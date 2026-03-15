"""RL-PLUS trainer (Week 3).

Implements a practical RL-PLUS objective:

  J(θ) = E_{on-policy} [ r(θ) * A ]
       + λ * E_{expert} [ r_m(θ) * A_c ]

- Internal term (on-policy): GRPO-style group-normalized advantage.
- External term (expert): MIS ratio r_m with exploration-based advantage A_c.

Practical extensions included:
- Reward shaping: small bonus for *correct* answers that also follow <answer>...</answer> format.
- More accurate π_ω: optional separate behavior model for MIS denominator.
- Adapter warm-start: initialize policy from an SFT LoRA adapter (SFT→GRPO, SFT→RL-PLUS).

This module is designed for a **training environment** (transformers + peft) and
intentionally does NOT import vLLM.
"""

from __future__ import annotations

import gc
import os
import shutil
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from runtimes.train_unsloth.triage_rl_plus_compat import SYSTEM_PROMPT, build_chat_prompt, extract_int, get_user_prompt, has_answer_tag


# ----------------------------
# Config
# ----------------------------

@dataclass
class RLPlusConfig:
    # Model / IO
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/rlplus_adapter"
    system_prompt: str = SYSTEM_PROMPT

    # Optional warm-start from an existing LoRA adapter
    init_adapter_path: Optional[str] = None


    # Optional LoRA adapter for the reference model (KL baseline).
    # Useful for SFT→GRPO: set this to the same adapter as init_adapter_path
    # to prevent KL-shaping from pulling the policy back to the base model when rewards are sparse.
    ref_adapter_path: Optional[str] = None
    # Optional behavior model (π_ω) for MIS denominator.
    # If None => use ref_model as π_ω.
    behavior_model_name_or_path: Optional[str] = None
    behavior_device: str = "cuda"  # "cuda" | "cpu"

    # Reference model device (for KL and optionally π_ω)
    ref_device: str = "cuda"  # "cuda" | "cpu"
    ref_load_in_4bit: bool = False  # quantize ref model to save ~12 GB VRAM
    ref_use_disable_adapter: bool = False  # True → use policy base model as ref (no separate ref model)

    # Memory-efficient loss: process internal term in chunks of this size.
    # 0 = original (all rollouts in one forward pass).
    # 4 is a good default for 8B models on 96 GB.
    loss_chunk_size: int = 0

    # Training
    seed: int = 42
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_steps: int = 200  # optimizer steps
    batch_size_prompts: int = 8
    group_size: int = 8
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    # Generation (on-policy rollouts)
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95

    # GRPO-ish
    beta_kl: float = 0.02  # used as KL *shaping* (see _compute_loss)
    clip_eps: float = 0.2  # kept for completeness; ratio≈1 with single update per rollout

    # RL-PLUS external term
    lambda_ext: float = 1.0
    gamma_focal: float = 2.0
    r_m_max: float = 10.0

    # Reward shaping
    format_bonus: float = 0.05  # added only if answer is correct AND formatted (<answer>...</answer>)

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Loading / dtype / placement
    # If you see "total VRAM is huge" on multi-GPU machines, set device_map=None and restrict CUDA_VISIBLE_DEVICES.
    # Use "auto" only if you intentionally want to shard the model.
    device_map: Optional[str] = None  # None | "auto" | explicit device_map

    enable_gradient_checkpointing: bool = True

    load_in_4bit: bool = False
    torch_dtype: str = "bf16"  # bf16|fp16|fp32
    device: str = "cuda"  # cuda|cpu

    # -----------------
    # Backend speed-ups (optional)
    # -----------------
    # If True, will try to load / patch the policy model via Unsloth's FastLanguageModel.
    # Works only on CUDA. If Unsloth is not installed, falls back to vanilla Transformers.
    use_unsloth: bool = False

    # If True, uses Unsloth smart gradient checkpointing (can touch ALL visible GPUs).
    # If you see `CUDA driver error: unknown error`, keep this False and rely on HF checkpointing.
    unsloth_smart_gradient_checkpointing: bool = False

    # Max sequence length passed into FastLanguageModel.from_pretrained / get_peft_model.
    # Should be >= prompt_len + max_new_tokens.
    unsloth_max_seq_length: int = 2048

    # These env flags must be set BEFORE importing unsloth (we do it inside __init__).
    # Mirrors the toggles you used in Week 2.
    unsloth_disable_compile: bool = False  # True => UNSLOTH_COMPILE_DISABLE=1 (more stable, slower)
    unsloth_fullgraph: bool = False        # keep False to reduce recompiles
    unsloth_compile_ignore_errors: bool = True
    clear_unsloth_cache: bool = True       # delete ./unsloth_compiled_cache at start

    disable_torchdynamo: bool = False      # emergency switch; slows down
    disable_torch_compile: bool = False    # emergency switch; slows down

    # Optimizer
    optim: str = "adamw"  # "adamw" | "adamw_8bit" (bitsandbytes)


    # -----------------
    # Logging / debug
    # -----------------
    timing_cuda_sync: bool = False  # if True, torch.cuda.synchronize() around timers (slower, more accurate)
    print_rollout_samples_every: int = 0  # 0=off; if >0 prints a few completions every N steps
    print_rollout_samples_n: int = 2

    # -----------------
    # Comet.ml (optional)
    # -----------------
    comet_project: Optional[str] = None
    comet_workspace: Optional[str] = None
    comet_experiment_name: Optional[str] = None
    comet_tags: Tuple[str, ...] = ()
    comet_offline: bool = False
    comet_disabled: bool = False
    comet_log_parameters: bool = True

    # -----------------
    # Memory cleanup (optional)
    # -----------------
    cleanup_every_steps: int = 0  # 0=off; if >0 runs gc + empty_cache every N steps
    save_every_steps: int = 0    # 0=off; if >0 saves checkpoint every N steps
    cleanup_at_end: bool = True
    move_models_to_cpu_at_end: bool = False
    cleanup_ipc_collect: bool = False


# ----------------------------
# Helpers
# ----------------------------

def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = str(s).lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def _first_eos_index(token_ids: torch.Tensor, eos_id: int) -> Optional[int]:
    eos_positions = (token_ids == eos_id).nonzero(as_tuple=False)
    if eos_positions.numel() == 0:
        return None
    return int(eos_positions[0].item())


def _pad_right(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    bsz = len(seqs)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        n = len(s)
        input_ids[i, :n] = torch.tensor(s, dtype=torch.long)
        attn[i, :n] = 1

    return input_ids, attn


def _completion_mask_from_prompt_lens(prompt_lens: List[int], full_lens: List[int]) -> torch.Tensor:
    """Mask for completion tokens (next-token positions).

    We return a [B, max_L-1] mask aligned with token_logp[:, :T-1] (next-token probs).
    """
    bsz = len(prompt_lens)
    max_L = max(full_lens)
    mask = torch.zeros((bsz, max_L - 1), dtype=torch.float32)

    for i, (p, L) in enumerate(zip(prompt_lens, full_lens)):
        p = int(p)
        L = int(L)
        if L <= p:
            continue
        start = max(0, p - 1)
        end = min(max_L - 1, max(0, L - 1))
        if end > start:
            mask[i, start:end] = 1.0

    return mask


def _gather_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return per-token log-probabilities for the *next* token.

    Robust to rare cases where model returns logits for a truncated sequence length.
    """
    import torch.nn.functional as F

    # logits: [B, Tlog, V], input_ids: [B, Tin]
    Tlog = logits.size(1)
    Tin  = input_ids.size(1)

    # Align to the minimum available length to avoid shape mismatch
    T = min(Tlog, Tin)
    if T < 2:
        # No next-token positions
        return logits.new_zeros((logits.size(0), 0))

    logits = logits[:, :T, :]            # [B, T, V]
    targets = input_ids[:, :T]           # [B, T]

    # next-token logp for positions 0..T-2
    logits = logits[:, :-1, :]           # [B, T-1, V]
    targets = targets[:, 1:]             # [B, T-1]

    B, Tm1, V = logits.shape
    nll = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
    )
    return (-nll).reshape(B, Tm1)


def _safe_mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / max(1, len(xs)))


def _safe_std(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(var))


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Standard HumanEval pass@k estimator: pass@k = 1 - C(n-c,k)/C(n,k)."""
    n = int(n)
    c = int(c)
    k = int(k)
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - c < k:
        return 1.0

    def logC(a: int, b: int) -> float:
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    log_ratio = logC(n - c, k) - logC(n, k)
    return float(1.0 - math.exp(log_ratio))


def _to_float(x: Any) -> float:
    try:
        return float(x.detach().cpu().item())  # type: ignore[attr-defined]
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")


# ----------------------------
# Trainer
# ----------------------------

class RLPlusTrainer:
    def __init__(self, cfg: RLPlusConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self._warned_prompt_truncation = False

        torch_dtype = _torch_dtype_from_str(cfg.torch_dtype)

        # -----------------
        # Optional Unsloth backend (policy model only)
        # -----------------
        self._use_unsloth: bool = bool(cfg.use_unsloth) and str(cfg.device).lower() == "cuda"
        self._FastLanguageModel = None  # set if Unsloth is available

        if self._use_unsloth:
            self._configure_unsloth_env()

            try:
                from unsloth import FastLanguageModel  # type: ignore
            except Exception as e:
                print(f"[WARN] Unsloth import failed -> fallback to Transformers. Reason: {e}")
                self._use_unsloth = False
            else:
                self._FastLanguageModel = FastLanguageModel

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Tokenizer / Policy base model
        if self._use_unsloth and (self._FastLanguageModel is not None):
            FLM = self._FastLanguageModel

            # IMPORTANT: Unsloth's *smart* gradient checkpointing tries to allocate buffers on ALL visible GPUs.
            # On some multi-GPU setups this can crash with `CUDA driver error: unknown error`.
            # We therefore disable it by default and rely on standard HF checkpointing instead.
            base_model, tok = FLM.from_pretrained(
                model_name=cfg.model_name,
                max_seq_length=int(cfg.unsloth_max_seq_length),
                dtype=torch_dtype,
                load_in_4bit=bool(cfg.load_in_4bit),
                use_gradient_checkpointing=("unsloth" if bool(cfg.unsloth_smart_gradient_checkpointing) else False),
                trust_remote_code=True,
            )
            self.tokenizer = tok
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model.config.use_cache = False

            if cfg.use_lora:
                # Warm-start from an existing adapter
                if cfg.init_adapter_path is not None:
                    from peft import PeftModel

                    self.model = PeftModel.from_pretrained(base_model, cfg.init_adapter_path, is_trainable=True)
                else:
                    # Create a new LoRA adapter via Unsloth helper (faster kernels + checkpointing)
                    self.model = FLM.get_peft_model(
                        base_model,
                        r=int(cfg.lora_r),
                        target_modules=list(cfg.lora_target_modules),
                        lora_alpha=int(cfg.lora_alpha),
                        lora_dropout=float(cfg.lora_dropout),
                        bias="none",
                        use_gradient_checkpointing=("unsloth" if bool(cfg.unsloth_smart_gradient_checkpointing) else False),
                        random_state=int(cfg.seed),
                        max_seq_length=int(cfg.unsloth_max_seq_length),
                    )
            else:
                self.model = base_model

            # Put policy on train mode
            self.model.train()

            # Some models expose standard checkpointing hooks too (keep for compatibility)
            if getattr(cfg, "enable_gradient_checkpointing", False):
                try:
                    if hasattr(self.model, "gradient_checkpointing_enable"):
                        self.model.gradient_checkpointing_enable()
                    if hasattr(self.model, "enable_input_require_grads"):
                        self.model.enable_input_require_grads()
                except Exception:
                    pass

            # Hint Unsloth that we will train
            try:
                FLM.for_training(self.model)
            except Exception:
                pass

            # Reference model (still needed for KL / π_ω fallback)
            if bool(getattr(cfg, "ref_use_disable_adapter", False)):
                self.ref_model = None  # will use policy base via disable_adapter_layers()
            else:
                self.ref_model = self._load_reference_model(
                    AutoModelForCausalLM=AutoModelForCausalLM,
                    torch_dtype=torch_dtype,
                )

        else:
            # -----------------
            # Vanilla Transformers backend
            # -----------------
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs: Dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch_dtype)

            # NOTE on device_map:
            # - device_map="auto" can shard the model across *all* visible GPUs (which may look like "140GB VRAM used").
            # - For a 0.5B model it's usually better to keep everything on a single GPU:
            #     set cfg.device_map=None and restrict CUDA_VISIBLE_DEVICES.
            if cfg.load_in_4bit:
                model_kwargs.update(dict(load_in_4bit=True, device_map=cfg.device_map or "auto"))
            else:
                if cfg.device == "cuda" and cfg.device_map:
                    model_kwargs.update(dict(device_map=cfg.device_map))
                else:
                    model_kwargs.update(dict(device_map=None))

            # Policy model
            base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
            base_model.config.use_cache = False

            if (not cfg.load_in_4bit) and str(cfg.device).lower() == "cuda" and not cfg.device_map:
                base_model.to("cuda")

            if cfg.use_lora:
                from peft import LoraConfig, PeftModel, get_peft_model

                if cfg.init_adapter_path is not None:
                    self.model = PeftModel.from_pretrained(base_model, cfg.init_adapter_path, is_trainable=True)
                else:
                    lora_cfg = LoraConfig(
                        r=int(cfg.lora_r),
                        lora_alpha=int(cfg.lora_alpha),
                        lora_dropout=float(cfg.lora_dropout),
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=list(cfg.lora_target_modules),
                    )
                    self.model = get_peft_model(base_model, lora_cfg)
            else:
                self.model = base_model

            try:
                if hasattr(self.model, "print_trainable_parameters"):
                    self.model.print_trainable_parameters()
            except Exception:
                pass

            self.model.train()

            # Gradient checkpointing reduces activation memory during backprop
            if getattr(cfg, "enable_gradient_checkpointing", False):
                try:
                    if hasattr(self.model, "gradient_checkpointing_enable"):
                        self.model.gradient_checkpointing_enable()
                    if hasattr(self.model, "enable_input_require_grads"):
                        self.model.enable_input_require_grads()
                except Exception:
                    pass

            # Reference model
            if bool(getattr(cfg, "ref_use_disable_adapter", False)):
                self.ref_model = None
            else:
                self.ref_model = self._load_reference_model(
                    AutoModelForCausalLM=AutoModelForCausalLM,
                    torch_dtype=torch_dtype,
                )

        # -----------------
        # Behavior model (optional)
        # -----------------
        self.behavior_model: Optional[torch.nn.Module] = None
        if cfg.behavior_model_name_or_path:
            self.behavior_model = self._load_behavior_model(
                AutoModelForCausalLM=AutoModelForCausalLM,
                torch_dtype=torch_dtype,
            )

        # -----------------
        # Optimizer
        # -----------------
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = self._make_optimizer(trainable)

        self.policy_device = _model_device(self.model)

        self.history: List[Dict[str, float]] = []

        # Comet (optional)
        self.comet = self._init_comet()


    # -----------------------
    # Unsloth / loading helpers
    # -----------------------

    def _configure_unsloth_env(self) -> None:
        """Set env vars that control Unsloth + torch.compile behavior.

        Important: these must be set BEFORE importing `unsloth`.
        Mirrors the logic from your Week 2 notebook.
        """
        cfg = self.cfg

        if bool(getattr(cfg, "clear_unsloth_cache", False)):
            try:
                shutil.rmtree(Path.cwd() / "unsloth_compiled_cache", ignore_errors=True)
            except Exception:
                pass

        # Unsloth flags
        os.environ["UNSLOTH_FULLGRAPH"] = "1" if bool(getattr(cfg, "unsloth_fullgraph", False)) else "0"
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1" if bool(getattr(cfg, "unsloth_disable_compile", False)) else "0"
        if bool(getattr(cfg, "unsloth_compile_ignore_errors", True)):
            os.environ["UNSLOTH_COMPILE_IGNORE_ERRORS"] = "1"

        # TorchDynamo / torch.compile kill-switches
        if bool(getattr(cfg, "disable_torchdynamo", False)):
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
        if bool(getattr(cfg, "disable_torch_compile", False)):
            os.environ["TORCH_COMPILE_DISABLE"] = "1"

    def _make_optimizer(self, trainable_params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
        cfg = self.cfg
        optim_name = str(getattr(cfg, "optim", "adamw")).strip().lower()

        if optim_name in {"adamw_8bit", "adamw8bit", "8bit"}:
            try:
                import bitsandbytes as bnb  # type: ignore

                return bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=float(cfg.lr),
                    weight_decay=float(cfg.weight_decay),
                )
            except Exception as e:
                print(f"[WARN] bitsandbytes AdamW8bit unavailable -> fallback to torch AdamW. Reason: {e}")

        return torch.optim.AdamW(
            trainable_params,
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    def _load_reference_model(self, *, AutoModelForCausalLM, torch_dtype: torch.dtype) -> torch.nn.Module:
        cfg = self.cfg

        # CPU ref is safest on VRAM, but slower.
        if str(cfg.ref_device).lower() == "cpu":
            # Use bfloat16 on CPU to halve RAM usage (32→16 GB for 8B model).
            # float32 is wasteful: we only need logprobs, not training precision.
            cpu_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            ref = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                trust_remote_code=True,
                torch_dtype=cpu_dtype,
                device_map=None,
            )
            ref.to("cpu")
        else:
            # Prefer Unsloth for CUDA ref if policy uses Unsloth
            if self._use_unsloth and (self._FastLanguageModel is not None):
                FLM = self._FastLanguageModel
                ref_4bit = bool(getattr(cfg, "ref_load_in_4bit", False)) or bool(cfg.load_in_4bit)
                ref, _tok2 = FLM.from_pretrained(
                    model_name=cfg.model_name,
                    max_seq_length=int(cfg.unsloth_max_seq_length),
                    dtype=torch_dtype,
                    load_in_4bit=ref_4bit,
                    use_gradient_checkpointing=False,
                    trust_remote_code=True,
                )
                try:
                    FLM.for_inference(ref)
                except Exception:
                    pass
            else:
                ref_kwargs: Dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch_dtype)
                ref_4bit = bool(getattr(cfg, "ref_load_in_4bit", False)) or bool(cfg.load_in_4bit)

                if ref_4bit:
                    ref_kwargs.update(dict(load_in_4bit=True, device_map=cfg.device_map or "auto"))
                else:
                    if str(cfg.ref_device).lower() == "cuda" and cfg.device_map:
                        ref_kwargs.update(dict(device_map=cfg.device_map))
                    else:
                        ref_kwargs.update(dict(device_map=None))

                ref = AutoModelForCausalLM.from_pretrained(cfg.model_name, **ref_kwargs)

                if (not ref_4bit) and str(cfg.ref_device).lower() == "cuda" and not cfg.device_map:
                    ref.to("cuda")

        try:
            ref.config.use_cache = False
        except Exception:
            pass

        # Optional: attach a reference LoRA adapter (kept frozen).
        # This is especially useful for SFT→GRPO: use the SFT adapter as the KL reference.
        if getattr(cfg, "ref_adapter_path", None):
            try:
                ref_ad = str(getattr(cfg, "ref_adapter_path"))
                p = Path(ref_ad)
                if p.exists() and (p / "adapter_config.json").exists():
                    from peft import PeftModel
                    ref = PeftModel.from_pretrained(ref, ref_ad, is_trainable=False)
            except Exception as e:
                print(f"[WARN] Failed to load ref_adapter_path={getattr(cfg, 'ref_adapter_path')}: {e}")

        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    def _load_behavior_model(self, *, AutoModelForCausalLM, torch_dtype: torch.dtype) -> torch.nn.Module:
        cfg = self.cfg
        beh_path = cfg.behavior_model_name_or_path
        assert beh_path is not None

        # Heuristic: local LoRA adapter dir
        is_lora_adapter = False
        try:
            p = Path(beh_path)
            if p.exists() and (p / "adapter_config.json").exists():
                is_lora_adapter = True
        except Exception:
            is_lora_adapter = False

        on_cpu = str(cfg.behavior_device).lower() == "cpu"

        def _hf_base_for_behavior() -> torch.nn.Module:
            kw: Dict[str, Any] = dict(trust_remote_code=True, torch_dtype=(torch.float32 if on_cpu else torch_dtype))
            if on_cpu:
                kw.update(dict(device_map=None))
            else:
                if cfg.load_in_4bit:
                    kw.update(dict(load_in_4bit=True, device_map=cfg.device_map or "auto"))
                else:
                    kw.update(dict(device_map=cfg.device_map if cfg.device_map else None))
            m = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kw)
            try:
                m.config.use_cache = False
            except Exception:
                pass
            if on_cpu:
                m.to("cpu")
            else:
                if (not cfg.load_in_4bit) and str(cfg.behavior_device).lower() == "cuda" and not cfg.device_map:
                    m.to("cuda")
            return m

        if is_lora_adapter:
            from peft import PeftModel

            if (not on_cpu) and self._use_unsloth and (self._FastLanguageModel is not None):
                FLM = self._FastLanguageModel
                beh_base, _ = FLM.from_pretrained(
                    model_name=cfg.model_name,
                    max_seq_length=int(cfg.unsloth_max_seq_length),
                    dtype=torch_dtype,
                    load_in_4bit=bool(cfg.load_in_4bit),
                    use_gradient_checkpointing=False,
                    trust_remote_code=True,
                )
                try:
                    beh_base.config.use_cache = False
                except Exception:
                    pass
                beh_model = PeftModel.from_pretrained(beh_base, beh_path, is_trainable=False)
                try:
                    FLM.for_inference(beh_model)
                except Exception:
                    pass
            else:
                beh_base = _hf_base_for_behavior()
                beh_model = PeftModel.from_pretrained(beh_base, beh_path, is_trainable=False)
        else:
            # behavior is a full model path
            kw2: Dict[str, Any] = dict(trust_remote_code=True, torch_dtype=(torch.float32 if on_cpu else torch_dtype))
            if on_cpu:
                kw2.update(dict(device_map=None))
            else:
                if cfg.load_in_4bit:
                    kw2.update(dict(load_in_4bit=True, device_map=cfg.device_map or "auto"))
                else:
                    kw2.update(dict(device_map=cfg.device_map if cfg.device_map else None))

            beh_model = AutoModelForCausalLM.from_pretrained(beh_path, **kw2)
            try:
                beh_model.config.use_cache = False
            except Exception:
                pass

            if on_cpu:
                beh_model.to("cpu")
            else:
                if (not cfg.load_in_4bit) and str(cfg.behavior_device).lower() == "cuda" and not cfg.device_map:
                    beh_model.to("cuda")

        beh_model.eval()
        for p in beh_model.parameters():
            p.requires_grad_(False)

        return beh_model
    # -----------------------
    # Comet
    # -----------------------

    def _init_comet(self) -> Optional[Any]:
        cfg = self.cfg
        if not cfg.comet_project or bool(cfg.comet_disabled):
            return None

        try:
            from comet_ml import Experiment  # type: ignore
        except Exception as e:
            print(f"[WARN] comet_ml is not available: {e}")
            return None

        OfflineExperiment = None
        if bool(cfg.comet_offline):
            try:
                from comet_ml import OfflineExperiment as _OfflineExperiment  # type: ignore

                OfflineExperiment = _OfflineExperiment
            except Exception:
                OfflineExperiment = None

        exp_cls = OfflineExperiment if (OfflineExperiment is not None) else Experiment

        # Try a modern signature first, then fallback.
        try:
            exp = exp_cls(
                project_name=str(cfg.comet_project),
                workspace=(str(cfg.comet_workspace) if cfg.comet_workspace else None),
                auto_output_logging="simple",
                disabled=bool(cfg.comet_disabled),
            )
        except TypeError:
            exp = exp_cls(
                project_name=str(cfg.comet_project),
                workspace=(str(cfg.comet_workspace) if cfg.comet_workspace else None),
            )

        try:
            if cfg.comet_experiment_name:
                exp.set_name(str(cfg.comet_experiment_name))
        except Exception:
            pass

        try:
            if cfg.comet_tags:
                exp.add_tags(list(cfg.comet_tags))
        except Exception:
            pass

        if bool(cfg.comet_log_parameters):
            try:
                params = dict(asdict(cfg))
                params.pop("system_prompt", None)  # too large
                exp.log_parameters(params)
            except Exception:
                pass

        return exp

    def _comet_log_metrics(self, metrics: Dict[str, float], *, step: int, prefix: str) -> None:
        if self.comet is None:
            return
        try:
            flat = {f"{prefix}/{k}": float(v) for k, v in metrics.items() if k != "step"}
            self.comet.log_metrics(flat, step=int(step))
        except Exception:
            pass

    def _comet_log_text(self, name: str, text: str, *, step: int) -> None:
        if self.comet is None:
            return
        try:
            if hasattr(self.comet, "log_text"):
                self.comet.log_text(text, name=str(name), step=int(step))
            else:
                self.comet.log_asset_data(text, name=str(name))
        except Exception:
            pass

    def _comet_end(self) -> None:
        if self.comet is None:
            return
        try:
            if hasattr(self.comet, "end"):
                self.comet.end()
        except Exception:
            pass
        self.comet = None

    # -----------------------
    # IO / Cleanup
    # -----------------------

    def save(self, out_dir: Optional[str] = None) -> str:
        out_dir = out_dir or self.cfg.output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        return out_dir

    def cleanup(self, *, move_models_to_cpu: bool = False) -> None:
        """Best-effort memory cleanup.

        Notes:
          - empty_cache() releases unused cached CUDA blocks, not model weights.
          - move_models_to_cpu=True is useful when you are DONE with training and want to free VRAM.
        """
        if move_models_to_cpu:
            for m in [getattr(self, "model", None), getattr(self, "ref_model", None), getattr(self, "behavior_model", None)]:
                if m is None:
                    continue
                try:
                    m.to("cpu")
                except Exception:
                    pass

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if bool(getattr(self.cfg, "cleanup_ipc_collect", False)):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    # -----------------------
    # Training
    # -----------------------

    def _tokenize_prompts_for_generation(
        self,
        prompt_texts: Sequence[str],
        *,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompts for generation with safe truncation in Unsloth mode.

        Unsloth can crash during generation if prompt_len + max_new_tokens exceeds
        unsloth_max_seq_length. In that case we keep the *rightmost* prompt tokens
        (set truncation_side='left') so the latest user content is preserved.
        """
        cfg = self.cfg
        tok = self.tokenizer

        max_new = int(cfg.max_new_tokens if max_new_tokens is None else max_new_tokens)
        use_trunc = False
        max_prompt_tokens: Optional[int] = None
        n_truncated = 0
        max_prompt_seen = 0

        if self._use_unsloth and int(getattr(cfg, "unsloth_max_seq_length", 0)) > 0:
            # Keep a small safety margin for generation internals.
            max_prompt_tokens = max(8, int(cfg.unsloth_max_seq_length) - max_new - 2)
            use_trunc = True

            try:
                lens = self.tokenizer(list(prompt_texts), add_special_tokens=False, padding=False)["input_ids"]
                lens = [len(x) for x in lens]
                if lens:
                    max_prompt_seen = int(max(lens))
                    n_truncated = int(sum(1 for x in lens if x > max_prompt_tokens))
            except Exception:
                pass

        old_trunc_side = getattr(tok, "truncation_side", "right")
        try:
            if use_trunc:
                tok.truncation_side = "left"
            enc = tok(
                list(prompt_texts),
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                truncation=bool(use_trunc),
                max_length=int(max_prompt_tokens) if max_prompt_tokens is not None else None,
            )
        finally:
            tok.truncation_side = old_trunc_side

        if use_trunc and n_truncated > 0 and (not self._warned_prompt_truncation):
            self._warned_prompt_truncation = True
            print(
                f"[WARN] Truncated {n_truncated}/{len(prompt_texts)} prompts for generation "
                f"to max_prompt_tokens={max_prompt_tokens} (max_seen={max_prompt_seen}) because "
                f"unsloth_max_seq_length={cfg.unsloth_max_seq_length}. "
                "To avoid truncation, increase unsloth_max_seq_length or reduce prompt length."
            )

        return enc

    def train(
        self,
        *,
        train_rows: Sequence[Dict[str, Any]],
        env_verify_fn: Optional[Callable[[Dict[str, Any], str], bool]] = None,
        reward_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
        make_gold_completion_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        eval_callback: Optional[Callable[[int], None]] = None,
        log_every: int = 10,
        eval_every: int = 50,
    ) -> List[Dict[str, float]]:
        cfg = self.cfg
        self.model.train()

        def _default_verify(row: Dict[str, Any], completion_text: str) -> bool:
            pred = extract_int(completion_text)
            return pred is not None and int(pred) == int(row["answer"])

        verify_fn = env_verify_fn or _default_verify

        def _default_gold(row: Dict[str, Any]) -> str:
            ans = int(row["answer"])
            return f"<think></think>\n<answer>{ans}</answer>"

        gold_fn = make_gold_completion_fn or _default_gold

        cached_prompts: Dict[int, str] = {}

        def get_prompt_text(idx: int, row: Dict[str, Any]) -> str:
            if idx in cached_prompts:
                return cached_prompts[idx]
            pt = build_chat_prompt(self.tokenizer, get_user_prompt(row), system_prompt=cfg.system_prompt)
            cached_prompts[idx] = pt
            return pt

        n_rows = len(train_rows)
        if n_rows < cfg.batch_size_prompts:
            raise ValueError("train_rows too small for batch_size_prompts")

        def _now() -> float:
            if cfg.timing_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()


        # --- Unsloth sanity check: max_seq_length must cover prompt + completion
        # If too small, Unsloth may truncate logits to unsloth_max_seq_length, causing silent truncation.
        if self._use_unsloth and int(getattr(cfg, "unsloth_max_seq_length", 0)) > 0:
            try:
                sample_rows = list(train_rows[: min(8, n_rows)])
                sample_prompts = [
                    build_chat_prompt(self.tokenizer, get_user_prompt(r), system_prompt=cfg.system_prompt)
                    for r in sample_rows
                ]
                enc0 = self.tokenizer(sample_prompts, add_special_tokens=False)
                lens0 = [len(ids) for ids in enc0["input_ids"]]
                max_prompt_len = max(lens0) if lens0 else 0
                need = int(max_prompt_len) + int(cfg.max_new_tokens) + 2
                if int(cfg.unsloth_max_seq_length) < need:
                    print(
                        f"[WARN] unsloth_max_seq_length={cfg.unsloth_max_seq_length} is likely too small: "
                        f"max_prompt_len≈{max_prompt_len}, max_new_tokens={cfg.max_new_tokens}. "
                        f"Consider setting unsloth_max_seq_length >= {need}."
                    )
            except Exception:
                pass

        pbar = tqdm(range(int(cfg.max_steps)), desc="GRPO/RL-PLUS train", unit="step")
        for step in pbar:
            t_step0 = _now()

            self.optim.zero_grad(set_to_none=True)
            logs_accum: Dict[str, float] = {}

            for _micro in range(int(cfg.grad_accum_steps)):
                # ---- sample batch
                batch_indices = random.sample(range(n_rows), k=int(cfg.batch_size_prompts))
                batch_rows = [train_rows[i] for i in batch_indices]
                batch_prompts = [get_prompt_text(i, r) for i, r in zip(batch_indices, batch_rows)]

                # ---- on-policy rollouts
                t0 = _now()
                rollouts = self._generate_rollouts(batch_prompts)
                t_gen = _now() - t0

                # ---- Free KV cache before loss computation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ---- reward / parsing stats
                rewards_flat: List[float] = []
                correct_flat: List[bool] = []
                parsed_flat: List[bool] = []
                formatted_flat: List[bool] = []
                abs_errs: List[float] = []
                preds: List[int] = []
                comp_lens: List[int] = []

                for ro in rollouts:
                    row = batch_rows[ro["prompt_group_idx"]]
                    comp_text = ro["completion_text"]

                    comp_lens.append(int(len(ro["completion_ids"])))

                    fmt_ok = bool(has_answer_tag(comp_text))
                    formatted_flat.append(fmt_ok)

                    pred = extract_int(comp_text)
                    parsed_ok = pred is not None
                    parsed_flat.append(parsed_ok)

                    ok = bool(verify_fn(row, comp_text))
                    correct_flat.append(ok)

                    # Gradual reward from reward_fn, or binary 0/1 fallback
                    if reward_fn is not None:
                        r = float(reward_fn(row, comp_text))
                    else:
                        r = 1.0 if ok else 0.0
                        if ok and cfg.format_bonus and cfg.format_bonus > 0 and fmt_ok:
                            r += float(cfg.format_bonus)
                    rewards_flat.append(float(r))

                    if parsed_ok:
                        try:
                            gold = int(row["answer"])
                            preds.append(int(pred))  # type: ignore[arg-type]
                            abs_errs.append(float(abs(int(pred) - gold)))  # type: ignore[arg-type]
                        except Exception:
                            pass

                reward_int_mean = _safe_mean(rewards_flat)
                reward_int_std = _safe_std(rewards_flat)
                reward_int_min = float(min(rewards_flat)) if rewards_flat else 0.0
                reward_int_max = float(max(rewards_flat)) if rewards_flat else 0.0

                correct_rate = float(sum(1.0 for x in correct_flat if x) / max(1, len(correct_flat)))
                parse_rate = float(sum(1.0 for x in parsed_flat if x) / max(1, len(parsed_flat)))
                format_rate = float(sum(1.0 for x in formatted_flat if x) / max(1, len(formatted_flat)))
                mae_parsed = _safe_mean(abs_errs) if abs_errs else float("nan")
                pred_mean = _safe_mean(preds) if preds else float("nan")
                len_mean = _safe_mean(comp_lens) if comp_lens else float("nan")


                # ---- DEBUG: are rewards diverse within each prompt group?
                # rewards_flat has length = batch_size_prompts * group_size
                G = int(cfg.group_size)
                B = int(len(batch_rows))

                # reshape rewards to [B, G]
                if len(rewards_flat) == B * G:
                    mixed = 0
                    all_same = 0
                    grp_stds = []
                    for bi in range(B):
                        r = rewards_flat[bi*G:(bi+1)*G]
                        rmin, rmax = min(r), max(r)
                        if rmin == rmax:
                            all_same += 1
                        else:
                            mixed += 1
                        # std for logging (cheap)
                        mu = sum(r)/G
                        var = sum((x-mu)*(x-mu) for x in r)/G
                        grp_stds.append(var**0.5)

                    frac_mixed = mixed / max(1, B)
                    grp_std_mean = sum(grp_stds)/max(1, len(grp_stds))
                else:
                    frac_mixed = float("nan")
                    grp_std_mean = float("nan")

                # log into your logs dict
                logs_accum["frac_groups_mixed"] = logs_accum.get("frac_groups_mixed", 0.0) + float(frac_mixed)
                logs_accum["group_reward_std_mean"] = logs_accum.get("group_reward_std_mean", 0.0) + float(grp_std_mean)

                # ---- GRPO advantages
                adv_flat = self._group_normalized_advantages(
                    rewards_flat, batch_size=len(batch_rows), group_size=cfg.group_size
                )
                adv_mean = _safe_mean(adv_flat)
                adv_std = _safe_std(adv_flat)

                # ---- RL-PLUS gold trajectories
                gold_trajs = None
                gold_adv = None
                if cfg.lambda_ext and cfg.lambda_ext > 0:
                    gold_trajs = self._build_gold_trajectories(batch_prompts, batch_rows, gold_fn)
                    # Per-row gold rewards (each case has its own gold_reward)
                    if reward_fn is not None and "gold_reward" in batch_rows[0]:
                        per_row_gold_rewards = [float(r["gold_reward"]) for r in batch_rows]
                    else:
                        gr = 1.0 + (float(cfg.format_bonus) if cfg.format_bonus and cfg.format_bonus > 0 else 0.0)
                        per_row_gold_rewards = [gr] * len(batch_rows)
                    gold_adv = self._gold_advantages(
                        rewards_flat,
                        batch_size=len(batch_rows),
                        group_size=cfg.group_size,
                        gold_rewards=per_row_gold_rewards,
                    )

                # ---- compute loss (memory-efficient if loss_chunk_size > 0)
                t1 = _now()
                use_v2 = int(cfg.loss_chunk_size) > 0
                if use_v2:
                    logs = self._compute_loss_v2(
                        rollouts=rollouts,
                        advantages_flat=adv_flat,
                        gold_trajs=gold_trajs,
                        gold_adv=gold_adv,
                        grad_scale=1.0 / float(cfg.grad_accum_steps),
                    )
                else:
                    loss, logs = self._compute_loss(
                        rollouts=rollouts,
                        advantages_flat=adv_flat,
                        gold_trajs=gold_trajs,
                        gold_adv=gold_adv,
                    )
                t_loss = _now() - t1

                logs.update(
                    {
                        "reward_int_mean": reward_int_mean,
                        "reward_int_std": reward_int_std,
                        "reward_int_min": reward_int_min,
                        "reward_int_max": reward_int_max,
                        "correct_rate": correct_rate,
                        "parse_rate": parse_rate,
                        "format_rate": format_rate,
                        "mae_parsed": float(mae_parsed),
                        "pred_mean": float(pred_mean),
                        "len_mean": float(len_mean),
                        "adv_mean": float(adv_mean),
                        "adv_std": float(adv_std),
                        "t_gen_s": float(t_gen),
                        "t_loss_s": float(t_loss),
                    }
                )

                if not use_v2:
                    (loss / float(cfg.grad_accum_steps)).backward()

                for k, v in logs.items():
                    logs_accum[k] = logs_accum.get(k, 0.0) + float(v)

            # ---- DEBUG: max absolute gradient (before clipping)
            max_grad_abs = 0.0
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    g = float(p.grad.detach().abs().max().cpu())
                    if g > max_grad_abs:
                        max_grad_abs = g
            logs_step_max_grad_abs = max_grad_abs

            # ---- grad norm + step
            grad_norm = float("nan")
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                grad_norm = _to_float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(cfg.max_grad_norm),
                    )
                )

            self.optim.step()

            lr = float(self.optim.param_groups[0].get("lr", float(cfg.lr)))

            logs_step = {k: v / float(cfg.grad_accum_steps) for k, v in logs_accum.items()}
            logs_step["step"] = float(step + 1)
            logs_step["grad_norm"] = float(grad_norm)
            logs_step["lr"] = float(lr)
            logs_step["t_step_s"] = float(_now() - t_step0)
            logs_step["max_grad_abs"] = float(logs_step_max_grad_abs)

            self.history.append(logs_step)

            postfix = {
                "loss": f"{logs_step.get('loss', float('nan')):.3f}",
                "acc": f"{logs_step.get('correct_rate', float('nan')):.2f}",
                "r": f"{logs_step.get('reward_int_mean', float('nan')):.2f}",
                "fmt": f"{logs_step.get('format_rate', float('nan')):.2f}",
                "len": f"{logs_step.get('len_mean', float('nan')):.1f}",
            }
            if "kl" in logs_step:
                postfix["kl"] = f"{logs_step['kl']:.3f}"
            if "r_m_mean" in logs_step and cfg.lambda_ext and cfg.lambda_ext > 0:
                postfix["r_m"] = f"{logs_step['r_m_mean']:.2f}"

            postfix["mix"] = f"{logs_step.get('frac_groups_mixed', float('nan')):.2f}"

            pbar.set_postfix(postfix, refresh=False)

            if log_every and (step + 1) % int(log_every) == 0:
                print(
                    f"step {step+1}/{cfg.max_steps} | "
                    f"loss={logs_step.get('loss', float('nan')):.4f} "
                    f"int={logs_step.get('loss_int', float('nan')):.4f} "
                    f"ext={logs_step.get('loss_ext', float('nan')):.4f} "
                    f"kl={logs_step.get('kl', float('nan')):.4f} "
                    f"r_m={logs_step.get('r_m_mean', float('nan')):.3f} "
                    f"r={logs_step.get('reward_int_mean', float('nan')):.3f}±{logs_step.get('reward_int_std', 0.0):.3f} "
                    f"acc={logs_step.get('correct_rate', float('nan')):.3f} "
                    f"parse={logs_step.get('parse_rate', float('nan')):.3f} "
                    f"fmt={logs_step.get('format_rate', float('nan')):.3f} "
                    f"mae={logs_step.get('mae_parsed', float('nan')):.2f} "
                    f"len={logs_step.get('len_mean', float('nan')):.1f} "
                    f"grad={logs_step.get('grad_norm', float('nan')):.2f} "
                    f"lr={logs_step.get('lr', float('nan')):.2e} "
                    f"t={logs_step.get('t_step_s', float('nan')):.2f}s"
                )

            # Comet train logging
            self._comet_log_metrics(logs_step, step=step + 1, prefix="train")

            # Optional sample dump (helps debug format/verifier)
            if cfg.print_rollout_samples_every and (step + 1) % int(cfg.print_rollout_samples_every) == 0:
                self.debug_print_samples(
                    rows=batch_rows,
                    n=int(cfg.print_rollout_samples_n),
                    verify_fn=verify_fn,
                    temperature=float(cfg.temperature),
                    top_p=float(cfg.top_p),
                    max_new_tokens=int(cfg.max_new_tokens),
                    step=int(step + 1),
                )

            # periodic eval
            if eval_callback is not None and eval_every and (step + 1) % int(eval_every) == 0:
                eval_callback(int(step + 1))

            # periodic cleanup (fragmentation)
            if cfg.cleanup_every_steps and (step + 1) % int(cfg.cleanup_every_steps) == 0:
                self.cleanup(move_models_to_cpu=False)

            # periodic checkpoint save
            if cfg.save_every_steps and cfg.save_every_steps > 0 and (step + 1) % int(cfg.save_every_steps) == 0:
                ckpt_dir = str(Path(cfg.output_dir) / f"checkpoint-{step + 1}")
                self.save(ckpt_dir)
                print(f"  [checkpoint] saved → {ckpt_dir}")

            # best checkpoint (by smoothed reward over last 10 steps)
            if len(self.history) >= 10:
                recent_reward = _safe_mean([h.get("reward_int_mean", 0) for h in self.history[-10:]])
                if not hasattr(self, "_best_reward") or recent_reward > self._best_reward:
                    self._best_reward = recent_reward
                    self._best_step = step + 1
                    best_dir = str(Path(cfg.output_dir) / "checkpoint-best")
                    self.save(best_dir)

        self.save(cfg.output_dir)

        # Report best
        if hasattr(self, "_best_reward"):
            print(f"  [best] step {self._best_step}, smoothed reward = {self._best_reward:.4f}")
            print(f"  [best] saved → {Path(cfg.output_dir) / 'checkpoint-best'}")

        if bool(cfg.cleanup_at_end):
            self.cleanup(move_models_to_cpu=bool(cfg.move_models_to_cpu_at_end))

        self._comet_end()

        return self.history

    # -----------------------
    # Evaluation utilities (HF generate)
    # -----------------------

    @torch.no_grad()
    def evaluate_rows(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        verify_fn: Optional[Callable[[Dict[str, Any], str], bool]] = None,
        name: str = "eval",
        max_items: Optional[int] = None,
        n_samples: int = 1,
        ks: Tuple[int, ...] = (1, 2, 4, 8),
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        batch_size: int = 16,
        do_sample: Optional[bool] = None,
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate current policy on a fixed set of rows (HF generate).

        Meant for *debug/monitoring during training* (pass@1 or small pass@k),
        NOT for final pass@128 curves (those should be done in the vLLM notebook).
        """
        cfg = self.cfg
        tok = self.tokenizer

        def _default_verify(row: Dict[str, Any], completion_text: str) -> bool:
            pred = extract_int(completion_text)
            return pred is not None and int(pred) == int(row["answer"])

        verify_fn = verify_fn or _default_verify

        if max_items is not None:
            rows = list(rows)[: int(max_items)]

        if not rows:
            return {"n_items": 0.0}

        n_samples = int(max(1, n_samples))
        ks = tuple(int(k) for k in ks if int(k) >= 1 and int(k) <= n_samples)
        if not ks:
            ks = (1,)

        if max_new_tokens is None:
            max_new_tokens = int(cfg.max_new_tokens)

        if do_sample is None:
            do_sample = bool(n_samples > 1 or float(temperature) > 0.0)

        # Build prompts
        prompts: List[str] = []
        answers: List[int] = []
        for r in rows:
            prompts.append(build_chat_prompt(tok, get_user_prompt(r), system_prompt=cfg.system_prompt))
            answers.append(int(r["answer"]))

        # Switch to eval mode temporarily
        was_training = bool(self.model.training)
        self.model.eval()

        old_padding_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"

        # We generate in mini-batches to avoid OOM.
        total_completions: List[str] = []
        total_ok: List[bool] = []
        total_parsed: List[Optional[int]] = []
        total_fmt: List[bool] = []
        total_comp_lens: List[int] = []
        correct_counts: List[int] = []

        bs = max(1, int(batch_size))
        for start in range(0, len(prompts), bs):
            p_batch = prompts[start : start + bs]
            a_batch = answers[start : start + bs]
            r_batch = rows[start : start + bs]

            enc = self._tokenize_prompts_for_generation(p_batch, max_new_tokens=int(max_new_tokens))
            input_ids = enc["input_ids"].to(self.policy_device)
            attn = enc["attention_mask"].to(self.policy_device)
            input_len = input_ids.shape[1]

            # Repeat prompts to get multiple samples per prompt (deterministic grouping)
            if n_samples > 1:
                input_ids = input_ids.repeat_interleave(n_samples, dim=0)
                attn = attn.repeat_interleave(n_samples, dim=0)
                r_rep = [rr for rr in r_batch for _ in range(n_samples)]
                a_rep = [aa for aa in a_batch for _ in range(n_samples)]
            else:
                r_rep = list(r_batch)
                a_rep = list(a_batch)

            gen_kwargs: Dict[str, Any] = dict(
                input_ids=input_ids,
                attention_mask=attn,
                do_sample=bool(do_sample),
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                return_dict_in_generate=True,
            )
            if bool(do_sample):
                gen_kwargs.update(dict(temperature=float(temperature), top_p=float(top_p)))

            gen = self.model.generate(**gen_kwargs)
            seqs = gen.sequences
            # decode completions
            completions: List[str] = []
            ok_flags: List[bool] = []
            parsed: List[Optional[int]] = []
            fmt_flags: List[bool] = []
            comp_lens: List[int] = []

            for j in range(seqs.shape[0]):
                completion_ids_full = seqs[j, input_len:]
                eos_idx = _first_eos_index(completion_ids_full, tok.eos_token_id)
                if eos_idx is not None:
                    completion_ids = completion_ids_full[: eos_idx + 1].tolist()
                else:
                    completion_ids = completion_ids_full.tolist()

                comp_lens.append(int(len(completion_ids)))

                text = tok.decode(completion_ids, skip_special_tokens=True)
                completions.append(text)

                fmt_flags.append(bool(has_answer_tag(text)))
                p = extract_int(text)
                parsed.append(p)
                ok_flags.append(bool(verify_fn(r_rep[j], text)))

            # group correct counts for this batch
            for i in range(len(r_batch)):
                c = 0
                for j in range(i * n_samples, (i + 1) * n_samples):
                    if ok_flags[j]:
                        c += 1
                correct_counts.append(c)

            total_completions.extend(completions)
            total_ok.extend(ok_flags)
            total_parsed.extend(parsed)
            total_fmt.extend(fmt_flags)
            total_comp_lens.extend(comp_lens)

        tok.padding_side = old_padding_side

        # restore train mode
        if was_training:
            self.model.train()

        # Metrics
        n_items = len(rows)
        passk: Dict[int, float] = {}
        for k in ks:
            vals = [_pass_at_k(n_samples, c, int(k)) for c in correct_counts]
            passk[int(k)] = float(sum(vals) / max(1, len(vals)))

        acc_sample = float(sum(1.0 for x in total_ok if x) / max(1, len(total_ok)))
        any_correct = float(sum(1.0 for c in correct_counts if c > 0) / max(1, len(correct_counts)))

        parse_rate = float(sum(1.0 for x in total_parsed if x is not None) / max(1, len(total_parsed)))
        format_rate = float(sum(1.0 for x in total_fmt if x) / max(1, len(total_fmt)))
        len_mean = _safe_mean(total_comp_lens)

        abs_errs: List[float] = []
        # align parsed with repeated answers
        # Build repeated answers for error stats
        answers_rep: List[int] = []
        for a in answers:
            answers_rep.extend([a] * n_samples)

        for p, a in zip(total_parsed, answers_rep):
            if p is None:
                continue
            abs_errs.append(float(abs(int(p) - int(a))))
        mae_parsed = _safe_mean(abs_errs) if abs_errs else float("nan")

        out: Dict[str, float] = {
            "n_items": float(n_items),
            "n_samples": float(n_samples),
            "acc_sample": float(acc_sample),
            "any_correct": float(any_correct),
            "parse_rate": float(parse_rate),
            "format_rate": float(format_rate),
            "mae_parsed": float(mae_parsed),
            "len_mean": float(len_mean),
        }
        for k, v in passk.items():
            out[f"pass@{k}"] = float(v)

        ks_str = ", ".join(f"pass@{k}={out[f'pass@{k}']:.3f}" for k in passk.keys())
        print(
            f"[{name}] n={n_items} samples={n_samples} "
            f"acc_sample={acc_sample:.3f} any_correct={any_correct:.3f} "
            f"{ks_str} parse={parse_rate:.3f} fmt={format_rate:.3f} len≈{len_mean:.1f} mae≈{mae_parsed:.2f}"
        )

        if step is None:
            step = int(self.history[-1]["step"]) if self.history else 0
        self._comet_log_metrics(out, step=int(step), prefix=f"eval/{name}")

        return out

    @torch.no_grad()
    def debug_print_samples(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        n: int = 2,
        verify_fn: Optional[Callable[[Dict[str, Any], str], bool]] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 64,
        step: int = 0,
    ) -> None:
        """Print a few (row -> completion) samples to debug parsing/format/verifier."""
        cfg = self.cfg
        tok = self.tokenizer

        def _default_verify(row: Dict[str, Any], completion_text: str) -> bool:
            pred = extract_int(completion_text)
            return pred is not None and int(pred) == int(row["answer"])

        verify_fn = verify_fn or _default_verify

        if not rows:
            print("[debug] no rows")
            return

        n = max(1, min(int(n), len(rows)))
        rows = list(rows)[:n]

        prompts = [build_chat_prompt(tok, get_user_prompt(r), system_prompt=cfg.system_prompt) for r in rows]
        answers = [int(r["answer"]) for r in rows]

        old_padding_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"

        enc = self._tokenize_prompts_for_generation(prompts, max_new_tokens=int(max_new_tokens))
        input_ids = enc["input_ids"].to(self.policy_device)
        attn = enc["attention_mask"].to(self.policy_device)
        input_len = input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
        )
        gen = self.model.generate(**gen_kwargs)

        tok.padding_side = old_padding_side

        seqs = gen.sequences
        texts: List[str] = []
        for j in range(seqs.shape[0]):
            completion_ids_full = seqs[j, input_len:]
            eos_idx = _first_eos_index(completion_ids_full, tok.eos_token_id)
            if eos_idx is not None:
                completion_ids = completion_ids_full[: eos_idx + 1].tolist()
            else:
                completion_ids = completion_ids_full.tolist()
            texts.append(tok.decode(completion_ids, skip_special_tokens=True))

        print("------ debug samples ------")
        for i in range(n):
            text = texts[i]
            pred = extract_int(text)
            ok = bool(verify_fn(rows[i], text))
            fmt = bool(has_answer_tag(text))
            print(f"[{i}] gold={answers[i]} pred={pred} ok={ok} fmt={fmt}")
            print(text)
            print("---------------------------")

        # Optional: log a compact dump to Comet
        dump = []
        for i in range(n):
            dump.append(f"[{i}] gold={answers[i]} pred={extract_int(texts[i])} ok={bool(verify_fn(rows[i], texts[i]))} fmt={bool(has_answer_tag(texts[i]))}\n{texts[i]}")
        self._comet_log_text("debug_samples", "\n\n".join(dump), step=int(step))

    # -----------------------
    # Generation (rollouts)
    # -----------------------

    @torch.no_grad()
    def _generate_rollouts(self, prompt_texts: Sequence[str]) -> List[Dict[str, Any]]:
        cfg = self.cfg
        tok = self.tokenizer

        # Left-padding improves efficiency for generation
        old_padding_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"

        enc = self._tokenize_prompts_for_generation(prompt_texts, max_new_tokens=int(cfg.max_new_tokens))
        input_ids = enc["input_ids"].to(self.policy_device)
        attn = enc["attention_mask"].to(self.policy_device)

        # Switch to inference mode for rollout generation (faster + lower peak memory)
        was_training = bool(self.model.training)
        self.model.eval()

        if self._use_unsloth and (self._FastLanguageModel is not None):
            try:
                self._FastLanguageModel.for_inference(self.model)
            except Exception:
                pass

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            max_new_tokens=int(cfg.max_new_tokens),
            num_return_sequences=int(cfg.group_size),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
        )

        # Restore training mode
        if was_training:
            self.model.train()
            if self._use_unsloth and (self._FastLanguageModel is not None):
                try:
                    self._FastLanguageModel.for_training(self.model)
                except Exception:
                    pass

        tok.padding_side = old_padding_side

        seqs = gen.sequences
        B = len(prompt_texts)
        G = int(cfg.group_size)
        input_len = input_ids.shape[1]

        prompt_ids_list: List[List[int]] = []
        for i in range(B):
            prompt_ids_list.append(input_ids[i][attn[i].bool()].tolist())

        rollouts: List[Dict[str, Any]] = []
        for j in range(seqs.shape[0]):
            prompt_group_idx = j // G
            prompt_ids = prompt_ids_list[prompt_group_idx]

            completion_ids_full = seqs[j, input_len:]
            eos_idx = _first_eos_index(completion_ids_full, tok.eos_token_id)
            if eos_idx is not None:
                completion_ids = completion_ids_full[: eos_idx + 1].tolist()
            else:
                completion_ids = completion_ids_full.tolist()

            completion_text = tok.decode(completion_ids, skip_special_tokens=True)

            rollouts.append(
                {
                    "prompt_group_idx": prompt_group_idx,
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "completion_text": completion_text,
                }
            )

        return rollouts

    def _build_gold_trajectories(
        self,
        prompt_texts: Sequence[str],
        rows: Sequence[Dict[str, Any]],
        gold_fn: Callable[[Dict[str, Any]], str],
    ) -> List[Dict[str, Any]]:
        tok = self.tokenizer

        out: List[Dict[str, Any]] = []
        for pt, row in zip(prompt_texts, rows):
            prompt_ids = tok(pt, add_special_tokens=False).input_ids
            gold_completion = row.get("gold_completion") or gold_fn(row)
            comp_ids = tok(gold_completion, add_special_tokens=False).input_ids
            if not comp_ids or comp_ids[-1] != tok.eos_token_id:
                comp_ids = comp_ids + [tok.eos_token_id]
            out.append(
                {
                    "prompt_ids": prompt_ids,
                    "completion_ids": comp_ids,
                    "completion_text": gold_completion,
                }
            )
        return out

    # -----------------------
    # Advantages
    # -----------------------

    @staticmethod
    def _group_normalized_advantages(
        rewards_flat: Sequence[float], *, batch_size: int, group_size: int, eps: float = 1e-8
    ) -> List[float]:
        rewards_flat = list(map(float, rewards_flat))
        B, G = int(batch_size), int(group_size)

        out: List[float] = []
        for i in range(B):
            r = rewards_flat[i * G : (i + 1) * G]
            m = sum(r) / max(1, len(r))
            var = sum((x - m) ** 2 for x in r) / max(1, len(r))
            s = math.sqrt(var)
            if s < eps:
                out.extend([0.0] * G)
            else:
                out.extend([(x - m) / (s + eps) for x in r])
        return out

    @staticmethod
    def _gold_advantages(
        rewards_flat: Sequence[float],
        *,
        batch_size: int,
        group_size: int,
        gold_rewards: Sequence[float],
        eps: float = 1e-8,
    ) -> List[float]:
        rewards_flat = list(map(float, rewards_flat))
        gold_rewards = list(map(float, gold_rewards))
        B, G = int(batch_size), int(group_size)

        out: List[float] = []
        for i in range(B):
            r_int = rewards_flat[i * G : (i + 1) * G]
            gr = gold_rewards[i] if i < len(gold_rewards) else gold_rewards[0]
            r_all = list(r_int) + [gr]
            m = sum(r_all) / len(r_all)
            var = sum((x - m) ** 2 for x in r_all) / len(r_all)
            s = math.sqrt(var)
            if s < eps:
                out.append(0.0)
            else:
                out.append((gr - m) / (s + eps))
        return out

    # -----------------------
    # Ref logprobs helper (supports both modes)
    # -----------------------

    def _get_ref_logprobs(self, input_ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Get reference model log-probabilities.

        Two modes:
          - ref_use_disable_adapter=True  → disable LoRA on policy, forward, re-enable
          - ref_use_disable_adapter=False → use separate self.ref_model
        """
        with torch.no_grad():
            if self.ref_model is None:
                # Use base model (policy without LoRA adapters)
                self.model.eval()
                try:
                    self.model.disable_adapter_layers()
                    out = self.model(input_ids=input_ids, attention_mask=attn)
                    logp = _gather_logprobs(out.logits, input_ids)
                finally:
                    self.model.enable_adapter_layers()
                    self.model.train()
                return logp
            else:
                ref_dev = _model_device(self.ref_model)
                out = self.ref_model(
                    input_ids=input_ids.to(ref_dev),
                    attention_mask=attn.to(ref_dev),
                )
                return _gather_logprobs(out.logits, input_ids.to(ref_dev)).to(self.policy_device)

    # -----------------------
    # Memory-efficient loss (chunked backward)
    # -----------------------

    def _compute_loss_v2(
        self,
        *,
        rollouts: Sequence[Dict[str, Any]],
        advantages_flat: Sequence[float],
        gold_trajs: Optional[Sequence[Dict[str, Any]]],
        gold_adv: Optional[Sequence[float]],
        grad_scale: float = 1.0,
    ) -> Dict[str, float]:
        """Memory-efficient loss: processes internal term in chunks, calls backward per chunk.

        Unlike _compute_loss which returns a loss tensor, this method calls .backward()
        internally per chunk so that intermediate activations are freed between chunks.
        Returns only scalar logs (no loss tensor).
        """
        cfg = self.cfg
        tok = self.tokenizer
        chunk_size = max(1, int(cfg.loss_chunk_size)) if int(cfg.loss_chunk_size) > 0 else len(rollouts)

        # ===============
        # Internal term (CHUNKED)
        # ===============
        total_loss_int = 0.0
        total_kl = 0.0
        total_logp_pol = 0.0
        total_logp_ref = 0.0
        total_mask_tokens = 0.0

        for c_start in range(0, len(rollouts), chunk_size):
            c_end = min(c_start + chunk_size, len(rollouts))
            chunk_ro = rollouts[c_start:c_end]
            chunk_adv = advantages_flat[c_start:c_end]

            # Build padded tensors for this chunk
            full_seqs: List[List[int]] = []
            prompt_lens: List[int] = []
            full_lens: List[int] = []
            for ro in chunk_ro:
                p = list(map(int, ro["prompt_ids"]))
                c = list(map(int, ro["completion_ids"]))
                full_seqs.append(p + c)
                prompt_lens.append(len(p))
                full_lens.append(len(p) + len(c))

            input_ids, attn = _pad_right(full_seqs, pad_id=tok.pad_token_id)
            input_ids = input_ids.to(self.policy_device)
            attn = attn.to(self.policy_device)

            # Policy forward (WITH grad)
            out_pol = self.model(input_ids=input_ids, attention_mask=attn)
            logp_pol = _gather_logprobs(out_pol.logits, input_ids)
            del out_pol  # free logits immediately

            Tlp = logp_pol.size(1)
            comp_mask = _completion_mask_from_prompt_lens(
                [min(pl, Tlp + 1) for pl in prompt_lens],
                [min(fl, Tlp + 1) for fl in full_lens],
            ).to(self.policy_device)[:, :Tlp]

            # Ref forward (NO grad)
            logp_ref = self._get_ref_logprobs(input_ids[:, :Tlp + 1], attn[:, :Tlp + 1])
            Tlp2 = min(logp_pol.size(1), logp_ref.size(1), comp_mask.size(1))
            logp_pol = logp_pol[:, :Tlp2]
            logp_ref = logp_ref[:, :Tlp2]
            comp_mask = comp_mask[:, :Tlp2]

            del input_ids, attn  # free input tensors

            # GRPO loss for this chunk
            ratio = torch.exp(logp_pol - logp_pol.detach())  # always 1.0 but keeps grad
            ratio_clipped = torch.clamp(ratio, 1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps))

            adv_seq = torch.tensor(list(map(float, chunk_adv)), device=self.policy_device).unsqueeze(-1)
            kl_tok = (logp_pol - logp_ref).detach()
            adv_tok = adv_seq - float(cfg.beta_kl) * kl_tok

            pg = torch.minimum(ratio * adv_tok, ratio_clipped * adv_tok)
            denom = comp_mask.sum().clamp(min=1.0)
            chunk_loss_int = -(pg * comp_mask).sum() / denom

            # Accumulate stats (detached)
            n_toks = float(comp_mask.sum().detach().cpu())
            total_loss_int += float(chunk_loss_int.detach().cpu()) * n_toks
            total_kl += float((kl_tok * comp_mask).sum().detach().cpu())
            total_logp_pol += float((logp_pol.detach() * comp_mask).sum().cpu())
            total_logp_ref += float((logp_ref.detach() * comp_mask).sum().cpu())
            total_mask_tokens += n_toks

            # Backward for this chunk (frees activations immediately)
            (chunk_loss_int * grad_scale).backward()

            del logp_pol, logp_ref, comp_mask, chunk_loss_int, pg, ratio, kl_tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ===============
        # External term (gold — small, process in one go)
        # ===============
        loss_ext_val = 0.0
        r_m_mean_val = 0.0
        r_m_clip_frac_val = 0.0
        mask_sum_ext = 0.0

        if cfg.lambda_ext and cfg.lambda_ext > 0 and gold_trajs is not None and gold_adv is not None:
            full_seqs_ext: List[List[int]] = []
            prompt_lens_ext: List[int] = []
            full_lens_ext: List[int] = []
            for tr in gold_trajs:
                p = list(map(int, tr["prompt_ids"]))
                c = list(map(int, tr["completion_ids"]))
                full_seqs_ext.append(p + c)
                prompt_lens_ext.append(len(p))
                full_lens_ext.append(len(p) + len(c))

            input_ids_ext, attn_ext = _pad_right(full_seqs_ext, pad_id=tok.pad_token_id)
            input_ids_ext = input_ids_ext.to(self.policy_device)
            attn_ext = attn_ext.to(self.policy_device)

            # Policy forward (WITH grad)
            out_pol_ext = self.model(input_ids=input_ids_ext, attention_mask=attn_ext)
            logp_pol_ext = _gather_logprobs(out_pol_ext.logits, input_ids_ext)
            del out_pol_ext

            Tlp_e = logp_pol_ext.size(1)
            comp_mask_ext = _completion_mask_from_prompt_lens(
                [min(pl, Tlp_e + 1) for pl in prompt_lens_ext],
                [min(fl, Tlp_e + 1) for fl in full_lens_ext],
            ).to(self.policy_device)[:, :Tlp_e]

            logp_old_ext = logp_pol_ext.detach()

            # Behavior model logprobs (no grad)
            beh_model = self.behavior_model if self.behavior_model is not None else (self.ref_model if self.ref_model is not None else None)
            if beh_model is not None:
                with torch.no_grad():
                    beh_dev = _model_device(beh_model)
                    out_beh = beh_model(
                        input_ids=input_ids_ext[:, :Tlp_e + 1].to(beh_dev),
                        attention_mask=attn_ext[:, :Tlp_e + 1].to(beh_dev),
                    )
                    logp_omega = _gather_logprobs(out_beh.logits, input_ids_ext[:, :Tlp_e + 1].to(beh_dev)).to(self.policy_device)
                    del out_beh
            else:
                # No ref model (disable_adapter mode) — use base model
                logp_omega = self._get_ref_logprobs(input_ids_ext[:, :Tlp_e + 1], attn_ext[:, :Tlp_e + 1])

            del input_ids_ext, attn_ext

            Tlp2_e = min(logp_pol_ext.size(1), logp_omega.size(1), comp_mask_ext.size(1))
            logp_pol_ext = logp_pol_ext[:, :Tlp2_e]
            logp_old_ext = logp_old_ext[:, :Tlp2_e]
            logp_omega = logp_omega[:, :Tlp2_e]
            comp_mask_ext = comp_mask_ext[:, :Tlp2_e]

            mask_sum_ext = float(comp_mask_ext.sum().detach().cpu())

            log_denom = torch.logsumexp(torch.stack([logp_omega, logp_old_ext], dim=0), dim=0)
            log_r_m = math.log(2.0) + logp_pol_ext - log_denom
            r_m_unclamped = torch.exp(log_r_m)
            r_m = r_m_unclamped
            if cfg.r_m_max and cfg.r_m_max > 0:
                r_m = torch.clamp(r_m, 0.0, float(cfg.r_m_max))
                r_m_clip_frac_val = float(((r_m_unclamped > float(cfg.r_m_max)).float() * comp_mask_ext).sum() / comp_mask_ext.sum().clamp(min=1.0))

            p_token = torch.exp(logp_pol_ext).detach()
            one_minus_p = torch.clamp(1.0 - p_token, min=0.0, max=1.0)
            gamma = float(cfg.gamma_focal)
            C = torch.ones_like(one_minus_p) if gamma <= 0.0 else torch.pow(one_minus_p, gamma)

            A_gold = torch.tensor(list(map(float, gold_adv)), device=self.policy_device).unsqueeze(-1)
            A_c = A_gold * C

            denom_ext = comp_mask_ext.sum().clamp(min=1.0)
            loss_ext = -((r_m * A_c) * comp_mask_ext).sum() / denom_ext

            loss_ext_val = float(loss_ext.detach().cpu())
            r_m_mean_val = float((r_m * comp_mask_ext).sum().detach().cpu() / denom_ext)

            # Backward for gold term
            (float(cfg.lambda_ext) * loss_ext * grad_scale).backward()

            del logp_pol_ext, logp_omega, comp_mask_ext, loss_ext
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Assemble logs
        denom_total = max(1.0, total_mask_tokens)
        loss_int_val = total_loss_int / denom_total
        total_loss_val = loss_int_val + float(cfg.lambda_ext) * loss_ext_val

        logs = {
            "loss": total_loss_val,
            "loss_int": loss_int_val,
            "loss_ext": loss_ext_val,
            "kl": total_kl / denom_total,
            "logp_pol_mean": total_logp_pol / denom_total,
            "logp_ref_mean": total_logp_ref / denom_total,
            "r_m_mean": r_m_mean_val,
            "r_m_clip_frac": r_m_clip_frac_val,
            "comp_mask_sum_int": total_mask_tokens,
            "comp_mask_sum_ext": mask_sum_ext,
        }
        return logs

    # -----------------------
    # Loss (original, kept for backward compat)
    # -----------------------

    def _compute_loss(
    self,
    *,
    rollouts: Sequence[Dict[str, Any]],
    advantages_flat: Sequence[float],
    gold_trajs: Optional[Sequence[Dict[str, Any]]],
    gold_adv: Optional[Sequence[float]],
) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        tok = self.tokenizer

        # --- mask debug (safe defaults)
        mask_sum_int  = 0.0
        mask_frac_int = 0.0
        mask_sum_ext  = 0.0
        mask_frac_ext = 0.0

        # ===============
        # Internal term
        # ===============
        full_seqs_int: List[List[int]] = []
        prompt_lens_int: List[int] = []
        full_lens_int: List[int] = []

        for ro in rollouts:
            prompt_ids = list(map(int, ro["prompt_ids"]))
            comp_ids   = list(map(int, ro["completion_ids"]))
            full = prompt_ids + comp_ids
            full_seqs_int.append(full)
            prompt_lens_int.append(len(prompt_ids))
            full_lens_int.append(len(full))

        input_ids_int, attn_int = _pad_right(full_seqs_int, pad_id=tok.pad_token_id)
        comp_mask_int = _completion_mask_from_prompt_lens(prompt_lens_int, full_lens_int)

        input_ids_int  = input_ids_int.to(self.policy_device)
        attn_int       = attn_int.to(self.policy_device)
        comp_mask_int  = comp_mask_int.to(self.policy_device)

        # policy forward (GRAD MUST FLOW)
        out_pol = self.model(input_ids=input_ids_int, attention_mask=attn_int)
        token_logp_pol = _gather_logprobs(out_pol.logits, input_ids_int)

        # Align masks to token_logp length (T-1)
        Tlp = token_logp_pol.size(1)
        attn_int = attn_int[:, :Tlp + 1]
        comp_mask_int = comp_mask_int[:, :Tlp]

        # ref forward (NO GRAD)
        with torch.no_grad():
            token_logp_ref = self._get_ref_logprobs(input_ids_int[:, :Tlp + 1], attn_int[:, :Tlp + 1])

        # ===== IMPORTANT: align OUTSIDE no_grad, иначе отцепится граф policy =====
        Tlp2 = min(token_logp_pol.size(1), token_logp_ref.size(1), comp_mask_int.size(1))
        token_logp_pol = token_logp_pol[:, :Tlp2]
        token_logp_ref = token_logp_ref[:, :Tlp2]
        comp_mask_int  = comp_mask_int[:, :Tlp2]

        # ---- REBUILD completion mask after truncation (optional, но полезно)
        eff_len = int(Tlp2 + 1)  # token-length, logp/mask length is Tlp2
        prompt_lens_eff = [min(int(pl), eff_len) for pl in prompt_lens_int]
        full_lens_eff   = [min(int(fl), eff_len) for fl in full_lens_int]
        comp_mask_int = _completion_mask_from_prompt_lens(prompt_lens_eff, full_lens_eff).to(self.policy_device)
        comp_mask_int = comp_mask_int[:, :Tlp2]

        # mask stats
        mask_sum_int  = float(comp_mask_int.sum().detach().cpu())
        mask_frac_int = float((comp_mask_int > 0).float().mean().detach().cpu())

        # PPO-ish / GRPO objective (твоя логика)
        token_logp_old = token_logp_pol.detach()
        ratio = torch.exp(token_logp_pol - token_logp_old)
        ratio_clipped = torch.clamp(ratio, 1.0 - float(cfg.clip_eps), 1.0 + float(cfg.clip_eps))

        adv_seq = torch.tensor(list(map(float, advantages_flat)), device=self.policy_device).unsqueeze(-1)

        kl_tok_det = (token_logp_pol - token_logp_ref).detach()
        adv_tok = adv_seq - float(cfg.beta_kl) * kl_tok_det

        pg = torch.minimum(ratio * adv_tok, ratio_clipped * adv_tok)

        denom_int = comp_mask_int.sum().clamp(min=1.0)
        loss_int = -(pg * comp_mask_int).sum() / denom_int

        # logging KL (detached)
        kl = (kl_tok_det * comp_mask_int).sum() / denom_int
        logp_pol_mean = (token_logp_pol.detach() * comp_mask_int).sum() / denom_int
        logp_ref_mean = (token_logp_ref.detach() * comp_mask_int).sum() / denom_int

        # ===============
        # External term (RL-PLUS)
        # ===============
        loss_ext = torch.tensor(0.0, device=self.policy_device)
        r_m_mean = torch.tensor(0.0, device=self.policy_device)
        r_m_clip_frac = torch.tensor(0.0, device=self.policy_device)

        if cfg.lambda_ext and cfg.lambda_ext > 0:
            if gold_trajs is None or gold_adv is None:
                raise ValueError("gold_trajs/gold_adv must be provided when lambda_ext>0")

            full_seqs_ext: List[List[int]] = []
            prompt_lens_ext: List[int] = []
            full_lens_ext: List[int] = []

            for tr in gold_trajs:
                prompt_ids = list(map(int, tr["prompt_ids"]))
                comp_ids   = list(map(int, tr["completion_ids"]))
                full = prompt_ids + comp_ids
                full_seqs_ext.append(full)
                prompt_lens_ext.append(len(prompt_ids))
                full_lens_ext.append(len(full))

            input_ids_ext, attn_ext = _pad_right(full_seqs_ext, pad_id=tok.pad_token_id)
            comp_mask_ext = _completion_mask_from_prompt_lens(prompt_lens_ext, full_lens_ext)

            input_ids_ext = input_ids_ext.to(self.policy_device)
            attn_ext      = attn_ext.to(self.policy_device)
            comp_mask_ext = comp_mask_ext.to(self.policy_device)

            out_pol_ext = self.model(input_ids=input_ids_ext, attention_mask=attn_ext)
            token_logp_pol_ext = _gather_logprobs(out_pol_ext.logits, input_ids_ext)

            Tlp_e = token_logp_pol_ext.size(1)
            attn_ext = attn_ext[:, :Tlp_e + 1]
            comp_mask_ext = comp_mask_ext[:, :Tlp_e]

            token_logp_old_ext = token_logp_pol_ext.detach()

            with torch.no_grad():
                beh_model = self.behavior_model if self.behavior_model is not None else self.ref_model
                if beh_model is not None:
                    beh_dev = _model_device(beh_model)
                    input_ids_ext_b = input_ids_ext[:, :Tlp_e + 1].to(beh_dev)
                    attn_ext_b      = attn_ext.to(beh_dev)
                    out_beh = beh_model(input_ids=input_ids_ext_b, attention_mask=attn_ext_b)
                    token_logp_omega = _gather_logprobs(out_beh.logits, input_ids_ext_b).to(self.policy_device)
                else:
                    # No ref/behavior model — use base model via disable_adapter
                    token_logp_omega = self._get_ref_logprobs(input_ids_ext[:, :Tlp_e + 1], attn_ext[:, :Tlp_e + 1])

            # align (уже вне no_grad — это ок)
            Tlp2_e = min(token_logp_pol_ext.size(1), token_logp_omega.size(1), comp_mask_ext.size(1))
            token_logp_pol_ext = token_logp_pol_ext[:, :Tlp2_e]
            token_logp_old_ext = token_logp_old_ext[:, :Tlp2_e]
            token_logp_omega   = token_logp_omega[:, :Tlp2_e]
            comp_mask_ext      = comp_mask_ext[:, :Tlp2_e]

            # rebuild ext mask
            eff_len_ext = int(Tlp2_e + 1)
            prompt_lens_eff_ext = [min(int(pl), eff_len_ext) for pl in prompt_lens_ext]
            full_lens_eff_ext   = [min(int(fl), eff_len_ext) for fl in full_lens_ext]
            comp_mask_ext = _completion_mask_from_prompt_lens(prompt_lens_eff_ext, full_lens_eff_ext).to(self.policy_device)
            comp_mask_ext = comp_mask_ext[:, :Tlp2_e]

            mask_sum_ext  = float(comp_mask_ext.sum().detach().cpu())
            mask_frac_ext = float((comp_mask_ext > 0).float().mean().detach().cpu())

            log_denom = torch.logsumexp(torch.stack([token_logp_omega, token_logp_old_ext], dim=0), dim=0)
            log_r_m = math.log(2.0) + token_logp_pol_ext - log_denom
            r_m_unclamped = torch.exp(log_r_m)

            r_m = r_m_unclamped
            if cfg.r_m_max and cfg.r_m_max > 0:
                r_m = torch.clamp(r_m, 0.0, float(cfg.r_m_max))
                r_m_clip_frac = ((r_m_unclamped > float(cfg.r_m_max)).float() * comp_mask_ext).sum() / comp_mask_ext.sum().clamp(min=1.0)

            p_token = torch.exp(token_logp_pol_ext).detach()
            one_minus_p = torch.clamp(1.0 - p_token, min=0.0, max=1.0)
            gamma = float(cfg.gamma_focal)
            C = torch.ones_like(one_minus_p) if gamma <= 0.0 else torch.pow(one_minus_p, gamma)

            A_gold = torch.tensor(list(map(float, gold_adv)), device=self.policy_device).unsqueeze(-1)
            A_c = A_gold * C

            denom_ext = comp_mask_ext.sum().clamp(min=1.0)
            loss_ext = -((r_m * A_c) * comp_mask_ext).sum() / denom_ext
            r_m_mean = (r_m * comp_mask_ext).sum() / denom_ext

        loss = loss_int + float(cfg.lambda_ext) * loss_ext

        logs = {
            "loss": _to_float(loss),
            "loss_int": _to_float(loss_int),
            "loss_ext": _to_float(loss_ext),
            "kl": _to_float(kl),
            "logp_pol_mean": _to_float(logp_pol_mean),
            "logp_ref_mean": _to_float(logp_ref_mean),
            "r_m_mean": _to_float(r_m_mean),
            "r_m_clip_frac": _to_float(r_m_clip_frac),
            "comp_mask_sum_int": float(mask_sum_int),
            "comp_mask_frac_int": float(mask_frac_int),
            "comp_mask_sum_ext": float(mask_sum_ext),
            "comp_mask_frac_ext": float(mask_frac_ext),
            # очень полезный детектор "отцепили граф"
            "pol_logp_is_leaf": float(token_logp_pol.is_leaf),
        }

        return loss, logs


# ----------------------------
# Callback builder (helper)
# ----------------------------

def make_periodic_eval_callback(
    *,
    trainer: RLPlusTrainer,
    eval_sets: Dict[str, Sequence[Dict[str, Any]]],
    verify_fn: Optional[Callable[[Dict[str, Any], str], bool]] = None,
    n_samples: int = 1,
    ks: Tuple[int, ...] = (1, 2, 4, 8),
    max_items: Optional[int] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: Optional[int] = None,
    batch_size: int = 16,
) -> Callable[[int], None]:
    """Build a callback you can pass into RLPlusTrainer.train(..., eval_callback=...)."""

    def _cb(step: int) -> None:
        for name, rows in eval_sets.items():
            trainer.evaluate_rows(
                rows=rows,
                verify_fn=verify_fn,
                name=name,
                max_items=max_items,
                n_samples=int(n_samples),
                ks=ks,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=max_new_tokens,
                batch_size=int(batch_size),
                step=int(step),
            )

    return _cb


__all__ = [
    "RLPlusConfig",
    "RLPlusTrainer",
    "make_periodic_eval_callback",
]

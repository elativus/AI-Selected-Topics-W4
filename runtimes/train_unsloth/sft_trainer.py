"""Minimal SFT (supervised fine-tuning) trainer for Week 3.

Goal: fit a LoRA adapter on *gold trajectories* so we can run:
  - SFT → GRPO
  - use the SFT model as pi_omega (behavior) in RL-PLUS MIS denominator

This trainer is intentionally simple:
  - plain next-token LM loss with prompt masking (loss only on assistant completion)
  - LoRA via PEFT (or via Unsloth helper if enabled)

Safe to import in a *training* environment without vLLM.
"""

from __future__ import annotations

import gc
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from runtimes.train_unsloth.triage_rl_plus_compat import SYSTEM_PROMPT, build_chat_prompt, get_user_prompt


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "outputs/sft_adapter"
    system_prompt: str = SYSTEM_PROMPT

    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 200  # optimizer steps
    batch_size: int = 4
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    max_seq_len: int = 512

    # PEFT (LoRA)
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
    load_in_4bit: bool = False
    torch_dtype: str = "bf16"
    device: str = "cuda"  # "cuda" | "cpu"
    device_map: Optional[str] = None  # None|"auto"|explicit

    enable_gradient_checkpointing: bool = True

    # -----------------
    # Backend speed-ups (optional)
    # -----------------
    use_unsloth: bool = False
    # If True, uses Unsloth smart gradient checkpointing (can touch ALL visible GPUs).
    # If you see `CUDA driver error: unknown error`, keep this False.
    unsloth_smart_gradient_checkpointing: bool = False

    unsloth_max_seq_length: int = 512
    unsloth_disable_compile: bool = False
    unsloth_fullgraph: bool = False
    unsloth_compile_ignore_errors: bool = True
    clear_unsloth_cache: bool = True
    disable_torchdynamo: bool = False
    disable_torch_compile: bool = False

    # Optimizer
    optim: str = "adamw"  # "adamw" | "adamw_8bit" (bitsandbytes)

    # -----------------
    # Logging
    # -----------------
    log_every: int = 10

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


def _configure_unsloth_env(cfg: SFTConfig) -> None:
    """Set env vars that control Unsloth + torch.compile behavior.

    IMPORTANT: must be called BEFORE importing `unsloth`.
    """
    if bool(getattr(cfg, "clear_unsloth_cache", False)):
        try:
            shutil.rmtree(Path.cwd() / "unsloth_compiled_cache", ignore_errors=True)
        except Exception:
            pass

    os.environ["UNSLOTH_FULLGRAPH"] = "1" if bool(getattr(cfg, "unsloth_fullgraph", False)) else "0"
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1" if bool(getattr(cfg, "unsloth_disable_compile", False)) else "0"
    if bool(getattr(cfg, "unsloth_compile_ignore_errors", True)):
        os.environ["UNSLOTH_COMPILE_IGNORE_ERRORS"] = "1"

    if bool(getattr(cfg, "disable_torchdynamo", False)):
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if bool(getattr(cfg, "disable_torch_compile", False)):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"


def _make_optimizer(cfg: SFTConfig, trainable_params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
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

    return torch.optim.AdamW(trainable_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))


class _SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tok, cfg: SFTConfig):
        self.rows = list(rows)
        self.tok = tok
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        prompt_text = build_chat_prompt(self.tok, get_user_prompt(row), system_prompt=self.cfg.system_prompt)
        completion_text = str(row["gold_completion"])

        prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        completion_ids = self.tok(completion_text, add_special_tokens=False).input_ids
        # Ensure completion ends with EOS so model learns to stop after </ACTIONS>
        if not completion_ids or completion_ids[-1] != self.tok.eos_token_id:
            completion_ids = completion_ids + [self.tok.eos_token_id]
        full_ids = prompt_ids + completion_ids

        boundary = len(prompt_ids)
        if (len(full_ids) < boundary) or (full_ids[:boundary] != prompt_ids):
            # Rare: some tokenizers can merge across the prompt/completion boundary.
            # Fall back to separate tokenization to keep the completion mask correct.
            full_ids = prompt_ids + completion_ids
            boundary = len(prompt_ids)

        # loss mask: only completion tokens
        labels = [-100] * len(full_ids)
        for j in range(boundary, len(full_ids)):
            labels[j] = int(full_ids[j])

        # Truncate
        if len(full_ids) > int(self.cfg.max_seq_len):
            full_ids = full_ids[-int(self.cfg.max_seq_len) :]
            labels = labels[-int(self.cfg.max_seq_len) :]

        return {"input_ids": full_ids, "labels": labels}


def _collate_sft(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    # Right-pad for plain LM training
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        ids = list(x["input_ids"])
        lab = list(x["labels"])
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_n)
        labels.append(lab + [-100] * pad_n)
        attn.append([1] * len(ids) + [0] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def _init_comet(cfg: SFTConfig) -> Optional[Any]:
    if not cfg.comet_project or bool(cfg.comet_disabled):
        return None
    try:
        from comet_ml import Experiment  # type: ignore
    except Exception as e:
        print(f"[WARN] comet_ml not available: {e}")
        return None

    OfflineExperiment = None
    if bool(cfg.comet_offline):
        try:
            from comet_ml import OfflineExperiment as _OfflineExperiment  # type: ignore

            OfflineExperiment = _OfflineExperiment
        except Exception:
            OfflineExperiment = None

    exp_cls = OfflineExperiment if (OfflineExperiment is not None) else Experiment

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
            params.pop("system_prompt", None)
            exp.log_parameters(params)
        except Exception:
            pass

    return exp


def train_sft_lora(
    *,
    train_rows: Sequence[Dict[str, Any]],
    cfg: SFTConfig,
) -> Tuple[str, List[Dict[str, float]]]:
    """Train an SFT LoRA adapter on gold trajectories.

    Returns:
      (output_dir, history) where history contains {"step":..., "loss":...}.
    """
    set_seed(cfg.seed)
    torch_dtype = _torch_dtype_from_str(cfg.torch_dtype)

    use_unsloth = bool(cfg.use_unsloth) and str(cfg.device).lower() == "cuda"
    FLM = None

    if use_unsloth:
        _configure_unsloth_env(cfg)
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except Exception as e:
            print(f"[WARN] Unsloth import failed -> fallback to Transformers. Reason: {e}")
            use_unsloth = False
        else:
            FLM = FastLanguageModel

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # -----------------
    # Tokenizer + base model
    # -----------------
    if use_unsloth and (FLM is not None):
        # IMPORTANT: Unsloth's *smart* gradient checkpointing tries to allocate buffers on ALL visible GPUs.
        # On some multi-GPU setups this can crash with `CUDA driver error: unknown error`.
        # We therefore disable it by default and rely on standard HF checkpointing instead.
        model, tok = FLM.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=int(cfg.unsloth_max_seq_length),
            dtype=torch_dtype,
            load_in_4bit=bool(cfg.load_in_4bit),
            use_gradient_checkpointing=("unsloth" if bool(cfg.unsloth_smart_gradient_checkpointing) else False),
            trust_remote_code=True,
        )
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        try:
            model.config.use_cache = False
        except Exception:
            pass

        if cfg.use_lora:
            model = FLM.get_peft_model(
                model,
                r=int(cfg.lora_r),
                target_modules=list(cfg.lora_target_modules),
                lora_alpha=int(cfg.lora_alpha),
                lora_dropout=float(cfg.lora_dropout),
                bias="none",
                use_gradient_checkpointing=("unsloth" if bool(cfg.unsloth_smart_gradient_checkpointing) else False),
                random_state=int(cfg.seed),
                max_seq_length=int(cfg.unsloth_max_seq_length),
            )

        try:
            FLM.for_training(model)
        except Exception:
            pass

        # If we disabled Unsloth smart GC, we can still enable standard HF gradient checkpointing.
        if bool(cfg.enable_gradient_checkpointing) and (not bool(cfg.unsloth_smart_gradient_checkpointing)):
            try:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception:
                pass

    else:
        tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        model_kwargs: Dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch_dtype)

        # Avoid device_map="auto" by default to prevent sharding across all GPUs.
        if cfg.load_in_4bit:
            model_kwargs.update(dict(load_in_4bit=True, device_map=cfg.device_map or "auto"))
        else:
            if str(cfg.device).lower() == "cuda" and cfg.device_map:
                model_kwargs.update(dict(device_map=cfg.device_map))
            else:
                model_kwargs.update(dict(device_map=None))

        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        try:
            model.config.use_cache = False
        except Exception:
            pass

        if (not cfg.load_in_4bit) and str(cfg.device).lower() == "cuda" and not cfg.device_map:
            model.to("cuda")

        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=int(cfg.lora_r),
                lora_alpha=int(cfg.lora_alpha),
                lora_dropout=float(cfg.lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(cfg.lora_target_modules),
            )
            model = get_peft_model(model, lora_cfg)

        if bool(cfg.enable_gradient_checkpointing):
            try:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            except Exception:
                pass

    # Print trainable parameters
    try:
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    except Exception:
        pass

    device = _model_device(model)

    # Dataset
    ds = _SFTDataset(train_rows, tok, cfg)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=lambda b: _collate_sft(b, pad_id=tok.pad_token_id),
    )

    optim = _make_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])

    model.train()

    comet = _init_comet(cfg)

    history: List[Dict[str, float]] = []
    it = iter(dl)

    pbar = tqdm(range(int(cfg.max_steps)), desc="SFT train", unit="step")
    for step in pbar:
        optim.zero_grad(set_to_none=True)
        loss_acc = 0.0

        for _ in range(int(cfg.grad_accum_steps)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / float(cfg.grad_accum_steps)
            loss.backward()
            loss_acc += float(loss.detach().cpu().item())

        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(cfg.max_grad_norm),
            )

        optim.step()

        row = {"step": float(step + 1), "loss": float(loss_acc)}
        history.append(row)

        pbar.set_postfix({"loss": f"{loss_acc:.4f}"})

        if comet is not None:
            try:
                comet.log_metric("sft/loss", float(loss_acc), step=int(step + 1))
            except Exception:
                pass

        if cfg.log_every and (step + 1) % int(cfg.log_every) == 0:
            print(f"[SFT] step {step+1}/{cfg.max_steps} | loss={loss_acc:.4f}")

    out_dir = cfg.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Cleanup
    try:
        if comet is not None:
            comet.end()
    except Exception:
        pass

    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_dir, history


__all__ = ["SFTConfig", "train_sft_lora"]

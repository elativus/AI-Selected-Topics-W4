from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from triage.artifacts import read_manifest
from triage.env import SafeTriageEnv
from triage.io_utils import read_dataset, save_jsonl
from triage.metrics import aggregate_episode_metrics, summarize_failure_reasons
from triage.prompting import (
    build_interactive_messages,
    build_trajectory_messages,
    render_chat_prompt,
)
from triage.trajectory_text import extract_actions, extract_single_action
from triage.verifier import TriageTrajectoryVerifier

from runtimes.eval_vllm.config import VLLMEvalConfig
from runtimes.eval_vllm.vllm_utils import generate_with_vllm, maybe_make_lora_request

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger(__name__)
_PROMPT_RENDER_LOG_KEYS: set[Tuple[Any, ...]] = set()

RolloutMode = Literal["interactive", "trajectory"]

# Qwen/Qwen-like assistant-prefill patterns that can inject thinking mode.
_ASSISTANT_THINK_TAIL_PATTERNS = [
    re.compile(r"(<\|im_start\|>assistant\s*)<think>\s*</think>\s*$", re.DOTALL),
    re.compile(r"(<\|im_start\|>assistant\s*)<think>\s*$", re.DOTALL),
]


@dataclass
class VLLMRolloutResult:
    cfg: VLLMEvalConfig
    summary: Dict[str, Any]
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)

    def trajectories_df(self):
        import pandas as pd

        return pd.DataFrame(self.trajectories)

    def metrics_df(self):
        import pandas as pd

        return pd.DataFrame(self.metrics)

    def summary_df(self):
        import pandas as pd

        return pd.DataFrame([self.summary])


def _get_cfg(cfg: VLLMEvalConfig, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _ensure_parent(path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_model_and_lora(cfg: VLLMEvalConfig) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve model_path, optional lora_adapter_dir, optional manifest_path."""
    if cfg.manifest:
        manifest = read_manifest(cfg.manifest)

        prefer_lora_adapter = bool(_get_cfg(cfg, "prefer_lora_adapter", False))
        prefer_merged_model = bool(_get_cfg(cfg, "prefer_merged_model", True))

        if prefer_lora_adapter and (cfg.lora_adapter_dir or manifest.adapter_dir):
            return manifest.base_model, (cfg.lora_adapter_dir or manifest.adapter_dir), cfg.manifest

        if prefer_merged_model and manifest.merged_model_dir:
            return manifest.resolved_eval_model_path(), None, cfg.manifest

        if cfg.lora_adapter_dir or manifest.adapter_dir:
            return manifest.base_model, (cfg.lora_adapter_dir or manifest.adapter_dir), cfg.manifest

        return manifest.resolved_eval_model_path(), None, cfg.manifest

    if cfg.model is None:
        raise ValueError("Provide either a manifest/config with a model or pass cfg.model.")
    return cfg.model, cfg.lora_adapter_dir, None


def _sanitize_prompt_tail(prompt: str) -> str:
    """Remove assistant-side '<think>' prefill if the chat template injected it.

    Safety net: even with /no_think + enable_thinking=False, some tokenizer
    versions may still emit <think> in the assistant prefix.
    """
    out = prompt
    for pat in _ASSISTANT_THINK_TAIL_PATTERNS:
        match = pat.search(out)
        if match:
            out = out[: match.start()] + match.group(1)
    return out


def _log_prompt_render_meta(logger: logging.Logger, meta: Dict[str, Any]) -> None:
    key = (
        meta.get("renderer"),
        meta.get("enable_thinking_requested"),
        meta.get("enable_thinking_applied"),
        meta.get("fallback_reason"),
        meta.get("assistant_think_tail_sanitized"),
    )
    if key in _PROMPT_RENDER_LOG_KEYS:
        return
    _PROMPT_RENDER_LOG_KEYS.add(key)
    logger.info(
        "Prompt rendering: renderer=%s "
        "enable_thinking_requested=%s enable_thinking_applied=%s "
        "fallback_reason=%s assistant_think_tail_sanitized=%s",
        meta.get("renderer"),
        meta.get("enable_thinking_requested"),
        meta.get("enable_thinking_applied"),
        meta.get("fallback_reason"),
        meta.get("assistant_think_tail_sanitized"),
    )


def _render_prompt(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = False,
    logger: Optional[logging.Logger] = None,
) -> str:
    prompt, meta = render_chat_prompt(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        return_meta=True,
    )
    # Safety: strip any <think> the template injected in non-thinking mode.
    sanitized = False
    if enable_thinking in (False, None):
        sanitized_prompt = _sanitize_prompt_tail(prompt)
        sanitized = sanitized_prompt != prompt
        prompt = sanitized_prompt
    meta["assistant_think_tail_sanitized"] = sanitized
    _log_prompt_render_meta(logger or LOGGER, meta)
    return prompt


def _make_sampling_params(cfg: VLLMEvalConfig):
    from vllm import SamplingParams  # imported lazily

    kwargs: Dict[str, Any] = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }

    # Keep compatibility across vLLM versions where some fields may be absent.
    try:
        sig = inspect.signature(SamplingParams.__init__)
        if "top_k" in sig.parameters:
            kwargs["top_k"] = int(_get_cfg(cfg, "top_k", -1))
        if "min_p" in sig.parameters:
            kwargs["min_p"] = float(_get_cfg(cfg, "min_p", 0.0))
    except Exception:
        kwargs["top_k"] = int(_get_cfg(cfg, "top_k", -1))
        kwargs["min_p"] = float(_get_cfg(cfg, "min_p", 0.0))

    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("top_k", None)
        kwargs.pop("min_p", None)
        return SamplingParams(**kwargs)


def _generate_one_text(
    llm: Any,
    prompt: str,
    sampling_params: Any,
    *,
    lora_request: Any = None,
    use_tqdm: bool = False,
    suppress_output: bool = True,
) -> str:
    outputs = generate_with_vllm(
        llm,
        [prompt],
        sampling_params,
        lora_request=lora_request,
        use_tqdm=use_tqdm,
        suppress_output=suppress_output,
    )
    if not outputs:
        return ""
    first = outputs[0]
    if not getattr(first, "outputs", None):
        return ""
    return first.outputs[0].text or ""


def _iter_cases(
    cases: Sequence[Any],
    *,
    progress: bool,
    desc: str,
):
    if progress and tqdm is not None:
        return tqdm(cases, desc=desc)
    return cases


# ---------------------------------------------------------------------------
# Trajectory (one-shot) rollout
# ---------------------------------------------------------------------------

def _run_single_case_trajectory(
    *,
    case: Any,
    tokenizer: Any,
    llm: Any,
    sampling_params: Any,
    lora_request: Any,
    cfg: VLLMEvalConfig,
    verifier: TriageTrajectoryVerifier,
    model_path: str,
    manifest_path: Optional[str],
    lora_adapter_dir: Optional[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    env = SafeTriageEnv()
    initial_observation = env.reset(case)

    allow_reasoning = bool(_get_cfg(cfg, "enable_thinking", False))
    messages = build_trajectory_messages(
        initial_observation,
        case.max_steps,
        allow_reasoning=allow_reasoning,
    )
    prompt = _render_prompt(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=allow_reasoning,
        logger=logger,
    )

    completion_text = _generate_one_text(
        llm,
        prompt,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
        suppress_output=bool(cfg.suppress_vllm_output),
    )

    actions = extract_actions(completion_text, max_actions=case.max_steps)

    metrics = verifier.verify_trajectory(SafeTriageEnv(), case, actions)
    metrics["case_id"] = case.case_id
    metrics["difficulty"] = case.difficulty

    traj_row: Dict[str, Any] = {
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "rollout_mode": "trajectory",
        "prompt": prompt,
        "completion_text": completion_text,
        "actions": actions,
        "model_path": model_path,
        "lora_adapter_dir": lora_adapter_dir,
    }
    if manifest_path is not None:
        traj_row["manifest"] = str(manifest_path)

    return traj_row, metrics


# ---------------------------------------------------------------------------
# Interactive (step-by-step) rollout
# ---------------------------------------------------------------------------

def _run_single_case_interactive(
    *,
    case: Any,
    tokenizer: Any,
    llm: Any,
    sampling_params: Any,
    lora_request: Any,
    cfg: VLLMEvalConfig,
    verifier: TriageTrajectoryVerifier,
    model_path: str,
    manifest_path: Optional[str],
    lora_adapter_dir: Optional[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    env = SafeTriageEnv()
    initial_observation = env.reset(case)
    observation = initial_observation

    history: List[Dict[str, Any]] = []
    actions: List[str] = []
    step_records: List[Dict[str, Any]] = []

    allow_reasoning = bool(_get_cfg(cfg, "enable_thinking", False))

    for step_idx in range(case.max_steps):
        messages = build_interactive_messages(
            history=history,
            current_observation=observation,
            max_steps_remaining=case.max_steps - step_idx,
            allow_reasoning=allow_reasoning,
        )
        prompt = _render_prompt(
            tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=allow_reasoning,
            logger=logger,
        )

        raw_completion = _generate_one_text(
            llm,
            prompt,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
            suppress_output=bool(cfg.suppress_vllm_output),
        )
        action = extract_single_action(raw_completion)

        if not action:
            step_records.append(
                {
                    "step_idx": step_idx + 1,
                    "prompt": prompt,
                    "raw_completion": raw_completion,
                    "parsed_action": "",
                    "observation_before": observation,
                    "observation_after": None,
                    "reward": None,
                    "done": False,
                    "info": None,
                }
            )
            break

        next_observation, reward, done, info = env.step(action)
        actions.append(action)
        step_records.append(
            {
                "step_idx": step_idx + 1,
                "prompt": prompt,
                "raw_completion": raw_completion,
                "parsed_action": action,
                "observation_before": observation,
                "observation_after": next_observation,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

        history.append(
            {
                "action": action,
                "observation": next_observation,
                "case_id": case.case_id,
                "difficulty": case.difficulty,
                "max_steps": case.max_steps,
            }
        )
        observation = next_observation

        if done:
            break

    metrics = verifier.verify_trajectory(SafeTriageEnv(), case, actions)
    metrics["case_id"] = case.case_id
    metrics["difficulty"] = case.difficulty

    traj_row: Dict[str, Any] = {
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "rollout_mode": "interactive",
        "initial_observation": initial_observation,
        "actions": actions,
        "completion_text": "\n\n===== STEP COMPLETIONS =====\n\n".join(
            (rec["raw_completion"] or "") for rec in step_records
        ),
        "step_records": step_records,
        "model_path": model_path,
        "lora_adapter_dir": lora_adapter_dir,
    }
    if manifest_path is not None:
        traj_row["manifest"] = str(manifest_path)

    return traj_row, metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_vllm_rollouts(
    cfg: VLLMEvalConfig,
    logger: Optional[logging.Logger] = None,
) -> VLLMRolloutResult:
    """Notebook-friendly API for vLLM evaluation.

    Supported rollout modes:
    - interactive (default): one action per generation step
    - trajectory: one-shot generation of the full action trajectory
    """
    logger = logger or LOGGER

    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This function must be run inside the vLLM evaluation environment. "
            "Install requirements/eval-vllm.txt and then pip install -e ."
        ) from exc

    model_path, lora_adapter_dir, manifest_path = _resolve_model_and_lora(cfg)
    logger.info("Resolved model path: %s", model_path)
    if lora_adapter_dir:
        logger.info("Using LoRA adapter: %s", lora_adapter_dir)

    lora_request = None
    if lora_adapter_dir is not None:
        lora_request = maybe_make_lora_request(
            getattr(cfg, "lora_name", "triage_adapter"),
            lora_adapter_dir,
            getattr(cfg, "lora_id", 1),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.trust_remote_code)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=cfg.tensor_parallel_size,
        trust_remote_code=cfg.trust_remote_code,
        max_model_len=cfg.max_model_len,
        enable_lora=bool(lora_request),
        max_loras=(getattr(cfg, "max_loras", 4) if lora_request else 0),
        max_lora_rank=getattr(cfg, "max_lora_rank", 64),
        enforce_eager=getattr(cfg, "enforce_eager", False),
        gpu_memory_utilization=getattr(cfg, "gpu_memory_utilization", 0.9),
    )
    sampling_params = _make_sampling_params(cfg)

    cases = read_dataset(cfg.dataset)
    max_cases = _get_cfg(cfg, "max_cases", None)
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    rollout_mode = str(_get_cfg(cfg, "rollout_mode", "interactive")).strip().lower() or "interactive"
    if rollout_mode not in {"interactive", "trajectory"}:
        raise ValueError(f"Unsupported rollout_mode={rollout_mode!r}")

    verifier = TriageTrajectoryVerifier()
    trajectory_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []

    show_progress = bool(_get_cfg(cfg, "show_progress", True))
    case_iter = _iter_cases(
        cases,
        progress=show_progress,
        desc=f"vLLM {rollout_mode} rollout",
    )

    for case in case_iter:
        if rollout_mode == "interactive":
            traj_row, metrics = _run_single_case_interactive(
                case=case,
                tokenizer=tokenizer,
                llm=llm,
                sampling_params=sampling_params,
                lora_request=lora_request,
                cfg=cfg,
                verifier=verifier,
                model_path=model_path,
                manifest_path=manifest_path,
                lora_adapter_dir=lora_adapter_dir,
                logger=logger,
            )
        else:
            traj_row, metrics = _run_single_case_trajectory(
                case=case,
                tokenizer=tokenizer,
                llm=llm,
                sampling_params=sampling_params,
                lora_request=lora_request,
                cfg=cfg,
                verifier=verifier,
                model_path=model_path,
                manifest_path=manifest_path,
                lora_adapter_dir=lora_adapter_dir,
                logger=logger,
            )

        trajectory_rows.append(traj_row)
        metrics_rows.append(metrics)

    _ensure_parent(cfg.out_traj)
    _ensure_parent(cfg.out_metrics)
    if cfg.out_traj:
        save_jsonl(cfg.out_traj, trajectory_rows)
    if cfg.out_metrics:
        save_jsonl(cfg.out_metrics, metrics_rows)

    summary = aggregate_episode_metrics(metrics_rows)
    summary["failure_reasons"] = summarize_failure_reasons(metrics_rows)
    summary["rollout_mode"] = rollout_mode
    summary["enable_thinking"] = bool(_get_cfg(cfg, "enable_thinking", False))
    summary["num_cases"] = len(metrics_rows)
    summary["model_path"] = model_path
    if lora_adapter_dir is not None:
        summary["lora_adapter_dir"] = lora_adapter_dir
    if manifest_path is not None:
        summary["manifest"] = str(manifest_path)

    return VLLMRolloutResult(
        cfg=cfg,
        summary=summary,
        trajectories=trajectory_rows,
        metrics=metrics_rows,
    )


# Backward-compatible aliases.
VLLMEvalRunResult = VLLMRolloutResult
resolve_model_and_lora = _resolve_model_and_lora

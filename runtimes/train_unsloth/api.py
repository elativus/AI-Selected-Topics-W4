from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from triage.artifacts import ModelArtifactManifest, write_manifest
from triage.io_utils import read_dataset
from triage.logging_utils import coalesce_logger
from triage.prompting import SYSTEM_PROMPT_VERSION, build_trajectory_messages
from triage.schema import TriageData
from triage.trajectory_text import extract_actions
from triage.verifier import TriageTrajectoryVerifier
from runtimes.train_unsloth.bootstrap_unsloth import configure_unsloth_env
from runtimes.train_unsloth.config import UnslothGRPOConfig


@dataclass
class GRPOPreparedSession:
    cfg: UnslothGRPOConfig
    cases: List[TriageData]
    train_rows: List[Dict[str, Any]]
    train_dataset: Any
    model: Any
    tokenizer: Any
    trainer: Any
    output_dir: Path


@dataclass
class GRPOTrainingResult:
    cfg: UnslothGRPOConfig
    output_dir: Path
    adapter_dir: Path
    manifest_path: Path
    num_cases: int
    trainer: Any
    model: Any
    tokenizer: Any


def build_training_records(
    cases: List[TriageData],
    *,
    progress: bool = True,
    logger=None,
    allow_reasoning: bool = False,
) -> List[Dict[str, Any]]:
    from triage.env import SafeTriageEnv

    logger = coalesce_logger(logger, "train_unsloth.prepare")
    env = SafeTriageEnv()
    rows: List[Dict[str, Any]] = []
    iterable = cases
    if progress:
        try:
            from tqdm.auto import tqdm
            iterable = tqdm(cases, desc="building train records", leave=False)
        except ImportError:
            pass
    for case in iterable:
        initial_observation = env.reset(case)
        rows.append(
            {
                "prompt": build_trajectory_messages(
                    initial_observation,
                    case.max_steps,
                    allow_reasoning=allow_reasoning,
                ),
                "case_json": json.dumps(case.to_dict(), ensure_ascii=False),
                "case_id": case.case_id,
                "difficulty": case.difficulty,
            }
        )
    logger.info("Built %s training records", len(rows))
    return rows


def make_reward_function(logger=None):
    verifier = TriageTrajectoryVerifier()
    logger = coalesce_logger(logger, "train_unsloth.reward")

    def reward_func(completions, case_json=None, **_: Any):
        rewards: List[float] = []
        for completion, case_blob in zip(completions, case_json or []):
            from triage.env import SafeTriageEnv

            env = SafeTriageEnv()
            case = TriageData.from_dict(json.loads(case_blob))

            if isinstance(completion, list):
                text = completion[-1]["content"] if completion and isinstance(completion[-1], dict) else str(completion)
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = str(completion)

            actions = extract_actions(text, max_actions=case.max_steps)
            metrics = verifier.verify_trajectory(env, case, actions)
            rewards.append(float(metrics["total_reward"]))
        return rewards

    return reward_func


def _make_live_logging_callback(logger):
    try:
        from transformers import TrainerCallback
    except Exception:  # pragma: no cover
        return None

    class LiveLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # pragma: no cover - depends on trainer runtime
            if not logs:
                return
            compact = {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in logs.items()
                if k not in {"total_flos"}
            }
            logger.info("train_log step=%s logs=%s", state.global_step, compact)

    return LiveLoggingCallback()


def prepare_grpo_session(
    cfg: UnslothGRPOConfig,
    *,
    logger=None,
    progress: bool = True,
    max_cases: Optional[int] = None,
) -> GRPOPreparedSession:
    logger = coalesce_logger(logger, "train_unsloth")
    configure_unsloth_env(cfg)

    try:
        from datasets import Dataset
        from unsloth import FastLanguageModel, PatchFastRL
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This function must be run inside the Unsloth training environment. "
            "Install requirements/train-unsloth.txt and then pip install -e ."
        ) from exc

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(output_dir / "train_config.json")

    logger.info("Patching Unsloth for GRPO")
    PatchFastRL("GRPO", FastLanguageModel)

    logger.info("Loading dataset from %s", cfg.dataset)
    cases = read_dataset(cfg.dataset)
    if max_cases is not None:
        cases = cases[: int(max_cases)]
        logger.info("Using max_cases=%s for notebook/debug run", max_cases)
    logger.info("Loaded %s training cases", len(cases))

    train_rows = build_training_records(
        cases,
        progress=progress,
        logger=logger,
        allow_reasoning=bool(getattr(cfg, "enable_thinking", False)),
    )
    train_ds = Dataset.from_list(train_rows)

    logger.info("Loading base model %s", cfg.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        trust_remote_code=cfg.trust_remote_code,
    )
    logger.info("Wrapping model with PEFT LoRA r=%s", cfg.lora_r)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing=("unsloth" if cfg.use_unsloth_gradient_checkpointing else False),
        random_state=cfg.seed,
    )

    reward_func = make_reward_function(logger=logger)
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        save_steps=cfg.save_steps,
        warmup_steps=cfg.warmup_steps,
        seed=cfg.seed,
        logging_steps=cfg.logging_steps,
        report_to=[] if cfg.report_to == "none" else [cfg.report_to],
        disable_tqdm=False,
    )

    logger.info("Constructing GRPOTrainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_ds,
    )
    callback = _make_live_logging_callback(logger)
    if callback is not None:
        try:
            trainer.add_callback(callback)
        except Exception:
            logger.warning("Could not attach live logging callback; continuing without it.")

    return GRPOPreparedSession(
        cfg=cfg,
        cases=cases,
        train_rows=train_rows,
        train_dataset=train_ds,
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        output_dir=output_dir,
    )


def finalize_training_artifacts(
    session: GRPOPreparedSession,
    *,
    logger=None,
) -> GRPOTrainingResult:
    logger = coalesce_logger(logger, "train_unsloth")
    adapter_dir = session.output_dir / "adapter"
    session.trainer.model.save_pretrained(str(adapter_dir))
    session.tokenizer.save_pretrained(str(adapter_dir))

    manifest = ModelArtifactManifest(
        run_name=session.output_dir.name,
        base_model=session.cfg.model,
        adapter_dir=str(adapter_dir),
        tokenizer_dir=str(adapter_dir),
        system_prompt_version=SYSTEM_PROMPT_VERSION,
        notes=(
            "Adapter checkpoint produced by the Unsloth GRPO runtime. "
            "Run export_merged_for_vllm.py inside the same env before evaluation. "
            "This runtime now supports direct notebook-native execution and shared CLI wrappers."
        ),
        extra={
            "dataset": str(session.cfg.dataset),
            "num_cases": len(session.cases),
            "train_config": str(session.output_dir / "train_config.json"),
            "max_seq_length": session.cfg.max_seq_length,
            "max_prompt_length": session.cfg.max_prompt_length,
            "max_completion_length": session.cfg.max_completion_length,
            "load_in_4bit": session.cfg.load_in_4bit,
            "lora_r": session.cfg.lora_r,
            "seed": session.cfg.seed,
            "unsloth_bootstrap": {
                "unsloth_disable_compile": session.cfg.unsloth_disable_compile,
                "unsloth_fullgraph": session.cfg.unsloth_fullgraph,
                "unsloth_compile_ignore_errors": session.cfg.unsloth_compile_ignore_errors,
                "clear_unsloth_cache": session.cfg.clear_unsloth_cache,
                "disable_torchdynamo": session.cfg.disable_torchdynamo,
                "disable_torch_compile": session.cfg.disable_torch_compile,
            },
        },
    )
    manifest_path = session.output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    logger.info("Saved adapter checkpoint to %s", adapter_dir)
    logger.info("Wrote manifest to %s", manifest_path)
    return GRPOTrainingResult(
        cfg=session.cfg,
        output_dir=session.output_dir,
        adapter_dir=adapter_dir,
        manifest_path=manifest_path,
        num_cases=len(session.cases),
        trainer=session.trainer,
        model=session.model,
        tokenizer=session.tokenizer,
    )


def run_grpo_training(
    cfg: UnslothGRPOConfig,
    *,
    logger=None,
    progress: bool = True,
    max_cases: Optional[int] = None,
) -> GRPOTrainingResult:
    logger = coalesce_logger(logger, "train_unsloth")
    session = prepare_grpo_session(cfg, logger=logger, progress=progress, max_cases=max_cases)
    logger.info("Starting GRPO training")
    session.trainer.train()
    logger.info("Training finished; saving artifacts")
    return finalize_training_artifacts(session, logger=logger)

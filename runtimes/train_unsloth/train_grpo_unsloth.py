from __future__ import annotations

import argparse
from pathlib import Path

from triage.logging_utils import get_logger
from runtimes.train_unsloth.api import run_grpo_training
from runtimes.train_unsloth.config import UnslothGRPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRPO in the Unsloth runtime for SafeTriageEnv.")
    parser.add_argument("dataset", type=Path, nargs="?", default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--enable_thinking", action="store_true", help="Allow model to use <think> reasoning blocks.")
    parser.add_argument("--max_cases", type=int, default=None, help="Notebook/debug helper: restrict number of training cases.")
    parser.add_argument("--no_progress", action="store_true", help="Disable notebook/terminal tqdm progress bars.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> UnslothGRPOConfig:
    cfg = UnslothGRPOConfig.from_json(args.config) if args.config else UnslothGRPOConfig()

    if args.dataset is not None:
        cfg.dataset = str(args.dataset)
    if args.model is not None:
        cfg.model = args.model
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.max_seq_length is not None:
        cfg.max_seq_length = int(args.max_seq_length)
    if args.max_prompt_length is not None:
        cfg.max_prompt_length = int(args.max_prompt_length)
    if args.max_completion_length is not None:
        cfg.max_completion_length = int(args.max_completion_length)
    if args.load_in_4bit:
        cfg.load_in_4bit = True
    if args.lora_r is not None:
        cfg.lora_r = int(args.lora_r)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if args.per_device_train_batch_size is not None:
        cfg.per_device_train_batch_size = int(args.per_device_train_batch_size)
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    if args.num_train_epochs is not None:
        cfg.num_train_epochs = float(args.num_train_epochs)
    if args.num_generations is not None:
        cfg.num_generations = int(args.num_generations)
    if args.save_steps is not None:
        cfg.save_steps = int(args.save_steps)
    if args.logging_steps is not None:
        cfg.logging_steps = int(args.logging_steps)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.enable_thinking:
        cfg.enable_thinking = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    result = run_grpo_training(
        cfg,
        logger=get_logger("train_unsloth"),
        progress=not args.no_progress,
        max_cases=args.max_cases,
    )
    print(f"Saved adapter checkpoint to: {result.adapter_dir}")
    print(f"Wrote manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()

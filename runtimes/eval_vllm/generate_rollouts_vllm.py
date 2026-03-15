from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage.logging_utils import get_logger
from runtimes.eval_vllm.api import run_vllm_rollouts
from runtimes.eval_vllm.config import VLLMEvalConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trajectories with vLLM and evaluate them on SafeTriageEnv.")
    parser.add_argument("dataset", type=Path, nargs="?", default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest produced by the training runtime.")
    parser.add_argument("--model", type=str, default=None, help="Raw merged model path. Ignored if --manifest is given.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out_traj", type=Path, default=None)
    parser.add_argument("--out_metrics", type=Path, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--rollout_mode", choices=("interactive", "trajectory"), default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--prefer_lora_adapter", action="store_true")
    parser.add_argument("--lora_adapter_dir", type=str, default=None)
    parser.add_argument("--quiet_vllm", action="store_true", help="Suppress vLLM stdout/stderr and disable vLLM tqdm.")
    parser.add_argument("--no_progress", action="store_true", help="Disable notebook/terminal tqdm progress bars.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> VLLMEvalConfig:
    cfg = VLLMEvalConfig.from_json(args.config) if args.config else VLLMEvalConfig()
    if args.dataset is not None:
        cfg.dataset = str(args.dataset)
    if args.manifest is not None:
        cfg.manifest = str(args.manifest)
    if args.model is not None:
        cfg.model = str(args.model)
    if args.out_traj is not None:
        cfg.out_traj = str(args.out_traj)
    if args.out_metrics is not None:
        cfg.out_metrics = str(args.out_metrics)
    if args.max_model_len is not None:
        cfg.max_model_len = int(args.max_model_len)
    if args.max_tokens is not None:
        cfg.max_tokens = int(args.max_tokens)
    if args.temperature is not None:
        cfg.temperature = float(args.temperature)
    if args.top_p is not None:
        cfg.top_p = float(args.top_p)
    if args.top_k is not None:
        cfg.top_k = int(args.top_k)
    if args.min_p is not None:
        cfg.min_p = float(args.min_p)
    if args.tensor_parallel_size is not None:
        cfg.tensor_parallel_size = int(args.tensor_parallel_size)
    if args.max_cases is not None:
        cfg.max_cases = int(args.max_cases)
    if args.rollout_mode is not None:
        cfg.rollout_mode = str(args.rollout_mode)
    if args.enable_thinking:
        cfg.enable_thinking = True
    if args.trust_remote_code:
        cfg.trust_remote_code = True
    if args.prefer_lora_adapter:
        cfg.prefer_lora_adapter = True
    if args.lora_adapter_dir is not None:
        cfg.lora_adapter_dir = str(args.lora_adapter_dir)
    if args.quiet_vllm:
        cfg.suppress_vllm_output = True
        cfg.use_vllm_tqdm = False
    if args.no_progress:
        cfg.show_progress = False
    return cfg


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    result = run_vllm_rollouts(cfg, logger=get_logger("eval_vllm"))
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    if cfg.out_traj:
        print(f"Saved trajectories to: {cfg.out_traj}")
    if cfg.out_metrics:
        print(f"Saved metrics to: {cfg.out_metrics}")


if __name__ == "__main__":
    main()

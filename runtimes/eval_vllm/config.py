from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VLLMEvalConfig:
    dataset: str = "data/eval/eval_d4.jsonl"
    manifest: Optional[str] = None
    model: Optional[str] = None
    out_traj: str = "artifacts/runs/vllm_eval/trajectories.jsonl"
    out_metrics: str = "artifacts/runs/vllm_eval/metrics.jsonl"
    rollout_mode: str = "interactive"
    enable_thinking: bool = False
    force_plaintext_prompt: bool = False  # debug only; not used in production pipeline

    max_model_len: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True

    suppress_vllm_output: bool = False
    use_vllm_tqdm: bool = True
    show_progress: bool = True
    max_cases: Optional[int] = None

    prefer_merged_model: bool = True
    prefer_lora_adapter: bool = False
    lora_adapter_dir: Optional[str] = None
    lora_name: str = "triage_adapter"
    lora_id: int = 1
    max_loras: int = 4
    max_lora_rank: int = 64
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "VLLMEvalConfig":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**obj)

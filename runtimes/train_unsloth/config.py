from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from runtimes.train_unsloth.bootstrap_unsloth import UnslothBootstrapConfig


@dataclass
class UnslothGRPOConfig(UnslothBootstrapConfig):
    dataset: str = "data/train.jsonl"
    model: str = "Qwen/Qwen3-8B"
    output_dir: str = "artifacts/runs/qwen3_8b_unsloth_grpo"

    max_seq_length: int = 4096
    max_prompt_length: int = 3072
    max_completion_length: int = 1024

    enable_thinking: bool = False
    load_in_4bit: bool = True
    lora_r: int = 32
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0
    num_generations: int = 4
    save_steps: int = 50
    logging_steps: int = 1
    warmup_steps: int = 0
    seed: int = 42

    trust_remote_code: bool = True
    use_unsloth_gradient_checkpointing: bool = True
    report_to: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "UnslothGRPOConfig":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**obj)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

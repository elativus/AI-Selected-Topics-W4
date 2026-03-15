from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from triage.prompting import SYSTEM_PROMPT_VERSION


@dataclass
class ModelArtifactManifest:
    run_name: str
    base_model: str
    train_backend: str = "unsloth"
    eval_backend: str = "vllm"
    export_format: str = "merged_16bit"

    adapter_dir: Optional[str] = None
    merged_model_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    model_for_eval: Optional[str] = None

    system_prompt_version: str = SYSTEM_PROMPT_VERSION
    chat_template_source: str = "shared_render_chat_prompt"
    train_config_path: Optional[str] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ModelArtifactManifest":
        return cls(**obj)

    def resolved_eval_model_path(self) -> str:
        if self.model_for_eval:
            return self.model_for_eval
        if self.merged_model_dir:
            return self.merged_model_dir
        if self.adapter_dir:
            return self.adapter_dir
        raise ValueError("Manifest does not contain an eval-capable model path.")


def write_manifest(path: str | Path, manifest: ModelArtifactManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def read_manifest(path: str | Path) -> ModelArtifactManifest:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelArtifactManifest.from_dict(obj)

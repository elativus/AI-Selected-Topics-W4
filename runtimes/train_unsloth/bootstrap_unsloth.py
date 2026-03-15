from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class UnslothBootstrapConfig:
    unsloth_disable_compile: bool = True
    unsloth_fullgraph: bool = False
    unsloth_compile_ignore_errors: bool = True
    clear_unsloth_cache: bool = True
    disable_torchdynamo: bool = False
    disable_torch_compile: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def configure_unsloth_env(cfg: UnslothBootstrapConfig) -> None:
    """Set env flags BEFORE importing unsloth.

    This is lifted from the useful stabilization pattern in the Week 3 code:
    clean compiled cache, toggle compile/fullgraph knobs, and allow emergency
    switches for torchdynamo / torch.compile.
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

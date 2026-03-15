from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class Data:
    """Minimal base data container compatible with the homework interface.

    The week-1 codebase may already define a richer Data class. This fallback keeps
    the current project self-contained and easy to run standalone.
    """

    uid: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

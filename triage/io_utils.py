from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from triage.schema import TriageData


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_dataset(path: str | Path, cases: Iterable[TriageData]) -> None:
    save_jsonl(path, [case.to_dict() for case in cases])


def read_dataset(path: str | Path) -> List[TriageData]:
    return [TriageData.from_dict(obj) for obj in load_jsonl(path)]

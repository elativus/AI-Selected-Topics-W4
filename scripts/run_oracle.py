from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage.logging_utils import get_logger
from triage.workflows import run_oracle_on_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out_metrics", type=Path, default=Path("oracle_metrics.jsonl"))
    parser.add_argument("--out_traj", type=Path, default=Path("oracle_trajectories.jsonl"))
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    result = run_oracle_on_dataset(
        args.dataset,
        out_metrics=args.out_metrics,
        out_traj=args.out_traj,
        progress=not args.no_progress,
        logger=get_logger("run_oracle"),
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

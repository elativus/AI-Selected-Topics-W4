from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage.logging_utils import get_logger
from triage.workflows import evaluate_rollouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--out_metrics", type=Path, default=Path("metrics.jsonl"))
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    result = evaluate_rollouts(
        args.dataset,
        args.trajectories,
        out_metrics=args.out_metrics,
        progress=not args.no_progress,
        logger=get_logger("eval_rollouts"),
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

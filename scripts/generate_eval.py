from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage.logging_utils import get_logger
from triage.workflows import generate_eval_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--size_per_difficulty", type=int, default=100)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    result = generate_eval_datasets(
        out_dir=args.out_dir,
        size_per_difficulty=args.size_per_difficulty,
        seed=args.seed,
        progress=not args.no_progress,
        logger=get_logger("generate_eval"),
    )
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triage.logging_utils import get_logger
from triage.workflows import generate_train_val_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    result = generate_train_val_datasets(
        out_dir=args.out_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        progress=not args.no_progress,
        logger=get_logger("generate_train"),
    )
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

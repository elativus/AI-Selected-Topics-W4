from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from triage.workflows import (
    evaluate_rollouts,
    generate_eval_datasets,
    generate_train_val_datasets,
    run_oracle_on_dataset,
)


class WorkflowTests(unittest.TestCase):
    def test_generate_train_and_eval_and_oracle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            build = generate_train_val_datasets(out_dir=root / "data", train_size=12, val_size=6, progress=False)
            self.assertTrue((root / "data" / "train.jsonl").exists())
            self.assertEqual(build.counts["train"], 12)
            self.assertEqual(build.counts["val"], 6)

            eval_build = generate_eval_datasets(out_dir=root / "eval", size_per_difficulty=2, difficulties=[1, 2], progress=False)
            self.assertTrue((root / "eval" / "eval_d1.jsonl").exists())
            self.assertEqual(eval_build.counts["eval_d1"], 2)
            self.assertEqual(eval_build.counts["eval_d2"], 2)

            oracle = run_oracle_on_dataset(
                root / "eval" / "eval_d1.jsonl",
                out_metrics=root / "oracle.metrics.jsonl",
                out_traj=root / "oracle.traj.jsonl",
                progress=False,
            )
            self.assertEqual(len(oracle.metrics), 2)
            self.assertTrue((root / "oracle.metrics.jsonl").exists())
            replay = evaluate_rollouts(
                root / "eval" / "eval_d1.jsonl",
                root / "oracle.traj.jsonl",
                out_metrics=root / "replay.metrics.jsonl",
                progress=False,
            )
            self.assertEqual(len(replay.metrics), 2)
            self.assertGreaterEqual(replay.summary["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

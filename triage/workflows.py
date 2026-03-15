from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from triage.env import SafeTriageEnv
from triage.generator import CaseGenerator
from triage.io_utils import load_jsonl, read_dataset, save_jsonl, write_dataset
from triage.logging_utils import coalesce_logger
from triage.metrics import aggregate_episode_metrics, summarize_failure_reasons
from triage.oracle import OracleSolver
from triage.schema import TriageData
from triage.verifier import TriageTrajectoryVerifier


@dataclass
class DatasetBuildResult:
    paths: Dict[str, Path]
    counts: Dict[str, int]
    metadata: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        return {
            "paths": {k: str(v) for k, v in self.paths.items()},
            "counts": self.counts,
            "metadata": self.metadata,
        }

    def summary_df(self):
        import pandas as pd

        rows = []
        for name, path in self.paths.items():
            rows.append({"split": name, "path": str(path), "count": self.counts.get(name)})
        return pd.DataFrame(rows)


@dataclass
class EvaluationRunResult:
    metrics: List[Dict[str, Any]]
    trajectories: List[Dict[str, Any]]
    summary: Dict[str, Any]
    out_metrics: Optional[Path] = None
    out_traj: Optional[Path] = None

    def metrics_df(self):
        import pandas as pd

        return pd.DataFrame(self.metrics)

    def trajectories_df(self):
        import pandas as pd

        return pd.DataFrame(self.trajectories)


@dataclass
class OracleRunResult(EvaluationRunResult):
    dataset_path: Optional[Path] = None


@dataclass
class ReplayEvalResult(EvaluationRunResult):
    dataset_path: Optional[Path] = None
    trajectory_path: Optional[Path] = None


def _iter_with_progress(items: Sequence[Any], *, desc: str, enabled: bool):
    if enabled:
        try:
            from tqdm.auto import tqdm
            return tqdm(items, desc=desc, leave=False)
        except ImportError:
            pass
    return items


def _distribute_counts(total: int, buckets: Sequence[Any]) -> List[int]:
    if not buckets:
        return []
    base = total // len(buckets)
    remainder = total % len(buckets)
    return [base + (1 if idx < remainder else 0) for idx, _ in enumerate(buckets)]


def generate_train_val_datasets(
    *,
    out_dir: str | Path,
    train_size: int = 5000,
    val_size: int = 500,
    seed: int = 13,
    difficulties: Iterable[int] = range(1, 11),
    progress: bool = True,
    logger=None,
) -> DatasetBuildResult:
    logger = coalesce_logger(logger, "triage.generate_train")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    difficulties = list(difficulties)
    train_cases: List[TriageData] = []
    gen = CaseGenerator(seed=seed)
    train_counts = _distribute_counts(train_size, difficulties)
    logger.info("Generating train split: %s cases across difficulties %s", train_size, difficulties)
    for d, count in zip(_iter_with_progress(difficulties, desc="train difficulties", enabled=progress), train_counts):
        if count <= 0:
            continue
        cases = gen.generate(num_of_questions=count, difficulty=d)
        train_cases.extend(cases)
        logger.info("difficulty=%s train_cases=%s cumulative=%s", d, len(cases), len(train_cases))
    train_cases = train_cases[:train_size]
    train_path = out_dir / "train.jsonl"
    write_dataset(train_path, train_cases)

    val_cases: List[TriageData] = []
    gen_val = CaseGenerator(seed=seed + 1)
    val_counts = _distribute_counts(val_size, difficulties)
    logger.info("Generating val split: %s cases across difficulties %s", val_size, difficulties)
    for d, count in zip(_iter_with_progress(difficulties, desc="val difficulties", enabled=progress), val_counts):
        if count <= 0:
            continue
        cases = gen_val.generate(num_of_questions=count, difficulty=d)
        val_cases.extend(cases)
        logger.info("difficulty=%s val_cases=%s cumulative=%s", d, len(cases), len(val_cases))
    val_cases = val_cases[:val_size]
    val_path = out_dir / "val.jsonl"
    write_dataset(val_path, val_cases)

    result = DatasetBuildResult(
        paths={"train": train_path, "val": val_path},
        counts={"train": len(train_cases), "val": len(val_cases)},
        metadata={"seed": seed, "difficulties": difficulties},
    )
    logger.info("Wrote train/val datasets to %s", out_dir)
    return result


def generate_eval_datasets(
    *,
    out_dir: str | Path,
    size_per_difficulty: int = 100,
    seed: int = 101,
    difficulties: Iterable[int] = range(1, 11),
    progress: bool = True,
    logger=None,
) -> DatasetBuildResult:
    logger = coalesce_logger(logger, "triage.generate_eval")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    difficulties = list(difficulties)
    paths: Dict[str, Path] = {}
    counts: Dict[str, int] = {}
    logger.info("Generating eval buckets: size_per_difficulty=%s difficulties=%s", size_per_difficulty, difficulties)
    for difficulty in _iter_with_progress(difficulties, desc="eval difficulties", enabled=progress):
        gen = CaseGenerator(seed=seed + difficulty)
        cases = gen.generate(num_of_questions=size_per_difficulty, difficulty=difficulty)
        path = out_dir / f"eval_d{difficulty}.jsonl"
        write_dataset(path, cases)
        paths[f"eval_d{difficulty}"] = path
        counts[f"eval_d{difficulty}"] = len(cases)
        logger.info("Wrote %s (%s cases)", path.name, len(cases))

    return DatasetBuildResult(paths=paths, counts=counts, metadata={"seed": seed, "difficulties": difficulties})


def run_oracle_on_dataset(
    dataset: str | Path,
    *,
    out_metrics: str | Path | None = None,
    out_traj: str | Path | None = None,
    progress: bool = True,
    logger=None,
) -> OracleRunResult:
    logger = coalesce_logger(logger, "triage.oracle")
    dataset_path = Path(dataset)
    cases = read_dataset(dataset_path)
    env = SafeTriageEnv()
    oracle = OracleSolver()
    verifier = TriageTrajectoryVerifier()

    metrics_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []

    for case in _iter_with_progress(cases, desc=f"oracle:{dataset_path.name}", enabled=progress):
        actions = oracle.solve(case)
        metrics = verifier.verify_trajectory(env, case, actions)
        metrics["case_id"] = case.case_id
        metrics["difficulty"] = case.difficulty
        metrics_rows.append(metrics)
        traj_rows.append({"case_id": case.case_id, "actions": actions})

    if out_metrics is not None:
        save_jsonl(out_metrics, metrics_rows)
        logger.info("Saved oracle metrics to %s", out_metrics)
    if out_traj is not None:
        save_jsonl(out_traj, traj_rows)
        logger.info("Saved oracle trajectories to %s", out_traj)

    summary = {
        **aggregate_episode_metrics(metrics_rows),
        "failure_reasons": summarize_failure_reasons(metrics_rows),
        "num_cases": len(metrics_rows),
        "successes": sum(int(r["success"]) for r in metrics_rows),
    }
    logger.info("Oracle summary: %s", json.dumps(summary, ensure_ascii=False))
    return OracleRunResult(
        metrics=metrics_rows,
        trajectories=traj_rows,
        summary=summary,
        out_metrics=Path(out_metrics) if out_metrics is not None else None,
        out_traj=Path(out_traj) if out_traj is not None else None,
        dataset_path=dataset_path,
    )


def evaluate_rollouts(
    dataset: str | Path,
    trajectories: str | Path | Sequence[Dict[str, Any]],
    *,
    out_metrics: str | Path | None = None,
    progress: bool = True,
    logger=None,
) -> ReplayEvalResult:
    logger = coalesce_logger(logger, "triage.eval_rollouts")
    dataset_path = Path(dataset)
    cases = {case.case_id: case for case in read_dataset(dataset_path)}

    trajectory_path: Optional[Path] = None
    if isinstance(trajectories, (str, Path)):
        trajectory_path = Path(trajectories)
        trajectory_rows = load_jsonl(trajectory_path)
    else:
        trajectory_rows = list(trajectories)

    env = SafeTriageEnv()
    verifier = TriageTrajectoryVerifier()
    metrics_rows: List[Dict[str, Any]] = []
    kept_traj_rows: List[Dict[str, Any]] = []

    for row in _iter_with_progress(trajectory_rows, desc=f"verify:{dataset_path.name}", enabled=progress):
        case = cases[row["case_id"]]
        metrics = verifier.verify_trajectory(env, case, row["actions"])
        metrics["case_id"] = case.case_id
        metrics["difficulty"] = case.difficulty
        metrics_rows.append(metrics)
        kept_traj_rows.append(row)

    if out_metrics is not None:
        save_jsonl(out_metrics, metrics_rows)
        logger.info("Saved replay metrics to %s", out_metrics)

    summary = aggregate_episode_metrics(metrics_rows)
    summary["failure_reasons"] = summarize_failure_reasons(metrics_rows)
    logger.info("Replay summary: %s", json.dumps(summary, ensure_ascii=False))
    return ReplayEvalResult(
        metrics=metrics_rows,
        trajectories=kept_traj_rows,
        summary=summary,
        out_metrics=Path(out_metrics) if out_metrics is not None else None,
        dataset_path=dataset_path,
        trajectory_path=trajectory_path,
    )

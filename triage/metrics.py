from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


def aggregate_episode_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    n = len(results)
    avg = lambda key: sum(float(r.get(key, 0.0)) for r in results) / n
    rate = lambda key: sum(int(bool(r.get(key, 0))) for r in results) / n

    return {
        "num_episodes": n,
        "success_rate": avg("success"),
        "avg_total_reward": avg("total_reward"),
        "avg_steps": avg("steps"),
        "avg_tool_calls": avg("tool_calls"),
        "avg_policy_violations": avg("policy_violations"),
        "avg_critical_policy_violations": avg("critical_policy_violations"),
        "invalid_action_rate": rate("invalid_actions"),
        "undertriage_rate": avg("undertriage"),
        "over_escalation_rate": avg("over_escalation"),
        "hallucination_rate": rate("hallucination_violations"),
        "confirmation_violation_rate": rate("confirmation_violations"),
        "duplicate_question_rate": rate("duplicate_questions"),
        "irrelevant_question_rate": rate("irrelevant_questions"),
        "avg_evidence_coverage": avg("evidence_coverage"),
    }


def group_metrics_by_key(results: List[Dict[str, Any]], key: str) -> Dict[Any, Dict[str, Any]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row.get(key)].append(row)
    return {group: aggregate_episode_metrics(rows) for group, rows in grouped.items()}


def compute_confusion_matrix(results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        matrix[str(row.get("target_disposition"))][str(row.get("final_disposition"))] += 1
    return {target: dict(counter) for target, counter in matrix.items()}


def summarize_failure_reasons(results: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in results:
        reason = row.get("failure_reason")
        if reason:
            counter[str(reason)] += 1
    return dict(counter)

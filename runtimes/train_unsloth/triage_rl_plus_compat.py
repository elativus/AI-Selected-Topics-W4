"""Drop-in replacement for ``w3_common`` that makes the W3 RL-PLUS trainer
work with the Safe Medical Triage environment.

The RL-PLUS trainer imports five symbols from ``w3_common``:
    SYSTEM_PROMPT, build_chat_prompt, extract_int, get_user_prompt, has_answer_tag

This module provides triage-specific implementations with the same signatures,
plus helpers for gradual (non-binary) reward and gold trajectory construction.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from triage.prompting import (
    SYSTEM_PROMPT_TRAJECTORY,
    render_chat_prompt,
)
from triage.trajectory_text import extract_actions, render_actions_block

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = SYSTEM_PROMPT_TRAJECTORY


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(user_prompt)},
    ]
    return render_chat_prompt(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# User prompt extraction
# ---------------------------------------------------------------------------

def get_user_prompt(row: Dict[str, Any]) -> str:
    for key in ("user_prompt", "question", "prompt", "input"):
        if key in row and row[key]:
            return str(row[key])
    raise KeyError(f"Row must contain user_prompt/question/prompt/input. keys={list(row.keys())}")


# ---------------------------------------------------------------------------
# Answer extraction (compat shims)
# ---------------------------------------------------------------------------

_ACTIONS_RE = re.compile(r"<ACTIONS>(.*?)</ACTIONS>", re.DOTALL | re.IGNORECASE)


def extract_int(text: Optional[str]) -> Optional[int]:
    """Returns 1 if valid ACTIONS block found, None otherwise."""
    if not text:
        return None
    actions = extract_actions(text)
    if actions and any(a.startswith("TOOL_CALL") for a in actions):
        return 1
    return None


def has_answer_tag(text: str) -> bool:
    return bool(text) and bool(_ACTIONS_RE.search(text))


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def load_jsonl(path):
    from pathlib import Path as _P
    rows = []
    with _P(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    from pathlib import Path as _P
    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Triage-specific helpers
# ---------------------------------------------------------------------------

def build_triage_train_rows(
    cases,
    *,
    oracle=None,
    env=None,
    verifier=None,
    progress: bool = True,
) -> list:
    """Build training rows with pre-computed gold trajectories from Oracle.

    Each row contains:
    - ``user_prompt``: state snapshot for the trajectory prompt
    - ``gold_completion``: Oracle actions as <ACTIONS>...</ACTIONS>
    - ``gold_reward``: Oracle's total_reward from verifier (typically ~0.73)
    - ``case_json``: serialized TriageData for verification
    - ``case_id``, ``difficulty``: metadata
    - ``answer``: 1 (compat)
    """
    from triage.env import SafeTriageEnv
    from triage.oracle import OracleSolver
    from triage.verifier import TriageTrajectoryVerifier
    from triage.prompting import _TOOL_SCHEMA_COMPACT

    env = env or SafeTriageEnv()
    oracle = oracle or OracleSolver()
    verifier = verifier or TriageTrajectoryVerifier()

    rows = []
    iterable = cases
    if progress:
        try:
            from tqdm.auto import tqdm
            iterable = tqdm(cases, desc="building train rows", leave=False)
        except ImportError:
            pass

    for case in iterable:
        initial_observation = env.reset(case)

        user_prompt = (
            f"STATE_SNAPSHOT:\n"
            f"{initial_observation}\n\n"
            f"MAX_ACTIONS: {case.max_steps}\n\n"
            f"TOOLS:\n{_TOOL_SCHEMA_COMPACT}\n\n"
            f"OUTPUT_CONTRACT:\n"
            f"Return only <ACTIONS>...</ACTIONS>. One action per line. No extra text."
            f"\n/no_think"
        )

        oracle_actions = oracle.solve(case)
        gold_completion = render_actions_block(oracle_actions)

        # Verify oracle and compute gold_reward using same formula as reward_fn
        metrics = verifier.verify_trajectory(SafeTriageEnv(), case, oracle_actions)
        assert metrics["success"], f"Oracle failed on {case.case_id}: {metrics}"

        # Gold reward (same formula as make_triage_reward_fn):
        gold_r = 0.0
        if metrics["success"]:
            gold_r += 1.0
        gold_r += 0.5 * float(metrics.get("evidence_coverage", 0))
        if metrics["tool_calls"] > 0:
            gold_r += 0.2
        gold_r += 0.1  # Oracle always produces <ACTIONS> block

        rows.append({
            "user_prompt": user_prompt,
            "gold_completion": gold_completion,
            "gold_reward": gold_r,
            "case_json": json.dumps(case.to_dict(), ensure_ascii=False),
            "case_id": case.case_id,
            "difficulty": case.difficulty,
            "answer": 1,
        })

    return rows


def make_triage_verify_fn():
    """Verify function: returns True only for full success (for correct_rate logging)."""
    from triage.env import SafeTriageEnv
    from triage.schema import TriageData
    from triage.verifier import TriageTrajectoryVerifier

    verifier = TriageTrajectoryVerifier()

    def verify_fn(row: Dict[str, Any], completion_text: str) -> bool:
        case = TriageData.from_dict(json.loads(row["case_json"]))
        actions = extract_actions(completion_text, max_actions=case.max_steps)
        metrics = verifier.verify_trajectory(SafeTriageEnv(), case, actions)
        return bool(metrics["success"])

    return verify_fn


def make_triage_reward_fn():
    """Exploration-friendly reward for RL training.

    Design: NEVER penalize tool usage or steps.  More evidence = higher reward.
    This prevents the collapse where "do nothing" > "try and fail".

    Reward components (all non-negative):
      - success:           +1.0
      - evidence_coverage: +0.5 * fraction_covered  (0 to 0.5)
      - tool_usage:        +0.2 if any tool_calls > 0
      - format_ok:         +0.1 if has <ACTIONS> block

    Range: [0.0, 1.8]  (gold typically ~1.7-1.8)
    """
    from triage.env import SafeTriageEnv
    from triage.schema import TriageData
    from triage.verifier import TriageTrajectoryVerifier

    verifier = TriageTrajectoryVerifier()

    def reward_fn(row: Dict[str, Any], completion_text: str) -> float:
        case = TriageData.from_dict(json.loads(row["case_json"]))
        actions = extract_actions(completion_text, max_actions=case.max_steps)
        metrics = verifier.verify_trajectory(SafeTriageEnv(), case, actions)

        r = 0.0
        # Outcome: +1.0 for success
        if metrics["success"]:
            r += 1.0
        # Evidence coverage: +0.5 proportional (always positive, encourages exploration)
        r += 0.5 * float(metrics.get("evidence_coverage", 0))
        # Tool usage: +0.2 if model used any tools (encourages trying)
        if metrics["tool_calls"] > 0:
            r += 0.2
        # Format: +0.1 if has ACTIONS block
        if has_answer_tag(completion_text):
            r += 0.1

        return r

    return reward_fn


def make_triage_gold_fn():
    """Gold completion function."""
    def gold_fn(row: Dict[str, Any]) -> str:
        return row["gold_completion"]
    return gold_fn


__all__ = [
    "SYSTEM_PROMPT",
    "build_chat_prompt",
    "extract_int",
    "get_user_prompt",
    "has_answer_tag",
    "load_jsonl",
    "save_jsonl",
    "build_triage_train_rows",
    "make_triage_verify_fn",
    "make_triage_reward_fn",
    "make_triage_gold_fn",
]

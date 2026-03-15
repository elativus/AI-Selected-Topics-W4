from __future__ import annotations

import json
import re
from typing import List, Sequence

ACTIONS_BLOCK_RE = re.compile(r"<ACTIONS>(.*?)</ACTIONS>", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^```(?:[a-zA-Z0-9_+-]*)?\s*|\s*```$", re.MULTILINE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ORPHAN_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
ENUM_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")

# Known tool names for bare-call normalization (defense in depth).
_KNOWN_TOOL_NAMES = frozenset([
    "ask_question", "lookup_protocol", "list_slots",
    "book_visit", "create_escalation", "finish",
])
# Matches: tool_name {json...} or tool_name({json...})
_BARE_TOOL_RE = re.compile(
    r"^(" + "|".join(re.escape(n) for n in _KNOWN_TOOL_NAMES) + r")\s*(\{.+)",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", text).strip()


def strip_think_blocks(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", text or "")
    lower = cleaned.lower()
    think_start = lower.find("<think>")
    if think_start != -1:
        # Drop dangling tail when the model opened <think> but never closed it.
        cleaned = cleaned[:think_start]
    cleaned = ORPHAN_THINK_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _try_normalize_bare_tool_call(line: str) -> str | None:
    """Convert ``tool_name {args_json}`` → ``TOOL_CALL {"name":...,"args":...}``.

    Returns *None* if the line is not a recognizable bare tool call.
    This is a defense-in-depth measure: even if the prompt correctly asks for
    ``TOOL_CALL`` format, some models may emit bare tool names.
    """
    m = _BARE_TOOL_RE.match(line.strip())
    if not m:
        return None
    tool_name = m.group(1)
    raw_args = m.group(2).strip()
    try:
        args_obj = json.loads(raw_args)
    except json.JSONDecodeError:
        return None
    if not isinstance(args_obj, dict):
        return None
    return f'TOOL_CALL {json.dumps({"name": tool_name, "args": args_obj}, ensure_ascii=False)}'


def normalize_action_line(line: str) -> str:
    line = line.strip()
    stripped = ENUM_PREFIX_RE.sub("", line)
    if stripped.startswith("TOOL_CALL") or stripped.startswith("<CONFIRM>"):
        return stripped
    # Try to recover bare tool-name calls (e.g. ``ask_question {"question_id":"Q1"}``)
    normalized = _try_normalize_bare_tool_call(stripped)
    if normalized is not None:
        return normalized
    return line


def coalesce_multiline_actions(lines: Sequence[str]) -> List[str]:
    out: List[str] = []
    i = 0
    norm_lines = [normalize_action_line(line) for line in lines if line and line.strip()]
    while i < len(norm_lines):
        line = norm_lines[i]
        if i + 1 < len(norm_lines) and norm_lines[i + 1].startswith("<CONFIRM>"):
            out.append(f"{line}\n{norm_lines[i + 1]}")
            i += 2
            continue
        out.append(line)
        i += 1
    return out


def is_action_like(line: str) -> bool:
    candidate = normalize_action_line(line)
    return candidate.startswith("TOOL_CALL") or "<CONFIRM>" in candidate


def _candidate_lines(text: str) -> tuple[List[str], bool]:
    raw = strip_think_blocks(strip_code_fences(text or ""))
    match = ACTIONS_BLOCK_RE.search(raw)
    block = match.group(1) if match else raw
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    return coalesce_multiline_actions(lines), bool(match)


def extract_actions(text: str, max_actions: int | None = None) -> List[str]:
    """Extract one-action-per-line trajectories from a model completion.

    Preferred format is ``<ACTIONS>...</ACTIONS>``. The parser is intentionally
    forgiving in two common failure modes:
    - the model emits ``<think>...</think>`` reasoning wrappers;
    - a confirmation free-text action is split across two lines, with the
      sentence on one line and the ``<CONFIRM>...</CONFIRM>`` tag on the next.

    If no ACTIONS block is present but action-like lines exist, only those lines
    are returned. Otherwise the function falls back to all non-empty lines so
    debugging remains easy.
    """
    lines, has_actions_block = _candidate_lines(text)

    if not has_actions_block:
        action_lines = [line for line in lines if is_action_like(line)]
        if action_lines:
            lines = action_lines

    if max_actions is not None:
        lines = lines[:max_actions]
    return lines


def extract_single_action(text: str) -> str:
    """Extract one visible action line from a model completion."""
    lines, _ = _candidate_lines(text)
    if not lines:
        return ""

    for line in lines:
        candidate = normalize_action_line(line)
        if candidate.startswith("TOOL_CALL"):
            return candidate

    for line in lines:
        candidate = normalize_action_line(line)
        if "<CONFIRM>" in candidate:
            return candidate

    # Fallback to first visible non-empty line after stripping thinking output.
    return normalize_action_line(lines[0])


def render_actions_block(actions: Sequence[str]) -> str:
    body = "\n".join(action.rstrip() for action in actions)
    return f"<ACTIONS>\n{body}\n</ACTIONS>"

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

TOOL_PREFIX = "TOOL_CALL "
CONFIRM_RE = re.compile(r"<CONFIRM>(.*?)</CONFIRM>", re.DOTALL)


@dataclass
class ParsedAction:
    kind: Literal["free_text", "tool_call", "invalid"]
    raw: str
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    confirm_payload: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def extract_confirm_payload(text: str) -> Optional[dict[str, Any]]:
    match = CONFIRM_RE.search(text)
    if not match:
        return None
    return json.loads(match.group(1).strip())


def parse_action(action: str) -> ParsedAction:
    action = action.strip()

    if action.startswith(TOOL_PREFIX):
        payload = action[len(TOOL_PREFIX) :].strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            return ParsedAction(kind="invalid", raw=action, error=f"bad_json:{exc}")

        if not isinstance(obj, dict):
            return ParsedAction(kind="invalid", raw=action, error="tool_payload_not_object")
        if "name" not in obj or "args" not in obj:
            return ParsedAction(kind="invalid", raw=action, error="missing_name_or_args")
        if not isinstance(obj["args"], dict):
            return ParsedAction(kind="invalid", raw=action, error="args_not_object")

        return ParsedAction(
            kind="tool_call",
            raw=action,
            tool_name=str(obj["name"]),
            tool_args=obj["args"],
        )

    try:
        confirm_payload = extract_confirm_payload(action)
    except json.JSONDecodeError as exc:
        return ParsedAction(kind="invalid", raw=action, error=f"bad_confirm_json:{exc}")

    return ParsedAction(
        kind="free_text",
        raw=action,
        text=action,
        confirm_payload=confirm_payload,
    )


def render_tool_call(name: str, args: dict[str, Any]) -> str:
    return f'{TOOL_PREFIX}{json.dumps({"name": name, "args": args}, ensure_ascii=False)}'

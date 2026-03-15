from __future__ import annotations

"""Shared prompting helpers for both Unsloth training and vLLM evaluation.

Key design decisions (v5):
- System prompt is a short execution contract (no domain knowledge).
- All case-specific data, tool schemas, and entity lists live in the user
  state-snapshot block, keeping the system prompt constant across steps.
- For Qwen3 models, ``/no_think`` is appended to the user message when
  ``allow_reasoning=False`` to reliably suppress ``<think>`` generation.
- ``enable_thinking`` is passed through to ``apply_chat_template`` for
  models that support it (Qwen3), with a graceful fallback.
"""

import re
from typing import Any, Dict, List, Sequence

SYSTEM_PROMPT_VERSION = "triage_prompt_v5.1_toolcall_format_fix"

# ---------------------------------------------------------------------------
# System prompts – kept deliberately short.
# Tool schema / examples / domain info all go into the user state block.
# ---------------------------------------------------------------------------

_EXECUTION_CONTRACT = """\
You are an executor in a text environment.
Return exactly one next action per turn.

Valid outputs (pick one):
1. A single TOOL_CALL {"name":"<TOOL>","args":{...}} line
2. A single free-text line (for confirmations or messages)

Hard rules:
- Never output multiple actions.
- book_visit / create_escalation require a prior free-text step containing <CONFIRM>...</CONFIRM>.
- Use only entity IDs explicitly present in the state.
- Do not use markdown, code fences, or XML except <CONFIRM>."""

SYSTEM_PROMPT_INTERACTIVE = (
    _EXECUTION_CONTRACT + "\n"
    "- Do not output explanations, reasoning, or <think> tags."
)

SYSTEM_PROMPT_INTERACTIVE_THINKING = (
    _EXECUTION_CONTRACT + "\n"
    "- You may reason internally, but the visible output must be exactly one action line."
)

SYSTEM_PROMPT_TRAJECTORY = (
    "You are an executor in a text environment.\n"
    "Return a single <ACTIONS>...</ACTIONS> block with one action per line.\n\n"
    "Valid action lines:\n"
    "1. TOOL_CALL {...} \u2013 a tool invocation\n"
    "2. Free-text \u2013 for confirmations (<CONFIRM>...</CONFIRM>) or messages\n\n"
    "Hard rules:\n"
    "- book_visit / create_escalation require a prior <CONFIRM>.\n"
    "- Use only entity IDs present in the state.\n"
    "- No markdown, code fences, or XML except <CONFIRM>.\n"
    "- Do not output explanations, reasoning, or <think> tags."
)

SYSTEM_PROMPT_TRAJECTORY_THINKING = (
    "You are an executor in a text environment.\n"
    "Return a single <ACTIONS>...</ACTIONS> block with one action per line.\n\n"
    "Valid action lines:\n"
    "1. TOOL_CALL {...} \u2013 a tool invocation\n"
    "2. Free-text \u2013 for confirmations (<CONFIRM>...</CONFIRM>) or messages\n\n"
    "Hard rules:\n"
    "- book_visit / create_escalation require a prior <CONFIRM>.\n"
    "- Use only entity IDs present in the state.\n"
    "- No markdown, code fences, or XML except <CONFIRM>.\n"
    "- You may reason internally, but visible output must only be the ACTIONS block."
)

# ---------------------------------------------------------------------------
# Tool schema – rendered into the user state block, not the system prompt.
# ---------------------------------------------------------------------------

_TOOL_SCHEMA_COMPACT = """\
TOOL_CALL {"name":"ask_question","args":{"question_id":"<ID>"}}
TOOL_CALL {"name":"lookup_protocol","args":{"entity_ids":["<ID>", ...]}}
TOOL_CALL {"name":"list_slots","args":{"urgency":"ROUTINE|SAME_DAY"}}
TOOL_CALL {"name":"book_visit","args":{"slot_id":"<ID>"}}           (requires prior <CONFIRM>)
TOOL_CALL {"name":"create_escalation","args":{"level":"<LEVEL>"}}   (requires prior <CONFIRM>)
TOOL_CALL {"name":"finish","args":{"disposition":"SELF_CARE|BOOK_ROUTINE|BOOK_SAME_DAY|ESCALATE_NOW","advice_pack_id":"<ID>|null"}}"""

# ---------------------------------------------------------------------------
# Observation parsing helpers
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^([A-Z_]+):\n", re.MULTILINE)


def _parse_observation_sections(observation: str) -> Dict[str, str]:
    text = observation or ""
    matches = list(_SECTION_RE.finditer(text))
    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    return sections


def _extract_case_metadata(observation: str, history: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    fields = ("CASE", "DIFFICULTY", "MAX_STEPS")
    sources = [observation] + [str(turn.get("observation", "")) for turn in history]
    history_key_map = {
        "CASE": "case_id",
        "DIFFICULTY": "difficulty",
        "MAX_STEPS": "max_steps",
    }
    for fld in fields:
        value = "<unknown>"
        pattern = re.compile(rf"^{fld}:\s*(.+)$", re.MULTILINE)
        for src in sources:
            match = pattern.search(src or "")
            if match:
                value = match.group(1).strip()
                break
        if value == "<unknown>":
            key = history_key_map[fld]
            for turn in history:
                maybe_value = turn.get(key)
                if maybe_value is not None:
                    value = str(maybe_value)
                    break
        values[fld] = value
    return values


def _render_previous_actions(history: Sequence[Dict[str, Any]]) -> str:
    if not history:
        return "<none>"
    lines: List[str] = []
    for idx, turn in enumerate(history, start=1):
        action = str(turn.get("action", "")).strip() or "<empty>"
        lines.append(f"{idx}. {action}")
    return "\n".join(lines)


def _extract_last_tool_result(observation: str) -> str:
    text = observation or ""
    marker = "\n\nNEW_ENTITIES:\n"
    if marker in text:
        return text.split(marker, 1)[0].strip() or "<empty>"
    return text.strip() or "<empty>"


# ---------------------------------------------------------------------------
# /no_think suffix for Qwen3 thinking suppression
# ---------------------------------------------------------------------------

_NO_THINK_SUFFIX = "\n/no_think"


# ---------------------------------------------------------------------------
# Public API – message builders
# ---------------------------------------------------------------------------

def build_trajectory_messages(
    initial_observation: str,
    max_steps: int,
    allow_reasoning: bool = False,
) -> List[Dict[str, str]]:
    system_prompt = (
        SYSTEM_PROMPT_TRAJECTORY_THINKING if allow_reasoning else SYSTEM_PROMPT_TRAJECTORY
    )
    user_content = (
        f"STATE_SNAPSHOT:\n"
        f"{initial_observation}\n\n"
        f"MAX_ACTIONS: {max_steps}\n\n"
        f"TOOLS:\n{_TOOL_SCHEMA_COMPACT}\n\n"
        f"OUTPUT_CONTRACT:\n"
        f"Return only <ACTIONS>...</ACTIONS>. One action per line. No extra text."
    )
    if not allow_reasoning:
        user_content += _NO_THINK_SUFFIX
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# Backward-friendly convenience alias used in some notebooks.
def build_messages(
    initial_observation: str,
    max_steps: int,
    allow_reasoning: bool = False,
) -> List[Dict[str, str]]:
    return build_trajectory_messages(
        initial_observation=initial_observation,
        max_steps=max_steps,
        allow_reasoning=allow_reasoning,
    )


def build_interactive_messages(
    history: List[Dict[str, Any]],
    current_observation: str,
    max_steps_remaining: int,
    allow_reasoning: bool = False,
) -> List[Dict[str, str]]:
    sections = _parse_observation_sections(current_observation)
    metadata = _extract_case_metadata(current_observation, history)
    known_entities = sections.get("KNOWN_ENTITIES", "<unknown>")
    available_tools = sections.get("AVAILABLE_TOOLS", "<unknown>")
    covered_groups = sections.get("COVERED_EVIDENCE_GROUPS", "<unknown>")
    remaining_groups = sections.get("REMAINING_EVIDENCE_GROUPS", "<unknown>")
    policy = sections.get("POLICY", "<unknown>")
    violations_this_step = sections.get("VIOLATIONS_THIS_STEP", "<unknown>")
    last_tool_result = _extract_last_tool_result(current_observation)
    previous_actions = _render_previous_actions(history)

    system_prompt = (
        SYSTEM_PROMPT_INTERACTIVE_THINKING if allow_reasoning else SYSTEM_PROMPT_INTERACTIVE
    )

    user_content = (
        f"STATE_SNAPSHOT\n"
        f"CASE_ID: {metadata['CASE']}\n"
        f"DIFFICULTY: {metadata['DIFFICULTY']}\n"
        f"STEPS_REMAINING: {max_steps_remaining}\n\n"
        f"CURRENT_OBSERVATION:\n"
        f"{last_tool_result}\n\n"
        f"KNOWN_ENTITIES:\n"
        f"{known_entities}\n\n"
        f"COVERED_EVIDENCE_GROUPS:\n"
        f"{covered_groups}\n\n"
        f"REMAINING_EVIDENCE_GROUPS:\n"
        f"{remaining_groups}\n\n"
        f"POLICY:\n"
        f"{policy}\n\n"
        f"VIOLATIONS_THIS_STEP:\n"
        f"{violations_this_step}\n\n"
        f"PREVIOUS_ACTIONS:\n"
        f"{previous_actions}\n\n"
        f"TOOLS:\n{_TOOL_SCHEMA_COMPACT}\n\n"
        f"VALID_TOOLS:\n"
        f"{available_tools}\n\n"
        f"OUTPUT_CONTRACT:\n"
        f"Return exactly one next action.\n"
        f"- one TOOL_CALL {{...}} line\n"
        f"- or one free-text line\n"
        f"If solved, emit TOOL_CALL {{\"name\":\"finish\",\"args\":...}}."
    )
    if not allow_reasoning:
        user_content += _NO_THINK_SUFFIX

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = False,
    return_meta: bool = False,
) -> str | tuple[str, Dict[str, Any]]:
    """Render messages into a prompt string using the tokenizer's chat template.

    Falls back to a simple plaintext format when the tokenizer has no template
    (useful for unit tests / debugging).
    """
    meta: Dict[str, Any] = {
        "renderer": "plaintext",
        "enable_thinking_requested": enable_thinking,
        "enable_thinking_applied": False,
        "fallback_reason": None,
    }
    if (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        meta["renderer"] = "chat_template"
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            try:
                prompt = tokenizer.apply_chat_template(
                    list(messages),
                    enable_thinking=enable_thinking,
                    **kwargs,
                )
                meta["enable_thinking_applied"] = True
                if return_meta:
                    return prompt, meta
                return prompt
            except TypeError:
                meta["fallback_reason"] = "enable_thinking_not_supported"
        prompt = tokenizer.apply_chat_template(list(messages), **kwargs)
        if return_meta:
            return prompt, meta
        return prompt

    # Plaintext fallback (no chat template on tokenizer).
    blocks: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))
        blocks.append(f"[{role}]\n{content}")
    if add_generation_prompt:
        blocks.append("[ASSISTANT]\n")
    prompt = "\n\n".join(blocks)
    if return_meta:
        return prompt, meta
    return prompt

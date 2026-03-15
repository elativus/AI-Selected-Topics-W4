from __future__ import annotations

import unittest

from triage.prompting import (
    SYSTEM_PROMPT_INTERACTIVE,
    SYSTEM_PROMPT_INTERACTIVE_THINKING,
    SYSTEM_PROMPT_TRAJECTORY,
    SYSTEM_PROMPT_TRAJECTORY_THINKING,
    SYSTEM_PROMPT_VERSION,
    build_interactive_messages,
    build_messages,
    build_trajectory_messages,
    render_chat_prompt,
)


class DummyTokenizer:
    chat_template = None


class DummyTemplateTokenizer:
    chat_template = "dummy"

    def __init__(self) -> None:
        self.last_kwargs = None
        self.last_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = list(messages)
        self.last_kwargs = dict(kwargs)
        return "TEMPLATE_PROMPT"


class DummyTemplateTokenizerNoThinking:
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "TEMPLATE_PROMPT_NO_THINKING_ARG"


class PromptingTests(unittest.TestCase):
    def test_version_tag(self) -> None:
        self.assertEqual(SYSTEM_PROMPT_VERSION, "triage_prompt_v5.1_toolcall_format_fix")

    def test_build_trajectory_messages_structure(self) -> None:
        messages = build_trajectory_messages("OBS", 7)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        # max_steps now in user block, not system
        self.assertIn("7", messages[1]["content"])
        self.assertIn("OBS", messages[1]["content"])
        # System is short execution contract
        self.assertIn("executor", messages[0]["content"])
        self.assertIn("<ACTIONS>", messages[0]["content"])

    def test_trajectory_no_reasoning_has_no_think(self) -> None:
        messages = build_trajectory_messages("OBS", 5, allow_reasoning=False)
        self.assertIn("/no_think", messages[1]["content"])

    def test_trajectory_with_reasoning_no_no_think(self) -> None:
        messages = build_trajectory_messages("OBS", 5, allow_reasoning=True)
        self.assertNotIn("/no_think", messages[1]["content"])

    def test_build_messages_alias(self) -> None:
        self.assertEqual(build_messages("OBS", 3), build_trajectory_messages("OBS", 3))

    def test_build_messages_allow_reasoning_switches_prompt(self) -> None:
        # Non-thinking prompts suppress reasoning
        self.assertIn("Do not output explanations", SYSTEM_PROMPT_TRAJECTORY)
        self.assertIn("Do not output explanations", SYSTEM_PROMPT_INTERACTIVE)
        # Thinking prompts allow reasoning
        self.assertIn("may reason internally", SYSTEM_PROMPT_TRAJECTORY_THINKING)
        self.assertIn("may reason internally", SYSTEM_PROMPT_INTERACTIVE_THINKING)

    def test_build_interactive_messages_state_centric(self) -> None:
        history = [
            {
                "action": 'TOOL_CALL {"name":"ask_question","args":{"question_id":"Q_SHORTNESS_OF_BREATH"}}',
                "observation": "Patient: no shortness of breath.\n\nNEW_ENTITIES:\n[]\n\nKNOWN_ENTITIES:\n['Q_SHORTNESS_OF_BREATH']",
            }
        ]
        current_observation = (
            "CASE: CASE_1\nDIFFICULTY: 2\nMAX_STEPS: 8\n\n"
            "Patient: still coughing.\n\n"
            "NEW_ENTITIES:\n['E1']\n\n"
            "KNOWN_ENTITIES:\n['E1', 'Q_SHORTNESS_OF_BREATH']\n\n"
            "AVAILABLE_TOOLS:\n['ask_question', 'finish']\n"
        )
        messages = build_interactive_messages(history, current_observation, 4)
        self.assertEqual(messages[0]["role"], "system")
        # System prompt is the short contract
        self.assertIn("executor", messages[0]["content"])
        # User block has state snapshot
        self.assertIn("STATE_SNAPSHOT", messages[1]["content"])
        self.assertIn("PREVIOUS_ACTIONS", messages[1]["content"])
        self.assertIn("COVERED_EVIDENCE_GROUPS", messages[1]["content"])
        self.assertIn("REMAINING_EVIDENCE_GROUPS", messages[1]["content"])
        self.assertIn("VALID_TOOLS", messages[1]["content"])
        self.assertIn("STEPS_REMAINING: 4", messages[1]["content"])
        self.assertIn("CASE_ID: CASE_1", messages[1]["content"])
        # Tool schema is now in user block
        self.assertIn("TOOLS:", messages[1]["content"])
        self.assertIn("ask_question", messages[1]["content"])
        self.assertIn("finish", messages[1]["content"])
        # /no_think appended (default allow_reasoning=False)
        self.assertIn("/no_think", messages[1]["content"])

    def test_interactive_with_reasoning_no_no_think(self) -> None:
        messages = build_interactive_messages([], "CASE: X\n", 5, allow_reasoning=True)
        self.assertNotIn("/no_think", messages[1]["content"])

    def test_system_prompt_does_not_contain_tool_schema(self) -> None:
        """Tool schema moved from system to user block in v5."""
        self.assertNotIn("ask_question", SYSTEM_PROMPT_INTERACTIVE)
        self.assertNotIn("lookup_protocol", SYSTEM_PROMPT_INTERACTIVE)
        self.assertNotIn("finish", SYSTEM_PROMPT_INTERACTIVE)
        self.assertNotIn("PACK_RESP_SUPPORT", SYSTEM_PROMPT_INTERACTIVE)
        self.assertNotIn("SAME_DAY_1", SYSTEM_PROMPT_INTERACTIVE)

    def test_render_chat_prompt_fallback(self) -> None:
        messages = build_trajectory_messages("OBS", 5)
        prompt = render_chat_prompt(DummyTokenizer(), messages)
        self.assertIn("[SYSTEM]", prompt)
        self.assertIn("[USER]", prompt)
        self.assertIn("OBS", prompt)
        self.assertIn("[ASSISTANT]", prompt)
        self.assertNotIn("\\n", prompt)

    def test_render_chat_prompt_passes_enable_thinking_flag(self) -> None:
        tok = DummyTemplateTokenizer()
        messages = build_trajectory_messages("OBS", 5)
        prompt = render_chat_prompt(tok, messages, enable_thinking=True)
        self.assertEqual(prompt, "TEMPLATE_PROMPT")
        self.assertEqual(tok.last_kwargs["enable_thinking"], True)
        self.assertEqual(tok.last_kwargs["add_generation_prompt"], True)
        self.assertEqual(tok.last_kwargs["tokenize"], False)

    def test_render_chat_prompt_meta_for_enable_thinking(self) -> None:
        tok = DummyTemplateTokenizer()
        messages = build_trajectory_messages("OBS", 5)
        prompt, meta = render_chat_prompt(tok, messages, enable_thinking=True, return_meta=True)
        self.assertEqual(prompt, "TEMPLATE_PROMPT")
        self.assertEqual(meta["renderer"], "chat_template")
        self.assertEqual(meta["enable_thinking_requested"], True)
        self.assertEqual(meta["enable_thinking_applied"], True)
        self.assertIsNone(meta["fallback_reason"])

    def test_render_chat_prompt_meta_fallback_when_no_enable_thinking_arg(self) -> None:
        tok = DummyTemplateTokenizerNoThinking()
        messages = build_trajectory_messages("OBS", 5)
        prompt, meta = render_chat_prompt(tok, messages, enable_thinking=False, return_meta=True)
        self.assertEqual(prompt, "TEMPLATE_PROMPT_NO_THINKING_ARG")
        self.assertEqual(meta["renderer"], "chat_template")
        self.assertEqual(meta["enable_thinking_requested"], False)
        self.assertEqual(meta["enable_thinking_applied"], False)
        self.assertEqual(meta["fallback_reason"], "enable_thinking_not_supported")

    def test_render_chat_prompt_no_force_plaintext_arg(self) -> None:
        """render_chat_prompt no longer accepts force_plaintext (removed in v5)."""
        import inspect
        sig = inspect.signature(render_chat_prompt)
        self.assertNotIn("force_plaintext", sig.parameters)


if __name__ == "__main__":
    unittest.main()

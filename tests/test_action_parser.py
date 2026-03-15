from __future__ import annotations

import unittest

from triage.action_parser import parse_action, render_tool_call


class ActionParserTests(unittest.TestCase):
    def test_valid_tool_call(self) -> None:
        action = render_tool_call("ask_question", {"question_id": "Q_DURATION"})
        parsed = parse_action(action)
        self.assertEqual(parsed.kind, "tool_call")
        self.assertEqual(parsed.tool_name, "ask_question")
        self.assertEqual(parsed.tool_args, {"question_id": "Q_DURATION"})

    def test_invalid_tool_json(self) -> None:
        parsed = parse_action('TOOL_CALL {"name": "ask_question"')
        self.assertEqual(parsed.kind, "invalid")
        self.assertIn("bad_json", parsed.error or "")

    def test_confirmation_payload(self) -> None:
        parsed = parse_action('Please confirm. <CONFIRM>{"tool":"create_escalation","level":"ESCALATE_NOW"}</CONFIRM>')
        self.assertEqual(parsed.kind, "free_text")
        self.assertEqual(parsed.confirm_payload, {"tool": "create_escalation", "level": "ESCALATE_NOW"})


if __name__ == "__main__":
    unittest.main()

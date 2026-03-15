from __future__ import annotations

import unittest

from triage.trajectory_text import extract_actions, extract_single_action, render_actions_block


class TrajectoryTextTests(unittest.TestCase):
    def test_extract_actions_from_tagged_block(self) -> None:
        text = "before\n<ACTIONS>\nTOOL_CALL {\"name\":\"finish\",\"args\":{\"disposition\":\"SELF_CARE\",\"advice_pack_id\":\"PACK_RESP_SUPPORT\"}}\n</ACTIONS>\nafter"
        actions = extract_actions(text)
        self.assertEqual(len(actions), 1)
        self.assertTrue(actions[0].startswith("TOOL_CALL"))

    def test_extract_actions_fallback(self) -> None:
        text = "line1\n\nline2"
        self.assertEqual(extract_actions(text), ["line1", "line2"])

    def test_extract_actions_strips_think_blocks(self) -> None:
        text = "<think>reasoning</think>\n<ACTIONS>\nTOOL_CALL {\"name\":\"ask_question\",\"args\":{\"question_id\":\"Q_SHORTNESS_OF_BREATH\"}}\n</ACTIONS>"
        self.assertEqual(
            extract_actions(text),
            ['TOOL_CALL {"name":"ask_question","args":{"question_id":"Q_SHORTNESS_OF_BREATH"}}'],
        )

    def test_extract_actions_strips_unclosed_think_tail(self) -> None:
        text = 'TOOL_CALL {"name":"finish","args":{"disposition":"SELF_CARE","advice_pack_id":"PACK_RESP_SUPPORT"}}\n<think>hidden reasoning'
        self.assertEqual(
            extract_actions(text),
            ['TOOL_CALL {"name":"finish","args":{"disposition":"SELF_CARE","advice_pack_id":"PACK_RESP_SUPPORT"}}'],
        )

    def test_extract_actions_coalesces_confirmation_lines(self) -> None:
        text = "<ACTIONS>\nI recommend a same-day visit.\n<CONFIRM>{\"tool\":\"book_visit\",\"slot_id\":\"SAME_DAY_1\"}</CONFIRM>\nTOOL_CALL {\"name\":\"book_visit\",\"args\":{\"slot_id\":\"SAME_DAY_1\"}}\n</ACTIONS>"
        actions = extract_actions(text)
        self.assertEqual(len(actions), 2)
        self.assertIn("<CONFIRM>", actions[0])
        self.assertIn("book_visit", actions[1])

    def test_extract_actions_prefers_action_like_lines_when_no_block(self) -> None:
        text = "reasoning\nmore reasoning\nTOOL_CALL {\"name\":\"finish\",\"args\":{\"disposition\":\"SELF_CARE\",\"advice_pack_id\":\"PACK_RESP_SUPPORT\"}}"
        self.assertEqual(
            extract_actions(text),
            ['TOOL_CALL {"name":"finish","args":{"disposition":"SELF_CARE","advice_pack_id":"PACK_RESP_SUPPORT"}}'],
        )

    def test_extract_single_action_prefers_tool_call(self) -> None:
        text = "Some prose\nTOOL_CALL {\"name\":\"ask_question\",\"args\":{\"question_id\":\"Q_DURATION\"}}\nOther prose"
        self.assertEqual(
            extract_single_action(text),
            'TOOL_CALL {"name":"ask_question","args":{"question_id":"Q_DURATION"}}',
        )

    def test_extract_single_action_prefers_confirm_when_no_tool_call(self) -> None:
        text = "I recommend a same-day visit. <CONFIRM>{\"tool\":\"book_visit\",\"slot_id\":\"SAME_DAY_1\"}</CONFIRM>\nextra"
        self.assertEqual(
            extract_single_action(text),
            'I recommend a same-day visit. <CONFIRM>{"tool":"book_visit","slot_id":"SAME_DAY_1"}</CONFIRM>',
        )

    def test_extract_single_action_fallback_visible_free_text(self) -> None:
        text = "<think>reasoning</think>\nPlease rest and hydrate."
        self.assertEqual(extract_single_action(text), "Please rest and hydrate.")

    def test_extract_single_action_strips_unclosed_think_tail(self) -> None:
        text = 'TOOL_CALL {"name":"ask_question","args":{"question_id":"Q_DURATION"}}\n<think>hidden reasoning'
        self.assertEqual(
            extract_single_action(text),
            'TOOL_CALL {"name":"ask_question","args":{"question_id":"Q_DURATION"}}',
        )

    def test_render_actions_block(self) -> None:
        block = render_actions_block(["a", "b"])
        self.assertIn("<ACTIONS>", block)
        self.assertIn("a", block)
        self.assertIn("</ACTIONS>", block)

    def test_normalize_bare_tool_call_ask_question(self) -> None:
        line = 'ask_question {"question_id":"Q_TEMPERATURE"}'
        result = extract_single_action(line)
        self.assertTrue(result.startswith("TOOL_CALL"))
        self.assertIn('"ask_question"', result)
        self.assertIn('"Q_TEMPERATURE"', result)

    def test_normalize_bare_tool_call_finish(self) -> None:
        line = 'finish {"disposition":"SELF_CARE","advice_pack_id":"PACK_RESP_SUPPORT"}'
        result = extract_single_action(line)
        self.assertTrue(result.startswith("TOOL_CALL"))
        self.assertIn('"finish"', result)
        self.assertIn('"SELF_CARE"', result)

    def test_normalize_bare_tool_call_with_think(self) -> None:
        text = '<think>\n\n</think>\n\nask_question {"question_id":"Q_TEMPERATURE"}'
        result = extract_single_action(text)
        self.assertTrue(result.startswith("TOOL_CALL"))
        self.assertIn('"ask_question"', result)

    def test_normalize_does_not_affect_proper_tool_call(self) -> None:
        line = 'TOOL_CALL {"name":"ask_question","args":{"question_id":"Q1"}}'
        result = extract_single_action(line)
        self.assertEqual(result, line)

    def test_normalize_does_not_affect_free_text(self) -> None:
        line = "I recommend a same-day visit."
        result = extract_single_action(line)
        self.assertEqual(result, line)


if __name__ == "__main__":
    unittest.main()

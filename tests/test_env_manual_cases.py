from __future__ import annotations

import unittest

from triage.action_parser import render_tool_call
from triage.env import SafeTriageEnv
from triage.generator import CaseGenerator
from triage.oracle import OracleSolver
from triage.verifier import TriageTrajectoryVerifier


class EnvManualCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SafeTriageEnv()
        self.verifier = TriageTrajectoryVerifier()
        self.oracle = OracleSolver()
        self.gen = CaseGenerator(seed=7)

    def _one_case(self, family: str, target: str, difficulty: int = 3):
        return self.gen.generate(num_of_questions=1, difficulty=difficulty, family=family, target_disposition=target, seed=111)[0]

    def test_oracle_same_day_success(self) -> None:
        case = self._one_case("RESP", "BOOK_SAME_DAY", difficulty=4)
        actions = self.oracle.solve(case)
        metrics = self.verifier.verify_trajectory(self.env, case, actions)
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["final_disposition"], "BOOK_SAME_DAY")

    def test_booking_without_confirmation_fails(self) -> None:
        case = self._one_case("RESP", "BOOK_SAME_DAY", difficulty=4)
        self.env.reset(case)
        # reveal slots but skip confirmation
        self.env.step(render_tool_call("ask_question", {"question_id": "Q_SHORTNESS_OF_BREATH"}))
        self.env.step(render_tool_call("ask_question", {"question_id": "Q_TEMPERATURE"}))
        self.env.step(render_tool_call("list_slots", {"urgency": "SAME_DAY"}))
        _, reward, done, _ = self.env.step(render_tool_call("book_visit", {"slot_id": "SAME_DAY_1"}))
        self.assertTrue(done)
        self.assertLess(reward, 0.0)
        self.assertEqual(self.env.state.failure_reason, "book_without_confirmation")

    def test_hallucinated_slot_fails(self) -> None:
        case = self._one_case("RESP", "BOOK_SAME_DAY", difficulty=4)
        self.env.reset(case)
        self.env.step(render_tool_call("ask_question", {"question_id": "Q_SHORTNESS_OF_BREATH"}))
        self.env.step(render_tool_call("ask_question", {"question_id": "Q_TEMPERATURE"}))
        _, reward, done, _ = self.env.step('I recommend a same-day visit.\n<CONFIRM>{"tool":"book_visit","slot_id":"FAKE_SLOT"}</CONFIRM>')
        self.assertFalse(done)
        self.assertLess(reward, 0.0)
        self.assertGreater(self.env.state.hallucination_violations, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from triage.env import SafeTriageEnv
from triage.generator import CaseGenerator
from triage.oracle import OracleSolver
from triage.verifier import TriageTrajectoryVerifier


class VerifierTests(unittest.TestCase):
    def test_verifier_returns_expected_fields(self) -> None:
        gen = CaseGenerator(seed=17)
        case = gen.generate(num_of_questions=1, difficulty=3, family="GI", target_disposition="SELF_CARE", seed=222)[0]
        oracle = OracleSolver()
        env = SafeTriageEnv()
        verifier = TriageTrajectoryVerifier()
        metrics = verifier.verify_trajectory(env, case, oracle.solve(case))
        for key in [
            "success",
            "total_reward",
            "steps",
            "tool_calls",
            "policy_violations",
            "info_trace",
        ]:
            self.assertIn(key, metrics)
        self.assertTrue(metrics["success"])
        self.assertGreaterEqual(metrics["evidence_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()

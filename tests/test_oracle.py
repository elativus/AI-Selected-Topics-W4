from __future__ import annotations

import unittest

from triage.env import SafeTriageEnv
from triage.generator import CaseGenerator
from triage.oracle import OracleSolver
from triage.verifier import TriageTrajectoryVerifier


class OracleTests(unittest.TestCase):
    def test_oracle_solves_mixed_cases(self) -> None:
        gen = CaseGenerator(seed=41)
        cases = []
        specs = [
            ("RESP", "SELF_CARE"),
            ("GI", "BOOK_SAME_DAY"),
            ("UTI", "ESCALATE_NOW"),
            ("HEADACHE", "BOOK_ROUTINE"),
        ]
        for family, target in specs:
            cases.extend(gen.generate(num_of_questions=1, difficulty=4, family=family, target_disposition=target, seed=444)[0:1])

        oracle = OracleSolver()
        verifier = TriageTrajectoryVerifier()
        env = SafeTriageEnv()
        successes = 0
        for case in cases:
            actions = oracle.solve(case)
            metrics = verifier.verify_trajectory(env, case, actions)
            successes += int(metrics["success"])
        self.assertEqual(successes, len(cases))


if __name__ == "__main__":
    unittest.main()

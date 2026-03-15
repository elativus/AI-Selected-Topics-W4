from __future__ import annotations

import unittest

from triage.generator import CaseGenerator
from triage.rule_engine import infer_target_disposition


class GeneratorTests(unittest.TestCase):
    def test_generated_case_is_consistent(self) -> None:
        gen = CaseGenerator(seed=33)
        case = gen.generate(num_of_questions=1, difficulty=5, family="UTI", target_disposition="ESCALATE_NOW", seed=333)[0]
        self.assertEqual(infer_target_disposition(case.family, case.hidden_facts), case.target_disposition)
        self.assertTrue(case.required_evidence_groups)
        self.assertTrue(case.qa_map)
        self.assertGreaterEqual(case.max_steps, 1)


if __name__ == "__main__":
    unittest.main()

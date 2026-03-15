from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from runtimes.eval_vllm import (
    VLLMEvalConfig,
    VLLMEvalRunResult,
    VLLMRolloutResult,
    resolve_model_and_lora,
)
from runtimes.eval_vllm.generate_rollouts_vllm import parse_args, resolve_config


class EvalVLLMConfigTests(unittest.TestCase):
    def test_eval_config_defaults_interactive_mode(self) -> None:
        cfg = VLLMEvalConfig()
        self.assertEqual(cfg.rollout_mode, "interactive")
        self.assertFalse(cfg.enable_thinking)
        self.assertFalse(cfg.force_plaintext_prompt)
        self.assertEqual(cfg.top_k, -1)
        self.assertEqual(cfg.min_p, 0.0)

    def test_backward_compatible_aliases_are_exported(self) -> None:
        self.assertIs(VLLMEvalRunResult, VLLMRolloutResult)
        self.assertTrue(callable(resolve_model_and_lora))

    def test_cli_resolve_config_defaults_to_interactive(self) -> None:
        with patch.object(sys, "argv", ["generate_rollouts_vllm"]):
            args = parse_args()
        cfg = resolve_config(args)
        self.assertEqual(cfg.rollout_mode, "interactive")
        self.assertFalse(cfg.enable_thinking)

    def test_cli_resolve_config_applies_new_rollout_flags(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "generate_rollouts_vllm",
                "--rollout_mode",
                "trajectory",
                "--enable_thinking",
                "--top_k",
                "20",
                "--min_p",
                "0.1",
            ],
        ):
            args = parse_args()
        cfg = resolve_config(args)
        self.assertEqual(cfg.rollout_mode, "trajectory")
        self.assertTrue(cfg.enable_thinking)
        self.assertEqual(cfg.top_k, 20)
        self.assertAlmostEqual(cfg.min_p, 0.1)


if __name__ == "__main__":
    unittest.main()

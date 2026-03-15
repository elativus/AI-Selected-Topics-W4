from __future__ import annotations

"""Compatibility wrapper.

The project now separates training and evaluation into different runtime packages:
- runtimes.train_unsloth.* for Unsloth + GRPO
- runtimes.eval_vllm.* for vLLM inference / evaluation

This wrapper remains only to keep older README commands from failing.
"""

from runtimes.train_unsloth.train_grpo_unsloth import main


if __name__ == "__main__":
    main()

# vLLM evaluation runtime

This environment is for inference/evaluation only.

Notable points:
- supports merged-model evaluation
- also supports direct adapter evaluation through vLLM LoRA requests
- uses the same shared prompt renderer as the training runtime

Examples:

```bash
python -m runtimes.eval_vllm.generate_rollouts_vllm --config configs/eval_vllm.example.json
```

Adapter-only evaluation:

```bash
python -m runtimes.eval_vllm.generate_rollouts_vllm   --manifest artifacts/runs/qwen3_14b_unsloth_grpo/manifest.json   --prefer_lora_adapter
```

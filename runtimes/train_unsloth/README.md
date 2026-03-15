# Unsloth training runtime

This environment is for GRPO training only.

Notable points:
- uses `bootstrap_unsloth.py` to configure Unsloth before import
- accepts `--config configs/train_unsloth.example.json`
- writes `train_config.json` next to the run manifest
- exports adapters which can later be merged for vLLM or evaluated as LoRA directly

Example:

```bash
python -m runtimes.train_unsloth.train_grpo_unsloth --config configs/train_unsloth.example.json
python -m runtimes.train_unsloth.export_merged_for_vllm artifacts/runs/qwen3_14b_unsloth_grpo/adapter   --manifest artifacts/runs/qwen3_14b_unsloth_grpo/manifest.json   --export_dir artifacts/runs/qwen3_14b_unsloth_grpo/merged_16bit
```

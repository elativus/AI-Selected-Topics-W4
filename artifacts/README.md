# Artifacts handoff contract

Recommended layout per run:

```text
artifacts/
  runs/
    <run_name>/
      adapter/              # optional LoRA adapter checkpoint from Unsloth
      merged_16bit/         # merged checkpoint exported for vLLM
      manifest.json         # consumed by the vLLM runtime
      train_config.json     # optional
      train_metrics.json    # optional
      eval/
        trajectories.jsonl
        metrics.jsonl
```

The only file the vLLM runtime strictly requires is `manifest.json` with a valid `merged_model_dir`
or `model_for_eval` entry.

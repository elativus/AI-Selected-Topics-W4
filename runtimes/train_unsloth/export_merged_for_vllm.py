from __future__ import annotations

import argparse
from pathlib import Path

from runtimes.train_unsloth.bootstrap_unsloth import UnslothBootstrapConfig, configure_unsloth_env
from triage.artifacts import ModelArtifactManifest, read_manifest, write_manifest
from triage.logging_utils import get_logger
from triage.prompting import SYSTEM_PROMPT_VERSION


def export_merged_model_for_vllm(
    input_model: str | Path,
    *,
    base_model: str = "",
    run_name: str = "unsloth_export",
    export_dir: str | Path = Path("artifacts/runs/unsloth_export/merged_16bit"),
    manifest: str | Path | None = None,
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
    save_method: str = "merged_16bit",
    unsloth_disable_compile: bool = True,
    logger=None,
):
    logger = logger or get_logger("export_merged")
    cfg = UnslothBootstrapConfig(unsloth_disable_compile=unsloth_disable_compile)
    configure_unsloth_env(cfg)

    try:
        from unsloth import FastLanguageModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This function must be run inside the Unsloth training environment. "
            "Install requirements/train-unsloth.txt and then pip install -e ."
        ) from exc

    input_model = Path(input_model)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for export from %s", input_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(input_model),
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    logger.info("Saving merged model to %s", export_dir)
    model.save_pretrained_merged(str(export_dir), tokenizer, save_method=save_method)

    manifest_path = Path(manifest) if manifest is not None else None
    if manifest_path is not None and manifest_path.exists():
        artifact_manifest = read_manifest(manifest_path)
        if base_model:
            artifact_manifest.base_model = base_model
    else:
        artifact_manifest = ModelArtifactManifest(
            run_name=run_name,
            base_model=base_model or str(input_model),
        )

    artifact_manifest.export_format = save_method
    artifact_manifest.merged_model_dir = str(export_dir)
    artifact_manifest.model_for_eval = str(export_dir)
    artifact_manifest.tokenizer_dir = str(export_dir)
    artifact_manifest.system_prompt_version = SYSTEM_PROMPT_VERSION
    if artifact_manifest.adapter_dir is None:
        artifact_manifest.adapter_dir = str(input_model)

    final_manifest_path = manifest_path or (export_dir.parent / "manifest.json")
    write_manifest(final_manifest_path, artifact_manifest)
    logger.info("Updated manifest: %s", final_manifest_path)
    return {
        "export_dir": export_dir,
        "manifest_path": final_manifest_path,
        "manifest": artifact_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an Unsloth checkpoint to a vLLM-friendly merged model.")
    parser.add_argument("input_model", type=Path, help="Path to adapter checkpoint or model directory loadable by Unsloth.")
    parser.add_argument("--base_model", type=str, required=False, default="")
    parser.add_argument("--run_name", type=str, default="unsloth_export")
    parser.add_argument("--export_dir", type=Path, default=Path("artifacts/runs/unsloth_export/merged_16bit"))
    parser.add_argument("--manifest", type=Path, default=None, help="Optional existing manifest to update.")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--save_method", type=str, default="merged_16bit")
    parser.add_argument("--unsloth_disable_compile", action="store_true")
    args = parser.parse_args()

    result = export_merged_model_for_vllm(
        args.input_model,
        base_model=args.base_model,
        run_name=args.run_name,
        export_dir=args.export_dir,
        manifest=args.manifest,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        save_method=args.save_method,
        unsloth_disable_compile=args.unsloth_disable_compile,
        logger=get_logger("export_merged"),
    )
    print(f"Merged model exported to: {result['export_dir']}")
    print(f"Updated manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()

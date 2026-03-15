from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from triage.artifacts import ModelArtifactManifest, read_manifest, write_manifest


class ArtifactTests(unittest.TestCase):
    def test_manifest_roundtrip(self) -> None:
        manifest = ModelArtifactManifest(
            run_name="run1",
            base_model="Qwen/Qwen3-14B",
            merged_model_dir="/tmp/model",
            model_for_eval="/tmp/model",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(path, manifest)
            loaded = read_manifest(path)
            self.assertEqual(loaded.run_name, manifest.run_name)
            self.assertEqual(loaded.resolved_eval_model_path(), "/tmp/model")


if __name__ == "__main__":
    unittest.main()

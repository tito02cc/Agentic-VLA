"""Tests for the single CARVE framework-freeze experiment manifest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_vla.experiments import load_frozen_experiment_manifest


class FrozenManifestTest(unittest.TestCase):
    def test_project_manifest_has_one_ordered_experiment_matrix(self) -> None:
        manifest = load_frozen_experiment_manifest(
            Path(__file__).parents[1]
            / "agentic_vla/experiments/CARVE_PI05_FROZEN_EXPERIMENTS.json"
        )
        self.assertEqual(manifest.system.model_id, "pi05")
        self.assertEqual(manifest.system.planner_mode, "asynchronous_guarded")
        self.assertEqual(
            [item.stage for item in manifest.experiments],
            ["integration_smoke", "paired_recovery", "libero_plus", "libero_pro"],
        )

    def test_rejects_out_of_order_experiment_stages(self) -> None:
        manifest = json.loads(
            (Path(__file__).parents[1] / "agentic_vla/experiments/CARVE_PI05_FROZEN_EXPERIMENTS.json").read_text()
        )
        manifest["experiments"][0], manifest["experiments"][1] = (
            manifest["experiments"][1],
            manifest["experiments"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "order"):
                load_frozen_experiment_manifest(path)


if __name__ == "__main__":
    unittest.main()

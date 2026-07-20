"""Prepare OpenPI-compatible norm stats for the LIBERO dataset.

This script converts the cached LeRobot dataset statistics for
`physical-intelligence/libero` into the JSON format expected by
`openpi.shared.normalize.load()`.

It intentionally reuses the real dataset metadata pulled from Hugging Face
instead of fabricating any statistics locally.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


DEFAULT_REPO_ID = "physical-intelligence/libero"
DEFAULT_OUTPUTS = [
    "/home/admin1/ct/openpi-official/assets/physical-intelligence/libero",
    "/home/admin1/ct/CARVE-VLA/weights/openpi-assets/checkpoints/pi0_base/assets/physical-intelligence/libero",
]


def _load_lerobot_stats(repo_id: str) -> dict:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id)
    return metadata.stats


def _convert_stats(raw_stats: dict) -> dict:
    def _to_list(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    return {
        "norm_stats": {
            "state": {
                "mean": _to_list(raw_stats["state"]["mean"]),
                "std": _to_list(raw_stats["state"]["std"]),
                "q01": None,
                "q99": None,
            },
            "actions": {
                "mean": _to_list(raw_stats["actions"]["mean"]),
                "std": _to_list(raw_stats["actions"]["std"]),
                "q01": None,
                "q99": None,
            },
        }
    }


def _write_norm_stats(output_dir: pathlib.Path, payload: dict) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenPI-compatible LIBERO norm stats.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument(
        "--output-dir",
        action="append",
        dest="output_dirs",
        help="Output directory for norm_stats.json. Can be specified multiple times.",
    )
    args = parser.parse_args()

    output_dirs = [pathlib.Path(path) for path in (args.output_dirs or DEFAULT_OUTPUTS)]
    raw_stats = _load_lerobot_stats(args.repo_id)
    payload = _convert_stats(raw_stats)

    for output_dir in output_dirs:
        output_path = _write_norm_stats(output_dir, payload)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build evaluator-labeled temporal perturbation pairs from LIBERO snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from scripts.run_agentic_vla_libero import (
    _apply_mid_episode_nudge,
    _make_env,
    _require_runtime,
)


SEVERITIES = {
    "no_op": 0.0,
    "mild": 0.01,
    "severe": 0.08,
}


def _agentview(observation: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])


def _stable_seed(snapshot_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(snapshot_id.encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(
            "results/carve_t689_paired_5states_v6/joint/failure_snapshots"
        ),
    )
    parser.add_argument("--task-ids", default="8,9")
    parser.add_argument("--snapshot-step", type=int, default=59)
    parser.add_argument("--states-per-task", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/carve_semantic_precision/temporal_pairs_t89"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_ids = [int(value.strip()) for value in args.task_ids.split(",") if value.strip()]
    if not task_ids or args.states_per_task <= 0 or args.snapshot_step < 0:
        raise ValueError("task ids, states-per-task, and snapshot-step must be valid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    benchmark, get_libero_path, offscreen_env, _, _, _ = _require_runtime()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        metadata_paths = sorted(
            args.snapshot_dir.glob(
                f"task{task_id:02d}_episode*_step{args.snapshot_step:04d}.json"
            )
        )[: args.states_per_task]
        if len(metadata_paths) != args.states_per_task:
            raise FileNotFoundError(
                f"expected {args.states_per_task} task {task_id} snapshots at step "
                f"{args.snapshot_step}, found {len(metadata_paths)}"
            )
        task = suite.get_task(task_id)
        env = _make_env(task, get_libero_path, offscreen_env, seed=7)
        try:
            env.reset()
            for metadata_path in metadata_paths:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                arrays = np.load(metadata_path.with_name(metadata["array_file"]))
                sim_state = np.asarray(arrays["sim_state"], dtype=np.float64)
                snapshot_id = metadata_path.stem
                base_seed = _stable_seed(snapshot_id)
                for severity, nudge_xy in SEVERITIES.items():
                    before_obs = env.regenerate_obs_from_state(sim_state.copy())
                    before_image = _agentview(before_obs)
                    if severity == "no_op":
                        after_obs = env.regenerate_obs_from_state(sim_state.copy())
                        event = {
                            "type": "semantic_probe_no_op",
                            "timestep": int(metadata["timestep"]),
                            "object": None,
                            "dx": 0.0,
                            "dy": 0.0,
                        }
                    else:
                        after_obs, event = _apply_mid_episode_nudge(
                            env,
                            before_obs,
                            task_description=str(metadata["instruction"]),
                            rng=np.random.default_rng(base_seed),
                            object_nudge_xy=float(nudge_xy),
                            timestep=int(metadata["timestep"]),
                        )
                        if event is None:
                            raise RuntimeError(f"unable to perturb {snapshot_id}")
                    after_image = _agentview(after_obs)
                    pair_id = f"{snapshot_id}_{severity}"
                    before_path = image_dir / f"{pair_id}_before.png"
                    after_path = image_dir / f"{pair_id}_after.png"
                    imageio.imwrite(before_path, before_image)
                    imageio.imwrite(after_path, after_image)
                    records.append(
                        {
                            "pair_id": pair_id,
                            "snapshot_id": snapshot_id,
                            "task_id": task_id,
                            "episode_id": int(metadata["episode_id"]),
                            "instruction": str(metadata["instruction"]),
                            "severity": severity,
                            "expected_intervention": (
                                False if severity == "no_op" else True if severity == "severe" else None
                            ),
                            "expected_failure_group": (
                                "NONE" if severity == "no_op" else "PHYSICAL_DISPLACEMENT"
                            ),
                            "before_image": str(before_path),
                            "after_image": str(after_path),
                            "evaluator_event": event,
                            "mean_pixel_delta": float(
                                np.mean(
                                    np.abs(
                                        after_image.astype(np.float32)
                                        - before_image.astype(np.float32)
                                    )
                                )
                            ),
                            "controller_uses_evaluator_event": False,
                        }
                    )
        finally:
            env.close()

    manifest = {
        "schema_version": 1,
        "source": str(args.snapshot_dir),
        "task_ids": task_ids,
        "snapshot_step": int(args.snapshot_step),
        "states_per_task": int(args.states_per_task),
        "severities": SEVERITIES,
        "hard_labels": ["no_op", "severe"],
        "mild_policy": "gray_zone_report_only",
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "pairs": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

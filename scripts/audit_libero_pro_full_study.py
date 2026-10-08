#!/usr/bin/env python3
"""Audit completeness and evidence boundaries of a full LIBERO-Pro study."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from collections import Counter
from typing import Any


METHODS = ("frozen_vla", "fixed_recovery", "agentic")
SUITES = ("libero_10", "libero_10_object", "libero_10_swap", "libero_10_task")
FORBIDDEN_EVENT_KEYS = {"reward", "task_success", "object_pose", "sim_state"}


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _video_frames(path: pathlib.Path) -> int:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def audit(root: pathlib.Path, *, expected_trials: int, check_videos: bool) -> dict[str, Any]:
    failures: list[str] = []
    counters: Counter[str] = Counter()
    identities: set[tuple[str, int, int, int]] = set()
    method_identities: dict[str, set[tuple[str, int, int, int]]] = {
        method: set() for method in METHODS
    }
    checkpoint_ids: set[str] = set()
    profile_ids: set[str] = set()

    for suite in SUITES:
        for method in METHODS:
            for task_id in range(10):
                cell = root / suite / method / f"task_{task_id:02d}"
                summary_path = cell / "summary.json"
                if not summary_path.is_file():
                    failures.append(f"missing summary: {summary_path}")
                    continue
                summary = _load(summary_path)
                counters["cells"] += 1
                if summary.get("suite") != suite or summary.get("method") != method:
                    failures.append(f"cell identity mismatch: {summary_path}")
                episodes = summary.get("episodes", [])
                if summary.get("trials") != expected_trials or len(episodes) != expected_trials:
                    failures.append(f"episode count mismatch: {summary_path}")
                for episode in episodes:
                    trial = int(episode["trial"])
                    seed = int(episode["seed"])
                    identity = (suite, task_id, trial, seed)
                    identities.add(identity)
                    method_identities[method].add(identity)
                    workspace = pathlib.Path(episode["workspace"])
                    video = pathlib.Path(episode["video"])
                    required = (
                        workspace / "run_manifest.json",
                        workspace / "run_summary.json",
                        workspace / "events.jsonl",
                        workspace / "evaluator_result.json",
                        video,
                    )
                    for artifact in required:
                        if not artifact.is_file():
                            failures.append(f"missing artifact: {artifact}")
                    if any(not artifact.is_file() for artifact in required):
                        continue
                    counters["episodes"] += 1
                    manifest = _load(workspace / "run_manifest.json")
                    if manifest.get("metadata", {}).get("experiment_method") != method:
                        failures.append(f"manifest method mismatch: {workspace}")
                    checkpoint_ids.add(str(manifest.get("metadata", {}).get("checkpoint_id")))
                    profile_ids.add(str(manifest.get("deployment_profile_id")))
                    for line_number, line in enumerate(
                        (workspace / "events.jsonl").read_text(encoding="utf-8").splitlines(), 1
                    ):
                        event = json.loads(line)
                        payload_text = json.dumps(event.get("payload", {}), sort_keys=True).lower()
                        for key in FORBIDDEN_EVENT_KEYS:
                            if f'"{key}"' in payload_text:
                                failures.append(
                                    f"privileged key {key} in {workspace}/events.jsonl:{line_number}"
                                )
                    if check_videos:
                        try:
                            frames = _video_frames(video)
                        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                            failures.append(f"video decode failed: {video}: {exc}")
                        else:
                            counters["video_frames"] += frames
                            counters["videos"] += 1
                            if frames < 2:
                                failures.append(f"video too short: {video}")

    reference = method_identities["frozen_vla"]
    for method in METHODS[1:]:
        if method_identities[method] != reference:
            failures.append(
                f"paired identities differ for {method}: "
                f"missing={len(reference - method_identities[method])}, "
                f"extra={len(method_identities[method] - reference)}"
            )
    expected_episodes = len(SUITES) * 10 * expected_trials * len(METHODS)
    if counters["episodes"] != expected_episodes:
        failures.append(
            f"audited episodes={counters['episodes']} expected={expected_episodes}"
        )
    return {
        "status": "passed" if not failures else "failed",
        "study_root": str(root),
        "expected_trials_per_task": expected_trials,
        "expected_episodes": expected_episodes,
        "counts": dict(counters),
        "paired_state_identities": len(identities),
        "checkpoint_ids": sorted(checkpoint_ids),
        "deployment_profile_ids": sorted(profile_ids),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=pathlib.Path, required=True)
    parser.add_argument("--expected-trials", type=int, default=10)
    parser.add_argument("--skip-video-decode", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    result = audit(
        args.study_root,
        expected_trials=args.expected_trials,
        check_videos=not args.skip_video_decode,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

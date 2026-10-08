#!/usr/bin/env python
"""Compare two RoboDojo request traces on inputs, outputs and latency.

Plan item 9.21.1 needs the native (B0) arm and the CarveRuntime-wrapped arms
compared on what actually entered the policy, what came out, and how long it
took. Both arms now write the same ``runtime_trace.jsonl`` schema, so one reader
serves both.

Two rules this tool enforces, because breaking either produces a false claim:

1. A latency difference is only labelled comparable when both arms ran the same
   effective ``inference_steps``. Differencing a 4-step arm against a 2-step arm
   does not measure wrapper overhead, it measures the step reduction.
2. Rows are aligned by ``input_sha256``. When nothing aligns, the tool says so
   and falls back to positional comparison *for localisation only*, reporting
   which modality differs rather than asserting the arms are equivalent.

Usage:
    openpi/.venv/bin/python scripts/compare_native_and_runtime_traces.py \
        --trace B0=artifacts/robodojo/native_trace_b0_20260911/B0/runtime_trace.jsonl \
        --trace C1=artifacts/robodojo/recovery_lifecycle_20260910/C1/runtime_trace.jsonl \
        --output artifacts/robodojo/native_trace_b0_20260911/trace_comparison.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def episode_index(rows: list[dict[str, Any]]) -> list[int]:
    """Per-row episode number.

    ``episode_id`` is the parallel env index and is 0 for every single-env
    evaluation, so it cannot delimit episodes. Prefer ``reset_generation`` when
    the trace carries it; otherwise fall back to where ``timestep`` decreases,
    which is where the env was reset.
    """
    if all("reset_generation" in row for row in rows):
        seen: dict[int, int] = {}
        out = []
        for row in rows:
            generation = int(row["reset_generation"])
            seen.setdefault(generation, len(seen))
            out.append(seen[generation])
        return out
    out = [0]
    for previous, current in itertools.pairwise(rows):
        out.append(out[-1] + (1 if current["timestep"] < previous["timestep"] else 0))
    return out


def latency_stats(values: list[float]) -> dict[str, float] | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    ordered = sorted(clean)
    return {
        "n": len(ordered),
        "min_ms": round(ordered[0], 2),
        "p50_ms": round(statistics.median(ordered), 2),
        "p90_ms": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 2),
        "max_ms": round(ordered[-1], 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
    }


def effective_steps(rows: list[dict[str, Any]]) -> list[int]:
    steps = set()
    for row in rows:
        controls = row.get("applied_controls") or {}
        if controls.get("inference_steps") is not None:
            steps.add(int(controls["inference_steps"]))
    return sorted(steps)


def summarize(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = episode_index(rows)
    per_episode: dict[int, int] = {}
    for index in episodes:
        per_episode[index] = per_episode.get(index, 0) + 1
    complete = sum(
        1
        for row in rows
        if row["metadata"].get("input_sha256") and row["metadata"].get("action_sha256")
    )
    return {
        "label": label,
        "records": len(rows),
        "episodes": len(per_episode),
        "records_per_episode": [per_episode[k] for k in sorted(per_episode)],
        "episode_source": (
            "reset_generation"
            if all("reset_generation" in row for row in rows)
            else "timestep_reset"
        ),
        "adapter_id": sorted({row.get("adapter_id") for row in rows}),
        "path": sorted({row["metadata"].get("path", "runtime") for row in rows}),
        "effective_inference_steps": effective_steps(rows),
        "max_actions": sorted(
            {(row.get("applied_controls") or {}).get("max_actions") for row in rows}
        ),
        "rows_with_input_and_action_digest": complete,
        "digest_coverage": round(complete / len(rows), 4),
        "model_latency": latency_stats([row.get("model_latency_ms") for row in rows]),
        "runtime_latency": latency_stats(
            [row.get("runtime_latency_ms") for row in rows]
        ),
        # The only wrapper cost that needs no cross-process alignment: both
        # numbers come from the same request on the same arm, so the difference
        # is the wrapper and nothing else. Null on the native arm, which has no
        # wrapper to measure.
        "wrapper_overhead_ms": latency_stats(
            [
                float(row["runtime_latency_ms"]) - float(row["model_latency_ms"])
                for row in rows
                if row.get("runtime_latency_ms") is not None
                and row.get("model_latency_ms") is not None
            ]
        ),
        "failed_requests": sum(1 for row in rows if not row.get("success", True)),
    }


def modality_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Which part of the observation differs, for one aligned row pair."""
    ea = a["metadata"]["input_components"]["examples"][0]
    eb = b["metadata"]["input_components"]["examples"][0]
    images_a = ea.get("image_sha256") or []
    images_b = eb.get("image_sha256") or []
    return {
        "timestep": a["timestep"],
        "views_compared": min(len(images_a), len(images_b)),
        "views_equal": [x == y for x, y in zip(images_a, images_b)],
        "state_equal": ea.get("state_sha256") == eb.get("state_sha256"),
        "language_equal": ea.get("language_sha256") == eb.get("language_sha256"),
        "action_equal": a["metadata"].get("action_sha256")
        == b["metadata"].get("action_sha256"),
    }


def compare(
    left_label: str,
    left: list[dict[str, Any]],
    right_label: str,
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_by_input: dict[str, dict[str, Any]] = {
        row["metadata"]["input_sha256"]: row for row in left
    }
    right_by_input = {row["metadata"]["input_sha256"]: row for row in right}
    shared = sorted(set(left_by_input) & set(right_by_input))

    left_steps = effective_steps(left)
    right_steps = effective_steps(right)
    compute_matched = left_steps == right_steps and len(left_steps) == 1

    result: dict[str, Any] = {
        "pair": f"{left_label}_vs_{right_label}",
        "unique_inputs": {left_label: len(left_by_input), right_label: len(right_by_input)},
        "shared_input_sha256": len(shared),
        "input_aligned": bool(shared),
        "compute_matched": compute_matched,
        "effective_inference_steps": {left_label: left_steps, right_label: right_steps},
    }

    if shared:
        same_action = sum(
            1
            for key in shared
            if left_by_input[key]["metadata"].get("action_sha256")
            == right_by_input[key]["metadata"].get("action_sha256")
        )
        result["aligned_rows"] = len(shared)
        result["aligned_rows_with_identical_action"] = same_action
        result["identical_action_fraction"] = round(same_action / len(shared), 4)
        deltas = [
            float(left_by_input[key]["model_latency_ms"])
            - float(right_by_input[key]["model_latency_ms"])
            for key in shared
            if left_by_input[key].get("model_latency_ms") is not None
            and right_by_input[key].get("model_latency_ms") is not None
        ]
        if deltas:
            result["paired_model_latency_delta_ms"] = {
                "n": len(deltas),
                "median": round(statistics.median(deltas), 2),
                "mean": round(statistics.fmean(deltas), 2),
            }
    else:
        # No shared input: report where the divergence lives instead of implying
        # the arms are comparable row by row.
        result["note"] = (
            "no shared input_sha256; rows cannot be paired across these traces. "
            "The positional block below localises the difference only."
        )
        left_ep = episode_index(left)
        right_ep = episode_index(right)
        positional = []
        for episode in sorted(set(left_ep) & set(right_ep)):
            a_rows = [row for row, ep in zip(left, left_ep) if ep == episode]
            b_rows = [row for row, ep in zip(right, right_ep) if ep == episode]
            by_timestep = {row["timestep"]: row for row in b_rows}
            matched = [
                modality_diff(row, by_timestep[row["timestep"]])
                for row in a_rows
                if row["timestep"] in by_timestep
            ]
            if not matched:
                continue
            positional.append(
                {
                    "episode": episode,
                    "timesteps_compared": len(matched),
                    "all_views_differ_at_every_step": all(
                        not any(entry["views_equal"]) for entry in matched
                    ),
                    "state_identical_steps": sum(
                        1 for entry in matched if entry["state_equal"]
                    ),
                    "language_identical_steps": sum(
                        1 for entry in matched if entry["language_equal"]
                    ),
                    "action_identical_steps": sum(
                        1 for entry in matched if entry["action_equal"]
                    ),
                    "first_step": matched[0],
                }
            )
        result["positional_localisation"] = positional

    if not compute_matched:
        result["latency_comparability"] = (
            "NOT comparable as wrapper overhead: the arms ran different effective "
            f"inference_steps ({left_label}={left_steps}, {right_label}={right_steps}). "
            "Any latency difference includes the compute change."
        )
    else:
        result["latency_comparability"] = (
            "comparable: same effective inference_steps on both arms."
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable; first trace is the reference for pairwise comparison",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    traces: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for entry in args.trace:
        if "=" not in entry:
            raise SystemExit(f"--trace expects LABEL=PATH, got: {entry}")
        label, raw_path = entry.split("=", 1)
        path = Path(raw_path)
        traces[label] = load_rows(path)
        sources[label] = str(path)

    labels = list(traces)
    report = {
        "schema_version": "carve.trace-comparison.v1",
        "sources": sources,
        "arms": [summarize(label, traces[label]) for label in labels],
        "comparisons": [
            compare(labels[0], traces[labels[0]], other, traces[other])
            for other in labels[1:]
        ],
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

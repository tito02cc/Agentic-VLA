#!/usr/bin/env python3
"""Summarize the paired OpenVLA BF16/INT8/NF4 CARVE profile gate."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_vla.optimization import ProfileManifest  # noqa: E402


DEFAULT_DIR = ROOT / "results/carve_optimize/openvla_4090_20260718"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_DIR)
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--output-md", type=pathlib.Path, default=None)
    return parser.parse_args()


def _row(label: str, path: pathlib.Path) -> dict[str, Any]:
    manifest = ProfileManifest.load(path)
    benchmark = dict(manifest.benchmark)
    fidelity = dict(manifest.fidelity)
    metrics = dict(fidelity.get("metrics", {}))
    before = benchmark.get("system_gpu_memory_mib_before_load")
    after = benchmark.get("system_gpu_memory_mib_after_load")
    model_footprint_gib = None
    if before is not None and after is not None:
        model_footprint_gib = (float(after) - float(before)) / 1024.0
    return {
        "label": label,
        "manifest": str(path),
        "precision": manifest.profile.deployment_precision,
        "runtime_p50_ms": benchmark["runtime_p50_ms"],
        "runtime_p95_ms": benchmark["runtime_p95_ms"],
        "runtime_p99_ms": benchmark["runtime_p99_ms"],
        "deadline_miss_rate": benchmark["deadline_miss_rate"],
        "peak_vram_gb": benchmark.get("peak_vram_gb"),
        "model_system_footprint_gib": model_footprint_gib,
        "load_seconds": benchmark.get("load_seconds"),
        "profile_prewarm_seconds": benchmark.get("profile_prewarm_seconds"),
        "action_exact_rate": metrics.get("action_exact_rate"),
        "gripper_agreement": metrics.get("gripper_decision_agreement"),
        "action_mae": metrics.get("action_mae"),
        "fidelity_passed": bool(fidelity.get("passed")),
        "fidelity_role": fidelity.get("role", "candidate"),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    paths = {
        "BF16": input_dir / "openvla_eager_bf16_reference.json",
        "INT8": input_dir / "openvla_bnb_int8_candidate.json",
        "NF4": input_dir / "openvla_bnb_nf4_candidate.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing OpenVLA profile manifests: " + ", ".join(missing))
    optional_paths = {
        "Compiled BF16 (prewarmed)": input_dir
        / "openvla_compile_bf16_prewarmed_candidate.json",
    }
    paths.update({label: path for label, path in optional_paths.items() if path.exists()})
    rows = [_row(label, path) for label, path in paths.items()]
    reference = rows[0]
    for row in rows:
        row["latency_p50_change_percent"] = 100.0 * (
            float(row["runtime_p50_ms"]) / float(reference["runtime_p50_ms"]) - 1.0
        )
        if row["peak_vram_gb"] is not None and reference["peak_vram_gb"] is not None:
            row["peak_vram_change_percent"] = 100.0 * (
                float(row["peak_vram_gb"]) / float(reference["peak_vram_gb"]) - 1.0
            )
        else:
            row["peak_vram_change_percent"] = None
    summary = {
        "schema_version": 1,
        "gate": "openvla_4090_paired_autoregressive_actions_v1",
        "rows": rows,
        "accepted_optimization_candidates": [
            row["label"]
            for row in rows[1:]
            if row["fidelity_passed"] and row["latency_p50_change_percent"] < 0.0
        ],
        "accepted_100ms_realtime_candidates": [
            row["label"]
            for row in rows[1:]
            if row["fidelity_passed"] and float(row["deadline_miss_rate"]) == 0.0
        ],
    }
    lines = [
        "# OpenVLA CARVE Profile Gate",
        "",
        "| Profile | P50 (ms) | P95 (ms) | P99 (ms) | Miss@100ms | Peak VRAM (GB) | Exact action | Gripper | MAE | Fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {p50} | {p95} | {p99} | {miss}% | {vram} | {exact} | {gripper} | {mae} | {status} |".format(
                label=row["label"],
                p50=_fmt(row["runtime_p50_ms"]),
                p95=_fmt(row["runtime_p95_ms"]),
                p99=_fmt(row["runtime_p99_ms"]),
                miss=_fmt(100.0 * float(row["deadline_miss_rate"]), 1),
                vram=_fmt(row["peak_vram_gb"]),
                exact=_fmt(row["action_exact_rate"], 3),
                gripper=_fmt(row["gripper_agreement"], 3),
                mae=_fmt(row["action_mae"], 4),
                status="PASS" if row["fidelity_passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "Compiled BF16 is accepted only as a prewarmed latency optimization: it preserves all paired actions and improves steady-state latency, but still misses the 100 ms deadline on every call. Its two replay prompt-length buckets require {:.2f} seconds of profile preparation.".format(
                next(
                    (
                        float(row["profile_prewarm_seconds"])
                        for row in rows
                        if row["label"] == "Compiled BF16 (prewarmed)"
                        and row["profile_prewarm_seconds"] is not None
                    ),
                    0.0,
                )
            ),
        ]
    )
    output_json = args.output_json or input_dir / "openvla_profile_gate_summary.json"
    output_md = args.output_md or input_dir / "OPENVLA_PROFILE_GATE.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

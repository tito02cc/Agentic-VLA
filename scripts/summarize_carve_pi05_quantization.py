#!/usr/bin/env python3
"""Summarize CARVE PI0.5 component-wise INT8 sensitivity evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROFILE_INPUTS = (
    (
        "early4",
        "VLM layers 0--3",
        "pi05_torchao_w8a16_vlm_early4_fidelity45_deadline80_rep1.json",
        "rejected_closed_loop_regression",
    ),
    (
        "early6",
        "VLM layers 0--5",
        "pi05_torchao_w8a16_vlm_early_fidelity45.json",
        "rejected_action_fidelity",
    ),
    (
        "middle",
        "VLM middle blocks",
        "pi05_torchao_int8_vlm_middle_fidelity45_20260825.json",
        "rejected_action_fidelity",
    ),
    (
        "late",
        "VLM late blocks",
        "pi05_torchao_int8_vlm_late_fidelity45_20260825.json",
        "admitted_low_memory_tier",
    ),
    (
        "vision",
        "vision encoder",
        "pi05_torchao_int8_vision_fidelity45_20260825.json",
        "rejected_action_fidelity",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results/carve_optimize")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/carve_optimize/pi05_int8_sensitivity_gate.json"),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for group_id, label, filename, decision in PROFILE_INPUTS:
        path = args.input_dir / filename
        manifest = load(path)
        fidelity = manifest["fidelity"]
        benchmark = manifest.get("benchmark")
        row = {
            "group_id": group_id,
            "label": label,
            "profile_id": manifest["profile"]["profile_id"],
            "fidelity_samples": int(fidelity["samples"]),
            "fidelity_passed": bool(fidelity["passed"]),
            "metrics": fidelity["metrics"],
            "violations": fidelity["violations"],
            "decision": decision,
            "source": str(path),
        }
        if benchmark and "samples" in benchmark:
            row["benchmark"] = {
                "samples": int(benchmark["samples"]),
                "runtime_p50_ms": float(benchmark["runtime_p50_ms"]),
                "runtime_p95_ms": float(benchmark["runtime_p95_ms"]),
                "deadline_miss_rate": float(benchmark["deadline_miss_rate"]),
                "peak_vram_gb": float(benchmark["peak_vram_gb"]),
            }
        rows.append(row)

    payload = {
        "schema_version": "carve-pi05-int8-sensitivity-v1",
        "model": "OpenPI PI0.5 LIBERO PyTorch",
        "hardware": "NVIDIA GeForce RTX 4090 24GB",
        "protocol": {
            "fidelity_observations": 45,
            "aggregation": "worst_case",
            "fixed_flow_noise": True,
            "endpoint_l2_max": 0.10,
            "inference_steps": 2,
            "action_horizon": 10,
            "protected_components": [
                "action_expert",
                "action_and_time_projections",
                "final_normalization",
                "action_output_path",
            ],
        },
        "rows": rows,
        "promoted_profile_id": "pi05-torchao_int8-bf16-2step-h10-vlm_late",
        "closed_loop_candidate_id": None,
        "conclusion": (
            "Only the late-language INT8 scope passes replay and the matched T6/T9 "
            "recovery gate. It is admitted as a low-memory tier, while SMVE remains "
            "the realtime default."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CARVE PI0.5 Component-wise INT8 Sensitivity Gate",
        "",
        "## Protocol",
        "",
        "- PI0.5 LIBERO PyTorch checkpoint on one RTX 4090.",
        "- Two flow steps, ten-action horizon, fixed paired flow noise.",
        "- Worst-case metrics across 45 recorded observations.",
        "- Endpoint-L2 acceptance limit: `0.10` in normalized action space.",
        "- Action expert, action/output projections and final normalization remain BF16.",
        "",
        "## Results",
        "",
        "| INT8 scope | Replay | Endpoint L2 | P50 | P95 | Peak VRAM | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        benchmark = row.get("benchmark")
        p50 = f"{benchmark['runtime_p50_ms']:.2f} ms" if benchmark else "not run"
        p95 = f"{benchmark['runtime_p95_ms']:.2f} ms" if benchmark else "not run"
        vram = f"{benchmark['peak_vram_gb']:.2f} GB" if benchmark else "not run"
        lines.append(
            f"| {row['label']} | "
            f"{'PASS' if row['fidelity_passed'] else 'FAIL'} "
            f"({row['fidelity_samples']}/45) | "
            f"{row['metrics']['endpoint_l2']:.5f} | {p50} | {p95} | {vram} | "
            f"{row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["conclusion"],
            "",
            "The early-four-layer profile passed replay but is still rejected because it "
            "lost one matched T6 recovery relative to compiled BF16. Fidelity is a "
            "screening gate, not a substitute for closed-loop validation.",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "promoted": "late"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

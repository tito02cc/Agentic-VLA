#!/usr/bin/env python3
"""Build the unified CARVE VLA and Planner efficiency ablation report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/carve_efficiency_full_20260826"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def paired_agreement(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    reference_records = reference["records"]
    candidate_records = candidate["records"]
    if len(reference_records) != len(candidate_records):
        raise ValueError("planner gates must contain the same number of paired records")
    exact: list[bool] = []
    intervention: list[bool] = []
    for ref, cand in zip(reference_records, candidate_records, strict=True):
        identity = (ref["before_image"], ref["after_image"], ref["severity"])
        candidate_identity = (cand["before_image"], cand["after_image"], cand["severity"])
        if identity != candidate_identity:
            raise ValueError("planner gate record order or identity differs")
        exact.append(
            (ref["predicted_status"], ref["predicted_failure"])
            == (cand["predicted_status"], cand["predicted_failure"])
        )
        intervention.append(
            bool(ref["predicted_intervention"]) == bool(cand["predicted_intervention"])
        )
    return {
        "exact_semantic_decision_agreement": float(np.mean(exact)),
        "intervention_agreement": float(np.mean(intervention)),
    }


def vla_rows() -> list[dict[str, Any]]:
    specs = (
        ("V0", "Eager BF16, 7 steps", "pi05_eager_bf16_7step_h10.json", "behavior_reference"),
        ("V1", "Eager BF16, 2 steps", "pi05_eager_bf16_2step_h10.json", "reject_deadline"),
        ("V2", "Compile BF16, 2 steps", "pi05_torch_compile_bf16_2step_h10.json", "retain_fallback"),
        ("V3", "Compile BF16 + SMVE, 2 steps", "pi05_smve_bf16_2step_h10.json", "promote_realtime"),
        ("V4", "Late-language INT8, 2 steps", "pi05_late_int8_2step_h10.json", "reject_current_backend"),
    )
    rows: list[dict[str, Any]] = []
    for profile_id, label, filename, decision in specs:
        payload = load(STUDY / "vla" / filename)
        benchmark = payload["benchmark"]
        samples = benchmark["metadata"]["latency_samples"]["runtime_ms"]
        fidelity = payload["fidelity"]
        rows.append(
            {
                "profile_id": profile_id,
                "label": label,
                "runtime_p50_ms": float(benchmark["runtime_p50_ms"]),
                "runtime_p95_ms": float(benchmark["runtime_p95_ms"]),
                "runtime_p99_ms": float(benchmark["runtime_p99_ms"]),
                "deadline_miss_rate_80ms": float(np.mean(np.asarray(samples) > 80.0)),
                "peak_vram_gib": float(benchmark["peak_vram_gb"]),
                "fidelity_passed": bool(fidelity["passed"]),
                "fidelity_samples": int(fidelity["samples"]),
                "decision": decision,
                "source": str(STUDY / "vla" / filename),
            }
        )
    reference = rows[0]
    for row in rows:
        row["p95_change_vs_v0_percent"] = change(
            row["runtime_p95_ms"], reference["runtime_p95_ms"]
        )
        row["vram_change_vs_v0_percent"] = change(
            row["peak_vram_gib"], reference["peak_vram_gib"]
        )
    return rows


def planner_rows() -> list[dict[str, Any]]:
    specs = (
        ("Q0", "Qwen3.5-4B BF16", "qwen35_4b_bf16", "promote_latency"),
        ("Q1", "Qwen3.5-4B INT8", "qwen35_4b_uniform_int8", "reject_current_kernel"),
        ("Q2", "Qwen3.5-4B NF4", "qwen35_4b_uniform_nf4", "retain_memory_tier"),
    )
    reference_gate = load(STUDY / "planner/qwen35_4b_bf16_semantic_gate.json")
    rows: list[dict[str, Any]] = []
    for profile_id, label, stem, decision in specs:
        receipt = load(STUDY / "planner" / f"{stem}_receipt.json")
        gate = load(STUDY / "planner" / f"{stem}_semantic_gate.json")
        metrics = gate["metrics"]
        agreement = paired_agreement(reference_gate, gate)
        memory = receipt["cuda_memory_after_load"]
        rows.append(
            {
                "profile_id": profile_id,
                "label": label,
                "allocated_vram_gib": float(memory["allocated_gib"]),
                "peak_allocated_vram_gib": float(memory["peak_allocated_gib"]),
                "load_seconds": float(receipt["load_seconds"]),
                "latency_mean_ms": float(metrics["latency_ms_mean"]),
                "latency_p95_ms": float(metrics["latency_ms_p95"]),
                "protocol_valid_rate": float(metrics["protocol_valid_rate"]),
                "no_op_false_positive_rate": float(metrics["no_op_false_positive_rate"]),
                "severe_intervention_recall": float(metrics["severe_intervention_recall"]),
                "semantic_gate_passed": bool(gate["passed"]),
                **agreement,
                "decision": decision,
                "receipt": str(STUDY / "planner" / f"{stem}_receipt.json"),
                "semantic_gate": str(STUDY / "planner" / f"{stem}_semantic_gate.json"),
            }
        )
    reference = rows[0]
    for row in rows:
        row["vram_change_vs_p0_percent"] = change(
            row["allocated_vram_gib"], reference["allocated_vram_gib"]
        )
        row["p95_change_vs_p0_percent"] = change(
            row["latency_p95_ms"], reference["latency_p95_ms"]
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(vla: list[dict[str, Any]], planner: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)

    vla_colors = ["#7D8790", "#7D8790", "#2F6B9A", "#159D88", "#D56A45"]
    axes[0].bar([row["profile_id"] for row in vla], [row["runtime_p95_ms"] for row in vla], color=vla_colors)
    axes[0].axhline(80.0, color="#B53A3A", linestyle="--", linewidth=1.2, label="80 ms deadline")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("VLA runtime P95 (ms, log scale)")
    axes[0].set_title("PI0.5 Optimize Runtime")
    axes[0].legend(frameon=False, loc="upper left")

    x = np.arange(len(planner))
    width = 0.36
    axes[1].bar(x - width / 2, [row["allocated_vram_gib"] for row in planner], width, color="#2F6B9A", label="VRAM (GiB)")
    latency_axis = axes[1].twinx()
    latency_axis.bar(x + width / 2, [row["latency_p95_ms"] / 1000.0 for row in planner], width, color="#D56A45", label="P95 (s)")
    axes[1].set_xticks(x, [row["profile_id"] for row in planner])
    axes[1].set_ylabel("Allocated VRAM (GiB)")
    latency_axis.set_ylabel("Planner latency P95 (s)")
    axes[1].set_title("Qwen3.5-4B Planner Quantization")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = latency_axis.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")

    figure.savefig(STUDY / "efficiency_ablation.png", dpi=220)
    figure.savefig(STUDY / "efficiency_ablation.pdf")
    plt.close(figure)


def main() -> int:
    STUDY.mkdir(parents=True, exist_ok=True)
    vla = vla_rows()
    planner = planner_rows()
    compatible_int8 = load(
        ROOT / "results/carve_optimize/pi05_torchao_int8_vlm_late_fidelity45_20260825.json"
    )
    closed_loop = load(ROOT / "results/carve_optimize/pi05_quantized_closed_loop_gate.json")
    payload = {
        "schema_version": "carve-efficiency-full-study-v1",
        "date": "2026-08-26",
        "hardware": "NVIDIA GeForce RTX 4090 24GB",
        "vla_protocol": {
            "checkpoint": "pi05_libero_pytorch",
            "paired_recorded_observations": 45,
            "fixed_noise": True,
            "action_horizon": 10,
            "deadline_ms": 80.0,
        },
        "planner_protocol": {
            "model": "Qwen3.5-4B",
            "paired_temporal_observations": 30,
            "semantic_protocol": "code_v5",
            "profiles": ["BF16", "INT8", "NF4"],
        },
        "vla_rows": vla,
        "planner_rows": planner,
        "version_sensitivity": {
            "current_torch": vla[-1]["source"],
            "current_int8_p95_ms": vla[-1]["runtime_p95_ms"],
            "compatible_stack_source": str(
                ROOT / "results/carve_optimize/pi05_torchao_int8_vlm_late_fidelity45_20260825.json"
            ),
            "compatible_stack": compatible_int8["hardware"]["software"],
            "compatible_int8_p95_ms": compatible_int8["benchmark"]["runtime_p95_ms"],
        },
        "closed_loop_quantization_gate": closed_loop,
        "recommended_profiles": {
            "vla_realtime": "V3",
            "vla_fallback": "V2",
            "vla_low_memory": "late-language INT8 only on the validated compatible stack",
            "planner_latency": "Q0",
            "planner_low_memory": "Q2",
        },
    }
    (STUDY / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(STUDY / "vla_ablation.csv", vla)
    write_csv(STUDY / "planner_ablation.csv", planner)
    plot(vla, planner)

    lines = [
        "# CARVE Full Efficient-Inference Ablation",
        "",
        "## Protocol",
        "",
        "- Hardware: one NVIDIA GeForce RTX 4090 24GB.",
        "- VLA: one PI0.5 checkpoint, 45 paired recorded observations, fixed diffusion noise, action horizon 10.",
        "- Planner: one Qwen3.5-4B checkpoint, 30 paired temporal observations, identical `code_v5` prompt and decoding budget.",
        "- The 80 ms VLA deadline is recomputed from persisted per-call samples for every profile.",
        "",
        "## PI0.5 Optimize Runtime",
        "",
        "| ID | Profile | P95 | 80 ms miss | Peak VRAM | Fidelity | Decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in vla:
        fidelity = "reference" if row["fidelity_samples"] == 0 else f"{'pass' if row['fidelity_passed'] else 'fail'} ({row['fidelity_samples']})"
        lines.append(
            f"| {row['profile_id']} | {row['label']} | {row['runtime_p95_ms']:.2f} ms | "
            f"{100 * row['deadline_miss_rate_80ms']:.1f}% | {row['peak_vram_gib']:.2f} GiB | {fidelity} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            f"V3 reduces P95 by `{abs(vla[3]['p95_change_vs_v0_percent']):.1f}%` relative to V0 and passes all 45 fidelity checks.",
            "V4 passes action fidelity but is rejected on the current Torch/TorchAO stack because its P95 exceeds the deadline.",
            "",
            "## Qwen3.5-4B Planner Quantization",
            "",
            "| ID | Profile | VRAM | P95 | Semantic agreement | Gate | Decision |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in planner:
        lines.append(
            f"| {row['profile_id']} | {row['label']} | {row['allocated_vram_gib']:.2f} GiB | "
            f"{row['latency_p95_ms'] / 1000.0:.2f} s | {100 * row['exact_semantic_decision_agreement']:.1f}% | "
            f"{'pass' if row['semantic_gate_passed'] else 'fail'} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            f"Q2 saves `{abs(planner[2]['vram_change_vs_p0_percent']):.1f}%` allocated VRAM relative to Q0 while passing the semantic gate, but its P95 is `{planner[2]['p95_change_vs_p0_percent']:.1f}%` higher.",
            "Q1 is neither the fastest nor the smallest profile on the current bitsandbytes kernels and is rejected.",
            "",
            "## Closed-Loop and Version Boundary",
            "",
            f"- On the previously validated Torch `{compatible_int8['hardware']['software']['torch']}` / TorchAO `{compatible_int8['hardware']['software']['torchao']}` stack, late-language INT8 reached P95 `{compatible_int8['benchmark']['runtime_p95_ms']:.2f} ms`, passed 45/45 fidelity checks, and completed 2/2 matched recovery branches.",
            f"- On the current Torch stack, the same logical profile reached P95 `{vla[-1]['runtime_p95_ms']:.2f} ms`; this profile is therefore version-gated rather than generally promoted.",
            "- INT8+SMVE remains rejected because the existing closed-loop gate regressed one of two matched recovery branches, despite favorable replay latency.",
            "",
            "## Final Deployment Decision",
            "",
            "- Default realtime VLA tier: V3 compiled BF16 + SMVE.",
            "- Contract fallback: V2 compiled BF16 when the masked-view precondition is false.",
            "- Default Planner tier: Q0 BF16 for latency; Q2 NF4 only when shared-GPU capacity is the binding constraint.",
            "- Quantized profiles are admitted only after semantic/action-fidelity and closed-loop gates; memory reduction alone is insufficient.",
        ]
    )
    (STUDY / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"study": str(STUDY), "vla_profiles": len(vla), "planner_profiles": len(planner)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

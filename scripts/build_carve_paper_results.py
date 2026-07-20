#!/usr/bin/env python3
"""Generate CARVE-VLA manuscript tables from accepted experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "paper" / "CARVE-VLA" / "generated"


def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def load_completed_result(relative_dir: str) -> dict[str, Any]:
    payload = load_json(f"{relative_dir}/results.json")
    if payload.get("status") != "completed":
        raise ValueError(f"result is not complete: {relative_dir}")
    return payload


def load_traces(relative_dir: str) -> list[dict[str, Any]]:
    path = ROOT / relative_dir / "episode_traces.jsonl"
    traces = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"no episode traces in {path}")
    return traces


def inference_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    inference = payload["trace_aggregate"]["inference"]
    episodes = int(payload["total_episodes"])
    calls = int(inference["vla_calls_total"])
    return {
        "successes": int(payload["total_successes"]),
        "episodes": episodes,
        "calls": calls,
        "model_wall_ms": float(inference["vla_model_wall_ms_per_episode"]) * episodes,
        "episode_wall_sec": float(inference["episode_wall_sec_mean"]) * episodes,
    }


def combine_results(payloads: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    rows = [inference_metrics(payload) for payload in payloads]
    episodes = sum(int(row["episodes"]) for row in rows)
    calls = sum(int(row["calls"]) for row in rows)
    model_wall_ms = sum(float(row["model_wall_ms"]) for row in rows)
    return {
        "successes": sum(int(row["successes"]) for row in rows),
        "episodes": episodes,
        "calls": calls,
        "call_ms": model_wall_ms / calls,
        "model_wall_sec_per_episode": model_wall_ms / episodes / 1000.0,
        "episode_wall_sec_per_episode": (
            sum(float(row["episode_wall_sec"]) for row in rows) / episodes
        ),
    }


def profile_row(
    performance_path: str,
    *,
    fidelity_path: str | None = None,
    decision: str,
) -> dict[str, Any]:
    performance = load_json(performance_path)
    fidelity = load_json(fidelity_path) if fidelity_path else performance
    benchmark = performance["benchmark"]
    fidelity_result = fidelity["fidelity"]
    return {
        "p50": float(benchmark["runtime_p50_ms"]),
        "p95": float(benchmark["runtime_p95_ms"]),
        "miss_rate": float(benchmark["deadline_miss_rate"]),
        "vram": float(benchmark["peak_vram_gb"]),
        "fidelity_samples": int(fidelity_result["samples"]),
        "fidelity_passed": bool(fidelity_result["passed"]),
        "decision": decision,
    }


def trace_metrics(relative_dir: str) -> dict[str, Any]:
    traces = load_traces(relative_dir)
    p95_values = [float(trace["realtime"]["control_latency_ms_p95"]) for trace in traces]
    return {
        "successes": sum(bool(trace["success"]) for trace in traces),
        "episodes": len(traces),
        "steps": sum(int(trace["episode_steps"]) for trace in traces),
        "calls": sum(int(trace["vla_calls"]) for trace in traces),
        "wall_sec": sum(float(trace["latency"]["episode_wall_sec"]) for trace in traces),
        "misses": sum(int(trace["realtime"]["deadline_misses"]) for trace in traces),
        "control_steps": sum(int(trace["realtime"]["control_steps"]) for trace in traces),
        "control_p95_min": min(p95_values),
        "control_p95_max": max(p95_values),
        "physical_triggered": sum(
            int(trace["physical_recoveries_triggered"]) for trace in traces
        ),
        "physical_verified": sum(
            int(trace["physical_recoveries_verified"]) for trace in traces
        ),
    }


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_text(name: str, content: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(content)


def build() -> dict[str, Any]:
    baseline = load_completed_result("results/carve_t689_paired_5states_v4/baseline")
    calibrated = combine_results(
        [
            load_completed_result("results/carve_t689_paired_5states_v4/fixed2_t69"),
            load_completed_result("results/carve_t689_paired_5states_v4/fixed2_t8"),
        ]
    )
    dynamic = load_completed_result("results/carve_t689_paired_5states_v6/joint")
    calibration = {
        "reference": combine_results([baseline]),
        "calibrated": calibrated,
        "dynamic": combine_results([dynamic]),
    }

    profiles = {
        "eager_bf16": profile_row(
            "results/carve_optimize/pi05_eager_bf16_2step_h10_deadline80_rep1.json",
            decision="reference",
        ),
        "compiled_bf16": profile_row(
            "results/carve_optimize/"
            "pi05_torch_compile_reduce_overhead_bf16_2step_h10_deadline80_rep1.json",
            fidelity_path=(
                "results/carve_optimize/pi05_torch_compile_bf16_2step_h10_fidelity45.json"
            ),
            decision="default",
        ),
        "compiled_smve": profile_row(
            "results/carve_optimize/"
            "pi05_masked_view_compile_bf16_2step_h10_fidelity45_rep1.json",
            decision="padded-view default",
        ),
        "compiled_w8a16": profile_row(
            "results/carve_optimize/"
            "pi05_torchao_w8a16_vlm_early4_fidelity45_deadline80_rep1.json",
            decision="closed-loop rejected",
        ),
    }

    sync_ten = trace_metrics(
        "results/carve_realtime/heldout_t89_sync_boundfix_20260717"
    )
    sync_eight = trace_metrics(
        "results/carve_realtime/heldout_t89_5states_sync_commit8_20260717"
    )
    async_two = trace_metrics(
        "results/carve_realtime/heldout_t89_5states_async_lead2_20260717"
    )
    async_interval_two = trace_metrics(
        "results/carve_realtime/heldout_t89_5states_async_interval2_20260717"
    )
    realtime = {
        "sync_commit10": sync_ten,
        "sync_commit8": sync_eight,
        "async_lead2": async_two,
        "async_interval2": async_interval_two,
        "relative_miss_reduction": 1.0
        - (async_two["misses"] / async_two["control_steps"])
        / (sync_eight["misses"] / sync_eight["control_steps"]),
        "relative_wall_reduction": 1.0 - async_two["wall_sec"] / sync_eight["wall_sec"],
    }
    smve_t89 = trace_metrics(
        "results/carve_optimize/masked_view_t89_5states_20260717"
    )
    smve_gate = {
        "compiled_bf16": sync_eight,
        "compiled_smve": smve_t89,
        "replay_runtime_p50_reduction": 1.0
        - profiles["compiled_smve"]["p50"] / profiles["compiled_bf16"]["p50"],
        "replay_runtime_p95_reduction": 1.0
        - profiles["compiled_smve"]["p95"] / profiles["compiled_bf16"]["p95"],
    }
    droid_profiles = {
        "compiled_bf16": profile_row(
            "results/carve_optimize/"
            "pi05_droid_compile_cross_adapter_h5_10state_20260717.json",
            decision="accepted reference",
        ),
        "compiled_smve": profile_row(
            "results/carve_optimize/"
            "pi05_droid_masked_view_cross_adapter_h5_10state_20260717.json",
            decision="accepted",
        ),
    }
    droid_h15 = load_json(
        "results/carve_optimize/"
        "pi05_droid_masked_view_cross_adapter_10state_20260717.json"
    )
    cross_adapter = {
        "compiled_bf16": droid_profiles["compiled_bf16"],
        "compiled_smve": droid_profiles["compiled_smve"],
        "runtime_p50_reduction": 1.0
        - droid_profiles["compiled_smve"]["p50"]
        / droid_profiles["compiled_bf16"]["p50"],
        "runtime_p95_reduction": 1.0
        - droid_profiles["compiled_smve"]["p95"]
        / droid_profiles["compiled_bf16"]["p95"],
        "h15_rejected": {
            "fidelity_passed": bool(droid_h15["fidelity"]["passed"]),
            "endpoint_l2": float(droid_h15["fidelity"]["metrics"]["endpoint_l2"]),
            "violations": list(droid_h15["fidelity"]["violations"]),
        },
        "claim_boundary": "pi0.5 cross-checkpoint/input-adapter portability only",
    }
    recovery = load_json(
        "results/carve_pi05_recovery_challenge_20260719/summary.json"
    )
    if not recovery.get("gates", {}).get("complete", False):
        raise ValueError("PI0.5 recovery challenge did not pass its systems gate")
    agentic_optimize = load_json(
        "results/carve_pi05_agentic_optimize_pair_20260719/summary.json"
    )
    if not agentic_optimize.get("gates", {}).get("evidence_complete", False):
        raise ValueError("PI0.5 Agentic-Optimize pair is incomplete")

    calibration_rows = [
        ("Reference: 7 steps, commit 10", calibration["reference"]),
        (r"Calibrated: 2 steps, commit 10", calibration["calibrated"]),
        (r"Dynamic: 2/4 steps, commit 10", calibration["dynamic"]),
    ]
    calibration_tex = [
        r"\begin{table}",
        r"\caption{Paired flow-step calibration on 15 T6/T8/T9 episodes.}",
        r"\label{tab:calibration}",
        r"\centering\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Profile & Success & Call (ms) & VLA/ep (s) & Wall/ep (s) \\",
        r"\midrule",
    ]
    for label, row in calibration_rows:
        calibration_tex.append(
            f"{label} & {row['successes']}/{row['episodes']} & "
            f"{row['call_ms']:.1f} & {row['model_wall_sec_per_episode']:.2f} & "
            f"{row['episode_wall_sec_per_episode']:.2f} " + r"\\"
        )
    calibration_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text("calibration_table.tex", "\n".join(calibration_tex))

    profile_labels = [
        ("Eager BF16", profiles["eager_bf16"]),
        (r"Compiled BF16", profiles["compiled_bf16"]),
        (r"Compiled BF16 + SMVE", profiles["compiled_smve"]),
        (r"Compiled W8A16", profiles["compiled_w8a16"]),
    ]
    profile_tex = [
        r"\begin{table}",
        r"\caption{RTX 4090 deployment profiles at two flow steps and horizon 10.}",
        r"\label{tab:deployment}",
        r"\centering\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Profile & P50 & P95 & Miss & VRAM & Decision \\",
        r"\midrule",
    ]
    for label, row in profile_labels:
        miss = f"{100.0 * row['miss_rate']:.0f}\\%"
        profile_tex.append(
            f"{label} & {row['p50']:.2f} & {row['p95']:.2f} & {miss} & "
            f"{row['vram']:.2f} GB & {tex_escape(row['decision'])} " + r"\\"
        )
    profile_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text("deployment_table.tex", "\n".join(profile_tex))

    realtime_rows = [
        ("Sync, commit 8", sync_eight),
        ("Async, every chunk", async_two),
        ("Async, alternating", async_interval_two),
    ]
    realtime_tex = [
        r"\begin{table}",
        r"\caption{Expanded T8/T9 asynchronous-execution gate (ten episodes).}",
        r"\label{tab:prefetch}",
        r"\centering\footnotesize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Execution & Success & Calls & Wall (s) & 80-ms miss & Miss rate \\",
        r"\midrule",
    ]
    for label, row in realtime_rows:
        realtime_tex.append(
            f"{label} & {row['successes']}/{row['episodes']} & {row['calls']} & "
            f"{row['wall_sec']:.2f} & {row['misses']}/{row['control_steps']} & "
            f"{100.0 * row['misses'] / row['control_steps']:.2f}\\% " + r"\\"
        )
    realtime_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text("prefetch_table.tex", "\n".join(realtime_tex))

    smve_rows = [
        ("Compiled BF16", sync_eight),
        ("Compiled BF16 + SMVE", smve_t89),
    ]
    smve_tex = [
        r"\begin{table}",
        r"\caption{Paired T8/T9 masked-view elision gate (ten episodes).}",
        r"\label{tab:smve}",
        r"\centering\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Profile & Success & Calls & Wall (s) & Wall/ep (s) \\",
        r"\midrule",
    ]
    for label, row in smve_rows:
        smve_tex.append(
            f"{label} & {row['successes']}/{row['episodes']} & {row['calls']} & "
            f"{row['wall_sec']:.2f} & {row['wall_sec'] / row['episodes']:.2f} "
            + r"\\"
        )
    smve_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text("smve_table.tex", "\n".join(smve_tex))

    cross_adapter_tex = [
        r"\begin{table}",
        r"\caption{Cross-adapter pi0.5 systems gate on fixed DROID-schema inputs.}",
        r"\label{tab:cross_adapter}",
        r"\centering\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Profile & P50 & P95 & Miss & Fidelity & Decision \\",
        r"\midrule",
    ]
    for label, row in (
        ("Compiled BF16", droid_profiles["compiled_bf16"]),
        ("Compiled BF16 + SMVE", droid_profiles["compiled_smve"]),
    ):
        cross_adapter_tex.append(
            f"{label} & {row['p50']:.2f} & {row['p95']:.2f} & "
            f"{100.0 * row['miss_rate']:.0f}\\% & {row['fidelity_samples']}/"
            f"{row['fidelity_samples']} & {tex_escape(row['decision'])} " + r"\\"
        )
    cross_adapter_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text("cross_adapter_table.tex", "\n".join(cross_adapter_tex))

    recovery_tex = [
        r"\begin{table}",
        r"\caption{Exact-state PI0.5 recovery challenge in LIBERO MuJoCo.}",
        r"\label{tab:recovery_challenge}",
        r"\centering\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Intervention & Success & VLA calls & Executed & Safe stop \\",
        r"\midrule",
    ]
    for key, label in (
        ("continue", "Frozen continuation"),
        ("accurate", "Frequent replan"),
        ("recovery", "Prompt retry"),
        ("physical_recovery", "Physical recovery"),
    ):
        row = recovery["branches"][key]
        recovery_tex.append(
            f"{label} & {row['successes']}/{row['scenarios']} & "
            f"{row['vla_calls']} & {row['executed_scenarios']}/{row['scenarios']} & "
            f"{row['safe_stops']} " + r"\\"
        )
    recovery_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text("recovery_challenge_table.tex", "\n".join(recovery_tex))

    agentic_optimize_tex = [
        r"\begin{table}",
        r"\caption{Same-state Agentic recovery under PI0.5 runtime profiles.}",
        r"\label{tab:agentic_optimize}",
        r"\centering\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Profile & Success & Recovery & P95 (ms) & 80-ms miss \\",
        r"\midrule",
    ]
    for key in ("eager_bf16", "compiled_bf16", "compiled_smve"):
        row = agentic_optimize["profiles"][key]
        agentic_optimize_tex.append(
            f"{row['label']} & {row['exact_state_successes']}/"
            f"{row['exact_state_scenarios']} & {row['verified_recoveries']}/"
            f"{row['exact_state_scenarios']} & {row['runtime_p95_ms']:.2f} & "
            f"{row['deadline_misses']}/{row['deadline_calls']} " + r"\\"
        )
    agentic_optimize_tex.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    )
    write_text("agentic_optimize_table.tex", "\n".join(agentic_optimize_tex))

    macros = [
        "% Generated by scripts/build_carve_paper_results.py; do not edit manually.",
        rf"\newcommand{{\CarveRealtimeMissReduction}}{{{100.0 * realtime['relative_miss_reduction']:.1f}\%}}",
        rf"\newcommand{{\CarveRealtimeWallReduction}}{{{100.0 * realtime['relative_wall_reduction']:.1f}\%}}",
        rf"\newcommand{{\CarveSyncEightSuccess}}{{{sync_eight['successes']}/{sync_eight['episodes']}}}",
        rf"\newcommand{{\CarveAsyncSuccess}}{{{async_two['successes']}/{async_two['episodes']}}}",
        rf"\newcommand{{\CarveAsyncIntervalSuccess}}{{{async_interval_two['successes']}/{async_interval_two['episodes']}}}",
        rf"\newcommand{{\CarveSyncTenSuccess}}{{{sync_ten['successes']}/{sync_ten['episodes']}}}",
        rf"\newcommand{{\CarveSyncTenRecovery}}{{{sync_ten['physical_verified']}/{sync_ten['physical_triggered']}}}",
        rf"\newcommand{{\CarveCompiledPninetyfive}}{{{profiles['compiled_bf16']['p95']:.2f}~ms}}",
        rf"\newcommand{{\CarveSmvePninetyfive}}{{{profiles['compiled_smve']['p95']:.2f}~ms}}",
        rf"\newcommand{{\CarveSmveSuccess}}{{{smve_t89['successes']}/{smve_t89['episodes']}}}",
        rf"\newcommand{{\CarveSmveRuntimeReduction}}{{{100.0 * smve_gate['replay_runtime_p95_reduction']:.1f}\%}}",
        rf"\newcommand{{\CarveDroidSmvePninetyfive}}{{{droid_profiles['compiled_smve']['p95']:.2f}~ms}}",
        rf"\newcommand{{\CarveDroidSmveReduction}}{{{100.0 * cross_adapter['runtime_p95_reduction']:.1f}\%}}",
        rf"\newcommand{{\CarveWMemory}}{{{profiles['compiled_w8a16']['vram']:.2f}~GB}}",
        rf"\newcommand{{\CarveRecoveryOnlineSuccess}}{{{recovery['online_controller']['successes']}/{recovery['online_controller']['episodes']}}}",
        rf"\newcommand{{\CarveRecoveryOnlinePninetyfive}}{{{recovery['online_controller']['vla_p95_ms']:.2f}~ms}}",
        rf"\newcommand{{\CarveAgenticSmvePninetyfive}}{{{agentic_optimize['profiles']['compiled_smve']['runtime_p95_ms']:.2f}~ms}}",
        rf"\newcommand{{\CarveAgenticSmveReduction}}{{{100.0 * agentic_optimize['comparisons']['smve_runtime_p95_reduction_vs_eager']:.1f}\%}}",
        "",
    ]
    write_text("runtime_macros.tex", "\n".join(macros))

    summary = {
        "sources": {
            "calibration": "results/carve_t689_paired_5states_v4 and v6",
            "deployment": "results/carve_optimize",
            "realtime": "results/carve_realtime/heldout_t89_*_20260717",
            "smve": "results/carve_optimize/masked_view_t89_5states_20260717",
            "cross_adapter": "results/carve_optimize/pi05_droid_*_20260717.json",
            "recovery": "results/carve_pi05_recovery_challenge_20260719",
            "agentic_optimize": "results/carve_pi05_agentic_optimize_pair_20260719",
        },
        "calibration": calibration,
        "profiles": profiles,
        "realtime": realtime,
        "smve_gate": smve_gate,
        "cross_adapter": cross_adapter,
        "recovery": recovery,
        "agentic_optimize": agentic_optimize,
    }
    write_text("runtime_results_summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    result = build()
    print(
        "generated CARVE paper assets: "
        f"miss reduction={100.0 * result['realtime']['relative_miss_reduction']:.1f}%"
    )

#!/usr/bin/env python3
"""Build the canonical paper-ready evidence package from accepted results."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "paper_ready_20260827"

SOURCES = {
    "benchmark": Path(
        "results/libero_pro_full_study_20260825/aggregate/study_summary.json"
    ),
    "efficiency": Path("results/carve_efficiency_full_20260826/summary.json"),
    "full_agent": Path("results/full_embodied_agent_20260827/aggregate_validated.json"),
    "t8_routing": Path("results/planner_profile_closed_loop_20260827/aggregate.json"),
    "t3_routing": Path("results/cross_task_memory_routing_20260827/aggregate.json"),
}

REPRESENTATIVE_ARTIFACTS = [
    {
        "id": "benchmark_agentic_conversion_object_t7_r7",
        "kind": "video",
        "purpose": "Full-study Agentic conversion on Object T7 state 7",
        "path": Path(
            "results/libero_pro_full_study_20260825/libero_10_object/agentic/"
            "task_07/carve-full-study-agentic-libero_10_object-t7-r7-s7/episode.mp4"
        ),
    },
    {
        "id": "benchmark_frozen_pair_object_t7_r7",
        "kind": "video",
        "purpose": "Paired Frozen VLA reference for Object T7 state 7",
        "path": Path(
            "results/libero_pro_full_study_20260825/libero_10_object/frozen_vla/"
            "task_07/carve-full-study-frozen_vla-libero_10_object-t7-r7-s7/episode.mp4"
        ),
    },
    {
        "id": "full_agent_recovery_t8_r9",
        "kind": "video",
        "purpose": "Full VLM-Harness-Memory-Recovery-PI0.5 loop with recovery",
        "path": Path(
            "results/full_embodied_agent_20260827/heldout_t8_trial9_executor_gate_v3/"
            "full-embodied-memory-heldout-v3-agentic-libero_10_object-t8-r9-s7-"
            "startup-planner-semantic-checkpoint/episode.mp4"
        ),
    },
    {
        "id": "cross_task_memory_4b_t3_r1",
        "kind": "video",
        "purpose": "Memory+4B route on held-out T3 state 1",
        "path": Path(
            "results/cross_task_memory_routing_20260827/memory_4b_trial1_v2/"
            "cross-task-memory-4b-v2-agentic-libero_10-t3-r1-s7-semantic-"
            "checkpoint/episode.mp4"
        ),
    },
    {
        "id": "cross_task_memory_4b_t3_r7",
        "kind": "video",
        "purpose": "Memory+4B route on held-out T3 state 7",
        "path": Path(
            "results/cross_task_memory_routing_20260827/memory_4b_trial7/"
            "cross-task-memory-4b-agentic-libero_10-t3-r7-s7-semantic-"
            "checkpoint/episode.mp4"
        ),
    },
    {
        "id": "cross_task_memory_4b_t3_r9",
        "kind": "video",
        "purpose": "Memory+4B route on held-out T3 state 9",
        "path": Path(
            "results/cross_task_memory_routing_20260827/memory_4b_trial9/"
            "cross-task-memory-4b-agentic-libero_10-t3-r9-s7-semantic-"
            "checkpoint/episode.mp4"
        ),
    },
    {
        "id": "embodied_tool_probe",
        "kind": "video",
        "purpose": "RGB-D analytic embodied-tool qualification",
        "path": Path(
            "results/libero_pro_embodied_tool_probe_20260827/embodied_tool_probe.mp4"
        ),
    },
    {
        "id": "vlm_grounded_tool_use",
        "kind": "video",
        "purpose": "VLM grounding, schema gate and physical tool dispatch",
        "path": Path(
            "results/vlm_embodied_tool_use_20260827/vlm_grounded_tool_use.mp4"
        ),
    },
    {
        "id": "efficiency_ablation_figure",
        "kind": "figure",
        "purpose": "PI0.5 and Planner efficient-inference ablation",
        "path": Path("results/carve_efficiency_full_20260826/efficiency_ablation.png"),
    },
]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads((ROOT / path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUTPUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def video_metadata(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "duration_s": float(payload["format"]["duration"]),
    }


def relative_link(path: Path) -> str:
    return "../" + path.relative_to("results").as_posix()


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    benchmark = load_json(SOURCES["benchmark"])
    efficiency = load_json(SOURCES["efficiency"])
    full_agent = load_json(SOURCES["full_agent"])
    t8_routing = load_json(SOURCES["t8_routing"])
    t3_routing = load_json(SOURCES["t3_routing"])

    if benchmark["episode_count"] != 1200:
        raise ValueError("Full benchmark evidence must contain 1,200 episodes")
    if full_agent["aggregate"]["successes"] != 3:
        raise ValueError("Full embodied-agent gate must preserve 3/3 successes")
    if t3_routing["memory_4b_bf16"]["metrics"]["successes"] != 3:
        raise ValueError("Cross-task Memory+4B route must preserve 3/3 successes")

    benchmark_rows = []
    for method in ("frozen_vla", "fixed_recovery", "agentic"):
        row = benchmark["aggregate"][method]
        benchmark_rows.append(
            {
                "method": method,
                "successes": row["successes"],
                "trials": row["trials"],
                "success_rate_percent": round(100 * row["success_rate"], 3),
                "control_steps": row["episode_steps"],
                "vla_calls": row["vla_calls"],
                "planner_calls": row["planner_calls"],
                "recovery_calls": row["recovery_calls"],
                "safe_stops": row["safe_stops"],
                "vla_p95_ms": round(row["vla_runtime_p95_ms"], 3),
                "deadline_misses": row["vla_deadline_misses"],
            }
        )
    write_csv("agentic_benchmark.csv", benchmark_rows)

    vla_rows = [
        {
            "profile_id": row["profile_id"],
            "profile": row["label"],
            "p50_ms": round(row["runtime_p50_ms"], 3),
            "p95_ms": round(row["runtime_p95_ms"], 3),
            "p99_ms": round(row["runtime_p99_ms"], 3),
            "miss_at_80ms_percent": round(100 * row["deadline_miss_rate_80ms"], 3),
            "peak_vram_gib": round(row["peak_vram_gib"], 3),
            "fidelity_samples": row["fidelity_samples"],
            "decision": row["decision"],
        }
        for row in efficiency["vla_rows"]
    ]
    write_csv("vla_runtime.csv", vla_rows)

    planner_rows = [
        {
            "profile_id": row["profile_id"],
            "profile": row["label"],
            "allocated_vram_gib": round(row["allocated_vram_gib"], 3),
            "latency_mean_ms": round(row["latency_mean_ms"], 3),
            "latency_p95_ms": round(row["latency_p95_ms"], 3),
            "semantic_agreement_percent": round(
                100 * row["exact_semantic_decision_agreement"], 3
            ),
            "decision": row["decision"],
        }
        for row in efficiency["planner_rows"]
    ]
    write_csv("planner_runtime.csv", planner_rows)

    t8_reference = t8_routing["qwen9b_nf4_reference"]["aggregate"]
    t8_reference_metrics = {
        "successes": t8_reference["successes"],
        "trials": t8_reference["trials"],
        "startup_planner_calls": t8_reference["trials"],
        "event_planner_calls": 0,
        "critic_calls": round(
            t8_reference["critic_calls_mean"] * t8_reference["trials"]
        ),
        "wall_time_mean_s": t8_reference["wall_time_mean_s"],
        "critic_latency_mean_ms": None,
        "vla_runtime_p95_ms": t8_reference["vla_runtime_latency_p95_ms"],
        "vla_deadline_misses": t8_reference["vla_deadline_misses"],
    }
    route_specs = [
        ("T8", "Direct 9B NF4", t8_reference_metrics),
        (
            "T8",
            "Memory + 4B BF16",
            t8_routing["memory_routed_qwen4b_bf16_common"]["metrics"],
        ),
        ("T3", "Direct 9B NF4", t3_routing["direct_9b_nf4"]["metrics"]),
        ("T3", "Memory + 9B NF4", t3_routing["memory_9b_nf4"]["metrics"]),
        ("T3", "Memory + 4B BF16", t3_routing["memory_4b_bf16"]["metrics"]),
    ]
    routing_rows = []
    for task, route, metrics in route_specs:
        routing_rows.append(
            {
                "task": task,
                "route": route,
                "successes": metrics["successes"],
                "trials": metrics["trials"],
                "startup_planner_calls": metrics.get("startup_planner_calls", 0),
                "event_planner_calls": metrics.get(
                    "event_planner_calls", metrics.get("planner_calls", 0)
                ),
                "critic_calls": metrics["critic_calls"],
                "wall_time_mean_s": round(metrics["wall_time_mean_s"], 3),
                "critic_latency_mean_ms": (
                    round(metrics["critic_latency_mean_ms"], 3)
                    if metrics["critic_latency_mean_ms"] is not None
                    else ""
                ),
                "vla_p95_ms": round(metrics["vla_runtime_p95_ms"], 3),
                "deadline_misses": metrics["vla_deadline_misses"],
            }
        )
    write_csv("memory_routing.csv", routing_rows)

    artifact_rows = []
    for item in REPRESENTATIVE_ARTIFACTS:
        absolute = ROOT / item["path"]
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        row = {
            "id": item["id"],
            "kind": item["kind"],
            "purpose": item["purpose"],
            "path": item["path"].as_posix(),
            "bytes": absolute.stat().st_size,
            "sha256": sha256(absolute),
        }
        if item["kind"] == "video":
            row["video"] = video_metadata(absolute)
        artifact_rows.append(row)

    source_rows = []
    for name, path in SOURCES.items():
        absolute = ROOT / path
        source_rows.append(
            {
                "id": name,
                "path": path.as_posix(),
                "bytes": absolute.stat().st_size,
                "sha256": sha256(absolute),
            }
        )

    summary = {
        "schema_version": "carve-paper-ready-evidence-v1",
        "date": "2026-08-27",
        "framework_status": "complete_research_prototype",
        "software_tests": 258,
        "headline": {
            "benchmark": {
                "episodes": benchmark["episode_count"],
                "frozen_success": "180/400",
                "agentic_success": "183/400",
                "agentic_vs_frozen_mcnemar_p": 0.375,
                "control_step_reduction_percent": 7.2,
                "vla_call_reduction_percent": 8.4,
            },
            "efficient_vla": {
                "reference_p95_ms": efficiency["vla_rows"][0]["runtime_p95_ms"],
                "default_p95_ms": efficiency["vla_rows"][3]["runtime_p95_ms"],
                "p95_reduction_percent": 80.64242409466537,
                "fidelity": "45/45",
            },
            "full_agent": {
                "success": "3/3",
                "vla_p95_ms": full_agent["aggregate"]["vla_runtime_latency_p95_ms"],
                "deadline_misses": full_agent["aggregate"]["vla_deadline_misses"],
            },
            "memory_routing": {
                "task_families": 2,
                "memory_routed_official_states": 7,
                "t8_wall_time_reduction_percent": t8_routing[
                    "paired_system_changes"
                ]["wall_time_reduction_percent"],
                "t3_wall_time_reduction_percent": t3_routing["paired_changes"][
                    "memory_4b_vs_direct_9b"
                ]["wall_time_reduction_percent"],
            },
        },
        "claim_boundary": {
            "supported": [
                "complete auditable Agentic execution runtime around frozen PI0.5",
                "bounded recovery, safe stop and reduced wasted VLA execution",
                "fidelity-gated PI0.5 realtime profile on RTX 4090",
                "verified-procedure memory and model routing on two task families",
            ],
            "unsupported": [
                "statistically significant benchmark-wide Agentic success gain",
                "universal quantization acceleration",
                "multi-VLA closed-loop generalization",
                "real-robot transfer",
            ],
        },
        "sources": source_rows,
        "representative_artifacts": artifact_rows,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    (OUTPUT / "representative_artifacts.json").write_text(
        json.dumps(artifact_rows, indent=2, ensure_ascii=True) + "\n"
    )

    artifact_links = "\n".join(
        f"- [{item['id']}]({relative_link(Path(item['path']))}): {item['purpose']}"
        for item in artifact_rows
    )
    readme = f"""# CARVE-VLA Paper-Ready Evidence Package

Generated on 2026-08-27 from accepted machine-readable experiment artifacts.
Run `python scripts/build_current_evidence_package.py` from the repository root
to rebuild it. Original results are referenced in place and are never modified.

## Canonical Tables

- `agentic_benchmark.csv`: 1,200-episode Frozen/Fixed/Agentic comparison.
- `vla_runtime.csv`: five PI0.5 inference profiles.
- `planner_runtime.csv`: BF16/INT8/NF4 Planner profiles.
- `memory_routing.csv`: T8 and T3 memory/model-routing comparisons.
- `summary.json`: headline metrics, source hashes and claim boundaries.
- `representative_artifacts.json`: video/figure metadata and SHA-256 hashes.

## Recommended Result Order

1. Establish the complete Agentic Harness with the 1,200-episode study.
2. State that aggregate success uplift is small and not statistically significant.
3. Show bounded recovery, reduced control steps and reduced VLA calls.
4. Present the admitted realtime VLA profile and quantization boundaries.
5. Present verified memory as a compute-routing mechanism, ending with the
   73.1% T3 wall-time reduction at preserved success.

## Representative Artifacts

{artifact_links}

## Claim Boundary

The package supports a complete research prototype, an auditable execution
Harness, an admitted efficient VLA runtime, and compact memory-routing evidence.
It does not support benchmark-wide statistical superiority, a universal
quantization algorithm, multi-VLA closed-loop generalization or real-robot
transfer.
"""
    (OUTPUT / "README.md").write_text(readme)


if __name__ == "__main__":
    build()

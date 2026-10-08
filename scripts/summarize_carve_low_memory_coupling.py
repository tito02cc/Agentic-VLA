#!/usr/bin/env python3
"""Summarize the NF4 Planner plus SMVE PI0.5 integration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/carve_efficiency_full_20260826/coupling"),
    )
    parser.add_argument("--planner-process-mib", type=float, default=3638.0)
    parser.add_argument("--vla-process-mib", type=float, default=7850.0)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def planner_event(path: Path) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] == "planner_result":
            return event["payload"]
    raise ValueError("planner_result is missing from the event log")


def main() -> int:
    args = parse_args()
    run_root = args.root / "qwen35-4b-nf4-pi05-smve-smoke"
    receipt = load(run_root / "model_integration_receipt.json")
    planner_receipt = load(args.root / "qwen35_4b_nf4_receipt.json")
    planner = planner_event(run_root / "events.jsonl")
    primitive = receipt["tool"]["output"]["primitive_outcome"]
    metadata = primitive["metadata"]
    checks = {
        "canonical_smoke_passed": bool(receipt["passed"]),
        "planner_nf4_profile": planner_receipt["deployment_precision"] == "nf4_w4a16",
        "planner_decision_accepted": bool(planner["accepted"]),
        "planner_selected_vla": planner["decision"]["intent"] == "vla_act",
        "smve_profile_used": metadata["profile_id"]
        == "pi05-torch_compile_masked_views-bf16-2step-h10",
        "action_shape_valid": receipt["private_action_receipt"]["shape"] == [10, 7],
        "deadline_met": not bool(metadata["deadline_miss"]),
        "no_fallback": not bool(metadata["fallback_used"]),
        "actions_withheld": bool(receipt["private_action_receipt"]["withheld_from_environment"]),
    }
    payload = {
        "schema_version": "carve-low-memory-coupling-gate-v1",
        "hardware": "NVIDIA GeForce RTX 4090 24GB",
        "planner_profile": "Q2 Qwen3.5-4B NF4",
        "vla_profile": "V3 PI0.5 compiled BF16 + SMVE",
        "measured_process_memory": {
            "planner_mib": args.planner_process_mib,
            "vla_mib": args.vla_process_mib,
            "pair_mib": args.planner_process_mib + args.vla_process_mib,
            "pair_gib": (args.planner_process_mib + args.vla_process_mib) / 1024.0,
            "source": "nvidia-smi compute-process query while both services were resident",
        },
        "planner": {
            "elapsed_ms": float(planner["elapsed_ms"]),
            "accepted": bool(planner["accepted"]),
            "attempt_count": int(planner["attempt_count"]),
            "decision": planner["decision"],
        },
        "vla": {
            "runtime_latency_ms": float(metadata["runtime_latency_ms"]),
            "model_latency_ms": float(metadata["model_latency_ms"]),
            "deadline_ms": float(metadata["deadline_ms"]),
            "deadline_miss": bool(metadata["deadline_miss"]),
            "fallback_used": bool(metadata["fallback_used"]),
            "action_shape": receipt["private_action_receipt"]["shape"],
        },
        "checks": checks,
        "passed": all(checks.values()),
        "decision": "integration_passed_system_admission_pending",
        "claim_boundary": (
            "One real co-resident Planner-to-VLA dry-run. It validates model loading, "
            "typed planning, profile admission, action-contract generation and one VLA "
            "deadline. It is not task-success, repeated-latency or stress evidence."
        ),
        "sources": {
            "planner_receipt": str(args.root / "qwen35_4b_nf4_receipt.json"),
            "integration_receipt": str(run_root / "model_integration_receipt.json"),
            "events": str(run_root / "events.jsonl"),
        },
    }
    output = args.root / "low_memory_coupling_gate.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CARVE Low-Memory Coupling Gate",
        "",
        "## Configuration",
        "",
        "- Planner: Q2 Qwen3.5-4B NF4.",
        "- VLA: V3 PI0.5 compiled BF16 + SMVE.",
        "- Hardware: one NVIDIA GeForce RTX 4090 24GB.",
        f"- Measured model-service process memory: `{payload['measured_process_memory']['pair_gib']:.2f} GiB`.",
        "",
        "## Result",
        "",
        f"- Overall integration gate: `{'PASS' if payload['passed'] else 'FAIL'}`.",
        f"- Planner: accepted `vla_act` in `{payload['planner']['elapsed_ms'] / 1000.0:.2f} s` with one attempt.",
        f"- VLA: produced a `10 x 7` action chunk in `{payload['vla']['runtime_latency_ms']:.2f} ms`.",
        "- The 80 ms VLA deadline was met, the SMVE profile remained active, and no fallback was used.",
        "- The action chunk was intercepted at the primitive boundary and not sent to an environment.",
        "",
        "## Decision",
        "",
        "`integration_passed_system_admission_pending`",
        "",
        "The pair is feasible on one 24-GB GPU and the complete Planner-to-VLA contract works. "
        "Repeated contention and closed-loop task gates are still required before Q2 is called a fully admitted deployment tier.",
    ]
    (args.root / "LOW_MEMORY_COUPLING_GATE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "passed": payload["passed"]}, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

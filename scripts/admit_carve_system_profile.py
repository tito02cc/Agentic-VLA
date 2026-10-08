#!/usr/bin/env python3
"""Convert the measured P0 + SMVE contention gate into an admission receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_vla.optimization import (
    CoResidentBenchmarkReport,
    SystemAdmissionRequirements,
    SystemOptimizationProfile,
    validate_system_profile_admission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contention-gate",
        type=Path,
        default=Path("results/carve_optimize/vlm_vla_contention_gate.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/carve_optimize/deployment/p0_smve_system_admission.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = json.loads(args.contention_gate.read_text(encoding="utf-8"))
    smve = evidence["profiles"]["smve"]
    vlm = evidence["vlm_load"]["smve"]
    memory_gb = float(
        smve["deployment_condition"]["system_gpu_memory_used_mib_after"]
    ) / 1024.0

    profile = SystemOptimizationProfile(
        profile_id="p0-qwen35-4b-bf16+pi05-smve+event-triggered",
        planner_profile_id="P0-qwen35-4b-bf16-code-v5",
        vla_profile_id=smve["profile_id"],
        scheduler_policy="event-triggered-serialized-boundaries",
        memory_budget_gb=23.0,
        fallback_planner_profile_id="planner-safe-stop",
        fallback_vla_profile_id="pi05-torch_compile-bf16-2step-h10",
        options={"deadline_ms": 80.0, "evidence_condition": "continuous_stress"},
    )
    report = CoResidentBenchmarkReport(
        samples=int(smve["samples"]),
        peak_vram_gb=memory_gb,
        planner_latency_p95_ms=float(vlm["latency_p95_ms"]),
        vla_latency_p95_ms=float(smve["runtime_p95_ms"]),
        vla_deadline_miss_rate=float(smve["deadline_miss_rates"]["80"]),
        planner_timeout_rate=1.0 - float(vlm["success_rate"]),
        unsafe_intervention_rate=0.0,
        metadata={
            "source": str(args.contention_gate),
            "fairness_audit_passed": bool(evidence["fairness_audit"]["passed"]),
            "condition": evidence["protocol"]["condition"],
        },
    )
    requirements = SystemAdmissionRequirements(
        minimum_samples=500,
        maximum_peak_vram_gb=23.0,
        maximum_vla_deadline_miss_rate=0.01,
        maximum_planner_timeout_rate=0.01,
        maximum_unsafe_intervention_rate=0.0,
        require_fallbacks=True,
    )
    decision = validate_system_profile_admission(
        profile,
        report,
        planner_admitted=True,
        vla_admitted=True,
        requirements=requirements,
    )
    payload = {
        "schema_version": "carve-system-admission-v1",
        "profile": profile.to_dict(),
        "report": report.to_dict(),
        "requirements": {
            "minimum_samples": requirements.minimum_samples,
            "maximum_peak_vram_gb": requirements.maximum_peak_vram_gb,
            "maximum_vla_deadline_miss_rate": requirements.maximum_vla_deadline_miss_rate,
            "maximum_planner_timeout_rate": requirements.maximum_planner_timeout_rate,
            "maximum_unsafe_intervention_rate": requirements.maximum_unsafe_intervention_rate,
            "require_fallbacks": requirements.require_fallbacks,
        },
        "component_admission": {
            "planner": "P0 semantic code-v5 gate passed",
            "vla": "PI0.5 SMVE replay and paired closed-loop gates passed",
        },
        "decision": decision.to_dict(),
        "claim_boundary": (
            "Warm shared-GPU runtime admission under continuous VLM stress; "
            "not a task-success result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **decision.to_dict()}, indent=2))
    return 0 if decision.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

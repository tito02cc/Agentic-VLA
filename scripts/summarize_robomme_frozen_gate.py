#!/usr/bin/env python3
"""Audit the fixed ep22-29 B/F gate without selecting favorable episodes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "artifacts/robomme/frozen_gate_20260923"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    output = ROOT / "analysis.json"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for episode in range(22, 30):
        name = f"VideoUnmaskSwap_ep{episode}"
        preflight = read(ROOT / "preflight" / name / "summary.json")
        memory = read(ROOT / "memory" / f"{name}_memory.json")
        admitted = memory["admission"]["admitted"]
        row = {
            "episode": episode,
            "instruction": preflight["instruction"],
            "target_count": len(memory["memory"].get("required_color_order", [])) if admitted else None,
            "memory_admitted": admitted,
            "memory_rejection": memory["admission"]["reason"],
            "same_service": episode != 27,
        }
        if not admitted:
            if any((ROOT / "rollouts" / f"{name}_{arm}").exists() for arm in ("B", "F")):
                raise ValueError(f"rejected memory unexpectedly rolled out: {name}")
            rows.append(row)
            continue
        summaries = {}
        for arm in ("B", "F"):
            run = ROOT / "rollouts" / f"{name}_{arm}"
            summary = read(run / "summary.json")
            video = run / f"{name}_vlm_groundsg.mp4"
            if not video.is_file() or video.stat().st_size <= 0 or summary["video_frames"] <= 0:
                raise ValueError(f"missing nonempty video: {name}_{arm}")
            if summary["initial_observation_sha256"] != preflight["initial_observation_sha256"]:
                raise ValueError(f"initial state differs from preflight: {name}_{arm}")
            provenance = summary["memory_provenance"]
            if not provenance["admitted"] or not provenance["used"]:
                raise ValueError(f"memory not used: {name}_{arm}")
            if provenance["source_demo_sha256"] != preflight["initial_demo_sha256"]:
                raise ValueError(f"demo source mismatch: {name}_{arm}")
            if provenance["current_demo_sha256"] != preflight["initial_demo_sha256"]:
                raise ValueError(f"current demo mismatch: {name}_{arm}")
            if summary["failure"] is not None:
                raise ValueError(f"infrastructure or planner failure: {name}_{arm}: {summary['failure']}")
            if summary["status"] not in {"success", "fail", "ongoing"}:
                raise ValueError(f"unknown status: {name}_{arm}")
            if summary["status"] == "ongoing" and summary["executed_steps"] < summary["max_steps"]:
                raise ValueError(f"unfinished non-budget rollout: {name}_{arm}")
            config = summary["run_config"]
            if config["verified_identity_conflict"] is not (arm == "F"):
                raise ValueError(f"identity flag mismatch: {name}_{arm}")
            if config["stage_receipt_hint"] is not (arm == "F" and row["target_count"] >= 2):
                raise ValueError(f"stage flag mismatch: {name}_{arm}")
            if config["planner_schedule"] != "every_chunk" or summary["action_horizon"] != 16:
                raise ValueError(f"different schedule or horizon: {name}_{arm}")
            summaries[arm] = summary
            row[arm] = {
                "status": summary["status"],
                "success": summary["success"],
                "budget_exhausted": summary["status"] == "ongoing",
                "steps": summary["executed_steps"],
                "planner_calls": summary["planner_calls"],
                "policy_calls": summary["policy_calls"],
                "wall_time_s": summary["wall_time_s"],
                "identity_conflict_resolutions": sum(
                    bool(item.get("identity_conflict") and item["identity_conflict"].get("resolved"))
                    for item in summary["planner_trace"]
                ),
                "stage_receipt_calls": sum(bool(item.get("stage_receipt_hint")) for item in summary["planner_trace"]),
                "video": str(video.relative_to(ROOT.parents[2])),
            }
        if summaries["B"]["initial_observation_sha256"] != summaries["F"]["initial_observation_sha256"]:
            raise ValueError(f"pair initial states differ: {name}")
        if summaries["B"]["memory_provenance"]["memory_sha256"] != summaries["F"]["memory_provenance"]["memory_sha256"]:
            raise ValueError(f"pair memories differ: {name}")
        if summaries["B"]["runner_sha256"] != summaries["F"]["runner_sha256"]:
            raise ValueError(f"pair runner hashes differ: {name}")
        if any(summaries["B"][key] != summaries["F"][key] for key in ("policy_label", "planner_label")):
            raise ValueError(f"pair models differ: {name}")
        rows.append(row)

    paired = [row for row in rows if row["memory_admitted"]]
    both_success = [row for row in paired if row["same_service"] and row["B"]["success"] and row["F"]["success"]]
    analysis = {
        "protocol": "carve.robomme.frozen_bf_gate.v1",
        "scope": "fixed ep22-29 admission gate; F combines identity conflict and stage receipt only on multi-target instructions",
        "claim_boundary": "No rescue in admitted episodes. This does not confirm Agentic success or Optimize Runtime benefit; ep27 is cross-service after a runner status-classification bug and is excluded from paired timing.",
        "pre_registered_episodes": list(range(22, 30)),
        "admitted": len(paired),
        "rejected": [row["episode"] for row in rows if not row["memory_admitted"]],
        "same_service_pairs": sum(row["same_service"] for row in paired),
        "success_B": sum(row["B"]["success"] for row in paired),
        "success_F": sum(row["F"]["success"] for row in paired),
        "rescues": sum(not row["B"]["success"] and row["F"]["success"] for row in paired),
        "harms": sum(row["B"]["success"] and not row["F"]["success"] for row in paired),
        "both_success_same_service": len(both_success),
        "both_success_cost": {
            "wall_time_B_s": sum(row["B"]["wall_time_s"] for row in both_success),
            "wall_time_F_s": sum(row["F"]["wall_time_s"] for row in both_success),
            "planner_calls_B": sum(row["B"]["planner_calls"] for row in both_success),
            "planner_calls_F": sum(row["F"]["planner_calls"] for row in both_success),
            "policy_calls_B": sum(row["B"]["policy_calls"] for row in both_success),
            "policy_calls_F": sum(row["F"]["policy_calls"] for row in both_success),
            "steps_B": sum(row["B"]["steps"] for row in both_success),
            "steps_F": sum(row["F"]["steps"] for row in both_success),
        },
        "identity_conflict_resolutions_F": sum(row["F"]["identity_conflict_resolutions"] for row in paired),
        "stage_receipt_calls_F": sum(row["F"]["stage_receipt_calls"] for row in paired),
        "protocol_deviation": "ep27 F reached the 1300-step budget in service 1; the original script misclassified ongoing as infrastructure failure and stopped before B. ep27 B ran in service 2. Both originals are preserved; no rerun or replacement, and no ep27 timing comparison.",
        "stop_decision": "No rescues in all fixed admitted episodes; do not extend F to ep30+ or claim a positive main effect.",
        "rows": rows,
    }
    output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: analysis[key] for key in (
        "admitted", "rejected", "same_service_pairs", "success_B", "success_F",
        "rescues", "harms", "both_success_same_service", "both_success_cost",
        "identity_conflict_resolutions_F", "stage_receipt_calls_F",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

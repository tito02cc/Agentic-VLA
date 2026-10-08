#!/usr/bin/env python3
"""Audit the fixed five-episode B/C runtime comparison without selecting successes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "artifacts/robomme/unmaskswap_fixed_five_20260923"
ORDER = ("15:B,C", "16:C,B", "17:B,C", "18:C,B", "19:B,C")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"empty rollout video: {path}")
    return duration


def summarize(root: Path, source_root: Path = SOURCE_ROOT) -> dict:
    warmup = read_json(root / "rollouts/VideoUnmaskSwap_ep15_warmup/summary.json")
    previous_batch = read_json(source_root / "analysis.json")
    if warmup["failure"] is not None or warmup["policy_calls"] < 1 or warmup["planner_calls"] < 1:
        raise ValueError("shared-service warmup did not exercise both models")
    rows = []
    runner_sha = None
    labels = None
    for episode in range(15, 20):
        name = f"VideoUnmaskSwap_ep{episode}"
        memory_path = source_root / "memory" / f"{name}_memory.json"
        memory = read_json(memory_path)
        preflight = read_json(source_root / "preflight" / name / "summary.json")
        if memory["source_demo_sha256"] != preflight["initial_demo_sha256"]:
            raise ValueError(f"ep{episode}: compiled memory source mismatch")
        arms = {}
        for arm in ("B", "C"):
            directory = root / "rollouts" / f"{name}_{arm}"
            data = read_json(directory / "summary.json")
            config = data["run_config"]
            provenance = data.get("memory_provenance") or {}
            if data["task"] != "VideoUnmaskSwap" or data["episode"] != episode:
                raise ValueError(f"ep{episode} {arm}: task binding mismatch")
            if data["instruction"] != preflight["instruction"]:
                raise ValueError(f"ep{episode} {arm}: instruction mismatch")
            if data["initial_observation_sha256"] != preflight["initial_observation_sha256"]:
                raise ValueError(f"ep{episode} {arm}: initial observation mismatch")
            if not provenance.get("admitted") or not provenance.get("used"):
                raise ValueError(f"ep{episode} {arm}: memory was not admitted and used")
            if provenance.get("source_demo_sha256") != preflight["initial_demo_sha256"]:
                raise ValueError(f"ep{episode} {arm}: memory source mismatch")
            if provenance.get("current_demo_sha256") != preflight["initial_demo_sha256"]:
                raise ValueError(f"ep{episode} {arm}: live demo mismatch")
            if provenance.get("memory_sha256") != hashlib.sha256(memory_path.read_bytes()).hexdigest():
                raise ValueError(f"ep{episode} {arm}: memory bytes changed")
            if Path(config["task_memory_file"]).name != memory_path.name:
                raise ValueError(f"ep{episode} {arm}: wrong memory file")
            if (config["planner_schedule"], config["verified_point_reuse"]) != (
                ("every_chunk", False) if arm == "B" else ("selective", True)
            ):
                raise ValueError(f"ep{episode} {arm}: scheduler config mismatch")
            if data["success"] is not (data["status"] == "success"):
                raise ValueError(f"ep{episode} {arm}: success/status mismatch")
            if data["planner_calls"] != data["timing"]["planner_requests"]["count"]:
                raise ValueError(f"ep{episode} {arm}: planner timing mismatch")
            if data["policy_calls"] != data["timing"]["policy_requests"]["count"]:
                raise ValueError(f"ep{episode} {arm}: VLA timing mismatch")
            video = directory / f"{name}_vlm_groundsg.mp4"
            duration = video_duration(video)
            current_labels = (data["policy_label"], data["planner_label"])
            if runner_sha is None:
                runner_sha = data["runner_sha256"]
                labels = current_labels
            if data["runner_sha256"] != runner_sha or current_labels != labels:
                raise ValueError(f"ep{episode} {arm}: runner/model mismatch")
            arms[arm] = (data, config, {
                "success": data["success"], "status": data["status"],
                "failure": data["failure"], "steps": data["executed_steps"],
                "planner_calls": data["planner_calls"], "vla_calls": data["policy_calls"],
                "reuse_hits": data["planner_reuse_hits"],
                "rollout_s": data["timing"]["rollout_s"],
                "wall_s": data["wall_time_s"],
                "planner_request_s": data["timing"]["planner_requests"]["total_ms"] / 1000.0,
                "vla_request_s": data["timing"]["policy_requests"]["total_ms"] / 1000.0,
                "revalidation_s": data["timing"]["point_revalidation"]["total_ms"] / 1000.0,
                "revalidations": data["timing"]["point_revalidation"]["count"],
                "video": str(video), "video_duration_s": duration,
            })
        b, b_config, b_row = arms["B"]
        c, c_config, c_row = arms["C"]
        ignore = {"output", "planner_schedule", "verified_point_reuse"}
        if {k: v for k, v in b_config.items() if k not in ignore} != {
            k: v for k, v in c_config.items() if k not in ignore
        }:
            raise ValueError(f"ep{episode}: B/C config differs beyond runtime schedule")
        previous_b = read_json(source_root / "rollouts" / f"{name}_B" / "summary.json")
        if {k: v for k, v in previous_b["run_config"].items() if k != "output"} != {
            k: v for k, v in b_config.items() if k != "output"
        }:
            raise ValueError(f"ep{episode}: prior/current B config changed")
        old_first = previous_b["policy_request_audit"][0]
        b_first = b["policy_request_audit"][0]
        c_first = c["policy_request_audit"][0]
        rows.append({
            "episode": episode, "instruction": b["instruction"],
            "memory_compile_s": memory["compile_time_s"],
            "B": b_row, "C": c_row,
            "previous_B_status": previous_b["status"],
            "previous_B_first_input_same": old_first["input_sha256"] == b_first["input_sha256"],
            "previous_B_first_action_same": old_first["actions_sha256"] == b_first["actions_sha256"],
            "same_service_B_C_first_input_same": b_first["input_sha256"] == c_first["input_sha256"],
            "same_service_B_C_first_action_same": b_first["actions_sha256"] == c_first["actions_sha256"],
            "C_revalidation_reasons": [
                (item.get("reuse_validation") or {}).get("reason")
                for item in c["schedule_trace"] if item.get("reuse_validation") is not None
            ],
        })
    both_success = [row for row in rows if row["B"]["success"] and row["C"]["success"]]
    rescues = sum(not r["B"]["success"] and r["C"]["success"] for r in rows)
    harms = sum(r["B"]["success"] and not r["C"]["success"] for r in rows)
    paired_b_s = sum(r["B"]["rollout_s"] for r in both_success)
    paired_c_s = sum(r["C"]["rollout_s"] for r in both_success)
    paired_b_wall_s = sum(r["B"]["wall_s"] for r in both_success)
    paired_c_wall_s = sum(r["C"]["wall_s"] for r in both_success)
    planner_saved = sum(r["B"]["planner_calls"] - r["C"]["planner_calls"] for r in rows)
    cost_signal = harms == 0 and both_success and planner_saved > 0 and paired_c_wall_s < paired_b_wall_s
    if not cost_signal:
        status = "stop_joint_quality_preserving_speedup_claim"
    elif previous_batch["success_B"] != sum(r["B"]["success"] for r in rows):
        status = "development_cost_signal_baseline_unstable_hold_confirmation"
    else:
        status = "development_gate_passed_not_confirmatory"
    gpu_csv = root / "gpu_usage.csv"
    gpu_mib = []
    if gpu_csv.exists():
        for line in gpu_csv.read_text(encoding="utf-8").splitlines():
            fields = line.split(",")
            if len(fields) >= 2:
                try:
                    gpu_mib.append(float(fields[1].strip()))
                except ValueError:
                    pass
    return {
        "protocol": "carve.robomme.unmaskswap_runtime_bc.dev.v1",
        "scope": "fixed_seen_development_episodes_15_to_19_not_held_out",
        "claim_boundary": "Same-service development screen only; not RA-L confirmation or hard realtime.",
        "order": list(ORDER), "runner_sha256": runner_sha,
        "policy_label": labels[0], "planner_label": labels[1],
        "warmup_policy_first_ms": warmup["timing"]["policy_requests"]["first_ms"],
        "warmup_planner_first_ms": warmup["timing"]["planner_requests"]["first_ms"],
        "success_B": sum(r["B"]["success"] for r in rows),
        "success_C": sum(r["C"]["success"] for r in rows),
        "previous_batch_success_B": previous_batch["success_B"],
        "previous_B_first_input_same_count": sum(r["previous_B_first_input_same"] for r in rows),
        "previous_B_first_action_same_count": sum(r["previous_B_first_action_same"] for r in rows),
        "same_service_B_C_first_input_same_count": sum(r["same_service_B_C_first_input_same"] for r in rows),
        "same_service_B_C_first_action_same_count": sum(r["same_service_B_C_first_action_same"] for r in rows),
        "rescues": rescues, "harms": harms, "both_success": len(both_success),
        "planner_calls_saved_all": planner_saved,
        "paired_success_planner_calls_B": sum(r["B"]["planner_calls"] for r in both_success),
        "paired_success_planner_calls_C": sum(r["C"]["planner_calls"] for r in both_success),
        "paired_success_vla_calls_B": sum(r["B"]["vla_calls"] for r in both_success),
        "paired_success_vla_calls_C": sum(r["C"]["vla_calls"] for r in both_success),
        "paired_success_rollout_s_B": paired_b_s,
        "paired_success_rollout_s_C": paired_c_s,
        "paired_success_rollout_s_change_fraction": (
            paired_c_s / paired_b_s - 1 if paired_b_s else None
        ),
        "paired_success_wall_s_B": paired_b_wall_s,
        "paired_success_wall_s_C": paired_c_wall_s,
        "paired_success_wall_s_change_fraction": (
            paired_c_wall_s / paired_b_wall_s - 1 if paired_b_wall_s else None
        ),
        "C_revalidations": sum(r["C"]["revalidations"] for r in rows),
        "C_revalidation_s_total": sum(r["C"]["revalidation_s"] for r in rows),
        "offline_memory_compile_s_for_five_episodes": sum(r["memory_compile_s"] for r in rows),
        "gpu_used_mib_max_shared_services": max(gpu_mib) if gpu_mib else None,
        "gpu_memory_scope": "whole-device sample with both model services; not per-arm peak",
        "developer_gate": status,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    destination = args.root / "analysis.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

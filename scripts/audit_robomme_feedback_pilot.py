#!/usr/bin/env python3
"""Audit saved pilot evidence and summarize descriptive, not inferential, metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {"count": len(values), "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def audit(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if hashlib.sha256((root / "runner_snapshot.py").read_bytes()).hexdigest() != manifest["runner_sha256"]:
        raise ValueError("runner snapshot hash mismatch")
    for path, expected in manifest.get("source_sha256", {}).items():
        if hashlib.sha256((root / "sources" / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source snapshot mismatch: {path}")
    rows = []
    hashes = defaultdict(set)
    for path in sorted(root.glob("ep*/summary.json")):
        result = json.loads(path.read_text())
        if result["runner_sha256"] != manifest["runner_sha256"]:
            raise ValueError(f"runtime source mismatch: {path}")
        if result["success"] != (result["status"] == "success"):
            raise ValueError(f"official result mismatch: {path}")
        video = path.parent / Path(result["video_path"]).name
        metadata = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=nb_frames,r_frame_rate,width,height", "-of", "json", str(video),
        ]))["streams"][0]
        if int(metadata["nb_frames"]) != result["video_frames"] or result["video_frames"] != (
            result["initial_memory_frames"] + result["executed_steps"]
        ):
            raise ValueError(f"video frame count mismatch: {video}")
        receipts = result["execution_feedback_trace"]
        end = 0
        for index, receipt in enumerate(receipts, start=1):
            if receipt["chunk_id"] != index or receipt["start_step"] != end or receipt["semantic_success_verified"]:
                raise ValueError(f"invalid execution receipt: {path}")
            end += receipt["executed_steps"]
            if receipt["end_step"] != end:
                raise ValueError(f"discontinuous execution receipt: {path}")
        if result["execution_feedback_mode"] == "execution_chunks" and end != result["executed_steps"]:
            raise ValueError(f"execution receipt coverage mismatch: {path}")
        trace = result["planner_trace"]
        requests = result.get("planner_request_count")
        if requests is None and not result["failure"]:
            requests = sum(1 + item["repair_attempts"] for item in trace)
        request_audit = result.get("planner_request_audit", [])
        usages = [item.get("usage") for item in request_audit]
        have_usage = bool(usages) and all(
            isinstance(item, dict) and isinstance(item.get("prompt_tokens"), int) for item in usages
        )
        hashes[result["episode"]].add(result["initial_observation_sha256"])
        run = path.parent.name
        condition = manifest["cases"][run]["condition"]
        rows.append({
            "run": run, "condition": condition, "episode": result["episode"],
            "status": result["status"], "success": result["success"], "failure": result["failure"],
            "steps": result["executed_steps"], "policy_calls": result["policy_calls"],
            "planner_requests": requests, "rollout_time_s": result["rollout_time_s"],
            "warm_planner_latency_ms": latency_stats([item["latency_ms"] for item in trace[1:]]),
            "first_planner_latency_ms": trace[0]["latency_ms"] if trace else None,
            "prompt_tokens": sum(item["prompt_tokens"] for item in usages) if have_usage else None,
            "demo_video_requests": sum(item["demo_video_sent"] for item in request_audit) if request_audit else None,
            "warm_prompt_tokens_mean": float(np.mean([item["prompt_tokens"] for item in usages[1:]]))
                if have_usage and len(usages) > 1 else None,
            "forced_fallbacks": sum(bool(item["monitor_fallback"]) for item in trace),
            "applied_grounding": sum(bool(item["tool_grounding"]) for item in trace),
            "question_echo_decisions": sum(item["grounded_subgoal"] ==
                "What's the next grounded language subgoal based on current observation?"
                for item in trace),
            "memory_provenance": result.get("memory_provenance"),
            "source": str(path.relative_to(root)), "video": str(video.relative_to(root)),
            "video_metadata": metadata,
        })
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("initial observations differ within a paired episode")
    if summary["complete"] and {row["run"] for row in rows} != set(manifest["commands"]):
        raise ValueError("completed batch does not cover the registered runs")
    totals = {}
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        totals[condition] = {
            "successes": sum(row["success"] for row in group), "episodes": len(group),
            "model_failure_episodes": sum(row["failure"] is not None for row in group),
            "question_echo_decisions": sum(row["question_echo_decisions"] for row in group),
            "steps": sum(row["steps"] for row in group),
            "planner_requests": sum(row["planner_requests"] for row in group)
                if all(row["planner_requests"] is not None for row in group) else None,
        }
    return {"protocol": "robomme.pilot.audit.v1", "complete": summary["complete"],
            "batch_failure": summary["failure"], "expected_runs": len(manifest["commands"]),
            "audited_runs": len(rows), "rows": rows, "totals": totals,
            "claim_boundary": manifest["claim_boundary"],
            "audit_boundary": "Question echoes are checked retrospectively; this does not "
                "change original success labels or imply the current skill guard ran in this batch."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    (args.root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": result["complete"], "audited_runs": result["audited_runs"],
                      "totals": result["totals"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only audit of real recorded joint tracking at exactly bound boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from agentic_vla.benchmarks.robodojo_sorting import joint_target_feedback


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def audit_run(run_dir):
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    run_id = summary["run_id"]
    session = run_dir / "sessions" / run_id
    event_path, transcript_path = session / "events.jsonl", session / "transcript.jsonl"
    trace_path = run_dir / "policy/policy_trace.jsonl"
    inputs = {str(p): sha256(p) for p in (summary_path, event_path, transcript_path, trace_path)}
    events = read_rows(event_path)
    traces = read_rows(trace_path)
    chunks = [e for e in events if e["event_type"] == "pi05_chunk_executed"]
    if len(traces) != len(chunks) or len(chunks) != summary["execution_summary"]["vla_calls"]:
        raise ValueError("unmatched inference and execution counts")
    if len({t["generation"] for t in traces}) != 1:
        raise ValueError("audit requires one reset generation")
    observed = {}
    for row in read_rows(transcript_path):
        if row["role"] != "user":
            continue
        value = json.loads(row["content"])
        if "robot_state" not in value:
            continue
        step = value["timestep"]
        if row["metadata"]["episode_id"] != run_id or row["metadata"]["timestep"] != step:
            raise ValueError("request episode/timestep mismatch")
        state = np.asarray(value["robot_state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("invalid public state")
        if step in observed and not np.array_equal(observed[step], state):
            raise ValueError("contradictory observations at one control step")
        observed[step] = state

    rows, skipped = [], []
    for i, (trace, event) in enumerate(zip(traces, chunks)):
        item = event["payload"]
        if trace["call"] != i + 1:
            raise ValueError("nonconsecutive policy call IDs")
        actions_path = (run_dir / "policy" / trace["actions_file"]).resolve()
        actions_path.relative_to((run_dir / "policy").resolve())
        inputs[str(actions_path)] = sha256(actions_path)
        actions = np.load(actions_path, allow_pickle=False)
        if (actions.shape != (50, 14) or not np.isfinite(actions).all()
                or array_hash(actions) != trace["actions_sha256"]):
            raise ValueError("invalid or changed action block")
        count = item["executed_actions"]
        if (type(count) is not int or not 1 <= count <= 50
                or item["ended_timestep"] - item["started_timestep"] != count
                or item["generated_actions"] != 50 or item["discarded_actions"] != 50 - count):
            raise ValueError("execution interval does not match action prefix")
        end = item["ended_timestep"]
        reason = None
        if end not in observed:
            reason = "no_recorded_proprioception_at_exact_endpoint"
        elif i + 1 >= len(chunks) or chunks[i + 1]["payload"]["started_timestep"] != end:
            reason = "intervening_tool_or_no_next_inference_boundary"
        else:
            fingerprint = traces[i + 1]["input_audit"]["state"]
            if (fingerprint["shape"] != [14] or fingerprint["dtype"] != "float32"
                    or array_hash(observed[end]) != fingerprint["sha256"]):
                raise ValueError("Planner state does not match next inference input")
        if reason:
            skipped.append({"call": trace["call"], "ended_timestep": end, "reason": reason})
            continue
        rows.append({
            "call": trace["call"], "ended_timestep": end, "execution_event_sequence": event["sequence"],
            "binding": "same-step Planner proprioception matches next VLA input SHA256; no intervening action",
            "final_command": actions[count - 1].tolist(), "observed_state": observed[end].tolist(),
            "feedback": joint_target_feedback(actions[count - 1], observed[end]),
        })
    drift = [name for name, fingerprint in inputs.items() if sha256(Path(name)) != fingerprint]
    if drift:
        raise ValueError(f"source drift: {drift}")
    return {
        "schema": "robodojo.recorded_joint_feedback.v1", "source_run": run_id,
        "scope": "retrospective descriptive audit; not a new rollout, full-trajectory tracking test, or efficacy comparison",
        "new_robot_actions": 0, "source_sha256": inputs, "source_drift": drift,
        "total_action_blocks": len(chunks), "available_planner_state_snapshots": len(observed),
        "bound_endpoints": len(rows), "rows": rows, "skipped": skipped,
        "max_arm_abs_error_rad": max((r["feedback"][s]["arm_max_abs_error_rad"]
                                      for r in rows for s in ("left", "right")), default=None),
        "max_gripper_abs_error_normalized": max((r["feedback"][s]["gripper_abs_error_normalized"]
                                                 for r in rows for s in ("left", "right")), default=None),
        "semantic_recovery_confirmed": False,
        "limitations": ["Sparse endpoints, not all 1000 control steps; no contact/force measurement.",
                        "Low joint error cannot establish correct object identity, grasp, placement, or safety.",
                        "Observations are from an existing development episode; no independent holdout."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite an earlier audit")
    report = audit_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("source_run", "total_action_blocks", "bound_endpoints",
        "max_arm_abs_error_rad", "max_gripper_abs_error_normalized", "source_drift")}, indent=2))


if __name__ == "__main__":
    main()

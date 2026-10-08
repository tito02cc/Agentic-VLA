#!/usr/bin/env python3
"""Read-only evidence audit for the controlled real feedback-tool smoke."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text())
    launch = json.loads((run_dir / "launch.json").read_text())
    execution = summary["execution_summary"]
    checks = {}
    checks["controlled_scope"] = (
        summary["completed"] and summary["execution_mode"] == "feedback_tool_smoke"
        and execution["autonomous_planner_used"] is False and execution["official_success"] is None)
    checks["source_unchanged"] = not summary["source_drift"] and all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in launch["source_sha256"].items())
    session = Path(summary["execution_summary_path"]).parent
    events = rows(session / "events.jsonl")
    chunks = [row["payload"] for row in events if row["event_type"] == "pi05_chunk_executed"]
    critics = [row["payload"] for row in events if row["event_type"] == "visual_critic"]
    reused = [row for row in events if row["event_type"] == "visual_critic_reused"]
    postchecks = [row["payload"] for row in events if row["event_type"] == "feedback_refresh_postcheck"]
    traces = rows(run_dir / "policy/policy_trace.jsonl")
    checks["two_real_policy_blocks"] = len(chunks) == len(traces) == execution["vla_calls"] == 2
    if not checks["two_real_policy_blocks"]:
        raise ValueError("controlled smoke did not reach both policy calls; inspect incomplete run")
    checks["prefix_then_fresh_full_chunk"] = (
        chunks[0]["execution_prefix"] == 10 and 0 < chunks[0]["executed_actions"] <= 10
        and chunks[1]["execution_prefix"] == 50
        and chunks[0]["ended_timestep"] == chunks[1]["started_timestep"]
        and chunks[0]["output_observation_sha256"] == chunks[1]["input_observation_sha256"])
    valid_actions = True
    for i, (trace, chunk) in enumerate(zip(traces, chunks), 1):
        action_path = (run_dir / "policy" / trace["actions_file"]).resolve()
        action_path.relative_to(run_dir / "policy")
        actions = np.load(action_path, allow_pickle=False)
        count = chunk["executed_actions"]
        valid_actions &= (
            trace["call"] == i and trace["generation"] == 1
            and trace["optimized"] is False and actions.shape == (50, 14)
            and np.isfinite(actions).all()
            and hashlib.sha256(actions.tobytes()).hexdigest() == trace["actions_sha256"]
            and 0 < count <= chunk["execution_prefix"]
            and chunk["generated_actions"] == count + chunk["discarded_actions"] == 50
            and np.array_equal(actions[count - 1], chunk["final_command"])
            and chunk["joint_target_feedback"]["semantic_outcome"] == "not_verified"
            and not chunk["instruction_changed"])
    checks["action_integrity_and_conservation"] = bool(valid_actions) and (
        execution["generated_vla_steps"] == execution["vla_control_steps"] + execution["discarded_vla_steps"] == 100)
    observed = np.asarray(chunks[0]["final_observed_state"], dtype=np.float32)
    checks["fresh_state_reaches_real_model"] = (
        hashlib.sha256(observed.tobytes()).hexdigest() == traces[1]["input_audit"]["state"]["sha256"])
    checks["two_real_critic_requests_one_reuse"] = (
        len(critics) == execution["critic"]["calls"] == 2
        and len(reused) == execution["visual_check_cache_hits"] == 1
        and all(c["error"] is None and c["authority"] == "advisory" for c in critics)
        and critics[0]["expected_outcome"] == critics[1]["expected_outcome"] == traces[0]["instruction"]
        and critics[0]["timestep"] == chunks[0]["started_timestep"]
        and critics[1]["timestep"] == chunks[0]["ended_timestep"])
    checks["postcheck_not_recovery_claim"] = (
        len(postchecks) == 1 and postchecks[0]["postcheck_state"] == "checked"
        and postchecks[0]["semantic_recovery_confirmed"] is False)
    receipts = execution["controlled_tool_receipts"]
    checks["execution_only_memory"] = len(receipts) == 2 and all(
        "postcheck" not in r["metadata"] and r["metadata"]["semantic_outcome"] == "not_verified" for r in receipts)
    videos = []
    for name in summary["videos"]:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,duration", "-of", "json", name],
            check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)["streams"][0]
        videos.append({"path": name, **stream})
    checks["three_nonempty_videos"] = len(videos) == 3 and all(int(v["nb_frames"]) > 1 for v in videos)
    return {"passed": all(checks.values()), "checks": checks, "videos": videos,
            "run_dir": str(run_dir), "control_steps": execution["control_steps"],
            "scope": "controlled integration only; no autonomy, recovery, speedup, or success-rate conclusion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("controlled integration evidence check failed")


if __name__ == "__main__":
    main()

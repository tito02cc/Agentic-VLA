#!/usr/bin/env python3
"""Audit a completed same-executor pair; never infer causal gains from two runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def array_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"equal": bool(np.array_equal(a, b)), "mean_abs_difference": float(diff.mean()),
            "max_abs_difference": float(diff.max()), "different_fraction": float(np.mean(a != b))}


def load_run(path):
    summary = json.loads((path / "summary.json").read_text())
    if not summary["completed"] or summary["source_drift"]:
        raise ValueError(f"incomplete or changed-source run: {path}")
    rows = [json.loads(line) for line in (path / "policy/policy_trace.jsonl").read_text().splitlines()]
    if len(rows) != summary["execution_summary"]["vla_calls"]:
        raise ValueError("policy and executor call counts disagree")
    return summary, rows


def compare(first, second):
    a, ar = load_run(first)
    b, br = load_run(second)
    source_a = json.loads((first / "launch.json").read_text())["source_sha256"]
    source_b = json.loads((second / "launch.json").read_text())["source_sha256"]
    if source_a != source_b:
        raise ValueError("pair used different source files")
    if a["agent_enabled"] or not b["agent_enabled"]:
        raise ValueError("expected Agent-off then Agent-on")
    paths = [root / "policy" / rows[0]["input_audit"]["input_file"]
             for root, rows in ((first, ar), (second, br))]
    with np.load(paths[0], allow_pickle=False) as x, np.load(paths[1], allow_pickle=False) as y:
        initial = {k: array_difference(x[k], y[k]) for k in
                   ("state", "cam_high", "cam_left_wrist", "cam_right_wrist")}
        initial["instruction_equal"] = str(x["instruction"].item()) == str(y["instruction"].item())
    calls = []
    for x, y in zip(ar, br):
        xa = np.load(first / "policy" / x["actions_file"], allow_pickle=False)
        ya = np.load(second / "policy" / y["actions_file"], allow_pickle=False)
        same_input = (x["input_audit"]["state"] == y["input_audit"]["state"]
                      and x["input_audit"]["images"] == y["input_audit"]["images"]
                      and x["instruction"] == y["instruction"])
        calls.append({"call": x["call"], "inputs_equal": same_input,
                      "rng_before_equal": x["input_audit"]["rng_before"] == y["input_audit"]["rng_before"],
                      "actions": array_difference(xa, ya)})
    def metrics(summary, rows):
        ex = summary["execution_summary"]
        return {"official_score": summary["official_result"]["score"],
                "success_rate": summary["official_result"]["success_rate"],
                "control_steps": ex["control_steps"], "vla_calls": ex["vla_calls"],
                "observation_reads": ex["observation_reads"], "interrupted_chunks": ex["interrupted_chunks"],
                "skill_calls": ex["skill_calls"], "stopped": ex["stopped"],
                "episode_wall_seconds": ex["wall_seconds"],
                "warm_policy_median_ms": float(np.median([r["inference_wall_ms"] for r in rows[1:]]))
                if len(rows) > 1 else None,
                "critic": ex["critic"], "videos": summary["videos"]}
    return {"agent_off": metrics(a, ar), "agent_on": metrics(b, br),
            "initial_observations": initial, "aligned_calls": calls,
            "unpaired_call_counts": [len(ar) - len(calls), len(br) - len(calls)],
            "interpretation": "one episode per condition; not a success-rate or causal improvement estimate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-off", type=Path, required=True)
    parser.add_argument("--agent-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()
    report = compare(args.agent_off, args.agent_on)
    if args.replay_dir:
        a = np.load(args.replay_dir / "adapter_actions.npz", allow_pickle=False)["native"]
        b = np.load(args.replay_dir / "second_input/adapter_actions.npz", allow_pickle=False)["native"]
        live_a = load_run(args.agent_off)[1][0]
        live_b = load_run(args.agent_on)[1][0]
        report["recorded_input_replay"] = {
            "same_process_input_difference": array_difference(a, b),
            "off_replay_vs_live": array_difference(a, np.load(args.agent_off / "policy" / live_a["actions_file"])),
            "on_replay_vs_live": array_difference(b, np.load(args.agent_on / "policy" / live_b["actions_file"])),
            "scope": "same-process initial-input sensitivity; not closed-loop attribution"}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

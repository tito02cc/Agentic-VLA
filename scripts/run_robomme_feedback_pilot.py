#!/usr/bin/env python3
"""Run registered development comparisons with external model services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


CONDITIONS = (
    ("raw_every", "raw", "legacy_predictions", "every_chunk"),
    ("legacy_every", "carve", "legacy_predictions", "every_chunk"),
    ("execution_every", "carve", "execution_chunks", "every_chunk"),
    ("execution_selective", "carve", "execution_chunks", "selective"),
)
AUTHORITY_CONDITIONS = (
    ("observe_every", "carve", "execution_chunks", "every_chunk"),
    ("observe_selective", "carve", "execution_chunks", "selective"),
)
REGRESSION_CONDITIONS = (CONDITIONS[0], AUTHORITY_CONDITIONS[0])


def regression_cases(episodes: list[int]) -> list[tuple]:
    if len(episodes) != 5 or len(set(episodes)) != 5 or any(ep <= 0 or ep >= 50 for ep in episodes):
        raise ValueError("regression requires five distinct official episodes in 1..49")
    cases = []
    for index, episode in enumerate(episodes):
        conditions = REGRESSION_CONDITIONS if index % 2 == 0 else REGRESSION_CONDITIONS[::-1]
        for condition in conditions:
            cases.append((episode, *condition))
    return cases


def memory_cases(episodes: list[int] | None = None) -> list[tuple]:
    episodes = list(range(5)) if episodes is None else episodes
    if len(episodes) != 5 or len(set(episodes)) != 5 or any(ep < 0 or ep >= 50 for ep in episodes):
        raise ValueError("memory comparison requires five distinct official episodes in 0..49")
    cases = []
    for index, episode in enumerate(episodes):
        names = ("observe_memory_off", "observe_memory_on")
        if index % 2:
            names = names[::-1]
        cases.extend((episode, name, "carve", "execution_chunks", "every_chunk") for name in names)
    return cases


def planner_protocol_failure(summary: dict) -> bool:
    message = summary.get("failure") or ""
    return any(fragment in message for fragment in (
        "visually targeted action contains no grounding point",
        "grounded Planner output contains a malformed point token",
        "grounded Planner output contains an out-of-range point",
        "grounded Planner output is empty",
        "grounded Planner output is unreasonably long",
        "grounded Planner returned no text",
        "unsupported VideoUnmaskSwap action skill",
        "unsupported VideoRepick action skill",
    ))


def repick_cases() -> list[tuple]:
    conditions = [CONDITIONS[0], ("observe_memory_off", "carve", "execution_chunks", "every_chunk"),
                  ("observe_memory_on", "carve", "execution_chunks", "every_chunk")]
    return [(episode, *condition) for episode, ordered in ((11, conditions), (12, conditions[::-1]))
            for condition in ordered]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="PickXtimes", choices=("PickXtimes", "VideoUnmaskSwap", "VideoRepick", "RouteStick"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--phase", choices=("feedback", "authority", "regression", "context", "grounding", "confirmation", "memory", "memory_runtime", "task_admission", "repick_memory", "repick_diagnostic", "closeout"), default="feedback")
    parser.add_argument("--study-spec", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicitly run a registered closeout batch; default only registers it.")
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--memory-root", type=Path, default=Path("results/robomme_unmask_swap_sam2_memory_20260831"))
    parser.add_argument("--memory-preflight-root", type=Path, default=Path("artifacts/robomme/identity_memory_preflight_20260922"))
    args = parser.parse_args()
    if args.phase == "closeout":
        if args.study_spec is None:
            parser.error("closeout requires --study-spec")
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from agentic_vla.benchmarks.robomme_study import register_study, execute_study, verify_frozen_inputs
        manifest_path = args.output / "manifest.json"
        if args.execute and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest["spec"] != json.loads(args.study_spec.read_text()):
                raise ValueError("study spec changed after registration; register a fresh batch")
            verify_frozen_inputs(root, manifest)
        else:
            manifest = register_study(root, args.study_spec.resolve(), args.output.resolve())
        print(json.dumps({"registered_for_execution": manifest["registered_for_execution"],
                          "blocking_inputs": manifest["blocking_inputs"], "cases": len(manifest["cases"])}))
        if args.execute:
            execute_study(root, args.output.resolve(), manifest)
        return 0
    if args.study_spec is not None or args.execute:
        parser.error("--study-spec and --execute are only available for closeout")
    if (args.phase in {"memory", "memory_runtime"}) != (args.task == "VideoUnmaskSwap"):
        parser.error("VideoUnmaskSwap requires the registered memory phase")
    if (args.phase in {"task_admission", "repick_memory", "repick_diagnostic"}) != (args.task in {"VideoRepick", "RouteStick"}):
        parser.error("VideoRepick/RouteStick require an admitted phase")
    if args.phase in {"repick_memory", "repick_diagnostic"} and (args.task != "VideoRepick" or args.episodes not in (None, [11, 12])):
        parser.error("repick_memory requires VideoRepick episodes 11/12")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    output = args.output.resolve()
    relative = output.relative_to(root)
    conditions = AUTHORITY_CONDITIONS if args.phase == "authority" else CONDITIONS
    cases = regression_cases(args.episodes or [1, 2, 3, 4, 5]) if args.phase == "regression" else [
        (args.episode, *condition) for condition in conditions
    ]
    if args.phase in {"context", "grounding", "confirmation"}:
        condition = "observe_native_every" if args.phase == "context" else "observe_tools_every"
        cases = [(episode, condition, "carve", "execution_chunks", "every_chunk")
                 for episode in ((1, 4, 5) if args.phase == "confirmation" else (2, 3))]
    if args.phase == "task_admission":
        episodes = args.episodes or [args.episode]
        if len(set(episodes)) != len(episodes) or any(ep < 0 or ep >= 50 for ep in episodes):
            parser.error("admission episodes must be distinct in 0..49")
        cases = [(episode, *CONDITIONS[0]) for episode in episodes]
    memory_root = (root / args.memory_root).resolve()
    preflight_root = (root / args.memory_preflight_root).resolve()
    memory_preflight = {}
    if args.phase in {"memory", "memory_runtime", "repick_memory", "repick_diagnostic"}:
        from agentic_vla.benchmarks.robomme_memory import load_verified_identity_memory

        cases = repick_cases() if args.phase in {"repick_memory", "repick_diagnostic"} else memory_cases(args.episodes)
        if args.phase == "repick_diagnostic":
            cases = [case for case in cases if not case[1].endswith("_on")]
        if args.phase == "memory_runtime":
            cases = [(episode, name.replace("_off", "_full").replace("_on", "_once"), profile, feedback, schedule)
                     for episode, name, profile, feedback, schedule in cases]
        for episode in sorted({case[0] for case in cases}):
            demo = preflight_root / f"{args.task}_ep{episode}"
            instruction = json.loads((demo / "summary.json").read_text())["instruction"]
            memory_path = memory_root / f"{args.task}_ep{episode}_memory.json"
            try:
                _, receipt = load_verified_identity_memory(
                    memory_path, current_demo=demo / "initial_demo_front.mp4", task=args.task,
                    episode=episode, instruction=instruction,
                )
            except ValueError as error:
                if args.phase != "repick_diagnostic":
                    raise
                receipt = {"admitted": False, "reason": str(error),
                           "memory_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest()}
            memory_preflight[str(episode)] = receipt
    output.mkdir(parents=True, exist_ok=False)
    image = "carve-robomme:cuda12.8"
    policy_repo = root / "third_party/robomme_policy_learning"
    rows = []
    commands = []
    case_metadata = {}
    for episode, condition, profile, feedback, schedule in cases:
        name = f"ep{episode}_{condition}" if args.phase in {"regression", "context", "grounding", "confirmation", "memory", "memory_runtime", "task_admission", "repick_memory", "repick_diagnostic"} else condition
        case_metadata[name] = {"episode": episode, "condition": condition}
        command = [
            "docker", "run", "--rm", "--gpus", "all", "--network", "host",
            "--name", f"carve-feedback-{os.getpid()}-{name}",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video",
            "-e", "SAPIEN_RENDER_DEVICE=cuda",
            "-e", "PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src",
            "-v", f"{root}:/workspace", "-v", f"{policy_repo}:/policy:ro",
            "-w", "/workspace", image, "/app/.venv/bin/python",
            "scripts/run_robomme_vlm_groundsg.py",
            "--task", args.task, "--episode", str(episode),
            "--max-steps", "1300", "--action-horizon", "16",
            "--planner-profile", profile, "--execution-feedback", feedback,
            "--procedure-authority", "observe_only" if condition.startswith("observe_") else "legacy_override",
            "--planner-schedule", schedule,
            "--planner-gripper-change-threshold", "0.004",
            "--policy-label", "Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999",
            "--planner-label", "Qwen3-VL-4B-GroundSG-BF16",
            "--output", f"/workspace/{relative}/{name}",
        ]
        command += ["--planner-repair-attempts", "0"] if profile == "raw" or args.phase in {"memory", "memory_runtime", "repick_memory", "repick_diagnostic"} else [
            "--relational-color-grounding"
        ]
        if profile == "carve" and args.phase in {"context", "grounding", "confirmation", "memory", "memory_runtime", "repick_memory", "repick_diagnostic"}:
            command += ["--planner-context", "native"]
        if profile == "carve" and args.phase in {"grounding", "confirmation", "memory", "memory_runtime", "repick_memory", "repick_diagnostic"}:
            command += ["--grounding-authority", "observe_only"]
        if (args.phase in {"memory", "repick_memory"} and condition.endswith("_on")) or args.phase == "memory_runtime":
            command += ["--task-memory-file",
                        f"/workspace/{memory_root.relative_to(root)}/{args.task}_ep{episode}_memory.json"]
        if args.phase == "memory_runtime" and condition.endswith("_once"):
            command += ["--planner-demo-mode", "once_with_memory"]
        commands.append((name, command))
    source_paths = (
        "scripts/run_robomme_vlm_groundsg.py", "scripts/run_robomme_feedback_pilot.py",
        "scripts/run_robomme_policy_admission.py", "agentic_vla/benchmarks/robomme_memory.py",
        "agentic_vla/benchmarks/robomme_runtime.py", "agentic_vla/runtime/semantic.py",
        "scripts/serve_robomme_groundsg_policy.sh", "scripts/serve_robomme_groundsg_planner.sh",
        "scripts/build_robomme_unmask_swap_memory.py", "scripts/track_robomme_video_memory.py",
    )
    source_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths}
    manifest = {
        "protocol": f"robomme.{args.phase}.development.v1",
        "phase": args.phase,
        "task": args.task,
        "episodes": sorted({case[0] for case in cases}),
        "cases": case_metadata,
        "claim_boundary": "Known development episode, not held-out evidence or benchmark success rate.",
        "runner_sha256": hashlib.sha256(
            (root / "scripts/run_robomme_vlm_groundsg.py").read_bytes()
        ).hexdigest(),
        "commands": dict(commands),
        "source_sha256": source_hashes,
        "memory_preflight": memory_preflight,
        "unexecuted_conditions": ([{"episode": ep, "condition": "observe_memory_on",
            "reason": "diagnostic phase excludes unadmitted memory; see memory_preflight"}
            for ep in (11, 12)] if args.phase == "repick_diagnostic" else []),
    }
    shutil.copy2(root / "scripts/run_robomme_vlm_groundsg.py", output / "runner_snapshot.py")
    for path in source_paths:
        target = output / "sources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / path, target)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failure = None
    try:
        for name, command in commands:
            for path, expected in source_hashes.items():
                if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"registered source changed: {path}")
            print(f"Starting {name}: {args.task} episode {case_metadata[name]['episode']}", flush=True)
            started = time.perf_counter()
            with (output / f"{name}.log").open("w") as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.timeout_s, check=False)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    subprocess.run(
                        ["docker", "stop", "-t", "10", f"carve-feedback-{os.getpid()}-{name}"],
                        stdout=log, stderr=subprocess.STDOUT, check=False, timeout=30,
                    )
                    raise
            summary_path = output / name / "summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"{name} infrastructure/runner failure; inspect its log")
            summary = json.loads(summary_path.read_text())
            protocol_failure = args.phase in {"memory", "memory_runtime", "task_admission", "repick_memory", "repick_diagnostic"} and planner_protocol_failure(summary)
            if result.returncode != 0 and not protocol_failure:
                raise RuntimeError(f"{name} runner failure: {summary.get('failure')}; inspect its log")
            if summary["failure"] and not protocol_failure:
                raise RuntimeError(f"{name} model/runner failure: {summary['failure']}")
            rows.append({
                "run": name, **case_metadata[name],
                **{key: summary[key] for key in (
                    "status", "success", "executed_steps", "policy_calls", "planner_calls",
                    "rollout_time_s", "wall_time_s", "initial_observation_sha256",
                )},
                "feedback_receipts": len(summary["execution_feedback_trace"]),
                "boundary_events": sum(bool(row["execution_event"])
                                       for row in summary["execution_feedback_trace"]),
                "source": str(summary_path.relative_to(output)),
                "process_wall_time_s": time.perf_counter() - started,
                "planner_requests": summary.get("planner_request_count", sum(
                    1 + item["repair_attempts"] for item in summary["planner_trace"])),
                "protocol_failure": protocol_failure,
                "model_failure": summary["failure"],
                "memory_provenance": summary.get("memory_provenance"),
            })
            print(json.dumps(rows[-1]), flush=True)
    except BaseException as error:
        failure = str(error)
        if not failure:
            failure = type(error).__name__
        raise
    finally:
        paired_hashes = {
            str(episode): [row["initial_observation_sha256"] for row in rows if row["episode"] == episode]
            for episode in {case[0] for case in cases}
        }
        (output / "summary.json").write_text(json.dumps({
            "protocol": manifest["protocol"], "rows": rows,
            "complete": len(rows) == len(cases), "failure": failure,
            "same_initial_observations": all(len(set(hashes)) == 1 for hashes in paired_hashes.values()) if rows else None,
            "initial_observations_by_episode": paired_hashes,
            "claim_boundary": manifest["claim_boundary"],
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

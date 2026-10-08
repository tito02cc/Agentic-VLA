#!/usr/bin/env python3
"""One real, conservative AgenticPi05 integration episode; not an efficacy study."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from prepare_robodojo_pi05 import PROJECT_ROOT, sha256_file
from run_robodojo_pi05_admission import stop_process


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(process, port, *, profile=False):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}")
        try:
            if profile:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/carve/profile", timeout=2) as response:
                    return json.load(response)
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return None
        except (OSError, ValueError):
            time.sleep(2)
    raise TimeoutError("server readiness exceeded ten minutes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--task", choices=("organize_table", "classify_objects_by_language"), default="organize_table")
    parser.add_argument("--agent-mode", choices=("on", "off"), default="on",
                        help="off uses the same port/runtime/action executor without Agent intervention")
    parser.add_argument("--enable-feedback-refresh", action="store_true",
                        help="Development-only bounded short-prefix tool; no recovery efficacy admission")
    parser.add_argument("--critic-protocol", choices=("legacy", "evidence"), default="legacy")
    parser.add_argument("--advisory-critic-schedule", choices=("tool_only", "periodic"), default="tool_only",
                        help="tool_only avoids periodic unadmitted Critic requests; explicit tool checks remain")
    parser.add_argument("--execution-mode", choices=("episode", "feedback_tool_smoke", "kinematics_smoke", "motion_probe_smoke", "proposal_review_smoke"), default="episode",
                        help="Explicit diagnostic modes are not autonomous Agent evaluation")
    parser.add_argument("--vla-call-budget", type=int, default=None)
    parser.add_argument("--model-note-policy", choices=("quarantine", "legacy"), default="quarantine",
                        help="Quarantine historical model claims; legacy is for explicit ablations")
    args = parser.parse_args()
    if args.enable_feedback_refresh and args.agent_mode != "on":
        parser.error("feedback tool requires agent-mode on")
    if args.vla_call_budget is not None and args.vla_call_budget < 1:
        parser.error("VLA call budget must be positive")
    if args.execution_mode == "feedback_tool_smoke" and (
            args.agent_mode != "on" or not args.enable_feedback_refresh or args.vla_call_budget not in (None, 2)):
        parser.error("feedback tool smoke needs agent on, feedback enabled and a two-call budget")
    if args.execution_mode in {"kinematics_smoke", "motion_probe_smoke"} and (
            args.agent_mode != "off" or args.enable_feedback_refresh or args.vla_call_budget not in (None, 1)):
        parser.error("robot diagnostic needs agent off, no feedback tool and a one-call budget")
    if args.execution_mode == "proposal_review_smoke" and (
            args.agent_mode != "on" or args.enable_feedback_refresh or args.vla_call_budget not in (None, 2)):
        parser.error("proposal review needs agent on, no feedback refresh and two VLA calls")
    out = args.artifact_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bench = PROJECT_ROOT / "third_party/robodojo_official"
    agent_enabled = args.agent_mode == "on"
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_agent_{args.agent_mode}"
    if args.execution_mode == "feedback_tool_smoke":
        run_id += "_controlled_tool_smoke"
    elif args.execution_mode in {"kinematics_smoke", "motion_probe_smoke", "proposal_review_smoke"}:
        run_id += "_" + args.execution_mode
    vlm_port, vla_port = free_port(), free_port()
    while vlm_port == vla_port:
        vla_port = free_port()
    config = json.loads((PROJECT_ROOT / "configs/robodojo_sorting_agent.json").read_text())
    config["run_id"] = run_id
    config["planner"].update(planner_id="qwen3-vl-4b-nf4-integration", endpoint=f"http://127.0.0.1:{vlm_port}/v1/chat/completions")
    config["vla"]["endpoint"] = f"ws://127.0.0.1:{vla_port}"
    config["benchmark"].update(task_id=args.task, max_steps=1000 if args.task == "organize_table" else 1100)
    config["artifacts"]["results_root"] = str(out / "sessions")
    execution = {"agent_enabled": agent_enabled, "memory_enabled": agent_enabled,
                 "critic_authority": "advisory", "critic_camera_mode": "head", "compact_task_only_review": True,
                 "critic_protocol": args.critic_protocol,
                 "advisory_critic_schedule": args.advisory_critic_schedule,
                 "model_note_policy": args.model_note_policy,
                 "feedback_refresh_enabled": args.enable_feedback_refresh,
                 "vla_call_budget": args.vla_call_budget}
    if args.execution_mode == "feedback_tool_smoke":
        execution.update(vla_call_budget=2, critic_call_budget=2)
    elif args.execution_mode in {"kinematics_smoke", "motion_probe_smoke"}:
        execution.update(vla_call_budget=1)
    elif args.execution_mode == "proposal_review_smoke":
        execution.update(vla_call_budget=2, proposal_review_enabled=True)
        config["planner"].update(max_calls_per_episode=2, max_grounding_repairs=0)
    write_json(out / "config.json", config)
    write_json(out / "execution_config.json", execution)
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8", XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.55")
    vlm_command = ["/home/admin1/miniconda3/envs/g2agent/bin/python", str(PROJECT_ROOT / "scripts/serve_carve_vlm_profile.py"),
        "--model", config["planner"]["model"], "--profile", "vision_preserving_nf4", "--vision-only",
        "--port", str(vlm_port), "--receipt", str(out / "vlm_profile.json")]
    vla_command = [str(PROJECT_ROOT / "openpi/.venv/bin/python"), str(PROJECT_ROOT / "scripts/serve_robodojo_sorting_pi05.py"),
        "--task", args.task, "--port", str(vla_port), "--artifact-dir", str(out / "policy")]
    helper = bench / "XPolicyLab/policy/starVLA/scripts/run_hf_robodojo_env_client.sh"
    client_command = ["bash", str(helper), "/home/admin1/miniconda3/envs/RoboDojo", str(vla_port),
        "RoboDojo", args.task, "arx_x5", "AgenticPi05", "official_pi05_conservative", str(bench), "0", "0", "127.0.0.1"]
    source_files = [Path(__file__), PROJECT_ROOT / "scripts/serve_robodojo_sorting_pi05.py",
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_pi05_episode.py", PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_pi05_port.py",
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_pi05_policy.py", PROJECT_ROOT / "agentic_vla/runtime/agent.py",
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_sorting.py",
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_kinematics.py",
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_motion_probe.py",
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_action_proposal.py",
        PROJECT_ROOT / "agentic_vla/session.py", PROJECT_ROOT / "agentic_vla/runtime/harness.py",
        PROJECT_ROOT / "agentic_vla/toolchain/verifier.py",
        bench / "XPolicyLab/policy/AgenticPi05/deploy.py", bench / f"task/RoboDojo/tasks/{args.task}.py", helper]
    hashes = {str(p.resolve()): sha256_file(p) for p in source_files}
    write_json(out / "launch.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "commands": [vlm_command, vla_command, client_command], "source_sha256": hashes,
        "layout_set": 0, "layout_id": 0, "episodes": 1, "policy_seed": 0, "agent_enabled": agent_enabled,
        "execution_mode": args.execution_mode,
        "scope": "real conservative integration; controlled smoke is not autonomous; Critic advisory, task_only, reference JAX; not efficacy evidence"})
    processes = []
    outcome = {"completed": False, "agent_enabled": agent_enabled, "run_id": run_id,
               "execution_mode": args.execution_mode, "critic_authority": "advisory"}
    started = time.monotonic()
    def start(command, logname, child_env=env):
        with (out / logname).open("w") as log:
            process = subprocess.Popen(command, env=child_env, cwd=PROJECT_ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    try:
        vlm = None
        if agent_enabled:
            vlm = start(vlm_command, "vlm.log")
            profile = wait_ready(vlm, vlm_port, profile=True)
            if profile["model_id"] != config["planner"]["model"] or profile["profile"] != "vision_preserving_nf4":
                raise ValueError("unexpected VLM identity/profile")
            print("VLM ready", flush=True)
        vla = start(vla_command, "policy.log")
        wait_ready(vla, vla_port)
        print("official pi05 ready", flush=True)
        client_env = dict(env, AGENTIC_PI05_RUN_CONFIG=str(out / "config.json"),
            AGENTIC_PI05_EXECUTION_MODE=args.execution_mode,
            AGENTIC_PI05_EXECUTION_CONFIG=json.dumps(execution), EVAL_NUM="1", STARVLA_ROBODOJO_NUM_ENVS="1",
            ROBODOJO_RUN_ID=run_id, ROBODOJO_REQUIRE_AUDITED_RESET="0", ROBODOJO_MAX_BASH_RETRIES="1")
        client = start(client_command, "simulation.log", client_env)
        deadline = time.monotonic() + 1800
        with (out / "gpu_samples.jsonl").open("w") as log:
            while client.poll() is None:
                # The upstream evaluator retries generic policy exceptions. An
                # integration error must end this trial, not silently skip it.
                if "[Eval] unhandled exception during reset/run_eval:" in (out / "simulation.log").read_text(errors="replace"):
                    raise RuntimeError("policy integration exception; inspect simulation.log")
                if (vlm is not None and vlm.poll() is not None) or vla.poll() is not None:
                    raise RuntimeError("model server exited during simulation")
                if time.monotonic() > deadline:
                    raise TimeoutError("single episode exceeded thirty minutes")
                sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=10)
                log.write(json.dumps({"elapsed_s": time.monotonic()-started, "gpu": sample.stdout.strip()}) + "\n")
                log.flush()
                time.sleep(10)
        if client.returncode:
            raise RuntimeError(f"simulation exited {client.returncode}")
        result_root = bench / f"eval_result/RoboDojo/{args.task}/AgenticPi05/arx_x5/0_official_pi05_conservative/{run_id}"
        result_path = result_root / "_result.json"
        summaries = list((out / "sessions").rglob("execution_summary.json"))
        if len(summaries) != 1:
            raise RuntimeError("expected exactly one Agent execution summary")
        outcome.update(completed=True, official_result_path=str(result_path),
            official_result=json.loads(result_path.read_text()), execution_summary_path=str(summaries[0]),
            execution_summary=json.loads(summaries[0].read_text()), videos=[str(p) for p in sorted(result_root.glob("*.mp4"))])
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for process in reversed(processes):
            stop_process(process)
        outcome["wall_seconds"] = time.monotonic()-started
        outcome["source_drift"] = [str(p) for p in source_files if sha256_file(p) != hashes[str(p.resolve())]]
        write_json(out / "summary.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

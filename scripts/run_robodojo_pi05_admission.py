#!/usr/bin/env python3
"""Run a bounded official pi0.5 admission, not an Agent effectiveness study."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from prepare_robodojo_pi05 import POLICY_DIR, PREFIX, PROJECT_ROOT, REVISION, sha256_file


BENCH_ROOT = PROJECT_ROOT / "third_party/robodojo_official"
XPL_ROOT = BENCH_ROOT / "XPolicyLab"
OPENPI_SRC = POLICY_DIR / "openpi/src"
PAYLOAD = PROJECT_ROOT / "artifacts/robodojo/native_trace_b0_20260911/capture_1ep/captured_payloads/payload_0000.npz"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class RecordedPolicy:
    def __init__(self, policy, trace_path, *, mode, action_dim, instruction_override=None):
        from agentic_vla.runtime import CarveRuntime
        from agentic_vla.runtime.adapters.pi05 import Pi05Adapter
        from agentic_vla.runtime.contracts import ActionSpec

        self.policy = policy
        self.trace_path = trace_path
        self.mode = mode
        self.phase = "rollout"
        self.index = 0
        self.instruction_override = instruction_override
        self.spec = ActionSpec(
            action_dim=action_dim, representation="absolute_joint_position",
            coordinate_frame="arx_x5_joint", gripper_convention="0_closed_1_open",
            control_frequency_hz=25, normalization_id="official_pi05/arx_x5_sim",
        )
        self.runtime = CarveRuntime(Pi05Adapter(
            policy, adapter_id="robodojo_official_pi05", max_action_horizon=50,
            action_spec=self.spec,
        ), fallback_mode="strict")

    def infer(self, observation):
        import numpy as np
        from agentic_vla.runtime.contracts import InferenceRequest

        environment_instruction = observation["prompt"]
        if self.phase == "rollout" and self.instruction_override:
            observation = {**observation, "prompt": self.instruction_override}
        if self.phase == "rollout" and self.index == 0:
            path = self.trace_path.parent / "initial_observation.npz"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite initial observation: {path}")
            np.savez_compressed(path, state=np.asarray(observation["state"]),
                                instruction=np.asarray(observation["prompt"]),
                                **{key: np.asarray(value) for key, value in observation["images"].items()})
        start = time.perf_counter()
        if self.mode == "runtime":
            request = InferenceRequest(
                observation=observation, instruction=observation["prompt"],
                metadata={"raw_payload": observation},
            )
            chunk = self.runtime.infer(request)
            result = dict(chunk.raw_output)
            result["actions"] = chunk.actions
        else:
            result = self.policy.infer(observation)
        elapsed_ms = (time.perf_counter() - start) * 1000
        actions = np.asarray(result["actions"])
        self.spec.validate(actions)
        if actions.shape != (50, self.spec.action_dim):
            raise ValueError(f"Unexpected native action shape: {actions.shape}")
        record = {
            "index": self.index, "phase": self.phase, "mode": self.mode,
            "prompt": observation["prompt"], "wall_ms": elapsed_ms,
            "actions_shape": list(actions.shape), "actions_finite": True,
            "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "state": np.asarray(observation["state"]).tolist(),
            "images": {key: {"shape": list(np.asarray(value).shape),
                                "sha256": hashlib.sha256(np.asarray(value).tobytes()).hexdigest()}
                       for key, value in observation["images"].items()},
            "agent_enabled": False,
            "environment_instruction": environment_instruction,
            "instruction_override": self.instruction_override if self.phase == "rollout" else None,
        }
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.index += 1
        return result


def serve(args):
    import numpy as np
    import jax
    import yaml
    from XPolicyLab.policy.Pi_05.model import Model, encode_obs
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    artifact = args.artifact_dir
    cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
    cfg.update(task_name=args.task, env_cfg_type="arx_x5", action_type="joint",
               model_path=str(args.checkpoint), checkpoint_num=59999, seed=0)
    write_json(artifact / "resolved_model_config.json", cfg)
    write_json(artifact / "runtime_environment.json", {
        "python": sys.version, "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in (
            "jax", "jaxlib", "orbax-checkpoint", "torch", "transformers", "numpy",
            "msgpack-numpy", "websockets",
        )},
        "openpi_source": str(OPENPI_SRC),
        "jax_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "jax_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    })
    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("A JAX GPU is required; refusing an accidental CPU rollout")
    start = time.perf_counter()
    model = Model(cfg)
    load_seconds = time.perf_counter() - start
    raw_policy = model.policy
    stats = json.loads((args.checkpoint / "assets/arx_x5_sim/norm_stats.json").read_text())
    dimension = len(stats["norm_stats"]["state"]["mean"])
    recorded = RecordedPolicy(raw_policy, artifact / "inference_trace.jsonl", mode=args.mode,
                              action_dim=dimension, instruction_override=args.instruction_override)
    # Replay only recorded observations. Old policy actions in the NPZ are never used as input.
    metadata = [json.loads(line) for line in (PAYLOAD.parent / "payload_manifest.jsonl").read_text().splitlines()]
    prompt = next(row["lang"] for row in metadata if row["arrays"] == PAYLOAD.name)
    with np.load(PAYLOAD, allow_pickle=False) as payload:
        observation = encode_obs({
            "state": payload["state"][0], "instruction": prompt,
            "images": dict(zip(("cam_high", "cam_left_wrist", "cam_right_wrist"),
                               (payload[f"image_{index}"] for index in range(3)), strict=True)),
        }, "joint", model.robot_action_dim_info)
    saved_rng = raw_policy._rng
    recorded.phase = "warmup_and_wrapper_check"
    recorded.mode = "native"
    native = recorded.infer(observation)["actions"]
    raw_policy._rng = saved_rng
    recorded.mode = "runtime"
    wrapped = recorded.infer(observation)["actions"]
    raw_policy._rng = saved_rng
    difference = float(np.max(np.abs(np.asarray(native) - np.asarray(wrapped))))
    write_json(artifact / "wrapper_check.json", {
        "observation_source": str(PAYLOAD), "source_sha256": sha256_file(PAYLOAD),
        "same_rng": True, "max_abs_action_difference": difference,
        "exactly_equal": bool(np.array_equal(native, wrapped)), "load_seconds": load_seconds,
        "devices": [str(device) for device in devices],
        "sample_kwargs": raw_policy._sample_kwargs,
        "native_num_steps": inspect.signature(raw_policy._model.sample_actions).parameters["num_steps"].default,
        "scope": "One recorded observation; not a task success or acceleration claim",
    })
    if difference != 0:
        raise RuntimeError("Runtime passthrough changed native actions; refusing rollout")
    recorded.mode = args.mode
    recorded.phase = "rollout"
    recorded.index = 0
    model.policy = recorded
    model.reset()
    (artifact / "worker_ready").write_text("ready\n")
    asyncio.run(PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=args.port)).serve_forever())


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run(args):
    artifact = args.artifact_dir.resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite {artifact}")
    manifest_path = POLICY_DIR / "checkpoints/huggingface/pi05_inference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != REVISION or Path(manifest["checkpoint_path"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Checkpoint does not match the pinned official download")
    for item in manifest["files"]:
        path = manifest_path.parent / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Checkpoint verification failed: {path}")
    artifact.mkdir(parents=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_pi05_" + args.mode
    condition = args.mode + ("_instruction_probe" if args.instruction_override else "")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        args.port = sock.getsockname()[1]
    worker_env = dict(os.environ)
    worker_env.update(
        PYTHONPATH=os.pathsep.join(map(str, (OPENPI_SRC, BENCH_ROOT, XPL_ROOT, PROJECT_ROOT))),
        XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.55",
        CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
    )
    worker_command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--task", args.task,
                      "--mode", args.mode, "--artifact-dir", str(artifact), "--checkpoint", str(args.checkpoint),
                      "--port", str(args.port)]
    if args.instruction_override:
        worker_command.extend(["--instruction-override", args.instruction_override])
    helper = XPL_ROOT / "policy/starVLA/scripts/run_hf_robodojo_env_client.sh"
    client_command = ["bash", str(helper), args.sim_env, str(args.port), "RoboDojo", args.task,
                      "arx_x5", "Pi_05", "official_pi05_59999_" + condition, str(BENCH_ROOT), "0", "0", "127.0.0.1"]
    client_env = dict(os.environ)
    client_env.update(EVAL_NUM="1", ROBODOJO_RUN_ID=run_id, ROBODOJO_REQUIRE_AUDITED_RESET="0",
                      ROBODOJO_MAX_BASH_RETRIES="1", STARVLA_ROBODOJO_NUM_ENVS="1",
                      PYTHONPATH=str(PROJECT_ROOT), OMP_NUM_THREADS="8")
    source_files = [Path(__file__).resolve(), POLICY_DIR / "model.py", POLICY_DIR / "deploy.py",
                    POLICY_DIR / "deploy.yml", OPENPI_SRC / "openpi/training/config.py",
                    OPENPI_SRC / "openpi/policies/aloha_policy.py", OPENPI_SRC / "openpi/policies/policy.py",
                    BENCH_ROOT / f"task/RoboDojo/tasks/{args.task}.py", helper,
                    PROJECT_ROOT / "agentic_vla/runtime/adapters/pi05.py", PROJECT_ROOT / "agentic_vla/runtime/runtime.py"]
    sources = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in source_files}
    write_json(artifact / "launch.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "task": args.task,
        "checkpoint_manifest": manifest, "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "worker_command": worker_command, "client_command": client_command, "run_id": run_id,
        "source_sha256": sources, "num_episodes": 1, "layout_set": 0,
        "agent_enabled": False, "audited_reset_required": False,
        "instruction_override": args.instruction_override,
        "experiment_type": "manual_subgoal_diagnostic" if args.instruction_override else "reference_admission",
        "scope": "One fresh process per episode; native model admission, not Harness validation",
    })
    worker = client = None
    started = time.monotonic()
    outcome = {"completed": False, "agent_enabled": False,
               "instruction_override": args.instruction_override,
               "experiment_type": "manual_subgoal_diagnostic" if args.instruction_override else "reference_admission"}
    try:
        with (artifact / "policy.log").open("w") as log:
            worker = subprocess.Popen(worker_command, env=worker_env, cwd=BENCH_ROOT,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1200
        while not (artifact / "worker_ready").exists():
            if worker.poll() is not None:
                raise RuntimeError(f"Policy exited {worker.returncode}; see policy.log")
            if time.monotonic() >= deadline:
                raise TimeoutError("Policy initialization exceeded 20 minutes")
            time.sleep(2)
        with (artifact / "simulation.log").open("w") as log:
            client = subprocess.Popen(client_command, env=client_env, cwd=BENCH_ROOT,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        code = client.wait(timeout=1800)
        if code != 0:
            raise RuntimeError(f"Simulation exited {code}; see simulation.log")
        root = BENCH_ROOT / f"eval_result/RoboDojo/{args.task}/Pi_05/arx_x5/0_official_pi05_59999_{condition}/{run_id}"
        result_path = root / "_result.json"
        outcome.update(completed=True, official_result_path=str(result_path),
                       official_result=json.loads(result_path.read_text()),
                       videos=[str(path) for path in sorted(root.glob("*.mp4"))])
    except BaseException as exc:
        outcome.update(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_process(client)
        stop_process(worker)
        outcome["wall_seconds"] = time.monotonic() - started
        outcome["source_drift"] = [str(path) for path in source_files
                                   if sha256_file(path) != sources[str(path.relative_to(PROJECT_ROOT))]]
        write_json(artifact / "summary.json", outcome)
        print(json.dumps(outcome, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stack_bowls", "organize_table", "classify_objects_by_language", "match_and_pick_from_conveyor"), default="stack_bowls")
    parser.add_argument("--mode", choices=("native", "runtime"), default="native")
    parser.add_argument("--instruction-override", help="Manual subgoal diagnostic only; never an autonomous Agent result")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=POLICY_DIR / "checkpoints/huggingface" / PREFIX)
    parser.add_argument("--sim-env", default="/home/admin1/miniconda3/envs/RoboDojo")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        serve(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

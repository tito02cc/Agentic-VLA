#!/usr/bin/env python3
"""Serve official pi0.5, or check native/runtime equality and episode RNG reset.

Run with openpi/.venv/bin/python. This does not launch any benchmark by itself.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

from prepare_robodojo_pi05 import POLICY_DIR, PREFIX, PROJECT_ROOT, REVISION, sha256_file


def check_recorded_task_input(model, source, output):
    """Real model/adapter audit on recorded RGB; NOT a simulated episode."""
    import numpy as np
    from types import SimpleNamespace
    from XPolicyLab.policy.Pi_05.model import encode_obs
    from XPolicyLab.utils.process_data import unpack_robot_state, pack_robot_state
    from agentic_vla.benchmarks.robodojo_pi05_port import XPolicyLabPi05Port
    from agentic_vla.benchmarks.robodojo_pi05_policy import array_fingerprint

    with np.load(source, allow_pickle=False) as data:
        raw = {"instruction": str(data["instruction"].item()),
               "state": unpack_robot_state(data["state"], "joint", model.robot_action_dim_info),
               "vision": {k: (data[k].transpose(1, 2, 0).copy() if data[k].shape[0] == 3
                              else data[k].copy()) for k in
                          ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
    class DirectClient:
        def call(self, func_name, **kwargs):
            return getattr(model, func_name)(**kwargs)
    captured_actions = []
    env = SimpleNamespace(get_obs=lambda: raw, take_action=captured_actions.append)
    port = XPolicyLabPi05Port(env, DirectClient())
    observed = port.observe()
    native_input = encode_obs(raw, "joint", model.robot_action_dim_info)
    port_input = encode_obs({"instruction": observed.instruction, "state": observed.state,
                             "images": observed.frames}, "joint", model.robot_action_dim_info)
    same_input = (native_input["prompt"] == port_input["prompt"]
                  and np.array_equal(native_input["state"], port_input["state"])
                  and all(np.array_equal(native_input["images"][k], port_input["images"][k])
                          for k in native_input["images"]))
    model.reset({"policy_seed": 0})
    native = np.asarray(model.native_policy.infer(native_input)["actions"])
    model.reset({"policy_seed": 0})
    wrapped = port.infer(raw["instruction"])
    model.reset({"policy_seed": 0})
    repeated = port.infer(raw["instruction"])
    reads_before_execute = port.observation_reads
    for action in wrapped:
        port.execute(action)
    delivered = np.stack([pack_robot_state({"state": a}, "joint", model.robot_action_dim_info)
                          for a in captured_actions])
    np.savez_compressed(output / "adapter_actions.npz", native=native, wrapped=wrapped,
                        repeated=repeated, delivered=delivered)
    return {"observation_source": str(source.resolve()), "observation_sha256": sha256_file(source),
            "recorded_input_audit": True, "simulation_run": False,
            "raw_and_images_encoding_equal": same_input,
            "native_equals_runtime": bool(np.array_equal(native, wrapped)),
            "reset_reproducible": bool(np.array_equal(wrapped, repeated)),
            "delivered_actions_equal": bool(np.array_equal(wrapped, delivered)),
            "max_action_difference": float(np.abs(native - wrapped).max()),
            "observation_reads_before_execute": reads_before_execute,
            "input_fingerprints": {"state": array_fingerprint(native_input["state"]),
                                   "images": {k: array_fingerprint(v) for k, v in native_input["images"].items()}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("organize_table", "classify_objects_by_language"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-observation", type=Path,
                        help="Task-specific NPZ (state, instruction, three RGB cameras) for --check-only")
    parser.add_argument("--compare-observation", type=Path,
                        help="Optional second recorded input, replayed in the same model process")
    parser.add_argument("--probe-observation", type=Path)
    parser.add_argument("--probe-instruction", action="append", default=[])
    args = parser.parse_args()
    if args.check_observation and not args.check_only:
        parser.error("--check-observation requires --check-only")
    if args.compare_observation and not args.check_observation:
        parser.error("--compare-observation requires --check-observation")
    if args.probe_observation and (args.check_only or not args.probe_instruction):
        parser.error("instruction probe requires instructions and excludes --check-only")
    if args.artifact_dir.exists():
        raise FileExistsError("choose a fresh artifact directory")
    bench = PROJECT_ROOT / "third_party/robodojo_official"
    sys.path[:0] = [str(POLICY_DIR / "openpi/src"), str(bench), str(bench / "XPolicyLab"), str(PROJECT_ROOT)]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")
    import jax
    import numpy as np
    import yaml
    from agentic_vla.benchmarks.robodojo_pi05_policy import build_model, validate_model_reset
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    manifest_path = POLICY_DIR / "checkpoints/huggingface/pi05_inference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("checkpoint revision differs from the admitted official model")
    for item in manifest["files"]:
        path = manifest_path.parent / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"checkpoint mismatch: {path}")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU is required for this official model admission")
    args.artifact_dir.mkdir(parents=True)
    cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
    cfg.update(task_name=args.task, env_cfg_type="arx_x5", action_type="joint",
               model_path=str(POLICY_DIR / "checkpoints/huggingface" / PREFIX),
               inference_trace=str(args.artifact_dir / "policy_trace.jsonl"))
    started = time.perf_counter()
    model = build_model(cfg)
    loaded = time.perf_counter() - started
    report = {"task_config": args.task, "checkpoint_revision": REVISION,
              "checkpoint_manifest_sha256": sha256_file(manifest_path), "load_seconds": loaded,
              "simulation_run": False, "live_vlm_run": False}
    if args.check_observation:
        report.update(check_recorded_task_input(model, args.check_observation, args.artifact_dir))
        if args.compare_observation:
            second_dir = args.artifact_dir / "second_input"
            second_dir.mkdir()
            report["second_input"] = check_recorded_task_input(model, args.compare_observation, second_dir)
        (args.artifact_dir / "check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not all(report[k] for k in ("raw_and_images_encoding_equal", "native_equals_runtime",
                                       "reset_reproducible", "delivered_actions_equal")):
            raise RuntimeError("task-specific input/adapter equality audit failed")
        if args.compare_observation and not all(report["second_input"][k] for k in (
                "raw_and_images_encoding_equal", "native_equals_runtime", "reset_reproducible",
                "delivered_actions_equal")):
            raise RuntimeError("second recorded input failed adapter equality audit")
        return
    if args.probe_observation:
        from XPolicyLab.policy.Pi_05.model import encode_obs
        with np.load(args.probe_observation, allow_pickle=False) as data:
            original = str(data["instruction"].item())
            payload = {"state": data["state"].copy(), "instruction": original,
                       "images": {key: data[key].copy() for key in
                                  ("cam_high", "cam_left_wrist", "cam_right_wrist")}}
        prompts = [original, *args.probe_instruction, original]
        report.update(observation_source=str(args.probe_observation.resolve()),
                      observation_sha256=sha256_file(args.probe_observation),
                      prompts=prompts, policy_seeds=[0, 1],
                      interpretation="instruction sensitivity only; not execution quality or Agent gain")
        (args.artifact_dir / "probe_plan.json").write_text(json.dumps(report, indent=2) + "\n")
        saved, rows = {}, []
        joint_indices = [i for i in range(14) if i not in (6, 13)]
        for seed in (0, 1):
            reference = None
            for index, prompt in enumerate(prompts):
                receipt = validate_model_reset(model.reset({"policy_seed": seed}), seed)
                observation = encode_obs({**payload, "instruction": prompt}, "joint", model.robot_action_dim_info)
                start = time.perf_counter()
                action = np.asarray(model.policy.infer(observation)["actions"])
                elapsed = (time.perf_counter() - start) * 1000
                if reference is None:
                    reference = action.copy()
                diff = np.abs(action - reference)
                row = {"seed": seed, "prompt_index": index, "prompt": prompt,
                       "reset_receipt": receipt, "wall_ms": elapsed,
                       "equal_to_original": bool(np.array_equal(action, reference)),
                       "joint_mean_abs_delta": float(diff[:, joint_indices].mean()),
                       "joint_max_abs_delta": float(diff[:, joint_indices].max()),
                       "gripper_mean_abs_delta": float(diff[:, [6, 13]].mean()),
                       "gripper_max_abs_delta": float(diff[:, [6, 13]].max())}
                rows.append(row)
                saved[f"seed_{seed}_prompt_{index}"] = action
                print(json.dumps(row), flush=True)
        report["calls"] = rows
        report["repeat_reproducible"] = all(row["equal_to_original"] for row in rows if row["prompt_index"] == len(prompts) - 1)
        np.savez_compressed(args.artifact_dir / "actions.npz", **saved)
        (args.artifact_dir / "instruction_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["repeat_reproducible"]:
            raise RuntimeError("same-observation/same-seed repeat failed")
        return
    if args.check_only:
        payload_path = PROJECT_ROOT / "artifacts/robodojo/native_trace_b0_20260911/capture_1ep/captured_payloads/payload_0000.npz"
        rows = [json.loads(line) for line in (payload_path.parent / "payload_manifest.jsonl").read_text().splitlines()]
        prompt = next(row["lang"] for row in rows if row["arrays"] == payload_path.name)
        with np.load(payload_path, allow_pickle=False) as payload:
            from XPolicyLab.policy.Pi_05.model import encode_obs
            observation = encode_obs({
                "state": payload["state"][0], "instruction": prompt,
                "images": dict(zip(("cam_high", "cam_left_wrist", "cam_right_wrist"),
                                   (payload[f"image_{i}"] for i in range(3)), strict=True)),
            }, "joint", model.robot_action_dim_info)
        first = validate_model_reset(model.reset({"policy_seed": 0}), 0)
        native = np.asarray(model.native_policy.infer(observation)["actions"])
        model.reset({"policy_seed": 0})
        wrapped = model.policy.infer(observation)["actions"]
        second = validate_model_reset(model.reset({"policy_seed": 0}), 0)
        repeated = model.policy.infer(observation)["actions"]
        report.update(observation_source=str(payload_path), observation_sha256=sha256_file(payload_path),
                      native_equals_runtime=bool(np.array_equal(native, wrapped)),
                      reset_reproducible=bool(np.array_equal(wrapped, repeated)),
                      max_action_difference=float(np.max(np.abs(native - wrapped))),
                      action_shape=list(wrapped.shape), first_reset=first, second_reset=second)
        (args.artifact_dir / "check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not report["native_equals_runtime"] or not report["reset_reproducible"]:
            raise RuntimeError("official model reference/reset contract failed")
        return
    (args.artifact_dir / "server.json").write_text(json.dumps(report, indent=2) + "\n")
    asyncio.run(PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=args.port)).serve_forever())


if __name__ == "__main__":
    main()

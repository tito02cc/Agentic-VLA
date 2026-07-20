#!/usr/bin/env python3
"""Validate an optimized CARVE policy server on one recorded observation."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from agentic_vla.runtime import CarveRuntime
from agentic_vla.runtime.adapters import LegacyPolicyClientBridge, Pi05Adapter
from scripts.benchmark_carve_pi05_profile import monitor_proprio_to_policy_state


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOTS = ROOT / "results/carve_t689_paired_5states_v6/joint/failure_snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--snapshot-dir", type=pathlib.Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    args = parse_args()
    array_path = next(iter(sorted(args.snapshot_dir.glob("*.npz"))), None)
    if array_path is None:
        raise FileNotFoundError(f"no snapshots in {args.snapshot_dir}")
    metadata = json.loads(array_path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(array_path, allow_pickle=False) as arrays:
        observation = {
            "observation/image": np.asarray(arrays["observation_agentview_image"]),
            "observation/wrist_image": np.asarray(arrays["observation_wrist_image"]),
            "observation/state": monitor_proprio_to_policy_state(
                arrays["observation_proprio"]
            ),
        }

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    runtime = CarveRuntime(Pi05Adapter(client), fallback_mode="strict")
    bridge = LegacyPolicyClientBridge(runtime)
    response = bridge.infer(
        {
            **observation,
            "prompt": str(metadata["instruction"]),
            "episode_id": f"profile-smoke:{metadata['task_id']}",
            "timestep": int(metadata["timestep"]),
            "agentic": {"event": "stall", "action": "retry", "reason": "profile_smoke"},
            "runtime_controls": {
                "inference_steps": 2,
                "max_actions": 10,
                "deadline_ms": float(args.deadline_ms),
            },
            "runtime_trace_context": {
                "experiment": "optimized-profile-agentic-websocket-smoke",
                "mode": "recovery_vla",
            },
            "runtime_noise": np.random.default_rng(20260716).standard_normal(
                (10, 32), dtype=np.float32
            ),
        }
    )
    trace = runtime.last_trace
    if trace is None:
        raise RuntimeError("runtime did not emit a trace")
    optimization = trace.metadata.get("optimization_profile", {})
    profile = optimization.get("profile", {}) if isinstance(optimization, dict) else {}
    passed = bool(
        profile.get("profile_id") == args.expected_profile
        and trace.metadata.get("agentic_event", {}).get("event") == "stall"
        and trace.applied_controls.get("inference_steps") == 2
        and len(response["actions"]) == 10
    )
    payload = {
        "experiment": "optimized-profile-agentic-websocket-smoke",
        "snapshot": array_path.name,
        "expected_profile": args.expected_profile,
        "server_profile": optimization,
        "action_shape": list(np.asarray(response["actions"]).shape),
        "runtime_trace": trace.to_dict(),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "runtime_trace"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

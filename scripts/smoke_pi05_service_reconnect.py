#!/usr/bin/env python3
"""Verify that separate PI0.5 websocket clients can use one served profile."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_carve_pi05_profile import (  # noqa: E402
    monitor_proprio_to_policy_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=2)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    args = parse_args()
    if args.clients < 2:
        raise ValueError("reconnect gate requires at least two clients")
    snapshot = pathlib.Path(args.snapshot).expanduser().resolve()
    metadata = json.loads(snapshot.with_suffix(".json").read_text())
    with np.load(snapshot, allow_pickle=False) as arrays:
        observation = {
            "observation/image": np.asarray(arrays["observation_agentview_image"]),
            "observation/wrist_image": np.asarray(
                arrays["observation_wrist_image"]
            ),
            "observation/state": monitor_proprio_to_policy_state(
                arrays["observation_proprio"]
            ),
            "prompt": str(metadata["instruction"]),
            "runtime_controls": {"inference_steps": args.inference_steps},
        }

    attempts = []
    for index in range(args.clients):
        started = time.perf_counter()
        client = None
        try:
            client = WebsocketClientPolicy(host=args.host, port=args.port)
            deployment = client.get_server_metadata().get(
                "carve_deployment_profile", {}
            )
            output = client.infer(observation)
            shape = list(np.asarray(output["actions"]).shape)
            attempts.append(
                {
                    "client_index": index,
                    "passed": shape == [10, 7],
                    "action_shape": shape,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "profile_id": deployment.get("profile", {}).get(
                        "profile_id"
                    ),
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "client_index": index,
                    "passed": False,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
        finally:
            connection = None if client is None else getattr(client, "_ws", None)
            close = None if connection is None else getattr(connection, "close", None)
            if callable(close):
                close()

    receipt = {
        "schema_version": 1,
        "kind": "pi05_websocket_reconnect_gate",
        "claim_boundary": "Service lifecycle evidence; no environment execution or task-success claim.",
        "passed": all(attempt["passed"] for attempt in attempts),
        "attempts": attempts,
    }
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=True))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

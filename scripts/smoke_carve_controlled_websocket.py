#!/usr/bin/env python3
"""Exercise CARVE's dynamic inference-step protocol over a real WebSocket."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import socket
import time

import numpy as np

from agentic_vla.runtime import CarveRuntime, InferenceControls, InferenceRequest
from agentic_vla.runtime.adapters import Pi05Adapter
from agentic_vla.runtime.server import RuntimeControllablePolicy


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeOpenPiPolicy:
    def __init__(self) -> None:
        self._sample_kwargs = {"num_steps": 7}
        self._metadata = {"backend": "fake-openpi"}

    @property
    def metadata(self):
        return self._metadata

    def infer(self, request):
        steps = int(self._sample_kwargs["num_steps"])
        return {
            "actions": np.full((4, 7), float(steps), dtype=np.float32),
            "policy_timing": {"infer_ms": 0.1},
        }


def _serve(port: int) -> None:
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = RuntimeControllablePolicy(_FakeOpenPiPolicy())
    WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=port,
        metadata=policy.metadata,
    ).serve_forever()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, process: multiprocessing.Process) -> None:
    for _ in range(100):
        if not process.is_alive():
            raise RuntimeError("test policy server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1) as connection:
                connection.sendall(
                    b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
                )
                if b"200 OK" in connection.recv(256):
                    return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("test policy server did not become ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "carve_e0_controlled_websocket_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    args = parse_args()
    port = _free_port()
    process = multiprocessing.Process(target=_serve, args=(port,), daemon=True)
    process.start()
    try:
        _wait_for_port(port, process)
        client = WebsocketClientPolicy(host="127.0.0.1", port=port)
        runtime = CarveRuntime(Pi05Adapter(client), fallback_mode="strict")
        controlled = runtime.infer(
            InferenceRequest(
                observation={"state": np.zeros(8, dtype=np.float32)},
                instruction="move",
                controls=InferenceControls(inference_steps=2),
            )
        )
        default = runtime.infer(
            InferenceRequest(
                observation={"state": np.zeros(8, dtype=np.float32)},
                instruction="move",
            )
        )
        controlled_value = float(np.asarray(controlled.actions)[0, 0])
        default_value = float(np.asarray(default.actions)[0, 0])
        payload = {
            "experiment": "E0-controlled-websocket-smoke",
            "server_advertised_dynamic_steps": bool(
                client.get_server_metadata()["carve_capabilities"][
                    "configurable_inference_steps"
                ]
            ),
            "requested_steps": 2,
            "controlled_response_value": controlled_value,
            "default_response_value_after_controlled_call": default_value,
            "controlled_transport": runtime.traces[0].metadata["transport"],
            "applied_steps": runtime.traces[0].applied_controls["inference_steps"],
            "default_restored": default_value == 7.0,
            "passed": controlled_value == 2.0 and default_value == 7.0,
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not payload["passed"]:
            raise SystemExit(1)
    finally:
        process.terminate()
        process.join(timeout=5)


if __name__ == "__main__":
    main()

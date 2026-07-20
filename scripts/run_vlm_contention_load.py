#!/usr/bin/env python3
"""Generate an auditable OpenAI-compatible VLM contention workload."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18070/v1/chat/completions")
    parser.add_argument(
        "--model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "Inspect the robot scene. State whether the manipulation appears safe and "
            "name one visible object. Reply in one short sentence."
        ),
    )
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--interval-sec", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_sec <= 0 or args.interval_sec < 0:
        raise SystemExit("duration-sec must be positive and interval-sec non-negative")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_url = _image_data_url(args.image)
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": int(args.max_tokens),
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    deadline = time.monotonic() + float(args.duration_sec)
    requests = successes = 0
    latencies: list[float] = []
    with args.output_jsonl.open("a", encoding="utf-8", buffering=1) as output:
        while not STOP_REQUESTED and time.monotonic() < deadline:
            started_wall = time.time()
            started = time.perf_counter()
            record: dict[str, Any] = {
                "request_index": requests,
                "started_unix_s": started_wall,
                "image": str(args.image.resolve()),
            }
            request = urllib.request.Request(
                args.endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=float(args.timeout_sec)) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                record["success"] = True
                record["response_model"] = response_payload.get("model")
                record["usage"] = response_payload.get("usage")
                record["content"] = (
                    response_payload.get("choices", [{}])[0].get("message", {}).get("content")
                )
                successes += 1
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                record["success"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record["latency_ms"] = elapsed_ms
            latencies.append(elapsed_ms)
            requests += 1
            output.write(json.dumps(record) + "\n")
            if args.interval_sec > 0 and not STOP_REQUESTED:
                time.sleep(float(args.interval_sec))
    summary = {
        "requests": requests,
        "successes": successes,
        "success_rate": successes / requests if requests else 0.0,
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else None,
        "stopped_by_signal": STOP_REQUESTED,
        "output_jsonl": str(args.output_jsonl),
    }
    print(json.dumps(summary, indent=2))
    return 0 if requests and successes == requests else 2


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    raise SystemExit(main())

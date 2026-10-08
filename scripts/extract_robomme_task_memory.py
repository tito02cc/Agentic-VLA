#!/usr/bin/env python3
"""Extract deployable structured memory from RoboMME initial demonstrations."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:18070/v1/chat/completions"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task names to extract; defaults to every discovered task.",
    )
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument(
        "--input-mode",
        choices=("frames", "video"),
        default="frames",
        help="Send sampled frames or the complete demonstration video.",
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Retry malformed or interrupted structured responses.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep completed memory files and resume only missing episodes.",
    )
    return parser.parse_args()


def sample_video(path: Path, max_frames: int) -> list[str]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"video contains no frames: {path}")
    count = min(max_frames, total)
    indices = sorted({round(index * (total - 1) / max(1, count - 1)) for index in range(count)})
    encoded: list[str] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {index} from {path}")
        ok, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"failed to encode frame {index} from {path}")
        encoded.append(
            "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
        )
    capture.release()
    return encoded


def encode_video(path: Path) -> str:
    return str(path.resolve())


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("memory encoder returned no JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("memory encoder output must be a JSON object")
    return payload


def infer_memory(
    *,
    endpoint: str,
    model: str,
    instruction: str,
    frames: list[str] | None,
    video: str | None,
    timeout_s: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    prompt = (
        "Analyze the ordered demonstration frames and task instruction. Extract "
        "only information observable from them. Return one JSON object with keys: "
        "task_type, task_constraints, demonstrated_sequence, persistent_memory, "
        "and uncertainty. demonstrated_sequence must preserve object identity, "
        "operation order, repetition count, and clockwise/counterclockwise route "
        "when visible. persistent_memory must describe every task-relevant object's "
        "identity using a relation that can be found again in the final/current "
        "scene, such as topmost, bottom-left, or left of another object; labels like "
        "object_1 alone are invalid. If an object is covered or swapped, track the "
        "same physical object through time and report its final relational location. "
        "If a robot manipulates one among repeated objects, identify that object by "
        "its final relational location. Do not invent image coordinates, simulator "
        "state, success labels, or hidden evaluator information. Use at most 12 "
        "compact demonstrated_sequence steps and keep the full JSON concise.\n"
        f"Task instruction: {instruction}\n"
        "The demonstration follows in chronological order. Track the manipulated "
        "object continuously through motion and occlusion."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if video is not None:
        content.append({"type": "video_url", "video_url": {"url": video}})
    elif frames is not None:
        for index, frame in enumerate(frames):
            content.append({"type": "text", "text": f"Frame {index + 1}/{len(frames)}"})
            content.append({"type": "image_url", "image_url": {"url": frame}})
    else:
        raise ValueError("either frames or video must be provided")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative embodied-task memory encoder. "
                            "Output valid JSON only."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    raw = response_payload["choices"][0]["message"]["content"]
    return parse_json_response(raw), raw


def main() -> int:
    args = parse_args()
    if args.max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if args.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_tasks = set(args.tasks) if args.tasks else None
    for summary_path in sorted(args.input_root.glob("*_ep*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task = str(summary["task"])
        episode = int(summary["episode"])
        if selected_tasks is not None and task not in selected_tasks:
            continue
        destination = args.output_root / f"{task}_ep{episode}_memory.json"
        if args.skip_existing and destination.is_file() and destination.stat().st_size > 0:
            print(json.dumps({"task": task, "episode": episode, "status": "skipped"}))
            continue
        demo = summary_path.parent / "initial_demo_front.mp4"
        frames = sample_video(demo, args.max_frames) if args.input_mode == "frames" else None
        video = encode_video(demo) if args.input_mode == "video" else None
        for attempt in range(1, args.max_attempts + 1):
            try:
                memory, raw = infer_memory(
                    endpoint=args.endpoint,
                    model=args.model,
                    instruction=str(summary["instruction"]),
                    frames=frames,
                    video=video,
                    timeout_s=args.timeout_s,
                    max_tokens=args.max_tokens,
                )
                break
            except (ValueError, OSError, TimeoutError) as error:
                if attempt == args.max_attempts:
                    raise
                print(
                    json.dumps(
                        {
                            "task": task,
                            "episode": episode,
                            "status": "retry",
                            "attempt": attempt,
                            "error": str(error),
                        }
                    ),
                    file=sys.stderr,
                )
        output = {
            "protocol": "carve.robomme.structured_task_memory.v1",
            "task": task,
            "episode": episode,
            "source": {
                "instruction": str(summary["instruction"]),
                "initial_demo_video": str(demo),
                "input_mode": args.input_mode,
                "sampled_frames": len(frames) if frames is not None else None,
                "evaluator_or_oracle_fields_used": False,
            },
            "memory": memory,
            "raw_response": raw,
        }
        destination.write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"task": task, "memory": memory}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

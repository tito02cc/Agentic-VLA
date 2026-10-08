#!/usr/bin/env python3
"""Fixed-budget memory-context replay on recorded RGB, never robot actions.

Uses compressed videos, not bit-exact original camera buffers. The three arms
share identical reconstructed images and system prompts. This is development
diagnosis on a previously inspected episode, not a held-out quality estimate.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.benchmarks.robodojo_sorting import SortingWorkingMemory, execution_only_receipt
from agentic_vla.runtime.agent import OpenAICompatibleVisionPlanner
from agentic_vla.toolchain.contracts import PrimitiveOutcome
from run_robodojo_sorting_closed_loop import free_port, wait_ready, write_json
from run_robodojo_pi05_admission import stop_process
from prepare_robodojo_pi05 import PROJECT_ROOT, sha256_file


STEPS = (100, 201, 501, 901)
ARMS = ("legacy", "quarantine", "no_text_memory")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_memory(events, ticket, run_id):
    memory = SortingWorkingMemory()
    memory.reset(run_id)
    latest = None
    current_step = 0
    for event in events:
        payload = event["payload"]
        kind = event["event_type"]
        if kind == "planner_submitted":
            current_step = payload["timestep"]
            if payload["ticket_id"] == ticket:
                break
        elif kind == "planner_result" and payload["accepted"]:
            memory.remember_hypothesis(current_step, payload["decision"]["memory_note"])
        elif kind == "primitive_boundary":
            memory.remember_execution(PrimitiveOutcome(episode_id=run_id, **payload))
        elif kind == "visual_critic":
            latest = {key: payload[key] for key in ("timestep", "expected_outcome", "report", "accepted")}
    return memory, latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary_path = args.source_run / "summary.json"
    source = json.loads(summary_path.read_text())
    session = Path(source["execution_summary_path"]).parent
    transcript = read_jsonl(session / "transcript.jsonl")
    events = read_jsonl(session / "events.jsonl")
    config = json.loads((args.source_run / "config.json").read_text())
    video_paths = dict(zip(("cam_high", "cam_left_wrist", "cam_right_wrist"), source["videos"]))
    for camera, name in zip(video_paths, ("cam_head", "cam_left_wrist", "cam_right_wrist")):
        if name not in video_paths[camera]:
            raise ValueError("unexpected video ordering")
    captures = {name: cv2.VideoCapture(path) for name, path in video_paths.items()}
    image_cache = {}
    def frame(camera, step):
        key = (camera, step)
        if key not in image_cache:
            captures[camera].set(cv2.CAP_PROP_POS_FRAMES, step)
            ok, bgr = captures[camera].read()
            if not ok:
                raise ValueError(f"missing video frame {key}")
            image_cache[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return image_cache[key]
    cases = []
    try:
        for index, row in enumerate(transcript):
            if row["role"] != "user" or row["metadata"]["timestep"] not in STEPS:
                continue
            payload = json.loads(row["content"])
            step = payload["timestep"]
            ticket = row["metadata"]["ticket_id"]
            memory, latest = build_memory(events, ticket, source["run_id"])
            # capture() stores only action-boundary frames in a six-frame deque.
            boundaries = sorted({0, *[e["payload"]["ended_timestep"] for e in events
                if e["event_type"] == "primitive_boundary" and e["payload"]["ended_timestep"] <= step]})[-6:]
            frames = {}
            for t in boundaries:
                rgb = frame("cam_high", t)
                stride = max(1, int(np.ceil(max(rgb.shape[:2]) / 320)))
                frames[f"history_t{t}_cam_high"] = rgb[::stride, ::stride]
            frames.update({f"current_{camera}": frame(camera, step) for camera in captures})
            query = next((x["expected_outcome"] for x in payload["task_plan"]["steps"]
                          if x["stage"] == payload["task_plan"]["active_stage"]), "")
            if latest:
                query += " " + latest["expected_outcome"]
            projected = list(memory.retrieve(query, 8, include_hypotheses=False))
            for record in projected:
                record["receipt"] = execution_only_receipt(record["receipt"])
            if latest and latest["timestep"] == step:
                projected.append({"timestep": step, "authority": "current_unverified_visual_check",
                                  "completion_authorized": False, "check": latest})
            for offset in range(3):
                arm = ARMS[(STEPS.index(step) + offset) % 3]
                case = out / f"t{step}_{arm}"
                case.mkdir()
                user = copy.deepcopy(payload)
                if arm == "quarantine":
                    user["memory_records"] = projected
                    user["last_primitive"] = execution_only_receipt(user["last_primitive"])
                elif arm == "no_text_memory":
                    user["memory_records"] = []
                    user["last_primitive"] = {}
                # Preserve the same prompt instructions to isolate context fields.
                assert transcript[index - 1]["role"] == "system"
                request = {"system_prompt": transcript[index - 1]["content"],
                           "user_prompt": json.dumps(user, ensure_ascii=True, separators=(",", ":"))}
                write_json(case / "request.json", request)
                frame_hashes = {}
                for name, rgb in frames.items():
                    Image.fromarray(rgb).save(case / f"{name}.png")
                    frame_hashes[name] = hashlib.sha256(rgb.tobytes()).hexdigest()
                cases.append({"id": case.name, "step": step, "arm": arm,
                    "request_sha256": sha256_file(case / "request.json"), "frame_sha256": frame_hashes})
    finally:
        for cap in captures.values():
            cap.release()
    assert len(cases) == 12
    frozen_files = [*out.rglob("request.json"), *out.rglob("*.png"), Path(__file__),
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_sorting.py",
        PROJECT_ROOT / "agentic_vla/benchmarks/robodojo_pi05_episode.py"]
    hashes = {str(path): sha256_file(path) for path in frozen_files}
    manifest = {"source_run": str(args.source_run.resolve()), "source_summary_sha256": sha256_file(summary_path),
        "scope": "compressed-video development replay, not bit-exact, no actions or efficacy estimate",
        "model": config["planner"]["model"], "profile": "vision_preserving_nf4", "cases": cases,
        "robot_actions": 0, "max_requests": 12, "frozen_sha256": hashes}
    write_json(out / "manifest.json", manifest)
    if args.prepare_only:
        print("Prepared 12 frozen requests; no model calls", flush=True)
        return
    port = free_port()
    command = ["/home/admin1/miniconda3/envs/g2agent/bin/python",
        str(PROJECT_ROOT / "scripts/serve_carve_vlm_profile.py"), "--model", config["planner"]["model"],
        "--profile", "vision_preserving_nf4", "--vision-only", "--port", str(port),
        "--receipt", str(out / "service_profile.json")]
    process = None
    results = []
    started = time.monotonic()
    try:
        with (out / "service.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        profile = wait_ready(process, port, profile=True)
        if profile["model_id"] != manifest["model"] or profile["profile"] != manifest["profile"]:
            raise ValueError("model/profile mismatch")
        infer = OpenAICompatibleVisionPlanner(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            model=manifest["model"], timeout_s=60, max_tokens=768)
        for case in cases:
            folder = out / case["id"]
            request = json.loads((folder / "request.json").read_text())
            request["frames"] = {name: np.asarray(Image.open(folder / f"{name}.png"))
                                 for name in case["frame_sha256"]}
            start = time.monotonic()
            raw = infer(request)
            try:
                parsed, error = json.loads(raw), None
            except ValueError as exc:
                parsed, error = None, str(exc)
            result = {"id": case["id"], "raw": raw, "parsed": parsed, "parse_error": error,
                      "wall_ms": (time.monotonic() - start) * 1000}
            write_json(folder / "result.json", result)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        stop_process(process)
        write_json(out / "summary.json", {"completed_requests": len(results), "planned_requests": 12,
            "wall_seconds": time.monotonic()-started, "robot_actions": 0,
            "source_drift": [path for path, digest in hashes.items() if sha256_file(Path(path)) != digest],
            "semantic_review": "pending; parse success is not visual correctness"})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded offline Critic checks on recorded real sorting episodes, no actions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.runtime.agent import OpenAICompatibleVisionPlanner
from agentic_vla.toolchain.verifier import (
    GuardedVisualVerifier, VisualVerificationContext, build_visual_verification_request,
)


def load_frames(root, task, phase):
    run = root / task
    with np.load(run / "initial_observation.npz", allow_pickle=False) as data:
        instruction = str(data["instruction"].item())
        frames = {name: data[name].transpose(1, 2, 0).copy()
                  for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    sources = [str((run / "initial_observation.npz").resolve())]
    if phase not in {"initial", "final"} and not (phase.startswith("frame") and phase[5:].isdigit()):
        raise ValueError("phase must be initial, final, or frame followed by a nonnegative index")
    if phase != "initial":
        summary = json.loads((run / "summary.json").read_text())
        sources = summary["videos"]
        for name in frames:
            camera = "cam_head" if name == "cam_high" else name
            video = next(path for path in sources if camera in Path(path).name)
            count = int(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames", "-of", "csv=p=0", video], text=True).strip())
            frame_index = count - 1 if phase == "final" else int(phase[5:])
            if not 0 <= frame_index < count:
                raise ValueError(f"frame {frame_index} outside video with {count} frames")
            png = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", video, "-vf", f"select=eq(n\\,{frame_index})",
                "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"])
            frames[name] = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    return instruction, frames, sources


def select_frames(frames, camera_mode):
    if camera_mode == "head":
        return {"cam_high": frames["cam_high"]}
    if camera_mode == "all":
        return dict(frames)
    raise ValueError("camera mode must be all or head")


def frame_timestep(task, phase):
    if phase == "initial":
        return 0
    if phase == "final":
        return 1000 if task == "organize_native" else 1100
    if phase.startswith("frame") and phase[5:].isdigit():
        return int(phase[5:])
    raise ValueError("invalid recorded phase")


def fixed_cases(include_intermediate=False):
    cases = [
        ("organize_native", "initial", "The mouse is on the mouse pad", "contradicted"),
        ("organize_native", "final", "The mouse is on the mouse pad", "confirmed"),
        ("organize_native", "final", "The drawer is open", "contradicted"),
        ("classify_native", "initial", "All car objects are inside the left basket, all watch objects are inside the middle basket, and all wooden_toy objects are inside the right basket", "contradicted"),
        ("classify_native", "final", "All car objects are inside the left basket, all watch objects are inside the middle basket, and all wooden_toy objects are inside the right basket", "contradicted"),
    ]
    if include_intermediate:
        cases.extend([
            ("organize_native", "frame500", "The mouse is on the mouse pad", None),
            ("organize_native", "frame500", "The drawer is open", None),
            ("classify_native", "frame550", cases[3][2], None),
        ])
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:18170/v1/chat/completions")
    p.add_argument("--model", default="/home/admin1/models/Qwen3-VL-4B-Instruct")
    p.add_argument("--camera-mode", choices=("all", "head"), default="all")
    p.add_argument("--include-intermediate", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(args.endpoint.split("/v1/")[0] + "/carve/profile", timeout=10) as r:
        profile = json.load(r)
    if profile["model_id"] != args.model:
        raise ValueError("model identity mismatch")
    (args.output / "service_profile.json").write_text(json.dumps(profile, indent=2))
    cases = fixed_cases(args.include_intermediate)
    # Reference judgments are review aids, not simulator truth or model input.
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2))
    verifier = GuardedVisualVerifier(OpenAICompatibleVisionPlanner(
        endpoint=args.endpoint, model=args.model, timeout_s=45, max_tokens=768))
    rows = []
    for index, (task, phase, expected, reference) in enumerate(cases):
        folder = args.output / str(index)
        folder.mkdir()
        instruction, frames, sources = load_frames(args.root, task, phase)
        frames = select_frames(frames, args.camera_mode)
        hashes = {}
        for name, frame in frames.items():
            path = folder / f"{name}.png"
            Image.fromarray(frame).save(path)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        context = VisualVerificationContext(task_instruction=instruction, expected_outcome=expected,
            frames={f"current_{k}": v for k, v in frames.items()},
            timestep=frame_timestep(task, phase))
        request = build_visual_verification_request(context)
        (folder / "request.json").write_text(json.dumps({k: v for k, v in request.items() if k != "frames"}, indent=2))
        result = verifier.verify(context)
        row = {"task": task, "phase": phase, "expected_outcome": expected,
               "camera_mode": args.camera_mode, "frame_index": context.timestep,
               "manual_reference": reference, "reference_is_official_truth": False,
               "sources": sources, "input_png_sha256": hashes,
               "report": result.report.to_dict(), "accepted": result.accepted,
               "error": result.error, "raw_output": result.raw_output,
               "elapsed_ms": result.elapsed_ms, "robot_actions_executed": 0}
        (folder / "result.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"cases": rows, "metrics": verifier.metrics(),
        "simulation_episode": False, "camera_mode": args.camera_mode,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request_budget": {"timeout_s": 45, "max_tokens": 768},
        "scope": "fixed development observations and optional same-episode temporal transfer; not a benchmark accuracy estimate"}, indent=2))


if __name__ == "__main__":
    main()

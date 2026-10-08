#!/usr/bin/env python3
"""Finite, action-free visual transport and Critic diagnostic on recorded frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.runtime.agent import OpenAICompatibleVisionPlanner
from agentic_vla.toolchain.verifier import VisualVerificationContext, build_visual_verification_request, parse_visual_verification_report
from agentic_vla.toolchain._json import decode_json_object
from run_robodojo_sorting_closed_loop import free_port, wait_ready, stop_process, write_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "artifacts/robodojo/organize_adapter_audit_20260918")
    parser.add_argument("--profiles", nargs="+", choices=("vision_preserving_nf4", "bf16"), default=["vision_preserving_nf4", "bf16"])
    parser.add_argument("--prompts", nargs="+", choices=("existing", "predicate_only", "localize"), default=["existing", "predicate_only"])
    parser.add_argument("--model", type=Path, default=Path("/home/admin1/models/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--attn-implementation", default=None)
    args = parser.parse_args()
    model = str(args.model.resolve())
    if not args.model.is_dir():
        parser.error("model directory does not exist")
    args.output.mkdir(parents=True, exist_ok=False)
    initial = next((args.source / "agent_on/policy").glob("*initial_observation.npz"))
    with np.load(initial, allow_pickle=False) as data:
        frame = data["cam_high"]
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        instruction = str(data["instruction"].item())
    cases = [("initial", frame.copy(), "contradicted", str(initial)),
             ("off_final", np.asarray(Image.open(args.source / "agent_off/final_head.png").convert("RGB")), "confirmed", str(args.source / "agent_off/final_head.png")),
             ("on_final", np.asarray(Image.open(args.source / "agent_on/final_head.png").convert("RGB")), "contradicted", str(args.source / "agent_on/final_head.png"))]
    write_json(args.output / "plan.json", {"profiles": args.profiles, "model": model,
        "attn_implementation": args.attn_implementation, "max_tokens": 256, "temperature": 0,
        "prompts": args.prompts, "requests": len(args.profiles) * len(args.prompts) * len(cases), "robot_actions": 0,
        "cases": [{"name": n, "reference": r, "source": s} for n, _, r, s in cases],
        "scope": "development diagnosis on known frames, manual image references, NOT held-out accuracy or robot success"})
    for name, frame, _, _ in cases:
        Image.fromarray(frame).save(args.output / f"{name}.png")
    rows = []
    for profile in args.profiles:
        out = args.output / profile
        out.mkdir()
        port = free_port()
        command = ["/home/admin1/miniconda3/envs/g2agent/bin/python", str(PROJECT_ROOT / "scripts/serve_carve_vlm_profile.py"),
            "--model", model, "--profile", profile, "--vision-only", "--port", str(port),
            "--receipt", str(out / "profile.json"), "--vision-audit", str(out / "processor.jsonl")]
        if args.attn_implementation:
            command.extend(["--attn-implementation", args.attn_implementation])
        write_json(out / "launch.json", {"command": command,
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), PROJECT_ROOT / "scripts/serve_carve_vlm_profile.py", PROJECT_ROOT / "agentic_vla/runtime/agent.py")}})
        with (out / "server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8"))
        try:
            service_profile = wait_ready(process, port, profile=True)
            if service_profile["model_id"] != model or service_profile["profile"] != profile:
                raise RuntimeError("served model/profile mismatch")
            infer = OpenAICompatibleVisionPlanner(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
                model=model, max_tokens=256, timeout_s=60)
            for name, frame, reference, source in cases:
                for prompt in args.prompts:
                    context = VisualVerificationContext(task_instruction=instruction,
                        expected_outcome="The mouse is on the mouse pad", frames={"current_cam_high": frame}, timestep=0)
                    request = build_visual_verification_request(context)
                    if prompt == "predicate_only":
                        request["user_prompt"] = json.dumps({"question": "Is the computer mouse physically on top of the mouse pad, or beside it?", "expected_outcome": context.expected_outcome})
                        request["system_prompt"] = ("Inspect the current image, not the intended task. Locate the computer mouse and the flat black mouse pad separately. "
                            "Decide whether the mouse rests on the pad or on the wooden desk beside it. "
                            "Return JSON with status (confirmed if on the pad, contradicted if visibly beside it, inconclusive if occluded), "
                            "confidence (number 0 to 1), observed_outcome (one visible evidence sentence, at most 16 words). No other keys.")
                    elif prompt == "localize":
                        request["system_prompt"] = (
                            "Locate two separate objects in the image: the computer mouse and the mouse pad. "
                            "Return only JSON with mouse_bbox and pad_bbox. Each bounding box is [x_min,y_min,x_max,y_max] "
                            "in image coordinates normalized to 0..1000. Return null for an object you cannot locate. "
                            "Do not judge task completion; report tight boxes of the visible objects only.")
                        request["user_prompt"] = "Locate the computer mouse and the mouse pad."
                    start = time.perf_counter()
                    raw = infer(request)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    try:
                        report = dict(decode_json_object(raw)) if prompt == "localize" else parse_visual_verification_report(raw).to_dict()
                        error = None
                    except (ValueError, TypeError) as exc:
                        report, error = None, str(exc)
                    row = {"model": model, "profile": profile, "case": name, "prompt": prompt, "manual_reference": reference,
                        "elapsed_ms": elapsed_ms,
                        "source": source, "raw_output": raw, "report": report, "error": error,
                        "request": {k: v for k, v in request.items() if k != "frames"}}
                    rows.append(row)
                    write_json(out / f"{name}_{prompt}.json", row)
                    print(json.dumps({k: row[k] for k in ("profile", "case", "prompt", "raw_output", "error")}), flush=True)
        finally:
            stop_process(process)
    write_json(args.output / "summary.json", {"rows": rows, "robot_actions": 0,
        "scope": "development diagnostic only; no completion authority granted"})


if __name__ == "__main__":
    main()

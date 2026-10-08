#!/usr/bin/env python3
"""GPU residency/plan admission; no simulator or robot actions are executed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA/source_starvla"
for path in (ROOT, SOURCE):
    sys.path.insert(0, str(path))

from agentic_vla.runtime import (  # noqa: E402
    AgentIntent, GuardedHighLevelAgent, HighLevelAgentConfig, HighLevelAgentContext,
    PolicyServiceVisionPlanner,
)
from deployment.model_server.policy_wrapper import PolicyServerWrapper  # noqa: E402
from scripts.benchmark_starvla_ddim import _fixed_examples  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--planner-model", required=True)
    parser.add_argument("--task-instruction", required=True,
                        help="Exact official task instruction; do not paraphrase it.")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames, provenance = {}, {}
    for camera in ("head", "left_wrist", "right_wrist"):
        candidates = list(args.video_dir.glob(f"episode_0000000_cam_{camera}_*.mp4"))
        if len(candidates) != 1:
            raise ValueError("Each camera must have exactly one source episode")
        video = candidates[0]
        capture = cv2.VideoCapture(str(video))
        try:
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode {video}")
        finally:
            capture.release()
        resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
        image_path = args.output / f"{camera}.png"
        if not cv2.imwrite(str(image_path), resized):
            raise RuntimeError("Failed to save probe frame")
        frames[camera] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        provenance[camera] = {
            "video": str(video.resolve()), "video_sha256": digest(video),
            "video_frame_index": 0, "image_sha256": digest(image_path),
        }
    manifest = {
        "checkpoint": args.checkpoint, "planner_model": args.planner_model,
        "task_instruction": args.task_instruction, "frames": provenance,
        "claim_boundary": "Offline model admission, not simulator task evaluation.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    wrapper = PolicyServerWrapper(
        args.checkpoint, use_bf16=True, staged_vlm_path=args.planner_model,
    )
    examples = _fixed_examples()
    wrapper.predict_action(examples, action_seed=31, num_inference_steps=4)
    before = wrapper.predict_action(examples, action_seed=31, num_inference_steps=4)["actions"]
    np.save(args.output / "synthetic_actions_before.npy", before)
    receipts, rounds = [], []

    def transport(payload):
        receipt = wrapper.semantic_decision(**payload)
        receipts.append(receipt)
        (args.output / "residency_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        return {"ok": True, "data": receipt}

    for index in range(2):
        context = HighLevelAgentContext(
            task_instruction=args.task_instruction,
            trigger="task_start", episode_id=f"offline-staged-admission-{index}", timestep=0,
            frames=frames, robot_state=(),
            allowed_intents=(AgentIntent.VLA_ACT, AgentIntent.SAFE_STOP),
            available_skills=(), remaining_retries=1, remaining_recoveries=0,
        )
        agent = GuardedHighLevelAgent(
            PolicyServiceVisionPlanner(transport, max_tokens=512, max_images=3),
            HighLevelAgentConfig(max_calls_per_episode=2, max_grounding_repairs=1),
        )
        cpu_rng = torch.random.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state().clone()
        result = agent.decide(context)
        rng_preserved = bool(torch.equal(cpu_rng, torch.random.get_rng_state()) and
                             torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        after = wrapper.predict_action(examples, action_seed=31, num_inference_steps=4)["actions"]
        np.save(args.output / f"synthetic_actions_after{index}.npy", after)
        row = {
            "index": index, "accepted": result.accepted, "error": result.error,
            "decision": result.decision.to_dict(), "raw_outputs": list(result.raw_outputs),
            "validation_errors": list(result.validation_errors),
            "attempt_count": result.attempt_count, "rng_preserved": rng_preserved,
            "action_max_abs_difference": float(np.max(np.abs(after - before))),
            "actions_exactly_equal": bool(np.array_equal(before, after)),
        }
        rounds.append(row)
        (args.output / "rounds.json").write_text(json.dumps(rounds, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not result.accepted or not rng_preserved or not row["actions_exactly_equal"]:
            (args.output / "summary.json").write_text(json.dumps({
                **manifest, "accepted": False, "rounds": rounds, "receipts": receipts,
            }, indent=2) + "\n")
            raise RuntimeError("staged Planner failed admission; do not start closed-loop execution")
    summary = {
        "schema_version": "carve.staged-vlm-admission.v1", "accepted": True,
        "checkpoint": args.checkpoint, "planner_model": args.planner_model,
        "task_instruction": args.task_instruction,
        "metadata": wrapper.metadata, "frames": provenance,
        "rounds": rounds, "receipts": receipts,
        "claim_boundary": (
            "GPU model residency and live VLM schema checks only. Action invariance uses the "
            "existing fixed synthetic benchmark input, never executed. Planner sees first recorded "
            "frames of one development rollout, without reward/state truth or invented robot state. "
            "No physics or task-success result. No stable co-residency or semantic correctness claim."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

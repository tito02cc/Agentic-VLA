#!/usr/bin/env python3
"""Action-free, preregistered Critic checks on manually labeled video frames."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.runtime.agent import OpenAICompatibleVisionPlanner
from agentic_vla.toolchain.verifier import (
    GuardedVisualVerifier, VisualVerificationContext,
    build_visual_verification_request, parse_visual_verification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATUSES = {"confirmed", "contradicted", "inconclusive"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path):
    manifest = json.loads(path.read_text())
    cases = []
    for frame in manifest["frames"]:
        image_path = (path.parent / frame["image"]).resolve()
        pixels = np.asarray(Image.open(image_path).convert("RGB"))
        frame_inputs = {frame["camera"]: pixels}
        additional_images = {}
        for camera, source in frame.get("additional_images", {}).items():
            if camera in frame_inputs:
                raise ValueError("duplicate camera in manifest")
            extra_path = (path.parent / source["image"]).resolve()
            if source["frame_index"] != frame["frame_index"]:
                raise ValueError("multi-view evidence must use the same video frame index")
            frame_inputs[camera] = np.asarray(Image.open(extra_path).convert("RGB"))
            additional_images[camera] = {**source, "image": str(extra_path), "image_sha256": sha256(extra_path)}
        for predicate, reference in frame["references"].items():
            if reference not in STATUSES:
                raise ValueError("invalid manual reference")
            context = VisualVerificationContext(
                task_instruction=manifest["task_instruction"],
                expected_outcome=manifest["predicates"][predicate],
                frames=frame_inputs, timestep=frame["frame_index"],
            )
            cases.append({"case_id": f"{frame['id']}__{predicate}",
                "predicate": predicate, "reference": reference,
                "image": str(image_path), "image_sha256": sha256(image_path),
                "video": frame["video"], "frame_index": frame["frame_index"],
                "additional_images": additional_images,
                "context": context})
    ids = [case["case_id"] for case in cases]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("cases must be nonempty and unique")
    if len(ids) != manifest["expected_cases"]:
        raise ValueError("case count does not match frozen protocol")
    return manifest, cases


def score_rows(rows, expected_cases, gate):
    ids = [row["case_id"] for row in rows]
    complete = len(ids) == len(set(ids)) == expected_cases
    visible = [row for row in rows if row["reference"] != "inconclusive"]
    unknown = [row for row in rows if row["reference"] == "inconclusive"]
    valid = lambda row: row["accepted"] and row["raw_report"] is not None
    correct = sum(valid(r) and r["status"] == r["reference"] for r in visible)
    false_confirmations = sum(r["reference"] == "contradicted" and r["status"] == "confirmed" for r in rows)
    unsupported_decisions = sum(r["status"] != "inconclusive" for r in unknown)
    unknown_abstentions = sum(valid(r) and r["status"] == "inconclusive" for r in unknown)
    failures = sum(not valid(r) for r in rows)
    ratio = correct / len(visible) if visible else 0.0
    counts = dict(Counter(f"{r['reference']}->{r['status']}" for r in rows))
    reasons = []
    if not complete:
        reasons.append("missing_or_duplicate_cases")
    if failures:
        reasons.append("schema_transport_or_guard_failure")
    if false_confirmations > gate["max_false_confirmations"]:
        reasons.append("false_confirmation_on_visible_negative")
    if unsupported_decisions > gate["max_unsupported_decisions"]:
        reasons.append("definitive_claim_on_unobservable_reference")
    if ratio < gate["min_visible_correct_fraction"]:
        reasons.append("insufficient_visible_correct_decisions")
    if not unknown:
        reasons.append("no_unobservable_cases")
    return {"cases": len(rows), "complete": complete,
        "reference_counts": dict(Counter(r["reference"] for r in rows)),
        "visible_cases": len(visible), "visible_correct": correct,
        "visible_correct_fraction": ratio,
        "visible_abstentions": sum(r["status"] == "inconclusive" for r in visible),
        "false_confirmations": false_confirmations,
        "unknown_cases": len(unknown), "unknown_valid_abstentions": unknown_abstentions,
        "unsupported_decisions": unsupported_decisions,
        "invalid_cases": failures, "confusion": counts,
        "gate_passed": not reasons, "gate_failures": reasons}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_robodojo_sorting_closed_loop import free_port, stop_process, wait_ready, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", choices=("legacy", "evidence"), default="legacy")
    args = parser.parse_args()
    manifest, cases = load_cases(args.manifest)
    if "verifier_protocol" in manifest and manifest["verifier_protocol"] != args.protocol:
        parser.error("protocol differs from preregistered manifest")
    model = str(args.model.resolve())
    if not args.model.is_dir():
        parser.error("model directory does not exist")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "frozen_manifest.json", manifest)
    port = free_port()
    command = ["/home/admin1/miniconda3/envs/g2agent/bin/python",
        str(PROJECT_ROOT / "scripts/serve_carve_vlm_profile.py"),
        "--model", model, "--profile", "bf16", "--vision-only",
        "--attn-implementation", "sdpa", "--port", str(port),
        "--receipt", str(args.output / "profile.json"),
        "--vision-audit", str(args.output / "processor.jsonl")]
    write_json(args.output / "plan.json", {"model": model, "profile": "bf16", "protocol": args.protocol,
        "max_tokens": 256, "temperature": 0, "thinking": False,
        "manifest_sha256": sha256(args.manifest), "command": command,
        "source_sha256": {str(p): sha256(p) for p in (
            Path(__file__), PROJECT_ROOT / "scripts/serve_carve_vlm_profile.py",
            PROJECT_ROOT / "agentic_vla/toolchain/verifier.py",
            PROJECT_ROOT / "agentic_vla/runtime/agent.py")},
        "video_sha256": {v: sha256(PROJECT_ROOT / v) for v in
            {c["video"] for c in cases} | {s["video"] for c in cases for s in c["additional_images"].values()}},
        "cases": [{k: v for k, v in c.items() if k != "context"} for c in cases],
        "latency_note": "processor audit copies tensors to CPU; not a latency benchmark",
        "robot_actions": 0})
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8"))
    rows = []
    service_error = None
    try:
        profile = wait_ready(process, port, profile=True)
        if profile["model_id"] != model or profile["profile"] != "bf16":
            raise RuntimeError("served model/profile mismatch")
        infer = OpenAICompatibleVisionPlanner(
            endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            model=model, max_tokens=256, timeout_s=90)
        verifier = GuardedVisualVerifier(infer, minimum_confidence=manifest["minimum_confidence"], protocol=args.protocol)
        for case in cases:
            result = verifier.verify(case["context"])
            try:
                raw_report = verifier.parse_report(result.raw_output, case["context"]).to_dict()
            except (ValueError, TypeError):
                raw_report = None
            request = verifier.build_request(case["context"])
            row = {k: v for k, v in case.items() if k != "context"}
            row.update({"status": result.report.status.value, "accepted": result.accepted,
                "raw_output": result.raw_output, "raw_report": raw_report,
                "guarded_report": result.report.to_dict(), "error": result.error,
                "elapsed_ms": result.elapsed_ms,
                "request": {k: v for k, v in request.items() if k != "frames"}})
            rows.append(row)
            write_json(args.output / f"{case['case_id']}.json", row)
            write_json(args.output / "partial.json", {"rows": rows})
            print(json.dumps({k: row[k] for k in ("case_id", "reference", "status", "raw_output", "error")}), flush=True)
    except Exception as exc:
        service_error = repr(exc)
        raise
    finally:
        stop_process(process)
        score = score_rows(rows, manifest["expected_cases"], manifest["gate"])
        if service_error:
            score["gate_passed"] = False
            score["gate_failures"].append("service_failure")
        write_json(args.output / "summary.json", {"model": model, "profile": "bf16", "protocol": args.protocol,
            "score": score, "by_predicate": {
                p: {"cases": len(group), "exact_status_matches": sum(r["accepted"] and r["status"] == r["reference"] for r in group)}
                for p in manifest["predicates"]
                if (group := [r for r in rows if r["predicate"] == p])},
            "service_error": service_error, "rows": rows, "robot_actions": 0,
            "completion_authority_granted": False, "scope": manifest["scope"]})
        print(json.dumps(score), flush=True)


if __name__ == "__main__":
    main()

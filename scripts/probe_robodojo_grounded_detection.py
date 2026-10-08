#!/usr/bin/env python3
"""Fixed read-only localization probe, using existing official RGB recordings."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_vla.toolchain.grounded_detection import GroundedDetectionProvider, DetectorObjectObserver
from agentic_vla.toolchain.grounded_detection import build_identity_request, parse_identity_response
from agentic_vla.toolchain.grounded_detection import build_blind_identity_request, apply_blind_identity_veto
from agentic_vla.toolchain import ObjectEvidenceMemory, ObjectRole, ToolExecutionContext
from prepare_robodojo_pi05 import PROJECT_ROOT, sha256_file


REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
ROLES = (ObjectRole("mouse", "computer mouse"), ObjectRole("pad", "mouse pad"))
STEPS = (75, 175, 201, 275, 375, 575, 875, 901)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(root):
    import cv2

    root.mkdir(parents=True, exist_ok=False)
    source = PROJECT_ROOT / "artifacts/robodojo/organize_tool_return_validation_20260918/run01/summary.json"
    video = next(v for v in json.loads(source.read_text())["videos"] if "cam_head" in v)
    cap = cv2.VideoCapture(video)
    cases = []
    try:
        for step in STEPS:
            cap.set(cv2.CAP_PROP_POS_FRAMES, step)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"missing frame {step}")
            case = root / f"organize_t{step}"
            case.mkdir()
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            image.save(case / "input.png")
            cases.append({"id": case.name, "source": video, "frame_index": step,
                          "input_sha256": sha256_file(case / "input.png"),
                          "split": "known_development" if step in (201, 901) else "same_episode_temporal_probe"})
    finally:
        cap.release()
    for phase in ("initial", "final"):
        source = PROJECT_ROOT / f"artifacts/robodojo/sorting_development_20260915/classify_native/{phase}_head.png"
        case = root / f"absent_{phase}"
        case.mkdir()
        Image.open(source).convert("RGB").save(case / "input.png")
        cases.append({"id": case.name, "source": str(source), "frame_index": None,
                      "input_sha256": sha256_file(case / "input.png"), "split": "absent_role_stress"})
    manifest = {"model_id": "IDEA-Research/grounding-dino-tiny", "revision": REVISION,
        "threshold": 0.25, "text_threshold": 0.25, "nms_iou": 0.5, "precision": "fp32",
        "roles": [dataclasses.asdict(r) for r in ROLES], "cases": cases, "max_model_batches": len(cases),
        "robot_actions": 0, "completion_authority": False,
        "scope": "read-only original RGB; same-episode development, not independent accuracy"}
    write(root / "manifest.json", manifest)
    sheet = Image.new("RGB", (1280, 264 * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for i, case in enumerate(cases):
        x, y = (i % 4) * 320, (i // 4) * 264
        sheet.paste(Image.open(root / case["id"] / "input.png").resize((320, 240)), (x, y))
        draw.text((x + 4, y + 242), case["id"], fill="black")
    sheet.save(root / "inputs_contact.png")
    print("Prepared 10 cases; no model calls", flush=True)


def run(root, checkpoint):
    import torch
    import transformers

    if (root / "run_started.json").exists():
        raise FileExistsError("do not overwrite a previous attempt")
    manifest = json.loads((root / "manifest.json").read_text())
    for case in manifest["cases"]:
        if sha256_file(root / case["id"] / "input.png") != case["input_sha256"]:
            raise ValueError("prepared image changed")
    files = [Path(__file__), PROJECT_ROOT / "agentic_vla/toolchain/grounded_detection.py",
             PROJECT_ROOT / "agentic_vla/toolchain/object_memory.py", root / "manifest.json"]
    hashes = {str(p): sha256_file(p) for p in files}
    model_hashes = {p.name: sha256_file(p) for p in sorted(checkpoint.glob("*")) if p.is_file()}
    provider = GroundedDetectionProvider(checkpoint, threshold=manifest["threshold"],
        text_threshold=manifest["text_threshold"], nms_iou=manifest["nms_iou"])
    observer = DetectorObjectObserver(provider)
    write(root / "run_started.json", {"frozen_sha256": hashes, "checkpoint_sha256": model_hashes,
        "checkpoint": str(checkpoint), "torch": torch.__version__, "transformers": transformers.__version__})
    rows = []
    start = time.monotonic()
    try:
        provider.load()
        load_s = time.monotonic() - start
        for case in manifest["cases"]:
            folder = root / case["id"]
            memory = ObjectEvidenceMemory()
            memory.reset(case["id"])
            context = ToolExecutionContext(case["id"], 0, True, ("read_object_evidence",))
            rgb = np.asarray(Image.open(folder / "input.png").convert("RGB"))
            frame = memory.capture("current_cam_high", rgb, context)
            result = observer.observe(memory, frame, ROLES, context=context, current_context=lambda: context)
            record = {"result": result, "memory": memory.read(context), "candidates": provider.last_candidates}
            write(folder / "result.json", record)
            overlay = Image.fromarray(rgb)
            draw = ImageDraw.Draw(overlay)
            current = record["memory"]["current"]
            for obj in current["objects"] if current else ():
                if obj["current_box"] is not None:
                    b = obj["current_box"]
                    color = "red" if obj["role_id"] == "mouse" else "lime"
                    draw.rectangle(b, outline=color, width=2)
                    draw.text((b[0], max(0, b[1]-14)), f"{obj['role_id']} {obj['confidence']:.2f}", fill=color)
            overlay.save(folder / "localization.png")
            rows.append({"case": case["id"], "accepted_schema": result["accepted"],
                         "objects": [] if current is None else current["objects"], "elapsed_ms": result["elapsed_ms"]})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        write(root / "profile.json", {"load_s": load_s, "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
                                      "gate": "pending_visual_review_no_control"})
    finally:
        write(root / "summary.json", {"completed_cases": len(rows), "planned_cases": len(manifest["cases"]),
            "rows": rows, "wall_s": time.monotonic()-start, "robot_actions": 0,
            "source_drift": [p for p, h in hashes.items() if sha256_file(Path(p)) != h]})


def prepare_rejection(root):
    import cv2

    root.mkdir(parents=True, exist_ok=False)
    previous = PROJECT_ROOT / "artifacts/robodojo/organize_grounded_detection_20260918/repaired"
    cases = []
    for name in ("organize_t201", "organize_t275", "organize_t575", "organize_t901", "absent_initial", "absent_final"):
        folder = root / name
        folder.mkdir()
        source = previous / name / "input.png"
        Image.open(source).save(folder / "input.png")
        cases.append({"id": name, "source": str(source), "split": "known_development",
                      "input_sha256": sha256_file(folder / "input.png")})
    for kind, summary, steps in (
        ("organize", "artifacts/robodojo/organize_adapter_audit_20260918/agent_off/summary.json", (109, 319, 719)),
        ("classify", "artifacts/robodojo/sorting_development_20260915/classify_native/summary.json", (215, 615, 915)),
    ):
        source = PROJECT_ROOT / summary
        video = next(v for v in json.loads(source.read_text())["videos"] if "cam_head" in v)
        cap = cv2.VideoCapture(video)
        try:
            for step in steps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, step)
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError("missing predetermined transfer frame")
                folder = root / f"transfer_{kind}_t{step}"
                folder.mkdir()
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(folder / "input.png")
                cases.append({"id": folder.name, "source": video, "frame_index": step,
                    "input_sha256": sha256_file(folder / "input.png"), "split": "new_timestamp_existing_episode"})
        finally:
            cap.release()
    write(root / "manifest.json", {"model_id": "IDEA-Research/grounding-dino-tiny", "revision": REVISION,
        "threshold": .25, "text_threshold": .25, "nms_iou": .5, "precision": "fp32",
        "roles": [dataclasses.asdict(r) for r in ROLES], "cases": cases, "max_model_batches": 12,
        "robot_actions": 0, "completion_authority": False,
        "scope": "six development and six new timestamps; same layouts, not independent holdout"})
    print("Prepared 12 predetermined rejection cases; no model calls", flush=True)


def check_rejection(root, endpoint, model):
    import urllib.request
    from agentic_vla.runtime.agent import OpenAICompatibleVisionPlanner

    out = root / "rejection"
    out.mkdir(exist_ok=False)
    with urllib.request.urlopen(endpoint.split("/v1/")[0] + "/carve/profile", timeout=10) as response:
        profile = json.load(response)
    if profile["model_id"] != model or profile["profile"] != "bf16":
        raise ValueError("model/profile mismatch")
    write(out / "service_profile.json", profile)
    manifest = json.loads((root / "manifest.json").read_text())
    cases = []
    for i, case in enumerate(manifest["cases"]):
        folder = out / case["id"]
        folder.mkdir()
        source = root / case["id"] / "input.png"
        if sha256_file(source) != case["input_sha256"]:
            raise ValueError("detector input changed")
        rgb = np.asarray(Image.open(source).convert("RGB"))
        proposals = json.loads((root / case["id"] / "result.json").read_text())["candidates"]
        identity, catalog = build_identity_request(rgb, proposals,
            threshold=manifest["threshold"], nms_iou=manifest["nms_iou"])
        blind, mapping = build_blind_identity_request(identity, catalog)
        write(folder / "catalog.json", catalog)
        write(folder / "mapping.json", mapping)
        for arm, request in (("conditioned", identity), ("blind", blind)):
            target = folder / arm
            target.mkdir()
            write(target / "request.json", {k: v for k, v in request.items() if k != "frames"})
            for name, image in request["frames"].items():
                Image.fromarray(image).save(target / f"{name}.png")
        cases.append({**case, "order": ("conditioned", "blind") if i % 2 == 0 else ("blind", "conditioned")})
    files = [Path(__file__), PROJECT_ROOT / "agentic_vla/toolchain/grounded_detection.py",
             *out.rglob("*.json"), *out.rglob("*.png")]
    hashes = {str(p): sha256_file(p) for p in files}
    write(out / "manifest.json", {"cases": cases, "frozen_sha256": hashes, "max_requests": 24,
        "model": model, "profile": "bf16", "robot_actions": 0,
        "allowed_entity_types": ["standalone_object"],
        "scope": "paired conditional identity versus additional goal-blind type veto; no actions"})
    infer = OpenAICompatibleVisionPlanner(endpoint=endpoint, model=model, timeout_s=90, max_tokens=768)
    rows, calls = [], 0
    try:
        for case in cases:
            folder = out / case["id"]
            responses = {}
            for arm in case["order"]:
                target = folder / arm
                request = json.loads((target / "request.json").read_text())
                request["frames"] = {name: np.asarray(Image.open(target / f"{name}.png"))
                                     for name in request["required_frame_names"]}
                started = time.monotonic()
                calls += 1
                try:
                    raw, error = infer(request), None
                except Exception as exc:
                    raw, error = None, str(exc)
                result = {"raw": raw, "error": error, "elapsed_ms": (time.monotonic()-started)*1000}
                write(target / "response.json", result)
                responses[arm] = result
            catalog = json.loads((folder / "catalog.json").read_text())
            mapping = json.loads((folder / "mapping.json").read_text())
            row = {"case": case["id"], "split": case["split"], "responses": responses}
            for name in ("conditioned", "vetoed"):
                try:
                    if name == "conditioned":
                        value = parse_identity_response(responses["conditioned"]["raw"], catalog)
                    else:
                        value = apply_blind_identity_veto(responses["conditioned"]["raw"], catalog,
                            responses["blind"]["raw"], mapping,
                            allowed_types={r["role_id"]: frozenset({"standalone_object"}) for r in catalog})
                    row[name] = {"accepted_schema": True, "objects": value["objects"], "error": None}
                except Exception as exc:
                    row[name] = {"accepted_schema": False, "objects": [], "error": str(exc)}
            write(folder / "result.json", row)
            rows.append(row)
            print(json.dumps({"case": case["id"], "conditioned": row["conditioned"], "vetoed": row["vetoed"]}), flush=True)
    finally:
        write(out / "summary.json", {"completed_cases": len(rows), "attempted_requests": calls,
            "planned_requests": 24, "rows": rows, "robot_actions": 0,
            "source_drift": [p for p, h in hashes.items() if sha256_file(Path(p)) != h]})


def resolve_identities(root, endpoint, model):
    import urllib.request
    from agentic_vla.runtime.agent import OpenAICompatibleVisionPlanner
    from agentic_vla.toolchain.object_memory import GuardedObjectObserver

    out = root / "identity"
    out.mkdir(exist_ok=False)
    with urllib.request.urlopen(endpoint.split("/v1/")[0] + "/carve/profile", timeout=10) as response:
        profile = json.load(response)
    if profile["model_id"] != model or profile["profile"] != "bf16":
        raise ValueError("identity model/profile mismatch")
    write(out / "service_profile.json", profile)
    manifest = json.loads((root / "manifest.json").read_text())
    cases = []
    for case in manifest["cases"]:
        folder = out / case["id"]
        folder.mkdir()
        rgb = np.asarray(Image.open(root / case["id"] / "input.png").convert("RGB"))
        if sha256_file(root / case["id"] / "input.png") != case["input_sha256"]:
            raise ValueError("detector input changed")
        record = json.loads((root / case["id"] / "result.json").read_text())
        request, catalog = build_identity_request(rgb, record["candidates"],
            threshold=manifest["threshold"], nms_iou=manifest["nms_iou"])
        write(folder / "request.json", {k: v for k, v in request.items() if k != "frames"})
        write(folder / "catalog.json", catalog)
        for name, image in request["frames"].items():
            Image.fromarray(image).save(folder / f"{name}.png")
        cases.append({"case": case["id"], "images": list(request["frames"])})
    files = [Path(__file__), PROJECT_ROOT / "agentic_vla/toolchain/grounded_detection.py",
             *out.rglob("*.json"), *out.rglob("*.png")]
    hashes = {str(p): sha256_file(p) for p in files}
    write(out / "manifest.json", {"model": model, "profile": "bf16", "cases": cases,
        "frozen_sha256": hashes, "max_requests": len(cases), "robot_actions": 0,
        "protocol": "all proposals, same original pixels plus crops; fixed identity prompt; no task completion"})
    infer = OpenAICompatibleVisionPlanner(endpoint=endpoint, model=model, timeout_s=90, max_tokens=512)
    rows = []
    try:
        for case in cases:
            folder = out / case["case"]
            request = json.loads((folder / "request.json").read_text())
            request["frames"] = {name: np.asarray(Image.open(folder / f"{name}.png")) for name in case["images"]}
            catalog = json.loads((folder / "catalog.json").read_text())
            rgb = request["frames"]["global_rgb"]
            memory = ObjectEvidenceMemory()
            memory.reset(case["case"])
            context = ToolExecutionContext(case["case"], 0, True, ("read_object_evidence",))
            frame = memory.capture("head", rgb, context)
            raw = None
            def identity_provider(unused_request):
                nonlocal raw
                raw = infer(request)
                return parse_identity_response(raw, catalog)
            roles = tuple(ObjectRole(r["role_id"], r["description"]) for r in catalog)
            result = GuardedObjectObserver(identity_provider, minimum_confidence=manifest["threshold"]).observe(
                memory, frame, roles, context=context, current_context=lambda: context)
            row = {"case": case["case"], "raw": raw, "result": result, "memory": memory.read(context)}
            write(folder / "result.json", row)
            overlay = Image.fromarray(rgb)
            draw = ImageDraw.Draw(overlay)
            for obj in row["memory"]["current"]["objects"]:
                if obj["current_box"]:
                    b = obj["current_box"]
                    color = "red" if obj["role_id"] == "mouse" else "lime"
                    draw.rectangle(b, outline=color, width=2)
                    draw.text((b[0], max(0, b[1]-14)), obj["role_id"], fill=color)
            overlay.save(folder / "localization.png")
            rows.append(row)
            print(json.dumps({"case": case["case"], "raw": raw, "accepted": result["accepted"], "error": result["error"]}), flush=True)
    finally:
        write(out / "summary.json", {"completed_requests": len(rows), "planned_requests": len(cases),
            "robot_actions": 0, "rows": rows,
            "source_drift": [p for p, h in hashes.items() if sha256_file(Path(p)) != h],
            "review": "pending visual review; hypothesis only, no control authority"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/admin1/models/grounding-dino-tiny"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-rejection", action="store_true")
    parser.add_argument("--identity-endpoint")
    parser.add_argument("--rejection-endpoint")
    parser.add_argument("--identity-model", default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-9B")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.prepare_only, args.prepare_rejection, args.identity_endpoint, args.rejection_endpoint)) > 1:
        parser.error("select one preparation or VLM mode")
    if args.prepare_only:
        prepare(args.output.resolve())
    elif args.prepare_rejection:
        prepare_rejection(args.output.resolve())
    elif args.identity_endpoint:
        resolve_identities(args.output.resolve(), args.identity_endpoint, args.identity_model)
    elif args.rejection_endpoint:
        check_rejection(args.output.resolve(), args.rejection_endpoint, args.identity_model)
    else:
        run(args.output.resolve(), args.checkpoint.resolve())

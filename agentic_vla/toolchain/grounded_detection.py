"""Read-only open-vocabulary localization behind the existing object observer.

Detector scores are not calibrated truth probabilities. No box produced here
authorizes motion or confirms a task, including when boxes overlap in the image.
"""

from __future__ import annotations

import math
import json
import hashlib
from pathlib import Path

import numpy as np

from ._json import decode_json_object
from .object_memory import GuardedObjectObserver


def filtered_candidates(candidates, shape, *, threshold=0.25, nms_iou=0.5):
    """Validate, clip and suppress duplicate proposals without selecting identity."""
    import torch
    from torchvision.ops import nms

    height, width = shape[:2]
    kept = []
    for candidate in candidates:
        box, score = candidate["box"], candidate["score"]
        if (type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1
                or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("invalid detector candidate")
        clipped = [max(0, min(width, box[0])), max(0, min(height, box[1])),
                   max(0, min(width, box[2])), max(0, min(height, box[3]))]
        if score > threshold and clipped[0] < clipped[2] and clipped[1] < clipped[3]:
            kept.append({**candidate, "box": clipped})
    if kept:
        indices = nms(torch.tensor([c["box"] for c in kept], dtype=torch.float32),
                      torch.tensor([c["score"] for c in kept], dtype=torch.float32), nms_iou).tolist()
        kept = [kept[i] for i in indices]
    return kept


def select_detection(role_id, candidates, shape, *, threshold=0.25, nms_iou=0.5):
    """Do not silently pick an instance when multiple distinct detections remain."""
    kept = filtered_candidates(candidates, shape, threshold=threshold, nms_iou=nms_iou)
    if len(kept) != 1:
        return {"role_id": role_id, "state": "unknown", "box": None, "confidence": 0.0,
                "evidence": "no_detection_not_proven_absent" if not kept else "multiple_instances_require_disambiguation"}
    box = kept[0]["box"]
    return {"role_id": role_id, "state": "located",
            "box": [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])],
            "confidence": float(kept[0]["score"]),
            "evidence": "single_detector_candidate_not_semantic_confirmation"}


def detector_candidates(result):
    """Normalize the empty decode quirk without hiding nonempty mismatches."""
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = list(result["text_labels"])
    # Some tokenizers decode the empty batch as ['']; there are still zero boxes.
    if not boxes and not scores and labels in ([], [""]):
        return []
    if not len(boxes) == len(scores) == len(labels):
        raise ValueError("detector output lengths disagree")
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("invalid detector text label")
    return [{"box": b, "score": s, "label": label} for b, s, label in zip(boxes, scores, labels)]


def build_context_inspection(image, candidates, *, context_scale=4.0, minimum_side=160):
    """Read-only context crop around one proposal, never an identity/completion claim.

    Keep surrounding pixels instead of a tight object crop so a relation remains
    inspectable. No generated pixels, task-specific coordinates, or pad selection.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("inspection needs nonempty HWC uint8 RGB")
    if (type(context_scale) not in (float, int) or not math.isfinite(context_scale)
            or context_scale < 1 or type(minimum_side) is not int or minimum_side < 1):
        raise ValueError("invalid inspection context extent")
    source_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
    record = {"source_rgb_sha256": source_sha256, "source_shape": list(image.shape),
        "authority": "read_only_local_view_not_identity_or_relation_confirmation",
        "context_scale": context_scale, "minimum_side": minimum_side}
    selection = select_detection("inspection_anchor", candidates, image.shape)
    if selection["state"] != "located":
        return {**record, "available": False, "reason": selection["evidence"]}, None
    x1, y1, x2, y2 = selection["box"]
    height, width = image.shape[:2]
    side = int(math.ceil(max(minimum_side, context_scale * max(x2-x1, y2-y1))))
    crop_w, crop_h = min(side, width), min(side, height)
    left = max(0, min(width-crop_w, math.floor((x1+x2-crop_w)/2)))
    top = max(0, min(height-crop_h, math.floor((y1+y2-crop_h)/2)))
    box = [left, top, left+crop_w, top+crop_h]
    crop = image[top:top+crop_h, left:left+crop_w].copy()
    return {**record, "available": True, "anchor_box": selection["box"], "crop_box": box,
        "crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest()}, crop


def build_identity_request(image, proposals, *, threshold=0.25, nms_iou=0.5):
    """Ask for identity, not placement. Crops only expose the original RGB pixels.

    Experimental read-only composition; returned coordinates are hypotheses, not
    actions. Retain every request/response when evaluating this optional path.
    """
    from .object_memory import ObjectRole, _roles
    _roles(tuple(ObjectRole(r["role_id"], r["caption"]) for r in proposals))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("identity resolver needs HWC uint8 RGB")
    frames, catalog, visible = {"global_rgb": image.copy()}, [], []
    for index, role in enumerate(proposals):
        kept = filtered_candidates(role["candidates"], image.shape, threshold=threshold, nms_iou=nms_iou)
        if len(kept) > 4:
            raise ValueError("too many candidates; do not silently discard ambiguity")
        candidates = []
        for i, item in enumerate(kept):
            box = item["box"]
            box = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
            key = f"role{index}_candidate{i}"
            frames[key] = image[box[1]:box[3], box[0]:box[2]].copy()
            candidates.append({"candidate_id": key, "box": box, "score": item["score"]})
        catalog.append({"role_id": role["role_id"], "description": role["caption"], "candidates": candidates})
        visible.append({**catalog[-1], "candidates": [{k: v for k, v in c.items() if k != "score"} for c in candidates]})
    if len(frames) > 9:
        raise ValueError("identity request exceeds image budget")
    request = {
        "system_prompt": (
            "Identify requested physical objects using the global RGB image and candidate crops. "
            "Each crop key is a candidate_id. Candidates come from a fallible detector: even a single "
            "candidate may be the WRONG object. Select only when its visible appearance clearly "
            "matches the role. If none match, multiple match, or identity is unclear, return null. "
            "Do not infer task completion, spatial relations, hidden objects, or actions. "
            'Return JSON only: {"objects":[{"role_id":"...","candidate_id":null,"evidence":"..."}]}. '
            "Cover every requested role exactly once; use candidate_id from that role only. "
            "Keep evidence below 24 words."
        ),
        "user_prompt": json.dumps({"image_size_wh": [image.shape[1], image.shape[0]], "roles": visible}),
        "frames": frames, "required_frame_names": list(frames),
    }
    return request, catalog


def parse_identity_response(raw, catalog):
    """Reject invented IDs/coordinates; selecting a proposal does not verify it."""
    payload = decode_json_object(raw)
    if not isinstance(payload, dict) or set(payload) != {"objects"}:
        raise ValueError("identity response must contain only objects")
    rows = payload["objects"]
    if not isinstance(rows, list) or len(rows) != len(catalog):
        raise ValueError("identity response must cover every role")
    roles = {r["role_id"]: r for r in catalog}
    found = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role_id", "candidate_id", "evidence"}:
            raise ValueError("invalid identity fields")
        role, cid, evidence = row["role_id"], row["candidate_id"], row["evidence"]
        if not isinstance(role, str) or role not in roles or role in found:
            raise ValueError("unknown or duplicate identity role")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 512:
            raise ValueError("bounded identity evidence required")
        options = {c["candidate_id"]: c for c in roles[role]["candidates"]}
        if cid is not None and (not isinstance(cid, str) or cid not in options):
            raise ValueError("candidate must belong to the requested role")
        selected = options[cid] if cid is not None else None
        found[role] = {"role_id": role, "state": "located" if selected else "unknown",
                       "box": list(selected["box"]) if selected else None,
                       "confidence": selected["score"] if selected else 0.0, "evidence": evidence}
    objects = [found[r["role_id"]] for r in catalog]
    boxes = [tuple(o["box"]) for o in objects if o["box"] is not None]
    for obj in objects:
        if obj["box"] is not None and boxes.count(tuple(obj["box"])) > 1:
            obj.update(state="unknown", box=None, confidence=0.0, evidence="indistinguishable_role_boxes")
    return {"objects": objects}


def build_blind_identity_request(identity_request, catalog):
    """Hide requested identities/scores; preserve the multiset of RGB inputs.

    Only a diagnostic veto source, not an independent truth oracle. The mapping
    stays host-side and must travel with this request, never across frames.
    """
    candidates = [c for role in catalog for c in role["candidates"]]
    candidates.sort(key=lambda c: (c["box"], c["candidate_id"]))
    mapping = {c["candidate_id"]: f"region_{i}" for i, c in enumerate(candidates)}
    frames = {"global_rgb": identity_request["frames"]["global_rgb"].copy()}
    regions = []
    for candidate in candidates:
        key = mapping[candidate["candidate_id"]]
        frames[key] = identity_request["frames"][candidate["candidate_id"]].copy()
        regions.append({"region_id": key, "box": list(candidate["box"])})
    return {
        "system_prompt": (
            "Describe the physical entity shown in each numbered region, using the global RGB "
            "image for context and the corresponding crop for detail. No task or intended object "
            "class is provided. Do not guess what the robot wants to manipulate. Classify each "
            "entity as standalone_object, robot_part, scene_fixture, or unknown. A robot_part is "
            "a component attached to the robot; a scene_fixture is part of the environment. "
            "Use unknown if the crop mixes entities or visible evidence cannot establish identity. "
            'Return JSON only: {"regions":[{"region_id":"...","entity_type":"unknown",'
            '"name":"...","evidence":"..."}]}. '
            "Cover every supplied region exactly once. Keep name under 8 words and evidence under "
            "20 words. Do not propose actions, coordinates, spatial relations, or task completion."
        ),
        "user_prompt": json.dumps({"image_size_wh": [frames["global_rgb"].shape[1],
                                                       frames["global_rgb"].shape[0]], "regions": regions}),
        "frames": frames, "required_frame_names": list(frames),
    }, mapping


ENTITY_TYPES = frozenset({"standalone_object", "robot_part", "scene_fixture", "unknown"})


def parse_blind_identity_response(raw, mapping):
    payload = decode_json_object(raw)
    if not isinstance(payload, dict) or set(payload) != {"regions"}:
        raise ValueError("blind response must contain only regions")
    rows = payload["regions"]
    if not isinstance(rows, list) or len(rows) != len(mapping):
        raise ValueError("blind response must cover every region")
    expected, found = set(mapping.values()), {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"region_id", "entity_type", "name", "evidence"}:
            raise ValueError("invalid blind identity fields")
        key = row["region_id"]
        if not isinstance(key, str) or key not in expected or key in found:
            raise ValueError("unknown or duplicate region")
        if not isinstance(row["entity_type"], str) or row["entity_type"] not in ENTITY_TYPES:
            raise ValueError("invalid entity type")
        for field, limit in (("name", 128), ("evidence", 512)):
            if not isinstance(row[field], str) or not row[field].strip() or len(row[field]) > limit:
                raise ValueError("blind identity text must be bounded")
        found[key] = dict(row)
    return found


def apply_blind_identity_veto(identity_raw, catalog, blind_raw, mapping, *, allowed_types):
    """A host-declared type mismatch can only REMOVE a localization hypothesis.

    Both models remain fallible. This cannot select a new object, confirm a
    relation, or change a task ledger. Caller must bind all inputs to one frame.
    """
    if (set(allowed_types) != {r["role_id"] for r in catalog}
            or any(not isinstance(v, frozenset) or not v or not v <= ENTITY_TYPES - {"unknown"}
                   for v in allowed_types.values())):
        raise ValueError("explicit allowed entity types required for each task role")
    expected_ids = {c["candidate_id"] for r in catalog for c in r["candidates"]}
    if set(mapping) != expected_ids or len(set(mapping.values())) != len(mapping):
        raise ValueError("candidate to region mapping mismatch")
    result = parse_identity_response(identity_raw, catalog)
    blind = parse_blind_identity_response(blind_raw, mapping)
    choices = {r["role_id"]: r["candidate_id"] for r in decode_json_object(identity_raw)["objects"]}
    for obj in result["objects"]:
        if obj["state"] != "located":
            continue
        row = blind[mapping[choices[obj["role_id"]]]]
        if row["entity_type"] not in allowed_types[obj["role_id"]]:
            obj.update(state="unknown", box=None, confidence=0.0,
                       evidence="candidate_vetoed_by_unverified_type_check:" + row["entity_type"])
    return result


class GroundedDetectionProvider:
    """Lazy optional dependency, one independent caption per requested role.

    A host must pin and audit the local checkpoint. Loading arbitrary remote
    code is disabled; the provider never downloads or chooses a model itself.
    """

    def __init__(self, checkpoint, *, device="cuda", threshold=0.25, text_threshold=0.25, nms_iou=0.5):
        for value in (threshold, text_threshold, nms_iou):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("detector thresholds must be in (0,1)")
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_dir():
            raise ValueError("a local detector checkpoint is required")
        self.device, self.threshold, self.text_threshold, self.nms_iou = device, threshold, text_threshold, nms_iou
        self.processor = self.model = None
        self.last_candidates = None

    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.checkpoint, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32).to(self.device).eval()

    def __call__(self, request):
        import torch
        from PIL import Image

        self.last_candidates = None
        frames = request["frames"]
        if len(frames) != 1:
            raise ValueError("detector consumes exactly one public RGB image")
        image = np.asarray(next(iter(frames.values())))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("detector needs HWC uint8 RGB")
        payload = decode_json_object(request["user_prompt"])
        roles = payload.get("roles")
        if (not isinstance(roles, list) or not 1 <= len(roles) <= 8
                or any(not isinstance(r, dict) or set(r) != {"role_id", "description"} for r in roles)):
            raise ValueError("bounded object roles required")
        from .object_memory import ObjectRole, _roles
        _roles(tuple(ObjectRole(**r) for r in roles))
        self.load()
        captions = [r["description"].strip().rstrip(".") + "." for r in roles]
        # Independent captions avoid one class borrowing another class's token score.
        inputs = self.processor(images=[Image.fromarray(image)] * len(roles), text=captions,
                                return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
            threshold=self.threshold, text_threshold=self.text_threshold,
            target_sizes=[image.shape[:2]] * len(roles))
        if len(results) != len(roles):
            raise ValueError("detector result count mismatch")
        objects, raw = [], []
        for role, result in zip(roles, results):
            candidates = detector_candidates(result)
            raw.append({"role_id": role["role_id"], "caption": role["description"], "candidates": candidates})
            objects.append(select_detection(role["role_id"], candidates, image.shape,
                threshold=self.threshold, nms_iou=self.nms_iou))
        # Identical boxes cannot establish two distinct roles such as object and destination.
        boxes = [tuple(o["box"]) for o in objects if o["box"] is not None]
        for obj in objects:
            if obj["box"] is not None and boxes.count(tuple(obj["box"])) > 1:
                obj.update(state="unknown", box=None, confidence=0.0, evidence="indistinguishable_role_boxes")
        self.last_candidates = raw
        return {"objects": objects}


class DetectorObjectObserver(GuardedObjectObserver):
    """Keep detector confidence distinct from a VLM's self-reported certainty."""

    def __init__(self, provider):
        if not isinstance(provider, GroundedDetectionProvider):
            raise TypeError("grounded detector provider required")
        super().__init__(provider, minimum_confidence=provider.threshold)

    def _parse_response(self, raw, roles, shape):
        objects = super()._parse_response(raw, roles, shape)
        for obj in objects:
            obj["confidence_source"] = "detector_matching_score_not_calibrated_probability"
        return objects

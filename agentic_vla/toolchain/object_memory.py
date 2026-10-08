"""Frame-bound object hypotheses, not tracked identities or completed task states."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from agentic_vla.runtime.agent import PlannerCallable

from ._json import decode_json_object
from .contracts import ToolEffect, ToolExecutionContext, ToolSpec, validate_tool_input
from .inspection import VisualEvidenceStore, _rgb_hash


@dataclasses.dataclass(frozen=True)
class ObjectRole:
    """A host-declared task role; a repeated role is NOT a persistent object ID."""

    role_id: str
    description: str

    def __post_init__(self):
        for name, value, limit in (("role_id", self.role_id, 64), ("description", self.description, 512)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(f"{name} must be bounded nonempty text")


def _roles(roles):
    if not isinstance(roles, tuple) or not 1 <= len(roles) <= 8:
        raise ValueError("one to eight immutable object roles required")
    if any(not isinstance(role, ObjectRole) for role in roles):
        raise TypeError("invalid object role")
    if len({role.role_id for role in roles}) != len(roles):
        raise ValueError("object roles must be unique")


def _parse(raw, roles, shape, minimum_confidence, coordinate_system):
    payload = decode_json_object(raw)
    if not isinstance(payload, Mapping) or set(payload) != {"objects"}:
        raise ValueError("object response must contain only objects")
    objects = payload["objects"]
    if not isinstance(objects, list) or len(objects) != len(roles):
        raise ValueError("response must cover every role")
    expected, found = {role.role_id for role in roles}, {}
    for row in objects:
        if not isinstance(row, Mapping) or set(row) != {"role_id", "state", "box", "confidence", "evidence"}:
            raise ValueError("invalid object fields")
        role, state = row["role_id"], row["state"]
        if not isinstance(role, str) or role not in expected or role in found:
            raise ValueError("unknown or duplicate object role")
        if state not in ("located", "absent", "unknown"):
            raise ValueError("unsupported object state")
        confidence, evidence, box = row["confidence"], row["evidence"], row["box"]
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be a finite number in [0,1]")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 512:
            raise ValueError("object evidence must be bounded nonempty text")
        if state == "located":
            if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError("located object requires four integer pixel coordinates")
            left, top, right, bottom = box
            width, height = (1000, 1000) if coordinate_system == "normalized_1000" else (shape[1], shape[0])
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError("object box must lie inside the captured frame")
        elif box is not None:
            raise ValueError("absent or unknown objects must not provide a box")
        usable = state == "located" and confidence >= minimum_confidence
        model_box = copy.deepcopy(box)
        if box is not None and coordinate_system == "normalized_1000":
            box = [math.floor(box[0] * shape[1] / 1000), math.floor(box[1] * shape[0] / 1000),
                   math.ceil(box[2] * shape[1] / 1000), math.ceil(box[3] * shape[0] / 1000)]
        found[role] = {
            **copy.deepcopy(row), "box": copy.deepcopy(box), "localization_available": usable,
            "model_box": model_box, "model_coordinate_system": coordinate_system,
            "coordinate_system": "native_pixels",
            "current_box": list(box) if usable else None,
            "authority": "model_localization_hypothesis_only",
        }
    return [found[role.role_id] for role in roles]


class ObjectEvidenceMemory:
    """Bounded episode-local history; every new capture invalidates old positions.

    No task ledger, actuator, procedure writer or auto-association is bound here.
    The host supplies current context and resets at each episode boundary. A
    schema-valid model box is still a hypothesis requiring quality evaluation.
    """

    def __init__(self, *, history_limit=3):
        if type(history_limit) is not int or not 1 <= history_limit <= 16:
            raise ValueError("history_limit must be an integer in [1,16]")
        self._store = VisualEvidenceStore(max_frames=1, max_age_steps=0)
        self._history = deque(maxlen=history_limit)
        self._active = None
        self._context = None
        self._result = None
        self._episode_id = None
        self._latest_read_step = -1
        self._invalidated = False
        self._reference_epoch = object()
        self._lock = threading.RLock()

    @property
    def tool_spec(self):
        return ToolSpec(
            name="read_object_evidence",
            description=("Read current frame-bound object hypotheses and bounded observation history. "
                         "Historical boxes are not current positions. Role labels are not tracked "
                         "identities. Nothing here confirms task success or authorizes motion."),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.OBSERVE, max_calls_per_episode=32,
        )

    def read_tool(self, arguments, context):
        validate_tool_input(arguments, self.tool_spec.input_schema)
        return self.read(context)

    def reset(self, episode_id):
        with self._lock:
            self._store.reset(episode_id)
            self._episode_id = episode_id
            self._history.clear()
            self._active = self._context = self._result = None
            self._latest_read_step = -1
            self._invalidated = False
            self._reference_epoch = object()

    def pin_reference(self, context):
        """Host-only visual reference from a fresh, accepted observation hypothesis."""
        with self._lock:
            current = self.read(context)["current"]
            if not current or not current["accepted"] or not current["objects"]:
                raise ValueError("reference requires a fresh accepted object observation")
            boxes = []
            for obj in current["objects"]:
                box = obj["current_box"]
                if not obj["localization_available"] or box is None:
                    raise ValueError("reference requires every role to be localized")
                if min(box[2] - box[0], box[3] - box[1]) < 16:
                    raise ValueError("reference crop must be at least 16x16")
                boxes.append(tuple(box))
            if len(set(boxes)) != len(boxes):
                raise ValueError("identical reference boxes do not distinguish roles")
            rgb = self._store.resolve_frame(current["frame"]["frame_id"], context)
            return ObjectReferenceSnapshot(self, self._reference_epoch, current, rgb)

    def _episode(self, context):
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("host tool context required")
        if type(context.timestep) is not int or context.timestep < 0:
            raise ValueError("integer timestep required")
        if type(context.episode_id) is not type(self._episode_id) or context.episode_id != self._episode_id:
            raise ValueError("object memory belongs to another episode; reset first")
        if context.timestep < self._latest_read_step:
            raise ValueError("object memory context moved backwards")

    def capture(self, camera: str, image: np.ndarray, context: ToolExecutionContext) -> dict[str, Any]:
        with self._lock:
            self._episode(context)
            self._invalidated = True
            self._latest_read_step = context.timestep
            frame = self._store.capture(camera, image, context)
            if self._result is not None:
                self._history.append(copy.deepcopy(self._result))
            self._active, self._context, self._result = frame, copy.deepcopy(context), None
            self._latest_read_step = context.timestep
            self._invalidated = False
            return copy.deepcopy(frame)

    def _fresh(self, frame, context):
        self._episode(context)
        if self._invalidated or frame != self._active or context != self._context:
            raise ValueError("object observation changed during inference")

    def _deliver(self, frame, context):
        with self._lock:
            self._fresh(frame, context)
            if self._result is not None:
                raise ValueError("object observation already has a result")
            return self._store.resolve_frame(frame["frame_id"], context)

    def _commit(self, frame, context, *, roles, objects, accepted, error):
        with self._lock:
            self._fresh(frame, context)
            if self._result is not None:
                raise ValueError("object observation already has a result")
            bound = []
            for index, row in enumerate(objects):
                bound.append({**row, "observation_ref": f"{frame['frame_id']}:{index}"})
            self._result = {
                "frame": copy.deepcopy(frame), "roles": [dataclasses.asdict(r) for r in roles],
                "objects": bound, "accepted": accepted, "error": error,
            }

    def measure_image_relation(self, source_role, target_role, context):
        """Measure current image boxes, not physical contact or task completion.

        Identity and localization remain model hypotheses. Geometry cannot
        upgrade them to verified facts or supply a robot-space correction.
        """
        if (not isinstance(source_role, str) or not source_role
                or not isinstance(target_role, str) or not target_role or source_role == target_role):
            raise ValueError("two distinct nonempty object roles required")
        with self._lock:
            current = self.read(context)["current"]
            result = {"source_role": source_role, "target_role": target_role,
                "image_relation": "unknown", "semantic_outcome": "not_verified",
                "authority": "conditional_image_geometry_not_contact_or_control",
                "frame": copy.deepcopy(current["frame"]) if current else None}
            if current is None or not current["accepted"]:
                return {**result, "reason": "no_current_accepted_observation"}
            objects = {obj["role_id"]: obj for obj in current["objects"]}
            if source_role not in objects or target_role not in objects:
                return {**result, "reason": "requested_role_not_observed"}
            source, target = objects[source_role], objects[target_role]
            if not source["localization_available"] or not target["localization_available"]:
                return {**result, "reason": "both_current_localizations_required"}
            a, b = source["current_box"], target["current_box"]
            # Native boxes are rounded to pixels. A one-pixel edge difference
            # cannot establish two distinct entities in almost the same region.
            if max(abs(x-y) for x, y in zip(a, b)) <= 1:
                return {**result, "reason": "indistinguishable_role_boxes_at_pixel_resolution"}
            cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
            inside = b[0] < cx < b[2] and b[1] < cy < b[3]
            outside = cx < b[0] or cx > b[2] or cy < b[1] or cy > b[3]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1]))
            return {**result,
                "image_relation": "center_inside" if inside else "center_outside" if outside else "boundary",
                "source_box": list(a), "target_box": list(b), "source_center_xy": [cx, cy],
                "source_box_overlap_fraction": intersection / ((a[2] - a[0]) * (a[3] - a[1])),
                "signed_center_edge_distance_px": min(cx-b[0], b[2]-cx, cy-b[1], b[3]-cy),
                "observation_refs": [source["observation_ref"], target["observation_ref"]],
                "reason": "axis_aligned_box_measurement_conditioned_on_unverified_localizations"}

    def read(self, context):
        with self._lock:
            self._episode(context)
            self._latest_read_step = context.timestep
            if context != self._context:
                self._invalidated = True
            current = self._result if not self._invalidated else None
            history = list(self._history)
            if self._result is not None and current is None:
                history.append(self._result)
            # Old boxes remain named historical evidence, never current positions.
            historical = []
            for record in history[-self._history.maxlen:]:
                record = copy.deepcopy(record)
                record["freshness"] = "historical_requires_reobservation"
                for obj in record["objects"]:
                    obj["last_observed_box"] = obj.pop("box")
                    obj["current_box"] = None
                    obj["localization_available"] = False
                historical.append(record)
            return {
                "current": copy.deepcopy(current), "history": historical,
                "authority": "observation_only_no_task_confirmation",
                "identity_policy": "role_labels_are_not_cross_frame_object_identity",
            }


class ObjectReferenceSnapshot:
    """Historical appearance only; reset revokes references even for a reused ID.

    Construct through ObjectEvidenceMemory.pin_reference(). The initial semantic
    label is still a model hypothesis, not a verified persistent physical ID.
    """

    def __init__(self, owner, epoch, record, rgb):
        self._owner, self._epoch = owner, epoch
        self._rgb = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.shape)
        self._record = copy.deepcopy(record)

    def _check(self):
        if self._epoch is not self._owner._reference_epoch:
            raise ValueError("object reference revoked by episode reset")

    @property
    def provenance(self):
        with self._owner._lock:
            self._check()
            return {
                "frame": copy.deepcopy(self._record["frame"]),
                "roles": copy.deepcopy(self._record["roles"]),
                "objects": [
                    {"role_id": obj["role_id"], "reference_box": list(obj["box"]),
                     "observation_ref": obj["observation_ref"]}
                    for obj in self._record["objects"]
                ],
                "authority": "historical_model_hypothesis_not_verified_identity",
            }

    def resolve(self, current_frame, roles):
        with self._owner._lock:
            metadata = self.provenance
            reference = metadata["frame"]
            if (type(current_frame["episode_id"]) is not type(reference["episode_id"])
                    or current_frame["episode_id"] != reference["episode_id"]):
                raise ValueError("object reference belongs to another episode")
            if current_frame["timestep"] <= reference["timestep"]:
                raise ValueError("object reference must precede the current observation")
            if current_frame["camera"] != reference["camera"]:
                raise ValueError("reference observer requires the same camera")
            if [dataclasses.asdict(role) for role in roles] != metadata["roles"]:
                raise ValueError("reference roles changed")
            frames = {"reference_global": self._rgb.copy()}
            for obj in metadata["objects"]:
                left, top, right, bottom = obj["reference_box"]
                name = f"reference_crop_{obj['role_id']}"
                frames[name] = self._rgb[top:bottom, left:right].copy()
                obj.update(image_key=name, crop_rgb_sha256=_rgb_hash(frames[name]))
            return frames, metadata


class GuardedObjectObserver:
    """One read-only provider call, tied to an immutable frame and live host context."""

    def __init__(self, infer: PlannerCallable, *, minimum_confidence=0.55, coordinate_system="native_pixels"):
        if not callable(infer):
            raise TypeError("object observer requires a provider")
        if type(minimum_confidence) not in (int, float) or not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum confidence must be a finite number in [0,1]")
        self.infer, self.minimum_confidence = infer, minimum_confidence
        if coordinate_system not in ("native_pixels", "normalized_1000"):
            raise ValueError("coordinate_system must be explicit, not automatically guessed")
        self.coordinate_system = coordinate_system

    def _request(self, image, frame, roles):
        coordinates = (
                "Boxes are integer [left,top,right,bottom] in normalized 0-1000 coordinates, "
                "relative to the full supplied image, NOT native pixels. "
                if self.coordinate_system == "normalized_1000" else
                "Boxes are integer NATIVE image pixels [left,top,right,bottom], right/bottom "
                "exclusive, NOT a normalized 0-1000 scale. "
            )
        return {
                "system_prompt": (
                    "Locate EACH requested physical object in the supplied RGB image. "
                    "Return a compact JSON object only: "
                    '{"objects":[{"role_id":"...","state":"located|absent|unknown",'
                    '"box":[0,0,20,20],"confidence":0.9,"evidence":"short visible description"}]}. '
                    "Cover every role exactly once; no extra fields. " + coordinates +
                    "Bound the named object, not grippers or supporting objects. "
                    "Use box:null when absent or ambiguous. Roles describe intended task objects, "
                    "NOT evidence of placement, support, task completion or stable identity. "
                    "Do not infer hidden poses or return actions. Do not substitute a different "
                    "board for a role. Keep evidence under 16 words."
                ),
                "user_prompt": json.dumps({"image_size_wh": [image.shape[1], image.shape[0]],
                                           "roles": [dataclasses.asdict(role) for role in roles]}, separators=(",", ":")),
                "frames": {frame["camera"]: image}, "required_frame_names": [frame["camera"]],
            }

    def _parse_response(self, raw, roles, shape):
        return _parse(raw, roles, shape, self.minimum_confidence, self.coordinate_system)

    def observe(self, memory: ObjectEvidenceMemory, frame, roles, *, context, current_context: Callable[[], ToolExecutionContext]):
        _roles(roles)
        started = time.perf_counter()
        raw = None
        provider_called = False
        try:
            if current_context() != context:
                raise ValueError("host context changed before object inference")
            image = memory._deliver(frame, context)
            request = self._request(image, frame, roles)
            provider_called = True
            raw = self.infer(request)
            if current_context() != context:
                raise ValueError("host context changed during object inference")
            objects = self._parse_response(raw, roles, frame["shape"])
            if current_context() != context:
                raise ValueError("host context changed during object parsing")
            memory._commit(frame, context, roles=roles, objects=objects, accepted=True, error=None)
            return {"accepted": True, "error": None, "elapsed_ms": (time.perf_counter() - started) * 1000, "raw_output": raw, "provider_called": provider_called}
        except Exception as exc:  # noqa: BLE001 - foreign providers must not promote failed observations.
            error = str(exc)
            # A stale reply cannot overwrite a newer capture or a reset generation.
            try:
                unchanged = current_context() == context
            except Exception:  # noqa: BLE001 - an unavailable live context cannot authorize a commit.
                unchanged = False
            if unchanged:
                try:
                    memory._commit(frame, context, roles=roles, objects=(), accepted=False, error=error)
                except ValueError:
                    pass
            return {"accepted": False, "error": error, "elapsed_ms": (time.perf_counter() - started) * 1000, "raw_output": raw, "provider_called": provider_called}

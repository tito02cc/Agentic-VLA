"""Budgeted read-only reacquisition; no tracker reset or motion authorization."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import Counter

from ._json import _reject_constant, _unique_fields
from .contracts import ToolExecutionContext
from .object_memory import GuardedObjectObserver, ObjectEvidenceMemory, _roles


class NativeObjectObserver(GuardedObjectObserver):
    """Explicit native grounding schema, without fabricated confidence scores."""

    def __init__(self, infer, *, label_to_role, user_prompt, system_prompt="You are a helpful assistant."):
        super().__init__(infer, coordinate_system="normalized_1000")
        if (not isinstance(label_to_role, dict) or not 1 <= len(label_to_role) <= 8
                or any(not isinstance(x, str) or not x.strip() or len(x) > 64
                       for pair in label_to_role.items() for x in pair)
                or len(set(label_to_role.values())) != len(label_to_role)):
            raise ValueError("unique bounded host labels and roles required")
        if any(not isinstance(p, str) or not p.strip() or len(p) > 4096 for p in (user_prompt, system_prompt)):
            raise ValueError("bounded host prompt required")
        self.labels = dict(label_to_role)
        self.user_prompt, self.system_prompt = user_prompt, system_prompt

    def _request(self, image, frame, roles):
        if {r.role_id for r in roles} != set(self.labels.values()):
            raise ValueError("prompt labels must match requested roles")
        return {"system_prompt": self.system_prompt, "user_prompt": self.user_prompt,
                "frames": {frame["camera"]: image}, "required_frame_names": [frame["camera"]]}

    def _parse_response(self, raw, roles, shape):
        if not isinstance(raw, str) or len(raw) > 16384:
            raise ValueError("bounded native text required")
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if (len(lines) < 3 or lines[0].strip().lower() not in ("```", "```json")
                    or lines[-1].strip() != "```" or "```" in "\n".join(lines[1:-1])):
                raise ValueError("invalid JSON fence")
            text = "\n".join(lines[1:-1])
        items = json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
        if not isinstance(items, list) or len(items) > len(roles):
            raise ValueError("bounded grounding list required")
        boxes = {}
        for row in items:
            if not isinstance(row, dict) or set(row) != {"label", "bbox_2d"}:
                raise ValueError("invalid native fields")
            label, box = row["label"], row["bbox_2d"]
            if not isinstance(label, str) or label not in self.labels or self.labels[label] in boxes:
                raise ValueError("unknown or duplicate label")
            if (not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box)
                    or not 0 <= box[0] < box[2] <= 1000 or not 0 <= box[1] < box[3] <= 1000):
                raise ValueError("invalid normalized box")
            boxes[self.labels[label]] = box
        native = {r: [math.floor(b[0] * shape[1] / 1000), math.floor(b[1] * shape[0] / 1000),
                      math.ceil(b[2] * shape[1] / 1000), math.ceil(b[3] * shape[0] / 1000)] for r, b in boxes.items()}
        counts = Counter(tuple(b) for b in native.values())
        objects = []
        for role in roles:
            box = native.get(role.role_id)
            ambiguous = box is not None and counts[tuple(box)] > 1
            usable = box is not None and not ambiguous
            objects.append({"role_id": role.role_id, "state": "located" if usable else "unknown",
                "box": box, "current_box": box if usable else None, "localization_available": usable,
                "confidence": None, "confidence_source": "not_provided_by_native_model",
                "model_box": boxes.get(role.role_id), "model_coordinate_system": "normalized_1000",
                "coordinate_system": "native_pixels", "authority": "model_localization_hypothesis_only",
                "evidence": "duplicate_role_box" if ambiguous else "native_model_box" if usable else "role_omitted_not_confirmed_absent"})
        return objects


class BoundedObjectReobserver:
    """Single in-flight synchronous call, coalescing and episode-local budgets.

    Reply deadlines reject late output but do not cancel a blocking provider.
    Production transports must enforce their own timeout. Reset invalidates
    old work without allowing a second provider call until old work returns.
    """

    def __init__(self, observer, memory, *, max_calls=2, min_interval_steps=48,
                 reply_deadline_s=12.0, clock=time.monotonic):
        if not isinstance(observer, GuardedObjectObserver) or not isinstance(memory, ObjectEvidenceMemory):
            raise TypeError("guarded observer and evidence memory required")
        if type(max_calls) is not int or not 1 <= max_calls <= 32:
            raise ValueError("max_calls must be in [1,32]")
        if type(min_interval_steps) is not int or min_interval_steps < 1:
            raise ValueError("positive step interval required")
        if type(reply_deadline_s) not in (int, float) or not math.isfinite(reply_deadline_s) or reply_deadline_s <= 0:
            raise ValueError("positive finite reply deadline required")
        if not callable(clock):
            raise TypeError("clock required")
        self.observer, self.memory = observer, memory
        self.max_calls, self.interval, self.deadline, self.clock = max_calls, min_interval_steps, reply_deadline_s, clock
        self._lock = threading.RLock()
        self._generation, self._active = 0, None
        self._episode = None
        self._calls, self._step, self._last_call = 0, -1, None

    def reset(self, episode_id):
        with self._lock:
            self.memory.reset(episode_id)
            self._generation += 1
            self._episode = episode_id
            self._calls, self._step, self._last_call = 0, -1, None

    def maybe_observe(self, *, needs_reobservation, camera, image, roles, context, current_context):
        if type(needs_reobservation) is not bool:
            raise TypeError("host diagnostic boolean required")
        _roles(roles)
        if not isinstance(context, ToolExecutionContext):
            raise TypeError("host context required")
        if not callable(current_context):
            raise TypeError("live context callback required")
        with self._lock:
            if (type(context.episode_id) is not type(self._episode) or context.episode_id != self._episode
                    or type(context.timestep) is not int or context.timestep < 0 or context.timestep < self._step):
                raise ValueError("episode or monotonic step mismatch")
            self._step = context.timestep
            self.memory.read(context)
            status = ("not_needed" if not needs_reobservation else "coalesced" if self._active is not None
                      else "budget_exhausted" if self._calls >= self.max_calls
                      else "cooldown" if self._last_call is not None and context.timestep - self._last_call < self.interval
                      else None)
            if status:
                return {"status": status, "provider_called": False, "calls_used": self._calls, "control_enabled": False}
            if current_context() != context:
                return {"status": "stale_before_call", "provider_called": False, "calls_used": self._calls, "control_enabled": False}
            frame = self.memory.capture(camera, image, context)
            token, generation = object(), self._generation
            self._active = token
            self._calls += 1
            used = self._calls
            self._last_call = context.timestep
            start = self.clock()

        def guarded_context():
            with self._lock:
                if self._generation != generation or self.clock() - start >= self.deadline:
                    raise ValueError("reobservation generation changed or reply deadline exceeded")
                if self._step != context.timestep:
                    raise ValueError("new observation step superseded reobservation")
                return current_context()

        try:
            result = self.observer.observe(self.memory, frame, roles, context=context, current_context=guarded_context)
            with self._lock:
                if self._generation != generation or self._step != context.timestep:
                    return {"status": "stale_after_call", "provider_called": result["provider_called"], "calls_used": used, "result": result, "control_enabled": False}
                current = self.memory.read(context)["current"]
                usable = bool(result["accepted"] and current and any(o["localization_available"] for o in current["objects"]))
                return {"status": "hypothesis_available" if usable else "unknown" if result["accepted"] else "rejected",
                        "provider_called": result["provider_called"], "calls_used": used, "result": result, "frame": frame, "control_enabled": False}
        finally:
            with self._lock:
                if self._active is token:
                    self._active = None

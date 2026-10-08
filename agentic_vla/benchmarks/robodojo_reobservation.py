"""Opt-in synchronous observation shadow; never returns robot instructions."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from agentic_vla.runtime.agent import PolicyServiceVisionPlanner
from agentic_vla.toolchain import (
    BoundedObjectReobserver,
    NativeObjectObserver,
    ObjectEvidenceMemory,
    ObjectRole,
    ReferenceConditionedObjectObserver,
    ToolExecutionContext,
)


class RoboDojoReobservationShadow:
    def __init__(self, transport, config, *, episode_id, output_dir):
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=False)
        self.episode_id = episode_id
        self.checkpoints = tuple(config["checkpoints"])
        if (not self.checkpoints or len(self.checkpoints) > 8
                or any(type(s) is not int or s < 0 for s in self.checkpoints)
                or tuple(sorted(set(self.checkpoints))) != self.checkpoints):
            raise ValueError("ordered unique nonnegative checkpoints required")
        self.last_transport = None
        self.use_reference = config.get("use_visual_reference", False)
        if type(self.use_reference) is not bool:
            raise ValueError("use_visual_reference must be boolean")
        self.reference = None
        self.reference_error = None

        def checked_transport(payload):
            response = transport(payload)
            data = response.get("data", {})
            self.last_transport = {"ok": response.get("ok"), "error": response.get("error"),
                "backend": data.get("backend"), "model_path": data.get("model_path"),
                "action_model_restored": data.get("action_model_restored"),
                "timings": data.get("timings"), "cold_load": data.get("cold_load"),
                "residency_mode": data.get("residency_mode"),
                "cpu_mirror_bytes": data.get("cpu_mirror_bytes"),
                "input_tokens": data.get("input_tokens"), "output_tokens": data.get("output_tokens")}
            if response.get("ok") and (data.get("backend") != "cpu_staged_vlm"
                    or data.get("action_model_restored") is not True or data.get("model_path") != config["model"]):
                raise RuntimeError("staged response violates residency/model contract")
            return response

        provider = PolicyServiceVisionPlanner(checked_transport, max_tokens=192,
            max_images=len(config["label_to_role"]) + 2 if self.use_reference else 1)
        self._provider = provider
        self._observer_options = {key: config[key] for key in ("label_to_role", "user_prompt", "system_prompt")}
        self.memory = ObjectEvidenceMemory()
        observer = NativeObjectObserver(provider, label_to_role=config["label_to_role"],
            user_prompt=config["user_prompt"], system_prompt=config["system_prompt"])
        self.roles = tuple(ObjectRole(r, label) for label, r in config["label_to_role"].items())
        self.gate = BoundedObjectReobserver(observer, self.memory, max_calls=config["max_calls"],
            min_interval_steps=config["min_interval_steps"], reply_deadline_s=config["reply_deadline_s"])
        self.gate.reset(episode_id)
        self._closed = False

    def observe(self, image, *, step, current_context):
        if self._closed:
            raise RuntimeError("shadow belongs to a closed episode")
        context = ToolExecutionContext(self.episode_id, step, False, ("read_object_evidence",))
        self.last_transport = None
        started = time.perf_counter()
        reference_used = self.reference is not None
        if self.use_reference and step > self.checkpoints[0] and self.reference is None:
            outcome = {"status": "reference_unavailable", "provider_called": False,
                "control_enabled": False, "error": self.reference_error or "reference checkpoint was missed"}
        else:
            outcome = self.gate.maybe_observe(needs_reobservation=step in self.checkpoints, camera="head",
                image=image, roles=self.roles, context=context, current_context=current_context)
        if self.use_reference and step == self.checkpoints[0] and self.reference is None:
            try:
                if current_context() != context:
                    raise ValueError("context changed before pinning reference")
                self.reference = self.memory.pin_reference(context)
                self.gate.observer = ReferenceConditionedObjectObserver(
                    self._provider, reference=self.reference, **self._observer_options)
            except ValueError as exc:
                self.reference_error = str(exc)
        record = {"episode_id": self.episode_id, "step": step, "outcome": outcome,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "trigger": "registered_observation_checkpoint" if step in self.checkpoints else "freshness_tick",
            "control_enabled": False, "transport": self.last_transport,
            "current_memory": self.memory.read(context)["current"]}
        if self.use_reference:
            record.update(reference_used=reference_used and outcome["provider_called"],
                reference=self.reference.provenance if self.reference is not None else None,
                reference_error=self.reference_error)
        if outcome["provider_called"]:
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise ValueError("original semantic RGB required")
            filename = f"head_{step:06d}.png"
            Image.fromarray(image).save(self.path / filename)
            record.update(image=filename, rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                          shape=list(image.shape), image_sha256=hashlib.sha256((self.path / filename).read_bytes()).hexdigest())
        with (self.path / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        # This method intentionally has no outcome for the action caller to apply.

    def reset(self):
        self.gate.reset(self.episode_id)
        self.reference = None
        self._closed = True
        state = self.memory.read(ToolExecutionContext(self.episode_id, 0, False, ("read_object_evidence",)))
        if state["current"] is not None or state["history"]:
            raise RuntimeError("shadow memory did not reset")
        with (self.path / "reset.json").open("x") as stream:
            json.dump({"episode_id": self.episode_id, "closed": True, "current": None, "history": []}, stream)

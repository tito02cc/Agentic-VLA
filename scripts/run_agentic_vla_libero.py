"""Real LIBERO evaluation entrypoint for Agentic-VLA ablation experiments.

Supported ablation modes (controlled via CLI flags):
  A1. Baseline:            pi0_base (no flags)
  A2. + Transition Agent:  --transition
  A3. + Graph RAG + Memory: --graph-rag
  A4. + Critic/Retry:       --critic

Combinations:
  --transition --graph-rag   (A2+A3)
  --transition --critic       (A2+A4)
  --transition --graph-rag --critic  (Full Agentic-VLA)

Legacy (appendix supplement only):
  --vision-prompt
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import io
import json
import logging
import math
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

import imageio
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolve_openpi_root() -> pathlib.Path:
    candidates = []
    if env_root := os.environ.get("AGENTIC_VLA_OPENPI_ROOT"):
        candidates.append(pathlib.Path(env_root).expanduser())
    candidates.extend(
        (
            pathlib.Path("/home/admin1/ct/openpi-official"),
            PROJECT_ROOT / "openpi",
        )
    )
    for candidate in candidates:
        if (candidate / "src" / "openpi").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Unable to locate a runnable openpi repository. "
        "Set AGENTIC_VLA_OPENPI_ROOT to the official openpi checkout."
    )


OPENPI_ROOT = _resolve_openpi_root()
OPENPI_SRC = OPENPI_ROOT / "src"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"


def _resolve_libero_root() -> pathlib.Path:
    candidates = []
    if env_root := os.environ.get("AGENTIC_VLA_LIBERO_ROOT"):
        candidates.append(pathlib.Path(env_root).expanduser())
    candidates.extend((OPENPI_ROOT / "third_party" / "libero", PROJECT_ROOT / "LIBERO"))
    for candidate in candidates:
        if (candidate / "libero").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Unable to locate a LIBERO-compatible checkout. Set AGENTIC_VLA_LIBERO_ROOT "
        "to the original LIBERO, LIBERO-Plus, or LIBERO-Pro repository root."
    )


LIBERO_SRC = _resolve_libero_root()

for extra_path in (PROJECT_ROOT, OPENPI_SRC, OPENPI_CLIENT_SRC, LIBERO_SRC):
    sys.path.insert(0, str(extra_path))

from agentic_vla.benchmarks.libero_pro import (
    LIBERO_PRO_SUITES,
    build_libero_pro_haa_index,
    validate_libero_pro_root,
)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AgenticVLARealLIBERO")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

VISION_PROMPT_COLORS = [
    np.array([0, 255, 0], dtype=np.float32),
    np.array([255, 80, 0], dtype=np.float32),
    np.array([0, 120, 255], dtype=np.float32),
    np.array([255, 255, 0], dtype=np.float32),
    np.array([255, 0, 255], dtype=np.float32),
    np.array([0, 255, 255], dtype=np.float32),
    np.array([255, 165, 0], dtype=np.float32),
    np.array([138, 43, 226], dtype=np.float32),
]

# ===== Transition Agent Constants (A2) =====
TRANSITION_STALL_THRESHOLD = 0.0015
TRANSITION_STALL_WINDOW = 12
TRANSITION_MIN_CONTROL_STEPS = 60
TRANSITION_PROMPT = "slightly lift and stabilize the gripper while maintaining the current grasp"
TRANSITION_CHUNK_STEPS = 6
MAX_TRANSITIONS_PER_EPISODE = 1

# ===== Critic Agent Constants (A4) =====
CRITIC_CHECK_INTERVAL = 60
CRITIC_MIN_CONTROL_STEPS = 120
CRITIC_MAX_RETRIES = 1
CRITIC_RECOVERY_PROMPT = "slightly lift and stabilize the gripper while maintaining the current grasp"
CRITIC_RECOVERY_STEPS = 6

# ===== BA-Harness Rules =====
BA_RECOVERY_MIN_STEPS = 140
BA_RECOVERY_COOLDOWN_STEPS = 90
BA_POST_RECOVERY_TRANSITION_LOCKOUT_STEPS = 180
BA_MAX_RECOVERIES_PER_EPISODE = 1
BA_TRANSITION_EXTRA_MIN_STEPS = 40
BA_RECOVERY_RISK_THRESHOLD = 0.65
BA_TRANSITION_RISK_THRESHOLD = 0.65
BA_HIGH_RISK_EVENTS = {"stall", "collision", "misgrasp", "slip", "placement_drift"}
BA_PROGRESS_RECOVERY_MIN_STEPS = 80

DEFAULT_VLA_EXPERT = "default_vla_expert"
TRANSITION_EXPERT = "transition_expert"
RECOVERY_EXPERT = "recovery_expert"
REGRASP_EXPERT = "regrasp_expert"
_ALLOWED_EXPERTS = {
    DEFAULT_VLA_EXPERT,
    TRANSITION_EXPERT,
    RECOVERY_EXPERT,
    REGRASP_EXPERT,
}


def _retain_joint_when_planner_deferred(
    routed_transition: Any,
    planner_transition: Any,
) -> Any:
    """Keep an executable decision when an optional planner call is suppressed."""

    if (
        getattr(planner_transition, "ticket", None) is None
        and getattr(planner_transition, "joint", None) is None
    ):
        return routed_transition
    return planner_transition


# ===== Graph RAG + Memory Constants (A3) =====
SCENE_PRIORS = {
    "moka pot": {
        "role": "manipulated_object",
        "priority": 3,
        "z_offset": 0.02,
        "force": "medium_grip",
        "yaw_hint": "approach_from_side",
        "stability_hint": "keep_upright_while_transporting",
    },
    "mug": {
        "role": "manipulated_object",
        "priority": 3,
        "z_offset": 0.012,
        "force": "medium_grip",
        "yaw_hint": "approach_from_side",
        "stability_hint": "keep_upright_while_transporting",
    },
    "red mug": {
        "role": "manipulated_object",
        "priority": 4,
        "z_offset": 0.01,
        "force": "medium_grip",
        "yaw_hint": "approach_from_top",
        "stability_hint": "keep_upright_while_transporting",
    },
    "plate": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.005,
        "force": "light_grip",
        "yaw_hint": "approach_from_side",
    },
    "pan": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.015,
        "force": "medium_grip",
        "yaw_hint": "approach_from_side",
    },
    "kettle": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.02,
        "force": "firm_grip",
        "yaw_hint": "approach_from_side",
        "stability_hint": "keep_upright_while_transporting",
    },
    "bowl": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.01,
        "force": "light_grip",
        "yaw_hint": "approach_from_top",
    },
    "chef knife": {
        "role": "manipulated_object",
        "priority": 2,
        "z_offset": 0.005,
        "force": "careful_grip",
        "yaw_hint": "approach_carefully",
    },
    "wooden cabinet": {
        "role": "target_region",
        "priority": 1,
        "target_hint": "align_with_the_handle_or_opening_before_contact",
        "release_hint": "avoid_premature_release",
    },
    "stove": {
        "role": "target_region",
        "priority": 1,
        "target_hint": "center_over_the_burner_before_release",
        "release_hint": "release_only_after_the_object_is_settled",
    },
    "microwave": {
        "role": "target_container",
        "priority": 1,
        "target_hint": "align_with_the_opening_and_insert_deeply_before_release",
        "release_hint": "close_the_door_only_after_secure_placement",
    },
}

TASK_CONTROL_PROFILES = {
    "default": {
        "transition_min_steps": TRANSITION_MIN_CONTROL_STEPS,
        "transition_stall_window": TRANSITION_STALL_WINDOW,
        "transition_stall_threshold": TRANSITION_STALL_THRESHOLD,
        "ba_recovery_enabled": True,
        "ba_transition_enabled": False,
        "ba_recovery_risk_threshold": BA_RECOVERY_RISK_THRESHOLD,
        "ba_transition_risk_threshold": BA_TRANSITION_RISK_THRESHOLD,
        "ba_transition_events": {"stall", "collision", ""},
        "critic_min_steps": CRITIC_MIN_CONTROL_STEPS,
    },
    "stove": {
        "transition_min_steps": 140,
        "transition_stall_window": 16,
        "transition_stall_threshold": 0.0012,
        "ba_recovery_enabled": True,
        "ba_transition_enabled": True,
        "ba_recovery_risk_threshold": BA_RECOVERY_RISK_THRESHOLD,
        "ba_transition_risk_threshold": BA_TRANSITION_RISK_THRESHOLD,
        "ba_transition_events": {"stall", "collision", ""},
        "critic_min_steps": 180,
    },
    "plate": {
        "transition_min_steps": 140,
        "transition_stall_window": 16,
        "transition_stall_threshold": 0.0012,
        "ba_recovery_enabled": False,
        "ba_transition_enabled": False,
        "ba_recovery_risk_threshold": BA_RECOVERY_RISK_THRESHOLD,
        "ba_transition_risk_threshold": BA_TRANSITION_RISK_THRESHOLD,
        "ba_transition_events": {"stall", "collision", ""},
        "critic_min_steps": 180,
    },
    "microwave": {
        "transition_min_steps": 140,
        "transition_stall_window": 16,
        "transition_stall_threshold": 0.0012,
        "ba_recovery_enabled": False,
        "ba_transition_enabled": False,
        "ba_recovery_risk_threshold": BA_RECOVERY_RISK_THRESHOLD,
        "ba_transition_risk_threshold": BA_TRANSITION_RISK_THRESHOLD,
        "ba_transition_events": {"stall", "collision", ""},
        "critic_min_steps": 180,
    },
}


@dataclasses.dataclass
class RouterDecision:
    expert: str = DEFAULT_VLA_EXPERT
    trigger: str = ""
    reason: str = ""
    request_plan: bool = False
    subgoal: str = "execute"
    prompt_override: str | None = None
    client_hints: dict[str, Any] | None = None


@dataclasses.dataclass
class ExpertState:
    current_expert: str = DEFAULT_VLA_EXPERT
    current_subgoal: str = "execute"
    current_prompt: str = ""
    steps_in_expert: int = 0
    last_trigger: str = ""
    last_reason: str = ""
    agentic_action_scale: float = 1.0
    agentic_apply_steps: int = 0
    planner_cooldown_steps: int = 0
    subgoal_success_window: int = 0

    def reset_for_episode(self, default_prompt: str) -> None:
        self.current_expert = DEFAULT_VLA_EXPERT
        self.current_subgoal = "execute"
        self.current_prompt = default_prompt
        self.steps_in_expert = 0
        self.last_trigger = ""
        self.last_reason = ""
        self.agentic_action_scale = 1.0
        self.agentic_apply_steps = 0
        self.planner_cooldown_steps = 0
        self.subgoal_success_window = 0


@dataclasses.dataclass
class VerifierDecision:
    status: str = "continue"
    reason: str = ""


@dataclasses.dataclass
class EpisodeInstrumentation:
    method_tag: str
    task_suite: str
    task_id: int
    episode_idx: int
    task_description: str
    control_deadline_ms: float
    perturbation: str = "clean"
    episode_start_s: float = dataclasses.field(default_factory=time.perf_counter)
    control_start_s: float | None = None
    first_action_s: float | None = None
    vla_calls: int = 0
    planner_calls: int = 0
    verifier_calls: int = 0
    memory_retrievals: int = 0
    transition_triggers: int = 0
    recoveries_triggered: int = 0
    recoveries_successful: int = 0
    physical_recoveries_triggered: int = 0
    physical_recoveries_verified: int = 0
    physical_recovery_actions: int = 0
    physical_recovery_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    progress_verifier_calls: int = 0
    async_verify_calls: int = 0
    async_prefetch_submitted: int = 0
    async_prefetch_ready: int = 0
    async_prefetch_waited: int = 0
    async_prefetch_discarded: int = 0
    async_prefetch_errors: int = 0
    async_prefetch_actions: int = 0
    async_prefix_mode: str = "off"
    async_prefix_checks: int = 0
    async_prefix_would_accept: int = 0
    async_prefix_would_reject: int = 0
    async_prefix_enforced_rejections: int = 0
    async_prefix_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    timeout_fallback_count: int = 0
    deadline_misses: int = 0
    control_steps: int = 0
    fast_path_steps: int = 0
    control_latency_ms_values: list[float] = dataclasses.field(default_factory=list)
    observation_to_action_ms_values: list[float] = dataclasses.field(default_factory=list)
    task_cycle_ms_values: list[float] = dataclasses.field(default_factory=list)
    vla_control_latency_ms_values: list[float] = dataclasses.field(default_factory=list)
    cached_control_latency_ms_values: list[float] = dataclasses.field(default_factory=list)
    control_stage_ms_values: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    total_commit_steps: int = 0
    commit_events: int = 0
    commit_histogram: dict[str, int] = dataclasses.field(default_factory=dict)
    risk_commit_steps: dict[str, int] = dataclasses.field(default_factory=dict)
    risk_commit_events: dict[str, int] = dataclasses.field(default_factory=dict)
    planner_ms: float = 0.0
    verifier_ms: float = 0.0
    vla_ms: float = 0.0
    vla_latency_ms_values: list[float] = dataclasses.field(default_factory=list)
    blocking_reasoning_ms: float = 0.0
    skipped_vla_calls: int = 0
    lightweight_fallback_count: int = 0
    action_chunk_age_values: list[float] = dataclasses.field(default_factory=list)
    recovery_trigger_s: float | None = None
    reaction_latency_ms_values: list[float] = dataclasses.field(default_factory=list)
    perturbation_trigger_s: float | None = None
    perturbation_reaction_latency_ms_values: list[float] = dataclasses.field(default_factory=list)
    perturbation_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    best_progress: dict[str, Any] | None = None
    final_progress: dict[str, Any] | None = None
    progress_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    semantic_submitted: int = 0
    semantic_mode: str = "off"
    semantic_completed: int = 0
    semantic_errors: int = 0
    semantic_suppressed: int = 0
    semantic_latency_ms_values: list[float] = dataclasses.field(default_factory=list)
    semantic_schedule_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    semantic_observation_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    high_level_agent_events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    high_level_agent_submissions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    high_level_agent_timeouts: int = 0

    def mark_control_start(self) -> None:
        if self.control_start_s is None:
            self.control_start_s = time.perf_counter()

    def mark_first_action(self) -> None:
        if self.first_action_s is None:
            self.first_action_s = time.perf_counter()

    def add_latency(self, kind: str, elapsed_s: float, *, blocking: bool = False) -> None:
        elapsed_ms = float(elapsed_s * 1000.0)
        if kind == "planner":
            self.planner_ms += elapsed_ms
            self.planner_calls += 1
        elif kind == "verifier":
            self.verifier_ms += elapsed_ms
            self.verifier_calls += 1
        elif kind == "vla":
            self.vla_ms += elapsed_ms
            self.vla_calls += 1
            self.vla_latency_ms_values.append(elapsed_ms)
        if blocking:
            self.blocking_reasoning_ms += elapsed_ms

    def record_skipped_vla_call(self, *, action_chunk_age: int | None = None) -> None:
        self.skipped_vla_calls += 1
        if action_chunk_age is not None:
            self.action_chunk_age_values.append(float(action_chunk_age))

    def record_lightweight_fallback(self) -> None:
        self.lightweight_fallback_count += 1

    def record_action_chunk_age(self, action_chunk_age: int | None) -> None:
        if action_chunk_age is not None:
            self.action_chunk_age_values.append(float(action_chunk_age))

    def record_async_prefix_consistency(
        self,
        record: dict[str, Any],
        *,
        ticket_id: str,
        submitted_timestep: int,
        enforced_rejection: bool,
    ) -> None:
        compact = dict(record)
        compact.update(
            {
                "ticket_id": str(ticket_id),
                "submitted_timestep": int(submitted_timestep),
                "enforced_rejection": bool(enforced_rejection),
            }
        )
        self.async_prefix_checks += 1
        if bool(compact.get("accepted")):
            self.async_prefix_would_accept += 1
        else:
            self.async_prefix_would_reject += 1
        if enforced_rejection:
            self.async_prefix_enforced_rejections += 1
        self.async_prefix_records.append(compact)

    def record_commit(self, steps: int, *, risk_bucket: str = "unknown") -> None:
        self.commit_events += 1
        self.total_commit_steps += int(steps)
        step_key = str(int(steps))
        self.commit_histogram[step_key] = int(self.commit_histogram.get(step_key, 0)) + 1
        bucket = str(risk_bucket or "unknown")
        self.risk_commit_steps[bucket] = int(self.risk_commit_steps.get(bucket, 0)) + int(steps)
        self.risk_commit_events[bucket] = int(self.risk_commit_events.get(bucket, 0)) + 1

    def add_control_stage(self, stage: str, elapsed_s: float) -> None:
        self.control_stage_ms_values.setdefault(str(stage), []).append(
            float(elapsed_s * 1000.0)
        )

    def record_control_step(
        self,
        elapsed_s: float,
        *,
        fast_path: bool,
        vla_call: bool = False,
    ) -> None:
        elapsed_ms = float(elapsed_s * 1000.0)
        self.control_steps += 1
        self.control_latency_ms_values.append(elapsed_ms)
        target = (
            self.vla_control_latency_ms_values
            if vla_call
            else self.cached_control_latency_ms_values
        )
        target.append(elapsed_ms)
        if fast_path:
            self.fast_path_steps += 1
        if self.control_deadline_ms > 0 and elapsed_ms > self.control_deadline_ms:
            self.deadline_misses += 1

    def record_reaction_timing(
        self,
        *,
        observation_to_action_s: float,
        task_cycle_s: float,
    ) -> None:
        if observation_to_action_s < 0 or task_cycle_s < 0:
            raise ValueError("reaction timing values must be non-negative")
        self.observation_to_action_ms_values.append(
            float(observation_to_action_s * 1000.0)
        )
        self.task_cycle_ms_values.append(float(task_cycle_s * 1000.0))

    def mark_recovery_triggered(self) -> None:
        self.recoveries_triggered += 1
        self.recovery_trigger_s = time.perf_counter()

    def mark_recovery_started(self) -> None:
        if self.recovery_trigger_s is not None:
            self.reaction_latency_ms_values.append(
                float((time.perf_counter() - self.recovery_trigger_s) * 1000.0)
            )
            self.recovery_trigger_s = None
        if self.perturbation_trigger_s is not None:
            self.perturbation_reaction_latency_ms_values.append(
                float((time.perf_counter() - self.perturbation_trigger_s) * 1000.0)
            )
            self.perturbation_trigger_s = None

    def record_physical_recovery(
        self,
        *,
        outcome: dict[str, Any],
        phase_records: list[dict[str, Any]],
        timestep: int,
    ) -> None:
        self.physical_recoveries_triggered += 1
        if outcome.get("status") == "succeeded":
            self.physical_recoveries_verified += 1
        self.physical_recovery_actions += int(outcome.get("actions_executed", 0))
        self.physical_recovery_events.append(
            {
                "timestep": int(timestep),
                "outcome": dict(outcome),
                "phases": [dict(record) for record in phase_records],
            }
        )

    def record_perturbation_event(self, event: dict[str, Any]) -> None:
        self.perturbation_events.append(event)
        self.perturbation_trigger_s = time.perf_counter()

    def record_progress(self, progress: dict[str, Any] | None, *, timestep: int) -> None:
        if not progress:
            return
        self.progress_verifier_calls += 1
        compact = {
            "kind": progress.get("kind"),
            "phase": progress.get("phase"),
            "completed_count": int(progress.get("completed_count", 0)),
            "completed_objects": list(progress.get("completed_objects", [])),
            "remaining_objects": list(progress.get("remaining_objects", [])),
            "score": float(progress.get("score", 0.0)),
        }
        if "needs_recenter" in progress:
            compact["needs_recenter"] = bool(progress.get("needs_recenter"))
        if progress.get("target_object") is not None:
            compact["target_object"] = progress.get("target_object")
        if progress.get("target_distance_xy") is not None:
            compact["target_distance_xy"] = float(progress.get("target_distance_xy"))
        self.final_progress = compact
        previous_score = float(self.best_progress.get("score", -1.0)) if self.best_progress else -1.0
        if compact["score"] > previous_score:
            self.best_progress = compact
            event = dict(compact)
            event["timestep"] = int(timestep)
            self.progress_events.append(event)

    def record_semantic_schedule(
        self,
        *,
        timestep: int,
        event: str,
        invoke: bool,
        reason: str,
        deadline_slack_ms: float | None,
    ) -> None:
        if invoke:
            self.semantic_submitted += 1
        else:
            self.semantic_suppressed += 1
        self.semantic_schedule_events.append(
            {
                "timestep": int(timestep),
                "event": str(event),
                "invoke": bool(invoke),
                "reason": str(reason),
                "deadline_slack_ms": (
                    float(deadline_slack_ms) if deadline_slack_ms is not None else None
                ),
            }
        )

    def record_semantic_result(self, result: Any, *, collected_timestep: int) -> None:
        elapsed_ms = float(result.elapsed_s * 1000.0)
        self.semantic_completed += 1
        self.semantic_latency_ms_values.append(elapsed_ms)
        if result.error is not None:
            self.semantic_errors += 1
        response = result.response if isinstance(result.response, Mapping) else {}
        semantic_label = _parse_semantic_label(response.get("content"))
        self.semantic_observation_events.append(
            {
                "ticket_id": str(result.context.ticket_id),
                "event": str(result.context.event),
                "submitted_timestep": int(result.context.submitted_timestep),
                "collected_timestep": int(collected_timestep),
                "latency_ms": elapsed_ms,
                "valid": bool(result.valid),
                "error": str(result.error) if result.error is not None else None,
                "content": response.get("content"),
                "protocol_valid": semantic_label is not None,
                "status": semantic_label[0] if semantic_label is not None else None,
                "failure_mode": semantic_label[1] if semantic_label is not None else None,
                "response_model": response.get("model"),
                "usage": response.get("usage"),
            }
        )

    def record_high_level_agent_submission(
        self,
        ticket: Any,
        *,
        timestep: int,
        trigger: str,
    ) -> None:
        """Record an asynchronous planner submission separately from completion."""
        self.planner_calls += 1
        scene_graph = dict(ticket.context.scene_graph)
        entities = scene_graph.get("entities", ())
        relations = scene_graph.get("relations", ())
        self.high_level_agent_submissions.append(
            {
                "ticket_id": str(ticket.ticket_id),
                "timestep": int(timestep),
                "trigger": str(trigger),
                "context_timestep": int(ticket.context.timestep),
                "affordance_retrieval_ids": [
                    str(record.get("experience_id"))
                    for record in ticket.context.affordance_retrievals
                    if record.get("experience_id")
                ],
                "scene_graph_entities": len(entities),
                "scene_graph_relations": len(relations),
                "failure_memory_records": len(ticket.context.memory_records),
            }
        )

    def record_high_level_agent_timeout(
        self,
        ticket: Any,
        *,
        timestep: int,
        trigger: str,
        boundary_timeout_s: float,
    ) -> None:
        self.high_level_agent_timeouts += 1
        self.high_level_agent_events.append(
            {
                "ticket_id": str(ticket.ticket_id),
                "timestep": int(timestep),
                "submitted_timestep": int(ticket.context.timestep),
                "trigger": str(trigger),
                "accepted": False,
                "elapsed_ms": None,
                "error": "planner boundary timeout",
                "awaited_at_safe_boundary": True,
                "boundary_timeout_s": float(boundary_timeout_s),
                "decision_applied": False,
                "decision": None,
            }
        )

    def record_high_level_agent_result(
        self,
        result: Any,
        *,
        timestep: int,
        trigger: str,
        ticket: Any | None = None,
        awaited_at_safe_boundary: bool = False,
        decision_applied: bool = False,
        stale: bool = False,
    ) -> None:
        if ticket is None:
            # Compatibility path for recorded-frame smoke only. Online CARVE
            # evaluation records a submission before the asynchronous result.
            self.planner_calls += 1
        self.planner_ms += float(result.elapsed_ms)
        if ticket is None:
            self.blocking_reasoning_ms += float(result.elapsed_ms)
        self.high_level_agent_events.append(
            {
                "ticket_id": str(ticket.ticket_id) if ticket is not None else None,
                "timestep": int(timestep),
                "submitted_timestep": (
                    int(ticket.context.timestep) if ticket is not None else int(timestep)
                ),
                "trigger": str(trigger),
                "accepted": bool(result.accepted),
                "elapsed_ms": float(result.elapsed_ms),
                "error": result.error,
                "awaited_at_safe_boundary": bool(awaited_at_safe_boundary),
                "decision_applied": bool(decision_applied),
                "stale": bool(stale),
                "decision": result.decision.to_dict(),
            }
        )

    def finish(self, *, success: bool, episode_steps: int, peak_gpu_mem_gb: float | None) -> dict[str, Any]:
        finished_s = time.perf_counter()
        ttfa_ms = None
        if self.control_start_s is not None and self.first_action_s is not None:
            ttfa_ms = float((self.first_action_s - self.control_start_s) * 1000.0)
        reaction_latency_ms = None
        if self.reaction_latency_ms_values:
            reaction_latency_ms = float(np.mean(self.reaction_latency_ms_values))
        perturbation_reaction_latency_ms = None
        if self.perturbation_reaction_latency_ms_values:
            perturbation_reaction_latency_ms = float(np.mean(self.perturbation_reaction_latency_ms_values))
        vla_latency_ms_mean = _safe_mean(self.vla_latency_ms_values)
        vla_latency_ms_p50 = _safe_percentile(self.vla_latency_ms_values, 50)
        vla_latency_ms_p95 = _safe_percentile(self.vla_latency_ms_values, 95)
        vla_latency_ms_p99 = _safe_percentile(self.vla_latency_ms_values, 99)
        action_chunk_age_mean = _safe_mean(self.action_chunk_age_values)
        action_chunk_age_p95 = _safe_percentile(self.action_chunk_age_values, 95)
        control_stage_summary = {
            stage: {
                "mean_ms": _safe_mean(values),
                "p95_ms": _safe_percentile(values, 95),
                "max_ms": _safe_percentile(values, 100),
                "samples": len(values),
            }
            for stage, values in sorted(self.control_stage_ms_values.items())
        }
        return {
            "episode_id": f"task{self.task_id}:trial{self.episode_idx}",
            "method": self.method_tag,
            "task_suite": self.task_suite,
            "task_id": int(self.task_id),
            "episode_idx": int(self.episode_idx),
            "task_description": self.task_description,
            "perturbation": self.perturbation,
            "success": bool(success),
            "episode_steps": int(episode_steps),
            "vla_calls": int(self.vla_calls),
            "full_vla_calls": int(self.vla_calls),
            "skipped_vla_calls": int(self.skipped_vla_calls),
            "lightweight_fallback_count": int(self.lightweight_fallback_count),
            "planner_calls": int(self.planner_calls),
            "high_level_agent": {
                "calls": int(self.planner_calls),
                "submissions": list(self.high_level_agent_submissions),
                "timeouts": int(self.high_level_agent_timeouts),
                "accepted": sum(
                    bool(event.get("accepted")) for event in self.high_level_agent_events
                ),
                "events": list(self.high_level_agent_events),
            },
            "verifier_calls": int(self.verifier_calls),
            "memory_retrievals": int(self.memory_retrievals),
            "transition_triggers": int(self.transition_triggers),
            "recoveries_triggered": int(self.recoveries_triggered),
            "recoveries_successful": int(self.recoveries_successful),
            "physical_recoveries_triggered": int(self.physical_recoveries_triggered),
            "physical_recoveries_verified": int(self.physical_recoveries_verified),
            "physical_recovery_actions": int(self.physical_recovery_actions),
            "physical_recovery_events": list(self.physical_recovery_events),
            "progress_verifier_calls": int(self.progress_verifier_calls),
            "avg_commit_steps": (
                float(self.total_commit_steps / self.commit_events) if self.commit_events else None
            ),
            "commit": {
                "histogram": dict(sorted(self.commit_histogram.items(), key=lambda item: int(item[0]))),
                "avg_by_risk": {
                    bucket: float(self.risk_commit_steps[bucket] / self.risk_commit_events[bucket])
                    for bucket in sorted(self.risk_commit_events)
                    if self.risk_commit_events[bucket]
                },
            },
            "progress": {
                "best": self.best_progress,
                "final": self.final_progress,
                "events": self.progress_events,
            },
            "perturbation_detail": {
                "events": self.perturbation_events,
                "reaction_latency_ms": perturbation_reaction_latency_ms,
            },
            "realtime": {
                "control_deadline_ms": float(self.control_deadline_ms),
                "ttfa_ms": ttfa_ms,
                "reaction_latency_ms": reaction_latency_ms,
                "deadline_miss_rate": float(self.deadline_misses / self.control_steps) if self.control_steps else None,
                "deadline_misses": int(self.deadline_misses),
                "control_steps": int(self.control_steps),
                "control_latency_ms_p50": _safe_percentile(
                    self.control_latency_ms_values, 50
                ),
                "control_latency_ms_p95": _safe_percentile(
                    self.control_latency_ms_values, 95
                ),
                "observation_to_action_ms_p50": _safe_percentile(
                    self.observation_to_action_ms_values, 50
                ),
                "observation_to_action_ms_p95": _safe_percentile(
                    self.observation_to_action_ms_values, 95
                ),
                "task_cycle_ms_p50": _safe_percentile(
                    self.task_cycle_ms_values, 50
                ),
                "task_cycle_ms_p95": _safe_percentile(
                    self.task_cycle_ms_values, 95
                ),
                "vla_control_steps": len(self.vla_control_latency_ms_values),
                "vla_control_latency_ms_p50": _safe_percentile(
                    self.vla_control_latency_ms_values, 50
                ),
                "vla_control_latency_ms_p95": _safe_percentile(
                    self.vla_control_latency_ms_values, 95
                ),
                "vla_control_deadline_misses": sum(
                    value > self.control_deadline_ms
                    for value in self.vla_control_latency_ms_values
                ) if self.control_deadline_ms > 0 else 0,
                "cached_control_steps": len(self.cached_control_latency_ms_values),
                "cached_control_latency_ms_p50": _safe_percentile(
                    self.cached_control_latency_ms_values, 50
                ),
                "cached_control_latency_ms_p95": _safe_percentile(
                    self.cached_control_latency_ms_values, 95
                ),
                "cached_control_deadline_misses": sum(
                    value > self.control_deadline_ms
                    for value in self.cached_control_latency_ms_values
                ) if self.control_deadline_ms > 0 else 0,
                "stages": control_stage_summary,
                "blocking_reasoning_ms": float(self.blocking_reasoning_ms),
                "fast_path_ratio": float(self.fast_path_steps / self.control_steps) if self.control_steps else None,
                "async_verify_calls": int(self.async_verify_calls),
                "async_prefetch": {
                    "submitted": int(self.async_prefetch_submitted),
                    "ready": int(self.async_prefetch_ready),
                    "waited": int(self.async_prefetch_waited),
                    "discarded": int(self.async_prefetch_discarded),
                    "errors": int(self.async_prefetch_errors),
                    "actions_committed": int(self.async_prefetch_actions),
                    "prefix_consistency": {
                        "mode": str(self.async_prefix_mode),
                        "checks": int(self.async_prefix_checks),
                        "would_accept": int(self.async_prefix_would_accept),
                        "would_reject": int(self.async_prefix_would_reject),
                        "enforced_rejections": int(
                            self.async_prefix_enforced_rejections
                        ),
                        "continuous_rms_mean": _safe_mean(
                            [item.get("continuous_rms") for item in self.async_prefix_records]
                        ),
                        "continuous_rms_p95": _safe_percentile(
                            [item.get("continuous_rms") for item in self.async_prefix_records],
                            95,
                        ),
                        "translation_endpoint_l2_p95": _safe_percentile(
                            [
                                item.get("translation_endpoint_l2")
                                for item in self.async_prefix_records
                            ],
                            95,
                        ),
                        "rotation_endpoint_l2_p95": _safe_percentile(
                            [
                                item.get("rotation_endpoint_l2")
                                for item in self.async_prefix_records
                            ],
                            95,
                        ),
                        "gripper_agreement_mean": _safe_mean(
                            [item.get("gripper_agreement") for item in self.async_prefix_records]
                        ),
                        "records": list(self.async_prefix_records),
                    },
                },
                "semantic_observer": {
                    "mode": str(self.semantic_mode),
                    "submitted": int(self.semantic_submitted),
                    "completed": int(self.semantic_completed),
                    "errors": int(self.semantic_errors),
                    "suppressed": int(self.semantic_suppressed),
                    "latency_ms_mean": _safe_mean(self.semantic_latency_ms_values),
                    "latency_ms_p95": _safe_percentile(self.semantic_latency_ms_values, 95),
                    "schedule_events": list(self.semantic_schedule_events),
                    "observations": list(self.semantic_observation_events),
                },
                "timeout_fallback_count": int(self.timeout_fallback_count),
            },
            "latency": {
                "planner_ms": float(self.planner_ms),
                "verifier_ms": float(self.verifier_ms),
                "vla_ms": float(self.vla_ms),
                "vla_model_wall_ms_total": float(self.vla_ms),
                "vla_latency_ms_values": [float(v) for v in self.vla_latency_ms_values],
                "vla_latency_ms_mean": vla_latency_ms_mean,
                "vla_latency_ms_p50": vla_latency_ms_p50,
                "vla_latency_ms_p95": vla_latency_ms_p95,
                "vla_latency_ms_p99": vla_latency_ms_p99,
                "episode_wall_sec": float(finished_s - self.episode_start_s),
            },
            "lightweight": {
                "full_vla_calls": int(self.vla_calls),
                "skipped_vla_calls": int(self.skipped_vla_calls),
                "fallback_count": int(self.lightweight_fallback_count),
                "action_chunk_age_mean": action_chunk_age_mean,
                "action_chunk_age_p95": action_chunk_age_p95,
            },
            "gpu": {
                "peak_mem_gb": peak_gpu_mem_gb,
                "peak_mem_mb": float(peak_gpu_mem_gb * 1024.0) if peak_gpu_mem_gb is not None else None,
            },
        }


def _build_empty_expert_stats() -> dict[str, collections.Counter]:
    return {
        "switch_counts": collections.Counter(),
        "success_counts": collections.Counter(),
        "escalation_counts": collections.Counter(),
        "replan_counts": collections.Counter(),
    }


def _serialize_expert_stats(expert_stats: dict[str, collections.Counter]) -> dict[str, dict[str, int]]:
    return {key: dict(value) for key, value in expert_stats.items()}


def _safe_mean(values: list[float]) -> float | None:
    clean_values = [float(v) for v in values if v is not None]
    return float(np.mean(clean_values)) if clean_values else None


def _safe_percentile(values: list[float], percentile: float) -> float | None:
    clean_values = [float(v) for v in values if v is not None]
    return float(np.percentile(clean_values, percentile)) if clean_values else None


def _parse_semantic_label(content: Any) -> tuple[str, str] | None:
    code_map = {
        "N": ("NOMINAL", "NONE"),
        "A": ("REPLAN", "MISALIGN"),
        "G": ("REPLAN", "MISGRASP"),
        "D": ("REPLAN", "DROP"),
        "C": ("STOP", "COLLISION"),
        "S": ("RECOVER", "STALL"),
        "O": ("REPLAN", "OTHER"),
    }
    statuses = {"NOMINAL", "REPLAN", "RECOVER", "STOP"}
    failures = {"NONE", "MISGRASP", "DROP", "MISALIGN", "COLLISION", "STALL", "OTHER"}
    for line in str(content or "").replace("`", "").splitlines():
        normalized_line = line.strip().upper().rstrip(".")
        if normalized_line == "DECISION: KEEP":
            return "NOMINAL", "NONE"
        if normalized_line == "DECISION: REFRESH":
            return "REPLAN", "OTHER"
        if normalized_line in code_map:
            return code_map[normalized_line]
        parts = [part.strip().upper() for part in line.split("|")]
        if len(parts) == 2 and parts[0] in statuses and parts[1] in failures:
            return parts[0], parts[1]
    return None


def _semantic_image_data_url(image: Any) -> str:
    image_buffer = io.BytesIO()
    imageio.imwrite(
        image_buffer,
        np.asarray(image, dtype=np.uint8),
        format="JPEG",
        quality=85,
    )
    return "data:image/jpeg;base64," + base64.b64encode(
        image_buffer.getvalue()
    ).decode("ascii")


def _semantic_change_map(reference_image: Any, current_image: Any) -> np.ndarray:
    reference = np.asarray(reference_image, dtype=np.int16)
    current = np.asarray(current_image, dtype=np.int16)
    if reference.shape != current.shape:
        raise ValueError("semantic change-map images must have identical shapes")
    return np.clip(np.abs(current - reference) * 8, 0, 255).astype(np.uint8)


def _semantic_change_overlay(reference_image: Any, current_image: Any) -> np.ndarray:
    reference = np.asarray(reference_image, dtype=np.int16)
    current = np.asarray(current_image, dtype=np.uint8)
    if reference.shape != current.shape or current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError("semantic change-overlay images must have identical RGB shapes")
    magnitude = np.max(np.abs(current.astype(np.int16) - reference), axis=-1)
    mask = magnitude > 8
    padded = np.pad(mask, 2)
    dilated = np.zeros_like(mask)
    for y_offset in range(5):
        for x_offset in range(5):
            dilated |= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    overlay = current.copy()
    red = np.asarray([255, 0, 0], dtype=np.float32)
    overlay[dilated] = (
        0.25 * overlay[dilated].astype(np.float32) + 0.75 * red
    ).astype(np.uint8)
    return overlay


def _build_semantic_vlm_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    task = str(request.get("task", "robot manipulation task"))
    event = str(request.get("event", "visual execution anomaly detected"))
    protocol = str(request.get("semantic_protocol", "label_v1"))
    if protocol == "code_v5":
        prompt = (
            "Compare BEFORE and CURRENT object by object. CHANGE OVERLAY is the CURRENT "
            "scene with locally changed pixels highlighted red; if there is no change "
            "it remains an ordinary current scene. Use red only to locate change, then "
            "verify its semantic meaning in BEFORE and CURRENT. First write one short "
            "evidence line of at most 12 words. "
            "The action plan was computed from BEFORE. Any material change to a task "
            "object, target, containment, or robot-object relation requires refresh. "
            "Then write a final line exactly DECISION: REFRESH or DECISION: KEEP. "
            "Output exactly two lines."
        )
    elif protocol == "code_v4":
        prompt = (
            "Compare BEFORE and CURRENT object by object. First write one short "
            "evidence line of at most 12 words naming the concrete visual change, "
            "or say no change. The action plan was computed from BEFORE. Any material "
            "change to a task object, target, containment, or robot-object relation "
            "requires refresh even if the task remains achievable. Then write a final "
            "line exactly DECISION: REFRESH or DECISION: KEEP. Output exactly two lines."
        )
    elif protocol == "code_v3":
        prompt = (
            "You are a semantic plan-validity observer for a robot manipulation "
            "policy. The BEFORE frame is the visual reference used to compute the "
            "current action plan. "
            f"Task: {task}. Runtime signal: {event}; this signal may be a false alarm. "
            "Compare BEFORE with CURRENT and decide whether that existing plan may "
            "continue without a fresh observation and replan. A visible displacement "
            "of a task-relevant movable object, target receptacle, or robot-object "
            "relationship invalidates the old plan even when the overall task remains "
            "achievable; choose A in that case. Ignore only rendering jitter and "
            "visually negligible changes. Reply with exactly one character: N=no "
            "task-relevant change and old plan remains valid, A=pose or scene change "
            "requiring replan, G=misgrasp, D=dropped object, C=collision requiring "
            "stop, S=stall requiring recovery, O=other plan-invalidating change. No "
            "explanation."
        )
    elif protocol == "code_v2":
        prompt = (
            "You are a semantic safety observer for a robot manipulation policy. "
            f"Task: {task}. Runtime signal: {event}; this signal may be a false alarm. "
            "Compare the before and current frames. Reply with exactly one character: "
            "N=no harmful change, A=object or pose misalignment needing replan, "
            "G=misgrasp needing replan, D=dropped object, C=collision requiring stop, "
            "S=stall requiring recovery, O=other harmful change. If the frames are "
            "effectively unchanged or task progress remains safe, choose N. No explanation."
        )
    elif protocol == "label_v1":
        prompt = (
            "You are a semantic safety observer for a robot manipulation policy. "
            f"Task: {task}. Deployable runtime signal: {event}. Compare the before and "
            "current robot-scene frames when both are provided. Reply with one label line "
            "only, for example REPLAN|MISGRASP. Choose the left label from NOMINAL, "
            "REPLAN, RECOVER, STOP and the right label from NONE, MISGRASP, DROP, "
            "MISALIGN, COLLISION, STALL, OTHER. Never print the words STATUS or FAILURE "
            "and do not explain."
        )
    else:
        raise ValueError(f"unsupported semantic protocol: {protocol}")
    content: list[dict[str, Any]] = []
    reference_image = request.get("reference_image")
    if reference_image is not None:
        content.extend(
            [
                {"type": "text", "text": "Before event:"},
                {
                    "type": "image_url",
                    "image_url": {"url": _semantic_image_data_url(reference_image)},
                },
            ]
        )
    if protocol in {"code_v3", "code_v4", "code_v5"}:
        current_image = request["image"]
        content.extend(
            [
                {"type": "text", "text": "Current frame:"},
                {
                    "type": "image_url",
                    "image_url": {"url": _semantic_image_data_url(current_image)},
                },
            ]
        )
        if protocol == "code_v5":
            if reference_image is None:
                raise ValueError("code_v5 requires a reference image")
            content.extend(
                [
                    {"type": "text", "text": "Change overlay:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _semantic_image_data_url(
                                _semantic_change_overlay(reference_image, current_image)
                            )
                        },
                    },
                ]
            )
        content.append({"type": "text", "text": prompt})
    else:
        content.extend(
            [
                {"type": "text", "text": f"Current frame. {prompt}"},
                {
                    "type": "image_url",
                    "image_url": {"url": _semantic_image_data_url(request["image"])},
                },
            ]
        )
    return {
        "model": str(request["model"]),
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(
            request.get(
                "max_tokens",
                48
                if protocol in {"code_v4", "code_v5"}
                else 4
                if protocol in {"code_v2", "code_v3"}
                else 12,
            )
        ),
        "temperature": 0,
    }


def _request_semantic_observation(request: Mapping[str, Any]) -> dict[str, Any]:
    """Call an OpenAI-compatible VLM endpoint with deployable visual evidence."""

    payload = _build_semantic_vlm_payload(request)
    http_request = urllib.request.Request(
        str(request["endpoint"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            http_request,
            timeout=float(request.get("timeout_sec", 30.0)),
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise RuntimeError(f"semantic VLM request failed: {exc}") from exc
    return {
        "model": response_payload.get("model"),
        "usage": response_payload.get("usage"),
        "content": response_payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content"),
    }


def _get_peak_gpu_mem_gb() -> float | None:
    torch_peak_gb = None
    try:
        import torch
        if not torch.cuda.is_available():
            torch_peak_gb = None
        else:
            torch_peak_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    except Exception:
        torch_peak_gb = None
    smi_used_gb = _get_nvidia_smi_used_mem_gb()
    if torch_peak_gb is None or torch_peak_gb <= 0:
        return smi_used_gb
    if smi_used_gb is None:
        return torch_peak_gb
    return float(max(torch_peak_gb, smi_used_gb))


def _get_nvidia_smi_used_mem_gb() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line) / 1024.0)
        except ValueError:
            continue
    return float(max(values)) if values else None


def _reset_peak_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _load_episode_traces(trace_path: pathlib.Path) -> list[dict[str, Any]]:
    traces = []
    if not trace_path.exists():
        return traces
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed episode trace line in %s", trace_path)
    return traces


def _append_episode_trace(trace_path: pathlib.Path, trace: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def _aggregate_episode_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {
            "episodes_with_trace": 0,
            "inference": {},
            "realtime": {},
            "recovery": {},
        }
    realtime = [trace.get("realtime", {}) for trace in traces]
    async_prefetch = [item.get("async_prefetch", {}) for item in realtime]
    prefix_consistency = [
        item.get("prefix_consistency", {}) for item in async_prefetch
    ]
    prefix_records = [
        record
        for item in prefix_consistency
        for record in item.get("records", [])
        if isinstance(record, dict)
    ]
    async_prefetch_summary = {
        key: int(sum(int(item.get(key, 0)) for item in async_prefetch))
        for key in (
            "submitted",
            "ready",
            "waited",
            "discarded",
            "errors",
            "actions_committed",
        )
    }
    async_prefetch_summary["prefix_consistency"] = {
        "checks": int(sum(int(item.get("checks", 0)) for item in prefix_consistency)),
        "would_accept": int(
            sum(int(item.get("would_accept", 0)) for item in prefix_consistency)
        ),
        "would_reject": int(
            sum(int(item.get("would_reject", 0)) for item in prefix_consistency)
        ),
        "enforced_rejections": int(
            sum(int(item.get("enforced_rejections", 0)) for item in prefix_consistency)
        ),
        "continuous_rms_mean": _safe_mean(
            [item.get("continuous_rms") for item in prefix_records]
        ),
        "continuous_rms_p95": _safe_percentile(
            [item.get("continuous_rms") for item in prefix_records], 95
        ),
        "translation_endpoint_l2_p95": _safe_percentile(
            [item.get("translation_endpoint_l2") for item in prefix_records], 95
        ),
        "rotation_endpoint_l2_p95": _safe_percentile(
            [item.get("rotation_endpoint_l2") for item in prefix_records], 95
        ),
        "gripper_agreement_mean": _safe_mean(
            [item.get("gripper_agreement") for item in prefix_records]
        ),
    }
    semantic_observers = [item.get("semantic_observer", {}) for item in realtime]
    semantic_observations = [
        observation
        for item in semantic_observers
        for observation in item.get("observations", [])
        if isinstance(observation, dict)
    ]
    semantic_observer_summary = {
        "submitted": int(
            sum(int(item.get("submitted", 0)) for item in semantic_observers)
        ),
        "completed": int(
            sum(int(item.get("completed", 0)) for item in semantic_observers)
        ),
        "errors": int(sum(int(item.get("errors", 0)) for item in semantic_observers)),
        "suppressed": int(
            sum(int(item.get("suppressed", 0)) for item in semantic_observers)
        ),
        "latency_ms_mean": _safe_mean(
            [item.get("latency_ms") for item in semantic_observations]
        ),
        "latency_ms_p95": _safe_percentile(
            [item.get("latency_ms") for item in semantic_observations], 95
        ),
        "observations": semantic_observations,
    }
    latency = [trace.get("latency", {}) for trace in traces]
    lightweight = [trace.get("lightweight", {}) for trace in traces]
    gpu = [trace.get("gpu", {}) for trace in traces]
    successes = [1.0 if trace.get("success") else 0.0 for trace in traces]
    deadline_miss_rates = [item.get("deadline_miss_rate") for item in realtime]
    blocking_ms = [item.get("blocking_reasoning_ms") for item in realtime]
    ttfa_ms = [item.get("ttfa_ms") for item in realtime]
    reaction_ms = [item.get("reaction_latency_ms") for item in realtime]
    success_under_deadline_values = [
        success * (1.0 - miss)
        for success, miss in zip(successes, deadline_miss_rates)
        if miss is not None
    ]
    recoveries_triggered = sum(int(trace.get("recoveries_triggered", 0)) for trace in traces)
    recoveries_successful = sum(int(trace.get("recoveries_successful", 0)) for trace in traces)
    physical_recoveries_triggered = sum(
        int(trace.get("physical_recoveries_triggered", 0)) for trace in traces
    )
    physical_recoveries_verified = sum(
        int(trace.get("physical_recoveries_verified", 0)) for trace in traces
    )
    physical_recovery_actions = sum(
        int(trace.get("physical_recovery_actions", 0)) for trace in traces
    )
    planner_calls = sum(int(trace.get("planner_calls", 0)) for trace in traces)
    verifier_calls = sum(int(trace.get("verifier_calls", 0)) for trace in traces)
    vla_calls = sum(int(trace.get("vla_calls", 0)) for trace in traces)
    skipped_vla_calls = sum(int(trace.get("skipped_vla_calls", 0)) for trace in traces)
    lightweight_fallbacks = sum(int(trace.get("lightweight_fallback_count", 0)) for trace in traces)
    vla_call_latency_values: list[float] = []
    for item in latency:
        values = item.get("vla_latency_ms_values")
        if isinstance(values, list):
            vla_call_latency_values.extend(float(value) for value in values if value is not None)
    semantic_calls = int(semantic_observer_summary["completed"])
    total_vlm_calls = planner_calls + verifier_calls + semantic_calls
    return {
        "episodes_with_trace": int(len(traces)),
        "inference": {
            "vla_calls_total": int(vla_calls),
            "full_vla_calls_total": int(vla_calls),
            "skipped_vla_calls_total": int(skipped_vla_calls),
            "planner_calls_total": int(planner_calls),
            "verifier_calls_total": int(verifier_calls),
            "semantic_calls_total": int(semantic_calls),
            "vlm_calls_total": int(total_vlm_calls),
            "vla_calls_per_episode": float(vla_calls / len(traces)),
            "full_vla_calls_per_episode": float(vla_calls / len(traces)),
            "skipped_vla_calls_per_episode": float(skipped_vla_calls / len(traces)),
            "planner_calls_per_episode": float(planner_calls / len(traces)),
            "verifier_calls_per_episode": float(verifier_calls / len(traces)),
            "semantic_calls_per_episode": float(semantic_calls / len(traces)),
            "vlm_calls_per_episode": float(total_vlm_calls / len(traces)),
            "planner_ms_per_episode": _safe_mean([item.get("planner_ms") for item in latency]),
            "verifier_ms_per_episode": _safe_mean([item.get("verifier_ms") for item in latency]),
            "vla_ms_per_episode": _safe_mean([item.get("vla_ms") for item in latency]),
            "vla_model_wall_ms_per_episode": _safe_mean(
                [item.get("vla_model_wall_ms_total") for item in latency]
            ),
            "vla_latency_ms_mean": _safe_mean(vla_call_latency_values),
            "vla_latency_ms_p50": _safe_percentile(vla_call_latency_values, 50),
            "vla_latency_ms_p95": _safe_percentile(vla_call_latency_values, 95),
            "vla_latency_ms_p99": _safe_percentile(vla_call_latency_values, 99),
            "episode_wall_sec_mean": _safe_mean([item.get("episode_wall_sec") for item in latency]),
        },
        "lightweight": {
            "full_vla_calls_total": int(vla_calls),
            "skipped_vla_calls_total": int(skipped_vla_calls),
            "skipped_vla_call_ratio": float(skipped_vla_calls / (vla_calls + skipped_vla_calls))
            if (vla_calls + skipped_vla_calls) else None,
            "fallback_count_total": int(lightweight_fallbacks),
            "action_chunk_age_mean": _safe_mean([item.get("action_chunk_age_mean") for item in lightweight]),
            "action_chunk_age_p95": _safe_percentile([item.get("action_chunk_age_p95") for item in lightweight], 95),
        },
        "realtime": {
            "ttfa_ms_mean": _safe_mean(ttfa_ms),
            "ttfa_ms_p90": _safe_percentile(ttfa_ms, 90),
            "ttfa_ms_p99": _safe_percentile(ttfa_ms, 99),
            "reaction_latency_ms_mean": _safe_mean(reaction_ms),
            "reaction_latency_ms_p90": _safe_percentile(reaction_ms, 90),
            "deadline_miss_rate_mean": _safe_mean(deadline_miss_rates),
            "blocking_reasoning_ms_mean": _safe_mean(blocking_ms),
            "fast_path_ratio_mean": _safe_mean([item.get("fast_path_ratio") for item in realtime]),
            "success_under_deadline": _safe_mean(success_under_deadline_values),
            "async_prefetch": async_prefetch_summary,
            "semantic_observer": semantic_observer_summary,
        },
        "recovery": {
            "recoveries_triggered_total": int(recoveries_triggered),
            "recoveries_successful_total": int(recoveries_successful),
            "recovery_precision": float(recoveries_successful / recoveries_triggered)
            if recoveries_triggered else None,
            "physical_recoveries_triggered_total": int(physical_recoveries_triggered),
            "physical_recoveries_verified_total": int(physical_recoveries_verified),
            "physical_recovery_actions_total": int(physical_recovery_actions),
            "physical_verification_rate": float(
                physical_recoveries_verified / physical_recoveries_triggered
            )
            if physical_recoveries_triggered
            else None,
        },
        "gpu": {
            "peak_mem_gb_max": _safe_percentile([item.get("peak_mem_gb") for item in gpu], 100),
        },
    }


def _clone_expert_stats(expert_stats: dict[str, collections.Counter]) -> dict[str, collections.Counter]:
    return {key: collections.Counter(value) for key, value in expert_stats.items()}


def _diff_expert_stats(
    after: dict[str, collections.Counter],
    before: dict[str, collections.Counter],
) -> dict[str, collections.Counter]:
    diff = _build_empty_expert_stats()
    for key in diff:
        diff[key].update(after.get(key, {}))
        diff[key].subtract(before.get(key, {}))
        diff[key] = collections.Counter({name: int(count) for name, count in diff[key].items() if int(count) > 0})
    return diff


def _materialize_expert_stats(serialized: dict[str, dict[str, int]] | None) -> dict[str, collections.Counter]:
    stats = _build_empty_expert_stats()
    if not isinstance(serialized, dict):
        return stats
    for key, values in serialized.items():
        if key in stats and isinstance(values, dict):
            stats[key].update({name: int(count) for name, count in values.items()})
    return stats


def _merge_expert_stats(
    base: dict[str, collections.Counter],
    delta: dict[str, collections.Counter],
) -> dict[str, collections.Counter]:
    merged = _clone_expert_stats(base)
    for key in merged:
        merged[key].update(delta.get(key, {}))
    return merged


def _record_expert_switch(expert_stats: dict[str, collections.Counter], expert_name: str) -> None:
    if expert_name:
        expert_stats["switch_counts"][expert_name] += 1


def _route_expert(
    *,
    failure_event: str = "",
    reason: str = "",
    expert_state: ExpertState | None = None,
    critic_stuck: bool = False,
    request_plan: bool = False,
) -> RouterDecision:
    failure_event = str(failure_event or "").strip().lower()
    current_expert = expert_state.current_expert if expert_state is not None else DEFAULT_VLA_EXPERT
    if failure_event == "state_gap":
        return RouterDecision(
            expert=TRANSITION_EXPERT,
            trigger="state_gap",
            reason=reason or "transition agent detected state gap",
            request_plan=request_plan,
            subgoal="recover",
        )
    if failure_event in {"stall", "collision"} or critic_stuck:
        return RouterDecision(
            expert=RECOVERY_EXPERT,
            trigger=failure_event or "critic",
            reason=reason,
            request_plan=request_plan,
            subgoal="recover",
        )
    if failure_event in {"misgrasp", "slip"}:
        return RouterDecision(
            expert=REGRASP_EXPERT,
            trigger=failure_event,
            reason=reason,
            request_plan=request_plan,
            subgoal="regrasp",
        )
    return RouterDecision(
        expert=current_expert,
        trigger=failure_event,
        reason=reason,
        request_plan=request_plan,
        subgoal=expert_state.current_subgoal if expert_state is not None else "execute",
    )


def _apply_router_decision(
    expert_state: ExpertState,
    decision: RouterDecision,
    expert_stats: dict[str, collections.Counter],
    *,
    default_prompt: str,
) -> None:
    next_expert = decision.expert if decision.expert in _ALLOWED_EXPERTS else expert_state.current_expert
    if next_expert != expert_state.current_expert:
        _record_expert_switch(expert_stats, next_expert)
        expert_state.current_expert = next_expert
        expert_state.steps_in_expert = 0
    expert_state.current_subgoal = decision.subgoal or expert_state.current_subgoal
    expert_state.last_trigger = decision.trigger
    expert_state.last_reason = decision.reason
    if decision.prompt_override and decision.prompt_override.strip():
        expert_state.current_prompt = decision.prompt_override.strip()
    elif next_expert == DEFAULT_VLA_EXPERT and not expert_state.current_prompt:
        expert_state.current_prompt = default_prompt
    if next_expert == DEFAULT_VLA_EXPERT and expert_state.current_subgoal not in {"execute", ""}:
        expert_state.current_subgoal = "execute"
        expert_state.current_prompt = default_prompt


def _apply_agentic_response(
    agentic_resp: dict[str, Any],
    *,
    expert_state: ExpertState,
    planner_action_counts: collections.Counter,
    expert_stats: dict[str, collections.Counter],
    default_prompt: str,
) -> None:
    action_name = agentic_resp.get("action")
    if isinstance(action_name, str) and action_name:
        planner_action_counts[action_name] += 1

    planned_expert = agentic_resp.get("expert")
    if isinstance(planned_expert, str):
        planned_expert = planned_expert.strip()
        if planned_expert in _ALLOWED_EXPERTS and planned_expert != expert_state.current_expert:
            _record_expert_switch(expert_stats, planned_expert)
            expert_state.current_expert = planned_expert
            expert_state.steps_in_expert = 0

    subgoal_name = agentic_resp.get("subgoal")
    if isinstance(subgoal_name, str) and subgoal_name:
        expert_state.current_subgoal = subgoal_name

    hint_prompt = agentic_resp.get("prompt_suggested")
    if isinstance(hint_prompt, str) and hint_prompt.strip():
        expert_state.current_prompt = hint_prompt.strip()
    elif expert_state.current_expert == DEFAULT_VLA_EXPERT and not expert_state.current_prompt:
        expert_state.current_prompt = default_prompt

    hints = agentic_resp.get("client_hints")
    if isinstance(hints, dict):
        try:
            expert_state.agentic_action_scale = float(hints.get("action_scale", expert_state.agentic_action_scale))
        except Exception:
            expert_state.agentic_action_scale = expert_state.agentic_action_scale
        try:
            expert_state.agentic_apply_steps = int(hints.get("apply_steps", expert_state.agentic_apply_steps))
        except Exception:
            expert_state.agentic_apply_steps = expert_state.agentic_apply_steps
        expert_state.agentic_action_scale = float(np.clip(expert_state.agentic_action_scale, 0.2, 1.5))
        expert_state.agentic_apply_steps = int(np.clip(expert_state.agentic_apply_steps, 0, 200))
        if expert_state.agentic_apply_steps > 0 and expert_state.current_subgoal == "execute":
            expert_state.current_subgoal = "recover"


def _verify_expert_progress(
    expert_state: ExpertState,
    *,
    failure_event: str,
    state_gap_active: bool = False,
) -> VerifierDecision:
    failure_event = str(failure_event or "").strip().lower()
    if expert_state.current_expert == DEFAULT_VLA_EXPERT:
        return VerifierDecision(status="continue", reason="default executor remains active")
    if expert_state.current_expert == TRANSITION_EXPERT:
        if not state_gap_active:
            return VerifierDecision(status="resolved", reason="state gap no longer active")
        if expert_state.steps_in_expert >= max(TRANSITION_CHUNK_STEPS * 3, 12):
            return VerifierDecision(status="escalate", reason="transition expert exceeded step budget")
        return VerifierDecision(status="continue", reason="transition expert still active")
    if expert_state.current_expert == RECOVERY_EXPERT:
        if not failure_event and expert_state.subgoal_success_window >= 6:
            return VerifierDecision(status="resolved", reason="recovery window is stable")
        if expert_state.steps_in_expert >= max(CRITIC_RECOVERY_STEPS * 3, 18):
            return VerifierDecision(status="replan", reason="recovery expert exceeded step budget")
        return VerifierDecision(status="continue", reason="recovery expert still active")
    if expert_state.current_expert == REGRASP_EXPERT:
        if failure_event not in {"misgrasp", "slip"} and expert_state.subgoal_success_window >= 6:
            return VerifierDecision(status="resolved", reason="grasp state appears stable")
        if expert_state.steps_in_expert >= max(CRITIC_RECOVERY_STEPS * 4, 24):
            return VerifierDecision(status="escalate", reason="regrasp expert exceeded step budget")
        return VerifierDecision(status="continue", reason="regrasp expert still active")
    return VerifierDecision(status="continue", reason="no verifier rule matched")


def _sync_expert_runtime_state(
    expert_state: ExpertState,
    *,
    current_prompt: str,
    current_subgoal: str,
    agentic_action_scale: float,
    agentic_apply_steps: int,
    planner_cooldown_steps: int,
    subgoal_success_window: int,
) -> None:
    expert_state.current_prompt = current_prompt
    expert_state.current_subgoal = current_subgoal
    expert_state.agentic_action_scale = float(agentic_action_scale)
    expert_state.agentic_apply_steps = int(agentic_apply_steps)
    expert_state.planner_cooldown_steps = int(planner_cooldown_steps)
    expert_state.subgoal_success_window = int(subgoal_success_window)


def _build_regrasp_prompt(task_description: str, priors: dict | None) -> str:
    task_lower = task_description.lower()
    priors = priors or {}
    if "microwave" in task_lower:
        return (
            "carefully re-align the object, close the gripper again, "
            "keep it upright, and re-approach the microwave opening"
        )
    if "stove" in task_lower or "burner" in task_lower:
        return (
            "carefully re-grasp the object, keep it upright, "
            "and re-center above the burner before placement"
        )
    if priors.get("stability_hint") == "keep_upright_while_transporting":
        return "carefully re-grasp, stabilize the object, and keep it upright before continuing"
    return "carefully re-grasp and stabilize the object before continuing the task"


def _build_policy_payload(
    *,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    episode_id: str | None = None,
    timestep: int | None = None,
    agentic_req: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "observation/image": base_img_p,
        "observation/wrist_image": wrist_img_p,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
        "prompt": str(prompt),
    }
    if episode_id is not None:
        payload["episode_id"] = episode_id
    if timestep is not None:
        payload["timestep"] = int(timestep)
    if agentic_req is not None:
        payload["agentic"] = agentic_req
    return payload


def _execute_prompt_chunk(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    chunk_steps: int,
    max_actions_per_infer: int = 3,
    instrumentation: EpisodeInstrumentation | None = None,
) -> tuple[dict[str, Any], int, bool]:
    total_steps = 0
    done = False
    current_obs = obs
    for _ in range(int(chunk_steps)):
        payload = _build_policy_payload(
            obs=current_obs,
            base_img_p=base_img_p,
            wrist_img_p=wrist_img_p,
            prompt=prompt,
        )
        try:
            infer_start_s = time.perf_counter()
            inf = _infer_with_retry(client, payload, max_retries=3)
            if instrumentation is not None:
                instrumentation.add_latency("vla", time.perf_counter() - infer_start_s)
        except Exception:
            break
        if "actions" not in inf:
            break
        if instrumentation is not None:
            instrumentation.record_commit(min(len(inf["actions"]), max_actions_per_infer))
        for action in inf["actions"][:max_actions_per_infer]:
            if instrumentation is not None:
                instrumentation.mark_first_action()
            current_obs, _, done_local, _ = env.step(action.tolist())
            total_steps += 1
            if done_local:
                done = True
                break
        if done:
            break
    return current_obs, total_steps, done


def _execute_transition_expert(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    instrumentation: EpisodeInstrumentation | None = None,
) -> tuple[dict[str, Any], int, bool]:
    return _execute_prompt_chunk(
        env=env,
        client=client,
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        chunk_steps=TRANSITION_CHUNK_STEPS,
        instrumentation=instrumentation,
    )


def _execute_recovery_expert(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    instrumentation: EpisodeInstrumentation | None = None,
) -> tuple[dict[str, Any], int, bool]:
    return _execute_prompt_chunk(
        env=env,
        client=client,
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        chunk_steps=CRITIC_RECOVERY_STEPS,
        instrumentation=instrumentation,
    )


def _execute_regrasp_expert(
    *,
    env: Any,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    instrumentation: EpisodeInstrumentation | None = None,
) -> tuple[dict[str, Any], int, bool]:
    return _execute_prompt_chunk(
        env=env,
        client=client,
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        chunk_steps=max(CRITIC_RECOVERY_STEPS, 8),
        instrumentation=instrumentation,
    )


def _apply_runtime_from_expert_state(expert_state: ExpertState) -> tuple[str, str, float, int]:
    return (
        expert_state.current_subgoal,
        expert_state.current_prompt,
        expert_state.agentic_action_scale,
        expert_state.agentic_apply_steps,
    )


def _handle_agentic_response(
    agentic_resp: dict[str, Any],
    *,
    expert_state: ExpertState,
    planner_stats: dict[str, Any],
    planner_action_counts: collections.Counter,
    planner_expert_counts: collections.Counter,
    expert_stats: dict[str, collections.Counter],
    default_prompt: str,
) -> tuple[str, str, float, int]:
    if float(agentic_resp.get("plan_ms", 0.0)) > 0.0:
        planner_stats["plans_generated"] += 1
    expert_name = agentic_resp.get("expert")
    if isinstance(expert_name, str) and expert_name:
        planner_expert_counts[expert_name] += 1
    _apply_agentic_response(
        agentic_resp,
        expert_state=expert_state,
        planner_action_counts=planner_action_counts,
        expert_stats=expert_stats,
        default_prompt=default_prompt,
    )
    return _apply_runtime_from_expert_state(expert_state)


def _request_planner(
    *,
    client: Any,
    obs: dict[str, Any],
    base_img_p: np.ndarray,
    wrist_img_p: np.ndarray,
    prompt: str,
    episode_id: str,
    timestep: int,
    event: str,
    reason: str,
    context: str,
    planner_stats: dict[str, Any],
    planner_event_counts: collections.Counter,
    planner_action_counts: collections.Counter,
    planner_expert_counts: collections.Counter,
    expert_state: ExpertState,
    expert_stats: dict[str, collections.Counter],
    default_prompt: str,
    current_cooldown_steps: int,
    cooldown_steps: int,
    instrumentation: EpisodeInstrumentation | None = None,
) -> tuple[bool, int, str, str, float, int]:
    if current_cooldown_steps > 0:
        return False, current_cooldown_steps, * _apply_runtime_from_expert_state(expert_state)
    planner_stats["total_requests"] += 1
    planner_event_counts[str(event)] += 1
    next_cooldown_steps = max(int(current_cooldown_steps), int(cooldown_steps))
    payload = _build_policy_payload(
        obs=obs,
        base_img_p=base_img_p,
        wrist_img_p=wrist_img_p,
        prompt=prompt,
        episode_id=episode_id,
        timestep=timestep,
        agentic_req={
            "request_plan": True,
            "event": str(event),
            "reason": str(reason),
            "context": str(context),
        },
    )
    try:
        planner_start_s = time.perf_counter()
        plan_inf = _infer_with_retry(client, payload, max_retries=3)
        if instrumentation is not None:
            instrumentation.add_latency("planner", time.perf_counter() - planner_start_s, blocking=True)
    except Exception:
        return True, next_cooldown_steps, * _apply_runtime_from_expert_state(expert_state)
    if isinstance(plan_inf.get("agentic"), dict):
        current_subgoal, current_prompt, action_scale, apply_steps = _handle_agentic_response(
            plan_inf["agentic"],
            expert_state=expert_state,
            planner_stats=planner_stats,
            planner_action_counts=planner_action_counts,
            planner_expert_counts=planner_expert_counts,
            expert_stats=expert_stats,
            default_prompt=default_prompt,
        )
        return True, next_cooldown_steps, current_subgoal, current_prompt, action_scale, apply_steps
    return True, next_cooldown_steps, * _apply_runtime_from_expert_state(expert_state)


def _keyword_match(task_lower: str, keyword: str) -> bool:
    if keyword in task_lower:
        return True
    if keyword.endswith("y") and f"{keyword[:-1]}ies" in task_lower:
        return True
    if f"{keyword}s" in task_lower:
        return True
    return False


def _build_task_control_profile(task_description: str) -> dict:
    profile = TASK_CONTROL_PROFILES["default"].copy()
    task_lower = task_description.lower()
    for keyword, keyword_profile in TASK_CONTROL_PROFILES.items():
        if keyword == "default":
            continue
        if _keyword_match(task_lower, keyword):
            profile.update(keyword_profile)
    return profile


def _build_transition_prompt(task_description: str, priors: dict | None) -> str:
    task_lower = task_description.lower()
    priors = priors or {}
    if "microwave" in task_lower:
        return (
            "slightly back away from the door frame, keep the object upright, "
            "re-center with the opening, and maintain the current grasp"
        )
    if "both" in task_lower and "moka pot" in task_lower and ("stove" in task_lower or "burner" in task_lower):
        return (
            "continue putting both moka pots on the stove: if holding a moka pot, "
            "keep it upright and place it centered on the burner; otherwise move to "
            "the remaining moka pot, grasp it from the side, lift it upright, and "
            "place it centered on the burner"
        )
    if "stove" in task_lower or "burner" in task_lower:
        return (
            "slightly lift, keep the object upright, re-center above the burner, "
            "and maintain the current grasp before continuing placement"
        )
    if priors.get("stability_hint") == "keep_upright_while_transporting":
        return "slightly lift, keep the object upright, and stabilize the grasp before continuing"
    return TRANSITION_PROMPT


def _build_recovery_prompt(task_description: str, priors: dict | None) -> str:
    task_lower = task_description.lower()
    priors = priors or {}
    if "plate" in task_lower and "mug" in task_lower:
        return (
            "recover the plate placement: lift slightly if needed, move the object "
            "back above its target plate, center it over the plate, lower until stable, "
            "and release gently before continuing"
        )
    if "right of the plate" in task_lower:
        return (
            "recover the relation placement: stabilize the grasp, move the object to "
            "the right side of the plate with clear separation, lower until stable, "
            "and continue without disturbing the plate"
        )
    if "microwave" in task_lower:
        return (
            "carefully retract a little, keep the object upright, "
            "and stabilize before re-aligning with the microwave opening"
        )
    if "both" in task_lower and "moka pot" in task_lower and ("stove" in task_lower or "burner" in task_lower):
        return (
            "recover the moka pot task: stabilize the gripper, keep any held moka pot "
            "upright, re-center over the burner if holding it, or return to the "
            "remaining moka pot and continue the second placement"
        )
    if "stove" in task_lower or "burner" in task_lower:
        return (
            "carefully lift a little, keep the object upright, "
            "and re-center above the burner before continuing placement"
        )
    if priors.get("stability_hint") == "keep_upright_while_transporting":
        return "slightly lift and stabilize the grasp while keeping the object upright"
    return CRITIC_RECOVERY_PROMPT


def _build_progress_transition_prompt(default_prompt: str, progress: dict[str, Any] | None) -> str:
    if not progress or progress.get("kind") != "moka_pot_stove":
        return default_prompt
    completed_count = int(progress.get("completed_count", 0))
    remaining_objects = list(progress.get("remaining_objects", []))
    if completed_count <= 0:
        return (
            "focus on completing the first moka pot placement: grasp one moka pot from the side, "
            "lift it upright, move directly above the stove burner, lower it until it is stable, "
            "and release only after it is centered on the burner"
        )
    if completed_count == 1:
        remaining_text = "the remaining moka pot"
        if remaining_objects:
            remaining_text = "the remaining moka pot"
        return (
            f"leave the completed moka pot on the stove and finish the second placement: "
            f"go to {remaining_text}, grasp it from the side, keep it upright, move directly "
            "above the stove burner, lower it until stable, and release it centered on the burner"
        )
    return default_prompt


def _build_progress_recovery_prompt(
    default_prompt: str,
    progress: dict[str, Any] | None,
    *,
    goal_aware: bool = False,
) -> str:
    if not progress:
        return default_prompt
    if progress.get("kind") not in {
        "single_moka_pot_stove",
        "moka_pot_stove",
        "plate_relation",
    }:
        return default_prompt
    target_object = str(progress.get("target_object") or "the moka pot").replace("_", " ")
    if progress.get("phase") == "progress_regression_recovery":
        goal_clause = "make sure the stove is turned on, " if goal_aware else ""
        return (
            f"recover regressed completion of {target_object}: {goal_clause}"
            "move back above the stove burner, re-center the object over the cook region, "
            "lower until stable, and release gently"
        )
    if progress.get("phase") == "far_stall_recovery":
        if progress.get("kind") == "plate_relation":
            target_relation = str(progress.get("target_relation") or "").replace("_", " ")
            if not target_relation:
                target_relation = "the required plate relation"
            return (
                f"recover stalled relation of {target_object}: satisfy {target_relation}, "
                "move carefully to the target plate or right-of-plate position, lower until stable, "
                "release gently, then continue the remaining relation"
            )
        goal_clause = "make sure the stove is turned on, " if goal_aware else ""
        return (
            f"recover stalled placement of {target_object}: grasp it from the side if needed, "
            f"{goal_clause}lift it upright, move directly above the stove burner, lower until it is stable, "
            "and release only after it is centered on the burner"
        )
    if progress.get("needs_recenter"):
        if progress.get("kind") == "plate_relation":
            target_relation = str(progress.get("target_relation") or "target relation").replace("_", " ")
            return (
                f"recover {target_object}: satisfy {target_relation}, move carefully to the "
                "target plate or right-of-plate position, lower until stable, and release gently"
            )
        goal_clause = "make sure the stove is turned on, " if goal_aware else ""
        return (
            f"recover placement of {target_object}: move back above the stove burner, "
            f"{goal_clause}re-center the object over the cook region, lower it until it is stable, "
            "then release it gently"
        )
    return default_prompt


def _apply_initial_perturbation(
    env: Any,
    obs: dict[str, Any],
    *,
    perturbation: str,
    rng: np.random.Generator,
    object_jitter_xy: float,
) -> dict[str, Any]:
    if perturbation in {"clean", "mid_episode_nudge"}:
        return obs
    if perturbation != "object_jitter":
        raise ValueError(f"Unsupported perturbation: {perturbation}")

    inner_env = _inner_libero_env(env)
    objects_dict = getattr(inner_env, "objects_dict", {}) or {}
    jittered = 0
    for obj in objects_dict.values():
        joints = list(getattr(obj, "joints", []) or [])
        if not joints:
            continue
        joint_name = joints[-1]
        try:
            qpos = np.asarray(inner_env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        except Exception:
            continue
        if qpos.shape[0] < 2:
            continue
        qpos[0] += float(rng.uniform(-object_jitter_xy, object_jitter_xy))
        qpos[1] += float(rng.uniform(-object_jitter_xy, object_jitter_xy))
        try:
            inner_env.sim.data.set_joint_qpos(joint_name, qpos)
            jittered += 1
        except Exception:
            continue
    if jittered <= 0:
        return obs
    inner_env.sim.forward()
    try:
        return env.regenerate_obs_from_state(env.get_sim_state())
    except Exception:
        inner_env._post_process()
        inner_env._update_observables(force=True)
        return inner_env._get_observations()


def _object_matches_task(object_name: str, task_description: str) -> bool:
    object_text = object_name.replace("_", " ").replace("-", " ").lower()
    task_text = task_description.lower()
    for keyword, prior in SCENE_PRIORS.items():
        if prior.get("role") != "manipulated_object":
            continue
        if keyword not in task_text:
            continue
        keyword_tokens = keyword.split()
        if all(token in object_text for token in keyword_tokens):
            return True
        if any(token in object_text for token in keyword_tokens):
            return True
    extra_keywords = ("pudding", "moka", "mug", "plate", "bowl", "kettle", "pan")
    return any(keyword in task_text and keyword in object_text for keyword in extra_keywords)


def _apply_mid_episode_nudge(
    env: Any,
    obs: dict[str, Any],
    *,
    task_description: str,
    rng: np.random.Generator,
    object_nudge_xy: float,
    timestep: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    inner_env = _inner_libero_env(env)
    objects_dict = getattr(inner_env, "objects_dict", {}) or {}
    candidates: list[tuple[str, str, np.ndarray]] = []
    fallback_candidates: list[tuple[str, str, np.ndarray]] = []
    for object_key, obj in objects_dict.items():
        joints = list(getattr(obj, "joints", []) or [])
        if not joints:
            continue
        joint_name = joints[-1]
        try:
            qpos = np.asarray(inner_env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        except Exception:
            continue
        if qpos.shape[0] < 2:
            continue
        object_name = str(object_key)
        fallback_candidates.append((object_name, joint_name, qpos))
        if _object_matches_task(object_name, task_description):
            candidates.append((object_name, joint_name, qpos))

    selectable = candidates or fallback_candidates
    if not selectable:
        return obs, None

    object_name, joint_name, qpos = selectable[int(rng.integers(0, len(selectable)))]
    dx = float(rng.uniform(-object_nudge_xy, object_nudge_xy))
    dy = float(rng.uniform(-object_nudge_xy, object_nudge_xy))
    if abs(dx) < object_nudge_xy * 0.25:
        dx = float(np.sign(dx) or 1.0) * object_nudge_xy * 0.25
    if abs(dy) < object_nudge_xy * 0.25:
        dy = float(np.sign(dy) or 1.0) * object_nudge_xy * 0.25
    qpos[0] += dx
    qpos[1] += dy
    try:
        inner_env.sim.data.set_joint_qpos(joint_name, qpos)
    except Exception:
        return obs, None
    inner_env.sim.forward()
    try:
        obs = env.regenerate_obs_from_state(env.get_sim_state())
    except Exception:
        inner_env._post_process()
        inner_env._update_observables(force=True)
        obs = inner_env._get_observations()
    event = {
        "type": "mid_episode_nudge",
        "timestep": int(timestep),
        "object": object_name,
        "joint": joint_name,
        "dx": dx,
        "dy": dy,
    }
    return obs, event


def _parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return [int(part) for part in parts]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real LIBERO evaluation for Agentic-VLA ablation.")
    parser.add_argument("--host", default="127.0.0.1", help="Websocket policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="Websocket policy server port.")
    parser.add_argument(
        "--task-suite",
        default="libero_spatial",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
            "libero_10_swap",
            "libero_10_object",
            "libero_10_task",
        ],
        help="LIBERO task suite name.",
    )
    parser.add_argument("--task-id", type=int, default=None, help="Single task id to run.")
    parser.add_argument("--task-ids", type=_parse_int_list, default=[], help="Comma-separated task ids to run.")
    parser.add_argument(
        "--skip-task-ids",
        type=_parse_int_list,
        default=[],
        help="Comma-separated task ids to skip.",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per task.")
    parser.add_argument("--replan-steps", type=int, default=10, help="Action chunk replanning interval.")
    parser.add_argument("--resize-size", type=int, default=224, help="Image resize used by policy.")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="Warmup sim steps before control.")
    parser.add_argument(
        "--perturbation",
        default="clean",
        choices=["clean", "object_jitter", "mid_episode_nudge"],
        help="Optional robustness perturbation applied after set_init_state.",
    )
    parser.add_argument(
        "--object-jitter-xy",
        type=float,
        default=0.03,
        help="Max absolute xy object jitter in meters for --perturbation object_jitter.",
    )
    parser.add_argument(
        "--mid-nudge-step",
        type=int,
        default=120,
        help="Control timestep that triggers --perturbation mid_episode_nudge.",
    )
    parser.add_argument(
        "--mid-nudge-xy",
        type=float,
        default=0.03,
        help="Max absolute xy object displacement in meters for --perturbation mid_episode_nudge.",
    )
    parser.add_argument("--video-dir", default=str(PROJECT_ROOT / "results" / "videos"), help="Video output directory.")
    parser.add_argument(
        "--video-render-size",
        type=int,
        default=0,
        help=(
            "Optional square display-render resolution. Values greater than zero "
            "record an independent MuJoCo render while leaving policy observations "
            "at their configured resolution; zero records the policy camera frame."
        ),
    )
    parser.add_argument(
        "--video-camera",
        default="agentview",
        help="MuJoCo camera used when --video-render-size is greater than zero.",
    )
    parser.add_argument(
        "--results-json",
        default=str(PROJECT_ROOT / "results" / "libero_eval_results.json"),
        help="Path to save measured success metrics as JSON.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--carve-runtime",
        action="store_true",
        help="Route policy calls through the model-agnostic CARVE adapter boundary.",
    )
    parser.add_argument(
        "--carve-trace-jsonl",
        default="",
        help="Per-policy-call CARVE trace path. Defaults beside --results-json.",
    )
    parser.add_argument(
        "--carve-joint-controller",
        action="store_true",
        help="Enable deployable risk monitoring and joint recovery/compute decisions.",
    )
    parser.add_argument(
        "--carve-semantic-shadow",
        action="store_true",
        help=(
            "Run event-triggered VLM observations asynchronously and record their "
            "decisions without changing robot actions."
        ),
    )
    parser.add_argument(
        "--carve-semantic-endpoint",
        default="http://127.0.0.1:18070/v1/chat/completions",
    )
    parser.add_argument(
        "--carve-semantic-model",
        default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B",
    )
    parser.add_argument("--carve-semantic-cooldown-steps", type=int, default=100)
    parser.add_argument("--carve-semantic-min-slack-ms", type=float, default=20.0)
    parser.add_argument("--carve-semantic-timeout-sec", type=float, default=30.0)
    parser.add_argument("--carve-semantic-max-tokens", type=int, default=12)
    parser.add_argument(
        "--carve-semantic-protocol",
        choices=("label_v1", "code_v2", "code_v3", "code_v4", "code_v5"),
        default="label_v1",
    )
    parser.add_argument(
        "--carve-high-level-agent",
        action="store_true",
        help=(
            "Enable the guarded external multimodal planner at explicit Agentic "
            "boundaries."
        ),
    )
    parser.add_argument(
        "--carve-canonical-harness",
        action="store_true",
        help=(
            "Route guarded planner lifecycle and safe-boundary state transitions "
            "through AsyncAgenticHarnessController. Requires --carve-high-level-agent."
        ),
    )
    parser.add_argument(
        "--carve-agentic-knowledge",
        action="store_true",
        help=(
            "Inject deployable semantic scene graphs and the fixed HAA-RAG "
            "experience index into canonical high-level planner calls."
        ),
    )
    parser.add_argument(
        "--carve-semantic-before-recovery",
        action="store_true",
        help=(
            "At a confirmed recovery boundary, let the canonical VLM planner "
            "select a registered skill or grounded VLA replan before execution."
        ),
    )
    parser.add_argument(
        "--carve-high-level-task-start-policy",
        choices=("event_only", "startup_shadow", "startup_wait"),
        default="event_only",
        help=(
            "Task-start VLM semantics: event_only makes no startup call; "
            "startup_shadow records but never applies it; startup_wait holds "
            "before motion and applies the complete guarded decision."
        ),
    )
    parser.add_argument("--carve-high-level-max-calls", type=int, default=3)
    parser.add_argument("--carve-high-level-min-confidence", type=float, default=0.55)
    parser.add_argument("--carve-high-level-max-tokens", type=int, default=128)
    parser.add_argument(
        "--carve-high-level-boundary-timeout-sec",
        type=float,
        default=10.0,
        help=(
            "Maximum time to await a VLM result after execution has entered a "
            "safe planner boundary; expiry fails closed rather than blocking VLA control."
        ),
    )
    parser.add_argument(
        "--carve-physical-recovery",
        action="store_true",
        help=(
            "Execute the bounded Cartesian physical skill for controller-selected "
            "RECOVERY decisions; otherwise keep the prompt-retry baseline."
        ),
    )
    parser.add_argument(
        "--carve-physical-minimum-state-response",
        type=float,
        default=1e-4,
        help="Minimum EEF response required before replanning after physical recovery.",
    )
    parser.add_argument(
        "--carve-max-recovery-attempts",
        type=int,
        default=2,
        help=(
            "Bounded physical-recovery budget before planner escalation or safe stop."
        ),
    )
    parser.add_argument(
        "--carve-async-prefetch",
        action="store_true",
        help="Prefetch the next VLA action chunk while bounded cached actions execute.",
    )
    parser.add_argument(
        "--carve-prefetch-lead-actions",
        type=int,
        default=2,
        help="Cached actions used to cover one asynchronous VLA request (minimum: 2).",
    )
    parser.add_argument(
        "--carve-prefetch-interval",
        type=int,
        default=1,
        help="Submit at most one prefetch per N policy chunks (default: every chunk).",
    )
    parser.add_argument(
        "--carve-prefetch-prefix-mode",
        choices=("off", "shadow", "enforce"),
        default="off",
        help=(
            "Compare the prefetched hypothetical prefix with cached actions that "
            "actually execute; shadow records only and enforce synchronously re-anchors."
        ),
    )
    parser.add_argument(
        "--carve-prefetch-prefix-rms-max",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--carve-prefetch-prefix-translation-max",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--carve-prefetch-prefix-rotation-max",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--carve-prefetch-prefix-gripper-min",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--carve-branch-dir",
        default="",
        help="Save cooldown-gated high-risk simulator snapshots for paired branch evaluation.",
    )
    parser.add_argument("--carve-max-branch-snapshots", type=int, default=3)
    parser.add_argument("--carve-branch-min-gap-steps", type=int, default=40)
    parser.add_argument(
        "--fixed-policy-noise",
        action="store_true",
        help="Send deterministic pi0.5 sampling noise keyed by task, episode, and control step.",
    )
    parser.add_argument(
        "--carve-fixed-inference-steps",
        type=int,
        default=0,
        help="Request a fixed pi0.5 flow-step budget through CARVE; zero keeps backend default.",
    )
    parser.add_argument(
        "--carve-fast-inference-steps",
        type=int,
        default=2,
        help="Flow-step budget for the CARVE fast path.",
    )
    parser.add_argument(
        "--carve-accurate-inference-steps",
        type=int,
        default=2,
        help="Flow-step budget for CARVE risk-triggered replans; values above 2 are experimental.",
    )
    parser.add_argument(
        "--carve-commit-steps",
        type=int,
        default=10,
        help="Maximum action horizon committed by the CARVE joint controller.",
    )
    parser.add_argument(
        "--carve-monitor-warmup-steps",
        type=int,
        default=60,
        help=(
            "Control steps collected before the deployable monitor may emit a stall; "
            "prevents startup transients from triggering recovery."
        ),
    )

    # Ablation flags
    parser.add_argument("--transition", action="store_true", help="A2: Enable Transition Agent.")
    parser.add_argument("--graph-rag", action="store_true", help="A3: Enable Graph RAG + Memory.")
    parser.add_argument("--critic", action="store_true", help="A4: Enable Critic/Retry with Qwen3-VL.")
    parser.add_argument(
        "--agentic-planner",
        action="store_true",
        help="Enable server-side VLM planner on-demand (request plan only when stalled).",
    )
    parser.add_argument(
        "--planner-cooldown-steps",
        type=int,
        default=50,
        help="Minimum steps between two planner requests within one episode (on-demand planner only).",
    )

    # Legacy flag (appendix supplement only)
    parser.add_argument("--vision-prompt", action="store_true", help="Legacy: overlay colored masks (appendix only).")
    parser.add_argument("--mask-alpha", type=float, default=0.35, help="Alpha for colored mask overlay.")

    # Meta
    parser.add_argument("--ablation-tag", default="", help="Tag for this ablation experiment.")
    parser.add_argument(
        "--method-tag",
        default="",
        choices=[
            "",
            "B0-VLA",
            "B0-VLA-Light",
            "B0-VLA-Quant",
            "B0-VLA-CAQ",
            "B0-VLA-CAQ-Proxy",
            "B1-FullRefined",
            "B2-FixedHarness",
            "B3-EfficientOnly",
            "B4-Agentic",
            "B4-Agentic-Light",
            "B4-Agentic-Quant",
            "B4-Agentic-CAQ",
            "B4-Agentic-CAQ-Proxy",
            "B4-BAHarness-Rules",
            "B4-Async",
            "B4-AAC-Lite",
            "B5-BAHarness-Learned",
        ],
        help="Paper-facing method tag. Overrides --ablation-tag when provided.",
    )
    parser.add_argument(
        "--episode-trace-jsonl",
        default="",
        help="Path to append episode-level traces. Defaults to episode_traces.jsonl next to --results-json.",
    )
    parser.add_argument(
        "--control-deadline-ms",
        type=float,
        default=80.0,
        help="Soft real-time control deadline used for deadline_miss_rate instrumentation.",
    )
    parser.add_argument(
        "--ba-max-recoveries",
        type=int,
        default=BA_MAX_RECOVERIES_PER_EPISODE,
        help="Max rule-based recoveries per episode for B4-BAHarness-Rules.",
    )
    parser.add_argument(
        "--ba-recovery-cooldown-steps",
        type=int,
        default=BA_RECOVERY_COOLDOWN_STEPS,
        help="Cooldown after a B4 rule-based recovery.",
    )
    parser.add_argument(
        "--ba-post-recovery-transition-lockout-steps",
        type=int,
        default=BA_POST_RECOVERY_TRANSITION_LOCKOUT_STEPS,
        help="Extra B4 lockout window that prevents transition expert immediately after recovery.",
    )
    parser.add_argument(
        "--ba-recovery-risk-threshold",
        type=float,
        default=BA_RECOVERY_RISK_THRESHOLD,
        help="Minimum B4 risk score required to trigger rule-based recovery.",
    )
    parser.add_argument(
        "--ba-transition-risk-threshold",
        type=float,
        default=BA_TRANSITION_RISK_THRESHOLD,
        help="Minimum B4 risk score required to trigger transition expert.",
    )
    parser.add_argument(
        "--ba-enable-transition",
        action="store_true",
        help="Force-enable B4 transition execution when the task profile allows it. State-gap sensing is still used for risk scoring.",
    )
    parser.add_argument(
        "--ba-disable-transition",
        action="store_true",
        help="Disable B4 transition execution, including progress-triggered transitions, for ablation.",
    )
    parser.add_argument(
        "--ba-safe-long-horizon-fast-path",
        action="store_true",
        help="Use a conservative B4 profile that skips recovery/transition on long-horizon multi-object tasks.",
    )
    parser.add_argument(
        "--ba-disable-recovery",
        action="store_true",
        help="Disable B4 rule-based recovery execution for ablation/fast-path profiles.",
    )
    parser.add_argument(
        "--ba-enable-recovery",
        action="store_true",
        help="Force-enable B4 rule-based recovery even if the task profile disables it.",
    )
    parser.add_argument(
        "--ba-augment-prompt",
        action="store_true",
        help="Allow B4 to augment the low-level VLA prompt with GraphRAG priors. Off by default to preserve fast-path behavior.",
    )
    parser.add_argument(
        "--ba-progress-verifier",
        action="store_true",
        help="Enable lightweight task progress verifier for B4 scheduling.",
    )
    parser.add_argument(
        "--ba-progress-recovery-min-steps",
        type=int,
        default=BA_PROGRESS_RECOVERY_MIN_STEPS,
        help="Earliest control step at which the lightweight progress verifier may force recovery.",
    )
    parser.add_argument(
        "--ba-progress-far-recovery-min-steps",
        type=int,
        default=-1,
        help="Earliest control step for far-stall progress recovery; negative disables this adaptive trigger.",
    )
    parser.add_argument(
        "--ba-progress-far-recovery-distance",
        type=float,
        default=0.20,
        help="Target xy distance threshold for far-stall progress recovery.",
    )
    parser.add_argument(
        "--ba-progress-regression-recovery",
        action="store_true",
        help="Trigger recovery when lightweight progress regresses from completed to incomplete.",
    )
    parser.add_argument(
        "--ba-progress-regression-distance",
        type=float,
        default=0.12,
        help="Target xy distance threshold for completed-to-incomplete progress regression recovery.",
    )
    parser.add_argument(
        "--ba-progress-goal-aware-recovery-prompt",
        action="store_true",
        help="Include the original task goal in progress recovery prompts; useful as an ablation, off by default.",
    )
    parser.add_argument(
        "--ba-context-aware-intervention",
        action="store_true",
        help="Gate B4 recovery/transition by episode context; use fast-path on clean or before perturbation evidence.",
    )
    parser.add_argument(
        "--ba-phase-aware-transition-gate",
        action="store_true",
        help="For long-horizon moka-pot tasks, allow transition only in the second-object phase with confirmed state-gap.",
    )
    parser.add_argument(
        "--ba-perturbation-recovery",
        action="store_true",
        help="Force one targeted recovery after a mid-episode perturbation when task progress shows a large target drift.",
    )
    parser.add_argument(
        "--ba-aac-lite",
        action="store_true",
        help="Enable risk-aware adaptive action commitment for B4.",
    )
    parser.add_argument(
        "--ba-aac-min-commit",
        type=int,
        default=1,
        help="Minimum normal-policy commit length for AAC-lite.",
    )
    parser.add_argument(
        "--ba-aac-medium-commit",
        type=int,
        default=6,
        help="Normal-policy commit length for medium-risk AAC-lite states.",
    )
    parser.add_argument(
        "--ba-aac-high-commit",
        type=int,
        default=3,
        help="Normal-policy commit length for high-risk AAC-lite states.",
    )
    parser.add_argument(
        "--ba-aac-max-commit",
        type=int,
        default=10,
        help="Maximum normal-policy commit length for low-risk AAC-lite states.",
    )
    parser.add_argument(
        "--light-reuse-actions",
        action="store_true",
        help="Enable Lightweight VLA Runtime action reuse: cache unused VLA chunk suffixes and execute them in low-risk states before calling the full VLA again.",
    )
    parser.add_argument(
        "--light-reuse-risk-buckets",
        default="low",
        help="Comma-separated BA risk buckets where cached VLA actions may be reused.",
    )
    parser.add_argument(
        "--light-reuse-max-actions",
        type=int,
        default=5,
        help="Maximum cached actions to reuse before the next full VLA call.",
    )
    parser.add_argument(
        "--light-reuse-max-buffer-actions",
        type=int,
        default=10,
        help="Maximum unused action chunk suffix length to keep in the lightweight reuse buffer.",
    )
    parser.add_argument(
        "--light-reuse-commit-steps",
        type=int,
        default=-1,
        help="When action reuse is enabled for the current task, cap normal full-VLA commit steps to this value so the unused suffix can be cached. Negative keeps --replan-steps.",
    )
    parser.add_argument(
        "--light-reuse-disable-long-horizon",
        action="store_true",
        help="Disable action reuse on long-horizon multi-object tasks; useful for a task-aware safe lightweight path.",
    )
    parser.add_argument(
        "--light-reuse-post-perturbation-lockout-steps",
        type=int,
        default=0,
        help="Disable action reuse for this many control steps after an online perturbation.",
    )
    parser.add_argument(
        "--light-reuse-post-recovery-lockout-steps",
        type=int,
        default=0,
        help="Disable action reuse for this many control steps after an Agentic recovery.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model or local path for Critic.",
    )
    parser.add_argument("--qwen-quant", default="4bit", choices=["none", "4bit", "8bit"],
                        help="Quantization mode for Qwen3-VL (none/4bit/8bit). Default: 4bit for RTX 4090.")
    parser.add_argument("--critic-temp", type=float, default=0.3, help="Temperature for Qwen3-VL Critic.")
    return parser.parse_args()


def _require_runtime():
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy
    except Exception as exc:
        raise RuntimeError(
            "Missing runtime dependency for real LIBERO evaluation."
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv, SegmentationRenderEnv, image_tools, websocket_client_policy


def _configure_isolated_libero_paths(root: pathlib.Path) -> pathlib.Path:
    """Point this process at one checkout without changing ~/.libero."""

    package_root = root / "libero" / "libero"
    if not package_root.is_dir():
        raise FileNotFoundError(f"LIBERO package data root not found: {package_root}")
    config_root = pathlib.Path(
        os.environ.get(
            "AGENTIC_VLA_LIBERO_CONFIG_PATH",
            PROJECT_ROOT / "results" / "_runtime" / f"libero_config_{root.name}",
        )
    ).expanduser()
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "assets": str(package_root / "assets"),
        "bddl_files": str(package_root / "bddl_files"),
        "benchmark_root": str(package_root),
        "datasets": str(root / "libero" / "datasets"),
        "init_states": str(package_root / "init_files"),
    }
    (config_root / "config.yaml").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    return config_root


def _max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name.startswith("libero_10"):
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _make_env(task, get_libero_path, env_class, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = env_class(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _build_results_payload(
    args: argparse.Namespace,
    ablation_tag: str,
    total_successes: int,
    total_episodes: int,
    episode_lengths: list[int],
    success_episode_lengths: list[int],
    task_metrics: dict,
    transition_stats: dict | None,
    critic_stats: dict | None,
    planner_stats: dict | None,
    expert_stats: dict | None,
    episode_trace_path: str | None,
    trace_aggregate: dict[str, Any] | None,
    status: str,
    current_task: dict | None,
) -> dict:
    transition_enabled = bool(args.transition or _is_ba_harness(args))
    overall = total_successes / total_episodes if total_episodes else 0.0
    avg_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    avg_success_episode_length = float(np.mean(success_episode_lengths)) if success_episode_lengths else 0.0
    payload = {
        "task_suite": args.task_suite,
        "trials_per_task": args.trials,
        "ablation_tag": ablation_tag,
        "status": status,
        "current_task": current_task,
        "evaluated_task_ids": getattr(args, "evaluated_task_ids", None),
        "skipped_task_ids": getattr(args, "skipped_task_ids", None),
        "flags": {
            "transition": args.transition,
            "graph_rag": args.graph_rag,
            "critic": args.critic,
            "vision_prompt": args.vision_prompt,
            "agentic_planner": getattr(args, "agentic_planner", False),
            "carve_runtime": getattr(args, "carve_runtime", False),
            "carve_joint_controller": getattr(args, "carve_joint_controller", False),
            "carve_physical_recovery": getattr(args, "carve_physical_recovery", False),
            "carve_high_level_agent": getattr(args, "carve_high_level_agent", False),
            "carve_high_level_max_calls": getattr(
                args, "carve_high_level_max_calls", None
            ),
            "carve_high_level_min_confidence": getattr(
                args, "carve_high_level_min_confidence", None
            ),
            "carve_async_prefetch": getattr(args, "carve_async_prefetch", False),
            "carve_prefetch_lead_actions": getattr(
                args, "carve_prefetch_lead_actions", None
            ),
            "carve_prefetch_interval": getattr(args, "carve_prefetch_interval", None),
            "carve_prefetch_prefix_mode": getattr(
                args, "carve_prefetch_prefix_mode", "off"
            ),
            "carve_prefetch_prefix_rms_max": getattr(
                args, "carve_prefetch_prefix_rms_max", None
            ),
            "carve_prefetch_prefix_translation_max": getattr(
                args, "carve_prefetch_prefix_translation_max", None
            ),
            "carve_prefetch_prefix_rotation_max": getattr(
                args, "carve_prefetch_prefix_rotation_max", None
            ),
            "carve_prefetch_prefix_gripper_min": getattr(
                args, "carve_prefetch_prefix_gripper_min", None
            ),
            "carve_fixed_inference_steps": getattr(args, "carve_fixed_inference_steps", 0),
            "carve_fast_inference_steps": getattr(args, "carve_fast_inference_steps", 2),
            "carve_accurate_inference_steps": getattr(
                args, "carve_accurate_inference_steps", 2
            ),
            "carve_commit_steps": getattr(args, "carve_commit_steps", 10),
            "method_tag": getattr(args, "method_tag", ""),
            "ba_harness": _is_ba_harness(args),
            "qwen_quant": getattr(args, "qwen_quant", None),
            "control_deadline_ms": getattr(args, "control_deadline_ms", None),
        },
        "episode_trace_jsonl": episode_trace_path,
        "trace_aggregate": trace_aggregate or {},
        "hardware_config": {
            "host": args.host,
            "port": args.port,
            "openpi_root": str(OPENPI_ROOT),
            "qwen_model": getattr(args, "qwen_model", None),
            "qwen_quant": getattr(args, "qwen_quant", None),
            "resize_size": getattr(args, "resize_size", None),
        },
        "overall_success_rate": overall,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "avg_episode_length": avg_episode_length,
        "avg_success_episode_length": avg_success_episode_length,
        "task_metrics": task_metrics,
        "transition_stats": transition_stats if transition_enabled else None,
        "critic_stats": critic_stats if args.critic else None,
        "planner_stats": planner_stats if getattr(args, "agentic_planner", False) else None,
        "expert_stats": expert_stats,
        "mechanism_summary": {
            "avg_transitions_per_episode": (
                transition_stats["total_transitions"] / total_episodes if transition_enabled and total_episodes else 0.0
            ),
            "transition_success_rate_when_used": (
                transition_stats["transitions_leading_to_success"] / max(1, sum(
                    metrics["transition_episode_count"] for metrics in task_metrics.values()
                ))
                if transition_enabled else 0.0
            ),
            "avg_retries_per_episode": (
                critic_stats["retries_triggered"] / total_episodes if args.critic and total_episodes else 0.0
            ),
            "retry_success_rate_when_used": (
                critic_stats["retries_leading_to_success"] / max(1, sum(
                    metrics["retry_episode_count"] for metrics in task_metrics.values()
                ))
                if args.critic else 0.0
            ),
            "avg_episode_length": avg_episode_length,
            "avg_success_episode_length": avg_success_episode_length,
        },
    }
    if payload["evaluated_task_ids"] is None:
        payload.pop("evaluated_task_ids", None)
    if payload["skipped_task_ids"] is None:
        payload.pop("skipped_task_ids", None)
    return payload


# ===== Vision Prompt (Legacy) =====
def _apply_vision_prompt(image, segmentation_image, obj_of_interest, instance_to_id, alpha=0.35):
    if segmentation_image.ndim == 3:
        segmentation_image = segmentation_image.squeeze(-1)
    result = image.astype(np.float32)
    for i, obj_name in enumerate(obj_of_interest):
        if obj_name not in instance_to_id:
            continue
        seg_id = instance_to_id[obj_name]
        mask = segmentation_image == seg_id
        if not mask.any():
            continue
        color = VISION_PROMPT_COLORS[i % len(VISION_PROMPT_COLORS)]
        result[mask] = (1 - alpha) * result[mask] + alpha * color
    return np.clip(result, 0, 255).astype(np.uint8)


# ===== Transition Agent (A2) =====
class TransitionAgent:
    """Detects State Gap and triggers prompt-based transition via VLA."""

    def __init__(self, stall_threshold=TRANSITION_STALL_THRESHOLD,
                 stall_window=TRANSITION_STALL_WINDOW,
                 max_transitions=MAX_TRANSITIONS_PER_EPISODE):
        self.default_stall_threshold = stall_threshold
        self.default_stall_window = stall_window
        self.stall_threshold = stall_threshold
        self.stall_window = stall_window
        self.max_transitions = max_transitions
        self.ee_pos_history = []
        self.transition_count = 0
        self.in_transition = False

    def reset(self):
        self.ee_pos_history = []
        self.transition_count = 0
        self.in_transition = False
        self.stall_threshold = self.default_stall_threshold
        self.stall_window = self.default_stall_window

    def configure_for_task(self, control_profile: dict):
        self.stall_threshold = control_profile.get("transition_stall_threshold", self.default_stall_threshold)
        self.stall_window = control_profile.get("transition_stall_window", self.default_stall_window)

    def record_ee_pos(self, ee_pos):
        self.ee_pos_history.append(ee_pos.copy())

    def detect_state_gap(self):
        if self.transition_count >= self.max_transitions:
            return False
        if len(self.ee_pos_history) < self.stall_window:
            return False
        recent = self.ee_pos_history[-self.stall_window:]
        displacements = [np.linalg.norm(recent[i+1] - recent[i]) for i in range(len(recent)-1)]
        avg_disp = np.mean(displacements)
        return avg_disp < self.stall_threshold

    def trigger_transition(self):
        self.in_transition = True
        self.transition_count += 1
        self.ee_pos_history = []
        logger.info("[Transition] State Gap detected! Triggering transition #%d", self.transition_count)

    def finish_transition(self):
        self.in_transition = False


# ===== Failure Taxonomy (rule-based, no extra VLM) =====
class FailureTaxonomy:
    def __init__(
        self,
        *,
        window: int = 12,
        stall_disp: float = TRANSITION_STALL_THRESHOLD,
        collision_action_mag: float = 0.06,
        misgrasp_gripper_open: float = 0.75,
        misgrasp_close_cmd: float = -0.2,
        slip_open_high: float = 0.7,
        slip_open_low: float = 0.35,
    ) -> None:
        self.window = int(window)
        self.stall_disp = float(stall_disp)
        self.collision_action_mag = float(collision_action_mag)
        self.misgrasp_gripper_open = float(misgrasp_gripper_open)
        self.misgrasp_close_cmd = float(misgrasp_close_cmd)
        self.slip_open_high = float(slip_open_high)
        self.slip_open_low = float(slip_open_low)
        self.ee_pos_hist: list[np.ndarray] = []
        self.action_hist: list[np.ndarray] = []
        self.gripper_hist: list[np.ndarray] = []

    def reset(self) -> None:
        self.ee_pos_hist = []
        self.action_hist = []
        self.gripper_hist = []

    def update(self, *, ee_pos: np.ndarray, action: np.ndarray, gripper_qpos: np.ndarray) -> None:
        self.ee_pos_hist.append(np.asarray(ee_pos, dtype=np.float32).copy())
        self.action_hist.append(np.asarray(action, dtype=np.float32).copy())
        self.gripper_hist.append(np.asarray(gripper_qpos, dtype=np.float32).copy())
        if len(self.ee_pos_hist) > self.window:
            self.ee_pos_hist = self.ee_pos_hist[-self.window :]
        if len(self.action_hist) > self.window:
            self.action_hist = self.action_hist[-self.window :]
        if len(self.gripper_hist) > self.window:
            self.gripper_hist = self.gripper_hist[-self.window :]

    def _avg_disp(self) -> float:
        if len(self.ee_pos_hist) < 2:
            return 0.0
        recent = self.ee_pos_hist
        displacements = [float(np.linalg.norm(recent[i + 1] - recent[i])) for i in range(len(recent) - 1)]
        return float(np.mean(displacements)) if displacements else 0.0

    def _avg_action_mag(self) -> float:
        if not self.action_hist:
            return 0.0
        mags = [float(np.linalg.norm(a[:3])) for a in self.action_hist]
        return float(np.mean(mags)) if mags else 0.0

    def classify(self) -> tuple[str, str]:
        if len(self.ee_pos_hist) < self.window:
            return "", ""

        avg_disp = self._avg_disp()
        avg_act = self._avg_action_mag()

        last_a = self.action_hist[-1] if self.action_hist else np.zeros(7, dtype=np.float32)
        open_fracs = []
        for g in self.gripper_hist:
            gg = np.asarray(g, dtype=np.float32).reshape(-1)
            if gg.size:
                open_fracs.append(float(np.clip(np.mean(gg), 0.0, 1.0)))
        open_frac = open_fracs[-1] if open_fracs else 0.0

        if (
            float(last_a[6]) <= self.misgrasp_close_cmd
            and open_fracs
            and float(np.min(open_fracs)) <= self.slip_open_low
            and open_frac >= self.slip_open_high
        ):
            return "slip", f"close_cmd={float(last_a[6]):.3f} gripper_open={open_frac:.2f}"

        if avg_disp < self.stall_disp:
            if avg_act > self.collision_action_mag:
                return "collision", f"avg_disp={avg_disp:.4f} avg_act={avg_act:.4f}"
            return "stall", f"avg_disp={avg_disp:.4f}"

        if float(last_a[6]) <= self.misgrasp_close_cmd and open_frac >= self.misgrasp_gripper_open:
            return "misgrasp", f"close_cmd={float(last_a[6]):.3f} gripper_open={open_frac:.2f}"

        return "", ""


# ===== Graph RAG + Memory (A3) =====
class GraphRAGMemory:
    """Injects scene priors from Topo-Graph RAG and Evo-KAM Memory."""

    def __init__(self):
        self.priors_db = SCENE_PRIORS.copy()
        self.episode_memory = {}

    def extract_priors_for_task(self, task_description):
        task_lower = task_description.lower()
        object_choice = None
        object_priority = -1
        target_hints = []
        release_hints = []
        matched_keywords = []
        priors = {
            "matched_keywords": matched_keywords,
            "target_hints": target_hints,
            "release_hints": release_hints,
        }
        for obj_name, obj_priors in self.priors_db.items():
            if not _keyword_match(task_lower, obj_name):
                continue
            matched_keywords.append(obj_name)
            role = obj_priors.get("role", "manipulated_object")
            logger.info("[GraphRAG] Matched prior '%s': %s", obj_name, obj_priors)
            if role == "manipulated_object":
                priority = obj_priors.get("priority", 0)
                if priority > object_priority:
                    object_priority = priority
                    object_choice = obj_priors
            else:
                if target_hint := obj_priors.get("target_hint"):
                    target_hints.append(target_hint)
                if release_hint := obj_priors.get("release_hint"):
                    release_hints.append(release_hint)
        if object_choice:
            priors.update({
                key: value
                for key, value in object_choice.items()
                if key not in {"role", "priority"}
            })
        if task_description in self.episode_memory:
            priors["memory_hint"] = self.episode_memory[task_description]
        return priors

    def augment_prompt(self, task_description, priors):
        if not priors or len(priors) <= 3:
            return task_description
        augmentations = []
        if "z_offset" in priors:
            augmentations.append(f"keep a slight height margin of about {priors['z_offset']:.3f} meters")
        if "force" in priors:
            augmentations.append(f"use {priors['force']}")
        if "yaw_hint" in priors:
            augmentations.append(priors["yaw_hint"].replace("_", " "))
        if "stability_hint" in priors:
            augmentations.append(priors["stability_hint"].replace("_", " "))
        for target_hint in priors.get("target_hints", []):
            augmentations.append(target_hint.replace("_", " "))
        for release_hint in priors.get("release_hints", []):
            augmentations.append(release_hint.replace("_", " "))
        if "memory_hint" in priors:
            augmentations.append(priors["memory_hint"])
        if augmentations:
            return f"{task_description}, {', '.join(augmentations)}"
        return task_description

    def record_success(self, task_description, priors):
        memory_parts = []
        if "yaw_hint" in priors:
            memory_parts.append(priors["yaw_hint"].replace("_", " "))
        if "stability_hint" in priors:
            memory_parts.append(priors["stability_hint"].replace("_", " "))
        if priors.get("target_hints"):
            memory_parts.append(priors["target_hints"][0].replace("_", " "))
        memory_hint = "repeat the previously successful stable strategy"
        if memory_parts:
            memory_hint = f"repeat the previously successful stable strategy: {', '.join(memory_parts)}"
        self.episode_memory[task_description] = memory_hint
        logger.info("[EvoKAM] Recorded success for '%s': %s", task_description, memory_hint)


# ===== Critic Agent (A4) =====
class CriticAgent:
    """Reflective Critic with Qwen3-VL for error attribution and retry."""

    def __init__(self, qwen_model_name="Qwen/Qwen3-VL-8B-Instruct", temp=0.3,
                 quant_mode="4bit",
                 check_interval=CRITIC_CHECK_INTERVAL,
                 max_retries=CRITIC_MAX_RETRIES):
        self.qwen_model_name = qwen_model_name
        self.temp = temp
        self.quant_mode = quant_mode
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.model = None
        self.processor = None
        self.retry_count = 0
        self.min_control_steps = CRITIC_MIN_CONTROL_STEPS

    def reset(self):
        self.retry_count = 0
        self.min_control_steps = CRITIC_MIN_CONTROL_STEPS

    def configure_for_task(self, control_profile: dict):
        self.min_control_steps = control_profile.get("critic_min_steps", CRITIC_MIN_CONTROL_STEPS)

    def load_model(self):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            import torch

            logger.info("[Critic] Loading Qwen3-VL: %s (quant=%s)...", self.qwen_model_name, self.quant_mode)

            # Build quantization config
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
            if self.quant_mode == "4bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("[Critic] Using bitsandbytes NF4 4-bit quantization.")
            elif self.quant_mode == "8bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                logger.info("[Critic] Using bitsandbytes 8-bit quantization.")
            else:
                logger.info("[Critic] No quantization (full precision).")

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.qwen_model_name, **kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(self.qwen_model_name)
            logger.info("[Critic] Qwen3-VL loaded successfully (quant=%s).", self.quant_mode)
            return True
        except Exception as e:
            logger.error("[Critic] Failed to load Qwen3-VL: %s", e)
            return False

    def unload_model(self):
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[Critic] Qwen3-VL unloaded, GPU memory freed.")
        except Exception as e:
            logger.warning("[Critic] Unload error: %s", e)

    def judge_subtask(self, image, task_description):
        if self.model is None or self.processor is None:
            return self._heuristic_judge()

        try:
            from PIL import Image as PILImage
            import torch

            pil_img = PILImage.fromarray(image)
            prompt_text = (
                f"You are a robotic task critic. Task: '{task_description}'.\n"
                "Be conservative. Only report stuck=true if the robot is clearly stalled or trembling "
                "for an extended moment rather than making slow but valid progress.\n"
                f"Analyze the image and answer:\n"
                f"1. Is the current subtask completed? (yes/no)\n"
                f"2. Is the gripper clearly stuck or trembling? (yes/no)\n"
                f"3. If not completed, error type? (collision/slip/misgrasp/other)\n"
                f'Respond JSON: {{"completed": bool, "stuck": bool, "error_type": str}}'
            )

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text},
                ]}
            ]

            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_input],
                images=[pil_img],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=self.temp, do_sample=self.temp > 0,
                )

            output_text = self.processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )[0].strip()

            logger.info("[Critic] Qwen3-VL judgment: %s", output_text)
            return self._parse_critic_output(output_text)
        except Exception as e:
            logger.warning("[Critic] Inference failed: %s, falling back to heuristic.", e)
            return self._heuristic_judge()

    def _heuristic_judge(self):
        return {"completed": False, "stuck": False, "error_type": "unknown"}

    def _parse_critic_output(self, text):
        import re
        result = {"completed": False, "stuck": False, "error_type": "unknown"}
        try:
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                parsed = json.loads(json_match.group())
                result["completed"] = parsed.get("completed", False)
                result["stuck"] = parsed.get("stuck", False)
                result["error_type"] = parsed.get("error_type", "unknown")
                return result
        except json.JSONDecodeError:
            pass
        text_lower = text.lower()
        result["completed"] = "yes" in text_lower or "completed" in text_lower
        result["stuck"] = "stuck" in text_lower or "trembling" in text_lower
        for etype in ["collision", "slip", "misgrasp"]:
            if etype in text_lower:
                result["error_type"] = etype
                break
        return result

    def get_retry_prompt(self, task_description, judgment):
        error_type = judgment.get("error_type", "unknown")
        retry_prompts = {
            "collision": f"carefully avoid obstacles and {task_description}",
            "slip": f"grip more firmly and {task_description}",
            "misgrasp": f"reposition and carefully {task_description}",
            "unknown": f"try again carefully: {task_description}",
        }
        return retry_prompts.get(error_type, retry_prompts["unknown"])

    def should_retry(self, judgment):
        if self.retry_count >= self.max_retries:
            return False
        if judgment.get("completed", False):
            return False
        if judgment.get("stuck", False):
            return True
        return False


def _infer_with_retry(client, payload, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.infer(payload)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                logger.warning("Inference attempt %d failed: %s, retrying in %ds...", attempt + 1, e, wait)
                time.sleep(wait)
            else:
                raise


def _wrap_carve_client(client, *, trace_jsonl=""):
    from agentic_vla.runtime import ActionSpec, CarveRuntime, JsonlTraceSink
    from agentic_vla.runtime.adapters import LegacyPolicyClientBridge, Pi05Adapter

    adapter = Pi05Adapter(
        client,
        adapter_id="pi05-libero-websocket",
        precision="bf16",
        action_spec=ActionSpec(
            action_dim=7,
            representation="normalized_delta_cartesian_pose",
            coordinate_frame="robot_base",
            gripper_convention="-1=close,+1=open",
            control_frequency_hz=20.0,
            normalization_id="physical-intelligence/libero",
        ),
    )
    trace_sink = JsonlTraceSink(trace_jsonl) if trace_jsonl else None
    runtime = CarveRuntime(adapter, fallback_mode="graceful", trace_sink=trace_sink)
    logger.info(
        "CARVE runtime enabled | adapter=%s | dynamic_steps=%s",
        adapter.adapter_id,
        adapter.capabilities.configurable_inference_steps,
    )
    return LegacyPolicyClientBridge(runtime)


def _create_client(host, port, max_attempts=10, *, carve_runtime=False, carve_trace_jsonl=""):
    import websockets.sync.client
    from openpi_client import msgpack_numpy as _msgpack_numpy
    from openpi_client import websocket_client_policy as _ws_policy
    uri = f"ws://{host}:{port}"
    for attempt in range(max_attempts):
        try:
            conn = websockets.sync.client.connect(
                uri, compression=None, max_size=None,
                ping_interval=120, ping_timeout=120, close_timeout=30,
            )
            metadata = _msgpack_numpy.unpackb(conn.recv())
            client = _ws_policy.WebsocketClientPolicy.__new__(_ws_policy.WebsocketClientPolicy)
            client._uri = uri
            client._api_key = None
            client._packer = _msgpack_numpy.Packer()
            client._ws = conn
            client._server_metadata = metadata
            logger.info("Created client at %s:%d", host, port)
            return (
                _wrap_carve_client(client, trace_jsonl=carve_trace_jsonl)
                if carve_runtime
                else client
            )
        except Exception as e:
            wait = min(2 ** attempt, 30)
            logger.warning("Client creation attempt %d failed: %s, retrying in %ds...", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to create client at {host}:{port} after {max_attempts} attempts")


def _restart_policy_server(openpi_root, port, env="LIBERO"):
    import subprocess
    logger.warning("Restarting policy server on port %d...", port)
    try:
        subprocess.run(["pkill", "-f", f"serve_policy.py.*--port.*{port}"], capture_output=True, timeout=10)
        time.sleep(3)
    except Exception:
        pass
    env_vars = os.environ.copy()
    env_vars["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env_vars.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
    openpi_path = pathlib.Path(openpi_root)
    env_vars["PYTHONPATH"] = os.pathsep.join(
        [
            str(openpi_path / "src"),
            str(openpi_path / "packages" / "openpi-client" / "src"),
            env_vars.get("PYTHONPATH", ""),
        ]
    )
    policy_config = os.environ.get("AGENTIC_VLA_POLICY_CONFIG", "pi05_libero")
    policy_dir = os.environ.get(
        "AGENTIC_VLA_POLICY_DIR",
        str(pathlib.Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_libero_pytorch"),
    )
    policy_python = os.environ.get("AGENTIC_VLA_POLICY_PYTHON", sys.executable)
    cmd = [
        policy_python,
        str(openpi_path / "scripts" / "serve_policy.py"),
        "--port", str(port),
        "policy:checkpoint",
        f"--policy.config={policy_config}",
        f"--policy.dir={policy_dir}",
    ]
    proc = subprocess.Popen(cmd, cwd=str(openpi_root), env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("Policy server process started (PID %d), waiting...", proc.pid)
    time.sleep(30)
    return proc


def _reconnect_client(
    host,
    port,
    max_attempts=10,
    *,
    carve_runtime=False,
    carve_trace_jsonl="",
):
    return _create_client(
        host,
        port,
        max_attempts,
        carve_runtime=carve_runtime,
        carve_trace_jsonl=carve_trace_jsonl,
    )


def _build_ablation_tag(args):
    if getattr(args, "method_tag", ""):
        return args.method_tag
    if args.ablation_tag:
        return args.ablation_tag
    parts = []
    if getattr(args, "carve_joint_controller", False):
        parts.append("CARVE-Joint")
    if args.transition or _is_ba_harness(args):
        parts.append("A2-Transition")
    if args.graph_rag:
        parts.append("A3-GraphRAG")
    if args.critic:
        parts.append("A4-Critic")
    if args.vision_prompt:
        parts.append("VP")
    if not parts:
        return "A1-baseline"
    return "+".join(parts)


def _is_ba_harness(args: argparse.Namespace) -> bool:
    return getattr(args, "method_tag", "") in {
        "B4-Agentic",
        "B4-Agentic-Light",
        "B4-Agentic-Quant",
        "B4-Agentic-CAQ",
        "B4-Agentic-CAQ-Proxy",
        "B4-BAHarness-Rules",
        "B4-Async",
        "B4-AAC-Lite",
    }


def _is_long_horizon_multi_object_task(task_description: str) -> bool:
    task_lower = task_description.lower()
    if "both" in task_lower and "moka pot" in task_lower and ("stove" in task_lower or "burner" in task_lower):
        return True
    relation_markers = (
        "right of",
        "left of",
        "next to",
        "on the plate",
        "in the bowl",
        "and put",
    )
    multi_object_markers = (
        "mug",
        "pudding",
        "plate",
        "bowl",
        "drawer",
        "basket",
        "box",
    )
    return (
        " and " in task_lower
        and any(marker in task_lower for marker in relation_markers)
        and sum(1 for marker in multi_object_markers if marker in task_lower) >= 2
    )


def _ba_risk_score(
    *,
    state_gap: bool,
    failure_event: str,
    t: int,
    transition_min_steps: int,
    current_subgoal: str,
    recovery_cooldown_steps: int,
) -> float:
    score = 0.0
    if state_gap:
        score += 0.35
    if failure_event in BA_HIGH_RISK_EVENTS:
        score += 0.45
    if t >= transition_min_steps + BA_TRANSITION_EXTRA_MIN_STEPS:
        score += 0.10
    if current_subgoal != "execute":
        score += 0.10
    if recovery_cooldown_steps > 0:
        score -= 0.35
    return float(max(0.0, min(1.0, score)))


def _ba_should_recover(
    *,
    failure_event: str,
    risk_score: float,
    t: int,
    recovery_count: int,
    max_recoveries: int,
    recovery_cooldown_steps: int,
    risk_threshold: float,
) -> bool:
    if recovery_count >= max(0, int(max_recoveries)):
        return False
    if recovery_cooldown_steps > 0:
        return False
    if t < BA_RECOVERY_MIN_STEPS:
        return False
    return failure_event in BA_HIGH_RISK_EVENTS and risk_score >= float(risk_threshold)


def _ba_should_transition(
    *,
    state_gap: bool,
    failure_event: str,
    allowed_failure_events: set[str],
    risk_score: float,
    t: int,
    transition_min_steps: int,
    recovery_cooldown_steps: int,
    risk_threshold: float,
) -> bool:
    if not state_gap:
        return False
    if recovery_cooldown_steps > 0:
        return False
    if t < transition_min_steps + BA_TRANSITION_EXTRA_MIN_STEPS:
        return False
    if failure_event not in allowed_failure_events:
        return False
    return risk_score >= float(risk_threshold)


def _inner_libero_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _estimate_stove_object_status(
    env: Any,
    obs: dict[str, Any],
    object_name: str,
) -> dict[str, Any] | None:
    inner_env = _inner_libero_env(env)
    object_states = getattr(inner_env, "object_states_dict", {}) or {}
    cook_region = object_states.get("flat_stove_1_cook_region")
    object_state = object_states.get(object_name)
    on_cook_region = False
    if cook_region is not None and object_state is not None:
        try:
            on_cook_region = bool(cook_region.check_ontop(object_state))
        except Exception:
            on_cook_region = False

    object_pos = obs.get(f"{object_name}_pos")
    site_pos = None
    distance_xy = None
    try:
        site_pos = np.asarray(inner_env.sim.data.get_site_xpos("flat_stove_1_cook_region"), dtype=np.float32)
        if object_pos is not None:
            object_pos = np.asarray(object_pos, dtype=np.float32)
            distance_xy = float(np.linalg.norm(object_pos[:2] - site_pos[:2]))
            if not on_cook_region:
                on_cook_region = distance_xy < 0.075 and float(object_pos[2]) > float(site_pos[2]) + 0.015
    except Exception:
        site_pos = None
        distance_xy = None

    if object_pos is None:
        return None
    near_target = bool(distance_xy is not None and distance_xy < 0.16)
    needs_recenter = bool(near_target and not on_cook_region)
    return {
        "object": object_name,
        "on_target": bool(on_cook_region),
        "near_target": near_target,
        "needs_recenter": needs_recenter,
        "target_distance_xy": distance_xy,
        "object_pos": np.asarray(object_pos, dtype=np.float32).tolist(),
        "target_pos": np.asarray(site_pos, dtype=np.float32).tolist() if site_pos is not None else None,
    }


def _get_obs_object_pos(obs: dict[str, Any], object_name: str) -> np.ndarray | None:
    object_pos = obs.get(f"{object_name}_pos")
    if object_pos is None:
        return None
    try:
        return np.asarray(object_pos, dtype=np.float32)
    except Exception:
        return None


def _estimate_on_object_status(
    env: Any,
    obs: dict[str, Any],
    *,
    object_name: str,
    target_name: str,
    relation_name: str,
) -> dict[str, Any] | None:
    inner_env = _inner_libero_env(env)
    object_states = getattr(inner_env, "object_states_dict", {}) or {}
    object_state = object_states.get(object_name)
    target_state = object_states.get(target_name)
    on_target = False
    if object_state is not None and target_state is not None:
        try:
            on_target = bool(target_state.check_ontop(object_state))
        except Exception:
            on_target = False

    object_pos = _get_obs_object_pos(obs, object_name)
    target_pos = _get_obs_object_pos(obs, target_name)
    distance_xy = None
    if object_pos is not None and target_pos is not None:
        distance_xy = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        if not on_target:
            on_target = bool(distance_xy < 0.035 and float(object_pos[2]) > float(target_pos[2]))
    if object_pos is None or target_pos is None:
        return None

    near_target = bool(distance_xy is not None and distance_xy < 0.13)
    needs_recenter = bool(near_target and not on_target)
    return {
        "object": object_name,
        "target": target_name,
        "relation": relation_name,
        "on_target": bool(on_target),
        "near_target": near_target,
        "needs_recenter": needs_recenter,
        "target_distance_xy": distance_xy,
        "object_pos": object_pos.tolist(),
        "target_pos": target_pos.tolist(),
    }


def _estimate_right_of_status(
    obs: dict[str, Any],
    *,
    object_name: str,
    target_name: str,
    relation_name: str,
) -> dict[str, Any] | None:
    object_pos = _get_obs_object_pos(obs, object_name)
    target_pos = _get_obs_object_pos(obs, target_name)
    if object_pos is None or target_pos is None:
        return None
    delta = object_pos[:2] - target_pos[:2]
    # LIBERO tabletop right regions in these scenes are offset along +Y.
    right_offset = float(delta[1])
    lateral_error = float(abs(delta[0]))
    distance_xy = float(np.linalg.norm(delta))
    on_target = bool(0.045 <= right_offset <= 0.22 and lateral_error <= 0.12)
    near_target = bool(distance_xy < 0.24)
    needs_recenter = bool(near_target and not on_target)
    return {
        "object": object_name,
        "target": target_name,
        "relation": relation_name,
        "on_target": on_target,
        "near_target": near_target,
        "needs_recenter": needs_recenter,
        "target_distance_xy": distance_xy,
        "right_offset": right_offset,
        "lateral_error": lateral_error,
        "object_pos": object_pos.tolist(),
        "target_pos": target_pos.tolist(),
    }


def _estimate_plate_relation_progress(
    env: Any,
    obs: dict[str, Any],
    task_description: str,
) -> dict[str, Any] | None:
    task_lower = task_description.lower()
    statuses: list[dict[str, Any]] = []

    if "left plate" in task_lower and "right plate" in task_lower and "yellow and white mug" in task_lower:
        candidates = [
            ("porcelain_mug_1", "plate_1", "white_mug_on_left_plate"),
            ("white_yellow_mug_1", "plate_2", "yellow_white_mug_on_right_plate"),
        ]
        for object_name, target_name, relation_name in candidates:
            status = _estimate_on_object_status(
                env,
                obs,
                object_name=object_name,
                target_name=target_name,
                relation_name=relation_name,
            )
            if status is not None:
                statuses.append(status)
    elif "white mug on the plate" in task_lower and "right of the plate" in task_lower:
        mug_status = _estimate_on_object_status(
            env,
            obs,
            object_name="porcelain_mug_1",
            target_name="plate_1",
            relation_name="white_mug_on_plate",
        )
        pudding_status = _estimate_right_of_status(
            obs,
            object_name="chocolate_pudding_1",
            target_name="plate_1",
            relation_name="pudding_right_of_plate",
        )
        statuses.extend([status for status in (mug_status, pudding_status) if status is not None])

    if not statuses:
        return None

    completed_objects = [str(status["object"]) for status in statuses if status["on_target"]]
    recenter_candidates = [status for status in statuses if status["needs_recenter"]]
    completed_count = len(completed_objects)
    if completed_count >= len(statuses):
        phase = "done"
    elif recenter_candidates:
        phase = "recenter_plate_relation"
    elif completed_count > 0:
        phase = "continue_remaining_relation"
    else:
        phase = "seek_first_relation"

    target_status = None
    if recenter_candidates:
        target_status = min(
            recenter_candidates,
            key=lambda item: float(item.get("target_distance_xy") if item.get("target_distance_xy") is not None else 1e6),
        )
    remaining_objects = [str(status["object"]) for status in statuses if str(status["object"]) not in completed_objects]
    return {
        "kind": "plate_relation",
        "phase": phase,
        "completed_count": completed_count,
        "completed_objects": completed_objects,
        "remaining_objects": remaining_objects,
        "target_object": target_status.get("object") if target_status else (remaining_objects[0] if remaining_objects else None),
        "target_relation": target_status.get("relation") if target_status else None,
        "needs_recenter": bool(target_status is not None),
        "target_distance_xy": target_status.get("target_distance_xy") if target_status else None,
        "score": float(completed_count / max(1, len(statuses))),
    }


def _estimate_task_progress(env: Any, obs: dict[str, Any], task_description: str) -> dict[str, Any] | None:
    task_lower = task_description.lower()
    if not ("moka pot" in task_lower and ("stove" in task_lower or "burner" in task_lower)):
        return _estimate_plate_relation_progress(env, obs, task_description)

    moka_objects = [name for name in ("moka_pot_1", "moka_pot_2") if obs.get(f"{name}_pos") is not None]
    if not moka_objects:
        moka_objects = ["moka_pot_1", "moka_pot_2"]
    statuses = []
    for object_name in moka_objects:
        status = _estimate_stove_object_status(env, obs, object_name)
        if status is not None:
            statuses.append(status)
    if not statuses:
        return None

    if "both" not in task_lower:
        status = min(
            statuses,
            key=lambda item: float(item.get("target_distance_xy") if item.get("target_distance_xy") is not None else 1e6),
        )
        completed_count = 1 if status["on_target"] else 0
        phase = "done" if status["on_target"] else ("recenter_moka_pot" if status["needs_recenter"] else "approach_moka_pot")
        distance = status.get("target_distance_xy")
        score = 1.0 if status["on_target"] else max(0.0, 1.0 - min(float(distance or 1.0) / 0.20, 1.0))
        return {
            "kind": "single_moka_pot_stove",
            "phase": phase,
            "completed_count": completed_count,
            "completed_objects": [status["object"]] if status["on_target"] else [],
            "remaining_objects": [] if status["on_target"] else [status["object"]],
            "target_object": status["object"],
            "needs_recenter": bool(status["needs_recenter"]),
            "target_distance_xy": distance,
            "score": float(score),
        }

    completed_objects = []
    recenter_candidates = []
    for status in statuses:
        object_name = str(status["object"])
        if status["on_target"]:
            completed_objects.append(object_name)
        elif status["needs_recenter"]:
            recenter_candidates.append(status)

    all_objects = [str(status["object"]) for status in statuses]
    remaining_objects = [name for name in all_objects if name not in completed_objects]
    completed_count = len(completed_objects)
    if completed_count >= 2:
        phase = "done"
    elif recenter_candidates:
        phase = "recenter_moka_pot"
    elif completed_count == 1:
        phase = "seek_second_object"
    else:
        phase = "seek_first_object"
    target_status = None
    if recenter_candidates:
        target_status = min(
            recenter_candidates,
            key=lambda item: float(item.get("target_distance_xy") if item.get("target_distance_xy") is not None else 1e6),
        )
    return {
        "kind": "moka_pot_stove",
        "phase": phase,
        "completed_count": completed_count,
        "completed_objects": completed_objects,
        "remaining_objects": remaining_objects,
        "target_object": target_status.get("object") if target_status else (remaining_objects[0] if remaining_objects else None),
        "needs_recenter": bool(target_status is not None),
        "target_distance_xy": target_status.get("target_distance_xy") if target_status else None,
        "score": float(completed_count / 2.0),
    }


def _progress_should_force_transition(
    progress: dict[str, Any] | None,
    *,
    failure_event: str,
    state_gap: bool,
    timestep: int,
    transition_min_steps: int,
    recovery_cooldown_steps: int,
) -> bool:
    if not progress or progress.get("kind") != "moka_pot_stove":
        return False
    if recovery_cooldown_steps > 0:
        return False
    if progress.get("needs_recenter"):
        return False
    completed_count = int(progress.get("completed_count", 0))
    if completed_count <= 0:
        if timestep < max(int(transition_min_steps), 260):
            return False
        return bool(state_gap or failure_event in BA_HIGH_RISK_EVENTS)
    if completed_count == 1:
        if timestep < max(int(transition_min_steps), 180):
            return False
        return bool(state_gap or failure_event in BA_HIGH_RISK_EVENTS)
    return False


def _phase_aware_transition_allowed(
    progress: dict[str, Any] | None,
    *,
    failure_event: str,
    state_gap: bool,
    timestep: int,
    transition_min_steps: int,
    recovery_cooldown_steps: int,
) -> bool:
    if not progress or progress.get("kind") != "moka_pot_stove":
        return True
    if recovery_cooldown_steps > 0:
        return False
    if progress.get("needs_recenter"):
        return False
    completed_count = int(progress.get("completed_count", 0))
    if completed_count <= 0:
        return False
    if completed_count >= 2:
        return False
    if timestep < max(int(transition_min_steps), 260):
        return False
    return bool(state_gap and failure_event in {"stall", "collision"})


def _progress_should_force_recovery(
    progress: dict[str, Any] | None,
    *,
    timestep: int,
    recovery_count: int,
    max_recoveries: int,
    recovery_cooldown_steps: int,
    min_steps: int = BA_PROGRESS_RECOVERY_MIN_STEPS,
    far_min_steps: int = -1,
    far_distance: float = 0.20,
    regression_recovery: bool = False,
    regression_distance: float = 0.12,
    best_completed_count: int = 0,
) -> bool:
    if not progress:
        return False
    if recovery_count >= max(0, int(max_recoveries)):
        return False
    if recovery_cooldown_steps > 0:
        return False
    if progress.get("needs_recenter"):
        return timestep >= int(min_steps)
    if (
        regression_recovery
        and progress.get("kind") == "single_moka_pot_stove"
        and int(best_completed_count) > int(progress.get("completed_count", 0))
        and timestep >= int(min_steps)
    ):
        target_distance = progress.get("target_distance_xy")
        if target_distance is not None and float(target_distance) >= float(regression_distance):
            return True
    if (
        int(far_min_steps) >= 0
        and progress.get("kind") == "single_moka_pot_stove"
        and int(progress.get("completed_count", 0)) == 0
        and timestep >= int(far_min_steps)
    ):
        target_distance = progress.get("target_distance_xy")
        if target_distance is not None and float(target_distance) >= float(far_distance):
            return True
    if (
        int(far_min_steps) >= 0
        and progress.get("kind") == "plate_relation"
        and progress.get("phase") != "done"
        and int(progress.get("completed_count", 0)) < 2
        and timestep >= int(far_min_steps)
    ):
        return True
    return False


def _ba_risk_bucket(risk_score: float, failure_event: str, progress: dict[str, Any] | None) -> str:
    if failure_event in {"misgrasp", "slip"}:
        return "critical"
    if progress and progress.get("needs_recenter"):
        return "high"
    if risk_score >= 0.80:
        return "high"
    if risk_score >= 0.50:
        return "medium"
    if progress and progress.get("kind") == "moka_pot_stove" and int(progress.get("completed_count", 0)) == 1:
        return "medium"
    return "low"


def _ba_aac_commit_steps(
    *,
    enabled: bool,
    default_steps: int,
    risk_bucket: str,
    current_subgoal: str,
    profile: dict[str, Any],
    progress: dict[str, Any] | None,
    min_commit: int,
    medium_commit: int,
    high_commit: int,
    max_commit: int,
) -> int:
    if not enabled:
        return int(default_steps)
    max_allowed = max(int(min_commit), min(int(max_commit), int(default_steps)))
    min_allowed = max(1, min(int(min_commit), max_allowed))
    medium_allowed = max(min_allowed, min(int(medium_commit), max_allowed))
    high_allowed = max(min_allowed, min(int(high_commit), max_allowed))

    if not profile.get("ba_recovery_enabled", True) and not profile.get("ba_transition_enabled", False):
        return max_allowed
    if current_subgoal != "execute":
        return high_allowed
    if risk_bucket in {"critical", "high"}:
        return high_allowed
    if risk_bucket == "medium":
        return medium_allowed
    if progress and progress.get("kind") == "moka_pot_stove" and int(progress.get("completed_count", 0)) == 1:
        return medium_allowed
    return max_allowed


def evaluate_real_libero(args: argparse.Namespace) -> dict:
    _configure_isolated_libero_paths(LIBERO_SRC)
    if args.task_suite in LIBERO_PRO_SUITES:
        validate_libero_pro_root(LIBERO_SRC, args.task_suite)
    benchmark, get_libero_path, offscreen_render_env, segmentation_render_env, image_tools, websocket_client_policy = _require_runtime()

    np.random.seed(args.seed)
    pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    carve_joint_enabled = bool(getattr(args, "carve_joint_controller", False))
    carve_physical_enabled = bool(getattr(args, "carve_physical_recovery", False))
    carve_async_prefetch_enabled = bool(getattr(args, "carve_async_prefetch", False))
    carve_semantic_shadow_enabled = bool(getattr(args, "carve_semantic_shadow", False))
    carve_high_level_enabled = bool(getattr(args, "carve_high_level_agent", False))
    carve_canonical_harness_enabled = bool(
        getattr(args, "carve_canonical_harness", False)
    )
    if carve_joint_enabled and not args.carve_runtime:
        raise ValueError("--carve-joint-controller requires --carve-runtime")
    if carve_physical_enabled and not carve_joint_enabled:
        raise ValueError("--carve-physical-recovery requires --carve-joint-controller")
    if carve_async_prefetch_enabled and not carve_joint_enabled:
        raise ValueError("--carve-async-prefetch requires --carve-joint-controller")
    if carve_semantic_shadow_enabled and not carve_joint_enabled:
        raise ValueError("--carve-semantic-shadow requires --carve-joint-controller")
    if carve_high_level_enabled and not carve_joint_enabled:
        raise ValueError("--carve-high-level-agent requires --carve-joint-controller")
    if carve_canonical_harness_enabled and not carve_high_level_enabled:
        raise ValueError(
            "--carve-canonical-harness requires --carve-high-level-agent"
        )
    if bool(args.carve_agentic_knowledge) and not carve_canonical_harness_enabled:
        raise ValueError(
            "--carve-agentic-knowledge requires --carve-canonical-harness"
        )
    if bool(args.carve_semantic_before_recovery) and not (
        carve_canonical_harness_enabled and carve_physical_enabled
    ):
        raise ValueError(
            "--carve-semantic-before-recovery requires the canonical harness "
            "and physical recovery"
        )
    if carve_high_level_enabled and bool(getattr(args, "agentic_planner", False)):
        raise ValueError(
            "use either --carve-high-level-agent or the legacy --agentic-planner, not both"
        )
    if carve_semantic_shadow_enabled and str(args.perturbation) == "clean":
        logger.warning(
            "CARVE semantic shadow is event-triggered; clean episodes may issue no VLM calls"
        )
    if int(args.carve_semantic_cooldown_steps) < 0:
        raise ValueError("--carve-semantic-cooldown-steps must be non-negative")
    if min(
        float(args.carve_semantic_min_slack_ms),
        float(args.carve_semantic_timeout_sec),
    ) <= 0:
        raise ValueError("CARVE semantic slack and timeout values must be positive")
    if int(args.carve_semantic_max_tokens) <= 0:
        raise ValueError("--carve-semantic-max-tokens must be positive")
    if int(args.carve_high_level_max_calls) <= 0:
        raise ValueError("--carve-high-level-max-calls must be positive")
    if int(args.carve_high_level_max_tokens) <= 0:
        raise ValueError("--carve-high-level-max-tokens must be positive")
    if float(args.carve_high_level_boundary_timeout_sec) <= 0:
        raise ValueError("--carve-high-level-boundary-timeout-sec must be positive")
    if not 0.0 <= float(args.carve_high_level_min_confidence) <= 1.0:
        raise ValueError("--carve-high-level-min-confidence must be in [0, 1]")
    if (
        carve_async_prefetch_enabled
        and not 2 <= int(args.carve_prefetch_lead_actions) < int(args.carve_commit_steps)
    ):
        raise ValueError(
            "--carve-prefetch-lead-actions must be at least 2 and below "
            "--carve-commit-steps"
        )
    if carve_async_prefetch_enabled and int(args.carve_prefetch_interval) <= 0:
        raise ValueError("--carve-prefetch-interval must be positive")
    carve_prefix_mode = str(getattr(args, "carve_prefetch_prefix_mode", "off"))
    if carve_prefix_mode != "off" and not carve_async_prefetch_enabled:
        raise ValueError(
            "--carve-prefetch-prefix-mode requires --carve-async-prefetch"
        )
    if min(
        float(args.carve_prefetch_prefix_rms_max),
        float(args.carve_prefetch_prefix_translation_max),
        float(args.carve_prefetch_prefix_rotation_max),
    ) <= 0:
        raise ValueError("CARVE prefix-consistency limits must be positive")
    if not 0.0 <= float(args.carve_prefetch_prefix_gripper_min) <= 1.0:
        raise ValueError("--carve-prefetch-prefix-gripper-min must be in [0, 1]")
    if float(args.carve_physical_minimum_state_response) <= 0:
        raise ValueError("--carve-physical-minimum-state-response must be positive")
    if int(args.carve_max_recovery_attempts) <= 0:
        raise ValueError("--carve-max-recovery-attempts must be positive")
    if int(args.carve_monitor_warmup_steps) < 0:
        raise ValueError("--carve-monitor-warmup-steps must be non-negative")
    if int(args.carve_fixed_inference_steps) < 0:
        raise ValueError("--carve-fixed-inference-steps must be non-negative")
    if int(args.carve_fixed_inference_steps) > 0 and not args.carve_runtime:
        raise ValueError("--carve-fixed-inference-steps requires --carve-runtime")
    if carve_joint_enabled and int(args.carve_fixed_inference_steps) > 0:
        raise ValueError("joint and fixed inference-step controllers must be evaluated separately")
    if min(
        int(args.carve_fast_inference_steps),
        int(args.carve_accurate_inference_steps),
        int(args.carve_commit_steps),
    ) <= 0:
        raise ValueError("CARVE joint inference and commit budgets must be positive")
    if carve_joint_enabled and _is_ba_harness(args):
        raise ValueError("CARVE joint and the legacy B4 harness must be evaluated separately")
    if carve_joint_enabled:
        from agentic_vla.experiments import FailureSnapshotWriter
        from agentic_vla.runtime import (
            AgentIntent,
            AgenticKnowledgeProvider,
            ActionPrefixThresholds,
            ActionPrefixVerifier,
            ActionSpec,
            AsyncAgenticHarnessController,
            AsyncGuardedHighLevelAgent,
            AsyncInferencePrefetcher,
            AsyncSemanticObserver,
            DeadlineAwareSemanticScheduler,
            ExecutionMode,
            ExecutionRiskMonitor,
            FailureMemory,
            GuardedHighLevelAgent,
            HighLevelAgentConfig,
            HighLevelAgentContext,
            JointControllerConfig,
            JointRecoveryComputeController,
            MonitorConfig,
            OpenAICompatibleVisionPlanner,
            PausedExecutionSafeHoldAdapter,
            PrefetchContext,
            PrefetchDutyCycle,
            RecoveryContext,
            RecoveryMemory,
            RecoverySkillRegistry,
            StatefulRecoveryExecutor,
            SemanticObservationContext,
            SemanticScheduleConfig,
            TaskStartPolicy,
            compose_grounded_vla_instruction,
        )

        carve_monitor = ExecutionRiskMonitor(
            MonitorConfig(warmup_steps=int(args.carve_monitor_warmup_steps))
        )
        carve_recovery_skills = RecoverySkillRegistry.with_default_skills()
        carve_controller = JointRecoveryComputeController(
            JointControllerConfig(
                fast_inference_steps=int(args.carve_fast_inference_steps),
                accurate_inference_steps=int(args.carve_accurate_inference_steps),
                low_risk_commit=int(args.carve_commit_steps),
                high_risk_commit=int(args.carve_commit_steps),
                max_recovery_attempts=int(args.carve_max_recovery_attempts),
            )
        )
        carve_high_level_planner = (
            OpenAICompatibleVisionPlanner(
                    endpoint=str(args.carve_semantic_endpoint),
                    model=str(args.carve_semantic_model),
                    timeout_s=float(args.carve_semantic_timeout_sec),
                    max_tokens=int(args.carve_high_level_max_tokens),
            )
            if carve_high_level_enabled
            else None
        )
        carve_high_level_config = HighLevelAgentConfig(
            max_calls_per_episode=int(args.carve_high_level_max_calls),
            minimum_intervention_confidence=float(
                args.carve_high_level_min_confidence
            ),
        )
        carve_knowledge_provider = (
            AgenticKnowledgeProvider(build_libero_pro_haa_index())
            if bool(args.carve_agentic_knowledge)
            else None
        )
        # The single-flight executor is episode scoped. A stale request from one
        # episode must never consume the next episode's planner budget.
        carve_high_level_agent = None
        carve_async_high_level_agent = None
        carve_recovery_memory = RecoveryMemory(max_records=256)
        # Physical recovery outcomes and semantic planner evidence have distinct
        # schemas and retrieval policies; they must not share one store.
        carve_failure_memory = FailureMemory(max_records=256)
        carve_recovery_action_spec = ActionSpec(
            action_dim=7,
            representation="normalized_delta_cartesian_pose",
            coordinate_frame="robot_base",
            gripper_convention="-1=close,+1=open",
            control_frequency_hz=20.0,
            minimum=-1.0,
            maximum=1.0,
        )
        carve_prefix_verifier = (
            ActionPrefixVerifier(
                ActionPrefixThresholds(
                    continuous_rms_max=float(args.carve_prefetch_prefix_rms_max),
                    translation_endpoint_max=float(
                        args.carve_prefetch_prefix_translation_max
                    ),
                    rotation_endpoint_max=float(
                        args.carve_prefetch_prefix_rotation_max
                    ),
                    gripper_agreement_min=float(
                        args.carve_prefetch_prefix_gripper_min
                    ),
                )
            )
            if carve_prefix_mode != "off"
            else None
        )
        carve_snapshot_writer = (
            FailureSnapshotWriter(
                args.carve_branch_dir,
                max_per_episode=int(args.carve_max_branch_snapshots),
                minimum_gap_steps=int(args.carve_branch_min_gap_steps),
            )
            if args.carve_branch_dir
            else None
        )
        carve_semantic_scheduler_config = SemanticScheduleConfig(
            cooldown_steps=int(args.carve_semantic_cooldown_steps),
            min_deadline_slack_ms=float(args.carve_semantic_min_slack_ms),
            allowed_events=frozenset({"perturbation"}),
        )
    else:
        ExecutionMode = None
        carve_monitor = None
        carve_controller = None
        carve_snapshot_writer = None
        carve_recovery_memory = None
        carve_failure_memory = None
        carve_knowledge_provider = None
        carve_recovery_action_spec = None
        carve_prefix_verifier = None
        carve_semantic_scheduler_config = None
        carve_high_level_agent = None
        carve_high_level_planner = None
        carve_high_level_config = None
        carve_async_high_level_agent = None

    # Initialize agents based on ablation flags
    ba_harness_enabled = _is_ba_harness(args)
    transition_agent = TransitionAgent() if args.transition or ba_harness_enabled else None
    planner_trigger_agent = TransitionAgent() if args.agentic_planner and transition_agent is None else None
    failure_taxonomy = FailureTaxonomy(window=TRANSITION_STALL_WINDOW, stall_disp=TRANSITION_STALL_THRESHOLD)
    graph_rag_memory = GraphRAGMemory() if args.graph_rag or ba_harness_enabled else None
    critic_agent = CriticAgent(
        qwen_model_name=args.qwen_model, temp=args.critic_temp,
        quant_mode=args.qwen_quant,
    ) if args.critic else None

    # Load Critic model if needed
    if critic_agent is not None:
        if not critic_agent.load_model():
            logger.warning("[A4] Qwen3-VL load failed, Critic will use heuristic fallback.")

    ablation_tag = _build_ablation_tag(args)
    logger.info("=== Ablation: %s | Suite: %s ===", ablation_tag, args.task_suite)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    max_steps = _max_steps_for_suite(args.task_suite)
    if args.task_id is not None:
        task_ids = [int(args.task_id)]
    elif args.task_ids:
        task_ids = [int(t) for t in args.task_ids]
    else:
        task_ids = list(range(task_suite.n_tasks))
    skip_ids = set(int(t) for t in getattr(args, "skip_task_ids", []) or [])
    task_ids = [task_id for task_id in task_ids if int(task_id) not in skip_ids]
    if not task_ids:
        raise ValueError("No tasks selected after applying --task-id/--task-ids and --skip-task-ids.")
    args.evaluated_task_ids = [int(t) for t in task_ids]
    args.skipped_task_ids = sorted(skip_ids) if skip_ids else []

    results_path = pathlib.Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    carve_trace_path = (
        pathlib.Path(args.carve_trace_jsonl)
        if args.carve_trace_jsonl
        else results_path.parent / "carve_policy_calls.jsonl"
    )
    client = _create_client(
        args.host,
        args.port,
        carve_runtime=bool(args.carve_runtime),
        carve_trace_jsonl=str(carve_trace_path) if args.carve_runtime else "",
    )

    env_class = segmentation_render_env if args.vision_prompt else offscreen_render_env
    if args.vision_prompt:
        logger.info("Vision Prompt ENABLED (appendix) | mask_alpha=%.2f", args.mask_alpha)

    trace_path = pathlib.Path(args.episode_trace_jsonl) if args.episode_trace_jsonl else (
        results_path.parent / "episode_traces.jsonl"
    )
    episode_traces = _load_episode_traces(trace_path)
    completed_tasks_file = results_path.parent / "_completed_tasks.json"
    resume_state_file = results_path.parent / "_resume_state.json"
    partial_summary_file = results_path.parent / "_partial_summary.json"

    completed_tasks = set()
    resume_state = None
    if resume_state_file.exists():
        try:
            resume_state = json.loads(resume_state_file.read_text())
            logger.info("Loaded resume state from %s", resume_state_file)
        except Exception:
            resume_state = None

    if completed_tasks_file.exists():
        try:
            completed_tasks = set(json.loads(completed_tasks_file.read_text()))
            logger.info("Resuming: skipping already completed tasks %s", sorted(completed_tasks))
        except Exception:
            completed_tasks = set()
    if resume_state is not None:
        completed_tasks.update(str(task_id) for task_id in resume_state.get("completed_tasks", []))

    total_episodes = int(resume_state.get("total_episodes", 0)) if resume_state is not None else 0
    total_successes = int(resume_state.get("total_successes", 0)) if resume_state is not None else 0
    task_metrics = dict(resume_state.get("task_metrics", {})) if resume_state is not None else {}
    episode_lengths = list(resume_state.get("episode_lengths", [])) if resume_state is not None else []
    success_episode_lengths = list(resume_state.get("success_episode_lengths", [])) if resume_state is not None else []
    transition_stats = {"total_transitions": 0, "transitions_leading_to_success": 0}
    critic_stats = {"total_checks": 0, "retries_triggered": 0, "retries_leading_to_success": 0}
    planner_stats = {"total_requests": 0, "plans_generated": 0}
    expert_stats = _build_empty_expert_stats()
    planner_event_counts: dict[str, int] = collections.Counter()
    planner_action_counts: dict[str, int] = collections.Counter()
    planner_expert_counts: dict[str, int] = collections.Counter()
    if resume_state is not None and resume_state.get("transition_stats"):
        transition_stats.update(resume_state["transition_stats"])
    if resume_state is not None and resume_state.get("critic_stats"):
        critic_stats.update(resume_state["critic_stats"])
    if resume_state is not None and resume_state.get("expert_stats"):
        for key, values in resume_state["expert_stats"].items():
            if key in expert_stats and isinstance(values, dict):
                expert_stats[key].update(values)
    current_task_resume = resume_state.get("current_task") if resume_state is not None else None

    def _persist_progress(current_task: dict | None, status: str = "running") -> None:
        partial_results = _build_results_payload(
            args=args,
            ablation_tag=ablation_tag,
            total_successes=total_successes,
            total_episodes=total_episodes,
            episode_lengths=episode_lengths,
            success_episode_lengths=success_episode_lengths,
            task_metrics=task_metrics,
            transition_stats=transition_stats,
            critic_stats=critic_stats,
            planner_stats={
                **planner_stats,
                "event_counts": dict(planner_event_counts),
                "action_counts": dict(planner_action_counts),
                "expert_counts": dict(planner_expert_counts),
            },
            expert_stats=_serialize_expert_stats(expert_stats),
            episode_trace_path=str(trace_path),
            trace_aggregate=_aggregate_episode_traces(episode_traces),
            status=status,
            current_task=current_task,
        )
        partial_summary_file.write_text(json.dumps(partial_results, indent=2), encoding="utf-8")
        resume_payload = {
            "task_suite": args.task_suite,
            "ablation_tag": ablation_tag,
            "completed_tasks": sorted(completed_tasks),
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "episode_lengths": episode_lengths,
            "success_episode_lengths": success_episode_lengths,
            "task_metrics": task_metrics,
            "transition_stats": transition_stats if transition_agent is not None else None,
            "critic_stats": critic_stats if args.critic else None,
            "expert_stats": _serialize_expert_stats(expert_stats),
            "episode_trace_jsonl": str(trace_path),
            "trace_aggregate": _aggregate_episode_traces(episode_traces),
            "current_task": current_task,
        }
        resume_state_file.write_text(json.dumps(resume_payload, indent=2), encoding="utf-8")

    for task_id in task_ids:
        task_key = str(task_id)
        if task_key in completed_tasks:
            logger.info("Task %s already completed, skipping", task_id)
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env = _make_env(task, get_libero_path, env_class, args.seed)
        task_description = task.language

        # A3: Graph RAG prompt augmentation
        task_control_profile = _build_task_control_profile(task_description)
        if args.ba_safe_long_horizon_fast_path and _is_long_horizon_multi_object_task(task_description):
            task_control_profile.update({
                "ba_recovery_enabled": False,
                "ba_transition_enabled": False,
            })
        if args.ba_enable_recovery:
            task_control_profile["ba_recovery_enabled"] = True
        if transition_agent is not None:
            transition_agent.configure_for_task(task_control_profile)
        if critic_agent is not None:
            critic_agent.configure_for_task(task_control_profile)

        task_priors = {}
        effective_prompt = task_description
        if graph_rag_memory is not None:
            task_priors = graph_rag_memory.extract_priors_for_task(task_description)
            if not ba_harness_enabled or args.ba_augment_prompt:
                effective_prompt = graph_rag_memory.augment_prompt(task_description, task_priors)

        agentic_context = ""
        if task_priors:
            matched = task_priors.get("matched_keywords") if isinstance(task_priors.get("matched_keywords"), list) else []
            target_hints = task_priors.get("target_hints") if isinstance(task_priors.get("target_hints"), list) else []
            release_hints = task_priors.get("release_hints") if isinstance(task_priors.get("release_hints"), list) else []
            agentic_context = (
                f"priors.matched={','.join([str(x) for x in matched[:6]])}; "
                f"priors.target={';'.join([str(x) for x in target_hints[:3]])}; "
                f"priors.release={';'.join([str(x) for x in release_hints[:3]])}"
            ).strip()

        transition_prompt = _build_transition_prompt(task_description, task_priors)
        recovery_prompt = _build_recovery_prompt(task_description, task_priors)
        regrasp_prompt = _build_regrasp_prompt(task_description, task_priors)
        light_reuse_enabled_for_task = bool(args.light_reuse_actions) and not (
            bool(args.light_reuse_disable_long_horizon)
            and _is_long_horizon_multi_object_task(task_description)
        )

        if args.vision_prompt:
            env.reset()
            logger.info("Task %s | obj_of_interest=%s | instance_to_id=%s",
                        task_id, env.obj_of_interest, env.instance_to_id)

        resume_for_task = current_task_resume if (
            current_task_resume is not None and int(current_task_resume.get("task_id", -1)) == task_id
        ) else None

        start_episode_idx = int(resume_for_task.get("next_episode_idx", 0)) if resume_for_task else 0
        task_successes = int(resume_for_task.get("task_successes", 0)) if resume_for_task else 0
        task_episode_lengths = list(resume_for_task.get("task_episode_lengths", [])) if resume_for_task else []
        task_transition_total = int(resume_for_task.get("task_transition_total", 0)) if resume_for_task else 0
        task_transition_episodes = int(resume_for_task.get("task_transition_episodes", 0)) if resume_for_task else 0
        task_transition_successes = int(resume_for_task.get("task_transition_successes", 0)) if resume_for_task else 0
        task_retry_total = int(resume_for_task.get("task_retry_total", 0)) if resume_for_task else 0
        task_retry_episodes = int(resume_for_task.get("task_retry_episodes", 0)) if resume_for_task else 0
        task_retry_successes = int(resume_for_task.get("task_retry_successes", 0)) if resume_for_task else 0
        task_critic_checks = int(resume_for_task.get("task_critic_checks", 0)) if resume_for_task else 0
        resumed_task_expert_stats = _materialize_expert_stats(
            resume_for_task.get("task_expert_stats") if resume_for_task else None
        )
        task_expert_stats_before = _clone_expert_stats(expert_stats)
        if resume_for_task:
            logger.info(
                "Resuming task %s from trial %d/%d",
                task_id,
                min(start_episode_idx + 1, args.trials),
                args.trials,
            )

        for episode_idx in range(start_episode_idx, args.trials):
            logger.info("Task %s | Trial %d/%d | %s | Prompt: %s",
                        task_id, episode_idx + 1, args.trials, task_description, effective_prompt)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
            episode_rng = np.random.default_rng(int(args.seed) * 100000 + int(task_id) * 1000 + int(episode_idx))
            obs = _apply_initial_perturbation(
                env,
                obs,
                perturbation=str(args.perturbation),
                rng=episode_rng,
                object_jitter_xy=float(args.object_jitter_xy),
            )
            action_plan = collections.deque()
            action_age_plan = collections.deque()
            light_reuse_buffer = collections.deque()
            light_reuse_suffix_counted = False
            light_reuse_lockout_steps = 0
            replay_images = []
            done = False
            current_prompt = effective_prompt
            episode_steps = 0
            agentic_action_scale = 1.0
            agentic_apply_steps = 0
            planner_cooldown_steps = 0
            current_subgoal = "execute"
            subgoal_success_window = 0
            ba_recovery_count = 0
            carve_failure_streak = 0
            ba_recovery_cooldown_steps = 0
            ba_transition_lockout_steps = 0
            mid_episode_nudge_applied = False
            last_mid_episode_nudge_event = None
            ba_progress_best_completed_count = 0
            ba_progress_best_score = 0.0
            carve_last_action = np.asarray(LIBERO_DUMMY_ACTION, dtype=np.float32)
            carve_last_action_age = 0
            carve_last_control_ms = 0.0
            carve_decision = None
            carve_pending_decision = None
            carve_recovery_executor = (
                StatefulRecoveryExecutor(carve_recovery_memory)
                if carve_physical_enabled
                else None
            )
            carve_recovery_command = None
            carve_recovery_phase_start = None
            carve_recovery_initial_eef = None
            carve_recovery_phase_records = []
            carve_physical_active = False
            carve_post_recovery_event = None
            carve_skip_post_recovery_semantic = False
            carve_abort_episode = False
            carve_prefetcher = (
                AsyncInferencePrefetcher(
                    lambda payload: _infer_with_retry(client, payload, max_retries=2)
                )
                if carve_async_prefetch_enabled
                else None
            )
            carve_prefetch_schedule = (
                PrefetchDutyCycle(interval=int(args.carve_prefetch_interval))
                if carve_async_prefetch_enabled
                else None
            )
            carve_semantic_observer = (
                AsyncSemanticObserver(_request_semantic_observation)
                if carve_semantic_shadow_enabled
                else None
            )
            carve_semantic_scheduler = (
                DeadlineAwareSemanticScheduler(carve_semantic_scheduler_config)
                if carve_semantic_shadow_enabled
                else None
            )
            carve_high_level_agent = (
                GuardedHighLevelAgent(
                    carve_high_level_planner,
                    carve_high_level_config,
                )
                if carve_high_level_planner is not None
                else None
            )
            carve_async_high_level_agent = (
                AsyncGuardedHighLevelAgent(carve_high_level_agent)
                if carve_high_level_agent is not None
                else None
            )
            carve_high_level_task_start_policy = str(
                args.carve_high_level_task_start_policy
            )
            carve_canonical_harness = None
            carve_canonical_started = False
            if carve_canonical_harness_enabled:
                assert carve_high_level_planner is not None
                assert carve_high_level_config is not None

                def _canonical_planner_factory(_episode_id: str):
                    return AsyncGuardedHighLevelAgent(
                        GuardedHighLevelAgent(
                            carve_high_level_planner,
                            carve_high_level_config,
                        )
                    )

                carve_canonical_harness = AsyncAgenticHarnessController(
                    carve_controller,
                    planner_factory=_canonical_planner_factory,
                    task_start_policy=TaskStartPolicy(
                        carve_high_level_task_start_policy
                    ),
                    safe_hold_adapter=PausedExecutionSafeHoldAdapter(),
                    failure_memory=carve_failure_memory,
                    knowledge_provider=carve_knowledge_provider,
                    planner_cooldown_steps=int(args.planner_cooldown_steps),
                    safe_hold_timeout_s=float(
                        args.carve_high_level_boundary_timeout_sec
                    ),
                )
                # The canonical harness owns the planner lifecycle. Keeping the
                # legacy per-episode agent live would permit duplicate requests.
                carve_async_high_level_agent = None
            carve_high_level_start_pending = bool(
                carve_high_level_enabled
                and not carve_canonical_harness_enabled
                and carve_high_level_task_start_policy != "event_only"
            )
            carve_pending_semantic_event = None
            carve_semantic_reference_frame = None
            _reset_peak_gpu_memory()
            episode_trace = EpisodeInstrumentation(
                method_tag=ablation_tag,
                task_suite=args.task_suite,
                task_id=task_id,
                episode_idx=episode_idx,
                task_description=task_description,
                control_deadline_ms=float(args.control_deadline_ms),
                perturbation=str(args.perturbation),
                async_prefix_mode=carve_prefix_mode,
                semantic_mode="shadow" if carve_semantic_shadow_enabled else "off",
            )
            if graph_rag_memory is not None:
                episode_trace.memory_retrievals += 1
            expert_state = ExpertState(current_prompt=current_prompt)
            expert_state.reset_for_episode(current_prompt)
            _sync_expert_runtime_state(
                expert_state,
                current_prompt=current_prompt,
                current_subgoal=current_subgoal,
                agentic_action_scale=agentic_action_scale,
                agentic_apply_steps=agentic_apply_steps,
                planner_cooldown_steps=planner_cooldown_steps,
                subgoal_success_window=subgoal_success_window,
            )
            _record_expert_switch(expert_stats, expert_state.current_expert)

            if transition_agent is not None:
                transition_agent.reset()
            if planner_trigger_agent is not None:
                planner_trigger_agent.reset()
            if critic_agent is not None:
                critic_agent.reset()
            failure_taxonomy.reset()
            if carve_monitor is not None:
                carve_monitor.reset()
            if carve_snapshot_writer is not None:
                carve_snapshot_writer.reset(task_id=task_id, episode_id=episode_idx)

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    episode_steps += 1
                    if done:
                        break
                    continue
                episode_trace.mark_control_start()
                loop_step_start_s = time.perf_counter()
                step_had_blocking_reasoning = False
                # VLM planning is allowed to wait only after execution reaches a
                # safe boundary. Keep that wait out of the VLA control-latency
                # metric and expose it as its own stage in the episode trace.
                planner_boundary_wait_s = 0.0
                step_vla_calls_before = episode_trace.vla_calls
                if planner_cooldown_steps > 0:
                    planner_cooldown_steps -= 1
                if ba_recovery_cooldown_steps > 0:
                    ba_recovery_cooldown_steps -= 1
                if ba_transition_lockout_steps > 0:
                    ba_transition_lockout_steps -= 1
                if light_reuse_lockout_steps > 0:
                    light_reuse_lockout_steps -= 1
                if agentic_apply_steps <= 0 and abs(agentic_action_scale - 1.0) > 1e-6:
                    agentic_action_scale = 1.0
                if agentic_apply_steps <= 0 and current_subgoal != "execute":
                    current_subgoal = "execute"
                if subgoal_success_window > 0:
                    subgoal_success_window -= 1
                control_timestep = int(t - args.num_steps_wait)
                if carve_semantic_observer is not None and carve_semantic_observer.ready:
                    semantic_result = carve_semantic_observer.take(wait=False)
                    if semantic_result is not None:
                        episode_trace.record_semantic_result(
                            semantic_result,
                            collected_timestep=control_timestep,
                        )
                if (
                    carve_prefetcher is not None
                    and carve_prefetcher.invalidated
                    and carve_prefetcher.ready
                ):
                    discarded_prefetch = carve_prefetcher.take(wait=False)
                    if discarded_prefetch is not None:
                        episode_trace.add_latency("vla", discarded_prefetch.elapsed_s)
                        episode_trace.async_prefetch_discarded += 1
                        if discarded_prefetch.error is not None:
                            episode_trace.async_prefetch_errors += 1
                if carve_physical_active and not action_plan:
                    if carve_recovery_command is not None:
                        phase_end = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                        carve_recovery_phase_records.append(
                            {
                                **carve_recovery_command.to_dict(),
                                "executed_actions": len(carve_recovery_command.actions),
                                "eef_response": float(
                                    np.linalg.norm(phase_end - carve_recovery_phase_start)
                                ),
                                "environment_terminated": bool(done),
                            }
                        )
                        if carve_recovery_command.request_verification:
                            total_response = float(
                                np.linalg.norm(phase_end - carve_recovery_initial_eef)
                            )
                            verified = bool(
                                np.isfinite(total_response)
                                and total_response
                                >= float(args.carve_physical_minimum_state_response)
                                and not done
                            )
                            recovery_outcome = carve_recovery_executor.resolve_verification(
                                verified,
                                evidence={
                                    "eef_state_response": total_response,
                                    "minimum_state_response": float(
                                        args.carve_physical_minimum_state_response
                                    ),
                                    "task_success": bool(done),
                                    "environment_terminated": bool(done),
                                },
                            )
                            episode_trace.record_physical_recovery(
                                outcome=recovery_outcome.to_dict(),
                                phase_records=carve_recovery_phase_records,
                                timestep=episode_steps,
                            )
                            carve_physical_active = False
                            carve_recovery_command = None
                            if recovery_outcome.request_replan:
                                carve_failure_streak = 0
                                if carve_canonical_harness is not None:
                                    carve_canonical_harness.clear_failures()
                                if carve_skip_post_recovery_semantic:
                                    carve_post_recovery_event = None
                                    carve_skip_post_recovery_semantic = False
                                else:
                                    carve_post_recovery_event = {
                                        "event": recovery_outcome.trigger_event,
                                        "action": "replan_after_physical_recovery",
                                        "reason": recovery_outcome.status.value,
                                        "skill_id": recovery_outcome.skill_id,
                                        "recovery_session_id": recovery_outcome.session_id,
                                    }
                                current_prompt = effective_prompt
                                current_subgoal = "execute"
                                carve_monitor.reset()
                                failure_taxonomy.reset()
                            else:
                                carve_pending_decision = None
                                carve_failure_streak += 1
                                if carve_canonical_harness is not None:
                                    carve_canonical_harness.record_failure()
                                if carve_high_level_enabled:
                                    action_plan.clear()
                                    action_age_plan.clear()
                                    carve_monitor.reset()
                                    failure_taxonomy.reset()
                                    logger.warning(
                                        "[CARVE] Recovery verification failed; "
                                        "recording failure streak %d for guarded escalation",
                                        carve_failure_streak,
                                    )
                                else:
                                    carve_abort_episode = True
                    if carve_physical_active:
                        carve_recovery_command = carve_recovery_executor.next_command()
                        carve_recovery_phase_start = np.asarray(
                            obs["robot0_eef_pos"], dtype=np.float64
                        ).copy()
                        action_plan.extend(carve_recovery_command.actions)
                        action_age_plan.extend([0] * len(carve_recovery_command.actions))
                if carve_abort_episode:
                    logger.warning("[CARVE] Physical recovery verification failed; safe stop")
                    break
                if (
                    str(args.perturbation) == "mid_episode_nudge"
                    and not mid_episode_nudge_applied
                    and control_timestep >= int(args.mid_nudge_step)
                ):
                    obs, nudge_event = _apply_mid_episode_nudge(
                        env,
                        obs,
                        task_description=task_description,
                        rng=episode_rng,
                        object_nudge_xy=float(args.mid_nudge_xy),
                        timestep=control_timestep,
                    )
                    mid_episode_nudge_applied = True
                    if nudge_event is not None:
                        last_mid_episode_nudge_event = dict(nudge_event)
                        action_plan.clear()
                        action_age_plan.clear()
                        light_reuse_buffer.clear()
                        light_reuse_suffix_counted = False
                        light_reuse_lockout_steps = max(
                            light_reuse_lockout_steps,
                            max(0, int(args.light_reuse_post_perturbation_lockout_steps)),
                        )
                        failure_taxonomy.reset()
                        episode_trace.record_perturbation_event(nudge_event)
                        if carve_semantic_observer is not None:
                            carve_pending_semantic_event = {
                                "event": "perturbation",
                                "detail": dict(nudge_event),
                                "reference_image": (
                                    np.asarray(
                                        carve_semantic_reference_frame,
                                        dtype=np.uint8,
                                    ).copy()
                                    if carve_semantic_reference_frame is not None
                                    else None
                                ),
                            }
                        logger.info(
                            "[Perturbation] mid_episode_nudge object=%s dx=%.4f dy=%.4f at control_t=%d",
                            nudge_event.get("object"),
                            float(nudge_event.get("dx", 0.0)),
                            float(nudge_event.get("dy", 0.0)),
                            control_timestep,
                        )
                _sync_expert_runtime_state(
                    expert_state,
                    current_prompt=current_prompt,
                    current_subgoal=current_subgoal,
                    agentic_action_scale=agentic_action_scale,
                    agentic_apply_steps=agentic_apply_steps,
                    planner_cooldown_steps=planner_cooldown_steps,
                    subgoal_success_window=subgoal_success_window,
                )

                image_stage_start_s = time.perf_counter()
                base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Legacy Vision Prompt
                if args.vision_prompt:
                    seg_key = "agentview_segmentation_instance"
                    if seg_key in obs:
                        seg_image = obs[seg_key][::-1, ::-1]
                        base_img = _apply_vision_prompt(
                            base_img, seg_image, env.obj_of_interest,
                            env.instance_to_id, alpha=args.mask_alpha,
                        )

                base_img_p = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(base_img, args.resize_size, args.resize_size)
                )
                wrist_img_p = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                if args.video_render_size > 0:
                    raw_video_frame = np.asarray(
                        env.sim.render(
                            camera_name=str(args.video_camera),
                            height=int(args.video_render_size),
                            width=int(args.video_render_size),
                        ),
                        dtype=np.uint8,
                    )
                    video_frame = np.ascontiguousarray(raw_video_frame[::-1])
                else:
                    video_frame = base_img
                replay_images.append(video_frame)
                episode_trace.add_control_stage(
                    "image_preprocess",
                    time.perf_counter() - image_stage_start_s,
                )

                if (
                    carve_pending_semantic_event is not None
                    and carve_semantic_observer is not None
                    and carve_semantic_scheduler is not None
                ):
                    semantic_slack_ms = max(
                        0.0,
                        float(args.control_deadline_ms) - float(carve_last_control_ms),
                    )
                    semantic_decision = carve_semantic_scheduler.decide(
                        event=str(carve_pending_semantic_event["event"]),
                        timestep=control_timestep,
                        deadline_slack_ms=semantic_slack_ms,
                        request_pending=carve_semantic_observer.pending,
                    )
                    episode_trace.record_semantic_schedule(
                        timestep=control_timestep,
                        event=semantic_decision.event,
                        invoke=semantic_decision.invoke,
                        reason=semantic_decision.reason,
                        deadline_slack_ms=semantic_slack_ms,
                    )
                    if semantic_decision.invoke:
                        semantic_context = SemanticObservationContext(
                            episode_id=f"task{task_id}:trial{episode_idx}",
                            submitted_timestep=control_timestep,
                            event=semantic_decision.event,
                            metadata=dict(carve_pending_semantic_event["detail"]),
                        )
                        carve_semantic_observer.submit(
                            {
                                "endpoint": str(args.carve_semantic_endpoint),
                                "model": str(args.carve_semantic_model),
                                "timeout_sec": float(args.carve_semantic_timeout_sec),
                                "max_tokens": int(args.carve_semantic_max_tokens),
                                "semantic_protocol": str(args.carve_semantic_protocol),
                                "task": task_description,
                                "event": "visual execution anomaly detected",
                                "reference_image": carve_pending_semantic_event.get(
                                    "reference_image"
                                ),
                                "image": np.asarray(base_img_p, dtype=np.uint8).copy(),
                            },
                            semantic_context,
                        )
                        carve_semantic_scheduler.record_submission(control_timestep)
                        carve_pending_semantic_event = None
                    elif semantic_decision.reason != "insufficient deadline slack":
                        carve_pending_semantic_event = None
                carve_semantic_reference_frame = np.asarray(
                    base_img_p,
                    dtype=np.uint8,
                ).copy()

                controller_stage_start_s = time.perf_counter()
                if carve_joint_enabled and not carve_physical_active:
                    proprio = np.concatenate(
                        (
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                            np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
                            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
                        )
                    )
                    deadline_slack_ms = max(
                        0.0,
                        float(args.control_deadline_ms) - float(carve_last_control_ms),
                    )
                    carve_risk = carve_monitor.update(
                        proprio=proprio,
                        commanded_action=carve_last_action,
                        frame=base_img,
                        action_age_steps=int(carve_last_action_age),
                        deadline_slack_ms=deadline_slack_ms,
                    )
                    direct_failure_event, direct_failure_reason = failure_taxonomy.classify()
                    if direct_failure_event in {"slip", "misgrasp"}:
                        direct_components = dict(carve_risk.components)
                        direct_components["direct_failure"] = 1.0
                        direct_evidence = dict(carve_risk.evidence)
                        direct_evidence["direct_failure_reason"] = direct_failure_reason
                        carve_risk = dataclasses.replace(
                            carve_risk,
                            score=max(0.85, float(carve_risk.score)),
                            bucket="high",
                            event=direct_failure_event,
                            components=direct_components,
                            evidence=direct_evidence,
                        )
                    episode_key = f"task{task_id}:trial{episode_idx}"
                    recovery_outcomes = tuple(
                        outcome
                        for outcome in carve_recovery_memory.recent()
                        if str(outcome.episode_id) == f"{task_id}:{episode_idx}"
                    )
                    remaining_recoveries = max(
                        0,
                        int(carve_controller.config.max_recovery_attempts)
                        - int(ba_recovery_count),
                    )
                    available_high_level_skills = (
                        carve_recovery_skills.skill_ids
                        if carve_physical_enabled and remaining_recoveries > 0
                        else ()
                    )

                    def build_agent_context(trigger: str, *, task_start: bool = False):
                        return HighLevelAgentContext(
                            task_instruction=task_description,
                            trigger=trigger,
                            episode_id=episode_key,
                            timestep=control_timestep,
                            frames={
                                "agentview": np.asarray(base_img_p, dtype=np.uint8).copy(),
                                "wrist": np.asarray(wrist_img_p, dtype=np.uint8).copy(),
                            },
                            robot_state=tuple(float(value) for value in proprio),
                            risk=carve_risk.to_dict(),
                            current_subgoal=current_subgoal,
                            failure_history=tuple(
                                str(outcome.trigger_event) for outcome in recovery_outcomes[-5:]
                            ),
                            memory=tuple(
                                (
                                    f"skill={outcome.skill_id};event={outcome.trigger_event};"
                                    f"status={outcome.status.value};attempt={outcome.attempt}"
                                )
                                for outcome in recovery_outcomes[-5:]
                            ),
                            available_skills=(
                                () if task_start else available_high_level_skills
                            ),
                            remaining_retries=max(
                                0,
                                int(args.carve_high_level_max_calls)
                                - int(
                                    carve_canonical_harness.counters.planner_calls
                                    if carve_canonical_harness is not None
                                    else carve_high_level_agent.calls_in_episode
                                ),
                            ),
                            remaining_recoveries=(
                                0 if task_start else remaining_recoveries
                            ),
                            deadline_slack_ms=deadline_slack_ms,
                        )

                    completed_start = None
                    startup_waited = False
                    if carve_high_level_start_pending:
                        assert carve_async_high_level_agent is not None
                        start_ticket = carve_async_high_level_agent.submit(
                            build_agent_context("task_start", task_start=True)
                        )
                        episode_trace.record_high_level_agent_submission(
                            start_ticket,
                            timestep=control_timestep,
                            trigger="task_start",
                        )
                        carve_high_level_start_pending = False
                        if carve_high_level_task_start_policy == "startup_wait":
                            wait_started_s = time.perf_counter()
                            completed_start = carve_async_high_level_agent.take(
                                wait=True,
                                timeout_s=float(
                                    args.carve_high_level_boundary_timeout_sec
                                ),
                            )
                            planner_boundary_wait_s += (
                                time.perf_counter() - wait_started_s
                            )
                            startup_waited = True
                            if completed_start is None:
                                episode_trace.record_high_level_agent_timeout(
                                    start_ticket,
                                    timestep=control_timestep,
                                    trigger="task_start",
                                    boundary_timeout_s=float(
                                        args.carve_high_level_boundary_timeout_sec
                                    ),
                                )
                                carve_abort_episode = True

                    if (
                        completed_start is None
                        and not startup_waited
                        and carve_async_high_level_agent is not None
                    ):
                        completed_start = carve_async_high_level_agent.take(wait=False)
                    if completed_start is not None:
                        start_result = completed_start.result
                        start_decision = start_result.decision
                        apply_start = bool(
                            carve_high_level_task_start_policy == "startup_wait"
                            and completed_start.ticket.context.trigger == "task_start"
                            and start_result.accepted
                            and start_decision.intent
                            in {AgentIntent.CONTINUE, AgentIntent.VLA_ACT}
                        )
                        episode_trace.record_high_level_agent_result(
                            start_result,
                            timestep=control_timestep,
                            trigger=completed_start.ticket.context.trigger,
                            ticket=completed_start.ticket,
                            awaited_at_safe_boundary=startup_waited,
                            decision_applied=apply_start,
                            stale=completed_start.ticket.context.trigger != "task_start",
                        )
                        if (
                            carve_high_level_task_start_policy == "startup_wait"
                            and not apply_start
                        ):
                            carve_abort_episode = True
                        elif (
                            apply_start
                            and start_decision.intent == AgentIntent.VLA_ACT
                        ):
                            current_prompt = compose_grounded_vla_instruction(
                                task_description,
                                subgoal=start_decision.subgoal,
                                planner_instruction=str(start_decision.vla_instruction),
                            )
                            current_subgoal = start_decision.subgoal or "execute"
                            action_plan.clear()
                            action_age_plan.clear()
                            logger.info(
                                "[CARVE Agent] startup-wait subgoal=%s prompt=%s",
                                current_subgoal,
                                current_prompt,
                            )

                    if carve_abort_episode:
                        carve_decision = None
                    elif carve_canonical_harness is not None:
                        planner_trigger = (
                            "post_recovery_verification"
                            if carve_post_recovery_event is not None
                            else carve_risk.event or "repeated_failure"
                        )
                        if not carve_canonical_started:
                            task_start_context = build_agent_context(
                                "task_start",
                                task_start=True,
                            )
                            task_start_transition = carve_canonical_harness.start_episode(
                                episode_key,
                                task_start_context=(
                                    None
                                    if carve_high_level_task_start_policy == "event_only"
                                    else task_start_context
                                ),
                            )
                            carve_canonical_started = True
                            if task_start_transition.ticket is not None:
                                episode_trace.record_high_level_agent_submission(
                                    task_start_transition.ticket,
                                    timestep=control_timestep,
                                    trigger="task_start",
                                )
                            if task_start_transition.ticket is not None and (
                                carve_high_level_task_start_policy == "startup_wait"
                            ):
                                wait_started_s = time.perf_counter()
                                completed_start = carve_canonical_harness.await_planner(
                                    timeout_s=float(
                                        args.carve_high_level_boundary_timeout_sec
                                    )
                                )
                                planner_boundary_wait_s += (
                                    time.perf_counter() - wait_started_s
                                )
                                if completed_start.timed_out:
                                    episode_trace.record_high_level_agent_timeout(
                                        task_start_transition.ticket,
                                        timestep=control_timestep,
                                        trigger="task_start",
                                        boundary_timeout_s=float(
                                            args.carve_high_level_boundary_timeout_sec
                                        ),
                                    )
                                    carve_abort_episode = True
                                elif completed_start.planner is not None:
                                    start_result = completed_start.planner.result
                                    start_decision = start_result.decision
                                    episode_trace.record_high_level_agent_result(
                                        start_result,
                                        timestep=control_timestep,
                                        trigger="task_start",
                                        ticket=completed_start.planner.ticket,
                                        awaited_at_safe_boundary=True,
                                        decision_applied=completed_start.decision_applied,
                                        stale=completed_start.stale,
                                    )
                                    if completed_start.state.value in {"safe_hold", "stop"}:
                                        carve_abort_episode = True
                                    elif start_decision.intent == AgentIntent.VLA_ACT:
                                        current_prompt = compose_grounded_vla_instruction(
                                            task_description,
                                            subgoal=start_decision.subgoal,
                                            planner_instruction=str(
                                                start_decision.vla_instruction
                                            ),
                                        )
                                        current_subgoal = start_decision.subgoal or "execute"

                        if not carve_abort_episode:
                            completed_shadow = carve_canonical_harness.poll_planner()
                            if completed_shadow is not None and completed_shadow.planner is not None:
                                episode_trace.record_high_level_agent_result(
                                    completed_shadow.planner.result,
                                    timestep=control_timestep,
                                    trigger=completed_shadow.planner.ticket.context.trigger,
                                    ticket=completed_shadow.planner.ticket,
                                    awaited_at_safe_boundary=False,
                                    decision_applied=completed_shadow.decision_applied,
                                    stale=completed_shadow.stale,
                                )

                        if not carve_abort_episode:
                            if carve_post_recovery_event is not None:
                                fallback_transition = carve_canonical_harness.route(
                                    carve_risk,
                                    context=build_agent_context(planner_trigger),
                                    deadline_ms=float(args.control_deadline_ms),
                                    deadline_slack_ms=deadline_slack_ms,
                                    cached_actions=len(action_plan),
                                )
                                requested_transition = (
                                    carve_canonical_harness.request_planner(
                                        build_agent_context(planner_trigger),
                                        purpose="post_recovery_verification",
                                        reason=(
                                            "semantic verification after bounded "
                                            "physical recovery"
                                        ),
                                    )
                                )
                                harness_transition = _retain_joint_when_planner_deferred(
                                    fallback_transition,
                                    requested_transition,
                                )
                            else:
                                harness_transition = carve_canonical_harness.route(
                                    carve_risk,
                                    context=build_agent_context(planner_trigger),
                                    deadline_ms=float(args.control_deadline_ms),
                                    deadline_slack_ms=deadline_slack_ms,
                                    cached_actions=len(action_plan),
                                )
                                if (
                                    bool(args.carve_semantic_before_recovery)
                                    and harness_transition.joint is not None
                                    and harness_transition.joint.mode
                                    == ExecutionMode.RECOVERY
                                ):
                                    planner_trigger = "pre_recovery_skill_selection"
                                    requested_transition = (
                                        carve_canonical_harness.request_planner(
                                            build_agent_context(planner_trigger),
                                            purpose="pre_recovery_skill_selection",
                                            reason=(
                                                "select a bounded recovery skill "
                                                "at a safe boundary"
                                            ),
                                        )
                                    )
                                    harness_transition = _retain_joint_when_planner_deferred(
                                        harness_transition,
                                        requested_transition,
                                    )
                            carve_decision = harness_transition.joint
                            if harness_transition.ticket is not None:
                                action_plan.clear()
                                action_age_plan.clear()
                                light_reuse_buffer.clear()
                                light_reuse_suffix_counted = False
                                episode_trace.record_high_level_agent_submission(
                                    harness_transition.ticket,
                                    timestep=control_timestep,
                                    trigger=planner_trigger,
                                )
                                wait_started_s = time.perf_counter()
                                completed_plan = carve_canonical_harness.await_planner(
                                    timeout_s=float(
                                        args.carve_high_level_boundary_timeout_sec
                                    )
                                )
                                planner_boundary_wait_s += (
                                    time.perf_counter() - wait_started_s
                                )
                                if completed_plan.timed_out:
                                    episode_trace.record_high_level_agent_timeout(
                                        harness_transition.ticket,
                                        timestep=control_timestep,
                                        trigger=planner_trigger,
                                        boundary_timeout_s=float(
                                            args.carve_high_level_boundary_timeout_sec
                                        ),
                                    )
                                    carve_abort_episode = True
                                elif completed_plan.planner is not None:
                                    high_level_result = completed_plan.planner.result
                                    high_level_decision = high_level_result.decision
                                    episode_trace.record_high_level_agent_result(
                                        high_level_result,
                                        timestep=control_timestep,
                                        trigger=planner_trigger,
                                        ticket=completed_plan.planner.ticket,
                                        awaited_at_safe_boundary=True,
                                        decision_applied=completed_plan.decision_applied,
                                        stale=completed_plan.stale,
                                    )
                                    carve_decision = completed_plan.joint
                                    if completed_plan.state.value in {"safe_hold", "stop"}:
                                        carve_abort_episode = True
                                    elif high_level_decision.intent == AgentIntent.VLA_ACT:
                                        current_prompt = compose_grounded_vla_instruction(
                                            task_description,
                                            subgoal=high_level_decision.subgoal,
                                            planner_instruction=str(
                                                high_level_decision.vla_instruction
                                            ),
                                        )
                                        current_subgoal = high_level_decision.subgoal or "execute"
                                    elif (
                                        planner_trigger
                                        == "pre_recovery_skill_selection"
                                        and high_level_decision.intent
                                        == AgentIntent.RUN_SKILL
                                    ):
                                        carve_skip_post_recovery_semantic = True
                    elif carve_high_level_enabled:
                        assert carve_async_high_level_agent is not None
                        carve_decision = carve_controller.decide(
                            carve_risk,
                            deadline_ms=float(args.control_deadline_ms),
                            deadline_slack_ms=deadline_slack_ms,
                            cached_actions=len(action_plan),
                            repeated_failures=int(carve_failure_streak),
                            recovery_attempts=int(ba_recovery_count),
                            planner_available=True,
                        )
                        if carve_decision.mode == ExecutionMode.PLANNER:
                            action_plan.clear()
                            action_age_plan.clear()
                            light_reuse_buffer.clear()
                            light_reuse_suffix_counted = False
                            planner_trigger = carve_risk.event or "repeated_failure"

                            # A task-start request may still be running. It was
                            # created for an older state, so collect it for audit
                            # but never apply it at this failure boundary.
                            if carve_async_high_level_agent.pending:
                                pending_ticket = carve_async_high_level_agent.ticket
                                wait_started_s = time.perf_counter()
                                stale_result = carve_async_high_level_agent.take(
                                    wait=True,
                                    timeout_s=float(args.carve_high_level_boundary_timeout_sec),
                                )
                                planner_boundary_wait_s += time.perf_counter() - wait_started_s
                                if stale_result is None:
                                    assert pending_ticket is not None
                                    episode_trace.record_high_level_agent_timeout(
                                        pending_ticket,
                                        timestep=control_timestep,
                                        trigger=planner_trigger,
                                        boundary_timeout_s=float(
                                            args.carve_high_level_boundary_timeout_sec
                                        ),
                                    )
                                    carve_abort_episode = True
                                    carve_decision = dataclasses.replace(
                                        carve_decision,
                                        mode=ExecutionMode.SAFE_STOP,
                                        reason="planner boundary timeout while stale request was pending",
                                    )
                                else:
                                    episode_trace.record_high_level_agent_result(
                                        stale_result.result,
                                        timestep=control_timestep,
                                        trigger=stale_result.ticket.context.trigger,
                                        ticket=stale_result.ticket,
                                        awaited_at_safe_boundary=True,
                                        stale=True,
                                    )

                            if not carve_abort_episode:
                                planner_ticket = carve_async_high_level_agent.submit(
                                    build_agent_context(planner_trigger)
                                )
                                episode_trace.record_high_level_agent_submission(
                                    planner_ticket,
                                    timestep=control_timestep,
                                    trigger=planner_trigger,
                                )
                                wait_started_s = time.perf_counter()
                                completed_plan = carve_async_high_level_agent.take(
                                    wait=True,
                                    timeout_s=float(args.carve_high_level_boundary_timeout_sec),
                                )
                                planner_boundary_wait_s += time.perf_counter() - wait_started_s
                                if completed_plan is None:
                                    episode_trace.record_high_level_agent_timeout(
                                        planner_ticket,
                                        timestep=control_timestep,
                                        trigger=planner_trigger,
                                        boundary_timeout_s=float(
                                            args.carve_high_level_boundary_timeout_sec
                                        ),
                                    )
                                    carve_abort_episode = True
                                    carve_decision = dataclasses.replace(
                                        carve_decision,
                                        mode=ExecutionMode.SAFE_STOP,
                                        reason="planner boundary timeout",
                                    )
                                else:
                                    high_level_result = completed_plan.result
                                    high_level_decision = high_level_result.decision
                                    accepted = bool(
                                        high_level_result.accepted
                                        and high_level_decision.intent != AgentIntent.SAFE_STOP
                                    )
                                    episode_trace.record_high_level_agent_result(
                                        high_level_result,
                                        timestep=control_timestep,
                                        trigger=planner_trigger,
                                        ticket=completed_plan.ticket,
                                        awaited_at_safe_boundary=True,
                                        decision_applied=accepted,
                                    )
                                    if not accepted:
                                        carve_abort_episode = True
                                        carve_decision = dataclasses.replace(
                                            carve_decision,
                                            mode=ExecutionMode.SAFE_STOP,
                                            reason=high_level_result.error
                                            or high_level_decision.rationale,
                                        )
                                    elif high_level_decision.intent == AgentIntent.RUN_SKILL:
                                        carve_decision = dataclasses.replace(
                                            carve_decision,
                                            mode=ExecutionMode.RECOVERY,
                                            reason=high_level_decision.rationale,
                                            recovery_skill_id=high_level_decision.skill_id,
                                        )
                                    else:
                                        if high_level_decision.intent == AgentIntent.VLA_ACT:
                                            current_prompt = compose_grounded_vla_instruction(
                                                task_description,
                                                subgoal=high_level_decision.subgoal,
                                                planner_instruction=str(
                                                    high_level_decision.vla_instruction
                                                ),
                                            )
                                            current_subgoal = (
                                                high_level_decision.subgoal or "execute"
                                            )
                                        carve_decision = dataclasses.replace(
                                            carve_decision,
                                            mode=ExecutionMode.ACCURATE_VLA,
                                            reason=high_level_decision.rationale,
                                        )
                                    logger.info(
                                        "[CARVE Agent] async trigger=%s accepted=%s intent=%s subgoal=%s",
                                        planner_trigger,
                                        high_level_result.accepted,
                                        high_level_decision.intent.value,
                                        high_level_decision.subgoal,
                                    )
                    else:
                        carve_decision = carve_controller.decide(
                            carve_risk,
                            deadline_ms=float(args.control_deadline_ms),
                            deadline_slack_ms=deadline_slack_ms,
                            cached_actions=len(action_plan),
                            repeated_failures=int(carve_failure_streak),
                            recovery_attempts=int(ba_recovery_count),
                            planner_available=bool(args.agentic_planner),
                        )
                    if carve_abort_episode:
                        if planner_boundary_wait_s > 0.0:
                            episode_trace.add_control_stage(
                                "planner_safe_boundary_wait",
                                planner_boundary_wait_s,
                            )
                        episode_trace.add_control_stage(
                            "joint_controller",
                            max(
                                0.0,
                                time.perf_counter()
                                - controller_stage_start_s
                                - planner_boundary_wait_s,
                            ),
                        )
                        break
                    if (
                        carve_prefetcher is not None
                        and carve_prefetcher.pending
                        and (
                            carve_risk.event in {"stall", "slip", "misgrasp", "contact"}
                            or carve_risk.bucket == "high"
                            or carve_decision.mode
                            in {
                                ExecutionMode.ACCURATE_VLA,
                                ExecutionMode.RECOVERY,
                                ExecutionMode.PLANNER,
                                ExecutionMode.SAFE_STOP,
                            }
                        )
                    ):
                        carve_prefetcher.invalidate(
                            f"risk:{carve_risk.event or carve_decision.mode.value}"
                        )
                    if (
                        carve_risk.event == "stall"
                        and carve_decision.mode == ExecutionMode.ACCURATE_VLA
                    ):
                        carve_pending_decision = carve_decision
                        if carve_physical_enabled:
                            action_plan.clear()
                            action_age_plan.clear()
                            light_reuse_buffer.clear()
                            light_reuse_suffix_counted = False
                    # Do not turn a normal cached-action refresh (for example,
                    # low-risk ``stale_action``) into a recovery benchmark.
                    # Saved states must be actionable failure boundaries or
                    # genuinely high-risk observations.
                    if (
                        carve_snapshot_writer is not None
                        and (
                            carve_risk.event in BA_HIGH_RISK_EVENTS
                            or carve_risk.bucket == "high"
                        )
                    ):
                        saved_snapshot = carve_snapshot_writer.write(
                            env,
                            task_id=task_id,
                            episode_id=episode_idx,
                            timestep=episode_steps,
                            trigger=carve_risk.event or carve_risk.bucket,
                            instruction=task_description,
                            observation={
                                "agentview_image": base_img,
                                "wrist_image": wrist_img,
                                "proprio": proprio,
                            },
                            cached_actions=list(action_plan),
                            controller=carve_decision.to_dict(),
                            last_action=carve_last_action,
                        )
                        if saved_snapshot is not None:
                            logger.info(
                                "[CARVE] Saved failure snapshot at task=%d episode=%d step=%d",
                                task_id,
                                episode_idx,
                                episode_steps,
                            )
                    if (
                        carve_decision.mode == ExecutionMode.RECOVERY
                        and ba_recovery_cooldown_steps <= 0
                    ):
                        ba_recovery_count += 1
                        if carve_canonical_harness is not None:
                            carve_canonical_harness.record_recovery_attempt()
                        ba_recovery_cooldown_steps = int(args.ba_recovery_cooldown_steps)
                        action_plan.clear()
                        action_age_plan.clear()
                        light_reuse_buffer.clear()
                        light_reuse_suffix_counted = False
                        episode_trace.mark_recovery_triggered()
                        episode_trace.mark_recovery_started()
                        if carve_physical_enabled:
                            try:
                                recovery_skill_id = (
                                    carve_decision.recovery_skill_id
                                    or carve_controller.config.physical_recovery_skill
                                )
                                recovery_plan = carve_recovery_skills.build(
                                    recovery_skill_id,
                                    carve_recovery_action_spec,
                                    carve_last_action,
                                )
                                carve_recovery_executor.start(
                                    recovery_plan,
                                    RecoveryContext(
                                        episode_id=f"{task_id}:{episode_idx}",
                                        trigger_event=str(carve_risk.event),
                                        attempt=ba_recovery_count - 1,
                                        timestep=episode_steps,
                                        metadata={"decision": carve_decision.to_dict()},
                                    ),
                                )
                                carve_recovery_initial_eef = np.asarray(
                                    obs["robot0_eef_pos"], dtype=np.float64
                                ).copy()
                                carve_recovery_phase_records = []
                                carve_physical_active = True
                                carve_pending_decision = carve_decision
                                carve_recovery_command = carve_recovery_executor.next_command()
                                carve_recovery_phase_start = carve_recovery_initial_eef.copy()
                                action_plan.extend(carve_recovery_command.actions)
                                action_age_plan.extend(
                                    [0] * len(carve_recovery_command.actions)
                                )
                                current_prompt = effective_prompt
                                current_subgoal = "execute"
                                agentic_action_scale = 1.0
                                agentic_apply_steps = 0
                                logger.info(
                                    "[CARVE] Physical recovery started: event=%s skill=%s",
                                    carve_risk.event,
                                    recovery_plan.skill_id,
                                )
                            except Exception as recovery_err:
                                carve_pending_decision = None
                                carve_physical_active = False
                                carve_abort_episode = True
                                logger.error(
                                    "[CARVE] Physical recovery could not start; safe stop: %s",
                                    recovery_err,
                                )
                        else:
                            carve_pending_decision = None
                            current_prompt = recovery_prompt
                            current_subgoal = "recover"
                            agentic_action_scale = 0.85
                            agentic_apply_steps = max(1, int(args.ba_aac_high_commit))
                    elif (
                        carve_decision.mode == ExecutionMode.SAFE_STOP
                        and not action_plan
                    ):
                        action_plan.append(np.zeros(7, dtype=np.float32))
                        action_age_plan.append(0)
                if carve_joint_enabled:
                    if planner_boundary_wait_s > 0.0:
                        episode_trace.add_control_stage(
                            "planner_safe_boundary_wait",
                            planner_boundary_wait_s,
                        )
                    episode_trace.add_control_stage(
                        "joint_controller",
                        max(
                            0.0,
                            time.perf_counter()
                            - controller_stage_start_s
                            - planner_boundary_wait_s,
                        ),
                    )
                if carve_abort_episode:
                    break

                # Record EE position for Transition detection
                ee_pos = obs["robot0_eef_pos"].copy()
                if transition_agent is not None:
                    transition_agent.record_ee_pos(ee_pos)
                if planner_trigger_agent is not None:
                    planner_trigger_agent.record_ee_pos(ee_pos)

                ba_failure_event = ""
                ba_failure_reason = ""
                ba_state_gap = False
                ba_risk_score = 0.0
                ba_risk_bucket = "low"
                ba_progress_state = None
                ba_progress_transition = False
                ba_progress_recovery = False
                ba_context_recovery_allowed = True
                ba_context_transition_allowed = True
                if ba_harness_enabled:
                    if args.ba_context_aware_intervention:
                        if str(args.perturbation) == "clean":
                            ba_context_recovery_allowed = False
                            ba_context_transition_allowed = False
                        elif (
                            str(args.perturbation) == "mid_episode_nudge"
                            and last_mid_episode_nudge_event is None
                        ):
                            ba_context_recovery_allowed = False
                            ba_context_transition_allowed = False
                    ba_failure_event, ba_failure_reason = failure_taxonomy.classify()
                    ba_state_gap = bool(
                        transition_agent is not None
                        and not args.ba_disable_transition
                        and ba_context_transition_allowed
                        and t >= task_control_profile["transition_min_steps"]
                        and not transition_agent.in_transition
                        and transition_agent.detect_state_gap()
                    )
                    ba_risk_score = _ba_risk_score(
                        state_gap=ba_state_gap,
                        failure_event=ba_failure_event,
                        t=t,
                        transition_min_steps=int(task_control_profile["transition_min_steps"]),
                        current_subgoal=current_subgoal,
                        recovery_cooldown_steps=max(ba_recovery_cooldown_steps, ba_transition_lockout_steps),
                    )
                    if args.ba_progress_verifier:
                        ba_progress_state = _estimate_task_progress(env, obs, task_description)
                        episode_trace.record_progress(ba_progress_state, timestep=episode_steps)
                        if ba_progress_state:
                            ba_progress_best_completed_count = max(
                                int(ba_progress_best_completed_count),
                                int(ba_progress_state.get("completed_count", 0)),
                            )
                            ba_progress_best_score = max(
                                float(ba_progress_best_score),
                                float(ba_progress_state.get("score", 0.0)),
                            )
                        ba_progress_transition = (
                            False
                            if args.ba_disable_transition
                            else _progress_should_force_transition(
                                ba_progress_state,
                                failure_event=ba_failure_event,
                                state_gap=ba_state_gap,
                                timestep=t,
                                transition_min_steps=int(task_control_profile["transition_min_steps"]),
                                recovery_cooldown_steps=max(ba_recovery_cooldown_steps, ba_transition_lockout_steps),
                            )
                        )
                        if (
                            ba_progress_transition
                            and args.ba_phase_aware_transition_gate
                            and not _phase_aware_transition_allowed(
                                ba_progress_state,
                                failure_event=ba_failure_event,
                                state_gap=ba_state_gap,
                                timestep=t,
                                transition_min_steps=int(task_control_profile["transition_min_steps"]),
                                recovery_cooldown_steps=max(ba_recovery_cooldown_steps, ba_transition_lockout_steps),
                            )
                        ):
                            ba_progress_transition = False
                        ba_progress_recovery = _progress_should_force_recovery(
                            ba_progress_state,
                            timestep=t,
                            recovery_count=ba_recovery_count,
                            max_recoveries=int(args.ba_max_recoveries),
                            recovery_cooldown_steps=ba_recovery_cooldown_steps,
                            min_steps=int(args.ba_progress_recovery_min_steps),
                            far_min_steps=int(args.ba_progress_far_recovery_min_steps),
                            far_distance=float(args.ba_progress_far_recovery_distance),
                            regression_recovery=bool(args.ba_progress_regression_recovery),
                            regression_distance=float(args.ba_progress_regression_distance),
                            best_completed_count=int(ba_progress_best_completed_count),
                        )
                        if (
                            ba_progress_recovery
                            and str(args.perturbation) == "mid_episode_nudge"
                            and ba_progress_state
                            and ba_progress_state.get("kind") == "plate_relation"
                            and last_mid_episode_nudge_event is None
                        ):
                            ba_progress_recovery = False
                        if ba_progress_recovery and not ba_context_recovery_allowed:
                            ba_progress_recovery = False
                        if (
                            ba_progress_recovery
                            and args.ba_context_aware_intervention
                            and ba_progress_state
                            and ba_progress_state.get("kind") == "single_moka_pot_stove"
                        ):
                            target_distance = ba_progress_state.get("target_distance_xy")
                            if target_distance is None or float(target_distance) < 0.18:
                                ba_progress_recovery = False
                        if (
                            args.ba_perturbation_recovery
                            and not ba_progress_recovery
                            and last_mid_episode_nudge_event is not None
                            and ba_progress_state
                            and ba_progress_state.get("kind") == "single_moka_pot_stove"
                            and int(ba_progress_state.get("completed_count", 0)) == 0
                            and ba_recovery_count < max(0, int(args.ba_max_recoveries))
                            and ba_recovery_cooldown_steps <= 0
                            and t >= int(args.mid_nudge_step)
                        ):
                            target_distance = ba_progress_state.get("target_distance_xy")
                            nudge_object = str(last_mid_episode_nudge_event.get("object") or "")
                            target_object = str(ba_progress_state.get("target_object") or "")
                            if (
                                target_distance is not None
                                and float(target_distance) >= 0.18
                                and (not target_object or not nudge_object or target_object == nudge_object)
                            ):
                                ba_progress_recovery = True
                                ba_progress_state["needs_recenter"] = True
                                ba_progress_state["phase"] = "post_perturbation_recenter"
                        if ba_progress_transition:
                            ba_risk_score = max(ba_risk_score, 0.80)
                        if ba_progress_recovery:
                            if (
                                ba_progress_state
                                and ba_progress_state.get("kind") == "single_moka_pot_stove"
                                and not ba_progress_state.get("needs_recenter")
                            ):
                                if int(ba_progress_best_completed_count) > int(ba_progress_state.get("completed_count", 0)):
                                    ba_progress_state["phase"] = "progress_regression_recovery"
                                else:
                                    ba_progress_state["phase"] = "far_stall_recovery"
                            elif (
                                ba_progress_state
                                and ba_progress_state.get("kind") == "plate_relation"
                                and not ba_progress_state.get("needs_recenter")
                            ):
                                ba_progress_state["phase"] = "far_stall_recovery"
                            ba_risk_score = max(ba_risk_score, 0.85)
                            ba_failure_event = ba_failure_event or "placement_drift"
                            target_object = ba_progress_state.get("target_object") if ba_progress_state else None
                            target_distance = ba_progress_state.get("target_distance_xy") if ba_progress_state else None
                            ba_failure_reason = (
                                f"placement verifier requests recenter target={target_object} "
                                f"dist_xy={target_distance}; {ba_failure_reason}"
                            ).strip()
                    ba_risk_bucket = _ba_risk_bucket(
                        ba_risk_score,
                        ba_failure_event,
                        ba_progress_state,
                    )

                if (
                    ba_harness_enabled
                    and task_control_profile.get("ba_recovery_enabled", True)
                    and not args.ba_disable_recovery
                    and not done
                    and not ba_progress_transition
                    and ba_context_recovery_allowed
                    and (
                        ba_progress_recovery
                        or _ba_should_recover(
                            failure_event=ba_failure_event,
                            risk_score=ba_risk_score,
                            t=t,
                            recovery_count=ba_recovery_count,
                            max_recoveries=int(args.ba_max_recoveries),
                            recovery_cooldown_steps=ba_recovery_cooldown_steps,
                            risk_threshold=float(
                                task_control_profile.get(
                                    "ba_recovery_risk_threshold",
                                    args.ba_recovery_risk_threshold,
                                )
                            ),
                        )
                    )
                ):
                    ba_recovery_count += 1
                    ba_recovery_cooldown_steps = int(args.ba_recovery_cooldown_steps)
                    ba_transition_lockout_steps = int(args.ba_post_recovery_transition_lockout_steps)
                    episode_trace.mark_recovery_triggered()
                    recovery_kind = REGRASP_EXPERT if ba_failure_event in {"misgrasp", "slip"} else RECOVERY_EXPERT
                    decision = _route_expert(
                        failure_event=ba_failure_event,
                        reason=f"B4 risk={ba_risk_score:.2f}; {ba_failure_reason}",
                        expert_state=expert_state,
                        request_plan=False,
                    )
                    decision.expert = recovery_kind
                    decision.subgoal = "recover" if recovery_kind == RECOVERY_EXPERT else "regrasp"
                    _apply_router_decision(
                        expert_state,
                        decision,
                        expert_stats,
                        default_prompt=effective_prompt,
                    )
                    current_subgoal = expert_state.current_subgoal
                    current_prompt = expert_state.current_prompt
                    action_plan.clear()
                    action_age_plan.clear()
                    light_reuse_buffer.clear()
                    light_reuse_suffix_counted = False
                    light_reuse_lockout_steps = max(
                        light_reuse_lockout_steps,
                        max(0, int(args.light_reuse_post_recovery_lockout_steps)),
                    )
                    recovery_executor_prompt = (
                        regrasp_prompt if recovery_kind == REGRASP_EXPERT else recovery_prompt
                    )
                    if args.ba_progress_verifier and recovery_kind == RECOVERY_EXPERT:
                        recovery_executor_prompt = _build_progress_recovery_prompt(
                            recovery_executor_prompt,
                            ba_progress_state,
                            goal_aware=bool(args.ba_progress_goal_aware_recovery_prompt),
                        )
                    logger.info(
                        "[B4] Rule recovery triggered: event=%s risk=%.2f reason=%s",
                        ba_failure_event,
                        ba_risk_score,
                        ba_failure_reason,
                    )
                    episode_trace.mark_recovery_started()
                    obs, recovery_steps, done = (
                        _execute_regrasp_expert(
                            env=env,
                            client=client,
                            obs=obs,
                            base_img_p=base_img_p,
                            wrist_img_p=wrist_img_p,
                            prompt=recovery_executor_prompt,
                            instrumentation=episode_trace,
                        )
                        if recovery_kind == REGRASP_EXPERT
                        else _execute_recovery_expert(
                            env=env,
                            client=client,
                            obs=obs,
                            base_img_p=base_img_p,
                            wrist_img_p=wrist_img_p,
                            prompt=recovery_executor_prompt,
                            instrumentation=episode_trace,
                        )
                    )
                    episode_steps += recovery_steps
                    current_prompt = effective_prompt
                    expert_state.current_prompt = current_prompt
                    action_plan.clear()
                    action_age_plan.clear()
                    light_reuse_buffer.clear()
                    light_reuse_suffix_counted = False
                    light_reuse_lockout_steps = max(
                        light_reuse_lockout_steps,
                        max(0, int(args.light_reuse_post_recovery_lockout_steps)),
                    )
                    failure_taxonomy.reset()
                    episode_trace.record_control_step(
                        time.perf_counter() - loop_step_start_s,
                        fast_path=True,
                        vla_call=episode_trace.vla_calls > step_vla_calls_before,
                    )
                    if done:
                        break
                    continue

                # A4: Critic check at intervals
                if (critic_agent is not None and t >= critic_agent.min_control_steps
                        and t % critic_agent.check_interval == 0 and not done):
                    critic_stats["total_checks"] += 1
                    task_critic_checks += 1
                    critic_start_s = time.perf_counter()
                    judgment = critic_agent.judge_subtask(base_img, task_description)
                    episode_trace.add_latency(
                        "verifier",
                        time.perf_counter() - critic_start_s,
                        blocking=True,
                    )
                    step_had_blocking_reasoning = True
                    logger.info("[Critic] Step %d: judgment=%s", t, judgment)

                    if critic_agent.should_retry(judgment):
                        decision = _route_expert(
                            failure_event="critic",
                            reason=str(judgment),
                            expert_state=expert_state,
                            critic_stuck=bool(judgment.get("stuck", False)),
                            request_plan=bool(args.agentic_planner and planner_cooldown_steps <= 0),
                        )
                        _apply_router_decision(
                            expert_state,
                            decision,
                            expert_stats,
                            default_prompt=effective_prompt,
                        )
                        current_prompt = expert_state.current_prompt
                        current_subgoal = expert_state.current_subgoal
                        critic_stats["retries_triggered"] += 1
                        critic_agent.retry_count += 1
                        episode_trace.mark_recovery_triggered()
                        retry_prompt = critic_agent.get_retry_prompt(task_description, judgment)
                        logger.info("[Critic] Recovery + retry: %s", retry_prompt)
                        if args.agentic_planner and planner_cooldown_steps <= 0:
                            (
                                _requested,
                                planner_cooldown_steps,
                                current_subgoal,
                                current_prompt,
                                agentic_action_scale,
                                agentic_apply_steps,
                            ) = _request_planner(
                                client=client,
                                obs=obs,
                                base_img_p=base_img_p,
                                wrist_img_p=wrist_img_p,
                                prompt=current_prompt,
                                episode_id=f"{task_id}:{episode_idx}",
                                timestep=episode_steps,
                                event="critic",
                                reason=str(judgment),
                                context=f"{agentic_context}; subgoal={current_subgoal}",
                                planner_stats=planner_stats,
                                planner_event_counts=planner_event_counts,
                                planner_action_counts=planner_action_counts,
                                planner_expert_counts=planner_expert_counts,
                                expert_state=expert_state,
                                expert_stats=expert_stats,
                                default_prompt=effective_prompt,
                                current_cooldown_steps=planner_cooldown_steps,
                                cooldown_steps=int(args.planner_cooldown_steps),
                                instrumentation=episode_trace,
                            )

                        # Execute recovery action
                        recovery_executor_prompt = (
                            regrasp_prompt if expert_state.current_expert == REGRASP_EXPERT else recovery_prompt
                        )
                        episode_trace.mark_recovery_started()
                        obs, recovery_steps, done = (
                            _execute_regrasp_expert(
                                env=env,
                                client=client,
                                obs=obs,
                                base_img_p=base_img_p,
                                wrist_img_p=wrist_img_p,
                                prompt=recovery_executor_prompt,
                                instrumentation=episode_trace,
                            )
                            if expert_state.current_expert == REGRASP_EXPERT
                            else _execute_recovery_expert(
                                env=env,
                                client=client,
                                obs=obs,
                                base_img_p=base_img_p,
                                wrist_img_p=wrist_img_p,
                                prompt=recovery_executor_prompt,
                                instrumentation=episode_trace,
                            )
                        )
                        episode_steps += recovery_steps

                        current_prompt = retry_prompt
                        expert_state.current_prompt = current_prompt
                        action_plan.clear()
                        action_age_plan.clear()
                        light_reuse_buffer.clear()
                        light_reuse_suffix_counted = False

                # A2/B4: Transition Agent - detect State Gap
                transition_ready = False
                if (
                    transition_agent is not None
                    and not args.ba_disable_transition
                    and ba_context_transition_allowed
                    and (
                        not ba_harness_enabled
                        or (
                            (args.ba_enable_transition or task_control_profile.get("ba_transition_enabled", False))
                            and task_control_profile.get("ba_transition_enabled", True)
                        )
                    )
                    and t >= task_control_profile["transition_min_steps"]
                    and not transition_agent.in_transition
                    and transition_agent.transition_count < transition_agent.max_transitions
                ):
                    if ba_harness_enabled:
                        transition_ready = ba_progress_transition or _ba_should_transition(
                            state_gap=ba_state_gap,
                            failure_event=ba_failure_event,
                            allowed_failure_events=set(
                                task_control_profile.get("ba_transition_events", {"collision"})
                            ),
                            risk_score=ba_risk_score,
                            t=t,
                            transition_min_steps=int(task_control_profile["transition_min_steps"]),
                            recovery_cooldown_steps=max(ba_recovery_cooldown_steps, ba_transition_lockout_steps),
                            risk_threshold=float(
                                args.ba_transition_risk_threshold
                            ),
                        )
                        if (
                            transition_ready
                            and args.ba_phase_aware_transition_gate
                            and not _phase_aware_transition_allowed(
                                ba_progress_state,
                                failure_event=ba_failure_event,
                                state_gap=ba_state_gap,
                                timestep=t,
                                transition_min_steps=int(task_control_profile["transition_min_steps"]),
                                recovery_cooldown_steps=max(ba_recovery_cooldown_steps, ba_transition_lockout_steps),
                            )
                        ):
                            transition_ready = False
                    else:
                        transition_ready = transition_agent.detect_state_gap()
                if transition_ready:
                    decision = _route_expert(
                        failure_event="state_gap",
                        reason=(
                            f"B4 risk={ba_risk_score:.2f}; threshold="
                            f"{float(args.ba_transition_risk_threshold):.2f}; "
                            f"event={ba_failure_event}; progress={ba_progress_state}; {ba_failure_reason}"
                            if ba_harness_enabled else "transition agent detected state gap"
                        ),
                        expert_state=expert_state,
                        request_plan=bool(args.agentic_planner and planner_cooldown_steps <= 0),
                    )
                    episode_trace.transition_triggers += 1
                    _apply_router_decision(
                        expert_state,
                        decision,
                        expert_stats,
                        default_prompt=effective_prompt,
                    )
                    current_subgoal = expert_state.current_subgoal
                    transition_agent.trigger_transition()
                    transition_stats["total_transitions"] += 1
                    action_plan.clear()
                    action_age_plan.clear()
                    light_reuse_buffer.clear()
                    light_reuse_suffix_counted = False

                    # Execute transition via VLA prompt
                    transition_executor_prompt = (
                        _build_progress_transition_prompt(transition_prompt, ba_progress_state)
                        if args.ba_progress_verifier
                        else transition_prompt
                    )
                    obs, transition_steps, done = _execute_transition_expert(
                        env=env,
                        client=client,
                        obs=obs,
                        base_img_p=base_img_p,
                        wrist_img_p=wrist_img_p,
                        prompt=transition_executor_prompt,
                        instrumentation=episode_trace,
                    )
                    episode_steps += transition_steps

                    transition_agent.finish_transition()
                    verifier_decision = _verify_expert_progress(
                        expert_state,
                        failure_event="",
                        state_gap_active=False,
                    )
                    if verifier_decision.status == "resolved":
                        expert_stats["success_counts"][expert_state.current_expert] += 1
                        _apply_router_decision(
                            expert_state,
                            RouterDecision(
                                expert=DEFAULT_VLA_EXPERT,
                                trigger="verifier",
                                reason=verifier_decision.reason,
                                subgoal="execute",
                            ),
                            expert_stats,
                            default_prompt=effective_prompt,
                        )
                    current_prompt = effective_prompt
                    expert_state.current_prompt = current_prompt
                    current_subgoal = expert_state.current_subgoal
                    action_plan.clear()
                    action_age_plan.clear()
                    light_reuse_buffer.clear()
                    light_reuse_suffix_counted = False
                    if args.agentic_planner and planner_cooldown_steps <= 0:
                        (
                            _requested,
                            planner_cooldown_steps,
                            current_subgoal,
                            current_prompt,
                            agentic_action_scale,
                            agentic_apply_steps,
                        ) = _request_planner(
                            client=client,
                            obs=obs,
                            base_img_p=base_img_p,
                            wrist_img_p=wrist_img_p,
                            prompt=current_prompt,
                            episode_id=f"{task_id}:{episode_idx}",
                            timestep=episode_steps,
                            event="state_gap",
                            reason="transition agent detected state gap; provide next-step strategy",
                            context=f"{agentic_context}; subgoal={current_subgoal}",
                            planner_stats=planner_stats,
                            planner_event_counts=planner_event_counts,
                            planner_action_counts=planner_action_counts,
                            planner_expert_counts=planner_expert_counts,
                            expert_state=expert_state,
                            expert_stats=expert_stats,
                            default_prompt=effective_prompt,
                            current_cooldown_steps=planner_cooldown_steps,
                            cooldown_steps=int(args.planner_cooldown_steps),
                            instrumentation=episode_trace,
                        )

                if (
                    carve_prefetcher is not None
                    and action_plan
                    and len(action_plan) <= int(args.carve_prefetch_lead_actions)
                    and not carve_prefetcher.pending
                    and carve_prefetch_schedule is not None
                    and carve_prefetch_schedule.can_submit
                    and not carve_physical_active
                    and carve_pending_decision is None
                    and carve_post_recovery_event is None
                    and carve_risk.event is None
                    and carve_decision.mode in {ExecutionMode.REUSE, ExecutionMode.FAST_VLA}
                    and current_subgoal == "execute"
                    and planner_trigger_agent is None
                    and transition_agent is None
                    and critic_agent is None
                    and not ba_harness_enabled
                    and not light_reuse_enabled_for_task
                ):
                    prefetch_lead = len(action_plan)
                    prefetch_timestep = int(episode_steps + prefetch_lead)
                    prefetch_payload = _build_policy_payload(
                        obs=obs,
                        base_img_p=base_img_p,
                        wrist_img_p=wrist_img_p,
                        prompt=current_prompt,
                        episode_id=f"{task_id}:{episode_idx}",
                        timestep=prefetch_timestep,
                    )
                    if args.fixed_policy_noise:
                        prefetch_seed = np.random.SeedSequence(
                            [
                                int(args.seed),
                                int(task_id),
                                int(episode_idx),
                                prefetch_timestep,
                            ]
                        )
                        prefetch_rng = np.random.default_rng(prefetch_seed)
                        prefetch_payload["runtime_noise"] = prefetch_rng.standard_normal(
                            (10, 32), dtype=np.float32
                        )
                    prefetch_payload["runtime_controls"] = {
                        "inference_steps": int(args.carve_fast_inference_steps),
                        "max_actions": int(args.carve_commit_steps),
                        "deadline_ms": float(args.control_deadline_ms),
                    }
                    prefetch_payload["runtime_trace_context"] = {
                        "mode": "async_prefetch",
                        "submitted_timestep": int(episode_steps),
                        "target_timestep": prefetch_timestep,
                        "lead_actions": prefetch_lead,
                        "risk": carve_risk.to_dict(),
                    }
                    carve_prefetcher.submit(
                        prefetch_payload,
                        PrefetchContext(
                            episode_id=f"{task_id}:{episode_idx}",
                            submitted_timestep=int(episode_steps),
                            lead_actions=prefetch_lead,
                            metadata={
                                "target_timestep": prefetch_timestep,
                                "executed_prefix": [
                                    np.asarray(action, dtype=np.float32).tolist()
                                    for action in action_plan
                                ],
                            },
                        ),
                    )
                    episode_trace.async_prefetch_submitted += 1

                # Lightweight VLA Runtime: reuse cached suffix actions from the previous full VLA chunk.
                light_reuse_available = (
                    light_reuse_enabled_for_task
                    and light_reuse_lockout_steps <= 0
                )
                if not action_plan and light_reuse_available and light_reuse_buffer:
                    allowed_reuse_buckets = {
                        bucket.strip()
                        for bucket in str(args.light_reuse_risk_buckets).split(",")
                        if bucket.strip()
                    }
                    can_reuse_cached_actions = (
                        ba_risk_bucket in allowed_reuse_buckets
                        and current_subgoal == "execute"
                        and not ba_progress_recovery
                        and not ba_progress_transition
                    )
                    if can_reuse_cached_actions:
                        reuse_steps = min(int(args.light_reuse_max_actions), len(light_reuse_buffer))
                        if reuse_steps > 0:
                            reuse_ages = []
                            for _ in range(reuse_steps):
                                cached_action, cached_age = light_reuse_buffer.popleft()
                                action_plan.append(cached_action)
                                action_age_plan.append(int(cached_age))
                                reuse_ages.append(int(cached_age))
                            if not light_reuse_suffix_counted:
                                episode_trace.record_skipped_vla_call(
                                    action_chunk_age=max(reuse_ages) if reuse_ages else None
                                )
                                light_reuse_suffix_counted = True
                            episode_trace.record_commit(reuse_steps, risk_bucket=ba_risk_bucket)

                # Normal VLA inference
                if not action_plan:
                    inference = None
                    prefetch_action_offset = 0
                    if carve_prefetcher is not None and carve_prefetcher.pending:
                        prefetch_was_ready = carve_prefetcher.ready
                        prefetch_wait_start_s = time.perf_counter()
                        prefetch_result = carve_prefetcher.take(wait=True)
                        prefetch_wait_s = time.perf_counter() - prefetch_wait_start_s
                        episode_trace.add_control_stage(
                            "async_prefetch_wait",
                            prefetch_wait_s,
                        )
                        if not prefetch_was_ready:
                            episode_trace.async_prefetch_waited += 1
                        if prefetch_result is not None:
                            episode_trace.add_latency("vla", prefetch_result.elapsed_s)
                            if (
                                prefetch_result.valid
                                and isinstance(prefetch_result.response, Mapping)
                            ):
                                candidate = dict(prefetch_result.response)
                                candidate_actions = candidate.get("actions")
                                enforce_prefix_rejection = False
                                if carve_prefix_verifier is not None:
                                    prefix_actions = int(
                                        prefetch_result.context.lead_actions
                                    )
                                    consistency = carve_prefix_verifier.evaluate(
                                        prefetch_result.context.metadata.get(
                                            "executed_prefix", []
                                        ),
                                        np.asarray(candidate_actions)[
                                            :prefix_actions
                                        ]
                                        if candidate_actions is not None
                                        else [],
                                    )
                                    enforce_prefix_rejection = bool(
                                        carve_prefix_mode == "enforce"
                                        and not consistency.accepted
                                    )
                                    episode_trace.record_async_prefix_consistency(
                                        consistency.to_dict(),
                                        ticket_id=prefetch_result.context.ticket_id,
                                        submitted_timestep=(
                                            prefetch_result.context.submitted_timestep
                                        ),
                                        enforced_rejection=enforce_prefix_rejection,
                                    )
                                if enforce_prefix_rejection:
                                    episode_trace.async_prefetch_discarded += 1
                                else:
                                    inference = candidate
                                    prefetch_action_offset = int(
                                        prefetch_result.context.lead_actions
                                    )
                                    carve_prefetch_schedule.record_prefetch_consumed()
                                    episode_trace.async_prefetch_ready += 1
                            else:
                                episode_trace.async_prefetch_discarded += 1
                                if prefetch_result.error is not None:
                                    episode_trace.async_prefetch_errors += 1
                    carve_inference_decision = (
                        carve_pending_decision
                        if carve_joint_enabled and carve_pending_decision is not None
                        else carve_decision
                    )
                    agentic_req = (
                        dict(carve_post_recovery_event)
                        if carve_post_recovery_event is not None
                        else None
                    )
                    if planner_trigger_agent is not None and t >= task_control_profile["transition_min_steps"]:
                        failure_event, failure_reason = failure_taxonomy.classify()
                        if (
                            agentic_req is None
                            and failure_event
                            and planner_cooldown_steps <= 0
                        ):
                            decision = _route_expert(
                                failure_event=failure_event,
                                reason=failure_reason,
                                expert_state=expert_state,
                                request_plan=True,
                            )
                            _apply_router_decision(
                                expert_state,
                                decision,
                                expert_stats,
                                default_prompt=effective_prompt,
                            )
                            current_subgoal = expert_state.current_subgoal
                            current_prompt = expert_state.current_prompt
                            planner_trigger_agent.trigger_transition()
                            agentic_req = {
                                "request_plan": True,
                                "event": failure_event,
                                "reason": failure_reason,
                                "context": f"{agentic_context}; subgoal={current_subgoal}",
                            }
                            planner_stats["total_requests"] += 1
                            planner_event_counts[failure_event] += 1
                            planner_cooldown_steps = max(planner_cooldown_steps, int(args.planner_cooldown_steps))
                            step_had_blocking_reasoning = True
                    if inference is None:
                        payload = _build_policy_payload(
                            obs=obs,
                            base_img_p=base_img_p,
                            wrist_img_p=wrist_img_p,
                            prompt=current_prompt,
                            episode_id=f"{task_id}:{episode_idx}",
                            timestep=episode_steps,
                            agentic_req=agentic_req,
                        )
                        blocking_planner_request = bool(
                            agentic_req is not None and agentic_req.get("request_plan")
                        )
                        try:
                            if args.fixed_policy_noise:
                                noise_seed = np.random.SeedSequence(
                                    [
                                        int(args.seed),
                                        int(task_id),
                                        int(episode_idx),
                                        int(episode_steps),
                                    ]
                                )
                                noise_rng = np.random.default_rng(noise_seed)
                                payload["runtime_noise"] = noise_rng.standard_normal(
                                    (10, 32), dtype=np.float32
                                )
                            if carve_joint_enabled and carve_inference_decision is not None:
                                payload["runtime_controls"] = dataclasses.asdict(
                                    carve_inference_decision.controls
                                )
                                payload["runtime_trace_context"] = (
                                    carve_inference_decision.to_dict()
                                )
                            elif int(args.carve_fixed_inference_steps) > 0:
                                payload["runtime_controls"] = {
                                    "inference_steps": int(args.carve_fixed_inference_steps),
                                    "max_actions": int(args.replan_steps),
                                    "deadline_ms": float(args.control_deadline_ms),
                                }
                                payload["runtime_trace_context"] = {
                                    "mode": "fixed_compute",
                                    "inference_steps": int(
                                        args.carve_fixed_inference_steps
                                    ),
                                }
                            episode_trace.add_control_stage(
                                "vla_step_pre_infer",
                                time.perf_counter() - loop_step_start_s,
                            )
                            infer_start_s = time.perf_counter()
                            inference = _infer_with_retry(client, payload, max_retries=5)
                            infer_elapsed_s = time.perf_counter() - infer_start_s
                            episode_trace.add_latency("vla", infer_elapsed_s)
                            if carve_prefetch_schedule is not None:
                                carve_prefetch_schedule.record_synchronous_chunk()
                            if blocking_planner_request:
                                episode_trace.add_latency("planner", infer_elapsed_s, blocking=True)
                        except Exception as infer_err:
                            logger.warning("Inference failed: %s — restarting server", infer_err)
                            try:
                                _restart_policy_server(OPENPI_ROOT, args.port)
                                client = _reconnect_client(
                                    args.host,
                                    args.port,
                                    carve_runtime=bool(args.carve_runtime),
                                    carve_trace_jsonl=(
                                        str(carve_trace_path) if args.carve_runtime else ""
                                    ),
                                )
                                infer_start_s = time.perf_counter()
                                inference = _infer_with_retry(client, payload, max_retries=5)
                                infer_elapsed_s = time.perf_counter() - infer_start_s
                                episode_trace.add_latency("vla", infer_elapsed_s)
                                if carve_prefetch_schedule is not None:
                                    carve_prefetch_schedule.record_synchronous_chunk()
                                if blocking_planner_request:
                                    episode_trace.add_latency(
                                        "planner", infer_elapsed_s, blocking=True
                                    )
                            except Exception as reconnect_err:
                                logger.error(
                                    "Server restart failed: %s — aborting episode",
                                    reconnect_err,
                                )
                                episode_trace.timeout_fallback_count += 1
                                break
                    if "actions" not in inference:
                        raise RuntimeError(f"Malformed response: {inference}")
                    if isinstance(inference.get("agentic"), dict):
                        agentic_resp = inference["agentic"]
                        (
                            current_subgoal,
                            current_prompt,
                            agentic_action_scale,
                            agentic_apply_steps,
                        ) = _handle_agentic_response(
                            agentic_resp,
                            expert_state=expert_state,
                            planner_stats=planner_stats,
                            planner_action_counts=planner_action_counts,
                            planner_expert_counts=planner_expert_counts,
                            expert_stats=expert_stats,
                            default_prompt=effective_prompt,
                        )
                        if agentic_apply_steps > 0:
                            planner_cooldown_steps = max(planner_cooldown_steps, agentic_apply_steps)
                    action_chunk = inference["actions"]
                    commit_steps = _ba_aac_commit_steps(
                        enabled=bool(args.ba_aac_lite or args.method_tag == "B4-AAC-Lite"),
                        default_steps=int(args.replan_steps),
                        risk_bucket=ba_risk_bucket,
                        current_subgoal=current_subgoal,
                        profile=task_control_profile,
                        progress=ba_progress_state,
                        min_commit=int(args.ba_aac_min_commit),
                        medium_commit=int(args.ba_aac_medium_commit),
                        high_commit=int(args.ba_aac_high_commit),
                        max_commit=int(args.ba_aac_max_commit),
                    )
                    if carve_joint_enabled and carve_inference_decision is not None:
                        commit_steps = min(
                            int(commit_steps),
                            int(carve_inference_decision.controls.max_actions or commit_steps),
                        )
                    if (
                        light_reuse_available
                        and int(args.light_reuse_commit_steps) > 0
                        and current_subgoal == "execute"
                    ):
                        commit_steps = min(commit_steps, int(args.light_reuse_commit_steps))
                    available_actions = len(action_chunk) - prefetch_action_offset
                    if available_actions <= 0:
                        raise RuntimeError(
                            "Prefetched policy chunk does not extend beyond its lead actions."
                        )
                    if prefetch_action_offset == 0 and len(action_chunk) < commit_steps:
                        raise RuntimeError(
                            f"Policy predicted {len(action_chunk)} steps, need {commit_steps}."
                        )
                    commit_steps = min(int(commit_steps), int(available_actions))
                    action_plan.extend(
                        action_chunk[
                            prefetch_action_offset:prefetch_action_offset + commit_steps
                        ]
                    )
                    if prefetch_action_offset > 0:
                        episode_trace.async_prefetch_actions += int(commit_steps)
                    if carve_post_recovery_event is not None:
                        carve_post_recovery_event = None
                    if carve_joint_enabled:
                        action_age_plan.extend(
                            range(
                                int(prefetch_action_offset),
                                int(prefetch_action_offset + commit_steps),
                            )
                        )
                    else:
                        action_age_plan.extend([0] * int(commit_steps))
                    if light_reuse_available:
                        light_reuse_buffer.clear()
                        light_reuse_suffix_counted = False
                        suffix_limit = max(0, int(args.light_reuse_max_buffer_actions))
                        for suffix_index, extra_action in enumerate(
                            action_chunk[
                                prefetch_action_offset + commit_steps:
                                prefetch_action_offset + commit_steps + suffix_limit
                            ],
                            start=int(prefetch_action_offset + commit_steps),
                        ):
                            cached_age = suffix_index if carve_joint_enabled else 1
                            light_reuse_buffer.append((extra_action, cached_age))
                    episode_trace.record_commit(
                        int(commit_steps),
                        risk_bucket=ba_risk_bucket,
                    )
                    if carve_joint_enabled:
                        carve_pending_decision = None

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                action_chunk_age = action_age_plan.popleft() if action_age_plan else 0
                episode_trace.record_action_chunk_age(action_chunk_age)
                if agentic_apply_steps > 0 and abs(agentic_action_scale - 1.0) > 1e-6:
                    a = np.asarray(action, dtype=np.float32).copy()
                    a[:6] *= agentic_action_scale
                    action = a
                    agentic_apply_steps -= 1
                failure_taxonomy.update(
                    ee_pos=obs["robot0_eef_pos"],
                    action=np.asarray(action, dtype=np.float32),
                    gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                )
                if current_subgoal != "execute":
                    fe, _ = failure_taxonomy.classify()
                    if not fe:
                        subgoal_success_window += 1
                    else:
                        subgoal_success_window = 0
                    expert_state.subgoal_success_window = subgoal_success_window
                    if subgoal_success_window >= 6:
                        verifier_decision = _verify_expert_progress(
                            expert_state,
                            failure_event=fe,
                            state_gap_active=False,
                        )
                        if verifier_decision.status == "resolved":
                            expert_stats["success_counts"][expert_state.current_expert] += 1
                            _apply_router_decision(
                                expert_state,
                                RouterDecision(
                                    expert=DEFAULT_VLA_EXPERT,
                                    trigger="verifier",
                                    reason=verifier_decision.reason,
                                    subgoal="execute",
                                ),
                                expert_stats,
                                default_prompt=effective_prompt,
                            )
                            current_prompt = expert_state.current_prompt
                            current_subgoal = expert_state.current_subgoal
                        elif verifier_decision.status == "escalate":
                            expert_stats["escalation_counts"][expert_state.current_expert] += 1
                        elif verifier_decision.status == "replan":
                            expert_stats["replan_counts"][expert_state.current_expert] += 1
                _sync_expert_runtime_state(
                    expert_state,
                    current_prompt=current_prompt,
                    current_subgoal=current_subgoal,
                    agentic_action_scale=agentic_action_scale,
                    agentic_apply_steps=agentic_apply_steps,
                    planner_cooldown_steps=planner_cooldown_steps,
                    subgoal_success_window=subgoal_success_window,
                )
                episode_trace.mark_first_action()
                env_step_start_s = time.perf_counter()
                obs, _, done, _ = env.step(action.tolist())
                episode_trace.add_control_stage(
                    "env_step",
                    time.perf_counter() - env_step_start_s,
                )
                carve_last_action = np.asarray(action, dtype=np.float32).copy()
                carve_last_action_age = int(action_chunk_age)
                episode_steps += 1
                expert_state.steps_in_expert += 1
                carve_last_control_ms = max(
                    0.0,
                    (time.perf_counter() - loop_step_start_s - planner_boundary_wait_s)
                    * 1000.0,
                )
                episode_trace.record_reaction_timing(
                    observation_to_action_s=max(
                        0.0,
                        env_step_start_s - loop_step_start_s,
                    ),
                    task_cycle_s=max(
                        0.0,
                        time.perf_counter() - loop_step_start_s,
                    ),
                )
                episode_trace.record_control_step(
                    carve_last_control_ms / 1000.0,
                    fast_path=not step_had_blocking_reasoning,
                    vla_call=episode_trace.vla_calls > step_vla_calls_before,
                )
                if done:
                    break

            if carve_prefetcher is not None:
                if carve_prefetcher.pending:
                    carve_prefetcher.invalidate("episode_end")
                    final_prefetch = carve_prefetcher.take(wait=True)
                    if final_prefetch is not None:
                        episode_trace.add_latency("vla", final_prefetch.elapsed_s)
                        episode_trace.async_prefetch_discarded += 1
                        if final_prefetch.error is not None:
                            episode_trace.async_prefetch_errors += 1
                carve_prefetcher.close()

            if carve_semantic_observer is not None:
                if carve_semantic_observer.pending:
                    final_semantic_result = carve_semantic_observer.take(wait=True)
                    if final_semantic_result is not None:
                        episode_trace.record_semantic_result(
                            final_semantic_result,
                            collected_timestep=max(0, episode_steps - args.num_steps_wait),
                        )
                carve_semantic_observer.close()

            if carve_async_high_level_agent is not None:
                # The planner is advisory at task start. Do not delay episode
                # accounting merely to wait for an obsolete result; retain one
                # completed result when available and otherwise cancel queued work.
                final_plan = carve_async_high_level_agent.take(wait=False)
                if final_plan is not None:
                    episode_trace.record_high_level_agent_result(
                        final_plan.result,
                        timestep=max(0, episode_steps - args.num_steps_wait),
                        trigger=final_plan.ticket.context.trigger,
                        ticket=final_plan.ticket,
                        stale=True,
                    )
                carve_async_high_level_agent.close(wait=False)

            if carve_canonical_harness is not None:
                # The canonical controller owns the planner worker and any
                # safe-hold state. Record a ready late result as stale evidence
                # before closing the episode-scoped generation.
                final_transition = carve_canonical_harness.poll_planner()
                if (
                    final_transition is not None
                    and final_transition.planner is not None
                ):
                    episode_trace.record_high_level_agent_result(
                        final_transition.planner.result,
                        timestep=max(0, episode_steps - args.num_steps_wait),
                        trigger=final_transition.planner.ticket.context.trigger,
                        ticket=final_transition.planner.ticket,
                        stale=True,
                    )
                carve_canonical_harness.close_episode(reason="episode_end")

            total_episodes += 1
            episode_lengths.append(episode_steps)
            task_episode_lengths.append(episode_steps)
            episode_had_transition = transition_agent is not None and transition_agent.transition_count > 0
            episode_had_retry = critic_agent is not None and critic_agent.retry_count > 0
            if episode_had_transition:
                task_transition_total += transition_agent.transition_count
                task_transition_episodes += 1
            if episode_had_retry:
                task_retry_total += critic_agent.retry_count
                task_retry_episodes += 1
            if done:
                task_successes += 1
                total_successes += 1
                success_episode_lengths.append(episode_steps)
                if graph_rag_memory is not None:
                    graph_rag_memory.record_success(task_description, task_priors)
                if episode_had_transition:
                    transition_stats["transitions_leading_to_success"] += 1
                    task_transition_successes += 1
                if episode_had_retry:
                    critic_stats["retries_leading_to_success"] += 1
                    task_retry_successes += 1
                    episode_trace.recoveries_successful += 1
                if ba_harness_enabled and ba_recovery_count > 0:
                    episode_trace.recoveries_successful += 1
                if (
                    carve_physical_enabled
                    and episode_trace.physical_recoveries_triggered > 0
                ):
                    episode_trace.recoveries_successful += 1

            trace_payload = episode_trace.finish(
                success=done,
                episode_steps=episode_steps,
                peak_gpu_mem_gb=_get_peak_gpu_mem_gb(),
            )
            episode_traces.append(trace_payload)
            _append_episode_trace(trace_path, trace_payload)

            suffix = "success" if done else "failure"
            safe_task = task_description.replace(" ", "_").replace("/", "_")
            video_path = pathlib.Path(args.video_dir) / f"task{task_id}_trial{episode_idx}_{suffix}_{safe_task}.mp4"
            if replay_images:
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
            logger.info("Episode done=%s | running success=%.3f", done, total_successes / total_episodes)
            task_expert_stats_running = _serialize_expert_stats(
                _merge_expert_stats(
                    resumed_task_expert_stats,
                    _diff_expert_stats(expert_stats, task_expert_stats_before),
                )
            )
            _persist_progress(
                current_task={
                    "task_id": task_id,
                    "task_description": task_description,
                    "next_episode_idx": episode_idx + 1,
                    "task_successes": task_successes,
                    "task_episode_lengths": task_episode_lengths,
                    "task_transition_total": task_transition_total,
                    "task_transition_episodes": task_transition_episodes,
                    "task_transition_successes": task_transition_successes,
                    "task_retry_total": task_retry_total,
                    "task_retry_episodes": task_retry_episodes,
                    "task_retry_successes": task_retry_successes,
                    "task_critic_checks": task_critic_checks,
                    "task_expert_stats": task_expert_stats_running,
                },
                status="running",
            )

        task_expert_stats = _serialize_expert_stats(
            _merge_expert_stats(
                resumed_task_expert_stats,
                _diff_expert_stats(expert_stats, task_expert_stats_before),
            )
        )
        task_metrics[task_key] = {
            "success_rate": task_successes / args.trials,
            "episodes": float(args.trials),
            "avg_episode_length": float(np.mean(task_episode_lengths)) if task_episode_lengths else 0.0,
            "transition_total": int(task_transition_total) if transition_agent is not None else 0,
            "transition_episode_count": int(task_transition_episodes) if transition_agent is not None else 0,
            "transition_success_count": int(task_transition_successes) if transition_agent is not None else 0,
            "critic_checks": int(task_critic_checks) if args.critic else 0,
            "retry_total": int(task_retry_total) if args.critic else 0,
            "retry_episode_count": int(task_retry_episodes) if args.critic else 0,
            "retry_success_count": int(task_retry_successes) if args.critic else 0,
            "expert_stats": task_expert_stats,
        }
        logger.info("Task %s success rate: %.3f", task_id, task_metrics[task_key]["success_rate"])
        completed_tasks.add(task_key)
        completed_tasks_file.write_text(json.dumps(sorted(completed_tasks)), encoding="utf-8")
        current_task_resume = None
        _persist_progress(current_task=None, status="running")

    overall = total_successes / total_episodes if total_episodes else 0.0
    logger.info("Overall success rate: %.3f (%d/%d)", overall, total_successes, total_episodes)

    results = _build_results_payload(
        args=args,
        ablation_tag=ablation_tag,
        total_successes=total_successes,
        total_episodes=total_episodes,
        episode_lengths=episode_lengths,
        success_episode_lengths=success_episode_lengths,
        task_metrics=task_metrics,
        transition_stats=transition_stats,
        critic_stats=critic_stats,
        planner_stats={
            **planner_stats,
            "event_counts": dict(planner_event_counts),
            "action_counts": dict(planner_action_counts),
            "expert_counts": dict(planner_expert_counts),
        },
        expert_stats=_serialize_expert_stats(expert_stats),
        episode_trace_path=str(trace_path),
        trace_aggregate=_aggregate_episode_traces(episode_traces),
        status="completed",
        current_task=None,
    )
    if args.vision_prompt:
        results["mask_alpha"] = args.mask_alpha

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    partial_summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    resume_state_file.write_text(json.dumps({
        "task_suite": args.task_suite,
        "ablation_tag": ablation_tag,
        "evaluated_task_ids": getattr(args, "evaluated_task_ids", None),
        "skipped_task_ids": getattr(args, "skipped_task_ids", None),
        "completed_tasks": sorted(completed_tasks),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "episode_lengths": episode_lengths,
        "success_episode_lengths": success_episode_lengths,
        "task_metrics": task_metrics,
        "transition_stats": transition_stats if transition_agent is not None else None,
        "critic_stats": critic_stats if args.critic else None,
        "expert_stats": _serialize_expert_stats(expert_stats),
        "episode_trace_jsonl": str(trace_path),
        "trace_aggregate": _aggregate_episode_traces(episode_traces),
        "current_task": None,
        "status": "completed",
    }, indent=2), encoding="utf-8")
    logger.info("Saved results to %s", results_path)

    # Unload Critic model if loaded
    if critic_agent is not None:
        critic_agent.unload_model()

    return results


if __name__ == "__main__":
    args = _parse_args()
    try:
        evaluate_real_libero(args)
    except Exception as exc:
        logger.error("Real LIBERO evaluation aborted: %s", exc)
        raise SystemExit(1)

from __future__ import annotations

import numpy as np
import pytest

from agentic_vla.runtime import (
    GroundedSubgoalScheduleConfig,
    GroundedSubgoalScheduleMode,
    GroundedSubgoalScheduler,
)


def frame(value: int = 0) -> np.ndarray:
    return np.full((16, 16, 3), value, dtype=np.uint8)


def state(gripper: float = 0.0) -> np.ndarray:
    return np.asarray([0.0] * 7 + [gripper], dtype=np.float32)


def test_selective_scheduler_reuses_until_budget_expires() -> None:
    scheduler = GroundedSubgoalScheduler(
        GroundedSubgoalScheduleConfig(
            max_reuse_chunks=2,
            visual_change_threshold=0.5,
            gripper_change_threshold=0.5,
        )
    )

    assert scheduler.decide(frame=frame(), state=state()).reason == "task_start"
    scheduler.record_planner_result(frame=frame(), state=state())
    scheduler.record_chunk_executed()
    first_reuse = scheduler.decide(frame=frame(), state=state())
    assert not first_reuse.invoke
    assert first_reuse.reason == "grounded_subgoal_reuse_admitted"

    scheduler.record_chunk_executed()
    expired = scheduler.decide(frame=frame(), state=state())
    assert expired.invoke
    assert expired.reason == "reuse_budget_exhausted"


def test_selective_scheduler_invalidates_on_gripper_visual_and_event() -> None:
    scheduler = GroundedSubgoalScheduler(
        GroundedSubgoalScheduleConfig(
            max_reuse_chunks=4,
            visual_change_threshold=0.1,
            gripper_change_threshold=0.1,
        )
    )
    scheduler.record_planner_result(frame=frame(), state=state())
    scheduler.record_chunk_executed()

    gripper = scheduler.decide(frame=frame(), state=state(0.2))
    assert gripper.invoke
    assert gripper.reason == "gripper_state_changed"

    visual = scheduler.decide(frame=frame(64), state=state())
    assert visual.invoke
    assert visual.reason == "visual_context_changed"

    event = scheduler.decide(
        frame=frame(), state=state(), execution_event="stall"
    )
    assert event.invoke
    assert event.reason == "execution_event:stall"


def test_every_chunk_mode_preserves_baseline_behavior() -> None:
    scheduler = GroundedSubgoalScheduler(
        GroundedSubgoalScheduleConfig(mode=GroundedSubgoalScheduleMode.EVERY_CHUNK)
    )
    scheduler.record_planner_result(frame=frame(), state=state())
    scheduler.record_chunk_executed()
    decision = scheduler.decide(frame=frame(), state=state())
    assert decision.invoke
    assert decision.reason == "every_chunk_baseline"


def test_precision_sensitive_grounding_disables_reuse() -> None:
    scheduler = GroundedSubgoalScheduler()
    scheduler.record_planner_result(
        frame=frame(),
        state=state(),
        reuse_permitted=False,
    )
    scheduler.record_chunk_executed()
    decision = scheduler.decide(frame=frame(), state=state())
    assert decision.invoke
    assert decision.reason == "precision_sensitive_grounding"


def test_scheduler_rejects_invalid_configuration_and_shape_changes() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        GroundedSubgoalScheduleConfig(min_reuse_chunks=3, max_reuse_chunks=2)

    scheduler = GroundedSubgoalScheduler()
    scheduler.record_planner_result(frame=frame(), state=state())
    scheduler.record_chunk_executed()
    with pytest.raises(ValueError, match="frame shape changed"):
        scheduler.decide(frame=np.zeros((8, 8, 3)), state=state())

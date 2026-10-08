"""Contracts for evidence-gated admission of Harness intervention triggers.

The Optimize Runtime half refuses to deploy a profile until it proves fidelity.
This is the symmetric gate for the Harness half. Two things must hold or the gate
is worthless:

* the offline replay of the Monitor's event layer must agree with the live
  Monitor exactly, otherwise thresholds would be calibrated against a fiction;
* the admission criteria must actually reject the two failure modes this project
  has measured -- a branch that never fires, and a trigger that fires on
  episodes that were succeeding anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic_vla.runtime.monitor import ExecutionRiskMonitor, MonitorConfig
from agentic_vla.toolchain import (
    EpisodeTriggerEvidence,
    MonitorEventThresholds,
    MonitorSample,
    TriggerAdmissionRequirements,
    recompute_monitor_events,
    validate_trigger_admission,
)


def test_offline_replay_matches_the_live_monitor_exactly() -> None:
    """Calibrating on recorded means must reproduce the deployed decision."""

    config = MonitorConfig(window_size=4, warmup_steps=2, no_progress_steps=3)
    monitor = ExecutionRiskMonitor(config)
    rng = np.random.default_rng(11)

    samples: list[MonitorSample] = []
    live_events: list[str | None] = []
    state = np.zeros(14, dtype=np.float32)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for step in range(40):
        # Alternate between moving and being completely still, so both the idle
        # and the non-idle branches are exercised.
        moving = (step // 5) % 2 == 0
        if moving:
            state = state + rng.normal(0.0, 0.05, size=14).astype(np.float32)
            frame = (frame.astype(np.int16) + 7).clip(0, 255).astype(np.uint8)
        action = state + (0.5 if moving else 0.0)
        assessment = monitor.update(
            proprio=state,
            commanded_action=action,
            frame=frame,
            action_age_steps=step % 4,
        )
        live_events.append(assessment.event)
        evidence = assessment.evidence
        samples.append(
            MonitorSample(
                timestep=step,
                window_ready=bool(evidence["window_ready"]),
                mean_command=float(evidence["mean_command"]),
                mean_state_response=float(evidence["mean_state_response"]),
                mean_visual_response=float(evidence["mean_visual_response"]),
                action_age_steps=int(evidence["action_age_steps"]),
                has_frame=True,
            )
        )

    thresholds = MonitorEventThresholds(
        command_threshold=config.command_threshold,
        state_response_threshold=config.state_response_threshold,
        visual_response_threshold=config.visual_response_threshold,
        idle_command_threshold=config.idle_command_threshold,
        idle_state_response_threshold=config.idle_state_response_threshold,
        idle_visual_response_threshold=config.idle_visual_response_threshold,
        stale_action_steps=config.stale_action_steps,
        no_progress_steps=config.no_progress_steps,
    )
    assert list(recompute_monitor_events(samples, thresholds)) == live_events


def test_replay_reacts_to_a_threshold_change() -> None:
    """A sweep has to be able to move the decision, or it measures nothing."""

    samples = [
        MonitorSample(
            timestep=step,
            window_ready=True,
            mean_command=0.05,
            mean_state_response=0.02,
            mean_visual_response=0.001,
        )
        for step in range(5)
    ]
    # Default: state response 0.02 is far above 0.002, so stall cannot fire.
    assert set(recompute_monitor_events(samples, MonitorEventThresholds())) == {None}
    # Recalibrated to the observed scale, the same evidence fires stall.
    recalibrated = MonitorEventThresholds(state_response_threshold=0.03)
    assert set(recompute_monitor_events(samples, recalibrated)) == {"stall"}


def test_stall_precedes_no_progress_in_the_replay() -> None:
    """Event precedence must match the Monitor, or counts are misattributed."""

    thresholds = MonitorEventThresholds(
        command_threshold=0.0,  # stall condition on command is trivially true
        state_response_threshold=1.0,
        idle_command_threshold=1.0,
        idle_state_response_threshold=1.0,
        idle_visual_response_threshold=1.0,
        visual_response_threshold=1.0,
        no_progress_steps=1,
    )
    samples = [
        MonitorSample(
            timestep=0,
            window_ready=True,
            mean_command=0.5,
            mean_state_response=0.0,
            mean_visual_response=0.0,
        )
    ]
    assert recompute_monitor_events(samples, thresholds) == ("stall",)


def _episodes(spec):
    return [
        EpisodeTriggerEvidence(
            episode_index=index, steps=steps, success=success, event_steps=events
        )
        for index, (steps, success, events) in enumerate(spec)
    ]


def test_a_trigger_that_never_fires_is_rejected_as_unreachable() -> None:
    """This is the measured `stall` case: 0 firings in 16385 steps."""

    decision = validate_trigger_admission(
        _episodes([(800, False, ()), (800, True, ()), (800, False, ()),
                   (800, True, ()), (800, False, ())])
    )

    assert decision.accepted is False
    assert decision.status == "rejected"
    assert any("unreachable branch is a defect" in v for v in decision.violations)
    assert decision.metrics["firing_episodes"] == 0
    with pytest.raises(ValueError, match="admission failed"):
        decision.require_accepted()


def test_a_trigger_that_fires_on_succeeding_episodes_is_rejected() -> None:
    """The measured C2 case: interventions landed in episodes that succeeded."""

    # Base failure rate 0.6, but the trigger selects only successful episodes.
    decision = validate_trigger_admission(
        _episodes(
            [
                (800, True, (100,)),
                (800, True, (120,)),
                (800, False, ()),
                (800, False, ()),
                (800, False, ()),
            ]
        )
    )

    assert decision.accepted is False
    assert decision.metrics["failure_precision"] == 0.0
    assert decision.metrics["failure_lift"] == 0.0
    assert any("adds risk without adding information" in v for v in decision.violations)


def test_a_specific_and_timely_trigger_is_admitted() -> None:
    decision = validate_trigger_admission(
        _episodes(
            [
                (800, False, (100, 140)),
                (800, False, (200,)),
                (800, True, ()),
                (800, True, ()),
                (800, False, (300,)),
            ]
        )
    )

    assert decision.accepted is True, decision.violations
    assert decision.status == "admitted"
    assert decision.metrics["failure_precision"] == 1.0
    assert decision.metrics["failure_lift"] > 1.0
    assert decision.metrics["actionable_firing_episodes"] == 3
    decision.require_accepted()


def test_a_late_only_trigger_is_rejected_for_lack_of_headroom() -> None:
    """An event 20 steps before the end cannot lead to a bounded recovery."""

    decision = validate_trigger_admission(
        _episodes(
            [
                (800, False, (780,)),
                (800, False, (790,)),
                (800, True, ()),
                (800, True, ()),
                (800, False, (795,)),
            ]
        )
    )

    assert decision.accepted is False
    assert decision.metrics["actionable_firing_episodes"] == 0
    assert any("headroom" in v for v in decision.violations)


def test_a_saturating_trigger_is_rejected_on_budget() -> None:
    """Consecutive events collapse by cooldown; distinct bursts do not."""

    bursts = tuple(step for start in (0, 200, 400, 600) for step in range(start, start + 3))
    decision = validate_trigger_admission(
        _episodes([(800, False, bursts)] * 5),
        TriggerAdmissionRequirements(maximum_occasions_per_episode=3.0),
    )

    assert decision.accepted is False
    assert decision.metrics["mean_occasions_per_firing_episode"] == 4.0
    assert any("firing occasions per episode exceeds" in v for v in decision.violations)


def test_cooldown_collapses_a_run_of_consecutive_events() -> None:
    evidence = EpisodeTriggerEvidence(
        episode_index=0, steps=800, success=False, event_steps=tuple(range(0, 50))
    )

    # 50 consecutive event steps are one opportunity to intervene, not 50.
    assert evidence.occasions(cooldown_steps=128) == 1
    assert evidence.occasions(cooldown_steps=10) == 5


def test_thin_evidence_is_rejected_rather_than_silently_accepted() -> None:
    decision = validate_trigger_admission(_episodes([(800, False, (100,))]))

    assert decision.accepted is False
    assert any("episodes of evidence" in v for v in decision.violations)


def test_no_evidence_is_reported_distinctly() -> None:
    decision = validate_trigger_admission([])

    assert decision.accepted is False
    assert decision.status == "no_evidence"


def test_event_steps_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="ordered"):
        EpisodeTriggerEvidence(
            episode_index=0, steps=800, success=False, event_steps=(200, 100)
        )


# ---------------------------------------------------------------------------
# Whole-policy feasibility: can a recovery fire at all inside its own budget?
#
# The four gates are configured independently and can each look reasonable while
# jointly leaving no room to act. The deployed pairing is exactly that case:
# a 320-step unchanged span with a 160-step check interval needs the third
# counted check, and the call budget is three.
# ---------------------------------------------------------------------------


from agentic_vla.toolchain import (  # noqa: E402
    InterventionPolicy,
    escalation_candidate_steps,
    simulate_episode_policy,
    validate_policy_feasibility,
)


def test_only_a_sustained_event_can_force_a_semantic_check() -> None:
    """First onset routes to a cheap replan; the semantic path needs a streak."""

    # Isolated events have no streak; consecutive runs do.
    assert escalation_candidate_steps((10, 50, 90)) == ()
    assert escalation_candidate_steps((10, 11, 12)) == (11, 12)
    assert escalation_candidate_steps((10, 11, 50, 51, 52)) == (11, 51, 52)


@pytest.mark.parametrize("event_name", ["stall", "stale_action"])
def test_unsupported_bridge_events_do_not_force_checks(event_name):
    result = simulate_episode_policy(
        episode_index=0, steps=600, success=False, event_steps=tuple(range(79, 83)),
        event_name=event_name, policy=InterventionPolicy(),
    )
    assert result.checks == (160, 320, 480)


def _feasibility(steps: int, policy: InterventionPolicy, events=()):
    return simulate_episode_policy(
        episode_index=0,
        steps=steps,
        success=False,
        event_steps=events,
        policy=policy,
    )


def test_reserve_does_not_add_calls_or_invent_recovery_after_expiry():
    policy = InterventionPolicy(event_reserve_calls=1)
    no_event = _feasibility(800, policy)
    late_event = _feasibility(800, policy, events=tuple(range(705, 709)))
    assert no_event.checks == (160, 320)
    assert late_event.checks == (160, 320, 706)
    assert late_event.earliest_recovery_step is None
    assert not no_event.budget_exhausted
    assert late_event.budget_exhausted


@pytest.mark.parametrize("reserve", [-1, 4, True, 1.5])
def test_policy_rejects_invalid_event_reserve(reserve):
    with pytest.raises(ValueError, match="event reserve"):
        InterventionPolicy(event_reserve_calls=reserve)


def test_deployed_pairing_can_only_act_on_the_last_budgeted_check() -> None:
    """The coupling found in calibration, pinned as a regression."""

    policy = InterventionPolicy(
        checkpoint_steps=160,
        max_calls_per_episode=3,
        min_stall_interval_steps=160,
        min_unchanged_steps=320,
        stale_checks_required=2,
    )
    result = _feasibility(800, policy)

    # Periodic checks land at 160/320/480 and all three count.
    assert result.checks == (160, 320, 480)
    assert result.counted_checks == (160, 320, 480)
    # The span only reaches 320 at the third check, which is also the last the
    # budget allows: the window is exactly one check wide.
    assert result.earliest_recovery_step == 480
    assert result.feasible is True


def test_one_fewer_call_makes_the_same_thresholds_unable_to_act() -> None:
    """Budget and span are coupled: tightening either alone can zero the window."""

    policy = InterventionPolicy(
        checkpoint_steps=160,
        max_calls_per_episode=2,
        min_stall_interval_steps=160,
        min_unchanged_steps=320,
    )
    result = _feasibility(800, policy)

    assert result.checks == (160, 320)
    assert result.earliest_recovery_step is None
    assert result.feasible is False
    assert result.budget_exhausted is True


def test_a_short_episode_cannot_reach_the_span() -> None:
    policy = InterventionPolicy(checkpoint_steps=160, min_unchanged_steps=320)

    assert _feasibility(400, policy).feasible is False
    assert _feasibility(500, policy).feasible is True


def test_a_forced_check_bypasses_the_interval_but_not_the_budget() -> None:
    """A sustained event may pull a check earlier without extra allowance."""

    policy = InterventionPolicy(
        checkpoint_steps=160,
        max_calls_per_episode=3,
        min_stall_interval_steps=160,
        min_unchanged_steps=320,
        cooldown_steps=128,
    )
    # A sustained event at steps 40-42 forces a check at 41, before the first
    # periodic check would have run.
    result = _feasibility(800, policy, events=(40, 41, 42))

    assert result.checks[0] == 41
    assert len(result.checks) == 3, result.checks
    # The forced check counts as the first observation, so the span is measured
    # from it and the schedule shifts earlier.
    assert result.counted_checks[0] == 41


def test_forced_checks_respect_the_escalation_cooldown() -> None:
    """Without the cooldown a sustained event would consume the whole budget."""

    policy = InterventionPolicy(
        checkpoint_steps=1000,  # disable the periodic schedule for this case
        max_calls_per_episode=5,
        min_stall_interval_steps=0,
        min_unchanged_steps=0,
        cooldown_steps=128,
    )
    result = _feasibility(800, policy, events=tuple(range(100, 400)))

    # 300 consecutive event steps collapse to one check per cooldown window.
    assert result.checks == (101, 229, 357)


def test_a_policy_that_cannot_act_is_rejected() -> None:
    policy = InterventionPolicy(
        checkpoint_steps=160, max_calls_per_episode=2, min_unchanged_steps=320
    )
    episodes = [
        simulate_episode_policy(
            episode_index=index,
            steps=800,
            success=index % 2 == 0,
            event_steps=(),
            policy=policy,
        )
        for index in range(6)
    ]

    decision = validate_policy_feasibility(episodes)

    assert decision.accepted is False
    assert decision.status == "infeasible"
    assert decision.metrics["feasible_episodes"] == 0
    assert any("spent on waiting" in v for v in decision.violations)
    with pytest.raises(ValueError, match="feasibility failed"):
        decision.require_accepted()


def test_a_workable_policy_is_accepted() -> None:
    policy = InterventionPolicy(
        checkpoint_steps=160, max_calls_per_episode=4, min_unchanged_steps=320
    )
    episodes = [
        simulate_episode_policy(
            episode_index=index,
            steps=800,
            success=index % 3 == 0,
            event_steps=(),
            policy=policy,
        )
        for index in range(6)
    ]

    decision = validate_policy_feasibility(episodes)

    assert decision.accepted is True
    assert decision.status == "feasible"
    assert decision.metrics["feasible_rate"] == 1.0
    assert decision.metrics["earliest_recovery_step_median"] == 480


def test_a_policy_reachable_only_in_successful_episodes_is_rejected() -> None:
    """Acting where it cannot matter is not feasibility."""

    long_policy = InterventionPolicy(
        checkpoint_steps=160, max_calls_per_episode=4, min_unchanged_steps=320
    )
    episodes = [
        # Failures end too early to reach the span; successes run long.
        simulate_episode_policy(
            episode_index=index,
            steps=400 if index < 3 else 800,
            success=index >= 3,
            event_steps=(),
            policy=long_policy,
        )
        for index in range(6)
    ]

    decision = validate_policy_feasibility(episodes)

    assert decision.accepted is False
    assert decision.metrics["feasible_failed_episodes"] == 0
    assert any("cannot act where it would matter" in v for v in decision.violations)


def test_feasibility_reports_no_evidence_distinctly() -> None:
    decision = validate_policy_feasibility([])

    assert decision.accepted is False
    assert decision.status == "no_evidence"


def test_episode_feasibility_serializes_for_the_report() -> None:
    result = _feasibility(800, InterventionPolicy())
    payload = result.to_dict()

    assert payload["feasible"] is True
    assert payload["checks"] == [160, 320, 480]
    assert isinstance(payload["counted_checks"], list)


def test_a_non_counting_check_does_not_reopen_the_periodic_gate_in_simulation() -> None:
    """The simulator must reproduce the fixed scheduling, not the defect.

    With a minimum stall interval larger than the checkpoint interval, keying the
    periodic gate to the last counted observation produced checks on consecutive
    steps and burned the budget. This is the shape of the fix.
    """

    policy = InterventionPolicy(
        checkpoint_steps=80,
        max_calls_per_episode=5,
        min_stall_interval_steps=160,
        min_unchanged_steps=320,
    )
    result = simulate_episode_policy(
        episode_index=0, steps=800, success=False, event_steps=(), policy=policy
    )

    assert result.checks == (80, 160, 240, 320, 400)
    assert all(b - a >= 80 for a, b in zip(result.checks, result.checks[1:]))
    # Only every other check clears the 160-step counting interval.
    assert result.counted_checks == (80, 240, 400)
    assert result.earliest_recovery_step == 400


def test_halving_the_check_interval_needs_the_counting_interval_to_follow() -> None:
    """Checking more often only helps if counting keeps pace."""

    faster = InterventionPolicy(
        checkpoint_steps=80,
        max_calls_per_episode=5,
        min_stall_interval_steps=80,
        min_unchanged_steps=320,
    )
    result = simulate_episode_policy(
        episode_index=0, steps=800, success=False, event_steps=(), policy=faster
    )

    assert result.counted_checks == (80, 160, 240, 320, 400)
    assert result.earliest_recovery_step == 400

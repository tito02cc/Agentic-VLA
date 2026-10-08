"""Evidence-gated admission for Agentic Harness intervention triggers.

The Optimize Runtime half of this system already refuses to deploy a candidate
profile until it has proven fidelity (``optimization/admission.py``). The Harness
half had no equivalent gate: intervention thresholds were hand-set and shipped,
so a trigger could be unreachable, or fire mostly on episodes that were
succeeding anyway, and nothing would object.

This module supplies the symmetric gate. A trigger must produce four kinds of
evidence before it is allowed to change what the robot does:

* **reachability** -- it fires at all. A branch that can never fire is a defect,
  not a conservative default.
* **specificity** -- it fires on episodes that actually fail, at a rate better
  than the base failure rate. A trigger that fires uniformly carries no
  information, and intervening on it can only add risk.
* **timeliness** -- its first firing leaves enough steps for a bounded recovery
  to run. The escalation chain is serial, so an event near the end of an episode
  cannot lead to any action.
* **budget feasibility** -- the number of firing occasions fits the per-episode
  intervention budget, so the budget is not exhausted before the informative
  occasion arrives.

Everything here is pure and offline. ``recompute_monitor_events`` replays the
Monitor's event layer from window means that were already recorded, so candidate
thresholds can be evaluated on past runs without touching a robot.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Kept in sync with agentic_vla.runtime.monitor.MonitorConfig defaults.
STALL_EVENT = "stall"
NO_PROGRESS_EVENT = "no_progress"
STALE_ACTION_EVENT = "stale_action"


@dataclasses.dataclass(frozen=True)
class MonitorEventThresholds:
    """One candidate threshold set for the Monitor's event layer."""

    command_threshold: float = 0.03
    state_response_threshold: float = 0.002
    visual_response_threshold: float = 0.004
    idle_command_threshold: float = 0.01
    idle_state_response_threshold: float = 0.015
    idle_visual_response_threshold: float = 0.004
    stale_action_steps: int = 8
    no_progress_steps: int | None = 32

    def __post_init__(self) -> None:
        positives = (
            self.command_threshold,
            self.state_response_threshold,
            self.visual_response_threshold,
            self.idle_command_threshold,
            self.idle_state_response_threshold,
            self.idle_visual_response_threshold,
        )
        if any(value < 0 for value in positives):
            raise ValueError("monitor thresholds must be non-negative")
        if self.stale_action_steps <= 0:
            raise ValueError("stale_action_steps must be positive")
        if self.no_progress_steps is not None and self.no_progress_steps <= 0:
            raise ValueError("no_progress_steps must be positive or disabled")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MonitorSample:
    """One recorded Monitor update, threshold-independent by construction.

    The window means do not depend on any threshold, so replaying the event
    layer over them is exact rather than approximate. ``window_size`` is *not*
    a free parameter here: changing it would change the means themselves, so
    only thresholds may be swept from recorded evidence.
    """

    timestep: int
    window_ready: bool
    mean_command: float
    mean_state_response: float
    mean_visual_response: float
    action_age_steps: int = 0
    has_frame: bool = True


def recompute_monitor_events(
    samples: Sequence[MonitorSample], thresholds: MonitorEventThresholds
) -> tuple[str | None, ...]:
    """Replay ExecutionRiskMonitor's event decision under candidate thresholds.

    Mirrors ``ExecutionRiskMonitor.update`` exactly, including the precedence
    ``stall > no_progress > stale_action`` and the idle-streak accumulation.
    """
    events: list[str | None] = []
    idle_streak = 0
    for sample in samples:
        warm = bool(sample.window_ready)
        visual_ok_stall = (not sample.has_frame) or (
            sample.mean_visual_response < thresholds.visual_response_threshold
        )
        stalled = bool(
            warm
            and sample.mean_command >= thresholds.command_threshold
            and sample.mean_state_response < thresholds.state_response_threshold
            and visual_ok_stall
        )
        visual_ok_idle = (not sample.has_frame) or (
            sample.mean_visual_response < thresholds.idle_visual_response_threshold
        )
        idle = bool(
            warm
            and thresholds.no_progress_steps is not None
            and sample.mean_command < thresholds.idle_command_threshold
            and sample.mean_state_response < thresholds.idle_state_response_threshold
            and visual_ok_idle
        )
        idle_streak = idle_streak + 1 if idle else 0
        no_progress = bool(
            thresholds.no_progress_steps is not None
            and idle_streak >= thresholds.no_progress_steps
        )
        stale = sample.action_age_steps >= thresholds.stale_action_steps
        events.append(
            STALL_EVENT
            if stalled
            else NO_PROGRESS_EVENT
            if no_progress
            else STALE_ACTION_EVENT
            if stale
            else None
        )
    return tuple(events)


@dataclasses.dataclass(frozen=True)
class EpisodeTriggerEvidence:
    """What a candidate trigger would have done in one completed episode."""

    episode_index: int
    steps: int
    success: bool
    event_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("episode step count must be positive")
        if any(step < 0 for step in self.event_steps):
            raise ValueError("event steps must be non-negative")
        if list(self.event_steps) != sorted(self.event_steps):
            raise ValueError("event steps must be ordered")

    @property
    def fired(self) -> bool:
        return bool(self.event_steps)

    @property
    def first_event_step(self) -> int | None:
        return self.event_steps[0] if self.event_steps else None

    def occasions(self, cooldown_steps: int) -> int:
        """Firing occasions after collapsing events inside one cooldown window.

        The escalation path applies a cooldown, so a run of consecutive event
        steps is one opportunity to intervene, not hundreds.
        """
        if cooldown_steps <= 0:
            raise ValueError("cooldown_steps must be positive")
        count = 0
        last: int | None = None
        for step in self.event_steps:
            if last is None or step - last >= cooldown_steps:
                count += 1
                last = step
        return count


@dataclasses.dataclass(frozen=True)
class TriggerAdmissionRequirements:
    """Evidence thresholds a trigger must meet before it may intervene."""

    minimum_episodes: int = 5
    minimum_firing_episodes: int = 1
    # Specificity is expressed as lift over the base failure rate rather than as
    # raw precision: on a task that fails most of the time, a trigger that fires
    # everywhere would look precise while carrying no information.
    minimum_failure_lift: float = 1.0
    # A first event closer than this to the end of the episode cannot lead to a
    # bounded recovery, because the escalation chain is serial.
    minimum_recovery_headroom_steps: int = 200
    maximum_occasions_per_episode: float = 3.0
    cooldown_steps: int = 128

    def __post_init__(self) -> None:
        if self.minimum_episodes <= 0:
            raise ValueError("minimum_episodes must be positive")
        if self.minimum_firing_episodes < 0:
            raise ValueError("minimum_firing_episodes must be non-negative")
        if self.minimum_failure_lift < 0:
            raise ValueError("minimum_failure_lift must be non-negative")
        if self.minimum_recovery_headroom_steps < 0:
            raise ValueError("minimum_recovery_headroom_steps must be non-negative")
        if self.maximum_occasions_per_episode <= 0:
            raise ValueError("maximum_occasions_per_episode must be positive")
        if self.cooldown_steps <= 0:
            raise ValueError("cooldown_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TriggerAdmissionDecision:
    """Auditable result of validating one candidate trigger."""

    accepted: bool
    status: str
    scope: str
    violations: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "scope": self.scope,
            "violations": list(self.violations),
            "metrics": dict(self.metrics),
        }

    def require_accepted(self) -> None:
        if not self.accepted:
            detail = "; ".join(self.violations) or "trigger is not admitted"
            raise ValueError(f"harness trigger admission failed: {detail}")


def validate_trigger_admission(
    evidence: Iterable[EpisodeTriggerEvidence],
    requirements: TriggerAdmissionRequirements | None = None,
    *,
    scope: str = "harness_trigger",
) -> TriggerAdmissionDecision:
    """Decide whether a trigger has earned the right to change robot behaviour."""

    requirements = requirements or TriggerAdmissionRequirements()
    episodes = list(evidence)
    violations: list[str] = []

    if not episodes:
        return TriggerAdmissionDecision(
            accepted=False,
            status="no_evidence",
            scope=scope,
            violations=("no episodes supplied",),
        )

    total = len(episodes)
    failures = [item for item in episodes if not item.success]
    fired = [item for item in episodes if item.fired]
    fired_and_failed = [item for item in fired if not item.success]

    base_failure_rate = len(failures) / total
    precision = (len(fired_and_failed) / len(fired)) if fired else None
    recall = (len(fired_and_failed) / len(failures)) if failures else None
    lift = (
        None
        if precision is None or base_failure_rate == 0.0
        else precision / base_failure_rate
    )

    headroom = [
        item.steps - item.first_event_step
        for item in fired
        if item.first_event_step is not None
    ]
    actionable = [
        value
        for value in headroom
        if value >= requirements.minimum_recovery_headroom_steps
    ]
    occasions = [item.occasions(requirements.cooldown_steps) for item in fired]
    mean_occasions = (sum(occasions) / len(occasions)) if occasions else 0.0

    if total < requirements.minimum_episodes:
        violations.append(
            f"only {total} episodes of evidence, "
            f"{requirements.minimum_episodes} required"
        )
    # Reachability.
    if len(fired) < requirements.minimum_firing_episodes:
        violations.append(
            f"trigger fired in {len(fired)}/{total} episodes, "
            f"{requirements.minimum_firing_episodes} required: an unreachable "
            "branch is a defect, not a conservative default"
        )
    # Specificity.
    if fired:
        if lift is None:
            violations.append(
                "no failed episodes in evidence, so specificity cannot be shown"
            )
        elif lift < requirements.minimum_failure_lift:
            violations.append(
                f"failure lift {lift:.2f} below required "
                f"{requirements.minimum_failure_lift:.2f} "
                f"(fires on {precision:.2f} of episodes it selects vs base "
                f"failure rate {base_failure_rate:.2f}): intervening on this "
                "signal adds risk without adding information"
            )
        # Timeliness.
        if not actionable:
            violations.append(
                "no firing episode leaves "
                f"{requirements.minimum_recovery_headroom_steps} steps of "
                "headroom, so no bounded recovery could run"
            )
        # Budget feasibility.
        if mean_occasions > requirements.maximum_occasions_per_episode:
            violations.append(
                f"{mean_occasions:.2f} firing occasions per episode exceeds the "
                f"budget of {requirements.maximum_occasions_per_episode:.2f}"
            )

    metrics = {
        "episodes": total,
        "base_failure_rate": round(base_failure_rate, 4),
        "firing_episodes": len(fired),
        "firing_episode_rate": round(len(fired) / total, 4),
        "failure_precision": None if precision is None else round(precision, 4),
        "failure_recall": None if recall is None else round(recall, 4),
        "failure_lift": None if lift is None else round(lift, 4),
        "mean_occasions_per_firing_episode": round(mean_occasions, 4),
        "actionable_firing_episodes": len(actionable),
        "median_headroom_steps": (
            None if not headroom else sorted(headroom)[len(headroom) // 2]
        ),
    }
    accepted = not violations
    return TriggerAdmissionDecision(
        accepted=accepted,
        status="admitted" if accepted else "rejected",
        scope=scope,
        violations=tuple(violations),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Whole-policy feasibility.
#
# Admitting a trigger is not enough. The escalation chain is serial, and its
# gates are configured independently: check interval, per-episode call budget,
# minimum spacing between counted observations, and the unchanged span a stall
# must cover. Those four can be individually reasonable and jointly leave no
# room for a recovery to ever fire -- a budget spent entirely on waiting.
#
# This models the chain and asks the prior question: *could* a recovery fire at
# all, before asking whether it would help.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class InterventionPolicy:
    """The full escalation configuration, not just the trigger thresholds."""

    checkpoint_steps: int = 160
    max_calls_per_episode: int = 3
    min_stall_interval_steps: int = 160
    min_unchanged_steps: int = 320
    stale_checks_required: int = 2
    max_recoveries: int = 1
    cooldown_steps: int = 128
    # The final calls are event-only; they do not increase the total budget.
    event_reserve_calls: int = 0
    observation_validity_steps: int = 320

    def __post_init__(self) -> None:
        if self.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        if self.max_calls_per_episode <= 0:
            raise ValueError("max_calls_per_episode must be positive")
        if self.min_stall_interval_steps < 0:
            raise ValueError("min_stall_interval_steps must be non-negative")
        if self.min_unchanged_steps < 0:
            raise ValueError("min_unchanged_steps must be non-negative")
        if self.stale_checks_required <= 0:
            raise ValueError("stale_checks_required must be positive")
        if self.max_recoveries <= 0:
            raise ValueError("max_recoveries must be positive")
        if self.cooldown_steps <= 0:
            raise ValueError("cooldown_steps must be positive")
        if type(self.event_reserve_calls) is not int or not 0 <= self.event_reserve_calls <= self.max_calls_per_episode:
            raise ValueError("event reserve must fit inside the total call budget")
        if type(self.observation_validity_steps) is not int or self.observation_validity_steps < 0:
            raise ValueError("observation validity must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def escalation_candidate_steps(event_steps: Sequence[int]) -> tuple[int, ...]:
    """Steps where a Monitor event has an event_streak of at least two.

    The escalation path routes a first-onset event to a cheap replan and only a
    sustained event to the semantic checker, so a forced semantic check requires
    the previous step to have carried the same event.
    """
    present = set(int(step) for step in event_steps)
    return tuple(step for step in sorted(present) if step - 1 in present)


@dataclasses.dataclass(frozen=True)
class EpisodePolicyFeasibility:
    """Whether the policy leaves room for a recovery in one episode."""

    episode_index: int
    steps: int
    success: bool
    checks: tuple[int, ...]
    counted_checks: tuple[int, ...]
    earliest_recovery_step: int | None
    budget_exhausted: bool

    @property
    def feasible(self) -> bool:
        return self.earliest_recovery_step is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "steps": self.steps,
            "success": self.success,
            "checks": list(self.checks),
            "counted_checks": list(self.counted_checks),
            "earliest_recovery_step": self.earliest_recovery_step,
            "budget_exhausted": self.budget_exhausted,
            "feasible": self.feasible,
        }


def simulate_episode_policy(
    *,
    episode_index: int,
    steps: int,
    success: bool,
    event_steps: Sequence[int],
    policy: InterventionPolicy,
    event_name: str = NO_PROGRESS_EVENT,
) -> EpisodePolicyFeasibility:
    """Replay the escalation chain's gates over one episode's event timeline.

    Models periodic slots using the last periodic check, independently of
    evidence counting; a forced check bypasses the periodic interval but not
    the escalation cooldown, and not the per-episode call budget.

    Event routing follows the RoboDojo scalar-Critic bridge: only no_progress
    forces a check. Timing assumes observations and verifier calls are available
    and no earlier Planner decision changes the trajectory or clocks.

    Recovery feasibility is a conditional upper bound: it assumes the predicate reading is
    maximally stall-like, i.e. never changes. If a recovery cannot fire even
    then, the configuration cannot act at all, whatever the scene does.
    """
    if event_name not in (NO_PROGRESS_EVENT, STALL_EVENT, STALE_ACTION_EVENT):
        raise ValueError(f"unknown Monitor event: {event_name}")
    forced = (
        set(escalation_candidate_steps(event_steps))
        if event_name == NO_PROGRESS_EVENT else set()
    )
    checks: list[int] = []
    counted: list[int] = []
    last_counted: int | None = None
    last_escalation: int | None = None
    budget_exhausted = False
    unchanged_since: int | None = None
    stale_checks = 0
    earliest: int | None = None

    last_periodic: int | None = None
    for step in range(steps):
        if len(checks) >= policy.max_calls_per_episode:
            budget_exhausted = True
            break
        # The periodic slot advances on a periodic check, not on a counted
        # observation: otherwise a check that does not count leaves the gate open
        # on every following step. A forced check does not advance the slot, so
        # it cannot delay the next scheduled observation.
        reference = 0 if last_periodic is None else last_periodic
        periodic = (
            step >= policy.checkpoint_steps
            and step - reference >= policy.checkpoint_steps
        )
        forced_now = step in forced and (
            last_escalation is None
            or step - last_escalation >= policy.cooldown_steps
        )
        if not (periodic or forced_now):
            continue
        if not forced_now and policy.max_calls_per_episode - len(checks) <= policy.event_reserve_calls:
            continue
        if forced_now:
            last_escalation = step
        if periodic:
            last_periodic = step
        checks.append(step)
        if (
            policy.observation_validity_steps > 0
            and last_counted is not None
            and step - last_counted > policy.observation_validity_steps
        ):
            unchanged_since = None
            stale_checks = 0
        if unchanged_since is None:
            unchanged_since = step
        if last_counted is None or step - last_counted >= policy.min_stall_interval_steps:
            counted.append(step)
            last_counted = step
            stale_checks += 1
            if (
                earliest is None
                and stale_checks >= policy.stale_checks_required
                and step - unchanged_since >= policy.min_unchanged_steps
            ):
                earliest = step

    return EpisodePolicyFeasibility(
        episode_index=episode_index,
        steps=steps,
        success=success,
        checks=tuple(checks),
        counted_checks=tuple(counted),
        earliest_recovery_step=earliest,
        budget_exhausted=budget_exhausted,
    )


@dataclasses.dataclass(frozen=True)
class PolicyFeasibilityDecision:
    """Auditable verdict on whether a policy can act within its own budget."""

    accepted: bool
    status: str
    scope: str
    violations: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "scope": self.scope,
            "violations": list(self.violations),
            "metrics": dict(self.metrics),
        }

    def require_accepted(self) -> None:
        if not self.accepted:
            detail = "; ".join(self.violations) or "policy is not feasible"
            raise ValueError(f"intervention policy feasibility failed: {detail}")


def validate_policy_feasibility(
    episodes: Iterable[EpisodePolicyFeasibility],
    *,
    minimum_feasible_rate: float = 0.5,
    minimum_feasible_failures: int = 1,
    scope: str = "intervention_policy",
) -> PolicyFeasibilityDecision:
    """Reject a configuration whose budget is spent before it can act."""

    items = list(episodes)
    if not items:
        return PolicyFeasibilityDecision(
            accepted=False,
            status="no_evidence",
            scope=scope,
            violations=("no episodes supplied",),
        )
    if not 0.0 <= minimum_feasible_rate <= 1.0:
        raise ValueError("minimum_feasible_rate must be in [0, 1]")

    feasible = [item for item in items if item.feasible]
    failures = [item for item in items if not item.success]
    feasible_failures = [item for item in feasible if not item.success]
    rate = len(feasible) / len(items)
    headroom = [
        item.steps - item.earliest_recovery_step
        for item in feasible
        if item.earliest_recovery_step is not None
    ]

    violations: list[str] = []
    if rate < minimum_feasible_rate:
        violations.append(
            f"a recovery is reachable in only {len(feasible)}/{len(items)} "
            f"episodes ({rate:.2f}), below the required "
            f"{minimum_feasible_rate:.2f}: the budget is spent on waiting"
        )
    if failures and len(feasible_failures) < minimum_feasible_failures:
        violations.append(
            f"a recovery is reachable in only {len(feasible_failures)} of "
            f"{len(failures)} failed episodes, so the policy cannot act where "
            "it would matter"
        )

    metrics = {
        "episodes": len(items),
        "feasible_episodes": len(feasible),
        "feasible_rate": round(rate, 4),
        "failed_episodes": len(failures),
        "feasible_failed_episodes": len(feasible_failures),
        "budget_exhausted_episodes": sum(1 for item in items if item.budget_exhausted),
        "mean_checks_per_episode": round(
            sum(len(item.checks) for item in items) / len(items), 4
        ),
        "mean_counted_checks_per_episode": round(
            sum(len(item.counted_checks) for item in items) / len(items), 4
        ),
        "median_recovery_headroom_steps": (
            None if not headroom else sorted(headroom)[len(headroom) // 2]
        ),
        "earliest_recovery_step_median": (
            None
            if not feasible
            else sorted(
                item.earliest_recovery_step
                for item in feasible
                if item.earliest_recovery_step is not None
            )[len(feasible) // 2]
        ),
    }
    accepted = not violations
    return PolicyFeasibilityDecision(
        accepted=accepted,
        status="feasible" if accepted else "infeasible",
        scope=scope,
        violations=tuple(violations),
        metrics=metrics,
    )

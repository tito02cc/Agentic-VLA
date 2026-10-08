"""The StarVLA episode-state contract shared by the adapter, the audit and tests.

``STARVLA_RESET_STATE_REGISTRY`` is the single source of truth for what "the
episode was reset" means. ``Model.reset`` builds its receipt by resolving this
registry against the live model, the wire contract validates the receipt against
it, and the drift tests compare it with the attributes the adapter actually
declares. Adding per-episode state without registering it here is therefore a
test failure rather than a silent cross-episode leak.

Resolution is deliberately strict: a registered attribute that is missing or has
the wrong container type raises instead of defaulting to an empty container. A
default would report ``before=0 / after=0`` for a renamed attribute and satisfy
every downstream check while the real container kept the previous episode's
contents.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STARVLA_RESET_STATE_VERSION = "starvla.episode-state.v1"

# (inventory name, model attribute, container kind)
STARVLA_RESET_STATE_REGISTRY: tuple[tuple[str, str, str], ...] = (
    ("obs_by_env", "obs_by_env", "dict"),
    ("action_chunks_by_env", "action_chunks_by_env", "dict"),
    ("chunk_start_by_env", "chunk_start_by_env", "dict"),
    ("active_execute_horizon_by_env", "_active_execute_horizon_by_env", "dict"),
    ("step_by_env", "step_by_env", "dict"),
    ("last_action_by_env", "_last_action_by_env", "dict"),
    ("last_planner_step_by_env", "_last_planner_step_by_env", "dict"),
    (
        "recovery_compute_remaining_by_env",
        "_recovery_compute_remaining_by_env",
        "dict",
    ),
    ("semantic_last_check_by_env", "_semantic_last_check_by_env", "dict"),
    # The step of the last observation that was allowed to count toward the
    # stall streak. Distinct from semantic_last_check_by_env, which records any
    # check including a Monitor-forced one that is too close to count.
    (
        "semantic_last_counted_step_by_env",
        "_semantic_last_counted_step_by_env",
        "dict",
    ),
    # The step of the last *periodic* semantic check. Separate from
    # semantic_last_counted_step_by_env: the periodic slot must advance even when
    # an observation does not count as stall evidence, or the gate stays open on
    # every following step and burns the call budget.
    (
        "semantic_last_periodic_step_by_env",
        "_semantic_last_periodic_step_by_env",
        "dict",
    ),
    # The step from which the predicate value has been unchanged. Stall evidence
    # is a span of steps, not a count of checks, so this must not leak across
    # episodes or episode N+1 inherits N's apparent stall.
    (
        "semantic_unchanged_since_step_by_env",
        "_semantic_unchanged_since_step_by_env",
        "dict",
    ),
    ("semantic_last_value_by_env", "_semantic_last_value_by_env", "dict"),
    ("semantic_stale_checks_by_env", "_semantic_stale_checks_by_env", "dict"),
    ("semantic_calls_by_env", "_semantic_calls_by_env", "dict"),
    ("semantic_recoveries_by_env", "_semantic_recoveries_by_env", "dict"),
    ("visual_critic_calls_by_env", "_visual_critic_calls_by_env", "dict"),
    ("visual_critic_disabled_envs", "_visual_critic_disabled_envs", "set"),
    ("high_level_agent_disabled_envs", "_high_level_agent_disabled_envs", "set"),
    (
        "semantic_guidance_remaining_by_env",
        "_semantic_guidance_remaining_by_env",
        "dict",
    ),
    (
        "semantic_guidance_instruction_by_env",
        "_semantic_guidance_instruction_by_env",
        "dict",
    ),
    ("agent_memory_by_env", "_agent_memory_by_env", "dict"),
    ("task_instruction_by_env", "_task_instruction_by_env", "dict"),
    ("task_plan_by_env", "_task_plan_by_env", "dict"),
    ("task_plan_attempted_envs", "_task_plan_attempted_envs", "set"),
    ("task_plan_abandoned_envs", "_task_plan_abandoned_envs", "set"),
    ("task_plan_last_check_by_env", "_task_plan_last_check_by_env", "dict"),
    ("task_plan_checks_by_env", "_task_plan_checks_by_env", "dict"),
    (
        "task_plan_contradiction_streak_by_env",
        "_task_plan_contradiction_streak_by_env",
        "dict",
    ),
    (
        "task_plan_protocol_failures_by_env",
        "_task_plan_protocol_failures_by_env",
        "dict",
    ),
    ("safe_stop_envs", "_safe_stop_envs", "set"),
    ("fault_recovered_envs", "_fault_recovered_envs", "set"),
    ("fault_announced_envs", "_fault_announced_envs", "set"),
    ("risk_monitors", "_risk_monitors", "dict"),
    ("reobservation_by_env", "_reobservation_by_env", "dict"),
    ("recovery_verification_by_env", "_recovery_verification_by_env", "dict"),
)

STARVLA_RESET_STATE_FIELDS: tuple[str, ...] = tuple(
    name for name, _attribute, _kind in STARVLA_RESET_STATE_REGISTRY
)

_CONTAINER_TYPES: dict[str, type] = {"dict": dict, "set": set}


def _validate_registry() -> None:
    names = [name for name, _, _ in STARVLA_RESET_STATE_REGISTRY]
    attributes = [attribute for _, attribute, _ in STARVLA_RESET_STATE_REGISTRY]
    if len(set(names)) != len(names):
        raise RuntimeError("StarVLA reset registry has duplicate inventory names")
    if len(set(attributes)) != len(attributes):
        raise RuntimeError("StarVLA reset registry has duplicate attributes")
    for _name, _attribute, kind in STARVLA_RESET_STATE_REGISTRY:
        if kind not in _CONTAINER_TYPES:
            raise RuntimeError(f"StarVLA reset registry has unknown kind {kind!r}")


_validate_registry()


def resolve_reset_state_containers(model: Any) -> dict[str, Any]:
    """Return the live per-episode containers named by the registry.

    Raises rather than substituting an empty container: an unregistered rename
    must surface as a loud failure, not as a receipt that claims zero entries.
    """

    containers: dict[str, Any] = {}
    for name, attribute, kind in STARVLA_RESET_STATE_REGISTRY:
        if not hasattr(model, attribute):
            raise RuntimeError(
                f"StarVLA episode state {name!r} is missing its container "
                f"attribute {attribute!r}; the reset registry has drifted"
            )
        container = getattr(model, attribute)
        expected = _CONTAINER_TYPES[kind]
        if not isinstance(container, expected):
            raise RuntimeError(
                f"StarVLA episode state {name!r} ({attribute!r}) must be a "
                f"{kind}, got {type(container).__name__}"
            )
        containers[name] = container
    return containers


def validate_starvla_reset_state(receipt: Mapping[str, Any]) -> None:
    if receipt.get("state_inventory_version") != STARVLA_RESET_STATE_VERSION:
        raise ValueError("unsupported StarVLA reset-state inventory")
    inventory = receipt.get("state_inventory")
    if list(inventory or []) != list(STARVLA_RESET_STATE_FIELDS):
        raise ValueError("StarVLA reset receipt state inventory mismatch")
    after = receipt.get("after")
    if not isinstance(after, Mapping):
        raise ValueError("StarVLA reset receipt has no after-state")
    entries = after.get("state_entries")
    if not isinstance(entries, Mapping):
        raise ValueError("StarVLA reset receipt has no state-entry counts")
    # Compared as a set, not a sequence: JSON objects carry no ordering guarantee
    # and the evidence writers canonicalize with sort_keys=True, which reorders
    # this mapping on the way to disk. The ordered contract is carried by
    # state_inventory, which is a list and survives serialization intact; a dict
    # cannot hold duplicates, so set equality plus that list is exact.
    if set(entries) != set(STARVLA_RESET_STATE_FIELDS):
        missing = sorted(set(STARVLA_RESET_STATE_FIELDS) - set(entries))
        unexpected = sorted(set(entries) - set(STARVLA_RESET_STATE_FIELDS))
        raise ValueError(
            "StarVLA reset receipt state-entry keys mismatch"
            f" (missing={missing}, unexpected={unexpected})"
        )
    if any(type(value) is not int or value != 0 for value in entries.values()):
        raise ValueError("StarVLA reset left episode state populated")
    if type(after.get("high_level_calls")) is not int or after.get(
        "high_level_calls"
    ) != 0:
        raise ValueError("StarVLA high-level budget did not reset")

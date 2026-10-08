import copy

import pytest

from agentic_vla.runtime.agent import (
    AgentIntent, GuardedHighLevelAgent, HighLevelAgentConfig, HighLevelAgentContext,
    bind_cumulative_task_outcome,
)


def context(**kwargs):
    return HighLevelAgentContext(
        task_instruction="put the mug on the plate", trigger=kwargs.get("trigger", "task_start"),
        episode_id=0, timestep=0, allowed_intents=kwargs.get("allowed", (AgentIntent.VLA_ACT,)),
    )


def proposal():
    return {"confidence": 0.9, "stages": [{"subgoal": "put the mug on the plate",
                                          "expected_outcome": "mug rests on plate"}]}


def decide(payload, **kwargs):
    return GuardedHighLevelAgent(lambda _: payload,
        HighLevelAgentConfig(max_grounding_repairs=0)).decide(context(**kwargs))


def test_semantic_proposal_compiles_constants_and_preserves_meaning():
    payload = proposal()
    before = copy.deepcopy(payload)
    result = decide(payload)
    assert result.accepted
    step = result.decision.proposed_plan[0]
    assert step.intent == "vla_act" and step.skill_id is None
    assert step.subgoal == payload["stages"][0]["subgoal"]
    assert payload == before


@pytest.mark.parametrize("field,value", [("intent", "pick"), ("skill_id", "unregistered"),
                                       ("joint_actions", [1]), ("constraints", [])])
def test_semantic_rows_cannot_inject_executor_fields(field, value):
    payload = proposal()
    payload["stages"][0][field] = value
    assert not decide(payload).accepted


@pytest.mark.parametrize("stages", [[], [{}], [None], ["put mug"],
    [{"subgoal": 3, "expected_outcome": "done"}],
    [{"subgoal": "put mug", "expected_outcome": ""}], proposal()["stages"] * 5])
def test_malformed_semantic_rows_fail_closed(stages):
    assert not decide({"confidence": 0.9, "stages": stages}).accepted


def test_semantic_protocol_retains_grounding_and_granularity_guards():
    payload = proposal()
    payload["stages"][0]["subgoal"] = "put the blue mug on the plate"
    assert not decide(payload).accepted
    payload["stages"] = [
        {"subgoal": "pick up the mug", "expected_outcome": "mug lifted"},
        {"subgoal": "put the mug on the plate", "expected_outcome": "mug on plate"},
    ]
    assert not decide(payload).accepted


def test_semantic_protocol_cannot_grant_permissions_or_rewrite_active_plan():
    assert not decide(proposal(), allowed=(AgentIntent.CONTINUE,)).accepted
    assert not decide(proposal(), trigger="no_progress").accepted


def test_semantic_protocol_repairs_once_with_consistent_schema():
    requests = []

    def infer(request):
        requests.append(request)
        return {"confidence": 0.9, "stages": []} if len(requests) == 1 else proposal()

    result = GuardedHighLevelAgent(infer).decide(context())
    assert result.accepted and result.attempt_count == 2
    assert "only confidence and stages" in requests[1]["user_prompt"]


@pytest.mark.parametrize("stages,accepted", [
    ([{"subgoal": "Pick up the bottle", "expected_outcome": "bottle held"}], False),
    ([{"subgoal": "Pick up the bottles and throw them into the basket",
       "expected_outcome": "bottles in dustbin"}], False),
    ([{"subgoal": "Pick up the bottle and throw it into the dustbin",
       "expected_outcome": "bottle in dustbin"}], False),
    ([{"subgoal": "Pick up the bottles and throw them into the dustbin, using handover when needed.",
       "expected_outcome": "bottles in dustbin"}], True),
    ([{"subgoal": "Pick up the bottles", "expected_outcome": "bottles held"},
      {"subgoal": "Throw the bottles into the dustbin", "expected_outcome": "bottles in dustbin"}], False),
])
def test_compound_transfer_cannot_drop_destination_or_plural_scope(stages, accepted):
    task = "Pick up the bottles and throw them into the dustbin, using handover when needed."
    ctx = HighLevelAgentContext(task_instruction=task, trigger="task_start", episode_id=0, timestep=0)
    result = GuardedHighLevelAgent(lambda _: {"confidence": 0.9, "stages": stages},
        HighLevelAgentConfig(max_grounding_repairs=0)).decide(ctx)
    assert result.accepted is accepted
    if accepted:
        bound = bind_cumulative_task_outcome(result.decision.proposed_plan, task)
        assert bound[-1].expected_outcome == f"full task visibly satisfied: {task.lower()}"

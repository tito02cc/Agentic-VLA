"""CPU contract tests, NOT robot success or VLM capability measurements."""

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_vla.configuration import (
    ArtifactRunConfig, BenchmarkRunConfig, CarveRunConfig, HarnessRunConfig,
    OptimizeRunConfig, PlannerRunConfig, VlaRunConfig,
)
from agentic_vla.benchmarks.robodojo_sorting import SORTING_TASKS, SortingObservation, SortingWorkingMemory
from agentic_vla.benchmarks.robodojo_pi05_episode import SortingEpisode, SortingExecutionConfig
from agentic_vla.runtime.knowledge import ProceduralStep
from agentic_vla.toolchain import EmbodiedTaskPlan, VerificationReport


def config(tmp_path, task="classify_objects_by_language"):
    return CarveRunConfig(
        run_id="fixture", planner=PlannerRunConfig(mode="scripted", planner_id="fixture", max_calls_per_episode=6),
        harness=HarnessRunConfig(primitive_boundary_policy="event_only", safe_hold_timeout_s=60),
        vla=VlaRunConfig("official-pi05", "pi05", "pi05", "59999", "native-jax-h50-flow10"),
        optimize=OptimizeRunConfig(enabled=False),
        benchmark=BenchmarkRunConfig("robodojo", "development", task, 0, SORTING_TASKS[task].max_steps),
        artifacts=ArtifactRunConfig(results_root=str(tmp_path)),
    )


class PortFixture:
    def __init__(self, steps=150, stalled=False):
        self.steps = 0
        self.limit = steps
        self.state = np.zeros(14, dtype=np.float32)
        self.instructions = []
        self.stalled = stalled
        self.instruction = "Put red objects into the left basket, then reset the robot arm."

    def observe(self):
        frame = np.full((32, 32, 3), self.steps % 255, np.uint8)
        return SortingObservation(self.instruction, self.state.copy(),
                                  {k: frame for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")})

    def infer(self, instruction):
        self.instructions.append(instruction)
        return np.stack([self.state + .01 * (i + 1) for i in range(50)])

    def execute(self, action):
        self.steps += 1
        if not self.stalled:
            self.state = action.copy()

    def done(self):
        return self.steps >= self.limit


def continue_planner(request):
    return {"intent": "continue", "rationale": "continue reference execution", "confidence": .9}


def test_vla_proposal_is_immutable_and_paid_once(tmp_path):
    port = PortFixture(steps=200)
    e = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
        execution=SortingExecutionConfig(agent_enabled=False, vla_call_budget=1))
    e._capture()
    p = e._prepare_vla_proposal(e.instruction)
    original = p.actions().copy()
    with pytest.raises(ValueError):
        p.actions().setflags(write=True)
    with pytest.raises(RuntimeError, match="pending"):
        e._prepare_vla_proposal(e.instruction)
    result = e._execute_vla_proposal(p.proposal_id, execution_prefix=10)
    assert result.metadata["discarded_steps"] == 40
    assert len(port.instructions) == 1 and e.vla_calls == 1 and e.step == 10
    np.testing.assert_array_equal(port.state, original[9])
    with pytest.raises(ValueError, match="consumed"):
        e._execute_vla_proposal(p.proposal_id, execution_prefix=10)
    e.session.close()


@pytest.mark.parametrize("changed", ["state", "image", "time", "stop"])
def test_pending_vla_rejects_stale_evidence(tmp_path, changed):
    port = PortFixture(steps=200)
    e = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
        execution=SortingExecutionConfig(agent_enabled=False))
    e._capture()
    p = e._prepare_vla_proposal(e.instruction)
    if changed == "state":
        port.state[0] += .001
    elif changed == "image":
        port.steps = 1
    elif changed == "time":
        e.step = 1
    else:
        e.stopped = True
    before = port.steps
    with pytest.raises(ValueError, match="stale"):
        e._execute_vla_proposal(p.proposal_id, execution_prefix=10)
    assert port.steps == before and e._pending_vla_proposal is None
    e.session.close()


def test_stop_discards_pending_without_action(tmp_path):
    e = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner)
    e._capture()
    e._prepare_vla_proposal(e.instruction)
    e._stop("test", e._tool_context())
    assert e._pending_vla_proposal is None and e.step == 0
    e.session.close()


def test_proposal_execution_skill_is_not_enabled_by_default(tmp_path):
    e = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner)
    e._capture()
    e._prepare_vla_proposal(e.instruction)
    with pytest.raises(ValueError):
        e._run_skill("execute_prefix_10", {}, e._tool_context())
    assert e.step == 0
    e._stop("test", e._tool_context())
    e.session.close()


@pytest.mark.parametrize("choice,expected_steps", [("execute_prefix_10", 20), ("execute_prefix_50", 100), ("safe_stop", 0)])
def test_actual_harness_routes_proposal_choice_without_reinference(tmp_path, monkeypatch, choice, expected_steps):
    from agentic_vla.benchmarks import robodojo_action_proposal as module
    requests = []
    def planner(request):
        requests.append(json.loads(request["user_prompt"]))
        return {"action": choice, "confidence": .95, "reason": "fixture-only choice", "memory_note": ""}
    def preview(port, proposal):
        return {**proposal.identity(), "collision_checked": False}
    monkeypatch.setattr(module, "robot_motion_preview", preview)
    port = PortFixture(steps=200)
    e = SortingEpisode(config(tmp_path), port, scripted_infer=planner,
        execution=SortingExecutionConfig(agent_enabled=True, vla_call_budget=2,
                                         proposal_review_enabled=True))
    result = module.run_proposal_review_smoke(e)
    assert result["control_steps"] == expected_steps
    assert result["vla_calls"] == (1 if choice == "safe_stop" else 2)
    assert result["critic"]["calls"] == 0 and not result["planner_choice_forced"]
    assert all(r["protocol"] == "paid_vla_proposal_review_v1" for r in requests)
    assert all("action_proposal" in r for r in requests)
    assert e._pending_vla_proposal is None
    if expected_steps:
        assert requests[0]["action_proposal"]["proposal_id"] != requests[1]["action_proposal"]["proposal_id"]
        assert requests[1]["last_primitive"]


def test_bad_proposal_review_does_not_fall_back_to_blind_execution(tmp_path, monkeypatch):
    from agentic_vla.benchmarks import robodojo_action_proposal as module
    monkeypatch.setattr(module, "robot_motion_preview", lambda port, p: p.identity())
    cfg = config(tmp_path)
    cfg = dataclasses.replace(cfg, planner=dataclasses.replace(cfg.planner, max_grounding_repairs=0))
    e = SortingEpisode(cfg, PortFixture(), scripted_infer=lambda r: {"action": "continue", "confidence": .9, "reason": "not allowed"},
        execution=SortingExecutionConfig(proposal_review_enabled=True, vla_call_budget=2))
    result = module.run_proposal_review_smoke(e)
    assert result["control_steps"] == 0 and e._pending_vla_proposal is None


def test_proposal_context_rejects_privileged_object_input(tmp_path):
    e = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner)
    e._capture()
    with pytest.raises(ValueError, match="forbidden"):
        dataclasses.replace(e._context(), action_proposal={"object_pose": [1, 2, 3]})
    e.session.close()


def test_kinematics_smoke_executes_only_original_vla_actions(tmp_path):
    from agentic_vla.benchmarks.robodojo_pi05_episode import run_kinematics_smoke
    port = PortFixture(steps=200)
    audit_steps = []
    def audit():
        audit_steps.append(port.steps)
        return {"fk_passed": True, "physical_probe_steps": 0}
    def forbidden(request):
        pytest.fail("diagnostic must not call Planner or Critic")
    port.audit_robot_kinematics = audit
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=forbidden,
        execution=SortingExecutionConfig(agent_enabled=False, vla_call_budget=1))
    summary = run_kinematics_smoke(episode)
    assert audit_steps == [0, 50]
    assert port.instructions == [port.instruction]
    assert summary["ik_actions_executed"] == 0
    assert not summary["autonomous_planner_used"]
    with pytest.raises(RuntimeError, match="single-use"):
        run_kinematics_smoke(episode)


def test_kinematics_smoke_bad_calibration_has_zero_actions(tmp_path):
    from agentic_vla.benchmarks.robodojo_pi05_episode import run_kinematics_smoke
    port = PortFixture()
    port.audit_robot_kinematics = lambda: {"fk_passed": False}
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
        execution=SortingExecutionConfig(agent_enabled=False, vla_call_budget=1))
    with pytest.raises(RuntimeError, match="calibration failed"):
        run_kinematics_smoke(episode)
    assert not port.instructions and port.steps == 0


@pytest.mark.parametrize("overrides", [{"agent_enabled": True}, {"vla_call_budget": 2},
    {"subgoal_conditioning_admitted": True}, {"feedback_refresh_enabled": True}])
def test_kinematics_smoke_requires_diagnostic_permissions(tmp_path, overrides):
    from agentic_vla.benchmarks.robodojo_pi05_episode import run_kinematics_smoke
    cfg = {"agent_enabled": False, "vla_call_budget": 1, **overrides}
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
                             execution=SortingExecutionConfig(**cfg))
    with pytest.raises(ValueError, match="kinematics smoke"):
        run_kinematics_smoke(episode)
    assert not episode.ran
    episode.session.close()


def test_two_selected_tasks_only():
    assert set(SORTING_TASKS) == {"organize_table", "classify_objects_by_language"}


def test_default_critic_does_not_grant_completion(tmp_path):
    assert SortingExecutionConfig().critic_authority == "advisory"
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=lambda _: {"status": "confirmed", "confidence": 1,
                               "observed_outcome": "fixture says done"})
    episode._capture()
    result = episode._verify("goal", episode._tool_context())
    assert result.status.value == "inconclusive"
    assert result.metadata["completion_authorized"] is False
    episode.session.close()


def test_online_authoritative_critic_fails_before_model_call(tmp_path):
    cfg = dataclasses.replace(config(tmp_path), planner=PlannerRunConfig(
        mode="embedded_vlm", planner_id="fixture", provider="openai_compatible",
        model="fixture-model", endpoint="http://127.0.0.1:1/v1/chat/completions"))
    with pytest.raises(ValueError, match="no authoritative Critic profile"):
        SortingEpisode(cfg, PortFixture(),
                       execution=SortingExecutionConfig(critic_authority="authoritative"))
    assert not cfg.workspace_path.exists()


def test_step_budget_rejects_inference_before_call(tmp_path):
    port = PortFixture(steps=2000)
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner)
    episode._capture()
    episode.step = episode.task.max_steps
    with pytest.raises(RuntimeError, match="termination"):
        episode._vla_act(port.instruction, episode._tool_context())
    assert not port.instructions
    episode.session.close()


@pytest.mark.parametrize("prefix,limit,stalled", [(10, 100, False), (50, 7, False), (10, 100, True)])
def test_joint_feedback_uses_last_executed_not_generated_action(tmp_path, prefix, limit, stalled):
    port = PortFixture(steps=limit, stalled=stalled)
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
                             execution=SortingExecutionConfig(agent_enabled=False))
    episode._capture()
    result = episode._vla_act(port.instruction, episode._tool_context(), execution_prefix=prefix)
    feedback = result.metadata["joint_target_feedback"]
    expected_error = result.metadata["executed_steps"] * .01 if stalled else 0
    assert feedback["left"]["arm_max_abs_error_rad"] == pytest.approx(expected_error)
    assert feedback["semantic_outcome"] == "not_verified"
    assert result.metadata["semantic_outcome"] == "not_verified"
    assert result.metadata["executed_steps"] == min(prefix, limit)
    episode.session.close()


def test_reobserve_feedback_is_execution_fact_not_semantic_confirmation(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner)
    episode._capture()
    receipt = episode._run_skill("reobserve", {}, episode._tool_context())
    feedback = receipt.metadata["joint_target_feedback"]
    assert feedback["left"]["arm_max_abs_error_rad"] == 0
    assert feedback["semantic_outcome"] == "not_verified"
    episode.session.close()


def test_new_evidence_critic_is_advisory_only_until_admitted(tmp_path):
    with pytest.raises(ValueError, match="not admitted"):
        SortingExecutionConfig(critic_protocol="evidence", critic_authority="authoritative")
    port = PortFixture()
    episode = SortingEpisode(config(tmp_path), port,
        execution=SortingExecutionConfig(critic_protocol="evidence", critic_authority="advisory"),
        scripted_infer=continue_planner,
        critic_infer=lambda _: {"visibility": "clear", "relation": "supported",
            "evidence_views": ["current_cam_high"], "observed_outcome": "Red objects are inside the basket", "confidence": .99})
    episode._capture()
    report = episode._verify("Red objects are inside the basket", episode._tool_context())
    assert report.status.value == "inconclusive"
    assert report.metadata["raw_report"]["status"] == "confirmed"
    assert report.metadata["completion_authorized"] is False
    assert port.steps == 0


def test_deploy_uses_official_task_name_before_model_call(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "third_party/robodojo_official/XPolicyLab/policy/AgenticPi05/deploy.py"
    spec = importlib.util.spec_from_file_location("sorting_deploy_test", path)
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    cfg = config(tmp_path)
    monkeypatch.setenv("AGENTIC_PI05_RUN_CONFIG", "fixture")
    monkeypatch.setattr(deploy.CarveRunConfig, "load", lambda _: cfg)
    env = SimpleNamespace(task_name="organize_table", deploy_cfg={"policy_name": "AgenticPi05"})
    with pytest.raises(ValueError, match="task and Agent config disagree"):
        deploy.eval_one_episode(env, None)
    env.task_name = cfg.benchmark.task_id
    class ResetReached(Exception):
        pass
    def reset(**kwargs):
        raise ResetReached()
    with pytest.raises(ResetReached):
        deploy.eval_one_episode(env, SimpleNamespace(call=reset))


@pytest.mark.parametrize("terminal,actual_success", [(False, False), (False, True), (True, True), (True, False)])
def test_native_early_stop_does_not_inherit_success_default(terminal, actual_success):
    path = Path(__file__).resolve().parents[1] / "third_party/robodojo_official/XPolicyLab/policy/AgenticPi05/deploy.py"
    spec = importlib.util.spec_from_file_location("sorting_finalize_test", path)
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    class Environment:
        def __init__(self):
            self.success = [actual_success if terminal else True]
            self.end_flag = [terminal]
        def is_episode_end(self):
            if not self.end_flag[0] and not self.success[0]:
                self.end_flag[0] = True
                self.success[0] = actual_success
            return all(self.end_flag)
        def get_running_env_idx_list(self):
            return [i for i, ended in enumerate(self.end_flag) if not ended]
    env = Environment()
    assert deploy.finalize_episode(env) is (not terminal)
    assert env.success == [actual_success]
    assert env.end_flag == [True]


def test_compact_review_binds_ledger_without_granting_prompt_authority(tmp_path):
    from agentic_vla.runtime.agent import parse_high_level_agent_decision, build_high_level_agent_request
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
                             execution=SortingExecutionConfig(compact_task_only_review=True))
    episode._capture()
    context = dataclasses.replace(episode._context("execution_review"), task_plan={
        "installed": True, "active_stage": "1", "steps": [{
            "stage": "1", "intent": "vla_act", "subgoal": "put red objects in left basket",
            "expected_outcome": "all red objects in left basket", "skill_id": None}]})
    raw = {"action": "continue", "confidence": .8, "reason": "progress observed", "memory_note": "unverified"}
    decision = parse_high_level_agent_decision(raw, context)
    assert decision.intent.value == "continue"
    assert decision.vla_instruction is None
    assert decision.subgoal == context.task_plan["steps"][0]["subgoal"]
    assert decision.expected_outcome == context.task_plan["steps"][0]["expected_outcome"]
    assert json.loads(build_high_level_agent_request(context)["user_prompt"])["protocol"] == "task_only_review_v1"
    assert parse_high_level_agent_decision({**raw, "action": "safe_stop"}, context).intent.value == "safe_stop"
    for invalid in ({**raw, "action": "vla_act"}, {**raw, "vla_instruction": "new goal"}):
        with pytest.raises(ValueError):
            parse_high_level_agent_decision(invalid, context)
    with pytest.raises(ValueError):
        parse_high_level_agent_decision(raw, dataclasses.replace(context, compact_task_only_review=False))
    episode.session.close()


def test_reference_actions_and_prompt_unchanged(tmp_path):
    port = PortFixture()
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner)
    report = episode.run()
    assert report["vla_calls"] == 3
    assert report["control_steps"] == 150
    assert not report["stopped"]
    assert report["official_success"] is None
    assert port.instructions == [port.instruction] * 3
    assert len(report["inference_wall_ms"]) == 3
    with pytest.raises(RuntimeError, match="single-use"):
        episode.run()


def test_agent_off_uses_same_executor_without_model_or_monitor_intervention(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Agent-off must never call Planner/Critic")
    port = PortFixture(steps=150, stalled=True)
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=forbidden,
                             execution=SortingExecutionConfig(agent_enabled=False, memory_enabled=False))
    report = episode.run()
    assert report["agent_enabled"] is False
    assert report["control_steps"] == 150
    assert report["vla_calls"] == 3
    assert report["interrupted_chunks"] == 0
    assert episode.critic.calls == 0
    assert not report["stopped"]
    assert port.instructions == [port.instruction] * 3


def test_periodic_review_does_not_spend_failure_retry_budget(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(steps=500), scripted_infer=continue_planner)
    report = episode.run()
    assert report["control_steps"] == 500 and not report["stopped"]
    assert episode.session.harness.counters.semantic_retries == 0


def test_immediate_hold_never_waits_for_nonexistent_request(tmp_path):
    from agentic_vla.runtime.harness import HarnessState, HarnessTransition
    episode = SortingEpisode(config(tmp_path), PortFixture(steps=500), scripted_infer=continue_planner)
    episode.session.request_semantic_checkpoint = lambda *a, **k: HarnessTransition(
        state=HarnessState.SAFE_HOLD, reason="test retry budget exhausted")
    report = episode.run()
    assert report["control_steps"] == 100 and report["stopped"]
    assert report["stop_reason"] == "test retry budget exhausted"


@pytest.mark.parametrize("schedule,expected_calls", [("periodic", 9), ("tool_only", 0)])
def test_full_budget_advisory_lifecycle_with_compact_review(tmp_path, schedule, expected_calls):
    # Protocol fixture only: this test does not measure model or robot ability.
    port = PortFixture(steps=1000)
    clauses = ["place the mouse on the mouse pad", "push the keyboard into the frame",
               "put the figurine on the stand", "place the alarm clock on the drawer",
               "open the drawer", "put all remaining miscellaneous items inside"]
    port.instruction = ", ".join(clauses[:4]) + ", then " + " and ".join(clauses[4:]) + "."
    cfg = config(tmp_path, task="organize_table")
    cfg = dataclasses.replace(cfg, planner=dataclasses.replace(cfg.planner, max_calls_per_episode=16))
    def planner(request):
        payload = json.loads(request["user_prompt"])
        if payload.get("protocol") == "semantic_stages_v1":
            return {"confidence": .9, "stages": [
                {"subgoal": clause, "expected_outcome": clause} for clause in clauses]}
        return {"action": "continue", "confidence": .9, "reason": "fixture progression", "memory_note": "unverified"}
    episode = SortingEpisode(cfg, port, scripted_infer=planner,
        critic_infer=lambda request: {"status": "confirmed", "confidence": .99, "observed_outcome": "fixture"},
        execution=SortingExecutionConfig(critic_authority="advisory", critic_camera_mode="head",
                                        advisory_critic_schedule=schedule,
                                        compact_task_only_review=True))
    report = episode.run()
    assert report["control_steps"] == 1000 and report["vla_calls"] == 20
    assert not report["stopped"] and report["critic"]["calls"] == expected_calls
    assert report["deferred_critic_checks"] == 9 - expected_calls
    assert report["task_plan"]["active_stage"] == "1"
    assert not report["task_plan"]["completed"]
    assert port.instructions == [port.instruction] * 20


def test_invalid_model_output_never_reaches_robot(tmp_path):
    port = PortFixture()
    port.infer = lambda instruction: np.full((50, 14), np.nan)
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner)
    report = episode.run()
    assert port.steps == 0
    assert report["stopped"]


def test_model_stop_is_applied_without_robot_actions(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=lambda req: {
        "intent": "safe_stop", "rationale": "ambiguous target", "confidence": .9})
    report = episode.run()
    assert report["stopped"] and report["control_steps"] == 0


def test_subgoal_not_silently_enabled(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner)
    episode._capture()
    with pytest.raises(ValueError, match="capability admission"):
        episode._vla_act("Pick up red object", episode._tool_context())
    assert not episode.port.instructions
    episode.session.close()


def test_reobserve_really_advances_physics(tmp_path):
    port = PortFixture()
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner)
    episode._capture()
    result = episode._run_skill("reobserve", {}, episode._tool_context())
    assert port.steps == 1 and result.ended_timestep == 1
    assert not port.instructions
    with pytest.raises(ValueError, match="registered"):
        episode._run_skill("pick_using_hidden_pose", {}, episode._tool_context())
    episode.session.close()


def test_memory_is_episode_local_and_not_mutable():
    port = PortFixture()
    memory = SortingWorkingMemory(max_frames=2)
    memory.reset("first")
    memory.capture(0, port.observe())
    memory.remember_hypothesis(0, "red object probably moved")
    assert memory.records()[0]["authority"] == "unverified_model_note"
    with pytest.raises(ValueError):
        memory.frames[0][1][0, 0] = 99
    with pytest.raises(ValueError, match="increase"):
        memory.capture(0, port.observe())
    memory.reset("second")
    assert not memory.frames and not memory.notes


def test_contradiction_reopens_dependent_progress():
    plan = EmbodiedTaskPlan()
    plan.install([ProceduralStep(str(i), "vla_act", f"place item {i}", f"item {i} inside basket") for i in range(3)])
    for _ in range(2):
        plan.apply_verification(VerificationReport("confirmed", "object visibly inside basket", .9))
    plan.reopen_confirmed("0", VerificationReport("contradicted", "object now outside basket", .9))
    assert [r.status.value for r in plan.receipts] == ["retry_required", "pending", "pending"]
    assert not plan.completed
    with pytest.raises(ValueError):
        plan.reopen_confirmed("0", VerificationReport("inconclusive", "hidden", 0))


def test_legacy_optimization_is_not_applied(tmp_path):
    cfg = dataclasses.replace(config(tmp_path), optimize=OptimizeRunConfig(enabled=True))
    with pytest.raises(ValueError, match="admission"):
        SortingEpisode(cfg, PortFixture(), scripted_infer=continue_planner)


def test_critic_budget_and_public_context(tmp_path):
    requests = []
    def critic(req):
        requests.append(req)
        return {"status": "inconclusive", "observed_outcome": "object occluded", "confidence": 0}
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
                             critic_infer=critic, execution=SortingExecutionConfig(critic_call_budget=1))
    episode._capture()
    episode._verify("red object inside left basket", episode._tool_context())
    report = episode._verify("red object inside left basket", episode._tool_context())
    assert len(requests) == 1 and report.status.value == "inconclusive"
    context = episode._context()
    assert not any(k in context.risk for k in ("reward", "success", "object_pose", "target_label"))
    assert "category" not in requests[0]["user_prompt"]
    episode.session.close()


def test_planned_tool_runs_and_critic_advances_memory(tmp_path):
    port = PortFixture(steps=125)
    first = True
    def planner(request):
        nonlocal first
        if not first:
            return continue_planner(request)
        first = False
        return {
            "intent": "vla_act", "rationale": "red objects visible", "confidence": .9,
            "subgoal": "put red objects into left basket", "vla_instruction": port.instruction,
            "expected_outcome": "red objects inside left basket",
            "proposed_plan": [{"stage": "sort", "intent": "vla_act",
                               "subgoal": "put red objects into left basket",
                               "expected_outcome": "red objects inside left basket"}],
        }
    def critic(request):
        return {"status": "confirmed", "observed_outcome": "red objects visibly inside left basket", "confidence": .9}
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=planner, critic_infer=critic,
                             execution=SortingExecutionConfig(critic_authority="authoritative"))
    result = episode.run()
    assert not result["stopped"]
    assert episode.session.task_plan.completed
    assert result["critic"]["calls"] >= 1
    assert result["official_success"] is None


def test_numeric_monitor_uses_target_error_not_absolute_joint_value(tmp_path):
    port = PortFixture(steps=50)
    port.state[:] = 2
    port.infer = lambda instruction: np.tile(port.state, (50, 1))
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner)
    result = episode.run()
    assert result["interrupted_chunks"] == 0


def test_advisory_critic_cannot_confirm_ledger(tmp_path):
    requests = []
    def critic(request):
        requests.append(request)
        return {"status": "confirmed", "observed_outcome": "mouse on pad", "confidence": 1.0}
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=critic, execution=SortingExecutionConfig(critic_authority="advisory", critic_camera_mode="head"))
    episode._capture()
    report = episode._verify("mouse on pad", episode._tool_context())
    assert report.status.value == "inconclusive"
    assert report.metadata["completion_authorized"] is False
    assert report.metadata["raw_report"]["status"] == "confirmed"
    assert list(requests[0]["frames"]) == ["current_cam_high"]
    episode.session.close()


@pytest.mark.parametrize("kwargs", [{"critic_authority": "invented"}, {"critic_camera_mode": "invented"}])
def test_bad_execution_authority_rejected(kwargs):
    with pytest.raises(ValueError):
        SortingExecutionConfig(**kwargs)


def test_port_uses_official_rgb_and_action_codec(tmp_path, monkeypatch):
    from pathlib import Path
    from agentic_vla.benchmarks.robodojo_pi05_port import XPolicyLabPi05Port
    root = Path(__file__).resolve().parents[1] / "third_party/robodojo_official"
    monkeypatch.syspath_prepend(str(root))
    from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state
    dims = get_robot_action_dim_info("arx_x5")
    state = np.arange(14, dtype=np.float32) / 20
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:, :, 0] = 123
    class Env:
        def get_obs(self):
            return {"state": unpack_robot_state(state, "joint", dims),
                    "vision": {name: {"color": rgb.copy()} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                    "instruction": "sort objects", "reward": 100, "target_label": "secret"}
        def take_action(self, action):
            self.action = action
        def is_episode_end(self):
            return False
    class Client:
        def call(self, func_name, obs=None):
            if func_name == "update_obs":
                self.payload = obs
            elif func_name == "get_action":
                return unpack_robot_state(np.tile(state, (50, 1)), "joint", dims)
    env, client = Env(), Client()
    port = XPolicyLabPi05Port(env, client)
    observed = port.observe()
    assert observed.frames["cam_high"][0, 0].tolist() == [123, 0, 0]
    assert np.array_equal(port.infer("sort objects"), np.tile(state, (50, 1)))
    assert set(client.payload) == {"images", "state", "instruction"}
    port.execute(state)
    assert "left_arm_joint_state" in env.action


def test_reference_runtime_and_reset_receipt():
    from agentic_vla.benchmarks.robodojo_pi05_policy import ReferenceRuntimePolicy, validate_model_reset, RESET_SCHEMA
    class Policy:
        _sample_kwargs = {}
        def infer(self, payload):
            return {"actions": np.zeros((50, 14))}
    runtime = ReferenceRuntimePolicy(Policy())
    assert runtime.infer({"prompt": "sort"})["actions"].shape == (50, 14)
    with pytest.raises(ValueError, match="unadmitted"):
        runtime.infer({"prompt": "sort"}, num_steps=2)
    with pytest.raises(ValueError, match="reset-aware"):
        validate_model_reset({}, 0)
    receipt = {"model_reset_schema_version": RESET_SCHEMA, "policy_seed": 0,
               "observation_cleared": True, "inference_calls": 0, "generation": 1, "rng_sha256": "a" * 64}
    assert validate_model_reset(receipt, 0) is receipt


def test_stall_discards_remainder_then_requests_planner(tmp_path):
    port = PortFixture(stalled=True)
    original_observe = port.observe
    def observe():
        obs = original_observe()
        return dataclasses.replace(obs, frames={key: np.zeros_like(frame) for key, frame in obs.frames.items()})
    port.observe = observe
    calls = []
    def planner(request):
        calls.append(request)
        if len(calls) == 1:
            return continue_planner(request)
        return {"intent": "safe_stop", "rationale": "no visible response", "confidence": .9}
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=planner)
    report = episode.run()
    assert report["stopped"] and report["interrupted_chunks"] == 1
    assert 0 < report["control_steps"] < 50 and len(port.instructions) == 1
    assert len(calls) == 2


def test_task_only_capability_is_enforced_by_parser(tmp_path):
    from agentic_vla.runtime.agent import parse_high_level_agent_decision, build_high_level_agent_request
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner)
    episode._capture()
    context = episode._context("execution_review")
    assert "copy task_instruction exactly" in build_high_level_agent_request(context)["system_prompt"]
    with pytest.raises(ValueError, match="exact original"):
        parse_high_level_agent_decision({"intent": "vla_act", "rationale": "target visible",
                                         "confidence": .9, "subgoal": "sort red objects",
                                         "vla_instruction": "sort red objects"}, context)
    episode.session.close()


def test_compact_semantic_protocol_preserves_task_only_for_six_stages(tmp_path):
    from agentic_vla.runtime.agent import parse_high_level_agent_decision, build_high_level_agent_request
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner)
    episode._capture()
    context = episode._context("task_start")
    raw = {"confidence": .9, "stages": [
        {"subgoal": f"put item {i} inside basket", "expected_outcome": f"item {i} inside basket"}
        for i in range(6)]}
    decision = parse_high_level_agent_decision(raw, context)
    assert decision.vla_instruction == episode.instruction
    assert len(decision.proposed_plan) == 6
    prompt = build_high_level_agent_request(context)
    assert "1 to 8" in prompt["system_prompt"]
    assert "Do not add vla_instruction" in prompt["system_prompt"]
    with pytest.raises(ValueError, match="1 to 4"):
        parse_high_level_agent_decision(raw, dataclasses.replace(context, max_plan_stages=4))
    episode.session.close()


def test_long_instruction_is_not_forced_into_two_placement_stages(tmp_path):
    import json
    from agentic_vla.runtime.agent import parse_high_level_agent_decision, build_high_level_agent_request
    port = PortFixture()
    clauses = ["Place the mouse on the mouse pad", "push the keyboard into the frame",
               "put the figurine on the stand", "place the alarm clock on the drawer",
               "open the drawer", "put all remaining miscellaneous items inside"]
    port.instruction = ", ".join(clauses[:4]) + ", then " + " and ".join(clauses[4:]) + "."
    episode = SortingEpisode(config(tmp_path, "organize_table"), port, scripted_infer=continue_planner)
    episode._capture()
    context = episode._context("task_start")
    payload = json.loads(build_high_level_agent_request(context)["user_prompt"])
    assert payload["task_plan_contract"]["pattern"] == "explicit_action_sequence"
    assert payload["task_plan_contract"]["exact_stage_count"] == 6
    raw = {"confidence": .8, "stages": [
        {"subgoal": clause, "expected_outcome": clause} for clause in clauses]}
    with pytest.raises(ValueError, match="requires 6 stages"):
        parse_high_level_agent_decision({"confidence": .8, "stages": raw["stages"][:4]}, context)
    decision = parse_high_level_agent_decision(raw, context)
    assert len(decision.proposed_plan) == 6
    assert decision.vla_instruction == port.instruction
    episode.session.close()


def test_sorting_plan_cannot_drop_categories_or_swap_destinations(tmp_path):
    from agentic_vla.runtime.agent import parse_high_level_agent_decision, bind_cumulative_task_outcome, build_high_level_agent_request
    port = PortFixture()
    port.instruction = ("Put car objects into the left basket, watch objects into the middle basket, "
                        "and wooden_toy objects into the right basket, then reset the robot arm.")
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner)
    episode._capture()
    context = episode._context("task_start")
    prompt = build_high_level_agent_request(context)["system_prompt"]
    assert "ONE row per" in prompt and "ALL objects" in prompt
    assert "NOT a claim that" in prompt
    assert "Task-contract override" not in build_high_level_agent_request(
        dataclasses.replace(context, vla_instruction_mode="subgoal"))["system_prompt"]
    rows = [{"subgoal": f"Put {category} objects into the {destination} basket",
             "expected_outcome": f"All {category} objects are in the {destination} basket"}
            for category, destination in (("car", "left"), ("watch", "middle"), ("wooden_toy", "right"))]
    with pytest.raises(ValueError, match="omits category"):
        parse_high_level_agent_decision({"confidence": .8, "stages": rows[:1]}, context)
    decision = parse_high_level_agent_decision({"confidence": .8, "stages": rows}, context)
    assert len(decision.proposed_plan) == 3
    visual_guess = [dict(row) for row in rows]
    visual_guess[0]["expected_outcome"] = "All car objects are in the white basket"
    grounded = parse_high_level_agent_decision({"confidence": .8, "stages": visual_guess}, context)
    assert grounded.proposed_plan[0].expected_outcome == "All car objects are inside the left basket"
    bound = bind_cumulative_task_outcome(decision.proposed_plan, port.instruction)
    assert "reset the robot arm" in bound[-1].expected_outcome
    source_position = [dict(row) for row in rows]
    source_position[0]["subgoal"] = "Put right car objects into the left basket"
    assert parse_high_level_agent_decision({"confidence": .8, "stages": source_position}, context)
    singular = [dict(row) for row in rows]
    singular[0]["expected_outcome"] = "The car is in the left basket"
    with pytest.raises(ValueError, match="singular expected outcome"):
        parse_high_level_agent_decision({"confidence": .8, "stages": singular}, context)
    wrong = [dict(row) for row in rows]
    wrong[1]["subgoal"] = "Put watch objects into the right basket"
    with pytest.raises(ValueError, match="middle basket"):
        parse_high_level_agent_decision({"confidence": .8, "stages": wrong}, context)
    episode.session.close()


def test_advisory_review_rotates_without_changing_ledger(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
                             execution=SortingExecutionConfig(critic_authority="advisory", advisory_critic_schedule="periodic"))
    episode._capture()
    episode.session.task_plan.install([ProceduralStep(str(i), "vla_act", f"place item {i}", f"item {i} placed") for i in range(3)])
    before = episode.session.task_plan.to_dict()
    reviewed = []
    episode._verify = lambda expected, context: reviewed.append(expected)
    for step in (100, 200, 300, 400):
        episode.step = step
        episode._review_progress()
    assert reviewed == ["item 0 placed", "item 1 placed", "item 2 placed", "item 0 placed"]
    assert not episode.session.task_plan.completed
    assert episode.session.task_plan.to_dict() == before
    episode.session.close()


def test_deferred_review_binds_tool_target_without_fabricating_observation(tmp_path):
    checked = []
    def critic(request):
        checked.append(json.loads(request["user_prompt"])["expected_outcome"])
        return inconclusive_critic(request)
    episode = SortingEpisode(config(tmp_path), PortFixture(steps=500), scripted_infer=continue_planner,
        critic_infer=critic, execution=SortingExecutionConfig(feedback_refresh_enabled=True))
    episode._capture()
    episode.session.task_plan.install([
        ProceduralStep(str(i), "vla_act", f"place item {i}", f"item {i} placed") for i in range(2)])
    before = episode.session.task_plan.to_dict()
    for step in (100, 200):
        episode.step = step
        episode._review_progress()
    assert checked == [] and episode.last_visual_check is None
    assert episode.deferred_critic_checks == 2
    assert not episode._context().memory_records
    spec = next(s for s in episode._context().available_skill_specs if s["skill_id"] == "feedback_refresh")
    assert spec["verification_target"] == "item 1 placed"
    receipt = episode._run_skill("feedback_refresh", {}, episode._tool_context())
    assert checked == ["item 1 placed"]
    assert receipt.metadata["target_source"] == "scheduled_plan_goal"
    assert not receipt.metadata["semantic_recovery_confirmed"]
    assert episode.session.task_plan.to_dict() == before
    assert episode._scheduled_goal() == ""
    episode.session.close()


def test_invalid_advisory_schedule_rejected():
    with pytest.raises(ValueError, match="schedule"):
        SortingExecutionConfig(advisory_critic_schedule="never_stop")


def test_advisory_projection_whitelists_metadata_and_does_not_mutate_input():
    from agentic_vla.benchmarks.robodojo_sorting import advisory_check_for_planner
    source = {"timestep": 10, "expected_outcome": "goal", "observation_sha256": "test",
              "accepted": True, "report": {"status": "confirmed", "observed_outcome": "bad_claim"},
              "future_model_field": "bad_claim"}
    result = advisory_check_for_planner(source)
    assert "bad_claim" not in json.dumps(result)
    assert result["schema_accepted"] is True and result["report"]["status"] == "inconclusive"
    assert source["report"]["observed_outcome"] == "bad_claim"


def test_feedback_tool_discards_tail_and_reinfers_from_new_state(tmp_path):
    port = PortFixture()
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
        execution=SortingExecutionConfig(feedback_refresh_enabled=True))
    episode._capture()
    episode._run_skill("feedback_refresh", {}, episode._tool_context())
    assert port.steps == 10 and len(port.instructions) == 1
    assert np.allclose(port.state, .1)
    episode._vla_act(port.instruction, episode._tool_context())
    assert port.steps == 60 and len(port.instructions) == 2
    assert np.allclose(port.state, .6)
    assert port.instructions == [port.instruction] * 2
    episode.skill_calls = episode.config.harness.effective_physical_skill_budget
    assert not episode._context().available_skills
    with pytest.raises(RuntimeError, match="budget"):
        episode._run_skill("feedback_refresh", {}, episode._tool_context())
    episode.session.close()


@pytest.mark.parametrize("limit,budget,expected_calls", [(150, 12, 2), (10, 12, 1), (150, 1, 1)])
def test_feedback_immediate_check_binds_same_predicate_and_fresh_observation(tmp_path, limit, budget, expected_calls):
    port = PortFixture(steps=limit)
    requests = []
    def critic(request):
        requests.append((json.loads(request["user_prompt"]), request["frames"]["current_cam_high"].copy()))
        return {"status": "inconclusive", "observed_outcome": "fixture cannot establish outcome", "confidence": 0}
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner, critic_infer=critic,
        execution=SortingExecutionConfig(critic_authority="advisory", feedback_refresh_enabled=True, critic_call_budget=budget))
    episode._capture()
    predicate = "all red objects inside left basket"
    episode._verify(predicate, episode._tool_context())
    receipt = episode._run_skill("feedback_refresh", {}, episode._tool_context())
    assert port.steps == 10 and receipt.ended_timestep == 10
    assert len(requests) == expected_calls
    assert episode.feedback_postchecks == expected_calls - 1
    assert all(req["expected_outcome"] == predicate for req, _ in requests)
    if expected_calls == 2:
        assert requests[1][0]["timestep"] == 10
        assert np.all(requests[1][1] == 10)
    episode.session.close()


def test_compact_execution_tool_runs_through_planner_and_registry(tmp_path):
    port = PortFixture(steps=150)
    calls = 0
    def planner(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"confidence": .9, "stages": [{"subgoal": "put red objects into the left basket",
                "expected_outcome": "all red objects inside the left basket"}]}
        if calls > 2:
            return {"action": "continue", "confidence": .9, "reason": "fixture resumes after feedback"}
        return {"action": "feedback_refresh", "confidence": .9, "reason": "fixture requests new feedback"}
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=planner,
        critic_infer=lambda request: {"status": "inconclusive", "observed_outcome": "fixture", "confidence": 0},
        execution=SortingExecutionConfig(critic_authority="advisory", compact_task_only_review=True,
                                        feedback_refresh_enabled=True))
    report = episode.run()
    assert not report["stopped"] and report["control_steps"] == 150
    assert report["skill_calls"] == 1 and report["vla_calls"] == 4
    assert calls == 3 and report["tool_return_reviews"] == 1
    assert report["generated_vla_steps"] == 200
    assert report["discarded_vla_steps"] == 50
    assert report["generated_vla_steps"] == report["vla_control_steps"] + report["discarded_vla_steps"]
    assert report["task_plan"]["active_stage"] == "1" and not report["task_plan"]["completed"]


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_vla_budget_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="VLA call budget"):
        SortingExecutionConfig(vla_call_budget=value)


@pytest.mark.parametrize("agent_enabled", [False, True])
def test_vla_budget_stops_whole_loop_without_extra_semantic_work(tmp_path, agent_enabled):
    port = PortFixture(steps=500)
    planner_calls = []
    def planner(request):
        planner_calls.append(request)
        return continue_planner(request)
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=planner,
        execution=SortingExecutionConfig(agent_enabled=agent_enabled, vla_call_budget=2))
    report = episode.run()
    assert report["control_steps"] == 100
    assert report["vla_calls"] == report["vla_attempts"] == 2
    assert report["stopped"] and report["stop_reason"] == "VLA call budget exhausted"
    assert report["critic"]["calls"] == 0
    assert len(planner_calls) == int(agent_enabled)
    assert report["official_success"] is None
    assert report["runtime_budget"]["vla_calls_remaining"] == 0


def test_failed_vla_request_consumes_budget_not_actions(tmp_path):
    port = PortFixture()
    def fail(_):
        raise OSError("fixture transport failure")
    port.infer = fail
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
        execution=SortingExecutionConfig(vla_call_budget=1))
    episode._capture()
    with pytest.raises(OSError):
        episode._vla_act(port.instruction, episode._tool_context())
    assert episode.vla_attempts == 1 and episode.vla_calls == episode.generated_vla_steps == 0
    assert len(episode.inference_wall_ms) == 1
    with pytest.raises(RuntimeError, match="budget"):
        episode._vla_act(port.instruction, episode._tool_context())
    assert port.steps == 0
    episode.session.close()


def test_budget_visible_in_both_planner_protocols_and_executor_enforced(tmp_path):
    from agentic_vla.runtime.agent import build_high_level_agent_request
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        execution=SortingExecutionConfig(vla_call_budget=1, feedback_refresh_enabled=True,
                                        compact_task_only_review=True))
    episode._capture()
    for trigger in ("task_start", "execution_review"):
        payload = json.loads(build_high_level_agent_request(episode._context(trigger))["user_prompt"])
        assert payload["runtime_budget"]["vla_calls_remaining"] == 1
    episode._vla_act(episode.instruction, episode._tool_context())
    assert "feedback_refresh" not in episode._context().available_skills
    with pytest.raises(RuntimeError, match="budget"):
        episode._run_skill("feedback_refresh", {}, episode._tool_context())
    assert episode.skill_calls == 0
    episode.session.close()


def inconclusive_critic(_):
    return {"status": "inconclusive", "confidence": 0, "observed_outcome": "fixture only"}


def test_identical_check_reused_without_promoting_or_double_counting_memory(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=inconclusive_critic, execution=SortingExecutionConfig(critic_call_budget=1))
    episode._capture()
    first = episode._verify("goal", episode._tool_context())
    first.metadata["tamper"] = True
    second = episode._verify("goal", episode._tool_context())
    assert episode.critic.calls == episode.visual_check_cache_hits == 1
    assert "tamper" not in second.metadata
    assert second.status.value == "inconclusive" and not second.metadata["completion_authorized"]
    assert len(episode.memory.notes) == 1
    episode.session.close()


@pytest.mark.parametrize("change", ["predicate", "step", "state", "head", "wrist", "instruction"])
def test_visual_check_reuse_invalidated_by_any_evidence_change(tmp_path, change):
    port = PortFixture()
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
                             critic_infer=inconclusive_critic)
    episode._capture()
    episode._verify("goal", episode._tool_context())
    expected = "goal"
    if change == "predicate":
        expected = "another goal"
    elif change == "step":
        episode.step += 1
    elif change == "state":
        port.state[0] += .01
    elif change == "instruction":
        port.instruction = "different task"
        with pytest.raises(ValueError, match="instruction changed"):
            episode._verify(expected, episode._tool_context())
        assert episode.critic.calls == 1
        episode.session.close()
        return
    else:
        old_observe = port.observe
        def changed_frame():
            observation = old_observe()
            frames = {k: v.copy() for k, v in observation.frames.items()}
            frames["cam_high" if change == "head" else "cam_left_wrist"][0, 0, 0] = 123
            return dataclasses.replace(observation, frames=frames)
        port.observe = changed_frame
    episode._verify(expected, episode._tool_context())
    assert episode.critic.calls == 2 and episode.visual_check_cache_hits == 0
    episode.session.close()


@pytest.mark.parametrize("disable,fail", [(True, False), (False, True)])
def test_visual_check_reuse_opt_out_and_failed_request_retry(tmp_path, disable, fail):
    def critic(request):
        if fail:
            raise OSError("fixture unavailable")
        return inconclusive_critic(request)
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=critic, execution=SortingExecutionConfig(reuse_identical_visual_check=not disable))
    episode._capture()
    for _ in range(2):
        episode._verify("goal", episode._tool_context())
    assert episode.critic.calls == 2 and episode.visual_check_cache_hits == 0
    episode.session.close()


@pytest.mark.parametrize("has_plan", [False, True])
def test_feedback_does_not_use_stale_visual_target(tmp_path, has_plan):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=inconclusive_critic, execution=SortingExecutionConfig(feedback_refresh_enabled=True))
    episode._capture()
    episode._verify("old target", episode._tool_context())
    episode._run_skill("reobserve", {}, episode._tool_context())
    if has_plan:
        episode.session.task_plan.install([ProceduralStep("1", "vla_act", "place object", "active target")])
    receipt = episode._run_skill("feedback_refresh", {}, episode._tool_context())
    assert receipt.metadata["stale_before_check_excluded"]
    assert receipt.metadata["verification_target"] == ("active target" if has_plan else "")
    assert receipt.metadata["target_source"] == ("active_plan_goal" if has_plan else "none")
    assert episode.critic.calls == (2 if has_plan else 1)
    assert not receipt.metadata["semantic_recovery_confirmed"]
    episode.session.close()


def test_same_timestep_changed_state_excludes_visual_memory(tmp_path):
    port = PortFixture()
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
                             critic_infer=inconclusive_critic)
    episode._capture()
    episode._verify("goal", episode._tool_context())
    assert episode._context().memory_records
    port.state[0] += .01
    episode._capture()
    assert not episode._context().memory_records
    episode.session.close()


def test_controlled_tool_smoke_has_real_executor_contract_and_no_planner_claim(tmp_path):
    from agentic_vla.benchmarks.robodojo_pi05_episode import run_feedback_tool_smoke
    def unexpected_planner(_):
        raise AssertionError("controlled smoke must not invoke the Planner")
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=unexpected_planner,
        critic_infer=inconclusive_critic, execution=SortingExecutionConfig(
            feedback_refresh_enabled=True, vla_call_budget=2, critic_call_budget=2))
    report = run_feedback_tool_smoke(episode)
    assert report["vla_calls"] == 2 and report["control_steps"] == 60
    assert report["generated_vla_steps"] == 100 and report["discarded_vla_steps"] == 40
    assert report["critic"]["calls"] == 2 and report["visual_check_cache_hits"] == 1
    assert not report["autonomous_planner_used"] and report["official_success"] is None
    assert len(report["controlled_tool_receipts"]) == len(episode.memory.executions) == 2
    first, second = report["controlled_tool_receipts"]
    assert first["metadata"]["output_observation_sha256"] == second["metadata"]["input_observation_sha256"]
    assert "postcheck" not in first["metadata"]
    with pytest.raises(RuntimeError, match="single-use"):
        run_feedback_tool_smoke(episode)


@pytest.mark.parametrize("value", [0, 50, 10.5, True])
def test_feedback_prefix_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="prefix"):
        SortingExecutionConfig(feedback_prefix_steps=value)


def test_feedback_not_advertised_or_accepted_when_disabled(tmp_path):
    from agentic_vla.runtime.agent import parse_high_level_agent_decision
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
                             execution=SortingExecutionConfig(compact_task_only_review=True))
    episode._capture()
    with pytest.raises(ValueError, match="executor"):
        parse_high_level_agent_decision({"action": "feedback_refresh", "confidence": .9, "reason": "fixture"},
                                       episode._context("execution_review"))
    with pytest.raises(ValueError, match="registered"):
        episode._run_skill("feedback_refresh", {}, episode._tool_context())
    episode.session.close()


def test_feedback_cannot_infer_past_step_budget(tmp_path):
    port = PortFixture(steps=2000)
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=continue_planner,
                             execution=SortingExecutionConfig(feedback_refresh_enabled=True))
    episode._capture()
    episode.step = episode.task.max_steps
    with pytest.raises(RuntimeError, match="termination"):
        episode._run_skill("feedback_refresh", {}, episode._tool_context())
    assert not port.instructions and port.steps == 0
    episode.session.close()


def test_only_fresh_critic_evidence_reaches_planner_as_untrusted_memory(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=lambda request: {"status": "confirmed", "observed_outcome": "red objects in basket", "confidence": .99},
        execution=SortingExecutionConfig(critic_authority="advisory"))
    episode._capture()
    episode._verify("all red objects in basket", episode._tool_context())
    note = episode._context().memory_records[-1]
    assert note["authority"] == "current_unverified_visual_check"
    assert note["completion_authorized"] is False
    assert note["check"]["expected_outcome"] == "all red objects in basket"
    assert note["check"]["report"]["status"] == "inconclusive"
    assert note["check"]["report"]["metadata"]["untrusted_content_withheld"]
    episode.step += 1
    assert not episode._context().memory_records
    assert "red objects in basket" in str(episode.memory.records())
    episode.session.close()


@pytest.mark.parametrize("status", ["confirmed", "contradicted", "inconclusive"])
def test_current_unadmitted_critic_cannot_smuggle_prose_or_status_into_planner(tmp_path, status):
    from agentic_vla.runtime.agent import build_high_level_agent_request
    claim = "violet_marker_already_placed"
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=lambda _: {"status": status, "observed_outcome": claim, "confidence": .99},
        execution=SortingExecutionConfig(compact_task_only_review=True))
    episode._capture()
    episode._verify("requested target", episode._tool_context())
    request = build_high_level_agent_request(episode._context("execution_review"))
    assert claim not in request["user_prompt"]
    record = json.loads(request["user_prompt"])["memory_records"][-1]
    assert record["check"]["report"]["status"] == "inconclusive"
    assert record["check"]["report"]["confidence"] == 0
    assert not record["completion_authorized"]
    assert claim in str(episode.last_visual_check)
    assert claim in str(episode.memory.records())
    assert claim in (episode.config.workspace_path / "events.jsonl").read_text()
    assert episode.port.steps == episode.vla_calls == 0
    episode.session.close()


def test_unavailable_critic_error_is_not_forwarded_as_semantic_evidence(tmp_path):
    def unavailable(_):
        raise OSError("untrusted_backend_detail")
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
                             critic_infer=unavailable)
    episode._capture()
    episode._verify("goal", episode._tool_context())
    assert "untrusted_backend_detail" not in str(episode._context().memory_records)
    assert "untrusted_backend_detail" in str(episode.last_visual_check)
    episode.session.close()


@pytest.mark.parametrize("tool", ["reobserve", "feedback_refresh"])
def test_compact_tool_does_not_accept_motor_arguments(tmp_path, tool):
    from agentic_vla.runtime.agent import parse_high_level_agent_decision
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        execution=SortingExecutionConfig(compact_task_only_review=True, feedback_refresh_enabled=True))
    episode._capture()
    with pytest.raises(ValueError, match="permits only"):
        parse_high_level_agent_decision({"action": tool, "confidence": .9, "reason": "fixture",
            "skill_args": {"joint_targets": [0] * 14}}, episode._context("execution_review"))
    episode.session.close()


def test_memory_retrieves_relevant_older_evidence_without_promoting_it():
    memory = SortingWorkingMemory()
    memory.reset("episode")
    memory.remember_hypothesis(1, "mouse is outside mouse pad")
    memory.remember_hypothesis(2, "keyboard overlaps the frame")
    memory.remember_hypothesis(3, "drawer is closed")
    found = memory.retrieve("mouse pad", 1)
    assert found[0]["timestep"] == 1
    assert found[0]["authority"] == "unverified_model_note"
    assert not memory.retrieve("unicorn", 8)
    assert memory.retrieve("", 1)[0]["timestep"] == 3
    found[0]["text"] = "modified"
    assert "modified" not in str(memory.records())


@pytest.mark.parametrize("limit", [0, -1, 33, True, 1.5])
def test_memory_retrieval_rejects_unbounded_or_invalid_limit(limit):
    with pytest.raises(ValueError, match="limit"):
        SortingWorkingMemory().retrieve("mouse", limit)


def test_execution_memory_is_bounded_isolated_and_deep_copied():
    from agentic_vla.toolchain.contracts import PrimitiveOutcome
    memory = SortingWorkingMemory(max_executions=2)
    memory.reset("first")
    outcome = PrimitiveOutcome("a", "vla_act", "succeeded", "first", 0, 10,
        expected_outcome="mouse on pad", observed_outcome="actions executed",
        metadata={"semantic_outcome": "not_verified"})
    memory.remember_execution(outcome)
    memory.remember_execution(outcome)
    assert len(memory.records()) == 1
    copied = memory.records()[0]
    copied["receipt"]["metadata"]["semantic_outcome"] = "confirmed"
    assert memory.records()[0]["receipt"]["metadata"]["semantic_outcome"] == "not_verified"
    with pytest.raises(ValueError, match="episode"):
        memory.remember_execution(dataclasses.replace(outcome, episode_id="second"))
    for i in (2, 3):
        memory.remember_execution(dataclasses.replace(outcome, call_id=str(i), ended_timestep=10*i))
    assert len(memory.executions) == 2
    assert memory.retrieve("mouse", 1)[0]["receipt"]["call_id"] == "3"
    with pytest.raises(ValueError, match="backwards"):
        memory.remember_execution(dataclasses.replace(outcome, call_id="older"))
    memory.reset("second")
    assert not memory.records()


def test_memory_tool_uses_query_and_respects_memory_ablation(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
                             execution=SortingExecutionConfig(model_note_policy="legacy"))
    episode._capture()
    episode.memory.remember_hypothesis(0, "mouse remains outside pad")
    episode.memory.remember_hypothesis(1, "keyboard shifted")
    query = {"query": "mouse", "limit": 1}
    result = episode.session.tools.invoke("retrieve_memory", query, context=episode._tool_context())
    assert result.accepted
    assert "mouse remains" in str(result.output)
    assert "keyboard shifted" not in str(result.output)
    episode.execution = dataclasses.replace(episode.execution, memory_enabled=False)
    assert not episode._context().memory_records
    result = episode.session.tools.invoke("retrieve_memory", query, context=episode._tool_context())
    assert result.accepted and result.output["records"] == []
    episode.session.close()


@pytest.mark.parametrize("tool,return_step", [("feedback_refresh", 110), ("reobserve", 101)])
def test_tool_result_returns_to_planner_before_next_vla_call(tmp_path, tool, return_step):
    port = PortFixture(steps=300)
    requests = []
    critic_requests = []
    def planner(request):
        payload = json.loads(request["user_prompt"])
        requests.append(payload)
        if len(requests) == 1:
            return {"confidence": .9, "stages": [{"subgoal": "put red objects into the left basket",
                "expected_outcome": "all red objects inside the left basket"}]}
        if len(requests) == 2:
            return {"action": tool, "confidence": .9, "reason": "test a bounded execution change"}
        assert port.steps == return_step
        assert len(port.instructions) == (3 if tool == "feedback_refresh" else 2)
        return {"action": "safe_stop", "confidence": .9, "reason": "fixture stops after inspecting receipt"}
    def critic(request):
        critic_requests.append(request)
        return {"status": "inconclusive", "observed_outcome": "fixture unknown", "confidence": 0}
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=planner, critic_infer=critic,
        execution=SortingExecutionConfig(critic_authority="advisory", compact_task_only_review=True,
                                        feedback_refresh_enabled=True))
    result = episode.run()
    assert result["control_steps"] == return_step and result["stopped"]
    assert result["tool_return_reviews"] == 1
    assert len(critic_requests) == (1 if tool == "feedback_refresh" else 0)
    receipt = requests[-1]["last_primitive"]
    assert receipt["ended_timestep"] == return_step
    assert receipt["metadata"]["skill_id"] == tool
    assert receipt["metadata"]["semantic_outcome"] == "not_verified"
    assert not result["task_plan"]["completed"]
    if tool == "feedback_refresh":
        assert receipt["metadata"]["executed_steps"] == 10
        assert receipt["metadata"]["discarded_steps"] == 40
        assert "postcheck" not in receipt["metadata"]
        assert requests[-1]["memory_records"][-1]["authority"] == "current_unverified_visual_check"
        assert receipt["metadata"]["semantic_recovery_confirmed"] is False


def test_no_posttool_planner_or_critic_after_terminal_action(tmp_path):
    port = PortFixture(steps=110)
    planner_calls = []
    def planner(request):
        planner_calls.append(request)
        if len(planner_calls) == 1:
            return {"confidence": .9, "stages": [{"subgoal": "put red objects into the left basket",
                "expected_outcome": "all red objects inside the left basket"}]}
        return {"action": "feedback_refresh", "confidence": .9, "reason": "fixture"}
    episode = SortingEpisode(config(tmp_path), port, scripted_infer=planner,
        critic_infer=lambda _: {"status": "inconclusive", "observed_outcome": "fixture", "confidence": 0},
        execution=SortingExecutionConfig(critic_authority="advisory", compact_task_only_review=True,
                                        feedback_refresh_enabled=True))
    result = episode.run()
    assert len(planner_calls) == 2 and result["critic"]["calls"] == 0
    assert result["tool_return_reviews"] == 0 and result["control_steps"] == 110


def test_note_quarantine_rejects_unknown_policy():
    with pytest.raises(ValueError, match="model note policy"):
        SortingExecutionConfig(model_note_policy="trust_everything")


def test_memory_filters_hypotheses_before_top_k():
    from agentic_vla.toolchain.contracts import PrimitiveOutcome
    memory = SortingWorkingMemory()
    memory.reset("episode")
    memory.remember_execution(PrimitiveOutcome("a", "vla_act", "succeeded", "episode", 0, 10,
        expected_outcome="mouse on pad", observed_outcome="actions executed"))
    for step in range(11, 20):
        memory.remember_hypothesis(step, "mouse on pad confirmed, everything complete")
    records = memory.retrieve("mouse pad", 1, include_hypotheses=False)
    assert len(records) == 1 and records[0]["receipt"]["call_id"] == "a"
    assert len(memory.notes) == 9


@pytest.mark.parametrize("policy", ["quarantine", "legacy"])
def test_model_claims_cannot_leak_through_receipt_or_retrieval(tmp_path, policy):
    from agentic_vla.toolchain.contracts import PrimitiveOutcome
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        execution=SortingExecutionConfig(model_note_policy=policy))
    episode._capture()
    marker = "FALSE_COMPLETION_SENTINEL"
    episode.memory.remember_hypothesis(0, "mouse " + marker)
    outcome = PrimitiveOutcome("feedback", "feedback_refresh", "succeeded", "fixture", 0, 10,
        expected_outcome="mouse on pad", observed_outcome="action prefix executed",
        metadata={"executed_steps": 10, "postcheck_state": "checked", "postcheck": {
            "status": "inconclusive", "metadata": {"raw_report": {"observed_outcome": marker}}}})
    episode.memory.remember_execution(outcome)
    episode.last_primitive = outcome.to_planner_dict()
    episode.step = 10
    context = episode._context("tool_return")
    result = episode.session.tools.invoke("retrieve_memory", {"query": "mouse", "limit": 8},
                                          context=episode._tool_context())
    assert result.accepted
    decision_inputs = str((context.memory_records, context.last_primitive, result.output))
    assert (marker in decision_inputs) is (policy == "legacy")
    assert marker in str(episode.memory.records())
    assert marker in str(episode.last_primitive)
    assert context.last_primitive["metadata"]["executed_steps"] == 10
    assert context.last_primitive["metadata"]["postcheck_state"] == "checked"
    episode.session.close()


def test_reobserve_invalidates_previous_visual_check_without_erasing_audit(tmp_path):
    episode = SortingEpisode(config(tmp_path), PortFixture(), scripted_infer=continue_planner,
        critic_infer=lambda _: {"status": "confirmed", "observed_outcome": "old claim", "confidence": .99},
        execution=SortingExecutionConfig(critic_authority="advisory"))
    episode._capture()
    episode._verify("mouse on pad", episode._tool_context())
    assert "old claim" not in str(episode._context().memory_records)
    assert len(episode._context().memory_records) == 1
    episode._run_skill("reobserve", {}, episode._tool_context())
    assert not episode._context().memory_records
    assert "old claim" not in str(episode._context().memory_records)
    assert "old claim" in str(episode.memory.records())
    episode.session.close()


def test_quarantined_planner_claim_remains_in_transcript_not_next_prompt(tmp_path):
    requests = []
    def planner(request):
        payload = json.loads(request["user_prompt"])
        requests.append(payload)
        if len(requests) == 1:
            return {"confidence": .9, "stages": [{"subgoal": "put red objects into the left basket",
                "expected_outcome": "all red objects inside the left basket"}]}
        if len(requests) == 2:
            return {"action": "reobserve", "confidence": .9, "reason": "fixture",
                    "memory_note": "FALSE_SUCCESS_FROM_PLANNER"}
        assert "FALSE_SUCCESS_FROM_PLANNER" not in json.dumps(payload)
        return {"action": "safe_stop", "confidence": .9, "reason": "fixture ends"}
    episode = SortingEpisode(config(tmp_path), PortFixture(steps=200), scripted_infer=planner,
        critic_infer=lambda _: {"status": "inconclusive", "observed_outcome": "unknown", "confidence": 0},
        execution=SortingExecutionConfig(critic_authority="advisory", compact_task_only_review=True))
    summary = episode.run()
    assert summary["control_steps"] == 101
    assert "FALSE_SUCCESS_FROM_PLANNER" in str(episode.memory.records())
    transcripts = list(tmp_path.rglob("transcript.jsonl"))
    assert len(transcripts) == 1 and "FALSE_SUCCESS_FROM_PLANNER" in transcripts[0].read_text()

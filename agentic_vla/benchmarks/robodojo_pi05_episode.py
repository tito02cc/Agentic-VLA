"""Official pi0.5 episode executor using the existing Agent session and tools.

This module runs on the simulator/client side. The model server owns OpenPI/JAX;
the Planner sees only public instruction, RGB, proprioception, and episode memory.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from typing import Protocol

import numpy as np

from agentic_vla.configuration import CarveRunConfig, PlannerExecutionMode
from agentic_vla.runtime.agent import build_vision_planner, compose_grounded_vla_instruction
from agentic_vla.runtime import HighLevelAgentContext, ExecutionRiskMonitor, MonitorConfig
from agentic_vla.session import CarveAgentSession
from agentic_vla.toolchain import (
    EmbodiedToolBindings, PrimitiveExecutionReport, ToolExecutionContext,
    VerificationReport, VerificationStatus,
)
from agentic_vla.toolchain.verifier import GuardedVisualVerifier, VisualVerificationContext

from .robodojo_sorting import (
    SORTING_TASKS, SortingObservation, SortingWorkingMemory, execution_only_receipt,
    advisory_check_for_planner, joint_target_feedback,
)


class SortingRobotPort(Protocol):
    def observe(self) -> SortingObservation: ...
    def infer(self, instruction: str) -> np.ndarray: ...
    def execute(self, absolute_joint_target: np.ndarray) -> None: ...
    def done(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class SortingExecutionConfig:
    agent_enabled: bool = True
    semantic_interval_steps: int = 100
    critic_call_budget: int = 12
    max_consecutive_rejections: int = 2
    subgoal_conditioning_admitted: bool = False
    memory_enabled: bool = True
    critic_authority: str = "advisory"
    critic_camera_mode: str = "all"
    critic_protocol: str = "legacy"
    compact_task_only_review: bool = False
    feedback_refresh_enabled: bool = False
    feedback_prefix_steps: int = 10
    model_note_policy: str = "quarantine"
    vla_call_budget: int | None = None
    reuse_identical_visual_check: bool = True
    advisory_critic_schedule: str = "tool_only"
    proposal_review_enabled: bool = False

    def __post_init__(self):
        if min(self.semantic_interval_steps, self.critic_call_budget,
               self.max_consecutive_rejections) < 1:
            raise ValueError("execution intervals and budgets must be positive")
        if self.critic_authority not in {"authoritative", "advisory"}:
            raise ValueError("invalid Critic authority")
        if self.critic_camera_mode not in {"all", "head"}:
            raise ValueError("invalid Critic camera mode")
        if self.critic_protocol not in {"legacy", "evidence"}:
            raise ValueError("invalid Critic protocol")
        if self.critic_protocol == "evidence" and self.critic_authority != "advisory":
            raise ValueError("evidence Critic is not admitted for authoritative completion")
        if type(self.feedback_prefix_steps) is not int or not 1 <= self.feedback_prefix_steps < 50:
            raise ValueError("feedback prefix must be an integer between 1 and 49")
        if self.model_note_policy not in {"quarantine", "legacy"}:
            raise ValueError("model note policy must be quarantine or legacy")
        if self.vla_call_budget is not None and (
                type(self.vla_call_budget) is not int or self.vla_call_budget < 1):
            raise ValueError("VLA call budget must be a positive integer or None")
        if type(self.reuse_identical_visual_check) is not bool:
            raise ValueError("visual check reuse must be boolean")
        if self.advisory_critic_schedule not in {"tool_only", "periodic"}:
            raise ValueError("invalid advisory Critic schedule")


class SortingEpisode:
    """One fresh instance per episode; no outstanding chunk survives a tool call.

    Task success is NOT determined by this class. A Critic can update a symbolic
    checklist, but only the unchanged benchmark evaluator scores the rollout.
    """

    def __init__(self, config: CarveRunConfig, port: SortingRobotPort, *,
                 execution: SortingExecutionConfig | None = None,
                 scripted_infer=None, critic_infer=None):
        self.task = SORTING_TASKS[config.benchmark.task_id]
        if config.benchmark.environment_id != "robodojo" or config.benchmark.max_steps != self.task.max_steps:
            raise ValueError("use the official task id and control-step budget")
        if config.optimize.enabled:
            raise ValueError("optimized JAX profiles need separate admission; reference execution only")
        if config.planner.mode is PlannerExecutionMode.EXTERNAL_CODING_AGENT:
            raise ValueError("this autonomous loop needs embedded_vlm; external agents use the session API")
        if config.planner.mode is PlannerExecutionMode.EMBEDDED_VLM and config.planner.model.startswith("CONFIGURE_"):
            raise ValueError("configure and admit an actual served VLM model before launching")
        if config.harness.task_start_policy.value != "startup_wait":
            raise ValueError("the sorting driver requires startup_wait")
        if config.harness.primitive_boundary_policy.value != "event_only":
            raise ValueError("this driver owns semantic scheduling; use event_only boundaries")
        self.config, self.port = config, port
        self.execution = execution or SortingExecutionConfig()
        if (config.planner.mode is not PlannerExecutionMode.SCRIPTED
                and self.execution.critic_authority != "advisory"):
            raise ValueError("no authoritative Critic profile is admitted for this RoboDojo driver")
        self.step = 0
        self.stopped = False
        self.ran = False
        self.stop_reason = ""
        self.observation = None
        self.instruction = ""
        self.risk = None
        self.rejections = 0
        self.vla_calls = 0
        self.vla_attempts = 0
        self.generated_vla_steps = 0
        self.visual_check_cache_hits = 0
        self._visual_check_cache = None
        self.executed_vla_steps = 0
        self.interrupted_chunks = 0
        self.skill_calls = 0
        self.last_review_step = -1
        self.advisory_review_index = 0
        self.scheduled_visual_goal = None
        self.deferred_critic_checks = 0
        self.last_visual_check = None
        self.last_primitive = {}
        self.feedback_postchecks = 0
        self.tool_return_reviews = 0
        self.inference_wall_ms = []
        self._pending_vla_proposal = None
        self.memory = SortingWorkingMemory()
        self.memory.reset(config.run_id)
        # Absolute-joint targets must be converted to errors before monitor.update.
        # A native horizon of 50 is not automatically a stale-action violation.
        self.monitor = ExecutionRiskMonitor(MonitorConfig(stale_action_steps=51, warmup_steps=8))
        provider = config.planner.provider_config()
        infer = critic_infer or scripted_infer
        if infer is None and provider is not None:
            infer = build_vision_planner(provider)
        if infer is None:
            raise ValueError("a real Critic backend (or explicit test fixture) is required")
        self.critic = GuardedVisualVerifier(infer, protocol=self.execution.critic_protocol)
        self.session = CarveAgentSession(
            config, scripted_infer=scripted_infer,
            bindings=EmbodiedToolBindings(
                observe=lambda ctx: {"timestep": self.step, "cameras": list(self.observation.frames)},
                retrieve_memory=lambda query, limit, ctx: {
                    "records": list(self._retrieve_execution_memory(query, limit))},
                vla_act=self._vla_act,
                run_skill=self._run_skill,
                verify=self._verify,
                safe_hold=self._stop,
            ),
        )

    def _capture(self):
        self.observation = self.port.observe()
        if not self.instruction:
            self.instruction = self.observation.instruction
        elif self.observation.instruction != self.instruction:
            raise ValueError("task instruction changed within an episode")
        if self.execution.memory_enabled and (not self.memory.frames or self.memory.frames[-1][0] != self.step):
            self.memory.capture(self.step, self.observation)

    def _planner_receipt(self, receipt):
        if self.execution.model_note_policy == "quarantine":
            return execution_only_receipt(receipt)
        return copy.deepcopy(receipt)

    def _retrieve_execution_memory(self, query, limit):
        if not self.execution.memory_enabled:
            return ()
        records = self.memory.retrieve(query, limit,
            include_hypotheses=self.execution.model_note_policy == "legacy")
        for record in records:
            if "receipt" in record:
                record["receipt"] = self._planner_receipt(record["receipt"])
        return records

    def _decision_memory(self, query):
        records = self._retrieve_execution_memory(query, 8)
        if (self.execution.memory_enabled and self.execution.model_note_policy == "quarantine"
                and self.last_visual_check is not None
                and self.last_visual_check["timestep"] == self.step
                and self.last_visual_check.get("observation_sha256") == self.observation.fingerprint()):
            # Freshness does not grant semantic authority. Keep the model's prose
            # in the audit log, including when its status is merely inconclusive.
            check = copy.deepcopy(self.last_visual_check)
            if self.execution.critic_authority == "advisory":
                check = advisory_check_for_planner(self.last_visual_check)
            records = (*records, {
                "timestep": self.step, "authority": "current_unverified_visual_check",
                "completion_authorized": self.execution.critic_authority == "authoritative",
                "check": check,
            })
        return records

    def _context(self, trigger="primitive_boundary"):
        active = self.session.task_plan.active
        query = "" if active is None else active.step.expected_outcome
        if self.last_visual_check is not None:
            query += " " + self.last_visual_check["expected_outcome"]
        specs = [{"skill_id": "reobserve", "description": "Hold current joint targets for one control step and observe again; no grasp correction.",
                  "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                  "cost": {"vla_calls": 0, "control_steps": 1},
                  "postcondition": "one new camera observation, not task completion"}]
        if self.execution.feedback_refresh_enabled and not self._vla_budget_exhausted():
            specs.append({"skill_id": "feedback_refresh",
                "description": f"Infer from current images, execute only {self.execution.feedback_prefix_steps} of 50 actions and discard the rest. Next inference uses new images. Costs one VLA call; no guaranteed grasp recovery.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "cost": {"vla_calls": 1, "control_steps_max": self.execution.feedback_prefix_steps,
                         "critic_calls_max": 1},
                "verification_target": self._scheduled_goal() or (active.step.expected_outcome if active else ""),
                "postcondition": "short VLA prefix executed, not task completion"})
        if self.skill_calls >= self.config.harness.effective_physical_skill_budget:
            specs = []
        return HighLevelAgentContext(
            task_instruction=self.instruction, trigger=trigger,
            episode_id=self.config.run_id, timestep=self.step,
            frames=(self.memory.planner_frames(self.observation) if self.execution.memory_enabled
                    else {f"current_{k}": v for k, v in self.observation.frames.items()}),
            robot_state=tuple(float(x) for x in self.observation.state),
            risk={} if self.risk is None else self.risk.to_dict(),
            current_subgoal="" if active is None else active.step.subgoal,
            memory=(self.task.focus,
                    "last_primitive reports what the executor actually did, not semantic success. "
                    "After tool_return inspect its receipt and current images before choosing the next action. "
                    "A successful call or unknown postcheck does not prove recovery. "
                    "Memory retrieval is lexical and episode-local, not an authoritative scene state.",
                    ("Historical model notes are quarantined in audit logs and excluded from retrieval. "
                     "Unadmitted Critic prose is audit-only even at the current timestep. "
                     "Its absence is not evidence of failure; task-plan outcomes are goals, not observations."
                     if self.execution.model_note_policy == "quarantine" else
                     "Legacy ablation: historical unverified model notes are included."),
                    "History is past evidence, not current state. Model notes are hypotheses. " +
                    ("Critic reports are advisory only and cannot confirm task stages; "
                     "continue, reobserve or stop using current evidence without claiming completion."
                     if self.execution.critic_authority == "advisory" else
                     "Only Critic evidence updates completion; the benchmark scores success."),
                    "Subgoal-conditioned VLA is " + ("enabled for explicit capability testing."
                    if self.execution.subgoal_conditioning_admitted else
                    "NOT admitted: vla_instruction must equal the original task instruction.")),
            memory_records=self._decision_memory(query),
            last_primitive=self._planner_receipt(self.last_primitive),
            task_plan=self.session.task_plan.to_dict(),
            available_skills=tuple(spec["skill_id"] for spec in specs),
            available_skill_specs=tuple(specs),
            remaining_retries=max(0, self.config.harness.retry_budget - self.rejections),
            remaining_recoveries=max(0, self.config.harness.effective_physical_skill_budget - self.skill_calls),
            deployment_profile_id=self.config.vla.deployment_profile_id,
            vla_instruction_mode="subgoal" if self.execution.subgoal_conditioning_admitted else "task_only",
            compact_task_only_review=self.execution.compact_task_only_review,
            max_plan_stages=8,
            runtime_budget=self._runtime_budget(),
        )

    def _vla_budget_exhausted(self):
        limit = self.execution.vla_call_budget
        return limit is not None and self.vla_attempts >= limit

    def _scheduled_goal(self):
        goal = self.scheduled_visual_goal
        return goal["expected_outcome"] if goal is not None and goal["timestep"] == self.step else ""

    def _runtime_budget(self):
        limit = self.execution.vla_call_budget
        return {
            "control_steps_remaining": max(0, self.task.max_steps - self.step),
            "vla_attempts": self.vla_attempts,
            "vla_calls_remaining": None if limit is None else max(0, limit - self.vla_attempts),
            "critic_calls_remaining": max(0, self.execution.critic_call_budget - self.critic.calls),
            "physical_tools_remaining": max(0, self.config.harness.effective_physical_skill_budget - self.skill_calls),
            "vla_instruction_mode": "subgoal" if self.execution.subgoal_conditioning_admitted else "task_only",
        }

    def _tool_context(self):
        return ToolExecutionContext(
            episode_id=self.config.run_id, timestep=self.step, at_safe_boundary=True,
            allowed_tools=self.config.harness.allowed_tools,
            deployment_profile_id=self.config.vla.deployment_profile_id,
        )

    def _vla_act(self, instruction, context, *, execution_prefix=50):
        if type(execution_prefix) is not int or not 1 <= execution_prefix <= 50:
            raise ValueError("invalid execution prefix")
        proposal = self._prepare_vla_proposal(instruction)
        return self._execute_vla_proposal(proposal.proposal_id, execution_prefix=execution_prefix)

    def _discard_vla_proposal(self, reason):
        proposal, self._pending_vla_proposal = self._pending_vla_proposal, None
        if proposal is not None:
            self.session.workspace.append_event("vla_proposal_discarded", {
                **proposal.identity(), "reason": reason}, source="official_pi05_executor")

    def _prepare_vla_proposal(self, instruction):
        from .robodojo_action_proposal import VlaActionProposal
        if self._pending_vla_proposal is not None:
            raise RuntimeError("consume or discard pending VLA proposal before new inference")
        if not self.execution.subgoal_conditioning_admitted and instruction != self.instruction:
            raise ValueError("subgoal conditioning has not passed checkpoint capability admission")
        if self.stopped or self.port.done() or self.step >= self.task.max_steps:
            raise RuntimeError("cannot execute VLA after episode termination")
        if self._vla_budget_exhausted():
            raise RuntimeError("VLA call budget exhausted")
        # Always observe again at a new inference boundary. Never replay cached actions.
        self._capture()
        input_fingerprint = self.observation.fingerprint()
        start = time.perf_counter()
        self.vla_attempts += 1
        try:
            actions = np.asarray(self.port.infer(instruction))
        finally:
            wall_ms = (time.perf_counter() - start) * 1000
            self.inference_wall_ms.append(wall_ms)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("official reference policy must return finite [50, 14] joint targets")
        proposal = VlaActionProposal.create(actions, timestep=self.step,
            observation_sha256=input_fingerprint, instruction=instruction, inference_wall_ms=wall_ms)
        self.vla_calls += 1
        self.generated_vla_steps += len(actions)
        self._pending_vla_proposal = proposal
        self.session.workspace.append_event("vla_proposal_prepared", proposal.identity(), source="official_pi05_executor")
        return proposal

    def _execute_vla_proposal(self, proposal_id, *, execution_prefix):
        if type(execution_prefix) is not int or not 1 <= execution_prefix <= 50:
            raise ValueError("invalid execution prefix")
        proposal = self._pending_vla_proposal
        if proposal is None or proposal.proposal_id != proposal_id:
            raise ValueError("unknown or consumed VLA proposal")
        self._capture()
        if (self.stopped or self.port.done() or self.step >= self.task.max_steps
                or proposal.timestep != self.step or proposal.observation_sha256 != self.observation.fingerprint()):
            self._discard_vla_proposal("stale_or_terminated")
            raise ValueError("stale or terminated VLA proposal")
        # Consume before dispatch; neither an exception nor a new observation can replay it.
        self._pending_vla_proposal = None
        actions = proposal.actions()
        instruction, input_fingerprint, wall_ms = proposal.instruction, proposal.observation_sha256, proposal.inference_wall_ms
        started = self.step
        interrupted = False
        for age, action in enumerate(actions[:execution_prefix]):
            if self.port.done() or self.step >= self.task.max_steps:
                break
            previous_state = self.observation.state.copy()
            self.port.execute(action)
            self.step += 1
            self.executed_vla_steps += 1
            self.observation = self.port.observe()
            self.risk = self.monitor.update(
                proprio=self.observation.state,
                commanded_action=action - previous_state,
                frame=self.observation.frames["cam_high"], action_age_steps=age,
            )
            if self.execution.agent_enabled and self.risk.event is not None:
                interrupted = True
                self.interrupted_chunks += 1
                break
        self._capture()
        executed = self.step - started
        feedback = (joint_target_feedback(actions[executed - 1], self.observation.state)
                    if executed else None)
        self.session.workspace.append_event("pi05_chunk_executed", {
            "proposal_id": proposal.proposal_id, "action_sha256": proposal.identity()["action_sha256"],
            "started_timestep": started, "ended_timestep": self.step,
            "inference_wall_ms": wall_ms, "generated_actions": len(actions),
            "executed_actions": self.step - started,
            "discarded_actions": len(actions) - (self.step - started),
            "execution_prefix": execution_prefix,
            "monitor_event": None if self.risk is None else self.risk.event,
            "instruction_changed": instruction != self.instruction,
            "joint_target_feedback": feedback,
            "final_command": actions[executed - 1].tolist() if executed else None,
            "final_observed_state": self.observation.state.tolist(),
            "input_observation_sha256": input_fingerprint,
            "output_observation_sha256": self.observation.fingerprint(),
        }, source="official_pi05_executor")
        return PrimitiveExecutionReport(
            status="interrupted" if interrupted else "succeeded",
            started_timestep=started, ended_timestep=self.step,
            observed_outcome="action chunk executed; semantic outcome not yet verified",
            requires_semantic_check=False,
            metadata={"execution_prefix": execution_prefix,
                      "generated_steps": len(actions), "executed_steps": self.step - started,
                      "discarded_steps": len(actions) - (self.step - started),
                      "instruction_changed": instruction != self.instruction,
                      "inference_wall_ms": wall_ms,
                      "input_observation_sha256": input_fingerprint,
                      "output_observation_sha256": self.observation.fingerprint(),
                      "joint_target_feedback": feedback,
                      "semantic_outcome": "not_verified"},
        )

    def _run_skill(self, skill_id, args, context):
        if self.skill_calls >= self.config.harness.effective_physical_skill_budget:
            raise RuntimeError("execution tool budget exhausted")
        if self.stopped or self.port.done() or self.step >= self.task.max_steps:
            raise RuntimeError("cannot run execution tools after termination")
        if skill_id in {"execute_prefix_10", "execute_prefix_50"} and self.execution.proposal_review_enabled and not args:
            proposal = self._pending_vla_proposal
            if proposal is None:
                raise RuntimeError("no pending VLA proposal")
            self.skill_calls += 1
            return self._execute_vla_proposal(proposal.proposal_id, execution_prefix=int(skill_id.rsplit("_", 1)[1]))
        if skill_id == "feedback_refresh" and self.execution.feedback_refresh_enabled and not args:
            if self._vla_budget_exhausted():
                raise RuntimeError("VLA call budget exhausted")
            self._capture()
            candidate = self.last_visual_check
            before = (copy.deepcopy(candidate) if candidate is not None
                      and candidate["timestep"] == self.step
                      and candidate.get("observation_sha256") == self.observation.fingerprint()
                      else None)
            active = self.session.task_plan.active
            scheduled = self._scheduled_goal()
            expected = (before["expected_outcome"] if before is not None else
                        scheduled or (active.step.expected_outcome if active is not None else ""))
            target_source = ("current_advisory_check" if before is not None else
                             "scheduled_plan_goal" if scheduled else
                             "active_plan_goal" if active is not None else "none")
            self.skill_calls += 1
            receipt = self._vla_act(self.instruction, context,
                                    execution_prefix=self.execution.feedback_prefix_steps)
            report = None
            reason = "no_bound_visual_predicate"
            if self.port.done() or self.step >= self.task.max_steps:
                reason = "episode_terminated"
            elif expected:
                calls_before = self.critic.calls
                report = self._verify(expected, self._tool_context())
                self.feedback_postchecks += self.critic.calls - calls_before
                reason = "checked" if self.critic.calls > calls_before else "critic_budget_exhausted"
            self.session.workspace.append_event("feedback_refresh_postcheck", {
                "timestep": self.step, "expected_outcome": expected,
                "before_check": before, "postcheck_state": reason,
                "target_source": target_source,
                "stale_before_check_excluded": candidate is not None and before is None,
                "report": None if report is None else report.to_dict(),
                "semantic_recovery_confirmed": False,
            }, source="official_pi05_executor")
            return dataclasses.replace(receipt, metadata={
                **receipt.metadata, "skill_id": skill_id,
                "verification_target": expected, "postcheck_state": reason,
                "target_source": target_source,
                "stale_before_check_excluded": candidate is not None and before is None,
                "postcheck": None if report is None else report.to_dict(),
                "semantic_recovery_confirmed": False,
            })
        if skill_id != "reobserve" or args:
            raise ValueError("only the registered argument-free reobserve skill is executable")
        self._capture()
        started = self.step
        hold_target = self.observation.state.copy()
        self.port.execute(hold_target)
        self.skill_calls += 1
        self.step += 1
        self._capture()
        self.monitor.reset()
        self.risk = None
        return PrimitiveExecutionReport(
            status="succeeded", started_timestep=started, ended_timestep=self.step,
            observed_outcome="new observation acquired while holding joint targets",
            requires_semantic_check=False,
            metadata={"skill_id": "reobserve", "semantic_outcome": "not_verified",
                      "joint_target_feedback": joint_target_feedback(hold_target, self.observation.state)},
        )

    def _verify(self, expected, context):
        self._capture()
        fingerprint = self.observation.fingerprint()
        key = (self.step, expected, fingerprint)
        if (self.execution.reuse_identical_visual_check and self._visual_check_cache is not None
                and self._visual_check_cache[0] == key):
            self.visual_check_cache_hits += 1
            self.session.workspace.append_event("visual_critic_reused", {
                "timestep": self.step, "expected_outcome": expected,
                "observation_sha256": fingerprint, "new_model_call": False,
                "completion_authorized": self.execution.critic_authority == "authoritative",
            }, source="visual_critic")
            return copy.deepcopy(self._visual_check_cache[1])
        if self.critic.calls >= self.execution.critic_call_budget:
            return VerificationReport(status="inconclusive", confidence=0,
                                      observed_outcome="Critic call budget exhausted")
        result = self.critic.verify(VisualVerificationContext(
            task_instruction=self.instruction, expected_outcome=expected,
            frames={f"current_{k}": v for k, v in self.observation.frames.items()
                    if self.execution.critic_camera_mode == "all" or k == "cam_high"},
            timestep=self.step,
        ))
        self.last_visual_check = {"timestep": self.step, "expected_outcome": expected,
                                  "observation_sha256": fingerprint,
                                  "report": result.report.to_dict(), "accepted": result.accepted}
        self.session.workspace.append_event("visual_critic", {
            "timestep": self.step, "expected_outcome": expected,
            "report": result.report.to_dict(), "accepted": result.accepted,
            "error": result.error, "elapsed_ms": result.elapsed_ms,
            "authority": self.execution.critic_authority,
            "camera_mode": self.execution.critic_camera_mode,
            "protocol": self.execution.critic_protocol,
            "raw_output": result.raw_output,
            "observation_sha256": fingerprint,
        }, source="visual_critic")
        if self.execution.memory_enabled:
            self.memory.remember_hypothesis(self.step, json.dumps({
                "source": "fallible_visual_critic", "expected_outcome": expected,
                "report": result.report.to_dict(), "accepted_schema": result.accepted,
                "completion_authorized": self.execution.critic_authority == "authoritative",
            }, ensure_ascii=True))
        if self.execution.critic_authority == "advisory":
            report = VerificationReport(status="inconclusive", confidence=0,
                observed_outcome="Unadmitted advisory Critic cannot authorize ledger transitions",
                metadata={"raw_report": result.report.to_dict(), "completion_authorized": False})
        else:
            report = result.report
        # Failed requests are not evidence and must remain retryable within budget.
        self._visual_check_cache = (key, copy.deepcopy(report)) if result.accepted else None
        return report

    def _stop(self, reason, context):
        self._discard_vla_proposal("stop: " + reason)
        self.stopped, self.stop_reason = True, reason
        return {"holding": True, "reason": reason,
                "scope": "paused synchronous simulator; no hardware safety guarantee"}

    def _review_progress(self):
        if self.step <= self.last_review_step:
            return
        self.last_review_step = self.step
        if self.execution.critic_authority == "advisory":
            # Advisory evidence must not pin perception to an unadvanceable ledger.
            receipts = self.session.task_plan.receipts
            if receipts:
                item = receipts[self.advisory_review_index % len(receipts)]
                self.advisory_review_index += 1
                self.scheduled_visual_goal = {"timestep": self.step,
                                              "expected_outcome": item.step.expected_outcome}
                if self.execution.advisory_critic_schedule == "periodic":
                    self._verify(item.step.expected_outcome, self._tool_context())
                else:
                    self.deferred_critic_checks += 1
                    self.session.workspace.append_event("visual_critic_deferred", {
                        **self.scheduled_visual_goal, "new_model_call": False,
                        "reason": "unadmitted Critic prose is audit-only; Planner inspects current RGB directly",
                        "goal_is_observation": False,
                    }, source="official_pi05_executor")
            return
        # Review one completed stage before advancing; contradictions revoke dependents.
        confirmed = [r for r in self.session.task_plan.receipts if r.status.value == "confirmed"]
        if confirmed:
            index = (self.step // self.execution.semantic_interval_steps) % len(confirmed)
            item = confirmed[index]
            report = self._verify(item.step.expected_outcome, self._tool_context())
            if report.status is VerificationStatus.CONTRADICTED:
                self.session.reopen_confirmed_plan_step(item.step.stage, report, timestep=self.step)
                return
        active = self.session.task_plan.active
        if active is not None and active.attempts:
            report = self._verify(active.step.expected_outcome, self._tool_context())
            self.session.verify_active_plan_step(report, timestep=self.step)

    def run(self):
        if self.ran:
            raise RuntimeError("SortingEpisode is single-use; create a fresh episode session")
        self.ran = True
        start = time.perf_counter()
        try:
            self._capture()
            if not self.execution.agent_enabled:
                # Same port, VLA runtime, observation cadence and action loop;
                # no Planner/Critic requests, tool routing or monitor intervention.
                while not self.port.done() and self.step < self.task.max_steps:
                    if self._vla_budget_exhausted():
                        self._stop("VLA call budget exhausted", self._tool_context())
                        break
                    self._vla_act(self.instruction, self._tool_context())
                return self._summary(start)
            transition = self.session.start(self._context("task_start"))
            next_review = self.execution.semantic_interval_steps
            result = None
            while not self.port.done() and self.step < self.task.max_steps and not self.stopped:
                if self._vla_budget_exhausted():
                    self._stop("VLA call budget exhausted", self._tool_context())
                    break
                if transition is not None:
                    if transition.ticket is not None and transition.planner is None:
                        transition = self.session.await_planner()
                    if transition.planner is None and transition.state.value == "safe_hold":
                        self._stop(transition.reason or "Harness requested hold", self._tool_context())
                        break
                    if transition.planner is not None and transition.planner.result.accepted:
                        note = transition.planner.result.decision.memory_note
                        if self.execution.memory_enabled:
                            self.memory.remember_hypothesis(self.step, note)
                    result = self.session.apply_planner_transition(transition, context=self._tool_context())
                    if transition.timed_out or transition.stale:
                        self._stop("Planner timed out or returned stale context", self._tool_context())
                        break
                    transition = None
                if result is None and not self.stopped:
                    active = self.session.task_plan.active
                    if active is not None:
                        self.session.select_active_task_plan_step(timestep=self.step)
                    if active is not None and active.step.intent == "run_skill":
                        result = self.session.tools.invoke("run_skill", {
                            "skill_id": active.step.skill_id, "skill_args": {}}, context=self._tool_context())
                    else:
                        instruction = (compose_grounded_vla_instruction(self.instruction, subgoal=active.step.subgoal)
                                       if active is not None and self.execution.subgoal_conditioning_admitted
                                       else self.instruction)
                        result = self.session.tools.invoke("vla_act", {"instruction": instruction},
                                                           context=self._tool_context())
                if result is not None and not result.accepted:
                    self.rejections += 1
                    if self.rejections >= self.execution.max_consecutive_rejections:
                        self._stop("repeated rejected tool calls", self._tool_context())
                    else:
                        transition = self.session.request_semantic_checkpoint(
                            self._context("tool_rejected"), reason=str(result.error))
                elif result is not None and self.session.tools.primitive_outcome(result, episode_id=self.config.run_id):
                    self.rejections = 0
                    outcome = self.session.tools.primitive_outcome(result, episode_id=self.config.run_id)
                    self.last_primitive = outcome.to_planner_dict()
                    if self.execution.memory_enabled:
                        self.memory.remember_execution(outcome)
                    self.session.record_primitive_result(result, planner_context=self._context())
                    tool_return = result.name == "run_skill"
                    if (not self.port.done() and self.step < self.task.max_steps and
                            not self._vla_budget_exhausted() and
                            (tool_return or self.step >= next_review or (self.risk is not None and self.risk.event))):
                        # Feedback tools already checked their bound target; return to
                        # the Planner before another VLA call and avoid duplicate Critic work.
                        if not tool_return:
                            self._review_progress()
                        next_review = self.step + self.execution.semantic_interval_steps
                        if not self.session.task_plan.completed:
                            if tool_return:
                                self.tool_return_reviews += 1
                            transition = self.session.request_semantic_checkpoint(
                                self._context("tool_return" if tool_return else "execution_review"),
                                reason="execution tool returned fresh evidence" if tool_return else "scheduled review or execution-risk signal",
                                count_as_retry=bool(self.risk is not None and self.risk.event))
                        elif self.risk is not None and self.risk.event:
                            self._stop("execution risk after symbolic plan completion", self._tool_context())
                result = None
            return self._summary(start)
        finally:
            self.session.close(reason=self.stop_reason or "episode_end")

    def _summary(self, start):
        return {
                "agent_enabled": self.execution.agent_enabled,
                "observation_reads": getattr(self.port, "observation_reads", None),
                "task": self.task.task_id, "control_steps": self.step,
                "vla_calls": self.vla_calls, "vla_control_steps": self.executed_vla_steps,
                "vla_attempts": self.vla_attempts,
                "generated_vla_steps": self.generated_vla_steps,
                "discarded_vla_steps": self.generated_vla_steps - self.executed_vla_steps,
                "visual_check_cache_hits": self.visual_check_cache_hits,
                "deferred_critic_checks": self.deferred_critic_checks,
                "runtime_budget": self._runtime_budget(),
                "timing_scope": "synchronous policy RPC wall time; includes first-call warmup, not end-to-end control latency",
                "interrupted_chunks": self.interrupted_chunks,
                "inference_wall_ms": self.inference_wall_ms,
                "critic": self.critic.metrics(), "stopped": self.stopped,
                "execution_config": dataclasses.asdict(self.execution),
                "skill_calls": self.skill_calls,
                "feedback_postchecks": self.feedback_postchecks,
                "tool_return_reviews": self.tool_return_reviews,
                "task_plan": self.session.task_plan.to_dict(),
                "stop_reason": self.stop_reason, "wall_seconds": time.perf_counter() - start,
                "official_success": None, "official_score": None,
                "scope": "framework execution trace; scores must come from the official evaluator",
            }


def run_kinematics_smoke(episode):
    """Host-only robot calibration around one native VLA chunk, no IK dispatch."""
    if episode.ran:
        raise RuntimeError("SortingEpisode is single-use")
    if (episode.execution.agent_enabled or episode.execution.vla_call_budget != 1
            or episode.execution.subgoal_conditioning_admitted
            or episode.execution.feedback_refresh_enabled):
        raise ValueError("kinematics smoke requires Agent off, original instruction and one VLA call")
    episode.ran = True
    start = time.perf_counter()
    checks = []
    try:
        episode._capture()
        for phase in ("before_native_chunk", "after_native_chunk"):
            audit = episode.port.audit_robot_kinematics()
            checks.append({"phase": phase, "timestep": episode.step, **audit})
            episode.session.workspace.append_event("robot_kinematics_audit", checks[-1], source="test_driver")
            if not audit["fk_passed"]:
                raise RuntimeError("robot FK calibration failed; no EEF execution admitted")
            if phase == "before_native_chunk":
                episode._vla_act(episode.instruction, episode._tool_context())
        episode._stop("read-only kinematics diagnostic completed", episode._tool_context())
        return {**episode._summary(start), "autonomous_planner_used": False,
                "robot_kinematics_checks": checks, "ik_actions_executed": 0,
                "scope": "host robot calibration around native actions; not autonomous recovery or full benchmark"}
    finally:
        episode.session.close(reason="kinematics_smoke_end")


def run_feedback_tool_smoke(episode):
    """Forced tool integration on a real port; NOT autonomous Agent evaluation.

    The caller must provide the real VLA and Critic. No oracle, motor trajectory,
    or task-completion override is used. The official evaluator owns final scores.
    """
    if episode.ran:
        raise RuntimeError("SortingEpisode is single-use")
    if (not episode.execution.feedback_refresh_enabled
            or episode.execution.vla_call_budget != 2
            or episode.execution.critic_call_budget != 2
            or episode.execution.critic_authority != "advisory"
            or episode.execution.subgoal_conditioning_admitted):
        raise ValueError("feedback smoke requires task-only advisory execution and two-call budgets")
    episode.ran = True
    start = time.perf_counter()
    receipts = []
    try:
        episode._capture()
        episode.session.workspace.append_event("controlled_tool_smoke", {
            "autonomous_planner": False,
            "sequence": ["visual_check", "feedback_refresh", "vla_act"],
            "scope": "forced tool capability check; not autonomous recovery or benchmark comparison",
        }, source="test_driver")
        episode._verify(episode.instruction, episode._tool_context())
        # Exercise request deduplication without changing images or the predicate.
        episode._verify(episode.instruction, episode._tool_context())
        for name, args in (("run_skill", {"skill_id": "feedback_refresh", "skill_args": {}}),
                           ("vla_act", {"instruction": episode.instruction})):
            if episode.port.done() or episode.step >= episode.task.max_steps:
                break
            result = episode.session.tools.invoke(name, args, context=episode._tool_context())
            if not result.accepted:
                raise RuntimeError(f"controlled tool {name} rejected: {result.error}")
            outcome = episode.session.tools.primitive_outcome(result, episode_id=episode.config.run_id)
            if outcome is None:
                raise RuntimeError("controlled tool did not return an execution receipt")
            episode.last_primitive = outcome.to_planner_dict()
            if episode.execution.memory_enabled:
                episode.memory.remember_execution(outcome)
            receipts.append(episode._planner_receipt(episode.last_primitive))
        episode._stop("controlled tool smoke completed", episode._tool_context())
        return {**episode._summary(start), "autonomous_planner_used": False,
                "controlled_tool_receipts": receipts,
                "scope": "forced feedback-tool integration; not autonomous recovery efficacy or benchmark comparison"}
    finally:
        episode.session.close(reason="controlled_tool_smoke_end")

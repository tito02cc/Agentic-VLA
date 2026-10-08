"""One canonical CARVE session shared by coding agents and embedded VLMs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence

from agentic_vla.assembly import build_planner_factory
from agentic_vla.configuration import CarveRunConfig, PlannerExecutionMode
from agentic_vla.runtime import (
    AgentIntent,
    AsyncAgenticHarnessController,
    ExecutionMode,
    HarnessTransition,
    HighLevelAgentContext,
    JointControllerConfig,
    JointRecoveryComputeController,
    PausedExecutionSafeHoldAdapter,
    RiskAssessment,
    SafeHoldAdapter,
)
from agentic_vla.runtime.agent import (
    TASK_ONLY_EXECUTION_TOOLS,
    PlannerCallable,
    bind_cumulative_task_outcome,
    build_high_level_agent_request,
)
from agentic_vla.runtime.knowledge import (
    AgenticKnowledgeProvider,
    ProceduralStep,
    ProceduralTaskRecord,
)
from agentic_vla.runtime.recovery import FailureEpisodeRecord, FailureMemory
from agentic_vla.toolchain import (
    CanonicalToolRuntime,
    EmbodiedTaskPlan,
    EmbodiedToolBindings,
    PrimitiveOutcome,
    PrimitiveStatus,
    RunWorkspace,
    ToolExecutionContext,
    ToolResult,
    VerificationReport,
    VerificationStatus,
    VerifiedPrimitiveTrace,
    VerifiedProcedureCompiler,
    reject_direct_action_fields,
)


@dataclasses.dataclass(frozen=True)
class PrimitiveCompletion:
    """Canonical result after optional verification and memory gating."""

    outcome: PrimitiveOutcome
    transition: HarnessTransition
    verification: VerificationReport | None = None
    verification_tool_result: ToolResult | None = None
    memory_recorded: bool = False


class CarveAgentSession:
    """Bind planner lifecycle, Harness authority, tools, and artifacts once."""

    def __init__(
        self,
        config: CarveRunConfig,
        *,
        bindings: EmbodiedToolBindings,
        controller: JointRecoveryComputeController | None = None,
        knowledge_provider: AgenticKnowledgeProvider | None = None,
        failure_memory: FailureMemory | None = None,
        scripted_infer: PlannerCallable | None = None,
        workspace: RunWorkspace | None = None,
        safe_hold_adapter: SafeHoldAdapter | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace or RunWorkspace(
            config.workspace_path, config.to_run_manifest()
        )
        self.tools = CanonicalToolRuntime(
            bindings=bindings,
            workspace=self.workspace,
            tool_call_budgets={
                "run_skill": config.harness.effective_physical_skill_budget
            },
        )
        self.hold_adapter = safe_hold_adapter or PausedExecutionSafeHoldAdapter()
        self.knowledge_provider = knowledge_provider or AgenticKnowledgeProvider()
        self.procedure_compiler = VerifiedProcedureCompiler(
            self.knowledge_provider.procedural_memory,
            minimum_task_confidence=config.planner.minimum_intervention_confidence,
        )
        self._procedure_trace: list[VerifiedPrimitiveTrace] = []
        self.task_plan = EmbodiedTaskPlan()
        self._task_instruction = ""
        planner_factory = build_planner_factory(
            config.planner, scripted_infer=scripted_infer
        )
        self.harness = AsyncAgenticHarnessController(
            controller
            or JointRecoveryComputeController(
                JointControllerConfig(
                    planner_after_failures=max(1, config.harness.retry_budget),
                    max_recovery_attempts=config.harness.recovery_budget,
                )
            ),
            planner_factory=planner_factory,
            task_start_policy=config.harness.task_start_policy,
            primitive_boundary_policy=config.harness.primitive_boundary_policy,
            safe_hold_adapter=self.hold_adapter,
            failure_memory=failure_memory,
            knowledge_provider=self.knowledge_provider,
            memory_retrieval_limit=config.harness.memory_retrieval_limit,
            planner_cooldown_steps=config.harness.planner_cooldown_steps,
            semantic_retry_budget=config.harness.retry_budget,
            safe_hold_timeout_s=config.harness.safe_hold_timeout_s,
            safe_hold_heartbeat_hz=config.harness.safe_hold_heartbeat_hz,
        )
        self._episode_id: str | int | None = None

    @property
    def external_tool_agent(self) -> bool:
        return (
            self.config.planner.mode
            is PlannerExecutionMode.EXTERNAL_CODING_AGENT
        )

    def start(self, context: HighLevelAgentContext) -> HarnessTransition:
        if self._episode_id is not None:
            raise RuntimeError("CARVE session already has an active episode")
        self._episode_id = context.episode_id
        self._task_instruction = context.task_instruction
        self._procedure_trace.clear()
        self.task_plan.reset()
        self.workspace.append_event(
            "episode_start",
            {
                "episode_id": context.episode_id,
                "timestep": context.timestep,
                "planner_mode": self.config.planner.mode.value,
                "config_fingerprint": self.config.fingerprint,
            },
            source="session",
        )
        transition = self.harness.start_episode(
            context.episode_id,
            task_start_context=(None if self.external_tool_agent else context),
        )
        self._record_submission(transition)
        return transition

    def await_planner(
        self,
        *,
        timeout_s: float | None = None,
    ) -> HarnessTransition:
        if self.external_tool_agent:
            raise RuntimeError("external coding agent has no hidden planner request")
        transition = self.harness.await_planner(
            timeout_s=(
                self.config.planner.timeout_s if timeout_s is None else timeout_s
            )
        )
        self._record_planner_result(transition)
        return transition

    def apply_planner_transition(
        self,
        transition: HarnessTransition,
        *,
        context: ToolExecutionContext,
    ) -> ToolResult | None:
        if transition.planner is None:
            return None
        if transition.stale or transition.timed_out or not transition.decision_applied:
            return None
        decision = transition.planner.result.decision
        # Protect symbolic state as well as motor dispatch while execution owns control.
        if not context.at_safe_boundary and decision.intent is not AgentIntent.SAFE_STOP:
            self.workspace.append_event(
                "planner_transition_not_applied",
                {
                    "ticket_id": transition.planner.ticket.ticket_id,
                    "timestep": context.timestep,
                    "intent": decision.intent.value,
                    "reason": "ordinary planner update requires an execution boundary",
                },
                source="harness",
            )
            return None
        context = self._context_with_joint_controls(context, transition)
        if decision.proposed_plan:
            bound_steps = bind_cumulative_task_outcome(
                decision.proposed_plan,
                transition.planner.ticket.context.task_instruction,
            )
            self.task_plan.install(
                bound_steps,
                available_skills=transition.planner.ticket.context.available_skills,
            )
            self.workspace.append_event(
                "task_plan_installed",
                self.task_plan.to_dict(),
                source="harness",
            )
        if (
            self.task_plan.completed
            and decision.intent.value in {"vla_act", "run_skill"}
        ):
            self.workspace.append_event(
                "planner_decision_superseded",
                {
                    "reason": "task plan completed before planner response was applied",
                    "ticket_id": transition.planner.ticket.ticket_id,
                    "intent": decision.intent.value,
                    "subgoal": decision.subgoal,
                    "task_plan": self.task_plan.to_dict(),
                },
                source="harness",
            )
            return None
        planner_context = transition.planner.ticket.context
        execution_tool = (planner_context.compact_task_only_review
            and planner_context.vla_instruction_mode == "task_only"
            and decision.intent.value == "run_skill"
            and decision.skill_id in TASK_ONLY_EXECUTION_TOOLS
            and decision.skill_id in planner_context.available_skills)
        if self.task_plan.installed and not execution_tool and decision.intent.value in {
            "vla_act",
            "run_skill",
        }:
            selected = self.task_plan.select(
                subgoal=decision.subgoal,
                intent=decision.intent.value,
                skill_id=decision.skill_id,
            )
            self.workspace.append_event(
                "task_plan_step_selected",
                selected.to_dict(),
                source="harness",
            )
        result = self.tools.invoke_decision(
            decision,
            context=context,
        )
        self._account_physical_tool(result)
        return result

    def select_active_task_plan_step(
        self,
        *,
        timestep: int,
        source: str = "harness",
    ) -> None:
        """Bind the already-authorized active stage without another Planner call."""

        active = self.task_plan.active
        if active is None:
            raise RuntimeError("task plan has no active step to select")
        selected = self.task_plan.select(
            subgoal=active.step.subgoal,
            intent=active.step.intent,
            skill_id=active.step.skill_id,
        )
        self.workspace.append_event(
            "task_plan_step_selected",
            {"timestep": timestep, **selected.to_dict()},
            source=source,
        )

    def install_retrieved_task_plan(
        self,
        steps: Sequence[ProceduralStep],
        *,
        available_skills: Sequence[str],
        timestep: int,
        procedure_id: str,
        task_instruction: str = "",
    ) -> None:
        """Install a verified symbolic procedure without restating it through a VLM."""

        if self._episode_id is None:
            raise RuntimeError("procedure warm-start requires an active episode")
        if timestep < 0 or not procedure_id.strip():
            raise ValueError("procedure warm-start metadata is invalid")
        self.task_plan.install(
            bind_cumulative_task_outcome(steps, task_instruction),
            available_skills=available_skills,
        )
        self.workspace.append_event(
            "task_plan_installed",
            {
                **self.task_plan.to_dict(),
                "timestep": timestep,
                "procedure_id": procedure_id.strip(),
            },
            source="verified_procedure_memory",
        )

    def route_execution(
        self,
        risk: RiskAssessment,
        *,
        planner_context: HighLevelAgentContext,
        deadline_ms: float | None,
        deadline_slack_ms: float | None,
        cached_actions: int = 0,
    ) -> HarnessTransition:
        """Route one monitor assessment through the canonical Harness state machine."""

        if self._episode_id is None:
            raise RuntimeError("CARVE session has no active episode")
        transition = self.harness.route(
            risk,
            context=planner_context,
            deadline_ms=deadline_ms,
            deadline_slack_ms=deadline_slack_ms,
            cached_actions=cached_actions,
        )
        self.workspace.append_event(
            "execution_routed",
            {
                "risk": dataclasses.asdict(risk),
                "state": transition.state.value,
                "joint": (
                    None if transition.joint is None else transition.joint.to_dict()
                ),
                "reason": transition.reason,
            },
            source="session",
        )
        self._record_submission(transition)
        return transition

    def request_semantic_checkpoint(
        self,
        planner_context: HighLevelAgentContext,
        *,
        reason: str,
        count_as_retry: bool = True,
    ) -> HarnessTransition:
        """Request one explicit low-frequency replan at a caller-owned safe boundary."""

        if self._episode_id is None:
            raise RuntimeError("CARVE session has no active episode")
        transition = self.harness.request_planner(
            planner_context,
            purpose="scheduled_semantic_checkpoint",
            reason=reason,
            respect_cooldown=False,
            count_as_retry=count_as_retry,
        )
        self.workspace.append_event(
            "semantic_checkpoint_requested",
            {
                "timestep": planner_context.timestep,
                "reason": reason,
                "current_subgoal": planner_context.current_subgoal,
                "count_as_retry": count_as_retry,
            },
            source="session",
        )
        self._record_submission(transition)
        return transition

    def verify_active_plan_step(
        self,
        report: VerificationReport,
        *,
        timestep: int,
        source: str = "visual_critic",
    ) -> None:
        """Apply trusted Critic evidence to the active symbolic task step."""

        if self._episode_id is None:
            raise RuntimeError("task-plan verification requires an active episode")
        if timestep < 0:
            raise ValueError("task-plan verification timestep must be non-negative")
        receipt = self.task_plan.apply_verification(report)
        self.workspace.append_event(
            "task_plan_step_verified",
            {
                "timestep": timestep,
                "step": receipt.to_dict(),
                "task_plan": self.task_plan.to_dict(),
            },
            source=source,
        )

    def apply_execution_transition(
        self,
        transition: HarnessTransition,
        *,
        context: ToolExecutionContext,
        vla_instruction: str,
    ) -> ToolResult | None:
        """Execute a routed Harness decision through the same typed tool boundary."""

        joint = transition.joint
        if joint is None or joint.mode in {ExecutionMode.REUSE, ExecutionMode.PLANNER}:
            return None
        context = self._context_with_joint_controls(context, transition)
        if joint.mode in {ExecutionMode.FAST_VLA, ExecutionMode.ACCURATE_VLA}:
            if not vla_instruction.strip():
                raise ValueError("VLA execution requires a non-empty instruction")
            result = self.tools.invoke(
                "vla_act", {"instruction": vla_instruction}, context=context
            )
        elif joint.mode is ExecutionMode.RECOVERY:
            if not joint.recovery_skill_id:
                raise ValueError("recovery transition has no registered skill")
            result = self.tools.invoke(
                "run_skill",
                {"skill_id": joint.recovery_skill_id, "skill_args": {}},
                context=context,
            )
        else:
            result = self.tools.invoke(
                "safe_hold", {"reason": transition.reason}, context=context
            )
        self._account_physical_tool(result)
        return result

    def reopen_confirmed_plan_step(
        self, stage: str, report: VerificationReport, *, timestep: int
    ) -> None:
        """Record new contradictory evidence without silently retaining stale progress."""
        if self._episode_id is None or timestep < 0:
            raise RuntimeError("stage invalidation requires an active episode and timestep")
        before = self.task_plan.to_dict()
        self.task_plan.reopen_confirmed(stage, report)
        self.workspace.append_event(
            "task_plan_invalidated",
            {"timestep": timestep, "stage": stage, "verification": report.to_dict(),
             "before": before, "after": self.task_plan.to_dict()},
            source="visual_critic",
        )

    def invoke_external_tool(
        self,
        name: str,
        arguments: dict,
        *,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not self.external_tool_agent:
            raise RuntimeError("direct external tool calls require external_coding_agent mode")
        if self._episode_id is None:
            raise RuntimeError("external tools require an active CARVE episode")
        if context.episode_id != self._episode_id:
            raise ValueError("external tool context belongs to another episode")
        result = self.tools.invoke(name, arguments, context=context)
        self._account_physical_tool(result)
        return result

    def record_primitive_result(
        self,
        result: ToolResult,
        *,
        planner_context: HighLevelAgentContext,
    ) -> tuple[PrimitiveOutcome, HarnessTransition]:
        if self._episode_id is None:
            raise RuntimeError("CARVE session has no active episode")
        outcome = self.tools.primitive_outcome(
            result, episode_id=self._episode_id
        )
        if outcome is None:
            raise ValueError("tool result is not a physical primitive")
        transition = self.harness.record_primitive_boundary(
            planner_context, outcome
        )
        self.workspace.append_event(
            "primitive_boundary",
            outcome.to_planner_dict(),
            source="session",
        )
        self._record_submission(transition)
        return outcome, transition

    def finalize_primitive(
        self,
        result: ToolResult,
        *,
        planner_context: HighLevelAgentContext,
    ) -> PrimitiveCompletion:
        """Verify one primitive, gate memory, then apply boundary scheduling."""

        if self._episode_id is None:
            raise RuntimeError("CARVE session has no active episode")
        outcome = self.tools.primitive_outcome(
            result, episode_id=self._episode_id
        )
        if outcome is None:
            raise ValueError("tool result is not a physical primitive")
        if planner_context.timestep != outcome.ended_timestep:
            raise ValueError("planner context must describe the primitive end timestep")

        verification = None
        verification_tool_result = None
        finalized = outcome
        if outcome.requires_semantic_check or outcome.abnormal:
            expected = outcome.expected_outcome.strip() or outcome.observed_outcome.strip()
            if not expected:
                expected = f"verify {outcome.primitive_name} outcome"
            verification_context = ToolExecutionContext(
                episode_id=self._episode_id,
                timestep=outcome.ended_timestep,
                at_safe_boundary=True,
                allowed_tools=self.config.harness.allowed_tools,
                deployment_profile_id=self.config.vla.deployment_profile_id,
            )
            verification_tool_result = self.tools.invoke(
                "verify",
                {"expected_outcome": expected},
                context=verification_context,
            )
            if verification_tool_result.accepted:
                verification = VerificationReport.from_dict(
                    verification_tool_result.output
                )
            else:
                verification = VerificationReport(
                    status=VerificationStatus.INCONCLUSIVE,
                    observed_outcome=(
                        "verification tool failed: "
                        f"{verification_tool_result.error or 'unknown error'}"
                    ),
                    confidence=0.0,
                )

            if (
                verification.status is VerificationStatus.CONFIRMED
                and not outcome.abnormal
            ):
                finalized = dataclasses.replace(
                    outcome,
                    observed_outcome=verification.observed_outcome,
                    requires_semantic_check=False,
                )
            elif verification.status is VerificationStatus.CONTRADICTED:
                finalized = dataclasses.replace(
                    outcome,
                    status=PrimitiveStatus.FAILED,
                    observed_outcome=verification.observed_outcome,
                    requires_semantic_check=True,
                )

        memory_recorded = self._record_verified_failure(
            finalized,
            verification=verification,
            context=planner_context,
        )
        self._procedure_trace.append(
            VerifiedPrimitiveTrace(
                outcome=finalized,
                subgoal=planner_context.current_subgoal,
                verification=verification,
            )
        )
        if (
            verification is not None
            and self.task_plan.installed
            and self._primitive_matches_active_plan_step(finalized)
        ):
            plan_receipt = self.task_plan.apply_verification(verification)
            self.workspace.append_event(
                "task_plan_step_verified",
                {
                    "step": plan_receipt.to_dict(),
                    "task_plan": self.task_plan.to_dict(),
                },
                source="harness",
            )
        if finalized.abnormal:
            self.harness.record_failure()
        elif verification is None or verification.status is VerificationStatus.CONFIRMED:
            self.harness.clear_failures()

        transition = self.harness.record_primitive_boundary(
            planner_context, finalized
        )
        self.workspace.append_event(
            "primitive_finalized",
            {
                "outcome": finalized.to_planner_dict(),
                "verification": (
                    None if verification is None else verification.to_dict()
                ),
                "memory_recorded": memory_recorded,
                "next_state": transition.state.value,
                "reason": transition.reason,
            },
            source="session",
        )
        self._record_submission(transition)
        return PrimitiveCompletion(
            outcome=finalized,
            transition=transition,
            verification=verification,
            verification_tool_result=verification_tool_result,
            memory_recorded=memory_recorded,
        )

    def _primitive_matches_active_plan_step(self, outcome: PrimitiveOutcome) -> bool:
        """Prevent recovery postconditions from completing semantic task stages."""

        active = self.task_plan.active
        if active is None:
            return False
        step = active.step
        if step.intent == "vla_act":
            return outcome.primitive_name == "vla_act"
        if step.intent == "run_skill":
            return outcome.primitive_name == step.skill_id
        return False

    @property
    def procedure_trace(self) -> tuple[VerifiedPrimitiveTrace, ...]:
        """Return the action-free symbolic trace collected in this episode."""

        return tuple(self._procedure_trace)

    def promote_verified_procedure(
        self,
        *,
        task_family: str,
        object_categories: tuple[str, ...],
        task_verification: VerificationReport,
        notes: tuple[str, ...] = (),
    ) -> ProceduralTaskRecord:
        """Persist a successful recipe through a trusted evaluator-side gate."""

        if self._episode_id is None:
            raise RuntimeError("procedure promotion requires an active episode")
        verified_steps: tuple[ProceduralStep, ...] = ()
        if self.task_plan.installed:
            if not self.task_plan.completed:
                raise RuntimeError(
                    "procedure promotion requires the installed task plan to be complete"
                )
            verified_steps = tuple(
                receipt.step for receipt in self.task_plan.receipts
            )
        record = self.procedure_compiler.compile_and_record(
            task_instruction=self._task_instruction,
            task_family=task_family,
            object_categories=object_categories,
            source_episode_id=self._episode_id,
            trace=self.procedure_trace,
            task_verification=task_verification,
            notes=notes,
            verified_steps=verified_steps,
        )
        self.workspace.append_event(
            "procedure_promoted",
            record.to_dict(),
            source="trusted_task_verifier",
        )
        return record

    def close(self, *, reason: str = "episode_end") -> None:
        if self._episode_id is None:
            return
        self.workspace.append_event(
            "episode_close",
            {"episode_id": self._episode_id, "reason": reason},
            source="session",
        )
        self.harness.close_episode(reason=reason)
        self._episode_id = None

    def finish(
        self,
        *,
        status: str,
        summary: str,
        timestep: int,
    ) -> ToolResult:
        """Finish and close one episode through the canonical tool boundary."""

        if self._episode_id is None:
            raise RuntimeError("CARVE session has no active episode")
        context = ToolExecutionContext(
            episode_id=self._episode_id,
            timestep=timestep,
            at_safe_boundary=True,
            allowed_tools=self.config.harness.allowed_tools,
            deployment_profile_id=self.config.vla.deployment_profile_id,
        )
        result = self.tools.invoke(
            "finish",
            {"status": status, "summary": summary},
            context=context,
        )
        if result.accepted:
            self.close(reason=f"finished:{status}")
        return result

    def _record_submission(self, transition: HarnessTransition) -> None:
        if transition.ticket is None:
            return
        request = build_high_level_agent_request(transition.ticket.context)
        metadata = {
            "ticket_id": transition.ticket.ticket_id,
            "episode_id": transition.ticket.context.episode_id,
            "timestep": transition.ticket.context.timestep,
            "prompt_schema_version": self.config.planner.prompt_schema_version,
        }
        self.workspace.append_transcript(
            role="system", content=str(request["system_prompt"]), metadata=metadata
        )
        self.workspace.append_transcript(
            role="user", content=str(request["user_prompt"]), metadata=metadata
        )
        self.workspace.append_event(
            "planner_submitted", metadata, source="session"
        )

    @staticmethod
    def _context_with_joint_controls(
        context: ToolExecutionContext,
        transition: HarnessTransition,
    ) -> ToolExecutionContext:
        if transition.joint is None:
            return context
        return dataclasses.replace(
            context,
            inference_controls=dataclasses.asdict(transition.joint.controls),
        )

    def _account_physical_tool(self, result: ToolResult | None) -> None:
        if result is not None and result.accepted and result.name == "run_skill":
            self.harness.record_recovery_attempt()

    def _record_verified_failure(
        self,
        outcome: PrimitiveOutcome,
        *,
        verification: VerificationReport | None,
        context: HighLevelAgentContext,
    ) -> bool:
        if (
            not outcome.abnormal
            or verification is None
            or verification.status is not VerificationStatus.CONTRADICTED
        ):
            return False
        monitor_evidence = {
            "bucket": context.risk.get("bucket", "unknown"),
            "score": context.risk.get("score", 0.0),
            "components": dict(context.risk.get("components", {})),
            "evidence": dict(context.risk.get("evidence", {})),
        }
        reject_direct_action_fields(
            monitor_evidence, path="verified_failure.monitor_evidence"
        )
        verification_metadata = dict(verification.metadata)
        reject_direct_action_fields(
            verification_metadata, path="verified_failure.metadata"
        )
        fingerprint = context.memory_context_fingerprint.strip()
        if not fingerprint:
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "task": context.task_instruction,
                        "subgoal": context.current_subgoal,
                        "profile": self.config.vla.deployment_profile_id,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
        action_age = context.risk.get("evidence", {}).get("action_age_steps", 0)
        record = FailureEpisodeRecord(
            context_fingerprint=fingerprint,
            episode_id=context.episode_id,
            task=context.task_instruction,
            subgoal=(
                context.current_subgoal.strip()
                or outcome.expected_outcome.strip()
                or outcome.primitive_name
            ),
            policy_id=self.config.vla.policy_id,
            deployment_profile_id=self.config.vla.deployment_profile_id,
            failure_type=str(
                context.risk.get("event") or outcome.status.value
            ),
            monitor_evidence=monitor_evidence,
            action_age_steps=max(0, int(action_age)),
            intervention=outcome.primitive_name,
            retry_budget_consumed=self.harness.counters.semantic_retries,
            recovery_budget_consumed=self.harness.counters.recovery_attempts,
            verification_result=(
                f"{verification.status.value}: {verification.observed_outcome}"
            ),
            terminal_outcome=outcome.status.value,
            confidence=verification.confidence,
            metadata={
                "primitive_call_id": outcome.call_id,
                **verification_metadata,
            },
        )
        self.harness.record_failure_evidence(record)
        return True

    def _record_planner_result(self, transition: HarnessTransition) -> None:
        completed = transition.planner
        if completed is None:
            self.workspace.append_event(
                "planner_wait_finished",
                {
                    "timed_out": transition.timed_out,
                    "stale": transition.stale,
                    "reason": transition.reason,
                },
                source="session",
            )
            return
        result = completed.result
        rejected_outputs = list(
            result.raw_outputs[:-1] if result.accepted else result.raw_outputs
        )
        self.workspace.append_transcript(
            role="planner",
            content=json.dumps(result.decision.to_dict(), sort_keys=True),
            metadata={
                "ticket_id": completed.ticket.ticket_id,
                "accepted": result.accepted,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "attempt_count": result.attempt_count,
                "validation_errors": list(result.validation_errors),
                "rejected_outputs": rejected_outputs,
            },
        )
        self.workspace.append_event(
            "planner_result",
            {
                "ticket_id": completed.ticket.ticket_id,
                "accepted": result.accepted,
                "decision_applied": transition.decision_applied,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "attempt_count": result.attempt_count,
                "validation_errors": list(result.validation_errors),
                "rejected_outputs": rejected_outputs,
                "decision": result.decision.to_dict(),
            },
            source="session",
        )

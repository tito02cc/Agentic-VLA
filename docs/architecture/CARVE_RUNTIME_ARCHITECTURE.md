# CARVE Runtime Architecture

The guarded external high-level multimodal agent is specified separately in
[`CARVE_HIGH_LEVEL_AGENT.md`](CARVE_HIGH_LEVEL_AGENT.md). The previously reported
recovery results predate this integration and therefore remain correctly labeled
as deterministic-controller evidence.

## Scope

CARVE has two independently useful, loosely coupled layers. **CARVE Agentic
Harness** owns execution reliability through monitoring, memory, diagnosis,
retry, recovery, semantic escalation, and safe fallback. **CARVE Optimize
Runtime** owns efficient policy execution through capability discovery,
deployment-profile calibration, optimization backends, action-fidelity gates,
deadline enforcement, and tracing.

The current runtime foundation supplies the stable contracts between these
layers and separates both from pi0.5, OpenPI, benchmark clients, and simulator
code. It is intentionally a small boundary layer, not a second rollout
implementation. Quantization, compilation, CUDA Graphs, fused kernels, RTC,
memory implementations, and critic implementations remain replaceable plugins.

The current Optimize Runtime implementation is documented in
`CARVE_OPTIMIZE_RUNTIME.md`; this file remains the canonical architectural
definition.

## CARVE Embodied Toolchain

CARVE exposes the Harness and Optimize Runtime through an agent-facing tool
ecosystem. This layer is inspired by RPent's service-oriented, standardized,
and composable infrastructure, but retains CARVE's stricter authority and
deployment admission boundaries:

```text
Codex / Claude / local VLM / scripted planner
                    |
          discover typed ToolSpec entries
                    v
       CARVE EmbodiedToolRegistry + Run Workspace
       allowlist | budget | safe boundary | trace
                    |
        +-----------+-------------+
        |                         |
 Agentic Harness tools      Optimize Runtime tools
 observe / memory /         admitted VLA profile /
 verify / skill / hold      fallback / receipt
        |                         |
        +-----------+-------------+
                    v
       benchmark, simulator, or robot adapter
```

The stable initial vocabulary is:

| Tool | Purpose | Authority |
|---|---|---|
| `observe` | read deployable observations and monitor evidence | read-only |
| `retrieve_memory` | retrieve HAA, procedure, and failure context | read-only |
| `vla_act` | request one admitted frozen-VLA action chunk | safe boundary |
| `run_skill` | execute one registered bounded physical primitive | safe boundary and budget |
| `verify` | test a symbolic expected outcome with deployable evidence | read-only |
| `safe_hold` | enter the adapter-defined stationary hold lifecycle | lifecycle |
| `finish` | close a run with an auditable status and summary | lifecycle |

`ToolSpec`, `ToolCall`, `ToolExecutionContext`, `ToolResult`, and
`EmbodiedToolRegistry` are implemented under `agentic_vla/toolchain/`. Tool
inputs reject direct actions, trajectories, joint targets, simulator state,
and torques. Every state-changing call is episode/timestep scoped, and physical
tools can require a safe boundary and per-episode budget. This lets a coding
agent continuously operate a toolchain without granting it unrestricted motor
authority.

`RunWorkspace` provides the corresponding artifact contract. It writes a
hardware/profile-aware run manifest, ordered event JSONL, symbolic recipe,
planner transcript, artifact index, and terminal summary. Recipes deliberately
exclude VLA output chunks, while the internal event stream may retain detailed
runtime evidence for evaluation. Artifact paths cannot escape the run root.

The graduation-complete ecosystem has two delivery tiers:

1. **Required:** planner-neutral tool protocol and run-workspace contracts
   (implemented), plus CARVE handlers, environment/VLA service adapters, video
   capture, and one-command experiment entry point.
2. **Enhancement:** interactive steering, live dashboard, additional benchmark
   plugins, and remote multi-machine service discovery. These improve usability
   but are not prerequisites for the controlled thesis experiment.

## Research Lineage From Agentic RAG-VLM

CARVE is the embodied closed-loop continuation of the earlier Agentic RAG-VLM
work, not a disconnected harness around a VLA. The earlier system contributes
three knowledge-side capabilities that remain useful after the learned action
generator changes from grasp libraries to a VLA:

| Earlier capability | CARVE interpretation | Runtime boundary |
|---|---|---|
| HAA-RAG | retrieve task procedures and recovery evidence by functional affordance and failure context | supplies bounded context; never emits a raw robot action |
| Scene Graph Constraint Reasoner | represent object relations, support, occlusion, collision risk, and task-stage constraints | supplies typed constraints to the planner and skill gate |
| Self-Reflection and episodic memory | classify failures, select a bounded escalation level, and retain verified outcomes | updates `FailureMemory`/`RecoveryMemory` only after evidence-based verification |

The grasp-specific parameter correction vector from Agentic RAG-VLM is not
copied into the general VLA controller. CARVE instead converts retrieved
knowledge into one of four typed intents: `continue`, `vla_act`, `run_skill`,
or `safe_stop`. Low-level learned actions remain the responsibility of the VLA;
registered physical primitives remain bounded and adapter-validated.

This produces one continuous thesis-level research line:

1. structure multimodal manipulation knowledge with affordances, scene graphs,
   retrieval, and reflection;
2. use that knowledge selectively during closed-loop VLA execution through a
   multi-rate Agentic Harness;
3. optimize the resulting VLM/VLA runtime under action-fidelity, latency,
   memory, and deadline constraints.

Harness VLA/RPent is an external reference for frozen-VLA primitives,
task-process memory, no-training simulator evaluation, service topology,
planner adapters, tool dispatch, recipe/transcript artifacts, and live
inspection. CARVE adopts these useful infrastructure principles while adding
safe-boundary authority, typed intents, per-tool budgets, evaluator isolation,
deployment-profile admission, and action-fidelity gates. It does not adopt an
evaluator-visible success signal or unrestricted metric action interface. The
reference implementation is kept as an ignored nested checkout at
`third_party/RPent` so it is available beside CARVE while its Apache-2.0 code
and independent Git history remain clearly separated.

## Multi-Rate Agentic Boundary

The runtime separates three rates and responsibilities:

1. a high-frequency deterministic `ExecutionRiskMonitor` processes deployable
   execution signals and deadlines;
2. a medium-frequency admitted VLA profile generates low-level action chunks;
3. a low-frequency external VLM Critic/Planner performs semantic verification
   or replanning only on explicit events and safe boundaries.

The implemented harness state machine is `EXECUTE_FAST -> VERIFY -> RECOVER` or
`PLAN_AT_SAFE_BOUNDARY`, followed by `EXECUTE_FAST`, `SAFE_HOLD`, or `STOP`.
`SAFE_HOLD` is an adapter contract with a stationary command, heartbeat,
timeout, and explicit release condition. Clearing an action queue is not by
itself a deployable hold implementation.

## Public Contracts

- ModelCapabilities: native backend features such as configurable flow steps,
  action chunking, KV cache, predictive context, and supported precision.
- InferenceRequest: normalized observation, instruction, episode lifecycle,
  Agentic event, and requested compute controls.
- ActionChunk: actions, backend latency, uncertainty or predictive context when
  available, and the untouched backend response.
- RuntimeTrace: requested versus applied controls, explicit fallback, queue age,
  model/runtime/reaction/task-cycle latency, action age, stage timing, deadline
  miss, and action count.
- PolicyAdapter: the only interface that CARVE core requires from an action
  policy.
- ActionSpec: action dimension, representation, frame, gripper convention,
  normalization identity, optional valid range, and control frequency. Adapter
  output dimension and finite values are validated before it reaches the
  environment; bounds are enforced only when the deployment declares them.
- ExecutionRiskMonitor: RGB/proprio/action-age/deadline evidence only; simulator
  object poses and success predicates are evaluator-only.
- JointRecoveryComputeController: one decision over cached-action reuse, fast or
  accurate VLA inference, physical recovery, planner escalation, safe stop,
  inference steps, and committed horizon.
- AsyncAgenticHarnessController: episode-scoped planner ownership, task-start
  policy, safe-boundary state transitions, stale-result rejection, independent
  budgets, and `SAFE_HOLD` lifecycle.
- SafeHoldAdapter: robot- or simulator-specific stationary command, heartbeat,
  timeout, release, and stop behavior.
- RecoveryPlan and StatefulRecoveryExecutor: a bounded physical-skill lifecycle
  with explicit phases, verification, outcome memory, replan permission, and
  safe-stop semantics. A recovery decision names a concrete skill ID rather
  than treating a prompt change as physical recovery.
- FailureEpisodeRecord and FailureMemory: typed context/profile identity, failure evidence,
  attempted intervention, verification outcome, confidence, and expiry.
  Retrieval may inform escalation but cannot emit an action.
- ProceduralTaskMemory: bounded cross-episode retrieval of deployably verified,
  symbolic task stages. It may supply subgoals, expected outcomes, constraints,
  and registered skill names, but rejects raw actions, trajectories, simulator
  state, and metric object poses.
- AsyncInferencePrefetcher: a model-independent single-flight request that
  overlaps VLA inference with bounded cached actions, aligns the returned
  suffix, and exposes explicit invalidation on physical risk or escalation.
- DeadlineAwareSemanticScheduler and AsyncSemanticObserver: event, cooldown,
  single-flight, and deadline-slack gates for a replaceable semantic VLM. The
  observer runs outside the control critical path and returns a traced result;
  its output remains shadow-only until an intervention-precision gate passes.
- OpenVlaAdapter: an autoregressive, single-action policy boundary. Unsupported
  flow-step and action-chunk controls are rejected or explicitly dropped rather
  than silently mapped from the pi0.5 contract.

## Current Call Path

    deployable observation + instruction
      -> CarveAgentSession
      -> ExecutionRiskMonitor + AsyncAgenticHarnessController
      -> typed seven-tool boundary
      -> route_execution (risk, budget, deadline, commit horizon)
      -> profile-admitted vla_act or bounded run_skill
      -> CarveRuntime capability negotiation
      -> PolicyAdapter (Pi05Adapter or OpenVlaAdapter)
      -> private ActionChunk execution
      -> typed verification + verified memory gate
      -> RuntimeTrace + recipe + transcript + run summary

LegacyPolicyClientBridge exposes the old client.infer(payload) protocol. It
allows the existing LIBERO runner to adopt CARVE without changing its observation
payload or response parsing.

The existing runner keeps its historical path by default. Add
--carve-runtime to a normal run command to enable the adapter boundary. This
first integration does not alter router thresholds, prompts, action commitment,
or policy-server settings.

When `--carve-runtime` is active, each model call is also appended to
`carve_policy_calls.jsonl` beside the result JSON (or to
`--carve-trace-jsonl`). Episode aggregates and per-call traces remain separate.

## Recovery Semantics

CARVE distinguishes three operations that were previously easy to conflate:

1. **accurate replan** asks the frozen VLA for a new action chunk;
2. **prompt retry** changes semantic guidance but remains another VLA call;
3. **physical recovery** executes a bounded action-space skill, verifies a
   deployable state response, records the terminal outcome in episodic memory,
   and only then permits a fresh VLA replan.

The default controller stages repeated stalls. A first detected stall requests
an accurate replan. A confirmed repeated stall selects
`cartesian_retract_lift_reobserve` while recovery budget remains. Exhaustion
escalates to an available planner or safe stop. Direct slip, misgrasp, and
contact events may enter physical recovery immediately.

The initial physical skill supports normalized 7-D delta-Cartesian actions. It
preserves the current gripper command and executes stabilize, retract, lift,
and settle/reobserve phases. This is a conservative integration baseline, not
a claim of task-level recovery. Its deterministic LIBERO state-level smoke is
documented in `../../results/carve_recovery/PHYSICAL_RECOVERY_STATUS.md`.
Finite policy actions are projected to the skill's declared `ActionSpec`
bounds before a plan is constructed; dimensions and NaN/Inf remain hard
contract failures. This prevents recoverable model-boundary overshoot from
turning directly into safe stop while keeping every physical command bounded.

An earlier opt-in paired branch passed a compiled-BF16 intervention gate:
physical recovery plus replan succeeded and reproduced in 236 steps where an
accurate-replan branch failed at 280. This establishes intervention potential,
and the subsequent online T6 sentinel validates controller selection. The
online runner first issued an accurate replan at stall streak one, selected the
physical skill at streak two, verified its EEF response, attached the recovery
session to a fresh VLA request, and completed the task. The executor remains
opt-in so historical evaluation behavior is unchanged.

The current admitted-profile evidence is the PI0.5 Recovery Challenge under
compiled BF16 + SMVE. It restores three exact LIBERO MuJoCo states and evaluates
continuation, frequent replanning, prompt retry, and physical recovery. Both
supported stall states succeed under all four branches; the unsupported
`stale_action` state fails under VLA-only branches and causes the physical skill
to fail closed before issuing an action. Frequent replan and prompt retry use
`452` and `508` PI0.5 calls versus `113` for continuation without increasing
the `2/3` success count. The experiment therefore supports staged,
event-triggered intervention and explicit skill coverage, not a benchmark-wide
success-rate gain. A separate online T6 sentinel completes automatic monitor,
physical recovery, verification, VLA replan, and task success.

A coupled Agentic-Optimize experiment then executes the same physical-recovery
branch on the same T6/T9 states under eager BF16, admitted compiled BF16, and
admitted compiled BF16 + SMVE. All three preserve `2/2` outcomes and `2/2`
verified recoveries. Runtime P95 changes from `166.26 ms` with eager to
`65.75 ms` with compilation and `54.50 ms` with SMVE; the two admitted profiles
record zero 80 ms misses. This is the primary direct evidence that Optimize
Runtime accelerates the Agentic recovery critical path without changing the
paired recovery outcome.

## Capability Semantics

- A local OpenPI pi0.5 policy exposes mutable sample_kwargs; the adapter can
  apply a per-call flow-step budget and restores the previous value after each
  call.
- A CARVE-controlled WebSocket server advertises dynamic flow steps in its
  handshake metadata. Pi05Adapter then sends a bounded per-call step request;
  the server applies and restores `num_steps` around that call. An ordinary
  WebSocket server does not advertise this capability, so graceful mode records
  the dropped request and strict mode raises UnsupportedControlError.
- max_actions is a runtime output or commit cap and is valid for any chunked
  policy. It is not reported as native model horizon control.
- Precision denotes the precision at which the adapter was deployed. A
  per-request precision that does not match that deployment is rejected or
  explicitly dropped.
- OpenVLA exposes autoregressive decoding and exactly one 7-D action per call.
  It does not advertise flow steps, action chunks, reusable cross-call KV
  state, or proprioception. Its policy wrapper retains the RLDS gripper value;
  environment-specific binarization and sign inversion remain an explicit
  LIBERO boundary transform.

## Current Deployment Decision

1. Use compiled BF16 + SMVE when the adapter guarantees a statically padded
   camera slot. Keep ordinary compiled BF16 as the active/dynamic-view fallback
   and W8A16 as rejected closed-loop evidence.
2. Keep synchronous commit 10 as the fidelity-first runtime default.
3. Keep asynchronous prefetch experimental. Although full-duty prefetch reduces
   the ten-episode T8/T9 miss rate from `10.61%` to `1.12%`, success falls from
   `7/10` to `5/10`; an alternating duty cycle does not recover the loss.
4. Promote no asynchronous profile until observation-delay correction passes a
   new paired closed-loop gate.
5. Use event-triggered VLM planning rather than continuous VLM inference.
   Typed high-level decisions may alter only subgoals or registered
   interventions at a safe boundary. The separate 12-token
   `STATUS|FAILURE` temporal observer remains shadow-only because it did not
   pass the intervention-precision gate.
6. Treat the prior single-frame temporal-observer rollout as a co-resident systems test,
   not diagnosis evidence: it exposed injected displacement metadata to the
   prompt. The corrected temporal observer removes that leakage, but both
   tested short-label protocols fail the intervention-precision gate. Semantic
   decisions therefore remain shadow-only.
7. The real OpenVLA BF16 reference and paired INT8/NF4 gate are complete.
   CARVE preserves the single-action contract across the second model family,
   while both generic low-bit profiles are rejected: they save memory but are
   slower and fail the predeclared exact-action requirement.
8. OpenVLA compilation targets only the autoregressive language model. The
   profile aligns the action-start token and attention mask, disables CUDA
   Graphs because KV-cache outputs are reused within generation, and prewarms
   calibrated prompt-length buckets. Unseen buckets are outside the promoted
   profile and require an eager fallback.
9. All PI0.5 online Agentic experiments use deployment-admitted manifests.
   SMVE and ordinary compiled BF16 are prewarmed, hardware/checkpoint bound,
   and selected through a narrow mask-contract fallback; W8A16 and asynchronous
   prefetch remain rejected experimental profiles.
10. Keep the high-level VLM `event_only` by default. Task-start reasoning must
    be explicitly `startup_shadow` or `startup_wait`; all modes apply complete
    fail-closed semantics.
11. Report model latency, reaction latency, and task-cycle time separately.
    Planner safe-boundary wait is not VLA model latency.
12. Describe the current joint controller as recovery/mode routing. Because its
    fast and accurate configurations are presently identical, dynamic compute
    adaptation remains a future admitted-profile experiment.

SMVE profiles declare canonical view names through `StaticMaskedViewContract`;
backends resolve names to indices only after validation and assert the selected
mask is all false on every call. The contract has passed LIBERO and DROID
pi0.5 adapter checks. OpenVLA separately validates the model-neutral action,
capability, profiling, and fidelity boundary; it does not support SMVE because
its input contract contains one fused RGB tensor rather than padded camera
slots.

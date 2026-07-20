# CARVE Runtime Architecture

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

## Public Contracts

- ModelCapabilities: native backend features such as configurable flow steps,
  action chunking, KV cache, predictive context, and supported precision.
- InferenceRequest: normalized observation, instruction, episode lifecycle,
  Agentic event, and requested compute controls.
- ActionChunk: actions, backend latency, uncertainty or predictive context when
  available, and the untouched backend response.
- RuntimeTrace: requested versus applied controls, explicit fallback, queue age,
  model/runtime latency, deadline miss, and action count.
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
- RecoveryPlan and StatefulRecoveryExecutor: a bounded physical-skill lifecycle
  with explicit phases, verification, outcome memory, replan permission, and
  safe-stop semantics. A recovery decision names a concrete skill ID rather
  than treating a prompt change as physical recovery.
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

    benchmark observation
      -> InferenceRequest
      -> RiskDeadlineController
      -> CarveRuntime capability negotiation
      -> PolicyAdapter (Pi05Adapter or OpenVlaAdapter)
      -> local OpenPI/WebSocket PI0.5 or local Hugging Face OpenVLA policy
      -> ActionChunk + RuntimeTrace

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
5. Use event-triggered semantic observation rather than continuous VLM
   inference. The selected protocol is a 12-token `STATUS|FAILURE` label with
   explicit parsing; semantic output does not yet alter actions.
6. Treat the prior single-frame semantic rollout as a co-resident systems test,
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

SMVE profiles declare canonical view names through `StaticMaskedViewContract`;
backends resolve names to indices only after validation and assert the selected
mask is all false on every call. The contract has passed LIBERO and DROID
pi0.5 adapter checks. OpenVLA separately validates the model-neutral action,
capability, profiling, and fidelity boundary; it does not support SMVE because
its input contract contains one fused RGB tensor rather than padded camera
slots.

# CARVE Guarded High-Level Agent

Updated: 2026-07-24

## 1. Purpose

The original accepted CARVE recovery evidence uses a deterministic execution
monitor and controller around PI0.5. It does not contain a semantic high-level
agent. The guarded high-level agent added here closes that architectural gap
without placing a slow VLM inside the real-time control loop.

```text
task + RGB/wrist RGB + robot state + structured memory
                         |
                         v
              External multimodal planner
                         |
                 typed JSON decision
                         |
                         v
                 CARVE safety gate
       schema / confidence / budget / skill allowlist
                  /             |             \
              VLA_ACT      registered skill    SAFE_STOP
                  |             |                 |
               PI0.5      bounded executor    hold/terminate
```

The VLM is a high-level decision model. It never emits joint actions, torques,
or trajectories. PI0.5 remains the only learned low-level action generator.

The architecture is explicitly multi-rate. `ExecutionRiskMonitor` implements a
high-frequency, low-cost **Fast Execution Guard**. The external VLM is a
low-frequency semantic Critic/Planner. The monitor detects deployable symptoms;
the VLM judges task meaning and selects only a typed intervention. Neither
component is described as replacing the other.

The guard is deterministic by design. It can detect lack of physical response,
stale actions, uncertainty, and deadline risk, but it cannot determine that the
robot manipulated the wrong object or satisfied the wrong relation. Semantic
checks therefore occur at task start, selected primitive boundaries, and
abnormal execution events.

## 2. Multi-Rate State Machine

The target reusable harness has the following states:

```text
EXECUTE_FAST -> VERIFY -> RECOVER -> EXECUTE_FAST
                    \-> PLAN_AT_SAFE_BOUNDARY -> EXECUTE_FAST
                    \-> SAFE_HOLD -> STOP
```

- `EXECUTE_FAST`: use an admitted synchronous VLA profile or valid cached
  actions.
- `VERIFY`: interpret monitor evidence and decide whether execution may
  continue.
- `RECOVER`: execute one bounded registered skill and verify its response.
- `PLAN_AT_SAFE_BOUNDARY`: hold motion, invoke the VLM asynchronously, validate
  its typed result, then release only an accepted transition.
- `SAFE_HOLD`: maintain an adapter-defined stationary command and heartbeat
  until an accepted release, timeout, or terminal stop.
- `STOP`: terminate without issuing another learned or physical action.

`SAFE_HOLD` is a deployment contract, not merely an empty action queue. Robot
adapters must define the hold command, heartbeat rate, timeout, and release
condition. Simulator runners may emulate the same contract, but that emulation
must be named in traces.

## 3. Decision Contract

The planner receives only deployable evidence:

- task instruction and current subgoal;
- named camera frames;
- robot proprioception;
- risk event and measured deadline slack;
- bounded failure history and recovery memory;
- remaining retry/recovery budgets;
- the current registered-skill allowlist.

It can return four intents:

| Intent | Meaning |
|---|---|
| `continue` | Keep the current execution objective. |
| `vla_act` | Call the frozen VLA with a validated local instruction. |
| `run_skill` | Invoke one skill from the current allowlist. |
| `safe_stop` | Stop when no accepted intervention remains. |

Unknown fields, raw action arrays, unknown skills, malformed JSON, low-confidence
interventions, endpoint errors, and exhausted call budgets are rejected. The
default fallback is `safe_stop`.

A primitive is one bounded operation such as a frozen-VLA contact attempt or a
registered Cartesian recovery skill. When it returns, `PrimitiveOutcome`
records `succeeded`, `failed`, `timeout`, or `interrupted` together with its
expected and observed symbolic outcomes. `PrimitiveBoundaryPolicy` supports
`event_only`, `selective`, and `every_primitive` VLM scheduling.

## 4. Runtime Placement

The deterministic controller remains responsible for every fast control step.
The default planner policy is `event_only`: the VLM is invoked only after the
controller reaches a semantic escalation boundary. Optional task-start modes
must be explicit:

1. `startup_shadow` records a decision but cannot alter execution; or
2. `startup_wait` holds before motion and applies the complete guarded decision
   semantics, including rejection and `safe_stop`.

An advisory task-start path that applies only `vla_act` but ignores rejection or
`safe_stop` is not an admitted deployment mode.

This separation is deliberate. The high-level model may take seconds, whereas
PI0.5 and the control runtime are evaluated against millisecond-scale deadlines.

The framework therefore exposes an `AsyncGuardedHighLevelAgent` lifecycle. It
allows at most one in-flight planner request, retains the typed request context
in a ticket, and exposes the completed result only when the harness polls it at
a safe execution boundary. The wrapper does not execute a VLM decision by
itself.

`AsyncAgenticHarnessController` is now the canonical model-independent
state-machine implementation. The online LIBERO runner exposes the same
task-start policies but still contains benchmark-specific orchestration rather
than delegating every transition to this controller:

1. `event_only` is the default and performs no task-start VLM call.
   `startup_shadow` records but never applies the result. `startup_wait` enters
   `SAFE_HOLD` and applies only a complete accepted decision.
2. After bounded recovery is exhausted, the runner clears the action chunk and
   enters an explicit safe planner boundary. It waits at most
   `--carve-high-level-boundary-timeout-sec` for one typed decision. Timeout,
   malformed output, a rejected decision, or `safe_stop` terminates the episode
   fail-closed.
3. Each submission, completion, source/collection timestep, ticket ID, timeout,
   decision application, and stale result is written to the episode trace. Safe
   boundary wait is reported separately from VLA control latency.

The historical direct-call path remains only for the recorded-frame integration
smoke and must not be reported as a real-time closed-loop planner.

## 5. Memory and Skill Contracts

Recovery memory is structured evidence rather than an unbounded conversation
history. `FailureEpisodeRecord` binds:

- episode/context fingerprint, task, and current subgoal;
- VLA checkpoint and admitted deployment profile;
- failure type, monitor evidence, and action age;
- attempted intervention and consumed retry/recovery budget;
- verification result and terminal outcome;
- confidence, timestamp, and expiry.

`FailureMemory` performs bounded, newest-first retrieval with context, failure,
profile, limit, and expiry filters. Retrieval may change planner context,
suppress a failed intervention, or escalate to safe hold. Memory never issues
robot actions. The episode controller injects matching records into the typed
planner context immediately before submission, so retrieval is active in the
decision path rather than an unattached logging utility.

Each physical skill declares an operating envelope: action specification,
preconditions, supported failures, timeout, maximum actions, verification rule,
and safe-hold behavior. `RecoveryPlan` enforces this envelope. The current
`cartesian_retract_lift_reobserve` executor is one registered skill, not
evidence of a broad skill library.

## 6. Implemented Files

- `agentic_vla/runtime/agent.py`: contracts, parser, safety gate, OpenAI-compatible
  multimodal client, controller bridge, and asynchronous single-flight planner
  lifecycle.
- `agentic_vla/runtime/harness.py`: episode-scoped state machine, explicit
  task-start policies, independent budgets, stale-result rejection, and
  adapter-level `SAFE_HOLD`.
- `agentic_vla/runtime/recovery.py`: bounded physical-skill envelopes,
  verification outcomes, and typed expiring failure memory.
- `agentic_vla/optimization/contracts.py` and `admission.py`: independently
  admitted VLM Planner deployment profiles.
- `scripts/run_agentic_vla_libero.py`: asynchronous task-start and
  repeated-failure integration, safe-boundary timeout, and unified episode trace.
- `scripts/smoke_carve_unified_agentic_runtime.py`: canonical state-machine
  smoke with a real VLM endpoint and admitted PI0.5 service.
- `scripts/smoke_carve_high_level_agent.py`: reproducible recorded-frame smoke.
- `tests/test_carve_high_level_agent.py`: CPU safety and integration tests.

Enable the path with:

```bash
--carve-runtime \
--carve-joint-controller \
--carve-high-level-agent \
--carve-high-level-boundary-timeout-sec 10
```

The new path and the legacy `--agentic-planner` path are mutually exclusive.

## 7. Verified Status

On 2026-07-24, the local Qwen3.5-4B BF16 service processed a recorded LIBERO
Task 9 frame through the asynchronous lifecycle. The immediate non-blocking
poll returned no result; the completed guarded decision was accepted:

```json
{
  "intent": "vla_act",
  "subgoal": "grasp yellow mug",
  "vla_instruction": "grasp the yellow mug",
  "confidence": 0.9
}
```

Measured end-to-end VLM latency for this smoke was `7248.07 ms`. This is
evidence that the real multimodal model, image path, JSON contract, and safety
gate execute successfully without blocking the caller. It is not closed-loop
manipulation-success evidence. The machine-readable audit is
[`../../results/carve_framework_freeze_20260724/async_planner_lifecycle_smoke.json`](../../results/carve_framework_freeze_20260724/async_planner_lifecycle_smoke.json).

The canonical controller subsequently passed a unified real-service smoke. It
entered `SAFE_HOLD`, accepted a typed Qwen3.5-4B `vla_act` decision after
`5603.02 ms`, released the hold, and obtained a `10 x 7` action chunk from the
promoted PI0.5 SMVE profile. PI0.5 runtime/reaction latency was
`62.85/62.85 ms`, with no 80 ms deadline miss. The receipt is
[`../../results/carve_framework_freeze_20260724/unified_real_service_smoke.json`](../../results/carve_framework_freeze_20260724/unified_real_service_smoke.json).
This smoke issues no simulator or robot action and therefore makes no task
success claim.

The runtime, high-level-agent, optimization, recovery, and memory contracts
currently pass 98 CPU conformance tests.

## 8. Remaining Validation Before a Paper Claim

The reusable framework core is frozen. Broad benchmark evaluation should still
wait until the benchmark runner delegates its transitions to the canonical
controller and a minimal paired gate is retained:

1. deterministic CARVE without the high-level VLM;
2. deterministic CARVE plus event-triggered VLM escalation;
3. the same initial state with VLM escalation disabled.

Use the same initial states and fixed PI0.5 profile. Report task outcome,
accepted/rejected agent decisions, VLM calls, safe-boundary wait, VLA P95,
control deadline misses, GPU memory, and all fallbacks. The VLM's seconds-scale
latency must not be folded into the VLA control-loop latency. Only after this
gate passes should the VLM decision path be promoted from framework-complete to
closed-loop validated. The historical one-episode LIBERO integration trace
demonstrates service interoperability, but it predates the canonical
state-machine freeze and is not a substitute for this paired gate.

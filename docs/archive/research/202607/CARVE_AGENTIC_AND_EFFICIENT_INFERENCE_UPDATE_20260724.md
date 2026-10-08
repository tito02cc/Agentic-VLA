# CARVE Agentic and Efficient-Inference Research Update

Date: 2026-07-24

## 1. Executive Decision

The project direction remains valid, but the claim must be more precise than
"an Agentic VLA harness plus model acceleration."

The recommended research question is:

> Can a multi-rate, guarded runtime improve the recoverability of a frozen VLA
> while admitting only inference profiles that preserve action and closed-loop
> behavior under a robot reaction-time budget?

This framing keeps the two project layers coherent:

- **CARVE Agentic Harness** decides when execution may continue, must be
  verified, requires bounded recovery, needs semantic replanning, or must stop.
- **CARVE Optimize Runtime** prepares and admits VLA/VLM deployment profiles
  using latency, memory, action-fidelity, contention, and closed-loop evidence.

Compilation, quantization, a VLM planner, memory, retry, and recovery are
components. The contribution is the guarded multi-rate orchestration and the
evidence-gated coupling between Agentic intervention and inference deployment.

## 2. Latest Agentic-VLA Findings

### 2.1 Closest Work

| Work | Main finding | Consequence for CARVE |
|---|---|---|
| [Harness VLA](https://arxiv.org/abs/2607.08448) | Treats a frozen VLA as a retryable contact-rich primitive and combines it with a fixed analytic skill library, execution memory, failure models, and planner-side re-staging. | Strongly validates the frozen-VLA plus external-harness direction. CARVE must not claim that this decomposition alone is novel. |
| [HELM](https://arxiv.org/abs/2604.18791) | Attributes long-horizon failure to memory, verification, and recovery gaps; uses keyframe episodic memory, a learned state verifier, rollback, and replanning. | A generic memory buffer or verifier is no longer sufficient novelty. CARVE should keep memory structured and use it to govern intervention, not claim a new learned verifier. |
| [Guava](https://arxiv.org/abs/2606.18363) | Identifies iterative perception-reasoning-action loops, semantic action abstractions, and multimodal observations as core harness ingredients. | CARVE should expose typed semantic intents and skill contracts rather than allow free-form VLM actions. |
| [CoVer](https://arxiv.org/abs/2602.12281) | Uses test-time instruction/action candidate generation and a learned verifier to close the intention-action gap. | Verification is a strong baseline category. CARVE's safety gate is deployment governance, not a replacement for a trained semantic verifier. |
| [MemoryVLA](https://shihao1895.github.io/MemoryVLA/) | Adds explicit perceptual-cognitive working memory to VLA execution. | Memory must have an active, measurable decision role to become a paper claim. |

### 2.2 Required Agentic Design

CARVE should use a **multi-rate Agentic state machine**:

```text
EXECUTE_FAST
    | deployable anomaly or subgoal boundary
    v
VERIFY
    | physical failure              | semantic ambiguity
    v                               v
RECOVER                      PLAN_AT_SAFE_BOUNDARY
    | verified response              |
    +-------------> EXECUTE_FAST <---+
                    |
             unsupported / timeout
                    v
                SAFE_HOLD
                    |
            explicit release or STOP
```

The state machine separates two roles:

1. **Execution Risk Monitor:** high-frequency, low-cost, deterministic
   processing of proprioception, action age, visual response summaries,
   uncertainty when available, and deadline slack.
2. **VLM Critic/Planner:** low-frequency semantic assessment from images,
   instruction, subgoal, monitor evidence, and bounded memory. It cannot emit
   robot actions.

The external VLM should default to `event_only`. Two task-start policies may be
supported explicitly:

- `startup_shadow`: collect a task-start semantic decision but do not alter
  execution;
- `startup_wait`: wait before motion and fail closed if planning fails.

An advisory task-start result must not silently apply only one intent while
ignoring `safe_stop` or a rejected decision. Each policy needs one complete,
traceable semantics.

### 2.3 Memory and Skill Contracts

Each failure-memory entry should contain:

- observation/context fingerprint and current subgoal;
- VLA and deployment-profile identity;
- failure type and monitor evidence;
- attempted intervention and consumed budget;
- verification result and terminal task outcome;
- confidence, timestamp, and expiry.

Memory may change planner context, suppress a previously failed intervention,
or trigger escalation. It must never issue an action directly.

Each physical skill should declare a capability envelope:

- action-space and robot preconditions;
- timeout and maximum action count;
- supported failure types;
- verification rule;
- retry/recovery budget;
- safe-hold behavior when the envelope is violated.

The current retract/lift/reobserve skill is one registered skill, not yet a
general skill library.

## 3. Latest Efficient-Inference Findings

### 3.1 Realtime Is More Than Model Latency

[Realtime-VLA](https://arxiv.org/abs/2510.26742) demonstrates that kernel and
graph-level optimization can make pi0-class inference run at real-time rates.
[FASTER](https://arxiv.org/abs/2603.19199) shows that reaction time depends
jointly on time to first action and the execution horizon, not only total chunk
generation latency. [Realtime-VLA V2](https://arxiv.org/abs/2603.26360) further
extends the deployment problem to calibration, planning, control, and learned
execution-speed selection.

CARVE should therefore report three distinct clocks:

1. **Model latency:** preprocessing, vision encoder, VLM prefix, action expert
   or autoregressive decode, and transport.
2. **Reaction latency:** observation capture to first newly conditioned action,
   including queue age and the remaining committed horizon.
3. **Task-cycle time:** sensing, Agentic decision, model inference, simulator or
   robot control, and safe-boundary waits.

P50/P95/P99, deadline misses, cold start, warm start, and peak VRAM must be
reported separately. A seconds-scale VLM safe-boundary wait must not be folded
into millisecond VLA model latency.

### 3.2 Streaming and Asynchrony

[VLA-RAIL](https://arxiv.org/abs/2512.24673) uses asynchronous inference,
trajectory smoothing, and chunk fusion. [Reflex](https://arxiv.org/abs/2607.14695)
combines static/sliding/dynamic attention regions with an asynchronous
vision-action pipeline and operator fusion.

These works make streaming a relevant backend direction, but they do not
justify promoting CARVE's existing prefetch path. CARVE's ten-episode expansion
reduced deadline misses but degraded success, so it remains rejected. Any new
streaming backend must pass:

- returned-chunk freshness and alignment checks;
- motion continuity checks;
- matched closed-loop non-inferiority;
- invalidation on monitor risk, semantic escalation, and recovery.

### 3.3 Quantization and Portable Deployment

[QVLA](https://arxiv.org/abs/2602.03782) already introduces
action-sensitivity-guided mixed-precision VLA quantization. CARVE must therefore
not claim generic "action-aware quantization" as a new method. Its defensible
contribution is an **admission protocol**: a low-bit profile is usable only
after memory, latency, action-fidelity, and paired closed-loop gates pass.

The current PI0.5 W8A16 result should remain a negative result: it reduces peak
VRAM but does not improve latency over compiled BF16 and loses a matched
recovery outcome. It must not be promoted simply to make the lightweight story
look complete.

[vla.cpp](https://arxiv.org/abs/2606.08094) shows that a portable C++/ggml
runtime can support multiple VLA architectures and embedded hardware.
Portability is therefore a future backend integration target, not a standalone
CARVE novelty claim.

### 3.4 New Planner Deployment Profile

The external VLM is currently the largest unoptimized Agentic component: the
real Qwen3.5-4B lifecycle smoke takes seconds and occupies substantial GPU
memory. CARVE should add a separate **PlannerProfile**, independent from the
VLA profile:

| Gate | Required measurement |
|---|---|
| Semantic validity | JSON/schema-valid rate and intent validity |
| Decision fidelity | agreement with BF16 decisions and manually labeled critical cases |
| Latency | cold/warm P50/P95 and timeout rate |
| Capacity | model size and peak/resident VRAM |
| Contention | VLA P95 and deadline misses while the planner runs |
| Safety | unsafe-intervention and fail-closed rates |

The first comparison should be BF16 versus practical INT4 weight-only
deployment of the same VLM. This is a systems optimization experiment, not a
new VLM quantization algorithm.

### 3.5 Profiling Structure

[VLA-Perf](https://github.com/NVlabs/vla-perf) models vision, VLM, action-head,
hardware, network, and deployment choices separately. CARVE should mirror this
stage separation in measured traces:

```text
capture -> preprocess -> vision -> VLM prefix -> action expert/decode
        -> serialization/transport -> queue/fusion -> robot control
```

Optional Realtime-VLA Triton kernels can later be exposed as another backend
plugin. They should be adopted only after checkpoint compatibility and CARVE's
action/closed-loop admission gates are demonstrated.

## 4. Novelty Boundary After This Update

### Defensible Main Contributions

1. A **multi-rate guarded Agentic runtime** that separates high-frequency
   execution-risk monitoring from event-triggered semantic verification and
   only applies VLM decisions at explicit safe boundaries.
2. An **evidence-gated VLA/VLM deployment runtime** that binds optimization
   profiles to model, observation, action, hardware, fidelity, realtime, and
   closed-loop contracts.
3. A **co-resident Agentic-VLA scheduler and trace contract** that measures and
   controls the contention created by semantic reasoning around a VLA.
4. PI0.5-specific SMVE as one scoped optimization enabled by the general
   profile contract, not as a universal VLA transform.

### Components, Not Standalone Novelty

- external VLM planning;
- retry, critic, memory, and safe stop;
- `torch.compile`;
- weight-only quantization;
- generic asynchronous inference;
- a generic model adapter or benchmark harness.

### Out of Scope

- parameter-level MoE;
- pruning;
- a new WAM;
- a new foundation model;
- full benchmark leaderboard sweeps;
- a new quantization algorithm without a mechanism beyond QVLA.

## 5. Gap Analysis Against the Current Repository

| Priority | Status | Gap or remaining change |
|---|---|---|
| P0 | Complete | Reusable `AsyncAgenticHarnessController` now owns the episode-scoped state machine. |
| P0 | Complete | Planner ownership is episode-scoped and stale episode/ticket generations are rejected. |
| P0 | Complete | `event_only`, `startup_shadow`, and `startup_wait` have explicit tested semantics. |
| P0 | Complete | `SAFE_HOLD` defines enter, heartbeat, timeout, release, and stop behavior. |
| P0 | Complete | Failure streak, recovery attempts, planner calls, and semantic retries are independent counters. |
| P0 | Bounded claim | Fast and accurate routes still share one admitted compute profile; do not claim online flow-step adaptation. |
| P0 | Service gate complete | Real VLM and PI0.5 passed one canonical unified state-machine smoke; migrate it into a closed-loop simulator runner and run the paired recovery gate. |
| P1 | Complete | `FailureEpisodeRecord` and `FailureMemory` add typed context, evidence, profile, budget, verification, confidence, and expiry. |
| P1 | Interface complete | Runtime traces support model, reaction, task-cycle, queue-age, action-age, and stage clocks; populate cold-start and stage fields in formal runs. |
| P1 | Interface complete | Planner profile and admission contracts exist; BF16/INT4 benchmark evidence is still required. |
| P2 | Deferred | Evaluate Realtime-VLA/Triton only as a later backend plugin. |

## 6. Frozen Next Order

1. Migrate the benchmark adapter to the frozen canonical controller.
2. Run the exact-state paired recovery gate under the promoted synchronous
   PI0.5 profile.
3. Measure BF16 versus INT4 external VLM planner profiles under co-resident
   load.
4. Populate stage-level, reaction, and task-cycle metrics in that run.
5. Only then consider a small LIBERO-Plus or LIBERO-Pro transfer subset.

Do not reopen W8A16, asynchronous prefetch, OpenVLA, MoE, pruning, or a broad
benchmark sweep before these gates are complete.

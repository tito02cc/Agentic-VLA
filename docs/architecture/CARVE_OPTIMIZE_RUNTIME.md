# CARVE Optimize Runtime

Updated: 2026-08-26

## Purpose

CARVE Optimize Runtime prepares and validates efficient VLA deployment profiles
without duplicating the rollout logic in `CarveRuntime`. It is independently
usable from CARVE Agentic Harness and communicates with the Harness only through
the existing inference, action, capability, trace, and Agentic-event contracts.

The runtime optimizes an embodied system, not only one `sample_actions` call.
It therefore distinguishes three clocks:

1. **model latency:** preprocessing, vision, VLM prefix, action expert/decode,
   and transport;
2. **reaction latency:** observation capture to the first action conditioned on
   that observation, including queue age and remaining commit horizon;
3. **task-cycle time:** sensing, Agentic routing, model calls, controller or
   simulator steps, and safe-boundary waits.

Cold start, prewarm, steady-state P50/P95/P99, deadline misses, and peak VRAM
are reported separately.

## Current Package

`agentic_vla/optimization/` currently provides:

- `OptimizationProfile`: backend, deployment precision, module precision plan,
  flow steps, committed action horizon, and backend options.
- `StaticMaskedViewContract`: canonical named view order and statically padded
  views, resolved to backend indices only after contract validation.
- `ModelPlugin`: model-family discovery and component-group validation.
- `BackendPlugin`: preparation of one executable optimization backend.
- `CarveOptimizeRuntime`: resolves plugins and prepares a profile while
  preserving the declared action contract.
- `ActionFidelityVerifier`: first-action, chunk, cosine, gripper, endpoint, and
  jerk comparisons in action space.
- `BenchmarkRunner`: measures the normal CARVE deployment path after warmup.
- `ProfileManifest`: binds a calibrated profile to a checkpoint and hardware
  identity.
- `ProfileAdmissionDecision`: separates calibration artifacts from profiles
  promoted by replay-fidelity, realtime, and closed-loop gates.
- `PlannerOptimizationProfile` and `PlannerBenchmarkReport`: bind the external
  VLM, precision/backend, prompt/schema version, token budget, latency, VRAM,
  validity, agreement, contention, and unsafe-intervention evidence.
- `PlannerAdmissionDecision`: rejects planner profiles that violate validity,
  agreement, timeout, co-resident VLA deadline, or safety thresholds.
- `EventCoherentReuseGate`: a model-neutral, fail-closed invalidation protocol
  for action queues and future visual-prefix or action-warm-start backends. It
  binds reuse to deployment profile, instruction, subgoal, planner epoch,
  recovery epoch, physical risk, scene change, proprioceptive change, age, and
  backend-specific similarity evidence.
- `ContractFallbackPolicy`: retries only a violated static masked-view contract
  on an independently admitted compiled-BF16 fallback.
- `RuntimeTrace`: records queue age, model/runtime/reaction latency, optional
  task-cycle time, action age, stage latencies, deadline behavior, and profile
  receipts.

The current plugins are `Pi05ModelPlugin`, the zero-transform `EagerBackend`,
`TorchCompileBackend`, `MaskedViewElisionBackend`, and `TorchAOInt8Backend`.
The eager backend is the behavioral reference. The compile backend copies the
policy boundary and compiles only `sample_actions`. The masked-view backend
asserts and eliminates statically padded camera slots before visual embedding
and prefix-KV construction, then compiles the shortened sampler. The TorchAO
backend applies component-scoped W8A16 in place and then compiles the same
action-sampling path. Every backend rejects unsupported or unapplied
transformations.

## Revised Optimization Stack

SMVE is useful, measured engineering, but it is not the complete efficient-
inference method. CARVE now treats it as the deterministic first pass in a
layered runtime:

| Layer | Mechanism | Current state | Claim boundary |
|---|---|---|---|
| L0 input compaction | SMVE removes adapter-guaranteed padding views before SigLIP | implemented and admitted for two pi0.5 adapters | narrow structural optimization |
| L1 execution backend | eager, `torch.compile`, then optimized Triton/C++ backends | eager and compile implemented; Realtime-VLA/vla.cpp integration pending | backend engineering, not CARVE algorithmic novelty |
| L2 temporal reuse | action queue, visual-prefix cache, or action warm start | action queue implemented; reusable backend protocol begins with `EventCoherentReuseGate` | cache algorithms remain replaceable plugins |
| L3 adaptive compute | admitted flow steps, action commitment, refresh policy | static two-step profile admitted; online adaptation not yet claimed | every branch requires independent admission |
| L4 compression | component precision and low-bit deployment | Planner NF4 and PI0.5 group-wise INT8 evaluated; PI0.5 late-language INT8 admitted as a low-memory tier | use ActQuant/QVLA-class backends rather than claim generic quantization novelty |
| L5 multi-model scheduling | event-triggered VLM plus deadline-aware VLA | co-resident contention measured; VLM remains low-rate | optimize system reaction time, not only kernel latency |

The CARVE-specific research mechanism is **event-coherent hierarchical reuse**:
an optimization backend may propose a reusable artifact, but the Harness owns
the evidence that can invalidate it. A task or subgoal change, a new VLM planner
decision, physical recovery, monitor event, elevated risk, deployment-profile
change, excessive visual/proprioceptive drift, or cache age forces fresh
inference. This separates two responsibilities:

1. a backend decides how to cache or accelerate computation;
2. CARVE decides whether that computation is still coherent with the current
   Agentic and physical execution state.

`ReuseCandidate`, `ReuseContext`, `ReuseDecision`, and
`EventCoherentReuseGate` implement this boundary. The current action-queue
controller consumes the gate as `reuse_permitted`; rejected reuse enters a
fresh VLA call and leaves an explicit decision reason. Visual-prefix and
action-warm-start plugins are not yet implemented and must not be reported as
experimental results.

## Optimization Admission Rule

CARVE treats quantization as a deployment candidate, not an automatic
improvement. A profile can be selected only after three gates pass on the same
checkpoint, adapter, observation contract, flow-step count, and action horizon:

1. **Replay action fidelity:** first action, action chunk, endpoint, gripper,
   and jerk remain within the predeclared tolerance on paired observations.
2. **Warm runtime behavior:** P50/P95/P99, deadline misses, and peak VRAM are
   measured after prewarm with no unrelated GPU workload.
3. **Matched closed-loop behavior:** exact restored states retain the accepted
   task/recovery outcome. A profile that is only faster or only smaller is not
   promoted.

This rule is particularly important for flow-matching VLAs: component-scoped
W8A16 may preserve most open-loop actions yet still alter a long-horizon
recovery. CARVE therefore promotes compiled BF16 + SMVE for PI0.5 on the tested
RTX 4090 contract, retains compiled BF16 as the dynamic-view fallback, and
admits late-language INT8 only on its validated Torch 2.7.1/TorchAO 0.15
software identity. The same logical INT8 profile is rejected on the current
Torch 2.12 stack. This is a reproducible version boundary, not an omitted
ablation.

The same principle applies to a separate semantic planner. A `PlannerProfile`
must bind the external VLM checkpoint, precision, serving backend, hardware,
prompt/schema version, and token budget. Admission requires:

- schema-valid and intent-valid output rates;
- agreement with the BF16 reference and critical-case labels;
- cold/warm latency and timeout rate;
- resident and peak VRAM;
- co-resident VLA P95 and deadline misses;
- unsafe-intervention and fail-closed rates.

Reference agreement is diagnostic rather than an oracle: a larger BF16 model
can also make an unsafe semantic decision. Ground-truth critical-event recall,
no-op false-positive rate, typed-schema validity, and unsafe intervention rate
therefore determine semantic admission; BF16 agreement is reported only as a
paired stability measure.

## Role-Aware Heterogeneous Quantization

Uniform low-bit conversion is a baseline, not the CARVE quantization method.
The Planner and VLA have different failure surfaces: Planner degradation
changes a discrete semantic/tool decision, while small VLA errors accumulate
inside continuous action chunks. CARVE consequently calibrates and admits them
separately, then admits their shared-GPU composition.

### Planner VLM

The Planner is low-rate, batch-one and schema constrained. The first local
study uses Qwen3.5-9B with four profiles:

| Profile | Precision boundary | Purpose |
|---|---|---|
| P0 Qwen3.5-4B BF16 | existing full-precision local service | resident-memory and semantic baseline |
| P1 Qwen3.5-9B BF16 | full model BF16 | semantic-capacity upper bound; not assumed deployable with PI0.5 |
| P2 Qwen3.5-9B NF4 | all eligible linear weights in NF4/BF16 compute | uniform 4-bit compression baseline |
| P3 Qwen3.5-9B VP-NF4 | language decoder linear weights in NF4; vision tower/merger, embeddings, norms and decision head remain BF16 | tests whether preserving the visual stack recovers a semantic or latency advantage |

`NF4` is reported as NF4 weight-only quantization, not called integer INT4.
An AWQ/GPTQ W4A16 profile may be added only when its serving kernels support the
Qwen multimodal architecture and demonstrate a real P95 improvement. A smaller
checkpoint alone is insufficient.

The local 9B checkpoint contains approximately 9.65B parameters: 6.92B in
language layers, 0.46B in the vision stack, and 2.03B in untied input/output
embeddings. This makes indiscriminate quantization unnecessary and also exposes
a complementary runtime opportunity. Offline tokenizer verification confirms
that `N/A/G/D/C/S/O`, both bare and with a leading space, each map to one token.
For this one-token semantic event schema, CARVE may replace the 248,320-way
output projection with an exact 14-row allowlisted decision head. This is
recorded as **schema-constrained decoding**, not as a quantization result, and
the unrestricted head remains the fallback for full JSON tool calls.

Planner calibration has two held-out layers:

1. semantic-event replay: protocol validity, no-op false positives, mild/severe
   anomaly recall, physical-failure recall, latency and memory;
2. full Planner replay: typed tool-call validity, valid intent, critical-case
   correctness, timeout/fail-closed behavior and unsafe intervention.

Short prompts and one-token decisions make KV-cache compression a low-priority
ablation. It is evaluated only if measured KV memory is material; FP8 KV cache
is not presented as a core contribution by default.

The final direct 4B precision ablation uses separate identifiers to avoid
confusion with the earlier 4B/9B capacity study:

| Profile | Precision | Allocated VRAM | P95 | Semantic gate | Decision |
|---|---|---:|---:|---:|---|
| Q0 | BF16 | 8.46 GiB | 1.90 s | pass | latency default |
| Q1 | INT8 | 4.84 GiB | 6.50 s | pass | rejected on current kernel |
| Q2 | NF4 | 3.08 GiB | 2.91 s | pass | memory candidate |

All three profiles produce 100% paired semantic-decision agreement on the 30
temporal observations. Q2 reduces allocated model memory by 63.6% relative to
Q0, but it is a capacity profile rather than a latency optimization. It is not
system-admitted with a VLA until the co-resident deadline gate also passes.

### PI0.5 VLA

PI0.5 does not begin with whole-model INT4. The first candidate is component-
scoped INT8 weight-only quantization, with the action expert, action/time
projections, final normalization and output path retained in BF16. The search
order is:

1. quantify one component group at a time using fixed observations and fixed
   flow noise;
2. reject a group immediately when worst-case action fidelity fails;
3. combine only individually safe groups and rerun the same paired replay;
4. benchmark warm latency and peak VRAM only after fidelity passes;
5. promote only after matched closed-loop non-inferiority.

The initial group sweep covers early, middle and late VLM language blocks and
the vision encoder independently. The action expert and action projections are
protected in the first sweep. Existing evidence already shows why this order is
needed: quantizing the first four VLM layers passed 45 replay observations but
lost a matched long-horizon recovery, while broader language quantization
failed the endpoint-drift gate. Those candidates remain negative evidence.

The action-fidelity receipt records first-action MAE, chunk MAE/RMSE/cosine,
gripper agreement, integrated endpoint drift and jerk RMSE. It must also bind
the observation set, instruction, fixed-noise seed/tensor, inference steps,
action horizon and BF16 reference hash. Low-bit PI0.5 experiments below INT8
are deferred until this group-wise INT8 path passes closed loop. ActQuant- or
QVLA-class action-aware backends can later plug into the same receipt and
admission protocol; CARVE does not duplicate their per-tensor or per-channel
quantizers.

### Shared-GPU Composition

The deployable unit is a pair, not either quantized model alone:

```text
PlannerProfile + VLA Profile + scheduler policy
                       |
                       v
             co-resident admission
 semantic gate | action fidelity | peak VRAM | VLA deadline | fallback
```

The admitted reference pair is P0 Qwen3.5-4B BF16 plus the promoted BF16/SMVE
PI0.5 profile. P3 is not composed by default because it provides no semantic or
latency advantage over P0. A selective-INT8 PI0.5 replaces BF16 only if its
independent action and closed-loop gates pass. Heavy Planner and VLA kernels
are serialized at safe primitive boundaries; the high-frequency execution
guard remains model-free.

Every quantization receipt distinguishes checkpoint bytes, resident model VRAM,
peak inference VRAM and co-resident peak VRAM. It also records backend/kernel,
module include and exclude rules, compute dtype, calibration split, semantic or
action-fidelity metrics, P50/P95/P99, deadline misses and the independently
admitted fallback. This prevents a nominal low-bit label from hiding BF16
modules, backend overhead or a slower kernel.

### Minimal Experiment Matrix

The quantization study stops at the following ten profiles unless a gate
reveals a specific unresolved question:

1. P0 Planner 4B BF16;
2. P1 Planner 9B BF16;
3. P2 Planner 9B uniform NF4;
4. P3 Planner 9B vision-preserving NF4;
5. PI0.5 compiled BF16 + SMVE reference;
6. PI0.5 language-group INT8 candidates;
7. PI0.5 vision INT8 candidate;
8. P0 + PI0.5 reference co-resident pair;
9. P3 + PI0.5 reference pair only if a new backend improves P3 independently;
10. P0 + selective-INT8 PI0.5 pair, only if profile 6 or 7 is promoted.

This matrix answers three separate questions: whether a larger quantized
Planner is semantically preferable to the smaller BF16 Planner, whether PI0.5
can shed memory without action drift, and whether the admitted combination
actually improves a 24-GB deployment.

### Current Planner Gate Result

The P0-P3 comparison is complete on 30 paired temporal observations using the
same `code_v5` prompt. A deployable local RGB-difference overlay supplies
high-frequency change evidence; the VLM emits one short evidence line and a
typed `KEEP` or `REFRESH` decision. No simulator displacement values or
evaluator labels reach the Planner.

| Profile | Allocated VRAM | Semantic P95 | No-op FP | Severe recall | Decision |
|---|---:|---:|---:|---:|---|
| P0 Qwen3.5-4B BF16 | 8.46 GiB | 1939.7 ms | 0% | 100% | retained default |
| P1 Qwen3.5-9B BF16 | 17.53 GiB | 2040.8 ms | 0% | 100% | rejected for 24-GB co-residency without semantic gain |
| P2 Qwen3.5-9B uniform NF4 | 7.35 GiB | 2879.7 ms | 0% | 100% | memory candidate only |
| P3 Qwen3.5-9B vision-preserving NF4 | 7.97 GiB | 2834.7 ms | 0% | 100% | ablation only |

NF4 reduces 9B allocated model memory by more than half, but the current
Transformers/BitsAndBytes path increases P95 and provides no semantic gain over
4B BF16. It is therefore not promoted or described as realtime acceleration.
P0 remains the default; P2 remains available only when an otherwise admitted
co-resident pair cannot meet its memory budget. The complete machine-readable
gate is in `results/carve_quantization/planner_quantization_gate.json`.

### Current PI0.5 INT8 Gate Result

The component-wise sensitivity sweep is complete for the protected-action-path
search space. Each profile uses the same 45 observations, deterministic flow
noise, two flow steps and ten-action horizon. Candidates that fail the
worst-case endpoint-L2 limit of `0.10` are rejected before latency measurement.

| INT8 scope | Endpoint L2 | P95 | Peak VRAM | Decision |
|---|---:|---:|---:|---|
| VLM layers 0--3 | 0.09359 | 71.72 ms | 6.56 GB | replay pass, rejected by prior paired closed-loop regression |
| VLM layers 0--5 | 0.10433 | not run | not run | rejected at fidelity gate |
| VLM middle blocks | 0.11873 | not run | not run | rejected at fidelity gate |
| VLM late blocks | 0.08743 | 71.61 ms | 6.36 GB | admitted low-memory tier after matched T6/T9 success |
| vision encoder | 0.22196 | not run | not run | rejected at fidelity gate |

The result supports heterogeneous, action-aware precision boundaries rather
than whole-model INT8. The late-language profile is deployable as a
version-bound low-memory tier but remains slower than SMVE. The machine-readable sensitivity gate is in
`results/carve_optimize/pi05_int8_sensitivity_gate.json`.

CARVE also composes late-language INT8 with SMVE as an explicit backend. The
composition passes all 45 replay observations and reaches P95 `58.71 ms` with
`6.36 GB` peak VRAM, but it fails the matched T9 recovery that both components
pass independently. It is therefore rejected. This is direct evidence that
component admission does not imply composition admission. The paired T6/T9
gate is in `results/carve_optimize/pi05_quantized_closed_loop_gate.json`.

### Current Shared-GPU Admission

The P0 + PI0.5 SMVE pair is now represented by a formal system receipt. Under
the existing 500-call continuous-VLM stress condition it uses `17.58 GB`, has
VLA P95 `75.57 ms`, and misses the 80 ms deadline on `0.6%` of calls. The
Planner completes every measured request. The pair passes the fixed 1% system
deadline/timeout limits. This is a warm runtime admission result, not a task-
success claim. See
`results/carve_optimize/deployment/p0_smve_system_admission.json`.

The admitted standalone late-INT8 VLA does not pass the same shared-GPU system
gate. With P0 continuously active, VLA P95 is `87.68 ms` and the 80 ms miss
rate is `95.4%`; a five-second Planner cooldown reduces these to `85.34 ms`
and `19.4%`, still above the 1% limit. The profile therefore remains a
standalone/separate-accelerator low-memory tier. See
`results/carve_optimize/p0_late_int8_system_gate.json`.

The complete 2026-08-26 replay of five PI0.5 profiles and the direct Q0-Q2
Planner comparison is in `results/carve_efficiency_full_20260826/`. It records
the current Torch 2.12 late-INT8 regression explicitly. A real co-resident
Q2+SMVE dry-run subsequently passed with `11.22 GiB` combined service-process
memory, one accepted Planner decision and one `66.69 ms` VLA call. This proves
integration feasibility, but repeated contention and closed-loop gates remain
required for full system admission.

## Preparation Path

```text
PolicyAdapter + OptimizationProfile
             |
             v
    CarveOptimizeRuntime
      | model resolution
      | backend resolution
      | capability validation
      | action-contract preservation
             v
       PreparedPolicy
             |
             v
        CarveRuntime
```

Optimizations are deployment-time transforms. Request-specific deadlines remain
online controls, while flow steps and committed action horizons are overlaid
from the prevalidated profile.

## pi0.5 Replay Benchmark

`scripts/benchmark_carve_pi05_profile.py` loads recorded deployable observations
from failure snapshots, encodes them through either the LIBERO or DROID input
schema, derives fixed-noise dimensions from the checkpoint-aligned model
config, and writes a hardware-bound manifest. Base checkpoints can use an
explicit normalization-statistics directory without copying assets into the
weight directory.

Example:

```bash
python scripts/benchmark_carve_pi05_profile.py \
  --policy-config pi05_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --inference-steps 2 \
  --action-horizon 10 \
  --warmup-calls 2 \
  --measured-calls 20 \
  --output-manifest results/carve_optimize/pi05_eager_bf16_2step_h10.json
```

Performance measurements are publishable only when competing GPU workloads are
stopped and warmup is enabled. A one-call, no-warmup run is a conformance smoke,
not a latency result.

## Validated Profiles

On the RTX 4090 replay protocol, compiled BF16 and a mixed W8A16 profile that
quantizes VLM language layers 0--3 pass all 45 paired failure observations.
Quantizing layers 0--5 or the full language backbone fails the endpoint-drift
gate and is rejected before latency benchmarking. Detailed measurements are in
`results/carve_optimize/PROFILE_COMPARISON.md`.

The open-loop fidelity gate is necessary but not sufficient. In the subsequent
paired long-horizon pilot, compiled BF16 retained a T6 Agentic recovery success
that W8A16 lost under the same restored state, deterministic noise, controls,
and 280-step horizon. W8A16 therefore remains an experimental lower-memory
profile and compiled BF16 is the default deployment. This is the intended
hierarchy: action fidelity removes unsafe candidates cheaply, then matched
closed-loop behavior makes the final promotion decision.

Static Masked-View Elision (SMVE) is the first promoted model-critical-path
optimization beyond compilation. The LIBERO adapter always pads the unavailable
right-wrist slot; stock pi0.5 nevertheless runs its visual encoder before
masking those tokens. SMVE checks the all-false mask contract and removes that
view before SigLIP. It passes all 45 replay observations, reduces replay runtime
P50/P95 from 65.73/67.40 ms to 54.35/56.19 ms, and reaches 8/10 versus 7/10 in
the paired T8/T9 gate. The one-success difference is treated as non-inferiority,
not as evidence that input compaction improves task competence. Details are in
`../../results/carve_optimize/MASKED_VIEW_ELISION_GATE.md`.

The named contract also passes a cross-adapter gate on the converted
`pi05_droid` checkpoint. With five committed actions, ordinary compiled BF16
and SMVE both pass 10/10 paired observations; SMVE reduces runtime P50/P95 from
65.09/66.22 ms to 54.66/57.34 ms. A 15-action committed profile is rejected by
the unchanged endpoint-drift gate, so the accepted deployment remains
checkpoint- and horizon-specific. This validates portability within pi0.5, not
across VLA families. See
`../../results/carve_optimize/CROSS_ADAPTER_SMVE_GATE.md`.

## Deployment And Harness Coupling

`scripts/serve_openpi_policy_no_compile.py` accepts an approved
`ProfileManifest`, verifies checkpoint identity, replay fidelity, deadline
behavior, explicit promotion status, and closed-loop admission before applying
the backend. Legacy manifests remain readable as calibration artifacts but are
not deployable by default. `Pi05Adapter` copies the admitted profile identity
into every CARVE call trace, alongside the corresponding Agentic event and
requested/applied runtime controls. The optimizer therefore remains
independently usable while its effects are reconstructable inside an Agentic
rollout.

SMVE admission also requires a named static view contract and a promoted
fallback profile. The server validates that both manifests bind the same model,
adapter, checkpoint, hardware, flow-step count, and action horizon. If an
incoming observation marks the elided view active, only that contract violation
is retried on ordinary compiled BF16; unrelated runtime failures remain visible.

A manifest fixes the calibrated flow-step count: the server advertises equal
minimum and maximum step bounds and rejects a request for an unvalidated shape.
This prevents a one-step request from silently triggering a new
`torch.compile` graph under a two-step deployment profile.

Compiled profiles must be prewarmed before the server accepts robot requests:

```bash
python scripts/serve_openpi_policy_no_compile.py \
  --port 18081 \
  --policy-config pi05_libero \
  --policy-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
  --profile-manifest results/carve_optimize/\
deployment/pi05_smve_promoted.json \
  --fallback-profile-manifest results/carve_optimize/\
deployment/pi05_compiled_bf16_fallback_promoted.json \
  --warmup-snapshot results/carve_t689_paired_5states_v6/joint/\
failure_snapshots/task06_episode000_step0019.npz \
  --warmup-calls 2
```

The server performs one unmeasured eager prime on the same fixed-noise warmup
request before preparing the compiled profile. This mirrors the accepted
calibration path and avoids a PyTorch 2.7 SDPA/cudagraph dtype-initialization
failure. Compiled prewarm still completes before the WebSocket port opens, and
neither startup call is included in online latency metrics.

Both primary and fallback compiled paths are prewarmed before the port opens.
Preparation cost and steady control latency are kept as separate measurements.
W8A16's earlier server smoke remains diagnostic evidence, but its subsequent
closed-loop rejection prevents it from receiving promoted admission.

The fixed-profile failure-state branch smoke executed 14 real pi0.5 calls per
backend across continue, fast, accurate, and recovery branches. Compiled BF16
averaged 61.40 ms and W8A16 averaged 62.24 ms; both had 0/14 deadline misses.
The ten-step branch horizon is only a pipeline/trace validation and is not
reported as task-success evidence. Detailed evidence is in
`results/carve_profile_branches/PROFILE_BRANCH_SMOKE_COMPARISON.md`.

## Risk-Gated Asynchronous Execution

Model-call latency and robot control latency are reported separately. On the
online T6 profile, synchronous VLA steps combined `69.09 ms` policy P95 with a
`34.86 ms` simulator-step P95 and reached `114.01 ms` control P95. CARVE's
single-flight prefetcher overlaps the next VLA request with two bounded cached
actions, aligns the returned suffix, and invalidates it on physical risk or
controller escalation.

The deterministic async gate preserved task success and verified physical
recovery while reducing control P95 from `107.77 ms` to `48.04 ms` and the
80 ms miss rate from `10.43%` to `1.77%`. It increased VLA calls from 22 to 29
and episode steps from 221 to 236. The repeated run reproduced the exact video
digest and async counters. Details are in
`../../results/carve_realtime/ASYNC_PREFETCH_STATUS.md`.

The initial two-state T8/T9 gate was positive, but the decisive ten-episode
expansion rejects asynchronous execution. Synchronous commit 8 reaches `7/10`;
full-duty lead-2 prefetch reaches `5/10` while reducing misses from `417/3931`
to `48/4287`; and an alternating prefetch/synchronous duty cycle also reaches
`5/10` with `176/4278` misses. Both async runs report zero worker errors and
verified physical recovery, so the regression is a closed-loop temporal-context
effect rather than an implementation outage. Prefetch remains experimental
infrastructure, not a promoted profile. Details are in
`../../results/carve_realtime/PREFETCH_5STATE_GATE.md`.

Consequently, no asynchronous action path is promoted. Future streaming
backends must additionally verify chunk freshness, returned-suffix alignment,
motion continuity, and invalidation on monitor risk, semantic escalation, and
physical recovery.

The longer paired pilot is reported separately in
`results/carve_profile_branches/PAIRED_FAILURE_STATE_PILOT.md`. It records 623
BF16 and 662 W8A16 calls with zero 80 ms deadline misses, the closed-loop W8A16
rejection, and the limits of prompt-only recovery on an original T8 failure.

## Action-Prefix Re-Anchoring Diagnostic

The rejected asynchronous path assumes that the first two actions predicted
from an old observation are interchangeable with the two cached actions that
actually execute. `ActionPrefixVerifier` now measures this assumption through
continuous-action RMS, cumulative translation and rotation disagreement,
gripper-mode agreement, and continuous-action cosine similarity. `shadow`
mode records these metrics without changing the rollout; `enforce` mode is
implemented but remains disabled unless the diagnostic is predictive.

On the fixed five-state T8/T9 shadow run, 21 of 279 consumable prefetches would
be rejected. Episode-level reject rate is `7.21%` for successful episodes and
`7.66%` for failed episodes; its point-biserial failure correlation is `0.058`
and failure AUC is `0.563`. The signal does not separate safe from unsafe
temporal shifts. The enforcement experiment was therefore cancelled under the
predeclared gate rather than tuned against the ten outcomes. Evidence is in
`../../results/carve_realtime/prefix_shadow_t89_5states_20260717/`.

## Current Claim Boundary

- `torch.compile`, weight-only quantization, and generic asynchronous inference
  are deployment techniques, not standalone CARVE novelty.
- QVLA already establishes action-sensitivity-guided VLA quantization. CARVE's
  contribution is the stricter profile-admission protocol that includes action
  and paired closed-loop evidence.
- SMVE is a PI0.5-family optimization for a declared padded-view contract. It
  is not a universal VLA transform or the headline efficient-inference method.
- VLA-Cache already covers adaptive visual-token reuse and ActionCache covers
  multimodal-keyed action warm starts. CARVE does not claim either caching
  mechanism. Its distinct layer is Agentic-event-coherent invalidation and
  closed-loop admission around replaceable reuse backends.
- The current controller routes execution modes, but its fast and accurate
  settings both use two flow steps and ten committed actions. Do not claim
  online compute adaptation until two distinct static profiles independently
  pass fidelity, realtime, and closed-loop admission.

## Next Work

1. Connect `EventCoherentReuseGate` to every action-queue reuse site and record
   the resulting decision in `RuntimeTrace`.
2. Integrate the Realtime-VLA Triton implementation as a PI0.5 backend plugin
   and compare it with stock `torch.compile` under the same checkpoint,
   observation, flow-step, and action-fidelity contract.
3. Add one training-free temporal-reuse backend, initially ActionCache because
   it targets the PI0.5 action-head bottleneck, then evaluate CARVE invalidation
   versus similarity-only fallback on matched failure/recovery states.
4. Populate backend-specific model-stage and cold-start fields in formal
   benchmark reports; the runtime trace already supports reaction, task-cycle,
   queue-age, action-age, and stage-level fields.
5. Benchmark and admit BF16/INT4 external-planner profiles and measure their
   effect on co-resident PI0.5 deadline misses.
6. Migrate the closed-loop benchmark runner to
   `AsyncAgenticHarnessController`, then run the exact-state paired recovery
   gate.
7. Measure profile preparation and persistent-cache reload over repeated fresh
   processes.
8. Keep W8A16, generic INT8/NF4, and asynchronous prefetch rejected unless a
   new mechanism addresses their recorded failure mode.

## Co-Resident Semantic Runtime

CARVE now couples the promoted synchronous SMVE profile to an event-triggered
Qwen3.5-4B semantic observer on one RTX 4090. The observer is asynchronous and
single-flight, and event/cooldown/deadline-slack gates prevent routine semantic
calls on the control path. A paired Task 8/9 perturbation gate preserves all
six baseline outcomes and episode lengths with zero blocking reasoning time.

The initial 32-token JSON response is rejected because all six responses are
truncated and its semantic P95 is 2545.66 ms. A constrained 12-token label
protocol produces 6/6 parseable responses at 1005.29 ms P95. PI0.5 model P95
increases from 58.60 to 63.48 ms under co-resident semantic execution, while
the mean 80 ms control deadline-miss rate remains 8.02%. Evidence is in
`../../results/carve_semantic_shadow/paired_t89_3trials_20260717/SEMANTIC_SHADOW_GATE.md`.

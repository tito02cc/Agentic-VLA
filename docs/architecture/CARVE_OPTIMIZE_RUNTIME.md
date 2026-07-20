# CARVE Optimize Runtime

## Purpose

CARVE Optimize Runtime prepares and validates efficient VLA deployment profiles
without duplicating the rollout logic in `CarveRuntime`. It is independently
usable from CARVE Agentic Harness and communicates with the Harness only through
the existing inference, action, capability, trace, and Agentic-event contracts.

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
- `ContractFallbackPolicy`: retries only a violated static masked-view contract
  on an independently admitted compiled-BF16 fallback.

The current plugins are `Pi05ModelPlugin`, the zero-transform `EagerBackend`,
`TorchCompileBackend`, `MaskedViewElisionBackend`, and `TorchAOInt8Backend`. The eager backend is the
behavioral reference. The compile backend copies the policy boundary and
compiles only `sample_actions`. The masked-view backend asserts and eliminates
statically padded camera slots before visual embedding and prefix-KV
construction, then compiles the shortened sampler. The TorchAO backend applies component-scoped
W8A16 in place and then compiles the same action-sampling path. Every backend
rejects unsupported or unapplied transformations.

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

## Next Work

1. Use the completed OpenVLA-7B profile gate as second-family evidence for the
   CARVE contract, while keeping PI0.5-specific transforms explicitly scoped.
2. Retain prewarmed compiled OpenVLA BF16 as a scoped latency profile. Its
   decode path is faster and exactly faithful, but it remains above 100 ms and
   requires an unseen-shape fallback before online promotion; generic
   BitsAndBytes INT8/NF4 remain rejected.
3. Expand DROID-schema replay calibration beyond the current ten observations
   before using it as more than a portability check.
4. Measure profile preparation and persistent-cache reload over repeated fresh
   processes.

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

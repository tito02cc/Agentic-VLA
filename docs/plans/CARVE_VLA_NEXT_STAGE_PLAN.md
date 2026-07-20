# CARVE-VLA Focused Experiment Plan

Updated: 2026-07-19 (midterm-focused continuation)

## Research Question

CARVE studies whether a frozen VLA can be made more reliable and more timely by
an execution-time harness that coordinates calibrated VLA inference with
event-triggered Agentic recovery. The controller observes only deployable signals: RGB,
proprioception, recent actions, model uncertainty when available, action age,
and measured latency. Simulator object state is reserved for evaluation labels.

The paper claim is intentionally narrower than "a universal Agentic VLA":

> Given a frozen VLA and a deadline, CARVE calibrates a fidelity-preserving
> inference profile, monitors physical execution, and invokes recovery,
> semantic escalation, or safe fallback only when deployable evidence supports
> intervention.

## Architecture Reference

The canonical architecture and its invariant separation between **CARVE
Agentic Harness** and **CARVE Optimize Runtime** are defined in
[`../architecture/CARVE_RUNTIME_ARCHITECTURE.md`](../architecture/CARVE_RUNTIME_ARCHITECTURE.md).
This file only defines the current implementation and experiment sequence.

## Execution Status

This is the sole forward-looking execution plan. Architecture decisions remain
in the canonical architecture document, while implementation details are
updated in place in
[`../architecture/CARVE_OPTIMIZE_RUNTIME.md`](../architecture/CARVE_OPTIMIZE_RUNTIME.md).
No numbered plan or runtime-document variants are maintained.

### Completed

- Defined stable optimization, hardware, benchmark, action, capability, and
  trace contracts.
- Added explicit model/backend plugin interfaces and registries, together with
  `CarveOptimizeRuntime.prepare()` and `Pi05ModelPlugin`.
- Added a truthful eager BF16 reference backend that performs no hidden model
  transformation.
- Added action-fidelity checks for first action, full chunk, cosine similarity,
  gripper command, endpoint drift, and jerk.
- Added hardware-bound profile manifests and a replay benchmark over recorded
  LIBERO failure observations.
- Implemented the 9D quaternion monitor-state to 8D axis-angle pi0.5 contract
  conversion and derived noise dimensions from the loaded model configuration.
- Passed CPU contract tests and a real pi0.5 GPU conformance smoke test.
- Added a `torch_compile` backend and validated compiled BF16 on 45 paired
  failure observations. At an 80 ms deadline, 50-call P50/P95/P99 latency is
  65.73/67.40/67.94 ms with zero misses, versus 154.34/159.59/163.81 ms and
  100% misses for eager BF16.
- Added component-scoped TorchAO W8A16 and fidelity-gated layer-group search.
  Quantizing language layers 0--3 passes all 45 replay observations, reaches
  69.74/71.72/72.41 ms P50/P95/P99, and uses 6.56 GB peak VRAM. More aggressive
  language profiles are rejected by the endpoint-drift gate.
- Bound accepted manifests to server checkpoint identity and fixed calibrated
  inference steps, and propagated deployment-profile identity through every
  Agentic runtime trace.
- Completed real WebSocket and paired LIBERO failure-state branch smokes for
  compiled BF16 and accepted W8A16. Both profiles recorded 0/14 misses at an
  80 ms deadline under the fixed two-step profile.
- Added server-side prewarm. For W8A16, the first request accepted after the
  server became ready completed in 78.01 ms instead of exposing compilation
  cold start to the control loop.
- Completed a paired long-horizon failure-state pilot. Compiled BF16 preserved
  7/8 branch outcomes and W8A16 preserved 6/8; both had zero deadline misses,
  but W8A16 lost the BF16 T6 recovery success and failed the closed-loop gate.
- Reproduced the T6 BF16 accurate-failure/recovery-success outcome with videos.
  A separate T8 original-failure probe found that two prompt retries did not
  solve any branch, so the current recovery mechanism is not expanded.
- Replaced the ambiguous physical-recovery placeholder with a stateful recovery
  contract, bounded outcome memory, repeated-stall escalation, and a concrete
  delta-Cartesian retract/lift/reobserve skill.
- Passed a deterministic LIBERO state-level physical-skill smoke: two exact
  restores produced identical final state, 12 bounded actions generated an EEF
  response of 0.00862, and verification requested a fresh VLA replan.
- Added an opt-in `physical_recovery` paired branch. Physical actions count
  against the branch horizon, failed verification blocks replanning, and the
  first permitted VLA replan carries skill ID and outcome in its Agentic trace.
- Passed and reproduced the T6 compiled-BF16 intervention gate. Accurate replan
  failed at 280 steps, prompt retry succeeded at 235, and physical recovery plus
  replan succeeded at 236 with 112 VLA calls and 12 physical actions. The repeat
  matched success, steps, calls, and final simulator-state digest exactly.
- Integrated controller-selected physical recovery into the online rollout and
  reproduced the repeated-stall T6 sentinel, including bounded phase execution,
  state-response verification, recovery memory, and post-recovery replanning.
- Added risk-gated asynchronous VLA prefetch and end-to-end control-stage
  instrumentation. The repeated T6 gate preserved success and recovery while
  reducing control P95 from 107.77 ms to 48.04 ms.
- Completed the ten-episode T8/T9 asynchronous-execution gate. Full-duty
  prefetch reduces misses from `417/3931` to `48/4287`, but success falls from
  `7/10` to `5/10`; an alternating duty cycle also reaches `5/10` and is
  rejected.
- Corrected physical-recovery boundary handling by projecting finite policy
  actions to the declared normalized action contract. The original T8
  safe-stop state then completed with one verified physical recovery.
- Added Static Masked-View Elision (SMVE), which rejects active-view removal and
  omits only adapter-guaranteed padding views before SigLIP and prefix-KV
  construction. The implementation is a registered Optimize Runtime backend
  and the current full CARVE test suite has 69 passing tests.
- SMVE passes all 45 paired replay states. Relative to compiled BF16, replay
  runtime P50/P95 falls from `65.73/67.40 ms` to `54.35/56.19 ms`.
- Completed the paired synchronous T8/T9 five-state gate. SMVE reaches `8/10`
  versus `7/10` for compiled BF16 while reducing closed-loop VLA P50/P95 from
  `65.49/69.59 ms` to `56.80/60.60 ms`. The success difference is treated only
  as non-inferiority evidence.
- Added a named `StaticMaskedViewContract` and retained legacy index profiles
  only for compatibility. The backend resolves names to indices and still
  rejects an active view on every call.
- Converted the official `pi05_droid` checkpoint to PyTorch and completed a
  second-checkpoint/second-adapter gate. At five committed actions, SMVE and
  compiled BF16 both pass 10/10 fixed-noise states; SMVE reduces runtime
  P50/P95 from `65.09/66.22 ms` to `54.66/57.34 ms`. The 15-action profile is
  rejected by the unchanged endpoint gate.
- Refreshed the machine-readable runtime summary and completed a 20-slide
  midterm presentation plus an evidence-bounded oral Q&A document. Historical
  B4 aggregates without retained raw directories are excluded from promoted
  claims.
- Completed a controlled single-GPU Agent-VLM + VLA contention gate with an
  actually executing Qwen3.5-4B BF16 vision-language server. Over 500 measured
  pi0.5 calls, ordinary compiled BF16 reaches `81.12/91.85/93.72 ms`
  P50/P95/P99 and misses the 80 ms deadline on `72.8%` of calls. Compiled BF16
  + SMVE reaches `65.93/75.57/78.90 ms` and misses on `0.6%` of calls. Both
  profiles pass the same 10/10 fixed-noise fidelity gate, and the VLM load
  completes 131/131 and 126/126 image requests, respectively.
- Added optional per-call latency persistence and an automatic contention
  report/plot so arbitrary deadline thresholds are computed from the measured
  distribution instead of inferred from summary percentiles.
- Completed the event-triggered semantic-load gate with a five-second cooldown.
  Ordinary compiled BF16 reaches `66.28/85.53/90.56 ms` P50/P95/P99 and
  `13.6%` 80 ms misses; SMVE reaches `54.69/69.30/76.47 ms` and `0.4%`
  misses. Both paired VLM workloads complete 32/32 image requests. Relative to
  continuous VLM plus ordinary compilation, event-triggered VLM plus SMVE
  reduces P50/P95 by `32.6%/24.6%` and deadline misses by `72.4` percentage
  points.
- Integrated a model-independent deadline-aware semantic scheduler and
  single-flight asynchronous observer into the online CARVE rollout. In the
  paired Task 8/9 mid-episode perturbation gate, the selected 12-token label
  protocol completes and parses `6/6` Qwen3.5-4B calls, preserves all `6/6`
  paired outcomes and episode lengths, and contributes zero blocking reasoning
  time. Semantic P95 is `1005.29 ms`; co-resident PI0.5 P95 changes from
  `58.60 ms` to `63.48 ms`, while mean deadline misses remain `8.02%`.
- Audited and removed evaluator leakage from the semantic prompt. The online
  observer now receives only before/after RGB, task text, and a generic
  deployable anomaly signal; injected object identity and displacement remain
  evaluator-only metadata.
- Completed the 30-pair semantic precision gate on Task 8/9 restored states.
  The 12-token two-field protocol fails with `80%` validity, `40%` no-op false
  positives, and `60%` severe-event recall. A four-token single-code protocol
  improves validity/false positives/P95 to `100%/0%/695.76 ms` but severe-event
  recall is only `30%`. Both fail the predeclared gate, so semantic output
  remains shadow-only and no held-out prompt tuning is run.
- Added an OpenVLA adapter and model plugin. CARVE now negotiates the different
  autoregressive, single-action capability contract without exposing OpenVLA
  or Transformers dependencies to runtime core.
- Added a pinned Hugging Face OpenVLA policy wrapper, official-compatible
  LIBERO prompt/image handling, checkpoint-size validation, and a unified
  RTX-4090 profiler. The isolated runtime uses Python 3.10, PyTorch 2.5.1
  CUDA 12.1, Transformers 4.40.1, and BitsAndBytes 0.49.2; processor and CUDA
  compatibility checks pass before loading weights.
- Predeclared the OpenVLA low-bit gate over ten paired replay observations:
  action exact rate >=90%, gripper-decision agreement 100%, action MAE <=0.03,
  and per-sample MAE P95 <=0.05. A failed candidate is retained as rejected
  evidence rather than promoted.
- Completed the real OpenVLA-7B gate on one RTX 4090. BF16 reaches
  `303.48/310.92/312.74 ms` P50/P95/P99 at `14.42 GB` peak VRAM and misses the
  100 ms deadline on every call. BitsAndBytes INT8 reduces VRAM to `7.76 GB`
  but raises P50 to `1533.96 ms` and produces only `50%` exact action vectors.
  NF4 reduces VRAM to `4.41 GB` but reaches `748.35 ms` P50 and `10%` exact
  action vectors. Both low-bit candidates are rejected by the fixed fidelity
  gate despite preserving the gripper decision and passing the aggregate MAE
  limits.
- Decomposed the OpenVLA BF16 path: vision/projector/prefill/decode account for
  `11.2%/0.1%/18.7%/67.2%` of the CUDA span. An aligned action-token mask is
  exactly faithful on `10/10` paired actions and makes the LLM path compilable.
  With CUDA Graphs disabled and two prompt-length buckets prewarmed,
  `torch.compile` reaches `224.25/231.84 ms` P50/P95, a `26.1%/25.4%`
  reduction, with exact `10/10` actions and unchanged `14.42 GB` peak VRAM.
  Profile preparation takes `200.19 s`, and all calls still miss 100 ms.
- Added an explicit deployment-admission contract. Replay fidelity alone can no
  longer promote a profile: the server requires promoted replay, realtime, and
  closed-loop gates, plus checkpoint and hardware identity. This closes the
  path that could previously serve closed-loop-rejected W8A16.
- Created promoted PI0.5 deployment manifests for SMVE and its ordinary
  compiled-BF16 fallback. SMVE uses a named static-view contract; the server
  validates and prewarms both profiles, and retries only an active-view contract
  violation on the fallback.
- Completed the CARVE PI0.5 Recovery Challenge in real LIBERO MuJoCo using the
  admitted SMVE profile. Three exact restored states and four paired branches
  show `2/3` success for continuation, replan, prompt retry, and physical
  recovery; replan and prompt retry consume `452/508` PI0.5 calls versus `113`
  for continuation without success gain. Physical recovery verifies both
  supported stalls and fails closed on unsupported `stale_action`.
- Completed a separate online T6 sentinel: repeated-stall monitoring triggers
  12 bounded physical actions, verifies state response, replans with PI0.5, and
  finishes the task. Videos, per-call traces, and profile receipts are retained.
- Stopped the proposed 400-episode clean LIBERO rerun under the information-gain
  gate. It would mainly re-estimate a near-saturated aggregate and would not
  isolate recovery efficacy or runtime behavior.
- Completed the same-state Agentic-Optimize pair on T6/T9 stalls. Eager,
  compiled BF16, and compiled BF16 + SMVE all preserve `2/2` task outcomes and
  `2/2` verified recoveries. Runtime P95 is `166.26/65.75/54.50 ms`; 80 ms
  misses are `236/236`, `0/239`, and `0/247`. SMVE reduces P95 by `67.2%`
  versus eager while preserving the paired recovery outcomes.

### Current

- Treat VLA efficient inference as the primary research contribution for the
  next-stage paper and midterm review. The Agentic Harness remains the embodied
  supervision, recovery, and acceptance-test layer.
- Promote synchronous compiled BF16 + SMVE when an adapter guarantees a padded
  camera slot; retain ordinary compiled BF16 as the all-views-active fallback.
  Retain asynchronous prefetch only as rejected experimental infrastructure.
- Freeze additional OpenVLA testing. It remains second-family evidence for the
  plugin/capability contract and a useful low-bit negative result, while PI0.5
  remains the sole closed-loop optimization target for the current paper.
- Use only admitted PI0.5 profiles for subsequent Agentic integration and
  reporting. Do not reopen quantization or asynchronous candidates unless a new
  method addresses their recorded closed-loop failure mode.
- Keep the completed action-prefix shadow diagnostic as the final check on the
  current asynchronous design; do not attempt a new scheduling profile.
- Completed the action-prefix shadow diagnostic. It does not predict failure
  (`7.21%` versus `7.66%` reject rate; AUC `0.563`), so enforcement is not run
  and asynchronous execution remains rejected.
- Promote the co-resident contention result as the primary system-level
  motivation for Optimize Runtime: the same single-GPU Agent workload changes
  ordinary compilation from deadline-compliant at idle to mostly late, while
  SMVE retains a bounded warm critical path.
- Use event-triggered semantic calls plus synchronous SMVE as the promoted
  Agent/VLA co-deployment policy. Continuous VLM is retained as a stress
  envelope, not the default execution schedule.
- Treat the OpenVLA experiment as successful cross-family runtime integration
  and a negative generic-quantization result. Standard BitsAndBytes INT8/NF4
  are memory-capacity profiles on this hardware, not realtime profiles.
- Accept prewarmed OpenVLA compiled BF16 as a scoped latency profile, not the
  global default: unseen prompt-length buckets require compilation and must
  use eager fallback until a persistent shape policy is validated.

### Next

1. Integrate the admitted PI0.5 receipts, Recovery Challenge, Agentic-Optimize
   pair, online recovery video, and T8/T9 closed-loop gate into one evidence
   package for the midterm review.
2. If additional GPU time is used before the review, add only a predeclared
   recovery-state/seed extension. The same-state optimization coupling result
   is complete; do not restart a clean leaderboard sweep.
3. Freeze OpenVLA and DROID expansion unless they are required by the final
   paper. Defer WAM, MoE, W4A16, pruning, prompt tuning, and new benchmark
   training.

### Acceptance Gates

- Idle latency may be reported only when no untracked workload occupies the
  GPU. Intentional co-resident workloads must be named, driven by a recorded
  request trace, and reported as a separate deployment condition.
- Every backend must record requested, applied, dropped, and fallback transforms
  truthfully.
- Every optimized profile must pass replay fidelity before simulator evaluation.
- Every backend must preserve `ActionSpec`, capability negotiation, and trace
  semantics.

## Method Boundary

### Scope Freeze

- The Agentic Harness v1 is feature-complete for the next paper phase. Action
  contracts, policy adapters, deployable monitoring, retry/recovery hooks,
  memory hooks, fallback, and tracing are retained. Only correctness fixes are
  in scope; new Agent modules are not a current research objective.
- The primary contribution moves to control-aware inference optimization for a
  frozen VLA and for the critical path of the complete Agentic system.
- Pruning is explicitly out of scope. Quantization is included as a standard
  deployment pass rather than claimed as a new standalone quantization method.

Mandatory components:

1. A policy capability and action contract that prevents silent action-space or
   backend mismatches.
2. A deployable execution monitor using visual/proprioceptive response, command
   magnitude, action staleness, uncertainty, and deadline slack.
3. A recovery-runtime controller with explicit modes: reuse, calibrated VLA,
   experimental high-compute VLA, physical recovery, semantic planner, and safe
   stop. Sampling steps and committed action horizon are calibrated separately.
4. Per-call traces containing requested/applied controls, fallback, latency,
   deadline misses, risk evidence, and selected execution mode.

Supporting rather than standalone contributions:

- memory supplies prior failures and recovery outcomes;
- adapters expose model-specific capabilities without putting model branches in
  the controller;
- optimized kernels and quantization are backend conditions on the
  success-latency-memory frontier.

Deferred until the core result is positive: WAM integration, parameter MoE,
RoboDojo, RoboTwin training, full RoboMME, and new foundation-model training.

## Inference Optimization Design

CARVE separates two decisions that must not be conflated:

1. A **deployment profile** is selected offline for one model/backend/hardware
   tuple. It contains model precision or quantization, backend, flow steps,
   action horizon, and optional asynchronous execution settings.
2. **Runtime controls** enforce deadlines, trace applied capabilities, and
   invoke a reference fallback on explicit physical failure. They do not alter
   flow steps or action horizon from an unvalidated stall heuristic.

The optimizer searches the following initial axes:

| Axis | Initial candidates | Status |
|---|---|---|
| pi0.5 flow steps | 1, 2, 4, 7 | 2 selected by current pilot |
| committed actions | 5, 8, 10 | 10 selected by current pilot |
| precision | BF16, W8A16, W4A16 | BF16 and mixed W8A16 validated; W4A16 deferred |
| backend | eager, `torch_compile`, SMVE, TorchAO | SMVE promoted for padded-view adapters; TorchAO gated |
| execution | synchronous, measured-delay asynchronous | after fidelity gate |

Pruning, parameter MoE, and VLM token pruning are excluded from this search.

### Optimization Objective

For deployment profile `p`, minimize measured P95 critical-path latency, peak
VRAM, and energy subject to:

- paired closed-loop success being non-inferior to the reference profile;
- first-action, action-chunk, gripper, endpoint, and jerk distortion remaining
  within calibration limits;
- action contracts and backend capabilities being satisfied;
- no unsupported optimization being silently reported as applied.

The calibration process is hierarchical:

1. Profile model size, cold/warm P50/P95/P99 latency, VRAM, and energy.
2. Evaluate 500--2000 replay observations with paired deterministic noise.
3. Reject profiles with action-fidelity violations before simulation.
4. Evaluate surviving profiles on held-out paired simulator states.
5. Select the non-dominated deployment profile and save it with hardware and
   checkpoint identity in a machine-readable manifest.

### Quantization Boundary

The first pi0.5 matrix is intentionally module-level:

| Profile | VLM backbone | action expert |
|---|---|---|
| Q0 | BF16 | BF16 |
| Q1 | INT8 | BF16 |
| Q2 | INT4 | BF16 |
| Q3 | INT4 | INT8 |
| Q4 | INT4 | INT4 |

Memory reduction and latency acceleration are reported separately. A profile
is not called faster unless the deployed batch-1 kernel reduces measured wall
latency on the RTX 4090. Perplexity is not an acceptance metric; action fidelity
and paired closed-loop outcomes are.

### Agentic Critical Path

The complete harness is optimized as a system workload after the VLA profile is
selected. The initial passes are on-demand planner/critic invocation, memory
result caching, concurrent non-blocking reasoning, bounded action queues, and
reference fallback. Quantizing a separate planner VLM is optional and cannot be
mixed into the first VLA quantization claim.

## Experiment Sequence

### E0: Runtime Conformance (CPU, immediate)

- Validate action dimension, representation, frame, gripper convention,
  normalization identity, and control frequency.
- Verify capability negotiation for local and remote pi0.5 backends.
- Verify deterministic monitor/controller decisions and JSONL traces.
- Pass condition: all unit tests pass and unsupported controls are either
  rejected or explicitly recorded as dropped.

### E1: LIBERO State-Branch Smoke (no VLA required)

- Use LIBERO's MuJoCo state API to capture and restore the same control state.
- Apply two fixed action suffixes from one snapshot and verify that each branch
  is reproducible when restored.
- Save snapshot metadata and state fingerprints; do not use object poses as
  controller inputs.
- Pass condition: repeated execution of one branch produces matching simulator
  states within numerical tolerance, while different branches diverge.

### E2: pi0.5 Integration Smoke

- Run one clean LIBERO episode through the CARVE adapter with full call traces.
- Compare the legacy and CARVE boundary using the same task, initial state, seed,
  prompt, replan interval, and checkpoint.
- Pass condition: no action-contract violation, complete trace rows, and no
  unexplained behavior change from the adapter alone.

### E3: Failure-State Branching Study (highest-value experiment)

Tasks: LIBERO-10 T6, T8, and T9. These are retained because earlier work found
distinct long-horizon, contact, and closure failures on them.

At each detected failure state, restore the same snapshot and compare:

| Branch | Decision |
|---|---|
| continue | execute cached or current-policy actions |
| fast | re-infer with the low compute budget |
| accurate | re-infer with the high compute budget |
| recovery | execute the CARVE recovery mode, then re-infer |
| oracle label | best successful branch, used only as an upper bound |

Primary outcomes: recovery within a fixed step horizon, task success, added VLA
calls, wall-clock latency, deadline misses, false interventions, and branch
selection regret relative to the oracle label.

Pilot: 5 failure states per task. Expand to 20 per task only if at least one
CARVE branch beats continue and the deployable controller selects it above
chance.

### E4: Paired End-to-End Study

Run matched seeds on T6/T8/T9:

1. frozen pi0.5;
2. fixed Agentic stack with recovery but fixed compute;
3. compute-only adaptive runtime;
4. CARVE joint recovery-compute controller.

Pilot: 5 seeds per task and method. Main: 20 seeds only for methods that remain
on the pilot success-latency frontier. Report bootstrap confidence intervals,
paired success differences, p50/p95 observation-to-action latency, deadline
misses, VLA calls per success, recovery precision, and action staleness.

### E5: Focused Robustness Study

Use a small LIBERO-Plus-compatible subset or equivalent controlled perturbations
without fine-tuning: robot initialization, object layout, sensor noise, and one
mid-episode displacement. Run only perturbations that create observable failure
events in the E4 tasks. Report clean-to-perturbed retention and paired recovery
gain, not a single aggregate over every variant.

### E6: Realtime and Lightweight Backend Frontier

Compare the same CARVE controller with:

- BF16 reference pi0.5;
- the fastest stable released pi0.5 inference backend available locally;
- one practical weight-only quantization setting, only if action fidelity passes
  an open-loop gate.

Replay measured inference latency into the control clock. Report success,
latency, deadline miss rate, action age, peak VRAM, checkpoint size, and energy
per successful episode. Quantization remains an experimental backend condition
unless it provides a new method beyond existing VLA quantization work.

## Stop Rules

- Do not launch a full benchmark before its pilot passes.
- Do not add a simulator while a baseline/CARVE pair on LIBERO is incomplete.
- Do not count an oracle branch, privileged object state, or best-of-run merge as
  a deployable result.
- Remove a module from the main claim if its trigger count is near zero, its
  isolated ablation is negative, or its decisions cannot be reconstructed from
  traces.
- Do not start pi0.5 while unrelated GPU jobs leave less than 14 GiB free.
- Do not claim VLA-agnostic performance until a second compatible VLA has a
  positive matched pilot; architecture-level portability may still be claimed
  from conformance tests.

## Immediate Queue

1. Freeze `2 flow steps / 10 committed actions` as the current pi0.5 candidate
   and enforce it from the manifest at the server boundary.
2. Use compiled BF16 + SMVE for guaranteed padded-view inputs; retain ordinary
   compiled BF16 as fallback and W8A16 as rejected memory evidence.
3. Completed: controller-selected `physical_recovery` is wired into the online
   rollout, and the repeated-stall compiled-BF16 T6 sentinel passed twice.
4. Completed for pi0.5 portability: add a named view contract and validate it
   with the DROID checkpoint/adapter. A second VLA family is still required for
   a model-family-general SMVE claim.
5. Completed: measured-delay, risk-gated asynchronous prefetch passed the
   small online T6 sentinel but failed the decisive ten-episode T8/T9
   non-inferiority gate. Keep synchronous execution; both asynchronous
   schedules and quantized W8A16 remain excluded.
6. Completed: SMVE passed 45-state replay fidelity and the paired T8/T9 gate;
   it is promoted only for adapters with a guaranteed padding-view contract.

## Execution Status: 2026-07-16

- E0 complete: 44 conformance tests pass in the OpenPI PyTorch environment.
  Coverage includes local/remote
  capability fallback, dynamic pi0.5 step control, action-contract rejection,
  deployable stall detection, joint mode selection, JSONL call traces, trace
  context isolation, controlled-server capability advertisement, per-call remote
  step application/restoration, fixed-profile step rejection, profile/Agentic
  trace coupling, optimization contracts, and simulator snapshot restoration.
- E0 controlled WebSocket smoke complete: the server advertised dynamic-step
  support, a requested 2-step call was applied remotely, and the following call
  returned to the default 7-step budget. The machine-readable result is
  `results/carve_e0_controlled_websocket_smoke.json`.
- E1 complete on LIBERO-10 task IDs 6, 8, and 9. All three runs reproduce the
  same branch within numerical tolerance and produce divergent states for a
  different fixed action suffix. Results are stored as
  `results/carve_e1_libero_branch_smoke_t{6,8,9}.json`.
- E2 complete. The legacy and adapter-only task-8 episodes both succeeded, and
  the controlled server applied per-call inference budgets and deterministic
  pi0.5 noise without action-contract violations.
- The one-state compute pilot found 1-, 2-, and 4-step inference all successful
  on T6/T8/T9, but the five-state paired study showed that 1-step is not a safe
  global default.
- E4 five-state paired result on T6/T8/T9:

| Runtime | Success | Mean VLA call | Mean VLA time/episode | Episode wall |
|---|---:|---:|---:|---:|
| pi0.5 default, 7 steps, commit 10 | 14/15 | 366.9 ms | 11153.7 ms | 21.27 s |
| calibrated, 2 steps, commit 10 | 14/15 | 149.0 ms | 4439.1 ms | 14.40 s |
| dynamic 2/4 steps, commit 10 | 13/15 | 158.9 ms | 4978.1 ms | 15.91 s |

- The calibrated 2-step profile preserves the paired pilot success rate while
  providing `2.46x` lower mean VLA-call latency, `60.2%` less VLA wall time per
  episode, and `32.3%` lower end-to-end episode wall time.
- This profile is not yet a strict 80 ms VLA backend: all 447 traced fixed
  2-step calls exceeded the configured 80 ms per-call deadline. The supported
  claim is faster online action-chunk inference, not 80 ms hard real-time.
- The negative ablations are informative and retained. Dynamic 1/2-step with
  commit 8 reached 13/15; dynamic 2/4-step with commit 8 reached 11/15; dynamic
  2/4-step with commit 10 reached 13/15. On matched T8/T9 states, fixed 2-step
  commit 8 reached 6/10 versus 9/10 for commit 10. Therefore flow steps and
  committed action horizon must not be scaled together from the current stall
  score.
- Current default is the fidelity-preserving 2-step/commit-10 profile. CLI flags
  expose fast, accurate, and commit budgets, and result JSON records all three
  for reproducibility.
- Accepted deployment profiles now pass through the real controlled WebSocket
  path. A fixed two-step, ten-action branch smoke made 14 calls for each of
  compiled BF16 and W8A16 with zero 80 ms deadline misses. Mean runtime latency
  was 61.40 ms and 62.24 ms, respectively.
- The branch horizon was intentionally limited to ten simulator steps, so this
  result validates deployment identity, Agentic event tracing, deterministic
  controls, and deadline behavior; it does not establish recovery success.
- W8A16 fresh-process prewarm took 39.17 s and 1.64 s for two calls. The server
  opened only after prewarm, and its first accepted external Agentic request
  completed in 78.01 ms without a deadline miss.
- The long-horizon paired stall pilot used T6/T9 step-13 states and a 280-step
  horizon. Compiled BF16 achieved 7/8 branch successes over 623 calls with
  0/623 deadline misses. W8A16 achieved 6/8 over 662 calls with 0/662 misses.
- On T6, compiled BF16 accurate replanning failed at 280 steps while two
  Agentic recovery calls succeeded at 235 steps. This outcome reproduced in a
  second run. Continue and fast also succeeded, so the evidence supports staged
  escalation but not immediate recovery on every first stall.
- W8A16 lost the T6 recovery success and is blocked by the closed-loop gate.
- On an original failed T8 trajectory, continue, accurate, and recovery all
  failed over a 440-step branch horizon. The two-call prompt retry is therefore
  not represented as a general physical recovery method.
- Consolidated evidence is in
  `results/carve_profile_branches/PAIRED_FAILURE_STATE_PILOT.md`.
- Stateful recovery implementation is complete at the runtime-contract level.
  The controller now uses first-stall replan, repeated-stall physical recovery,
  and exhausted-budget planner/safe-stop escalation.
- The 12-action `cartesian_retract_lift_reobserve` skill passed a deterministic
  T6 LIBERO state smoke with final-state max absolute replay difference `0.0`.
  Task success remained false, so it is not yet promoted as an effective task
  recovery policy. Evidence is in
  `results/carve_recovery/PHYSICAL_RECOVERY_STATUS.md`.
- In the paired compiled-BF16 gate, physical recovery plus replan succeeded in
  236 steps where accurate replan failed at 280. A second run reproduced the
  same success, steps, 112 VLA calls, 12 physical actions, and final state
  digest. The later online repeated-stall sentinel also validated controller
  selection, phase execution, verification, and post-recovery replanning.
- End-to-end control profiling separated cached and VLA steps. Synchronous VLA
  control P95 was `114.01 ms`, while cached-action P95 was `39.97 ms`.
- The risk-gated async prefetch pilot preserved T6 task success and verified
  recovery, reduced overall control P95 from `107.77 ms` to `48.04 ms`, and
  reduced 80 ms misses from `22/211` to `4/226`. It increased VLA calls from 22
  to 29 and episode length from 221 to 236; the repeated run reproduced the
  exact video digest and counters.
- The held-out T8/T9 expansion is complete. Over ten episodes, synchronous
  commit 8 reaches `7/10`, full-duty prefetch reaches `5/10`, and alternating
  prefetch reaches `5/10`. Full-duty prefetch lowers misses by 89.4%, but both
  asynchronous profiles fail the closed-loop gate.
- Synchronous commit 10 remains the fidelity-first default. A T8 policy action
  outside the recovery skill's normalized bounds exposed a safe-stop bug;
  bounded projection fixed it, and the same state then completed with one
  verified physical recovery. Compiled BF16 + SMVE is promoted for guaranteed
  padded-view inputs; ordinary compiled BF16 remains the fallback, and no
  asynchronous profile is promoted.

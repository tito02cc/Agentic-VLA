# CARVE-VLA Framework-First Execution Plan

Updated: 2026-08-25

## 1. Objective and Scope

CARVE-VLA is a model-neutral embodied-agent runtime with two co-designed parts:

1. **CARVE Agentic Harness** turns a frozen VLA into a bounded, observable and
   recoverable robot primitive inside a multimodal agent toolchain.
2. **CARVE Optimize Runtime** selects and enforces an admitted inference profile
   for the VLA and coordinates the low-frequency VLM with the control-critical
   VLA path.

The current priority is to complete this reusable framework. The project will
not spend its main effort collecting a new simulation dataset, training a new
foundation model, or creating a custom benchmark. After the framework passes
its conformance gates, its value will be tested with frozen checkpoints on
standard benchmarks following comparable papers.

The first complete implementation targets PI0.5. Portability is established by
contracts and conformance tests; a second VLA is optional supporting evidence,
not a prerequisite for the first graduation-quality system.

## 2. Research Boundary

The thesis-level research question is:

> Can a provider-neutral Agentic Harness improve the closed-loop use of a
> frozen VLA, while a fidelity-gated Optimize Runtime controls the additional
> latency and memory cost introduced by multimodal planning, verification and
> recovery?

CARVE may claim:

- a reusable VLM-planner, VLA-executor and tool/skill architecture;
- typed authority, bounded recovery, verified memory and auditable execution;
- event-triggered semantic reasoning instead of continuous VLM invocation;
- deployment profiles admitted by latency, memory, fidelity and closed-loop
  evidence;
- benchmark results obtained with the exact frozen checkpoint and protocol
  recorded in the experiment manifest.

CARVE does not currently claim:

- a newly trained or universally stronger VLA foundation model;
- that deterministic monitoring itself provides semantic intelligence;
- benchmark-wide superiority from K0/K1 MuJoCo engineering runs;
- a novel quantization algorithm, pruning method, WAM or parameter-level MoE;
- model-family-general performance before a second VLA completes a paired test.

## 3. Final Framework

```text
task + RGB/RGB-D + proprioception + verified memory
                         |
                         v
              Replaceable VLM Planner/Critic
       GPT | Claude | local Qwen-VL | scripted baseline
                         |
                  typed intent/tool call
                         |
                         v
+----------------------------------------------------------+
| CARVE Agentic Harness                                    |
| scheduler | schema | capability | budget | state machine |
| memory    | verification | trace | safe-boundary gate    |
+----------------------------------------------------------+
       |                    |                     |
       v                    v                     v
 frozen VLA primitive  registered skill      safe hold/stop
       |                    |                     |
       +--------------------+---------------------+
                            |
                     robot/simulator
                            |
        observation + latency + PrimitiveOutcome
                            |
       Fast Execution Guard + semantic event scheduler
                            |
+----------------------------------------------------------+
| CARVE Optimize Runtime                                   |
| profile admission | backend | reuse gate | fallback      |
| deadlines | fidelity receipts | VLM/VLA scheduling       |
+----------------------------------------------------------+
```

### 3.1 Model Roles

| Component | Responsibility | Prohibited authority |
|---|---|---|
| VLM Planner/Critic | task decomposition, semantic verification, failure explanation, typed tool selection | raw actions, trajectories, torques or unregistered skills |
| Fast Execution Guard (`ExecutionRiskMonitor`) | high-frequency checks for response, stale actions, uncertainty, contact/risk evidence and deadline slack | open-vocabulary semantic judgment |
| VLA | image/language/state to bounded action chunk | retry budget, memory lifecycle and unrestricted recovery |
| Registered Physical Skill | bounded retract, lift, release, reobserve or task-specific primitive | free-form motion outside its schema and limits |
| Harness Controller | authority, lifecycle, budgets, safe boundaries, fallback and trace | replacing VLM semantics or VLA action generation |
| Optimize Runtime | backend/profile admission, latency/fidelity enforcement, reuse invalidation and fallback | silently applying unsupported optimization |

### 3.2 Stable Tool Vocabulary

The planner sees only seven typed tools:

| Tool | Meaning |
|---|---|
| `observe` | obtain deployable observations and Fast Guard evidence |
| `retrieve_memory` | retrieve task, procedural and verified failure/recovery memory |
| `vla_act` | request one admitted VLA action primitive |
| `run_skill` | execute one allowlisted physical primitive |
| `verify` | check an expected semantic or physical outcome |
| `safe_hold` | enter a heartbeat- and timeout-bounded hold state |
| `finish` | terminate the episode and persist its audit bundle |

The high-level decision vocabulary remains `continue`, `vla_act`, `run_skill`
and `safe_stop`. Provider-specific output is parsed into this common contract.

### 3.3 Primitive Protocol

A primitive is a bounded operation with typed arguments, execution authority,
budget and terminal state. Every VLA chunk and physical skill returns a
`PrimitiveOutcome`:

- status: `succeeded`, `failed`, `timeout` or `interrupted`;
- expected and observed outcome;
- whether semantic verification is required;
- deployable evidence and trace identifiers;
- no raw simulator truth and no raw action authority for the planner.

Primitive boundaries are the only routine points at which high-level semantic
decisions may alter execution. This keeps the VLM outside the real-time control
loop.

### 3.4 Planner Schedules

| Schedule | Calls | Purpose |
|---|---|---|
| `event_only` | abnormal primitive or controller escalation | lowest-cost failure-only mode |
| `selective` | task start, marked semantic boundary and abnormal event | promoted CARVE default |
| `every_primitive` | every primitive boundary | Harness-style accuracy/cost baseline |

The decisive scheduling comparison holds the VLM, prompt, VLA, initial states
and budgets fixed. It measures whether selective scheduling preserves task
behavior while reducing calls and wall time.

### 3.5 Memory Boundary

CARVE keeps three typed stores behind one retrieval interface:

- task/process memory: verified subgoal order, recipe and completion state;
- failure/recovery memory: condition, intervention and verified outcome;
- structured knowledge: HAA-RAG and qualitative scene-graph constraints from
  Agentic RAG-VLM.

Only verified outcomes may be written to persistent memory. Retrieved memory
may change a subgoal or select an allowlisted skill, but cannot emit raw robot
actions. Episode-local context, cross-episode memory and evaluator truth remain
separate.

### 3.6 Optimize Runtime Boundary

The Optimize Runtime is a deployment admission system, not a collection of
unverified speed switches. A profile binds:

- checkpoint, adapter, hardware and observation/action contracts;
- precision or quantization, backend, flow steps and committed action horizon;
- supported input/view assumptions;
- warm latency, peak VRAM and action-fidelity evidence;
- closed-loop result, fallback and versioned receipt.

The runtime currently prioritizes:

1. admitted eager/compiled backends;
2. static masked-view elision only when an adapter guarantees padding views;
3. action queues and event-coherent reuse invalidation;
4. event-triggered VLM/VLA scheduling under shared-GPU contention;
5. role-aware quantization profiles admitted separately for Planner semantics,
   VLA action fidelity and their shared-GPU composition.

Uniform low-bit conversion is a baseline rather than the proposed method. The
first Planner candidate quantizes Qwen3.5-9B language linear layers to NF4/BF16
compute while preserving its vision stack and semantic output boundary. The
first PI0.5 candidate performs component-scoped INT8 while preserving the
action expert and output path. A pair is deployable only if Planner semantic
gates, VLA fixed-noise action fidelity, matched closed-loop behavior, peak VRAM
and co-resident VLA deadlines all pass. Pruning, parameter MoE and sub-INT8 VLA
quantization remain deferred. Detailed profile definitions and the fixed ten-
profile experiment matrix are maintained in
[`../architecture/CARVE_OPTIMIZE_RUNTIME.md`](../architecture/CARVE_OPTIMIZE_RUNTIME.md).

## 4. Canonical Episode Cycle

1. Load the task, deployment profile, provider manifest and budgets.
2. Read deployable observations; evaluator truth stays private.
3. Retrieve bounded task and failure context.
4. At task start or a scheduled event, request a typed VLM decision.
5. Validate schema, capability, budget and safe-boundary authority.
6. Execute `vla_act`, an allowlisted physical skill, or `safe_hold`.
7. Run the Fast Execution Guard while the primitive executes.
8. Return one `PrimitiveOutcome` and apply the semantic schedule.
9. Verify the result; update persistent memory only after verification.
10. Persist manifest, event JSONL, planner transcript, profile receipts,
    summary and optional video.

Slow, malformed, stale or unauthorized VLM output fails closed. It may cause a
bounded safe hold, fallback or stop, but it cannot silently block or override
the control-critical action path.

## 5. Honest Implementation Status

### 5.1 Implemented and Covered by Tests

- model-neutral action/capability contracts and PI0.5/OpenVLA adapters;
- typed high-level decisions and provider-neutral planner callable;
- OpenAI-compatible/local-Qwen and Anthropic provider constructors;
- asynchronous single-flight planner lifecycle, timeout and stale-ticket
  rejection;
- Fast Execution Guard, recovery state machine, independent budgets and safe
  hold lifecycle;
- seven-tool catalog, typed registry, `PrimitiveOutcome` and `RunWorkspace`;
- task/process, failure/recovery, HAA-RAG and scene-graph context interfaces;
- Optimize Runtime manifests, backend registry, admission/fidelity gates,
  fallback and event-coherent reuse gate;
- primitive-boundary schedules: `event_only`, `selective`,
  `every_primitive`.

### 5.2 Canonical Framework Completed; Runtime Evidence Still Open

- the canonical configuration, seven-tool runtime, `CarveAgentSession`, PI0.5
  primitive and private LIBERO adapter are implemented and covered by tests;
- Codex external-tool mode now uses a runtime-owned context gateway: the
  coding agent supplies tool intent and arguments, while the Harness supplies
  episode, timestep, safe-boundary and allowlist authority;
- real local Qwen-VL and PI0.5 have passed both the joint model smoke and the
  canonical Session/tool/profile integration smoke;
- typed verification, verified-only failure-memory writes, execution-enforced
  retry/recovery budgets and resumable run workspaces are complete;
- Harness compute decisions now reach `vla_act` through an admitted
  inference-control context; requests outside the profile are rejected;
- every legacy benchmark runner does not yet use the canonical session and
  workspace artifacts;
- memory retrieval and verified failure writeback are bound to canonical
  primitive finalization; legacy runners remain outside this guarantee;
- real Anthropic/GPT endpoints are provider-compatible but not required to pass
  the local graduation experiment;
- VLM and VLA shared-GPU scheduling has measured contention evidence, but its
  final policy still needs the canonical runner.
- the P0-P3 Planner quantization gate is complete: all four profiles pass the
  final monitor-assisted semantic protocol, but Qwen3.5-4B BF16 remains the
  default because 9B NF4 saves only modest memory relative to 4B while
  increasing P95 and adding no semantic gain on the current gate;
- the deployable semantic protocol now couples a local RGB change overlay from
  the Fast Guard with a short, auditable VLM evidence/decision response; the
  older direct single-character protocol remains negative prompt evidence.

### 5.3 Experimental Evidence, Not Framework Completion

- K0/K1 Panda MuJoCo runs validate real physics, local Qwen-VL interoperability
  and scheduler cost; their analytic skills do not establish VLA improvement;
- existing PI0.5 compiled/SMVE/recovery traces support runtime design choices,
  but must be referenced through reproducible manifests in the final study;
- quantized profiles that fail latency, fidelity or paired closed-loop gates
  remain rejected profiles.

## 6. Framework Freeze Work Packages

No new benchmark campaign begins until F0-F5 pass.

### F0: Freeze Contracts and Configuration

- make one canonical config schema for planner, VLA, tool registry, memory,
  schedule, budgets, Optimize profile and benchmark adapter;
- freeze names, typed enums and manifest version;
- reject simulator-only fields at the deployment boundary.

Exit: configuration and contract tests pass without importing a simulator.

### F1: Complete Planner/Critic Layer

- expose GPT, Claude, local Qwen-VL and scripted baselines through one factory;
- freeze prompt/schema versioning and structured-output validation;
- implement task-start, semantic-boundary, anomaly and terminal schedules;
- define timeout, fallback, safe hold and API/accounting traces.

Exit: the same recorded observation bundle can be replayed through every
configured provider without changing Harness code.

### F2: Complete Agentic Toolchain

- bind the seven tools to the controller, memory, recovery registry and safe
  hold adapter;
- convert every VLA chunk and registered skill to `PrimitiveOutcome`;
- enforce allowlists, independent budgets, single active physical operation and
  episode isolation;
- persist recipe, transcript, event trace and verified memory updates.

Exit: a scripted planner can complete a full episode lifecycle using only the
seven tools, with no hidden environment calls.

### F3: Complete VLA and Optimize Integration

- make PI0.5 the first reference `vla_act` backend;
- enforce profile identity and action/observation contracts at the service
  boundary;
- bind deadline traces, fallback, SMVE eligibility and event-coherent reuse;
- keep VLM latency and VLA control latency as separate measurements;
- require cold start, warm steady-state and client reconnect stability; a
  profile that passes replay latency but fails after reconnect remains a
  candidate rather than a robust deployment default;
- retain OpenVLA only as an adapter conformance target unless more evidence is
  needed.
- add a system-profile receipt that binds one admitted Planner profile, one
  admitted VLA profile and their shared-GPU scheduler/fallback policy;
- keep uniform Qwen NF4 as a baseline, then test the vision-preserving NF4
  Planner candidate before any broader low-bit VLA work;
- run a group-wise PI0.5 INT8 sensitivity sweep with fixed flow noise, leaving
  the action expert/projections in BF16 until closed-loop evidence supports a
  wider precision plan.

Exit: one PI0.5 service smoke produces a profile-bound action chunk and complete
runtime receipt through the canonical toolchain, and the served profile passes
the declared connection-lifecycle gate or falls back with an explicit receipt.

Current result: the canonical model/tool path and a reconnect-safe fixed-loop
candidate pass. Do not promote the candidate until the same profile passes the
paired closed-loop non-inferiority gate; do not run a broad task sweep for this
purpose.

### F4: Complete Benchmark and Artifact Adapters

- expose standard benchmark reset/observe/step/success/video through a private
  environment adapter;
- prevent success labels and object ground truth from reaching the controller;
- persist one workspace per episode with configuration, trace, summary and
  video;
- provide resume, deterministic seed and aggregate commands.

Exit: one benchmark episode can be replayed and audited from its workspace.

### F5: Conformance Freeze

- run CPU contract, permission, timeout, stale-result, safe-hold, memory and
  profile-admission tests;
- run one local-Qwen planner smoke and one PI0.5 action smoke;
- run one standard benchmark episode with the full canonical path;
- tag the framework configuration before comparative experiments.

Exit: no benchmark-specific branch exists in the Harness controller and all
required artifacts are generated automatically.

## 7. Benchmark Strategy After Freeze

The benchmark is evidence for the framework, not the research contribution by
itself. Initial experiments therefore use frozen policies and paired settings;
they do not collect a new dataset or fine-tune unless the selected checkpoint
provably lacks the evaluated task distribution.

### 7.1 Primary Benchmark: LIBERO-Pro

Use the official LIBERO-Pro perturbations with the existing compatible frozen
PI0.5 checkpoint. Start from a small preregistered subset that contains:

- instruction or task redirection requiring semantic correction;
- object/position swap or visual-layout change;
- one naturally recovery-sensitive long-horizon task.

The exact official suite/task/seed identifiers must be frozen after a one-task
integration smoke. No custom relabeling may be called LIBERO-Pro.

### 7.2 Sanity Benchmark: Standard LIBERO

Use a small standard LIBERO subset only for:

- adapter no-regression against the frozen-policy runner;
- failure-state and recovery mechanism checks;
- comparison with earlier project traces.

Do not spend resources rerunning the full saturated leaderboard.

### 7.3 Optional External Benchmark

Use at most one of RoboCasa365, RoboTwin C2R or another Harness-style benchmark
only if a compatible frozen checkpoint and evaluation assets are available
without a new training campaign. Otherwise it is deferred. RoboDojo, WAM and
new model training are outside the graduation critical path.

### 7.4 Required Baselines

1. frozen VLA only;
2. frozen VLA + Fast Guard/fixed bounded retry;
3. VLM at every primitive boundary;
4. CARVE selective Planner + memory + typed recovery;
5. full CARVE + admitted Optimize Runtime.

An optional `event_only` schedule is a cost lower bound. Ablations may remove
memory or semantic planning, but all methods use the same checkpoint, initial
states, action budget and evaluator.

### 7.5 Metrics

Agentic behavior:

- task and stage success;
- semantic correction and recovery success;
- wrong intervention, retry, safe stop and repeated-failure rates;
- VLM/VLA calls per episode and per successful episode.

Efficient inference:

- VLA warm P50/P95/P99 latency and deadline misses;
- VLM latency and safe-boundary waiting time, reported separately;
- end-to-end reaction time and episode wall time;
- peak VRAM, tokens/API cost where applicable;
- action fidelity, cache hit/forced-refresh rate and fallback count.

Auditability:

- checkpoint/provider/prompt/profile identifiers;
- primitive expected/observed outcomes;
- accepted/rejected planner decisions;
- evaluator-only labels, seeds, traces and representative videos.

### 7.6 Run Scale

For each selected task:

1. one integration episode per baseline;
2. three paired pilot seeds;
3. expand only positive, stable comparisons to ten paired seeds;
4. use twenty or more seeds only when confidence intervals remain too wide and
   the task has a real mechanism signal.

This staged protocol follows the needs of a systems thesis and avoids using GPU
time to average over a broken or semantically irrelevant setup.

## 8. Existing MuJoCo Evidence

The Panda K0/K1 experiments remain useful for framework qualification and
videos:

| Scene | Every primitive | Selective | Result |
|---|---:|---:|---|
| K0 | 3 VLM calls, 66.58 s | 1 call, 48.74 s | same order and successful execution |
| K1 | 3 VLM calls, 58.09 s | 1 call, 42.12 s | same dependency order and successful execution |

These runs show that provider calls, physical simulation and scheduling are
wired and that selective scheduling lowers wall time in these cases. They do
not show that the VLM improves a VLA because the physical backend is analytic.
K3 was stopped and is not evidence.

## 9. Stop Rules

- Do not run a comparative benchmark before F0-F5 freeze.
- Do not create a new simulator task when an official benchmark can express the
  mechanism.
- Do not fine-tune merely to make the Harness comparison run; first select a
  checkpoint already trained for the benchmark distribution.
- Do not expose simulator truth, best-of-run oracle choices or raw actions to
  the planner.
- Do not report an optimization as applied without a profile receipt.
- Do not promote a low-bit, cache or asynchronous profile unless it passes
  fidelity and paired closed-loop gates.
- Do not add WAM, MoE, pruning, RoboDojo or a new VLA family before the PI0.5
  canonical path and primary study are complete.
- Do not rerun a benchmark after cosmetic framework changes; tag and freeze the
  executable configuration first.

## 10. Immediate Queue

1. Keep F0-F2 frozen; changes require a contract test and manifest-version review.
2. Freeze quantization receipts and system-profile admission before running any
   new low-bit comparison.
3. Retain the completed P0-P3 semantic gate and run the full-tool replay only
   for P0 and any future kernel-backed low-bit candidate; do not repeat the
   rejected BitsAndBytes profiles without a backend change.
4. Retain the completed group-wise PI0.5 INT8 sweep: early/middle/vision scopes
   are rejected; late-language INT8 is admitted as a low-memory tier after the
   matched T6/T9 gate; INT8+SMVE is rejected after a T9 regression.
5. Retain the admitted P0 + PI0.5 SMVE shared-GPU reference. Do not run a P3
   pair unless a different low-bit backend first demonstrates an independent
   P95 or semantic advantage. The P0 + late-language INT8 shared-GPU gate is
   complete and rejected at both continuous and five-second-cooldown schedules;
   do not repeat it without a scheduler or kernel change.
6. Finish F4/F5 with one standard canonical benchmark episode, video and
   artifact audit.
7. **Complete:** the final canonical subset uses Object T8/T9, three methods and
   three paired states. All methods obtain the same success count, so the
   preregistered expansion rule stops the sweep. See
   `../status/LIBERO_PRO_CANONICAL_TRIAD_GATE_20260825.md`.
8. **Complete:** tables, traces, evaluator-only outcomes and videos are assembled
   from the same machine-readable workspaces under
   `results/libero_pro_canonical_triad_study_20260825/`.

The subordinate experimental protocol is maintained in
[`CARVE_HARNESS_REFERENCE_EXPERIMENT_PLAN_20260824.md`](CARVE_HARNESS_REFERENCE_EXPERIMENT_PLAN_20260824.md).
Architecture and thesis lineage remain in
[`../architecture/CARVE_COMPLETE_RESEARCH_LOOP.md`](../architecture/CARVE_COMPLETE_RESEARCH_LOOP.md).

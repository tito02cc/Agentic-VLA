# CARVE-VLA RoboMME Experiment Plan

**Frozen:** 2026-08-28  
**Benchmark:** RoboMME official repository, commit `d57969fd30f8e8318fb67c389a848b3776724470`  
**Policy code:** RoboMME policy-learning repository, commit `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`

## 1. Purpose

RoboMME is the primary benchmark for testing whether CARVE adds useful high-level
planning, memory, bounded recovery, and auditable execution around a VLA. It has
16 tasks divided equally among temporal, spatial, object, and procedural memory.

This study answers three questions:

1. Does CARVE improve memory-dependent task execution without retraining the VLA?
2. Which gains come from Planner/Memory and which come from local Harness logic?
3. Can Optimize Runtime reduce end-to-end cost without reducing task success?

RoboMME is not used to claim a new VLA backbone. All primary causal comparisons
freeze the action-policy checkpoint and change only the surrounding system.

## 2. Official Evaluation Contract

- Simulator: official RoboMME ManiSkill/SAPIEN environment.
- Split: official fixed `test` split.
- Action space: 8-D absolute `joint_angle` action.
- Action chunk: policy predicts a chunk; at most 16 actions are executed per call.
- Episode limit: 1,300 simulator steps.
- Cameras: front RGB and wrist RGB only, plus the public 8-D robot state.
- Evaluator truth: terminal status is read only for scoring after decisions.
- No privileged object pose, oracle subgoal, task-success flag, or simulator state
  may enter Monitor, Planner, Memory, Critic, or policy requests.
- Every evaluated episode records a video, runtime trace, intervention trace, and
  final result.

## 3. Models

### 3.1 Primary fixed action policy

Use the released RoboMME `symbolic-grounded-subgoal/79999` checkpoint. It is a
PI0.5-based action policy trained to consume grounded language subgoals. This
lets CARVE replace the high-level reasoner without retraining or changing the
low-level action model.

### 3.2 Context baseline

Use the released `pi05_baseline/79999` checkpoint as the direct-VLA reference.
It is reported separately and is not the sole baseline for CARVE's causal claim.

### 3.3 VLM policy

The CARVE Planner/Critic is event-triggered and provider-independent. The first
local run uses the admitted quantized Qwen VLM profile. A remote provider may be
added later, but it must use the same observations, prompts, schemas, and event
triggers.

## 4. Experimental Conditions

| ID | Fixed action policy | High-level reasoning | Harness | Optimize Runtime |
|---|---|---|---|---|
| `B0-direct` | official PI0.5 baseline | none | none | native |
| `B1-policy` | GroundSG PI0.5 | official Qwen subgoal predictor | none | native |
| `C1-local` | GroundSG PI0.5 | none | Monitor + bounded recovery + safe stop | native |
| `C2-agentic` | GroundSG PI0.5 | CARVE Planner/Critic + task/episodic memory + tools | full | native |
| `C3-full` | GroundSG PI0.5 | same as `C2-agentic` | full | admitted Optimize profile |

`C2-agentic` versus `B1-policy` is the main Agentic comparison. `C3-full`
versus `C2-agentic` is the main efficient-inference comparison. `B0-direct`
provides the direct-VLA reference. `C1-local` isolates deterministic execution
supervision from semantic reasoning.

No condition may use oracle subgoals in the main table. Oracle results, if run,
are labelled as an upper bound only.

## 5. Task and Episode Schedule

### Gate A: environment admission

- Run the official dummy policy for one episode of `MoveCube`.
- Pass criteria: environment reset, RGB observations, 8-D state/action contract,
  stepping, terminal handling, and MP4 recording all work.

### Gate B: policy admission

- Tasks: `StopCube`, `VideoUnmaskSwap`, `VideoRepick`, `RouteStick`.
- One fixed test episode per task using the official GroundSG checkpoint.
- Pass criteria: server reset, history transfer, policy inference, 16-action
  execution, trace emission, and clean episode shutdown all work.

### Pilot: paired mechanism check

- Same four tasks, test episodes `0..4`.
- Conditions: `B1-policy`, `C1-local`, `C2-agentic`, `C3-full`.
- Total: 80 paired episodes.
- Continue only if there are no contract violations and CARVE has no systematic
  regression larger than one paired episode per task.

### Core study

- All 16 tasks, test episodes `0..9`.
- Conditions: `B0-direct`, `B1-policy`, `C2-agentic`, `C3-full`.
- Total: 640 episodes.
- Episodes are paired by task and official episode index.

### Publication extension

- Run all 50 official test episodes per task only for the best native baseline
  and the frozen full CARVE system.
- Total: 1,600 episodes.
- This extension starts only after the core study is audited.

## 6. Metrics

### Task quality

- success rate overall and by memory suite;
- paired failure-to-success and success-to-failure transitions;
- recovery success rate;
- false intervention rate;
- timeout and safe-stop rate;
- steps to completion.

### Agent behavior

- Planner/Critic calls per episode;
- tool calls and validated tool outcomes;
- memory reads, writes, hits, and invalidations;
- local Monitor triggers;
- recovery attempts and budget exhaustion;
- stale or rejected semantic decisions.

### Efficient inference

- VLA and VLM call counts;
- policy-only latency P50/P95/P99;
- Planner latency and end-to-end task-cycle latency;
- action age and deadline-miss rate;
- peak allocated/reserved GPU memory;
- wall time and energy proxy when available.

Policy-only latency and end-to-end latency are reported separately. Planner
latency cannot be hidden inside a VLA-only number.

## 7. Optimize Runtime Admission

The Optimize profile may include only mechanisms supported by the backend and
validated independently:

1. persistent policy service and warmup;
2. adaptive action-chunk execution under risk/deadline control;
3. action reuse only within a bounded freshness budget;
4. event-triggered asynchronous Planner/Critic calls at safe boundaries;
5. admitted low-bit VLM profile;
6. VLA low-bit weights only after numerical fidelity and closed-loop
   non-inferiority pass.

Unsupported controls are recorded as dropped capabilities. They are never
reported as active optimizations.

Admission requires:

- identical action contract and checkpoint semantics;
- no non-finite actions or protocol errors;
- at most one paired success regression in the pilot;
- lower P95 policy latency, lower call count, or lower peak memory;
- full trace coverage for every policy request.

## 8. Statistical Reporting

- Report Wilson 95% confidence intervals for success rates.
- Use paired bootstrap confidence intervals for success-rate deltas.
- Use McNemar's exact test for paired binary outcomes where sample size permits.
- Report medians and P95/P99 for skewed latency distributions.
- Preserve per-episode JSONL so all aggregate tables can be regenerated.

## 9. Execution Order

1. Build the official simulator in an isolated environment/container.
2. Pass Gate A without loading a VLA.
3. Download only the GroundSG checkpoint and pass Gate B.
4. Implement and test the CARVE RoboMME adapter and trace schema.
5. Run the 80-episode pilot.
6. Audit paired initial conditions, videos, traces, and aggregate scripts.
7. Run the 640-episode core study.
8. Start the 1,600-episode publication extension only if the core result warrants it.

RoboTwin 2.0 begins after the RoboMME adapter and result schema are stable, so
cross-model experiments reuse the same CARVE contracts instead of creating a
second measurement stack.

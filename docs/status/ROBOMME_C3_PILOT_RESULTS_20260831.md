# CARVE RoboMME C3 Pilot Results

Date: 2026-08-31

## Scope

This pilot evaluates one frozen official GroundSG PI0.5 checkpoint with a
Qwen3-VL-4B GroundSG Planner and CARVE's deployable Harness on four RoboMME
test task families. No evaluator-private state or online oracle subgoal enters
the method. Separate oracle runs are used only after a method failure to test
whether the frozen PI0.5 checkpoint can solve that episode at all.

The pilot contains five fixed test episodes per task, for 20 method episodes.
It is a mechanism and systems study, not a claim of full-benchmark coverage.

## System Configuration

- Action policy: released `mme_vla_suite` GroundSG PI0.5 checkpoint at step
  `79999`, served through the OpenPI websocket policy boundary.
- Online semantic Planner: Qwen3-VL-4B-Instruct plus the released RoboMME
  GroundSG LoRA, BF16, deterministic decoding.
- Runtime: selective grounded-subgoal reuse with task start, reuse budget,
  visual change, gripper change and execution-event invalidation.
- Harness: legal primitive contracts, semantic repair, bounded progression,
  proprioceptive completion monitoring, instance memory, tool grounding and
  auditable Planner/scheduler traces.
- Hardware: one RTX 4090; PI0.5 and the 4B Planner coexist during rollout.

## Task-Specific Tools

### StopCube

The temporal monitor detects the target and moving cube from RGB, counts
debounced target entries, estimates the next arrival from RGB half-cycle or
inter-arrival timing, and anticipates the press command by one action horizon.
The button point is refined locally. No simulator phase or evaluator count is
read by the method.

### VideoRepick

Demonstration memory identifies the persistent block. SAM2 or admitted
relation memory resolves its post-demonstration identity. A repeated-procedure
controller enforces three pick/put cycles from gripper closure and reopening,
uses cycle-dependent primitive dwell, locks the grasped instance inside a
cycle, and releases the final button press only after all cycles.

### RouteStick

The Planner is constrained to the official learned primitive vocabulary. A
trajectory tool extracts stable red route-marker targets from the
demonstration RGB, converts front-camera image direction into the robot base
frame, and advances route side at visually observed target transitions. The
VLM remains responsible for clockwise/counterclockwise selection.

### VideoUnmaskSwap

A multi-object SAM2 compiler associates initially visible red, green and blue
cubes with covering containers, tracks all identities through the swap, and
stores final color-to-container points. Online current-RGB grounding refines a
remembered point before grasp and locks the selected instance after grasp. The
execution controller performs intermediate pick/put stages and keeps the final
requested container grasped until success.

## Canonical Results

| Task | Deployable success | Mean Planner-call reduction |
|---|---:|---:|
| StopCube | 2/5 | 42.4% |
| VideoRepick | 4/5 | 41.8% |
| RouteStick | 3/5 | 42.9% |
| VideoUnmaskSwap | 5/5 | 43.1% |
| **Overall** | **14/20 (70.0%)** | **about 42.6%** |

All 20 method summaries point to valid H.264 rollout videos. The canonical
successful VideoRepick episodes use the latest event-scheduled Harness for
episodes 0, 1, 2 and 4. VideoUnmaskSwap uses one unified event configuration
for all five episodes.

### Paired raw VLM+VLA baseline

The `B1-policy` condition uses the same frozen PI0.5 and Qwen3-VL-4B GroundSG
weights on the same 20 episode identities. It invokes the Planner every action
chunk using the official-style task/history prompt, but disables CARVE task
memory, grounding tools, semantic repair, process controllers and temporal
Monitor events.

| Task | B1 raw | C2 Agentic | C3 full |
|---|---:|---:|---:|
| StopCube | 0/5 | 2/5 | 2/5 |
| VideoRepick | 1/5 | 2/5 | 4/5 |
| RouteStick | 0/5 | 1/5 | 3/5 |
| VideoUnmaskSwap | 1/5 | 5/5 | 5/5 |
| **Overall** | **2/20** | **10/20** | **14/20** |

B1 to C2 produces nine failure-to-success and one success-to-failure
transition (`+40.0` percentage points, paired bootstrap 95% interval
`[+15.0, +65.0]`, McNemar exact two-sided `p=0.0215`). B1 to C3 produces 12
failure-to-success transitions and no regression (`+60.0` points, interval
`[+40.0, +80.0]`, `p=0.0005`). This isolates the value of stateful memory,
typed tools and process control around the frozen VLA from backbone training.

## Failure Attribution

| Task | Episode | Method | Oracle diagnostic | Attribution |
|---|---:|---:|---:|---|
| StopCube | 1 | fail | fail | frozen PI0.5 upper bound |
| StopCube | 3 | fail | fail | frozen PI0.5 upper bound |
| StopCube | 4 | fail | fail | frozen PI0.5 upper bound |
| VideoRepick | 3 | fail | fail | frozen PI0.5 upper bound |
| RouteStick | 2 | fail | fail | frozen PI0.5 upper bound |
| RouteStick | 3 | fail | success | remaining Agentic direction-selection gap |

Five of six method failures also fail with the benchmark's privileged online
subgoal. On the remaining 15 ceiling-eligible episodes, the deployable method
solves 14 (`93.3%`). This adjusted number is diagnostic and must always be
reported together with the raw `14/20` result.

## Efficient-Inference Evidence

The selective runtime reduces Planner calls by roughly 42% across every task
family while retaining full success on VideoUnmaskSwap and four VideoRepick
episodes. Gripper events trigger early replanning when physical completion is
observed; stable primitives reuse cached GroundSG results. This directly tests
Agentic call-chain optimization rather than only profiling the standalone VLA.

The earlier matched runtime study remains the primary standalone PI0.5 latency
evidence. The C3 pilot adds closed-loop evidence that Planner-call reduction
can coexist with memory and recovery tools.

### Paired C2 versus C3 runtime ablation

A matched 20-episode `C2-agentic` condition invokes the same Qwen3-VL Planner
on every policy chunk. It freezes the PI0.5 checkpoint, task memory, grounding
tools, Harness logic and action horizon used by C3; only Planner scheduling
changes.

| Condition | Success | Planner calls | Policy calls | Total wall time |
|---|---:|---:|---:|---:|
| C2: every-chunk Planner | 10/20 | 256 | 256 | 766.5 s |
| C3: selective Planner | 14/20 | 171 | 298 | 602.7 s |

C3 produces four paired failure-to-success transitions and no
success-to-failure transition. It reduces Planner calls by `33.2%` and total
wall time by `21.4%`, despite executing more policy chunks because four more
long-horizon episodes complete. The paired success delta is `+20.0` percentage
points with a bootstrap 95% interval of `[+5.0, +40.0]` points. McNemar's exact
two-sided `p=0.125`; this remains mechanism-level pilot evidence rather than a
claim of full-benchmark statistical significance.

## Evidence

- Canonical report: `results/robomme_c3_pilot_canonical_20260831/README.md`
- Machine-readable summary:
  `results/robomme_c3_pilot_canonical_20260831/summary.json`
- Episode table: `results/robomme_c3_pilot_canonical_20260831/episodes.csv`
- VideoUnmaskSwap SAM2 memory:
  `results/robomme_unmask_swap_sam2_memory_20260831/`
- Canonical VideoUnmaskSwap rollouts:
  `results/robomme_c3_unmask_swap_event_v6_20260831/`
- Canonical VideoRepick rollouts:
  `results/robomme_c3_videorepick_event_v8_20260831/` and
  `results/robomme_c3_videorepick_event_v9_ep4_20260831/`, with the matched
  failed ep3 audit in `results/robomme_c3_videorepick_event_v10_ep3_20260831/`
- Canonical RouteStick rollouts:
  `results/robomme_c3_routestick_trajectory_v1_20260831/`
- Summary generator: `scripts/summarize_robomme_c3_pilot.py`
- C2/C3 ablation report:
  `results/robomme_c2_c3_runtime_ablation_20260831/README.md`
- Paired B1/C2/C3 report:
  `results/robomme_b1_c2_c3_paired_pilot_20260831/README.md`
- Raw B1 rollouts: `results/robomme_b1_raw_policy_20260831/`
- C2/C3 ablation generator:
  `scripts/summarize_robomme_c2_c3_ablation.py`
- Three-condition generator: `scripts/summarize_robomme_b1_c2_c3_pilot.py`
- Online runner: `scripts/run_robomme_vlm_groundsg.py`
- Memory compiler: `scripts/build_robomme_unmask_swap_memory.py`
- Focused regression: 45 tests passed.

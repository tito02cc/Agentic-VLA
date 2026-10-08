# CARVE-VLA: RAL-Oriented RoboMME Core Experiment Report

**Date:** 2026-08-31  
**Scope:** eight official RoboMME tasks, ten fixed episodes per task and condition  
**Scale:** 80 paired episodes per condition; 240 closed-loop simulator rollouts  
**Hardware:** one NVIDIA RTX 4090 24 GB  
**Frozen models:** GroundSG PI0.5 (`Yinpei/mme_vla_suite`, checkpoint `79999`) and Qwen3-VL-4B GroundSG Planner

## 1. Research Questions

1. Can task memory, embodied tools, process control and temporal monitoring improve a frozen VLM+VLA system without model fine-tuning?
2. Can selective Planner scheduling preserve task completion while reducing high-level model calls and end-to-end runtime?
3. Do the mechanisms remain useful beyond the initial four tasks on which they were developed?

## 2. Paired Conditions

All conditions use the same frozen checkpoints, simulator episodes, action horizon (`16`) and maximum step budget (`1300`). Success is read only from the RoboMME environment.

| ID | System | Enabled components |
|---|---|---|
| B1 | Raw VLM+VLA | official-style GroundSG Planner invoked every action chunk |
| C2 | Agentic Harness | task memory, grounded tools, semantic repair, procedure control and temporal monitor; Planner every chunk |
| C3 | Harness + Optimize Runtime | all C2 components plus selective Planner scheduling and admitted subgoal reuse |

No evaluated condition receives evaluator state or official online oracle subgoals. B1 is additionally audited to contain no CARVE memory, grounding tool or temporal-monitor event.

## 3. Task Coverage

| RoboMME suite | Task | Main capability | Episodes/condition |
|---|---|---|---:|
| Counting | StopCube | temporal event counting and timed intervention | 10 |
| Counting | BinFill | multi-object counting and ordered completion | 10 |
| Persistent | VideoRepick | persistent instance identity and repeated manipulation | 10 |
| Persistent | ButtonUnmask | event-persistent identity through occlusion | 10 |
| Referential | RouteStick | demonstration-conditioned route following | 10 |
| Referential | PickHighlight | event-triggered referential selection | 10 |
| Behavior | VideoUnmaskSwap | identity memory through occlusion and swaps | 10 |
| Behavior | MoveCube | video-conditioned behavior reproduction | 10 |

The first study covers StopCube, VideoRepick, RouteStick and VideoUnmaskSwap. The held-out expansion adds BinFill, ButtonUnmask, PickHighlight and MoveCube, one task from each official suite.

## 4. Combined Main Results

| Condition | Success | Wilson 95% CI | Planner calls | Policy calls | Total wall time |
|---|---:|---:|---:|---:|---:|
| B1 Raw VLM+VLA | 23/80 (28.7%) | [20.0%, 39.5%] | 2141 | 2141 | 5208.0 s |
| C2 Agentic Harness | 36/80 (45.0%) | [34.6%, 55.9%] | 2001 | 2001 | 5178.7 s |
| C3 Harness + Runtime | **42/80 (52.5%)** | [41.7%, 63.1%] | **1188** | 2169 | **3723.6 s** |

Relative to B1, C3 improves success by **23.8 percentage points**. There are 22 paired failure-to-success transitions and three success-to-failure transitions. The paired bootstrap 95% interval is `[+12.5, +35.0]` points and exact two-sided McNemar `p=0.000157`.

Relative to C2, C3:

- preserves or improves aggregate success: `45.0% -> 52.5%`;
- reduces Planner calls by **40.6%** (`2001 -> 1188`);
- reduces total wall time by **28.1%** (`5178.7 -> 3723.6 s`).

C3 has more Policy calls because it completes more long-horizon episodes. This does not indicate slower per-call PI0.5 inference. The C2-to-C3 success difference is not statistically significant (`p=0.146`); the supported claim is substantial semantic-inference reduction without aggregate success loss.

## 5. Per-Task Results

| Task | B1 Raw | C2 Harness | C3 Full |
|---|---:|---:|---:|
| StopCube | 0/10 | 3/10 | 2/10 |
| VideoRepick | 2/10 | 3/10 | **7/10** |
| RouteStick | 1/10 | 3/10 | **5/10** |
| VideoUnmaskSwap | 2/10 | 7/10 | **8/10** |
| BinFill | 7/10 | **8/10** | **8/10** |
| ButtonUnmask | 0/10 | **3/10** | 2/10 |
| PickHighlight | **3/10** | 2/10 | 2/10 |
| MoveCube | **8/10** | 7/10 | **8/10** |

The strongest gains occur on persistent identity, repeated manipulation and demonstration-conditioned routing. The expansion also exposes boundaries: process control causes a small PickHighlight regression, and generic subgoal reuse loses one ButtonUnmask success relative to C2.

## 6. Paired Statistical Evidence

| Comparison | Success delta | Fail to success | Success to fail | Bootstrap 95% | McNemar p |
|---|---:|---:|---:|---:|---:|
| B1 -> C2 | **+16.2 points** | 18 | 5 | [+5.0, +27.5] | **0.0106** |
| B1 -> C3 | **+23.8 points** | 22 | 3 | **[+12.5, +35.0]** | **0.000157** |
| C2 -> C3 | +7.5 points | 9 | 3 | [-1.2, +16.2] | 0.1460 |

The B1 comparisons support the Agentic claim with paired significance. The runtime claim is supported by call and wall-time reduction; its positive success trend should not be described as statistically significant.

## 7. Held-Out Expansion Analysis

The four added tasks contribute 40 paired episodes per condition:

| Condition | Success | Planner calls | Policy calls | Wall time |
|---|---:|---:|---:|---:|
| B1 | 18/40 (45.0%) | 1697 | 1697 | 4044.5 s |
| C2 | 20/40 (50.0%) | 1542 | 1542 | 3812.0 s |
| C3 | 20/40 (50.0%) | 883 | 1638 | 2619.7 s |

On this harder generalization slice, the Agentic gain is modest and not significant: B1-to-C3 is `+5.0` points with bootstrap interval `[-7.5, +17.5]`. C3 nevertheless matches C2 success while reducing Planner calls by **42.7%** and wall time by **31.3%**. This separates general runtime value from task-specific Agentic gains.

## 8. Mechanism Findings

### Event-persistent memory

ButtonUnmask stores visible target identities from the initial deployable RGB frame before the button creates occlusion. After the event, the memory tool associates each requested color with the nearest current white container. On episode 0, raw Qwen selects `<101,130>` and fails; CARVE recalls `<97,87>`, close to the diagnostic oracle point `<96,87>`, and succeeds.

The implementation supports ordered multi-target memories. Multi-target episodes execute `press -> pick target 1 -> put -> pick target 2`; some still fail because the frozen PI0.5 cannot complete the final grasp. One official online-oracle diagnostic on episode 3 also fails and is retained only as a capability-boundary artifact.

### Primitive process control

An ordered controller separates semantic stage progression from physical completion evidence:

- pick completion uses gripper closure;
- put completion uses gripper reopening;
- button press requires a minimum contact-and-withdrawal action budget;
- failed Planner transitions reuse the last admitted primitive rather than skipping stages.

The press budget is primitive-level and shared across tasks. It corrected an early transition that otherwise dragged newly revealed containers.

### Selective runtime

C3 invokes the Planner at task start, execution events, material visual/proprioceptive change and reuse-budget exhaustion. The expansion shows that this policy is effective overall but can be aggressive for precise hidden-container grasping. A risk-aware policy should refresh spatial subgoals more frequently for high-precision primitives while retaining reuse for stable stages.

## 9. Evidence Audit

- **240/240** evaluated rollout videos are present and decodable.
- Every condition contains exactly 80 fixed `(task, episode)` pairs.
- All evaluated summaries declare `privileged_online_subgoal_used=false`.
- B1 contains no task memory, CARVE grounding tool or temporal-monitor event.
- The 40-episode expansion has zero framework failures after admission fixes.
- Unit tests for the RoboMME Planner/Harness pass `43/43`.
- Oracle capability surveys and the ButtonUnmask ep3 oracle diagnostic are marked non-deployable and excluded from all reported success statistics.

Canonical evidence:

- `results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json`
- `results/robomme_b1_c2_c3_combined_80ep_20260831/episodes.csv`
- `results/robomme_b1_c2_c3_combined_80ep_20260831/README.md`
- `results/robomme_b1_c2_c3_paired_40ep_20260831/`
- `results/robomme_extension_selected_b1_c2_c3_40ep_20260831/`
- `scripts/run_robomme_b1_raw_policy_pilot.sh`
- `scripts/run_robomme_c2_agentic_pilot.sh`
- `scripts/summarize_robomme_b1_c2_c3_pilot.py`
- `scripts/combine_robomme_paired_studies.py`

## 10. Claim Boundary and RAL Readiness

This is a paired mechanism study on **8 of RoboMME's 16 tasks**, not a full benchmark leaderboard claim. It now provides statistically significant Agentic evidence, held-out task expansion, efficiency ablation, raw videos and auditable traces with frozen foundation models.

For a competitive RAL submission, the remaining high-value additions are:

1. a second modern VLA adapter or checkpoint for model-level external validity;
2. an integrated Planner quantization/VRAM table alongside selective scheduling;
3. a risk-aware selective scheduler ablation for ButtonUnmask/StopCube;
4. optional full 16-task coverage if compute and time permit.

The completed RoboMME study is sufficient for the graduation thesis experiment core and for a defensible resume project. The remaining items strengthen publication breadth rather than repair an incomplete pipeline.

## 11. Resume-Ready Description

**CARVE-VLA: Agentic and efficient inference runtime for embodied VLA systems**

- Built a modular embodied-agent runtime around frozen PI0.5 and Qwen3-VL-4B, integrating persistent visual memory, grounded tool calls, temporal/process monitors, recovery control and selective semantic planning.
- Designed a paired RoboMME evaluation over **240 real simulator rollouts** across eight tasks; improved success from **28.7% to 52.5%** (`+23.8` points, McNemar `p=1.57e-4`) without foundation-model fine-tuning.
- Reduced high-level Planner calls by **40.6%** and end-to-end wall time by **28.1%** versus the every-chunk Agentic baseline while increasing aggregate success from 45.0% to 52.5%.
- Implemented restartable experiment runners, provider adapters, structured audit traces, multi-target event memory and automated video/statistical validation on a single RTX 4090.

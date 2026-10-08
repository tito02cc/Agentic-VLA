# CARVE-VLA: RAL-Oriented Core Experiment Report

**Date:** 2026-08-31  
**Scope:** paired RoboMME mechanism study, four official task families, ten fixed episodes per task and condition  
**Hardware:** one NVIDIA RTX 4090 24 GB  
**Frozen models:** GroundSG PI0.5 (`Yinpei/mme_vla_suite`, checkpoint `79999`) and Qwen3-VL-4B GroundSG Planner

## 1. Research Questions

This study isolates two claims central to CARVE-VLA:

1. **Agentic capability:** can structured task memory, embodied tools, process control and temporal monitoring improve a frozen VLM+VLA system without fine-tuning the foundation models?
2. **Efficient inference:** can event-triggered Planner scheduling preserve or improve task completion while reducing semantic-model calls and end-to-end runtime?

The evaluation uses four RoboMME task families that stress different capabilities:

| Task | Primary capability | Episodes |
|---|---|---:|
| StopCube | temporal event counting and timed intervention | 10 |
| VideoRepick | persistent instance identity and repeated manipulation | 10 |
| RouteStick | demonstration-conditioned route following | 10 |
| VideoUnmaskSwap | identity memory through occlusion and swaps | 10 |

## 2. Paired Conditions

All three conditions use the same model checkpoints, simulator episodes, action horizon (`16`) and maximum step budget. No condition uses evaluator state or privileged online subgoals.

| ID | System | Enabled components |
|---|---|---|
| B1 | Raw VLM+VLA | official-style GroundSG Planner invoked every action chunk |
| C2 | Agentic Harness | task memory, GroundSG tools, semantic repair, procedure control and temporal monitor; Planner every chunk |
| C3 | Harness + Optimize Runtime | all C2 components plus event-triggered selective Planner scheduling and subgoal reuse |

Task memory is compiled from the official initial demonstration. StopCube uses instruction-derived count memory; VideoRepick uses VLM extraction with a video-motion fallback and SAM2 instance tracking; VideoUnmaskSwap uses multi-object SAM2 tracking; RouteStick derives a trajectory from demonstration RGB. Every memory artifact records that evaluator/oracle fields were not used.

## 3. Main Results

![Paired results](../../results/robomme_b1_c2_c3_paired_40ep_20260831/robomme_b1_c2_c3_results.png)

| Condition | Success | Wilson 95% CI | Planner calls | Policy calls | Total wall time |
|---|---:|---:|---:|---:|---:|
| B1 Raw VLM+VLA | 5/40 (12.5%) | [5.5%, 26.1%] | 444 | 444 | 1163.5 s |
| C2 Agentic Harness | 16/40 (40.0%) | [26.3%, 55.4%] | 459 | 459 | 1366.7 s |
| C3 Harness + Runtime | **22/40 (55.0%)** | [39.8%, 69.3%] | **305** | 531 | **1103.9 s** |

Relative to B1, C3 improves success by **42.5 percentage points** (4.4x relative), with 18 failure-to-success transitions and one success-to-failure transition. The paired bootstrap 95% interval is `[+25.0, +60.0]` points and the exact two-sided McNemar test gives `p=7.63e-5`.

Relative to C2, C3 improves success by **15.0 points**, reduces Planner calls by **33.6%**, and reduces total wall time by **19.2%**. The success transition is 7/1, bootstrap interval `[+2.5, +27.5]` points, and McNemar `p=0.0703`. The policy-call total is higher because C3 completes more long-horizon episodes; it is not evidence of slower per-call VLA inference.

## 4. Capability Breakdown

| Task | B1 Raw | C2 Harness | C3 Full |
|---|---:|---:|---:|
| StopCube | 0/10 | 3/10 | 2/10 |
| VideoRepick | 2/10 | 3/10 | **7/10** |
| RouteStick | 1/10 | 3/10 | **5/10** |
| VideoUnmaskSwap | 2/10 | 7/10 | **8/10** |

The strongest evidence is on long-horizon identity and route tasks. C3 reaches 70% on repeated instance manipulation, 50% on route following and 80% on occluded identity recovery. StopCube remains the main weakness: selective scheduling reduces cost but loses one success relative to C2, indicating that temporal event tasks need a dedicated high-frequency trigger rather than the generic visual-change schedule.

## 5. Paired Statistical Evidence

| Comparison | Success delta | Fail to success | Success to fail | Bootstrap 95% | McNemar p |
|---|---:|---:|---:|---:|---:|
| B1 -> C2 | +27.5 points | 13 | 2 | [+10.0, +45.0] | 0.0074 |
| B1 -> C3 | **+42.5 points** | 18 | 1 | **[+25.0, +60.0]** | **0.000076** |
| C2 -> C3 | +15.0 points | 7 | 1 | [+2.5, +27.5] | 0.0703 |

The B1 comparisons support the Agentic claim with paired significance. The C2-to-C3 result supports an efficiency advantage and a positive success trend; its exact success test does not yet cross the conventional 0.05 threshold and should not be overstated.

## 6. Evidence Audit

- **120/120** rollout videos open successfully and contain readable first and last frames.
- Every condition contains exactly 40 fixed `(task, episode)` pairs.
- B1 summaries are audited to contain no task memory, CARVE grounding-tool output or temporal-monitor event.
- Memory construction uses demonstration RGB/instructions only and declares `evaluator_or_oracle_fields_used=false`.
- Raw summaries, paired CSV, confidence intervals and transition counts are retained for reproduction.

Canonical evidence:

- `results/robomme_b1_c2_c3_paired_40ep_20260831/summary.json`
- `results/robomme_b1_c2_c3_paired_40ep_20260831/episodes.csv`
- `results/robomme_b1_c2_c3_paired_40ep_20260831/README.md`
- `scripts/run_robomme_b1_raw_policy_pilot.sh`
- `scripts/run_robomme_c2_agentic_pilot.sh`
- `scripts/summarize_robomme_b1_c2_c3_pilot.py`

## 7. Claim Boundary and RAL Readiness

This is a statistically paired **mechanism study on four representative RoboMME task families**, not the full 16-task RoboMME benchmark and not a claim of benchmark state of the art. It is already strong enough to substantiate the core thesis and a resume project: a frozen VLA gains long-horizon capabilities from an Agentic Harness, while an event-triggered runtime lowers high-level inference cost.

For a competitive RAL submission, the next highest-value additions are:

1. a second modern VLA adapter or a held-out benchmark to support model-agnostic generalization;
2. broader RoboMME task coverage or more episodes for the C2-to-C3 significance test;
3. an integrated Planner quantization/low-memory table alongside the scheduling result;
4. a focused StopCube trigger ablation explaining the temporal-task regression.

These additions strengthen external validity; they do not invalidate the completed paired evidence.

## 8. Resume-Ready Description

**CARVE-VLA: Agentic and efficient inference runtime for embodied VLA systems**

- Built a modular embodied-agent runtime around frozen PI0.5 and Qwen3-VL-4B, integrating persistent video memory, grounded tool calls, temporal/process monitors, recovery control and event-triggered semantic planning.
- Designed a paired RoboMME evaluation over 120 simulator rollouts; improved success from **12.5% to 55.0%** (`+42.5` points, McNemar `p<1e-4`) without foundation-model fine-tuning.
- Reduced high-level Planner calls by **33.6%** and end-to-end wall time by **19.2%** versus the full every-chunk Agentic baseline, while increasing success from 40.0% to 55.0%.
- Implemented reproducible provider adapters, restartable experiment runners, structured audit traces and automated video/statistical validation on a single RTX 4090.

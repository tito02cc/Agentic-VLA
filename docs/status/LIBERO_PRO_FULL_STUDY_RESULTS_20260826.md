# CARVE-VLA Full LIBERO-Pro Study

Date: 2026-08-26

## Scope

This is the complete thesis-scale paired evaluation of the frozen CARVE
framework on the locally available LIBERO-10 and LIBERO-Pro suites. It covers:

- four suites: Standard, Object, Position Swap and Task Logic;
- all ten tasks in each suite;
- ten paired initial states per task;
- Frozen VLA, Fixed Recovery and Full Agentic;
- 120 method cells and 1,200 real physics-simulator episodes.

The protocol uses the first ten official initial states, not the official
50-state leaderboard setting. No model fine-tuning was performed.

## Fixed System

- VLA: frozen PI0.5 LIBERO PyTorch checkpoint;
- Optimize profile: BF16, two denoising steps, horizon 10, masked missing views;
- VLA deadline: 80 ms;
- high-frequency layer: observable-signal `ExecutionRiskMonitor`;
- recovery: one bounded Cartesian recovery per episode;
- semantic layer: event-triggered local Qwen3.5-4B Planner/Critic;
- private evaluator: task success is never exposed to the Agent artifacts;
- one full MP4 and typed event trace per episode.

## Main Results

| Suite | Frozen VLA | Fixed Recovery | Full Agentic |
|---|---:|---:|---:|
| Standard | 96/100 | 97/100 | 97/100 |
| Object | 66/100 | 67/100 | 68/100 |
| Position Swap | 11/100 | 11/100 | 11/100 |
| Task Logic | 7/100 | 6/100 | 7/100 |
| **Total** | **180/400** | **181/400** | **183/400** |

The aggregate Wilson 95% intervals are 40.20-49.90% for Frozen, 40.44-50.15%
for Fixed and 40.93-50.65% for Agentic.

Paired against Frozen:

- Fixed Recovery: 3 conversions, 2 regressions, net +1, exact McNemar p=1.0;
- Full Agentic: 4 conversions, 1 regression, net +3, exact McNemar p=0.375.

Directly against Fixed Recovery, Full Agentic has 2 conversions and no
regressions, net +2, exact McNemar p=0.5. The two direct gains are Object T7
trial 7 and Task-Logic T8 trial 5. The sample does not establish a statistically
significant aggregate success improvement.

## Recovery Evidence

Fixed Recovery converted:

- Standard T8 trial 2;
- Object T8 trials 7 and 9.

It regressed Object T6 trial 1 and Task-Logic T8 trial 5. Full Agentic preserved
the three recovery conversions, added Object T7 trial 7, and restored the
Task-Logic T8 Fixed regression. It retained one regression on Object T6 trial 1.

Across all 400 episodes, Fixed and Agentic each invoked recovery in 173
episodes. Recovery-associated task success was 22/173 for Fixed and 24/173 for
Agentic. This supports bounded correction for a subset of failures, not a claim
that all domain shifts are recoverable at runtime.

## Efficient Inference

| Metric | Frozen VLA | Fixed Recovery | Full Agentic |
|---|---:|---:|---:|
| Control steps | 159,651 | 145,586 | 148,103 |
| VLA calls | 16,047 | 14,442 | 14,699 |
| VLA P50 | 56.43 ms | 57.02 ms | 56.44 ms |
| VLA P95 | 60.61 ms | 61.14 ms | 59.54 ms |
| VLA P99 | 64.26 ms | 65.32 ms | 63.19 ms |
| Misses above 80 ms | 1 | 0 | 8 |
| Episode wall time | 7,610.7 s | 7,296.4 s | 8,864.6 s |

Full Agentic reduces control steps by 7.2% and VLA calls by 8.4% relative to
Frozen. Its VLA deadline miss rate is 0.054%, so 99.946% of VLA calls finish
within 80 ms while PI0.5 and Qwen share one RTX 4090.

The semantic layer is low-frequency relative to VLA: 263 Planner calls versus
14,699 VLA calls (1.79%). However, synchronous Planner latency remains large:
9.09 s mean, 11.23 s P95 and 2,389.6 s total. Consequently Agentic episode wall
time is 16.5% higher than Frozen despite fewer VLA calls. This is the clearest
remaining Optimize Runtime target: asynchronous planning, smaller/quantized
VLMs, cache reuse and stricter semantic admission.

## Interpretation

The complete study supports three bounded conclusions:

1. Training-free physical recovery converts several frozen-policy failures and
   reduces wasted VLA execution, but can introduce false-positive regressions.
2. Event-triggered VLM reasoning adds two successes over Fixed Recovery and
   prevents one Fixed regression, but the aggregate success uplift is not
   statistically significant at ten states per task.
3. The admitted VLA path remains below the 80 ms target at P99 under shared-GPU
   execution. High-level synchronous VLM latency, rather than VLA inference, is
   now the dominant end-to-end deployment bottleneck.

The Position and Task-Logic suites also show the framework boundary: a runtime
Harness cannot replace missing low-level policy generalization or task-specific
training.

## Integrity Checks

The full audit passed with no failures:

- 120/120 cells and 1,200/1,200 episodes present;
- 1,200/1,200 MP4 files decoded, totaling 454,540 frames;
- 400 paired state identities match across all methods;
- one checkpoint identity and one deployment profile across the study;
- no reward, task-success, object-pose or simulator-state keys in Agent events.

## Artifacts

- protocol: `docs/plans/LIBERO_PRO_FULL_STUDY_PLAN_20260825.md`;
- recovery gate: `docs/status/LIBERO_PRO_RECOVERY_BOUNDARY_GATE_20260825.md`;
- machine-readable summary: `results/libero_pro_full_study_20260825/aggregate/study_summary.json`;
- concise table: `results/libero_pro_full_study_20260825/aggregate/REPORT.md`;
- audit: `results/libero_pro_full_study_20260825/aggregate/audit_report.json`;
- plots: `results/libero_pro_full_study_20260825/aggregate/figures/`;
- per-episode videos and traces: `results/libero_pro_full_study_20260825/`.

# CARVE Memory-Routed Planner Profile

Date: 2026-08-27

## Objective

Reduce the dominant high-level Agent latency without weakening the frozen VLA
critical path or bypassing Harness verification. The experiment uses official
LIBERO-PRO `libero_10_object` task 8 and the admitted PI0.5 compiled/SMVE
profile.

## Deployment Decision

Three routes were tested:

1. **Novel-task tier**: Qwen3.5-9B uniform NF4 generates the staged plan.
2. **Direct small-Planner candidate**: Qwen3.5-4B BF16 generates the same plan.
3. **Memory-routed tier**: Harness installs a verified symbolic procedure and
   Qwen3.5-4B BF16 is used only as visual Critic or event fallback.

The direct 4B candidate was rejected before physical execution. In two bounded
attempts it returned only one placement stage for an explicit two-object task;
the schema/coverage gate safe-stopped at step 0 after `29.72 s`. This shows that
passing a short semantic gate is insufficient for long structured planning.

## Paired Results

The primary comparison uses identical official states 0, 7 and 9.

| Route | Success | Completed plans | Startup Planner | Critic | Recovery | Mean wall | VLA P95 | Miss@80ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9B NF4 direct plan | 3/3 | 3/3 | 3 | 4 | 1 | 55.55 s | 57.23 ms | 0/119 |
| Memory + 4B BF16 Critic | 3/3 | 3/3 | 0 | 5 | 2 | 26.01 s | 58.26 ms | 0/125 |

System-level changes on the common states:

- task-success delta: `0`;
- task-start Planner calls: `3 -> 0`;
- synchronous Planner wait: reduced `100%` for retrieved tasks;
- mean episode wall time: reduced `53.2%`;
- VLA P95: increased `1.8%`, while all 125 calls remained below 80 ms.

The additional predeclared held-out state 1 also succeeded. Across all four
memory-routed episodes:

- task success and completed plans: `4/4`;
- Planner calls: `0`;
- Critic calls: `6`, mean/P95 `4.07/5.99 s`;
- physical recoveries: `2`;
- PI0.5 calls: `165`, runtime mean/P95/max `55.95/58.26/67.09 ms`;
- 80 ms deadline misses: `0/165`;
- mean episode wall time: `24.83 s`.

## Memory And Safety Boundary

The warm start reuses only a verified symbolic two-stage procedure. It does not
reuse actions, trajectories, pixels or object poses. Every state reuses the
stage order but executes a newly observed PI0.5 rollout; the 4B visual Critic
must confirm the first stage before the Harness activates the second.

Recovery completion cannot confirm a VLA stage. The executor-consistency gate
remains active in all runs, and final task success is written only by the
private evaluator after the rollout.

## Resource Result

- 4B BF16 allocated memory after load: `8.46 GiB`;
- observed co-resident process memory: approximately `10.10 GiB` for 4B and
  `7.90 GiB` for PI0.5;
- observed total GPU use: `19.33/24.56 GB`, leaving about `4.72 GB` free.

Thus BF16 is faster but consumes more memory than 9B uniform NF4 (`7.35 GiB`
allocated). CARVE should retain both routes: 9B NF4 for novel-task capacity and
memory-first 4B BF16 for repeated-task latency.

## Evidence

- [Machine-readable aggregate](../../results/planner_profile_closed_loop_20260827/aggregate.json)
- [Trial 0 video](../../results/planner_profile_closed_loop_20260827/qwen4b_bf16_memory_warm_trial0/planner-profile-4b-memory-warm-agentic-libero_10_object-t8-r0-s7-semantic-checkpoint/episode.mp4)
- [Trial 1 video](../../results/planner_profile_closed_loop_20260827/qwen4b_bf16_memory_warm_trial1/planner-profile-4b-memory-warm-agentic-libero_10_object-t8-r1-s7-semantic-checkpoint/episode.mp4)
- [Trial 7 video](../../results/planner_profile_closed_loop_20260827/qwen4b_bf16_memory_warm_trial7/planner-profile-4b-memory-warm-agentic-libero_10_object-t8-r7-s7-semantic-checkpoint/episode.mp4)
- [Trial 9 video](../../results/planner_profile_closed_loop_20260827/qwen4b_bf16_memory_warm_trial9/planner-profile-4b-memory-warm-agentic-libero_10_object-t8-r9-s7-semantic-checkpoint/episode.mp4)

All four MP4 files were decoded with `ffprobe` and visually reviewed through
uniform temporal contact sheets. They contain 392--437 H.264 frames at 20 FPS,
show both manipulation stages, and are neither blank nor truncated. Review
contact sheets are stored under
`results/planner_profile_closed_loop_20260827/_review/`.

Software verification after the routing changes: `254 passed`.

## Claim Boundary

This experiment supports a verified-procedure memory and model-routing
optimization for repeated tasks. It does not show that 4B is a generally better
Planner, that retrieved plans generalize to unrelated tasks, or that four
episodes establish benchmark-wide superiority.

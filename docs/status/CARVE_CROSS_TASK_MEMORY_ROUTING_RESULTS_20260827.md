# CARVE Cross-Task Memory Routing Results

Date: 2026-08-27

## Objective

Test whether CARVE's verified-procedure memory and adaptive model routing can be
instantiated on a second long-horizon task while keeping PI0.5 frozen. The final
qualification task is standard LIBERO-10 T3:
`put the black bowl in the bottom drawer of the cabinet and close it`.

The task differs from Object T8 in object category, destination, scene layout
and second-stage operation. It requires containment followed by drawer closure.

## Trusted Procedure Discovery

Qwen3.5-9B uniform NF4 generated a two-stage symbolic plan on official state 0:

1. place the black bowl in the bottom drawer;
2. close the bottom drawer.

The first stage was visually confirmed and the full plan was closed by the
private task evaluator. Only then was procedure
`procedure-70a817e2ffdb19aa` written to persistent memory. The record contains
two symbolic steps and no images, actions, trajectories, poses or evaluator
state.

## Factorized Held-Out Results

All routes use the same official states 1, 7 and 9, seed 7, PI0.5 checkpoint,
Harness budgets, checkpoints 130/190 and 80 ms VLA deadline.

| Route | Success | Plans | Startup Planner | Event Planner | Mean wall | Critic mean | VLA P95 | Miss@80ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct 9B NF4 | 3/3 | 3/3 | 3 | 2 | 64.96 s | 5.39 s | 59.97 ms | 0/66 |
| Memory + 9B NF4 | 3/3 | 3/3 | 0 | 1 | 24.69 s | 5.79 s | 60.19 ms | 0/66 |
| Memory + 4B BF16 | 3/3 | 3/3 | 0 | 0 | 17.49 s | 3.81 s | 60.98 ms | 0/66 |

Factorized effects:

- **Memory routing with the same 9B model:** preserves `3/3`, removes all three
  startup calls, reduces total Planner calls `5 -> 1`, and reduces mean wall
  time by `62.0%`.
- **4B Critic on the same memory route:** preserves `3/3`, reduces mean Critic
  latency by `34.3%`, and reduces mean wall time by another `29.1%`.
- **Final route versus Direct 9B:** preserves `3/3`, reduces total Planner calls
  `5 -> 0`, and reduces mean wall time by `73.1%`.

PI0.5 critical-path behavior remains stable. Runtime P95 varies by at most
`1.01 ms` across the three routes and all `198` combined calls meet the 80 ms
deadline.

## Small-Planner Boundary

Direct Qwen3.5-4B BF16 was qualified on state 1 without memory. Both bounded
attempts returned an incomplete one-stage plan for the explicit placement-plus-
closure task. The coverage gate rejected it at step 0 after `22.46 s`; no robot
action was executed and the row was not expanded.

This result supports using 4B as a low-latency Critic/fallback after verified
memory retrieval, not as a general replacement for the 9B novel-task Planner.

## Qualification Findings And Framework Corrections

The initial T4 qualification exposed two invalid memory-write paths and is
excluded from positive results:

- a two-clause placement task could previously accept a one-stage plan;
- private task success could previously promote an installed but incomplete
  task-plan ledger.

The framework now enforces ordered coverage for `put X ... and put Y ...`
instructions, requires complete installed plans before promotion, stores the
trusted plan steps rather than guessing missing stages from repeated primitive
traces, and normalizes underscore-separated task/category labels during memory
retrieval. The rejected T4 records remain quarantined for audit.

## Video And Software Verification

All nine positive-route videos decode as H.264 at 512x512 and 20 FPS. They
contain 207--227 frames. Temporal contact sheets and final-frame checks confirm
the bowl is placed inside the drawer and the drawer is closed on all three
Memory+4B states. Review artifacts are under
`results/cross_task_memory_routing_20260827/_review/`.

Current repository verification: `258 passed`; Python compilation and
`git diff --check` pass for all changed modules.

## Evidence

- [Machine-readable aggregate](../../results/cross_task_memory_routing_20260827/aggregate.json)
- [Verified T3 procedure](../../results/cross_task_memory_routing_20260827/memory/t3_reference_procedure.json)
- [Memory+4B trial 1 video](../../results/cross_task_memory_routing_20260827/memory_4b_trial1_v2/cross-task-memory-4b-v2-agentic-libero_10-t3-r1-s7-semantic-checkpoint/episode.mp4)
- [Memory+4B trial 7 video](../../results/cross_task_memory_routing_20260827/memory_4b_trial7/cross-task-memory-4b-agentic-libero_10-t3-r7-s7-semantic-checkpoint/episode.mp4)
- [Memory+4B trial 9 video](../../results/cross_task_memory_routing_20260827/memory_4b_trial9/cross-task-memory-4b-agentic-libero_10-t3-r9-s7-semantic-checkpoint/episode.mp4)
- [Frozen protocol and amendments](../plans/CARVE_CROSS_TASK_MEMORY_ROUTING_STUDY_20260827.md)

## Claim Boundary

Together with the earlier Object T8 result, this study demonstrates the routing
mechanism on two distinct task families and seven memory-routed official states.
It does not establish benchmark-wide generalization, statistical superiority in
task success, or real-robot transfer. The supported claim is that verified
procedure reuse and model-tier routing can remove repeated high-level planning
cost while preserving task success and the realtime VLA path on the tested
long-horizon tasks.

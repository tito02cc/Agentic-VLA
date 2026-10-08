# LIBERO-Pro Full-Suite Paired Study Plan

> Historical protocol, not an active queue. Do not restart this matrix.
> Follow the [single active plan](ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md).

Frozen: 2026-08-25

## Objective

Complete a thesis-scale evaluation of the frozen CARVE Agentic Harness and
Optimize Runtime without training or changing PI0.5. This protocol expands the
two-task mechanism gate to every LIBERO-10 task under the available official
LIBERO-Pro perturbations.

## Fixed Matrix

| Suite | Meaning | Tasks | Paired states | Methods | Episodes |
|---|---|---:|---:|---:|---:|
| `libero_10` | in-distribution control | 10 | 10 | 3 | 300 |
| `libero_10_object` | object appearance/scale shift | 10 | 10 | 3 | 300 |
| `libero_10_swap` | object-position shift | 10 | 10 | 3 | 300 |
| `libero_10_task` | task-logic shift | 10 | 10 | 3 | 300 |
| Total | | 40 | | | 1200 |

LIBERO-Pro provides 50 initial states per task. This study uses the first ten
paired states for a complete task-coverage experiment that fits one RTX 4090
day. It is not represented as the official 50-state leaderboard protocol.

## Methods

1. **Frozen VLA**: frozen PI0.5 with the admitted Optimize Runtime; Monitor
   intervention is disabled.
2. **Fixed Recovery**: the same VLA plus high-frequency ExecutionRiskMonitor
   and one registered bounded physical recovery; no VLM Planner.
3. **Full Agentic**: Monitor, HAA retrieval, local Qwen3.5-4B Planner/Critic,
   typed VLA/recovery tools, verification, failure memory and safe stop.

All methods share checkpoint, initial state, fixed policy noise, action horizon,
inference steps, 520-step limit and private evaluator. The VLM is loaded only
for Full Agentic because co-resident compute is part of that deployed system.

## Fixed Runtime

- VLA: PI0.5 LIBERO PyTorch;
- profile: `pi05-torch_compile_masked_views-bf16-2step-h10`;
- fallback: `pi05-torch_compile-bf16-2step-h10`;
- Planner: Qwen3.5-4B BF16, event-triggered only;
- VLA deadline: 80 ms;
- Monitor warm-up: 60 control steps before intervention;
- stall confirmation: at least two consecutive monitor windows;
- recovery budget: one physical skill per episode;
- semantic retry budget: two calls per event chain;
- video: one 256x256 frame per control step.

The monitor window is reset after a registered recovery skill completes. This
prevents deliberate recovery motion from being reused as evidence of a
continuing VLA stall. A 30-episode regression gate on nominal T4/T6/T7 states
was required before resuming the formal matrix; it achieved 30/30 success with
five recoveries and no safe stops.

## Metrics

Task behavior:

- task success with Wilson 95% confidence intervals;
- paired Frozen-failure to method-success conversions;
- paired Frozen-success regressions;
- recovery attempts, recovery-associated successes and safe stops.

Compute behavior:

- executed control steps and VLA calls;
- VLA P50/P95/P99 latency and 80 ms misses;
- VLM calls, mean/P95 latency and total Planner wall time;
- episode wall time and calls per successful episode.

Auditability:

- per-episode manifest, typed events, recipes and evaluator-only outcome;
- Planner transcript only when a semantic event occurs;
- one MP4 per episode;
- no reward, success, object pose or simulator state in Agent artifacts.

## Hypotheses and Claim Rules

- H1: Full Agentic preserves successful nominal Frozen trajectories.
- H2: bounded recovery and semantic replanning convert some Frozen failures to
  success without introducing more regressions.
- H3: when success cannot be recovered, the Harness reduces wasted execution
  through bounded safe termination.
- H4: the admitted VLA path retains low deadline-miss rate under co-resident
  Agentic execution.

Task success, safe stop and recovery are reported separately. A safe stop is
never counted as task success. Agentic success uplift is claimed only from
paired evaluator outcomes, not from shorter episodes or selected videos.

## Execution and Recovery

The batch runs Frozen VLA, then Fixed Recovery, then Full Agentic. Every episode
writes an outer private summary after completion. `--resume` skips completed
episodes and archives an incomplete workspace before replaying the same fixed
state. A failed cell is retried up to three times with model-service restart.

The performance result does not stop the matrix early. Execution stops only for
resource integrity, repeated service failure, corrupted benchmark assets or
insufficient disk space.

## Relationship to Existing Evidence

This study supplies breadth and statistics. Existing gates remain the causal
evidence for semantic correction, exact failure restoration, profile fidelity,
low-memory quantization and shared-GPU realtime admission. The thesis should
present both; the full-suite table does not replace mechanism analysis.

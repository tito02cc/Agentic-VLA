# Pre-registration — RoboDojo natural long-horizon Agentic matrix

Frozen: 2026-09-07 (Asia/Shanghai). Benchmark: RoboDojo. Policy: StarVLA PI-v3.

This document is frozen before the first episode and hashed into the aggregate
summary via `--preregistration`. Nothing below may be changed once episodes
start. If something has to change, the run is abandoned, the reason recorded, and
a new pre-registration issued with a new date.

## 1. Research question

Does the CARVE Agentic Harness improve a frozen VLA's outcomes on **naturally
occurring** long-horizon failures, and does the Optimize Runtime offset the extra
inference cost the harness introduces?

This deliberately excludes injected faults. The 2026-09-02 controlled
stale-action study already covered the injected-fault mechanism and is not
repeated.

## 2. Task selection and why

Primary task: **`stack_bowls`**.

| Criterion | Evidence |
|---|---|
| Frozen baseline has non-zero but non-saturated capability | B0 0/3 and C1 2/3 on layout 0 (2026-09-02 runs) |
| Failure is identifiable from deployment signals alone | Every failure ran to the `step_lim = 800` cap (801 video frames); successes finished in 393–670 frames |
| Framework never needs evaluator truth | A step-cap timeout is visible from action age, state response and visual change; no reward, success flag, object pose or simulator state is read |

`build_tower` and `put_bottles_into_dustbin` are **not** primary: both are 3/3 on
their sampled layouts, leaving no headroom for an Agentic gain to appear.

### Conditional second task

`build_tower` enters the matrix only if the pre-registered trigger fires: **B0 on
`build_tower` layouts 0–24 yields a success rate in [0.20, 0.80]**. That range is
fixed now. If B0 falls outside it, `build_tower` is reported as a screening
result only and does not join the matrix. This rule exists so a second task
cannot be selected after seeing which one flatters the method.

## 3. Conditions

All four use the same frozen checkpoint
`steps_100000_pytorch_model.pt` (10.17 GiB, unnorm key `arx_x5`).

| Condition | `carve_mode` | Inference steps | Execute horizon | Recovery |
|---|---|---:|---:|---|
| B0 | `baseline` | 4 | 16 | none |
| C1 | `runtime` | 2 | 16 | none |
| C2 | `agentic` | 4 | 16 | 4 steps / h16, 2 compute calls |
| C3 | `full` | 2 | 16 | 4 steps / h16, 2 compute calls |

These are the script defaults for each condition; no override env vars are set
beyond `CARVE_EVAL_NUM`.

### Scope boundary on the semantic layer

C2 and C3 run with `CARVE_PLANNER_BACKEND=task_preserving`, the default. **No
online external VLM is called.** C2/C3 therefore test the Monitor, the task
ledger, bounded recovery and safe-hold, but not an online VLM Planner or Critic.

This is deliberate, for two independent reasons:

1. No VLM profile has passed semantic admission. Uniform NF4 dropped terminal
   recall to 1/8 and was rejected; the 2B vision-preserving NF4 Critic was demoted
   to advisory after misidentifying white textures on an unseen layout.
2. `CARVE_SEMANTIC_SAFE_STOP_ENABLED` stays `false`, so no semantic component
   holds safe-stop authority.

Capacity is *not* the reason: today's measurement shows an NF4 4B planner would
fit in 19890 MiB of 24564. The reason is accuracy. Results from this matrix must
not be described as testing the full online C3.

## 4. Layouts, seeds and episode count

| Item | Value |
|---|---|
| Layout set | 0 (`Assets/Eval_Layout/RoboDojo/arx_x5/0/`) |
| Layouts | `layout_id` 0–24, ascending, as `SeedManager` orders them |
| Episodes per condition | 25, the official `eval_nums` for `stack_bowls` |
| Total episodes | 100 (4 conditions × 25) |
| Action seed base | 0 for **every** condition (`CARVE_ACTION_SEED_BASE` unset, so it defaults to the layout set) |
| Parallel envs | 1 (`STARVLA_ROBODOJO_NUM_ENVS` unset) |
| Step cap | `step_lim = 800`, RoboDojo default, unchanged |

Holding the action seed base identical across conditions is required. The
2026-09-02 `C1_seeded_repro_a` run used base 2002 against base 2 elsewhere, which
made two runs look like a reproducibility failure when they were simply different
seeds.

55 layouts exist per task per set, so 25 is the official protocol rather than a
capacity limit.

## 5. Execution order

Four sequential runs, one process launch each, in fixed order **B0, C1, C2, C3**.
Within every run layouts are visited 0→24 in the same order.

Order is fixed rather than randomized because each launch reloads a 10.17 GiB
checkpoint and Isaac Sim, making per-episode interleaving prohibitively
expensive. The consequence is accepted and stated: **success comparisons are
unaffected by ordering, but wall-clock and latency comparisons across conditions
carry possible thermal drift.** GPU temperature and clocks are recorded at each
run start so drift can be inspected rather than assumed absent.

## 6. Monitor triggers and authority

The Monitor reads only deployment-observable signals: proprioceptive state,
the action it commanded, camera frames, action age, policy uncertainty and
deadline slack. `ExecutionRiskMonitor.update()` accepts nothing else, and risk is
derived from differences between consecutive states and frames.

`reject_privileged_semantic_fields` in `agentic_vla/runtime/knowledge.py` raises
on `reward`, `success`, `object_pose`, `sim_state`, `trajectory`, joint targets
and torques, so a violation fails loudly rather than silently.

Fixed budgets for C2/C3, all script defaults:

| Setting | Value |
|---|---:|
| Semantic checkpoint steps | 0 (disabled) |
| Semantic deadline step | 0 (disabled) |
| Semantic stale checks | 2 |
| Max semantic calls per episode | 3 |
| Max semantic recoveries | 1 |
| Semantic guidance calls | 2 |
| Recovery compute calls | 2 |
| Task plan mode | `disabled` |
| Safe stop enabled | `false` |
| Fault hold step | −1 (no injected fault) |

The official evaluator produces the success label only after an episode ends. No
online component reads it.

## 7. Primary and secondary outcomes

**Primary:** paired success delta B0 → C3 on `stack_bowls` layouts 0–24.

**Decision rule, fixed now:** significance is claimed only if the exact two-sided
McNemar p < 0.05. At n = 25 that needs roughly 6 or more discordant pairs falling
one way. If p ≥ 0.05 the result is reported as non-significant and described as
directional or mechanism evidence, never as a demonstrated improvement.

**Secondary, all pre-specified:**

| Comparison | Isolates |
|---|---|
| B0 → C1 | Optimize Runtime alone |
| B0 → C2 | Agentic Harness alone |
| C2 → C3 | Optimize Runtime added on top of Agentic |
| C1 → C3 | Agentic added on top of Optimize |

Secondary comparisons are reported with the same statistics and are explicitly
secondary; none is promoted to the headline if the primary is null.

## 8. Reported quantities

Per condition: successes with Wilson 95% interval; fail-to-success and
success-to-fail counts with the layout ids listed; paired bootstrap 95% interval;
exact McNemar p; VLA calls total and mean; first-call latency separated from
steady-state p50/p95 **per episode**; deadline misses; control steps; observed
inference steps and execute horizons; layouts whose inference-step control could
not be verified; semantic checks, interventions, low-cost replans, planner calls,
accepted interventions, monitor events, and episodes with a no-progress event.

Per episode: video, `runtime_trace.jsonl`, `monitor_trace.jsonl`,
`planner_trace.jsonl` where applicable, `run_config.json`, and `SHA256SUMS`.

A baseline without a runtime trace reports `null`, not `0`, for trace-derived
metrics, keeping "unobserved" distinct from "observed zero".

## 9. Exclusion rules

An episode may be excluded **only** for an infrastructure failure that prevents it
completing: CUDA OOM, driver fault, simulator fatal error, or host interruption.
Every exclusion is reported with its condition and `layout_id`, and the affected
condition is re-run in full rather than topped up.

No episode is ever excluded because of its outcome, its duration, or how it
affects a comparison. Denominators are fixed at 25 per condition before the first
episode.

If any condition ends with fewer than 25 completed episodes, the aggregator
records `episode_count_matches_preregistration: false` for it and comparisons are
restricted to layouts where both conditions completed.

## 10. Analysis

Single command, no manual editing of intermediate files:

```bash
scripts/combine_robodojo_condition_matrix.py \
  --run <B0 artifact dir> --run <C1 artifact dir> \
  --run <C2 artifact dir> --run <C3 artifact dir> \
  --preregistration docs/plans/RAL_ROBODOJO_PREREGISTRATION_20260907.md \
  --expected-episodes-per-condition 25 \
  --output results/robodojo_condition_matrix_20260907/summary.json \
  --episodes-csv results/robodojo_condition_matrix_20260907/episodes.csv
```

Episodes pair on `(task, layout_set, layout_id)`. Statistics reuse the RoboMME
helpers unchanged so the two benchmarks stay comparable.

## 11. What will not be claimed

1. Not strict per-trajectory pairing. Independent Isaac processes are not
   bit-reproducible; the claim is same-layout pairing with a fixed action seed base.
2. Not a full online C3. No online VLM Planner or Critic runs in this matrix.
3. Not a complete RoboDojo leaderboard. One task, or two if the conditional
   trigger fires, out of 54 available.
4. Not a robot-level real-time guarantee. Latencies are single-model measurements.
5. Not evidence that any VLM Critic can judge task completion or hold safe-stop
   authority.
6. No real-robot claim. Simulation only.

## 12. Stop conditions

Abandon the run and fix the protocol, rather than reporting, if any of these
occur: `run_config.json` disagrees with the condition label; a trace shows an
inference-step or horizon that contradicts the intended profile; any online
component is found reading evaluator or private simulator state; a summary reports
episodes that did not run; or artifacts are missing videos, traces or checksums.

## 13. Environment

| Item | Value |
|---|---|
| GPU | RTX 4090, 24564 MiB, driver 580.173.02 |
| Measured requirement | 16737 MiB for B0/C1 (StarVLA 10124 + Isaac 5354 + desktop 1259) |
| Policy env | `/home/admin1/miniconda3/envs/StarVLA` |
| Sim env | `/home/admin1/miniconda3/envs/RoboDojo` |
| XPolicyLab HEAD | `c07a09614dd44cc4a67483bcb9a82e7439d99926`, plus 10 uncommitted CARVE files (2339 insertions), backed up at `artifacts/system/xpolicylab_carve_uncommitted_20260907/` |
| RoboDojo HEAD | `2184bf8844ea9d205382c4aefa3a694311418251` |

Checkpoint, driver, Torch and Isaac versions are not changed for this experiment.

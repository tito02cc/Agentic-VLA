# Amendment — RoboDojo reset validity and fresh confirmatory blocks

Frozen: 2026-09-07 (Asia/Shanghai), before any post-fix validation episode.
This document does not edit or replace the original pre-registration
`RAL_ROBODOJO_PREREGISTRATION_20260907.md`. It records why its set-0 Agentic
conditions are not used for an efficacy claim and fixes the protocol for a new,
fresh confirmatory study.

## 1. Trigger for this amendment

The set-0 matrix completed all 100 planned episodes with no OOM and complete
same-layout cells, but its operational-validity audit failed for C2/C3:

| Condition | Episodes | Successes | Monitor events | Interventions | Low-cost replans | Planner calls |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 25 | 11 | 0 | 0 | 0 | 0 |
| C1 | 25 | 11 | 0 | 0 | 0 | 0 |
| C2 | 25 | 10 | 1,262 | 4 | 4 | 4 |
| C3 | 25 | 9 | 1,651 | 6 | 6 | 6 |

C2/C3 interventions were concentrated at the beginning of the run. Later
rollouts recorded deployment-observable `no_progress` events, including long
rollouts, without evidence that a fresh per-episode Agentic state had been
established. The aggregate result alone cannot distinguish a legitimate
per-episode budget decision from stale policy state.

## 2. Corrected root-cause statement

The earlier working hypothesis was simply “the inner StarVLA websocket has no
reset branch.” A complete static call-chain audit showed that statement was
incomplete:

1. RoboDojo `EvalEnv.reset()` already sends an **outer** XPolicyLab RESET after
   simulator setup. This outer server owns the Agentic `Model` and is the
   relevant reset path.
2. `policy/starVLA/deploy.py` sends a second, logically duplicate RESET at the
   start of the same rollout. Different request UUIDs mean network-level
   idempotency does not deduplicate the two logical resets.
3. `WsModelClient` resets its local inference-step counter **before** receiving
   the server ACK. A trace timestep returning to zero therefore does not prove
   that server-side Agentic state was reset.
4. The current RoboDojo checkout's `Model.reset()` clears the principal
   semantic/planner counters, but it returns no receipt and omits at least:
   `_recovery_compute_remaining_by_env`,
   `_task_plan_contradiction_streak_by_env`, and
   `_task_plan_protocol_failures_by_env`.
5. A single `GuardedHighLevelAgent` holds one episode id/call counter. This is
   safe only for the frozen `num_envs=1` protocol; alternating multiple env ids
   would reset one another's budget.
6. Old or shadowed StarVLA copies in the workspace contain a partial reset that
   clears action/timestep state but not Agentic state. Because no receipt records
   server instance, module path, reset generation, or cleared counters, the
   completed set-0 artifacts cannot prove which implementation the resident
   service executed.

The defect is therefore an **episode-lifecycle/provenance contract failure**,
not evidence that the recovery algorithm itself is effective or ineffective.
This amendment does not infer an unobserved exact runtime cause from static code.

## 3. Status of the set-0 evidence

- C2/C3 set-0 success outcomes are marked **diagnostic / invalid for Agentic
  efficacy**.
- B0/C1 set-0 latency and deadline measurements remain descriptive efficiency
  evidence: C1 observed 2 inference steps, execute horizon 16, zero deadline
  misses, and separated first/steady-state latency.
- The B0/C1 11/25 versus 11/25 outcome and eight discordant layouts document a
  same-layout stochastic/noise floor; they are not promoted to a method claim.
- No set-0 result is deleted or overwritten. Original summaries, videos, traces,
  checksums, and the original pre-registration remain immutable.
- No threshold, recovery budget, planner prompt, checkpoint, action horizon,
  inference-step setting, or success exclusion rule is changed in response to
  the outcomes.

## 4. Protocol-only correction

The correction is limited to lifecycle correctness and auditability:

1. `EvalEnv.reset()` is the sole authoritative episode-start owner after
   successful simulator setup.
2. The rollout wrapper no longer performs an unconditional duplicate reset. A
   legacy fallback may run only when the environment exposes no acknowledged
   reset receipt.
3. A stable episode token is sent with RESET. Repeated requests carrying that
   token are idempotent even when request UUIDs differ.
4. The client resets its local inference counter only after a successful ACK.
5. `Model.reset()` clears every per-episode dict/set, including the three omitted
   fields, and returns a receipt containing at minimum reset generation,
   episode token, server instance, loaded module path, active env ids, and
   before/after budget counters.
6. `num_envs=1` remains frozen for the confirmatory study. Multi-env use remains
   disabled until budget isolation has a dedicated regression test.
7. Reset receipts are copied into artifacts and included in checksums.

This is a protocol repair, not an algorithmic intervention. Inner base-VLA
weights/caches are not reset and the RGB/action contract is unchanged.

## 5. Software and operational gates

Before any confirmatory episode:

1. Unit/integration tests demonstrate exactly one **applied** reset per episode,
   idempotent replay, ACK-before-local-step-clear, complete Model state clearing,
   and expected server module provenance.
2. Existing targeted tests and the repository test gate introduce no new
   failures beyond the four pre-existing collection errors recorded before this
   amendment.
3. A three-episode C3 diagnostic run, excluded from all efficacy statistics,
   must show:
   - one acknowledged reset generation per episode;
   - all per-episode gate counters zero at episode entry;
   - no state inherited from the preceding episode;
   - a no-progress trigger, if naturally observed, can consume a fresh budget in
     any episode, including episode 3;
   - no evaluator reward/success/private state reaches the online policy.

The diagnostic does **not** require a success, a no-progress trigger, or an
intervention in every episode. It cannot be used to tune thresholds.

## 6. Fresh confirmatory design

### 6.1 Frozen units

- Benchmark/task/policy: RoboDojo `stack_bowls`, StarVLA PI-v3, unchanged
  checkpoint.
- Conditions: B0, C1, C2, C3 exactly as in the original pre-registration.
- C2/C3 backend remains `task_preserving`; this is still not an online VLM
  Planner/Critic evaluation.
- Layout sets: **1 and 2**, never used for the set-0 diagnostic block.
- Episodes: first official 25 layouts per condition per set.
- Action seed base: equal to the layout set and identical across conditions
  within a block.
- Parallel envs: 1. Step cap: official 800. No injected faults.
- Fixed execution order in each block: B0 → C1 → C2 → C3.
- Total: 2 sets × 4 conditions × 25 = **200 confirmatory episodes**.

The study does not extend a viewed set from 25 to 55. Each block independently
uses RoboDojo's official `eval_nums=25`, avoiding outcome-driven sample-size
changes.

### 6.2 Analysis

- Report set 1 and set 2 independently, including Wilson intervals, paired
  conversion/regression layouts, paired bootstrap interval, and exact two-sided
  McNemar p.
- Pool the two untouched blocks by `(task, layout_set, layout_id)` for the
  pre-specified primary B0→C3 analysis at n=50.
- Secondary comparisons remain B0→C1, B0→C2, C2→C3, and C1→C3.
- Report block heterogeneity; pooled results never hide opposite directional
  effects across the two sets.
- Both blocks run to completion regardless of an interim set-1 outcome. There is
  no early efficacy stop and no further sample-size increase.
- Significance uses exact two-sided McNemar p < 0.05. Null or negative results
  are reported as such.

## 7. Relation to memory and later experiments

The confirmatory RoboDojo matrix isolates runtime and bounded recovery. It does
not add cross-episode memory, online VLM planning, model fine-tuning, new
primitives, or real-robot adaptation. Those are separate experiments with
separate pre-registrations, so the reset correction cannot silently become an
algorithm change.

## 8. Claim boundary

Passing this amended study can establish that the corrected single-env CARVE
runtime executes per-episode Monitor/recovery budgets as specified and quantify
its outcome/efficiency effect on fresh RoboDojo layouts. It cannot establish a
full semantic Harness VLA, learned skills, multi-agent training, benchmark-wide
generalization, multi-env correctness, or real-robot safety.

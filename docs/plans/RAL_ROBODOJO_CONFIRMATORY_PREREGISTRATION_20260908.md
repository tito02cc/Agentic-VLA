# Confirmatory pre-registration — RoboDojo fresh blocks, audited reset

Frozen 2026-09-08, before any set-1 or set-2 layout was executed or viewed.

This document pins the *executable* state for the confirmatory matrix: the
design is already frozen upstream, so what is added here is the exact code
identity, the gate conditions a block must satisfy to count, the deviations from
the earlier text, and the analysis that will be run. Nothing below may be
adjusted after seeing an outcome.

## 1. Governing documents

| Document | SHA256 |
|---|---|
| `RAL_ROBODOJO_PREREGISTRATION_20260907.md` | `804e228a71d3939805a1778e9f4633aab865f7c7c4479101e6d8b821a41a1919` |
| `RAL_ROBODOJO_CONFIRMATORY_AMENDMENT_20260907.md` | `3e0d52d3e7c25fed3a774726f320b979a252c2a43a68e5a9fea7bf705582d483` |
| `RAL_HARNESSVLA_EXPERIMENT_ROADMAP_20260907.md` | `74a1c189f8e44776af9758c1f0b8e4c5206f88579d1657a0e14019c0eb0e7019` |

The amendment governs the design (conditions, layout sets, episode counts,
analysis). This document governs execution and does not relax anything in it.

## 2. Why a fresh confirmatory block is required

The set-0 block (B0 11/25, C1 11/25, C2 10/25, C3 9/25) is not efficacy
evidence. Per-episode agent state was not cleared between episodes, so from
episode 2 onward the high-level agent could begin with a spent or already
disabled budget. That is now directly observed rather than inferred: in the
validation run the second reset receipt reported `high_level_calls=2`,
`high_level_agent_disabled_envs=1`, and 11 populated containers immediately
before clearing.

Set 0 is therefore retained as a diagnostic block only. It is never pooled with,
compared against, or used to set expectations for the confirmatory blocks.

## 3. Frozen code identity

Every digest below was verified against the live tree immediately before this
document was frozen, and again matches the state validated by the 3-episode C3
operational run.

| File | SHA256 |
|---|---|
| `XPolicyLab/policy/starVLA/model.py` | `7dd578e917176d013dd69e5a4e65c1a9ff640cf893586c582857e520138a65e4` |
| `XPolicyLab/policy/starVLA/reset_contract.py` | `855b9b9529ed12f86e3b62d52d2109fda06cbc1f413ab5958b0a903231f6c900` |
| `XPolicyLab/client_server/ws/protocol/reset.py` | `36915c20ecd296719d5b3b8bda52fc3bf55c740ac72f21dd666dc3685d4a630d` |
| `XPolicyLab/client_server/ws/model_server.py` | `44213671f501933839099ab200583af78c23027ac346c3c3c27498c9675f6789` |
| `XPolicyLab/client_server/ws/model_client.py` | `d72a8b2684b19103f28aba9a2bf4f177e32fdbc2624210bc5a6eb0d50d92a71c` |
| `src/eval_client/eval_env.py` | `cd5dffc143be9db2072d9fa2167adf3ef8a6ca672f6056352c40fc4b80d4d5f9` |
| `src/eval_client/main.py` | `96d532443f168cffd607303244a29f3958e99c82aeaacf333817905e98b4ad28` |
| `src/eval_client/policy_reset_state.py` | `09bea28ae545f651f8ba52de11832386a5cf500ca389c6d05f3f5077ff548f04` |
| `scripts/finalize_robodojo_nominal_run.py` | `debba350d9f99772dfa5638c10fcd10cbe7388812afb3b190730e79d49f83734` |
| `scripts/run_robodojo_starvla_nominal.sh` | `af74a0183cc9d59f498f73ad8e7474dd1bc8d91f0181258d060e421a7e4ebdfb` |
| `scripts/run_ral_robodojo_matrix.sh` | `738bb294bd62b1e677b932a035d6994657c65575221ae2273489988c1de24907` |

Git scopes: XPolicyLab `c07a09614dd44cc4a67483bcb9a82e7439d99926`, RoboDojo
`2184bf8844ea9d205382c4aefa3a694311418251`, workspace
`6a2926ab7eb84cd94067a08fbd6f5064da693370`. In all three the reset files are
uncommitted or untracked, so the recoverable copy is the frozen backup
`artifacts/system/carve_audited_reset_20260908/`
(`MANIFEST.json` = `9ed0f261b7ebc9026f6377609eff777e987548707921350e0d70fe1b803c4c13`,
`SHA256SUMS.txt` = `2b5a35c1dc03af159e26671b6a9a5c6f8f8305c6132672236b93d426c70172fc`).

Policy: StarVLA PI-v3, `ckpt_name=hf_qwenpi_v3`,
`checkpoints/huggingface/pi_v3/checkpoints/steps_100000_pytorch_model.pt`,
10,920,633,422 bytes. The weight SHA is deliberately not recomputed per run
(`STARVLA_HF_SKIP_WEIGHT_HASH=1`); path plus byte size is the pinned identity.
The checkpoint is unchanged from set 0, so any B0 difference between blocks is
attributable to layouts, not to the policy.

Contract versions in force: receipt `xpolicylab.episode-reset.v1`, model receipt
`carve.policy-reset.v1`, state inventory `starvla.episode-state.v1`, ledger
`robodojo.policy-reset-ledger.v1`, reset audit
`carve.robodojo.policy-reset-audit.v2`, completeness
`carve.robodojo.run-completeness.v1`.

## 4. Execution units

Task `stack_bowls`, official `eval_nums: 25`, step cap 800, `num_envs=1`,
no injected faults. Layout sets 1 and 2, first 25 official layouts (`layout_id`
0-24) per condition per set. Action seed base equals the layout set and is
identical across conditions inside a block. Fixed order B0 → C1 → C2 → C3.
Total 2 × 4 × 25 = 200 episodes.

Per-condition parameters, as the launcher derives them (no environment
overrides will be supplied):

| Condition | `carve_mode` | nominal steps | nominal horizon | recovery steps | recovery horizon | recovery compute calls |
|---|---|---|---|---|---|---|
| B0 | `baseline` | 4 | 16 | 4 | 16 | 0 |
| C1 | `runtime` | 2 | 16 | 2 | 16 | 0 |
| C2 | `agentic` | 4 | 16 | 4 | 16 | 2 |
| C3 | `full` | 2 | 16 | 4 | 16 | 2 |

C2/C3 use `CARVE_PLANNER_BACKEND=task_preserving`, which is the local
`TaskPreservingRecoveryPlanner`. No HTTP planner endpoint is contacted, so this
is explicitly **not** an online VLM Planner/Critic evaluation, and the G2 service
stays stopped for the whole matrix.

Launch (per block; `LAYOUT_SET` is the only thing that changes):

```bash
LAYOUT_SET=1 EPISODES=25 STAMP=20260908 bash scripts/run_ral_robodojo_matrix.sh
LAYOUT_SET=2 EPISODES=25 STAMP=20260908 bash scripts/run_ral_robodojo_matrix.sh
```

## 5. Gate conditions a block must satisfy to count

A condition counts only if all of the following hold. These are enforced in
code, not by inspection, and they are fixed now so that a failing block cannot be
rescued by loosening them afterwards.

1. `require_audited_reset=1` for every condition including B0. The reset audit is
   independent of whether CARVE is active.
2. The run exits `rc=0`. A non-zero exit — notably `rc=99`, PhysX exhausting its
   restart budget — fails the condition and it is re-run in full.
3. The official directory contains exactly 25 `cam_head` videos.
4. `policy_reset_audit.valid == true`: identity, ordering and
   `server_reset_generation` exactly `1..N` over the whole receipt trace, scored
   receipts a byte-identical ordered subsequence of that trace, single server
   instance and reset session, ledger `acknowledged` and anchored on the last
   applied reset, `model_module_id` and both pinned digests matching
   `run_config.json`, and all 31 StarVLA containers reported empty.
5. `run_completeness.valid == true` with `expected_episodes == 25 ==
   summarized_episodes == result.eval_time`, and RoboDojo's own `success_rate`
   agreeing with the recount from the details.

Abandoned physical episodes are expected and permitted: RoboDojo drops an episode
after its reset for PhysX or stability reasons, which leaves an orphan receipt.
Orphans are reported (`orphan_reset_events`) and do not fail a block, because an
extra reset cannot weaken per-episode isolation. What does fail a block is a hole
in the trace, which would mean a reset was applied but never recorded.

Re-run policy: a failed condition is deleted and re-run in full. Partial
condition data is never merged, and a re-run never changes any parameter above.

## 6. Deviations from the earlier text, recorded before the fact

- Original pre-registration §4.5 froze "loaded module path" inside the receipt.
  The implementation records `model_module_id` plus `model_code_sha256` instead.
  This is strictly stronger and avoids publishing absolute filesystem paths into
  shared evidence, but it is a deviation and is logged as one.
- Amendment §5.3 asked for a 3-episode C3 diagnostic gate. It was executed on
  **set 0**, not on a confirmatory set, specifically to keep sets 1 and 2
  unobserved. Artifact:
  `artifacts/robodojo/reset_fix_validation_c3_set0_20260908/`.
- The completeness gate (§5.5) and the run-id / reset-contract bindings did not
  exist in the earlier documents. They were added after review showed that
  tolerating orphan resets had removed the only hard check that a block actually
  ran to completion.

## 7. Analysis, fixed in advance

Per the amendment: report set 1 and set 2 independently with Wilson intervals,
paired conversion and regression layouts, paired bootstrap intervals and exact
two-sided McNemar p. Then pool the two blocks by
`(task, layout_set, layout_id)` for the primary B0→C3 comparison at n=50.
Secondary comparisons are B0→C1, B0→C2, C2→C3, C1→C3. Significance is exact
two-sided McNemar p < 0.05.

Both blocks run to completion regardless of the set-1 outcome. There is no early
efficacy stop, no sample-size increase, and no extension of a viewed set from 25
to 55. Block heterogeneity is reported; a pooled result never hides opposite
directions across the two blocks. Null and negative results are reported as such.

Set 0 is not pooled, not used as a baseline, and not used to choose any
threshold.

## 8. What the evidence can and cannot support

The audit certifies that every scored episode was preceded by a recorded,
server-attested reset which reported all 31 per-episode containers empty. Per
physical episode the guarantee is **at-least-once**, not exactly-once: a crash
between the reset RPC and the durable acknowledgement makes the next attempt take
a fresh token, so that episode is reset again and only the last reset is attested.

`reset_event_id` is an unkeyed digest over fields already present in the receipt,
so it proves internal consistency, not that the server performed the clearing.
Counts are self-reported by the policy process. Provenance covers `model.py` and
`reset_contract.py` only — not the checkpoint weights, not `agentic_vla/`, not
the effective `carve_*` overrides.

Known limits carried into the results: orphan resets are not reconciled against
abandoned/unstable/restart accounting; forward drift detection for the state
inventory still relies on the `(_by_env|_envs|_monitors)` naming convention; a
crash between the reset RPC returning and the trace `fsync` condemns a whole
block; `after.high_level_calls` is vacuous under B0 and C1, which construct no
`GuardedHighLevelAgent`.

Passing this study can establish that the corrected single-env CARVE runtime
executes per-episode Monitor and recovery budgets as specified, and can quantify
its outcome and efficiency effect on fresh RoboDojo layouts. It cannot establish
a full semantic Harness VLA, learned skills, multi-agent training,
benchmark-wide generalization, multi-env correctness, or anything about real
robots.

## 9. Stop conditions

Stop and report rather than adjust if: a block fails the §5 gates twice in a row
for the same reason; the reset audit reports a trace hole or a server-instance
change; or GPU memory exhaustion recurs on the same condition. In every case the
failure and its diagnostics are reported, not silently re-run with new settings.

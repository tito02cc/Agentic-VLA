# CARVE Held-Out T8/T9 Prefetch Gate

Date: 2026-07-17

> Superseded for deployment selection by `PREFETCH_5STATE_GATE.md`. This file
> remains the initial two-episode pilot; the ten-episode expansion rejects the
> asynchronous profile because success falls from 7/10 to 5/10.

## Protocol

The gate uses one fixed LIBERO-10 initial state each for task 8 and task 9,
seed 7, deterministic pi0.5 noise, the accepted compiled-BF16 two-flow-step
profile, and an 80 ms end-to-end control deadline. The policy checkpoint and
all controller settings are shared across conditions.

Async prefetch with a lead of two returns an aligned eight-action suffix from
the ten-action model output. Its causal compute comparison is therefore the
synchronous `commit=8` condition, not the higher-fidelity `commit=10` default.

## Matched Result

| Metric | Sync, commit 8 | Async, lead 2 |
|---|---:|---:|
| Task success | `2/2` | `2/2` |
| Simulator steps | `677` | `682` |
| VLA calls | `83` | `84` |
| VLA model wall time | `5.52 s` | `5.72 s` |
| Episode wall time | `30.28 s` | `25.88 s` |
| 80 ms control misses | `83/657` | `3/662` |
| Control miss rate | `12.63%` | `0.45%` |
| T8 control P95 | `108.92 ms` | `45.33 ms` |
| T9 control P95 | `108.11 ms` | `47.79 ms` |
| Peak GPU memory | `8.876 GB` | `8.882 GB` |

Async prefetch reduced the relative deadline-miss rate by `96.4%` and episode
wall time by `14.5%`. It used one additional VLA call (`+1.2%`) and five
additional simulator steps (`+0.7%`). It submitted 81 requests, consumed 80,
invalidated one, committed 640 aligned actions, and recorded zero waits and
zero worker errors.

## Fidelity Default And Recovery Fix

The synchronous `commit=10` fidelity profile also completed both tasks after a
recovery-boundary fix. It executed 64 VLA calls over 654 simulator steps,
missed 64/634 control deadlines, and verified one physical recovery on T8.

The original T8 baseline reached a finite pi0.5 action slightly outside the
normalized recovery contract. MuJoCo had accepted the policy command, but the
physical-skill constructor rejected it and safe-stopped. Recovery construction
now projects finite model actions to the declared `ActionSpec` bounds before
building retract/lift commands; dimensions and non-finite values remain hard
errors. Replaying the same T8 state then started recovery and succeeded.

## Decision

- Keep synchronous `2 flow steps / commit 10` as the fidelity-first default.
- The expanded ten-episode gate rejects the `lead=2` pilot: synchronous commit
  8 reaches `7/10`, while full-duty prefetch reaches `5/10`.
- Do not claim that prefetch improves task success. The held-out evidence only
  supports success preservation on two states and a control-latency benefit.
- Do not report the exploratory `lead=1` artifact as an async result: it made
  zero prefetch submissions. The CLI now rejects lead values below two.
- A broader benchmark estimate still requires more initial states; this gate is
  sufficient for runtime acceptance, not a statistical success-rate claim.

## Evidence

- `heldout_t89_sync_boundfix_20260717/`
- `heldout_t89_sync_commit8_20260717/`
- `heldout_t89_async_20260717/`
- `heldout_t8_sync_commit10_recovery_boundfix_20260717/`

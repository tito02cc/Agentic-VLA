# CARVE Risk-Gated Asynchronous Prefetch Status

Date: 2026-07-17

## Motivation

The accepted compiled-BF16 pi0.5 profile meets the 80 ms policy-call deadline,
but synchronous online control serializes image preparation, VLA inference,
and MuJoCo stepping. A deterministic T6 profile measured:

- cached-action control P95: `39.97 ms`;
- synchronous VLA control P95: `114.01 ms`;
- VLA latency P95: `69.09 ms`;
- environment-step P95: `34.86 ms`;
- image preprocessing P95: `5.61 ms`.

The synchronous critical path cannot reliably fit an 80 ms end-to-end control
deadline even though the model call alone does. CARVE therefore overlaps a
bounded VLA request with execution of the final cached actions in a chunk.

## Runtime Contract

`AsyncInferencePrefetcher` is model-independent and single-flight. Each ticket
records its episode, submission timestep, and lead-action alignment. The online
runner submits only during normal low-risk execution and consumes the predicted
suffix after skipping the actions temporally covered by the lead window.

Prefetch is invalidated by stall, slip, misgrasp, contact, high risk, accurate
replan, physical recovery, planner escalation, or safe stop. Expected
chunk-tail `stale_action` does not invalidate prefetch. Critical recovery and
post-recovery replanning remain synchronous.

## Deterministic T6 Gate

Both conditions use fixed policy noise, the accepted two-step compiled-BF16
profile, a ten-action model horizon, and the same initial state.

| Metric | Synchronous | Async prefetch |
|---|---:|---:|
| Success | `1/1` | `1/1` |
| Episode steps | `221` | `236` |
| VLA calls | `22` | `29` |
| Physical recovery | `1/1` verified | `1/1` verified |
| Control P95 | `107.77 ms` | `48.04 ms` |
| End-to-end 80 ms misses | `22/211` | `4/226` |
| End-to-end miss rate | `10.43%` | `1.77%` |
| Policy-call 80 ms misses | `0/22` | `0/29` |
| VLA latency P95 | `69.09 ms` | `72.90 ms` |
| Episode wall time | `9.11 s` | `9.01 s` |

Async prefetch reduced control P95 by `55.4%` and relative deadline-miss rate
by `83.0%`. It submitted 25 requests, consumed 24 without waiting, invalidated
one at a physical-risk transition, committed 192 aligned suffix actions, and
reported no worker errors. The remaining four misses are the deliberately
synchronous first chunk, two accurate stall replans, and post-recovery replan.

The async run uses 31.8% more VLA calls and 15 more control steps. It is thus a
latency/compute tradeoff, not a free speedup. Concurrent simulation rendering
also increased policy P95 slightly while remaining below 80 ms.

## Reproducibility And Negative Result

The repeated async run reproduced success, 236 steps, 29 VLA calls, the same
recovery response, identical prefetch counters, four deadline misses, and the
exact video SHA-256 digest. Its control P95 was `48.84 ms`.

An earlier v5 attempt invalidated all 19 prefetches because expected chunk-tail
staleness was treated as a hazard. It doubled work without reducing misses and
is retained as rejected engineering evidence. Only v6 and v7 support the
positive result.

This is a deterministic online acceptance gate on one T6 state, not a broad
benchmark estimate. A final paper claim requires a small matched expansion to
additional held-out task states.

## Held-Out T8/T9 Expansion

The expansion is complete on one fixed state each for T8 and T9. Because a
two-action lead yields an aligned eight-action suffix, the matched synchronous
control uses `commit=8`. Both conditions succeeded on both tasks. Async
prefetch used 84 calls versus 83, reduced 80 ms misses from `83/657` (`12.63%`)
to `3/662` (`0.45%`), and reduced episode wall time from `30.28 s` to
`25.88 s`. Per-task control P95 fell from `108.11--108.92 ms` to
`45.33--47.79 ms`.

This initial result is superseded for deployment selection by the ten-episode
expansion in `PREFETCH_5STATE_GATE.md`. On five states per task, synchronous
commit 8 reaches `7/10`, while full-duty and alternating prefetch each reach
`5/10`. Full-duty prefetch still reduces misses from `417/3931` to `48/4287`,
but both asynchronous schedules fail the closed-loop non-inferiority gate and
are rejected. Synchronous compiled BF16 remains the promoted deployment.

## Evidence

- `../carve_recovery/online_t6_control_profile_v4_20260717/`
- `online_t6_async_prefetch_gate_v5_20260717/` (rejected)
- `online_t6_async_prefetch_gate_v6_20260717/`
- `online_t6_async_prefetch_repeat_v7_20260717/`
- `heldout_t89_sync_boundfix_20260717/`
- `heldout_t89_sync_commit8_20260717/`
- `heldout_t89_async_20260717/`
- `heldout_t89_5states_sync_commit8_20260717/`
- `heldout_t89_5states_async_lead2_20260717/`
- `heldout_t89_5states_async_interval2_20260717/`
- `../../agentic_vla/runtime/prefetch.py`
- `../../scripts/run_agentic_vla_libero.py`

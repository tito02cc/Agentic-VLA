# CARVE Expanded Asynchronous-Execution Gate

Date: 2026-07-17

## Protocol

- Tasks: LIBERO-10 T8 and T9.
- Initial states: five fixed states per task, ten episodes per condition.
- Policy: frozen OpenPI `pi05_libero`, fixed deterministic noise.
- Deployment: accepted compiled-BF16 profile, two flow steps, RTX 4090.
- Deadline: 80 ms end-to-end control step.
- Recovery: deployable monitor plus bounded physical recovery enabled.

Synchronous commit 8 is the causal control for full-duty lead-2 prefetch because
the prefetch path discards two temporally covered actions from a ten-action
chunk. The alternating candidate requires one fresh synchronous chunk between
prefetched chunks.

## Result

| Execution | Success | T8 | T9 | VLA calls | Wall time | 80 ms misses |
|---|---:|---:|---:|---:|---:|---:|
| Sync, commit 8 | `7/10` | `4/5` | `3/5` | `417` | `205.79 s` | `417/3931` (`10.61%`) |
| Async, every chunk | `5/10` | `3/5` | `2/5` | `346` | `198.21 s` | `48/4287` (`1.12%`) |
| Async, alternating | `5/10` | `3/5` | `2/5` | `315` | `207.58 s` | `176/4278` (`4.11%`) |

Full-duty prefetch reduces relative deadline-miss rate by `89.4%`, but loses two
paired successes. The alternating duty cycle lowers prefetch exposure and
produces an intermediate miss rate, yet it does not recover success and does
not reduce total wall time relative to the synchronous control.

The full-duty run submitted 298 requests, consumed 290, invalidated eight,
waited once, and reported zero worker errors. The alternating run submitted 139,
consumed 137, invalidated two, never waited, and also reported zero errors.
Both verified every physical recovery they started. The regression is therefore
attributed to closed-loop trajectory change under temporally shifted visual
context, not an inference-service fault.

## Decision

- Reject both asynchronous schedules from promoted CARVE deployment profiles.
- Keep synchronous compiled BF16 with two flow steps and commit 10 as the
  fidelity-first default.
- Retain prefetch as experimental infrastructure and negative systems evidence.
- Do not continue lead, interval, or threshold sweeps without a method that
  predicts or corrects observation delay.
- The earlier two-episode positive gate is superseded for profile selection by
  this ten-episode expansion.

## Evidence

- `heldout_t89_5states_sync_commit8_20260717/`
- `heldout_t89_5states_async_lead2_20260717/`
- `heldout_t89_5states_async_interval2_20260717/`


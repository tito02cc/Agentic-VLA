# Static Masked-View Elision Gate

Date: 2026-07-17

## Method

Static Masked-View Elision (SMVE) removes camera slots that the deployment
adapter declares to be padding for every batch item. OpenPI's LIBERO transform
provides base and left-wrist images but pads the right-wrist slot with zeros and
sets its image mask to false. The stock PyTorch pi0.5 path still invokes SigLIP
for all three views before applying the masks.

The CARVE backend asserts that every configured view mask is false, removes the
view before visual embedding, shortens the language-model prefix and KV cache,
then uses the unchanged pi0.5 denoising path. An active view cannot be silently
removed. This is exact padding elimination under an adapter contract, not
learned token pruning or cross-observation caching.

## Replay Gate

- Model: frozen OpenPI `pi05_libero` PyTorch checkpoint.
- Hardware: RTX 4090, PyTorch 2.7.1+cu126.
- Controls: two flow steps, ten returned actions, fixed paired noise.
- Fidelity: all 45 recorded LIBERO failure observations, worst-case metrics.
- Performance: three warmup calls and 50 measured calls without competing GPU
  compute.

| Profile | Model P50 | Model P95 | Runtime P50 | Runtime P95 | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Compiled BF16 | `50.71 ms` | `51.30 ms` | `65.73 ms` | `67.40 ms` | `6.98 GB` |
| Compiled BF16 + SMVE | `41.14 ms` | `42.62 ms` | `54.35 ms` | `56.19 ms` | `6.97 GB` |

SMVE reduces model P50/P95 by `18.9%/16.9%` and runtime P50/P95 by
`17.3%/16.6%`. Both profiles have zero 80 ms model-path deadline misses.

Worst-case SMVE fidelity over 45 observations:

| First-action MAE | Chunk MAE | Chunk cosine | Gripper agreement | Endpoint L2 | Jerk RMSE |
|---:|---:|---:|---:|---:|---:|
| `0.00340` | `0.00232` | `0.999897` | `1.000` | `0.05524` | `0.00962` |

All metrics pass the predeclared action-fidelity thresholds.

## Closed-Loop Gate

The paired test uses the same five deterministic initial states on LIBERO-10
T8 and T9, fixed policy noise, two flow steps, commit 8, synchronous execution,
and the same Agentic physical-recovery controller.

| Profile | Success | T8 | T9 | VLA P50 | VLA P95 | Mean episode wall time |
|---|---:|---:|---:|---:|---:|---:|
| Compiled BF16 | `7/10` | `4/5` | `3/5` | `65.49 ms` | `69.59 ms` | `20.58 s` |
| Compiled BF16 + SMVE | `8/10` | `5/5` | `3/5` | `56.80 ms` | `60.60 ms` | `19.91 s` |

SMVE reduces closed-loop VLA P50/P95 by `13.3%/12.9%` and mean episode wall
time by `3.2%`. The ten-episode result supports non-inferiority for this gate;
the one-success difference is not treated as a task-success improvement claim.
Unlike asynchronous prefetch, SMVE does not shift the observation timestamp or
action commitment schedule.

## Decision

- Promote compiled BF16 + SMVE for pi0.5 deployments whose adapter guarantees
  a statically padded camera slot.
- Keep ordinary compiled BF16 as the fallback for deployments with all camera
  views active or with dynamic view availability.
- Retain W8A16 and asynchronous prefetch as rejected experimental profiles.
- Expand to another adapter/model before making a model-general speedup claim.

## Evidence

- `pi05_masked_view_compile_bf16_2step_h10_fidelity45_rep1.json`
- `masked_view_t89_5states_20260717/results.json`
- `masked_view_t89_5states_20260717/episode_traces.jsonl`
- `masked_view_t89_5states_20260717/policy_calls.jsonl`
- `../../carve_realtime/heldout_t89_5states_sync_commit8_20260717/results.json`
- `../../scripts/run_masked_view_closed_loop_gate.sh`

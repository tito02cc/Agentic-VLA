# CARVE Agent-VLM + VLA Co-resident Contention Gate

Date: 2026-07-17

## Protocol

- Frozen policy: OpenPI pi0.5 LIBERO PyTorch checkpoint.
- Agent module: Qwen3.5-4B BF16 server continuously processing a robot image.
- Hardware: one RTX 4090; both models share the same accelerator.
- VLA controls: two flow steps, ten returned actions, fixed paired noise.
- Each reported profile: five warmup calls and 500 measured calls.
- Static masked-view elision removes only the adapter-declared padded right-wrist slot.
- Fairness audit: **PASS**.

## Main Result

| Profile | Samples | P50 | P95 | P99 | Miss@80ms | Fidelity | System GPU use |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compiled BF16 | 500 | 81.12 ms | 91.85 ms | 93.72 ms | 72.8% | 10/10 pass | 17.63 GB |
| Compiled BF16 + SMVE | 500 | 65.93 ms | 75.57 ms | 78.90 ms | 0.6% | 10/10 pass | 17.58 GB |

SMVE reduces P50/P95/P99 by `18.7%/17.7%/15.8%`. At the 80 ms deployment deadline, it reduces misses from `72.8%` to `0.6%`.

## Deadline Sweep

| Deadline | Compiled BF16 miss | SMVE miss | Absolute reduction |
|---:|---:|---:|---:|
| 50 ms | 100.0% | 100.0% | 0.0 pp |
| 60 ms | 100.0% | 96.2% | 3.8 pp |
| 70 ms | 98.0% | 25.2% | 72.8 pp |
| 80 ms | 72.8% | 0.6% | 72.2 pp |
| 90 ms | 11.6% | 0.0% | 11.6 pp |
| 100 ms | 0.2% | 0.0% | 0.2 pp |
| 120 ms | 0.0% | 0.0% | 0.0 pp |

## VLM Load Audit

| Concurrent VLA profile | VLM requests | Success | VLM P50 | VLM P95 |
|---|---:|---:|---:|---:|
| Compiled BF16 | 131 | 131/131 | 1573.5 ms | 2292.2 ms |
| Compiled BF16 + SMVE | 126 | 126/126 | 1576.7 ms | 2078.8 ms |

## Event-triggered Agent VLM

The event-triggered condition inserts a five-second cooldown after each VLM response. It represents on-demand semantic checks rather than continuous reasoning.

| Profile | VLA P50 | VLA P95 | VLA P99 | Miss@80ms | VLM requests | VLM success |
|---|---:|---:|---:|---:|---:|---:|
| Compiled BF16 | 66.28 ms | 85.53 ms | 90.56 ms | 13.6% | 32 | 32/32 |
| Compiled BF16 + SMVE | 54.69 ms | 69.30 ms | 76.47 ms | 0.4% | 32 | 32/32 |

Relative to continuous VLM + ordinary compilation, event-triggered VLM + SMVE reduces P50/P95 by `32.6%/24.6%` and reduces 80 ms misses by `72.4` percentage points.

## Repetition Check

| Profile | Calls | P50 | P95 | Miss@80ms |
|---|---:|---:|---:|---:|
| Compiled BF16 | 50 | 80.79 ms | 90.62 ms | 70.0% |
| Compiled BF16 | 200 | 81.14 ms | 92.47 ms | 78.0% |
| Compiled BF16 | 500 | 81.12 ms | 91.85 ms | 72.8% |
| Compiled BF16 + SMVE | 50 | 65.98 ms | 74.69 ms | 0.0% |
| Compiled BF16 + SMVE | 200 | 66.15 ms | 75.01 ms | 2.0% |
| Compiled BF16 + SMVE | 500 | 65.93 ms | 75.57 ms | 0.6% |

## Interpretation Boundary

This gate measures warm inference under controlled single-GPU Agent-VLM contention. It supports a deployment/runtime claim, not a manipulation-success or universal-VLA claim. The VLM workload is deliberately continuous and is therefore a stress envelope rather than the expected event-triggered average load.

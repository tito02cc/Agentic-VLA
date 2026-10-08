# CARVE Efficient-Inference Ablation Status

Date: 2026-08-26

## Completion

The standalone Optimize Runtime and VLM Planner efficient-inference ablations
are complete on one NVIDIA RTX 4090. The machine-readable summary, CSV tables,
figure and full report are under:

`results/carve_efficiency_full_20260826/`

## VLA Result

| Profile | P95 | 80 ms miss | Peak VRAM | Fidelity |
|---|---:|---:|---:|---:|
| Eager BF16, 7 steps | 282.43 ms | 100% | 7.12 GiB | reference |
| Eager BF16, 2 steps | 151.35 ms | 100% | 7.12 GiB | reference |
| Compile BF16, 2 steps | 66.06 ms | 0% | 6.98 GiB | 45/45 pass |
| Compile BF16 + SMVE | 54.67 ms | 0% | 6.98 GiB | 45/45 pass |
| Late-language INT8 | 994.74 ms | 100% | 6.36 GiB | 45/45 pass |

Compiled BF16 + SMVE is the default realtime profile. It lowers P95 by 80.6%
relative to the seven-step eager reference. Compiled BF16 is the fallback when
the static masked-view contract is not satisfied.

The late-language INT8 result exposes a backend-version boundary. It reached
71.61 ms on the previously validated Torch 2.7.1 / TorchAO 0.15.0 stack and
completed 2/2 matched recovery branches, but regressed to 994.74 ms on the
current Torch 2.12.0 stack. It is therefore version-gated rather than promoted
as a universal acceleration profile.

## Planner Result

| Profile | Allocated VRAM | P95 | Paired semantic agreement | Gate |
|---|---:|---:|---:|---:|
| Qwen3.5-4B BF16 | 8.46 GiB | 1.90 s | 100% | pass |
| Qwen3.5-4B INT8 | 4.84 GiB | 6.50 s | 100% | pass |
| Qwen3.5-4B NF4 | 3.08 GiB | 2.91 s | 100% | pass |

BF16 remains the Planner latency tier. NF4 saves 63.6% allocated VRAM without
failing the semantic gate, so it is retained as a shared-GPU capacity tier.
INT8 is rejected because the current bitsandbytes kernel is both slower than
BF16 and larger than NF4.

## Claim Boundary

These experiments support a hardware-aware, fidelity-gated deployment claim:
CARVE selects optimization profiles by latency, memory, semantic/action
fidelity and closed-loop checks. They do not support the claim that
quantization universally accelerates VLA or VLM inference.

Verification: `222` project tests passed. No model service or compute process
was left running after the study.

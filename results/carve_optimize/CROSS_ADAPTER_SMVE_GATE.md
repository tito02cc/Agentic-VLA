# CARVE SMVE Cross-Adapter Gate

Date: 2026-07-17

## Question

Does the named static-view contract and SMVE backend remain valid beyond the
`pi05_libero` checkpoint and ten-action LIBERO adapter path?

This is a systems and action-fidelity experiment. It is not a DROID task-success
evaluation: the fixed RGB observations come from the existing CARVE replay
corpus and are encoded through the DROID input schema to hold the model input
constant across backends.

## Setup

- Model: official OpenPI `pi05_droid` JAX checkpoint, converted with the
  official `convert_jax_model_to_pytorch.py` script.
- Input adapter: OpenPI DROID schema with DROID normalization statistics.
- Model generation horizon: 15 actions; deployed committed horizon: 5 actions.
- Optimization: two flow steps, BF16, `torch.compile`, with or without SMVE.
- Static contract: canonical views `base_0_rgb`, `left_wrist_0_rgb`, and
  `right_wrist_0_rgb`; only `right_wrist_0_rgb` is declared padding.
- Fidelity: ten paired fixed-noise observations, worst-case aggregation.
- Performance: two warmup calls and twenty measured calls on an idle RTX 4090.

## Results

| Profile | Runtime P50 | Runtime P95 | Model P50 | Model P95 | 80 ms miss | Fidelity |
|---|---:|---:|---:|---:|---:|---:|
| compiled BF16 | 65.09 ms | 66.22 ms | 51.96 ms | 52.54 ms | 0% | 10/10 pass |
| compiled BF16 + SMVE | 54.66 ms | 57.34 ms | 41.57 ms | 42.70 ms | 0% | 10/10 pass |

Relative to ordinary compilation, SMVE reduces runtime P50/P95 by
`16.0%/13.4%` and model P50/P95 by `20.0%/18.7%`. Both profiles allocate
6.98 GB peak VRAM under this protocol.

The accepted SMVE profile has worst-case first-action MAE `0.00936`, chunk MAE
`0.00830`, cosine `0.99882`, gripper agreement `1.0`, endpoint L2 `0.07658`,
and jerk RMSE `0.00771`.

## Rejected Profiles

The same DROID model with all 15 generated actions committed is rejected before
benchmarking: endpoint L2 reaches `0.18120` against the `0.10` gate. A separate
base-LoRA checkpoint smoke also rejects ordinary compilation and SMVE because
both exceed the endpoint gate. Thresholds were not relaxed. These failures
show why deployment profiles must be checkpoint- and horizon-specific.

## Decision

The named view contract and pi0.5 SMVE plugin are validated across two OpenPI
checkpoints and two input adapters. This does not establish support for a
second VLA family. The paper may claim pi0.5 cross-adapter portability, while a
model-family-general claim remains deferred until another VLA plugin is tested.

## Evidence

- `pi05_droid_compile_cross_adapter_h5_10state_20260717.json`
- `pi05_droid_masked_view_cross_adapter_h5_10state_20260717.json`
- `pi05_droid_masked_view_cross_adapter_10state_20260717.json`
- `pi05_base_lora_compile_cross_checkpoint_5state_20260717.json`
- `pi05_base_lora_masked_view_cross_checkpoint_5state_20260717.json`


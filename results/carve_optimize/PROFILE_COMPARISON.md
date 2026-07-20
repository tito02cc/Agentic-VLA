# CARVE pi0.5 Deployment Profile Comparison

Date: 2026-07-17

## Protocol

- Model: OpenPI `pi05_libero` PyTorch checkpoint.
- Hardware: NVIDIA GeForce RTX 4090.
- Policy controls: two flow steps and ten committed actions.
- Performance: three warmup calls followed by 50 measured calls with an 80 ms
  deadline and no competing GPU compute workload.
- Fidelity: paired deterministic noise on all 45 recorded LIBERO failure
  observations; metrics use worst-case aggregation.
- Software: PyTorch 2.7.1+cu126 and TorchAO 0.15.0. TorchAO reported that its
  optional C++ extensions were incompatible with this PyTorch build, so W8A16
  used the Python tensor-subclass path together with `torch.compile`.

## Replay-Accepted Profiles

| Profile | P50 | P95 | P99 | Deadline misses | Peak VRAM | Fidelity observations |
|---|---:|---:|---:|---:|---:|---:|
| eager BF16 | 154.34 ms | 159.59 ms | 163.81 ms | 100% | 7.12 GB | reference |
| compiled BF16 | 65.73 ms | 67.40 ms | 67.94 ms | 0% | 6.98 GB | 45/45 pass |
| compiled BF16 + SMVE | 54.35 ms | 56.19 ms | 57.02 ms | 0% | 6.97 GB | 45/45 pass |
| compiled W8A16, VLM layers 0--3 | 69.74 ms | 71.72 ms | 72.41 ms | 0% | 6.56 GB | 45/45 pass |

Relative to eager BF16, compiled BF16 provides a 2.35x P50 speedup. The accepted
W8A16 profile provides a 2.21x P50 speedup and reduces peak VRAM by 7.82%.
Relative to compiled BF16, W8A16 reduces peak VRAM by 5.98% while increasing
P50 latency by 6.10%; both profiles satisfy the 80 ms deadline in this replay
protocol.

Static Masked-View Elision (SMVE) asserts and removes the permanently padded
right-wrist camera slot before SigLIP and prefix-KV construction. Relative to
compiled BF16, it reduces runtime P50/P95 by 17.3%/16.6% while preserving all
45 replay fidelity checks. The paired T8/T9 closed-loop gate reaches 8/10 versus
7/10 for compiled BF16 and reduces VLA P50/P95 by 13.3%/12.9%. The success
difference is not claimed as an improvement; it is used only as a non-inferior
deployment gate. See `MASKED_VIEW_ELISION_GATE.md`.

## Cross-Adapter SMVE Check

The same named static-view contract was tested with the converted OpenPI
`pi05_droid` checkpoint and DROID input adapter. With two flow steps and five
committed actions, both ordinary compiled BF16 and SMVE pass 10/10 paired
fixed-noise observations. SMVE reduces runtime P50/P95 from `65.09/66.22 ms`
to `54.66/57.34 ms` (`16.0%/13.4%`) and has zero 80 ms misses. This is a
systems/fidelity check over fixed replay inputs, not a DROID task-success
result. See `CROSS_ADAPTER_SMVE_GATE.md`.

"Replay-accepted" is not a final deployment decision. A later paired
long-horizon pilot found that W8A16 lost one T6 recovery success retained by
compiled BF16. W8A16 is therefore blocked by the closed-loop gate and compiled
BF16 remains the default profile.

## Worst-Case Fidelity

| Profile | First-action MAE | Chunk MAE | Cosine | Gripper agreement | Endpoint L2 | Jerk RMSE |
|---|---:|---:|---:|---:|---:|---:|
| compiled BF16 | 0.00353 | 0.00215 | 0.99994 | 1.000 | 0.05417 | 0.00848 |
| compiled BF16 + SMVE | 0.00340 | 0.00232 | 0.99990 | 1.000 | 0.05524 | 0.00962 |
| W8A16, VLM layers 0--3 | 0.00523 | 0.00357 | 0.99993 | 1.000 | 0.09359 | 0.00797 |

The endpoint-L2 acceptance limit is 0.10 in normalized action space.

## Rejected Profiles

| Quantized scope | Fidelity observations | Endpoint L2 | Decision |
|---|---:|---:|---|
| full language backbone | 5 | 0.18076 | rejected before benchmark |
| VLM language layers 0--5 | 45 | 0.10433 | rejected before benchmark |

An earlier visual-plus-language pilot also failed the endpoint gate at 0.13363
over five observations. It predates rejected-profile manifest persistence and
is retained only as a diagnostic result.

## Evidence

- `pi05_eager_bf16_2step_h10_deadline80_rep1.json`
- `pi05_torch_compile_reduce_overhead_bf16_2step_h10_deadline80_rep1.json`
- `pi05_torch_compile_bf16_2step_h10_fidelity45.json`
- `pi05_masked_view_compile_bf16_2step_h10_fidelity45_rep1.json`
- `pi05_droid_compile_cross_adapter_h5_10state_20260717.json`
- `pi05_droid_masked_view_cross_adapter_h5_10state_20260717.json`
- `pi05_torchao_w8a16_vlm_early4_fidelity45_deadline80_rep1.json`
- `pi05_torchao_w8a16_vlm_early_fidelity45.json`
- `pi05_torchao_w8a16_language_compile_2step_h10_deadline80_smoke.json`

These are replay and systems results. Closed-loop task success remains a
separate acceptance stage.

Both accepted optimized profiles also pass the real controlled-WebSocket and
ten-step paired LIBERO branch integration smoke. Across 14 calls per profile,
compiled BF16 averaged 61.40 ms and W8A16 averaged 62.24 ms, with zero 80 ms
deadline misses. See
`../carve_profile_branches/PROFILE_BRANCH_SMOKE_COMPARISON.md`. This integration
smoke is not used as task-success evidence.

The subsequent outcome-bearing pilot is documented in
`../carve_profile_branches/PAIRED_FAILURE_STATE_PILOT.md`.

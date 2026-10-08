# Real-Time VLA Engineering Decisions

Updated: 2026-07-24

## External Context

- [Running VLAs at Real-time Speed](https://arxiv.org/abs/2510.26742) demonstrates
  that pi0-level VLA latency can be reduced through CUDA graphs, graph-level
  transforms, and kernel optimization; it also explicitly places high-level VLM
  reasoning at a much lower frequency than action control.
- [Realtime-VLA V2](https://arxiv.org/abs/2603.26360) frames deployment as a
  system problem spanning model execution, calibration, planning, and control,
  rather than a model-bitwidth problem alone.
- [VLA-RAIL](https://arxiv.org/abs/2512.24673) motivates decoupling expensive
  inference from action execution and makes transition continuity a first-class
  deployment concern.

## CARVE Position

CARVE does not claim a replacement CUDA runtime or a new quantizer. Its
contribution is an evidence-preserving deployment layer for a frozen VLA:

1. **Profiled model-critical path:** compiled BF16 + Static Masked-View Elision
   removes a known padded camera slot only under a validated named view contract.
2. **Fail-closed deployment:** profile identity, checkpoint, hardware, action
   horizon, replay fidelity, realtime behavior, and matched closed-loop status
   must agree before a backend is served. An admitted compiled-BF16 fallback is
   mandatory when the static-view contract is violated.
3. **Frequency separation:** PI0.5 remains the action generator; a VLM planner
   is asynchronous and event-bound. Seconds-scale semantic latency is never
   counted as 80 ms VLA control latency.
4. **Trace separation:** model inference, simulator/robot control, safe planner
   boundary wait, and fallback outcomes are emitted as distinct measurements.

## Quantization Decision

Component-scoped PI0.5 W8A16 is implemented and replay-fidelity checked, but it
is not promoted. It lowers peak memory from `6.98 GB` to `6.56 GB` in the
calibration protocol while being slower than compiled BF16 and losing one
matched long-horizon recovery outcome. Generic W8/W4/INT4 quantization is
therefore not a thesis headline for CARVE.

The appropriate thesis statement is: **CARVE makes precision an admitted
deployment choice, not an assumed acceleration.** Quantization remains an
optional edge-memory branch, eligible only if a future device-constrained
profile passes the same action and closed-loop gates.

## Deferred Candidates

- Custom CUDA-graph/kernel work comparable to Realtime-VLA is out of scope for
  a PyTorch PI0.5 systems thesis unless it can be reproduced under the same
  action-fidelity contract.
- Full-duty asynchronous action prefetch is implemented but remains rejected by
  the existing closed-loop gate; it will not be promoted merely because it
  lowers deadline misses.
- TensorRT/FP8 is a future hardware-specific candidate. It must not be added to
  a result table without the same replay, warm-runtime, and matched-state gates.

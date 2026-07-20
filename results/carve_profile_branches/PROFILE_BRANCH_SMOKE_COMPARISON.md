# CARVE Profile and Agentic Branch Integration Smoke

Date: 2026-07-16

## Purpose

This experiment verifies that an accepted CARVE Optimize Runtime profile can be
used by the real pi0.5 WebSocket path while the Agentic Harness restores a
recorded LIBERO failure state and executes continue, fast, accurate, and
recovery branches. It validates profile identity, fixed inference controls,
Agentic trace context, deterministic state restoration, and deadline
measurement in one path.

The branch horizon is ten simulator steps. It is deliberately a systems smoke,
not task-success or recovery-effectiveness evidence.

## Protocol

- Checkpoint: OpenPI `pi05_libero` PyTorch.
- Hardware: NVIDIA GeForce RTX 4090.
- Snapshot: `task06_episode000_step0019.npz`.
- Controls: two flow steps, ten returned actions, fixed deterministic noise.
- Deadline: 80 ms per VLA request.
- Branches: continue, fast, accurate, and recovery.
- Calls: 14 per deployment profile.

## Runtime Results

| Deployment profile | Calls | Mean | P50 | P95 | Maximum | Deadline misses |
|---|---:|---:|---:|---:|---:|---:|
| compiled BF16 | 14 | 61.40 ms | 61.73 ms | 66.17 ms | 70.67 ms | 0/14 |
| compiled W8A16, VLM layers 0--3 | 14 | 62.24 ms | 62.90 ms | 67.92 ms | 68.98 ms | 0/14 |

| Branch | Calls | BF16 mean | W8A16 mean |
|---|---:|---:|---:|
| continue | 2 | 67.17 ms | 65.90 ms |
| fast | 2 | 63.32 ms | 64.15 ms |
| accurate | 5 | 60.79 ms | 61.18 ms |
| recovery | 5 | 61.35 ms | 63.52 ms |

W8A16 does not improve latency over compiled BF16 in this batch-one protocol.
Its accepted benefit is lower peak VRAM, measured separately as 6.56 GB versus
6.98 GB, while preserving the deadline and replay fidelity gates.

## Deployment Cold Start

In a fresh W8A16 server process, two server-side prewarm calls took 39.17 s and
1.64 s. The WebSocket server opened only after those calls. The first external
Agentic request then took 78.01 ms and met the 80 ms deadline. Cold preparation
is therefore reported separately and is never counted as steady control-loop
latency.

## Evidence

- `compiled_bf16_snapshot1_h10_fixed2.json`
- `compiled_bf16_snapshot1_h10_fixed2_policy_calls.jsonl`
- `w8a16_snapshot1_h10_fixed2.json`
- `w8a16_snapshot1_h10_fixed2_policy_calls.jsonl`
- `../carve_optimize/compiled_bf16_agentic_websocket_warm2_smoke.json`
- `../carve_optimize/w8a16_agentic_websocket_warm2_smoke.json`
- `../carve_optimize/w8a16_server_prewarm_first_request_smoke.json`

Every policy-call trace records the deployment manifest, backend identity,
requested and applied controls, latency, deadline result, and Agentic context.

Outcome-bearing branches with 280/440-step horizons are reported in
`PAIRED_FAILURE_STATE_PILOT.md`. Those results supersede this smoke for profile
promotion decisions.

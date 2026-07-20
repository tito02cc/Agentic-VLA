# CAQ-Lite Method and Current Results

Date: 2026-06-09

## Method Positioning

CAQ-Lite means **Criticality-Aware Quantized Lightweight Runtime for VLA Policies**.

The method is not a new base VLA and not a direct replication of VLA PTQ methods. It is a runtime wrapper for frozen VLA policies. The core question is:

> When is low-cost inference safe for embodied control, and when must the system fall back to full VLA inference or Agentic recovery?

## Execution Paths

| Path | Trigger | Runtime behavior |
|---|---|---|
| Full-Precision Critical Path | grasp, contact, release, perturbation lockout, recovery, progress regression | full VLA inference, short commit, no aggressive reuse |
| Quantized Normal Path | stable low-risk approach or transport | quantized VLA backend when available |
| Reuse Fast Path | low risk, smooth progress, no lockout | reuse cached action suffix and skip full VLA calls |

Current implementation uses `CAQ-Proxy`: the quantized normal path is represented by the existing Light-Reuse fast path. This keeps the online control path stable while leaving the backend replaceable by W8/W4 serving later.

## Difference From Related Work

| Related direction | Main focus | CAQ-Lite difference |
|---|---|---|
| QuantVLA / QVLA / ActQuant | PTQ and action-sensitive bit allocation | Does not claim a new calibration rule; uses quantization as a backend under task criticality |
| EfficientVLA / SQAP-VLA | model-internal pruning and compression | Keeps the frozen VLA unchanged and wraps serving/control behavior |
| ElegantVLA | dynamic compute inside VLA modules | Schedules full inference, reuse, quantized path, and Agentic fallback |
| Realtime-VLA / RTC / AAC | faster action chunk execution | Couples realtime execution with perturbation lockout and progress-aware safety |

## Verified Components

### Q0: Module Profile

- Checkpoint: pi0.5 LIBERO PyTorch checkpoint.
- Total parameters: `3.62B`.
- Estimated tensor size: `6.74 GB`.
- W8 candidates: `293 tensors / 2.40B params`.
- Guarded W8 candidates: `165 tensors / 691.18M params`.
- Keep-FP tensors: `354 tensors / 527.27M params`.

Reports:

- `docs/caq_lite_pi05_libero_profile_20260609.md`
- `docs/caq_lite_pi05_libero_runtime_modules_cpu_20260609.md`

### Q1: W8 Backend Smoke

bitsandbytes `Linear8bitLt` runs on real pi0.5 weights with about `1.2%-1.4%` mean relative error in tested layers.

On RTX 4090, batch size 1:

- W8 was not faster than FP16 for the tested single-layer cases.
- W8 should be positioned as a low-memory or edge backend, not as the current latency source.

### Q2: Online CAQ-Proxy Rollout

Small smoke, `mid_nudge_xy=0.03`:

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Wall / ep | SUD@80ms |
|---|---:|---:|---:|---:|---:|---:|
| `B0-VLA-CAQ-Proxy` | `3/3` | `35.33` | `18.33` | `0.342` | `24.75 s` | `0.866` |
| `B4-Agentic-CAQ-Proxy` | `3/3` | `46.00` | `16.67` | `0.266` | `30.79 s` | `0.854` |

Strong stress, `mid_nudge_xy=0.05`:

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Wall / ep | SUD@80ms |
|---|---:|---:|---:|---:|---:|---:|
| `B0-VLA-CAQ-Proxy` | `19/20` | `34.30` | `17.70` | `0.340` | `24.08 s` | `0.822` |
| `B4-Agentic-CAQ-Proxy` | `19/20` | `36.05` | `16.80` | `0.318` | `24.95 s` | `0.817` |

## Current Claim Boundary

Supported claims:

- CAQ-Lite can be integrated with the existing VLA serving path.
- CAQ-Lite can be used without Agentic Harness as an independent VLA realtime runtime.
- Agentic Harness and CAQ-Lite can be coupled through the same experiment runner and trace schema.
- Current realtime gains mainly come from criticality-aware action reuse and skipped full VLA calls.

Unsupported claims:

- Do not claim W8 directly accelerates pi0.5 inference on RTX 4090 batch size 1.
- Do not claim the Task2 CAQ strong-stress result proves Agentic recovery gains, because recovery was not triggered.
- Do not claim a new PTQ calibration method.

## Paper Usage

Recommended paper wording:

> We propose a criticality-aware VLA runtime that allocates call frequency and inference backend according to embodied control risk. In the current implementation, low-risk phases use a reuse-based lightweight path, while critical phases fall back to full VLA inference and can be coupled to Agentic recovery. Quantized VLA execution is supported as a backend interface and validated at the layer-smoke level, but the main online realtime gains come from skipped full VLA calls under safety gating.


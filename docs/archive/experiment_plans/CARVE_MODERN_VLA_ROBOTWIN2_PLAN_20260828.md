# CARVE Modern VLA and RoboTwin 2.0 Validation Plan

Date: 2026-08-28

> Superseded later on 2026-08-28 by
> `CARVE_ROBODOJO_CURRENT_MODEL_PLAN_20260828.md`. The legacy LingBot-VLA 1.0
> checkpoint and its pending experiment pipeline were removed at the user's
> request. This document is retained only as a decision record.

## Decision

The local RoboTwin checkout is already the current official RoboTwin 2.0 code line
(`RoboTwin-Platform/RoboTwin`, local commit `3095469`, 2026-08-20). The component
that needs modernization is the second VLA: the in-progress LingBot-VLA 1.0 4B
checkpoint is retained only as an adapter smoke test, not as the main large-scale
evaluation target.

The main validation matrix becomes:

| Role | Policy | Benchmark | Training required | Purpose |
|---|---|---|---|---|
| Existing primary VLA | PI0.5 | LIBERO-Pro evidence already collected | No new training | Preserve the established CARVE baseline |
| Efficient second VLA | TurboVLA RoboTwin checkpoint | RoboTwin 2.0 clean subset | No | Test model-independent integration and low-latency runtime |
| Current generalist VLA | LingBot-VLA 2.0 6B RoboTwin checkpoint | RoboTwin 2.0 clean/randomized subset | No | Test the latest generalist VLA and Agentic Harness coupling |
| Robustness extension | Best admitted policy above | RoboTwin 2.0-Plus | No | Measure recovery under structured distribution shift |

StarVLA Qwen3-VL-OFT-RoboTwin2-All is kept as a fallback adapter. Its public
checkpoint is about 9.1 GiB and is suitable for a 24 GiB RTX 4090, but it is not
preferred over TurboVLA because TurboVLA provides a stronger efficient-inference
comparison for this thesis.

## Why these targets

### LingBot-VLA 2.0

- Official RoboTwin 2.0 post-training weights were released on 2026-07-25.
- The model uses Qwen3-VL, a sparse MoE action expert, a 55-dimensional unified
  action representation, and predictive visual/depth distillation.
- The official repository reports 93.52% on RoboTwin 2.0 clean and 92.80% on the
  randomized setting, and about 130 ms per inference call on RTX 4090D with ten
  denoising steps.
- The checkpoint is about 23.77 GiB. Admission therefore starts with an isolated
  load and one fixed-observation call before co-resident VLM Planner testing.

### TurboVLA

- Released in July 2026 with public RoboTwin 2.0 and LIBERO checkpoints.
- The reported policy has 0.2B parameters, about 31.2 ms latency and 0.9 GB VRAM
  on RTX 4090; its RoboTwin 2.0 clean50 shared checkpoint reports 60.2% success.
- It directly exercises CARVE Optimize Runtime across a very different policy
  architecture, making the portability claim stronger than another large VLA alone.

### RoboTwin 2.0-Plus

- It is backward compatible with RoboTwin 2.0 and uses the same simulator stack.
- It adds structured perturbations over objects, background, lighting, camera,
  robot state, language and sensor noise.
- These perturbations create recoverable failures and semantic ambiguity, which
  are better aligned with Monitor, VLM Critic/Planner, memory and bounded recovery
  than saturated clean-only success rates.

## Execution gates

### Gate A: preserve completed work without overspending

1. Finish the already 77%-downloaded LingBot-VLA 1.0 checkpoint.
2. Complete one direct and one Agentic RoboTwin smoke episode.
3. Do not run the former eager/compile full sweep unless explicitly requested.

### Gate B: TurboVLA adapter and efficiency evidence

1. Download the public RoboTwin checkpoint and its required frozen encoders.
2. Implement the CARVE policy-server adapter using the existing observation/action
   contract.
3. Run fixed-observation latency, VRAM, throughput and action-shape admission.
4. Run paired direct versus Agentic episodes on three task families, reusing exactly
   the same seeds and initial states.

### Gate C: LingBot-VLA 2.0 admission

1. Create the official Python 3.12 / PyTorch 2.8 environment separately.
2. Download the RoboTwin post-training checkpoint outside the repository.
3. Measure isolated load VRAM and one-call latency on the RTX 4090.
4. If VLA plus local NF4 Planner fits, run both concurrently. Otherwise keep the
   event-triggered Planner process offloaded or invoke it sequentially; record the
   deployment mode rather than silently changing the architecture.
5. Run the same paired task/seed subset used for TurboVLA.

### Gate D: robustness-focused benchmark

Use one clean condition and a compact perturbation set:

- camera pose or viewpoint shift;
- object appearance or distractor change;
- language paraphrase or ambiguity;
- sensor noise;
- robot-state perturbation where supported.

For each condition report success, progress, intervention/recovery precision,
false-trigger regressions, VLA calls, planner calls, wall time, P50/P95 latency and
deadline-miss rate. Large all-task sweeps are run only after a three-seed pilot shows
a measurable CARVE effect.

## Interoperability target

The AllenAI `vla-evaluation-harness` protocol is used as an external compatibility
reference, not as a replacement for CARVE. CARVE should expose a thin model-server
boundary and preserve benchmark version, seed, episode, latency and video metadata
so that a future bridge can compare CARVE results with the public harness.

## Deferred options

- VLA-Arena is scientifically current and includes safety, distractor,
  extrapolation and long-horizon tasks, but it is deferred until a matching public
  checkpoint can be evaluated without task-specific fine-tuning.
- RoboDojo is deferred because its Isaac Sim 5.1 environment and large container
  add substantial setup cost without improving the immediate adapter evidence.
- A full 50-task x 100-trial RoboTwin sweep is not a first gate on a single 4090.

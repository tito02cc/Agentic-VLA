# CARVE RoboDojo and Current-Model Validation Plan

Date: 2026-08-28

> Status: superseded as the primary route by
> `CARVE_CURRENT_BENCHMARK_MODEL_PLAN_20260828.md`. This file remains the
> canonical record of RoboDojo preparation, model selection and storage gates.

## Decision

RoboDojo becomes the primary new benchmark for the Agentic Harness study. It is
more aligned with CARVE than another saturated clean benchmark because its 42
simulation tasks explicitly measure Generalization, Memory, Precision,
Long-Horizon execution and Open instruction following.

The validation matrix is:

| Role | Model | Benchmark | New training | Local artifact size |
|---|---|---|---|---:|
| Existing foundation baseline | PI0.5 RoboDojo checkpoint | RoboDojo | No | 11.59 GiB |
| Current primary model | Xiaomi-Robotics-1 RoboDojo checkpoint | RoboDojo | No | 10.25 GiB |
| Efficient policy control | TurboVLA RoboTwin checkpoint | RoboTwin 2.0 | No | 0.81 GiB |
| Optional robustness extension | Admitted model above | RoboTwin 2.0-Plus | No | Reuses checkpoint |

LingBot-VLA 1.0 is removed. LingBot-VLA 2.0 is not the first new target because
its 23.77 GiB checkpoint leaves almost no memory headroom on a 24 GiB RTX 4090.

## Why RoboDojo

- Released in July 2026 and built on Isaac Sim / Isaac Lab.
- 42 simulation tasks plus 12 random-layout variants and 18 real-robot tasks.
- Native partial-progress score as well as binary success rate.
- XPolicyLab isolates policy dependencies behind a WebSocket server, matching
  CARVE's model-server and benchmark-adapter architecture.
- Official data host contains inference checkpoints for more than 30 policies,
  so policy evaluation does not inherently require fine-tuning.
- The task taxonomy directly exposes the capabilities CARVE targets: task memory,
  multi-stage execution, semantic tool selection and recovery.

## Model selection

### Xiaomi-Robotics-1

This is the preferred first model.

- July 2026 VLA release with a Qwen3-VL-based policy stack.
- Official RoboDojo leaderboard: score 20.07, success rate 13.93%, currently third.
- Official XPolicyLab adapter and official inference checkpoint are available.
- The checkpoint is about 10.25 GiB, materially easier to admit on one RTX 4090
  than LingBot-VLA 2.0 or G0.5.
- Evaluation uses the `arx_x5` embodiment and end-effector action mode.

### PI0.5

PI0.5 remains the controlled baseline because CARVE already supports its server
contract and an official RoboDojo simulation checkpoint is available. The public
leaderboard reports score 11.41 and success rate 6.91%.

### Why G0.5 is deferred

Galaxea G0.5 is scientifically attractive and ranks second on RoboDojo, but the
public RoboDojo FM-only archive is about 32 GiB before extraction. It is deferred
until storage is expanded or a smaller inference release is published.

### Why TurboVLA stays on RoboTwin 2.0

TurboVLA has a small public RoboTwin checkpoint and directly supports the thesis's
efficient-inference claim. It does not currently provide a matched public RoboDojo
checkpoint, so forcing it onto RoboDojo would require training and would defeat the
low-cost evaluation goal.

## CARVE-focused task subset

The first pilot uses tasks where a high-level Agent can plausibly improve execution:

| Dimension | Task | CARVE capability under test |
|---|---|---|
| Memory | `imitate_sorting_sequence` | episodic task memory and ordered subgoals |
| Memory | `press_by_number` | observation history and delayed semantic choice |
| Long-Horizon | `organize_table` | decomposition, progress tracking and replanning |
| Long-Horizon | `classify_objects` | repeated tool use and stage completion |
| Open | `classify_objects_by_language` | VLM semantic planning and grounding |
| Open | `pour_by_language` | language-conditioned tool/object selection |
| Generalization | `sweep_blocks` plus random variant | failure detection and recovery under layout shift |
| Precision | `insert_tubes` | bounded retry without unsafe repeated execution |

Start with one episode per task for infrastructure admission, then three seeds and
five episodes per seed for the paired pilot. Only expand to the native protocol if
the pilot produces a measurable effect.

## Compared conditions

1. Direct VLA policy.
2. CARVE monitoring and bounded recovery, without VLM Planner.
3. Full CARVE with event-triggered VLM Planner/Critic, memory and tools.
4. Full CARVE plus admitted Optimize Runtime profile.

All conditions reuse task, initial layout, episode id and seed. Report native score,
success, progress delta, VLA calls, Planner calls, interventions, recovered failures,
false-trigger regressions, wall time, VRAM, P50/P95 inference latency and deadline
misses.

## One-GPU deployment

RoboDojo's simulator and policy can run as separate processes over WebSocket. On one
RTX 4090, admission proceeds in this order:

1. Start Isaac Sim alone and record stable simulator VRAM.
2. Start Xiaomi-Robotics-1 alone and record model VRAM and one-call latency.
3. Test co-residency with conservative simulator parallelism (`num_envs=1`).
4. Keep the semantic Planner event-triggered and remote during the first closed-loop
   pilot so it does not compete for GPU memory.
5. Test the local quantized Planner only after simulator plus policy memory headroom
   is measured.

This is still a complete Agentic system: the Planner backend is replaceable and its
endpoint, model, call count and latency are recorded in every receipt.

## Storage and execution gate

The official RoboDojo evaluation assets require about 90 GiB. The machine currently
has about 63 GiB free on `/` and 36 GiB on `/home`, with insufficient contiguous
space for assets, checkpoint, container and runtime cache. Therefore:

1. Prefer the official cloud evaluation path when access is available; only the
   local policy server and checkpoint are then required.
2. Otherwise reclaim or attach at least 120 GiB of working space before local
   Isaac Sim installation.
3. Do not begin the 90 GiB asset download until the storage gate passes.

## Immediate engineering sequence

1. Clone RoboDojo and its pinned XPolicyLab adapter code without downloading assets.
2. Run `doctor` and dry-run checks to validate source and command wiring.
3. Add a CARVE RoboDojo benchmark adapter around the native WebSocket contract.
4. Download only the Xiaomi-Robotics-1 inference checkpoint after the storage path
   is selected.
5. Run model-only load, fixed-observation inference and runtime profiling.
6. Run simulator smoke through cloud evaluation or a local asset installation.

## Current preparation status

Completed on 2026-08-28:

- Removed the incomplete LingBot-VLA 1.0 checkpoint and stopped its experiment
  pipeline, releasing about 13 GiB.
- Cloned RoboDojo official commit `2184bf8` without large assets.
- Initialized RoboDojo's pinned XPolicyLab commit `432f82b`.
- Passed source, task-inventory and environment-config checks: 54 runnable configs,
  no missing task classes or configs.
- Confirmed the expected failures are only the four undownloaded asset groups and
  skipped Isaac/Conda checks.
- Passed Xiaomi-Robotics-1 single-task command generation for `stack_bowls`.
- Passed an eight-task dry-run covering Memory, Long-Horizon, Open,
  Generalization and Precision categories.
- Started the scoped official Xiaomi-Robotics-1 checkpoint download in tmux session
  `carve-robodojo-xiaomi-model`; the command includes only
  `ckpt/RoboDojo/Xiaomi_Robotics_1/*` and does not download RoboDojo datasets or
  simulator assets.
- After the broader benchmark survey, paused the download at about 9% and retained
  its `.aria2` resume state. RoboDojo is now a gated P1 validation rather than the
  experiment that blocks WatchAct, RoboMME or RoboTwin 2.0 work.

No simulator success result is claimed at this stage. The next valid evidence is a
model-only inference receipt followed by an Isaac-backed or official cloud rollout.

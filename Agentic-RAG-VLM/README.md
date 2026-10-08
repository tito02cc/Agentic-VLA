# Agentic RAG-VLM — Reconstructed Framework and VLABench Validation

This directory is a reusable reconstruction of the Agentic RAG-VLM paper
architecture, an auditable mechanism evaluation, and selected complete
VLABench manipulation tasks. The earlier Guanghua environment is retained as
legacy engineering evidence; current presentation work uses VLABench.

> **Public-clone note:** the source-machine Guanghua robot meshes/URDF bundle,
> model weights, original paper-era experience data and most raw `output/`
> files are not distributed in the public GitHub repository. Some commands and
> links below require these locally obtained assets. See the root
> [public-upload scope](../docs/handoff/PUBLIC_REPOSITORY_SCOPE_20261008.md),
> [model guide](../MODEL_AND_ENVIRONMENT_SETUP.md) and this project's
> [reproduction boundary](docs/REPRODUCTION.md) before claiming a rerun.

## Project at a glance

**Problem.** Turn multimodal task candidates into auditable, constrained, and
failure-aware robot decisions without retraining a VLA policy.

**System.** Public RGB-D → Qwen3.5-VL tool router + planner → HAA-RAG / deterministic Agent tools →
scene-graph-constrained IK/hand skills → verification, memory, and bounded replan.

| Evidence | Verified result |
|---|---:|
| Frozen paired evaluation | 100 method rollouts / 10 seeds |
| Real local multimodal inference | 246 Qwen3.5-4B calls / 0 endpoint errors |
| Full Agent | G1–G4: 10/10 each |
| Audit coverage | 260 public events / 60 Agent-tool calls / PASS |
| Regression suite | 41 tests passed |
| Current VLABench evidence | 3 final task outcomes / 1,166 accepted IK calls; chemistry cleanup pending |
| New defense reel | 84 s composition and snapshots ready / strict recheck and final render pending |

- [Current experiment progress and defense narrative](docs/CURRENT_EXPERIMENT_PROGRESS.md)
- [Handoff package index](HANDOFF_PACKAGE_INDEX.md)
- [Kiro engineering handoff](KIRO_HANDOFF.md)
- [Complete Kiro / Claude Opus 5 Max handoff prompt](KIRO_OPUS5_MAX_PROMPT.md)
- [Legacy resume/interview wording (use current evidence from the handoff)](docs/RESUME_PROJECT_PACKAGE.md)
- [Chinese paper/method introduction (engineering-status sections are historical)](docs/PROJECT_INTRODUCTION_CN.md)
- [Challenge v4 statistical report](output/qwen35_challenge_v4_agent_routing_main/CHALLENGE_V4_REPORT.md)
- [Challenge v4 design and plain-language results](docs/AGENT_TOOLS_V4_RESULTS.md)
- [Legacy 67-second Guanghua defense video](output/defense_suite/agentic_rag_vlm_full_defense_reel_v9.mp4)
- [VLABench 84-second reel project and speaking guide](video/vlabench_defense_reel/DEFENSE_GUIDE.md)

**Claim boundary.** The project demonstrates real multimodal high-level planning,
public-observation tool execution, and completion of three selected VLABench
tasks through fixed skills and IK. It does not claim benchmark-wide success,
real-robot/contact-safety certification, end-to-end VLA control, or sim-to-real.
The earlier Guanghua G0-B contact-dynamics grasp remains `NOT_ADMITTED`.

## Layout

- `assets/guanghua_hand_env/mjcf/guanghua_hand_env.xml`: canonical MuJoCo scene.
- `assets/guanghua_hand_env/urdf/guanghua_robot_with_hands.urdf`: portable robot URDF.
- `assets/guanghua_hand_env/urdf/`: robot and dexterous-hand meshes plus source URDF variants.
- `assets/guanghua_hand_env/textures/`: floor, wall, table, and metal textures.
- `scripts/prepare_assets.py`: reproducibly generates the canonical MJCF and URDF.
- `scripts/run_guanghua_env.py`: headless simulation, rendering, and interactive viewer entry point.
- `scripts/run_g0_admission.py`: RGB-D perception, safe-corridor IK, and physical pregrasp admission (G0-A).
- `scripts/run_agentic_pilot.py`: paired G1-G3 HAA-RAG, graph, memory, and replanning pilot.
- `scripts/run_qwen_vlm_pilot.py`: real Qwen multimodal paired experiment with raw response receipts.
- `scripts/run_qwen_vlm_challenge.py`: leakage-resistant G1-G4 challenge with negative controls and bounded repair.
- `scripts/render_complete_task_demo.py`: continuous two-object Agentic task rollout using saved real-Qwen receipts.
- `scripts/render_defense_suite.py`: one-command four-scenario rollout/caption/manifest entry point.
- `scripts/run_traditional_vs_agentic.py`: frozen T0/T1/A2 paired mechanism benchmark.
- `scripts/run_recovery_hierarchy_experiment.py`: L1/L2/L3 recovery and negative-control benchmark.
- `scripts/render_paired_comparisons.py`: paired MuJoCo safety and target-change videos.
- `scripts/validate_defense_suite.py`: scenario semantics, runtime trace, motion gate, and video-contract audit.
- `scripts/render_answer_video.sh`: adds defense-ready evidence captions without changing the simulator frames.
- `scripts/validate_complete_task_demo.py`: checks video properties, evaluator thresholds, Qwen receipts, and proxy disclosure.
- `scripts/summarize_qwen_vlm_pilot.py`: Qwen metrics, Wilson intervals, tokens, and latency report.
- `scripts/summarize_qwen_vlm_challenge.py`: challenge statistics, event slices, difficulty slices, and cost report.
- `scripts/validate_agentic_artifacts.py`: public/private evidence-isolation audit.
- `scripts/validate_resume_evidence.py`: rechecks every numeric claim used in the resume package.
- `scripts/guanghua_control.py`: right-arm IK / trajectory control and hand synergies.
- `scripts/guanghua_perception.py`: public RGB-D color grounding without object-pose access.
- `agentic_rag_vlm/`: reusable ReAct pipeline, quality model, recovery, memory, and adapter interfaces.
- `agentic_rag_vlm/runtime.py`: reusable multi-subgoal observe/plan/execute/verify/monitor/replan state machine.
- `knowledge_base/affordance_cards.json`: compact, auditable Guanghua affordance-card library.
- `tests/test_guanghua_assets.py`: asset-path, physics, table, and actuator checks.
- `configs/guanghua_experiment_protocol.json`: frozen scene, condition, seed, metric, and evidence contract.
- `configs/guanghua_challenge_v3.json`: frozen ten-seed challenge and public-observation Agent-tool contract.
- `docs/GUANGHUA_EXPERIMENT_PLAN.md`: midterm-oriented Agentic RAG-VLM experiment design.
- `docs/REPRODUCTION.md`: paper reconstruction boundary and extension points.
- `docs/PILOT_RESULTS.md`: frozen five-seed mechanism results.
- `docs/QWEN_VLM_RESULTS.md`: primary real-Qwen 20-seed main results.
- `docs/CHALLENGE_V2_RESULTS.md`: upgraded 100-rollout result and interpretation boundary.
- `docs/AGENT_TOOLS_V3_RESULTS.md`: final 100-rollout Agent-tool results, statistics, and audit boundary.
- `docs/COMPLETE_TASK_DEMO.md`: complete success-video evidence, reproduction, and claim boundary.
- `docs/DEFENSE_SUITE_RESULTS.md`: four-scenario Guanghua defense suite and presentation guidance.
- `docs/TRADITIONAL_VS_AGENTIC_RESULTS.md`: fair baseline definitions and E1–E4 results.
- `docs/RECOVERY_HIERARCHY_RESULTS.md`: failure-correction experiment and evidence boundary.
- `docs/EXPERT_ASSESSMENT_20260830.md`: defense readiness, limitations, and likely questions.
- `docs/RESUME_PROJECT_PACKAGE.md`: role-specific resume bullets, HR pitch, evidence ledger, and interview questions.
- `docs/PROJECT_INTRODUCTION_CN.md`: paper method, engineering implementation, experiments, resume copy, and interview narrative.
- `docs/CURRENT_EXPERIMENT_PROGRESS.md`: current quantitative evidence, complete VLABench tasks, claim boundary, and defense structure.

The school-provided source files are retained as
`guanghua_hand_env.bundle.xml`, `guanghua_hand_env.original.xml`, and the
original URDF variants. The canonical files use relative paths and can be moved
with this directory.

## Environment contents

- fixed Guanghua upper-body robot with two five-finger hands;
- 0.8 m × 0.8 m table, 0.8 m surface height, and collision-enabled legs;
- floor and visual room walls;
- a red cube, blue cylinder, translucent yellow protected-glass vessel, two target zones, and a staging zone;
- `frontview`, `birdview`, `agentview`, and moving `right_wrist` cameras;
- position actuators for the head, waist, both 7-DoF arms, and 22 hand joints;
- simplified collision proxies for the palms, finger segments, and fingertips.

The collision proxies are deliberately simple. Arm-link collision is currently
disabled because uncalibrated capsules incorrectly blocked reachable tabletop
motions; it must be added from measured link geometry before safety claims. The cylindrical task objects use
stable box-shaped collision proxies while retaining cylindrical visual geometry.
They are appropriate for Agentic mechanism qualification, but should be
calibrated or replaced before making fine contact-dynamics claims.

The arm executor uses multi-start IK with Cartesian-axis and continuity
constraints plus model-based gravity feed-forward. Hand actuators use compliant
position control (`kp=30`, force range ±8) and padded contact margins. These
choices are generated by `prepare_assets.py` and appear in the public execution
trace rather than being hidden simulator interventions.

## Prepare and verify

```bash
cd Agentic-RAG-VLM
python scripts/prepare_assets.py
pytest -q tests/test_guanghua_assets.py
```

## Run

The canonical Challenge v4 result must not be overwritten. The service observed
on port 18070 after the archived run did not have the same recorded precision
profile, so do not reuse that process as provenance. For a new run, use a new
port, a truthful BF16/NF4 deployment label, a new server receipt, and a dated
output directory. Follow the admission-first commands in
[`KIRO_HANDOFF.md`](KIRO_HANDOFF.md#10-重跑-challenge-v4).

Challenge v4 contains 100 method rollouts in a paired frozen-seed design and
246 real multimodal calls.
It varies object geometry, fragile-object distance/bearing, instruction wording,
online event type, mass, friction, pose, and yaw. It includes no-change and
irrelevant-change controls, does not provide G2's answer offset, and records all
bounded schema/consistency repairs as cost. An independent Qwen routing turn
selects the required tools before planning; scene labels do not enter the runtime
dispatcher. The Full Agent executes three public-observation-only contracts for
RAG-to-skill routing, protected-relation motion constraints, and role-aware
change/memory verification. It passes G1–G4 at 10/10 each with 100% tool-selection
accuracy; the audit passes all 100 run directories, 260 public events and 60 tool
calls with zero endpoint errors. See `docs/AGENT_TOOLS_V4_RESULTS.md`. Challenge v3
is retained as the pre-explicit-router comparison.

The original easier v1 study remains reproducible as a mechanism diagnostic:

```bash
MUJOCO_GL=egl python scripts/run_qwen_vlm_pilot.py \
  --output output/qwen35_nf4_main --seed-split main \
  --endpoint http://127.0.0.1:18110/v1/chat/completions \
  --model /home/admin1/g2_multimodal_agent/models/Qwen3.5-4B \
  --deployment-label qwen3.5-4b-bnb-nf4-local
python scripts/summarize_qwen_vlm_pilot.py output/qwen35_nf4_main
python scripts/validate_agentic_artifacts.py output/qwen35_nf4_main
```

The v1 study made 140 real multimodal calls with 100% valid JSON and
no endpoint failures. Full passed 20/20 in G1, G2, and G3; each isolated
ablation passed 0/20. Every paired McNemar exact test gives `p=1.90735e-6`.
See `docs/QWEN_VLM_RESULTS.md`. Its 20/20 versus 0/20 result should not replace
the harder Challenge v4 in the main defense table. Both are still high-level
mechanism experiment, not physical pick success.

The deterministic adapter remains as a fast offline diagnostic:

```bash
MUJOCO_GL=egl python scripts/run_agentic_pilot.py --output output/agentic_pilot
python scripts/summarize_agentic_pilot.py output/agentic_pilot
python scripts/validate_agentic_artifacts.py output/agentic_pilot
```

Its output in `docs/PILOT_RESULTS.md` must not be substituted for the real-Qwen
result in a paper or presentation.

## Complete task-success video

The current primary presentation project is
[`video/vlabench_defense_reel/`](video/vlabench_defense_reel/). It combines
three selected VLABench task outcomes: chemistry, cook dishes, and
take out a healthy drink. The 84-second HyperFrames composition, source clips,
and visual snapshots exist, but the strict check must be rerun and the final
MP4 has **not** been rendered. Read its `AGENTS.md`, `BRIEF.md`, and
[`DEFENSE_GUIDE.md`](video/vlabench_defense_reel/DEFENSE_GUIDE.md); show the
preview to the user before rendering.

The chemistry source reaches the benchmark goal after the second forward pour,
but the executor stops after 14 of 20 planned stages instead of performing the
second tube's return/release sequence. It also records an `HCl_tag` unexpected
contact proxy. Both issues are P0 handoff items and must be resolved before the
video is described as a fully cleaned-up safety sequence.

The 67-second Guanghua reel and the following defense-suite material are legacy
artifacts, retained for historical mechanism evidence rather than current main
presentation use:

`output/defense_suite/agentic_rag_vlm_full_defense_reel_v9.mp4`

It is accompanied by Nominal, Fragile-aware, and Recovery controls under
`output/defense_suite/`. Generate all four raw rollouts, captioned videos, and
their manifest with one command, then audit every artifact:

```bash
MUJOCO_GL=egl python scripts/render_defense_suite.py --render-rollouts
python scripts/validate_defense_suite.py
```

All four are continuous 22.5 s, 1280×720, 30 fps rollouts. Their runtime
semantics differ as intended: Nominal/Fragile perform zero replans, while
Recovery/Showcase perform exactly one public-observation-triggered L3 replan.
All finish with both targets completed, the fragile object unmoved, and the
visual motion gate admitted. See `docs/DEFENSE_SUITE_RESULTS.md`.

The earlier single-video v2 artifact remains available for compatibility.

The defense-ready 1280×720, 30 fps video is:

`output/complete_task_demo_v2_seed101/guanghua_agentic_rag_vlm_complete_success.mp4`

It is one continuous 21.3 s MuJoCo rollout. The saved real Qwen3.5-4B NF4
receipts select POWER/PINCH skills, the scene graph protects the fragile
neighbor, and a declared blue-object displacement triggers memory-preserving L3
replanning. Version 2 uses a full tool-orientation constraint, hand-relative
grasp centres calibrated from the imported fingertip geometry, a natural-posture
IK prior, high-level hand pre-shaping, strict IK tolerances, and a reachable task
layout. Its motion gate reports at least 2.0 mm minimum
fingertip/table clearance, sub-millimetre terminal IK error, and no target or
fragile-object error under the declared grasp-skill proxy.

Reproduce the rollout and captioned answer video:

```bash
MUJOCO_GL=egl python scripts/render_complete_task_demo.py \
  --seed 101 --output output/complete_task_demo_v2_seed101
bash scripts/render_answer_video.sh
python scripts/validate_complete_task_demo.py output/complete_task_demo_v2_seed101
```

The original `complete_task_demo_seed101` directory is retained only as a
diagnostic baseline. It contains under-constrained arm postures and visible
table/object penetration and must not be used in a defense or reported as the
final execution result.

The transport phase is an explicitly logged kinematic grasp-skill proxy, not a
contact-dynamics success claim. This disclosure remains visible throughout the
answer video. See `docs/COMPLETE_TASK_DEMO.md` for the evidence contract.

Headless one-second physics check:

```bash
python scripts/run_guanghua_env.py --duration 1
```

Render the front camera:

```bash
MUJOCO_GL=egl python scripts/run_guanghua_env.py \
  --scene G2 --seed 13 --duration 1 \
  --camera agentview --render output/g2_agent.png
```

Render the declared G3 displacement for environment diagnostics:

```bash
MUJOCO_GL=egl python scripts/run_guanghua_env.py \
  --scene G3 --seed 17 --duration 1 --apply-perturbation \
  --camera frontview --render output/g3_displaced.png
```

`--apply-perturbation` applies the declared displacement halfway through this
diagnostic simulation. The formal Agentic experiment must instead trigger it
after the first visually verified placement, as frozen in the protocol.

Open the interactive viewer on a desktop session:

```bash
python scripts/run_guanghua_env.py --viewer
```

Run the first physical admission gate:

```bash
MUJOCO_GL=egl python scripts/run_g0_admission.py \
  --seed 7 --output output/g0_admission_seed7
```

The admission writes `public_trace.jsonl` separately from
`private_evaluator.json`. The controller target is estimated from rendered
RGB-D; MuJoCo object poses are used only by the private evaluator. It also
writes a 640×480, 20 fps `g0a_pregrasp_front.mp4` diagnostic video using the
system `ffmpeg` executable. The filename and role say `G0-A/pregrasp` so it
cannot be mistaken for a successful grasp video.

Current frozen result for seed 7: G0-A (RGB-D → safe physical pregrasp) passes
with sub-millimetre end-effector error and no non-support contact with the
fragile proxy. G0-B (contact closure → lift) is deliberately not marked as
passed: first fingertip contact can destabilize the imported free-object
dynamics. See `docs/GUANGHUA_CALIBRATION_STATUS.md`. Full task success rates
must not be reported until G0-B passes without a MuJoCo reset.

## Traditional vision + IK paired comparison

The current comparison includes both a fixed centroid+IK baseline and a stronger
reactive geometry+IK baseline. The latter is allowed a primitive-specific grasp
heuristic, a globally tuned fixed wrist yaw, and visual replanning on any scene
change. This avoids attributing gains to an intentionally weak comparator.

```bash
python scripts/run_traditional_vs_agentic.py
python scripts/render_paired_comparisons.py
```

The frozen E1–E4 results, the exact role of the translucent yellow protected
glass, negative controls, and claim boundary are documented in
`docs/TRADITIONAL_VS_AGENTIC_RESULTS.md`.

## Failure-correction hierarchy

The recovery experiment compares no recovery, one unchanged retry, and the
Agentic L1 parameter retry / L2 method switch / L3 full-replan hierarchy on
three recoverable failures plus no-failure and timeout controls:

```bash
python scripts/run_recovery_hierarchy_experiment.py
```

Frozen recoverable-failure results are T0 `0/3`, T1 `1/3`, and Agentic `3/3`.
The 36-second 1920×1080/30 fps evidence reel is:

`video/recovery_evidence_reel/renders/guanghua_recovery_evidence_reel_v1.mp4`

L1/L2 are explicitly disclosed as controlled verifier fault injection; the L3
segment uses the paired MuJoCo target-displacement video. This experiment tests
recovery-state selection, not contact-dynamics grasp success. See
`docs/RECOVERY_HIERARCHY_RESULTS.md`.

## Modeling boundary

This environment is intended for high-level Agentic RAG-VLM evaluation with an
IK / skill-based low-level executor. It does not contain or require a trained
VLA policy. MuJoCo private body poses and contacts should remain evaluator-only;
the agent should consume rendered RGB-D, proprioception, gripper state, and
structured experience memory.

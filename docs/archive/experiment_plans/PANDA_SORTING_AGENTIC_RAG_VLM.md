# Panda Sorting Agentic RAG-VLM Demonstration

## Purpose

Build a recordable, physics-based demonstration for the accepted Agentic RAG-VLM
work before coupling the same execution interface to a VLA. The task is a Panda
robot sorting two visually distinct household objects into their assigned bins in
Robosuite / MuJoCo.

This is a near-deployment physics simulation, not a real-robot result and not a
LIBERO leaderboard evaluation.

The complete three-family evaluation design is maintained in
`docs/plans/AGENTIC_RAG_VLM_PHYSICS_EVALUATION.md`. This document remains the
implementation status for the initial Panda sorting prototype.

## Closed-loop boundary

The control-side agent receives only:

- external RGB-D and wrist RGB observations;
- Panda proprioception and gripper state;
- monitor measurements derived from those observations; and
- task specification plus stored, structured experience cards.

Object poses, contact state, and success labels from MuJoCo are kept inside the
environment adapter for reset, fault injection, and final evaluation. They are
never included in a planner request or a skill command.

## Agentic RAG-VLM execution path

1. A low-frequency VLM planner identifies the next semantic subgoal and can
   critique an event snapshot. It returns a typed subgoal or recovery choice,
   never joint commands. At a recovery boundary, the router restricts it to the
   event-compatible registered skill or `safe_stop`.
2. HAA-RAG retrieves an experience card by object affordance and failure context.
3. The scene-graph reasoner converts estimated relative relations into safe
   approach and placement constraints.
4. A closed-loop Cartesian skill executor runs approach, grasp, lift, transport,
   place, and retract primitives while the high-frequency monitor watches
   progress and action response.
5. Reflection maps evidence to a bounded L1 parameter retry, L2 skill switch, or
   L3 replan. Successful outcomes are written back as experience cards.

## Demonstration scenarios

The final video will include ordinary sequential sorting and one recovery branch:

- object displacement after the plan is formed;
- failed lift / slip after gripper closure; or
- placement deviation requiring a corrected placement attempt.

Each fault is injected into MuJoCo physics. The resulting recovery is selected
from the same observation-only monitor and agent interfaces used in the nominal
run.

## Staged deliverables

1. **Foundation (complete 2026-07-29):** isolated task contracts,
   non-privileged observation adapter, RAG / scene graph / reflection components,
   a typed Agentic RAG-VLM harness, a real simulator smoke run, and one audited
   low-frequency Qwen VLM shadow call.
2. **Closed-loop skills (S1 complete):** RGB-D grounding plus a two-condition
   lift verifier (lateral displacement and RGB-D height increase) for Panda
   Cartesian primitives, with video recording and structured traces.
3. **Recovery video:** nominal and fault-injected rollouts with annotated event
   traces; publish only measured success and latency values.
4. **VLA coupling:** preserve the same task and harness contracts, then replace
   the analytic skill executor with Pi0.5 or another VLA adapter.

## Foundation evidence

- Runner: `scripts/run_panda_sorting_rag_vlm.py`.
- Physics smoke: `results/panda_sorting_rag_vlm_smoke_20260729/`.
- VLM-shadow smoke: `results/panda_sorting_rag_vlm_shadow_20260729/`.
  The local Qwen service returned a schema-accepted semantic decision in
  approximately 3.6 seconds; the result was logged but deliberately not executed.
- Regression: 106 unit tests passed after integrating the Panda sorting package.

## Closed-loop evidence

- Nominal two-object rollout:
  `results/panda_two_object_agentic_rag_vlm_ordered_20260729/`.
  RGB-D grounding and Panda Cartesian skills sorted Cereal then Bread in one
  MuJoCo episode; the evaluator reported both target objects in their correct
  bins. The raw physics video is `panda_two_object_sorting.mp4` (38.2 seconds,
  20 FPS).
- Recovery rollout:
  `results/panda_two_object_agentic_rag_vlm_bread_recovery_20260729/`.
  After the first object, the evaluator displaced Bread by a configured
  `(0.020, 0.010)` m offset. The control-side RGB-D estimate changed by
  0.01625 m, crossed the 0.015 m event threshold, and triggered the reflection
  policy's `L3_full_replan`. Both target objects subsequently passed evaluator
  placement checks.
- Hard negatives retained, not aggregated as successes:
  `results/panda_two_object_agentic_rag_vlm_recovery_20260729/` and
  `results/panda_two_object_agentic_rag_vlm_recovery_small_20260729/`.
  They demonstrate that Cereal can become unreachable after a displacement in
  crowded random layouts; future work must add obstacle-aware visual nodes and
  an L2 alternative approach skill before reporting recovery rates.

## Current mechanism evidence

- **Executed VLM-authorized L1 retry:**
  `results/panda_s1_l1_full_vlm_authorized_seed13_20260729/` records a real
  Qwen decision at `grasp_missed`. The router allowed only
  `retract_and_regrasp` or safe stop; the accepted decision was physically
  executed and Cereal reached its target bin.
- **Scene-graph physical constraint:**
  `results/panda_s2_full_seed13_20260729/` derives `near:milk`, an escape
  route, and high clearance from RGB-D before succeeding with no Cereal--Milk
  object contact. The matched no-graph route at
  `results/panda_s2_no_graph_seed13_20260729/` contacts Milk and fails
  placement.
- **Memory-grounded VLM-authorized L3 replan:**
  `results/panda_s3_compact_l3_real_vlm_seed13_20260729/` completes Cereal and
  Can. After a physical Can displacement, RGB-D detects the stale plan, Qwen
  authorizes `reobserve_scene`, and the harness rebuilds the subgoal before
  resumed execution.
- **Compound L1 + L3 integration smoke test:**
  `results/panda_s3_compound_l1_l3_real_vlm_seed13_20260730/` combines a
  declared Cereal grasp-height bias and a later Can displacement in one
  two-object rollout. Cereal's first visual lift check fails and its bounded
  L1 retry succeeds; Can then triggers L3 from a `2.09 cm` visual estimate
  change. At each safe boundary, the real local Qwen VLM is capability-limited
  to the event-compatible recovery skill and authorizes it. Both task objects
  pass evaluator-only destination checks. The VLM calls take `7.12 s` and
  `5.37 s`, so they are semantic, low-frequency decisions rather than a
  realtime control component. Because L1 is activated by a fixed internal
  grasp bias, this artifact validates wiring only and is excluded from the
  representative-task evidence.

# Agentic RAG-VLM Physics Evaluation Plan

## Scope and claim boundary

This plan evaluates the accepted Agentic RAG-VLM concepts in a **Robosuite /
MuJoCo Panda physics simulation**. It is a new, controlled demonstration suite,
not a replacement for the analytical 12-task evaluation reported in the IROS
paper and not a real-robot result.

The low-level executor is an observation-driven Cartesian skill library. VLA
integration is explicitly out of scope for this phase. The goal is to establish
whether affordance retrieval, scene constraints, execution memory, and bounded
reflection have distinct, observable effects in the same physical environment.

Control code may use only external RGB-D, wrist RGB, Panda proprioception,
gripper state, the declared task fixture layout, and agent memory. MuJoCo body
poses, contacts, rewards, and success flags are evaluator-only signals used for
initialization, disturbance injection, and scoring.

## Design principles

1. Test a mechanism with a task that can actually fail without it.
2. Use fixed, versioned scene specifications and a frozen seed manifest. A
   perception failure counts as failure; trials are never re-sampled based on a
   simulator success label.
3. Fault injection changes only MuJoCo physics. The agent must rediscover the
   outcome from its normal sensor channels.
4. Each recovery level has its own canonical scenario. Do not manufacture one
   episode containing every failure solely for a video.
5. The VLM is event-boundary semantic planning / critique, never a continuous
   action generator. A VLM comparison is deferred until its typed decision is
   actually allowed to select an enabled skill.

## Common task setup

- **Robot:** Franka Panda with parallel jaw gripper and OSC pose controller.
- **Objects:** Robosuite PickPlace Milk, Bread, Cereal, and Can meshes. Fixed
  color task labels are rendered only for RGB-D grounding; their mass, collision
  geometry, and joint dynamics are unchanged.
- **Destinations:** four physical PickPlace target compartments. A fixture map
  supplies the destination frame; it is equivalent to a calibrated bin layout
  on a real table and does not expose object pose.
- **Sensors:** external calibrated RGB-D camera, wrist RGB, robot proprioception,
  gripper aperture.
- **Semantic evidence:** HAA-RAG cards store affordance profile, context tag,
  approach height, gripper mode, result, and failure note. Execution memory
  stores completed subgoals and bounded recovery outcomes.
- **Safety evaluator:** contact between the gripper / target and an object marked
  `fragile_proxy`, plus placement outside the assigned compartment. These labels
  are logged after the rollout and are not planner inputs.

## Scenario family S1: Affordance-aware single-object manipulation

### Objective

Show that retrieval changes a measurable grasp parameter rather than merely
adding textual narration. This maps to the paper's Single-Grasp category and
isolates HAA-RAG plus Level-1 retry.

### Canonical tasks

| ID | Target | Manipulation distinction | Intended retrieval effect |
|---|---|---|---|
| S1-A | Bread | flat package / side-edge grasp | lower approach and longer closure hold |
| S1-B | Cereal | upright box / body-center grasp | center approach and larger clearance |
| S1-C | Can | cylindrical container / body grasp | conservative vertical approach |

Each target is evaluated in a clear source bin with no semantic neighbor.

### Controlled fault and recovery

For S1-B, inject a one-attempt grasp-height bias into the skill executor. The
object remains physically unmodified. A visual lift verifier observes that the
object did not leave the source region and triggers:

`L1_parameter_retry`: retrieve the appropriate card, lower / raise the approach
height according to the failure type, and retry once.

### Comparisons

- **Full:** HAA-RAG selected parameters plus L1 retry.
- **Generic prior:** one fixed grasp configuration plus L1 retry.
- **No reflection:** HAA-RAG parameters but only one attempt.

### Primary measurements

- object-to-destination success;
- first-attempt success and L1 recovery success;
- affordance-card selection accuracy;
- action count and wall-clock duration.

### Implementation status (2026-07-29)

**S1-B physics prototype is complete; the statistical pilot is not yet run.**
The implementation uses a fixed, declared Robosuite layout with Cereal at
`(0.10, -0.20)` and three non-target objects placed outside its approach lane.
The controller receives the layout only through RGB-D; the coordinates exist in
`scene_spec.json` as experiment initialization, not as planner input.

The full run at frozen seed 13 uses an `+0.11 m` one-attempt grasp-height bias.
The first attempt has visual lateral / height changes of `6.7 mm / 0.7 mm`, so
the RGB-D monitor rejects it. It then triggers L1, opens the gripper, applies
the retrieval-conditioned `-1 cm` height update to a `+1 cm` base parameter,
and retries from a fresh visual estimate. The retry has `5.7 cm` lateral motion
and `15.7 cm` height increase; the evaluator labels the Cereal destination as
successful. Artifacts are stored in
`results/panda_s1_l1_full_canonical_seed13_20260729/`.

The matched `no_reflection` run has the same first visual failure and stops
without placement; its evaluator label is unsuccessful:
`results/panda_s1_l1_no_reflection_seed13_20260729/`.

The current `generic` prior also succeeds on this sparse canonical layout.
Therefore the present S1 evidence supports the visual L1 failure detector and
bounded retry, but **does not yet support a stronger HAA-RAG-versus-generic
parameter superiority claim**. That comparison requires S1-A/C layouts whose
valid grasp-height intervals differ materially.

The full S1 run is also available with **executed semantic authorization**:
the local Qwen3.5-4B VLM sees the `grasp_missed` safe-boundary observation and
is restricted to `run_skill(retract_and_regrasp)` or `safe_stop`. It selected
the registered recovery skill with confidence `0.92`; the resulting physical
retry completed Cereal placement. Its `5.67 s` VLM latency is logged separately,
so this is semantic-agent evidence and explicitly not a realtime VLM claim:
`results/panda_s1_l1_full_vlm_authorized_seed13_20260729/`.

## Scenario family S2: Scene-constrained manipulation

### Objective

Create cases in which a direct top-down trajectory is physically unsafe or
blocked. This maps to the paper's Interactive category and isolates scene graph
relations and Level-2 skill switching.

### Canonical tasks

| ID | Physical layout | Expected scene relation | Required response |
|---|---|---|---|
| S2-A | Cereal adjacent to a `fragile_proxy` Milk object | `near:fragile_proxy` | increase clearance, use an offset approach |
| S2-B | Can partially hidden behind Bread in the approach corridor | `occluded_by` / `blocked_by` | use a staged lateral waypoint before descent |
| S2-C | Correct destination compartment initially occupied by a non-target object | `destination_occupied` | select a free staging location then complete placement |

The obstacle and proxy are physical MuJoCo bodies; collision and placement
constraints are evaluated from contact records after execution.

### Controlled fault and recovery

The no-scene-graph direct approach is expected to stall or violate the safety
clearance in S2-A/B. The full system receives only monitor / visual evidence and
uses:

`L2_skill_switch`: retract, move to a graph-derived lateral waypoint, re-approach
from above, then verify the target state.

### Comparisons

- **Full:** graph relations -> clearance / waypoint constraints -> L2 skill.
- **No scene graph:** direct approach with identical low-level controller.
- **No L2:** graph detects a conflict but only allows L1 parameter retry.

### Primary measurements

- end-to-end success;
- fragile-proxy contact rate and constraint-satisfaction rate;
- L2 recovery success conditional on a triggered event;
- additional travel distance / actions relative to direct approach.

### Implementation status (2026-07-29)

**S2-A physics prototype is complete; the statistical pilot is not yet run.**
Milk is rendered with a high-visibility red fragile-proxy label while retaining
its original MuJoCo mesh, mass, and collision geometry. Cereal and Milk are
placed in a fixed adjacent layout. The full condition derives `near:milk`, an
escape direction, and a `+3 cm` clearance adjustment from RGB-D scene estimates.
The no-graph ablation uses a fixed low-clearance shortcut toward the neighbor.

The frozen seed-13 full rollout places Cereal successfully with zero recorded
Cereal--Milk object contacts. The matched no-graph rollout records a
Cereal--Milk contact and fails target-bin evaluation. Contacts are
evaluator-only and never exposed to the agent. Artifacts are
`results/panda_s2_full_seed13_20260729/`,
`results/panda_s2_no_graph_seed13_20260729/`, and
`scripts/run_panda_s2_scene_graph.py`. This is physical
constraint-satisfaction evidence, not yet an L2 recovery-rate claim.

## Scenario family S3: Memory-grounded long-horizon table clearing

### Objective

Demonstrate task progress tracking and correction of a stale plan in a single
three-object rollout. This becomes the main presentation video.

### Physical task

Sort Cereal, Bread, and Can into three assigned target compartments. The scene
contains a fourth object as a non-target context / obstacle. The task uses a
deliberate execution ordering chosen from the scene graph and destination
availability rather than a hard-coded temporal action script.

### Memory state

- **Execution memory:** completed object IDs, assigned destination, attempt
  count, and verified outcome.
- **Retrieval memory:** successful grasp parameters and context tag for the next
  object.
- **Recent visual memory:** keyframes before grasp, after lift, and after place;
  used by the verifier to avoid acting on an occluded / stale target estimate.

### Controlled fault and recovery

After the first placement and after the second subgoal is planned, move the
second target by 1.5--2.5 cm in the MuJoCo world. The agent receives a new RGB-D
observation, compares it to the planned target estimate, and triggers:

`L3_full_replan`: re-observe, reconstruct the local graph, retrieve context-
matched memory, update the active target estimate, and resume from the safe
subgoal boundary.

The existing successful Bread-displacement rollout is the initial S3 prototype;
the final S3 scene must contain three targets and nontrivial obstacle context.

### Implementation status (2026-07-29)

**A compact S3 physics demonstration is complete; the three-target statistical
protocol remains future work.** The release rollout executes Cereal followed by
Can, with Bread and Milk retained as physical context objects. After Cereal is
visually verified and placed, the evaluator physically displaces Can by
`(2 cm, 1 cm)` after its plan has been created. The controller receives only a
new RGB-D frame, estimates a `2.29 cm` target change, triggers
`L3_full_replan`, rebuilds the Can subgoal, verifies its lift, and places it in
the assigned compartment.

The evaluator independently labels both active objects successful. The complete
release artifacts are:

- `results/panda_s3_compact_l3_release_seed13_20260729/scene_spec.json`;
- `results/panda_s3_compact_l3_release_seed13_20260729/trace.json`;
- `results/panda_s3_compact_l3_release_seed13_20260729/evaluator.json`; and
- `results/panda_s3_compact_l3_release_seed13_20260729/panda_s3_long_horizon_annotated.mp4`.

The video is `384 x 384` (higher than the earlier `256 x 256` rollouts) and
contains lettered segments A--G for scene initialization, retrieval/execution,
the physical disturbance, L3 replan, resumed execution, and completion. This
is a presentation-quality mechanism demonstration, **not** evidence for the
planned three-target success rate or a real-robot claim.

The latest release executes a real local Qwen3.5-4B semantic decision at the
L3 safe boundary. The router permits only `run_skill(reobserve_scene)` or
`safe_stop`; Qwen selects the registered reobservation skill, after which the
controller rebuilds the Can subgoal and finishes both active targets. The VLM
call is separately timed while physics is paused at the safe boundary, so this
is semantic-agent evidence rather than a realtime VLM claim:
`results/panda_s3_compact_l3_real_vlm_seed13_20260729/`.

### Compound recovery mechanism smoke test (2026-07-30)

`results/panda_s3_compound_l1_l3_real_vlm_seed13_20260730/` is a
**two-object compound-recovery integration smoke test**, not a representative
experiment or a new statistical condition: Cereal receives a declared
one-attempt `+11 cm` grasp-height bias and Can receives the declared `(2 cm,
1 cm)` post-plan displacement. The controller uses visual evidence only:

- Cereal's first lift check fails (`5.3 mm` lateral motion and negative height
  change), then L1 releases, re-observes, applies the one permitted parameter
  correction, and passes the second lift check.
- Can's refreshed RGB-D estimate differs from its pre-fault estimate by
  `2.09 cm`, triggering L3 full replan before its grasp.
- The local Qwen3.5-4B VLM is executed at both safe boundaries. The typed
  router authorizes only `retract_and_regrasp` for `grasp_missed` and only
  `reobserve_scene` for `target_displaced`; both returned authorized skills.
  Their logged latencies are `7.12 s` and `5.37 s`, respectively.
- Evaluator-only labels report Cereal and Can in their assigned bins. The
  environment-wide PickPlace flag remains false because Bread and Milk are
  intentionally non-target context objects.

The annotated video is
`panda_s3_long_horizon_annotated.mp4`. This compact scenario is suitable for a
mechanism regression because it exposes two different recovery levels in one
auditable physics rollout. The fixed internal grasp bias means it does not
establish natural recovery ability. It must not be reported as an L1/L3 success
rate, a three-object success rate, or a realtime VLM-control result.

### Three-target exploratory branch

The optional third-object branch remains exploratory. Bread is poorly grounded
by the current color-label detector under the default camera geometry and can
be physically coupled to Can in the original layout. Milk is visually grounded
with a red task tag but has an unstable pick-and-place primitive. These are
useful engineering findings, but neither branch is included in the formal S3
claim until its perception and object-specific skills pass the pilot gate.

### Comparisons

- **Full:** execution memory + visual verification + L3 replan.
- **No execution memory:** recompute task progress from the current frame only.
- **No L3:** retain stale target estimate after displacement.
- **No retrieval memory:** retain state memory but use generic action parameters.

### Primary measurements

- full-episode success and completed-subgoal count;
- stale-plan detection precision / recall over nominal and perturbed episodes;
- L3 recovery success conditional on injected displacement;
- repeated-action / wrong-object rate;
- VLM critic calls, memory retrievals, actions, and wall-clock latency.

## Evaluation protocol

### Pilot gate

Before measuring success rates, run 5 frozen seeds for each canonical task and
configuration. A task advances only after:

- perception and video are valid for all five trials;
- failure injection is visible from control observations;
- full and ablated methods share identical initial state / perturbation seeds;
- every metric can be regenerated from the saved trace.

### Final measured runs

| Stage | Runs | Purpose |
|---|---:|---|
| Mechanism pilot | 5 seeds / configuration | debug and reject invalid task designs |
| Main comparison | 20 seeds / configuration | stable effect-size estimate |
| Headline confirmation | 30 seeds for Full vs. key ablation | align with the original paper's trial convention |

The headline comparison is Full vs. the mechanism-specific ablation: generic
prior for S1, no-scene-graph for S2, and no-L3 for S3. Broader ablations remain
at 20 seeds unless the pilot variance requires more.

### Reporting

Report Wilson 95% confidence intervals for binary success, paired seed-level
deltas where states are matched, and raw trial traces. Never report a mean across
S1/S2/S3 without also showing the three family-level results.

## Required artifacts

For every frozen run:

- `scene_spec.json`: object labels, fixture geometry, seed, and injected fault;
- `trace.json`: sensor-only plan/retrieval/graph/recovery events and timings;
- `evaluator.json`: success and contact labels kept separate from control trace;
- front-camera rollout video; and
- first / failure / recovery / final keyframes.

For each scenario family, produce one concise annotated video. The S3 video must
show an ordinary subgoal completion, the physical displacement, visual detection
of the stale target, L3 replan, and final completion.

## Implementation order

1. **Freeze the scene-spec and trace schemas.** Generalize the existing two-
   object runner without changing its control-state boundary. **Completed for
   S1:** each run writes `scene_spec.json`, `trace.json`, `evaluator.json`, and
   a front-camera video.
2. **Implement S1.** Add a visual lift verifier, one bounded grasp-height fault,
   and L1 retry. Establish retrieval and reflection logging. **Completed for
   the S1-B prototype:** the verifier requires both lateral movement and RGB-D
   height increase, preventing tabletop sliding from being counted as a grasp.
3. **Implement S2.** Add context-object RGB-D nodes, contact evaluator, and a
   staged lateral-approach skill. Only then claim scene graph benefit.
4. **Implement S3.** Add three-object state memory, destination occupancy and a
   displacement recovery event. Reuse S1/S2 skills instead of inventing a new
   controller.
5. **Run pilots, lock seed manifests, then run final comparisons.**

## What this plan will and will not establish

It can establish physical-simulator evidence that the Agentic RAG-VLM system
uses retrieval, constraints, memory, and bounded reflection to improve defined
failure modes. It does not establish VLA generalization, a standard benchmark
ranking, sim-to-real transfer, or real-robot performance. Those claims require
the later VLA and hardware phases.

# CARVE Representative Experiment: Dynamic Kitting and Workspace Reconfiguration

## 1. Research question

Can a multimodal Agentic harness improve completion of a long-horizon robot
task under occlusion, occupied destinations, and unannounced scene changes,
while an optimized runtime bounds semantic-model calls and control latency?

The experiment must test this question directly. It must not create a recovery
success by injecting an error into the controller or by scheduling a known
retry branch.

## 2. Task definition

The robot receives one natural-language order and an RGB-D observation of a
tabletop kitting workspace. Two requested objects have assigned kit slots. A
third movable object may block an approach corridor or occupy a required slot;
a fourth object is a distractor. Completing the order can require the following
state-dependent subgoals:

1. identify the requested objects and their destinations;
2. move an occluder or slot occupant to a legal staging region;
3. manipulate the first requested object and verify placement;
4. retain completed-subgoal state while the scene changes;
5. re-observe and revise the remaining plan when required;
6. manipulate the second requested object and verify the final kit state.

This is not scored by the number of grasped objects. It is scored by whether the
system resolves task dependencies, avoids unsafe contacts, preserves task
progress, and reaches the requested final state.

## 3. Environment conditions

Five scene families share the same instruction and evaluator:

| Family | Physical condition | Required capability |
| --- | --- | --- |
| K0 | clear nominal workspace | ordinary planning and execution |
| K1 | requested Can constrains Cereal's approach | scene graph and prerequisite target order |
| K2 | Cereal slot occupied by requested Can | dependency-aware target ordering |
| K3 | remaining target moved after the first verified placement | stale-plan detection and replanning |
| K4 | K1/K2 plus randomized pose, yaw, friction, and mass | combined long-horizon robustness |

K3 is a valid environment perturbation because it is external to the
controller, policy-independent, randomized from a frozen seed manifest, and
observable through the normal sensors. No condition modifies grasp offsets,
action outputs, retry counters, or model logits to force a failure.

Natural failures are those produced by the frozen policy under pose,
appearance, friction, occlusion, or interaction variation. A recovery is
counted only when a monitor detects such a failure from public observations and
the task subsequently succeeds within the fixed recovery budget.

## 4. Framework roles

- **VLM Planner/Critic:** low-frequency semantic decomposition and event-level
  judgment. It chooses only from registered skills or safe stop.
- **Scene Graph:** represents `occluded_by`, `near`, `destination_occupied`, and
  `inside` relations and exposes prerequisite constraints.
- **Execution Memory:** records completed subgoals, failed attempts, staged
  obstacles, and verified outcomes; it stores no joint trajectories.
- **VLA or skill policy:** produces physical manipulation actions. The initial
  environment qualification may use analytic Cartesian skills, but an
  Agentic-VLA paper claim requires at least one real frozen VLA backend.
- **Monitor:** runs at high frequency on RGB-D, robot state, gripper state, and
  action response; it raises evidence rather than semantic plans.
- **Optimize Runtime:** asynchronously schedules VLM decisions at safe
  boundaries, suppresses unnecessary semantic calls, enforces budgets, and
  records end-to-end latency and GPU memory.

The high-level planner is not given a fixed object sequence. It must infer the
next valid subgoal from the instruction, scene graph, execution memory, and
registered capability set.

## 5. Comparisons

The minimum causal comparison set is:

| Method | Purpose |
| --- | --- |
| Frozen policy only | establish base task competence |
| Policy + fixed retry | separate repeated execution from Agentic reasoning |
| Full harness | test planner, graph, memory, monitor, and bounded recovery |
| Full harness without memory | test repeated/wrong-subgoal failures |
| Full harness without scene graph | test occlusion and occupied-slot reasoning |
| Full harness without runtime optimization | isolate efficiency contribution |

The fixed-retry method retries only after the same observation-side monitor
event and receives the same retry budget. It does not receive privileged
failure labels.

## 6. Measurements

Primary task metrics:

- complete-order success;
- verified subgoals completed;
- prerequisite-order violations;
- unsafe object-object contact rate;
- duplicate-action and wrong-object rates;
- conditional natural-recovery success.

Efficiency metrics:

- VLM and VLA calls per episode;
- monitor-to-intervention latency;
- VLM/VLA latency P50/P95 and end-to-end control-cycle P50/P95;
- deadline miss rate, GPU peak memory, and total episode wall time;
- success versus latency / call-budget Pareto curve.

Every reported outcome must be regenerated from `scene_spec.json`, public
observation traces, agent decisions, and a separate evaluator file.

## 7. Protocol and scale

1. Qualify each scene family with analytic skills on five frozen seeds. This
   validates physics, perception, evaluator separation, and video capture; it
   is not the final VLA result.
2. Freeze the task manifest before method comparison. A perception or planning
   failure counts as an episode failure; do not resample based on outcome.
3. Run 10 frozen seeds per family and method (`50` episodes per method) for the
   main controlled study. Increase only if confidence intervals remain too
   wide.
4. Report Wilson 95% intervals for success metrics and paired bootstrap
   intervals for latency/call differences because all methods share seeds.
5. Publish three qualitative videos selected by scene ID before observing
   method outcomes: nominal, dependency resolution, and dynamic replanning.

## 8. Acceptance gates

The representative experiment is ready only when:

- the full harness chooses at least two different valid subgoal orders across
  scene families from observations rather than a scripted episode sequence;
- K1/K2 require a prerequisite action and the base policy cannot pass by direct
  target grasp alone;
- K3 displacement is detectable from RGB-D without MuJoCo object poses;
- natural failure/recovery events are traceable and no controller-internal
  failure injection is enabled;
- the same task interface runs with an analytic skill backend and a frozen VLA
  backend;
- runtime metrics include actual model calls, not simulated sleep or estimated
  FLOPs.

## 9. Status

As of 2026-07-30, K0 and K1 run end to end in MuJoCo with RGB-D state
estimation, final-state planning, execution memory, visual lift and placement
verification, a real capability-gated Qwen VLM, private evaluator labels, and
recorded videos. The controller receives no MuJoCo object poses.

K1 is the first representative dependency result: the observed scene causes
the planner to execute `Can -> Cereal -> verify`, while K0 executes
`Cereal -> Can -> verify`. No grasp offset, action, retry event, or model output
is modified to create this order change.

The paired runtime experiment also passes in both K0 and K1. Calling the VLM at
every subgoal boundary uses three calls per episode. Event-triggered scheduling
retains the necessary initial semantic decision and uses one call while
preserving the same selected subgoals, final-state success, and zero-contact
evaluator outcome. Exact latency deltas are generated from the run traces by
`scripts/summarize_panda_kitting_runtime.py`.

K3-v2 also passes with event-triggered VLM scheduling. After Can placement is
visually verified, the environment relocates the unfinished Cereal without
providing event metadata to the controller. RGB-D observes a 11.63 cm
displacement, completed-subgoal memory is retained, and the agent acts from the
new object estimate. The final evaluator reports both targets in their slots
and zero object-object or protected-object contacts. The first K3 qualification
attempt remains recorded as a calibration failure because its relocation
placed Cereal in a poorly reachable area.

K2 is currently a capability gate rather than a positive result. The
dependency planner correctly identifies the requested Can occupying Cereal's
slot, but the analytic Cartesian primitive cannot extract an object from the
walled slot. Bounded natural L1 recovery also fails. This scene should be used
to qualify a slot-extraction skill or a frozen VLA backend; it must not be
reported as a successful harness episode.

The earlier Panda S1/S2/S3 fixed-bias artifacts remain mechanism regression
tests only. They are excluded from the main evidence table because the recovery
event is deliberately injected. The next milestones are K3 external
scene-change repeatability across held-out seeds and replacement of the
analytic physical skill backend with a frozen VLA adapter.

The Pi0.5 preparation pipeline now records public-observation transitions and
converts real external/wrist views, robot state, and 7D OSC actions to LeRobot.
A two-episode randomized-XY, fixed-yaw pilot passes for K0 and K1. This only
qualifies the data interface. The next VLA milestone is a 5+5 expert acceptance
gate, followed by a 30+30 collection and Pi0.5 LoRA fine-tuning if acceptance
is at least 80%.

# Panda Dynamic Kitting Pipeline Status

## Scope

This status separates three claims that must not be conflated:

1. the Agentic harness and Optimize Runtime work in a real MuJoCo physics loop;
2. the analytic expert can generate Pi0.5-compatible demonstrations;
3. a frozen Pi0.5 policy completes the same task.

Items 1 and 2 are now qualified. Item 3 is not yet complete.

## Agentic harness qualification

The representative task uses a Panda robot in Robosuite PickPlace / MuJoCo.
The controller receives external/front RGB-D, wrist RGB, Panda state, fixture
geometry, and bounded execution memory. MuJoCo object poses, contacts, rewards,
and success labels remain evaluator-only.

Qualified evidence:

- K0 nominal: VLM chooses from two valid initial subgoals; success and zero
  object contacts.
- K1 dependency: observed relations change the plan from K0 ordering to
  `Can -> Cereal -> verify`; success and zero object contacts.
- K3 dynamic change: after the first verified placement, the environment moves
  the remaining target without sending event metadata. RGB-D observes a
  11.63 cm displacement, the completed-subgoal memory is retained, and the
  final task succeeds with zero contacts.
- K2 occupied slot: dependency detection works, but the analytic primitive
  cannot extract an object from the walled slot. This remains a physical-policy
  capability gate.

Evidence index:
`results/panda_kitting_representative_20260730/README.md`.

## Optimize Runtime qualification

The paired K0 and K1 runs compare a real Qwen VLM call at every subgoal boundary
against event-triggered semantic scheduling.

| Scene | Calls: baseline -> optimized | Wall time: baseline -> optimized | Result |
| --- | ---: | ---: | --- |
| K0 | 3 -> 1 | 60.321 s -> 49.310 s | same subgoals, success, zero contacts |
| K1 | 3 -> 1 | 60.531 s -> 48.119 s | same subgoals, success, zero contacts |

This is evidence for system-level inference optimization: deterministic
transitions are discharged by RGB-D state and execution memory, while the VLM
is retained at ambiguous or dependency-bearing boundaries.

It is not yet evidence for Pi0.5 model compression or quantization.

## Pi0.5 data pipeline

The environment now exposes an optional transition sink. It records only
public observations and applied controls:

- external RGB: `256 x 256 x 3`;
- real wrist RGB: `256 x 256 x 3`;
- Robosuite proprio: 32 dimensions;
- normalized OSC_POSE action: 7 dimensions.

The converter maps proprio to the 8-dimensional OpenPI state
`[eef_position, eef_axis_angle, gripper_qpos]` and preserves both real camera
views.

Final pilot:

- source HDF5:
  `results/panda_kitting_pi05_data_pilot_20260730/kitting_pi05_pilot.hdf5`;
- LeRobot dataset:
  `results/panda_kitting_pi05_data_pilot_20260730/lerobot`;
- two accepted episodes, 2396 frames;
- K0 seed 201: 1123 frames;
- K1 seed 10201: 1273 frames with required `Can -> Cereal -> verify` order.

The pilot uses randomized reachable XY positions and fixed zero yaw. This is
stage A of a data curriculum, not a claim of yaw robustness. A prior randomized
yaw pilot produced K0 1/5 and K1 0/5 expert success, revealing that the current
analytic grasp primitive is not suitable as a random-yaw demonstration oracle.

## Gates before Pi0.5 fine-tuning

1. Collect at least 30 accepted K0 and 30 accepted K1 stage-A episodes.
2. Freeze 10 held-out seeds per family before training.
3. Check action/state distributions and compute OpenPI norm statistics.
4. Fine-tune Pi0.5 with LoRA on the 4-A100 machine; the two-episode pilot is
   only a loader and loss smoke test.
5. Evaluate frozen Pi0.5, Pi0.5 + Agentic harness, and Pi0.5 + harness +
   Optimize Runtime on the same held-out scenes.
6. Add visual yaw estimation or a stronger expert before introducing
   random-yaw stage-B data.

The existing Robosuite Stack Pi0.5 checkpoint is not a substitute. Its recorded
closed-loop baseline is 0/5 on Stack and it was not trained for dynamic
kitting.

## Recommended next execution

The next resource-efficient action is a five-episode-per-family collection
qualification. Continue to 30 per family only if both families achieve at least
80% expert acceptance and K1 semantic-order checks pass. This avoids spending
GPU time on a dataset whose expert labels are physically or semantically
inconsistent.

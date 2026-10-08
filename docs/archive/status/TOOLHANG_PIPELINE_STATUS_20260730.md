# ToolHang Pipeline Status - 2026-07-30

## Current result

Robosuite 1.4.1 ToolHang launches locally with the Panda robot, OSC pose
control, three RGB-D views, and 20 Hz stepping. The initial evaluator state is
`frame_unassembled`.

The new `agentic_vla.tool_hang.ToolHangEnvironment` exposes only camera and
robot observations. Robosuite object poses, object-state, reward, and built-in
stage checks remain behind the evaluator API. A live MuJoCo smoke test confirms
the 7D action interface, version-stable 9D end-effector/gripper state, camera
calibration, stepping, and private phase evaluation.

## Why this replaces simple grasping as the main demo

ToolHang requires two different object interactions and a strict physical
dependency: the hook frame must be grasped, reoriented, and inserted before the
wrench can be aligned and hung. Grasp recovery and alignment recovery are
meaningfully different. This makes it suitable for testing semantic stage
planning, monitor events, execution memory, bounded recovery, and runtime
scheduling in one task.

## External demonstration audit

The local LeRobot pilot contains:

- 23 episodes;
- 10,870 frames;
- 20 Hz videos;
- 7D OSC actions;
- agent-view and wrist-view 256x256 video.

The audit blocks direct final-policy training because the 43D state has no
semantic names, a separate 44D environment-state field is present, and the
dataset page lacks complete source and citation metadata. It is retained only
for task qualification until provenance is resolved.

An official Robomimic PH HDF5 has also been downloaded from
`robomimic/robomimic_datasets`. It contains 200 successful ToolHang
demonstrations and 95,962 frames. A local Robosuite 1.5.1 state replay of
`demo_0` reaches frame assembly at source frame 392 and complete tool hanging
at source frame 677; the final built-in evaluator reports success. This is a
dataset replay, not an online policy result.

The recorded 7D actions were then executed online from the same official
initial state. This control rollout assembles the frame at action step 391 but
fails to hang the tool; the wrench falls to the table and the final phase
remains `frame_assembled`. No action corruption or scheduled failure was used.
This natural long-horizon drift is the first frozen recovery case for the
ToolHang harness. The action stream contains five observable gripper-delimited
segments: frame approach, frame grasp/reorientation/insertion, frame release
and tool approach, tool grasp/alignment/hanging, and release/verification.

## Next implementation gate

Implement and qualify the physical action-chunk interface for:

1. frame grasp and reorientation;
2. frame insertion;
3. wrench grasp;
4. wrench alignment and hanging;
5. observation-driven grasp and alignment recovery.

Only after a full evaluator-success trajectory exists will VLM planning,
monitor events, memory, and Optimize Runtime be compared. This avoids repeating
experiments against a changing physical backend.

The immediate target is not an unconditional replay retry. It is an
observation-conditioned tool recovery policy that re-observes the fallen or
misaligned wrench, chooses grasp recovery or alignment recovery, and reaches
the final evaluator state without resetting the completed frame stage.

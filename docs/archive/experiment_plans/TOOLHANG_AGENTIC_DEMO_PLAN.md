# CARVE ToolHang Representative Demo Plan

## 1. Why ToolHang

The Panda ToolHang task is the next representative physical experiment. It is
not scored as a sequence of independent grasps. A successful episode must:

1. localize and grasp a loose hook frame;
2. reorient the frame and insert it onto a narrow stand;
3. verify that the frame is physically assembled;
4. preserve that completed stage in execution memory;
5. localize and grasp a wrench by a usable region;
6. align the wrench hole with the assembled hook;
7. release and verify that the wrench remains hanging without gripper support.

The task therefore exposes stage dependence, contact-rich alignment,
orientation sensitivity, and two distinct recovery classes. These properties
are more suitable for Agentic Harness evaluation than the previous tabletop
kitting qualification task.

## 2. Information boundary

Control modules receive only:

- agent-view RGB-D;
- front-view RGB-D;
- wrist RGB-D;
- version-stable 9D state: end-effector position, quaternion, and gripper;
- gripper position;
- end-effector position and quaternion;
- the natural-language instruction and bounded execution memory.

Object poses, the 44D Robosuite object-state vector, reward, simulator state,
and the built-in success checks are private evaluator data. The VLM planner,
monitor, VLA, and recovery policy may not consume them.

## 3. Framework decomposition

### Low-frequency semantic layer

The event-triggered VLM Planner/Critic selects a registered capability, checks
whether the current stage is semantically complete, and revises the plan after
evidence-backed anomalies. It does not output joint commands.

### High-frequency execution layer

The monitor consumes public camera and robot signals. It detects lack of
object motion after closure, loss of the grasped object, stalled insertion,
excessive command-response mismatch, and post-release instability. It raises
events but does not choose semantic goals.

### Physical policy layer

The initial qualification backend contains heterogeneous policies:

- frame grasp and reorientation;
- frame insertion;
- wrench grasp;
- hole-to-hook alignment and hanging;
- retract/reobserve;
- grasp recovery;
- alignment recovery.

The backend must implement the same action-chunk interface used by a frozen
VLA. Analytic or demonstration-based policies qualify the physics and recovery
protocol but are not reported as final VLA evidence.

## 4. Frozen scene families

| ID | Condition | Capability under test |
| --- | --- | --- |
| T0 | nominal sampled reset | complete multi-stage execution |
| T1 | frame pose and yaw variation | reorientation and insertion robustness |
| T2 | wrench pose and yaw variation | grasp choice and hole alignment |
| T3 | natural grasp or insertion failure | event detection and bounded recovery |
| T4 | safe external scene change at a stage boundary | re-observation and memory |

T3 contains no forced action corruption. It counts only failures that arise
from frozen pose, dynamics, perception, or policy variation. T4 perturbations
are environment-side, declared in a frozen seed manifest, and visible through
normal sensors.

## 5. Comparisons

1. physical policy only;
2. policy plus unconditional fixed retry;
3. full Agentic Harness;
4. full harness without execution memory;
5. full harness without VLM semantic review;
6. full harness without Optimize Runtime scheduling.

All methods share scene seeds and the same action/recovery budgets.

## 6. Metrics

Task metrics:

- full ToolHang success;
- frame-assembly success;
- conditional tool-hang success after frame assembly;
- stage-order violations;
- natural failure-detection precision and recall;
- conditional recovery success;
- duplicate and unnecessary action rate.

Runtime metrics:

- VLM and VLA calls per episode;
- monitor-to-intervention latency;
- model and control-loop P50/P95 latency;
- deadline miss rate;
- GPU peak memory;
- episode wall time and success-efficiency Pareto curves.

## 7. Data qualification

The downloaded LeRobot pilot contains 23 ToolHang episodes, 10,870 frames, 7D
OSC actions, and two 256x256 camera streams at 20 Hz. It is useful for task and
video qualification, but its converted 43D state lacks semantic field names,
contains a separate 44D environment-state field, and has incomplete provenance
and citation metadata.

Until those fields are traced to an official Robomimic source, this copy is
classified as `reference_video_and_interface_only`. It must not silently train
the final policy or support a no-privileged-state claim.

## 8. Acceptance gates

- a real Robosuite episode reaches both built-in evaluator stages;
- no policy input contains object poses, object-state, reward, or success;
- the full harness can distinguish grasp recovery from alignment recovery;
- recovery is triggered by public evidence, not a scheduled failure branch;
- the same environment and skill contract accepts a frozen VLA adapter;
- system videos contain planner events, monitor evidence, recovery decisions,
  stage memory, and private evaluator results in separate overlays or logs.

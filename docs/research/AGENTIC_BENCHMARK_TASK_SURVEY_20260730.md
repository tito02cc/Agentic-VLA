# Agentic Benchmark Task Survey

Date: 2026-07-30

## 1. Scope

This survey selects tasks that can test the mechanisms inherited from
Agentic RAG-VLM and extended by the CARVE Agentic Harness. It does not select
benchmarks for leaderboard coverage.

The target mechanisms are:

1. HAA-RAG: retrieving object affordance, material, fragility, grasp-region,
   and prior-experience knowledge.
2. Scene-graph reasoning: reasoning about support, occlusion, containment,
   proximity, and articulation constraints.
3. Episodic memory: retaining target identity, completed stages, and prior
   failures over a long-horizon task.
4. Self-reflective recovery: diagnosing a naturally occurring failure and
   choosing parameter adjustment, skill switching, or replanning.
5. Agentic process control: deciding when to continue, verify, retry, replan,
   or stop without exposing simulator state to the agent.

The low-level VLA remains frozen unless a benchmark has no compatible policy.
Simulator reward and object poses are evaluator-only signals.

## 2. Decision Summary

### Immediate main experiment: LIBERO-PRO task subset

Use three LIBERO-10 task families:

- T3: place the black bowl in the bottom drawer and close it.
- T9: place the mug in the microwave and close it.
- T8: place both moka pots on the stove.

Run only mechanism-relevant LIBERO-PRO variations:

- clean/original;
- position swap;
- object or task variation;
- selected combined variation after the single-factor gate passes.

This is the fastest credible next step because the local LIBERO-PRO assets and
initial states are already available, its perturbations are controlled, and it
does not require retraining pi0.5 before the Agentic comparison.

### Preferred recent external validation: RoboDojo Cover Blocks

Use `Cover Blocks` as the first RoboDojo task:

- it requires remembering a hidden color sequence;
- it has multiple ordered stages;
- it provides graded progress rather than only binary success;
- official RoboDojo/XPolicyLab resources include a pi0.5 adapter and released
  RoboDojo pi0.5 checkpoints.

The second task, after the first passes, is `Imitate Sorting Sequence`. It
isolates long-horizon visual memory and ordered execution.

RoboDojo is not the immediate experiment because its official environment is
based on Isaac Sim. The present machine has adequate GPU capacity for policy
or simulation in isolation, but its Ubuntu version, available disk, and
same-GPU simulator-policy contention make setup a separate engineering gate.

### Deferred options

- LIBERO-Plus: use later for selected robustness transfer, not all 10,030
  variants. It is broad, but less diagnostic of memory and recovery, and the
  local asset archive is currently incomplete.
- RoboTwin 2.0: use `Hanging Mug` only if task-specific training resources are
  accepted. The official protocol fine-tunes each policy on 50 demonstrations
  per task, so this is not a low-cost zero-shot test.
- RMBench: retain as a memory-specialized fallback. Its task design is highly
  relevant, but using the released Mem-0 policy would evaluate another
  architecture, while integrating our own pi0.5 path may require training.
- ToolHang: retain as a qualitative mechanism demo. It is useful for videos
  and recovery traces but is not a substitute for a public benchmark result.

## 3. Benchmark Comparison

| Benchmark | Relevant capability | Data / released policy | Fine-tuning for our first test | Local readiness | Recommended role |
|---|---|---|---|---|---|
| LIBERO-PRO | Controlled object, position, language, task, and environment shifts | Official BDDL and initial states; pi0.5 reported by authors | No, for frozen-policy comparison | High | Immediate main experiment |
| LIBERO-Plus | Large-scale robustness under seven perturbation families | Official data and selected model resources | Not necessarily | Medium-low; local assets incomplete | Secondary robustness transfer |
| RoboDojo | Generalization, memory, precision, long horizon, and open tasks | XPolicyLab adapters and released RoboDojo pi0.5 checkpoints | Potentially no for checkpoint-supported tasks | Low; Isaac setup gate | Preferred recent external validation |
| RMBench | Explicit memory-dependent manipulation | Official data and Mem-0 checkpoints | Likely for a CARVE-owned pi0.5 comparison | Low-medium | Memory fallback or later study |
| RoboTwin 2.0 | Diverse bimanual manipulation | More than 100k trajectories; official pi0.5 integration | Yes in the standard protocol | Low | Later transfer, not first-line |

## 4. Task-to-Mechanism Matrix

Scores indicate how directly a task can expose the mechanism: high, medium, or
low. Cost includes setup, training, and evaluation risk.

| Task | HAA-RAG | Scene graph | Memory / ordering | Reflective recovery | Cost | Decision |
|---|---|---|---|---|---|---|
| LIBERO-PRO T3 drawer | Medium | High | Medium | High | Low | Main |
| LIBERO-PRO T9 microwave | High | High | High | High | Low | Main |
| LIBERO-PRO T8 two moka pots | Medium | Medium | High | Medium | Low | Main |
| RoboDojo Cover Blocks | Low | Medium | Very high | Medium | High | External main |
| RoboDojo Imitate Sorting Sequence | Low | Medium | Very high | Medium | High | External secondary |
| RoboDojo Store Tools in Toolbox | Very high | High | High | High | Very high; evaluation-only | Stretch |
| RoboDojo Sweep Blocks | High | High | High | Very high | Very high; binary and difficult | Stretch |
| RMBench Battery Try | High | Medium | High | High | High | Alternative |
| RoboTwin Hanging Mug | High | High | High | High | Very high; bimanual training | Deferred |
| ToolHang | High | High | High | High | Medium | Qualitative demo |

### Why these LIBERO-PRO tasks

T3 and T9 contain an articulated container, placement verification, and a
required closure stage. They can distinguish a planner that merely repeats an
action from one that identifies an incomplete stage or a failed interaction.

T8 requires tracking two similar objects and completed subgoals. It can expose
duplicate actions, target confusion, and stale memory. T6 is retained only as
a smoke task because two independent placements do not sufficiently stress the
Agentic mechanisms.

### Why Cover Blocks comes before the more attractive open tasks

`Store Tools in Toolbox` is conceptually closer to HAA-RAG, but it is an
evaluation-only task and therefore risks conflating policy skill coverage with
Agentic reasoning. `Cover Blocks` has a released task pipeline, explicit
memory demand, ordered stages, and graded scoring. It provides a cleaner first
test of whether the planner and memory improve execution.

## 5. Experimental Protocol

### 5.1 Agent inputs

The evaluated system may use:

- RGB or RGB-D observations;
- robot proprioception;
- language instruction;
- action and decision history;
- retrieved long-term experience records.

The agent may not use simulator object poses, ground-truth stage labels, or
reward. Those signals are reserved for scoring and post-hoc failure labeling.

### 5.2 Compared systems

Run the smallest comparison that can identify Agentic value:

1. Frozen pi0.5 VLA.
2. Frozen pi0.5 with fixed or unconditional retry.
3. Full Agentic system: VLM planner/critic, HAA-RAG, scene graph, episodic
   memory, monitor, and reflective recovery.

After the full system shows a positive signal, run targeted ablations only on
the affected task-condition pairs:

- without HAA-RAG;
- without scene-graph reasoning;
- without episodic memory;
- without reflective recovery.

This order prevents spending GPU time on ablations before the main effect is
established.

### 5.3 Evaluation sequence

1. Qualification: three fixed seeds for every selected task-condition pair.
2. Admission gate: continue only if the full system improves stage completion
   or recovers at least one natural baseline failure without increasing unsafe
   or redundant actions.
3. Main evaluation: freeze ten held-out seeds per admitted pair.
4. Expansion: add a combined perturbation only after the corresponding
   single-factor conditions pass.
5. External validation: move to RoboDojo only after the LIBERO-PRO mechanism
   trace is correct and reproducible.

Do not choose seeds because a recovery happened. Task, perturbation, seed set,
retry budget, and stop policy must be fixed before the main run.

### 5.4 Required metrics

Effectiveness:

- task success and stage completion;
- conditional recovery rate on baseline-failure episodes;
- task-order violations;
- duplicate or redundant actions;
- false intervention rate;
- unsafe-stop rate.

Efficiency:

- VLM and VLA calls per episode and per success;
- recovery attempts per success;
- end-to-end latency p50 and p95;
- deadline-miss rate and action age;
- peak GPU memory;
- successful episodes per wall-clock hour.

The primary Agentic claim must be supported by recovery or stage-progress
metrics, not by additional retries alone. The Optimize Runtime claim must be
reported as a success-efficiency frontier rather than latency in isolation.

## 6. Recommended Execution Order

### P0: Freeze the Agentic observation and decision boundary

Verify that monitor, planner/critic, memory, and evaluator use separate
interfaces. Record every trigger, diagnosis, decision, retrieval, and VLA call
in the episode trace.

### P1: LIBERO-PRO qualification

Run:

- T3: clean, position, and task/object variation;
- T9: clean, position, and task/object variation;
- T8: clean and position variation.

Use the three-system comparison above. Stop conditions:

- stop a task-condition pair if the underlying VLA never reaches the relevant
  stage in any qualification seed;
- stop if the Agentic system can only improve by reading privileged state;
- stop if added calls raise latency without improving recovery or progress.

### P2: LIBERO-PRO held-out study

Promote only positive qualification pairs to ten held-out seeds. Add targeted
ablations on the minimum set needed to attribute the gain.

### P3: RoboDojo environment gate

Before downloading a full dataset:

1. verify native or containerized Isaac Sim launch;
2. download only the required assets and one pi0.5 parameter checkpoint;
3. start simulator and policy in isolated processes;
4. verify one unmodified Cover Blocks rollout;
5. integrate the Agentic decision interface only after the baseline works.

If this gate exceeds the storage or environment budget, use RMBench Cover
Blocks as the fallback and keep the same memory-oriented protocol.

### P4: Optional transfer

Choose one, not all:

- LIBERO-Plus selected perturbations for broad robustness;
- RoboDojo Imitate Sorting Sequence for memory transfer;
- RoboTwin Hanging Mug for bimanual contact-rich recovery.

## 7. Claims This Route Can Support

If the experiments pass, the defensible claims are:

1. A frozen VLA can be augmented by an Agentic process-control layer that
   improves selected long-horizon and perturbed manipulation tasks.
2. The gain comes from selective semantic planning, memory, and diagnosed
   recovery rather than unconditional retries.
3. The runtime reduces unnecessary model calls or deadline violations while
   preserving the recovery benefit.
4. The architecture transfers across at least one public task family without
   changing the low-level VLA interface.

This route does not support a claim that CARVE establishes a new overall
benchmark state of the art. That claim is neither required nor aligned with
the mechanism-first objective.

## 8. Official Sources

- LIBERO-Plus repository:
  https://github.com/sylvestf/LIBERO-plus
- LIBERO-Plus CVPR 2026 paper:
  https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html
- LIBERO-PRO repository:
  https://github.com/Zxy-MLlab/LIBERO-PRO
- RoboTwin 2.0 repository:
  https://github.com/robotwin-Platform/RoboTwin
- RoboTwin 2.0 tasks:
  https://robotwin-platform.github.io/doc/tasks/
- RoboTwin pi0.5 integration:
  https://robotwin-platform.github.io/doc/usage/Pi05.html
- RMBench repository:
  https://github.com/RoboTwin-Platform/RMBench
- RoboDojo documentation:
  https://robodojo-benchmark.com/doc/
- RoboDojo Cover Blocks:
  https://robodojo-benchmark.com/doc/sim-tasks/cover-blocks/
- RoboDojo Imitate Sorting Sequence:
  https://robodojo-benchmark.com/doc/sim-tasks/imitate-sorting-sequence/
- XPolicyLab repository:
  https://github.com/XPolicyLab/XPolicyLab

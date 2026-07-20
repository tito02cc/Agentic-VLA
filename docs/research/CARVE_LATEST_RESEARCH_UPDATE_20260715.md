# CARVE-VLA Latest Research Update

Date: 2026-07-15

## Executive Decision

The project should continue, but its contribution must be narrowed.

CARVE should no longer be presented as the first general Agentic VLA harness.
Recent work has already occupied generic hierarchical VLA orchestration,
inference-time visual verification, adaptive action chunking, speculative VLA
inference, VLA-specific quantization, and physical-AI harness terminology.

The defensible central question is now:

> Can a frozen VLA recover from execution deviations while meeting a robot
> control deadline, by using one risk signal to jointly select an execution mode
> and allocate inference compute?

Recommended paper identity:

**CARVE: Compute-Adaptive Recovery and Verification for Frozen
Vision-Language-Action Policies**

CARVE is a post-hoc runtime. It does not train a foundation model, synthesize
online rewards, or require a world-model branch. Its potential contribution is
the coupling of execution supervision and deadline-aware computation.

## What Harness Means in 2026

Four different meanings must be separated.

1. **Evaluation harness**: [vla-eval](https://arxiv.org/abs/2603.13966)
   isolates benchmark environments from model servers. It is experiment
   infrastructure, not an Agentic controller.
2. **Physical-AI middleware harness**:
   [Harness Engineering for Physical AI](https://arxiv.org/abs/2606.09416)
   argues that ROS 2 middleware should enforce projection, isolation, and
   transfer constraints for learned models. It is primarily a systems position
   and profile proposal.
3. **Agentic robot runtime**:
   [OpenRAL](https://discourse.openrobotics.org/t/openral-the-agentic-harness-for-physical-ai-ros-2-native/56352)
   is a recent ROS 2-native implementation announcement for typed, traceable,
   safety-oriented integration of policies, reasoning, skills, and sensors.
4. **Portable embodied inference runtime**:
   [Embodied.cpp](https://arxiv.org/abs/2607.02501) defines input adapters,
   sequence builders, backbone execution, head plugins, deployment adapters,
   and multi-rate execution for VLA and WAM deployment. It evaluates both
   pi0.5 and HY-VLA.

Therefore, "VLA Harness" alone is no longer a sufficient research novelty.
CARVE may still use harness engineering, but the paper must name the algorithmic
behavior it contributes.

## Closest Agentic and Hierarchical Work

| Work | Date | Main mechanism | Relation to CARVE |
|---|---:|---|---|
| [Agentic-VLA](https://arxiv.org/abs/2605.22896) | 2026-05 | Online adaptation through reward synthesis, language-guided exploration, and memory of policy weights | Occupies the name and Agentic training story, but not frozen inference-time recovery |
| [What Matters in Orchestrating Robot Policies](https://arxiv.org/abs/2606.10267) | 2026-06 | Systematic study of VLM planner, VLA controller, switching interface, observation, and memory in hierarchical VLA agents | Directly pressures generic planner-controller orchestration claims |
| [HiVLA](https://arxiv.org/abs/2604.14125) | 2026-04 | VLM task decomposition and visual grounding connected to a flow-matching action expert | Pressures claims based only on a VLM plus low-level VLA hierarchy |
| [Steerable Policies](https://arxiv.org/abs/2602.13193) | 2026-02 | Trains VLAs to accept commands at several abstraction levels, including subtasks, motions, and pixel coordinates | Shows that a planner-to-policy interface must be physically expressive, not prompt rewriting alone |
| [Maestro](https://arxiv.org/abs/2511.00917) | 2025-11 | VLM coding agent composes perception, planning, and control modules as a programmatic policy | Occupies broad tool and skill orchestration for zero-shot robots |
| [VERITAS](https://arxiv.org/abs/2606.18247) | 2026-06 | Gradient-free visual verifier selects or steers generated policy actions and creates data for self-improvement | Directly pressures a generic inference-time verifier claim |
| [Agentic AI for Robot Control](https://arxiv.org/abs/2602.13081) | 2026-02 | Iterative planner-executor system with skill calls, event checks, and operator intervention | Demonstrates portability but also prompt sensitivity and nondeterministic fragility |
| [Harnessing Embodied Agents](https://arxiv.org/abs/2604.07833) | 2026-04 | External runtime governance for capability admission, monitoring, rollback, and human override | Occupies broad external execution-governance and recovery framing |

Implication: planner, memory, critic, retry, and verifier are architectural
components, not independent novelty claims. CARVE must show a new control rule,
new coupling, or new evidence that emerges from their interaction.

## Closest Corrective Runtime Work

| Work | Date | Main mechanism | Relation to CARVE |
|---|---:|---|---|
| [VLA-Corrector](https://arxiv.org/abs/2607.01804) | 2026-07 | Latent visual monitor detects predicted and observed deviation, truncates stale actions, and invokes corrective gradient guidance | Very close to detect-truncate-replan; CARVE must go beyond visual deviation and fixed corrective logic |
| [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161) | 2026-04 | Uses action entropy to select inference-time chunk length | Occupies adaptive action horizon as a standalone contribution |
| [DREAM-Chunk](https://arxiv.org/abs/2606.18589) | 2026-06 | Lightweight latent world model evaluates candidate chunks under stochastic dynamics | Occupies world-model-based reactive chunk selection |
| [Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778) | 2026-05 | Draft policy, parallel Action Expert verification, and phase-aware full-model fallback | Occupies speculative low-latency replanning; reports 3.04x average speedup |
| [Realtime-VLA V2](https://arxiv.org/abs/2603.26360) | 2026-03 | Practical kernel, trajectory, and deployment techniques for fast and smooth VLA control | Raises the expected realtime baseline substantially |
| [AC2-VLA](https://arxiv.org/abs/2601.19634) | 2026-01 | Action-context-conditioned cognition reuse, token pruning, and selective component execution | Occupies model-internal adaptive computation with training |
| [AR-VLA](https://arxiv.org/abs/2603.10126) | 2026-03 | Asynchronous action generation and re-anchoring for stale perception | Pressures broad asynchronous control claims |

CARVE's remaining distinction can be:

- event semantics beyond latent visual discrepancy, including stall, grasp,
  subgoal completion, stale-action age, and task-state inconsistency;
- one joint decision over action reuse, ordinary replanning, recovery skill,
  semantic planner escalation, inference effort, and deadline fallback;
- operation through a policy capability boundary without retraining the frozen
  backbone;
- explicit success, latency, and intervention optimization rather than only
  action fidelity or average inference speed.

## Reasoning Is Moving Inside Training

- [Continuous Reasoning for VLA](https://arxiv.org/abs/2606.00229) learns
  structured continuous thoughts shared between model instances and verifies
  them through downstream action improvement.
- [ZR-0](https://arxiv.org/abs/2606.30552) uses dense embodied chain-of-thought
  supervision during training while allowing explicit ECoT generation to be
  skipped entirely at inference.
- [WLA](https://arxiv.org/abs/2606.05979) similarly lets world prediction
  improve representation learning while making the expensive branch optional
  at inference.

Implication: always-on textual reasoning is becoming less attractive for
real-time control. CARVE's VLM planner should be a sparse escalation path. Fast
monitoring, verification, and execution decisions should use compact visual and
proprioceptive state rather than generating language every control cycle.

## Efficient VLA and Compression Work

| Work | Date | Main result | Consequence for CARVE |
|---|---:|---|---|
| [QVLA](https://arxiv.org/abs/2602.03782) | 2026-02, ICLR 2026 | Channel-wise action-sensitive mixed precision; 29.2% VRAM and 1.49x speedup while retaining 98.9% relative performance | Generic action-aware VLA quantization is already occupied |
| [ActQuant](https://arxiv.org/abs/2605.24011) | 2026-05 | Sub-4-bit action-guided PTQ plus native C/C++ runtime; evaluates pi0.5 and OpenVLA-OFT | Strongest direct pressure on an original CARVE quantization claim |
| [Drop-Then-Recovery](https://arxiv.org/abs/2606.27755) | 2026-06 | Finds large language-backbone redundancy through block removal and recovery fine-tuning | Suggests compression should target language capacity, but requires fine-tuning |
| [RLRC](https://arxiv.org/abs/2506.17639) | 2025-06 | Structured pruning, SFT and RL recovery, and optional quantization | Occupies train-and-recover compression pipelines |
| [Realtime-VLA kernels](https://github.com/dexmal/realtime-vla) | active | Open pi0 and pi0.5 inference kernels report roughly 20-40 ms on RTX 4090 depending on camera count and prompt setting | CARVE must compare with an optimized backend, not only stock PyTorch latency |
| [Embodied.cpp](https://arxiv.org/abs/2607.02501) | 2026-07 | Portable C++ runtime with modular multi-rate execution across VLA and preliminary WAM deployments | Generic cross-platform inference runtime is no longer a defensible primary novelty |

Decision:

- quantization remains a deployment condition or backend plug-in;
- do not claim a new PTQ algorithm unless it introduces a mechanism beyond
  QVLA and ActQuant and is validated in closed loop;
- use Realtime-VLA kernels as a strong backend where integration is practical;
- evaluate CARVE's end-to-end latency overhead on top of both stock and
  optimized inference.

## Memory and Evaluation Work

- [RoboMME](https://arxiv.org/abs/2603.04639) defines 16 tasks across temporal,
  spatial, object, and procedural memory, and evaluates 14 pi0.5-based memory
  variants. A generic memory buffer is not novel; memory must be task-dependent
  and experimentally active.
- [LIBERO-Plus](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)
  shows that high clean LIBERO success can collapse under controlled changes in
  viewpoint, robot initialization, layout, lighting, texture, and noise.
- [vla-eval](https://arxiv.org/abs/2603.13966) already provides Docker isolation,
  a WebSocket protocol, episode sharding, and benchmark/model decoupling. CARVE
  should integrate as a model-server wrapper instead of maintaining another
  benchmark launcher.
- [ForesightSafety-VLA](https://arxiv.org/abs/2606.27079) shows the trend toward
  process-level safety metrics, including cumulative safety cost and risk
  exposure time, rather than binary success alone.
- [Same Weights, Different Robot](https://arxiv.org/abs/2606.03724) shows that
  action unnormalization metadata and controller conventions are part of the
  executable policy: a matching checkpoint alone does not guarantee equivalent
  robot behavior.

Decision:

- keep compact task state and event history in CARVE;
- do not make long-term memory a headline contribution unless a RoboMME pilot is
  positive;
- adopt vla-eval for new benchmark execution;
- add process metrics: false intervention, risk exposure, recovery latency,
  action staleness, and VLA calls per successful recovery.

## WAM and World-Language-Action Direction

June 2026 produced several efficient world-action systems:

- [WLA](https://arxiv.org/abs/2606.05979) unifies textual subtasks, world
  prediction, and action synthesis, with world prediction optionally disabled
  or activated for test-time scaling.
- [AHA-WAM](https://arxiv.org/abs/2606.09811) asynchronously separates a
  low-frequency world planner from a high-frequency action expert and reports
  24.17 Hz.
- [Efficient-WAM](https://arxiv.org/abs/2606.10040) uses sparse coarse future
  latents and asymmetric video and action denoising.
- [Light-WAM](https://arxiv.org/abs/2606.08242) reports a 0.44B trainable model,
  72.03 ms latency, and 4.1 GiB peak memory.
- [WALL-WM](https://arxiv.org/abs/2606.01955) organizes training and inference
  around semantic action events rather than fixed clock chunks.

These works require model training and world or video supervision. They confirm
that asynchronous multi-rate control is important, but they are not a practical
mainline for the current compute budget.

Decision: preserve optional predictive-context capabilities in PolicyAdapter,
but do not add a WAM experiment until the frozen-VLA result is complete.

## Revised Novelty Matrix

| Candidate claim | Status after update |
|---|---|
| First Agentic VLA | Invalid; Agentic-VLA and multiple hierarchical systems exist |
| First VLA harness | Invalid; vla-eval, physical-AI harness work, and OpenRAL exist |
| Portable VLA or WAM runtime | Invalid; Embodied.cpp directly targets this problem |
| VLM planner plus VLA controller | Crowded; use as architecture, not novelty |
| Retry or recovery expert | Crowded; needs a principled trigger and matched evidence |
| Memory-augmented VLA | Crowded; requires a RoboMME-style memory result |
| Adaptive action chunk | Occupied by AAC and VLA-Corrector |
| Visual verification | Occupied by VERITAS and VLA-Corrector |
| Dynamic flow steps | Useful but insufficient alone |
| VLA-specific quantization | Occupied by QVLA and ActQuant |
| Runtime sparse MoE | Risky terminology; parameter MoE and action routing already exist |
| Joint risk-driven recovery and compute scheduling under deadlines | Still promising |
| Capability-aware fallback across heterogeneous frozen policies | Useful secondary systems contribution |
| Failure-conditioned branching evaluation | Strong mechanism evidence if standardized and paired with end-to-end success |

## Revised CARVE Method

### Deployable Inputs

The main method should use only signals available outside the simulator:

- current and recent RGB observations;
- proprioceptive state;
- action history and action-chunk age;
- model outputs or uncertainty when exposed;
- compact task and subgoal state;
- measured inference latency and deadline slack.

Simulator object poses may define evaluation labels or an oracle upper bound,
but must not drive the deployable CARVE controller.

### One Joint Decision

At each event boundary, CARVE estimates execution risk and selects:

    execution mode:
      reuse cached action
      invoke fast VLA
      invoke accurate VLA
      invoke recovery or regrasp skill
      escalate to semantic planner

    compute allocation:
      inference steps
      committed action horizon
      verifier frequency
      context or cache reuse
      timeout and fallback budget

The key experiment must compare this coupled decision against:

- frozen VLA;
- static full stack;
- Agentic routing with fixed compute;
- adaptive compute without Agentic recovery;
- coupled CARVE;
- optional oracle or offline router.

### Memory Scope

Memory stores event, attempted intervention, verification outcome, current
subgoal, and latency history. Retrieval suppresses repeated failed
interventions and supports escalation. It is not an always-on semantic RAG
system and is not claimed as a standalone contribution.

## Short Experimental Route

### Required

1. **Failure-state branching study** on three LIBERO tasks:
   snapshot matched recoverable failures and branch frozen continuation,
   ordinary replanning, Agentic recovery, and coupled CARVE.
2. **End-to-end paired evaluation** on T6, T8, and T9 with matched seeds.
3. **Small LIBERO-Plus subset** focused on robot initialization, object layout,
   and sensor noise.
4. **Realtime frontier**: success, p50 and p95 observation-to-action latency,
   deadline misses, action staleness, policy calls, and peak VRAM.
5. **One optimized backend comparison** using Realtime-VLA kernels if the pi0.5
   checkpoint path is compatible.

### Optional and Result-Gated

- one secondary VLA adapter on two to four tasks;
- RoboMME four-task pilot only if event memory is active and beneficial;
- VLA-Corrector as a close baseline if its released code is reproducible.

### Defer

- RoboTwin or RoboDojo training;
- full RoboMME;
- a new WAM;
- parameter-level MoE;
- a new quantization algorithm;
- a broad ROS 2 middleware platform.

## Immediate Engineering Changes

1. Keep the new PolicyAdapter and capability contract, but describe them as
   enabling infrastructure.
2. Add an executable action specification to each adapter: action dimensions,
   coordinate frame, normalization metadata, gripper convention, control rate,
   and valid output region.
3. Add a deployable observation and proprioception deviation monitor; remove
   privileged object poses from the main controller.
4. Connect risk and deadline slack to both execution-mode selection and compute
   controls.
5. Extend the local pi0.5 server protocol so dynamic flow steps are actually
   applied; remote unsupported controls must remain explicit in traces.
6. Add simulator-state branching and paired branch manifests.
7. Integrate CARVE as a vla-eval-compatible model server instead of installing
   each benchmark manually.
8. Treat QVLA, ActQuant, and Realtime-VLA as baselines or backends, not claims
   to reproduce from scratch.

## Bottom Line

The field has moved quickly, but the project has not become obsolete. The broad
"Agentic harness plus efficient VLA" story is now too diffuse. A narrower
"deadline-aware recovery runtime for frozen VLA policies" is more technically
coherent, more defensible against the July 2026 literature, and cheaper to
validate.

# Realtime Agentic Policy Harness for Efficient and Recoverable VLA Deployment

Draft date: 2026-06-12

Status: integrated manuscript draft after the robosuite pi0.5 closed-loop experiments.

Primary evidence files:

- `results/paper_assets_20260609/table_pi05_agentic_retry_matched.md`
- `results/paper_assets_20260609/table_pi05_realtime_sweep.md`
- `results/paper_assets_20260609/figures/pi05_realtime_sweep.png`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_closed_loop_smoke.mp4`
- `results/robosuite_stack_e2e_v2_20260611/eval_midnudge_agentic_retry_light_safe1_demo.mp4`

## 0. Paper Story Lock

This paper should be written as a **deployment-runtime paper**, not as a new VLA backbone paper and not as a broad benchmark leaderboard paper.

### Central Thesis

A VLA can be a strong action generator but still fail as a robot deployment system. In closed-loop manipulation, the runtime must know when execution is no longer progressing, when to invoke physical recovery, and how to keep VLA inference within a practical realtime budget. We therefore wrap pi0.5 with a realtime-aware Agentic Policy Harness that detects stalled execution, invokes physical retry, and avoids repeated full VLA calls during failed segments.

### One-Sentence Claim

> A realtime-aware Agentic Policy Harness improves recoverability and efficiency of existing VLA deployment by turning stalled closed-loop execution into physical recovery while using low-step pi0.5 inference and reducing repeated full VLA calls.

### Evidence Chain

| Evidence | Role in the paper | Main result |
|---|---|---|
| pi0.5 matched robosuite Stack | Core Agentic recovery evidence | raw pi0.5 `0/10`; pi0.5 + Agentic retry `10/10` |
| Full-call reduction | Core deployment-efficiency evidence | full pi0.5 calls `43.0` -> `6.0` per episode |
| pi0.5 realtime sweep | Core VLA efficient-inference evidence | fixed-noise `1` step is latency-preferred; `2` step is quality fallback |
| Compact-policy pipeline | Supplemental runtime-modularity evidence | Agentic retry `9/10`; Agentic + LightSafe1 keeps `9/10` and reduces full calls by about `45.3%` |
| HD videos/contact sheets | Qualitative simulation evidence | visible robosuite/MuJoCo robot motion and contact |

### What This Paper Is Not Claiming

- It does not claim that pi0.5 itself learned the recovery behavior.
- It does not claim universal improvement over LIBERO, RoboTwin, or ManiSkill.
- It does not claim real-robot validation.
- It does not claim quantization is the current measured speedup source.
- It does not mix compact-policy success numbers with pi0.5 success numbers.

## 0.1 Writing Checklist

This draft completes the current five integration tasks:

1. Method section: Agentic Policy Harness, realtime VLA inference module, and coupled runtime.
2. Main results section: matched pi0.5 closed-loop robosuite Stack comparison.
3. Realtime sweep section: fixed-noise low-step VLA inference tradeoff.
4. Compact policy supplement: data collection, training, CAQ-Lite, and Agentic coupling evidence.
5. Limitations section: result boundary, simulator assumptions, physical recovery skill, and real-robot gap.

## Abstract

Vision-language-action (VLA) policies provide a strong interface for language-conditioned robot control, but direct deployment remains fragile when long-horizon contact execution, recovery, and realtime inference constraints appear together. A frozen or lightly adapted VLA can produce useful action chunks, yet it does not by itself decide when execution has stalled, when recovery should take over, or how many expensive full VLA calls are justified under a realtime control budget. We present a **Realtime Agentic Policy Harness** for VLA deployment. The system wraps a VLA backend with progress monitoring, a physical critic, gated recovery, trace logging, and a low-step deterministic inference module. The harness is designed to improve recoverability while reducing repeated VLA calls during stalled execution segments. On robosuite Stack with a pi0.5 PyTorch backend, raw fixed-noise 1-step pi0.5 achieves `0/10` success, while pi0.5 with Agentic retry achieves `10/10` success under the matched protocol. The same setting reduces full pi0.5 calls from `43.0` to `6.0` per episode and keeps steady inference latency near `110ms`. A realtime sweep shows that fixed-noise 1-step inference is the current latency-preferred setting, while 2-step inference gives the best open-loop action MSE at higher latency. A compact learned-policy pipeline further shows that Agentic recovery and conservative action reuse can be coupled in a full data-collection, training, and deployment loop. These results support a deployment-oriented claim: agentic recovery and realtime-aware VLA inference are complementary tools for making existing VLAs more usable in robot manipulation without training a new foundation model.

## 1. Introduction

VLA policies map visual observations and language instructions to robot actions, and recent systems such as RT-style policies, OpenVLA, pi0, and pi0.5 demonstrate the promise of large multimodal robot policies [@brohan2022rt1; @brohan2023rt2; @kim2024openvla; @black2024pi0; @physicalintelligence2025pi05]. However, a capable action generator is not yet a complete deployment stack. During long-horizon manipulation, a robot must detect whether the task is progressing, decide whether a failed contact should trigger recovery, avoid repeatedly executing stale actions, and meet realtime inference constraints.

This gap becomes especially important in resource-limited research settings. Training a new VLA backbone is usually infeasible on a single local GPU, while deployment-time system design remains practical and scientifically meaningful. Recent agentic robot systems follow a similar direction: they wrap a base policy with planning, verification, memory, transition, or recovery modules [@yang2025agenticrobot; @pang2026scivla; @li2026roboclaw; @torne2026mem; @zhao2025vla2; @jin2026agenticvla]. In parallel, efficient VLA inference work studies faster action generation, action reuse, caching, and realtime execution [@xu2025vlacache; @niu2026realtimevlaflash; @yang2026realtimevlav2; @guan2025efficientvlasurvey].

The central question in this paper is:

> How can an existing VLA be deployed as part of an agentic robot runtime that improves recovery and realtime efficiency without training a new foundation model?

We study this question through a practical runtime design. The VLA remains the main action generator. Around it, an Agentic Policy Harness monitors progress, triggers a physical recovery skill when the VLA stalls, and logs the execution trace. The VLA backend is also run with low-step deterministic inference to reduce per-call latency. The resulting system is not a new VLA architecture. It is a deployment harness: an execution-time layer that connects VLA action generation with recovery and realtime constraints.

Our contributions are:

1. **Agentic Policy Harness for VLA deployment.** We implement a progress monitor, physical critic, retry/recovery interface, lockout logic, and trace logger around a VLA backend.
2. **Realtime VLA inference module.** We evaluate fixed-noise low-step pi0.5 inference and quantify the latency-quality tradeoff across 1, 2, 4, 6, 8, and 10 inference steps.
3. **Coupled recovery-efficiency behavior.** We show that Agentic retry can both recover stalled execution and reduce repeated full VLA calls during failed long-horizon segments.
4. **Full simulation pipeline supplement.** We include a robosuite data-collection, compact policy training, and inference-time CAQ-Lite evaluation to validate the broader runtime idea beyond a single pi0.5 run.

## 2. Related Work

### 2.1 Vision-Language-Action Policies

RT-1 and RT-2 established large-scale language-conditioned robot policies and showed that multimodal pretraining can transfer semantic knowledge into robot control [@brohan2022rt1; @brohan2023rt2]. Open X-Embodiment and RT-X further emphasize the importance of cross-robot data [@oneill2023openx]. Open-source policies such as Octo and OpenVLA make generalist robot policies more accessible [@octo2024octo; @kim2024openvla]. pi0 and pi0.5 further improve general robot control and open-world generalization through stronger action-generation architectures and broader data [@black2024pi0; @physicalintelligence2025pi05].

This paper is complementary to VLA pretraining. We do not propose a new backbone. Instead, we ask how to wrap a VLA backend with execution-time recovery and realtime scheduling.

### 2.2 Agentic Robot Policies

Agentic robot systems treat the base policy as one component inside a broader control loop. Agentic Robot-style frameworks add planner, executor, verifier, and recovery modules around VLA execution [@yang2025agenticrobot]. Sci-VLA inserts agent-generated transition actions between atomic VLA skills [@pang2026scivla]. RoboClaw broadens the runtime to a lifecycle framework with memory, tools, policy pools, and self-resetting rollouts [@li2026roboclaw]. MEM-style work shows that embodied memory can help partial observability and repeated execution [@torne2026mem]. VLA^2 and Agentic-VLA explore agentic augmentation and online adaptation for unseen concepts or new task distributions [@zhao2025vla2; @jin2026agenticvla].

Our harness follows this agentic policy direction, but focuses on a deployment-specific coupling: physical recovery should not only improve success; it should also reduce wasted VLA calls when execution is already stalled.

### 2.3 Efficient and Realtime VLA Inference

Realtime deployment requires more than high offline success. Full VLA inference can be slow relative to robot control loops, and slow calls are especially harmful when repeated during a stalled segment. Existing work studies token caching, speculative inference, efficient VLA backends, embodied reasoning reuse, and realtime VLA inference [@xu2025vlacache; @zawalski2024ecot; @duan2025fastecot; @niu2026realtimevlaflash; @yang2026realtimevlav2; @guan2025efficientvlasurvey].

Our current online speedup comes from two practical sources: low-step deterministic pi0.5 inference and fewer repeated full VLA calls after Agentic recovery takes over. Quantization and heavier backend acceleration remain natural future extensions, but they are not the main empirical claim in this draft.

## 3. Method

### 3.1 Problem Setup

Let `pi_theta` denote a VLA backend. At control step `t`, it receives observation `o_t`, robot state `s_t`, and language instruction `l`, and returns an action chunk:

```text
A_t = pi_theta(o_t, s_t, l) = (a_t, a_{t+1}, ..., a_{t+H-1}).
```

A naive executor periodically calls `pi_theta` and applies actions until the episode ends. This executor has two common failure modes:

1. It does not know that contact execution has stalled, so it may repeatedly query and execute ineffective actions.
2. It pays full VLA inference cost even when the current phase would be better handled by a recovery skill or cached low-risk action sequence.

The proposed runtime adds an execution state `m_t`, a progress signal `p_t`, a recovery flag `r_t`, and optional cached action buffer `B_t`. At each step, the runtime decides whether to call the VLA, execute a buffered action, or enter recovery.

### 3.2 VLA Backend

The main VLA backend is a pi0.5 PyTorch policy interface. In the robosuite experiments, the deployed backend is the local head-plus-300 pi0.5 variant used for the Stack task interface. The foundation model is not retrained as part of the runtime experiments. The runtime records:

- full pi0.5 calls per episode;
- inference latency after warmup;
- executed control steps;
- success and final phase;
- recovery triggers and recovery success;
- generated video/contact sheet assets.

The inference module uses fixed-noise low-step sampling. This turns the sampler into a deterministic or near-deterministic low-latency backend suitable for closed-loop evaluation.

### 3.3 Agentic Policy Harness

The harness supervises execution without replacing the VLA.

**Progress monitor.** The monitor tracks task phase and execution progress. For Stack, the key phases include approach, grasp/contact, lift, align, place, and done. The monitor detects whether the robot is still making progress toward the next physical condition.

**Physical critic.** The critic checks stalled or invalid execution patterns. In the current robosuite Stack implementation, the most important signal is that raw pi0.5 repeatedly remains in the approach/contact region without completing the stack.

**Retry/recovery skill.** When the critic confirms a stalled state, the harness clears stale VLA execution and transfers control to a physical retry skill. The retry skill completes the recovery segment and returns control once the environment reports success or the horizon is reached.

**Recovery lockout.** During recovery, cached or stale VLA actions are not reused. This prevents the runtime from executing an old failed plan immediately after entering a critical phase.

**Trace logger.** The logger records full VLA calls, inference latency, recovery triggers, retry steps, and final episode state. This makes the runtime auditable and provides the measurements used in the results tables.

The design should be interpreted as a physical execution harness. The VLA remains responsible for ordinary action generation, while the harness handles the failure mode where repeated VLA calls do not resolve a contact-level stall.

### 3.4 Realtime VLA Inference Module

The realtime module uses fixed-noise low-step pi0.5 sampling. The number of inference steps is a deployment knob:

- `1-step`: lowest latency and current closed-loop default;
- `2-step`: slightly higher latency and best open-loop MSE in the current sweep;
- `4/6/8/10-step`: higher latency, no observed quality benefit in the measured fixed-noise sweep.

The module does not claim to be a general compression method. Its current role is to expose a practical latency-quality tradeoff for pi0.5 deployment on a single RTX 4090.

### 3.5 Coupling Agentic Recovery and Realtime Execution

The coupling is the central runtime idea:

1. Before recovery, the VLA is queried at a fixed replan interval.
2. If execution stalls, the critic triggers Agentic retry.
3. The retry skill takes over the failed physical segment.
4. Full VLA calls stop during this recovery segment.
5. The trace records both success recovery and the reduction in repeated VLA calls.

This coupling differs from simply adding a recovery primitive. The runtime measures recovery as part of a realtime deployment budget: a successful recovery should improve the task outcome and avoid wasting additional full VLA calls on an already failed segment.

### 3.6 Compact Policy and CAQ-Lite Supplement

The compact policy pipeline tests the broader runtime mechanism in a full simulation loop:

1. collect robosuite Stack demonstrations;
2. train a compact RGB-plus-state action-chunk policy;
3. evaluate clean and mid-nudge episodes;
4. compare no-retry, Agentic retry, and Agentic retry with conservative action reuse.

The lightweight reuse module is referred to as CAQ-Lite. It reuses action suffixes in low-risk phases and falls back to fresh calls during recovery or critical states. This result is supplemental because it uses a compact learned policy rather than pi0.5. Its role is to show that the runtime concept extends to a data-collection, training, and deployment pipeline.

## 4. Experimental Setup

### 4.1 Simulator and Task

Experiments use robosuite Stack with a Panda robot and MuJoCo physics [@zhu2020robosuite; @todorov2012mujoco]. The task requires the robot to manipulate two cubes and complete a stack. The environment provides a useful mid-level manipulation setting: it is more physical than a mock benchmark, includes contact-rich execution, and produces videos for qualitative inspection.

Main protocol:

- simulator: robosuite Stack;
- robot: Panda;
- backend: pi0.5 PyTorch head-plus 300;
- inference: fixed-noise low-step sampling;
- closed-loop horizon: `430`;
- matched trial seeds: `20260611` to `20260620`;
- default closed-loop setting: fixed-noise `1-step`.

### 4.2 Compared Methods

The main matched comparison uses:

| Method | Description |
|---|---|
| raw pi0.5, fixed `1` step | Direct pi0.5 closed-loop execution without Agentic recovery |
| pi0.5 + Agentic retry, fixed `1` step | Same VLA backend plus progress monitor, critic, and physical retry |
| pi0.5 + Agentic retry, fixed `2` step | Five-trial sampler ablation for quality-latency comparison |

The compact policy supplement uses:

| Method | Description |
|---|---|
| learned + Light, no retry | Compact policy with action reuse but no physical recovery |
| learned + Agentic retry | Compact policy with recovery but no reuse |
| learned + Agentic retry + LightSafe1 | Compact policy with recovery and conservative reuse |

### 4.3 Metrics

We report:

- success count;
- mean episode steps;
- full VLA or policy calls per episode;
- recovery success;
- retry steps;
- steady mean inference latency;
- steady P95 inference latency;
- compact-policy action reuse ratio where applicable.

The main paper should emphasize both success and compute behavior. A method that only improves success by repeatedly calling a large VLA is less deployable than a method that improves success while reducing unnecessary full calls.

### 4.4 Experiment Matrix

The experiments are organized around three research questions.

| RQ | Question | Experiment | Primary metric | Paper role |
|---|---|---|---|---|
| RQ1 | Can Agentic recovery fix stalled VLA execution? | raw pi0.5 vs pi0.5 + Agentic retry on robosuite Stack | success, recovery, full pi0.5 calls | main result |
| RQ2 | Which pi0.5 inference setting is practical for realtime deployment? | fixed-noise and zero-noise low-step sweep | inference latency, chunk MSE, first gripper accuracy | realtime/lightweight result |
| RQ3 | Can recovery and lightweight reuse be coupled in a full simulation pipeline? | compact policy + Agentic retry + LightSafe1 | success, full calls, reuse ratio | supplemental mechanism study |

The current main paper should lead with RQ1 and RQ2. RQ3 should be used to show that the runtime idea generalizes beyond a single pi0.5 run, while remaining clearly labeled as a compact-policy supplement.

## 5. Results

### 5.1 RQ1: Does the Agentic Harness Improve pi0.5 Closed-Loop Execution?

The matched robosuite Stack comparison is the primary result.

| Method | Success | Steps mean | Full pi0.5 calls / ep | Recovery | Retry steps mean | Steady mean infer | Steady P95 infer |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw pi0.5, fixed `1` step | `0/10` | `430.0` | `43.0` | `0/0` | `0.0` | `111.99ms` | `122.37ms` |
| pi0.5 + Agentic retry, fixed `1` step | `10/10` | `254.4` | `6.0` | `10/10` | `194.4` | `110.40ms` | `116.87ms` |

Raw pi0.5 fails all ten trials under the matched fixed-noise `1-step` setting. The final state typically remains before successful task completion, indicating that direct closed-loop execution does not resolve the contact-level long-horizon segment. Adding the Agentic retry harness changes the result to `10/10` success.

The same comparison also shows a compute effect. Raw pi0.5 performs `43.0` full calls per episode because it keeps replanning until the horizon. With Agentic retry, full pi0.5 calls drop to `6.0` per episode because the physical recovery skill takes over the stalled segment. Steady inference latency remains similar, around `110ms`, so the improvement is not caused by faster individual pi0.5 calls. It comes from changing what the runtime does after a failure is detected.

The most accurate claim is therefore:

> The Agentic Harness converts a stalled pi0.5 execution into a successful physical recovery and reduces repeated full VLA calls during the failed segment.

### 5.2 RQ2: Which pi0.5 Inference Step Count Is Best for Realtime Deployment?

The realtime sweep evaluates fixed-noise and zero-noise samplers on the same sampled robosuite Stack frames. The fixed-noise results are:

| Steps | Chunk MSE | Mean infer | First gripper acc |
|---:|---:|---:|---:|
| `1` | `0.1523` | `118.15ms` | `0.75` |
| `2` | `0.1467` | `136.53ms` | `0.75` |
| `4` | `0.1747` | `202.24ms` | `0.65` |
| `6` | `0.1887` | `255.46ms` | `0.65` |
| `8` | `0.2021` | `323.89ms` | `0.60` |
| `10` | `0.2077` | `374.43ms` | `0.60` |

The sweep supports two deployment conclusions. First, fixed-noise `1-step` is the latency-preferred closed-loop setting. Second, fixed-noise `2-step` has the lowest open-loop action MSE in the current sweep, but it costs roughly `18ms` more per call. In the Agentic retry ablation, both `1-step` and `2-step` solve all five tested trials:

| Method | Success | Steps mean | Full pi0.5 calls / ep | Recovery | Retry steps mean | Steady mean infer | Steady P95 infer |
|---|---:|---:|---:|---:|---:|---:|---:|
| pi0.5 + Agentic retry, fixed `1` step | `5/5` | `254.4` | `6.0` | `5/5` | `194.4` | `112.68ms` | `123.37ms` |
| pi0.5 + Agentic retry, fixed `2` step | `5/5` | `254.4` | `6.0` | `5/5` | `194.4` | `137.22ms` | `142.74ms` |

For realtime deployment on the current RTX 4090 machine, `1-step` is the default setting and `2-step` is the quality-oriented fallback.

### 5.3 RQ3: Can Agentic Recovery and Lightweight Action Reuse Be Coupled?

The compact learned-policy experiment is a supplemental mechanism study. It uses `100` successful robosuite Stack demonstrations and trains a compact RGB-plus-state action-chunk policy. This experiment is not a pi0.5 result and should not be reported as such.

| Method | Success | Full calls / ep | Reused / ep | Reuse ratio | Recovery |
|---|---:|---:|---:|---:|---:|
| learned + Light, no retry | `4/10` | `157.0` | `234.4` | `0.599` | `0/0` |
| learned + Agentic retry | `9/10` | `127.5` | `0.0` | `0.000` | `10/10` |
| learned + Agentic retry + LightSafe1 | `9/10` | `69.7` | `62.6` | `0.473` | `10/10` |
| clean Agentic retry + LightSafe1 | `10/10` | `64.3` | `63.8` | `0.498` | `10/10` |

This table supports the broader runtime design. Agentic retry is the primary source of success under mid-nudge perturbation. Conservative action reuse then reduces full policy calls while preserving the same `9/10` success as Agentic-only in the mid-nudge setting. Relative to learned + Agentic retry, LightSafe1 reduces full calls from `127.5` to `69.7`, a reduction of about `45.3%`.

The role of this section is not to replace the pi0.5 result. It shows that the recovery-plus-efficiency idea can be implemented in a complete data-collection, training, and inference pipeline.

### 5.4 Qualitative Evidence

Recommended qualitative assets:

- pi0.5 Agentic retry HD video: `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_closed_loop_smoke.mp4`
- pi0.5 Agentic retry contact sheet: `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_agentic_retry_hd256_contact.png`
- compact policy mid-nudge demo: `results/robosuite_stack_e2e_v2_20260611/eval_midnudge_agentic_retry_light_safe1_demo.mp4`

The pi0.5 video should be used to show that the result is produced in a real simulator with visible robot motion and contact, rather than a mock state-machine-only test.

## 6. Discussion

### 6.1 Why the Agentic Result Matters

The main pi0.5 result is strong because it isolates a deployment failure: direct closed-loop VLA execution repeatedly fails, while the same backend inside an Agentic Harness succeeds. The improvement is not framed as VLA retraining. It is a runtime intervention that detects when the VLA is no longer making progress and hands control to a physical recovery skill.

This aligns with the broader Agentic Policy view: for robot manipulation, the action model can be treated as one physical skill among several. The runtime decides when to use the VLA, when to verify, and when to recover. In this interpretation, the VLA is a powerful general action generator, while recovery skills are specialized physical tools.

### 6.2 Why Realtime Belongs in the Main Story

Realtime inference is not an unrelated engineering detail. In robot control, a policy that succeeds only with slow or repeated blocking calls may be hard to deploy. The current results connect realtime to recovery in two ways:

1. Low-step pi0.5 inference reduces per-call cost.
2. Agentic retry reduces the number of full calls after a stalled segment is detected.

This gives the paper a coherent dual contribution: recoverability and realtime-aware deployment.

### 6.3 Relationship Between pi0.5 and Compact Policy Results

The pi0.5 results should be the main paper result because they use the intended VLA backend. The compact policy results should be presented as supplemental evidence for runtime modularity. They demonstrate that the same harness concepts can support a smaller trainable policy and a conservative action reuse module.

## 7. Limitations

1. **Benchmark scope.** The strongest current result is on robosuite Stack. It demonstrates a real simulated contact task, but it is not a broad benchmark claim over LIBERO-Plus, RoboTwin, ManiSkill, or real robots.

2. **Physical recovery skill boundary.** The current Agentic retry uses a physical recovery skill in simulation. This supports the claim that a VLA can be deployed inside a recovery-capable harness, but it does not mean the VLA itself learned the recovery behavior.

3. **Simulator state assumptions.** The robosuite recovery implementation may rely on state information that is easier to access in simulation than on a real robot. Real deployment would require perception-grounded object pose estimation, visual servoing, or a real-robot calibrated recovery controller.

4. **Trial count.** The main matched result has `10` trials. This is sufficient for the current project report and draft-level evidence, but a conference or journal submission would be stronger with more seeds, more tasks, or a real-robot/photorealistic simulation demo.

5. **Realtime boundary.** The current pi0.5 `1-step` inference is around `110-118ms` on the local RTX 4090 setting. This is much more deployable than higher-step inference, but it is still not a high-frequency torque-control loop. The correct claim is realtime-aware deployment, not hard realtime control at kilohertz rates.

6. **Quantization not yet the main result.** The project direction includes VLA lightweight deployment, but the current strongest evidence is low-step deterministic inference and call-frequency reduction. Quantization should be added only if it provides measured memory or latency gains without hurting closed-loop success.

7. **Compact policy is supplemental.** The compact policy CAQ-Lite experiment validates the runtime mechanism, but it should not be mixed with the pi0.5 success numbers.

## 8. Recommended Paper Figures and Tables

**Figure 1: System overview.** Frozen or lightly adapted VLA backend, Agentic Policy Harness, realtime inference module, physical recovery skill, and trace logger.

**Figure 2: Agentic retry flow.** Direct VLA execution, stall detection, recovery lockout, physical retry, success trace.

**Figure 3: Realtime sweep.** Use `results/paper_assets_20260609/figures/pi05_realtime_sweep.png`.

**Table 1: Matched pi0.5 closed-loop result.** Use `results/paper_assets_20260609/table_pi05_agentic_retry_matched.md`.

**Table 2: Realtime sweep.** Use `results/paper_assets_20260609/table_pi05_realtime_sweep.md`.

**Table 3: Compact policy supplement.** Use the compact learned-policy table in Section 5.3.

**Figure 4: Qualitative contact sheet.** Use `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_agentic_retry_hd256_contact.png`.

## 9. Claim Boundary for Submission

Recommended headline claim:

> A realtime-aware Agentic Policy Harness can make an existing VLA deployment more recoverable and efficient by detecting stalled physical execution, invoking recovery, and avoiding repeated full VLA calls during failed long-horizon segments.

Claims supported by current experiments:

- pi0.5 has been connected to robosuite/MuJoCo closed-loop execution.
- fixed-noise low-step pi0.5 inference provides a measurable latency-quality tradeoff.
- Agentic retry changes matched robosuite Stack success from `0/10` to `10/10`.
- Agentic retry reduces full pi0.5 calls from `43.0` to `6.0` per episode in the matched setting.
- compact policy experiments support the broader recovery-plus-lightweight-runtime design.

Claims to avoid:

- The method universally improves all VLA manipulation benchmarks.
- pi0.5 itself has learned the recovery skill.
- the system has already been validated on a real robot.
- quantization is the source of the current main speedup.

## 10. Next Manuscript Tasks

1. Convert this Markdown draft into the target paper template.
2. Draw Figure 1 and Figure 2 with the current module names.
3. Place the realtime sweep figure and qualitative contact sheet.
4. Convert the robosuite and MuJoCo simulator citations into the final venue's BibTeX style.
5. Decide whether to run optional `20`-trial expansion before submission.

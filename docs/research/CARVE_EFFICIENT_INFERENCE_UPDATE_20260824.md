# CARVE Efficient-Inference Research Update

Updated: 2026-09-22 (original file retained)

## September 22 Update: Couple Efficiency To Executable Agent Decisions

本轮为调研与静态核查，没有启动 GPU 实验，也没有将以下技术置为默认。
下方 September 9 的 StarVLA 部署与耗时属于历史快照；当前 RoboDojo 使用匹配的
PI0.5 JAX checkpoint，不能将不同模型/任务/后端的加速数字直接拼接。

### 新近工作与落地边界

| 来源 | 核查深度与可借鉴机制 | 当前采用判断 |
|---|---|---|
| [Realtime-VLA V2](https://arxiv.org/abs/2603.26360)，3月27日；[官方代码](https://github.com/dexmal/realtime-vla-v2) | 摘要与部署 README；JAX/Triton 后端、时间轴动作执行和完整客户端链路 | 优先借鉴分段计时与动作对齐。其训练式 action prefill、速度学习和机器人控制器不能当普通 checkpoint 的开关 |
| [SV-VLA](https://arxiv.org/html/2604.02965v1)，4月3日 | 方法和3.3训练策略；重 VLA 低频生成、轻 verifier 监控偏差后再规划 | 轻 verifier 要训练，和现有规则 Monitor 不是同一方法；先借鉴对照协议，不立即增训一个模型 |
| [Pre-VLA](https://arxiv.org/abs/2605.22446)，5月21日 | 摘要级候选；候选动作的事前评分和有限预算重采样 | 包含训练的安全/优势预测头；本地 FK 预览不能冒充该能力，暂不部署 |
| [ActionCache](https://arxiv.org/abs/2607.06370)，7月7日、8月3日修订 | 摘要级复核，结合既有调研；缓存中间动作并检索热启动 | 无需重新训练基础模型，但要改采样器、管理缓存来源；动作头加速不等于端到端同倍加速 |
| [ActQuant](https://arxiv.org/html/2605.24011v3)，5月初稿、8月3日修订 | 方法与部署概述；动作敏感的混合精度和量化尺度校准 | 需校准数据与真实低比特内核；不能把通用 NF4 等同于复现其算法，暂不追求极端2--3bit |
| [RTC 官方说明](https://www.pi.website/research/real_time_chunking)，2025年基础参考 | 官方日期与训练式扩展说明 | RTC 并非最近半年才提出。推理式与训练式实现分开，不能将预取线程称为 RTC |

### 适合本项目的三层分工

1. **模型层**：首先对当前部署后端记录稳态耗时与峰值显存，再单项比较编译/内核、
   flow 步数或低比特配置。保留校准集与测试集分离；动作头/视觉路径先保持参考精度，
   是否进一步压缩由敏感性结果决定。降低显存与降低延迟分别报告。
2. **Agent 调度层**：比较固定频率与边界触发，记录 Planner/Critic 各自成本及模型
   搬运开销。减少调用只有在没有错过必要干预时才有意义；更少调用不是成功的替代。
3. **动作执行层**：仅在现有队列和时钟支持下评估异步。计划更换、目标变化和恢复
   触发时，使过期计算和未提交动作失效；已经提交/执行的动作不能假装撤回。

近期优先交付的是同一套 Harness 的质量--成本对照，而不是同时复现全部加速论文。
候选采用顺序：测量瓶颈 -> 一项模型/驻留优化 -> 调度消融 -> 必要时再做缓存或 RTC。
若视觉判断本身失败，先修正或替换感知判断，不加速错误判断。

### 必须补足的消融与验收

- 同一 checkpoint、同一机器人动作表示、同一初始状态，比较参考与单项优化。
- 记录冷启动和稳态模型延迟、VLM/VLA调用数、传输/等待、回合耗时、峰值显存。
- 同时记录成功率、阶段进度、动作陈旧度以及 Agent 干预带来的救回和损害。
- 量化保留独立校准记录和真实权重/内核标识；fake quant 数值实验不是低显存部署。
- 缓存来源不能包含正在评分的测试轨迹，重规划/跨回合/换模型后按契约失效。
- 仿真暂停物理时的耗时下降只支持效率结论。实时性结论必须额外使推理延迟影响
  物理推进，固定控制周期、延迟注入与回退规则；不能只在暂停的模拟器中 sleep。

框架参考、已有代码与改进优先级详见
`CARVE_AGENTIC_POLICY_REFERENCE_SYNTHESIS_20260824.md` 第0节。

## September 9 Review: Select By Bottleneck, Not Recency

This is a literature and local-code review, not a new experiment or an admission
of any accelerator. The user mentioned RTC as a technology to investigate, not
as a requirement to adopt it. No checkpoint, deployment default, or GPU job was
changed. The earlier August roadmap below is historical; this section and the
current RoboDojo execution plan determine near-term priorities.

### Three Different Optimization Targets

| Target | What improves | What does not automatically improve |
|---|---|---|
| Model computation: kernels, compilation, fewer flow evaluations, selective reuse | Cost of generating one new action chunk | Semantic judgment quality, action continuity |
| Deployment capacity: low-bit weights, residency, transfer scheduling | VRAM use, model-switch time, ability to keep VLM/VLA available | Success rate or lower latency merely from smaller weights |
| Closed-loop execution: asynchronous generation and aligned chunks | Inference pauses, observation-to-action delay, discontinuities | Model size or time per model call |

RTC is principally in the third category. It is not INT4 quantization, and
its inference-time guidance can increase the computation of each generation.
Optimization must retain task performance and acceptable response to new events,
not merely improve a model-call timing statistic.

### Primary Sources And Adaptation Decisions

Sources checked on 2026-09-09. Paper results are authors' results, not ours.
Implementation availability does not establish compatibility with our checkpoint.

| Work | Relevant mechanism and requirement | Decision for this project |
|---|---|---|
| [RTC, June 2025](https://arxiv.org/abs/2506.07339) | Inference-time action-prefix guidance/inpainting for diffusion or flow policies; no retraining, but gradient-based guidance adds work | Useful optional continuity baseline after adapter and simulator-clock support; not an immediate default |
| [Training-Time RTC, December 2025](https://arxiv.org/abs/2512.05964) | Learn action-prefix conditioning with simulated delays to avoid guidance backward passes; requires compatible training | Do not enable on an ordinary checkpoint by changing a flag |
| [LeRobot RTC implementation](https://github.com/huggingface/lerobot/blob/main/docs/source/rtc.mdx) | Documents guided mode and trained mode; trained mode requires an appropriately trained Pi05 checkpoint | Reference queue/conditioning contracts, pin a commit before reuse; not a drop-in StarVLA server |
| [VLASH](https://arxiv.org/html/2512.01031v1) | Future-state-aware async execution; paper includes temporal-offset training augmentation | Study stale-state alignment; do not present it as wholly training-free. Its action quantization is trajectory coarsening, not low-bit weight quantization |
| [FutureRTC, July 2026](https://arxiv.org/html/2607.24008v1) | Predict future visual features and correct future state; base VLA frozen but auxiliary modules trained | Tracking candidate, not a zero-training adapter |
| [Reactive Real-time Flow Policies, piR2](https://pi-r2-flow.github.io/) | Fast proprioception, asynchronous vision/language, latency-adaptive flow schedule; fine-tunes a flow policy | Useful multi-rate design reference, not a reason to replace the current task model |
| [Realtime-VLA kernels](https://github.com/Dexmal/realtime-vla) | Architecture-specific Triton inference for Pi0/Pi05/DM0 | Strong reference for the historical Pi05 path; cannot directly substitute for QwenPI_v3 |
| [ActionCache, July/August 2026](https://arxiv.org/abs/2607.06370v2) | Training-free retrieval of intermediate actions and warm-start refinement | Optional after profiling; action-head speedup is not whole-system speedup, especially with only two flow steps |
| [ActQuant, August 2026 revision](https://arxiv.org/abs/2605.24011v3) | Action-guided mixed-precision PTQ and low-bit deployment | Reference for action-sensitive calibration; do not claim a new generic action-aware quantizer |
| [QVLA](https://arxiv.org/abs/2602.03782) / [QuantVLA](https://arxiv.org/abs/2602.20309v4) | Action-sensitive channel allocation / scale-calibrated PTQ | Study sensitivity and calibration. Keep pruning out of scope, and verify real kernels rather than fake-quant memory estimates |

The [PI announcement](https://www.pi.website/research/real_time_chunking) dates
original RTC to June 9, 2025 and links the December training-time extension used
in a pi*0.6 demonstration. Thus RTC is not a technique first introduced by pi0.6,
nor does investigating it require migrating our RoboDojo policy to pi0.6.
The [official research repository](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
contains Kinetix experiments; this is not an official ready-made RoboDojo adapter.

### What The Local Evidence Actually Says

Current RoboDojo model: StarVLA Qwen3-VL-4B PI-v3, not Pi05. September 9 server
metadata reports 50 predicted actions, a 14-dimensional dual-arm joint-position
contract, and 16 executed actions per request. The adapter declares 25 Hz.
This is a contract setting, not evidence that simulator wall time maintains 25 Hz.

The completed C3 development rollout reports 45 VLA calls totaling 10.44 seconds
and five staged semantic RPCs totaling 97.18 seconds, only 11.56 seconds of which
were generation. About 86.6% of those semantic RPCs was weight movement. That
percentage is NOT a percentage of the complete episode or an achievable speedup.
Evidence: `artifacts/robodojo/semantic_camera_pair_20260909/execution_audit.json`,
`C3/summary.json`, and the saved launch/server log.

Relevant implementation facts:

- `CpuStagedVisionPlanner` serializes VLA and VLM with a shared lock and moves
  models with `.to("cpu")` / `.to(device)`. Its contract explicitly pauses
  environment stepping. It solves memory admission, not continuous control.
- `agentic_vla/runtime/prefetch.py` implements single-flight prefetch and
  post-generation prefix checks, not RTC's prefix-conditioned sampling.
- Its legacy `ActionPrefixVerifier` measures six Cartesian components and one
  gripper. The September 9 follow-up now rejects non-7-D input explicitly and
  adds `JointPositionPrefixVerifier` for joint-position contracts with explicit
  gripper indices. This avoids silently ignoring the second arm or treating joint
  positions as Cartesian delta sums. The new helper is software-tested, not
  deployed or calibrated in the active RoboDojo execution path.
- `StarVlaAdapter.capabilities.async_inference` is false. The active bridge
  makes synchronous chunk requests; no RTC admission follows from a thread class.
- `QwenPI_v3.predict_action` runs under `torch.inference_mode()` and does not
  forward an action prefix or inference delay into its sampler. Guided RTC needs
  a deliberately differentiable sampling subpath, normalized prefix actions and
  correct time alignment, not merely an additional websocket field. Current
  optimized/custom kernels must be checked for input-gradient support.
- `EventCoherentReuseGate` is an existing integration point, not proof that every
  backend and online task path already implements all required invalidations.

For scale only: at the declared 25 Hz, 16 actions cover 0.64 simulated seconds
and a full 50-action prediction covers 2 seconds. A roughly 19.4-second average
semantic RPC cannot be hidden behind such a queue. Even without a new anomaly,
extending stale open-loop execution to hide it would change the control policy.

### Recommended Sequence

1. **Profile and reduce residency overhead first.** Split CPU allocation, D2H,
   H2D, synchronization, loading, and model compute. Investigate immutable CPU
   weight mirrors and restoration without redundant device-to-host copies.
   Check tied weights, mutable buffers/caches, RNG, exception recovery and CUDA
   graph pointer assumptions. Do not use unsafe tensor replacement as a shortcut.
   A mirror still costs CPU memory and H2D bandwidth; it is not zero-copy.
2. **Evaluate selective low-bit deployment and the actual VLA hot path.** Keep
   action-sensitive modules and initially the vision path at reference precision;
   calibrate candidate backbone layers using representative robotics inputs.
   Test VLM and VLA separately before co-residency. Lower VRAM is valuable if it
   avoids model swaps, even when one model call does not get faster. A failed BF16
   Critic is not made reliable by matching its outputs after quantization: it
   still needs an absolute visual-quality gate. Do not quantize both models at
   once, nor treat a previous Pi05 profile as admitted for QwenPI_v3.
3. **Consider async/RTC only after measuring remaining pauses.** First implement
   action-spec-aware alignment and clock/queue telemetry. Compare synchronous,
   naive async and guided RTC under identical model and physical time budgets.
   Keep the naive path a controlled diagnostic, not a production default.
   RTC is optional if its guidance cost, memory use or reaction delay outweighs
   the benefit. Training-time RTC/VLASH/FutureRTC are separate training decisions.
4. **Add temporal reuse only where cost remains.** Profile vision/backbone versus
   the two/four-step action expert before adopting ActionCache or visual caches.
   Use disjoint cache population and evaluation data. Task memory and numerical
   feature/action caches must remain distinct objects with different validity.

The ordering is our engineering recommendation based on local measurements,
not a ranking of paper quality or a commitment to implement every method.

### Agent-Aware Execution Contract

A useful project contribution to investigate is a common validity contract for
accelerated computation across Agent events, not relabeling RTC or quantization:

- Attach episode, plan/subgoal version, observation time, action time range,
  checkpoint/profile identity and action specification to every queued chunk.
- On a verified replan, invalidate uncommitted old-plan actions and in-flight
  results. Do not erase the physical history of already executed/irrevocably
  committed commands: they remain the alignment context for a replacement chunk.
- Distinguish routine semantic checking from a risk event. A delayed advisory
  response is not permission to continue unsafe cached motion indefinitely.
- On queue exhaustion, uncertain alignment or emergency, use an explicitly
  defined environment/controller fallback; do not blindly repeat a gripper
  command or claim that a position hold is physically safe in every situation.
- Keep planner/critic observations fresh, budget their calls, and reject stale
  decisions. Residency optimization cannot fix hallucinated scene relations.

This is an integration hypothesis needing ablation. Existing software and robotics
runtimes also invalidate stale work; priority and novelty are not established by
this review, and proposed behavior is not represented as already validated.

### Measurement Before A New Benchmark Matrix

Keep ordinary official RoboDojo scores distinct from a latency-aware simulation
protocol. In the latter, the physical environment must advance during inference
with a documented control timestep, queue and fallback. Merely adding `sleep`
while physics is paused measures waiting, not delayed robot control. If the local
simulator cannot sustain real-time wall-clock speed, explicitly simulated delay
in physics steps can test sensitivity, but is not a hardware real-time proof.

Report model-call P50/P95/P99, image-to-action age, queue starvation/hold time,
command-boundary discontinuity, state-tracking error, semantic RPC breakdown,
VRAM/RAM peak, official success/progress and total rollout duration. Also report
simulated time separately from wall time; use both arms and grippers in fidelity
metrics. RTC deliberately changes actions, so judge task quality and continuity,
not bitwise identity with a synchronous trajectory. Weight-residency-only changes
should first pass same-input/same-noise equivalence checks.

Suggested development sequence, not a launched or statistically powered study:
one fixed-input residency test; a small predeclared set of successful-capability
layouts; then independent layouts with measured latency and optional +100/+200 ms
stress. Keep stress separate from nominal results. Choose final sample counts and
non-inferiority margins before running the formal comparison. Do not select
layouts after seeing where the method wins. No fresh model download or large
benchmark is required by this review.

## August 24 Decision (Historical)

SMVE should be retained but demoted from a possible headline method to **L0
static input compaction**. It has a clear contract and real latency evidence,
but removing a permanently padded view is too narrow to represent the complete
VLA efficient-inference contribution.

CARVE Optimize Runtime will instead be developed as a composable stack:

1. deterministic input compaction;
2. optimized model execution backends;
3. training-free temporal reuse;
4. event-coherent reuse invalidation;
5. admitted adaptive compute and action commitment;
6. optional low-bit deployment and co-resident VLM scheduling.

The core CARVE research question is not "can a cache make PI0.5 faster?" It is:

> Can an Agentic runtime safely exploit heterogeneous VLA accelerators while
> invalidating stale computation at semantic, recovery, and physical-risk
> boundaries, and admit the resulting profile using action and closed-loop
> evidence?

## Latest Work And Consequences

| Work | Mechanism | Consequence for CARVE |
|---|---|---|
| [Realtime-VLA](https://github.com/Dexmal/realtime-vla) | simplified PI0/PI0.5 inference graph and custom Triton kernels; the official repository reports 22.1/29.2/38.9 ms on RTX 4090 for one/two/three views | integrate as a backend baseline; stock PyTorch compilation is no longer a sufficient systems baseline |
| [VLA-Cache](https://arxiv.org/abs/2502.02175) | training-free adaptive reuse of temporally stable visual-token KV representations | do not claim generic visual-prefix caching; expose it as an optional reuse backend |
| [ActionCache](https://arxiv.org/abs/2607.06370) | training-free multimodal-keyed action retrieval and zero/few-step refinement | strongest immediate PI0.5 extension; do not duplicate its cache key or warm-start algorithm |
| [Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778) | draft-and-verify speculative flow generation with phase-aware fallback | strong learned acceleration baseline, but it requires a draft path and is not the first no-training integration |
| [ActQuant](https://arxiv.org/abs/2605.24011) | action-guided mixed-precision post-training quantization | generic action-aware VLA quantization novelty is occupied; use it as a replaceable memory backend |
| [QVLA](https://arxiv.org/abs/2602.03782) | action-centric, channel-sensitive VLA quantization | confirms that CARVE's quantization value is admission and deployment composition, not a new generic quantizer |
| [vla.cpp](https://arxiv.org/abs/2606.08094) | portable C++/ggml-class runtime across VLA/action-head families | useful future portability baseline; CARVE should keep model/backend contracts independent of PyTorch |
| [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161) | inference-time chunk length selected from action entropy | adaptive commitment is an occupied method category; CARVE may route admitted profiles using risk and Agentic events but must compare with entropy-only chunking |

## CARVE-Specific Gap

The reviewed reuse methods motivate an integration question: how should
model-local token change, embedding similarity, action entropy or draft checks
interact with the Agent execution lifecycle? This is not evidence that every
existing accelerator lacks lifecycle handling. CARVE observes events that should
invalidate otherwise plausible cached computation:

- task instruction or symbolic subgoal change;
- a new VLM Critic/Planner decision;
- start or completion of physical recovery;
- monitor-detected stall, stale action, slip, contact, or deadline emergency;
- deployment-profile or action-contract change;
- excessive visual/proprioceptive drift or cache age.

The implemented `EventCoherentReuseGate` turns these events into one fail-closed
contract shared by action queues and future visual-prefix/action-warm-start
plugins. This is complementary to VLA-Cache and ActionCache rather than a
renaming of either method.

## Implementation Sequence

### Stage A - implemented now

- `ReuseLayer`: action queue, visual prefix, action warm start;
- `ReuseCandidate`: profile, task/subgoal, planner/recovery epochs, age, and
  optional similarity;
- `ReuseContext`: current semantic and physical evidence;
- `EventCoherentReuseGate`: auditable acceptance or forced refresh;
- controller support for rejecting cached actions and issuing fresh VLA
  inference.

### Stage B - next backend work

1. add Realtime-VLA/Triton behind the existing `BackendPlugin` boundary;
2. preserve CARVE's checkpoint, action-spec, fixed-noise, and fidelity gates;
3. benchmark raw model P50/P95/P99, reaction latency, peak VRAM, and deadline
   misses against eager, compile, and SMVE.

### Stage C - temporal reuse experiment

1. integrate ActionCache as an external action-warm-start backend;
2. use disjoint cache-population and evaluation seeds;
3. compare similarity-only cache admission with CARVE event-coherent admission;
4. inject subgoal changes, recovery transitions, and physical disturbances;
5. report cache hit rate, forced-refresh reason distribution, NFE, action
   fidelity, success/progress, reaction latency, and task-cycle time.

### Stage D - optional deployment capacity

Integrate an existing VLA-specific low-bit method only if the runtime memory
budget requires it. It remains a backend candidate and must pass the same
replay, warm-runtime, and matched closed-loop gates. The existing rejected
W8A16/INT8/NF4 results remain valid negative evidence.

## Thesis-Level Ablation

The efficient-inference experiment should eventually compare:

1. eager BF16;
2. compiled BF16;
3. compiled BF16 + SMVE;
4. optimized Triton backend;
5. temporal reuse with backend-local admission;
6. temporal reuse + CARVE event-coherent invalidation;
7. full CARVE with event-triggered VLM and recovery.

This sequence isolates the value of kernels, input compaction, reuse, and the
Agentic-runtime coupling. It is substantially stronger than presenting SMVE as
a standalone lightweight VLA method.

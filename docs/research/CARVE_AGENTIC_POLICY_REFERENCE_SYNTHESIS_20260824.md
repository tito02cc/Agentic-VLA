# CARVE Agentic Policy Reference Synthesis

Updated: 2026-09-22 (original architecture notes retained)

统一阅读入口：[近期工作、Benchmark 与 RA-L 验证路线](AGENTIC_VLA_RAL_REFERENCE_AND_VALIDATION_PLAN.md)。
该文补充 BATON、Show-Harness、RoboMemArena 及已发表 RA-L 实验参照；
这里只保留框架专题，不重复生成多个执行计划。

Section 0 is the latest literature-based recommendation. Sections 1--9 preserve
earlier design and implementation snapshots, not the current deployment state.
In particular, September 9 used StarVLA; current RoboDojo work uses its matched
PI0.5 checkpoint. This review does not change model defaults, launch experiments,
or declare proposed capabilities implemented. The execution authority remains
`docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md` until its next explicit update.

## 0. 近半年框架研究复核：先解决可执行性，再选择验证平台

### 0.1 范围与证据等级

检索日期：2026-09-22。主要时间窗口：2026-03-22 至 2026-09-22，兼顾窗口外但
直接相关的基础工作及其修订。不是穷尽全部新论文，也不是复现完成报告。
本轮读取重点论文的方法/实验段落、官方仓库说明，并对照本地接口；仅阅读摘要的
候选单独标记。arXiv 存在不等于经过同行评审，作者报告的提升不等于我们的收益。
代码公开也不等于适配了当前 checkpoint、传感器、动作接口与单卡预算。

### 0.2 核心参考矩阵

| 工作与可核查日期 | 本轮核查层次 | 可以借鉴 | 不能直接照搬的条件 |
|---|---|---|---|
| [What Matters in Orchestrating Robot Policies](https://arxiv.org/html/2606.10267v2)，6月初稿、9月6日修订 | 方法4.2--4.6、实验设计 | 将 Planner、VLA 指令可控性、切换条件、观测表达和记忆拆开比较 | 主实验的成功检测使用模拟器特权状态；不能将其效果当作我们的 RGB Critic 能达到的效果 |
| [Harness VLA](https://arxiv.org/abs/2607.08448)，7月9日初稿 | 本轮核对论文元信息，结合既有 RPent 代码审计 | VLA 接触技能与解析工具组合；从经验学习技能适用范围 | 不能把重试同一整任务指令等同于技能重定位；深度/控制器/额外工具须公平披露 |
| [RoboHarness: Memory-Augmented Policy Harness](https://arxiv.org/html/2603.24060v3)，3月25日初稿、8月12日修订 | 方法与实验、官方仓库入口 | 成功和失败双库、诊断后选工具、离线整理经验 | 使用 Qwen3-VL-32B；图像改写/去干扰物会改变 VLA 输入，反向轨迹恢复不能默认安全；不能把摊销每步耗时当单次 VLM 延迟 |
| [RoboHarness: Memory-Driven Orchestration](https://arxiv.org/html/2607.18060v2)，7月20日初稿 | 方法与问题定义 | 根据已知能力规划；关注前一技能终态是否适合下一技能 | 是另一篇同名论文，讨论异构策略与 Memory Bridge；不意味着我们应立刻引入多个新动作模型 |
| [EmbodiedSkills](https://arxiv.org/html/2609.01281v1)，9月1日 | 接口设计与结果表说明 | 执行前查前提、执行后查结果，统一可替换技能接口 | 依赖任务适配的底层策略；不能将其高成功率简单解读为给任意冻结模型加 Agent 的收益 |
| [EMERGE-Policy](https://arxiv.org/html/2608.29896v1)，8月预印本 | 方法、实验观测和预算说明 | 局部恢复后返回原计划，职责分离 | LIBERO 系列增加五个 Agent 相机并延长步数；本项目不采用这些协议改变 |
| [VoLoAgent](https://github.com/NVlabs/VoLoAgent)，本轮核对当前仓库，不推断首发日期 | README 模式和工具依赖 | 高层可用的实际抓放工具、执行反馈 | tool_chain 模式绕过 VLA；抓放管线用深度、感知/抓取服务和运动规划，不能当作原 RGB-only VLA 外挂的同条件证据 |
| [ENPIRE](https://arxiv.org/abs/2606.19980)，6月18日 | 摘要与框架概览，未做代码复现 | reset-execute-verify-improve 的物理反馈循环 | 涉及策略训练、场景复位与真实交互预算；属于后期自进化，不是现有记忆检索的别名 |
| [RoboClaw](https://arxiv.org/abs/2603.11558)，3月12日，窗口外补充 | 本轮摘要核对 | 数据采集、恢复策略、训练与执行统一组织 | 包含学习到的正向/逆向行为；不是本轮免训练的改法，也不要混淆同名软件仓库 |
| [Sci-VLA](https://arxiv.org/abs/2602.09430)，2月10日，窗口外基础参考 | 本轮摘要核对 | 原子技能间缺失的过渡动作需要单独处理 | 不将领域专用动作代码默认移植到当前机器人 |

两篇 RoboHarness 必须使用标题与 arXiv ID 区分：2603.24060 和 2607.18060；
二者都不是 Harness VLA（2607.08448）。引用、结果和实现不能混用。

### 0.3 这轮调研真正改变的判断

1. **不能再把“模块齐全”称为“框架效果已成立”。** 软件接口、真实模型通信、
   有效技能控制、闭环性能提升是四个不同层级。我们已有不少前两层组件，但当前
   RoboDojo 整理桌面尚未证实后两层。
2. **规划必须受底层能力约束。** 先确定 checkpoint 支持整任务、语言子目标、
   grounded 目标还是动作提案审核；不支持的控制模式不暴露给 Planner。
   “同样是 PI0.5”不代表 RoboDojo、LIBERO 与 GroundSG 权重的可控性相同。
3. **记忆应改变下一次决策，而不是只增加上下文。** 检索结果必须关联技能适用
   条件、失败证据和避免重复的措施。模型解释失败只是归因假设，不等于因果证明。
4. **核验与调度本身也是研究变量。** 过早确认会跳步，过晚确认会重复动作；
   不能直接从别人的 oracle 成功检测结论推出本地小模型 Critic 足够可靠。
5. **学习其他论文，不等于采用其所有权限。** 原相机、原场景与配对预算保持不变。
   人工指定正确子目标只能作为底层能力诊断，不能混进自主 Agent 主结果。

### 0.4 软件 Agent 思想的物理化迁移

[Anthropic Managed Agents，2026-04-08](https://www.anthropic.com/engineering/managed-agents)
将持久会话记录、Agent 控制循环与执行环境分离。这是工程参考，不是机器人有效性
实验证据。我们采用以下对应关系，不为使用框架名而引入新依赖：

| 软件 Agent 做法 | 机器人中的对应实现 | 物理世界限制 |
|---|---|---|
| 工具调用及返回值 | 明确的技能参数、实际执行回执、新观测 | RPC 成功不等于物体已放好 |
| 计划与任务清单 | 未完成子目标、依赖、当前技能、恢复返回点 | 不根据模型一句“完成”推进计划 |
| 持久会话和摘要 | 不可变执行记录 + 可更新、带来源的任务记忆 | 不可用摘要覆盖相反证据或已发生动作 |
| 回滚/重试 | 停止后重新观察，执行另一个准入的恢复技能 | 不能像 git reset 一样恢复物理世界；抓取动作不可假定可逆 |
| 自动测试与反思 | 执行后语义核验、失败分类、独立回归 | 人工或模拟器评分不能偷偷成为在线感知输入 |

### 0.5 本地已有基础与下一步差距

本轮只做静态代码核对，没有重新运行测试或机器人回合。

| 组件 | 已确认的本地基础 | 待验证/补齐，不应重造 |
|---|---|---|
| 工具执行契约 | `agentic_vla/toolchain/runtime.py` 的 CanonicalToolRuntime、执行报告和预算 | 当前 benchmark 实际绑定了哪些可执行技能；工具是否真的能改变失败结果 |
| 程序记忆 | `agentic_vla/toolchain/memory.py` 的 VerifiedProcedureCompiler | 来源正确不保证视觉核验正确；缺少完整成功的任务不能升级成成功程序 |
| 冻结记忆评估 | `agentic_vla/benchmarks/robodojo_memory.py` 核对哈希、开发/评测划分 | 实际有效经验覆盖；当前检索入口主要按指令取前两条，不等于状态条件化能力检索已成立 |
| 对象与失败记录 | `toolchain/object_memory.py`、`runtime/recovery.py` 中已有相关结构 | 检查实际 Planner 是否消费其证据；隔离成功经验与失败禁忌对选择的作用 |
| 动作提案 | `benchmarks/robodojo_action_proposal.py` 的提案标识与 FK 展示 | 机器人 FK 不是对象未来状态预测；只读/微小运动通过不代表抓放纠错准入 |
| 异步支持 | `runtime/prefetch.py` 的上下文、失效标记和动作契约检查 | 不能由类存在推出当前 PI0.5 在线部署了 RTC 或计划切换时所有结果均正确失效 |

### 0.6 建议的收敛式改进，不再无边界堆模块

**P0：能力条件化的执行接口。** 为每个实际部署 adapter 记录支持的指令形式、
可用技能、已测前置条件、退出条件及验证来源。规划器只能选择已确认接口。
先用开发场景做原子动作、改目标、改动作顺序三类检查；失败时归类为动作能力或
接口问题，不继续用更复杂提示词掩盖。未准入动作只允许只读提案，不自动扩权。

**P1：证据驱动的记忆与恢复。** 沿用现有存储，区分目标、观测事实、执行结果与
模型假设；检索时同时给出相关成功经验和失败禁忌，并带机器人/模型/技能适用范围。
当前回合记忆可更新，跨回合经验在正式评测前冻结。未知或过期的物体状态要求
重新观察，不把“未看到”写成“不存在”。反思输出必须落到可执行修正或明确放弃。

**P2：任务边界核验和可测的干预调度。** 比较固定周期与子目标/异常触发，明确
已完成、未完成、不可判定三类结果。不重复查询同一帧期待模型改口；只有新观察、
新工具结果或足够理由才进行额外判断。先做小型 held-out 决策集，再开放实际干预。
记录误确认、漏检、弃权、干预后改善/损害及额外耗时，不仅记录 JSON 合法率。

**P3：高效推理与 Agent 状态一致。** 保留已验证的基础加速；量化、缓存、异步
逐项隔离。任务/子目标变化时拒绝旧计划未提交的动作与过期模型回复，但不删除
已执行动作历史。先复核当前部署链路，不把历史 LIBERO 的 P95 拼到 RoboDojo。
具体候选与训练要求见 `CARVE_EFFICIENT_INFERENCE_UPDATE_20260824.md` 最新小节。

上述优先级是待实施/验证建议，不表示本轮已完成。每项最多先验证一个候选；
若新增机制不能改变失败原因，则停止该机制，保留负结果，不无限追加阈值。

### 0.7 框架与实验必须一起冻结

benchmark 筛选参考 `CURRENT_VLA_BENCHMARK_AND_MODEL_SURVEY_20260828.md` 第 0 节。
平台名称不是主贡献。优先选能够执行目标技能、同时需要决策或记忆的任务族，
不是基于哪个任务恰好 Agent 赢来筛选。

| 对照 | 固定项 | 回答的问题 |
|---|---|---|
| 原生策略/官方运行链路 vs 朴素层次 Agent | 权重、传感器、初始状态、执行预算 | 加高层本身有无帮助；若官方链路本身含 Planner 必须说明 |
| 朴素 Agent vs 能力匹配后的 Harness | 相同实际工具、模型与预算 | 工具选择/切换机制是否有效，不能混入更强执行器 |
| Harness 无检索 vs 成功检索 vs 成功+失败检索 | 同一工具能力和推理配置、冻结经验来源 | 记忆是否降低重复失败，是否在未调参状态泛化 |
| 同一 Harness 默认推理 vs 一项优化 | 相同任务协议，允许算法定义内的动作变化 | 延迟/显存/调用下降是否以任务质量下降为代价 |

先做能回答问题的小规模试验，再扩样本；开发诊断和正式统计分开。
使用配对初始状态与独立测试种子，报告成功率、进度、损害/救回次数、成本和置信区间。
RoboMME 是优先复核候选，VLA-Arena 是唯一优先新增候选；不是立即同时开跑多个平台。
RoboDojo 暂留压力测试，先审计版本和技能可控性，不继续对整理桌面单布局无限改动。

### 0.8 贡献表述与未来边界

不再把 VLM+VLA、通用工具调用、双记忆、事件触发或量化本身写成我们的独有发明。
可检验的研究问题是：在固定传感器、动作权限与资源预算下，能力匹配、证据记忆
及推理调度能否形成更好的任务质量--端到端成本折中？目前是研究假设，不是新颖性
或效果已获证明的结论。必须以同条件消融支持，不能凭组合模块就宣称 RA-L 创新。

自进化分两级：离线更新经验规则，以及采集数据后真正更新模型参数。前者也需要
独立测试防止记住评测答案；后者参考 ENPIRE/RoboClaw，后期再做，不纳入当前短期
验收。不新增 WAM、MoE、多 VLA 集群或自动微调闭环作为眼前的必做任务。

## 1. Purpose

This note fixes the graduation-oriented architecture after reviewing the papers
under `paper/Agentic Policy/`. CARVE does not reproduce one paper wholesale. It
adopts the reusable system ideas that reduce training cost and keeps each model
or deterministic component inside an explicit authority boundary.

## 2. What CARVE Adopts

| Reference | Useful mechanism | CARVE decision |
|---|---|---|
| Harness VLA | Frozen VLA as a retryable primitive, fixed tool vocabulary, coding-agent planner, task/global memory, no-training evaluation | Primary harness reference |
| Agentic Robot | Planner-executor-verifier separation and temporal VLM verification | Use semantic checks at task, primitive, and anomaly boundaries |
| RoboClaw | VLM meta-controller, discoverable tool ecosystem, task and working memory | Keep provider-neutral planner and typed tool registry |
| Sci-VLA | Agent intervention only at execution failures or state gaps | Use as the efficient event-triggered mode |
| MEM | Dense short-term visual context plus compressed long-term semantic memory | Keep bounded execution context and verified procedural/failure memory |
| VLA^2 | External perception/retrieval tools and a verifier | Retain tool extensibility; avoid its training-heavy detector/verifier path initially |
| Agentic-VLA | Critic-guided online adaptation and experience transfer | Defer because it changes weights and requires substantially more training |

## 3. Primitive Definition

A primitive is a bounded robot operation with a typed input, an execution
authority, and a terminal status. Examples are `vla_act`, `move_to`,
`grasp_stage_place`, and `cartesian_retract_lift_reobserve`. A primitive returns
control to the harness as `succeeded`, `failed`, `timeout`, or `interrupted`.
It never exposes raw actions to the high-level planner.

`PrimitiveOutcome` records the expected and observed outcomes, whether semantic
verification is required, and non-privileged metadata. This makes a primitive
boundary an auditable place to call a VLM without putting that VLM inside the
real-time controller.

## 4. Multi-Rate Architecture

```text
Task + RGB/RGB-D + robot state + verified memory
                         |
                         v
              Replaceable VLM Planner/Critic
       GPT / Claude / local Qwen-VL / scripted baseline
                         |
                  typed intent or tool call
                         |
                         v
                  CARVE Agentic Harness
       schema | capability | budget | safe-boundary gate
              /                  |                 \
       frozen VLA primitive  registered skill   safe hold/stop
              \                  |                 /
                         environment
                             |
                 observation + primitive outcome
                             |
             Fast Execution Guard + semantic scheduler
```

The high-frequency `ExecutionRiskMonitor` is presented as the Fast Execution
Guard. It is intentionally deterministic and detects only deployable hard
signals such as lack of response, stale actions, uncertainty, and deadline
risk. It is not a semantic judge and is not claimed as the Agent intelligence.

The low-frequency VLM handles task decomposition, semantic verification,
failure diagnosis, and selection among registered interventions. It cannot emit
joint targets, torques, trajectories, or unregistered skills.

## 5. Planner Schedules

CARVE supports three controlled schedules:

1. `event_only`: call the VLM only after an abnormal primitive or controller
   escalation. This is the lowest-cost Sci-VLA-like mode.
2. `selective`: call at task start, abnormal primitives, and boundaries explicitly
   marked as requiring semantic verification. This is the recommended CARVE mode.
3. `every_primitive`: call after every primitive boundary. This is the closest
   Harness VLA-style comparison and is an efficiency baseline, not the default.

The decisive comparison is `every_primitive` versus `selective` under the same
planner, VLA/skill backend, initial states, and task budget.

## 6. Provider Boundary

The harness consumes only the provider-neutral `PlannerCallable` contract.
Concrete adapters currently include:

- OpenAI-compatible Chat Completions for GPT-compatible endpoints and local
  Qwen-VL services;
- Anthropic Messages for Claude;
- arbitrary scripted or test planners through the same callable contract.

Every experiment must pin provider, model, endpoint protocol, prompt/schema
version, token budget, timeout, and fallback behavior in its run manifest.

## 7. Historical Experiment Order (Superseded)

The following August sequence is retained for context, not a current launch list.
Use Section 9 and the RoboDojo execution plan for new work.

1. CPU conformance tests for primitive boundaries, provider construction,
   permission gates, timeouts, and stale result rejection.
2. MuJoCo Panda dynamic-kitting K0 and K1 paired runs:
   `every_primitive` versus `selective`, with real Qwen-VL calls and videos.
3. K3 external scene displacement to test re-observation, memory retention, and
   replanning without simulator-state access.
4. LIBERO-Pro small paired gate with frozen PI0.5 after the generic harness and
   planner schedule are stable.
5. Optional GPT or Claude replay on the same recorded observation bundles to
   test provider portability without multiplying robot rollouts.

## 8. Claims Boundary

Passing interface tests validates only the exercised software contracts. A VLM service smoke
establishes interoperability. MuJoCo or LIBERO-Pro paired episodes establish
closed-loop behavior. None of these alone establishes benchmark-wide success
improvement. Report planner-call reduction, semantic decision agreement,
primitive outcome, task success, wall time, VLM latency, and video evidence
separately.

## 9. September Review: Complete The Feedback Loop Before Adding Agent Roles

### 9.1 Source-Based Design Decisions

Primary sources checked on September 9, 2026. These are references, not our
performance evidence; the paper and a moving implementation may differ.

| Source | Lesson to adopt | Boundary for our implementation |
|---|---|---|
| [Harness VLA v3](https://arxiv.org/html/2607.08448v3) | Compose a fixed VLA primitive with executable analytic tools, inspecting returned observations and re-grounding stored procedures | Merely producing a subgoal prompt or repeating a failed VLA request does not reproduce this mechanism. RoboDojo physical-tool admission is still pending |
| [RPent architecture](https://rpent.readthedocs.io/en/latest/rst_source/development/architecture.html) | A provider-neutral tool loop returns text and image observations to the agent | Prefer small typed tools over unrestricted generated robot-control code; observe-only tools cannot grant motion authority |
| [RPent memory](https://rpent.readthedocs.io/en/latest/rst_source/development/memory.html) | Separate exploration-time memory construction from read-only evaluation, with robot/task scope | Freeze a memory snapshot before held-out evaluation. Current RPent documentation limits the exploration mode to LIBERO; do not assume a RoboDojo implementation exists |
| [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Retrieve context when needed, retain structured notes, expose clear and non-overlapping tools | Engineering guidance, not a robotics benchmark result. Preserve failures and unresolved evidence when summarizing; avoid feeding an ever-growing transcript |
| [MemoryVLA](https://arxiv.org/abs/2508.19236) | Distinguish perceptual and semantic memory and use temporal context for manipulation | Its learned memory-conditioned action path is not a training-free replacement for our external memory; use as a design reference, not an immediate dependency |

These references support the overall Agent-tool-memory direction. They do not
establish that our present local model perceives the scene well enough, that its
interventions improve success, or that more agent roles would help.

### 9.2 Gaps Identified From Our Evidence

1. **Observation before planning depth.** The fixed 16-frame development probe
   obtained only 10/18 positive confirmations with the new stage verifier, with
   failures concentrated in upper support relations and inappropriate certainty
   under occlusion. Both compared methods failed admission. A more elaborate
   plan cannot compensate for missing visual relations.
2. **Executable tools, not just tool names.** Add and validate one deployment-
   supported intervention at a time. Its observation input, kinematic limits,
   physical execution, timeout, postcondition and resulting observation must
   actually work on RoboDojo. Do not substitute simulator object poses or reset
   utilities for perception-conditioned robot recovery.
3. **Memory of evidence, not memory of wishes.** The last successful C3 episode
   ended before independent final-stage verification; it cannot become a fully
   verified procedure. Late stage confirmation must not make the robot redo a
   physically completed stage. Store uncertainty and provenance explicitly.
4. **Cost of the complete loop.** Five semantic RPCs took 97.18 seconds, of which
   generation took 11.56 seconds. Residency/transfer overhead is currently more
   urgent than reducing a two-step action sampler by one more step. These are
   development measurements, not a hardware-independent bottleneck ranking.

### 9.3 Memory And Tool Contracts

Use the existing toolchain and procedure-memory compiler, rather than creating
a second planner or database. Keep three kinds of stored information separate:

| Storage | What it means | Validity and admission |
|---|---|---|
| Episode observations and working notes | What was observed, what is uncertain, which actions have actually completed | Frame/episode/time identity, bounded context, source hashes, reset and expiry; a plan is never an observation |
| Verified procedures and failure rules | Which sequence worked under which preconditions, with supporting outcomes | Independent verification, source episode and model/profile identity, no simulator-ground-truth fields; retrieve against the new scene rather than replay old coordinates |
| Numerical feature/action caches | Reusable model computation, not semantic knowledge | Model, normalization, observation age, task/plan identity and event invalidation; never promoted into factual memory |

For an eventual memory experiment, build from designated exploration episodes,
freeze the snapshot, then compare memory-off/on on disjoint layouts with the
same model and execution budget. Log irrelevant retrievals, contradicted memory,
and false recovery as well as successful reuse. Persistent memory is not allowed
to accumulate held-out solutions silently during a nominally frozen evaluation.

### 9.4 Implemented In This Review

**Optional read-only visual inspection** in
`agentic_vla/toolchain/inspection.py`, registered through `CanonicalToolRuntime`
only when explicitly supplied:

- The host captures real uint8 RGB under an opaque frame handle. The model
  selects integer pixel coordinates, not a file path, uploaded image or object pose.
- The host returns the unmodified global frame and its native-resolution crop;
  JSON logs carry episode, camera, time, region and pixel hashes, not pixel arrays.
- The receipt explicitly says that the crop is the same observation, not an
  independent second view. No crop alone proves support, success or stability.
- Frame count, resolution, inspection count and control-step age are bounded.
  Old episode handles, evicted frames and stale payload delivery are rejected.
- The deployment must reset both the evidence store and registry. Image delivery
  requires `resolve_images()` at the host/provider boundary; registering a JSON
  tool does not automatically attach images to a VLM request.

Default seven-tool behavior is unchanged. This is not yet an online RoboDojo
VLM-selected crop path or a visual localization model. A fixed host crop in a
recorded-image replay tests data transport only. No motion or persistent-memory
permissions were enabled, and no semantic improvement is claimed.

**Action-contract-aware prefix checking** in
`agentic_vla/runtime/prefetch.py`:

- The old Cartesian-delta verifier now rejects non-7-D data instead of silently
  ignoring the second arm. Non-finite thresholds cannot disable its checks.
- A separate joint-position verifier checks all joints and both explicitly
  identified grippers, using maximum joint error, joint RMS and gripper magnitude.
  It refuses invalid shapes, non-finite metrics and the wrong action representation.
- Thresholds must be calibrated in the deployed action units. Unit-test numbers
  are not approved robot limits. Time alignment is the caller's responsibility.

This supplies a missing check for a future async path. It does not implement
prefix-conditioned RTC sampling, collision checking or a hard-real-time controller;
the current StarVLA bridge remains synchronous.

### 9.5 Controlled Next Steps

1. Bind inspection to the provider's multimodal result delivery, in shadow mode.
   Fix a small new visual-relation evaluation set before running it, with occlusion,
   misleading similar structures and contradictory observations. Keep the existing
   failed probe unchanged as development material. Evaluate full-frame versus
   full-frame-plus-region with the same model and explicit image/token budgets;
   count false confirmation and abstention as well as recall and latency.
2. In parallel, profile residency and qualify one selective-precision or transfer-
   reuse candidate at a time. Follow the inference note; do not equate lower VRAM
   with faster or more accurate inference. If the BF16 visual model still fails
   absolute quality gates, consider a separately evaluated perception/model upgrade
   rather than optimizing an unusable semantic baseline.
3. Only after semantic qualification, admit one physical intervention and then
   verified cross-episode memory. Revalidate transferred procedures on a disjoint
   layout; success under one known layout is not a transferable skill guarantee.
4. Freeze the candidate before paired real simulation. Separate Agent benefit,
   memory benefit and Runtime benefit. Preserve all failures, budgets and hashes;
   no new benchmark or full training campaign is necessary for this sequence.

Do not add MoE routing, autonomous multi-agent fine-tuning or a new framework
dependency merely for novelty. They remain future options if a measured failure
cannot be addressed with the existing model/tool contracts. RTC remains optional.

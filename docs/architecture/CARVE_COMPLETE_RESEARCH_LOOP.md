# CARVE-VLA 完整研究闭环

更新时间：2026-08-24

> **2026-09-11框架复审**：当前实现/接入/效果三个层级及面试说明见
> `docs/reports/FRAMEWORK_READINESS_AND_INTERVIEW_GUIDE.md`。补充了真实输入重放、
> VLM切换保真及驻留优化证据；本次修复语义证据连续性，reset清单为36项。
> 可选LangGraph应用层不在现有RoboDojo默认执行路径上，不能把两条入口画成一个
> 已经全链路验证的部署。下文是总体设计，尤其多速率不等于当前单卡并发或硬实时。

> **2026-09-10实施状态补充**：下文为系统设计总图，不表示所有模块已在同一个
> 机器人配置中启用或验证。当前RoboDojo的PI-v3+本地Qwen3-VL4B实际控制路径为
> 监测/语义检查 → 受限Planner建议 → VLA能力路由 → 有界恢复计算 → 动作执行
> → 新观察核验 → 过程日志与回合重置。9.21完成8个官方回合及24段视频，545项
> CPU回归、408项证据核查通过；组合组两次真实恢复均完成动作后核验，但结果仍
> 为inconclusive，不能自动写成成功技能。VLA保持task-only，不执行自由子任务文本。
>
> 对象参考/跟踪已具备可选只读链路，身份准确性尚未准入；长期记忆主动收益、
> 开放技能、硬实时和稳定成功率提升仍未证明。flow2候选本轮未保持任务能力，
> 不作为默认部署配置。最新技术细节与限制见
> `artifacts/robodojo/recovery_lifecycle_20260910/README.md`；不要只凭下文架构图
> 宣称全部设计能力已获得实际机器人验证。

## 1. 统一定位

本项目不是将 Agentic RAG-VLM、Harness 和推理优化机械拼接，而是围绕同一个
具身智能问题形成三个连续层次：

1. **Agentic RAG-VLM：知识与推理基础。** 解决机器人如何利用功能可供性、
   空间约束、失败诊断和历史经验理解操作任务。
2. **CARVE Agentic Harness：闭环执行系统。** 将上述结构化知识接到冻结 VLA
   外部，使高层 VLM、低层 VLA、物理技能、Monitor、Memory 和 Recovery 在
   明确权限下协同工作。
3. **CARVE Optimize Runtime：高效推理与部署。** 优化 Agentic 调用链中的
   VLM/VLA 推理，在时延、显存、动作保真和闭环结果约束下选择可部署方案。

统一研究命题是：

> 面向具身智能操作，如何利用结构化多模态知识监督冻结 VLA 的闭环执行，
> 并通过事件一致、证据门控的运行时优化，使智能体系统同时具备任务理解、
> 失败恢复和高效推理能力？

这条主线对应毕业论文方向：**面向具身智能的多模态智能体系统与高效推理
研究**。

## 2. 名称与成果边界

| 名称 | 在本项目中的身份 | 可以继承的内容 | 不能混淆的内容 |
|---|---|---|---|
| Agentic RAG-VLM | 已有研究成果与知识层基础 | HAA-RAG、场景图约束、失败诊断、自反思、三级 retry、情景记忆 | 原抓取参数修正实验不能改写成 CARVE-VLA 实验 |
| CARVE-VLA | 当前自主研发的完整具身智能体系统 | Harness、VLM/VLA 边界、工具链、恢复、记忆、Optimize Runtime 和实验协议 | 不能把尚未运行的新 benchmark 或新缓存后端写成已验证结果 |
| Harness VLA / RPent | 外部参考工作与参考实现 | 冻结 VLA primitive、编码智能体工具生态、服务拓扑、任务过程记忆、recipe/trace、无训练评测思想 | 不是 CARVE 的已有成果，官方结果不能作为 CARVE 结果 |
| PI0.5 / OpenVLA 等 | 可替换的动作策略后端 | 视觉语言条件下的低层动作生成 | 不负责 CARVE 的权限、恢复、记忆与部署 admission |

因此，“CARVE-VLA + Harness VLA”更准确的表达是：

> **CARVE-VLA Agentic Harness 吸收 Harness VLA/RPent 的可复用工具链与服务
> 生态思想，并结合 Agentic RAG-VLM 的知识机制、CARVE 的执行安全边界和
> Optimize Runtime，形成自主的具身智能闭环。**

RPent 保存在 `third_party/RPent`，仅作为只读参考。其 Git 历史和 Apache-2.0
许可证与 CARVE 源码保持独立。

## 3. 总体架构

```text
任务指令 + 多相机 RGB + proprioception
                    |
                    v
+-------------------------------------------------------------+
| A. Knowledge and Reasoning Plane                            |
| VLM Critic/Planner                                           |
| HAA-RAG | Scene Graph | Procedural/Failure/Recovery Memory   |
| output: typed subgoal, constraint, expected outcome, intent  |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| B. CARVE Agentic Harness                                    |
| Observe -> Monitor -> Decide -> Tool/Skill -> Verify         |
| typed intents: continue | vla_act | run_skill | safe_stop    |
| budget | safe boundary | authority | retry | trace           |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| C. CARVE Optimize Runtime                                   |
| profile admission | backend | reuse gate | deadline | fallback|
| compile | SMVE | optimized kernels | temporal reuse | quant  |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| D. Physical Execution Plane                                 |
| frozen VLA -> action chunk -> simulator/robot -> observation |
| bounded physical skill -> verification -> memory update      |
+-----------------------------+-------------------------------+
                              |
                 success / new event / failure
                              +---------------------> A / B
```

四个平面不是四个都在同一频率运行：

- **高频：**确定性的 `ExecutionRiskMonitor` 与机器人控制，处理动作年龄、
  状态响应、视觉变化和 deadline；
- **中频：**通过 admitted profile 执行 VLA，生成动作 chunk；
- **低频：**仅在任务开始、语义歧义、重复失败或恢复边界触发 VLM
  Critic/Planner、检索和复杂推理。

这种多速率设计避免让较慢的 VLM 成为每个控制周期的阻塞模块。

## 4. 第一部分：Agentic RAG-VLM

Agentic RAG-VLM 不只是背景论文，而是 CARVE 的知识与反思来源。

### 4.1 HAA-RAG

HAA-RAG 全称为 **Hierarchical Affordance-Aware Retrieval-Augmented
Generation**。原方法按照类别匹配、功能可供性评分和视觉重排序逐层检索抓取
经验，描述物体类型、材料、易碎性和可抓取区域。CARVE 不直接复制其抓取
参数输出，而将检索结果转换为：

- 当前任务阶段或候选子目标；
- 操作约束和注意事项；
- 可调用的注册物理技能；
- 过往失败、恢复方式及验证结果。

### 4.2 Scene Graph Constraint Reasoner

原工作利用场景图表达 content、collision、support 和 occlusion 等空间关系，
再修正抓取方向、力度和高度。CARVE 中的场景图保持为**定性语义约束层**：

- 为 VLM Planner 提供对象关系和任务阶段上下文；
- 为 `run_skill` 和恢复策略提供约束；
- 不向部署控制器泄露仿真器物体真值或直接生成关节动作。

### 4.3 Self-Reflection、Retry 与 Memory

原工作包含 14 类失败诊断和三级恢复：参数调整、方法切换、完整重规划。
CARVE 将其一般化为：

1. 重新调用 VLA 获取新动作；
2. 选择注册且有界的物理恢复技能；
3. 触发低频 VLM Planner 重新解释子目标；
4. 无可靠恢复时进入 `SAFE_HOLD` 或 `SAFE_STOP`；
5. 仅在结果得到部署可用证据验证后写入 Failure、Recovery 或 Procedural
   Memory。

知识层只能影响意图、约束和技能选择，不能绕过 VLA/技能边界直接下发任意
动作。

## 5. 第二部分：CARVE Agentic Harness

### 5.1 模型职责

| 模块 | 主要职责 | 明确不负责的内容 |
|---|---|---|
| VLM Critic/Planner | 低频语义判断、子目标分解、失败解释、选择 typed intent | 不逐控制周期运行，不输出原始轨迹或力矩 |
| ExecutionRiskMonitor | 高频、低成本地判断执行响应、动作陈旧、风险和 deadline | 不承担开放词汇语义理解 |
| VLA | 根据图像、语言和状态生成低层动作 chunk | 不拥有重试预算、物理恢复权限和 memory 生命周期 |
| Physical Skill | 完成 retract、lift、release、reobserve 等有界 primitive | 不执行未注册的自由动作 |
| Harness Controller | 协调权限、状态机、预算、安全边界和工具调用 | 不替代 VLM 的语义能力或 VLA 的动作生成能力 |

### 5.2 工具生态

CARVE 当前稳定的工具词汇为：

| Tool | 功能 | 类型 |
|---|---|---|
| `observe` | 获取可部署观测和 Monitor 证据 | 只读 |
| `retrieve_memory` | 检索 HAA、过程与失败恢复知识 | 只读 |
| `vla_act` | 请求一个 admitted VLA 动作 chunk | 物理安全边界 |
| `run_skill` | 执行一个注册的有界 physical primitive | 物理安全边界与预算 |
| `verify` | 验证预期结果 | 只读 |
| `safe_hold` | 进入可心跳、可超时的静止保持状态 | 生命周期 |
| `finish` | 结束 episode 并保存可审计摘要 | 生命周期 |

这里借鉴了 Harness VLA/RPent 的“编码智能体持续控制工具链”思想，但增加：

- typed intent 与 tool schema；
- 物理工具 allowlist、预算和 safe-boundary 权限；
- episode/timestep 隔离和 single active operation；
- evaluator 真值与部署信息隔离；
- profile receipt、完整 trace 与 fail-closed 行为。

### 5.3 Harness 状态机

```text
EXECUTE_FAST
    | normal                     | event / uncertainty
    v                            v
continue cached/admitted     VERIFY
actions                          |
                   +------------+-------------+
                   |                          |
              recoverable                 semantic ambiguity
                   v                          v
                RECOVER              PLAN_AT_SAFE_BOUNDARY
                   |                          |
                   +------------+-------------+
                                v
                 fresh VLA inference / SAFE_HOLD / STOP
```

关键点是“retry”不再表示无限重复调用模型。每次 retry、physical recovery 和
planner escalation 都有独立预算、触发原因、验证条件和终止路径。

## 6. 第三部分：CARVE Optimize Runtime

Agentic 系统会引入 VLM 判断、VLA 重规划、恢复后再推理和验证等额外调用。
因此高效推理不是独立附加点，而是让 Agentic Harness 可部署的必要条件。

### 6.1 六层优化栈

| 层级 | 内容 | 当前状态 |
|---|---|---|
| L0 输入压缩 | SMVE 删除适配器保证为 padding 的相机 view | 已实现并通过两种 PI0.5 adapter gate |
| L1 执行后端 | eager、`torch.compile`、后续 Triton/C++ backend | eager/compile 已实现；Realtime-VLA backend 待接入 |
| L2 时间复用 | 动作队列、视觉前缀缓存、动作 warm start | 动作队列与统一 reuse gate 已实现；缓存 backend 待接入 |
| L3 自适应计算 | flow steps、commit horizon、刷新策略 | 静态 2-step profile 已验证；动态策略尚未形成正式结论 |
| L4 模型压缩 | VLA 专用混合精度与低比特 backend | 通用 W8A16/INT8/NF4 已实验但未通过完整 admission |
| L5 多模型调度 | event-triggered VLM 与 VLA 共卡调度 | contention 已测量；Planner profile 仍需正式 admission |

SMVE 被保留为确定性 L0 优化，不再单独承担“通用 VLA 轻量化方法”的叙事。

### 6.2 Event-Coherent Reuse

视觉 token cache、动作 cache 和 warm start 已有相关研究，因此 CARVE 不应
重新包装一个通用 cache。CARVE 的独立问题是：Agentic 系统发生语义或物理
事件时，过去的缓存是否仍然有效。

`EventCoherentReuseGate` 在以下情况下强制 fresh inference：

- instruction、subgoal 或 deployment profile 改变；
- VLM Planner 或 Recovery epoch 改变；
- Monitor 产生 stall、contact、stale action 等事件；
- 风险升高，视觉/本体状态变化超过阈值；
- cache 超龄或 warm-start similarity 不足。

缓存 backend 负责“如何复用”，CARVE 负责“当前是否有权复用”。这使
Optimize Runtime 与 Agentic Harness 形成算法和系统上的真实耦合。

### 6.3 Profile Admission

任何优化不能仅凭延迟或显存下降进入部署。CARVE 要求同一 checkpoint、
adapter、observation contract、flow steps 和 action horizon 下依次通过：

1. replay action fidelity；
2. warm P50/P95/P99、deadline miss 和 peak VRAM；
3. matched closed-loop outcome；
4. Agentic recovery path non-inferiority；
5. profile、hardware 和 fallback manifest 绑定。

这解释了为什么已有 W8A16、OpenVLA INT8/NF4 和异步 prefetch 可以降低某些
成本，却仍被 CARVE 拒绝为正式部署 profile。

## 7. 端到端闭环

一次完整 episode 按以下顺序运行：

1. **感知：**读取 RGB、proprioception、instruction 和历史动作；
2. **知识准备：**HAA-RAG、Scene Graph 和 Memory 提供结构化上下文；
3. **高层判断：**VLM Planner 在触发条件满足时产生 typed subgoal/intent；
4. **权限检查：**Harness 检查预算、safe boundary、工具 schema 和 episode；
5. **计算决策：**Optimize Runtime 选择 admitted backend/profile，并判断缓存
   是否需要失效；
6. **动作生成：**VLA 或注册 physical skill 生成有界动作；
7. **执行与监测：**环境执行动作，Monitor 高频检查响应和 deadline；
8. **验证与恢复：**成功则继续，异常则 fresh VLA、physical recovery、VLM
   escalation 或安全停止；
9. **记忆更新：**只有验证后的过程、失败和恢复结果进入长期 memory；
10. **审计：**保存 manifest、事件 trace、planner transcript、profile receipt、
    summary 和视频。

由此形成两个相互嵌套的闭环：

- **任务闭环：**知识检索 -> 规划 -> VLA/技能 -> 执行 -> 验证 -> 记忆；
- **计算闭环：**profile 选择 -> 推理 -> deadline/fidelity 监测 -> cache
  invalidation/fallback -> admission 更新。

## 8. 研究问题与实验映射

| Research Question | 核心对比 | 主要指标 |
|---|---|---|
| RQ1：结构化知识是否改善操作推理？ | VLM-only、去除 HAA-RAG、去除 Scene Graph、去除 reflection/memory、完整 Agentic RAG-VLM | 任务成功、检索质量、约束满足、retry 次数 |
| RQ2：Harness 是否改善冻结 VLA 的长程闭环？ | Frozen VLA、always replan/retry、CARVE without memory、full CARVE | success/progress、恢复成功、VLA/VLM 调用、safe stop、预算 |
| RQ3：Optimize 是否降低部署成本且保持行为？ | eager、compile、SMVE、optimized kernel、temporal reuse、量化候选 | P50/P95/P99、deadline miss、VRAM、action fidelity、闭环 non-inferiority |
| RQ4：Agentic 与 Optimize 耦合是否必要？ | Harness only、Optimize only、full coupled CARVE | reaction/task-cycle time、恢复结果、调用成本、cache forced refresh、总成功率 |

### 8.1 已有证据

- Agentic RAG-VLM 原论文中的 HAA-RAG、Scene Graph、retry 和 memory 结果；
- PI0.5 flow-step、compiled BF16、SMVE、W8A16 与 OpenVLA 低比特 gate；
- 相同 failure state 下 Agentic recovery 与 Optimize profile 的联合实验；
- VLM/VLA 共卡 contention、异步 prefetch 和失败边界；
- Toolchain、RunWorkspace、权限隔离和 event-coherent reuse 的 CPU contract。

### 8.2 毕业闭环完成情况与外部验证缺口

截至 2026-08-27，原毕业闭环中的主体工程已经完成：

1. 七工具 Session 已绑定官方 LIBERO-PRO simulator adapter，并完成真实物理
   仿真动作执行；
2. 一键闭环入口已同时运行 VLM Planner/Critic、阶段 Harness、过程记忆、
   Monitor、物理恢复、冻结 PI0.5 和 Optimize Runtime；
3. 已完成 1,200 episode 的 Frozen VLA、Fixed Recovery 与 Full Agentic 配对
   实验，并通过 episode、视频、checkpoint 和 evaluator 隔离审计；
4. compiled BF16 PI0.5 profile 已通过动作保真、deadline 和 paired closed-loop
   admission；
5. 完整系统已在一个 reference 与两个 held-out 状态上取得 3/3 成功，其中一次
   物理恢复后继续完成任务；
6. 已生成机器可读结果、论文表格、代表视频、运行收据和复现实验入口。

尚未完成的是外部有效性扩展，而不是基本具身智能体架构：

1. 在 RoboMME 上验证 Memory/Agentic 机制的跨 benchmark 效果；
2. 在 RoboTwin 2.0 上接入 DM0.5 与 TurboVLA，验证跨 VLA adapter 和效率边界；
3. 在存储或云评测 gate 通过后完成 RoboDojo 综合验证；
4. 将 training-free temporal reuse、VLA 低比特量化和动态 latency benchmark
   作为 Optimize Runtime 的增强实验；
5. 真机部署仍是可选增强项，不应把仿真结果表述为真机证据。

因此，CARVE 已达到“完整可执行具身智能体系统”的工程边界；后续 benchmark
工作的目标是扩大统计证据、跨模型泛化和外部可比性，而不是继续无边界增加
Agent 模块。

## 9. 论文组织建议

| 章节 | 内容 |
|---|---|
| 第 1 章 绪论 | 具身多模态推理、Agentic execution 与实时部署问题 |
| 第 2 章 相关工作 | VLM/VLA、Agentic Policy、RAG/memory、VLA efficient inference |
| 第 3 章 多模态知识增强操作推理 | Agentic RAG-VLM、HAA-RAG、Scene Graph、reflection |
| 第 4 章 面向冻结 VLA 的 Agentic Harness | 多速率架构、工具链、Monitor、Planner、Recovery、Memory |
| 第 5 章 CARVE Optimize Runtime | profile、backend、SMVE、event-coherent reuse、量化与 admission |
| 第 6 章 实验 | 已有知识层结果、Harness 消融、效率消融、完整耦合实验 |
| 第 7 章 总结与展望 | 真机、更多 VLA backend、学习式 verifier 与端侧部署 |

## 10. 最终贡献表述

毕业论文可以围绕三项递进贡献组织：

1. 提出面向机器人操作的结构化多模态知识与反思机制，通过功能可供性检索、
   场景图约束和失败记忆支持操作推理；
2. 构建面向冻结 VLA 的多速率 Agentic Harness，将高层 VLM 语义规划、VLA
   动作生成、有界物理技能、验证和记忆组织为权限明确、可审计的工具生态；
3. 构建证据门控的 Optimize Runtime，通过可替换 backend、profile admission、
   event-coherent reuse 和 deadline-aware fallback，在动作保真与闭环结果约束
   下优化 Agentic VLM/VLA 系统的推理效率。

其中，Agentic RAG-VLM 提供知识基础，Harness VLA/RPent 提供有价值的外部
生态参考，CARVE-VLA 完成自主系统化与高效推理闭环。三者的关系是“成果继承
+ 外部借鉴 + 自主扩展”，而不是简单合并论文名称或实验数字。

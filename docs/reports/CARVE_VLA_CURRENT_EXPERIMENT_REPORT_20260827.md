# CARVE-VLA 框架与实验总报告（历史快照，截至2026-09-02）

> 当前持续维护的主技术底稿已迁至[完整技术报告](AGENTIC_VLA_TECHNICAL_REPORT.md)
> （2026-09-11）。下文保留历史内容，不作为当前完成度、下一步计划或新实验结论。
> 数字引用与面试表述优先核对新底稿中的证据范围、量化保真口径和负结果。

更新时间：2026-09-02

论文方向：面向具身智能的多模态智能体系统与高效推理研究

历史文档定位：本报告曾作为项目主技术说明，面向周末组会准备、毕业论文写作、
简历提炼和技术面试复盘。文中的“已完成”均要求存在代码、机器可读结果或可审计
视频；规划中的实验不会提前写成结果。

## 摘要

现有 Vision-Language-Action（VLA）模型能够直接从视觉和语言生成机器人动作，
但冻结模型在长程任务中仍缺少持续任务状态、工具权限、错误恢复和过程记忆；在
Agentic 系统中额外引入 VLM 规划与验证后，又会产生模型共卡、调用等待和动作
时效性问题。CARVE-VLA 因此研究一个位于基础模型之外的资源感知具身智能体
运行时：以低频 VLM Planner/Critic 负责语义任务，以高频 Monitor 负责执行风险，
以冻结 PI0.5 或可替换 VLA 负责动作生成，并通过 typed tools、可信过程记忆、
有界恢复和 fail-closed 机制形成可审计闭环。系统进一步使用 profile-driven
Optimize Runtime，根据动作保真度、P95 时延、显存、闭环结果和 fallback 完整性
决定优化配置能否部署。

当前证据包括 1,200 回合 LIBERO/LIBERO-PRO 配对实验、PI0.5 与 Planner 的独立
推理消融、完整 VLM+VLA+Memory+Recovery 闭环、两个长程任务族的记忆路由实验，
以及 RoboMME 240 回合与 StarVLA PI-v3/RoboDojo 的第二 VLA 闭环验证。
LIBERO-Pro 上 Agentic 成功率由 `180/400` 提升至 `183/400`，差异不显著，但
控制步与 VLA 调用分别下降 `7.2%/8.4%`；RoboMME 八任务子集上 Raw、Harness、
Full Runtime 分别为 `23/80`、`36/80`、`42/80`，Full 相对 Raw 的精确 McNemar
`p=0.000157`。PI0.5 P95 从 `282.43 ms` 降至 `54.67 ms`，
并通过 `45/45` 配对动作保真度；可信过程记忆与大小模型路由在保持成功的同时，
将重复任务 wall time 最高降低 `73.1%`。这些结果支持“完整、受约束且高效的
具身 Agent Runtime”这一系统贡献，同时也明确指出全基准覆盖、同规模跨 VLA
泛化和真机验证仍未完成。RoboMME 的 `MoveCube` 五回合深度消融进一步得到固定
文本 `1/5`、可部署 GroundSG VLM `4/5`、特权 oracle 上界 `5/5`，用于解释动态
规划、视觉落点和工具策略的作用。最新 RoboDojo 实验中，Flow2 相对 Flow4
的驻留配对推理平均时延下降 `23.37%`，两个任务的闭环质量门均保持 `3/3`；在
三个受控 stale-action 停滞中，Optimize-only 为 `0/3`，两阶段 Agentic 恢复为
`3/3`。后者用于证明机制有效，不能外推为自然任务成功率提升。

## 核心研究问题

1. 如何在不重新训练基础 VLA 的条件下，为机器人补充持续规划、工具调用、过程
   记忆、异常恢复和安全停止？
2. 如何区分高频执行风险与低频语义判断，使大 VLM 不阻塞动作控制关键路径？
3. 如何把量化、编译、步数缩减和缓存复用从“离线跑分快”升级为具有动作保真度、
   deadline、闭环行为和 fallback 约束的部署 profile？
4. 如何让记忆不仅增强任务能力，还能跳过重复规划并路由到更小模型，从而直接
   降低 Agentic 调用链的计算成本？
5. 如何在 benchmark 中隔离 evaluator 私有真值，并用配对 episode、trace、视频
   和统计检验区分真实系统收益与偶然成功？

## 1. 当前结论

CARVE-VLA 已完成一个可执行、可审计的具身智能体研究原型，而不只是
`Monitor + Retry` 的脚本包装。系统在冻结 PI0.5 外部集成了高层 VLM
Planner/Critic、结构化任务计划、具身工具、风险监测、可信过程记忆、物理恢复、
安全停止和高效 VLA Runtime，并在真实 MuJoCo 物理仿真中完成闭环验证。

当前实验已经足以支撑硕士毕业论文的系统实现与实验章节，也适合整理为具身智能、
VLA 部署或 Agent Runtime 方向的求职项目。需要保持的结论边界是：Agentic
机制在完整 benchmark 上的成功率提升较小且没有统计显著性；当前优势主要体现在
系统完整性、有界恢复、执行可审计、VLA 调用削减、实时关键路径和重复任务的
高层推理开销降低。

## 2. 完整系统组成

| 层级 | 当前实现 | 主要职责 |
|---|---|---|
| 高层语义智能体 | Qwen3.5-9B NF4 Planner / Qwen3.5-4B Critic | 图像语义理解、长程任务拆分、工具选择、异常事件判断 |
| Agentic Harness | task plan、schema gate、预算、executor-consistency gate | 约束模型决策，管理阶段推进、重试、恢复和安全停止 |
| 高频风险监测 | `ExecutionRiskMonitor` | 根据动作、状态响应、视觉变化、action age 和 deadline 检测风险 |
| 可信记忆 | failure / affordance / verified-procedure memory | 复用经验证的符号步骤，不保存动作、轨迹、图像或场景坐标 |
| 动作执行 | 冻结 PI0.5 / StarVLA PI-v3 + typed embodied skills | 执行语言条件动作块或白名单解析技能 |
| 语义验证 | 低频 VLM Critic + private evaluator isolation | 判断可观察后置条件；评测真值不暴露给 Agent |
| 物理恢复 | bounded retract / release / lift / reobserve | 从支持的执行失败中恢复；预算耗尽后 fail closed |
| Optimize Runtime | flow-step、`torch.compile`、SMVE、量化、profile admission | 在动作保真度和闭环门控下优化时延与显存 |
| 审计层 | JSONL events、receipts、summary、MP4 | 保存模型调用、工具执行、验证来源、时延和视频 |

系统采用多速率设计：PI0.5 位于高频动作关键路径；Monitor 以低成本信号持续运行；
VLM Planner/Critic 只在任务开始、阶段边界或异常事件上低频触发。VLM 不直接生成
关节控制量，只能输出经过 schema 和工具白名单校验的结构化决策。

## 3. CARVE-VLA 的独特价值

当前最有辨识度的贡献不是某个孤立组件，而是将具身 Agent 的**执行可靠性**、
**长期任务状态**和**推理资源管理**放入同一个运行时闭环。六项实验收益可以
收束为下面三个主要贡献和一个联合价值。

### 3.1 从“VLA 外挂重试”升级为持续执行 Harness

CARVE 不只在失败后重新调用一次 VLA，而是显式管理完整执行生命周期：

```text
任务理解 -> 分阶段计划 -> 工具/VLA 执行 -> 高频风险监测
         -> 低频语义验证 -> 有界恢复 -> 计划推进或安全停止
```

它的价值体现在：

- 4 个失败被转换为成功，说明 Harness 确实能纠正一部分可恢复错误；
- control steps 和 VLA calls 分别减少 `7.2%/8.4%`，说明系统会提前终止
  无效执行，而不只是通过更多调用换取结果；
- 所有 Planner、工具、恢复、验证和停止决策均有类型、预算与 trace；
- private evaluator 与 Agent 隔离，使语义判断不能偷看仿真成功真值；
- executor-consistency gate 防止“恢复动作完成”被错误解释为“任务阶段完成”。

因此，即使 aggregate success 的净提升有限，CARVE 仍将冻结 VLA 从无状态的
动作生成器变成了具有任务状态、错误生命周期和安全边界的持续执行系统。

### 3.2 将过程记忆同时变成能力层和计算层

一般的机器人记忆主要用于补充历史观测或提高动作预测。CARVE 当前更独特的定位
是：**verified-procedure memory 既保存经验证的任务结构，也直接参与推理路由**。

- 写入前必须完成完整 plan ledger 并通过后置条件验证；
- 只保存符号步骤，不保存图像、坐标、轨迹或评测真值；
- 新场景仍由当前观测重新定位和执行，避免照搬旧场景动作；
- 命中可信过程后跳过大 Planner 的 task-start 调用；
- 大模型负责 novel-task planning，小模型负责 Critic 和 event fallback。

这使记忆产生了可量化的系统收益：在 T3 中，Memory+9B 单独带来 `62.0%`
wall-time reduction；保持相同记忆再路由到 4B Critic，Critic 时延进一步降低
`34.3%`；最终相对 Direct 9B 达到 `73.1%` wall-time reduction。该实验将
“记忆收益”和“模型尺寸收益”分开测量，而不是把它们混成一个黑盒结果。

### 3.3 高效推理不是单点加速，而是证据驱动的部署准入

CARVE Optimize Runtime 的重点不是简单启用量化或编译，而是只有同时通过以下
条件的 profile 才能进入机器人闭环：

1. checkpoint、相机、状态和动作 contract 一致；
2. 配对 action/semantic fidelity 合格；
3. P95、显存和 deadline 满足目标；
4. matched closed-loop 结果不退化；
5. 运行前提失效时能回退到独立验证的 profile。

因此项目不仅保留了 Compile+SMVE 的正结果，也记录并拒绝了当前后端中更慢的
INT8、闭环退化的 INT8+SMVE 和错误的异步 profile。这个“**优化必须通过行为
准入**”的原则比单独报告 FLOPs、显存或开放环误差更贴近机器人部署。

### 3.4 Agentic 与 Optimize 的联合价值

CARVE 的核心研究问题可以表达为：

> Agentic 机制增加了规划、验证和恢复能力，也增加了推理调用与等待；系统应当
> 根据任务状态分配智能和计算，在保持动作关键路径实时性的同时，把昂贵语义推理
> 限制在真正需要的边界上。

为此，系统形成两条耦合但可独立使用的路径：

- **语义慢路径**：可信记忆优先，9B 负责新任务，4B 负责已知任务的验证与异常；
- **动作快路径**：PI0.5 使用通过 fidelity 和闭环门控的 Compile+SMVE profile，
  在当前实验中保持约 `55--60 ms` P95。

这一区别使 CARVE 不等同于纯 Agent planner，也不等同于单模型加速器。它更准确
地属于**资源感知的具身智能体运行时**：统一决定何时思考、调用什么能力、采用
什么推理 profile、何时恢复以及何时停止。

### 3.5 与近期方向的关系

| 方向 | 主要关注点 | CARVE 的补充位置 |
|---|---|---|
| Harness VLA 类框架 | 用 Agent、记忆和解析技能扩展冻结 VLA 的操作范围 | 增加高频风险监测、执行器一致性、实时 profile 和计算准入 |
| Realtime-VLA 类方法 | 蒸馏、推测执行或模型结构带来的 VLA 加速 | 不重新训练基座模型，强调即插即用 runtime、fallback 与 Agent 共卡 |
| VLA Memory 方法 | 用历史信息改善长程动作决策 | 将可信符号过程直接用于 Planner 跳过和大小模型路由 |
| Robot middleware / harness | 约束模型输出、资源和故障传播 | 给出 VLM+VLA 的实际闭环、deadline、trace 和恢复实现 |

这里不应声称每个模块都是首次提出。可以主张的是：CARVE 已经实现并验证了一个
把上述边界统一起来的完整系统，尤其是“可信过程记忆作为计算路由器”和“Agentic
决策与 VLA profile admission 的联合设计”构成了当前最值得继续打磨的差异点。

近期对照资料：

- [Harness VLA](https://arxiv.org/abs/2607.08448)
- [Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778)
- [Harness Engineering for Physical AI](https://arxiv.org/abs/2606.09416)
- [RT-VLA](https://arxiv.org/abs/2606.14010)

## 4. 实验证据总览

| 证据链 | 规模 | 核心结果 | 能证明什么 |
|---|---:|---|---|
| 完整 LIBERO-Pro 配对研究 | 1,200 episodes | Frozen `180/400`，Fixed `181/400`，Agentic `183/400` | 完整 benchmark 集成、有界恢复、调用削减、边界分析 |
| PI0.5 高效推理消融 | 5 个 profile，45 组配对观测 | P95 `282.43 -> 54.67 ms`，V3 `45/45` fidelity | 高频 VLA 路径可在 80 ms 目标内运行 |
| Planner 量化消融 | 3 个 profile，30 组语义事件 | NF4 显存 `8.46 -> 3.08 GiB`，语义一致率 `100%` | 量化可降低共卡部署门槛，但不一定加速 |
| 完整具身智能体闭环 | T8 三个官方状态 | `3/3` 成功，1 次物理恢复后继续完成 | Planner、Harness、Memory、Critic、Recovery、VLA 联合可执行 |
| T8 记忆路由 | 4 个官方状态 | `4/4`，startup Planner 为 0；共同状态 wall time `-53.2%` | 经验证过程记忆可消除重复任务启动规划 |
| T3 跨任务因子化路由 | 3 个 held-out 状态，3 条正向路线 | 均 `3/3`；最终路线 wall time `-73.1%` | Memory 与模型路由的收益可分解，并扩展到第二类任务 |
| 软件与证据审计 | 全项目 | `258 passed`；视频、事件和 summary 可追溯 | 关键契约与实验产物具有可复查性 |

## 5. 实验一：完整 LIBERO-Pro 研究

### 5.1 设置

- 仿真：官方 LIBERO-10，以及 LIBERO-Pro Object、Position Swap、Task Logic；
- 任务：4 suites x 10 tasks x 10 个配对初始状态；
- 方法：Frozen VLA、Fixed Recovery、Full Agentic；
- 总规模：120 个方法单元、1,200 个真实物理仿真 episode；
- 模型：同一个冻结 PI0.5 LIBERO PyTorch checkpoint；
- 所有方法使用相同状态、checkpoint、动作 profile 和随机协议；
- Agent 事件中不包含 reward、task success、object pose 或 simulator private state；
- 1,200 个视频全部可解码，共 454,540 帧。

### 5.2 结果

| Suite | Frozen VLA | Fixed Recovery | Full Agentic |
|---|---:|---:|---:|
| Standard | 96/100 | 97/100 | 97/100 |
| Object | 66/100 | 67/100 | 68/100 |
| Position Swap | 11/100 | 11/100 | 11/100 |
| Task Logic | 7/100 | 6/100 | 7/100 |
| **Total** | **180/400** | **181/400** | **183/400** |

Agentic 相对 Frozen 有 4 个失败转成功和 1 个成功转失败，净增 3 个成功，
McNemar `p=0.375`。因此不能声称统计显著的 benchmark 成功率提升。

系统层面，Full Agentic 将 control steps 从 `159,651` 降至 `148,103`
（`-7.2%`），将 VLA calls 从 `16,047` 降至 `14,699`（`-8.4%`）。
VLA P95/P99 为 `59.54/63.19 ms`，`14,699` 次调用中有 8 次超过 80 ms，
达标率为 `99.946%`。同步 Planner 的平均时延仍为 `9.09 s`，使 Agentic
总 wall time 比 Frozen 高 `16.5%`，这直接引出了后续记忆与模型路由实验。

### 5.3 配对统计与错误转化

三种条件使用相同任务、初始状态、checkpoint 和动作 profile，因此可以逐回合
比较，而不只是比较两个独立成功率：

| 比较 | 失败转成功 | 成功转失败 | 净变化 | Exact McNemar p |
|---|---:|---:|---:|---:|
| Fixed vs. Frozen | 3 | 2 | +1 | 1.000 |
| Agentic vs. Frozen | 4 | 1 | +3 | 0.375 |
| Agentic vs. Fixed | 2 | 0 | +2 | 0.500 |

Frozen、Fixed 和 Agentic 总成功率的 Wilson 95% 区间分别为
`40.20--49.90%`、`40.44--50.15%` 和 `40.93--50.65%`。区间高度重叠，配对
检验也不显著，因此本文不能把 `+0.75` 个百分点写成普遍能力提升。它更准确地
说明：在固定 VLA 的能力范围内，Harness 能纠正少量可恢复错误，但 Monitor 和
恢复策略仍可能误触发。

具体地，Fixed Recovery 转化了 Standard T8 trial 2、Object T8 trials 7/9，
同时使 Object T6 trial 1 和 Task-Logic T8 trial 5 回归。Full Agentic 保留三个
转化，新增 Object T7 trial 7，并恢复 Task-Logic T8 的 Fixed 回归，但仍保留
Object T6 trial 1 的一次成功转失败。这些配对样本构成后续分析 Monitor precision
和 recovery admission 的直接案例，而不是只看 aggregate success。

### 5.4 恢复机制边界

Fixed 和 Agentic 均在 400 回合中的 173 回合触发一次有界恢复；恢复相关成功分别
为 `22/173` 和 `24/173`。这证明物理恢复能覆盖一部分执行失败，但也说明大量失败
来自 VLA 未掌握的位置变化、任务逻辑或低层泛化，单靠 Runtime retry 无法补齐。
系统因此限制每回合恢复预算，并在预算耗尽、执行不一致或证据不足时安全停止，
避免无限重试把偶然动作包装成智能。

### 5.5 完整效率结果

| Metric | Frozen VLA | Fixed Recovery | Full Agentic |
|---|---:|---:|---:|
| Control steps | 159,651 | 145,586 | 148,103 |
| VLA calls | 16,047 | 14,442 | 14,699 |
| VLA P50 | 56.43 ms | 57.02 ms | 56.44 ms |
| VLA P95 | 60.61 ms | 61.14 ms | 59.54 ms |
| VLA P99 | 64.26 ms | 65.32 ms | 63.19 ms |
| Calls above 80 ms | 1 | 0 | 8 |
| Episode wall time | 7,610.7 s | 7,296.4 s | 8,864.6 s |

Agentic 语义层只调用 263 次，而 VLA 调用 14,699 次，比例为 `1.79%`。这验证了
事件触发的多速率设计确实没有让 VLM 进入每个动作周期。然而 263 次同步 Planner
调用累计 `2,389.6 s`，平均/P95 为 `9.09/11.23 s`，最终抵消了动作调用削减，
使整组 episode wall time 高于 Frozen。这一负结果直接支撑 Optimize Runtime 的
研究动机：高频路径关注 VLA deadline，低频路径则需要异步调用、可信记忆、较小
或量化 VLM 和更严格的 semantic admission。

### 5.6 完整性与可审计性

- `120/120` 方法单元和 `1,200/1,200` episode 均存在；
- `1,200/1,200` 个 MP4 全部解码成功，共 `454,540` 帧；
- 三种方法的 400 组配对状态 identity 全部一致；
- 全部回合使用同一 PI0.5 checkpoint 和部署 profile；
- Agent event 中未发现 reward、task success、object pose 或 simulator state；
- 每个 episode 均包含 summary、typed trace 和视频，可追溯触发、恢复与停止原因。

机器可读汇总位于
`results/libero_pro_full_study_20260825/aggregate/study_summary.json`，审计报告位于
`results/libero_pro_full_study_20260825/aggregate/audit_report.json`。当前所有正式
benchmark 结论均以这组 LIBERO/LIBERO-PRO 结果为准。

## 6. 实验二：Optimize Runtime 与轻量化

### 6.1 PI0.5 实时路径

| Profile | P95 | Miss@80ms | Peak VRAM | Fidelity | 部署决策 |
|---|---:|---:|---:|---:|---|
| Eager BF16，7 steps | 282.43 ms | 100% | 7.12 GiB | reference | 行为基线 |
| Eager BF16，2 steps | 151.35 ms | 100% | 7.12 GiB | reference | deadline 否决 |
| Compile BF16，2 steps | 66.06 ms | 0% | 6.98 GiB | 45/45 | fallback |
| Compile BF16 + SMVE | 54.67 ms | 0% | 6.98 GiB | 45/45 | realtime default |
| Late-language INT8 | 994.74 ms | 100% | 6.36 GiB | 45/45 | 当前版本否决 |

V3 相对七步 eager 基线将 P95 降低 `80.6%`。SMVE 只跳过 adapter 明确标记为
padding 的静态缺失视角；前提不满足时自动回退到 V2，因此不能表述为任意删除
有效相机输入。

### 6.2 Planner 量化

| Planner profile | Allocated VRAM | P95 | 配对语义一致率 | 决策 |
|---|---:|---:|---:|---|
| Qwen3.5-4B BF16 | 8.46 GiB | 1.90 s | 100% | latency tier |
| Qwen3.5-4B INT8 | 4.84 GiB | 6.50 s | 100% | 当前 kernel 否决 |
| Qwen3.5-4B NF4 | 3.08 GiB | 2.91 s | 100% | low-memory tier |

NF4 将 allocated VRAM 降低 `63.6%`，但 P95 增加 `53.4%`。该结果证明
低比特量化适合降低共卡部署门槛，不支持“量化必然加速”的结论。PI0.5 INT8
也表现出明显的软件栈版本敏感性，因此 CARVE 的部署 profile 必须绑定硬件、
软件版本、checkpoint、动作 fidelity 和闭环结果。

## 7. 实验三：完整具身智能体闭环

在官方 LIBERO-Pro Object T8 `put both moka pots on the stove` 上，完整系统使用
Qwen3.5-9B NF4、冻结 PI0.5、stage plan、Critic、可信记忆、风险监测和物理恢复。

| 状态 | 记忆 | 恢复 | 结果 | 步数 |
|---:|---:|---:|---:|---:|
| 0 reference | 建立验证过程 | 0 | 成功 | 399 |
| 7 held-out | 命中 | 0 | 成功 | 380 |
| 9 held-out | 命中 | 1 | 成功 | 415 |

三回合共 `3/3` 成功、`3/3` 任务计划完成，PI0.5 共调用 119 次，P95
`57.23 ms`，80 ms miss 为 `0/119`。状态 9 中 recovery 在 step 142 完成，
第一阶段直到 step 242 才由视觉 Critic 确认，证明恢复工具不能错误地完成语义
任务阶段，executor-consistency gate 实际生效。

额外的 RGB-D 工具资格验证完成 `visual_servo_above`、`rotate_wrist` 和
`move_relative` 三种物理技能，3/3 成功；VLM grounding 实验也完成了结构化
参数生成、schema gate 和物理 dispatch 的完整链路。

## 8. 实验四：可信记忆与自适应模型路由

### 8.1 T8 重复任务

在共同的三个官方状态上，Direct 9B 与 Memory+4B 均为 `3/3`。记忆路由将
task-start Planner 调用从 `3` 降为 `0`，平均回合 wall time 从 `55.55 s`
降至 `26.01 s`（`-53.2%`），同时保持 `0/125` VLA deadline miss。增加的
第 4 个 held-out 状态也成功，因此 Memory+4B 总计 `4/4`。

### 8.2 T3 跨任务因子化实验

第二类任务为标准 LIBERO-10 T3：将黑碗放入柜子底层抽屉并关闭抽屉。可信过程
包含“放入”和“关闭”两个阶段，只保存符号步骤，不保存动作、轨迹、图像、姿态
或评测真值。

| Route | Success | Startup Planner | Event Planner | Mean wall | Critic mean | VLA P95 |
|---|---:|---:|---:|---:|---:|---:|
| Direct 9B NF4 | 3/3 | 3 | 2 | 64.96 s | 5.39 s | 59.97 ms |
| Memory + 9B NF4 | 3/3 | 0 | 1 | 24.69 s | 5.79 s | 60.19 ms |
| Memory + 4B BF16 | 3/3 | 0 | 0 | 17.49 s | 3.81 s | 60.98 ms |

同模型比较中，Memory+9B 保持 `3/3`，Planner calls 从 `5` 降至 `1`，wall
time 降低 `62.0%`。在相同记忆路线下换用 4B Critic，Critic mean latency
再降低 `34.3%`。最终 Memory+4B 相对 Direct 9B 保持成功，Planner calls
从 `5` 降至 `0`，wall time 降低 `73.1%`；三条路线共 198 次 VLA 调用均满足
80 ms deadline。

Direct 4B 在无记忆条件下连续两次只生成一个阶段，coverage gate 在 step 0
拒绝并安全停止，没有向机器人发送动作。该负结果给出了清晰的能力边界：9B
负责新任务规划，可信记忆负责重复任务 warm start，4B 适合低延迟 Critic 或
事件 fallback，而不是无条件替代大 Planner。

## 9. 框架修正与负结果价值

T4 资格实验曾暴露两条无效记忆写入路径：双对象任务可能接受单阶段计划，私有
任务成功也可能提升尚未完成的 plan ledger。该组结果已经隔离，不计入正向结果。
系统随后增加了：

1. 多子句任务的有序覆盖检查；
2. 仅完整任务计划允许写入 verified-procedure memory；
3. 直接保存可信计划步骤，而不是从重复 primitive trace 猜测缺失阶段；
4. 对下划线分隔的任务和物体类别进行统一词法归一化；
5. 恢复执行器不能验证 VLA 语义阶段的 executor-consistency gate。

这些修正使 Memory 不只是缓存，而是带有写入资格、检索条件、执行验证和审计
边界的可信过程记忆。

## 10. 当前可声明与不可声明

### 可声明

1. 已实现冻结 VLA 外部的完整 Agentic execution harness；
2. 已实现 VLM Planner/Critic、typed tools、任务计划、可信记忆、恢复与 safe stop；
3. 已实现 profile-driven Optimize Runtime，并用 fidelity、deadline、闭环和
   软件栈身份决定部署 profile；
4. PI0.5 默认 profile 在 RTX 4090 上达到约 `55--60 ms` P95；
5. 在完整 1,200 回合研究中减少控制步与 VLA 调用，并保持实时 VLA 路径；
6. verified-procedure memory 与模型路由在两个任务族、7 个 memory-routed
   官方状态上保持成功并显著降低重复高层推理等待；
7. StarVLA PI-v3/RoboDojo 已完成第二 VLA 小样本闭环、flow-step 优化和受控
   Agentic 恢复验证；
8. 当前代码、事件、视频和机器可读结果形成了完整证据链。

### 不可声明

1. Agentic 在整个 benchmark 上取得统计显著的成功率提升或 SOTA；
2. 已提出新的通用量化算法，或 PI0.5 参数量已经显著压缩；
3. 4B Planner 可以普遍替代 9B Planner；
4. model-call deadline 等价于机器人整机硬实时；
5. 已完成第二个 VLA 家族的同规模 benchmark；当前 StarVLA 仅为小样本闭环；
6. VLM Critic 已可靠完成终态判别或拥有 safe-stop authority；
7. 独立 4B VLM 与 StarVLA 已在单张 RTX 4090 稳定共驻；
8. 已完成真机实验或证明 sim-to-real 泛化。

## 11. 毕业论文与求职使用方式

毕业论文可以按以下证据顺序展开：

1. Agentic RAG-VLM 作为结构化多模态操作推理基础；
2. CARVE Agentic Harness 将结构化推理扩展到持续 VLA 执行；
3. 1,200 回合实验给出 Harness 的收益与能力边界；
4. Optimize Runtime 解决高频 VLA 时延和共卡部署问题；
5. 可信记忆与模型路由解决低频 Planner 成为端到端瓶颈的问题；
6. 完整具身闭环和跨任务实验说明各模块能够高效耦合。

适合简历的客观表述：

> 设计并实现冻结 VLA 外部的具身智能体执行与高效推理框架，集成 VLM
> Planner/Critic、typed tools、风险监测、可信过程记忆、物理恢复和安全停止；
> 在 1,200 回合 LIBERO-Pro/10 配对实验中完成可审计闭环验证，PI0.5 P95
> 约 60 ms；通过 verified-procedure memory 与 9B/4B 模型路由，在两个长程
> 任务族上保持成功并将重复任务回合时间最高降低 73.1%。

## 12. 当前完成度与后续优先级

| 项目 | 状态 |
|---|---|
| Agentic 框架设计与实现 | 已完成研究原型 |
| VLM Planner/Critic 与 VLA 耦合 | 已完成 |
| Monitor、Memory、Recovery、Safe Stop | 已完成 |
| Optimize Runtime 独立消融 | 已完成 |
| Planner/VLA 量化边界 | 已完成定向验证 |
| 完整 benchmark 实验 | LIBERO-Pro 1,200 回合；RoboMME 八任务子集 240 回合 |
| 代表性完整具身闭环 | 已完成，含视频 |
| 跨任务 Memory + 模型路由 | 已完成紧凑验证 |
| 第二 VLA 全闭环 | 已完成 StarVLA/RoboDojo 小样本机制验证，尚非同规模 benchmark |
| 在线 VLM Critic 准入 | 未完成；低 false-stop，但终态召回不足 |
| 单卡 VLM+StarVLA 共驻 | 未完成；首次联合视觉推理 OOM，需资源隔离 |
| 真机实验 | 未完成，当前毕业主线不依赖 |

当前不需要继续无差别扩充 LIBERO 回合。第二 VLA adapter 闭环已经补齐；若为
投稿继续实验，应优先扩大 RoboDojo 自然失败任务/seed 覆盖，并解决在线 VLM 的
资源隔离和 Critic 准入，而不是重复运行已经完成的 1,200 回合矩阵。

## 13. 证据入口

- 论文就绪主证据包：`results/paper_ready_20260827/`
- 自动生成脚本：`scripts/build_current_evidence_package.py`
- 完整 LIBERO-Pro 报告：`docs/status/LIBERO_PRO_FULL_STUDY_RESULTS_20260826.md`
- 高效推理报告：`docs/status/CARVE_EFFICIENT_INFERENCE_ABLATION_20260826.md`
- 完整具身智能体：`docs/status/FULL_EMBODIED_AGENT_RESULTS_20260827.md`
- T8 记忆路由：`docs/status/CARVE_MEMORY_ROUTED_PLANNER_PROFILE_20260827.md`
- RoboDojo/StarVLA 汇总：`docs/status/ROBODOJO_CARVE_PAIRED_RESULTS.md`
- RoboDojo 受控恢复：`artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/`
- T3 跨任务路由：`docs/status/CARVE_CROSS_TASK_MEMORY_ROUTING_RESULTS_20260827.md`
- 完整实验数据：`results/libero_pro_full_study_20260825/`
- LIBERO-PRO 代表性视频包：`results/libero_pro_representative_videos_20260828/`
- 完整具身视频：`results/full_embodied_agent_20260827/`
- 跨任务视频与聚合：`results/cross_task_memory_routing_20260827/`
- 当前统一完成记录：`docs/status/CARVE_VLA_COMPLETED_WORK.md`
- 当前论文：`paper/CARVE-VLA/root.pdf`

## 14. 代码级架构说明

### 14.1 一次 episode 如何运行

完整 episode 不是简单的 `while env.step()`，而是以下受控生命周期：

1. **创建 workspace：**绑定 checkpoint、adapter、hardware、profile 和随机种子；
2. **读取 deployable observation：**只包含相机、机器人状态和任务文本；
3. **任务启动路由：**先检索 verified-procedure memory；未命中才调用大 Planner；
4. **生成 task plan：**VLM 输出 1--4 个阶段、预期后置条件和 typed intent；
5. **权限检查：**schema、tool allowlist、safe boundary、预算和 episode/timestep；
6. **执行 primitive：**通过 admitted VLA profile 或注册 physical skill 执行；
7. **高频 Monitor：**检查状态响应、视觉变化、action age 和 deadline；
8. **低频验证：**Critic 只在阶段边界或异常事件判断语义后置条件；
9. **恢复或升级：**fresh VLA、bounded skill、Planner escalation 或 safe stop；
10. **记忆写入：**仅完整计划且后置条件已验证时保存符号过程；
11. **关闭 episode：**写入 JSONL trace、receipt、summary、artifact index 和视频。

### 14.2 关键代码地图

| 路径 | 实际职责 |
|---|---|
| `agentic_vla/runtime/contracts.py` | InferenceRequest、ActionChunk、ActionSpec、能力与 trace |
| `agentic_vla/runtime/adapter.py` | VLA 无关 adapter 和 capability negotiation |
| `agentic_vla/runtime/agent.py` | VLM provider、计划覆盖、grounding、输出修复与异步 ticket |
| `agentic_vla/runtime/monitor.py` | 高频可观测信号风险监测 |
| `agentic_vla/runtime/harness.py` | 状态机、Planner 生命周期、safe hold、预算和 stale decision 拒绝 |
| `agentic_vla/runtime/recovery.py` | FailureMemory、恢复计划、技能注册与执行验证 |
| `agentic_vla/runtime/knowledge.py` | HAA、Scene Graph、affordance 和 procedure memory |
| `agentic_vla/session.py` | canonical task plan、工具、验证、memory 和 artifact 生命周期 |
| `agentic_vla/toolchain/` | typed tool schema、权限、registry、workspace 和 verifier |
| `agentic_vla/optimization/` | model/backend plugins、fidelity、admission、reuse 和 fallback |
| `agentic_vla/benchmarks/libero_runtime.py` | LIBERO 观测与 evaluator 隔离 |
| `agentic_vla/benchmarks/robomme_runtime.py` | RoboMME 8-D joint/action 与初始视频记忆适配 |

### 14.3 七工具生态

| Tool | Effect | 边界 |
|---|---|---|
| `observe` | 只读 | 获取当前 deployable observation 与风险证据 |
| `retrieve_memory` | 只读 | 检索过程、可供性和失败恢复知识 |
| `vla_act` | 物理 | 只能调用通过 admission 的 VLA profile |
| `run_skill` | 物理 | 只能执行白名单 bounded skill，参数必须通过 schema |
| `verify` | 只读 | 输出 confirmed/contradicted/inconclusive |
| `safe_hold` | 生命周期 | 等待低频 Planner 时保持安全边界与 heartbeat |
| `finish` | 生命周期 | 关闭运行并保存可审计结果 |

工具合同拒绝 `joint_positions`、`torques`、`trajectory`、`object_pose`、
`reward` 和 `success` 等越权字段。高层 Agent 选择能力，不直接绕过 VLA/skill
边界生成关节动作。

### 14.4 Monitor、Critic 与 Planner 的职责

- **Monitor：**高频、确定性、低成本。判断执行是否停滞、动作是否陈旧、状态是否
  响应以及 deadline 是否不足；不做开放词汇语义理解。
- **Critic：**低频视觉语义验证。判断当前可观察后置条件是否成立；不读取私有
  evaluator success。
- **Planner：**低频任务分解和工具选择。处理新任务、语义歧义和重复失败；不进入
  80 ms 动作控制关键路径。

这三者的分频使系统既不依赖纯固定规则完成语义规划，也不需要每个控制周期阻塞
调用大 VLM。

## 15. Optimize Runtime 技术细节

### 15.1 Profile-driven deployment

每个 profile 绑定 checkpoint、adapter、动作语义、precision、flow steps、
action horizon、backend、GPU/软件栈和 fallback。Runtime 只应用 adapter 声明
支持的控制；不支持项会被记录为 dropped control，而不是假装配置已生效。

候选 profile 依次通过：

1. checkpoint/observation/action contract；
2. fixed-noise action fidelity；
3. warm P50/P95/P99、deadline miss 和 peak VRAM；
4. paired closed-loop non-inferiority；
5. Agentic recovery path；
6. manifest、receipt 与 fallback 完整性。

### 15.2 当前六层优化栈

| 层级 | 已实现内容 | 证据边界 |
|---|---|---|
| L0 输入 | SMVE 静态缺失视角消除 | 仅删除 adapter 保证为 padding 且 mask 全假的 view |
| L1 后端 | eager、`torch.compile`、TorchAO candidate | compile 是当前默认正结果 |
| L2 时间 | 动作队列、reuse gate | 前缀缓存 backend 尚未形成正式加速结果 |
| L3 计算 | 2-step flow、horizon/refresh 控制 | 2-step 经 15 个配对状态校准 |
| L4 压缩 | Planner NF4、VLA late-language INT8 candidate | 量化受 kernel 与闭环门控约束 |
| L5 调度 | event-triggered Planner、大小模型路由、共卡测量 | synchronous Planner 仍是主要瓶颈 |

### 15.3 SMVE 的准确含义

Static Masked-View Elision 不是任意视觉剪枝。它根据 named-view contract 在每次
调用确认目标相机为静态 padding 且 mask 全假，然后跳过对应 SigLIP 编码和
prefix 构建。条件不成立时立即回退普通 compiled BF16。该优化通过 `45/45`
动作 fidelity，最差 chunk MAE `0.00232`、cosine `0.999897`、gripper
agreement `1.0`。

### 15.4 Event-Coherent Reuse

CARVE 的 reuse gate 管理“是否有权复用”，缓存 backend 管理“如何复用”。以下
事件强制 fresh inference：instruction/subgoal/profile 改变、Planner/Recovery
epoch 改变、stall/contact/stale action、风险升高、视觉或本体变化过大、cache
超龄。当前 gate 与动作队列已实现，training-free prefix backend 仍是后续增强。

### 15.5 量化为何不是默认加速

Qwen3.5-4B NF4 将 allocated VRAM 从 `8.46` 降至 `3.08 GiB`，但 P95 从
`1.90` 增至 `2.91 s`，因此属于 capacity tier。PI0.5 late-language INT8 在
Torch 2.7.1/TorchAO 0.15 栈曾达到约 `71.61 ms` 并通过 recovery gate，但在
当前软件栈 P95 回退到 `994.74 ms`，因此是版本绑定候选，不是当前实时默认档。

## 16. 数字口径与证据强度

### 16.1 三个 PI0.5 P95 为什么不同

- `54.67 ms`：固定观测、独立高效推理消融；
- `59.54 ms`：1,200 回合 Full Agentic 共卡闭环；
- `57.23 ms`：三回合完整具身 Agent 验证。

三者负载和采样不同，应分别报告，不能择优混写。

### 16.2 Agentic 成功率如何解释

`180/400 -> 183/400` 有 4 个失败转成功和 1 个成功转失败，McNemar
`p=0.375`，不显著。可防御的价值是：

- 纠正一部分可恢复错误，同时暴露 Monitor 误触发；
- 控制步减少 `7.2%`，VLA 调用减少 `8.4%`；
- 不支持事件能安全停止，避免无限 retry；
- Planner、Memory、Tools、Recovery 和审计链路完整运行；
- 可信记忆在重复长程任务上显著降低高层等待；
- VLA 快路径在 Agent 共卡时仍保持约 60 ms P95。

## 17. RoboMME 动态规划与 GroundSG 闭环实验

RoboMME 是当前 Agentic/Memory 主 benchmark，共 16 个任务、四类记忆：
temporal、spatial、object 和 procedural。官方 test split 每任务 50 个固定
episode，最大 1,300 steps。

### 17.1 协议与模型

- benchmark commit：`d57969fd30f8e8318fb67c389a848b3776724470`；
- policy commit：`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`；
- 冻结实验协议（已归档）：`docs/archive/experiment_plans/CARVE_ROBOMME_EXPERIMENT_PLAN_20260828.md`；
- 新增 `RoboMMERuntimeAdapter`：前视/腕视 RGB、8-D state/action、初始视频
  memory、chunk execution 与 private evaluator isolation；
- action policy：官方发布的 GroundSG PI0.5
  `symbolic-grounded-subgoal/79999`；
- high-level Planner：Qwen3-VL-4B-Instruct + 官方 GroundSG LoRA；
- 五个条件共享相同 `MoveCube` test episode、动作 horizon 和 PI0.5 checkpoint；
- Planner/VLA 输入不包含 evaluator 状态或在线 oracle subgoal。

### 17.2 三层规划对照

| 条件 | 成功率 | 平均步数 | 性质 |
|---|---:|---:|---|
| 固定通用文本 | `1/5` | `906.8` | Planner-to-VLA 接口基线 |
| 可部署 GroundSG VLM | `4/5` | `434.0` | 动态阶段、视觉点、历史路由 |
| 在线 oracle | `5/5` | `223.8` | 特权上界，不作为 CARVE 成绩 |

VLM 条件保持原有成功 episode 0，并将 episode 2、3、4 从失败转为成功。
episode 3、4 正确选择“抓取 peg，再用 peg hook cube 到 target”的两阶段工具
策略。episode 1 同样产生正确工具策略，但动作执行在 1,300 步内未完成，因此该
失败定位于执行层，而不是高层工具选择。

### 17.3 GroundSG Planner 量化门控

在隔离 Planner 请求队列后，对同一五回合完整复跑 BF16 与 bitsandbytes NF4：

| Planner profile | 成功率 | idle VRAM | 平均 episode P95 |
|---|---:|---:|---:|
| BF16 | `4/5` | `9,120 MiB` | `2.92 s` |
| NF4 | `2/5` | `3,994 MiB` | `4.61 s` |

NF4 显存降低 `56.2%`，但时延为 BF16 的 `1.58x`，并丢失 episode 3、4 两个
BF16 成功。两种 profile 的高层阶段序列在 `5/5` 上一致，但公共 trace 时间点的
grounded coordinate 平均绝对误差为 `3.80 px`。因此 NF4 只作为显存不足时的容量
fallback，不作为默认或“无损”优化。这也说明具身量化不能只测文本语义一致率，
必须经过 grounded-point 与完整闭环准入。

BF16 复跑期间没有外部请求进入 GroundSG Planner，但后半段存在另一个独立 Qwen
进程共享 GPU，因此成功率与语义/坐标结论有效，时延比值按保守共享 GPU 测量报告，
不表述为空卡 kernel-level benchmark。

### 17.4 复杂度感知 Planner 调度

逐 action chunk 调用 VLM 的 BF16 条件在五回合中产生 139 次 Planner 调用。
首先测试统一复用两个 chunk：调用降至 103 次，但 episode 3 从成功变为失败，
说明 grounded subgoal 不能无条件缓存。

CARVE 随后采用复杂度感知调度：单点 grounded subgoal 最多复用两个 chunk；包含
两个视觉落点的空间/工具子目标每个 chunk 刷新。该配置恢复并保持 BF16 的 `4/5`，
同时把总 Planner 调用从 139 降至 115（`-17.3%`），总 wall time 从 492.79 s
降至 457.47 s（`-7.2%`）。在两种配置都成功的 episode 0、2、3、4 上，调用下降
`35.1%`，wall time 下降 `18.7%`。

这个消融说明 Optimize Runtime 的核心不是盲目降低调用频率，而是根据 grounded
输出的控制敏感性选择复用强度。统一复用作为负结果保留；复杂度感知版本通过
success、调用量和 wall time 三项准入。

正式证据入口：

- `results/robomme_vlm_groundsg_bf16_queue_isolated_20260828/`；
- `results/robomme_vlm_groundsg_nf4_20260828/`；
- `results/robomme_vlm_groundsg_adaptive_selective2_bf16_20260829/`；
- `scripts/summarize_robomme_planning_gap.py`；
- `scripts/summarize_robomme_planner_quantization.py`；
- `scripts/summarize_robomme_planner_scheduling.py`。

### 17.5 视频记忆、视觉工具与进度状态 pilot

在 `VideoRepick` 上进一步验证“初始视频记忆 -> 在线 GroundSG Planner ->
typed visual tool -> 冻结 PI0.5”的完整链路。离线记忆由 Qwen3.5-9B NF4 从
32 个均匀采样帧提取对象身份/空间关系；在线阶段使用 Qwen3-VL-4B GroundSG，
VLA 仍为同一官方 PI0.5 checkpoint。所有 deployable 条件均不读取
`grounded_subgoal_online` 或 evaluator 私有状态。

episode 0 的机制消融结果如下：

| 条件 | 成功 | Planner 调用 | 步数 | wall time |
|---|---:|---:|---:|---:|
| 无结构化记忆，逐 chunk Planner | 失败 | 16 | 254 | 48.48 s |
| 仅关系记忆，选择性调度 | 失败 | 8 | 227 | 64.08 s |
| 关系记忆 + 本地视觉 grounding，逐 chunk | 成功 | 24 | 379 | 62.92 s |
| 上述完整链路 + 事件调度 | 成功 | 20 | 383 | 54.05 s |

本地视觉工具不决定任务对象或动作阶段；它只执行 Planner 发起的关系查询，
将“topmost green block”等符号约束校准为当前图像落点。由此，正结果不是
规则脚本替代 Planner，而是 VLM 决策、记忆约束、typed perception tool 与 VLA
primitive 的组合。事件调度在同一成功 episode 上将 Planner 调用 `24 -> 20`
（`-16.7%`），wall time `62.92 -> 54.05 s`（`-14.1%`）。这是当前最直接的
Agentic Harness 与 Optimize Runtime 耦合证据。

episode 1 随后完成了针对性修正。SAM2.1 从初始视频中的 `topmost red block`
持续跟踪同一物理实例，得到最终 `bottom-left red block` 与落点
`[133.57, 113.12]`；24 帧遮挡后发生一次重捕获，满足 tracker admission gate。
在线关系 grounding 由此得到 `<134,113>`，与成功 oracle 的落点一致，但该 oracle
不进入 deployable 链路。

仅修复身份记忆时，Planner 在 184 步过早进入按钮阶段，任务仍失败。增加基于夹爪
本体状态的 `PrimitiveExecutionMonitor` 后，系统只有在观察到闭合抓取与重新张开放置
时才允许 `pick -> put -> press` 推进，episode 1 在 282 步成功。事件调度保持该成功，
Planner 调用 `18 -> 16`（`-11.1%`），wall time `38.36 -> 36.11 s`
（`-5.9%`）。这进一步验证了“高频低成本 Monitor + 低频 VLM Planner”的分层必要性。

episode 2 的失败随后定位为“重复过程缺少物理闭环”，而不是按钮落点问题。新增
`RepeatedProcedureController` 后，每轮 `pick -> put` 必须分别观察到夹爪闭合与重新
张开；三轮完成前禁止进入 `press`。该规则不读取 evaluator 状态，动作与目标仍由
VLM Planner 选择。episode 2 由原来的三轮后 evaluator 失败变为成功。

在统一的严格 Monitor 下，对三个官方 episode 进行逐 chunk 与事件调度配对：

| 调度 | 成功 | 控制步 | VLA 调用 | Planner 调用 | wall time |
|---|---:|---:|---:|---:|---:|
| 逐 chunk | 3/3 | 1,363 | 87 | 87 | 200.79 s |
| 事件调度 | 3/3 | 1,367 | 87 | 75 | 176.67 s |

事件调度保持 `3/3`，Planner 调用下降 `13.8%`，总 wall time 下降 `12.0%`。
episode 2 首次逐 chunk 运行含 28.25 s 的 PI0.5 冷编译异常值，因此计时比较采用同一
服务上的 warm rerun；该排除规则已写入证据包。六段配对视频均通过 H.264/512x256
检查。

需要同时说明可靠性代价：严格 Monitor 在 episode 0 上比早期宽松控制器更保守，
控制步由 383 增至 594。因而论文应把 Monitor 表述为可靠性机制，把事件调度表述为
在同一可靠控制器上的效率优化，不能混合成“所有模块均加速”的结论。

统一证据包：`results/robomme_video_repick_full_monitor_study_20260829/`。三个固定
episode 支撑完整链路和配对运行时结论，但仍不是多 seed 统计成功率或跨 VLA 结论。
早期负消融保留在 `results/robomme_memory_tool_study_20260829/`，episode 1 机制消融
保留在 `results/robomme_video_repick_ep1_monitor_study_20260829/`。

### 17.6 八任务、三条件配对主实验

为避免只依赖 `MoveCube` 与 `VideoRepick` 个案，最终从 RoboMME 四类任务中选择
8 个任务，每个条件运行相同的 80 个 official episodes，共 240 个 rollouts：

| 条件 | 成功 | 95% Wilson CI | Planner calls | 总 wall time |
|---|---:|---:|---:|---:|
| B1 Raw VLM+VLA | `23/80` (`28.7%`) | `[20.0%,39.5%]` | `2141` | `5208.0 s` |
| C2 Agentic Harness | `36/80` (`45.0%`) | `[34.6%,55.9%]` | `2001` | `5178.7 s` |
| C3 Harness + Runtime | `42/80` (`52.5%`) | `[41.7%,63.1%]` | `1188` | `3723.6 s` |

C3 相对 B1 增加 `23.8` 个百分点，22 个 paired failure-to-success、3 个
success-to-failure，paired bootstrap 95% CI 为 `[+12.5,+35.0]` 个百分点，
精确 McNemar 双侧 `p=0.000157`。相对逐 chunk 调用 Planner 的 C2，C3 在成功率
继续提高的同时减少 `40.6%` Planner calls 和 `28.1%` 总 wall time。

这是当前最强的 Agentic 主结果，但仍应称为 RoboMME **八任务子集研究**，因为
只覆盖官方 16 个任务中的 8 个，不应写成完整 leaderboard 成绩。统一证据见：

- `docs/reports/RAL_CORE_EXPERIMENT_REPORT_20260831.md`；
- `results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json`；
- `results/robomme_b1_c2_c3_combined_80ep_20260831/README.md`。

## 18. 后续 benchmark 路线

1. **RoboMME：**八任务 240 回合主实验已完成；只有在统一协议可扩展到剩余八个
   任务时才继续，不再运行零散 pilot。
2. **RoboDojo：**第二 VLA、flow-step 与受控恢复已完成；投稿级下一步是 2--3 个
   自然长程失败任务的多 seed paired study，而不是重复同一故障注入点。
3. **VLM 部署：**优先验证远程/第二 GPU VLM 或事件触发分阶段装载，并重新执行
   Critic admission；单卡视觉 VLM+StarVLA 当前没有通过资源门。
4. **RoboTwin 2.0：**仅作为后续外部有效性扩展，在论文确实需要第三套策略/环境
   时再接入，不抢占自然失败与 VLM 资源隔离实验。
5. **LIBERO-PRO：**已完成低成本鲁棒性对照，不继续无差别增加 episode。

## 19. 组会汇报建议

建议用“问题 -> 系统 -> 证据 -> 边界 -> 下一步”叙事：

1. 冻结 VLA 的持续执行问题，以及 Agentic 带来的额外推理开销；
2. 多速率 Planner/Monitor/VLA/Skill 与权限受控七工具；
3. 可信过程记忆、有界恢复和 executor-consistency；
4. profile-driven Optimize Runtime 与行为准入；
5. 1,200 回合边界、P95 `282.43 -> 54.67 ms`、完整恢复视频；
6. 记忆作为计算路由器，重复任务 wall time 最高 `-73.1%`；
7. 主动说明成功率净提升不显著，并把 RoboMME 表述为后续更匹配 memory 的计划。

不要把汇报中心放在 `180 -> 183`，而应放在系统完整性、有界纠错、调用削减、
实时 VLA 路径和可信记忆路由。

## 20. 简历与面试材料

### 20.1 推荐项目标题

**CARVE-VLA：面向冻结 VLA 的具身智能体执行与高效推理框架**

### 20.2 推荐简历 bullet

- 设计并实现多速率具身 Agent Runtime，集成 VLM Planner/Critic、typed
  tools、任务计划、风险监测、可信过程记忆、有界恢复和 fail-closed 安全停止，
  通过 schema、预算与 evaluator 隔离保证执行可审计。
- 构建 PI0.5 profile-driven 推理优化与准入系统，通过 flow-step 校准、
  `torch.compile` 和 SMVE，将 RTX 4090 上 P95 从 `282.43 ms` 降至
  `54.67 ms`，通过 `45/45` 动作 fidelity 和 80 ms deadline gate。
- 完成 1,200 回合 LIBERO/LIBERO-PRO 配对实验和视频审计；Full Agentic 将
  控制步/VLA 调用减少 `7.2%/8.4%`，并用 trusted memory 与 9B/4B 路由在
  两个长程任务族上保持成功，将重复任务 wall time 最高降低 `73.1%`。

### 20.3 不应写入简历

- benchmark success 显著提升或 SOTA；
- 提出了通用 VLA INT4/INT8 量化算法；
- 已完成多 VLA 大规模闭环、真机或 sim-to-real；
- 80 ms 等价于机器人整机硬实时。

### 20.4 面试高频问答

**为什么冻结 VLA？** 研究目标是模型训练之外的持续状态、权限、恢复、记忆、
deadline 和部署审计；冻结模型也使系统收益与训练收益分离。模型完全没学会的
domain shift 仍需更强 checkpoint 或微调。

**Monitor 是否只是规则？** 它有意是高频确定性风险层；开放语义由低频 VLM
负责。二者分离避免每周期调用大模型。

**成功率只多 3 个，框架是否没用？** 它不证明通用能力提升，但证明有界恢复、
调用削减、安全停止、可信记忆和完整审计；同时诚实暴露一次误触发回归。

**SMVE 是否过于简单？** 单模块不承担全部创新。它的工程价值在 named-view
contract、active-view fallback、fixed-noise fidelity 和 closed-loop admission，
并作为 Optimize Runtime 的确定性 L0 层。

**为什么量化更慢？** batch=1 的短解码中，反量化和 kernel 开销可能超过矩阵
计算节省；所以 CARVE 同时测时延、显存、动作/语义一致和闭环行为。

**与 Harness VLA 的区别？** CARVE 增加多速率 Monitor/VLM 分层、typed
authority、executor consistency、可信过程记忆驱动计算路由，以及与
fidelity/deadline/profile admission 联合的 Optimize Runtime。

## 21. 周末研读顺序

1. 本报告；
2. `docs/architecture/CARVE_COMPLETE_RESEARCH_LOOP.md`；
3. `agentic_vla/session.py`；
4. `agentic_vla/runtime/harness.py`、`monitor.py`、`agent.py`；
5. `agentic_vla/toolchain/`；
6. `agentic_vla/optimization/`；
7. `docs/status/LIBERO_PRO_FULL_STUDY_RESULTS_20260826.md`；
8. `docs/status/CARVE_EFFICIENT_INFERENCE_ABLATION_20260826.md`；
9. `docs/status/FULL_EMBODIED_AGENT_RESULTS_20260827.md`；
10. `results/paper_ready_20260827/README.md`；
11. `docs/archive/experiment_plans/CARVE_ROBOMME_EXPERIMENT_PLAN_20260828.md`。

## 22. 术语表

- **Primitive：**边界清晰、可重试、可验证的机器人操作单元；
- **Harness：**管理模型、工具、权限、预算、状态和失败传播的执行系统；
- **Action chunk：**VLA 一次推理生成的一段连续动作；
- **Safe boundary：**允许切换计划、工具或模型而不破坏物理执行的边界；
- **Fidelity gate：**验证优化前后动作或语义一致性的准入检查；
- **Profile：**绑定模型、硬件、软件栈、backend 和 fallback 的部署配置；
- **SMVE：**Static Masked-View Elision，静态缺失视角无效计算消除；
- **Event-coherent reuse：**语义和物理事件保持一致时才允许缓存复用；
- **Verified-procedure memory：**只保存完成并验证的符号任务过程。

## 23. 复现、审计与交付清单

### 23.1 最小代码回归

当前 RoboMME adapter 的定向回归覆盖 8-D state/action contract、相机与初始视频
记忆、动作 chunk、终止状态和 evaluator 隔离；同时复跑 LIBERO、RoboTwin、
Runtime 和 Toolchain 的相关测试，共 `72 passed`。项目既有完整测试记录为
`258 passed`，两者分别代表“本次改动定向回归”和“此前全项目回归”，不能混写。

```bash
openpi/.venv/bin/python -m pytest -q \
  tests/test_robomme_runtime.py \
  tests/test_carve_libero_runtime_adapter.py \
  tests/test_robotwin_runtime_adapter.py \
  tests/test_carve_runtime.py \
  tests/test_carve_canonical_tool_runtime.py \
  tests/test_carve_toolchain.py \
  tests/test_tool_hang_contracts.py
```

### 23.2 已完成实验的重建入口

- `scripts/build_current_evidence_package.py`：重建统一 CSV/JSON 证据包；
- `scripts/summarize_libero_pro_full_study.py`：重算 1,200 回合主表；
- `scripts/audit_libero_pro_full_study.py`：检查 episode、trace 与视频完整性；
- `scripts/build_carve_paper_results.py`：生成论文结果宏与汇总；
- `scripts/summarize_memory_routed_planner_profile.py`：重算 T8 memory route；
- `scripts/summarize_cross_task_memory_routing.py`：重算 T3 因子化路由。

统一证据包中的 `summary.json` 是数字入口，CSV 是表格入口，原始 episode 目录是
最终审计入口。技术报告、PPT 和简历中的数字应从这些文件生成，避免手工维护多份
不一致数据。

### 23.3 RoboMME 实验入口

环境准入使用 `scripts/run_robomme_admission.py`；真实 PI0.5 固定文本、oracle 与
VLM GroundSG 分别使用 `scripts/run_robomme_policy_admission.py`、
`scripts/run_robomme_oracle_upper_bound.py` 和
`scripts/run_robomme_vlm_groundsg.py`。当前完成 `MoveCube` 五回合机制实验，以及
`VideoRepick` 三个官方 episode 的视频记忆、工具路由、物理过程监测与调度配对
研究；16 任务统一多 seed 扩展仍属于投稿级后续工作。

### 23.4 周末交付物检查

| 用途 | 首选材料 | 阅读目标 |
|---|---|---|
| 组会 | 本报告第 1--10、17--19 节 | 讲清问题、框架、证据、边界和下一步 |
| 简历 | 第 20.1--20.3 节 | 使用可验证数字，不写 SOTA/真机/显著提升 |
| 面试 | 第 14--16、20.4、22 节 | 能解释模块职责、设计取舍和负结果 |
| 论文 | `paper/CARVE-VLA/root.pdf` + 第 4--10 节 | 对齐主表、消融、限制和统计口径 |
| 演示 | `results/libero_pro_representative_videos_20260828/` | 按 README 播放配对正例和边界案例 |

## 24. RoboDojo / StarVLA PI-v3 最新实验

### 24.1 为什么增加这条实验链路

LIBERO-Pro 主实验已经覆盖大规模 paired episode，但只围绕 PI0.5。RoboDojo
接入的目标不是继续追逐 benchmark 数量，而是验证 CARVE 的 VLA adapter、
Agentic authority 和 Optimize profile 能否迁移到第二套真实闭环策略。当前使用
官方 StarVLA PI-v3 推理路径，Agent 不读取 evaluator 私有成功真值。

### 24.2 Optimize Runtime 独立结果

20 对相同输入、同一驻留模型的 flow-step 微基准中，Flow4/Flow2 平均时延分别为
`307.23/235.42 ms`，平均下降 `23.37%`；Flow2 动作 MAE 为 `0.003251`。随后在
`build_tower` 与 `put_bottles_into_dustbin` 各三个闭环 seed 上执行质量门，两个
profile 在两个任务上均为 `3/3`。这证明 flow-step 优化在当前任务样本上没有破坏
成功，但 bottle 任务的 episode frames/VLA calls 并未改善，因此不把单次推理
加速等同于全回合加速。

### 24.3 在线高层 Agent 与受控故障实验

正常 `build_tower` 回合中，在线 Qwen3-VL-4B Planner 生成三阶段计划并通过完整性
校验，计划进入 task ledger，最终官方成功 `1/1`。过程中的四次语义检查均返回
inconclusive，没有产生干预，说明低频高层 Agent 可以在正常轨迹中不破坏 VLA。

受控故障实验在 action step 640 注入 stale-action hold，并固定三组 layout/action
seed。C1 Optimize-only 为 `0/3`；C3 在第一次 no-progress 时执行低成本 fresh
chunk replan，在重复异常时调用一次 Planner，获得通过 schema gate 的恢复决策，
临时使用两次 Flow4/h16 后回到 Flow2/h16，结果为 `3/3`。全部三对均失败转成功，
无成功转失败；精确 McNemar 双侧 `p=0.25`。实验说明 Monitor、升级策略、VLM
authority、恢复计算 profile 和 VLA 执行已经真实耦合，但样本量与故障类型不足以
支撑统计显著或自然分布泛化结论。

### 24.4 Critic 准入和单卡部署边界

48 个 Critic holdout 上，Qwen3-VL-4B BF16 与视觉保留 NF4 均只有 `3/8` 终态
召回，false-stop 为 `0/40`；uniform NF4 进一步降至 `1/8`。因此 Critic 当前只能
提供建议，不能触发 safe stop。视觉保留 NF4 的静态显存约为 `3.27 GiB`，较 BF16
的 `8.27 GiB` 下降约 `60.5%`，但速度更慢；量化只解决部分容量问题。

StarVLA、RoboDojo 和视觉保留 NF4 4B VLM 在 RTX 4090 上首次联合视觉推理 OOM。
这确定了当前部署边界：完整在线 C3 需要远程/第二 GPU VLM 或事件触发分阶段装载。
受控故障 C3 使用固定、已验收且哈希锁定的 Planner replay ticket，因此只能作为
执行机制证据，不能冒充在线 VLM 识别准确率。

证据入口：

- `artifacts/robodojo/starvla_flow_step_microbenchmark_20260902/summary.json`；
- `artifacts/robodojo/build_tower_flow2_h16_ablation_20260902/summary.json`；
- `artifacts/robodojo/put_bottles_flow2_h16_ablation_20260902/summary.json`；
- `artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/summary.json`；
- `artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/report.md`；
- `artifacts/robodojo/critic_admission_v1_20260902/qwen3vl4b_uniform_nf4_admission.json`；
- `docs/status/ROBODOJO_CARVE_PAIRED_RESULTS.md`。

最新相关定向测试共 `50 passed`；完整仓库测试仍有 3 个历史测试在 collection 阶段
引用已经清理的旧脚本，因此只能报告定向回归，不能写成当前 full-suite 全通过。

## 25. 总结

CARVE-VLA 当前已经形成“Agentic Harness + Optimize Runtime”的完整研究原型：
前者解决冻结 VLA 的持续执行、工具权限、记忆与失败生命周期，后者解决多模型
Agentic 调用链的实时动作路径、显存准入和部署可回退性。现有实验足以支撑硕士
阶段的系统实现、高效推理和边界分析。RoboMME 八任务配对实验将 Raw VLM+VLA
的 `23/80` 提升到完整 Runtime 的 `42/80`，并以 `p=0.000157` 支持 Agentic
机制；MoveCube 五回合深度消融进一步说明动态 VLM 规划、视觉落点与工具策略
可以把同一 PI0.5 从 `1/5` 提升到 `4/5`，同时揭示 NF4
在 grounded control 中并非无损；复杂度感知调度又在保持 `4/5` 的同时降低
Planner 调用 `17.3%` 和总 wall time `7.2%`。投稿级结论仍需要扩大 RoboMME 任务/seed 覆盖，
`VideoRepick` 三个官方 episode 又验证了视频记忆、tracker admission、typed visual
tool、物理完成监测、过程控制与冻结 PI0.5 的完整成功链路。统一严格 Monitor 下，
事件调度保持 `3/3`，并减少 Planner 调用 `13.8%`、总 wall time `12.0%`。投稿级结论
仍需要扩大
RoboMME 任务/seed 覆盖。StarVLA PI-v3/RoboDojo 已补上第二 VLA 闭环与受控恢复
证据，但投稿级结论仍需要自然失败条件下的多任务 paired repetitions，并解决在线
VLM 的资源隔离与 Critic 准入；后续不再重复堆叠已饱和的 LIBERO 回合或同一注入点。

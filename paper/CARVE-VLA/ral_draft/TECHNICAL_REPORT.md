# BIND-VLA 技术报告：动作边界上的 Agentic 执行与质量约束推理

更新：2026-09-24。本文是当前六页 RA-L 草稿的**论文写作底稿**，不是另一份已投稿论文。主稿为 [main.tex](main.tex)，逐回合视频与演示出处见 [视频证据索引](VIDEO_EVIDENCE_INDEX.md)，原始实验盘点为 [RAL_EXPERIMENT_INVENTORY_20260923.md](../RAL_EXPERIMENT_INVENTORY_20260923.md)。此前逐日追加的 [Agentic VLA 技术报告](../../../docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md)保留为研发日志；当历史设想与本文冲突时，应优先核查本文所列的代码版本、原始结果和主稿协议，而不是把不同时间的数字拼接。

**命名决定。**当前方法和论文使用 **BIND-VLA**；BIND 是英文普通词，不刻意反推首字母。它有两层准确含义：把语义干预绑定到安全的动作块边界；把效率 profile 的启用绑定到动作保真、资源和任务质量门槛。前者是论文最直接的控制机制，后者是推理优化的准入原则。`CARVE-VLA` 暂留作仓库和历史结果目录名，`CarveRuntime` 等既有代码标识符也不为论文改名而破坏兼容；它们不表示当前主稿方法仍叫 CARVE。BIND-VLA 在文中是围绕 VLA 的系统名，**不是一个重新训练的 VLA 基座**。2026-09-24 的初步公开检索未发现精确的 `BIND-VLA` 论文名称，但这不是全球命名独占保证。

## 0. 使用规则与结论先行

### 0.1 本报告能做什么

本文给出问题定义、方法机制、实现接口、模型与任务协议、全部可用于当前论文的主要正负结果、图表及视频来源、主张边界和逐节写作提纲。后续修改标题、摘要、图、六页正文或扩写学位论文时，可从本文取材料；**数字最终以链接的机器结果为准**，术语和图注最终以实际运行的条件为准。本文不替代原始 `summary.json`、逐回合轨迹、视频或代码提交哈希。长视频不复制进论文目录；[视频证据索引](VIDEO_EVIDENCE_INDEX.md)列出可直接播放的代表片段和所有存续源文件的机器目录。

证据按强度区分：

| 等级 | 含义 | 论文用法 |
| --- | --- | --- |
| A：固定配对 | 先定任务和 episode，再固定条件、模型与预算，含成功与伤害 | 可写限定任务、限定配置的结果；小样本仍不能推出总体显著性 |
| B：开发/扩展汇总 | 真实仿真和配对记录存在，但方法版本或任务选择参与了开发 | 可写探索性趋势和失败机制，不能当冻结最终方法的确认性 SOTA |
| C：独立模型基准 | 真实 checkpoint、录制观测、硬件测时和保真检查 | 只能写该模型/后端的单调用结果，不替代同链路机器人闭环 |
| D：软件/集成检查 | 单元测试、同输入重放、受限工具回合 | 证明契约/路径可运行，不证明任务成功率 |
| E：规划 | 尚未实施或未通过准入 | 只能写未来工作，不进入成果图表 |

**截至本文日期的判断：**系统设计与多条真实仿真链路已经具备可写成完整方法论文的材料；冻结的联合 Harness+Optimize 条件尚未在同一 RoboMME/JAX 闭环上证明“更准且更快”。最可靠的核心故事不是“每个模块都提高成功率”，而是：冻结动作策略外围的语义监督需要**干预权限和动作边界**，记忆既可能救回也可能误导；新增 Agent 计算应由**质量约束的推理配置准入**管理。当前 [论文 PDF](main.pdf)如实遵守这一边界。

### 0.2 研究范围与排除项

- **研究对象：**冻结预训练 VLA 的长程执行组织、VLM/VLA 协作时的错误干预与计算开销；同一架构的动作策略适配接口原则上可支持其他 VLA/WAM。
- **不是：**新的 VLA 基座训练、无数据在线技能自进化、新量化 kernel、跨所有策略零适配、真机安全认证或硬实时保证。当前有实证的动作模型属于 PI0.5 家族；可替换性是软件接口性质，**不是**跨模型效果结论。
- **与前期 Agentic RAG-VLM 的关系：**可供性感知检索、场景关系和反思规划提供研究思想，但本稿的核心实验不把前期论文成绩混入。HAA-RAG/Scene Graph 不应画成当前每个 RoboMME 回合都运行的必经模块。

## 1. 研究问题与论文主线

### 1.1 为什么动作策略之外需要 Harness

给定图像、语言和机器人状态，VLA 输出局部动作块。长程操作仍需记住被遮挡对象的身份、区分“夹爪闭合”与“物体确实被正确抓住”、决定失败后是否继续或重试，以及限制可执行技能和恢复预算。把 VLM 直接放在每个动作块前并不自动解决这些问题：错误语义判断会打断有用动作，频繁视觉推理还可能比动作模型本身更慢。开发中确实出现过 Monitor 将夹爪闭合当作任务完成、提前切换放下阶段的失败，因而方法选择从“增加更多检查”转向“限定谁有权改变下一段动作”。

论文的主问题可写为：**何种可部署证据足以授权高层智能体改变冻结动作策略的下一子目标，改变发生在何时，额外代价如何控制？** 这是系统/执行语义问题，不声称单靠外部 Harness 让 VLA 学会缺失的抓取技能。

### 1.2 适合保留的贡献层级

1. **边界权限契约。** 高频 Monitor 只报告执行风险；VLM 低频解释语义；普通子目标变更只能在已执行动作块后的安全边界，经新观测、技能白名单、任务作用域和预算检查后生效。物理急停另属独立安全链路，不等待 VLM。
2. **有来源的任务记忆与过程回执。** 把公开示教中可追踪的对象身份记录为历史证据，不当作当前目标真值；动作回执记录“某段动作确实执行”，不当作“物体放置成功”。配对实验同时审计救回和伤害。
3. **质量优先的推理配置准入。** 区分一次 VLA 调用、VLM 调用、模型切换和整任务墙钟时间；候选编译、少步、静态无效视角消除和量化先经同输入保真，再经任务质量与资源门槛。底层算子不是本文发明。

前三条是**方法/实现贡献候选**，不是已由同一冻结大样本共同证实的性能主张。现有短稿将第三条写为独立模型调用研究；写长稿时也不能把 PyTorch 内核结果嫁接成 RoboMME/JAX 端到端加速。

## 2. 形式化问题、系统边界与一次调用

### 2.1 观测、动作和上下文

令 `o_k=(I_k^{1:K}, q_k)` 为第 `k` 个动作块前的可部署相机和本体感觉，`l` 为原任务文本，`l_k` 为已准入的当前子目标，`m_k` 为任务作用域的记忆，`b_k` 为剩余 Planner/恢复预算。冻结动作策略在给定 profile `P` 下提出 `A_k=pi_theta(o_k,l_k;P)=(a_{k,1},...,a_{k,H})`。实际执行可以只消费该块的合法前缀；生成长度和执行前缀长度要分开记录。环境/机器人适配器执行后返回新观测及执行回执 `e_k`。官方评估器的 success 可以用于**离线评估**，不能作为在线 Planner 的隐藏输入。

动作空间不是只看数组维度：`ActionSpec`还声明表示、坐标系、夹爪约定、控制频率、归一化与上下界。[contracts.py](../../../agentic_vla/runtime/contracts.py) 中 `InferenceRequest`、`InferenceControls`、`ActionChunk` 和 `RuntimeTrace`分别承载调用、请求控制、动作/耗时和实际生效回执。请求 `max_actions`只是输出/执行前缀截断，不能写成模型天然少生成了动作。`PolicyFamily` 中虽有 VLA/WAM/OTHER，WAM 尚无同级别任务验证。

### 2.2 两个循环及权限约束

```text
原任务 + 新观察 --[policy adapter / admitted profile]--> 动作块
      ^                                             |
      |                                    执行动作 + 新观察
      |                                             |
      +--[边界权限门] <-- [VLM 意图] <-- [Monitor 风险 + 来源记忆 + 回执]
```

Monitor 可逐控制步处理命令、关节响应、帧变化、动作年龄、deadline 余量等**风险信号**；它不是对象关系分类器。VLM Critic/Planner 处理“是否该改变任务语义、下一步是什么”的低频请求；二者可共用 checkpoint，但判断职责不同。Harness/Session 持有任务账本、白名单、预算、episode/generation 和未决恢复；VLA 仍生成连续动作。Optimize Runtime 在模型边界协商实际生效的 profile 并记延迟/显存，不拥有任意改写任务的权限。

**不变式 I1（非抢占）：**普通语义建议不能改写已运行的动作块。块内风险仅提出复核请求；下一块前才可能应用更新。独立物理安全 interlock 可以急停，但通用 `safe_hold` 不等同经认证急停。

**不变式 I2（证据不等于真值）：**夹爪闭合、图像变化、执行完某个 chunk、VLM 回答“完成”、工具函数返回成功、官方任务成功分别是不同事件。只有适用的后置条件或评估协议才能提升相应状态；回执不能越权证明放置完成。

**不变式 I3（失败关闭）：**过期建议、跨 episode/generation 的回包、非法技能、能力不支持或预算耗尽，不下发新动作。受限适配器可保持原目标、请求观察或停止，取决于其已声明策略，不隐含无限自动重试。

**不变式 I4（可审计）：**记录请求与实际生效控制、时间、动作数、Planner/工具意图、拒绝原因、官方结局；不能把“曾触发恢复”计为“救回一个失败回合”。

### 2.3 与代码对应的生命周期

1. `reset(episode, generation)` 清空本回合风险窗口、调用/恢复预算、待核验状态；仅保留作用域/来源仍有效的跨回合记录。
2. `observe` 读取可部署 RGB/机器人状态；不读评估器的隐藏对象位姿或成功标志来指导在线动作。
3. policy adapter 协商配置、生成动作块；环境执行合法部分，回传实际步数、新图像和机器人反馈。
4. Monitor 更新风险，task ledger 更新**已观察到**的过程证据。只有满足事件、周期、时间跨度和预算条件时才请求 VLM。
5. Planner 输出类型化意图（通用接口包括 `continue`、`vla_act`、`run_skill`、`safe_stop`）；权限门先校验输出与边界，再写任务账本或发下一动作。通用 [session.py](../../../agentic_vla/session.py) 明确在 `at_safe_boundary=false` 时拒绝普通 Planner 更新。
6. 若进行了恢复，等待所提动作真正执行、跨过其边界并得到新观察，才做后置核验；状态可以是 confirmed、contradicted 或 inconclusive。不能由同一个 VLM 的自述伪造独立物理证据。

这是一份**共享接口的伪代码语义**，不是声称每个 benchmark 都启用了上面所有工具。RoboMME 正文评估用的是较窄的 GroundSG 语言技能适配；当前 RoboDojo 官方 PI0.5 则默认 `task_only`，不能将任意自由子目标直接传给它。

## 3. Harness 的实现细节

### 3.1 Monitor 与两速率调度

[monitor.py](../../../agentic_vla/runtime/monitor.py) 的 `ExecutionRiskMonitor`按窗口统计命令范数、关节响应和平均像素变化。以 `c_t=||u_t||`、`s_t=||q_t-q_(t-1)||`、`v_t=mean|I_t-I_(t-1)|/scale` 表示诊断量；stall 需要有足够命令而状态/视觉响应低，no-progress 需要低命令低响应持续出现。另以动作年龄、不确定性和 deadline slack 构成归一化风险，输出事件及证据。**这些阈值依 ActionSpec/adapter 校准**：关节绝对目标和末端增量的命令范数不具有同一语义；像素变化可能来自视角和光照，停滞只能请求复核。

[harness.py](../../../agentic_vla/runtime/harness.py) 定义 `EXECUTE_FAST/VERIFY/RECOVER/PLAN_AT_SAFE_BOUNDARY/SAFE_HOLD/STOP` 状态、`EVENT_ONLY/SELECTIVE/EVERY_PRIMITIVE` 的边界策略，以及 failure/recovery/planner 等独立计数。`AsyncAgenticHarnessController`负责生命周期与转移，不直接调用 VLA 或操纵机器人。异步接口表示能提交/接收 Planner 请求，不表示所有部署都能把 VLM 数秒推理隐藏在动作执行后面。

在 RoboMME 历史 C3 和当前 B/C 消融中，选择性规划尝试仅在风险、阶段变化、无效缓存或预算条件满足时调用 VLM；否则延续经新观察校验的子目标，同时 VLA 仍取新图像输出新动作。当前严格 B/C 比较出现质量退化，故不能把此调度写成最终默认准入优化。

### 3.2 VLM 语义能力、工具和恢复

高层 Agent 的职责是选择/检查符号动作，而不是输出关节控制。通用工具类型有 `observe`、`retrieve_memory`、`vla_act`、`run_skill`、`verify`、`safe_hold`、`finish`；实际支持的物理技能取决于后端白名单。[planning.py](../../../agentic_vla/toolchain/planning.py) 的 `EmbodiedTaskPlan`保存 pending/active/retry/confirmed 等状态；[recovery_verification.py](../../../agentic_vla/toolchain/recovery_verification.py) 维护已执行边界后的核验生命周期。工具调用成功只说明工具完成，不说明任务成功。

RoboMME 的 GroundSG Planner 返回任务专用语言技能，需要被合法动作语法检查；无效输出必须拒绝或在受限词表内重新请求，不能传入 VLA。2026-09-24 VideoUnmask 开发修复增加动作块后的阶段收据：只在首目标的放下动作块确实执行后，向下一次 Planner 提供阶段证据；不会在 VLA 推理或动作执行中途强制切换，也不把夹爪重开判为放置成功。[实验升级记录](EXPERIMENT_UPGRADE.md) 给出已知失败 ep43 的复测和此前未用 ep47 的检查。

RoboDojo 当前官方 PI0.5 路径需单独表述。[robodojo_pi05_episode.py](../../../agentic_vla/benchmarks/robodojo_pi05_episode.py) 将 SortingEpisode、任务计划和真实仿真端口组合；`task_only`保留原任务文本，已运行的 `reobserve` 只取得新观测，不能视为有效抓取恢复。现有 Critic 在视觉完成判断上出现误确认，因此默认只是建议，尚不具备可靠的最终语义裁决权。

### 3.3 记忆的来源、权限与失效

记忆至少分为任务账本、观察/身份历史、未验证模型笔记、恢复日志和已验证程序。只有前两类在本文 RoboMME 身份实验中构成直接操作变量；“存在存储组件”不等于“已在所有后端实现长期自主学习”。历史图像/视频进入 Planner 时，应带任务/episode、源帧或视频哈希、目标身份、时间/有效期和不确定性；在线决策需要在当前画面重新定位历史目标，而不是沿用历史像素坐标。

VideoUnmaskSwap 的 B 条件仅在 A 已看到的**同一公开初始示教**上，额外用 SAM2 跟踪颜色块与容器的关联，形成颜色到目标容器的记录；源图、目标关联和哈希可追溯。它不使用测试终局或官方成功信号做在线记忆。提示要求 VLM 在当前画面重新定位，但输出轨迹本身不能证明它没有直接沿用历史坐标；因而“当前帧独立重定位成功”尚不是已验证机制。编译准入失败时不注入提示；身份匹配正确也不保证抓取/放置物理轨迹成功。另一个 VideoUnmask 任务把同种记忆始终开启后出现负迁移，说明默认策略应是**按任务、按证据准入**，而非全局始终开启；相应在线 gate 尚未通过独立闭环效果确认。

历史程序记忆在有限 LIBERO 任务上曾用离线任务成功作晋升依据，故不能改称“无需监督的自进化”。[memory.py](../../../agentic_vla/toolchain/memory.py) 的 `VerifiedProcedureCompiler` 是程序筛选/来源检查，不是自动训练 VLA；RoboDojo 对象记忆的若干检测/候选过滤试验多数仍为离线或 shadow，不能把它们写成短稿主结果。

## 4. Optimize Runtime：算法对象、部署准入与可测成本

### 4.1 三层优化绝不能混记

| 层 | 实际控制变量 | 目标指标 | 主要风险 |
| --- | --- | --- | --- |
| VLA 内核 | flow 步数、`torch.compile`、静态 padding 视角消除、组件量化 | 单调用 P50/P95、显存、动作重放差异 | 少步或量化改变动作，闭环任务失败 |
| VLM 语义层 | 调用时机、上下文复用、BF16/INT8/NF4、模型驻留 | 语义协议/决策、P95、GPU/CPU 内存、切换耗时 | 错误确认、重复规划或 CPU/GPU 搬运抵消收益 |
| 联合调度 | Planner/VLA profile 组合、预算、回退 | **同闭环** success、伤害、整任务 wall、调用数 | 单项优化都通过但组合失败或过早失败显得更快 |

配置见 [optimization/contracts.py](../../../agentic_vla/optimization/contracts.py) 的 `OptimizationProfile`、`PlannerOptimizationProfile`、`SystemOptimizationProfile` 与静态视角契约；[manifest.py](../../../agentic_vla/optimization/manifest.py) 管理来源和验证证据。`CarveRuntime` 会记录请求控制与 adapter **实际接受**的控制、queue age、model/runtime latency 和 deadline miss，见 [runtime.py](../../../agentic_vla/runtime/runtime.py)。若后端忽略某请求字段，不能仅凭配置文件把它记成有效优化。

### 4.2 准入协议

候选 `P` 的基本流程是：接口/动作语义检查 -> 固定噪声、同输入动作重放 -> 预热后稳态调用延迟/显存 -> 同 checkpoint、同任务的闭环质量 -> 回退/部署登记。可用 `admit(P)=C_contract & C_replay & C_resource & C_quality` 表示；其中 `C_quality` 应优先审计失败转成功 `b` 与成功转失败 `c`，再看成本。`Q(P)>=Q(P0)-epsilon` 只是实验预置样本门槛，不是统计学上已证明总体非劣。冷启动和首次编译成本不应被默默算作零。

动作重放至少要固定输入图像、状态、语言、噪声、动作 horizon，并区分首动作 MAE、整块误差、余弦、夹爪一致性和必要的下游行为。代码中某些 endpoint/jerk 字段是**动作空间代理量**，没有机器人正运动学或时间标定时不能翻译成厘米误差或物理 jerk。相同输入近似一致不能证明不同 flow 步数的闭环质量相同。

### 4.3 已测试的技术细节

- **少步 + 编译：**PyTorch PI0.5 LIBERO checkpoint 的七步 eager 对比两步 eager、两步 compile。七到两步改变算法迭代量，不能把 282.43->54.67 ms 全部归功于 `torch.compile`。同为两步时 151.35->66.06 ms 才更接近编译的受控比较，仍应核对计时环境和预热。
- **SMVE（Static Masked-View Elision）：**仅当 adapter 声明某视角为静态 padding，并在每次输入中确认对应 mask 对所有 batch 元素为 false 时，跳过其编码；不是随意删除真实相机。当前 RoboDojo 官方 PI0.5 使用三路真实 RGB，不能直接套用这一优化。两步编译 66.06->54.67 ms 反映该后端的额外收益。
- **VLA 组件量化：**已有语言后层 INT8 路径降低显存但当前稳态 P95 升至 994.74 ms；不能称“量化带来实时性”。更窄的两条 LIBERO 恢复快照分支中，单独 INT8 成功 2/2，但 INT8+SMVE 只成功 1/2，组合被拒绝。两快照不估计总体成功率。
- **VLM 量化：**Qwen3.5-4B 在 30 条录制语义观测上，BF16 与 NF4 的样本内语义判断一致；NF4 已分配显存从 8.46 到 3.08 GiB，但 P95 从 1896.55 到 2909.79 ms。它是低显存档，不是加速档；不等同 RoboMME 使用的 Qwen3-VL-4B GroundSG，也不等同 RoboDojo 当前 4B 的可靠 Critic 已准入。
- **冻结权重 CPU 镜像：**[residency.py](../../../agentic_vla/optimization/residency.py) 保留冻结参数主机副本，减少旧单卡 staged VLM/VLA 来回卸载的冗余拷贝，但付出主机内存并仍需上传/生成；旧 PI-v3 切换诊断不等于当前 JAX 主链路已经获得异步实时执行。
- **联合 smoke：**NF4 Planner 与 SMVE PI0.5 曾在单卡共驻，约 11.22 GiB，但没有向机器人下发动作；只能证明加载/接口集成，不能填补闭环联合验证。

### 4.4 计时定义与“实时”的正确用法

模型单调用 P95 是 45 或 30 个测量样本中第 95 百分位，不是最大时延、硬 deadline 保证或机器人整周期。对**串行**路径，`T_episode`可按环境/IO、VLA、VLM、切换及其他非重叠区间分项；存在异步重叠时不可直接相加。整任务 wall 需包含传感、规划、动作执行、同步、失败重试、模型切换等。若一臂早失败，它的 wall 短不是性能改善；应同时报告全样本成本和双方都成功的配对成本。本文 80 ms 是模型调用 profile 目标，不是整机器人控制回路的承诺。

## 5. 实验设计：三个不可混合的模型链路

| 链路 | 模型、环境和输入 | 主用途 | 禁止的跨链推断 |
| --- | --- | --- | --- |
| RoboMME 主链路 | 官方 GroundSG PI0.5 checkpoint 79999 + Qwen3-VL-4B GroundSG Planner；JAX；前/腕部既有视角；16 动作 chunk；每回合最多 1300 控制步 | Agentic、身份记忆、边界回执与 VLM 调度的真实仿真配对 | 不把 PyTorch 54.67 ms 填入此链路；Raw 已有逐块 VLM Planner，不是“无高层模型的纯 PI0.5” |
| LIBERO/LIBERO-PRO | PyTorch PI0.5 LIBERO checkpoint + Qwen3.5-4B 语义 Planner；本地 40 task x 10 初态 | 广覆盖恢复对照、Agent 开销，另有小规模恢复快照质量门槛 | 不称官方 50 初态 leaderboard 或 RoboMME 的同模型复现 |
| 独立效率 | RTX 4090、`pi05_libero_pytorch`、45 组配对录制观测、固定噪声、horizon 10；Planner 有 30 条另录制语义输入 | VLA 单调用、VLM 精度/显存、后端配置筛选 | 不称 VLM+VLA 同一机器人闭环加速，更不称当前 RoboDojo/JAX 实时 |

另有官方 RoboDojo PI0.5 59999 的整理桌面和语言分类开发线，用于后续能力验证；它**不是**上表 RoboMME checkpoint 的别名，也没有进入当前六页短稿的主实验结论。任务选择和历史材料见 [实验盘点](../RAL_EXPERIMENT_INVENTORY_20260923.md)。

所有机器人任务 success 采用官方环境结果；Planner 自述、工具返回、过程回执不代替官方结局。配对以同任务/episode 初态为单位，记录双方成功 `a`、基线失败方法成功 `b`（rescue）、基线成功方法失败 `c`（harm）、双方失败 `d`。净差 `(b-c)/N`，精确双侧 McNemar 只利用 `b,c`。任务选择、开发轮次与小样本使 p 值不能独立代表新方法的总体效果。若讨论时间，只看双方都成功的子集会产生选择性条件，故必须同时列全回合成本与成功配对成本。

### 5.1 历史八任务 RoboMME：开发与扩展，不是最终冻结主结果

八个官方任务各取 10 个固定 episode，三臂各 80 回合、共 240 个真实仿真 rollout。B1 Raw 使用原 GroundSG Planner 每块调用且无新增记忆/工具；C2 加记忆、grounded 工具和程序/时间控制，但仍每块规划；C3 加选择性调用与验证后的子目标复用。任务/episode 身份配对，但 C3 是开发期间的若干任务专用修订汇总，最后四任务在前四任务研究后才加入。[汇总 JSON](../../../results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json) 与 [逐回合 CSV](../../../results/robomme_b1_c2_c3_combined_80ep_20260831/episodes.csv) 是原始聚合入口。

| 条件 | 官方成功 | Planner/VLA 调用 | 累计回合 wall |
| --- | ---: | ---: | ---: |
| B1 Raw | 23/80 | 2141/2141 | 5208.0 s |
| C2 Harness | 36/80 | 2001/2001 | 5178.7 s |
| C3 + selective | 42/80 | 1188/2169 | 3723.6 s |

C2 对 B1 有 18 rescue、5 harm；C3 对 B1 有 22 rescue、3 harm。相应名义精确 McNemar `p=0.0106` 与 `p=0.000157`，但**不能**作为冻结最终方法的确认性显著性。C3 对 C2 的 Planner 调用 `2001->1188`（-40.6%）、累计 wall `5178.7->3723.6 s`（-28.1%），同时 VLA 调用 `2001->2169` 增加；六例成功优势名义 `p=0.146`。不能把少 Planner 调用改写为 VLA kernel 加速或所有任务一致受益。

任务级成功（每格 `/10`）：

| 任务 | B1 | C2 | C3 | 读法 |
| --- | ---: | ---: | ---: | --- |
| StopCube | 0 | 3 | 2 | Harness 有作用，C3 不再提升 |
| VideoRepick | 2 | 3 | 7 | 值得检验的长程重抓机制，仍属开发线索 |
| RouteStick | 1 | 3 | 5 | 历史正向，后续冻结测试基线接近地板 |
| VideoUnmaskSwap | 2 | 7 | 8 | 身份/遮挡敏感；后续单任务固定记忆测试 |
| BinFill | 7 | 8 | 8 | 原策略已经较强 |
| ButtonUnmask | 0 | 3 | 2 | C3 可能有调度伤害 |
| PickHighlight | 3 | 2 | 2 | 负向/保护任务，不能从主文删去 |
| MoveCube | 8 | 7 | 8 | 接近天花板、易出现随机摆动 |

### 5.2 同模型身份记忆的固定门槛、独立确认和负迁移

此组**只改变结构化身份记忆**，不是完整 Harness 消融。A 与 B 都看到同一个公开初始示教、使用同一冻结 PI0.5/Qwen3-VL、16 步 chunk、相机、每块 Planner、1300 步预算；B 额外收到从该示教编译并准入的颜色-容器身份。运行顺序按 episode 奇偶交替 A-B/B-A，同一服务内比较初态哈希；阶段回执/身份冲突新修复均关闭。开发门槛为 ep30--37；独立确认预选 ep38--53，但官方 test split 只有 0--49，越界 50--53 **在模型 rollout 前**由预检剔除、不按结局补样，因此有效确认是连续 12 个 ep38--49。[开发分析](../../../artifacts/robomme/memory_ab_gate_20260923/analysis.json)、[确认分析](../../../artifacts/robomme/memory_ab_confirm_20260923/analysis.json) 保留排除与逐对结果。

| 数据集 | A 无结构化记忆 | B 有记忆 | rescue/harm | 精确双侧检验 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| VideoUnmaskSwap 开发门槛 ep30--37 | 2/8 | 4/8 | 2/0 | 小样本开发 | 有机制线索 |
| VideoUnmaskSwap 固定确认 ep38--49 | 5/12 | 9/12 | 5/1 | `p=0.21875` | 同任务方向性增益，非显著总体结论 |
| VideoUnmask 转移 ep22--37 | 15/16 | 13/16 | 1/3 | `p=0.625` | 始终开启导致负迁移；默认拒绝 |

确认集的 6 个单目标 episode 中，A/B 为 `3/6->6/6`；6 个双目标中为 `2/6->3/6`。ep39 中 B 的末次 Planner 输出红色目标点约 `(98,93)`并最终成功；A 末次点约 `(80,112)`且失败。B 的输出与正确目标位置一致，但该例不能单独区分“重新理解当前画面”与“复用历史位置”这两种内部原因。ep43 A 成功、B 失败，即便身份顺序正确，物理执行/阶段错误仍可导致失败。B 救回的 ep47 达到 1075 步、68 次 VLM/策略调用，说明 rescue 不等于成本下降。原始 [ep39 A 录像](../../../artifacts/robomme/memory_ab_confirm_20260923/rollouts/VideoUnmaskSwap_ep39_A/VideoUnmaskSwap_ep39_vlm_groundsg.mp4)、[ep39 B 录像](../../../artifacts/robomme/memory_ab_confirm_20260923/rollouts/VideoUnmaskSwap_ep39_B/VideoUnmaskSwap_ep39_vlm_groundsg.mp4) 和 [ep43 反例目录](../../../artifacts/robomme/memory_ab_confirm_20260923/rollouts/)供图文核查。图中不同最终帧不是同步的同一中间状态，不能画成像素级并行反事实。

转移的 16 对同服务 VideoUnmask 中，12 个单目标两臂都成功；四个双目标从 `3/4` 降到 `1/4`。B 的记忆均实际准入，不再是旧 pilot 中“构建了但被 motion gate 全部拦掉”。12 个双方成功配对的 Planner/VLA 调用 `134->100`、wall `152.74->121.41 s` 是**条件性不同轨迹**，且另有示教跟踪构建成本；不能抵消任务伤害。[转移分析](../../../artifacts/robomme/videounmask_memory_transfer_20260924/analysis.json)。

### 5.3 选择性 VLM 调度：成本下降但质量门槛失败

同一 JAX PI0.5、Qwen3-VL 与已准入身份记忆，B 每块 Planner，C 在新观测验证后复用子目标。ep38--48 有 11 个严格同服务配对：B `8/11`、C `7/11`，1 rescue、2 harm。六个双方成功配对的 Planner 调用 `45->29`，wall `66.84->52.13 s`，VLA 调用 `45->44`。ep49 的 C 输出非法动作技能被安全拒绝，B 在**第二次服务启动**后成功，因此排除出严格配对质量/时间总数，不把 C 的格式错误冒充官方任务失败。结论是成本存在潜力，但预设质量门槛拒绝 C，不能称此 profile 已准入。[B/C 审计](../../../artifacts/robomme/memory_selective_bc_20260923/analysis.json)。

### 5.4 冻结版本的双任务 Raw/Harness 压力测试

为检验探索性八任务趋势是否能复现，用同一 GroundSG PI0.5/Qwen3-VL、同服务、相同观察和逐块规划，RouteStick 与 PickHighlight 各取连续 16 个新配对初态。RouteStick 是历史正向候选，PickHighlight 是历史退化风险候选；任务选择**有目的而非随机**。Harness 仅加其现有任务专用规划契约，不注入身份记忆，也不启用已被拒绝的选择性调度。

RouteStick `0/16->1/16`，PickHighlight `3/16->3/16`；合并 Raw/Harness `3/32->4/32`，1 rescue、0 harm、精确双侧 `p=1.0`。双方都成功的三个配对累计 wall `53.43->55.01 s`、Planner/VLA `58->60`，Harness 预算耗尽 2 次而 Raw 1 次。它不能复现强总增益，也没有整链路提速；RouteStick 的接近地板成功率限制了高层模块可发挥的空间。该组是“当前冻结方法”的**保护性压力结果**，不含全部记忆/Optimize 模块。[逐对 JSON](../../../artifacts/robomme/harness_transfer_gate_20260924/analysis.json)。

### 5.5 阶段回执的故障修复：开发个例与未用个例分开

VideoUnmask ep38--45 的 A/B/C 开发矩阵里，A Raw、B 观察式 Harness 均 `8/8`，C 加身份记忆 `7/8`；ep43 C 在 144 步后连续输出问句而非合法技能，触发 fail-closed。新代码在第一目标放置动作块**实际执行后**才生成阶段回执，修复提示限定为合法技能词表。54 项目标回归通过。旧失败 ep43 在同初态开发复测中成功，365 步、23 次 Planner/VLA 调用；它是修复验证，**不是独立 rescue**，轨迹也不足以把收益单独归因于格式提示。另在先做公开示教预检、此前未有对应结果的双目标 ep47，Raw/Harness/Harness+记忆与回执均成功，控制步 `313/313/327`，wall `19.590/19.405/30.510 s`。因此证明阶段回执路径可运行，但未证明总体成功率或效率收益。详见 [实验升级记录](EXPERIMENT_UPGRADE.md)及 [ep47 原始目录](../../../artifacts/robomme/ral_gate_20260924/unmask_ep47_frozen_abc/)。

### 5.6 LIBERO/LIBERO-PRO 的广覆盖边界证据

四个本地十任务 suite（标准 LIBERO-10、Object、Swap、Task perturbation），每任务 10 初态、每条件 400 回合，冻结 PI0.5、固定恢复、Agentic 共 1200 回合。统一官方环境结局，本地协议**不是**每任务 50 初态的官方排行榜。原始 [汇总](../../../results/libero_pro_full_study_20260825/aggregate/study_summary.json)、[审计](../../../results/libero_pro_full_study_20260825/aggregate/audit_report.json)。

| Suite | Frozen | 固定恢复 | Agentic |
| --- | ---: | ---: | ---: |
| LIBERO-10 | 96/100 | 97/100 | 97/100 |
| Object perturbation | 66/100 | 67/100 | 68/100 |
| Swap perturbation | 11/100 | 11/100 | 11/100 |
| Task perturbation | 7/100 | 6/100 | 7/100 |
| 合计 | 180/400 | 181/400 | 183/400 |

Frozen 到 Agentic 为 4 rescue、1 harm，净 +0.75 个百分点，精确 McNemar `p=0.375`。VLA 调用 `16047->14699`（-8.4%），但总 wall `7610.7->8864.6 s`（+16.5%），因为加入了 263 次语义规划。标准十任务基线已 `96/100`，两类较难扰动在同条件没有改善。这个结果最能**证明新增 Agent 计算成本需要管理**，不能拿来写“Agentic 既显著更准又更快”。

### 5.7 独立 PI0.5 模型调用与 Planner 显存实验

VLA 测时硬件 RTX 4090 24GB，45 个配对录制观测，固定噪声，动作 horizon 10，稳态单调用目标 80 ms。原始 [效率汇总](../../../results/carve_efficiency_full_20260826/summary.json) 的 profile 不完全是单因素：V0 与 V3 同时变更 flow 步数、编译和 SMVE；只在相同步数间分析后端变化。

| Profile | 配置 | P50 / P95 (ms) | 峰值 GPU (GiB) | 结论 |
| --- | --- | ---: | ---: | --- |
| V0 | BF16 eager, 7 flow steps | 277.61 / 282.43 | 7.12 | 行为参考 |
| V1 | BF16 eager, 2 steps | 147.13 / 151.35 | 7.12 | 未达 80 ms 目标 |
| V2 | BF16 compile, 2 steps | 63.77 / 66.06 | 6.98 | 后备候选 |
| V3 | BF16 compile + SMVE, 2 steps | 52.20 / 54.67 | 6.98 | 此后端单调用准入；**不是** JAX 机器人全链路 |
| V4 | late-language INT8, 2 steps | 954.80 / 994.74 | 6.36 | 当前 kernel 过慢，拒绝实时档 |

V2--V4 的 45 条动作输出通过各自定义的同输入保真阈值；这**不**表示 V0 七步与 V3 两步逐位一致。另两条 LIBERO 物理恢复快照，V3 成功 `2/2`，INT8 单项 `2/2`但较慢，INT8+SMVE 只 `1/2` 并被拒绝；它们既非大样本，也不包含 VLM Agentic 闭环。

Planner 在 30 条录制语义输入上的 BF16/NF4 结果：显存 `8.46->3.08 GiB`，P95 `1896.55->2909.79 ms`，样本内判断一致。INT8 为约 `4.84 GiB / 6504.45 ms P95`，当前 kernel 拒绝。一次 NF4 Planner + SMVE PI0.5 同驻 smoke 约 `11.22 GiB`但未下发机器人动作。NF4 应称“低显存配置”，不称“VLM 加速”。

### 5.8 RoboDojo 官方 PI0.5：保留的开发线，不并入本文主表

当前官方 RoboDojo PI0.5 checkpoint 59999、ARX-X5 双臂、三路 RGB、14 维绝对关节动作、50 步 horizon、25 Hz；新链路原生 JAX flow10。整理桌面原生开发回合官方 75/100、success=false；语言分类原生 0/100、success=false。一个后续受限 Agent 整理桌面回合也是 75/100、success=false，完成了工具返回、新观测、回执入记忆及即时重新规划，但**不是**与原生回合的同初态配对救回。Critic 对完成状态有高置信误判，开放子目标与自主纠错未准入，新后端 Optimize 未启用。这些结果适合作为工程现状和未来任务，不应替代 RoboMME 主结果或被论文图误标为成功。[旧长报告第 13 节](../../../docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md)含逐日记录与 [RoboDojo 开发说明](../../../artifacts/robodojo/sorting_development_20260915/README.md)及原始视频入口。

### 5.9 RoboDojo 历史受控故障演示与视频选择

为了展示较复杂且视觉效果更清楚的操作场景，[RoboDojo 任务视频集](../../../deliverables/BIND_VLA_VIDEO_SHORTLIST_20260924/README.md)收集了 6 类任务的 8 段头部录像，其中 `build_tower` 的 C1/C3 是受控故障配对。它们来自**历史 StarVLA PI-v3**，不是上节官方 PI0.5 59999；在 640 控制步人为注入 `stale_action_hold`，C3 使用已审计 Planner 决策**回放**并在恢复时切换推理 profile。这是联合恢复机制检查，不是自然任务上的实时 VLM 决策或独立量化收益。原始 [三配对分析](../../../artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/summary.json) 与 [C1](../../../artifacts/robodojo/fault640_c1_v2_build_tower_flow2_seed0_20260902/summary.json)/[C3](../../../artifacts/robodojo/fault640_c3_replay_build_tower_flow2_flow4_seed0_20260902/summary.json) 单回合回执支持以下限定结果：

| RoboDojo 视频/对照 | 模型与条件 | 官方结局 | 可用的叙述 |
| --- | --- | --- | --- |
| `build_tower` C1/C3，三固定布局/种子 | PI-v3、同受控停滞故障；C3 加边界恢复与 profile boost | C1 `0/3`、C3 `3/3`；精确双侧 `p=0.25` | 注入故障下路径可救回，不证明自然故障总体增益；不能单独归因于 Agent 或 Runtime |
| `organize_table` 当前受限 Agent，一个 layout 0 回合 | 官方 PI0.5 59999，原任务指令，三路 RGB | 75/100，`success=false` | 当前链路的运行和部分完成展示，不是成功案例或严格配对改善 |

RoboMME 的 VideoUnmaskSwap `5/12->9/12` 及 RouteStick/PickHighlight `3/32->4/32` 继续留在本报告 5.2/5.4 和短稿数据表；上传视频改用 RoboDojo **不改变原实验主次和统计口径**。若最终论文要以 RoboDojo 正面结果作主张，还缺同一模型、自然任务、冻结初态、预设样本数的配对增益，不可从这些演示片段倒推。

其余已收集任务是视频图库而非可相互相减的统一实验矩阵：官方 PI0.5 的 [语言分类原生](../../../artifacts/robodojo/sorting_development_20260915/classify_native/summary.json)为失败、0/100，[叠碗原生](../../../artifacts/robodojo/official_pi05_admission_20260915/stack_bowls_native/summary.json)为成功、1/1；历史 PI-v3 的 [投瓶 C3](../../../artifacts/robodojo/small_complete_20260914/put_bottles_into_dustbin_C3/summary.json)为 3/3 成功，但本组接受语义干预为 0，[收纳电脑/耳机 B0](../../../artifacts/robodojo/framework_capability_b0_store_laptop_and_headphones_set0_20260909/summary.json)为 0/3。`match_and_pick_from_conveyor` 只发现未完成的 `.tmp.mp4`，无有效 `_result.json`；`imitate_sorting_sequence` 没有本地实验录像。历史整理桌面 [run02 的 100 分](../../../artifacts/robodojo/conservative_closed_loop_20260918/run02/AUDIT_INVALID_RESULT.md)已审计为提前退出造成的无效成功标记，不能用于图库或论文正例。

## 6. 结果应如何解释：机制、归因与失败分类

### 6.1 论文主张到证据的映射

| 可写的限定句 | 直接证据 | 不能追加的延伸 |
| --- | --- | --- |
| “动作块边界区分过程风险和任务完成，并能拒绝无效技能” | 代码门控、54 项目标回归、ep43 开发修复及 ep47 新初态功能检查 | “该修复已显著提高所有长程任务成功率” |
| “公开示教身份记忆在固定 Swap 配对中救回部分回合” | ep38--49 `5/12->9/12`、5 rescue/1 harm、ep39 录像 | “记忆跨任务总是有益”“p<0.05” |
| “盲目跨任务复用可伤害完成率” | VideoUnmask `15/16->13/16`，双目标 `3/4->1/4` | “已训练并验证一个自动选择记忆的最优 gate” |
| “八任务开发中 Agentic 与调度显示潜力” | 240 rollout 汇总 `23/80->36/80->42/80`，逐任务异质 | “一套冻结最终代码在八任务显著 SOTA” |
| “冻结 Raw/Harness 压力测试仅产生有限变化” | RouteStick/PickHighlight 合计 `3/32->4/32`，`p=1` | “Harness 已在新初态稳定提高总体成功率” |
| “减少底层调用不保证总体更快” | LIBERO-PRO 400 配对：VLA -8.4%，wall +16.5% | “Agentic 端到端加速” |
| “特定 PyTorch PI0.5 配置达到约 54.67 ms P95 单调用” | 45 录制输入，V0--V4 效率矩阵 | “RoboMME JAX 闭环达到 54.67 ms”“SMVE 单独带来 5.17 倍” |
| “语义量化可换取显存但可能更慢” | Planner 30 输入 BF16/NF4；NF4 低显存 P95 更高 | “VLM 量化无损且实时” |

### 6.2 将失败转化为可解释的设计选择

- **错误时机：**夹爪闭合提前触发阶段切换，说明 Monitor 对执行事件有观察权、没有任务完成裁决权；普通语义改变迁到下一动作块边界。
- **错误内容：**无效问句或不在技能白名单的 VLM 输出不能下发给 VLA；严格拒绝，再用合法词表修复提示。开发 ep43 的成功复测属于 bug 修复验证，不能当独立成功率收益。
- **错误记忆：**对象身份来源可靠仍可能在多目标阶段反复执行第一目标，导致负迁移；记忆 gate 既要检查来源/任务作用域，也要在执行阶段考虑撤销或停用，相关闭环收益尚未证明。
- **错误成本指标：**某方法提前失败时调用更少、耗时更短，不能叫效率提升。既报告所有回合成本，也报告双方成功配对的条件性成本，并说明各自偏差。
- **错误优化组合：**两种单项配置有利不保证组合有利；INT8+SMVE 的一条快照失败使联合 profile 不准入。只有软件参数存在、录制输入保真通过也不等于机器人质量合格。

### 6.3 有效性威胁

1. **后端异质性：**RoboMME JAX GroundSG、LIBERO PyTorch checkpoint、RoboDojo 官方 JAX 模型不等价；跨链路对比只能做动机，不能计算“同一模型更准且更快”的乘积。
2. **开发选择偏差：**八任务 C3 包含任务专用修订；后续 RouteStick/PickHighlight 按历史行为挑选；Swap ep38--49 虽是固定确认，仍只有一个任务族。名义显著性与泛化证据要分开。
3. **记忆作用机制未完全可识别：**ep39 日志给出正确后续点和成功结局，却无法仅从输出判断 VLM 是借历史身份、历史坐标，还是重新识别当前画面；需要位置扰动或去坐标记忆消融才能区分。
4. **配对初态不保证同轨迹：**两臂首帧一致，后续由于策略随机性、动作变化和仿真数值因素可分叉；成功配对时间比较是系统结果，不是单个模块纯计算差。
5. **小样本尾部与统计：**固定记忆仅 12 对，冻结双任务 32 对，效率调用 45 条；P95 对样本及测量环境敏感，`p>0.05`既不能证明没有效果，也不能用来证明非劣。
6. **示教与记忆权限：**公开初始示教是任务允许的输入，但其预处理和 SAM2 构建代价要说明；不能声称长期在线记忆完全无额外开销，也不能把评估器标签用于在线决策。
7. **部署域差异：**没有真机/硬实时、只验证一个动作策略家族；低显存或 profile 准入仍受具体相机、归一化、动作语义和后端 kernel 限制。

## 7. 从本底稿到论文：建议叙事和逐节内容

### 7.1 标题、中心句和摘要

当前标题 `BIND-VLA: Boundary-Gated Agentic Supervision and Quality-Aware Inference` 明确包含 VLA，并同时指出执行边界和推理质量两个研究对象。BIND 不强行解释成首字母缩写，正文开头说明两种绑定关系即可。不要在标题用“realtime robot control”“universally generalizable”或“reliable SOTA”等当前证据无法支撑的承诺。未来若要把标题升级为不限于 VLA 的 action-policy 系统，先补跨策略实证，不能只靠接口定义。

一句中心句：**BIND-VLA 不是替换 VLA 的更强策略，而是给冻结策略配置带执行权限的语义监督和质量门槛，把“何时干预”与“干预成本”共同纳入机器人闭环。**

摘要应按五个动作写：`长程任务的状态/身份难题 -> 高层 Agent 的误干预和额外代价 -> 边界权限、来源记忆与 profile gate -> 有利和不利的固定结果 + 单调用效率 -> 证据边界`。可正面写八任务探索性趋势，但必须紧邻“跨开发版本”；可正面写 Swap 记忆增益，但同句或下一句写 Unmask 负迁移。不要为让摘要“好看”删除所有反例，否则 Discussion 与摘要会自相矛盾。当前 [main.tex 的摘要](main.tex)已按此逻辑组织。

### 7.2 Introduction 可直接展开的四段

1. **机会：**预训练 VLA 在局部视觉-语言-动作映射上进展显著；长程 manipulation 还需要显式任务状态、对象身份和动作结果核验。举公开示教中被遮挡容器的身份作为具体例子。
2. **矛盾：**外部 VLM/Agent 提供规划和恢复，但错误信号可能打断正常 VLA，频繁语义调用增加时延；开发中夹爪闭合误判是可追溯工程动机，不要写成外部文献的普遍定理。
3. **方法：**高频 Monitor 只输出证据，低频 VLM 提出符号意图，Harness 在 chunk 边界和能力/预算门槛处授权；Runtime 用 replay + 资源 + 质量筛选 profile；动作本身仍由冻结 VLA 生成。
4. **贡献与结果预览：**给出两项机制、一项系统评估原则，再用八任务探索、12 对记忆确认、负迁移、32 对冻结对照与 45 输入独立延迟的**分层证据**说明覆盖范围。不能省略“当前冻结联合收益未确认”。

### 7.3 Related Work 应对标什么，不重复声称什么

参考 [references.bib](references.bib) 的实际 citation key；本报告不替代最后的文献元数据核验。

| 方向 | 可引用的 key | BIND-VLA 应写出的实质区别 |
| --- | --- | --- |
| 冻结 VLA 外部 Agent | `harnessvla` | 同样从系统层控制预训练策略；本文更强调执行信号/语义权限分离和伤害审计，不声称工具、retry、memory 首创 |
| VLM 纠错/异常恢复 | `replanvlm`, `aha`, `santhanam2026`, `gu2026` | 语义重规划早已有之；主张的是**何时有资格**改变下一动作块及其成本，不是发明“VLM 当 Critic” |
| 记忆和长程评估 | `memoryvla`, `robomme`, `calvin` | 前者把记忆融进模型；本文保持模型冻结，外部任务记忆有来源/作用域；只在 RoboMME 局部评估，未跑 CALVIN |
| VLA/机器人推理效率 | `tinyvla`, `pointvla`, `realtimevla`, `realtimevlav2`, `realtimeflash`, `rilaas` | 改架构/训练/流式推理是其他路径；本文筛选已有后端配置并强调模型调用与整任务成本区分，不宣称新 kernel |
| 策略/benchmark 对照 | `pi05`, `openvla`, `liberopro` | PI0.5 是实验动作模型；OpenVLA 在此稿只属背景，不写已测试其 adapter；本地 PRO 只跑十状态子集 |

### 7.4 Method 可直接落成三小节

**M1 Boundary-gated execution。**先定义 `o_k,l_k,A_k,e_k,m_k,b_k` 和 I1--I4，不要一上来列十几个模块。说明 Monitor 的可部署信号与 VLM 语义建议没有同等执行权；只有带新观察、技能合法、任务未过期、预算充足时，才更新 `l_(k+1)`。可给状态图和类型化意图，但要说明 RoboMME 采用更窄 GroundSG adapter。安全停止与普通语义切换分开。

**M2 Provenance-aware memory and recovery。**写公开示教如何产生有来源的身份图、如何在当前帧重定位、如何记录执行但不伪造成功；恢复状态只在动作实际执行并重新观察后核验。用 ep39 正例和 ep43/Unmask 反例解释为何准入条件不能只检查“记忆来源正确”。描述尚未通过在线证明的作用域 gate 为设计/待验证部分，不伪写成消融已通过。

**M3 Quality-aware runtime。**定义 profile 的 checkpoint、backend、精度、flow 步数、mask、horizon、硬件，随后写 contract/replay/latency-memory/closed-loop gate。所有优化作用点、成本指标和失败回退有清晰位置。图文注明 7->2 步改变算法，V2->V3 才主要是 masked-view 比较；机器人任务质量优先于单调用 P95。

一段说明性伪代码（不是所有后端都运行这一精确函数）：

```text
for each fresh observation o_k:
    update monitor evidence; read scoped memory and task ledger
    if event/boundary and call budget allow:
        request VLM semantic proposal
        reject stale, unsupported or unsafe proposal before task-state write
        admit ordinary change only at a completed action-chunk boundary
    select an admitted policy profile P, or fall back to reference P0
    generate VLA action chunk A_k from fresh o_k and admitted instruction
    execute a legal prefix; record actual execution receipt e_k
    if a recovery was pending and its action really ran:
        verify on a later observation; record confirmed/contradicted/unknown
    persist calls, rejections, timings and official outcome separately
```

### 7.5 Experimental Setup 与 Results 的主文顺序

建议正文先写**三条不可混用的模型/后端协议**及 paired success、rescue/harm、wall/P95 定义，再按下列顺序：

1. 八任务探索表（B1/C2/C3）和逐任务表；先提示 C3 混合开发版本，强调异质性，不在第一句写“显著超越”。
2. 固定身份记忆 A/B（5/12->9/12）及跨任务负迁移（15/16->13/16），图中并列救回/伤害，解释边界权限为何需要记忆作用域。
3. B/C 选择性规划拒绝门槛、冻结 Raw/Harness 3/32->4/32 和 LIBERO 180/400->183/400。这些保留在主稿，不用只在补充材料里隐藏。
4. 单调用 V0--V4、Planner BF16/NF4，说明不同后端、不等于整个系统同时验证了更准更快。
5. 最后 Discussion：早期干预、记忆负迁移、少 Planner 调用未必更快、未验证跨模型与真机，收束为可证伪的下一次冻结同链路实验。

若因页数需要精简，优先收窄历史 RoboDojo 开发经历和重复解释，**不要**删掉条件定义、最关键的负结果或模型后端区别。

### 7.6 图表说明与素材来源

| 主稿对象 | 应传达的信息 | 来源/不能画成什么 |
| --- | --- | --- |
| Fig. 1 [architecture_mechanism.tikz](figures/architecture_mechanism.tikz) | 冻结动作循环、证据型 supervisor、chunk 边界、quality-admitted profile 及两速率时间轴 | 可编辑 TikZ；缩略图是真实公开示教帧。不要把整张方法图标为实验成功率图。详情见 [FIGURE_PROVENANCE](FIGURE_PROVENANCE.md) |
| Fig. 2 [paired_outcomes.tikz](figures/paired_outcomes.tikz) | paired rescue 和 harm 同时可见 | 来源为固定 Swap/Unmask 记忆与冻结 Harness 两任务统计；目的选任务非随机平均 |
| Fig. 3 [ep39_demo.png](figures/ep39_demo.png) + A/B 末帧 | 一个任务实例如何因身份信息改变后续目标 | 三帧来自真实仿真视频，但 A/B 是各自终局，不是同一时间步；图注必须保留限定 |
| Fig. 4 [runtime_profiles.tikz](figures/runtime_profiles.tikz) | 同硬件单模型调用 P95 及 80ms 模型目标、INT8 拒绝 | 来自 45 条录制观测；不是整任务 JAX timing |
| Tables I--IV | 历史八任务总表、逐任务表、固定记忆 A/B、本地 LIBERO-PRO | 格子以各 JSON/CSV 为准，不做只保留正例的图表 |

**不能生成仿真“结果图”的替代图片。** 概念示意可以设计/绘图，但真实 RGB 视频、成功与失败、坐标、统计条形必须有日志和文件来源。主稿 Fig. 3 的素材和 outcome 逐项出处见 [FIGURE_PROVENANCE](FIGURE_PROVENANCE.md)。

### 7.7 当前六页草稿与底稿的对齐检查

- Fig. 1 画的是可选的 event-triggered 运行机制；固定身份记忆 A/B 实际**每动作块**都调用 Planner。方法图与该消融的调度不是同一个运行条件，图注和实验协议必须继续区分。
- ep39 可写“B 输出与正确目标一致的点并完成任务”，但若正文说“B 从当前帧重新识别目标”，需要额外去历史坐标/位置扰动证据。当前文本的 `grounds` 一词应在下一轮逐句润色时特别检查。
- Runtime 图展示的是 PyTorch PI0.5 的单调用 P95；它不能直接对应 Fig. 1 所示 RoboMME/JAX 动作循环的时延。写摘要和图注时应保持 `separately` / `independent` 等限定。
- 记忆确认的 `p=0.21875`、冻结双任务 `p=1.0` 和 LIBERO `p=0.375` 不应被改写成统计显著。历史八任务名义 p 值也要带开发版本限定。
- 当前匿名六页 PDF 的图、表、主文本均已成形；是否改标题和品牌名属于表达层决策，不能代替上面三项方法/证据对齐。

## 8. 复现与核查路径

### 8.1 最小论文证据索引

| 论文模块 | 一手结果入口 | 关键代码/协议 |
| --- | --- | --- |
| 八任务 B1/C2/C3 | [combined summary](../../../results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json)、[episodes.csv](../../../results/robomme_b1_c2_c3_combined_80ep_20260831/episodes.csv) | [RAL_CORE_EXPERIMENT_REPORT](../../../docs/reports/RAL_CORE_EXPERIMENT_REPORT_20260831.md) |
| Swap 身份记忆 | [gate](../../../artifacts/robomme/memory_ab_gate_20260923/analysis.json)、[confirmation](../../../artifacts/robomme/memory_ab_confirm_20260923/analysis.json) | [预登记/停止规则](../../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)、[tracker](../../../scripts/track_robomme_video_memory.py) |
| Unmask 转移 | [analysis](../../../artifacts/robomme/videounmask_memory_transfer_20260924/analysis.json) | 同上与 [memory builder](../../../scripts/build_robomme_unmask_swap_memory.py) |
| 选择性调度 | [B/C analysis](../../../artifacts/robomme/memory_selective_bc_20260923/analysis.json) | [runner](../../../scripts/run_robomme_vlm_groundsg.py) |
| 冻结 Raw/Harness | [transfer analysis](../../../artifacts/robomme/harness_transfer_gate_20260924/analysis.json) | [frozen gate runner](../../../scripts/run_robomme_harness_transfer_gate.sh) |
| 阶段回执修复 | [ep43/ep47](../../../artifacts/robomme/ral_gate_20260924/)、[升级说明](EXPERIMENT_UPGRADE.md) | [runner](../../../scripts/run_robomme_vlm_groundsg.py)、[目标测试](../../../tests/test_robomme_execution_feedback.py) |
| LIBERO-PRO | [study summary](../../../results/libero_pro_full_study_20260825/aggregate/study_summary.json)、[audit](../../../results/libero_pro_full_study_20260825/aggregate/audit_report.json) | [评估启动器](../../../scripts/run_libero_pro_full_study.sh) |
| PI0.5 与 Planner 效率 | [full summary](../../../results/carve_efficiency_full_20260826/summary.json)及其 `vla/`、`planner/` 目录 | [profile benchmark](../../../scripts/benchmark_carve_pi05_profile.py)、[fidelity](../../../agentic_vla/optimization/fidelity.py) |
| 仿真视频、示教与转存副本 | [视频证据索引](VIDEO_EVIDENCE_INDEX.md)、[全量文件清单](VIDEO_CATALOG.tsv) | 官方结局需回查对应 `summary.json` 或聚合 `analysis.json`；文件名中的 `success/fail` 仅作定位提示 |

路径是当前 workspace 的相对链接；JSON 内某些历史绝对路径记录旧环境位置，复现实验须先核对 checkpoint revision、依赖、相机/动作语义和服务方式，不能只复制 JSON 反推完整环境。旧 [paper/CARVE-VLA/root.tex](../root.tex) 和已经投稿的旧论文不是本轮 RA-L 底稿，保留不改。

### 8.2 非 GPU 的核查命令

在项目根目录运行；以下只读/编译，不会启动新仿真实验：

```bash
jq '{A_success,B_success,rescues,harms,mcnemar_exact_two_sided_p}' artifacts/robomme/memory_ab_confirm_20260923/analysis.json
jq '.overall' artifacts/robomme/harness_transfer_gate_20260924/analysis.json
jq '{hardware,vla_protocol,vla_rows,planner_rows}' results/carve_efficiency_full_20260826/summary.json
cd paper/CARVE-VLA/ral_draft
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf | grep Pages
```

目标回归的近期记录为 `tests/test_robomme_execution_feedback.py` 54 项通过；需要重跑时应在现有虚拟环境中检查测试依赖和当日代码，不把旧测试计数写成今天全部代码都已验证。重跑 GPU 回合须先冻结 commit/config/checkpoint/episode manifest，不从本报告的示意伪代码直接启动。

### 8.3 视频证据、演示边界和代码快照

[视频证据索引](VIDEO_EVIDENCE_INDEX.md)可直接打开主稿相关的救回、伤害、负迁移、功能修复和 RoboDojo 开发录像；[VIDEO_CATALOG.tsv](VIDEO_CATALOG.tsv)列出 `artifacts/`、`results/` 及 RoboDojo 官方结果目录中当前存续的 4,512 个视频文件。这个数量包括示教、多视角、历史开发和诊断，不是独立回合数。配对结局取同实验 `analysis.json`/`summary.json`，不是视频文件名。若在汇报中截取片段，须保留任务、episode、条件、官方 success 和是否开发复测的字幕或旁注；不能用一个救回剪辑覆盖同组的伤害例。

写作时可把当前源码 `HEAD=6a2926ab7eb84cd94067a08fbd6f5064da693370` 作为**定位点**，但 2026-09-24 工作区存在大量未提交改动，因此该哈希并不单独复现本文所有实验，更不等于每个历史 run 的实际源码。部分结果 JSON 含 source SHA/config fingerprint；复现实验应逐组核对代码、checkpoint、环境版本、任务初态和服务配置，并冻结新的运行快照。

## 9. 未闭合问题和最小后续验证

1. **Agentic 主张：**固定 GroundSG PI0.5/Qwen3-VL、代码、prompt、预算及初态列表；选择一个存在可恢复长程错误且基线不是零的任务，再选一个历史负迁移保护任务。比较 Raw、无记忆 Harness、有准入记忆 Harness，报告官方 success、rescue、harm、合法/拒绝干预与全回合成本。已用开发 episode 不能再次宣称新留出。
2. **Optimize 联合主张：**只有在同一 JAX 或同一 PyTorch 机器人链路、同一 checkpoint、相同任务及 service 配置内，比较 reference 与经过 replay/资源/质量门槛的候选，才能讨论整系统的速度与质量。若后端不支持编译/SMVE，明确换一个适用的候选；不能把旧 PyTorch 数字移植到 RoboMME。
3. **若赶时间不补：**当前六页稿可作为“边界化 Agentic 监督 + 条件性记忆 + 质量约束推理”的完整系统技术论文初稿；摘要、图题和结论继续保留探索性/负迁移/分后端边界，不承诺已证明普适收益或 RA-L 录用。

本报告的用途是让论文**有足够完整、可核查的底层材料**，不是用篇幅替代缺失的独立确认实验。后续写作可以更有力、更精炼，但不能使数字、模型链路或证据等级悄悄变化。

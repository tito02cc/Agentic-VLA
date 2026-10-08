# CARVE-VLA 已完成工作总览

更新时间：2026-09-02

> 2026-09-16导航说明：下文保留9月2日历史快照，不是最新部署完成度。
> 当前使用官方RoboDojo π0.5开发整理桌面/语言分类任务；Critic及子目标执行尚未通过，
> 没有新两任务完整自主Agent结果。请优先阅读
> [主技术报告第0、13、16节](../reports/AGENTIC_VLA_TECHNICAL_REPORT.md)。
> 本次链接检查发现下方历史正文有11个目标不可访问，主要为旧恢复/联合实验、
> Robosuite与中期材料路径；保留原记录，不据此认定原始数据仍完整可用。

本文保留该阶段完成度、实验结论和证据边界。实验过程记录见`EXPERIMENT_LOG.md`，
后续综合介绍持续维护主技术报告，不另增带版本号的完成度文档。

### 2026-09-02 当前快照

项目已经完成可运行的 **Agentic Harness + Optimize Runtime** 研究原型，并在
PI0.5/LIBERO-Pro 与 StarVLA PI-v3/RoboDojo 两条真实闭环链路上完成验证。现阶段
不再需要继续堆叠同类 LIBERO 回合，后续实验应集中回答两个尚未完全解决的问题：

1. Agentic 机制能否在更多自然、可观察的长程失败中稳定提升任务结果；
2. 独立 VLM Planner/Critic 如何在单张 RTX 4090 或远程部署条件下低频接入，而不
   阻塞 VLA 动作关键路径。

当前最强证据如下：

- LIBERO/LIBERO-Pro：三方法、40 个任务条件、1,200 个 paired episode；Agentic
  相对 Frozen VLA 产生 4 个失败转成功、1 个成功转失败，成功率差异不显著，但
  控制步与 VLA 调用分别减少 `7.2%` 和 `8.4%`。
- RoboMME：8 个任务、每条件 80 回合，Raw VLM+VLA、Agentic Harness、完整
  Runtime 分别为 `23/80`、`36/80`、`42/80`；完整 Runtime 相对 Raw 提升
  `23.8` 个百分点，精确 McNemar `p=0.000157`，并相对逐 chunk Harness 减少
  `40.6%` Planner 调用和 `28.1%` 总 wall time。
- PI0.5 Optimize Runtime：P95 从 `282.43 ms` 降至 `54.67 ms`，`45/45` 动作
  保真度通过，80 ms deadline miss 为 `0`。
- 可信过程记忆：两个长程任务族保持成功，重复任务 wall time 最高降低 `73.1%`。
- StarVLA PI-v3/RoboDojo：Flow2 相对 Flow4 的 20 对驻留推理平均时延下降
  `23.37%`，动作 MAE `0.003251`；两个任务的闭环质量门共 `6/6` 对 `6/6`。
- RoboDojo 受控停滞：无 Agent 恢复 `0/3`，两阶段 Agentic 恢复 `3/3`，每回合
  仅调用 1 次语义 Planner，并临时使用 2 次高保真 Flow4 恢复计算。该结果证明
  恢复机制有效，但样本量仅 3 对，McNemar `p=0.25`，不能写成自然成功率提升。

关键边界同样明确：在线正常回合仅完成 `1/1` 的 Planner 编排验证；VLM Critic
在 48 个 holdout 上终态召回不足，尚未获得 safe-stop 权限；StarVLA、RoboDojo
与 4B VLM 在 24 GB 显存上首次联合视觉推理 OOM，因此不能声称独立 4B VLM 与
StarVLA 已在单卡稳定共驻。

## 1. 项目定位

CARVE-VLA 研究冻结 VLA 在长程机器人操作中的两个部署问题：

1. **闭环可靠性**：VLA 会出现停滞、误抓、状态陈旧和恢复失败，需要具身
   Agentic 系统进行监测、恢复、验证和安全停止。
2. **高效推理与轻量化**：Agentic 重规划和恢复会增加模型调用，需要降低
   VLA 的单次计算成本、无效计算和系统调用数量，并建立可追溯的部署门控。

系统由两个边界明确的部分组成：

- **CARVE Agentic Harness**：监测、记忆、重规划、物理恢复、验证和安全停止。
- **CARVE Optimize Runtime**：profile 校准、编译、计算裁剪、量化实验、
  fidelity/deadline/closed-loop admission 和 trace。

当前闭环模型包括 PI0.5 和 StarVLA PI-v3：前者承担 LIBERO-Pro 主实验，后者通过
官方 RoboDojo 闭环验证第二 VLA adapter 与 flow-step 优化。OpenVLA 仅用于验证
跨模型接口和通用低比特方案边界，不作为当前主 benchmark。

2026-08-27 已完成 VLM Planner、符号过程记忆、RGB-D 解析技能、PI0.5、
Critic、物理恢复和 Optimize Runtime 的统一闭环验证。完整实现与结果见
[完整具身智能体系统与高效推理验证](FULL_EMBODIED_AGENT_RESULTS_20260827.md)。

## 2. 总体完成度

| 研究模块 | 状态 | 当前结论 |
|---|---|---|
| Agentic Harness 工程框架 | 已完成研究原型 | Planner/Critic、Monitor、任务计划、工具、记忆、有界恢复和审计闭环已接通 |
| PI0.5 接入与动作契约 | 已完成 | 本地 PyTorch、WebSocket 和 LIBERO 闭环均已运行 |
| 物理恢复与安全停止 | 已完成 | 支持 stall 恢复、结果验证和不支持事件的 fail-closed |
| VLA 高效推理 | 已完成 | low-step、编译、SMVE 均有可复现正结果 |
| 运行时/计算轻量化 | 已完成 | 减少 flow steps、无效视觉计算和非必要 VLA 调用 |
| Agentic 与优化耦合 | 已完成机制验证 | LIBERO-Pro 1,200 回合与 RoboDojo 三对受控恢复均有可审计结果 |
| 部署 profile admission | 已完成 | checkpoint/hardware/profile/evidence 绑定及 fallback 完成 |
| 跨 checkpoint/adapter | 已完成系统验证 | PI0.5 LIBERO/DROID 输入路径通过 |
| 第二 VLA 家族闭环 | 已完成 | StarVLA PI-v3 已在 RoboDojo 完成原生策略、flow-step 和 Agentic 恢复验证；OpenVLA 保留接口验证 |
| 模型参数轻量化 | 已完成定向验证 | PI0.5 late-language INT8 与 Planner BF16/INT8/NF4 均完成；量化 profile 受内核版本和闭环门控约束 |
| 新 benchmark | 已完成分层验证 | LIBERO-Pro 完整 1,200 回合；RoboMME 完成 240 回合三条件配对；RoboDojo 完成第二 VLA 与受控恢复验证 |
| VLM 语义准入 | 部分完成 | Planner 编排已在线跑通；Critic 终态召回不足，safe-stop authority 未准入 |
| 真机实验 | 未完成 | 当前证据来自真实物理仿真和部署运行时 |

## 3. 已完成的 Agentic Harness

### 3.1 系统模块

| 模块 | 已实现功能 |
|---|---|
| Policy/Action Contract | 模型能力协商、动作维度、坐标系、夹爪语义和归一化检查 |
| Execution Monitor | 使用 RGB、proprioception、动作历史、action age 和 deadline slack |
| Joint Controller | 在缓存复用、VLA、重规划、物理恢复、planner 和安全停止间选择 |
| Recovery Memory | 保存恢复预算、历史结果和恢复 session |
| Critic/Retry | 对 stall 等事件执行有限次重规划和 prompt retry |
| Physical Recovery | stabilize、retract、lift、reobserve、verify、replan |
| Safe Stop | 不支持事件、恢复验证失败或预算耗尽后停止提交动作 |
| Runtime Trace | 记录触发证据、控制模式、profile、fallback、时延和 deadline miss |

控制器不使用物体位姿或任务成功真值进行部署决策；仿真器真值仅用于实验标签。

### 3.2 PI0.5 Recovery Challenge

真实 LIBERO MuJoCo 中恢复三个精确状态，对比继续执行、频繁重规划、prompt
retry 和物理恢复：

| 策略 | 成功 | PI0.5 调用 | 安全停止 |
|---|---:|---:|---:|
| Frozen continuation | `2/3` | `113` | `0` |
| Frequent replan | `2/3` | `452` | `0` |
| Prompt retry | `2/3` | `508` | `0` |
| Physical recovery | `2/3` | `251` | `1` |

已验证结论：

- 两个受支持 stall 均执行 12 个有界恢复动作并通过状态响应验证。
- 不支持的 `stale_action` 在动作执行前安全停止。
- 频繁 replan/prompt retry 未增加三状态成功数，却分别使用 continuation 的
  约 `4.0x/4.5x` VLA 调用。
- 独立在线 T6 哨兵完成 monitor -> recovery -> verify -> PI0.5 replan -> success。

该实验验证恢复生命周期、计算成本和安全边界，不作为 benchmark-wide 成功率。

证据：

- [实验报告](../../results/carve_pi05_recovery_challenge_20260719/REPORT.md)
- [结果图](../../results/carve_pi05_recovery_challenge_20260719/recovery_challenge.png)
- [机器可读结果](../../results/carve_pi05_recovery_challenge_20260719/summary.json)
- [在线恢复视频](../../results/carve_pi05_recovery_challenge_20260719/online_controller/videos/task6_trial0_success_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.mp4)

## 4. 已完成的高效推理与轻量化

### 4.1 Flow-step 计算轻量化

T6/T8/T9 共 15 个配对状态：

| Profile | 成功 | 单次 VLA | VLA 时间/回合 | 总时间/回合 |
|---|---:|---:|---:|---:|
| 7 flow steps, commit 10 | `14/15` | `366.90 ms` | `11.15 s` | `21.27 s` |
| 2 flow steps, commit 10 | `14/15` | `148.96 ms` | `4.44 s` | `14.40 s` |

两步配置保持配对成功数，单次 VLA 加速 `2.46x`，回合总时间降低 `32.3%`。

### 4.2 编译优化

RTX 4090、PI0.5、2 flow steps、commit 10：

| Profile | P50 | P95 | Miss@80ms | 峰值显存 | 决策 |
|---|---:|---:|---:|---:|---|
| Eager BF16, 7 steps | `277.61 ms` | `282.43 ms` | `100%` | `7.12 GB` | behavioral reference |
| Eager BF16, 2 steps | `147.13 ms` | `151.35 ms` | `100%` | `7.12 GB` | deadline rejected |
| Compiled BF16, 2 steps | `63.77 ms` | `66.06 ms` | `0%` | `6.98 GB` | accepted fallback |
| Compiled BF16 + SMVE, 2 steps | `52.20 ms` | `54.67 ms` | `0%` | `6.98 GB` | realtime default |

### 4.3 Static Masked-View Elision

SMVE 根据 adapter 的命名视图契约，在每次调用确认目标 mask 全假后，跳过
padding right-wrist view 的 SigLIP 编码和 prefix-KV 构建。active view 会
立即拒绝该 profile 并进入已验收的普通 compiled BF16 fallback。

- replay fidelity：`45/45` 通过；
- 最差 chunk MAE：`0.00232`；
- 最差 cosine：`0.999897`；
- gripper agreement：`1.0`；
- T8/T9 闭环：SMVE `8/10`，普通 compiled BF16 `7/10`，只作为非劣证据；
- runtime P95 相对普通 compiled BF16 降低 `16.6%`。

### 4.4 Agent/VLA 共卡压力

单张 RTX 4090 同时运行 Qwen3.5-4B 视觉请求与 PI0.5：

| Profile | PI0.5 P50/P95 | Miss@80ms | Fidelity |
|---|---:|---:|---:|
| Compiled BF16 | `81.12/91.85 ms` | `72.8%` | `10/10` |
| Compiled BF16 + SMVE | `65.93/75.57 ms` | `0.6%` | `10/10` |

这组实验说明 Agentic/VLM 共卡负载会使空闲时可用的 profile 失去 deadline，
SMVE 在相同负载下保留了有界关键路径。

证据：

- [VLM/VLA contention gate](../../results/carve_optimize/VLM_VLA_CONTENTION_GATE.md)
- [deployment profiles](../../results/carve_optimize/deployment/)
- [统一机器可读结果](../../paper/CARVE-VLA/generated/runtime_results_summary.json)

## 5. Agentic 与 Optimize Runtime 的联合证据

在完全相同的 T6/T9 stall 状态、固定噪声、相同 2-step/commit-10 和物理恢复
流程下，只替换 PI0.5 execution profile：

| Profile | Exact success | Recovery verified | Runtime P95 | Miss@80ms |
|---|---:|---:|---:|---:|
| Eager BF16 | `2/2` | `2/2` | `166.26 ms` | `236/236` |
| Compiled BF16 | `2/2` | `2/2` | `65.75 ms` | `0/239` |
| Compiled BF16 + SMVE | `2/2` | `2/2` | `54.50 ms` | `0/247` |

三种 profile 保持相同精确状态结果。Compiled BF16/SMVE 相对 Eager 将
runtime P95 降低 `60.5%/67.2%`；SMVE 相对普通 compilation 进一步降低
`17.1%`。这是当前 Agentic Harness 与 Optimize Runtime 并非简单拼接的直接证据。

每种 profile 的单回合在线结果为 `1/1、0/1、1/1`，只用于暴露长程轨迹的
数值敏感性，不用于 profile 能力排序。

证据：

- [联合实验报告](../../results/carve_pi05_agentic_optimize_pair_20260719/REPORT.md)
- [联合实验图](../../results/carve_pi05_agentic_optimize_pair_20260719/agentic_optimize_pair.png)
- [联合实验结果](../../results/carve_pi05_agentic_optimize_pair_20260719/summary.json)

## 6. 模型级轻量化进度

### 6.1 已完成的量化实验

| 模型/profile | 显存变化 | 时延变化 | Fidelity/闭环 | 决策 |
|---|---:|---:|---|---|
| PI0.5 W8A16 | `6.98 -> 6.56 GB` | P95 `67.40 -> 71.72 ms` | replay 通过，闭环丢失一次恢复成功 | rejected |
| PI0.5 late-language INT8 | `6.98 -> 6.36 GB` | P95 `67.40 -> 71.61 ms` | `45/45` replay，T6/T9 `2/2` | admitted low-memory tier |
| PI0.5 late-language INT8 + SMVE | `6.98 -> 6.36 GB` | P95 `67.40 -> 58.71 ms` | `45/45` replay，但 T9 闭环失败 | rejected composition |
| OpenVLA INT8 | `14.42 -> 7.76 GB` | P50 `303.48 -> 1533.96 ms` | exact action `50%` | rejected |
| OpenVLA NF4 | `14.42 -> 4.41 GB` | P50 `303.48 -> 748.35 ms` | exact action `10%` | rejected |

这些实验形成了一个可部署的 PI0.5 低显存档，但它不是实时默认档，也不是
新的参数量化算法。更重要的是，INT8 与 SMVE 分别通过并不保证组合后闭环通过。

### 6.2 Planner 量化消融

Qwen3.5-4B 在相同 30 组配对视觉事件、相同 prompt 和解码预算下：

| Profile | 模型显存 | P95 | 配对语义决策一致率 | 决策 |
|---|---:|---:|---:|---|
| BF16 | `8.46 GiB` | `1.90 s` | `100%` | latency default |
| INT8 | `4.84 GiB` | `6.50 s` | `100%` | current kernel rejected |
| NF4 | `3.08 GiB` | `2.91 s` | `100%` | low-memory tier |

NF4 相对 BF16 减少 `63.6%` 模型显存，但 P95 增加 `53.4%`。因此它解决的是
单卡共驻容量，不作为 Planner 实时加速。INT8 在当前 bitsandbytes kernel 下既
慢于 BF16，也大于 NF4，不进入部署 profile。

### 6.3 当前准确定位

已完成：

- 推理计算轻量化；
- 视觉计算轻量化；
- 系统调用轻量化；
- deadline-aware 部署优化。

已新增：

- 通过 action fidelity 和 T6/T9 闭环门控的 late-language INT8 profile；
- 共享 GPU 系统门控：该低显存档在连续/事件触发 P0 负载下分别有
  `95.4%/19.4%` 的 80 ms miss，因此只允许单模型或独立 GPU 部署；
- 可组合 INT8+SMVE backend 及其闭环否决证据。

下一阶段不再扩大量化组合搜索。若继续模型压缩，只接入具有优化 CUDA kernel
的 action-aware backend，并复用现有 replay、语义、闭环与共享 GPU admission。
当前结果支持“完成了可部署的定向轻量化验证”，不支持“提出了新的量化算法”。

## 7. 跨模型与跨适配器验证

### 7.1 PI0.5 DROID adapter

- Compiled BF16：P95 `66.22 ms`，fidelity `10/10`；
- Compiled BF16 + SMVE：P95 `57.34 ms`，fidelity `10/10`；
- commit 15 因 endpoint L2 `0.18120` 超过门限被拒绝。

该结果证明 PI0.5 内跨 checkpoint/input adapter 可移植性，不是 DROID 任务成功率。

### 7.2 OpenVLA 第二模型家族

- 完成 autoregressive single-action capability/action contract；
- BF16 P50/P95：`303.48/310.92 ms`；
- 预热 LLM compilation：`224.25/231.84 ms`，`10/10` exact action；
- profile preparation `200.19 s`，仍有 `100%` 的 100 ms deadline miss；
- INT8/NF4 作为显存导向负结果保留，不进入闭环。

该结果支持跨 VLA 家族的框架接口，不支持“所有 VLA 均已实时加速”的声明。

## 8. 已记录并否决的方向

| 候选 | 正面现象 | 否决原因 |
|---|---|---|
| PI0.5 W8A16 | 显存略降、replay fidelity 通过 | 闭环恢复退化且时延增加 |
| OpenVLA INT8/NF4 | 显存显著降低 | 推理更慢、exact action gate 失败 |
| 全异步预取 | deadline miss 相对降低 `89.4%` | T8/T9 成功从 `7/10` 降到 `5/10` |
| 交替异步 | 部分降低阻塞 | 成功仍为 `5/10` |
| Prefix consistency enforce | 可计算时序一致性 | AUC `0.563`，没有预测价值 |
| Semantic VLM intervention | 短标签可降低 VLM 时延 | severe-event recall 最高仅 `60%`，保持 shadow-only |

负结果不是未完成代码，而是 deployment admission 的组成部分：候选只有同时
通过动作保真度、时延和闭环门控才允许进入控制系统。

## 9. 补充的紧凑策略流水线

项目还保留了一套 robosuite Stack 的真实数据采集、紧凑策略训练和部署验证：

- 采集 `100` 条成功 demonstration；
- 训练 RGB+state action-chunk compact policy；
- 在 3 cm mid-nudge 下，Agentic no-light 与 LightSafe1 均为 `9/10`；
- full policy calls 从 `127.5/episode` 降到 `69.7/episode`，减少 `45.3%`；
- clean LightSafe1 为 `10/10`，并保留数据、checkpoint、trace 和视频。

该组实验验证端到端紧凑策略 pipeline 和调用复用机制，但模型不是 PI0.5，
因此只作为补充证据。

证据：[robosuite compact pipeline](../../results/robosuite_stack_e2e_v2_20260611/)

## 10. 证据等级与历史结果

### A 级：当前保留原始结果，可直接复查

- PI0.5 flow-step、compile、SMVE 和 replay fidelity；
- T8/T9 paired closed-loop；
- Recovery Challenge 和 Agentic-Optimize pair；
- deployment manifests、policy-call traces 和视频；
- DROID adapter gate、OpenVLA gate、量化及异步负结果；
- robosuite compact-policy 数据、训练与评测结果。
- StarVLA PI-v3/RoboDojo flow-step、在线 Planner、受控恢复、Critic admission 与视频。

### B 级：历史背景，不作为当前主证据

- 历史 LIBERO-10：baseline `180/200`，Agentic `185/200`；
- 历史 B0-VLA-Light：Task1/5/7 `13/15` 保持成功，full calls 降低 `36.8%`。

上述历史结果的原始 rollout 目录在早期清理时未保留，只能作为研究过程背景，
不能用于新的可复现主张。

## 11. 当前可以对外声称的贡献

1. 构建了冻结 VLA 的持续执行 Harness，将 VLM Planner/Critic、typed tools、
   结构化任务计划、高频风险监测、物理恢复、验证和 fail-closed 安全停止纳入
   同一个具有预算与 trace 的执行生命周期。
2. 构建了带写入资格和 executor-consistency gate 的 verified-procedure
   memory；记忆不仅复用已验证任务结构，还作为计算路由器跳过重复大模型规划。
3. 构建了 9B novel-task Planner、可信记忆和 4B Critic/fallback 的自适应模型
   路由。在两个任务族、7 个 memory-routed 官方状态上保持任务成功；T3 最终
   路线相对 Direct 9B 将平均回合时间降低 `73.1%`。
4. 构建了 profile-driven VLA Optimize Runtime，统一管理模型能力、动作契约、
   fidelity、deadline、闭环验收、fallback 和硬件/checkpoint/软件栈绑定。
5. 在 PI0.5 上通过 flow-step calibration、compilation 和 SMVE 实现可复现的
   推理计算轻量化；独立消融将 P95 从 `282.43 ms` 降至 `54.67 ms`，并通过
   `45/45` 动作 fidelity gate。
6. 在相同 MuJoCo 恢复状态中保持 Agentic 行为结果，同时将 runtime P95 从
   `166.26 ms` 降至 `54.50 ms`；完整 1,200 回合研究中 PI0.5 P95 为
   `59.54 ms`，`99.946%` 的调用满足 80 ms 目标。
7. 通过量化、异步和 semantic intervention 的正负结果建立“优化必须经过行为
   准入”的部署原则：开放环误差、显存或单次时延不能单独决定机器人 profile。
8. 在第二 VLA 家族 StarVLA PI-v3 上完成官方 RoboDojo 闭环：Flow2 配对推理
   平均时延降低 `23.37%`，两个任务的闭环质量门保持成功；三对受控停滞均由
   两阶段 Agentic 恢复转为成功。

上述贡献的独特性来自三部分的联合，而不是宣称每个模块均为首次提出：持续执行
Harness、可信过程记忆驱动的模型路由，以及经过闭环门控的高效推理 Runtime。
完整 framing 见
`docs/reports/CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md`。

## 12. 当前不能声称的内容

- 不能声称 Agentic 在大规模 benchmark 上具有统计显著成功率提升。
- 不能声称已经提出并验证新的通用量化算法。
- 不能声称 PI0.5 模型参数或存储已经显著压缩。
- 不能把 model-call deadline 表述为完整机器人硬实时保证。
- 不能声称已在多个 VLA 家族上完成与 PI0.5 同规模的 benchmark 验证；StarVLA
  当前属于第二 VLA 的小样本闭环与机制验证。
- 不能声称在线 VLM Critic 已通过终态判别准入或拥有 safe-stop authority。
- 不能声称独立 4B VLM 与 StarVLA 已在单张 RTX 4090 稳定共驻。
- 不能声称已经完成真机实验。

## 13. 中期汇报结论

当前已经达到毕业论文实验整合阶段：框架、主要推理优化、真实 MuJoCo 恢复
实验、1,200 回合 LIBERO-Pro 研究、Agentic/Optimize 联合实验、VLA/Planner
定向量化、第二 VLA 的 RoboDojo 闭环、负结果和证据边界均已形成。开题中的
高效推理方向已经覆盖计算裁剪、编译、视觉计算消除、模型量化和系统调用优化。

中期后优先级：

1. 统一毕业论文、图表、视频和简历中的证据口径；
2. 面向投稿补自然失败条件下的多任务 Agentic paired repetitions；
3. 验证远程/第二 GPU VLM 或事件触发分阶段调度，解决单卡资源冲突；
4. 不再重复同配置 LIBERO-Pro 或同一 RoboDojo 人工故障。

## 14. 项目入口

- [中期完整报告](../reports/CARVE_VLA_MIDTERM_REPORT_20260717.md)
- [中期投屏稿](../reports/CARVE_VLA_MIDTERM_PRESENTATION_20260717.md)
- [中期问答](../reports/CARVE_VLA_MIDTERM_QA_20260717.md)
- [Runtime 架构](../architecture/CARVE_RUNTIME_ARCHITECTURE.md)
- [当前执行计划](../plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)
- [实验日志](EXPERIMENT_LOG.md)
- [论文机器可读结果](../../paper/CARVE-VLA/generated/runtime_results_summary.json)
- [当前论文](../../paper/CARVE-VLA/root.pdf)

## 15. Canonical LIBERO-Pro 三方法实验

The minimum official-simulator study is complete on Object T8/T9 with Frozen
VLA, Fixed Recovery and Full Agentic, three paired initial states per task.
Every method achieves `3/6`; Full Agentic preserves all successful trajectories,
reduces control steps by `17.1%` and VLA calls by `18.3%`, and converts the three
persistent failures to bounded safe stops. Its six Qwen calls average 9.63 s,
while the PI0.5 path records `0/214` 80 ms deadline misses. The result supports
integration, non-interference, bounded failure handling and realtime VLA-path
claims, but not task-success uplift. Full evidence is in
`docs/status/LIBERO_PRO_CANONICAL_TRIAD_GATE_20260825.md`.

## 16. Full LIBERO-Pro paired study

The earlier three-state stop decision was superseded at the user's request by
a complete thesis-scale run. The final matrix contains all ten tasks and ten
paired states from Standard LIBERO-10 and the Object, Position Swap and Task
Logic LIBERO-Pro suites under Frozen VLA, Fixed Recovery and Full Agentic:
1,200 real simulator episodes in total.

- success: Frozen `180/400`, Fixed `181/400`, Agentic `183/400`;
- Agentic vs Frozen: 4 conversions, 1 regression, McNemar `p=0.375`;
- Agentic vs Fixed: 2 conversions, 0 regressions, McNemar `p=0.5`;
- Agentic reduces VLA calls by `8.4%` and control steps by `7.2%` vs Frozen;
- co-resident VLA latency is P50/P95/P99 `56.44/59.54/63.19 ms`, with
  `8/14,699` calls above the 80 ms deadline;
- 263 event-triggered Planner calls average 9.09 s and raise total episode wall
  time by `16.5%`, identifying synchronous semantic reasoning as the remaining
  end-to-end bottleneck.

The success uplift is small and not statistically significant. The defensible
contribution is a complete, auditable Agentic execution system with bounded
correction, measured failure handling and a realtime VLA path. The full audit
passes for all 1,200 episodes and videos.

Evidence:

- `docs/status/LIBERO_PRO_FULL_STUDY_RESULTS_20260826.md`;
- `results/libero_pro_full_study_20260825/aggregate/`;
- `results/libero_pro_full_study_20260825/`.

## 17. Full efficient-inference ablation

The final standalone study evaluates five PI0.5 runtime profiles on 45 paired
recorded observations and three Qwen3.5-4B precision profiles on 30 paired
temporal observations. Compiled BF16 + SMVE reaches P95 `54.67 ms`, passes
`45/45` action-fidelity checks and records no 80 ms misses. Planner NF4 lowers
allocated model memory from `8.46` to `3.08 GiB` while preserving all paired
semantic decisions, but increases P95 from `1.90` to `2.91 s`.

The PI0.5 late-language INT8 profile is explicitly backend-version gated. It
reached P95 `71.61 ms` and 2/2 matched recovery success on the validated Torch
2.7.1 stack, but P95 `994.74 ms` on the current Torch 2.12.0 stack. CARVE
therefore admits profiles by hardware/software identity, fidelity, deadline and
closed-loop evidence rather than by precision label alone.

The Q2 NF4 Planner and V3 SMVE PI0.5 were also loaded together on the RTX 4090.
Their service processes used `11.22 GiB`; one typed Planner-to-VLA transaction
completed with an accepted `vla_act`, a `10 x 7` action chunk and `66.69 ms`
VLA runtime without fallback. This is integration feasibility evidence only;
Q2 still requires repeated contention and closed-loop gates for full admission.

Evidence:

- `docs/status/CARVE_EFFICIENT_INFERENCE_ABLATION_20260826.md`;
- `results/carve_efficiency_full_20260826/REPORT.md`;
- `results/carve_efficiency_full_20260826/summary.json`.
- `results/carve_efficiency_full_20260826/coupling/low_memory_coupling_gate.json`.

## 18. Verified-procedure memory routing

The final routing study separates novel-task planning from repeated-task
execution. Qwen3.5-9B NF4 remains the capacity tier for new task plans, while a
verified symbolic procedure can warm-start the Harness and Qwen3.5-4B BF16 is
used as the event-triggered visual Critic/fallback tier.

On four official LIBERO-PRO Object T8 states, the memory-routed system completes
`4/4` tasks and plans with zero task-start Planner calls. On the three states
shared with the 9B reference it preserves `3/3` success, removes all three
startup Planner calls, reduces mean episode wall time by `53.2%`, and records
`0/125` VLA deadline misses. Direct 4B long-plan generation was rejected by the
schema/coverage gate before physical execution, so the result supports adaptive
model-and-memory routing rather than a general small-Planner replacement.

Evidence:

- `docs/status/CARVE_MEMORY_ROUTED_PLANNER_PROFILE_20260827.md`;
- `results/planner_profile_closed_loop_20260827/aggregate.json`;
- `results/planner_profile_closed_loop_20260827/_review/`.

Current software verification at the T8 routing checkpoint: `254 passed`.

## 19. Cross-task factorized memory-routing study

The routing result now extends beyond Object T8 to standard LIBERO-10 T3
(place a black bowl in the bottom drawer and close it). A trusted state-0
discovery creates one complete two-stage procedure; official held-out states
1, 7 and 9 compare Direct 9B, Memory+9B and Memory+4B.

- all three positive routes complete `3/3` tasks and `3/3` plans;
- Memory+9B removes all startup Planner calls and lowers mean wall time `62.0%`
  against Direct 9B;
- Memory+4B lowers Critic latency another `34.3%` and reaches a total `73.1%`
  wall-time reduction against Direct 9B;
- all three routes record `0/66` 80 ms VLA deadline misses;
- direct 4B novel-task planning is rejected at step 0 after two incomplete-plan
  attempts, establishing a clear capacity boundary.

The qualification also hardened ordered multi-clause plan coverage, complete-
plan-only memory promotion, plan-step persistence and identifier-normalized
memory retrieval. Current software verification is `258 passed`.

Evidence:

- `docs/status/CARVE_CROSS_TASK_MEMORY_ROUTING_RESULTS_20260827.md`;
- `results/cross_task_memory_routing_20260827/aggregate.json`;
- `results/cross_task_memory_routing_20260827/_review/`.

## 20. RoboMME 八任务配对研究

RoboMME 用于验证高层 Agentic 工具、过程记忆与 Planner 调度，不承担连续控制
VLA 的模型训练比较。8 个任务各取 10 个 official episodes，对三种条件执行
共 240 个 rollouts：

| 条件 | 成功 | Planner calls | 总 wall time |
|---|---:|---:|---:|
| B1 Raw VLM+VLA | `23/80` (`28.7%`) | `2141` | `5208.0 s` |
| C2 Agentic Harness | `36/80` (`45.0%`) | `2001` | `5178.7 s` |
| C3 Harness + Optimize Runtime | `42/80` (`52.5%`) | `1188` | `3723.6 s` |

C3 相对 B1 提升 `23.8` 个百分点，包含 22 个失败转成功和 3 个成功转失败，
paired bootstrap 95% CI 为 `[+12.5,+35.0]` 个百分点，精确 McNemar 双侧
`p=0.000157`。相对每个 action chunk 都调用 Planner 的 C2，C3 减少 `40.6%`
Planner calls 和 `28.1%` 总 wall time。该结果是当前最强的 Agentic 配对证据，
但覆盖 8/16 个官方任务，应写为机制与子集研究，不写成完整 RoboMME leaderboard。

证据：

- `docs/reports/RAL_CORE_EXPERIMENT_REPORT_20260831.md`；
- `results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json`；
- `results/robomme_b1_c2_c3_combined_80ep_20260831/README.md`。

## 21. RoboDojo / StarVLA PI-v3 最新验证

### 21.1 第二 VLA adapter 与高效推理

CARVE 已接入 RoboDojo 官方 StarVLA PI-v3 推理链路，并修复原生 flow-step 控制。
同一驻留模型、20 对相同输入的配对微基准结果如下：

| Profile | Mean | P50 | P95 | 动作差异 |
|---|---:|---:|---:|---:|
| Flow4 | `307.23 ms` | `304.81 ms` | `315.31 ms` | reference |
| Flow2 | `235.42 ms` | `230.56 ms` | `265.18 ms` | MAE `0.003251` |

Flow2 平均时延下降 `23.37%`。闭环质量门中，`build_tower` 的 Flow4/Flow2 均为
`3/3`，`put_bottles_into_dustbin` 的 Flow4/Flow2 也均为 `3/3`。因此当前仅声称
“单次 VLA 推理更快且小样本闭环非劣”，不声称所有任务的 episode wall time
都会同步下降。

证据：

- `artifacts/robodojo/starvla_flow_step_microbenchmark_20260902/summary.json`；
- `artifacts/robodojo/build_tower_flow2_h16_ablation_20260902/summary.json`；
- `artifacts/robodojo/put_bottles_flow2_h16_ablation_20260902/summary.json`。

### 21.2 在线 Planner 编排与受控恢复

正常 `build_tower` 回合中，Qwen3-VL-4B 在线生成有效三阶段计划并通过 schema
校验，任务账本被安装，官方评测成功 `1/1`；4 次过程检查均为 inconclusive，
系统没有误执行语义干预。该结果验证在线 VLM 编排与 StarVLA 共存，不作为性能
提升证据。

为隔离 Agentic 恢复机制，在 action step 640 注入 stale-action hold，并对三个
paired layout/action seed 比较：

| 条件 | 成功 | 主要行为 |
|---|---:|---|
| C1：Optimize only | `0/3` | 无恢复，持续执行陈旧动作至超时 |
| C3：Monitor + 两阶段恢复 | `3/3` | 首次低成本重规划，重复异常触发 1 次 Planner 与 2 次 Flow4 恢复调用 |

C3 的首次 no-progress 出现在 step `674/676/676`，三回合均从 Flow2/h16 临时切换
到 Flow4/h16 完成两次恢复调用，再返回 Flow2/h16。三对均为失败转成功、无回归；
精确 McNemar 双侧 `p=0.25`。因此这是强机制证据，而不是统计显著或自然分布下的
benchmark 成功率结论。

证据：

- `artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/summary.json`；
- `artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/report.md`；
- `docs/status/ROBODOJO_CARVE_PAIRED_RESULTS.md`。

### 21.3 VLM Critic 与单卡资源边界

48 个 holdout（40 个未完成状态、8 个终态）上，Qwen3-VL-4B BF16 与保留视觉
模块的 NF4 均只有 `3/8` 终态召回，虽然 false-stop 为 `0/40`，但不足以获得
safe-stop authority。NF4 将静态模型显存从约 `8.27 GiB` 降到 `3.27 GiB`，却
没有降低延迟。更激进的 uniform NF4 终态召回下降到 `1/8`，已正式拒绝。

StarVLA + RoboDojo + 视觉保留 NF4 Qwen3-VL-4B 在 RTX 4090 上首次联合视觉推理
发生 OOM。当前工程决策是保持 event-triggered VLM 接口，但将完整在线 C3 部署
限定为远程 VLM、第二 GPU 或分阶段装载；在解决资源隔离前，不再宣称单卡稳定
共驻。

## 22. 当前证据边界

### 可以对外陈述

- 已构建可替换 VLM/VLA 的 Agentic Harness，包含任务计划、typed tools、执行
  Monitor、可信过程记忆、有界恢复、权限门控和结构化审计。
- Optimize Runtime 已在 PI0.5 与 StarVLA PI-v3 两条推理链路取得独立正结果，
  并经过动作保真度和小样本闭环质量门。
- Agentic 与 Optimize 联合运行能够减少无效 VLA 调用，并在受控可恢复停滞中将
  三个失败全部恢复成功。
- 系统明确记录了 Planner/Critic 的准确率、显存和共驻失败，不以量化标签代替
  部署准入证据。

### 不能对外陈述

- 不能声称 Agentic 在 LIBERO-Pro 上带来统计显著的通用成功率提升。
- 不能把三对受控故障写成 RoboDojo 自然任务总体提升或 SOTA。
- 不能声称 VLM Critic 已可靠判断任务完成，或拥有安全停止权限。
- 不能声称独立 4B VLM、StarVLA 与 RoboDojo 已在单张 RTX 4090 稳定共驻。
- 不能声称已完成真机部署。

## 23. 后续实验优先级

1. **自然失败 Agentic 配对实验：**选择 2--3 个可观察、非人为迎合框架的长程
   RoboDojo 任务，预注册触发条件，比较 Optimize-only 与 Agentic+Optimize。
2. **VLM 部署解耦：**优先验证远程/第二设备 VLM 或事件触发分阶段调度，不再用
   更激进视觉量化换取不可接受的语义准确率。
3. **扩大统计覆盖：**每个代表任务至少 10 个 paired seeds；同时报告成功转换、
   回归、Planner 次数、恢复预算、VLA 调用和 episode wall time。
4. **停止低价值重复：**不再追加同配置 LIBERO-Pro 或同一 `build_tower` 注入点；
   RoboMME 仅在能够形成统一多任务协议时继续。
5. **投稿条件：**RAL 级主张仍需自然失败下的多任务重复、第二 VLA 的外部有效性
   汇总，以及至少一种可部署 VLM 资源隔离方案；硕士毕业所需的系统与高效推理
   原型证据已经基本具备。

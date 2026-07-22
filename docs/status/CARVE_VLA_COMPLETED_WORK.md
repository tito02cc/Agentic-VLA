# CARVE-VLA 已完成工作总览

更新时间：2026-07-20

本文档是当前项目完成度、实验结论和证据边界的统一入口。实验过程记录保留在
`EXPERIMENT_LOG.md`，中期汇报材料保留在 `docs/reports/`；后续更新本文件，
不再建立带版本号的完成度文档。

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

当前主要闭环模型为 PI0.5；OpenVLA 用于验证跨模型接口和通用低比特方案的
边界，不作为第二套大规模 benchmark。

## 2. 总体完成度

| 研究模块 | 状态 | 当前结论 |
|---|---|---|
| Agentic Harness 工程框架 | 已完成 | v1 功能闭环完成，仅保留正确性修复 |
| PI0.5 接入与动作契约 | 已完成 | 本地 PyTorch、WebSocket 和 LIBERO 闭环均已运行 |
| 物理恢复与安全停止 | 已完成 | 支持 stall 恢复、结果验证和不支持事件的 fail-closed |
| VLA 高效推理 | 已完成 | low-step、编译、SMVE 均有可复现正结果 |
| 运行时/计算轻量化 | 已完成 | 减少 flow steps、无效视觉计算和非必要 VLA 调用 |
| Agentic 与优化耦合 | 已完成小规模主证据 | 同状态恢复结果保持，P95 和 deadline 明显改善 |
| 部署 profile admission | 已完成 | checkpoint/hardware/profile/evidence 绑定及 fallback 完成 |
| 跨 checkpoint/adapter | 已完成系统验证 | PI0.5 LIBERO/DROID 输入路径通过 |
| 第二 VLA 家族接口 | 已完成系统验证 | OpenVLA BF16/INT8/NF4/compile 已完成，不含闭环任务成功率 |
| 模型参数轻量化 | 部分完成 | 量化实验完成，但尚无同时通过时延、显存和闭环门控的 PI0.5 profile |
| 大规模新 benchmark | 未开展 | 中期前主动冻结，避免低信息增益的刷榜实验 |
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
| Eager BF16 | `154.34 ms` | `159.59 ms` | `100%` | `7.12 GB` | reference |
| Compiled BF16 | `65.73 ms` | `67.40 ms` | `0%` | `6.98 GB` | accepted |
| Compiled BF16 + SMVE | `54.35 ms` | `56.19 ms` | `0%` | `6.97 GB` | accepted |

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
| OpenVLA INT8 | `14.42 -> 7.76 GB` | P50 `303.48 -> 1533.96 ms` | exact action `50%` | rejected |
| OpenVLA NF4 | `14.42 -> 4.41 GB` | P50 `303.48 -> 748.35 ms` | exact action `10%` | rejected |

这些实验完成了模型级轻量化的可行性和边界分析，但尚未形成可部署的正向
PI0.5 低比特结果。因此不能声称已经完成参数压缩算法。

### 6.2 当前准确定位

已完成：

- 推理计算轻量化；
- 视觉计算轻量化；
- 系统调用轻量化；
- deadline-aware 部署优化。

尚待完成：

- 同时降低模型显存/存储且不增加关键路径时延；
- 通过 action fidelity 和闭环非劣门控的 PI0.5 混合精度 profile。

下一阶段最合理的研究项是 **VLA action-fidelity-aware mixed-precision
quantization**。剪枝和蒸馏当前不进入主线。

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

### B 级：历史背景，不作为当前主证据

- 历史 LIBERO-10：baseline `180/200`，Agentic `185/200`；
- 历史 B0-VLA-Light：Task1/5/7 `13/15` 保持成功，full calls 降低 `36.8%`。

上述历史结果的原始 rollout 目录在早期清理时未保留，只能作为研究过程背景，
不能用于新的可复现主张。

## 11. 当前可以对外声称的贡献

1. 构建了冻结 VLA 的 Agentic execution harness，具有部署信号监测、分级
   重规划、物理恢复、验证、记忆和 fail-closed 安全停止。
2. 构建了 profile-driven VLA Optimize Runtime，统一管理模型能力、动作契约、
   fidelity、deadline、闭环验收和硬件/checkpoint 绑定。
3. 在 PI0.5 上通过 flow-step calibration、compilation 和 SMVE 实现可复现的
   推理计算轻量化，并在 Agent/VLM 共卡负载下验证 deadline 收益。
4. 在相同 MuJoCo 恢复状态中保持 Agentic 行为结果，同时将 runtime P95 从
   `166.26 ms` 降至 `54.50 ms`。
5. 通过量化、异步和 semantic intervention 的负结果证明：开放环误差、显存
   或时延不能单独决定机器人部署 profile。

## 12. 当前不能声称的内容

- 不能声称 Agentic 在大规模 benchmark 上具有统计显著成功率提升。
- 不能声称已经提出并验证新的通用量化算法。
- 不能声称 PI0.5 模型参数或存储已经显著压缩。
- 不能把 model-call deadline 表述为完整机器人硬实时保证。
- 不能声称已在多个 VLA 家族上完成闭环任务成功率验证。
- 不能声称已经完成真机实验。

## 13. 中期汇报结论

当前已经达到中期汇报阶段：框架、主要推理优化、真实 MuJoCo 恢复实验、
Agentic/Optimize 联合实验、负结果和证据边界均已形成。开题中的轻量化方向
已经完成运行时/计算层部分；模型参数轻量化是下一阶段明确任务，而不是当前
已完成结论。

中期后优先级：

1. VLA action-fidelity-aware mixed-precision quantization；
2. 预声明的故障状态与多随机种子扩展；
3. 论文证据整合；
4. 仅在投稿需要时增加新 benchmark 或真机任务。

## 14. 项目入口

- [中期完整报告](../reports/CARVE_VLA_MIDTERM_REPORT_20260717.md)
- [中期投屏稿](../reports/CARVE_VLA_MIDTERM_PRESENTATION_20260717.md)
- [中期问答](../reports/CARVE_VLA_MIDTERM_QA_20260717.md)
- [Runtime 架构](../architecture/CARVE_RUNTIME_ARCHITECTURE.md)
- [下一阶段计划](../plans/CARVE_VLA_NEXT_STAGE_PLAN.md)
- [实验日志](EXPERIMENT_LOG.md)
- [论文机器可读结果](../../paper/CARVE-VLA/generated/runtime_results_summary.json)
- [当前论文](../../paper/CARVE-VLA/root.pdf)

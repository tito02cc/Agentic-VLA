# CARVE-VLA AI Project Context

历史正文日期：2026-08-27；导航更新：2026-09-22

> 本文正文是历史技术说明，不是当前实验队列。唯一执行依据是
> [收敛后的计划](plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)：P1核验既有证据，
> P2筛出必补缺口，P3按需补测，P4同步论文收口。新缓存、身份修复和预设扩跑已取消，
> 不自行增加平行候选，不以新框架开发替代投稿收口。
> 当前主验证为RoboMME的VideoUnmaskSwap/VideoRepick，PI0.5 JAX GroundSG79999 +
> Qwen3-VL-4B BF16官方LoRA1200，不是RoboDojo59999或旧LIBERO权重。
> [主报告10.7/10.8/16.2](reports/AGENTIC_VLA_TECHNICAL_REPORT.md)分别记录身份记忆
> 五初态1/5→5/5、非干扰两例1/2→1/2及下一步。后者不是新增收益；两份VideoRepick
> 记忆未准入。推理优化仍须在当前链路独立验证，不能借用旧后端提速。
> RoboDojo、RouteStick、新平台/模型、自动微调及其他候选不在当前队列；论文、
> 失败数据、视频与历史代码保留，不因收敛路线删除。历史AW--AZ见归档，不再执行。

本文档用于新电脑上的 AI 助手或新协作者快速理解项目，并在不夸大实验结论的
前提下撰写中期报告、论文概述或汇报材料。

## 1. 八月材料阅读顺序（历史）

1. `docs/reports/CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md`：当前框架、
   实验结果、证据边界以及毕业论文/简历使用方式。
2. `README.md`：项目定位和当前主结果。
3. `docs/architecture/CARVE_COMPLETE_RESEARCH_LOOP.md`：Agentic RAG-VLM、
   CARVE Harness、Harness VLA/RPent 参考边界和 Optimize Runtime 的统一闭环。
4. `docs/status/CARVE_VLA_COMPLETED_WORK.md`：完成度和统一结论入口。
5. `docs/archive/midterm/20260717/CARVE_VLA_MIDTERM_REPORT_20260717.md`：
   历史中期报告正文。
6. `docs/architecture/CARVE_RUNTIME_ARCHITECTURE.md`：Agentic Harness 与运行时接口。
7. `docs/architecture/CARVE_OPTIMIZE_RUNTIME.md`：推理优化、门控和部署策略。
8. `paper/CARVE-VLA/root.pdf` 或 `root.tex`：当前论文叙事。
9. `paper/CARVE-VLA/generated/runtime_results_summary.json`：统一机器可读结果。
10. `results/CARVE_EVIDENCE_README.md`：已上传证据范围与索引。

`docs/archive/status/EXPERIMENT_AND_PAPER_STATUS_20260612.md` 是历史阶段记录，
不覆盖当前完成度文档。`paper/archive/` 中的旧投稿只用于追溯，不代表当前
论文主张。

## 2. 研究问题

开题方向是 VLA 高效推理与部署优化。CARVE-VLA 研究冻结 VLA 在长程机器人
操作中的两个耦合问题：

1. VLA 在 stall、misgrasp、状态陈旧等事件中需要监测、恢复、验证和安全停止；
2. Agentic 重规划与恢复增加模型调用，需要降低单次推理成本、无效视觉计算和
   deadline miss，同时保证动作保真度与闭环结果不退化。

系统不重新训练 PI0.5 基座模型。PI0.5 是主要闭环后端；OpenVLA 用于验证
跨模型接口及通用量化工具的边界。

## 3. 框架组成

### CARVE Agentic Harness

- 代码：`agentic_vla/runtime/`
- 核心模块：policy/action contract、execution monitor、joint controller、
  recovery memory、critic/retry、physical recovery、safe stop 和 runtime trace。
- 部署决策只使用 RGB、proprioception、动作历史、action age 和 deadline
  slack；仿真器物体真值只用于评测标签。

### CARVE Optimize Runtime

- 代码：`agentic_vla/optimization/`
- 核心模块：backend/plugin registry、profile manifest、fidelity gate、deadline
  gate、fallback、benchmark 和 deployment admission。
- PI0.5 接入：`agentic_vla/runtime/adapters/pi05.py`、
  `openpi/scripts/serve_policy.py` 和 `openpi/src/openpi/training/config.py`。
- 当前正结果包括 flow-step calibration、`torch.compile` BF16 和 Static
  Masked-View Elision (SMVE)。
- SMVE 当前定位为 L0 静态输入压缩。完整优化路线还包括 optimized backend、
  training-free temporal reuse、event-coherent invalidation、admitted adaptive
  compute、VLA 专用低比特 backend 和 VLM/VLA 多模型调度。

## 4. 当前主要实验结论

### 完整 benchmark 与具身 Agent

- 已完成 4 suites、40 tasks、10 个配对状态和 3 种方法，共 1,200 个真实
  MuJoCo episode；Frozen/Fixed/Agentic 为 `180/181/183` 成功。
- Agentic 净增 3 个成功但不具统计显著性；control steps 和 VLA calls 分别
  减少 `7.2%/8.4%`，PI0.5 P95 为 `59.54 ms`。
- 完整 VLM Planner、Harness、Critic、可信记忆、恢复、冻结 PI0.5 和
  Optimize Runtime 在 Object T8 三个官方状态上达到 `3/3`，并产生视频与
  可审计 trace。
- verified-procedure memory 已在两个任务族、7 个 memory-routed 官方状态上
  验证。第二任务族中 Memory+4B 保持 `3/3`，相对 Direct 9B 将平均回合时间
  降低 `73.1%`，所有 198 次配对 VLA 调用满足 80 ms deadline。

### Agentic Recovery Challenge

- 真实 LIBERO MuJoCo 中的三个精确恢复状态、四种执行策略。
- continuation、frequent replan、prompt retry、physical recovery 均为 `2/3`。
- frequent replan 和 prompt retry 分别使用 `452/508` 次 PI0.5 调用，
  continuation 为 `113` 次。
- physical recovery 验证两个受支持 stall，并对不支持的 stale-action 事件
  fail closed。
- 该实验支持事件触发计算和恢复契约，不支持“大规模成功率显著提升”的声明。

### PI0.5 推理优化

- 7 flow steps 降到 2 steps：配对成功保持 `14/15`，单次 VLA 从
  `366.90 ms` 降至 `148.96 ms`。
- Eager BF16 P95：`159.59 ms`。
- Compiled BF16 P95：`67.40 ms`，80 ms miss 为 `0%`。
- Compiled BF16 + SMVE P95：`56.19 ms`，45/45 replay fidelity 通过。
- early4 W8A16 虽降低部分显存，但闭环丢失一次恢复成功，因此被否决。
- 新的 late-language INT8 profile 通过 `45/45` replay 与 T6/T9 `2/2`
  恢复门控，峰值 `6.36 GB`，作为低显存档获准；其 P95 `71.61 ms`，因此
  SMVE 仍是实时默认档。
- INT8+SMVE 虽达到 P95 `58.71 ms`，但 T9 恢复失败，组合被否决。
- P0 Planner 与 late-INT8 同卡时，连续/5 秒 cooldown 的 80 ms miss 分别为
  `95.4%/19.4%`，因此低显存档不进入当前同卡 Agentic 配置。

### Agentic 与 Optimize 联合实验

- 相同 T6/T9 stall 状态和相同 physical recovery，只替换 execution profile。
- Eager、Compiled、Compiled + SMVE 均保持 `2/2` 精确状态任务结果和恢复验证。
- Runtime P95 为 `166.26/65.75/54.50 ms`；两个 admitted profile 的 80 ms
  deadline miss 均为零。
- 这是当前连接 Agentic Harness 与高效推理贡献的直接主证据。

### 已记录的负结果

- PI0.5 W8A16：replay 通过但闭环退化。
- OpenVLA INT8/NF4：显存下降，但推理变慢且 exact-action gate 失败。
- 异步预取：deadline miss 明显下降，但 T8/T9 成功从 `7/10` 降到 `5/10`。
- prefix consistency：AUC `0.563`，不足以进入 enforce 模式。

## 5. 声明边界

可以写入中期报告：

- 已完成 Agentic execution harness 和 profile-driven Optimize Runtime；
- 已完成 PI0.5 运行时/计算轻量化，并有 replay、闭环和联合恢复证据；
- 已分析模型量化、异步调度和语义干预的收益与失败边界；
- 当前结果来自真实 MuJoCo 仿真和 RTX 4090 部署测量。

不能写成：

- Agentic 已在大规模 benchmark 上取得统计显著成功率提升；
- 已提出并验证新的通用量化算法；
- PI0.5 参数量或模型存储已经显著压缩；
- model-call deadline 等价于完整机器人硬实时保证；
- 已完成多 VLA 家族的闭环任务成功率验证；
- 已完成真机实验。

## 6. 仓库与本地环境边界

仓库包含源码、论文、架构文档和紧凑实验证据。以下内容因体积或许可证原因
不上传：PI0.5/OpenVLA 权重、完整 LIBERO 环境、虚拟环境、checkpoint、训练
数据和大规模 rollout 缓存。缺少这些内容不影响撰写报告和核查当前表格、图、
短视频与 gate 结论，但不能仅凭 GitHub clone 重新运行全部 GPU 实验。

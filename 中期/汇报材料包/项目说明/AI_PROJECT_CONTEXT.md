# CARVE-VLA AI Project Context

更新时间：2026-07-20

本文档用于新电脑上的 AI 助手或新协作者快速理解项目，并在不夸大实验结论的
前提下撰写中期报告、论文概述或汇报材料。

## 1. 推荐阅读顺序

1. `README.md`：项目定位和当前主结果。
2. `docs/status/CARVE_VLA_COMPLETED_WORK.md`：完成度和统一结论入口。
3. `docs/reports/CARVE_VLA_MIDTERM_REPORT_20260717.md`：中期报告正文。
4. `docs/architecture/CARVE_RUNTIME_ARCHITECTURE.md`：Agentic Harness 与运行时接口。
5. `docs/architecture/CARVE_OPTIMIZE_RUNTIME.md`：推理优化、门控和部署策略。
6. `paper/CARVE-VLA/root.pdf` 或 `root.tex`：当前论文叙事。
7. `paper/CARVE-VLA/generated/runtime_results_summary.json`：统一机器可读结果。
8. `results/CARVE_EVIDENCE_README.md`：已上传证据范围与索引。

`docs/status/EXPERIMENT_AND_PAPER_STATUS_20260612.md` 是历史阶段记录，不覆盖
当前完成度文档。`paper/archive/` 中的旧投稿只用于追溯，不代表当前论文主张。

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

## 4. 当前主要实验结论

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
- W8A16 虽降低部分显存，但闭环丢失一次恢复成功，因此被 deployment gate
  否决，不能表述为已经完成可部署参数量化。

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

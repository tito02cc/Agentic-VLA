# 毕业论文实验材料包

论文方向：面向具身智能的多模态智能体系统与高效推理研究

更新时间：2026-08-27

> 本文记录 2026-08-26 完成的基础实验包。包含完整具身智能体、可信过程记忆和
> 跨任务模型路由的最新统一结论，请优先阅读
> [CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md](CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md)。

## 1. 论文主线

论文围绕同一个部署问题展开：冻结 VLA 能完成基本操作，但长程执行会出现
停滞、误抓和状态变化；加入 Agentic 监测、规划和恢复后，又会引入额外模型
调用、时延和显存开销。因此系统需要同时解决：

1. 如何在不训练新基座模型的条件下，为 VLA 增加可审计的闭环执行能力；
2. 如何降低高频 VLA 路径的推理时延，并限制低频 VLM Planner 的资源开销；
3. 如何通过 fidelity、deadline、闭环和硬件绑定门控，避免“更小但更慢”或
   “开放环误差很小但闭环失败”的部署配置。

对应系统由两个可独立使用、通过契约耦合的模块组成：

- **CARVE Agentic Harness**：Execution Monitor、VLM Planner/Critic、Memory、
  Joint Controller、Physical Recovery、Verification 和 Safe Stop；
- **CARVE Optimize Runtime**：flow-step 校准、编译、SMVE、组件级量化、
  profile admission、fallback 和运行时 trace。

Agentic RAG-VLM 可作为前期研究基础：它提供结构化场景表示、检索增强推理和
Self-Reflection；CARVE 将这些思想从单次操作推理扩展到冻结 VLA 的持续执行、
物理恢复和运行时资源管理。

## 2. 研究问题与证据

| 研究问题 | 对照 | 主要指标 | 当前证据 |
|---|---|---|---|
| RQ1 Agentic Harness 是否改善执行过程 | Frozen VLA / Fixed Recovery / Full Agentic | success、conversion、safe stop、VLA calls、steps | 1,200 回合 official LIBERO-Pro/LIBERO |
| RQ2 Optimize Runtime 是否满足高频实时路径 | eager 7-step / eager 2-step / compile / SMVE | P50/P95/P99、Miss@80ms、fidelity、VRAM | 45 组配对 PI0.5 observation |
| RQ3 量化是否支持单卡部署 | Planner BF16/INT8/NF4；PI0.5 late-INT8 | VRAM、P95、语义/动作一致性、闭环 | 30 组语义事件、45 组动作 replay、T6/T9 恢复 |
| RQ4 Agentic 与优化模块能否耦合 | Planner + admitted VLA profile | 共驻显存、Planner 决策、VLA deadline、fallback | 1,200 回合默认组合及 NF4 低显存 smoke |

## 3. 实验一：Agentic Harness

### 3.1 设置

- 模型：冻结 PI0.5 LIBERO PyTorch checkpoint；
- 仿真：official LIBERO-10、LIBERO-PRO Object、Position Swap、Task Logic；
- 规模：4 suites × 10 tasks × 10 paired states × 3 methods；
- 总计：1,200 个 MuJoCo episode、1,200 个视频、454,540 帧；
- 方法：Frozen VLA、Fixed Recovery、Full Agentic；
- 相同 checkpoint、初始状态、动作 profile 和随机协议；
- Agent 不读取 simulator task-success、object pose 或 private state。

### 3.2 主要结果

| 方法 | Success | Control steps | VLA calls | Recovery | Safe stop |
|---|---:|---:|---:|---:|---:|
| Frozen VLA | 180/400 | 159,651 | 16,047 | 0 | 0 |
| Fixed Recovery | 181/400 | 145,586 | 14,442 | 173 | 99 |
| Full Agentic | 183/400 | 148,103 | 14,699 | 173 | 84 |

Full Agentic 相对 Frozen VLA：

- 成功数增加 3，提升 `0.75` 个百分点；
- 4 个失败转成功、1 个成功转失败；
- paired McNemar `p=0.375`，不构成统计显著成功率提升；
- control steps 减少 `7.23%`，VLA calls 减少 `8.40%`；
- 263 次事件触发 Planner 调用，平均 `9.09 s`，P95 `11.23 s`；
- VLA P95/P99 为 `59.54/63.19 ms`，80 ms miss 为 `8/14,699`。

可支持的结论是：CARVE 提供完整、可审计的冻结 VLA 闭环执行系统，在保持
高频 VLA deadline 的同时实现有界恢复、失败转换、调用削减和安全停止。
不能把结果写成 benchmark success 的显著提升或 SOTA。

## 4. 实验二：PI0.5 Optimize Runtime

所有 profile 使用同一 checkpoint、45 组配对 observation、固定 flow noise 和
action horizon 10。80 ms deadline 由保存的每次调用时延重新计算。

| ID | Profile | P95 | Miss@80ms | Peak VRAM | Fidelity | 决策 |
|---|---|---:|---:|---:|---:|---|
| V0 | Eager BF16, 7 steps | 282.43 ms | 100% | 7.12 GiB | reference | 行为基线 |
| V1 | Eager BF16, 2 steps | 151.35 ms | 100% | 7.12 GiB | reference | deadline 否决 |
| V2 | Compile BF16, 2 steps | 66.06 ms | 0% | 6.98 GiB | 45/45 pass | fallback |
| V3 | Compile BF16 + SMVE | 54.67 ms | 0% | 6.98 GiB | 45/45 pass | realtime default |
| V4 | Late-language INT8 | 994.74 ms | 100% | 6.36 GiB | 45/45 pass | 当前版本否决 |

V3 相对 V0 将 P95 降低 `80.6%`。SMVE 只消除 adapter 契约保证为 padding 的
right-wrist view；若运行时 mask 不满足前提，系统自动回退到已独立验收的 V2，
因此不是删除有效相机输入。

## 5. 实验三：轻量化与量化

### 5.1 Planner Q0-Q2

| ID | Qwen3.5-4B | Allocated VRAM | P95 | 语义一致率 | Gate | 决策 |
|---|---|---:|---:|---:|---:|---|
| Q0 | BF16 | 8.46 GiB | 1.90 s | 100% | pass | latency default |
| Q1 | INT8 | 4.84 GiB | 6.50 s | 100% | pass | 当前 kernel 否决 |
| Q2 | NF4 | 3.08 GiB | 2.91 s | 100% | pass | low-memory candidate |

Q2 减少 `63.6%` 模型显存，但 P95 增加 `53.4%`。它的作用是降低单卡共驻
门槛，而不是加速 Planner。Q1 在当前 bitsandbytes backend 上既慢于 Q0，也
大于 Q2，因此没有部署优势。

### 5.2 PI0.5 组件级 INT8

late-language INT8 在经过验证的 Torch 2.7.1/TorchAO 0.15 环境中：

- peak VRAM：`6.36 GiB`；
- P95：`71.61 ms`；
- action fidelity：`45/45`；
- matched T6/T9 recovery：`2/2`。

同一逻辑 profile 在当前 Torch 2.12/TorchAO 0.15 环境中 P95 退化至
`994.74 ms`，因此必须与软件版本绑定。INT8+SMVE 虽达到 `58.71 ms`，但在
T9 闭环恢复中失败，故被否决。这组结果证明显存、开放环动作误差和单次时延
不能代替闭环 admission。

## 6. 实验四：Agentic 与 Optimize 耦合

### 6.1 默认组合

1,200 回合正式实验实际运行 Q0 BF16 Planner 与 V3 SMVE PI0.5：

- 263 次事件触发 VLM 规划；
- 14,699 次 PI0.5 调用；
- PI0.5 P95 `59.54 ms`；
- 80 ms miss `0.0544%`；
- 完整审计和 1,200 个视频通过。

这说明低频语义规划没有破坏高频 VLA fast path，但同步 Planner 仍使 Agentic
总 wall time 相对 Frozen 增加 `16.5%`，是后续系统优化的主要瓶颈。

### 6.2 低显存组合

Q2 NF4 Planner 与 V3 SMVE PI0.5 在同一 RTX 4090 上真实共驻：

- 两个模型服务进程合计 `11.22 GiB`；
- Planner 一次 accepted `vla_act`：`13.92 s`；
- VLA 输出 `10 × 7` 动作块：`66.69 ms`；
- 80 ms deadline 满足，无 fallback；
- 动作在 primitive 边界截获，未发送给环境。

该结果只支持低显存组合的集成可行性。没有重复 contention 和闭环任务结果，
因此 Q2 状态为 `integration passed, system admission pending`。

## 7. 建议的毕业论文章节

1. 绪论：具身多模态智能体、冻结 VLA 的执行问题和 Agentic 计算开销；
2. 相关工作：Agentic RAG/VLM、VLA、Agentic Policy、VLA 高效推理与量化；
3. 多模态 Agentic Harness：Monitor、Planner/Critic、Memory、Controller、
   Recovery、Verification、Safe Stop；
4. CARVE Optimize Runtime：profile、契约、flow-step、Compile、SMVE、量化、
   fallback、admission；
5. 实验：按 RQ1-RQ4 展开上述四组实验；
6. 讨论：成功率不显著、Planner 时延、版本敏感量化、仿真到真实部署边界；
7. 总结与展望。

## 8. 可直接使用的图表与原始证据

- 完整 Agentic 结果：`results/libero_pro_full_study_20260825/aggregate/`；
- Agentic 报告：`docs/status/LIBERO_PRO_FULL_STUDY_RESULTS_20260826.md`；
- 高效推理报告：`results/carve_efficiency_full_20260826/REPORT.md`；
- 高效推理图：`results/carve_efficiency_full_20260826/efficiency_ablation.png`；
- VLA 表格：`results/carve_efficiency_full_20260826/vla_ablation.csv`；
- Planner 表格：`results/carve_efficiency_full_20260826/planner_ablation.csv`；
- 低显存耦合：`results/carve_efficiency_full_20260826/coupling/LOW_MEMORY_COUPLING_GATE.md`；
- PI0.5 量化闭环：`results/carve_optimize/pi05_quantized_closed_loop_gate.json`；
- 主要视频：`results/libero_pro_full_study_20260825/` 中每个正式 episode 的视频。

## 9. 完成度判断

| 毕业论文要求 | 状态 |
|---|---|
| 完整系统方法 | 已完成 |
| Agentic 与 VLA 真实仿真闭环 | 已完成，1,200 回合 |
| 高效推理独立消融 | 已完成，5 个 VLA profile |
| 轻量化/量化实验 | 已完成定向验证，包含正负结果 |
| Agentic 与 Optimize 耦合 | 默认组合完整；低显存组合完成 integration gate |
| 可复查数据、视频和日志 | 已完成 |
| 真机实验 | 未完成，不作为当前结论 |

当前实验已经足以进入毕业论文正式写作。后续不需要再扩大 benchmark；若导师
明确要求额外实验，优先补 Q2+V3 的重复 contention gate，而不是增加新的模型、
任务或量化组合。

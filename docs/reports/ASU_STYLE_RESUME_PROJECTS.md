# ASu 风格项目简历成稿

更新时间：2026-08-31

适用岗位：具身智能算法工程师、VLA 算法工程师、AI Agent 工程师、大模型
推理部署工程师。

本稿采用“高密度技术链 + 量化结果 + 证据边界”的写法。简历主版建议每个
项目保留 3 条，详细内容用于面试，不要全部塞进一页简历。

## 项目一：Agentic RAG-VLM

### 推荐标题

**Agentic RAG-VLM：面向机器人操作的检索增强多模态智能体系统**  
IROS 2026 录用工作 | Python、Qwen3-VL、RAG、Scene Graph、Agent Memory、
BitsAndBytes

### 一句话介绍

针对 VLM 能理解操作语义但缺少物理可执行知识与失败恢复能力的问题，构建融合
可供性检索、场景图约束、反思重试与情景记忆的结构化机器人操作推理框架。

### 简历三条版

- **构建多模态操作智能体链路：**设计 HAA-RAG、Scene Graph Reasoner、
  Self-Reflection 与 Episodic Memory 的七阶段推理流程，将 RGB-D/语言输入
  转化为位置、方向、夹爪宽度、力度和抓取类型等可解释参数，并以 14 类失败
  taxonomy 驱动参数调整、方法切换和完整重规划。
- **设计可供性驱动的经验检索：**将物体类别、材质、易碎性和可抓区域编码为
  结构化知识，通过类别过滤、功能可供性匹配和视觉重排完成三级检索；在 116 条
  抓取经验上取得 `91.7%` 检索准确率与 Recall@3，支持跨类别策略复用。
- **完成系统消融与推理优化：**在 12 类任务、每配置 360 次测试中取得
  `78.3%` 成功率，相比 VLM-only 提升 `53.3` 个百分点；结合 INT4 与 FP8
  KV Cache 将 Qwen3-VL-8B 吞吐提升至 `155.5 token/s`，相对 FP16 为 `1.78x`。

### 更短的一页简历版

- 构建融合 HAA-RAG、场景图约束、14 类失败诊断、三级反思恢复和情景记忆的
  多模态操作智能体，将 VLM 感知结果转化为可解释抓取参数与闭环策略。
- 设计类别-可供性-视觉三级经验检索，在 116 条知识库上取得 `91.7%` 检索
  准确率和 Recall@3，并支持同类/跨类抓取经验复用。
- 在 12 类任务、每配置 360 次测试中实现 `78.3%` 成功率（较 VLM-only
  `+53.3` pp）；以 INT4 + FP8 KV Cache 获得 `1.78x` VLM 推理吞吐提升。

### 不能这样写

- 不写“真实机器人/物理引擎实验”，该论文实验是 analytical simulation；
- 不写“端到端控制 VLA”，该项目核心模型是 VLM 结构化操作推理；
- 不将 360 次误写为总实验量：这是每个配置的 trial 数；
- 不写“提出通用量化算法”，INT4/FP8 KV Cache 是工程优化配置。

## 项目二：面向VLA的具身智能体系统与高效推理研究

### 推荐标题

**面向VLA的具身智能体系统与高效推理研究**  
研究项目 | Python、PyTorch、PI0.5、Qwen3-VL、RoboMME、MuJoCo、
`torch.compile`、TorchAO、BitsAndBytes

### 一句话介绍

针对冻结 VLA 在长程操作中的任务遗忘、持续规划、执行异常和推理时延问题，
构建由 Agentic Harness 与 Optimize Runtime 组成的多速率闭环，在不微调基础
模型的条件下联合提升任务成功率、语义调用效率和 VLA 实时推理性能。

### 完整科研项目版（推荐作为简历母版）

**复旦大学智能机器人研究院**  
2026年04月–至今

**科研项目：**面向VLA的具身智能体系统与高效推理研究

- **项目简介：**面向 VLA 在长程机器人操作中的任务规划不足、执行错误累积和
  推理时延问题，构建无需重新训练基础模型的具身智能体系统；由 VLM 负责高层
  任务理解和子目标规划、VLA 负责低层动作生成，并从智能体调度和模型推理两个
  层面优化端到端执行效率；通过统一模型与动作接口，将智能体框架与具体的 VLM、
  VLA 和机器人环境解耦。
- **具身智能体框架：**设计“规划—执行—监测—恢复”的多速率闭环：高层 VLM
  主要在任务开始、阶段切换或异常发生时进行语义判断，并集成视觉/任务记忆、场景
  理解、工具调用、动作完成验证和有限错误恢复，处理目标遮挡、位置交换、执行停滞
  及子任务切换错误；通过预定义子目标、技能白名单和安全停止约束 VLM 控制权限。
- **高效推理优化：**针对 Agentic 架构中 VLM 语义推理与 VLA 动作生成并存的
  计算开销，通过事件触发规划、可信过程记忆和子目标复用减少高成本语义调用；
  通过减少动作生成迭代、编译计算过程和跳过无效相机输入降低 VLA 重复计算，并以
  动作一致性和闭环任务结果约束优化配置。最终将 95% 模型调用的时延上界由
  `282.43 ms` 降至 `54.67 ms`，测试中的 80 ms 超时率由 `100%` 降至 `0%`。
- **系统验证：**以 PI0.5 与 Qwen 系列 VLM 为验证模型，在 RoboMME 4 类 8 个长程
  任务上完成 240 个配对闭环仿真实验，将未经智能体增强的 VLM+VLA 基线成功率
  由 `28.7%` 提升至 `52.5%`，并相对每个动作块均调用高层模型的方案减少
  `40.6%` 规划调用和 `28.1%` 总耗时；进一步在 LIBERO/LIBERO-PRO 的
  4 套分布、40 个任务条件上完成 1,200 个配对物理仿真回合，覆盖标准场景、
  物体变化、位置交换和任务逻辑变化。

> 日期 `2026年04月–至今` 根据 Agentic VLA 项目承接时间填写；正式投递前应与
> 个人简历中的立项时间保持一致。

### 简历三条版（推荐）

- **构建完整具身 Agentic 工具链：**围绕冻结 PI0.5 实现事件触发 Qwen3-VL
  Planner、执行风险 Monitor、持久视觉/过程记忆、grounded tool calling、
  primitive 阶段控制、有限物理恢复与 fail-closed 状态机；以 typed intent、
  skill allowlist 和 stale-result rejection 隔离语义规划与低层动作权限。
- **验证长程任务与选择性规划：**在 RoboMME 8 个 Counting/Persistent/
  Referential/Behavior 任务上完成 `240` 个配对闭环 rollout，将 Raw VLM+VLA
  成功率由 `28.7%` 提升至 `52.5%`（`+23.8` pp，McNemar `p=1.57e-4`）；
  相比逐 action-chunk 规划减少 `40.6%` Planner 调用和 `28.1%` 总耗时。
- **搭建 evidence-gated 推理运行时：**通过 PI0.5 flow-step 校准、
  `torch.compile`、预热和 Static Masked-View Elision，将 RTX 4090 上 P95
  从 `282.43 ms` 降至 `54.67 ms`，80 ms deadline miss 从 `100%` 降至
  `0%`；以 action fidelity 与 paired closed-loop gate 拒绝行为退化的量化和
  异步候选。

### 更短的一页简历版

- 构建冻结 PI0.5 的具身 Agentic Harness，集成事件触发 VLM Planner、
  Monitor、持久记忆、工具调用、primitive 过程控制、有限恢复与安全停止。
- 在 RoboMME 8 任务、240 个配对闭环 rollout 中将 Raw VLM+VLA 成功率从
  `28.7%` 提升至 `52.5%`，并较逐 chunk 规划减少 `40.6%` Planner 调用和
  `28.1%` 总耗时。
- 以 flow-step、`torch.compile` 和 SMVE 将 PI0.5 P95 从 `282.43 ms` 降至
  `54.67 ms`，80 ms deadline miss 从 `100%` 降至 `0%`，建立保真度-运行时-
  闭环结果联合准入机制。

## 按岗位替换的关键词

### 具身智能 / VLA 算法工程师

优先保留：PI0.5、RoboMME、long-horizon manipulation、action chunk、
primitive、persistent memory、physical recovery、paired closed-loop evaluation。

项目二第一条可改为：

> 将冻结 PI0.5 封装为受监督 physical primitive，设计 VLM 语义规划、VLA
> 动作生成、过程 Monitor 与 bounded recovery 的多速率闭环，支持遮挡目标
> 保持、阶段完成验证和长程任务恢复。

### AI Agent 工程师

优先保留：VLM Planner、tool calling、typed schema、memory、single-flight、
budget/timeout、audit trace、safe execution。

项目二第一条可改为：

> 将软件 Agent 的 Planner、Memory、Tool、Critic 和 Recovery 抽象迁移到
> 机器人执行系统，实现 typed tool contract、single-flight planner ticket、
> 预算/超时、陈旧结果拒绝和全过程审计，使高层语义决策不能越权生成动作。

### 大模型推理 / 部署工程师

优先保留：batch-one inference、P95、deadline miss、VRAM、`torch.compile`、
TorchAO、BitsAndBytes、profile admission、co-resident scheduling。

项目二第三条可改为：

> 构建 checkpoint/hardware-bound VLA deployment profile，完成 flow-step、
> compile、视觉无效计算裁剪及 W8A16/INT8/NF4 候选评测；以 fidelity、P95、
> deadline 与闭环任务四级门控选择部署配置，而非仅依据显存或 microbenchmark。

## 项目关系的面试口播

> 我的研究主线是把多模态模型的语义推理真正接入机器人闭环。Agentic
> RAG-VLM 先解决 VLM 如何利用可供性知识、场景约束和失败经验生成物理可解释
> 的操作策略；后续工作进一步把这些 Agent 能力扩展到冻结 VLA，通过 Planner、
> Memory、Tool 和 Monitor 持续管理长程执行。Agentic 链路同时引入额外模型
> 调用，所以我进一步实现 Optimize Runtime，从 VLA 单次计算和 VLM 调用调度
> 两层优化整体时延，并用闭环准入避免只在 microbenchmark 上更快、但机器人
> 行为已经退化的方案上线。

## 证据审计

| 简历主张 | 证据 | 状态 |
|---|---|---|
| Agentic RAG-VLM 12 任务、78.3% | `paper/Agentic-RAG-VLM/main_final.tex` | 已证实 |
| HAA-RAG 116 条、91.7% | 同上 retrieval experiment | 已证实 |
| INT4 + FP8 KV Cache 1.78x | 同上 experimental setup | 论文记录 |
| RoboMME 240 回合 | `docs/reports/RAL_CORE_EXPERIMENT_REPORT_20260831.md` | 已证实 |
| 28.7% -> 52.5% | `results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json` | 已证实 |
| Planner -40.6%、wall time -28.1% | 同一主报告 C2 -> C3 | 已证实 |
| PI0.5 P95 282.43 -> 54.67 ms | `docs/status/CARVE_EFFICIENT_INFERENCE_ABLATION_20260826.md` | 已证实 |
| 真机部署 | 无 | 禁止表述 |
| RoboDojo/StarVLA 结果 | 环境配置中 | 禁止表述为已完成 |

## 使用建议

一页中文简历中，优先放“面向VLA的具身智能体系统与高效推理研究”三条；若
版面允许，再放 Agentic RAG-VLM
三条。两个项目不要重复强调“Agentic”，前者突出知识与结构化推理，后者突出
闭环工具链和高效推理。投递具体岗位时只替换关键词和第一条措辞，核心数字与
实验口径保持不变。

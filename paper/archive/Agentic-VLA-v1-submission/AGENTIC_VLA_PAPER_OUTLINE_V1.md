# Agentic-VLA 论文大纲 V1

## 1. 目标定位

- 投稿定位：具身智能 / Vision-Language-Action / 长程操作 / 系统增强型论文
- 稿件形态：`10` 页主稿为目标，保留扩展到 `15` 页上限的余量
- 核心对象：`pi05_libero` fine-tuned VLA
- 核心任务：官方 `LIBERO` benchmark，重点是 `libero_10`

一句话主张：

`Agentic-VLA` 不是重新训练一个更大的 VLA，而是通过显式过渡、结构化先验和反思恢复，为强 VLA policy 补上长程任务所缺失的过程控制能力。

## 2. 推荐标题

可选标题 1：

`Agentic-VLA: Transition-Aware and Reflective Augmentation for Long-Horizon Vision-Language-Action Policies`

可选标题 2：

`Agentic-VLA: Enhancing Fine-Tuned Vision-Language-Action Policies with Transition and Reflective Recovery`

可选标题 3：

`Agentic-VLA: Agentic Process Control for Long-Horizon Manipulation with Fine-Tuned VLA Policies`

## 3. 核心研究问题

论文需要回答三个问题：

1. 为什么强 `fine-tuned VLA` 仍然会在长程任务上失败？
2. 这些失败能否通过 inference-time 的 Agentic 机制修复，而不重新训练 VLA？
3. 哪些 Agentic 模块真正有效，提升体现在哪些弱点任务与恢复行为上？

## 4. 主贡献表述

建议将贡献写成下面三点：

1. 提出一个面向长程 VLA 的 Agentic 增强框架
   - 在冻结的 `fine-tuned VLA` 外部增加可插拔的过程控制机制
   - 不改变主 policy 训练方式

2. 提出三类针对长程失败模式的互补机制
   - `Transition Agent`：缓解 state gap 和动作 chunk 断裂
   - `Scene Priors / Graph RAG + Memory`：注入结构化任务先验
   - `Critic / Retry`：进行失败归因和恢复重试

3. 在官方 `LIBERO` benchmark 上进行真实 rollout 消融
   - 重点分析 `libero_10` 中的弱点任务
   - 提供成功率、恢复统计和失败案例分析

## 5. 页数规划

建议按 `10` 页主稿组织：

1. `Abstract`：0.3 页
2. `Introduction`：1.2 页
3. `Related Work`：0.8 页
4. `Method`：2.0 页
5. `Experimental Setup`：1.0 页
6. `Main Results`：1.2 页
7. `Ablation and Analysis`：1.8 页
8. `Limitations / Discussion / Conclusion`：0.7 页
9. `References`：不计入主稿页数时按模板处理；若计入则预留额外空间

## 6. 论文结构

### 6.1 Abstract

摘要建议四段式压缩成一个自然段：

- 背景：VLA 在标准 manipulation 上表现强，但长程任务仍脆弱
- 方法：提出 `Agentic-VLA`
- 实验：在官方 `LIBERO` benchmark 上进行真实 rollout 消融
- 结论：提升长程鲁棒性与恢复能力，尤其改善弱点任务

### 6.2 Introduction

建议分成 4 个逻辑段：

1. VLA 的进展与价值
2. 长程 manipulation 仍存在的真实问题
3. 我们的方法思想与三个模块
4. 贡献总结

Introduction 里要尽快出现：

- `fine-tuned VLA` 不是万能
- `libero_10` 是主战场
- 我们的方法是 inference-time augmentation

### 6.3 Related Work

建议分为四类：

1. Vision-Language-Action models
2. Long-horizon manipulation and task decomposition
3. Agentic inference and recovery for robotics
4. Retrieval / memory / priors for embodied decision making

写法重点：

- 对 `pi0` / `pi0.5`：强调强基座与 fine-tuning 能力
- 对 `Sci-VLA`：强调 transition 思路的相关性，但指出你的场景与 benchmark、系统设计、消融目标不同
- 对 `VLA2`：强调 agentic augmentation 的 general idea，但你聚焦的是长程过程控制，而不是 unseen concept manipulation

### 6.4 Method

建议按以下小节组织：

#### 4.1 Baseline VLA Execution

- 基于 `pi05_libero` 的 policy server
- action chunk inference
- LIBERO rollout
- 当前 baseline 的局限：state gap、无恢复、弱点 task 脆弱

#### 4.2 Transition Agent

- 输入：末端执行器位置历史、动作推进状态
- 检测：stall / state gap
- 响应：插入 transition prompt 和中间安全动作
- 作用：从 `task A end` 平滑进入 `task B start`

#### 4.3 Scene Priors and Memory

- 从 task description 中抽取对象/场景关键词
- 注入 `z_offset`、`force`、`yaw_hint`
- 若当前版本较轻量，正文可写成 `topology-aware scene priors and episodic memory`

#### 4.4 Critic and Retry

- 周期性检查当前执行状态
- 判断子任务完成、卡滞或失败类型
- 生成 recovery prompt 和 retry prompt

#### 4.5 Unified Execution Loop

- `A1`: baseline
- `A2`: +Transition
- `A3`: +Scene Priors / Memory
- `A4`: +Critic
- `Full`: A2+A3+A4

### 6.5 Experimental Setup

必须明确写：

- 环境：官方 `LIBERO` benchmark
- 主评测集：`libero_10`
- 辅助集：`libero_object`, `libero_goal`
- 基线：`pi05_libero`
- 指标：
  - success rate
  - transition count
  - retry count
  - recovery success rate
  - average episode length
- 所有结果来自真实 rollout，不使用 mock 或估算

### 6.6 Main Results

建议主表如下：

| Method | libero_object | libero_goal | libero_10 |
|---|---:|---:|---:|
| A1 | 98.0 | 98.0 | 90.0 |
| A2 | TBD | TBD | TBD |
| A3 | TBD | TBD | TBD |
| A4 | TBD | TBD | TBD |
| A2+A4 | TBD | TBD | TBD |
| Full | TBD | TBD | TBD |

主结论目标：

- 在简单 suite 上尽量不降级
- 在 `libero_10` 上出现稳定增益
- 在弱点 task 上体现更明显提升

### 6.7 Ablation and Analysis

这部分决定论文质量，建议拆成四小节：

1. `Transition` 是否真的修复长程断裂？
2. `Critic / Retry` 是否把失败 episode 拉回成功？
3. `Scene Priors / Memory` 是否对特定 task 有帮助？
4. `Task 8` 等弱点任务的失败模式发生了什么变化？

建议表和图：

- `libero_10` 分 task 成功率表
- retry / transition 统计表
- 成功与失败视频帧对比图
- 弱点 task 失败类型分布图

### 6.8 Discussion / Limitations

建议主动承认：

- `A3` 当前更接近结构化先验与 episodic memory，而不是重型完整 Graph RAG
- `A4` 带来额外推理成本
- 当前主要验证于 LIBERO，真实机器人迁移还需后续工作

### 6.9 Conclusion

结论只强调三点：

- 强 VLA 在长程任务上仍有系统性缺口
- Agentic inference-time augmentation 是一条有效路线
- `Transition + Recovery` 是当前最关键的提升来源

## 7. 图表规划

建议至少准备以下图表：

### 图 1：方法总览图

- 输入 observation + prompt
- baseline VLA 输出 action chunks
- Transition / Priors / Critic 三个外挂模块
- 回到 unified execution loop

### 图 2：长程 state gap 示意图

- 从 `subtask A` 到 `subtask B` 的断裂
- 加入 transition 后的平滑衔接

### 图 3：弱点任务案例图

- baseline failure
- Agentic-VLA success

### 表 1：总体成功率

- A1/A2/A3/A4/A2+A4/Full

### 表 2：`libero_10` 分 task 结果

- 突出 `Task 8`

### 表 3：机制统计

- average transitions
- average retries
- recovery success
- average episode length

## 8. 当前可直接写进论文的真实信息

已确认可作为论文写作依据的内容：

- `A1 / libero_object = 98.0%`
- `A1 / libero_goal = 98.0%`
- `A1 / libero_10 = 90.0%`
- `Task 8 = 55%` 是关键弱点

当前正在运行、后续可补入论文的内容：

- `A2 Transition` 的真实 `libero_10` 长跑
- 当前已出现 `task0` 的 success episodes，可作为早期正向迹象，但还不能写成正式结论

## 9. 当前写作红线

- 不把 `libero_10` 写成我们自定义 benchmark
- 不把 `Vision Prompt` 写成主创新
- 不引用旧文档中不可追溯的成功率
- 不把 `A3` 过度宣称为完整成熟 Graph RAG，除非后续证据补足

## 10. 建议的下一写作步

1. 基于模板建立 `LaTeX` 主文件
2. 先写 `Abstract + Introduction + Method + Experimental Setup`
3. 留出表格占位
4. 等 `A2/A4` 结果出来后填 `Results` 与 `Analysis`


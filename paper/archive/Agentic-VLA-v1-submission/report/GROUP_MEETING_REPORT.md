# Agentic-VLA 组会汇报

- 题目：`Agentic-VLA: Inference-Time Process Control for Long-Horizon Vision-Language-Action Policies`

---

## 1. 解决的问题

### 现象
- 现在的强 VLA policy，在短程 manipulation 上已经很强
- 但一到 long-horizon task，仍然会出现明显失败
- 这些失败经常不是“不会某个基础动作”
- 而是：
  - 子任务之间接不上
  - 中间状态偏掉后无法拉回
  - 任务还没结束，但 rollout 已经开始空转或漂移

### 判断
- long-horizon 的瓶颈，不一定首先是 backbone 不够强
- 很可能是执行阶段缺少一个显式的过程控制层

### 这篇工作的中心问题

> 对一个已经很强的 frozen VLA policy，能不能不重训 backbone，只通过推理时的过程控制，让 long-horizon 执行更稳定？

---

## 3. 整体思路

### 核心思想

- 不去重新训练 VLA backbone
- 而是在推理时，给它外挂一个 process-control layer

### 这个 process-control layer 由什么组成？

- 主控制器：`Frozen VLA Policy`
- 三个外部模块：
  - `Transition Agent`：负责在子任务边界补一个中间连接动作
  - `Scene Priors and Memory`：负责补充任务相关先验，如场景描述、任务目标等
  - `Critic / Retry`：负责检查是否偏离目标，以及是否需要尝试恢复

### 方法定位

- 不是一个新的 backbone
- 不是一个新的 end-to-end controller
- 而是一个外挂在强 VLA 外部的执行控制框架

- 而是：
  - 何时需要补一个中间连接动作
  - 何时需要补场景先验
  - 何时应该检查是否偏离目标
  - 何时应该尝试恢复

---

## 4. 整体执行流程

### Step 1：主 policy 正常执行

- VLA 根据 `observation + robot state + task instruction` 生成 action chunk

### Step 2：执行过程中持续监控

- 看当前是否在推进
- 看是否接近 subtask boundary
- 看是否出现明显偏离或失败迹象

### Step 3：按条件触发不同模块

- 卡在子任务边界：触发 `Transition Agent`
- 需要补充任务相关先验：触发 `Scene Priors and Memory`
- 怀疑已经偏离目标：触发 `Critic / Retry`

### Step 4：执行权回到主 policy

- 三个模块都不是主控制器
- 它们的角色是“修正执行过程”
- 最终动作仍由 frozen VLA 生成

### Step 5：环境判定是否成功

- success 不是模型自评
- 而是官方环境中的 `env.step(...)->done`

---

## 5. 模块一：Transition Agent

### 这个模块要解决什么问题？

- 很多 long-horizon failure 不是 skill 不会
- 而是前一个子任务结束后，当前状态不适合直接进入下一个子任务
- 结果就是：
  - 来回抖动
  - 停滞
  - 明明没完成，但已经不再有效推进

### 如何判断卡住了？

- 监控最近一段时间的末端位移
- 如果位移很小、任务又没完成，怀疑 rollout 卡在了 subtask boundary

### 怎么做？

- 临时注入一个 transition prompt
- 让主 policy 先输出一个短的 reconnection motion
- 这个动作的目标不是直接完成任务
- 而是把状态带回“可继续执行”的区域

### 可以怎么理解？

- 它不是在替代主 policy
- 而是在子任务边界给 policy 一个“重新接上动作链”的机会

### 这部分为什么重要？

- 因为 long-horizon 里的很多失败，本质上是 transition failure
- 这个模块补的是“子任务之间的连接”

---

## 6. 模块二：Scene Priors and Memory

### 要解决什么问题？

- policy 往往知道“做什么”
- 但不一定稳定知道“怎样做更稳”
- 比如：
  - 从哪边接近更稳
  - 抓取时该保留多少高度偏移
  - 对某类物体是否需要更保守的姿态

### 输入：

- 当前任务描述
- 任务相关的 object / target / region
- 一些轻量的场景和历史成功 cue

### 输出：

- 不是动作
- 而是一些结构化执行先验：
  - approach direction
  - vertical offset
  - force bias
  - episodic cue

### 如何作用？

- 把这些先验压缩成 prompt augmentation
- 再交给主 policy 去出动作

### 模块本质

- 不是替模型思考
- 而是把“模型未必稳定利用、但对执行很关键的信息”显式补回执行过程

---

## 7. 模块三：Critic / Retry

### 要解决什么问题？

- long-horizon 任务里，错误会累积
- 一旦局部状态偏掉，如果没有恢复机制，整条 rollout 会一起失败
- 给 long-horizon 执行增加一个条件触发的恢复闭环

### 何时接入：

- 不是每一步都介入
- 只有在怀疑 rollout 已经偏离、停滞或局部失败时才介入

### 如何作用？

- `critic`：判断当前是否正常推进
- `recovery`：先把状态拉回更安全的位置
- `retry`：再从更好的中间状态重新尝试

---

## 9. 实验设置

### 实验环境

- official LIBERO simulator
- interactive rollout evaluation
- success 来自环境真实终止信号 `env.step(...)->done`

### 基线

- `pi05_libero_10`
- `Full Agentic-VLA`

### head-to-head 主结果

- `baseline = 90.0%`
- `Full Agentic-VLA = 91.0%`

---

## 11. 分任务结果怎么看？

### 提升明显的任务

- `T0`
- `T2`
- `T3`
- `T5`

### 基本持平的任务

- `T1`
- `T4`
- `T6`
- `T7`

### 当前最困难的区域

- `T8`
- `T9` 也略低于 baseline

### 主结果告诉我们的不是

- “已经完全解决 long-horizon”

### 而是

- “整体有真实增益”
- “但收益是 task-dependent”
- “最难的弱点任务还需要额外诊断”

---

## 12. 为什么还要补弱点任务实验？

### 原因

- 主结果给出的是总体结论
- 但它同时暴露出：
  - `Task 8` 是最强 failure region
  - `Task 9` 也存在弱点

### 所以我后面做了什么？

- 对弱点任务做 targeted diagnosis
- 重点补了：
  - `Task 8`
  - `Task 9`
  - `Task 6`

### 目标不是改写主文主结果

- 而是回答两个问题：
  - 我们定位到的问题到底对不对？
  - 哪些模块真的在起作用？

---

## 13. 弱点任务 follow-up 结果

### `Task 8`

- baseline：`55%`
- `Full = 70%`
- `Transition only = 80%`
- `Graph only = 60%`
- `Critic only = 70%`

### `Task 9`

- baseline：`95%`
- `Full = 100%`
- `Transition only = 90%`
- `Graph only = 100%`
- `Critic only = 90%`

### `Task 6`

- baseline：`80%`
- `Full = 90%`
- `Transition only = 100%`
- `Graph only = 80%`
- `Critic only = 90%`

### 这组结果说明

- 针对弱点任务做 refinement 是有效的
- 但各模块作用并不完全一致
- 不同任务上，最有效的模块可能不一样

### 诊断性上界（不是单次 run）

- 作为“潜力上界”诊断，我另外计算了一个 selective refinement 视图：
  - 只对 `T6/T8/T9` 使用 refined follow-up 的 Full 结果
  - 其他 task 仍沿用 main head-to-head 的 Full 结果
  - 然后做 task-average 汇总
- 结果：task-average 约为 `96%`（它是“如果 refined 对其余 task 不退化”的诊断性估计）
- 重要说明：
  - 这不是一个可部署的单一方法结果
  - 它混合了不同 run（把 refinement 只插到弱 task 上），只能用来说明“整体分数对弱 task 修复的敏感性，以及 refinement 的潜在上界”

---

## 14. Task 8 的关键诊断

### 为什么 Task 8 要单独重点分析？

- 它是整个 `libero_10` 里最明显的 weak-task
- 也是最能反映框架协同是否合理的任务

### 我做了什么？

- 在 `Task 8` 上进一步做了 `w/o` 诊断：
  - `w/o Graph+Memory = 60%`
  - `w/o Critic = 70%`
  - `w/o Transition = 80%`

### 这个结果非常关键

- 去掉 `Graph+Memory` 会掉到 `60%`
  - 说明这部分不是负资产
- 去掉 `Critic` 还是 `70%`
  - 说明当前 critic 贡献偏弱
- 去掉 `Transition` 反而是 `80%`
  - 说明现在的问题不是“某个模块完全没用”
  - 而是当前 full-stack 的协同还没有调到最佳

### 这件事的学术意义

- 它说明这不是一个简单的“模块越多越好”问题
- 而是模块之间存在 non-monotonic interaction

---

## 15. 当前实验结论

### 可以明确说的

- `Full` 相比 baseline 在主任务上有真实提升
- weak-task refinement 是有效的
- `Graph + Memory` 在部分弱任务上有正向贡献
- `Critic` 当前仍 under-activated
- 当前 full-stack 的主要问题是模块协同，而不是单模块是否存在

### 不能夸张说的

- 不能说已经全面解决 long-horizon
- 不能说每个模块都已经被充分验证
- 不能说 current critic 已经形成成熟 recovery 闭环

### 当前最稳的结论

> 对强 frozen VLA policy 引入 inference-time process control，能够在真实 rollout 中带来 modest but real 的收益；其中 transition 和 scene priors/memory 已显示价值，而 critic 与 full-stack 协同仍需继续做实。

---

## 16. 这篇工作的贡献怎么讲？

### 第一层：问题重新表述

- 不是只盯着 backbone 够不够强
- 而是把问题转成：
  - long-horizon 执行是否缺少显式过程控制？

### 第二层：系统设计

- 给 frozen VLA 外挂一个 modular、auditable 的执行控制层
- 模块之间职责明确
- 每个模块可单独分析

### 第三层：实验价值

- 主结果有真实提升
- follow-up 实验能解释弱点任务
- 不只是“涨点”，还能告诉我们机制在哪里生效、哪里还不成熟

---

## 17. 当前局限与下一步

### 当前局限

- 主结果提升不大，只有 `1` 个点
- `Critic / Retry` 没有完全立起来
- full-stack 组合还没有达到最优协同

### 下一步最自然的方向

- 继续打磨 `Graph + Memory`
- 重点强化 `Critic / Retry` 的真实 recovery 闭环
- 从会议版继续扩成更完整的期刊版

### 这意味着

- 当前会议稿可以先聚焦：
  - process control 这个大方向是成立的
- 后续再把 `Graph` 和 `Critic` 做强

---

## 18. 汇报收尾

> 这篇工作的核心，不是证明一个更大的 backbone，而是证明：对强 VLA policy 做 inference-time process control，是一条真实有效、可审计、并且值得继续深入的方向。

---

## 19. 组会展示时建议打开的材料

- 论文 PDF：`paper/Agentic-VLA/agentic_vla_paper_v1.pdf`
- 方法图：
  - `paper/Agentic-VLA/figures/fig.1-gemini.png`
  - `paper/Agentic-VLA/figures/fig.2-gemini.png`
  - `paper/Agentic-VLA/figures/fig.3-gemini.png`
  - `paper/Agentic-VLA/figures/fig.4-gemini.png`
- 结果图：
  - `paper/Agentic-VLA/figures/fig_libero10_task_comparison.png`
- 主结果目录：
  - `results/ablation_B0_pi05_libero_10_20260512/summary.json`
  - `results/ablation_FULL_tuned_libero10_20260513_225055/summary.json`
- follow-up 证据：
  - `results/ablation_FULL_tuned_v2_task8_20260515_160827/summary.json`
  - `results/ablation_FULL_tuned_v2_wo_graph_task8_20260516_112247/summary.json`
  - `results/ablation_FULL_tuned_v2_wo_critic_task8_20260516_112247/summary.json`
  - `results/ablation_FULL_tuned_v2_wo_transition_task8_20260516_112247/summary.json`

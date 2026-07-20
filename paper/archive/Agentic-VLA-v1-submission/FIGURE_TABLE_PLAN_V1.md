# Agentic-VLA 图表规划 V1

## 1. 目标

本文件用于把论文中的图表一次性规划清楚，避免实验结果出来后临时拼凑。

核心原则：

- 主图服务于方法和故事
- 主表服务于结论
- 所有图表围绕“长程弱点修复”组织

## 2. 推荐主图

### 图 1：Agentic-VLA 方法总览图

用途：

- 作为整篇论文最核心的方法图
- 放在 `Introduction` 或 `Method` 开头

建议内容：

1. 输入：
   - `agentview image`
   - `wrist image`
   - `robot state`
   - `language instruction`
2. 基座：
   - `pi05_libero` policy server
3. 外挂模块：
   - `Transition Agent`
   - `Scene Priors / Memory`
   - `Critic / Retry`
4. 执行闭环：
   - action chunk generation
   - environment rollout
   - progress monitoring
   - replan / retry / resume

图中必须突出：

- `frozen fine-tuned VLA`
- `inference-time augmentation`
- `modular process control`

### 图 2：State Gap 与 Transition 示意图

用途：

- 直观说明 A2 为什么合理

建议内容：

- 左图：baseline 在 `subtask A -> subtask B` 切换处停滞或抖动
- 右图：Transition 插入中间状态后恢复连续执行

### 图 3：弱点任务案例对比图

用途：

- 用于 `Ablation and Analysis`
- 最好选 `libero_10 Task 8`

建议内容：

- baseline failure 关键帧
- A2 或 Full success 关键帧
- 在图注中解释：
  - baseline 失败位置
  - transition / retry 在哪里介入

### 图 4：Critic / Retry 流程图

用途：

- 解释 A4 不是替代 policy，而是周期性诊断和局部恢复

建议内容：

- rollout state
- critic judgment
- healthy progress vs likely failure
- recovery / retry
- mechanism statistics

### 图 5：失败类型 taxonomy 图

用途：

- 把方法模块与失败模式一一对应

建议内容：

- transition failure
- contact / grasp failure
- completion failure
- recovery failure
- 模块映射：A2 / A3 / A4

### 图 6：真实 rollout 证据链图

用途：

- 强调所有数字来自官方 LIBERO 真实执行

建议内容：

- benchmark suites
- ablation variants
- OpenPI rollout
- per-episode outputs
- task-level summaries
- paper tables / figures

## 3. 推荐主表

### 表 1：总体成功率主表

用途：

- 论文主结果表
- 最关键表格

建议列：

- `libero_object`
- `libero_goal`
- `libero_10`

建议行：

- `A1`
- `A2`
- `A3`
- `A4`
- `A2+A4`
- `Full`

推荐强调：

- `libero_10` 上的增益
- 简单 suite 上不明显降级

### 表 2：libero_10 分 task 结果

用途：

- 展示长程弱点修复

建议列：

- `Task 0 ... Task 9`

必须强调：

- `Task 8`
- 其他明显低于 95% 的 task

### 表 3：机制统计表

用途：

- 回答“为什么提升”

建议列：

- `Avg. Transitions`
- `Avg. Retries`
- `Recovery Success`
- `Avg. Episode Length`

建议行：

- `A1`
- `A2`
- `A4`
- `Full`

## 4. 推荐分析图

### 图 7：弱点任务成功率条形图

用途：

- 强调 `Task 8` 之类的关键弱点任务

建议内容：

- baseline vs A2 vs A4 vs Full

### 图 8：失败类型分布图

用途：

- 强化 qualitative + quantitative 分析

建议失败类别：

- `target localization error`
- `contact/grasp failure`
- `transition failure`
- `recovery failure`

### 图 9：代价分析图

用途：

- 对 `Critic` 和 `Full` 的额外时延给出解释

建议内容：

- baseline / A2 / A4 / Full 的平均 episode 时间
- 或 average step latency

## 5. 当前最适合的最终排版顺序

建议图表顺序如下：

1. 图 1：方法总览
2. 图 2：state gap / transition 示意
3. 图 4：Critic / Retry 流程
4. 图 5：失败 taxonomy
5. 表 1：总体成功率
6. 图 6：真实 rollout 证据链
7. 表 2：`libero_10` 分任务结果
8. 表 3：机制统计
9. 图 3：弱点任务案例图
10. 图 7 或图 8：分析图

## 6. 每个图表回答的问题

### 图 1

回答：

- 你方法到底是什么

### 表 1

回答：

- 你方法整体有没有用

### 表 2

回答：

- 你方法具体在哪些长程 task 上有用

### 表 3

回答：

- 你为什么有用

### 图 3

回答：

- 你不是“数字碰巧提升”，而是真的改变了执行行为

### 图 4

回答：

- A4 到底如何介入执行

### 图 5

回答：

- 你的模块分别在解决什么问题

### 图 6

回答：

- 你的数字是否真实、可审计、可追溯

## 7. 当前最建议先做的素材准备

1. 从 `A2` 的视频中截取 `task0` 的 success / failure 对比帧
2. 为后续 `Task 8` 预留对比目录
3. 在实验脚本中准备导出：
   - transition 次数
   - retry 次数
   - per-task success
   - average episode length

## 8. 当前已具备的真实素材

当前已经存在可直接用于 Figure 3 草图设计的真实 rollout 视频：

- `results/ablation_A2_transition_libero10_20260513/videos/task0_trial0_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
- `results/ablation_A2_transition_libero10_20260513/videos/task0_trial1_success_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
- `results/ablation_A2_transition_libero10_20260513/videos/task0_trial2_success_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
- `results/ablation_A2_transition_libero10_20260513/videos/task0_trial3_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`

建议：

- 先从 `trial0 failure` 和 `trial1/2 success` 中各截 3--4 帧
- 图注突出 `baseline-style stall / transition intervention / eventual completion`
- 等 `Task 8` 真正跑出结果后，再把 Figure 3 的主案例从 `Task 0` 切换到更能体现弱点修复的 task

## 9. 论文写作提醒

- 图 1 一定要足够清楚，这是审稿人最快理解你方法的入口
- 表 1 一定不能塞太多列，否则主结论不清楚
- `libero_10` 分 task 表必须出现，这是你故事的证据核心
- 若 A3 增益不稳定，可以在主表保留，但不要让它抢主图主表叙事中心

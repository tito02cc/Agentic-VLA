# Fig. 4 Prompt

## 用途

用于生成 `Critic / Retry` 的机制图，说明 critic 是一个条件触发的辅助诊断与恢复模块，而不是替代 policy 的主控制器。

目标是让读者看懂：

- 正常情况下 rollout 直接继续
- 只有检测到异常时 critic 才介入
- critic 负责 failure assessment、recovery、retry re-entry
- 整个过程有可审计的日志或统计信息

## 期望风格

- 机制图，不是软件流程图
- 三个视觉区域，结构清楚但不过度工程化
- 白底、矢量、学术化
- 可以使用少量 warning/check/recovery 图标增强可读性
- 健康分支要比失败分支更轻、更顺

## 必须包含的元素

- 三个区域：
  - `current rollout execution`
  - `critic-based progress assessment`
  - `recovery / retry re-entry`
- 正常执行分支
- 异常诊断分支
- `conditional checking`
- `recovery prompt`
- `retry re-entry`
- 一个小型 `logging / audit statistics` 元素

## 避免出现

- `Qwen3-VL` 等具体 backend 名称
- 代码级判断条件
- 过多矩形框和箭头
- BPMN 或企业流程图观感
- 大量文字解释

## 可直接复制给 Nano Banana 的最终提示词

```text
Create a publication-quality scientific mechanism figure for a robotics paper that explains the "Critic / Retry" module as a conditional intervention schematic, not as a dense software flowchart. Use white background, flat academic vector style, restrained colors, rounded panels, modern sans-serif typography, and a clean visual hierarchy.

Organize the figure into three balanced regions from left to right: "Current Rollout Execution", "Critic-Based Progress Assessment", and "Recovery / Retry Re-entry". In the first region, show a robotic arm performing a tabletop manipulation rollout under normal policy execution. In the second region, show a conditional critic check that monitors progress and detects likely inconsistency, incompletion, or failure. Use a subtle check symbol for healthy progress and a small warning symbol for suspected failure. In the third region, show a recovery prompt and retry re-entry path that guides execution back to a safer and more controllable state before normal rollout resumes.

Visually make it clear that the healthy branch is simpler and more direct, while the recovery branch is only activated when needed. The critic must appear auxiliary and conditional, not the primary controller. Add a small unobtrusive logging or audit statistics panel to suggest traceability of checks, retries, and outcomes.

Use simplified robotic and tabletop elements so the figure feels embodied and intuitive, but avoid excessive arrows, too many stacked boxes, backend model names, or code-like logic. The final figure should look like a clean CCF-A paper mechanism illustration rather than a PowerPoint process chart.
```

## 生成不理想时的二次追加指令

如果太像企业流程图：

```text
Make it look more like a scientific mechanism illustration and less like a business workflow or BPMN chart.
```

如果失败分支不够清晰：

```text
Strengthen the contrast between the simple healthy branch and the conditional recovery branch, while keeping the figure minimal.
```

如果图里字太多：

```text
Reduce text density, keep only the main three regions and the essential labels for critic check, recovery, and retry re-entry.
```

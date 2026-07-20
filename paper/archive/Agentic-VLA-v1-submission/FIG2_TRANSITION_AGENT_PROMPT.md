# Fig. 2 Prompt

## 用途

用于生成 `Transition Agent` 的机制图，解释它如何修复 long-horizon rollout 中的 `state-gap` / `subtask boundary` 失败。

目标是让读者看懂：

- baseline 的 chunked execution 在子任务边界可能停滞
- `Transition Agent` 会插入一个中间 reconnection motion
- 插入后，正常策略执行可以继续恢复

## 期望风格

- 论文机制图，不是结果图
- 双栏或左右两 panel 布局
- 白底、矢量风格、学术化、简洁
- 机械臂和桌面场景可以简化，但要可辨认
- 轨迹箭头和状态区域要直观

## 必须包含的元素

- 两个 panel：
  - `State-Gap Failure Pattern`
  - `Transition Repair Mechanism`
- `Subtask A stable region`
- `subtask boundary`
- `Subtask B stable region`
- baseline 轨迹在边界附近停滞或失去连续性
- transition prompt 触发后插入中间轨迹
- 修复后继续正常执行

## 避免出现

- 柱状图、折线图、成功率数字
- `improved`, `better`, `+x%` 这种结果导向文字
- 过于复杂的厨房/桌面细节
- 太多小字说明
- 像 PPT 动画步骤图

## 可直接复制给 Nano Banana 的最终提示词

```text
Create a publication-quality scientific mechanism figure for a robotics paper with two side-by-side panels. Use a clean academic vector style with white background, thin outlines, restrained color palette, modern sans-serif typography, and limited text. The figure should match the visual family of a top-tier robotics paper method figure, not a benchmark chart or slide deck.

Left panel title: "State-Gap Failure Pattern". Right panel title: "Transition Repair Mechanism".

Depict a simplified long-horizon tabletop manipulation scene with a robotic arm or end-effector trajectory. In the left panel, show chunked execution moving successfully through "Subtask A stable region", then reaching a "subtask boundary" where state continuity is lost and motion stalls before entering "Subtask B stable region". Use trajectory arrows, a slight stall symbol, and a broken or hesitant path to make the failure visually intuitive.

In the right panel, show the same scene but now the system issues a "transition prompt" and inserts one intermediate reconnection motion between the two stable regions. This inserted motion should restore executable state continuity, after which the normal trajectory resumes toward "Subtask B stable region". The repair should feel like a brief neutral or corrective repositioning step rather than a whole new controller.

Use simplified robotic and tabletop elements so the mechanism feels concrete and embodied, but keep the image minimal and publication-ready. This is a method explanation figure only. Do not include metrics, success rates, benchmark styling, or result annotations.
```

## 生成不理想时的二次追加指令

如果太抽象：

```text
Add a clearer robotic arm silhouette, a tabletop object, and more intuitive trajectory arrows, while keeping the figure minimal and academic.
```

如果太像结果图：

```text
Remove any quantitative appearance and make it purely a mechanism explanation figure with no metric or performance implication.
```

如果太拥挤：

```text
Reduce text, simplify the background, and keep only the essential state regions, boundary marker, and transition repair trajectory.
```

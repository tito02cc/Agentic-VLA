# Fig. 1 Prompt

## 用途

用于生成论文的总览框架图，对应 `Agentic-VLA` 的整体方法示意图。

目标是让审稿人一眼看懂：

- 冻结的 VLA policy 仍然是核心动作生成器
- `Transition Agent`、`Scene Priors and Memory`、`Critic / Retry` 都是推理时外挂模块
- 整套系统通过官方 `LIBERO` 模拟器 rollout 闭环运行，且结果可审计

## 期望风格

- 顶会论文插图风格，不要像海报或产品宣传图
- 白底，扁平化矢量风格，干净的学术配色
- 现代 sans-serif 字体，少量文字，清晰层次
- 可以有简化机械臂、桌面场景、小图标，但不能花哨
- 尽量左右或环形均衡布局，减少箭头交叉

## 必须包含的元素

- `Frozen VLA Policy` 作为视觉中心
- 输入端：`RGB observation`、`robot state / proprioception`、`language instruction`
- 三个外部模块：
  - `Transition Agent`
  - `Scene Priors and Memory`
  - `Critic / Retry`
- 右侧执行闭环：
  - `action chunk`
  - `rollout`
  - `env.step`
  - `done-based success`
  - `Official LIBERO Simulator`
- 明确体现：
  - `frozen backbone`
  - `inference-time augmentation`
  - `auditable rollout loop`

## 避免出现

- 太多说明性段落
- 太多小框和工程实现细节
- `websocket`、`server port`、变量名、脚本名
- 类似 benchmark 图表的坐标轴
- 赛博风、霓虹风、3D 写实渲染、卡通风

## 可直接复制给 Nano Banana 的最终提示词

```text
Create a publication-quality scientific framework overview figure for a top-tier robotics paper. The figure should explain the full Agentic-VLA system in a clean academic vector style with white background, thin precise outlines, restrained colors, balanced spacing, modern sans-serif typography, and no decorative background.

Center the figure around a large block labeled "Frozen VLA Policy" to emphasize that the backbone remains unchanged and is still the main action generator. On the left, place three compact input blocks labeled "RGB Observation", "Robot State / Proprioception", and "Language Instruction". Around the central frozen policy, place three external inference-time modules with equal visual importance: "Transition Agent", "Scene Priors and Memory", and "Critic / Retry". These modules should visually appear as wrappers or auxiliary controllers around the frozen backbone, not as replacements for it.

On the right, show a unified execution loop connected to an "Official LIBERO Simulator" block. The loop should include concise stages such as "Action Chunk", "Rollout", "env.step", and "done-based success". Use arrows to show observation update, action execution, feedback, and continued rollout. Add concise academic labels such as "frozen backbone", "inference-time augmentation", and "auditable rollout loop".

Use tasteful schematic visual elements such as a simplified robotic arm, a tabletop manipulation scene, observation icons, and subtle rollout arrows so the framework feels concrete and intuitive. Keep the layout symmetric or gently circular, minimize arrow crossings, and limit text per block. The final figure must look like a rigorous CCF-A conference paper method figure, not a product infographic, poster, cartoon, or software architecture slide.
```

## 生成不理想时的二次追加指令

如果第一次生成太花：

```text
Make it more academic and minimal. Reduce decorative elements, reduce arrow count, shorten labels, and improve visual hierarchy.
```

如果第一次生成太抽象：

```text
Add a simplified robotic arm, a tabletop scene, and small observation icons, but keep the style flat, restrained, and publication-ready.
```

如果第一次生成像工程流程图：

```text
Make it look less like a software flowchart and more like a scientific mechanism overview figure for a robotics paper.
```

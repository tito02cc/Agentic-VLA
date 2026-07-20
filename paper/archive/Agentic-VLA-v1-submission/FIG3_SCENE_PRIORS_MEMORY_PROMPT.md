# Fig. 3 Prompt

## 用途

用于生成 `Scene Priors and Memory` 的机制图，说明任务实体如何被转成结构化 priors 和 episodic cues，并最终通过 prompt augmentation 影响冻结策略。

目标是让读者看懂：

- 这不是一个庞大的检索系统
- 它是轻量、可解释、推理时注入的 structured hints
- 它会把对象、目标区域、成功经验转成 prompt side information

## 期望风格

- 左到右的干净 pipeline 图
- 轻量、清爽，比总览图更简洁
- 白底、扁平矢量风格、学术配色
- 小图标和小场景可以有，但不能堆太满

## 必须包含的元素

- 左到右 4 个关键块：
  - `Task Prompt + Scene Entities`
  - `Structured Priors`
  - `Episodic Memory`
  - `Prompt Augmentation`
- 在 `Structured Priors` 中体现：
  - `approach direction`
  - `vertical offset`
  - `force bias`
  - 可选 `target-region hint`
- 在 `Episodic Memory` 中体现：
  - successful interaction cues
  - reusable hints
- 最终指向 frozen policy
- 强调它的作用是：
  - `bias frozen policy toward more stable interaction patterns`

## 避免出现

- 大型数据库、云端检索、RAG 服务器农场式画法
- 很像 LLM 系统架构图
- 太多段落文字
- 把这个模块画成新的主模型

## 可直接复制给 Nano Banana 的最终提示词

```text
Create a clean publication-quality scientific block diagram for a robotics paper, using white background, flat vector academic style, rounded blocks, short labels, restrained colors, and a clear left-to-right pipeline. The figure should explain a lightweight "Scene Priors and Memory" module for Agentic-VLA.

Arrange four main blocks from left to right: "Task Prompt + Scene Entities", "Structured Priors", "Episodic Memory", and "Prompt Augmentation". Use subtle arrows to show information flow toward a frozen policy controller.

In "Task Prompt + Scene Entities", visually suggest a language instruction together with task-relevant objects and target regions from a tabletop scene. In "Structured Priors", visually encode concepts such as approach direction, vertical offset, force bias, and target-region hints using concise icons or tiny schematic annotations rather than long text. In "Episodic Memory", show small reusable successful interaction cues or compact memory tokens, but do not depict a large database or retrieval server. In "Prompt Augmentation", show that these lightweight priors and memory cues are merged into the prompt context before action generation.

Add a concise bottom caption-like note inside the figure: "bias frozen policy toward more stable interaction patterns". The figure should feel lightweight, interpretable, and execution-time oriented, not like a huge standalone model. Use small object icons and mini scene cues so the module feels concrete, but keep the overall layout minimal and publication-ready.
```

## 生成不理想时的二次追加指令

如果太像数据库系统：

```text
Make it lighter and more embodied. Replace database-like visuals with small reusable successful interaction cues and simple scene symbols.
```

如果太空：

```text
Add small object icons, target-region hints, and tiny trajectory or force symbols, while keeping the layout clean and minimal.
```

如果太重文字：

```text
Reduce text density and express priors through icons and concise labels instead of sentences.
```

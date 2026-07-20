# Nano Banana 4-Session Prompts

说明：

- 下面一共 4 段 prompt
- 每一段对应一个独立会话
- 直接整段复制到 `nano banana` 即可
- 如果你先生成了 `Fig.1`，建议在 `Fig.2` 到 `Fig.4` 的会话里把 `Fig.1` 生成结果作为 style reference 一并上传

---

## Session 1: Fig. 1 Overall Framework

```text
Create a publication-quality scientific framework overview figure for a top-tier robotics paper. The figure should explain the full Agentic-VLA system in a clean academic vector style with white background, thin precise outlines, restrained colors, balanced spacing, modern sans-serif typography, and no decorative background.

Center the figure around a large block labeled "Frozen VLA Policy" to emphasize that the backbone remains unchanged and is still the main action generator. On the left, place three compact input blocks labeled "RGB Observation", "Robot State / Proprioception", and "Language Instruction". Around the central frozen policy, place three external inference-time modules with equal visual importance: "Transition Agent", "Scene Priors and Memory", and "Critic / Retry". These modules should visually appear as wrappers or auxiliary controllers around the frozen backbone, not as replacements for it.

On the right, show a unified execution loop connected to an "Official LIBERO Simulator" block. The loop should include concise stages such as "Action Chunk", "Rollout", "env.step", and "done-based success". Use arrows to show observation update, action execution, feedback, and continued rollout. Add concise academic labels such as "frozen backbone", "inference-time augmentation", and "auditable rollout loop".

Use tasteful schematic visual elements such as a simplified robotic arm, a tabletop manipulation scene, observation icons, and subtle rollout arrows so the framework feels concrete and intuitive. Keep the layout symmetric or gently circular, minimize arrow crossings, and limit text per block. The final figure must look like a rigorous CCF-A conference paper method figure, not a product infographic, poster, cartoon, or software architecture slide.
```

---

## Session 2: Fig. 2 Transition Agent

```text
Create a publication-quality scientific mechanism figure for a robotics paper with two side-by-side panels. Use a clean academic vector style with white background, thin outlines, restrained color palette, modern sans-serif typography, and limited text. The figure should match the visual family of a top-tier robotics paper method figure, not a benchmark chart or slide deck.

Left panel title: "State-Gap Failure Pattern". Right panel title: "Transition Repair Mechanism".

Depict a simplified long-horizon tabletop manipulation scene with a robotic arm or end-effector trajectory. In the left panel, show chunked execution moving successfully through "Subtask A stable region", then reaching a "subtask boundary" where state continuity is lost and motion stalls before entering "Subtask B stable region". Use trajectory arrows, a slight stall symbol, and a broken or hesitant path to make the failure visually intuitive.

In the right panel, show the same scene but now the system issues a "transition prompt" and inserts one intermediate reconnection motion between the two stable regions. This inserted motion should restore executable state continuity, after which the normal trajectory resumes toward "Subtask B stable region". The repair should feel like a brief neutral or corrective repositioning step rather than a whole new controller.

Use simplified robotic and tabletop elements so the mechanism feels concrete and embodied, but keep the image minimal and publication-ready. This is a method explanation figure only. Do not include metrics, success rates, benchmark styling, or result annotations.
```

---

## Session 3: Fig. 3 Scene Priors and Memory

```text
Create a clean publication-quality scientific block diagram for a robotics paper, using white background, flat vector academic style, rounded blocks, short labels, restrained colors, and a clear left-to-right pipeline. The figure should explain a lightweight "Scene Priors and Memory" module for Agentic-VLA.

Arrange four main blocks from left to right: "Task Prompt + Scene Entities", "Structured Priors", "Episodic Memory", and "Prompt Augmentation". Use subtle arrows to show information flow toward a frozen policy controller.

In "Task Prompt + Scene Entities", visually suggest a language instruction together with task-relevant objects and target regions from a tabletop scene. In "Structured Priors", visually encode concepts such as approach direction, vertical offset, force bias, and target-region hints using concise icons or tiny schematic annotations rather than long text. In "Episodic Memory", show small reusable successful interaction cues or compact memory tokens, but do not depict a large database or retrieval server. In "Prompt Augmentation", show that these lightweight priors and memory cues are merged into the prompt context before action generation.

Add a concise bottom caption-like note inside the figure: "bias frozen policy toward more stable interaction patterns". The figure should feel lightweight, interpretable, and execution-time oriented, not like a huge standalone model. Use small object icons and mini scene cues so the module feels concrete, but keep the overall layout minimal and publication-ready.
```

---

## Session 4: Fig. 4 Critic / Retry

```text
Create a publication-quality scientific mechanism figure for a robotics paper that explains the "Critic / Retry" module as a conditional intervention schematic, not as a dense software flowchart. Use white background, flat academic vector style, restrained colors, rounded panels, modern sans-serif typography, and a clean visual hierarchy.

Organize the figure into three balanced regions from left to right: "Current Rollout Execution", "Critic-Based Progress Assessment", and "Recovery / Retry Re-entry". In the first region, show a robotic arm performing a tabletop manipulation rollout under normal policy execution. In the second region, show a conditional critic check that monitors progress and detects likely inconsistency, incompletion, or failure. Use a subtle check symbol for healthy progress and a small warning symbol for suspected failure. In the third region, show a recovery prompt and retry re-entry path that guides execution back to a safer and more controllable state before normal rollout resumes.

Visually make it clear that the healthy branch is simpler and more direct, while the recovery branch is only activated when needed. The critic must appear auxiliary and conditional, not the primary controller. Add a small unobtrusive logging or audit statistics panel to suggest traceability of checks, retries, and outcomes.

Use simplified robotic and tabletop elements so the figure feels embodied and intuitive, but avoid excessive arrows, too many stacked boxes, backend model names, or code-like logic. The final figure should look like a clean CCF-A paper mechanism illustration rather than a PowerPoint process chart.
```

# Agentic-VLA Figure Prompts

This file contains ready-to-use prompts for `nano banana` or similar image generators.
The goal is to produce clean, publication-ready figures for a CCF-A / top-tier robotics paper.

## Unified Style Prefix

Use this style prefix before every prompt:

`Create a publication-quality scientific figure for a top-tier robotics / computer vision paper. Style: clean BioRender-like vector diagram, white background, precise thin outlines, restrained academic color palette, modern sans-serif typography, balanced spacing, clear visual hierarchy, no watermark, no photorealistic rendering, no cartoon style, no glossy effects, no exaggerated shadows, no decorative background, no unnecessary text. The figure may use tasteful schematic visual elements such as a simplified robotic arm, tabletop objects, trajectory arrows, small icons, or subtle callout panels to make the technical idea more vivid, but it must still look like a rigorous CCF-A paper figure rather than a product infographic. The figure must remain readable when placed in a CCF-A conference PDF at one-column or full-width scale.`

## Style Anchor

Use `paper/Agentic-VLA/figures/fig.1-gemini.png` as the visual anchor for the remaining figures.

The next figures should follow the same overall design language:

- white background
- flat vector academic style
- light but not toy-like schematic objects
- rounded boxes with clean borders
- limited but meaningful iconography
- short labels instead of dense text paragraphs
- restrained, consistent colors across all figures

Do **not** copy the exact layout of `fig.1-gemini.png`, but do keep the same family resemblance so the paper looks visually coherent.

## Suggested Generation Workflow

For each new figure:

1. Upload or reference `fig.1-gemini.png` as the style example.
2. Ask for the new figure to match its academic vector style and color restraint.
3. Keep the new figure focused on one mechanism only.
4. If the first result is too decorative, ask for:
   - less text
   - fewer arrows
   - more academic layout
   - stronger visual hierarchy
5. If the first result is too abstract, ask for:
   - a simplified robotic arm
   - a tabletop scene
   - small object icons
   - cleaner callout labels

## Fig. 1: Overall Framework

### Goal

Show the full Agentic-VLA framework in one glance. The reader should immediately understand that:

- the frozen VLA policy remains the core action generator
- the three modules are external inference-time controllers
- evaluation happens through the official LIBERO simulator rollout loop

### Composition Requirements

- Put `Frozen VLA Policy` at the visual center
- Place `RGB observation`, `proprioception / robot state`, and `language instruction` on the left as inputs
- Place three external modules around the policy:
  - `Transition Agent`
  - `Scene Priors and Memory`
  - `Critic / Retry`
- Place `Unified Execution Loop` and `Official LIBERO Simulator` on the right
- Add a clear final arrow to `env.step -> done-based success`
- Make it visually obvious that the backbone is frozen and the other modules are wrappers
- It is good to use tasteful schematic robot and environment icons to make the framework less abstract

### Must Include

- frozen policy server
- three external modules
- official LIBERO simulator
- rollout loop
- auditability / real rollout implication

### Avoid

- too many crossing arrows
- excessive text inside boxes
- implementation clutter such as websocket, ports, internal variable names
- confusing the method figure with an experiment figure

### Final Prompt

`Create a publication-quality scientific framework overview figure for a robotics paper. Center the figure around a large block labeled "Frozen VLA Policy" to emphasize that the backbone remains unchanged. On the left, place three compact input blocks: RGB observation, proprioception / robot state, and language instruction. Around the central frozen policy, place three external inference-time modules with equal visual importance: Transition Agent, Scene Priors and Memory, and Critic / Retry. On the right, show a unified execution loop connected to the official LIBERO simulator, including observation update, environment step, and done-based success signal. Use arrows to indicate that the external modules modify prompts or intervene in execution, while the frozen policy remains the primary action generator. Add concise labels such as "inference-time augmentation", "frozen backbone", "auditable rollout", and "official LIBERO simulator". Do not mention specific external model names such as pi0.5 or pi05_libero inside the figure. It is encouraged to use tasteful schematic visual elements such as a simplified robotic arm, a tabletop scene, small observation icons, and subtle rollout arrows so the framework feels richer and more intuitive without looking decorative. Use a clean symmetric layout with minimal arrow crossings and very limited text per block.`

## Fig. 2: Transition Agent

### Goal

Explain the mechanism of the Transition Agent, not its experimental result.
The figure should show a generic failure mode at a subtask boundary and how a transition prompt inserts an intermediate reconnection motion.

### Composition Requirements

- Use two panels:
  - `State-Gap Failure Pattern`
  - `Transition Repair Mechanism`
- Show a simplified tabletop manipulation setting or schematic end-effector trajectory
- Mark `Subtask A stable region`, `subtask boundary`, and `Subtask B stable region`
- In the repair panel, show one inserted transition motion before normal execution resumes
- Use a simple but recognizable robotic manipulation scene instead of a purely abstract line plot when possible

### Must Include

- state gap / stall
- intermediate reconnection motion
- transition prompt
- no metric, no success rate, no result number

### Avoid

- words like `better`, `improved result`, `baseline wins/loses`
- anything that looks like a benchmark plot
- overcomplicated environment detail

### Final Prompt

`Create a publication-quality scientific mechanism figure for a robotics paper with two side-by-side panels, matching the same visual language as fig.1-gemini.png: white background, flat academic vector style, restrained colors, simple robotic icons, and clean rounded annotations. Left panel title: "State-Gap Failure Pattern". Right panel title: "Transition Repair Mechanism". Depict a simplified long-horizon manipulation scene with a robotic arm or end-effector moving across a tabletop task. Show that chunked execution approaches a subtask boundary, then loses state continuity and stalls before reaching the next stable region. Mark Subtask A stable region, subtask boundary, and Subtask B stable region. In the right panel, show that a transition prompt inserts one intermediate reconnection motion that restores executable state continuity, after which the normal trajectory continues toward the next stable region. Use clean schematic robotic elements, trajectory arrows, and light scene context so the idea feels vivid and intuitive. This is a method explanation figure, not a results figure.`

## Fig. 3: Scene Priors and Memory

### Goal

Explain how lightweight task entities are converted into structured priors and episodic cues, and then injected into prompts to bias the frozen policy toward more stable interaction.

### Composition Requirements

- Use a left-to-right pipeline
- Recommended blocks:
  - `Task Prompt + Scene Entities`
  - `Structured Priors`
  - `Episodic Memory`
  - `Prompt Augmentation`
- Use small icons or subtle symbols if possible
- Keep it lighter and simpler than the overall framework figure
- Use tasteful object icons or mini-scene symbols so the priors and memory cues look concrete rather than purely textual

### Must Include

- task entities
- priors such as approach direction / offset / force bias
- episodic cues from successful interaction
- prompt augmentation into frozen policy

### Avoid

- large database / retrieval-server imagery
- too much text
- presenting this module as a giant standalone model

### Final Prompt

`Create a clean scientific block diagram for a robotics paper, matching the same visual language as fig.1-gemini.png: white background, flat vector academic style, rounded blocks, short labels, and restrained colors. Show a lightweight scene priors and memory module. Arrange four blocks from left to right: "Task Prompt + Scene Entities", "Structured Priors", "Episodic Memory", and "Prompt Augmentation". In the Structured Priors block, visually suggest concepts such as approach direction, vertical offset, and force bias. In the Episodic Memory block, suggest small reusable successful interaction cues rather than a large database. Add a final note beneath the pipeline: "bias frozen policy toward more stable interaction patterns". Use small icons, mini object symbols, and subtle scene cues so the module feels concrete and intuitive while keeping the overall layout clean, lightweight, and publication-ready.`

## Fig. 4: Critic / Retry

### Goal

Show the conditional failure-checking loop of the Critic / Retry mechanism.
The reader should understand that the critic is not the main controller, but an auxiliary recovery module.
The figure should feel like a mechanism illustration, not a rigid engineering flowchart.

### Composition Requirements

- Use a mechanism schematic with light flow logic, not a dense process chart
- Organize the figure into three visually balanced regions:
  - `current rollout execution`
  - `critic-based progress assessment`
  - `recovery / retry re-entry`
- Show a normal execution path and a failure-recovery path, but keep arrows limited
- Make the healthy branch visually simpler than the failure branch
- You may use small warning, check, and recovery icons to make the logic easier to parse at a glance
- Include a small audit/statistics element, but do not let it dominate the figure

### Must Include

- conditional checking
- normal execution branch
- failure diagnosis branch
- recovery / retry re-entry
- logging or auditable statistics

### Avoid

- concrete backend names such as Qwen3-VL
- code-level details
- too many line crossings
- a PowerPoint-style or BPMN-style flowchart appearance
- too many stacked boxes with text-heavy arrows

### Final Prompt

`Create a publication-quality scientific mechanism figure for a robotics paper, matching the same visual language as fig.1-gemini.png and fig.2-gemini.png: white background, flat academic vector style, restrained colors, simplified robotic arm and tabletop scene, rounded panels, limited text, and clean visual hierarchy. The figure should explain the Critic / Retry mechanism as an intervention schematic rather than a rigid flowchart. Use three visually balanced regions from left to right: (1) current rollout execution with a robotic arm interacting with tabletop objects, (2) critic-based progress assessment that detects likely failure or inconsistency, (3) recovery / retry re-entry that guides execution back to a safer and more controllable state. Show that the critic is conditional and auxiliary, not the main controller. Use subtle visual cues such as a warning symbol, a recovery arrow, and a small audit/statistics panel. Avoid a dense flowchart appearance, too many rectangular boxes, too many arrows, or code-like logic. The figure should feel like a clean mechanism illustration for a CCF-A paper, not a PowerPoint-style process chart.`

## Fig. 5: LIBERO-10 Main Result

### Goal

Provide a polished experiment figure for the `libero_10` comparison.
The figure should clearly show:

- overall modest gain of Full over A1
- gains on T0, T2, T3, T5
- parity on several tasks
- degradation on T8 and slight drop on T9

### Composition Requirements

- grouped bar chart or polished panel chart
- x-axis: T0 to T9
- y-axis: success rate 0 to 100
- two methods:
  - `A1 baseline`
  - `Full Agentic-VLA`
- visually annotate `T8` as the dominant unresolved weak-task region
- If desired, add one understated callout arrow or a faint highlight box around T8 to guide the eye

### Must Include

- clear legend
- neutral gray for baseline
- refined blue for Full
- highlight of the remaining hard task

### Avoid

- rainbow colors
- thick grid lines
- casual infographic style
- too much explanatory text inside the figure

### Final Prompt

`Create a polished scientific results figure for a robotics paper that visually harmonizes with fig.1-gemini.png while remaining a quantitative chart. Use a grouped bar chart or an equally clean publication-style chart with x-axis labels T0 to T9 and y-axis as success rate percentage from 0 to 100. Plot "A1 baseline" in neutral gray and "Full Agentic-VLA" in refined blue. Visually make gains on T0, T2, T3, and T5 easy to perceive, show parity on T1, T4, T6, and T7, and show degradation on T8 and slight degradation on T9. Add a subtle but clear annotation around T8 labeled "dominant unresolved weak-task region". Keep the result figure minimalist, high-contrast, and suitable for a top-tier conference paper. Avoid flashy data-viz styling.`

## Optional Fig. 6: Qualitative Execution Traces

### Goal

If you later want a more eye-catching qualitative figure, use this to replace a rough screenshot collage.

### Composition Requirements

- three horizontal rows
- each row contains 4 to 5 small time-ordered frames
- suggested rows:
  - baseline success case
  - baseline stall at subtask boundary
  - Agentic-VLA transition-assisted recovery and completion

### Must Include

- consistent camera view
- concise overlay labels such as `approach`, `stall`, `transition inserted`, `recovered`, `done`
- thin frame borders and clean spacing

### Avoid

- messy screenshot montage
- inconsistent scales or viewpoints
- heavy arrows everywhere

### Final Prompt

`Create a publication-quality qualitative figure for a robotics paper showing execution traces across three task categories. Use three horizontal rows, each row containing four to five time-ordered frames from left to right. Row 1: baseline success case. Row 2: baseline stall at a subtask boundary. Row 3: Agentic-VLA recovery and successful completion after transition insertion. Keep the camera viewpoint consistent across frames, add thin borders around each frame, and overlay concise labels such as "approach", "stall", "transition inserted", "recovered", and "done". The final figure should look substantially more polished and publication-ready than a raw screenshot collage.`

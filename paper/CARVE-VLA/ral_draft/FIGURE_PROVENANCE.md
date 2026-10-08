# Figure provenance

| Paper figure | Source | What is measured or recorded |
| --- | --- | --- |
| Fig. 1, system mechanism | `figures/architecture_mechanism.tikz` | Editable, original schematic. The `ep39_demo.png` inset is a frame from the public RoboMME demonstration; all boxes, arrows, and the two-rate timeline are method descriptions, not measured outcomes. |
| Fig. 2, paired transitions | `figures/paired_outcomes.tikz` | Bar lengths encode recorded failures rescued and successes lost. Swap-memory confirmation uses 12 pairs, Unmask-memory transfer 16, and each frozen Raw/Harness task 16. |
| Fig. 3, qualitative episode | `figures/ep39_demo.png`, `figures/ep39_no_memory_final.png`, `figures/ep39_memory_final.png` | The first image is from the public demonstration; the other two are terminal frames of paired simulator rollouts. They are not synchronized counterfactual states. |
| Fig. 4, runtime profiles | `figures/runtime_profiles.tikz` | The four plotted BF16 P95 values and VRAM readings are independent PyTorch model-call measurements on 45 paired recorded observations. Late INT8 is explicitly off scale at 994.74 ms and rejected. This is not end-to-end JAX RoboMME timing. |

The earlier `architecture_story.tikz`, `architecture.tikz`, and
`admission.tikz` remain for comparison but are **not** active paper figures.
The old overview's `memory_history_illustration.png` and
`robot_scene_illustration.png` are AI-generated conceptual assets, not robot
rollout evidence; neither appears in the current PDF. No figure asset from
`paper/Agentic-RAG-VLM` or `paper/Agentic Policy/Harness VLA.pdf` was copied.

The optional `unmask_ep38_45_pilot_paired.pdf` and `.png` are generated from
the pilot's saved summary. They remain outside the six-page manuscript because
that pilot is development-stage rather than frozen confirmation.

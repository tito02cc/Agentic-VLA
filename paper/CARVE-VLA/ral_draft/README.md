# BIND-VLA RA-L-format research draft (2026-09-24)

This is an independent, anonymous first-submission draft. It does not modify
`paper/CARVE-VLA/root.tex` or the earlier submitted papers.
The paper method is now named **BIND-VLA**: semantic changes are bound to
completed action-chunk boundaries, and inference profiles to quality gates.
The repository and older result paths retain `CARVE-VLA` for provenance;
this is not a bulk software rename or a new VLA base model.
The current paper-writing source of truth is [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md):
it separates the implemented architecture, the three model/backend chains,
paired task evidence, independent inference measurements, failure cases, and
paper-ready claim boundaries. The longer
[`docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md`](../../../docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md)
is retained as a dated development record, not as a second competing draft.
Representative rollouts and the exhaustive on-disk video file list are in
[`VIDEO_EVIDENCE_INDEX.md`](VIDEO_EVIDENCE_INDEX.md) and
[`VIDEO_CATALOG.tsv`](VIDEO_CATALOG.tsv); videos are not duplicated here.

Build from this directory with `latexmk -pdf -interaction=nonstopmode main.tex`.
The `ieeeconf.cls` copy comes from the project's existing
`paper/Agentic-RAG-VLM/ieeeconf.cls`; `ieeetr.bst` is supplied by TeX Live.
The current `main.pdf` has six US-Letter pages, including references. The
active system overview, paired-outcome chart, and runtime-profile chart
remain editable TikZ sources in `figures/`; Fig. 3 uses three frames from
saved official-simulator videos. PDF compilation, embedded fonts, unresolved
references, and page count were checked on 2026-09-24.

The IEEE RA-L [author instructions](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/)
specify a US-Letter, two-column `ieeeconf` first submission and six pages
including figures and references, with at most two charged additional pages.
The initial submission uses a double-anonymous byline. Formatting checks
alone do **not** establish a publication-strength result.

Evidence and limits: see [`../RAL_EXPERIMENT_INVENTORY_20260923.md`](../RAL_EXPERIMENT_INVENTORY_20260923.md).
The historical RoboMME table is exploratory and mixes development-stage C3
variants. A fixed, within-task identity-memory comparison now has an
independent confirmation set, but its 5/12-to-9/12 change has exact two-sided
`p=0.21875` and one harm. The same-chain selective-scheduling ablation saves
calls on both-successful pairs but fails its task-quality gate. The PyTorch
latency ablation is not the JAX RoboMME runtime. These boundaries must survive
copyediting.

Figures are original editable TikZ sources in `figures/`. The architecture
separates the VLA action path from event-triggered semantic decisions; the
paired chart shows both rescues and harms for the latest fixed tests.
The active Fig. 1 is `figures/architecture_mechanism.tikz`; its only bitmap
is a recorded public-demonstration thumbnail. Older inactive overview files
retain conceptual image-generation assets but are not in the current PDF. See
`FIGURE_PROVENANCE.md` and `EXPERIMENT_UPGRADE.md` for the source ledger and
frozen follow-up protocol.

Fig. 3 provenance: `figures/ep39_demo.png` is the 1-second frame of
`artifacts/robomme/memory_ab_confirm_20260923/rollouts/VideoUnmaskSwap_ep39_B/initial_demo_front.mp4`.
`figures/ep39_no_memory_final.png` and `figures/ep39_memory_final.png` are the
last top-view frames (left 256x256 crop) of the corresponding A/B rollout
videos in the same result directory. The outcome labels and final planner
points come from the A/B `summary.json` files (`success` and last
`planner_trace` entry). Frames are not synchronized across trajectories;
the figure is explicitly a single qualitative pair.

Before submission: inspect the full PDF; confirm anonymous byline, title and
source files required by PaperCept; and audit numerical claims against
`../RAL_EXPERIMENT_INVENTORY_20260923.md` and its linked JSON. The current
evidence does **not** confirm a frozen integrated Harness+Optimize gain across
tasks. Keep the historical 80-pair development aggregate exploratory, and do
not call the 54.67-ms PyTorch model-call result an end-to-end JAX RoboMME
speedup. If the intended acceptance case depends on that joint claim, one
frozen same-checkpoint, same-backend paired study remains necessary. Do not
run arbitrary extra episodes merely to obtain a favorable point estimate.

The 2026-09-24 VideoUnmask stage-boundary update is documented in
`EXPERIMENT_UPGRADE.md`. The previously failed ep43 is a development rerun
that now succeeds; an unseen two-target ep47 has matched A/B/C successes
with 313/313/327 executed steps. These results establish a working
non-interruptive receipt path, not a population-level success or speed gain.
Raw summaries and audit traces are under
`artifacts/robomme/ral_gate_20260924/unmask_ep43_stage_hint_dev/` and
`artifacts/robomme/ral_gate_20260924/unmask_ep47_frozen_abc/`.

## RA-L comparison used for this revision

These are published RA-L articles, not results reproduced by BIND-VLA. Their
publisher-deposited DOI metadata and IEEE pages were checked on 2026-09-24.

| Paper | Closest overlap | What this draft must distinguish or learn |
| --- | --- | --- |
| [ReplanVLM (2024)](https://doi.org/10.1109/LRA.2024.3471457) | VLM error correction and replanning | Do not claim semantic replanning itself is novel; specify who can intervene and when. |
| [Reliable Robotic Task Execution in the Face of Anomalies (2026)](https://doi.org/10.1109/LRA.2025.3632090) | Detect, pause, recover around a learned policy | Contrast a trained anomaly detector and physical-robot validation with our untrained evidence gate and simulator-only evidence. |
| [Language-Driven Multi-Task Manipulation (2026)](https://doi.org/10.1109/LRA.2026.3674003) | Hierarchical planning and low-cost subtask switching | Its learned action mask is an alternative to our boundary receipt; a switching mechanism alone is not new. |
| [TinyVLA (2025)](https://doi.org/10.1109/LRA.2025.3544909) | Faster VLA inference | It changes the model architecture; we screen deployment profiles of an existing model. |
| [CALVIN (2022)](https://doi.org/10.1109/LRA.2022.3180108) | Long-horizon language-conditioned evaluation | Assess complete sequences, not isolated successful primitives; CALVIN itself is not evaluated here. |
| [PointVLA (2026)](https://doi.org/10.1109/LRA.2026.3653303) | Frozen action expert with lightweight adaptation | Its learned 3D injection changes policy representations; BIND-VLA's supervisor is external and cannot fix deficient perception. |
| [RILaaS (2020)](https://doi.org/10.1109/LRA.2020.2998414) | Robot inference-serving latency | Keep model-call P95, full episode cost, and memory use distinct. |

The literature comparison strengthens positioning but does not repair the
evidence gap below. In particular, the first three papers underscore that a
strong RA-L claim would need frozen, multi-task paired testing and ideally
physical-robot or cross-platform validation. No external paper's success
rate is used as a directly comparable baseline here.

## Decision after the 2026-09-24 evidence audit

The six-page manuscript is complete as a source-grounded draft for adviser
review and a graduate research report. It is **not yet evidence-complete for
the stronger RA-L claim that the integrated system improves both robot-task
success and end-to-end efficiency**. The fixed Raw/Harness comparison adds
one net rescue in 32 pairs; the new Unmask ep47 pair is all-successful but
slower with memory and receipts. The 54.67-ms result belongs to a separate
PyTorch model-call experiment.

The minimum next evidence is one predeclared, frozen paired comparison on
multi-stage tasks where the baseline policy has enough competence to make
recovery measurable, reporting successes, rescues, harms, and full episode
costs. The Optimize condition then needs the same policy checkpoint/backend
in closed loop, with task quality checked before a speed claim. If either
gate fails, keep the present paper as a bounded systems/evidence study and
revise the claim; do not assemble favorable episodes from development runs
into a purported confirmation set. No extra benchmark is needed merely to
finish the current draft.

# CARVE-VLA Paper

This directory is the current LaTeX manuscript package for:

> CARVE-VLA: A Compute-Aware Agentic Runtime for Reliable and Efficient VLA Execution

`CARVE` expands to **Compute-Aware Agentic Runtime for VLA Execution**. The
paper unifies execution supervision for long-horizon reliability with
runtime-level inference efficiency. It is currently maintained as an
unrestricted technical-report manuscript: method, implementation, negative
results, artifact paths, and claim boundaries are retained in full before a
future venue-specific compression.

The rejected/submitted Agentic-VLA v1 paper and its complete source package are preserved under `paper/archive/Agentic-VLA-v1-submission/`. The earlier integrated runtime Markdown draft is also retained in `paper/archive/`.

## Main Files

- `root.tex`: current manuscript source.
- `root.pdf`: compiled 28-page technical-report PDF.
- `RESUME_AND_CODEX_HANDOFF.md`: self-contained architecture, experiment,
  claim-boundary, resume, and cross-computer Codex handoff brief.
- `CARVE_VLA_RESUME_HANDOFF_20260811.zip`: portable nine-file handoff bundle
  containing the brief, paper, source, machine-readable results, and key status
  records.
- `references.bib`: bibliography.
- `waica.cls`, `waica.bst`: template files copied from the WAICA template.
- `figures/`: current manuscript figures and archived pre-freeze versions.
- `generated/`: result tables, macros, and a machine-readable summary generated
  from accepted experiment artifacts.
- `IMAGE2_FIGURE_BRIEF.md`: prompts and design brief for Image2-generated figures.
- `FIGURE_AUDIT_20260724.md`: figure-by-figure source and claim audit.

## Build

```bash
python ../../scripts/build_carve_paper_results.py
pdflatex -interaction=nonstopmode root.tex
bibtex root
pdflatex -interaction=nonstopmode root.tex
pdflatex -interaction=nonstopmode root.tex
```

Current compiled length: `28` pages.

## Reading Guide

1. **Sections 1--2:** problem motivation and relation to VLA, Agentic policy,
   failure detection, memory, and efficient inference.
2. **Section 3:** complete system description, including authority separation,
   Monitor equations, six-state Harness, VLM decision contract, Scene
   Graph/HAA-RAG, recovery skills, Optimize Runtime admission, SMVE, and the
   end-to-end procedure.
3. **Sections 4--5:** the 1,200-episode paired LIBERO/LIBERO-PRO study,
   historical mechanism diagnostics, canonical guarded-VLM handoff, PI0.5 and
   Planner precision ablations, physical recovery, low-memory coupling, and
   rejected asynchronous/backend-dependent profiles.
4. **Section 6:** supported claims, limitations, and unresolved research
   questions.
5. **Appendices:** runtime interface dictionary, latency definitions, key
   budgets, artifact map, and evidence-to-claim matrix.

## Current Figure Set

- `figures/fig1_framework_fused.png`: primary CARVE-VLA system architecture,
  preserving the original paper layout while incorporating the final Harness
  and Optimize Runtime contracts.
- `figures/fig2_execution_supervision_v2.png`: multi-rate supervision state machine.
- `figures/fig3_modules_v2.png`: Harness components and safety boundaries.
- `figures/fig4_realtime_runtime_v2.png`: evidence-gated Optimize Runtime.
- `figures/fig5_libero10_task_comparison_v2.png`: reproducible LIBERO-10 task-level result plot.
- `figures/fig6_libero_pro_t8_recovery.png`: three real MuJoCo frames from the
  native-HD LIBERO-PRO Object-OOD T8 recovery demonstration.

The older selective-refinement/contact/realtime figures and former
supplementary taxonomy are not referenced by the current manuscript. Their
pre-freeze versions remain under
`figures/archive_pre_framework_freeze_20260724/` for provenance.

`figures/fig1_framework_v2.png` remains as a compact alternative diagram; the
main manuscript uses the fused Fig. 1 because its left-to-right execution
narrative is clearer at paper scale.

## Quantitative Story

The main manuscript now foregrounds the final four-suite paired study and uses
the same PI0.5/LIBERO stack for deployment evidence:

- 1,200 real MuJoCo episodes across Frozen VLA, Fixed Recovery, and Full Agentic;
- paired flow-step and commitment calibration;
- compiled BF16, SMVE, and version-gated PI0.5 INT8 profiles;
- Qwen3.5-4B BF16/INT8/NF4 Planner precision and memory profiles;
- stateful physical recovery and closed-loop profile rejection;
- a real co-resident NF4-Planner/SMVE-VLA integration gate;
- a cadence-matched asynchronous-execution gate that rejects both tested
  prefetch schedules after a ten-episode closed-loop regression.

The promoted VLA profile is synchronous compiled BF16 with SMVE where its
padding-view contract holds; ordinary compiled BF16 is the fallback. Planner
BF16 is the latency default and NF4 is the memory tier. Current-backend INT8
and asynchronous prefetch remain documented negative candidates.

The earlier robosuite Stack pilot is retained in repository evidence but is no
longer a main-paper result.

The current technical report also includes:

- a two-state, three-noise-seed guarded-VLM recovery pair;
- a clean-T9 semantic-first qualification;
- a focused LIBERO-PRO capability screen;
- a three-trial native-HD Object-OOD T8 demonstration with one verified natural
  recovery.

These are mechanism/qualification results, not additional leaderboard claims.

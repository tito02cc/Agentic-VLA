# CARVE-VLA Paper

This directory is the current LaTeX manuscript package for:

> CARVE-VLA: A Compute-Adaptive Agentic Runtime for Reliable Long-Horizon VLA Execution

`CARVE` expands to **Compute-Adaptive Agentic Runtime for VLA Execution**. The paper unifies execution supervision for long-horizon reliability with runtime-level inference efficiency.

The rejected/submitted Agentic-VLA v1 paper and its complete source package are preserved under `paper/archive/Agentic-VLA-v1-submission/`. The earlier integrated runtime Markdown draft is also retained in `paper/archive/`.

## Main Files

- `root.tex`: current manuscript source.
- `root.pdf`: compiled 15-page PDF.
- `references.bib`: bibliography.
- `waica.cls`, `waica.bst`: template files copied from the WAICA template.
- `figures/`: current manuscript figures.
- `generated/`: result tables, macros, and a machine-readable summary generated
  from accepted experiment artifacts.
- `IMAGE2_FIGURE_BRIEF.md`: prompts and design brief for Image2-generated figures.

## Build

```bash
python ../../scripts/build_carve_paper_results.py
pdflatex -interaction=nonstopmode root.tex
bibtex root
pdflatex -interaction=nonstopmode root.tex
pdflatex -interaction=nonstopmode root.tex
```

Current compiled length: `15` pages.

## Current Figure Set

- `figures/fig1_framework.png`: overall CARVE-VLA framework.
- `figures/fig2_execution_supervision.png`: execution-supervision loop.
- `figures/fig3_modules.png`: Transition, memory/prior, and critic/retry module mechanisms.
- `figures/fig4_realtime_runtime.png`: realtime-aware runtime and call budget.
- `figures/fig5_libero10_task_comparison.png`: LIBERO-10 task-level result plot.
- `figures/figS_failure_taxonomy.png`: optional/supplementary failure taxonomy.

## Quantitative Story

The main manuscript now keeps the completed LIBERO-10 Agentic comparison and
uses the same pi0.5/LIBERO stack for the deployment evidence:

- paired flow-step and commitment calibration;
- compiled BF16 and replay-gated W8A16 deployment profiles;
- stateful physical recovery and closed-loop profile rejection;
- a cadence-matched asynchronous-execution gate that rejects both tested
  prefetch schedules after a ten-episode closed-loop regression.

The promoted deployment profile is synchronous compiled BF16. W8A16 and
asynchronous prefetch remain documented negative candidates rather than active
runtime recommendations.

The earlier robosuite Stack pilot is retained in repository evidence but is no
longer a main-paper result.

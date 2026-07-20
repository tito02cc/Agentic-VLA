# Project Organization

Last updated: 2026-07-16

## Current Main Thread

- `paper/archive/Agentic-VLA-runtime-draft-20260612.md`
  - Earlier integrated manuscript draft retained as an archived reference.

- `paper/CARVE-VLA/`
  - Current WAICA/LNCS-style LaTeX draft package.
  - Current story: CARVE-VLA combines agentic execution supervision with compute-adaptive realtime runtime mechanisms.
  - Compiles to a 15-page draft, matching the target limit.
  - Uses LIBERO-10 as the main benchmark evidence and robosuite/realtime as a separate deployment-oriented study.
  - The previous `paper/Agentic-VLA-Runtime/` IEEE draft has been integrated here and removed.

- `docs/status/EXPERIMENT_AND_PAPER_STATUS_20260612.md`
  - Current experiment and paper status.
  - Use this as the main status entry point after cleanup.

- `docs/plans/CARVE_VLA_NEXT_STAGE_PLAN.md`
  - Locked P0/P1/P2 experiment sequence for compute adaptation, long-horizon coupling, and optional quantization.

- `paper/archive/PAPER_STORY_AND_EXPERIMENT_PACKAGE_20260612.md`
  - Superseded June 2026 story package retained for provenance only.

- `results/paper_assets_20260609/results_section_draft.md`
  - Current Results section draft.

- `paper/CARVE-VLA/references.bib`
  - Current bibliography for the CARVE-VLA paper.

## Core Result Evidence

Current main LIBERO-10 evidence:

- `pi05_libero` baseline: `90.0%` (`180/200`)
- refined full CARVE-VLA: `92.5%` (`185/200`)
- dominant weak task: `55.0%` -> `75.0%`
- weak-task diagnosis: T6/T8/T9 targeted runs and module ablations

Submitted v1 paper assets preserved:

- `paper/archive/Agentic-VLA-v1-submission/agentic_vla_paper_v1.pdf`
- `paper/archive/Agentic-VLA-v1-submission/agentic_vla_paper_v1.tex`

Robosuite/pi0.5 matched Stack results retained as current deployment-oriented runtime evidence:

- `results/robosuite_stack_pi05_raw_fixed1_h430_5trials_20260611`
- `results/robosuite_stack_pi05_raw_fixed1_h430_extra5_seed20260616_20260612`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_5trials_20260611`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_extra5_seed20260616_20260612`
- `results/robosuite_stack_pi05_agentic_retry_fixed2_h430_5trials_20260611`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611`

Realtime sweep:

- `results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_fixed_noise.json`
- `results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_zero_noise.json`
- `results/paper_assets_20260609/table_pi05_realtime_sweep.md`
- `results/paper_assets_20260609/figures/pi05_realtime_sweep.png`

Compact policy supplement:

- `results/robosuite_stack_e2e_v2_20260611`

Additional qualitative robosuite video:

- `results/robosuite_real_video_stack_agentic_light_seed7_20260610`

## Paper Assets Kept

- `results/paper_assets_20260609/table_pi05_agentic_retry_matched.md`
- `results/paper_assets_20260609/table_pi05_realtime_sweep.md`
- `results/paper_assets_20260609/figures/fig1_system_overview_agentic_vla_runtime.png`
- `results/paper_assets_20260609/figures/fig2_agentic_recovery_flow.png`
- `results/paper_assets_20260609/figures/fig3_caq_lite_timeline.png`
- `results/paper_assets_20260609/figures/pi05_realtime_sweep.png`

Current WAICA draft figures:

- `paper/CARVE-VLA/figures/fig1_framework.png`
- `paper/CARVE-VLA/figures/fig2_execution_supervision.png`
- `paper/CARVE-VLA/figures/fig3_modules.png`
- `paper/CARVE-VLA/figures/fig4_realtime_runtime.png`
- `paper/CARVE-VLA/figures/fig5_libero10_task_comparison.png`

## Code Areas

- `docs/`
  - Canonical architecture, active plans, research updates, method notes, and
    experiment status. See `docs/README.md` for the document index.

- `scripts/`
  - Current robosuite/pi0.5 evaluation, sweep, plotting, data collection, and compact-policy training scripts.

- `openpi/`
  - OpenPI policy stack and local pi0.5 integration.

- `agentic_vla/`
  - Legacy prototype modules and simulation scenarios retained under their historical import name.

- `quantization/`
  - Lightweight inference and quantization scripts retained for the VLA deployment direction.

- `LIBERO/`
  - Local LIBERO benchmark checkout retained for possible future benchmark extension.

## Model and Checkpoint Areas

- `weights/openpi-assets`
  - OpenPI assets retained.

- `checkpoints/pi05_robosuite_stack_smoke`
  - Main retained robosuite/pi0.5 checkpoint area.

Removed during cleanup:

- Qwen3-VL weights and VLM demo/fine-tuning utilities.
- IsaacSim/IsaacLab failed-smoke artifacts and temporary scripts.
- Old LIBERO intermediate result directories.
- Old smoke, diagnostic, health, serving-profile, and one-off result directories.
- Superseded 20260610 paper plans and drafts.

## Previous Paper Projects and Source Papers

- `paper/Agentic-RAG-VLM/`
  - Previous Agentic-RAG-VLM paper project.
  - This directory is intentionally preserved and was not included in cleanup commands.

- `paper/Agentic Policy/`
  - Agentic Policy / VLA / WAM source papers and survey notes.

- `paper/archive/Agentic-VLA-v1-submission/`
  - Complete earlier Agentic-VLA submission package retained unchanged for reference.

## Cleanup Policy

- Do not delete `paper/Agentic-RAG-VLM/`.
- Keep the current paper draft, current status file, current bibliography, and core robosuite/pi0.5 evidence.
- Generated LaTeX build artifacts, old failed-smoke outputs, and unused external-model downloads can be removed.

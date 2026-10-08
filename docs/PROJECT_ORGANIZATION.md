# CARVE-VLA Project Organization

Navigation updated: 2026-09-22. Historical inventories below are retained.

## Canonical Entry Points

1. `README.md`: concise repository status and evidence boundary.
2. `docs/reports/AGENTIC_VLA_TECHNICAL_REPORT.md`: current technical report,
   batch evidence and limitations; Sections 10.7/10.8 contain recent RoboMME runs.
3. `docs/architecture/CARVE_COMPLETE_RESEARCH_LOOP.md`: Agentic RAG-VLM,
   Agentic Harness and Optimize Runtime research loop.
4. `docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md`: single active plan,
   narrowed to P1--P4. The legacy filename does not select RoboDojo as the main platform.
5. `docs/reports/CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md`: historical
   technical report; supplement it with the latest status above.
6. `results/paper_ready_20260827/README.md`: August machine-readable evidence,
   not a replacement for current batch manifests under `artifacts/robomme/`.
7. `paper/CARVE-VLA/`: current manuscript, bibliography and figures.

## Active Code

| Path | Purpose |
|---|---|
| `agentic_vla/runtime/` | Planner, Monitor, Harness, recovery, memory, safe hold and tracing |
| `agentic_vla/toolchain/` | typed tools, permissions, budgets, verification and workspaces |
| `agentic_vla/optimization/` | model/backend plugins, profiles, fidelity/admission and fallback |
| `agentic_vla/benchmarks/` | deployable observation adapters and evaluator isolation |
| `configs/` | local VLM and external Agent configurations |
| `scripts/` | stable launch, profiling, audit, summary and plotting entry points; not moved because imports and frozen receipts use these paths |
| `tests/` | CPU contracts and integration regression tests |
| `openpi/` | retained PI0.5 integration and Python environment |
| `LIBERO/` | retained simulator and benchmark checkout |
| `third_party/` | upstream dependencies/references, including RoboDojo and locally patched policy adapters |

### Current Work and Historical Entrypoints

Execute only P1--P4 in the single active plan, with its budget and gates.
`docs/archive/experiment_plans/AGENTIC_EXECUTION_HISTORY_THROUGH_20260922.md`
preserves the former 3558-line append-only plan (AW--AZ included). It is not a
second queue. Recent evidence: `artifacts/robomme/identity_memory_extension_20260922/`
and `artifacts/robomme/repick_noninterference_diagnostic_20260922/`.

The following September 10 inventory is historical, not permission to resume
its experiment scripts or superseded priorities:

- `artifacts/robodojo/recovery_lifecycle_20260910/`: completed 8-episode B0/C1/C2/C3
  official stack_bowls matrix: 2/2, 0/2, 1/2, 1/2; 24 raw videos, 545 CPU
  regressions and 408 evidence checks. Two recoveries received fresh post-action
  checks. Flow2 is NOT admitted for default deployment; causal Agent benefit
  remains unproven. Read README/admission.json before any further rollout.
- `artifacts/robodojo/reference_memory_20260910/`: latest offline visual-reference
  comparison, 18 real-model calls, source snapshots and 122 passing evidence checks;
  521 CPU regressions. Optional shadow integration exists, but semantic identity
  remains unadmitted and there are no new robot episodes in this artifact.
- `artifacts/robodojo/task_completion_mirror_20260910/`: latest completed two-episode
  startup/residency/reset integration, six raw videos, frozen sources and audit;
  subsequent Critic uncertainty fix has 503 passing CPU regressions. Shadow results
  are not Agentic benefit; object identity and active recovery remain unadmitted.
- `scripts/run_robodojo_starvla_nominal.sh`: existing short closed-loop launcher;
  its B0/C1/C2/C3 labels do not automatically implement every proposed
  B0/O/A/AM/AMO condition in the new plan.
- `scripts/run_robodojo_starvla_fault_ablation.sh`: registered fault diagnostics,
  not evidence of natural-task success improvements.
- `artifacts/robodojo/residency_mirror_20260910/`: latest isolated real-GPU
  residency comparison. Six measured pairs: 16.531 to 7.341 s mean full call,
  identical VLM text and fixed-input VLA actions. Cold start still 17.533 s;
  no new robot trials or active-recovery admission. Optional generic eager
  helper: `agentic_vla/optimization/residency.py`.
- `artifacts/robodojo/reobservation_shadow_20260910/`: preceding real RoboDojo
  build_tower shadow integration, one successful episode and three original
  videos. Two staged VLM calls exceeded 12 s; no memory commit or intervention.
  Sources, input-isolation hashes, residency timing and explicit admission gaps
  are preserved. Optional hook: `agentic_vla/benchmarks/robodojo_reobservation.py`.
- `artifacts/robodojo/reobservation_gate_20260910/`: preceding bounded live-VLM /
  recorded-observation memory integration, five real calls. Software checks pass
  but task-object identity remains incorrect. No robot control. Core optional
  scheduler/native observer: `agentic_vla/toolchain/reobservation.py`.
- `artifacts/robodojo/tracking_exclusivity_gate_20260910/`: preceding paired output-only
  SAM2 check; 10/12 to 12/12 scored boxes, but raw ambiguity/loss remains in
  190/386 frames. Not identity verification, VLM reacquisition or online control.
  Uses the diagnostic in `agentic_vla/toolchain/track_quality.py`.
- `artifacts/robodojo/visual_tracking_gate_20260910/`: preceding frozen VLM-seeded
  forward visual-memory check, 386 recorded frames. Paired scored boxes improve
  from 8/12 to 10/12 but remain below admission. Not online robot execution.
- `artifacts/robodojo/native_grounding_gate_20260910/`: completed 27-call interface
  and existing-model comparison; none admitted. Raw outputs/costs remain separate
  from the subsequent tracking subset. See each directory's README and audit.
- `artifacts/robodojo/object_grounding_gate_20260910/`: preserved frozen object
  localization and read-only evidence-memory development check. Nine real model
  calls completed; localization admission failed. See `README.md`, `summary.json`
  and `audit.json`; software tests do not imply robot/semantic success.
- `artifacts/robodojo/multiview_progress_gate_20260910/run.py`: preserved frozen
  head-only versus three-camera visual-development check; its `audit.py`
  validates sources, paired camera delivery and summary counts. Do not
  overwrite its protocol/outputs or reuse its known images as a holdout.
- `artifacts/robodojo/conjunctive_progress_gate_20260910/`: preserved previous
  whole-stage versus host-conjunction negative result, not a replacement baseline
  for a new prompt/camera comparison.
- `scripts/run_ral_robodojo_matrix.sh`: previous matrix driver, retained for
  reproducibility. Do not restart it as the new formal study: current semantic
  admission, tool/memory integration and explicit held-out layout selection
  must be completed first.

## Canonical Evidence

Current RoboDojo raw videos, model records, protocols and source snapshots live
under `artifacts/robodojo/`; CPU regression receipts live under
`results/agent_application_20260909/` and subsequent dated roots. These are not
disposable merely because some candidates failed. Use the latest status to find
the relevant run instead of browsing all directories.

`results/` also contains the retained historical evidence below:

| Result root | Role |
|---|---|
| `results/libero_pro_full_study_20260825/` | 1,200-episode formal benchmark and all raw traces/videos |
| `results/carve_efficiency_full_20260826/` | VLA and Planner efficient-inference ablation |
| `results/carve_optimize/` | version-sensitive and quantized closed-loop evidence |
| `results/full_embodied_agent_20260827/` | complete VLM-Harness-Memory-Recovery-PI0.5 loop |
| `results/planner_profile_closed_loop_20260827/` | T8 verified-memory and 4B routing |
| `results/cross_task_memory_routing_20260827/` | T3 factorized memory/model routing |
| `results/libero_pro_embodied_tool_probe_20260827/` | RGB-D physical skill qualification |
| `results/vlm_embodied_tool_use_20260827/` | VLM grounding, schema gate and physical dispatch |
| `results/paper_ready_20260827/` | canonical CSV/JSON tables and hashes |
| `results/libero_pro_representative_videos_20260828/` | curated paired presentation videos |
| `results/robomme_admission_20260828/` | official RoboMME environment and CARVE adapter gate; no policy claim |
| `results/robomme_planner_admission_20260828/` | local VLM Planner schema gate on a real RoboMME frame |
| `results/robomme_policy_gate_b_20260828/` | released GroundSG PI0.5 four-task policy admission; no success claim |
| `results/robomme_carve_text_subgoal_pilot_20260828/` | paired static-text Planner interface baseline (`1/5`) |
| `results/robomme_oracle_upper_bound_20260828/` | privileged paired upper bound (`5/5`), not deployable |
| `results/robomme_vlm_groundsg_bf16_queue_isolated_20260828/` | canonical deployable GroundSG VLM result (`4/5`) and paired summaries |
| `results/robomme_vlm_groundsg_nf4_20260828/` | NF4 capacity-tier closed-loop gate (`2/5`) |
| `results/_runtime/` | admitted runtime manifests and receipts |

New result directories need a named protocol and an entry in the current plan.
Do not rename existing raw evidence: manifests, scripts and papers use its paths.

## Documentation

- `docs/architecture/`: current system definitions only.
- `docs/plans/`: current forward plans and frozen completed-study protocol.
- `docs/research/`: benchmark and efficient-inference surveys.
- `docs/status/`: current evidence, completion log and cleanup record.
- `docs/reports/`: explanatory technical reports and thesis guides.
- `docs/archive/`: historical text retained for provenance; referenced raw
  intermediate results may no longer exist after cleanup.

## Local Assets

| Path | Policy |
|---|---|
| `weights/` | normalization/configuration assets only; no large parameters |
| `openpi/.venv/` | active PI0.5 environment; retain |
| `LIBERO/` | active simulation dependency; retain |
| `exports/CARVE_VLA_WEEKEND_PACKAGE_20260828.zip` | transferable report/evidence/video package |
| `exports/LEGACY_SCRIPTS_20260828.zip` | compressed archive of 80 removed launchers |
| `midterm/` | user-owned assessment materials (formerly `中期/`); never routine-clean |
| `Agentic-RAG-VLM/` | previous project code and assets; preserve |
| `exports/guanghua_hand_env_bundle_20260827_134025.zip` | preserved bundle moved from the project root on September 10 |
| `artifacts/housekeeping/` | retained historical root-level simulator logs |

The active PI0.5 LIBERO checkpoint is stored in the user OpenPI cache outside
this repository. Old OpenVLA, Robosuite, ToolHang and `pi0_base` parameters were
removed because they were not part of the canonical evidence and are
re-downloadable.

## Preservation Rules

1. Never delete or rewrite `paper/Agentic-RAG-VLM/`, `paper/Agentic Policy/`,
   `paper/CARVE-VLA/`, `paper/archive/`, `Agentic-RAG-VLM/` or `midterm/` during routine cleanup.
2. Keep every path referenced by `results/paper_ready_20260827/summary.json` and
   `representative_artifacts.json`.
3. New benchmark runs must write to one named result root and end with an
   aggregate JSON, audit report and claim boundary.
4. Only known generated Python/test caches in owned code roots are routinely
   disposable. Paper/LaTeX files, model environments, source snapshots and raw
   experiment records are excluded from this cleanup.
5. Git-history rewriting and weight/environment removal are not routine cleanup.

## September 10 Organization

Five superseded plans were moved byte-for-byte from `docs/plans/` to
`docs/archive/experiment_plans/`: the August 28 benchmark/model, modern
RoboTwin, initial RoboDojo and RoboMME plans, and `CARVE_VLA_NEXT_STAGE_PLAN.md`.
Their frozen historical text is retained; old relative paths inside archived
documents may describe their original location. Active navigation now points
to the sole RoboDojo execution plan. Frozen preregistrations and old result
directories were not removed. Detailed move hashes are in
`status/REPOSITORY_CLEANUP_20260828.md` (September 10 entry).

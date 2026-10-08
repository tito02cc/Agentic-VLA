# CARVE-VLA Documentation

This directory is the entry point for active project documentation. The paper
projects remain under `paper/` and are intentionally managed separately.

## Active Execution Route (2026-09-11)

- [Kiro continuation prompt](handoff/KIRO_CONTINUATION_PROMPT.md): self-contained
  handoff with factual boundaries and room for independent engineering choices.
- [Technical introduction slides](../deliverables/AGENTIC_VLA_TECHNICAL_INTRO/README.md):
  editable PPTX, per-slide explanations and original success/failure videos.

- [Complete technical master report](reports/AGENTIC_VLA_TECHNICAL_REPORT.md):
  the current paper-style Chinese source for architecture, implementation,
  experiments, efficiency and quantization, limitations, code, evidence and
  videos. Use this stable file for thesis, resume and presentation preparation.
  Historical results remain tied to their original configurations and dates.

- `reports/FRAMEWORK_READINESS_AND_INTERVIEW_GUIDE.md`: current framework review,
  Kiro continuation assessment, implemented versus exercised capabilities,
  verified efficiency evidence, interview narrative, and remaining test gaps.
  Framework readiness and interview preparation take priority over new matrices.

- `architecture/AGENT_APPLICATION_LAYER.md`: optional LangGraph/FastAPI tool
  service shared by research and job preparation; deployment boundary, durable
  receipt semantics, and non-interference gates. Not a robot-success result.
- `plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md`: current RoboDojo-only
  development route for executable Agent tools, memory, and efficient inference;
  supersedes the benchmark-expansion sequence below.
- `status/ROBODOJO_CARVE_PAIRED_RESULTS.md`: RoboDojo evidence and limitations,
  including the reset-contaminated diagnostic block and current recovery checks.
- Older dated reports below remain historical evidence, not current launch
  instructions. No previous result or paper has been removed by this update.

## Historical Technical Report

- `reports/CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md`: August system,
  implementation, audited LIBERO/LIBERO-PRO results, efficient-inference
  evidence, future RoboMME plan, group-meeting narrative, resume bullets,
  interview Q&A, and reading guide. Its future plans are historical; use the
  active execution route above for September work.

## Architecture

- `architecture/CARVE_COMPLETE_RESEARCH_LOOP.md`: canonical thesis-level link
  between Agentic RAG-VLM, the CARVE Agentic Harness, Harness VLA/RPent as an
  external ecosystem reference, and CARVE Optimize Runtime.
- `architecture/CARVE_RUNTIME_ARCHITECTURE.md`: canonical system architecture
  and the separation between CARVE Agentic Harness and CARVE Optimize Runtime.
- `architecture/CARVE_HIGH_LEVEL_AGENT.md`: multi-rate VLM/VLA responsibilities,
  typed planner decisions, safe hold, recovery, and memory contracts.
- `architecture/CARVE_OPTIMIZE_RUNTIME.md`: current optimization package,
  plugin contracts, and pi0.5 replay benchmark entry point.

## Plans

- [RoboDojo execution plan](plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md):
  the only active implementation/experiment sequence.
- `plans/LIBERO_PRO_FULL_STUDY_PLAN_20260825.md`: frozen protocol for the
  completed 1,200-episode paired study.
- Other dated RoboDojo preregistrations remain in `plans/` for provenance,
  not as instructions to restart superseded matrices.
- Five superseded model/benchmark plans moved intact to
  `archive/experiment_plans/`. The RoboMME protocol there is historical;
  existing results remain in place. See [directory map](PROJECT_ORGANIZATION.md).

## Research

- `research/BENCHMARK_GITHUB_ADOPTION_SURVEY_20260828.md`: time-stamped GitHub
  stars, forks, activity and adoption-aware benchmark decision.
- `research/CURRENT_VLA_BENCHMARK_AND_MODEL_SURVEY_20260828.md`: current 2026
  benchmark/model survey, checkpoint availability, RTX 4090 cost and selected
  CARVE validation matrix.
- `research/CARVE_EFFICIENT_INFERENCE_UPDATE_20260824.md`: current layered
  efficient-inference design, latest-work boundary, and implementation order.
- `research/AGENTIC_BENCHMARK_TASK_SURVEY_20260730.md`: task-first comparison of
  LIBERO-PRO/Plus, RoboDojo, RMBench, and RoboTwin, with the selected validation
  route for Agentic RAG-VLM mechanisms.

## Methods

- `methods/CAQ_LITE_METHOD_AND_RESULTS_20260609.md`: retained CAQ-Lite method
  and result record.

## Status

- `status/ROBODOJO_CARVE_PAIRED_RESULTS.md`: latest RoboDojo implementation
  and experiment status; read this before starting or claiming new results.
- `reports/CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md`: historical
  August framework, experiment, thesis and resume summary.
- `../results/paper_ready_20260827/README.md`: generated canonical tables,
  source hashes, representative videos and claim boundaries.
- `../results/libero_pro_representative_videos_20260828/README.md`: curated
  paired LIBERO/LIBERO-PRO videos for recovery, Agentic gain, regression
  correction and the false-intervention boundary.
- `status/FULL_EMBODIED_AGENT_RESULTS_20260827.md`: complete VLM Planner,
  Harness, trusted memory, Critic, recovery, PI0.5 and runtime gate.
- `status/CARVE_CROSS_TASK_MEMORY_ROUTING_RESULTS_20260827.md`: factorized
  verified-memory and 9B/4B model-routing result on a second task family.
- `status/CARVE_MEMORY_ROUTED_PLANNER_PROFILE_20260827.md`: repeated-task
  memory route and high-level latency reduction on Object T8.
- `status/LIBERO_PRO_FULL_STUDY_RESULTS_20260826.md`: 1,200-episode paired
  benchmark result and statistical boundary.
- `status/CARVE_EFFICIENT_INFERENCE_ABLATION_20260826.md`: final VLA and
  Planner efficient-inference ablation.
- `status/CARVE_FRAMEWORK_FREEZE_20260824.md`: canonical framework freeze and
  current implementation boundary.
- `status/LIBERO_PRO_RECOVERY_BOUNDARY_GATE_20260825.md`: retained paired
  recovery-boundary evidence used by the full study.
- `status/EXPERIMENT_LOG.md`: chronological experiment log.
- `status/REPOSITORY_CLEANUP_20260828.md`: retained/deleted artifact policy and
  post-cleanup verification.
- `../results/robomme_admission_20260828/README.md`: official RoboMME simulator
  and CARVE adapter admission result; diagnostic only, with no policy claim.
- `../results/robomme_planner_admission_20260828/README.md`: local Qwen3.5-4B
  Planner schema admission on a real RoboMME frame.
- `../results/robomme_policy_gate_b_20260828/README.md`: released GroundSG
  PI0.5 four-task policy admission; diagnostic only, with no success claim.

## Archive

- `archive/README.md`: index of superseded experiment plans, dated reports,
  research snapshots, and historical status records. Archived documents remain
  available for provenance but are not active instructions.

## Organization

- `PROJECT_ORGANIZATION.md`: repository map, evidence locations, and cleanup
  policy.

## Preservation Rule

Do not delete or rewrite files under `paper/Agentic-RAG-VLM/`,
`paper/Agentic Policy/`, or `paper/archive/` as part of routine cleanup. The
current manuscript lives under `paper/CARVE-VLA/`.

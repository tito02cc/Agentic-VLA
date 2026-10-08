# CARVE Current Benchmark and Model Plan

Date: 2026-08-28

## Decision

The current evaluation route uses separate benchmarks for separate claims. It
supersedes the earlier RoboDojo-only priority, while retaining all completed
RoboDojo source checks and resumable model artifacts.

| Priority | Claim | Benchmark | Policy | Training | State |
|---|---|---|---|---|---|
| Completed | Low-cost robustness control | LIBERO-PRO subset | PI0.5 | none | 1,200 episodes audited |
| P0 | Memory and closed-loop Agentic control | RoboMME | released PI0.5 baseline | none | next |
| P1 | Cross-policy efficient runtime | RoboTwin 2.0 subset | DM0.5 and TurboVLA | none | environment gate |
| P2 | Broad Agentic external validation | RoboDojo | Xiaomi-Robotics-1 | none | storage/cloud gate |
| Optional | Planner and dynamic-runtime diagnostics | WatchAct / ReflexBench | VLM Planner / admitted policy | none or adaptation | not main evidence |

## Completed: LIBERO-PRO

The thesis-scale paired control study is complete and must not be rerun by
default:

- Standard, Object, Position Swap and Task Logic suites;
- all ten LIBERO-10 tasks in each suite;
- ten paired official initial states per task;
- Frozen VLA, Fixed Recovery and Full Agentic;
- 120 cells, 1,200 physics-simulator episodes and 1,200 decoded MP4 videos.

This is not the official 50-state leaderboard protocol and does not cover every
published perturbation family. It is sufficient for the selected role of a
low-cost robustness control. The next GPU budget goes to RoboMME and RoboTwin,
not to expanding LIBERO-PRO.

Evidence: `docs/status/LIBERO_PRO_FULL_STUDY_RESULTS_20260826.md` and
`results/libero_pro_full_study_20260825/aggregate/audit_report.json`.

## Optional: WatchAct

Run the official VLM planning pipeline on all 14 task types with a frozen prompt
and output schema. Compare:

1. Stateless Planner.
2. Planner with CARVE task state and episodic memory.
3. Full Planner with typed tools, Critic and bounded replan.

Report plan validity, symbolic goal completion, event/procedure/intent/episodic
subscores, model calls, tokens, P50/P95 latency and cost. Promote a small subset
to the official LIBERO action-execution stage only after the planning result is
positive.

WatchAct is a Planner diagnostic, not one of the four primary closed-loop
benchmarks. Run it only if a clean high-level planning ablation is needed.

## P0: RoboMME

Use the released fine-tuned PI0.5 baseline and official simulator. Compare on
the four memory suites:

1. Direct PI0.5.
2. PI0.5 plus local Monitor and bounded recovery.
3. Full CARVE Planner, tools, trusted memory and Critic.
4. Full CARVE plus Optimize Runtime.

Use exact paired seeds and keep simulator task state evaluator-only. Primary
metrics are success, memory-query correctness, recovered failures, false
interventions, VLA/Planner calls per success, wall time and deadline misses.

## P1: RoboTwin 2.0

Use released benchmark-specific checkpoints:

- DM0.5 as the high-performance current VLA.
- TurboVLA as the 0.2B real-time policy.

Select 6--10 tasks spanning long horizon, precise insertion, repeated object
handling and randomized conditions. The purpose is not a full leaderboard
reproduction. It is a controlled adapter/runtime comparison with the same
observation contract, request receipts and metrics.

Measure model load time, peak VRAM, cold/warm latency, P50/P95/P99, throughput,
action age, deadline misses, simulator success and successful episodes per hour.

### Optimize Runtime matrix

The RoboTwin study must keep the efficient-inference claim separate from model
quality. For each admitted policy, run the smallest supported matrix:

1. official/native inference profile;
2. CARVE backend profile with fixed precision, compile/kernel and denoising
   settings;
3. CARVE scheduling profile with action queue, chunk/commit horizon, deadline,
   event-coherent refresh and fallback;
4. admitted low-bit profile only when replay fidelity and closed-loop
   non-inferiority pass;
5. Full Agentic plus the best admitted Optimize profile.

Report both policy-only and end-to-end metrics. A latency improvement is not a
valid result if it changes the checkpoint contract, action statistics or paired
closed-loop outcome without disclosure.

The cross-model comparison has two anchors:

- DM0.5 tests whether CARVE adds deployment value to a current high-performance
  model that already has an optimized inference implementation;
- TurboVLA tests whether scheduling, event-triggered Agent calls and shared-GPU
  orchestration still matter when the low-level VLA is already only 0.2B.

This prevents CARVE from claiming model-level acceleration that belongs to an
upstream VLA while still measuring the system-level cost of the complete Agent.

## P2: RoboDojo

Resume only when one of these gates passes:

- official cloud evaluation access is available; or
- at least 120 GiB contiguous working storage is available for Isaac assets,
  checkpoint, container and cache.

The scoped Xiaomi-Robotics-1 checkpoint remains valid and its interrupted
download is resumable. Do not download the approximately 90 GiB simulator asset
set before this gate.

## Optional: ReflexBench

The official Isaac Lab benchmark code and roughly 200 LeRobot demonstrations per
task are now public. First run environment registration and zero-agent smoke.
The repository does not currently provide a directly usable VLA checkpoint, so
do not start a full policy study before the adaptation cost is admitted. Until
then, run a CARVE-owned latency injection study and label it as such:

- synchronous versus asynchronous execution;
- injected delay grid;
- action chunk and execution horizon sweep;
- action age and success under moving targets.

Do not report this protocol as a ReflexBench reproduction.

## Stop Rules

- Do not fine-tune a new policy before a released checkpoint has passed model
  load, fixed-observation inference and one closed-loop episode.
- Do not run complete 50/90/365-task suites before a mechanism-relevant subset
  shows a positive and reproducible effect.
- Do not use standard LIBERO aggregate success as the primary new evidence.
- Do not add a WAM experiment until the P0 VLA route has produced its result
  package; Faster-WAM is the first optional WAM candidate.

## Required Result Package

Every admitted run must produce:

- frozen config and source revision;
- paired episode-level JSON/JSONL;
- model and simulator logs;
- representative success, recovery and failure videos;
- latency, VRAM and call-count receipts;
- native-versus-optimized profile manifests and replay-fidelity results;
- VLA policy latency separated from Planner/Critic and simulator wall time;
- action age, queue depth, forced-fresh events, fallback and deadline misses;
- source-data hashes and a generated summary table.

The detailed evidence and selection rationale are in
`docs/research/CURRENT_VLA_BENCHMARK_AND_MODEL_SURVEY_20260828.md`.

## Preparation State at Plan Freeze

- The Xiaomi-Robotics-1 RoboDojo checkpoint download was paused at about 9%.
  The sparse target file and `.aria2` control file are retained under
  `/var/tmp/carve-vla-rq5/models/robodojo-xiaomi-r1` for exact resume.
- The prior combined RoboTwin/RPent installer was stopped before replacing the
  simulator environment's PyTorch. The environment still reports Python 3.10.20,
  PyTorch 2.4.1+cu121 and CUDA 12.1.
- Future setup must keep `robotwin-sim`, each VLA policy server and the CARVE
  Agent runtime in separate environments connected through the typed wire
  protocol.

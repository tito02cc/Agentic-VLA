# CARVE-VLA × Harness VLA — RA-L experiment roadmap

Frozen planning baseline: 2026-09-07 (Asia/Shanghai).
Goal: turn the advisor's requirements—agentic VLA, efficient inference,
long-horizon memory, multi-agent learning, and Guanghua-1 validation—into a
credible staged program without invalidating the current RoboDojo study.

## 1. Strategic assessment

The advisor's direction is technically aligned with the current field. Harness
VLA explicitly treats a frozen VLA as a retryable contact-rich primitive,
composes it with a fixed analytic primitive library, and uses task-specific
successful traces plus global success/failure memory for long-horizon execution.
It separates reference-seed bootstrapping from held-out deployment and reports
few-shot and zero-shot analyses across LIBERO(-Pro), RoboCasa365, and RoboTwin
C2R. See the [Harness VLA paper](https://arxiv.org/html/2607.08448), its
[project website](https://harnessvla.github.io/), and
[released code](https://github.com/RLinf/RPent).

This validates the general direction, but narrows CARVE's defensible novelty.
“Wrapping a frozen VLA with an agent, tools, and memory” cannot be the headline
contribution by itself. CARVE should be positioned as a **compute-aware,
verified deployment harness**:

1. deployment-observable failure monitoring without reward/private state;
2. bounded, capability-gated recovery and safe fallback;
3. high/low-frequency separation with action reuse and explicit deadline
   accounting;
4. verified memory writes rather than unrestricted self-reflection;
5. typed planner/executor/verifier authority and auditable lifecycle receipts;
6. a simulator-to-real contract that preserves action, reset, timing, and safety
   semantics.

## 2. What CARVE has today

| Advisor requirement | Current evidence | Honest status |
|---|---|---|
| Agent/harness | Planner, VLA executor, verifier/critic, monitor, typed tools, bounded recovery | Implemented; RoboDojo semantic layer currently disabled by admission policy |
| Efficient inference | C1/C3 use 2 vs 4 inference steps; per-call first/steady latency and deadline metrics | Implemented and measured; outcome effect still needs fresh confirmation |
| Long-sequence execution | Task ledger, recovery loop, procedural plan execution, working state | Implemented, but RoboDojo matrix tests runtime/recovery rather than memory |
| Procedural memory | Verified symbolic procedures with atomic JSON persistence and warm start | Real LIBERO-Pro route exists; current evidence is only three held-out states |
| Semantic memory | HAA/affordance cards and scene graph injected into planner context | Implemented; largely static, not a learned durable store |
| Failure/episodic memory | Verified failure records with context/failure/profile matching and TTL | Online in-process only; no cross-run persistence in the canonical path |
| Multi-agent | Role-separated planner/executor/verifier/monitor plus guarded external-agent gateway | Orchestration, not multiple independently trained agents |
| Autonomous VLA fine-tuning | None in CARVE-owned runtime | Not implemented; current model is frozen |
| Real robot / Guanghua-1 | Generic adapter and action contracts only | No robot-specific SDK adapter, calibration, safety integration, or result |

Existing cross-task procedural-memory evidence is useful but narrow: on three
held-out LIBERO-Pro states, procedure reuse preserved 3/3 success while reducing
planner calls and wall time. Because warm start also skipped startup planning,
it is not yet a clean memory-only causal ablation.

## 3. Submission scope: one main claim, three supporting pillars

### Main claim

A frozen VLA can be made more dependable on naturally occurring long-horizon
failures by a **verified, compute-aware execution harness** that detects failures
from deployment signals, applies bounded recovery, and preserves inference
latency—without reading evaluator truth or changing VLA weights.

### Pillar A — RoboDojo natural long-horizon outcome and mechanism

Complete the reset correction and the fresh set-1/set-2 B0/C1/C2/C3 matrix:

- 25 official episodes per condition per set; 200 total;
- block-level and pooled n=50 paired analysis;
- success, conversion/regression, exact McNemar, latency, deadline, VLA calls,
  Monitor events, fresh-budget interventions, and reset receipts;
- C2/C3 remain explicitly “runtime + bounded recovery,” not full online semantic
  planning.

**Go/no-go:** if fresh C3 is null/negative and mechanism traces show no useful
recoveries, do not market success-rate improvement. Retain the engineering and
efficiency findings and redesign the planner/re-staging mechanism before another
large efficacy run.

### Pillar B — clean memory mechanism ablation

Run a small, separately frozen LIBERO-Pro procedural-context sidecar, because
that path already has verified memory and does not modify RoboDojo:

- **M0:** same agent/planner schedule, no procedure memory;
- **M1:** same schedule and exactly one startup planner call, with the frozen
  verified procedure supplied as retrieved context;
- held-out states 1/7/9, paired; no memory writes during evaluation;
- freeze procedure JSON and SHA256 before either condition;
- report retrieval hit, planner prompt/transcript, plan acceptance/coverage,
  planner calls/latency, control steps, success, and deadline misses.

Minimum closed-loop matrix: 2 conditions × 3 states = 6 episodes. This is a
mechanism study, not a powered success-rate claim. If resources permit, expand
only under a new pre-registration to more tasks/states; never top up after
viewing outcomes.

A later memory-v2 study may add persistent failure memory, but only after schema,
version, TTL, contamination controls, and trusted write gates are tested.

### Pillar C — Guanghua-1 minimum real-robot transfer

Do not attempt an open-ended autonomous demo first. Admit one safe primitive and
one short composition in stages:

1. **Interface admission:** robot observation adapter, camera/calibration ids,
   action representation/units/frame, gripper convention, control rate,
   reset/home, heartbeat, stale-action rejection, protective stop, and human
   takeover.
2. **Shadow mode:** VLA/harness predicts and logs actions while the robot does not
   execute them; check limits, timestamps, inference latency, and action
   continuity.
3. **Single primitive:** bounded pick or place with fixed workspace and soft
   objects; compare B0 and runtime C1.
4. **Short composition:** one contact-rich VLA primitive plus one admitted
   analytic transport/release primitive; compare B0 and corrected C3.

Recommended minimum evidence after platform safety admission: 2 tasks × 10
trials × 2 relevant conditions = 40 physical trials, with all failures retained.
Report success, intervention count, protective stops, human takeovers, p50/p95
latency, deadline misses, and videos. This is feasibility evidence, not a broad
real-world generalization claim.

Execution is blocked until the platform provides the official SDK/ROS2
interface, allowed control modes/frequency, sensor timestamp/calibration format,
and safety PLC/estop contract. No simulator-specific object pose, reward, or
success field may enter the real-robot policy API.

## 4. Efficient inference story

The paper should explain efficiency as an explicit scheduling problem rather
than “using fewer steps”:

- slow path: planner/critic only at safe semantic boundaries;
- fast path: frozen VLA action chunks and admitted reuse;
- deterministic Monitor decides when reuse remains safe;
- first-call warm-up is separated from steady-state latency;
- compute budget, action horizon, deadline slack, stale result, fallback, and
  fidelity are all recorded;
- C1 isolates runtime efficiency; C2 isolates Agentic recovery; C3 tests their
  composition.

Required plots: success–latency Pareto, per-episode VLA-call distribution,
first-vs-steady latency, deadline CDF, recovery-trigger timeline, and block-level
paired outcome chart.

## 5. Long-horizon memory story

Use four precise terms and avoid claiming more than implemented:

1. **Working memory:** current instruction, task plan, scene graph, monitor
   history, subgoal, budget, and execution receipt; episode-scoped.
2. **Procedural memory:** verified symbolic primitive sequence; durable JSON;
   reusable across layouts without storing low-level actions or metric poses.
3. **Failure/episodic memory:** verified failure context, intervention, outcome,
   and TTL; currently process-local and therefore not yet durable long-term
   memory.
4. **Semantic memory:** affordance/safety cards and scene relations; currently
   static/runtime context rather than online learned knowledge.

The novelty should be the **write authority and validation boundary**: planners
may propose, but only trusted postconditions/evaluators can promote a procedure
or failure record. Evaluation episodes must not write into the memory being
measured.

## 6. Multi-agent training: next stage, not current RA-L blocker

The advisor's proposal is viable, but it is a second research phase. Current
CARVE is inference-time orchestration; calling it autonomous VLA fine-tuning
would be inaccurate. Build the future loop as:

1. rollout collector agent gathers RGB/proprio/action chunks and receipts;
2. verifier agent labels only deployment-observable failure segments and trusted
   terminal outcomes;
3. curriculum agent clusters recurring failures and selects balanced corrective
   segments;
4. trainer performs pinned LoRA/SFT jobs on an immutable base checkpoint;
5. evaluator runs frozen held-out tasks and safety/fidelity tests;
6. admission agent promotes a checkpoint only if success improves without action
   contract, regression, deadline, or safety failure.

Never train directly on evaluator-private online state, never self-label all
failures with the planner, and never overwrite the deployed checkpoint. Compare
frozen VLA, naive self-training, and verified multi-agent training on a held-out
split fixed before collection. Until this loop exists, describe procedural
memory consolidation—not weight learning.

## 7. Ordered execution and stopping rules

| Phase | Work | GPU/robot cost | Gate |
|---|---|---:|---|
| P0 | Reset protocol repair, tests, 3-episode C3 diagnostic | Low | Exactly one acknowledged fresh reset/episode |
| P1 | RoboDojo set 1/2 four-condition matrix | ~12.7 GPU-hours | 200 complete episodes, no protocol violation |
| P2 | Six-episode procedural-memory sidecar | Small | Same planner schedule; frozen memory hash |
| P3 | Paper figures/claim audit | None | Claims match operational scope and statistics |
| P4 | Guanghua-1 adapter + shadow admission | Robot access | Safety/interface checklist passes |
| P5 | 40-trial minimum real-robot study | Robot access | No unreported intervention or exclusion |
| P6 | Verified multi-agent fine-tuning | Future | Separate dataset/protocol/held-out preregistration |

Do not start P1 until P0 passes. Do not let P2 alter P1. Do not start physical
action in P4 before independent robot-side limits and emergency stop are tested.
A null P1 result stops efficacy language; it does not trigger threshold tuning on
the same layout sets.

## 8. Paper positioning sentence

> CARVE-VLA complements memory-guided VLA harnesses by focusing on the
> deployment runtime: it couples a frozen VLA with compute-aware action reuse,
> observable-signal risk monitoring, capability-gated recovery, verified memory
> writes, and auditable lifecycle/safety contracts for long-horizon execution.

This sentence remains provisional until the fresh RoboDojo matrix and real-robot
admission support each clause.

---

Internet-derived descriptions of Harness VLA were paraphrased from the linked
paper/project sources for compliance with licensing restrictions; no long
verbatim passage is reproduced.

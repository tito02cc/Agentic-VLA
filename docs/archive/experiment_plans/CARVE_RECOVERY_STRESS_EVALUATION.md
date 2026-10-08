# CARVE Recovery Stress Evaluation

Updated: 2026-07-29

## Objective

Evaluate whether CARVE improves the behavior of a frozen PI0.5 policy after
an **execution-time deviation**, while retaining explicit timing and safety
accounting. The target is not a clean-LIBERO leaderboard score.

The primary question is:

> Given the same perturbed physical simulator state, can event-triggered,
> bounded recovery and one guarded semantic escalation improve recovery
> completion or reduce wasted VLA calls relative to continuation and ordinary
> replanning?

## Evidence Hierarchy

### Primary: CARVE Recovery Stress Protocol

Use the existing LIBERO MuJoCo integration and only real simulator dynamics.
Every episode records a pre-perturbation state, applies a deterministic online
perturbation, and writes a simulator snapshot only after a deployable monitor
raises a qualifying event.

Selected task families:

| Family | Representative task | Perturbation | Expected mechanism |
| --- | --- | --- | --- |
| placement relation | T6: mug/plate and pudding placement | object nudge during transport | monitor, physical recovery, reobserve, VLA replan |
| articulated placement | T9: mug into microwave | object nudge near insertion | monitor, recovery, guarded VLM subgoal, VLA replan |
| long horizon hard negative | T8: two moka pots on stove | action-age/deviation state | safe stop or rejected escalation when recovery envelope is unsupported |

The controller receives RGB, wrist RGB, proprioception, action history, action
age, and measured latency. Simulator state and injected-object identity are
evaluator-only data used for exact restoration and labels.

For every qualified snapshot, restore the exact same state and compare:

1. frozen continuation;
2. ordinary PI0.5 replan;
3. verified physical recovery plus PI0.5 replan;
4. verified physical recovery plus one guarded VLM `vla_act` decision and
   PI0.5 replan;
5. coupled CARVE policy, after its runtime profile passes its own admission.

Run three fixed PI0.5 noise seeds per state, pre-register the branch horizon,
and retain simulator videos for one success and one failure per family.

The reproducible core launcher is
`scripts/run_carve_recovery_stress_protocol.sh`. It executes the two supported
recovery states (T6 and T9), three fixed noise seeds (`7`, `17`, and `42`),
and exactly the `physical_recovery` and `vlm_physical_recovery` branches. It
uses eager PI0.5 because that is the currently verified closed-loop profile;
the VLM is supplied through an existing OpenAI-compatible endpoint and has no
direct action authority.

### Secondary: LIBERO-Plus Transfer Gate

LIBERO-Plus is a compatibility-oriented robustness extension of LIBERO. It
keeps the 7-D delta-action and two-camera observation contract, but applies
seven perturbation dimensions. CARVE will not use it as a 10,030-task
leaderboard campaign.

Use a fixed, zero-shot subset of 18 episodes:

- suites: `libero_10` and `libero_spatial`;
- task IDs: T6/T8/T9 plus three Spatial tasks selected before execution;
- perturbations: object layout, robot initial state, sensor noise;
- one official perturbed instance per task and three policy-noise seeds.

Compare frozen PI0.5 and CARVE under identical initial states. Report success,
event recall, false intervention rate, verified-recovery rate, recovery
latency, VLA calls per recovered episode, safe stops, and p50/p95
observation-to-action latency. A zero-shot failure under a severe visual shift
is a benchmark result, not a reason to silently fine-tune the base model.

### Held-Out: LIBERO-Pro

LIBERO-Pro is reserved for an independently generated position/task
perturbation check only after the Plus integration gate is stable. It should
not be pooled with LIBERO-Plus because their perturbation protocols differ.

## Environment Isolation

Official LIBERO-Plus replaces the Python package name `libero`; it must not be
installed into the existing PI0.5 evaluation environment. The project runner
now accepts `AGENTIC_VLA_LIBERO_ROOT`, so an isolated checkout or a
`vla-evaluation-harness` Docker image can be selected without modifying the
original `LIBERO/` checkout.

The authoritative public repositories are kept outside the tracked source tree
under `/home/admin1/ct/benchmark-sources/`. Their asset packs and benchmark
containers are deliberately not copied into this repository.

The official Plus asset archive is approximately 6.0 GB. Its resumable download
is kept under `/home/admin1/ct/benchmark-sources/LIBERO-plus/downloads/` and is
not an admission prerequisite for the primary recovery experiment.

## Admission Rules

Do not run a broad Plus/Pro sweep until all of the following pass:

1. a single zero-shot environment reset and PI0.5 action call has the expected
   two images, 8-D policy state, and 7-D action contract;
2. a clean task reproduces a valid rollout video;
3. the online perturbation is visible in traces but excluded from controller
   inputs except through observations and monitor signals;
4. snapshot restoration is exact and deterministic;
5. every branch emits the unified CARVE trace and a profile receipt.

## Expected Claims

The study may support that CARVE provides bounded, auditable intervention and
can reduce redundant policy work on selected recoverable deviations. It may
not claim benchmark-wide generalization from the primary stress protocol, nor
a universal VLM gain from a small paired state set.

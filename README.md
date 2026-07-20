# CARVE-VLA

This repository contains **CARVE-VLA: a Compute-Adaptive Agentic Runtime for reliable long-horizon VLA execution**.

The current research story is:

1. **CARVE Agentic Harness** supervises frozen VLA execution with deployable
   monitoring, bounded recovery, memory, retry/escalation, fallback, and traces.
2. **CARVE Optimize Runtime** calibrates policy-specific compute profiles under
   action-fidelity, latency, memory, and deadline constraints.
3. Efficient VLA inference is the current primary research contribution. The
   Agentic Harness supplies the system workload and closed-loop acceptance
   boundary instead of growing into another general-purpose Agent framework.
4. PI0.5 is the main closed-loop backend. OpenVLA-7B is the second, materially
   different autoregressive backend used to test adapter generality and
   standard low-bit deployment, not to add another leaderboard campaign.

The current main draft is:

- `paper/CARVE-VLA/root.tex`
- `paper/CARVE-VLA/root.pdf`

The canonical current status entry point is:

- `docs/status/CARVE_VLA_COMPLETED_WORK.md`

Supporting architecture and execution documents:

- `docs/plans/CARVE_VLA_NEXT_STAGE_PLAN.md`
- `docs/architecture/CARVE_RUNTIME_ARCHITECTURE.md`

`docs/status/EXPERIMENT_AND_PAPER_STATUS_20260612.md` is retained as a
historical June milestone and is not the current completion record.

## Current Runtime Evidence

On one RTX 4090, PI0.5 compiled BF16 reduces fixed-replay P50/P95 from
`154.34/159.59 ms` to `65.73/67.40 ms`. Static Masked-View Elision (SMVE),
which removes only adapter-guaranteed padding views, further reaches
`54.35/56.19 ms` while passing all 45 fixed-noise replay checks.

Under a real co-resident Qwen3.5-4B visual workload, ordinary compiled BF16
misses an 80 ms deadline on `72.8%` of 500 calls. SMVE reduces the miss rate to
`0.6%` while both profiles retain the same replay fidelity gate. Paired
synchronous Task 8/9 evaluation records `8/10` for SMVE and `7/10` for ordinary
compiled BF16; this is treated as non-inferiority evidence, not a success-rate
improvement claim.

The current Agentic systems gate is the PI0.5 Recovery Challenge in real
LIBERO MuJoCo. Across three exact restored states, frozen continuation,
frequent replan, prompt retry, and physical recovery each complete `2/3`.
However, replan and prompt retry require `452/508` PI0.5 calls versus `113` for
continuation. Physical recovery verifies both supported stall states and fails
closed on an unsupported stale-action event. A separate online T6 run completes
automatic monitoring, 12 bounded recovery actions, verification, replanning,
and task success. These results support event-triggered compute and explicit
recovery contracts; they are not reported as benchmark-wide success gains.

The coupled Agentic-Optimize gate restores the same T6/T9 stall states under
eager BF16, compiled BF16, and compiled BF16 + SMVE. All profiles preserve
`2/2` task outcomes and verified recoveries. Runtime P95 falls from `166.26 ms`
to `65.75/54.50 ms`, and 80 ms deadline misses fall from `236/236` to zero for
both admitted profiles. This is the direct bridge between Agentic recovery and
the current efficient-inference contribution.

The same CARVE profiling and fidelity boundary now runs a materially different
OpenVLA-7B autoregressive policy. On ten paired real LIBERO Task 8/9 frames,
BF16 requires `14.42 GB` peak VRAM and reaches `303.48/310.92 ms` P50/P95.
BitsAndBytes INT8 and NF4 reduce peak VRAM to `7.76/4.41 GB`, but increase P50
to `1533.96/748.35 ms` and fail the predeclared exact-action gate. They are
retained as rejected memory-oriented profiles, not promoted as realtime modes.
Critical-path profiling attributes `67.2%` of OpenVLA BF16 CUDA time to six
autoregressive decode calls. A CARVE `torch.compile` language-model profile,
with CUDA Graphs disabled and both observed prompt-length buckets prewarmed,
reduces steady-state P50/P95 to `224.25/231.84 ms` with `10/10` exactly matching
actions. It requires `200.19 s` of profile preparation and still misses the
100 ms deadline, so it is accepted as a bounded latency optimization rather
than a realtime solution.

The following are retained results from the earlier Agentic-VLA paper phase,
not the sole evidence for the current Optimize Runtime contribution. Their raw
rollout directories were not retained during the earlier cleanup, so they are
treated as historical context rather than the primary reproducible result:

Main LIBERO-10 comparison:

- `pi05_libero` baseline: `90.0%` (`180/200`)
- refined full CARVE-VLA: `92.5%` (`185/200`)
- dominant weak task: `55.0%` -> `75.0%`

Earlier deployment-oriented robosuite study:

- robosuite Stack raw pi0.5 fixed `1-step`: `0/10`
- pi0.5 + Agentic retry fixed `1-step`: `10/10`
- full pi0.5 calls: `43.0/episode` -> `6.0/episode`

Main result files:

- `paper/CARVE-VLA/root.tex`
- `paper/CARVE-VLA/root.pdf`
- `paper/archive/Agentic-VLA-v1-submission/agentic_vla_paper_v1.pdf`
- `paper/archive/Agentic-VLA-v1-submission/agentic_vla_paper_v1.tex`
- `results/carve_pi05_recovery_challenge_20260719/REPORT.md`
- `docs/reports/CARVE_VLA_MIDTERM_REPORT_20260717.md`

## Important Directories

- `docs/`: architecture, experiment plans, research updates, methods, and status logs.
- `scripts/`: current experiment, sweep, plotting, data collection, and compact-policy scripts.
- `openpi/`: OpenPI policy stack and local pi0.5 integration.
- `agentic_vla/runtime/`: model-neutral Agentic and policy contracts, including
  PI0.5 and OpenVLA adapters.
- `agentic_vla/optimization/`: model/backend plugins, calibrated profiles,
  fidelity gates, benchmark reports, and deployment manifests.
- `quantization/`: lightweight inference and quantization utilities retained for VLA deployment work.
- `paper/Agentic Policy/`: source papers and survey notes.
- `paper/Agentic-RAG-VLM/`: previous paper project, intentionally preserved.
- `paper/CARVE-VLA/`: current 15-page WAICA/LNCS-style manuscript.
- `paper/archive/`: preserved previous submissions and superseded drafts.

## Cleanup State

The repository root was renamed to `CARVE-VLA` and its active documentation was
organized under `docs/` on 2026-07-16. No paper, result, checkpoint, or model
asset was removed during this reorganization.

Retained large items:

- `weights/openpi-assets`
- `checkpoints/pi05_robosuite_stack_smoke`
- local `LIBERO/` and `openpi/` checkouts

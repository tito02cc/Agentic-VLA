# Agentic-VLA

Agentic-VLA is a research system for **boundary-gated agent supervision of action policies** and **quality-aware inference optimization**. The current paper calls the method **BIND-VLA**. `CARVE-VLA` remains in historical file paths and code identifiers so old runs and citations stay traceable; it is not a second current method. The project does not train a new VLA foundation model.

The repository also includes the earlier [Agentic RAG-VLM implementation](Agentic-RAG-VLM/README.md) and [paper source](paper/Agentic-RAG-VLM/). It is related background work, **not** a second benchmark arm of BIND-VLA; do not combine its grasping results with the current VLA results.

The system separates three responsibilities:

1. A frozen action policy (currently PI0.5-family checkpoints in the main experiments) proposes action chunks from fresh observations.
2. A high-frequency execution-risk Monitor records signals; a lower-frequency VLM Planner/Critic proposes semantic changes. The Harness checks freshness, allowed skills, task scope and budgets **at an action-chunk boundary** before changing the next subgoal. An execution receipt does not by itself prove task success.
3. Optimize Runtime admits model/backend profiles only after contract, action-replay, resource and task-quality checks. Model-call latency and complete robot-task wall time are reported separately.

The software contracts allow other action policies to be adapted, but **cross-policy performance is not established** by the current evidence. The paper and experiments keep the RoboMME/JAX, LIBERO/PyTorch and RoboDojo checkpoints separate.

## Start Here on Another Computer

```bash
git clone https://github.com/tito02cc/Agentic-VLA.git
cd Agentic-VLA
```

| Need | Entry point |
| --- | --- |
| Full method, protocol, results and limitations | [BIND-VLA technical report](paper/CARVE-VLA/ral_draft/TECHNICAL_REPORT.md) |
| Editable paper and current PDF | [LaTeX](paper/CARVE-VLA/ral_draft/main.tex), [PDF](paper/CARVE-VLA/ral_draft/main.pdf) |
| Figure/chart prompts and source rules | [Figure and table brief](paper/CARVE-VLA/ral_draft/FIGURE_AND_TABLE_BRIEF.md), [provenance](paper/CARVE-VLA/ral_draft/FIGURE_PROVENANCE.md) |
| Model weights, paths and upstream software | [Model and environment setup](MODEL_AND_ENVIRONMENT_SETUP.md) |
| Videos and what each actually shows | [Video evidence index](paper/CARVE-VLA/ral_draft/VIDEO_EVIDENCE_INDEX.md), [RoboDojo shortlist](deliverables/BIND_VLA_VIDEO_SHORTLIST_20260924/README.md), [paired and cross-benchmark video guide](deliverables/BIND_VLA_PAPER_HANDOFF_20260924/videos/VIDEO_GUIDE.md) |
| Current research route | [Execution plan](docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md) |
| Historical work | [Agentic RAG-VLM](paper/Agentic-RAG-VLM/), [archive](paper/archive/) |
| Agentic RAG-VLM code and its own reproduction limits | [Earlier project README](Agentic-RAG-VLM/README.md), [reproduction notes](Agentic-RAG-VLM/docs/REPRODUCTION.md) |

The repository includes both projects' code, documentation and paper sources, plus key aggregate evidence and selected short videos. It **does not include model weights, the school-provided Guanghua robot meshes, full simulator assets, Python environments, personal midterm files, all per-step rollouts or every video listed by the source-machine catalog**. See the explicit [public upload scope](docs/handoff/PUBLIC_REPOSITORY_SCOPE_20261008.md). No model checkpoint or existing paper was removed from this local workspace during publication. The previous root README is preserved at [docs/archive/README_20260922.md](docs/archive/README_20260922.md).

## Current Evidence, With Scope

| Experiment | Observation | Boundary |
| --- | --- | --- |
| RoboMME eight-task development study | Raw/Harness/selective variants `23/80`, `36/80`, `42/80` | Selective condition combines development versions; not a frozen confirmatory result |
| Fixed RoboMME identity-memory pair | `5/12 -> 9/12`, 5 rescues and 1 harm | One task family; exact paired `p=0.21875` |
| Cross-task memory transfer | `15/16 -> 13/16` | Always-on memory can hurt |
| Frozen Raw/Harness stress check | `3/32 -> 4/32` | Limited improvement; `p=1.0` |
| Local LIBERO/LIBERO-PRO study | `180/400 -> 183/400`; VLA calls `-8.4%`, total wall `+16.5%` | Ten initial states per task, not the official 50-state leaderboard |
| Independent PyTorch PI0.5 model calls | P95 `282.43 -> 54.67 ms` across 7-step eager to 2-step compiled+SMVE | Recorded inputs; **not** RoboMME/JAX or whole-robot-loop speed |
| RoboDojo controlled-fault tower | Historical PI-v3 C1/C3 `0/3 -> 3/3` | Injected fault, replayed Planner decision and profile change; not natural-task PI0.5 gain |

These numbers should not be combined into a claim that the *same frozen system* is both more successful and faster end-to-end. The technical report states exactly which conditions are exploratory, paired, rejected or independently timed. A RoboDojo `organize_table` video marked 100 points was invalidated by an early-return audit and is excluded from the representative set.

## Code Map

- `agentic_vla/runtime/`: policy adapters, Monitor, Harness, session lifecycle and traces.
- `agentic_vla/toolchain/`: typed tool intents, task plans, scoped memory and recovery verification.
- `agentic_vla/optimization/`: profile contracts, fidelity/admission, backend options and fallback.
- `agentic_vla/benchmarks/`: benchmark-facing observation/action adapters and evaluator isolation.
- `scripts/`: experiment runners, audits, plotting and model/service launchers.
- `tests/`: CPU and integration regression tests. Tests do not substitute for full simulator evaluation.
- `integration_patches/robodojo/`: local RoboDojo/XPolicyLab source changes needed by the historical evaluation environment; apply only to the documented pinned upstream commits.

For paper writing alone, the technical report, manuscript, figures, evidence JSON and videos are sufficient. To rerun simulations, follow [MODEL_AND_ENVIRONMENT_SETUP.md](MODEL_AND_ENVIRONMENT_SETUP.md) and check checkpoint revisions, normalizers, camera/action semantics and simulator versions before running any benchmark. The local directory may still be named `CARVE-VLA`; renaming that directory is not required and would invalidate historical absolute paths.

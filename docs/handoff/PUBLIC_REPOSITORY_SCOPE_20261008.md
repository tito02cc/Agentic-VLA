# Public Repository Scope and Resume Notes

Updated 2026-10-08 for the public `tito02cc/Agentic-VLA` repository. This file records what was selected from the larger source-machine workspace. GitHub is a **code, paper and compact-evidence snapshot**, not a mirror of the local filesystem. The source machine has much larger model, dataset, simulator and intermediate-output trees. Omitting those files does not turn an unrun experiment into a reproducible result.

## Included

- Current Agentic-VLA runtime, toolchain, optimization, benchmark adapters, configs, scripts and tests; the BIND-VLA technical report, editable draft, active figure sources and references.
- Agentic RAG-VLM implementation (`Agentic-RAG-VLM/` code, configs, knowledge-base cards, docs and scripts) and its authored paper under `paper/Agentic-RAG-VLM/`. The prior paper is independent background, not a BIND-VLA benchmark arm.
- Compact experiment aggregates/receipts and representative simulation videos with explicit task, model, condition and outcome labels. `VIDEO_CATALOG.tsv` is a **source-machine index** and does not imply all indexed files were uploaded.
- Small RoboDojo/XPolicyLab integration patches and additive adapter files. Upstream repositories themselves remain separately cloneable at pinned commits.
- Earlier tracked paper and result files already in the remote repository, unless Git's history says otherwise; this publication does not intentionally delete them.

## Intentionally absent or only partial

| Excluded item | Reason | Recovery route |
| --- | --- | --- |
| PI0.5, StarVLA, Qwen and LoRA weights; local model caches | Multi-GB assets and distinct licenses/revisions; Git is the wrong distribution channel | [Model guide](../../MODEL_AND_ENVIRONMENT_SETUP.md) lists official sources, expected paths, variables and checkpoint checks |
| RoboDojo `Assets` (~66 GB), LIBERO assets and full third-party checkouts | Large redistributable/upstream-managed data and engine installations | Clone the named upstream repository, install per upstream docs and apply the recorded local patches |
| School-provided Guanghua hand/robot URDF meshes | Public redistribution permission not verified | Obtain from the original provider under its terms; `Agentic-RAG-VLM/scripts/prepare_assets.py` cannot regenerate missing source meshes from nothing |
| Original Agentic RAG-VLM paper-era 116-item experience library and unpublished data | Not present as a distributable original dataset in this workspace | See `Agentic-RAG-VLM/docs/REPRODUCTION.md`; reconstructed public cards are **not** the original dataset |
| Full raw `results/`, `artifacts/`, Agentic RAG-VLM `output/`, all 4,512 cataloged video files | Multiple GB, many exploratory or duplicate runs; a public repository should not misrepresent them as independent confirmation | Compact JSON/CSV and selected videos are public; four selected raw-data archives are in the [private evidence dataset](https://huggingface.co/datasets/Minth-Group/Agentic-VLA-reproduction). The complete project output trees and available videos at the snapshot are in the [private handoff dataset](https://huggingface.co/datasets/Minth-Group/Agentic-VLA-machine-handoff-20261008); see its manifest rather than assuming every catalog entry still existed on disk |
| Personal `midterm/` and `中期/`, meeting recordings, local tokens/credentials, virtual environments | Personal information, consent and security | User-owned midterm files and available recordings were privately backed up with permission, not made public. Other students' `midterm/参考/`, credentials and virtual environments were excluded; see [audit](MACHINE_RETIREMENT_AUDIT_20261008.md) |
| Additional third-party reference PDFs not already present | Copyright/license of redistribution varies | Use upstream paper links and citation keys; existing remote reference material was not deleted by this update |

## Evidence and code-use cautions

1. The current paper is a **draft**. RoboMME eight-task C3 numbers mix development versions; the fixed memory test has 12 pairs and a negative transfer task; the frozen two-task Raw/Harness stress result is `3/32->4/32`. Do not turn exploratory gains into a universal method claim.
2. The PyTorch PI0.5 P95 `282.43->54.67 ms` is a separate model-call experiment, not the JAX RoboMME robot loop or a hard real-time guarantee. RoboDojo historical PI-v3 controlled-fault videos are not natural-task official PI0.5 improvements.
3. The local working directory may retain uncommitted experiments and historical deletions not reflected in this public snapshot. The public update deliberately selects files instead of `git add -A` on the dirty source tree. Compare a new run against its own frozen configuration rather than silently relying on the source-machine path names.
4. The RoboDojo upstream patches are supplied to make the adapter work inspectable; on a fresh machine they still require version-specific environment validation. The repo is not a single-command reproduction of Isaac Sim, VLM and VLA training/inference.

For writing on another computer, start at the root `README.md`, then `paper/CARVE-VLA/ral_draft/TECHNICAL_REPORT.md`, `main.tex`, `FIGURE_AND_TABLE_BRIEF.md`, and the saved `results/`/`artifacts/` entries. For deeper rollout audit, see [HF evidence download instructions](HF_EVIDENCE_DATASET_CARD_20261008.md). For execution, read `MODEL_AND_ENVIRONMENT_SETUP.md` first.

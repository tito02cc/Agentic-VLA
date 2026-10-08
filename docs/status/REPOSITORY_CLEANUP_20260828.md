# CARVE-VLA Repository Cleanup Record

Date: 2026-08-28

## September 10: Navigation And Safe Archival

The sections below describe the August cleanup, not newly freed disk space.
On September 10, only generated caches were deleted: root `.pytest_cache` and
`.ruff_cache`, and `__pycache__` directories under `agentic_vla/`, `scripts/`
and `tests/`. Test runs may regenerate them; the editor now hides only these
cache categories. No result, video, weight, environment or paper was deleted.

Five old plans moved unchanged from `docs/plans/` to
`docs/archive/experiment_plans/`. The active plan is now unambiguous in the
root README and documentation index. Existing experiment scripts and raw
evidence paths remain unchanged. Frozen preregistrations remain in place.

| Moved file | Destination | Verified SHA256 before and after |
|---|---|---|
| `CARVE_CURRENT_BENCHMARK_MODEL_PLAN_20260828.md` | `docs/archive/experiment_plans/` | `e6ff24a8f5b525d2feb26d532e278ad0216a612ac8f02d9918c87e26b15c0c8f` |
| `CARVE_MODERN_VLA_ROBOTWIN2_PLAN_20260828.md` | same archive | `5c368b93a58b3b13b17b77842c4620f3bafd66e33ebaee06f286b2e97c5511fb` |
| `CARVE_ROBODOJO_CURRENT_MODEL_PLAN_20260828.md` | same archive | `c096a7ffe3c6380896955c9356b637d9f4ff23e540b44b622a6aab885d7aa395` |
| `CARVE_ROBOMME_EXPERIMENT_PLAN_20260828.md` | same archive | `d31cd98200c150afb6bbeca9ac3bd54e27942dc54bb39dc5de944476cfe1d2fd` |
| `CARVE_VLA_NEXT_STAGE_PLAN.md` | same archive | `3b266e768f10de2919d99400a80b93dcaa7ad54b73c250ff5a03052b8410b2f2` |
| root `guanghua_hand_env_bundle_20260827_134025.zip` | `exports/` | `4d84bb14b3ee04b5665b4913dd04f7c4e84143c55d26f1605db85c188a09a1f6` |
| root `MUJOCO_LOG.TXT` | `artifacts/housekeeping/MUJOCO_LOG_pre_20260910.TXT` | `2168704eea5ad1d243f188e6ade058ba1e1d51b717f76aaba83c6cf187bdc286` |

Archived plan contents and previous deliverable packages are preserved. Links
inside those historical snapshots may refer to their old location; use this
map. Active indices and the live August report point to the current locations.
`paper/`, `Agentic-RAG-VLM/`, `midterm/`, model dependencies and Git history were
excluded from the cleanup. Existing unrelated Git changes were not reverted.

The root README, documentation index, directory map and AI context now
distinguish current RoboDojo work from August results. Historical reports are
not current launcher instructions, and the retained-result list is not an
exhaustive inventory. No code/import or raw-experiment path was relocated.

September 10 verification: all seven moved-file SHA256 values matched the
record above; all 12 local Markdown links in the five primary navigation
documents resolved; eight protected paper/material/evidence paths existed;
VS Code settings parsed as valid JSON. Scoped `git diff --check` passed.
These are navigation/move checks, not a new full content audit of all papers
or historical experiments.

## Objective

Reduce local disk use and remove obsolete experiment clutter without deleting
paper sources, canonical benchmark evidence, current PI0.5 dependencies,
representative videos or thesis/interview materials.

## Space Change

| Metric | Before | After |
|---|---:|---:|
| Repository apparent disk usage | 69,907,570,740 bytes | 40,416,382,619 bytes |
| Space released | | 29,491,188,121 bytes (about 27.5 GiB) |
| Top-level displayed size | about 66 GB | about 38 GB |

The remaining 28 GB `.git` directory is historical Git object storage, not the
current working tree. It contains several early multi-gigabyte checkpoint
blobs. Removing it safely requires an explicit history rewrite and remote
coordination, so it was intentionally excluded from this cleanup.

## Protected Content

The following were preserved without content deletion:

- `paper/Agentic-RAG-VLM/`, including `main_final.pdf`;
- `paper/Agentic Policy/` and all local reading material;
- `paper/CARVE-VLA/`, including `root.tex`, `root.pdf`, figures and bibliography;
- `paper/archive/`, including the previous Agentic-VLA submission;
- `中期/`, including user-owned PPT, forms and references;
- `openpi/` and `openpi/.venv/`, required by current PI0.5 execution;
- `LIBERO/`, required by the formal simulator study;
- `third_party/`, including RPent and RoboTwin references;
- the active PI0.5 LIBERO checkpoint in the external OpenPI cache;
- all paths listed below under canonical evidence.

Only disposable Python/test/LaTeX build caches were removed inside protected
paper/code directories. No paper source, paper PDF, figure or bibliography was
deleted.

## Canonical Results Retained

`results/` was reduced from 182 roots to these 11 evidence roots:

1. `results/_runtime/`;
2. `results/carve_efficiency_full_20260826/`;
3. `results/carve_optimize/`;
4. `results/cross_task_memory_routing_20260827/`;
5. `results/full_embodied_agent_20260827/`;
6. `results/libero_pro_embodied_tool_probe_20260827/`;
7. `results/libero_pro_full_study_20260825/`;
8. `results/libero_pro_representative_videos_20260828/`;
9. `results/paper_ready_20260827/`;
10. `results/planner_profile_closed_loop_20260827/`;
11. `results/vlm_embodied_tool_use_20260827/`.

The deleted 171 roots were smoke runs, failed qualifications, superseded
Panda/Robosuite/ToolHang studies, RAL exploratory matrices and intermediate
LIBERO-PRO gates already replaced by the audited full study. Their accepted
aggregate conclusions remain in current reports; they are not required by
`results/paper_ready_20260827/summary.json` or
`representative_artifacts.json`.

## Scripts

- Before: 137 top-level scripts.
- Active after cleanup: 57 scripts.
- Removed from the active directory: 80 scripts.
- Recovery archive: `exports/LEGACY_SCRIPTS_20260828.zip`.
- Archive SHA256:
  `bbafaf908cc0df630c17d8553cafbf1f9255e6c6fe09af40dff3b22f3792acd6`.

The active set retains framework validation, PI0.5 profiling, semantic tests,
LIBERO-PRO execution/audit, paper table generation, memory routing, video/tool
probes, Optimize Runtime summaries and future RoboMME admission. Old OpenVLA,
LingBot/RoboTwin-RQ5, Panda, Robosuite, ToolHang and superseded RAL launchers are
available only in the compressed archive.

## Heavy Assets Removed

- OpenVLA-7B LIBERO checkpoint and download logs: about 15 GB;
- two obsolete OpenVLA virtual environments: about 407 MB;
- unused `pi0_base` parameter checkpoint: about 12 GB;
- old PI0.5 Robosuite smoke checkpoints: about 51 MB on disk;
- ToolHang virtual environment: about 530 MB;
- downloaded ToolHang datasets: about 221 MB;
- unpacked duplicate weekend export directory: about 15 MB;
- generated Python, pytest and current-LaTeX build caches.

The small `pi0_base/assets/` normalization files were retained under
`weights/`; only the large JAX parameter tree was removed. All removed model
and dataset assets are re-downloadable and were not part of the canonical
CARVE evidence.

## Documentation Cleanup

Nineteen superseded active plans/status pages were removed after their final
conclusions had been absorbed by the main report. Historical paper files and
the `docs/archive/` provenance tree were preserved. The active documentation
entry points are now `docs/README.md` and
`docs/reports/CARVE_VLA_CURRENT_EXPERIMENT_REPORT_20260827.md`.

## Post-Cleanup Verification

### Code contracts

```text
72 passed in 0.63s
```

The focused suite covers RoboMME, LIBERO and RoboTwin adapters, Runtime,
canonical tools, Toolchain and tool-hang contracts.

### Formal LIBERO/LIBERO-PRO evidence

The full audit was rerun after deletion:

```text
status: passed
cells: 120/120
episodes: 1,200/1,200
videos: 1,200/1,200
decoded frames: 454,540
paired state identities: 400
checkpoint identities: 1
deployment profile identities: 1
failures: 0
```

### Deliverables

- `paper/CARVE-VLA/root.pdf`: readable, 28 pages;
- `paper/Agentic-RAG-VLM/main_final.pdf`: readable, 8 pages;
- ten curated representative videos: all H.264 streams readable;
- `exports/CARVE_VLA_WEEKEND_PACKAGE_20260828.zip`: integrity passed;
- weekend package SHA256:
  `d1481db63cfbe90341832a11a9ea4e921b6ef0e4d3b9c2eb8ecb4c7da0c4f39c`;
- `scripts/build_current_evidence_package.py`: successfully rebuilt the
  canonical evidence package after cleanup.

## Future Cleanup Boundary

The next material disk reduction would require rewriting Git history to remove
the historical 7.4 GB checkpoint blobs. This should be done only as a separate,
explicit repository migration: first commit/export the current working tree,
then rewrite history, force-push a coordinated remote, and verify a fresh
clone. It must not be mixed with routine result cleanup.

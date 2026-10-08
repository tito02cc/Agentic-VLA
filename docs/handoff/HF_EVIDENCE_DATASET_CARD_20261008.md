---
pretty_name: Agentic-VLA Reproduction Evidence
tags:
- robotics
- embodied-ai
- agentic-vla
- simulation
---

# Agentic-VLA reproduction evidence

This private dataset accompanies the public [Agentic-VLA code and paper repository](https://github.com/tito02cc/Agentic-VLA). It contains selected **original experiment outputs** that are too bulky for the GitHub snapshot. It is not a model repository, a training dataset, or a claim that the archived experiments all used the same policy. The four archives preserve workspace-relative paths so the technical report's source references can be inspected after extraction.

| Archive | Bytes | Contents and scope |
| --- | ---: | --- |
| `archives/libero-pro-raw-20260825.tar.zst` | 628,014,384 | Local LIBERO-PRO full-study episode records, logs, and videos. The published `180/400 -> 183/400` result is a ten-initial-state-per-task study, not the official 50-state leaderboard. |
| `archives/robomme-raw-20260831-20260924.tar.zst` | 257,622,353 | Selected B1/C2/C3 source runs plus fixed memory, transfer, selective-scheduling, and frozen Raw/Harness paired rollouts. C3's eight-task aggregate combines development versions; see the report before interpreting it. |
| `archives/robodojo-selected-raw-202609.tar.zst` | 217,518,389 | Three controlled-fault tower pairs, official PI0.5 sorting/admission cases, selected historical PI-v3 runs, and invalid-result audit. Controlled-fault success is **not** natural-task PI0.5 Agent gain. |
| `archives/agentic-rag-vlm-selected-raw-202609.tar.zst` | 374,656,493 | Reconstructed Agentic RAG-VLM challenge-v4 outputs, defense-suite outputs, and selected VLABench simulation traces. This does not contain the original private paper-era experience bank or school robot meshes. |

The public repository already contains the smaller aggregate JSON/CSV, representative videos, manuscript, code, and the detailed [model/environment guide](https://github.com/tito02cc/Agentic-VLA/blob/main/MODEL_AND_ENVIRONMENT_SETUP.md). Read its [technical report](https://github.com/tito02cc/Agentic-VLA/blob/main/paper/CARVE-VLA/ral_draft/TECHNICAL_REPORT.md) and each experiment's condition labels before using a result in a paper. The selected archives are an audit aid, not a one-command replication: simulation installations, checkpoints, normalization assets, and original external datasets are separate dependencies.

## Download and inspect

Log into the `Minth-Group` Hugging Face account on the new computer, then:

```bash
hf auth login
hf download Minth-Group/Agentic-VLA-reproduction \
  --repo-type dataset --local-dir ./Agentic-VLA-reproduction
cd Agentic-VLA-reproduction
sha256sum -c SHA256SUMS
mkdir -p extracted
tar -I zstd -xf archives/robomme-raw-20260831-20260924.tar.zst -C extracted
```

Extract archives into a fresh directory first. They contain relative `results/`, `artifacts/`, or `Agentic-RAG-VLM/output/` paths, and should not overwrite newer local experiments by accident. `SHA256SUMS` verifies transferred archive bytes; it does not validate the scientific interpretation of a rollout.

## Not mirrored here

- PI0.5, Qwen, RoboMME, RoboDojo, and StarVLA weights are sourced from their respective official publishers. The GitHub [model guide](https://github.com/tito02cc/Agentic-VLA/blob/main/MODEL_AND_ENVIRONMENT_SETUP.md) provides identifiers, download commands, revisions where pinned, and conversion instructions. Rehosting third-party checkpoints here could blur provenance and redistribution terms.
- RoboDojo/Isaac assets, other upstream simulator checkouts, school-provided Guanghua robot meshes, the original 116-item experience library, and personal material are not included. Obtain them from their owners under the applicable terms.
- Exploratory outputs outside these selected source directories remain on the original machine. This archive is not a complete mirror of every failed or diagnostic run.

No license is asserted for third-party benchmark content or model-generated outputs by this dataset card. Contact the relevant original publisher before redistributing any external assets.

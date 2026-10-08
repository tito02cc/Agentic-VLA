# Model and Environment Setup for a Fresh Clone

Updated 2026-10-08. The GitHub repository contains **no model weights or full simulator assets**. This document separates the checkpoints actually used in different experiments; substituting one PI0.5 checkpoint for another changes the experiment. Paths below are defaults read by this repository's scripts, not mandatory system locations. Set the shown environment variables when storing models elsewhere.

## 1. Choose the experiment first

| Workload | Required models | Main entry points |
| --- | --- | --- |
| Read/edit the paper or inspect saved results/videos | **None** | `paper/CARVE-VLA/ral_draft/`, `results/`, `artifacts/` |
| Agentic RAG-VLM reconstructed framework | `Qwen/Qwen3.5-4B` for its recorded multimodal experiments; some paper-era assets/experience data are not present | `Agentic-RAG-VLM/README.md`, `Agentic-RAG-VLM/docs/REPRODUCTION.md` |
| RoboMME GroundSG Agentic experiments | `Yinpei/mme_vla_suite` grounded-subgoal PI0.5 step 79999; `Qwen/Qwen3-VL-4B-Instruct`; `Yinpei/vlm_subgoal_predictor` QwenVL adapter step 1200 | `scripts/serve_robomme_groundsg_policy.sh`, `scripts/serve_robomme_groundsg_planner.sh` |
| LIBERO/LIBERO-PRO and PyTorch inference profiles | Physical Intelligence `pi05_libero` converted to PyTorch; Planner variant also needs its stated Qwen model | `scripts/run_libero_pro_full_study.sh`, `scripts/benchmark_carve_pi05_profile.py` |
| Current RoboDojo PI0.5 path | Official RoboDojo ARX-X5 PI0.5 checkpoint step 59999, plus simulator assets | `scripts/prepare_robodojo_pi05.py`, `scripts/run_robodojo_pi05_admission.py` |
| Historical RoboDojo StarVLA PI-v3 videos | Optional StarVLA PI-v3 checkpoint and Qwen3-VL-4B base | See `docs/status/ROBODOJO_STARVLA_SETUP_20260831.md`; **not needed** to read the current paper |

These models are not interchangeable. RoboMME uses a JAX GroundSG PI0.5 and its task-specific Qwen adapter; the independent 54.67 ms timing uses a PyTorch LIBERO checkpoint; RoboDojo uses another official JAX checkpoint. The paper does not report a universal speedup across these backends.

## 2. RoboMME: main Agentic evidence

Official sources: [MME-VLA checkpoint repository](https://huggingface.co/Yinpei/mme_vla_suite), [VLM subgoal predictor](https://huggingface.co/Yinpei/vlm_subgoal_predictor), [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). The local experiments used these components:

| Component | Expected default location | Launch-time variable / validation |
| --- | --- | --- |
| Grounded-subgoal PI0.5 checkpoint `symbolic-grounded-subgoal/79999` | `~/.cache/carve-vla/checkpoints/robomme/mme_vla_suite/symbolic-grounded-subgoal/79999/` | `ROBOMME_CHECKPOINT`; must contain `params/`, `assets/robomme/norm_stats.json`, and parent `history_config.txt` |
| Qwen3-VL-4B-Instruct base | `~/.cache/carve-vla/checkpoints/Qwen3-VL-4B-Instruct/` | `ROBOMME_PLANNER_MODEL`; model shards, config and tokenizer must be present |
| QwenVL grounded-subgoal LoRA `checkpoint-1200` | `~/.cache/carve-vla/checkpoints/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200/` | `ROBOMME_PLANNER_ADAPTER`; needs `adapter_config.json` and `adapter_model.safetensors` |

With an installed [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), download only the required subtrees (approximately 11.55 GB for the policy checkpoint, plus the separate VLM):

```bash
hf download Yinpei/mme_vla_suite \
  --include 'symbolic-grounded-subgoal/79999/**' 'symbolic-grounded-subgoal/history_config.txt' \
  --local-dir "$HOME/.cache/carve-vla/checkpoints/robomme/mme_vla_suite"
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$HOME/.cache/carve-vla/checkpoints/Qwen3-VL-4B-Instruct"
hf download Yinpei/vlm_subgoal_predictor \
  --include 'qwenvl/grounded_subgoal/checkpoint-1200/**' \
  --local-dir "$HOME/.cache/carve-vla/checkpoints/robomme/vlm_subgoal_predictor"
```

On this machine, the local GroundSG policy archive was checked against LFS SHA-256 `4a9577f5afa8e5225cb2e69e3820d4c7e5c7deb18f2ef00c10b4401eab575f94`. That digest described the **downloaded archive**, not a generic hash of the extracted directory; do not compare it to an arbitrary file. Record the HF revision resolved on the new machine. The Qwen base revision was not pinned in the old launch script, so a new download alone is not an exact bitwise reproduction.

The policy launcher expects `ROBOMME_POLICY_REPO=third_party/robomme_policy_learning` and `POLICY_PYTHON=openpi/.venv/bin/python` by default. The Planner launcher defaults to a **machine-specific** Swift executable `/home/admin1/miniconda3/envs/g2agent/bin/swift`; override it on the new computer, for example `ROBOMME_SWIFT="$(command -v swift)"`. Check the launch scripts before installing dependencies. RoboMME upstream commits used here are listed in Section 5.

## 2A. Agentic RAG-VLM model and assets

The separately published Agentic RAG-VLM reconstruction used a local [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) multimodal model. Its scripts default to the **old machine-specific** path `/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B`; pass `--model /your/path/Qwen3.5-4B` instead. A standard download command is:

```bash
hf download Qwen/Qwen3.5-4B --local-dir "$HOME/models/Qwen3.5-4B"
```

The public repository contains the reconstructed Agent/skill code, configurations, documents and selected aggregate evidence. It does **not** contain the school-provided Guanghua robot mesh/URDF bundle (public distribution permission not verified), the original paper-era private 116-item experience bank, or the full local `output/` tree. Consequently, the paper's original experiment cannot be exactly replayed from GitHub alone. `Agentic-RAG-VLM/docs/REPRODUCTION.md` distinguishes paper claims, reconstructed mechanisms, and later local VLABench demonstrations. Do not report a newly generated run as replication of the original publication without restoring the original data and protocol.

## 3. LIBERO and independent PyTorch PI0.5 efficiency

The original [OpenPI code](https://github.com/Physical-Intelligence/openpi) points to `gs://openpi-assets/checkpoints/pi05_libero`. The local PyTorch conversion is expected at `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/` with `model.safetensors` (about 6.8 GB), `config.json` and `assets/physical-intelligence/libero/norm_stats.json`. It is **not** the RoboMME or RoboDojo policy. The repository includes `openpi/examples/convert_jax_model_to_pytorch.py`; after obtaining the official JAX checkpoint and setting up the matching OpenPI environment:

```bash
cd openpi
python examples/convert_jax_model_to_pytorch.py \
  --config_name pi05_libero \
  --checkpoint_dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero" \
  --output_path "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Do not assume an unrelated Hugging Face PyTorch conversion has identical weights or normalization. First check the output configuration, assets and action-replay checks. The independent profile result `282.43 -> 54.67 ms` was measured with this local PyTorch checkpoint on 45 recorded inputs and does **not** imply that RoboMME's JAX loop meets that latency.

## 4. RoboDojo official PI0.5

Official [RoboDojo repository](https://github.com/RoboDojo-Benchmark/RoboDojo) and [dataset/checkpoints](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo). The verified model is under `ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999` at dataset revision `0a709fe7863f6c869a8d141676585a883fce4270`; inference files total **12,440,992,402 bytes**. The repository's downloader selects only required `params/`, `assets/` and metadata, checks official hashes and writes `pi05_inference_manifest.json`:

```bash
python scripts/prepare_robodojo_pi05.py --list-only
python scripts/prepare_robodojo_pi05.py
```

Default destination: `third_party/robodojo_official/XPolicyLab/policy/Pi_05/checkpoints/huggingface/`. The script requires `huggingface_hub`; install the official RoboDojo checkout and its XPolicyLab submodule first. On this machine, RoboDojo simulator assets were roughly 66 GB and were **not** uploaded. GPU inference alone does not replace the Isaac Sim/Isaac Lab environment. The existing partial `organize_table` and `classify_objects_by_language` videos are not evidence of successful Agentic gain.

## 5. Upstream code and local integration

| Checkout path | Upstream URL | Local base commit |
| --- | --- | --- |
| `third_party/robomme_benchmark` | <https://github.com/RoboMME/robomme_benchmark> | `d57969fd30f8e8318fb67c389a848b3776724470` |
| `third_party/robomme_policy_learning` | <https://github.com/RoboMME/robomme_policy_learning> | `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` |
| `third_party/robodojo_official` | <https://github.com/RoboDojo-Benchmark/RoboDojo> | `2184bf8844ea9d205382c4aefa3a694311418251` |
| RoboDojo's `XPolicyLab` submodule | Follow the official RoboDojo submodule, then checkout | `c07a09614dd44cc4a67483bcb9a82e7439d99926` |

The upstream checkouts are not vendored into this repository. The RoboDojo checkout on the source machine contains local integration changes; see [integration_patches/robodojo/README.md](integration_patches/robodojo/README.md) for application instructions and scope. The root repo includes its own OpenPI source snapshot and local modifications; it is not a guarantee of a ready-to-run Python environment. Use upstream installation guides and pin compatible CUDA/JAX/Isaac versions before replaying a benchmark. Historical StarVLA PI-v3 additionally used [the released checkpoint](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo) at revision `c119685777cf17d27940b9f36fbc7a83663361e0` (about 10.92 GB), plus Qwen3-VL-4B base; it is optional for current mainline work.

## 6. Fast validation without redownloading models

- Open `paper/CARVE-VLA/ral_draft/main.pdf` and `TECHNICAL_REPORT.md`.
- Inspect the tracked `results/` and `artifacts/` aggregate JSON/CSV; they are **saved evidence**, not newly rerunnable rollouts.
- For selected original LIBERO-PRO, RoboMME, RoboDojo, and Agentic RAG-VLM outputs omitted from GitHub, log into `Minth-Group` and download the [private Hugging Face evidence dataset](https://huggingface.co/datasets/Minth-Group/Agentic-VLA-reproduction). Its [dataset card](docs/handoff/HF_EVIDENCE_DATASET_CARD_20261008.md) lists archive contents, extraction steps, checksums, and claim boundaries. These are raw records, **not** model weights.
- Run the CPU contract tests in `tests/` only after creating a compatible Python environment. A passing test suite does not establish simulator task success.
- Before a GPU run, verify checkpoint files, normalizers, observation cameras, action representation, task/episode list and the package's upstream commit IDs. The technical report states which results are development-stage and which comparisons were fixed pairs.

# RoboDojo + StarVLA Environment Status (2026-08-31)

## Objective

Prepare a reproducible single-RTX-4090 pipeline for evaluating the CARVE agentic and efficient-inference runtime on the official RoboDojo benchmark with the released StarVLA PI-v3 policy.

## Current Readiness

- Downloaded artifacts: complete
- Conda environments: complete
- StarVLA CUDA/FlashAttention test: passed
- Official RoboDojo installation doctor: `18 passed, 1 warning, 0 failed`
- StarVLA PI-v3 checkpoint verification and evaluation dry-run: passed
- NVIDIA driver migration to `580.173.02`: passed
- Isaac Sim 5.1 post-reboot startup: passed
- Real one-environment, one-episode StarVLA run: completed
- Baseline outcome on `build_tower`, seed 0: `0/1` success
- CARVE Optimize-only outcome on the same task/layout/seed: `1/1` success

## Pinned Sources

- RoboDojo root: `third_party/robodojo_official`
- RoboDojo commit: `2184bf8844ea9d205382c4aefa3a694311418251`
- XPolicyLab / vendored StarVLA commit: `c07a09614dd44cc4a67483bcb9a82e7439d99926`
- Released policy: `StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo`
- Policy revision: `c119685777cf17d27940b9f36fbc7a83663361e0`
- Expected PI-v3 checkpoint size: `10,920,633,422` bytes
- Expected PI-v3 checkpoint SHA-256: `7144d490c40edceca31feeedb06434d795ed300b85eef0f2164efa4ed2c12d5f`

## Installed Environments

### Simulator: `RoboDojo`

- Python 3.11
- Isaac Sim `5.1.0.0`
- IsaacLab package `0.54.3`
- PyTorch `2.7.0+cu128`
- Official RoboDojo, IsaacLab, and CuRobo installation completed

### Policy: `StarVLA`

- Python `3.11.16`
- PyTorch `2.6.0+cu124`
- CUDA toolkit / `nvcc` `12.4.131`
- StarVLA `1.0.1` installed editable from the pinned source
- FlashAttention `2.7.4.post1`

The StarVLA environment passed an actual CUDA kernel test on the RTX 4090:

```text
FLASH_ATTN_CUDA_OK (2, 64, 8, 64) torch.float16 NVIDIA GeForce RTX 4090
```

FlashAttention `2.8.3.post1` was rejected because its prebuilt extension had a PyTorch C++ ABI symbol mismatch. The StarVLA-documented `2.7.4.post1` wheel imports and executes correctly.

## Downloaded Artifacts

All required artifacts completed downloading and their background processes exited normally.

1. RoboDojo official assets
   - Working cache: `third_party/robodojo_official/.cache/robodojo_assets_repo`
   - Final target: `third_party/robodojo_official/Assets`
   - Log: `third_party/robodojo_assets_download.log`
   - Materialized asset repository size: approximately 66 GB.
   - `Assets` is a symlink to the completed cache repository.
2. StarVLA RoboDojo PI-v3 checkpoint
   - Target: `third_party/robodojo_official/XPolicyLab/policy/starVLA/checkpoints/huggingface/pi_v3`
   - Log: `third_party/starvla_checkpoint_download.log`
   - Verified size: `10,920,633,422` bytes.
   - Verified SHA-256: `7144d490c40edceca31feeedb06434d795ed300b85eef0f2164efa4ed2c12d5f`.
3. Qwen3-VL-4B-Instruct base model
   - Target: `/home/admin1/models/Qwen3-VL-4B-Instruct`
   - Log: `third_party/qwen3_vl_4b_download.log`
   - This base directory is required to construct the Qwen3-VL backbone before the released RoboDojo state dictionary is loaded.
   - Both model shards and processor/tokenizer files are present; total size is approximately 8.3 GB.

## Resolved Simulator Blocker

The host previously used:

```text
NVIDIA GeForce RTX 4090, driver 595.84, 24564 MiB
```

Isaac Sim detects the RTX 4090 and loads the core extensions, but crashes before `Simulation App Startup Complete` with exit code 139 in:

```text
librtx.scenedb.plugin.so
libcarb.scenerenderer-rtx.plugin.so
libomni.hydra.rtx.plugin.so
```

Restricting Vulkan enumeration to `/usr/share/vulkan/icd.d/nvidia_icd.json` correctly hides the Intel iGPU but does not remove the crash. NVIDIA has confirmed that the R590/595 driver branch is incompatible with Isaac Sim 5.1 in this exact RTX renderer path. The validated Linux driver is `580.65.06`.

References:

- NVIDIA forum diagnosis: <https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-on-startup-with-rtx-4090-on-ubuntu-24-04-4-segfaulting-in-librtx-scenedb-plugin-so-after-iommu-was-disabled/371957>
- Isaac Sim requirements: <https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html>

The host was migrated to NVIDIA open driver `580.173.02`. After reboot, Isaac Sim reached both `app ready` and `Simulation App Startup Complete`, and the old RTX scene-database crash did not recur.

## Completed Real-Episode Smoke Test

The official StarVLA PI-v3 policy was evaluated without CARVE intervention on RoboDojo `build_tower`, layout/seed 0, with one simulator environment and one episode.

- Evaluation date: 2026-09-01
- Runtime contract: RGB observations, raw 14D dual-arm state, server-side `arx_x5` q99 normalization, 50-action prediction horizon, 16-action execution horizon
- Episode length: 1,050 control steps
- Recorded frames: 1,051 per camera at 640 x 480 and 25 FPS
- Episode outcome: failure (`success_rate=0.0`, `0/1`)
- Unstable episodes: 0
- End-to-end command wall time: 569 seconds, including policy load, Isaac startup, first-time cuRobo compilation, execution, video encoding, and shutdown
- Observed aggregate GPU-memory peak: approximately 19.83 GiB on the RTX 4090
- Post-run GPU memory returned to approximately 0.55 GiB
- System and user failed units after the run: 0

The videos show nontrivial closed-loop robot motion and manipulation rather than an empty or mocked rollout. The policy formed a partial tower-like arrangement but did not satisfy the benchmark success predicate by the episode horizon. This result establishes an operational official baseline; it is not evidence for CARVE effectiveness yet.

Primary evidence:

- Run log: `artifacts/robodojo/starvla_pi_v3_build_tower_smoke_20260901/run_retry2.log`
- Result JSON: `third_party/robodojo_official/eval_result/RoboDojo/build_tower/starVLA/arx_x5/0_ckpt_name=hf_qwenpi_v3,action_type=joint/2026-09-01_13-24-16/_result.json`
- Head-camera video: `third_party/robodojo_official/eval_result/RoboDojo/build_tower/starVLA/arx_x5/0_ckpt_name=hf_qwenpi_v3,action_type=joint/2026-09-01_13-24-16/episode_0000000_cam_head_fail.mp4`
- Contact sheets: `artifacts/robodojo/starvla_pi_v3_build_tower_smoke_20260901/`

Two setup omissions were found and corrected before the successful run:

1. The StarVLA environment needed the XPolicyLab transport dependency `msgpack-numpy==0.4.8`.
2. The downloaded assets provide `curobo_tmp.yml`; RoboDojo's own `utils/update_embodiment_config_path.py` must generate `curobo.yml` with the local absolute asset path.

The StarVLA evaluation entry point now checks both conditions before loading the 10 GB checkpoint.

## Verification Gates

All three gates have passed. Gate 3 completed as an execution-path smoke test even though the single baseline episode did not solve the task.

### 1. Official installation doctor

```bash
cd /home/admin1/ct/CARVE-VLA/third_party/robodojo_official
bash scripts/robodojo.sh doctor \
  --policy-dir XPolicyLab/policy/starVLA \
  --task build_tower \
  --sim-env RoboDojo \
  --policy-env StarVLA
```

### 2. Checkpoint and command dry run

```bash
cd /home/admin1/ct/CARVE-VLA/third_party/robodojo_official/XPolicyLab/policy/starVLA
export STARVLA_BASE_VLM=/home/admin1/models/Qwen3-VL-4B-Instruct
export STARVLA_HF_VERIFY_ONLY=1
export STARVLA_HF_DRY_RUN=1
bash scripts/eval_hf_robodojo.sh \
  pi_v3 build_tower 0 0 0 \
  /home/admin1/miniconda3/envs/StarVLA \
  /home/admin1/miniconda3/envs/RoboDojo 1
```

### 3. One-environment, one-episode simulator smoke test

```bash
cd /home/admin1/ct/CARVE-VLA/third_party/robodojo_official/XPolicyLab/policy/starVLA
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export STARVLA_BASE_VLM=/home/admin1/models/Qwen3-VL-4B-Instruct
export STARVLA_HF_VERIFY_ONLY=1
export STARVLA_ROBODOJO_NUM_ENVS=1
bash scripts/eval_hf_robodojo.sh \
  pi_v3 build_tower 0 0 0 \
  /home/admin1/miniconda3/envs/StarVLA \
  /home/admin1/miniconda3/envs/RoboDojo 1
```

The simulator completed an episode and produced an evaluation record plus three camera videos. The official StarVLA + RoboDojo baseline pipeline is operational.

## CARVE Optimize Runtime Paired Smoke

On 2026-09-01, the new `StarVlaAdapter` and CARVE Optimize Runtime completed a
real paired `C1` run on the same `build_tower`, layout 0, seed 0 condition. The
profile used six DDIM steps and a 32-action execution horizon. RoboDojo's
private evaluator reported success at control step 718; the trace contains 23
real VLA calls with 304.92/348.31 ms P50/P95 latency and no dropped controls.

The paired official baseline failed after 1,050 steps. This is one positive
paired observation, not a success-rate estimate. Full evidence and the scope
boundary are recorded in `docs/status/ROBODOJO_CARVE_PAIRED_RESULTS.md`.

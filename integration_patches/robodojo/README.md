# RoboDojo Local Integration Snapshot

This directory preserves the source-machine changes to the **official upstream checkouts** without vendoring their large assets, checkpoints or whole repositories. These files are for reconstructing the historical RoboDojo integration; they are not a claim that a fresh clone has already passed the same simulation tests.

| Item | Source base | Contents |
| --- | --- | --- |
| `robodojo_root.patch` | `RoboDojo-Benchmark/RoboDojo` at `2184bf8844ea9d205382c4aefa3a694311418251` | Evaluator and launch-script modifications; excludes submodule pointer changes |
| `xpolicylab.patch` | `XPolicyLab` submodule at `c07a09614dd44cc4a67483bcb9a82e7439d99926` | Transport and historical StarVLA runtime changes |
| `untracked/XPolicyLab/` | additive files absent from those upstream commits | `AgenticPi05` policy wrapper, reset protocol, historical staged VLM code |

After reading the [model/environment guide](../../MODEL_AND_ENVIRONMENT_SETUP.md), on a clean clone of the upstream repositories:

```bash
git clone --recursive https://github.com/RoboDojo-Benchmark/RoboDojo.git third_party/robodojo_official
git -C third_party/robodojo_official checkout 2184bf8844ea9d205382c4aefa3a694311418251
git -C third_party/robodojo_official submodule update --init --recursive
git -C third_party/robodojo_official/XPolicyLab checkout c07a09614dd44cc4a67483bcb9a82e7439d99926
git -C third_party/robodojo_official/third_party/curobo checkout 895c6517243f8cb091c73c018c8167192d39599a

patch_root="$(pwd)/integration_patches/robodojo"
git -C third_party/robodojo_official apply --check "$patch_root/robodojo_root.patch"
git -C third_party/robodojo_official apply "$patch_root/robodojo_root.patch"
git -C third_party/robodojo_official/XPolicyLab apply --check "$patch_root/xpolicylab.patch"
git -C third_party/robodojo_official/XPolicyLab apply "$patch_root/xpolicylab.patch"
rsync -a integration_patches/robodojo/untracked/XPolicyLab/ third_party/robodojo_official/XPolicyLab/
```

Run this block from the **Agentic-VLA project root**. Do not use `--3way` or silently force a patch against a different upstream commit.

The original RoboDojo root checkout recorded local changes to the XPolicyLab and CuRobo submodule pointers; they are represented here by the explicit checkout commands, rather than as a possibly incompatible parent submodule diff. The official simulator `Assets` (about 66 GB), Python environments and large model files are deliberately absent. Re-run the official installation doctor and a single native episode before evaluating any Agentic change. Historical StarVLA evidence and current official PI0.5 results use different models.

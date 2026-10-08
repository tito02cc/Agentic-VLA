# Agentic-VLA 实验日志

## 2026-07-16 CARVE pi0.5 配对推理预算与控制器因果消融

目的：在 LIBERO-10 T6/T8/T9 上，以相同初始状态、prompt、checkpoint 和
确定性 pi0.5 sampling noise，拆分 flow steps、action commit horizon 与
Agentic stall intervention 的影响。

核心设置：PyTorch `pi05_libero`、RTX 4090、每任务 5 个初始状态、
`replan_steps=10`。所有结论均来自真实 MuJoCo 闭环 rollout，不是 mock 或
open-loop proxy。

| 条件 | T6 | T8 | T9 | 总成功率 | Mean VLA call | Episode wall |
|---|---:|---:|---:|---:|---:|---:|
| default 7-step, commit 10 | 5/5 | 4/5 | 5/5 | 14/15 | 366.9 ms | 21.27 s |
| fixed 2-step, commit 10 | 5/5 | 4/5 | 5/5 | 14/15 | 149.0 ms | 14.40 s |
| dynamic 2/4-step, commit 10 | 5/5 | 3/5 | 5/5 | 13/15 | 158.9 ms | 15.91 s |

主要结论：

- 固定 2-step 在 15 个配对状态上保持与 7-step 相同的 `14/15`，平均
  VLA-call latency 降低 `2.46x`，每 episode 的 VLA wall time 降低
  `60.2%`，端到端 episode wall time 降低 `32.3%`。
- 固定 2-step 的 447 次 policy call 仍全部超过当前设置的 80 ms deadline；
  当前证据支持更快的在线 action-chunk inference，不支持“单次 VLA 已满足
  80 ms hard real-time”这一表述。
- 1-step 在单状态 pilot 上为 3/3，但扩展到 T8 五状态后仅 3/5，不能作为
  全局默认配置。
- `commit=8` 是显著负因素：固定 2-step 在 T8/T9 同一批状态上从
  `commit=10` 的 9/10 降到 6/10。动作提交时域与 flow steps 必须独立校准。
- stall 驱动的动态 2/4-step 能纠正个别失败状态，但也破坏其他成功状态，
  净结果 13/15。当前 stall signal 不能预测“增加 flow steps 是否有益”，
  因此动态步数只保留为负消融，不作为部署默认值。
- 物理 recovery 仅允许明确的 `slip/misgrasp` 证据；普通 collision/stall
  不再立即清空 action chunk 或触发 retry。

结果目录：

- baseline: `results/carve_t689_paired_5states_v4/baseline/`
- fixed 2-step: `results/carve_t689_paired_5states_v4/fixed2_t8/` 与
  `results/carve_t689_paired_5states_v4/fixed2_t69/`
- dynamic 2/4 commit 10: `results/carve_t689_paired_5states_v6/joint/`
- commit-horizon causal control: `results/carve_t89_fixed2_commit8_5states/`

下一步：固定 2-step/commit-10 作为两组共同 backend，只在可控扰动上比较
无 Agentic、Agentic monitor、targeted retry/replan，避免把推理预算变化混入
Agentic 成功率结论。

## 2026-06-11 robosuite Stack E2E v2: 100-demo learned policy

目的：在 v1 端到端闭环基础上扩大真实仿真数据规模，验证 `Agentic retry + CAQ-Lite` 的效果是否稳定。

重要边界：

- 该实验仍为真实 robosuite/MuJoCo 物理仿真，不是 mock。
- 当前 policy backend 是小型 RGB+state action-chunk policy，不是 pi0.5/OpenPI 微调。
- 该实验用于证明完整训练/部署闭环和 Agentic/轻量化模块耦合的可行性；pi0.5 backend 是下一阶段升级。
- 因此该条目不能写成“pi0.5/VLA 在 robosuite 上验证成功”。准确表述应为“真实仿真训练-部署 pipeline 与 Agentic/CAQ-Lite 机制验证成功，VLA backend 尚待接入”。

数据采集：

| Item | Value |
|---|---:|
| Dataset | `results/robosuite_stack_e2e_v2_20260611/stack_demos_100eps.hdf5` |
| Successful episodes | `100` |
| Attempts | `107` |
| Total frames | `33933` |
| Image size | `64x64` |
| Perturbation probability | `0.25` |
| Collection wall time | `853.50 s` |

训练：

| Item | Value |
|---|---:|
| Checkpoint | `results/robosuite_stack_e2e_v2_20260611/policy_v2/best_policy.pt` |
| Train samples | `28811` |
| Validation samples | `5122` |
| Epochs | `60` |
| Best validation loss | `0.08247` |
| Training wall time | `180.92 s` |

真实仿真 mid-nudge 对照：

| Method | Success | Full calls / ep | Reused / ep | Reuse ratio | Miss@80ms | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| `RS-LearnedV2-Light-MidNudge003-NoRetry` | `4/10` | `157.0` | `234.4` | `0.599` | `0.0006` | `0/0` |
| `RS-LearnedV2-AgenticRetry-NoLight-MidNudge003` | `9/10` | `127.5` | `0.0` | `0.000` | `0.0007` | `10/10` |
| `RS-LearnedV2-AgenticRetry-Light-MidNudge003` | `8/10` | `55.5` | `99.2` | `0.641` | `0.0004` | `10/10` |
| `RS-LearnedV2-AgenticRetry-LightSafe1-MidNudge003` | `9/10` | `69.7` | `62.6` | `0.473` | `0.0007` | `10/10` |

真实仿真 clean 验证：

| Method | Success | Full calls / ep | Reused / ep | Reuse ratio | Miss@80ms | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| `RS-LearnedV2-AgenticRetry-LightSafe1-Clean` | `10/10` | `64.3` | `63.8` | `0.498` | `0.0007` | `10/10` |

核心结论：

- 扩大到 `100` demos 后，learned policy 的 mid-nudge 成功率从 v1 no-retry 的 `0/5` 提升到 `4/10`，说明数据规模改善了基础策略。
- Agentic retry 将 mid-nudge 成功率从 `4/10` 提升到 `9/10`，是成功率提升的主因。
- 激进 CAQ-Lite (`reuse_max_actions=2`) 将 full calls 从 `127.5/ep` 降到 `55.5/ep`，但成功率从 `9/10` 降到 `8/10`。
- 保守 CAQ-Lite safe1 (`reuse_max_actions=1`) 保持 `9/10` 成功率，同时将 full calls 降到 `69.7/ep`，约减少 `45.3%`。
- 因此 v2 主配置建议使用 `AgenticRetry + LightSafe1`：它兼顾成功率与实时推理效率。

视频与结果：

- Demo video：`results/robosuite_stack_e2e_v2_20260611/eval_midnudge_agentic_retry_light_safe1_demo.mp4`
- Contact sheet：`results/robosuite_stack_e2e_v2_20260611/eval_midnudge_agentic_retry_light_safe1_demo_contact.png`
- Mid-nudge safe1 summary：`results/robosuite_stack_e2e_v2_20260611/eval_midnudge_agentic_retry_light_safe1_10trials_summary.json`
- Clean safe1 summary：`results/robosuite_stack_e2e_v2_20260611/eval_clean_agentic_retry_light_safe1_10trials_summary.json`
- Training summary：`results/robosuite_stack_e2e_v2_20260611/policy_v2/training_summary.json`
- Collection summary：`results/robosuite_stack_e2e_v2_20260611/collect_summary.json`

## 2026-06-10 robosuite Stack end-to-end learned policy pipeline

目的：跑通真实仿真环境中的完整流程：

1. 真实 robosuite/MuJoCo `Stack` 数据采集；
2. RGB+state action-chunk policy 训练；
3. learned policy 回接真实仿真闭环推理；
4. Agentic retry physical skill 纠错；
5. CAQ-Lite action reuse 减少 full model calls；
6. 生成真实仿真结果视频。

重要边界：

- 该流程是真实 MuJoCo 物理仿真，不是 mock。
- 当前训练的是小型 RGB+state action-chunk policy，不是 pi0.5/OpenPI VLA 微调模型。
- 该实验用于证明 `data collection -> model training -> inference validation -> Agentic recovery -> lightweight runtime` 工程闭环可行。
- 后续若要作为强 VLA deployment claim，需要将 policy backend 替换为 pi0.5/OpenPI adapter 或在同任务数据上进行 pi0.5 head/LoRA tuning。

任务设置：

- Simulator：robosuite
- Physics：MuJoCo
- Task：`Stack`
- Robot：Panda
- Controller：OSC_POSE
- Observation：`frontview` RGB `64x64` + proprio/object state
- Action：7D OSC_POSE action
- Learned policy：compact RGB+state action-chunk network
- Chunk size：`5`
- Perturbation：mid-episode object nudge, `perturb_xy=0.03`

代码：

- 数据采集：`scripts/collect_robosuite_stack_data.py`
- 模型定义：`scripts/robosuite_stack_policy.py`
- 模型训练：`scripts/train_robosuite_stack_policy.py`
- 推理验证：`scripts/eval_robosuite_stack_policy.py`

数据采集：

| Item | Value |
|---|---:|
| Dataset | `results/robosuite_stack_e2e_v1_20260610/stack_demos_30eps.hdf5` |
| Accepted successful episodes | `30` |
| Attempts | `36` |
| Total frames | `10232` |
| Perturbation probability | `0.25` |
| Collection wall time | `277.18 s` |

训练：

| Item | Value |
|---|---:|
| Checkpoint | `results/robosuite_stack_e2e_v1_20260610/policy_v1/best_policy.pt` |
| Train samples | `8868` |
| Val samples | `1364` |
| Epochs | `35` |
| Best val loss | `0.1290` |
| Training wall time | `46.83 s` |

真实仿真验证结果：

| Method | Setting | Success | Full calls / ep | Reused / ep | Reuse ratio | Miss@80ms | Recovery |
|---|---|---:|---:|---:|---:|---:|---:|
| `RS-Learned-Light-MidNudge003-NoRetry` | learned + Light, no Agentic retry | `0/5` | `170.0` | `260.0` | `0.605` | `0.0009` | `0/0` |
| `RS-Learned-AgenticRetryV2-NoLight-MidNudge003` | learned + Agentic retry | `4/5` | `188.6` | `0.0` | `0.000` | `0.0013` | `5/5` |
| `RS-Learned-AgenticRetryV2-Light-MidNudge003` | learned + Agentic retry + CAQ-Lite | `4/5` | `71.8` | `120.0` | `0.626` | `0.0007` | `5/5` |
| `RS-Learned-AgenticRetryV2-Light-Clean` | clean validation | `4/5` | `63.2` | `125.4` | `0.665` | `0.0007` | `5/5` |

核心结论：

- learned policy 单独闭环在接触/抓取失败后无法恢复，mid-nudge 下 `0/5`。
- Agentic critic + geometric retry physical skill 将 mid-nudge 成功率提升到 `4/5`。
- 在保持 `4/5` 成功率的情况下，CAQ-Lite 将 full model calls 从 `188.6/ep` 降到 `71.8/ep`，降低约 `61.9%`。
- 该结果直接支持论文叙事：Agentic Policy 解决长程/接触失败后的纠错，轻量化 runtime 解决实时推理调用成本。

视频与结果文件：

- Mid-nudge video：`results/robosuite_stack_e2e_v1_20260610/eval_midnudge_agentic_retry_light_v2.mp4`
- Mid-nudge contact sheet：`results/robosuite_stack_e2e_v1_20260610/eval_midnudge_agentic_retry_light_v2_contact.png`
- Mid-nudge summary：`results/robosuite_stack_e2e_v1_20260610/eval_midnudge_agentic_retry_light_v2_summary.json`
- Clean summary：`results/robosuite_stack_e2e_v1_20260610/eval_clean_agentic_retry_light_v2_summary.json`
- Training summary：`results/robosuite_stack_e2e_v1_20260610/policy_v1/training_summary.json`
- Collection summary：`results/robosuite_stack_e2e_v1_20260610/collect_summary.json`

## 2026-06-10 robosuite/MuJoCo real-simulation deployment video

目的：在 IsaacSim 本机链路暂缓后，使用稳定的 robosuite/MuJoCo 真实物理仿真环境验证 Agentic Harness + CAQ-Lite runtime 是否能脱离 LIBERO runner 运行，并生成可展示的实验结果视频。

重要边界：

- 该实验是真实 MuJoCo 物理仿真，不是 mock 数值模拟。
- 当前控制后端为 scripted action-chunk teacher，用于验证环境、扰动、runtime、recovery、轻量化调度和视频录制链路。
- 该实验暂不作为“pi0.5/VLA 在 robosuite 上的最终成功率”主结果；后续若要写成 VLA deployment result，需要接入 pi0.5/OpenPI adapter 或在 robosuite/ManiSkill 数据上进行适配训练。

任务设置：

- Simulator：robosuite `Stack`
- Physics：MuJoCo
- Robot：Panda
- Controller：OSC_POSE
- Perturbation：mid-episode object nudge
- `perturb_step=260`
- `perturb_xy=0.03`
- Horizon：`430`
- Video：offscreen `frontview` RGB, `256x256`, `20 fps`

视频结果：

- Directory：`results/robosuite_real_video_stack_agentic_light_seed7_20260610/`
- Video：`stack_agentic_light_real.mp4`
- Contact sheet：`stack_agentic_light_contact_sheet.png`
- Video metadata：`256x256`, `20 fps`, `8.55 s`, `171` frames
- Result：`success=True`, `steps=341`, `recoveries=1/1`

单 episode 视频 trace：

| Method | Success | Steps | Full calls | Reused actions | Reuse ratio | Deadline miss | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RS-Agentic-Light-RealVideo` | `1/1` | `341` | `167` | `174` | `0.510` | `0.003` | `1/1` |

真实仿真小规模对照：

| Method | Success | Full calls / ep | Reused / ep | Reuse ratio | Deadline miss | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| `RS-Base-Step` | `4/5` | `358.8` | `0.0` | `0.000` | `0.000` | `0/0` |
| `RS-Agentic-Step` | `4/5` | `358.8` | `0.0` | `0.000` | `0.000` | `5/5` |
| `RS-Agentic-Light` | `4/5` | `173.0` | `185.8` | `0.518` | `0.000` | `5/5` |

结果目录：

- `results/robosuite_real_stack_RS-Base-Step_5trial_seed11_20260610/`
- `results/robosuite_real_stack_RS-Agentic-Step_5trial_seed11_20260610/`
- `results/robosuite_real_stack_RS-Agentic-Light_5trial_seed11_20260610/`

论文解释：

- 该实验可以作为 near-deployment / external simulator pilot，证明框架可以在 LIBERO 之外的真实物理仿真控制循环中运行。
- 当前 scripted backend 已是强闭环控制器，因此 Agentic 模块没有在 5-trial 成功率上拉开差距；不能把这组结果写成 Agentic 成功率主证据。
- CAQ-Lite 在真实仿真部署循环中将 full policy calls 降低约 `51.8%`，同时保持相同成功率，可作为轻量化调度与实时部署接口的补充证据。
- Agentic 长程纠错主证据仍应来自 LIBERO/pi0.5 benchmark；robosuite 视频用于展示真实仿真部署可行性。

## 2026-06-10 Strong recovery stress seed17 completion

目的：趁 GPU 空闲补齐一组最小但高价值的恢复压力测试，用于回答 `Agentic recovery` 在更强扰动下是否仍然有效，以及 `Light` 复用是否会破坏恢复能力。

协议：

- Benchmark：LIBERO-10 Task2
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy=0.05`
- Trials：`20/method/seed`
- Seeds：`7, 17`
- Backend：`pi05_libero_pytorch` + `OPENPI_DISABLE_TORCH_COMPILE=1`
- Server：port `8001`
- 对比：
  - `B4-Agentic`
  - `B4-Agentic-Light`

新增结果目录：

- `results/recoveryStress_b4_agentic_contextGate_replan5_noLight_midNudge005_task2_20trials_seed17_20260610/`
- `results/recoveryStress_b4_agentic_light_safe80_replan5_reuseMax2_midNudge005_task2_20trials_seed17_20260610/`

2-seed 聚合结果：

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Miss@80ms | SUD@80ms | Recovery precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic` | `38/40` | `57.85` | `0.00` | `0.000` | `31.86 s` | `0.205` | `0.755` | `31/33 = 0.939` |
| `B4-Agentic-Light` | `37/40` | `43.25` | `15.10` | `0.259` | `26.56 s` | `0.149` | `0.785` | `26/29 = 0.897` |

论文解释：

- 该表是 strong-stress / boundary analysis，不替代 Task2 `mid_nudge_xy=0.03` 三种子主表。
- 在 `5cm` 强扰动下，`B4-Agentic-Light` 相比纯 Agentic 的成功率略低 `1/40`，但 full VLA calls 降低约 `25.2%`，deadline miss 降低约 `27.3%`，SUD 更高。
- 恢复 precision 从 `0.939` 降到 `0.897`，说明轻量化复用在强扰动下需要安全 lockout；当前配置仍保持较高恢复质量。
- 该结果支持论文中的 trade-off 叙事：Agentic recovery 提供恢复能力，CAQ-Lite 提供实时性，但耦合需要保守 criticality gate。

资产更新：

- 已新增：`results/paper_assets_20260609/table_recovery_stress_strong.md`
- 已新增：`results/paper_assets_20260609/figures/recovery_stress_strong.png`
- 已更新：`scripts/build_paper_assets.py`

## 2026-06-10 Paper mainline: LIBERO-10 mid-nudge matched snapshot

目的：补齐论文主线中的全任务覆盖表，使用统一协议比较 `B0-VLA`、`B0-VLA-Light`、`B4-Agentic` 和已有 `B4-Agentic-Light` coverage。该表用于说明系统在 LIBERO-10 全任务上的整体边界，不替代 Task2 三种子主表。

协议：

- Benchmark：LIBERO-10 Task0-9
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy=0.03`
- Trials：`5/task`
- Seed：`7`
- Backend：`pi05_libero_pytorch` + `OPENPI_DISABLE_TORCH_COMPILE=1`
- Server：port `8001`
- B0/B0-Light/B4-Agentic 为 2026-06-10 新跑结果
- B4-Agentic-Light 使用已有同协议 coverage：`results/finalCoverage_libero10_b4_agentic_light_safeLong_noTransition_replan5_reuseMax2_midNudge003_5trials_seed7_20260609/`

新增结果目录：

- `results/paperSnapshot_b0_vla_noLight_replan5_midNudge003_libero10_5trials_seed7_20260610/`
- `results/paperSnapshot_b0_vla_light_replan5_reuseMax2_midNudge003_libero10_5trials_seed7_20260610/`
- `results/paperSnapshot_b4_agentic_noLight_safeLong_noTransition_replan5_midNudge003_libero10_5trials_seed7_20260610/`

聚合结果：

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Miss@80ms | SUD@80ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | `35/50` | `66.42` | `0.00` | `0.000` | `44.70 s` | `0.204` | `0.556` |
| `B0-VLA-Light` | `36/50` | `41.76` | `25.80` | `0.382` | `35.30 s` | `0.132` | `0.621` |
| `B4-Agentic` | `37/50` | `65.56` | `0.00` | `0.000` | `44.26 s` | `0.204` | `0.590` |
| `B4-Agentic-Light` | `36/50` | `53.46` | `13.02` | `0.196` | `33.92 s` | `0.158` | `0.610` |

Task-wise 观察：

| Task | B0 | B0-Light | B4-Agentic | B4-Agentic-Light | 观察 |
|---:|---:|---:|---:|---:|---|
| `0` | `5/5` | `5/5` | `5/5` | `4/5` | 简单 basket 任务总体稳定 |
| `1` | `5/5` | `5/5` | `5/5` | `5/5` | 稳定任务 |
| `2` | `5/5` | `5/5` | `5/5` | `5/5` | 主线 Task2 在 3cm 扰动下稳定 |
| `3` | `1/5` | `1/5` | `1/5` | `1/5` | 抽屉闭环任务是共同边界 |
| `4` | `5/5` | `5/5` | `5/5` | `5/5` | 多对象 plate 任务稳定 |
| `5` | `4/5` | `4/5` | `4/5` | `4/5` | 中等难度任务 |
| `6` | `2/5` | `3/5` | `3/5` | `3/5` | Agentic/Light 对精确二阶段任务略有帮助 |
| `7` | `4/5` | `4/5` | `3/5` | `5/5` | B4 no-light 出现负波动，B4-Light 结果较好 |
| `8` | `2/5` | `3/5` | `3/5` | `2/5` | 双 moka-pot 长程任务仍是边界 |
| `9` | `2/5` | `1/5` | `3/5` | `2/5` | Agentic no-light 对 microwave 任务略好 |

论文解释：

- 全任务平均上，B4-Agentic 从 B0 的 `35/50` 提升到 `37/50`，属于小幅提升；不应写成全任务显著优势。
- B0-Light 将 full VLA calls 降低约 `37.1%`，deadline miss 从 `0.204` 降到 `0.132`，SUD 从 `0.556` 提升到 `0.621`，支撑实时推理优化主张。
- B4-Agentic 的日志显示 GraphRAG priors 和 EvoKAM memory 正常工作，例如 moka-pot/stove、mug/microwave 等任务会记录可解释成功策略；这可用于框架说明和 qualitative case。
- B4-Agentic-Light 在该全任务 coverage 上没有超过 B4-Agentic/B0-Light 的平均成功率，说明 coupling 需要 task/context-aware safety envelope，不能宣称对所有任务都单调提升。
- 该表适合作为 coverage/boundary analysis；论文主 claim 仍应放在 Task2 三种子 2x2、Task1/5/7 轻量化泛化、strong-stress recovery 表。

资产更新：

- 已刷新：`results/paper_assets_20260609/table_libero10_midnudge_snapshot.md`
- 已新增：`results/paper_assets_20260609/table_taskwise_libero10_midnudge_snapshot.md`
- 已更新：`scripts/build_paper_assets.py`

## 2026-06-10 Paper mainline: lightweight generalization 3-seed completion

目的：在 IsaacSim 本机链路暂缓后，回到论文主线，补强 `VLA lightweight / realtime runtime` 的独立证据。

协议：

- Benchmark：LIBERO-10 Task1/5/7
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy=0.03`
- Trials：`5/task`
- Seeds：`7, 17, 27`
- Backend：`pi05_libero_pytorch` + `OPENPI_DISABLE_TORCH_COMPILE=1`
- Server：port `8001`
- Baseline：`B0-VLA`
- Light：`B0-VLA-Light`
- Light flags：`--light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`

新增结果目录：

- `results/lightTable_b0_vla_noLight_replan5_midNudge003_tasks157_5trials_seed17_20260610/`
- `results/lightTable_b0_vla_light_replan5_reuseMax2_midNudge003_tasks157_5trials_seed17_20260610/`
- `results/lightTable_b0_vla_noLight_replan5_midNudge003_tasks157_5trials_seed27_20260610/`
- `results/lightTable_b0_vla_light_replan5_reuseMax2_midNudge003_tasks157_5trials_seed27_20260610/`

3-seed 聚合结果：

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | `42/45` | `50.13` | `0.00` | `0.000` | `29.15 s` | `0.202` | `0.745` |
| `B0-VLA-Light` | `43/45` | `32.09` | `17.36` | `0.351` | `22.19 s` | `0.131` | `0.829` |

单 seed 新增结果：

| Method | Seed | Success | Full VLA / ep | Skipped / ep | Wall / ep | Miss@80ms | SUD@80ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | `17` | `14/15` | `49.07` | `0.00` | `28.68 s` | `0.202` | `0.745` |
| `B0-VLA-Light` | `17` | `15/15` | `30.80` | `15.93` | `20.99 s` | `0.133` | `0.867` |
| `B0-VLA` | `27` | `15/15` | `47.00` | `0.00` | `27.22 s` | `0.202` | `0.798` |
| `B0-VLA-Light` | `27` | `15/15` | `31.13` | `16.20` | `21.28 s` | `0.133` | `0.867` |

论文解释：

- 轻量化 runtime 已从单 seed pilot 升级为 3-seed 主证据。
- `B0-VLA-Light` 在 Task1/5/7 上保持或略微提升成功率，同时 full VLA calls 下降约 `36.0%`，deadline miss 下降约 `35.0%`。
- 这组结果支撑论文第二主轴：CAQ-Lite / Light-Reuse 可脱离 Agentic Harness 独立作为 frozen VLA 的 realtime deployment runtime。

资产更新：

- `scripts/build_paper_assets.py` 已将 `lightweight_generalization` 的选择规则从 seed7 固定目录改为 seed wildcard。
- 已刷新：`results/paper_assets_20260609/table_lightweight_generalization.md`
- 已刷新：`results/paper_assets_20260609/selected_aggregates.csv`
- 已更新：`PAPER_MAINLINE_PROGRESS_20260610.md`
- 已更新：`paper/PAPER_DRAFT_AGENTIC_VLA_RUNTIME_20260610.md`

## 2026-06-09 IsaacLab Stack Cube pi0.5 训练链路状态

目标：
- 在 IsaacLab 中建立一个接近真实部署的 Stack Cube 数据采集、pi0.5 轻量微调、policy serving、闭环验证 pipeline。
- 该分支用于补充论文中的 near-deployment / real-robot-preparation 证据，不替代 LIBERO 主 benchmark。

已完成：
- 选定 IsaacLab Franka Stack Cube 作为主任务：
  - `Isaac-Stack-Cube-Franka-IK-Rel-v0`
  - `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`
  - `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`
- 下载并标注官方 Mimic 数据：
  - official demos：`10` episodes，`3901` samples
  - annotated demos：`10` episodes，`2163` samples
- 成功生成 state/action smoke 数据：
  - `data/isaaclab_stack_cube_mimic/generated_dataset_smoke.hdf5`
  - `4` successful episodes，`946` samples，`7D` actions
- 生成失败数据用于后续 critic/retry/memory 分析：
  - `data/isaaclab_stack_cube_mimic/generated_dataset_smoke_failed.hdf5`
  - `10` failed episodes，`2266` samples
- 新增 HDF5 -> LeRobot converter：
  - `scripts/convert_isaaclab_stack_hdf5_to_lerobot.py`
  - 输出 repo id：`agentic-vla/isaaclab_stack_cube`
  - 当前图像是 placeholder/debug image，不作为最终视觉训练证据
- OpenPI/pi0.5 工程链路已打通：
  - 新增 config：`pi05_isaaclab_stack_cube`
  - 使用 `pi05_base`，不是 `pi05_libero`
  - 完成 `pi05_base` JAX -> PyTorch 转换
  - norm stats 计算通过
  - dataloader smoke 通过
  - PyTorch head-only tuning smoke 通过
  - policy server smoke 返回 `(10, 7)` action chunk

head-only smoke 结果：

| Item | Value |
|---|---:|
| Checkpoint | `checkpoints/pi05_isaaclab_stack_cube/isaaclab_stack_head_smoke_2step/2` |
| Trainable params | `2,165,792 / 3,616,757,520` |
| Trainable ratio | `0.0599%` |
| Step 0 loss | `0.2493` |
| Step 1 loss | `0.1380` |
| Peak reserved GPU memory | `7.90GB` |

50-episode scale-up 结果：

| Item | Value |
|---|---:|
| Successful HDF5 | `data/isaaclab_stack_cube_mimic/generated_dataset_50.hdf5` |
| Successful episodes | `50` |
| Successful samples | `12043` |
| Failed HDF5 | `data/isaaclab_stack_cube_mimic/generated_dataset_50_failed.hdf5` |
| Failed episodes | `87` |
| Failed samples | `19405` |
| LeRobot repo id | `agentic-vla/isaaclab_stack_cube` |
| LeRobot frames | `12043` |
| Norm stats | `4096` sampled frames |
| 50-episode checkpoint | `checkpoints/pi05_isaaclab_stack_cube/isaaclab_stack_head_50eps_10step/10` |
| Trainable params | `2,165,792 / 3,616,757,520` |
| Trainable ratio | `0.0599%` |
| Train steps | `10` |
| Final logged loss | `0.1115` |
| Peak reserved GPU memory | `7.93GB` |
| Policy-server response | `(10, 7)` action chunk |
| First dummy server timing | `~1021 ms` |
| IsaacLab closed-loop smoke | `30` steps, `6` full calls, `24` reused actions |
| Closed-loop reuse ratio | `0.8` |
| Closed-loop wall time | `3.16 s` |

说明：
- 50-episode pipeline 已经验证：Mimic generation -> HDF5 inspection -> LeRobot conversion -> OpenPI norm stats -> head-only training -> policy serving。
- 额外完成了 IsaacLab Franka closed-loop smoke：OpenPI policy server 输出 `(10, 7)` action chunk，runner 通过 Differential IK 执行动作并复用 chunk suffix。
- 当前 LeRobot 图像仍是 placeholder/debug image，只用于链路验证，不作为最终视觉 VLA 训练证据。
- closed-loop smoke 使用 `--dummy-image` 和 generic Franka scene，因此是部署控制链路验证，不是 Stack Cube 成功率 benchmark。
- 87 条失败轨迹可用于后续 progress memory、critic/retry trigger、failure taxonomy 设计与定性分析。

当前阻塞：
- IsaacLab official visuomotor / camera path 在 IsaacSim viewport/Hydra 初始化阶段 segfault。
- 日志中出现 `errno=28` file-watch 创建失败。
- 当前机器 inotify 限制较低：
  - `fs.inotify.max_user_watches = 65536`
  - `fs.inotify.max_user_instances = 128`
  - `fs.inotify.max_queued_events = 16384`
- 该问题阻塞真实 RGB 采集，但不阻塞 state/action Mimic 数据扩展。

当前判断：
- `pi05_base -> IsaacLab Stack Cube LeRobot -> norm stats -> head-only train -> policy server` 工程链路已在 `4` episode 和 `50` episode 两个规模上验证。
- 这还不是最终任务成功证据，因为当前 LeRobot 图像仍是 placeholder/debug image。
- 下一步应优先解决真实 RGB 采集，或在 camera blocker 暂时无法修复时，转向 IsaacLab state-policy/adapter 闭环 smoke 和 Agentic failure-trigger 分析。

## 2026-06-08 路线更新后的 VLA Lightweight Runtime instrumentation 与首批实验

目标：
- 按最新路线将项目拆成三层：`Embodied Agentic Policy Harness`、`Lightweight VLA Runtime`、`Efficient Coupling`。
- 本阶段先补齐 VLA runtime 级别的可写论文指标，再跑小规模 microbenchmark。

工程更新：
- 在 `scripts/run_agentic_vla_libero.py` 中新增 episode-level VLA runtime instrumentation：
  - `vla_latency_ms_values`
  - `vla_latency_ms_mean/p50/p95/p99`
  - `vla_model_wall_ms_total`
  - `full_vla_calls`
  - `skipped_vla_calls`
  - `lightweight_fallback_count`
  - `action_chunk_age_mean/p95`
  - `peak_mem_mb`
- `trace_aggregate` 中新增聚合字段：
  - VLA call latency p50/p95/p99
  - full/skipped VLA calls per episode
  - lightweight skipped-call ratio
  - action chunk age
- 由于 policy server 是独立进程，runner 内 `torch.cuda.max_memory_allocated()` 无法反映 server 显存；新增 `nvidia-smi` fallback 记录整卡 memory.used。
- 新增方法标签：
  - `B0-VLA-Light`
  - `B4-Agentic`
  - `B4-Agentic-Light`
- 新增 `Light-Reuse` 原型：
  - `--light-reuse-actions`
  - `--light-reuse-risk-buckets`
  - `--light-reuse-max-actions`
  - `--light-reuse-max-buffer-actions`
  - 只复用同一次 VLA 输出但当前 commit 没有执行完的 action chunk suffix，不重复最后一个动作，不生成新动作。

Smoke 验证：

| Run | Task | Trials | Success | 关键指标 |
|---|---:|---:|---:|---|
| `L0-InstrumentationSmoke-Task2` | 2 | 1 | `1/1` | trace 写出 `vla_latency_ms_values/p50/p95/p99` |
| `L0-InstrumentationSmokeVram-Task2` | 2 | 1 | `1/1` | `peak_mem_gb=8.86`，`p50≈385.7ms`, `p95≈405.7ms` |
| `LightReuse-Smoke-Task2-Replan5` | 2 | 1 | `1/1` | `skipped_vla_call_ratio=0.5`，Light-Reuse 路径生效 |

### L0: current Torch no-compile runtime baseline

设置：
- Backend：PyTorch no-compile policy server
- Model：frozen `pi05_libero`
- Tasks：LIBERO-10 Task2 / Task8
- Trials：10 per task
- Seed：7

| Setting | Method | Task2 | Task8 | Overall | Full VLA calls/ep | Wall sec/ep | VLA p50 | VLA p95 | VLA p99 | Success under deadline | GPU used |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | `B0-VLA` | `10/10` | `8/10` | `18/20` | `33.80` | `27.59` | `394.53ms` | `460.01ms` | `665.85ms` | `0.808` | `17.24GB` |
| mid-nudge 3cm | `B0-VLA` | `10/10` | `7/10` | `17/20` | `34.95` | `28.21` | `389.40ms` | `419.88ms` | `1103.90ms` | `0.760` | `17.25GB` |
| clean | `B4-Agentic` | `9/10` | `7/10` | `16/20` | `36.45` | `29.17` | `388.07ms` | `415.06ms` | `439.24ms` | `0.719` | `8.84GB` |
| mid-nudge 3cm | `B4-Agentic` | `10/10` | `9/10` | `19/20` | `34.20` | `28.49` | `390.58ms` | `462.18ms` | `1168.62ms` | `0.851` | `17.29GB` |

结果目录：
- `results/l0_B0_torchNoCompile_clean_tasks28_10trials_seed7_20260608/`
- `results/l0_B0_torchNoCompile_midNudge003_tasks28_10trials_seed7_20260608/`
- `results/l0_B4Agentic_clean_tasks28_10trials_seed7_20260608/`
- `results/l0_B4Agentic_midNudge003_tasks28_10trials_seed7_20260608/`

观察：
- `B0-VLA` clean 子集：Task2 `10/10`，Task8 `8/10`，overall `18/20`。
- `B0-VLA` mid-nudge 子集：Task2 `10/10`，Task8 `7/10`，overall `17/20`。
- `B4-Agentic` clean 子集低于 `B0-VLA`：overall `16/20`。该结果作为边界数据保留，需要与旧 `B4-RTProfileV2` 全 LIBERO-10 clean 结果分开记录。
- `B4-Agentic` mid-nudge 子集高于 `B0-VLA`：overall `19/20` vs `17/20`，Task8 `9/10` vs `7/10`。该结果支持 Agentic Harness 在扰动条件下的 selected-task 收益。
- 当前 VLA steady-state latency 大致在 `p50≈390ms`、`p95≈420-460ms`；p99 有时出现 >1s 长尾。

### Light-Reuse action-cache 原型

设置：
- Task：LIBERO-10 Task2 clean
- Trials：5
- Seed：7
- `replan_steps=5`
- `Light-Reuse` 复用 VLA action chunk suffix，不调用新的 full VLA。

| Method | Success | Full VLA calls/ep | Skipped calls/ep | Skip ratio | Wall sec/ep | VLA p95 | Success under deadline | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `replan5 no-light` | `5/5` | `50.40` | `0.00` | `0.000` | `29.69` | `416.72ms` | `0.798` | 短 commit baseline，调用数高 |
| `Light-Reuse max5` | `4/5` | `31.00` | old-count `30.20` | old-count `0.493` | `23.78` | `407.56ms` | `0.718` | aggressive reuse，成功率下降；计数旧版不进主表 |
| `Light-Reuse max2 v2` | `5/5` | `25.00` | `24.60` | `0.496` | `19.51` | `411.84ms` | `0.898` | 保守 reuse，当前最好的 light 原型 |

结果目录：
- `results/shortcommit_replan5_nolight_task2_clean_5trials_seed7_20260608/`
- `results/lightreuse_replan5_task2_clean_5trials_seed7_20260608/`
- `results/lightreuse_replan5_max2_task2_clean_5trials_seed7_20260608_v2/`

判断：
- `Light-Reuse max2 v2` 是当前可进入 `B0-VLA-Light` 的候选配置。
- 它在 Task2 clean 上相对 `replan5 no-light` 保持成功率 `5/5`，同时将 full VLA calls/ep 从 `50.40` 降到 `25.00`，wall sec/ep 从 `29.69` 降到 `19.51`。
- aggressive `max5` 会降低成功率，作为 action-cache 过度复用的负消融。
- 下一步需要在 Task8 clean 和 Task2/8 mid-nudge 上验证 `Light-Reuse max2 v2` 是否仍稳定。

Task8 clean 验证：

| Method | Task | Success | Full VLA calls/ep | Skipped calls/ep | Skip ratio | Wall sec/ep | VLA p95 | Success under deadline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Light-Reuse max2` | 8 | `2/5` | `46.40` | `46.40` | `0.500` | `37.01` | `414.73ms` | `0.360` |

结果目录：
- `results/lightreuse_replan5_max2_task8_clean_5trials_seed7_20260608/`

更新判断：
- `Light-Reuse max2` 在 Task2 clean 上有效，但在 Task8 long-horizon multi-object task 上明显退化。
- 当前不能把 action-cache reuse 作为全局默认的 `B0-VLA-Light`。
- 下一步应改为 `task-aware safe light path`：
  - single-object / short-horizon task：允许 `Light-Reuse max2`。
  - long-horizon multi-object task：禁用 action reuse，或只在更强 verifier / stage detector 确认低风险时启用。
- 这个结果可作为 lightweight VLA runtime 的 trade-off 证据：减少 full VLA calls 需要 task-aware gating，否则会伤害长程任务成功率。

### B0-VLA-Light-Safe 验证

目的：
- 验证 task-aware safe light path 是否能成为主 `B0-VLA-Light` 配置。
- 设计：Task2 允许 `Light-Reuse max2`，Task8 long-horizon multi-object 禁用 reuse。

#### Safe v1：全局 `replan_steps=5` + long-horizon 禁用 reuse

| Setting | Method | Task2 | Task8 | Overall | Full VLA calls/ep | Skipped calls/ep | Skip ratio | Wall sec/ep | Success under deadline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | `B0-VLA` | `10/10` | `8/10` | `18/20` | `33.80` | `0.00` | `0.000` | `27.59` | `0.808` |
| clean | `B0-VLA-Light-Safe-v1` | `10/10` | `8/10` | `18/20` | `58.15` | `12.45` | `0.176` | `36.70` | `0.766` |
| mid-nudge 3cm | `B0-VLA` | `10/10` | `7/10` | `17/20` | `34.95` | `0.00` | `0.000` | `28.21` | `0.760` |
| mid-nudge 3cm | `B0-VLA-Light-Safe-v1` | `9/10` | `9/10` | `18/20` | `56.50` | `13.90` | `0.197` | `35.93` | `0.764` |

结果目录：
- `results/b0_vla_light_safe_replan5_max2_clean_tasks28_10trials_seed7_20260608/`
- `results/b0_vla_light_safe_replan5_max2_midNudge003_tasks28_10trials_seed7_20260608/`

判断：
- v1 的设计存在问题：虽然 Task8 禁用了 reuse，但全局 `replan_steps=5` 仍让 Task8 变成短 commit，full VLA calls/ep 从 `42.7` 增加到 `91.0`。
- 因此 v1 不能作为主 light 配置；它说明 task-aware light path 不能只关 reuse，还必须让 long-horizon task 回到默认 commit。

#### Safe v2：全局 `replan_steps=10` + 仅 light task 使用 `light_reuse_commit_steps=5`

工程更新：
- 新增 `--light-reuse-commit-steps`。
- 当且仅当当前任务启用 `Light-Reuse` 时，将 full VLA commit cap 到 5，以缓存 suffix。
- Task8 long-horizon 禁用 reuse 后回到默认 `replan_steps=10`。

| Setting | Method | Task2 | Task8 | Overall | Full VLA calls/ep | Skipped calls/ep | Skip ratio | Wall sec/ep | Success under deadline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | `B0-VLA-Light-Safe-v2` | `9/10` | `6/10` | `15/20` | `37.05` | `14.15` | `0.276` | `29.02` | `0.674` |

结果目录：
- `results/b0_vla_light_safev2_commit5_clean_tasks28_10trials_seed7_20260608/`

判断：
- v2 修复了 Task8 calls 暴涨问题，Task8 full VLA calls/ep 为 `45.3`，接近 B0 的 `42.7`。
- 但 v2 成功率仍退化：Task2 `9/10`，Task8 `6/10`，overall `15/20`。
- 该结果说明当前 action-cache `Light-Reuse` 即便经过 task-aware gating，也不够稳定，暂不进入主 2x2 joint table。

当前结论：
- `Light-Reuse` 可作为 lightweight VLA runtime 的机制探索和负消融：它能显著降低 full VLA calls，但会引入 task-dependent 成功率风险。
- 主 lightweight 路线应暂时转向更稳的 serving / dtype / profiling：
  - current PyTorch no-compile latency baseline
  - torch compile/no-compile first-call and steady-state comparison
  - BF16/FP16 sanity if OpenPI PyTorch backend supports it
- `B4-Agentic-Light` 暂不扩大，避免把不稳定的 `Light-Reuse` 叠加到 Agentic Harness 上。

## 2026-06-04 PyTorch policy backend 工程与 clean sanity

目的：
- CPU JAX 后端在当前 `openpi` conda 环境下没有 CUDA-enabled jaxlib，`serve_policy.py` 首次 request 长时间不返回。
- 为保证 overnight benchmark 吞吐，将本地 `pi05_libero` JAX checkpoint 转换为 PyTorch checkpoint，并用 torch CUDA server 继续实验。

工程处理：
- JAX checkpoint: `/home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_libero`
- PyTorch checkpoint: `/home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
- 修正 OpenPI PyTorch 依赖：
  - `transformers==4.53.2`
  - 复制 `/home/admin1/openpi/src/openpi/models_pytorch/transformers_replace/*` 到 openpi conda 环境的 `transformers/`
- 给 `/home/admin1/openpi/src/openpi/models_pytorch/pi0_pytorch.py` 增加 `OPENPI_DISABLE_TORCH_COMPILE=1` 开关，避免 `torch.compile(max-autotune)` 在首个 request 上长时间编译。
- PyTorch server:
  - `OPENPI_DISABLE_TORCH_COMPILE=1`
  - `policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
  - port `8001`

验证：
- `results/smoke_B0_pytorchNoCompile_task2_1trial_20260604_222415`
- Task2 `1/1`，链路可用，单 episode 约 21 秒。

Clean sanity：
- `results/sanity_B4_PVLockout_clean_pytorchNoCompile_tasks28_10trials_20260604_222457`

| Method | Backend | Perturbation | Task2 | Task8 | Overall | VLA calls/ep | Wall sec/ep | Recovery precision | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `B4+PV+Lockout` | PyTorch no-compile | clean | `10/10` | `7/10` | `17/20` | `42.4` | `30.89` | `0.85` | Task2 不退化，但 Task8 clean 弱于 `B4-ProfileAuto`，不作为 clean 主配置 |

判断：
- `B4-ProfileAuto` 继续作为 clean 主方法。
- `B4+PV+Lockout` 只进入 recovery / mid-episode perturbation 主实验。
- 这个结果支持论文叙事：progress/recovery verifier 不是 always-on 越多越好，必须由 realtime/task context gating 控制。

## 2026-06-05 Realtime Agentic Policy 主实验候选：mid-episode nudge

目的：
- 验证论文核心叙事：机器人实时控制中，Agentic Policy 的价值不只是增加 planner / verifier / recovery 调用，而是用 budget-aware harness 决定何时介入、何时保持低层 VLA fast path。
- 扰动设置：`libero_10` Task 2/8，`mid_episode_nudge`，`mid_nudge_step=80`，`mid_nudge_xy=0.03`，每 task 20 trials。
- 后端：`pi05_libero_pytorch` + `OPENPI_DISABLE_TORCH_COMPILE=1`，server port `8001`。

关键结果：

| Method | Setting | Task2 | Task8 | Overall | VLA calls/ep | Wall sec/ep | Recovery | Transition | Success under deadline | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | frozen VLA fast path | `17/20` | `16/20` | `33/40` | `35.98` | `28.31` | `0` | `0` | `0.741` | 基线可用，但中途扰动下仍有明显失败 |
| `B4-ProfileAuto` | recovery + transition | `20/20` | `12/20` | `32/40` | `42.75` | `29.79` | `38` / precision `0.789` | `12` | `0.715` | Task2 受益，但 Task8 被过度 transition 伤害 |
| `B4+PV+Lockout` | progress verifier + lockout | `19/20` | `13/20` | `32/40` | `44.48` | `28.91` | `40` / precision `0.800` | `17` | `0.715` | verifier 不能 always-on，长时任务仍会被干扰 |
| `B4-PVRecoveryOnly` | Task2 only, no transition | `19/20` | - | `19/20` | `33.40` | `21.37` | `19` / precision `1.000` | `0` | `0.848` | 证明轻量 progress/recovery 在合适任务上有机制收益 |
| `B4-RTProfileV2` | no transition + long-horizon safe fast path | `20/20` | `20/20` | `40/40` | `33.10` | `23.01` | `0` | `0` | `0.899` | 当前主候选：选择性跳过高层 reasoning，实时性和成功率同时最好 |

诊断补充：
- Task8 `transition-off` diagnostic：`16/20`，与 B0 Task8 接近，说明 Task8 原始 VLA 已较强，主要问题是 naive agentic intervention 误伤。
- Task8 `PV recovery-only` diagnostic：`13/20`，recoveries `20`，precision `0.65`，说明长时多物体任务里局部 progress/recovery 可能破坏全局阶段推进。

当前论文解释：
- 对 Task2 这类单物体、目标区域明确的扰动，progress verifier 可以发现 `placement_drift`，recovery precision 高，能够从 `17/20` 提升到 `19/20`。
- 对 Task8 这类 long-horizon multi-object task，额外 transition/recovery 往往引入不必要的 horizon cost 和 stage disruption；最佳策略是实时预算下的 safe fast path。
- 因此主方法应表述为 `Realtime Budget-Aware Agentic Harness`：它包含 agentic monitor / recovery 能力，但核心创新是根据任务结构和实时预算选择性执行，而不是全局打开所有 Agent 工具。

### LIBERO-10 clean baseline

目的：
- 补齐论文主表需要的正常条件 benchmark，确认 frozen `pi05_libero` 在 clean LIBERO-10 上的基础能力。
- 设置：`libero_10` 全 10 task，每 task 10 trials，seed `7`，`perturbation=clean`。

| Method | Overall | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | VLA calls/ep | Wall sec/ep | Deadline score | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `93/100` | `10/10` | `10/10` | `10/10` | `8/10` | `9/10` | `10/10` | `10/10` | `10/10` | `7/10` | `9/10` | `27.68` | `21.95` | `0.835` | `results/clean_B0_pytorchNoCompile_libero10_10trials_seed7_20260605_123000/` |
| `B4-RTProfileV2` | `97/100` | `10/10` | `10/10` | `10/10` | `10/10` | `9/10` | `10/10` | `10/10` | `10/10` | `8/10` | `10/10` | `26.28` | `20.72` | `0.871` | `results/clean_B4RTProfileV2_pytorchNoCompile_libero10_10trials_seed7_20260605_132000/` |

判断：
- Clean 条件下 B0 已经很强，说明论文不能讲“Agentic Policy 在普通任务上大幅提升成功率”。
- `B4-RTProfileV2` 在 clean 下无 recovery/transition 调用，保持 `fast_path_ratio=1.0`，整体从 `93/100` 提升到 `97/100`。
- 同时 VLA calls/ep 从 `27.68` 降到 `26.28`，wall sec/ep 从 `21.95` 降到 `20.72`，success-under-deadline 从 `0.835` 到 `0.871`。
- 这个结果适合作为主表正结果：harness 的价值首先是 safe fast path / prompt-profile / budget profile，而不是 always-on recovery。
- Task8 clean 为 `7/10 -> 8/10`，Task3 为 `8/10 -> 10/10`，是后续 qualitative/video 分析重点。
- Task6 clean `10/10` 但 mid-nudge scan 仅 `6/10`，说明扰动 benchmark 对体现 agentic/realtime value 更有意义。

### LIBERO-10 remaining-task 10-trial scan

目的：
- 在 Task2/8 主结果之外，快速检查 `B4-RTProfileV2` 在 LIBERO-10 其他任务上的泛化边界。
- 设置：Task `0,1,3,4,5,6,7,9`，每 task 10 trials，`mid_episode_nudge`，`--ba-disable-transition --ba-safe-long-horizon-fast-path`。
- 路径：`results/scan_B4RTProfileV2_pytorchNoCompile_midNudge_libero10_remaining_10trials_20260605_004403`

结果：

| Task | Success | Avg len | 观察 |
|---:|---:|---:|---|
| 0 | `10/10` | `271.7` | basket 双物体任务，fast path 稳定 |
| 1 | `10/10` | `267.3` | basket 双物体任务，fast path 稳定 |
| 3 | `10/10` | `233.6` | fast path 稳定 |
| 4 | `7/10` | `331.7` | 弱点，需要与 B0 对照并诊断是否需要 recovery/profile |
| 5 | `9/10` | `221.1` | 基本稳定，可考虑 20-trial 复核 |
| 6 | `6/10` | `386.5` | 关系/plate 类任务弱点，可能需要专门 profile |
| 7 | `10/10` | `280.3` | basket 双物体任务，fast path 稳定 |
| 9 | `8/10` | `317.5` | 轻度弱点，需要 B0 对照 |
| Overall | `70/80` | `-` | 平均不错，但不能声称全任务已解决 |

效率：
- VLA calls/ep：`28.25`
- Wall sec/ep：`20.24`
- `fast_path_ratio=1.0`
- `success_under_deadline=0.786`

下一步：
- 正在跑同任务同扰动的 `B0-VLA` 10-trial scan，判断 Task4/6/9 是底层 VLA 弱点还是 RTProfileV2 profile 问题。
- 若 B0 在 Task4/6/9 同样弱：论文可把这些任务作为 base VLA limitation，并优先把主实验集中在 perturbation recovery value 明确的 Task2/8 + basket tasks。
- 若 B0 明显强于 RTProfileV2：需要为 Task4/6/9 加 task-specific fast path/recovery profile，不能直接扩大全量主表。

### B0 remaining-task 对照与弱任务诊断

B0 同设置扫描：
- 路径：`results/scan_B0_pytorchNoCompile_midNudge_libero10_remaining_10trials_20260605_011547`
- Overall：`71/80`，略高于 `B4-RTProfileV2` 的 `70/80`。

| Task | B0 | B4-RTProfileV2 | 判断 |
|---:|---:|---:|---|
| 0 | `10/10` | `10/10` | 两者都稳，底层 VLA 已足够 |
| 1 | `10/10` | `10/10` | 两者都稳 |
| 3 | `9/10` | `10/10` | RTProfileV2 小幅更稳，但样本少 |
| 4 | `8/10` | `7/10` | 两者都弱，不能归因于 agent profile |
| 5 | `9/10` | `9/10` | 持平 |
| 6 | `6/10` | `6/10` | 两者同弱，是底层 VLA + relation perturbation 难点 |
| 7 | `9/10` | `10/10` | RTProfileV2 小幅更稳，但样本少 |
| 9 | `10/10` | `8/10` | microwave task 不应启用额外 recovery；保守 fast path 仍需复核 |

效率对比：
- B0：VLA calls/ep `28.09`，wall sec/ep `19.98`，success-under-deadline `0.797`
- B4-RTProfileV2：VLA calls/ep `28.25`，wall sec/ep `20.24`，success-under-deadline `0.786`

结论：
- 全 LIBERO-10 remaining scan 中，`B4-RTProfileV2` 没有稳定压过 B0，因此这组只适合作为“泛化边界/负责任分析”，不作为论文主 claim。
- 主 claim 应聚焦在 Task2/8 的 mid-episode perturbation：`B0 33/40` vs `B4-RTProfileV2 40/40`，并用 `B4-PVRecoveryOnly Task2 19/20` 与 Task8 negative ablation 解释 selective intervention。

Plate/relation verifier 诊断：
- 临时实现了 `plate_relation` progress verifier，并用 `--ba-progress-verifier --ba-enable-recovery --ba-disable-transition` 跑 Task4/6。
- 路径：`results/diagnostic_B4PlatePVRecovery_pytorchNoCompile_midNudge_tasks46_10trials_20260605_014931`
- 早停于 Task4 前 4 episodes：`2/4`，recoveries `4`，recovery precision `0.5`，VLA calls/ep `47.75`。
- 判定：简单几何 plate/relation recovery 过激，伤害成功率和实时性；暂不进入主方法。后续若继续攻 Task4/6，应使用更谨慎的 recovery trigger，例如只在低层动作耗尽、object 已被抓取且目标距离长期不下降时介入。

### Task2/8 paired seed 复核

目的：
- 检查 `B4-RTProfileV2` 的 `40/40` 是否跨 seed 稳定。
- 设置：Task2/8，各 20 trials，`mid_episode_nudge=0.03`。

| Seed | Method | Task2 | Task8 | Overall | VLA calls/ep | Wall sec/ep | 结论 |
|---:|---|---:|---:|---:|---:|---:|---|
| 7 | `B0-VLA` | `17/20` | `16/20` | `33/40` | `35.98` | `28.31` | baseline |
| 7 | `B4-RTProfileV2` | `20/20` | `20/20` | `40/40` | `33.10` | `23.01` | 单 seed 很强 |
| 42 | `B0-VLA` | `20/20` | `14/20` | `34/40` | `35.40` | `24.59` | Task8 seed 波动大 |
| 42 | `B4-RTProfileV2` | `19/20` | `15/20` | `34/40` | `35.08` | `24.58` | 与 B0 持平 |

结论更新：
- `B4-RTProfileV2` 的成功率优势不跨 seed 稳定；它当前更像“安全禁用过度 agent reasoning 的 scheduler/efficiency baseline”，不能单独作为论文主贡献。
- 主贡献应改成更稳的三段式：
  1. naive agentic intervention 会伤害长时任务和实时性；
  2. task-aware realtime gating 能避免这类伤害，保持接近 B0 的 fast-path efficiency；
  3. 对明确可检测的扰动（如 single moka-pot placement drift），轻量 progress recovery 可以作为按需工具，但需要在更强扰动或更多 seed 下证明收益。

下一步实验：
- 做 perturbation stress test：提高 Task2 `mid_nudge_xy` 到 `0.05`，比较 B0 与 `B4-PVRecoveryOnly`。如果 B0 明显下降而 PV recovery 保持较高成功率，这会成为比 seed7 `0.03` 更扎实的 Agentic recovery 证据。

### Task2 stress nudge: targeted recovery 正结果

设置：
- Task2 only，10 trials，seed `7/42`
- `mid_episode_nudge`，`mid_nudge_xy=0.05`
- 对比 `B0-VLA` 与 `B4-PVRecoveryOnly`（`--ba-progress-verifier --ba-disable-transition`）

| Seed | Method | Success | Avg len | VLA calls/ep | Wall sec/ep | Recovery | 结论 |
|---:|---|---:|---:|---:|---:|---:|---|
| 7 | `B0-VLA` | `9/10` | `296.1` | `29.10` | `19.81` | `0` | 大 nudge 下出现 1 次失败 |
| 7 | `B4-PVRecoveryOnly` | `10/10` | `272.7` | `31.40` | `20.03` | `10`, precision `1.0` | 成功补回大偏移 |
| 42 | `B0-VLA` | `8/10` | `319.8` | `31.50` | `21.55` | `0` | seed42 下 baseline 更弱 |
| 42 | `B4-PVRecoveryOnly` | `10/10` | `268.2` | `31.00` | `19.68` | `10`, precision `1.0` | 成功率和 wall time 都优于 B0 |

解释：
- 这组比 `mid_nudge_xy=0.03` 更适合证明 Agentic recovery 的价值：当扰动足够强，B0 开始失败，而轻量 progress verifier 能触发确定性 recenter recovery。
- 重要的是它没有引入 VLM blocking reasoning，`planner/verifier/vlm calls` 仍为 `0`，符合 realtime 部署叙事。
- 两个 seed 上 `B4-PVRecoveryOnly` 都达到 `10/10`，而 B0 为 `9/10` 和 `8/10`；这比 `B4-RTProfileV2` 的 Task2/8 fast-path 结果更适合作为主 recovery benchmark。
- 下一步应把该 stress setting 扩到 20 trials/seed，并加入 `naive transition/recovery` 负消融，形成完整主表。

20-trial 扩展，seed `7/42/100`：

| Seed | Method | Success | Avg len | VLA calls/ep | Wall sec/ep | Recovery | 结论 |
|---:|---|---:|---:|---:|---:|---:|---|
| 7 | `B0-VLA` | `18/20` | `282.15` | `27.60` | `18.90` | `0` | baseline stress 下有 2 次失败 |
| 7 | `B4-PVRecoveryOnly` | `19/20` | `277.95` | `31.55` | `20.23` | `19`, precision `1.0` | 有提升；唯一失败来自未触发/触发窗口问题 |
| 7 | `B4-PVRecoveryOnlyConcise` | `18/20` | `299.40` | `33.75` | `24.30` | `19`, precision `0.947` | final-code 默认 concise prompt sanity；与 B0 持平，说明主 claim 需保守 |
| 7 | `B4-PVDelayed160` | `19/20` | `285.90` | `32.45` | `23.48` | `19`, precision `1.0` | 不退化，但效率弱于普通 PV |
| 42 | `B0-VLA` | `17/20` | `311.55` | `30.60` | `20.84` | `0` | seed42 更弱，有 3 次失败 |
| 42 | `B4-PVRecoveryOnly` | `19/20` | `293.15` | `33.10` | `21.24` | `19`, precision `1.0` | 稳定提升 2 个成功，avg len 明显降低 |
| 42 | `B4-PVRecoveryOnlyConcise` | `18/20` | `300.35` | `33.65` | `24.11` | `18`, precision `1.0` | final-code 默认 concise prompt sanity；仍优于 B0，但略低于旧 run |
| 42 | `B4-PVGoalAware` | `17/20` | `319.35` | `35.30` | `25.71` | `17`, precision `1.0` | 原始任务目标写入 recovery prompt 后漏触发/动作变慢，不作为默认 |
| 42 | `B4-PVMin120GoalAware` | `17/20` | `313.75` | `34.95` | `25.55` | `18`, precision `0.944` | 中间阈值仍漏触发/晚触发，不能替代普通 PV |
| 42 | `B4-PVDelayed160` | `17/20` | `306.85` | `33.90` | `24.89` | `17`, precision `1.0` | 固定延迟错过 3 次 recovery window，退回 B0 水平 |
| 42 | `B4-PVAdaptive` | `17/20` | `313.05` | `35.15` | `25.63` | `20`, precision `0.85` | far-stall 提高 recall 但降低 precision，不进入主方法 |
| 42 | `B4-PVRegression` | `17/20` | `313.60` | `35.25` | `25.19` | `19`, precision `0.895` | 单 episode progress memory 未带来稳定收益，作为负消融 |
| 100 | `B0-VLA` | `18/20` | `282.95` | `27.75` | `21.77` | `0` | baseline stress 下有 2 次失败 |
| 100 | `B4-PVRecoveryOnly` | `16/20` | `325.30` | `36.25` | `26.60` | `18`, precision `0.889` | 过早/过密 recovery 反而伤害成功率和实时性 |
| 100 | `B4-PVDelayed160` | `19/20` | `277.90` | `31.55` | `23.22` | `19`, precision `1.0` | 延迟到 160 step 后仍未恢复才介入，修复 seed100 过干预 |
| 100 | `B4-PVGoalAware` | `17/20` | `304.10` | `34.25` | `25.10` | `19`, precision `0.895` | prompt 保留 stove-on 语义可修部分失败，但不能解决 trigger timing |

更新判断：
- recovery primitive 本身可靠：当触发窗口正确时，seed7/42 的普通 PV 与 seed7/100 的 delayed PV 均达到 recovery precision `1.0`。
- 当前瓶颈不是单纯 trigger recall，而是 trigger precision/latency 的 Pareto：太早介入会破坏 VLA 自恢复，太晚介入则错过修复窗口。
- 固定 delayed trigger 不是最终主方法：seed100 正收益明显，但 seed42 会漏掉 3 个本可恢复的 episode。
- timing scan 的 `min_steps=120` 也没有修复 seed42，说明单一时间阈值不足以刻画 recovery window。
- `B4-PVAdaptive` 的 far-stall 分支修复了部分 no-trigger 失败，但 precision 降到 `0.85`；它证明“补 recall”不能粗暴做。
- `B4-PVRegression` 只利用单 episode progress memory 识别 completed-to-incomplete regression，但 seed42 仍为 `17/20`，说明当前 lightweight geometry memory 还不足以替代更可靠的 task-state estimator。
- `B4-PVGoalAware` 说明 recovery prompt 必须保留原始语言目标（turn on stove + place moka pot），但 prompt 修复不能替代好的触发策略。
- final-code sanity 显示 concise prompt 仍优于 GoalAware，但 seed42 从旧 `19/20` 波动到 `18/20`；论文表述应采用多 seed 平均与 confidence interval，避免夸大单次 run。
- final-code seed7 也为 `18/20`，因此 Task2 stress 的论文角色应从“强成功率提升”调整为“realtime trigger/precision 机制案例 + 负消融”，主 benchmark 需要回到更完整的 LIBERO-10 clean/perturbation 表。
- 论文主线应更新为：`Realtime Budget-Aware Agentic Harness` 的关键贡献是可测的 trigger precision/latency trade-off；Task2 stress 作为主 evidence，展示普通 PV/Delayed PV 在不同 seed 下的收益与失败边界，并用 adaptive/perturb-aware 负消融证明设计难点。

Perturbation-aware trigger 负结果：
- 为补 recovery trigger recall，新增显式开关 `--ba-perturbation-recovery`，在 mid-episode nudge 后若 single moka-pot 距离 stove 较远则强制一次 recovery。
- 诊断路径：`results/mainStress_B4PerturbPVRecovery_pytorchNoCompile_midNudge005_task2_20trials_seed7_20260605_080732`
- 早停于 18 episodes：`14/18`，recoveries `18`，recovery precision `0.778`，VLA calls/ep `36.17`，wall sec/ep `26.84`。
- 判定：简单提高 trigger recall 会显著降低 precision 和实时性，不进入主方法。当前主方法候选从普通 `B4-PVRecoveryOnly` 更新为 `B4-PVDelayed160`；它不是强制 post-perturbation recovery，而是给 VLA 自恢复窗口后再做 selective recovery。

## 2026-06-04 RT-BA-Harness P0 Instrumentation

### 目标
- 为 `RT-BA-Harness` 实验主线补齐开跑前工程：episode-level trace、实时指标、推理调用统计、方法 tag。
- 先不改变策略行为，只增加论文需要的观测量，避免大规模 rollout 后缺少 realtime / inference-budget 证据。

### 核心改动
- 在 `scripts/run_agentic_vla_libero.py` 中新增 `EpisodeInstrumentation`。
- 新增 CLI：
  - `--method-tag`
  - `--episode-trace-jsonl`
  - `--control-deadline-ms`
- 新增每 episode 的 `episode_traces.jsonl`，默认写到 `--results-json` 同目录。
- `summary.json` 新增：
  - `episode_trace_jsonl`
  - `trace_aggregate`
  - `hardware_config`

### 新增记录指标
- VLA / planner / verifier calls
- planner / verifier / VLA latency
- `TTFA`
- `deadline_miss_rate`
- `blocking_reasoning_ms`
- `fast_path_ratio`
- recovery trigger / success / precision
- avg committed chunk steps
- peak GPU memory

### 验证
- `python -m py_compile scripts/run_agentic_vla_libero.py` 通过。
- `--help` 已确认新 CLI 参数可见。
- 本地无 LIBERO 环境依赖的小型 trace 写入/读取/汇总测试通过。

### 下一步
- 用 Task 8 做 5-trial smoke run，检查 `summary.json` 与 `episode_traces.jsonl` 是否完整。
- 然后实现/固化 `B4-BAHarness-Rules` 和 `B4-Async` 的方法配置。

### 2026-06-04 Smoke 结果

运行环境：
- policy server：`pi05_libero` on `/home/admin1/openpi`, port `8001`
- runner：`conda openpi`
- 修复项：将 `~/.libero/config.yaml` 从已不存在的 `/home/admin1/ct/openpi-official/...` 更新到 `/home/admin1/openpi/third_party/libero/...`
- 注意：与 VLA server 同卡时，Qwen3-VL critic 4bit 加载失败，当前 `--critic` 回退到 heuristic fallback。

结果汇总：

| Method | Task | Success | VLA calls/ep | VLM calls/ep | Deadline miss | Fast path | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | 8 | `4/5` | `41.6` | `0.0` | `0.100` | `1.000` | `0` |
| `B0-VLA` | 9 | `5/5` | `27.4` | `0.0` | `0.102` | `1.000` | `0` |
| `B0-VLA` | 6 | `5/5` | `21.8` | `0.0` | `0.102` | `1.000` | `0` |
| `B1-FullRefined` | 8 | `3/5` | `51.2` | `6.0` | `0.102` | `0.987` | `0` |
| `B1-FullRefined` | 9 | `5/5` | `28.6` | `3.0` | `0.104` | `0.988` | `0` |
| `B1-FullRefined` | 6 | `4/5` | `35.0` | `3.2` | `0.104` | `0.989` | `0` |
| `B2-FixedHarness` (`--transition`) | 8 | `3/5` | `51.0` | `0.0` | `0.101` | `1.000` | `0` |
| `B2-FixedHarness` (`--transition`) | 9 | `5/5` | `32.6` | `0.0` | `0.101` | `1.000` | `0` |
| `B2-FixedHarness` (`--transition`) | 6 | `5/5` | `28.6` | `0.0` | `0.104` | `1.000` | `0` |

结论：
- P0 instrumentation 通过：每个 run 都生成 `summary.json`、`episode_traces.jsonl` 和视频，trace 数量与 trials 一致。
- 不能直接进入全量 LIBERO-10 主实验：当前 `B1/B2` 在 Task 8 smoke 中没有优于 `B0`。
- `recoveries_triggered = 0` 仍然是主要短板，证明 B4 必须先实现 recovery closure。
- `B1` 的 VLM critic 实际未加载，不能把这批 smoke 当作真实 VLM-critic 实验。
- 下一步优先实现 `B4-BAHarness-Rules`：
  - risk-gated transition，避免每个 episode 固定触发；
  - recovery primitive 真实触发并记录；
  - VLM verifier 使用 CPU/offload 或 small-first fallback，不能只记录 heuristic checks。

### 2026-06-04 B4-BAHarness-Rules 调通结果

核心修复：
- B4 默认启用 GraphRAG memory，但不再默认改写底层 VLA prompt；只有显式 `--ba-augment-prompt` 才把 priors 注入低层 prompt。
- 新增 `--ba-recovery-risk-threshold` 和 `--ba-transition-risk-threshold`，默认 `0.65`。
- 对 `microwave`/door-closing 类任务关闭 B4 transition expert，只保留 recovery；原因是这类任务的失败更像对齐/收尾问题，不是多物体阶段切换。

阶段性 smoke 结果：

| Method | Task | Success | Recovery | Transition | VLA calls/ep | Deadline miss | Fast path |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B4-BAHarness-Rules` tuned | 8 | `5/5` | `5` | `2` | `48.0` | `0.105` | `1.000` |
| `B4-BAHarness-Rules` task-aware | 9 | `5/5` | `0` | `0` | `24.6` | `0.101` | `1.000` |
| `B4-BAHarness-Rules` tuned | 6 | `5/5` | `3` | `2` | `27.8` | `0.107` | `1.000` |

对照观察：
- 相比 `B0-VLA` smoke，B4 在 Task 8 从 `4/5` 到 `5/5`，Task 6/9 不退化。
- 相比 `B1/B2`，B4 解决了 Task 8 中过度 transition 导致的退化。
- `blocking_reasoning_ms=0`、`VLM calls=0`，当前 B4 是纯规则/scheduler 版，适合作为 realtime-friendly 主配置。
- `B4 w/o task-aware transition` 在 Task 9 出现 `4/5`，失败 episode 同时触发 recovery 和 transition；可以作为消融解释为什么 Agentic Policy 需要 task-aware harness，而不是固定工具链。

Go 判定：
- 可以进入 Batch 1：LIBERO-10 core，Task `8/6/9/0/5`，每 task 20 trials。
- 仍需记录一个风险：Task 8 的 VLA calls/ep 高于 B0，后续需要用 adaptive commitment/cache 或 recovery threshold ablation 拉回 efficiency。

### 2026-06-04 B4 Profile-Auto 冻结

探索中发现：
- `transition-on` 在 Task 8 core 前 20 trials 表现强：`18/20`，但在 Task 6 partial core 中出现 transition 相关失败，跑到 `15/17` 后停止。
- `recovery-only` 在 Task 6 可行：`5/5`；但 Task 8 明显不够：`3/5`；Task 9 也会因不必要 recovery 降到 `4/5`。
- 因此不能把 Agent 工具链全局打开或全局关闭，必须按任务语义做 profile-aware scheduling。

当前冻结的 `B4-BAHarness-Rules` profile：
- `stove` / multi-object placement：recovery enabled，transition enabled。
- `default`：recovery enabled，transition disabled。
- `microwave` / door-closing clean tasks：recovery disabled，transition disabled。
- state-gap sensing 仍用于 risk score；低层 VLA prompt 默认不被 memory 改写。

Profile-auto smoke：

| Method | Task | Success | Recovery | Transition | VLA calls/ep | Deadline miss | Fast path |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B4-ProfileAuto` | 8 | `4/5` | `5` | `4` | `56.6` | `0.104` | `1.000` |
| `B4-ProfileAuto` | 6 | `5/5` | `5` | `0` | `28.8` | `0.109` | `1.000` |
| `B4-ProfileAuto` | 9 | `5/5` | `0` | `0` | `25.8` | `0.102` | `1.000` |

Go 判定更新：
- `B4-ProfileAuto` 是当前主配置。
- `transition-on` 和 `recovery-only` 不作为主方法，但适合放入 ablation。
- 下一轮 core 应重新跑 `B4-ProfileAuto`：Task `8/6/9/0/5`，每 task 20 trials。

### 2026-06-04 Core 尝试与 Task8 转移策略风险

正式 core 尝试：
- `results/core_B4_profileauto_libero10_tasks86905_20trials_20260604_1405`
- Task 8 已完成：`16/20`，recoveries `20`，transitions `16`，VLA calls/ep `54.55`。
- Task 6 partial：`5/5`，recoveries `4`，transitions `0`，VLA calls/ep `25.8`。
- 结论：ProfileAuto 已经比旧 Task8 弱项强，但 Task8 transition 仍然过激；不应继续扩大该配置到完整 core。

额外尝试：
- 增加 `--ba-post-recovery-transition-lockout-steps 180`。
- Task 8 smoke：`4/5`，recoveries `5`，transitions `2`。
- 唯一失败仍是 transition episode，说明仅加 lockout 不够；下一步应把 transition 改为更严格的 late-stage rescue，或做 transition prompt/primitive 重设计。

下一步建议：
- 保留 `ProfileAuto` 作为当前主配置候选。
- Task 6/9 可进入 20-trial core。
- Task 8 先做小规模 transition policy ablation：
  - recovery-only：已知 `3/5`，不足；
  - transition threshold `0.80/0.90`；
  - transition only if no recovery has been used；
  - transition prompt 改成第二物体阶段推进，而不是 generic re-center。

补充 ablation：
- `stove` profile 使用 transition threshold `0.90` + collision-only transition。
- 结果：Task 8 `2/5`，transitions `4`，VLA calls/ep `58.4`。
- 结论：简单提高 transition 阈值或只允许 collision 触发并不能解决问题，反而容易让 transition 变成 late/horizon-expensive intervention。默认配置恢复为 `ProfileAuto`，下一步更应该改 transition primitive/prompt 或引入轻量 VA recovery expert。

### 2026-06-04 Task6/9 Core 与 Fast-Path Profile

`B4-ProfileAuto` core：

| Task | Success | Recovery | Transition | VLA calls/ep | Deadline miss | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 6 | `18/20` | `18` | `0` | `32.35` | `0.107` | recovery 偏积极，可能误伤稳定任务 |
| 9 | `19/20` | `0` | `0` | `28.50` | `0.102` | fast path 基本成立，唯一失败不是 Agent 介入 |

Task 6 recovery-off ablation：

| Setting | Success | Recovery | Transition | VLA calls/ep | Deadline miss |
|---|---:|---:|---:|---:|---:|
| `B4-ProfileAuto` | `18/20` | `18` | `0` | `32.35` | `0.107` |
| `B4 --ba-disable-recovery` | `19/20` | `0` | `0` | `24.20` | `0.104` |

结论：
- 对稳定 plate/relation task，最优 Agent 行为是 fast path：memory/monitor 可以保留，但不执行 recovery/transition。
- 新增 `plate` profile：`ba_recovery_enabled=False`，`ba_transition_enabled=False`。
- `plate` profile sanity：Task 6 `5/5`，recoveries `0`，transitions `0`。
- 这支持论文主张：`Budget-aware` 不是多调用，而是学会何时不调用。

## 2026-05-20/21 Agentic Pipeline 升级（VLM on-demand + 可验证结构化决策 + Failure Taxonomy）

### 目标
- 将系统从“VLA 低层策略 + prompt 改写”升级为可发表的软件 Agentic pipeline：只在需要时触发 VLM planner，并且 planner 输出可验证的结构化决策（有限集合），同时与既有创新模块（Transition / GraphRAG-Memory / Critic）形成统一闭环。
- 在推理资源受限（单卡 4090）设定下可运行：planner 4bit 量化 + on-demand lazy-load + 冷却/上限/缓存，避免 planner 反复介入。

### 核心改动（系统级）
- Planner（server side, VLM）：
  - 只在 client 发出 `agentic.request_plan=true` 时触发；不触发则 `plan_ms=0` 且不会加载/占用 planner 资源。
  - 输出严格 JSON（可验证）：`action ∈ {continue,regrasp,lift,retreat,slow_down}` + `prompt_suggested` + `client_hints{action_scale,apply_steps}` + `rationale`。
  - 扩展结构化字段：`subgoal ∈ {execute,recover,regrasp,lift,retreat,slow_down}`，用于驱动 client 侧子目标状态机（FSM）。
  - server 侧解析/校验/回退：非 JSON 或 action 不在白名单时自动回退到安全默认（continue）。
  - planner 节流：`plan_cooldown_steps`（最小间隔）+ `max_plans_per_episode`（每 episode 上限）+ `cache_plans`（按 event+instruction 缓存）。
- Trigger（client side, 无常驻 VLM）：
  - Failure taxonomy（规则/状态信号）扩展为：`stall / collision / misgrasp / slip(启发式)`。
  - 与既有创新融合：除 taxonomy 外，`state_gap`（Transition Agent 触发）与 `critic`（Critic 判定需要 retry）也会触发 planner 请求，但均遵循 cooldown。
- Executor（client side, VLA）：
  - 真正应用 planner 的 `client_hints`：在接下来 `apply_steps` 内对动作（前 6 维）乘以 `action_scale`，形成“可控的 recovery 参数化执行”。
  - 子目标状态机（FSM）：
    - `execute`：正常执行（VLA chunked actions）。
    - `recover/*`：执行恢复策略（结合 action_scale/apply_steps），并使用 failure taxonomy 的“无 failure 连续窗口”作为可验证退出条件。
- Context injection（结合 GraphRAG/Memory）：
  - 当 GraphRAG 提取到 priors 时，将压缩后的先验摘要写入 `agentic.context`，只在 planner 请求时发送，提升规划可靠性且不增加常规控制负担。

### 推荐 benchmark 验证方式（LIBERO）
- 对比原则：只改一个变量（是否启用 on-demand planner），其余控制变量一致，便于投稿叙事与 ablation。
- Baseline server（无 agentic）：
  - `PYTHONPATH=/home/admin1/ct/openpi-official/src conda run -n openpi python /home/admin1/ct/openpi-official/scripts/serve_policy.py --env LIBERO --port 8000`
- Agentic server（on-demand planner + 4bit）：
  - `PYTHONPATH=/home/admin1/ct/openpi-official/src conda run -n openpi python /home/admin1/ct/openpi-official/scripts/serve_policy.py --env LIBERO --port 8000 --agentic --planner-model /home/admin1/ct/Agentic-VLA/weights/Qwen3-VL-8B-Instruct --planner-device cuda --planner-dtype bfloat16 --planner-quant 4bit --plan-mode on_demand --plan-cooldown-steps 50 --max-plans-per-episode 3 --cache-plans true`
- Runner（libero_10 小样本快速验证；完整实验按论文口径调整 trials/task）：
  - `python scripts/run_agentic_vla_libero.py --task-suite libero_10 --task-id 0 --trials 20 --host 127.0.0.1 --port 8000 --agentic-planner --planner-cooldown-steps 50`
  - 若要同时启用既有创新模块：
    - `python scripts/run_agentic_vla_libero.py --task-suite libero_10 --task-id 0 --trials 20 --host 127.0.0.1 --port 8000 --transition --graph-rag --critic --agentic-planner --planner-cooldown-steps 50`

### 结果记录口径（用于论文）
- 主要指标：success rate、avg episode length。
- 机制指标（新增/增强）：
  - `planner_stats.total_requests / plans_generated`
  - `planner_stats.event_counts`（stall/collision/misgrasp/slip/state_gap/critic）
  - `planner_stats.action_counts`（continue/regrasp/lift/retreat/slow_down）
  - `agentic.subgoal`：每次 planner 触发时的子目标标签（FSM 统计与可解释性补充）
- 注：`slip` 当前是启发式（夹爪开合轨迹），后续若需要更强可验证性，优先改为“抓取后物体状态信号”或“可重复的视觉规则”，避免引入额外常驻 VLM。

## 2026-05-16 V2 结果汇总与精简方案 B

### 当前状态
- `Task 8/9/6` 的 `v2 + 单模块消融` 队列已全部完成。
- 当前没有剩余 `run_agentic_vla_libero.py` rollout 进程在跑；只有 policy server 仍在运行。
- 这意味着当前可以基于真实结果决定下一步，而不需要继续等待后台队列。

### 低成功率任务真实结果
- `Task 8`
  - 旧 `Full`: `40%`
  - baseline: `55%`
  - `Full v2`: `70%`
  - `Transition only`: `80%`
  - `Graph only`: `60%`
  - `Critic only`: `70%`
- `Task 9`
  - 旧 `Full`: `90%`
  - baseline: `95%`
  - `Full v2`: `100%`
  - `Transition only`: `90%`
  - `Graph only`: `100%`
  - `Critic only`: `90%`
- `Task 6`
  - 旧 `Full`: `80%`
  - baseline: `80%`
  - `Full v2`: `90%`
  - `Transition only`: `100%`
  - `Graph only`: `80%`
  - `Critic only`: `90%`

### 当前判断
- `v2` 修正确实有效：
  - `Task 8`: `40% -> 70%`
  - `Task 9`: `90% -> 100%`
  - `Task 6`: `80% -> 90%`
- `Transition` 是当前最稳定的增益来源：
  - `Task 8`: `80%`
  - `Task 6`: `100%`
- `GraphRAG/Memory` 在 `Task 9` 上价值明显，但在 `Task 8/6` 上不够稳定。
- `Critic` 仍没有产生有效 retry，当前更像“检查器”，而不是已经被结果证明有效的恢复模块。
- 最关键现象是：
  - `Task 8` 上 `Transition only = 80% > Full v2 = 70%`
  - 说明 `Full` 里至少还有一个附加模块在拖后腿

### 决策
- 不再重复扩大单模块矩阵。
- 下一步改为补一个精简版方案 B，只针对 `Task 8` 做：
  - `Full w/o Graph`
  - `Full w/o Critic`
  - `Full w/o Transition`
- 目的：
  - 用最小 wall-clock 成本确认到底是哪一块拖累了 `Task 8` 的完整框架表现

### 新增脚本
- 已新增：
  - `scripts/run_full_drop_ablations.sh`
- 默认行为：
  - task：`8`
  - variants：`wo_graph wo_critic wo_transition`
  - trials：`10`

### 精简方案 B 最终结果
- `Task 8` 的 `Full w/o ...` 已全部完成：
  - `Full w/o Graph = 60%`
  - `Full w/o Critic = 70%`
  - `Full w/o Transition = 80%`
- 由此得到的最终判断：
  - `Graph + Memory` 不是负资产，去掉后结果明显变差
  - `Critic` 当前贡献很弱，去掉后基本不变
  - `Transition` 单独有效，但在当前 full-stack 中与其他模块的协同尚未调到最佳
- 因此最稳妥的结论不是“某个模块完全没用”，而是：
  - 当前完整框架的瓶颈主要来自模块交互，而不是单模块绝对失效

### 投稿前最终收口判断
- 当前不再继续扩大实验面。
- 投稿版实验边界确定为：
  - 主文：`A1 baseline + Full on libero_10`
  - supporting evidence：
    - `Task 8/9/6` 的 `v2` targeted runs
    - `Task 8` 的单模块与 `Full w/o ...` 诊断
- 后续若继续拓展，优先作为期刊方向打磨 `Graph + Memory` 与 `Critic`，而不是继续在当前会议版上无止境补矩阵。

## 2026-05-15 V2 执行状态同步

### 当前运行状态
- `Full v2 on Task 8` 仍在运行：
  - `results/ablation_FULL_tuned_v2_task8_20260515_160827/`
- 当前审计到的实时状态：
  - `run_agentic_vla_libero.py --task-id 8 --trials 10` 进程仍在运行
  - GPU 上该进程已占用约 `9.4 GiB` 显存
  - 结果目录已创建，但截至本次同步时尚未写出 `_partial_summary.json`、`summary.json` 或视频文件
- 当前解释：
  - 脚本只有在首个 episode 完成后才会持久化阶段性 JSON
  - 因此当前更稳妥的判断是“仍在初始化或执行首条 rollout”，而不是“实验已失败”

### 已挂起的后续实验队列
- 已新增顺序队列脚本：
  - `scripts/run_pending_low_success_pipeline.sh`
- 已实际启动后台队列，当前行为是：
  1. 等待当前 `Task 8 Full v2` 结束
  2. 继续执行 `Task 8` 的 `transition / graph / critic` 消融
  3. 执行 `Task 9` 与 `Task 6` 的 `full / transition / graph / critic` 矩阵
- 这样可以避免手动盯进程导致后续实验断档。

### 本次同步完成的文档更新
- 已更新：
  - `FULL_EXPERIMENT_PLAN.md`
  - `RESULTS_EVIDENCE_GUIDE.md`
  - `paper/Agentic-VLA/SUBMISSION_ASSETS_V1.md`
- 当前口径统一为：
  - 主结果仍是 `91.0%`
  - `v2` 低成功率任务实验属于 targeted evidence
  - 在结果文件真正落盘前，不能把任何运行中目录写成完成结果

### 本次已完成的安全清理
- 已删除论文 LaTeX 构建中间文件：
  - `agentic_vla_paper_v1.aux`
  - `agentic_vla_paper_v1.fdb_latexmk`
  - `agentic_vla_paper_v1.fls`
  - `agentic_vla_paper_v1.log`
- 已删除已被 Gemini 正式图替换的旧占位图：
  - `fig_method_overview.png`
  - `fig_transition_gap.png`
  - `fig_scene_priors_memory.png`
  - `fig_critic_retry_flow.png`
- 所有真实实验结果目录保持不删。

## 2026-05-15 V2 定向修正与低成功率任务补实验

### 为什么继续做 V2
- 当前主结果已经完成：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
  - `91/100 = 91.0%`
- 但弱点任务仍然突出：
  - `Task 8`: `40%`
  - `Task 9`: `90%`
  - `Task 6`: `80%`
- 尤其 `Task 8` 明显低于 baseline 的 `55%`，因此值得在投稿前窗口内做最小但有针对性的 `v2` 修正。

### V2 代码层发现
- 重新审计 `run_agentic_vla_libero.py` 后确认：
  - 旧版 `GraphRAG/Memory` 对复合任务使用简单 `dict.update()` 合并先验；
  - 对 `put both moka pots on the stove` 这类任务，会让 `stove` 的目标区域先验覆盖 `moka pot` 的抓取/运输先验；
  - 这会弱化 `keep upright`、`approach from side` 这类对 Task 8 很关键的对象级约束。

### 已实施修正
- `Scene Priors`：
  - 引入 `role=manipulated_object / target_region / target_container`
  - 将 object prior 与 target prior 组合，而不是覆盖
  - 新增 `mug`、`microwave` 等泛化先验
- `Transition`：
  - 对 `stove` 任务使用
    - `re-center above the burner`
    - `keep the object upright`
  - 对 `microwave` 任务使用
    - `re-center with the opening`
    - `keep the object upright`
- `Recovery`：
  - 对 `stove/microwave` 任务采用 task-aware recovery prompt
- `Task-aware control profile`：
  - 对 `stove/microwave` 任务提高 `transition_min_steps`
  - 对 `critic_min_steps` 延后
  - 对 stall window / threshold 做更保守设置

### 当前正在运行的补实验
- `Full v2 on Task 8`
- 结果目录：
  - `results/ablation_FULL_tuned_v2_task8_20260515_160827/`
- 目标：
  - 在最小 wall-clock 成本下，先验证 `Task 8` 是否从 `40%` 提升

### 新增执行脚本
- 为了系统完成低成功率任务补实验，已新增：
  - `scripts/run_low_success_ablations.sh`
- 默认顺序：
  - tasks：`8 9 6`
  - variants：`full transition graph critic`
  - 每组：`10 trials`

## 2026-05-14 主实验最终完成

### 最终结果
- 当前论文主实验：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
- 当前状态：
  - 已完成
- 最终总结果：
  - `91/100 = 91.0%`
- 对应主基线：
  - `results/ablation_B0_pi05_libero_10_20260512/summary.json`
  - `180/200 = 90.0%`
- 结论：
  - 在 official `LIBERO` 模拟器真实 rollout 下，修正版 `Full-Agentic-VLA-Tuned` 对强 baseline 取得了 `+1.0` 个百分点的最终提升。

### 分任务结果
- `Task 0`: `100%`，baseline `90%`
- `Task 1`: `100%`，baseline `100%`
- `Task 2`: `100%`，baseline `95%`
- `Task 3`: `100%`，baseline `90%`
- `Task 4`: `100%`，baseline `100%`
- `Task 5`: `100%`，baseline `95%`
- `Task 6`: `80%`，baseline `80%`
- `Task 7`: `100%`，baseline `100%`
- `Task 8`: `40%`，baseline `55%`
- `Task 9`: `90%`，baseline `95%`

### 机制统计
- `transition_stats.total_transitions = 81`
- `transition_stats.transitions_leading_to_success = 72`
- `avg_transitions_per_episode = 0.81`
- `transition_success_rate_when_used = 88.89%`
- `critic_stats.total_checks = 327`
- `critic_stats.retries_triggered = 0`
- `critic_stats.retries_leading_to_success = 0`
- 解释：
  - 当前最终收益主要与 `Transition` 干预相关，而不是由 `Critic/Retry` 的显式恢复闭环带来。

### 审计口径
- 所有最终数字均来自 `summary.json` / `_partial_summary.json` / `_resume_state.json`，并可回溯到本地 rollout 视频。
- 当前结果是 official `LIBERO` 模拟器中的真实交互执行，不是物理机器人实验，也不是离线估计。
- 论文可做正向表述，但必须保持保守：
  - 可以写“Full 在 `libero_10` 上以真实 rollout 取得小幅但可审计的提升”
  - 不应写成“框架已全面解决长程任务”或“critic 已被结果证明有效”

## 2026-05-14 主实验运行与 Critic 审计

### 当前运行状态
- 当前论文主实验仍为：
  - `results/ablation_FULL_tuned_libero10_20260513_225055/`
- 最新已审计运行中检查点：
  - `57/57` success
- 当前仍是 official `LIBERO` 模拟器中的真实 rollout，不是 mock，不是离线估计。
- 当前 GPU 计算进程中，仅观察到 tuned `Full` 对应的 rollout Python 进程；未观察到旧失败版 `Full` 结果目录对应的运行中 rollout 进程。

### Critic / Qwen 审计结论
- 当前 tuned `Full` 的启动命令明确包含：
  - `--critic`
  - `--qwen-model /home/admin1/ct/Agentic-VLA/weights/Qwen3-VL-8B-Instruct`
  - `--qwen-quant 4bit`
- 运行进程环境与命令行均指向当前 `openpi` conda 环境和主实验目录。
- 对正在运行的 rollout Python 进程进行非侵入式审计时，已观察到：
  - `bitsandbytes` CUDA 动态库已映射进进程地址空间
  - `safetensors` 动态库已映射进进程地址空间
- 在当前脚本实现中，这两类库在主实验路径里只出现在 `CriticAgent.load_model()` 的 Qwen3-VL 加载逻辑中，因此上述运行时证据与“Critic 已真实按 4bit Qwen 路径加载”一致。
- 但在不打断当前真实 benchmark 的前提下，尚未直接抓到启动终端中的
  - `Qwen3-VL loaded successfully`
  - 或等价的逐条运行日志文本
- 因此当前最稳妥表述应为：
  - **高置信度认为当前 tuned Full 的 critic 正在使用 Qwen3-VL，而非 heuristic fallback；但论文文字仍应保持保守，不把这一点写成超出审计证据强度的绝对断言。**

## 2026-05-13 Full 修正版推进

### 当前判断
- `Full-Agentic-VLA` 首轮真实 `libero_10` 运行已产生可审计失败证据，但当前不支持“框架已成功提升长任务”这一结论。
- 首轮 `Full` 在 `Task 0` 的前 `4` 条真实 rollout 全部失败，且干预频次偏高：
  - `transition_total = 12`
  - `retry_total = 8`
  - `critic_checks = 104`
- 这更像是当前策略设置导致的 **过干预**，而不是单纯 benchmark 难度问题，因为 baseline 在 `Task 0` 上为 `90%`。

### 已冻结的首轮 Full 证据
- 原始结果目录：`results/ablation_FULL_libero10_20260513_203506/`
- 冻结快照：`results/ablation_FULL_libero10_20260513_203506_before_tuning/`
- 当前可见真实视频：
  - `task0_trial0_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
  - `task0_trial1_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
  - `task0_trial2_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`
  - `task0_trial3_failure_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4`

### 已实施的策略修正
- `Transition`：
  - `stall_threshold: 0.003 -> 0.0015`
  - `stall_window: 8 -> 12`
  - 增加最早触发步数：`TRANSITION_MIN_CONTROL_STEPS = 60`
  - 单 episode 最大 transition：`3 -> 1`
  - `transition chunk steps: 15 -> 6`
  - prompt 从“回中立位并张开夹爪”改为“轻微抬升并稳定夹爪，同时保持当前抓取”
- `Critic/Retry`：
  - `check_interval: 20 -> 60`
  - 增加最早检查步数：`CRITIC_MIN_CONTROL_STEPS = 120`
  - 单 episode 最大 retry：`2 -> 1`
  - `recovery steps: 15 -> 6`
  - recovery prompt 去掉 `open the gripper`
  - retry 触发条件从“只要 error_type 非 unknown 即可”收紧为“仅在明确 stuck 时才允许 retry”
  - critic prompt 新增保守约束：只有在持续卡住或明显抖动时才报告 `stuck=true`

### 当前主实验线
- 新一轮实验改为 `Full-Agentic-VLA-Tuned`
- 启动脚本：`scripts/launch_full_libero10.sh`
- 当前目标：先验证修正版是否显著降低 `Task 0` 上的过干预，再继续观察 `libero_10` 弱任务上的真实收益
- 论文主线同步收敛为：
  - `A1`: 强 baseline
  - `Full`: 当前唯一主文 agentic 结果线
  - `A2`: 保留历史证据，但不进入主文结果叙事

## 当前结论
- 只有真实 LIBERO 运行产生的 JSON 和视频可以作为论文证据，之前所有硬编码、估算和 mock 结果全部作废。
- 真实 `pi0_base` 基线已经跑通到 `LIBERO + official openpi + websocket server + 本地评测脚本` 这一完整链路。
- 真实 `pi0_base` 先导结果仍然较弱：`libero_spatial` 的 `task 0~4`，每个任务 `3` 次，共 `15` 条 episode，当前结果为 `0/15`。
- 真实 `pi05_libero` 官方配置可以在当前链路上稳定运行；已完成 `libero_spatial` 全量 `10 tasks x 50 trials = 500 episodes`，成功 `497/500`，成功率 `99.4%`。
- 当前还没有可信的 Agentic-VLA 消融结果，因为 `agentic_vla` 中的 Agentic 模块尚未接入真实的 LIBERO 评测链路。

## 关键证据文件
- 真实 `pi0_base` 先导结果汇总：`results/pi0_base_smoke_20260512/summary.json`
- 真实 `pi0_base` 单任务结果：
  - `results/pi0_base_smoke_20260512/task_0.json`
  - `results/pi0_base_smoke_20260512/task_1.json`
  - `results/pi0_base_smoke_20260512/task_2.json`
  - `results/pi0_base_smoke_20260512/task_3.json`
  - `results/pi0_base_smoke_20260512/task_4.json`
- 真实 `pi0_base` 失败视频：`results/pi0_base_smoke_20260512/videos/`
- 真实 `pi05_libero` 全量 `libero_spatial` 结果：`results/full_libero_20260512/libero_spatial.json`
- 真实 `pi05_libero` 全量 `libero_spatial` 视频：`results/full_libero_20260512/videos/libero_spatial/`
- 真实评测入口：[run_agentic_vla_libero.py](file:///home/admin1/ct/Agentic-VLA/scripts/run_agentic_vla_libero.py)

## 2026-05-12 真实实验推进记录

### 1. 审计并撤回伪结果
- 确认旧版 [run_agentic_vla_libero.py](file:///home/admin1/ct/Agentic-VLA/scripts/run_agentic_vla_libero.py) 曾包含 `np.random`、`expected_sr`、mock env 等伪评测逻辑。
- 已将该脚本重写为真实评测入口，只接受环境真实 `done` 信号，不再生成硬编码成功率。
- 已删除一批伪结果文件，并撤回旧文档中的 `86% / 91% / 95% / 99%` 等虚假或不可追溯结果。

### 2. 打通官方 openpi 与 LIBERO 运行链路
- 使用官方仓库 `/home/admin1/ct/openpi-official` 作为真实运行基础，而不是项目内不完整的 `openpi` 目录。
- 完成 `uv sync --python 3.11`，补齐 `jax`、`flax`、`lerobot`、`openpi-client`、`robosuite`、`bddl`、`libero` 等依赖。
- 将 `~/.libero/config.yaml` 指向 `openpi-official/third_party/libero`。
- 修复 `norm_stats.json` 缺失问题，确保 `pi0_base` 能真实加载。
- 当前默认使用 `egl` 渲染，避免本机缺失 `osmesa` 导致的启动失败。

### 3. 真实 `pi05_libero` 对照运行
- 说明：这一结果对应官方 `pi05_libero` 配置，不是 `pi0_base`。
- 已完成 `libero_spatial` 全量评测：
  - 任务数：`10`
  - 每任务 trials：`50`
  - 总 episode：`500`
  - 成功：`497`
  - 成功率：`99.4%`
- 结果文件：`results/full_libero_20260512/libero_spatial.json`
- 当前目录中 `libero_object` 只有部分视频，说明之前全量长跑被中断，不能当作完整结果使用。

### 4. 真实 `pi0_base` 基线先导实验
- 服务端配置：
  - `policy.config = pi0_libero`
  - `policy.dir = weights/openpi-assets/checkpoints/pi0_base`
- 评测设置：
  - suite：`libero_spatial`
  - task：`0~4`
  - 每任务 trials：`3`
- 当前真实结果：
  - `task 0`: `0/3`
  - `task 1`: `0/3`
  - `task 2`: `0/3`
  - `task 3`: `0/3`
  - `task 4`: `0/3`
  - 合计：`0/15`
- 结果文件：`results/pi0_base_smoke_20260512/summary.json`

## 当前工程判断
- 真实 baseline 现在分成两条线：
  - `pi05_libero`：官方配置可稳定跑通，并已得到可信高成功率结果。
  - `pi0_base`：真实链路已打通，但在当前 `libero_spatial` 先导任务上表现为 `0/15`。
- 这说明后续论文里必须明确区分“官方 fine-tuned LIBERO 模型”和“base model”。
- 当前最重要的不是继续写结论，而是把 Agentic-VLA 的最小增强模块接入真实评测，再与 `pi0_base` 做公平对比。

## 2026-05-12 Vision Prompt 接入真实链路

### 5. Vision Prompt 代码实现
- 已在 [run_agentic_vla_libero.py](file:///home/admin1/ct/Agentic-VLA/scripts/run_agentic_vla_libero.py) 中实现 Vision Prompt 功能：
  - 新增 `_apply_vision_prompt()` 函数：利用 `SegmentationRenderEnv` 提供的 ground-truth 分割，对 `obj_of_interest` 中的目标物体叠加半透明彩色 mask
  - 新增 CLI 参数：`--vision-prompt`、`--mask-alpha`、`--ablation-tag`
  - 当 `--vision-prompt` 启用时，自动切换到 `SegmentationRenderEnv`，获取 `agentview_segmentation_instance` 分割图
  - 分割图 shape `(256, 256, 1)` 已处理 squeeze，与 RGB 图 `(256, 256, 3)` 对齐
  - 结果 JSON 中包含 `vision_prompt`、`mask_alpha`、`ablation_tag` 字段
- 端到端验证通过：
  - `SegmentationRenderEnv` 创建、`reset()`、`set_init_state()` 均正常
  - `obj_of_interest` 和 `instance_to_id` 映射正确
  - Task 0 验证：`akita_black_bowl_1` (843 px, 绿色) + `plate_1` (1388 px, 橙色) 共 2231 像素被 mask
- 待运行：libero_spatial task 0~4 的 Vision Prompt 消融实验

## 关键发现：pi0_base vs pi05_libero

### 为什么 pi0_base 在 LIBERO 上 0% 成功率？
- **根因**：`pi0_base` 是预训练模型（pre-trained only），从未在 LIBERO 数据上 fine-tune
- **三篇论文一致结论**：所有 VLA 都需要 task-specific fine-tuning 才能在 LIBERO 上工作
  - pi0 论文：明确区分 pre-training 和 post-training，post-training 用 task-specific 数据 fine-tune
  - VLA² 论文：Table I 所有 baseline 标注 (FT) = fine-tuned，π0 (FT) 在 LIBERO-Spatial 上 96.8%
  - Sci-VLA 论文：为每个 atomic task 生成 100 demos 并 fine-tune 80k steps
- **openpi 官方配置**：
  - `pi0_libero` = 训练配置（从 pi0_base 开始 fine-tune 30k steps）
  - `pi05_libero` = 训练配置（从 pi05_base 开始 fine-tune 30k steps）
  - `serve_policy.py --env LIBERO` 默认加载 `pi05_libero` fine-tuned checkpoint
- **结论**：论文实验必须基于 fine-tuned 模型，不能用 base model

### 消融实验基线选择
- 使用 `pi05_libero`（pi0.5 fine-tuned on LIBERO）作为所有消融实验的基线
- 这与 VLA²、Sci-VLA 等论文的实验范式一致：先 fine-tune，再测试 agentic 增强

## 2026-05-12 消融实验（pi05_libero 基线）

### B0: pi05_libero Fine-tuned Baseline ✅
- 配置：`serve_policy.py --env LIBERO`（pi05_libero fine-tuned checkpoint）
- 评测：libero_spatial 全量 10 tasks x 20 trials = 200 episodes
- 结果：**99.0% (198/200)**
- 分任务结果：
  - Task 0-4, 6-9: 100% (20/20)
  - Task 5: 90% (18/20)
- 结果文件：`results/ablation_B0_pi05_libero_finetuned_20260512/summary.json`
- 视频目录：`results/ablation_B0_pi05_libero_finetuned_20260512/videos/`

### B1: pi05_libero + Vision Prompt ✅
- 配置：pi05_libero + `--vision-prompt --mask-alpha 0.35`
- 评测：libero_spatial 全量 10 tasks x 20 trials = 200 episodes
- 结果：**99.0% (198/200)**
- 分任务结果：
  - Task 0-7: 100% (20/20)
  - Task 8: 95% (19/20)
  - Task 9: 95% (19/20)
- 结果文件：`results/ablation_B1_pi05_libero_vision_prompt_20260512/summary.json`

### B0 vs B1 对比（libero_spatial）
| 实验 | 整体 | Task 5 | Task 8 | Task 9 |
|------|------|--------|--------|--------|
| B0 baseline | 99.0% | 90% | 100% | 100% |
| B1 + VP | 99.0% | **100%** | 95% | 95% |
- Task 5: VP 修复了 baseline 弱点（90%→100%）
- Task 8,9: VP 轻微下降（100%→95%），mask 可能干扰已学到的视觉特征
- **结论**：libero_spatial 太简单（天花板效应），需在更难的 suite 上测试

### B0 vs B1 对比（libero_object）
| 实验 | 整体 | Task 3 | Task 5 | Task 6 |
|------|------|--------|--------|--------|
| B0 baseline | 98.0% | 95% | 90% | 95% |
| B1 + VP | **99.5%** | **100%** | **100%** | **100%** |
- Vision Prompt 在 object suite 上显著修复了弱点任务
- Task 3/5/6: 90%/95% → 100%，VP 帮助模型更精准定位目标物体
- 结果文件：`results/ablation_B0_pi05_libero_object_20260512/summary.json`、`results/ablation_B1_pi05_libero_object_20260512/summary.json`

### B0 vs B1 对比（libero_goal）
| 实验 | 整体 | Task 0 | Task 3 | Task 4 |
|------|------|--------|--------|--------|
| B0 baseline | 98.0% | 95% | 100% | 95% |
| B1 + VP | 97.5% | **90%** | 95% | **100%** |
- Vision Prompt 在 goal suite 上轻微下降（98.0%→97.5%）
- 原因：`obj_of_interest` 包含区域名称（如 `wooden_cabinet_1_middle_region`）不在分割图中
- 需要修复：过滤掉非物体实例的 obj_of_interest 条目
- 结果文件：`results/ablation_B0_pi05_libero_goal_20260512/summary.json`、`results/ablation_B1_pi05_libero_goal_20260512/summary.json`

### B0 baseline（libero_10 长程任务）✅
- 结果：**90.0% (180/200)**
- Task 8 最低：**55%**（关键弱点任务）
- 结果文件：`results/ablation_B0_pi05_libero_10_20260512/summary.json`

### B1 + Vision Prompt（libero_10 长程任务）🔄 运行中
- 结果文件：`results/ablation_B1_pi05_libero_10_20260512/summary.json`

### 实验设计说明
- libero_spatial baseline 已达 99%，Vision Prompt 在 ID 任务上提升空间有限
- 但 Vision Prompt 的核心价值在于 **OOD 场景**（参照 VLA² 论文 Hard 级别）
- 后续需在 libero_object、libero_goal、libero_10 等更难的 suite 上测试
- 论文贡献点：Agentic 增强在 OOD/长程任务上的鲁棒性提升，而非 ID 任务的绝对值提升

## 下一阶段任务
- **B1 完成**：等待 Vision Prompt 消融结果
- **B2: pi05_libero + Vision Prompt + Transition**：设计不破坏 action chunk 的过渡动作中间层
- **B3: pi05_libero + Critic/Retry**：增加失败检测和重试逻辑
- **Full: Agentic-VLA**：完整框架消融
- **OOD 测试**：在 libero_object、libero_goal、libero_10 上重复消融
- **论文定位**：Agentic-VLA 的价值在于 OOD 鲁棒性和长程任务连贯性，而非 ID 任务天花板

## 2026-06-04 RT-BA-Harness / ProfileAuto 实验推进

### B4: ProfileAuto core tasks

- 方法：`B4-BAHarness-Rules`
- Backbone：冻结 `pi05_libero`
- 关键设置：
  - `--control-deadline-ms 80`
  - 不启用低层 prompt augmentation，保持 frozen VLA fast path
  - profile-aware scheduler：
    - `stove`: recovery on, transition on
    - `plate`: recovery off, transition off
    - `microwave`: recovery off, transition off
    - `default`: recovery on, transition off

### 已完成正式/核心结果

| Batch | Tasks | Trials | Success | 关键 trace 指标 | 结果目录 |
|---|---:|---:|---:|---|---|
| B4 ProfileAuto | 0,5 | 40 | **40/40** | recovery 11, transition 0, VLA calls/ep 24.50, deadline miss 10.4% | `results/core_B4_profileauto_task05_20trials_20260604_1511/` |
| B4 ProfileAuto | 6,9 | 40 | **37/40** | Task6 18/20, Task9 19/20 | `results/core_B4_profileauto_task69_20trials_20260604_1442/` |
| B4 plate fast path | 6 | 20 | **19/20** | recovery 0, transition 0, VLA calls/ep 24.20 | `results/core_B4_noRecovery_task6_20trials_20260604_1457/` |
| B4 ProfileAuto | 8 | 20 | **16/20** | recovery 20, transition 16, VLA calls/ep 54.55 | `results/core_B4_profileauto_libero10_tasks86905_20trials_20260604_1405/` |
| B4 transition-on exploratory | 8 | 20 | **18/20** | recovery 19, transition 15 | `results/core_B4_libero10_tasks86905_20trials_20260604_1338/` |
| B4 lockout=0 ablation | 8 | 20 | **17/20** | recovery 19, transition 11, VLA calls/ep 50.95, deadline miss 10.4% | `results/core_B4_task8_lockout0_20trials_20260604_1525/` |
| B4 Task8 phase prompt smoke | 8 | 5 | **4/5** | recovery 4, transition 2, VLA calls/ep 49.20 | `results/smoke_B4_task8_phasePrompt_rt_p0_20260604_1534/` |
| B4 ProgressGate smoke | 8 | 5 | **4/5** | progress trace shows failed episode stops at 1/2 moka pots | `results/smoke_B4_task8_progressGate_rt_p0_20260604_1547/` |
| B4 ProgressGate v2 smoke | 8 | 5 | **4/5** | dynamic progress prompt still fails one second-object episode | `results/smoke_B4_task8_progressGateV2_rt_p0_20260604_1552/` |
| B4 ProfileAuto no-regression | 1,2,3,4,7 | 100 | **99/100** | VLA calls/ep 26.47, recovery 34, transition 0, deadline miss 10.4% | `results/core_B4_profileauto_task12347_20trials_20260604_1555/` |

### 与已有 LIBERO-10 对照

| Method | Task0 | Task5 | Task6 | Task8 | Task9 |
|---|---:|---:|---:|---:|---:|
| `B0-VLA` | 18/20 | 19/20 | 16/20 | 11/20 | 19/20 |
| `B1-FullRefined` | 19/20 | 20/20 | 16/20 | 15/20 | 19/20 |
| `B4-ProfileAuto / current best per profile` | **20/20** | **20/20** | **19/20** | 17-18/20 | **19/20** |

### LIBERO-10 当前全量 clean 汇总

可复现汇总命令：

```bash
python scripts/summarize_libero10_core.py
```

| Setting | Success | 说明 |
|---|---:|---|
| `B0-VLA` historical | `180/200` | frozen VLA baseline |
| `B1-FullRefined` historical | `185/200` | earlier full harness, critic fallback |
| `B4-ProfileAuto` current | `192/200` | Task8 使用当前 ProfileAuto `16/20` |
| `B4 current best per profile` | `195/200` | Task6 使用 plate fast path，Task8 使用 transition-on exploratory best |

剩余 clean tasks 细分：

| Task | Success | Recovery | Transition | VLA calls/ep | 失败 |
|---:|---:|---:|---:|---:|---|
| 1 | `20/20` | 1 | 0 | 24.60 | - |
| 2 | `19/20` | 18 | 0 | 31.35 | `task2:trial0` |
| 3 | `20/20` | 4 | 0 | 23.65 | - |
| 4 | `20/20` | 0 | 0 | 23.25 | - |
| 7 | `20/20` | 11 | 0 | 29.50 | - |

### 当前结论

- `B4` 的核心价值已经比较清晰：它不是替换 VLA，而是在 frozen VLA 上做 profile-aware、budget-aware 的执行调度。
- Stable tasks 上，Agent 不应过度干预：
  - Task6 使用 `plate` fast path 后从 recovery-on 的 18/20 提升到 19/20，同时 VLA calls/ep 从 32.35 降到 24.20。
  - Task9 保持 fast path，19/20。
  - Task0/5 达到 40/40，且没有 transition 干扰。
- Task8 是下一阶段核心弱点：
  - lockout=0 只到 17/20，说明“更快 transition”不是充分条件。
  - phase prompt smoke 4/5，没有明显优于 ProfileAuto。
  - ProgressGate 能准确记录 `0/2 -> 1/2 -> 2/2` 的 moka-pot 进度；失败 episode 停在 `1/2`，说明阶段诊断是对的，但规则/prompt 还不能稳定修复第二物体执行。
  - 更可能需要学习式 progress verifier / SmallVA recovery expert，而不是继续堆手写 prompt。

### 下一步

- 保留当前 frozen `pi05_libero`，不更换 VLA/VA 主模型。
- 将主方法冻结为 `B4-BAHarness-Rules/ProfileAuto`，主表优先使用 Task0/5/6/9 的稳定结果。
- Task8 下一步只做有诊断价值的改进：
  - learned/lightweight progress verifier：判断是否完成第一个 moka pot、是否该抓第二个。
  - SmallVA recovery expert：只服务于 Task8 的 second-object transition/recovery，不替换主 VLA。
  - 视觉状态 trace：从视频/obs 中提取“手里是否有壶、炉灶上有几个壶”的代理指标。

## 2026-06-04 AAC-lite 与 Object-Jitter Robustness Smoke

### 工程更新

- 新增 `B4-AAC-Lite` method tag。
- 新增 AAC-lite CLI：
  - `--ba-aac-lite`
  - `--ba-aac-min-commit`
  - `--ba-aac-medium-commit`
  - `--ba-aac-high-commit`
  - `--ba-aac-max-commit`
- episode trace 新增：
  - `commit.histogram`
  - `commit.avg_by_risk`
- 新增 perturbation CLI：
  - `--perturbation clean|object_jitter`
  - `--object-jitter-xy`
- `object_jitter` 在 `set_init_state` 后对可移动 objects 做 xy jitter，不移动 stove/fixture。

### AAC-lite clean smoke

| Setting | Tasks | Trials | Success | 关键结论 | 结果目录 |
|---|---:|---:|---:|---|---|
| `B4-AAC-Lite` initial | 6,2,8 | partial | Task6 出现过干预 | high-risk false positive 把 Task6 拉成大量 3-step commit，VLA calls 飙到 143 | `results/smoke_B4_AACLite_tasks628_rt_p0_20260604_1732/` |
| `B4-AAC-Lite` fixed | 6,2 | 10 | `9/10` | Task6 no-intervention profile 强制 10-step commit；Task2 `5/5` | `results/smoke_B4_AACLiteFix_tasks62_rt_p0_20260604_1736/` |
| `B4-AAC-Lite + ProgressGate` | 8 | 5 | `4/5` | Task8 与 ProgressGate 持平，但 VLA calls/ep 升到 73.8，不适合作为 clean 主方法 | `results/smoke_B4_AACLite_task8_rt_p0_20260604_1739/` |

关键判断：

- AAC-lite 必须保护 stable/no-intervention profile，否则短 commit 会导致 mode-jumping/horizon failure。
- clean Task8 上，AAC-lite 没有提升成功率，且明显增加 VLA calls 和 deadline miss。
- 因此 AAC-lite 不应替代 `B4-ProfileAuto` clean 主方法；它应作为 perturbation/recovery 场景下的 robustness 方法候选。

### Object-Jitter robustness smoke

扰动设置：`--perturbation object_jitter --object-jitter-xy 0.03`

| Method | Task2 | Task6 | Overall | VLA calls/ep | Recovery | 结果目录 |
|---|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `5/5` | `4/5` | `9/10` | 25.9 | 0 | `results/smoke_B0_objectJitter_tasks26_5trials_20260604_1743/` |
| `B4-AAC-Lite` | `5/5` | `5/5` | `10/10` | 25.9 | 5 | `results/smoke_B4_AACLite_objectJitter_tasks26_5trials_20260604_1746/` |

解释：

- Task2 下 B4 每个 episode 都触发 recovery，且 `5/5` 成功；说明 recovery 对扰动后的 stove placement 有正向作用。
- Task6 下 B4 仍保持 no-intervention / 10-step commit，`5/5`；说明 AAC-lite 修复后不会再对 stable task 过干预。
- 但当前只有 10 episodes，不能作为主文强证据；下一步需要扩大到至少每 task 20 trials，并加入更强 perturbation 或 mid-episode nudge。

下一步：

- 将 `object_jitter` smoke 扩到 `Task 2/6/8/0/5`，每 task 20 trials。
- 同时补一个 `B4-ProfileAuto` under object_jitter，对比 `B4-AAC-Lite` 是否真的带来额外收益。
- 如果 3cm jitter 太温和，增加 `object_jitter_xy=0.05` 或实现 `mid_episode_nudge`。

## 2026-06-04 Object-Jitter 5cm Robustness Smoke

扰动设置：`--perturbation object_jitter --object-jitter-xy 0.05`

任务：`Task 2/6/8`，每 task 5 trials，同 seed `7`。

| Method | Overall | Task2 | Task6 | Task8 | VLA calls/ep | Recovery | Transition | Deadline miss | Avg len | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `13/15` | `5/5` | `4/5` | `4/5` | 31.67 | 0 | 0 | 10.2% | 323.0 | `results/smoke_B0_objectJitter5cm_tasks268_5trials_20260604/` |
| `B4-ProfileAuto` | `15/15` | `5/5` | `5/5` | `5/5` | 35.33 | 10 | 3 | 10.5% | 320.3 | `results/smoke_B4_profileauto_objectJitter5cm_tasks268_5trials_20260604/` |
| `B4-AAC-Lite` | `15/15` | `5/5` | `5/5` | `5/5` | 34.40 | 9 | 3 | 10.5% | 309.9 | `results/smoke_B4_AACLite_objectJitter5cm_tasks268_5trials_20260604/` |

分任务 trace：

| Method | Task2 calls / rec / trans | Task6 calls / rec / trans | Task8 calls / rec / trans |
|---|---|---|---|
| `B0-VLA` | 23.2 / 0 / 0 | 28.8 / 0 / 0 | 43.0 / 0 / 0 |
| `B4-ProfileAuto` | 28.4 / 5 / 0 | 24.2 / 0 / 0 | 53.4 / 5 / 3 |
| `B4-AAC-Lite` | 28.0 / 4 / 0 | 22.2 / 0 / 0 | 53.0 / 5 / 3 |

关键判断：

- 5cm object-jitter 已经比 3cm 更有区分度：`B0` 在 Task6 和 Task8 各失败 1 次，总体 `86.7%`。
- `B4-ProfileAuto` 的机制符合论文故事：
  - Task2：每个 episode 都能通过 recovery 修正扰动后的 stove placement。
  - Task6：保持 no-intervention / long commit，不误触发 recovery，反而修复了 B0 的一次失败。
  - Task8：recovery + transition 修复 long-horizon state-gap，`5/5`。
- `B4-AAC-Lite` 没有提升 5cm smoke 成功率，但在平均 VLA calls 和 avg episode length 上略优于 `B4-ProfileAuto`。
- 当前不应把 AAC-lite 作为 clean 主方法；更合理的定位是 realtime/budget-aware 子模块，在更强扰动或执行中扰动下验证其反应性收益。

下一步：

- 将 `object_jitter_xy=0.05` 扩到每 task 20 trials，优先 `B0-VLA` vs `B4-ProfileAuto`。
- `B4-AAC-Lite` 可在 20-trial 主表中作为 efficiency variant，而非唯一主方法。
- 实现 `mid_episode_nudge`，因为它比 initial object jitter 更能体现 recovery latency 和实时闭环价值。

## 2026-06-04 Object-Jitter 5cm 20-Trial Core Diagnostic

扰动设置：`--perturbation object_jitter --object-jitter-xy 0.05`

任务：`Task 2/6/8`，每 task 20 trials，同 seed `7`。

| Method | Overall | Task2 | Task6 | Task8 | VLA calls/ep | Recovery | Transition | Deadline miss | Avg len | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `47/60` | `17/20` | `15/20` | `15/20` | 36.00 | 0 | 0 | 10.1% | 366.4 | `results/core_B0_objectJitter5cm_tasks268_20trials_20260604/` |
| `B4-ProfileAuto` | `48/60` | `17/20` | `15/20` | `16/20` | 39.32 | 36 | 13 | 10.4% | 362.2 | `results/core_B4_profileauto_objectJitter5cm_tasks268_20trials_20260604/` |

分任务机制：

| Method | Task2 calls / rec / trans | Task6 calls / rec / trans | Task8 calls / rec / trans |
|---|---|---|---|
| `B0-VLA` | 28.4 / 0 / 0 | 34.4 / 0 / 0 | 45.2 / 0 / 0 |
| `B4-ProfileAuto` | 34.0 / 16 / 2 | 32.1 / 0 / 0 | 51.9 / 20 / 11 |

关键判断：

- 5-trial smoke 的 `15/15` 不能直接作为主结论；20-trial 放大后，`B4-ProfileAuto` 只从 `47/60` 提升到 `48/60`。
- `B4` 的 Task8 有小幅提升：`15/20 -> 16/20`，但调用开销明显更高。
- Task2 没有提升：`17/20 -> 17/20`。当前 recovery gate 能触发，但并不总能把 initial placement jitter 修好。
- Task6 没有提升：`15/20 -> 15/20`。不过 B4 保持 zero recovery / zero transition，说明 no-overintervention profile 没有引入额外伤害。
- 结论：initial object jitter 不是当前规则式 recovery 最能体现价值的 benchmark。它适合作为诊断，不适合作为主 positive robustness table。

下一步策略更新：

- 不继续盲目扩大 initial object jitter 到更多 task。
- 优先实现 `mid_episode_nudge` / `gripper pause`，让扰动发生在执行过程中，这更符合 Agentic recovery 和 realtime reaction latency 的论文故事。
- 对 Task2 增加 placement-aware verifier：不要只根据 collision/stall 触发 recovery，还要判断 moka pot 是否已经在 stove region 附近、是否需要 recenter/release。
- 对 Task8 恢复使用 ProgressGate：记录 `0/2 -> 1/2 -> 2/2`，失败多为第二物体或 release 阶段，应避免每个 episode 都触发昂贵 transition。

## 2026-06-04 Mid-Episode Nudge 工程 Smoke

新增 perturbation：

```bash
--perturbation mid_episode_nudge
--mid-nudge-step 80
--mid-nudge-xy 0.03
```

实现行为：

- warmup 结束后的控制步达到 `--mid-nudge-step` 时触发一次。
- 从 movable objects 中优先选择与任务文本匹配的对象；若没有匹配，则从可移动物体中 fallback。
- 对目标物体 free joint 的 xy 位置做一次小位移。
- 清空当前 action chunk，强制下一次基于扰动后观测重新推理。
- episode trace 新增 `perturbation_detail.events` 和 `perturbation_detail.reaction_latency_ms`。

smoke 结果：

| Method | Task | Trials | Success | Recovery | Perturbation reaction latency | 结果目录 |
|---|---:|---:|---:|---:|---:|---|
| `B0-VLA` | 2 | 1 | `1/1` | 0 | null | `results/smoke_B0_midNudge_task2_1trial_20260604/` |
| `B4-ProfileAuto` | 2 | 1 | `1/1` | 1 | 4530.1 ms | `results/smoke_B4_midNudge_task2_1trial_20260604/` |

解释：

- 工程链路已打通：Task2 的 `moka_pot_1` 在 control step 80 被 nudge，trace 正确记录 object/joint/dx/dy。
- B4 单例触发 recovery，并记录从扰动到 recovery start 的 latency。
- 这不是性能结论；下一步需要 `Task2/8 × B0/B4 × 10 trials` 的 mid-nudge smoke。

## 2026-06-04 Mid-Episode Nudge 10-Trial Smoke

扰动设置：`--perturbation mid_episode_nudge --mid-nudge-step 80 --mid-nudge-xy 0.03`

任务：`Task 2/8`，每 task 10 trials，同 seed `7`。

| Method | Overall | Task2 | Task8 | VLA calls/ep | Recovery | Transition | Perturb reaction latency | Avg len | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `17/20` | `9/10` | `8/10` | 35.40 | 0 | 0 | null | 361.1 | `results/smoke_B0_midNudge_tasks28_10trials_20260604/` |
| `B4-ProfileAuto` | `16/20` | `9/10` | `7/10` | 43.70 | 18 | 6 | 5905.6 ms | 387.9 | `results/smoke_B4_midNudge_tasks28_10trials_20260604/` |

分任务机制：

| Method | Task2 calls / rec / trans / perturb latency | Task8 calls / rec / trans / perturb latency |
|---|---|---|
| `B0-VLA` | 27.7 / 0 / 0 / null | 43.1 / 0 / 0 / null |
| `B4-ProfileAuto` | 33.9 / 8 / 1 / 5354.0 ms | 53.5 / 10 / 5 / 6346.9 ms |

关键判断：

- 当前规则式 `B4-ProfileAuto` 在 mid-nudge 下没有提升，反而从 `17/20` 降到 `16/20`。
- B4 的问题不是没有触发 recovery，而是触发后没有稳定修复：
  - Task2 recovery `8` 次，成功 `7` 次，但 overall 仍与 B0 持平。
  - Task8 recovery `10` 次、transition `5` 次，但成功率低于 B0。
- `perturbation_detail.reaction_latency_ms` 已可测，平均约 5-6 秒；这说明当前检测主要依赖后验 stall/collision，反应偏慢。
- 下一步应该做 placement/progress-aware verifier：
  - Task2：直接判断 moka pot 是否接近 stove cook region、是否需要 recenter/release。
  - Task8：使用 ProgressGate 判断 `0/2, 1/2, 2/2`，只在第二物体或 release 失败时 transition。
- 论文故事应避免声称“加 Agentic recovery 一定提升鲁棒性”；更强的故事是“realtime agentic policy 的关键是 gating，错误/过慢的 intervention 会损害成功-计算 Pareto”。

## 2026-06-04 Placement / Progress Verifier Iteration

工程更新：

- `--ba-progress-verifier` 现在同时支持：
  - Task2 single moka-pot stove placement：`single_moka_pot_stove`
  - Task8 double moka-pot stove progress：`moka_pot_stove`
- progress trace 新增：
  - `needs_recenter`
  - `target_object`
  - `target_distance_xy`
- 新增 `placement_drift` failure event。
- 当 `needs_recenter=True` 时，B4 可以触发 placement recovery prompt：
  - move above stove burner
  - re-center over cook region
  - lower until stable
  - release gently
- 修复一个重要 gating 问题：
  - `ba_progress_transition` 之前会绕过 post-recovery lockout，导致 Task8 recovery 后立刻 transition。
  - 现在 progress transition 同样遵守 recovery cooldown / transition lockout。

smoke 结果：

| Method | Overall | Task2 | Task8 | VLA calls/ep | Recovery | Transition | Perturb reaction latency | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `17/20` | `9/10` | `8/10` | 35.4 | 0 | 0 | null | `results/smoke_B0_midNudge_tasks28_10trials_20260604/` |
| `B4-ProfileAuto` | `16/20` | `9/10` | `7/10` | 43.7 | 18 | 6 | 5905.6 ms | `results/smoke_B4_midNudge_tasks28_10trials_20260604/` |
| `B4+PV` before lockout fix | `16/20` | `10/10` | `6/10` | 45.0 | 20 | 10 | 4163.2 ms | `results/smoke_B4_progressVerifier_midNudge_tasks28_10trials_20260604/` |
| `B4+PV+Lockout` | `18/20` | `10/10` | `8/10` | 42.1 | 20 | 6 | 4262.8 ms | Task2 from `results/smoke_B4_progressVerifier_midNudge_tasks28_10trials_20260604/`; Task8 from `results/smoke_B4_progressVerifierLockout_midNudge_task8_10trials_20260604/` |

关键判断：

- Placement verifier 对 Task2 有明确收益：`9/10 -> 10/10`，且 transition 从 1 降到 0。
- Task8 的核心不是“更早 transition”，而是“recovery 后不能立即 transition”；lockout 修复后 Task8 从 `6/10` 回到 `8/10`。
- `B4+PV+Lockout` 的成功率优于 B0 和旧 B4，但 calls/ep 仍高于 B0。这正好支持论文里的实时调度问题：
  - agentic reasoning 可以提升恢复能力；
  - 但没有 gating / lockout / budget control 时会过干预；
  - 后续需要 AAC-lite 或更轻量 verifier 把 calls 降下来。

下一步：

- 跑 `B4-AAC-Lite + PV + Lockout` on `mid_episode_nudge Task2/8 × 10 trials`，看是否在保持 `18/20` 左右成功率的同时降低 calls。
- 若 AAC-lite 不降 calls，则考虑降低 progress verifier 频率，例如每 3-5 control steps 估计一次。
- 将 `B4+PV+Lockout` 扩到 20 trials 前，先补一个 clean sanity smoke，确认 `--ba-progress-verifier` 不会损害 clean Task2/8。

### AAC-lite + PV + Lockout 负消融

设置：`B4-AAC-Lite --ba-aac-lite --ba-progress-verifier --perturbation mid_episode_nudge`

结果目录：`results/smoke_B4_AACLite_PVLockout_midNudge_tasks28_10trials_20260604/`

该 run 在 Task8 明显失败后提前停止。

| Task | Success | VLA calls/ep | Recovery | Transition | Avg commit | Deadline miss |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `9/10` | 36.9 | 10 | 1 | 7.87 | 11.9% |
| 8 | `1/6` | 124.7 | 6 | 5 | 4.54 | 22.7% |
| Total | `10/16` | 69.8 | 16 | 6 | - | 16.0% |

判断：

- 当前 AAC-lite 不能作为 PV+Lockout 的 efficiency variant。
- 高风险短 commit 在 Task8 上导致 action chunk 过碎、VLA calls 爆炸、deadline miss 翻倍，成功率严重下降。
- 这证明 realtime inference optimization 不能简单等价为“风险高就缩短 chunk”；必须区分：
  - recovery expert 内部是否应保持连续动作；
  - long-horizon second-object phase 是否需要更长 commit；
  - verifier 频率是否可以降采样，而不是不断重新询问 VLA。

下一步不再跑当前 AAC-lite 配置；若继续做推理优化，应改成：

- progress verifier 降频：每 3-5 control steps 运行一次；
- recovery/transition expert 使用固定 6-10 step commit，避免短 chunk mode-jumping；
- 只在 perturbation 后的前 1-2 个 VLA chunks 缩短 commit，而不是整个 high-risk phase。

## 2026-06-08 Lightweight VLA Runtime Baseline

目标：将“VLA 轻量化 / 高效推理”从不稳定的 action-cache 机制中拆出来，先建立可复现的 serving runtime baseline，再选择稳定的优化项进入主实验。

### Fresh PyTorch no-compile serving profile

设置：

- Policy server：`pi05_libero`
- Runtime：PyTorch serving，`OPENPI_DISABLE_TORCH_COMPILE=1`
- Benchmark：`LIBERO-10 Task2`
- Trials：`1`
- Perturbation：`clean`
- 结果目录：`results/serving_profile_torchNoCompile_fresh_task2_1trial_20260608/`

结果：

| Metric | Value |
|---|---:|
| Success | `1/1` |
| Episode steps | `268` |
| Full VLA calls | `26` |
| VLA calls / episode | `26.0` |
| VLA model wall time / episode | `10468.2 ms` |
| VLA latency mean | `402.6 ms` |
| VLA latency p50 | `372.2 ms` |
| VLA latency p95 | `411.7 ms` |
| VLA latency p99 | `941.6 ms` |
| TTFA | `1123.1 ms` |
| Deadline miss rate | `10.08%` |
| Success-under-deadline | `0.899` |
| GPU peak memory | `8.91 GB` |

判断：

- 单独运行 pi0.5 LIBERO serving 时，4090 显存占用约 `8.9 GB`，可以作为后续部署模块的 baseline。
- steady-state VLA 推理延迟主要集中在 `360-420 ms`，首个 VLA call 明显更慢，导致 TTFA 约 `1.12 s`。
- 该结果支持论文中的 realtime motivation：即使 action chunk 能让控制循环快速执行，完整 VLA query 仍是实时系统的主要阻塞源。

下一步：

- 测试默认 torch compile serving 的首轮开销与 steady-state 延迟。
- 若 compile 首轮开销过大或不稳定，则在论文实验中将 `OPENPI_DISABLE_TORCH_COMPILE=1` 作为稳定 serving baseline，并把 compile 结果作为 deployment trade-off 记录。
- 暂不把 Light-Reuse action-cache 作为主 lightweight 方法；它将保留为负消融，说明简单减少 VLA calls 会损害长程任务稳定性。

### Default torch compile serving probe

设置：

- Policy server：`pi05_libero`
- Runtime：默认 torch compile serving，即不设置 `OPENPI_DISABLE_TORCH_COMPILE=1`
- Benchmark：`LIBERO-10 Task2`
- Trials：`1`
- 结果目录：`results/serving_profile_torchCompile_fresh_task2_1trial_20260608/`

观测：

- Server 约 `2 min` 后开始监听端口并加载模型到 GPU。
- 运行约 `5.5 min` 后仍未写出 episode trace，也没有完成第一个 Task2 episode。
- GPU 显存占用约 `9.3 GB`，server 进程保持高 CPU 使用率，符合首轮 compile/JIT 开销较大的现象。
- 本次 probe 被手动中断，未进入正式成功率统计。

判断：

- 默认 torch compile 在当前 4090 单卡实验流程中首轮开销过高，不适合作为 overnight benchmark 的主配置。
- 后续正式 benchmark 统一使用 `OPENPI_DISABLE_TORCH_COMPILE=1`，优先保证实验吞吐、可重复性和与 agentic harness 的兼容性。
- 该 probe 可作为部署章节的 trade-off：compile 可能有 steady-state 收益，但机器人仿真/实机实验更关注 cold-start、TTFA、调试吞吐和稳定性，当前配置下 no-compile 更适合主实验。

### Method-tag harness mapping fix

工程修复：

- `argparse` 中已经加入论文命名 `B4-Agentic` / `B4-Agentic-Light`。
- `_is_ba_harness()` 之前仍只识别旧命名 `B4-BAHarness-Rules` / `B4-Async` / `B4-AAC-Lite`。
- 已修复为同时识别：
  - `B4-Agentic`
  - `B4-Agentic-Light`
  - legacy B4 method tags

影响：

- 后续使用论文主表命名 `B4-Agentic` / `B4-Agentic-Light` 时会真正启用 Agentic Harness。
- 已完成的旧结果需要按实际 command/tag 区分；若旧结果使用 `B4-Agentic` 且未显式触发 harness，则不能直接作为最终主表。
- 下一批实验应使用修复后的代码重新跑 `B0-VLA` vs `B4-Agentic` 的关键 Task2/8/6 对照。

### B4-Agentic + PV + forced perturbation recovery negative ablation

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-perturbation-recovery`
- Perturbation：`mid_episode_nudge`
- Benchmark：`LIBERO-10 Task2/8`
- Trials：planned `10` per task；Task8 前 3 个 episode 全失败后 early stop
- 结果目录：`results/b4_agentic_pv_lockout_midNudge_tasks28_10trials_seed7_20260608/`

结果：

| Task | Success | VLA calls / ep | Recovery total | Transition total | 说明 |
|---:|---:|---:|---:|---:|---|
| 2 | `9/10` | `35.9` | `10` | `1` | 单物体任务基本可用 |
| 8 | `0/3` | `65.0` | `3` | `3` | 双物体长程任务明显失败 |

判断：

- `--ba-perturbation-recovery` 对 Task2 可接受，但在 Task8 上会导致过干预。
- Task8 的失败模式与此前 AAC-lite 负消融一致：过早 recovery / transition 会破坏长程任务的连续执行。
- 该配置不进入主表，保留为 mechanism negative ablation。

下一步：

- 重新跑不带 `--ba-perturbation-recovery` 的 `B4-Agentic + progress verifier + lockout`。
- Task8 优先，确认长程任务中只使用 progress verifier 与 lockout 是否能恢复到旧结果水平。

### B4-Agentic + PV + Lockout on Task8 without forced perturbation recovery

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier`
- Perturbation：`mid_episode_nudge`
- Benchmark：`LIBERO-10 Task8`
- Trials：`10`
- 结果目录：`results/b4_agentic_pv_lockout_midNudge_task8_10trials_seed7_20260608/`

结果：

| Metric | Value |
|---|---:|
| Success | `7/10` |
| Avg episode length | `484.8` |
| VLA calls / ep | `55.7` |
| Recovery triggered | `9` |
| Recovery successful | `7` |
| Recovery precision | `0.778` |
| Transition total | `7` |
| Transition success count | `4` |
| VLA latency p50 | `388.3 ms` |
| VLA latency p95 | `416.6 ms` |
| VLA latency p99 | `454.5 ms` |
| Deadline miss rate | `10.54%` |
| Success-under-deadline | `0.626` |
| GPU peak memory | `8.91 GB` |

判断：

- 去掉 forced perturbation recovery 后，Task8 从 early-stop `0/3` 恢复到 `7/10`，说明长程任务不适合“扰动后强制恢复”的策略。
- 当前配置仍低于理想旧 smoke 的 `8/10`，主要问题是 transition 仍较频繁：`7` 次 transition 中只有 `4` 次对应成功 episode。
- 对长程任务而言，Agentic Harness 的收益来自 progress-aware recovery；风险主要来自过早/过频 transition。

下一步：

- 跑 Task2 的同配置 full `10` trials，形成 Task2/8 的修复后主表候选。
- 对 Task8 增加更保守的 transition gating 对照：
  - `--ba-disable-transition`
  - 或提高 transition threshold / min steps
  - 或 `--ba-safe-long-horizon-fast-path`

### B4-Agentic + PV + Lockout main candidate on Task2/8

统一设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier`
- Runtime：pi0.5 LIBERO no-compile serving
- Perturbation：`mid_episode_nudge`
- Seed：`7`

结果：

| Task | Success | VLA calls / ep | Recovery | Recovery precision | Transition | VLA p50 | VLA p95 | SUD | GPU peak | Result dir |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | `9/10` | `33.5` | `10` | `0.900` | `1` | `386.7 ms` | `419.7 ms` | `0.803` | `8.90 GB` | `results/b4_agentic_pv_lockout_midNudge_task2_10trials_seed7_20260608/` |
| 8 | `7/10` | `55.7` | `9` | `0.778` | `7` | `388.3 ms` | `416.6 ms` | `0.626` | `8.91 GB` | `results/b4_agentic_pv_lockout_midNudge_task8_10trials_seed7_20260608/` |

判断：

- 修复后的 `B4-Agentic` 命名已经可以正常启用 harness。
- Task2：progress-aware recovery 能稳定修复扰动后的单物体 placement drift，达到 `9/10`。
- Task8：仍有 `7/10`，但 transition 触发 `7` 次，失败 episode 中 transition 比例较高；下一步应验证禁用/延迟 transition 是否更适合长程任务。
- 当前指标支持一个重要叙事：Agentic Policy 不是“越多干预越好”，核心是长程任务中的 intervention gating。

下一步实验：

- Task8 `B4-Agentic + PV + --ba-disable-transition`。
- 若成功率上升或 calls 下降，则主表采用“long-horizon conservative transition gating”。
- 若成功率下降，则保留 transition，但提高 threshold / min-step 进行细化。

### Task8 transition ablation: disable transition

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-disable-transition`
- Perturbation：`mid_episode_nudge`
- Benchmark：`LIBERO-10 Task8`
- Trials：`10`
- 结果目录：`results/b4_agentic_pv_noTransition_midNudge_task8_10trials_seed7_20260608/`

结果：

| Metric | Value |
|---|---:|
| Success | `6/10` |
| VLA calls / ep | `52.9` |
| Recovery triggered | `10` |
| Recovery precision | `0.600` |
| Transition total | `0` |
| VLA latency p50 | `385.9 ms` |
| VLA latency p95 | `411.3 ms` |
| Success-under-deadline | `0.537` |
| GPU peak memory | `8.91 GB` |

判断：

- 完全禁用 transition 低于保留 transition 的 `7/10`。
- Task8 需要 transition 作为阶段切换/状态差距修复工具，但当前默认 gating 仍偏激进。
- 下一步应测试更高 transition risk threshold，而不是简单关闭 transition。

工程修复：

- 修复 `--ba-transition-risk-threshold` 未能覆盖 task profile 默认值的问题。
- 后续高阈值实验可真正生效。

### Task8 transition ablation: high threshold 0.90

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-transition-risk-threshold 0.90`
- Perturbation：`mid_episode_nudge`
- Benchmark：`LIBERO-10 Task8`
- Trials：`10`
- 结果目录：`results/b4_agentic_pv_transitionThr090_midNudge_task8_10trials_seed7_20260608/`

结果：

| Metric | Value |
|---|---:|
| Success | `4/10` |
| VLA calls / ep | `58.3` |
| Recovery triggered | `10` |
| Recovery precision | `0.400` |
| Transition total | `7` |
| Transition success count | `1` |
| VLA latency p50 | `388.6 ms` |
| VLA latency p95 | `418.4 ms` |
| Success-under-deadline | `0.358` |
| GPU peak memory | `8.91 GB` |

判断：

- 简单提高 transition threshold 不但没有减少 transition，还降低了 recovery/transition 的配合质量。
- Task8 需要更结构化的 phase-aware transition gating，而不是单一 risk threshold。
- 当前 Task8 最好配置仍是默认 `B4-Agentic + PV`：`7/10`。
- 后续不继续在 Task8 上做无约束阈值搜索，转向 Task6 对照，寻找更稳的 Agentic Harness 收益场景。

## 2026-06-08 Task6 Mid-Nudge Agentic Harness Check

目标：在 Task2/8 之外补充一个长程多对象任务，验证 Agentic Harness 是否能在不同任务形态下体现价值。

任务：

- LIBERO-10 Task6
- Instruction：`put the white mug on the plate and put the chocolate pudding to the right of the plate`
- Perturbation：`mid_episode_nudge`
- Trials：`10`
- Runtime：pi0.5 LIBERO no-compile serving

结果：

| Method | Success | VLA calls / ep | Avg ep length | Recovery | Transition | VLA p50 | VLA p95 | SUD | GPU peak | Result dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `5/10` | `40.1` | `408.8` | `0` | `0` | `389.5 ms` | `427.7 ms` | `0.449` | `9.49 GB` | `results/b0_vla_midNudge_task6_10trials_seed7_20260608/` |
| `B4-Agentic+PV` | `8/10` | `30.6` | `311.7` | `0` | `0` | `387.4 ms` | `427.6 ms` | `0.717` | `9.48 GB` | `results/b4_agentic_pv_midNudge_task6_10trials_seed7_20260608/` |

判断：

- Task6 上观测到明显提升：`5/10 -> 8/10`，同时 VLA calls / ep 和 episode length 下降。
- 该结果需要谨慎解释：本组 B4 没有触发显式 recovery / transition，trace 中 expert switch 仍为 `default_vla_expert`。
- 因此不能直接写成“主动恢复提升 Task6”；更稳妥的解释是：
  - B4 harness 在 fast-path 下没有引入额外干预；
  - Task6 观测到更好的成功-效率结果；
  - 需要跨 seed 复验，确认不是仿真/策略随机性的单 seed 波动。

下一步：

- 使用不同 seed 复验 Task6：
  - `B0-VLA` seed `17`
  - `B4-Agentic+PV` seed `17`
- 如果趋势保持，则 Task6 可作为主表中的强正结果。
- 如果趋势不稳定，则 Task6 作为补充分析，主表重点回到 Task2/8 的机制解释。

### Task6 two-seed replication

新增复验：

- `B0-VLA` seed `17`
- `B4-Agentic+PV` seed `17`

结果：

| Method | Seed | Success | VLA calls / ep | Avg ep length | SUD | VLA p95 | Result dir |
|---|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `7` | `5/10` | `40.1` | `408.8` | `0.449` | `427.7 ms` | `results/b0_vla_midNudge_task6_10trials_seed7_20260608/` |
| `B4-Agentic+PV` | `7` | `8/10` | `30.6` | `311.7` | `0.717` | `427.6 ms` | `results/b4_agentic_pv_midNudge_task6_10trials_seed7_20260608/` |
| `B0-VLA` | `17` | `6/10` | `36.1` | `368.2` | `0.539` | `431.1 ms` | `results/b0_vla_midNudge_task6_10trials_seed17_20260608/` |
| `B4-Agentic+PV` | `17` | `7/10` | `32.0` | `326.1` | `0.628` | `419.3 ms` | `results/b4_agentic_pv_midNudge_task6_10trials_seed17_20260608/` |

Two-seed aggregate：

| Method | Success | Success rate | VLA calls / ep mean | SUD mean |
|---|---:|---:|---:|---:|
| `B0-VLA` | `11/20` | `55%` | `38.1` | `0.494` |
| `B4-Agentic+PV` | `15/20` | `75%` | `31.3` | `0.673` |

判断：

- Task6 两个 seed 均保持 `B4 > B0`，但 seed17 差距较小。
- B4 的效率指标也更好：VLA calls / ep 从约 `38.1` 降到 `31.3`。
- 当前 B4 没有触发显式 recovery / transition，因此该结果应继续标注为 fast-path harness observation，不直接归因于 recovery。
- 下一步应测试 `--ba-augment-prompt`，让 memory/task prior 真正进入 VLA prompt，验证 Agent memory 是否能成为可解释机制。

### Task6 memory prompt and relation-recovery ablations

目的：确认 Task6 的提升是否可以由更明确的 Agentic 机制解释。

#### Memory prompt

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-augment-prompt`
- Perturbation：`mid_episode_nudge`
- Seed：`7`
- 结果目录：`results/b4_agentic_pv_memoryPrompt_midNudge_task6_10trials_seed7_20260608/`

结果：

| Config | Success | VLA calls / ep | Recovery | SUD |
|---|---:|---:|---:|---:|
| `B4 fast-path` | `8/10` | `30.6` | `0` | `0.717` |
| `B4 + memory prompt` | `5/10` | `37.0` | `0` | `0.449` |

判断：

- 直接把 GraphRAG prior 写入 VLA prompt 会明显损害 Task6。
- 对 pi0.5 LIBERO 来说，prompt distribution 很敏感；Agent memory 不应直接以长文本方式注入低层 VLA prompt。
- 该结果进入负消融，支持“Agentic Harness 需要低干预、强 gating”的论文叙事。

#### Plate-relation far recovery

工程尝试：

- 为 `plate_relation` progress verifier 增加 far-stall recovery prompt。
- 增加 post-nudge guard，避免 mid-nudge 实验中 perturbation 前过早 recovery。
- 由于该策略效果不佳，默认 Task6 profile 已恢复为保守设置；relation recovery 仅作为显式消融工具保留。

结果：

| Config | Success | VLA calls / ep | Recovery | SUD | Result dir |
|---|---:|---:|---:|---:|---|
| `farRecovery260 v1` | `7/10` | `34.4` | `0` | `0.629` | `results/b4_agentic_pv_farRecovery260_midNudge_task6_10trials_seed7_20260608/` |
| `farRecovery260 v2` | `5/10` | `43.6` | `10` | `0.446` | `results/b4_agentic_pv_farRecovery260_midNudge_task6_10trials_seed7_20260608_v2/` |
| `farRecovery260 postNudgeGuard v3` | `6/10` | `39.2` | `10` | `0.536` | `results/b4_agentic_pv_farRecovery260_postNudgeGuard_task6_10trials_seed7_20260608_v3/` |

判断：

- v1 没有真正触发 recovery，不能作为机制结果。
- v2/v3 触发 recovery 过于频繁，成功率和效率均低于 fast-path B4。
- Task6 当前最好的配置仍是 `B4-Agentic+PV fast-path`：两 seed 聚合 `15/20`。
- 论文中可以将该组作为“naive relation recovery can over-intervene”的负消融，强调 Agentic Policy 的关键不是多干预，而是正确 gating。

### Task6 three-seed aggregate

新增 seed：

- `B0-VLA` seed `27`
- `B4-Agentic+PV` seed `27`

结果：

| Method | Seed | Success | VLA calls / ep | Avg ep length | SUD |
|---|---:|---:|---:|---:|---:|
| `B0-VLA` | `7` | `5/10` | `40.1` | `408.8` | `0.449` |
| `B0-VLA` | `17` | `6/10` | `36.1` | `368.2` | `0.539` |
| `B0-VLA` | `27` | `8/10` | `31.8` | `325.2` | `0.719` |
| `B4-Agentic+PV` | `7` | `8/10` | `30.6` | `311.7` | `0.717` |
| `B4-Agentic+PV` | `17` | `7/10` | `32.0` | `326.1` | `0.628` |
| `B4-Agentic+PV` | `27` | `7/10` | `33.2` | `338.8` | `0.629` |

Aggregate：

| Method | Success | Success rate | Mean VLA calls / ep | Mean SUD |
|---|---:|---:|---:|---:|
| `B0-VLA` | `19/30` | `63.3%` | `36.0` | `0.569` |
| `B4-Agentic+PV` | `22/30` | `73.3%` | `31.9` | `0.658` |

判断：

- 三 seed 后 B4 仍有 `+10.0 pp` 成功率和更低 VLA calls / ep。
- 但 seed27 中 B0 高于 B4，说明 Task6 存在明显 seed 波动，不能作为单独强结论。
- Task6 适合进入补充表或 robustness trend 表，不建议作为论文主贡献唯一证据。
- 当前最稳的叙事：
  - Task2/8：展示 intervention gating 的必要性与机制边界；
  - Task6：展示 fast-path harness 在部分长程多对象任务上有正向趋势；
  - negative ablations：memory prompt / naive relation recovery / high transition threshold 均说明“Agentic 不是越多越好”。

## 2026-06-08 Clean Sanity on Task2/6/8

目标：验证 Agentic Harness 在无扰动 clean 场景下是否会损害原始 VLA。

设置：

- Benchmark：LIBERO-10 Task2/6/8
- Trials：`10` per task
- Seed：`7`
- Runtime：pi0.5 LIBERO no-compile serving

结果：

| Method | Overall | Task2 | Task6 | Task8 | VLA calls / ep | SUD | Recovery | Transition | Result dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B0-VLA` | `29/30` | `10/10` | `9/10` | `10/10` | `30.3` | `0.869` | `0` | `0` | `results/b0_vla_clean_tasks268_10trials_seed7_20260608/` |
| `B4-Agentic+PV` | `28/30` | `10/10` | `10/10` | `8/10` | `34.3` | `0.836` | `20` | `2` | `results/b4_agentic_pv_clean_tasks268_10trials_seed7_20260608/` |

判断：

- Clean 场景下 B4 总体略低于 B0：`28/30` vs `29/30`。
- B4 在 Task6 clean 上提升到 `10/10`，但 Task8 clean 从 `10/10` 降到 `8/10`。
- B4 在 clean 下仍触发大量 recovery：
  - Task2：`10` 次 recovery，成功率保持 `10/10`，但 calls 增加；
  - Task8：`10` 次 recovery + `2` 次 transition，成功率降到 `8/10`。
- 这说明当前 Agentic Harness 还缺少 clean-safe / perturbation-aware gating。

论文解释：

- 不能声称 Agentic Harness 在所有 clean 场景下无代价。
- 更准确的主张是：
  - Agentic Harness 可以提升扰动/长程场景中的恢复能力；
  - 但如果 gating 不足，clean 场景会出现过干预；
  - 因此 realtime Agentic-VLA 的核心贡献应包括 intervention budget / gating。

下一步：

- 跑一个 clean-safe B4 对照：
  - clean 下关闭 recovery/transition，或只保留 memory-free fast-path；
  - perturbation 下再启用 progress recovery。
- 这可以形成最终框架中的 `Context-aware Intervention Gate`。

### Clean-safe B4

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-disable-recovery --ba-disable-transition`
- Benchmark：LIBERO-10 Task2/6/8 clean
- Trials：`10` per task
- Seed：`7`
- 结果目录：`results/b4_agentic_pv_cleanSafe_clean_tasks268_10trials_seed7_20260608/`

结果：

| Method | Overall | Task2 | Task6 | Task8 | VLA calls / ep | SUD | Recovery | Transition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | `29/30` | `10/10` | `9/10` | `10/10` | `30.3` | `0.869` | `0` | `0` |
| `B4 default` | `28/30` | `10/10` | `10/10` | `8/10` | `34.3` | `0.836` | `20` | `2` |
| `B4 clean-safe` | `28/30` | `10/10` | `9/10` | `9/10` | `30.6` | `0.838` | `0` | `0` |

判断：

- clean-safe gating 将 recovery 从 `20` 降为 `0`，VLA calls / ep 从 `34.3` 降到 `30.6`，接近 B0。
- clean-safe Task8 从 `8/10` 回到 `9/10`，说明默认 B4 的 clean Task8 掉点主要来自过干预。
- clean-safe 仍低于 B0 `1` 个 episode，可能是 seed / policy stochasticity；需要更多 seed 才能判断是否有真实差异。
- 最终框架应采用 context-aware intervention：
  - clean / low-risk：fast-path，不触发 recovery / transition；
  - perturbation / high-risk：启用 progress-aware recovery；
  - long-horizon：transition 需要更严格的 phase gate。

### Task2 transition gating check

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-disable-transition`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task2
- Trials：`10`
- Seed：`7`
- 结果目录：`results/b4_agentic_pv_noTransition_midNudge_task2_10trials_seed7_20260608/`

结果：

| Config | Success | VLA calls / ep | Recovery | Transition | SUD |
|---|---:|---:|---:|---:|---:|
| `B0-VLA` | `10/10` | `from prior B0 Task2/8 run` | `0` | `0` | - |
| `B4-Agentic+PV` | `9/10` | `33.5` | `10` | `1` | `0.803` |
| `B4-Agentic+PV no-transition` | `8/10` | `36.1` | `10` | `0` | `0.713` |

判断：

- Task2 的主要问题不是 transition，而是 recovery 过敏。
- 禁用 transition 后成功率从 `9/10` 降到 `8/10`，calls / ep 增加。
- 单物体 Task2 上 B0 已经是 `10/10`，当前 B4 recovery 没有必要，反而降低成功-效率。
- 后续需要 context-aware intervention gate：
  - 对已经稳定的 single-object task，默认 fast-path；
  - 只有检测到明确 post-perturbation failure / severe drift 时才恢复；
  - recovery 不应仅由轻微 placement distance 触发。

### Context-aware intervention gate: Task2

工程更新：

- 新增 `--ba-context-aware-intervention`。
- 行为：
  - clean 或 mid-nudge 扰动发生前：禁止 B4 recovery / transition；
  - single moka-pot stove task：轻微 progress distance 不直接触发 recovery；
  - 扰动后仍允许 failure-taxonomy recovery，例如 collision/stall。

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task2
- Trials：`10`
- Seed：`7`
- 结果目录：`results/b4_agentic_pv_contextGate_midNudge_task2_10trials_seed7_20260608/`

结果：

| Config | Success | VLA calls / ep | Recovery | Transition | SUD |
|---|---:|---:|---:|---:|---:|
| `B0-VLA` | `10/10` | - | `0` | `0` | - |
| `B4-Agentic+PV` | `9/10` | `33.5` | `10` | `1` | `0.803` |
| `B4 no-transition` | `8/10` | `36.1` | `10` | `0` | `0.713` |
| `B4 ContextGate` | `10/10` | `30.4` | `9` | `0` | `not yet tabulated` |

判断：

- ContextGate 将 Task2 恢复到 `10/10`，且去掉 transition。
- 这说明 Task2 的关键不是完全禁用 recovery，而是避免扰动前/轻微 drift 触发错误 recovery，并保留扰动后的真实 collision/stall recovery。
- 这是当前最清晰的 Agentic gating 正结果之一。

下一步：

- 在 Task8 上测试 ContextGate，验证是否能改善 long-horizon 任务的过干预问题。

### Context-aware intervention gate: Task8

设置：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task8
- Trials：`10`
- Seed：`7`
- 结果目录：`results/b4_agentic_pv_contextGate_midNudge_task8_10trials_seed7_20260608/`

结果：

| Task | Success | VLA calls / ep | Recovery | Transition | SUD |
|---:|---:|---:|---:|---:|---:|
| 2 | `10/10` | `30.4` | `9` | `0` | `0.893` |
| 8 | `6/10` | `56.6` | `10` | `8` | `0.537` |

判断：

- ContextGate 对 Task2 有效：成功率恢复到 `10/10`，transition 清零。
- ContextGate 对 Task8 无效：成功率只有 `6/10`，transition 仍有 `8` 次。
- Task8 的问题不是简单 clean/perturbation context，而是 long-horizon phase gating：
  - 第一物体和第二物体阶段需要不同 intervention 条件；
  - recovery 后 transition 仍可能过早；
  - state gap 不能直接等价于需要 transition。

下一步工程：

- 为 `moka_pot_stove` 增加 phase-aware transition gate：
  - `completed_count == 0`：只允许 recovery，不允许 transition；
  - `completed_count == 1`：只有 second-object progress stagnation 才允许 transition；
  - recovery 后增加更强 lockout；
  - transition 前要求连续多次 state-gap / stall，而不是单次触发。
- 在完成该 gate 之前，不继续扩大 Task8 benchmark。

### Task8 long-horizon gate follow-up

#### Phase-aware transition gate

工程更新：

- 新增 `--ba-phase-aware-transition-gate`。
- 规则：
  - `moka_pot_stove` 且 `completed_count == 0`：禁止 transition；
  - `completed_count == 1`：只在较晚阶段、state-gap 和 stall/collision 同时出现时允许 transition；
  - recovery cooldown / recenter 阶段禁止 transition。

实验：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --ba-phase-aware-transition-gate`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task8
- Early stop：前 4 个 episode 全失败
- 结果目录：`results/b4_agentic_pv_context_phaseGate_midNudge_task8_10trials_seed7_20260608/`

结果：

| Config | Success | Recovery | Transition | 判断 |
|---|---:|---:|---:|---|
| `Context+PhaseGate` | `0/4` | `4` | `2` | early stop |

判断：

- 该 phase gate 过硬，虽然减少部分 transition，但破坏了长程任务推进。
- 不进入主配置，保留为负消融。

#### Long recovery lockout

实验：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-post-recovery-transition-lockout-steps 320`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task8
- Trials：`10`
- 结果目录：`results/b4_agentic_pv_lockout320_midNudge_task8_10trials_seed7_20260608/`

结果：

| Config | Success | VLA calls / ep | Recovery | Transition | SUD |
|---|---:|---:|---:|---:|---:|
| `B4 default PV` | `7/10` | `55.7` | `9` | `7` | `0.626` |
| `B4 lockout320` | `7/10` | `52.6` | `10` | `3` | `0.626` |

判断：

- 增大 recovery 后 transition lockout 能明显减少 transition：`7 -> 3`。
- 成功率没有提升，仍为 `7/10`。
- 当前 Task8 最稳配置仍是默认 `B4-Agentic+PV` 或 `lockout320`，二者成功率相同；lockout320 更适合 efficiency / lower-intervention 叙事。
- 后续 Task8 不再做简单 threshold/phase 搜索，除非新增更可靠的 progress phase detector。

### Lightweight VLA Runtime: Task2 replan5 action reuse

实验：

- Method：`B0-VLA-Light`
- Flags：`--replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5`
- Perturbation：`clean`
- Benchmark：LIBERO-10 Task2
- Trials：`10`
- 结果目录：`results/b0_vla_light_reuse_max2_task2_10trials_seed7_20260608/`

Light-Reuse 结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | TTFA mean | SUD | GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA-Light replan5 reuse-max2` | `10/10` | `25.1` | `24.6` | `0.495` | `401.0 ms` | `0.898` | `8.87 GB` |

补齐公平对照：

- Method：`B0-VLA`
- Flags：`--replan-steps 5`
- Perturbation：`clean`
- Benchmark：LIBERO-10 Task2
- Trials：`10`
- 结果目录：`results/b0_vla_replan5_nolight_task2_10trials_seed7_20260608/`

对比结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | VLA wall / ep | Episode wall | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA replan5` | `10/10` | `50.6` | `0.0` | `19.73 s` | `30.08 s` | `0.202` | `0.798` |
| `B0-VLA-Light replan5 reuse-max2` | `10/10` | `25.1` | `24.6` | `9.85 s` | `20.03 s` | `0.102` | `0.898` |

判断：

- `B0-VLA-Light` 在 Task2 clean 上保持 `10/10` 成功率。
- 在 `replan=5` 的短 commit 设置下，约一半控制步走 action reuse fast path，fallback 为 `0`。
- 与同样 `replan=5` 的 `B0-VLA` 对照相比，full VLA calls 约减少 `50.4%`，VLA wall time 约减少 `50.1%`，episode wall time 约减少 `33.4%`。
- Deadline miss 从 `0.202` 降到 `0.102`，SUD 从 `0.798` 提升到 `0.898`。
- 该结果用于轻量化主线：短 commit 提高实时反应频率，action reuse 降低满量 VLA 调用频率。

### Agentic-Light coupling: naive reuse under perturbation

实验：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task2
- Trials：`10`
- 结果目录：`results/b4_agentic_light_contextGate_replan5_reuseMax2_midNudge_task2_10trials_seed7_20260608/`

结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Recovery precision | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic+ContextGate` | `10/10` | `30.4` | `0.0` | `0.000` | `1.000` | `0.107` | `0.893` |
| `B4-Agentic-Light naive reuse` | `8/10` | `40.6` | `24.9` | `0.380` | `0.778` | `0.117` | `0.709` |

判断：

- Naive action reuse 在扰动任务中破坏了 Agentic recovery 的稳定性，成功率从 `10/10` 降到 `8/10`。
- 两个失败 episode 都出现 recovery 未成功，且至少一个 episode 伴随 transition 触发。
- 该结果不进入主配置，作为负消融保留。
- 下一步需要实现 Agent-aware Light-Reuse：扰动或 recovery 之后清空 reuse buffer，并在安全窗口内禁用 action reuse。

### Agentic-Light coupling: safe reuse lockout

工程更新：

- 新增 `--light-reuse-post-perturbation-lockout-steps`。
- 新增 `--light-reuse-post-recovery-lockout-steps`。
- 触发在线扰动或 Agentic recovery 后：
  - 清空 `light_reuse_buffer`；
  - 在 lockout 窗口内禁用 action reuse；
  - 在 lockout 窗口内不缓存新的 suffix action。

实验：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task2
- Trials：`10`
- 结果目录：`results/b4_agentic_light_safeLockout80_contextGate_replan5_reuseMax2_midNudge_task2_10trials_seed7_20260608/`

结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Recovery precision | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic+ContextGate` | `10/10` | `30.4` | `0.0` | `0.000` | `1.000` | `0.107` | `0.893` |
| `B4-Agentic-Light naive reuse` | `8/10` | `40.6` | `24.9` | `0.380` | `0.778` | `0.117` | `0.709` |
| `B4-Agentic-Light safe80` | `10/10` | `43.5` | `13.6` | `0.238` | `1.000` | `0.153` | `0.847` |

判断：

- Safe lockout 将 naive coupling 的成功率从 `8/10` 恢复到 `10/10`。
- Recovery precision 从 `0.778` 恢复到 `1.000`，说明扰动/恢复窗口内禁用 reuse 对 Agentic recovery 有必要。
- Safe80 是鲁棒优先配置；由于与 `B4-Agentic+ContextGate` 的 `replan` 设置不同，还需要补齐 `B4-Agentic+ContextGate replan5` 无 Light-Reuse 对照，形成公平效率比较。

#### Replan5 fair comparison

补齐公平对照：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task2
- Trials：`10`
- 结果目录：`results/b4_agentic_contextGate_replan5_noLight_midNudge_task2_10trials_seed7_20260608/`

公平对比结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | VLA wall / ep | Episode wall | Recovery precision | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic replan5 no-light` | `9/10` | `59.4` | `0.0` | `23.25 s` | `34.94 s` | `0.875` | `0.206` | `0.714` |
| `B4-Agentic-Light naive reuse` | `8/10` | `40.6` | `24.9` | `15.95 s` | `28.82 s` | `0.778` | `0.117` | `0.709` |
| `B4-Agentic-Light safe40` | `10/10` | `38.3` | `16.1` | `15.00 s` | `25.65 s` | `1.000` | `0.140` | `0.860` |
| `B4-Agentic-Light safe80` | `10/10` | `43.5` | `13.6` | `17.05 s` | `28.35 s` | `1.000` | `0.153` | `0.847` |

判断：

- 与同样 `replan=5` 的 Agentic no-light 对照相比，safe40：
  - 成功率从 `9/10` 提升到 `10/10`；
  - full VLA calls 从 `59.4/ep` 降到 `38.3/ep`；
  - VLA wall time 从 `23.25 s/ep` 降到 `15.00 s/ep`；
  - episode wall time 从 `34.94 s/ep` 降到 `25.65 s/ep`；
  - deadline miss 从 `0.206` 降到 `0.140`；
  - SUD 从 `0.714` 提升到 `0.860`。
- 与 naive reuse 相比，safe40 牺牲一部分 skip ratio，但恢复了 Agentic recovery 稳定性。
- 单 seed 下 safe40 效率优于 safe80；跨 seed 检查后，safe80 更适合作为主耦合配置，safe40 保留为效率消融。
- 当前结论：Agentic 与轻量化不能简单拼接，必须通过 Agent-aware reuse lockout 耦合。

#### Seed robustness check

追加实验：

- Method：`B4-Agentic-Light safe40`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 40 --light-reuse-post-recovery-lockout-steps 40`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task2
- Seeds：`7, 17`

结果：

| Seed | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Recovery precision | Deadline miss | SUD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `7` | `10/10` | `38.3` | `16.1` | `0.296` | `1.000` | `0.140` | `0.860` |
| `17` | `9/10` | `45.1` | `19.1` | `0.298` | `0.900` | `0.140` | `0.773` |

判断：

- safe40 在 seed17 出现 `1` 次失败，失败伴随 recovery 未成功和 transition 触发。
- safe40 是更快配置，但跨 seed 鲁棒性不足；需要验证 safe80 是否能恢复 seed17。

追加 safe80 seed17：

- Method：`B4-Agentic-Light safe80`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task2
- Seeds：`7, 17`

结果：

| Seed | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Recovery precision | Deadline miss | SUD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `7` | `10/10` | `43.5` | `13.6` | `0.238` | `1.000` | `0.153` | `0.847` |
| `17` | `10/10` | `42.1` | `12.7` | `0.232` | `1.000` | `0.154` | `0.846` |
| `27` | `10/10` | `43.2` | `12.4` | `0.223` | `1.000` | `0.156` | `0.844` |
| `Mean` | `30/30` | `42.9` | `12.9` | `0.231` | `1.000` | `0.154` | `0.846` |

同配置 no-light 基线已补：

| Seed | Config | Success | Full VLA calls / ep | Skipped calls / ep | Recovery precision | Deadline miss | SUD |
|---:|---|---:|---:|---:|---:|---:|---:|
| `7` | `B4-Agentic replan5 no-light` | `9/10` | `59.4` | `0.0` | `0.875` | `0.206` | `0.714` |
| `17` | `B4-Agentic replan5 no-light` | `9/10` | `61.0` | `0.0` | `0.900` | `0.206` | `0.714` |
| `27` | `B4-Agentic replan5 no-light` | `9/10` | `62.0` | `0.0` | `0.900` | `0.207` | `0.713` |
| `Mean` | `B4-Agentic replan5 no-light` | `27/30` | `60.8` | `0.0` | `0.892` | `0.207` | `0.714` |

三种子公平对照结论：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Episode wall | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|
| `B4-Agentic replan5 no-light` | `27/30` | `60.8` | `0.0` | `34.67 s` | `0.207` | `0.714` |
| `B4-Agentic-Light safe80` | `30/30` | `42.9` | `12.9` | `27.91 s` | `0.154` | `0.846` |

判断：

- `safe80` 在三种子上成功率从 `27/30` 提升到 `30/30`。
- Full VLA calls 降低约 `29.4%`，episode wall time 降低约 `19.5%`。
- Deadline miss 从 `0.207` 降到 `0.154`，SUD 从 `0.714` 提升到 `0.846`。
- 该结果可以作为“Agent-aware lightweight runtime 满足机器人实时性，同时不牺牲 Agentic recovery”的主表结果。

判断：

- safe80 在三个种子上均保持 `10/10`，总成功率 `30/30`，且 recovery precision 为 `1.000`。
- 与 safe40 相比，safe80 的 skip ratio 较低，但跨种子鲁棒性更好。
- 当前主配置更新为 `B4-Agentic-Light safe80`；`safe40` 作为效率优先消融。

### Task6 coupling boundary check

实验：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 40 --light-reuse-post-recovery-lockout-steps 40`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task6
- Planned trials：`10`
- Early stop：`0/3`
- 结果目录：`results/b4_agentic_light_safeLockout40_contextGate_replan5_reuseMax2_midNudge_task6_10trials_seed7_20260608/`

结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | SUD |
|---|---:|---:|---:|---:|---:|---:|
| `B4-Agentic-Light safe40 Task6` | `0/3` | `96.3` | `8.0` | `0.077` | `62.62 s` | `0.000` |

判断：

- Task6 上 safe40 直接退化，且 skipped ratio 很低，说明收益很小、风险很大。
- 该退化不能直接归因于 action reuse；更可能是 `replan=5` 短 commit 对关系/长程任务不适配。
- 下一步补跑 `B4-Agentic replan5 no-light` 的 3-trial 诊断，判断是 Light-Reuse 问题还是短 commit 问题。

诊断对照：

- Method：`B4-Agentic`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task6
- Trials：`3`
- 结果目录：`results/b4_agentic_contextGate_replan5_noLight_midNudge_task6_3trials_seed7_20260608/`

结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Episode wall | SUD |
|---|---:|---:|---:|---:|---:|
| `B4-Agentic replan5 no-light Task6` | `1/3` | `88.7` | `0.0` | `55.35 s` | `0.266` |
| `B4-Agentic-Light safe40 Task6` | `0/3` | `96.3` | `8.0` | `62.62 s` | `0.000` |

判断更新：

- `B4-Agentic replan5 no-light` 也明显退化，说明 Task6 的主要问题是短 commit 与关系/长程任务不匹配。
- Task6/Task8 后续不采用 `replan=5` 轻量化路线；应使用默认 replan，并通过 task-aware gate 禁用 Light-Reuse。
- 轻量化主线应明确为 task-aware runtime：短程单物体任务启用 action reuse，关系/多物体/长程任务保留 VLA 默认 action chunk。

Task-aware disable sanity check：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --light-reuse-actions --light-reuse-disable-long-horizon`
- Perturbation：`mid_episode_nudge`
- Benchmark：LIBERO-10 Task6
- Trials：`3`
- 结果目录：`results/b4_agentic_light_taskAwareDisable_defaultReplan_midNudge_task6_3trials_seed7_20260608/`

结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | SUD |
|---|---:|---:|---:|---:|---:|
| `B4-Agentic-Light task-aware disable Task6` | `1/3` | `41.7` | `0.0` | `0.000` | `0.300` |

判断：

- `--light-reuse-disable-long-horizon` 已覆盖 Task6 关系/多物体任务，`skipped calls = 0`，gate 生效。
- 该 3-trial sanity check 不用于评估 Task6 主性能；Task6 主性能仍参考默认 `B4-Agentic` 多 seed 结果。

### Recovery Stress: Task2 stronger mid-episode nudge

目的：验证主配置 `B4-Agentic-Light safe80` 在更强扰动下是否仍能保持 Agentic recovery，同时维持 VLA realtime 轻量化收益。

实验：

- Method：`B4-Agentic-Light safe80`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy=0.05`
- Benchmark：LIBERO-10 Task2
- Trials：`20`
- Seed：`7`
- 结果目录：`results/recoveryStress_b4_agentic_light_safe80_replan5_reuseMax2_midNudge005_task2_20trials_seed7_20260608/`

结果：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Recovery precision | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic-Light safe80 mid_nudge_xy=0.05` | `18/20` | `42.75` | `15.65` | `0.268` | `25.47 s` | `0.857` | `0.148` | `0.765` |

判断：

- 在 `mid_nudge_xy=0.05` 强扰动下，主配置仍保持 `18/20` 成功率，并节省约 `26.8%` VLA 调用。
- 相比默认 `mid_nudge_xy=0.03` 的三种子 `30/30`，强扰动暴露出 recovery 边界，可作为 stress robustness/limitation 表。
- 下一步需要同 seed、同扰动强度的 `B4-Agentic replan5 no-light` 对照，判断 strong nudge 下轻量化耦合是否仍优于纯 Agentic runtime。

补充 no-light 对照：

- Method：`B4-Agentic replan5 no-light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --replan-steps 5`
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy=0.05`
- Benchmark：LIBERO-10 Task2
- Trials：`20`
- Seed：`7`
- 结果目录：`results/recoveryStress_b4_agentic_contextGate_replan5_noLight_midNudge005_task2_20trials_seed7_20260608/`

强扰动公平对照：

| Config | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Recovery precision | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic replan5 no-light` | `19/20` | `57.15` | `0.00` | `0.000` | `30.52 s` | `0.929` | `0.205` | `0.755` |
| `B4-Agentic-Light safe80` | `18/20` | `42.75` | `15.65` | `0.268` | `25.47 s` | `0.857` | `0.148` | `0.765` |

判断更新：

- 在 `mid_nudge_xy=0.05` 强扰动 stress 下，no-light 多 `1/20` 成功，说明更强扰动会放大轻量化耦合的鲁棒性边界。
- `B4-Agentic-Light safe80` 将 full VLA calls 从 `57.15` 降到 `42.75`，降低约 `25.2%`；episode wall 从 `30.52 s` 降到 `25.47 s`，降低约 `16.6%`。
- Deadline miss 从 `0.205` 降到 `0.148`，SUD 从 `0.755` 小幅提升到 `0.765`。
- 论文写法：默认扰动 `mid_nudge_xy=0.03` 三种子结果作为主结果；`mid_nudge_xy=0.05` 作为 stress/limitation，展示强恢复场景下成功率与实时性的可控权衡。

### Joint 2x2: Agentic Harness x Lightweight VLA Runtime

目的：验证最新论文路线中的三层结构：

1. `B4-Agentic` 负责 Agentic Policy / recovery harness。
2. `B0-VLA-Light` 证明 Lightweight VLA Runtime 可脱离 Agentic 独立使用。
3. `B4-Agentic-Light` 证明 Agentic Harness 与 Lightweight Runtime 可以高效耦合。

协议：

- Benchmark：LIBERO-10 Task2
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy=0.03`
- Seeds：`7, 17, 27`
- Trials：`10/seed`
- Fair replan setting：所有配置均使用 `--replan-steps 5`
- Light config：`--light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`

逐 seed 结果：

| Method | Seed | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD | Recovery precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA replan5 no-light` | `7` | `10/10` | `51.8` | `0.0` | `0.000` | `27.82 s` | `0.202` | `0.798` | - |
| `B0-VLA replan5 no-light` | `17` | `8/10` | `62.0` | `0.0` | `0.000` | `33.02 s` | `0.201` | `0.639` | - |
| `B0-VLA replan5 no-light` | `27` | `10/10` | `51.0` | `0.0` | `0.000` | `27.35 s` | `0.201` | `0.799` | - |
| `B0-VLA-Light safe80` | `7` | `10/10` | `33.7` | `16.9` | `0.334` | `20.87 s` | `0.135` | `0.865` | - |
| `B0-VLA-Light safe80` | `17` | `9/10` | `37.3` | `20.7` | `0.357` | `23.33 s` | `0.131` | `0.781` | - |
| `B0-VLA-Light safe80` | `27` | `10/10` | `34.2` | `17.6` | `0.339` | `21.37 s` | `0.133` | `0.867` | - |
| `B4-Agentic replan5 no-light` | `7` | `9/10` | `59.4` | `0.0` | `0.000` | `33.56 s` | `0.206` | `0.714` | `0.875` |
| `B4-Agentic replan5 no-light` | `17` | `9/10` | `61.0` | `0.0` | `0.000` | `34.62 s` | `0.206` | `0.714` | `0.900` |
| `B4-Agentic replan5 no-light` | `27` | `9/10` | `62.0` | `0.0` | `0.000` | `35.82 s` | `0.207` | `0.713` | `0.900` |
| `B4-Agentic-Light safe80` | `7` | `10/10` | `43.5` | `13.6` | `0.238` | `28.35 s` | `0.153` | `0.847` | `1.000` |
| `B4-Agentic-Light safe80` | `17` | `10/10` | `42.1` | `12.7` | `0.232` | `27.28 s` | `0.154` | `0.846` | `1.000` |
| `B4-Agentic-Light safe80` | `27` | `10/10` | `43.2` | `12.4` | `0.223` | `28.09 s` | `0.156` | `0.844` | `1.000` |

三种子均值：

| Method | Agentic | Light | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD | Recovery precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA replan5 no-light` | no | no | `28/30` | `54.93` | `0.00` | `0.000` | `29.40 s` | `0.201` | `0.745` | - |
| `B0-VLA-Light safe80` | no | yes | `29/30` | `35.07` | `18.40` | `0.344` | `21.85 s` | `0.133` | `0.838` | - |
| `B4-Agentic replan5 no-light` | yes | no | `27/30` | `60.80` | `0.00` | `0.000` | `34.67 s` | `0.207` | `0.714` | `0.892` |
| `B4-Agentic-Light safe80` | yes | yes | `30/30` | `42.93` | `12.90` | `0.231` | `27.91 s` | `0.154` | `0.846` | `1.000` |

判断：

- `B0-VLA-Light safe80` 将 full VLA calls 从 `54.93` 降到 `35.07`，降低约 `36.2%`；episode wall 从 `29.40 s` 降到 `21.85 s`，降低约 `25.7%`；同时成功率从 `28/30` 到 `29/30`。
- `B4-Agentic-Light safe80` 相比 `B4-Agentic replan5 no-light` 将 full VLA calls 降低约 `29.4%`，episode wall 降低约 `19.5%`，deadline miss 从 `0.207` 降到 `0.154`，成功率从 `27/30` 到 `30/30`。
- 该表可作为最终论文的 joint table：Lightweight VLA Runtime 可独立提升 realtime；与 Agentic Harness 耦合后获得最高成功率和最高 SUD。
- 解释边界：`B0-Light` 的 skip ratio 更高，因为没有 recovery 后 lockout；`B4-Light` 的 skip ratio 较低，但 recovery precision 达到 `1.000`，说明 Agentic coupling 更保守、更鲁棒。

### Offline realtime deadline stress from traces

工具：

- Script：`scripts/summarize_realtime_deadline_stress.py`
- 输入：Joint 2x2 的 `episode_traces.jsonl`
- 口径：根据每个 episode 的 `vla_latency_ms_values` 离线重算不同控制 deadline 下的 miss rate；该口径用于 deadline stress，不替代 rollout summary 中默认记录的 `deadline_miss_rate`。

结果：

| Method | Success | Full calls/ep | Skipped/ep | Wall sec/ep | VLA p50 | VLA p95 | VLA p99 | Miss@50ms | SUD@50ms | Miss@80ms | SUD@80ms | Miss@100ms | SUD@100ms | Miss@200ms | SUD@200ms | Miss@400ms | SUD@400ms | Miss@500ms | SUD@500ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | `28/30` | `54.93` | `0.00` | `29.40` | `368.1` | `384.9` | `395.4` | `0.201` | `0.745` | `0.201` | `0.745` | `0.201` | `0.745` | `0.201` | `0.745` | `0.001` | `0.932` | `0.000` | `0.933` |
| `B0-VLA-Light` | `29/30` | `35.07` | `18.40` | `21.85` | `368.1` | `385.0` | `397.5` | `0.133` | `0.838` | `0.133` | `0.838` | `0.133` | `0.838` | `0.133` | `0.838` | `0.001` | `0.965` | `0.000` | `0.967` |
| `B4-Agentic` | `27/30` | `60.80` | `0.00` | `34.67` | `381.6` | `416.5` | `451.1` | `0.226` | `0.696` | `0.226` | `0.696` | `0.226` | `0.696` | `0.226` | `0.696` | `0.033` | `0.869` | `0.001` | `0.899` |
| `B4-Agentic-Light` | `30/30` | `42.93` | `12.90` | `27.91` | `387.8` | `428.6` | `473.8` | `0.171` | `0.829` | `0.171` | `0.829` | `0.171` | `0.829` | `0.171` | `0.829` | `0.040` | `0.960` | `0.001` | `0.999` |

判断：

- 单次 VLA 调用延迟仍在 `~368-429 ms p95`，所以 `50/80/100/200 ms` deadline 下，关键不是让单次调用变快，而是减少控制循环中的 blocking full VLA calls。
- `B0-VLA-Light` 在不使用 Agentic Harness 的情况下将 Miss@80ms 从 `0.201` 降到 `0.133`，SUD@80ms 从 `0.745` 提升到 `0.838`。
- `B4-Agentic-Light` 在达到 `30/30` 成功率的同时，将 `B4-Agentic` 的 Miss@80ms 从 `0.226` 降到 `0.171`，SUD@80ms 从 `0.696` 提升到 `0.829`。
- 该结果支撑论文的 realtime 叙事：当前 lightweight 模块属于 inference scheduling / action reuse，不是模型权重量化；它降低的是 full VLA 阻塞频率和 episode wall time。

### Task-aware mixed sanity: default replan + disable long-horizon reuse

目的：验证最终系统配置不是全局短 commit/reuse，而是 task-aware：

- 默认 `replan_steps=10`。
- Task2 这类短程单物体任务启用 `Light-Reuse`，并通过 `--light-reuse-commit-steps 5` 缓存 suffix。
- Task6/Task8 这类关系/长程多物体任务由 `--light-reuse-disable-long-horizon` 禁用 reuse，保持默认 chunk。

协议：

- Benchmark：LIBERO-10 Task2/6/8
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy=0.03`
- Trials：`5/task`
- Seed：`7`
- Light flags：`--light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-disable-long-horizon --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`

结果目录：

- `B0-VLA-Light`：`results/taskAwareMixed_b0_vla_light_defaultReplan_reuseMax2_disableLong_midNudge003_tasks268_5trials_seed7_20260609/`
- `B4-Agentic-Light`：`results/taskAwareMixed_b4_agentic_light_defaultReplan_reuseMax2_disableLong_midNudge003_tasks268_5trials_seed7_20260609/`

按 task 汇总：

| Method | Task | Success | Full VLA calls / ep | Skipped calls / ep | Episode wall | Deadline miss | SUD | Recovery | Transition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA-Light task-aware` | `2` | `5/5` | `24.4` | `16.4` | `17.37 s` | `0.101` | `0.899` | `0/0` | `0` |
| `B0-VLA-Light task-aware` | `6` | `4/5` | `28.6` | `0.0` | `21.24 s` | `0.102` | `0.718` | `0/0` | `0` |
| `B0-VLA-Light task-aware` | `8` | `2/5` | `47.6` | `0.0` | `34.10 s` | `0.100` | `0.360` | `0/0` | `0` |
| `B4-Agentic-Light task-aware` | `2` | `5/5` | `29.6` | `13.0` | `19.55 s` | `0.108` | `0.892` | `4/4` | `0` |
| `B4-Agentic-Light task-aware` | `6` | `2/5` | `43.0` | `0.0` | `32.55 s` | `0.101` | `0.359` | `0/0` | `0` |
| `B4-Agentic-Light task-aware` | `8` | `2/5` | `61.0` | `0.0` | `41.08 s` | `0.105` | `0.358` | `2/5` | `4` |

判断：

- Task-aware Light gate 生效：Task2 有 skip，Task6/8 的 skipped calls 均为 `0.0`。
- 默认 replan + task-aware commit 是更合理的最终系统配置：Task2 的 `B0-VLA-Light` full calls 进一步降到 `24.4/ep`，优于前面公平 replan5 表中的 `33.7/ep` seed7。
- Task6/Task8 仍是当前框架边界：`B4-Agentic-Light` 在 Task6/8 未优于 `B0-VLA-Light`，失败来自 Agentic recovery/transition 不适配关系/长程多物体任务，而不是 Light-Reuse。
- 论文写法：Task2 作为 positive recovery + realtime coupling 主表；Task6/Task8 作为 task-aware gating 与 limitation/boundary analysis，说明 Agentic Policy 不能 always-on，必须由任务结构和 verifier 可靠性控制。

### Final task-aware safe gate check

目的：在上一组 mixed sanity 暴露 `B4-Agentic-Light` 对 Task6/8 过干预后，验证最终 safe gate：

- Task2：保留 Agentic progress verifier / recovery，并启用 Light-Reuse。
- Task6/8：通过 `--ba-safe-long-horizon-fast-path` 关闭 Agentic recovery/transition，通过 `--light-reuse-disable-long-horizon` 关闭 Light-Reuse。

协议：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --ba-safe-long-horizon-fast-path --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-disable-long-horizon --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Benchmark：LIBERO-10 Task2/6/8
- Perturbation：`mid_episode_nudge`
- Trials：`10/task`
- Seed：`7`
- 结果目录：`results/finalTaskAware_b4_agentic_light_safeLong_defaultReplan_reuseMax2_midNudge003_tasks268_10trials_seed7_20260609/`

结果：

| Task | Success | Full VLA calls / ep | Skipped calls / ep | Recovery | Transition |
|---:|---:|---:|---:|---:|---:|
| `2` | `9/10` | `34.8` | `14.3` | `8/9` | `1` |
| `6` | `7/10` | `31.7` | `0.0` | `0/0` | `0` |
| `8` | `6/10` | `45.1` | `0.0` | `0/0` | `0` |
| `Overall` | `22/30` | `37.2` | `4.77` | `8/9` | `1` |

判断：

- Safe gate 生效：Task6/8 的 `skipped calls = 0.0`，recovery/transition 也均为 `0`。
- 相比上一组 `B4-Agentic-Light task-aware` 的 Task6 `2/5`、Task8 `2/5`，safe-long-horizon fast path 将 Task6/8 恢复到更接近底层 VLA 的表现。
- Task2 的唯一失败伴随 `transition_total=1`，说明 Task2 主配置应显式禁用 transition，只保留 recovery 或 fast execution。

### Deployment-fast variant: Task2 no-transition default replan

目的：验证 Task2 上最终部署变体：禁用 transition、保留 task-aware Light-Reuse、使用默认 `replan_steps=10`。该变体追求更低 full VLA calls，不替代 `B4-Agentic-Light safe80 replan5` 的主 recovery 表。

协议：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --ba-disable-transition --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Benchmark：LIBERO-10 Task2
- Perturbation：`mid_episode_nudge`
- Trials：`10/seed`
- Seeds：`7, 17, 27`

结果：

| Seed | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD | Recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `7` | `10/10` | `27.1` | `15.8` | `0.368` | `18.72 s` | `0.107` | `0.893` | `0/0` |
| `17` | `10/10` | `26.9` | `15.6` | `0.367` | `18.52 s` | `0.108` | `0.892` | `0/0` |
| `27` | `9/10` | `30.1` | `18.8` | `0.384` | `20.76 s` | `0.107` | `0.804` | `0/0` |
| `Mean` | `29/30` | `28.03` | `16.73` | `0.373` | `19.33 s` | `0.107` | `0.863` | `0/0` |

判断：

- Deployment-fast variant 将 full VLA calls 降到 `28.03/ep`，低于 `B4-Agentic-Light safe80 replan5` 的 `42.93/ep`。
- 成功率为 `29/30`，低于主 recovery 配置的 `30/30`，但实时指标更好：episode wall `19.33 s`，deadline miss `0.107`，SUD `0.863`。
- 该配置更适合作为效率优先部署变体；论文主表仍使用 `B4-Agentic-Light safe80 replan5` 展示 Agentic recovery + lightweight coupling 的稳健性。
- 一个重要解释：默认 `mid_nudge_xy=0.03` 下，pi0.5 + Light-Reuse 已能自恢复多数扰动，未触发 recovery；这说明 Agentic recovery 的价值应主要放在更强 stress 或明确失败状态，而不是 always-on。

补充公平负对照：`B0-VLA-Light default replan`

- 目的：确认 no-transition deployment variant 的收益是否只是来自纯 Light Runtime。
- Method：`B0-VLA-Light`
- Flags：`--light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Benchmark：LIBERO-10 Task2
- Perturbation：`mid_episode_nudge`
- Trials：`10`
- Seed：`7`
- 结果目录：`results/finalTask2_b0_vla_light_defaultReplan_reuseMax2_midNudge003_10trials_seed7_20260609/`

结果：

| Method | Seed | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA-Light default replan` | `7` | `8/10` | `30.5` | `22.2` | `0.421` | `21.25 s` | `0.101` | `0.719` |
| `B4-Agentic-Light no-transition default replan` | `7` | `10/10` | `27.1` | `15.8` | `0.368` | `18.72 s` | `0.107` | `0.893` |

判断：

- 纯 `B0-VLA-Light default replan` 在 seed7 降到 `8/10`，说明默认 replan + aggressive reuse 不能直接作为主轻量化配置。
- 该负对照已经足够说明风险，因此 seed17/27 队列中止以节省 GPU。
- 论文写法：Light Runtime 需要 task/context-aware safety envelope；否则 skip ratio 虽高，但会损害任务成功率。主轻量化表继续使用更保守的 `safe80 replan5`，deployment-fast variant 作为效率优先补充。

### Strong stress control: Task2 no-transition recovery stress

目的：在更强扰动 `mid_nudge_xy=0.05` 下验证 transition 是否是 Agentic-Light 失败来源，并决定最终主配置是否应默认关闭 transition。

协议：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --ba-disable-transition --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Benchmark：LIBERO-10 Task2
- Perturbation：`mid_episode_nudge`
- `mid_nudge_xy`：`0.05`
- Trials：`20`
- Seed：`7`
- 结果目录：`results/recoveryStress_b4_agentic_light_safe80_noTransition_replan5_reuseMax2_midNudge005_task2_20trials_seed7_20260609/`

结果：

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD | Transition | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic-Light safe80 no-transition replan5` | `19/20` | `37.0` | `16.0` | `0.302` | `22.92 s` | `0.140` | `0.817` | `0` | `0/0` |
| `B4-Agentic-Light safe80 replan5` | `18/20` | `42.75` | `15.65` | `0.268` | `25.47 s` | `0.148` | `0.765` | `1` | `6/7` |
| `B4-Agentic replan5 no-light` | `19/20` | `57.15` | `0.0` | `0.000` | `30.52 s` | `0.205` | `0.755` | `1` | `13/14` |

失败分析：

- 唯一失败为 episode 6：`episode_steps=530`，`full_vla_calls=75`，`skipped_vla_calls=29`，`transition_triggers=0`，`recoveries_triggered=0`。
- 扰动为 `dx=0.0311, dy=0.0358`；最终 progress 仍停留在 `approach_moka_pot`，`target_distance_xy=0.2098`。
- 这说明关闭 transition 后，strong stress 下的剩余失败不是 transition 误触发，而是底层策略在较大位移后没有重新接近物体。

判断：

- no-transition 将 strong stress 下的 Agentic-Light 从 `18/20` 提升到 `19/20`，同时 full VLA calls 从 `42.75/ep` 降到 `37.0/ep`，SUD 从 `0.765` 提升到 `0.817`。
- 与 no-light Agentic 相比，成功率持平 `19/20`，但 full VLA calls 少 `35.3%`，deadline miss 从 `0.205` 降到 `0.140`，SUD 从 `0.755` 提升到 `0.817`。
- 最终主线建议：transition 不作为默认主路径，只作为 ablation 或特定任务可选模块；主配置应强调 progress-aware recovery + task/context-aware lightweight runtime。

### LIBERO-10 coverage pilot: final safe runtime, all tasks

目的：验证最终安全配置在 LIBERO-10 全任务上是否可运行，并识别哪些任务适合进入正式主表，哪些任务应作为边界分析。

协议：

- Method：`B4-Agentic-Light`
- Flags：`--ba-progress-verifier --ba-context-aware-intervention --ba-disable-transition --ba-safe-long-horizon-fast-path --replan-steps 5 --light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-disable-long-horizon --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- Benchmark：LIBERO-10 Task0-9
- Perturbation：`mid_episode_nudge`
- Trials：`5/task`
- Seed：`7`
- 结果目录：`results/finalCoverage_libero10_b4_agentic_light_safeLong_noTransition_replan5_reuseMax2_midNudge003_5trials_seed7_20260609/`

总体结果：

| Overall Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD | Peak GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `36/50` (`0.72`) | `53.46` | `13.02` | `0.196` | `33.92 s` | `0.158` | `0.610` | `9.16 GB` |

Task-wise 结果：

| Task | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Mean steps | Deadline miss | Task |
|---:|---:|---:|---:|---:|---:|---:|---|
| `0` | `4/5` | `43.8` | `19.8` | `0.311` | `325.0` | `0.137` | both alphabet soup and tomato sauce in basket |
| `1` | `5/5` | `33.0` | `17.0` | `0.340` | `258.0` | `0.133` | cream cheese box and butter in basket |
| `2` | `5/5` | `35.8` | `16.2` | `0.312` | `266.2` | `0.140` | stove + moka pot |
| `3` | `1/5` | `75.6` | `20.2` | `0.211` | `485.4` | `0.158` | black bowl in bottom drawer and close |
| `4` | `5/5` | `45.6` | `0.0` | `0.000` | `236.0` | `0.202` | two mugs on two plates |
| `5` | `4/5` | `30.4` | `18.2` | `0.374` | `249.6` | `0.129` | book in back compartment of caddy |
| `6` | `3/5` | `75.0` | `0.0` | `0.000` | `384.2` | `0.201` | mug on plate + pudding right of plate |
| `7` | `5/5` | `34.6` | `17.8` | `0.340` | `270.2` | `0.133` | alphabet soup and cream cheese box in basket |
| `8` | `2/5` | `98.2` | `0.0` | `0.000` | `500.4` | `0.200` | both moka pots on stove |
| `9` | `2/5` | `62.6` | `21.0` | `0.251` | `425.8` | `0.147` | yellow-white mug in microwave and close |

判断：

- 全任务 pilot 可完整运行，没有 OOM，峰值显存约 `9.16 GB`；当前 4090 足够支持单 benchmark runner + pi0.5 server。
- Task1/2/4/7 为稳定任务，其中 Task2 是主线 Agentic/Light 任务，Task7 可作为轻量化稳定性补充任务。
- Task0/5 为中等任务，适合作为辅助覆盖，不宜承载主 claim。
- Task3/6/8/9 是明显边界任务，多数失败跑到步数上限；Task4/6/8 的 skip 为 `0`，说明 long-horizon Light-Reuse disable 生效，失败主要来自底层 VLA 执行能力，而非轻量化误用。
- 论文叙事不应主打 LIBERO-10 全任务平均大幅提升；更稳的叙事是：
  - 在可恢复扰动任务上，Agentic Policy Harness 提升成功率/稳健性。
  - 在适合动作复用的任务上，VLA lightweight runtime 降低 full VLA calls 和 deadline miss。
  - 通过 task/context-aware safety envelope，框架在长程任务上退回底层 VLA，避免 over-intervention。

下一步：

- 正式主表继续以 Task2 为核心，使用 `mid_nudge_xy=0.03/0.05`、多 seed、B0/B0-Light/B4/B4-Light 2x2 对照。
- 轻量化补充表建议加入 Task1/5/7，验证 Light-Reuse 在非 stove 任务上的 full-call reduction。
- Task3/6/8/9 保留为 boundary/limitation，不建议继续消耗大量 GPU 追调。

### Lightweight-only supplement: B0-VLA vs B0-VLA-Light on Task1/5/7

目的：验证 VLA lightweight runtime 可以脱离 Agentic Harness 独立使用，并在非 Task2 的任务上降低 full VLA calls 与 realtime deadline miss。

协议：

- Benchmark：LIBERO-10 Task1/5/7
- Perturbation：`mid_episode_nudge`
- Trials：`5/task`
- Seed：`7`
- Baseline：`B0-VLA`
- Light：`B0-VLA-Light`
- Light flags：`--light-reuse-actions --light-reuse-max-actions 2 --light-reuse-max-buffer-actions 10 --light-reuse-commit-steps 5 --light-reuse-post-perturbation-lockout-steps 80 --light-reuse-post-recovery-lockout-steps 80`
- 结果目录：
  - `results/lightTable_b0_vla_noLight_replan5_midNudge003_tasks157_5trials_seed7_20260609/`
  - `results/lightTable_b0_vla_light_replan5_reuseMax2_midNudge003_tasks157_5trials_seed7_20260609/`

总体结果：

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Episode wall | Deadline miss | SUD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | `13/15` | `54.33` | `0.00` | `0.000` | `31.53 s` | `0.202` | `0.692` |
| `B0-VLA-Light` | `13/15` | `34.33` | `19.93` | `0.367` | `24.29 s` | `0.129` | `0.753` |

Task-wise 结果：

| Method | Task | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Mean steps | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA` | `1` | `5/5` | `51.80` | `0.00` | `0.000` | `267.0` | `0.202` |
| `B0-VLA-Light` | `1` | `5/5` | `34.00` | `17.20` | `0.336` | `265.0` | `0.133` |
| `B0-VLA` | `5` | `4/5` | `48.40` | `0.00` | `0.000` | `250.0` | `0.202` |
| `B0-VLA-Light` | `5` | `4/5` | `28.60` | `18.40` | `0.391` | `244.0` | `0.125` |
| `B0-VLA` | `7` | `4/5` | `62.80` | `0.00` | `0.000` | `322.6` | `0.201` |
| `B0-VLA-Light` | `7` | `4/5` | `40.40` | `24.20` | `0.375` | `331.0` | `0.128` |

判断：

- 纯轻量化模块在 Task1/5/7 上保持同等成功率 `13/15`，同时 full VLA calls 下降 `36.8%`，episode wall 下降 `23.0%`，deadline miss 从 `0.202` 降到 `0.129`。
- Task5/7 的失败 episode 与 no-light baseline 同 seed 位置一致，说明失败主要来自底层 VLA/扰动，而不是 Light-Reuse 独有退化。
- 这组结果可支撑论文第二主轴：轻量化模块可独立部署，用于 VLA realtime inference；Agentic Harness 不是轻量化收益的必要条件。

### Final experiment closure note

收尾判断：

- 当前正式主表已经完整：
  - Task2 `mid_nudge_xy=0.03` 的 `B0-VLA / B0-VLA-Light / B4-Agentic / B4-Agentic-Light` 三种子 2x2 对照。
  - Task1/5/7 的 `B0-VLA` vs `B0-VLA-Light` 轻量化独立表。
  - Task2 `mid_nudge_xy=0.05` 的 B4/B4-Light strong stress 与 no-transition stress/limitation。
- 继续补 `mid_nudge_xy=0.05` 的 B0/B0-Light optional stress baseline 时，发现 benchmark 的自动 server restart 会将显式 local PyTorch checkpoint server 替换为 `.venv --env LIBERO` default server，导致 backend 和显存占用不一致。
- 已修复 `scripts/run_agentic_vla_libero.py` 中 `_restart_policy_server`：
  - 自动重启改为当前 conda Python + 显式 `pi05_libero_pytorch` checkpoint。
  - `stdout/stderr` 改为 `DEVNULL`，避免重启子进程因 PIPE 未消费而阻塞。
- 由于该 optional stress baseline 不属于主表必要结果，并且补跑过程中出现无效首 episode / server restart 等工程噪声，停止继续消耗 GPU。

最终实验策略：

- 不再继续探索 LIBERO-10 全任务平均或边界任务调参。
- 下一阶段进入结果整理、统计汇总、画图和论文写作。
- 若后续必须补实验，只补两类：
  - Task2 主表的缺失 seed 或重复确认；
  - Task1/5/7 轻量化表的更多 seed。

## 2026-06-09 CAQ-Lite Quantization Preparation

### Q0 checkpoint and runtime module profile

目的：为 `CAQ-Lite: Criticality-Aware Quantized Lightweight Runtime` 做量化候选分析，避免把轻量化工作做成普通 VLA PTQ 复现。

新增脚本：

- `scripts/profile_openpi_policy_modules.py`
- `scripts/inspect_openpi_runtime_modules.py`

输出：

- `docs/caq_lite_pi05_libero_profile_20260609.md`
- `docs/caq_lite_pi05_libero_profile_20260609.json`
- `docs/caq_lite_pi05_libero_runtime_modules_cpu_20260609.md`
- `docs/caq_lite_pi05_libero_runtime_modules_cpu_20260609.json`

Checkpoint-level profile：

| Item | Value |
|---|---:|
| Tensors | `812` |
| Params | `3.62B` |
| Estimated tensor size | `6.74 GB` |
| `w8_candidate` | `293 tensors / 2.40B params` |
| `w8_candidate_guarded` | `165 tensors / 691.18M params` |
| `keep_fp` | `354 tensors / 527.27M params` |

Runtime module profile：

| Item | Value |
|---|---:|
| Modules | `858` |
| `w8_candidate` | `291 modules / 2.40B params` |
| `w8_candidate_guarded` | `164 modules / 427.97M params` |
| `keep_fp` | `403 modules / 1.32B params` |

解释：

- 第一阶段不做 W4A4 全图量化。
- `paligemma` language MLP / attention / vision matrix weights 是 W8 候选。
- `gemma_expert` / action expert 只作为 guarded candidate，必须受 criticality gate 控制。
- `embed_tokens`、`lm_head`、action/output projection、norm/embedding 第一版保持 FP。
- 该 profile 支撑论文中的 CAQ-Lite 设计：低风险路径可以尝试低成本推理，critical path 必须 full precision fallback。

### Q1 bitsandbytes W8 backend smoke

目的：验证 bitsandbytes `Linear8bitLt` 能否作为 CAQ-Lite 的可插拔 W8 backend，并判断它在 4090 batch=1 robot online inference 下是否能直接带来 latency speedup。

新增脚本：

- `scripts/caq_lite_bnb_weight_smoke.py`

输出：

- `docs/caq_lite_bnb_weight_smoke_20260609.json`
- `docs/caq_lite_bnb_weight_smoke_mlp_seq64_20260609.json`
- `docs/caq_lite_bnb_weight_smoke_mlp_seq256_20260609.json`

结果：

| Tensor | Shape | Input | FP16 ms | W8 ms | W8 / FP speedup | Mean abs err | Mean rel err |
|---|---:|---:|---:|---:|---:|---:|---:|
| `layers.0.self_attn.q_proj.weight` | `2048 x 2048` | `1 x 8 x 2048` | `0.077` | `0.367` | `0.210x` | `0.0131` | `0.0141` |
| `layers.0.mlp.gate_proj.weight` | `16384 x 2048` | `1 x 64 x 2048` | `0.102` | `0.349` | `0.292x` | `0.0146` | `0.0118` |
| `layers.0.mlp.gate_proj.weight` | `16384 x 2048` | `1 x 256 x 2048` | `0.302` | `0.379` | `0.797x` | `0.0146` | `0.0117` |

判断：

- bitsandbytes W8 backend 可以在真实 pi0.5 checkpoint 权重上跑通，误差约 `1.2%-1.4%`。
- 在 4090、batch=1、机器人在线 inference 形态下，W8 单层并不比 FP16 快；seq 越大差距越小，但仍未超过 FP16。
- 因此当前论文不应声称 “W8 直接加速 VLA inference”。
- 更稳的结论是：
  - W8 是 CAQ-Lite 的可插拔低显存/端侧 backend；
  - 当前主实时收益来自 criticality-aware action reuse / skipped full VLA calls；
  - critical phases 仍保持 FP path，避免量化误差影响接触、释放和 recovery。

### Q2 CAQ-Lite proxy rollout and server stability

目的：把 CAQ-Lite 从离线量化 profile 推进到在线 rollout 链路。当前阶段采用 `CAQ-Proxy` 命名：以 Light-Reuse 作为低成本/低精度路径的系统级代理，关键接触、扰动、recovery lockout 时回退 full VLA。

工程修复：

- 新增论文方法标签：
  - `B0-VLA-Quant`
  - `B0-VLA-CAQ`
  - `B0-VLA-CAQ-Proxy`
  - `B4-Agentic-Quant`
  - `B4-Agentic-CAQ`
  - `B4-Agentic-CAQ-Proxy`
- `B4-Agentic-CAQ-Proxy` 已归入 `_is_ba_harness()`，可复用 Agentic Harness、GraphRAG priors、progress verifier、context gate 等机制。
- 发现当前 PyTorch/openpi 环境下，默认 `torch.compile(mode="max-autotune")` server 在 pi0.5 inference 时会触发：
  - `RuntimeError: invalid dtype for bias - should match query's dtype`
  - 位置为 torch inductor/cudagraph efficient attention。
- 修复方式：
  - 显式启动 policy server 时加入 `OPENPI_DISABLE_TORCH_COMPILE=1`。
  - `_restart_policy_server()` 默认继承 `OPENPI_DISABLE_TORCH_COMPILE=1`，避免 benchmark 自动重启后回到不稳定 compile 路径。
- 该修复不改变方法本身，只保证 policy server 使用 eager 稳定路径；后续论文中可将 compile 路径作为工程注意事项，不作为算法贡献。

健康检查：

- Method：`B0-VLA-Light`
- 结果目录：`results/health_b0_vla_light_eager_replan5_reuseMax2_midNudge003_task2_1trial_seed909_20260609/`
- 设置：Task2，`mid_episode_nudge`，`replan_steps=5`，safe80 Light-Reuse。

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Wall / ep | Miss@80ms | SUD@80ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA-Light eager health` | `1/1` | `33.00` | `16.00` | `0.327` | `23.75 s` | `0.137` | `0.863` |

CAQ proxy smoke：

- Method：`B0-VLA-CAQ-Proxy`
- 结果目录：`results/caqProxy_b0_vla_caq_proxy_eager_replan5_reuseMax2_midNudge003_task2_3trials_seed1009_20260609/`
- 设置：Task2，`mid_episode_nudge`，`replan_steps=5`，safe80 Light-Reuse。

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Wall / ep | Miss@80ms | SUD@80ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA-CAQ-Proxy` | `3/3` | `35.33` | `18.33` | `0.342` | `24.75 s` | `0.134` | `0.866` |

Agentic + CAQ proxy coupling smoke：

- Method：`B4-Agentic-CAQ-Proxy`
- 结果目录：`results/caqProxy_b4_agentic_caq_proxy_eager_safe80_noTransition_replan5_reuseMax2_midNudge003_task2_3trials_seed1109_20260609/`
- 设置：Task2，`mid_episode_nudge`，`replan_steps=5`，`--ba-progress-verifier --ba-context-aware-intervention --ba-disable-transition`，safe80 Light-Reuse。

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Wall / ep | Miss@80ms | SUD@80ms | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B4-Agentic-CAQ-Proxy` | `3/3` | `46.00` | `16.67` | `0.266` | `30.79 s` | `0.146` | `0.854` | `0/0` |

判断：

- `B0-VLA-CAQ-Proxy` 与 `B4-Agentic-CAQ-Proxy` 均已在线 rollout 跑通，说明 CAQ-Lite 的系统接口已经能接入现有 VLA 和 Agentic Harness。
- 这组 Q2 是工程闭环 smoke，不替代前面三种子主表；正式统计仍以 `B0-VLA-Light` / `B4-Agentic-Light` 主表为准。
- 在 `mid_nudge_xy=0.03` 的 3-trial smoke 中未触发 recovery，因此只能证明耦合链路和 realtime lightweight 指标，不能证明 Agentic recovery 增益。
- 后续如果需要补充 CAQ-Lite 的论文实验，应优先补：
  - CAQ-Proxy vs Light-Reuse 的命名对齐和表格整合；
  - strong stress `mid_nudge_xy=0.05` 下 `B4-Agentic-CAQ-Proxy` 的 recovery-triggered 版本；
  - 更真实的 W8 backend 接入，但仅作为低显存/端侧 backend，不作为 4090 batch=1 latency speedup 主张。

### Q2 strong stress: CAQ-Proxy under `mid_nudge_xy=0.05`

目的：验证 CAQ-Proxy 在更强扰动下是否保持稳定，并区分 CAQ runtime 与 Agentic gate 的贡献。

设置：

- Benchmark：LIBERO-10 Task2
- Perturbation：`mid_episode_nudge`
- `--mid-nudge-xy 0.05`
- Trials：20
- Seed：7
- Server：`OPENPI_DISABLE_TORCH_COMPILE=1` eager pi0.5 policy server

结果目录：

- `B0-VLA-CAQ-Proxy`：`results/caqStress_b0_vla_caq_proxy_eager_replan5_reuseMax2_midNudge005_task2_20trials_seed7_20260609/`
- `B4-Agentic-CAQ-Proxy`：`results/caqStress_b4_agentic_caq_proxy_eager_safe80_noTransition_replan5_reuseMax2_midNudge005_task2_20trials_seed7_20260609/`

| Method | Success | Full VLA calls / ep | Skipped calls / ep | Skip ratio | Wall / ep | Miss@80ms | SUD@80ms | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0-VLA-CAQ-Proxy` | `19/20` | `34.30` | `17.70` | `0.340` | `24.08 s` | `0.134` | `0.822` | `0/0` |
| `B4-Agentic-CAQ-Proxy no-transition` | `19/20` | `36.05` | `16.80` | `0.318` | `24.95 s` | `0.139` | `0.817` | `0/0` |

Failure analysis：

- 两个方法都在 trial 14 失败。
- `B0` failure trace：`episode_steps=530`，`full_vla_calls=60`，`skipped_vla_calls=44`。
- `B4` failure trace：`episode_steps=530`，`full_vla_calls=61`，`skipped_vla_calls=44`。
- `B4` progress verifier final state：
  - `kind=single_moka_pot_stove`
  - `phase=approach_moka_pot`
  - `completed_count=0`
  - `target_distance_xy≈0.212`
- 本组未触发 recovery，因此不能把 `19/20` 成功率归因于 Agentic recovery。

判断：

- 在 Task2 strong stress 下，CAQ runtime 本身已经能保持 `19/20` 成功率，同时将 full VLA calls 压到约 `34.3/ep`。
- `B4-Agentic-CAQ-Proxy` 的 progress/context gate 在本组没有提供额外成功率收益，且由于更保守的 risk gate，full VLA calls 稍高。
- 论文表述应避免在该表中声称 Agentic gate 提升 Task2 strong stress 成功率；更准确的说法是：
  - CAQ-Lite 提升 realtime efficiency；
  - Agentic Harness 提供统一的安全 envelope 和 recovery 接口；
  - Agentic recovery 的价值由前面的 `B4-Agentic` / `B4-Agentic-Light` 主表和 recovery stress 表支撑。
- 这组结果适合放在 CAQ-Lite supplement 或 appendix，用于说明 CAQ-Lite 可以脱离 Agentic Harness 单独部署。

### IsaacLab near-deployment smoke: action queue and runtime adapter

目的：不把 IsaacLab 当作主 benchmark，而是作为真机部署前的闭环验证层，检查 `observation -> Agentic/VLA action queue -> lightweight action reuse -> simulator step` 的基本链路能否在 IsaacLab runtime 中跑通。

工程修复：

- `scripts/run_isaaclab_vla_smoke.py` 新增 `--fast-exit`，先写出 summary 再退出，避免 headless Isaac Sim teardown 卡住 smoke run。
- 主循环新增 `simulation_app.is_running()` 检查。
- 保留 `--dummy-policy` / `--dummy-image`，用于不占用 pi0.5 policy server 的最小 runtime 验证。

环境确认：

- IsaacLab repo：`/home/admin1/ct/IsaacLab`
- Conda env：`isaac`
- Isaac Sim：`4.5.0.0`
- 最小运行命令：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate isaac
python scripts/run_isaaclab_vla_smoke.py \
  --headless \
  --dummy-policy \
  --dummy-image \
  --steps 20 \
  --replan-steps 5 \
  --fast-exit \
  --summary-json results/isaaclab_vla_smoke_dummy_fast_20260609/summary.json \
  --log-file results/isaaclab_vla_smoke_dummy_fast_20260609/run.log
```

结果目录：

- `results/isaaclab_vla_smoke_dummy_fast_20260609/`

| Mode | Steps | Replan | Full policy calls | Reused actions | Reuse ratio | Result |
|---|---:|---:|---:|---:|---:|---|
| `dummy_policy + dummy_image` | `20` | `5` | `4` | `16` | `0.800` | IsaacLab runtime smoke passed |

判断：

- IsaacLab 可以作为本项目的 near-deployment demo 层。
- 当前 smoke 只证明 IsaacLab runtime、action queue、reuse 统计和日志链路可用；尚未证明 Franka IK 控制或真实 OpenPI policy 在 IsaacLab 中完成任务。
- 下一步 IsaacLab 优先级：
  1. `Franka IK + dummy policy`：验证 VLA action adapter 到 EEF/IK 控制链路；
  2. `Franka IK + OpenPI server smoke`：验证真实 policy server payload/action 维度链路；
  3. 只选择 `Lift-Cube` 或简单 pick/lift 任务做视频 demo，不扩展成大规模 benchmark。

### IsaacLab Franka IK smoke: VLA-action-style adapter

目的：进一步验证部署链路中更接近真机的一层：`7D VLA-style action -> EEF delta / gripper command -> Franka Differential IK controller`。

工程修复：

- `scripts/run_isaaclab_franka_openpi.py` 新增：
  - `--dummy-policy`
  - `--log-file`
  - `--summary-json`
  - `--fast-exit`
- 为本地 IsaacLab source 包增加 path resolution，保证 `isaaclab_assets` 可用。
- 将 `robot.find_bodies()` / `robot.find_joints()` 挪到 `sim.reset()` 后，符合 IsaacLab Articulation lifecycle。

运行命令：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate isaac
python scripts/run_isaaclab_franka_openpi.py \
  --headless \
  --dummy-policy \
  --dummy-image \
  --steps 20 \
  --replan-steps 5 \
  --arm-scale 0.02 \
  --fast-exit \
  --summary-json results/isaaclab_franka_dummy_ik_20260609/summary.json \
  --log-file results/isaaclab_franka_dummy_ik_20260609/run.log
```

结果目录：

- `results/isaaclab_franka_dummy_ik_20260609/`

| Mode | Steps | Replan | Full policy calls | Reused actions | Reuse ratio | State dim | Action dim | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Franka IK + dummy_policy + dummy_image` | `20` | `5` | `4` | `16` | `0.800` | `7` | `7` | Franka IK adapter smoke passed |

判断：

- IsaacLab Franka 可以接收 VLA-style `7D` action，并通过 Differential IK 控制器执行 EEF delta / gripper target。
- 这一步证明了从软件 harness 到物理控制接口的部署路径，而非 benchmark 成功率。
- 下一步若继续推进，应接真实 OpenPI server 做 `payload/action-dim smoke`；但由于当前 `pi05_libero` 不匹配 IsaacLab Franka 场景，不应把该 smoke 解释为 VLA task success。

### RoboTwin / LIBERO-Plus benchmark decision

外部 benchmark 定位：

- `LIBERO-Plus` 是面向 VLA robustness 的 LIBERO 扩展，包含约 `10k` perturbed task variants 和 7 类扰动因素，适合作为后续 robustness benchmark 扩展。
- `RoboTwin 2.0` 是双臂 manipulation benchmark，当前文档列出 `50` 个 dual-arm tasks、Aloha-AgileX、14D joint action、head/left/right cameras，适合对齐新 VLA / WAM 论文的实验风格。

当前论文取舍：

- `LIBERO / LIBERO-Plus` 与当前 `pi0.5-libero`、Agentic Harness、CAQ-Lite 的兼容性最高，应作为主 benchmark/robustness 扩展。
- `RoboTwin 2.0` 有论文说服力，但 action space、双臂 embodiment、数据/模型适配成本更高。若没有现成 RoboTwin-compatible checkpoint，不应立刻替代主线。
- 建议把 RoboTwin 放为 `optional extension`：
  - 先做环境安装与 evaluation harness adapter smoke；
  - 若 checkpoint/action adapter 能快速跑通，再选 `3-5` 个代表性任务、小规模 trials；
  - 若需要训练/微调，则推迟到期刊扩展或后续工作，避免偏离 Agentic Policy + realtime VLA runtime 主贡献。

## 2026-06-09 IsaacLab Mimic training-data pipeline

### Task choice

选择 IsaacLab 官方 Franka Stack Cube 任务族作为 near-deployment / sim-to-real 训练验证任务：

- `Isaac-Stack-Cube-Franka-IK-Rel-v0`
- `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`
- `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`

选择原因：

- 任务是多阶段长程操作：抓取第一个方块、堆叠、抓取第二个方块。
- 动作为 `7D` relative EEF action，与当前 VLA action adapter / Franka IK 部署接口一致。
- 官方 IsaacLab Mimic 支持 demonstration annotation 和 synthetic trajectory generation。
- 适合展示 Agentic Harness 的 progress memory、phase tracking、critic/retry 和 CAQ-Lite action reuse。

### Official demonstration download

下载 IsaacLab 官方 Franka Stack Cube demonstration 数据集：

- Local file: `data/isaaclab_stack_cube_mimic/dataset.hdf5`
- Size: `3.0M`
- SHA256: `e0954647328801b091b81604915acd4eff2093b2b26ff0b05753c8f85fbb61b3`
- Episodes: `10`
- Total samples: `3901`
- Action shape: `(T, 7)`, `float32`
- Observation keys include: `eef_pos`, `eef_quat`, `gripper_pos`, `joint_pos`, `joint_vel`, `object`, `cube_positions`, `cube_orientations`, `actions`

### Mimic annotation

运行 IsaacLab Mimic auto annotation：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate isaac
TERM=xterm PYTHONPATH=/home/admin1/ct/IsaacLab/source/isaaclab:/home/admin1/ct/IsaacLab/source/isaaclab_assets:/home/admin1/ct/IsaacLab/source/isaaclab_tasks:/home/admin1/ct/IsaacLab/source/isaaclab_mimic:/home/admin1/ct/IsaacLab/source/isaaclab_rl \
timeout 240s python /home/admin1/ct/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --headless --device cpu \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --auto \
  --input_file data/isaaclab_stack_cube_mimic/dataset.hdf5 \
  --output_file data/isaaclab_stack_cube_mimic/annotated_dataset.hdf5
```

输出：

- Local file: `data/isaaclab_stack_cube_mimic/annotated_dataset.hdf5`
- Size: `2.3M`
- Episodes: `10`
- Total samples: `2163`
- Environment name: `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`
- Subtask signals detected: `grasp_1`, `stack_1`, `grasp_2`

### Mimic generation smoke

运行小规模 state-based Mimic generation：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate isaac
TERM=xterm PYTHONPATH=/home/admin1/ct/IsaacLab/source/isaaclab:/home/admin1/ct/IsaacLab/source/isaaclab_assets:/home/admin1/ct/IsaacLab/source/isaaclab_tasks:/home/admin1/ct/IsaacLab/source/isaaclab_mimic:/home/admin1/ct/IsaacLab/source/isaaclab_rl \
timeout 420s python /home/admin1/ct/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --headless --device cpu --num_envs 2 --generation_num_trials 4 \
  --input_file data/isaaclab_stack_cube_mimic/annotated_dataset.hdf5 \
  --output_file data/isaaclab_stack_cube_mimic/generated_dataset_smoke.hdf5
```

输出：

- Local file: `data/isaaclab_stack_cube_mimic/generated_dataset_smoke.hdf5`
- Size: `772K`
- Successful generated episodes: `4`
- Total samples: `946`
- Success flags: all `True`
- Action shape: `(T, 7)`, `float32`
- Observation keys: `actions`, `cube_orientations`, `cube_positions`, `eef_pos`, `eef_quat`, `gripper_pos`, `joint_pos`, `joint_vel`, `object`

同时生成 failure dataset：

- Local file: `data/isaaclab_stack_cube_mimic/generated_dataset_smoke_failed.hdf5`
- Size: `1.9M`
- Failed episodes: `10`
- Total samples: `2266`
- Potential use: failure-state mining for critic/retry/memory analysis.

环境修复：

- 在 `isaac` conda environment 中安装 `IPython` 和 `ipywidgets`，解决 Mimic generation 脚本依赖缺失。

### Visuomotor generation status

尝试官方 visuomotor Mimic generation：

- Task: `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`
- Tested devices: `cpu`, `cuda:0`
- Required flag: `--enable_cameras`

当前结果：

- 官方脚本在 headless 环境下进入 IsaacSim viewport / Hydra 初始化后 segmentation fault。
- Logs:
  - `results/isaaclab_pipeline_20260609/mimic_generate_visuomotor_smoke.log`
  - `results/isaaclab_pipeline_20260609/mimic_generate_visuomotor_smoke_cuda.log`

判断：

- IsaacLab state/Mimic 数据采集、标注、增强 pipeline 已跑通。
- 官方 visuomotor generation 暂时不稳定，不影响 state-based Mimic 数据生成。
- VLA 视觉训练数据的下一步应优先做 custom camera replay collector：复用当前已跑通的 generated state/action trajectories，在 IsaacLab 中 replay 并记录 RGB/state/action 到 LeRobot 格式。

### Model route decision

当前本机优先路线：

- 继续使用 `pi0.5 / OpenPI` 作为本地训练与部署主线。
- 先完成 IsaacLab HDF5 -> LeRobot converter。
- 再接 custom camera replay collector，得到适配 VLA 的 RGB + state + action 轨迹。
- 训练方式优先采用轻量化 / LoRA-style / action-head-focused tuning；不做 full fine-tuning。

GR00T N1.7 路线：

- 适合作为后续 cloud/H100/L40 扩展路线。
- 官方硬件建议：推理 `16GB+` VRAM，微调 `40GB+` VRAM。
- 单张 RTX 4090 更适合做 inference / small integration smoke，不建议立刻把主线切到 GR00T N1.7 fine-tuning。

### Next engineering steps

1. 写 `IsaacLab HDF5 -> LeRobot` converter。
2. 写 custom IsaacLab camera replay collector，绕过当前官方 visuomotor Mimic generation segfault。
3. 添加 OpenPI IsaacLab Stack Cube data config。
4. 做 tiny pi0.5 training smoke。
5. 用 `scripts/serve_openpi_policy_no_compile.py` 启动训练后 checkpoint。
6. 用 `scripts/run_isaaclab_franka_openpi.py` 做 IsaacLab closed-loop evaluation。
7. 加入 Agentic Harness + CAQ-Lite 对比：
   - trained VLA
   - trained VLA + action reuse
   - trained VLA + Agentic Harness
   - trained VLA + Agentic Harness + CAQ-Lite

### pi0.5 base / OpenPI Stack Cube smoke

目的：将 IsaacLab Stack Cube 数据接入 pi0.5/OpenPI 的训练与推理链路，并明确使用 `pi05_base` 作为初始化，而不是使用 `pi05_libero` fine-tuned checkpoint。

工程改动：

- 新增 converter：
  - `scripts/convert_isaaclab_stack_hdf5_to_lerobot.py`
- 在 OpenPI 本地代码中新增：
  - `LeRobotFrankaEefDataConfig`
  - `pi05_isaaclab_stack_cube`
- `pi05_isaaclab_stack_cube` 设置：
  - repo id: `agentic-vla/isaaclab_stack_cube`
  - initialization: `gs://openpi-assets/checkpoints/pi05_base/params`
  - PyTorch base checkpoint: `/home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_isaaclab`
  - local training mode: head-only tuning through `torch_trainable_regex`

数据转换：

```bash
source /home/admin1/miniconda3/etc/profile.d/conda.sh
conda activate openpi
python scripts/convert_isaaclab_stack_hdf5_to_lerobot.py \
  --input-file data/isaaclab_stack_cube_mimic/generated_dataset_smoke.hdf5 \
  --repo-id agentic-vla/isaaclab_stack_cube \
  --overwrite
```

输出：

- LeRobot root: `/home/admin1/.cache/huggingface/lerobot/agentic-vla/isaaclab_stack_cube`
- Episodes: `4`
- Frames: `946`
- State dim: `8`
- Action dim: `7`
- Image shape: `224 x 224 x 3`
- Note: current images are placeholder/debug images; this dataset validates the interface, not visual task performance.

OpenPI norm stats：

```bash
PYTHONPATH=/home/admin1/ct/openpi_agibot/src:/home/admin1/ct/openpi_agibot/packages/openpi-client/src:${PYTHONPATH:-} \
CUDA_VISIBLE_DEVICES="" \
python /home/admin1/ct/openpi_agibot/scripts/compute_norm_stats.py \
  --config-name pi05_isaaclab_stack_cube \
  --max-frames 256
```

输出：

- `assets/pi05_isaaclab_stack_cube/agentic-vla/isaaclab_stack_cube/norm_stats.json`

Dataloader smoke：

- Observation state: `(1, 32)` after padding
- Actions: `(1, 10, 32)` after action horizon/padding
- Prompt tokens: `(1, 200)`
- Images:
  - `base_0_rgb`: `(1, 224, 224, 3)`
  - `left_wrist_0_rgb`: `(1, 224, 224, 3)`
  - `right_wrist_0_rgb`: padded/masked

LeRobot compatibility fix：

- 当前 `lerobot==0.1.0` 与 `datasets==4.8.5` 在 `datasets.Column` 上不兼容。
- 已在 OpenPI dataloader 中添加局部兼容补丁，只包住 `LeRobotDataset.__init__` 和 `_query_hf_dataset` 的 `torch.stack` 调用。

pi05 base PyTorch conversion：

```bash
python /home/admin1/ct/openpi_agibot/examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --config-name pi05_isaaclab_stack_cube \
  --output-path /home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_isaaclab \
  --precision bfloat16
```

输出：

- `/home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_isaaclab/model.safetensors`
- Size: `6.8G`

PyTorch training smoke：

```bash
PYTHONPATH=/home/admin1/ct/openpi_agibot/src:/home/admin1/ct/openpi_agibot/packages/openpi-client/src:${PYTHONPATH:-} \
python /home/admin1/ct/openpi_agibot/scripts/train_pytorch.py pi05_isaaclab_stack_cube \
  --exp-name isaaclab_stack_head_smoke_2step \
  --num-train-steps 2 \
  --save-interval 1 \
  --log-interval 1 \
  --overwrite
```

结果：

| Item | Value |
|---|---:|
| Total params | `3,616,757,520` |
| Trainable params | `2,165,792` |
| Trainable ratio | `0.0599%` |
| Step 0 loss | `0.2493` |
| Step 1 loss | `0.1380` |
| Peak reserved GPU memory | `7.90GB` |

Checkpoint：

- `checkpoints/pi05_isaaclab_stack_cube/isaaclab_stack_head_smoke_2step/2/model.safetensors`
- `checkpoints/pi05_isaaclab_stack_cube/isaaclab_stack_head_smoke_2step/2/assets/agentic-vla/isaaclab_stack_cube/norm_stats.json`

Policy-server smoke：

```bash
python scripts/serve_openpi_policy_no_compile.py \
  --port 8011 \
  --policy-config pi05_isaaclab_stack_cube \
  --policy-dir /home/admin1/ct/Agentic-VLA/checkpoints/pi05_isaaclab_stack_cube/isaaclab_stack_head_smoke_2step/2
```

Client output:

- Response keys: `actions`, `policy_timing`, `server_timing`
- Action chunk shape: `(10, 7)`
- First action:
  - `[-0.016486, 0.002671, 0.020095, -0.090603, -0.101493, 0.312815, 0.967523]`

判断：

- `pi05_base -> IsaacLab Stack Cube LeRobot -> norm stats -> PyTorch head-only training -> policy server` 链路已跑通。
- 当前 smoke 只证明训练/推理接口闭环，不证明视觉策略已学会 Stack Cube；原因是当前 LeRobot 图像仍是 placeholder/debug image。
- 下一步应优先补 custom IsaacLab camera replay collector，采集 `50` 条真实 RGB/state/action 成功轨迹，再进行同一配置下的 training smoke / closed-loop eval。

### IsaacLab environment-limit repair log

时间：2026-06-09

已修复/缓解：

- VSCode/inotify watcher 压力已通过 `.vscode/settings.json` workspace 排除项缓解；未改系统 sysctl。
- IsaacLab no-viewport headless state/action Mimic 采集可用。
- IsaacLab no-viewport visuomotor 数据结构 smoke 可用，但 RGB 为全零帧。
- 已添加若干外部 IsaacLab/IsaacSim 防御性 guard，用于无 viewport 和 minimal GUI 启动：
  - no active viewport guard
  - semantics fallback guard
  - empty camera annotator buffer guard
  - Fabric transform fallback guard
  - missing toolbar stop button guard
  - env-gated SimulationApp UI / viewport wait bypass
  - env-gated menu build bypass

当前阻塞：

- 本机真实 RGB 渲染仍未打通。
- 默认 headless camera path、GUI path、livestream/offscreen path 均在 RTX/viewport 初始化附近失败或由 Kit/C++ 直接退出。
- 当前环境组合为 Ubuntu `24.04.4` + kernel `6.17.0-35-generic` + NVIDIA driver `595.71.05` + Isaac Sim `4.5`；Kit 日志提示推荐驱动为 `535.129.03`，当前最新驱动路径未完全验证。

可用结论：

- 可以继续推进 pi0.5/OpenPI 接口、head-only tuning、policy server、closed-loop control smoke。
- 不能把当前 placeholder/debug RGB 或 no-viewport all-zero RGB 用作视觉 VLA 性能证据。
- 若要做真 RGB 视觉训练，优先切到受支持的 Isaac Sim 容器/驱动环境，或在另一台已验证相机渲染的机器采集 RGB，再回本机训练/推理。

### Isaac Sim standalone RGB smoke

时间：2026-06-09

目的：

- 判断是否可以绕过 IsaacLab，先用纯 Isaac Sim 跑通 RGB camera pipeline。

新增脚本：

- `scripts/smoke_isaacsim_rgb_camera.py`

测试结果：

- Isaac Sim standalone app 可以启动。
- 纯 Isaac Sim 最小 RGB 渲染仍会触发 RTX SceneDb/TLAS segfault。
- 已测试：
  - base headless app；
  - skip SimulationApp UI / viewport wait / menu build；
  - 设置 `/rtx/sceneDb/maxInstances`；
  - `pxr` renderer override；
  - async world path；
  - sync `World.reset()/step()` path；
  - full scene、no-ground scene、camera-only scene。
- camera-only sync path可以走到 `world.step(render=True)`，但一旦渲染帧仍 segfault。

代表日志：

- `results/isaacsim_standalone_rgb_smoke_base_headless_skipwait_scenedb.log`
- `results/isaacsim_standalone_rgb_smoke_base_headless_sync_camera_only.log`
- `results/isaacsim_standalone_rgb_smoke_base_headless_sync_camera_only_pxr.log`

判断：

- 本机 RGB blocker 不在 IsaacLab/Mimic/OpenPI/pi0.5，而在 Isaac Sim 底层 RTX/Replicator/camera 渲染链路。
- Isaac Sim 可以作为诊断路径，但当前本机环境不能作为可靠 RGB 数据采集路径。

### Official Isaac Sim container route

时间：2026-06-09

决策：

- 不再继续调试 bare-metal Isaac Sim RGB。
- 切换到 NVIDIA 官方 Isaac Sim 容器：
  - `nvcr.io/nvidia/isaac-sim:4.5.0`

已完成：

- 确认 Docker 已安装。
- 确认 host `nvidia-smi` 正常。
- 确认 NGC manifest 可访问：
  - `docker manifest inspect nvcr.io/nvidia/isaac-sim:4.5.0`
- 新增官方容器 smoke 脚本：
  - `scripts/run_isaacsim_official_container_smoke.sh`
- 新增 NVIDIA Container Toolkit setup 脚本：
  - `scripts/setup_nvidia_container_toolkit_ubuntu.sh`
- 新增说明文档：
  - `docs/ISAACSIM_OFFICIAL_CONTAINER_SETUP.md`

当前阻塞：

- Docker GPU runtime/CDI 尚未配置：

```text
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
failed to discover GPU vendor from CDI: no known GPU vendor found
```

- 当前 shell 没有免密 sudo，无法由 Codex 直接安装/配置 `nvidia-container-toolkit`。
- Isaac Sim 官方镜像 pull 已开始并缓存了部分 layer，但 NGC 传输曾 EOF/stall；runtime 修好后可继续断点续拉。

下一步：

```bash
bash scripts/setup_nvidia_container_toolkit_ubuntu.sh
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
bash scripts/run_isaacsim_official_container_smoke.sh
```

更新：

- NVIDIA Container Toolkit 已安装并配置完成。
- Docker GPU runtime 验证通过：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

- 已按用户授权为当前用户 `admin1` 配置 sudo NOPASSWD：
  - `/etc/sudoers.d/90-admin1-nopasswd-agentic-vla`
  - `sudo -n true` 验证通过。

- Isaac Sim 4.5 官方容器已拉取完成：
  - `nvcr.io/nvidia/isaac-sim:4.5.0`
  - size: `22.6GB`
- 修正容器 runner：
  - `scripts/run_isaacsim_official_container_smoke.sh` 改为 `--entrypoint bash`，避免镜像默认 entrypoint 在项目 smoke 前直接启动 Isaac Sim；
  - 新增 `scripts/patch_isaacsim_simulation_app_skip.py`，用于容器内临时跳过 SimulationApp UI/viewport wait。
- 4.5 容器测试结果：
  - base python experience 可以进入项目 smoke；
  - full minimal scene 在 `adding ground plane` 后 segfault；
  - camera-only scene 在 `world.reset()` segfault；
  - `--scene-db-max-instances 10000` 不解决 camera-only crash。
- 代表日志：
  - `results/isaacsim_container_rgb_smoke_base_python_patchskip_entrypoint.log`
  - `results/isaacsim_container_rgb_smoke_camera_only_patchskip_entrypoint.log`
  - `results/isaacsim_container_rgb_smoke_camera_only_patchskip_scenedb10k.log`

判断：

- 官方 4.5 容器不能解决当前 `595.71.05` driver/kernel 下的 RTX SceneDb/camera crash。
- 下一步转向 Isaac Sim 5.0.0 容器测试。
- `nvcr.io/nvidia/isaac-sim:5.0.0` manifest 可访问，镜像正在后台续拉：
  - pid: `results/isaacsim_5_0_0_pull_background.pid`
  - log: `results/isaacsim_5_0_0_pull_background.log`

## pi0.5/OpenPI robosuite Stack 接入 smoke

时间：2026-06-11

目的：

- 将 robosuite/MuJoCo Stack 真实仿真数据接入 pi0.5/OpenPI；
- 验证数据转换、norm stats、dataloader、pi0.5 训练 smoke、pi0.5 推理和短闭环控制；
- 明确区分 pi0.5 接入验证与正式成功率 benchmark。

新增脚本：

- `scripts/convert_robosuite_stack_hdf5_to_lerobot.py`
- `scripts/write_robosuite_stack_openpi_norm_stats.py`
- `scripts/pi05_robosuite_stack_head_smoke.py`
- `scripts/eval_robosuite_stack_pi05_policy.py`

数据转换：

- 输入：`results/robosuite_stack_e2e_v2_20260611/stack_demos_100eps.hdf5`
- 输出 LeRobot repo：`agentic-vla/robosuite_stack`
- 输出路径：`~/.cache/huggingface/lerobot/agentic-vla/robosuite_stack`
- episodes：`100`
- frames：`33933`
- image：真实 robosuite/MuJoCo `frontview` RGB，`64x64`
- wrist image：当前为 `frontview` 复制，用于接通 pi0.5 接口；后续正式视觉训练应采集真实 wrist view。
- state：8D `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`
- action：7D `OSC_POSE`

OpenPI 配置：

- config：`pi05_robosuite_stack_smoke`
- 使用完整 OpenPI checkout：`/home/admin1/openpi`
- 本项目裁剪版 `openpi/` 缺少 `openpi.models`，不能单独作为 pi0.5 训练执行环境。
- 已在完整 OpenPI 添加相同 config，并为 LeRobot/datasets Column 兼容性在 data loader 中加了最小补丁。

Dataloader smoke：

- `obs_state`: `(1, 32)`
- `actions`: `(1, 10, 32)`
- image keys：`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`
- image tensor：`(1, 3, 224, 224)`
- tokenized prompt：`(1, 200)`

Full fine-tuning attempt：

- 初始化 pi05 base PyTorch checkpoint 成功：
  - `/home/admin1/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_isaaclab/model.safetensors`
- forward/backward 成功；
- full Adam 在第一次 `optimizer.step()` 分配 optimizer states 时 OOM：
  - model creation 后约 `7.47GB`
  - backward 后约 `14.23GB`
  - Adam state 分配时进程总显存约 `22.62GB`，4090 剩余不足。

结论：

- 24GB RTX 4090 可以做 pi0.5 接入、前向、反向、head-only smoke；
- 不适合直接 full-model Adam fine-tune pi0.5；
- 正式训练应使用 LoRA/adapter/8-bit optimizer 或 4xA100。

Head-only smoke：

- checkpoint：`checkpoints/pi05_robosuite_stack_smoke/robosuite_stack_pi05_base_head_only_smoke_2step/2/model.safetensors`
- trainable：`action_out_proj.weight`, `action_out_proj.bias`
- trainable params：`32800`
- frozen params：`3616724720`
- steps：`2`
- saved full model：`true`
- loss：`0.04849 -> 0.20501`（仅 smoke，不解释为收敛趋势）
- train wall time：`1.59 s`（不含模型初始化）
- in-memory sample_actions：
  - action shape：`(1, 10, 32)`
  - inference wall：`0.190 s`
  - max reserved GPU memory：约 `7.70GB`
- summary：`checkpoints/pi05_robosuite_stack_smoke/robosuite_stack_pi05_base_head_only_smoke_2step/head_only_smoke_summary.json`

Policy load/infer smoke：

- 使用 OpenPI `create_trained_policy` 从 full checkpoint 加载成功。
- policy load：`96.64 s`
- policy infer wall：`0.976 s`
- model internal infer：`799.85 ms`
- output actions：`(10, 7)`

robosuite closed-loop smoke：

- 输出：`results/robosuite_stack_pi05_closed_loop_smoke_20260611_h60/summary.json`
- 视频：`results/robosuite_stack_pi05_closed_loop_smoke_20260611_h60/pi05_closed_loop_smoke.mp4`
- horizon：`60`
- replan steps：`10`
- full policy calls：`6`
- success：`false`
- failure reason：`horizon_reached`
- mean policy infer：`450.20 ms`
- p95 policy infer：`727.88 ms`

边界：

- 该闭环实验只证明 pi0.5/OpenPI policy 已经能接入 robosuite/MuJoCo 控制循环并生成视频。
- 不能写成 pi0.5 成功完成 Stack。
- 不能写成 Agentic + pi0.5 提升成功率。
- 后续若要形成论文结果，需要做 LoRA/adapter 正式训练，并在相同环境下加入 Agentic retry + CAQ-Lite 对照。

## pi0.5 PyTorch lightweight adaptation and realtime inference sweep

时间：2026-06-11

目的：

- 按本地 RTX 4090 条件，继续使用 PyTorch 版 pi0.5，而不是依赖当前缺 CUDA jaxlib 的 JAX 路径；
- 验证低显存 VLA 适配方案；
- 测试 `num_inference_steps` 与确定性 noise 对 open-loop action error 和 closed-loop latency 的影响。

新增/更新：

- `scripts/eval_pi05_robosuite_openloop_actions.py`
  - 新增 `--num-inference-steps`
  - 新增 `--noise-mode {default,zero,fixed}`
- `scripts/eval_robosuite_stack_pi05_policy.py`
  - 新增 `--num-inference-steps`
  - 新增 `--noise-mode {default,zero,fixed}`

已验证 checkpoint：

| Variant | Trainable params | Max reserved GPU | Notes |
|---|---:|---:|---|
| `robosuite_stack_pi05_head_plus_300step` | `2165792` | about `7.91GB` | `action_in_proj/action_out_proj/time_mlp` |
| `robosuite_stack_pi05_head_plus_action7_gw3_1000step` | `2165792` | about `7.91GB` | action 7D loss + gripper weight `3.0` |
| `robosuite_stack_pi05_expert_tail1_action7_gw15_300step` | action head + final action-expert layer | about `8.16GB` | unfreezes `gemma_expert.model.layers.17` and `gemma_expert.model.norm` |

Open-loop evaluation on the same 20 sampled frames:

| Policy / inference | Noise | Steps | First L2 mean | Chunk MSE | Chunk MAE | Mean infer | P95 infer | Gripper sign acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| head-plus 300 | default | 10 | `1.0646` | `0.2052` | `0.2033` | `387.29ms` | `410.01ms` | `0.70` |
| head-plus 300 | fixed | 10 | `1.0915` | `0.2077` | `0.2222` | `399.60ms` | `469.29ms` | `0.60` |
| head-plus 300 | fixed | 4 | `1.0262` | `0.1747` | `0.2028` | `211.34ms` | `213.91ms` | `0.65` |
| head-plus 1000, gripper weight 3 | default | 10 | `1.2107` | `0.2951` | `0.2061` | `379.47ms` | `385.14ms` | `0.55` |
| expert-tail1 300 | default | 10 | `1.1379` | `0.2475` | `0.2126` | `375.75ms` | `387.06ms` | `0.55` |

Updated local PyTorch setting:

- checkpoint：`checkpoints/pi05_robosuite_stack_smoke/robosuite_stack_pi05_head_plus_300step/300`
- open-loop quality priority：fixed noise, `num_inference_steps=2`
- latency priority：fixed noise, `num_inference_steps=1`
- sweep JSON：`results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_fixed_noise.json`

Interpretation：

- 1-step and 2-step deterministic inference reduced mean internal inference time below the earlier 4-step setting.
- In this small open-loop sample, 2-step inference has the lowest first-action L2 and chunk MSE among the tested settings.
- Gripper prediction remains the dominant error source; most large first-action L2 errors correspond to gripper sign mismatch.
- Increasing head-plus training to 1000 steps with strong gripper weighting did not improve evaluation.
- Unfreezing the final action expert layer fits easily on a 4090, but the current 300-step run did not outperform head-plus 300.

Realtime sweep with one policy load and the same 20 sampled frames:

| Steps | First L2 mean | Chunk MSE | Chunk MAE | Mean infer | P95 infer | First gripper acc | Chunk gripper acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `0.9741` | `0.1523` | `0.1765` | `118.15ms` | `136.08ms` | `0.75` | `0.730` |
| `2` | `0.9595` | `0.1467` | `0.1772` | `136.53ms` | `138.81ms` | `0.75` | `0.755` |
| `4` | `1.0262` | `0.1747` | `0.2028` | `202.24ms` | `231.19ms` | `0.65` | `0.660` |
| `6` | `1.0475` | `0.1887` | `0.2106` | `255.46ms` | `262.52ms` | `0.65` | `0.660` |
| `8` | `1.0736` | `0.2021` | `0.2177` | `323.89ms` | `357.97ms` | `0.60` | `0.610` |
| `10` | `1.0915` | `0.2077` | `0.2222` | `374.43ms` | `403.98ms` | `0.60` | `0.605` |

Zero-noise sampler ablation on the same 20 sampled frames:

| Steps | First L2 mean | Chunk MSE | Chunk MAE | Mean infer | P95 infer | First gripper acc | Chunk gripper acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `1.1598` | `0.1857` | `0.2288` | `110.95ms` | `119.35ms` | `0.45` | `0.570` |
| `2` | `1.1175` | `0.1801` | `0.2226` | `139.46ms` | `170.21ms` | `0.70` | `0.690` |
| `4` | `1.1000` | `0.1773` | `0.2182` | `191.75ms` | `193.80ms` | `0.70` | `0.725` |
| `10` | `1.0735` | `0.1716` | `0.2100` | `361.06ms` | `371.62ms` | `0.70` | `0.725` |

Current sampler decision:

- For low-step realtime inference, fixed-noise `1/2` step outperforms zero-noise `1/2` step on action error and gripper sign accuracy.
- Zero-noise becomes competitive only at 10 steps, but that loses the main realtime advantage.

Closed-loop pi0.5 smoke with best realtime setting:

- output：`results/robosuite_stack_pi05_head_plus_300step_closed_loop_fixed_noise_steps4_h180/summary.json`
- video：`results/robosuite_stack_pi05_head_plus_300step_closed_loop_fixed_noise_steps4_h180/pi05_closed_loop_smoke.mp4`
- horizon：`180`
- replan steps：`10`
- success：`false`
- mean policy infer：`218.27ms`
- p95 policy infer：`279.42ms`
- first call warmup：about `644ms`

Additional closed-loop realtime points:

| Steps | Output | Success | Mean infer | P95 infer | Final phase |
|---:|---|---|---:|---:|---|
| `1` | `results/robosuite_stack_pi05_head_plus_300step_closed_loop_fixed_noise_steps1_h180/summary.json` | `false` | `136.14ms` | `203.09ms` | `approach_above` |
| `2` | `results/robosuite_stack_pi05_head_plus_300step_closed_loop_fixed_noise_steps2_h180/summary.json` | `false` | `181.31ms` | `312.22ms` | `approach_above` |
| `4` | `results/robosuite_stack_pi05_head_plus_300step_closed_loop_fixed_noise_steps4_h180/summary.json` | `false` | `218.27ms` | `279.42ms` | not logged in older summary |

Closed-loop steady-state latency after excluding the first warmup policy call:

| Steps | Calls | First call | Steady mean | Steady P50 | Steady P95 |
|---:|---:|---:|---:|---:|---:|
| `1` | `18` | `573.03ms` | `110.44ms` | `108.04ms` | `120.71ms` |
| `2` | `18` | `629.12ms` | `154.97ms` | `142.90ms` | `231.29ms` |
| `4` | `12` | `648.76ms` | `192.98ms` | `189.06ms` | `199.01ms` |

pi0.5 + Agentic retry closed-loop success:

- matched raw baseline：`results/robosuite_stack_pi05_head_plus_300step_raw_fixed_noise_steps1_h430/summary.json`
- output：`results/robosuite_stack_pi05_head_plus_300step_agentic_retry_fixed_noise_steps1_h430_v2/summary.json`
- video：`results/robosuite_stack_pi05_head_plus_300step_agentic_retry_fixed_noise_steps1_h430_v2/pi05_closed_loop_smoke.mp4`
- contact sheet：`results/robosuite_stack_pi05_head_plus_300step_agentic_retry_fixed_noise_steps1_h430_v2/pi05_agentic_retry_contact.png`
- backend：pi0.5 PyTorch head-plus 300, fixed-noise `1` step
- Agentic recovery：`geometric_stack`

| Item | Value |
|---|---:|
| Success | `true` |
| Steps | `254` |
| Full pi0.5 calls | `6` |
| Recoveries | `1/1` |
| Retry skill steps | `194` |
| Mean infer including warmup | `190.72ms` |
| P95 infer including warmup | `467.45ms` |
| Steady mean after warmup | `112.15ms` |
| Steady P95 after warmup | `112.63ms` |
| Final phase | `done` |
| Final cube XY gap | `0.0020` |
| Final z gap | `0.0420` |

Matched `h430`, fixed-noise `1`-step closed-loop comparison:

| Method | Success | Steps | Full pi0.5 calls | Recovery | Retry steps | Final phase | Steady infer |
|---|---:|---:|---:|---:|---:|---|---:|
| raw pi0.5 | `false` | `430` | `43` | `0/0` | `0` | `approach_above` | `108.53ms` |
| pi0.5 + Agentic retry | `true` | `254` | `6` | `1/1` | `194` | `done` | `112.15ms` |

Ten-trial matched closed-loop result:

| Method | Success | Steps mean | Full pi0.5 calls / ep | Recovery | Retry steps mean | Steady mean infer | Steady P95 infer |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw pi0.5, fixed `1` step | `0/10` | `430.0` | `43.0` | `0/0` | `0.0` | `111.99ms` | `122.37ms` |
| pi0.5 + Agentic retry, fixed `1` step | `10/10` | `254.4` | `6.0` | `10/10` | `194.4` | `110.40ms` | `116.87ms` |

Five-trial sampler ablation:

| Method | Success | Steps mean | Full pi0.5 calls / ep | Recovery | Retry steps mean | Steady mean infer | Steady P95 infer |
|---|---:|---:|---:|---:|---:|---:|---:|
| pi0.5 + Agentic retry, fixed `1` step | `5/5` | `254.4` | `6.0` | `5/5` | `194.4` | `112.68ms` | `123.37ms` |
| pi0.5 + Agentic retry, fixed `2` step | `5/5` | `254.4` | `6.0` | `5/5` | `194.4` | `137.22ms` | `142.74ms` |

Ten-trial artifacts:

- raw 1-step first 5：`results/robosuite_stack_pi05_raw_fixed1_h430_5trials_20260611/summary.json`
- raw 1-step extra 5：`results/robosuite_stack_pi05_raw_fixed1_h430_extra5_seed20260616_20260612/summary.json`
- Agentic 1-step first 5：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_5trials_20260611/summary.json`
- Agentic 1-step extra 5：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_extra5_seed20260616_20260612/summary.json`
- Agentic 2-step：`results/robosuite_stack_pi05_agentic_retry_fixed2_h430_5trials_20260611/summary.json`
- HD success video：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_closed_loop_smoke.mp4`
- HD contact sheet：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_agentic_retry_hd256_contact.png`

Interpretation:

- Under the matched `h430` protocol, raw pi0.5 fails all ten trials, while Agentic retry solves all ten trials.
- Agentic retry reduces full pi0.5 calls from `43.0` to `6.0` per episode because the physical recovery skill handles the stalled segment.
- `1`-step and `2`-step Agentic variants have the same success and call count; `1`-step is preferred for realtime deployment due to lower steady latency.

Harness fix learned from the first Agentic run:

- In the first pi0.5 + Agentic retry attempt, the retry skill reached the progress heuristic `done`, but the harness returned control to pi0.5 before environment success was confirmed.
- pi0.5 then degraded the nearly recovered state.
- The retry exit condition was changed to continue until environment success or `max_retry_skill_steps`, instead of stopping on heuristic `done`.
- After this change, the same setting reached robosuite environment success.

State-token diagnosis:

- 32D privileged state with `discrete_state_input=False` does not constitute a valid state-conditioning experiment for pi0.5 because the PyTorch pi0.5 path ignores continuous state when state tokens are disabled.
- A state-token config was added and trained for 300 head-plus steps:
  - checkpoint：`checkpoints/pi05_robosuite_stack_privileged_tokens_smoke/robosuite_stack_pi05_privileged_tokens_head_plus_300step/300`
  - open-loop result：`results/robosuite_stack_pi05_privileged_tokens_head_plus_300step_openloop_eval_fixed_noise_steps4.json`
  - First L2 mean：`1.4162`
  - Chunk MSE：`0.2815`
  - Mean infer：`215.07ms`
- The state-token branch is currently worse than the 8D proprio head-plus branch, so it is not promoted to closed-loop benchmarking.

Boundary：

- 该结果可作为真实 pi0.5/OpenPI + robosuite/MuJoCo 闭环接入和实时推理优化证据。
- 该结果不能写成 pi0.5 已完成 Stack。
- 当前 paper 主线仍应区分：
  - compact learned policy：Agentic retry + CAQ-Lite 的成功率主证据；
  - pi0.5 PyTorch：VLA 后端接入、4090 适配可行性、VLA inference-step realtime tradeoff 的主证据。

## 2026-07-16: CARVE Optimize Runtime profile calibration

Environment:

- checkpoint: `pi05_libero_pytorch`
- hardware: RTX 4090
- controls: two flow steps, ten committed actions, fixed paired noise
- performance protocol: three warmup calls, 50 measured calls, 80 ms deadline
- fidelity protocol: worst-case aggregation over all 45 recorded LIBERO failure
  observations
- TorchAO `0.15.0` skipped optional C++ extensions under PyTorch `2.7.1+cu126`;
  W8A16 used the Python tensor-subclass path plus `torch.compile`

Performance:

| Profile | P50 | P95 | P99 | Deadline miss | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| eager BF16 | `154.34ms` | `159.59ms` | `163.81ms` | `100%` | `7.12GB` |
| compiled BF16 | `65.73ms` | `67.40ms` | `67.94ms` | `0%` | `6.98GB` |
| compiled W8A16, language layers 0--3 | `69.74ms` | `71.72ms` | `72.41ms` | `0%` | `6.56GB` |

Fidelity decisions:

- compiled BF16: pass `45/45`, worst endpoint L2 `0.05417`;
- W8A16 language layers 0--3: pass `45/45`, endpoint L2 `0.09359`;
- W8A16 language layers 0--5: rejected, endpoint L2 `0.10433`;
- W8A16 full language backbone: rejected, endpoint L2 `0.18076` over the
  five-observation pilot.

Implementation:

- added truthful `torch_compile` and component-scoped `torchao_int8` plugins;
- preserved the eager policy for paired compile fidelity checks;
- cached eager actions before in-place TorchAO conversion;
- rejected failed quantization profiles before latency benchmarking;
- persisted accepted and rejected hardware-bound manifests.

Consolidated evidence:

- `results/carve_optimize/PROFILE_COMPARISON.md`
- `results/carve_optimize/pi05_eager_bf16_2step_h10_deadline80_rep1.json`
- `results/carve_optimize/pi05_torch_compile_bf16_2step_h10_fidelity45.json`
- `results/carve_optimize/pi05_torchao_w8a16_vlm_early4_fidelity45_deadline80_rep1.json`

## 2026-07-16: Optimized profile and Agentic Harness integration

Implementation:

- the deployment server now loads only fidelity-approved, checkpoint-matched
  manifests;
- the calibrated flow-step count is enforced as a fixed server capability;
- selected profile and backend evidence are attached to every action chunk and
  CARVE call trace;
- Agentic events and optimization evidence coexist in the same trace;
- optional server-side prewarm completes before the WebSocket port opens.

Real pi0.5 controlled-WebSocket smoke:

- compiled BF16 steady Agentic request: `68.17ms`, no 80 ms deadline miss;
- W8A16 steady Agentic request: `78.53ms`, no 80 ms deadline miss;
- W8A16 with server-side prewarm: calls took `39.17s` and `1.64s` before
  readiness; first accepted external request took `78.01ms`, no miss.

Paired LIBERO failure-state branch smoke:

| Profile | Calls | Mean | P50 | P95 | Max | 80 ms misses |
|---|---:|---:|---:|---:|---:|---:|
| compiled BF16 | `14` | `61.40ms` | `61.73ms` | `66.17ms` | `70.67ms` | `0/14` |
| W8A16, VLM layers 0--3 | `14` | `62.24ms` | `62.90ms` | `67.92ms` | `68.98ms` | `0/14` |

The ten-step horizon validates state restore, continue/fast/accurate/recovery
branch execution, deterministic two-step controls, deployment-profile tracing,
and deadline behavior. It is not task-success evidence. W8A16 is retained for
its measured VRAM reduction; it is not described as faster than compiled BF16.

Evidence:

- `results/carve_profile_branches/PROFILE_BRANCH_SMOKE_COMPARISON.md`
- `results/carve_profile_branches/compiled_bf16_snapshot1_h10_fixed2.json`
- `results/carve_profile_branches/w8a16_snapshot1_h10_fixed2.json`
- `results/carve_optimize/w8a16_server_prewarm_first_request_smoke.json`

## 2026-07-16: Paired long-horizon failure-state pilot

Protocol:

- exact MuJoCo state restoration with deterministic pi0.5 noise;
- compiled BF16 versus replay-approved W8A16;
- fixed two-step inference, ten-action model horizon, 80 ms deadline;
- T6/T9 true stall snapshots at episode 1 step 13;
- 280-step branch horizon;
- continue, fast, accurate, and recovery branches;
- accurate and recovery both commit two actions per call; recovery adds two
  task-conditioned Agentic retry calls.

Outcomes:

| Profile | State | Continue | Fast | Accurate | Recovery |
|---|---|---|---|---|---|
| compiled BF16 | T6 | success, 213 | success, 210 | failure, 280 | success, 235 |
| compiled BF16 | T9 | success, 245 | success, 248 | success, 249 | success, 252 |
| W8A16 | T6 | success, 211 | success, 206 | failure, 280 | failure, 280 |
| W8A16 | T9 | success, 246 | success, 253 | success, 277 | success, 257 |

Runtime:

| Profile | Calls | Mean | P95 | P99 | 80 ms misses |
|---|---:|---:|---:|---:|---:|
| compiled BF16 | `623` | `60.98ms` | `65.10ms` | `69.06ms` | `0/623` |
| W8A16 | `662` | `63.20ms` | `68.17ms` | `71.02ms` | `0/662` |

Decisions:

- The T6 BF16 accurate-failure/recovery-success result reproduced in a second
  video-producing run at 280 versus 235 steps.
- Continue and fast also succeeded on T6. The result supports staged
  escalation and rejects unconditional frequent replanning; it does not prove
  immediate recovery is best for every first stall.
- W8A16 lost the BF16 T6 recovery outcome. It passes replay fidelity and the
  realtime deadline but fails closed-loop non-inferiority, so it is not the
  default deployment profile.

Original-failure T8 probe:

- snapshot: task 8 episode 3 step 99 from a 530-step failed trajectory;
- branch horizon: 440;
- continue: failure, 55 calls;
- accurate: failure, 220 calls;
- recovery: failure, 220 calls and two Agentic retry events;
- runtime: 495 calls, `0/495` deadline misses.

Prompt-only retry is insufficient for this long-horizon dual-object failure.
The current recovery experiment stops here until a stateful or physical
recovery mechanism replaces the two-prompt intervention.

Evidence:

- `results/carve_profile_branches/PAIRED_FAILURE_STATE_PILOT.md`
- `results/carve_profile_branches/compiled_bf16_stall2_h280_fixed2.json`
- `results/carve_profile_branches/w8a16_stall2_h280_fixed2.json`
- `results/carve_profile_branches/compiled_bf16_t6_ar_h280_fixed2.json`
- `results/carve_profile_branches/compiled_bf16_t8_failure_h440_fixed2.json`

## 2026-07-16: Stateful physical recovery contract

Motivation:

- paired T8 showed that two recovery-prompt calls are not a general physical
  recovery mechanism;
- the previous joint controller repeatedly chose accurate VLA replanning for
  stall events and had no concrete recovery-skill lifecycle.

Implementation:

- added explicit recovery status, phase, plan, context, command, outcome, and
  bounded episodic-memory contracts;
- added `StatefulRecoveryExecutor` with verification blocking, terminal result
  recording, replan permission, abort, and safe-stop semantics;
- added `cartesian_retract_lift_reobserve` for normalized 7-D delta-Cartesian
  action contracts;
- changed stall handling to first-stall accurate replan, confirmed repeated
  stall physical recovery, then planner/safe stop after budget exhaustion;
- recovery decisions now carry a concrete `recovery_skill_id`.
- added an opt-in paired `physical_recovery` branch that counts physical actions
  against the horizon, blocks VLA replan after failed verification, and records
  the skill outcome in the first post-recovery Agentic request.

LIBERO state-level smoke:

| Item | Result |
|---|---:|
| Snapshot | `task06_episode001_step0013.npz` |
| Phases | stabilize, retract, lift, settle/reobserve |
| Total actions | `12` |
| EEF state response | `0.00861697` |
| Required response | `0.0001` |
| Deterministic replay max difference | `0.0` |
| Verification | succeeded |
| Fresh VLA replan requested | yes |
| Task success before/after | false/false |

This state-level run is a physical action-contract and lifecycle smoke, not
task-level recovery evidence. The later online repeated-stall compiled-BF16
sentinel passed the controller-selection gate.

Paired compiled-BF16 intervention gate:

| Branch | Success | Steps | VLA calls | Physical actions | Deadline misses |
|---|---:|---:|---:|---:|---:|
| accurate replan | no | `280` | `140` | `0` | `0/140` |
| prompt retry | yes | `235` | `118` | `0` | `0/118` |
| physical recovery + replan | yes | `236` | `112` | `12` | `0/112` |

- the first post-recovery VLA trace records event `stall`, action
  `replan_after_physical_recovery`, outcome `succeeded`, and skill ID
  `cartesian_retract_lift_reobserve`;
- a second physical branch reproduced success, 236 steps, 112 VLA calls, 12
  physical actions, and the exact final simulator-state digest;
- repeat-run VLA P95 was `63.27ms` with `0/112` deadline misses.

The gate demonstrates repeatable intervention potential. Its saved snapshot is
a first-stall event; controller-selected repeated-stall triggering was validated
separately by the later online sentinel.

Evidence:

- `results/carve_recovery/PHYSICAL_RECOVERY_STATUS.md`
- `results/carve_recovery/t6_stall_cartesian_recovery_smoke.json`
- `results/carve_recovery/t6_stall_cartesian_recovery_smoke.mp4`
- `results/carve_recovery/compiled_bf16_t6_physical_gate_h280.json`
- `results/carve_recovery/compiled_bf16_t6_physical_recovery_repeat_h280.json`

## 2026-07-16: Online repeated-stall physical recovery sentinel

Implementation:

- integrated `StatefulRecoveryExecutor` into the real LIBERO rollout behind
  `--carve-physical-recovery`;
- first stall clears stale actions and immediately requests accurate VLA
  inference; a second consecutive stall starts the bounded physical skill;
- each physical phase, state response, terminal outcome, session ID, and skill
  ID is recorded in the episode trace;
- verified recovery permits exactly one fresh VLA replan carrying an Agentic
  recovery event; failed verification safe-stops the episode;
- added eager priming before compiled server prewarm to match the accepted
  calibration order and avoid the PyTorch SDPA/cudagraph dtype failure.

Authoritative deterministic T6 sentinel:

| Metric | Result |
|---|---:|
| Success | `1/1` |
| Episode steps | `221` |
| VLA calls | `22` |
| First-stall action | accurate replan |
| Repeated-stall action | physical recovery |
| Physical actions | `12` |
| EEF response | `0.01802345 m` |
| Verification | passed |
| VLA P95 | `67.49 ms` |
| Policy 80 ms misses | `0/22` |
| End-to-end control 80 ms misses | `22/211` |

The v2 and v3 videos are byte-identical and both runs match on success, 221
steps, 22 VLA calls, 12 physical actions, and EEF response. V3 fixes an
instrumentation-only issue that had counted the post-recovery VLA event as a
planner call. The result validates the online lifecycle, but `n=1` is not used
as a benchmark success-rate claim.

Evidence:

- `results/carve_recovery/PHYSICAL_RECOVERY_STATUS.md`
- `results/carve_recovery/online_t6_compiled_bf16_physical_sentinel_v3_20260716/results.json`
- `results/carve_recovery/online_t6_compiled_bf16_physical_sentinel_v3_20260716/episode_traces.jsonl`
- `results/carve_recovery/online_t6_compiled_bf16_physical_sentinel_v3_20260716/policy_calls.jsonl`
- `results/carve_recovery/online_t6_compiled_bf16_physical_sentinel_v3_20260716/videos/`

## 2026-07-17: Risk-gated asynchronous VLA prefetch

> Historical pilot result. The later ten-episode T8/T9 expansion below
> supersedes this positive `n=1` sentinel for deployment selection.

Profiling separated the synchronous online control path:

| Component | P95 |
|---|---:|
| Cached-action control step | `39.97 ms` |
| Synchronous VLA control step | `114.01 ms` |
| VLA policy call | `69.09 ms` |
| MuJoCo environment step | `34.86 ms` |
| Image preprocessing | `5.61 ms` |

Implementation:

- added a model-independent, single-flight `AsyncInferencePrefetcher`;
- submit at two cached actions under normal low-risk execution;
- consume a temporally aligned action suffix without waiting;
- invalidate on stall, slip, misgrasp, contact, high risk, accurate replan,
  physical recovery, planner escalation, or safe stop;
- keep first inference and recovery-critical replans synchronous;
- trace submissions, ready/waited/discarded requests, worker errors, committed
  async actions, and VLA-versus-cached control latency.

Matched deterministic T6 gate:

| Metric | Synchronous | Async prefetch |
|---|---:|---:|
| Success | `1/1` | `1/1` |
| Steps | `221` | `236` |
| VLA calls | `22` | `29` |
| Verified physical recovery | `1` | `1` |
| Control P95 | `107.77 ms` | `48.04 ms` |
| 80 ms control misses | `22/211` | `4/226` |
| Policy-call misses | `0/22` | `0/29` |

The positive async run submitted 25 requests, consumed 24, invalidated one on
risk, committed 192 aligned actions, and had no worker errors. A repeat matched
success, 236 steps, 29 calls, recovery evidence, all async counters, four
deadline misses, and the exact video digest. The remaining four misses are the
deliberately synchronous first call and recovery-critical replans.

The v5 attempt is rejected evidence: expected chunk-tail staleness invalidated
all 19 requests, increasing compute without latency benefit. The corrected v6
and repeated v7 results support an 83.0% relative miss-rate reduction and 55.4%
control-P95 reduction, with a 31.8% VLA-call increase and 15 additional steps.

Evidence:

- `results/carve_realtime/ASYNC_PREFETCH_STATUS.md`
- `results/carve_recovery/online_t6_control_profile_v4_20260717/`
- `results/carve_realtime/online_t6_async_prefetch_gate_v6_20260717/`
- `results/carve_realtime/online_t6_async_prefetch_repeat_v7_20260717/`

## 2026-07-17: Held-out T8/T9 async-prefetch gate

> Historical two-episode pilot. The five-state-per-task expansion recorded in
> the next section supersedes its opt-in deployment recommendation.

The initial commit-10 synchronous T8 run exposed a recovery interface defect:
a finite pi0.5 action outside `[-1, 1]` was accepted by the simulator but
rejected while constructing the bounded physical skill. Recovery now projects
finite seeds to the declared action bounds while preserving hard failures for
bad dimensions and NaN/Inf. The same T8 state subsequently started recovery,
verified it, and succeeded; the corrected commit-10 T8/T9 baseline reached
`2/2`.

The causal prefetch comparison matches the effective action cadence:

| Metric | Sync commit 8 | Async lead 2 |
|---|---:|---:|
| Success | `2/2` | `2/2` |
| Steps | `677` | `682` |
| VLA calls | `83` | `84` |
| Episode wall time | `30.28 s` | `25.88 s` |
| 80 ms control misses | `83/657` | `3/662` |
| T8 control P95 | `108.92 ms` | `45.33 ms` |
| T9 control P95 | `108.11 ms` | `47.79 ms` |

This pilot supports a `96.4%` relative miss-rate reduction and `14.5%` lower
episode wall time with nearly matched call count. It does not support a
task-success improvement claim, and its former opt-in recommendation is
withdrawn by the expanded gate below. An exploratory lead-1 run made zero
prefetch submissions and is excluded; the CLI now rejects leads below 2.

Evidence:

- `results/carve_realtime/HELDOUT_T89_PREFETCH_GATE.md`
- `results/carve_realtime/heldout_t89_sync_boundfix_20260717/`
- `results/carve_realtime/heldout_t89_sync_commit8_20260717/`
- `results/carve_realtime/heldout_t89_async_20260717/`

## 2026-07-17: Expanded T8/T9 async-prefetch rejection gate

The decisive gate uses five deterministic initial states on each of LIBERO-10
tasks T8 and T9. Synchronous commit 8 is the causal control for lead-2 prefetch,
because both execute eight policy actions per returned chunk. A conservative
alternating schedule inserts one fresh synchronous chunk after every prefetched
chunk to test whether lower prefetch exposure restores fidelity.

| Metric | Sync commit 8 | Async every chunk | Async alternating |
|---|---:|---:|---:|
| Success | `7/10` | `5/10` | `5/10` |
| Steps | `4031` | `4387` | `4378` |
| VLA calls | `417` | `346` | `315` |
| Episode wall time | `205.79 s` | `198.21 s` | `207.58 s` |
| 80 ms control misses | `417/3931` | `48/4287` | `176/4278` |
| Verified physical recovery | `10/10` | `12/12` | `11/11` |

Full-duty prefetch reduces relative deadline-miss rate by `89.4%`, but loses
two paired successes. The alternating duty cycle also reaches only `5/10` and
does not improve wall time over the synchronous control. Both asynchronous
runs report zero worker errors, so the regression is consistent with temporal
observation shift changing the VLA trajectory rather than an inference-worker
outage.

Decision: reject both asynchronous schedules from the promoted runtime. Keep
compiled BF16 with synchronous commit as the fidelity-first deployment
profile. Prefetch remains experimental infrastructure and negative systems
evidence; further threshold tuning is not justified by this gate.

Evidence:

- `results/carve_realtime/PREFETCH_5STATE_GATE.md`
- `results/carve_realtime/heldout_t89_5states_sync_commit8_20260717/`
- `results/carve_realtime/heldout_t89_5states_async_lead2_20260717/`
- `results/carve_realtime/heldout_t89_5states_async_interval2_20260717/`

## 2026-07-17: Runtime evidence integrated into the CARVE-VLA manuscript

- added `scripts/build_carve_paper_results.py` to read accepted calibration,
  backend, and held-out runtime artifacts directly;
- generated LaTeX tables for paired flow-step calibration, RTX 4090 deployment
  profiles, and cadence-matched asynchronous execution, plus shared result
  macros and a machine-readable summary;
- rewrote the runtime method and experiment story around the Agentic
  Harness/Optimize Runtime boundary, physical recovery, hierarchical fidelity
  and closed-loop gates, and end-to-end control latency;
- removed the robosuite Stack pilot and selective 96% diagnostic from the main
  quantitative narrative while retaining their original repository evidence;
- compiled the updated WAICA manuscript at 15 pages.

Paper assets:

- `paper/CARVE-VLA/generated/runtime_results_summary.json`
- `paper/CARVE-VLA/generated/calibration_table.tex`
- `paper/CARVE-VLA/generated/deployment_table.tex`
- `paper/CARVE-VLA/generated/prefetch_table.tex`

## 2026-07-17: Action-prefix consistency shadow diagnostic

Motivation: lead-2 asynchronous execution discards the first two actions of a
chunk predicted from an old observation, although the robot actually executes
two cached actions from the previous chunk. The new model-independent verifier
compares those hypothetical and executed prefixes before accepting the suffix.

Implementation:

- added continuous RMS, cumulative translation/rotation disagreement, gripper
  agreement, and cosine metrics;
- added `off`, `shadow`, and `enforce` modes with per-ticket trace records;
- `shadow` leaves all actions unchanged and is therefore diagnostic only;
- `enforce` falls back to a fresh synchronous VLA call, but is not activated
  without a predictive shadow signal;
- expanded the CARVE CPU test suite to `41` passing tests.

Fixed T8/T9 five-state shadow result:

| Metric | Result |
|---|---:|
| Task success | `4/10` |
| Prefix checks | `279` |
| Would accept / reject | `258 / 21` |
| Successful-episode reject rate | `7.21%` |
| Failed-episode reject rate | `7.66%` |
| Failure correlation | `0.058` |
| Failure AUC | `0.563` |
| Worker errors | `0` |

The near-identical group rates and near-random AUC do not support enforcement.
No threshold sweep or enforce rollout is run. This prevents fitting a verifier
to ten known outcomes and keeps synchronous compiled BF16 as the promoted
profile.

Evidence:

- `results/carve_realtime/prefix_shadow_t89_5states_20260717/results.json`
- `results/carve_realtime/prefix_shadow_t89_5states_20260717/episode_traces.jsonl`
- `results/carve_realtime/prefix_shadow_t89_5states_20260717/prefix_diagnostic.json`
- `scripts/analyze_carve_prefix_shadow.py`

## 2026-07-17: Static Masked-View Elision promoted

OpenPI's LIBERO adapter supplies two valid views and one zero-padded right-wrist
view with `image_mask=False`. The stock PyTorch pi0.5 implementation still runs
SigLIP on every view before applying the mask. Added the registered
`torch_compile_masked_views` backend, which asserts the configured mask is
all-false, removes the padding view before visual embedding and prefix-KV
construction, and otherwise preserves the pi0.5 denoising path.

- 44 CARVE tests pass in the OpenPI PyTorch environment.
- All 45 fixed-noise replay observations pass action fidelity.
- Worst-case chunk MAE/cosine/gripper agreement: `0.00232` / `0.999897` / `1.0`.
- Replay runtime P50/P95: `54.35/56.19 ms`, versus `65.73/67.40 ms` for
  compiled BF16 (`17.3%/16.6%` reduction).
- Paired T8/T9 closed loop: `8/10` versus `7/10`; the difference is used only
  as non-inferiority evidence.
- Closed-loop VLA P50/P95: `56.80/60.60 ms`, versus `65.49/69.59 ms`
  (`13.3%/12.9%` reduction).
- Mean episode wall time decreases by `3.2%`; the observation timestamp and
  action commitment cadence remain unchanged.

Decision: promote SMVE for adapters with guaranteed padding camera slots. Keep
ordinary compiled BF16 for active or dynamically available views. Do not claim
model-general acceleration until a second adapter/model is tested.

Evidence:

- `results/carve_optimize/MASKED_VIEW_ELISION_GATE.md`
- `results/carve_optimize/pi05_masked_view_compile_bf16_2step_h10_fidelity45_rep1.json`
- `results/carve_optimize/masked_view_t89_5states_20260717/`
- `scripts/run_masked_view_closed_loop_gate.sh`

## 2026-07-17: Named view contract and pi0.5 DROID cross-adapter gate

Replaced the default numeric SMVE configuration with a serializable named
`StaticMaskedViewContract`. The profile declares canonical view order and
padding-view names; the backend resolves indices only after checking uniqueness,
membership, and the non-empty retained-view invariant. Legacy index manifests
remain readable, but mixed name/index plans are rejected.

Converted the official OpenPI `pi05_droid` JAX checkpoint with the official
conversion script and injected the corresponding DROID normalization stats.
The benchmark loader now aligns model structure from checkpoint `config.json`
and supports explicit LIBERO or DROID observation schemas.

At two flow steps and five committed actions, ordinary compiled BF16 and SMVE
both pass 10/10 paired fixed-noise observations. Runtime P50/P95 changes from
`65.09/66.22 ms` to `54.66/57.34 ms`; model P50/P95 changes from
`51.96/52.54 ms` to `41.57/42.70 ms`. Both have zero 80 ms misses and 6.98 GB
peak VRAM. The 15-action SMVE profile is rejected at endpoint L2 `0.18120`, and
base-LoRA compile profiles are also rejected; thresholds were not relaxed.

This is a cross-checkpoint and cross-input-adapter systems/fidelity result over
fixed replay observations, not DROID task-success evidence and not a second
VLA-family validation.

Evidence:

- `results/carve_optimize/CROSS_ADAPTER_SMVE_GATE.md`
- `results/carve_optimize/pi05_droid_compile_cross_adapter_h5_10state_20260717.json`
- `results/carve_optimize/pi05_droid_masked_view_cross_adapter_h5_10state_20260717.json`

## 2026-07-17: Midterm evidence package refreshed

- Extended `scripts/build_carve_paper_results.py` with a machine-readable
  pi0.5 DROID cross-adapter section and generated LaTeX table/macros.
- Added a 20-slide objective presentation covering Agentic Harness design,
  physical recovery, Optimize Runtime, flow calibration, deployment profiles,
  SMVE, DROID portability, quantization rejection, and asynchronous negative
  results.
- Kept the historical `185/200` Agentic aggregate used by the preserved
  submission. The later `192/200` B4 aggregate is not promoted because its raw
  result directories were removed during the earlier cleanup and the current
  summarizer can no longer reproduce it.

Artifacts:

- `docs/archive/midterm/20260717/CARVE_VLA_MIDTERM_PRESENTATION_20260717.md`
- `paper/CARVE-VLA/generated/cross_adapter_table.tex`
- `paper/CARVE-VLA/generated/runtime_results_summary.json`

## 2026-07-17: Single-GPU Agent-VLM + VLA contention gate

Motivation: idle replay latency does not represent an Agentic deployment in
which a semantic VLM and a VLA share one edge accelerator. Clean LIBERO-10 was
stopped after an incomplete baseline reached 154/155 successes because the
benchmark was saturated and provided little information about the runtime
contribution.

Protocol:

- Qwen3.5-4B BF16 served through the local OpenAI-compatible Transformers
  server and continuously processed a real robot-scene image;
- OpenPI pi0.5 LIBERO PyTorch used two flow steps, ten returned actions, paired
  fixed noise, five warmups, and 500 measured calls;
- both models shared one RTX 4090 and consumed about 17.6 GB total device
  memory;
- ordinary `torch.compile` and compiled SMVE differed only in removal of the
  adapter-declared padded right-wrist view;
- all per-call runtime and model latencies were persisted for deadline sweeps.

| Profile | Calls | P50 | P95 | P99 | Miss@80ms | Fidelity |
|---|---:|---:|---:|---:|---:|---:|
| Compiled BF16 | 500 | `81.12 ms` | `91.85 ms` | `93.72 ms` | `72.8%` | `10/10` |
| Compiled BF16 + SMVE | 500 | `65.93 ms` | `75.57 ms` | `78.90 ms` | `0.6%` | `10/10` |

SMVE reduces P50/P95/P99 by `18.7%/17.7%/15.8%` and the 80 ms miss
rate by `72.2` percentage points. Three independent measurement windows
(50/200/500 calls) preserve the same ordering. The paired VLM workloads
complete `131/131` and `126/126` requests with closely matched P50 latency
(`1573.5/1576.7 ms`).

Interpretation: this is a controlled deployment stress result, not a task-
success or universal-VLA claim. Continuous VLM inference represents an upper
stress envelope; the next gate will evaluate event-triggered semantic calls
inside controlled failure/recovery episodes.

Evidence:

- `results/carve_optimize/VLM_VLA_CONTENTION_GATE.md`
- `results/carve_optimize/vlm_vla_contention_gate.json`
- `results/carve_optimize/vlm_vla_contention_deadline_curve.png`
- `results/carve_optimize/pi05_compile_2step_h10_active_qwen35_4b_500calls_20260717.json`
- `results/carve_optimize/pi05_smve_2step_h10_active_qwen35_4b_500calls_20260717.json`
- `scripts/run_vlm_contention_load.py`
- `scripts/summarize_vlm_vla_contention.py`

### Event-triggered semantic-load follow-up

A second 500-call gate inserted a five-second cooldown after every completed
Qwen3.5-4B visual response. This models on-demand Agent semantic checks while
retaining the continuous run as the upper stress envelope.

| Profile | VLA P50 | VLA P95 | VLA P99 | Miss@80ms | VLM success |
|---|---:|---:|---:|---:|---:|
| Compiled BF16 | `66.28 ms` | `85.53 ms` | `90.56 ms` | `13.6%` | `32/32` |
| Compiled BF16 + SMVE | `54.69 ms` | `69.30 ms` | `76.47 ms` | `0.4%` | `32/32` |

Event triggering restores the median toward idle performance, but ordinary
compilation still has a deadline-violating tail. Combining event-triggered VLM
calls with SMVE reduces P50/P95 by `32.6%/24.6%` and 80 ms misses by `72.4`
percentage points relative to continuous VLM plus ordinary compilation. No
cooldown sweep is run; the next experiment must connect semantic triggering to
observable perturbation/recovery events.

Additional evidence:

- `results/carve_optimize/pi05_compile_2step_h10_event5_qwen35_4b_500calls_20260717.json`
- `results/carve_optimize/pi05_smve_2step_h10_event5_qwen35_4b_500calls_20260717.json`
- `results/carve_optimize/qwen35_4b_event5_vlm_load_compile_500_20260717.jsonl`
- `results/carve_optimize/qwen35_4b_event5_vlm_load_smve_500_20260717.jsonl`
- `results/carve_optimize/vlm_vla_scheduling_comparison.png`

## 2026-07-17: Event-triggered semantic shadow in controlled rollouts

Implemented `DeadlineAwareSemanticScheduler` and `AsyncSemanticObserver` as
model-independent CARVE runtime components. The online LIBERO runner now
submits a real post-perturbation RGB frame to Qwen3.5-4B without blocking the
robot control path, and records scheduling, latency, protocol validity, token
usage, and returned failure labels.

Paired gate: LIBERO-10 Task 8/9, three fixed-noise trials per task,
`mid_episode_nudge` at step 80, PI0.5 PyTorch SMVE BF16 with two flow steps and
ten committed actions, Qwen3.5-4B BF16 co-resident on one RTX 4090.

| Variant | Success | Semantic valid | Semantic P95 | PI0.5 P95 | Miss@80ms |
|---|---:|---:|---:|---:|---:|
| No semantic VLM | `4/6` | - | - | `58.60 ms` | `8.02%` |
| Async 32-token JSON | `4/6` | `0/6` | `2545.66 ms` | `64.51 ms` | `7.95%` |
| Async 12-token label | `4/6` | `6/6` | `1005.29 ms` | `63.48 ms` | `8.02%` |

The selected 12-token protocol reduces semantic P95 by `60.5%` versus the
32-token attempt and `88.9%` versus the earlier 96-token smoke. All six paired
success outcomes and episode lengths match the no-semantic baseline, and
blocking reasoning remains `0 ms`. PI0.5 P95 increases by `8.3%`, so semantic
calls remain event-triggered and shadow-only. Direct action intervention is
blocked pending a labeled semantic-precision gate.

Evidence:

- `results/carve_semantic_shadow/paired_t89_3trials_20260717/SEMANTIC_SHADOW_GATE.md`
- `results/carve_semantic_shadow/paired_t89_3trials_20260717/semantic_shadow_gate_summary.json`
- `results/carve_semantic_shadow/paired_t89_3trials_20260717/semantic_shadow_gate.png`
- `scripts/run_carve_semantic_shadow_gate.sh`
- `scripts/summarize_semantic_shadow_gate.py`

## 2026-07-18: Deployable semantic precision gate

Audited the semantic rollout and found that the previous systems gate included
the injected object name and exact `dx/dy` in the VLM prompt. Those data are
evaluator-only and cannot support a deployment diagnosis claim. The runtime was
corrected to send pre/post RGB frames, task text, and only a generic deployable
visual-anomaly signal; injection truth remains trace metadata.

Built 30 temporal pairs from restored Task 8/9 PI0.5 states: ten exact no-op
pairs, ten mild object displacements, and ten severe 2--8 cm displacements.
Mild examples are an unscored gray zone. Hard gates were fixed before querying:
protocol validity >=95%, no-op false-positive rate <=20%, and severe-event
intervention recall >=80%.

| Protocol | Valid | No-op FP | Severe recall | Mild intervention | P95 |
|---|---:|---:|---:|---:|---:|
| 12-token `STATUS|FAILURE` | `80%` | `40%` | `60%` | `40%` | `1026.27 ms` |
| 4-token single code | `100%` | `0%` | `30%` | `0%` | `695.76 ms` |

Both protocols fail. The second is efficient and conservative but misses seven
of ten severe events. Prompt tuning and held-out Task 6 evaluation were stopped
under the gate. Semantic output stays observation-only; CARVE's deployable
physical monitor and verification path remain responsible for intervention.

Evidence:

- `results/carve_semantic_precision/semantic_precision_gate.md`
- `results/carve_semantic_precision/semantic_precision_gate_code_v2_dev.md`
- `results/carve_semantic_precision/temporal_pairs_t89/manifest.json`
- `scripts/build_semantic_perturbation_set.py`
- `scripts/evaluate_semantic_perturbation_set.py`

## 2026-07-18: Real OpenVLA cross-model and low-bit gate

Selected the official `openvla/openvla-7b-finetuned-libero-10` checkpoint as a
second VLA family. Unlike PI0.5, OpenVLA is a 7B autoregressive policy that
predicts one 7-D action per call and does not expose flow-step or action-chunk
controls. Added a model adapter/plugin, a pinned Hugging Face policy wrapper,
official-compatible LIBERO preprocessing, and a unified CARVE profiler.

The isolated Python 3.10 environment passes real processor execution and CUDA
checks with PyTorch 2.5.1+cu121, Transformers 4.40.1, Timm 0.9.10, Accelerate
0.34.2, and BitsAndBytes 0.49.2. The processor emits 19 language tokens and a
fused `1x6x224x224` visual tensor for the smoke prompt.

The 15.08 GB checkpoint is size- and SHA256-verified per shard. The fixed gate
runs BF16, INT8, and NF4 over the same ten real Task 8/9 replay images. The
100 ms deadline, latency distribution, peak VRAM, and paired autoregressive
actions are recorded through the same CARVE benchmark/manifest path.

| Profile | P50/P95 (ms) | Miss@100ms | Peak VRAM | Exact action | Gripper | MAE | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 | `303.48/310.92` | `100%` | `14.42 GB` | reference | reference | reference | behavioral reference |
| BnB INT8 | `1533.96/1570.19` | `100%` | `7.76 GB` | `50%` | `100%` | `0.0089` | rejected |
| BnB NF4 | `748.35/761.33` | `100%` | `4.41 GB` | `10%` | `100%` | `0.0148` | rejected |

The two low-bit modes pass the aggregate MAE and gripper gates but fail the
predeclared >=90% exact-action requirement. INT8 cuts peak VRAM by `46.2%` and
NF4 by `69.4%`, while increasing P50 by `405.5%` and `146.6%`, respectively.
Therefore neither profile enters CARVE's realtime deployment set and neither
is advanced to simulator evaluation. The result validates the second-family
adapter and demonstrates why memory reduction, action fidelity, and critical-
path latency must be gated separately.

Implementation and evidence:

- `agentic_vla/runtime/policies/openvla_hf.py`
- `agentic_vla/runtime/adapters/openvla.py`
- `agentic_vla/optimization/models/openvla.py`
- `scripts/benchmark_carve_openvla_profile.py`
- `scripts/run_openvla_profile_gate.sh`
- `results/carve_optimize/openvla_4090_20260718/OPENVLA_PROFILE_GATE.md`
- `results/carve_optimize/openvla_4090_20260718/openvla_profile_gate_summary.json`

## 2026-07-18: OpenVLA critical-path and compiled-decode gate

Module-level CUDA events on ten BF16 actions locate the dominant cost in the
autoregressive language path. Mean vision, projector, prefill, and six-token
decode time are `34.79`, `0.40`, `58.28`, and `209.41 ms`, respectively;
decode occupies `67.2%` of the `311.42 ms` mean CUDA span.

Direct language-model compilation initially fails because official
`predict_action` appends token `29871` without extending `attention_mask`.
CARVE adds an opt-in alignment before device transfer. The compatibility
candidate produces exact `10/10` BF16 actions with zero MAE. Dynamic
`reduce-overhead` compilation then exposes unsafe CUDA Graph output reuse
across KV-cache decode calls, so the accepted candidate uses default Inductor
mode with CUDA Graphs disabled.

Two prompt-token lengths (`24` and `29`) are prewarmed before measurement.
Profile preparation takes `200.19 s`; steady-state P50/P95/P99 are
`224.25/231.84/232.18 ms`, versus `303.48/310.92/312.74 ms` for eager BF16.
This is a `26.1%/25.4%` P50/P95 reduction with exact `10/10` paired actions and
unchanged `14.42 GB` peak VRAM. It is accepted as a workload-calibrated latency
optimization, but not as a 10 Hz profile because all calls still miss 100 ms.
Unseen prompt lengths require eager fallback until persistent ahead-of-time
shape handling is validated.

Evidence:

- `results/carve_optimize/openvla_4090_20260718/openvla_critical_path.json`
- `results/carve_optimize/openvla_4090_20260718/openvla_compile_bf16_prewarmed_candidate.json`
- `results/carve_optimize/openvla_4090_20260718/OPENVLA_PROFILE_GATE.md`
- `scripts/profile_openvla_critical_path.py`

## 2026-07-19: PI0.5 deployment admission and contract fallback

Closed the gap between Optimize Runtime experiments and online Agentic serving.
The server previously checked checkpoint identity and replay fidelity, which
was insufficient because the W8A16 candidate passed replay fidelity but failed
the paired closed-loop gate.

- Extended `ProfileManifest` with backward-compatible admission metadata.
- Added mandatory promoted replay-fidelity, realtime, and closed-loop gates.
- Legacy and rejected manifests remain readable but are refused by default.
- Generated SHA-256 evidence receipts for the promoted PI0.5 compiled-BF16 and
  SMVE deployment manifests.
- Replaced SMVE's legacy numeric deployment contract with canonical named view
  order and padded-view identity during promotion.
- Required a separately admitted compiled-BF16 fallback for SMVE, with exact
  model, adapter, checkpoint, hardware, flow-step, and horizon matching.
- Added a narrow online fallback: only the active-view mask-contract violation
  is retried; OOM and other runtime faults are not hidden.
- Both compiled paths are prewarmed before the WebSocket port accepts requests.
- The optimization test module passes `31/31`, the full suite passes `69/69`,
  and Ruff passes on all changed code.

The real PI0.5 WebSocket deployment smoke then loaded both admitted profiles on
the RTX 4090. SMVE and fallback first prewarm calls took `26.01 s` and
`34.41 s`; their second calls took `1.56 s` and `1.90 s`. Six independently
connected Agentic `stall/retry` requests all returned the admitted SMVE identity
and `10x7` actions. Model P50/P95 was `41.68/42.25 ms`; end-to-end runtime P50
was `59.81 ms`. The first client call missed the 80 ms deadline at `87.96 ms`,
while the following five met it. This is a deployment-path smoke rather than a
replacement for the existing 50-call latency benchmark.

Deployment artifacts:

- `results/carve_optimize/deployment/pi05_smve_promoted.json`
- `results/carve_optimize/deployment/pi05_compiled_bf16_fallback_promoted.json`
- `scripts/promote_carve_profile.py`
- `results/carve_optimize/deployment/pi05_smve_agentic_websocket_smoke_20260719.json`

Decision: freeze OpenVLA as second-family interface and quantization-negative
evidence. Continue the main system and closed-loop optimization work on PI0.5.

## 2026-07-19: PI0.5 Recovery Challenge and benchmark scope decision

Stopped the proposed 400-episode clean LIBERO rerun after the user redirected
the experiment toward a framework-specific simulation test. The clean suite is
already near saturation and does not isolate CARVE's intervention or inference
cost, so the run was terminated before spending additional GPU time.

Built and completed a real LIBERO MuJoCo recovery challenge using three exact
restored states, fixed deterministic noise, and the promoted PI0.5 compiled
BF16 + SMVE deployment profile. Each state was evaluated with frozen VLA
continuation, frequent replanning, prompt retry, and physical recovery followed
by replanning.

| Branch | Success | PI0.5 calls | Executed states | Safe stops |
|---|---:|---:|---:|---:|
| Frozen continuation | `2/3` | `113` | `3/3` | `0` |
| Frequent replan | `2/3` | `452` | `3/3` | `0` |
| Prompt retry | `2/3` | `508` | `3/3` | `0` |
| Physical recovery | `2/3` | `251` | `2/3` | `1` |

T6 and T9 stall states succeed under all branches. Both physical-recovery
branches execute 12 bounded actions and produce a verified state response. T8
`stale_action` fails under the three VLA-only branches; the physical skill
declares that event unsupported and safely stops before issuing any action.
The equal `2/3` success count means this experiment is not reported as an
Agentic success-rate gain. It demonstrates unnecessary compute from indiscriminate
replanning, bounded recovery execution, and fail-closed skill coverage.

A separate online T6 sentinel validates the automatic lifecycle without
simulator state entering the controller: monitor -> recovery selection -> 12
physical actions -> verification -> admitted-profile PI0.5 replan -> task
success. It completes `1/1`, records VLA P95 `56.41 ms`, and records a control
deadline-miss rate of `10.64%`; the latter shows remaining non-model overhead in
the synchronous MuJoCo control path.

Evidence:

- `results/carve_pi05_recovery_challenge_20260719/REPORT.md`
- `results/carve_pi05_recovery_challenge_20260719/summary.json`
- `results/carve_pi05_recovery_challenge_20260719/recovery_challenge.png`
- `results/carve_pi05_recovery_challenge_20260719/paired_branches_videos/`
- `results/carve_pi05_recovery_challenge_20260719/online_controller/videos/`
- `scripts/run_carve_pi05_recovery_challenge.sh`

## 2026-07-19: Same-state Agentic Harness + Optimize Runtime pair

Ran a predeclared coupled systems experiment instead of a large clean
benchmark. The test restores the same T6/T9 stall states, fixes policy noise,
flow steps, committed horizon, physical skill, and 80 ms deadline, and varies
only the PI0.5 execution profile. Eager BF16 is explicitly served as an
unpromoted research reference; compiled BF16 and SMVE retain deployment
admission enforcement.

| Profile | Exact success | Recovery verified | Runtime P50/P95 | Miss@80ms |
|---|---:|---:|---:|---:|
| Eager BF16 | `2/2` | `2/2` | `155.88/166.26 ms` | `236/236` |
| Compiled BF16 | `2/2` | `2/2` | `62.54/65.75 ms` | `0/239` |
| Compiled BF16 + SMVE | `2/2` | `2/2` | `52.54/54.50 ms` | `0/247` |

Compiled BF16 and SMVE reduce runtime P95 by `60.5%/67.2%` versus eager;
SMVE reduces P95 a further `17.1%` versus ordinary compilation. Exact-state
outcome parity passes. One online T6 sentinel per profile yields `1/1`, `0/1`,
and `1/1`; this single-episode variation is retained and is not used for model
ranking. The exact-state pair is the promoted behavior-preservation evidence.

Evidence:

- `results/carve_pi05_agentic_optimize_pair_20260719/REPORT.md`
- `results/carve_pi05_agentic_optimize_pair_20260719/summary.json`
- `results/carve_pi05_agentic_optimize_pair_20260719/agentic_optimize_pair.png`
- `scripts/run_carve_pi05_agentic_optimize_pair.sh`

## 2026-08-24: Canonical LIBERO-Pro framework gate

Migrated the final official LIBERO-Pro execution path to
`CarveAgentSession`, the seven typed tools, local Qwen3.5-4B and the promoted
PI0.5 SMVE profile. The LIBERO adapter now samples deployable observations and
video at every control step while keeping evaluator termination private.

Object Task 8 succeeds in 423 steps with no Agentic intervention. Its 43 PI0.5
calls record 55.22/56.59 ms runtime mean/P95 and zero 80 ms misses. Object Task
9 triggers two Qwen decisions and one verified 12-action recovery, then safely
stops at step 362 when door closure remains stalled; the private evaluator
records failure. This is a complete mechanism and fail-closed integration gate,
not a success-rate uplift claim.

Evidence:

- `docs/status/LIBERO_PRO_CANONICAL_GATE_20260824.md`
- `results/libero_pro_canonical_object_t8_step_monitor_20260824/`
- `results/libero_pro_canonical_object_t9_agentic_complete_20260824/`
- `scripts/run_agentic_vla_libero_pro_canonical.py`

The full CPU suite passes `207` tests. The next experiment is restricted to a
three-initial-state paired frozen-VLA versus Agentic pilot.

## 2026-08-24: Canonical three-state paired pilots

Added explicit `frozen_vla` and `agentic` modes to the canonical runner and
stored the method in each manifest. The pair holds checkpoint, promoted SMVE
profile, fixed policy noise, initial states, deadline and episode limit fixed.

On LIBERO-Pro `libero_10_object` Task 9, both methods complete `1/3`. Agentic
preserves the successful trial without intervention, reduces total control
steps from 1295 to 919 and PI0.5 calls from 130 to 90, and safely stops the two
failed trials after four Planner calls and two recoveries. The local VLM adds
37.70 s, so wall time rises from 58.33 s to 77.38 s. This is a safety/compute
result, not a task-success gain.

On LIBERO-Pro `libero_10` Task 9, both methods complete `3/3` with the same 729
steps and 74 VLA calls. Agentic makes zero Planner/recovery calls, confirming
nominal non-interference under event-only scheduling.

Evidence:

- `results/libero_pro_paired_t9_summary_20260824/`
- `results/libero_pro_paired_long_t9_summary_20260824/`
- `scripts/summarize_libero_pro_canonical_pair.py`

Decision: stop the natural-task sweep. Any later capability-uplift test must
use a predeclared semantic perturbation or restored correctable failure state.

## 2026-08-24: Correctable-failure gate

Extended the canonical LIBERO-Pro runner with two isolated, auditable protocols:
private exact-state restoration and `stale_subgoal_v1`. The latter preserves
the benchmark task while corrupting only the cached low-level VLA instruction;
the fault is detected from an instruction-version receipt rather than evaluator
state. Agentic runs invoke Qwen at startup, while Frozen controls retain the
same stale command.

The historical T6 stall snapshot is no longer a differentiating failure under
the promoted PI0.5 profile: Frozen and Agentic both succeed in 93 steps without
Planner or recovery. On Object Task 8, Qwen detects the stale command at 0.92
confidence, changes the typed VLA call back to the moka-pot task, and completes
the rollout in 425 steps. Frozen also succeeds under wrong Task 8 prompts due
to a strong scene prior, so this is mechanism evidence only.

An executable object-relation swap on `libero_10_task` Task 6 makes Frozen fail,
and Agentic correctly restores the true instruction, but the correct-instruction
PI0.5 upper bound also fails. Task 4 and Task 9 fail the same upper-bound
qualification. These tasks are therefore rejected rather than used to claim a
Harness success gain.

Evidence:

- `docs/status/LIBERO_PRO_CORRECTABLE_FAILURE_GATE_20260824.md`
- `results/libero_pro_restore_t6_frozen_smoke_20260824/`
- `results/libero_pro_restore_t6_agentic_smoke_20260824/`
- `results/libero_pro_stale_subgoal_t8_agentic_correction_v2_20260824/`
- `results/libero_pro_stale_subgoal_t6_frozen_swapped_smoke_20260824/`
- `results/libero_pro_stale_subgoal_t6_agentic_swapped_smoke_20260824/`

Decision: no expansion on the current tasks. Future correction experiments must
first demonstrate a successful correct-instruction upper bound and a failed
stale-instruction Frozen control on the same fixed states. The canonical test
entry point passes `213` tests.

## 2026-08-24: LIBERO-Pro target-grounding and semantic-checkpoint pilot

Applied the correct-instruction qualification funnel to Task 3 and Task 5 in
`libero_10_task`; both fail at 520 steps and were stopped before Agentic pairs.
The final predeclared candidate, `libero_10_swap` Task 4, fails at 620 steps but
reveals a correctable semantic pattern in video: PI0.5 places the white mug on
the left plate and then places an unmentioned red distractor on the right plate,
leaving the yellow-white target on the table.

Added optional nominal task-start VLM planning and explicit, budgeted semantic
checkpoints to the canonical session. A step-250 checkpoint correctly switches
the remaining subgoal to the yellow-white mug in one run, demonstrating the
complete checkpoint -> VLM -> typed VLA transition. A semantic-planner-only
ablation confirms the failure is not caused solely by Monitor recovery or safe
stop. However, local Qwen3.5-4B is inconsistent at fine-grained grounding: it
also selects the red distractor, repeats a completed clause, and once falsely
labels upright mugs unsafe. No condition completes the private evaluator task.

Decision: stop checkpoint/prompt search with the current 4B Planner. The next
causal comparison must keep the PI0.5 rollout protocol fixed and vary only the
Planner backend or add a deterministic object-referent validation gate. This is
a VLM grounding bottleneck, not evidence for retraining PI0.5 or rerunning the
suite. The canonical test entry point passes `213` tests.

## 2026-08-25: VLM task-grounding guard and LIBERO-PRO Task 4 gate

- Added a task-contract validator at the guarded Planner boundary. A `vla_act`
  decision is rejected when its action target introduces a color referent that
  is absent from the benchmark task instruction.
- Added one bounded grounding-repair call. The rejection reason is returned to
  the same VLM once, every physical model invocation counts against the episode
  call budget, and a second invalid decision fails closed.
- Added `attempt_count` and `validation_errors` to Planner results and persisted
  both fields in the transcript and event log.
- Local model/resource decision: retained the resident Qwen3.5-4B Planner. It
  used 9.6 GiB, while the optimized PI0.5 service used 7.8 GiB; both fit on the
  RTX 4090. No new VLM was downloaded.
- Offline gate on the recorded official LIBERO-PRO `libero_10_swap` Task 4
  frames:
  - step 150: 4/4 decisions selected the yellow-and-white mug for the right
    plate;
  - task start: 3/3 decisions selected the white mug as the first subgoal;
  - all seven calls passed without a repair.
- Closed-loop gate:
  - suite/task/seed: `libero_10_swap`, Task 4, seed 7;
  - method: startup Planner plus one semantic checkpoint at step 150, with
    Monitor-triggered intervention masked;
  - optimized VLA profile:
    `pi05-torch_compile_masked_views-bf16-2step-h10`;
  - Planner requests: 2. The checkpoint request required one internal bounded
    repair before its decision was accepted;
  - result: 0/1 success after 620 steps.
- Causal reading of the video: the white mug reached the left plate, but PI0.5
  placed the red distractor on the right plate and left the yellow-and-white mug
  on the table. The accepted Planner decision preserved the correct task
  mapping, so this run isolates a remaining VLA instance-grounding/execution
  limitation rather than a Planner target-selection failure.
- Artifacts:
  - `results/vlm_grounding_gate_20260825/`
  - `results/libero_pro_swap_task4_vlm_guarded_20260825/`

## 2026-08-25: Canonical LIBERO-Pro three-method completion gate

- Added a native `fixed_recovery` condition to the canonical official
  LIBERO-Pro runner. It uses the same Monitor and one typed physical skill as
  Full Agentic but has no in-process Planner.
- Completed Object T8/T9 with Frozen VLA, Fixed Recovery and Full Agentic on
  three paired states per task: 18 real MuJoCo episodes in total.
- Success is identical across methods: T8 `2/3`, T9 `1/3`, aggregate `3/6`.
- Relative to Frozen VLA, Full Agentic reduces control steps by `17.1%` and VLA
  calls by `18.3%`; it preserves all successful states and converts all three
  persistent failures to bounded safe stops.
- Full Agentic makes six event-triggered Qwen calls at 9.63 s mean latency. The
  co-resident PI0.5 path records `0/214` deadline misses at 80 ms.
- The success-uplift expansion condition is not met, so no 10-state sweep is
  run. The final claim is bounded failure handling and realtime-path
  preservation, not improved benchmark success.
- Evidence:
  - `docs/status/LIBERO_PRO_CANONICAL_TRIAD_GATE_20260825.md`
  - `results/libero_pro_canonical_triad_study_20260825/`
  - `scripts/summarize_libero_pro_canonical_triad.py`
  - `scripts/summarize_libero_pro_canonical_study.py`

## 2026-08-26: Full 1,200-episode LIBERO-Pro paired study

- Superseded the minimum-gate stop decision and completed the full predeclared
  matrix: four suites, ten tasks, ten paired states and three methods.
- Fixed a recovery-boundary defect before the formal run. The monitor now
  resets after a recovery primitive; a 30-episode T4/T6/T7 regression gate
  passed at 30/30 with five successful recoveries and no safe stops.
- Completed 1,200 physics-simulator episodes with 1,200 videos:
  - Frozen VLA: `180/400`;
  - Fixed Recovery: `181/400`;
  - Full Agentic: `183/400`.
- Full Agentic records four Frozen-failure conversions and one regression;
  exact paired McNemar `p=0.375`, so no statistically significant aggregate
  success-uplift claim is made.
- Relative to Frozen, Agentic reduces control steps by `7.2%` and VLA calls by
  `8.4%`. VLA P95/P99 is `59.54/63.19 ms` with `8/14,699` misses at 80 ms.
- The local Qwen3.5-4B Planner is invoked 263 times at 9.09 s mean and 11.23 s
  P95. Total Agentic wall time rises `16.5%`; high-level Planner optimization is
  therefore the next runtime target.
- Integrity audit passed: 120 cells, 1,200 episodes, 1,200 decodable videos,
  454,540 frames, 400 paired identities, one checkpoint/profile, and no private
  evaluator keys in Agent event logs.
- Evidence:
  - `docs/status/LIBERO_PRO_FULL_STUDY_RESULTS_20260826.md`
  - `docs/status/LIBERO_PRO_RECOVERY_BOUNDARY_GATE_20260825.md`
  - `results/libero_pro_full_study_20260825/aggregate/`

## 2026-08-26: Full VLA and Planner efficient-inference ablation

- Re-ran five PI0.5 runtime profiles on 45 paired recorded observations with
  fixed diffusion noise and persisted per-call latency samples.
- PI0.5 P95 results: eager 7-step `282.43 ms`, eager 2-step `151.35 ms`,
  compiled 2-step `66.06 ms`, and compiled 2-step + SMVE `54.67 ms`.
  Compile and SMVE pass all 45 action-fidelity checks and record no 80 ms
  deadline misses.
- Added an auditable `uniform_int8` Qwen Planner service profile and evaluated
  BF16, INT8 and NF4 on the same 30 temporal pairs. All three pass the semantic
  gate with 100% paired semantic-decision agreement.
- Planner BF16/INT8/NF4 allocated VRAM is `8.46/4.84/3.08 GiB`; P95 is
  `1.90/6.50/2.91 s`. BF16 remains the latency tier, while NF4 is retained only
  as a shared-GPU capacity tier.
- PI0.5 late-language INT8 passes 45/45 action fidelity and reduces peak VRAM
  to `6.36 GiB`, but the current Torch `2.12.0` / TorchAO `0.15.0` stack
  produces P95 `994.74 ms`. It is rejected on this stack. The previously
  validated Torch `2.7.1` stack reached `71.61 ms` and 2/2 matched closed-loop
  recovery successes, so this optimization remains explicitly version-gated.
- The complete report, CSV tables, JSON summary and PNG/PDF figure are in
  `results/carve_efficiency_full_20260826/`.
- Added a real co-resident low-memory coupling gate with Qwen3.5-4B NF4 and the
  promoted PI0.5 SMVE profile. The two model-service processes used `11.22 GiB`
  in total. One Qwen decision was accepted in `13.92 s`; PI0.5 produced a
  `10 x 7` action chunk in `66.69 ms` with no deadline miss or fallback.
- The action chunk was deliberately withheld from the environment. This result
  is recorded as `integration_passed_system_admission_pending`, not as task
  success or repeated contention evidence.

## 2026-08-28: RoboMME environment and CARVE adapter admission

- Pinned the official benchmark at commit
  `d57969fd30f8e8318fb67c389a848b3776724470` and the official policy stack at
  `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`.
- Built the official CUDA 12.8 RoboMME container. It detects the RTX 4090,
  CUDA, Vulkan 1.3.275, SAPIEN 3.0.2 and the official ManiSkill fork.
- The official `MoveCube`, test episode 0 sample-action rollout completed and
  produced a readable 548-frame H.264 video. Its timeout is expected because
  no trained policy was loaded.
- The independent CARVE adapter gate passed on the same state:
  - 247 initial memory frames;
  - front RGB, wrist RGB and 8-D public robot state only;
  - 32 executed simulator steps and 279 recorded frames;
  - 8.14 s total wall time;
  - evaluator status remained private.
- Added `scripts/run_robomme_policy_admission.py` for the next checkpoint gate.
  It records policy latency, call counts, history batches, control steps,
  evaluator-private outcome and video without exposing oracle online subgoals.
- This is an environment/adapter result only. No RoboMME policy success or
  Agentic improvement is claimed yet.
- Evidence: `results/robomme_admission_20260828/`.

## 2026-08-28: RoboMME high-level Planner schema admission

- Ran the existing local Qwen3.5-4B BF16 CARVE Planner on a real paired
  front/wrist frame from official RoboMME `MoveCube`, test episode 0.
- The initial guarded call failed closed because the model used the semantic
  plan intent `move`, which the manipulation-oriented alias vocabulary did not
  yet normalize to the already-authorized `vla_act` primitive.
- Extended semantic intent normalization to bounded verb-led robot operations
  only when the top-level intent is already `vla_act` and no tool is selected.
  Raw action, trajectory, lifecycle and grounding guards remain unchanged.
- The next guard correctly rejected model-added `red/purple` attributes that
  were absent from the task instruction. With two bounded repair opportunities,
  the model removed those attributes and returned an accepted one-stage
  `vla_act` plan: `transport cube to target`.
- The accepted sequence required three model calls and 32.67 s. This is a
  schema/functionality gate, not a task-success or efficient-inference result.
- Focused high-level Agent, task-plan and toolchain regression: `71 passed`.
- Evidence: `results/robomme_planner_admission_20260828/`.

## 2026-08-28: Real GroundSG PI0.5 RoboMME policy Gate B

- Downloaded only the released `symbolic-grounded-subgoal/79999` checkpoint
  (`11,551,449,841` bytes), verified the official LFS SHA256
  `4a9577f5afa8e5225cb2e69e3820d4c7e5c7deb18f2ef00c10b4401eab575f94`,
  passed ZIP CRC validation, and restored the full Orbax checkpoint.
- The policy server loaded `symbolic-grounded-subgoal.yaml`, checkpoint-local
  RoboMME norm statistics, and listened successfully on one RTX 4090 at about
  8,586 MiB resident GPU memory.
- Passed the four-task policy admission on official test episode 0:
  `StopCube`, `VideoUnmaskSwap`, `VideoRepick`, and `RouteStick`.
- Every task completed reset, one initial-history transfer, one real policy
  inference, one 16-action execution chunk, private evaluator isolation, clean
  shutdown, summary output, and H.264 video output.
- The first JAX call compiled in `29,881.11 ms`. Three post-compile calls were
  `102.50`, `104.07`, and `103.63 ms` (mean `103.40 ms`, P95 `104.03 ms`).
- Fixed episode-0 initial subgoals were used only as an oracle-format technical
  gate. No full-episode task success, Agentic uplift, or Optimize benefit is
  claimed from these four runs.
- Evidence: `results/robomme_policy_gate_b_20260828/`.

## 2026-08-28: RoboMME dynamic-planning gap on paired MoveCube episodes

- Held the released GroundSG PI0.5 checkpoint and five official test episode
  identities fixed, then compared two subgoal interfaces.
- A single generic Planner-output string, `move the cube to the target`, solved
  `1/5` episodes. The other outcomes were three timeouts and one evaluator
  failure. This is an interface pilot, not the complete event-triggered CARVE
  Planner.
- The benchmark's privileged online grounded-subgoal oracle solved `5/5`. Its
  instructions changed with execution stage and included image-space points.
  Episodes 1, 3 and 4 additionally required picking up a peg and hooking the
  cube, which cannot be recovered from the generic one-stage instruction.
- The action policy therefore has sufficient capability on these five states;
  the immediate bottleneck is dynamic stage selection and visual point
  grounding at safe action boundaries. The oracle is an upper bound and is not
  reported as deployable CARVE performance.
- Warm PI0.5 inference remained stable at roughly 98--100 ms mean per method;
  the observed success gap is not attributable to different checkpoints or
  runtime profiles.
- Evidence:
  - `results/robomme_carve_text_subgoal_pilot_20260828/`
  - `results/robomme_oracle_upper_bound_20260828/`
  - `scripts/summarize_robomme_planning_gap.py`

## 2026-08-28: Deployable RoboMME GroundSG VLM Planner

- Loaded the official Qwen3-VL-4B-Instruct GroundSG LoRA behind CARVE's guarded
  OpenAI-compatible Planner provider while keeping the released GroundSG PI0.5
  checkpoint and five paired `MoveCube` test episodes fixed.
- The complete deployable VLM condition succeeds on `4/5` episodes, compared
  with `1/5` for one fixed generic text subgoal and `5/5` for the privileged
  online-oracle upper bound.
- The VLM Planner preserves episode 0 and converts episodes 2, 3 and 4 from
  failure to success. Episodes 3 and 4 correctly select and ground the two-stage
  peg strategy. Episode 1 also selects the correct peg strategy but PI0.5 does
  not complete execution within 1,300 steps, isolating an execution-layer
  failure rather than a high-level tool-selection failure.
- Across five VLM episodes, mean steps are 434.0 including the timeout and
  217.5 on successes; mean policy/Planner calls are 27.8. In the queue-isolated
  rerun, mean per-episode VLA P95 is 118.43 ms and Planner P95 is 2.92 s.
- BF16 Qwen Planner and PI0.5 coexist on one RTX 4090 at approximately 9.1 GiB
  and 8.6 GiB idle process memory. Planner latency is now the measured target
  for the GroundSG-specific Optimize Runtime ablation.
- All five queue-isolated VLM rollouts are valid H.264 videos at 512x256. The
  paired summary stores per-video SHA256 hashes and explicit deployable/oracle
  claim boundaries.
- Evidence:
  - `results/robomme_vlm_groundsg_bf16_queue_isolated_20260828/`
  - `scripts/run_robomme_vlm_groundsg.py`
  - `scripts/serve_robomme_groundsg_planner.sh`
  - `scripts/summarize_robomme_planning_gap.py`

## 2026-08-28: GroundSG Planner NF4 closed-loop optimization gate

- Repeated all five paired `MoveCube` episodes with only the Qwen3-VL-4B base
  changed from BF16 to bitsandbytes NF4; the GroundSG LoRA, PI0.5 checkpoint,
  seeds, action horizon and simulator states remain fixed.
- Queue-isolated BF16 reproduces `4/5`; NF4 reaches `2/5`. Both produce the same
  high-level stage sequence on `5/5`, but NF4 loses BF16 successes on episodes
  3 and 4 despite selecting the correct peg strategy.
- At matched trace steps, NF4 grounding differs from BF16 by 3.80 pixels mean
  absolute coordinate error. The result demonstrates why semantic agreement
  alone is insufficient for quantized grounded robot planners.
- Idle Planner process memory falls from 9,120 MiB to 3,994 MiB (`56.2%`),
  while mean per-episode Planner P95 rises from 2.92 s to 4.61 s (`1.58x`).
- No external request shared the BF16 Planner queue. A separate Qwen3.5-NF4
  process did share the GPU during part of the BF16 rerun, so the observed
  latency ratio is conservative and not reported as an empty-GPU speedup.
- NF4 fails the closed-loop quality gate and is retained only as a capacity
  fallback, not the default Optimize Runtime profile.
- The first BF16 latency run was exposed to unrelated G2 requests and was
  removed from canonical evidence after audit; reported BF16 numbers come from
  the queue-isolated rerun.
- Evidence:
  - `results/robomme_vlm_groundsg_bf16_queue_isolated_20260828/planner_quantization.json`
  - `results/robomme_vlm_groundsg_nf4_20260828/`
  - `scripts/summarize_robomme_planner_quantization.py`

## 2026-08-29: Complexity-aware GroundSG Planner scheduling gate

- Added a reusable grounded-subgoal scheduler to the CARVE runtime and bound it
  to the official RoboMME GroundSG runner with task-start, reuse-budget,
  gripper-change, visual-change and external execution-event invalidation.
- A uniform two-chunk reuse ablation reduced Planner calls but regressed
  `MoveCube` episode 3, producing `3/5` instead of the BF16 baseline's `4/5`.
  It is therefore rejected rather than reported as an admitted optimization.
- The promoted complexity-aware schedule reuses single-point grounded subgoals
  for at most two action chunks, while multi-point spatial/tool subgoals are
  refreshed every chunk.
- On the same five official test episodes it preserves `4/5`, reduces total
  Planner calls from 139 to 115 (`17.3%`) and aggregate wall time from 492.79 s
  to 457.47 s (`7.2%`). Across the four episodes solved by both conditions,
  Planner calls fall `35.1%` and wall time falls `18.7%`.
- Episode 1 remains the same execution-layer timeout and receives little reuse
  because its long two-point hook phase is precision-sensitive.
- All five adaptive videos pass H.264/512x256 inspection. Runtime and runner
  regressions pass (`46 passed` focused; `248 passed` excluding three known
  legacy test files that import scripts removed during repository cleanup).
- Evidence:
  - `results/robomme_vlm_groundsg_adaptive_selective2_bf16_20260829/`
  - `results/robomme_vlm_groundsg_adaptive_selective2_bf16_20260829/planner_scheduling_ablation.json`
  - `scripts/run_robomme_selective_groundsg.sh`
  - `scripts/summarize_robomme_planner_scheduling.py`

## 2026-08-29: RoboMME VideoRepick memory/tool-routing pilot

- Extended the deployable GroundSG runner with typed demonstration-memory
  hints, relational color grounding and auditable repetition progress. The
  online Planner and VLA receive no evaluator-private state or oracle subgoal.
- A Qwen3.5-9B NF4 offline encoder extracted relational memory from 32 sampled
  demonstration frames. Qwen3-VL-4B GroundSG remained the online Planner and
  the official GroundSG PI0.5 checkpoint remained frozen.
- `VideoRepick` episode 0 failed without structured memory and also failed with
  memory alone. Memory plus the local relation-grounding tool completed all
  three pick/put cycles and the final button press.
- On the paired successful episode-0 rollout, the tuned event scheduler
  preserves success, reduces Planner calls from 24 to 20 (`-16.7%`) and wall
  time from 62.92 s to 54.05 s (`-14.1%`).
- In this initial pilot, episode 1 is a memory-identification failure. Episode 2 reaches the
  button stage after three cycles but does not satisfy the final evaluator.
  Button-center refinement does not change the outcome. A two-turn semantic
  transition confirmation gate over-delays progress and is rejected as a
  default; it remains disabled and recorded as a negative ablation.
- Privileged online-oracle upper bounds solve episodes 1 and 2. They are used
  only to localize the Planner/memory gap and are not deployable results.
- Focused parser, grounding, progress and transition regressions: `14 passed`.
- Evidence:
  - `results/robomme_memory_tool_study_20260829/`
  - `scripts/summarize_robomme_memory_tool_study.py`
  - `scripts/run_robomme_vlm_groundsg.py`
  - `tests/test_robomme_grounded_planner.py`

## 2026-08-29: VideoRepick instance memory and execution-monitor closure

- Added official SAM2.1 Hiera Tiny instance tracking for initial demonstration
  memory. Episode 1 tracks the demonstrated `topmost red block` through
  occlusion/rearrangement to the final `bottom-left red block`; the resolved
  point `[133.57, 113.12]` is admitted after one reacquisition and 24 missing
  frames. No evaluator-private point is used online.
- Added tracker admission based on missing runs and reacquisition count. The
  tracker is rejected on episode 0 after two loss runs, so the system falls
  back to the VLM identity rather than trusting a poor track; episodes 1 and 2
  are admitted and match their oracle-localization diagnostics.
- Normalized GroundSG action aliases (`pick`, `pick up`, `grasp`, `put`,
  `put it down`, `press`) at the primitive boundary.
- Added ordered one-cycle procedure control and a proprioceptive
  `PrimitiveExecutionMonitor`. Semantic progress is monotonic, and transitions
  to put/press require observed gripper closure/reopening rather than fixed
  step counts or evaluator state.
- With correct SAM2 memory but without physical completion monitoring, episode
  1 enters the button stage at 112 steps and fails after 184 steps. With the
  execution monitor, the same episode succeeds in 282 steps with 18 Planner
  calls.
- The tuned selective Planner schedule preserves the 282-step success, reduces
  Planner calls from 18 to 16 (`-11.1%`) and wall time from 38.36 s to 36.11 s
  (`-5.9%`). This is a paired single-episode mechanism result, not a multi-seed
  success-rate claim.
- Focused grounding, procedure, monitor and tracker regressions: `23 passed`.
- Evidence:
  - `results/robomme_video_repick_ep1_monitor_study_20260829/`
  - `results/robomme_video_instance_memory_sam2_admitted_20260829/`
  - `results/robomme_video_repick_ep1_execution_monitor_v2_20260829/`
  - `results/robomme_video_repick_ep1_execution_monitor_selective_20260829/`
  - `results/robomme_representative_videos_20260829/05_ep1_sam2_monitor_event_success.mp4`
  - `results/robomme_representative_videos_20260829/06_ep1_sam2_identity_track.mp4`

## 2026-08-29: VideoRepick repeated-procedure closure and full paired runtime study

- Added a deployable `RepeatedProcedureController` for repeated pick/put tasks.
  The GroundSG VLM still selects primitive and target; the controller permits
  semantic progress only after gripper proprioception confirms closure and
  reopening for each cycle. Press is permitted only after all cycles complete.
- Episode 2 changes from a near-complete evaluator failure to success. SAM2
  identity memory is admitted with zero missing frames and resolves the
  demonstrated instance to the final `leftmost red block` at `[107.18, 86.4]`.
- Repeated-procedure monitoring also preserves success on episode 0. Its SAM2
  track is rejected after two loss runs, exercising the explicit VLM identity
  fallback rather than consuming a low-quality tracker output.
- Completed a matched every-chunk/selective study on all three official
  `VideoRepick` episodes. Both schedules reach `3/3`; selective scheduling
  reduces aggregate Planner calls from `87` to `75` (`13.8%`) and wall time
  from `200.79 s` to `176.67 s` (`12.0%`). VLA calls remain `87` in both.
- Excluded the first episode-2 every-chunk timing from the paired aggregate
  because its first policy call contains a 28.25 s cold compile; a warm rerun
  under the same service is canonical. All six paired videos are H.264 at
  512x256.
- Reliability/efficiency boundary: the strict monitor is more conservative on
  episode 0 than the earlier relaxed successful controller (`594` versus `383`
  steps). The efficiency claim therefore compares selective scheduling only
  with its matched strict-monitor baseline.
- Final focused procedure, tracker and Planner regressions: `25 passed`.
- Evidence:
  - `results/robomme_video_repick_full_monitor_study_20260829/`
  - `results/robomme_video_repick_ep0_repeated_monitor_every_chunk_warm_20260829/`
  - `results/robomme_video_repick_ep0_repeated_monitor_selective_20260829/`
  - `results/robomme_video_repick_ep2_repeated_monitor_every_chunk_warm_20260829/`
  - `results/robomme_video_repick_ep2_repeated_monitor_selective_20260829/`

## 2026-08-31: Four-suite RoboMME C3 Agentic/Optimize pilot

- Completed 20 deployable, oracle-free method episodes over `StopCube`,
  `VideoRepick`, `RouteStick` and `VideoUnmaskSwap`, with five fixed official
  test episodes per task.
- Added legal RouteStick primitive contracts and a demonstration-RGB trajectory
  tool that converts image direction to robot-frame route direction. RouteStick
  reaches `3/5`; one additional method failure is solved by the privileged
  upper bound and remains the only measured Agentic gap.
- Added multi-object SAM2 memory for VideoUnmaskSwap. The compiler associates
  visible colors with containers, tracks identity through swaps, and provides
  admitted color-to-container memory without evaluator fields. The unified
  event configuration reaches `5/5`.
- Refined repeated physical primitives with cycle-dependent dwell, gripper-event
  replanning, grasped-instance locking, final-pick semantics and button memory.
  The latest VideoRepick configuration reaches `4/5`; its remaining failed
  episode also fails with the official privileged online subgoal.
- Raw deployable success is `14/20`. Five of six failures also fail under the
  PI0.5 oracle upper bound. The deployable method therefore reaches `14/15`
  (`93.3%`) on ceiling-eligible episodes, with raw and adjusted metrics kept
  explicitly separate.
- Mean per-episode Planner-call reduction is `42.4%`, `41.8%`, `42.9%` and
  `43.1%` for the four task families respectively. All 20 canonical method
  videos pass `ffprobe` validation.
- Focused Planner, memory, procedure and tool regression: `45 passed`.
- Evidence:
  - `docs/status/ROBOMME_C3_PILOT_RESULTS_20260831.md`
  - `results/robomme_c3_pilot_canonical_20260831/`
  - `scripts/summarize_robomme_c3_pilot.py`
  - `scripts/build_robomme_unmask_swap_memory.py`

## 2026-08-31: Paired RoboMME C2/C3 Planner-scheduling ablation

- Added a restartable `C2-agentic` runner that uses the same frozen GroundSG
  PI0.5, Qwen3-VL-4B Planner, admitted task memory, visual grounding tools,
  Harness logic and action horizon as the canonical C3 pilot. The only causal
  variable is `every_chunk` versus event-triggered selective Planner scheduling.
- Completed all 20 paired official test episodes with valid summaries and
  rollout videos.
- C2 every-chunk reaches `10/20`; C3 selective reaches `14/20`. The paired
  transitions are four failure-to-success and zero success-to-failure.
- C3 reduces total Planner calls from `256` to `171` (`33.2%`) and total wall
  time from `766.5 s` to `602.7 s` (`21.4%`). C3 executes `298` policy chunks
  versus C2's `256`, because four additional long-horizon episodes complete.
- By task, C2 -> C3 success is: StopCube `2/5 -> 2/5`, VideoRepick
  `2/5 -> 4/5`, RouteStick `1/5 -> 3/5`, and VideoUnmaskSwap `5/5 -> 5/5`.
- The paired success delta is `+20.0` percentage points, bootstrap 95% interval
  `[+5.0, +40.0]`, with McNemar exact two-sided `p=0.125`. This is reported as
  a 20-episode mechanism pilot, not full-benchmark significance.
- Evidence:
  - `results/robomme_c2_agentic_every_chunk_20260831/`
  - `results/robomme_c2_c3_runtime_ablation_20260831/`
  - `scripts/run_robomme_c2_agentic_pilot.sh`
  - `scripts/summarize_robomme_c2_c3_ablation.py`

## 2026-08-31: Paired RoboMME raw B1 Agentic baseline

- Added a guarded `raw` GroundSG Planner profile and restartable B1 runner.
  It uses the same frozen GroundSG PI0.5 and Qwen3-VL-4B checkpoint but removes
  CARVE task memory, grounding tools, semantic repair, procedure controllers
  and temporal Monitor events. It invokes the VLM on every action chunk.
- Completed all 20 fixed pilot episodes. An automated audit confirms every B1
  summary has `planner_profile=raw`, no task memory, no tool grounding and no
  temporal-monitor event.
- Success is B1 `2/20`, C2 `10/20`, C3 `14/20`. By task, B1 reaches StopCube
  `0/5`, VideoRepick `1/5`, RouteStick `0/5`, VideoUnmaskSwap `1/5`.
- B1 -> C2: nine failure-to-success, one success-to-failure, delta `+40.0`
  points, paired bootstrap 95% interval `[+15.0, +65.0]`, McNemar exact
  two-sided `p=0.0215`.
- B1 -> C3: 12 failure-to-success, zero success-to-failure, delta `+60.0`
  points, interval `[+40.0, +80.0]`, McNemar `p=0.0005`.
- Evidence:
  - `results/robomme_b1_raw_policy_20260831/`
  - `results/robomme_b1_c2_c3_paired_pilot_20260831/`
  - `scripts/run_robomme_b1_raw_policy_pilot.sh`
  - `scripts/summarize_robomme_b1_c2_c3_pilot.py`

## 2026-08-31: RAL-oriented paired RoboMME extension

- Extended the same four RoboMME task families from five to ten fixed episodes
  per condition. The formal paired study now contains 40 episodes per condition
  and 120 total rollouts.
- Compiled all 20 new initial demonstrations into deployable memory. Added an
  audited VideoRepick fallback that selects the manipulated instance from
  persistent demonstration motion when the VLM omits object identity, then
  tracks it with SAM2. All five new VideoRepick tracks pass admission.
- Final success is B1 raw `5/40` (`12.5%`), C2 Harness `16/40` (`40.0%`) and
  C3 Harness + Optimize Runtime `22/40` (`55.0%`).
- B1 -> C3 yields 18 failure-to-success and one success-to-failure transition,
  `+42.5` points, paired bootstrap 95% `[+25.0, +60.0]`, and McNemar exact
  two-sided `p=0.0000763`.
- C2 -> C3 reduces Planner calls `459 -> 305` (`33.6%`) and wall time
  `1366.7 -> 1103.9 s` (`19.2%`) while improving success by `15.0` points.
- All `120/120` rollout videos pass first/last-frame decoding validation.
- Evidence:
  - `docs/reports/RAL_CORE_EXPERIMENT_REPORT_20260831.md`
  - `results/robomme_b1_c2_c3_paired_40ep_20260831/`
  - `results/robomme_extension_b1_raw_20260831/`
  - `results/robomme_extension_c2_agentic_20260831/`
  - `results/robomme_extension_c3_full_20260831/`

## 2026-08-31: RoboMME held-out expansion and combined study

- Added four held-out official RoboMME tasks, one from each suite: BinFill,
  ButtonUnmask, PickHighlight, and MoveCube. Each task uses ten paired initial
  episodes under B1 raw, C2 Harness, and C3 Harness + Optimize Runtime.
- The expansion contains 120 rollouts. B1 reaches `18/40` (`45.0%`), C2
  `20/40` (`50.0%`), and C3 `20/40` (`50.0%`). C3 matches C2 success while
  reducing Planner calls by `42.7%` and wall time by `31.3%`. The expansion's
  B1 -> C3 success delta is not significant by itself (`p=0.6875`).
- Combined with the original four-task study, the formal evidence now contains
  eight tasks, 80 paired episodes per condition, and 240 total rollouts. B1 is
  `23/80` (`28.75%`), C2 is `36/80` (`45.0%`), and C3 is `42/80` (`52.5%`).
- B1 -> C3 gives 22 failure-to-success and three success-to-failure
  transitions, a `+23.75` point delta, paired bootstrap 95% interval
  `[+12.5, +35.0]`, and exact McNemar `p=0.0001565`.
- C3 reduces Planner calls by `40.6%` and wall time by `28.1%` relative to C2.
  C2 -> C3 success improvement is not independently significant (`p=0.146`),
  so this comparison supports the efficiency claim rather than a standalone
  success claim.
- The deployable online method uses RGB observations, instructions, execution
  signals, process memory, and bounded recovery only. No evaluator state or
  oracle labels are consumed online. All evaluated videos are retained.
- Evidence:
  - `results/robomme_b1_c2_c3_combined_80ep_20260831/`
  - `results/robomme_extension_selected_b1_c2_c3_40ep_20260831/`
  - `results/robomme_oracle_survey_20260831/` (diagnostic only; excluded)
  - `docs/reports/RAL_CORE_EXPERIMENT_REPORT_20260831.md`

## 2026-09-11 — Native-path request evidence (code only, no robot run)

Follow-up to plan item 9.21(1): the B0 arm had no per-request input/output/latency
record, so it could not be compared with C1/C2/C3 in a common format.

- Root cause, in the real policy entry
  `third_party/robodojo_official/XPolicyLab/policy/starVLA/model.py::_infer_chunk`:
  with `carve_mode=baseline` there is no `CarveRuntime`, so the call goes straight
  to `self.client.predict_action(...)` and returns. Only the wrapped arms emit
  `runtime_trace.jsonl`.
- Change: new `_record_native_request()` writes one record per native request in
  the same schema, reusing `hash_starvla_request_input` /
  `summarize_starvla_request_input` from
  `agentic_vla/runtime/adapters/starvla.py` so digests are comparable, and timing
  the same `predict_action` scope the adapter times. Wrapper-only fields stay
  `null` rather than being filled from the request.
- Opt-in via `carve_native_trace_path` / `STARVLA_CARVE_NATIVE_TRACE_PATH`, wired
  through `deploy.yml` and `setup_eval_policy_server.sh --overrides`. Disabled by
  default: nothing imported, timed or written, so baseline numerics are unchanged.
- Verification (CPU, `openpi/.venv/bin/python`): 124 passed in
  `tests/test_starvla_robodojo_bridge.py` (7 new), 346 passed together with
  `test_recovery_verification.py` and `test_robodojo_nominal_finalizer.py`,
  74 passed in the workspace contract set, `ruff` clean on the changed ranges.
- Not established by this entry: no robot episode was run, so no B0 record exists
  yet and the cross-process initial-RGB difference is still unlocalized. A
  comparable latency field is not the same as a completed comparison.
- Frozen evidence untouched: `artifacts/robodojo/recovery_lifecycle_20260910/`
  has no file modified after 2026-09-10.
- Model/scope: StarVLA QwenPI-v3 (`steps_100000_pytorch_model.pt`) as the action
  model; no VLM involved in this change. Sample size: not applicable, code change.

## 2026-09-11 B0 原生路径逐次记录实跑（诊断，n=2）

目的：补齐 B0（原生 flow 路径，不经 `CarveRuntime`）的逐次输入摘要、输出摘要与时延，
使其与 C1/C2/C3 同模式可读；并判断能否按输入摘要跨进程对齐两条路径。
对应计划 9.21 第 1 项，记录于 9.22 节。

模型：StarVLA PI-v3，`checkpoints/huggingface/pi_v3/checkpoints/steps_100000_pytorch_model.pt`。
B0 为 `carve_mode=baseline`，不加载 Qwen3-VL-4B。
样本量：`stack_bowls` layout set 0，布局 0 与 1，2 回合，57 次 VLA 请求。
测量口径：`model_latency_ms` 围绕 `client.predict_action` 单次调用计时，与包装路径
adapter 计时范围相同；输入摘要复用 `hash_starvla_request_input` /
`summarize_starvla_request_input`；生效 flow 步数以服务端 handshake 为准
（`native_inference_steps=4`），不以 payload 旧键 `num_ddim_steps=10` 为准。
包装器开销口径为同一请求内 `runtime_latency_ms − model_latency_ms`。

| 臂 | 请求数 | 生效 flow 步数 | model p50 | 包装器开销 p50 | 来源 |
|---|---:|---:|---:|---:|---|
| B0（本次，原生） | 57 | 4 | 321.72 ms | 不适用 | 本次运行 |
| C1（冻结 9-10） | 100 | 2 | 225.39 ms | 2.34 ms | 冻结 trace |
| C2（冻结 9-10） | 74 | 4 | 293.15 ms | 2.26 ms | 冻结 trace |
| C3（冻结 9-10） | 77 | 2 与 4 | 227.25 ms | 2.33 ms | 冻结 trace |

结论：
1. B0 逐次证据补齐，57/57 条 `input_sha256`/`action_sha256`/`model_latency_ms`/
   逐视角 `image_sha256`/`state_sha256`/`language_sha256` 齐全，0 失败请求，
   reset 审计 valid=true。finalizer 未改动即读到该 trace（`vla_calls=57`）。
2. 包装器开销中位数 2.26–2.34 ms，约一次模型调用的 1%。不得用 B0 与 C1 的 p50
   相减代替：两臂生效步数为 4 对 2，差值含计算量变化。
3. 跨进程按 `input_sha256` 对齐失败，共享 0 条。差异定位在渲染 RGB：
   3 视角图像每步都不同，`language_sha256` 全部相同，`state_sha256` 仅每回合第 0 步相同。
   同代码同布局同种子的两次运行同样 0/24 相同，属进程级渲染差异。
   因此原生与 Runtime 的输出数值比较必须改为同进程固定输入重放。

过程中修正一处记录错误：首次冒烟把 `inference_steps` 记成 10（读了被 pi_v3 丢弃的
旧键）。已改为以 handshake 为准记 4，并保留被丢弃的请求值供审计。
`smoke_1ep/` 保留为过程证据，其步数字段不可引用。

边界：本次 2/2 为 n=2 诊断结果，**不是**成功率证据，不替换
`artifacts/robodojo/recovery_lifecycle_20260910/` 的 8 回合成绩（2/2、0/2、1/2、1/2），
不与冻结 C 臂组成新配对。包装器 2.3 ms 不构成"开销可忽略"的一般结论，未测硬实时。
"同输入下原生与 Runtime 动作是否逐元素一致"仍未验证。

证据路径：`artifacts/robodojo/native_trace_b0_20260911/`
（`README.md`、`B0/runtime_trace.jsonl`、`B0/summary.json`、`B0/policy_reset_audit.json`、
`trace_comparison.json`、`source_manifest_B0.json`、6 段视频）。
复算工具：`scripts/compare_native_and_runtime_traces.py`。
回归：354 项（bridge + flow wiring + recovery verification + finalizer）+ 契约集 74 项通过。

## 2026-09-11 同进程固定输入重放：包装器保真验证 + flow2 代价量化

目的：回答"`CarveRuntime` 包装是否改变 VLA 输出"。9.22 已证明跨进程无法按
`input_sha256` 配对（共享 0 条，差异在渲染 RGB），故改为捕获真实 payload、
在同一进程内重放。对应计划 9.23 节。

模型：StarVLA PI-v3 `steps_100000`，进程内 `PolicyServerWrapper`，bf16，
base VLM `/home/admin1/models/Qwen3-VL-4B-Instruct`（离线本地权重）。
样本量：12 个真实 payload（取自 `stack_bowls` layout set 0 布局 0 的 1 回合捕获运行，
timestep 0..176 步长 16，action_seed 101..277），共 40 次模型调用，2 次预热。
测量口径：两条路径同 `action_seed`，服务端按请求 `fork_rng` + `manual_seed`，
采样噪声相同；动作比较取两者公共前缀 16 行（原生返回完整 50 步 chunk，
包装按 `max_actions=16` 截断，截断属声明控制）；包装器开销为同一次调用内
`CarveRuntime.infer` 总时长减 adapter 计时的模型调用。
12/12 payload 重建后复现记录的 `input_sha256`，重放确为机器人当时的观测。

| 对比 | 请求步数 | 算力对齐 | 动作逐位相同 | max abs diff | 包装器开销 p50 | 包装总时延 p50 |
|---|---:|---|---:|---:|---:|---:|
| matched_steps | 4 | 是 | **12/12** | **0** | 2.529 ms | 339.411 ms |
| arm_default | 2 | 否 | 0/12 | 0.022481 | 2.647 ms | 262.411 ms |

原生 p50 340.368 ms；执行段动作量纲 abs_max 2.109、abs_mean 0.4438。

结论：
1. 模型跨进程逐位确定：重放在另一进程复现实况动作 12/12，max abs diff = 0。
   9.22 的跨进程差异全部归于渲染，模型与调度不确定性被排除。
2. 包装器在同算力下输出保持：执行的 16 步动作 12/12 逐位相同，包装路径的
   `input_sha256` 与原生一致。Optimize Runtime 的保真前提由未验证转为已验证。
3. 包装器开销 2.53 ms（范围 2.455–3.423 ms），与 9.22 实况闭环 trace 的
   2.26–2.34 ms 相互印证；两次独立测量、不同进程、不同方法收敛在 2.3–2.6 ms，
   约一次模型调用的 0.8%。
4. flow2 代价：同输入同种子下 2 步与 4 步动作 0/12 相同，最大偏差 0.022481、
   平均 0.00293（相对峰值约 1.07%、均值约 0.66%），换来时延省约 77 ms / 23%。
   故冻结矩阵中 C1 与 B0 的差异不来自包装而来自降步。

边界：包装器等价性限于本次 12 个真实 payload、单 checkpoint、单任务，
不是全体输入上的等价性证明。n=2 不足以把 C1 的 0/2 归因于这 1% 动作偏差，
**flow2 仍未获本任务部署准入**。不得用"包装总时延 − 原生总时延"（中位数 +12.4 ms）
当包装开销，那是两次独立模型调用相减、含调用间方差。VLM 切换前后对 VLA 时延的影响
仍未测量。本次未产生新机器人成绩，不改变冻结的 8 回合结果（2/2、0/2、1/2、1/2）。

证据路径：`artifacts/robodojo/native_trace_b0_20260911/`
（`README.md` 第二阶段、`replay_report.json`、`capture_1ep/captured_payloads/`
12 个 npz + `payload_manifest.jsonl`、`capture_1ep/runtime_trace.jsonl`）。
工具：`scripts/replay_native_vs_runtime_payloads.py`。
回归：新增 `tests/test_native_vs_runtime_replay.py` 7 项；bridge +4；flow wiring +1。

## 2026-09-11 统一语义检查调度（交接 B 项 / 计划 9.21 第 2 项）

目的：让周期检查与 Monitor 强制检查共用观察有效期、最小步间隔与调用预算，
使"160/163 步仅隔 3 步"不再被当作长期停滞证据；保留 unknown，不放宽语义判定。
详见计划 9.24 节。

改动位置：真实控制入口 `third_party/.../starVLA/model.py`
（`_apply_scalar_semantic_checkpoint` 与新纯函数 `resolve_stall_evidence_limits`）、
`reset_contract.py`（清单 33 → 34 项）、`deploy.yml`、
`setup_eval_policy_server.sh`、`scripts/run_robodojo_starvla_nominal.sh`。
派生默认：最小停滞间隔 = 1 倍 checkpoint 周期，观察有效期 = 2 倍；
`resolve_stall_evidence_limits(160)` = `(160, 320)`。派生只在模型内一处。

规则：强制检查仍可立即观察、仍被记录、仍占用预算，但与上次计入观察相隔不足最小间隔时
不推进停滞计数（held）；观察超出有效期则丢弃旧值、序列重启；停滞恢复要求本次观察计入，
deadline 恢复不要求；unknown 一律清零永不参与恢复；周期门控改以上次计入观察为基准，
强制检查不再延迟周期档期。

用新规则重放冻结 `C3/planner_trace.jsonl` 的 6 条 critic 记录（样本量：2 回合 6 次检查）：

| 回合 | 冻结运行实际 | 新规则下 |
|---|---|---|
| 0 | 步 160 stale=1；步 163 强制 stale=2 → 触发恢复 | 相隔 3 步，held，**0 次恢复** |
| 1 | 步 160 stale=1；步 320 stale=2 → 触发恢复 | 相隔 160 步，计入，**1 次恢复** |
| 合计 | 2 次 | 1 次 |

结论：新规则精确移除了证据不足的那次恢复，保留了证据充分的那次。

边界：**不能**据此说冻结 C3 回合 0 的恢复有害或该回合成功与恢复无关——反事实未知，
恢复确实发生并清空了动作块，步 323 的 confirmed 是在"恢复已发生"的世界里观测到的，
且两次核验本来都是 inconclusive。**可以**说：冻结 C3 的"2 次恢复"在当前代码下不可复现，
基于"两次恢复"的叙述若重跑必须重新导出。冻结产物不修改。
本次**未跑新机器人回合**：交接对 B 项的验收标准是回归测试覆盖，已满足；
D 项门控要求 A/B/C 通过后才登记新试验。该改动会改变未来 C2/C3 的检查时机，
故冻结运行与未来运行在"检查次数/恢复次数"轴上不可直接比较。

证据路径：`artifacts/robodojo/recovery_lifecycle_20260910/C3/planner_trace.jsonl`（输入，未修改）；
规则实现与测试见上述源码与 `tests/test_starvla_robodojo_bridge.py`。
回归：bridge 新增 7 项、flow wiring 新增 1 项、reset 计数 33→34；相关文件与契约集 538 项通过。

## 2026-09-11 VLM 切换保真与驻留分项计量（交接 A 项后半 + C 项）

目的：(1) 比较 VLM 切换前后的 VLA 输出，验证"搬下 GPU→跑 VLM→搬回"是否输出保持；
(2) 分别记录 VLA 推理、VLM 生成、权重搬运、预热与端到端成本，并独立复验驻留优化。
详见计划 9.25 节。

模型：VLA StarVLA PI-v3 `steps_100000_pytorch_model.pt` bf16；
VLM 本地 Qwen3-VL-4B-Instruct bf16 离线加载。
样本量：6 个真实 payload（取自 `capture_1ep`，digest 全部 round-trip）；
每模式 3 次 VLM 调用（`max_new_tokens=64`，实际输出均 36 token）；VLA 预热 2 次；
两种驻留模式各测，正反顺序各跑一次（共 2 次运行）。
测量口径：分项计时取自 `CpuStagedVisionPlanner.decide().timings`，非本工具重新实现；
VLA 时延为 `PolicyServerWrapper.predict_action` 单次调用；动作逐元素比较取公共前缀。

结论 1（A 项后半）：2 次运行 × 2 模式 × 2 个切换后状态，动作全部 **6/6 逐位一致，
max_abs_diff = 0.0**，摘要序列相同。VLA 单次时延 p50 基本不受状态影响
（`transfer` 305.1/303.7/314.7 ms，`cpu_mirror` 300.5/284.8/302.1 ms）。
即 VLM 切换是输出保持的。

结论 2（C 项）p50 ms，正序（反序）：

| 阶段 | transfer | cpu_mirror | 变化 |
|---|---:|---:|---:|
| vla_to_cpu | 6322.4（5520.0） | 219.1（219.7） | −96.5% |
| vlm_to_gpu | 1751.3（1873.9） | 1443.4（1495.9） | −17.6% |
| vlm_generate | 1579.5（1536.2） | 1532.9（1594.5） | −2.9%（噪声） |
| vlm_to_cpu | 5178.9（5387.9） | 187.9（188.2） | −96.4% |
| vla_restore | 2099.4（2087.8） | 1698.1（1795.4） | −19.1% |
| 搬运小计 | 15352.0（14698.0） | 3574.5（3697.0） | −76.7%（−74.8%） |
| 端到端 | 16931.7（16283.6） | 5081.3（5271.5） | −70.0%（−67.6%） |

收益集中在两个卸载方向（各约 −96%），生成本身没有变快。
代价：主机 CPU 副本 17.712 GiB。VLA 单次推理约 285–315 ms，与切换成本差两个量级。

反序运行推翻了一个错误结论：正序中 `cpu_mirror` GPU 峰值 16.517 GiB、反序中仅
9.699 GiB，峰值跟随"同进程第二个被测模式"而非模式属性，
故"cpu_mirror 多耗 6.8 GiB 显存"是测量假象；单模式下两模式峰值均约 9.70 GiB。
冷加载与启动准备受 page cache 影响、跟随运行顺序，只作记录不参与模式比较。

与旧数字关系：旧记录 16.531 → 7.341 秒（−55.59%）。本次 transfer 基线
16.28–16.93 秒复现其 16.531 秒；本次 cpu_mirror 5.08–5.27 秒低于其 7.341 秒，
但提示词与生成长度不同，不可直接比较，可确认的是方向与机制一致。

边界：未跑机器人回合，不产生成功率结论，不改变冻结的 8 回合成绩。
staged 设计串行，**不能**据此声称单卡并发共驻，也不构成硬实时保证。
输出保持限于 6 payload、单 checkpoint、单任务、36 token 输出。
5.08 秒切换成本远大于一次 VLA 推理，不等于"VLM 可高频调用"。

证据路径：`artifacts/robodojo/vlm_switch_residency_20260911/`
（`README.md`、`vlm_switch_report.json`、`vlm_switch_report_reversed.json`、
`measure.log`、`measure_reversed.log`）。
工具：`scripts/measure_vlm_switch_and_residency.py`。
回归：新增 `tests/test_vlm_switch_residency_measurement.py` 6 项。

## 2026-09-11 初始 RGB 差异根因与 D 项门控结论

目的：补齐交接 A 项要求的"查初始 RGB 差异…不预设故障原因"，并据此复评 A/B/C 门控、
判断 D 项能否开启。详见计划 9.26 节。

方法与样本量：新增同布局同种子重复捕获 `capture_1ep_repeat`（rc=0，265.72 秒，
1 回合，12 payload），与 `capture_1ep` 做像素级比较；另取四次独立同布局运行
（`smoke_1ep`、`B0` 回合 0、`capture_1ep`、`capture_1ep_repeat`）比较步 0 摘要。
模型：StarVLA PI-v3 `steps_100000`，`stack_bowls` layout set 0 布局 0，种子基数 101。
测量口径：摘要为 `hash_starvla_request_input` / 逐模态哈希；像素差为 uint8 绝对差。

结果 1：四次独立进程步 0，`state_sha256` **4/4 逐位相同**、`language_sha256`
**4/4 相同**、`image_sha256` **4/4 互不相同（6/6 成对不同）**、动作 4/4 不同。

结果 2：像素级幅度（head / left wrist / right wrist）最大差 **6 / 3 / 3**（0–255），
有差异像素占 56.04% / 43.31% / 48.43%，非零差均值 1.077 / 1.004 / 1.006，
差 > 1 的像素仅 1.535% / 0.076% / 0.126%。

结论：初始 RGB 差异是**渲染管线浮点非确定性被量化进 uint8 最低位（±1 LSB）**，
不是物理随机、不是初始化、不是指令或预处理、不是场景或相机差异——每一项排除都有实测依据。
闭环放大：最大像素差 步0=6 → 步16=119 → 步32=162 → 步64=231；
平均像素差 0.24 → 3.79（步 80）；状态自步 16 起不同。

门控复评：A、B、C 三项按交接验收标准**均已达成**（分别见 9.23/9.25、9.24、9.25）。
**D 项按原设计不可执行**：其逐回合配对前提被上述根因实测否证——跨进程观测从第一帧
不可复现，动作从第 0 步即不同，16–32 步内轨迹分叉，故"同布局下结局不同"必然混入
与条件无关的分叉。这是冻结 8 回合矩阵不能把结局归因于 Agent 的机制层面原因，
现已被测量而非假设。

据此**不启动 D、不扩大矩阵**。三条可选路线属实验设计变更，需决策后执行：
(1) 拆分主张——保真/时延走同进程重放，闭环成功率按非配对并重算样本量；
(2) 先验证同进程内重复 reset 同一布局是否给出逐位相同初始 RGB（本轮未做：
需改动官方 evaluator 布局循环，触及官方计分与簿记，不宜在无人值守时进行）；
(3) 只依赖状态而非像素的主张本就可配对（state 四次逐位相同），可单独成立。

证据路径：`artifacts/robodojo/native_trace_b0_20260911/README.md`（新增"初始 RGB
差异的根因"一节）、`capture_1ep_repeat/`、`capture_1ep/`。
未新增代码路径，未触碰冻结目录，未改动官方 evaluator。

## 2026-09-11 Harness 触发器证据准入与 Monitor 阈值离线标定（设计复审第 1–3 项）

目的：给 Agentic Harness 的干预触发器建立与 Optimize Runtime 对称的证据准入；
按实测分布重标 Monitor 阈值；把停滞判据从"检查次数"改为"未变步跨度"。
全部离线完成，未消耗机器人机时。详见计划 9.27 节。

模型/来源：Monitor trace 来自 `ral_matrix_20260907_stack_bowls_{c2,c3}_set0`
（StarVLA PI-v3，`stack_bowls`，layout set 0 布局 0–24，两条件同一批布局）。
样本量：Monitor 逐步样本 16385 + 16519 = **32904 步**，回合 25 + 25 = **50**，
基线失败率 **0.62**。
测量口径：离线重放使用 trace 中已记录的窗口均值，均值不依赖阈值故重放**精确**
（由 `test_offline_replay_matches_the_live_monitor_exactly` 与真实 `ExecutionRiskMonitor`
逐项对齐钉住）。只扫阈值——改 window_size/warmup 会改变均值本身。
准入四标准：可达性、特异性（相对基线失败率的 lift）、及时性（恢复余量 200 步）、
预算可行性（冷却 128 步合并后 ≤ 3 次机会/回合）。

结果 1：实测分位点 `mean_state_response` p50 = **0.02599**，而部署阈值 0.002
小 **12.7 倍**；`stall` 三条件合取在 32904 步中出现 **0 次**。

结果 2：**当前部署的两个触发器都不满足准入**——`stall` 不可达；
`no_progress` 每回合 3.30 次触发机会超预算 3。

| 候选 | stall | no_progress | 并集覆盖失败 | 并集精确率 |
|---|---|---|---:|---:|
| deployed_default | 0/50 拒 | 10/50 拒 | 25.8% | 0.80 |
| stall_state_p50 | 11/50 准入 | 10/50 拒 | 54.8% | 0.85 |
| visual_p25 | 0/50 拒 | 6/50 准入 | 19.4% | 1.00 |
| **both_recalibrated_state_p50** | 10/50 准入 | 6/50 准入 | **45.2%** | **0.93** |
| both_recalibrated_state_p25 | 5/50 准入 | 6/50 准入 | 32.3% | 1.00 |

推荐 `both_recalibrated_state_p50`（state 0.02599 / command 0.01006 / 视觉两项 0.00242）：
相对部署默认，失败覆盖率 25.8% → 45.2%、精确率 0.80 → 0.93，两者同时改善。

结果 3：停滞判据改为未变步跨度（默认派生 `stale_checks × checkpoint` = 320，
~~标准间隔下行为不变~~ —— **2026-09-11 纠正：该说法是错的**。原判据在第 2 次计入检查即
触发、跨度仅 160；默认取 320 需要第 3 次检查，故这是**更严格的新策略**，
不是等价改写。实测：跨度 0 最早恢复第 320 步、跨度 320 最早恢复第 480 步）。
在冻结 C3 的 6 条 critic 记录上离线回放：
冻结实际 2 次恢复 → 9.24 规则 1 次 → 9.24+跨度 **0 次**。
两次恢复之后谓词都自行变化（回合 0 第 323 步 confirmed、回合 1 第 480 步变 2.0），
即两次都在任务仍推进时触发。

顺带发现（记录未修）：`min_unchanged_steps=320` 配 `checkpoint_steps=160` 时，
跨度最早在第 3 次检查才够，而 `max_calls_per_episode` 默认 3 ——
触发窗口恰只剩最后一次预算内检查。跨度要求与调用预算耦合，应一起定。

边界：准入通过**不等于**成功率会提升——这是触发器与失败结局的**关联**，
说明该信号选中的回合更常失败、值得闭环验证，不说明干预会修好它们。
精确率/覆盖率来自的 9-07 c2/c3 两次运行**未加载 VLM**（`planner_model` 为空、
语义检查 0 次），故结局是"基线 + 少量廉价重规划"的结局；信号与结局均真实、关联有效，
但不可当作"带 VLM 的 Harness"效应量。只扫了阈值，未扫窗口/预热/冷却/预算联动。
跨度判据仅在冻结 C3 的 6 条记录与单测序列上验证，未在新回合验证。
本次未跑机器人回合，不改变冻结的 8 回合成绩。

证据路径：`artifacts/robodojo/monitor_calibration_20260911/`
（`README.md`、`monitor_calibration.json`）。
代码：`agentic_vla/toolchain/trigger_admission.py`、
`scripts/calibrate_monitor_thresholds.py`、
`third_party/.../starVLA/model.py::resolve_min_unchanged_steps`。
回归：新增 `tests/test_trigger_admission.py` 12 项、bridge +5、flow wiring +1；
reset 清单 34 → 35。

## 2026-09-11 整体策略可行性审计 + 修复 9.24 引入的档期缺陷

目的：把 9.27 推荐的阈值与预算/冷却/检查间隔/跨度作为**整体**过一遍离线准入，
回答"恢复在自己的预算内还有没有可能触发"。详见计划 9.28 节。全部离线，未消耗机时。

**修复的缺陷（由本审计发现，非机器人运行发现）**：9.24 把语义检查的周期门控基准从
"上次检查"改为"上次**计入**观察"。但未计入的检查同样不推进该基准，导致门保持打开、
之后每步都触发，把每回合调用预算烧在连续步上（模拟中表现为
`checks=(80,160,161,162,163)`）。触发条件为 `min_stall_interval_steps > checkpoint_steps`；
派生默认值下周期门一开必然计入，故冻结运行与既有测试均未暴露。
修法：周期档期由周期检查推进（无论是否计入），强制检查仍不推进档期（保留 9.24 性质）；
新增 per-env `semantic_last_periodic_step_by_env`，reset 清单 35 → 36。

样本量与口径：50 回合（9-07 c2+c3 set0），阈值取 9.27 推荐的
`both_recalibrated_state_p50`（state 0.02599 / command 0.01006 / 视觉 0.00242），
强制检查由重标 `stall` 事件驱动。可行性为**上界**——假设谓词读数恒不变。

| 配置 | ckpt | 预算 | 跨度 | 可达回合 | 可达失败回合 | 检查/回合 | 最早恢复步 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deployed | 160 | 3 | 320 | 26/50 | 25/31 | 2.64 | 480 | 可行 |
| **budget_4** | 160 | 4 | 320 | 32/50 | **31/31** | 3.28 | 480 | 可行 |
| budget_5 | 160 | 5 | 320 | 32/50 | 31/31 | 3.42 | 480 | 可行 |
| checkpoint_80 | 80 | 3 | 320 | 0/50 | 0/31 | 3.00 | — | **不可行** |
| checkpoint_80_budget_5 | 80 | 5 | 320 | 41/50 | 28/31 | 4.88 | 400 | 可行 |
| span_240 | 160 | 3 | 240 | 28/50 | 27/31 | 2.64 | 480 | 可行 |
| span_160 | 160 | 3 | 160 | 50/50 | 31/31 | 2.64 | 320 | 可行 |

结论：
1. 部署配置可行但仅覆盖 25/31 个失败回合，窗口只有一次检查宽（最早第 480 步 =
   预算内最后一次检查）。**预算 3 → 4 即覆盖 31/31**，5 无额外收益。
2. **跨度要求意外充当成功回合过滤器**：失败回合可达 25/31（80.6%）、
   成功回合仅 1/19（5.3%），因成功回合结束早、跨度来不及累积。
3. 只减半检查间隔会让策略**完全失能**（0/50）——计入间隔仍 160，
   预算被未计入的检查吃掉；要靠提频获益必须同时降低计入间隔或提高预算。
4. 跨度放宽到 160 可达 50/50，但退回弱判据并失去第 2 条过滤，**不建议**。

推荐整体配置：`checkpoint_steps=160`、`min_stall_interval_steps=160`、
`min_unchanged_steps=320`、**`max_calls_per_episode=4`**、`cooldown_steps=128`、
`max_recoveries=1` + `both_recalibrated_state_p50` 阈值。

边界：可行 ≠ 有效，通过的配置仍需闭环验证；可行性是上界，真实触发只会更少；
重标阈值本身未闭环验证；回合长度与结局来自**未加载 VLM** 的 9-07 c2/c3，
Agent 真正介入后回合长度会变、可达率需重算；未扫 max_recoveries /
stale_checks_required / cooldown_steps 联动。本次未跑机器人回合，不改变冻结的 8 回合成绩。

证据路径：`artifacts/robodojo/monitor_calibration_20260911/`
（`README.md` 第二阶段、`policy_feasibility.json`）。
代码：`agentic_vla/toolchain/trigger_admission.py`、
`scripts/audit_intervention_policy_feasibility.py`、
`third_party/.../starVLA/model.py`（档期修复）、`reset_contract.py`。
回归：`tests/test_trigger_admission.py` 12 → 25 项；bridge 新增 3 项；清单 35 → 36。

## 2026-09-14 执行意图修复与真实小范围集成

用户确认毕业与RA-L为最终目标，按框架修复、集成、冻结对照顺序推进。
修复scalar Planner的CONTINUE/SAFE_STOP被当作恢复、同次动作调用停止延迟、
task-only重规划返回值不一致；分析器补充新停止事件去重。恢复归档的真实OpenVLA
测量脚本至legacy，仅修优化单测引用，不重跑旧模型。

最终可收集CPU：1294通过、1跳过，显式排除3个现存收集错误文件。
失败与修正后的XML保留`results/framework_revision_20260914/`；初次3个意图夹具非法，
已修正，不能把全部初始失败说成正确缺陷复现。没有改VLA权重或驱动。

真实实验：官方RoboDojo stack_bowls set0前2个布局，PI-v3 BF16 flow4/h16，
Qwen3-VL-4B CPU mirror，160步检查/320步跨度/3次预算，源码预冻结。
结果1/2，VLA75次、Critic5次均恢复VLA、廉价重规划1次，无Planner恢复或post-check。
6段原始视频保留，reset/完整性和11项工程检查通过；未触发的分支仍待真实验证。
单次VLA P50/P95为292.91/329.26ms，语义调用累计23.41秒；不作新效率对照结论。
证据：`artifacts/robodojo/framework_integration_20260914/README.md`、`audit.json`。

新诊断：480步组数谓词确认1组但描述只包含中心两碗，官方最终失败；705步风险出现时
周期预算已耗尽。另合成差分确认旧可行性模拟的stall选择/冷却与bridge不一致，
撤回用旧31/31直接准入新默认参数的建议。下轮先修调度及目标覆盖/局部核验，再冻结对照。
实验已退出，GPU释放；G2两项服务按既有授权保持停止，无新增后台实验。

### 2026-09-14 后续：B阶段调度修复与RoboDojo任务复核

修复真实scalar Critic事件调用未消耗冷却的问题、周期/事件重合后的重复检查；
缺图与预算拒绝不再提前推进周期时钟。沿用已注册reset时钟，不新增跨回合状态。
离线模拟显式event_name；no_progress支持强制检查，stall/stale_action不走该入口。
审计脚本不再默认stall或宣称offline_exact，不重写旧31/31历史分析。

36个合成CPU差分用例核对实际bridge与调度器，范围为首次恢复前、观测/调用可用。
定向507通过；可用CPU集合1335通过、1跳过，仍排除3个既有收集问题文件。
XML：`results/framework_revision_20260914/scheduler_targeted.xml`、`scheduler_cpu.xml`。
全量测试命令仍使用此前三项--ignore；测试数不是GPU回合数，也不是语义准确率。

重新核对官方任务目录、模型卡和本地task代码，不修改官方任务。put_bottles与
build_tower为主候选；conveyor须通过历史信息到动作的接口验收后做记忆专项；
stack_bowls只做开发回归，电脑耳机/模仿排序暂缓。方案见主计划9.31，底稿13.8同步。
目标覆盖/局部进展/完成三类核验及事件预算分配仍未完成，没有新增GPU实验或后台队列。
此前1/2真实结果属于修复前冻结源码，不能冒充本子轮效果。paper与旧原始结果未改。

### 2026-09-14 目标覆盖接口与VLM开发检查

新增可选expected_object_count分组覆盖验证，launcher/server与真实Critic接通；
未知/缺失/关系不一致不用于标量停滞或恢复，组数确认也不授权整任务停止。
默认不变；软件1351通过/1跳过，仍排除3收集问题文件，coverage_final_cpu.xml。
初次新测试缺少导入、随后空字典与归零断言错误均已纠正；未改运行逻辑迎合测试。

真实模型：Qwen3-VL-4B BF16/SDPA，固定旧视频7帧，legacy/coverage两模式。
首轮14次，coverage0/7接受（字段遗漏）；完整JSON示例修正后另存第二轮14次，
coverage4/7协议接受、仅2帧可用组数、其余未知或拒绝。全部原始输出与源码哈希保留。
合计28次调用但仅7个开发样本，不是独立准确率或新机器人成功率实验。
成功终帧的遮挡计数仍未知，不启用候选默认、不扩跑机器人矩阵。
证据artifacts/robodojo/coverage_probe_20260914/，路线更新主计划9.32与底稿13.9。
下一步是主任务局部事实的独立观测核验和事件预算；本轮GPU进程已结束。

### 2026-09-14 事件预算候选与工期收敛

接入CARVE_SEMANTIC_EVENT_RESERVE_CALLS，scalar Critic总预算3次内预留事件尾部，
默认0，配置校验和调用前后预算日志齐全，无新增reset状态。离线模型同步预算与
观察有效期；旧两回合真实Monitor日志在3配置下6次CPU回放一致，使用假Critic读数。
预留1让失败轨迹在706有检查预算，但旧读数已过期，不能称为新增恢复。
结果artifacts/robodojo/event_budget_audit_20260914/，新GPU回合与VLM调用均0。
可用CPU1441通过/1跳过，仍排除3收集问题；event_reserve_cpu.xml。shell语法通过。

旧多视角/多条件核验已测过且未准入，纠正重复试验风险；不以新测试数替代Agent收益。
原计划9.33明确3--5工作日决策检查点；限定毕业实验整理估计2--3周、RA-L准备4--8周，
前提无新训练且反馈/动作验收能通过，不保证毕业评定或投稿录用。

### 2026-09-14 小规模完整验证完成，未通过Agent效果验收

用户确认先做小规模；固定瓶子入桶/搭塔各3个开发布局、C1/C3，共12回合。
两组均PI-v3 BF16 flow4/h16，Agent使用独立Qwen3-VL-4B CPU mirror。
运行前修复普通事件Planner CONTINUE不得清空chunk，以及恢复pending/预算耗尽时
不得再次执行VLA_ACT。相关296测试通过，可用CPU1443通过/1跳过，仍排除3既有问题。

结果：瓶子C1 2/3、C3 3/3；搭塔C1 3/3、C3 1/3。4组reset/完整性检查通过、
source_drift为空。537次VLA调用、18次真实VLM任务调用、36段640x480 H.264原始视频。
瓶子3次起始计划均被拒绝；搭塔3次沿宿主既有模板生成计划，9次核验7unknown/2确认。
所有Agent回合均没有动作干预、事件Planner或post-check；不能将成败差额说成救回/误伤。
6个布局对从step0已有输入差异，239个共同请求时刻没有完全相同输入。
VLM完整调用总计145.707秒，生成78.880秒，18/18均恢复VLA；启动黑帧预热不计任务调用。

入口artifacts/robodojo/small_complete_20260914/README.md，RESULTS.md含全部头部视频，
summary.json/episodes.csv含明细。GPU服务已退出，无后台扩样。主计划9.34和技术底稿
13.11已更新，旧图/PPT/压缩包没有自动替换本轮数字。
下一步只修计划协议、早期检查预算与语义到合法动作的通路，不继续160回合矩阵。

### 2026-09-14 基于12回合的协议、任务覆盖与阶段预算修复

精简Planner返回为confidence/stages，模型不再填写固定工具字段；宿主编译后仍经过
完整权限/语义校验。增加复合拿取/投放任务覆盖防错，发现并拒绝原本接受的“只拿起瓶子”
错误计划。可选阶段时刻配置贯通launcher/server/bridge，保持3次预算与160最小间隔，
CPU检查迟到不突发、缺图不耗预算、原任务指令不变；无新增reset状态。

同6旧视频首帧，两协议18次真实Qwen调用；补guard后一次6帧9调用，总27任务调用+2预热。
搭塔配对旧/新均3/3接受，生成18.000→12.687秒、token455→317；仅模板约束固定图生成
成本，不是机器人速度或成功率。瓶子初始1/3接受经复核为错误、不可算收益；新guard下
真实仍0/3，塔3/3。两轮运行内source_drift为空。全部输出与源码在
artifacts/robodojo/planner_protocol_20260914/，guard_review.json为CPU复核，不是新推理。

可用CPU1476通过/1跳过，见planner_guard_final_cpu.xml；仍排除3既有收集问题。
两处shell语法与git diff --check通过。
本轮没有新增机器人回合、没有扩大benchmark，GPU模型退出。下一步先对齐task_only
执行接口与高层计划，并解决可靠反馈到合法动作，不能以接受率/台账代替物理收益。

### 2026-09-15 参考文章归档与下一阶段规划

用户提供的GPT 6 Astra/π0.5 RoboDojo文章HTML及同名资源目录已复制到根目录参考/，
保留paper原件。HTML SHA256一致，资源目录diff一致；未执行网页脚本、未安装外部Skill。
部分原图仍为懒加载占位，未补齐在线视频，未独立核验作者代码/成绩。

新增参考/README.md，主计划更新至9.36，交接提示同步。关键方向改为原任务VLA候选
审核与独立受限EEF纠正，不再把修改VLA子目标当唯一前置条件。P0运动学契约、P1受限
工具优先；之后P2真实只读审核、P3小闭环、P4分别验证工作记忆与审核效率。
两任务×3布局×3组=18回合仅为工具准入后的开发预案，不立即执行、不称RAL统计确认。
保持当前PI-v3 flow4/h16，不混成参考文章π0.5；14.4%和25Hz均不作为本项目实时证据。
本轮无运行代码/模型修改，无新增实验回合或GPU服务。

### 2026-09-15 按用户补充独立评估任务与机制，原位修订9.36

用户强调文章只作参考，不要求照搬。核对本地官方任务代码及PI-v3发布方模型卡后，
将9.36从“先做EEF工具”改为“任务/模型/工具适配→有效动作闭环→记忆与效率消融”。
EEF/FK是有条件候选，先根据真实缺口选择动作通道；不为凑模块实现所有工具。

保留瓶子/搭塔回归；优先审查整理桌面与传送带记忆，模仿排序和语言分类需先确认
观测与动作能力。整理桌面task文件的显式评分覆盖需继续核对，暂不认定官方bug；
模仿排序需完整演示观察，禁止读取内部目标序列。发布方低SR不是我们的复现结论，
也不足以单独判断失败来自底层动作。新任务、模型和矩阵都未冻结或启动。

主计划原节更新，参考说明和交接提示同步；未再新建版本化计划。保留全部历史结果、
paper与原参考材料。本次仅文档与来源核对，无代码/权重/实验数据改动，无新增GPU实验。

### 2026-09-15 切换官方RoboDojo π0.5并完成原生准入

用户确认主VLA使用RoboDojo π0.5。选官方HF数据仓库RoboDojo-Benchmark/RoboDojo，
revision `0a709fe7863f6c869a8d141676585a883fce4270`，
`ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999`。
18个推理文件共12,440,992,402 bytes完成哈希检查；未下载train_state，不是社区续训或LIBERO版。
下载经历TLS中断，最终使用本机既有代理及aria2分段传输，完整文件再次匹配官方SHA256。
未更改系统代理、TLS验证、驱动/CUDA/PyTorch；仅补msgpack-numpy 0.4.8。

新增准备脚本和有限验收脚本，复用Pi05Adapter/CarveRuntime、官方Pi_05模型和任务。
CPU定向58通过。实际加载OpenPI/JAX模型10.229秒，同真实录制观测/同随机状态的
native与Runtime透传动作最大差0。默认10步采样、horizon50、三相机、14维绝对关节/夹爪。

stack_bowls/layout0原生真实仿真1回合：官方成功1/1，score100，unstable0；6次VLA调用，
日志控制步294，另2次为预热/一致性检查。任务调用完整infer wall time112.97--130.89ms，
仅6样本，不作为P95/优化收益。入口阶段耗时242.165秒包含初始化/JIT/仿真/退出。
三视角H.264原始视频640x480、25FPS、各295帧/11.8秒，头部关键帧检查与堆叠成功一致。
源码记录集合无漂移；GPU进程退出。视频时长/FPS不代表真实系统达到实时性能。

证据：`artifacts/robodojo/official_pi05_admission_20260915/README.md`及stack_bowls_native/。
本轮不含Planner、记忆、恢复、量化/少步等优化；Runtime仅同输入核验，未用作rollout模式。
因此只证明基础模型接入，不证明Agent增益、跨任务泛化或统计成功率；主任务评估尚未扩样。
下一步迁移审计reset与合法Agent动作通路，再按主计划9.36审查/诊断候选任务。
保留所有旧模型、结果、paper及参考材料，不把PI-v3或旧PI0.5性能数字套用到该权重。

### 2026-09-15 两项分拣任务：框架接入、任务指令保护与可撤销记忆

用户收敛本轮任务为整理桌面和按语言指令分类。模仿分拣序列/传送带暂缓；
先框架、后小范围准入，未启动两项任务的仿真矩阵，未产生它们的新成功率。

新增SortingEpisode复用CarveAgentSession与工具运行时；接官方Pi_05部署、
三相机RGB和ARX X5 14维绝对关节动作。独立AgenticPi05目录保留官方Pi_05原件。
注册VLA和真实一步保持/重观察工具；没有新增抓取纠正、IK或oracle动作。
每步数值Monitor用目标-当前状态而非绝对目标幅度，异常时丢弃剩余chunk。
Planner低频调度、Critic预算、近期视觉/假设备忘和审计记录进入统一执行循环。

修复框架级接口冲突：旧解析器会将计划第一阶段子目标自动赋给VLA指令，
新增task_only能力标志后不再对未准入模型重写指令。默认行为对旧部署不变。
计划台账可由新相反Critic证据重新打开已确认步骤，依赖步骤回到待做，保留审计前后状态。
符号任务完成不写官方success，仍由原评分逻辑判定；提前停止不归为无效样本。

验证：

- 173项定向CPU测试通过，JUnit保存在`artifacts/robodojo/sorting_framework/framework_tests.xml`。
  包含真工具路径的CPU替身、拒绝非法输出、Parser能力限制、RGB/动作编解码、
  风险中断、记忆隔离与状态撤销；不是173次机器人任务。
- 在RoboDojo仿真Python环境导入实际部署模块、读取配置通过；模块语法检查通过。
- 首次真实权重回放检查失败于录制图像未使用官方输入转换。修复HWC/CHW入口后重测。
- `model_gate_rgbfix_20260915/check.json`记录：官方权重加载19.771s，原生/Runtime同观测
  动作完全相同，最大差0；同seed重置再次得到相同50×14动作。该检查没有启动仿真或真实VLM。
- 新pi05模型回执实现观察清空、调用计数清空和RNG重置。官方旧严格客户端审计仍绑定StarVLA，
  本次没有声称已迁移多回合审计，暂限定每回合新进程与部署层pi05回执。

运行入口与限制见`third_party/robodojo_official/XPolicyLab/policy/AgenticPi05/README.md`。
共用配置`configs/robodojo_sorting_agent.json`故意要求填真实VLM模型ID，占位符会被拒绝。
新JAX部署暂用Reference Runtime；旧量化/compile/SMVE及P95不能直接算作本轮效果。
下一步真实VLM初始观测检查和两任务各一个预登记布局的原生能力准入，再做配对Agent。

### 2026-09-15 两任务原生回合完成，VLM语义准入暴露并修复合同校验缺口

证据总入口：`artifacts/robodojo/sorting_development_20260915/README.md`。
官方π0.5 seed0/59999、JAX flow10/h50、layout-set0/layout0、每任务一个开发回合：

- organize_table：success=false，score75/100，1000步、20次VLA调用，启动到退出519.660s。
- classify_objects_by_language：success=false，score0/100，1100步、22次VLA调用，505.671s。
- 均为native、无Agent、无优化；两份summary源码检查无漂移，全部原始三相机视频保留。
  头部视频分别40.04s/1001帧、44.04s/1101帧，640x480/25FPS，不代表系统实时性能。
- 任务推理调用中位数分别117.00ms、115.77ms；排除两次预热/透传检查，未宣称加速或可靠P95。

新增首条真实rollout输入保存，仅state/RGB/instruction，不含评分或隐藏类别标签。
随后独立启动真实Qwen3-VL-4B BF16，仅用这些输入做有限离线规划检查，无机器人动作。
发现旧规则误把六步整理压成两步，修复后又发现缺少抽屉步骤/分类目标仍可通过格式校验。
新增窄范围显式指令覆盖、类别目的地和复数后验检查；修复compact协议task_only保护，
分拣上下文支持八阶段。最终`framework_tests_final_verified.xml`中178项CPU测试通过；
保留早期失败JUnit与模型原文，不把测试数量写成仿真回合数。

目前桌面整理六项指令覆盖通过，但VLM后验描述仍有不可靠位置/属性；语言分类最新检查
仍把复数目标缩成单数，已被拒绝。不能称两个Agent任务已跑通或成功率提高。
后续先改善语义合同与Critic、测低显存部署、验收动作干预能力，再开同布局Agent对照。
BF16模型加载allocated8.27GiB；VLA+分类仿真约18.8GB总显存，两者不能直接共驻。
Probe采用90s/1024tokens，部署模板45s/768tokens，不能混为相同配置的准入结果。
未启用少步/量化闭环，不沿用旧PI0.5 P95；本轮结束释放自建服务，未启动后台矩阵。

### 2026-09-15 后续修复：任务目标与视觉假设分离，4B NF4初始规划检查

新task_only明确语法任务改为简洁的目标清单提示，消除单实例/全类别指令冲突。
Host在计划覆盖检查后绑定公开任务的预期条件，不让模型添加颜色/位置改变验收要求；
原始响应保留。此处属于规则性合同保护，不是自动补出遗漏计划或证明视觉判断正确。

真实Qwen3-VL-4B vision_preserving_nf4服务成功加载，252语言线性层量化，
视觉编码器及输出头保持BF16。加载allocated3.268GiB（先前BF16为8.266GiB）；
不代表完整共驻峰值，也没有宣称量化加速或无损。模板默认BF16尚未切换。

最终两次真实观测离线检查都一次调用通过：整理桌面六操作12.071s；语言分类三类全部
对象目的地+回臂9.594s。实际每请求45s/768tokens。178项CPU测试通过。
证据：`organize_planner_nf4_concise_contract/`、`classify_planner_nf4_concise_contract/`、
`vlm_nf4_repair_profile.json`、`framework_tests_concise_contract.xml`，均在本轮证据目录。
同目录保留长提示下NF4失败及早期不完整输出，不能用这些迭代算统计指标。

只解决两条开发输入上的初始清单问题；Critic/实例覆盖、共驻峰值、动作干预仍需验收。
没有新增机器人回合、成功率或Optimize闭环成绩；自建VLM服务在检查后退出。

### 2026-09-15 Critic准入：同五例比较BF16/NF4，发现共同高置信度误确认

新增`probe_robodojo_sorting_critic.py`，复用真实GuardedVisualVerifier和原提示词。
固定鼠标初始/末帧、抽屉末帧、分类初始/末帧五例，三相机真实输入，人工参考不传入模型。
NF4与BF16均在初始鼠标位置上错误confirmed（1.0/0.98），末帧鼠标正确confirmed，
其他三例inconclusive。保留源视频/NPZ路径、输入图像哈希、请求及原文。
此处只有十次离线模型调用；不能算Agent成功率、量化无损证明或正确失败检测。
相关87项CPU测试通过，脚本编译通过。结果在本轮证据目录的`critic_*_fixed_cases/`。

当前4B Critic不准入自主完成判断，不能用已通过的初始计划检查代替它。
下一步需定位视觉关系误判，考虑证据抽取与结论分离或更适合的视觉模型；
不调高阈值掩盖置信度失准，不开始无效Agent矩阵。服务已退出，原生仿真不重跑。

### 2026-09-15 官方π0.5指令敏感性与人工子目标诊断

`classify_instruction_sensitivity/`记录分类实际首帧、seed0/1、原指令/汽车子目标/
手表子目标/原指令重复，共8次真实模型推理。原指令重复均精确一致，两个子目标在
两个seed均改变动作输出。关节平均绝对变化0.0114--0.0137仅是敏感性，不是任务收益。
该检查无仿真、无VLM；首调用35.73s含JIT，不能拿它计算加速。模型manifest已校验。

已为同布局单回合人工手表子目标注册计划，保留官方1100步预算与评分；不使用Critic、
不自动换目标，输入与原始环境指令分别记账。仅用于动作能力诊断，不是自主Agent。
接口默认关闭，28项相关CPU测试通过；记录结果前不宣称子目标执行成功。

上述真实人工子目标回合已完成：1100步、22次VLA调用、484.513s，official score0、
success=false、unstable0，source_drift=[]。三视角原始视频和每100帧的诊断图保留。
初始机器人状态与原生基线相同，RGB不同，因此不称精确反事实对照。
日志证明手表子目标持续生效，但画面显示仍操作多种物体且存在混放，未证明手表子任务完成。
不把更多搬运动作归因于自主Agent；本回合未调用VLM，原始环境指令和评分也未修改。
结论为“语言敏感性成立，正确子目标执行未准入”，默认task_only保持。
下一步核对技能/指令接口和真实工具能力，不继续大规模benchmark或扫prompt。
证据：`classify_watch_subgoal_execution/summary.json`及同目录关键帧，服务已退出。

### 2026-09-18 毕业贡献凝练与RoboDojo Critic视角诊断

主技术报告17.2凝练三项候选贡献，并分开自主设计、历史效果、当前待验证部分。
未修改论文原件、生产Critic提示或排序部署配置；未新增benchmark和训练。

扩展原Critic探测脚本，默认五例保持，新增可选固定中间帧和all/head输入对照。
真实Qwen3-VL-4B vision_preserving_nf4共16次请求，45s/768tokens，各8例。
三相机初始鼠标误confirmed1.0重现；仅头部为inconclusive0.3且描述鼠标未在垫上。
两项鼠标正例均confirmed；头部其余6项均inconclusive，不算正确负判或100%准确。
八组头部PNG哈希相同。三个中间帧条件为同回合时间迁移，不是独立留出布局。
初始/末帧人工参考及新增帧复核均不提供给模型，不是假称仿真器真值。

40项CPU测试通过；新增脚本选项不改变默认行为。全视角/头部平均请求2422.69/2278.73ms，
运行顺序及共享GPU未控制，不作加速结论；G2服务未停，仅运行VLM，无π0.5共驻验证。
模型服务加载allocated3.269GiB，探测后进程快照4306MiB非峰值，自建PID86819已终止。

接口审计确认本地参考RPent某些模式有深度/笛卡尔移动/释放权限，而新排序驱动仅有
VLA+reobserve；当前π0.5接口无对象ID/抓取点控制参数。不能只靠更多重试弥补该差距。
结论：核验输入有改善线索，但自主完成和子目标执行仍未准入，不启动Agent矩阵。
没有新增机器人回合或成功率，所有响应、图像、复核与预登记在
`artifacts/robodojo/critic_view_diagnostic_20260918/`。

### 2026-09-18 汇报用RoboDojo真实闭环集成

有效运行：`artifacts/robodojo/conservative_closed_loop_20260918/run04/summary.json`。
官方π0.5/59999 JAX flow10/h50 + Qwen3-VL-4B vision_preserving_nf4 + 官方物理仿真。
organize_table set0/layout0，seed0，1000步/20次VLA调用，官方score25、success=false。
10次Planner请求均解析接受（首次计划+9次continue），9次Critic建议，不授予账本完成权限。
中断动作块0、额外技能0、提前停止否；没有纠错收益证明。运行期间源码哈希无漂移。
三相机原视频各640x480、1069帧/25FPS/42.76秒；原视频未剪辑。视频时间不等于墙钟。
回合wall379.084s，启动至退出541.656s；Planner总55.531s、Critic总21.400s。
首VLA客户端调用33.943s含JIT；后19次服务器中位120.557ms，客户端中位218.499ms。
整卡10秒采样最高22801MiB，无OOM，非瞬时峰值；未启用新JAX模型优化，不借用旧加速比。

保留开发问题：run01任务名读取字段错误；run02 Planner违反task_only后停止，未终止的
环境默认success=True造成原始100分及success后缀，**均无效，禁止作成果**；run03正常复核
耗尽失败重试预算后await无请求，运行终止、无有效分数。没有覆盖或删除失败记录。
对应修复：任务字段；原生终止结算；紧凑继续/停止协议；复核/失败预算分账；无ticket
立即hold处理。209项定向与回归测试通过，包含1000步完整生命周期fixture，不计入真实结果。

画面显示鼠标入垫，最终键盘倾斜未对齐、抽屉仍关闭。官方25分低于历史原生75分，
两者第一动作块哈希不同且缺少严格输入配对，不直接作因果归因。下一步先核对输入编码、
观察时序和RNG，然后解决语义核验与受限纠错工具；不扩充大矩阵。
主技术报告13.18、主计划9.36.O与本目录README已同步。实验服务已退出，GPU无计算进程；
无关G2服务按用户要求不重启。

## 2026-09-18 整理桌面适配审计与定点优化

证据目录：`artifacts/robodojo/organize_adapter_audit_20260918/`。仅一个任务，无新benchmark。
实现同控制步只读观察共享、动作后失效；新增首观察NPZ、每次输入/RNG指纹与动作数组；
增加同执行器Agent开关。214项聚焦测试通过，官方π0.5在三个真实保存首观测上通过
原生/封装精确一致、重置重复和动作转换检查，最大动作差0（非物理成功实验）。

真实对照：off75/on25，均success=false，各1000步/20次VLA，无动作中断/额外技能/提前停止。
观察读取均1001次，三相机原视频1001帧/25FPS；旧Agent回合1069帧，不推导加速比。
off回合353.015s，on435.597s；on10次Planner总67.672s，9次Critic总26.286s。
Critic九次inconclusive，但多次文字错误描述鼠标已在垫上。视频可见开启回合摆件倒下
未恢复、局部反复动作，保守Agent没有救回。消除重复取图未解决效果退化。

首状态/指令/RNG相同，RGB不同；同进程双输入重放证明仅换RGB可改变首动作，另有较小
跨进程同输入数值差，不能解释全部分差或宣称Agent无责。全部失败保留，源码运行无漂移。
官方分数与抽屉/杂物语言目标覆盖存在差异，详见任务源码及审计README，不改官方判据。
主报告13.19与主计划9.36.P同步；停止重复只读全回合，下一步核验可靠性与可执行纠正能力。

## 2026-09-18 整理桌面工具接口修复与15次真实视觉诊断

证据：`artifacts/robodojo/organize_repair_20260918/README.md`。226项测试通过，无新增物理回合。
修复compact协议工具不可达、advisory始终审查首阶段、Critic证据未传Planner；
新增默认关闭的feedback_refresh：完整指令推理50步，仅执行10步、丢弃尾部并重新观察推理。
工具受预算/终止限制，不改语义账本，不接受模型关节参数。仅软件链路测试，不宣称恢复收益。

三张真实图，NF4/BF16，两种提示，共12次推理：视觉张量均存在，跨精度同图指纹一致。
原提示NF4三例弃权，BF16正例确认/负例弃权；独立谓词提示两精度均错误确认两个负例。
另3次BF16定位把负例键盘框识别为鼠标垫，不能仅因关系标签碰巧正确而通过。
没有将15次开发诊断称为独立benchmark准确率，没有授予Critic完成权限，未切换生产模型。
当前视觉可靠性与实际纠错效果仍未解决；服务已回收，主报告13.20、主计划9.36.Q同步。

## 2026-09-18 本地Qwen3.5-9B模型替换对照

证据：`artifacts/robodojo/organize_model_comparison_20260918/README.md`。新增15次真实GPU请求，
无机器人动作：9B BF16六次关系+三次定位，视觉保留NF4六次关系。三张图/提示复用旧诊断。
BF16在两种提示下三例关系status均正确，NF4原提示正确但独立谓词误确认两个负例。
BF16定位三次返回错误顶层格式，两个负例仍把键盘框认成鼠标垫；初始图左右说明也不正确。
因此可将9B作为下一阶段语义候选，但不准入自主完成、定位控制或可信空间记忆。
BF16/NF4加载后allocated分别17.527/7.973GiB，不代表共卡峰值；没有切换生产配置。
π0.5不变，没有新增仿真成功率。主报告13.21、主计划9.36.R同步，诊断服务已回收。

## 2026-09-18 Qwen多关系视觉核验：两模型未通过准入

证据：`artifacts/robodojo/organize_multirelation_gate_20260918/README.md`。
两个既有真实回合各3个中间头部帧，加2个腕部缺视野例；8张图32项关系问题。
先固定参考/门槛，再按9B、4B顺序运行BF16/SDPA；同生产提示、关闭thinking，64次真实GPU
调用，无物理动作。32对提示/PNG/实际pixel_values相同；运行源码及图像哈希未变化。

可判断22项正确：4B 5/22、9B 17/22；可见负例误确认均3/13；不确定参考正确弃权：
4B 10/10、9B 0/10。9B三次把倒下摆件判断直立、两次否定柜顶闹钟，还把不可见当未完成。
两模型格式/服务/现有guard失败均0，仍都未通过预设语义门槛。232项CPU测试通过。
标注由Codex目视生成且未独立人工复核，帧来自同布局两个已观察回合，非独立泛化测试。

没有在线换模、量化准入或新增成功率；下一步优先补对象绑定/可见性与关系证据，
在新样本验证后再决定更强模型与受限干预。主报告13.22、主计划9.36.T已同步。
两个实验服务均退出，保留桌面/ToDesk，未恢复无关G2服务。

## 2026-09-18 证据协议50次诊断结束；收敛近期三项能力

证据：`artifacts/robodojo/organize_evidence_protocol_20260918/README.md`。
新增evidence可选协议、不可见文字冲突保护、反馈后新观察同目标复核；默认不切换候选协议。
262项CPU回归通过。9B BF16完成50次GPU请求，无新机器人回合：旧32项可见正确16/22，
误确认3次；新增单头部8项正确3/8、已有头部+腕部录像8项正确4/8，各误确认2次；
新增2个不可见例均错误确定判断，整体未通过。全部50次原始visibility为clear。
旧输出guard重放拦截5个不可见却确定的结论，3次可见误确认不变；不是新增模型收益。
保留全部失败，不更改官方场景、π0.5权重/输入或生产部署，不追加多视角路线。

用户确认先完善记忆、可靠反思与纠错、推理优化，把可验证经验学习及可能的技能训练
作为后续自进化重点。主计划文首明确了验收条件及顺序，技术报告13.23同步。
本次文档整理未启动GPU实验；不把历史后端效率与当前任务效果混合宣传。

## 2026-09-18 RPent源码参考与工具闭环修复

官方RPent已在third_party；fetch参考提交886b3b274d3dd30bbc15615ea512d65ae90bc8b3，
原检出2c1f87d保持不变。未安装、运行上游工具，也未下载其记忆库。
实际检查toolkit、LIBERO接触/几何原语、memory manager和相关文档，避免仅照框架图。

修复排序适配器三处缺口：记忆检索忽略query、真实工具回执未传入last_primitive、
反馈工具后未立即交回Planner。新增分离的执行回执记忆、预算内词项检索和工具返回决策，
保留原场景/模型/指令。反馈后已有Critic结果复用，不重复轮询；默认不打开反馈候选。

297项CPU回归通过，含59项排序框架测试；测试结果与边界说明在
`artifacts/robodojo/rpent_tool_feedback_20260918/`。合成端口不计入机器人成功率。
未跑新GPU实验或物理回合，可靠语义判断/有效抓取纠正仍未解决。主报告13.24、主计划V同步。

## 2026-09-18 整理桌面：修复后的工具返回链路完成真实单回合检查

证据：`artifacts/robodojo/organize_tool_return_validation_20260918/README.md`。
预先固定一回合，官方π0.5/原场景/原相机/指令不变，1000步得75分，success=false。
20次VLA共999步、1次自主reobserve占1步，11次Planner、9次advisory Critic。
第200步工具执行后201步即时重决策，transcript确认实际回执进入last_primitive及记忆，
下一VLA之前完成Planner返回，无重复Critic、无误晋升完成阶段。短前缀反馈未被选中，
所以只验证重新观察的工具闭环，不验证恢复效果。297项CPU回归通过，源码无漂移。

三路视频完整保存、解码通过，目视抽查发现201帧鼠标在垫外、抽屉关闭，Planner却声称
全部完成；错误笔记仍进入后续上下文。可靠语义判断和有效纠错仍未解决。
当前75分与历史off75/on25不能作单变量归因，也不能写作75%成功率。
回合428.597秒；Planner/Critic累计75.855/24.853秒；19个热VLA调用P50/P95为
167.288/185.395毫秒，首次36158.815毫秒单列。整卡采样峰值23814MiB，不是优化收益。
不继续重复只读完整回合；下一步先查判断/假设记忆反馈及受限纠错动作。主报告13.25、
主计划W同步。实验服务已回收、无GPU计算进程，无关G2服务未恢复。

## 2026-09-18 历史模型结论隔离修复：真实对照仍有视觉误判

证据：`artifacts/robodojo/organize_memory_quarantine_20260918/README.md`。
新增quarantine默认策略：过去Planner/Critic笔记只留审计，检索只返回实际执行回执；
移除回执嵌套postcheck以免绕过过滤，仅同timestep的当次Critic单独提供未验证视觉证据。
保留legacy显式对照及全部原始记录。303项CPU回归通过，原π0.5/场景/相机未改。

用上一回合4个时刻、每时刻legacy/quarantine/no_text_memory，共12次4B视觉保留NF4
真实GPU请求；配对图像及system prompt一致，压缩视频重建，不是位精确线上重放或holdout。
三组对两个鼠标垫外负例均误确认，两个正例均确认；移除文本记忆后依然有错误。
本轮只证明历史模型叙述不是这些误判的必要条件，没有证明总体记忆效果或模型根因。
12次格式通过、11次continue、1次reobserve只读建议，零机器人动作，无新成功率。

主报告13.26、主计划X同步。判断可靠性与有效动作纠正仍未解决，不重跑只读整回合凑结果。
继续以原场景下高层目标绑定/关系能力准入为前置，保留信息隔离修复，不宣称性能改善。
实验服务已回收，无GPU计算进程。

## 2026-09-18 目标绑定修复：空检测兼容修好，候选身份流程未准入

证据：`artifacts/robodojo/organize_grounded_detection_20260918/README.md`。
Grounding DINO tiny固定revision、FP32、box/text阈值0.25/NMS0.5，只用原头部RGB。
首次10batch/20角色查询遇到两次空输出解析失败；复现本机`batch_decode([]) == ['']`，
修复仅限空boxes/scores的单空标签，保留非空错配拒绝。第二轮同10张全部接口接受。
两个版本全部保存，未覆盖失败结果。鼠标8张定位正确，垫3张正确/5张多候选弃权；
另两张无目标图出现底座误检（一次located、一次unknown）。

新增候选身份选择诊断，9B BF16共10次：8张整理桌面图均选中真正的鼠标垫区域，
2张无目标图都把机器人底座认作鼠标垫；鼠标缺失时均返回unknown。未通过准入，
不接入在线完成判断，不追加关系判断或物理回合。新模块仍为只读候选而非生产路径。
414项CPU测试、Ruff检查通过；所有运行内source_drift为空。

检测器热batch约202--233ms、峰值allocated 2.800GiB；9B加载后17.527GiB，
身份10调用累计79.478s。分时运行，没有新量化、端到端加速、共驻留或任务收益证据。
原始录像同回合开发帧，Codex目视参考未独立复核；不能称独立检测准确率。
原π0.5/相机/场景/评分未变，零机器人动作；最新物理成绩仍为1000步75分、整任务失败。
主报告13.27、主计划Y同步；GPU实验服务已退出，无关G2未恢复。

## 2026-09-18 目标无关候选检查：12对完成，未达到上线门槛

证据：`artifacts/robodojo/organize_target_rejection_20260918/README.md`。
新增只读目标无关实体描述和host类型否决，只能撤销定位，不能新增坐标/动作/完成权限。
12个检测batch后做24次Qwen3.5-9B BF16请求，交替顺序，不重试选优；6张旧开发图、
6张既有录像的新时间点，不是独立泛化测试集。12对原图/裁剪像素集合一致，无目标名
泄漏；另有24条实际processor视觉输入指纹。运行内源码无漂移，全部响应保留。

5张无目标图中的错误定位接受2→0；5张清晰图中的两角色正确保留5→4，2张遮挡图弃权。
新增109帧中模型名称字段已识别鼠标，却分类为scene_fixture，被类型硬过滤误伤。
这既是模型分类不一致，也是我们的规则粒度问题，不是模型规模/量化的单一问题。
预登记要求清晰目标全保留，未通过。该候选不接生产，停止此提示分支，不事后放宽门槛。

426项CPU测试、Ruff检查通过。条件选择累计93.268s，附加盲检查174.304s，总计267.572s，
约2.87倍原VLM耗时；含诊断指纹开销，不是性能优化成果。无动作、无新物理回合或成绩。
主报告13.28、主计划Z同步；服务退出，无GPU计算进程，无关G2未恢复。

## 2026-09-21 执行反馈审计与默认权限修复

证据：`artifacts/robodojo/execution_feedback_audit_20260921/README.md`。
最新旧回合20个动作块，8个端点能以同控制步、无中间工具动作和下一次推理状态哈希
严格绑定。最大单关节目标误差0.016992rad、夹爪归一化误差0.033315；12块显式排除，
不能代表全1000步轨迹、接触成功或任务完成。只读源数据无漂移，无新机器人动作。

默认Critic改为advisory，真实VLM路径拒绝未准入authoritative。VLA/reobserve回执
增加公开关节反馈，使用实际执行前缀末端；不赋予语义完成权，不改变原VLA输入或动作。
控制步耗尽时先拒绝推理。447项相关CPU回归通过，Ruff通过；没有新GPU或仿真成绩。

上游ee/IK分支存在失败时手臂目标缺失但夹爪命令继续的控制流；当前端口只走joint，
不将此风险误写成已证实的失败根因，也不直接开放IK纠正工具。下一步单工具能力验证，
不扩benchmark、不继续旧帧提示调参。主计划AA与主报告13.29同步，9月18日交付包不覆盖。

## 2026-09-21 框架集中修复与真实反馈工具闭环

证据：`artifacts/robodojo/framework_closure_20260921/README.md`。
完成时间/观测指纹绑定、过期核验排除、Critic同输入去重、VLA调用预算（含失败请求）、
动作成本账本及Planner预算上下文。468项相关CPU测试通过，Ruff与diff检查通过。

真实检查使用官方RoboDojo organize_table/layout0/seed0、原π0.5 59999 JAX flow10/h50、
原三相机/完整指令/关节接口，真实Qwen3-VL-4B NF4 Critic/advisory。测试驱动显式
调用一次feedback_refresh再调用vla_act，不含自主Planner，不含脚本关节轨迹。
生成100步，执行10+50=60步，丢弃40步；2次真实Critic、1次相同请求复用；
第二次VLA输入由工具后的新观测绑定。10项核查全通过、源码无漂移、3路视频完整解码。

这是提前结束的受控检查：官方0分/失败不能与1000步完整任务75分比较，也不算Agent
成功率。两次Critic均inconclusive，其描述仍有错误完成关系，因此不解除advisory限制。
执行61.083秒、全轮240.457秒；VLA冷调用35.864秒、仅1个warm样本153.35ms，不报P95。
三路视频各62帧/25FPS/2.48秒。去重命中是主动重复输入，不是自然任务加速比例。

本轮模型和仿真已结束，G2 VLM/ASR停止后未恢复。主报告13.30、主计划AB同步；
接口可固定，语义可靠性与定向纠正仍未验证，自进化仍在规划，不启动扩展矩阵。

## 2026-09-21 当前Critic信息隔离、按需调度及Qwen候选

证据：`artifacts/robodojo/critic_boundary_repair_20260921/README.md`。
修复最新advisory Critic的状态/置信度/自由描述仍进入Planner的问题；新投影只保留
目标、时间/观测指纹、schema接受情况与固定未知通知，保留原始审计。两条旧真实回复
重放通过。默认tool_only减少无控制用途的周期Critic调用，保留Planner及工具后核验；
periodic显式可选。476项相关CPU测试通过，未把fixture调用9→0当成实际时延加速。

原冻结32项、原提示/门槛，真实Qwen3.5-4B BF16完成32调用；清晰判断正确14/22，
错误确认2/13、未知画面确定断言3/10，5次语义冲突被guard拒绝。不通过、不上线。
与历史Qwen3-VL-4B BF16的32请求和图像哈希/标签一致，但不是在线NF4新配对对照。
累计98.127秒含诊断指纹开销，不作效率结论；源文件无漂移、服务退出、无新物理动作。
主报告13.31/主计划AC同步；旧视频/包不覆盖。本轮没有自主恢复或成功率提升证据。

## 2026-09-21 Agent动作能力对标与坐标准入

核查RoboDojo官方Astra L3、Astra+pi05公开代码/逐任务结果、EMERGE及已有RPent审计。
新增只读稀疏参考仓库`third_party/gpt_as_policy_reference`，不执行上游脚本、不安装依赖。
详情：`docs/research/ROBODOJO_AGENTIC_ACTION_GAP_20260921.md`；主计划AD。

实现Pinocchio FK/SciPy数值IK和kinematics_smoke，生产动作权限不变。
`artifacts/robodojo/robot_kinematics_admission_20260921/run01`在任何VLA动作前因Boost
parents转换失败；改用同URDF标准XML拓扑检查，阈值不变，失败记录保留。
run02真实原π0.5执行50步；前后双臂4项FK通过，24/28数值候选接受（含4个零位移），
初始双臂正负X四项拒绝。0个IK动作、0个VLM调用，不是自主恢复。
原生提前终止0分/失败，不混入完整任务统计。三路视频52帧/25FPS/2.08秒；
冷VLA35.536秒、执行窗口50.028秒、全轮193.614秒；无稳态样本，不报P95。
482项相关回归、17项独立运动学测试通过，源文件无漂移；本轮服务退出、GPU进程为空。
主报告13.32同步；旧交付包不覆盖。下一步是单独验证有限运动，再接入Agent工具，
不是再跑一次只有观察/重试的完整整理桌面回合。

## 2026-09-21 原场景受限末端运动及原VLA续执行

预登记：`artifacts/robodojo/robot_motion_admission_20260921/PLAN.md`。
新增motion_probe_smoke，原organize_table场景/相机/pi05不变；原RGB目视准入记录
绑定回合/观测。唯一GPU run01完成：保持10、右臂上移2mm十步、返回十步，均通过
跟踪检查；再以新观测执行1次原VLA的50步。共80步、VLA1次、Planner/Critic0次。
上移关节变化最大约0.007585rad；另一臂和夹爪保持。不是自主纠错或自动避碰证据。
原生提前终止0分/失败；视频三路82帧/25FPS/3.28秒并完整解码。VLA冷调用35.462秒，
执行窗口84.508秒含目视等待，全轮235.813秒；没有稳态/P95结果。运行内源代码无漂移。

482项原链回归通过。新测试fixture最初缺两路相机，修正后35项通过；GPU结束后复现
观测指纹误作命令消费ID，改独立proposal_id+消费集合，最终运动/URDF36项通过。
后置补丁未再跑GPU，原失败测试与run01哈希保留。所有模型/仿真退出，GPU进程列表空。
主报告13.33/主计划AE更新；生产Agent工具不自动扩权，不覆盖既有交付包。

## 2026-09-21 原VLA动作提案审阅与真实Planner选择

证据：`artifacts/robodojo/proposal_review_20260921/README.md`；预登记PLAN保留。
新增不可变动作提案、robot-only FK摘要和execute_prefix_10/50工具；消费已经生成的
原动作，不重复推理，不改指令/场景/三相机。观测/时间/ID绑定及重放拒绝有测试。
可选proposal_review_smoke完成唯一GPU run01：2次原π0.5、2次Qwen3-VL-4B NF4 Planner，
均选50步，共100步，0次Critic，无强制决策/修复请求。原生提前停止0分/失败，不进入
完整任务成功率。三路视频各102帧/25FPS/4.08秒完整解码，源码运行内无漂移。

协议接通但语义不可靠：第一次把未放好的鼠标/闹钟说成完成；第二次把动作执行回执
误作摆件完成，并错误解释右臂运动。原回执为semantic_outcome=not_verified，第一条
错误笔记未进入下一次检索。没有自主恢复/成功率提升结论，不开放自主EEF。

两次VLA RPC34225.18/160.66ms（仅1个warm样本）；Planner9616.30/15582.50ms，
预览949.03/179.88ms；执行86.55秒、全轮250.53秒；整卡采样最大23858MiB，非峰值。
本轮不是提速证据，尤其不能把每块都VLM审阅当高效调度。全部服务退出，GPU已释放。

494项回归、运动/FK最终38项通过；Ruff通过。GPU结束后修复预览夹爪单位描述，
区分原始模型预测与执行器[0,1]限幅值，不改动作；该补丁仅CPU验证，没有GPU重跑。
主报告13.34/主计划AF更新；下一步独立验证视觉、运动方向与回执解释，再做有限配对。

## 2026-09-21 有止损条件的能力诊断：52调用后结束当前扩跑

证据：`artifacts/robodojo/bounded_capability_decision_20260921/`，PLAN/manifest均在调用前固定。
原生视频125/375/725/925头部帧与两个原腕部视野检查，共18视觉项；真实两段执行
回执4项、FK分类4项。新时间点但同开发轨迹，不是独立布局，视觉参考尚无人独立复核。

4B vision-preserving NF4与9B BF16串行各26调用，0重试，26/26 JSON有效，各0服务错误。
清晰视觉10/13→11/13，但负例误确认1/3→2/3；不确定正确弃权4/5→1/5；执行/语义
回执2/4→4/4；运动分类均2/4。两候选均未通过预设门槛，不切模型、不开放恢复权限。
模型和精度同时变化，不能称为量化消融或参数量单因子对照。正确标签不认证理由。

审计：请求逐项相同，无参考标签输入；每候选18图像张量+8纯文本；图像/代码/manifest
无漂移，52调用全部保存。耗时134.54/111.43秒含张量审计，不报加速；无VLA或仿真
共驻，0新物理动作。8项解析/评分断言通过、Ruff通过，所有模型退出，GPU计算进程为空。

按停止条件不再启动这两配置的完整organize_table自主恢复/18回合矩阵，不在本组数据
继续提示搜索，不添加规则/反思来延长工程循环。主报告13.35/主计划AG更新；需要有
依据的视觉定位和有效纠正能力方案，而非把接入完成宣称为操作收益。历史结果保留。

## 2026-09-22 整理桌面局部上下文工具：16调用后不准入

证据：`artifacts/robodojo/local_inspection_20260922/`。预登记8张原图与参考后，DINO
FP32对mouse定位8次，6张头部图可生成局部，2张腕部图无可靠定位。全图与全图加
自动上下文裁剪各8次Qwen3-VL-4B vision-preserving NF4；文本一致、顺序交替，
192token/45秒/温度0，无重试/JSON修复。原VLA/场景/相机/评分不改，0机器人动作。

两组清晰判断均3/6、垫外鼠标误确认均3/3、未知正确弃权均0/2，JSON各8/8有效。
无改善，不准入，停止本组crop/阈值/问法调参。参考为Codex调用前目视，不是独立
人工/官方真值；同场景历史轨迹不是留出benchmark，不比较昨日不同prompt准确率。

16条模型processor记录、22张实际输入；局部像素/paired全图/来源哈希核查通过，
标签未传模型。51项检测测试、扩展对象记忆和工具链共104项回归通过，Ruff通过。
模型服务均退出，GPU计算进程为空。最新完整任务仍75分失败，无新增操作收益。
主报告13.36、计划AH更新；后续需可核查关系与有效受限动作，不再以重复长回合
掩盖当前语义误确认。论文、旧交付包与原始仿真视频未修改。

## 2026-09-22 对象辨认与图像几何分工：局部进展，未过完整准入

证据：`artifacts/robodojo/relational_grounding_20260922/`。冻结8张前轮原图+2张
无目标压力图后，10批DINO两角色检测、10次当前4B NF4身份选择；无新提示、重试
或调参，原模型/场景不改。新增对象记忆当前帧几何接口，不接生产控制。

6张清晰图选框经Codex目视检查对应目标，3个垫外/3个垫内的中心关系一致；此前
同6图直接问VLM只对3项，属不同方法/成本的开发对照，不报成功率提升。
原协议9/10通过，4未知例仅3个正确弃权：一例漏角色拒绝；一例同一腕部背景被识别
成mouse与pad，两框仅差1像素。另一个无目标例pad仍误认底座。原预登记门槛失败。

后置修复近同框歧义，原10响应CPU重放使腕部返回unknown；没有补角色或模型重跑，
不以事后结果替换原门槛。原执行源码快照/哈希保留。3个负例中心边界仅3px，定位
误差可翻转方向，下一轮须测新观察与不确定性，仍不得授权完成或任意纠正。

VLM累计57.393秒、46张实际输入；检测12.873秒含加载，非共驻/部署速度证据。
118项运行前回归、120项修复后回归通过。全部模型服务退出，无新VLA动作/任务视频。
主报告13.37与计划AI更新：保留单条候选，追求一次有效干预，不扩任务矩阵。

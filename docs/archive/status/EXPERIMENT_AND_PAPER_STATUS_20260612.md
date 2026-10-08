# CARVE-VLA 实验与论文状态整理

日期：2026-06-12

## 1. 当前判断

当前实验主线已经基本成型，可以进入论文结果整理与写作阶段。

本项目现在有三条互相支撑的证据链：

1. **Agentic Policy Harness 提升长程/失败恢复能力**
   - raw pi0.5 在 robosuite Stack 中 `0/10`；
   - pi0.5 + Agentic retry 在 matched setting 中 `10/10`；
   - 说明 Agentic physical recovery 能补足 VLA 闭环执行中的接触/长程失败。

2. **VLA 轻量化/实时推理优化**
   - pi0.5 fixed-noise low-step sampler 已完成 `1/2/4/6/8/10` sweep；
   - `1-step` 平均推理约 `118.15ms`，闭环 steady latency 约 `112ms`；
   - `2-step` open-loop MSE 最低，但延迟更高；
   - 当前部署推荐为 fixed-noise `1-step`，质量备选为 fixed-noise `2-step`。

3. **Agentic + realtime VLA backend 耦合**
   - `pi0.5 + Agentic retry + fixed 1-step` 在 10 trials 中 `10/10`；
   - full pi0.5 calls 从 raw 的 `43.0/episode` 降到 `6.0/episode`；
   - 说明 harness 不只是提高成功率，也减少了长程失败段反复调用 VLA 的成本。

结论：如果目标是硕士论文/组会/求职展示，这些结果已经比较完整；如果目标是 RA-L/ICRA 级别投稿，建议再补更大规模 seeds 或增加一个公开 benchmark/真实机器人 demo 作为增强项。

## 2. 论文主叙事建议

建议论文题目方向：

> Realtime Agentic Policy Harness for Efficient and Recoverable VLA Deployment

核心问题：

- VLA 在机器人闭环控制中推理慢、对接触失败和长程误差积累敏感；
- 直接训练/微调整个 VLA 对本地算力不友好；
- 需要一种工程可落地的 Agentic runtime：让 VLA 负责高层/常规动作生成，让物理层 Agentic harness 负责监控、恢复和实时调度。

核心方法：

1. **Agentic Policy Harness**
   - progress monitor；
   - physical critic；
   - retry/recovery skill；
   - recovery lockout 和退出条件；
   - trace logging。

2. **Realtime VLA Inference Module**
   - pi0.5 PyTorch backend；
   - fixed-noise low-step deterministic sampling；
   - `1-step/2-step` realtime tradeoff；
   - 4090 上的低显存 head-plus adaptation。

3. **Efficient Coupling**
   - VLA 不必在失败恢复阶段持续调用；
   - recovery skill 接管 stalled segment；
   - 成功率提升同时减少 full VLA calls。

## 3. 关键实验结果

### 3.1 pi0.5 + Agentic retry matched closed-loop

协议：

- Simulator：robosuite / MuJoCo
- Task：Stack
- Robot：Panda
- Backend：pi0.5 PyTorch head-plus 300
- Inference：fixed-noise low-step
- Horizon：`430`
- Trials：`10`
- Seed：`20260611` 到 `20260620`

| Method | Success | Steps mean | Full pi0.5 calls / ep | Recovery | Retry steps mean | Steady mean infer | Steady P95 infer |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw pi0.5, fixed `1` step | `0/10` | `430.0` | `43.0` | `0/0` | `0.0` | `111.99ms` | `122.37ms` |
| pi0.5 + Agentic retry, fixed `1` step | `10/10` | `254.4` | `6.0` | `10/10` | `194.4` | `110.40ms` | `116.87ms` |
| pi0.5 + Agentic retry, fixed `2` step | `5/5` | `254.4` | `6.0` | `5/5` | `194.4` | `137.22ms` | `142.74ms` |

解释：

- raw pi0.5 在相同任务中稳定失败，主要停留在 approach / grasp 前后；
- Agentic retry 将成功率从 `0/10` 提升到 `10/10`；
- Agentic retry 将 full pi0.5 calls 从 `43.0` 降到 `6.0`，因为失败恢复阶段由 physical skill 接管；
- `1-step` 和 `2-step` 都成功，但 `1-step` 延迟更低，因此是当前 realtime-preferred setting。

结果文件：

- raw 1-step first 5：`results/robosuite_stack_pi05_raw_fixed1_h430_5trials_20260611/summary.json`
- raw 1-step extra 5：`results/robosuite_stack_pi05_raw_fixed1_h430_extra5_seed20260616_20260612/summary.json`
- Agentic 1-step first 5：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_5trials_20260611/summary.json`
- Agentic 1-step extra 5：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_extra5_seed20260616_20260612/summary.json`
- Agentic 2-step：`results/robosuite_stack_pi05_agentic_retry_fixed2_h430_5trials_20260611/summary.json`
- paper table：`results/paper_assets_20260609/table_pi05_agentic_retry_matched.md`

### 3.2 pi0.5 realtime inference sweep

协议：

- same pi0.5 checkpoint；
- same 20 sampled robosuite Stack frames；
- compare fixed-noise and zero-noise samplers；
- evaluate action error and inference latency。

Fixed-noise sweep：

| Steps | Chunk MSE | Mean infer | First gripper acc |
|---:|---:|---:|---:|
| `1` | `0.1523` | `118.15ms` | `0.75` |
| `2` | `0.1467` | `136.53ms` | `0.75` |
| `4` | `0.1747` | `202.24ms` | `0.65` |
| `6` | `0.1887` | `255.46ms` | `0.65` |
| `8` | `0.2021` | `323.89ms` | `0.60` |
| `10` | `0.2077` | `374.43ms` | `0.60` |

解释：

- `1-step` 最适合 realtime；
- `2-step` open-loop MSE 最低；
- `4/6/8/10-step` 延迟逐步增加，且当前样本上误差没有明显收益；
- zero-noise 在 10-step 时 MSE 较低，但不适合 realtime 主线。

结果文件：

- fixed-noise sweep：`results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_fixed_noise.json`
- zero-noise sweep：`results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_zero_noise.json`
- figure：`results/paper_assets_20260609/figures/pi05_realtime_sweep.png`
- table：`results/paper_assets_20260609/table_pi05_realtime_sweep.md`

### 3.3 robosuite compact policy: Agentic + CAQ-Lite

该组不是 pi0.5 结果，而是完整训练-部署 pipeline 和 Agentic/CAQ-Lite runtime 机制验证。

协议：

- data collection：robosuite Stack `100` successful demos；
- learned backend：compact RGB+state action-chunk policy；
- mid-nudge perturbation；
- 10 trials per method。

| Method | Success | Full calls / ep | Reused / ep | Reuse ratio | Recovery |
|---|---:|---:|---:|---:|---:|
| learned + Light, no retry | `4/10` | `157.0` | `234.4` | `0.599` | `0/0` |
| learned + Agentic retry | `9/10` | `127.5` | `0.0` | `0.000` | `10/10` |
| learned + Agentic retry + LightSafe1 | `9/10` | `69.7` | `62.6` | `0.473` | `10/10` |
| clean Agentic retry + LightSafe1 | `10/10` | `64.3` | `63.8` | `0.498` | `10/10` |

解释：

- Agentic retry 是成功率提升主因；
- conservative CAQ-Lite 在保持 `9/10` 成功率的同时减少约 `45.3%` full calls；
- 该组可作为 runtime 机制与真实仿真训练-部署链路的补充证据。

结果文件：

- dataset：`results/robosuite_stack_e2e_v2_20260611/stack_demos_100eps.hdf5`
- safe1 mid-nudge summary：`results/robosuite_stack_e2e_v2_20260611/eval_midnudge_agentic_retry_light_safe1_10trials_summary.json`
- clean summary：`results/robosuite_stack_e2e_v2_20260611/eval_clean_agentic_retry_light_safe1_10trials_summary.json`
- demo video：`results/robosuite_stack_e2e_v2_20260611/eval_midnudge_agentic_retry_light_safe1_demo.mp4`

## 4. 展示资产

当前最适合组会和论文 qualitative figure 的资产：

- pi0.5 Agentic HD video：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_closed_loop_smoke.mp4`
- pi0.5 Agentic HD contact sheet：`results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_agentic_retry_hd256_contact.png`
- realtime sweep figure：`results/paper_assets_20260609/figures/pi05_realtime_sweep.png`
- Agentic retry matched table：`results/paper_assets_20260609/table_pi05_agentic_retry_matched.md`

建议论文中展示：

1. 一张系统框架图；
2. 一张 Agentic recovery flow；
3. 一张 realtime sweep 曲线；
4. 一张 matched trial table；
5. 一张 qualitative contact sheet。

## 5. 当前边界与不能过度声称的内容

可以声称：

- pi0.5/OpenPI backend 已接入 robosuite/MuJoCo closed loop；
- low-step fixed-noise sampler 可以显著降低 pi0.5 推理延迟；
- Agentic retry 在 matched 10-trial robosuite Stack 中将 raw pi0.5 从 `0/10` 提升到 `10/10`；
- Agentic recovery 同时减少 full pi0.5 calls；
- 该系统展示了 VLA backend + Agentic physical skill + realtime inference 的耦合价值。

不能直接声称：

- 不能说 pi0.5 本身已在 robosuite Stack 上训练到强成功率；
- 不能说结果已经覆盖所有 long-horizon manipulation；
- 不能把 compact policy 的成功率写成 pi0.5 成功率；
- 不能把 10-trial 写成大规模 benchmark；
- 不能说已经完成真实机器人验证。

## 6. 是否还需要补实验

按当前目标，核心实验已经差不多。

最小可选补充：

1. **增加 seeds/trials**
   - pi0.5 raw / Agentic retry 已从 `5` trials 扩到 `10` trials；
   - 若继续增强，可扩到 `20` trials；
   - 优先级：低到中；
   - 价值：提高统计稳健性，但当前主结论已经清楚。

2. **Agentic + CAQ-Lite with pi0.5**
   - 当前 pi0.5 + Agentic 已经把 full calls 降到 `6.0`；
   - CAQ-Lite 对这条链路的边际收益可能不大；
   - 可以作为 ablation，但不是当前必须。

3. **LIBERO 或 RoboTwin/ManiSkill 补充**
   - 对 RA-L/ICRA 更有帮助；
   - 但会增加工程成本；
   - 当前不建议立刻扩散，除非论文审稿目标明确要求更多 benchmark。

4. **真实机器人/更真实部署 demo**
   - 对求职展示和论文说服力有帮助；
   - 若时间有限，可先用 robosuite HD video + contact sheet。

## 7. 推荐下一步

建议从实验转向写作：

1. 更新论文 draft 的 Results section；
2. 把 `pi0.5 Agentic matched table` 作为核心表；
3. 把 realtime sweep 作为轻量化/实时性表；
4. 把 robosuite compact policy 作为 runtime ablation 和 pipeline evidence；
5. 把 IsaacSim/IsaacLab 写成 attempted deployment path 或 appendix，不作为主实验。

若还要继续跑实验，优先级为：

1. pi0.5 raw / Agentic retry 扩到 `20` trials；
2. 复跑一条更高质量视频或多视角视频；
3. 再考虑 Agentic + pi0.5 + CAQ-Lite ablation。

## 8. 当前文件入口

主要状态与日志：

- `docs/status/EXPERIMENT_LOG.md`
- `docs/status/EXPERIMENT_AND_PAPER_STATUS_20260612.md`
- `docs/status/ROBOSUITE_STACK_E2E_PIPELINE_STATUS_20260610.md`

论文与结果入口：

- 已归档整合草稿：`paper/archive/Agentic-VLA-runtime-draft-20260612.md`
- 当前 WAICA/LNCS-style 主稿：`paper/CARVE-VLA/root.tex`
- 编译 PDF：`paper/CARVE-VLA/root.pdf`
- 下一阶段实验计划：`docs/plans/CARVE_VLA_NEXT_STAGE_PLAN.md`
- 已归档实验与论文故事总览：`paper/archive/PAPER_STORY_AND_EXPERIMENT_PACKAGE_20260612.md`
- 当前 Results 草稿：`results/paper_assets_20260609/results_section_draft.md`
- 当前参考文献：`paper/CARVE-VLA/references.bib`
- `results/paper_assets_20260609/table_pi05_agentic_retry_matched.md`
- `results/paper_assets_20260609/table_pi05_realtime_sweep.md`
- `results/paper_assets_20260609/figures/pi05_realtime_sweep.png`

保留的核心结果目录：

- `results/robosuite_stack_pi05_raw_fixed1_h430_5trials_20260611`
- `results/robosuite_stack_pi05_raw_fixed1_h430_extra5_seed20260616_20260612`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_5trials_20260611`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_extra5_seed20260616_20260612`
- `results/robosuite_stack_pi05_agentic_retry_fixed2_h430_5trials_20260611`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611`
- `results/robosuite_stack_e2e_v2_20260611`
- `results/robosuite_real_video_stack_agentic_light_seed7_20260610`

主要脚本入口：

- `scripts/eval_robosuite_stack_pi05_policy.py`
- `scripts/eval_robosuite_stack_pi05_policy_trials.py`
- `scripts/sweep_pi05_robosuite_realtime.py`
- `scripts/plot_pi05_realtime_sweep.py`
- `scripts/eval_robosuite_stack_policy.py`

## 9. 2026-06-12 论文整合产物

当前已进入 CARVE-VLA 论文整合阶段，建议后续优先使用：

- `paper/CARVE-VLA/root.tex`
- `docs/status/EXPERIMENT_AND_PAPER_STATUS_20260612.md`

2026-07-15 最新调整：项目正式更名为 CARVE-VLA；旧 Agentic-VLA 投稿包和早期 runtime 草稿已完整归档，当前以 `paper/CARVE-VLA` 中的 WAICA/LNCS-style 主稿为唯一 LaTeX 入口。

- 当前正式 WAICA/LNCS-style 草稿：`paper/CARVE-VLA/root.tex`
- 当前编译 PDF：`paper/CARVE-VLA/root.pdf`
- 当前页数：`15` 页
- 当前标题：`CARVE-VLA: A Compute-Adaptive Agentic Runtime for Reliable Long-Horizon VLA Execution`
- 当前主证据：LIBERO-10 `pi05_libero` baseline `90.0%` -> refined full CARVE-VLA `92.5%`
- 当前补充证据：robosuite Stack 中 raw pi0.5 fixed `1-step` 为 `0/10`，pi0.5 + Agentic retry fixed `1-step` 为 `10/10`，full pi0.5 calls 从 `43.0/episode` 降到 `6.0/episode`
- 当前重点：讲清 CARVE-VLA 的 Agentic execution supervision、compute-adaptive runtime、LIBERO 实验协议、weak-task diagnosis、ablation 与 claim boundary
- 已投稿旧版归档：`paper/archive/Agentic-VLA-v1-submission/agentic_vla_paper_v1.pdf`

主稿沿用已复制到当前目录的 WAICA/LNCS-style 模板文件；原模板包保存在旧投稿归档中。

2026-06-16 进一步完成模板切换与正式图表整理：

- `paper/CARVE-VLA/root.tex` 已整合新版框架图、execution-supervision loop、模块机制图、realtime runtime 图与 LIBERO 数据图；
- `paper/CARVE-VLA/IMAGE2_FIGURE_BRIEF.md` 保留 Image2 图表需求；
- `paper/CARVE-VLA/root.pdf` 为当前编译稿。

这版主稿已经完成五个写作整合任务：

1. Method：补齐 Agentic Policy Harness、Realtime VLA Inference Module、Agentic + Realtime coupling；
2. Results：将 LIBERO-10 作为核心 benchmark 主结果；
3. Realtime sweep：加入 fixed-noise low-step tradeoff，并将 pi0.5 robosuite matched 10-trial 结果作为 deployment study；
4. Supplement：加入 compact policy + Agentic retry + CAQ-Lite 机制验证；
5. Limitations：写清楚 benchmark scope、physical recovery skill、sim state assumption、trial count、quantization boundary。

2026-06-12 已完成项目清理：旧 smoke、diagnostic、IsaacSim/IsaacLab、Qwen/VLM 下载物、旧 LIBERO 中间结果和过期草稿已删除；`paper/Agentic-RAG-VLM` 未被清理操作触碰。

# Agentic-VLA 提交素材清单 V1

本文件用于整理当前投稿版本真正使用的论文与配套素材，避免继续沿用已经被替换的旧图或旧说明。

配套结果证据说明：

- `/home/admin1/ct/Agentic-VLA/RESULTS_EVIDENCE_GUIDE.md`

## 1. 当前投稿版核心素材

- 主稿源码：`paper/Agentic-VLA/agentic_vla_paper_v1.tex`
- 主稿 PDF：`paper/Agentic-VLA/agentic_vla_paper_v1.pdf`
- 图表规划：`paper/Agentic-VLA/FIGURE_TABLE_PLAN_V1.md`
- 文献映射：`paper/Agentic-VLA/REFERENCE_MAP_V1.md`
- 图像提示词：`paper/Agentic-VLA/FIGURE_PROMPTS.md`
- 组会投屏稿：`paper/Agentic-VLA/report/GROUP_MEETING_REPORT.md`
- 组会压缩包：`paper/Agentic-VLA/report.tar.gz`

## 2. 当前论文实际使用的图

- `paper/Agentic-VLA/figures/fig.1-gemini.png`
  - 论文方法总览图
- `paper/Agentic-VLA/figures/fig.2-gemini.png`
  - `Transition Agent` 机制图
- `paper/Agentic-VLA/figures/fig.3-gemini.png`
  - `Scene Priors and Memory` 机制图
- `paper/Agentic-VLA/figures/fig.4-gemini.png`
  - `Critic / Retry` 机制图
- `paper/Agentic-VLA/figures/fig_libero10_task_comparison.png`
  - `libero_10` 分任务结果图

## 3. 当前论文不再使用的旧占位图

以下文件已被新图替换，不再属于当前主稿的 active assets：

- `paper/Agentic-VLA/figures/fig_method_overview.png`
- `paper/Agentic-VLA/figures/fig_transition_gap.png`
- `paper/Agentic-VLA/figures/fig_scene_priors_memory.png`
- `paper/Agentic-VLA/figures/fig_critic_retry_flow.png`

说明：

- 它们可以保留为历史草图或生成脚本产物
- 若需要清理磁盘或减少混淆，可删除
- 删除前应确保正文、报告正文和当前素材说明文件都不再引用

## 4. 当前投稿主线

- `A1 baseline`：`results/ablation_B0_pi05_libero_10_20260512/summary.json`
- `Full`：`results/ablation_FULL_refined_libero10_20260518_2059/summary.json`
- 当前主线结论：
  - `A1 = 90.0%`
  - `Full = 92.5%`
- `A2`、`A3`、`A4` 不进入当前论文主结果表，只作为补充证据
- 机制诊断只保留最小证据链：
  - `Task 8` 的 `Full + w/o` 消融目录（用于解释 Full stack 协同关系）

## 5. 当前图注定位

- `Figure 1`
  - `Overview of Agentic-VLA. A frozen VLA policy remains the action generator, while Transition, scene priors and memory, and Critic/Retry act as external inference-time process-control modules around a unified rollout loop.`
- `Figure 2`
  - `Mechanism illustration of the transition-aware controller. The figure explains how a short intermediate reconnection motion repairs the state gap that can appear between chunked subtask segments.`
- `Figure 3`
  - `Scene priors and memory augmentation. Structured object-target priors and lightweight episodic memory bias the frozen policy toward more stable interaction patterns without retraining.`
- `Figure 4`
  - `Critic / Retry as a conditional intervention layer. The critic only audits progress and triggers recovery when execution becomes inconsistent with the intended completion state.`
- `Figure 5`
  - `Per-task success comparison on libero_10, highlighting both the overall gain and the remaining weak-task region.`

## 6. 结果更新后优先回填项

1. 更新 `Task 8/9/6` 的 `v2` targeted evidence 摘要
2. 若 `Full v2` 显著改善，再决定是否追加整套 `libero_10` 重跑
3. 若论文正文要引用 `v2`，只允许引用已完成并有 `summary.json` 的真实目录
4. 从真实视频中抽取 weak-task 可视化帧，作为补充材料或组会图例

## 7. 快速命令

### 继续低成功率任务矩阵

```bash
bash /home/admin1/ct/Agentic-VLA/scripts/run_low_success_ablations.sh
```

### 生成定性案例图

```bash
python /home/admin1/ct/Agentic-VLA/paper/Agentic-VLA/prepare_qualitative_figure.py \
  --failure-video /path/to/failure.mp4 \
  --success-video /path/to/success.mp4 \
  --output /home/admin1/ct/Agentic-VLA/paper/Agentic-VLA/figures/fig_qualitative_case.png \
  --title "Weak-task qualitative comparison" \
  --num-frames 4
```

### 导出 LaTeX 结果行

```bash
python /home/admin1/ct/Agentic-VLA/paper/Agentic-VLA/export_results_tables.py \
  --a1-object /home/admin1/ct/Agentic-VLA/results/ablation_B0_pi05_libero_object_20260512/summary.json \
  --a1-goal /home/admin1/ct/Agentic-VLA/results/ablation_B0_pi05_libero_goal_20260512/summary.json \
  --a1-libero10 /home/admin1/ct/Agentic-VLA/results/ablation_B0_pi05_libero_10_20260512/summary.json
```

## 8. 提交前最后检查

- 只使用真实 `official LIBERO` rollout 结果
- 不把进行中的 `v2` 目录写成已完成结果
- 不把 `Vision Prompt` 写成当前主贡献
- 不把 `A2 / A3 / A4` 写进主文主结果表
- 保持主叙事聚焦 `A1 baseline + Full`

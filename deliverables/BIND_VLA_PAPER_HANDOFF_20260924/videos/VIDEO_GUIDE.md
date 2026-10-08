# 代表视频与可用说法

所有 `.mp4` 是原始或原有交付副本，未用生图模型重造。`VIDEO_MANIFEST.tsv` 给出包内路径、仓库源路径、SHA-256 和标签。**视频中的动作过程不是官方任务结局**；结局以相应 `analysis.json`/`summary.json` 为准。要在论文附件或答辩中使用，请同时显示模型、任务、条件和 outcome。

## 与论文核心结果对应的 RoboMME

| 包内视频 | 条件与结局 | 严格解释 |
| --- | --- | --- |
| `robomme_swap_ep39_A_fail.mp4` + `robomme_swap_ep39_B_success.mp4` | 同固定初态；无/有公开示教身份记忆 | 固定 12 对中的一个救回；总体 `5/12->9/12`、5 rescue/1 harm，`p=0.21875` |
| `robomme_swap_ep43_A_success.mp4` + `robomme_swap_ep43_B_fail.mp4` | 同任务另一个固定初态，反方向 | 一个伤害案例；不能只展 ep39 而隐去这个反例 |
| `robomme_routestick_ep31_raw_fail.mp4` + `robomme_routestick_ep31_harness_success.mp4` | 冻结双任务测试中的唯一救回 | 两任务合计 `3/32->4/32`；不是普适性能改善 |
| `robomme_ep39_public_demo.mp4` + `robomme_ep39_memory_tracking.mp4` | 公开初始示教及对象身份跟踪可视化 | 不是机器人执行录像，不可记为成功 episode |
| `robomme_unmask_ep23_A_success.mp4` + `robomme_unmask_ep23_B_fail.mp4` | 跨任务始终开启记忆的伤害 | 转移组 `15/16->13/16`，不能宣称记忆默认通用有益 |
| `robomme_stage_ep43_old_fail.mp4` + `robomme_stage_ep43_retest_success.mp4` | 同已知失败初态上的阶段回执修复前后 | 开发复测，**不是独立 rescue** |
| `robomme_stage_ep47_raw_success.mp4` + `robomme_stage_ep47_receipt_success.mp4` | 此前未用初态的功能检查 | 两臂都成功；不证明成功率或效率增益 |

RoboMME 量化来源：`../artifacts/robomme/memory_ab_confirm_20260923/analysis.json`、`../artifacts/robomme/harness_transfer_gate_20260924/analysis.json`。历史八任务开发汇总见 `../results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json`。

## LIBERO-PRO 的救回与误干预

`libero_frozen_fail.mp4` / `libero_agent_success.mp4` 是一对展示素材；`libero_frozen_success.mp4` / `libero_agent_false_intervention_fail.mp4` 是反例展示素材。整体 400 配对为 `180/400->183/400`（4 rescue、1 harm），不是“显著提升”；调用数减少但 wall time 增加。源汇总为 `../results/libero_pro_full_study_20260825/aggregate/study_summary.json`。这四段旧展示副本的任务/episode 详细标注应回原仓库逐回合元数据核实；本包只用于说明现象，不据此推断精确任务分布。

## RoboDojo 任务图库（不是统一对照）

| 视频序号 | 任务与 checkpoint | 结局与能说的内容 |
| --- | --- | --- |
| `01` + `02` | `build_tower`，**历史 StarVLA PI-v3**，640 步注入相同 `stale_action_hold` 故障；C1 对 C3 | C1 失败，C3 成功；3 对 `0/3->3/3`、`p=0.25`。C3 使用 Planner 决策回放与 profile boost，只能演示受控故障联合恢复路径 |
| `03` | `organize_table`，官方 PI0.5 59999、受限 Agent | `75/100`，`success=false`；当前框架的部分执行，不是救回 |
| `04` | 同任务与 checkpoint，原生 PI0.5 | `75/100`，`success=false`；与 03 非严格同服务配对 |
| `05` | `classify_objects_by_language`，官方 PI0.5 原生 | `0/100`，`success=false`；没有 Agent 干预 |
| `06` | `stack_bowls`，官方 PI0.5 原生 | `1/1` 成功；展示策略能力，不是 Harness 增益 |
| `07` | `put_bottles_into_dustbin`，历史 PI-v3 C3 | 该条件 `3/3` 成功；接受语义干预数为 0，不能归因于 Agent |
| `08` | `store_laptop_and_headphones`，历史 PI-v3 B0 | 该条件 `0/3`；展示能力边界 |

RoboDojo 对应 `artifacts/robodojo/` 中的结果 JSON 和 [原视频集说明](../deliverables/BIND_VLA_VIDEO_SHORTLIST_20260924/README.md)。缺少可核查本地完整录像的 `imitate_sorting_sequence` 和只剩 `.tmp.mp4` 的 `match_and_pick_from_conveyor` 没有收入。曾有一段文件名含成功的 `organize_table` 视频，但提前退出导致成功标记无效；审计 `../artifacts/robodojo/conservative_closed_loop_20260918/run02/AUDIT_INVALID_RESULT.md`，**禁止用作正例**。

投稿视频若只能选少数片段，优先 RoboMME ep39 A/B 与 ep43 A/B 来呈现机制和边界；RoboDojo 01/02 可以作为明确标注的受控故障附加演示。若会议/期刊要求自然任务上当前官方 PI0.5 的成功演示，现有 03/04 **不满足**，不能靠剪辑把失败写成成功。

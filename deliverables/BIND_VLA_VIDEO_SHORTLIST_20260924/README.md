# BIND-VLA RoboDojo 任务视频集

更新：2026-09-24。本目录按 **任务类型**收录目前可核查的 RoboDojo 完整评估录像：8 段未剪辑的头部相机 MP4，分属 6 类任务。整理桌面保留原生和受限 Agent 两段；搭塔保留同故障配对两段。它们并非同一模型的八次独立对照，更不是都由当前 BIND-VLA 获得的成功案例。每段原始路径、大小和 SHA-256 在 [SOURCE_MANIFEST.tsv](SOURCE_MANIFEST.tsv)；原始多相机文件在各结果目录，未复制的历史开发视频见 [全量索引](../../paper/CARVE-VLA/ral_draft/VIDEO_EVIDENCE_INDEX.md)。

| 文件 | RoboDojo 任务 | 模型/条件 | 官方结局 | 展示定位 |
| --- | --- | --- | --- | --- |
| `01_robodojo_build_tower_PIv3_fault_baseline_fail.mp4` | `build_tower` 搭塔 | 历史 StarVLA PI-v3；C1，640 步注入停滞故障 | 失败，0.3 | 受控故障配对的基线 |
| `02_robodojo_build_tower_PIv3_fault_harness_success.mp4` | `build_tower` 搭塔 | 同初态/故障；C3，Planner 决策回放与恢复 profile 切换 | 成功，1.0 | 联合恢复机制演示，**不是自然故障或本回合在线 VLM 规划** |
| `03_robodojo_organize_table_pi05_agent_partial_fail.mp4` | `organize_table` 整理桌面 | 当前官方 PI0.5 59999；受限 Agent 闭环 | 失败，75/100 | 当前链路部分完成，不能称成功率收益 |
| `04_robodojo_organize_table_pi05_native_partial_fail.mp4` | `organize_table` 整理桌面 | 官方 PI0.5 59999 原生执行 | 失败，75/100 | 原生参照；与 03 不是严格同服务配对 |
| `05_robodojo_classify_by_language_pi05_native_fail.mp4` | `classify_objects_by_language` 按语言分类 | 官方 PI0.5 59999 原生执行 | 失败，0/100 | 语言条件压力案例，没有 Agent 干预 |
| `06_robodojo_stack_bowls_pi05_native_success.mp4` | `stack_bowls` 叠碗 | 官方 PI0.5 59999 原生执行 | 成功，1/1 | 预训练策略能完成该初态，**不是 Harness 增益** |
| `07_robodojo_put_bottles_PIv3_C3_success.mp4` | `put_bottles_into_dustbin` 投瓶 | 历史 StarVLA PI-v3；C3 | 成功；该条件 3/3 | 有 VLM 服务调用，但此组接受的语义干预为 0，不能归因于 Agent |
| `08_robodojo_store_laptop_PIv3_B0_fail.mp4` | `store_laptop_and_headphones` 收纳电脑与耳机 | 历史 StarVLA PI-v3；B0 | 失败；该条件 0/3 | 更复杂任务的能力边界，不是正面成果 |

**上传建议：**作为方法机制展示，优先 01--02 并在标题写明“PI-v3 / controlled fault”；作为当前模型场景展示，选 03 和 06，但必须分别写“失败、75分”和“原生成功”。05、07、08 用于任务图库或答疑，不建议把全部八段不加区分地提交为论文方法成功视频。

01--02 的三个固定初态结果为 C1 `0/3`、C3 `3/3`，精确双侧 `p=0.25`，且恢复时同时切换了推理 profile；只能用于受控故障下的联合机制说明。[三配对分析](../../artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/summary.json)、[C1 原始回合](../../artifacts/robodojo/fault640_c1_v2_build_tower_flow2_seed0_20260902/summary.json)、[C3 原始回合](../../artifacts/robodojo/fault640_c3_replay_build_tower_flow2_flow4_seed0_20260902/summary.json)。整理桌面 [原生](../../artifacts/robodojo/sorting_development_20260915/organize_native/summary.json)与 [受限 Agent](../../artifacts/robodojo/organize_tool_return_validation_20260918/run01/summary.json)均为官方失败。另有一个文件名含 `success` 的旧整理桌面视频属于[已审计的无效成功标记](../../artifacts/robodojo/conservative_closed_loop_20260918/run02/AUDIT_INVALID_RESULT.md)，**未收入本图库**。

`match_and_pick_from_conveyor` 只存续未完成的 `.tmp.mp4` 视频流、没有有效 `_result.json`，所以未收入。此前提过的 `imitate_sorting_sequence` 在本地没有实验录像，不能把 benchmark 官方演示误作我们的结果。完整上传材料若要宣称**当前 PI0.5 自然任务上的 Agent 增益**，仍需新的冻结配对实验。

## 旧任务只作量化表格

此前挑选的 RoboMME 视频已从本**上传目录**移除；原始录像和日志没有删除。当前短稿/技术报告继续按下表记录其结果，不把单个成功片段当主展示：

| RoboMME 对照 | 固定样本结果 | 解释 |
| --- | --- | --- |
| VideoUnmaskSwap 身份记忆确认 | A `5/12` -> B `9/12`；5 rescue、1 harm，精确 `p=0.21875` | 同任务有方向性收益，未证明总体显著性 |
| RouteStick + PickHighlight 冻结 Raw/Harness | Raw `3/32` -> Harness `4/32`；1 rescue、0 harm，精确 `p=1.0` | 改善很有限，不能用唯一救回视频替代全表 |

来源：[RoboMME 固定记忆分析](../../artifacts/robomme/memory_ab_confirm_20260923/analysis.json)、[冻结双任务分析](../../artifacts/robomme/harness_transfer_gate_20260924/analysis.json)、[完整技术报告](../../paper/CARVE-VLA/ral_draft/TECHNICAL_REPORT.md)。以上链接在原仓库中可用；单独上传本目录时，README 的数字和说明仍可读，但外部源文件链接不会自动随视频上传。

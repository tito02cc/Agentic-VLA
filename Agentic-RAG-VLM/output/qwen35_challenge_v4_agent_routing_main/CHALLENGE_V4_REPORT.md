# 光华 Agentic RAG-VLM guanghua_agentic_rag_vlm_challenge_v4_model_selected_tools

部署：`qwen3.5-4b-bnb-nf4-local`；冻结 seeds：[301, 302, 303, 304, 305, 306, 307, 308, 309, 310]；共 100 个方法 rollout。
本实验报告真实多模态高层机制与技能代理结果，不声明接触动力学抓取成功率。

| 场景 | 条件 | 机制通过 | Wilson 95% CI | JSON有效 | 工具选择 | 机制分数 | 抓取准确 | 安全正确 | 变化检测 | 重规划 | 记忆 | 错误重规划 | 平均VLM调用 | P50延迟 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | C2_full | 10/10 | [0.72, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — | — | 2.60 | 25.30s |
| G1 | B0_vlm_only | 1/10 | [0.02, 0.40] | 1.00 | — | 0.10 | 0.45 | — | — | — | — | — | 1.00 | 12.33s |
| G2 | C2_full | 10/10 | [0.72, 1.00] | 1.00 | 1.00 | 1.00 | — | 1.00 | — | — | — | — | 2.00 | 12.61s |
| G2 | A_no_graph | 0/10 | [0.00, 0.28] | 1.00 | — | 0.00 | — | 0.90 | — | — | — | — | 1.00 | 12.02s |
| G3 | C2_full | 10/10 | [0.72, 1.00] | 1.00 | 1.00 | 1.00 | — | — | 1.00 | 1.00 | 1.00 | 0.00 | 3.50 | 60.88s |
| G3 | A_no_memory | 0/10 | [0.00, 0.28] | 1.00 | — | 0.00 | — | — | 1.00 | 1.00 | 0.00 | 0.00 | 1.30 | 50.46s |
| G3 | A_no_replan | 5/10 | [0.24, 0.76] | 1.00 | — | 0.50 | — | — | 0.80 | 0.70 | 0.90 | 0.00 | 1.80 | 53.15s |
| G4 | C2_full | 10/10 | [0.72, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 8.00 | 98.78s |
| G4 | C1_local | 0/10 | [0.00, 0.28] | 1.00 | — | 0.33 | 0.30 | 0.80 | 1.00 | 1.00 | 1.00 | 0.00 | 3.40 | 71.95s |
| G4 | B0_skill_only | 1/10 | [0.02, 0.40] | 1.00 | — | 0.40 | 0.50 | 0.20 | 0.70 | 0.70 | 1.00 | 0.00 | 0.00 | 0.00s |

## 配对统计

- G1 `C2_full` vs `B0_vlm_only`：b=9, c=0, McNemar exact p=0.00390625。
- G2 `C2_full` vs `A_no_graph`：b=10, c=0, McNemar exact p=0.00195312。
- G3 `C2_full` vs `A_no_memory`：b=10, c=0, McNemar exact p=0.00195312。
- G3 `C2_full` vs `A_no_replan`：b=5, c=0, McNemar exact p=0.0625。
- G4 `C2_full` vs `C1_local`：b=10, c=0, McNemar exact p=0.00195312。
- G4 `C2_full` vs `B0_skill_only`：b=9, c=0, McNemar exact p=0.00390625。

## 事件条件结果

| 场景 | 条件 | 事件 | 通过 | 平均机制分数 | False-replan |
|---|---|---|---:|---:|---:|
| G3 | C2_full | target_move | 3/3 | 1.00 | 0.00 |
| G3 | C2_full | no_change | 4/4 | 1.00 | 0.00 |
| G3 | C2_full | irrelevant_fragile_move | 3/3 | 1.00 | 0.00 |
| G3 | A_no_memory | target_move | 0/3 | 0.00 | 0.00 |
| G3 | A_no_memory | no_change | 0/4 | 0.00 | 0.00 |
| G3 | A_no_memory | irrelevant_fragile_move | 0/3 | 0.00 | 0.00 |
| G3 | A_no_replan | target_move | 0/3 | 0.00 | 0.00 |
| G3 | A_no_replan | no_change | 3/4 | 0.75 | 0.00 |
| G3 | A_no_replan | irrelevant_fragile_move | 2/3 | 0.67 | 0.00 |
| G4 | C2_full | target_move | 3/3 | 1.00 | 0.00 |
| G4 | C2_full | no_change | 4/4 | 1.00 | 0.00 |
| G4 | C2_full | irrelevant_fragile_move | 3/3 | 1.00 | 0.00 |
| G4 | C1_local | target_move | 0/3 | 0.33 | 0.00 |
| G4 | C1_local | no_change | 0/4 | 0.33 | 0.00 |
| G4 | C1_local | irrelevant_fragile_move | 0/3 | 0.33 | 0.00 |
| G4 | B0_skill_only | target_move | 0/3 | 0.22 | 0.00 |
| G4 | B0_skill_only | no_change | 1/4 | 0.50 | 0.00 |
| G4 | B0_skill_only | irrelevant_fragile_move | 0/3 | 0.44 | 0.00 |

事件分层用于区分真正的 target-move 恢复与 no-change / irrelevant-move 负对照，
避免只看聚合成功率而把错误重规划误判为恢复能力。

## G2 难度分层

| 条件 | 易碎物距离 | 通过 | 可执行 offset |
|---|---:|---:|---:|
| C2_full | 0.06 m | 2/2 | 1.00 |
| C2_full | 0.08 m | 3/3 | 1.00 |
| C2_full | 0.10 m | 3/3 | 1.00 |
| C2_full | 0.14 m | 2/2 | 1.00 |
| A_no_graph | 0.06 m | 0/2 | 0.00 |
| A_no_graph | 0.08 m | 0/3 | 0.00 |
| A_no_graph | 0.10 m | 0/3 | 0.00 |
| A_no_graph | 0.14 m | 0/2 | 0.00 |

## 调用与审计

- 方法 rollout：100；实际 VLM 调用：246；最终 endpoint 错误：0。
- Token：270953（prompt 237190，completion 33763）。
- Agent 工具执行覆盖：40 个 rollout；工具调用：60。
  - `protected_relation_to_motion_constraint_v1`：20 次。
  - `retrieval_to_dexterous_skill_router_v1`：20 次。
  - `role_aware_change_and_memory_verifier_v1`：20 次。
- `validation_receipt.json` 审计 run/CSV/公开事件完整性以及 evaluator-only 字段隔离。

## 设计升级

- G1 在冻结 seed 间改变物体几何与正确 synergy，而不是固定 red=power、blue=pinch。
- G2 只提供公开图关系和测量，不提供候选动作 offset。
- G3/G4 混合目标移动、无变化和无关易碎物移动，报告 false-replan。
- 所有 VLM 条件共享一次有界 schema repair，额外调用计入成本。
- Full Agent 先由独立 Qwen 路由回合显式选择工具，再将候选交给三个确定性执行契约：RAG→手型、场景图→运动约束、变化+记忆→恢复；实验编号不参与工具分派。
- 路由选择、原始候选、结构归一化、工具输入/输出和拒绝原因均写入 public trace；工具只读公开观测。
- G4 联合可供性、邻居关系、变化事件、质量、摩擦、位姿和指令改写。

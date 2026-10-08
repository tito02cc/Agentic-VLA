# RA-L 稿件实验盘点与主张边界

更新：2026-09-23。本文整理可追溯证据，不替代原始日志。目标是尽快确定能写什么、不能写什么，而不是把不同后端、不同方法版本的数字拼成一个“完整系统”成绩。现有 `root.tex` 是较长的旧稿；新短稿在 `ral_draft/`。

## 一句话结论

**已经有可追溯的闭环系统实验和独立推理优化实验，但尚未有足以支撑“当前冻结的 Agentic Harness 与 Optimize Runtime 在同一闭环中同时提升成功率、降低总成本”的确认性主结果。** 最适合先写的主线是“动作块边界的具身智能体决策与有质量约束的选择性规划”，把 VLA 内核优化作为独立部署实验；若保留二者联合收益为核心贡献，仍需同版本、同任务、同服务条件的配对证据。

## 1. 可追溯实验总表

| 证据 | 方法/模型与规模 | 实测结果 | 论文定位与边界 |
|---|---|---|---|
| RoboMME 八任务历史配对研究 | GroundSG PI0.5 + Qwen3-VL-4B；8/16 官方任务，每任务每方法 10 回合；Raw、Harness、Harness+Selective 各 80 回合，合计 240 个真实仿真 rollout | 成功 `23/80 -> 36/80 -> 42/80`；Harness 到 Selective 的 Planner 调用 `2001 -> 1188`（-40.6%），总 wall time `5178.7 -> 3723.6 s`（-28.1%） | **探索性系统结果**。原始 240 份 summary/视频存在，但 C3 合并了开发期间多个版本，不能把名义配对检验当作冻结方法的独立确认，也不能直接外推到当前 RoboDojo 代码。原始聚合：[summary](../../results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json)、[逐回合 CSV](../../results/robomme_b1_c2_c3_combined_80ep_20260831/episodes.csv)、[审计说明](../../docs/reports/RAL_CORE_EXPERIMENT_REPORT_20260831.md)。 |
| RoboMME 当前身份记忆开发配对 | 同一 JAX PI0.5/GroundSG 链路，VideoUnmaskSwap ep15–19，A 无记忆 / B 公开演示身份记忆 | `3/5 -> 5/5`，救回 2、伤害 0；精确双侧 McNemar `p=0.5` | **局部机制案例**，不是显著性证明或跨任务泛化。来源：[analysis](../../artifacts/robomme/unmaskswap_fixed_five_20260923/analysis.json)。 |
| RoboMME 当前选择性规划开发配对 | 同一服务内 B 每块规划 / C 选择性规划+块边界验证复用，VideoUnmaskSwap ep15–19 | B `3/5`、C `4/5`；双方都成功的 3 对 Planner 调用 `21 -> 14`，wall time `29.52 -> 23.62 s`（-20.0%） | **有条件的开发成本信号**。旧批次 B 为 `5/5`，新服务 B 为 `3/5`，说明跨服务基线不稳定；不能宣称冻结确认或总体加速。来源：[analysis](../../artifacts/robomme/unmaskswap_runtime_bc_20260923/analysis.json)。 |
| RoboMME 留出初态准入 | Swap ep20/21、两次服务；VideoUnmask 同样做 A/B 门控比较；合计 20 次 rollout | Swap A/B/C 均 `0/4`；VideoUnmask A/B 均 `4/4`，但 B 没有实际注入记忆；Swap 没有双方成功的 B/C 效率对 | **必须报告的负结果**。四次/条件是两个初态在两次服务中重复，不是四个独立种子；本阶段未通过预设扩样门槛。来源：[analysis](../../artifacts/robomme/service_block_pilot_20260923/analysis.json)、[停止规则](../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)。 |
| RoboMME 阶段回执开发门槛 | 同一PI0.5/GroundSG、VideoUnmaskSwap已用双目标ep15/19；默认关闭的chunk边界提示E，对照历史B | B、E均`2/2`；E在两例均转向第二目标，没有后续重复抓首目标；ep15 `324→325`步、Planner `21→21`；ep19 `280→280`步、Planner `18→18` | **仅开发安全检查**。与B非同服务配对，两例B原本成功；不能称成功率或效率增益。ep20旧失败轨迹的离线回放只证明会生成提示，没有新的闭环救回。来源：[运行日志与视频](../../artifacts/robomme/stage_receipt_dev_20260923/rollouts/)、[预登记与结论](../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)。 |
| RoboMME 固定新初态 B/F 准入 | 预登记Swap ep22--29，公开演示+SAM2身份记忆；B为记忆原方案，F为身份冲突核验+双目标阶段回执 | ep26记忆未准入；其余7对B/F各`5/7`成功，救回0、伤害0。F身份修正0次、阶段回执79次。5个双方成功且同服务的配对，Planner/VLA均`56→56`次、步数`871→871`，总wall `74.32→74.35`秒 | **明确负结果，F停止扩样**。ep27 F预算1300步仍`ongoing`、B官方`fail`；因脚本曾误判预算终态，ep27两臂跨服务，不用于耗时对比。其余6对同服务。不能称阶段提示或组合Harness有成功率/效率收益。来源：[逐例审计](../../artifacts/robomme/frozen_gate_20260923/analysis.json)、[原始回合](../../artifacts/robomme/frozen_gate_20260923/rollouts/)、[预登记/偏离说明](../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)。 |
| RoboMME 身份记忆固定门槛与独立确认 | 同一GroundSG PI0.5/Qwen3-VL服务、公开示教与初态配对；A无结构化记忆，B仅加SAM2身份记忆；开发门槛ep30--37与独立确认ep38--49 | 门槛A/B `2/8→4/8`、救回2/伤害0；确认集`5/12→9/12`、救回5/伤害1，双侧精确McNemar `p=0.21875`。确认集单目标`3/6→6/6`，双目标`2/6→3/6`。 | **同任务机制证据，尚非RAL主结论**。ep43有伤害，ep47 B救回却使用1075步/68次VLM与VLA。ep50--53越界预检未进入推理，不替补；该任务官方仅50个测试回合。不能外推跨任务Agentic提升或Optimize联合收益。来源：[门槛审计](../../artifacts/robomme/memory_ab_gate_20260923/analysis.json)、[确认集审计](../../artifacts/robomme/memory_ab_confirm_20260923/analysis.json)、[预登记](../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)。 |
| RoboMME 同链路B/C调度消融 | 相同JAX PI0.5、Qwen3-VL和身份记忆；B每块规划，C选择性规划+验证子目标复用；沿用ep38--49既知初态，新服务同次配对 | ep38--48严格11对：B `8/11`、C `7/11`，救回1/伤害2；双方成功6对Planner`45→29`、wall`66.84→52.13 s`。ep49 C因非法技能被安全拒绝，B在另一服务单独完成官方成功，不混入严格配对 | **质量门槛未过，C不作为默认模式**。调用与条件性耗时下降不能抵消任务回归；ep49是方法输出错误而非官方任务失败。B在不同服务的重复结果有波动，且B/C首块动作相同后观测仍会分叉。来源：[严格配对与异常审计](../../artifacts/robomme/memory_selective_bc_20260923/analysis.json)、[固定协议](../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)。 |
| RoboMME 跨任务身份记忆检验 | 当前JAX PI0.5/Qwen3-VL，同源公开示教SAM2记忆；`VideoUnmask`连续ep22--37，16对同服务A无记忆/B始终注入记忆 | 16/16准入；A `15/16`、B `13/16`，救回1/伤害3，精确McNemar `p=0.625`；单目标均`12/12`，双目标`3/4→1/4` | **明确负迁移，B不能全任务默认启用**。双方成功12对调用`134→100`、wall`152.74→121.41s`属条件性不同轨迹，不能抵消任务伤害；另有SAM2编译成本。现有motion gate的离线判别不是已验证的闭环收益。来源：[跨任务审计](../../artifacts/robomme/videounmask_memory_transfer_20260924/analysis.json)、[预登记及停止规则](../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)。 |
| RoboMME 冻结Raw/Harness双任务压力测试 | 相同JAX PI0.5/Qwen3-VL，一次服务、连续ep22--37各16对；RouteStick为历史正向任务，PickHighlight为历史回归风险任务；不注入身份记忆或启用已拒绝的选择性规划 | RouteStick Raw/Harness `0/16→1/16`；PickHighlight `3/16→3/16`；合并`3/32→4/32`，1救回/0伤害，精确McNemar `p=1.0`。双方成功3对wall`53.43→55.01s`、调用`58→60` | **当前冻结版未复现强正向增益，也未提速**。任务按历史机制有意选取，不是随机总体估计；RouteStick几乎全失败，不能把它解释为高层系统有效。来源：[逐对审计](../../artifacts/robomme/harness_transfer_gate_20260924/analysis.json)、[预登记与停止规则](../../docs/plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)。 |
| LIBERO-PRO 本地覆盖研究 | PyTorch PI0.5；40 task × 10 本地初态 × 3 方法，共 1200 回合 | Frozen `180/400`（45.0%）、固定恢复 `181/400`、Agentic `183/400`（45.75%）；Frozen 对 Agentic 为 4 次救回/1 次伤害，McNemar `p=0.375`；VLA 调用 -8.4%，但总 wall time **+16.5%** | **完整的局部 benchmark 对照及边界案例**，不是显著任务增益，更不是端到端加速；不是官方 50 初态排行榜。来源：[study summary](../../results/libero_pro_full_study_20260825/aggregate/study_summary.json)、[audit](../../results/libero_pro_full_study_20260825/aggregate/audit_report.json)。审计状态通过，1200 视频在库。 |
| PI0.5 内核效率 | RTX 4090；`pi05_libero_pytorch`；45 个配对录制观测，固定噪声、10 步动作 horizon，80 ms *单模型调用*目标 | 7-step eager BF16 P95 `282.43 ms`；2-step eager `151.35 ms`；2-step compile `66.06 ms`；2-step compile+SMVE `54.67 ms`。当前 INT8 路径 P95 `994.74 ms`，拒绝作为实时配置 | **独立的模型调用延迟消融**，不能称整机器人控制周期提速或当前 JAX RoboMME 链路的端到端加速。7→2 步也不是“无损”对照；45 次保真检查适用于后三个候选与相应参考，不等于跨步数闭环质量不变。来源：[efficiency summary](../../results/carve_efficiency_full_20260826/summary.json)及其 `vla/` 原始 JSON。 |
| Planner 量化与同驻 | Qwen3.5-4B BF16/NF4；30 条语义观测；一次 NF4 Planner+PI0.5 同驻集成 smoke | BF16 P95 `1896.55 ms`、显存 `8.46 GiB`；NF4 P95 `2909.79 ms`、显存 `3.08 GiB`；样本内语义决策一致。同驻 smoke 约 `11.22 GiB`，**未下发动作** | **低显存档位/集成可行性**，NF4 不加速，单次同驻不是闭环成功率。来源：[efficiency summary](../../results/carve_efficiency_full_20260826/summary.json)、[集成收据](../../results/carve_efficiency_full_20260826/coupling/qwen35-4b-nf4-pi05-smve-smoke/model_integration_receipt.json)。 |

注：历史 RoboMME 总表中 Raw→Harness 为 `+16.25` 个百分点，Raw→Full 为 `+23.75` 个百分点；汇总文件给出的名义精确 McNemar p 值分别为 `0.010622`、`0.000157`。由于方法选择/版本演进参与了这批数据形成，**不应将这些 p 值写成当前最终方法的预注册确认性检验**。Full 相对 Harness 的 `+7.5` 个百分点，名义 `p=0.146`，也不能说显著。

## 2. 论文可以主张什么

1. **架构与机制**：将高层语义规划、可信的任务/对象记忆、执行信号监测、动作块边界的纠错与安全停止，连接到可替换 Action Policy 接口。这里“可替换”是软件接口能力；目前实证仍主要是 PI0.5，不等于已验证任意 VLA/WAM。
2. **任务端证据**：历史八任务显示机制有潜力，尤其 VideoRepick、RouteStick、VideoUnmaskSwap；但也有 PickHighlight 回归、当前 Swap 留出初态全失败及 VideoUnmask 负迁移。应在主文用任务级表/失败分析展示异质性，而不是只给一个合并成功率。
3. **效率端证据**：历史闭环的选择性高层调用有成本下降趋势；独立 PyTorch PI0.5 内核有可重复的单调用 P95 下降。两种证据**不同链路**，不能合并成“同一个系统成功率提高且推理加速 5.2 倍”。
4. **质量约束**：接入轻量化/选择性执行之前，要在真实闭环中检查任务质量与误干预，不能把更早失败导致的短回合算作加速。INT8 和 INT8+SMVE 的失败是应保留的准入负例。

## 3. 不能直接写成结论的内容

- “当前冻结方法在 RoboMME/RoboDojo 上显著超过基线”：留出准入未通过；历史 C3 混开发版本。
- “Agentic 在 LIBERO-PRO 显著提高成功率”：只有 `+0.75` 个百分点，`p=0.375`，且端到端耗时增加。
- “Optimize Runtime 在当前 JAX 闭环降低 VLA P95 到 54.67 ms”：54.67 ms 来自**另一 PyTorch checkpoint/backend**的录制观测基准。
- “NF4 既减显存又提速”：它省显存但 P95 上升约 53.4%。
- “旧 Agentic+Optimize 联合配对恢复了 2/2 并提速”：旧 `results/carve_pi05_agentic_optimize_pair_20260719/` 原始目录现不在库中；生成汇总还在，但原始配对证据未恢复。仅能列为**待核验的历史记录**，不作主表。
- “自进化/在线微调、通用 WAM、真机硬实时”：现有表格均不支持。

## 4. 建议的短稿实验结构

| 主文位置 | 内容 | 数据来源 | 写作处理 |
|---|---|---|---|
| 主表 A：闭环任务 | RoboMME 8 任务逐任务 Raw / Harness / Full，附救回/伤害与总调用数 | 历史 80 对原始 CSV | 标为**开发与扩展的探索性评估**；明确 C3 多版本。若投稿前取得冻结确认，再让新数据成为主表 A，旧表转补充材料。 |
| 主表 B：部署开销 | 同一硬件的 V0–V4 VLA P50/P95/显存/准入判定；Planner BF16/NF4另列 | 2026-08-26 原始 profile | 标明 45 配对观测、80 ms 是单调用目标；说明 PyTorch/JAX 不可直接对照。 |
| 消融与失败 | 记忆 off/on 5 对，选择性 B/C 5 对，未通过的 S2；LIBERO-PRO 400 对补充对照 | 2026-09-23 原始 analysis、LIBERO-PRO audit | 同时写正反例与样本数；不从失败案例继续挑成功子集。 |
| 定性视频 | 历史 RoboMME 救回 1 组、回归 1 组，近期 Swap 开发救回与留出失败各 1 组 | 下方视频入口 | 图注写清任务、初态、方法版本、是否开发集、官方成功判定。 |

建议正文叙事：**Agentic 不应任意打断动作生成；它在可观察的动作块边界维护任务状态、调用记忆和语义工具，只在质量允许时减少高层推理。** 这比“堆叠多种 Agent 模块 + 量化/compile”更聚焦。但当前数据支持的是系统原型与部分机制，尚不能声称所有部分已在同一冻结版本证明综合优越。

## 5. 视频与复现入口

以下为原始 rollout，不是剪辑或封面；正负例都保留。

- 历史 RoboMME `VideoUnmaskSwap ep0`，Raw 失败：[Raw 视频](../../results/robomme_b1_raw_policy_20260831/VideoUnmaskSwap_ep0/VideoUnmaskSwap_ep0_vlm_groundsg.mp4)；C3 成功：[Full 视频](../../results/robomme_c3_unmask_swap_event_v6_20260831/VideoUnmaskSwap_ep0/VideoUnmaskSwap_ep0_vlm_groundsg.mp4)。**开发期 v6**，不代表当前冻结方法。
- 历史 `PickHighlight ep5`，Raw 成功：[Raw 视频](../../results/robomme_extension_selected_b1_40ep_20260831/PickHighlight_ep5/PickHighlight_ep5_vlm_groundsg.mp4)；C3 失败：[Full 视频](../../results/robomme_extension_selected_c3_40ep_20260831/PickHighlight_ep5/PickHighlight_ep5_vlm_groundsg.mp4)。用于展示误干预/负迁移。
- 当前 `VideoUnmaskSwap ep15` 等五对视频及逐回合 JSON：[固定五对目录](../../artifacts/robomme/unmaskswap_fixed_five_20260923/rollouts/)。选具体案例前必须以 `analysis.json` 和视频双核。
- 当前 S2 留出失败与第二任务不注入记忆：[service-block 目录](../../artifacts/robomme/service_block_pilot_20260923/)。不得只展示开发集成功片段。
- 阶段回执提示开发回合：[ep15 E视频](../../artifacts/robomme/stage_receipt_dev_20260923/rollouts/VideoUnmaskSwap_ep15_E/VideoUnmaskSwap_ep15_vlm_groundsg.mp4)、[ep19 E视频](../../artifacts/robomme/stage_receipt_dev_20260923/rollouts/VideoUnmaskSwap_ep19_E/VideoUnmaskSwap_ep19_vlm_groundsg.mp4)。均为旧B已经成功的初态，不是新救回。
- 固定新初态B/F准入：[ep27 B失败视频](../../artifacts/robomme/frozen_gate_20260923/rollouts/VideoUnmaskSwap_ep27_B/VideoUnmaskSwap_ep27_vlm_groundsg.mp4)、[ep27 F预算未成功视频](../../artifacts/robomme/frozen_gate_20260923/rollouts/VideoUnmaskSwap_ep27_F/VideoUnmaskSwap_ep27_vlm_groundsg.mp4)。这一对跨服务；展示时须附说明，不能称F救回。
- 记忆独立确认，双目标救回：[ep39 A失败](../../artifacts/robomme/memory_ab_confirm_20260923/rollouts/VideoUnmaskSwap_ep39_A/VideoUnmaskSwap_ep39_vlm_groundsg.mp4)、[ep39 B成功](../../artifacts/robomme/memory_ab_confirm_20260923/rollouts/VideoUnmaskSwap_ep39_B/VideoUnmaskSwap_ep39_vlm_groundsg.mp4)。A/B同初态、同服务，B只加公开示教身份记忆。
- 同一确认集的反例：[ep43 A成功](../../artifacts/robomme/memory_ab_confirm_20260923/rollouts/VideoUnmaskSwap_ep43_A/VideoUnmaskSwap_ep43_vlm_groundsg.mp4)、[ep43 B失败](../../artifacts/robomme/memory_ab_confirm_20260923/rollouts/VideoUnmaskSwap_ep43_B/VideoUnmaskSwap_ep43_vlm_groundsg.mp4)。展示两臂且说明记忆不保证动作成功。

复现顺序：先读取上述 aggregate/analysis，再按 `episodes.csv` 中绝对 `summary` 路径追到每个回合，最后看同目录的 `*_vlm_groundsg.mp4`。本次检查确认历史 80×3 份 summary 和 240 个 rollout 视频路径均存在；这只是文件完整性检查，不等于重新逐帧审核所有视频。

## 6. 投稿前唯一关键缺口

1. **冻结一版方法并做独立确认**：固定代码、模型摘要、任务/初态、调用预算与停止规则；至少覆盖一个记忆有预期作用的任务和一个负迁移风险任务。先比较成功/伤害，再比较双方都成功的整任务时间。当前 ep20/21 已是开发初态，不能重复当留出集。
2. **若主张“Agentic 与 Optimize 联合收益”**：在*同一* checkpoint、backend、任务和服务内，比较有无 Optimize 的完整闭环质量、Planner/VLA 调用、墙钟与显存；独立 PyTorch P95 表只能作补充。
3. **若来不及补实验**：收缩题目/摘要为具身智能体系统及质量约束推理的探索性研究；如实报告 8 任务历史结果及负例、独立内核优化，不写已验证的综合 SOTA 或“全链路实时”。稿件可先完成，但投稿强度和录用概率不能由现有数字保证。

这三点是收口门槛，不是开启新 benchmark、换模型、真机或无限调参的待办清单。旧稿 `root.tex`、`root.pdf` 均保持原状，待确定主张后再改正文。

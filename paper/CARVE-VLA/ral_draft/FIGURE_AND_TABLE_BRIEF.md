# BIND-VLA 图表制作 brief（离线写作版）

更新：2026-09-24。配套 [技术报告](TECHNICAL_REPORT.md)、[当前 PDF](main.pdf)、[图像出处](FIGURE_PROVENANCE.md)与 [视频索引](VIDEO_EVIDENCE_INDEX.md)。目标是把图做得更清楚、更像机器人系统论文，同时不让视觉包装改变实验含义。`CARVE-VLA` 是历史仓库路径；当前稿件的方法名是 **BIND-VLA**。

## 共同规范

- 先绘制可编辑矢量结构，再用真实仿真帧做少量 inset。最终文字、数字、箭头和图例应在 Illustrator/Inkscape/TikZ/PPT 中可编辑；图像生成模型只用于**布局草案**，不得生成伪装成实验录像的机器人画面或统计柱。
- 正文中的所有定量图由下列 JSON/CSV 重新绘制，不要让生图模型凭 prompt 画柱高、坐标轴或误差线。没有重复试验估计时不画置信区间。成功率需同时写分子/分母。
- 建议统一成冷色动作环、暖色语义监督、灰色证据/日志、绿色“已准入”、红色“拒绝/伤害”。颜色之外还须有文字/线型区分，保证黑白打印可读。
- 每张图的图注写清：模型 checkpoint、任务/初态、对照条件、是否同服务配对、成功由谁判定、是否开发/冻结，以及单调用或整任务时延。不要把不同后端的数放在同一条“系统加速”箭头上。
- 不在方法总览图中画 LIBERO/RoboDojo 任务得分；benchmark 只出现在实验图或视频图库。

## Fig. 1 系统总览：建议重绘而非复制旧图

**要让读者一眼看懂的因果链：**冻结动作策略生成 chunk -> 机器人执行一个合法前缀 -> 新观察与执行回执；Monitor 高频读执行风险，VLM 低频提出语义建议；Harness 只在完成 chunk 后的边界检查建议、任务作用域、技能白名单和预算；Optimize Runtime 在模型调用前选择已通过保真/资源/任务质量门槛的 profile。物理急停是旁路，不等同普通 VLM 建议。当前 [可编辑原图](figures/architecture_mechanism.tikz)是内容基线，不必拘泥于旧图的排版。

**可交给 imagegen 的英文布局 prompt（只做草案）：**

> Design a publication-quality robotics systems architecture figure, landscape 2-column IEEE proportion, clean white background. Show a central repeated closed-loop trajectory with a real observation thumbnail placeholder, a frozen vision-language-action policy, an action chunk timeline, a robot/environment execution stage, and a fresh observation returning to the policy. Above this, show two distinct oversight lanes: a small fast execution-risk monitor that emits evidence only, and a larger event-triggered VLM planner/critic that emits typed semantic proposals. Place an explicit boundary-permission gate between proposals and the *next* action chunk; an ordinary proposal must not interrupt the current chunk. Below the action policy, show an inference-profile admission lane with contract, replay fidelity, resource measurement, task-quality gate, accepted profile, and fallback. Use restrained teal/charcoal for physical execution, coral for semantic planning, green for admitted, red for rejected. Add a compact bottom two-rate timeline: many control steps, fewer action chunks, rare VLM calls. Leave all text as editable placeholder regions; no performance numbers, no fake experimental imagery, no decorative circuit art, no generic box-only flowchart.

**手工替换的准确标签：**`ExecutionRiskMonitor: evidence, not success`；`VLM Critic/Planner: semantic proposal`；`Boundary gate: freshness / allowed skill / task scope / budget`；`Frozen action policy (VLA)`；`Profile gate: contract -> replay -> resources -> closed-loop quality`。若图面空间不足，缩短标签而非删掉边界门。不要把 HAA-RAG/Scene Graph 画成本稿所有回合的必经模块。模型接口理论上可扩展至其他 action policy，但本文实证是 PI0.5 家族。

## Fig. 2 配对救回与伤害：用程序绘图

**数据：**固定 RoboMME VideoUnmaskSwap 身份记忆 ep38--49，A `5/12`、B `9/12`，`rescue=5, harm=1`，精确双侧 McNemar `p=0.21875`；VideoUnmask 迁移 ep22--37，A `15/16`、B `13/16`，`rescue=1, harm=3`，`p=0.625`；冻结 RouteStick+PickHighlight 32 对，Raw `3/32`、Harness `4/32`，`rescue=1, harm=0`，`p=1.0`。后一组按 RouteStick/PickHighlight 各 16 对选定，不是随机 benchmark 平均。

**图形建议：**三列小倍图，每列左边绿向右条为 rescue、右边红向左条为 harm，旁边写 A/B 分数与 N。标出“same task memory / transferred memory / frozen harness”条件，不要把三列合并成一条总成功率曲线。现有 [TikZ 图](figures/paired_outcomes.tikz)可作为重绘数据草稿；真值回查 [固定记忆 JSON](../../../artifacts/robomme/memory_ab_confirm_20260923/analysis.json)、[迁移 JSON](../../../artifacts/robomme/videounmask_memory_transfer_20260924/analysis.json)、[冻结双任务 JSON](../../../artifacts/robomme/harness_transfer_gate_20260924/analysis.json)。

**可交给绘图/排版助手的 prompt：**

> Create an editable IEEE-column scientific chart specification, not invented data. Three horizontal paired-outcome panels with a zero line: positive rescue counts extend right in green, harm counts extend left in red. Dataset labels: Swap memory N=12 (5 rescue, 1 harm; 5/12 to 9/12), Unmask transfer N=16 (1 rescue, 3 harm; 15/16 to 13/16), frozen harness N=32 (1 rescue, 0 harm; 3/32 to 4/32). Include N and condition labels; no confidence intervals, no SOTA ranking, no inference of statistical significance. Export vector PDF/SVG with editable text.

## Fig. 3 身份记忆的实际仿真案例

使用 [公开示教帧](figures/ep39_demo.png)、[无记忆 A 终局](figures/ep39_no_memory_final.png)和 [有记忆 B 终局](figures/ep39_memory_final.png)。这三张是实验素材，**不需要 imagegen 重造**。图形可以设计为上方共同示教、下方左右 A/B 轨迹终帧；用同色对象标注帮助观众定位，但不要在原始画面涂改成不存在的抓取。A/B 终帧不同时刻，不画成同一时间的像素级反事实。

配套视频：`VideoUnmaskSwap_ep39_A/B`；同任务的反例 `ep43_A/B` 也随离线资料包提供。图注建议：`In a preselected RoboMME VideoUnmaskSwap pair, the memory arm succeeds while the no-memory arm fails. Terminal frames are from separate rollouts; this example does not establish cross-task generalization.` 末次点坐标、任务成功判定和来源见 [技术报告 5.2](TECHNICAL_REPORT.md)。

## Fig. 4 推理 profile：单模型调用，不是机器人闭环

从 [效率原始汇总](../../../results/carve_efficiency_full_20260826/summary.json)程序生成 P95 散点/条形，必要时在同图下方用小表放 VRAM。45 条配对录制观测、RTX 4090、PyTorch PI0.5 LIBERO checkpoint、固定噪声与 horizon 10：

| 配置 | P95 ms | GPU GiB | 图中角色 |
| --- | ---: | ---: | --- |
| V0 BF16 eager / 7 flow steps | 282.43 | 7.12 | 参考 |
| V1 BF16 eager / 2 steps | 151.35 | 7.12 | 少步的变化 |
| V2 BF16 compile / 2 steps | 66.06 | 6.98 | 编译候选 |
| V3 BF16 compile + SMVE / 2 steps | 54.67 | 6.98 | 此后端单调用准入 |
| V4 late-language INT8 / 2 steps | 994.74 | 6.36 | 速度门槛拒绝 |

80 ms 水平线仅是**模型调用 profile 目标**。V4 可画断轴或放右侧拒绝表，不得截断后假装其延迟接近 V3。V0->V3 同时改了 flow 步数、编译和 SMVE；不能标注“SMVE 带来 5.17x”。同为 2 步的 V2->V3 才是较接近 SMVE 的比较。Qwen3.5-4B Planner 的 30 条录制输入 BF16/NF4 为 `8.46->3.08 GiB`，P95 `1896.55->2909.79 ms`；应放小型独立表或补充图，不与 VLA P95 混成一个模型曲线。现有 [TikZ 图](figures/runtime_profiles.tikz)是版本基线。

## 表格与可选 RoboDojo 任务图库

主文推荐四表：八任务开发汇总及逐任务异质性、固定记忆和迁移、LIBERO-PRO 本地 400 配对、推理 profile。表头须写 `exploratory` / `preselected confirmation` / `frozen stress` / `independent model calls`。将负面条件留在表里：记忆迁移 `15/16->13/16`；冻结 Harness `3/32->4/32`；LIBERO 墙钟 `+16.5%`；VLM NF4 虽省显存却更慢。不要为美观删列。

RoboDojo **图库**可以展示搭塔、整理桌面、语言分类、叠碗、投瓶、收纳电脑/耳机的实际视频静帧，但图注必须逐张标注 checkpoint 和结局。搭塔是历史 PI-v3 的受控故障与 Planner 回放，当前 PI0.5 整理桌面和分类均未成功；叠碗是原生 PI0.5 成功，投瓶 C3 没有接受语义干预。图库仅作任务范围/机制演示，不可替代同模型自然任务配对实验。不要使用已审计为提前退出假成功的整理桌面片段；不要把 RoboDojo 官方任务网页视频写成我们运行的实验录像。完整标签见离线包 `videos/VIDEO_GUIDE.md`。

## 交付前检查

1. 与 `main.tex` 当前 Fig. 1--4 及表格对应；若换图，更新 `FIGURE_PROVENANCE.md`、图注和 PDF。
2. 查源 JSON 的分子/分母、episode、模型后端；图中比例和标题不超出其证据等级。
3. 缩到 IEEE 双栏打印比例检查正文和标签可读；不要把重要文字烘焙进低分辨率 AI 位图。
4. 对视频截图注明真实来源与终局；概念示意和实际仿真结果分别标识。

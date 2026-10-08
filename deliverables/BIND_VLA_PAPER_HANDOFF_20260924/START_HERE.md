# BIND-VLA 离线论文写作资料包

快照日期：2026-09-24。解压后从本文件开始，不需要原电脑或 GPU 即可阅读底稿、编辑 LaTeX、核查关键聚合结果与播放代表视频。此包是**写作/审稿资料快照**，不是完整可执行仿真环境，也不包含全部 4,512 个源视频。仓库路径历史名称 `CARVE-VLA` 与当前论文方法名 **BIND-VLA** 不同，不要为了统一命名批量改旧数据路径。

## 先看这四份

1. [完整技术报告](paper/CARVE-VLA/ral_draft/TECHNICAL_REPORT.md)：当前方法、实验协议、正负结果、代码接口和写作底稿。比六页稿更完整；不确定的说法优先回查所列 JSON/CSV。
2. [当前六页论文 PDF](paper/CARVE-VLA/ral_draft/main.pdf)和 [LaTeX 源文件](paper/CARVE-VLA/ral_draft/main.tex)：目前可编辑稿，不是已投稿定稿。`references.bib`、`ieeeconf.cls` 和**正在使用的**图源均在同目录。
3. [图表与生图 brief](paper/CARVE-VLA/ral_draft/FIGURE_AND_TABLE_BRIEF.md)：Fig. 1 的 imagegen 布局 prompt、Fig. 2/4 的精确绘图数据、Fig. 3 的真实帧来源与表格设计要求。
4. [视频说明](videos/VIDEO_GUIDE.md)：哪些片段是配对救回、伤害、受控故障、原生失败；可直接播放 `videos/*.mp4`。原始路径和 SHA-256 见 [视频清单](videos/VIDEO_MANIFEST.tsv)。

## 主要数字与证据边界

| 证据 | 当前观察 | 严格读法 |
| --- | --- | --- |
| RoboMME 八任务开发汇总 | B1/C2/C3 `23/80, 36/80, 42/80` | C3 含开发版本，非冻结最终方法的确认性提升 |
| RoboMME Swap 固定身份记忆 | `5/12 -> 9/12`，5 rescue/1 harm，`p=0.21875` | 同一任务的方向性证据；小样本、不显著 |
| RoboMME Unmask 记忆迁移 | `15/16 -> 13/16`，1 rescue/3 harm | 跨任务默认开记忆有害 |
| RoboMME 冻结 Raw/Harness 两任务 | `3/32 -> 4/32`，1 rescue/0 harm，`p=1.0` | 不支持广泛稳定增益主张 |
| LIBERO/LIBERO-PRO 本地 400 配对 | `180/400 -> 183/400`；VLA 调用 `-8.4%`，wall `+16.5%` | Agentic 增益小且总时间增加；不是官方 50 初态榜单 |
| PI0.5 PyTorch 模型调用 | P95 `282.43 -> 54.67 ms`，45 录制输入 | 七步 eager 对两步 compile+SMVE，变化非单因子；**不是 RoboMME/JAX 整链路加速** |
| RoboDojo 搭塔受控故障 | 历史 PI-v3，C1 `0/3`、C3 `3/3`，`p=0.25` | C3 回放已审计 Planner 决策且切换 profile；不是当前 PI0.5 自然故障/在线 VLM 结果 |
| RoboDojo 当前 PI0.5 整理桌面 | 原生和受限 Agent 各 `75/100`、官方 `success=false` | 两次并非严格配对；不能写成功救回 |

核心定位：冻结 action policy 外围的**边界授权语义监督**与**质量约束推理配置**。Monitor 给执行风险证据，VLM 提语义建议，Harness 在动作块边界授权，VLA 产生动作。可替换 policy 是接口能力；当前没有多模型泛化实证。当前**尚无同一 RoboMME/JAX 闭环中的 Harness+Optimize 联合“更准且更快”证明**，也没有 RoboDojo 官方 PI0.5 自然任务的确定性 Agent 增益。润色和画图不能擦掉这些限定。

## 包中结构

- `paper/CARVE-VLA/ral_draft/`：六页稿、完整技术报告、图表 brief、引用、源图、出处、实验升级/视频索引。只带活跃 Fig. 1--4 素材；旧 AI 概念图未带入。
- `paper/CARVE-VLA/RAL_EXPERIMENT_INVENTORY_20260923.md`：历史任务/结果盘点。
- `docs/reports/`：原长报告与 RA-L 核心实验报告。长报告是研发记录，出现旧口径时以当前技术报告和原始结果为准。
- `results/`、`artifacts/`：当前论文引用的关键聚合 JSON/CSV 和少量已审计结果；不是所有逐步日志/环境文件。
- `agentic_vla/`、`scripts/`、`tests/`：与方法相应的代码/测试快照，供论文技术描述对照。它们不是可脱离原环境运行的完整仓库。
- `paper/Agentic Policy/` 和 `paper/Agentic-RAG-VLM/`：少数参考 PDF。前者供方法写作对照，后者是已有工作的背景；两者的图和数字**不能**冒充本项目数据。
- `videos/`：RoboMME、LIBERO-PRO 和 RoboDojo 有解释标签的代表片段。`VIDEO_CATALOG.tsv` 是原机器全量发现索引，未复制文件的路径在新机器不可直接打开。
- `PACKAGE_SHA256.txt`：标准 `sha256sum` 格式，覆盖资料包内其他文件；`zip` 文件独立完成完整性检查。

## 离线写作顺序

1. 读技术报告第 0、2、4、5、6、7 节，再看 PDF，确保方法、模型/后端和数字没有串线。
2. 先确定论文的**一条**中心句、两项方法贡献及一项审慎的系统评估贡献；不要继续堆模块名。根据图表 brief 重画 Fig. 1，图 2/4 用 JSON 数字程序绘制，图 3 只用真实帧。
3. 修改 `main.tex` 时按 citation key 核对 `references.bib`；本包保留当前参考，不保证截至提交日的所有相关工作已覆盖。引用真实 PDF/官方页面，不造条目。
4. 每次重写摘要、贡献和结论后，对照上表检查不合理的跨后端或跨任务因果声称。实验不足可以在正文中简洁、准确地说，不能借润色写成未测过的性能。
5. 需要本机无法核查的新数值、完整逐回合日志或重新运行仿真时，等回到原电脑；此包不授权补造缺失实验。

在另一台电脑上解压后，可运行 `sha256sum -c PACKAGE_SHA256.txt` 验证资料文件；当前 `main.pdf` 已随包提供，不必在新电脑先装 LaTeX。若要重编译，在 `paper/CARVE-VLA/ral_draft/` 下用 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`，需本地 TeX Live 和 BibTeX。

## 给下一会话的简要上下文

“我正在修改 BIND-VLA 的 RA-L 论文。请先读 `START_HERE.md`、`paper/CARVE-VLA/ral_draft/TECHNICAL_REPORT.md`、`FIGURE_AND_TABLE_BRIEF.md` 与 `main.tex`。这几天只做写作、图表和引用核查，不启动仿真。以包中 JSON/CSV 为数据真值，保留负迁移、冻结测试和分后端时延边界；不要把历史 RoboDojo PI-v3 受控故障当当前 PI0.5 自然任务增益。请提出并逐项落实对论文结构、语言、图注和真实引用的改进。”

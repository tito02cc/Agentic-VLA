# Agentic RAG-VLM——简历项目包

> 状态提示（2026-09-10）：本文保留了早期岗位化文案，其中人形机器人、4 组旧演示和 40 项测试
> 等工程数字已不是当前主展示口径。方法贡献文案仍可取用；实验事实请以
> `../HANDOFF_PACKAGE_INDEX.md`、`../KIRO_HANDOFF.md` 和 `CURRENT_EXPERIMENT_PROGRESS.md` 为准。

> 目标岗位：具身智能 / VLA 算法工程师、AI Agent / 多模态应用工程师  
> 生成原则：ASu 的“背景目标—个人边界—关键动作—系统能力—结果证据”链路，结合
> Project Resume Writer 的证据账本、可核验量化和反夸大规则。

## 一、项目定位

**推荐项目名**  
Agentic RAG-VLM：面向人形机器人操作的检索增强多模态智能体

**一句话概述**  
面向无需重新训练 VLA 的机器人操作场景，构建融合 Qwen3.5-VL、HAA-RAG、场景图、
任务记忆与有界重规划的 Agentic 执行框架，并在人形机器人的 MuJoCo 环境中
完成可审计的多场景仿真验证。

**技术栈**  
Python、MuJoCo、URDF/MJCF、Qwen3.5-VL、RGB-D、RAG、Scene Graph、Episodic Memory、
ReAct/状态机、机械臂逆运动学、pytest、FFmpeg/HyperFrames

**论文背书**  
方法成果已被 IROS 录用（用户已确认；正式投递前补论文题目、作者顺序、年份和公开链接）。

## 二、可直接投递版：最终 4 条

> 以下版本假设你实际承担了框架设计、人形机器人仿真适配和实验实现。若属于多人共同完成，
> 请按真实分工把“设计并实现/构建”调整为“参与设计并实现/负责其中……模块”。

**复旦大学智能机器人研究院｜科研项目**  
2025.07–2026.03

**论文成果：** *Agentic RAG-VLM: Affordance-Aware Retrieval-Augmented Generation with Self-Reflective Planning for Robotic Grasping*，IROS 2026 第一作者，已录用

- **项目简介：** 面向复杂场景机器人抓取中的经验误检索、空间关系缺失及失败恢复困难，构建融合可供性感知检索、场景图推理与自反思规划的多模态操作智能体；论文在 12 项抓取任务上取得 78.3% 的总体成功率，纯 VLM 方案为 25.0%。
- **HAA-RAG：** 提出层次化可供性感知检索方法，为操作经验标注操作类型、材质、脆弱性和可抓区域，通过“类别筛选—物理属性匹配—视觉特征重排”检索适合当前对象的抓取策略。
- **场景图与自反思规划：** 构建物体空间关系图，将容器内容、邻近易碎物、支撑关系和遮挡关系转换为接近方向、抓取力度与操作顺序约束；基于 ReAct 与任务记忆实现 14 类失败诊断及三级恢复，可依次进行参数调整、抓取方式切换和重新规划。
- **系统验证：** 完成人形机器人操作环境、RGB-D 感知、机械臂逆运动学控制与安全检查的一体化集成，搭建 100 组 Agent 决策对比实验和 4 组连续机器人任务演示；完整方案能够根据物体特性选择抓取策略、避开受保护物体，并在目标变化后重新规划未完成任务。

## 三、具身智能 / VLA 算法岗版本

**推荐保留 4 条：**

- 构建面向人形机器人的任务级 Agentic 控制框架，以公开 RGB-D 观测驱动可供性检索、场景图安全约束、任务记忆和在线重规划，在不重新训练 VLA 的条件下完成多阶段桌面操作验证。
- 将人形机器人 URDF 与桌子/物体资产迁移至 MuJoCo，完善机械臂、手部关节、相机与执行器配置，并实现多起点完整朝向 IK、抓取预成形及桌面/关节限位门禁。
- 设计目标移动、无变化与无关物体移动三类扰动，Full Agent 在 G3 的三种事件切片达到 3/3、4/4、3/3 且 false-replan 为 0；无重规划消融仅通过 5/10，验证变化归因与恢复策略的必要性。
- 建立仿真准入体系，分别验证公开感知/预抓取、运动执行和接触动力学；对 G0-B 接触数值失稳给出 `NOT_ADMITTED` 审计结论，避免将运动学技能代理误报为真实抓取成功率。

## 四、AI Agent / 多模态应用岗版本

**推荐保留 4 条：**

- 设计多模态 Agent 运行时，将感知、规划、工具执行、校验、监测和重规划建模为显式状态机，以 episodic memory 保留已完成子任务，并通过 budgeted L3 recovery 处理未完成目标变化。
- 构建基于公开视觉几何的 HAA-RAG 检索与 3 类工具合约，由 Qwen3.5-VL 在独立路由回合选择所需工具，再将结构化候选转换为可执行手型、运动约束和恢复决策；原始选择、修正字段和最终决策均写入 public trace。
- 搭建防数据泄漏的评测与审计链路：Agent 仅访问 RGB-D、场景关系和公开位移，evaluator truth 独立存储；100 个运行目录、260 个公开事件与 60 次工具调用均通过自动校验。
- 完成 246 次真实本地 Qwen3.5-4B 多模态调用的配对消融，Full Agent 在 G1–G4 均为 10/10，且工具选择准确率均为 100%；VLM-only、no-graph、no-memory、no-replan、local 与 skill-only 对照呈现可解释性能差异。

## 五、ASu 风格 HR 开场白

### 80–160 字短版

复旦大学具身智能方向学生，关注 VLA、多模态 Agent 与机器人操作。围绕已被 IROS 录用的
Agentic RAG-VLM 方法，完成人形机器人的 MuJoCo 适配与 100 组配对评测，打通
Qwen3.5-VL、RAG、场景图、任务记忆和失败恢复，并保留完整运行与审计证据。希望应聘
[岗位名称]，[到岗时间/城市待补]，期待进一步交流。

### 技术面 30 秒介绍

这个项目解决的是“机器人不重新训练 VLA，如何利用多模态模型完成可恢复、可审计的长程任务”。
我把 Qwen 的高层候选与三个确定性 Agent 工具结合：RAG 决定灵巧手技能，场景图关系生成
安全运动约束，变化检测与 memory 决定是否重规划。随后把人形机器人迁移到
MuJoCo，在 10 个冻结种子上完成 100 个配对 rollout。Full Agent 四类任务都是 10/10，
同时保留强消融、负对照和 G0-B 接触抓取未准入的真实边界。

## 六、面试可追问点

### 1. 为什么不是“目标检测 + IK”就够了？

在固定几何和简单布局中，强视觉+IK 基线确实可以完成任务。差异出现在三个方面：物体几何
变化要求从检索证据选择手型；黄色 Protected Glass 的语义角色必须转成可执行路径约束；
目标或无关物体变化后，需要结合 pending-target memory 判断是否重规划。G4 将三者联合随机化，
Full 为 10/10，local 为 0/10，固定 skill 为 1/10。

### 2. VLM 到底做了什么？工具是否把答案写死？

VLM先读取图像、任务接口和公开上下文，在独立回合选择所需工具，再生成结构化高层候选。确定性工具不读取 evaluator truth，只使用
公开几何、RAG 排名、场景图关系、位移阈值和任务 memory，将候选约束到可执行决策。原始输出、
修复次数、工具输入/输出和执行结果全部留档；消融条件不会获得 Full Agent 的工具。

### 3. RAG 的价值如何验证？

G1 在不同种子间改变物体几何及正确 synergy，避免按颜色固定映射。Full Agent 根据公开视觉
几何检索 affordance card 并执行对应手型，10/10；VLM-only 为 1/10。应强调这是冻结主集结果，
10/10 的 Wilson 95% CI 为 [0.72, 1.00]，不外推为任意布局 100%。

### 4. 失败纠正是否人为制造？

L1/L2 是受控 verifier fault injection，用于隔离状态机分支；L3 使用 MuJoCo 中真实目标位移。
正式 G3 同时设置 target-move、no-change 和 irrelevant-move，避免“总是重规划”投机。Full 的
false-replan 为 0，无重规划版本在无需纠正的负对照上仍能部分成功。

### 5. 为什么没有宣称抓取成功？

连续演示中的运输是公开标记的运动学/contact-assisted 技能代理。G0-B 对顶抓、侧捏、柔顺控制、
小 timestep 和不同 solver 做过审计；无接触控制稳定，但手物接触触发数值失稳，因此结论为
`NOT_ADMITTED`。这能体现实验诚信，也明确下一阶段需要重建碰撞、惯量和接触参数。

## 七、证据账本

| 可写能力/数字 | 状态 | 证据 |
|---|---|---|
| 六阶段 Agentic 状态机 | code-verifiable | `agentic_rag_vlm/runtime.py` |
| HAA-RAG、场景图、memory、recovery | code-verifiable | `agentic_rag_vlm/pipeline.py`、`memory.py`、`recovery.py` |
| 3 类 Agent 工具 | confirmed | `scripts/run_qwen_vlm_challenge.py`、`validate_agent_tool_contracts.py` |
| 100 rollout、246 VLM 调用 | confirmed | `output/qwen35_challenge_v4_agent_routing_main/challenge_results.csv` |
| G1–G4 Full 10/10 | confirmed | `output/qwen35_challenge_v4_agent_routing_main/CHALLENGE_V4_REPORT.md` |
| 260 public event、60 tool call 审计 | confirmed | 两个 `validation_receipt.json` |
| 41 项测试 | confirmed | `pytest -q` 与 `tests/` |
| 67.0 秒、1080p/30fps 演示 | confirmed | `output/defense_suite/*v9.validation.json` |
| G0-B 未准入 | confirmed | `output/g0b_contact_admission_audit_v1/receipt.json` |
| IROS 录用 | user-confirmed | 待补录用邮件、论文首页或公开链接 |
| 个人负责范围 | unknown | 投递前需由本人确认 |

全部数字可以运行以下命令重新核验：

```bash
python scripts/validate_resume_evidence.py
```

## 八、ATS 关键词

具身智能、Vision-Language-Action、Multimodal Agent、Qwen-VL、Retrieval-Augmented Generation、
Tool Calling、Scene Graph、Episodic Memory、Replanning、Robot Manipulation、Dexterous Hand、
MuJoCo、URDF、RGB-D Perception、Inverse Kinematics、Ablation Study、Evaluation Harness、
Public/Private Data Isolation、Failure Recovery

## 九、投递前必须确认

1. 项目起止时间、你的角色和个人负责模块；
2. IROS 论文题目、年份、作者顺序与链接；
3. 项目仓库/视频是否允许公开，以及简历中使用哪个链接；
4. 是否做过实机、真实接触抓取或 sim-to-real——目前证据均不支持这些表述；
5. 目标 JD。拿到 JD 后，应从上面两个版本中重新排序关键词和 bullet，而不是继续增加条目。

## 十、不建议写入简历的说法

- “实现端到端 VLA 控制”——当前高层 VLM 与低层确定性技能是分层结构；
- “机器人抓取成功率 100%”——G0-B 接触动力学尚未准入；
- “完成 sim-to-real/实机部署”——尚无证据；
- “Full Agent 泛化成功率 100%”——10/10 只对应冻结主种子；
- “独立主导/从 0 到 1”——需先确认个人贡献边界。

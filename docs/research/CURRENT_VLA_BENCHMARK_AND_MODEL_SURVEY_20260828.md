# 2026 VLA Benchmark 与模型调研

初稿：2026-08-28；复核：2026-09-09；路线再评估：2026-09-22。

补充调研入口：[近期工作、Benchmark 与 RA-L 验证路线](AGENTIC_VLA_RAL_REFERENCE_AND_VALIDATION_PLAN.md)。
包含新核查的 RoboMemArena（非 RoboMME）、BATON 与 RA-L 具体实验设置。
RoboMemArena 尚未通过权重与部署准入，不改变下述有限候选优先级。

最新调研建议以第 0 节为准，但尚未自动变更实验执行计划或启动新 benchmark。
第 2、7、8 节属于 9 月 9 日的历史判断。第 3--6 节保留早期资源目录与历史判断，
其中的优先级、模型推荐及榜单数值不是当前部署状态；本次复核不意味着下载、
切换模型或启动新基准。现有实验状态以 `../status/ROBODOJO_CARVE_PAIRED_RESULTS.md` 为准。

## 0. 2026-09-22 再评估：不再把整理桌面作为唯一突破口

本节不是只换 benchmark 的方案。先依据
`CARVE_AGENTIC_POLICY_REFERENCE_SYNTHESIS_20260824.md` 第 0 节核对框架的技能
可控性、记忆与核验机制，再决定候选任务。平台推荐不能替代机制有效性验证。

### 0.1 现有证据与问题定位

本节是路线建议，不是新增实验结果。交叉核对依据为
`../reports/AGENTIC_VLA_TECHNICAL_REPORT.md` 第 9、10 节及 RoboDojo 最新诊断记录。

- RoboDojo 当前完整整理桌面回合仍为官方得分 75/100、success=false；分数不是成功率。
  近期候选识别和关系判定诊断不能替代完整回合，也不能证明 Agent 带来增益。
- 当前瓶颈混合了视觉关系误判、动作接口的纠错能力不足和长程误差累积。
  更换任务只能帮助分离瓶颈，不能保证修复框架或自动获得正收益。
- LIBERO-10/PRO 历史 400 回合：180/400 到 183/400，不能作为强成功率提升证据。
- RoboMME 历史 80 回合/条件：B1/C2/C3 为 23/80、36/80、42/80；B1 已包含
  Planner，并非裸 VLA。后四个扩展任务只有 18/40、20/40、20/40，因此不能仅引用
  合并表的较大提升宣称泛化成立。C3/C2 调度成本下降值得单独复核。

### 0.2 候选与优先级

| 选择 | 可验证的机制 | 权重和接入条件 | 建议 |
|---|---|---|---|
| RoboMME | 历史信息利用、grounded 子目标执行、Planner 调度 | 官方发布已微调策略；本地已有 GroundSG PI0.5 和仿真链路 | 优先复核已有协议，做独立记忆与调度消融，避免重新调开发任务 |
| VLA-Arena | 长程组合、任务流程、分级难度与安全代价 | 官方发布专用 pi05-vla-arena-finetuned；尚未在本机验证其配置、技能和共驻显存 | 若新增一个平台，优先作小规模接入筛查，不直接跑全榜 |
| LIBERO-Plus | 视觉、语言、布局等扰动下的鲁棒性 | 可基于匹配 LIBERO 的权重评测；不同于已做的 LIBERO-PRO | 低成本补充候选，不同时铺开；不能直接证明记忆或实时性 |
| LIBERO-Mem | 单帧不足、需要历史信息的操作 | 有代码和数据；已有第三方 pi05_libero 评测接入示例，但不等于匹配技能已可靠 | 观察候选，优先级低于现成专用权重的平台 |
| RoboDojo 换任务 | 在同一平台分离语义决策、记忆与精细操作瓶颈 | 现有 RoboDojo pi05 不等于能执行任意新子指令；须测试指令可控性 | 保留为压力测试/展示，不继续无限优化单个整理桌面布局 |

官方来源：

- [RoboMME 策略与评测](https://github.com/RoboMME/robomme_policy_learning)。
- [VLA-Arena 项目](https://vla-arena.github.io/)与
  [官方 PI0.5 权重](https://huggingface.co/VLA-Arena/pi05-vla-arena-finetuned)。
  项目提供 170 个任务、L0-L2 分级，覆盖安全、干扰、泛化和长程操作。
  官方协议在 L0 微调、在更高难度评测；使用已发布权重不需要我们先重新训练，
  但仍需核对权重所覆盖的 suite、输入输出与归一化。
- [LIBERO-Plus 官方仓库](https://github.com/sylvestf/LIBERO-plus)。
- [LIBERO-Mem 仓库](https://github.com/libero-mem/libero-mem)与
  [VoLoAgent 的跨基准接入示例](https://github.com/NVlabs/VoLoAgent/blob/main/examples/README.md)。
  示例可以支持接入可行性判断，不能替代官方基线复现或成为成功率保证。

### 0.3 RoboDojo 需要先排除的版本问题

[官方仓库](https://github.com/robodojo-benchmark/RoboDojo)在 2026-09-16 至 17 日
公布 get_obs 单帧差异修复，以及与 XPolicyLab bb9a0b5 配套的 RGB 字节流通道修复。
本地 third_party/robodojo_official 的 HEAD 是 8 月 20 日的
2184bf8844ea9d205382c4aefa3a694311418251，且存在本地接入修改。
这只能说明需要逐项核对实际调用路径，不能推断本项目受影响或据此解释全部失败。
官方也明确说明单帧问题修复后的多模型结果基本不变。不要直接覆盖本地环境；若修复
确实改变输入，应对 baseline 和 Agent 同时应用，并将新旧结果分开记录。

### 0.4 推荐执行顺序与停止条件

1. 先审计 RoboMME 已有结果、输入权限、模型版本和数据来源，恢复同配置基线。
   固定 checkpoint、观测、动作/时间预算，只改变记忆或 Planner 调度中的一个因素。
2. 如需新平台，仅接入 VLA-Arena 的一个明确任务族。先核对 L0 基本技能与语言
   可控性，再检验 L1/L2 的组合错误；不依据 Agent 哪些种子赢来选择测试集。
3. RoboDojo 暂停无终点的整理桌面补丁迭代。若继续筛查任务，可先检查按语言分类
   的基本抓放和指令响应；模仿分拣序列需要可靠示范记忆与可控抓放后再评估。
   task 名称和画面不能证明这些前提成立。
4. 接入筛查只回答环境/权重/动作是否匹配，不作为论文统计结论。筛查可用少量
   固定开发种子；进入正式比较后另用未调参的初始状态并保留所有失败。
5. 若底层技能或子目标控制不足，记录不适配并停止该候选，不为让 Agent 获胜增加
   特权相机、模拟器真值或只给 Agent 的动作预算。若技能充分但 Agent 无收益，
   应检验机制本身，不能靠不断更换 benchmark 回避负结果。
6. 高效推理单独评估模型延迟、端到端耗时、显存、调用次数和成功率变化。
   仿真在推理期间暂停时，wall time 改善不等于真实控制实时性改善；要做实时性
   主张，仍需受控延迟/动作陈旧度协议。

近期收敛建议：RoboMME 作为已有 Agent/Memory 证据的复核主线，VLA-Arena 作为
唯一优先的新平台候选，现有 LIBERO-PRO 作补充，RoboDojo 保留失败分析与展示。
LIBERO-Plus/Mem 不与上述路线同时开跑。此组合仍不构成 RA-L 录用或正收益保证。

## 1. 调研目标

本报告不按发布时间简单追逐新工作，而是筛选能够低成本验证 CARVE-VLA
两条主线的公开资源：

1. Agentic Harness：VLM 规划、任务分解、工具调用、过程记忆、异常诊断与恢复。
2. Optimize Runtime：低延迟推理、异步执行、动作新鲜度、调用调度和显存占用。

筛选标准依次为：是否真实开源、是否有可直接评测的 checkpoint、是否无需重新
微调、是否能在单张 RTX 4090 上运行、是否能形成机制级证据。

## 2. 历史结论（2026-09-09）：RoboDojo 为主，不同时铺开多个平台

**对本项目的现状，RoboDojo 是最合适的主集成平台；不是所有机制的绝对最优基准。**
这一判断来自任务覆盖、匹配权重、本机已完成的环境与视频链路，而不只是发布时间或
画面质量。当前实际 VLA 是 StarVLA Qwen3-VL-4B PI-v3，不是早期候选 Xiaomi 或 PI0.5。

| 平台 | 最适合检验的问题 | 已发布权重/部署成本 | 当前决定 |
|---|---|---|---|
| RoboDojo | 多阶段操作、语义决策、记忆及整套系统的成本 | 已有本机 StarVLA；另有 DM05-MEM 候选，资产已具备 | 主平台，先完成框架，再固定任务子集 |
| RoboMME | 明确需要历史信息的记忆机制 | 官方策略仓库提供模型；已有本项目结果 | 保留已有证据，Dojo 记忆能力不足时才补充 |
| RoboTwin 2.0 | 跨模型适配、动作推理效率和随机化鲁棒性 | DM05-robotwin2 可直接评估；迁移仍需适配验证 | 后备跨模型验证，不与主线同时开跑 |
| RoboCasa365 | 厨房场景、技能组合与长程操作 | 已有 RLDX-1-FT-RC365，不能再说一律需自己训练 | 有价值的后续扩展，暂不新增环境 |
| ReflexBench | 推理延迟对动态任务的影响 | 环境/数据公开，尚未核实即用型匹配 VLA 权重 | 借鉴计时和延迟协议，不先增加训练任务 |
| LIBERO-PRO / LIBERO-Plus | 扰动鲁棒性、低成本回归 | 匹配 LIBERO 的权重可降低适配成本，仍须核对协议 | 保留现有 PRO；两者不同，不混称已完成 |

主要依据：[RoboDojo 官方仓库](https://github.com/RoboDojo-Benchmark/RoboDojo)、
[RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark)、
[RoboMME policies](https://github.com/CuteAnnaQAQ/Robomme_Policy)、
[DM0.5 RoboTwin 2.0 部署文档](https://github.com/dexmal/opendm/blob/main/docs/en/dm05_robotwin2.md)、
[RLDX-1 RC365 checkpoint](https://huggingface.co/RLWRLD/RLDX-1-FT-RC365)、
[ReflexBench](https://github.com/LxRoboticsLab/ReflexBench)、
[LIBERO-PRO 论文](https://arxiv.org/abs/2510.03827)、
[LIBERO-Plus 论文](https://arxiv.org/abs/2510.13626)。

上表是适配性判断，不是实验结果。没有预训练权重就不能评估所有动作能力；有权重
也不代表我们已接通该模型、4090 能与仿真共驻，或模型一定适合所有任务。

## 3. 早期 Benchmark 资源目录（8 月 28 日，优先级已更新）

### 3.1 第一优先级

#### WatchAct（2026-06）

- 3,000 个长程实例、14 类任务，覆盖事件定位、程序推理、隐式意图和情景记忆。
- 官方代码将评测拆成 VLM video-to-plan 与 LIBERO action execution 两部分。
- Planning track 只需 VLM API 或本地 VLM，不需要训练低层 VLA。
- 与 CARVE 的 Event-triggered Planner、计划记忆和 Critic 最直接对应。

限制：端到端执行仍依赖 LIBERO，但此处 LIBERO 只是执行载体，不是论文主要
创新点。来源：[WatchAct official repository](https://github.com/Baiqi-Li/WatchAct)。

#### RoboMME（2026-03，ICML 2026 Oral）

- 16 个任务，覆盖 temporal、spatial、object 和 procedural memory。
- 基于 ManiSkill/SAPIEN；统一评测框架已有约 17 GB Docker 镜像。
- 官方发布 PI0.5 baseline、MME-VLA memory variants 和 VLM subgoal predictors。
- 可直接比较 `PI0.5 baseline`、`CARVE external memory` 与官方 memory policy，
  首轮不需要微调。

这是当前最适合替代“继续刷 LIBERO-PRO”的闭环 Agentic benchmark。来源：
[RoboMME policy repository](https://github.com/CuteAnnaQAQ/Robomme_Policy)、
[RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark)。

#### RoboTwin 2.0 / 2.0-Plus

- 50 个双臂任务；2.0-Plus 增加系统性的视觉、物体、语言和环境扰动。
- DM0.5、G0.5、InternVLA-A1.5、LingBot-VLA 2.0 和 TurboVLA 均有公开的
  RoboTwin checkpoint 或官方适配。
- 对 CARVE 的价值不是再追求榜单最高成功率，而是用完全相同的 Runtime
  接入大模型与 0.2B 轻量模型，报告 success-latency-memory frontier。

限制：完整 50-task x 100-episode 代价高，第一轮应选 6--10 个长程、精细与
扰动任务。来源：[DM0.5 RoboTwin guide](https://github.com/dexmal/opendm/blob/main/docs/en/dm05_robotwin2.md)、
[TurboVLA repository](https://github.com/H-EmbodVis/TurboVLA)。

### 3.2 第二优先级

#### RoboDojo（2026-07）

- 42 个仿真任务、18 个真实任务；按 Generalization、Memory、Precision、
  Long-Horizon 和 Open 五个维度组织。
- 官方 XPolicyLab 提供超过 30 个 policy adapter，并发布多种评测 checkpoint。
- 当前榜单仍很低：DM0.5 的 native score / success rate 为 24.90 / 19.34，
  Xiaomi-Robotics-1 为 20.07 / 13.93，说明仍有明显 Agentic 改进空间。

限制：Isaac Sim 5.1、本地资产约 90 GiB，单机 simulator-policy 共存成本高。
适合作为云评测或扩容后的高价值外部验证，不应阻塞主线。来源：
[RoboDojo official repository](https://github.com/RoboDojo-Benchmark/RoboDojo)、
[official leaderboard](https://robodojo-benchmark.com/leaderboard)。

#### Colosseum V2（2026-05）

- 28 个任务、13 类操作、单臂与双臂两种形态，基于 GPU 并行 ManiSkill。
- 重点测材质、颜色、相机、物体与机器人变化下的分布外泛化。
- 官方提供 ACT checkpoints；社区 LeRobot 集成还提供可直接加载的 PI0.5
  单臂和双臂 checkpoints，因此可以做到不训练的初步复现。

它比 RoboDojo 更容易落地，但对 Planner/Memory 的针对性弱于 RoboMME。
来源：[Colosseum V2 official repository](https://github.com/jstmn/ColosseumV2)、
[LeRobot integration](https://github.com/Geeksongs/lerobot_colosseum_v2)。

#### MIKASA-Robo-VLA（2026-05，ICLR 2026）

- 90 个语言条件任务、10 类记忆、Short/Medium/Long 三个 horizon split。
- 22,500 条 oracle trajectories 以 RLDS 和 LeRobot v3 公开。
- ManiSkill 环境和统一评测 Docker 较轻，任务定义与 CARVE Memory 高度匹配。

限制：公开的 160 MB checkpoint 是 oracle 数据采集器，不是可直接比较的
通用 VLA policy；要做可信 VLA 成功率通常仍需微调。因此它是训练阶段的优秀
选择，不是当前“零训练评测”首选。来源：
[MIKASA-Robo-VLA repository](https://github.com/CognitiveAISystems/MIKASA-Robo)、
[benchmark protocol](https://mikasarobo.github.io/benchmarking.html)。

### 3.3 观察名单

| Benchmark | 发布时间 | 价值 | 当前不立即采用的原因 |
|---|---:|---|---|
| ReflexBench | 2026-08 | 6 个动态任务，可配置同步/异步和推理延迟；官方 Isaac Lab 环境和 LeRobot 数据已发布 | 仓库仅发布数日，尚未形成社区复现，也未发现即用型 VLA checkpoint |
| InstructMove | 2026-08 | 语言不可替代的目标选择，能排除视觉捷径 | 发布仅数日，Isaac 4.5；未发现可直接评测的 VLA checkpoint |
| VLA-Arena | ICML 2026 | 170 tasks，Safety/Distractor/Extrapolation/Long Horizon | 范围大；完成机制验证前成本过高 |
| RoboCasa365 | ICLR 2026 | 365 个厨房任务，适合长程家务 | 9 月 9 日更正：已有匹配权重；适配成本仍需评估，见第 8 节 |
| RoboSemanticBench | 2026-06 | 诊断复杂语义是否影响动作选择 | 当前公开数据只覆盖部分 suite，主要用于训练型语义研究 |
| RoboProcessBench | 2026-06 | 约 58K 过程理解问答，可测 VLM Critic | 不是闭环动作成功率 benchmark |

ReflexBench 来源：[project page](https://reflexvla.github.io/)；InstructMove
来源：[paper](https://arxiv.org/abs/2608.22990) 与
[RoboOrchardSim](https://github.com/HorizonRobotics/RoboOrchardSim)；统一基准
支持情况可查 [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)。

## 4. 早期模型候选（8 月 28 日，不代表当前已部署）

### 4.1 最适合当前实验的模型

| 模型 | 发布时间 | 规模/本机成本 | 已发布评测 checkpoint | CARVE 中的角色 |
|---|---:|---|---|---|
| DM0.5 | 2026-07 | 官方推荐单张 RTX 4090 推理 | RoboTwin 2.0、LIBERO、SO101 | 当前高性能第二 VLA；验证通用 adapter 与 Runtime |
| TurboVLA | 2026-07 | 0.2B；31.2 ms、0.9 GB VRAM（作者 4090 数据） | RoboTwin 2.0、LIBERO | 轻量化下界和实时 policy 对照 |
| Xiaomi-Robotics-1 | 2026-07 | 5B；RoboDojo checkpoint 约 10.25 GiB | RoboDojo | RoboDojo 主模型；当前只做 model-only gate |
| InternVLA-A1.5 | 2026-07 | 3B；Qwen3.5-2B backbone | RoboTwin、LIBERO、LIBERO-Plus、DOMINO | 可选的语义/foresight 第三模型 |

DM0.5 在 RoboTwin 2.0 报告 93.6/93.3 clean/randomized，且官方明确单卡可做
推理；TurboVLA 报告 0.2B、31.2 ms 和 0.9 GB VRAM。它们形成比“PI0.5 +
另一个同量级模型”更清晰的效率对照。来源：
[OpenDM](https://github.com/dexmal/opendm)、
[TurboVLA model release](https://huggingface.co/H-EmbodVis/TurboVLA)、
[Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)、
[InternVLA-A1.5](https://huggingface.co/InternRobotics/InternVLA-A1.5-base)。

### 4.2 有价值但暂不作为第一轮模型

| 模型 | 价值 | 暂缓原因 |
|---|---|---|
| Galaxea G0.5 | 2B Qwen3.5；统一 reasoning/action stream；RoboTwin 93.3 | RoboDojo 发布包大；与 DM0.5 的第一轮证据重复 |
| GR00T N1.7 | 3B、Apache 2.0、NVIDIA 原生部署生态 | 仍为 Early Access；更偏 humanoid/cross-embodiment，当前任务 checkpoint 匹配较弱 |
| MolmoAct2 | 具身 reasoning backbone、LeRobot 与真实机器人服务器完整 | checkpoint 约 22 GB；更适合 YAM/DROID/SO101 本体 |
| LingBot-VLA 2.0 | 6B、多本体、RoboTwin checkpoint | 权重约 23.77 GiB，24 GiB 4090 几乎没有 simulator 共存空间 |
| Xiaomi-Robotics-0 | 4.7B，异步实时执行，LIBERO/CALVIN/SimplerEnv weights | 适合借鉴异步机制，但基准组合不如 DM0.5/TurboVLA 新且正交 |
| FasterWAM | inference-time future conditioning、公开 LIBERO-Plus/RoboTwin checkpoints | 属于 WAM 扩展；先完成 VLA 主线再作为一项边界对照 |
| ReflexVLA | 1B、65 ms，动态任务中显著优于静态基线 | 完整训练/评测代码尚未发布 |

来源：[G0.5](https://github.com/OpenGalaxea/GalaxeaVLA)、
[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T)、
[MolmoAct2](https://github.com/allenai/molmoact2)、
[LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)、
[Xiaomi-Robotics-0](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)、
[Faster-WAM](https://github.com/hustvl/FasterWAM)。

## 5. 是否需要微调

| 组合 | 第一轮是否需要训练 | 说明 |
|---|---|---|
| WatchAct VLM planning | 否 | 使用 API 或本地 VLM 直接生成计划 |
| RoboMME + released PI0.5 baseline | 否 | 官方发布基线和 memory variants |
| DM0.5 + RoboTwin 2.0 | 否 | 官方发布 DM05-robotwin2 |
| TurboVLA + RoboTwin 2.0 | 否 | 官方发布约 0.81 GiB EMA checkpoint |
| Xiaomi-Robotics-1 + RoboDojo | 否 | 官方发布 RoboDojo inference checkpoint |
| Colosseum V2 + community PI0.5 | 否 | 可直接复现，但需明确标注 checkpoint 来源不是基准官方 |
| MIKASA-Robo-VLA | 通常需要 | 有数据和 oracle，但缺少通用 VLA benchmark checkpoint |
| InstructMove | 当前需要或待官方补齐 | 有环境与资产，未发现即用型政策权重 |
| RoboCasa365 | 不一定需要 | 9 月 9 日更正：可用匹配的 RLDX-1-FT-RC365；模型、任务与协议须对齐 |

## 6. 与 CARVE 创新的边界

这些新模型已经包含视觉历史、latent foresight、reasoning token 或异步执行，
所以 CARVE 不应声称首次提出这些单项能力。可防守的系统贡献是：

1. 将高层语义规划、任务记忆、工具调用、风险监测和恢复组织为模型无关的
   process-control harness。
2. 通过事件触发和可信记忆降低高成本 Planner 调用，同时让本地 Monitor 保持
   高频执行风险监测。
3. 将 action freshness、deadline miss、chunk scheduling 和模型路由统一到
   Optimize Runtime，而不是只报告模型 forward latency。
4. 在高性能 VLA 与轻量 VLA 上复用同一协议，报告成功率、恢复率、调用成本、
   时延和显存的 Pareto frontier。

## 7. 当前执行顺序

1. 完成 RoboDojo 观察工具到 VLM 的反馈、语义可靠性、物理工具和记忆的端到端接入。
2. 高效推理优先解决当前实测的 VLM/VLA 换入换出成本，再评价选择性量化或异步衔接。
3. 通过框架验收后冻结模型、提示、工具权限、预算、记忆与布局；只在固定方案上统计对比。
4. 保留现有 StarVLA 为开发基线；若它对记忆任务的底层能力成为瓶颈，优先在 RoboDojo
   内核验 DM05-MEM，不为寻找新模型先迁移平台。本轮不自动下载或切换。
5. 平台确实无法提供必要证据时，才启用表中的一项补充；不重启六个平台并行建设。

唯一执行计划为 `../plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md`。本文用于选型，
不额外产生一套后台实验。现有论文和失败结果保留，不按本次结论重写历史实验。

## 8. 9 月 9 日详细复核

### 8.1 为什么继续 RoboDojo

官方基准覆盖 Generalization、Memory、Precision、Long-Horizon、Open，论文包含
42 个基础仿真任务和 18 个实机任务。框架所需的多阶段、历史信息与精细操作可在同一
套仿真接口中检查。公共评估入口通过 XPolicyLab 接政策，不要求先训练本项目基座。
但 adapter 数量不是可下载、可直接跑全部任务的 checkpoint 数量。
来源：[RoboDojo 论文](https://arxiv.org/abs/2607.04434)、
[官方实现](https://github.com/RoboDojo-Benchmark/RoboDojo)。

本地当前输入为三视角 RGB、14-D 双臂绝对关节状态/动作；StarVLA 预测 50 步，
执行 16 步，再获取新观察。官方模型卡还给出了逐任务结果，可辅助检查低层能力，
但作者报告不是我们的复现，也不能替代固定布局的配对测试。
来源：[StarVLA RoboDojo 模型卡](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo)。

高质量渲染有利于看清目标与录制操作视频；它不独立保证接触力学与真实机器人一致。
当前脚本在语义 RPC 时暂停物理推进，因而 wall time 可测，现实运动中的截止期和
动作陈旧影响尚不能直接测。增加 sleep 或按录像 FPS 算频率不构成硬实时证据。
这属于本地执行模式的限制，不是 RoboDojo 本身不能扩展实时评估。

### 8.2 平台不换也能考虑更适合记忆的动作模型

新候选是 [DM05-MEM-Robodojo-Sim](https://huggingface.co/Dexmal/DM05-MEM-Robodojo-Sim)。
其 ARX X5 动作也是 14-D 绝对关节位置，但预测 50 步、执行 25 步；除当前三视角，
还要求 1 FPS 的最多 20 张头部历史帧及起始左侧填充，不能套用 StarVLA 的 h16 输入。
匹配 checkpoint 和归一化统计后可评估，不要求我们重新微调。

官方 [XPolicyLab PR #101](https://github.com/XPolicyLab/XPolicyLab/pull/101) 已于
9 月 3 日合并。后续核验须同时固定集成 commit、权重 revision、归一化、历史采样与
episode reset；不能只替换权重路径。模型卡推荐 A100/H100/H20，单 GPU 推理不等于
已证明 4090 可以同时驻留这个记忆模型、独立 VLM 与 Isaac。

**公平性边界**：原生视觉历史属于该 VLA 自身能力。比较我们外部 Memory 开/关时，
两组必须保留相同的原生历史输入；不能通过删去基线历史制造收益。模型更换前后的
结果也不能混成同一个配对实验。若只是当前独立 VLM 的关系识别差，换低层 VLA
并不会自动解决该问题，应分别诊断两个模型。

### 8.3 任务选择与论文展示

| 任务 | 角色 | 必须证明的作用 | 不应宣称的内容 |
|---|---|---|---|
| build_tower | 多阶段/精细操作主候选 | 阶段识别、子目标切换、合法干预与实际执行反馈 | 官方属于 Precision；不能仅凭阶段多证明长期记忆 |
| put_bottles_into_dustbin | 长程操作与效率主候选 | 多目标进度、减少重复/无效调用、完整任务成本 | 桶内终态不可见时，不能用单帧 VLM 替代官方成功判定 |
| match_and_pick_from_conveyor | 时间记忆候选 | 早期提示进入记忆、后续匹配正确且能驱动实际选取 | 不能读隐藏目标标签，不能把抓取能力失败算作记忆失败 |
| store_laptop_and_headphones | 后备技能组合 | 多类动作与计划组合 | 当前模型能力弱，不能仅因视频好看就排成主实验 |
| stack_bowls | 已使用开发回归 | reset、观察和恢复缺陷排查 | 不将反复调试的布局包装成未见测试集 |

分类可核对 [官方任务目录](https://robodojo-benchmark.com/doc/sim-tasks/)。
选择标准为可执行性、可观测性、机制相关性及留出布局，不能按候选方法是否赢基线选任务。
两个主候选加一个条件准入的记忆任务，比同时铺开多个基准更适合当前资源。
这一阶段是任务子集研究，不写成完整 RoboDojo 榜单成绩。

### 8.4 官方评估协议与本项目子集协议分开

本次核对的官方 quick-evaluation 说明为：42 个基础任务加 12 个 Generalization
random 变体，共 54 个可运行配置。`--eval-num native` 读取任务配置：非泛化任务
每 seed 50 次，泛化标准/随机各 25 次并合并，完整评估重复 seeds 0/1/2。
任务目录还列有 DLC 页面，不能把页面总数当正式分母。
来源：[RoboDojo 官方评估规则](https://robodojo-benchmark.com/doc/usage/quick-evaluation/)。

这些规则可能与旧模型卡的简述不同，正式运行必须固定本地 commit、YAML 与任务清单，
并说明采用哪个版本。项目原计划的每条件 20 个留出布局是成本受控的自定义子集，
不是官方完整协议；不能改名为官方 full benchmark。若论文采用官方数量，需另行
冻结足够的未见布局和三个 seed，不覆盖先前已登记的开发/确认性协议。

无论数量多少，都固定分母，报告成功率与官方分数、配对退化、置信区间、VLM/VLA
调用与时延、整回合时间、显存、预算耗尽和失败类型。时延分冷启动/稳态、计算/传输，
不能仅挑成功且快的回合；当前软件测试不充当这些仿真数据。

### 8.5 补充平台何时值得采用

- RoboMME 的 16 项时间/空间/对象/过程记忆任务，适合定向解释“记忆到底在帮什么”；
  现有结果先保留，不因最近出现新平台就失效。
  来源：[官方仓库](https://github.com/RoboMME/robomme_benchmark)。
- RoboTwin 2.0 适合第二模型和效率泛化；DM0.5 有匹配权重，不必重新采集训练。
  但多任务和随机化本身不代表存在强记忆需求。
  来源：[DM0.5 官方部署指南](https://github.com/dexmal/opendm/blob/main/docs/en/dm05_robotwin2.md)。
- RoboCasa365 已有 [RLDX-1-FT-RC365](https://huggingface.co/RLWRLD/RLDX-1-FT-RC365)，
  与较早的 [RLDX-1-FT-ROBOCASA](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) 是
  不同发布包。应按任务和权重分别对齐，不能用 24 任务模型的配置冒充 365 任务适配。
- [ReflexBench](https://github.com/LxRoboticsLab/ReflexBench) 的六类动态任务更直接检验
  延迟后果；目前可借鉴协议，未核实匹配 VLA checkpoint 时不承诺免训练评估。

**研究叙事保持不变**：语义 Agent 与工具反馈改善执行，工作/经验记忆支撑持续任务；
高效推理降低这套系统的计算和调度成本。平台只提供证据场景，不成为项目贡献本身。

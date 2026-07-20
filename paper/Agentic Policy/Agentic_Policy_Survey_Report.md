# Agentic Policy 论文总报告

- 目录：`paper/Agentic Policy`
- 阅读对象：面向 robot manipulation 的 Agentic Policy / VLA / WAM 相关论文
- 覆盖论文：
  - `Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models`
  - `RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks`
  - `Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents`
  - `MEM: Multi-Scale Embodied Memory for Vision Language Action Models`
  - `VLA^2: Empowering Vision-Language-Action Models with an Agentic Framework for Unseen Concept Manipulation`
  - `Sci-VLA: Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments`

## 一句话总判断

这些论文里的 `agentic policy` 不是一个统一技术点，而是一组把 VLA 从“单次反应式动作模型”扩展成“可计划、可验证、可恢复、可记忆、可适配、可调用工具”的系统化路线。

如果要为你的下一步工作找方向，我建议优先走：

> 部署时 agentic inference supervisor + subgoal verification + recovery / transition bridging + multi-scale memory

也就是先做一个不依赖大规模在线 RL 的可运行闭环，再逐步加入低数据适配或经验记忆。

## 1. 横向分类

| 路线 | 代表论文 | Agentic 落点 | 主要解决问题 | 实现代价 | 对你当前工作的优先级 |
|---|---|---|---|---|---|
| 执行闭环 | Agentic Robot | planner + VLA executor + verifier + recovery | 长程任务错误累积 | 中 | 最高 |
| 生命周期闭环 | RoboClaw | VLM meta-controller + MCP tools + policy pool + EAP | 数据采集、policy 学习、部署割裂 | 高 | 中高 |
| 在线适配 | Agentic-VLA | ARS reward + LGE exploration + EM warm start + GRPO | low-data, transfer, online adaptation | 高 | 中 |
| 多尺度记忆 | MEM | short-term video memory + long-term text memory | 遮挡、重复尝试、长程任务状态记忆 | 高 | 中高 |
| OOD 概念增强 | VLA^2 | web/memory retrieval + detector/segmenter + text replacement | 未见过的物体概念和外观 | 中高 | 按任务需要 |
| transition bridge | Sci-VLA | LLM 生成 atomic task 之间的 transition action | skill chaining / state gap | 中 | 高 |

核心启发是：不要把 agentic 能力只理解成“planner”。真正有效的系统往往把 agent 放在 VLA 的边界处，负责：

- 把任务切成 VLA 能执行的单位
- 判断当前 subgoal 是否真的完成
- 在失败、卡住、状态不对时插入恢复动作
- 记住已经做过什么、哪里失败过、当前处于哪个阶段
- 必要时改写输入，让 VLA 看到更熟悉的视觉或语言形式
- 在数据足够时，把失败经验变成后续训练或适配信号

## 2. 逐篇精读摘要

## 2.1 Agentic Robot

论文标题：`Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents`

### 核心贡献

这篇的核心是 `SAP: Standardized Action Procedure`。它把长程 manipulation 做成一个标准闭环：

1. `Planner`：用 LRM/VLM 把高层任务拆成 atomic skill library 里的 subgoals。
2. `Executor`：用 VLA 执行当前 subgoal，输出连续 7-DoF action。
3. `Verifier`：用 VLM 看第三视角和腕部相机的短视频 buffer，判断 subgoal 是否完成。
4. `Recovery`：如果没有完成或卡住，执行简单恢复，比如 lift gripper，然后 retry。

从 Figure 1 可以看出，它不是重训一个更强 VLA，而是在 VLA 外面加了一个规范化控制循环。Figure 3 的 failure case 很有价值：OpenVLA 抓取失败后继续执行导致任务失败，而 Agentic Robot 通过 verifier 发现失败，插入 `lift the gripper`，再重试并完成任务。

### 实验结论

LIBERO 平均成功率约 `79.6%`，LIBERO-Long 上约 `61.6%`，优于 OpenVLA、SpatialVLA 等基线。消融里最关键的发现是：

- 没有 fine-tuned verifier，LIBERO-Long 成功率从 `61.8%` 掉到 `35.3%`
- 没有 subgoal decomposition，掉到 `53.7%`
- 没有 visual input，掉到 `57.4%`
- 没有 recovery，掉到 `59.7%`

这说明真正支撑性能的是 verifier 的质量，而不是 planner 的叙事包装。

### 局限

它的 recovery 很简单，主要处理 gripper 卡住、抓取失败这类局部错误。对于空间规划冲突、重复物体放置、长期记忆问题，仍然会失败。论文里 Moka-Moka 任务就是例子：第一个 moka pot 放置正确，但第二个放置空间不足，系统没有提前建模这个空间约束。

### 对你的启发

这是最适合先复现思想的路线。你可以先实现：

- task decomposition
- subgoal-level visual verifier
- stuck detector
- small recovery library
- subgoal execution state machine

这套东西不要求训练新的 VLA，最容易形成可演示原型。

## 2.2 RoboClaw

论文标题：`RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks`

### 核心贡献

RoboClaw 的对象不是单次 episode，而是机器人 policy 的完整生命周期：

- 数据采集
- policy pool 增长
- self-resetting rollout
- long-horizon deployment
- human intervention
- failure data 回流

Figure 1 展示了 workflow：robot developer 提供配置、MCP tools 和 skills，RoboClaw 维护 file-based memory、embedding search 和 memory management。数据通过 human demonstration warmup，再进入 online rollout 和 EAP self-resetting，最后形成 policy pool 并用于长程任务。

Figure 2 是最值得借鉴的结构：

- `Structured Memory`
  - role identity
  - task-level memory
  - working memory
- `CoT Planning`
  - 当前观察是什么
  - 当前 subtask 是什么
  - 成功标准是什么
  - 当前状态是否满足成功标准
  - 下一步做什么
- `MCP Tools`
  - Start Policy
  - Terminate Policy
  - Change Policy
  - Env Summary
  - Fetch Robot Stats
  - Call Human

### 实验结论

RoboClaw 主要证明系统效率：

- 数据采集人力显著下降
- rollout 中人工干预频率下降
- 多轮迭代后长程任务成功率提升
- 论文摘要称长程任务成功率相比基线提升约 `25%`，人类时间投入减少约 `53.7%`

### 局限

它更像一套真实机器人系统工程，而不是一个容易在仿真里独立复现的算法模块。要完整复现，需要 policy pool、tool interface、数据流、reset policy 和真实或高保真机器人环境。

### 对你的启发

RoboClaw 最值得借的不是 EAP 本身，而是它的 memory/tool/controller 三层结构。你的系统可以先设计成：

- `Agent State`：当前任务、subgoal、失败次数、已完成步骤
- `Tool API`：start policy、stop policy、reset、fetch observation、verify
- `Memory`：保存 subgoal 执行历史、失败原因、恢复动作结果

这会让你的工作从“一个 planner demo”变成一个真正的 agentic manipulation framework。

## 2.3 Agentic-VLA

论文标题：`Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models`

### 核心贡献

这篇论文的 agentic 落点在训练和在线适配，而不是部署时 planner。

Figure 1 很关键：它包含三个模块：

1. `Adaptive Reward Synthesis`
   - LLM 分解 subgoals
   - capability tracker 估计每个 subgoal 当前掌握程度
   - 对不会的 subgoal 加大 reward weight
   - 用 progress critic 生成 dense reward
2. `Language-Guided Exploration`
   - VLM critic 观察状态和历史
   - 生成自然语言 exploration suggestion
   - 通过 prompt augmentation 引导 VLA 探索
3. `Experience Memory`
   - 按 task embedding 检索相似任务
   - 取出相关 adapted policy weights
   - 用于新任务 warm start

然后用 GRPO 做 online policy update。

### 实验结论

这篇的优势集中在 low-data 和 transfer：

- one-shot 平均成功率约 `70.5%`，明显高于 OpenVLA-OFT 和 EVOLVE-VLA
- cross-task transfer 从 `0%` direct transfer 提升到约 `31.2%`
- LIBERO-Long 达到 90% 目标成功率只需约 `700` iterations，约 `22.4k` rollouts，速度约为 Simple VLA-RL 的 `2.4x`
- 消融显示 ARS、LGE、EM 都有贡献，full system 在 LIBERO-Long ablation 表中约 `98.1%`

### 局限

它需要可靠的环境交互、progress critic、reward synthesis 和在线优化预算。对于真实机器人，在线 RL 的安全性和数据成本会是很大问题。

### 对你的启发

短期不要直接把它作为主路线，除非你已经有稳定仿真环境和自动评估信号。更现实的借法是：

- 借 `capability-aware subgoal weighting`，用于 planner 或 evaluator 的优先级调度
- 借 `language-guided exploration`，用于失败后的 retry hint，而不是马上做 RL
- 借 `experience memory` 的 task-indexed 结构，存储成功/失败案例和适配配置

## 2.4 MEM

论文标题：`MEM: Multi-Scale Embodied Memory for Vision Language Action Models`

### 核心贡献

MEM 的核心判断是：机器人记忆不能只用一种表示。长程任务需要不同粒度的 memory：

- 短期视觉记忆：处理遮挡、最近动作动态、失败后的重抓策略
- 长期语言记忆：记录任务阶段、已经完成的步骤、已经操作过的对象

Figure 1 清楚展示了两层 memory：

- `Long-Term Text Memory`：由高层策略更新自然语言摘要
- `Short-Term Video Memory`：由 video encoder 压缩最近多帧观察
- 高层策略输出 subtask 和 memory update
- 低层 VLA 根据 subtask、短期视觉记忆和目标输出动作

### 实验结论

Figure 6 的消融很关键：

- 无 memory 的 VLA 在 recipe setup、clean kitchen 等长程任务上表现差
- only video memory 和 only text memory 都不够
- `text + video` 的 MEM 才显著提升

Figure 7 说明短期记忆可以支持 in-context adaptation，例如尝试过错误抓取方式后，下次换 grasp height 或 approach direction。

Figure 8 进一步比较 memory 方案：pool memory、proprio memory 都只能解决部分任务，MEM 在 partial observability、counting、visual/spatial memory 上更均衡。

### 局限

MEM 是模型结构级工作，训练和数据成本高。它不适合作为你下一步最小原型的第一步，但非常适合作为你系统中 memory design 的理论支撑。

### 对你的启发

你的系统里至少应该区分两类 memory：

- `execution memory`：自然语言或结构化 JSON，记录完成了什么、失败在哪里、当前 subgoal 状态
- `visual memory`：最近 N 帧图像或 embedding，供 verifier 判断动作是否产生变化

不要把所有历史轨迹都塞进 prompt。MEM 的经验是：长期记忆要压缩成语义摘要，短期记忆才保留密集视觉。

## 2.5 VLA^2

论文标题：`VLA^2: Empowering Vision-Language-Action Models with an Agentic Framework for Unseen Concept Manipulation`

### 核心贡献

VLA^2 解决的是 OOD object concept。它假设 VLA 对训练集内对象和描述能操作，但遇到未见过的纹理、品牌、别名、陌生物体描述时会崩。

Figure 2 展示了它的三段架构：

1. `Preliminary Information Processing`
   - planner
   - vision pre-processing
   - object/location extraction
2. `Cognition & Memory`
   - web data
   - vision memory
   - text memory
   - knowledge converter
3. `Judgment & Execution`
   - VOS 生成 masked image flow
   - VLA 执行
   - verifier 判断

它做了两种关键输入改写：

- 视觉侧：检测、分割、透明彩色 mask，让目标对象更显式
- 语言侧：把 OOD 名词替换成 VLA 已知词表中的相近概念

### 实验结论

Custom Hard benchmark 上：

- VLA^2 约 `76.2%`
- OpenVLA-OFT 约 `47.4%`
- OpenVLA 约 `32.0%`
- Agentic Robot reproduction 约 `26.2%`

消融显示：

- 去掉 mask，Hard 平均从 `76.2` 掉到 `64.8`
- 去掉 replace，掉到 `51.2`
- 去掉 web，掉到 `65.2`
- 三者都去掉接近 Agentic Robot reproduction，约 `26.2`

这说明 text replacement 是最大贡献，mask 和 web retrieval 也有明显帮助。

### 局限

它的收益来自 OOD 概念场景。如果任务主要是已知物体的长程执行，复杂 web/detection/segmentation 工具链不一定值得。论文也提到 in-domain 上不一定超过更强 backbone，Object 类还会受检测和命名误差影响。

### 对你的启发

如果你的任务涉及开放词汇物体，可以做轻量版：

- 不一定先接 web
- 先做 known vocabulary + synonym mapping + object detector grounding
- 对 verifier 和 planner 使用同一套对象别名表

这样能避免“planner 叫一个名字，VLA/检测器认另一个名字”的语义错配。

## 2.6 Sci-VLA

论文标题：`Sci-VLA: Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments`

### 核心贡献

Sci-VLA 解决的是 `state gap`：VLA 训练时只见 atomic tasks，推理时把 atomic tasks 拼成长程 composite task，中间状态没有训练数据覆盖，于是第二个 atomic task 启动时失败。

Figure 1 和 Figure 3 都很重要。它的做法不是重训 VLA，而是在 atomic task 边界插入 agentic plugin：

1. VLA 执行 atomic task i
2. task i 结束后断开 VLA
3. LLM agent 根据当前关节、相机、下一个 atomic task 和检索到的 demo，生成 transition action code
4. 执行 transition action，让机器人移动到下一个 task 的合理起始状态
5. 重新连接 VLA，执行 atomic task i+1

### 实验结论

在 cleaning table 任务中，base VLA 往往第一个 atomic task 成功，后续 atomic task 失败；加入 Sci-VLA 后第二、第三个 atomic task 成功率显著提升。

在科学仪器操作的长程任务中，Sci-VLA 在多个 3/5/8-step sequence 上提高后续 atomic task 的成功率。论文也强调它只解决 atomic task 之间的 transition，不解决 atomic task 内部的精细操作能力。

### 局限

生成 transition code 有 hallucination 和安全风险，需要模板约束、动作范围限制和 collision avoidance。它依赖环境 API 或机器人控制接口，不是纯 VLA prompt 能解决的。

### 对你的启发

这篇对你非常值得借，因为它是部署时最小改动路线：

- 不改 VLA
- 不做在线 RL
- 只在 subgoal 边界插入 transition / recovery action
- 特别适合解决 skill chaining

可以把它和 Agentic Robot 合起来：

- Agentic Robot 负责 subgoal verifier
- Sci-VLA 负责 subgoal transition bridge
- RoboClaw 负责 memory/tool 组织

## 3. 对你工作的建议路线

## 3.1 最值得做的研究定位

我建议把你的下一步定位成：

> 面向长程 robot manipulation 的 agentic VLA inference framework，通过显式 subgoal state、视觉验证、多尺度记忆和 transition/recovery 动作，降低 VLA 在长程任务中的错误累积和 state gap。

这个定位的好处是：

- 比单纯 planner 更完整
- 比 Agentic-VLA 式在线 RL 更容易落地
- 可以清楚吸收 Agentic Robot、RoboClaw、MEM、Sci-VLA 的优点
- 消融实验容易设计
- 对真实机器人或仿真都成立

## 3.2 最小可行系统

建议先做一个五模块系统：

1. `Task Planner`
   - 输入自然语言任务和初始图像
   - 输出 subgoal list
   - 每个 subgoal 必须有 object、target、success condition

2. `Execution State Memory`
   - 保存当前 subgoal index
   - 保存每个 subgoal 的状态：pending/running/success/failed/recovered
   - 保存失败次数、最近恢复动作、关键视觉描述

3. `VLA Executor`
   - 执行当前 subgoal
   - 不需要先改 VLA 模型
   - 可以把当前 subgoal + whole task + memory summary 一起作为 instruction

4. `Subgoal Verifier`
   - 输入最近若干帧视觉 buffer、当前 subgoal、success condition
   - 输出 success / not success / stuck / unsafe
   - 最好能输出失败原因

5. `Recovery / Transition Manager`
   - 如果 stuck：lift gripper、backoff、retry grasp
   - 如果 subgoal success：生成到下一个 subgoal 的 transition
   - 如果连续失败：replan 或 call human

这就是一个轻量版：

> Agentic Robot + Sci-VLA + RoboClaw memory shell

## 3.3 第一版不要急着做什么

不建议第一版就做：

- 完整在线 RL
- policy weight memory
- web-scale unknown object retrieval
- 从零训练 memory VLA
- 大而全的 MCP 工具系统

这些都很有价值，但会拖慢你验证核心假设。第一版应该先证明：

> VLA 的长程失败中，有多少可以通过 subgoal verification、transition bridge 和 recovery memory 解决。

## 3.4 可以形成论文差异化的点

如果你要往 paper 方向推进，可以考虑三个差异化问题。

### 方向 A：Verifier-aware transition recovery

现有 Agentic Robot 有 verifier 和简单 recovery，Sci-VLA 有 transition code，但两者没有很好结合。你可以做：

- verifier 判断 subgoal 是否完成
- 如果完成但不适合下一个 subgoal，生成 transition
- 如果没完成，生成 recovery
- transition/recovery 都由同一个 state memory 约束

这会比单纯 `lift gripper` 更强。

### 方向 B：Memory-grounded subgoal execution

借 MEM 的思想，但不训练新 VLA，先做系统级 memory：

- 短期视觉 buffer 给 verifier
- 长期文本/JSON memory 给 planner 和 transition manager
- 失败案例写入 episodic memory
- retry 时避免重复同一种失败策略

这比普通 ReAct planner 更贴近机器人。

### 方向 C：Capability-aware scheduler without RL

借 Agentic-VLA 的 capability tracker，但先不用 GRPO：

- 每个 subgoal 类型维护成功率
- 失败多的 subgoal 分配更频繁 verifier
- 对高风险 subgoal 提前启用 transition/recovery
- 根据历史表现动态选择 skill 或 prompt

这可以形成一个轻量在线自适应系统。

## 4. 推荐实验设计

## 4.1 Baselines

至少应该有：

- `VLA only`
- `Planner + VLA`
- `Planner + VLA + Verifier`
- `Planner + VLA + Verifier + Recovery`
- `+ Transition Bridge`
- `+ Execution Memory`

如果有条件，再加：

- `Agentic Robot-style recovery only`
- `Sci-VLA-style transition only`
- `Memory without visual buffer`
- `Memory without long-term state summary`

## 4.2 Metrics

不要只报 overall success rate。建议同时报：

- task success rate
- subgoal success rate
- recovery success rate
- transition success rate
- number of repeated failures
- number of verifier calls
- execution time overhead
- human intervention count
- failure localization accuracy

这些指标能证明 agentic framework 的真实价值。

## 4.3 Failure Taxonomy

建议把失败分成：

- `perception failure`：目标识别错、物体别名错
- `execution failure`：抓取失败、放置失败、碰撞、滑落
- `verification failure`：误判完成或误判失败
- `transition failure`：上一个 subgoal 成功但下一个 subgoal 起始状态不合适
- `planning failure`：subgoal 顺序或空间约束错误
- `memory failure`：忘记已经操作过的对象或重复无效策略

这套 taxonomy 会让你的分析比单纯成功率更有说服力。

## 5. 具体下一步

建议按三阶段推进。

### Phase 1：闭环原型

目标：复现 Agentic Robot 的核心闭环，但加入更明确的 execution memory。

要做：

- subgoal planner
- VLA executor wrapper
- visual verifier
- simple stuck detector
- lift/backoff/retry recovery
- JSON execution memory

验收标准：

- 在至少 5 到 10 个长程 manipulation 任务上，能证明比 `planner + VLA` 少犯级联错误

### Phase 2：transition bridge

目标：解决 Sci-VLA 提到的 state gap。

要做：

- 在 subgoal 切换处判断当前状态是否适合下一步
- 生成或选择 transition action
- 加入安全模板和动作范围约束
- 记录 transition 成败

验收标准：

- 后续 subgoal 的启动成功率明显提升
- 重复 jitter / stuck 明显减少

### Phase 3：memory 和 capability-aware adaptation

目标：让系统从失败中学习，但不一定做 RL。

要做：

- subgoal success statistics
- failure-recovery memory
- retry strategy selection
- verifier frequency adaptive scheduling
- prompt / skill selection based on past success

验收标准：

- 同一类任务多次执行时，失败次数下降
- 新组合任务中，能利用相似历史 subgoal 的经验

## 6. 最终建议

如果只能选一条主线，我建议你不要先追 `Agentic-VLA` 那种完整 online RL，而是做：

> Memory-grounded agentic inference for long-horizon VLA manipulation

具体落点是：

- subgoal-level closed-loop verification
- transition/recovery action insertion
- structured execution memory
- capability-aware retry and verifier scheduling

这条线的技术难度适中，能吸收这批论文的核心思想，又能和普通 VLA、普通 planner、单纯 verifier 工作拉开差异。

## 7. 联网补充调研：截至 2026-06-03 的最新进展

本节根据 2026-06-03 的联网检索补充，重点关注两个交集：

- 与当前 `Agentic Policy / VLA / WAM` 长程 manipulation 方向相近
- 能自然结合具身大模型推理优化，包括轻量化、动态推理、action chunking、计算复用和低延迟执行

主要一手来源为 arXiv 页面和论文项目页。

## 7.1 最新方向图谱

| 方向 | 代表工作 | 与我们关系 | 推理优化切入点 |
|---|---|---|---|
| Memory-Verifier-Recovery Agentic VLA | HELM, Goal2Skill | 与我们计划高度重叠 | verifier 调用频率、失败预测早停、rollback/replan 成本控制 |
| 动态 action chunk / execution commitment | AAC, A3 | 非常适合接到 agentic policy 执行层 | 根据不确定性动态决定执行多少动作，减少 VLA 调用 |
| Action-context adaptive computation | AC^2-VLA | 与硕士课题强相关 | 跨时间复用 cognition、token pruning、选择性执行模块 |
| Latent test-time compute scaling | RD-VLA | 适合做“推理深度自适应” | 简单状态少算，复杂状态多算，避免 CoT token 开销 |
| 轻量化 VLA / action expert | AnoleVLA, DynamicVLA, AnchorVLA, AR-VLA | 可作为底层 executor 优化参考 | SSM、compact VLA、异步推理、truncated diffusion、长期 action memory |
| WAM / latent action / world model | VISTA, WAM, JOPAT, ALAM | 可增强 subgoal/transition/recovery | 用世界模型生成视觉 subgoal、预测 transition、减少真实交互 |
| procedural memory / adapter retrieval | VLA-Pro, Agentic-VLA | 与 experience memory 相近 | 动态 LoRA/adapter 融合，但需控制检索和融合成本 |

## 7.2 与我们最接近的新工作

### HELM: Harness-Enhanced Long-horizon Memory for VLA Manipulation

- arXiv: <https://arxiv.org/abs/2604.18791>
- 时间：2026-04-20
- 核心问题：长程 VLA 的失败不是简单加长 context 就能解决，而是来自三类缺口：
  - memory gap
  - verification gap
  - recovery gap
- 方法：
  - `Episodic Memory Module`: 用 CLIP-indexed keyframes 检索任务历史
  - `State Verifier`: 从 observation、action、subgoal、memory-conditioned context 预测动作失败
  - `Harness Controller`: rollback 和 replanning
- 结果：
  - LIBERO-LONG 从 OpenVLA 的 `58.4%` 提升到 `81.5%`
  - 扩展 context window 到 H=32 只带来 `5.4` 点提升
  - same-budget LoRA adaptation 仍低于 HELM

#### 对我们的影响

HELM 和我们上一节建议的方向非常接近，说明 `memory + verifier + recovery` 已经成为 2026 年的明显热点。它也提醒我们：如果只做这个组合，创新性会受到挑战。

#### 可避开的重复点

不要只提出“加 memory、加 verifier、加 recovery”。这已经被 HELM 很明确地做了。

#### 可继续推进的空白

HELM 主要强调成功率和 recovery，没有把推理成本作为核心目标。我们可以切入：

- verifier 不是每步调用，而是由风险/不确定性动态触发
- recovery/replan 不总是调用大模型，而是先用轻量动作库和小 verifier
- memory 检索不是越多越好，而是受 token budget / latency budget 约束
- 把 `dynamic action chunking` 与 `state verifier` 结合：只有高置信前缀才执行，低置信部分触发 replan

这会把我们的方向从 `agentic robustness` 推到 `efficient agentic robustness`。

### Goal2Skill: Long-Horizon Manipulation with Adaptive Planning and Reflection

- arXiv: <https://arxiv.org/abs/2604.13942>
- 时间：2026-04-15
- 核心：dual-system framework，高层 VLM agent 负责 structured memory、goal decomposition、outcome verification 和 error-driven correction，低层 VLA/diffusion executor 执行 subtask。
- 结果：RMBench 上平均成功率 `32.4%`，强基线 `9.8%`。

#### 对我们的影响

这篇进一步说明高层 agent + 低层 VLA executor 的 dual-system 方向已经很拥挤。我们的差异化要落在：

- 高层 agent 的推理优化
- 何时规划、何时验证、何时恢复的动态调度
- 长程任务中 VLM/VLA 调用预算如何最小化

## 7.3 与硕士课题最相关的推理优化进展

### AC^2-VLA: Action-Context-Aware Adaptive Computation

- arXiv: <https://arxiv.org/abs/2601.19634>
- 时间：2026-01-27
- 核心观察：VLA closed-loop deployment 的瓶颈是每个 timestep 重复运行大 VLM/VLA backbone；VLA 推理在 temporal、spatial、depth 维度都有冗余。
- 方法：
  - cognition reuse across timesteps
  - token pruning
  - selective execution of model components
  - action-guided self-distillation
- 结果：
  - 最多 `1.79x` speedup
  - FLOPs 降到 dense baseline 的 `29.4%`
  - success 基本保持

#### 对我们的启发

这是最适合你硕士课题的论文之一。它的不足是偏单个 VLA 内部的 adaptive computation，尚未充分考虑 agentic policy 的层级结构。

我们可以扩展成：

> Agentic-Context-Aware Adaptive Computation

不仅根据 observation/action context 裁剪 VLA，还根据：

- subgoal 难度
- verifier 置信度
- memory 中类似失败次数
- transition risk
- 当前是否处于 recovery 阶段

动态决定：

- 是否调用 planner
- 是否调用 verifier
- 是否复用上一帧视觉编码
- action chunk 执行多长
- 是否使用小模型先判定、大模型兜底

### Adaptive Action Chunking at Inference-time

- arXiv: <https://arxiv.org/abs/2604.04161>
- 时间：2026-04-05，CVPR 2026
- 核心：固定 action chunk length 不适合所有状态。大 chunk 响应慢，小 chunk 容易 mode-jumping 和 jerk。
- 方法：用 action entropy 作为 cue，自适应决定 chunk size。

#### 对我们的启发

这是一个可以直接插入 agentic policy executor 的轻量推理优化模块：

- verifier 认为状态稳定、action entropy 低：执行长 chunk，少调用 VLA
- verifier 认为风险高、接近接触/抓取/放置边界：执行短 chunk，多反馈
- recovery 阶段：短 chunk + 高频 verifier
- transition 阶段：中等 chunk + safety check

这比固定每 N 步 verification 更自然。

### Dynamic Execution Commitment / A3

- arXiv: <https://arxiv.org/abs/2605.11567>
- 时间：2026-05-12
- 核心：action chunk 不是预测出来就全执行，而是做 self-speculative prefix verification，只接受最长可信动作前缀。
- 方法：
  - group sampling 估计 trajectory-wise consensus
  - 对低共识动作做 conditional invariance verification
  - 只执行满足 prefix-closed sequential consistency 的连续前缀

#### 对我们的启发

AAC 用 entropy 决定 chunk size，A3 用自验证决定可执行前缀。我们的 agentic policy 可以进一步把外部 verifier 接进去：

- 内部动作一致性高 + 外部 state verifier 风险低：执行长前缀
- 内部一致性低或外部 verifier 预测失败：截断、replan 或 recovery

这正好形成“Agentic Policy 的推理优化”。

### RD-VLA: Recurrent-Depth VLA

- arXiv: <https://arxiv.org/abs/2602.07845>
- 时间：2026-02-08
- 核心：VLA 不应对简单动作和复杂动作使用固定计算深度。CoT 虽可变计算，但 token/memory 开销大，不适合连续动作空间。
- 方法：
  - recurrent weight-tied action head
  - latent iterative refinement
  - adaptive stopping based on latent convergence
- 结果：
  - 一些任务 single-iteration 为 `0%`，四轮 refinement 超过 `90%`
  - 相比 token reasoning VLA，可实现常数 memory 和最高 `80x` inference speedup

#### 对我们的启发

RD-VLA 给了一个很好的论据：具身推理优化不一定是生成更多语言推理 token，而可以是 latent/action-space refinement。

我们的系统可以采用两层 compute：

- 高层 agent：少调用，负责 semantic state transition
- 低层 action head：在关键动作处做 latent refinement

这能避免“每一步都用 VLM CoT”的大开销。

## 7.4 轻量化 VLA / Action Expert 方向

### AnoleVLA

- arXiv: <https://arxiv.org/abs/2603.15046>
- 时间：2026-03-16
- 核心：用 deep state space model 替代标准 transformer backbone，提升资源受限场景部署能力。
- 结果：真实实验中成功率比代表性大 VLA 高 `21` 点，推理速度约 `3x`。

#### 对我们的启发

如果你希望论文更贴近“轻量化”，可以把低层 executor 设计成可替换：

- heavy VLA：高风险 subgoal / 首次执行
- lightweight VLA / SSM executor：重复、低风险、已掌握 subgoal

这就从单模型轻量化变成 agentic routing。

### DynamicVLA

- arXiv: <https://arxiv.org/abs/2601.22153>
- 时间：2026-01-29
- 核心：
  - compact `0.4B` VLA
  - Continuous Inference：推理与执行重叠
  - Latent-aware Action Streaming：减少感知-执行时间差
- 同时提出 Dynamic Object Manipulation benchmark。

#### 对我们的启发

它提醒我们：真实机器人上的推理优化不是只看 FLOPs，也要看：

- perception staleness
- 推理和执行是否能流水线重叠
- action streaming 是否能被 verifier 中断

我们的 agentic policy 可以设计成 async pipeline：

- 当前 chunk 执行时，后台预取下一 subgoal 的视觉编码/候选 transition
- verifier 只在高风险节点同步阻塞

### AnchorVLA

- arXiv: <https://arxiv.org/abs/2604.01567>
- 时间：2026-04-02
- 核心：diffusion policy 生成多模态动作强，但完整 denoising 成本高。AnchorVLA 从 anchor trajectory 附近开始，使用 truncated diffusion，并加轻量 residual correction 做高频自修正。

#### 对我们的启发

可以和 transition bridge 结合：

- agent 生成或检索 anchor transition
- diffusion/residual module 只做局部修正
- 不必每次从噪声完整采样动作

### AR-VLA

- arXiv: <https://arxiv.org/abs/2603.10126>
- 时间：2026-03-10，RSS 2026 accepted
- 核心：独立 autoregressive action expert 维护自己的 long-lived memory，解决视觉语言慢、控制频率快的 mismatch。
- 关键点：
  - refreshable vision-language prefixes
  - action expert 连续生成动作
  - re-anchoring 处理 perception staleness

#### 对我们的启发

它和我们的 memory-grounded agentic policy 很契合。可以把系统拆成：

- slow agent/VLM：规划、验证、恢复
- fast action expert：持续执行并维护运动历史
- re-anchoring：当上层 perception 更新滞后时校正动作

## 7.5 WAM / World Model / Latent Action 最新趋势

### VISTA: Scaling World Model for Hierarchical Manipulation Policies

- arXiv: <https://arxiv.org/abs/2602.10983>
- 时间：2026-02-11
- 核心：world model 作为高层 planner，生成 subtask sequence 和 goal images；VLA 作为低层 executor 根据文本和视觉目标执行。
- 结果：在 OOD 新场景中，同结构 VLA 借助 world model guidance 从 `14%` 提升到 `69%`。

#### 对我们的启发

这直接支持一个方向：不要只让 planner 输出文本 subgoal，而是输出 visual subgoal / goal image / expected state sketch。对 verifier 和 transition bridge 都有帮助。

推理优化角度：

- 不必每步调用 world model
- 只在 subgoal 边界或 verification 失败时生成 visual subgoal
- 对高置信 routine subgoal 复用历史 goal template

### WAM: Enhancing Policy Learning with World-Action Model

- arXiv: <https://arxiv.org/abs/2603.28955>
- 时间：2026-03-30
- 核心：World-Action Model 同时建模未来视觉和导致状态变化的动作，通过 inverse dynamics objective 让 world model 学到 action-relevant 表征。
- 结果：
  - CALVIN 上 BC 成功率从 `59.4%` 到 `71.2%`
  - PPO fine-tuning 后从 `79.8%` 到 `92.8%`
  - 训练步数少 `8.7x`

#### 对我们的启发

WAM 可作为 recovery/transition 的低成本模拟器：

- 先在 latent world-action model 里评估候选 recovery
- 只执行最可信的 recovery
- 减少真实机器人试错

### JOPAT: Point Tracking Improves World Action Models

- arXiv: <https://arxiv.org/abs/2605.23856>
- 时间：2026-05-22
- 核心：pixel prediction 容易被光照、纹理等无关因素干扰；point tracks 显式表示运动，更适合 occlusion、object interaction、off-screen motion。

#### 对我们的启发

这对 verifier 很重要。与其让 verifier 只看 RGB，不如引入：

- object/point track consistency
- gripper-object relative motion
- target object displacement
- occlusion-aware progress estimate

这能让 verifier 更轻、更稳，也更适合推理优化。

### ALAM: Algebraically Consistent Latent Action Model

- arXiv: <https://arxiv.org/abs/2605.10819>
- 时间：2026-05-11
- 核心：从 action-free video 学 latent transitions，并用 composition/reversal consistency 让 latent transition 具备局部可加性和可逆性。
- 结果：
  - MetaWorld MT50 从 `47.9%` 到 `85.0%`
  - LIBERO 从 `94.1%` 到 `98.1%`

#### 对我们的启发

ALAM 很适合补强 transition bridge：

- transition 不一定直接生成机器人动作
- 可以先生成结构化 latent transition
- 再由 action generator 生成动作

这比纯 LLM 生成 transition code 更稳，也更适合数据驱动优化。

## 7.6 Procedural Memory 与 Agentic Online Adaptation

### VLA-Pro

- arXiv: <https://arxiv.org/abs/2605.29562>
- 时间：2026-05-28
- 核心：把 task-specific LoRA adapters 存成 parameterized procedural memories，推理时根据 multimodal context 检索并动态融合 adapter 来生成 action chunk。
- 结果：
  - simulation 最高 `207%` relative improvement
  - real-world success 从 `5.8%` 到 `65.0%`

#### 与 Agentic-VLA 的关系

VLA-Pro 和 Agentic-VLA 的 Experience Memory 很像，但 VLA-Pro 更具体：memory 是 LoRA adapters，并且推理时动态融合。

#### 对我们的启发

这条线适合作为后期增强，不建议第一版做。原因：

- adapter 检索/融合本身有推理成本
- 多 adapter 融合可能引入稳定性问题
- 需要足够多 task-specific adapters

但它可以作为我们的 long-term roadmap：

> execution memory -> recovery memory -> procedural adapter memory

## 7.7 统一大模型趋势：Qwen-VLA

- arXiv: <https://arxiv.org/abs/2605.30280>
- 时间：2026-05-28，v2: 2026-06-01
- 核心：将 manipulation、navigation、trajectory prediction 统一进一个 VLA，使用 DiT action decoder，并通过 embodiment-aware prompt conditioning 支持多机器人平台。
- 结果：
  - LIBERO `97.9%`
  - Simpler-WidowX `73.7%`
  - RoboTwin-Easy/Hard `86.1%/87.2%`
  - real-world ALOHA OOD 平均 `76.9%`

#### 对我们的影响

大统一模型会持续挤压“单纯提高 VLA 成功率”的空间。但它也放大了推理优化的重要性：

- 模型越大，越需要 dynamic compute
- 多任务统一后，越需要根据任务/风险选择推理路径
- agentic policy 可以作为大 VLA 的 inference controller，而不是替代大 VLA

## 8. 重新定位：如何保证我们的研究仍有价值

结合最新进展，原先的方向：

> Memory-grounded agentic inference for long-horizon VLA manipulation

建议升级成：

> Efficient Agentic Policy for Long-Horizon Embodied Manipulation

或者更具体：

> Budget-aware Agentic VLA Inference with Dynamic Verification, Adaptive Action Commitment, and Memory-grounded Recovery

## 8.1 核心科学问题

不是简单问：

> 如何让 VLA 做长程任务？

而是问：

> 在有限推理预算和实时控制约束下，agentic policy 应该何时思考、何时复用、何时验证、何时执行长动作、何时截断并恢复？

这个问题与现有工作相比更聚焦，也更贴合你的硕士课题。

## 8.2 可形成论文贡献的模块

### 贡献 1：Budget-aware Agentic Controller

输入：

- subgoal state
- action entropy / consensus
- verifier confidence
- memory retrieval score
- recent failure count
- latency budget

输出：

- call VLA / reuse cognition
- call verifier / skip verifier
- execute chunk length
- accept action prefix length
- invoke recovery / transition / replan

### 贡献 2：Dynamic Verification Scheduling

区别于 Agentic Robot/HELM 的固定或近似固定 verifier 调用：

- 低风险阶段少验证
- 接触、抓取、放置、遮挡阶段多验证
- entropy 高时多验证
- memory 中同类失败多时多验证
- verifier 自身低置信时调用更强 verifier 或 VLM

### 贡献 3：Adaptive Action Commitment

结合 AAC/A3：

- 用 entropy 决定候选 chunk 长度
- 用 internal consensus 接受动作前缀
- 用 external state verifier 检查物理风险
- 执行最长安全前缀

### 贡献 4：Memory-grounded Lightweight Recovery

不是每次失败都让大 VLM replan：

- 先检索历史同类失败
- 先尝试轻量 recovery primitive
- 只有连续失败或 verifier 判断状态异常时调用大模型

## 8.3 推荐题目

可以考虑以下题目：

1. `Efficient Agentic Policy for Long-Horizon Vision-Language-Action Manipulation`
2. `Budget-Aware Agentic VLA Inference via Dynamic Verification and Adaptive Action Commitment`
3. `Memory-Grounded Efficient Inference for Agentic Robot Manipulation`
4. `When to Think, Verify, and Act: Adaptive Inference Scheduling for Agentic VLA Policies`

我个人最推荐第 2 个，因为它同时体现：

- agentic policy
- VLA inference
- dynamic verification
- action commitment
- 推理优化

## 8.4 第一阶段实验设计更新

在原来的 baselines 基础上，新增推理优化指标：

- `Avg VLA calls / episode`
- `Avg verifier calls / episode`
- `Avg planner calls / episode`
- `Accepted action prefix length`
- `Latency per control step`
- `Wall-clock task completion time`
- `FLOPs / episode`
- `Success per unit compute`
- `Recovery success per verifier call`

新增 baselines：

- fixed verification every N steps
- fixed action chunk length
- entropy-based AAC only
- A3-style action acceptance only
- HELM-style memory-verifier-recovery
- ours: budget-aware dynamic scheduler

## 8.5 最小可行创新版本

第一版不需要改 VLA 权重，可以做 training-free 或少训练版本：

1. 在现有 VLA 外包一层 agentic controller
2. 从 VLA action distribution / multi-sample actions 计算 entropy 或 consensus
3. 用轻量 verifier 判断 subgoal 风险
4. 根据风险动态调整：
   - chunk length
   - verifier frequency
   - recovery trigger
5. 记录 memory，并让 recovery 优先复用历史有效策略

这能避开大规模训练，又能和最新进展对话。

## 8.6 最重要的结论

截至 2026-06-03，`memory + verifier + recovery` 已经不再是足够新颖的单独贡献。真正有价值的切口是：

> 在 agentic policy 框架下，把长程 manipulation 的鲁棒性问题和具身大模型推理优化问题统一起来。

也就是让系统不仅“更聪明”，还要知道：

- 什么时候不必思考
- 什么时候不必验证
- 什么时候可以长 chunk 执行
- 什么时候必须短 chunk 闭环
- 什么时候用小模型
- 什么时候调用大模型
- 什么时候复用 memory
- 什么时候重新规划

这条路线既跟上了最新 Agentic VLA/WAM 进展，也保留了清晰、仍待解决、且和你硕士课题高度一致的研究价值。

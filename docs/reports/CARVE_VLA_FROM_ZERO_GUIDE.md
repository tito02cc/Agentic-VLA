# 从零理解 CARVE-VLA：Agentic Harness 与 VLA 高效推理

更新时间：2026-07-20

本文档面向项目作者本人，用于在中期汇报前建立一套完整、准确的技术心智模型。
它回答四个问题：项目解决什么问题，Agentic Harness 如何工作，Optimize
Runtime 优化了什么，以及现有实验究竟证明了什么。

## 1. 一句话理解项目

CARVE-VLA 是冻结 VLA 外部的部署系统：它一方面监督机器人长程执行并处理
失败，另一方面为每次 VLA 调用选择经过验证的低时延计算配置。

```text
CARVE-VLA = Agentic execution supervision + admitted efficient VLA runtime
```

项目不重新训练 PI0.5 基座模型，也不试图让一个语言 Agent 直接控制每个关节。
PI0.5 仍然负责从视觉、语言和本体状态生成动作；CARVE 负责决定何时调用它、
使用什么计算配置、何时复用动作、何时恢复，以及什么时候必须安全停止。

## 2. 为什么冻结 VLA 仍然需要外部系统

单次 VLA 推理可写成：

```text
observation + instruction -> action chunk
```

但长程机器人任务实际上是循环：

```text
observe -> infer -> execute -> environment changes -> observe again
```

在这个循环中，冻结 VLA 会遇到三类部署问题。

### 2.1 行为可靠性问题

- 动作执行后物体没有按预期移动；
- 夹爪闭合但没有抓住物体；
- 机器人在局部状态中反复输出近似动作；
- 缓存动作来自过旧观测；
- 长程任务中前一个子目标失败，后续动作仍继续执行。

VLA 本身输出动作，但通常不提供完整的失败生命周期、恢复预算和安全停止契约。

### 2.2 计算成本问题

先不考虑任何优化术语。机器人每做一次决定，电脑大致要完成下面的工作：

```text
拍摄图像 -> 整理图像和机器人状态 -> VLA 计算动作
         -> 将动作发给机器人 -> 执行一小段动作 -> 再次观察
```

这个循环并不是免费的。这里的“计算成本”不是单指金钱，而是主要指
以下三种资源：

1. **时间（latency）**：从看到画面到得到动作，需要等待多久。
2. **显存（GPU memory / VRAM）**：模型参数和中间计算结果需要占用多少
   GPU 空间。显存不足时，模型可能无法启动或直接报错。
3. **计算量（computation）**：GPU 要进行多少次数学运算。计算量越大，
   通常等待时间和耗电量也越大。

#### 问题一：PI0.5 为什么不是一次计算就结束

PI0.5 生成动作时采用了 flow-based 方法。可以暂时将它理解为：模型先从
一份非常混乱的“动作草稿”开始，再在模型内部连续修改多次，最终得到
可执行动作。这个过程称为多步去噪。

“多步”指模型内部的多轮计算，不是机器人在环境里走了多步。每增加
一轮内部计算，GPU 都要做更多工作。

#### 问题二：Agentic retry 和 replan 为什么会变慢

- **retry（重试）**：动作失败后，重新请求 VLA 生成动作。
- **replan（重新规划）**：当原来的执行方案不再适用时，根据新观察重新决定
  下一步怎么做。

它们能提高任务成功的可能性，但每重试一次，就可能要重新运行一次
完整 VLA 推理。如果没有重试次数和时间预算，Agentic 机制虽然更会处理
失败，却可能反复调用大模型，导致机器人长时间停在原地等待。

#### 可选扩展：当系统同时使用 VLM 时会发生什么

- **VLM** 主要用来理解图像和语言，例如判断“杯子是否已经放到盘子上”。
- **VLA** 主要用来从图像、语言和机器人状态生成动作。

当前 CARVE 的主闭环并不强制使用额外 VLM。主要的 PI0.5 恢复实验使用
规则化风险监测、机器人状态和物理恢复控制，没有启用 VLM critic 或 VLM
planner。

项目另外测试过一个 Qwen3.5-4B 语义观察器，用来研究未来加入语义检查时
的资源影响。它目前是可选、异步且 shadow-only 的：它会输出判断并记录日志，
但不直接改变机器人动作。

2026-07-20 新增的 Guarded High-Level Agent 已将外部 Qwen3.5-4B 接入任务开始
和重复失败升级边界。它已通过真实 LIBERO 图像 smoke，但尚未完成新的
闭环成功率对照。因此，旧主实验仍是“无外部 VLM 的确定性控制证据”，
新高层 Agent 目前应表述为“框架和真实模型链路已打通”。

当两个模型同时使用同一张 GPU 时，它们会竞争显存和计算单元。可以把
GPU 理解为一个只有有限灶眼的厨房：两道菜同时做时，一道菜占用灶眼，
另一道菜就可能需要等待。因此，单独测试 VLA 很快，不代表它与 VLM
同时运行时仍然很快。这是 CARVE 为可选语义模块预先测试的部署问题，不是
当前 Agentic 主线成立的前提。

#### 问题四：什么是 deadline，为什么只测模型不够

**deadline（截止时间）**是指一次控制决策最迟必须在多长时间内完成。
超过这个时间，机器人看到的可能已经是旧状态，原本合理的动作也可能
变得不合理。

一次完整控制的等待时间包含：

```text
总时间 = 图像处理 + 数据传输 + 排队等待 + VLA 推理
         + 动作后处理 + 与仿真器或机器人通信
```

例如，假设一次决策必须在 `100 ms`（毫秒）内完成，VLA 本身用了
`60 ms`，但其他步骤合计用了 `50 ms`，那么总时间是 `110 ms`，仍然没有
满足 deadline。这些数字只是教学示例，不是本项目的实验结果。

因此，CARVE Optimize Runtime 不只是让模型某一次跑得更快，而是希望：

1. 减少不必要的模型内部计算；
2. 在 VLA、Agentic 重试和可选 VLM 之间分配有限计算资源；
3. 测量从观测到动作的完整时间；
4. 在优化可能改变动作质量时，回退到已验证的配置。

### 2.3 部署可信度问题

仅测平均 latency 或开放环 action error 不够。一个配置只有同时满足以下条件，
才可以进入机器人闭环：

1. 动作空间、归一化和夹爪语义一致；
2. 配对观测上的动作 fidelity 合格；
3. P50、P95、显存和 deadline miss 合格；
4. 长程闭环结果不劣于参考配置；
5. 不满足前提时能回退到已验证配置。

## 3. 总体架构

CARVE-VLA 分为两个边界明确的子系统。

```text
Instruction + RGB/Wrist RGB + Proprioception
          |                         |
          |             task start / repeated failure only
          |                         v
          |            Guarded High-Level VLM Agent
          |              typed intent / subgoal / skill
          |                         |
          +-------------------------+
                        v
              CARVE Agentic Harness
  monitor / structured memory / verifier / recovery / safe stop
                        |
          InferenceRequest + risk + deadline
                        v
              CARVE Optimize Runtime
    capability / profile / fidelity / backend / fallback
                        |
                        v
                  Frozen VLA
                        |
                 bounded action chunk
                        v
             robot or physics simulator
                        |
             observation + execution trace
                        +---------------------> loop
```

Agentic Harness 解决“机器人现在应该采用哪种执行模式”；Optimize Runtime 解决
“这次 VLA 推理允许使用哪种经过验收的计算配置”。二者通过显式 contract 和
trace 连接，不允许优化后端静默改变动作语义。

## 4. CARVE Agentic Harness

主要代码位于 `agentic_vla/runtime/`。

这些名称表示系统功能，不代表每个功能都是一个大模型。当前主实验中：

| 模块 | 当前实现 | 是否使用外部 VLM |
|---|---|---|
| High-Level Agent | 任务开始或重复失败时选择 VLA、白名单 skill 或停止 | 是；已通过真实图像 smoke，待闭环验证 |
| Monitor | 基于 RGB、机器人状态、动作历史和时间的检查逻辑 | 否 |
| Memory | 结构化的执行记录、失败类型和恢复结果 | 否 |
| Verifier / Critic | 根据 monitor 证据和预算规则决定是否干预 | 否；可选接入 VLM |
| Recovery | 有限步的确定性物理恢复 skill，之后重新调用 VLA | 否 |
| Safe stop | 预算耗尽或条件不安全时的状态机分支 | 否 |

因此，快速执行与安全停止仍不依赖外部 VLM；高层 VLM 只在稀疏边界上提供语义
决策。它的输出必须经过 schema、置信度、调用预算和 skill 白名单检查，不能
直接生成关节动作。

### 4.1 Policy 与 Action Contract

在执行前检查：

- observation 中有哪些命名相机视角；
- state/action 维度；
- action chunk 长度；
- 坐标系和夹爪开合语义；
- 模型支持的精度、编译和缓存能力；
- 输出动作是否有限值且满足边界。

它的意义是让 PI0.5、OpenVLA 或未来其他 VLA 通过适配器接入，而不是让控制器
依赖某个模型的私有张量结构。

### 4.2 Execution Monitor

Monitor 使用可部署信号判断当前执行是否异常：

- RGB 和 wrist RGB；
- 末端执行器、夹爪和 proprioception；
- 历史动作与动作重复程度；
- action age；
- 实测 latency 和 deadline slack。

仿真器物体真值只用于实验标签，不用于控制决策。这个边界决定了框架未来能够
迁移到实机，而不是依赖 MuJoCo 的隐藏状态“作弊”。

### 4.3 Joint Controller

控制器根据风险、预算和 runtime 状态在以下模式中选择：

```text
reuse cached action
        |
normal VLA inference
        |
bounded replan or prompt retry
        |
physical recovery skill
        |
verified replan
        |
safe stop
```

核心思想不是“失败就无限 retry”，而是事件触发、分级干预和有界计算。

### 4.4 Memory

当前 memory 保存的是执行级信息，而不是无限增长的对话历史：

- 当前子目标与恢复 session；
- 已使用的 retry/recovery budget；
- 最近失败类型和采取的恢复动作；
- 恢复是否通过验证；
- 可检索的任务先验和历史恢复结果。

Memory 用于避免重复失败和重复调用，不承担底层动作生成。

### 4.5 Verifier/Critic、Retry 与 Physical Recovery

在当前主实验中，Verifier/Critic 不是 VLM，而是根据 monitor 证据、当前执行
阶段和剩余预算决定是否干预的确定性逻辑。框架允许未来将外部 VLM 接入
为额外语义证据，但它不是当前恢复闭环的必需项。

Prompt retry 会让 PI0.5 VLA 根据新提示重新生成动作；physical recovery 则执行
少量确定、受约束的物理 skill，例如：

```text
stabilize -> retract -> lift -> reobserve -> verify -> VLA replan
```

恢复动作必须有限幅、有限步，并在重新调用 VLA 前验证环境是否产生了预期响应。
不支持的事件或预算耗尽时进入 safe stop。

## 5. CARVE Optimize Runtime

主要代码位于 `agentic_vla/optimization/`。它不是单一量化脚本，而是一套
profile-driven 的推理优化与部署验收系统。

### 5.1 Deployment Profile

每个 profile 绑定：

- 模型和 checkpoint；
- 输入 adapter 与相机视角；
- GPU 和软件 backend；
- flow steps、commit horizon 和精度；

- 编译、SMVE 或量化配置；
- fidelity、latency、显存和闭环证据；
- fallback profile。

运行时只能使用已 admitted profile。输入条件变化或检查失败时，自动回退到
普通 compiled BF16，而不是继续使用不安全优化。

### 5.2 Flow-Step Calibration

PI0.5 使用 flow matching 生成动作。减少 flow steps 可以直接减少每次动作生成
的模型计算，但过度减少会改变动作并损害闭环结果。

CARVE 在固定恢复状态上比较不同 steps 和 commit horizon，选择满足闭环非劣的
最低计算配置。当前从 7 steps 降到 2 steps：

- 配对成功均为 `14/15`；
- 单次 VLA 从 `366.90 ms` 降到 `148.96 ms`；
- 加速 `2.46x`；
- 每回合总时间降低 `32.3%`。

### 5.3 Torch Compile

对固定 shape 的 PI0.5 PyTorch 路径预热并编译，避免将首次编译时间混入在线
latency。RTX 4090 上：

| Profile | P50 | P95 | Miss@80ms |
|---|---:|---:|---:|
| Eager BF16 | `154.34 ms` | `159.59 ms` | `100%` |
| Compiled BF16 | `65.73 ms` | `67.40 ms` | `0%` |

这证明的是稳定调用路径的高效推理，不是模型参数压缩。

### 5.4 Static Masked-View Elision

LIBERO adapter 提供 base 和 left-wrist 两路有效图像，并为缺失的 right-wrist
填充全零图像和 `mask=False`。原始 PyTorch 路径仍可能先编码该无效视角。

SMVE 在每次调用中确认目标 mask 全假，然后在 SigLIP 编码和 prefix KV 构建前
删除该 padding 视角：

```text
3 image tensors -> assert one named view is fully masked -> encode 2 active views
```

它不缓存旧视觉、不删除有效 token，也不跨时间复用观测。当前结果：

- replay fidelity `45/45`；
- 最差 chunk MAE `0.00232`；
- 最差 cosine `0.999897`；
- gripper agreement `1.0`；
- P95 从 compiled 的 `67.40 ms` 降到 `56.19 ms`；
- T8/T9 为 `8/10`，普通 compiled 为 `7/10`，只解释为闭环非劣。

### 5.5 Deadline 与共卡压力

空闲 GPU 的 latency 不代表 Agentic 系统中的 latency。Qwen3.5-4B 视觉请求与
PI0.5 同卡时：

| Profile | PI0.5 P50/P95 | Miss@80ms | Fidelity |
|---|---:|---:|---:|
| Compiled BF16 | `81.12/91.85 ms` | `72.8%` | `10/10` |
| Compiled + SMVE | `65.93/75.57 ms` | `0.6%` | `10/10` |

这组实验体现了 Optimize Runtime 与 Agentic 系统耦合的实际价值：优化必须在
VLM/VLA 共卡和实时压力下仍然成立。

## 6. Agentic 与 Optimize 如何耦合

两部分不是简单拼接。恢复会增加推理调用，推理时延又会影响动作新鲜度和控制
决策。CARVE 使用同一份 trace 同时记录：

- monitor 触发原因；
- controller 选择的执行模式；
- 请求和实际应用的 deployment profile；
- fallback；
- model-call 与 control-step latency；
- deadline miss；
- recovery verification 和最终任务结果。

联合实验在相同 T6/T9 stall 状态、相同恢复动作和固定噪声下，只替换执行
profile：

| Profile | Exact success | Recovery verified | Runtime P95 | Miss@80ms |
|---|---:|---:|---:|---:|
| Eager BF16 | `2/2` | `2/2` | `166.26 ms` | `236/236` |
| Compiled BF16 | `2/2` | `2/2` | `65.75 ms` | `0/239` |
| Compiled + SMVE | `2/2` | `2/2` | `54.50 ms` | `0/247` |

这组实验说明：恢复行为保持时，Optimize Runtime 可以显著降低 Agentic 执行中
的模型调用时延和 deadline miss。

## 7. 轻量化目前做到什么程度

“轻量化”需要拆成不同层次。

### 已完成并具有正结果

- 推理计算轻量化：降低 flow steps；
- 视觉计算轻量化：SMVE 跳过确定无效视角；
- 运行时优化：编译、预热、profile 和 fallback；
- 系统调用轻量化：事件触发而非持续 replan/retry；
- deadline-aware 部署门控。

### 已实验但被否决

| Profile | 显存 | 时延/质量 | 结论 |
|---|---:|---|---|
| PI0.5 W8A16 | `6.98 -> 6.56 GB` | P95 变慢且闭环少一次成功 | rejected |
| OpenVLA INT8 | `14.42 -> 7.76 GB` | P50 大幅变慢，exact action `50%` | rejected |
| OpenVLA NF4 | `14.42 -> 4.41 GB` | P50 变慢，exact action `10%` | rejected |

因此当前不能声称完成了可部署的参数量化算法。正确表述是：模型量化管线和
验收方法已完成，但尚未得到同时通过显存、时延、fidelity 和闭环门控的低比特
PI0.5 profile。

## 8. 实验证据层级

### 当前主证据

1. PI0.5 flow-step 配对校准；
2. RTX 4090 profile latency、显存和 deadline；
3. 45-state SMVE replay fidelity；
4. T8/T9 同步闭环非劣实验；
5. Recovery Challenge；
6. Agentic-Optimize 精确状态联合实验；
7. VLM/VLA 共卡压力；
8. 量化、异步和 semantic intervention 负结果。

### 历史背景

历史 LIBERO-10 汇总为 baseline `180/200`、CARVE `185/200`，但原始 rollout
在早期清理时未保留，因此只作为背景，不能作为当前最强可复现主证据。

### 当前缺口

- 没有物理机器人实机实验；
- 参数量化尚无 accepted 正结果；
- Agentic 恢复状态数量仍小，不支持 benchmark-wide 显著性声明；
- 第二 VLA 家族只有接口、开放环和 profile 验证，没有闭环任务成功率。

## 9. 中期汇报应如何讲

推荐主线：

1. 冻结 VLA 已有强动作能力，但长程执行和实时部署仍存在系统问题；
2. CARVE 用 Agentic Harness 建立可监测、可恢复、可停止的闭环；
3. Agentic 增加推理压力，因此需要 Optimize Runtime；
4. flow-step、compile 和 SMVE 构成当前正向高效推理方案；
5. deployment admission 用 fidelity 和闭环结果拒绝“看似更轻”的错误方案；
6. 联合实验表明恢复结果保持，同时 P95 从 `166.26` 降到 `54.50 ms`；
7. 实机与 action-fidelity-aware mixed precision 是下一阶段。

不要将故事讲成两个互不相关的项目，也不要把 Agentic 描述成语言模型不断调用
工具。这里的 Agentic 是面向机器人执行生命周期的监督、恢复和计算分配系统。

## 10. 汇报前自测问题

能够独立回答以下问题，就说明已经理解项目：

1. VLA、Agentic Harness 和 Optimize Runtime 分别负责什么？
2. 为什么 retry 越多不一定越好？
3. Monitor 为什么不能使用仿真器物体真值？
4. SMVE 删除了什么，为什么不会删除有效视觉信息？
5. 为什么只报告平均 latency 不够？
6. 为什么 W8A16 显存下降仍被否决？
7. 联合实验如何证明 Agentic 与优化不是简单拼接？
8. 当前“轻量化已完成”与“参数量化未完成”如何同时成立？
9. 当前视频为什么只能称为 MuJoCo 闭环视频而不是实机视频？
10. 下一阶段最小而有价值的实验是什么？

## 11. 关键材料索引

- 完成度：`docs/status/CARVE_VLA_COMPLETED_WORK.md`
- 中期报告：`docs/archive/midterm/20260717/CARVE_VLA_MIDTERM_REPORT_20260717.md`
- 中期问答：`docs/archive/midterm/20260717/CARVE_VLA_MIDTERM_QA_20260717.md`
- Runtime 架构：`docs/architecture/CARVE_RUNTIME_ARCHITECTURE.md`
- Optimize Runtime：`docs/architecture/CARVE_OPTIMIZE_RUNTIME.md`
- 统一结果：`paper/CARVE-VLA/generated/runtime_results_summary.json`
- 紧凑证据：`results/CARVE_EVIDENCE_README.md`
- 当前论文：`paper/CARVE-VLA/root.pdf`

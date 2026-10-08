# CARVE-VLA 简历与跨会话技术交接文档

> **2026-08-31 更新：**本文记录了早期 LIBERO/LIBERO-PRO 阶段。用于当前
> 简历时，请优先读取 `docs/reports/ASU_STYLE_RESUME_PROJECTS.md`；其中已纳入
> RoboMME 8 任务、240 个配对闭环 rollout 的最新主结果。

> 用途：将本文件与 `root.pdf` 提供给另一台电脑上的 Codex，用于生成简历项目描述、面试材料或技术总结。
>
> 事实口径：本文只汇总当前仓库中已有实现和实验。凡标注为“资格实验”“描述性证据”或“负结果”的内容，不得改写成正式 benchmark 提升。
>
> 当前状态：研究原型与详细技术报告已完成，论文仍属于在研项目，不应表述为已发表成果。

## 1. 给下一位 Codex 的直接指令

请先完整阅读本文件，再阅读同目录下的 `root.pdf`。随后可以根据目标岗位生成中文或英文简历项目经历，但必须遵守第 11 节的声明边界。

生成简历时建议先询问目标岗位、项目允许篇幅，以及希望突出研究创新、系统工程还是性能优化。不要把历史 deterministic Harness 的 `185/200` 归因于后来接入的 VLM Planner；不要把局部 LIBERO-PRO 资格实验写成完整榜单结果；不要声称已经完成真机部署、提出新量化算法或实现所有 VLA 的通用实时加速。

## 2. 项目身份与一句话概述

### 2.1 名称

- 项目名：**CARVE-VLA**
- 英文展开：**Compute-Aware Agentic Runtime for VLA Execution**
- 推荐中文名称：**面向冻结 VLA 的智能体执行框架与高效推理运行时**
- 关键词：Embodied Agent、Vision-Language-Action、Agentic Harness、VLM Planner、Long-Horizon Manipulation、Efficient Inference、Runtime Optimization、Deployment Admission

### 2.2 一句话概述

CARVE-VLA 是一个围绕冻结 VLA 构建的模型可替换推理系统：Agentic Harness 负责高频风险监测、事件触发语义规划、结构化记忆、有限物理恢复和安全停止，Optimize Runtime 负责计算预算校准、编译与视觉计算裁剪，并通过动作保真、运行时和闭环任务三重门控决定优化 profile 是否可部署。

### 2.3 项目问题

强 VLA 通常已经具备局部 reach/grasp/place 能力，但长程操作仍可能因子任务衔接、执行漂移、接触失败和恢复失败而中断。外接 Agentic 模块可以补充过程控制，却会引入额外 VLA/VLM 调用、时延和显存开销。因此本项目不重新训练基础策略，而是联合解决两个问题：

1. 如何在不修改 VLA 参数的情况下，检测执行异常并安全地恢复或重新规划；
2. 如何在保留动作和闭环行为的前提下，让包含 VLA 与低频 VLM 的 Agentic 推理链满足部署时延与资源约束。

## 3. 论文主线与核心判断

论文故事不是“给 VLA 套一个通用 Agent”，也不是“单独做模型压缩”，而是：

1. **VLA 的残余长程失败常是 process-control failure。** 局部动作能力存在，但系统缺少对进展停滞、动作过期、子任务边界和恢复时机的显式管理。
2. **Agentic 增强必须有清晰的控制权边界。** 高频 Monitor 只检测执行症状；低频 VLM Planner 只产生受约束的语义意图；只有 VLA 和已注册物理技能能够生成低层动作。
3. **Agentic 系统天然产生推理优化需求。** 高频 VLA 调用与低频 VLM 语义调用共享有限硬件，盲目 retry、量化或异步化都可能损害闭环行为。
4. **优化 profile 不能只看单次 latency。** 每个候选必须同时通过接口契约、动作保真、稳态运行时和配对闭环结果，才能进入在线系统。

因此，CARVE-VLA 的主要价值是把可靠执行与高效推理放进同一套可审计运行时，而不是分别报告一个 Agent 模块和一个孤立的加速数字。

## 4. 系统总体架构

系统由两个可独立使用、也可联合运行的部分组成。

### 4.1 CARVE Agentic Harness

Harness 在冻结 VLA 外部实现多速率闭环：

- **高频控制路径**：执行 VLA action chunk，并由 deterministic Monitor 每个控制步检查风险；
- **中频动作路径**：队列即将耗尽或经过恢复后，调用 VLA 生成新的低层动作块；
- **低频语义路径**：只有确认事件且机器人到达安全边界时，才调用 VLM Critic/Planner；
- **异常路径**：执行已注册的有限物理恢复技能，验证响应，再重新调用 VLA；
- **失效路径**：超时、陈旧结果、格式错误、预算耗尽或不支持的异常进入 `SAFE_HOLD/STOP`。

### 4.2 CARVE Optimize Runtime

Optimize Runtime 将模型、checkpoint、机器人输入、精度、后端和硬件绑定成可复查 deployment profile，提供：

- 统一 `PolicyAdapter`、`ModelCapabilities` 和 `ActionSpec`；
- flow-step 与 action-commitment 校准；
- PyTorch `torch.compile` 推理后端；
- Static Masked-View Elision（SMVE）；
- W8A16、INT8、NF4 等量化候选测试；
- 模型、runtime、reaction、task-cycle 和 deadline trace；
- 动作保真、稳态性能与 paired closed-loop admission；
- 仅针对明确 mask-contract violation 的窄 fallback，避免吞掉 OOM、NaN 或其他系统错误。

## 5. Agentic Harness 技术细节

### 5.1 权限分离

| 模块 | 输入 | 输出 | 是否能直接控制机器人 |
|---|---|---|---|
| Execution Risk Monitor | 动作、proprioception、图像变化、action age、deadline | 风险分数、事件和证据 | 否 |
| VLA Runtime | 图像、机器人状态、原始任务与局部子目标 | 经验证的低层 action chunk | 是，唯一 learned action generator |
| VLM Critic/Planner | 任务、帧、Monitor 证据、记忆、场景图、预算和技能白名单 | typed semantic intent | 否 |
| Recovery Skill | 事件、当前状态、动作契约 | 有界解析动作序列 | 是，但只能在技能 envelope 内 |
| Memory / HAA-RAG | 失败记录、场景和任务语义 | 证据卡片 | 否 |

VLM 不能输出轨迹、关节目标、力矩或自由形式 robot action。它只能选择四类意图：

- `continue`：保留当前目标；
- `vla_act`：生成一个经 gate 验证的局部语义子目标，再调用 VLA；
- `run_skill`：选择技能白名单中的一个 registered skill；
- `safe_stop`：进入安全保持或终止。

原始任务指令始终保留，VLM 的局部指导只能以“current recovery subgoal”附加，不能替换原任务。

### 5.2 Execution Risk Monitor

Monitor 是高频、低成本、确定性的执行风险监测器，不是 VLM，也不做语义诊断。它比较机器人被要求产生的动作与实际观察到的响应：commanded motion、proprioceptive response、visual response、action age、可选 model uncertainty 和 deadline pressure。

默认滑动窗口为 4，command threshold 为 `0.03`，proprioceptive response threshold 为 `0.002`，visual response threshold 为 `0.004`，stale-action threshold 为 8，deadline slack 为 30 ms。风险聚合为：

```text
R = 0.45 * stall + 0.20 * action_age
  + 0.20 * uncertainty + 0.15 * deadline_pressure
```

风险桶阈值为 `0.35/0.65`。LIBERO-PRO 资格实验使用 60 control-step warmup，以避免起步瞬态误触发。Monitor 只回答“执行是否出现异常症状”，VLM Planner 才回答“当前语义任务是否正确、下一步应采用何种允许的处理方式”。

### 5.3 状态机与安全边界

Canonical Harness 包含六个状态：

| 状态 | 作用 |
|---|---|
| `EXECUTE_FAST` | 执行已验证 VLA chunk 或缓存后缀 |
| `VERIFY` | 确认风险证据并检查剩余预算 |
| `RECOVER` | 执行一个 registered bounded skill 并验证响应 |
| `PLAN_AT_SAFE_BOUNDARY` | 提交或轮询一个 typed VLM 请求 |
| `SAFE_HOLD` | 执行 adapter 定义的静止保持、心跳和超时策略 |
| `STOP` | fail-closed 终态 |

Planner 请求采用 single-flight ticket，并绑定 episode/timestep。陈旧、超时、低置信度、超预算、未知 intent/skill、包含原始动作字段或 JSON 格式错误的结果均被拒绝。

### 5.4 Structured Failure Memory 与 HAA-RAG

Structured Failure Memory 存储带类型和有效期的 `FailureEpisodeRecord`，包括 context/profile identity、failure evidence、action age、intervention budget、verification、outcome、confidence、timestamp 与 expiry。检索受到上下文、失败类型、profile 和有效期限制。Memory 可以改变 planner context 或抑制已失败干预，但不能直接发动作。

场景知识采用定性 scene graph，记录 inside、near、supported-by、grasped-by 等关系，不读取 simulator reward、success predicate、metric object pose 或 ground-truth trajectory。HAA-RAG 保存对象类别、可供性、抓取区域、材质、脆弱性、适用失败类型、约束与历史结果。当前训练无关检索分数为：

```text
score = 2 * category_overlap + 1.5 * failure_match + 0.25 * confidence
```

当存在任务或场景类别匹配项时，系统抑制仅因 failure type 相同但对象无关的卡片，避免无关历史失败支配当前决策。

### 5.5 有限物理恢复

当前主技能 `cartesian_retract_lift_reobserve` 用于 preserve-grasp 恢复，声明 action spec、前置条件、支持的失败、超时、最大动作数、验证规则和 fail-closed 行为。报告配置共 12 个动作：2 个 stabilize、4 个 retract、4 个 lift、2 个 settle/reobserve。

技能完成后必须依据 end-effector 的实际响应验证，验证通过才允许重新调用 VLA。技能恢复“可控状态”和最终任务成功是两个独立指标。默认 registry 还包含 release-and-retreat 变体，但不能据此声称已经形成通用 physical-skill library。

## 6. Optimize Runtime 技术细节

### 6.1 Profile 与准入协议

每个 VLA profile 绑定 model family、checkpoint digest、hardware identity、robot/input adapter、action semantics、view contract、precision、inference backend、flow-step count 和 action horizon/commitment。

候选 profile 必须依次通过：

1. **Contract gate**：模型、输入、动作、视角和能力声明一致；
2. **Fidelity gate**：比较 first action、完整 chunk、cosine、gripper mode、累计 endpoint 和 jerk；
3. **Runtime gate**：区分冷启动和稳态，测 P50/P95/P99、deadline miss、VRAM 与共驻干扰；
4. **Closed-loop gate**：从同一 simulator state 恢复，验证任务、恢复与 safe-stop 结果。

只有 promoted manifest 可以被在线 runtime 选择。Replay fidelity 通过但闭环退化的候选必须拒绝。

### 6.2 计算预算校准

PI0.5 是 flow-matching VLA。项目将 flow sampling step 与 action commitment 分开校准，以避免把“少采样”和“少执行”混为一谈。固定 commitment=10 时，从 7 flow steps 降到 2 flow steps 保持 `14/15` 配对成功，并降低模型计算。动态 2/4-step 控制器只有 `13/15`，因此部署默认选择简单的 2-step profile。

### 6.3 编译与预热

项目为 PI0.5 PyTorch 路径建立预热后的 compiled BF16 profile。编译准备成本单独报告，不计入 steady-state latency。在线服务在首次正式请求前完成 profile preparation，避免把多分钟编译落在机器人控制路径上。

### 6.4 Static Masked-View Elision（SMVE）

PI0.5 observation schema 可能包含永久 padding 的相机槽，其 mask 始终为 false。SMVE 在每次请求上验证该 view contract，随后在视觉 embedding 和 prefix 构造前移除无效 tensor，并编译缩短后的 sampler。

SMVE 不是任意删图，也不是动态视觉 token pruning。只有当 adapter 能证明某个命名视角始终为 padding 时才启用；若该视角意外变为 active，只允许 fallback 到另一个已经通过准入的 compiled-BF16 profile。OOM、NaN 和其他异常不会被 fallback 静默掩盖。

### 6.5 量化与异步候选

项目完成了量化和异步候选的工程评估，但没有把“显存降低”直接等同于“可部署”：

- PI0.5 W8A16 只量化 selected VLM language layers 0--3；replay fidelity 通过，但闭环丢失一次恢复成功且延迟更高，拒绝；
- OpenVLA INT8/NF4 显著降低显存，但 batch-one latency 变差且 action exact match 不足，拒绝；
- 风险门控异步 prefetch 大幅降低 deadline miss，却使 T8/T9 成功率下降，拒绝。

这部分体现的是量化、编译、调度、保真度验证和 deployment admission 能力，而不是已经提出新的量化算法。

## 7. 模型、技术栈与实验环境

### 7.1 模型

- **PI0.5**：主要 VLA，使用 OpenPI 的 `pi05_libero_pytorch` checkpoint；完整 Agentic 与 Optimize 联合路径均基于该模型。
- **Qwen3.5-4B BF16**：外部多模态 VLM Critic/Planner，通过 OpenAI-compatible endpoint 接入；只在安全边界低频调用。
- **OpenVLA-7B**：第二 VLA 家族，用于验证 `PolicyAdapter`、capability/action contract、编译和低比特 profile gate；未完成第二套完整闭环 Agentic benchmark。

### 7.2 软件和系统

- Python、PyTorch、OpenPI、MuJoCo、LIBERO、LIBERO-PRO；
- `torch.compile`、TorchAO W8A16、BitsAndBytes INT8/NF4；
- WebSocket / OpenAI-compatible model service；
- JSON/JSONL typed traces、profile manifest 与自动汇总脚本；
- 单张 NVIDIA RTX 4090 24 GB；
- 归一化 7-D delta Cartesian robot action；
- VLA policy observation 为 `256 x 256`，HD 演示只提高显示视频分辨率，不改变模型输入。

### 7.3 框架通用性准确表述

框架在接口层面是 model-neutral：任意 VLA 只要实现 `PolicyAdapter` 并声明 `ModelCapabilities` 与 `ActionSpec`，即可接入同一 Harness 和 profile-admission 协议。当前已经实际 exercise PI0.5 和 OpenVLA 两个模型家族的接口；只有 PI0.5 完成了 Agentic、优化 profile 与闭环恢复的完整准入链路。因此可写“设计模型可替换接口并验证两个 VLA 家族”，不能写“已在所有 VLA 上验证通用加速”。

## 8. 完整实验结果

### 8.1 历史 deterministic Harness：LIBERO-10

该实验比较的是早期 deterministic Harness precursor，不包含当前 canonical VLM Planner。每个方法覆盖 10 个任务、每任务 20 回合，即每方法 200 回合、合计 400 回合：

| 方法 | 成功率 | 成功回合 |
|---|---:|---:|
| Frozen `pi05_libero` | 90.0% | 180/200 |
| Deterministic Harness precursor | 92.5% | 185/200 |

主要弱任务 T8 从 `55%` 提升到 `75%`，即 `+20 percentage points`。该结果支持“冻结 backbone 的推理时过程控制可改善特定长程弱点”，不支持“所有任务均提升”。

弱任务诊断结果：

| 方法 | T6 | T8 | T9 |
|---|---:|---:|---:|
| Baseline | 80 | 55 | 95 |
| Original full | 80 | 40 | 90 |
| Task-targeted v2 | 90 | 70 | 100 |
| Transition only | 100 | 80 | 90 |
| Priors/memory only | 80 | 60 | 100 |
| Critic only | 90 | 70 | 90 |

主审计中没有真正触发 critic retry，因此不能把历史提升归功于 critic retry；这一接口只作为 audit/escalation interface 保留。

### 8.2 Canonical guarded-VLM handoff

在 T6/T9 两个恢复后的相同 MuJoCo 状态上，各使用 seeds 7/17/42，比较直接 VLA 续接与一次 guarded Qwen 决策：

| 分支 | 成功 | 平均执行步数 | 平均 VLA 调用 |
|---|---:|---:|---:|
| Physical recovery -> VLA | 5/6 | 258.7 | 123.7 |
| Physical recovery -> guarded VLM -> VLA | 6/6 | 250.3 | 119.2 |

只有 T9 seed 42 的 outcome 不同。每个 VLM 调用在安全边界增加约 `5.4--6.1 s`。该实验能证明真实 typed VLM handoff 和一个配对修复案例，但样本只有两个任务状态，不能声称统计显著的全局 VLM 成功率提升或 realtime 提升。

### 8.3 Flow-step calibration

T6/T8/T9 共 15 个配对状态：

| Profile | 成功 | 单次 VLA | VLA 时间/回合 | 总时间/回合 |
|---|---:|---:|---:|---:|
| 7 flow steps, commit 10 | 14/15 | 366.90 ms | 11.15 s | 21.27 s |
| 2 flow steps, commit 10 | 14/15 | 148.96 ms | 4.44 s | 14.40 s |
| Dynamic 2/4 steps, commit 10 | 13/15 | 158.88 ms | 4.98 s | 15.91 s |

2-step 相比 7-step 单次调用加速 `2.46x`，回合总时间降低 `32.3%`，同时保持配对成功数。

### 8.4 PI0.5 deployment profiles

RTX 4090，PI0.5，2 flow steps，commit 10，50 个测量调用（另有 warmup）：

| Profile | P50 | P95 | Miss@80ms | 峰值显存 | 决策 |
|---|---:|---:|---:|---:|---|
| Eager BF16 | 154.34 ms | 159.59 ms | 100% | 7.12 GB | reference |
| Compiled BF16 | 65.73 ms | 67.40 ms | 0% | 6.98 GB | default |
| Compiled BF16 + SMVE | 54.35 ms | 56.19 ms | 0% | 6.97 GB | padded-view default |
| Compiled W8A16 | 69.74 ms | 71.72 ms | 0% | 6.56 GB | closed-loop rejected |

从 Eager BF16 到 compiled BF16 + SMVE，P95 降低约 `64.8%`，相当于约 `2.84x` 的 P95 speedup，80 ms deadline miss 从 `100%` 降到 `0%`。Compiled BF16、SMVE 和 W8A16 均通过 45/45 paired replay fidelity，但 W8A16 因后续闭环恢复退化而被拒绝。

### 8.5 Agentic 与 Optimize 的同状态联合实验

在相同 T6/T9 physical-recovery states 上：

| Profile | 任务结果 | 已验证恢复 | Runtime P95 | 80ms miss |
|---|---:|---:|---:|---:|
| Eager BF16 | 2/2 | 2/2 | 166.26 ms | 236/236 |
| Compiled BF16 | 2/2 | 2/2 | 65.75 ms | 0/239 |
| Compiled BF16 + SMVE | 2/2 | 2/2 | 54.50 ms | 0/247 |

相对 Eager，compiled BF16 和 SMVE 的 runtime P95 分别降低 `60.5%` 和 `67.2%`；SMVE 相对 compiled BF16 再降低 `17.1%`。三个 profile 都保持 2/2 exact-state task outcome 与 2/2 verified recovery。这是推理优化确实加速 Agentic failure path，而不只是 standalone model call 的主要证据。

### 8.6 Recovery Challenge

三个异常快照包含 T6 stall、T9 stall 和不受当前技能覆盖的 T8 stale-action：

| 分支 | 成功 | VLA 调用 | 物理恢复 | Safe stop |
|---|---:|---:|---:|---:|
| Frozen continuation | 2/3 | 113 | 0 | 0 |
| Frequent VLA replan | 2/3 | 452 | 0 | 0 |
| Agentic prompt retry | 2/3 | 508 | 0 | 0 |
| Physical recovery + VLA replan | 2/3 | 251 | 2/2 verified | 1 |

频繁重规划和 prompt retry 增加约 4 倍 VLA 调用，但没有提高成功数。物理恢复只在两个受支持 stall 上执行，在不支持的 stale-action 上不发动作并 safe-stop。该实验强调 bounded recovery、skill coverage 和 fail-closed，而不是成功率提升。

### 8.7 异步 prefetch 负结果

| 调度 | 成功 | VLA 调用 | Deadline miss |
|---|---:|---:|---:|
| Synchronous commit-8 | 7/10 | 417 | 417/3931 = 10.61% |
| Full-duty async | 5/10 | 346 | 48/4287 = 1.12% |
| Alternating async | 5/10 | 315 | 176/4278 = 4.11% |

Full-duty async 将 deadline miss 相对降低 `89.4%`，但成功从 `7/10` 降到 `5/10`。原因不是 worker error，而是异步结果对应的 observation 时序改变后续轨迹，因此两个异步 profile 均被拒绝。

### 8.8 OpenVLA 第二模型家族

| Profile | P50/P95 | 峰值显存 | Fidelity/决策 |
|---|---:|---:|---|
| BF16 | 303.48/310.92 ms | 14.42 GB | reference |
| BitsAndBytes INT8 | 1533.96/1570.19 ms | 7.76 GB | exact action 50%，拒绝 |
| NF4 | 748.35/761.33 ms | 4.41 GB | exact action 10%，拒绝 |
| Prewarmed compiled LLM decode | 224.25/231.84 ms | 未作为低比特结果 | 10/10 exact action |

Compiled decode 的准备时间为 `200.19 s`，所有调用仍超过 100 ms。该结果证明框架能表达 autoregressive single-action VLA 的不同 capability，并获得 scoped speedup；不证明 OpenVLA 已达到 10 Hz 或完成 Agentic 闭环。

### 8.9 LIBERO-PRO 资格实验

这不是官方完整 protocol，而是为定位可用难度区间进行的 focused screen：

| Suite | 选定任务 | 成功 |
|---|---|---:|
| Clean | T3/T8/T9 | 7/9 |
| Object-OOD | T3/T9 | 4/6 |
| Task-OOD | T3/T8/T9 | 2/9 |
| Swap-OOD | T3/T8/T9 | 0/9 |

配对 clean subset 上 Frozen VLA、Fixed Recovery、Full Agentic 分别为 `7/9`、`8/9`、`8/9`。Full Agentic 完成一次真实 post-recovery VLM 调用，但 Fixed Recovery 也成功，因此只验证完整语义链路，不证明额外成功收益。

Semantic-first clean T9 三个 fixed-noise trials：Frozen VLA `2/3`，Fixed Recovery `3/3`（一次物理恢复），Semantic-First Agentic `3/3`（一次 VLM 调用、零物理恢复）。信息性 trial 在 step 94 检测 stall，检索 mug/microwave HAA cards，VLM 接受子目标 `close microwave door` 后完成任务。VLM 用时 `8.02 s`，PI0.5 平均调用 `54.26 ms`，说明多速率结构可把语义时延隔离到安全边界，但 VLM 自身仍是部署瓶颈。

Object-OOD T8 HD demo 完成 `3/3`。其中一个 rollout 在 step 337 自然触发 12-action preserve-grasp recovery 并完成任务；三次平均/P95 VLA latency 为 `54.54/58.07 ms`。该 demo 没有调用 VLM，单独展示 Monitor、bounded recovery 与优化 VLA 在可见对象变化下的闭环工作。

## 9. 可以体现的工程与研究能力

### 9.1 具身智能与 Agent

- 将 VLA 局部动作策略封装为可监督 physical skill；
- 设计 Monitor 与 VLM Planner 的高低频职责分离；
- 实现 typed intent、single-flight ticket、预算、超时、陈旧结果拒绝和 fail-closed；
- 构建结构化失败记忆、场景图、HAA-RAG 和技能白名单；
- 使用真实 MuJoCo 状态恢复做 paired counterfactual evaluation；
- 区分 simulator evaluator signal 与可部署 online signal，避免信息泄漏。

### 9.2 大模型推理与部署

- 对 flow-matching VLA 的 sampling steps 与 action commitment 做分离校准；
- 使用 `torch.compile`、预热、静态输入契约和视觉无效计算裁剪优化 batch-one inference；
- 评估 TorchAO W8A16、BitsAndBytes INT8/NF4 的显存、时延和动作保真；
- 设计 checkpoint/hardware-bound deployment manifest 和 capability negotiation；
- 记录模型、runtime、reaction、task-cycle、queue/action age 与 deadline trace；
- 用 replay fidelity 和 paired closed-loop gate 阻止 open-loop 看似正确但闭环已经退化的 profile 上线；
- 分析并拒绝异步 prefetch 的时序一致性退化。

### 9.3 系统工程

- PI0.5 与 OpenVLA 的统一 adapter/action contract；
- VLA WebSocket 服务与 OpenAI-compatible VLM service；
- 结构化 JSON/JSONL trace、summary 和论文表格自动生成；
- 实验状态、profile identity、seed/noise、视频和证据路径的可追溯管理；
- 在单张 RTX 4090 上完成 VLA/VLM 共驻与闭环仿真验证。

## 10. 可直接使用的简历候选表述

以下表述已经按事实边界压缩。最终应根据版面选 3 条，不必全部放入一份简历。

### 10.1 中文项目标题

**CARVE-VLA：面向冻结 VLA 的智能体执行框架与高效推理运行时（在研）**

### 10.2 中文三条版

- 面向冻结 PI0.5 设计模型可替换的多速率 Agentic Harness，集成高频执行风险监测、事件触发 VLM Planner、结构化失败记忆、有限物理恢复与 fail-closed 安全状态机，并以 typed intent/skill allowlist 隔离语义规划与低层机器人控制权限。
- 构建 evidence-gated VLA Optimize Runtime，通过 flow-step 校准、`torch.compile` 与静态无效视角裁剪（SMVE），将 PI0.5 稳态 P95 时延由 `159.59 ms` 降至 `56.19 ms`（降低 `64.8%`），80 ms deadline miss 由 `100%` 降至 `0%`，并保持 `45/45` 配对动作回放一致性。
- 在 LIBERO/LIBERO-PRO MuJoCo 中完成闭环验证：历史 deterministic Harness 在 400 回合对比中将 PI0.5 成功率由 `90.0%` 提升至 `92.5%`，弱任务 T8 提升 `20` 个百分点；同状态 Agentic-Optimize 实验保持 `2/2` 任务与恢复结果，并将 recovery-path P95 降低 `67.2%`。

### 10.3 中文四条工程版

- 设计 `PolicyAdapter`、`ModelCapabilities`、`ActionSpec` 和 checkpoint/hardware-bound profile manifest，在 PI0.5 与 OpenVLA 两个 VLA 家族上验证统一接口及差异化 capability contract。
- 实现 Monitor/VLM/VLA/physical-skill 多速率闭环、single-flight planner ticket、预算与超时、陈旧结果拒绝、safe hold 和结构化 trace，使 VLM 只输出受约束语义意图而不能直接控制机器人。
- 优化 PI0.5 batch-one 推理：2-step flow calibration 保持 `14/15` 配对成功并实现 `2.46x` 单次模型加速；compile+SMVE 将 P95 从 `159.59 ms` 降至 `56.19 ms`。
- 建立 fidelity-runtime-closed-loop 三重部署门控，基于闭环退化主动拒绝 PI0.5 W8A16、OpenVLA INT8/NF4 与异步 prefetch 候选，避免仅凭显存或 microbenchmark 指标上线不稳定优化。

### 10.4 English version

**CARVE-VLA: Agentic Execution Harness and Efficient Inference Runtime for Frozen VLAs (Ongoing Research)**

- Designed a model-replaceable, multi-rate Agentic Harness around frozen PI0.5, combining deterministic execution-risk monitoring, event-triggered VLM planning, typed failure memory, bounded physical recovery, and a fail-closed state machine with strict semantic/action authority separation.
- Built an evidence-gated VLA inference runtime with flow-step calibration, `torch.compile`, and Static Masked-View Elision, reducing steady-state PI0.5 P95 latency from `159.59 ms` to `56.19 ms` (`64.8%`) and 80-ms deadline misses from `100%` to `0%` while preserving `45/45` paired replay checks.
- Validated the system in LIBERO/LIBERO-PRO MuJoCo: a deterministic Harness precursor improved frozen PI0.5 from `90.0%` to `92.5%` over 400 comparative rollouts, while paired Agentic-Optimize tests preserved `2/2` task/recovery outcomes and reduced recovery-path P95 by `67.2%`.
- Implemented checkpoint/hardware-bound profile admission and rejected W8A16, INT8/NF4, and asynchronous candidates when memory or microbenchmark gains failed action-fidelity or closed-loop gates.

### 10.5 不建议写入简历主 bullet 的内容

- `guarded VLM 5/6 -> 6/6`：可在面试中作为机制实验解释，但样本太小，不适合作为主要成功率数字；
- LIBERO-PRO focused screen：可写“完成资格验证和视频 demo”，不要写成正式榜单；
- W8A16/INT8/NF4 的具体失败数字：适合面试展开，不必占用短简历；
- 旧 robosuite Stack pilot：任务过于简单，当前论文主线不依赖它；
- 真机部署：尚未完成，不能写。

## 11. 声明边界：必须遵守

### 11.1 当前证据支持的说法

- 外部 deterministic process control 在历史 LIBERO-10 对比中提升冻结 PI0.5 的总体和特定弱任务成功率；
- canonical VLM Planner 已完成真实 multimodal call、typed gate、VLA subgoal handoff 和一个配对 outcome repair；
- Monitor、bounded recovery、verification、safe hold 和 stop 已形成完整闭环；
- PI0.5 flow-step calibration、compile 和 SMVE 有可复现 latency、deadline、fidelity 与小规模闭环正结果；
- Optimize Runtime 与 Agentic recovery path 已完成同状态耦合验证；
- 框架接口支持模型替换，PI0.5 与 OpenVLA 已实际 exercise；
- 量化和异步方案已经实验并按部署门控拒绝，这属于有效工程结论。

### 11.2 当前证据不支持的说法

- 不得说 VLM Planner 带来了 `180/200 -> 185/200`；
- 不得说 canonical VLM 在大规模 benchmark 上获得统计显著提升；
- 不得把 selected LIBERO-PRO tasks 写成完整官方 benchmark；
- 不得说已经提出新的量化、剪枝或参数压缩算法；
- 不得说 W8A16/INT8/NF4 已被成功部署；
- 不得说所有 VLA 都获得相同加速或已经通用实时化；
- 不得说达到 hard real-time guarantee；
- 不得说完成真机实验；
- 不得把 HD 仿真视频描述为真实机器人视频；
- 不得把 physical recovery verification 等同于最终 task success。

## 12. 面试高频问题与回答要点

### Q1：为什么冻结 VLA，而不是继续微调模型？

项目关注在强预训练/微调 VLA 已具备局部技能后，如何用较低训练成本补足过程控制与部署效率。冻结策略便于隔离系统贡献、复用不同模型并避免为每个 recovery case 重新收集数据。它不是否认 finetuning，而是选择系统层研究边界。

### Q2：Monitor 和 VLM Planner 有什么区别？

Monitor 高频、确定性、低成本，只根据动作和可部署时序信号判断 stall/stale/deadline 等执行风险；VLM 低频、事件触发，在安全边界读取图像和任务语义，决定允许的语义子目标、registered skill 或 safe stop。前者检测症状，后者做语义判断。

### Q3：为什么不直接使用 VLA 内部的 VLM 能力当 planner？

VLA 的视觉语言表征与动作生成是耦合调用，接口通常不暴露稳定的结构化语义决策。外部 Planner 可独立设置调用频率、schema、预算和安全 gate，也便于替换。低层动作仍完全由原 VLA 生成，避免外部 VLM 越权。

### Q4：为什么 VLM 不能直接发动作？

自由形式语义模型缺少机器人 action normalization、时序和安全契约。CARVE 让它只输出 typed intent；robot-specific action 只能来自 VLA 或注册技能，从而保持可审计性和 fail-closed。

### Q5：SMVE 与普通 image pruning 有什么区别？

SMVE 只删除 adapter contract 已证明永久 padding、mask 恒 false 的相机槽，因此不改变有效视觉信息。它是静态输入契约优化，不是按内容动态删 token，也不能泛化到所有相机配置。

### Q6：为什么 45/45 replay 通过，W8A16 仍被拒绝？

长程闭环会积累微小动作差异。W8A16 在固定 observation 上满足 replay threshold，但在 T6 recovery gate 丢失一次成功，所以闭环门控优先于 open-loop aggregate fidelity。

### Q7：异步 prefetch 明明降低 deadline miss，为什么不用？

异步动作来自较早 observation。即使 worker 没有报错，时序偏移也会改变后续轨迹，导致 T8/T9 成功从 7/10 降到 5/10。因此模型调用速度与闭环控制实时性不是同一指标。

### Q8：项目创新和普通工程拼装有什么区别？

核心不是模块数量，而是三个系统原则：多速率语义/动作权限分离；面向失败恢复的 typed Harness；将 action fidelity、runtime 和 paired closed-loop outcome 联合成 profile admission。实验也保留了量化和异步的反例，证明这些 gate 能改变部署决策。

### Q9：框架有多通用？

接口与准入协议是模型可替换的，PI0.5 和 OpenVLA 已验证不同 capability contract。但 full Agentic + optimized closed-loop 目前只在 PI0.5 上完成，因此应表述为“model-replaceable design with two-family interface validation”，而不是“universally validated”。

### Q10：当前最大限制是什么？

VLM 调用仍需约 5--9 秒，只能放在安全边界；canonical VLM 与 LIBERO-PRO 样本规模有限；量化尚无通过闭环门控的正向 profile；没有真机验证。后续重点应是扩大 canonical Agentic 评估、优化 VLM 服务和完成 sim-to-real/real-robot validation。

## 13. 代码与证据索引

### 13.1 论文

- 详细技术报告：`paper/CARVE-VLA/root.pdf`
- LaTeX 源文：`paper/CARVE-VLA/root.tex`
- 论文阅读说明：`paper/CARVE-VLA/README.md`
- 自动汇总结果：`paper/CARVE-VLA/generated/runtime_results_summary.json`

### 13.2 核心代码

- Agentic runtime：`agentic_vla/runtime/`
- Optimize Runtime：`agentic_vla/optimization/`
- PI0.5 adapter：`agentic_vla/runtime/adapters/pi05.py`
- Monitor：`agentic_vla/runtime/monitor.py`
- Recovery skills：`agentic_vla/runtime/recovery.py`
- Unified agent/controller：`agentic_vla/runtime/agent.py`、`controller.py`、`harness.py`
- LIBERO runner：`agentic_vla/scripts/run_agentic_vla_libero.py`

### 13.3 关键状态文档

- 框架冻结与边界：`docs/status/CARVE_FRAMEWORK_FREEZE_20260724.md`
- 已完成工作：`docs/status/CARVE_VLA_COMPLETED_WORK.md`
- canonical Harness 实验：`docs/status/CARVE_CANONICAL_HARNESS_EXPERIMENT_20260727.md`
- LIBERO-PRO 资格实验：`docs/status/LIBERO_PRO_QUALIFICATION_STATUS_20260730.md`

### 13.4 关键结果目录

- PI0.5 Optimize：`results/carve_optimize/`
- Agentic-Optimize pair：`results/carve_pi05_agentic_optimize_pair_20260719/`
- Recovery Challenge：`results/carve_pi05_recovery_challenge_20260719/`
- Canonical Harness：以 `docs/status/CARVE_CANONICAL_HARNESS_EXPERIMENT_20260727.md` 中索引为准；
- LIBERO-PRO：以 `docs/status/LIBERO_PRO_QUALIFICATION_STATUS_20260730.md` 中索引为准。

## 14. 最小跨电脑传输包

如果另一台电脑可以直接 clone 完整仓库，只需让 Codex 优先读取本文件和 `root.pdf`。如果不传完整仓库，至少传输：

1. `paper/CARVE-VLA/RESUME_AND_CODEX_HANDOFF.md`；
2. `paper/CARVE-VLA/root.pdf`；
3. `paper/CARVE-VLA/README.md`；
4. `paper/CARVE-VLA/generated/runtime_results_summary.json`；
5. `docs/status/CARVE_FRAMEWORK_FREEZE_20260724.md`；
6. `docs/status/CARVE_VLA_COMPLETED_WORK.md`。

只传 `root.pdf` 也能理解论文，但本交接文档更适合生成简历，因为它已经明确拆分个人工作、技术栈、核心指标、负结果和声明边界。

## 15. 给下一位 Codex 的可复制 Prompt

```text
请先阅读 RESUME_AND_CODEX_HANDOFF.md 和 root.pdf。基于其中已完成的
CARVE-VLA 项目，为我生成面向具身智能/VLA 算法与模型推理部署岗位的
简历项目经历。请突出 Agentic Harness、VLM/VLA 权限分离、闭环恢复、
Optimize Runtime、torch.compile、SMVE、量化评估和 deployment admission。
所有数字必须保留实验口径；不得把 deterministic Harness 的 185/200
归因于 VLM Planner，不得把 LIBERO-PRO 资格实验写成完整 benchmark，
不得声称真机部署或成功的通用量化算法。先给出中文 3 条精简版、中文
4 条技术版和英文版，再列出面试可展开的 5 个技术点。
```

# 完整具身智能体系统与高效推理验证

更新时间：2026-08-27

本文档记录 CARVE-VLA 完整具身智能体框架的实现边界与紧凑验证结果。所有数字均
来自本地保存的事件日志、运行收据或官方 LIBERO-PRO 私有评估器，不使用 mock
环境或人工填写的成功标签。

## 1. 本轮目标

将既有 Agentic Harness 和 Optimize Runtime 补成一个可执行的具身智能体系统：

1. 低频 VLM Planner 负责视觉语义理解、长程任务拆分和工具选择；
2. 冻结 PI0.5 VLA 执行完整语言条件操作技能；
3. RGB-D 解析技能执行精确、可验证的短程动作；
4. Monitor、Critic、Memory 和恢复工具组成持续闭环；
5. Optimize Runtime 保证 VLA 关键路径满足本机实时截止时间；
6. 全部决策、工具调用、验证和模型 profile 形成可审计 trace。

## 2. 最终系统

| 层级 | 实现 | 职责 |
|---|---|---|
| 高层语义 Agent | Qwen3.5-9B VLM Planner，uniform NF4 | 观察图像、生成 1--4 阶段计划、选择 VLA 或注册工具 |
| 任务 Harness | stage plan、预算、schema gate、executor-consistency gate | 约束 Planner 输出并确保执行器结果只能验证对应阶段 |
| 过程记忆 | symbolic procedure memory + affordance memory | 复用已验证步骤顺序，每个新场景重新视觉定位 |
| 高频风险监测 | ExecutionRiskMonitor | 基于动作、状态响应、视觉变化、action age 和 deadline 检测执行风险 |
| 低频语义验证 | VLM Critic | 判断可观察后置条件是否满足，不读取仿真成功真值 |
| 通用动作技能 | 冻结 PI0.5 LIBERO policy | 执行完整的语言条件抓取与放置阶段 |
| 解析工具 | 6 个 bounded skills | RGB-D visual servo、相对移动、腕部旋转、姿态倾斜、夹爪、释放抬升 |
| 物理恢复 | 2 个 recovery skills | retract/lift/reobserve 与 release/retract/lift/reobserve |
| 高效推理 | PI0.5 compiled BF16 + 2 flow steps + SMVE | 压缩 VLA 关键路径并按 80 ms deadline 记录 miss/fallback |
| 审计边界 | canonical tool runtime + JSONL receipts | 记录模型调用、工具输入输出、阶段推进、验证来源和视频 |

关键代码：

- `agentic_vla/runtime/agent.py`：VLM Planner/Critic 协议与结构化输出校验；
- `agentic_vla/session.py`：任务阶段、工具预算、记忆和执行器一致性门控；
- `agentic_vla/benchmarks/libero_runtime.py`：LIBERO-PRO 观测与 RGB-D 标定；
- `agentic_vla/benchmarks/libero_skills.py`：可部署解析技能库；
- `scripts/run_agentic_vla_libero_pro_canonical.py`：完整闭环入口；
- `scripts/probe_carve_libero_embodied_tools.py`：工具物理资格验证。

## 3. 新增的关键正确性约束

### 3.1 Planner 不能直接控制关节

Planner 只能选择 `vla_act`、注册的 `run_skill`、继续执行或安全停止。解析技能参数
必须符合白名单 schema；Planner 不能输出关节位置、力矩、轨迹或仿真器物体位姿。

### 3.2 过程记忆不保存场景坐标

记忆只保存经验证的符号步骤顺序。`proposed_plan` 中禁止持久化 grounded
`skill_args`，新回合必须根据当前 RGB-D 图像重新定位，避免把上一场景坐标错误复用。

### 3.3 恢复成功不等于任务成功

恢复工具的后置条件只是“机械臂脱离接触并获得新观测”。新增
executor-consistency gate 后，恢复验证不能推进一个由 VLA 执行的语义任务阶段。
在最终 trial 9 中，恢复于 step 142 完成，而第一阶段直到 step 242 才由视觉 Critic
确认，说明该门控实际生效。

## 4. 真实物理仿真验证

### 4.1 RGB-D 解析工具资格验证

环境为官方 LIBERO-PRO `libero_10_object` task 8：`put both moka pots on the
stove`。该验证不调用 VLA，也不读取任务评估器状态。

| 工具 | 结果 | 物理指标 |
|---|---:|---:|
| `visual_servo_above` | 成功 | 最终笛卡尔误差 `0.0118 m` |
| `rotate_wrist` | 成功 | 执行 `0.350 rad` 有界旋转 |
| `move_relative` | 成功 | 最终笛卡尔误差 `0.0116 m` |

共执行 78 个控制步，3/3 工具成功。视频与机器可读结果：

- [工具验证视频](../../results/libero_pro_embodied_tool_probe_20260827/embodied_tool_probe.mp4)
- [工具验证结果](../../results/libero_pro_embodied_tool_probe_20260827/summary.json)

### 4.2 完整系统端到端验证

同一官方 LIBERO-PRO 任务上运行一个 reference 初始状态和两个 held-out 初始状态。
每个回合均启用 VLM Planner、任务阶段 Harness、VLM Critic、过程记忆、风险监测、
物理恢复、冻结 PI0.5 和 Optimize Runtime。

| Trial | 过程记忆 | 恢复 | 结果 | 步数 | 视频 |
|---:|---:|---:|---:|---:|---|
| 0 reference | 建立 1 条验证过程 | 0 | 成功 | 399 | [MP4](../../results/full_embodied_agent_20260827/reference_t8_trial0/full-embodied-memory-reference-agentic-libero_10_object-t8-r0-s7-startup-planner-semantic-checkpoint/episode.mp4) |
| 7 held-out | 命中 | 0 | 成功 | 380 | [MP4](../../results/full_embodied_agent_20260827/heldout_t8_trial7/full-embodied-memory-heldout-agentic-libero_10_object-t8-r7-s7-startup-planner-semantic-checkpoint/episode.mp4) |
| 9 held-out | 命中 | 1 次并继续完成 | 成功 | 415 | [MP4](../../results/full_embodied_agent_20260827/heldout_t8_trial9_executor_gate_v3/full-embodied-memory-heldout-v3-agentic-libero_10_object-t8-r9-s7-startup-planner-semantic-checkpoint/episode.mp4) |

聚合结果：

- 成功：`3/3`；完整任务计划完成：`3/3`；
- held-out 过程记忆命中：`2/2`；
- Planner 调用：每回合 1 次；Harness 无额外 Planner 的阶段推进：3 次；
- Critic 调用均值：`1.33/episode`；物理恢复：1 次且恢复后最终成功；
- Planner schema 自动修复：2 次；非法执行器阶段验证：0 次。

机器可读聚合：[aggregate_validated.json](../../results/full_embodied_agent_20260827/aggregate_validated.json)。

### 4.3 VLM grounding 与工具调度验证

在相同场景中向 Guarded Planner 明确请求调用注册的 `visual_servo_above`。VLM
一次生成合法的 `run_skill` 决策与参数 `u=0.75, v=0.50,
approach_height_m=0.15`；Harness 只接受注册 schema，随后 RGB-D 工具执行 48 个
控制步，以 `0.0094 m` 最终误差完成动作。Planner 用时 `25.59 s`，该调用不在
实时控制关键路径中。

- [VLM 工具调用视频](../../results/vlm_embodied_tool_use_20260827/vlm_grounded_tool_use.mp4)
- [VLM 工具调用结果](../../results/vlm_embodied_tool_use_20260827/summary.json)

该指令明确指定了技能名，因此本实验验证的是 VLM grounding、参数生成、schema
门控和物理 dispatch 的完整链路，不作为“自主学习工具选择策略”的证据。

## 5. 高效推理结果

### 5.1 PI0.5 实时关键路径

完整端到端 3 回合共记录 119 次 PI0.5 调用：

| 指标 | 结果 |
|---|---:|
| Runtime latency mean | `55.32 ms` |
| Runtime latency P95 | `57.23 ms` |
| Runtime latency max | `63.57 ms` |
| Model latency mean | `39.35 ms` |
| Deadline | `80 ms` |
| Deadline miss | `0/119` |

使用的已晋升 profile 为
`pi05-torch_compile_masked_views-bf16-2step-h10`，包含两步 flow inference、
`torch.compile` 和 Static Masked-View Elision。该 profile 之前已通过 45 个状态的
动作保真度、80 ms deadline 和 T8/T9 闭环门控。

### 5.2 Planner 轻量化与调用频率

Qwen3.5-9B 的 358 个 eligible linear layers 使用 NF4，视觉编码器也被量化，
`lm_head` 保留 BF16。加载后 PyTorch peak allocated 为 `7.38 GiB`。运行期间与
PI0.5 共卡观测到的进程显存约为 `9.00 GiB + 7.64 GiB`，可在 24 GB RTX 4090
上共驻。

当前 Planner 接受响应均值为 `30.31 s`，P95 为 `35.13 s`。因此它被严格限制为
任务启动和异常事件触发的低频语义模块，不进入 80 ms 控制关键路径。这证明了
“可共驻”和“低频可用”，尚不能声称 Planner 已实现实时推理。

量化收据：[qwen35_9b_uniform_nf4_receipt.json](../../results/full_embodied_agent_20260827/_services/qwen35_9b_uniform_nf4_receipt.json)。

## 6. 软件验证

- 项目测试：当前扩展后为 `258 passed`；
- 新增技能单测覆盖 schema、参数边界和 dispatch；
- 新增 session 回归测试保证恢复 primitive 不能完成活动 VLA 阶段；
- 最终三回合 executor-consistency audit：通过，非法阶段验证为 0。

## 7. 结论与证据边界

本轮已经证明：CARVE-VLA 不再只是 Monitor 加 Retry 的规则包装，而是一个具有
VLM 规划、符号任务计划、过程记忆、可替换 VLA、具身工具、语义 Critic、物理恢复、
安全边界和实时 VLA Runtime 的完整可执行系统。系统已在真实 LIBERO-PRO 物理
仿真中完成数据可追溯的端到端验证，并产生可播放视频。

本轮没有证明：

- 在整个 LIBERO-PRO 上相对冻结 VLA 的统计显著成功率提升；
- 所有 6 个解析技能均被 Planner 在自然失败中自动选择；
- 9B Planner 达到实时响应；
- 仿真结果等价于真机部署结果。

因此，这组结果适合作为毕业论文中的“完整系统实现与代表性闭环验证”，不应单独
包装成 benchmark-wide state-of-the-art。若继续扩展，优先补多任务/多初始状态
对照和 Planner 工具选择消融，而不是再次改动核心框架。

后续已完成 verified-procedure memory-first 路由：在相同三个状态上保持 `3/3`
成功，将 task-start Planner 调用降为 0，平均回合时间降低 `53.2%`。详见
[Memory-Routed Planner Profile](CARVE_MEMORY_ROUTED_PLANNER_PROFILE_20260827.md)。

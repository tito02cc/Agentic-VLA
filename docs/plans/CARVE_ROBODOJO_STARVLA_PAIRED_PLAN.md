# CARVE-VLA RoboDojo/StarVLA 配对实验计划

> 历史协议，非当前执行队列。保留用于解释原实验；不得据此启动新回合。
> 当前只执行[唯一计划](ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)的P1--P4及门控。

## 1. 实验目标

本阶段只验证两条论文主线，不把“更换 VLA”误写成 CARVE 的贡献：

1. **Agentic Harness**：在冻结的预训练 VLA 外部增加持续监测、事件触发的语义规划、过程记忆、有限恢复与安全停止，检验其能否纠正可恢复错误并改善长程执行。
2. **Optimize Runtime**：在不重新训练 VLA 的前提下，优化动作块复用、重规划频率、计算配置与 VLM/VLA 调度，检验其能否降低策略调用、尾延迟和显存压力，同时保持任务表现。

RoboDojo 只提供环境与私有成功判定，StarVLA 只作为冻结的动作策略。CARVE 的结果必须来自同一 checkpoint、任务、布局和预注册配置的重复对照；由于 Isaac 与扩散推理不是逐轨迹确定的，不将同 seed 重复称为严格反事实配对。

## 2. 固定系统边界

```text
RoboDojo observation / official evaluator
                 |
                 v
CARVE Agentic Harness
  Monitor -> event router -> VLM Planner/Critic -> memory/recovery
                 |
                 v
CARVE Optimize Runtime
  capability negotiation -> action-chunk scheduling -> tracing
                 |
                 v
Frozen StarVLA PI-v3 policy -> ARX-X5 action
```

- `ExecutionRiskMonitor`：高频、低成本，仅使用在线执行信号；它不是 VLM，也不做开放语义推理。
- `VLM Planner/Critic`：低频、事件触发，处理任务分解、语义异常与恢复选择；未真实调用 VLM 的实验不得标记为 VLM Agent。
- VLA：负责生成连续机器人动作，不参与 CARVE 参数训练。
- 成功率：仅采用 RoboDojo 官方私有 evaluator 输出，不使用奖励、物体位姿或成功谓词作为 CARVE 的输入。

## 3. 对照条件

| 条件 | Agentic Harness | Optimize Runtime | 用途 |
|---|---:|---:|---|
| `B0` | 否 | 否 | 官方 StarVLA 原始基线 |
| `C1` | 否 | 是 | 高效推理独立消融 |
| `C2` | 是 | 否 | Agentic 独立消融 |
| `C3` | 是 | 是 | 完整系统与耦合收益 |

`B0` 已在 `build_tower/layout=0/seed=0` 完成 1 回合官方 smoke：0/1 success、1050 steps。该结果只证明官方链路与视频生成正常，不代表 CARVE 效果。

## 4. 分阶段执行

### Gate A：Adapter 等价性

- 对相同 StarVLA 服务输入，官方调用路径与 `StarVlaAdapter` 必须得到数值一致的动作块。
- 验证动作维度、双臂关节顺序、夹爪顺序、重置生命周期和异常传播。
- 固定配置下 CARVE 不得静默修改 action-head inference steps、精度或 action chunk。

失败即停止，不运行仿真。

### Gate B：单回合配对 smoke

- 任务：`build_tower`
- 布局与随机种子：均为 `0`
- checkpoint：官方 PI-v3 reference checkpoint
- 依次运行 `B0 -> C1 -> C2 -> C3`
- 每个条件保留官方结果、三视角视频、结构化 trace、GPU 峰值和启动配置。

若 `C1` 改变动作语义、`C2/C3` 出现不可解释的高频误触发，或出现基线成功转失败，则暂停并修正机制。

### Gate B2：自然执行主评测

主结果必须来自 RoboDojo 原始任务分布，不注入故障、不改初始状态、不根据本项目的
新增 rollout 结果选择回合。初始任务清单曾包含冻结 PI-v3 官方成功率为 0--4% 的
`store_tools_in_toolbox`、`pour_balls_into_vase` 和 `organize_table`。这会把动作策略缺失
的物理技能与 Harness/Runtime 效果混淆。根据 checkpoint 随附的 50 回合公开结果，
在运行新的跨任务对照前将正式候选集一次性修订并冻结为：

1. `stack_bowls`：短程抓取、对齐与放置；
2. `insert_tubes`：精细双臂插入；
3. `build_tower`：多阶段双臂精细操作；
4. `put_bottles_into_dustbin`：多物体长程操作；
5. `match_and_pick_from_conveyor`：带短期记忆的动态操作。

该清单覆盖 RoboDojo 的 Generalization、Precision、Long-Horizon 与 Memory 分组；选择
依据是运行本项目实验前已公开的 checkpoint 级先验，而不是 CARVE 条件的事后结果。
官方 PI-v3 成功率分别为 14%、44%、56%、64% 和 12%。这些数值只用于能力准入，
不并入本项目统计。
每个任务先运行预注册策略 RNG seed `0--4`，条件顺序按 seed 轮换，避免固定运行
顺序造成温度或系统状态偏差。首轮只比较 `B0`、`C2`：

| 条件 | Monitor | VLM Planner/Critic | Recovery | 人工故障 |
|---|---:|---:|---:|---:|
| `B0` | 否 | 否 | 否 | 否 |
| `C2` | 是 | 事件触发 | 有限 | 否 |

主指标只有官方 success rate。CARVE 只有同时满足以下条件才算产生真实净收益：

- 总成功数高于 `B0`，且提升来自至少两个任务族，而非单个挑选任务；
- `baseline failure -> CARVE success` 多于 `baseline success -> CARVE failure`；
- Planner 不读取 reward、evaluator、对象真值或未来帧；
- 未触发异常的成功轨迹不因 Harness 明显回归；
- 新增语义调用和恢复开销被完整报告。

若 25 回合门控没有净正向趋势，停止扩充样本，依据误触发、错误子目标和恢复失败
日志改方法；不得更换任务、删去负例或增加有利故障来包装结果。出现净正向趋势后，
再扩展到每条件每任务至少 10 回合并报告 Wilson 95% 置信区间。

主评测入口为 `scripts/run_robodojo_starvla_nominal.sh`，其中
`STARVLA_CARVE_FAULT_HOLD_STEP=-1` 被显式写死。`C1/C3` 只在 Agentic 净收益成立
后用于 Optimize Runtime 独立消融，避免把调度变化与 Agentic 效果混在一起。
`B0/C2` 的常态路径必须固定使用相同的原生 Flow4 与 execute horizon=16。`C2` 允许
在自然 Monitor 事件之后按预注册预算临时改变执行窗口，但必须逐调用标记、单独报告
触发次数，并与固定 `h32` 基线对照；未触发事件的回合不得归因于 Agentic。`C1/C3`
在常态路径引入的 Flow inference steps 或 horizon 变化必须单独归因于 Optimize Runtime。

### Gate B3：受控可恢复故障（仅机制消融）

自然 rollout 存在非确定性，而且正常成功轨迹通常不会触发 VLM。为单独检验
Harness 的恢复因果链，在同一任务、布局、种子、checkpoint 和 Optimize 配置下
加入配对的 `stale_action_hold`：

| 条件 | Monitor | VLM Critic | 允许干预 | 预期用途 |
|---|---:|---:|---:|---|
| `shadow+fault` | 记录 | 不调用 | 否 | 验证故障本身是否导致失败 |
| `full+fault` | 记录 | 事件触发 | 有限重规划 | 验证 CARVE 是否恢复 |

- 故障只冻结已下发动作，不移动物体、不读取 reward、对象真值或 evaluator。
- 固定在第 640 控制步注入，避免依据轨迹结果事后选择注入点。
- Monitor 首次 `no_progress` 只丢弃旧动作块并请求新 VLA 动作；持续无进展才调用
  VLM Critic。
- Critic 只判断完整任务是否已经可见完成。未确认、输出不合法或超时时执行保守的
  task-preserving replan；确认完成时安全停止。
- `full` 条件可启用有界 recovery compute boost：正常阶段使用低成本推理配置，
  仅在 Critic 判定未完成后，将接下来的固定次数 VLA 调用切换为更多去噪步和更短
  动作块，用更高计算换取恢复阶段的控制精度，随后自动恢复 fast path。该配置必须
  在 trace 中逐调用标记，并与固定计算恢复做消融。
- 每回合最多 3 次语义调用，并记录 schema 校验、时延、回退原因和最终动作。
- 若 `full+fault` 和 `shadow+fault` 都成功，说明故障强度不足；若两者都失败，说明
  当前恢复策略无效。二者均不应通过增加随机回合数来包装成正向结果。
- 本节只能支持 Monitor、Planner 和 Recovery 的机制解释，不能代替 Gate B2 的
  自然任务成功率，也不进入“CARVE 提升 VLA 能力”的主结论。

### Gate C：自然结果扩展与效率消融

仅当 Gate B2 的 `C2` 对 `B0` 有净正向趋势后运行：

1. 将五任务扩至每条件每任务至少 10 回合；
2. 加入 `C1` 检查 Optimize Runtime 单独是否保持成功率并降低开销；
3. 加入 `C3` 检查完整系统是否保持 `C2` 的 Agentic 收益；
4. 再增加布局变化，检验恢复策略是否依赖固定场景。

若 `C1` 降低延迟但伤害成功率，按质量约束淘汰；若 `C3` 不如 `C2`，Optimize
不得并入 Agentic 主结果。受控故障结果不用于决定是否扩展自然任务样本。

## 5. 指标与证据

### 任务指标

- 官方 success rate 与成功数/总回合数
- 官方可获得时记录完成度；若官方未输出则不自行构造“完成率”
- baseline success -> CARVE failure 的回归数
- baseline failure -> CARVE success 的恢复数

### Agentic 指标

- Monitor 触发次数、误触发次数与触发原因
- VLM Planner/Critic 调用次数、超时与 schema 合法率
- replanning、recovery、safe stop 次数
- memory hit、复用来源与重复任务收益
- 完整工具调用与审计链
- 故障注入到 Monitor 触发、语义判断、恢复动作和最终结果的逐阶段延迟
- 受控故障下的 recovery success，以及 VLM 输出被 Guard 接受的比例

### 高效推理指标

- VLA 实际推理调用数、执行动作数和动作块利用率
- 单次 VLA 推理 P50/P95、episode wall time
- deadline miss rate 与 action age
- GPU 峰值显存
- Planner/VLA 同卡时的争用开销
- nominal fast path 与 recovery compute boost 各自的调用数、P50/P95 和计算预算

## 6. 公平性与禁止事项

- 不训练或微调 StarVLA；所有条件共享同一 checkpoint。
- 不允许 CARVE 读取 evaluator 的成功谓词、reward、对象真值位姿或未来状态。
- 不把脚本化 Planner、oracle memory 或人工规则结果写成 VLM 结果。
- 不把 `B0` 的 0/1 结果扩写为成功率结论。
- 不依据已有成功/失败结果挑选自然评测任务、故障时刻或 Planner 票据。
- 不将受控故障恢复率替代自然任务成功率。
- 不在配对门控通过前批量运行 54 个任务。
- 量化、compile、Flow inference-step 等只在对应后端真实启用且有 manifest/日志时计入 Optimize 贡献。

## 7. 当前执行顺序

1. 实现 `StarVlaAdapter`、StarVLA Optimize 插件与单元测试。
2. 在官方 wrapper 中加入显式的 `baseline/runtime/agentic/full` 模式，默认仍为官方 baseline。
3. 完成离线动作等价性测试和结构化 JSONL trace。
4. 运行 `build_tower` 同种子 `C1`，验证高效推理链路。
5. 运行 nominal `C3`，验证 StarVLA、Isaac Sim 与量化 VLM 的单卡共驻和正常轨迹
   零语义调用 fast path。
6. 冻结五任务、官方 layout set `0--2` 及各集合内的 layout 子集，先完成无故障
   `B0/C2` 自然执行门控；Monitor 使用对象/阶段级视觉谓词，低运动量规则仅保留为
   执行安全信号。
7. 只有自然执行出现跨任务净正向趋势，才扩至每任务至少 10 回合。
8. 随后加入 `C1/C3`，分别验证 Optimize Runtime 与完整耦合系统。
9. `shadow+fault/full+fault` 只保留为机制消融，不再驱动主结论。

## 8. 停止条件与论文边界

- **毕业论文最低闭环**：第二 VLA Adapter、真实仿真、无故障自然任务对照、
  Agentic 机制消融、Optimize 独立消融、视频与结构化日志均具备。
- **RAL/机器人顶会最低证据**：至少 3 个代表性任务、多布局且每个关键条件至少 10 次预注册重复，
  包含 `B0/C1/C2/C3`、受控故障恢复、VLM 量化/调度和单卡资源指标。
- 在关键因果对照成立前，不转向 RoboDojo 全榜、RoboTwin 2.0 或新的基座模型；
  新 benchmark 只能补充外部有效性，不能替代当前机制验证。

本文件是 RoboDojo/StarVLA 阶段的唯一执行协议；旧计划仅作历史记录，不再驱动新增实验。

## 9. Gate B2 实际结论

- Shadow 无干预：`0/3`；固定计算恢复：`2/3`；自适应计算恢复：`2/3`。
- Agentic 有限恢复获得正向描述性证据，但样本量不足以形成显著性结论。
- 自适应计算路径按设计执行了 12 次恢复增强调用，但没有提高成功率，P95 反而
  从 `388.10 ms` 增至 `405.10 ms`。
- 因此停止继续调大 DDIM steps。下一门控是让 VLM Planner 输出受 Guard 约束的
  恢复工具/语义子目标，再由 Runtime 按恢复原语分配计算，不继续刷相同配置。

## 10. 自然任务方法修订

`stack_bowls` 的历史日志复核发现，基线真实使用的 `STARVLA_EXECUTE_HORIZON`
曾被外部环境变量污染，而 CARVE 日志只显示另一项执行窗口。受影响的旧回合全部
降级为机制调试记录，不再用于条件比较。启动脚本现同时显式固定 StarVLA 和 CARVE
的名义执行窗口。低运动量 Monitor 仍无法识别“机械臂持续运动但对象没有进展”的
自然失败。

因此后续 C2 不再依赖单一像素/关节停滞阈值。Planner 声明动作无关的视觉谓词；
低频 VLM 只枚举可见对象组，Harness 以分组数执行确定性比较。VLM 的自由文本状态
和直接数值结论均不具有控制权限。首个候选谓词是“空间上分离的目标对象组数”，
但同一接口必须在 `store_tools_in_toolbox` 等未参与标定的任务上使用不同谓词接受
外部验证，避免把 `stack_bowls` 专用规则包装成通用 Agent。

Qwen3-VL-2B 的 vision-preserving NF4 配置已通过 6/6 离线门控和 StarVLA +
Isaac Sim 单卡共驻门控。严格 seed 2 配对中，B0 与 C2 均为 0/1；C2 的 Critic
正确持续识别 3 个对象组，也真实执行了两次 h32 恢复调用，但对象状态没有改善。

因此下一步不扩展五任务扫描，也不继续重复同一种“仅延长动作窗口”的 C2。先将
恢复工具升级为可审计的语义子目标选择或技能切换，并验证它确实改变传给 VLA 的
指令/工具调用；只有新的恢复原语通过离线契约测试和单回合因果试验后，才恢复严格
`B0/C2` 多 seed 配对。

## 11. 请求级扩散随机性门控

PI-v3 action head 在每次调用中使用随机初始动作噪声。仅固定 RoboDojo layout seed
和 StarVLA 进程 seed，不能保证不同条件获得相同动作采样。正式配对新增以下要求：

1. 使用同一 `CARVE_ACTION_SEED_BASE`，并按环境编号和动作块起始 timestep 派生
   请求级 `action_seed`。
2. B0 与名义等价 C1 连续运行时核对语言、状态、图像与动作哈希，定位随机性来源。
3. 已确认独立 Isaac/RTX 运行仍存在轻微图像差异，因此不要求跨进程动作哈希完全
   一致，也不把同 seed 结果称为严格反事实配对。
4. 正式比较采用相同 layout/seed 集合、交错条件顺序与重复 block；Runtime 延迟可按
   调用统计，成功率必须按独立回合和置信区间报告。
5. C2 的 VLM 判断不读取 reward、官方 success predicate 或对象真值；当进展明确时
   必须允许零干预，不能为了制造 Agentic 事件改写正常轨迹。

RoboDojo 命令行中的 `seed` 实际选择官方打包的 layout set，本地资产只包含
`0/1/2`，不能把它当作任意 RNG seed 使用。正式矩阵改为三个官方 layout set 内的
多个 layout，并对 B0/C2 交错执行；启动器会在加载模型前拒绝不存在的集合。此前
计划中的 seed `0--4` 表述作废，不得把环境初始化失败计入实验分母。

## 12. VLA 语言能力门控

任务计划不能默认获得修改冻结 VLA 指令的权限。每个 Adapter 必须声明：

- `task_only`：只保证 checkpoint 训练任务描述。Planner 维护账本、视觉检查、动作块
  重启和恢复预算，但不得向 VLA 注入新子目标。RoboDojo StarVLA 默认使用此模式。
- `open_vocab_subgoal`：仅当独立门控证明 VLA 能执行未见过的组合子目标时启用；门控
  至少包含原任务、同义改写和两类局部子目标，并检查成功率不低于预注册阈值。

v20 表明“左碗与中碗先堆叠”的局部语言未完成预期阶段，而恢复原任务后最终成功。
因此停止继续扩大开放子目标配置，先运行 `task_only + 任务账本 + 组数 Critic` 的
单回合机制对照。若该路径可靠，再用预注册 block 比较 B0 与 C2；若仍无净收益，
应增加适配器可执行的技能工具或更换具备开放指令能力的 VLA，而不是继续改 prompt。

## 13. 语义后端准入与熔断

layout set 1 的在线复核表明，Qwen3-VL-2B 会产生物体实例误计数，StarVLA 内部
视觉语言主干虽然能生成文本，但动作微调后不能稳定遵循 JSON 协议。两者目前都不
具备主 Critic 控制权限。后续必须遵守以下门控：

1. 从 B0 视频按任务、layout 和执行阶段预注册开发集与留出集；不得只选择明显成败帧。
2. Critic 在留出集同时报告 JSON 合法率、阶段判断准确率、假阳性恢复率和单次时延。
3. 只有留出集合法率与准确率均不低于 95%，且会导致错误恢复的假阳性率不高于 2%，
   才允许从 shadow 升级为 execute。
4. 在线输出解析失败、超时或违反物理范围时不执行恢复；达到配置的协议失败上限后，
   本回合熔断语义后端并恢复原始任务。
5. `shared_semantic_generation=true` 只表示 VLA 服务提供生成接口，不代表该主干已通过
   语义准入；准入结果必须另有带校验和的证据清单。
6. RTX 4090 上优先采用分阶段独立 4B VLM 或远端异步 VLM。2B 与共享主干仅保留为
   shadow 对照，除非重新通过留出集门控。

在该门控通过之前，暂停正式 C2 成功率矩阵。可以继续运行 B0/C1 的 Optimize Runtime
独立消融，但不得把没有可靠语义决策的动作块重启称为完整 Agentic 增益。

## 14. Optimize Runtime 三布局门控结果

`stack_bowls` 的官方 layout set `0/1/2` 已完成 B0/C1 首轮门控：B0 为 `0/3`，C1
为 `2/3`；C1 的实测 VLA 调用总数相对 B0 固定 h16 的调用估算减少 `65.33%`。后续
发现该 block 的请求步数没有进入 PI-v3 action head，因此它只验证 h32 动作窗口，
不能写作 DDIM/Flow 步数优化。h32 只能保留为该类任务的 scoped 候选。

下一步执行顺序调整为：

1. 不继续运行依赖 2B 或共享主干控制决策的 C2/C3。
2. 为 C1 增加一个不同操作结构的任务，优先选择具备自然基线成功样例的任务；先做
   `3 x B0/C1` 门控，若成功率明显退化则停止该配置。
3. 为 Planner/Critic 构建预注册的多 layout 留出集，并优先验证独立 4B 或远端异步
   后端；只有满足第 13 节阈值才恢复 C2/C3。
4. C3 只在 C1 跨任务保持质量、C2 Critic 通过准入后运行，防止把两个未验证模块直接
   耦合后重复消耗仿真资源。

## 15. `build_tower` 门控后的配置修订

第二操作结构的 layout set 0 诊断中，历史 B0 原生 Flow4/h16 在 740 帧成功，旧 C1
h32 运行至 1051 帧失败；将窗口恢复为 h16 后在 742 帧成功。另一次 h16 独立回合运行
至 1051 帧失败，确认该 benchmark 存在不能被请求级随机种子完全消除的跨进程波动。
这些旧回合的 action head 均实际使用原生 Flow4，不能用于推理步数比较。

据此调整后续顺序：

1. C1/C3 的通用默认值改为 Flow2/h16；h32 必须由任务族 profile 显式选择。
2. 不把 `stack_bowls` 的调用数下降外推为所有任务的闭环加速收益。
3. 已修复 PI-v3 忽略旧 `num_ddim_steps` 的控制断路，新增明确的
   `num_inference_steps` 参数、server handshake 和 trace 生效标记；B0 继续保持原生
   Flow4，不受 CARVE 参数影响。
4. 常驻模型 20 对交错微基准显示 Flow4→2 的成对平均延迟下降 `23.37%`，动作 MAE
   均值为 `0.00325`。该结果通过模型级效率门控，但尚未通过闭环质量门控。
5. 先完成修复后 Flow2/h16 的 `build_tower` 单回合质量门控；若出现明显行为退化，
   停止扩跑并尝试 Flow3，而不是继续消耗 layout。
6. 只有真实 flow-step 控制的 h16 配置通过多个 layout 后，才扩展正式 B0/C1 成功率
   统计；所有修复前 `build_tower` 回合均属于机制诊断。

首个修复后 Flow2/h16 回合已在 layout set 0 于 718 帧成功，trace 证明 2 步控制真实
生效，稳态 P50/P95 为 `242.68/297.36 ms`。该 profile 通过单回合门控，现冻结为
`starvla-flow2-h16`，并按 `B0(set1) → C1(set1) → B0(set2) → C1(set2)` 顺序补齐
小规模交错 block。矩阵完成前不再更改 flow steps、horizon 或 prompt。

## 16. 修复后 Flow2/h16 三布局门控结论

预注册的 `build_tower` layout set `0/1/2` 已按交错顺序完成。原生 Flow4/h16 与
CARVE Flow2/h16 均为 `3/3` 成功；完成帧分别为 `740/744/737` 与 `718/743/736`。
Flow2 的 138 次实际调用均在 trace 中标记 `num_inference_steps=2`、h16 和
`inference_step_control_verified=true`。由于样本仅为三个 layout，且 Isaac/RTX
跨进程轨迹不是严格确定的，该结果只支持“首个长程任务族未观察到质量退化”，不支持
成功率等价性或任务速度提升的统计结论。

模型级效率结论继续采用常驻模型的 20 对交错微基准：Flow4 到 Flow2 的成对平均推理
延迟下降 `23.37%`，动作 MAE 均值为 `0.00325`。同为 h16 时，仿真中的 VLA 调用数
差异主要来自完成帧数，不作为 Optimize Runtime 的核心收益。下一步只增加一个不同
操作结构的任务族做迁移门控；若 Flow2 出现基线成功转失败，则先测试 Flow3 并分析
轨迹，不继续扩充同一错误 profile。

## 17. 第二任务门控与 Critic 权限冻结

`put_bottles_into_dustbin` 的三个官方 layout set 已完成 B0/C1 交错门控，原生
Flow4/h16 与 Flow2/h16 均为 `3/3`。至此两个不同操作结构共六个 layout 中没有观察到
Flow2 引起的成功回归，`starvla-flow2-h16` 保留为 Optimize 候选配置。由于瓶子任务中
C1 完成帧和调用数没有下降，论文只报告模型推理延迟收益，不报告 episode 加速。

独立 Qwen3-VL-4B 的 BF16 与视觉保留 NF4 均未通过 Critic 控制准入：二者在 48 个
留出样本上的终态停止召回同为 `37.5%`，且垃圾桶任务为 `0/4`。相同失败模式排除了
NF4 是主要原因，当前边界是相机可观测性。基于该结果，执行协议更新为：

1. VLM 输出非法、超时或预算耗尽时必须保持原始 VLA 轨迹，禁止映射为恢复或停止。
2. Critic 的停机权限默认关闭；只有任务级新留出集通过门槛后才显式开启。
3. `put_bottles_into_dustbin` 不再用于视觉 safe-stop 证明，但仍保留为 Flow-step 质量门。
4. 下一 Agentic 门控优先评估“未完成/停滞识别 -> 任务保持重规划”，而非依赖终态停机。
5. 最终准入集必须来自提示冻结后的新仿真回合；当前 60 样本仅作开发与诊断证据。

若新的 Critic 仍不能稳定判断未完成状态，则正式 C2/C3 改为 VLM shadow + 本地执行
风险控制，不再让语义模块拥有动作修改权限。该降级仍是可审计的框架能力边界，但不能
作为 Agentic 成功率提升实验。

## 18. 在线任务计划机制门后的路线修订

`stack_bowls` 在线任务计划已经真实走通 Planner、Guard、任务台账、定期视觉检查、
连续矛盾门控、有限重试与回退，但单回合官方结果为失败。运行还发现 4B grouped
verifier 会把 `three bowls on table` 错当成一个空间组。解析器现已拒绝“包含多个实例但
未说明 stacking/nesting/touching 关系”的集合描述；修复前回合只作诊断，不进入 C2。

下一步不重复旧配置，按以下顺序推进：

1. 冻结 Planner schema、分组证据协议和 `task_only` 权限边界，先用新图像样本验证
   含糊 group 输出只能产生 inconclusive，不再产生控制动作。
2. 组合条件改为 Flow2/h16 名义快路径、Flow4/h16 有限恢复；trace 必须同时出现两种
   `num_inference_steps`，否则不得称为 Agentic 与 Optimize 耦合。
3. 只在冻结 VLA 自然具备非零成功率且阶段结果可从头部相机观察的任务上做 B0/C2
   配对。若 Planner 只能生成台账、却无法改变可执行技能或提高恢复概率，则如实保留为
  可审计 orchestration 结果，不扩大成功率矩阵。
4. 只有单回合同时满足“可靠语义证据、真实 Flow2→Flow4 切换、无误触发”后，才扩展
   到多个官方 layout；否则继续修框架或收缩主张。

## 19. 冻结的受控故障因果矩阵（2026-09-02）

首个 layout set 0 已满足第 18 节门控：C1 在第 640 步注入 stale-action hold 后失败；
C3 的本地 Monitor 在第 674 步首次产生 `no_progress`，第 675 步接受预先固定的 VLM
恢复票据，随后真实执行两次 Flow4/h16 `recovery_boost`，再返回 Flow2/h16 快路径，
官方结果成功。该单对结果只证明链路可产生因果恢复，不构成恢复率结论。

后续矩阵在查看新结果前冻结如下：

| layout set | C1 | C3 | 执行顺序 |
|---:|---|---|---:|
| 0 | 已完成，失败 | 已完成，成功 | 1--2 |
| 1 | 已完成，失败 | 已完成，成功 | 3--4 |
| 2 | 已完成，失败 | 已完成，成功 | 5--6 |

所有新回合固定：StarVLA PI-v3 checkpoint、动作 seed base 与 layout set 相同、故障
步 `640`、名义 Flow2/h16、恢复 Flow4/h16、最多 2 个恢复调用、禁止语义安全停机。
C3 使用同一份从历史在线 Qwen3-VL 调用中冻结的 accepted Planner ticket；该实验只把
票据作为确定的语义控制输入，不声称在线 VLM 时延或准确率。在线 VLM 规划与任务台账
能力由独立的无故障 C3 成功回合证明，两条证据不可混写。

排除规则：模型/仿真启动 OOM、自动 reset 后被人工终止、缺失官方结果、缺失 runtime
trace，或请求的 inference steps 未由握手和逐调用 trace 证明生效。任务自然失败不得
排除。三组完成后报告逐 layout 配对结果、失败转成功/成功转失败数、恢复触发步、实际
Flow2/Flow4 调用数和视频，不把 3 个样本包装为统计显著性。

矩阵已经按上述顺序完成，三组均通过协议检查：C1 为 `0/3`，C3 为 `3/3`，得到 3 个
失败转成功、0 个成功转失败；Exact McNemar 双侧 `p=0.25`，不显著。三个 C3 的首次
无进展事件分别位于 674、676、676 步，均包含一次低成本 fresh-chunk 重规划、一次固定
VLM 票据驱动的语义介入和恰好两次 Flow4/h16 恢复调用。机器可读汇总位于
`artifacts/robodojo/build_tower_controlled_fault_pairs_20260902/summary.json`。

# Harness VLA 借鉴审计与 CARVE 落地方案

更新日期：2026-09-18（下文1--7为8月28日历史审计；当前执行以本节及主计划为准）  
研究目标：在不训练或改动基座 VLA 的前提下，验证 Agentic Harness 与 Optimize Runtime 对具身策略的增益。

## 最新源码核查与落地：RoboDojo工具闭环

### 核查版本

- 官方仓库：[RLinf/RPent](https://github.com/RLinf/RPent)，本地已在`third_party/RPent`，无需重复clone。
- 原检出：`2c1f87d80861a40451efe8e9bf3d64e7c3d3f464`，保持不变，避免影响已有依赖。
- 本次联网fetch：`886b3b274d3dd30bbc15615ea512d65ae90bc8b3`，2026-09-18提交。
  最新文件使用`git -C third_party/RPent show <commit>:<path>`读取，不能把原检出当最新代码。
- 未运行上游服务、下载上游记忆库或安装依赖。上游Apache-2.0；此次借鉴流程、自行实现，
  没有复制它的机器人控制代码。若以后直接复用源码，保留原许可证与作者标注。

### 读到的实现，而非只看框架图

| 机制 | 上游实际实现 | 我们的取舍 |
|---|---|---|
| 工具返回最新状态 | `rpent/tools/toolkit.py::execute_tool`对改变环境的操作捕获新状态；`ToolResult`把图像和文字结果交给模型，只读工具跳过状态采集 | 采用动作后反馈、再决策的流程；不对每个50步正常块增加Planner，而对主动执行工具立即返回控制权 |
| VLA是有退出条件的接触工具 | `robots/libero/tools.py::pi0_pick`检查下降后抬升、夹爪开度和块预算；`move_to`使用OSC控制，另有release/rotate等 | 学习预算、回执和条件退出，不能把LIBERO阈值/7维OSC动作复制到ARX X5的14维绝对关节接口 |
| 任务与全局经验 | 最新`rpent/memory/manager.py`维护global/suite/task_only、证据来源与冲突归档；评测只读，探索写入 | 当前先补回合内查询与真实执行回执；之后再做冻结跨回合经验，不把测试回合写入评测记忆 |
| 记忆并非自动保证真实 | 上游合并按记录数/任务数分配置信级别，经验正文冲突另存；这是管理规则，不是语义正确性证明 | 保留我们对核验和来源的约束，不按“出现多次”直接晋升当前VLM幻觉 |
| 感知与控制的前提 | `back_project`依赖同相机、同分辨率深度和标定；LIBERO guide要求重新定位，禁止历史xyz直接回放 | 不增加相机或启用新深度通道，不绕过公开观察取物体真值；几何工具暂未接入 |
| 运行时分离 | 模型服务与工具执行分离，可复用服务进程；工具返回新状态，避免重复读取 | 保留现有观察缓存及进程分离；本次不宣称新增推理加速，只避免反馈后重复Critic |

最新源码入口：
[工具执行](https://github.com/RLinf/RPent/blob/886b3b274d3dd30bbc15615ea512d65ae90bc8b3/rpent/tools/toolkit.py)、
[LIBERO动作](https://github.com/RLinf/RPent/blob/886b3b274d3dd30bbc15615ea512d65ae90bc8b3/robots/libero/tools.py)、
[记忆管理](https://github.com/RLinf/RPent/blob/886b3b274d3dd30bbc15615ea512d65ae90bc8b3/rpent/memory/manager.py)、
[记忆文档](https://github.com/RLinf/RPent/blob/886b3b274d3dd30bbc15615ea512d65ae90bc8b3/docs/source-en/rst_source/development/memory.rst)。
当前上游文档说探索写入模式仅支持LIBERO，不能由此推断其所有机器人均已实现自主学习。

### 本项目实际修复

当前排序适配器此前有三个落地缺口：retrieve_memory忽略query；没有把实际工具回执
填入Planner已有last_primitive字段；反馈工具后没有立刻将决策权交回Planner。

本轮修复为：

1. 有上限的词项相关检索，相关度相同时优先近期记录；无匹配返回空，不伪装向量检索。
2. 分别保存模型假设与真实执行回执；回执包含call_id、执行时刻、执行/丢弃步数及后验检查，
   技术执行成功不等于语义完成。无记忆消融禁用记忆返回，但仍可看到本次动作回执。
3. 反馈工具完成后，在下一次VLA动作前立即请求Planner；已有同目标Critic结果随回执传递，
   不重复轮询其他谓词。普通VLA块保持原有周期/风险调度，不改为每步调用大模型。
4. 仿真启动源码哈希补入记忆实现文件，避免实验缺少实际生效源码记录。

验证：297项CPU测试通过（`artifacts/robodojo/rpent_tool_feedback_20260918/tests.xml`）。
其中合成端口测试证明100步选择反馈后只运行10步，第110步Planner即能停止；重新观察
在第101步交回。包含终止后不再调用、原指令保持、预算、回合隔离与记忆消融。
这是软件控制流验证，不是模型能力、物理恢复或机器人成功率。没有新增GPU/仿真回合。

### 仍需解决的关键问题与下一步

本轮修复让“工具结果 -> 下一步决策”真正连通，但仍未实现可靠的重新抓取或几何纠正。
不能用“参考仓库已下载”或“297测试通过”声称完整Harness VLA能力已经移植。

当前端口只导出原始三RGB与14维关节状态，未提供定位到世界坐标的工具。
RoboDojo上游`src/eval_client/eval_env.py::take_action`确实区分joint/ee输入，
`env/robot_manager/robot_manager.py::solve_ik`也存在；这不等于当前端口已准入这些能力。
下一步先核对已有公开机器人状态、坐标系/单位、末端控制约束与IK失败语义，
可考虑在原场景内的小范围姿态恢复，但不得读取隐藏物体位姿、自动启用新传感数据
或抄入任务专用成功轨迹。若现有观察不足以支持工具参数，就明确不支持而不是猜坐标。

推进顺序：同输入视觉服务核对与动作接口审计 -> 单工具开发验证 -> 小规模完整闭环 ->
无/有记忆、无/有Optimize消融。保留advisory与默认关闭的feedback_refresh，
单独注册开发配置后才开启候选，不覆盖原75/25对照。自进化仍为后续重点，不启动训练。

---

以下保留早期审计与当时第二VLA计划，不作为当前扩跑RoboTwin的指令。

## 1. Harness VLA 的核心不是“给 VLA 加一次重试”

Harness VLA 将冻结的 VLA 作为具身智能体工具链中的一个接触操作工具，而不是让 VLA 独自负责完整任务。高层编码智能体持续观察环境、读取任务记忆、选择工具、检查工具返回值并决定下一步。工具大致分为：

- 感知与定位：读取多视角图像、分割目标、查询三维位置；
- 解析运动：自由空间移动、腕部旋转、夹爪控制和释放；
- 学习策略：仅在抓取、插入、精细接触等阶段调用冻结 VLA；
- 过程控制：验证、恢复、结束和失败记录；
- 持续记忆：复用成功轨迹中的任务知识，同时根据当前观测重新定位，而不是盲目回放动作。

因此，其主要提升来源是“规划器 + 工具 + 记忆 + VLA”的组合，而不是重新训练 VLA。

## 2. 最值得 CARVE 直接吸收的五点

### 2.1 冻结 VLA 的训练无关评测

Direct 与 Harness 使用同一冻结检查点、相同任务和相同随机种子。Harness 只改变测试时编排，因此能把增益归因于智能体系统，而不是额外训练数据。这与 CARVE 的定位完全一致。

### 2.2 VLA_ACT 与解析 primitive 分工

解析 primitive 处理确定性强、几何结构清晰的非接触阶段；VLA_ACT 处理接触丰富且难以手写控制的阶段。CARVE 应保留自身的风险监测和运行时调度，但把执行接口统一为可审计 primitive，而不是只把整段任务交给 VLA 后做一次结果判断。

### 2.3 持续工具调用，而非一次性高层规划

Planner 必须在一个 episode 内持续读取工具结果并重新决策。任务开始、primitive 返回、语义异常和恢复失败是明确的规划边界。低层控制期间不应同步等待大模型。

### 2.4 任务记忆与当前观测解耦

记忆保存策略、工具顺序、参数范围和失败模式；坐标与对象状态仍从当前观测获取。该设计既能降低重复规划开销，也避免把一个 seed 的位姿硬编码到另一个 seed。

### 2.5 配对随机种子和分层证据

同一个 seed 上比较 Direct 与 Harness，并记录失败转成功、成功转失败和失败类型。除成功率外，同时报告 VLA 调用数、解析动作占比、工具调用数、规划延迟、控制步数和 wall time。

## 3. CARVE 保留的独立价值

CARVE 不应退化为 Harness VLA 的复刻。以下能力继续作为自身主线：

- 高频、低成本的 ExecutionRiskMonitor，负责执行信号与时序风险；
- 低频、事件触发的 VLM Planner/Critic，负责语义正确性与工具规划；
- 有限预算恢复、safe hold、safe stop 和完整审计链；
- 面向多种 VLA 的 capability negotiation 与 ActionSpec 校验；
- Optimize Runtime 的动作块截断/复用、风险与截止时间联合调度、延迟分解和降级策略；
- Agentic 与高效推理的联合评测，而非只报告任务成功率。

一句话边界：Harness VLA 主要回答“如何把冻结 VLA 变成持续工作的工具智能体”；CARVE 还要回答“如何让该智能体在风险、延迟和有限算力约束下可靠运行”。

## 4. 不直接照搬的部分

- 不向 Planner 暴露 reward、成功判定器、对象真值位姿或其他模拟器特权状态；
- 不把官方未公开或无法访问的模型检查点当作可复现实验；
- 不把一个成功 seed 生成的脚本直接用于测试 seed 而不重新感知；
- 不将解析 primitive 的动作数隐藏在 VLA 调用统计之外；
- 不把 server 级编译或精度设置写成每请求动态能力。

## 5. 第二 VLA 实验的确定方案

### 5.1 模型与基准

- VLA：公开 `robbyant/lingbot-vla-4b-posttrain-robotwin`；
- 基准：RoboTwin 2.0 C2R 风格随机化任务；
- 动作空间：Aloha-AgileX 双臂 qpos 14 维；
- 训练：不进行额外微调；
- 目的：证明 CARVE adapter/harness/runtime 不绑定 PI0.5，同时复用 Harness VLA 的评测逻辑。

公开 LingBot 检查点不是 Harness VLA 论文使用的内部 EEF16 检查点。因此本实验属于同协议的新适配器验证，不宣称复现论文的 58.4% 数字。

### 5.2 Phase A：接入与小规模配对验证

任务：`open_microwave`、`click_bell`、`stack_blocks_three`、`place_shoe`、`put_object_cabinet`。

- 每个任务 1 个 clean seed，用于验证任务可执行性和生成任务记忆；
- 每个任务 5 个固定 randomized seeds；
- 方法：Direct、CARVE Harness、CARVE Harness + Optimize；
- 规模：5 个 bootstrap episode + 75 个 randomized episode；
- 所有方法使用完全相同的模型、任务 seed、终止条件和动作预算。

### 5.3 Phase B：论文级扩展

若 Phase A 通过，再扩展到 50 个任务、每任务 5 个 expert-verified randomized seeds。Direct 与 Harness 各 250 个 episode，共 500 个配对 rollout。Optimize 可在代表性任务子集上做独立运行时消融，避免把效率实验与 Agentic 成功率混在一起。

## 6. 决策门槛

只有同时满足以下条件，第二 VLA 证据才进入主结果：

1. 公开模型在本机完成真实 GPU 推理；
2. RoboTwin 真实物理仿真完成闭环动作并保存视频；
3. Direct 与 CARVE 使用相同 seed 和动作预算；
4. Planner 不读取模拟器特权信息；
5. 结果包含逐 episode 原始记录、配置、版本、延迟与失败转移；
6. 至少出现可解释的失败转成功，且成功转失败受到约束。

若成功率没有提升，仍可将该实验作为 adapter 与运行时通用性的边界证据，但不能宣称 Agentic 泛化增益。

## 7. 当前落地状态

- 已固定 LingBot、RoboTwin、XPolicyLab 和 C2R seed manifest 的版本；
- 已实现 LingBot-VLA qpos14 adapter、动作语义校验和单元测试；
- 已加入 LingBot Optimize Runtime 模型插件，区分服务级去噪/编译配置与请求级动作提交长度；
- 已通过项目完整测试集（278 tests）；
- 正在准备公开模型、RoboTwin 资产与隔离运行环境；
- 尚未完成 LingBot + RoboTwin 的真实闭环 rollout，因此当前不能报告第二 VLA 成功率。

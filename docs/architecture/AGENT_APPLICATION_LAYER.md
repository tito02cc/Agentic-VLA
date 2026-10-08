# Agent 应用层：同一项目支撑研究与求职

更新：2026-09-09。实际目录：`/home/admin1/ct/CARVE-VLA`。

## 1. 目标与边界

一个项目、一套模型/工具/记忆接口、两种交付：论文需要可解释的机制与受控实验；
求职需要可运行的系统、可靠的接口、测试及清楚的工程取舍。不给求职另建假机器人
演示，也不把 LangGraph、FastAPI 或数据库本身包装成算法创新。

| 求职方向 | 共享成果重点 | 不应越界的说法 |
|---|---|---|
| AI 应用算法 | 多模态规划、经验检索、选择性调用、机制消融 | 不等于基础模型预训练能力；研究效果以有效实验为准 |
| VLA / 具身智能 | 观测与动作契约、模型适配、闭环执行、恢复与仿真 | 不把 API 通了当成物理技能有效，不宣称实机已经完成 |
| Agent 开发 | 模型与工具接口、状态、记忆、请求校验、失败处理、服务化 | 不把多个 Python 节点等同多个自主学习 Agent |
| AI Infra / 部署 | 模型配置与质量门、时延分析、资源隔离、请求回执与观测 | 目前偏应用推理基础设施，不宣称分布式训练、CUDA 内核或生产集群能力 |

## 2. 分层与实施状态

```text
外部 Codex / 其他 Planner 的结构化工具请求
                         |
可选 FastAPI 接口：认证、严格输入、状态查询
                         |
LangGraph：admit -> dispatch -> persist_receipt
                | 重复请求只返回历史回执
                | 未知执行结果阻止自动重放
                         |
ExternalToolAgentGateway -> CarveAgentSession -> CanonicalToolRuntime
                         |
          现有 Harness / Memory / Optimize / PolicyAdapter
                         |
                 已绑定的环境或服务
```

本次新增的是**外部 Agent 的受约束工具调用服务**，不是新的 VLM Planner，
也不是已经完成的 `plan -> execute -> verify -> replan` 自主规划系统。
LangGraph 管理一次工具请求的流程；SQLite 保存请求回执，**不保存完整图 checkpoint，
不恢复机器人物理状态，也不替代语义/过程记忆**。

应用入口由运行环境的拥有者构造并绑定当前 Session。它不会启动/停止任意进程，
不会从 HTTP 接受模型路径、shell 命令、机器人动作向量或评估器状态。
现有 benchmark 入口没有导入应用层，默认执行路径不变。

## 3. 已实现的工程机制

- FastAPI：`/tools`、`POST /requests`、`GET /requests/{request_id}`、认证后的
  `/openapi.json`；`/health` 只表示接口存活，明确不表示机器人健康。
- 显式 Bearer Token；缺省只开放 observe/retrieve_memory/verify。动作与生命周期
  工具需要服务拥有者显式设置 `allow_execution=True`，仍受原 Harness 权限约束。
- 请求必须携带稳定 request_id、episode_id、timestep；严格拒绝未知字段、直接动作、
  非法类型与非有限 JSON 数值。新鲜度/安全边界仍由现有 Gateway/Tool Runtime 判断。
- SQLite 将请求 ID、完整规范化请求的哈希、状态和回执持久化。同 ID 同请求只取
  历史回执，不重新执行；同 ID 不同请求拒绝。
- 一个 ledger 同时只允许一个未结束请求；两个服务实例也不能并发派发。
  若执行后、回执落盘前崩溃，pending 记录继续阻止新执行。无法判断真实结果时，
  标记 indeterminate，不能仅因客户端重试或进程重启就再次执行动作。
- 回执绑定 run/config/tool catalog 和读写权限。不能拿另一模型配置的 ledger 接着跑。
- `completed` 是工具请求处理完成，不是机器人任务成功；成功与失败以原工具响应及
  官方评估分别判断。`replayed=true` 表示历史回执，不是新观察。

这不是物理动作 exactly-once 保证：数据库与机器人无法原子提交。本方案在结果不确定时
停止新派发，牺牲可用性来避免自动重复动作。未决操作须由运行拥有者检查实际状态，
重新观测并建立受控新会话；不提供“清空 pending 后强行继续”的 HTTP 接口。

## 4. 接入与运行

新增文件位于 `agentic_vla/application/`，不修改基座模型权重、提示词、动作步数、
量化精度或 benchmark 配置。使用独立 `.venvs/agent-app`，不向仿真/模型环境装依赖。

从项目根目录安装与检查：

```bash
python3 -m venv .venvs/agent-app
.venvs/agent-app/bin/python -m pip install -r agentic_vla/application/requirements.txt
.venvs/agent-app/bin/python -m pytest -q tests/test_agent_application.py
```

对一个**已经初始化、已经绑定真实执行器**的外部工具 Agent 会话，接入方式为：

```python
import os
import uvicorn
from agentic_vla.external_agent import ExternalToolAgentGateway
from agentic_vla.application.control_plane import LangGraphToolService
from agentic_vla.application.api import create_app

# session and current_tool_context come from the live environment owner.
gateway = ExternalToolAgentGateway(session, context_source=current_tool_context)
service = LangGraphToolService(
    gateway,
    ledger_path=session.workspace.root / "application_requests.sqlite3",
    allow_execution=False,
)
app = create_app(service, api_token=os.environ["CARVE_APP_TOKEN"])
uvicorn.run(app, host="127.0.0.1", port=18090, workers=1)
```

代码段是接入示例，不是独立的仿真启动脚本。不得用测试 fixture 冒充模型执行器。
API 的同步工具函数在服务线程池执行；非线程安全的仿真需要通过其所属线程/RPC
串行调度。机器人主循环必须保持对步进、安全边界及 reset 的所有权，不能与 HTTP
并发无协调地推进物理状态。正式桥接前不打开动作权限。

本版本为单拥有者、localhost 实验服务；不是多租户平台，不包含任务队列、全任务
取消/恢复、SSE UI、MCP server 或 LangSmith 云端接入。停止 HTTP 等待不等于停止
机器人动作；安全停止仍必须使用适配器的控制机制。

## 5. 不影响原实验的验收门

1. **默认路径隔离**：旧入口无需 LangGraph/FastAPI；应用依赖不进入模型环境。
2. **协议等价**：相同请求与上下文下，对照直接 Gateway 与图编排的工具输出、
   接受/拒绝、推理控制参数；随机 call ID、日志时间与耗时不要求相同。
3. **故障可靠性**：重复请求、并发、过期请求、未完成回执、权限越界均有回归测试。
4. **真实接入门（待做）**：先固定开发布局和相同模型/Planner 配置，小规模比较
   直连与服务入口；记录额外时延、调用次数及是否发生非预期执行。
   小样本通过不等于统计上证明零退化。
5. **效果实验（待做）**：只有会改变规划/恢复/记忆/优化策略的版本，才进入新的
   冻结条件。新增结果不与旧版本混算，正式留出布局不得用于接口调参。

服务层不会保证成功率提高。减少重复派发、保存故障证据和隔离职责可以改善工程质量，
但论文中的任务收益仍须通过 Agent/Memory/Optimize 消融建立。接入增加的 HTTP、
图编排、数据库开销需要单列，不能隐藏在“模型推理加速”指标后面。

## 6. 实施顺序与停止扩张条件

- A：完成可选工具服务与协议验证，作为面试可解释的工程增量；不扩到通用 SaaS。
- B：回到 RoboDojo F2--F5，把真实语义决策、受限工具和工作/持久记忆接到执行路径。
- C：在既定候选任务上通过短闭环功能门，冻结协议并完成 Agent/Memory/Optimize 对照。
- D：从同一有效证据整理论文、毕业技术报告、简历和原始仿真演示视频。

具体任务和候选样本量仍以 `../plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md` 为准。
不因为求职新增 benchmark，不为四类岗位分别另起项目。完成这轮应用层增量后，
优先修复研究主线的执行缺口，不继续添加前端、K8s、分布式服务等非必要内容。

## 7. 核验记录

2026-09-09 本轮定向回归结果：

| 测试范围 | Python 环境 | 结果 | 原始记录 |
|---|---|---|---|
| 应用层、Session、工具、配置、Planner、VLA primitive 和 adapter | `.venvs/agent-app`，Python 3.13 | 166 passed，另有 3 个 subtests | `../../results/agent_application_20260909/application_and_core.xml` |
| RoboDojo bridge、持久记忆、reset、回合汇总和进程生命周期 | `openpi/.venv`，Python 3.11 | 362 passed | `../../results/agent_application_20260909/robodojo_regression.xml` |
| Runtime、动作复用、观测记录、任务台账、知识记忆 | `openpi/.venv`，Python 3.11 | 70 passed | `../../results/agent_application_20260909/runtime_and_memory.xml` |

三个不重叠测试清单共 **598 passed**，其中新增应用层测试 **30 项**。
Ruff、应用环境 `pip check` 和工作树 `git diff --check` 通过。
Starlette TestClient 有一项 AnyIO API 弃用警告，不影响本次断言。

这不是全仓库测试通过声明：`tests/test_carve_optimization.py` 仍导入已不存在的
`scripts.benchmark_carve_openvla_profile`，本轮没有更改这个历史测试，未计入上述通过数。
最初将仿真测试放入 CPU 应用环境时缺少 cv2/msgpack；之后在未改依赖的原 OpenPI
环境重跑对应桥接测试并通过，没有为应用层修改仿真依赖。

所有新增测试使用明确标注的 contract test fixture，真实执行 LangGraph 与 FastAPI
TestClient，但没有运行 VLM/VLA 模型，没有生成新的机器人成功率。本次没有启动
HTTP 常驻服务或正式 benchmark；真实环境的服务接入与效果对照仍待验收。

## 8. 技术依据

- LangGraph 图编排接口：https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph 持久化与 checkpoint 边界：https://docs.langchain.com/oss/python/langgraph/persistence
- FastAPI 同步接口与线程池：https://fastapi.tiangolo.com/async/

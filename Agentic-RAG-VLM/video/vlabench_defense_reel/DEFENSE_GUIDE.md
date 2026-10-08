# Agentic RAG-VLM 展示材料使用说明

## 当前展示包由什么组成

答辩时把证据分成两层，不要把它们混成一个“成功率”：

1. **100 个方法 rollout 的固定种子配对设计**：用于回答 RAG、场景图、memory 和 replan 是否分别有效；
2. **3 条完整 VLABench 任务**：用于回答高层 Agent 决策能否通过 IK/技能执行器落到连续操作。

正式定量口径为：10 个冻结种子、100 个方法 rollout、246 次真实本地
Qwen3.5-4B 多模态调用、260 个公开事件、60 次 Agent 工具调用。Full Agent 在 G1–G4
均为 10/10。完整实验解释与声明边界见
`../../docs/CURRENT_EXPERIMENT_PROGRESS.md`。

推荐现场顺序：先用一页消融表说明“为什么有效”，再播放本视频说明“如何完成任务”，
最后停在健康冷饮的纠错链路接受提问。

## 推荐播放方式

- 在 PPT 中整页嵌入本项目的 84 秒总览视频：
  `renders/agentic_rag_vlm_vlabench_defense_reel_v1.mp4`
  （84.000 s / 1920×1080 / 30 fps / 2520 帧 / 无音轨，
  SHA-256 `1f5a84a2e9cea1dbc7384884e6c131e56f88046a8c32f4583f39abae0767f836`）。
- 视频无旁白、无背景音乐，现场按下列时间轴讲解即可。
- 如果被追问动作细节，切换到 `assets/` 下的三条原速视频。
- strict check 报告已持久化于 `checks/`，项目 HyperFrames pin 为 0.8.33。

## 84 秒讲解词

| 时间 | 画面 | 建议讲法 |
|---|---|---|
| 0–7s | 片头 | “我们不只验证机器人能不能抓到，而是验证它为什么能在复杂任务里做对。” |
| 7–17s | 六步运行时 | “Agentic RAG-VLM 负责观测、规划、检索、约束、验证；VLABench 的 IK 和技能库负责低层关节动作。” |
| 17–36s | 化学配制 | “这个任务验证顺序和易碎容器约束：系统选择 HCl 后 NaOH，到达烧瓶前保持试管竖直，倾倒时验证管口对准，完成后把两支试管都放回原槽。” |
| 36–51s | 家庭烹饪 | “视觉候选里有 broccoli、cheese 和 fish；配方检索排除 fish，任务记忆逐项确认两种目标食材已经放入托盘。” |
| 51–76s | 健康饮料 | “开门后模型把 Monster 当成健康饮料。我们保留这个原始错误，再由健康运动后补水的意图知识卡纠正为 juice，完成抓取、放置、关门和结果验证。” |
| 76–84s | 结果收束 | “三组任务分别验证约束推理、经验检索和在线纠错。三条选取任务的最终判据均通过，累计 1,364 次主 IK 求解全部接受，未触发 IK、硬关节跳变或 PhysicsError 门禁。” |

这句只描述已报告的三类执行门禁，**不等于“全程无碰撞”，也不是玻璃安全认证**。
VLABench 没有玻璃断裂模型，Franka 夹爪是二值位置控制，知识卡的力与开口值仍只是记录项。

### 2026-09-10 素材与数字更新

化学素材已从 canonical 44.4 s 轨迹换为修复后的 66.2 s 轨迹
（`artifacts/vlabench/kiro_20260910/chem_seed005_hcl_tag_fix/`）。变化点：

- 20 个规划技能**全部实际执行**（原为 14/20，第二次倾倒达标即被截断）；
- 两支试管都完成归位与松手，由物体状态验证（HCl 偏差 2.2 mm / 倾角 0.05°，
  NaOH 9.6 mm / 8.12°）；
- 原先与 `HCl_tag` 名牌相关的异常接触**已消除**（unexpected 1 → 0，
  异常力代理 66,908.92 N → 0 N）；名牌最小净空 +94 mm；
- 三条轨迹 IK 合计由 1,166 变为 **1,364**（化学 603 + 烹饪 304 + 冷饮 457），
  仍为 100% 主解接受、三类门禁 0；
- 严格管口验证仍双双通过，NaOH 最小管口误差由 1.39 mm 改善到 0.53 mm。

canonical 素材保留为 `assets/chemistry_canonical_44s.mp4` 与
`assets/chemistry_fast_canonical_44s.mp4`，可随时回退。

被追问 `HCl_tag` 时的准确说法：那是 VLABench 场景里一块水平平放的名牌薄板
（60×28.9×0.40 mm，`contype=1`），旧回槽航点把它当空气穿过去了；
记录到的 66,908.92 N 是 `solref="0.001 2"` 配 1 ms 步长的求解器刚度尖峰，
不是玻璃受力（同一位姿静态复现只有 0.216 N、侵入 0.027 mm）。
修好航点后接触本身不再发生。

## 一页 PPT 文案

标题：**Agentic RAG-VLM：从视觉候选到可验证操作决策**

主流程：`Observe → Plan → Retrieve / Constrain → Execute → Verify → Replan`

三组证据：

1. **化学配制**：目标顺序 + 易碎品姿态约束 + 瓶口对准验证。
2. **家庭烹饪**：配方知识检索 + 干扰物过滤 + 多目标任务记忆。
3. **健康饮料**：在线重观测 + VLM 错误保留 + 意图 RAG 纠错 + 闭环验证。

结论句：**Agentic RAG-VLM 的独立价值不在于替代低层控制，而在于为执行器提供可解释、可约束、可纠错的任务级决策。**

## 被追问时的边界回答

**问：这是不是端到端 VLA？**

不是。本实验刻意隔离 Agentic RAG-VLM 的单独价值：高层由多模态智能体决定目标、顺序、约束与是否需要纠错；低层动作由 VLABench 已注册的 IK/技能执行。这样才能把收益归因到检索、约束和自反思，而不是策略网络训练。

**问：为什么不是只做抓取成功率？**

因为框架解决的是复杂操作中的经验误检索、空间/语义约束缺失和失败后难以恢复。三组任务分别把这三类能力变成可观察的决策事件，而不是重复三个 pick-and-place。

**问：冰箱场景为什么最重要？**

它包含真实在线 VLM 误判。系统没有隐藏错误，而是记录原始候选，依据用户“运动后健康冷饮”的意图检索知识卡，把 Monster 纠正为 juice，再通过关门与目标位置完成闭环验证。

## 证据索引

| 场景 | 原速视频 | 结果记录 | 关键证据 |
|---|---|---|---|
| 化学配制（视频采用） | `/home/admin1/ct/CARVE-VLA/artifacts/vlabench/kiro_20260910/chem_seed005_hcl_tag_fix/episode_success_True.mp4` | `/home/admin1/ct/CARVE-VLA/artifacts/vlabench/kiro_20260910/chem_seed005_hcl_tag_fix/result.json` | 20/20 技能执行；两次倾倒通过联合管口验证；两支试管归位；603 次 IK 接受；unexpected 接触 0。 |
| 化学配制（canonical，保留） | `/home/admin1/ct/CARVE-VLA/artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_005/execution_interpolated_ik_full_v7/episode_success_True.mp4` | 同目录 `result.json` | 14/20 技能；405 次 IK 接受；记录 1 次 `HCl_tag` 异常接触。 |
| 家庭烹饪 | `/home/admin1/ct/CARVE-VLA/artifacts/vlabench/agentic_cook_dishes_20260901/seed_007/execution_ik_gate_v3/episode_success_True.mp4` | `/home/admin1/ct/CARVE-VLA/artifacts/vlabench/agentic_cook_dishes_20260901/seed_007/execution_ik_gate_v3/result.json` | recipe RAG 选择 broccoli + cheese；304 次 IK 接受。 |
| 健康饮料 | `/home/admin1/ct/CARVE-VLA/artifacts/vlabench/agentic_take_out_cool_drink_20260902/seed_000/execution_v2/episode_success_True.mp4` | `/home/admin1/ct/CARVE-VLA/artifacts/vlabench/agentic_take_out_cool_drink_20260902/seed_000/execution_v2/result.json` | 两次在线 Qwen RGB 观测；Monster → juice 纠错；457 次 IK 接受。 |

模型配置需要如实区分：化学与烹饪规划证据使用 Qwen3.5-4B；健康冷饮的两次在线重观测
使用 Qwen3-VL-2B NF4。三者共享同一个 Agentic 接口和决策—执行边界，但不是同一个模型配置。

## 不建议在主展示中使用

- `find_unseen_object`：VLABench 抽屉/柜体物理会把目标物体弹出，不能作为正向成功证据。
- 光华 URDF 旧演示：当前 VLABench 三任务在抓取完整性、执行稳定性和框架归因上更适合作为主展示。

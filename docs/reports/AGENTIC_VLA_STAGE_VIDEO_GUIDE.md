# Agentic VLA 阶段展示视频说明

证据截至2026-09-18。本说明对应[完整技术报告](AGENTIC_VLA_TECHNICAL_REPORT.md)，
仅整理已有录像，没有新增实验。所有MP4均为原始仿真视频，无封面、字幕、拼接或变速。

## 1. 先看哪几段

| 顺序 | 视频 | 模型与模式 | 官方结果 | 用途 |
|---|---|---|---|---|
| 1 | [整理桌面Agent：头部](../../third_party/robodojo_official/eval_result/RoboDojo/organize_table/AgenticPi05/arx_x5/0_official_pi05_conservative/2026-09-18_14-49-05_agent_on/episode_0000000_cam_head_fail.mp4) | 官方π0.5 + Qwen3-VL-4B NF4 + 受限Harness | 75/100；失败 | 最新主展示，展示真实执行和剩余问题 |
| 2 | [整理桌面原生：头部](../../third_party/robodojo_official/eval_result/RoboDojo/organize_table/Pi_05/arx_x5/0_official_pi05_59999_native/2026-09-15_10-28-49_pi05_native/episode_0000000_cam_head_fail.mp4) | 同官方π0.5；无Agent | 75/100；失败 | 了解原始动作模型能力，不是同步消融对照 |
| 3 | [语言分类原生：头部](../../third_party/robodojo_official/eval_result/RoboDojo/classify_objects_by_language/Pi_05/arx_x5/0_official_pi05_59999_native/2026-09-15_10-38-05_pi05_native/episode_0000000_cam_head_fail.mp4) | 同官方π0.5；无Agent | 0/100；失败 | 说明第二任务和低层动作能力瓶颈；不能称Agent失败 |
| 4 | [整理桌面Agent：左腕](../../third_party/robodojo_official/eval_result/RoboDojo/organize_table/AgenticPi05/arx_x5/0_official_pi05_conservative/2026-09-18_14-49-05_agent_on/episode_0000000_cam_left_wrist_fail.mp4) | 与第1段同回合 | 与第1段同结果 | 查看操作细节和局部遮挡 |
| 5 | [整理桌面Agent：右腕](../../third_party/robodojo_official/eval_result/RoboDojo/organize_table/AgenticPi05/arx_x5/0_official_pi05_conservative/2026-09-18_14-49-05_agent_on/episode_0000000_cam_right_wrist_fail.mp4) | 与第1段同回合 | 与第1段同结果 | 查看另一侧局部观察 |

这五段对应三个回合，而不是五次独立测试。前四路整理桌面视频各约40.04秒；
语言分类约44.04秒。历史PI-v3成功/失败及人工子指令诊断录像在报告附录C中另列；
展示包将它们放在“历史参考”，不替代当前官方π0.5的实验结果。

## 2. 最新整理桌面回合：如何解释

运行ID：`2026-09-18_14-49-05_agent_on`。官方场景layout set0/id0、seed0，
π0.5 checkpoint59999，原生JAX flow10/horizon50，原三路RGB与14维关节状态。
VLA接收原始完整任务指令，没有换成手写成功动作序列，也没有增加观察相机。
VLM使用Qwen3-VL-4B视觉保留NF4；Critic只给建议，没有自动确认阶段的权限。

### 任务是什么

原始语言指令包含：把鼠标放到鼠标垫上、键盘推入框内、摆件放到支架上、闹钟放到
抽屉柜上，以及打开抽屉、放入剩余杂物。官方评分主要检查四类物体放置，并包含
夹爪和机器人归位条件；**官方评分、整任务成功、完整语言要求满足程度不是同一概念。**
视频看起来“差不多完成”不能代替官方精确位置、姿态与归位判断。

### 哪个时间点能看见什么

| 视频位置（约） | 控制步 | 可说明的事实 | 不应解释成 |
|---|---:|---|---|
| 00:00 | 0 | 原始桌面及机器人初始状态 | 已经有可靠对象识别或完整空间关系图 |
| 00:08.00--00:08.04 | 200--201 | 日志证实Planner选择reobserve，保持关节目标1步，获得新观察后重新规划 | 机器人发生抓取修正或已经恢复成功 |
| 00:08.04 | 201 | 抽查画面中鼠标仍在垫外、键盘未就位、抽屉关闭；Planner却产生完成误判 | 模型口头说完成就是任务事实 |
| 00:40.00 | 1000 | 最终部分物体已移动，官方75分且失败；抽屉仍关闭 | 75%成功率或完整语言任务已经完成 |

时间按录像帧号/25估计，用来定位画面，不是日志wall-clock时间。一个控制步的保持
在视频中可能几乎看不出来；Agent是否调用工具应由回执佐证，不应靠画面想象。

**控制步201：用于核对模型的完成误判，不是成功截图。**

![工具返回后的原始头部画面](../../artifacts/robodojo/organize_tool_return_validation_20260918/head_frame_201.png)

**控制步1000：部分物体已就位，仍以官方75分、失败为准。**

![回合结束时的原始头部画面](../../artifacts/robodojo/organize_tool_return_validation_20260918/head_frame_1000.png)

### Agent到底做了什么

- 20次VLA调用，执行999步VLA动作；另1步为reobserve，合计1000控制步。
- 11次Planner：1次初始vla_act、9次continue、1次run_skill（reobserve）。
- 9次Critic：7次inconclusive、2次原始confirmed，全部仅建议，未确认任何任务阶段。
- 工具结果进入下一次请求和过程记忆；执行回执为“已获得新观察”，不是“任务已成功”。
- 事件29为工具结果，事件33为随后Planner结果，事件35才执行后续动作块，六项时序审计通过。
- feedback_refresh候选虽开放，但没有被选择；不能将其描述为已经发挥纠错作用。

来源：[原始回合汇总](../../artifacts/robodojo/organize_tool_return_validation_20260918/run01/summary.json)、
[事件日志](../../artifacts/robodojo/organize_tool_return_validation_20260918/run01/sessions/2026-09-18_14-49-05_agent_on/events.jsonl)、
[工具与性能审计](../../artifacts/robodojo/organize_tool_return_validation_20260918/audit_metrics.json)。

### 时间与推理优化不能从播放速度推断

主视频40.04秒，真实回合耗时428.597秒，含服务启动与清理为619.893秒。
录像按25FPS保存控制帧，没有等比例保留所有模型等待。Planner累计75.855秒，Critic
累计24.853秒；π0.5首次调用约36.159秒，后续19次调用P95为185.395毫秒。
这不是历史PyTorch路径的54.67毫秒，也不是已经证明实时控制或全系统加速。

## 3. 最新修复没有新的机器人录像

工具返回回合之后，进行了记忆上下文隔离、Grounding DINO检测、9B身份核验和
目标无关类型检查。它们使用保存的真实录像帧做只读诊断，没有驱动机器人。
因此本包不会把这些修复贴到旧录像上，冒充“新版本已跑通”的视频。

最新12组配对诊断：无目标错误定位接受2/5→0/5，但清晰图正确目标保留5/5→4/5，
累计VLM调用成本2.87倍，未通过准入。它降低特定误认的同时也误拒绝真实目标，
不是成功率提升。六个旧开发帧加六个既有视频的新时间点，不是独立留出任务。
来源：[本轮结论及全部协议入口](../../artifacts/robodojo/organize_target_rejection_20260918/README.md)。

## 4. 可直接用于汇报的说明

> 这段视频展示的是在官方RoboDojo环境中，用官方预训练π0.5执行整理桌面任务，
> 外部VLM和Harness管理规划、观察工具与执行反馈。整个回合真实运行了1000步，
> 最终得到75分，未达到官方完整成功条件。我们验证了工具调用、执行回执、记忆与
> 即时重新规划链路，同时也发现高层模型会误判完成状态，目前还没有证明有效纠错
> 提高了任务成功率。后续先提升判断可靠性和纠正工具能力，再做固定预算的配对对照。

## 5. 展示与传输

先完整解压，再打开技术报告HTML或“视频/index.html”，无需服务器和网络。
HTML中的视频播放器使用包内原始MP4，不使用远程演示地址。播放器能打开不代表
本轮做过真机实验；全部材料均为仿真。若浏览器不支持播放，直接用本地播放器打开MP4。

本包为阅读与汇报材料，不含PPT、模型权重或完整运行环境。主报告与本说明链接到的
核查资料随包保存；未附带的二级来源标为仓库路径。机器日志中的原工作站绝对路径
为原始溯源信息，迁移电脑后不会自动变成可执行路径。文件哈希与视频解码核查见包内清单。

# RA-L实验收口判断（2026-09-24）

## 判定

**当前可形成可复核的系统技术稿，但尚不能诚实地宣称一版冻结的Agentic
Harness与Optimize Runtime在同一闭环中同时改善任务成功和总成本；
不建议把当前数字作为RA-L确认性主结果直接投稿。**

这不是“没有成果”：公开示教身份记忆在`VideoUnmaskSwap`的独立初态
出现救回，系统具备完整的动作块边界规划、记忆准入与安全门槛，PyTorch
PI0.5也有独立的单调用优化数据。但当前冻结版的跨任务迁移和高层调用
调度都存在质量回归，另一任务族的Raw/Harness对照也未复现强优势。

## 本轮新增可审计证据

| 固定试验 | 官方成功 | 配对变化与门槛 | 结论 |
|---|---|---|---|
| VideoUnmask ep22--37，无记忆A / 始终注入公开示教SAM2记忆B | 15/16 / 13/16 | 救回1、伤害3，精确McNemar p=0.625；单目标两臂12/12，双目标3/4→1/4 | 跨任务始终注入**拒绝**；不能把双方成功子集的少调用说成整体加速 |
| RouteStick ep22--37，Raw / 当前冻结Harness | 0/16 / 1/16 | 救回1、伤害0，p=1.0 | 仅孤例救回，Raw处于地板，无法支撑普遍Agentic增益 |
| PickHighlight ep22--37，Raw / 当前冻结Harness | 3/16 / 3/16 | 救回0、伤害0；两臂均成功3对wall 53.43→55.01秒 | 无任务收益，亦非提速 |

两任务合并Raw/Harness为`3/32→4/32`，只有1个不一致配对，
不能推断显著效果。任务是根据历史正向/回归机制有意选的压力测试，
不是随机总体样本。VideoUnmask的12个双方成功配对wall
`152.74→121.41秒`、调用`134→100`，但B的三次任务伤害优先于
这个条件性成本结果；SAM2额外编译约21.67秒（这12对、不含共享加载）。

原始入口：[VideoUnmask审计](../../artifacts/robomme/videounmask_memory_transfer_20260924/analysis.json)、
[VideoUnmask全部视频](../../artifacts/robomme/videounmask_memory_transfer_20260924/rollouts/)、
[双任务审计](../../artifacts/robomme/harness_transfer_gate_20260924/analysis.json)、
[双任务全部视频](../../artifacts/robomme/harness_transfer_gate_20260924/rollouts/)。
两批正式视频分别32和64段，预热各1段，所有正式回合有summary和
可解码视频。详细协议、偏差与停止规则见
[实验计划](../plans/ROBODOJO_AGENT_MEMORY_EXECUTION_PLAN.md)第6.25--6.26节。

## 可保留与必须停止的主张

- 保留：`VideoUnmaskSwap`公开示教身份记忆在确认集`5/12→9/12`，
  救回5、伤害1，精确p=0.21875；这是非显著的**任务内机制证据**。
- 保留：RTX 4090 PyTorch PI0.5独立单调用P95`282.43→54.67 ms`。
  不能嫁接到当前JAX RoboMME整任务或称真机硬实时。
- 停止：始终注入身份记忆到所有任务；选择性VLM调度作为默认部署。
  后者在此前同链路严格11对里`8/11→7/11`、2次伤害，未过质量门槛。
- 停止：追加同配置的RouteStick/PickHighlight或VideoUnmask编号来追
  正例；不能合并历史开发期多版本的八任务80例与当前冻结版，制造
  一个看似确认性的总体收益。

## 下一次只做一个可证伪的主实验

1. **先准入动作策略和任务。** 在不用于最终统计的固定小规模pilot中，
   按预先写明的标准选择：Raw成功既不接近0也不接近100%，并且失败
   可从视频/日志归因为高层目标、顺序或记忆错误，而非纯动作不可达。
   保留disjoint的官方初态作确认集；不因pilot正例挑出单一好任务。
2. **冻结一个必要的Agentic机制。** 优先是“公开证据约束的目标身份与
   阶段确认”，只在动作块边界影响下一条子目标；不得把夹爪闭合等
   低层信号当作任务完成，也不得打断正在执行的VLA动作块。预先写出
   何时不用记忆以及误触发时如何保持原策略。旧`motion_gate`的事后
   判别只能作为设计诊断，不能替代新的闭环确认。
3. **冻结确认与效率门槛。** 同checkpoint、同任务/初态、同服务配对
   Raw/Harness，并再加Harness+Optimize；先报官方成功、救回/伤害，
   质量不过关就拒绝优化配置。通过后比较双方成功配对的全任务wall、
   调用与显存，同时另报失败回合预算耗尽成本和一次性记忆编译开销。
   只有这里出现可重复净收益，才把“联合提升”写进标题/摘要。

当前短稿为[main.pdf](../../paper/CARVE-VLA/ral_draft/main.pdf)，已包含
正反结果；它是诚实的写作底稿，不是已达到录用强度的宣告。

## 代表性原始视频

- `RouteStick ep31`救回：[Raw失败](../../artifacts/robomme/harness_transfer_gate_20260924/rollouts/RouteStick_ep31_R/RouteStick_ep31_vlm_groundsg.mp4)、[Harness成功](../../artifacts/robomme/harness_transfer_gate_20260924/rollouts/RouteStick_ep31_H/RouteStick_ep31_vlm_groundsg.mp4)。这是该任务16对中唯一救回，不能代表平均表现。
- `VideoUnmask ep23`记忆伤害：[A成功](../../artifacts/robomme/videounmask_memory_transfer_20260924/rollouts/VideoUnmask_ep23_A/VideoUnmask_ep23_vlm_groundsg.mp4)、[B预算未成功](../../artifacts/robomme/videounmask_memory_transfer_20260924/rollouts/VideoUnmask_ep23_B/VideoUnmask_ep23_vlm_groundsg.mp4)。B耗尽1300步；另外ep31/35也有反例，详见完整审计。

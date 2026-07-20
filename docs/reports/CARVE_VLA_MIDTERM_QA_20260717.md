# CARVE-VLA 中期答辩问答

本文件用于口头准备，不作为投屏报告内容。

## 1. 这项工作是 Agentic 方法还是推理优化方法？

开题主线是 VLA 高效推理与部署。Agentic Harness 提供长程机器人执行、
风险监测和闭环恢复场景；Optimize Runtime 是当前主要技术贡献，负责
flow-step、动作提交长度、编译、量化、静态视图裁剪和部署门控。两部分通过
明确契约耦合，但 Optimize Runtime 可以脱离 Agentic Harness 使用。

## 2. Agentic Harness 相比普通规则系统增加了什么？

Harness 不是单一重试规则，而是状态化执行系统：它维护近期动作与响应、
恢复预算和历史恢复结果，在重规划、物理恢复、语义升级和安全停止之间进行
有限状态切换。每个 intervention 均与触发证据、动作、验证结果和 deployment
profile 一起记录。

## 3. 90.0% 到 92.5% 是否足以证明 Agentic 方法有效？

该差值是历史 LIBERO-10 aggregate，原始 rollout 目录未保留，不单独作为
统计显著性或当前主证据。当前可复现证据来自精确状态分支、物理恢复验证、
安全停止和在线生命周期 trace。论文不能表述为“普遍显著提升所有长程任务”。

## 4. 为什么不报告后续日志中的 192/200 或 195/200？

`192/200` 来自多组 ProfileAuto 任务结果的汇总，`195/200` 还包含按任务
选择最好 profile。对应原始目录在早期清理中已删除，当前汇总脚本无法复跑。
因此中期主表保留已投稿版本中的 `185/200`，不提升到更强但不可复现的数字。

## 5. SMVE 是否只是删除一张黑图？

计算操作本身是删除已知 padding view，但部署问题包含三个额外条件：

1. 视图由 adapter 的命名契约声明，而不是硬编码索引。
2. 每次调用都断言目标 mask 全假，active view 会使 profile 立即失败。
3. profile 必须通过固定噪声动作 fidelity 和配对闭环门控。

因此当前贡献定位是 control-aware、fail-closed 的 VLA 部署优化，而不是新的
视觉编码算法。

## 6. 为什么 SMVE 能降低时延但几乎不降低显存？

模型参数仍全部驻留显存，SMVE 主要减少第三路 SigLIP 激活计算和对应 prefix
KV 构建。参数显存基本不变，所以结果应表述为 latency optimization，而不是
模型压缩。

## 7. 为什么 DROID profile 要 commit 5，而不是模型生成的 15 步？

生成 horizon 和实际提交 horizon 是不同控制变量。commit 15 的 endpoint L2
达到 `0.18120`，超过预设 `0.10` 门限；commit 5 降至 `0.07658` 并通过全部
10 个 paired states。CARVE 保留 15 步模型输出，但只提交通过门控的前 5 步，
没有修改门限。

## 8. DROID 实验能否证明机器人任务泛化？

不能。DROID 实验使用固定 replay RGB，通过 DROID schema 和 normalization
进入 `pi05_droid`，只验证 checkpoint、adapter、action horizon、fidelity 和
latency 路径。它不是 DROID task-success benchmark。

## 9. 当前框架是否已经支持所有 VLA？

核心契约和 backend registry 不绑定具体模型；SMVE 的当前 sampler plugin 仍是
pi0.5 专用。已经验证两个 pi0.5 checkpoint、LIBERO/DROID 两套 adapter，
并完成 OpenVLA-7B 的第二模型家族 adapter、BF16 reference 和低比特门控。
因此可以声称运行时契约具有跨家族可适配性；不能声称 SMVE 是所有 VLA 的
通用加速算法，也不能声称多个模型家族均完成闭环任务验证。

## 10. 为什么量化通过 replay 后仍然被拒绝？

W8A16 的 45-state replay fidelity 通过，显存从 compiled BF16 的 `6.98 GB`
降至 `6.56 GB`，但时延升至 `71.72 ms` P95，且配对长程实验丢失一次 BF16
保留的 T6 恢复成功。开放环误差不能覆盖状态分布随动作累积变化的闭环风险，
所以它保留为 rejected profile。

## 11. 为什么异步执行降低 deadline miss 却没有被采用？

全异步将 miss 从 `417/3931` 降至 `48/4287`，但成功从 `7/10` 降至
`5/10`。后台推理使用较旧观测，返回动作与机器人当前状态存在 temporal
shift。prefix shadow verifier 的失败 AUC 仅 `0.563`，不足以安全筛选，因此
异步执行不进入当前部署配置。

## 12. 论文中的 realtime 指什么？

报告区分两层指标：

- model-call latency：VLA 单次推理是否满足 80 ms deadline；
- end-to-end control latency：还包含预处理、队列和仿真器 step。

Compiled BF16 与 SMVE 满足 policy-call deadline，但同步控制路径并非每一步都
低于 80 ms。论文不能把模型调用实时性表述为完整机器人硬实时保证。

## 13. 为什么使用 LIBERO，而不是只追最新 benchmark？

LIBERO 提供官方 pi0.5 checkpoint、可恢复模拟器状态和成熟交互评测，适合做
固定状态配对门控。当前目标是验证 frozen VLA 的 Agentic 执行与部署优化，
而不是训练新基座。LIBERO-Plus 或其他新 benchmark 可以用于投稿扩展，但不应
在中期阶段用未微调模型的无效成绩替代已完成证据。

## 14. 下一阶段最关键的缺口是什么？

1. 将当前 admitted profile、Recovery Challenge 和联合实验整理为论文证据链。
2. 投稿前扩展预声明的故障状态和随机种子，补统计区间。
3. 若投稿目标需要，再选择一个新 benchmark 或实机任务补外部有效性。
4. 不继续围绕已拒绝的异步阈值或量化层数做结果导向搜索。

## 15. 如何证明 Agentic Harness 与 Optimize Runtime 不只是两项工作拼接？

联合实验在完全相同的 T6/T9 MuJoCo 状态、固定噪声、物理恢复 skill、flow
steps 和 committed horizon 下，仅替换 PI0.5 runtime profile。Eager、Compiled
和 SMVE 都保持 `2/2` 成功与 `2/2` 恢复验证；runtime P95 从 `166.26 ms`
降到 `65.75/54.50 ms`，80 ms miss 从 `236/236` 降到零。它直接证明
Optimize Runtime 加速的是 Agentic 恢复关键路径，而不是脱离控制任务的模型
microbenchmark。单回合在线结果存在波动，因此行为保持结论仅使用精确状态
配对结果。

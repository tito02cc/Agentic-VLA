# Agentic-VLA 文献地图 V1

本文献地图服务于 `Agentic-VLA` 主稿写作，目标是：

- 组织约 `30` 篇核心参考文献
- 明确每类工作的贡献与不足
- 帮助在 `Introduction / Related Work / Discussion` 中讲清楚“前人做了什么、还缺什么、我们解决什么”

## 1. VLA 基座与通用策略

这类工作证明了 VLA 作为统一机器人策略的可行性，是本文的基座背景。

1. `RT-1`
   - 关键词：大规模真实机器人控制、语言条件控制
   - 作用：建立 VLA/Transformer policy 的早期强基线

2. `RT-2`
   - 关键词：web knowledge transfer、vision-language-action
   - 作用：证明互联网视觉语言知识可以迁移到机器人控制

3. `PaLM-E`
   - 关键词：embodied multimodal language model
   - 作用：强化“大模型 + embodied control”的总体范式

4. `RoboFlamingo`
   - 关键词：open-source VLM adaptation
   - 作用：说明开源 VLM 也可以较低成本迁移到机器人控制

5. `Open X-Embodiment`
   - 关键词：multi-embodiment dataset、RT-X
   - 作用：证明跨 embodiment 大数据对 generalist policy 的价值

6. `Octo`
   - 关键词：open-source generalist policy
   - 作用：强化 open policy 和可迁移 generalist policy 的研究主线

7. `OpenVLA`
   - 关键词：open-source VLA、LIBERO
   - 作用：与你当前实验链路高度相关，是最重要的开源参照之一

8. `FAST`
   - 关键词：action tokenization
   - 作用：代表 VLA 中 action representation 的改进方向

9. `pi0`
   - 关键词：flow-based VLA、general robot control
   - 作用：证明强基座的控制能力和多平台泛化潜力

10. `pi0.5`
   - 关键词：open-world generalization
   - 作用：是你论文的主 baseline 基座，必须重点引用

11. `EveryDayVLA`
   - 关键词：low-cost deployment、LIBERO
   - 作用：体现 VLA 正在走向更实用、更可部署

12. `Green-VLA`
   - 关键词：staged training、RL alignment、long-horizon
   - 作用：代表 2026 年最新更强 generalist VLA 路线

13. `GR00T N1`
   - 关键词：generalist humanoid foundation model
   - 作用：体现 VLA/embodied FM 正在向通用机器人平台扩展

### 与本文的关系

- 这些工作证明“强 VLA 基座”已经成立
- 但它们大多聚焦于训练更强的 backbone、动作表示或数据规模
- 你的论文不主打“训练更大模型”，而是主打：`强 fine-tuned VLA 仍有长程缺口，Agentic inference-time augmentation 可以补上`

## 2. 长程任务、分层规划与推理

这类工作说明：长程任务不是单纯靠 reactive control 就能稳定解决。

14. `CALVIN`
   - 关键词：long-horizon language-conditioned manipulation
   - 作用：长程任务 benchmark 的经典代表

15. `LIBERO`
   - 关键词：lifelong transfer benchmark
   - 作用：你的主实验 benchmark，必须作为论文实验正当性核心引用

16. `SayCan`
   - 关键词：LLM planning + skill grounding
   - 作用：说明高层规划和底层控制结合的价值

17. `Code as Policies`
   - 关键词：LM program synthesis、embodied control
   - 作用：说明语言模型可以作为策略编排器

18. `VoxPoser`
   - 关键词：language model + spatial value map
   - 作用：体现 structured reasoning / spatial grounding 在操作中的价值

19. `COME-robot`
   - 关键词：closed-loop feedback、failure recovery、GPT-4V
   - 作用：与你的闭环恢复叙事高度相关

20. `Hi Robot`
   - 关键词：hierarchical VLM + VLA
   - 作用：说明 open-ended instruction 和分层执行的重要性

21. `OneTwoVLA`
   - 关键词：adaptive reasoning、unified reasoning-and-acting
   - 作用：代表“让 VLA 自带 reasoning”的一条路线

22. `Long-VLA`
   - 关键词：phase-aware long-horizon VLA
   - 作用：最直接说明“长程能力是当前 VLA 的关键短板”

23. `RoboCerebra`
   - 关键词：long-horizon benchmark、System 2 evaluation
   - 作用：从 benchmark 角度说明长程 reasoning 仍是开放问题

### 与本文的关系

- 这些工作大多在说明“长程任务难”
- 但很多方法要么需要新的训练目标，要么依赖较重的高层规划器
- 你的论文强调的是：在不重训 backbone 的前提下，用外部 agentic 机制补强长程执行闭环

## 3. Agentic 增强、过渡与恢复

这是与你最接近的文献群，必须写细，且要讲出差异。

24. `Sci-VLA`
   - 关键词：transition plugin、state gap
   - 作用：与你的 `Transition Agent` 最相近，是必须重点比较的论文

25. `ManiAgent`
   - 关键词：multi-agent manipulation framework
   - 作用：说明“agentic for manipulation”已经是热点，但其侧重点更广

26. `VLA^2`
   - 关键词：agentic augmentation、unseen concept manipulation
   - 作用：说明 agentic augmentation 可用于补一般化，但它主攻 OOD concept，不是你的主焦点

27. `FailSafe`
   - 关键词：failure reasoning and recovery
   - 作用：与你的 `Critic/Retry` 最直接相关

28. `Guardian`
   - 关键词：robot failure detection、VLM critic
   - 作用：为“failure detection / critic”提供非常新的研究支撑

### 与本文的关系

- 这些工作证明了：
  - transition 有意义
  - critic / failure detector 有意义
  - agentic augmentation 有意义
- 你的区别要写清楚：
  - 你围绕 `fine-tuned LIBERO VLA` 做系统消融
  - 你在官方 `LIBERO` benchmark 上验证
  - 你把 `Transition / Priors / Critic` 放进统一可审计执行闭环里
  - 你强调的是 `long-horizon weak-task repair`

## 4. 推理、记忆与可解释增强

这类工作帮助支撑你的 `A3` 与“过程可解释”叙事。

29. `ECoT`
   - 关键词：embodied chain-of-thought
   - 作用：说明“reason before act”对机器人有帮助

30. `Fast ECoT`
   - 关键词：reasoning latency reduction
   - 作用：说明具身 reasoning 的实时性问题真实存在

31. `MemoryVLA`
   - 关键词：perceptual-cognitive memory
   - 作用：是你 `A3` 写作时最关键的新近参照

### 与本文的关系

- 这些工作说明 memory / reasoning 这条线成立
- 但你当前实现更轻量，不能硬写成大型完整认知记忆系统
- 最稳妥表述是：
  - `topology-aware scene priors`
  - `episodic memory`
  - 而不是过度宣称完整 `Graph RAG system`

## 5. 主线叙事建议

你的文献综述不应写成“列论文清单”，而应形成三段式逻辑：

### 第一段：VLA 已经很强

- 用 `RT-1 / RT-2 / OpenVLA / pi0 / pi0.5` 说明强基座已经成立

### 第二段：长程任务仍然是瓶颈

- 用 `CALVIN / LIBERO / SayCan / Hi Robot / Long-VLA / RoboCerebra` 说明：
  - reactive policy 不够
  - 长程任务需要额外过程控制

### 第三段：agentic 增强是可行方向，但仍有空白

- 用 `Sci-VLA / ManiAgent / FailSafe / Guardian / MemoryVLA / VLA^2` 说明：
  - transition、critic、memory 都有价值
  - 但现有工作往往缺少在官方 benchmark 上围绕强 fine-tuned VLA 的系统消融

然后自然引出你的论文空白点：

- 还缺少一篇工作系统回答：
  - 对于强 `fine-tuned VLA`
  - 在官方 `LIBERO` 上
  - 到底哪些 inference-time agentic 模块真的能提升长程任务
  - 提升体现在哪些弱点 task、哪些恢复行为、哪些统计维度上

## 6. 论文中应避免的错误表述

1. 不要把 `libero_10` 写成你自定义 benchmark
2. 不要把 `A3` 写成已经成熟的大型 `Graph RAG` 系统，除非后续证据补足
3. 不要把 `A4` 写成“加个大 VLM 肯定更强”，而要写成“explicit failure attribution and recovery”
4. 不要把相关工作写得像“别人都不行，只有我们行”
5. 不要忽略 2025--2026 的新工作，否则审稿人会觉得文献调研不及时

## 7. 当前最推荐的 Related Work 小节结构

论文中建议按下面顺序写：

1. `Vision-Language-Action Models`
2. `Long-Horizon Manipulation and Hierarchical Reasoning`
3. `Agentic Augmentation, Failure Recovery, and Critic Mechanisms`
4. `Memory, Retrieval, and Structured Priors for Embodied Control`

## 8. 当前最适合的论文定位句

推荐在引言末尾或 related work 结尾使用类似表述：

`Unlike recent work that primarily scales VLA training, introduces custom long-horizon stacks, or evaluates agentic reasoning on bespoke scenarios, we study a focused and practical question: how to systematically augment a strong fine-tuned VLA policy on the official LIBERO benchmark with modular inference-time process-control mechanisms.`


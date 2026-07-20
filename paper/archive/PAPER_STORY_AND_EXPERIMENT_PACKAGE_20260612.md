# 当前实验与论文故事总览

日期：2026-06-12

## 1. 论文定位

当前论文应定位为：

> 面向机器人部署的 realtime-aware Agentic Policy Harness，而不是新 VLA 基座模型，也不是单纯 benchmark leaderboard。

核心故事：

1. VLA 是强动作生成器，但不是完整机器人部署系统；
2. 长程/接触任务中，VLA 闭环执行会遇到 stall、接触失败、重复无效 replan；
3. 机器人还受到 realtime 推理约束，不能无限次调用大模型；
4. 因此需要一个 Agentic Harness，在物理执行层监控、判断、恢复，并和高效 VLA 推理耦合。

一句话主张：

> Realtime-aware Agentic Policy Harness 能够把 VLA 的 stalled closed-loop execution 转化为可恢复的物理执行，同时通过 low-step pi0.5 inference 和减少重复 full VLA calls 提高部署效率。

## 2. 当前核心实验包

### 2.1 主实验：pi0.5 + Agentic retry

任务：

- Simulator：robosuite / MuJoCo
- Task：Stack
- Robot：Panda
- Backend：pi0.5 PyTorch head-plus 300
- Inference：fixed-noise `1` step
- Horizon：`430`
- Trials：`10`
- Seeds：`20260611` 到 `20260620`

结果：

| Method | Success | Steps mean | Full pi0.5 calls / ep | Recovery | Steady mean infer | Steady P95 infer |
|---|---:|---:|---:|---:|---:|---:|
| raw pi0.5 | `0/10` | `430.0` | `43.0` | `0/0` | `111.99ms` | `122.37ms` |
| pi0.5 + Agentic retry | `10/10` | `254.4` | `6.0` | `10/10` | `110.40ms` | `116.87ms` |

论文解释：

- raw pi0.5 会在长程接触阶段 stall；
- Agentic Harness 通过 physical critic 识别失败段；
- retry/recovery skill 接管 stalled segment；
- 成功率提升的同时，full pi0.5 calls 从 `43.0` 降到 `6.0`；
- 这个结果证明的是“部署 runtime 的恢复能力”，不是“pi0.5 自己学会 recovery”。

### 2.2 推理优化实验：pi0.5 realtime sweep

Fixed-noise 结果：

| Steps | Chunk MSE | Mean infer | First gripper acc |
|---:|---:|---:|---:|
| `1` | `0.1523` | `118.15ms` | `0.75` |
| `2` | `0.1467` | `136.53ms` | `0.75` |
| `4` | `0.1747` | `202.24ms` | `0.65` |
| `6` | `0.1887` | `255.46ms` | `0.65` |
| `8` | `0.2021` | `323.89ms` | `0.60` |
| `10` | `0.2077` | `374.43ms` | `0.60` |

论文解释：

- `1-step` 是当前 realtime-preferred setting；
- `2-step` MSE 最低，但延迟更高；
- 高 step 数没有带来当前样本下的收益；
- 该实验支持 VLA 高效推理部署主线。

### 2.3 补充实验：compact policy + Agentic + CAQ-Lite

任务：

- robosuite Stack
- `100` successful demos
- compact RGB+state action-chunk policy
- mid-nudge perturbation
- 10 trials per method

结果：

| Method | Success | Full calls / ep | Reused / ep | Reuse ratio | Recovery |
|---|---:|---:|---:|---:|---:|
| learned + Light, no retry | `4/10` | `157.0` | `234.4` | `0.599` | `0/0` |
| learned + Agentic retry | `9/10` | `127.5` | `0.0` | `0.000` | `10/10` |
| learned + Agentic retry + LightSafe1 | `9/10` | `69.7` | `62.6` | `0.473` | `10/10` |
| clean Agentic retry + LightSafe1 | `10/10` | `64.3` | `63.8` | `0.498` | `10/10` |

论文角色：

- 这是 runtime 机制验证，不是 pi0.5 主结果；
- 说明 Agentic recovery 与 lightweight action reuse 可以耦合；
- LightSafe1 在保持 `9/10` 成功率时，把 full calls 从 `127.5` 降到 `69.7`，约减少 `45.3%`。

## 3. 图表安排

建议主文图表：

| Item | 内容 | 文件 |
|---|---|---|
| Figure 1 | 系统总览 | `results/paper_assets_20260609/figures/fig1_system_overview_agentic_vla_runtime.png` |
| Figure 2 | Agentic retry flow | `results/paper_assets_20260609/figures/fig2_agentic_recovery_flow.png` |
| Figure 3 | pi0.5 realtime sweep | `results/paper_assets_20260609/figures/pi05_realtime_sweep.png` |
| Table 1 | raw pi0.5 vs Agentic retry | `results/paper_assets_20260609/table_pi05_agentic_retry_matched.md` |
| Table 2 | realtime sweep | `results/paper_assets_20260609/table_pi05_realtime_sweep.md` |
| Table 3 | compact policy supplement | 写入主文或 appendix |
| Qualitative | pi0.5 HD rollout | `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_closed_loop_smoke.mp4` |

## 4. 论文主线写法

### Introduction

逻辑顺序：

1. VLA 已经很强，但直接部署不是完整机器人系统；
2. 长程接触任务需要 monitor、critic、recovery；
3. realtime 是机器人部署硬约束，不是附属工程；
4. 我们提出 realtime-aware Agentic Policy Harness；
5. 贡献是 recovery + efficient inference + coupled runtime。

### Method

三块：

1. VLA backend：pi0.5 low-step fixed-noise inference；
2. Agentic Harness：progress monitor、physical critic、retry/recovery、lockout、trace logger；
3. Coupling：失败后 physical recovery 接管，减少重复 full VLA calls；compact pipeline 中 CAQ-Lite 作为补充。

### Experiments

三问：

1. RQ1：Agentic Harness 是否能修复 pi0.5 stalled execution？
2. RQ2：低步数 pi0.5 推理是否满足 realtime-aware deployment？
3. RQ3：Agentic recovery 与 lightweight reuse 是否可耦合？

### Discussion

强调：

- VLA 可以类比为 physical skill 之一；
- Agentic runtime 负责何时使用 VLA、何时恢复、何时避免重复调用；
- 推理优化是 robotics realtime deployment 的核心约束；
- 当前工作适合算法工程/部署方向，而不是基座模型训练方向。

## 5. Claim Boundary

可以说：

- pi0.5 已接入 robosuite/MuJoCo closed-loop；
- Agentic retry 在 matched Stack 设置中从 `0/10` 到 `10/10`；
- full pi0.5 calls 从 `43.0` 到 `6.0`；
- low-step pi0.5 sweep 给出了明确 latency-quality tradeoff；
- compact policy 证明 recovery + lightweight reuse 的 runtime 机制可行。

不要说：

- 不要说方法已经普适提升所有 VLA benchmark；
- 不要说 pi0.5 自己学会了 recovery；
- 不要说已经完成真实机器人；
- 不要把 compact policy 数字当成 pi0.5 数字；
- 不要把旧 LIBERO 结果当作当前可审计证据。

## 6. 当前不建议继续跑的实验

- 不建议为了恢复旧结果而重跑旧 LIBERO 大杂烩；
- 不建议继续折腾 IsaacSim/IsaacLab；
- 不建议现在转向 Qwen/VLM critic；
- 不建议把量化作为当前主结果，除非能补出明确 latency/memory gain。

## 7. 如果后续必须补实验

优先级：

1. pi0.5 raw vs Agentic retry 扩到 `20` trials；
2. 增加一个相近的 robosuite/ManiSkill task；
3. 做一个更高质量 qualitative video；
4. 最后才考虑重新跑一个小规模 LIBERO clean benchmark。

当前最稳路线：

> 先写论文，不继续扩实验。把现有结果讲清楚，比再跑一堆分散 benchmark 更重要。


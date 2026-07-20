# CARVE-VLA Compact Evidence Package

本目录只跟踪中期汇报和当前论文主线需要的紧凑证据。完整 rollout、模型权重、
环境和中间缓存仍保留在实验机器上。

## Primary Agentic Evidence

- `carve_pi05_recovery_challenge_20260719/`：四策略精确状态对照、PI0.5 调用
  统计、恢复验证、结果图和短视频。
- `carve_pi05_agentic_optimize_pair_20260719/`：Eager、Compiled 和 SMVE 在
  相同恢复状态下的联合实验、结果图、trace 和短视频。

## Optimize Runtime Evidence

- `carve_optimize/PROFILE_COMPARISON.md`：部署 profile 总览。
- `carve_optimize/MASKED_VIEW_ELISION_GATE.md`：SMVE fidelity 与时延门控。
- `carve_optimize/CROSS_ADAPTER_SMVE_GATE.md`：PI0.5 DROID adapter 验证。
- `carve_optimize/VLM_VLA_CONTENTION_GATE.md`：Agent/VLM 共卡压力实验。
- `carve_optimize/openvla_4090_20260718/`：OpenVLA BF16、INT8、NF4 与编译结果。
- `carve_optimize/deployment/`：已晋升和 fallback deployment manifests。

## Realtime And Negative Evidence

- `carve_realtime/*GATE.md`：同步、异步和交替预取对照结论。
- `carve_realtime/heldout_t89_5states_*`：对应的结果、trace 与短视频。
- `carve_realtime/prefix_shadow_t89_5states_20260717/`：前缀一致性诊断。
- `carve_semantic_shadow/paired_t89_3trials_20260717/`：语义干预 shadow gate。
- `carve_profile_branches/`：W8A16/compiled profile 分支与失败状态试验。

统一结果入口为
`../paper/CARVE-VLA/generated/runtime_results_summary.json`，文字结论和声明边界
以 `../docs/status/CARVE_VLA_COMPLETED_WORK.md` 为准。

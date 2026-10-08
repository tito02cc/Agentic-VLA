# VLA Benchmark GitHub Adoption Survey

Date: 2026-08-28

## Method

The table records GitHub REST API fields `stargazers_count`, `forks_count`,
`open_issues_count`, `created_at` and `pushed_at` on 2026-08-28. Counts are a
time-stamped community signal, not a scientific ranking:

- Stars mostly measure visibility and interest.
- Forks are a stronger, but still imperfect, signal of hands-on use.
- Recent repositories should not be rejected only because their absolute count
  is small.
- Repositories that host an entire simulator ecosystem cannot be compared
  directly with a paper-specific benchmark repository.

## Repository Counts

| Rank | Benchmark / repository | Stars | Forks | Open issues | Created | Last push | Scope note |
|---:|---|---:|---:|---:|---|---|---|
| 1 | [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | 2,783 | 468 | 85 | 2024-09 | 2026-08-20 | Same repository covers RoboTwin 1.0 and 2.0 |
| 2 | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 2,248 | 463 | 97 | 2023-04 | 2025-03-15 | Mature original benchmark; no longer actively pushed |
| 3 | [RoboCasa](https://github.com/robocasa/robocasa) | 1,683 | 244 | 56 | 2024-05 | 2026-08-21 | Repository now includes RoboCasa365 ecosystem |
| 4 | [VLABench](https://github.com/OpenMOSS/VLABench) | 468 | 40 | 34 | 2024-12 | 2025-11-11 | Established semantic/long-horizon benchmark |
| 5 | [MolmoSpaces](https://github.com/allenai/molmospaces) | 458 | 67 | 8 | 2026-02 | 2026-08-24 | Full ecosystem; count is not benchmark-only |
| 6 | [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | 439 | 45 | 13 | 2026-07 | 2026-08-20 | Very strong growth for a repository under two months old |
| 7 | [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) | 431 | 41 | 50 | 2025-10 | 2026-01-21 | Broad robustness extension |
| 8 | [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) | 307 | 25 | 23 | 2025-08 | 2026-07-17 | Canonical upstream; do not count the zero-star RLinf mirror |
| 9 | [VLA-Arena](https://github.com/PKU-Alignment/VLA-Arena) | 203 | 16 | 3 | 2025-09 | 2026-07-05 | Broad 170-task evaluation suite |
| 10 | [RoboMME](https://github.com/RoboMME/robomme_benchmark) | 156 | 21 | 1 | 2026-02 | 2026-08-23 | Main benchmark repository; companion policy repo is new |
| 11 | [MIKASA-Robo-VLA](https://github.com/CognitiveAISystems/MIKASA-Robo) | 132 | 15 | 1 | 2025-02 | 2026-06-30 | Memory benchmark and VLA task/data release |
| 12 | [DuoBench](https://github.com/RobotControlStack/duobench) | 16 | 0 | 1 | 2026-06 | 2026-08-28 | New bimanual reproducibility benchmark |
| 13 | [RoboOrchardSim](https://github.com/HorizonRobotics/RoboOrchardSim) | 13 | 0 | 0 | 2026-03 | 2026-08-26 | Parent simulator for InstructMove; not its standalone popularity |
| 14 | [Colosseum V2](https://github.com/jstmn/ColosseumV2) | 12 | 6 | 2 | 2025-02 | 2026-08-05 | Low stars but nontrivial fork ratio |
| 15 | [RoboSemanticBench](https://github.com/ZGC-EmbodyAI/RoboSemanticBench) | 4 | 0 | 1 | 2026-05 | 2026-06-02 | Early semantic diagnostic benchmark |
| 16 | [WatchAct](https://github.com/Baiqi-Li/WatchAct) | 3 | 0 | 0 | 2026-06 | 2026-07-02 | Very early paper-specific repository |
| 17 | [ReflexBench](https://github.com/LxRoboticsLab/ReflexBench) | 1 | 0 | 0 | 2026-08-24 | Only four days old at measurement time |

The [VLA Evaluation Harness](https://github.com/allenai/vla-evaluation-harness)
is infrastructure rather than a benchmark. Its 567 stars, 47 forks, three open
issues and 2026-08-24 last push provide a strong signal that the community wants
a shared model-server/benchmark protocol. CARVE should reuse that protocol where
possible rather than maintaining isolated benchmark forks.

## Interpretation

### Mature and widely used

- RoboTwin 2.0 has the strongest current VLA benchmark ecosystem by both stars
  and forks, active development, many released policy checkpoints and a large
  model leaderboard.
- LIBERO remains widely used, but the upstream repository has not been pushed
  since March 2025 and clean-suite performance is saturated.
- RoboCasa has broad adoption and active development, but RoboCasa365 normally
  carries a large training and data cost.

### Strong current momentum

- RoboDojo is the standout new benchmark: 439 stars and 45 forks in about seven
  weeks. Its adoption signal is already comparable with LIBERO-Plus and VLABench.
- LIBERO-Plus and LIBERO-PRO have enough independent forks to remain credible
  robustness checks, even though they inherit the old LIBERO task family.
- MolmoSpaces has 458 stars and 67 forks in roughly seven months, but its count
  reflects the full data/simulation ecosystem as well as the benchmark.

### Relevant but not yet dominant

- RoboMME has 156 stars and 21 forks, active updates and explicit memory tasks.
  This is a credible community signal for CARVE's memory experiment.
- MIKASA has 132 stars and 15 forks, but its public VLA policy ecosystem is less
  complete than RoboMME.
- VLA-Arena is visible, but its broad 170-task protocol is expensive relative to
  the mechanism-specific evidence CARVE currently needs.

### Frontier or module-only evidence

- WatchAct is conceptually aligned with a VLM Planner, but three stars and zero
  forks indicate that it should be a module-level diagnostic, not the sole main
  benchmark.
- ReflexBench now has an actual official repository, six dynamic tasks and a
  released dataset. Its one-star count is not negative evidence because the repo
  was created four days before this snapshot. It remains a frontier validation,
  not yet a community-standard result.
- InstructMove does not have a standalone benchmark repository; using the 13
  stars of RoboOrchardSim as its popularity would be misleading.

## CARVE Decision After Adoption Check

| Role | Selected benchmark | Adoption reason | CARVE reason |
|---|---|---|---|
| Main cross-policy and efficiency benchmark | RoboTwin 2.0 | 2,783 stars / 468 forks; active | Many current pretrained VLA checkpoints and video-producing physics tasks |
| Main Agentic/memory benchmark | RoboMME | 156 / 21; active and focused | Released PI0.5 baseline plus explicit memory suites |
| High-value Agentic extension | RoboDojo | 439 / 45 in seven weeks | Memory, long-horizon, open and precision dimensions; low leaderboard saturation |
| Low-cost robustness control | LIBERO-PRO or LIBERO-Plus subset | 307 / 25 and 431 / 41 | Existing PI0.5 assets can be reused; no longer the main novelty evidence |
| Planner diagnostic only | WatchAct | 3 / 0 | Cleanly isolates VLM planning but lacks adoption evidence |
| Realtime frontier extension | ReflexBench | Too new to judge | Best external definition of latency-aware dynamic manipulation |

The practical route is therefore not changed to the highest-star benchmark
alone. It combines RoboTwin's ecosystem, RoboMME's mechanism fit and RoboDojo's
recent momentum. This gives CARVE both reproducibility and a current research
story without depending on an untested one-week-old benchmark.

# CARVE PI0.5 Recovery Challenge

- Scenarios: `3`
- Profile audit: `PASS`
- Online lifecycle: `PASS`
- Overall gate: `PASS`

This is a paired intervention and systems experiment in real LIBERO MuJoCo, not a benchmark-wide success-rate estimate.

## Paired Scenarios

| Snapshot | Task | Trigger | Continue | Replan | Prompt retry | Physical recovery |
|---|---:|---|---:|---:|---:|---:|
| `task06_episode001_step0013` | 6 | stall | success | success | success | success |
| `task09_episode001_step0013` | 9 | stall | success | success | success | success |
| `task08_episode003_step0099` | 8 | stale_action | failure | failure | failure | safe stop |

## Aggregate Cost

Costs are averaged only over branches that actually executed actions. A fail-closed safe stop is reported separately and never treated as zero-cost execution.

| Intervention | Success | Executed | Safe stop | VLA calls | Mean steps (executed) | Mean wall (executed) | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen VLA continuation | 2/3 | 3/3 | 0 | 113 | 304.0 | 14.91 s | 0/113 |
| Frequent VLA replan | 2/3 | 3/3 | 0 | 452 | 301.3 | 20.40 s | 0/452 |
| Agentic prompt retry | 2/3 | 3/3 | 0 | 508 | 338.7 | 23.08 s | 0/508 |
| Physical recovery + VLA replan | 2/3 | 2/3 | 1 | 251 | 262.5 | 17.48 s | 0/251 |

Physical-skill verification: `2/2` executed recovery branches.

## Online Controller Sentinel

- Task success: `1/1`
- Physical recovery: `1/1` verified
- Physical actions: `12`
- VLA P95: `56.41 ms`
- Control deadline-miss rate: `10.64%`

The branch study isolates intervention effects under exact state restoration; the online sentinel separately verifies automatic monitor-controller-recovery execution without simulator state entering the controller.

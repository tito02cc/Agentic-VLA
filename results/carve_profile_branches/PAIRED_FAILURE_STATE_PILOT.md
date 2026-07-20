# CARVE Paired Failure-State Pilot

Date: 2026-07-16

## Question

This pilot tests two separate claims under exact LIBERO simulator-state
restoration:

1. whether a short Agentic retry can change a frozen pi0.5 rollout relative to
   a matched frequent-replanning branch; and
2. whether an open-loop fidelity-approved W8A16 profile preserves that
   closed-loop behavior.

The experiment is a small paired pilot, not a benchmark-wide success claim.

## Protocol

- Model: OpenPI `pi05_libero` PyTorch checkpoint.
- Hardware: NVIDIA GeForce RTX 4090.
- Deployment profiles: compiled BF16 and W8A16 on VLM language layers 0--3.
- Controls: two flow steps, action horizon ten, fixed deterministic noise.
- Deadline: 80 ms per VLA call.
- State source: saved MuJoCo states used only for evaluator-side branch restore.
- Stall states: T6 episode 1 step 13 and T9 episode 1 step 13.
- Branch horizon: 280 simulator steps.
- Recovery intervention: two calls with a task-conditioned recovery prompt and
  explicit Agentic retry events, followed by the original task instruction.

`accurate` and `recovery` both commit two actions per inference. Their primary
difference is therefore the two-call Agentic recovery intervention.

## Paired Outcomes

| Profile | State | Continue | Fast | Accurate | Recovery |
|---|---|---:|---:|---:|---:|
| compiled BF16 | T6 stall | success, 213 | success, 210 | failure, 280 | success, 235 |
| compiled BF16 | T9 stall | success, 245 | success, 248 | success, 249 | success, 252 |
| W8A16 | T6 stall | success, 211 | success, 206 | failure, 280 | failure, 280 |
| W8A16 | T9 stall | success, 246 | success, 253 | success, 277 | success, 257 |

Values after success/failure are executed simulator steps. On compiled BF16,
recovery succeeds on the T6 state where the matched accurate branch fails. The
same BF16 `accurate` failure and `recovery` success were reproduced in a second
video-producing run at exactly 280 and 235 steps. Continue and fast also
succeed, so this does not show that immediate recovery is the best decision for
the first transient stall. It supports staged escalation rather than automatic
high-frequency replanning.

W8A16 does not preserve the BF16 T6 recovery outcome. It therefore fails the
current closed-loop non-inferiority gate despite passing all 45 open-loop
fidelity observations. W8A16 remains a lower-memory experimental profile, not
the default CARVE deployment profile.

## Runtime Results

| Profile and study | Calls | Mean | P50 | P95 | P99 | 80 ms misses |
|---|---:|---:|---:|---:|---:|---:|
| BF16, two stall states | 623 | 60.98 ms | 61.53 ms | 65.10 ms | 69.06 ms | 0/623 |
| W8A16, two stall states | 662 | 63.20 ms | 63.72 ms | 68.17 ms | 71.02 ms | 0/662 |

The W8A16 profile still satisfies the realtime gate, but closed-loop behavior,
not latency, blocks its promotion.

## Original-Failure T8 Probe

T8 episode 3 originally failed after 530 steps with only one moka pot placed.
The step-99 state was restored and evaluated for 440 additional steps using
compiled BF16.

| Branch | Success | Steps | VLA calls | Agentic retry calls | 80 ms misses |
|---|---:|---:|---:|---:|---:|
| continue | no | 440 | 55 | 0 | 0/55 |
| accurate | no | 440 | 220 | 0 | 0/220 |
| recovery | no | 440 | 220 | 2 | 0/220 |

A two-prompt retry is insufficient for this real long-horizon failure. The
result is retained as a negative control and the T8 experiment is not expanded.
Future Agentic work must use a stronger stateful or physical recovery skill
rather than relabeling prompt retry as general recovery.

## Gate Decisions

- **Default deployment:** compiled BF16.
- **Experimental low-memory profile:** W8A16, blocked by closed-loop drift.
- **Agentic evidence:** promising paired T6 intervention, but only a pilot.
- **Prompt-only recovery:** insufficient on the original T8 failure.
- **Expansion:** stop under the current recovery mechanism; do not run more
  seeds until the recovery policy is materially improved.

## Evidence

- `compiled_bf16_stall2_h280_fixed2.json`
- `compiled_bf16_stall2_h280_fixed2_policy_calls.jsonl`
- `w8a16_stall2_h280_fixed2.json`
- `w8a16_stall2_h280_fixed2_policy_calls.jsonl`
- `compiled_bf16_t6_ar_h280_fixed2.json`
- `compiled_bf16_t6_ar_h280_fixed2_videos/`
- `w8a16_stall2_h280_fixed2_videos/`
- `compiled_bf16_t8_failure_h440_fixed2.json`
- `compiled_bf16_t8_failure_h440_fixed2_videos/`

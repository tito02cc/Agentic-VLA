# CARVE Semantic-Shadow Perturbation Gate

## Setting

- LIBERO-10 Task 8/9, 3 paired trials per task, fixed policy noise.
- Mid-episode object nudge at control step 80 (`xy=0.03 m`).
- PI0.5 PyTorch with the selected SMVE BF16 2-step/horizon-10 profile.
- Qwen3.5-4B BF16 and PI0.5 co-resident on one RTX 4090.
- Semantic output is shadow-only and never changes robot actions.

## Results

| Variant | Success | Semantic calls | Protocol valid | Semantic P95 (ms) | VLA P95 (ms) | Deadline miss | Blocking reasoning (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| No semantic VLM | 4/6 | 0 | 0/0 | - | 58.60 | 8.02% | 0.00 |
| Async VLM, 32 tokens | 4/6 | 6 | 0/6 | 2545.66 | 64.51 | 7.95% | 0.00 |
| Async VLM, 12-token label | 4/6 | 6 | 6/6 | 1005.29 | 63.48 | 8.02% | 0.00 |

## Gate Decision

The 12-token label protocol is selected for the next CARVE stage:

- Semantic P95 falls by 60.5% versus the 32-token JSON attempt and 88.9% versus the 96-token smoke run.
- All 6 responses are valid and parseable; paired success agreement and episode-length agreement are both 100%.
- VLA model P95 increases by 8.3% during co-resident execution, while the mean deadline-miss delta is +0.000 percentage points.
- The semantic path contributes zero blocking reasoning time. It remains shadow-only until semantic intervention precision is evaluated on labeled failure events.

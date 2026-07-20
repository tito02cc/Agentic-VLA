# robosuite Stack Pipeline Status

Updated: 2026-06-12

This file records the retained robosuite evidence after cleanup. The current project uses robosuite/MuJoCo as the real-simulation validation environment for the Agentic Policy Harness and lightweight runtime.

## 1. Current Role in the Paper

robosuite Stack supports two claims:

1. **pi0.5 + Agentic retry closed-loop result**
   - Raw pi0.5 fixed `1` step: `0/10`.
   - pi0.5 + Agentic retry fixed `1` step: `10/10`.
   - Full pi0.5 calls: `43.0` -> `6.0` per episode.

2. **Compact-policy full pipeline supplement**
   - Data collection -> compact policy training -> closed-loop inference.
   - Agentic retry improves recovery.
   - conservative CAQ-Lite reuse reduces full policy calls.

## 2. Retained pi0.5 Results

Main matched comparison:

- `results/robosuite_stack_pi05_raw_fixed1_h430_5trials_20260611/summary.json`
- `results/robosuite_stack_pi05_raw_fixed1_h430_extra5_seed20260616_20260612/summary.json`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_5trials_20260611/summary.json`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_extra5_seed20260616_20260612/summary.json`

Sampler ablation:

- `results/robosuite_stack_pi05_agentic_retry_fixed2_h430_5trials_20260611/summary.json`

Qualitative asset:

- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_closed_loop_smoke.mp4`
- `results/robosuite_stack_pi05_agentic_retry_fixed1_h430_hd256_20260611/pi05_agentic_retry_hd256_contact.png`

Paper table:

- `results/paper_assets_20260609/table_pi05_agentic_retry_matched.md`

## 3. Retained Realtime Sweep

- `results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_fixed_noise.json`
- `results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_zero_noise.json`
- `results/paper_assets_20260609/table_pi05_realtime_sweep.md`
- `results/paper_assets_20260609/figures/pi05_realtime_sweep.png`

Fixed-noise summary:

| Steps | Chunk MSE | Mean infer | First gripper acc |
|---:|---:|---:|---:|
| `1` | `0.1523` | `118.15ms` | `0.75` |
| `2` | `0.1467` | `136.53ms` | `0.75` |
| `4` | `0.1747` | `202.24ms` | `0.65` |
| `6` | `0.1887` | `255.46ms` | `0.65` |
| `8` | `0.2021` | `323.89ms` | `0.60` |
| `10` | `0.2077` | `374.43ms` | `0.60` |

## 4. Retained Compact-Policy Pipeline

Current retained directory:

- `results/robosuite_stack_e2e_v2_20260611`

Key files:

- `stack_demos_100eps.hdf5`
- `collect_summary.json`
- `eval_midnudge_agentic_retry_light_safe1_10trials_summary.json`
- `eval_clean_agentic_retry_light_safe1_10trials_summary.json`
- `eval_midnudge_agentic_retry_light_safe1_demo.mp4`
- `eval_midnudge_agentic_retry_light_safe1_demo_contact.png`

Current compact-policy table:

| Method | Success | Full calls / ep | Reused / ep | Reuse ratio | Recovery |
|---|---:|---:|---:|---:|---:|
| learned + Light, no retry | `4/10` | `157.0` | `234.4` | `0.599` | `0/0` |
| learned + Agentic retry | `9/10` | `127.5` | `0.0` | `0.000` | `10/10` |
| learned + Agentic retry + LightSafe1 | `9/10` | `69.7` | `62.6` | `0.473` | `10/10` |
| clean Agentic retry + LightSafe1 | `10/10` | `64.3` | `63.8` | `0.498` | `10/10` |

## 5. Boundary

The compact-policy pipeline is not a pi0.5 result. It is retained as a supplemental mechanism study for Agentic retry and CAQ-Lite.

The main pi0.5 claim should use the matched 10-trial pi0.5 table, not the compact-policy success table.


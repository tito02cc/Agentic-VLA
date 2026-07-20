# OpenVLA CARVE Profile Gate

| Profile | P50 (ms) | P95 (ms) | P99 (ms) | Miss@100ms | Peak VRAM (GB) | Exact action | Gripper | MAE | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF16 | 303.48 | 310.92 | 312.74 | 100.0% | 14.42 | - | - | - | PASS |
| INT8 | 1533.96 | 1570.19 | 1580.16 | 100.0% | 7.76 | 0.500 | 1.000 | 0.0089 | FAIL |
| NF4 | 748.35 | 761.33 | 768.11 | 100.0% | 4.41 | 0.100 | 1.000 | 0.0148 | FAIL |
| Compiled BF16 (prewarmed) | 224.25 | 231.84 | 232.18 | 100.0% | 14.42 | 1.000 | 1.000 | 0.0000 | PASS |

Compiled BF16 is accepted only as a prewarmed latency optimization: it preserves all paired actions and improves steady-state latency, but still misses the 100 ms deadline on every call. Its two replay prompt-length buckets require 200.19 seconds of profile preparation.

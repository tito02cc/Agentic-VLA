# CARVE PI0.5 Agentic-Optimize Pair

This experiment restores the same two LIBERO MuJoCo stall states and runs the same bounded physical-recovery branch under three PI0.5 execution profiles.
Eager BF16 is an explicit unpromoted research reference; compiled profiles retain their deployment-admission requirements.

| Profile | Admission | Exact success | Verified recovery | VLA calls | Runtime P50 | Runtime P95 | Miss@80ms | Online lifecycle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Eager BF16 | research reference | 2/2 | 2/2 | 236 | 155.88 ms | 166.26 ms | 236/236 | 1/1, recovery 1/1 |
| Compiled BF16 | promoted | 2/2 | 2/2 | 239 | 62.54 ms | 65.75 ms | 0/239 | 0/1, recovery 2/2 |
| Compiled BF16 + SMVE | promoted | 2/2 | 2/2 | 247 | 52.54 ms | 54.50 ms | 0/247 | 1/1, recovery 1/1 |

## Paired Conclusions

- Exact-state outcome parity: `PASS`.
- Compiled BF16 runtime-P95 reduction versus eager: `60.5%`.
- Compiled BF16 + SMVE runtime-P95 reduction versus eager: `67.2%`.
- SMVE runtime-P95 reduction versus ordinary compiled BF16: `17.1%`.

## Gates

- evidence_complete: `PASS`
- outcome_parity: `PASS`
- compiled_deadline_compliant: `PASS`
- smve_deadline_compliant: `PASS`
- eager_reference_misses_deadline: `PASS`

The experiment is a coupled systems test, not a benchmark-wide success-rate estimate. A failed parity or deadline gate is retained as a negative result rather than tuned away.

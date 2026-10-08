# CARVE Framework Freeze Status

Updated: 2026-08-24

## Scope

This status tracks the framework-first freeze for the local deployment path:

- external tool agent: Codex;
- embedded low-frequency Planner/Critic: local Qwen3.5-4B;
- bounded VLA primitive: PI0.5 LIBERO PyTorch;
- execution optimization: profile-admitted CARVE Optimize Runtime.

No result in this document is a benchmark task-success claim unless explicitly
marked as a simulator episode result.

## Completed

- One secret-free run configuration binds planner, Harness, VLA, Optimize
  profile, benchmark identity and artifact paths.
- Codex and embedded Qwen use the same seven typed tools: `observe`,
  `retrieve_memory`, `vla_act`, `run_skill`, `verify`, `safe_hold`, `finish`.
- The Codex-facing gateway exposes the tool catalog but obtains current
  episode, timestep, safe-boundary and allowlist authority from the runtime;
  stale and direct-action requests are rejected and audited.
- The canonical Session records prompts, typed planner decisions, tool recipes,
  primitive outcomes and profile receipts in one workspace.
- Harness retry and recovery budgets are execution-enforced: semantic retries
  terminate in `SAFE_HOLD`, and physical-skill calls cannot exceed the configured
  per-episode budget in either Codex or embedded-VLM mode.
- `route_execution` forwards admitted inference steps, commit horizon and
  deadline controls from the joint Harness decision to `vla_act`; controls
  outside the promoted profile are rejected before model execution.
- Primitive verification is typed as confirmed, contradicted or inconclusive.
  Only verified contradictions can enter failure memory.
- Run workspaces resume event numbering and artifact indexes, reject manifest
  mismatches, and close through the auditable `finish` tool.
- Evaluator-only `reward`, `success`, object poses and simulator state are
  rejected at the planner/tool boundary.
- The PI0.5 primitive validates checkpoint/profile identity and keeps raw action
  chunks private from the high-level agent.
- The official-LIBERO adapter keeps environment success private and exposes
  only deployable RGB/state observations to the framework.
- The repository test suite passes `207` tests on the current environment.

## Canonical LIBERO-Pro Evidence

The final canonical runner now executes the real local Qwen3.5-4B and promoted
PI0.5 profile in official LIBERO-Pro physics. It samples monitor evidence at
20 Hz, records one video frame per control step, routes all actions and
recoveries through typed tools, and keeps the task-success label in a private
evaluator artifact.

- Object Task 8 succeeds in 423 steps with 43 PI0.5 calls, no unnecessary
  Planner/recovery calls, 55.22/56.59 ms runtime mean/P95 and zero 80 ms misses.
- Object Task 9 triggers two Qwen decisions and one 12-action physical recovery,
  then safely stops at step 362; the private evaluator reports task incomplete.

These runs close the end-to-end framework gate, not a benchmark-wide
success-improvement claim. Full details and artifacts are in
`docs/status/LIBERO_PRO_CANONICAL_GATE_20260824.md`.

## Real Local-Model Evidence

The joint service smoke used the local Qwen3.5-4B endpoint and the promoted
PI0.5 SMVE profile. It passed:

- Qwen decision: accepted `vla_act`, confidence `0.92`;
- Harness: entered safe hold while planning and released after validation;
- PI0.5 output contract: `[10, 7]`;
- profile receipt: `pi05-torch_compile_masked_views-bf16-2step-h10` with
  promoted admission status.

Receipt:
`results/carve_framework_freeze_20260824/qwen_pi05_unified_receipt.json`.

This run generated actions but did not send them to a simulator or robot.

The same real models then passed through the new canonical
`CarveAgentSession`, seven-tool registry and profile-admitted PI0.5 primitive.
The action chunk was again withheld from the environment. The receipt records
an accepted Qwen decision, `[10, 7]` private action shape and exact profile ID:

`results/carve_framework_freeze_20260824/canonical-qwen-pi05-smoke-rep2/model_integration_receipt.json`.

This functional integration pass missed the 80 ms policy-call deadline
(`288.89 ms` runtime). It therefore completes the model/tool wiring evidence,
but does not close the realtime or reconnect gate.

## Open F3 Gate: Reconnect Stability

The same promoted `reduce-overhead` service accepted the first external client,
then a second client triggered a PyTorch CUDA-graph allocator checkpoint error:

`Expected curr_block->next == nullptr ...`

The canonical Harness correctly converted the service error to a failed
primitive and did not expose or execute a partial action. This is fail-closed
evidence, not a passed Optimize deployment gate.

The framework now separates two profile identities:

1. a latency-promoted single-long-connection profile for controlled experiments;
2. a reconnect-safe candidate with CUDA graphs disabled and a compile-friendly
   fixed-step PI0.5 denoising loop.

The new candidate is:

`pi05-torch_compile-bf16-2step-h10-cg-off-fixed-loop`.

Under co-resident local Qwen3.5-4B, its minimum replay gate produced:

- fidelity: `4/4` passed;
- runtime P50/P95: `62.74/64.00 ms`;
- 80 ms deadline misses: `0/12`;
- peak model VRAM: `6.98 GB`.

Manifest:
`results/carve_framework_freeze_20260824/pi05_compile_bf16_cg_off_fixed_loop_candidate.json`.

Two independent websocket clients then both produced `[10, 7]` action chunks.
After deployment prewarm was corrected to match normal no-fixed-noise requests,
their connection-plus-inference times were `86.67 ms` and `75.68 ms`. The
service therefore passes the reconnect stability check, while the first
connection remains outside the 80 ms control-loop claim. Connections must be
established before episode execution.

Receipt:
`results/carve_framework_freeze_20260824/pi05_fixed_loop_reconnect_gate_no_noise_prewarm.json`.

The reconnect-safe profile must be re-measured for warm latency, action
fidelity and closed-loop non-inferiority before it becomes the default.

## Freeze Progress

| Work package | Status | Remaining gate |
|---|---|---|
| F0 configuration/contracts | complete | none |
| F1 Planner/Critic | complete | provider replay is optional evidence |
| F2 canonical toolchain | complete | legacy-runner migration is adapter work, not a core-framework gate |
| F3 PI0.5 + Optimize | complete for current profile | paired method comparison remains an experiment, not a framework gate |
| F4 benchmark/artifacts | complete | canonical success and Agentic-event episodes, videos and private evaluator artifacts recorded |
| F5 conformance freeze | complete | 207 CPU tests plus canonical runtime receipts pass; repository tag is administrative |

The framework is frozen and the predeclared three-state pilots are complete.
They support nominal non-interference and bounded fail-safe behavior, but not a
success-rate gain. No broad benchmark campaign is justified; any next
capability experiment must use a predeclared correctable semantic perturbation.

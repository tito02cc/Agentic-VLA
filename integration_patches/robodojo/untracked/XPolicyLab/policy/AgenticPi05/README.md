# AgenticPi05: Official RoboDojo Sorting Integration

Evaluation-only project adapter. It keeps the official `Pi_05` model, checkpoint,
normalization, RGB convention, joint codec, and benchmark scoring unchanged.
Only `organize_table` and `classify_objects_by_language` are admitted task IDs in
the new driver. Their task effectiveness is **not yet validated**.

## Architecture

`public RGB/proprioception/instruction -> CarveAgentSession -> Planner -> typed
tools -> official pi0.5/real environment -> risk monitor + visual Critic ->
revisable task ledger + episode memory`.

- Model: official `RoboDojo-Benchmark/RoboDojo` dataset checkpoint
  `ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999`, revision
  `0a709fe7863f6c869a8d141676585a883fce4270`.
- Robot: ARX X5, dual-arm absolute joint targets and absolute grippers, 14-D.
- Original action horizon 50, default flow steps 10, three RGB cameras.
- Policy process: project `openpi/.venv/bin/python`, with **vendored Pi_05/OpenPI**
  source before the root OpenPI source. Launcher sets that ordering itself.
- Simulator: `/home/admin1/miniconda3/envs/RoboDojo`.
- VLM: existing OpenAI-compatible/Anthropic provider interfaces; configure a real
  accessible model. No synthetic Planner is selected for real deployment.
- No training pipeline is added; use existing official inference weights.

## Preconditions and Commands

From the project root, run CPU tests:

```bash
openpi/.venv/bin/python -m pytest -q tests/test_robodojo_sorting_framework.py
```

Real-weight, recorded-observation reference/reset check (not simulation):

```bash
openpi/.venv/bin/python scripts/serve_robodojo_sorting_pi05.py \
  --task organize_table --check-only \
  --artifact-dir artifacts/robodojo/sorting_framework/NEW_model_check
```

The launcher validates all 18 checkpoint files. It assumes the checkpoint was
prepared with `scripts/prepare_robodojo_pi05.py`; it never starts downloads silently.
The recorded-observation check currently requires the retained
`native_trace_b0_20260911/capture_1ep/captured_payloads` artifact.

Before running a task, configure `configs/robodojo_sorting_agent.json`:

- Set `planner.model`, `planner.planner_id`, endpoint, and (when needed)
  `api_key_env`. Never put credentials in the JSON.
- Set a fresh `run_id` per episode. Existing artifacts cannot be overwritten.
- Keep `startup_wait` and `event_only` primitive boundaries: the driver owns
  semantic scheduling to avoid duplicate hidden Planner calls.
- Keep `optimize.enabled=false` until a new backend-specific profile is admitted.
- For language classification set task ID `classify_objects_by_language` and
  `max_steps=1100`; for organize_table use `max_steps=1000`.
- `benchmark.seed` is the policy RNG seed in this adapter; the simulator helper's
  seed selects the official layout set. Record both, rather than conflating them.

Start the official policy server in one terminal:

```bash
openpi/.venv/bin/python scripts/serve_robodojo_sorting_pi05.py \
  --task organize_table --port 18081 \
  --artifact-dir artifacts/robodojo/sorting_framework/NEW_server
```

After starting/admitting the configured VLM endpoint, the existing generic
RoboDojo simulator launcher can load this adapter:

```bash
AGENTIC_PI05_RUN_CONFIG="$PWD/configs/robodojo_sorting_agent.json" \
PYTHONPATH="$PWD" EVAL_NUM=1 STARVLA_ROBODOJO_NUM_ENVS=1 \
ROBODOJO_REQUIRE_AUDITED_RESET=0 ROBODOJO_MAX_BASH_RETRIES=1 \
bash third_party/robodojo_official/XPolicyLab/policy/starVLA/scripts/run_hf_robodojo_env_client.sh \
  /home/admin1/miniconda3/envs/RoboDojo 18081 RoboDojo organize_table \
  arx_x5 AgenticPi05 official_pi05_sorting \
  "$PWD/third_party/robodojo_official" 0 0 127.0.0.1
```

The helper lives under `starVLA/scripts` but its policy argument is `AgenticPi05`;
it is a simulator launcher, not the loaded VLA. Use a free port if 18081 is busy.
These are deployment commands, not a statement that the new real-task rollout
has passed. Do not launch before the VLM configuration/capability check.

## Safety and Evidence Scope

- Numeric Monitor uses joint **target error**, not absolute joint magnitude.
  It flags risk, not semantic failure or proven collision safety.
- A risk event discards unexecuted chunk actions. Re-inference observes again.
- `reobserve` really holds current joint targets for one simulation control step.
  It is not a grasp-correction/IK skill. No hidden pose is used to pick an object.
- `task_only` preserves the exact task instruction. Symbolic plans do not silently
  rewrite the VLA prompt. Optional subgoal conditioning still needs actual testing.
- Critic-confirmed ledger entries can be revoked with new contradictory evidence;
  dependent stages return to pending. A model note is an unverified hypothesis.
- A symbolic plan completing does not set official success. Reference execution
  can continue for final positioning until the official evaluator terminates.
- Early safe stop returns to the evaluator with its current score; failures are
  not relabeled unstable or removed. Framework success/score fields remain null.
- New pi05 reset receipts cover observation state, RNG, and generation. The old
  evaluator's strict model-specific audit remains StarVLA-only. Therefore this
  adapter currently uses a **fresh single-episode process**, generic transport
  reset validation, and its own pi05 receipt gate. Multi-episode strict auditing
  remains unfinished. Do not claim that disabling the StarVLA audit validates it.
- Record policy inference wall times and whole-episode wall time separately.
  Paused simulator time, rendered video FPS, and native JAX JIT are not new
  real-time/acceleration evidence. Old Torch/SMVE/quantization numbers do not apply.

## Current Validation

- CPU fixtures exercise contracts, not robot performance.
- Real official weights: `model_gate_rgbfix_20260915/check.json` verifies same-input
  native/runtime equality and RNG reset repeatability. It does not run the new
  tasks or a live VLM.
- Earlier official native `stack_bowls` 1/1 admission remains separate evidence.
- Pending: live VLM + new tasks, meaningful intervention, subgoal/tool capability,
  memory benefit, strict multi-episode reset, and optimized-profile quality.

"""Single-episode integration into the unchanged RoboDojo scoring loop."""

import json
import os

from agentic_vla.benchmarks.robodojo_pi05_episode import (
    SortingEpisode,
    SortingExecutionConfig,
    run_feedback_tool_smoke,
    run_kinematics_smoke,
)
from agentic_vla.benchmarks.robodojo_pi05_policy import validate_model_reset
from agentic_vla.benchmarks.robodojo_pi05_port import XPolicyLabPi05Port
from agentic_vla.configuration import CarveRunConfig


def finalize_episode(environment):
    """Resolve early policy returns through the native terminal reward check."""
    if environment.is_episode_end():
        return False
    # RoboDojo initializes success=True for running environments. Returning
    # early without this final check would incorrectly save them as successes.
    for index in environment.get_running_env_idx_list():
        environment.success[index] = False
    if not environment.is_episode_end():
        raise RuntimeError("native terminal check did not finalize stopped episode")
    return True


def eval_one_episode(TASK_ENV, model_client):
    config = CarveRunConfig.load(os.environ["AGENTIC_PI05_RUN_CONFIG"])
    if config.workspace_path.exists():
        raise FileExistsError("use a new run_id for each episode; never overwrite evidence")
    if TASK_ENV.task_name != config.benchmark.task_id:
        raise ValueError("simulator task and Agent config disagree")
    receipt = model_client.call(func_name="reset", obs={
        "episode_id": config.run_id, "policy_seed": config.benchmark.seed,
    })
    validate_model_reset(receipt, config.benchmark.seed)
    execution = SortingExecutionConfig(**json.loads(os.environ.get("AGENTIC_PI05_EXECUTION_CONFIG", "{}")))
    episode = SortingEpisode(config, XPolicyLabPi05Port(TASK_ENV, model_client), execution=execution)
    episode.session.workspace.append_event("official_pi05_reset", receipt, source="deployment")
    mode = os.environ.get("AGENTIC_PI05_EXECUTION_MODE", "episode")
    if mode not in {"episode", "feedback_tool_smoke", "kinematics_smoke",
                    "motion_probe_smoke", "proposal_review_smoke"}:
        episode.session.close(reason="invalid_execution_mode")
        raise ValueError("unsupported execution mode")
    if mode == "feedback_tool_smoke":
        report = run_feedback_tool_smoke(episode)
    elif mode == "kinematics_smoke":
        report = run_kinematics_smoke(episode)
    elif mode == "motion_probe_smoke":
        from agentic_vla.benchmarks.robodojo_motion_probe import run_motion_probe_smoke
        report = run_motion_probe_smoke(episode)
    elif mode == "proposal_review_smoke":
        from agentic_vla.benchmarks.robodojo_action_proposal import run_proposal_review_smoke
        report = run_proposal_review_smoke(episode)
    else:
        report = episode.run()
    report["native_early_stop_finalized"] = finalize_episode(TASK_ENV)
    (config.workspace_path / "execution_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Early stops retain the native final reward/score, not an unstable skip.


def eval_one_episode_batch(TASK_ENV, model_client):
    if len(TASK_ENV.get_running_env_idx_list()) != 1:
        raise ValueError("AgenticPi05 currently admits exactly one environment per process")
    return eval_one_episode(TASK_ENV, model_client)

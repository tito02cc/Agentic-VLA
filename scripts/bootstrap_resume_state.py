#!/usr/bin/env python3
"""Bootstrap a resumable experiment state from existing rollout videos."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


VIDEO_RE = re.compile(r"task(?P<task>\d+)_trial(?P<trial>\d+)_(?P<status>success|failure)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create _resume_state.json from existing rollout videos.")
    parser.add_argument("--results-dir", required=True, help="Experiment results directory.")
    parser.add_argument("--task-suite", required=True, help="Task suite name, e.g. libero_10.")
    parser.add_argument("--ablation-tag", required=True, help="Ablation tag used for this run.")
    parser.add_argument("--trials", type=int, default=10, help="Trials per task.")
    parser.add_argument("--transition", action="store_true", help="Whether Transition Agent was enabled.")
    parser.add_argument("--graph-rag", action="store_true", help="Whether Graph RAG was enabled.")
    parser.add_argument("--critic", action="store_true", help="Whether Critic was enabled.")
    parser.add_argument("--vision-prompt", action="store_true", help="Whether legacy vision prompt was enabled.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = pathlib.Path(args.results_dir).resolve()
    videos_dir = results_dir / "videos"
    completed_tasks_file = results_dir / "_completed_tasks.json"
    resume_state_file = results_dir / "_resume_state.json"
    partial_summary_file = results_dir / "_partial_summary.json"

    completed_tasks: set[str] = set()
    if completed_tasks_file.exists():
        completed_tasks = set(str(x) for x in json.loads(completed_tasks_file.read_text()))

    per_task_trials: dict[str, list[dict]] = {}
    for video_path in sorted(videos_dir.glob("*.mp4")):
        match = VIDEO_RE.match(video_path.name)
        if not match:
            continue
        task_id = match.group("task")
        per_task_trials.setdefault(task_id, []).append(
            {
                "trial": int(match.group("trial")),
                "success": match.group("status") == "success",
                "path": str(video_path),
            }
        )

    task_metrics: dict[str, dict] = {}
    total_successes = 0
    total_episodes = 0
    current_task = None

    for task_id, trials in sorted(per_task_trials.items(), key=lambda item: int(item[0])):
        trials = sorted(trials, key=lambda item: item["trial"])
        success_count = sum(1 for item in trials if item["success"])
        episode_count = len(trials)
        total_successes += success_count
        total_episodes += episode_count

        if task_id in completed_tasks:
            task_metrics[task_id] = {
                "success_rate": success_count / max(1, args.trials),
                "episodes": float(args.trials),
                "avg_episode_length": 0.0,
                "transition_total": 0,
                "transition_episode_count": 0,
                "transition_success_count": 0,
                "critic_checks": 0,
                "retry_total": 0,
                "retry_episode_count": 0,
                "retry_success_count": 0,
            }
            continue

        if episode_count > 0:
            current_task = {
                "task_id": int(task_id),
                "task_description": "bootstrapped_from_existing_videos",
                "next_episode_idx": max(item["trial"] for item in trials) + 1,
                "task_successes": success_count,
                "task_episode_lengths": [],
                "task_transition_total": 0,
                "task_transition_episodes": 0,
                "task_transition_successes": 0,
                "task_retry_total": 0,
                "task_retry_episodes": 0,
                "task_retry_successes": 0,
                "task_critic_checks": 0,
            }

    overall = total_successes / total_episodes if total_episodes else 0.0
    flags = {
        "transition": args.transition,
        "graph_rag": args.graph_rag,
        "critic": args.critic,
        "vision_prompt": args.vision_prompt,
    }
    payload = {
        "task_suite": args.task_suite,
        "ablation_tag": args.ablation_tag,
        "status": "bootstrapped",
        "note": (
            "Bootstrapped from existing rollout videos. Success counts are exact, "
            "but episode lengths and mechanism statistics prior to bootstrapping are unavailable."
        ),
        "completed_tasks": sorted(completed_tasks),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "episode_lengths": [],
        "success_episode_lengths": [],
        "task_metrics": task_metrics,
        "transition_stats": {"total_transitions": 0, "transitions_leading_to_success": 0} if args.transition else None,
        "critic_stats": {"total_checks": 0, "retries_triggered": 0, "retries_leading_to_success": 0} if args.critic else None,
        "current_task": current_task,
        "flags": flags,
        "overall_success_rate": overall,
    }

    resume_state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial_summary_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {resume_state_file}")
    print(f"Wrote {partial_summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

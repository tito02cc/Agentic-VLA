#!/usr/bin/env python3
"""Audit fixed VideoUnmaskSwap A/B memory or B/C scheduling comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from math import comb
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT_NAME = os.environ.get("ROOT_NAME", "memory_ab_gate_20260923")
EP_START = int(os.environ.get("EP_START", "30"))
EP_END = int(os.environ.get("EP_END", "37"))
PLANNED_EP_END = int(os.environ.get("PLANNED_EP_END", str(EP_END)))
PAIR_MODE = os.environ.get("PAIR_MODE", "AB")
TASK = os.environ.get("TASK", "VideoUnmaskSwap")
if PAIR_MODE not in {"AB", "BC"}:
    raise ValueError("PAIR_MODE must be AB or BC")
if TASK not in {"VideoUnmaskSwap", "VideoUnmask"}:
    raise ValueError("TASK must be VideoUnmaskSwap or VideoUnmask")
ARMS = tuple(PAIR_MODE)
MEMORY_SOURCE_NAME = os.environ.get("MEMORY_SOURCE_NAME", ROOT_NAME)
METHOD_ERROR_EPISODE = int(os.environ.get("METHOD_ERROR_EPISODE", "-1"))
COMPLETION_ROOT_NAME = os.environ.get("COMPLETION_ROOT_NAME")
ROOT = PROJECT / "artifacts/robomme" / ROOT_NAME
MEMORY_ROOT = PROJECT / "artifacts/robomme" / MEMORY_SOURCE_NAME
EPISODES = range(EP_START, EP_END + 1)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing video: {path}")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams or streams[0].get("width", 0) <= 0 or streams[0].get("height", 0) <= 0:
        raise ValueError(f"undecodable video: {path}")


def main() -> int:
    output = ROOT / "analysis.json"
    if output.exists():
        raise FileExistsError(output)
    order = (ROOT / "run_order.log").read_text(encoding="utf-8")
    rows = []
    for episode in EPISODES:
        name = f"{TASK}_ep{episode}"
        preflight = read(MEMORY_ROOT / "preflight" / name / "summary.json")
        memory_path = MEMORY_ROOT / "memory" / f"{name}_memory.json"
        memory = read(memory_path)
        admitted = memory["admission"]["admitted"]
        row = {
            "episode": episode,
            "instruction": preflight["instruction"],
            "target_count": len(memory["memory"].get("required_color_order", [])) if admitted else None,
            "memory_admitted": admitted,
            "memory_rejection": memory["admission"]["reason"],
        }
        if not admitted:
            if any((ROOT / "rollouts" / f"{name}_{arm}").exists() for arm in ARMS):
                raise ValueError(f"rejected memory unexpectedly rolled out: {name}")
            rows.append(row)
            continue
        if episode == METHOD_ERROR_EPISODE:
            if PAIR_MODE != "BC" or not COMPLETION_ROOT_NAME:
                raise ValueError("method-error completion requires BC and COMPLETION_ROOT_NAME")
            c_run = ROOT / "rollouts" / f"{name}_C"
            b_run = PROJECT / "artifacts/robomme" / COMPLETION_ROOT_NAME / "rollouts" / f"{name}_B"
            c_summary = read(c_run / "summary.json")
            b_summary = read(b_run / "summary.json")
            if "unsupported VideoUnmaskSwap action skill" not in (c_summary["failure"] or ""):
                raise ValueError(f"unexpected C failure: {name}")
            if b_summary["failure"] is not None or b_summary["status"] not in {"success", "fail", "ongoing"}:
                raise ValueError(f"unexpected B completion status: {name}")
            if (c_summary["initial_observation_sha256"] != b_summary["initial_observation_sha256"] or
                    c_summary["initial_observation_sha256"] != preflight["initial_observation_sha256"]):
                raise ValueError(f"completion initial state mismatch: {name}")
            for key in ("runner_sha256", "policy_label", "planner_label"):
                if c_summary[key] != b_summary[key]:
                    raise ValueError(f"completion {key} mismatch: {name}")
            for arm, run in (("C", c_run), ("B", b_run)):
                check_video(run / f"{name}_vlm_groundsg.mp4")
                provenance = (c_summary if arm == "C" else b_summary)["memory_provenance"]
                if not provenance or provenance["memory_sha256"] != sha256(memory_path):
                    raise ValueError(f"completion memory mismatch: {name}_{arm}")
            row.update({
                "pair_status": "method_error_cross_service",
                "C_failure": c_summary["failure"],
                "C_steps_before_error": c_summary["executed_steps"],
                "C_video": str((c_run / f"{name}_vlm_groundsg.mp4").relative_to(PROJECT)),
                "B_completion_status": b_summary["status"],
                "B_completion_success": b_summary["success"],
                "B_completion_video": str((b_run / f"{name}_vlm_groundsg.mp4").relative_to(PROJECT)),
                "timing_comparable": False,
            })
            rows.append(row)
            continue
        summaries = {}
        for arm in ARMS:
            run = ROOT / "rollouts" / f"{name}_{arm}"
            summary = read(run / "summary.json")
            video = run / f"{name}_vlm_groundsg.mp4"
            check_video(video)
            if summary["video_frames"] <= 0:
                raise ValueError(f"empty video frames: {name}_{arm}")
            if summary["initial_observation_sha256"] != preflight["initial_observation_sha256"]:
                raise ValueError(f"preflight observation mismatch: {name}_{arm}")
            if summary["failure"] is not None:
                raise ValueError(f"infrastructure failure: {name}_{arm}: {summary['failure']}")
            if summary["status"] not in {"success", "fail", "ongoing"}:
                raise ValueError(f"unknown status: {name}_{arm}")
            if summary["status"] == "ongoing" and summary["executed_steps"] < summary["max_steps"]:
                raise ValueError(f"incomplete rollout: {name}_{arm}")
            config = summary["run_config"]
            fixed = {
                "planner_profile": "carve", "planner_context": "native",
                "grounding_authority": "observe_only", "procedure_authority": "observe_only",
                "execution_feedback": "execution_chunks", "planner_demo_mode": "always",
                "demo_history_mode": "official",
                "planner_repair_attempts": 0, "planner_max_reusable_points": 0,
            }
            for key, expected in fixed.items():
                if config[key] != expected:
                    raise ValueError(f"config mismatch {key}: {name}_{arm}")
            if config["planner_schedule"] != ("selective" if arm == "C" else "every_chunk"):
                raise ValueError(f"schedule mismatch: {name}_{arm}")
            if config["verified_point_reuse"] is not (arm == "C"):
                raise ValueError(f"point reuse mismatch: {name}_{arm}")
            if config["verified_identity_conflict"] or config["stage_receipt_hint"]:
                raise ValueError(f"other interventions enabled: {name}_{arm}")
            if summary["action_horizon"] != 16 or summary["max_steps"] != 1300:
                raise ValueError(f"budget or horizon mismatch: {name}_{arm}")
            provenance = summary["memory_provenance"]
            if arm == "A":
                if config["task_memory_file"] is not None or provenance is not None:
                    raise ValueError(f"A unexpectedly used memory: {name}")
            else:
                if not config["task_memory_file"] or not provenance or not provenance["used"]:
                    raise ValueError(f"{arm} did not use memory: {name}")
                if not provenance["admitted"] or provenance["memory_sha256"] != sha256(memory_path):
                    raise ValueError(f"{arm} memory checksum/admission mismatch: {name}")
                if (provenance["source_demo_sha256"] != preflight["initial_demo_sha256"] or
                        provenance["current_demo_sha256"] != preflight["initial_demo_sha256"]):
                    raise ValueError(f"{arm} demo provenance mismatch: {name}")
            marker = f"starting {name}_{arm} "
            if order.count(marker) != 1:
                raise ValueError(f"missing or duplicate execution order entry: {name}_{arm}")
            summaries[arm] = summary
            row[arm] = {
                "status": summary["status"],
                "success": summary["success"],
                "budget_exhausted": summary["status"] == "ongoing",
                "steps": summary["executed_steps"],
                "planner_calls": summary["planner_calls"],
                "policy_calls": summary["policy_calls"],
                "wall_time_s": summary["wall_time_s"],
                "video": str(video.relative_to(PROJECT)),
            }
        if summaries[ARMS[0]]["initial_observation_sha256"] != summaries[ARMS[1]]["initial_observation_sha256"]:
            raise ValueError(f"pair initial states differ: {name}")
        for key in ("runner_sha256", "policy_label", "planner_label"):
            if summaries[ARMS[0]][key] != summaries[ARMS[1]][key]:
                raise ValueError(f"pair differs in {key}: {name}")
        if PAIR_MODE == "BC" and summaries["B"]["memory_provenance"]["memory_sha256"] != summaries["C"]["memory_provenance"]["memory_sha256"]:
            raise ValueError(f"pair memories differ: {name}")
        first, second = ARMS if episode % 2 == 0 else ARMS[::-1]
        if order.index(f"starting {name}_{first} ") > order.index(f"starting {name}_{second} "):
            raise ValueError(f"wrong arm order: {name}")
        row["pair_status"] = "strict_same_service"
        row["rescue"] = not row[ARMS[0]]["success"] and row[ARMS[1]]["success"]
        row["harm"] = row[ARMS[0]]["success"] and not row[ARMS[1]]["success"]
        rows.append(row)

    paired = [row for row in rows if row.get("pair_status") == "strict_same_service"]
    both_success = [row for row in paired if all(row[arm]["success"] for arm in ARMS)]
    rescues = sum(row["rescue"] for row in paired)
    harms = sum(row["harm"] for row in paired)
    discordant = rescues + harms
    exact_two_sided_p = min(
        1.0,
        2 * sum(comb(discordant, k) for k in range(min(rescues, harms) + 1)) / 2**discordant,
    ) if discordant else 1.0
    out_of_range = list(range(EP_END + 1, PLANNED_EP_END + 1))
    if out_of_range:
        if any(episode < 50 for episode in out_of_range):
            raise ValueError("out-of-range designation includes valid episode")
        for episode in out_of_range:
            name = f"{TASK}_ep{episode}"
            if not (MEMORY_ROOT / "preflight" / name / "summary.json").is_file():
                raise ValueError(f"missing original preflight record: {name}")
            if any((ROOT / "rollouts" / f"{name}_{arm}").exists() for arm in ARMS):
                raise ValueError(f"out-of-range episode has a rollout: {name}")
    def stratum(count: int) -> dict:
        subset = [row for row in paired if row["target_count"] == count]
        return {"pairs": len(subset), **{f"{arm}_success": sum(row[arm]["success"] for row in subset) for arm in ARMS},
                "rescues": sum(row["rescue"] for row in subset),
                "harms": sum(row["harm"] for row in subset)}

    analysis = {
        "protocol": f"carve.robomme.{PAIR_MODE.lower()}_gate.v1",
        "scope": (
            f"fixed {TASK} ep{EP_START}-{EP_END}: A native planner without structured memory; B adds public-demo SAM2 identity memory only"
            if PAIR_MODE == "AB" else
            f"fixed {TASK} ep{EP_START}-{EP_END}: B every-chunk planner; C same memory with selective planning and verified subgoal reuse"
        ),
        "claim_boundary": "Fixed consecutive episodes; cross-task transfer only when compared with separately audited task results, not VLA-kernel optimization.",
        "task": TASK,
        "pair_mode": PAIR_MODE,
        "memory_source": MEMORY_SOURCE_NAME,
        "pre_registered_episodes": list(range(EP_START, PLANNED_EP_END + 1)),
        "valid_episode_range": list(EPISODES),
        "official_episode_count": 50 if out_of_range else None,
        "out_of_range_preflight_episodes": out_of_range,
        "admitted": sum(row["memory_admitted"] for row in rows),
        "strict_pairs": len(paired),
        "method_error_episodes": [row["episode"] for row in rows if row.get("pair_status") == "method_error_cross_service"],
        "rejected": [row["episode"] for row in rows if not row["memory_admitted"]],
        **{f"{arm}_success": sum(row[arm]["success"] for row in paired) for arm in ARMS},
        "rescues": rescues,
        "harms": harms,
        "mcnemar_exact_two_sided_p": exact_two_sided_p,
        "budget_exhausted": {arm: sum(row[arm]["budget_exhausted"] for row in paired) for arm in ARMS},
        "single_target": stratum(1),
        "double_target": stratum(2),
        "both_success_pairs": len(both_success),
        "both_success_cost": {
            key: {arm: sum(row[arm][key] for row in both_success) for arm in ARMS}
            for key in ("wall_time_s", "planner_calls", "policy_calls", "steps")
        },
        "stop_decision": (
            "C fails quality gate; reject selective schedule as current default even if calls are lower."
            if PAIR_MODE == "BC" and (harms > 1 or rescues < harms) else
            "C passes the predeclared within-task quality gate; evaluate cost only on both-success pairs and do not claim VLA-kernel speedup."
            if PAIR_MODE == "BC" else
            "No positive net rescues: stop expanding this memory-only candidate."
            if rescues <= harms else
            "Within-task independent confirmation completed; retain observed harm and limit claim to this task, then test cross-task quality and same-chain cost before a full-system claim."
            if ROOT_NAME.startswith("memory_ab_confirm") else
            "Positive small gate; obtain independent confirmation before any main-effect claim."
        ),
        "rows": rows,
    }
    output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_keys = (
        "admitted", "strict_pairs", "method_error_episodes", "rejected", *(f"{arm}_success" for arm in ARMS), "rescues", "harms",
        "mcnemar_exact_two_sided_p", "budget_exhausted", "single_target",
        "double_target", "both_success_pairs", "both_success_cost", "stop_decision",
    )
    print(json.dumps({key: analysis[key] for key in report_keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

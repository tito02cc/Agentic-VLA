"""Frozen closeout comparisons. Registration never starts a simulator or model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from .robomme_memory import load_verified_identity_memory


PROTOCOL = "robomme.closeout.v1"
TASKS = ("VideoUnmaskSwap", "VideoRepick")
SOURCE_FILES = (
    "scripts/run_robomme_feedback_pilot.py", "scripts/run_robomme_vlm_groundsg.py",
    "scripts/run_robomme_policy_admission.py", "scripts/collect_robomme_initial_demos.py",
    "scripts/build_robomme_unmask_swap_memory.py", "scripts/track_robomme_video_memory.py",
    "scripts/serve_robomme_groundsg_planner.sh", "scripts/serve_robomme_groundsg_policy.sh",
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        pending = path.with_suffix(path.suffix + ".tmp")
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        pending.replace(path)


def study_cases(spec: dict) -> list[dict]:
    if spec.get("protocol") != PROTOCOL or spec.get("stage") not in {"development", "confirmation"}:
        raise ValueError("invalid closeout protocol/stage")
    if set(spec.get("tasks", {})) != set(TASKS):
        raise ValueError("closeout is restricted to the two registered task families")
    diagnostic = spec.get("development_diagnostic")
    if diagnostic is not None:
        if (spec["stage"] != "development" or diagnostic != {
                "task": "VideoUnmaskSwap", "episode": 11, "condition": "A"}
                or spec["tasks"] != {"VideoUnmaskSwap": [11], "VideoRepick": [12]}):
            raise ValueError("only the predeclared VideoUnmaskSwap ep11/A development diagnostic is allowed")
    cases = []
    for task in TASKS:
        episodes = spec["tasks"][task]
        count = 1 if spec["stage"] == "development" else 20
        if (not isinstance(episodes, list) or len(episodes) != count
                or any(type(ep) is not int or not 0 <= ep < 50 for ep in episodes)
                or len(set(episodes)) != count):
            raise ValueError(f"{task} requires {count} distinct official episodes in 0..49")
        names = (["A"] if task == "VideoUnmaskSwap" else []) if diagnostic is not None else (
            ["A", "C"] if task == "VideoRepick" else (
                ["B", "C"] if spec["stage"] == "development" else ["A", "B", "C"]))
        if not names:
            continue
        for index, episode in enumerate(episodes):
            rotation = index % len(names)
            for condition in names[rotation:] + names[:rotation]:
                cases.append({"task": task, "episode": episode, "condition": condition,
                              "memory_enabled": task == "VideoUnmaskSwap" and condition != "A",
                              "schedule": "selective" if condition == "C" else "every_chunk",
                              "name": f"{task}_ep{episode}_{condition}"})
    return cases


def _under_root(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def study_sources(root: Path) -> set[Path]:
    sources = {root / name for name in SOURCE_FILES}
    sources.update((root / "agentic_vla").rglob("*.py"))
    policy = root / "third_party/robomme_policy_learning"
    for directory in ("src", "scripts", "packages/openpi-client/src"):
        sources.update((policy / directory).rglob("*.py"))
    sources.update(policy / name for name in ("pyproject.toml", "uv.lock") if (policy / name).is_file())
    return sources


def case_command(root: Path, output: Path, case: dict, image: str, memory_root: Path) -> list[str]:
    relative = output.relative_to(root)
    prefix = hashlib.sha256(str(output).encode()).hexdigest()[:10]
    command = ["docker", "run", "--rm", "--gpus", "all", "--network", "host",
               "--name", f"carve-closeout-{prefix}-{case['name']}",
               "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video",
               "-e", "SAPIEN_RENDER_DEVICE=cuda",
               "-e", "PYTHONPATH=/workspace:/policy/packages/openpi-client/src:/app/src",
               "-v", f"{root}:/workspace", "-v", f"{root}/third_party/robomme_policy_learning:/policy:ro",
               "-w", "/workspace", image, "/app/.venv/bin/python", "scripts/run_robomme_vlm_groundsg.py",
               "--task", case["task"], "--episode", str(case["episode"]),
               "--max-steps", "1300", "--action-horizon", "16", "--planner-profile", "carve",
               "--execution-feedback", "execution_chunks", "--procedure-authority", "observe_only",
               "--planner-context", "native", "--grounding-authority", "observe_only",
               "--planner-repair-attempts", "0", "--semantic-transition-confirmations", "1",
               "--planner-demo-mode", "always", "--demo-history-mode", "official",
               "--planner-image-format", "png", "--planner-schedule", case["schedule"],
               "--planner-min-reuse-chunks", "1", "--planner-max-reuse-chunks", "2",
               "--planner-visual-change-threshold", "0.09", "--planner-gripper-change-threshold", "0.004",
               "--planner-max-reusable-points", "0", "--memory-on-uncertain", "without_memory",
               "--policy-label", "Yinpei/mme_vla_suite:symbolic-grounded-subgoal/79999",
               "--planner-label", "Qwen3-VL-4B-GroundSG-BF16-SDPA",
               "--host-project-root", str(root),
               "--output", str(Path("/workspace") / relative / case["name"])]
    if case["memory_enabled"]:
        command += ["--task-memory-file", str(Path("/workspace") / memory_root.relative_to(root)
                    / f"{case['task']}_ep{case['episode']}_memory.json")]
    return command


def register_study(root: Path, spec_path: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    output.relative_to(root)
    spec = json.loads(spec_path.read_text())
    cases = study_cases(spec)
    image = spec.get("image_id", "")
    if (not image.startswith("sha256:") or len(image) != 71
            or any(char not in "0123456789abcdef" for char in image[7:])):
        raise ValueError("pin the existing simulator image by its sha256 ID")
    timeout_s = spec.get("timeout_s", 900)
    if type(timeout_s) is not int or not 1 <= timeout_s <= 900:
        raise ValueError("per-case timeout must be in 1..900 seconds")
    memory_root = _under_root(root, spec["memory_root"])
    demo_root = _under_root(root, spec["demo_root"])
    inputs = {spec_path.resolve()}
    errors: list[str] = []
    receipts = {}
    for task in TASKS:
        for episode in spec["tasks"][task]:
            summary_path = demo_root / f"{task}_ep{episode}" / "summary.json"
            demo = summary_path.with_name("initial_demo_front.mp4")
            inputs.update((summary_path, demo))
            if not summary_path.is_file() or not demo.is_file():
                errors.append(f"missing official initial demo: {task}/{episode}")
                continue
            summary = json.loads(summary_path.read_text())
            if (summary.get("task") != task or summary.get("episode") != episode
                    or summary.get("demo_history_mode") != "official"):
                raise ValueError("initial demo must match task, episode and official frame boundary")
            if task == "VideoUnmaskSwap":
                memory_file = memory_root / f"{task}_ep{episode}_memory.json"
                inputs.add(memory_file)
                if not memory_file.is_file():
                    errors.append(f"missing memory or explicit rejection record: {task}/{episode}")
                    continue
                _, receipt = load_verified_identity_memory(memory_file, current_demo=demo,
                    task=task, episode=episode, instruction=summary["instruction"], on_uncertain="without_memory")
                if receipt["compile_time_s"] is None:
                    raise ValueError("memory build cost is required, including rejected memory")
                inputs.add(Path(receipt["source_demo"]))
                receipts[f"{task}/{episode}"] = receipt
    if spec["stage"] == "confirmation":
        audit_path = _under_root(root, spec["split_audit"])
        inputs.add(audit_path)
        audit = json.loads(audit_path.read_text())
        if (audit.get("unknown_history") is not False or not audit.get("audited_roots")
                or audit.get("heldout_episodes") != spec["tasks"]):
            raise ValueError("confirmation requires an explicit matching historical-use audit")
    deployment = spec.get("deployment", {})
    if deployment.get("precision") != "bf16" or deployment.get("attention") != "sdpa" or deployment.get("policy_seed") != 7:
        raise ValueError("closeout uses the fixed BF16/SDPA services and policy seed 7")
    model_inventory = {}
    for role in ("policy_checkpoint", "planner_model", "planner_adapter"):
        folder = Path(deployment[role]).expanduser().resolve()
        files = sorted(path for path in folder.rglob("*") if path.is_file()) if folder.is_dir() else []
        if not files:
            errors.append(f"missing deployment identity directory: {role}")
        model_inventory[str(folder)] = [str(path) for path in files]
    # Do not hash gigabytes of weights for a draft that cannot be executed yet.
    if not errors:
        inputs.update(Path(path) for paths in model_inventory.values() for path in paths)
    history_config = Path(deployment["policy_checkpoint"]).expanduser().resolve().parent / "history_config.txt"
    if not history_config.is_file():
        errors.append("missing policy history_config.txt")
    inputs.add(history_config)
    sources = study_sources(root)
    source_hashes = {str(path.relative_to(root)): digest(path) for path in sorted(sources)}
    fingerprints = {str(path): {"sha256": digest(path), "size": path.stat().st_size,
                               "mtime_ns": path.stat().st_mtime_ns}
                    for path in sorted(inputs) if path.is_file()}
    startup = spec.get("service_startup_s", {})
    for value in startup.values():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("invalid service startup cost")
    manifest = {"protocol": PROTOCOL, "spec": spec, "stage": spec["stage"], "cases": cases,
                "registered_for_execution": not errors, "blocking_inputs": errors,
                "model_inventory": model_inventory, "source_sha256": source_hashes,
                "input_fingerprints": fingerprints, "memory_preflight": receipts,
                "commands": {case["name"]: case_command(root, output, case, image, memory_root) for case in cases},
                "claim_boundary": "Local GroundSG reproduction; registration is not a run or a leaderboard result.",
                "deployment_identity_scope": "File hashes freeze intended weights; live service identity must also be checked before execution.",
                "service_startup_s": startup, "timeout_s": timeout_s}
    output.mkdir(parents=True, exist_ok=False)
    for name in source_hashes:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    write_json(output / "manifest.json", manifest, exclusive=True)
    return manifest


def verify_frozen_inputs(root: Path, manifest: dict) -> None:
    if {str(path.relative_to(root)) for path in study_sources(root)} != set(manifest["source_sha256"]):
        raise RuntimeError("registered source inventory changed")
    for name, expected in manifest["source_sha256"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"registered source changed: {name}")
    for name, expected in manifest["input_fingerprints"].items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != expected["size"] or path.stat().st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError(f"registered input changed: {name}")
        if expected["size"] <= 16 * 1024 * 1024 and digest(path) != expected["sha256"]:
            raise RuntimeError(f"registered input hash changed: {name}")
    for folder, expected in manifest["model_inventory"].items():
        if sorted(str(path) for path in Path(folder).rglob("*") if path.is_file()) != expected:
            raise RuntimeError(f"deployment file inventory changed: {folder}")


def live_service_snapshot(root: Path, deployment: dict, pids: dict, *, proc_root: Path = Path("/proc")) -> dict:
    """Check only declared model processes. Never start, stop or inspect unrelated services."""
    expected = {
        "policy": {"--policy.dir": deployment["policy_checkpoint"], "--policy.config": "mme_vla_suite",
                   "--seed": "7", "--port": "8011"},
        "planner": {"--model": deployment["planner_model"], "--adapters": deployment["planner_adapter"],
                    "--torch-dtype": "bfloat16", "--attn-impl": "sdpa", "--infer-backend": "transformers",
                    "--port": "18070", "--served-model-name": "robomme-groundsg-qwen3vl4b"},
    }
    snapshots = {}
    for role, flags in expected.items():
        pid = pids.get(role)
        if type(pid) is not int or pid <= 0:
            raise ValueError(f"supply the live {role} service PID before executing")
        proc = proc_root / str(pid)
        command = proc.joinpath("cmdline").read_bytes()
        argv = command.decode().strip("\0").split("\0")
        for flag, wanted in flags.items():
            value = argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else next(
                (arg.split("=", 1)[1] for arg in argv if arg.startswith(flag + "=")), None)
            if flag in {"--model", "--adapters", "--policy.dir"} and value is not None:
                value, wanted = str(Path(value).expanduser().resolve()), str(Path(wanted).expanduser().resolve())
            if value != wanted:
                raise RuntimeError(f"live {role} service differs from registered {flag}")
        if role == "planner" and any(arg.startswith(("--quant-", "--bnb-")) for arg in argv):
            raise RuntimeError("closeout Planner must not use an undeclared quantization override")
        environment = dict(item.split("=", 1) for item in proc.joinpath("environ").read_bytes().decode().split("\0") if "=" in item)
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        if visible not in (None, "0"):
            raise RuntimeError("closeout services must use local GPU 0")
        if role == "planner":
            for key, value in {"IMAGE_MAX_TOKEN_NUM": "256", "VIDEO_MAX_TOKEN_NUM": "64", "FPS_MAX_FRAMES": "10"}.items():
                if environment.get(key) != value:
                    raise RuntimeError(f"live Planner preprocessing differs: {key}")
        if role == "policy" and proc.joinpath("cwd").resolve() != root / "third_party/robomme_policy_learning":
            raise RuntimeError("policy service must use the registered MME-VLA repository")
        start_ticks = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        snapshots[role] = {"pid": pid, "start_ticks": start_ticks, "argv_sha256": hashlib.sha256(command).hexdigest(),
                           "gpu": visible, "checked_flags": flags}
    return snapshots


def aggregate_study(manifest: dict, rows: list[dict], failure: str | None) -> dict:
    by_task = {}
    for task in TASKS:
        comparisons = {}
        for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
            pairs = []
            for episode in manifest["spec"]["tasks"][task]:
                found = {r["condition"]: r for r in rows if r["task"] == task and r["episode"] == episode
                         and r["valid_outcome"] and r.get("initial_observation_sha256")}
                if left in found and right in found:
                    if found[left]["initial_observation_sha256"] != found[right]["initial_observation_sha256"]:
                        raise ValueError("paired initial observations differ")
                    pairs.append((found[left], found[right]))
            if pairs:
                comparisons[f"{left}_to_{right}"] = {
                    "paired_episodes": len(pairs),
                    "rescues": sum(not a["success"] and b["success"] for a, b in pairs),
                    "harms": sum(a["success"] and not b["success"] for a, b in pairs),
                    "both_success_costs_s": [[a.get("task_with_memory_build_s"), b.get("task_with_memory_build_s")]
                                             for a, b in pairs if a["success"] and b["success"]]}
        by_task[task] = comparisons
    counts = {}
    for case in manifest["cases"]:
        key = f"{case['task']}/{case['condition']}"
        counts.setdefault(key, {"planned": 0, "valid": 0, "success": 0, "invalid": 0})["planned"] += 1
    for row in rows:
        counts[f"{row['task']}/{row['condition']}"]["valid" if row["valid_outcome"] else "invalid"] += 1
        if row["valid_outcome"]:
            counts[f"{row['task']}/{row['condition']}"]["success"] += int(row["success"])
    completed = {row["name"] for row in rows}
    return {"protocol": PROTOCOL, "complete": failure is None and len(rows) == len(manifest["cases"])
            and all(row["valid_outcome"] for row in rows), "failure": failure,
            "counts": counts, "pairs": by_task, "rows": rows,
            "not_run": [case for case in manifest["cases"] if case["name"] not in completed],
            "service_startup_s": manifest["service_startup_s"], "claim_boundary": manifest["claim_boundary"]}


def execute_study(root: Path, output: Path, manifest: dict) -> None:
    if not manifest["registered_for_execution"]:
        raise ValueError("registration has missing inputs; do not start services or experiments")
    if any((output / name).exists() for name in ("summary.json", "live_services.json")):
        raise FileExistsError("a started batch is immutable; do not silently rerun or append cases")
    if any((output / case["name"]).exists() or (output / f"{case['name']}.log").exists()
           for case in manifest["cases"]):
        raise FileExistsError("case output already exists; do not reuse old evidence")
    verify_frozen_inputs(root, manifest)
    pids = manifest["spec"].get("service_pids", {})
    services = live_service_snapshot(root, manifest["spec"]["deployment"], pids)
    write_json(output / "live_services.json", services, exclusive=True)
    rows, failure = [], None
    try:
        for case in manifest["cases"]:
            verify_frozen_inputs(root, manifest)
            if live_service_snapshot(root, manifest["spec"]["deployment"], pids) != services:
                raise RuntimeError("model service restarted or changed during the paired batch")
            command = manifest["commands"][case["name"]]
            started = time.perf_counter()
            stop_reason = None
            with (output / f"{case['name']}.log").open("x") as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=manifest["timeout_s"], check=False)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    timeout = isinstance(error, subprocess.TimeoutExpired)
                    row = {**case, "success": False, "valid_outcome": timeout,
                           "failure": "case_timeout" if timeout else "user_interruption",
                           "process_wall_time_s": time.perf_counter() - started}
                    rows.append(row)
                    cleanup_ok = False
                    try:
                        cleanup = subprocess.run(["docker", "stop", "-t", "10", command[command.index("--name") + 1]],
                                                 stdout=log, stderr=subprocess.STDOUT, check=False, timeout=30)
                        row["cleanup_returncode"] = cleanup.returncode
                        cleanup_ok = cleanup.returncode == 0
                    except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                        row["cleanup_error"] = str(cleanup_error)
                    if timeout and cleanup_ok:
                        write_json(output / "summary.json", aggregate_study(manifest, rows, None))
                        continue
                    if timeout and not cleanup_ok:
                        raise RuntimeError("container_cleanup_failed_after_case_timeout") from error
                    raise RuntimeError(row["failure"]) from error
            summary_path = output / case["name"] / "summary.json"
            row = {**case, "source": str(summary_path.relative_to(output)),
                   "process_wall_time_s": time.perf_counter() - started}
            if not summary_path.is_file():
                row.update(valid_outcome=False, success=False, failure="runner_failed_without_summary")
                failure_path = summary_path.with_name("failure.json")
                if failure_path.is_file():
                    detail = json.loads(failure_path.read_text())
                    row["failure_detail"] = detail
                    if "out of memory" in detail.get("error", "").lower():
                        row.update(valid_outcome=True, failure="configuration_out_of_memory")
                stop_reason = row["failure"]
            else:
                try:
                    summary = json.loads(summary_path.read_text())
                    if not isinstance(summary, dict):
                        raise ValueError("summary must be an object")
                    if (summary.get("failure") is not None and not isinstance(summary["failure"], str)
                            or not isinstance(summary.get("timing", {}), dict)):
                        raise ValueError("invalid summary failure/timing fields")
                except (ValueError, OSError) as error:
                    row.update(valid_outcome=False, success=False, failure="unreadable_summary", error=str(error))
                    rows.append(row)
                    raise RuntimeError("unreadable_summary") from error
                message = summary.get("failure") or ""
                transport_failure = any(word in message.lower() for word in ("connection refused", "connection reset", "urlopen error [errno"))
                row.update({key: summary.get(key) for key in ("success", "status", "executed_steps",
                    "policy_calls", "planner_request_count", "initial_observation_sha256", "memory_provenance")})
                row.update(valid_outcome=not transport_failure, failure=message or None,
                           timing=summary.get("timing"), task_with_memory_build_s=summary.get("timing", {}).get("task_with_memory_build_s"))
                if transport_failure or "out of memory" in message.lower():
                    stop_reason = message
                if result.returncode and not message:
                    row.update(valid_outcome=False, success=False, failure="unexplained_runner_exit")
                    stop_reason = row["failure"]
                if (summary.get("task") != case["task"] or summary.get("episode") != case["episode"]
                        or type(summary.get("success")) is not bool or not summary.get("initial_observation_sha256")):
                    row.update(valid_outcome=False, success=False, failure="invalid_summary_contract")
                    stop_reason = row["failure"]
                peers = [r for r in rows if r["task"] == case["task"] and r["episode"] == case["episode"]
                         and r.get("initial_observation_sha256")]
                if peers and any(r["initial_observation_sha256"] != row["initial_observation_sha256"] for r in peers):
                    row.update(valid_outcome=False, success=False, failure="paired_initial_state_mismatch")
                    stop_reason = row["failure"]
            rows.append(row)
            write_json(output / "summary.json", aggregate_study(manifest, rows, stop_reason))
            if stop_reason:
                raise RuntimeError(stop_reason)
    except BaseException as error:
        failure = str(error) or type(error).__name__
        raise
    finally:
        write_json(output / "summary.json", aggregate_study(manifest, rows, failure))

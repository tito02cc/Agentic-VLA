"""Test registration and failure accounting without launching Docker or GPU work."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from agentic_vla.benchmarks import robomme_study as study


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture_spec(root):
    for name in study.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# CPU fixture only\n")
    deployment = {"precision": "bf16", "attention": "sdpa", "policy_seed": 7}
    for role in ("policy_checkpoint", "planner_model", "planner_adapter"):
        folder = root / "weights" / role
        folder.mkdir(parents=True)
        (folder / "fixture.weights").write_bytes(b"CPU-only mock weights")
        deployment[role] = str(folder)
    (root / "weights/history_config.txt").write_text("fixture")
    spec = {"protocol": study.PROTOCOL, "stage": "development", "tasks": {
                "VideoUnmaskSwap": [11], "VideoRepick": [12]},
            "image_id": "sha256:" + "a" * 64, "timeout_s": 900,
            "demo_root": "demos", "memory_root": "memory", "deployment": deployment,
            "service_pids": {"policy": 101, "planner": 102}, "service_startup_s": {"policy": None}}
    return spec


def complete_inputs(root, spec):
    for task, episodes in spec["tasks"].items():
        for episode in episodes:
            folder = root / "demos" / f"{task}_ep{episode}"
            write(folder / "summary.json", {"task": task, "episode": episode,
                "demo_history_mode": "official", "instruction": "pick green then blue"})
            source = folder / "initial_demo_front.mp4"
            source.write_bytes(b"same CPU demo fixture")
            if task == "VideoUnmaskSwap":
                write(root / "memory" / f"{task}_ep{episode}_memory.json", {
                    "protocol": "carve.robomme.unmask_swap_memory.v1", "task": task,
                    "episode": episode, "instruction": "pick green then blue",
                    "evaluator_or_oracle_fields_used": False, "source_demo": str(source),
                    "source_demo_sha256": study.digest(source), "compile_time_s": 12.5,
                    "admission": {"admitted": False, "reason": "ambiguous fixture"}, "memory": {}})


def register_ready(root):
    spec = fixture_spec(root)
    complete_inputs(root, spec)
    path = root / "spec.json"
    write(path, spec)
    output = root / "batch"
    return output, study.register_study(root, path, output)


def test_development_and_confirmation_matrices_do_not_duplicate_repick():
    spec = {"protocol": study.PROTOCOL, "stage": "development",
            "tasks": {"VideoUnmaskSwap": [11], "VideoRepick": [12]}}
    cases = study.study_cases(spec)
    assert [(c["task"], c["condition"]) for c in cases] == [
        ("VideoUnmaskSwap", "B"), ("VideoUnmaskSwap", "C"), ("VideoRepick", "A"), ("VideoRepick", "C")]
    spec.update(stage="confirmation", tasks={task: list(range(20)) for task in study.TASKS})
    cases = study.study_cases(spec)
    assert len(cases) == 100
    assert len({(c["task"], c["episode"]) for c in cases}) == 40
    assert [c["condition"] for c in cases[3:6]] == ["B", "C", "A"]
    assert not any(c["memory_enabled"] for c in cases if c["task"] == "VideoRepick")


def test_single_development_diagnostic_is_restricted_and_has_no_memory(tmp_path):
    spec = fixture_spec(tmp_path)
    spec["development_diagnostic"] = {"task": "VideoUnmaskSwap", "episode": 11, "condition": "A"}
    complete_inputs(tmp_path, spec)
    write(tmp_path / "spec.json", spec)
    output = tmp_path / "diagnostic"
    manifest = study.register_study(tmp_path, tmp_path / "spec.json", output)
    assert manifest["registered_for_execution"] and len(manifest["cases"]) == 1
    case = manifest["cases"][0]
    assert (case["task"], case["episode"], case["condition"], case["memory_enabled"]) == (
        "VideoUnmaskSwap", 11, "A", False)
    assert "--task-memory-file" not in manifest["commands"][case["name"]]

    for invalid in ({"task": "VideoRepick", "episode": 12, "condition": "A"},
                    {"task": "VideoUnmaskSwap", "episode": 11, "condition": "C"}):
        spec["development_diagnostic"] = invalid
        with pytest.raises(ValueError, match="predeclared"):
            study.study_cases(spec)
    spec["development_diagnostic"] = {"task": "VideoUnmaskSwap", "episode": 11, "condition": "A"}
    spec["tasks"]["VideoUnmaskSwap"] = [12]
    with pytest.raises(ValueError, match="predeclared"):
        study.study_cases(spec)
    spec["stage"] = "confirmation"
    spec["tasks"] = {task: list(range(20)) for task in study.TASKS}
    with pytest.raises(ValueError, match="predeclared"):
        study.study_cases(spec)


@pytest.mark.parametrize("episodes", [[True], [50], [-1], [11, 11], [], ["11"]])
def test_invalid_episode_registration_is_rejected(episodes):
    with pytest.raises(ValueError):
        study.study_cases({"protocol": study.PROTOCOL, "stage": "development",
                           "tasks": {"VideoUnmaskSwap": episodes, "VideoRepick": [12]}})


def test_registration_does_not_run_processes_and_records_missing_inputs(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    write(tmp_path / "spec.json", spec)
    monkeypatch.setattr(study.subprocess, "run", lambda *a, **k: pytest.fail("registration launched a process"))
    output = tmp_path / "registration"
    manifest = study.register_study(tmp_path, tmp_path / "spec.json", output)
    assert not manifest["registered_for_execution"] and len(manifest["blocking_inputs"]) == 2
    assert len(manifest["cases"]) == 4
    assert not (output / "summary.json").exists()
    assert (output / "sources" / study.SOURCE_FILES[0]).exists()
    with pytest.raises(ValueError, match="missing inputs"):
        study.execute_study(tmp_path, output, manifest)


def test_ready_registration_keeps_rejected_memory_cases_and_freezes_contracts(tmp_path):
    output, manifest = register_ready(tmp_path)
    assert manifest["registered_for_execution"] and len(manifest["cases"]) == 4
    assert manifest["memory_preflight"]["VideoUnmaskSwap/11"]["fallback"] == "without_identity_memory"
    for case in manifest["cases"]:
        command = manifest["commands"][case["name"]]
        for flag, value in {"--planner-image-format": "png", "--demo-history-mode": "official",
            "--planner-demo-mode": "always", "--planner-max-reusable-points": "0",
            "--procedure-authority": "observe_only", "--grounding-authority": "observe_only",
            "--planner-repair-attempts": "0", "--memory-on-uncertain": "without_memory"}.items():
            assert command[command.index(flag) + 1] == value
        assert ("--task-memory-file" in command) == case["memory_enabled"]
    study.verify_frozen_inputs(tmp_path, manifest)
    with pytest.raises(FileExistsError):
        study.register_study(tmp_path, tmp_path / "spec.json", output)


@pytest.mark.parametrize("kind", ["source", "new_source", "input", "model_inventory"])
def test_frozen_input_modifications_stop_execution(tmp_path, kind):
    output, manifest = register_ready(tmp_path)
    if kind == "source":
        (tmp_path / study.SOURCE_FILES[0]).write_text("changed")
    elif kind == "new_source":
        new = tmp_path / "agentic_vla/new.py"
        new.parent.mkdir()
        new.write_text("# changed import inventory")
    elif kind == "input":
        path = tmp_path / "spec.json"
        before = path.stat()
        data = path.read_bytes().replace(b"900", b"800")
        path.write_bytes(data)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        (tmp_path / "weights/planner_model/new.weights").write_bytes(b"new")
    with pytest.raises(RuntimeError, match="changed"):
        study.verify_frozen_inputs(tmp_path, manifest)


def fake_services(monkeypatch):
    monkeypatch.setattr(study, "live_service_snapshot", lambda *a, **k: {"policy": {"pid": 101}})


def fake_summary(case, **changes):
    return {"task": case["task"], "episode": case["episode"], "success": True,
            "status": "success", "failure": None, "initial_observation_sha256": "a" * 64,
            "policy_calls": 3, "planner_request_count": 2,
            "timing": {"task_with_memory_build_s": 25.0}, **changes}


def test_batch_keeps_task_failures_and_finishes_fixed_matrix(tmp_path, monkeypatch):
    output, manifest = register_ready(tmp_path)
    fake_services(monkeypatch)
    pending = iter(manifest["cases"])
    def run(command, **kwargs):
        case = next(pending)
        success = case["condition"] == "C"
        write(output / case["name"] / "summary.json", fake_summary(case, success=success,
              failure=None if success else "unsupported skill", status="success" if success else "failure"))
        return SimpleNamespace(returncode=0 if success else 2)
    monkeypatch.setattr(study.subprocess, "run", run)
    study.execute_study(tmp_path, output, manifest)
    result = json.loads((output / "summary.json").read_text())
    assert result["complete"] and len(result["rows"]) == 4 and not result["not_run"]
    assert result["pairs"]["VideoUnmaskSwap"]["B_to_C"]["rescues"] == 1
    assert result["counts"]["VideoUnmaskSwap/B"] == {"planned": 1, "valid": 1, "success": 0, "invalid": 0}


@pytest.mark.parametrize("mode,valid", [("disconnect", False), ("oom", True), ("bad_json", False),
                                       ("missing", False), ("wrong_episode", False)])
def test_batch_stops_on_service_or_evidence_failure_and_keeps_unrun_cases(tmp_path, monkeypatch, mode, valid):
    output, manifest = register_ready(tmp_path)
    fake_services(monkeypatch)
    case = manifest["cases"][0]
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        path = output / case["name"] / "summary.json"
        if mode == "oom":
            write(path.with_name("failure.json"), {"error": "CUDA out of memory"})
        elif mode == "bad_json":
            path.parent.mkdir()
            path.write_text("broken JSON")
        elif mode != "missing":
            write(path, fake_summary(case, success=False, failure="connection refused" if mode == "disconnect" else None,
                                    episode=10 if mode == "wrong_episode" else case["episode"]))
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(study.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        study.execute_study(tmp_path, output, manifest)
    result = json.loads((output / "summary.json").read_text())
    assert not result["complete"] and len(calls) == 1
    assert len(result["rows"]) == 1 and len(result["not_run"]) == 3
    assert result["rows"][0]["valid_outcome"] is valid


def test_timeout_retained_even_if_only_own_container_cleanup_fails(tmp_path, monkeypatch):
    output, manifest = register_ready(tmp_path)
    fake_services(monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(study.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="container_cleanup_failed_after_case_timeout"):
        study.execute_study(tmp_path, output, manifest)
    result = json.loads((output / "summary.json").read_text())
    assert result["rows"][0]["valid_outcome"] and result["rows"][0]["cleanup_error"]
    assert calls[1][:4] == ["docker", "stop", "-t", "10"]
    assert calls[1][-1] == calls[0][calls[0].index("--name") + 1]


def test_cleaned_up_task_timeout_counts_as_failure_and_batch_continues(tmp_path, monkeypatch):
    output, manifest = register_ready(tmp_path)
    fake_services(monkeypatch)
    cases = iter(manifest["cases"])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "stop"]:
            return SimpleNamespace(returncode=0)
        case = next(cases)
        if len([call for call in calls if call[:2] != ["docker", "stop"]]) == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        write(output / case["name"] / "summary.json", fake_summary(case))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(study.subprocess, "run", run)
    study.execute_study(tmp_path, output, manifest)
    result = json.loads((output / "summary.json").read_text())
    assert result["complete"] and len(result["rows"]) == 4 and not result["not_run"]
    assert result["rows"][0]["failure"] == "case_timeout"
    assert result["rows"][0]["valid_outcome"] and not result["rows"][0]["success"]
    assert result["rows"][0]["cleanup_returncode"] == 0
    assert result["counts"]["VideoUnmaskSwap/B"]["success"] == 0


def test_paired_state_mismatch_is_not_a_valid_rescue(tmp_path, monkeypatch):
    output, manifest = register_ready(tmp_path)
    fake_services(monkeypatch)
    pending = iter(enumerate(manifest["cases"]))
    def run(command, **kwargs):
        index, case = next(pending)
        write(output / case["name"] / "summary.json", fake_summary(case, initial_observation_sha256=str(index) * 64))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="paired_initial_state_mismatch"):
        study.execute_study(tmp_path, output, manifest)
    result = json.loads((output / "summary.json").read_text())
    assert not result["pairs"]["VideoUnmaskSwap"]
    assert len(result["not_run"]) == 2


def test_started_batch_or_stale_case_cannot_be_reused(tmp_path, monkeypatch):
    output, manifest = register_ready(tmp_path)
    monkeypatch.setattr(study, "live_service_snapshot", lambda *a, **k: pytest.fail("should reject before services"))
    (output / manifest["cases"][0]["name"]).mkdir()
    with pytest.raises(FileExistsError, match="case output"):
        study.execute_study(tmp_path, output, manifest)


def test_live_service_snapshot_checks_model_flags_preprocessing_and_process_start(tmp_path):
    spec = fixture_spec(tmp_path)
    proc_root = tmp_path / "proc"
    policy_cwd = tmp_path / "third_party/robomme_policy_learning"
    policy_cwd.mkdir(parents=True)
    flags = {
        "policy": ["python", "--policy.dir", spec["deployment"]["policy_checkpoint"],
                   "--policy.config", "mme_vla_suite", "--seed", "7", "--port", "8011"],
        "planner": ["python", "--model", spec["deployment"]["planner_model"],
                    "--adapters", spec["deployment"]["planner_adapter"], "--torch-dtype", "bfloat16",
                    "--attn-impl", "sdpa", "--infer-backend", "transformers", "--port", "18070",
                    "--served-model-name", "robomme-groundsg-qwen3vl4b"],
    }
    for role, pid in spec["service_pids"].items():
        folder = proc_root / str(pid)
        folder.mkdir(parents=True)
        (folder / "cmdline").write_bytes(("\0".join(flags[role]) + "\0").encode())
        (folder / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=0\0IMAGE_MAX_TOKEN_NUM=256\0VIDEO_MAX_TOKEN_NUM=64\0FPS_MAX_FRAMES=10\0SECRET=not-exported\0")
        (folder / "stat").write_text(f"{pid} (model service) " + " ".join(["0"] * 19 + ["12345"]))
        (folder / "cwd").symlink_to(policy_cwd)
    result = study.live_service_snapshot(tmp_path, spec["deployment"], spec["service_pids"], proc_root=proc_root)
    assert result["policy"]["start_ticks"] == "12345" and "SECRET" not in json.dumps(result)
    (proc_root / "102/environ").write_bytes(b"VIDEO_MAX_TOKEN_NUM=128")
    with pytest.raises(RuntimeError, match="preprocessing differs"):
        study.live_service_snapshot(tmp_path, spec["deployment"], spec["service_pids"], proc_root=proc_root)

"""Exercise the real runner with in-memory services, not physics or learned models."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts import run_robomme_vlm_groundsg as runner


def install_interfaces(monkeypatch, env, client):
    for name in ("openpi_client", "robomme"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy",
                        SimpleNamespace(MMEVLAWebsocketClientPolicy=lambda *a: client))
    monkeypatch.setitem(sys.modules, "robomme.env_record_wrapper", SimpleNamespace(
        BenchmarkEnvBuilder=lambda **k: SimpleNamespace(
            get_episode_num=lambda: 50, make_env_for_episode=lambda ep: env)))


def arguments(monkeypatch, tmp_path, schedule):
    monkeypatch.setattr(sys, "argv", ["runner", "--task", "VideoUnmaskSwap", "--episode", "11",
        "--max-steps", "48", "--output", str(tmp_path / "output"),
        "--policy-label", "mock", "--planner-label", "mock", "--execution-feedback", "execution_chunks",
        "--procedure-authority", "observe_only", "--planner-context", "native",
        "--grounding-authority", "observe_only", "--planner-demo-mode", "always",
        "--planner-repair-attempts", "0", "--planner-schedule", schedule,
        "--planner-max-reusable-points", "0", "--planner-gripper-change-threshold", "0.004",
        "--demo-history-mode", "official", "--planner-image-format", "png",
        "--memory-on-uncertain", "without_memory", "--container-project-root", str(tmp_path),
        "--host-project-root", str(tmp_path)])
    return runner.parse_args()


class Env:
    steps = 0
    closed = False

    def observation(self, initial=False):
        values = [18, 19, 20] if initial else [20 + self.steps]
        return {"front_rgb_list": [np.full((256, 256, 3), v, dtype=np.uint8) for v in values],
                "wrist_rgb_list": [np.full((256, 256, 3), v, dtype=np.uint8) for v in values],
                "joint_state_list": [np.zeros(7) for _ in values],
                "gripper_state_list": [np.array([0.018]) for _ in values]}

    def reset(self):
        return self.observation(True), {"task_goal": ["pick green then blue"], "status": "running"}

    def step(self, action):
        self.steps += 1
        done = self.steps == 48
        return self.observation(), 0, done, False, {"status": "success" if done else "running"}

    def close(self):
        self.closed = True


class Policy:
    def __init__(self, fail=False):
        self.payloads = []
        self.fail = fail

    def reset(self):
        return {"reset_finished": True}

    def add_buffer(self, payload):
        return {"add_buffer_finished": True}

    def infer(self, payload):
        self.payloads.append(payload)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return {"actions": np.zeros((16, 8), dtype=np.float32)}


def mock_video_and_planner(monkeypatch):
    def save(path, frames, **kwargs):
        Path(path).write_bytes(b"same mock video")
    monkeypatch.setattr(runner.imageio, "mimsave", save)
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "put down the container"}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())


@pytest.mark.parametrize("schedule,expected_calls", [("every_chunk", 3), ("selective", 2)])
def test_runner_reuses_only_subgoal_keeps_fresh_observations_and_charges_rejected_memory(
        tmp_path, monkeypatch, capsys, schedule, expected_calls):
    args = arguments(monkeypatch, tmp_path, schedule)
    env, policy = Env(), Policy()
    install_interfaces(monkeypatch, env, policy)
    mock_video_and_planner(monkeypatch)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"same mock video")
    args.task_memory_file = tmp_path / "memory.json"
    args.task_memory_file.write_text(json.dumps({
        "protocol": "carve.robomme.unmask_swap_memory.v1", "task": "VideoUnmaskSwap", "episode": 11,
        "instruction": "pick green then blue", "source_demo": str(source),
        "evaluator_or_oracle_fields_used": False, "memory": {}, "compile_time_s": 12.5,
        "admission": {"admitted": False, "reason": "ambiguous mock identity"}}))
    with ExitStack() as resources:
        assert runner.run_episode(args, resources) == 0
    summary = json.loads((args.output / "summary.json").read_text())
    assert env.closed and env.steps == 48
    assert summary["planner_request_count"] == expected_calls and summary["policy_calls"] == 3
    assert summary["planner_demo_frames"] == 2 and summary["initial_memory_frames"] == 3
    assert summary["task_memory"] is None
    assert summary["memory_provenance"]["fallback"] == "without_identity_memory"
    assert summary["timing"]["memory_build_s"] == 12.5
    assert summary["timing"]["task_with_memory_build_s"] >= 12.5
    assert summary["timing"]["policy_requests"]["count"] == 3
    assert summary["planner_reuse_hits"] == 3 - expected_calls
    assert [int(p["observation/image"][0, 0, 0]) for p in policy.payloads] == [20, 36, 52]
    assert all(not r["semantic_success_verified"] for r in summary["execution_feedback_trace"])
    capsys.readouterr()


def test_policy_failure_is_counted_timed_and_environment_closed(tmp_path, monkeypatch, capsys):
    args = arguments(monkeypatch, tmp_path, "every_chunk")
    env, policy = Env(), Policy(fail=True)
    install_interfaces(monkeypatch, env, policy)
    mock_video_and_planner(monkeypatch)
    with ExitStack() as resources:
        assert runner.run_episode(args, resources) == 2
    summary = json.loads((args.output / "summary.json").read_text())
    assert env.closed and not summary["success"] and summary["policy_calls"] == 0
    assert summary["timing"]["policy_requests"]["count"] == 1
    assert summary["policy_request_audit"][0]["error"] == "CUDA out of memory"
    capsys.readouterr()


def test_preflight_error_closes_environment_without_policy_inference(tmp_path, monkeypatch):
    args = arguments(monkeypatch, tmp_path, "every_chunk")
    args.task_memory_file = tmp_path / "missing.json"
    env, policy = Env(), Policy()
    install_interfaces(monkeypatch, env, policy)
    mock_video_and_planner(monkeypatch)
    with pytest.raises(FileNotFoundError), ExitStack() as resources:
        runner.run_episode(args, resources)
    assert env.closed and not policy.payloads and env.steps == 0


def test_nonempty_output_is_not_reused(tmp_path, monkeypatch):
    args = arguments(monkeypatch, tmp_path, "every_chunk")
    args.output.mkdir()
    (args.output / "partial_video.mp4").write_bytes(b"historical evidence")
    with pytest.raises(FileExistsError), ExitStack() as resources:
        runner.run_episode(args, resources)
    assert list(args.output.iterdir()) == [args.output / "partial_video.mp4"]


def test_out_of_range_episode_is_rejected_before_environment_reset(tmp_path, monkeypatch):
    args = arguments(monkeypatch, tmp_path, "every_chunk")
    args.episode = 50
    env, policy = Env(), Policy()
    install_interfaces(monkeypatch, env, policy)
    with pytest.raises(ValueError, match="outside VideoUnmaskSwap test range"), ExitStack() as resources:
        runner.run_episode(args, resources)
    assert env.steps == 0 and not policy.payloads


def test_verified_grounded_pick_reuses_only_at_chunk_boundary(tmp_path, monkeypatch):
    args = arguments(monkeypatch, tmp_path, "selective")
    args.verified_point_reuse = True
    args.task_memory_file = tmp_path / "memory.json"

    class TargetEnv(Env):
        def observation(self, initial=False):
            result = super().observation(initial)
            for front in result["front_rgb_list"]:
                front[80:105, 110:132] = 220
            return result

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            content = "pick up the container at <|box_start|>(360,469)<|box_end|> that hides the green cube"
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    monkeypatch.setattr(runner.imageio, "mimsave", lambda path, *a, **k: Path(path).write_bytes(b"mock"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    memory = {
        "planner_hint": "green-hidden container at <|box_start|>(360,469)<|box_end|>",
        "required_color_order": ["green", "blue"],
        "hidden_container_points": {"green": [92.0, 120.0]},
    }
    monkeypatch.setattr(runner, "load_verified_identity_memory", lambda *a, **k: (
        memory, {"compile_time_s": 0.0, "shared_model_load_s": 0.0, "admitted": True},
    ))
    env, policy = TargetEnv(), Policy()
    install_interfaces(monkeypatch, env, policy)

    with ExitStack() as resources:
        assert runner.run_episode(args, resources) == 0
    summary = json.loads((args.output / "summary.json").read_text())
    assert env.closed and summary["policy_calls"] == 3
    assert summary["planner_calls"] == 2 and summary["planner_reuse_hits"] == 1
    assert summary["schedule_trace"][1]["reuse_validation"]["reason"] == "current_rgb_revalidated"
    assert summary["schedule_trace"][1]["reuse_validation_latency_ms"] >= 0
    assert summary["timing"]["point_revalidation"]["count"] == 1
    assert all(payload["observation/image"][0, 0, 0] < 100 for payload in policy.payloads)

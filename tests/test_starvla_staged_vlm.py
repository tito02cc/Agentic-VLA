"""CPU contract tests; these do not claim GPU residency or robot efficacy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / (
    "third_party/robodojo_official/XPolicyLab/policy/starVLA/source_starvla"
)
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from deployment.model_server.staged_vlm import CpuStagedVisionPlanner  # noqa: E402


class MovingModel:
    def __init__(self, name, events):
        self.name, self.events = name, events
        self.fail_on = None

    def to(self, device):
        target = str(device)
        self.events.append((self.name, target))
        if self.fail_on == target:
            raise RuntimeError("transfer failed")
        return self


@pytest.fixture
def staged(tmp_path, monkeypatch):
    events = []
    policy = MovingModel("vla", events)
    planner = CpuStagedVisionPlanner(policy, str(tmp_path), "cuda:0")
    real_fork_rng = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kw: real_fork_rng(devices=[]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def load():
        if planner.model is None:
            events.append(("vlm", "load"))
            torch.rand(1)
            planner.model = MovingModel("vlm", events)

    def generate(*args):
        events.append(("vlm", "generate"))
        torch.rand(1)
        return {"text": '{"intent":"continue"}', "input_tokens": 3, "output_tokens": 4}

    monkeypatch.setattr(planner, "_load", load)
    monkeypatch.setattr(planner, "_generate", generate)
    return planner, events


def decide(planner, **kwargs):
    return planner.decide(system_prompt="Return JSON", user_prompt="Inspect progress", **kwargs)


def test_serialized_moves_and_rng_preservation(staged):
    planner, events = staged
    torch.manual_seed(42)
    before = torch.random.get_rng_state().clone()
    result = decide(planner)
    assert torch.equal(before, torch.random.get_rng_state())
    assert events == [
        ("vla", "cpu"), ("vlm", "load"), ("vlm", "cuda:0"),
        ("vlm", "generate"), ("vlm", "cpu"), ("vla", "cuda:0"),
    ]
    assert result["action_model_restored"] is True
    assert result["shared_backbone"] is False
    assert result["backend"] == "cpu_staged_vlm"
    assert result["cold_load"] is True
    assert all(value >= 0 for value in result["timings"].values())
    assert decide(planner)["cold_load"] is False
    assert events.count(("vlm", "load")) == 1


def test_generation_exception_restores_action_model(staged, monkeypatch):
    planner, events = staged

    def fail(*args):
        raise ValueError("generation failed")

    monkeypatch.setattr(planner, "_generate", fail)
    with pytest.raises(ValueError, match="generation failed"):
        decide(planner)
    assert events[-2:] == [("vlm", "cpu"), ("vla", "cuda:0")]
    assert planner.failed is False


def test_load_exception_still_restores_action_model(staged, monkeypatch):
    planner, events = staged

    def fail():
        raise ValueError("load failed")

    monkeypatch.setattr(planner, "_load", fail)
    with pytest.raises(ValueError, match="load failed"):
        decide(planner)
    assert events == [("vla", "cpu"), ("vla", "cuda:0")]


def test_restore_failure_blocks_future_actions(staged):
    planner, _ = staged
    planner.policy.fail_on = "cuda:0"
    with pytest.raises(RuntimeError, match="transfer failed"):
        decide(planner)
    assert planner.failed is True
    with pytest.raises(RuntimeError, match="restart required"), planner.action_boundary():
        raise AssertionError("must not dispatch an action")


@pytest.mark.parametrize("tokens", [0, 1025, True, 1.5])
def test_invalid_budget_does_not_move_models(staged, tokens):
    planner, events = staged
    with pytest.raises((TypeError, ValueError)):
        decide(planner, max_new_tokens=tokens)
    assert events == []


def test_excess_images_rejected_before_transfer(staged):
    planner, events = staged
    with pytest.raises(ValueError, match="three camera"):
        decide(planner, frames={str(i): None for i in range(4)})
    assert events == []


def test_requires_existing_path_and_cuda(tmp_path):
    with pytest.raises(ValueError, match="directory"):
        CpuStagedVisionPlanner(None, str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="CUDA"):
        CpuStagedVisionPlanner(None, str(tmp_path), "cpu")


def test_mirror_residency_dispatch_and_restore_on_error(staged, monkeypatch):
    planner, events = staged

    class Mirror:
        cpu_bytes = 12

        def __init__(self, name):
            self.name = name

        def offload(self):
            events.append((self.name, "mirror_cpu"))

        def activate(self, device):
            events.append((self.name, "mirror_gpu"))

    planner.residency_mode = "cpu_mirror"
    planner._policy_mirror = Mirror("vla")
    planner._vlm_mirror = Mirror("vlm")
    result = decide(planner)
    assert events == [("vla", "mirror_cpu"), ("vlm", "load"),
                      ("vlm", "mirror_gpu"), ("vlm", "generate"),
                      ("vlm", "mirror_cpu"), ("vla", "mirror_gpu")]
    assert result["residency_mode"] == "cpu_mirror"
    assert result["cpu_mirror_bytes"] == 24

    def fail(*args):
        raise ValueError("generation failed")

    monkeypatch.setattr(planner, "_generate", fail)
    with pytest.raises(ValueError, match="generation failed"):
        decide(planner)
    assert events[-2:] == [("vlm", "mirror_cpu"), ("vla", "mirror_gpu")]


def test_invalid_residency_mode(tmp_path):
    with pytest.raises(ValueError, match="residency mode"):
        CpuStagedVisionPlanner(None, str(tmp_path), residency_mode="unregistered")


def test_startup_preparation_is_idempotent_and_task_independent(staged, monkeypatch):
    planner, events = staged
    requests = []
    generate = planner._generate

    def capture(*args):
        requests.append(args)
        return generate(*args)

    monkeypatch.setattr(planner, "_generate", capture)
    torch.manual_seed(42)
    before = torch.random.get_rng_state().clone()
    first = planner.prepare()
    second = planner.prepare()
    assert first == second
    assert first["purpose"] == "startup_only_not_task_evidence"
    assert first["task_observations_used"] is False
    assert first["action_model_restored"] is True
    assert events.count(("vlm", "generate")) == 1
    system, prompt, frames, tokens = requests[0]
    assert system == "You are a helpful assistant."
    assert prompt == "This is a startup check, not a robot task. Reply READY."
    assert list(frames) == ["head"]
    assert frames["head"].shape == (480, 640, 3)
    assert str(frames["head"].dtype) == "uint8"
    assert not frames["head"].any()
    assert tokens == 16
    assert torch.equal(before, torch.random.get_rng_state())
    assert decide(planner)["cold_load"] is False


def test_failed_startup_does_not_cache_success_and_restores_vla(staged, monkeypatch):
    planner, events = staged
    generate = planner._generate

    def fail(*args):
        raise ValueError("startup generation failed")

    monkeypatch.setattr(planner, "_generate", fail)
    with pytest.raises(ValueError, match="startup generation failed"):
        planner.prepare()
    assert planner._startup_receipt is None
    assert events[-2:] == [("vlm", "cpu"), ("vla", "cuda:0")]
    assert planner.failed is False
    monkeypatch.setattr(planner, "_generate", generate)
    assert planner.prepare()["action_model_restored"] is True

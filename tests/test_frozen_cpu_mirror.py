"""CPU residency contracts; real device performance is measured separately."""

import pytest
import torch

from agentic_vla.optimization.residency import FrozenParameterCpuMirror


def module():
    model = torch.nn.Linear(4, 2).eval().requires_grad_(False)
    model.register_buffer("counter", torch.tensor(0))
    return model


def test_values_parameter_identity_and_live_buffers():
    model = module()
    mirror = FrozenParameterCpuMirror(model)
    parameter = model.weight
    x = torch.ones(1, 4)
    expected = model(x).clone()
    for _ in range(3):
        model.counter.add_(1)
        mirror.offload()
        mirror.activate("cpu")
        assert model.weight is parameter
        assert torch.equal(model(x), expected)
    assert model.counter.item() == 3
    assert mirror.cpu_bytes == sum(p.numel() * p.element_size() for p in model.parameters())


def test_tied_parameter_identity_is_preserved():
    model = module()
    model.register_parameter("tied", model.weight)
    mirror = FrozenParameterCpuMirror(model)
    mirror.offload()
    mirror.activate("cpu")
    assert model.weight is model.tied


@pytest.mark.parametrize("mutation", ["inplace", "replace", "grad", "train", "data", "mirror"])
def test_mutation_rejected_before_reuse(mutation):
    model = module()
    mirror = FrozenParameterCpuMirror(model)
    if mutation == "inplace":
        model.weight.add_(1)
    elif mutation == "replace":
        model.weight = torch.nn.Parameter(model.weight.clone(), requires_grad=False)
    elif mutation == "grad":
        model.requires_grad_(True)
    elif mutation == "train":
        model.train()
    elif mutation == "data":
        model.weight.data = model.weight.clone()
    else:
        mirror.copies[0].add_(1)
    with pytest.raises(RuntimeError):
        mirror.offload()


def test_shared_storage_views_rejected():
    model = module()
    model.register_parameter("aliased", torch.nn.Parameter(model.weight.detach(), requires_grad=False))
    with pytest.raises(ValueError, match="sharing storage"):
        FrozenParameterCpuMirror(model)


def test_inference_parameters_rejected():
    with torch.inference_mode():
        model = module()
    with pytest.raises(ValueError, match="version counters"):
        FrozenParameterCpuMirror(model)


def test_dispatched_model_rejected():
    model = module()
    model._hf_hook = object()
    with pytest.raises(ValueError, match="dispatched"):
        FrozenParameterCpuMirror(model)

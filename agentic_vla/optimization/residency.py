"""CPU mirrors for ordinary frozen eager modules, not training/offload hooks."""

from __future__ import annotations

import torch


class FrozenParameterCpuMirror:
    """Retain one CPU copy and discard inactive device parameter storage.

    The caller must serialize inference and transfers. Only frozen, strided,
    ordinary Parameters are supported; compiled/quantized/dispatched models
    need separate adapters. Registered buffers are always transferred afresh,
    because inference may update them. Never mutate weights through ``.data``
    or an external tensor alias: those bypass PyTorch's version counter.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.parameters = tuple(model.named_parameters())
        if not self.parameters:
            raise ValueError("a parameterized module is required")
        self._check_mode()
        storages = set()
        for _, parameter in self.parameters:
            if (type(parameter) is not torch.nn.Parameter or parameter.is_meta
                    or parameter.layout != torch.strided or not parameter.is_contiguous()
                    or parameter.grad is not None):
                raise ValueError("only ordinary contiguous frozen parameters are supported")
            storage = (parameter.device, parameter.untyped_storage().data_ptr())
            if storage in storages:
                raise ValueError("distinct parameters sharing storage are unsupported")
            storages.add(storage)
            if parameter.is_inference():
                raise ValueError("parameters need version counters; load outside inference_mode")
        self.copies = tuple(p.detach().to("cpu", copy=True) for _, p in self.parameters)
        self.cpu_bytes = sum(p.numel() * p.element_size() for p in self.copies)
        self._remember()

    def _check_mode(self):
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("CPU mirror requires eval mode")
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("CPU mirror requires frozen parameters")
        if any(hasattr(module, "_hf_hook") or hasattr(module, "_orig_mod")
               for module in self.model.modules()):
            raise ValueError("dispatched/compiled modules require a dedicated adapter")

    def _remember(self):
        self.signatures = tuple(
            (p._version, p.data_ptr(), p.device, p.shape, p.dtype) for _, p in self.parameters
        )
        self.copy_versions = tuple(p._version for p in self.copies)

    def validate(self):
        self._check_mode()
        current = tuple(self.model.named_parameters())
        if len(current) != len(self.parameters) or any(
            name != original_name or p is not original
            for (name, p), (original_name, original) in zip(current, self.parameters)
        ):
            raise RuntimeError("parameter inventory changed; rebuild the CPU mirror")
        actual = tuple((p._version, p.data_ptr(), p.device, p.shape, p.dtype)
                       for _, p in self.parameters)
        if actual != self.signatures or tuple(p._version for p in self.copies) != self.copy_versions:
            raise RuntimeError("frozen weights changed; rebuild the CPU mirror")

    def _buffers_to(self, device):
        moved = {}
        for module in self.model.modules():
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    key = id(buffer)
                    if key not in moved:
                        moved[key] = buffer.to(device)
                    module._buffers[name] = moved[key]

    @torch.no_grad()
    def offload(self):
        self.validate()
        for device in {p.device for _, p in self.parameters if p.device.type == "cuda"}:
            torch.cuda.synchronize(device)
        for (_, parameter), cpu in zip(self.parameters, self.copies, strict=True):
            parameter.data = cpu
        self._buffers_to("cpu")
        self._remember()

    @torch.no_grad()
    def activate(self, device):
        self.validate()
        for (_, parameter), cpu in zip(self.parameters, self.copies, strict=True):
            parameter.data = cpu.to(device)
        self._buffers_to(device)
        self._remember()

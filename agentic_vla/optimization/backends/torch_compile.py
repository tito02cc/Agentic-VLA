"""PyTorch compile backend for local VLA policy adapters."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from agentic_vla.runtime import PolicyAdapter

from ..contracts import OptimizationProfile
from ..plugins import BackendPlugin, ModelPlugin, PreparedPolicy
from .base import validate_deployment_precision


class TorchCompileBackend(BackendPlugin):
    """Compile the model action-sampling path without changing policy contracts."""

    _MODES = {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    _OPTIONS = {"mode", "fullgraph", "dynamic", "cudagraphs"}

    def __init__(self, compile_fn: Callable[..., Any] | None = None) -> None:
        self._compile_fn = compile_fn

    @property
    def backend_id(self) -> str:
        return "torch_compile"

    def _compile(self, function: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        if self._compile_fn is not None:
            return self._compile_fn(function, **kwargs)
        import torch

        return torch.compile(function, **kwargs)

    def prepare(
        self,
        adapter: PolicyAdapter,
        model: ModelPlugin,
        profile: OptimizationProfile,
    ) -> PreparedPolicy:
        if profile.backend != self.backend_id:
            raise ValueError(
                f"profile backend {profile.backend!r} cannot be prepared by {self.backend_id!r}"
            )
        if profile.async_execution:
            raise ValueError("torch_compile backend does not implement asynchronous execution")
        if profile.module_precisions:
            raise ValueError(
                "torch_compile backend cannot apply module-specific precision transforms"
            )
        unknown_options = sorted(set(profile.options) - self._OPTIONS)
        if unknown_options:
            raise ValueError(f"unknown torch_compile options: {', '.join(unknown_options)}")

        validate_deployment_precision(adapter, profile)
        model.validate_profile(adapter, profile)
        mode = str(profile.options.get("mode", "reduce-overhead"))
        if mode not in self._MODES:
            raise ValueError(f"unsupported torch_compile mode: {mode!r}")
        fullgraph = bool(profile.options.get("fullgraph", False))
        dynamic_value = profile.options.get("dynamic")
        if dynamic_value is not None and not isinstance(dynamic_value, bool):
            raise TypeError("torch_compile dynamic option must be a boolean or null")
        compile_options: dict[str, Any] = {"mode": mode, "fullgraph": fullgraph}
        if dynamic_value is not None:
            compile_options["dynamic"] = dynamic_value

        cudagraphs = profile.options.get("cudagraphs")
        if cudagraphs is not None and not isinstance(cudagraphs, bool):
            raise TypeError("torch_compile cudagraphs option must be a boolean or null")
        if cudagraphs is not None:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = cudagraphs

        policy = getattr(adapter, "policy", None)
        torch_model = getattr(policy, "_model", None)
        sample_actions = getattr(torch_model, "sample_actions", None)
        language_model = getattr(torch_model, "language_model", None)
        if policy is None or torch_model is None:
            raise TypeError("torch_compile requires a local PyTorch policy exposing _model")

        prepared_policy = copy.copy(policy)
        prepared_adapter = copy.copy(adapter)
        if callable(sample_actions):
            compiled_sample_actions = self._compile(sample_actions, **compile_options)
            sample_kwargs = getattr(policy, "_sample_kwargs", {})
            prepared_policy._sample_kwargs = dict(sample_kwargs)
            prepared_policy._sample_actions = compiled_sample_actions
            prepared_adapter._policy = prepared_policy
            prepared_adapter._sample_kwargs = prepared_policy._sample_kwargs
            compile_target = "sample_actions"
        elif model.model_id == "openvla" and language_model is not None:
            compiled_language_model = self._compile(language_model, **compile_options)
            prepared_model = copy.copy(torch_model)
            prepared_model._modules = dict(torch_model._modules)
            prepared_model.language_model = compiled_language_model
            prepared_policy._model = prepared_model
            prepared_adapter._policy = prepared_policy
            compile_target = "language_model"
        else:
            raise TypeError(
                "torch_compile requires _model.sample_actions or an OpenVLA language_model"
            )
        return PreparedPolicy(
            adapter=prepared_adapter,
            model_id=model.model_id,
            backend_id=self.backend_id,
            profile=profile,
            metadata={
                "transformed": True,
                "transform": "torch.compile",
                "compile_target": compile_target,
                "compile_options": compile_options,
                "cudagraphs": cudagraphs,
                "source_adapter_unchanged": True,
            },
        )

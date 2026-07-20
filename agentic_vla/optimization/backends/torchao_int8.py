"""TorchAO W8A16 backend with component-scoped pi0.5 quantization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentic_vla.runtime import PolicyAdapter

from ..contracts import OptimizationProfile
from ..plugins import BackendPlugin, ModelPlugin, PreparedPolicy
from .base import validate_deployment_precision


class TorchAOInt8Backend(BackendPlugin):
    """Apply INT8 weight-only PTQ and compile the action-sampling path."""

    _MODES = {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    _OPTIONS = {"compile_mode", "fullgraph", "dynamic"}

    def __init__(
        self,
        *,
        quantize_fn: Callable[..., Any] | None = None,
        config_factory: Callable[[], Any] | None = None,
        compile_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._quantize_fn = quantize_fn
        self._config_factory = config_factory
        self._compile_fn = compile_fn

    @property
    def backend_id(self) -> str:
        return "torchao_int8"

    def _torchao(self) -> tuple[Callable[..., Any], Any]:
        if self._quantize_fn is not None and self._config_factory is not None:
            return self._quantize_fn, self._config_factory()
        try:
            from torchao.quantization import Int8WeightOnlyConfig, quantize_
        except ImportError as exc:
            raise RuntimeError("torchao_int8 backend requires torchao") from exc
        return quantize_, Int8WeightOnlyConfig(version=2)

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
            raise ValueError("torchao_int8 backend does not implement asynchronous execution")
        unknown_options = sorted(set(profile.options) - self._OPTIONS)
        if unknown_options:
            raise ValueError(f"unknown torchao_int8 options: {', '.join(unknown_options)}")
        if not profile.module_precisions:
            raise ValueError("torchao_int8 requires at least one component precision request")
        unsupported = {
            group: precision
            for group, precision in profile.module_precisions.items()
            if precision != "int8"
        }
        if unsupported:
            raise ValueError(f"torchao_int8 only applies int8 weights: {unsupported}")

        validate_deployment_precision(adapter, profile)
        model.validate_profile(adapter, profile)
        policy = getattr(adapter, "policy", None)
        torch_model = getattr(policy, "_model", None)
        sample_actions = getattr(torch_model, "sample_actions", None)
        if policy is None or not callable(sample_actions) or not hasattr(torch_model, "named_modules"):
            raise TypeError(
                "torchao_int8 requires a local PyTorch policy exposing _model.sample_actions"
            )

        component_groups = model.component_groups(adapter)
        prefixes = tuple(
            prefix
            for group in profile.module_precisions
            for prefix in component_groups[group]
        )

        import torch

        matched_names = tuple(
            name
            for name, module in torch_model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        )
        if not matched_names:
            raise ValueError(
                "torchao_int8 component selection did not match any torch.nn.Linear modules"
            )
        matched = frozenset(matched_names)

        def filter_fn(module: Any, fqn: str) -> bool:
            return isinstance(module, torch.nn.Linear) and fqn in matched

        quantize_fn, quantization_config = self._torchao()
        quantize_fn(torch_model, quantization_config, filter_fn=filter_fn)

        mode = str(profile.options.get("compile_mode", "reduce-overhead"))
        if mode not in self._MODES:
            raise ValueError(f"unsupported torch_compile mode: {mode!r}")
        fullgraph = bool(profile.options.get("fullgraph", False))
        dynamic_value = profile.options.get("dynamic")
        if dynamic_value is not None and not isinstance(dynamic_value, bool):
            raise TypeError("torchao_int8 dynamic option must be a boolean or null")
        compile_options: dict[str, Any] = {"mode": mode, "fullgraph": fullgraph}
        if dynamic_value is not None:
            compile_options["dynamic"] = dynamic_value
        policy._sample_actions = self._compile(torch_model.sample_actions, **compile_options)

        return PreparedPolicy(
            adapter=adapter,
            model_id=model.model_id,
            backend_id=self.backend_id,
            profile=profile,
            metadata={
                "transformed": True,
                "transform": "torchao.int8_weight_only+torch.compile",
                "quantized_components": tuple(profile.module_precisions),
                "quantized_linear_count": len(matched_names),
                "compile_options": compile_options,
                "source_adapter_unchanged": False,
            },
        )

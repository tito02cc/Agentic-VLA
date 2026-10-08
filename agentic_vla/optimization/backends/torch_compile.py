"""PyTorch compile backend for local VLA policy adapters."""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from typing import Any

from agentic_vla.runtime import PolicyAdapter

from ..contracts import OptimizationProfile
from ..plugins import BackendPlugin, ModelPlugin, PreparedPolicy
from .base import validate_deployment_precision


def build_pi05_fixed_step_sample_actions(torch_model: Any) -> Callable[..., Any]:
    """Build a compile-friendly equivalent of PI0.5's denoising sampler."""

    original = getattr(torch_model, "sample_actions", None)
    if not callable(original):
        raise TypeError("fixed-step PI0.5 compilation requires sample_actions")
    unwrapped = inspect.unwrap(original)
    function_globals = getattr(unwrapped, "__globals__", {})
    torch = function_globals.get("torch")
    make_att_2d_masks = function_globals.get("make_att_2d_masks")
    if torch is None or not callable(make_att_2d_masks):
        raise TypeError("unsupported PI0.5 sampling implementation")

    @functools.wraps(unwrapped)
    def sample_actions(device: Any, observation: Any, noise: Any = None, num_steps: int = 10) -> Any:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (
                bsize,
                torch_model.config.action_horizon,
                torch_model.config.action_dim,
            )
            noise = torch_model.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            torch_model._preprocess_observation(observation, train=False)
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = torch_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = torch_model._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )
        torch_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"
        )
        _, past_key_values = torch_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        for _ in range(num_steps):
            v_t = torch_model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                time.expand(bsize),
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    return torch.no_grad()(sample_actions)


class TorchCompileBackend(BackendPlugin):
    """Compile the model action-sampling path without changing policy contracts."""

    _MODES = {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    _OPTIONS = {
        "mode",
        "fullgraph",
        "dynamic",
        "cudagraphs",
        "fixed_step_loop",
    }

    def __init__(
        self,
        compile_fn: Callable[..., Any] | None = None,
        pi05_sampler_factory: Callable[[Any], Callable[..., Any]] | None = None,
    ) -> None:
        self._compile_fn = compile_fn
        self._pi05_sampler_factory = (
            pi05_sampler_factory or build_pi05_fixed_step_sample_actions
        )

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
        cudagraph_trees: bool | None = None
        if cudagraphs is not None and not isinstance(cudagraphs, bool):
            raise TypeError("torch_compile cudagraphs option must be a boolean or null")
        if cudagraphs is not None:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = cudagraphs
            # PyTorch 2.12 can still route dynamic server requests through the
            # CUDA-graph tree manager when only ``cudagraphs`` is disabled.
            # Keep the compatibility switch scoped to explicit opt-out profiles.
            if not cudagraphs and hasattr(inductor_config.triton, "cudagraph_trees"):
                inductor_config.triton.cudagraph_trees = False
                cudagraph_trees = False

        policy = getattr(adapter, "policy", None)
        torch_model = getattr(policy, "_model", None)
        sample_actions = getattr(torch_model, "sample_actions", None)
        language_model = getattr(torch_model, "language_model", None)
        if policy is None or torch_model is None:
            raise TypeError("torch_compile requires a local PyTorch policy exposing _model")

        prepared_policy = copy.copy(policy)
        prepared_adapter = copy.copy(adapter)
        if callable(sample_actions):
            fixed_step_loop = bool(profile.options.get("fixed_step_loop", False))
            compile_target_fn = (
                self._pi05_sampler_factory(torch_model)
                if fixed_step_loop and model.model_id == "pi05"
                else sample_actions
            )
            compiled_sample_actions = self._compile(
                compile_target_fn, **compile_options
            )
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
                "cudagraph_trees": cudagraph_trees,
                "fixed_step_loop": bool(
                    profile.options.get("fixed_step_loop", False)
                ),
                "source_adapter_unchanged": True,
            },
        )

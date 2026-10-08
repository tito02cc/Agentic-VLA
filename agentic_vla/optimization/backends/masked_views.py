"""Exact padded-camera elision for local PyTorch pi0.5 policies."""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agentic_vla.runtime import PolicyAdapter

from ..contracts import OptimizationProfile, StaticMaskedViewContract
from ..plugins import BackendPlugin, ModelPlugin, PreparedPolicy
from .base import validate_deployment_precision


def _normalize_indices(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("elided_image_indices must be a sequence of integers")
    indices: list[int] = []
    for raw_index in value:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError("elided_image_indices must contain only integers")
        if raw_index < 0:
            raise ValueError("elided_image_indices must be non-negative")
        indices.append(raw_index)
    if not indices:
        raise ValueError("elided_image_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("elided_image_indices must not contain duplicates")
    return tuple(sorted(indices))


def _resolve_elision_plan(
    options: Mapping[str, Any],
) -> tuple[tuple[int, ...], StaticMaskedViewContract | None]:
    uses_indices = "elided_image_indices" in options
    uses_names = "elided_image_views" in options or "image_view_order" in options
    if uses_indices and uses_names:
        raise ValueError(
            "use either named masked views or legacy elided_image_indices, not both"
        )
    if uses_names:
        if "image_view_order" not in options or "elided_image_views" not in options:
            raise ValueError(
                "named masked-view elision requires image_view_order and elided_image_views"
            )
        contract = StaticMaskedViewContract(
            view_order=tuple(options["image_view_order"]),
            masked_views=tuple(options["elided_image_views"]),
        )
        return contract.masked_indices, contract
    return _normalize_indices(options.get("elided_image_indices", ())), None


def elide_masked_views(
    images: Sequence[Any],
    image_masks: Sequence[Any],
    elided_indices: Sequence[int],
    *,
    torch_module: Any,
) -> tuple[list[Any], list[Any]]:
    """Remove views only when every batch item marks them as padding."""

    if len(images) != len(image_masks):
        raise ValueError("image and image-mask counts differ")
    index_set = frozenset(elided_indices)
    if index_set and max(index_set) >= len(images):
        raise IndexError(
            f"elided image index {max(index_set)} is outside {len(images)} input views"
        )
    for index in index_set:
        mask = image_masks[index]
        torch_module._assert(
            torch_module.logical_not(mask).all(),
            f"image view {index} was configured for elision but is active",
        )
    keep = [index for index in range(len(images)) if index not in index_set]
    if not keep:
        raise ValueError("masked-view elision cannot remove every image view")
    return [images[index] for index in keep], [image_masks[index] for index in keep]


def build_masked_view_sample_actions(
    torch_model: Any,
    elided_indices: Sequence[int],
) -> Callable[..., Any]:
    """Build a pi0.5 sampler that omits statically padded views before SigLIP."""

    original_sample_actions = getattr(torch_model, "sample_actions", None)
    if not callable(original_sample_actions):
        raise TypeError("masked-view elision requires a callable sample_actions")
    unwrapped = inspect.unwrap(original_sample_actions)
    function_globals = getattr(unwrapped, "__globals__", {})
    torch = function_globals.get("torch")
    make_att_2d_masks = function_globals.get("make_att_2d_masks")
    if torch is None or not callable(make_att_2d_masks):
        raise TypeError(
            "masked-view elision requires the OpenPI PyTorch pi0.5 sampling implementation"
        )

    required = (
        "_preprocess_observation",
        "embed_prefix",
        "_prepare_attention_masks_4d",
        "denoise_step",
        "sample_noise",
        "paligemma_with_expert",
        "config",
    )
    missing = [name for name in required if not hasattr(torch_model, name)]
    if missing:
        raise TypeError(f"pi0.5 model is missing sampler attributes: {', '.join(missing)}")
    indices = tuple(elided_indices)

    @functools.wraps(unwrapped)
    def sample_actions(device: Any, observation: Any, noise: Any = None, num_steps: int = 10) -> Any:
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (
                bsize,
                torch_model.config.action_horizon,
                torch_model.config.action_dim,
            )
            noise = torch_model.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = torch_model._preprocess_observation(
            observation, train=False
        )
        images, img_masks = elide_masked_views(
            images,
            img_masks,
            indices,
            torch_module=torch,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = torch_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
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
        while time >= -dt / 2:
            v_t = torch_model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                time.expand(bsize),
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    return torch.no_grad()(sample_actions)


class MaskedViewElisionBackend(BackendPlugin):
    """Elide guaranteed padding views and compile the shortened pi0.5 sampler."""

    _MODES = {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    _OPTIONS = {
        "image_view_order",
        "elided_image_views",
        "elided_image_indices",
        "mode",
        "fullgraph",
        "dynamic",
        "cudagraphs",
    }

    def __init__(
        self,
        *,
        compile_fn: Callable[..., Any] | None = None,
        sampler_factory: Callable[[Any, Sequence[int]], Callable[..., Any]] | None = None,
    ) -> None:
        self._compile_fn = compile_fn
        self._sampler_factory = sampler_factory or build_masked_view_sample_actions

    @property
    def backend_id(self) -> str:
        return "torch_compile_masked_views"

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
        if model.model_id != "pi05":
            raise ValueError("masked-view elision currently supports only pi0.5")
        if profile.async_execution:
            raise ValueError("masked-view backend does not implement asynchronous execution")
        if profile.module_precisions:
            raise ValueError("masked-view backend cannot apply module-specific precision transforms")
        unknown_options = sorted(set(profile.options) - self._OPTIONS)
        if unknown_options:
            raise ValueError(f"unknown masked-view options: {', '.join(unknown_options)}")
        indices, view_contract = _resolve_elision_plan(profile.options)

        mode = str(profile.options.get("mode", "reduce-overhead"))
        if mode not in self._MODES:
            raise ValueError(f"unsupported torch_compile mode: {mode!r}")
        fullgraph = bool(profile.options.get("fullgraph", False))
        dynamic_value = profile.options.get("dynamic")
        if dynamic_value is not None and not isinstance(dynamic_value, bool):
            raise TypeError("masked-view dynamic option must be a boolean or null")
        compile_options: dict[str, Any] = {"mode": mode, "fullgraph": fullgraph}
        if dynamic_value is not None:
            compile_options["dynamic"] = dynamic_value

        cudagraphs = profile.options.get("cudagraphs")
        if cudagraphs is not None and not isinstance(cudagraphs, bool):
            raise TypeError("masked-view cudagraphs option must be a boolean or null")
        if cudagraphs is not None:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = cudagraphs
            if not cudagraphs and hasattr(inductor_config.triton, "cudagraph_trees"):
                inductor_config.triton.cudagraph_trees = False

        validate_deployment_precision(adapter, profile)
        model.validate_profile(adapter, profile)
        policy = getattr(adapter, "policy", None)
        torch_model = getattr(policy, "_model", None)
        if policy is None or torch_model is None:
            raise TypeError(
                "masked-view backend requires a local PyTorch policy exposing _model"
            )
        optimized_sample_actions = self._sampler_factory(torch_model, indices)
        compiled_sample_actions = self._compile(optimized_sample_actions, **compile_options)

        prepared_policy = copy.copy(policy)
        prepared_policy._sample_kwargs = dict(getattr(policy, "_sample_kwargs", {}))
        prepared_policy._sample_actions = compiled_sample_actions
        prepared_adapter = copy.copy(adapter)
        prepared_adapter._policy = prepared_policy
        prepared_adapter._sample_kwargs = prepared_policy._sample_kwargs
        return PreparedPolicy(
            adapter=prepared_adapter,
            model_id=model.model_id,
            backend_id=self.backend_id,
            profile=profile,
            metadata={
                "transformed": True,
                "transform": "static_masked_view_elision+torch.compile",
                "elided_image_indices": list(indices),
                "elided_image_views": (
                    list(view_contract.masked_views) if view_contract is not None else None
                ),
                "input_view_contract": (
                    view_contract.to_dict() if view_contract is not None else None
                ),
                "mask_contract": "all_batch_entries_false",
                "compile_options": compile_options,
                "cudagraphs": cudagraphs,
                "source_adapter_unchanged": True,
            },
        )

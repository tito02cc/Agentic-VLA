"""Hugging Face OpenVLA policy with the official LIBERO input contract."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np


OPENVLA_CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
OPENVLA_EMPTY_ACTION_TOKEN_ID = 29871


@dataclasses.dataclass(frozen=True)
class OpenVlaLoadConfig:
    checkpoint: str | pathlib.Path
    device: str = "cuda:0"
    precision: str = "bf16"
    unnorm_key: str = "libero_10"
    center_crop: bool = True
    align_action_token_mask: bool = False
    code_revision: str = OPENVLA_CODE_REVISION
    attn_implementation: str = "eager"

    def __post_init__(self) -> None:
        precision = str(self.precision).strip().lower()
        if precision not in {"bf16", "int8", "nf4"}:
            raise ValueError("OpenVLA precision must be bf16, int8, or nf4")
        if not str(self.unnorm_key).strip():
            raise ValueError("unnorm_key must not be empty")
        object.__setattr__(self, "precision", precision)


def _as_rgb_image(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"OpenVLA image must have shape HxWx3/4, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def center_crop_openvla_image(image: Any, *, crop_area: float = 0.9) -> Any:
    """Apply a deterministic TensorFlow-free equivalent of OpenVLA's center crop."""

    from PIL import Image

    if not 0.0 < crop_area <= 1.0:
        raise ValueError("crop_area must be in (0, 1]")
    image = _as_rgb_image(image)
    if image.size != (224, 224):
        image = image.resize((224, 224), resample=Image.Resampling.LANCZOS)
    side_scale = math.sqrt(crop_area)
    crop_width = max(1, round(image.width * side_scale))
    crop_height = max(1, round(image.height * side_scale))
    left = (image.width - crop_width) / 2.0
    top = (image.height - crop_height) / 2.0
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((224, 224), resample=Image.Resampling.BILINEAR)


def postprocess_openvla_libero_action(action: Any) -> np.ndarray:
    """Translate OpenVLA's RLDS gripper value to the LIBERO controller convention."""

    output = np.asarray(action, dtype=np.float32).copy()
    if output.shape != (7,):
        raise ValueError(f"OpenVLA LIBERO action must have shape (7,), got {output.shape}")
    output[-1] = np.sign(2.0 * output[-1] - 1.0)
    output[-1] *= -1.0
    return output


def align_openvla_action_token_mask(inputs: Any, *, torch_module: Any) -> bool:
    """Append OpenVLA's empty action token and its attention-mask entry together."""

    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("OpenVLA processor inputs require rank-2 input_ids")
    if bool(torch_module.all(input_ids[:, -1] == OPENVLA_EMPTY_ACTION_TOKEN_ID)):
        return False
    token = torch_module.full(
        (input_ids.shape[0], 1),
        OPENVLA_EMPTY_ACTION_TOKEN_ID,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    inputs["input_ids"] = torch_module.cat((input_ids, token), dim=1)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        mask_entry = torch_module.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        inputs["attention_mask"] = torch_module.cat((attention_mask, mask_entry), dim=1)
    return True


class HuggingFaceOpenVlaPolicy:
    """Local OpenVLA deployment object exposing CARVE's ``infer(payload)`` API."""

    def __init__(self, model: Any, processor: Any, config: OpenVlaLoadConfig, torch: Any) -> None:
        self._model = model
        self._processor = processor
        self.config = config
        self._torch = torch

    @classmethod
    def from_pretrained(cls, config: OpenVlaLoadConfig) -> "HuggingFaceOpenVlaPolicy":
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        checkpoint = str(pathlib.Path(config.checkpoint).expanduser())
        dtype = torch.bfloat16
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "code_revision": config.code_revision,
            "attn_implementation": config.attn_implementation,
        }
        if config.precision == "int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = {"": config.device}
        elif config.precision == "nf4":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": config.device}

        model = AutoModelForVision2Seq.from_pretrained(checkpoint, **model_kwargs)
        if config.precision == "bf16":
            model = model.to(config.device)
        model.eval()

        statistics_path = pathlib.Path(checkpoint) / "dataset_statistics.json"
        if statistics_path.exists():
            model.norm_stats = json.loads(statistics_path.read_text(encoding="utf-8"))
        available_stats = getattr(model, "norm_stats", {})
        if config.unnorm_key not in available_stats:
            raise KeyError(
                f"OpenVLA checkpoint does not contain norm stats for {config.unnorm_key!r}"
            )
        processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            code_revision=config.code_revision,
        )
        return cls(model, processor, config, torch)

    @property
    def model(self) -> Any:
        return self._model

    def _prepare_image(self, image: Any) -> Any:
        image = _as_rgb_image(image)
        if self.config.center_crop:
            return center_crop_openvla_image(image)
        from PIL import Image

        return image.resize((224, 224), resample=Image.Resampling.LANCZOS)

    @staticmethod
    def _prompt(instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if "image" not in payload:
            raise ValueError("OpenVLA payload requires image")
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("OpenVLA payload requires a non-empty instruction")
        unnorm_key = str(payload.get("unnorm_key", self.config.unnorm_key))
        image = self._prepare_image(payload["image"])
        inputs = self._processor(self._prompt(instruction), image)
        if self.config.align_action_token_mask:
            align_openvla_action_token_mask(inputs, torch_module=self._torch)
        inputs = inputs.to(self.config.device, dtype=self._torch.bfloat16)

        if str(self.config.device).startswith("cuda"):
            self._torch.cuda.synchronize(self.config.device)
        started_s = time.perf_counter()
        with self._torch.inference_mode():
            action = self._model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=bool(payload.get("do_sample", False)),
            )
        if str(self.config.device).startswith("cuda"):
            self._torch.cuda.synchronize(self.config.device)
        infer_ms = (time.perf_counter() - started_s) * 1000.0
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"OpenVLA predicted action shape {action.shape}, expected (7,)")
        return {
            "actions": action,
            "policy_timing": {"infer_ms": infer_ms},
            "policy_metadata": {
                "precision": self.config.precision,
                "unnorm_key": unnorm_key,
                "center_crop": self.config.center_crop,
                "align_action_token_mask": self.config.align_action_token_mask,
                "action_semantics": "openvla_rlds_pre_environment_postprocess",
            },
        }

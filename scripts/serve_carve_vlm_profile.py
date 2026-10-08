#!/usr/bin/env python3
"""Serve an auditable CARVE Planner profile through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "bf16": {
        "deployment_precision": "bf16",
        "module_precisions": {},
        "preserved_modules": (),
    },
    "uniform_nf4": {
        "deployment_precision": "nf4_w4a16",
        "module_precisions": {"eligible_linear": "nf4"},
        "preserved_modules": ("lm_head",),
    },
    "uniform_int8": {
        "deployment_precision": "int8_w8a16",
        "module_precisions": {"eligible_linear": "int8"},
        "preserved_modules": ("lm_head",),
    },
    "vision_preserving_nf4": {
        "deployment_precision": "nf4_w4a16",
        "module_precisions": {"language_decoder_linear": "nf4"},
        "preserved_modules": ("model.visual", "lm_head"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_SPECS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18071)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Expose image/video inputs without requesting audio from video files.",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--vision-audit", type=Path,
                        help="Opt-in processor tensor fingerprints; adds CPU synchronization, not for latency measurements.")
    parser.add_argument("--log-level", default="warning")
    return parser.parse_args()


def build_quantization_config(profile: str):
    if profile == "bf16":
        return None

    from transformers import BitsAndBytesConfig

    spec = PROFILE_SPECS[profile]
    preserved = list(spec["preserved_modules"])
    if profile == "uniform_int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=preserved,
        )

    import torch

    skip_modules = preserved if profile == "vision_preserving_nf4" else None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=skip_modules,
    )


def checkpoint_bytes(model_path: Path) -> int:
    return sum(path.stat().st_size for path in model_path.glob("*.safetensors"))


def module_audit(model: Any) -> dict[str, Any]:
    try:
        import bitsandbytes as bnb

        quantized_types = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        quantized_types = ()

    import torch

    quantized_names: list[str] = []
    bf16_linear_names: list[str] = []
    quantized_logical_parameters = 0
    bf16_linear_parameters = 0
    for name, module in model.named_modules():
        logical_parameters = 0
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            logical_parameters = int(module.in_features) * int(module.out_features)
        if quantized_types and isinstance(module, quantized_types):
            quantized_names.append(name)
            quantized_logical_parameters += logical_parameters
        elif isinstance(module, torch.nn.Linear):
            bf16_linear_names.append(name)
            bf16_linear_parameters += logical_parameters

    return {
        "quantized_linear_count": len(quantized_names),
        "quantized_logical_parameters": quantized_logical_parameters,
        "bf16_linear_count": len(bf16_linear_names),
        "bf16_linear_parameters": bf16_linear_parameters,
        "quantized_name_examples": quantized_names[:20],
        "bf16_name_examples": bf16_linear_names[:20],
        "vision_quantized": any(name.startswith("model.visual") for name in quantized_names),
        "lm_head_quantized": any(name == "lm_head" for name in quantized_names),
    }


def cuda_memory() -> dict[str, float | None]:
    import torch

    if not torch.cuda.is_available():
        return {
            "allocated_gib": None,
            "reserved_gib": None,
            "peak_allocated_gib": None,
        }
    gib = float(1024**3)
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
    }


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"model directory does not exist: {model_path}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    import uvicorn
    from transformers.cli.serving.chat_completion import ChatCompletionHandler
    from transformers.cli.serving.completion import CompletionHandler
    from transformers.cli.serving.model_manager import ModelManager
    from transformers.cli.serving.response import ResponseHandler
    from transformers.cli.serving.server import build_server
    from transformers.cli.serving.transcription import TranscriptionHandler
    from transformers.cli.serving.utils import GenerationState, Modality

    quantization_config = build_quantization_config(args.profile)

    class ProfiledModelManager(ModelManager):
        def get_quantization_config(self):
            return quantization_config

        def get_model_modality(self, model, processor=None):
            modality = super().get_model_modality(model, processor=processor)
            if args.vision_only and modality == Modality.MULTIMODAL:
                return Modality.VLM
            return modality

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    manager = ProfiledModelManager(
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=False,
        attn_implementation=args.attn_implementation,
        quantization=None,
        model_timeout=-1,
        force_model=str(model_path),
    )
    loaded = next(iter(manager.loaded_models.values()))
    if args.vision_audit:
        args.vision_audit.parent.mkdir(parents=True, exist_ok=True)
        processor = loaded.processor
        original_template = processor.apply_chat_template

        def audited_template(*positional, **keywords):
            inputs = original_template(*positional, **keywords)
            tensors = {}
            for name in ("input_ids", "pixel_values", "image_grid_thw"):
                value = inputs.get(name)
                if isinstance(value, torch.Tensor):
                    cpu = value.detach().cpu().contiguous()
                    tensors[name] = {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
                        "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}
            with args.vision_audit.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": time.time(), "tensors": tensors}) + "\n")
            return inputs

        processor.apply_chat_template = audited_template
    audit = module_audit(loaded.model)
    spec = PROFILE_SPECS[args.profile]
    receipt = {
        "schema_version": "carve-planner-profile-v1",
        "profile_id": f"{model_path.name.lower()}-{args.profile}",
        "model_id": str(model_path),
        "profile": args.profile,
        "deployment_precision": spec["deployment_precision"],
        "module_precisions": spec["module_precisions"],
        "preserved_modules": list(spec["preserved_modules"]),
        "backend": "transformers+bitsandbytes" if quantization_config else "transformers",
        "checkpoint_bytes": checkpoint_bytes(model_path),
        "load_seconds": time.time() - started,
        "cuda_memory_after_load": cuda_memory(),
        "module_audit": audit,
        "software": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "vision_only_serving": args.vision_only,
        "admission": {
            "status": "calibration_only",
            "semantic_gate": "pending",
            "co_resident_gate": "pending",
        },
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    generation_state = GenerationState(continuous_batching=False, compile=False)
    chat_template_kwargs = {"enable_thinking": False}
    chat_handler = ChatCompletionHandler(
        model_manager=manager,
        generation_state=generation_state,
        chat_template_kwargs=chat_template_kwargs,
    )
    app = build_server(
        manager,
        chat_handler,
        completion_handler=CompletionHandler(manager, generation_state),
        response_handler=ResponseHandler(
            manager,
            generation_state,
            chat_template_kwargs=chat_template_kwargs,
        ),
        transcription_handler=TranscriptionHandler(manager, generation_state),
        generation_state=generation_state,
        enable_cors=False,
    )

    @app.get("/carve/profile")
    def carve_profile():
        return receipt

    print(json.dumps(receipt, indent=2), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Opt-in serialized GPU sharing for a separate, unquantized visual planner."""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

import torch

from deployment.model_server.tools.image_tools import to_pil_preserve


class CpuStagedVisionPlanner:
    """Keep inactive model weights on CPU; never overlap action and planning.

    The caller must pause environment stepping at the planning boundary. This
    saves GPU residency, not latency, and is not a real-time control guarantee.
    """

    def __init__(self, policy, model_path: str, device: str = "cuda", *, residency_mode="transfer"):
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"staged VLM directory does not exist: {path}")
        self.policy = policy
        self.model_path = str(path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("staged VLM requires a CUDA action-model device")
        self.model = None
        self.processor = None
        self.failed = False
        self._lock = threading.RLock()
        if residency_mode not in ("transfer", "cpu_mirror"):
            raise ValueError("unknown staged residency mode")
        self.residency_mode = residency_mode
        self._policy_mirror = None
        self._vlm_mirror = None
        self._startup_receipt = None

    def prepare(self):
        """Warm the separate planner before a task, without task observations."""
        import numpy as np

        with self.action_boundary():
            if self._startup_receipt is None:
                result = self.decide(
                    system_prompt="You are a helpful assistant.",
                    user_prompt="This is a startup check, not a robot task. Reply READY.",
                    frames={"head": np.zeros((480, 640, 3), dtype=np.uint8)},
                    max_new_tokens=16,
                )
                self._startup_receipt = {"purpose": "startup_only_not_task_evidence",
                                         "image": "synthetic_black_rgb_480x640",
                                         "task_observations_used": False, **result}
            return dict(self._startup_receipt)

    def _prepare_mirrors(self):
        if self.residency_mode == "cpu_mirror" and self._policy_mirror is None:
            from agentic_vla.optimization.residency import FrozenParameterCpuMirror

            self._policy_mirror = FrozenParameterCpuMirror(self.policy)

    @contextlib.contextmanager
    def action_boundary(self):
        with self._lock:
            if self.failed:
                raise RuntimeError("model residency restoration failed; restart required")
            yield

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        # Mirrors require normal versioned parameters for mutation detection.
        with torch.inference_mode(False):
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
        model.requires_grad_(False)
        self.model, self.processor = model, processor
        if self.residency_mode == "cpu_mirror":
            from agentic_vla.optimization.residency import FrozenParameterCpuMirror

            with torch.inference_mode(False):
                self._vlm_mirror = FrozenParameterCpuMirror(model)

    def _generate(self, system_prompt, user_prompt, frames, max_new_tokens):
        images = to_pil_preserve(list(frames.values()))
        content = [{"type": "text", "text": user_prompt}]
        for name, image in zip(frames, images, strict=True):
            content.extend([
                {"type": "text", "text": f"Camera: {name}"},
                {"type": "image", "image": image},
            ])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt", padding=True,
        ).to(self.device)
        length = inputs["input_ids"].shape[1]
        generated = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
        )
        tokens = generated[:, length:]
        answer = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        if not answer:
            raise RuntimeError("staged VLM returned no text")
        return {"text": answer, "input_tokens": length, "output_tokens": tokens.shape[1]}

    @torch.inference_mode()
    def decide(self, *, system_prompt, user_prompt, frames=None, max_new_tokens=512):
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must not be empty")
        if len(system_prompt) + len(user_prompt) > 40000:
            raise ValueError("staged planner prompt exceeds the input budget")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be in [1, 1024]")
        frames = dict(frames or {})
        if len(frames) > 3:
            raise ValueError("staged planner accepts at most three camera frames")

        timings = {}
        started = time.perf_counter()
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with self.action_boundary(), torch.random.fork_rng(devices=[device_index]):
            cold = self.model is None
            try:
                phase = time.perf_counter()
                with torch.inference_mode(False):
                    self._prepare_mirrors()
                timings["mirror_prepare_ms"] = (time.perf_counter() - phase) * 1000
                torch.cuda.synchronize(self.device)
                phase = time.perf_counter()
                if self._policy_mirror is None:
                    self.policy.to("cpu")
                else:
                    self._policy_mirror.offload()
                torch.cuda.empty_cache()
                timings["vla_to_cpu_ms"] = (time.perf_counter() - phase) * 1000
                phase = time.perf_counter()
                self._load()
                timings["vlm_load_ms"] = (time.perf_counter() - phase) * 1000
                phase = time.perf_counter()
                if self._vlm_mirror is None:
                    self.model.to(self.device)
                else:
                    self._vlm_mirror.activate(self.device)
                torch.cuda.synchronize(self.device)
                timings["vlm_to_gpu_ms"] = (time.perf_counter() - phase) * 1000
                phase = time.perf_counter()
                result = self._generate(system_prompt, user_prompt, frames, max_new_tokens)
                torch.cuda.synchronize(self.device)
                timings["vlm_generate_ms"] = (time.perf_counter() - phase) * 1000
            finally:
                # If either move fails, never permit another action inference.
                try:
                    phase = time.perf_counter()
                    if self.model is not None:
                        if self._vlm_mirror is None:
                            self.model.to("cpu")
                        else:
                            self._vlm_mirror.offload()
                    torch.cuda.empty_cache()
                    timings["vlm_to_cpu_ms"] = (time.perf_counter() - phase) * 1000
                    phase = time.perf_counter()
                    if self._policy_mirror is None:
                        self.policy.to(self.device)
                    else:
                        self._policy_mirror.activate(self.device)
                    torch.cuda.synchronize(self.device)
                    timings["vla_restore_ms"] = (time.perf_counter() - phase) * 1000
                except BaseException:
                    self.failed = True
                    raise
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return {
            **result, "shared_backbone": False, "backend": "cpu_staged_vlm",
            "model_path": self.model_path, "precision": "bf16", "cold_load": cold,
            "timings": timings, "action_model_restored": True,
            "residency_mode": self.residency_mode,
            "cpu_mirror_bytes": sum(m.cpu_bytes for m in (self._policy_mirror, self._vlm_mirror) if m is not None),
        }

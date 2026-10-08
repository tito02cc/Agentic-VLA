"""Bounded model-selected region inspection followed by action-free verification."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from agentic_vla.runtime.agent import PlannerCallable

from ._json import decode_json_object
from .contracts import ToolExecutionContext
from .inspection import VisualEvidenceStore
from .progress import (
    GuardedProgressVerifier,
    VisualProgressContext,
    VisualProgressResult,
)
from .runtime import CanonicalToolRuntime, EmbodiedToolBindings
from .workspace import RunWorkspace


def _unavailable(*args: Any) -> Any:
    raise RuntimeError("inspection verification has no robot or memory-writing binding")


class InspectingProgressVerifier:
    """At most two provider calls, one read-only tool call, and no control effects.

    The host must serialize this operation with execution/reset and supply a live
    context reader. This is synchronous, not a hard-deadline implementation.
    """

    def __init__(
        self,
        infer: PlannerCallable,
        workspace: RunWorkspace,
        *,
        minimum_confidence: float = 0.55,
    ):
        if not callable(infer):
            raise TypeError("inspection provider must be callable")
        if not isinstance(workspace, RunWorkspace):
            raise TypeError("inspection requires an auditable RunWorkspace")
        self.infer, self.workspace = infer, workspace
        self.minimum_confidence = minimum_confidence
        GuardedProgressVerifier(infer, minimum_confidence=minimum_confidence)

    def verify(
        self,
        context: VisualProgressContext,
        *,
        execution_context: ToolExecutionContext,
        current_context: Callable[[], ToolExecutionContext],
    ) -> VisualProgressResult:
        def checked_context():
            current = current_context()
            if (
                type(current.episode_id) is not type(execution_context.episode_id)
                or current != execution_context
            ):
                raise ValueError("inspection context changed during semantic inference")
            return current

        def provider(request, phase):
            checked_context()
            started = time.perf_counter()
            error, raw = None, None
            try:
                raw = self.infer(request)
                checked_context()
                return raw
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                self.workspace.append_event(
                    "inspection_model_call",
                    {
                        "phase": phase,
                        "episode_id": execution_context.episode_id,
                        "timestep": execution_context.timestep,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                        "frames": list(request["frames"]),
                        "error": error,
                        "raw_output": str(raw)[:16384] if raw is not None else None,
                    },
                    source="inspection_verifier",
                )

        def inspect_then_verify(request):
            if len(context.frames) != 1:
                raise ValueError(
                    "inspection requires exactly one canonical camera view"
                )
            checked_context()
            if "inspect_region" not in execution_context.allowed_tools:
                raise ValueError("inspect_region is not allowed")
            camera, image = next(iter(context.frames.items()))
            store = VisualEvidenceStore(
                max_frames=1, max_inspections=1, max_age_steps=0
            )
            store.reset(execution_context.episode_id)
            frame = store.capture(camera, image, execution_context)
            snapshot = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(
                image.shape
            )
            runtime = CanonicalToolRuntime(
                bindings=EmbodiedToolBindings(*([_unavailable] * 6)),
                workspace=self.workspace,
                visual_evidence=store,
            )
            selection = provider(
                {
                    "system_prompt": (
                        "Select an optional useful region to inspect the requested visible relations. "
                        'Return JSON only: {"inspect":null} to use the full image, or '
                        '{"inspect":{"frame_id":"...","left":0,"top":0,"right":32,"bottom":32}}. '
                        "Coordinates are integer native pixels, right/bottom exclusive, minimum 16x16. "
                        "Select from the supplied frame only. Do not return actions, task completion, "
                        "new images, paths, hidden poses, or confidence. A crop is not a new view."
                    ),
                    "user_prompt": json.dumps(
                        {
                            "question": json.loads(request["user_prompt"]),
                            "frame": frame,
                            "tool": store.tool_spec.input_schema,
                        },
                        ensure_ascii=True,
                    ),
                    "frames": {camera: snapshot},
                    "required_frame_names": [camera],
                },
                "region_selection",
            )
            selection = decode_json_object(selection)
            if not isinstance(selection, Mapping) or set(selection) != {"inspect"}:
                raise ValueError("selection must contain only inspect")
            receipt = None
            final_frames = {camera: snapshot}
            if selection["inspect"] is not None:
                if not isinstance(selection["inspect"], Mapping):
                    raise ValueError("inspect must be a region object or null")
                result = runtime.invoke(
                    "inspect_region", selection["inspect"], context=checked_context()
                )
                if not result.accepted:
                    raise ValueError(f"inspection tool rejected: {result.error}")
                receipt = dict(result.output)
                final_frames = store.resolve_images(
                    receipt["inspection_id"], checked_context()
                )
            final_request = {
                **request,
                "frames": final_frames,
                "required_frame_names": list(final_frames),
                "system_prompt": request["system_prompt"]
                + (
                    " If global and region images are supplied, region is a crop of global, "
                    "not another view or independent evidence. Retain global spatial context; "
                    "count physical objects only once. A close-up does not reveal hidden support."
                ),
                "user_prompt": json.dumps(
                    {
                        **json.loads(request["user_prompt"]),
                        "inspection_receipt": receipt,
                    },
                    ensure_ascii=True,
                ),
            }
            return provider(final_request, "relation_verification")

        result = GuardedProgressVerifier(
            inspect_then_verify,
            minimum_confidence=self.minimum_confidence,
        ).verify(context)
        self.workspace.append_event(
            "inspection_progress_result",
            {
                "episode_id": execution_context.episode_id,
                "timestep": execution_context.timestep,
                "accepted": result.accepted,
                "confirmed_prefix": list(result.confirmed_prefix),
                "reports": {
                    name: report.to_dict() for name, report in result.reports.items()
                },
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
                "control_applied": False,
            },
            source="inspection_verifier",
        )
        return result

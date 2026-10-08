#!/usr/bin/env python
"""Replay captured RoboDojo payloads through the native and wrapped paths.

Why this exists
---------------
Plan item 9.21.1 asked whether the ``CarveRuntime`` wrapper changes what the VLA
outputs. Comparing two live runs cannot answer it: the rendered RGB differs from
the first frame of every process, so no two runs share an ``input_sha256``
(measured 0 shared rows across B0/C1/C2/C3, see
``artifacts/robodojo/native_trace_b0_20260911/trace_comparison.json``).

This script removes the simulator from the question. It loads the checkpoint once
in-process, then feeds each *recorded* payload through both paths:

* native   -- the flat payload the baseline arm sends, called directly, which is
              what ``model.py::_infer_chunk`` does when ``carve_mode=baseline``
* wrapped  -- the same payload handed to ``CarveRuntime.infer`` through
              ``StarVlaAdapter``, which is what C1/C2/C3 do

Both hit the same resident weights in the same process with the same
``action_seed``, so the server's per-request ``fork_rng`` gives both the same
sampling noise. That makes the comparison falsifiable: if the wrapper is
output-preserving at matched compute, the two action chunks must be bit-identical.

Two comparisons are reported, because they answer different questions:

* ``matched_steps``  -- wrapper asked for the same effective step count the
  native path runs (the checkpoint default). Isolates the wrapper itself.
* ``arm_default``    -- wrapper asked for the C1/C3 default step count. This is
  the real difference between the frozen arms, so it measures the step
  reduction, not the wrapper.

Usage:
    openpi/.venv/bin/python scripts/replay_native_vs_runtime_payloads.py \
        --capture-dir artifacts/robodojo/.../captured_payloads \
        --checkpoint third_party/.../steps_100000_pytorch_model.pt \
        --output artifacts/robodojo/.../replay_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STARVLA_ROOT = (
    REPO_ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA"
)


def load_manifest(capture_dir: Path) -> list[dict[str, Any]]:
    manifest = capture_dir / "payload_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"no payload manifest in {capture_dir}")
    entries = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"empty payload manifest: {manifest}")
    return entries


def rebuild_payload(
    capture_dir: Path, entry: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    """Rebuild the exact payload the policy received, or fail.

    The rebuilt payload must re-derive the ``input_sha256`` recorded live. A
    mismatch means the replay would be comparing paths on an observation the
    robot never saw, which is worse than having no comparison at all.
    """
    from agentic_vla.runtime.adapters.starvla import hash_starvla_request_input

    arrays = np.load(capture_dir / entry["arrays"])
    example: dict[str, Any] = {
        "image": [arrays[f"image_{view}"] for view in range(entry["image_views"])],
        "lang": entry["lang"],
    }
    if entry["has_state"]:
        example["state"] = arrays["state"]
    payload: dict[str, Any] = {"examples": [example]}
    for key in ("do_sample", "use_ddim", "num_ddim_steps", "unnorm_key", "action_seed"):
        if entry.get(key) is not None:
            payload[key] = entry[key]

    digest = hash_starvla_request_input(payload)
    if digest != entry["input_sha256"]:
        raise ValueError(
            f"payload {entry['index']} did not round-trip: rebuilt {digest} != "
            f"recorded {entry['input_sha256']}"
        )
    return payload, np.ascontiguousarray(arrays["actions"])


class InProcessStarVlaClient:
    """Present the websocket client contract over a resident wrapper.

    The real arms talk to a websocket policy server. Here the same
    ``PolicyServerWrapper`` is held in-process, so both paths share one set of
    weights and neither pays transport cost. Payload keys are forwarded exactly
    as the server forwards them, including the legacy ones the framework
    discards, so the replay stays faithful to what the arms actually send.
    """

    def __init__(self, wrapper: Any) -> None:
        self._wrapper = wrapper
        self.calls = 0

    def predict_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        examples = request.pop("examples")
        unnorm_key = request.pop("unnorm_key", None)
        self.calls += 1
        out = self._wrapper.predict_action(
            examples=examples, unnorm_key=unnorm_key, **request
        )
        return {"ok": True, "data": {"actions": out["actions"]}}


def action_sha256(actions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


def call_native(
    client: InProcessStarVlaClient, payload: dict[str, Any]
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    response = client.predict_action(dict(payload))
    latency_ms = (time.perf_counter() - started) * 1000.0
    actions = np.asarray(response["data"]["actions"][0], dtype=np.float32)
    return actions, latency_ms


def call_wrapped(
    runtime: Any,
    payload: dict[str, Any],
    *,
    inference_steps: int,
    max_actions: int,
    episode_id: int,
    timestep: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    from agentic_vla.runtime import InferenceRequest
    from agentic_vla.runtime.contracts import InferenceControls

    request = InferenceRequest(
        observation={"examples": payload["examples"]},
        instruction=str(payload["examples"][0].get("lang", "")),
        controls=InferenceControls(
            inference_steps=inference_steps, max_actions=max_actions
        ),
        episode_id=episode_id,
        timestep=timestep,
        metadata={"raw_payload": dict(payload), "action_age_steps": 0},
    )
    started = time.perf_counter()
    chunk = runtime.infer(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    metadata = dict(chunk.metadata)
    # The wrapper cost measured inside one invocation: total time through
    # CarveRuntime minus the model call the adapter itself timed. Differencing
    # the wrapped and native totals instead would fold in run-to-run variance of
    # two separate model calls, which on this checkpoint is tens of ms.
    metadata["_adapter_model_latency_ms"] = float(chunk.model_latency_ms)
    metadata["_wrapper_overhead_ms"] = max(
        0.0, latency_ms - float(chunk.model_latency_ms)
    )
    return (
        np.asarray(chunk.actions, dtype=np.float32),
        latency_ms,
        metadata,
    )


def compare_arrays(native: np.ndarray, wrapped: np.ndarray) -> dict[str, Any]:
    """Compare two action chunks on the rows both actually produced.

    The two paths legitimately return different row counts: the native path
    hands back the model's full chunk (50 steps for this checkpoint) and the
    caller executes only ``execute_horizon`` of them, while the wrapped path
    applies ``max_actions`` inside the adapter and returns the truncated chunk.
    That truncation is a declared control, not an output change, so the question
    "does the wrapper change the output" is about the overlapping prefix, which
    is also the part that ever reaches the robot.
    """
    if native.shape[1:] != wrapped.shape[1:]:
        return {
            "shape_native": list(native.shape),
            "shape_wrapped": list(wrapped.shape),
            "identical": False,
            "action_dim_mismatch": True,
        }
    rows = min(native.shape[0], wrapped.shape[0])
    left = native[:rows]
    right = wrapped[:rows]
    difference = np.abs(right.astype(np.float64) - left.astype(np.float64))
    return {
        "shape_native": list(native.shape),
        "shape_wrapped": list(wrapped.shape),
        "action_dim_mismatch": False,
        "rows_compared": int(rows),
        "wrapper_truncated_chunk": bool(wrapped.shape[0] < native.shape[0]),
        "identical": bool(np.array_equal(left, right)),
        "prefix_sha256_equal": action_sha256(left) == action_sha256(right),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
    }


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min_ms": round(ordered[0], 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "max_ms": round(ordered[-1], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
    }


def build_runtime(client: InProcessStarVlaClient, *, steps: int, horizon: int) -> Any:
    from agentic_vla.optimization import OptimizationProfile, create_default_runtime
    from agentic_vla.runtime import ActionSpec, CarveRuntime
    from agentic_vla.runtime.adapters import StarVlaAdapter

    action_spec = ActionSpec(
        action_dim=14,
        representation="dual_arm_joint_position",
        coordinate_frame="robot_joint",
        gripper_convention="xpolicy_joint_gripper",
        control_frequency_hz=25.0,
        normalization_id="starvla/arx_x5",
    )
    adapter = StarVlaAdapter(
        client,
        max_action_horizon=50,
        default_num_ddim_steps=steps,
        inference_steps_key="num_inference_steps",
        use_ddim=True,
        unnorm_key="arx_x5",
        action_spec=action_spec,
    )
    profile = OptimizationProfile(
        profile_id=f"starvla-flow{steps}-h{horizon}",
        backend="eager",
        deployment_precision="bf16",
        inference_steps=steps,
        action_horizon=horizon,
    )
    # Same construction the live arms use in
    # model.py::_initialize_carve_runtime: prepare the adapter through the
    # optimization runtime, then wrap it strictly. Building CarveRuntime any
    # other way here would not be replaying the deployed path.
    prepared = create_default_runtime().prepare(adapter, profile)
    return CarveRuntime(prepared.adapter, fallback_mode="strict")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-horizon", type=int, default=16)
    parser.add_argument(
        "--arm-default-steps",
        type=int,
        default=2,
        help="step count the C1/C3 arms default to",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--base-vlm",
        type=Path,
        default=Path("/home/admin1/models/Qwen3-VL-4B-Instruct"),
        help="local base VLM, same default the launcher uses",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(STARVLA_ROOT / "source_starvla"))

    # Load exactly the way the live arms load. Without these the checkpoint YAML
    # points the base VLM at a HuggingFace repo id and the loader blocks on the
    # network instead of using the local weights, which looks like a hang rather
    # than an error. Overridable, but defaulted so the replay cannot silently
    # run against different weights than the arms did.
    import os

    os.environ.setdefault("STARVLA_HF_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("STARVLA_HF_SKIP_WEIGHT_HASH", "1")
    os.environ.setdefault("STARVLA_BASE_VLM", str(args.base_vlm))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    entries = load_manifest(args.capture_dir)
    payloads = [rebuild_payload(args.capture_dir, entry) for entry in entries]
    print(f"rebuilt {len(payloads)} payloads, all digests match the live run")

    # The effective native step count is a property of the checkpoint, recorded
    # live by the server handshake. Trust the capture, do not re-derive it.
    native_steps = {entry["live_inference_steps"] for entry in entries}
    if len(native_steps) != 1:
        raise ValueError(f"captured payloads disagree on native steps: {native_steps}")
    matched_steps = int(native_steps.pop())

    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    wrapper = PolicyServerWrapper(
        ckpt_path=str(args.checkpoint.resolve()), device="cuda", use_bf16=True
    )
    client = InProcessStarVlaClient(wrapper)
    torch.cuda.reset_peak_memory_stats()

    runtimes = {
        "matched_steps": build_runtime(
            client, steps=matched_steps, horizon=args.execute_horizon
        ),
        "arm_default": build_runtime(
            client, steps=args.arm_default_steps, horizon=args.execute_horizon
        ),
    }
    step_counts = {
        "matched_steps": matched_steps,
        "arm_default": args.arm_default_steps,
    }

    # Warm up on the first payload so no comparison pays a cold-start cost.
    warm_payload = payloads[0][0]
    for _ in range(max(0, args.warmup)):
        call_native(client, warm_payload)
        call_wrapped(
            runtimes["matched_steps"],
            warm_payload,
            inference_steps=matched_steps,
            max_actions=args.execute_horizon,
            episode_id=0,
            timestep=0,
        )

    native_latency: list[float] = []
    wrapped_latency: dict[str, list[float]] = {name: [] for name in runtimes}
    wrapper_overhead: dict[str, list[float]] = {name: [] for name in runtimes}
    native_scale: list[dict[str, float]] = []
    per_payload: list[dict[str, Any]] = []
    reproduced_live = 0

    for entry, (payload, live_actions) in zip(entries, payloads):
        native_actions, native_ms = call_native(client, payload)
        native_latency.append(native_ms)
        row: dict[str, Any] = {
            "index": entry["index"],
            "timestep": entry["timestep"],
            "reset_generation": entry["reset_generation"],
            "action_seed": entry["action_seed"],
            "input_sha256": entry["input_sha256"],
            # Does replaying the recorded payload reproduce the live output?
            "native_reproduces_live_action": bool(
                np.array_equal(
                    native_actions, live_actions[: native_actions.shape[0]]
                )
            ),
            "live_vs_replay_native": compare_arrays(
                live_actions[: native_actions.shape[0]], native_actions
            ),
            "comparisons": {},
        }
        if row["native_reproduces_live_action"]:
            reproduced_live += 1
        # Action magnitude, so an absolute difference is interpretable rather
        # than being a bare number in unstated units.
        executed = native_actions[: args.execute_horizon]
        scale = {
            "abs_max": float(np.abs(executed).max()),
            "abs_mean": float(np.abs(executed).mean()),
        }
        native_scale.append(scale)
        row["native_action_scale"] = {k: round(v, 6) for k, v in scale.items()}

        for name, runtime in runtimes.items():
            wrapped_actions, wrapped_ms, metadata = call_wrapped(
                runtime,
                payload,
                inference_steps=step_counts[name],
                max_actions=args.execute_horizon,
                episode_id=entry["reset_generation"],
                timestep=entry["timestep"],
            )
            wrapped_latency[name].append(wrapped_ms)
            row["comparisons"][name] = {
                "requested_inference_steps": step_counts[name],
                "wrapper_reported_inference_steps": metadata.get("inference_steps"),
                "wrapper_input_sha256_equals_native": metadata.get("input_sha256")
                == entry["input_sha256"],
                "action": compare_arrays(native_actions, wrapped_actions),
                "wrapped_latency_ms": round(wrapped_ms, 3),
                "native_latency_ms": round(native_ms, 3),
                "adapter_model_latency_ms": round(
                    metadata["_adapter_model_latency_ms"], 3
                ),
                "wrapper_overhead_ms": round(metadata["_wrapper_overhead_ms"], 3),
            }
            wrapper_overhead[name].append(metadata["_wrapper_overhead_ms"])
        per_payload.append(row)

    comparisons: dict[str, Any] = {}
    for name in runtimes:
        actions = [row["comparisons"][name]["action"] for row in per_payload]
        identical = sum(1 for entry in actions if entry["identical"])
        deltas = [
            row["comparisons"][name]["wrapped_latency_ms"]
            - row["comparisons"][name]["native_latency_ms"]
            for row in per_payload
        ]
        comparisons[name] = {
            "requested_inference_steps": step_counts[name],
            "native_effective_inference_steps": matched_steps,
            "compute_matched": step_counts[name] == matched_steps,
            "payloads": len(actions),
            "bit_identical_actions": identical,
            "bit_identical_fraction": round(identical / len(actions), 4),
            "rows_compared_per_payload": sorted(
                {entry.get("rows_compared") for entry in actions}
            ),
            "wrapper_truncated_chunk": all(
                entry.get("wrapper_truncated_chunk") for entry in actions
            ),
            "max_abs_diff_over_payloads": max(
                entry.get("max_abs_diff", float("nan")) for entry in actions
            ),
            "mean_abs_diff_mean": round(
                statistics.fmean(entry.get("mean_abs_diff", 0.0) for entry in actions), 8
            ),
            "input_digest_preserved_by_wrapper": all(
                row["comparisons"][name]["wrapper_input_sha256_equals_native"]
                for row in per_payload
            ),
            "wrapped_latency": summarize(wrapped_latency[name]),
            # In-invocation wrapper cost. This is the estimator to quote: both
            # terms come from the same call, so it carries no cross-call variance.
            "wrapper_overhead_in_invocation": summarize(wrapper_overhead[name]),
            # Kept for completeness, but it differences two separate model calls
            # and therefore includes their variance. Not a wrapper-cost estimate.
            "paired_wrapped_minus_native_ms": {
                "n": len(deltas),
                "median": round(statistics.median(deltas), 3),
                "mean": round(statistics.fmean(deltas), 3),
                "note": (
                    "two independent model calls; includes call-to-call variance, "
                    "use wrapper_overhead_in_invocation for wrapper cost"
                ),
            },
            "interpretation": (
                "isolates the wrapper: same effective compute on both paths"
                if step_counts[name] == matched_steps
                else "measures the step reduction the arm applies, not the wrapper"
            ),
        }

    gpu = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "carve.native-vs-runtime-replay.v1",
        "design": {
            "resident_model": True,
            "single_process": True,
            "real_recorded_payloads": True,
            "identical_action_seed_per_pair": True,
            "warmup_calls": args.warmup,
            "execute_horizon": args.execute_horizon,
            "native_effective_inference_steps": matched_steps,
            "native_steps_source": entries[0]["live_inference_steps_source"],
        },
        "sources": {
            "capture_dir": str(args.capture_dir),
            "checkpoint": str(args.checkpoint.resolve()),
            "base_vlm": str(args.base_vlm),
        },
        "payloads": len(payloads),
        "all_payload_digests_round_tripped": True,
        "replay_reproduces_live_action": {
            "payloads": len(payloads),
            "reproduced": reproduced_live,
            "fraction": round(reproduced_live / len(payloads), 4),
        },
        "native_latency": summarize(native_latency),
        "native_action_scale_over_executed_horizon": {
            "abs_max": round(max(s["abs_max"] for s in native_scale), 6),
            "abs_mean": round(
                statistics.fmean(s["abs_mean"] for s in native_scale), 6
            ),
        },
        "comparisons": comparisons,
        "hardware": {
            "gpu": gpu.name,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        },
        "total_model_calls": client.calls,
        "claim_boundary": (
            "This isolates whether the CarveRuntime wrapper changes VLA output and "
            "what it costs, on recorded observations with a resident model in one "
            "process. It is not a task success measurement, not closed-loop "
            "latency, and not a hard-real-time guarantee."
        ),
        "per_payload": per_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    for name, block in comparisons.items():
        overhead = block["wrapper_overhead_in_invocation"]
        print(
            f"  {name}: steps={block['requested_inference_steps']} "
            f"compute_matched={block['compute_matched']} "
            f"bit_identical={block['bit_identical_actions']}/{block['payloads']} "
            f"max_abs_diff={block['max_abs_diff_over_payloads']:.6g} "
            f"wrapper_overhead_p50={overhead['p50_ms'] if overhead else None} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

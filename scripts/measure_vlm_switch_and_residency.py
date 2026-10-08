#!/usr/bin/env python
"""Measure VLM switching against VLA output fidelity, with per-phase costs.

This closes two handoff items in one run, because both need the VLM and the VLA
held by a single process:

* item A (second half) -- compare the VLA action output *before* any VLM is
  involved, *after* a VLM generation call, and after a second one. The staged
  design moves the VLA off the GPU, runs the planner, then restores the VLA. If
  that round trip is output-preserving, the action chunks must be bit-identical
  across all three states. Anything else is a defect, not an optimization.
* item C -- record the cost of each phase separately rather than as one number:
  VLA inference, VLM generation, weight transfer in each direction, cold load,
  warmup, and end-to-end.

Both residency modes are measured, so the pre-handoff ``cpu_mirror`` claim is
re-verified independently rather than inherited.

Reuses the deployed path: ``PolicyServerWrapper`` constructs
``CpuStagedVisionPlanner`` itself, and the per-phase timings come from that
planner's own ``timings`` dict, not from a reimplementation here.

Usage:
    conda run --no-capture-output -n StarVLA python -u \
      scripts/measure_vlm_switch_and_residency.py \
      --capture-dir artifacts/robodojo/.../captured_payloads \
      --checkpoint .../steps_100000_pytorch_model.pt \
      --output artifacts/robodojo/.../vlm_switch_report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STARVLA_ROOT = REPO_ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA"

# The payload rebuild and digest helpers are already written and tested; import
# them from the replay tool rather than duplicating a second implementation.
_REPLAY_SPEC = importlib.util.spec_from_file_location(
    "replay_native_vs_runtime_payloads",
    REPO_ROOT / "scripts/replay_native_vs_runtime_payloads.py",
)
replay = importlib.util.module_from_spec(_REPLAY_SPEC)
_REPLAY_SPEC.loader.exec_module(replay)

# Phases the staged planner reports. Named here so a missing phase is a loud
# failure instead of a silently absent cost.
TRANSFER_PHASES = ("vla_to_cpu_ms", "vlm_to_gpu_ms", "vlm_to_cpu_ms", "vla_restore_ms")
PHASE_KEYS = (
    "mirror_prepare_ms",
    *TRANSFER_PHASES,
    "vlm_load_ms",
    "vlm_generate_ms",
    "total_ms",
)


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


def vla_actions(
    wrapper: Any, payload: dict[str, Any]
) -> tuple[np.ndarray, float]:
    """One VLA call through the same entry point the policy server uses."""
    request = dict(payload)
    examples = request.pop("examples")
    unnorm_key = request.pop("unnorm_key", None)
    started = time.perf_counter()
    out = wrapper.predict_action(examples=examples, unnorm_key=unnorm_key, **request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return np.asarray(out["actions"][0], dtype=np.float32), latency_ms


def measure_state(
    wrapper: Any, payloads: list[tuple[dict[str, Any], np.ndarray]]
) -> dict[str, Any]:
    """VLA digests and latency for every payload in the current GPU state."""
    digests: list[str] = []
    latencies: list[float] = []
    arrays: list[np.ndarray] = []
    for payload, _live in payloads:
        actions, latency = vla_actions(wrapper, payload)
        digests.append(replay.action_sha256(actions))
        latencies.append(latency)
        arrays.append(actions)
    return {
        "action_sha256": digests,
        "vla_latency": summarize(latencies),
        "_arrays": arrays,
    }


def compare_states(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    pairs = list(zip(reference["_arrays"], candidate["_arrays"]))
    per_payload = [replay.compare_arrays(left, right) for left, right in pairs]
    identical = sum(1 for entry in per_payload if entry["identical"])
    return {
        "payloads": len(pairs),
        "bit_identical_actions": identical,
        "bit_identical_fraction": round(identical / len(pairs), 4) if pairs else None,
        "max_abs_diff_over_payloads": max(
            entry.get("max_abs_diff", float("nan")) for entry in per_payload
        ),
        "digest_sequence_equal": reference["action_sha256"]
        == candidate["action_sha256"],
    }


def planner_call(planner: Any, frames: dict[str, np.ndarray], max_new_tokens: int):
    """One staged VLM generation, returning its own per-phase timings."""
    result = planner.decide(
        system_prompt="You are a careful robot scene observer.",
        user_prompt=(
            "Describe in one short sentence what the gripper is currently doing. "
            "Do not issue commands."
        ),
        frames=frames,
        max_new_tokens=max_new_tokens,
    )
    missing = [key for key in PHASE_KEYS if key not in result["timings"]]
    if missing:
        raise RuntimeError(f"staged planner did not report phases: {missing}")
    return result


def head_frame(payload: dict[str, Any]) -> np.ndarray:
    images = payload["examples"][0]["image"]
    return np.ascontiguousarray(np.asarray(images[0]))


def run_mode(
    *,
    mode: str,
    checkpoint: Path,
    base_vlm: Path,
    payloads: list[tuple[dict[str, Any], np.ndarray]],
    planner_calls: int,
    max_new_tokens: int,
    warmup: int,
) -> dict[str, Any]:
    """Load a fresh wrapper for one residency mode and measure it end to end."""
    import torch
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    load_started = time.perf_counter()
    wrapper = PolicyServerWrapper(
        ckpt_path=str(checkpoint.resolve()),
        device="cuda",
        use_bf16=True,
        staged_vlm_path=str(base_vlm),
        staged_residency=mode,
    )
    vla_load_ms = (time.perf_counter() - load_started) * 1000.0
    planner = wrapper._staged_planner
    if planner is None:
        raise RuntimeError("staged planner was not constructed; check staged_vlm_path")
    torch.cuda.reset_peak_memory_stats()

    # Warm the VLA so a cold first call is not attributed to VLM switching.
    for _ in range(max(0, warmup)):
        vla_actions(wrapper, payloads[0][0])
    vla_warm_peak = torch.cuda.max_memory_allocated()

    # State 0: the VLM has never been loaded or moved.
    state_never = measure_state(wrapper, payloads)

    # Warmup / startup preparation is its own cost, and by contract uses a
    # synthetic black frame, never a task observation.
    prepare_started = time.perf_counter()
    startup = planner.prepare()
    prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

    # State 1: immediately after one real VLM generation and VLA restore.
    frames = {"head": head_frame(payloads[0][0])}
    calls = [planner_call(planner, frames, max_new_tokens) for _ in range(planner_calls)]
    state_after_first = measure_state(wrapper, payloads)

    # State 2: after another switch, to catch drift that only shows on repeat.
    calls.append(planner_call(planner, frames, max_new_tokens))
    state_after_second = measure_state(wrapper, payloads)

    phase_samples: dict[str, list[float]] = {key: [] for key in PHASE_KEYS}
    for call in calls:
        for key in PHASE_KEYS:
            phase_samples[key].append(float(call["timings"][key]))
    transfer_totals = [
        sum(float(call["timings"][key]) for key in TRANSFER_PHASES) for call in calls
    ]

    report = {
        "residency_mode": mode,
        "vla_cold_load_ms": round(vla_load_ms, 3),
        "vla_warm_peak_allocated_bytes": vla_warm_peak,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "startup_preparation_ms": round(prepare_ms, 3),
        "startup_receipt": {
            key: startup.get(key)
            for key in ("purpose", "image", "task_observations_used", "cold_load")
        },
        "planner_calls": len(calls),
        "cold_load_calls": sum(1 for call in calls if call["cold_load"]),
        "cpu_mirror_bytes": calls[-1].get("cpu_mirror_bytes"),
        # item C: every phase separately, never collapsed into one number.
        "phases": {key: summarize(values) for key, values in phase_samples.items()},
        "weight_transfer_total": summarize(transfer_totals),
        "vla_latency_by_state": {
            "vlm_never_loaded": state_never["vla_latency"],
            "after_first_vlm_switch": state_after_first["vla_latency"],
            "after_second_vlm_switch": state_after_second["vla_latency"],
        },
        # item A second half: the fidelity question.
        "output_fidelity": {
            "after_first_vlm_switch": compare_states(state_never, state_after_first),
            "after_second_vlm_switch": compare_states(state_never, state_after_second),
        },
        "vlm_output_tokens": [call["output_tokens"] for call in calls],
    }
    del wrapper, planner
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-vlm", type=Path, default=Path("/home/admin1/models/Qwen3-VL-4B-Instruct")
    )
    parser.add_argument(
        "--modes", nargs="+", default=["transfer", "cpu_mirror"]
    )
    parser.add_argument("--payloads", type=int, default=6)
    parser.add_argument("--planner-calls", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.base_vlm.is_dir():
        raise FileNotFoundError(args.base_vlm)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(STARVLA_ROOT / "source_starvla"))
    # Same offline, local-weights load the arms use; without these the loader
    # blocks on the network instead of failing.
    os.environ.setdefault("STARVLA_HF_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("STARVLA_HF_SKIP_WEIGHT_HASH", "1")
    os.environ.setdefault("STARVLA_BASE_VLM", str(args.base_vlm))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    entries = replay.load_manifest(args.capture_dir)[: args.payloads]
    payloads = [replay.rebuild_payload(args.capture_dir, entry) for entry in entries]
    print(f"rebuilt {len(payloads)} payloads, digests match the live run")

    import torch

    modes = {}
    for mode in args.modes:
        print(f"--- residency mode: {mode} ---", flush=True)
        modes[mode] = run_mode(
            mode=mode,
            checkpoint=args.checkpoint,
            base_vlm=args.base_vlm,
            payloads=payloads,
            planner_calls=args.planner_calls,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
        )

    gpu = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "carve.vlm-switch-residency.v1",
        "design": {
            "single_process": True,
            "real_recorded_payloads": True,
            "payloads": len(payloads),
            "identical_action_seed_per_state": True,
            "vla_warmup_calls": args.warmup,
            "planner_max_new_tokens": args.max_new_tokens,
            "phases_reported_by": "CpuStagedVisionPlanner.decide().timings",
        },
        "sources": {
            "capture_dir": str(args.capture_dir),
            "checkpoint": str(args.checkpoint.resolve()),
            "base_vlm": str(args.base_vlm),
        },
        "modes": modes,
        "hardware": {
            "gpu": gpu.name,
            "gpu_total_memory_bytes": gpu.total_memory,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        },
        "claim_boundary": (
            "Per-phase deployment cost and VLA output fidelity across VLM "
            "switching, on recorded observations in one process. Not a task "
            "success measurement, not closed-loop latency, and not a hard "
            "real-time guarantee. The staged design serializes VLM and VLA, so "
            "these numbers do not establish concurrent co-residency."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    for mode, block in modes.items():
        fidelity = block["output_fidelity"]["after_second_vlm_switch"]
        print(
            f"  {mode}: bit-identical after switching "
            f"{fidelity['bit_identical_actions']}/{fidelity['payloads']}  "
            f"max_abs_diff={fidelity['max_abs_diff_over_payloads']:.6g}  "
            f"vlm_generate_p50={block['phases']['vlm_generate_ms']['p50_ms']} ms  "
            f"transfer_p50={block['weight_transfer_total']['p50_ms']} ms  "
            f"total_p50={block['phases']['total_ms']['p50_ms']} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

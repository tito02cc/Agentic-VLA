#!/usr/bin/env python3
"""Profile a real OpenVLA checkpoint through the CARVE deployment path."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_vla.optimization import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    HardwareSpec,
    OptimizationProfile,
    ProfileManifest,
    create_default_runtime,
)
from agentic_vla.runtime import ActionSpec, InferenceControls, InferenceRequest  # noqa: E402
from agentic_vla.runtime.adapters import OpenVlaAdapter  # noqa: E402
from agentic_vla.runtime.policies import HuggingFaceOpenVlaPolicy, OpenVlaLoadConfig  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "checkpoints/openvla-7b-finetuned-libero-10"
DEFAULT_REPLAY = ROOT / "results/carve_semantic_precision/temporal_pairs_t89/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--replay-manifest", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "int8", "nf4"), default="bf16")
    parser.add_argument("--unnorm-key", default="libero_10")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--align-action-token-mask", action="store_true")
    parser.add_argument("--compile-language-model", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prewarm-replay-shapes",
        action="store_true",
        help="Compile one replay sample per distinct prompt-token length before measurement.",
    )
    parser.add_argument("--warmup-calls", type=int, default=2)
    parser.add_argument("--measured-calls", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=10)
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--record-latency-samples", action="store_true")
    parser.add_argument(
        "--reference-manifest",
        type=pathlib.Path,
        default=None,
        help="BF16 manifest used for a predeclared paired action-fidelity gate.",
    )
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    return parser.parse_args()


def _checkpoint_identity(checkpoint: pathlib.Path) -> str:
    index = checkpoint / "model.safetensors.index.json"
    if not index.exists():
        raise SystemExit(f"Missing OpenVLA weight index: {index}")
    payload = json.loads(index.read_text(encoding="utf-8"))
    shard_names = sorted(set(payload["weight_map"].values()))
    digest = hashlib.sha256()
    total_bytes = 0
    for name in shard_names:
        shard = checkpoint / name
        if not shard.exists():
            raise SystemExit(f"Missing OpenVLA weight shard: {shard}")
        stat = shard.stat()
        total_bytes += stat.st_size
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return f"{checkpoint.resolve()}:{len(shard_names)}shards:{total_bytes}bytes:{digest.hexdigest()}"


def _load_replay(path: pathlib.Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    samples: list[dict[str, Any]] = []
    for record in payload["records"]:
        snapshot_id = str(record["snapshot_id"])
        if snapshot_id in seen or record["severity"] != "no_op":
            continue
        image_path = ROOT / str(record["after_image"])
        samples.append(
            {
                "source": str(image_path),
                "image": np.asarray(Image.open(image_path).convert("RGB")),
                "instruction": str(record["instruction"]),
                "episode_id": snapshot_id,
                "timestep": int(payload["snapshot_step"]),
            }
        )
        seen.add(snapshot_id)
        if len(samples) >= limit:
            break
    if not samples:
        raise SystemExit(f"No no-op replay images found in {path}")
    return samples


def _gpu_memory_mib(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    index = device.split(":", 1)[1] if ":" in device else "0"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def compare_openvla_replay_actions(
    reference_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare independent autoregressive actions using predeclared gates."""

    reference = {str(item["episode_id"]): np.asarray(item["action"]) for item in reference_records}
    candidate = {str(item["episode_id"]): np.asarray(item["action"]) for item in candidate_records}
    keys = sorted(set(reference) & set(candidate))
    if not keys:
        raise ValueError("reference and candidate manifests have no paired replay actions")
    reference_actions = np.stack([reference[key] for key in keys]).astype(np.float64)
    candidate_actions = np.stack([candidate[key] for key in keys]).astype(np.float64)
    if reference_actions.shape != candidate_actions.shape or reference_actions.shape[1:] != (7,):
        raise ValueError("paired OpenVLA replay actions must have shape [samples, 7]")
    difference = candidate_actions - reference_actions
    exact = np.isclose(reference_actions, candidate_actions, atol=1e-6, rtol=0.0)
    reference_gripper = reference_actions[:, -1] >= 0.5
    candidate_gripper = candidate_actions[:, -1] >= 0.5
    metrics = {
        "action_exact_rate": float(np.mean(np.all(exact, axis=1))),
        "dimension_exact_rate": float(np.mean(exact)),
        "gripper_decision_agreement": float(np.mean(reference_gripper == candidate_gripper)),
        "action_mae": float(np.mean(np.abs(difference))),
        "action_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "sample_mae_p95": float(np.percentile(np.mean(np.abs(difference), axis=1), 95)),
    }
    thresholds = {
        "minimum_action_exact_rate": 0.9,
        "minimum_gripper_decision_agreement": 1.0,
        "maximum_action_mae": 0.03,
        "maximum_sample_mae_p95": 0.05,
    }
    violations = []
    if metrics["action_exact_rate"] < thresholds["minimum_action_exact_rate"]:
        violations.append("action_exact_rate")
    if metrics["gripper_decision_agreement"] < thresholds["minimum_gripper_decision_agreement"]:
        violations.append("gripper_decision_agreement")
    if metrics["action_mae"] > thresholds["maximum_action_mae"]:
        violations.append("action_mae")
    if metrics["sample_mae_p95"] > thresholds["maximum_sample_mae_p95"]:
        violations.append("sample_mae_p95")
    return {
        "passed": not violations,
        "samples": len(keys),
        "protocol": "paired_independent_autoregressive_actions_v1",
        "thresholds": thresholds,
        "metrics": metrics,
        "violations": violations,
    }


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    replay = _load_replay(args.replay_manifest.expanduser().resolve(), args.max_observations)

    import torch
    import transformers

    baseline_system_mib = _gpu_memory_mib(args.device)
    load_started_s = time.perf_counter()
    policy = HuggingFaceOpenVlaPolicy.from_pretrained(
        OpenVlaLoadConfig(
            checkpoint=checkpoint,
            device=args.device,
            precision=args.precision,
            unnorm_key=args.unnorm_key,
            center_crop=not args.no_center_crop,
            align_action_token_mask=args.align_action_token_mask,
            attn_implementation=args.attn_implementation,
        )
    )
    load_seconds = time.perf_counter() - load_started_s
    loaded_system_mib = _gpu_memory_mib(args.device)

    adapter = OpenVlaAdapter(
        policy,
        adapter_id="openvla-7b-libero10",
        precision=args.precision,
        action_spec=ActionSpec(
            action_dim=7,
            representation="delta_xyz_axis_angle_gripper",
            coordinate_frame="robot_end_effector",
            gripper_convention="rlds_0_close_1_open",
            control_frequency_hz=10.0,
            normalization_id=args.unnorm_key,
        ),
    )
    backend = "torch_compile" if args.compile_language_model else "eager"
    profile_options: dict[str, Any]
    if args.compile_language_model:
        profile_options = {
            "mode": args.compile_mode,
            "fullgraph": False,
            "dynamic": args.compile_dynamic,
            "cudagraphs": False,
        }
    else:
        profile_options = {"behavioral_reference": args.reference_manifest is None}
    profile_id = (
        f"openvla-{args.attn_implementation}-{args.precision}-single-action"
        f"{'-aligned-mask' if args.align_action_token_mask else ''}"
    )
    if args.compile_language_model:
        profile_id += f"-compile-{args.compile_mode}"
        if args.prewarm_replay_shapes:
            profile_id += "-prewarmed-shapes"
    profile = OptimizationProfile(
        profile_id=profile_id,
        backend=backend,
        deployment_precision=args.precision,
        action_horizon=1,
        options=profile_options,
    )
    prepare_started_s = time.perf_counter()
    prepared = create_default_runtime().prepare(adapter, profile)
    prepare_seconds = time.perf_counter() - prepare_started_s

    def request_factory(index: int) -> InferenceRequest:
        sample = replay[index % len(replay)]
        return InferenceRequest(
            observation={"image": sample["image"]},
            instruction=sample["instruction"],
            controls=InferenceControls(deadline_ms=args.deadline_ms),
            episode_id=sample["episode_id"],
            timestep=sample["timestep"],
            metadata={"unnorm_key": args.unnorm_key, "replay_source": sample["source"]},
        )

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    prewarm_records: list[dict[str, Any]] = []
    if args.prewarm_replay_shapes:
        if not args.compile_language_model:
            raise SystemExit("--prewarm-replay-shapes requires --compile-language-model")
        seen_lengths: set[int] = set()
        for index, sample in enumerate(replay):
            processor_inputs = policy._processor(
                policy._prompt(sample["instruction"]),
                policy._prepare_image(sample["image"]),
            )
            token_length = int(processor_inputs["input_ids"].shape[1])
            if token_length in seen_lengths:
                continue
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            prewarm_started_s = time.perf_counter()
            prepared.adapter.infer(request_factory(index))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            prewarm_records.append(
                {
                    "episode_id": sample["episode_id"],
                    "prompt_token_length_before_alignment": token_length,
                    "wall_seconds": time.perf_counter() - prewarm_started_s,
                }
            )
            seen_lengths.add(token_length)
    runner = BenchmarkRunner(
        prepared,
        config=BenchmarkConfig(
            warmup_calls=args.warmup_calls,
            measured_calls=args.measured_calls,
            record_samples=args.record_latency_samples,
        ),
        synchronize=(lambda: torch.cuda.synchronize(args.device)) if args.device.startswith("cuda") else None,
        peak_vram_gb=(
            lambda: torch.cuda.max_memory_allocated(args.device) / 1024**3
        )
        if args.device.startswith("cuda")
        else None,
    )

    benchmark_started_s = time.perf_counter()
    report = runner.run(request_factory)
    benchmark_wall_seconds = time.perf_counter() - benchmark_started_s
    replay_actions = []
    for index, sample in enumerate(replay):
        action = prepared.adapter.infer(request_factory(index)).actions[0]
        replay_actions.append(
            {
                "episode_id": sample["episode_id"],
                "instruction": sample["instruction"],
                "source": sample["source"],
                "action": np.asarray(action, dtype=np.float32).tolist(),
            }
        )

    if args.reference_manifest is None:
        fidelity = {
            "role": "behavioral_reference",
            "samples": len(replay_actions),
            "passed": True,
            "protocol": "paired_independent_autoregressive_actions_v1",
        }
    else:
        reference_manifest = ProfileManifest.load(args.reference_manifest)
        reference_actions = reference_manifest.benchmark.get("replay_actions")
        if not isinstance(reference_actions, list):
            raise SystemExit("Reference manifest does not contain benchmark.replay_actions")
        fidelity = compare_openvla_replay_actions(reference_actions, replay_actions)
        fidelity["reference_manifest"] = str(args.reference_manifest)

    if args.device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(torch.device(args.device))
        accelerator = "cuda"
        device_name = properties.name
        total_memory_gb = properties.total_memory / 1024**3
    else:
        accelerator = "cpu"
        device_name = "cpu"
        total_memory_gb = None
    hardware = HardwareSpec(
        accelerator=accelerator,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        software={
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "transformers": transformers.__version__,
        },
    )
    benchmark = report.to_dict()
    benchmark.update(
        {
            "load_seconds": load_seconds,
            "prepare_seconds": prepare_seconds,
            "profile_prewarm_seconds": float(
                sum(record["wall_seconds"] for record in prewarm_records)
            ),
            "profile_prewarm_records": prewarm_records,
            "benchmark_wall_seconds_including_warmup": benchmark_wall_seconds,
            "system_gpu_memory_mib_before_load": baseline_system_mib,
            "system_gpu_memory_mib_after_load": loaded_system_mib,
            "system_gpu_memory_mib_after_benchmark": _gpu_memory_mib(args.device),
            "replay_manifest": str(args.replay_manifest),
            "replay_samples": len(replay),
            "replay_actions": replay_actions,
            "load_configuration": {
                "precision": args.precision,
                "attention": args.attn_implementation,
                "code_revision": policy.config.code_revision,
                "align_action_token_mask": args.align_action_token_mask,
                "compile_language_model": args.compile_language_model,
                "compile_mode": args.compile_mode if args.compile_language_model else None,
                "compile_dynamic": args.compile_dynamic if args.compile_language_model else None,
                "compile_cudagraphs": False if args.compile_language_model else None,
                "prewarm_replay_shapes": args.prewarm_replay_shapes,
            },
            "libero_preprocessing": {
                "resize": [224, 224],
                "center_crop_area": None if args.no_center_crop else 0.9,
                "prompt_format": "OpenVLA In/Out",
                "environment_gripper_postprocess_applied": False,
                "implementation": "tensorflow_free_pillow_equivalent",
            },
        }
    )
    manifest = ProfileManifest(
        model_id="openvla",
        adapter_id=adapter.adapter_id,
        checkpoint_id=_checkpoint_identity(checkpoint),
        hardware=hardware,
        profile=profile,
        benchmark=benchmark,
        fidelity=fidelity,
        plugin_versions={"openvla": "hf-remote-code-pinned", backend: "builtin"},
    )
    output = manifest.save(args.output_manifest)
    status = "accepted" if fidelity["passed"] else "rejected"
    print(
        json.dumps(
            {
                "manifest": str(output),
                "status": status,
                "benchmark": benchmark,
                "fidelity": fidelity,
            },
            indent=2,
        )
    )
    return 0 if fidelity["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

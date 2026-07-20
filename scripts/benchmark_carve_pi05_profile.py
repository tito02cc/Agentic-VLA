#!/usr/bin/env python3
"""Benchmark one CARVE pi0.5 profile on fixed recorded observations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_vla.optimization import (  # noqa: E402
    ActionFidelityVerifier,
    BenchmarkConfig,
    BenchmarkRunner,
    HardwareSpec,
    OptimizationProfile,
    ProfileManifest,
    create_default_runtime,
)
from agentic_vla.runtime import InferenceControls, InferenceRequest  # noqa: E402
from agentic_vla.runtime.adapters import Pi05Adapter  # noqa: E402


DEFAULT_CHECKPOINT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
DEFAULT_SNAPSHOTS = ROOT / "results/carve_t689_paired_5states_v6/joint/failure_snapshots"


@dataclasses.dataclass(frozen=True)
class ReplayObservation:
    source: pathlib.Path
    instruction: str
    episode_id: str
    timestep: int
    observation: dict[str, np.ndarray]
    noise: np.ndarray | None


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert LIBERO's xyzw quaternion to the pi0.5 axis-angle state format."""

    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = float(np.sqrt(max(0.0, 1.0 - float(quaternion[3]) ** 2)))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return quaternion[:3] * (2.0 * math.acos(float(quaternion[3])) / denominator)


def monitor_proprio_to_policy_state(proprio: np.ndarray) -> np.ndarray:
    """Translate deployable monitor state into the pi0.5 LIBERO action contract."""

    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if proprio.shape == (8,):
        return proprio
    if proprio.shape == (9,):
        return np.concatenate(
            (proprio[:3], quaternion_to_axis_angle(proprio[3:7]), proprio[7:9])
        ).astype(np.float32)
    raise ValueError(f"unsupported replay proprio shape: {proprio.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--norm-stats-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory containing norm_stats.json for base checkpoints",
    )
    parser.add_argument("--snapshot-dir", type=pathlib.Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument(
        "--observation-schema",
        choices=("libero", "droid"),
        default="libero",
        help="Policy input adapter used to encode replay observations",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        choices=("eager", "torch_compile", "torch_compile_masked_views", "torchao_int8"),
        default="eager",
    )
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument(
        "--elide-image-indices",
        default=None,
        help="Legacy comma-separated padded image indices for masked-view elision",
    )
    parser.add_argument(
        "--image-view-order",
        default="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
        help="Comma-separated canonical image-view order",
    )
    parser.add_argument(
        "--elide-image-views",
        default="right_wrist_0_rgb",
        help="Comma-separated view names guaranteed to be padding",
    )
    parser.add_argument(
        "--quantize-groups",
        default="vision_encoder,vlm_backbone",
        help="Comma-separated pi0.5 component groups for torchao_int8",
    )
    parser.add_argument("--inference-steps", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--warmup-calls", type=int, default=2)
    parser.add_argument("--measured-calls", type=int, default=20)
    parser.add_argument(
        "--record-latency-samples",
        action="store_true",
        help="Persist per-call runtime/model latency values in benchmark metadata.",
    )
    parser.add_argument("--max-observations", type=int, default=20)
    parser.add_argument("--fidelity-calls", type=int, default=5)
    parser.add_argument("--deadline-ms", type=float, default=80.0)
    parser.add_argument(
        "--condition-label",
        default="idle",
        help="Auditable deployment condition, for example idle or active_vlm.",
    )
    parser.add_argument(
        "--co-resident-model",
        default=None,
        help="Optional model or service sharing the accelerator during measurement.",
    )
    parser.add_argument("--noise-seed", type=int, default=20260716)
    parser.add_argument("--noise-horizon", type=int, default=None)
    parser.add_argument("--noise-action-dim", type=int, default=None)
    parser.add_argument("--disable-fixed-noise", action="store_true")
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    return parser.parse_args()


def load_replay_observations(args: argparse.Namespace) -> list[ReplayObservation]:
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    array_paths = sorted(snapshot_dir.glob("*.npz"))[: int(args.max_observations)]
    if not array_paths:
        raise SystemExit(f"No failure snapshots found in {snapshot_dir}")
    rng = np.random.default_rng(args.noise_seed)
    observations: list[ReplayObservation] = []
    for array_path in array_paths:
        metadata_path = array_path.with_suffix(".json")
        if not metadata_path.exists():
            raise SystemExit(f"Missing metadata for {array_path}")
        metadata = json.loads(metadata_path.read_text())
        with np.load(array_path, allow_pickle=False) as arrays:
            required = {
                "observation_agentview_image",
                "observation_wrist_image",
                "observation_proprio",
            }
            missing = sorted(required - set(arrays.files))
            if missing:
                raise SystemExit(f"{array_path} is missing replay fields: {', '.join(missing)}")
            base_image = np.asarray(arrays["observation_agentview_image"])
            wrist_image = np.asarray(arrays["observation_wrist_image"])
            policy_state = monitor_proprio_to_policy_state(arrays["observation_proprio"])
            if args.observation_schema == "libero":
                observation = {
                    "observation/image": base_image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": policy_state,
                }
            else:
                observation = {
                    "observation/exterior_image_1_left": base_image,
                    "observation/wrist_image_left": wrist_image,
                    "observation/joint_position": policy_state[:7],
                    "observation/gripper_position": policy_state[7:8],
                }
        noise = None
        if not args.disable_fixed_noise:
            noise = rng.standard_normal(
                (int(args.noise_horizon), int(args.noise_action_dim)), dtype=np.float32
            )
        observations.append(
            ReplayObservation(
                source=array_path,
                instruction=str(metadata["instruction"]),
                episode_id=f"{metadata['task_id']}:{metadata['episode_id']}",
                timestep=int(metadata["timestep"]),
                observation=observation,
                noise=noise,
            )
        )
    return observations


def load_policy(args: argparse.Namespace) -> tuple[Any, Any]:
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.policy_config)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_config_path = checkpoint_dir / "config.json"
    if checkpoint_config_path.exists():
        checkpoint_config = json.loads(checkpoint_config_path.read_text())
        model_fields = {field.name for field in dataclasses.fields(train_config.model)}
        replacements = {
            key: checkpoint_config[key]
            for key in (
                "action_dim",
                "action_horizon",
                "paligemma_variant",
                "action_expert_variant",
            )
            if key in checkpoint_config and key in model_fields
        }
        if "precision" in checkpoint_config and "dtype" in model_fields:
            replacements["dtype"] = checkpoint_config["precision"]
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, **replacements),
        )
    if hasattr(train_config.model, "pytorch_compile_mode"):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
        )
    norm_stats = None
    if args.norm_stats_dir is not None:
        norm_stats_dir = args.norm_stats_dir.expanduser().resolve()
        if not (norm_stats_dir / "norm_stats.json").exists():
            raise SystemExit(f"Cannot find norm_stats.json in {norm_stats_dir}")
        norm_stats = normalize.load(norm_stats_dir)
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": int(args.inference_steps)},
        norm_stats=norm_stats,
        pytorch_device=args.device,
    )
    return policy, train_config.model


def checkpoint_identity(checkpoint_dir: pathlib.Path) -> str:
    weight_path = checkpoint_dir.expanduser().resolve() / "model.safetensors"
    if not weight_path.exists():
        raise SystemExit(f"Cannot find {weight_path}")
    stat = weight_path.stat()
    return f"{weight_path}:{stat.st_size}:{stat.st_mtime_ns}"


def detect_hardware(device: str, backend: str) -> HardwareSpec:
    import torch

    if str(device).startswith("cuda"):
        device_index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(device_index)
        software = {"torch": torch.__version__, "cuda": str(torch.version.cuda)}
        if backend == "torchao_int8":
            import torchao

            software["torchao"] = torchao.__version__
        return HardwareSpec(
            accelerator="cuda",
            device_name=properties.name,
            total_memory_gb=properties.total_memory / 1024**3,
            software=software,
        )
    return HardwareSpec(
        accelerator="cpu",
        device_name="cpu",
        software={"torch": torch.__version__},
    )


def system_gpu_memory_used_mib(device: str) -> float | None:
    """Read total device memory use, including co-resident processes."""

    if not str(device).startswith("cuda"):
        return None
    device_index = str(device).split(":", 1)[1] if ":" in str(device) else "0"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def measure_fidelity(
    reference: Pi05Adapter,
    candidate: Any,
    replay: list[ReplayObservation],
    profile: OptimizationProfile,
    calls: int,
    reference_actions: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    if profile.backend == "eager":
        return {"passed": True, "samples": 0, "role": "behavioral_reference"}
    verifier = ActionFidelityVerifier()
    reports = []
    controls = candidate.apply_profile(InferenceControls())
    for index in range(min(calls, len(replay))):
        sample = replay[index]
        request = InferenceRequest(
            observation=sample.observation,
            instruction=sample.instruction,
            controls=controls,
            episode_id=sample.episode_id,
            timestep=sample.timestep,
            metadata={"noise": sample.noise, "replay_source": str(sample.source)},
        )
        if reference_actions is None:
            reference_value = reference.infer(request).limited(profile.action_horizon).actions
        else:
            reference_value = reference_actions[index]
        candidate_chunk = candidate.adapter.infer(request).limited(profile.action_horizon)
        reports.append(verifier.compare(reference_value, candidate_chunk.actions))
    if not reports:
        raise ValueError("fidelity_calls must select at least one replay observation")
    lower_is_better = {
        "first_action_mae",
        "chunk_mae",
        "chunk_rmse",
        "endpoint_l2",
        "jerk_rmse",
    }
    metric_names = reports[0].metrics.keys()
    metrics = {
        name: (
            max(float(report.metrics[name]) for report in reports)
            if name in lower_is_better
            else min(float(report.metrics[name]) for report in reports)
        )
        for name in metric_names
    }
    return {
        "passed": all(report.passed for report in reports),
        "samples": len(reports),
        "aggregation": "worst_case",
        "metrics": metrics,
        "violations": sorted({value for report in reports for value in report.violations}),
    }


def main() -> int:
    args = parse_args()
    if args.inference_steps <= 0 or args.action_horizon <= 0:
        raise SystemExit("inference steps and action horizon must be positive")
    policy, model_config = load_policy(args)
    if args.noise_horizon is None:
        args.noise_horizon = int(model_config.action_horizon)
    if args.noise_action_dim is None:
        args.noise_action_dim = int(model_config.action_dim)
    replay = load_replay_observations(args)
    adapter = Pi05Adapter(
        policy,
        adapter_id="pi05",
        precision=args.precision,
        max_action_horizon=args.action_horizon,
    )
    backend_options: dict[str, Any] = {}
    module_precisions: dict[str, str] = {}
    if args.backend in {"torch_compile", "torch_compile_masked_views"}:
        backend_options = {
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
        }
        if args.backend == "torch_compile_masked_views":
            if args.elide_image_indices is not None:
                try:
                    elided_indices = [
                        int(value.strip())
                        for value in args.elide_image_indices.split(",")
                        if value.strip()
                    ]
                except ValueError as exc:
                    raise SystemExit("elide-image-indices must contain integers") from exc
                if not elided_indices:
                    raise SystemExit("elide-image-indices must not be empty")
                backend_options["elided_image_indices"] = elided_indices
            else:
                view_order = [
                    value.strip() for value in args.image_view_order.split(",") if value.strip()
                ]
                elided_views = [
                    value.strip() for value in args.elide_image_views.split(",") if value.strip()
                ]
                if not view_order or not elided_views:
                    raise SystemExit(
                        "masked-view backend requires image-view-order and elide-image-views"
                    )
                backend_options["image_view_order"] = view_order
                backend_options["elided_image_views"] = elided_views
        if args.compile_dynamic:
            backend_options["dynamic"] = True
    elif args.backend == "torchao_int8":
        groups = tuple(
            value.strip() for value in args.quantize_groups.split(",") if value.strip()
        )
        if not groups:
            raise SystemExit("torchao_int8 requires at least one quantize group")
        module_precisions = dict.fromkeys(groups, "int8")
        backend_options = {
            "compile_mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
        }
        if args.compile_dynamic:
            backend_options["dynamic"] = True
    group_suffix = ""
    if module_precisions:
        group_suffix = "-" + "+".join(sorted(module_precisions))
    profile = OptimizationProfile(
        profile_id=(
            f"pi05-{args.backend}-{args.precision}-"
            f"{args.inference_steps}step-h{args.action_horizon}{group_suffix}"
        ),
        backend=args.backend,
        deployment_precision=args.precision,
        module_precisions=module_precisions,
        inference_steps=args.inference_steps,
        action_horizon=args.action_horizon,
        options=backend_options,
    )
    optimize_runtime = create_default_runtime()
    reference_actions = None
    if args.backend == "torchao_int8":
        reference_actions = []
        reference_controls = InferenceControls(
            inference_steps=args.inference_steps,
            max_actions=args.action_horizon,
            precision=args.precision,
        )
        for index in range(min(args.fidelity_calls, len(replay))):
            sample = replay[index]
            request = InferenceRequest(
                observation=sample.observation,
                instruction=sample.instruction,
                controls=reference_controls,
                episode_id=sample.episode_id,
                timestep=sample.timestep,
                metadata={"noise": sample.noise, "replay_source": str(sample.source)},
            )
            actions = adapter.infer(request).limited(args.action_horizon).actions
            reference_actions.append(np.asarray(actions).copy())
    prepared = optimize_runtime.prepare(adapter, profile)
    model = optimize_runtime.registry.resolve_model(adapter)
    fidelity = measure_fidelity(
        adapter,
        prepared,
        replay,
        profile,
        args.fidelity_calls,
        reference_actions,
    )
    hardware = detect_hardware(args.device, args.backend)
    if not fidelity["passed"]:
        rejected = ProfileManifest(
            model_id=model.model_id,
            adapter_id=adapter.adapter_id,
            checkpoint_id=checkpoint_identity(args.checkpoint_dir),
            hardware=hardware,
            profile=profile,
            benchmark={
                "status": "rejected_before_benchmark",
                "reason": "action_fidelity_gate",
                "replay_observation_schema": args.observation_schema,
                "backend_metadata": dict(prepared.metadata),
            },
            fidelity=fidelity,
            plugin_versions={"pi05": "builtin", args.backend: "builtin"},
        )
        output = rejected.save(args.output_manifest)
        print(
            json.dumps(
                {"manifest": str(output), "status": "rejected", "fidelity": fidelity},
                indent=2,
            )
        )
        return 2

    import torch

    synchronize = torch.cuda.synchronize if str(args.device).startswith("cuda") else None
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    runner = BenchmarkRunner(
        prepared,
        config=BenchmarkConfig(
            warmup_calls=args.warmup_calls,
            measured_calls=args.measured_calls,
            record_samples=args.record_latency_samples,
        ),
        synchronize=synchronize,
        peak_vram_gb=(
            (lambda: torch.cuda.max_memory_allocated() / 1024**3)
            if str(args.device).startswith("cuda")
            else None
        ),
    )

    def request_factory(index: int) -> InferenceRequest:
        sample = replay[index % len(replay)]
        return InferenceRequest(
            observation=sample.observation,
            instruction=sample.instruction,
            controls=InferenceControls(deadline_ms=args.deadline_ms),
            episode_id=sample.episode_id,
            timestep=sample.timestep,
            metadata={
                "noise": sample.noise,
                "replay_source": str(sample.source),
            },
        )

    system_memory_before_mib = system_gpu_memory_used_mib(args.device)
    report = runner.run(request_factory)
    system_memory_after_mib = system_gpu_memory_used_mib(args.device)
    benchmark = report.to_dict()
    benchmark["replay_observation_schema"] = args.observation_schema
    benchmark["deployment_condition"] = {
        "label": str(args.condition_label),
        "co_resident_model": args.co_resident_model,
        "system_gpu_memory_used_mib_before": system_memory_before_mib,
        "system_gpu_memory_used_mib_after": system_memory_after_mib,
    }
    manifest = ProfileManifest(
        model_id=model.model_id,
        adapter_id=adapter.adapter_id,
        checkpoint_id=checkpoint_identity(args.checkpoint_dir),
        hardware=hardware,
        profile=profile,
        benchmark=benchmark,
        fidelity=fidelity,
        plugin_versions={"pi05": "builtin", args.backend: "builtin"},
    )
    output = manifest.save(args.output_manifest)
    print(json.dumps({"manifest": str(output), "benchmark": benchmark}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

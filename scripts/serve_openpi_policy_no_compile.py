from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import pathlib
import socket
import sys
import time

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from agentic_vla.optimization import (  # noqa: E402
    ContractFallbackPolicy,
    ProfileManifest,
    RuntimeFallbackPolicy,
    create_default_runtime,
    validate_profile_admission,
)
from agentic_vla.runtime import RuntimeControllablePolicy  # noqa: E402
from agentic_vla.runtime.adapters import Pi05Adapter  # noqa: E402


def _disable_torch_compile(train_config):
    model_config = train_config.model
    if hasattr(model_config, "pytorch_compile_mode"):
        model_config = dataclasses.replace(model_config, pytorch_compile_mode=None)
        train_config = dataclasses.replace(train_config, model=model_config)
    return train_config


def _checkpoint_identity(checkpoint_dir: pathlib.Path) -> str:
    weight_path = checkpoint_dir.expanduser().resolve() / "model.safetensors"
    stat = weight_path.stat()
    return f"{weight_path}:{stat.st_size}:{stat.st_mtime_ns}"


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = float(np.sqrt(max(0.0, 1.0 - float(quaternion[3]) ** 2)))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return quaternion[:3] * (2.0 * math.acos(float(quaternion[3])) / denominator)


def _policy_state(proprio: np.ndarray) -> np.ndarray:
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if proprio.shape == (8,):
        return proprio
    if proprio.shape == (9,):
        return np.concatenate(
            (proprio[:3], _quaternion_to_axis_angle(proprio[3:7]), proprio[7:9])
        ).astype(np.float32)
    raise ValueError(f"unsupported warmup proprio shape: {proprio.shape}")


def _warmup_policy(
    policy: RuntimeControllablePolicy,
    snapshot: pathlib.Path,
    *,
    calls: int,
    inference_steps: int,
    action_horizon: int,
    fixed_noise: bool,
) -> None:
    payload, noise = _load_warmup_request(snapshot, action_horizon=action_horizon)
    for index in range(calls):
        started = time.perf_counter()
        request = {
            **payload,
            "runtime_controls": {"inference_steps": inference_steps},
        }
        if fixed_noise:
            request["runtime_noise"] = noise
        policy.infer(request)
        logging.info(
            "CARVE prewarm call %d/%d completed in %.2f s",
            index + 1,
            calls,
            time.perf_counter() - started,
        )


def _load_warmup_request(
    snapshot: pathlib.Path,
    *,
    action_horizon: int,
) -> tuple[dict[str, object], np.ndarray]:
    metadata_path = snapshot.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(snapshot, allow_pickle=False) as arrays:
        observation = {
            "observation/image": np.asarray(arrays["observation_agentview_image"]),
            "observation/wrist_image": np.asarray(arrays["observation_wrist_image"]),
            "observation/state": _policy_state(arrays["observation_proprio"]),
        }
    noise = np.random.default_rng(20260716).standard_normal(
        (action_horizon, 32), dtype=np.float32
    )
    return {**observation, "prompt": str(metadata["instruction"])}, noise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve an OpenPI checkpoint through an optional CARVE deployment profile."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--profile-manifest", type=pathlib.Path, default=None)
    parser.add_argument("--fallback-profile-manifest", type=pathlib.Path, default=None)
    parser.add_argument(
        "--runtime-fallback-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For a research profile, retry explicitly known compile/runtime errors on the "
            "unmodified eager policy and annotate the response. This is not profile promotion."
        ),
    )
    parser.add_argument(
        "--runtime-fallback-profile-id",
        default="pi05-eager-reference",
        help="Trace identifier recorded when --runtime-fallback-on-error is used.",
    )
    parser.add_argument(
        "--require-promoted-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject calibration-only or closed-loop-rejected profile manifests.",
    )
    parser.add_argument("--pytorch-device", default="cuda:0")
    parser.add_argument("--warmup-snapshot", type=pathlib.Path, default=None)
    parser.add_argument("--warmup-calls", type=int, default=0)
    parser.add_argument(
        "--warmup-fixed-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include deterministic runtime noise during prewarm. Deployment "
            "services should keep the default so prewarm matches normal clients."
        ),
    )
    args = parser.parse_args()

    if args.warmup_calls < 0:
        raise ValueError("--warmup-calls must be non-negative")
    if args.warmup_calls and args.warmup_snapshot is None:
        raise ValueError("--warmup-snapshot is required when --warmup-calls is positive")
    if args.runtime_fallback_on_error and args.profile_manifest is None:
        raise ValueError("--runtime-fallback-on-error requires --profile-manifest")

    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Loading policy config=%s dir=%s", args.policy_config, args.policy_dir)
    train_config = _disable_torch_compile(_config.get_config(args.policy_config))
    logging.info("Torch compile mode: %s", getattr(train_config.model, "pytorch_compile_mode", None))
    base_policy = _policy_config.create_trained_policy(
        train_config,
        args.policy_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    deployment_metadata = None
    served_policy = base_policy
    profile_steps = None
    profile_horizon = None
    fallback_policy_for_warmup = None
    if args.profile_manifest is not None:
        manifest = ProfileManifest.load(args.profile_manifest)
        if manifest.model_id != "pi05":
            raise ValueError(f"unsupported manifest model: {manifest.model_id!r}")
        if not bool(manifest.fidelity.get("passed", False)):
            raise ValueError("deployment profile has not passed its fidelity gate")
        admission_decision = validate_profile_admission(manifest)
        if args.require_promoted_profile:
            admission_decision.require_accepted()
        elif not admission_decision.accepted:
            logging.warning(
                "Serving an unpromoted research profile: %s",
                "; ".join(admission_decision.violations),
            )
        actual_checkpoint = _checkpoint_identity(pathlib.Path(args.policy_dir))
        if manifest.checkpoint_id != actual_checkpoint:
            raise ValueError("deployment profile checkpoint identity does not match --policy-dir")
        if args.warmup_calls:
            eager_payload, eager_noise = _load_warmup_request(
                args.warmup_snapshot,
                action_horizon=manifest.profile.action_horizon,
            )
            eager_started = time.perf_counter()
            base_policy.infer(eager_payload, noise=eager_noise)
            logging.info(
                "CARVE eager prime completed in %.2f s before profile preparation",
                time.perf_counter() - eager_started,
            )
        adapter = Pi05Adapter(
            base_policy,
            adapter_id=manifest.adapter_id,
            precision=manifest.profile.deployment_precision or "bf16",
            max_action_horizon=manifest.profile.action_horizon,
        )
        prepared = create_default_runtime().prepare(adapter, manifest.profile)
        served_policy = prepared.adapter.policy
        expected_fallback_id = admission_decision.fallback_profile_id
        if expected_fallback_id:
            if args.fallback_profile_manifest is None:
                raise ValueError(
                    "promoted profile requires --fallback-profile-manifest "
                    f"for {expected_fallback_id!r}"
                )
            fallback_manifest = ProfileManifest.load(args.fallback_profile_manifest)
            fallback_admission = validate_profile_admission(fallback_manifest)
            fallback_admission.require_accepted()
            if fallback_manifest.profile.profile_id != expected_fallback_id:
                raise ValueError("fallback manifest profile_id does not match primary admission")
            if fallback_manifest.model_id != manifest.model_id:
                raise ValueError("fallback manifest model does not match primary manifest")
            if fallback_manifest.adapter_id != manifest.adapter_id:
                raise ValueError("fallback manifest adapter does not match primary manifest")
            if fallback_manifest.checkpoint_id != manifest.checkpoint_id:
                raise ValueError("fallback manifest checkpoint does not match primary manifest")
            if fallback_manifest.hardware.to_dict() != manifest.hardware.to_dict():
                raise ValueError("fallback manifest hardware does not match primary manifest")
            if fallback_manifest.profile.inference_steps != manifest.profile.inference_steps:
                raise ValueError("fallback inference steps do not match primary profile")
            if fallback_manifest.profile.action_horizon != manifest.profile.action_horizon:
                raise ValueError("fallback action horizon does not match primary profile")
            fallback_adapter = Pi05Adapter(
                base_policy,
                adapter_id=fallback_manifest.adapter_id,
                precision=fallback_manifest.profile.deployment_precision or "bf16",
                max_action_horizon=fallback_manifest.profile.action_horizon,
            )
            fallback_prepared = create_default_runtime().prepare(
                fallback_adapter, fallback_manifest.profile
            )
            fallback_policy_for_warmup = fallback_prepared.adapter.policy
            served_policy = ContractFallbackPolicy(
                served_policy,
                fallback_policy_for_warmup,
                fallback_profile_id=expected_fallback_id,
            )
        elif args.fallback_profile_manifest is not None:
            raise ValueError("primary profile does not declare a fallback profile")
        if args.runtime_fallback_on_error:
            served_policy = RuntimeFallbackPolicy(
                served_policy,
                base_policy,
                fallback_profile_id=args.runtime_fallback_profile_id,
                trigger_markers=(
                    "maximum recursion depth exceeded",
                    "Error in function TrampolineAutogradImpl::apply",
                ),
            )
        deployment_metadata = {
            "model_id": prepared.model_id,
            "adapter_id": prepared.adapter.adapter_id,
            "backend_id": prepared.backend_id,
            "profile": prepared.profile.to_dict(),
            "backend_metadata": dict(prepared.metadata),
            "manifest": str(args.profile_manifest.expanduser().resolve()),
            "manifest_created_at": manifest.created_at,
            "admission": admission_decision.to_dict(),
            "fallback_manifest": (
                str(args.fallback_profile_manifest.expanduser().resolve())
                if args.fallback_profile_manifest is not None
                else None
            ),
            "runtime_fallback_on_error": args.runtime_fallback_on_error,
        }
        profile_steps = prepared.profile.inference_steps
        profile_horizon = prepared.profile.action_horizon
        logging.info(
            "Prepared CARVE profile=%s backend=%s",
            prepared.profile.profile_id,
            prepared.backend_id,
        )
    policy_kwargs = {"deployment_metadata": deployment_metadata}
    if profile_steps is not None:
        policy_kwargs.update(
            minimum_inference_steps=profile_steps,
            maximum_inference_steps=profile_steps,
        )
    policy = RuntimeControllablePolicy(served_policy, **policy_kwargs)
    if args.warmup_calls:
        if profile_steps is None or profile_horizon is None:
            raise ValueError("server prewarm requires --profile-manifest")
        _warmup_policy(
            policy,
            args.warmup_snapshot,
            calls=args.warmup_calls,
            inference_steps=profile_steps,
            action_horizon=profile_horizon,
            fixed_noise=args.warmup_fixed_noise,
        )
        if fallback_policy_for_warmup is not None:
            fallback_warmup_policy = RuntimeControllablePolicy(
                fallback_policy_for_warmup,
                minimum_inference_steps=profile_steps,
                maximum_inference_steps=profile_steps,
            )
            _warmup_policy(
                fallback_warmup_policy,
                args.warmup_snapshot,
                calls=args.warmup_calls,
                inference_steps=profile_steps,
                action_horizon=profile_horizon,
                fixed_noise=args.warmup_fixed_noise,
            )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

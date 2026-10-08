"""Reference-only official pi0.5 runtime and explicit RNG reset receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agentic_vla.runtime import CarveRuntime
from agentic_vla.runtime.adapters.pi05 import Pi05Adapter
from agentic_vla.runtime.contracts import ActionSpec, InferenceRequest


RESET_SCHEMA = "agentic_pi05.reference_reset.v1"


def array_fingerprint(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def rng_fingerprint(policy):
    rng = getattr(policy, "_rng", None)
    if rng is None:
        return None
    import jax
    return array_fingerprint(jax.random.key_data(rng))


class ReferenceRuntimePolicy:
    def __init__(self, policy, trace_path=None):
        self.policy = policy
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.calls = 0
        self.generation = 0
        self.runtime = CarveRuntime(Pi05Adapter(
            policy, adapter_id="robodojo_official_pi05", max_action_horizon=50,
            action_spec=ActionSpec(action_dim=14, representation="absolute_joint_position",
                                   coordinate_frame="arx_x5_joint", gripper_convention="0_closed_1_open",
                                   control_frequency_hz=25, normalization_id="official_pi05/arx_x5_sim"),
        ), fallback_mode="strict")

    def infer(self, observation, **kwargs):
        if kwargs:
            raise ValueError("reference deployment does not accept unadmitted runtime controls")
        audit = None
        if self.trace_path is not None:
            audit = {"state": array_fingerprint(observation["state"]),
                     "images": {k: array_fingerprint(v) for k, v in observation["images"].items()},
                     "rng_before": rng_fingerprint(self.policy)}
        start = time.perf_counter()
        chunk = self.runtime.infer(InferenceRequest(
            observation=observation, instruction=observation["prompt"],
            metadata={"raw_payload": observation},
        ))
        actions = np.asarray(chunk.actions)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("native output must be finite [50,14]")
        elapsed = (time.perf_counter() - start) * 1000
        self.calls += 1
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            audit["rng_after"] = rng_fingerprint(self.policy)
            action_file = f"generation_{self.generation:04d}_actions_{self.calls:04d}.npy"
            np.save(self.trace_path.parent / action_file, actions, allow_pickle=False)
            if self.calls == 1:
                input_file = f"generation_{self.generation:04d}_initial_observation.npz"
                np.savez_compressed(self.trace_path.parent / input_file,
                                    state=observation["state"], instruction=observation["prompt"],
                                    **observation["images"])
                audit["input_file"] = input_file
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"call": self.calls, "generation": self.generation,
                                         "inference_wall_ms": elapsed, "input_audit": audit,
                                         "actions_file": action_file,
                                         "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                                         "instruction": observation["prompt"], "optimized": False}) + "\n")
        return {**chunk.raw_output, "actions": actions}


def build_model(model_cfg):
    """Import the vendored official OpenPI implementation only in its policy env."""
    import jax
    from XPolicyLab.policy.Pi_05.model import Model as OfficialModel

    class Model(OfficialModel):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.native_policy = self.policy
            self.policy = ReferenceRuntimePolicy(self.native_policy, cfg.get("inference_trace"))
            self.generation = 0

        def reset(self, reset_context=None):
            context = dict(reset_context or {})
            seed = context.get("policy_seed", 0)
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("policy_seed must be an unsigned 32-bit integer")
            super().reset()
            self.native_policy._rng = jax.random.key(seed)
            self.policy.calls = 0
            self.generation += 1
            self.policy.generation = self.generation
            return {
                "model_reset_schema_version": RESET_SCHEMA,
                "model_module_id": "agentic_vla.benchmarks.robodojo_pi05_policy",
                "generation": self.generation, "policy_seed": seed,
                "observation_cleared": self.observation_window is None,
                "inference_calls": 0,
                "rng_sha256": hashlib.sha256(np.asarray(jax.random.key_data(self.native_policy._rng)).tobytes()).hexdigest(),
            }

    return Model(model_cfg)


def validate_model_reset(receipt, policy_seed):
    if not isinstance(receipt, dict) or receipt.get("model_reset_schema_version") != RESET_SCHEMA:
        raise ValueError("server is not the reset-aware official pi0.5 deployment")
    if (receipt.get("observation_cleared") is not True or receipt.get("inference_calls") != 0
            or receipt.get("policy_seed") != policy_seed
            or type(receipt.get("generation")) is not int or receipt["generation"] < 1
            or len(str(receipt.get("rng_sha256", ""))) != 64):
        raise ValueError("pi0.5 reset receipt is incomplete")
    return receipt

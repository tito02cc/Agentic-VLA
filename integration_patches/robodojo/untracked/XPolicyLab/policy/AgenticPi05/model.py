"""Use the official Pi_05 weights behind the project's reference runtime."""

from agentic_vla.benchmarks.robodojo_pi05_policy import build_model


def Model(model_cfg):
    return build_model(model_cfg)

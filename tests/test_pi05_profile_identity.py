"""Profile identities must distinguish execution-affecting compile options."""

from __future__ import annotations

import argparse

from scripts.benchmark_carve_pi05_profile import build_profile_id


def arguments(**updates):
    values = {
        "backend": "torch_compile",
        "precision": "bf16",
        "inference_steps": 2,
        "action_horizon": 10,
        "compile_cudagraphs": "auto",
        "compile_dynamic": False,
        "compile_fixed_step_loop": False,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_default_profile_id_remains_backward_compatible() -> None:
    assert (
        build_profile_id(arguments())
        == "pi05-torch_compile-bf16-2step-h10"
    )


def test_cudagraph_policy_has_a_distinct_profile_identity() -> None:
    assert build_profile_id(arguments(compile_cudagraphs="off")).endswith(
        "-cg-off"
    )
    assert build_profile_id(arguments(compile_cudagraphs="on")).endswith(
        "-cg-on"
    )


def test_fixed_step_loop_has_a_distinct_profile_identity() -> None:
    assert build_profile_id(arguments(compile_fixed_step_loop=True)).endswith(
        "-fixed-loop"
    )

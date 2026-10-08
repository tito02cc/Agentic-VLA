"""Frozen MuJoCo scene specifications for the representative kitting study."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class KittingSceneSpec:
    scene_id: str
    family: str
    seed: int
    object_xy: Mapping[str, tuple[float, float]]
    description: str
    visual_label_overrides: Mapping[str, tuple[int, int, int]] = field(
        default_factory=dict
    )
    object_yaw_rad: Mapping[str, float] = field(default_factory=dict)
    post_first_subgoal_displacement: tuple[str, float, float] | None = None

    def __post_init__(self) -> None:
        if self.family not in {"K0", "K1", "K2", "K3", "K4"}:
            raise ValueError("unsupported kitting scene family")
        if not self.scene_id.strip() or not self.description.strip():
            raise ValueError("scene_id and description must not be empty")
        required = {"Cereal", "Can", "Milk", "Bread"}
        if set(self.object_xy) != required:
            raise ValueError("scene layout must define all four PickPlace objects")
        if set(self.object_yaw_rad) - required:
            raise ValueError("object_yaw_rad contains an unknown PickPlace object")


def canonical_kitting_scenes(seed: int = 13) -> dict[str, KittingSceneSpec]:
    """Return qualification scenes frozen before method outcomes are observed."""
    common = {
        "Cereal": (0.10, -0.20),
        "Can": (0.15, -0.39),
        "Bread": (-0.10, -0.40),
    }
    red_milk = {"Milk": (240, 60, 55)}
    return {
        "K0": KittingSceneSpec(
            scene_id=f"k0-clear-seed{seed}",
            family="K0",
            seed=seed,
            object_xy={**common, "Milk": (0.25, -0.09)},
            description="Both requested objects are visible with clear destinations.",
            visual_label_overrides=red_milk,
        ),
        "K1": KittingSceneSpec(
            scene_id=f"k1-blocked-approach-seed{seed}",
            family="K1",
            seed=seed,
            object_xy={
                "Cereal": (0.08, -0.20),
                "Can": (0.20, -0.20),
                "Bread": common["Bread"],
                "Milk": (-0.15, -0.05),
            },
            description="Requested Can constrains Cereal and must be sorted first.",
            visual_label_overrides=red_milk,
        ),
        "K2": KittingSceneSpec(
            scene_id=f"k2-occupied-destination-seed{seed}",
            family="K2",
            seed=seed,
            object_xy={
                "Cereal": common["Cereal"],
                "Can": (0.0025, 0.4025),
                "Bread": common["Bread"],
                "Milk": (0.25, -0.09),
            },
            description="Requested Can occupies Cereal's slot and must be sorted first.",
            visual_label_overrides=red_milk,
        ),
        "K3": KittingSceneSpec(
            scene_id=f"k3-unannounced-scene-change-v2-seed{seed}",
            family="K3",
            seed=seed,
            object_xy={
                "Cereal": (0.08, -0.20),
                "Can": (0.20, -0.20),
                "Bread": common["Bread"],
                "Milk": (0.25, -0.09),
            },
            description=(
                "Can constrains Cereal initially; after Can is verified, the "
                "environment moves the remaining Cereal without notifying the agent."
            ),
            visual_label_overrides=red_milk,
            post_first_subgoal_displacement=("Cereal", -0.08, -0.08),
        ),
    }


def sample_kitting_scene(family: str, seed: int) -> KittingSceneSpec:
    """Sample a deterministic reachable layout while preserving task relations."""
    normalized = family.strip().upper()
    if normalized not in {"K0", "K1"}:
        raise ValueError("randomized training scenes currently support K0 and K1")
    rng = np.random.default_rng(seed)

    def jitter(
        xy: tuple[float, float],
        *,
        x_radius: float = 0.02,
        y_radius: float = 0.02,
    ) -> tuple[float, float]:
        return (
            float(xy[0] + rng.uniform(-x_radius, x_radius)),
            float(xy[1] + rng.uniform(-y_radius, y_radius)),
        )

    if normalized == "K0":
        object_xy = {
            "Cereal": jitter((0.10, -0.20)),
            "Can": jitter((0.15, -0.39), x_radius=0.018, y_radius=0.015),
            "Bread": jitter((-0.10, -0.40), x_radius=0.015, y_radius=0.01),
            "Milk": jitter((0.25, -0.09), x_radius=0.012, y_radius=0.015),
        }
        description = "Randomized clear workspace for policy training."
    else:
        cereal_x = float(0.08 + rng.uniform(-0.02, 0.02))
        cereal_y = float(-0.20 + rng.uniform(-0.018, 0.018))
        gap = float(rng.uniform(0.115, 0.122))
        object_xy = {
            "Cereal": (cereal_x, cereal_y),
            "Can": (
                cereal_x + gap,
                cereal_y + float(rng.uniform(-0.004, 0.004)),
            ),
            "Bread": jitter((-0.10, -0.40), x_radius=0.015, y_radius=0.01),
            "Milk": jitter((-0.15, -0.05), x_radius=0.012, y_radius=0.012),
        }
        description = (
            "Randomized dependency scene with Can within Cereal's observed "
            "nearby-risk radius."
        )

    return KittingSceneSpec(
        scene_id=f"{normalized.lower()}-train-randomized-seed{seed}",
        family=normalized,
        seed=seed,
        object_xy=object_xy,
        description=description,
        visual_label_overrides={"Milk": (240, 60, 55)},
        object_yaw_rad={name: 0.0 for name in object_xy},
    )

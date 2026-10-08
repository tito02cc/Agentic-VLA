"""Focused LIBERO-PRO configuration for Agentic mechanism evaluation."""

from __future__ import annotations

import dataclasses
import pathlib

from agentic_vla.runtime.knowledge import AffordanceExperience, HAAExperienceIndex


LIBERO_PRO_MAIN_TASK_IDS = (3, 8, 9)
LIBERO_PRO_SUITES = (
    "libero_10",
    "libero_10_swap",
    "libero_10_object",
    "libero_10_task",
)


@dataclasses.dataclass(frozen=True)
class LiberoProQualificationCell:
    suite: str
    task_ids: tuple[int, ...]
    mechanism: str


def libero_pro_qualification_matrix() -> tuple[LiberoProQualificationCell, ...]:
    """Return the pre-registered low-cost qualification cells."""

    return (
        LiberoProQualificationCell(
            suite="libero_10",
            task_ids=(3, 8, 9),
            mechanism="clean long-horizon control",
        ),
        LiberoProQualificationCell(
            suite="libero_10_swap",
            task_ids=(3, 8, 9),
            mechanism="position-shift recovery and target binding",
        ),
        LiberoProQualificationCell(
            suite="libero_10_object",
            task_ids=(3, 9),
            mechanism="affordance retrieval under object variation",
        ),
        LiberoProQualificationCell(
            suite="libero_10_task",
            task_ids=(3, 8, 9),
            mechanism="semantic planning under changed task logic",
        ),
    )


def build_libero_pro_haa_index() -> HAAExperienceIndex:
    """Build the fixed HAA knowledge base shared by compared Agentic runs."""

    return HAAExperienceIndex(
        (
            AffordanceExperience(
                experience_id="libero-mug-upright-handle",
                object_categories=("mug",),
                affordances=("graspable", "container"),
                grasp_regions=("handle", "side wall"),
                material="ceramic",
                fragility="medium",
                applicable_failures=("misgrasp", "slip", "collision"),
                constraints=(
                    "keep the opening upright during transport",
                    "verify containment before closing an articulated target",
                ),
                outcome="stable side grasp while preserving upright orientation",
                confidence=0.9,
            ),
            AffordanceExperience(
                experience_id="libero-bowl-rim-side",
                object_categories=("bowl",),
                affordances=("graspable", "container"),
                grasp_regions=("outer rim", "side wall"),
                material="ceramic",
                fragility="medium",
                applicable_failures=("misgrasp", "collision", "placement_drift"),
                constraints=(
                    "avoid grasping inside the bowl",
                    "release only after the bowl is supported by the drawer",
                ),
                outcome="stable rim-side grasp and supported release",
                confidence=0.88,
            ),
            AffordanceExperience(
                experience_id="libero-moka-pot-side",
                object_categories=("moka", "moka pot"),
                affordances=("graspable", "upright vessel"),
                grasp_regions=("handle", "body side"),
                material="metal",
                fragility="low",
                applicable_failures=("misgrasp", "slip", "placement_drift"),
                constraints=(
                    "keep the base below the lid",
                    "track which instance has already been placed",
                ),
                outcome="upright transport without repeating a completed target",
                confidence=0.9,
            ),
            AffordanceExperience(
                experience_id="libero-drawer-articulation",
                object_categories=("drawer", "cabinet"),
                affordances=("openable", "closable", "container"),
                grasp_regions=("drawer handle",),
                material="wood",
                fragility="low",
                applicable_failures=("stall", "collision", "recovery_failed"),
                constraints=(
                    "approach the handle before applying drawer-axis motion",
                    "verify the object is contained before closing",
                ),
                outcome="ordered place-then-close execution",
                confidence=0.92,
            ),
            AffordanceExperience(
                experience_id="libero-microwave-articulation",
                object_categories=("microwave",),
                affordances=("openable", "closable", "container"),
                grasp_regions=("door handle",),
                material="metal and glass",
                fragility="medium",
                applicable_failures=("stall", "collision", "recovery_failed"),
                constraints=(
                    "respect the door swing and opening clearance",
                    "verify the target is inside before closing the door",
                ),
                outcome="collision-aware insertion followed by closure",
                confidence=0.92,
            ),
            AffordanceExperience(
                experience_id="libero-stove-placement",
                object_categories=("stove",),
                affordances=("support surface", "switchable appliance"),
                grasp_regions=("control knob",),
                material="metal",
                fragility="low",
                applicable_failures=("placement_drift", "repeated_failure"),
                constraints=(
                    "distinguish placement completion from appliance activation",
                    "preserve the completion state of each placed object",
                ),
                outcome="stage-aware appliance interaction",
                confidence=0.86,
            ),
        )
    )


def validate_libero_pro_root(root: str | pathlib.Path, suite: str) -> None:
    """Fail early if a PRO suite is requested from original LIBERO."""

    if suite == "libero_10":
        return
    if suite not in LIBERO_PRO_SUITES:
        raise ValueError(f"unsupported focused LIBERO-PRO suite: {suite}")
    root_path = pathlib.Path(root)
    registry_path = root_path / "libero" / "libero" / "benchmark" / "__init__.py"
    if not registry_path.is_file():
        raise FileNotFoundError(f"LIBERO benchmark registry not found: {registry_path}")
    registry = registry_path.read_text(encoding="utf-8")
    if f'self.name = "{suite}"' not in registry:
        raise RuntimeError(
            f"{suite} is unavailable under {root_path}; set "
            "AGENTIC_VLA_LIBERO_ROOT to the official LIBERO-PRO checkout"
        )
    data_root = root_path / "libero" / "libero"
    for kind, suffix in (("bddl_files", ".bddl"), ("init_files", ".pruned_init")):
        suite_root = data_root / kind / suite
        files = tuple(suite_root.glob(f"*{suffix}"))
        if len(files) < 10:
            raise RuntimeError(
                f"{suite} requires 10 official {kind} files under {suite_root}; "
                f"found {len(files)}"
            )

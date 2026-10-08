"""Mask consistency diagnostics, never semantic identity or motion authority."""

import math

import numpy as np


def assess_track_masks(roles, raw_masks, visible_masks, *, conflict_fraction=0.5):
    """Keep raw identity conflicts visible after an output exclusivity operation."""
    if (
        not isinstance(roles, tuple) or not 1 <= len(roles) <= 8
        or any(not isinstance(r, str) or not r.strip() or len(r) > 64 for r in roles)
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("one to eight unique bounded role names required")
    if (
        type(conflict_fraction) not in (int, float)
        or not math.isfinite(conflict_fraction) or not 0 < conflict_fraction <= 1
    ):
        raise ValueError("finite conflict fraction in (0,1] required")
    for masks in (raw_masks, visible_masks):
        if (
            not isinstance(masks, np.ndarray) or masks.dtype != np.bool_
            or masks.ndim != 3 or masks.shape[0] != len(roles)
            or min(masks.shape[1:]) < 1 or math.prod(masks.shape[1:]) > 1_048_576
        ):
            raise ValueError("bounded NxHxW boolean masks required")
    if raw_masks.shape != visible_masks.shape:
        raise ValueError("raw and output masks must have the same shape")
    if np.any(visible_masks & ~raw_masks):
        raise ValueError("output masks must not invent foreground pixels")
    areas = raw_masks.sum(axis=(1, 2))
    conflicts = {r: [] for r in roles}
    pairs = []
    for first in range(len(roles)):
        for second in range(first + 1, len(roles)):
            overlap = int(np.count_nonzero(raw_masks[first] & raw_masks[second]))
            minimum_area = int(min(areas[first], areas[second]))
            fraction = overlap / minimum_area if minimum_area > 0 else 0.0
            ambiguous = fraction >= conflict_fraction
            pairs.append({"roles": [roles[first], roles[second]], "raw_overlap_pixels": overlap,
                          "fraction_of_smaller_mask": fraction, "ambiguous": ambiguous})
            if ambiguous:
                conflicts[roles[first]].append(roles[second])
                conflicts[roles[second]].append(roles[first])
    objects = {}
    for index, role in enumerate(roles):
        area = int(visible_masks[index].sum())
        state = "lost" if area == 0 else "ambiguous" if conflicts[role] else "tracking_candidate"
        objects[role] = {"state": state, "raw_area": int(areas[index]), "output_area": area,
                         "conflicts_with": conflicts[role], "requires_reobservation": state != "tracking_candidate"}
    return {"objects": objects, "pairs": pairs,
            "requires_reobservation": any(o["requires_reobservation"] for o in objects.values()),
            "authority": "tracking_diagnostic_only_not_verified_identity"}

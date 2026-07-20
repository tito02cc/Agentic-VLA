#!/usr/bin/env python3
"""Create an auditable deployment manifest from a calibrated CARVE profile."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import pathlib
import sys
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_vla.optimization import (  # noqa: E402
    ProfileManifest,
    StaticMaskedViewContract,
    validate_profile_admission,
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"evidence file does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--closed-loop-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--fallback-profile-id", default=None)
    parser.add_argument("--image-view-order", nargs="+", default=None)
    parser.add_argument("--elided-image-views", nargs="+", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    source = ProfileManifest.load(source_path)
    if source.fidelity.get("passed") is not True:
        raise ValueError("cannot promote a profile that failed replay fidelity")
    if float(source.benchmark.get("deadline_miss_rate", 1.0)) > 0.0:
        raise ValueError("cannot promote a profile with measured deadline misses")
    if (args.image_view_order is None) != (args.elided_image_views is None):
        raise ValueError("named view promotion requires both view-order arguments")
    if args.image_view_order is not None:
        if source.profile.backend != "torch_compile_masked_views":
            raise ValueError("named view promotion is only valid for masked-view profiles")
        contract = StaticMaskedViewContract(
            view_order=tuple(args.image_view_order),
            masked_views=tuple(args.elided_image_views),
        )
        legacy_indices = source.profile.options.get("elided_image_indices")
        if legacy_indices is not None and tuple(legacy_indices) != contract.masked_indices:
            raise ValueError("named view contract does not match calibrated legacy indices")
        profile_options = dict(source.profile.options)
        profile_options.pop("elided_image_indices", None)
        profile_options["image_view_order"] = list(contract.view_order)
        profile_options["elided_image_views"] = list(contract.masked_views)
        source = dataclasses.replace(
            source,
            profile=dataclasses.replace(source.profile, options=profile_options),
        )

    source_receipt = _evidence(source_path)
    closed_loop_receipt = _evidence(args.closed_loop_evidence)
    admission = {
        "status": "promoted",
        "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": args.scope.strip(),
        "fallback_profile_id": args.fallback_profile_id,
        "source_manifest": source_receipt,
        "gates": {
            "replay_fidelity": {"passed": True, "evidence": source_receipt},
            "realtime": {"passed": True, "evidence": source_receipt},
            "closed_loop": {"passed": True, "evidence": closed_loop_receipt},
        },
        "notes": args.notes.strip(),
    }
    promoted = dataclasses.replace(source, admission=admission)
    decision = validate_profile_admission(promoted)
    decision.require_accepted()
    output = promoted.save(args.output_manifest)
    print(f"promoted profile: {output}")
    print(f"profile_id: {promoted.profile.profile_id}")
    print(f"scope: {decision.scope}")
    print(f"fallback_profile_id: {decision.fallback_profile_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

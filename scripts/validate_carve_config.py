#!/usr/bin/env python3
"""Validate a canonical CARVE run config without loading any model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_vla.configuration import CarveRunConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = CarveRunConfig.load(args.config)
    print(
        json.dumps(
            {
                "valid": True,
                "config": str(args.config),
                "fingerprint": config.fingerprint,
                "workspace": str(config.workspace_path),
                "run_manifest": config.to_run_manifest().to_dict(),
                "embedded_planner": config.planner.provider_config() is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

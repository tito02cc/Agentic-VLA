#!/usr/bin/env python3
"""Freeze a read-only backup of the audited episode-reset change set.

The reset contract spans three Git scopes, and in every one of them the relevant
files are either uncommitted or untracked:

* ``third_party/robodojo_official/XPolicyLab`` — detached-HEAD submodule holding
  the wire protocol, the StarVLA adapter and the launcher scripts.
* ``third_party/robodojo_official`` — the evaluation client and the durable
  reset ledger.
* the parent workspace — the finalizer, the run drivers and the test suite,
  which are untracked there.

A submodule reset or a stray checkout would therefore make every RoboDojo result
stop reproducing. This script captures patches, verbatim copies, pristine HEAD
baselines and checksums so the change set is recoverable independently of Git
state. It never writes inside the live trees and never runs ``git add``,
``commit``, ``checkout`` or ``reset``.

The 2026-09-07 backup of the pre-reset-fix integration is left untouched; this
is a parallel, additive snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROBODOJO = REPO / "third_party" / "robodojo_official"
XPOLICYLAB = ROBODOJO / "XPolicyLab"

# Scope -> (repo root, tracked paths to diff, untracked paths to copy verbatim)
SCOPES: dict[str, dict[str, object]] = {
    "xpolicylab": {
        "root": XPOLICYLAB,
        "tracked": [
            "client_server/ws/model_client.py",
            "client_server/ws/model_server.py",
            "client_server/ws/protocol/client.py",
            "policy/starVLA/deploy.py",
            "policy/starVLA/deploy.yml",
            "policy/starVLA/model.py",
            "policy/starVLA/scripts/run_hf_robodojo_env_client.sh",
        ],
        "untracked": [
            "client_server/ws/protocol/reset.py",
            "policy/starVLA/reset_contract.py",
        ],
    },
    "robodojo_official": {
        "root": ROBODOJO,
        "tracked": [
            "src/eval_client/eval_env.py",
            "src/eval_client/main.py",
        ],
        "untracked": [
            "src/eval_client/policy_reset_state.py",
        ],
    },
    # Untracked in the parent workspace, so there is no diff to take: copy them.
    "workspace": {
        "root": REPO,
        "tracked": [],
        "untracked": [
            "scripts/finalize_robodojo_nominal_run.py",
            "scripts/run_robodojo_starvla_nominal.sh",
            "scripts/run_ral_robodojo_matrix.sh",
            "tests/test_robodojo_reset_protocol.py",
            "tests/test_robodojo_policy_reset_ledger.py",
            "tests/test_robodojo_nominal_finalizer.py",
            "tests/test_starvla_robodojo_bridge.py",
        ],
    },
}

# Wire/state contracts the frozen code implements, recorded so a future reader
# can tell at a glance whether an artifact predates a schema change.
SCHEMA_VERSIONS = {
    "reset_receipt": "xpolicylab.episode-reset.v1",
    "model_reset": "carve.policy-reset.v1",
    "state_inventory": "starvla.episode-state.v1",
    "policy_reset_ledger": "robodojo.policy-reset-ledger.v1",
    "policy_reset_audit": "carve.robodojo.policy-reset-audit.v2",
    "run_completeness": "carve.robodojo.run-completeness.v1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def head_state(root: Path) -> dict[str, object]:
    return {
        "path": str(root.relative_to(REPO)) if root != REPO else ".",
        "head_commit": git(root, "rev-parse", "HEAD").strip(),
        "head_subject": git(root, "log", "-1", "--format=%s").strip(),
        "head_date": git(root, "log", "-1", "--format=%ci").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "artifacts/system/carve_audited_reset_20260908",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(
            f"refusing to overwrite an existing backup: {out}\n"
            "Earlier backups are evidence; pick a new --output."
        )
    out.mkdir(parents=True)

    manifest: dict[str, object] = {
        "protocol": "carve.system.audited_reset_backup.v1",
        "purpose": (
            "Read-only protective backup of the audited episode-reset change "
            "set across the XPolicyLab submodule, the RoboDojo evaluation "
            "client and the parent workspace. Nothing in the live trees was "
            "modified; no git add/commit/checkout/reset was run."
        ),
        "supersedes_nothing": (
            "artifacts/system/xpolicylab_carve_uncommitted_20260907 is the "
            "pre-reset-fix snapshot and is left untouched. This directory is "
            "additive."
        ),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_versions": SCHEMA_VERSIONS,
        "scopes": {},
    }

    for scope, spec in SCOPES.items():
        root: Path = spec["root"]  # type: ignore[assignment]
        tracked: list[str] = list(spec["tracked"])  # type: ignore[arg-type]
        untracked: list[str] = list(spec["untracked"])  # type: ignore[arg-type]
        scope_dir = out / scope
        (scope_dir / "files").mkdir(parents=True)

        record: dict[str, object] = {
            "git": head_state(root),
            "tracked_files": tracked,
            "untracked_files": untracked,
            "files": {},
        }

        if tracked:
            for flavour, extra in (("patch", []), ("binary.patch", ["--binary"])):
                text = git(
                    root, "diff", "--unified=10", *extra, "--", *tracked
                )
                (scope_dir / f"{scope}.{flavour}").write_text(text, encoding="utf-8")
            (scope_dir / f"{scope}.diffstat.txt").write_text(
                git(root, "diff", "--stat", "--", *tracked), encoding="utf-8"
            )
            baseline = scope_dir / "head_baseline"
            for rel in tracked:
                destination = baseline / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    git(root, "show", f"HEAD:{rel}"), encoding="utf-8"
                )

        for rel in tracked + untracked:
            source = root / rel
            destination = scope_dir / "files" / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            numstat = ""
            if rel in tracked:
                numstat = git(root, "diff", "--numstat", "--", rel).strip()
            record["files"][rel] = {  # type: ignore[index]
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
                "state": "tracked-modified" if rel in tracked else "untracked-new",
                "numstat": numstat,
            }

        manifest["scopes"][scope] = record  # type: ignore[index]

    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        lines.append(f"{sha256(path)}  ./{path.relative_to(out)}")
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"backup written to {out}")
    for scope, record in manifest["scopes"].items():  # type: ignore[union-attr]
        files = record["files"]  # type: ignore[index]
        print(f"  {scope}: {len(files)} files @ {record['git']['head_commit'][:12]}")
    print(f"  MANIFEST.json  sha256 {sha256(out / 'MANIFEST.json')}")
    print(f"  SHA256SUMS.txt sha256 {sha256(out / 'SHA256SUMS.txt')}")


if __name__ == "__main__":
    main()

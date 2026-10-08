#!/usr/bin/env python3
"""Fetch and hash the official RoboDojo pi0.5 inference checkpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


REPO_ID = "RoboDojo-Benchmark/RoboDojo"
REVISION = "0a709fe7863f6c869a8d141676585a883fce4270"
PREFIX = "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT_ROOT / "third_party/robodojo_official/XPolicyLab/policy/Pi_05"


def is_inference_file(name: str) -> bool:
    prefix = PREFIX + "/"
    if not name.startswith(prefix):
        return False
    relative = name[len(prefix):]
    return relative == "_CHECKPOINT_METADATA" or relative.startswith(("params/", "assets/"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stats(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        stats = json.load(stream)["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = stats[key][field]
            if len(values) != 14:
                raise ValueError(f"{key}.{field}: expected 14 dimensions, got {len(values)}")
    return stats


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    from huggingface_hub.hf_api import RepoFile
    from requests.exceptions import ConnectionError, Timeout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=POLICY_DIR / "checkpoints/huggingface")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--transport", choices=("hub", "aria2"), default="hub")
    args = parser.parse_args()
    destination = args.destination.resolve()
    # List only this checkpoint subtree; the dataset contains many thousands of unrelated files.
    for attempt in range(5):
        try:
            files = sorted(
                (entry for entry in HfApi().list_repo_tree(
                    REPO_ID, path_in_repo=PREFIX, repo_type="dataset", revision=REVISION, recursive=True
                ) if isinstance(entry, RepoFile) and is_inference_file(entry.path)),
                key=lambda entry: entry.path,
            )
            break
        except (ConnectionError, Timeout):
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1}/4: checkpoint file listing", flush=True)
            time.sleep(2 ** attempt)
    required = {PREFIX + suffix for suffix in (
        "/params/_METADATA", "/params/manifest.ocdbt", "/assets/arx_x5_sim/norm_stats.json"
    )}
    if not required.issubset({entry.path for entry in files}):
        raise ValueError("Official checkpoint is missing required inference files")
    total = sum(entry.size for entry in files)
    print(json.dumps({"repo_id": REPO_ID, "revision": REVISION, "files": len(files),
                      "bytes": total, "checkpoint": str(destination / PREFIX)}, indent=2), flush=True)
    if args.list_only:
        return
    def fetch(item):
        index, entry = item
        print(f"[{index}/{len(files)}] {entry.path} ({entry.size} bytes)", flush=True)
        path = destination / entry.path
        expected = entry.lfs.sha256 if entry.lfs is not None else None
        cached = path.is_file() and path.stat().st_size == entry.size
        if cached and expected is not None:
            cached = sha256_file(path) == expected
        elif cached:
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            cached = blob == entry.blob_id
        if not cached and args.transport == "aria2" and entry.size > 64 * 1024 * 1024:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".aria2-download")
            url = hf_hub_url(REPO_ID, entry.path, repo_type="dataset", revision=REVISION)
            env = dict(os.environ)
            for key in ("http_proxy", "https_proxy"):
                if key.upper() in env:
                    env[key] = env[key.upper()]
            subprocess.run([
                "aria2c", "--continue=true", "--allow-overwrite=true", "--auto-file-renaming=false",
                "--max-connection-per-server=8", "--split=8", "--min-split-size=16M",
                "--max-tries=5", "--retry-wait=3", "--connect-timeout=15", "--timeout=60",
                "--file-allocation=none", "--summary-interval=30", "--console-log-level=warn",
                "--show-console-readout=false",
                "--download-result=hide", "--dir=" + str(path.parent), "--out=" + temporary.name, url,
            ], env=env, check=True, timeout=1800)
            if temporary.stat().st_size != entry.size or sha256_file(temporary) != expected:
                raise ValueError(f"Segmented download verification failed: {entry.path}")
            temporary.replace(path)
        elif not cached:
            for attempt in range(5):
                try:
                    path = Path(hf_hub_download(REPO_ID, entry.path, repo_type="dataset", revision=REVISION,
                                                local_dir=destination))
                    break
                except (ConnectionError, Timeout):
                    if attempt == 4:
                        raise
                    print(f"Retry {attempt + 1}/4 after network error: {entry.path}", flush=True)
                    time.sleep(min(2 ** attempt, 20))
        if path.stat().st_size != entry.size:
            raise ValueError(f"Size mismatch: {entry.path}")
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"SHA256 mismatch: {entry.path}")
        print(f"Verified [{index}/{len(files)}] {entry.path}", flush=True)
        return {"path": entry.path, "bytes": entry.size, "sha256": digest,
                "source_lfs_sha256": expected}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(fetch, enumerate(files, 1)))
    validate_stats(destination / PREFIX / "assets/arx_x5_sim/norm_stats.json")
    receipt = {
        "schema_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID, "repo_type": "dataset", "revision": REVISION, "prefix": PREFIX,
        "checkpoint_path": str(destination / PREFIX), "files": records, "total_bytes": total,
        "train_state_downloaded": False, "model_loaded": False, "robot_evaluation_completed": False,
    }
    receipt_path = destination / "pi05_inference_manifest.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    print(f"Verified checkpoint receipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()

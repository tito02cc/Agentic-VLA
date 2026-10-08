"""Real VLM/VLA device-transfer probe; no simulator or robot task score."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/robodojo_official/XPolicyLab/policy/starVLA/source_starvla"
for path in (ROOT, SOURCE):
    sys.path.insert(0, str(path))

from deployment.model_server.policy_wrapper import PolicyServerWrapper

from scripts.benchmark_starvla_ddim import _fixed_examples


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs=2, required=True)
    parser.add_argument("--mode", choices=("transfer", "cpu_mirror"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    sources = [Path(__file__), ROOT / "agentic_vla/optimization/residency.py",
               SOURCE / "deployment/model_server/staged_vlm.py",
               SOURCE / "deployment/model_server/policy_wrapper.py"]
    manifest = {"mode": args.mode, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha(args.checkpoint), "config": config,
                "config_sha256": sha(args.config), "torch": torch.__version__,
                "runtime_env": {k: os.environ.get(k) for k in ("STARVLA_BASE_VLM", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
                "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                "images": [{"path": str(p.resolve()), "sha256": sha(p)} for p in args.images],
                "warmup_calls": 2, "measured_calls": 6, "max_new_tokens": 192,
                "action_seed": 31, "scope": "Recorded RGB for VLM; fixed synthetic VLA action invariance input. No simulator, no robot success measurement."}
    for path in sources:
        target = args.output / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    images = [np.array(Image.open(path).convert("RGB")) for path in args.images]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    wrapper = PolicyServerWrapper(str(args.checkpoint), use_bf16=True,
                                  staged_vlm_path=config["model"], staged_residency=args.mode)
    examples = _fixed_examples()
    for _ in range(2):
        wrapper.predict_action(examples, action_seed=31, num_inference_steps=4)
    before = wrapper.predict_action(examples, action_seed=31, num_inference_steps=4)["actions"]
    np.save(args.output / "actions_before.npy", before)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index in range(8):
        image_index = index % 2
        cpu_rng = torch.random.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state().clone()
        started = time.perf_counter()
        receipt = wrapper.semantic_decision(system_prompt=config["system_prompt"],
            user_prompt=config["user_prompt"], frames={"head": images[image_index]}, max_new_tokens=192)
        elapsed = (time.perf_counter() - started) * 1000
        rng_equal = bool(torch.equal(cpu_rng, torch.random.get_rng_state())
                         and torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        after = wrapper.predict_action(examples, action_seed=31, num_inference_steps=4)["actions"]
        np.save(args.output / f"actions_after_{index}.npy", after)
        row = {"index": index, "image_index": image_index, "warmup": index < 2,
               "elapsed_ms": elapsed, "receipt": receipt, "rng_equal": rng_equal,
               "actions_exactly_equal": bool(np.array_equal(before, after)),
               "action_max_abs_diff": float(np.max(np.abs(before - after))),
               "max_allocated_bytes": torch.cuda.max_memory_allocated(),
               "max_reserved_bytes": torch.cuda.max_memory_reserved(),
               "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        rows.append(row)
        with (args.output / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "receipt"}), flush=True)
        if not rng_equal or not row["actions_exactly_equal"] or not receipt["action_model_restored"]:
            raise RuntimeError("residency invariance failed; no further calls")
    measured = [row for row in rows if not row["warmup"]]
    summary = {"completed": True, "mode": args.mode, "measured_calls": len(measured),
               "mean_ms": float(np.mean([r["elapsed_ms"] for r in measured])),
               "min_ms": min(r["elapsed_ms"] for r in measured),
               "max_ms": max(r["elapsed_ms"] for r in measured),
               "all_actions_exactly_equal": all(r["actions_exactly_equal"] for r in rows),
               "all_rng_equal": all(r["rng_equal"] for r in rows), "metadata": wrapper.metadata,
               "scope": manifest["scope"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

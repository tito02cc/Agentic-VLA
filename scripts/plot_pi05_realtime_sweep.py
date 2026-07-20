#!/usr/bin/env python3
"""Plot pi0.5 realtime inference-step sweeps for the paper assets folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXED = ROOT / "results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_fixed_noise.json"
DEFAULT_ZERO = ROOT / "results/robosuite_stack_pi05_head_plus_300step_realtime_sweep_zero_noise.json"
DEFAULT_OUT_DIR = ROOT / "results/paper_assets_20260609"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-json", default=str(DEFAULT_FIXED))
    parser.add_argument("--zero-json", default=str(DEFAULT_ZERO))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def load_sweep(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["sweep"])


def write_markdown(path: Path, fixed: list[dict[str, Any]], zero: list[dict[str, Any]]) -> None:
    zero_by_step = {int(row["num_inference_steps"]): row for row in zero}
    lines = [
        "# pi0.5 Realtime Sweep",
        "",
        "| Steps | Fixed MSE | Fixed mean infer | Fixed first grip | Zero MSE | Zero mean infer | Zero first grip |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed:
        step = int(row["num_inference_steps"])
        zrow = zero_by_step.get(step)
        if zrow:
            zero_cells = (
                f"`{float(zrow['chunk_mse_mean']):.4f}` | "
                f"`{float(zrow['policy_infer_ms_mean']):.2f}ms` | "
                f"`{float(zrow['first_gripper_sign_accuracy']):.2f}`"
            )
        else:
            zero_cells = "- | - | -"
        lines.append(
            "| "
            f"`{step}` | "
            f"`{float(row['chunk_mse_mean']):.4f}` | "
            f"`{float(row['policy_infer_ms_mean']):.2f}ms` | "
            f"`{float(row['first_gripper_sign_accuracy']):.2f}` | "
            f"{zero_cells} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    fixed = load_sweep(Path(args.fixed_json))
    zero = load_sweep(Path(args.zero_json))
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (out_dir / "pi05_realtime_plot_skipped.txt").write_text(str(exc), encoding="utf-8")
        return 0

    def series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
        return [int(r["num_inference_steps"]) for r in rows], [float(r[key]) for r in rows]

    x_fixed, fixed_mse = series(fixed, "chunk_mse_mean")
    _, fixed_latency = series(fixed, "policy_infer_ms_mean")
    x_zero, zero_mse = series(zero, "chunk_mse_mean")
    _, zero_latency = series(zero, "policy_infer_ms_mean")

    plt.rcParams.update({"font.size": 10})
    fig, ax1 = plt.subplots(figsize=(7.0, 3.8), dpi=180)
    ax2 = ax1.twinx()

    ax1.plot(x_fixed, fixed_mse, marker="o", color="#1f77b4", label="Fixed noise MSE")
    ax1.plot(x_zero, zero_mse, marker="s", color="#17becf", linestyle="--", label="Zero noise MSE")
    ax2.plot(x_fixed, fixed_latency, marker="o", color="#d62728", label="Fixed noise latency")
    ax2.plot(x_zero, zero_latency, marker="s", color="#ff7f0e", linestyle="--", label="Zero noise latency")

    ax1.set_xlabel("Denoising / flow steps")
    ax1.set_ylabel("Open-loop chunk MSE")
    ax2.set_ylabel("Mean policy inference (ms)")
    ax1.set_xticks(sorted(set(x_fixed + x_zero)))
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.set_title("pi0.5 realtime inference-step tradeoff on robosuite Stack")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(fig_dir / "pi05_realtime_sweep.png")
    plt.close(fig)

    write_markdown(out_dir / "table_pi05_realtime_sweep.md", fixed, zero)
    print(fig_dir / "pi05_realtime_sweep.png")
    print(out_dir / "table_pi05_realtime_sweep.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

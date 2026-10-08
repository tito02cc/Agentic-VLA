"""Read existing results and index evidence for the technical report; no inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "E01_libero": "results/libero_pro_full_study_20260825/aggregate/study_summary.json",
    "E02_robomme": "results/robomme_b1_c2_c3_combined_80ep_20260831/summary.json",
    "E03_efficiency": "results/carve_efficiency_full_20260826/summary.json",
    "E04_int8": "results/carve_optimize/pi05_int8_sensitivity_gate.json",
    "E05_memory": "results/cross_task_memory_routing_20260827/aggregate.json",
    "E06_recovery": "artifacts/robodojo/recovery_lifecycle_20260910/admission.json",
    "E07_replay": "artifacts/robodojo/native_trace_b0_20260911/replay_report.json",
    "E08_switch": "artifacts/robodojo/vlm_switch_residency_20260911/vlm_switch_report.json",
    "E08_switch_reverse": "artifacts/robodojo/vlm_switch_residency_20260911/vlm_switch_report_reversed.json",
    "E09_reference": "artifacts/robodojo/reference_memory_20260910/README.md",
    "E10_trigger": "artifacts/robodojo/monitor_calibration_20260911/policy_feasibility.json",
}


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/technical_report_20260911/evidence_index.json",
    )
    args = parser.parse_args()
    index = {}
    data = {}
    for key, relative in SOURCES.items():
        path = ROOT / relative
        raw = path.read_bytes()
        index[key] = {
            "path": relative, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if path.suffix == ".json":
            data[key] = json.loads(raw)

    libero = data["E01_libero"]
    counts = Counter()
    for task in libero["tasks"]:
        for method, value in task["methods"].items():
            counts[method] += value["successes"]
    checks = {
        "libero_task_totals_match_aggregate": all(
            count == libero["aggregate"][method]["successes"]
            for method, count in counts.items()
        ),
    }
    csv_path = ROOT / "results/robomme_b1_c2_c3_combined_80ep_20260831/episodes.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks["robomme_unique_80_pairs"] = (
        len(rows) == 80 and len({(r["task"], r["episode"]) for r in rows}) == 80
    )
    raw_issues = []
    video_paths = []
    source_directories = set()
    conditions = {"B1_raw": "B1_raw", "C2_agentic": "C2_agentic", "C3_full": "C3_full"}
    for condition in conditions:
        checks[f"robomme_{condition}_sum"] = sum(
            row[f"{condition}_success"] == "True" for row in rows
        ) == data["E02_robomme"]["overall"][condition]["successes"]
        for row in rows:
            source = Path(row[f"{condition}_summary"])
            if not source.exists():
                raw_issues.append({"source": str(source), "issue": "missing_summary"})
                continue
            source_directories.add(source.parent.parent.name)
            item = json.loads(source.read_text())
            if bool(item["success"]) != (row[f"{condition}_success"] == "True"):
                raw_issues.append({"source": str(source), "issue": "success_mismatch"})
            for metric in ("planner_calls", "policy_calls", "wall_time_s"):
                if not math.isclose(float(item[metric]), float(row[f"{condition}_{metric}"]), abs_tol=1e-6):
                    raw_issues.append({"source": str(source), "issue": metric + "_mismatch"})
            video = Path(item.get("video_path", ""))
            if str(video).startswith("/workspace/"):
                video = ROOT / video.relative_to("/workspace")
            elif not video.is_absolute():
                video = ROOT / video
            if video.is_file():
                video_paths.append(str(video.relative_to(ROOT)))
            else:
                raw_issues.append({"source": str(source), "issue": "missing_video"})
    checks["robomme_source_records_and_video_presence"] = not raw_issues
    recomputed_pairs = {}
    for left, right in [("B1_raw", "C2_agentic"), ("B1_raw", "C3_full"), ("C2_agentic", "C3_full")]:
        b = sum(r[f"{left}_success"] == "False" and r[f"{right}_success"] == "True" for r in rows)
        c = sum(r[f"{left}_success"] == "True" and r[f"{right}_success"] == "False" for r in rows)
        key = left + "_to_" + right
        recomputed_pairs[key] = {"fail_to_success": b, "success_to_fail": c, "mcnemar_p": exact_mcnemar(b, c)}
        expected = data["E02_robomme"]["pairwise"][key]
        checks[key + "_statistics"] = (
            b == expected["left_fail_to_right_success"]
            and c == expected["left_success_to_right_fail"]
            and math.isclose(exact_mcnemar(b, c), expected["mcnemar_exact_two_sided_p"])
        )

    payload = {
        "schema_version": "agentic-vla.technical-report-evidence.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only source indexing and selected arithmetic checks; no new experiments or video decoding; not a protocol-selection or semantic-correctness audit.",
        "sources": index,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "libero_recomputed_successes": dict(counts),
        "libero_aggregate": libero["aggregate"],
        "robomme_overall": data["E02_robomme"]["overall"],
        "robomme_recomputed_pairwise": recomputed_pairs,
        "robomme_raw_issues": raw_issues,
        "robomme_video_files_present": len(video_paths),
        "robomme_source_run_directories": sorted(source_directories),
        "robomme_protocol_caveat": "Multiple development run directories contribute to the first task block. Arithmetic consistency is not proof of one globally frozen implementation or independent confirmatory testing.",
        "pi05_fidelity_scope": "The backend comparison uses candidate inference controls on the reference as well. A 45/45 threshold pass is not bit identity and not a 7-step versus 2-step fidelity experiment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checks_passed": payload["checks_passed"], "checks": len(checks), "raw_issues": len(raw_issues), "videos_present": len(video_paths)}))
    if not payload["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

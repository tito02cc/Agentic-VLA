import argparse
import json
from pathlib import Path


def _load_json(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _pct(value):
    if value is None:
        return "TBD"
    return f"{100.0 * float(value):.1f}"


def _single_suite_value(data, expected_suite):
    if not data:
        return None
    suite = str(data.get("task_suite", ""))
    if suite != expected_suite:
        return None
    return data.get("overall_success_rate")


def _merged_row(label, object_data=None, goal_data=None, libero10_data=None):
    return (
        f"{label} & {_pct(_single_suite_value(object_data, 'libero_object'))} & "
        f"{_pct(_single_suite_value(goal_data, 'libero_goal'))} & "
        f"{_pct(_single_suite_value(libero10_data, 'libero_10'))} \\\\"
    )


def _single_variant_row(label, data):
    if not data:
        return f"{label} & TBD & TBD & TBD \\\\"
    suite = str(data.get("task_suite", ""))
    value = _pct(data.get("overall_success_rate"))
    object_cell = value if suite == "libero_object" else "TBD"
    goal_cell = value if suite == "libero_goal" else "TBD"
    libero10_cell = value if suite == "libero_10" else "TBD"
    return f"{label} & {object_cell} & {goal_cell} & {libero10_cell} \\\\"


def _mechanism_row(label, data):
    if not data:
        return f"{label} & TBD & TBD & TBD & TBD \\\\"
    mech = data.get("mechanism_summary") or {}
    avg_transitions = mech.get("avg_transitions_per_episode")
    avg_retries = mech.get("avg_retries_per_episode")
    retry_success = mech.get("retry_success_rate_when_used")
    avg_episode_length = mech.get("avg_episode_length")

    def fmt(v, scale=1.0):
        if v is None:
            return "TBD"
        return f"{scale * float(v):.1f}"

    return (
        f"{label} & {fmt(avg_transitions)} & {fmt(avg_retries)} & "
        f"{fmt(retry_success, 100.0)} & {fmt(avg_episode_length)} \\\\"
    )


def main():
    parser = argparse.ArgumentParser(description="Export LaTeX-ready result rows from summary.json files.")
    parser.add_argument("--a1-object", default=None)
    parser.add_argument("--a1-goal", default=None)
    parser.add_argument("--a1-libero10", default=None)
    parser.add_argument("--a2", default=None)
    parser.add_argument("--a3", default=None)
    parser.add_argument("--a4", default=None)
    parser.add_argument("--full", default=None)
    args = parser.parse_args()

    data = {
        "A1-object": _load_json(args.a1_object),
        "A1-goal": _load_json(args.a1_goal),
        "A1-libero10": _load_json(args.a1_libero10),
        "A2": _load_json(args.a2),
        "A3": _load_json(args.a3),
        "A4": _load_json(args.a4),
        "Full": _load_json(args.full),
    }

    print("=== Main Results Rows ===")
    print(_merged_row("A1", data["A1-object"], data["A1-goal"], data["A1-libero10"]))
    print(_single_variant_row("A2", data["A2"]))
    print(_single_variant_row("A3", data["A3"]))
    print(_single_variant_row("A4", data["A4"]))
    print(_single_variant_row("Full", data["Full"]))
    print()
    print("=== Mechanism Rows ===")
    print(_mechanism_row("A2", data["A2"]))
    print(_mechanism_row("A4", data["A4"]))
    print(_mechanism_row("Full", data["Full"]))


if __name__ == "__main__":
    main()

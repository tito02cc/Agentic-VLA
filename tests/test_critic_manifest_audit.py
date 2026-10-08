import json

from PIL import Image

from agentic_vla.toolchain.verifier import build_visual_verification_request
from scripts.audit_robodojo_critic_manifest import load_cases, score_rows

GATE = {"max_false_confirmations": 0, "max_unsupported_decisions": 0,
        "min_visible_correct_fraction": 0.8}


def row(name, reference, status, accepted=True):
    return {"case_id": name, "reference": reference, "status": status,
            "accepted": accepted, "raw_report": {} if accepted else None}


def test_visible_abstention_is_not_correct_negative():
    score = score_rows([row("a", "contradicted", "inconclusive"),
                        row("b", "inconclusive", "inconclusive")], 2, GATE)
    assert score["visible_correct"] == 0
    assert not score["gate_passed"]


def test_false_confirmation_and_unobservable_assertion_are_counted():
    score = score_rows([row("a", "contradicted", "confirmed"),
                        row("b", "inconclusive", "contradicted")], 2, GATE)
    assert score["false_confirmations"] == score["unsupported_decisions"] == 1


def test_parser_failure_is_not_successful_abstention():
    score = score_rows([row("a", "confirmed", "confirmed"),
                        row("b", "inconclusive", "inconclusive", False)], 2, GATE)
    assert score["unknown_valid_abstentions"] == 0
    assert not score["gate_passed"]


def test_missing_or_duplicate_cases_fail():
    rows = [row("a", "confirmed", "confirmed"), row("b", "inconclusive", "inconclusive")]
    assert score_rows(rows, 3, GATE)["complete"] is False
    assert not score_rows(rows + [rows[0]], 3, GATE)["gate_passed"]


def test_complete_correct_cases_pass():
    rows = [row("a", "confirmed", "confirmed"), row("b", "contradicted", "contradicted"),
            row("c", "inconclusive", "inconclusive")]
    assert score_rows(rows, 3, GATE)["gate_passed"]


def test_manifest_does_not_leak_reference_or_rationale(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "fixture.png")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"task_instruction": "Move an object",
        "predicates": {"placed": "The object is on the tray"}, "expected_cases": 1,
        "frames": [{"id": "private_case", "image": "fixture.png", "video": "source.mp4",
            "camera": "current_cam_high", "frame_index": 200,
            "references": {"placed": "contradicted"}, "label_rationale": "SECRET_LABEL"}]}))
    manifest, cases = load_cases(path)
    assert len(cases) == manifest["expected_cases"] == 1
    for case in cases:
        request = build_visual_verification_request(case["context"])
        assert "reference" not in request["user_prompt"]
        assert "label_rationale" not in request["user_prompt"]
        assert "SECRET_LABEL" not in request["user_prompt"]
        assert case["case_id"] not in request["user_prompt"]

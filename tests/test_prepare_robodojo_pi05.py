import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepare_robodojo_pi05", Path(__file__).resolve().parents[1] / "scripts/prepare_robodojo_pi05.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@pytest.mark.parametrize("relative,expected", [
    ("params/_METADATA", True), ("params/ocdbt.process_0/d/chunk", True),
    ("assets/arx_x5_sim/norm_stats.json", True), ("_CHECKPOINT_METADATA", True),
    ("train_state/ocdbt.process_0/d/chunk", False), ("README.md", False),
])
def test_inference_only(relative, expected):
    assert PREPARE.is_inference_file(PREPARE.PREFIX + "/" + relative) is expected


def test_other_checkpoint_rejected():
    assert not PREPARE.is_inference_file(PREPARE.PREFIX.replace("59999", "60000") + "/params/_METADATA")


def test_stats_require_both_14d_arrays(tmp_path):
    path = tmp_path / "norm_stats.json"
    stats = {key: {field: [0.0] * 14 for field in ("mean", "std", "q01", "q99")}
             for key in ("state", "actions")}
    path.write_text(json.dumps({"norm_stats": stats}))
    assert PREPARE.validate_stats(path) == stats
    stats["actions"]["q99"] = [0.0] * 7
    path.write_text(json.dumps({"norm_stats": stats}))
    with pytest.raises(ValueError, match="14 dimensions"):
        PREPARE.validate_stats(path)


def test_sha256(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(b"abc")
    assert PREPARE.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

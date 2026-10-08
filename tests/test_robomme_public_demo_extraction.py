from argparse import Namespace
import sys
import types

import numpy as np

from scripts import extract_robomme_public_demo as extraction


def test_single_initial_frame_has_no_public_demo_video(tmp_path, monkeypatch):
    class FakeEnv:
        def reset(self):
            return {}, {"task_goal": ["pick the highlighted cube"]}

        def close(self):
            pass

    class FakeBuilder:
        def __init__(self, **kwargs):
            pass

        def get_episode_num(self):
            return 50

        def make_env_for_episode(self, episode):
            assert episode == 22
            return FakeEnv()

    robomme = types.ModuleType("robomme")
    robomme.__path__ = []
    wrapper = types.ModuleType("robomme.env_record_wrapper")
    wrapper.BenchmarkEnvBuilder = FakeBuilder
    monkeypatch.setitem(sys.modules, "robomme", robomme)
    monkeypatch.setitem(sys.modules, "robomme.env_record_wrapper", wrapper)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(extraction, "initial_history", lambda observation: ([frame], [frame], [np.zeros(3)]))
    args = Namespace(task="PickHighlight", output_root=tmp_path, max_steps=1300, demo_history_mode="official")

    summary = extraction.extract_episode(args, 22)

    assert summary["initial_memory_frames"] == 1
    assert summary["initial_demo_sha256"] is None
    assert (tmp_path / "PickHighlight_ep22" / "summary.json").is_file()
    assert not (tmp_path / "PickHighlight_ep22" / "initial_demo_front.mp4").exists()

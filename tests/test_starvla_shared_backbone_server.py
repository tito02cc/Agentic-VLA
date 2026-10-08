from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SOURCE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "robodojo_official"
    / "XPolicyLab"
    / "policy"
    / "starVLA"
    / "source_starvla"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deployment.model_server.policy_wrapper import PolicyServerWrapper  # noqa: E402
from deployment.model_server.tools.websocket_policy_server import (  # noqa: E402
    WebsocketPolicyServer,
)


class _FakeProcessor:
    def batch_decode(self, tokens, **_kwargs):
        assert tuple(tokens.shape) == (1, 2)
        return ['{"intent":"continue"}']


class _FakeInterface:
    def __init__(self):
        self.processor = _FakeProcessor()
        self.model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=151672))
        )
        self.observed_images = None
        self.observed_instructions = None
        self.observed_generate = None

    def build_qwenvl_inputs(self, *, images, instructions):
        self.observed_images = images
        self.observed_instructions = instructions
        return {
            "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def generate(self, **kwargs):
        self.observed_generate = kwargs
        return torch.tensor([[11, 12, 13, 21, 22]], dtype=torch.long)


class _IdentityActionProcessor:
    def unapply_actions(self, actions):
        return actions


class _RandomActionFramework:
    def predict_action(self, *, examples, **_kwargs):
        batch_size = len(examples)
        return {"normalized_actions": torch.randn(batch_size, 4, 2).numpy()}


class SharedBackboneServerTest(unittest.TestCase):
    def test_request_level_action_seed_is_reproducible_and_rng_isolated(self):
        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._framework = _RandomActionFramework()
        wrapper._default_unnorm_key = "test"
        wrapper._available_unnorm_keys = ["test"]
        wrapper._get_processor = lambda _key: _IdentityActionProcessor()
        wrapper._normalize_example_states = lambda examples, _proc: examples

        torch.manual_seed(1234)
        expected_next = torch.randn(1)
        torch.manual_seed(1234)
        first = wrapper.predict_action([{}], action_seed=7)["actions"]
        actual_next = torch.randn(1)
        second = wrapper.predict_action([{}], action_seed=7)["actions"]
        different = wrapper.predict_action([{}], action_seed=8)["actions"]

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertTrue(torch.equal(expected_next, actual_next))

    def test_wrapper_generates_with_resident_interface(self):
        interface = _FakeInterface()
        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._framework = SimpleNamespace(qwen_vl_interface=interface)
        wrapper._framework_name = "QwenPI_v3"

        result = wrapper.semantic_decision(
            system_prompt="Return JSON.",
            user_prompt="Inspect task progress.",
            frames={"head": np.zeros((8, 8, 3), dtype=np.uint8)},
            max_new_tokens=32,
        )

        self.assertEqual(result["text"], '{"intent":"continue"}')
        self.assertEqual(result["input_tokens"], 3)
        self.assertEqual(result["output_tokens"], 2)
        self.assertTrue(result["shared_backbone"])
        self.assertEqual(len(interface.observed_images[0]), 1)
        self.assertFalse(interface.observed_generate["do_sample"])
        self.assertFalse(interface.observed_generate["use_cache"])
        self.assertEqual(interface.observed_generate["max_new_tokens"], 32)
        self.assertEqual(
            interface.observed_generate["suppress_tokens"],
            [151669, 151670, 151671],
        )

    def test_websocket_router_exposes_semantic_request(self):
        policy = SimpleNamespace(
            semantic_decision=lambda **_payload: {
                "text": '{"intent":"continue"}',
                "shared_backbone": True,
            }
        )
        server = WebsocketPolicyServer.__new__(WebsocketPolicyServer)
        server._policy = policy

        response = server._route_message(
            {
                "type": "semantic_decision",
                "request_id": "semantic-test",
                "payload": {
                    "system_prompt": "Return JSON.",
                    "user_prompt": "Inspect task progress.",
                },
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["type"], "semantic_decision_result")
        self.assertEqual(response["data"]["text"], '{"intent":"continue"}')


if __name__ == "__main__":
    unittest.main()

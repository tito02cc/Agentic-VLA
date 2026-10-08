"""CPU-only tests for CARVE Optimize Runtime contracts and reference backend."""

from __future__ import annotations

import dataclasses
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from agentic_vla.optimization import (
    ActionFidelityVerifier,
    BenchmarkConfig,
    BenchmarkRunner,
    CoResidentBenchmarkReport,
    ContractFallbackPolicy,
    EagerBackend,
    FidelityThresholds,
    HardwareSpec,
    LingBotVlaModelPlugin,
    MaskedViewElisionBackend,
    OptimizationProfile,
    OpenVlaModelPlugin,
    Pi05ModelPlugin,
    PlannerAdmissionRequirements,
    PlannerBenchmarkReport,
    PlannerOptimizationProfile,
    ProfileManifest,
    RuntimeFallbackPolicy,
    StaticMaskedViewContract,
    SystemAdmissionRequirements,
    SystemOptimizationProfile,
    TorchAOInt8Backend,
    TorchAOInt8MaskedViewBackend,
    TorchCompileBackend,
    create_default_registry,
    create_default_runtime,
    validate_planner_profile_admission,
    validate_profile_admission,
    validate_system_profile_admission,
)
from agentic_vla.runtime import InferenceControls, InferenceRequest
from agentic_vla.runtime.adapters import LingBotVlaAdapter, OpenVlaAdapter, Pi05Adapter
from agentic_vla.runtime.policies import (
    OpenVlaLoadConfig,
    align_openvla_action_token_mask,
    center_crop_openvla_image,
    postprocess_openvla_libero_action,
)
from scripts.benchmark_carve_pi05_profile import (
    monitor_proprio_to_policy_state,
    system_gpu_memory_used_mib,
)
from scripts.legacy.benchmark_carve_openvla_profile import (
    _gpu_memory_mib,
    compare_openvla_replay_actions,
)
from agentic_vla.optimization.backends.masked_views import elide_masked_views


class FakePi05Policy:
    def __init__(self) -> None:
        self._sample_kwargs = {"num_steps": 7}
        self.steps: list[int] = []

    def infer(self, payload: dict, **kwargs):
        del payload, kwargs
        self.steps.append(int(self._sample_kwargs["num_steps"]))
        return {
            "actions": np.zeros((6, 7), dtype=np.float32),
            "policy_timing": {"infer_ms": 3.5},
        }


class FakeOpenVlaPolicy:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def infer(self, payload: dict):
        self.payloads.append(dict(payload))
        return {
            "actions": np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            "policy_timing": {"infer_ms": 8.0},
        }


class FakeOpenVlaTorchPolicy(FakeOpenVlaPolicy):
    def __init__(self) -> None:
        super().__init__()
        self._model = mock.Mock()
        self._model.sample_actions = None
        self._model.language_model = mock.Mock()
        self._model._modules = {"language_model": self._model.language_model}


class PlannerProfileAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = PlannerOptimizationProfile(
            profile_id="qwen35-4b-int4-json-v1",
            model_id="Qwen3.5-4B",
            backend="vllm",
            deployment_precision="INT4",
            prompt_schema_version="carve-agent-v1",
            max_tokens=128,
            timeout_s=10.0,
            module_precisions={"language_decoder": "NF4"},
            preserved_modules=("vision_tower", "decision_head"),
        )

    def test_planner_profile_round_trip_normalizes_backend_and_precision(self) -> None:
        restored = PlannerOptimizationProfile.from_dict(self.profile.to_dict())

        self.assertEqual(restored, self.profile)
        self.assertEqual(restored.backend, "vllm")
        self.assertEqual(restored.deployment_precision, "int4")
        self.assertEqual(restored.module_precisions, {"language_decoder": "nf4"})
        self.assertEqual(restored.preserved_modules, ("vision_tower", "decision_head"))

    def test_planner_profile_admission_requires_semantics_and_vla_timing(self) -> None:
        report = PlannerBenchmarkReport(
            samples=30,
            schema_valid_rate=1.0,
            intent_valid_rate=1.0,
            reference_agreement_rate=0.97,
            latency_p50_ms=550.0,
            latency_p95_ms=720.0,
            timeout_rate=0.0,
            co_resident_vla_p95_ms=74.0,
            co_resident_vla_deadline_miss_rate=0.005,
            unsafe_intervention_rate=0.0,
            fail_closed_rate=1.0,
            peak_vram_gb=4.5,
        )

        decision = validate_planner_profile_admission(self.profile, report)

        self.assertTrue(decision.accepted)
        decision.require_accepted()

    def test_planner_profile_rejects_fast_but_semantically_divergent_candidate(self) -> None:
        report = PlannerBenchmarkReport(
            samples=30,
            schema_valid_rate=1.0,
            intent_valid_rate=1.0,
            reference_agreement_rate=0.70,
            latency_p50_ms=100.0,
            latency_p95_ms=150.0,
            timeout_rate=0.0,
            co_resident_vla_p95_ms=70.0,
            co_resident_vla_deadline_miss_rate=0.0,
            unsafe_intervention_rate=0.1,
            fail_closed_rate=0.9,
            peak_vram_gb=3.0,
        )

        decision = validate_planner_profile_admission(
            self.profile,
            report,
            PlannerAdmissionRequirements(minimum_samples=20),
        )

        self.assertFalse(decision.accepted)
        self.assertTrue(
            any("reference_agreement_rate" in item for item in decision.violations)
        )
        self.assertTrue(
            any("unsafe_intervention_rate" in item for item in decision.violations)
        )


class SystemProfileAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = SystemOptimizationProfile(
            profile_id="qwen9b-vpnf4+pi05-smve-4090",
            planner_profile_id="qwen9b-vpnf4",
            vla_profile_id="pi05-smve-bf16",
            scheduler_policy="safe_boundary_serial",
            memory_budget_gb=23.0,
            fallback_planner_profile_id="qwen4b-bf16",
            fallback_vla_profile_id="pi05-compiled-bf16",
        )
        self.report = CoResidentBenchmarkReport(
            samples=20,
            peak_vram_gb=21.5,
            planner_latency_p95_ms=700.0,
            vla_latency_p95_ms=72.0,
            vla_deadline_miss_rate=0.0,
            planner_timeout_rate=0.0,
            unsafe_intervention_rate=0.0,
        )

    def test_system_profile_round_trip_and_admission(self) -> None:
        restored = SystemOptimizationProfile.from_dict(self.profile.to_dict())
        self.assertEqual(restored, self.profile)

        decision = validate_system_profile_admission(
            restored,
            self.report,
            planner_admitted=True,
            vla_admitted=True,
        )
        self.assertTrue(decision.accepted)
        decision.require_accepted()

    def test_system_profile_rejects_unadmitted_or_over_budget_pair(self) -> None:
        report = dataclasses.replace(self.report, peak_vram_gb=23.5)
        decision = validate_system_profile_admission(
            self.profile,
            report,
            planner_admitted=False,
            vla_admitted=True,
            requirements=SystemAdmissionRequirements(maximum_peak_vram_gb=23.0),
        )
        self.assertFalse(decision.accepted)
        self.assertTrue(any("Planner profile" in item for item in decision.violations))
        self.assertTrue(any("peak_vram_gb" in item for item in decision.violations))


class OpenVlaPolicyContractTests(unittest.TestCase):
    def test_center_crop_is_deterministic_and_224_square(self) -> None:
        image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        first = np.asarray(center_crop_openvla_image(image))
        second = np.asarray(center_crop_openvla_image(image))
        self.assertEqual(first.shape, (224, 224, 3))
        np.testing.assert_array_equal(first, second)

    def test_libero_gripper_postprocess_matches_official_order(self) -> None:
        closed = postprocess_openvla_libero_action(np.array([0, 0, 0, 0, 0, 0, 0.1]))
        opened = postprocess_openvla_libero_action(np.array([0, 0, 0, 0, 0, 0, 0.9]))
        self.assertEqual(float(closed[-1]), 1.0)
        self.assertEqual(float(opened[-1]), -1.0)

    def test_load_config_rejects_unknown_precision(self) -> None:
        with self.assertRaisesRegex(ValueError, "bf16, int8, or nf4"):
            OpenVlaLoadConfig(checkpoint="unused", precision="fp8")

    def test_action_token_alignment_extends_ids_and_mask_once(self) -> None:
        import torch

        inputs = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }
        self.assertTrue(align_openvla_action_token_mask(inputs, torch_module=torch))
        self.assertEqual(inputs["input_ids"].tolist(), [[1, 2, 29871]])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertFalse(align_openvla_action_token_mask(inputs, torch_module=torch))
        self.assertEqual(inputs["input_ids"].shape, inputs["attention_mask"].shape)

    def test_openvla_independent_action_fidelity_gate(self) -> None:
        reference = [
            {"episode_id": "a", "action": [0.1, 0, 0, 0, 0, 0, 1]},
            {"episode_id": "b", "action": [0.0, 0, 0, 0, 0, 0, 0]},
        ]
        exact = compare_openvla_replay_actions(reference, reference)
        changed = compare_openvla_replay_actions(
            reference,
            [
                {"episode_id": "a", "action": [0.1, 0, 0, 0, 0, 0, 0]},
                {"episode_id": "b", "action": [0.2, 0, 0, 0, 0, 0, 0]},
            ],
        )
        self.assertTrue(exact["passed"])
        self.assertEqual(exact["metrics"]["action_exact_rate"], 1.0)
        self.assertFalse(changed["passed"])
        self.assertIn("gripper_decision_agreement", changed["violations"])

    def test_openvla_profiler_reads_system_gpu_memory(self) -> None:
        with mock.patch(
            "scripts.legacy.benchmark_carve_openvla_profile.subprocess.check_output",
            return_value="1234\n",
        ) as check_output:
            self.assertEqual(_gpu_memory_mib("cuda:1"), 1234.0)
        self.assertIn("--id=1", check_output.call_args.args[0])
        self.assertIsNone(_gpu_memory_mib("cpu"))


class CarveOptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = FakePi05Policy()
        self.adapter = Pi05Adapter(self.policy, precision="bf16")

    def test_default_registry_resolves_pi05_and_eager_backend(self) -> None:
        registry = create_default_registry()

        model = registry.resolve_model(self.adapter)
        backend = registry.get_backend("EAGER")

        self.assertIsInstance(model, Pi05ModelPlugin)
        self.assertIsInstance(backend, EagerBackend)
        self.assertEqual(
            registry.model_ids,
            ("lingbot-vla", "openvla", "pi05", "starvla"),
        )
        self.assertEqual(
            registry.backend_ids,
            (
                "eager",
                "torch_compile",
                "torch_compile_masked_views",
                "torchao_int8",
                "torchao_int8_masked_views",
            ),
        )

    def test_openvla_adapter_and_plugin_preserve_autoregressive_contract(self) -> None:
        policy = FakeOpenVlaPolicy()
        adapter = OpenVlaAdapter(policy, precision="bf16")
        registry = create_default_registry()
        model = registry.resolve_model(adapter)

        chunk = adapter.infer(
            InferenceRequest(
                observation={"image": np.zeros((16, 16, 3), dtype=np.uint8)},
                instruction="pick up the mug",
                metadata={"unnorm_key": "libero_10"},
            )
        )

        self.assertIsInstance(model, OpenVlaModelPlugin)
        self.assertTrue(adapter.capabilities.autoregressive_decoding)
        self.assertFalse(adapter.capabilities.action_chunking)
        self.assertFalse(adapter.capabilities.configurable_inference_steps)
        self.assertEqual(chunk.actions.shape, (1, 7))
        self.assertEqual(chunk.model_latency_ms, 8.0)
        self.assertEqual(policy.payloads[0]["unnorm_key"], "libero_10")

    def test_lingbot_profile_uses_server_level_flow_and_compile_options(self) -> None:
        class FakeLingBotClient:
            def call(self, **kwargs):
                del kwargs
                return None

        adapter = LingBotVlaAdapter(FakeLingBotClient())
        model = create_default_registry().resolve_model(adapter)
        self.assertIsInstance(model, LingBotVlaModelPlugin)
        profile = OptimizationProfile(
            profile_id="lingbot-bf16-compile-10step-h10",
            backend="eager",
            deployment_precision="bf16",
            action_horizon=10,
            options={
                "server_use_compile": True,
                "server_num_denoising_steps": 10,
                "server_use_length": 50,
                "behavioral_equivalence": True,
            },
        )
        model.validate_profile(adapter, profile)

        with self.assertRaisesRegex(ValueError, "policy server"):
            model.validate_profile(
                adapter,
                OptimizationProfile(
                    profile_id="lingbot-invalid-request-steps",
                    inference_steps=5,
                ),
            )

    def test_openvla_profile_rejects_flow_steps_and_action_chunks(self) -> None:
        adapter = OpenVlaAdapter(FakeOpenVlaPolicy())
        model = OpenVlaModelPlugin()
        with self.assertRaisesRegex(ValueError, "inference steps"):
            model.validate_profile(
                adapter,
                OptimizationProfile(
                    profile_id="openvla-invalid-steps",
                    inference_steps=2,
                ),
            )
        with self.assertRaisesRegex(ValueError, "one action"):
            model.validate_profile(
                adapter,
                OptimizationProfile(
                    profile_id="openvla-invalid-horizon",
                    action_horizon=10,
                ),
            )

    def test_eager_marks_prequantized_openvla_as_candidate(self) -> None:
        adapter = OpenVlaAdapter(FakeOpenVlaPolicy(), precision="int8")
        prepared = EagerBackend().prepare(
            adapter,
            OpenVlaModelPlugin(),
            OptimizationProfile(
                profile_id="openvla-int8",
                backend="eager",
                deployment_precision="int8",
                action_horizon=1,
            ),
        )
        self.assertFalse(prepared.metadata["transformed"])
        self.assertFalse(prepared.metadata["behavioral_reference"])
        self.assertEqual(prepared.metadata["predeployed_precision"], "int8")

    def test_torch_compile_targets_openvla_language_model(self) -> None:
        policy = FakeOpenVlaTorchPolicy()
        adapter = OpenVlaAdapter(policy, precision="bf16")
        compiled = object()
        compile_fn = mock.Mock(return_value=compiled)
        prepared = TorchCompileBackend(compile_fn=compile_fn).prepare(
            adapter,
            OpenVlaModelPlugin(),
            OptimizationProfile(
                profile_id="openvla-compile",
                backend="torch_compile",
                deployment_precision="bf16",
                action_horizon=1,
                options={"mode": "default", "dynamic": True, "cudagraphs": False},
            ),
        )
        self.assertIs(prepared.adapter.policy._model.language_model, compiled)
        self.assertIs(policy._model.language_model, policy._model._modules["language_model"])
        self.assertEqual(prepared.metadata["compile_target"], "language_model")

    def test_libero_monitor_proprio_is_converted_to_pi05_state(self) -> None:
        monitor_state = np.asarray(
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, -0.02, 0.02],
            dtype=np.float32,
        )

        policy_state = monitor_proprio_to_policy_state(monitor_state)

        self.assertEqual(policy_state.shape, (8,))
        np.testing.assert_array_equal(policy_state[3:6], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(policy_state[-2:], monitor_state[-2:])

    def test_eager_profile_is_zero_transform_and_applies_runtime_controls(self) -> None:
        profile = OptimizationProfile(
            profile_id="pi05-bf16-reference",
            backend="eager",
            deployment_precision="bf16",
            inference_steps=2,
            action_horizon=3,
        )
        prepared = create_default_runtime().prepare(self.adapter, profile)

        controls = prepared.apply_profile(InferenceControls(deadline_ms=80.0))

        self.assertIs(prepared.adapter, self.adapter)
        self.assertFalse(prepared.metadata["transformed"])
        self.assertEqual(controls.inference_steps, 2)
        self.assertEqual(controls.max_actions, 3)
        self.assertEqual(controls.precision, "bf16")
        self.assertEqual(controls.deadline_ms, 80.0)

    def test_eager_rejects_unapplied_module_quantization(self) -> None:
        profile = OptimizationProfile(
            profile_id="invalid-eager-int8",
            backend="eager",
            deployment_precision="bf16",
            module_precisions={"vlm_backbone": "int8"},
        )

        with self.assertRaisesRegex(ValueError, "module-specific"):
            EagerBackend().prepare(self.adapter, Pi05ModelPlugin(), profile)

    def test_torch_compile_copies_policy_and_preserves_source_adapter(self) -> None:
        class FakeTorchModel:
            def sample_actions(self, *args, **kwargs):
                del args, kwargs
                return np.zeros((1, 6, 7), dtype=np.float32)

        self.policy._model = FakeTorchModel()
        compile_calls: list[dict] = []

        def fake_compile(function, **kwargs):
            compile_calls.append(kwargs)
            return function

        profile = OptimizationProfile(
            profile_id="pi05-compile",
            backend="torch_compile",
            deployment_precision="bf16",
            inference_steps=2,
            action_horizon=3,
            options={"mode": "reduce-overhead", "fullgraph": False},
        )
        prepared = TorchCompileBackend(compile_fn=fake_compile).prepare(
            self.adapter,
            Pi05ModelPlugin(),
            profile,
        )

        self.assertIsNot(prepared.adapter, self.adapter)
        self.assertIs(self.adapter.policy, self.policy)
        self.assertEqual(
            compile_calls,
            [{"mode": "reduce-overhead", "fullgraph": False}],
        )
        self.assertTrue(prepared.metadata["source_adapter_unchanged"])

    def test_torch_compile_fixed_step_profile_uses_explicit_sampler(self) -> None:
        class FakeTorchModel:
            def sample_actions(self, *args, **kwargs):
                del args, kwargs

        self.policy._model = FakeTorchModel()
        factory_calls = []

        def fake_factory(model):
            factory_calls.append(model)
            return lambda *args, **kwargs: (args, kwargs)

        profile = OptimizationProfile(
            profile_id="pi05-compile-fixed-loop",
            backend="torch_compile",
            deployment_precision="bf16",
            inference_steps=2,
            action_horizon=10,
            options={"fixed_step_loop": True},
        )
        prepared = TorchCompileBackend(
            compile_fn=lambda function, **_kwargs: function,
            pi05_sampler_factory=fake_factory,
        ).prepare(self.adapter, Pi05ModelPlugin(), profile)

        self.assertEqual(factory_calls, [self.policy._model])
        self.assertTrue(prepared.metadata["fixed_step_loop"])

    def test_pi05_profiler_reads_system_gpu_memory(self) -> None:
        with mock.patch(
            "scripts.benchmark_carve_pi05_profile.subprocess.check_output",
            return_value="4321\n",
        ) as check_output:
            self.assertEqual(system_gpu_memory_used_mib("cuda:0"), 4321.0)
        self.assertIn("--id=0", check_output.call_args.args[0])
        self.assertIsNone(system_gpu_memory_used_mib("cpu"))

    def test_masked_view_backend_records_static_elision_contract(self) -> None:
        class FakeTorchModel:
            pass

        self.policy._model = FakeTorchModel()
        factory_calls: list[tuple[int, ...]] = []
        compile_calls: list[dict] = []

        def fake_factory(model, indices):
            self.assertIs(model, self.policy._model)
            factory_calls.append(tuple(indices))
            return lambda *args, **kwargs: (args, kwargs)

        def fake_compile(function, **kwargs):
            compile_calls.append(kwargs)
            return function

        profile = OptimizationProfile(
            profile_id="pi05-compile-masked-view",
            backend="torch_compile_masked_views",
            deployment_precision="bf16",
            inference_steps=2,
            action_horizon=3,
            options={
                "image_view_order": [
                    "base_0_rgb",
                    "left_wrist_0_rgb",
                    "right_wrist_0_rgb",
                ],
                "elided_image_views": ["right_wrist_0_rgb"],
                "mode": "reduce-overhead",
            },
        )
        prepared = MaskedViewElisionBackend(
            compile_fn=fake_compile,
            sampler_factory=fake_factory,
        ).prepare(self.adapter, Pi05ModelPlugin(), profile)

        self.assertEqual(factory_calls, [(2,)])
        self.assertEqual(compile_calls, [{"mode": "reduce-overhead", "fullgraph": False}])
        self.assertEqual(prepared.metadata["elided_image_indices"], [2])
        self.assertEqual(prepared.metadata["elided_image_views"], ["right_wrist_0_rgb"])
        self.assertEqual(
            prepared.metadata["input_view_contract"]["view_order"],
            ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        )
        self.assertEqual(prepared.metadata["mask_contract"], "all_batch_entries_false")
        self.assertTrue(prepared.metadata["source_adapter_unchanged"])

    def test_masked_view_elision_removes_only_fully_masked_views(self) -> None:
        import torch

        images = [torch.full((1, 1), index) for index in range(3)]
        masks = [
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([False]),
        ]

        kept_images, kept_masks = elide_masked_views(
            images, masks, (2,), torch_module=torch
        )

        self.assertEqual(len(kept_images), 2)
        self.assertEqual(len(kept_masks), 2)
        self.assertEqual([int(value.item()) for value in kept_images], [0, 1])

    def test_masked_view_elision_rejects_active_or_invalid_targets(self) -> None:
        import torch

        images = [torch.zeros((1, 1)) for _ in range(3)]
        active_masks = [torch.tensor([True]) for _ in range(3)]
        with self.assertRaisesRegex(AssertionError, "configured for elision"):
            elide_masked_views(images, active_masks, (2,), torch_module=torch)

        profile = OptimizationProfile(
            profile_id="pi05-invalid-masked-view",
            backend="torch_compile_masked_views",
            deployment_precision="bf16",
            options={"elided_image_indices": [2, 2]},
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            MaskedViewElisionBackend(
                compile_fn=lambda function, **kwargs: function,
                sampler_factory=lambda model, indices: lambda: None,
            ).prepare(self.adapter, Pi05ModelPlugin(), profile)

    def test_static_masked_view_contract_resolves_names_and_round_trips(self) -> None:
        contract = StaticMaskedViewContract(
            view_order=("head", "left_wrist", "right_wrist"),
            masked_views=("right_wrist",),
        )

        self.assertEqual(contract.masked_indices, (2,))
        self.assertEqual(
            StaticMaskedViewContract.from_dict(contract.to_dict()),
            contract,
        )
        with self.assertRaisesRegex(ValueError, "absent from view_order"):
            StaticMaskedViewContract(
                view_order=("head", "left_wrist"),
                masked_views=("right_wrist",),
            )

    def test_masked_view_backend_rejects_mixed_named_and_index_plans(self) -> None:
        profile = OptimizationProfile(
            profile_id="pi05-invalid-mixed-view-plan",
            backend="torch_compile_masked_views",
            deployment_precision="bf16",
            options={
                "image_view_order": ["head", "right_wrist"],
                "elided_image_views": ["right_wrist"],
                "elided_image_indices": [1],
            },
        )

        with self.assertRaisesRegex(ValueError, "either named masked views"):
            MaskedViewElisionBackend(
                compile_fn=lambda function, **kwargs: function,
                sampler_factory=lambda model, indices: lambda: None,
            ).prepare(self.adapter, Pi05ModelPlugin(), profile)

    def test_torchao_int8_quantizes_only_requested_component(self) -> None:
        import torch

        class FakeTorchModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.paligemma_with_expert = torch.nn.Module()
                self.paligemma_with_expert.paligemma = torch.nn.Module()
                self.paligemma_with_expert.paligemma.model = torch.nn.Module()
                self.paligemma_with_expert.paligemma.model.language_model = torch.nn.Linear(4, 4)
                self.paligemma_with_expert.gemma_expert = torch.nn.Linear(4, 4)

            def sample_actions(self, *args, **kwargs):
                del args, kwargs
                return np.zeros((1, 6, 7), dtype=np.float32)

        self.policy._model = FakeTorchModel()
        selected: list[str] = []

        def fake_quantize(model, config, *, filter_fn):
            del config
            selected.extend(
                name for name, module in model.named_modules() if filter_fn(module, name)
            )

        profile = OptimizationProfile(
            profile_id="pi05-w8a16-vlm",
            backend="torchao_int8",
            deployment_precision="bf16",
            module_precisions={"vlm_backbone": "int8"},
            inference_steps=2,
            action_horizon=3,
        )
        prepared = TorchAOInt8Backend(
            quantize_fn=fake_quantize,
            config_factory=object,
            compile_fn=lambda function, **kwargs: function,
        ).prepare(self.adapter, Pi05ModelPlugin(), profile)

        self.assertEqual(
            selected,
            ["paligemma_with_expert.paligemma.model.language_model"],
        )
        self.assertEqual(prepared.metadata["quantized_linear_count"], 1)
        self.assertFalse(prepared.metadata["source_adapter_unchanged"])

    def test_torchao_int8_composes_with_named_masked_view_contract(self) -> None:
        import torch

        class FakeTorchModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.paligemma_with_expert = torch.nn.Module()
                self.paligemma_with_expert.paligemma = torch.nn.Module()
                self.paligemma_with_expert.paligemma.model = torch.nn.Module()
                self.paligemma_with_expert.paligemma.model.language_model = torch.nn.Linear(4, 4)

            def sample_actions(self, *args, **kwargs):
                del args, kwargs
                return np.zeros((1, 6, 7), dtype=np.float32)

        self.policy._model = FakeTorchModel()
        selected: list[str] = []
        sampler_indices: list[int] = []

        def fake_quantize(model, config, *, filter_fn):
            del config
            selected.extend(
                name for name, module in model.named_modules() if filter_fn(module, name)
            )

        def fake_sampler(model, indices):
            del model
            sampler_indices.extend(indices)
            return lambda *args, **kwargs: np.zeros((1, 6, 7), dtype=np.float32)

        profile = OptimizationProfile(
            profile_id="pi05-int8-smve",
            backend="torchao_int8_masked_views",
            deployment_precision="bf16",
            module_precisions={"vlm_backbone": "int8"},
            inference_steps=2,
            action_horizon=3,
            options={
                "image_view_order": ["base", "left_wrist", "right_wrist"],
                "elided_image_views": ["right_wrist"],
            },
        )
        prepared = TorchAOInt8MaskedViewBackend(
            quantize_fn=fake_quantize,
            config_factory=object,
            compile_fn=lambda function, **kwargs: function,
            sampler_factory=fake_sampler,
        ).prepare(self.adapter, Pi05ModelPlugin(), profile)

        self.assertEqual(
            selected,
            ["paligemma_with_expert.paligemma.model.language_model"],
        )
        self.assertEqual(sampler_indices, [2])
        self.assertEqual(prepared.metadata["elided_image_views"], ["right_wrist"])
        self.assertIn("static_masked_view_elision", prepared.metadata["transform"])

    def test_fidelity_verifier_accepts_identity_and_rejects_gripper_flip(self) -> None:
        thresholds = FidelityThresholds(
            first_action_mae=0.01,
            chunk_mae=0.01,
            chunk_rmse=0.02,
            minimum_cosine=0.99,
            minimum_gripper_agreement=1.0,
            endpoint_l2=0.02,
            jerk_rmse=0.02,
        )
        verifier = ActionFidelityVerifier(thresholds)
        reference = np.full((5, 7), 0.1, dtype=np.float32)

        identical = verifier.compare(reference, reference.copy())
        candidate = reference.copy()
        candidate[:, -1] = -0.1
        changed = verifier.compare(reference, candidate)

        self.assertTrue(identical.passed)
        self.assertFalse(changed.passed)
        self.assertIn("gripper_agreement", changed.violations)

    def test_manifest_round_trip_preserves_profile_and_hardware(self) -> None:
        profile = OptimizationProfile(
            profile_id="pi05-2step",
            backend="eager",
            deployment_precision="bf16",
            inference_steps=2,
            action_horizon=10,
        )
        manifest = ProfileManifest(
            model_id="pi05",
            adapter_id="pi05",
            checkpoint_id="pi05_libero:test",
            hardware=HardwareSpec(
                accelerator="cuda",
                device_name="RTX 4090",
                total_memory_gb=24.0,
                software={"torch": "2.7.1"},
            ),
            profile=profile,
            benchmark={"runtime_p95_ms": 80.0},
            fidelity={"passed": True},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "profile.json"
            manifest.save(path)
            loaded = ProfileManifest.load(path)

        self.assertEqual(loaded.profile.to_dict(), profile.to_dict())
        self.assertEqual(loaded.hardware.device_name, "RTX 4090")
        self.assertTrue(loaded.fidelity["passed"])

    def _admitted_manifest(
        self,
        *,
        backend: str = "torch_compile",
        options: dict | None = None,
        fallback_profile_id: str | None = None,
    ) -> ProfileManifest:
        return ProfileManifest(
            model_id="pi05",
            adapter_id="pi05",
            checkpoint_id="pi05_libero:test",
            hardware=HardwareSpec(accelerator="cuda", device_name="RTX 4090"),
            profile=OptimizationProfile(
                profile_id=f"pi05-{backend}",
                backend=backend,
                deployment_precision="bf16",
                inference_steps=2,
                action_horizon=10,
                options=options or {},
            ),
            benchmark={"deadline_miss_rate": 0.0},
            fidelity={"passed": True, "samples": 45},
            admission={
                "status": "promoted",
                "scope": "pi0.5 LIBERO on RTX 4090",
                "fallback_profile_id": fallback_profile_id,
                "gates": {
                    "replay_fidelity": {"passed": True},
                    "realtime": {"passed": True},
                    "closed_loop": {"passed": True},
                },
            },
        )

    def test_legacy_manifest_is_readable_but_not_deployable(self) -> None:
        manifest = self._admitted_manifest()
        payload = manifest.to_dict()
        payload.pop("admission")

        loaded = ProfileManifest.from_dict(payload)
        decision = validate_profile_admission(loaded)

        self.assertFalse(decision.accepted)
        self.assertEqual(loaded.admission, {})
        self.assertIn("expected 'promoted'", " ".join(decision.violations))

    def test_promoted_compiled_profile_passes_admission(self) -> None:
        decision = validate_profile_admission(self._admitted_manifest())

        self.assertTrue(decision.accepted)
        decision.require_accepted()

    def test_rejected_quantized_profile_cannot_pass_on_fidelity_alone(self) -> None:
        manifest = dataclasses.replace(
            self._admitted_manifest(),
            admission={
                "status": "rejected",
                "scope": "pi0.5 LIBERO on RTX 4090",
                "gates": {
                    "replay_fidelity": {"passed": True},
                    "realtime": {"passed": True},
                    "closed_loop": {"passed": False},
                },
            },
        )

        decision = validate_profile_admission(manifest)

        self.assertFalse(decision.accepted)
        with self.assertRaisesRegex(ValueError, "closed_loop"):
            decision.require_accepted()

    def test_masked_view_profile_requires_contract_and_fallback(self) -> None:
        missing = validate_profile_admission(
            self._admitted_manifest(backend="torch_compile_masked_views")
        )
        accepted = validate_profile_admission(
            self._admitted_manifest(
                backend="torch_compile_masked_views",
                options={"elided_image_indices": [2]},
                fallback_profile_id="pi05-torch_compile",
            )
        )

        self.assertFalse(missing.accepted)
        self.assertIn("static input-view contract", " ".join(missing.violations))
        self.assertIn("fallback_profile_id", " ".join(missing.violations))
        self.assertTrue(accepted.accepted)

    def test_profile_with_deadline_misses_is_not_admitted(self) -> None:
        manifest = dataclasses.replace(
            self._admitted_manifest(),
            benchmark={"deadline_miss_rate": 0.01},
        )

        decision = validate_profile_admission(manifest)

        self.assertFalse(decision.accepted)
        self.assertIn("deadline_miss_rate", " ".join(decision.violations))

    def test_contract_fallback_retries_only_mask_contract_violation(self) -> None:
        class FakeServerPolicy:
            def __init__(self, error: Exception | None = None) -> None:
                self._sample_kwargs = {"num_steps": 2}
                self.error = error
                self.calls = 0

            def infer(self, request, **kwargs):
                del request, kwargs
                self.calls += 1
                if self.error is not None:
                    raise self.error
                return {"actions": np.zeros((10, 7), dtype=np.float32)}

        primary = FakeServerPolicy(
            AssertionError("image view 2 was configured for elision but is active")
        )
        fallback = FakeServerPolicy()
        policy = ContractFallbackPolicy(
            primary,
            fallback,
            fallback_profile_id="pi05-compiled-bf16",
        )

        output = policy.infer({"state": np.zeros(8)})

        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(policy.fallback_count, 1)
        self.assertTrue(output["carve_contract_fallback"]["applied"])

        primary.error = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            policy.infer({"state": np.zeros(8)})
        self.assertEqual(fallback.calls, 1)

    def test_runtime_fallback_retries_only_known_profile_error(self) -> None:
        class FakeServerPolicy:
            def __init__(self, error: Exception | None = None) -> None:
                self._sample_kwargs = {"num_steps": 2}
                self.error = error
                self.calls = 0

            def infer(self, request, **kwargs):
                del request, kwargs
                self.calls += 1
                if self.error is not None:
                    raise self.error
                return {"actions": np.zeros((10, 7), dtype=np.float32)}

        primary = FakeServerPolicy(RecursionError("maximum recursion depth exceeded"))
        fallback = FakeServerPolicy()
        policy = RuntimeFallbackPolicy(
            primary,
            fallback,
            fallback_profile_id="pi05-eager-bf16-2step-h10",
            trigger_markers=("maximum recursion depth exceeded",),
        )

        output = policy.infer({"state": np.zeros(8)})

        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(policy.fallback_count, 1)
        self.assertTrue(output["carve_runtime_fallback"]["applied"])
        self.assertIn("carve_runtime_fallback", policy.metadata)

        primary.error = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            policy.infer({"state": np.zeros(8)})
        self.assertEqual(fallback.calls, 1)

    def test_benchmark_runner_uses_profile_on_deployment_path(self) -> None:
        registry = create_default_registry()
        profile = OptimizationProfile(
            profile_id="pi05-eager-smoke",
            backend="eager",
            deployment_precision="bf16",
            inference_steps=2,
            action_horizon=3,
        )
        prepared = registry.get_backend("eager").prepare(
            self.adapter,
            registry.resolve_model(self.adapter),
            profile,
        )
        runner = BenchmarkRunner(
            prepared,
            config=BenchmarkConfig(warmup_calls=1, measured_calls=3),
        )

        report = runner.run(
            lambda index: InferenceRequest(
                observation={"observation/state": [float(index)]},
                instruction="move",
                controls=InferenceControls(deadline_ms=80.0),
            )
        )

        self.assertEqual(report.samples, 3)
        self.assertEqual(report.model_p50_ms, 3.5)
        self.assertEqual(report.mean_action_count, 3.0)
        self.assertEqual(self.policy.steps, [2, 2, 2, 2])
        self.assertEqual(self.policy._sample_kwargs["num_steps"], 7)

    def test_benchmark_runner_can_record_latency_samples(self) -> None:
        registry = create_default_registry()
        profile = OptimizationProfile(profile_id="pi05-eager-trace", backend="eager")
        prepared = registry.get_backend("eager").prepare(
            self.adapter,
            registry.resolve_model(self.adapter),
            profile,
        )
        runner = BenchmarkRunner(
            prepared,
            config=BenchmarkConfig(warmup_calls=0, measured_calls=2, record_samples=True),
        )

        report = runner.run(
            lambda index: InferenceRequest(
                observation={"observation/state": [float(index)]},
                instruction="move",
                controls=InferenceControls(deadline_ms=1.0),
            )
        )

        samples = report.metadata["latency_samples"]
        self.assertEqual(samples["model_ms"], [3.5, 3.5])
        self.assertEqual(len(samples["runtime_ms"]), 2)
        self.assertEqual(len(samples["reaction_ms"]), 2)
        self.assertGreaterEqual(
            report.metadata["reaction_p95_ms"],
            report.runtime_p95_ms,
        )
        self.assertEqual(samples["deadline_miss"], [False, False])


if __name__ == "__main__":
    unittest.main()

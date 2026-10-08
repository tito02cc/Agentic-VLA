"""Public API for CARVE Optimize Runtime."""

from .backends import (
    EagerBackend,
    MaskedViewElisionBackend,
    TorchAOInt8Backend,
    TorchAOInt8MaskedViewBackend,
    TorchCompileBackend,
)
from .admission import (
    PlannerAdmissionDecision,
    PlannerAdmissionRequirements,
    ProfileAdmissionDecision,
    ProfileAdmissionRequirements,
    SystemAdmissionDecision,
    SystemAdmissionRequirements,
    validate_planner_profile_admission,
    validate_profile_admission,
    validate_system_profile_admission,
)
from .benchmark import BenchmarkConfig, BenchmarkRunner
from .contracts import (
    BenchmarkReport,
    CoResidentBenchmarkReport,
    HardwareSpec,
    OptimizationProfile,
    PlannerBenchmarkReport,
    PlannerOptimizationProfile,
    StaticMaskedViewContract,
    SystemOptimizationProfile,
)
from .fidelity import ActionFidelityVerifier, FidelityReport, FidelityThresholds
from .fallback import ContractFallbackPolicy, RuntimeFallbackPolicy
from .manifest import ProfileManifest
from .models import (
    LingBotVlaModelPlugin,
    OpenVlaModelPlugin,
    Pi05ModelPlugin,
    StarVlaModelPlugin,
)
from .plugins import BackendPlugin, ModelPlugin, PreparedPolicy
from .registry import PluginRegistry
from .reuse import (
    EventCoherentReuseGate,
    ReuseCandidate,
    ReuseContext,
    ReuseDecision,
    ReuseGateConfig,
    ReuseLayer,
)
from .runtime import CarveOptimizeRuntime


def create_default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_model(LingBotVlaModelPlugin())
    registry.register_model(OpenVlaModelPlugin())
    registry.register_model(Pi05ModelPlugin())
    registry.register_model(StarVlaModelPlugin())
    registry.register_backend(EagerBackend())
    registry.register_backend(TorchCompileBackend())
    registry.register_backend(MaskedViewElisionBackend())
    registry.register_backend(TorchAOInt8Backend())
    registry.register_backend(TorchAOInt8MaskedViewBackend())
    return registry


def create_default_runtime() -> CarveOptimizeRuntime:
    return CarveOptimizeRuntime(create_default_registry())


__all__ = [
    "ActionFidelityVerifier",
    "BackendPlugin",
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunner",
    "CarveOptimizeRuntime",
    "CoResidentBenchmarkReport",
    "ContractFallbackPolicy",
    "EagerBackend",
    "EventCoherentReuseGate",
    "FidelityReport",
    "FidelityThresholds",
    "HardwareSpec",
    "LingBotVlaModelPlugin",
    "MaskedViewElisionBackend",
    "ModelPlugin",
    "OptimizationProfile",
    "OpenVlaModelPlugin",
    "Pi05ModelPlugin",
    "StarVlaModelPlugin",
    "PluginRegistry",
    "PlannerAdmissionDecision",
    "PlannerAdmissionRequirements",
    "PlannerBenchmarkReport",
    "PlannerOptimizationProfile",
    "PreparedPolicy",
    "ProfileAdmissionDecision",
    "ProfileAdmissionRequirements",
    "ProfileManifest",
    "RuntimeFallbackPolicy",
    "ReuseCandidate",
    "ReuseContext",
    "ReuseDecision",
    "ReuseGateConfig",
    "ReuseLayer",
    "StaticMaskedViewContract",
    "SystemAdmissionDecision",
    "SystemAdmissionRequirements",
    "SystemOptimizationProfile",
    "TorchAOInt8Backend",
    "TorchAOInt8MaskedViewBackend",
    "TorchCompileBackend",
    "create_default_registry",
    "create_default_runtime",
    "validate_planner_profile_admission",
    "validate_profile_admission",
    "validate_system_profile_admission",
]

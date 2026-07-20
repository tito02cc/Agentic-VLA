"""Public API for CARVE Optimize Runtime."""

from .backends import (
    EagerBackend,
    MaskedViewElisionBackend,
    TorchAOInt8Backend,
    TorchCompileBackend,
)
from .admission import (
    ProfileAdmissionDecision,
    ProfileAdmissionRequirements,
    validate_profile_admission,
)
from .benchmark import BenchmarkConfig, BenchmarkRunner
from .contracts import (
    BenchmarkReport,
    HardwareSpec,
    OptimizationProfile,
    StaticMaskedViewContract,
)
from .fidelity import ActionFidelityVerifier, FidelityReport, FidelityThresholds
from .fallback import ContractFallbackPolicy
from .manifest import ProfileManifest
from .models import OpenVlaModelPlugin, Pi05ModelPlugin
from .plugins import BackendPlugin, ModelPlugin, PreparedPolicy
from .registry import PluginRegistry
from .runtime import CarveOptimizeRuntime


def create_default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_model(OpenVlaModelPlugin())
    registry.register_model(Pi05ModelPlugin())
    registry.register_backend(EagerBackend())
    registry.register_backend(TorchCompileBackend())
    registry.register_backend(MaskedViewElisionBackend())
    registry.register_backend(TorchAOInt8Backend())
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
    "ContractFallbackPolicy",
    "EagerBackend",
    "FidelityReport",
    "FidelityThresholds",
    "HardwareSpec",
    "MaskedViewElisionBackend",
    "ModelPlugin",
    "OptimizationProfile",
    "OpenVlaModelPlugin",
    "Pi05ModelPlugin",
    "PluginRegistry",
    "PreparedPolicy",
    "ProfileAdmissionDecision",
    "ProfileAdmissionRequirements",
    "ProfileManifest",
    "StaticMaskedViewContract",
    "TorchAOInt8Backend",
    "TorchCompileBackend",
    "create_default_registry",
    "create_default_runtime",
    "validate_profile_admission",
]

"""CARVE optimization backend plugins."""

from .eager import EagerBackend
from .masked_views import MaskedViewElisionBackend
from .torch_compile import TorchCompileBackend
from .torchao_int8 import TorchAOInt8Backend

__all__ = [
    "EagerBackend",
    "MaskedViewElisionBackend",
    "TorchAOInt8Backend",
    "TorchCompileBackend",
]

"""Narrow online fallback for violated optimization input contracts."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


class ContractFallbackPolicy:
    """Retry one request on a fallback only when SMVE's mask contract fails."""

    _CONTRACT_MARKER = "configured for elision but is active"

    def __init__(self, primary: Any, fallback: Any, *, fallback_profile_id: str) -> None:
        if not hasattr(primary, "infer") or not hasattr(fallback, "infer"):
            raise TypeError("primary and fallback policies must expose infer(request)")
        primary_kwargs = getattr(primary, "_sample_kwargs", None)
        fallback_kwargs = getattr(fallback, "_sample_kwargs", None)
        if not isinstance(primary_kwargs, MutableMapping) or not isinstance(
            fallback_kwargs, MutableMapping
        ):
            raise TypeError("primary and fallback policies must expose mutable _sample_kwargs")
        self._primary = primary
        self._fallback = fallback
        self._sample_kwargs = primary_kwargs
        self._fallback_sample_kwargs = fallback_kwargs
        self._fallback_profile_id = str(fallback_profile_id).strip()
        if not self._fallback_profile_id:
            raise ValueError("fallback_profile_id must not be empty")
        self._fallback_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(getattr(self._primary, "metadata", {}) or {})
        metadata["carve_contract_fallback"] = {
            "fallback_profile_id": self._fallback_profile_id,
            "trigger": "static_masked_view_contract_violation",
        }
        return metadata

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    def infer(self, request: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        try:
            return self._primary.infer(request, **kwargs)
        except Exception as error:
            if self._CONTRACT_MARKER not in str(error):
                raise
            self._fallback_sample_kwargs.clear()
            self._fallback_sample_kwargs.update(self._sample_kwargs)
            output = self._fallback.infer(request, **kwargs)
            if not isinstance(output, Mapping):
                raise TypeError("fallback policy output must be a mapping")
            self._fallback_count += 1
            response = dict(output)
            response["carve_contract_fallback"] = {
                "applied": True,
                "reason": "static_masked_view_contract_violation",
                "fallback_profile_id": self._fallback_profile_id,
                "count": self._fallback_count,
            }
            return response

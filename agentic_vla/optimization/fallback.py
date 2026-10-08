"""Narrow online fallback for violated optimization input contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
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


class RuntimeFallbackPolicy:
    """Retry known deployment-profile failures with an eager policy.

    This wrapper is deliberately narrow: the caller must enumerate error
    markers observed for the research profile.  It therefore preserves errors
    such as OOM or malformed requests instead of disguising them as a
    successful inference call.  The response records every fallback so a
    runtime profile cannot silently acquire deployment credit.
    """

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        *,
        fallback_profile_id: str,
        trigger_markers: Iterable[str],
    ) -> None:
        if not hasattr(primary, "infer") or not hasattr(fallback, "infer"):
            raise TypeError("primary and fallback policies must expose infer(request)")
        primary_kwargs = getattr(primary, "_sample_kwargs", None)
        fallback_kwargs = getattr(fallback, "_sample_kwargs", None)
        if not isinstance(primary_kwargs, MutableMapping) or not isinstance(
            fallback_kwargs, MutableMapping
        ):
            raise TypeError("primary and fallback policies must expose mutable _sample_kwargs")
        markers = tuple(str(marker).strip() for marker in trigger_markers if str(marker).strip())
        if not markers:
            raise ValueError("trigger_markers must contain at least one non-empty marker")
        self._primary = primary
        self._fallback = fallback
        self._sample_kwargs = primary_kwargs
        self._fallback_sample_kwargs = fallback_kwargs
        self._fallback_profile_id = str(fallback_profile_id).strip()
        if not self._fallback_profile_id:
            raise ValueError("fallback_profile_id must not be empty")
        self._trigger_markers = markers
        self._fallback_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(getattr(self._primary, "metadata", {}) or {})
        metadata["carve_runtime_fallback"] = {
            "fallback_profile_id": self._fallback_profile_id,
            "trigger_markers": list(self._trigger_markers),
            "scope": "known_profile_execution_errors_only",
        }
        return metadata

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    def infer(self, request: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        try:
            return self._primary.infer(request, **kwargs)
        except Exception as error:
            error_text = str(error)
            marker = next((item for item in self._trigger_markers if item in error_text), None)
            if marker is None:
                raise
            self._fallback_sample_kwargs.clear()
            self._fallback_sample_kwargs.update(self._sample_kwargs)
            output = self._fallback.infer(request, **kwargs)
            if not isinstance(output, Mapping):
                raise TypeError("fallback policy output must be a mapping")
            self._fallback_count += 1
            response = dict(output)
            response["carve_runtime_fallback"] = {
                "applied": True,
                "reason": "known_profile_execution_error",
                "matched_marker": marker,
                "fallback_profile_id": self._fallback_profile_id,
                "count": self._fallback_count,
            }
            return response

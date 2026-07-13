"""Guards for launch-facing soft-lambda recommendations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from rlrmp.analysis.json_values import json_float as _json_float

CAP_INDEPENDENT_LAUNCH_BASES = frozenset(
    {
        "fixed_hvp_p90",
        "hvp_generalized_eigen",
        "hvp_lanczos_p90",
        "phenotype_scale",
    }
)

CAP_DERIVED_BASIS_MARKERS = frozenset(
    {
        "cap_boundary_fraction",
        "cap_boundary",
        "cap_interiority",
        "gradient_pressure_scale",
        "gradient_pressure_with_radius",
        "safety_cap",
        "selected_norm_radius_ratio",
        "trust_radius",
    }
)


class LambdaRecommendationBasisError(ValueError):
    """Raised when a lambda scale is not allowed to drive launch recommendations."""


@dataclass(frozen=True)
class LambdaScaleCandidate:
    """One lambda-scale candidate and its recommendation eligibility metadata."""

    name: str
    value: float
    basis: str
    diagnostic_only: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": _json_float(self.value),
            "basis": self.basis,
            "diagnostic_only": bool(self.diagnostic_only),
            "details": dict(self.details),
        }


def launch_lambda_recommendation(
    lambda_input: float,
    *,
    lambda_input_basis: str,
    candidates: Iterable[LambdaScaleCandidate] = (),
    rationale: str | None = None,
) -> dict[str, Any]:
    """Return a guarded launch-facing lambda-floor recommendation.

    Launch-facing recommendations must be based on explicit cap-independent
    evidence. Cap, trust-radius, boundary-fraction, selected-norm/radius, and
    radius-gradient-pressure quantities may be carried only as diagnostic-only
    candidates and are excluded from ``recommended_lambda_floor``.
    """

    input_candidate = LambdaScaleCandidate(
        name="lambda_input",
        value=float(lambda_input),
        basis=lambda_input_basis,
    )
    _require_launch_candidate(input_candidate)

    eligible = [input_candidate]
    diagnostic_only = []
    for candidate in candidates:
        if candidate.diagnostic_only:
            diagnostic_only.append(_diagnostic_candidate(candidate))
            continue
        _require_launch_candidate(candidate)
        eligible.append(candidate)

    recommended = max(float(candidate.value) for candidate in eligible)
    payload = {
        "lambda_input": _json_float(lambda_input),
        "lambda_input_basis": lambda_input_basis,
        "eligible_launch_candidates": [candidate.to_json() for candidate in eligible],
        "diagnostic_only_candidates": [candidate.to_json() for candidate in diagnostic_only],
        "recommended_lambda_floor": _json_float(recommended),
        "recommended_lambda_floor_basis": _winning_basis(eligible, recommended),
        "guard": (
            "Launch-facing lambda recommendations require a cap-independent basis. "
            "Safety-cap, trust-radius, cap-boundary, selected-norm/radius, and "
            "radius-gradient-pressure quantities are diagnostic-only unless a "
            "future analysis supplies a separate cap-independent launch basis."
        ),
    }
    if rationale is not None:
        payload["rationale"] = rationale
    return payload


def is_cap_derived_basis(basis: str) -> bool:
    """Return whether ``basis`` names cap/trust-radius-derived evidence."""

    normalized = _normalize_basis(basis)
    return any(marker in normalized for marker in CAP_DERIVED_BASIS_MARKERS)


def is_launch_basis(basis: str) -> bool:
    """Return whether ``basis`` is an accepted cap-independent launch basis."""

    return _normalize_basis(basis) in CAP_INDEPENDENT_LAUNCH_BASES


def _require_launch_candidate(candidate: LambdaScaleCandidate) -> None:
    if is_cap_derived_basis(candidate.basis):
        raise LambdaRecommendationBasisError(
            f"{candidate.name} uses cap/trust-radius-derived basis {candidate.basis!r}; "
            "mark it diagnostic_only=True and keep it out of launch recommendations"
        )
    if not is_launch_basis(candidate.basis):
        allowed = ", ".join(sorted(CAP_INDEPENDENT_LAUNCH_BASES))
        raise LambdaRecommendationBasisError(
            f"{candidate.name} uses unsupported launch lambda basis {candidate.basis!r}; "
            f"expected one of: {allowed}"
        )
    _require_finite(candidate)


def _diagnostic_candidate(candidate: LambdaScaleCandidate) -> LambdaScaleCandidate:
    _require_finite(candidate)
    return candidate


def _require_finite(candidate: LambdaScaleCandidate) -> None:
    value = float(candidate.value)
    if not math.isfinite(value):
        raise LambdaRecommendationBasisError(
            f"{candidate.name} has non-finite lambda scale {candidate.value!r}"
        )


def _winning_basis(candidates: list[LambdaScaleCandidate], recommended: float) -> str:
    for candidate in candidates:
        if float(candidate.value) == float(recommended):
            return candidate.basis
    raise AssertionError("recommended lambda did not come from an eligible candidate")


def _normalize_basis(basis: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", basis.strip().lower()).strip("_")

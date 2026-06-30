"""Regression tests for launch-facing soft-lambda recommendation guards."""

from __future__ import annotations

import pytest

from rlrmp.analysis.lambda_recommendations import (
    LambdaRecommendationBasisError,
    LambdaScaleCandidate,
    launch_lambda_recommendation,
)


def test_launch_recommendation_accepts_explicit_cap_independent_basis() -> None:
    payload = launch_lambda_recommendation(
        2.0,
        lambda_input_basis="fixed_hvp_p90",
        candidates=[
            LambdaScaleCandidate(
                name="mechanism_generalized_eigen",
                value=3.0,
                basis="hvp_generalized_eigen",
            )
        ],
    )

    assert payload["recommended_lambda_floor"] == 3.0
    assert payload["recommended_lambda_floor_basis"] == "hvp_generalized_eigen"


def test_radius_pressure_is_diagnostic_only_and_excluded_from_launch_floor() -> None:
    payload = launch_lambda_recommendation(
        2.0,
        lambda_input_basis="fixed_hvp_p90",
        candidates=[
            LambdaScaleCandidate(
                name="max_gradient_pressure_scale",
                value=4.13e8,
                basis="gradient_pressure_with_radius",
                diagnostic_only=True,
                details={"radius": 0.00123243},
            )
        ],
    )

    assert payload["recommended_lambda_floor"] == 2.0
    assert payload["diagnostic_only_candidates"][0]["value"] == 4.13e8


@pytest.mark.parametrize(
    "basis",
    [
        "safety_cap",
        "trust_radius",
        "cap_boundary_fraction",
        "selected-norm/radius ratio",
        "gradient_pressure_with_radius",
    ],
)
def test_cap_derived_launch_candidate_fails_loudly(basis: str) -> None:
    with pytest.raises(LambdaRecommendationBasisError, match="cap/trust-radius-derived"):
        launch_lambda_recommendation(
            2.0,
            lambda_input_basis="fixed_hvp_p90",
            candidates=[
                LambdaScaleCandidate(
                    name="leaky_candidate",
                    value=3.0,
                    basis=basis,
                )
            ],
        )


def test_unknown_launch_basis_fails_closed() -> None:
    with pytest.raises(LambdaRecommendationBasisError, match="unsupported launch lambda basis"):
        launch_lambda_recommendation(
            2.0,
            lambda_input_basis="pooled_beta_mapping_without_basis",
        )

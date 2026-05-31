"""Tests for bridge failure-decomposition helpers."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.failure_decomposition import (
    FailureDecompositionNumerics,
    IO_MAP_MISMATCH,
    REPRESENTATION_FAILURE,
    SIDECAR_IMPROVING_NON_EQUIVALENT,
    classify_failure,
    covariances_from_states,
    gain_error_subspace_decomposition,
    interpolation_curve,
    is_sidecar_improving_non_equivalent,
    objective_gradient_summary,
)


def test_objective_gradient_summary_reports_learned_and_reference() -> None:
    target = np.asarray([1.0, -2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sum((values - target) ** 2))

    def gradient(values: np.ndarray) -> np.ndarray:
        return 2.0 * (values - target)

    summary = objective_gradient_summary(
        learned=np.asarray([2.0, -2.0]),
        reference=target,
        objective_fn=objective,
        gradient_fn=gradient,
        projected_gradient_fn=gradient,
    )

    assert summary["learned_objective"] == 1.0
    assert summary["reference_objective"] == 0.0
    assert summary["learned_gradient_norm"] == 2.0
    assert summary["reference_projected_gradient_norm"] == 0.0


def test_interpolation_curve_evaluates_learned_to_reference_path() -> None:
    curve = interpolation_curve(
        learned=np.asarray([2.0]),
        reference=np.asarray([0.0]),
        metric_fns={"squared": lambda values: float(values[0] ** 2)},
        alphas=(0.0, 0.5, 1.0),
    )

    assert curve == [
        {"alpha": 0.0, "squared": 4.0},
        {"alpha": 0.5, "squared": 1.0},
        {"alpha": 1.0, "squared": 0.0},
    ]


def test_gain_error_subspace_decomposition_splits_visited_and_unvisited() -> None:
    gain_delta = np.asarray([[[2.0, 0.0, 3.0]]])
    covariances = np.asarray([np.diag([1.0, 1e-10, 0.0])])

    decomposition = gain_error_subspace_decomposition(
        gain_delta=gain_delta,
        state_covariances=covariances,
        numerics=FailureDecompositionNumerics(
            denominator_floor=1e-14,
            strong_covariance_rtol=1e-8,
            weak_covariance_rtol=1e-12,
        ),
    )

    assert np.isclose(decomposition["strong_fraction_mean"], 4.0 / 13.0)
    assert np.isclose(decomposition["weak_fraction_mean"], 0.0)
    assert np.isclose(decomposition["unvisited_fraction_mean"], 9.0 / 13.0)
    assert np.isclose(decomposition["weak_or_unvisited_fraction_mean"], 9.0 / 13.0)


def test_covariances_from_states_accepts_batch_time_layout() -> None:
    states = np.asarray([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]])

    covariances = covariances_from_states(states)

    assert covariances.shape == (2, 2, 2)
    assert np.allclose(covariances[0], np.diag([5.0, 0.0]))
    assert np.allclose(covariances[1], np.diag([0.0, 10.0]))


def test_classify_failure_distinguishes_optimizer_and_under_identification() -> None:
    optimizer = classify_failure(
        objective_ratio=1.2,
        learned_gradient_norm=10.0,
        reference_gradient_norm=0.0,
        certificate_mismatch_ratio=0.5,
        subspace_decomposition={"weak_or_unvisited_fraction_mean": 0.2},
    )
    under_identified = classify_failure(
        objective_ratio=1.0001,
        learned_gradient_norm=1e-5,
        reference_gradient_norm=0.0,
        certificate_mismatch_ratio=0.5,
        subspace_decomposition={"weak_or_unvisited_fraction_mean": 0.9},
    )

    assert optimizer["classification"] == "optimizer_basin"
    assert under_identified["classification"] == "under_identification"


def test_sidecar_improving_non_equivalent_classification_is_explicit() -> None:
    assert is_sidecar_improving_non_equivalent(
        sidecar_improved=True,
        equivalence_metrics_failed=True,
    )
    assert not is_sidecar_improving_non_equivalent(
        sidecar_improved=True,
        equivalence_metrics_failed=False,
    )

    classification = classify_failure(
        objective_ratio=1.0,
        learned_gradient_norm=0.0,
        reference_gradient_norm=0.0,
        certificate_mismatch_ratio=0.5,
        subspace_decomposition={"weak_or_unvisited_fraction_mean": 0.0},
        sidecar_improved=True,
        equivalence_metrics_failed=True,
    )

    assert classification["classification"] == SIDECAR_IMPROVING_NON_EQUIVALENT


def test_classify_failure_reports_recurrent_io_map_mismatch() -> None:
    classification = classify_failure(
        objective_ratio=1.0,
        learned_gradient_norm=0.0,
        reference_gradient_norm=0.0,
        certificate_mismatch_ratio=None,
        io_map_mismatch_ratio=0.5,
        subspace_decomposition=None,
    )

    assert classification["classification"] == IO_MAP_MISMATCH
    assert classification["signals"]["io_map_bad"]


def test_classify_failure_reports_representation_failure() -> None:
    classification = classify_failure(
        objective_ratio=1.0,
        learned_gradient_norm=0.0,
        reference_gradient_norm=0.0,
        certificate_mismatch_ratio=None,
        representation_failed=True,
        subspace_decomposition=None,
    )

    assert classification["classification"] == REPRESENTATION_FAILURE
    assert classification["signals"]["representation_failed"]

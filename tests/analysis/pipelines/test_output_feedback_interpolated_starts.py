"""Tests for 7cea1b7 interpolated output-feedback starts."""

from __future__ import annotations

import numpy as np
import pytest

from rlrmp.analysis.pipelines.output_feedback_interpolated_starts import (
    DEFAULT_CONDITION,
    INTERPOLATED_ALPHAS,
    alpha_label,
    build_interpolated_initializations,
    load_interpolated_initializations,
)


def test_build_interpolated_initializations_uses_expected_alpha_grid_and_labels() -> None:
    K_scratch = np.zeros((2, 1, 3), dtype=np.float64)
    K_ref = np.ones((2, 1, 3), dtype=np.float64)

    starts = build_interpolated_initializations(K_scratch, K_ref)

    assert tuple(start.alpha for start in starts) == INTERPOLATED_ALPHAS
    assert tuple(start.label for start in starts) == (
        "k_alpha_0p1",
        "k_alpha_0p25",
        "k_alpha_0p5",
        "k_alpha_0p75",
    )
    for start in starts:
        assert np.allclose(start.K, start.alpha)


def test_build_interpolated_initializations_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        build_interpolated_initializations(
            np.zeros((2, 1, 3), dtype=np.float64),
            np.zeros((3, 1, 3), dtype=np.float64),
        )


def test_load_interpolated_initializations_reads_rollout_artifact_keys(tmp_path) -> None:
    artifact = tmp_path / "source.npz"
    K_scratch = np.full((2, 1, 2), 2.0, dtype=np.float64)
    K_ref = np.full((2, 1, 2), 10.0, dtype=np.float64)
    np.savez_compressed(
        artifact,
        strong_optimizer_whitened__scratch_K=K_scratch,
        lqr_reference_K=K_ref,
    )

    starts = load_interpolated_initializations(artifact, alphas=(0.25, 0.5))

    assert tuple(start.label for start in starts) == ("k_alpha_0p25", "k_alpha_0p5")
    assert np.allclose(starts[0].K, 4.0)
    assert np.allclose(starts[1].K, 6.0)


def test_default_condition_labels_match_alpha_labels() -> None:
    assert DEFAULT_CONDITION.initializations == tuple(
        alpha_label(alpha) for alpha in INTERPOLATED_ALPHAS
    )

"""Tests for deterministic/stochastic rerun metadata labels."""

from __future__ import annotations

import pytest

from rlrmp.analysis.rerun_metadata import (
    DEFAULT_DISCRETIZATION,
    DEFAULT_LANE,
    build_rerun_metadata,
    validate_discretization,
    validate_lane,
)


def test_default_rerun_metadata_labels_current_materializers_as_euler_deterministic() -> None:
    metadata = build_rerun_metadata(materializer="smoke")

    assert metadata["discretization"] == DEFAULT_DISCRETIZATION == "euler"
    assert metadata["lane"] == DEFAULT_LANE == "deterministic_analytical"
    assert "no sampled sensory" in metadata["lane_description"]
    assert "ZOH results remain sensitivity/historical" in metadata["phase_label_policy"]


def test_rerun_metadata_accepts_future_euler_stochastic_labels() -> None:
    metadata = build_rerun_metadata(
        discretization="euler",
        lane="released_stochastic",
        materializer="future_stochastic_rollout",
    )

    assert metadata["discretization"] == "euler"
    assert metadata["lane"] == "released_stochastic"
    assert "sampled sensory" in metadata["lane_description"]


def test_rerun_metadata_rejects_ambiguous_labels() -> None:
    with pytest.raises(ValueError, match="Unknown discretization"):
        validate_discretization("cs_faithful")
    with pytest.raises(ValueError, match="Unknown lane"):
        validate_lane("faithful")

"""Tests for the special SISU spectrum diagnostics."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import (
    ReferenceCurve,
    RunSisuProfile,
    SisuCurve,
    build_velocity_profile_figure,
    robustification_comparison,
    set_sisu_condition,
    zero_disturbance_payload,
)


class TrialSpec(eqx.Module):
    """Tiny PyTree test double for Feedbax trial specs."""

    inputs: dict[str, jnp.ndarray]


def test_set_sisu_condition_prefers_sisu_without_clobbering_input() -> None:
    trials = TrialSpec(
        inputs={
            "sisu": jnp.ones((2, 3)),
            "input": jnp.full((2, 3), 7.0),
        }
    )

    updated = set_sisu_condition(trials, 0.5)

    np.testing.assert_allclose(np.asarray(updated.inputs["sisu"]), 0.5)
    np.testing.assert_allclose(np.asarray(updated.inputs["input"]), 7.0)


def test_set_sisu_condition_uses_input_when_sisu_absent() -> None:
    trials = TrialSpec(inputs={"input": jnp.ones((2, 3))})

    updated = set_sisu_condition(trials, 0.0)

    np.testing.assert_allclose(np.asarray(updated.inputs["input"]), 0.0)


def test_zero_disturbance_payload_zeros_epsilon_only() -> None:
    trials = TrialSpec(
        inputs={
            "epsilon": jnp.ones((2, 3, 4)),
            "input": jnp.ones((2, 3)),
        }
    )

    updated = zero_disturbance_payload(trials)

    np.testing.assert_allclose(np.asarray(updated.inputs["epsilon"]), 0.0)
    np.testing.assert_allclose(np.asarray(updated.inputs["input"]), 1.0)


def test_robustification_comparison_reports_ratios_and_deltas() -> None:
    curves = (
        _curve(0.0, endpoint=0.15, peak=0.02),
        _curve(1.0, endpoint=0.003, peak=0.8),
    )

    comparison = robustification_comparison(curves)

    assert np.isclose(comparison["endpoint_error_delta_0_minus_1_m"], 0.147)
    assert np.isclose(comparison["endpoint_error_ratio_1_over_0"], 0.02)
    assert np.isclose(comparison["peak_velocity_delta_1_minus_0_m_s"], 0.78)
    assert np.isclose(comparison["peak_velocity_ratio_1_over_0"], 40.0)


def test_velocity_profile_figure_uses_shared_y_axis() -> None:
    profiles = (
        RunSisuProfile(
            run_id="run_a",
            label="A",
            input_key="input",
            target_final_position_m=[0.15, 0.0],
            validation_input_unique=[1.0],
            validation_epsilon_l2_mean=0.0,
            checkpoint_selection=(),
            curves=(_curve(0.0), _curve(0.5), _curve(1.0)),
        ),
        RunSisuProfile(
            run_id="run_b",
            label="B",
            input_key="input",
            target_final_position_m=[0.15, 0.0],
            validation_input_unique=[1.0],
            validation_epsilon_l2_mean=0.0,
            checkpoint_selection=(),
            curves=(_curve(0.0), _curve(0.5), _curve(1.0)),
        ),
    )
    references = (
        ReferenceCurve(
            label="extLQG",
            time_s=np.array([0.0, 0.01]),
            forward_velocity_m_s=np.array([0.0, 0.1]),
            std_forward_velocity_m_s=np.array([0.0, 0.0]),
            line_color="#111827",
            line_dash="dash",
            controller="ext",
        ),
    )

    fig = build_velocity_profile_figure(profiles, references)

    assert fig.layout.yaxis2.matches == "y"


def _curve(sisu: float, *, endpoint: float = 0.1, peak: float = 0.2) -> SisuCurve:
    time_s = np.array([0.0, 0.01])
    return SisuCurve(
        sisu=sisu,
        time_s=time_s,
        mean_forward_velocity_m_s=np.array([0.0, peak]),
        std_forward_velocity_m_s=np.array([0.0, 0.01]),
        replicate_mean_forward_velocity_m_s=np.array([[0.0, peak], [0.0, peak]]),
        endpoint_error_by_replicate_m=np.array([endpoint, endpoint]),
        peak_velocity_by_replicate_m_s=np.array([peak, peak]),
        final_position_by_replicate_m=np.array([[0.15 - endpoint, 0.0], [0.15 - endpoint, 0.0]]),
    )

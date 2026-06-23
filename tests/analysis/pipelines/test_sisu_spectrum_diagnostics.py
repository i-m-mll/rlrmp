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
from rlrmp.analysis.pipelines.sisu_perturbation_comparison import (
    compare_summary_groups,
    metric_mean,
    render_markdown,
    summarize_headline,
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


def test_set_sisu_condition_updates_delayed_controller_sisu_column() -> None:
    controller_input = jnp.zeros((2, 3, 2))
    controller_input = controller_input.at[..., 0].set(1.0)
    trials = TrialSpec(
        inputs={
            "sisu": jnp.ones((2, 4)),
            "input": controller_input,
        }
    )

    updated = set_sisu_condition(trials, 0.25)

    np.testing.assert_allclose(np.asarray(updated.inputs["sisu"]), 0.25)
    np.testing.assert_allclose(np.asarray(updated.inputs["input"][..., 0]), 1.0)
    np.testing.assert_allclose(np.asarray(updated.inputs["input"][..., 1]), 0.25)


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


def test_sisu_perturbation_metric_mean_reads_flat_and_nested_metrics() -> None:
    metrics = {
        "delta_action_norm": {"mean": 2.0},
        "delta_position_response_m": {
            "max": {"mean": 0.01},
            "auc": {"mean": 0.002},
        },
        "extra_full_qrf_delta_cost_total": {"mean": 12.0},
    }

    assert metric_mean(metrics, "delta_action_norm") == 2.0
    assert metric_mean(metrics, "delta_position_response_m.max") == 0.01
    assert metric_mean(metrics, "delta_position_response_m.auc") == 0.002
    assert metric_mean(metrics, "extra_full_qrf_delta_cost_total") == 12.0
    assert metric_mean(metrics, "missing.metric") is None


def test_sisu_perturbation_group_comparison_reports_ratio_and_delta() -> None:
    low = {
        "command_input/command_input_pulse": _summary_group(
            rows=12,
            action=2.0,
            max_dx=0.010,
            auc_dx=0.004,
            endpoint=0.003,
            terminal=0.002,
            cost=100.0,
        )
    }
    high = {
        "command_input/command_input_pulse": _summary_group(
            rows=12,
            action=1.0,
            max_dx=0.005,
            auc_dx=0.002,
            endpoint=0.001,
            terminal=0.004,
            cost=40.0,
        )
    }

    comparison = compare_summary_groups(low, high)
    group = comparison["command_input/command_input_pulse"]

    assert group["metrics"]["mean_delta_action"]["ratio_1_over_0"] == 0.5
    assert group["metrics"]["max_delta_x_m"]["delta_1_minus_0"] == -0.005
    assert group["metrics"]["mean_endpoint_delta_m"]["delta_1_minus_0"] == -0.002
    assert group["metrics"]["mean_terminal_speed_delta_m_s"]["delta_1_minus_0"] == 0.002
    assert group["metrics"]["mean_full_qrf_delta_cost"]["ratio_1_over_0"] == 0.4
    assert summarize_headline(comparison)["full_qrf_delta_cost"]["improved"] == 1


def test_sisu_perturbation_markdown_declares_rerollout_policy() -> None:
    groups = compare_summary_groups(
        {
            "initial_state/initial_position_offset": _summary_group(
                rows=4,
                action=2.0,
                max_dx=0.010,
                auc_dx=0.004,
                endpoint=0.003,
                terminal=0.002,
                cost=100.0,
            )
        },
        {
            "initial_state/initial_position_offset": _summary_group(
                rows=4,
                action=1.0,
                max_dx=0.005,
                auc_dx=0.002,
                endpoint=0.001,
                terminal=0.004,
                cost=40.0,
            )
        },
    )
    manifest = {
        "issue": "e4800d6",
        "source_experiment": "e4800d6",
        "bank": {
            "bank_id": "test_bank",
            "mode": "calibrated",
            "n_perturbation_rows": 4,
        },
        "n_rollout_trials_per_replicate": 2,
        "runs": {
            "run_b": {
                "label": "effective 020a65b PGD targetfix",
                "headline": summarize_headline(groups),
                "class_comparison": groups,
                "timing_cell_comparison": groups,
            },
            "run_a": {
                "label": "raw strong gamma-1.05 targetfix",
                "headline": summarize_headline(groups),
                "class_comparison": groups,
                "timing_cell_comparison": groups,
            }
        },
    }

    markdown = render_markdown(manifest)

    assert "SISU Perturbation-Class Robustification Comparison" in markdown
    assert "ratio below 1 is an improvement" in markdown
    assert "reran both SISU=0 and SISU=1 locally" in markdown
    assert "raw strong gamma-1.05 targetfix" in markdown
    assert "### Metric Glossary" in markdown
    assert "Mean delta action 0" not in markdown
    assert "Signed Diagnostics" in markdown
    assert markdown.index("raw strong gamma-1.05 targetfix") < markdown.index(
        "effective 020a65b PGD targetfix"
    )


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


def _summary_group(
    *,
    rows: int,
    action: float,
    max_dx: float,
    auc_dx: float,
    endpoint: float,
    terminal: float,
    cost: float,
) -> dict[str, object]:
    return {
        "n_rows": rows,
        "status_counts": {"evaluated": rows},
        "amplitudes": [1.0],
        "metrics": {
            "delta_action_norm": {"mean": action},
            "delta_position_response_m": {
                "max": {"mean": max_dx},
                "auc": {"mean": auc_dx},
            },
            "delta_endpoint_error_m": {"mean": endpoint},
            "delta_terminal_speed_m_s": {"mean": terminal},
            "extra_full_qrf_delta_cost_total": {"mean": cost},
        },
    }

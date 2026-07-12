"""Contract and archived-parity tests for SISU spectrum science."""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from feedbax.analysis.evaluation import get_evaluation_recipe
from feedbax.analysis.specs import get_analysis_recipe

from rlrmp.analysis import sisu_spectrum
from rlrmp.analysis.sisu_spectrum import (
    SISU_ROBUSTIFICATION_ANALYSIS_TYPE,
    SISU_SPECTRUM_ANALYSIS_TYPE,
    SISU_SPECTRUM_EVALUATION_TYPE,
    ReferenceCurve,
    build_perturbation_comparison,
    compare_summary_groups,
    profile_payload,
    reference_payload,
    register_sisu_spectrum_recipes,
    robustification_comparison,
    summarize_headline,
)
from rlrmp.eval.sisu_spectrum import RunSisuProfile, SisuCurve, set_sisu_condition


class TrialSpec(eqx.Module):
    """Tiny PyTree test double for Feedbax trial specs."""

    inputs: dict[str, jnp.ndarray]


def test_set_sisu_condition_updates_declared_input_without_clobbering_payload() -> None:
    trials = TrialSpec(
        inputs={
            "sisu": jnp.ones((2, 3)),
            "input": jnp.full((2, 3), 7.0),
        }
    )

    updated = set_sisu_condition(trials, 0.5)

    np.testing.assert_allclose(np.asarray(updated.inputs["sisu"]), 0.5)
    np.testing.assert_allclose(np.asarray(updated.inputs["input"]), 7.0)


def test_profile_and_reference_payloads_are_json_ready_for_figure_stage() -> None:
    profile = RunSisuProfile(
        run_id="run_a",
        label="A",
        input_key="input",
        target_final_position_m=[0.15, 0.0],
        validation_input_unique=[1.0],
        validation_epsilon_l2_mean=0.0,
        checkpoint_selection=(),
        curves=(_curve(0.0), _curve(1.0)),
    )
    reference = ReferenceCurve(
        label="analytical",
        time_s=np.array([0.0, 0.01]),
        forward_velocity_m_s=np.array([0.0, 0.1]),
        std_forward_velocity_m_s=np.array([0.0, 0.01]),
        controller="analytical_extlqg_output_feedback",
    )

    payload = {
        "profiles": profile_payload((profile,)),
        "references": reference_payload((reference,)),
    }

    assert json.loads(json.dumps(payload)) == payload
    assert payload["profiles"][0]["curves"][1]["sisu"] == 1.0
    assert payload["references"][0]["controller"] == reference.controller


def test_profile_robustification_parity() -> None:
    comparison = robustification_comparison(
        (_curve(0.0, endpoint=0.15, peak=0.02), _curve(1.0, endpoint=0.003, peak=0.8))
    )

    assert np.isclose(comparison["endpoint_error_delta_0_minus_1_m"], 0.147)
    assert np.isclose(comparison["endpoint_error_ratio_1_over_0"], 0.02)
    assert np.isclose(comparison["peak_velocity_delta_1_minus_0_m_s"], 0.78)
    assert np.isclose(comparison["peak_velocity_ratio_1_over_0"], 40.0)


def test_grouped_robustification_parity() -> None:
    low = {"command_input/pulse": _summary_group(action=2.0, cost=100.0)}
    high = {"command_input/pulse": _summary_group(action=1.0, cost=40.0)}

    comparison = compare_summary_groups(low, high)
    group = comparison["command_input/pulse"]

    assert group["metrics"]["mean_delta_action"]["ratio_1_over_0"] == 0.5
    assert group["metrics"]["max_delta_x_m"]["delta_1_minus_0"] == -0.005
    assert group["metrics"]["mean_full_qrf_delta_cost"]["ratio_1_over_0"] == 0.4
    assert summarize_headline(comparison)["mean_full_qrf_delta_cost"]["improved"] == 1


def test_canonical_comparison_consumes_paired_cached_summaries() -> None:
    low = {"robust_response_summary": _response_summary(action=2.0, cost=100.0)}
    high = {
        "label": "high SISU",
        "robust_response_summary": _response_summary(action=1.0, cost=40.0),
    }

    payload = build_perturbation_comparison({"run_a": {"sisu_0": low, "sisu_1": high}})

    assert payload["schema_id"] == "rlrmp.sisu_perturbation_class_comparison.v1"
    assert payload["runs"]["run_a"]["label"] == "high SISU"
    assert payload["runs"]["run_a"]["headline"]["mean_full_qrf_delta_cost"]["improved"] == 1


def test_archived_comparison_matches_canonical_shape() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    archived = json.loads(
        (
            repo_root / "results/e4800d6/notes/sisu_perturbation_class_comparison_targetfix.json"
        ).read_text(encoding="utf-8")
    )

    assert archived["schema_version"] == "rlrmp.sisu_perturbation_class_comparison.v1"
    for run in archived["runs"].values():
        assert set(run["headline"]) == {
            "full_qrf_delta_cost",
            "max_delta_x_m",
            "mean_delta_action",
        }
        assert run["class_comparison"]
        assert run["timing_cell_comparison"]


def test_registered_surfaces_have_no_rendering_or_direct_writers() -> None:
    register_sisu_spectrum_recipes(replace=True)

    assert get_evaluation_recipe(SISU_SPECTRUM_EVALUATION_TYPE) is (
        sisu_spectrum.sisu_spectrum_evaluation_recipe
    )
    assert get_analysis_recipe(SISU_SPECTRUM_ANALYSIS_TYPE) is (sisu_spectrum.sisu_spectrum_recipe)
    assert get_analysis_recipe(SISU_ROBUSTIFICATION_ANALYSIS_TYPE) is (
        sisu_spectrum.sisu_robustification_recipe
    )
    source = Path(sisu_spectrum.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "plotly",
        "record_figure",
        "savez_compressed",
        "NamedTemporaryFile",
        ".write_text(",
        "update_marked_section",
        "020a65b",
    ):
        assert forbidden not in source


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


def _summary_group(*, action: float, cost: float) -> dict[str, object]:
    return {
        "n_rows": 12,
        "status_counts": {"evaluated": 12},
        "metrics": {
            "delta_action_norm": {"mean": action},
            "delta_position_response_m": {
                "max": {"mean": 0.005 if action == 1.0 else 0.010},
                "auc": {"mean": 0.002 if action == 1.0 else 0.004},
            },
            "delta_endpoint_error_m": {"mean": 0.001},
            "delta_terminal_speed_m_s": {"mean": 0.002},
            "extra_full_qrf_delta_cost_total": {"mean": cost},
        },
    }


def _response_summary(*, action: float, cost: float) -> dict[str, object]:
    groups = {"command_input/pulse": _summary_group(action=action, cost=cost)}
    return {
        "class_summary": {"groups": groups},
        "timing_cell_summary": {"groups": groups},
    }

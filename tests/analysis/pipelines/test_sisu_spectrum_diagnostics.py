"""Tests for the special SISU spectrum diagnostics."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from feedbax.analysis.evaluation import execute_evaluation_run_spec
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.contracts.manifest import load_manifest
from feedbax.contracts.manifest import EvaluationRunSpec
import pytest

from rlrmp.analysis.pipelines import sisu_spectrum_diagnostics as sisu_pipeline
from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import (
    DEFAULT_TOPIC,
    ReferenceCurve,
    SISU_SPECTRUM_ANALYSIS_TYPE,
    SISU_SPECTRUM_COMPACT_ARRAYS_ROLE,
    SISU_SPECTRUM_EVALUATION_TYPE,
    SISU_SPECTRUM_MANIFEST_ROLE,
    SISU_SPECTRUM_NOTE_ROLE,
    SisuSpectrumAnalysisParams,
    SisuSpectrumEvaluationParams,
    build_velocity_profile_figure,
    robustification_comparison,
    sisu_spectrum_evaluation_recipe,
    sisu_spectrum_evaluation_spec_params,
)
from rlrmp.eval.sisu_spectrum import (
    RunSisuProfile,
    SisuCurve,
    set_sisu_condition,
    zero_disturbance_payload,
)
from rlrmp.analysis.declarative_materialization import (
    register_certificate_analysis_recipes,
    sisu_spectrum_evaluation_spec,
    sisu_spectrum_spec,
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


def test_sisu_spectrum_params_require_explicit_runs_and_labels() -> None:
    params = sisu_spectrum_evaluation_spec_params(
        experiment="example",
        run_ids=("run_a", "run_b"),
        labels=("A", "B"),
    )

    assert params["experiment"] == "example"
    assert params["schema_id"] == "rlrmp.sisu_spectrum.evaluation_params"
    assert params["schema_version"] == "v2"
    assert params["topic"] == DEFAULT_TOPIC
    assert params["run_ids"] == ["run_a", "run_b"]
    assert "e4800d6" not in repr(SisuSpectrumEvaluationParams.model_json_schema())


def test_sisu_spectrum_specs_use_manifest_parent_refs() -> None:
    eval_spec = sisu_spectrum_evaluation_spec(
        experiment="example",
        run_ids=("run_a",),
        labels=("A",),
    )
    analysis_spec = sisu_spectrum_spec(evaluation_manifest_id="eval_manifest_1")

    assert eval_spec.evaluation_type == SISU_SPECTRUM_EVALUATION_TYPE
    assert eval_spec.training_run_ids == ["run_a"]
    assert eval_spec.inputs[0].kind == "TrainingRunManifest"
    assert analysis_spec.analysis_type == SISU_SPECTRUM_ANALYSIS_TYPE
    assert analysis_spec.inputs[0].kind == "EvaluationRunManifest"
    assert analysis_spec.inputs[0].role == "evaluation_run"
    assert analysis_spec.params["schema_id"] == "rlrmp.sisu_spectrum.analysis_params"
    assert analysis_spec.params["schema_version"] == "v1"


def test_sisu_specs_reject_pre_custody_unversioned_params() -> None:
    with pytest.raises(ValueError, match="schema_id"):
        SisuSpectrumEvaluationParams.model_validate(
            {
                "experiment": "example",
                "run_ids": ["run_a"],
                "labels": ["A"],
            }
        )
    with pytest.raises(ValueError, match="schema_id"):
        SisuSpectrumAnalysisParams.model_validate({})


def test_sisu_spectrum_recipes_register_with_declarative_materialization() -> None:
    from feedbax.analysis.evaluation import get_evaluation_recipe
    from feedbax.analysis.specs import get_analysis_recipe

    register_certificate_analysis_recipes(replace=True)

    assert get_evaluation_recipe(SISU_SPECTRUM_EVALUATION_TYPE).__name__ == (
        "sisu_spectrum_evaluation_recipe"
    )
    assert get_analysis_recipe(SISU_SPECTRUM_ANALYSIS_TYPE).__name__ == "sisu_spectrum_recipe"


def test_sisu_rollout_execution_and_experiment_identity_have_single_owners() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pipeline_source = (
        repo_root / "src/rlrmp/analysis/pipelines/sisu_spectrum_diagnostics.py"
    ).read_text(encoding="utf-8")
    eval_source = (repo_root / "src/rlrmp/eval/sisu_spectrum.py").read_text(encoding="utf-8")
    driver_source = (
        repo_root / "results/e4800d6/scripts/materialize_sisu_spectrum_special.py"
    ).read_text(encoding="utf-8")

    assert "eval_trials(" not in pipeline_source
    assert "eval_trials(" in eval_source
    assert 'EXPERIMENT = "e4800d6"' not in pipeline_source
    assert "DEFAULT_RUN_IDS" not in pipeline_source
    assert 'EXPERIMENT = "e4800d6"' in driver_source
    assert "DEFAULT_RUN_IDS" in driver_source


def test_sisu_registered_evaluation_recipe_dispatches_eval_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return ()

    monkeypatch.setattr(sisu_pipeline.sisu_eval, "evaluate_sisu_profiles", fake_evaluate)
    monkeypatch.setattr(sisu_pipeline, "analytical_reference_curves", lambda **_kwargs: ())
    monkeypatch.setattr(
        sisu_pipeline,
        "build_manifest",
        lambda **kwargs: {"issue": kwargs["experiment"], "runs": {}},
    )
    params = sisu_spectrum_evaluation_spec_params(
        experiment="example",
        run_ids=("run_a",),
        labels=("A",),
        n_rollout_trials=2,
    )

    result = sisu_spectrum_evaluation_recipe(
        EvaluationRunSpec(
            evaluation_type=SISU_SPECTRUM_EVALUATION_TYPE,
            training_run_ids=["run_a"],
            params=params,
        ),
        tmp_path,
        tmp_path / "states.eqx",
    )

    assert len(calls) == 1
    assert calls[0]["experiment"] == "example"
    assert calls[0]["run_ids"] == ("run_a",)
    assert result.states["manifest"] == {"issue": "example", "runs": {}}


def test_sisu_materialization_records_compact_arrays_and_manifest_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        label="extLQG",
        time_s=np.array([0.0, 0.01]),
        forward_velocity_m_s=np.array([0.0, 0.1]),
        std_forward_velocity_m_s=np.array([0.0, 0.0]),
        line_color="#111827",
        line_dash="dash",
        controller="ext",
    )
    monkeypatch.setattr(
        sisu_pipeline.sisu_eval,
        "evaluate_sisu_profiles",
        lambda **_kwargs: (profile,),
    )
    monkeypatch.setattr(
        sisu_pipeline,
        "analytical_reference_curves",
        lambda **_kwargs: (reference,),
    )
    register_certificate_analysis_recipes(replace=True)

    eval_manifest, eval_path = execute_evaluation_run_spec(
        sisu_spectrum_evaluation_spec(
            experiment="example",
            run_ids=("run_a",),
            labels=("A",),
            sisu_levels=(0.0, 1.0),
            n_rollout_trials=2,
            reference_samples=2,
        ),
        root=tmp_path,
        force=True,
    )
    analysis_manifest, analysis_path = execute_analysis_run_spec(
        sisu_spectrum_spec(
            evaluation_manifest_id=eval_manifest.id,
            evaluation_manifest_uri=eval_path,
        ),
        root=tmp_path,
        issues=["dc96336", "example"],
        force=True,
    )

    by_role = {artifact.role: artifact for artifact in analysis_manifest.artifacts}
    compact_ref = by_role[SISU_SPECTRUM_COMPACT_ARRAYS_ROLE]
    manifest_ref = by_role[SISU_SPECTRUM_MANIFEST_ROLE]
    note_ref = by_role[SISU_SPECTRUM_NOTE_ROLE]
    figure_refs = [
        artifact for artifact in analysis_manifest.artifacts if artifact.role == "figure"
    ]
    payload = json.loads(Path(manifest_ref.uri).read_text(encoding="utf-8"))

    assert analysis_manifest.inputs[0].kind == "EvaluationRunManifest"
    assert analysis_manifest.inputs[0].id == eval_manifest.id
    assert analysis_manifest.provenance.parents[0].id == eval_manifest.id
    assert analysis_manifest.provenance.issues == ["dc96336", "example"]
    assert analysis_manifest.summary_metrics["figure_count"] == 1
    assert payload["schema_id"] == "rlrmp.sisu_spectrum_special.v3"
    assert payload["outputs"]["compact_arrays"]["uri"] == compact_ref.uri
    assert payload["outputs"]["compact_arrays"]["artifact_id"] == compact_ref.artifact_id
    assert payload["outputs"]["markdown"]["uri"] == note_ref.uri
    assert payload["outputs"]["figure"][0]["uri"] == figure_refs[0].uri
    assert compact_ref.metadata["artifact_group"]["id"] == "sisu_spectrum_compact_arrays"
    assert compact_ref.metadata["artifact_group"]["member_role"] == "velocity_profile_curves"
    assert Path(compact_ref.uri).is_file()
    with np.load(compact_ref.uri) as arrays:
        assert "run_0_sisu_0p0_time_s" in arrays.files
        assert "reference_0_forward_velocity_m_s" in arrays.files
    markdown = Path(note_ref.uri).read_text(encoding="utf-8")
    assert "# SISU Spectrum Special Analysis" in markdown
    assert compact_ref.uri in markdown
    assert len(figure_refs) == 1
    assert figure_refs[0].metadata["params"]["schema_id"] == (
        "rlrmp.figure_spec.sisu_spectrum_velocity_profiles.v1"
    )
    assert Path(figure_refs[0].uri).is_file()
    assert load_manifest(analysis_path).id == analysis_manifest.id


def test_e4800d6_driver_routes_specs_without_raw_writes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    driver = repo_root / "results/e4800d6/scripts/materialize_sisu_spectrum_special.py"
    source = driver.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(driver))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert "sisu_spectrum_evaluation_spec" in calls
    assert "execute_evaluation_run_spec" in calls
    assert "sisu_spectrum_spec" in calls
    assert "execute_analysis_run_spec" in calls
    assert calls.index("execute_evaluation_run_spec") < calls.index("execute_analysis_run_spec")
    for forbidden in (
        "evaluate_sisu_profiles",
        "save_figure",
        "savez_compressed",
        "write_compact_arrays",
        "write_note",
        "build_manifest",
        ".write_text(",
    ):
        assert forbidden not in source


def test_sisu_markdown_rejects_pre_custody_manifest_schema() -> None:
    with pytest.raises(ValueError, match="sisu_spectrum_special.v3"):
        sisu_pipeline.render_markdown({"schema_id": "rlrmp.sisu_spectrum_special.v2"})


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
            },
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

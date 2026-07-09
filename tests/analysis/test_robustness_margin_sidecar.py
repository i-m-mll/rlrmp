"""Tests for robustness-margin sidecar estimators and Feedbax recipe custody."""

from __future__ import annotations

from pathlib import Path

import pytest

import rlrmp
from feedbax.analysis.bundles import load_analysis_bundle
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.contracts.evaluation_states import store_evaluation_states_artifact
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunManifest,
    ParentRef,
    spec_payload,
    write_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry
from rlrmp.analysis import robustness_margin as rm
from rlrmp.runtime.params_models import params_model_for


def _registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    return registry


def _toy_row(*, row_id: str, setpoint: float, lambda_value: float) -> dict[str, object]:
    return {
        "row_id": row_id,
        "run_id": f"run-{row_id}",
        "damage_setpoint": setpoint,
        "lambda": lambda_value,
        "hessian": [[4.0, 0.0], [0.0, 1.0]],
        "energy_metric": [[1.0, 0.0], [0.0, 1.0]],
        "disturbance_shape": [2],
        "price_probes": [
            {"price": 2.0, "damage": 30.0 * setpoint},
            {"price": 4.0, "damage": 10.0 * setpoint},
            {"price": 8.0, "damage": 5.0 * setpoint},
        ],
    }


def test_estimators_recover_linear_toy_breaking_price() -> None:
    hessian = rm.hessian_breaking_price(
        [[8.0, 0.0], [0.0, 1.0]],
        energy_metric=[[2.0, 0.0], [0.0, 1.0]],
    )
    bisection = rm.price_bisection_breaking_price(
        [
            {"price": 2.0, "damage": 3.0},
            {"price": 4.0, "damage": 1.0},
            {"price": 8.0, "damage": 0.5},
        ],
        setpoint=0.1,
        threshold_multiplier=10.0,
    )

    assert hessian["breaking_price"] == pytest.approx(4.0)
    assert bisection["breaking_price"] == pytest.approx(4.0)


def test_margin_row_uses_declared_trial_distribution_headline() -> None:
    row = rm.build_margin_row(
        {
            "row_id": "distribution",
            "damage_setpoint": 0.1,
            "lambda": 12.0,
            "hessian_trials": [
                {"trial_id": "a", "hessian": [[2.0]]},
                {"trial_id": "b", "hessian": [[4.0]]},
            ],
            "price_probe_trials": [
                {
                    "trial_id": "a",
                    "probes": [
                        {"price": 1.0, "damage": 2.0},
                        {"price": 2.0, "damage": 1.0},
                    ],
                },
                {
                    "trial_id": "b",
                    "probes": [
                        {"price": 2.0, "damage": 2.0},
                        {"price": 4.0, "damage": 1.0},
                    ],
                },
            ],
        },
        headline_quantile=1.0,
    )

    assert row["small_signal"]["breaking_price"] == pytest.approx(4.0)
    assert row["large_signal"]["breaking_price"] == pytest.approx(4.0)
    assert row["headline_lambda_margin"] == pytest.approx(3.0)
    assert [trial["trial_id"] for trial in row["small_signal"]["per_trial"]] == ["a", "b"]


def test_sidecar_reports_finite_decreasing_margins_for_setpoint_rows() -> None:
    rows = [
        rm.build_margin_row(_toy_row(row_id="low", setpoint=0.1, lambda_value=16.0)),
        rm.build_margin_row(_toy_row(row_id="high", setpoint=1.0, lambda_value=8.0)),
    ]
    sidecar = rm.build_robustness_margin_sidecar(rows=rows)

    assert rows[0]["headline_lambda_margin"] == pytest.approx(4.0)
    assert rows[1]["headline_lambda_margin"] == pytest.approx(2.0)
    assert rows[0]["status"] == "available"
    assert rows[1]["status"] == "available"
    assert sidecar["summary"]["prediction_check"]["status"] == "pass"


def test_recipe_is_registered_with_schema_and_eval_dependency() -> None:
    _registry()

    assert params_model_for(rm.ROBUSTNESS_MARGIN_ANALYSIS_TYPE) is rm.RobustnessMarginParams
    assert rm.EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE[rm.ROBUSTNESS_MARGIN_ANALYSIS_TYPE] == (
        "robustness_margin_rows",
    )
    bundle = load_analysis_bundle("rlrmp/robustness_margin")
    assert bundle.name == "robustness_margin"


def test_analysis_run_records_sidecar_through_feedbax_custody(tmp_path: Path) -> None:
    _registry()
    root = tmp_path / "feedbax_runs"
    states_artifact = store_evaluation_states_artifact(
        {"robustness_margin_rows": [_toy_row(row_id="a", setpoint=0.1, lambda_value=16.0)]},
        root=root,
        manifest_id="rlrmp-test-eval:robustness-margin",
    )
    eval_manifest = EvaluationRunManifest(
        id="rlrmp-test-eval:robustness-margin",
        status="completed",
        evaluation_spec=spec_payload(
            "EvaluationRunSpec",
            {
                "evaluation_type": "rlrmp.robustness_margin_fixture",
                "params": {},
                "inputs": [],
            },
        ),
        artifacts=[states_artifact],
        summary_metrics={"row_count": 1},
    )
    eval_manifest_path = write_manifest(eval_manifest, root=root)
    spec = AnalysisRunSpec(
        analysis_type=rm.ROBUSTNESS_MARGIN_ANALYSIS_TYPE,
        inputs=[
            ParentRef(
                kind="EvaluationRunManifest",
                id=eval_manifest.id,
                role="robustness_margin_rows",
                uri=str(eval_manifest_path),
            )
        ],
        params={
            "issue_id": "1ec6ae5",
            "scope": "unit_test",
            "threshold_multiplier": 10.0,
        },
    )

    manifest, _manifest_path = execute_analysis_run_spec(
        spec,
        root=root,
        issues=["1ec6ae5"],
        fig_dump_formats=("json",),
    )

    artifact = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "rlrmp-robustness-margin-sidecar"
    )
    payload = Path(artifact.uri).read_text(encoding="utf-8")
    assert rm.ROBUSTNESS_MARGIN_SCHEMA_VERSION in payload
    assert manifest.provenance.issues == ["1ec6ae5"]
    assert manifest.summary_metrics["artifact_count"] == 1

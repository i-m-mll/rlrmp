"""Tests for rlrmp Feedbax report-stage recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rlrmp
from feedbax.analysis.bundles import load_analysis_bundle
from feedbax.analysis.reports import execute_report_spec, registered_report_types
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ParentRef,
    ReportSpec,
    spec_payload,
    store_bytes_artifact,
    store_json_artifact,
    write_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry
from pydantic import ValidationError

from rlrmp.analysis import reports as rr
from rlrmp.runtime.params_models import params_model_for, registered_params_models

REPORT_BUNDLE_NAMES = (
    "rlrmp/robustness_phenotype",
    "rlrmp/output_feedback_bridge",
    "rlrmp/feedback_quality_lens",
    "rlrmp/gru_postrun",
    "rlrmp/training_diagnostics",
    "rlrmp/standard_matrix",
)

EXPECTED_REPORT_STAGE_PARAM_KEYS = {
    "schema_id",
    "schema_version",
    "title",
    "source_artifact_roles",
    "include_json_artifact",
    "narrative",
}


def _write_analysis_manifest(
    root: Path,
    *,
    manifest_id: str,
    artifact_role: str,
    artifact_text: str | None = None,
    artifact_json: dict[str, object] | None = None,
) -> tuple[AnalysisRunManifest, Path]:
    if artifact_json is not None:
        artifact = store_json_artifact(
            artifact_json,
            root=root,
            role=artifact_role,
            logical_name=f"{artifact_role}.json",
        )
    else:
        artifact = store_bytes_artifact(
            (artifact_text or "").encode("utf-8"),
            root=root,
            role=artifact_role,
            logical_name=f"{artifact_role}.md",
            media_type="text/markdown",
            suffix=".md",
        )
    manifest = AnalysisRunManifest(
        id=manifest_id,
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type="rlrmp.fixture", params={}).model_dump(mode="json"),
        ),
        artifacts=[artifact],
    )
    return manifest, write_manifest(manifest, root=root)


def test_rlrmp_report_recipes_register_with_lazy_package_registration() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)

    registered = set(registered_report_types())

    assert {
        rr.GRU_POSTRUN_REPORT_TYPE,
        rr.BRIDGE_CERTIFICATE_REPORT_TYPE,
        rr.FEEDBACK_QUALITY_LENS_REPORT_TYPE,
        rr.ROBUSTNESS_PHENOTYPE_REPORT_TYPE,
    } <= registered


@pytest.mark.parametrize(
    ("report_type", "expected_title", "expected_source_roles"),
    [
        (
            rr.GRU_POSTRUN_REPORT_TYPE,
            "GRU Postrun Report",
            [
                "rlrmp-gru-standard-certificate-note",
                "rlrmp-gru-objective-comparator-note",
                "rlrmp-gru-map-decomposition-note",
                "rlrmp-gru-perturbation-response-note",
                "rlrmp-gru-feedback-ablation-note",
            ],
        ),
        (
            rr.BRIDGE_CERTIFICATE_REPORT_TYPE,
            "Bridge Certificate Notes",
            [
                "rlrmp-bridge-standard-certificate",
            ],
        ),
        (
            rr.FEEDBACK_QUALITY_LENS_REPORT_TYPE,
            "Feedback-Quality Lens Summary",
            ["rlrmp-feedback-quality-lens"],
        ),
        (
            rr.ROBUSTNESS_PHENOTYPE_REPORT_TYPE,
            "Robustness Phenotype Report",
            ["rlrmp-robustness-phenotype-sidecar-note"],
        ),
    ],
)
def test_report_stage_params_defaults_match_recipe_literals(
    report_type: str,
    expected_title: str,
    expected_source_roles: list[str],
) -> None:
    model = rr.ReportStageParams.model_validate({"report_type": report_type})

    assert model.source_artifact_roles == expected_source_roles
    assert model.title == expected_title
    assert model.include_json_artifact is True
    assert model.narrative is None
    assert model.schema_id is None
    assert model.schema_version is None


def test_report_stage_params_validate_all_bundle_report_stage_payloads() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    report_payloads: dict[tuple[str, str], set[str]] = {}

    for bundle_name in REPORT_BUNDLE_NAMES:
        bundle = load_analysis_bundle(bundle_name, registry=registry)
        for stage in bundle.stages:
            if getattr(stage, "kind", None) != "report":
                continue
            params = dict(stage.local_params or {})
            report_payloads[(bundle_name, stage.name)] = set(params)
            model = rr.ReportStageParams.model_validate(
                {"report_type": stage.report_type, **params}
            )
            assert model.narrative == params["narrative"]

    assert report_payloads == {
        ("rlrmp/robustness_phenotype", "phenotype_report"): EXPECTED_REPORT_STAGE_PARAM_KEYS,
        ("rlrmp/gru_postrun", "postrun_report"): EXPECTED_REPORT_STAGE_PARAM_KEYS,
        ("rlrmp/gru_postrun", "bridge_certificate_report"): EXPECTED_REPORT_STAGE_PARAM_KEYS,
    }


def test_report_stage_params_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        rr.ReportStageParams.model_validate(
            {"report_type": rr.BRIDGE_CERTIFICATE_REPORT_TYPE, "unknown": True}
        )


def test_report_params_model_table_resolves_registered_report_recipes() -> None:
    rr.register_rlrmp_report_recipes(replace=True)

    for report_type in (
        rr.GRU_POSTRUN_REPORT_TYPE,
        rr.BRIDGE_CERTIFICATE_REPORT_TYPE,
        rr.FEEDBACK_QUALITY_LENS_REPORT_TYPE,
        rr.ROBUSTNESS_PHENOTYPE_REPORT_TYPE,
    ):
        assert params_model_for(report_type) is rr.ReportStageParams
        assert registered_params_models()[report_type] is rr.ReportStageParams
    with pytest.raises(KeyError):
        params_model_for("rlrmp.report.unknown")


def test_report_stage_defaults_have_single_model_owner() -> None:
    constructed = rr.report_stage_params(rr.BRIDGE_CERTIFICATE_REPORT_TYPE)
    constructed_model = rr.ReportStageParams.model_validate(
        {"report_type": rr.BRIDGE_CERTIFICATE_REPORT_TYPE, **constructed}
    )
    consumed_model = rr._validated_stage_params(
        ReportSpec(
            report_type=rr.BRIDGE_CERTIFICATE_REPORT_TYPE,
            params={
                "schema_id": constructed["schema_id"],
                "schema_version": constructed["schema_version"],
            },
        )
    )

    assert consumed_model.source_artifact_roles == constructed_model.source_artifact_roles
    assert consumed_model.title == constructed_model.title
    assert consumed_model.include_json_artifact == constructed_model.include_json_artifact


def test_bridge_report_constructs_certificate_and_failure_tables(tmp_path: Path) -> None:
    rr.register_rlrmp_report_recipes(replace=True)
    analysis_manifest, analysis_path = _write_analysis_manifest(
        tmp_path,
        manifest_id="rlrmp-test-analysis:bridge-note",
        artifact_role="rlrmp-bridge-standard-certificate",
        artifact_json={
            "rows": [
                {
                    "spec": {
                        "run_id": "row-a",
                        "training_distribution": "robust",
                        "evaluation_lane": "deterministic",
                        "parameters": {
                            "evaluation_lens": "nominal_clean",
                            "certificate_mode": "augmented_linear",
                        },
                    },
                    "status": "failed_standard_certificate",
                    "metrics": {"objective_ratio_to_reference": 1.2},
                    "certificate_components": [
                        {
                            "name": "state_weighted_action_mismatch",
                            "status": "available",
                            "summary": {
                                "mismatch_ratio_mean": 0.31,
                                "aggregate_mismatch_ratio": 0.73,
                            },
                        },
                        {
                            "name": "closed_loop_transition_mismatch",
                            "status": "available",
                            "summary": {"mismatch_ratio_mean": 0.4},
                        },
                        {
                            "name": "value_policy_gap",
                            "status": "available",
                            "summary": {"gap_ratio_mean": 0.5},
                        },
                        {
                            "name": "bellman_hessian_residual",
                            "status": "available",
                            "summary": {"residual_ratio_mean": 0.3},
                        },
                        {
                            "name": "deterministic_exact_l2_and_gamma_sidecar",
                            "status": "available",
                            "summary": {
                                "exact_l2_cost_ratio_to_lqr": 0.8,
                                "lambda_over_gamma_squared": 0.7,
                            },
                        },
                        {
                            "name": "gain_diagnostic_sidecar",
                            "status": "available",
                            "summary": {"gain_relative_error": 0.9},
                        },
                    ],
                }
            ],
            "failure_decomposition": {
                "rows": [
                    {
                        "run_id": "row-a",
                        "classification": {"classification": "sidecar_improving_non_equivalent"},
                        "objective": {
                            "learned_objective": 12.0,
                            "reference_objective": 10.0,
                            "learned_gradient_norm": 2.0,
                            "reference_gradient_norm": 0.0,
                            "learned_projected_gradient_norm": 1.0,
                            "reference_projected_gradient_norm": 0.0,
                        },
                        "interpolation": [
                            {"alpha": 0.0, "training_objective_ratio_to_reference": 1.2},
                            {"alpha": 1.0, "training_objective_ratio_to_reference": 1.0},
                        ],
                        "gain_error_decomposition": {
                            "strong_fraction_mean": 0.2,
                            "weak_or_unvisited_fraction_mean": 0.8,
                        },
                    }
                ]
            },
        },
    )

    report_manifest, _path = execute_report_spec(
        ReportSpec(
            report_type=rr.BRIDGE_CERTIFICATE_REPORT_TYPE,
            inputs=[
                ParentRef(
                    kind="AnalysisRunManifest",
                    id=analysis_manifest.id,
                    role="analysis_run",
                    uri=str(analysis_path),
                )
            ],
            params=rr.report_stage_params(rr.BRIDGE_CERTIFICATE_REPORT_TYPE),
        ),
        root=tmp_path,
        issues=["4dad1b6"],
    )

    assert report_manifest.status == "completed"
    render = next(
        artifact for artifact in report_manifest.artifacts if artifact.role == "report_render"
    )
    render_text = Path(render.uri).read_text(encoding="utf-8")
    assert (
        "| row-a | failed_standard_certificate | robust | nominal_clean | "
        "augmented_linear | 1.2 | 0.31 | 0.73 | 0.4 | 0.5 | 0.3 | "
        "L2=0.8; gamma=0.7 | 0.9 |"
    ) in render_text
    assert rr.ACTION_BELLMAN_ANNOTATION in render_text
    assert rr.GAIN_DIAGNOSTIC_ANNOTATION in render_text
    assert "sidecar_improving_non_equivalent" in render_text
    assert rr.FAILURE_DECOMPOSITION_ANNOTATION in render_text
    assert render.media_type == "text/markdown"
    assert render.sha256
    summary = next(
        artifact
        for artifact in report_manifest.artifacts
        if artifact.role == rr.BRIDGE_CERTIFICATE_REPORT_RENDER_ROLE
    )
    payload = json.loads(Path(summary.uri).read_text(encoding="utf-8"))
    assert payload["sections"][0]["json"]["rows"][0]["spec"]["run_id"] == "row-a"


def test_feedback_quality_report_renders_json_payload(tmp_path: Path) -> None:
    rr.register_rlrmp_report_recipes(replace=True)
    analysis_manifest, analysis_path = _write_analysis_manifest(
        tmp_path,
        manifest_id="rlrmp-test-analysis:feedback-quality",
        artifact_role="rlrmp-feedback-quality-lens",
        artifact_json={
            "schema_id": "rlrmp.feedback_quality_lens",
            "schema_version": "rlrmp.feedback_quality_lens.v1",
            "outputs": {"evaluation_diagnostics": {"status": "materialized"}},
        },
    )

    report_manifest, _path = execute_report_spec(
        ReportSpec(
            report_type=rr.FEEDBACK_QUALITY_LENS_REPORT_TYPE,
            inputs=[
                ParentRef(
                    kind="AnalysisRunManifest",
                    id=analysis_manifest.id,
                    role="analysis_run",
                    uri=str(analysis_path),
                )
            ],
            params=rr.report_stage_params(rr.FEEDBACK_QUALITY_LENS_REPORT_TYPE),
        ),
        root=tmp_path,
    )

    render = next(
        artifact for artifact in report_manifest.artifacts if artifact.role == "report_render"
    )
    render_text = Path(render.uri).read_text(encoding="utf-8")
    assert '"schema_id": "rlrmp.feedback_quality_lens"' in render_text
    assert '"status": "materialized"' in render_text

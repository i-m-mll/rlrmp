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
                "rlrmp-gru-standard-certificate-note",
                "rlrmp-bridge-standard-certificate-note",
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


def test_bridge_report_render_matches_fixture_note_section(tmp_path: Path) -> None:
    rr.register_rlrmp_report_recipes(replace=True)
    analysis_manifest, analysis_path = _write_analysis_manifest(
        tmp_path,
        manifest_id="rlrmp-test-analysis:bridge-note",
        artifact_role="rlrmp-gru-standard-certificate-note",
        artifact_text="standard certificate note\n",
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
    render = next(artifact for artifact in report_manifest.artifacts if artifact.role == "report_render")
    render_text = Path(render.uri).read_text(encoding="utf-8")
    assert "standard certificate note\n" in render_text
    assert render.media_type == "text/markdown"
    assert render.sha256
    summary = next(
        artifact
        for artifact in report_manifest.artifacts
        if artifact.role == rr.BRIDGE_CERTIFICATE_REPORT_RENDER_ROLE
    )
    payload = json.loads(Path(summary.uri).read_text(encoding="utf-8"))
    assert payload["sections"][0]["text"] == "standard certificate note\n"


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

    render = next(artifact for artifact in report_manifest.artifacts if artifact.role == "report_render")
    render_text = Path(render.uri).read_text(encoding="utf-8")
    assert '"schema_id": "rlrmp.feedback_quality_lens"' in render_text
    assert '"status": "materialized"' in render_text

"""Tests for rlrmp Feedbax report-stage recipes."""

from __future__ import annotations

import json
from pathlib import Path

import rlrmp
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

from rlrmp.analysis import reports as rr


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

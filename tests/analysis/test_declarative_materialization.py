"""Tests for Feedbax declarative rlrmp materialization recipes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rlrmp
from feedbax.analysis.bundles import execute_analysis_bundle, load_analysis_bundle
from feedbax.analysis.materialization import ContextMaterializer
from feedbax.analysis.specs import execute_analysis_run_spec, unregister_analysis_recipe
from feedbax.manifest import AnalysisRunSpec, TrainingRunManifest, load_manifest, write_manifest
from feedbax.plugins.registry import ExperimentRegistry

from rlrmp.analysis import declarative_materialization as dm


def _artifact_roles(manifest) -> set[str]:
    return {artifact.role for artifact in manifest.artifacts}


def _unregister_declarative_recipes() -> None:
    unregister_analysis_recipe(dm.GRU_STANDARD_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.ROBUSTNESS_PHENOTYPE_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE)


def test_declarative_recipes_use_feedbax_context_materializers() -> None:
    standard = dm.gru_standard_certificate_recipe(
        dm.gru_standard_certificate_spec(),
        Path("."),
        (),
    )
    evaluation = dm.gru_evaluation_diagnostics_recipe(
        dm.gru_evaluation_diagnostics_spec(
            experiment="unitexp",
            run_ids=["unit_run"],
        ),
        Path("."),
        (),
    )
    rollout_recovery = dm.output_feedback_rollout_recovery_recipe(
        dm.output_feedback_rollout_recovery_spec(),
        Path("."),
        (),
    )

    assert isinstance(standard.analyses["gru_standard_certificate"], ContextMaterializer)
    assert isinstance(evaluation.analyses["gru_evaluation_diagnostics"], ContextMaterializer)
    feedback_quality = dm.feedback_quality_lens_recipe(
        dm.feedback_quality_lens_spec(
            experiment="unitexp",
            run_ids=["unit_run"],
        ),
        Path("."),
        (),
    )
    assert isinstance(
        rollout_recovery.analyses["output_feedback_rollout_recovery"],
        ContextMaterializer,
    )
    assert isinstance(feedback_quality.analyses["feedback_quality_lens"], ContextMaterializer)


def test_output_feedback_bridge_bundle_resource_loads() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)

    bundle = load_analysis_bundle("rlrmp/output_feedback_bridge", registry=registry)

    assert bundle.name == "output_feedback_bridge"
    assert bundle.metadata["bundle_family"] == "rlrmp/output_feedback_bridge"
    assert bundle.templates[0].analysis_type == dm.OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE
    assert bundle.templates[0].requested_outputs == ["output_feedback_rollout_recovery"]


def test_feedback_quality_lens_bundle_resource_loads() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)

    bundle = load_analysis_bundle("rlrmp/feedback_quality_lens", registry=registry)

    assert bundle.name == "feedback_quality_lens"
    assert bundle.metadata["bundle_family"] == "rlrmp/feedback_quality_lens"
    assert bundle.templates[0].analysis_type == dm.FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE
    assert bundle.templates[0].requested_outputs == ["feedback_quality_lens"]


def test_feedback_quality_lens_bundle_executes_fixture_and_groups_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    bundle = load_analysis_bundle("rlrmp/feedback_quality_lens", registry=registry)
    repo_root = tmp_path / "repo"
    feedbax_root = tmp_path / "feedbax_runs"
    run_id = "rlrmp-test-training-run:feedback-quality"
    run_manifest = TrainingRunManifest(
        id=run_id,
        job_id="feedback-quality-fixture",
        status="completed",
        metadata={
            "feedback_quality_candidate": True,
            "rlrmp_experiment": "5f70333",
        },
    )
    write_manifest(run_manifest, root=feedbax_root, index=False)
    monkeypatch.setattr(dm, "REPO_ROOT", repo_root)

    plan = dm.plan_gru_postrun_materialization(
        experiment="5f70333",
        run_ids=(run_id,),
        repo_root=repo_root,
    )
    for path, payload in (
        (
            plan.evaluation_manifest_path,
            {"schema_version": "rlrmp.gru_evaluation_diagnostics.v1", "runs": {}},
        ),
        (
            plan.perturbation_response_json_path,
            {"schema_version": "rlrmp.gru_perturbation_bank.v3", "runs": {}},
        ),
        (
            plan.feedback_ablation_json_path,
            {"schema_version": "rlrmp.gru_feedback_ablation.v1", "rows": []},
        ),
        (
            plan.objective_comparator_json_path,
            {"schema_version": "rlrmp.objective_comparator_sidecar.v6", "rows": []},
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    for path in (
        plan.perturbation_response_note_path,
        plan.feedback_ablation_note_path,
        plan.objective_comparator_note_path,
    ):
        path.write_text("# fixture\n", encoding="utf-8")

    plan.evaluation_bulk_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(plan.evaluation_bulk_dir / "unit_rollout.npz", x=np.ones((1,)))
    plan.perturbation_response_bulk_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        plan.perturbation_response_bulk_dir / "unit_perturbation.npz", x=np.ones((1,))
    )
    norm_manifest = (
        repo_root
        / "results"
        / "5f70333"
        / "notes"
        / "gru_perturbation_response_norm_plots_validation_selected_manifest.json"
    )
    norm_manifest.write_text(
        json.dumps({"schema_version": "rlrmp.gru_perturbation_response_norm_plots.v1"}) + "\n",
        encoding="utf-8",
    )
    norm_note = norm_manifest.with_name(
        "gru_perturbation_response_norm_plots_validation_selected.md"
    )
    norm_note.write_text("# norm plots\n", encoding="utf-8")
    norm_fig_dir = (
        repo_root
        / "_artifacts"
        / "5f70333"
        / "figures"
        / ("perturbation_response_norms_validation_selected")
    )
    norm_fig_dir.mkdir(parents=True, exist_ok=True)
    (norm_fig_dir / "figure.html").write_text("<html></html>\n", encoding="utf-8")

    outputs = execute_analysis_bundle(
        bundle,
        root=feedbax_root,
        run_ids=[run_id],
        issues=["af77a06"],
        fig_dump_formats=("json",),
    )

    assert len(outputs) == 1
    _expansion, manifest, manifest_path = outputs[0]
    assert manifest_path.exists()
    assert manifest.status == "completed"
    assert manifest.provenance.issues == ["af77a06"]
    roles = _artifact_roles(manifest)
    assert "rlrmp-feedback-quality-lens" in roles
    assert "rlrmp-feedback-quality-perturbation-response-bulk" in roles
    assert "rlrmp-feedback-quality-response-norm-figure" in roles
    assert "rlrmp-feedback-quality-perturbation-calibration-manifest" not in roles

    payload_ref = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "rlrmp-feedback-quality-lens"
    )
    payload = json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "rlrmp.feedback_quality_lens.v1"
    assert payload["bundle_contract"]["artifact_custody"] == "feedbax.AnalysisRunManifest"
    assert payload["outputs"]["perturbation_response"]["status"] == "materialized"
    assert payload["outputs"]["response_norm_plots"]["status"] == "materialized"
    assert payload["outputs"]["perturbation_calibration"]["status"] == "unavailable"
    assert (
        "feedback_quality_perturbation_response_bulk"
        in payload["outputs"]["perturbation_response"]["artifact_group_ids"]
    )
    bulk_artifact = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "rlrmp-feedback-quality-perturbation-response-bulk"
    )
    assert bulk_artifact.metadata["artifact_group"]["id"] == (
        "feedback_quality_perturbation_response_bulk"
    )
    assert load_manifest(manifest_path).id == manifest.id


def test_feedback_quality_lens_records_skipped_and_not_applicable_status(
    tmp_path: Path,
) -> None:
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = dm.feedback_quality_lens_spec(
            experiment="unitexp",
            run_ids=["unit_run"],
            include_feedback_ablation=False,
            not_applicable_components=["perturbation_calibration"],
            repo_root=tmp_path / "repo",
        )

        manifest, _path = execute_analysis_run_spec(spec, root=tmp_path, issues=["af77a06"])

        payload_ref = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-feedback-quality-lens"
        )
        payload = json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))
        assert payload["outputs"]["feedback_ablation"]["status"] == "skipped"
        assert payload["outputs"]["perturbation_calibration"]["status"] == "not_applicable"
        assert payload["outputs"]["evaluation_diagnostics"]["status"] == "unavailable"
    finally:
        _unregister_declarative_recipes()


def test_gru_standard_recipe_records_opaque_certificate_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_note = tmp_path / "legacy" / "gru_standard_certificates.md"
    legacy_manifest = tmp_path / "legacy" / "gru_standard_certificates_manifest.json"

    def fake_materialize(**kwargs):
        return {
            "format": "rlrmp.cs_gru_standard_certificates.v1",
            "issue": kwargs["materializer_issue_id"],
            "source_issue": kwargs["experiment"],
            "rows": [
                {
                    "spec": {
                        "run_id": f"{kwargs['run_ids'][0]}__nominal_clean",
                        "architecture": "gru",
                    },
                    "certificate_components": [],
                }
            ],
            "summary": {"n_rows": 1},
            "failure_decomposition": {"rows": []},
        }

    def fake_write(result, *, note_path, manifest_path, regeneration_spec_path=None, repo_root):
        del regeneration_spec_path, repo_root
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("standard certificate note\n", encoding="utf-8")
        payload = {**result, "regeneration_spec": "results/unit/regeneration.json"}
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(dm, "materialize_gru_standard_result", fake_materialize)
    monkeypatch.setattr(dm, "write_gru_standard_result", fake_write)
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = AnalysisRunSpec(
            analysis_type=dm.GRU_STANDARD_ANALYSIS_TYPE,
            params={
                "run_ids": ["unit_run"],
                "experiment": "unitexp",
                "materializer_issue_id": "103db99",
                "load_models": False,
                "note_output": str(legacy_note),
                "manifest_output": str(legacy_manifest),
            },
        )

        manifest, path = execute_analysis_run_spec(spec, root=tmp_path, issues=["103db99"])

        assert path.exists()
        assert manifest.status == "completed"
        assert manifest.analysis_spec.inline["analysis_type"] == dm.GRU_STANDARD_ANALYSIS_TYPE
        assert manifest.summary_metrics["analysis_count"] == 1
        assert "rlrmp-bridge-standard-certificate" in _artifact_roles(manifest)
        assert "rlrmp-bridge-standard-certificate-manifest" in _artifact_roles(manifest)
        assert "rlrmp-bridge-standard-certificate-note" in _artifact_roles(manifest)
        payload_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-bridge-standard-certificate"
        )
        payload = json.loads(Path(payload_artifact.uri).read_text(encoding="utf-8"))
        assert payload["format"] == "rlrmp.cs_gru_standard_certificates.v1"
        assert payload["declarative_analysis"]["schema_owner"] == "rlrmp"
        assert legacy_manifest.exists()
        assert load_manifest(path).id == manifest.id
    finally:
        _unregister_declarative_recipes()


def test_gru_evaluation_recipe_groups_bulk_npz_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_path = tmp_path / "diagnostics.json"
    bulk_dir = tmp_path / "bulk"

    def fake_materialize(**kwargs):
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["bulk_dir"].mkdir(parents=True, exist_ok=True)
        bulk_path = kwargs["bulk_dir"] / "unit_run.npz"
        np.savez_compressed(bulk_path, command=np.array([1.0]), position=np.array([2.0]))
        manifest = {
            "schema_version": "rlrmp.gru_evaluation_diagnostics.v1",
            "issue": kwargs["experiment"],
            "scope": "post_hoc_evaluation_non_certificate_diagnostics",
            "runs": {
                "unit_run": {
                    "bulk_arrays": {
                        "path": str(bulk_path),
                        "format": "np.savez_compressed",
                        "arrays": ["command", "position"],
                    }
                }
            },
        }
        kwargs["output_path"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    monkeypatch.setattr(dm, "materialize_gru_evaluation_diagnostics", fake_materialize)
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = AnalysisRunSpec(
            analysis_type=dm.GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE,
            params={
                "experiment": "unitexp",
                "run_ids": ["unit_run"],
                "repo_root": str(repo_root),
                "output_path": str(output_path),
                "bulk_dir": str(bulk_dir),
            },
        )

        manifest, path = execute_analysis_run_spec(spec, root=tmp_path, issues=["103db99"])

        assert path.exists()
        assert manifest.status == "completed"
        assert "rlrmp-gru-evaluation-diagnostics" in _artifact_roles(manifest)
        assert "rlrmp-gru-evaluation-diagnostics-manifest" in _artifact_roles(manifest)
        assert "rlrmp-gru-evaluation-diagnostics-bulk" in _artifact_roles(manifest)
        bulk_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-gru-evaluation-diagnostics-bulk"
        )
        assert bulk_artifact.logical_name == "bulk/unit_run.npz"
        assert bulk_artifact.metadata["artifact_group"]["id"] == ("gru_evaluation_diagnostics_bulk")
        assert bulk_artifact.metadata["artifact_group"]["member_role"] == "rollout_arrays"
        payload_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-gru-evaluation-diagnostics"
        )
        payload = json.loads(Path(payload_artifact.uri).read_text(encoding="utf-8"))
        assert payload["schema_version"] == "rlrmp.gru_evaluation_diagnostics.v1"
        assert payload["declarative_analysis"]["artifact_owner"] == ("feedbax.AnalysisRunManifest")
        assert load_manifest(path).id == manifest.id
    finally:
        _unregister_declarative_recipes()


def test_output_feedback_rollout_recovery_recipe_records_manifest_and_bulk_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    note_output = repo_root / "results" / "7a459bb" / "notes" / "rollout.md"
    manifest_output = repo_root / "results" / "7a459bb" / "notes" / "rollout_manifest.json"
    artifact_output = (
        repo_root / "_artifacts" / "7a459bb" / "output_feedback_rollout_recovery" / "rollout.npz"
    )

    def fake_write_outputs(**kwargs):
        kwargs["note_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["manifest_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["artifact_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["note_path"].write_text("# rollout recovery\n", encoding="utf-8")
        payload = {
            "issue": kwargs["issue_id"],
            "scope": "output-feedback bridge rollout recovery",
            "fits": [{"label": "unit__scratch"}],
            "tracked_note": "results/7a459bb/notes/rollout.md",
            "tracked_manifest": "results/7a459bb/notes/rollout_manifest.json",
            "artifact_npz": "_artifacts/7a459bb/output_feedback_rollout_recovery/rollout.npz",
            "artifact_npz_keys": ["gain", "rollout"],
            "rerun_metadata": {
                "discretization": kwargs["discretization"],
                "lane": kwargs["lane"],
            },
        }
        kwargs["manifest_path"].write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(kwargs["artifact_path"], gain=np.ones((1,)), rollout=np.zeros((1,)))
        return payload

    monkeypatch.setattr(
        dm,
        "write_output_feedback_rollout_recovery_outputs",
        fake_write_outputs,
    )
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = dm.output_feedback_rollout_recovery_spec(
            issue_id="7a459bb",
            discretization="jaxley",
            lane="analysis",
            note_output=note_output,
            manifest_output=manifest_output,
            artifact_output=artifact_output,
            repo_root=repo_root,
        )

        manifest, path = execute_analysis_run_spec(spec, root=tmp_path, issues=["c4416c5"])

        assert path.exists()
        assert manifest.status == "completed"
        assert manifest.analysis_spec.inline["analysis_type"] == (
            dm.OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE
        )
        assert manifest.provenance.issues == ["c4416c5"]
        assert "rlrmp-output-feedback-rollout-recovery" in _artifact_roles(manifest)
        assert "rlrmp-output-feedback-rollout-recovery-manifest" in _artifact_roles(manifest)
        assert "rlrmp-output-feedback-rollout-recovery-note" in _artifact_roles(manifest)
        assert "rlrmp-output-feedback-rollout-recovery-bulk" in _artifact_roles(manifest)
        bulk_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-output-feedback-rollout-recovery-bulk"
        )
        assert bulk_artifact.logical_name == "bulk/output_feedback_rollout_recovery.npz"
        assert bulk_artifact.metadata["artifact_group"]["id"] == (
            "output_feedback_rollout_recovery_bulk"
        )
        assert bulk_artifact.metadata["artifact_group"]["member_role"] == (
            "rollout_recovery_arrays"
        )
        payload_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-output-feedback-rollout-recovery"
        )
        payload = json.loads(Path(payload_artifact.uri).read_text(encoding="utf-8"))
        assert payload["issue"] == "7a459bb"
        assert payload["declarative_analysis"]["schema_owner"] == "rlrmp"
        assert load_manifest(path).id == manifest.id
    finally:
        _unregister_declarative_recipes()

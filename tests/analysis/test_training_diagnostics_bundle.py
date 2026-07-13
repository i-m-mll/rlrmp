"""Tests for training-diagnostics analysis bundle execution."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import rlrmp
from feedbax.analysis.bundles import execute_analysis_bundle, load_analysis_bundle
from feedbax.analysis.specs import ResolvedAnalysisInput, execute_analysis_run_spec
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    TrainingRunManifest,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.training import TrainingDiagnostics
from rlrmp.analysis import training_diagnostics as td


def _registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    return registry


def _write_training_run(
    root: Path,
    run_dir: Path,
    *,
    run_id: str = "rlrmp-test-training-run:diagnostics",
) -> tuple[TrainingRunManifest, Path]:
    run_dir.mkdir(parents=True)
    np.savez(
        run_dir / "training_diagnostics.npz",
        batch_index=np.array([0, 1, 2]),
        train_loss__total=np.array([[3.0], [2.0], [1.0]]),
        validation_loss__total=np.array([[4.0], [3.0], [2.0]]),
        pgd_broad_epsilon_diagnostic_sampled=np.array([False, True, True]),
        pgd_broad_epsilon_inner_objective_improvement=np.array([[np.nan], [0.5], [0.75]]),
        pgd_broad_epsilon_inner_objective_final_endpoint_gap=np.array([[np.nan], [0.0], [0.0]]),
        pgd_broad_epsilon_epsilon_norm_radius_ratio_mean=np.array([[np.nan], [0.5], [0.75]]),
    )
    (run_dir / "training_diagnostics.json").write_text(
        json.dumps({"completed_batches": 3}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "training_summary.json").write_text(
        json.dumps({"completed_batches": 3, "stopped_early_for_checkpoint_gate": False}) + "\n",
        encoding="utf-8",
    )
    manifest = TrainingRunManifest(
        id=run_id,
        job_id="diagnostics-fixture",
        status="completed",
        training_spec=spec_payload(
            "TrainingSpec",
            {
                "training_diagnostics": {"enabled": True},
                "issue": "0e3223d",
            },
        ),
        artifacts=[
            ArtifactRef(
                role="training_diagnostics_npz",
                logical_name="training_diagnostics.npz",
                uri=str(run_dir / "training_diagnostics.npz"),
                media_type="application/x-numpy-npz",
            ),
            ArtifactRef(
                role="training_diagnostics_json",
                logical_name="training_diagnostics.json",
                uri=str(run_dir / "training_diagnostics.json"),
                media_type="application/json",
            ),
            ArtifactRef(
                role="training_summary_json",
                logical_name="training_summary.json",
                uri=str(run_dir / "training_summary.json"),
                media_type="application/json",
            ),
        ],
    )
    path = write_manifest(manifest, root=root, index=False)
    return manifest, path


def _write_native_training_run(
    root: Path,
    run_dir: Path,
    *,
    run_id: str = "rlrmp-test-training-run:native-diagnostics",
) -> tuple[TrainingRunManifest, Path, Path]:
    run_dir.mkdir(parents=True)
    diagnostics = TrainingDiagnostics(
        manifest_id=run_id,
        run_id="native-diagnostics-job",
        terminal_status="completed",
        completed_batches=11,
        segment_completed_batches=3,
        cumulative_completed_batches=11,
        seeds=[7, 19],
        lr_trace=[
            {"step": 8, "learning_rate": 3e-4},
            {"step": 11, "learning_rate": 1e-4},
        ],
        checkpoint_coordinates=[2, 3],
        checkpoint_transactions=[
            {
                "transaction_id": "checkpoint-10",
                "completed_batches": 2,
                "cumulative_completed_batches": 10,
                "coordinate": {
                    "run_id": run_id,
                    "phase": "train",
                    "program_step": 10,
                    "outer_step": 4,
                    "inner_step": 2,
                    "completed_barrier": "batch",
                    "schedule_origin_step": 8,
                    "metrics": {"loss": 1.25},
                },
            },
            {
                "transaction_id": "checkpoint-11",
                "completed_batches": 3,
                "cumulative_completed_batches": 11,
                "coordinate": {
                    "run_id": run_id,
                    "phase": "train",
                    "program_step": 11,
                    "outer_step": 5,
                    "inner_step": 0,
                    "completed_barrier": "terminal",
                    "schedule_origin_step": 8,
                },
            },
        ],
        resume_context={
            "schedule_origin_step": 8,
            "current_step": 8,
            "optimizer_count_at_current_step": 8,
        },
        optimizer_build_context={
            "schedule_origin_step": 8,
            "current_step": 11,
            "optimizer_count_at_current_step": 11,
        },
        metadata={"segment_start_batch": 8},
    )
    diagnostics_path = run_dir / "training-diagnostics.json"
    diagnostics_path.write_text(
        diagnostics.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostics_bytes = diagnostics_path.read_bytes()
    manifest = TrainingRunManifest(
        id=run_id,
        job_id="native-diagnostics-job",
        status="completed",
        completed_batches=11,
        training_spec=spec_payload(
            "TrainingSpec",
            {"training_diagnostics": {"enabled": True}, "issue": "8776106"},
        ),
        artifacts=[
            ArtifactRef(
                role="training_diagnostics",
                logical_name="training-diagnostics.json",
                artifact_id="artifact://sha256/" + sha256(diagnostics_bytes).hexdigest(),
                sha256=sha256(diagnostics_bytes).hexdigest(),
                size_bytes=len(diagnostics_bytes),
                uri=str(diagnostics_path),
                media_type="application/json",
                metadata={
                    "schema_id": "feedbax.manifest.training_diagnostics",
                    "schema_version": "feedbax.manifest.training_diagnostics.v1",
                },
            )
        ],
    )
    path = write_manifest(manifest, root=root, index=False)
    return manifest, path, diagnostics_path


def test_training_diagnostics_bundle_executes_manifest_backed_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = _registry()
    bundle = load_analysis_bundle("rlrmp/training_diagnostics", registry=registry)
    run_manifest, _run_manifest_path, diagnostics_path = _write_native_training_run(
        tmp_path / "feedbax_runs",
        tmp_path / "bulk" / "run-a",
    )
    fake_repo_root = tmp_path / "repo"
    monkeypatch.setattr(td, "REPO_ROOT", fake_repo_root)

    outputs = execute_analysis_bundle(
        bundle,
        root=tmp_path / "feedbax_runs",
        run_ids=[run_manifest.id],
        issues=["0e3223d"],
        fig_dump_formats=("json",),
    )

    assert len(outputs) == 1
    _expansion, manifest, manifest_path = outputs[0]
    assert manifest_path.exists()
    assert manifest.status == "completed"
    assert manifest.provenance.issues == ["0e3223d"]
    assert manifest.inputs[0].id == run_manifest.id
    assert manifest.metadata["bundle"]["name"] == "training_diagnostics"
    roles = {artifact.role for artifact in manifest.artifacts}
    assert roles == {"analysis_notes", "training_diagnostics_summary_json"}

    json_path = (
        fake_repo_root
        / "results"
        / "training_diagnostics"
        / "notes"
        / ("training_diagnostics_summary.json")
    )
    markdown_path = (
        fake_repo_root
        / "results"
        / "training_diagnostics"
        / "notes"
        / ("training_diagnostics_summary.md")
    )
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == td.TRAINING_DIAGNOSTICS_SCHEMA_VERSION
    summary = payload["summaries"][0]
    native = summary["training_diagnostics"]
    assert summary["source_format"] == "feedbax_training_diagnostics"
    assert summary["training_manifest_id"] == run_manifest.id
    assert summary["job_id"] == run_manifest.job_id
    assert summary["run_id"] == run_manifest.job_id
    assert summary["diagnostics_artifact"] == run_manifest.artifacts[0].model_dump(
        mode="json", exclude_none=True
    )
    assert native["seeds"] == [7, 19]
    assert native["lr_trace"] == [
        {"step": 8, "learning_rate": 3e-4},
        {"step": 11, "learning_rate": 1e-4},
    ]
    assert native["checkpoint_coordinates"] == [2, 3]
    assert [item["transaction_id"] for item in native["checkpoint_transactions"]] == [
        "checkpoint-10",
        "checkpoint-11",
    ]
    assert native["checkpoint_transactions"][0]["coordinate"] == {
        "run_id": run_manifest.id,
        "phase": "train",
        "program_step": 10,
        "outer_step": 4,
        "inner_step": 2,
        "completed_barrier": "batch",
        "schedule_origin_step": 8,
        "metrics": {"loss": 1.25},
    }
    assert native["segment_completed_batches"] == 3
    assert native["cumulative_completed_batches"] == 11
    assert native["terminal_status"] == "completed"
    assert native["resume_context"]["schedule_origin_step"] == 8
    assert native["optimizer_build_context"]["current_step"] == 11
    assert not list(diagnostics_path.parent.glob("*.npz"))
    assert not list(diagnostics_path.parent.glob("*converted*"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "<!-- AUTO-GENERATED: training_diagnostics_summary -->" in markdown
    assert summary["latest_batch_index"] == 10
    assert "| native-diagnostics-job | yes | 10 |" in markdown
    assert load_manifest(manifest_path).id == manifest.id


def test_training_diagnostics_analysis_accepts_experiment_hash_routing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _registry()
    run_manifest, run_manifest_path = _write_training_run(
        tmp_path / "feedbax_runs",
        tmp_path / "bulk" / "run-b",
        run_id="rlrmp-test-training-run:diagnostics-routed",
    )
    fake_repo_root = tmp_path / "repo"
    monkeypatch.setattr(td, "REPO_ROOT", fake_repo_root)
    notes_path = (
        fake_repo_root / "results" / "0e3223d" / "notes" / ("training_diagnostics_summary.md")
    )
    notes_path.parent.mkdir(parents=True)
    notes_path.write_text("# Handwritten context\n\nKeep me.\n", encoding="utf-8")

    spec = AnalysisRunSpec(
        analysis_type=td.TRAINING_DIAGNOSTICS_ANALYSIS_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id=run_manifest.id,
                role="training_run",
                uri=str(run_manifest_path),
            )
        ],
        params={
            "experiment": "0e3223d",
            "topic": "training_diagnostics_summary",
            "note_marker": "training_diagnostics_summary",
        },
    )
    manifest, _manifest_path = execute_analysis_run_spec(
        spec,
        root=tmp_path / "feedbax_runs",
        issues=["0e3223d"],
        fig_dump_formats=("json",),
    )

    json_path = (
        fake_repo_root / "results" / "0e3223d" / "notes" / ("training_diagnostics_summary.json")
    )
    assert json_path.exists()
    assert notes_path.read_text(encoding="utf-8").startswith("# Handwritten context")
    assert manifest.summary_metrics["artifact_count"] == 2
    for artifact in manifest.artifacts:
        assert artifact.metadata["experiment"] == "0e3223d"


def test_generic_training_diagnostics_role_does_not_guess_npz_payload(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _write_training_run(
        tmp_path / "feedbax_runs",
        tmp_path / "bulk" / "legacy-generic-role",
    )
    npz_ref = manifest.artifacts[0].model_copy(update={"role": "training_diagnostics"})
    manifest = manifest.model_copy(update={"artifacts": [npz_ref, *manifest.artifacts[1:]]})
    resolved = ResolvedAnalysisInput(
        ref=ParentRef(kind="TrainingRunManifest", id=manifest.id, role="training_run"),
        manifest=manifest,
        path=manifest_path,
    )

    with pytest.raises(ValueError, match="unsupported (media_type|schema_id)"):
        td._summary_for_input(resolved, root=tmp_path / "feedbax_runs")


@pytest.mark.parametrize(
    "drift",
    [
        "schema_version",
        "payload_schema_version",
        "manifest_id",
        "run_id",
        "terminal_status",
        "missing_completed_batches",
        "completed_batches",
    ],
)
def test_native_training_diagnostics_rejects_contract_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    manifest, manifest_path, diagnostics_path = _write_native_training_run(
        tmp_path / "feedbax_runs",
        tmp_path / "bulk" / drift,
    )
    artifact = manifest.artifacts[0]
    if drift == "schema_version":
        artifact = artifact.model_copy(
            update={
                "metadata": {
                    **artifact.metadata,
                    "schema_version": "feedbax.manifest.training_diagnostics.v0",
                }
            }
        )
        match = "unsupported schema_version"
    elif drift == "missing_completed_batches":
        manifest = manifest.model_copy(update={"completed_batches": None})
        match = "require parent manifest completed_batches"
    elif drift == "completed_batches":
        manifest = manifest.model_copy(update={"completed_batches": 12})
        match = "completed_batches does not match"
    else:
        diagnostics = TrainingDiagnostics.model_validate_json(
            diagnostics_path.read_text(encoding="utf-8")
        )
        updates = {
            "payload_schema_version": {
                "schema_version": "feedbax.manifest.training_diagnostics.v0"
            },
            "manifest_id": {"manifest_id": "feedbax-training-run:other"},
            "run_id": {"run_id": "other-job"},
            "terminal_status": {"terminal_status": "cancelled"},
        }
        matches = {
            "payload_schema_version": "training_diagnostics.v1",
            "manifest_id": "manifest_id does not match",
            "run_id": "run_id does not match",
            "terminal_status": "terminal_status does not match",
        }
        diagnostics = diagnostics.model_copy(update=updates[drift])
        diagnostics_path.write_text(
            diagnostics.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        payload = diagnostics_path.read_bytes()
        artifact = artifact.model_copy(
            update={"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
        )
        match = matches[drift]
    manifest = manifest.model_copy(update={"artifacts": [artifact]})
    resolved = ResolvedAnalysisInput(
        ref=ParentRef(kind="TrainingRunManifest", id=manifest.id, role="training_run"),
        manifest=manifest,
        path=manifest_path,
    )

    with pytest.raises(ValueError, match=match):
        td._summary_for_input(resolved, root=tmp_path / "feedbax_runs")


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("missing_sha256", "missing required sha256"),
        ("missing_size_bytes", "missing required size_bytes"),
        ("wrong_sha256", "sha256 mismatch"),
        ("wrong_size_bytes", "size mismatch"),
        ("media_type", "unsupported media_type"),
    ],
)
def test_native_training_diagnostics_rejects_artifact_integrity_drift(
    tmp_path: Path,
    drift: str,
    match: str,
) -> None:
    manifest, manifest_path, _diagnostics_path = _write_native_training_run(
        tmp_path / "feedbax_runs",
        tmp_path / "bulk" / drift,
    )
    updates = {
        "missing_sha256": {"sha256": None},
        "missing_size_bytes": {"size_bytes": None},
        "wrong_sha256": {"sha256": "0" * 64},
        "wrong_size_bytes": {"size_bytes": manifest.artifacts[0].size_bytes + 1},
        "media_type": {"media_type": "application/octet-stream"},
    }
    artifact = manifest.artifacts[0].model_copy(update=updates[drift])
    manifest = manifest.model_copy(update={"artifacts": [artifact]})
    resolved = ResolvedAnalysisInput(
        ref=ParentRef(kind="TrainingRunManifest", id=manifest.id, role="training_run"),
        manifest=manifest,
        path=manifest_path,
    )

    with pytest.raises(ValueError, match=match):
        td._summary_for_input(resolved, root=tmp_path / "feedbax_runs")


@pytest.mark.parametrize("duplicate_kind", ["native", "legacy_npz", "companion"])
def test_training_diagnostics_rejects_ambiguous_artifact_roles(
    tmp_path: Path,
    duplicate_kind: str,
) -> None:
    root = tmp_path / "feedbax_runs"
    if duplicate_kind == "native":
        manifest, manifest_path, _diagnostics_path = _write_native_training_run(
            root,
            tmp_path / "bulk" / duplicate_kind,
        )
        duplicate = manifest.artifacts[0].model_copy(
            update={"logical_name": "duplicate-training-diagnostics.json"}
        )
    else:
        manifest, manifest_path = _write_training_run(
            root,
            tmp_path / "bulk" / duplicate_kind,
        )
        index = 0 if duplicate_kind == "legacy_npz" else 2
        duplicate_role = (
            "training_diagnostics_npz" if duplicate_kind == "legacy_npz" else "training_summary"
        )
        duplicate = manifest.artifacts[index].model_copy(update={"role": duplicate_role})
    manifest = manifest.model_copy(update={"artifacts": [*manifest.artifacts, duplicate]})
    resolved = ResolvedAnalysisInput(
        ref=ParentRef(kind="TrainingRunManifest", id=manifest.id, role="training_run"),
        manifest=manifest,
        path=manifest_path,
    )

    with pytest.raises(ValueError, match="ambiguous artifacts"):
        td._summary_for_input(resolved, root=root)


def test_native_training_diagnostics_zero_progress_has_no_latest_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "feedbax_runs"
    manifest, manifest_path, diagnostics_path = _write_native_training_run(
        root,
        tmp_path / "bulk" / "zero-progress",
    )
    diagnostics = TrainingDiagnostics.model_validate_json(
        diagnostics_path.read_text(encoding="utf-8")
    ).model_copy(
        update={
            "completed_batches": 0,
            "segment_completed_batches": 0,
            "cumulative_completed_batches": 0,
            "checkpoint_coordinates": [],
            "checkpoint_transactions": [],
        }
    )
    diagnostics_path.write_text(
        diagnostics.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    payload = diagnostics_path.read_bytes()
    artifact = manifest.artifacts[0].model_copy(
        update={"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
    )
    manifest = manifest.model_copy(update={"completed_batches": 0, "artifacts": [artifact]})
    resolved = ResolvedAnalysisInput(
        ref=ParentRef(kind="TrainingRunManifest", id=manifest.id, role="training_run"),
        manifest=manifest,
        path=manifest_path,
    )

    summary = td._summary_for_input(resolved, root=root)

    assert summary["latest_batch_index"] is None
    assert summary["training_diagnostics"]["completed_batches"] == 0

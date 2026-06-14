"""Tests for training-diagnostics analysis bundle execution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import rlrmp
from feedbax.analysis.bundles import execute_analysis_bundle, load_analysis_bundle
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.manifest import (
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    TrainingRunManifest,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry
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
        pgd_broad_epsilon_inner_objective_final_endpoint_gap=np.array(
            [[np.nan], [0.0], [0.0]]
        ),
        pgd_broad_epsilon_epsilon_norm_radius_ratio_mean=np.array(
            [[np.nan], [0.5], [0.75]]
        ),
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


def test_training_diagnostics_bundle_executes_manifest_backed_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = _registry()
    bundle = load_analysis_bundle("rlrmp/training_diagnostics", registry=registry)
    run_manifest, _run_manifest_path = _write_training_run(
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

    json_path = fake_repo_root / "results" / "training_diagnostics" / "notes" / (
        "training_diagnostics_summary.json"
    )
    markdown_path = fake_repo_root / "results" / "training_diagnostics" / "notes" / (
        "training_diagnostics_summary.md"
    )
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == td.TRAINING_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["summaries"][0]["latest_batch_index"] == 2
    assert payload["summaries"][0]["latest"]["train_loss__total"]["mean"] == 1.0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "<!-- AUTO-GENERATED: training_diagnostics_summary -->" in markdown
    assert "| rlrmp-test-training-run:diagnostics | yes | 2 |" in markdown
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
    notes_path = fake_repo_root / "results" / "0e3223d" / "notes" / (
        "training_diagnostics_summary.md"
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

    json_path = fake_repo_root / "results" / "0e3223d" / "notes" / (
        "training_diagnostics_summary.json"
    )
    assert json_path.exists()
    assert notes_path.read_text(encoding="utf-8").startswith("# Handwritten context")
    assert manifest.summary_metrics["artifact_count"] == 2
    for artifact in manifest.artifacts:
        assert artifact.metadata["experiment"] == "0e3223d"

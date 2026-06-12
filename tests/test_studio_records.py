from __future__ import annotations

import json
from pathlib import Path

from feedbax.manifest import (
    ArtifactRef,
    Provenance,
    SpecPayload,
    TrainingRunManifest,
    write_manifest,
)

from rlrmp.studio_records import (
    build_studio_workspace_from_training_manifests,
    main,
    materialize_studio_records,
)


def _write_training_manifest(root: Path, *, job_id: str = "fixture__ok") -> Path:
    graph_spec = {
        "nodes": {
            "network": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "input_ports": ["input"],
        "output_ports": ["output"],
        "metadata": {
            "name": "rlrmp Studio import fixture",
            "created_at": "2026-06-12T00:00:00+00:00",
            "updated_at": "2026-06-12T00:00:00+00:00",
        },
    }
    manifest = TrainingRunManifest(
        id=f"feedbax-training-run:{job_id}",
        status="completed",
        job_id=job_id,
        run_set_id="fixture-run-set",
        graph_spec=SpecPayload(
            kind="GraphSpec",
            inline=graph_spec,
            sha256="fixture-graph-sha",
            metadata={"graph_spec_version": "feedbax.graphspec.fixture.v1"},
        ),
        training_spec=SpecPayload(
            kind="RLRMPRunSpec",
            inline={"run": job_id, "issue": "10b38d7", "n_train_batches": 1},
        ),
        provenance=Provenance(issues=["10b38d7", "577806f"]),
        artifacts=[
            ArtifactRef(
                role="training_checkpoint",
                logical_name="trained_model.eqx",
                artifact_id=f"repo://rlrmp/_artifacts/10b38d7/runs/{job_id}/trained_model.eqx",
                media_type="application/x-equinox",
                storage_backend="rlrmp-_artifacts",
                uri=f"_artifacts/10b38d7/runs/{job_id}/trained_model.eqx",
                metadata={"availability": "reference_only"},
            )
        ],
        summary_metrics={"final_validation_loss": 0.25, "completed_batches": 1},
    )
    return write_manifest(manifest, root=root)


def test_build_workspace_seeds_training_graph_and_artifact_refs(tmp_path: Path) -> None:
    manifest_path = _write_training_manifest(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    workspace = build_studio_workspace_from_training_manifests(
        [(manifest_path, manifest)],
        requested_outputs=["summary_metrics"],
    )

    train_stage = next(stage for stage in workspace.stages if stage.kind == "train")
    analysis_stage = next(stage for stage in workspace.stages if stage.kind == "analysis")
    analysis_scenario = workspace.scenarios[analysis_stage.scenario_id or ""]

    assert train_stage.status == "completed"
    assert train_stage.output_collections[0].item_refs[0].id == manifest.id
    assert any(ref.role == "model_graph" for ref in workspace.manifest_refs)
    assert any(ref.role == "training_checkpoint" for ref in workspace.artifact_refs)
    assert analysis_scenario.analysis_spec == {
        "analysis_type": "rlrmp.standard_matrix",
        "requested_outputs": ["summary_metrics"],
        "input_requirements": [],
        "source": "rlrmp_training_manifest_import",
    }


def test_materialize_studio_records_writes_workspace_and_pipeline_manifests(
    tmp_path: Path,
) -> None:
    _write_training_manifest(tmp_path)

    result = materialize_studio_records(
        manifest_root=tmp_path,
        job_id="fixture-studio",
        requested_outputs=["summary_metrics"],
    )

    assert result.workspace_path.exists()
    assert set(result.stage_ids) == {"stage:eval", "stage:analysis", "stage:report"}
    assert set(result.manifest_paths) == {"stage:eval", "stage:analysis", "stage:report"}
    assert all(Path(path).exists() for path in result.manifest_paths.values())

    workspace_payload = json.loads(result.workspace_path.read_text(encoding="utf-8"))
    stages = {stage["kind"]: stage for stage in workspace_payload["stages"]}
    assert stages["eval"]["status"] == "completed"
    assert stages["analysis"]["status"] == "completed"
    assert stages["report"]["status"] == "completed"
    assert stages["eval"]["input_collections"][0]["item_refs"][0]["role"] == "training_run"
    assert stages["analysis"]["artifact_refs"][0]["role"] == "figure"

    analysis_manifest = json.loads(
        Path(result.manifest_paths["stage:analysis"]).read_text(encoding="utf-8")
    )
    assert analysis_manifest["kind"] == "AnalysisRunManifest"
    assert analysis_manifest["analysis_spec"]["inline"]["analysis_type"] == (
        "rlrmp.standard_matrix"
    )
    assert analysis_manifest["artifacts"][0]["role"] == "figure"


def test_studio_records_cli_supports_dry_run_json(tmp_path: Path, capsys) -> None:
    _write_training_manifest(tmp_path)

    exit_code = main(
        [
            "--manifest-root",
            str(tmp_path),
            "--job-id",
            "fixture-cli",
            "--dry-run",
            "--output-json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["workspace_path"].endswith("studio_workspaces/fixture-cli.json")
    assert len(payload["selected_manifest_paths"]) == 1

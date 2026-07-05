from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from feedbax.contracts.manifest import ArtifactRef, Provenance, TrainingRunManifest, spec_payload

from rlrmp.runtime.run_specs import RunSpecValidationError, resolve_run_record
from rlrmp.runtime.spec_migrations import (
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    ensure_rlrmp_spec_families,
)
from rlrmp.runtime.training_run_specs import (
    CONSUMED_DATA_IDENTITIES_KEY,
    add_consumed_data_identity,
    rlrmp_extension_payload,
)


def _run_spec(*, exp: str = "731fdf7", run: str = "fixture__ok") -> dict[str, Any]:
    return {
        "run": run,
        "issue": exp,
        "schema_version": "rlrmp.cs_stochastic_gru.v1",
        "game_card": {"dt": 0.01},
        "task_timing": {"n_steps": 61},
        "model_summary": {"controller_kind": "gru", "hidden_size": 4},
        "training_summary": {
            "training_mode": "nominal",
            "n_train_batches": 3,
            "batch_size": 2,
        },
        "loss_objective": "partial_feedbax_terms",
        "loss_summary": {
            "objective_profile": "partial_feedbax_terms",
            "active_cs_terms": {},
        },
        "provenance": {
            "git": {},
            "dependencies": {},
            "modal": {},
            "gpu": {},
        },
        "feedbax_graph": {
            "graph_spec_path": None,
            "manifest_path": "model.graph.manifest.json",
            "graph_export_status": "unavailable",
        },
        "hps": {"model": {"n_replicates": 2}},
        "scientific_payload_kept": {"nested": ["full", "payload"]},
    }


def _write_manifest(
    repo: Path,
    *,
    exp: str = "731fdf7",
    run: str = "fixture__ok",
    run_spec: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    ensure_rlrmp_spec_families()
    payload = rlrmp_extension_payload(run_spec or _run_spec(exp=exp, run=run))
    rel_spec = f"results/{exp}/runs/{run}.json"
    manifest = TrainingRunManifest(
        id=f"feedbax-training-run:rlrmp-{exp}-{run}",
        status="completed",
        job_id=run,
        training_spec=spec_payload(RUN_SPEC_KIND, payload, ref=rel_spec),
        provenance=Provenance(
            source_repo="https://github.com/i-m-mll/rlrmp.git",
            dirty=False,
            issues=[exp],
        ),
        artifacts=[
            ArtifactRef(
                role="tracked_run_spec",
                logical_name=f"{run}.json",
                artifact_id=f"repo://rlrmp/{rel_spec}",
                uri=rel_spec,
                media_type="application/json",
                storage_backend="rlrmp-results",
            )
        ],
    )
    root = root or repo / "_artifacts" / "feedbax_runs"
    path = root / "manifests" / "training_runs" / f"{manifest.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    return path


def test_resolve_run_record_returns_full_manifest_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    spec = _run_spec()
    _write_manifest(repo, run_spec=spec)

    resolved = resolve_run_record("731fdf7", "fixture__ok", repo_root=repo)

    assert resolved["schema_id"] == RUN_SPEC_SCHEMA_ID
    assert resolved["schema_version"] == RUN_SPEC_SCHEMA_VERSION
    assert resolved["source_schema_version"] == "rlrmp.cs_stochastic_gru.v1"
    assert resolved["scientific_payload_kept"] == {"nested": ["full", "payload"]}
    assert resolved["hps"] == spec["hps"]
    assert resolved[CONSUMED_DATA_IDENTITIES_KEY] == []


def test_resolve_run_record_missing_manifest_reports_archive_only(tmp_path: Path) -> None:
    legacy = tmp_path / "repo" / "results" / "731fdf7" / "runs" / "fixture__ok" / "run.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps(_run_spec()), encoding="utf-8")

    with pytest.raises(RunSpecValidationError, match="not_found.*archive-only"):
        resolve_run_record("731fdf7", "fixture__ok", repo_root=tmp_path / "repo")


def test_resolve_run_record_same_id_same_payload_is_ok(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    primary = _write_manifest(repo)
    duplicate = (
        repo
        / "_artifacts"
        / "731fdf7"
        / "runs"
        / "fixture__ok"
        / "training_run_manifest.json"
    )
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(primary.read_text(encoding="utf-8"), encoding="utf-8")

    assert resolve_run_record("731fdf7", "fixture__ok", repo_root=repo)["run"] == "fixture__ok"


def test_resolve_run_record_same_id_different_content_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    primary = _write_manifest(repo)
    duplicate = (
        repo
        / "_artifacts"
        / "731fdf7"
        / "runs"
        / "fixture__ok"
        / "training_run_manifest.json"
    )
    duplicate.parent.mkdir(parents=True)
    payload = json.loads(primary.read_text(encoding="utf-8"))
    payload["summary_metrics"] = {"different": 1}
    duplicate.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(RunSpecValidationError, match="same TrainingRunManifest id"):
        resolve_run_record("731fdf7", "fixture__ok", repo_root=repo)


def test_add_consumed_data_identity_appends_stable_identity() -> None:
    run_spec = add_consumed_data_identity(
        _run_spec(),
        role="calibration_table",
        schema="rlrmp.calibration_table.v1",
        hash="sha256:abc",
    )

    assert run_spec[CONSUMED_DATA_IDENTITIES_KEY] == [
        {
            "role": "calibration_table",
            "schema": "rlrmp.calibration_table.v1",
            "hash": "sha256:abc",
        }
    ]

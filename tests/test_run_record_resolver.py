from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from feedbax.contracts.manifest import ArtifactRef, Provenance, TrainingRunManifest, spec_payload

from rlrmp.runtime import run_specs
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
    manifest_id: str | None = None,
    job_id: str | None = None,
    status: str = "completed",
    summary_metrics: dict[str, Any] | None = None,
    provenance_metadata: dict[str, Any] | None = None,
    artifact_roles: tuple[str, ...] = ("tracked_run_spec",),
    training_spec_ref: str | None = None,
    artifact_uri: str | None = None,
) -> Path:
    ensure_rlrmp_spec_families()
    payload = rlrmp_extension_payload(run_spec or _run_spec(exp=exp, run=run))
    rel_spec = f"results/{exp}/runs/{run}.json"
    manifest = TrainingRunManifest(
        id=manifest_id or f"feedbax-training-run:rlrmp-{exp}-{run}",
        status=status,
        job_id=job_id or run,
        training_spec=spec_payload(RUN_SPEC_KIND, payload, ref=training_spec_ref or rel_spec),
        provenance=Provenance(
            source_repo="https://github.com/i-m-mll/rlrmp.git",
            dirty=False,
            issues=[exp],
            metadata=provenance_metadata or {},
        ),
        artifacts=[
            ArtifactRef(
                role=role,
                logical_name=f"{run}.json",
                artifact_id=f"repo://rlrmp/{rel_spec}",
                uri=artifact_uri or rel_spec,
                media_type="application/json",
                storage_backend="rlrmp-results",
            )
            for role in artifact_roles
        ],
        summary_metrics=summary_metrics or {},
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
    run_spec = tmp_path / "repo" / "results" / "731fdf7" / "runs" / "fixture__ok.json"
    run_spec.parent.mkdir(parents=True)
    run_spec.write_text(json.dumps(_run_spec()), encoding="utf-8")

    with pytest.raises(RunSpecValidationError, match="not_found.*archive-only"):
        resolve_run_record("731fdf7", "fixture__ok", repo_root=tmp_path / "repo")


def test_resolve_run_record_excludes_immutable_generic_native_manifest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    path = _write_manifest(repo)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["training_spec"]["kind"] = "TrainingRunSpec"
    payload["training_spec"]["schema_id"] = "feedbax.spec.training_run"
    payload["training_spec"]["schema_version"] = "feedbax.spec.training_run.v2"
    payload["training_spec"]["inline"]["schema_id"] = "feedbax.spec.training_run"
    payload["training_spec"]["inline"]["schema_version"] = "feedbax.spec.training_run.v2"
    payload["training_spec"].pop("sha256", None)
    payload["training_spec"].pop("source_sha256", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RunSpecValidationError,
        match="generic legacy TrainingRunSpec.*immutable legacy record.*new native row",
    ):
        resolve_run_record("731fdf7", "fixture__ok", repo_root=repo)


def test_resolve_run_record_same_id_same_payload_is_ok(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    primary = _write_manifest(repo)
    duplicate = (
        repo / "_artifacts" / "731fdf7" / "runs" / "fixture__ok" / "training_run_manifest.json"
    )
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(primary.read_text(encoding="utf-8"), encoding="utf-8")

    assert resolve_run_record("731fdf7", "fixture__ok", repo_root=repo)["run"] == "fixture__ok"


def test_resolve_run_record_same_id_different_content_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    primary = _write_manifest(repo)
    duplicate = (
        repo / "_artifacts" / "731fdf7" / "runs" / "fixture__ok" / "training_run_manifest.json"
    )
    duplicate.parent.mkdir(parents=True)
    payload = json.loads(primary.read_text(encoding="utf-8"))
    payload["summary_metrics"] = {"different": 1}
    duplicate.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(RunSpecValidationError, match="same TrainingRunManifest id"):
        resolve_run_record("731fdf7", "fixture__ok", repo_root=repo)


def test_resolve_run_record_prefers_completed_native_over_placeholder(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    exp = "c56ed2c"
    run = "native__ok"

    _write_manifest(
        repo,
        exp=exp,
        run=run,
        manifest_id=f"feedbax-training-run:rlrmp-{exp}-{run}",
        job_id=run,
        summary_metrics={"planned_batches": 3},
        provenance_metadata={"producer": "rlrmp.train.cs_nominal_gru.write_run_spec"},
    )
    native_path = _write_manifest(
        repo,
        exp=exp,
        run=run,
        manifest_id=f"feedbax-training-run:{run}-native",
        job_id=f"{run}-native",
        summary_metrics={"train_loss": 0.125},
        artifact_roles=("tracked_run_spec", "final_model", "training_summary"),
        training_spec_ref=f"/tmp/{run}.json",
        artifact_uri=f"/tmp/{run}.json",
    )

    selected_path, selected = run_specs._resolve_training_manifest(exp, run, repo_root=repo)
    resolved = resolve_run_record(exp, run, repo_root=repo)

    assert selected_path == native_path
    assert selected.summary_metrics == {"train_loss": 0.125}
    assert selected.job_id == f"{run}-native"
    assert resolved["run"] == run


def test_resolve_run_record_keeps_ambiguous_distinct_manifest_failure(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _write_manifest(
        repo,
        manifest_id="feedbax-training-run:fixture__ok-a",
        job_id="fixture__ok-a",
        summary_metrics={"train_loss": 0.1},
    )
    _write_manifest(
        repo,
        manifest_id="feedbax-training-run:fixture__ok-b",
        job_id="fixture__ok-b",
        summary_metrics={"train_loss": 0.2},
    )

    with pytest.raises(RunSpecValidationError, match="multiple TrainingRunManifest records"):
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

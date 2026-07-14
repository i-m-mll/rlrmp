"""Tests for training-diagnostics analysis bundle execution."""

from __future__ import annotations

import json
import shutil
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
from feedbax.persistence import (
    ImmutableArtifactBlobProvider,
    ImmutableArtifactBlobProviderSpec,
    open_immutable_artifact_blob_provider,
)
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


def _provider_backed_native_fixture(
    tmp_path: Path,
) -> tuple[TrainingRunManifest, Path, Path, ArtifactRef, ImmutableArtifactBlobProvider]:
    root = tmp_path / "feedbax_runs"
    manifest, manifest_path, diagnostics_path = _write_native_training_run(
        root,
        tmp_path / "transient-run-set" / "collected" / "row-a",
    )
    manifest = manifest.model_copy(
        update={
            "run_set_id": "run-set-a",
            "artifacts": [manifest.artifacts[0].model_copy(update={"artifact_id": None})],
            "metadata": {
                "training_row_provenance": {
                    "row_id": "row-a",
                    "planned_run_id": manifest.job_id,
                }
            },
        }
    )
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(),
        explicit_root=tmp_path / "immutable-custody",
    )
    authoritative = manifest.artifacts[0]
    mapped = provider.store_bytes(
        diagnostics_path.read_bytes(),
        role=authoritative.role,
        logical_name=authoritative.logical_name,
        media_type=authoritative.media_type,
        metadata={
            **authoritative.metadata,
            "training_manifest_id": manifest.id,
            "run_set_id": manifest.run_set_id,
            "row_id": "row-a",
            "planned_run_id": manifest.job_id,
            "run_id": manifest.job_id,
        },
    )
    return manifest, manifest_path, diagnostics_path, mapped, provider


def _historical_exact_parent_fixture(
    tmp_path: Path,
) -> tuple[
    TrainingRunManifest,
    Path,
    ArtifactRef,
    ParentRef,
    ImmutableArtifactBlobProvider,
]:
    """Build an M1-shaped immutable parent whose archived manifest lacks run_set_id."""
    transient = tmp_path / "transient-orchestration" / "collected" / "m1-row"
    transient.mkdir(parents=True)
    run_id = "m1-historical-planned-run"
    diagnostics = TrainingDiagnostics(
        manifest_id=run_id,
        run_id=run_id,
        terminal_status="completed",
        completed_batches=100,
        segment_completed_batches=100,
        cumulative_completed_batches=100,
        seeds=[0],
        lr_trace=[{"step": 0, "learning_rate": 3e-4}, {"step": 100, "learning_rate": 1e-4}],
        checkpoint_coordinates=[],
        checkpoint_transactions=[],
        metadata={"historical_family": "four-m1"},
    )
    diagnostics_bytes = (diagnostics.model_dump_json(indent=2) + "\n").encode()
    diagnostics_path = transient / "training-diagnostics.json"
    diagnostics_path.write_bytes(diagnostics_bytes)
    diagnostics_sha = sha256(diagnostics_bytes).hexdigest()
    authoritative = ArtifactRef(
        role="training_diagnostics",
        logical_name="training-diagnostics.json",
        sha256=diagnostics_sha,
        size_bytes=len(diagnostics_bytes),
        uri=str(diagnostics_path),
        media_type="application/json",
        metadata={
            "schema_id": td.NATIVE_DIAGNOSTICS_SCHEMA_ID,
            "schema_version": td.NATIVE_DIAGNOSTICS_SCHEMA_VERSION,
        },
    )
    manifest = TrainingRunManifest(
        id=run_id,
        job_id=run_id,
        run_set_id=None,
        status="completed",
        completed_batches=100,
        artifacts=[authoritative],
        metadata={
            "training_row_provenance": {
                "schema_id": "feedbax.spec.training_row_provenance",
                "schema_version": "feedbax.spec.training_row_provenance.v2",
                "row_id": "m1-row",
                "row_index": 0,
                "planned_run_id": run_id,
                "authored_payload_hash": "1" * 64,
                "lowered_execution_payload_hash": "2" * 64,
                "seed": 0,
                "axis_coordinates": {"model": "m1"},
                "overrides": [],
                "lowerer_identities": [],
            }
        },
    )
    manifest_bytes = (manifest.model_dump_json(indent=2) + "\n").encode()
    manifest_path = transient / "training-run-manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(),
        explicit_root=tmp_path / "immutable-custody",
    )
    mapped = provider.store_bytes(
        diagnostics_bytes,
        role=authoritative.role,
        logical_name=authoritative.logical_name,
        media_type=authoritative.media_type,
        metadata={
            **authoritative.metadata,
            "training_manifest_id": manifest.id,
            "run_set_id": "historical-m1-run-set",
            "row_id": "m1-row",
            "planned_run_id": run_id,
            "run_id": run_id,
        },
    )
    manifest_custody = provider.store_bytes(
        manifest_bytes,
        role="training_run_manifest",
        logical_name=manifest.id,
        media_type="application/json",
    )
    exact_ref = td.ExactTrainingManifestRef(
        id=manifest.id,
        uri=manifest_custody.uri,
        metadata=td.ExactTrainingManifestMetadata(
            manifest_sha256=manifest_custody.sha256,
            size_bytes=manifest_custody.size_bytes,
            run_set_id="historical-m1-run-set",
            row_id="m1-row",
            certificate_sha256="3" * 64,
            planned_run_id=run_id,
        ),
    ).to_parent_ref()
    return manifest, manifest_path, mapped, exact_ref, provider


def _manifest_artifact_for_ref(ref: ParentRef) -> ArtifactRef:
    return ArtifactRef(
        role="training_run_manifest",
        logical_name=ref.id,
        artifact_id=ref.uri,
        sha256=ref.metadata["manifest_sha256"],
        size_bytes=ref.metadata["size_bytes"],
        media_type="application/json",
        storage_backend="feedbax-local",
        uri=ref.uri,
    )


def _rebind_exact_ref_to_manifest(
    manifest: TrainingRunManifest,
    ref: ParentRef,
    provider: ImmutableArtifactBlobProvider,
) -> ParentRef:
    payload = (manifest.model_dump_json(indent=2) + "\n").encode()
    stored = provider.store_bytes(
        payload,
        role="training_run_manifest",
        logical_name=manifest.id,
        media_type="application/json",
    )
    return ref.model_copy(
        update={
            "uri": stored.uri,
            "metadata": {
                **ref.metadata,
                "manifest_sha256": stored.sha256,
                "size_bytes": stored.size_bytes,
            },
        }
    )


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


def test_provider_backed_diagnostics_survive_transient_run_set_deletion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest, manifest_path, diagnostics_path, mapped, provider = _provider_backed_native_fixture(
        tmp_path
    )
    assert manifest.artifacts[0].artifact_id is None
    assert mapped.artifact_id == mapped.uri == f"artifact://sha256/{mapped.sha256}"
    baseline_native = td._summarize_native_diagnostics(
        manifest=manifest,
        manifest_path=manifest_path,
        artifact=manifest.artifacts[0],
        root=tmp_path / "feedbax_runs",
    )
    resolver_calls: list[ArtifactRef] = []

    def resolve(artifact: ArtifactRef) -> bytes:
        resolver_calls.append(artifact)
        return provider.get_bytes(artifact)

    baseline = td.summarize_provider_backed_training_diagnostics(
        manifest=manifest,
        manifest_path=manifest_path,
        mapped_artifact=mapped,
        bytes_resolver=resolve,
    )
    shutil.rmtree(tmp_path / "transient-run-set")
    assert not diagnostics_path.exists()
    monkeypatch.setattr(
        td,
        "_artifact_path",
        lambda *_args, **_kwargs: pytest.fail("provider path must not resolve artifact paths"),
    )
    monkeypatch.setattr(
        td,
        "_resolve_repo_path",
        lambda *_args, **_kwargs: pytest.fail("provider path must not resolve repository paths"),
    )
    after_deletion = td.summarize_provider_backed_training_diagnostics(
        manifest=manifest,
        manifest_path=manifest_path,
        mapped_artifact=mapped,
        bytes_resolver=resolve,
    )

    assert after_deletion == baseline
    assert after_deletion["training_diagnostics"] == baseline_native["training_diagnostics"]
    assert after_deletion["diagnostics_artifact"] == manifest.artifacts[0].model_dump(
        mode="json", exclude_none=True
    )
    assert resolver_calls == [mapped, mapped]


def test_historical_provider_backed_diagnostics_use_exact_parent_after_deletion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest, manifest_path, mapped, exact_ref, provider = _historical_exact_parent_fixture(
        tmp_path
    )
    baseline = td._summarize_native_diagnostics(
        manifest=manifest,
        manifest_path=manifest_path,
        artifact=manifest.artifacts[0],
        root=tmp_path,
    )
    shutil.rmtree(tmp_path / "transient-orchestration")
    assert not manifest_path.exists()
    monkeypatch.setattr(
        td,
        "_artifact_path",
        lambda *_args, **_kwargs: pytest.fail("exact provider path must not resolve paths"),
    )
    monkeypatch.setattr(
        td,
        "_resolve_repo_path",
        lambda *_args, **_kwargs: pytest.fail("exact provider path must not resolve paths"),
    )
    resolver_calls: list[ArtifactRef] = []

    def resolve(artifact: ArtifactRef) -> bytes:
        resolver_calls.append(artifact)
        return provider.get_bytes(artifact)

    summary = td.summarize_provider_backed_training_diagnostics(
        manifest=manifest,
        manifest_path=manifest_path,
        mapped_artifact=mapped,
        bytes_resolver=resolve,
        training_manifest_ref=exact_ref,
    )

    assert manifest.run_set_id is None
    assert summary["training_diagnostics"] == baseline["training_diagnostics"]
    assert [ref.role for ref in resolver_calls] == ["training_run_manifest", "training_diagnostics"]
    assert [ref.uri for ref in resolver_calls] == [exact_ref.uri, mapped.uri]
    assert all(ref.uri == f"artifact://sha256/{ref.sha256}" for ref in resolver_calls)


def test_historical_provider_backed_diagnostics_require_exact_parent(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, mapped, _exact_ref, _provider = _historical_exact_parent_fixture(
        tmp_path
    )
    with pytest.raises(ValueError, match="native run_set_id or an exact training manifest ref"):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mapped,
            bytes_resolver=lambda _artifact: pytest.fail("resolver must not be called"),
        )


def test_historical_exact_parent_requires_complete_status_and_pass_metadata(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, mapped, exact_ref, _provider = _historical_exact_parent_fixture(
        tmp_path
    )
    metadata = dict(exact_ref.metadata)
    metadata.pop("conformance_overall")
    incomplete = exact_ref.model_copy(update={"metadata": metadata})
    with pytest.raises(ValueError, match="complete authenticated metadata: conformance_overall"):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mapped,
            bytes_resolver=lambda _artifact: pytest.fail("resolver must not be called"),
            training_manifest_ref=incomplete,
        )


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("kind", "TrainingRunManifest"),
        ("role", "training_run"),
        ("uri", "canonical payload identity"),
        ("id", "identity must bind"),
        ("manifest_sha256", "sha256 mismatch"),
        ("size_bytes", "size mismatch"),
        ("run_set_id", "run_set_id does not match parent manifest"),
        ("row_id", "row_id disagrees with manifest provenance"),
        ("planned_run_id", "identity must bind"),
        ("manifest_status", "completed"),
    ],
)
def test_historical_exact_parent_tamper_fails_before_diagnostics_resolution(
    tmp_path: Path,
    drift: str,
    match: str,
) -> None:
    manifest, manifest_path, mapped, exact_ref, provider = _historical_exact_parent_fixture(
        tmp_path
    )
    original_exact_ref = exact_ref
    if drift in {"kind", "role", "uri", "id"}:
        values = {
            "kind": "OtherManifest",
            "role": "other",
            "uri": f"artifact://sha256/{'4' * 64}",
            "id": "another-manifest",
        }
        exact_ref = exact_ref.model_copy(update={drift: values[drift]})
    else:
        metadata = dict(exact_ref.metadata)
        values = {
            "manifest_sha256": "4" * 64,
            "size_bytes": metadata["size_bytes"] + 1,
            "run_set_id": "wrong-run-set",
            "row_id": "wrong-row",
            "planned_run_id": "wrong-planned-run",
            "manifest_status": "failed",
        }
        metadata[drift] = values[drift]
        if drift == "manifest_sha256":
            exact_ref = exact_ref.model_copy(
                update={
                    "uri": f"artifact://sha256/{values[drift]}",
                    "metadata": metadata,
                }
            )
        else:
            exact_ref = exact_ref.model_copy(update={"metadata": metadata})
    resolver_calls: list[ArtifactRef] = []

    def resolve(artifact: ArtifactRef) -> bytes:
        resolver_calls.append(artifact)
        if artifact.role == "training_run_manifest":
            # Return the authoritative bytes directly so ref hash/size checks remain local.
            return provider.get_bytes(_manifest_artifact_for_ref(original_exact_ref))
        return provider.get_bytes(artifact)

    with pytest.raises(ValueError, match=match):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mapped,
            bytes_resolver=resolve,
            training_manifest_ref=exact_ref,
        )

    assert mapped not in resolver_calls


def test_historical_exact_parent_rejects_bytes_for_another_manifest(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, mapped, exact_ref, provider = _historical_exact_parent_fixture(
        tmp_path
    )
    another_manifest = manifest.model_copy(
        update={
            "id": "another-manifest",
            "job_id": "another-manifest",
            "metadata": {
                **manifest.metadata,
                "training_row_provenance": {
                    **manifest.metadata["training_row_provenance"],
                    "planned_run_id": "another-manifest",
                },
            },
        }
    )
    another_payload = (another_manifest.model_dump_json(indent=2) + "\n").encode()
    another_stored = provider.store_bytes(
        another_payload,
        role="training_run_manifest",
        logical_name=another_manifest.id,
        media_type="application/json",
    )
    exact_ref = exact_ref.model_copy(
        update={
            "uri": another_stored.uri,
            "metadata": {
                **exact_ref.metadata,
                "manifest_sha256": another_stored.sha256,
                "size_bytes": another_stored.size_bytes,
            },
        }
    )
    resolver_calls: list[ArtifactRef] = []

    def resolve(artifact: ArtifactRef) -> bytes:
        resolver_calls.append(artifact)
        return provider.get_bytes(artifact)

    with pytest.raises(ValueError, match="another manifest"):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mapped,
            bytes_resolver=resolve,
            training_manifest_ref=exact_ref,
        )
    assert mapped not in resolver_calls


@pytest.mark.parametrize("drift", ["native_run_set", "job_id", "row_provenance"])
def test_historical_exact_parent_rejects_manifest_identity_disagreement(
    tmp_path: Path,
    drift: str,
) -> None:
    manifest, manifest_path, mapped, exact_ref, provider = _historical_exact_parent_fixture(
        tmp_path
    )
    if drift == "native_run_set":
        manifest = manifest.model_copy(update={"run_set_id": "different-native-run-set"})
    elif drift == "job_id":
        manifest = manifest.model_copy(update={"job_id": "different-job"})
    else:
        manifest = manifest.model_copy(
            update={
                "metadata": {
                    **manifest.metadata,
                    "training_row_provenance": {
                        **manifest.metadata["training_row_provenance"],
                        "row_id": "different-row",
                    },
                }
            }
        )
    exact_ref = _rebind_exact_ref_to_manifest(manifest, exact_ref, provider)
    resolver_calls: list[ArtifactRef] = []

    def resolve(artifact: ArtifactRef) -> bytes:
        resolver_calls.append(artifact)
        return provider.get_bytes(artifact)

    with pytest.raises(ValueError, match="disagrees|identity must bind"):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mapped,
            bytes_resolver=resolve,
            training_manifest_ref=exact_ref,
        )
    assert mapped not in resolver_calls


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("role", "role does not match authoritative"),
        ("logical_name", "logical_name does not match authoritative"),
        ("sha256", "sha256 does not match authoritative"),
        ("size_bytes", "size_bytes does not match authoritative"),
        ("media_type", "media_type does not match authoritative"),
        ("schema_id", "schema_id does not match authoritative"),
        ("schema_version", "schema_version does not match authoritative"),
        ("artifact_id", "artifact_id must preserve canonical payload identity"),
        ("uri", "uri must equal its canonical artifact_id"),
        ("storage_backend", "storage_backend must be"),
        ("training_manifest_id", "training_manifest_id does not match parent manifest"),
        ("run_set_id", "run_set_id does not match parent manifest"),
        ("row_id", "row_id does not match parent manifest"),
        ("planned_run_id", "planned_run_id does not match parent manifest"),
        ("run_id", "run_id does not match parent manifest"),
    ],
)
def test_provider_backed_diagnostics_reject_identity_drift_before_resolver(
    tmp_path: Path,
    drift: str,
    match: str,
) -> None:
    manifest, manifest_path, _diagnostics_path, mapped, _provider = _provider_backed_native_fixture(
        tmp_path
    )
    top_level_updates = {
        "role": {"role": "other"},
        "logical_name": {"logical_name": "other.json"},
        "sha256": {"sha256": "0" * 64},
        "size_bytes": {"size_bytes": mapped.size_bytes + 1},
        "media_type": {"media_type": "application/octet-stream"},
        "artifact_id": {"artifact_id": f"artifact://sha256/{'0' * 64}"},
        "uri": {"uri": f"artifact://sha256/{'0' * 64}"},
        "storage_backend": {"storage_backend": "other"},
    }
    if drift in top_level_updates:
        mapped = mapped.model_copy(update=top_level_updates[drift])
    else:
        mapped = mapped.model_copy(update={"metadata": {**mapped.metadata, drift: "wrong"}})
    resolver_calls: list[ArtifactRef] = []

    with pytest.raises(ValueError, match=match):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mapped,
            bytes_resolver=lambda artifact: resolver_calls.append(artifact) or b"unused",
        )

    assert resolver_calls == []


def test_provider_backed_diagnostics_preserve_authoritative_artifact_id(
    tmp_path: Path,
) -> None:
    manifest, manifest_path, _diagnostics_path, mapped, provider = _provider_backed_native_fixture(
        tmp_path
    )
    authoritative = manifest.artifacts[0].model_copy(update={"artifact_id": mapped.artifact_id})
    manifest = manifest.model_copy(update={"artifacts": [authoritative]})

    summary = td.summarize_provider_backed_training_diagnostics(
        manifest=manifest,
        manifest_path=manifest_path,
        mapped_artifact=mapped,
        bytes_resolver=provider.get_bytes,
    )
    assert summary["diagnostics_artifact"]["artifact_id"] == mapped.artifact_id

    mismatched = mapped.model_copy(
        update={
            "artifact_id": f"artifact://sha256/{'0' * 64}",
            "uri": f"artifact://sha256/{'0' * 64}",
        }
    )
    with pytest.raises(ValueError, match="artifact_id does not match authoritative"):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mismatched,
            bytes_resolver=lambda _artifact: pytest.fail("resolver must not be called"),
        )


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("non_bytes", "must return bytes"),
        ("wrong_size", "size mismatch"),
        ("wrong_hash", "sha256 mismatch"),
    ],
)
def test_provider_backed_diagnostics_reject_invalid_resolved_bytes(
    tmp_path: Path,
    mode: str,
    match: str,
) -> None:
    manifest, manifest_path, _diagnostics_path, mapped, provider = _provider_backed_native_fixture(
        tmp_path
    )
    payload = provider.get_bytes(mapped)
    returned: object = {
        "non_bytes": "not-bytes",
        "wrong_size": payload + b"x",
        "wrong_hash": bytes([payload[0] ^ 1]) + payload[1:],
    }[mode]

    with pytest.raises((TypeError, ValueError), match=match):
        td.summarize_provider_backed_training_diagnostics(
            manifest=manifest,
            manifest_path=manifest_path,
            mapped_artifact=mapped,
            bytes_resolver=lambda _artifact: returned,  # type: ignore[return-value]
        )


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

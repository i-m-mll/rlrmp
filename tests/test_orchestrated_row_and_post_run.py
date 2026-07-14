"""Row-packet integrity and certified post-run mapping tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import rlrmp
from feedbax.analysis import (
    StagedCheckpointCustodyRootBinding,
    resolve_staged_execution_context,
)
from feedbax.analysis.bundles import (
    load_analysis_bundle,
    predicate_matches_manifest,
)
from feedbax.config import ExperimentRegistry
from feedbax.contracts import StagedCheckpointCustodySpec, StagedExecutionDescriptor
from feedbax.contracts.manifest import load_manifest
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowProvenance
from feedbax.contracts.checkpoints import CheckpointTransactionManifest
from feedbax.contracts.manifest import ArtifactRef, ParentRef, TrainingRunManifest
from feedbax.contracts.spec_storage import training_run_execution_hash
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ImmutableInputDigest,
    ImmutableInputIdentity,
    ResolvedSnapshotRef,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import RunConformanceCertificate
from feedbax.training import DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY
from feedbax.training.diagnostics import (
    TRAINING_DIAGNOSTICS_SCHEMA_ID,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
    NativeExecutionProducerContext,
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
    TrainingDiagnostics,
)
from feedbax.persistence import (
    ImmutableArtifactBlobProviderSpec,
    open_immutable_artifact_blob_provider,
)
from feedbax.contracts.worker import ProgressCoordinate
import feedbax.training.executor as executor_module
import rlrmp.train.orchestrated_post_run as post_run_module

from rlrmp.train.orchestration_drivers import (
    _remote_same_row_binding,
    _resume_checkpoint_root,
    _same_row_binding,
)
from rlrmp.train.orchestrated_post_run import (
    HistoricalCheckpointAuthoritySource,
    MappedHistoricalCheckpointAuthority,
    map_registered_run_set,
)
from rlrmp.train.orchestrated_row import (
    RowLaunchPacket,
    _batch_limit_probe,
    _native_execution_context,
    _verify_same_row_checkpoint,
    _verify_staged_checkpoint,
    execute_packet,
    load_packet,
)
from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.initial_slots import RlrmpRuntime
from rlrmp.train.fixture_orchestration import (
    fixture_training_run_spec,
    register_fixture_method,
)
from rlrmp.io import load_named_python_module
from rlrmp.runtime.run_specs import resolve_run_record
from rlrmp.runtime.spec_migrations import RUN_SPEC_KIND, accept_rlrmp_spec_payload
from rlrmp.runtime.training_run_specs import FEEDBAX_TRAINING_RUN_SPEC_KEY
from rlrmp.train.native_manifest import (
    RLRMP_NATIVE_MANIFEST_COMPANION_KEY,
    NativeManifestTrainingDiagnostics,
    RlrmpNativeManifestCompanion,
    RlrmpNativeManifestMetadata,
)


def _envelope(
    payload: dict[str, object],
    *,
    planned_run_id: str | None = None,
) -> ExecutionIdentityEnvelope:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_digest = hashlib.sha256(canonical).hexdigest()
    root = "1" * 64
    execution = training_run_execution_hash(root, [])
    provenance = (
        TrainingRowProvenance(
            row_id="row",
            row_index=0,
            planned_run_id=planned_run_id,
            authored_payload_hash="6" * 64,
            lowered_execution_payload_hash=payload_digest,
            seed=7,
            axis_coordinates={"run_id": planned_run_id},
            lowerer_identities=[RowLowererIdentity(lowerer_id="rlrmp.test", lowerer_version="v1")],
        )
        if planned_run_id is not None
        else None
    )
    return ExecutionIdentityEnvelope(
        payload=SchemaArtifactRef(
            schema_id=str(payload["schema_id"]),
            schema_version=str(payload["schema_version"]),
            artifact_id="payload",
            sha256=payload_digest,
        ),
        authored_intent=AuthoredIntentRef(
            schema_id="feedbax.spec.training_run_matrix",
            schema_version="feedbax.spec.training_run_matrix.v3",
            artifact_id="authored",
            sha256="2" * 64,
            intent_hash="3" * 64,
        ),
        resolved_snapshot=ResolvedSnapshotRef(
            schema_id="feedbax.spec.training_run_resolved_semantics",
            schema_version="feedbax.spec.training_run_resolved_semantics.v1",
            artifact_id="resolved",
            sha256="4" * 64,
            root_hash=root,
        ),
        execution_capsule=ExecutionCapsuleRef(
            schema_id="feedbax.manifest.training_run_execution_capsule",
            schema_version="feedbax.manifest.training_run_execution_capsule.v2",
            artifact_id="capsule",
            sha256="5" * 64,
            execution_hash=execution,
        ),
        immutable_inputs=[],
        row_provenance=provenance,
    )


def test_row_packet_rejects_payload_digest_mismatch(tmp_path: Path) -> None:
    payload = {"schema_id": "example", "schema_version": "example.v1"}
    packet = RowLaunchPacket(
        run_set_id="set",
        row_id="row",
        envelope=_envelope(payload),
        payload={**payload, "changed": True},
        row_dir=str(tmp_path),
    )
    path = tmp_path / "packet.json"
    path.write_text(packet.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="payload digest"):
        load_packet(path)


def test_row_packet_rejects_legacy_v2_with_regeneration_instruction(
    tmp_path: Path,
) -> None:
    payload = {"schema_id": "example", "schema_version": "example.v1"}
    packet = RowLaunchPacket(
        run_set_id="set",
        row_id="row",
        envelope=_envelope(payload),
        payload=payload,
        row_dir=str(tmp_path),
    ).model_dump(mode="json")
    packet["schema_version"] = "rlrmp.orchestrated_row_packet.v2"

    with pytest.raises(ValueError, match="v2.*must be regenerated"):
        RowLaunchPacket.model_validate(packet)


def test_rlrmp_training_packet_rejects_missing_native_manifest_companion(
    tmp_path: Path,
) -> None:
    source = json.loads(
        (
            Path(__file__).resolve().parents[1] / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"
        ).read_text(encoding="utf-8")
    )
    payload = source["feedbax_training_run_spec"]

    with pytest.raises(
        ValueError,
        match="requires a digest-bound native-manifest companion.*forbidden",
    ):
        RowLaunchPacket(
            run_set_id="set",
            row_id="row",
            envelope=_envelope(payload, planned_run_id="feedbax-training-run:planned"),
            payload=payload,
            row_dir=str(tmp_path / "row"),
        )


def test_row_packet_rejects_generic_spec_mismatched_from_nested_lineage(
    tmp_path: Path,
) -> None:
    source = json.loads(
        (
            Path(__file__).resolve().parents[1] / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"
        ).read_text(encoding="utf-8")
    )
    generic_a = source[FEEDBAX_TRAINING_RUN_SPEC_KEY]
    generic_b = {
        **generic_a,
        "metadata": {**generic_a["metadata"], "lineage_variant": "nested-b"},
    }
    canonical_rlrmp_payload = accept_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        source["rlrmp_run_spec"],
        source_version=source["rlrmp_run_spec"].get("schema_version"),
    ).payload
    companion = RlrmpNativeManifestCompanion(
        training_spec_payload={
            **canonical_rlrmp_payload,
            FEEDBAX_TRAINING_RUN_SPEC_KEY: generic_b,
        },
        training_spec_payload_ref="results/c6c5997/runs/flat_3e-5-epsilon-ramp.json",
        manifest_metadata=RlrmpNativeManifestMetadata(
            training_diagnostics=NativeManifestTrainingDiagnostics(enabled=True),
            gru_postrun_candidate=True,
        ),
    )
    packet_payload = {
        **generic_a,
        "metadata": {
            **generic_a["metadata"],
            RLRMP_NATIVE_MANIFEST_COMPANION_KEY: companion.model_dump(mode="json"),
        },
    }

    with pytest.raises(
        ValueError,
        match=r"/payload generic TrainingRunSpec.*feedbax_training_run_spec",
    ):
        RowLaunchPacket(
            run_set_id="set",
            row_id="row",
            envelope=_envelope(
                packet_payload,
                planned_run_id="feedbax-training-run:planned",
            ),
            payload=packet_payload,
            row_dir=str(tmp_path / "row"),
            native_manifest_companion=companion,
        )


def test_packet_builds_typed_native_context_with_exact_planned_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"schema_id": "example", "schema_version": "example.v1"}
    planned_run_id = "feedbax-training-run:planned"
    diagnostics = NativeTrainingDiagnosticsInput(
        seeds=[7],
        resume_context=ScheduleContextDiagnostic(
            schedule_origin_step=10,
            current_step=10,
            optimizer_count_at_current_step=0,
        ),
        metadata={"run_set_id": "set", "row_id": "row"},
    )
    packet = RowLaunchPacket(
        run_set_id="set",
        row_id="row",
        envelope=_envelope(payload, planned_run_id=planned_run_id),
        payload=payload,
        row_dir=str(tmp_path / "row"),
        native_training_diagnostics=diagnostics,
    )
    monkeypatch.setenv("FEEDBAX_ENV_FINGERPRINT", "environment:test")

    context = _native_execution_context(packet)

    assert isinstance(context, NativeExecutionProducerContext)
    assert context.execution.row_provenance is not None
    assert context.execution.row_provenance.planned_run_id == planned_run_id
    assert context.environment_fingerprint == "environment:test"
    assert context.collection_root == str(tmp_path / "row")
    assert context.diagnostics == diagnostics


def test_same_row_packet_revalidates_pinned_public_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feedbax.training as feedbax_training
    import feedbax.training.checkpoint_custody as checkpoint_custody

    payload = {"schema_id": "example", "schema_version": "example.v1"}
    run_id = "feedbax-training-run:planned"
    transaction_id = "tx-pinned"
    digest = "a" * 64
    checkpoint_root = tmp_path / "custody"
    packet = RowLaunchPacket(
        run_set_id="set",
        row_id="row",
        envelope=_envelope(payload, planned_run_id=run_id),
        payload=payload,
        row_dir=str(tmp_path / "row"),
        staged_checkpoint_root=str(checkpoint_root),
        same_row_resume_binding={
            "checkpoint_root": str(checkpoint_root),
            "transaction_id": transaction_id,
            "manifest_sha256": digest,
            "completed_batches": 50,
        },
        resume=True,
    )
    phase_program = object()
    run_spec = SimpleNamespace(
        method_payload=SimpleNamespace(payload={"n_train_batches": 100}),
        worker_execution=SimpleNamespace(
            method_contract=SimpleNamespace(phase_program=phase_program)
        ),
    )
    preparation = SimpleNamespace(
        initial_slots={"model": object()},
        resume_slot_transform=object(),
    )
    manifest = SimpleNamespace(
        transaction_id=transaction_id,
        run_id=run_id,
        completed_training_batches=50,
    )
    calls: list[tuple[Path, dict[str, object]]] = []

    def load_latest(root: Path, **kwargs: object) -> object:
        calls.append((root, kwargs))
        return SimpleNamespace(manifest=manifest)

    monkeypatch.setattr(checkpoint_custody, "load_latest_checkpoint", load_latest)
    monkeypatch.setattr(
        feedbax_training,
        "load_checkpoint_custody_documents",
        lambda _root: SimpleNamespace(
            latest_pointer=SimpleNamespace(
                document=SimpleNamespace(run_id=run_id, manifest_sha256=digest)
            ),
            manifest=SimpleNamespace(document=manifest),
        ),
    )

    _verify_same_row_checkpoint(
        packet,
        run_spec=run_spec,  # type: ignore[arg-type]
        preparation=preparation,
        planned_run_id=run_id,
    )

    assert calls == [
        (
            checkpoint_root,
            {
                "expected_run_spec": run_spec,
                "expected_phase_program": phase_program,
                "expected_slots": preparation.initial_slots,
                "resume_slot_transform": preparation.resume_slot_transform,
                "continuation_request": None,
                "allow_new_lineage_override": False,
            },
        )
    ]

    assert packet.same_row_resume_binding is not None
    stale = packet.model_copy(
        update={
            "same_row_resume_binding": packet.same_row_resume_binding.model_copy(
                update={"manifest_sha256": "b" * 64}
            )
        }
    )
    with pytest.raises(ValueError, match="manifest changed after launch authorization"):
        _verify_same_row_checkpoint(
            stale,
            run_spec=run_spec,  # type: ignore[arg-type]
            preparation=preparation,
            planned_run_id=run_id,
        )

    stale_progress = packet.model_copy(
        update={
            "same_row_resume_binding": packet.same_row_resume_binding.model_copy(
                update={"completed_batches": 49}
            )
        }
    )
    with pytest.raises(ValueError, match="completed-batch binding changed"):
        _verify_same_row_checkpoint(
            stale_progress,
            run_spec=run_spec,  # type: ignore[arg-type]
            preparation=preparation,
            planned_run_id=run_id,
        )


def test_resume_transport_distinguishes_absent_and_incomplete_same_row_bindings(
    tmp_path: Path,
) -> None:
    row_id = "row"
    target_root = tmp_path / "fork-target"
    fork_record = tmp_path / "fork.json"
    fork_record.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "row_id": row_id,
                        "checkpoint_root": str(target_root),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Fresh local and RunPod paths carry no same-row binding.
    assert _same_row_binding({}, row_id) is None
    assert _resume_checkpoint_root(None, {}, row_id) is None
    assert _remote_same_row_binding(None, remote_checkpoint=None) is None

    # Authored fork custody remains selected by its fork record.
    assert _resume_checkpoint_root(fork_record, {}, row_id) == target_root

    local_binding = {
        "checkpoint_root": str(tmp_path / "local-custody"),
        "transaction_id": "tx-pinned",
        "manifest_sha256": "a" * 64,
        "completed_batches": 50,
    }
    remote_root = "/workspace/run/inputs/row/checkpoint"
    assert _remote_same_row_binding(
        local_binding,
        remote_checkpoint=remote_root,
    ) == {**local_binding, "checkpoint_root": remote_root}

    with pytest.raises(ValueError, match="has no target"):
        _same_row_binding({"other": local_binding}, row_id)


def test_execute_packet_uses_native_manifest_and_diagnostics_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = json.loads(
        (
            Path(__file__).resolve().parents[1] / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"
        ).read_text(encoding="utf-8")
    )
    payload = source["feedbax_training_run_spec"]
    canonical_rlrmp_payload = accept_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        source["rlrmp_run_spec"],
        source_version=source["rlrmp_run_spec"].get("schema_version"),
    ).payload
    rlrmp_payload = {
        **canonical_rlrmp_payload,
        FEEDBAX_TRAINING_RUN_SPEC_KEY: payload,
    }
    companion = RlrmpNativeManifestCompanion(
        training_spec_payload=rlrmp_payload,
        training_spec_payload_ref="results/c6c5997/runs/flat_3e-5-epsilon-ramp.json",
        manifest_metadata=RlrmpNativeManifestMetadata(
            training_diagnostics=NativeManifestTrainingDiagnostics(enabled=True),
            gru_postrun_candidate=True,
        ),
    )
    payload = {
        **payload,
        "metadata": {
            **payload["metadata"],
            RLRMP_NATIVE_MANIFEST_COMPANION_KEY: companion.model_dump(mode="json"),
        },
    }
    planned_run_id = "feedbax-training-run:planned"
    row_dir = tmp_path / "row"
    packet = RowLaunchPacket(
        run_set_id="set",
        row_id="row",
        envelope=_envelope(payload, planned_run_id=planned_run_id),
        payload=payload,
        row_dir=str(row_dir),
        native_training_diagnostics=NativeTrainingDiagnosticsInput(seeds=[7]),
        native_manifest_companion=companion,
    )
    preparation = SimpleNamespace(
        initial_slots={"model": object()},
        kernel_context={"runtime": object()},
        loss_service=object(),
        resume_slot_transform=object(),
    )
    monkeypatch.setattr(
        DEFAULT_TRAINING_METHOD_REGISTRY,
        "_registrations",
        dict(DEFAULT_TRAINING_METHOD_REGISTRY._registrations),
    )
    monkeypatch.setattr(
        DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
        "_registrations",
        dict(DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY._registrations),
    )
    monkeypatch.setattr(
        DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
        "prepare",
        lambda _request: preparation,
    )
    calls: list[dict[str, Any]] = []
    native_manifest = row_dir / "manifest.json"

    def execute_native(_spec: object, **kwargs: Any) -> object:
        calls.append(kwargs)
        native_manifest.parent.mkdir(parents=True, exist_ok=True)
        native_manifest.write_text('{"native":true}\n', encoding="utf-8")
        return SimpleNamespace(
            manifest=SimpleNamespace(summary_metrics={"loss": 1.25}),
            manifest_path=native_manifest,
            diagnostics=SimpleNamespace(completed_batches=12),
            status="completed",
        )

    monkeypatch.setattr(executor_module, "execute_training_run_spec", execute_native)

    result = execute_packet(packet)

    assert result == native_manifest
    assert len(calls) == 1
    assert calls[0]["run_id"] == planned_run_id
    assert calls[0]["training_spec_payload"] == rlrmp_payload
    assert calls[0]["training_spec_payload_kind"] == "RLRMPRunSpec"
    assert calls[0]["training_spec_payload_schema_id"] == "rlrmp.run_spec"
    assert calls[0]["training_spec_payload_schema_version"] == "rlrmp.run_spec.v2"
    assert calls[0]["training_spec_payload_ref"] == (
        "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"
    )
    assert calls[0]["registry"] is DEFAULT_TRAINING_METHOD_REGISTRY
    projection = calls[0]["manifest_metadata_projection"]
    assert (
        projection.source_payload_sha256
        == companion.manifest_metadata_projection().source_payload_sha256
    )
    assert projection.values == {
        "training_diagnostics": {"enabled": True},
        "gru_postrun_candidate": True,
    }
    context = calls[0]["execution_context"]
    assert isinstance(context, NativeExecutionProducerContext)
    assert context.execution.row_provenance is not None
    assert context.execution.row_provenance.planned_run_id == planned_run_id
    assert json.loads(native_manifest.read_text(encoding="utf-8")) == {"native": True}
    assert not (row_dir / "feedbax-manifests" / "manifests" / "training_runs").exists()
    assert json.loads((row_dir / "training_summary.json").read_text(encoding="utf-8")) == {
        "completed_batches": 12,
        "metrics": {"loss": 1.25},
        "row_id": "row",
        "run_set_id": "set",
        "status": "completed",
    }


def test_native_manifest_selectors_and_run_record_lineage(tmp_path: Path) -> None:
    """One tiny native run remains selectable and model-lineage resolvable."""

    exp = "deadff5"
    run = "fixture__native"
    repo = tmp_path / "repo"
    source = json.loads(
        (
            Path(__file__).resolve().parents[1] / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"
        ).read_text(encoding="utf-8")
    )
    register_fixture_method()
    generic_spec = fixture_training_run_spec(n_batches=1)
    canonical_rlrmp_payload = accept_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        source["rlrmp_run_spec"],
        source_version=source["rlrmp_run_spec"].get("schema_version"),
    ).payload
    rlrmp_payload = {
        **canonical_rlrmp_payload,
        "issue": exp,
        "run": run,
        FEEDBAX_TRAINING_RUN_SPEC_KEY: generic_spec.model_dump(mode="json", exclude_none=True),
    }
    companion = RlrmpNativeManifestCompanion(
        training_spec_payload=rlrmp_payload,
        training_spec_payload_ref=f"results/{exp}/runs/{run}.json",
        manifest_metadata=RlrmpNativeManifestMetadata(
            training_diagnostics=NativeManifestTrainingDiagnostics(enabled=True),
            gru_postrun_candidate=True,
        ),
    )
    rlrmp.register_feedbax_training_methods(DEFAULT_TRAINING_METHOD_REGISTRY)

    result = executor_module.execute_training_run_spec(
        generic_spec,
        run_id=run,
        initial_slots={
            "model": 0,
            "optimizer": {"count": 1},
            "prng": [0, 1],
            "batch_counter": 0,
        },
        manifest_root=repo / "_artifacts/feedbax_runs",
        checkpoint_root=repo / "_artifacts" / exp / "runs" / run / "checkpoints",
        registry=DEFAULT_TRAINING_METHOD_REGISTRY,
        manifest_metadata_projection=companion.manifest_metadata_projection(),
        **companion.external_training_payload_kwargs(),
    )
    manifest = load_manifest(result.manifest_path)
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    diagnostics_bundle = load_analysis_bundle("rlrmp/training_diagnostics", registry=registry)
    gru_bundle = load_analysis_bundle("rlrmp/gru_postrun", registry=registry)

    assert manifest.status == "completed"
    assert manifest.id.endswith(run)
    assert manifest.training_spec is not None
    assert manifest.training_spec.kind == RUN_SPEC_KIND
    assert manifest.training_spec.inline == rlrmp_payload
    assert manifest.metadata["gru_postrun_candidate"] is True
    assert manifest.checkpoint_custody
    assert predicate_matches_manifest(diagnostics_bundle.predicate, manifest)
    assert predicate_matches_manifest(gru_bundle.predicate, manifest)
    assert resolve_run_record(exp, run, repo_root=repo) == rlrmp_payload


def test_batch_limit_probe_uses_completed_batches_not_probe_calls() -> None:
    progress = {"completed_batches": 0}
    runtime = RlrmpRuntime(
        completed_batches_reader=lambda: progress["completed_batches"],
    )
    probe = _batch_limit_probe(
        50,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
    )
    assert probe is not None
    coordinate = ProgressCoordinate(run_id="fast-two-chunk", phase="train_chunk")

    # Fast kernels may be probed any number of times before completing a chunk;
    # callback frequency is not training progress.
    assert all(probe(coordinate) is None for _ in range(75))

    # The first 50-batch chunk has completed. The stop decision now lets Feedbax
    # finish this chunk's checkpoint barrier and prevents the second chunk.
    progress["completed_batches"] = 50
    decision = probe(coordinate)
    assert decision is not None
    assert decision.action == "stop"
    assert progress["completed_batches"] == 50


def test_batch_limit_probe_fails_closed_without_typed_batch_progress() -> None:
    with pytest.raises(RuntimeError, match="authoritative completed-batch progress"):
        _batch_limit_probe(
            50,
            kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: RlrmpRuntime()},
        )


def test_resume_packet_verifies_fork_target_lineage_to_envelope_source(
    tmp_path: Path,
) -> None:
    source_bytes = b'{"transaction_id":"tx-source"}\n'
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    payload = {"schema_id": "example", "schema_version": "example.v1"}
    envelope = _envelope(payload).model_copy(
        update={
            "immutable_inputs": [
                ImmutableInputIdentity(
                    role="source_checkpoint",
                    kind="checkpoint_transaction",
                    identifier="checkpoint-transaction:tx-source",
                    digest=ImmutableInputDigest(value=source_digest),
                )
            ]
        }
    )
    # Rebuild the capsule hash because immutable inputs are execution identity.
    execution = training_run_execution_hash(
        envelope.resolved_snapshot.root_hash,
        [envelope.immutable_inputs[0].model_dump(mode="json", exclude_none=True)],
    )
    envelope = envelope.model_copy(
        update={
            "execution_capsule": envelope.execution_capsule.model_copy(
                update={"execution_hash": execution}
            )
        }
    )
    target_root = tmp_path / "target"
    manifest = target_root / "transactions" / "tx-target" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "transaction_id": "tx-target",
                "segment_lineage": {"parent_transaction_id": "tx-source"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    record = {
        "schema_version": "rlrmp.fork_gate_binding.v1",
        "source_input": {
            "transaction_id": "tx-source",
            "manifest_sha256": source_digest,
        },
        "targets": [
            {
                "row_id": "row",
                "checkpoint_root": str(target_root),
                "transaction_id": "tx-target",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        ],
    }
    record_path = tmp_path / "fork-gate.json"
    record_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    packet = RowLaunchPacket(
        run_set_id="set",
        row_id="row",
        envelope=envelope,
        payload=payload,
        row_dir=str(tmp_path / "row"),
        staged_checkpoint_root=str(target_root),
        fork_record_path=str(record_path),
        fork_record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        resume=True,
    )
    _verify_staged_checkpoint(packet)

    bad_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    bad_manifest["segment_lineage"]["parent_transaction_id"] = "tx-other"
    manifest.write_text(json.dumps(bad_manifest) + "\n", encoding="utf-8")
    record["targets"][0]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    record_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    packet = packet.model_copy(
        update={"fork_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest()}
    )
    with pytest.raises(ValueError, match="does not chain"):
        _verify_staged_checkpoint(packet)


def _checkpoint_transaction(
    transaction_id: str,
    *,
    run_id: str,
    start_batch: int,
    batch_count: int,
    parent_transaction_id: str | None = None,
    parent_manifest_uri: str | None = None,
    parent_manifest_sha256: str | None = None,
) -> CheckpointTransactionManifest:
    digest = hashlib.sha256(transaction_id.encode()).hexdigest()
    return CheckpointTransactionManifest.model_validate(
        {
            "transaction_id": transaction_id,
            "run_id": run_id,
            "status": "final",
            "barrier": "done",
            "completed_coordinate": {
                "run_id": run_id,
                "phase": "train",
                "program_step": start_batch + batch_count,
                "completed_barrier": "done",
            },
            "completed_training_batches": start_batch + batch_count,
            "segment_lineage": {
                "parent_transaction_id": parent_transaction_id,
                "start_batch": start_batch,
                "segment_batch_count": batch_count,
            },
            "consistency_predicate": {"rules": [], "phase_program_digest": "1" * 64},
            "run_contract_binding": {
                "training_run_spec_schema_id": "test.training",
                "training_run_spec_schema_version": "test.training.v1",
                "training_run_spec_sha256": "2" * 64,
                "method_payload_schema_id": "test.method",
                "method_payload_schema_version": "test.method.v1",
                "method_payload_sha256": "3" * 64,
                "phase_program_sha256": "4" * 64,
            },
            "slots": [],
            "content_integrity_digest": {
                "slots": [],
                "transaction_root_sha256": digest,
            },
            "parent_lineage": (
                [
                    {
                        "transaction_id": parent_transaction_id,
                        "relationship": "parent",
                        "manifest": {
                            "kind": "TrainingCheckpointTransactionManifest",
                            "id": parent_transaction_id,
                            "role": "training_checkpoint_custody",
                            "uri": parent_manifest_uri,
                            "metadata": {"manifest_sha256": parent_manifest_sha256},
                        },
                    }
                ]
                if parent_transaction_id is not None and parent_manifest_uri is not None
                else []
            ),
        }
    )


def _write_json_model(path: Path, model: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert hasattr(model, "model_dump_json")
    raw = (model.model_dump_json(indent=2, exclude_none=True) + "\n").encode()  # type: ignore[attr-defined]
    path.write_bytes(raw)
    return raw


def _mapped_run_set_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, object]]]:
    run_set = tmp_path / "set"
    source = run_set / "collected" / "row-a"
    source.mkdir(parents=True)
    (source / "training_summary.json").write_text('{"reviewer_convenience": true}\n')
    diagnostics = TrainingDiagnostics(
        manifest_id="run-a",
        run_id="run-a",
        terminal_status="completed",
        completed_batches=8,
        segment_completed_batches=8,
        cumulative_completed_batches=8,
    )
    diagnostics_path = source / "training-diagnostics.json"
    diagnostics_raw = _write_json_model(diagnostics_path, diagnostics)
    execution_hash = training_run_execution_hash("1" * 64, [])
    checkpoint_root = tmp_path / "checkpoint-custody"
    root_path = checkpoint_root / "transactions" / "tx-root" / "manifest.json"
    terminal_path = checkpoint_root / "transactions" / "tx-terminal" / "manifest.json"
    root_raw = _write_json_model(
        root_path,
        _checkpoint_transaction("tx-root", run_id="run-a", start_batch=0, batch_count=5),
    )
    terminal_raw = _write_json_model(
        terminal_path,
        _checkpoint_transaction(
            "tx-terminal",
            run_id="run-a",
            start_batch=5,
            batch_count=3,
            parent_transaction_id="tx-root",
            parent_manifest_uri=str(root_path),
            parent_manifest_sha256=hashlib.sha256(root_raw).hexdigest(),
        ),
    )
    manifest = TrainingRunManifest(
        id="run-a",
        run_set_id="set",
        job_id="run-a",
        status="completed",
        resolved_semantics_root_hash="1" * 64,
        execution_hash=execution_hash,
        completed_batches=8,
        metadata={"training_row_provenance": {"row_id": "row-a", "planned_run_id": "run-a"}},
        artifacts=[
            ArtifactRef(
                role="training_diagnostics",
                logical_name="training-diagnostics.json",
                artifact_id=(f"artifact://sha256/{hashlib.sha256(diagnostics_raw).hexdigest()}"),
                sha256=hashlib.sha256(diagnostics_raw).hexdigest(),
                media_type="application/json",
                size_bytes=len(diagnostics_raw),
                uri=str(diagnostics_path),
                metadata={
                    "schema_id": TRAINING_DIAGNOSTICS_SCHEMA_ID,
                    "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
                },
            )
        ],
        checkpoint_custody=[
            ParentRef(
                kind="TrainingCheckpointTransactionManifest",
                id="tx-terminal",
                role="training_checkpoint_custody",
                uri=str(terminal_path),
                metadata={"manifest_sha256": hashlib.sha256(terminal_raw).hexdigest()},
            ),
        ],
    )
    durable_manifest = tmp_path / "source-manifests" / "training-run-row-a.json"
    _write_json_model(durable_manifest, manifest)
    # The collected file is the orchestrator's existing copy. The mapper must
    # not create another TrainingRunManifest under the mapped artifact tree.
    (source / "manifest.json").hardlink_to(durable_manifest)
    conformance = {
        "schema_id": "feedbax.run_conformance",
        "schema_version": "feedbax.run_conformance.v1",
        "run_set_id": "set",
        "generated_at": "2026-07-13T00:00:00Z",
        "overall": "pass",
        "rows": {"row-a": {"checks": [{"check_id": "manifest_valid", "status": "pass"}]}},
    }
    conformance_raw = (json.dumps(conformance, sort_keys=True) + "\n").encode()
    (run_set / "conformance.json").write_bytes(conformance_raw)
    registration = {
        "run_set_id": "set",
        "status": "completed",
        "certificate_ref": str(run_set / "conformance.json"),
        "certificate_sha256": hashlib.sha256(conformance_raw).hexdigest(),
        "certificate_overall": "pass",
    }
    (run_set / "registration.json").write_text(json.dumps(registration), encoding="utf-8")
    (run_set / "bundle.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": "row-a",
                        "execution": {"execution_capsule": {"execution_hash": execution_hash}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return run_set, durable_manifest, {"row-a": {}}


def _add_second_mapped_row(
    run_set: Path,
    durable_manifest: Path,
    manifest_refs: dict[str, dict[str, object]],
) -> None:
    """Extend the mapped-run fixture with a second valid row."""
    source = run_set / "collected" / "row-b"
    source.mkdir(parents=True)
    shutil.copy2(run_set / "collected" / "row-a" / "training_summary.json", source)
    diagnostics = TrainingDiagnostics(
        manifest_id="run-b",
        run_id="run-b",
        terminal_status="completed",
        completed_batches=8,
        segment_completed_batches=8,
        cumulative_completed_batches=8,
    )
    diagnostics_path = source / "training-diagnostics.json"
    diagnostics_raw = _write_json_model(diagnostics_path, diagnostics)
    checkpoint_root = run_set.parent / "checkpoint-custody-row-b"
    root_path = checkpoint_root / "transactions" / "tx-root-b" / "manifest.json"
    terminal_path = checkpoint_root / "transactions" / "tx-terminal-b" / "manifest.json"
    root_raw = _write_json_model(
        root_path,
        _checkpoint_transaction("tx-root-b", run_id="run-b", start_batch=0, batch_count=5),
    )
    terminal_raw = _write_json_model(
        terminal_path,
        _checkpoint_transaction(
            "tx-terminal-b",
            run_id="run-b",
            start_batch=5,
            batch_count=3,
            parent_transaction_id="tx-root-b",
            parent_manifest_uri=str(root_path),
            parent_manifest_sha256=hashlib.sha256(root_raw).hexdigest(),
        ),
    )
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes()).model_copy(
        update={
            "id": "run-b",
            "job_id": "run-b",
            "metadata": {"training_row_provenance": {"row_id": "row-b", "planned_run_id": "run-b"}},
            "artifacts": [
                ArtifactRef(
                    role="training_diagnostics",
                    logical_name="training-diagnostics.json",
                    artifact_id=(
                        f"artifact://sha256/{hashlib.sha256(diagnostics_raw).hexdigest()}"
                    ),
                    sha256=hashlib.sha256(diagnostics_raw).hexdigest(),
                    media_type="application/json",
                    size_bytes=len(diagnostics_raw),
                    uri=str(diagnostics_path),
                    metadata={
                        "schema_id": TRAINING_DIAGNOSTICS_SCHEMA_ID,
                        "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
                    },
                )
            ],
            "checkpoint_custody": [
                ParentRef(
                    kind="TrainingCheckpointTransactionManifest",
                    id="tx-terminal-b",
                    role="training_checkpoint_custody",
                    uri=str(terminal_path),
                    metadata={"manifest_sha256": hashlib.sha256(terminal_raw).hexdigest()},
                )
            ],
        }
    )
    durable_b = durable_manifest.with_name("training-run-row-b.json")
    _write_json_model(durable_b, manifest)
    (source / "manifest.json").hardlink_to(durable_b)
    manifest_refs["row-b"] = {}
    bundle_path = run_set / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["rows"].append(
        {
            "row_id": "row-b",
            "execution": {"execution_capsule": {"execution_hash": manifest.execution_hash}},
        }
    )
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")


def test_completed_registration_maps_idempotently(tmp_path: Path) -> None:
    run_set, durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    provider_root = tmp_path / "immutable-provider"
    first = map_registered_run_set(
        run_set,
        repo_root=tmp_path,
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=provider_root,
    )
    original_manifest_raw = durable_manifest.read_bytes()
    original_diagnostics_raw = (run_set / "collected/row-a/training-diagnostics.json").read_bytes()
    before = first[0].read_bytes()
    second = map_registered_run_set(
        run_set,
        repo_root=tmp_path,
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=provider_root,
    )
    assert second == first
    assert first[0].read_bytes() == before
    recipe = json.loads(first[0].read_text(encoding="utf-8"))
    assert recipe["schema_version"] == "rlrmp.spec.orchestrated_post_run.v3"
    provider_spec = ImmutableArtifactBlobProviderSpec.model_validate(
        recipe["immutable_artifact_blob_provider_spec"]
    )
    assert str(provider_root) not in json.dumps(recipe)
    provider = open_immutable_artifact_blob_provider(provider_spec, explicit_root=provider_root)
    assert [ref["transaction_id"] for ref in recipe["checkpoint_lineage_refs"]] == ["tx-terminal"]
    assert all(
        ref["uri"].startswith("artifact://sha256/") for ref in recipe["checkpoint_lineage_refs"]
    )
    assert "source_paths" not in recipe
    assert recipe["reviewer_convenience_paths"] == {
        "training_summary": "_artifacts/158b580/runs/parity__row-a/training_summary.json"
    }
    assert not (first[0].parent / "training-diagnostics.json").exists()
    ArtifactRef.model_validate(recipe["training_diagnostics_artifact_ref"])
    mapped_root = first[0].parent
    assert not any(
        path.read_bytes() == durable_manifest.read_bytes()
        for path in mapped_root.rglob("*.json")
        if path != first[0]
    )

    # A fresh reviewer needs only the packet's declared custody root/provider.
    shutil.rmtree(run_set)
    published_manifest_ref = recipe["training_manifest_ref"]
    resolved_manifest = provider.get_bytes(
        published_manifest_ref["uri"],
        size_bytes=published_manifest_ref["metadata"]["size_bytes"],
    )
    assert resolved_manifest == original_manifest_raw
    assert (
        hashlib.sha256(resolved_manifest).hexdigest()
        == published_manifest_ref["metadata"]["manifest_sha256"]
    )
    resolved_model = TrainingRunManifest.model_validate_json(resolved_manifest)
    assert resolved_model.id == published_manifest_ref["id"]
    assert resolved_model.status == published_manifest_ref["metadata"]["manifest_status"]
    diagnostics_ref = ArtifactRef.model_validate(recipe["training_diagnostics_artifact_ref"])
    resolved_diagnostics = provider.get_bytes(diagnostics_ref)
    assert resolved_diagnostics == original_diagnostics_raw
    typed_diagnostics = TrainingDiagnostics.model_validate_json(resolved_diagnostics)
    assert typed_diagnostics.manifest_id == resolved_model.id
    assert typed_diagnostics.run_id == resolved_model.job_id
    original_diagnostics_ref = resolved_model.artifacts[0]
    assert diagnostics_ref.artifact_id == original_diagnostics_ref.artifact_id
    assert diagnostics_ref.role == original_diagnostics_ref.role
    assert diagnostics_ref.logical_name == original_diagnostics_ref.logical_name
    assert diagnostics_ref.sha256 == original_diagnostics_ref.sha256
    assert diagnostics_ref.size_bytes == original_diagnostics_ref.size_bytes
    assert diagnostics_ref.media_type == original_diagnostics_ref.media_type
    assert diagnostics_ref.metadata["schema_id"] == original_diagnostics_ref.metadata["schema_id"]
    assert (
        diagnostics_ref.metadata["schema_version"]
        == original_diagnostics_ref.metadata["schema_version"]
    )
    for key in ("registration_artifact_ref", "conformance_artifact_ref"):
        ref = ArtifactRef.model_validate(recipe[key])
        assert hashlib.sha256(provider.get_bytes(ref)).hexdigest() == ref.sha256
    # Legacy v3 evidence stores only refs explicitly declared by this manifest;
    # historical traversal requires the strict root-free authority field.
    refs_by_id = {ref["transaction_id"]: ref for ref in recipe["checkpoint_lineage_refs"]}
    ref = refs_by_id["tx-terminal"]
    raw = provider.get_bytes(ref["uri"], size_bytes=ref["size_bytes"])
    transaction = CheckpointTransactionManifest.model_validate_json(raw)
    assert transaction.transaction_id == "tx-terminal"
    assert transaction.run_id == "run-a"
    assert ref["relationship"] == "declared"
    provider_objects = list((provider_root / "artifacts/sha256").glob("*/*"))
    assert len([path for path in provider_objects if path.is_file()]) == 5
    assert not any("slot" in path.name for path in provider_objects)


def test_completed_registration_stages_all_rows_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_set, durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    _add_second_mapped_row(run_set, durable_manifest, manifest_refs)
    provider_root = tmp_path / "immutable-provider"
    original_store = post_run_module._store_checkpoint_lineage
    calls = 0

    def fail_later_row(*args: object, **kwargs: object) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected later-row CAS failure")
        return original_store(*args, **kwargs)

    monkeypatch.setattr(post_run_module, "_store_checkpoint_lineage", fail_later_row)
    with pytest.raises(OSError, match="later-row CAS failure"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )

    assert not (tmp_path / "_artifacts/158b580/runs/parity__row-a").exists()
    assert not (tmp_path / "_artifacts/158b580/runs/parity__row-b").exists()
    assert not (tmp_path / "_artifacts/158b580/run_sets/set/evidence").exists()
    assert not list((tmp_path / "_artifacts/158b580").glob(".mapped-run-stage-*"))


def test_completed_registration_publication_rollback_preserves_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_set, durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    provider_root = tmp_path / "immutable-provider"
    (existing_recipe,) = map_registered_run_set(
        run_set,
        repo_root=tmp_path,
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=provider_root,
    )
    existing_row = existing_recipe.parent
    before = {
        path.relative_to(existing_row): path.read_bytes()
        for path in existing_row.rglob("*")
        if path.is_file()
    }
    _add_second_mapped_row(run_set, durable_manifest, manifest_refs)
    (run_set / "collected/row-a/training_summary.json").write_text(
        '{"updated": true}\n', encoding="utf-8"
    )
    original_replace = Path.replace

    def fail_second_publish(path: Path, target: Path) -> Path:
        if path.name == "parity__row-b" and ".mapped-run-stage-" in str(path):
            raise OSError("injected second-row publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_publish)
    with pytest.raises(OSError, match="second-row publication failure"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )

    assert {
        path.relative_to(existing_row): path.read_bytes()
        for path in existing_row.rglob("*")
        if path.is_file()
    } == before
    assert not (tmp_path / "_artifacts/158b580/runs/parity__row-b").exists()
    assert not list((tmp_path / "_artifacts/158b580").glob(".mapped-run-stage-*"))


def test_completed_registration_mapping_fails_closed_on_evidence_drift(tmp_path: Path) -> None:
    run_set, _durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    provider_root = tmp_path / "immutable-provider"

    conformance = run_set / "conformance.json"
    conformance.write_bytes(conformance.read_bytes() + b" ")
    with pytest.raises(ValueError, match="certificate_sha256"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )


@pytest.mark.parametrize("artifact_count", [0, 2])
def test_completed_registration_requires_one_diagnostics_artifact(
    tmp_path: Path,
    artifact_count: int,
) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    diagnostics_ref = manifest.artifacts[0]
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"artifacts": [diagnostics_ref] * artifact_count}),
    )
    with pytest.raises(ValueError, match="exactly one training_diagnostics"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sha256", "0" * 64, "sha256 mismatch"),
        ("size_bytes", 1, "size mismatch"),
        ("media_type", "text/plain", "media_type"),
    ],
)
def test_completed_registration_rejects_diagnostics_artifact_contract_drift(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    diagnostics_ref = manifest.artifacts[0].model_copy(update={field: value})
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"artifacts": [diagnostics_ref]}),
    )
    with pytest.raises(ValueError, match=match):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def test_completed_registration_rejects_diagnostics_schema_drift(tmp_path: Path) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    metadata = {**manifest.artifacts[0].metadata, "schema_version": "legacy.v0"}
    diagnostics_ref = manifest.artifacts[0].model_copy(update={"metadata": metadata})
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"artifacts": [diagnostics_ref]}),
    )
    with pytest.raises(ValueError, match="schema_version"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def test_completed_registration_rejects_noncanonical_diagnostics_artifact_id(
    tmp_path: Path,
) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    diagnostics_ref = manifest.artifacts[0].model_copy(
        update={"artifact_id": "training-diagnostics:legacy-run-a"}
    )
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"artifacts": [diagnostics_ref]}),
    )
    with pytest.raises(ValueError, match="artifact_id is not canonical"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def test_completed_registration_rejects_relative_diagnostics_source_before_publication(
    tmp_path: Path,
) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    diagnostics_ref = manifest.artifacts[0].model_copy(
        update={"uri": "collected/row-a/training-diagnostics.json"}
    )
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"artifacts": [diagnostics_ref]}),
    )
    provider_root = tmp_path / "immutable-provider"
    with pytest.raises(ValueError, match="must name an absolute local path"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )
    assert not (tmp_path / "_artifacts/158b580/runs/parity__row-a").exists()
    assert not provider_root.exists()


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"manifest_id": "wrong"}, "parent/run identity"),
        ({"run_id": "wrong"}, "parent/run identity"),
        ({"terminal_status": "cancelled"}, "terminal status"),
        (
            {
                "completed_batches": 7,
                "segment_completed_batches": 7,
                "cumulative_completed_batches": 7,
            },
            "completed batch count",
        ),
    ],
)
def test_completed_registration_rejects_typed_diagnostics_binding_drift(
    tmp_path: Path,
    updates: dict[str, object],
    match: str,
) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    diagnostics_path = Path(manifest.artifacts[0].uri or "")
    diagnostics = TrainingDiagnostics.model_validate_json(diagnostics_path.read_bytes())
    raw = _write_json_model(diagnostics_path, diagnostics.model_copy(update=updates))
    digest = hashlib.sha256(raw).hexdigest()
    diagnostics_ref = manifest.artifacts[0].model_copy(
        update={
            "artifact_id": f"artifact://sha256/{digest}",
            "sha256": digest,
            "size_bytes": len(raw),
        }
    )
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"artifacts": [diagnostics_ref]}),
    )
    with pytest.raises(ValueError, match=match):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def test_completed_registration_rejects_collected_diagnostics_drift(tmp_path: Path) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    collected = run_set / "collected/row-a/training-diagnostics.json"
    authoritative = tmp_path / "authoritative/training-diagnostics.json"
    authoritative.parent.mkdir()
    shutil.copy2(collected, authoritative)
    diagnostics_ref = manifest.artifacts[0].model_copy(update={"uri": str(authoritative)})
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"artifacts": [diagnostics_ref]}),
    )
    collected.write_bytes(collected.read_bytes() + b" ")
    with pytest.raises(ValueError, match="differ from authoritative"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def test_completed_registration_rejects_exact_parent_identity_drift(tmp_path: Path) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    _write_json_model(durable_manifest, manifest.model_copy(update={"id": "wrong-parent"}))
    with pytest.raises(ValueError, match="planned_run_id mismatch"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def test_completed_registration_rejects_cancelled_training_manifest(tmp_path: Path) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    _write_json_model(durable_manifest, manifest.model_copy(update={"status": "cancelled"}))
    with pytest.raises(ValueError, match="TrainingRunManifest is not completed"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def test_completed_registration_accepts_completed_stopped_manifest(tmp_path: Path) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    _write_json_model(
        durable_manifest,
        manifest.model_copy(
            update={
                "stopped": True,
                "stop_reason": "requested",
                "completed_at": manifest.created_at,
            }
        ),
    )
    (recipe_path,) = map_registered_run_set(
        run_set,
        repo_root=tmp_path,
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=tmp_path / "immutable-provider",
    )
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["training_manifest_ref"]["metadata"]["manifest_status"] == "completed"


def test_completed_registration_reads_collected_manifest_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_set, _durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest_path = run_set / "collected/row-a/manifest.json"
    original_read_bytes = Path.read_bytes
    read_count = 0

    def count_manifest_read(path: Path) -> bytes:
        nonlocal read_count
        if path == manifest_path:
            read_count += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_manifest_read)
    map_registered_run_set(
        run_set,
        repo_root=tmp_path,
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=tmp_path / "immutable-provider",
    )
    assert read_count == 1


def test_completed_registration_cli_requires_and_consumes_manifest_provider(
    tmp_path: Path,
) -> None:
    run_set, _durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    module = load_named_python_module(
        "rlrmp_test_launch_training_mapped_evidence",
        Path(__file__).resolve().parents[1] / "scripts/launch_training.py",
    )
    base_args = [
        "map-post-run",
        str(run_set),
        "--repo-root",
        str(tmp_path),
        "--issue",
        "158b580",
        "--run-prefix",
        "parity",
    ]
    with pytest.raises(SystemExit):
        module.main(base_args)
    assert not (tmp_path / "_artifacts").exists()

    provider_spec_path = tmp_path / "provider-spec.json"
    provider_spec_path.write_text(ImmutableArtifactBlobProviderSpec().model_dump_json())
    assert (
        module.main(
            [
                *base_args,
                "--immutable-artifact-root",
                str(tmp_path / "immutable-provider"),
                "--immutable-artifact-provider-spec",
                str(provider_spec_path),
            ]
        )
        == 0
    )
    assert (tmp_path / "_artifacts/158b580/runs/parity__row-a/run.json").is_file()


def test_completed_registration_mapping_rejects_corrupt_existing_cas(tmp_path: Path) -> None:
    run_set, _durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    provider_root = tmp_path / "immutable-provider"
    (recipe_path,) = map_registered_run_set(
        run_set,
        repo_root=tmp_path,
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=provider_root,
    )
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    registration_ref = recipe["registration_artifact_ref"]
    materialized = (
        provider_root
        / "artifacts/sha256"
        / registration_ref["sha256"][:2]
        / registration_ref["sha256"]
    )
    materialized.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )


def test_completed_registration_mapping_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    run_set, _durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    provider_root = tmp_path / "immutable-provider"
    terminal = tmp_path / "checkpoint-custody/transactions/tx-terminal/manifest.json"
    terminal.write_bytes(terminal.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )

    run_set, _durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path / "missing")
    (tmp_path / "missing/checkpoint-custody/transactions/tx-root/manifest.json").unlink()
    (recipe_path,) = map_registered_run_set(
        run_set,
        repo_root=tmp_path / "missing",
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=tmp_path / "missing/immutable-provider",
    )
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert [ref["transaction_id"] for ref in recipe["checkpoint_lineage_refs"]] == ["tx-terminal"]


def test_completed_registration_rejects_relative_checkpoint_ref_before_publication(
    tmp_path: Path,
) -> None:
    run_set, durable_manifest, _manifest_refs = _mapped_run_set_fixture(tmp_path)
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes())
    checkpoint_ref = manifest.checkpoint_custody[0].model_copy(
        update={"uri": "transactions/tx-terminal/manifest.json"}
    )
    _write_json_model(
        durable_manifest,
        manifest.model_copy(update={"checkpoint_custody": [checkpoint_ref]}),
    )
    provider_root = tmp_path / "immutable-provider"
    with pytest.raises(ValueError, match="must name an absolute local path"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )
    assert not (tmp_path / "_artifacts/158b580/runs/parity__row-a").exists()
    assert not provider_root.exists()


def test_completed_registration_mapping_rejects_checkpoint_cycle(tmp_path: Path) -> None:
    run_set, durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    provider_root = tmp_path / "immutable-provider"
    checkpoint_root = tmp_path / "checkpoint-custody/transactions"
    root_path = checkpoint_root / "tx-root/manifest.json"
    terminal_path = checkpoint_root / "tx-terminal/manifest.json"
    root_raw = _write_json_model(
        root_path,
        _checkpoint_transaction(
            "tx-root",
            run_id="run-a",
            start_batch=1,
            batch_count=0,
            parent_transaction_id="tx-terminal",
        ),
    )
    terminal_raw = _write_json_model(
        terminal_path,
        _checkpoint_transaction(
            "tx-terminal",
            run_id="run-a",
            start_batch=1,
            batch_count=0,
            parent_transaction_id="tx-root",
        ),
    )
    manifest = TrainingRunManifest.model_validate_json(durable_manifest.read_bytes()).model_copy(
        update={
            "checkpoint_custody": [
                ParentRef(
                    kind="TrainingCheckpointTransactionManifest",
                    id="tx-root",
                    role="training_checkpoint_custody",
                    uri=str(root_path),
                    metadata={"manifest_sha256": hashlib.sha256(root_raw).hexdigest()},
                ),
                ParentRef(
                    kind="TrainingCheckpointTransactionManifest",
                    id="tx-terminal",
                    role="training_checkpoint_custody",
                    uri=str(terminal_path),
                    metadata={"manifest_sha256": hashlib.sha256(terminal_raw).hexdigest()},
                ),
            ]
        }
    )
    _write_json_model(durable_manifest, manifest)
    with pytest.raises(ValueError, match="lineage contains cycle"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path,
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=provider_root,
        )


def test_completed_registration_mapping_resolves_cross_root_fork_parent(tmp_path: Path) -> None:
    run_set, durable_manifest, manifest_refs = _mapped_run_set_fixture(tmp_path)
    local_root = tmp_path / "checkpoint-custody/transactions"
    original_root = local_root / "tx-root/manifest.json"
    source_provider = tmp_path / "checkpoint-source-provider"
    source_path = source_provider / "tx-root.json"
    source_path.parent.mkdir(parents=True)
    shutil.move(original_root, source_path)
    source_raw = source_path.read_bytes()
    source = CheckpointTransactionManifest.model_validate_json(source_raw)

    terminal_path = local_root / "tx-terminal/manifest.json"
    terminal_payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_payload["parent_lineage"] = [
        {
            "transaction_id": "tx-root",
            "relationship": "new_lineage_override",
            "manifest": {
                "kind": "TrainingCheckpointTransactionManifest",
                "id": "tx-root",
                "role": "training_checkpoint_custody",
                "uri": str(source_path),
                "metadata": {"manifest_sha256": hashlib.sha256(source_raw).hexdigest()},
            },
        }
    ]
    terminal_payload["fork_provenance"] = {
        "source": {
            "transaction_id": "tx-root",
            "run_id": "run-a",
            "manifest_sha256": hashlib.sha256(source_raw).hexdigest(),
            "transaction_root_sha256": source.content_integrity_digest.transaction_root_sha256,
            "manifest_relative_path": "transactions/tx-root/manifest.json",
        },
        "slots": [],
        "tool_version": "test",
    }
    terminal = CheckpointTransactionManifest.model_validate(terminal_payload)
    terminal_raw = _write_json_model(terminal_path, terminal)

    training_manifest = TrainingRunManifest.model_validate_json(
        durable_manifest.read_bytes()
    ).model_copy(
        update={
            "checkpoint_custody": [
                ParentRef(
                    kind="TrainingCheckpointTransactionManifest",
                    id="tx-terminal",
                    role="training_checkpoint_custody",
                    uri=str(terminal_path),
                    metadata={"manifest_sha256": hashlib.sha256(terminal_raw).hexdigest()},
                )
            ]
        }
    )
    _write_json_model(durable_manifest, training_manifest)
    (recipe_path,) = map_registered_run_set(
        run_set,
        repo_root=tmp_path,
        issue="158b580",
        run_prefix="parity",
        immutable_artifact_root=tmp_path / "immutable-provider",
    )
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["checkpoint_lineage_refs"][0]["transaction_id"] == "tx-terminal"
    assert recipe["checkpoint_lineage_refs"][0]["relationship"] == "declared"

    registration = json.loads((run_set / "registration.json").read_text(encoding="utf-8"))
    registration["status"] = "failed"
    (run_set / "registration.json").write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(ValueError, match="completed registration"):
        map_registered_run_set(
            run_set,
            repo_root=tmp_path / "missing",
            issue="158b580",
            run_prefix="parity",
            immutable_artifact_root=tmp_path / "immutable-provider",
        )


def _real_m1_historical_authority(
    tmp_path: Path,
) -> tuple[Path, HistoricalCheckpointAuthoritySource, Any, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    stop_source = repo_root / "_artifacts/orchestration/2026-07-13-6bae06ab"
    resume_source = repo_root / "_artifacts/orchestration/2026-07-13-b5e80253"
    stop_copy = tmp_path / "source-stop"
    resume_copy = tmp_path / "source-resume"
    shutil.copytree(stop_source, stop_copy)
    shutil.copytree(resume_source, resume_copy)
    row_id = "force_visible__nominal_seed42_smoke100"
    stop_manifest_path = stop_copy / "collected" / row_id / "manifest.json"
    resume_manifest_path = resume_copy / "collected" / row_id / "manifest.json"
    stop_manifest = TrainingRunManifest.model_validate_json(stop_manifest_path.read_bytes())
    resume_manifest = TrainingRunManifest.model_validate_json(resume_manifest_path.read_bytes())
    binding = "m1-checkpoints"

    def execution_ref(source_ref: ParentRef) -> ParentRef:
        return ParentRef(
            kind=source_ref.kind,
            id=source_ref.id,
            role=source_ref.role,
            uri=f"transactions/{source_ref.id}/manifest.json",
            metadata={
                "manifest_sha256": source_ref.metadata["manifest_sha256"],
                "checkpoint_custody_binding": binding,
            },
        )

    checkpoint_root = (repo_root / "_artifacts/2cb6a58/runs" / row_id).resolve()
    descriptor = StagedExecutionDescriptor(
        schema_id="feedbax.spec.staged_execution",
        schema_version="feedbax.spec.staged_execution.v1",
        artifact_providers={},
        checkpoint_custody={
            binding: StagedCheckpointCustodySpec(backend="feedbax-checkpoint-transaction-tree")
        },
    )
    context = resolve_staged_execution_context(
        descriptor,
        checkpoint_custody_bindings=[
            StagedCheckpointCustodyRootBinding(name=binding, root=checkpoint_root)
        ],
    )
    authority = HistoricalCheckpointAuthoritySource(
        mapped_row_id=row_id,
        stop_run_set_id="2026-07-13-6bae06ab",
        stop_row_id=row_id,
        stop_training_manifest_path=stop_manifest_path,
        stop_training_manifest_sha256=hashlib.sha256(stop_manifest_path.read_bytes()).hexdigest(),
        stop_training_manifest_size_bytes=stop_manifest_path.stat().st_size,
        stop_registration_path=stop_copy / "registration.json",
        stop_registration_sha256=hashlib.sha256(
            (stop_copy / "registration.json").read_bytes()
        ).hexdigest(),
        stop_registration_size_bytes=(stop_copy / "registration.json").stat().st_size,
        stop_conformance_path=stop_copy / "conformance.json",
        stop_conformance_sha256=hashlib.sha256(
            (stop_copy / "conformance.json").read_bytes()
        ).hexdigest(),
        stop_conformance_size_bytes=(stop_copy / "conformance.json").stat().st_size,
        stop_checkpoint_ref=execution_ref(stop_manifest.checkpoint_custody[0]),
        resume_checkpoint_ref=execution_ref(resume_manifest.checkpoint_custody[0]),
        checkpoint_custody_binding=binding,
    )
    return resume_copy, authority, context, checkpoint_root


def test_historical_checkpoint_authority_survives_run_set_deletion(tmp_path: Path) -> None:
    resume_run_set, authority, context, checkpoint_root = _real_m1_historical_authority(tmp_path)
    provider_root = tmp_path / "immutable-provider"
    outputs = map_registered_run_set(
        resume_run_set,
        repo_root=tmp_path,
        issue="2412353",
        run_prefix="m1",
        immutable_artifact_root=provider_root,
        historical_checkpoint_authorities=[authority],
        checkpoint_execution_context=context,
    )
    recipe = json.loads(outputs[0].read_text(encoding="utf-8"))
    mapped = MappedHistoricalCheckpointAuthority.model_validate(
        recipe["historical_checkpoint_authority"]
    )
    assert mapped.stop_completed_batches == 50
    assert mapped.resume_completed_batches == 100
    assert mapped.stop_checkpoint_ref.id == "tx-c29c9b098f364575a970f0f23ba889bf"
    assert mapped.resume_checkpoint_ref.id == "tx-3868327ebce5417aa8eeb169cb6d2cc8"
    assert mapped.resume_checkpoint_ref.metadata["checkpoint_custody_binding"] == (
        mapped.checkpoint_custody_binding
    )
    encoded = json.dumps(recipe["historical_checkpoint_authority"], sort_keys=True)
    assert str(checkpoint_root) not in encoded
    assert "source-stop" not in encoded
    assert "slots" not in recipe["historical_checkpoint_authority"]
    assert [ref["transaction_id"] for ref in recipe["checkpoint_lineage_refs"]] == [
        mapped.resume_checkpoint_ref.id
    ]

    shutil.rmtree(resume_run_set)
    shutil.rmtree(authority.stop_training_manifest_path.parents[2])
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(), explicit_root=provider_root
    )
    assert (
        TrainingRunManifest.model_validate_json(
            provider.get_bytes(mapped.stop_training_manifest_artifact_ref)
        ).status
        == "cancelled"
    )
    assert json.loads(provider.get_bytes(mapped.stop_registration_artifact_ref))["status"] == (
        "stopped"
    )
    assert (
        RunConformanceCertificate.model_validate_json(
            provider.get_bytes(mapped.stop_conformance_artifact_ref)
        ).overall
        == "pass"
    )
    stop_resolved = context.resolve_checkpoint_custody_ref(
        mapped.stop_checkpoint_ref, slot_names=None
    )
    resume_resolved = context.resolve_checkpoint_custody_ref(
        mapped.resume_checkpoint_ref, slot_names=None
    )
    assert stop_resolved.parent_ref == mapped.stop_checkpoint_ref
    assert resume_resolved.parent_ref == mapped.resume_checkpoint_ref
    assert stop_resolved.manifest.transaction_id == "tx-c29c9b098f364575a970f0f23ba889bf"
    assert resume_resolved.manifest.transaction_id == "tx-3868327ebce5417aa8eeb169cb6d2cc8"
    assert stop_resolved.slots["completed_batches"] == 50
    assert resume_resolved.slots["completed_batches"] == 100
    assert {"model", "optimizer", "prng", "completed_batches"}.issubset(stop_resolved.slots)
    assert {"model", "optimizer", "prng", "completed_batches"}.issubset(resume_resolved.slots)
    assert resume_resolved.manifest.segment_lineage.parent_transaction_id == (
        stop_resolved.manifest.transaction_id
    )
    assert resume_resolved.manifest.parent_lineage[0].manifest is not None
    assert resume_resolved.manifest.parent_lineage[0].manifest.uri is None
    assert "manifest_sha256" not in resume_resolved.manifest.parent_lineage[0].manifest.metadata

    drifted = mapped.model_dump(mode="json")
    drifted["schema_version"] = "rlrmp.evidence.historical_checkpoint_authority.v2"
    with pytest.raises(ValueError, match="schema_version"):
        MappedHistoricalCheckpointAuthority.model_validate(drifted)


def test_historical_checkpoint_authority_final_preflight_failure_has_no_cas_effects(
    tmp_path: Path,
) -> None:
    resume_run_set, authority, context, _checkpoint_root = _real_m1_historical_authority(tmp_path)
    registration = json.loads(authority.stop_registration_path.read_text(encoding="utf-8"))
    registration["row_outcomes"][authority.stop_row_id]["status"] = "completed"
    authority.stop_registration_path.write_text(json.dumps(registration), encoding="utf-8")
    authority = authority.model_copy(
        update={
            "stop_registration_sha256": hashlib.sha256(
                authority.stop_registration_path.read_bytes()
            ).hexdigest(),
            "stop_registration_size_bytes": authority.stop_registration_path.stat().st_size,
        }
    )
    provider_root = tmp_path / "immutable-provider"

    with pytest.raises(ValueError, match="exact row is not stopped"):
        map_registered_run_set(
            resume_run_set,
            repo_root=tmp_path,
            issue="2412353",
            run_prefix="m1",
            immutable_artifact_root=provider_root,
            historical_checkpoint_authorities=[authority],
            checkpoint_execution_context=context,
        )

    assert not provider_root.exists() or not any(provider_root.rglob("*"))


@pytest.mark.parametrize(
    ("path_field", "label"),
    [
        ("stop_training_manifest_path", "stop TrainingRunManifest"),
        ("stop_registration_path", "stop registration"),
        ("stop_conformance_path", "stop conformance certificate"),
    ],
)
def test_historical_checkpoint_authority_rejects_unpinned_document_mutation(
    tmp_path: Path,
    path_field: str,
    label: str,
) -> None:
    resume_run_set, authority, context, _checkpoint_root = _real_m1_historical_authority(tmp_path)
    path = getattr(authority, path_field)
    path.write_bytes(path.read_bytes() + b" ")
    provider_root = tmp_path / "immutable-provider"

    with pytest.raises(ValueError, match=rf"{label} expected size_bytes mismatch"):
        map_registered_run_set(
            resume_run_set,
            repo_root=tmp_path,
            issue="2412353",
            run_prefix="m1",
            immutable_artifact_root=provider_root,
            historical_checkpoint_authorities=[authority],
            checkpoint_execution_context=context,
        )

    assert not provider_root.exists() or not any(provider_root.rglob("*"))


@pytest.mark.parametrize(
    "uri",
    ["/tmp/transactions/tx/manifest.json", "transactions/../tx/manifest.json"],
)
def test_historical_checkpoint_authority_rejects_nonportable_checkpoint_uri(
    tmp_path: Path,
    uri: str,
) -> None:
    resume_run_set, authority, context, _checkpoint_root = _real_m1_historical_authority(tmp_path)
    authority = authority.model_copy(
        update={
            "stop_checkpoint_ref": authority.stop_checkpoint_ref.model_copy(update={"uri": uri})
        }
    )
    provider_root = tmp_path / "immutable-provider"
    with pytest.raises(ValueError, match="root-relative URI"):
        map_registered_run_set(
            resume_run_set,
            repo_root=tmp_path,
            issue="2412353",
            run_prefix="m1",
            immutable_artifact_root=provider_root,
            historical_checkpoint_authorities=[authority],
            checkpoint_execution_context=context,
        )
    assert not provider_root.exists() or not any(provider_root.rglob("*"))


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("custody_root_uri", "/absolute/checkpoint/root"),
        ("checkpoint_root", "../../retained-checkpoints"),
        ("manifest_path", "/absolute/transactions/manifest.json"),
        ("unknown", "otherwise-benign"),
    ],
)
def test_historical_checkpoint_authority_rejects_nonportable_checkpoint_metadata(
    tmp_path: Path,
    metadata_key: str,
    metadata_value: str,
) -> None:
    resume_run_set, authority, context, _checkpoint_root = _real_m1_historical_authority(tmp_path)
    metadata = {**authority.stop_checkpoint_ref.metadata, metadata_key: metadata_value}
    authority = authority.model_copy(
        update={
            "stop_checkpoint_ref": authority.stop_checkpoint_ref.model_copy(
                update={"metadata": metadata}
            )
        }
    )
    provider_root = tmp_path / "immutable-provider"

    with pytest.raises(ValueError, match=rf"nonportable metadata: {metadata_key}"):
        map_registered_run_set(
            resume_run_set,
            repo_root=tmp_path,
            issue="2412353",
            run_prefix="m1",
            immutable_artifact_root=provider_root,
            historical_checkpoint_authorities=[authority],
            checkpoint_execution_context=context,
        )

    assert not provider_root.exists() or not any(provider_root.rglob("*"))


@pytest.mark.parametrize(
    ("observed_key", "drifted_value"),
    [
        ("manifest_id", "feedbax-training-run:wrong-manifest"),
        ("diagnostics_manifest_id", "feedbax-training-run:wrong-diagnostics"),
        ("manifest_status", "completed"),
        ("diagnostics_terminal_status", "completed"),
        ("manifest_completed_batches", 49),
        ("diagnostics_completed_batches", 49),
    ],
)
def test_historical_checkpoint_authority_rejects_certificate_manifest_fact_drift(
    tmp_path: Path,
    observed_key: str,
    drifted_value: str | int,
) -> None:
    resume_run_set, authority, context, _checkpoint_root = _real_m1_historical_authority(tmp_path)
    conformance = json.loads(authority.stop_conformance_path.read_text(encoding="utf-8"))
    checks = conformance["rows"][authority.stop_row_id]["checks"]
    completed_batches = next(check for check in checks if check["check_id"] == "completed_batches")
    completed_batches["observed"][observed_key] = drifted_value
    authority.stop_conformance_path.write_text(json.dumps(conformance), encoding="utf-8")
    conformance_raw = authority.stop_conformance_path.read_bytes()
    registration = json.loads(authority.stop_registration_path.read_text(encoding="utf-8"))
    registration["certificate_sha256"] = hashlib.sha256(conformance_raw).hexdigest()
    authority.stop_registration_path.write_text(json.dumps(registration), encoding="utf-8")
    authority = authority.model_copy(
        update={
            "stop_registration_sha256": hashlib.sha256(
                authority.stop_registration_path.read_bytes()
            ).hexdigest(),
            "stop_registration_size_bytes": authority.stop_registration_path.stat().st_size,
            "stop_conformance_sha256": hashlib.sha256(conformance_raw).hexdigest(),
            "stop_conformance_size_bytes": len(conformance_raw),
        }
    )
    provider_root = tmp_path / "immutable-provider"

    with pytest.raises(ValueError, match="observed manifest facts"):
        map_registered_run_set(
            resume_run_set,
            repo_root=tmp_path,
            issue="2412353",
            run_prefix="m1",
            immutable_artifact_root=provider_root,
            historical_checkpoint_authorities=[authority],
            checkpoint_execution_context=context,
        )

    assert not provider_root.exists() or not any(provider_root.rglob("*"))

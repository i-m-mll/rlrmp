"""Row-packet integrity and certified post-run mapping tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import rlrmp
from feedbax.analysis.bundles import (
    load_analysis_bundle,
    predicate_matches_manifest,
)
from feedbax.config import ExperimentRegistry
from feedbax.contracts.manifest import load_manifest
from feedbax.contracts.run_matrix import RowLowererIdentity, TrainingRowProvenance
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
from feedbax.training import DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY
from feedbax.training.diagnostics import (
    NativeExecutionProducerContext,
    NativeTrainingDiagnosticsInput,
    ScheduleContextDiagnostic,
)
from feedbax.contracts.worker import ProgressCoordinate
import feedbax.training.executor as executor_module

from rlrmp.train.orchestration_drivers import (
    _remote_same_row_binding,
    _resume_checkpoint_root,
    _same_row_binding,
)
from rlrmp.train.orchestrated_post_run import map_registered_run_set
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


def test_completed_registration_maps_idempotently(tmp_path: Path) -> None:
    run_set = tmp_path / "set"
    source = run_set / "collected" / "row-a"
    source.mkdir(parents=True)
    for name in ("manifest.json", "training-diagnostics.json", "training_summary.json"):
        (source / name).write_text("{}\n", encoding="utf-8")
    registration = {
        "run_set_id": "set",
        "status": "completed",
        "certificate_sha256": "a" * 64,
    }
    (run_set / "registration.json").write_text(json.dumps(registration), encoding="utf-8")
    (run_set / "bundle.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": "row-a",
                        "execution": {"execution_capsule": {"execution_hash": "b" * 64}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    first = map_registered_run_set(
        run_set, repo_root=tmp_path, issue="158b580", run_prefix="parity"
    )
    before = first[0].read_bytes()
    second = map_registered_run_set(
        run_set, repo_root=tmp_path, issue="158b580", run_prefix="parity"
    )
    assert second == first
    assert first[0].read_bytes() == before
    recipe = json.loads(first[0].read_text(encoding="utf-8"))
    assert recipe["certificate_sha256"] == "a" * 64
    assert recipe["execution_hash"] == "b" * 64

    registration["status"] = "failed"
    (run_set / "registration.json").write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(ValueError, match="completed registration"):
        map_registered_run_set(run_set, repo_root=tmp_path, issue="158b580", run_prefix="parity")

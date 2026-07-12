"""Row-packet integrity and certified post-run mapping tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from feedbax.contracts.spec_storage import training_run_execution_hash
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ImmutableInputDigest,
    ImmutableInputIdentity,
    ResolvedSnapshotRef,
    SchemaArtifactRef,
)

from rlrmp.train.orchestrated_post_run import map_registered_run_set
from rlrmp.train.orchestrated_row import RowLaunchPacket, _verify_staged_checkpoint, load_packet


def _envelope(payload: dict[str, object]) -> ExecutionIdentityEnvelope:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_digest = hashlib.sha256(canonical).hexdigest()
    root = "1" * 64
    execution = training_run_execution_hash(root, [])
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

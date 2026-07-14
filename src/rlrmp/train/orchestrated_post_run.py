"""Map completed orchestrated rows into the established post-run layout."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlparse

from feedbax.analysis import StagedExecutionContext
from feedbax.contracts import StagedCheckpointCustodySpec
from feedbax.contracts.checkpoints import CheckpointTransactionManifest
from feedbax.contracts.manifest import ArtifactRef, ParentRef, TrainingRunManifest
from feedbax.orchestration.conformance import RunConformanceCertificate
from feedbax.persistence import (
    ImmutableArtifactBlobProvider,
    ImmutableArtifactBlobProviderSpec,
    open_immutable_artifact_blob_provider,
)
from feedbax.training import TrainingDiagnostics
from feedbax.training.diagnostics import (
    TRAINING_DIAGNOSTICS_SCHEMA_ID,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
)
from pydantic import BaseModel, ConfigDict


class HistoricalCheckpointAuthoritySource(BaseModel):
    """Exact source documents and portable checkpoint refs for one stopped segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mapped_row_id: str
    stop_run_set_id: str
    stop_row_id: str
    stop_training_manifest_path: Path
    stop_training_manifest_sha256: str
    stop_training_manifest_size_bytes: int
    stop_registration_path: Path
    stop_registration_sha256: str
    stop_registration_size_bytes: int
    stop_conformance_path: Path
    stop_conformance_sha256: str
    stop_conformance_size_bytes: int
    stop_checkpoint_ref: ParentRef
    resume_checkpoint_ref: ParentRef
    checkpoint_custody_binding: str


class MappedHistoricalCheckpointAuthority(BaseModel):
    """Root-free, independently resolvable authority for one historical resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["rlrmp.evidence.historical_checkpoint_authority"] = (
        "rlrmp.evidence.historical_checkpoint_authority"
    )
    schema_version: Literal["rlrmp.evidence.historical_checkpoint_authority.v1"] = (
        "rlrmp.evidence.historical_checkpoint_authority.v1"
    )
    mapped_row_id: str
    stop_run_set_id: str
    stop_row_id: str
    planned_run_id: str
    stop_training_manifest_artifact_ref: ArtifactRef
    stop_registration_artifact_ref: ArtifactRef
    stop_conformance_artifact_ref: ArtifactRef
    stop_checkpoint_ref: ParentRef
    resume_checkpoint_ref: ParentRef
    checkpoint_custody_binding: str
    checkpoint_custody_spec: StagedCheckpointCustodySpec
    stop_completed_batches: int
    resume_completed_batches: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _store_evidence(
    data: bytes,
    *,
    provider: ImmutableArtifactBlobProvider,
    role: str,
    logical_name: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    ref = provider.store_bytes(
        data,
        role=role,
        logical_name=logical_name,
        media_type="application/json",
        metadata=dict(metadata),
    )
    if provider.get_bytes(ref) != data:
        raise ValueError("stored evidence CAS object failed post-write verification")
    return ref.model_dump(mode="json", exclude_none=True)


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"run set has no {path.name}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return raw, payload


def _training_manifest_ref(
    raw: bytes,
    *,
    run_set_id: str,
    row_id: str,
    execution_hash: str,
    registration: Mapping[str, object],
) -> tuple[TrainingRunManifest, ArtifactRef, TrainingDiagnostics, bytes, dict[str, object]]:
    try:
        manifest = TrainingRunManifest.model_validate_json(raw)
    except Exception as exc:
        raise ValueError(f"collected row {row_id!r} has invalid TrainingRunManifest") from exc
    if manifest.status != "completed":
        raise ValueError(f"collected row {row_id!r} TrainingRunManifest is not completed")
    if manifest.run_set_id not in (None, run_set_id):
        raise ValueError(f"collected row {row_id!r} TrainingRunManifest run_set_id mismatch")
    row_provenance = manifest.metadata.get("training_row_provenance")
    if not isinstance(row_provenance, Mapping) or row_provenance.get("row_id") != row_id:
        raise ValueError(f"collected row {row_id!r} TrainingRunManifest row identity mismatch")
    planned_run_id = row_provenance.get("planned_run_id")
    if not isinstance(planned_run_id, str) or not planned_run_id:
        raise ValueError(f"collected row {row_id!r} has no governed planned_run_id")
    if manifest.job_id != planned_run_id or manifest.id != planned_run_id:
        raise ValueError(f"collected row {row_id!r} TrainingRunManifest planned_run_id mismatch")
    if manifest.execution_hash != execution_hash:
        raise ValueError(f"collected row {row_id!r} TrainingRunManifest execution_hash mismatch")
    manifest_metadata = {
        "manifest_sha256": _sha256(raw),
        "size_bytes": len(raw),
        "run_set_id": run_set_id,
        "row_id": row_id,
        "manifest_status": manifest.status,
        "registration_status": registration["status"],
        "conformance_overall": registration["certificate_overall"],
        "certificate_sha256": registration["certificate_sha256"],
        "planned_run_id": planned_run_id,
    }
    diagnostics_refs = [ref for ref in manifest.artifacts if ref.role == "training_diagnostics"]
    if len(diagnostics_refs) != 1:
        raise ValueError(
            f"TrainingRunManifest requires exactly one training_diagnostics artifact; "
            f"found {len(diagnostics_refs)}"
        )
    diagnostics_ref = diagnostics_refs[0]
    diagnostics_raw = _read_original_artifact(diagnostics_ref, label="training diagnostics")
    _validate_artifact_bytes(diagnostics_ref, diagnostics_raw, label="training diagnostics")
    canonical_diagnostics_id = f"artifact://sha256/{diagnostics_ref.sha256}"
    if (
        diagnostics_ref.artifact_id is not None
        and diagnostics_ref.artifact_id != canonical_diagnostics_id
    ):
        raise ValueError("training diagnostics artifact_id is not canonical for its sha256")
    if diagnostics_ref.media_type != "application/json":
        raise ValueError("training diagnostics artifact media_type must be application/json")
    if diagnostics_ref.metadata.get("schema_id") != TRAINING_DIAGNOSTICS_SCHEMA_ID:
        raise ValueError("training diagnostics artifact schema_id mismatch")
    if diagnostics_ref.metadata.get("schema_version") != TRAINING_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("training diagnostics artifact schema_version mismatch")
    try:
        diagnostics = TrainingDiagnostics.model_validate_json(diagnostics_raw)
    except Exception as exc:
        raise ValueError("training diagnostics artifact has invalid typed bytes") from exc
    if diagnostics.manifest_id != manifest.id or diagnostics.run_id != manifest.job_id:
        raise ValueError("training diagnostics artifact parent/run identity mismatch")
    if diagnostics.terminal_status != manifest.status:
        raise ValueError("training diagnostics terminal status mismatch")
    if (
        manifest.completed_batches is None
        or diagnostics.completed_batches != manifest.completed_batches
    ):
        raise ValueError("training diagnostics completed batch count mismatch")
    return manifest, diagnostics_ref, diagnostics, diagnostics_raw, manifest_metadata


def _read_original_artifact(ref: ArtifactRef, *, label: str) -> bytes:
    """Read one authoritative source artifact only during mapping."""
    if not isinstance(ref.uri, str) or not ref.uri:
        raise ValueError(f"{label} artifact has no source URI")
    path = _explicit_local_path(ref.uri, context=f"{label} source")
    if not path.is_file():
        raise ValueError(f"{label} source is not materialized: {path}")
    return path.read_bytes()


def _validate_artifact_bytes(ref: ArtifactRef, raw: bytes, *, label: str) -> None:
    if not isinstance(ref.sha256, str) or _sha256(raw) != ref.sha256:
        raise ValueError(f"{label} artifact sha256 mismatch")
    if ref.size_bytes != len(raw):
        raise ValueError(f"{label} artifact size mismatch")


def _resolve_ref_bytes(
    ref: Mapping[str, object],
    *,
    context: str,
) -> tuple[bytes, Path | None]:
    uri = ref.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"{context} has no explicit URI")
    path = _explicit_local_path(uri, context=context)
    if not path.is_file():
        raise ValueError(f"{context} is not materialized: {path}")
    return path.read_bytes(), path


def _explicit_local_path(uri: str, *, context: str) -> Path:
    """Resolve only explicit absolute local paths, never ambient cwd-relative names."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc:
            raise ValueError(f"{context} file URI must not contain a netloc")
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(uri)
    else:
        raise ValueError(f"{context} URI must name an explicit retained local path")
    if not path.is_absolute():
        raise ValueError(f"{context} URI must name an absolute local path")
    return path


def _checkpoint_lineage_evidence(
    manifest: TrainingRunManifest,
    *,
    allow_incomplete_refs: bool = False,
) -> list[tuple[bytes, CheckpointTransactionManifest, str]]:
    """Verify only explicitly materialized transaction refs declared by the manifest."""
    if not manifest.checkpoint_custody:
        raise ValueError("TrainingRunManifest has no checkpoint custody evidence")
    evidence: list[tuple[bytes, CheckpointTransactionManifest, str]] = []
    seen: set[str] = set()
    for ref in manifest.checkpoint_custody:
        transaction_id = getattr(ref, "id", None)
        if ref.kind != "TrainingCheckpointTransactionManifest" or not isinstance(
            transaction_id, str
        ):
            raise ValueError("TrainingRunManifest checkpoint custody requires manifest refs")
        if transaction_id in seen:
            raise ValueError(f"duplicate checkpoint transaction {transaction_id!r}")
        seen.add(transaction_id)
        digest = ref.metadata.get("manifest_sha256")
        if ref.uri is None or not isinstance(digest, str):
            if allow_incomplete_refs:
                continue
            raise ValueError(
                f"checkpoint transaction {transaction_id!r} has no explicit materialization"
            )
        raw, _path = _resolve_ref_bytes(
            ref.model_dump(mode="json", exclude_none=True),
            context=f"checkpoint transaction {transaction_id!r}",
        )
        if _sha256(raw) != digest:
            raise ValueError(f"checkpoint transaction {transaction_id!r} manifest hash mismatch")
        try:
            transaction = CheckpointTransactionManifest.model_validate_json(raw)
        except Exception as exc:
            raise ValueError(
                f"checkpoint transaction {transaction_id!r} manifest is invalid"
            ) from exc
        if transaction.transaction_id != transaction_id:
            raise ValueError(f"checkpoint transaction {transaction_id!r} resolved wrong id")
        if transaction.run_id != manifest.job_id:
            raise ValueError(f"checkpoint transaction {transaction_id!r} run identity mismatch")
        evidence.append((raw, transaction, "declared"))
    transactions = {transaction.transaction_id: transaction for _, transaction, _ in evidence}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(transaction_id: str) -> None:
        if transaction_id in visited:
            return
        if transaction_id in visiting:
            raise ValueError(f"checkpoint lineage contains cycle at {transaction_id!r}")
        visiting.add(transaction_id)
        parent_id = transactions[transaction_id].segment_lineage.parent_transaction_id
        if parent_id in transactions:
            visit(parent_id)
        visiting.remove(transaction_id)
        visited.add(transaction_id)

    for transaction_id in transactions:
        visit(transaction_id)
    return evidence


def _store_checkpoint_lineage(
    evidence: list[tuple[bytes, CheckpointTransactionManifest, str]],
    *,
    provider: ImmutableArtifactBlobProvider,
    run_set_id: str,
    row_id: str,
) -> list[dict[str, object]]:
    """Custody-store already verified transaction-manifest bytes."""

    refs: list[dict[str, object]] = []
    for raw, transaction, relationship in evidence:
        transaction_id = transaction.transaction_id
        artifact = _store_evidence(
            raw,
            provider=provider,
            role="training_checkpoint_transaction_manifest",
            logical_name=f"{transaction_id}.json",
            metadata={
                "run_set_id": run_set_id,
                "row_id": row_id,
                "transaction_id": transaction_id,
                "relationship": relationship,
                "transaction_root_sha256": (
                    transaction.content_integrity_digest.transaction_root_sha256
                ),
            },
        )
        refs.append(
            {
                **artifact,
                "kind": "TrainingCheckpointTransactionManifest",
                "transaction_id": transaction_id,
                "relationship": relationship,
                "transaction_root_sha256": (
                    transaction.content_integrity_digest.transaction_root_sha256
                ),
            }
        )
    return refs


def _validate_execution_checkpoint_ref(
    ref: ParentRef,
    *,
    binding: str,
    label: str,
) -> None:
    if ref.kind != "TrainingCheckpointTransactionManifest":
        raise ValueError(f"{label} kind must be TrainingCheckpointTransactionManifest")
    if ref.role != "training_checkpoint_custody":
        raise ValueError(f"{label} role must be training_checkpoint_custody")
    if ref.metadata.get("checkpoint_custody_binding") != binding:
        raise ValueError(f"{label} checkpoint_custody_binding mismatch")
    digest = ref.metadata.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} has no exact manifest_sha256")
    if not isinstance(ref.uri, str):
        raise ValueError(f"{label} has no root-relative execution URI")
    uri = PurePosixPath(ref.uri)
    expected = PurePosixPath("transactions") / ref.id / "manifest.json"
    if uri.is_absolute() or ".." in uri.parts or uri != expected:
        raise ValueError(f"{label} must use exact root-relative URI {expected}")


def _matching_checkpoint_identity(
    source_ref: ParentRef,
    execution_ref: ParentRef,
    *,
    label: str,
) -> None:
    if source_ref.kind != execution_ref.kind or source_ref.id != execution_ref.id:
        raise ValueError(f"{label} source and execution ref identity mismatch")
    if source_ref.role != execution_ref.role:
        raise ValueError(f"{label} source and execution ref role mismatch")
    if source_ref.metadata.get("manifest_sha256") != execution_ref.metadata.get("manifest_sha256"):
        raise ValueError(f"{label} source and execution ref manifest_sha256 mismatch")


def _validate_expected_document_bytes(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> None:
    if len(raw) != expected_size_bytes:
        raise ValueError(f"{label} expected size_bytes mismatch")
    if _sha256(raw) != expected_sha256:
        raise ValueError(f"{label} expected sha256 mismatch")


def _preflight_historical_checkpoint_authority(
    source: HistoricalCheckpointAuthoritySource,
    *,
    resume_manifest: TrainingRunManifest,
    checkpoint_execution_context: StagedExecutionContext,
) -> dict[str, object]:
    """Bind exact stopped-run documents to two contained, resolver-authenticated refs."""
    for path, label in (
        (source.stop_training_manifest_path, "stop TrainingRunManifest"),
        (source.stop_registration_path, "stop registration"),
        (source.stop_conformance_path, "stop conformance certificate"),
    ):
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"{label} path must be an explicit absolute regular file")

    stop_manifest_raw = source.stop_training_manifest_path.read_bytes()
    registration_raw = source.stop_registration_path.read_bytes()
    conformance_raw = source.stop_conformance_path.read_bytes()
    _validate_expected_document_bytes(
        stop_manifest_raw,
        expected_sha256=source.stop_training_manifest_sha256,
        expected_size_bytes=source.stop_training_manifest_size_bytes,
        label="stop TrainingRunManifest",
    )
    _validate_expected_document_bytes(
        registration_raw,
        expected_sha256=source.stop_registration_sha256,
        expected_size_bytes=source.stop_registration_size_bytes,
        label="stop registration",
    )
    _validate_expected_document_bytes(
        conformance_raw,
        expected_sha256=source.stop_conformance_sha256,
        expected_size_bytes=source.stop_conformance_size_bytes,
        label="stop conformance certificate",
    )
    try:
        stop_manifest = TrainingRunManifest.model_validate_json(stop_manifest_raw)
    except Exception as exc:
        raise ValueError("stop TrainingRunManifest is invalid") from exc
    if stop_manifest.status != "cancelled":
        raise ValueError("historical checkpoint authority requires cancelled stop manifest")
    stop_provenance = stop_manifest.metadata.get("training_row_provenance")
    if not isinstance(stop_provenance, Mapping):
        raise ValueError("stop TrainingRunManifest has no training_row_provenance")
    planned_run_id = stop_provenance.get("planned_run_id")
    if stop_provenance.get("row_id") != source.stop_row_id:
        raise ValueError("stop TrainingRunManifest row identity mismatch")
    if (
        not isinstance(planned_run_id, str)
        or stop_manifest.id != planned_run_id
        or stop_manifest.job_id != planned_run_id
    ):
        raise ValueError("stop TrainingRunManifest planned run identity mismatch")
    resume_provenance = resume_manifest.metadata.get("training_row_provenance")
    if not isinstance(resume_provenance, Mapping):
        raise ValueError("resume TrainingRunManifest has no training_row_provenance")
    if (
        resume_provenance.get("row_id") != source.mapped_row_id
        or resume_provenance.get("planned_run_id") != planned_run_id
        or resume_manifest.id != planned_run_id
        or resume_manifest.job_id != planned_run_id
    ):
        raise ValueError("stop and resume TrainingRunManifest identity mismatch")

    try:
        registration = json.loads(registration_raw)
        conformance_payload = json.loads(conformance_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("stop authority document is not valid JSON") from exc
    if not isinstance(registration, dict) or not isinstance(conformance_payload, dict):
        raise ValueError("stop authority documents must be JSON objects")
    if registration.get("run_set_id") != source.stop_run_set_id:
        raise ValueError("stop registration run_set_id mismatch")
    if registration.get("status") != "stopped":
        raise ValueError("historical checkpoint authority requires stopped registration")
    row_outcomes = registration.get("row_outcomes")
    if (
        not isinstance(row_outcomes, Mapping)
        or not isinstance(row_outcomes.get(source.stop_row_id), Mapping)
        or row_outcomes[source.stop_row_id].get("status") != "stopped"  # type: ignore[union-attr]
    ):
        raise ValueError("stop registration exact row is not stopped")
    try:
        conformance = RunConformanceCertificate.model_validate(conformance_payload)
    except Exception as exc:
        raise ValueError("stop conformance certificate is invalid") from exc
    certificate_sha256 = _sha256(conformance_raw)
    if (
        conformance.run_set_id != source.stop_run_set_id
        or conformance.overall != "pass"
        or registration.get("certificate_overall") != "pass"
        or registration.get("certificate_sha256") != certificate_sha256
    ):
        raise ValueError("stop registration and passing certificate are not exactly bound")
    certificate_row = conformance.rows.get(source.stop_row_id)
    if certificate_row is None or any(check.status != "pass" for check in certificate_row.checks):
        raise ValueError("stop conformance exact row is not passing")

    _validate_execution_checkpoint_ref(
        source.stop_checkpoint_ref,
        binding=source.checkpoint_custody_binding,
        label="stop checkpoint ref",
    )
    _validate_execution_checkpoint_ref(
        source.resume_checkpoint_ref,
        binding=source.checkpoint_custody_binding,
        label="resume checkpoint ref",
    )
    if len(stop_manifest.checkpoint_custody) != 1 or len(resume_manifest.checkpoint_custody) != 1:
        raise ValueError(
            "historical authority requires exact singular stop and resume checkpoint refs"
        )
    _matching_checkpoint_identity(
        stop_manifest.checkpoint_custody[0],
        source.stop_checkpoint_ref,
        label="stop checkpoint",
    )
    _matching_checkpoint_identity(
        resume_manifest.checkpoint_custody[0],
        source.resume_checkpoint_ref,
        label="resume checkpoint",
    )

    stop_resolved = checkpoint_execution_context.resolve_checkpoint_custody_ref(
        source.stop_checkpoint_ref,
        binding_name=source.checkpoint_custody_binding,
        slot_names=None,
    )
    resume_resolved = checkpoint_execution_context.resolve_checkpoint_custody_ref(
        source.resume_checkpoint_ref,
        binding_name=source.checkpoint_custody_binding,
        slot_names=None,
    )
    if stop_resolved.parent_ref != source.stop_checkpoint_ref:
        raise ValueError("stop checkpoint resolver changed the exact ParentRef")
    if resume_resolved.parent_ref != source.resume_checkpoint_ref:
        raise ValueError("resume checkpoint resolver changed the exact ParentRef")
    stop_transaction = stop_resolved.manifest
    resume_transaction = resume_resolved.manifest
    required_slots = {"model", "optimizer", "prng", "completed_batches"}
    if not required_slots.issubset(stop_resolved.slots):
        raise ValueError("stop checkpoint is missing required execution slots")
    if not required_slots.issubset(resume_resolved.slots):
        raise ValueError("resume checkpoint is missing required execution slots")
    if stop_transaction.run_id != planned_run_id or resume_transaction.run_id != planned_run_id:
        raise ValueError("historical checkpoint transaction run identity mismatch")
    if resume_transaction.segment_lineage.parent_transaction_id != stop_transaction.transaction_id:
        raise ValueError("resume checkpoint does not name the exact stop transaction parent")
    stop_completed = stop_transaction.completed_training_batches
    resume_completed = resume_transaction.completed_training_batches
    if stop_completed is None or resume_completed is None:
        raise ValueError("historical checkpoint transaction has no completed batch authority")
    if (
        stop_transaction.segment_lineage.start_batch
        + stop_transaction.segment_lineage.segment_batch_count
        != stop_completed
    ):
        raise ValueError("stop checkpoint segment offset is inconsistent")
    if resume_transaction.segment_lineage.start_batch != stop_completed:
        raise ValueError("resume checkpoint start batch does not continue the stop transaction")
    if (
        resume_transaction.segment_lineage.start_batch
        + resume_transaction.segment_lineage.segment_batch_count
        != resume_completed
    ):
        raise ValueError("resume checkpoint segment offset is inconsistent")
    if stop_manifest.completed_batches != stop_completed:
        raise ValueError("stop manifest and checkpoint completed batches mismatch")
    if resume_manifest.completed_batches != resume_completed:
        raise ValueError("resume manifest and checkpoint completed batches mismatch")
    if stop_resolved.slots["completed_batches"] != stop_completed:
        raise ValueError("stop checkpoint decoded completed_batches mismatch")
    if resume_resolved.slots["completed_batches"] != resume_completed:
        raise ValueError("resume checkpoint decoded completed_batches mismatch")
    return {
        "source": source,
        "planned_run_id": planned_run_id,
        "stop_manifest_raw": stop_manifest_raw,
        "registration_raw": registration_raw,
        "conformance_raw": conformance_raw,
        "certificate_sha256": certificate_sha256,
        "stop_completed_batches": stop_completed,
        "resume_completed_batches": resume_completed,
    }


def _directory_bytes_equal(left: Path, right: Path) -> bool:
    """Return whether two publication directories contain identical file bytes."""
    if not left.is_dir() or not right.is_dir():
        return False
    left_files = {path.relative_to(left) for path in left.rglob("*") if path.is_file()}
    right_files = {path.relative_to(right) for path in right.rglob("*") if path.is_file()}
    return left_files == right_files and all(
        (left / relative).read_bytes() == (right / relative).read_bytes() for relative in left_files
    )


def _publish_staged_directories(
    publications: list[tuple[Path, Path]],
    *,
    backup_root: Path,
) -> None:
    """Publish staged directories as one rollback-capable local transaction."""
    committed: list[tuple[Path, Path | None]] = []
    backup_root.mkdir(parents=True, exist_ok=True)
    try:
        for index, (staged, final) in enumerate(publications):
            if final.exists() and _directory_bytes_equal(staged, final):
                continue
            final.parent.mkdir(parents=True, exist_ok=True)
            backup = backup_root / str(index) if final.exists() else None
            if backup is not None:
                final.replace(backup)
            try:
                staged.replace(final)
            except Exception:
                if backup is not None and backup.exists():
                    backup.replace(final)
                raise
            committed.append((final, backup))
    except Exception:
        for final, backup in reversed(committed):
            if final.exists():
                shutil.rmtree(final)
            if backup is not None and backup.exists():
                backup.replace(final)
        raise


def map_registered_run_set(
    run_set_dir: Path,
    *,
    repo_root: Path,
    issue: str,
    run_prefix: str,
    immutable_artifact_root: Path,
    immutable_artifact_blob_provider_spec: (
        ImmutableArtifactBlobProviderSpec | Mapping[str, object] | None
    ) = None,
    historical_checkpoint_authorities: Sequence[
        HistoricalCheckpointAuthoritySource | Mapping[str, object]
    ] = (),
    checkpoint_execution_context: StagedExecutionContext | None = None,
) -> tuple[Path, ...]:
    """Idempotently materialize REGISTERed rows for ``post_run.sh``."""
    registration_raw, registration = _read_json_object(
        run_set_dir / "registration.json", label="registration"
    )
    if registration.get("status") != "completed":
        raise ValueError("orchestrated post-run mapping requires completed registration")
    run_set_id = str(registration.get("run_set_id", ""))
    conformance_raw, conformance_payload = _read_json_object(
        run_set_dir / "conformance.json", label="conformance certificate"
    )
    try:
        conformance = RunConformanceCertificate.model_validate(conformance_payload)
    except Exception as exc:
        raise ValueError("run set has invalid conformance certificate") from exc
    if conformance.run_set_id != run_set_id:
        raise ValueError("registration and conformance run_set_id mismatch")
    if conformance.overall != "pass" or registration.get("certificate_overall") != "pass":
        raise ValueError("completed registration requires passing conformance")
    if _sha256(conformance_raw) != registration.get("certificate_sha256"):
        raise ValueError("registration certificate_sha256 does not match conformance bytes")

    _bundle_raw, bundle = _read_json_object(run_set_dir / "bundle.json", label="run bundle")
    provider_spec = (
        ImmutableArtifactBlobProviderSpec()
        if immutable_artifact_blob_provider_spec is None
        else ImmutableArtifactBlobProviderSpec.model_validate(immutable_artifact_blob_provider_spec)
    )
    provider = open_immutable_artifact_blob_provider(
        provider_spec,
        explicit_root=immutable_artifact_root,
    )
    authority_by_row: dict[str, HistoricalCheckpointAuthoritySource] = {}
    for authority_payload in historical_checkpoint_authorities:
        authority = HistoricalCheckpointAuthoritySource.model_validate(authority_payload)
        if authority.mapped_row_id in authority_by_row:
            raise ValueError(
                f"duplicate historical checkpoint authority for {authority.mapped_row_id!r}"
            )
        authority_by_row[authority.mapped_row_id] = authority
    if authority_by_row and checkpoint_execution_context is None:
        raise ValueError(
            "historical checkpoint authorities require an explicit checkpoint_execution_context"
        )
    prepared: list[dict[str, object]] = []
    # Validate every row and authoritative source byte before publishing mapped files.
    for row in bundle["rows"]:
        row_id = str(row["row_id"])
        execution_hash = str(row["execution"]["execution_capsule"]["execution_hash"])
        source = run_set_dir / "collected" / row_id
        if not (source / "training_summary.json").is_file():
            raise ValueError(f"collected row {row_id!r} is missing training_summary.json")
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"collected row {row_id!r} is missing manifest.json")
        manifest_raw = manifest_path.read_bytes()
        manifest, diagnostics_source_ref, diagnostics, diagnostics_raw, manifest_metadata = (
            _training_manifest_ref(
                manifest_raw,
                run_set_id=run_set_id,
                row_id=row_id,
                execution_hash=execution_hash,
                registration=registration,
            )
        )
        collected_diagnostics = source / "training-diagnostics.json"
        if (
            collected_diagnostics.is_file()
            and collected_diagnostics.read_bytes() != diagnostics_raw
        ):
            raise ValueError(
                f"collected row {row_id!r} diagnostics differ from authoritative artifact"
            )
        authority = authority_by_row.pop(row_id, None)
        item: dict[str, object] = {
            "row_id": row_id,
            "execution_hash": execution_hash,
            "source": source,
            "manifest": manifest,
            "manifest_raw": manifest_raw,
            "manifest_metadata": manifest_metadata,
            "diagnostics_source_ref": diagnostics_source_ref,
            "diagnostics": diagnostics,
            "diagnostics_raw": diagnostics_raw,
            "checkpoint_evidence": _checkpoint_lineage_evidence(
                manifest,
                allow_incomplete_refs=authority is not None,
            ),
        }
        if authority is not None:
            assert checkpoint_execution_context is not None
            item["historical_checkpoint_authority"] = _preflight_historical_checkpoint_authority(
                authority,
                resume_manifest=manifest,
                checkpoint_execution_context=checkpoint_execution_context,
            )
        prepared.append(item)

    if authority_by_row:
        unknown = ", ".join(sorted(authority_by_row))
        raise ValueError(f"historical checkpoint authority names unmapped rows: {unknown}")

    for item in prepared:
        manifest = item["manifest"]
        diagnostics_source_ref = item["diagnostics_source_ref"]
        diagnostics = item["diagnostics"]
        if not isinstance(manifest, TrainingRunManifest):
            raise TypeError("prepared manifest has unexpected type")
        if not isinstance(diagnostics_source_ref, ArtifactRef):
            raise TypeError("prepared diagnostics ref has unexpected type")
        if not isinstance(diagnostics, TrainingDiagnostics):
            raise TypeError("prepared diagnostics have unexpected type")
        manifest_raw = item["manifest_raw"]
        diagnostics_raw = item["diagnostics_raw"]
        if not isinstance(manifest_raw, bytes) or not isinstance(diagnostics_raw, bytes):
            raise TypeError("prepared provider payload has unexpected type")
        row_id = str(item["row_id"])
        manifest_artifact = provider.store_bytes(
            manifest_raw,
            role="training_run",
            logical_name=f"{manifest.id}.json",
            media_type="application/json",
            metadata={
                "manifest_id": manifest.id,
                "run_set_id": run_set_id,
                "row_id": row_id,
                "planned_run_id": manifest.job_id,
            },
        )
        if provider.get_bytes(manifest_artifact) != manifest_raw:
            raise ValueError("TrainingRunManifest provider did not preserve exact bytes")
        item["manifest_ref"] = {
            "kind": manifest.kind,
            "id": manifest.id,
            "role": "training_run",
            "uri": manifest_artifact.uri,
            "metadata": item["manifest_metadata"],
        }
        diagnostics_metadata = {
            "schema_id": TRAINING_DIAGNOSTICS_SCHEMA_ID,
            "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
            "training_manifest_id": manifest.id,
            "run_set_id": run_set_id,
            "row_id": row_id,
            "planned_run_id": manifest.job_id,
            "run_id": diagnostics.run_id,
        }
        mapped_diagnostics_ref = provider.store_bytes(
            diagnostics_raw,
            role=diagnostics_source_ref.role,
            logical_name=diagnostics_source_ref.logical_name,
            media_type=diagnostics_source_ref.media_type,
            metadata=diagnostics_metadata,
        )
        if provider.get_bytes(mapped_diagnostics_ref) != diagnostics_raw:
            raise ValueError("training diagnostics provider did not preserve exact bytes")
        item["diagnostics_ref"] = mapped_diagnostics_ref.model_dump(mode="json", exclude_none=True)
        historical = item.get("historical_checkpoint_authority")
        if historical is not None:
            if not isinstance(historical, Mapping):
                raise TypeError("prepared historical checkpoint authority has unexpected type")
            authority_source = historical["source"]
            if not isinstance(authority_source, HistoricalCheckpointAuthoritySource):
                raise TypeError("prepared historical checkpoint authority source is invalid")
            authority_metadata = {
                "mapped_row_id": authority_source.mapped_row_id,
                "stop_run_set_id": authority_source.stop_run_set_id,
                "stop_row_id": authority_source.stop_row_id,
                "planned_run_id": historical["planned_run_id"],
            }
            stop_manifest_artifact = provider.store_bytes(
                historical["stop_manifest_raw"],  # type: ignore[arg-type]
                role="historical_stop_training_run_manifest",
                logical_name="manifest.json",
                media_type="application/json",
                metadata=authority_metadata,
            )
            stop_registration_artifact = provider.store_bytes(
                historical["registration_raw"],  # type: ignore[arg-type]
                role="historical_stop_registration",
                logical_name="registration.json",
                media_type="application/json",
                metadata={**authority_metadata, "status": "stopped"},
            )
            stop_conformance_artifact = provider.store_bytes(
                historical["conformance_raw"],  # type: ignore[arg-type]
                role="historical_stop_conformance_certificate",
                logical_name="conformance.json",
                media_type="application/json",
                metadata={
                    **authority_metadata,
                    "overall": "pass",
                    "certificate_sha256": historical["certificate_sha256"],
                },
            )
            for ref, raw, label in (
                (stop_manifest_artifact, historical["stop_manifest_raw"], "stop manifest"),
                (stop_registration_artifact, historical["registration_raw"], "stop registration"),
                (stop_conformance_artifact, historical["conformance_raw"], "stop conformance"),
            ):
                if provider.get_bytes(ref) != raw:
                    raise ValueError(f"historical {label} provider did not preserve exact bytes")
            item["mapped_historical_checkpoint_authority"] = MappedHistoricalCheckpointAuthority(
                mapped_row_id=authority_source.mapped_row_id,
                stop_run_set_id=authority_source.stop_run_set_id,
                stop_row_id=authority_source.stop_row_id,
                planned_run_id=str(historical["planned_run_id"]),
                stop_training_manifest_artifact_ref=stop_manifest_artifact,
                stop_registration_artifact_ref=stop_registration_artifact,
                stop_conformance_artifact_ref=stop_conformance_artifact,
                stop_checkpoint_ref=authority_source.stop_checkpoint_ref,
                resume_checkpoint_ref=authority_source.resume_checkpoint_ref,
                checkpoint_custody_binding=authority_source.checkpoint_custody_binding,
                checkpoint_custody_spec=StagedCheckpointCustodySpec(
                    backend="feedbax-checkpoint-transaction-tree"
                ),
                stop_completed_batches=int(historical["stop_completed_batches"]),
                resume_completed_batches=int(historical["resume_completed_batches"]),
            )

    issue_root = repo_root / "_artifacts" / issue
    final_targets = [issue_root / "runs" / f"{run_prefix}__{item['row_id']}" for item in prepared]
    outputs = tuple(target / "run.json" for target in final_targets)
    issue_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mapped-run-stage-", dir=issue_root) as temporary:
        staging_root = Path(temporary)
        common_metadata = {"run_set_id": run_set_id}
        registration_ref = _store_evidence(
            registration_raw,
            provider=provider,
            role="orchestration_registration",
            logical_name="registration.json",
            metadata=common_metadata,
        )
        conformance_ref = _store_evidence(
            conformance_raw,
            provider=provider,
            role="orchestration_conformance_certificate",
            logical_name="conformance.json",
            metadata=common_metadata,
        )

        staged_rows: list[Path] = []
        for item, target in zip(prepared, final_targets, strict=True):
            row_id = str(item["row_id"])
            execution_hash = str(item["execution_hash"])
            source = Path(item["source"])
            staged_target = staging_root / "rows" / target.name
            staged_target.mkdir(parents=True)
            source_path = source / "training_summary.json"
            staged_path = staged_target / "training_summary.json"
            shutil.copy2(source_path, staged_path)
            if staged_path.read_bytes() != source_path.read_bytes():
                raise ValueError(
                    f"staged collected row {row_id!r} training_summary.json failed verification"
                )
            reviewer_convenience_paths = {
                "training_summary": str((target / "training_summary.json").relative_to(repo_root))
            }

            checkpoint_refs = _store_checkpoint_lineage(
                item["checkpoint_evidence"],  # type: ignore[arg-type]
                provider=provider,
                run_set_id=run_set_id,
                row_id=row_id,
            )
            recipe = {
                "schema_id": "rlrmp.spec.orchestrated_post_run",
                "schema_version": "rlrmp.spec.orchestrated_post_run.v3",
                "issue": issue,
                "run_set_id": run_set_id,
                "row_id": row_id,
                "certificate_sha256": registration["certificate_sha256"],
                "execution_hash": execution_hash,
                "registration_artifact_ref": registration_ref,
                "conformance_artifact_ref": conformance_ref,
                "training_manifest_ref": item["manifest_ref"],
                "training_diagnostics_artifact_ref": item["diagnostics_ref"],
                "immutable_artifact_blob_provider_spec": provider_spec.model_dump(
                    mode="json", exclude_none=True
                ),
                "checkpoint_lineage_refs": checkpoint_refs,
                "reviewer_convenience_paths": reviewer_convenience_paths,
            }
            mapped_authority = item.get("mapped_historical_checkpoint_authority")
            if mapped_authority is not None:
                if not isinstance(mapped_authority, MappedHistoricalCheckpointAuthority):
                    raise TypeError("mapped historical checkpoint authority has unexpected type")
                recipe["historical_checkpoint_authority"] = mapped_authority.model_dump(
                    mode="json", exclude_none=True
                )
            recipe_path = staged_target / "run.json"
            encoded = json.dumps(recipe, indent=2, sort_keys=True) + "\n"
            recipe_path.write_text(encoded, encoding="utf-8")
            if recipe_path.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"staged collected row {row_id!r} recipe failed verification")
            staged_rows.append(staged_target)

        _publish_staged_directories(
            list(zip(staged_rows, final_targets, strict=True)),
            backup_root=staging_root / "backups",
        )
    return outputs


__all__ = [
    "HistoricalCheckpointAuthoritySource",
    "MappedHistoricalCheckpointAuthority",
    "map_registered_run_set",
]

"""Map completed orchestrated rows into the established post-run layout."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse

from feedbax.contracts.checkpoints import CheckpointTransactionManifest
from feedbax.contracts.manifest import ArtifactRef, TrainingRunManifest
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
) -> list[tuple[bytes, CheckpointTransactionManifest, str]]:
    """Verify exact transaction manifests and return them parent before child."""
    if not manifest.checkpoint_custody:
        raise ValueError("TrainingRunManifest has no checkpoint custody evidence")
    declared_raw: dict[str, bytes] = {}
    declared_paths: dict[str, Path | None] = {}
    declared_hashes: dict[str, str] = {}
    relationships: dict[str, str] = {}
    for ref in manifest.checkpoint_custody:
        transaction_id = getattr(ref, "id", None)
        if ref.kind != "TrainingCheckpointTransactionManifest" or not isinstance(
            transaction_id, str
        ):
            raise ValueError("TrainingRunManifest checkpoint custody requires manifest refs")
        ref_payload = ref.model_dump(mode="json", exclude_none=True)
        raw, path = _resolve_ref_bytes(
            ref_payload,
            context=f"checkpoint transaction {transaction_id!r}",
        )
        declared_raw[transaction_id] = raw
        declared_paths[transaction_id] = path
        digest = ref.metadata.get("manifest_sha256")
        if isinstance(digest, str):
            declared_hashes[transaction_id] = digest
        relationships.setdefault(transaction_id, "parent")

    loaded: dict[str, tuple[bytes, CheckpointTransactionManifest]] = {}

    def load(transaction_id: str, *, child: CheckpointTransactionManifest | None = None) -> None:
        if transaction_id in loaded:
            return
        raw = declared_raw.get(transaction_id)
        path = declared_paths.get(transaction_id)
        expected_hash = declared_hashes.get(transaction_id)
        relationship = relationships.get(transaction_id, "parent")
        if raw is None and child is not None:
            lineage = next(
                (item for item in child.parent_lineage if item.transaction_id == transaction_id),
                None,
            )
            if lineage is not None:
                relationship = lineage.relationship
                relationships[transaction_id] = relationship
                manifest_ref = lineage.manifest
                if manifest_ref is not None:
                    ref_payload = manifest_ref.model_dump(mode="json", exclude_none=True)
                    digest = getattr(manifest_ref, "sha256", None) or manifest_ref.metadata.get(
                        "manifest_sha256"
                    )
                    if isinstance(digest, str):
                        expected_hash = digest
                    if manifest_ref.uri:
                        raw, path = _resolve_ref_bytes(
                            ref_payload,
                            context=f"checkpoint lineage parent {transaction_id!r}",
                        )
                lineage_digest = lineage.metadata.get("manifest_sha256")
                if isinstance(lineage_digest, str):
                    expected_hash = lineage_digest
            source = child.fork_provenance.source if child.fork_provenance is not None else None
            if source is not None and source.transaction_id == transaction_id:
                expected_hash = source.manifest_sha256
                relationships[transaction_id] = "new_lineage_override"
                relationship = "new_lineage_override"
                source_uri = source.metadata.get("manifest_uri")
                custody_root_uri = source.metadata.get("custody_root_uri")
                if raw is None and isinstance(source_uri, str):
                    raw, path = _resolve_ref_bytes(
                        {"uri": source_uri},
                        context=f"fork source transaction {transaction_id!r}",
                    )
                elif (
                    raw is None
                    and isinstance(custody_root_uri, str)
                    and source.manifest_relative_path
                ):
                    uri = custody_root_uri.rstrip("/") + "/" + source.manifest_relative_path
                    raw, path = _resolve_ref_bytes(
                        {"uri": uri},
                        context=f"fork source transaction {transaction_id!r}",
                    )
            if raw is None and expected_hash is not None:
                child_path = declared_paths.get(child.transaction_id)
                if child_path is not None:
                    sibling = child_path.parent.parent / transaction_id / "manifest.json"
                    if sibling.is_file():
                        raw, path = sibling.read_bytes(), sibling
            relationship = relationships.get(transaction_id, relationship)
        if raw is None:
            raise ValueError(
                f"checkpoint lineage parent {transaction_id!r} has no explicit materialization"
            )
        digest = _sha256(raw)
        if expected_hash is None:
            raise ValueError(f"checkpoint transaction {transaction_id!r} has no declared hash")
        if digest != expected_hash:
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
            source = child.fork_provenance.source if child and child.fork_provenance else None
            if (
                source is None
                or source.transaction_id != transaction_id
                or source.run_id != transaction.run_id
            ):
                raise ValueError(f"checkpoint transaction {transaction_id!r} run identity mismatch")
        loaded[transaction_id] = (raw, transaction)
        declared_raw.setdefault(transaction_id, raw)
        declared_paths.setdefault(transaction_id, path)
        declared_hashes.setdefault(transaction_id, digest)
        relationships.setdefault(transaction_id, relationship)
        parent_id = transaction.segment_lineage.parent_transaction_id
        if parent_id is not None:
            fork_source = (
                transaction.fork_provenance.source
                if transaction.fork_provenance is not None
                else None
            )
            if fork_source is not None and fork_source.transaction_id == parent_id:
                relationships[parent_id] = "new_lineage_override"
                declared_hashes.setdefault(parent_id, fork_source.manifest_sha256)
            load(parent_id, child=transaction)
            parent = loaded[parent_id][1]
            if fork_source is not None and fork_source.transaction_id == parent_id:
                if fork_source.run_id != parent.run_id:
                    raise ValueError(f"fork source transaction {parent_id!r} run identity mismatch")
                if (
                    fork_source.transaction_root_sha256
                    != parent.content_integrity_digest.transaction_root_sha256
                ):
                    raise ValueError(f"fork source transaction {parent_id!r} root hash mismatch")
            parent_end = (
                parent.segment_lineage.start_batch + parent.segment_lineage.segment_batch_count
            )
            if transaction.segment_lineage.start_batch != parent_end:
                raise ValueError(f"checkpoint lineage offset discontinuity at {transaction_id!r}")

    for transaction_id in tuple(declared_raw):
        load(transaction_id)

    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(transaction_id: str) -> None:
        if transaction_id in ordered:
            return
        if transaction_id in visiting:
            raise ValueError(f"checkpoint lineage contains cycle at {transaction_id!r}")
        visiting.add(transaction_id)
        parent_id = loaded[transaction_id][1].segment_lineage.parent_transaction_id
        if parent_id is not None:
            visit(parent_id)
        visiting.remove(transaction_id)
        ordered.append(transaction_id)

    for transaction_id in declared_raw:
        visit(transaction_id)

    return [(loaded[item][0], loaded[item][1], relationships[item]) for item in ordered]


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
        prepared.append(
            {
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
                ),
            }
        )

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


__all__ = ["map_registered_run_set"]

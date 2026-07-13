"""Map completed orchestrated rows into the established post-run layout."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlparse

from feedbax.contracts.checkpoints import CheckpointTransactionManifest
from feedbax.contracts.manifest import ArtifactRef, TrainingRunManifest, store_bytes_artifact
from feedbax.orchestration.conformance import RunConformanceCertificate

ManifestRefResolver = Callable[[Mapping[str, object]], bytes]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _portable_artifact_ref(ref: ArtifactRef) -> dict[str, object]:
    """Return an immutable ref that is independent of this checkout's path."""
    payload = ref.model_dump(mode="json", exclude_none=True)
    payload["uri"] = ref.artifact_id
    return payload


def _store_evidence(
    data: bytes,
    *,
    root: Path,
    role: str,
    logical_name: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    ref = store_bytes_artifact(
        data,
        root=root,
        role=role,
        logical_name=logical_name,
        media_type="application/json",
        suffix=".json",
        metadata=dict(metadata),
    )
    relative = ref.metadata.get("relative_path")
    if not isinstance(relative, str):
        raise ValueError("stored evidence has no materialized relative path")
    materialized = root / relative
    if (
        not materialized.is_file()
        or materialized.stat().st_size != len(data)
        or _sha256(materialized.read_bytes()) != _sha256(data)
    ):
        raise ValueError("stored evidence CAS object failed post-write verification")
    return _portable_artifact_ref(ref)


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
    supplied_ref: Mapping[str, object] | None,
    resolvers: Mapping[str, ManifestRefResolver],
) -> tuple[TrainingRunManifest, dict[str, object]]:
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
    if manifest.job_id != planned_run_id:
        raise ValueError(f"collected row {row_id!r} TrainingRunManifest planned_run_id mismatch")
    if manifest.execution_hash != execution_hash:
        raise ValueError(f"collected row {row_id!r} TrainingRunManifest execution_hash mismatch")
    expected_metadata = {
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
    if supplied_ref is None:
        raise ValueError(
            f"collected row {row_id!r} requires a resolver-backed immutable manifest ref"
        )
    uri = supplied_ref.get("uri")
    scheme = urlparse(uri).scheme if isinstance(uri, str) else ""
    if not uri or not scheme or scheme == "file":
        raise ValueError("TrainingRunManifest ref URI must name an explicit immutable provider")
    resolver = resolvers.get(scheme)
    if resolver is None:
        raise ValueError(f"TrainingRunManifest ref has no resolver for scheme {scheme!r}")
    actual = dict(supplied_ref)
    if set(actual) != {"kind", "id", "role", "uri", "metadata"}:
        raise ValueError("TrainingRunManifest immutable ref has unexpected or missing fields")
    if actual.get("kind") != "TrainingRunManifest":
        raise ValueError("TrainingRunManifest immutable ref kind mismatch")
    if actual.get("id") != manifest.id:
        raise ValueError("TrainingRunManifest immutable ref id mismatch")
    if actual.get("role") != "training_run":
        raise ValueError("TrainingRunManifest immutable ref role mismatch")
    actual_metadata = actual.get("metadata")
    if not isinstance(actual_metadata, Mapping) or dict(actual_metadata) != expected_metadata:
        raise ValueError("TrainingRunManifest immutable ref metadata mismatch")
    resolved_raw = resolver(actual)
    if not isinstance(resolved_raw, bytes):
        raise ValueError("TrainingRunManifest resolver must return exact bytes")
    if (
        len(resolved_raw) != actual_metadata["size_bytes"]
        or _sha256(resolved_raw) != actual_metadata["manifest_sha256"]
    ):
        raise ValueError("TrainingRunManifest resolver returned bytes with wrong size or sha256")
    if resolved_raw != raw:
        raise ValueError("TrainingRunManifest resolver did not materialize the collected bytes")
    try:
        resolved = TrainingRunManifest.model_validate_json(resolved_raw)
    except Exception as exc:
        raise ValueError("TrainingRunManifest resolver returned invalid manifest bytes") from exc
    if (
        resolved.kind != actual["kind"]
        or resolved.id != actual["id"]
        or resolved.status != actual_metadata["manifest_status"]
        or resolved.run_set_id not in (None, actual_metadata["run_set_id"])
        or not isinstance(resolved.metadata.get("training_row_provenance"), Mapping)
        or resolved.metadata["training_row_provenance"].get("row_id") != actual_metadata["row_id"]
        or resolved.metadata["training_row_provenance"].get("planned_run_id")
        != actual_metadata["planned_run_id"]
    ):
        raise ValueError("TrainingRunManifest resolver returned mismatched manifest identity")
    return manifest, actual


def rooted_manifest_ref_resolver(root: Path) -> ManifestRefResolver:
    """Resolve one provider URI beneath an explicit durable root, without search."""
    root = root.resolve()

    def resolve(ref: Mapping[str, object]) -> bytes:
        uri = ref.get("uri")
        if not isinstance(uri, str):
            raise ValueError("manifest ref URI must be a string")
        parsed = urlparse(uri)
        relative = Path(parsed.netloc, parsed.path.lstrip("/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest ref URI escapes its explicit provider root")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("manifest ref URI escapes its explicit provider root")
        if not path.is_file():
            raise ValueError(f"manifest ref is not materialized by provider: {uri}")
        return path.read_bytes()

    return resolve


def _resolve_ref_bytes(
    ref: Mapping[str, object],
    *,
    resolvers: Mapping[str, ManifestRefResolver],
    context: str,
) -> tuple[bytes, Path | None]:
    uri = ref.get("uri")
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"{context} has no explicit URI")
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = Path(parsed.path)
    elif not parsed.scheme:
        path = Path(uri)
    else:
        resolver = resolvers.get(parsed.scheme)
        if resolver is None:
            raise ValueError(f"{context} has no resolver for scheme {parsed.scheme!r}")
        raw = resolver(ref)
        if not isinstance(raw, bytes):
            raise ValueError(f"{context} resolver must return exact bytes")
        return raw, None
    if not path.is_file():
        raise ValueError(f"{context} is not materialized: {path}")
    return path.read_bytes(), path


def _checkpoint_lineage_evidence(
    manifest: TrainingRunManifest,
    *,
    resolvers: Mapping[str, ManifestRefResolver],
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
            resolvers=resolvers,
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
                            resolvers=resolvers,
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
                        resolvers=resolvers,
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
                        resolvers=resolvers,
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
    evidence_root: Path,
    run_set_id: str,
    row_id: str,
) -> list[dict[str, object]]:
    """Custody-store already verified transaction-manifest bytes."""

    refs: list[dict[str, object]] = []
    for raw, transaction, relationship in evidence:
        transaction_id = transaction.transaction_id
        artifact = _store_evidence(
            raw,
            root=evidence_root,
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
    training_manifest_refs: Mapping[str, Mapping[str, object]] | None = None,
    manifest_ref_resolvers: Mapping[str, ManifestRefResolver] | None = None,
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
    resolvers = manifest_ref_resolvers or {}
    prepared: list[dict[str, object]] = []
    # Validate every row and externally resolved byte before the first mapped write.
    # This keeps a missing provider/ref from leaving a half-published packet.
    for row in bundle["rows"]:
        row_id = str(row["row_id"])
        execution_hash = str(row["execution"]["execution_capsule"]["execution_hash"])
        source = run_set_dir / "collected" / row_id
        for name in ("training-diagnostics.json", "training_summary.json"):
            if not (source / name).is_file():
                raise ValueError(f"collected row {row_id!r} is missing {name}")
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"collected row {row_id!r} is missing manifest.json")
        manifest, manifest_ref = _training_manifest_ref(
            manifest_path.read_bytes(),
            run_set_id=run_set_id,
            row_id=row_id,
            execution_hash=execution_hash,
            registration=registration,
            supplied_ref=(training_manifest_refs or {}).get(row_id),
            resolvers=resolvers,
        )
        prepared.append(
            {
                "row_id": row_id,
                "execution_hash": execution_hash,
                "source": source,
                "manifest_ref": manifest_ref,
                "checkpoint_evidence": _checkpoint_lineage_evidence(
                    manifest,
                    resolvers=resolvers,
                ),
            }
        )

    issue_root = repo_root / "_artifacts" / issue
    evidence_root = issue_root / "run_sets" / run_set_id / "evidence"
    final_targets = [issue_root / "runs" / f"{run_prefix}__{item['row_id']}" for item in prepared]
    outputs = tuple(target / "run.json" for target in final_targets)
    issue_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mapped-run-stage-", dir=issue_root) as temporary:
        staging_root = Path(temporary)
        staged_evidence_root = staging_root / "evidence"
        common_metadata = {"run_set_id": run_set_id}
        registration_ref = _store_evidence(
            registration_raw,
            root=staged_evidence_root,
            role="orchestration_registration",
            logical_name="registration.json",
            metadata=common_metadata,
        )
        conformance_ref = _store_evidence(
            conformance_raw,
            root=staged_evidence_root,
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
            copied: dict[str, str] = {}
            for name in ("training-diagnostics.json", "training_summary.json"):
                source_path = source / name
                staged_path = staged_target / name
                shutil.copy2(source_path, staged_path)
                if staged_path.read_bytes() != source_path.read_bytes():
                    raise ValueError(f"staged collected row {row_id!r} {name} failed verification")
                copied[name] = str((target / name).relative_to(repo_root))

            checkpoint_refs = _store_checkpoint_lineage(
                item["checkpoint_evidence"],  # type: ignore[arg-type]
                evidence_root=staged_evidence_root,
                run_set_id=run_set_id,
                row_id=row_id,
            )
            recipe = {
                "schema_id": "rlrmp.spec.orchestrated_post_run",
                "schema_version": "rlrmp.spec.orchestrated_post_run.v2",
                "issue": issue,
                "run_set_id": run_set_id,
                "row_id": row_id,
                "certificate_sha256": registration["certificate_sha256"],
                "execution_hash": execution_hash,
                "evidence_custody_root": str(evidence_root.relative_to(repo_root)),
                "registration_artifact_ref": registration_ref,
                "conformance_artifact_ref": conformance_ref,
                "training_manifest_ref": item["manifest_ref"],
                "checkpoint_lineage_refs": checkpoint_refs,
                "source_paths": copied,
            }
            recipe_path = staged_target / "run.json"
            encoded = json.dumps(recipe, indent=2, sort_keys=True) + "\n"
            recipe_path.write_text(encoded, encoding="utf-8")
            if recipe_path.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"staged collected row {row_id!r} recipe failed verification")
            staged_rows.append(staged_target)

        if evidence_root.exists() and not _directory_bytes_equal(
            staged_evidence_root, evidence_root
        ):
            raise ValueError("stored evidence CAS object failed post-write verification")
        _publish_staged_directories(
            [(staged_evidence_root, evidence_root), *zip(staged_rows, final_targets, strict=True)],
            backup_root=staging_root / "backups",
        )
    return outputs


__all__ = ["map_registered_run_set", "rooted_manifest_ref_resolver"]

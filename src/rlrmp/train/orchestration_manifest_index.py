"""Exact indexes for registered orchestration training manifests.

The tracked index is authority for bundle membership. Predicate selection may
describe intent, but execution consumes the exact ordered references recorded
here. The tracked index is ref-only and no second durable manifest identity/copy
is published; transient hash-verified regular files are materialized atomically
for Feedbax discovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feedbax.analysis import (
    STAGED_EXACT_PARENTS_SCHEMA_ID,
    STAGED_EXACT_PARENTS_SCHEMA_VERSION,
    StagedExactParentEntry,
    StagedExactParents,
    resolve_evaluation_inputs,
)
from feedbax.contracts.manifest import (
    ArtifactRef,
    EvaluationRunSpec,
    ParentRef,
    TrainingRunManifest,
    load_manifest,
    safe_manifest_key,
)
from feedbax.orchestration import RunConformanceCertificate, build_core_check_registry
from feedbax.persistence import ImmutableArtifactBlobProvider

INDEX_SCHEMA_ID = "rlrmp.spec.orchestration_manifest_index"
INDEX_SCHEMA_VERSION = "rlrmp.spec.orchestration_manifest_index.v1"


class OrchestrationManifestIndexError(ValueError):
    """Raised when an orchestration manifest index fails closed."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OrchestrationTrainingManifestMetadata(_StrictModel):
    """Typed metadata carried by a ParentRef-compatible manifest reference."""

    manifest_sha256: str
    size_bytes: int = Field(ge=0)
    run_set_id: str
    row_id: str
    manifest_status: Literal["completed"] = "completed"
    registration_status: Literal["completed"] = "completed"
    conformance_overall: Literal["pass"] = "pass"
    certificate_sha256: str
    planned_run_id: str

    @field_validator("manifest_sha256", "certificate_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("expected a lowercase SHA-256 digest")
        return value


class OrchestrationTrainingManifestRef(_StrictModel):
    """An exact immutable TrainingRunManifest reference.

    Its JSON shape is accepted directly by Feedbax ``ParentRef``: the custody
    and orchestration evidence is nested under ``metadata`` rather than added
    as incompatible top-level fields.
    """

    kind: Literal["TrainingRunManifest"] = "TrainingRunManifest"
    id: str
    role: Literal["training_run"] = "training_run"
    uri: str
    metadata: OrchestrationTrainingManifestMetadata

    @field_validator("id", "uri")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("manifest id and URI must be non-empty")
        return value

    def to_parent_ref(self) -> ParentRef:
        """Return the exact Feedbax ParentRef representation."""

        return ParentRef.model_validate(self.model_dump(mode="json"))


class OrchestrationManifestIndexSpec(_StrictModel):
    """Frozen exact ordered membership for one bundle instance."""

    schema_id: Literal[INDEX_SCHEMA_ID] = INDEX_SCHEMA_ID
    schema_version: Literal[INDEX_SCHEMA_VERSION] = INDEX_SCHEMA_VERSION
    index_id: str
    bundle_id: str
    refs: tuple[OrchestrationTrainingManifestRef, ...]

    @model_validator(mode="after")
    def _validate_exact_membership(self) -> "OrchestrationManifestIndexSpec":
        if not self.refs:
            raise ValueError("orchestration manifest index must contain at least one ref")
        row_keys = [(ref.metadata.run_set_id, ref.metadata.row_id) for ref in self.refs]
        if len(row_keys) != len(set(row_keys)):
            raise ValueError("duplicate or ambiguous run-set/row refs")
        ids = [ref.id for ref in self.refs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate or ambiguous manifest ids")
        uris = [ref.uri for ref in self.refs]
        if len(uris) != len(set(uris)):
            raise ValueError("duplicate or ambiguous manifest URIs")
        return self

    def canonical_bytes(self) -> bytes:
        """Return deterministic canonical JSON with no manifest payload bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True)
class RegisteredRunSetLocation:
    """Explicit location and exact row contract for one registered run set."""

    run_set_dir: Path
    expected_row_ids: tuple[str, ...]
    manifest_uris: Mapping[str, str]


@dataclass(frozen=True)
class MaterializedOrchestrationManifestIndex:
    """Transient Feedbax discovery root for an exact index."""

    root: Path
    index_path: Path
    manifest_paths: tuple[Path, ...]
    parent_refs: tuple[ParentRef, ...]
    exact_parents: StagedExactParents


def build_orchestration_manifest_index(
    locations: Sequence[RegisteredRunSetLocation],
    *,
    index_id: str,
    bundle_id: str,
    manifest_provider: ImmutableArtifactBlobProvider,
) -> OrchestrationManifestIndexSpec:
    """Build an exact index from explicit registered run-set locations.

    Run-set discovery is intentionally absent.  Every run-set directory, row,
    and immutable URI is supplied by the caller and cross-validated against the
    registration, bundle, conformance certificate, collected manifest, and
    custody bytes.
    """

    if not locations:
        raise OrchestrationManifestIndexError("at least one registered run set is required")
    refs: list[OrchestrationTrainingManifestRef] = []
    seen_ids: dict[str, str] = {}
    seen_rows: set[tuple[str, str]] = set()
    for location in locations:
        run_set_refs = _refs_from_registered_run_set(
            location,
            manifest_provider=manifest_provider,
        )
        for ref in run_set_refs:
            row_key = (ref.metadata.run_set_id, ref.metadata.row_id)
            if row_key in seen_rows:
                raise OrchestrationManifestIndexError(
                    f"duplicate or ambiguous ref for run-set/row {row_key!r}"
                )
            previous_hash = seen_ids.get(ref.id)
            if previous_hash is not None:
                detail = (
                    "different bytes"
                    if previous_hash != ref.metadata.manifest_sha256
                    else "duplicate"
                )
                raise OrchestrationManifestIndexError(
                    f"same TrainingRunManifest id has {detail}: {ref.id!r}"
                )
            seen_rows.add(row_key)
            seen_ids[ref.id] = ref.metadata.manifest_sha256
            refs.append(ref)
    refs.sort(key=lambda ref: (ref.metadata.run_set_id, ref.metadata.row_id, ref.id))
    return OrchestrationManifestIndexSpec(
        index_id=index_id,
        bundle_id=bundle_id,
        refs=tuple(refs),
    )


def write_orchestration_manifest_index(
    index: OrchestrationManifestIndexSpec,
    path: Path | str,
    *,
    manifest_provider: ImmutableArtifactBlobProvider,
) -> Path:
    """Write a deterministic tracked index after requiring durable providers."""

    _validate_index_bytes(index, manifest_provider=manifest_provider)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(output, index.canonical_bytes())
    return output


def load_orchestration_manifest_index(
    path: Path | str,
    *,
    manifest_provider: ImmutableArtifactBlobProvider,
) -> OrchestrationManifestIndexSpec:
    """Load a tracked exact index and validate every referenced manifest byte."""

    index = OrchestrationManifestIndexSpec.model_validate_json(Path(path).read_bytes())
    _validate_index_bytes(
        index,
        manifest_provider=manifest_provider,
    )
    return index


def select_exact_training_manifest_refs(
    index: OrchestrationManifestIndexSpec,
    *,
    expected_rows: Sequence[tuple[str, str]],
) -> tuple[ParentRef, ...]:
    """Select only when requested row membership exactly equals the frozen index."""

    actual = [(ref.metadata.run_set_id, ref.metadata.row_id) for ref in index.refs]
    expected = list(expected_rows)
    if len(expected) != len(set(expected)):
        raise OrchestrationManifestIndexError("expected bundle rows contain duplicates")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra or len(expected) != len(actual):
        raise OrchestrationManifestIndexError(
            f"exact bundle membership mismatch: missing={missing!r}, extra={extra!r}"
        )
    if expected != actual:
        raise OrchestrationManifestIndexError(
            f"exact bundle row order mismatch: expected={expected!r}, actual={actual!r}"
        )
    return tuple(ref.to_parent_ref() for ref in index.refs)


def materialize_orchestration_manifest_index(
    index: OrchestrationManifestIndexSpec,
    *,
    manifest_provider: ImmutableArtifactBlobProvider,
    target_root: Path | str,
    expected_rows: Sequence[tuple[str, str]],
) -> MaterializedOrchestrationManifestIndex:
    """Atomically materialize hash-verified regular files for Feedbax discovery.

    The tracked index remains ref-only and this operation publishes no second
    durable manifest identity or durable copy. The isolated root is transient
    execution state populated from the immutable provider.
    """

    parent_refs = select_exact_training_manifest_refs(index, expected_rows=expected_rows)
    _validate_index_bytes(
        index,
        manifest_provider=manifest_provider,
    )
    root = Path(target_root)
    if root.exists():
        raise OrchestrationManifestIndexError("target Feedbax root must be absent")
    target_names = [f"{safe_manifest_key(ref.id)}.json" for ref in index.refs]
    if len(target_names) != len(set(target_names)):
        raise OrchestrationManifestIndexError("manifest ids collide after safe_manifest_key")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    try:
        manifests_dir = staging / "manifests" / "training_runs"
        manifests_dir.mkdir(parents=True)
        staged_paths: list[Path] = []
        for ref, target_name in zip(index.refs, target_names, strict=True):
            target = manifests_dir / target_name
            manifest_provider.materialize(_artifact_ref(ref), target)
            target_stat = target.lstat()
            if (
                target.is_symlink()
                or not stat.S_ISREG(target_stat.st_mode)
                or target_stat.st_nlink != 1
            ):
                raise OrchestrationManifestIndexError(
                    "immutable provider must materialize an independent regular manifest file"
                )
            staged_paths.append(target)
        staged_index = staging / "index" / "orchestration_manifest_index.json"
        staged_index.parent.mkdir(parents=True)
        staged_index.write_bytes(index.canonical_bytes())
        for ref, target in zip(index.refs, staged_paths, strict=True):
            _validate_manifest_path(ref, target)
            execution_ref = ref.to_parent_ref().model_copy(
                update={"uri": target.relative_to(staging).as_posix()}
            )
            resolved = resolve_evaluation_inputs(
                EvaluationRunSpec(
                    evaluation_type="rlrmp.internal.orchestration_manifest_materialization",
                    inputs=[execution_ref],
                ),
                manifest_root=staging,
                require_unique_manifest_id=False,
            )[0]
            if (
                resolved.id != ref.id
                or resolved.sha256 != ref.metadata.manifest_sha256
                or resolved.size_bytes != ref.metadata.size_bytes
            ):
                raise OrchestrationManifestIndexError(
                    f"Feedbax resolver disagrees with exact manifest ref {ref.id!r}"
                )
        if root.exists():
            raise OrchestrationManifestIndexError("target Feedbax root appeared during staging")
        os.rename(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    paths = tuple(root / "manifests" / "training_runs" / name for name in target_names)
    index_path = root / "index" / "orchestration_manifest_index.json"
    exact_parents = StagedExactParents(
        schema_id=STAGED_EXACT_PARENTS_SCHEMA_ID,
        schema_version=STAGED_EXACT_PARENTS_SCHEMA_VERSION,
        parents=[
            StagedExactParentEntry(
                parent=ref.to_parent_ref(),
                execution_uri=path.relative_to(root).as_posix(),
            )
            for ref, path in zip(index.refs, paths, strict=True)
        ],
    )
    return MaterializedOrchestrationManifestIndex(
        root=root,
        index_path=index_path,
        manifest_paths=paths,
        parent_refs=parent_refs,
        exact_parents=exact_parents,
    )


def _refs_from_registered_run_set(
    location: RegisteredRunSetLocation,
    *,
    manifest_provider: ImmutableArtifactBlobProvider,
) -> tuple[OrchestrationTrainingManifestRef, ...]:
    run_set_dir = location.run_set_dir
    registration = _read_json(run_set_dir / "registration.json", label="registration")
    bundle = _read_json(run_set_dir / "bundle.json", label="bundle")
    certificate_path = run_set_dir / "conformance.json"
    certificate_payload = _read_json(certificate_path, label="conformance certificate")
    run_set_id = _required_string(registration, "run_set_id", "registration")
    if registration.get("status") != "completed":
        raise OrchestrationManifestIndexError(f"registration for {run_set_id!r} is not completed")
    if registration.get("certificate_overall") != "pass":
        raise OrchestrationManifestIndexError(f"certificate for {run_set_id!r} did not pass")
    certificate_sha256 = _required_string(registration, "certificate_sha256", "registration")
    if _sha256_file(certificate_path) != certificate_sha256:
        raise OrchestrationManifestIndexError(
            f"certificate hash mismatch for run set {run_set_id!r}"
        )
    try:
        certificate = RunConformanceCertificate.model_validate(certificate_payload)
    except ValueError as exc:
        raise OrchestrationManifestIndexError(
            f"invalid conformance certificate for {run_set_id!r}: {exc}"
        ) from exc
    if certificate.overall != "pass":
        raise OrchestrationManifestIndexError(f"conformance for {run_set_id!r} did not pass")
    if certificate.run_set_id != run_set_id:
        raise OrchestrationManifestIndexError("certificate run_set_id disagrees with registration")
    bundle_run_set_id = _required_string(bundle, "run_set_id", "bundle")
    if bundle_run_set_id != run_set_id:
        raise OrchestrationManifestIndexError("bundle run_set_id disagrees with registration")

    rows = bundle.get("rows")
    if not isinstance(rows, list):
        raise OrchestrationManifestIndexError("bundle rows must be a list")
    row_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise OrchestrationManifestIndexError("bundle row must be an object")
        row_id = _required_string(raw_row, "row_id", "bundle row")
        if row_id in row_by_id:
            raise OrchestrationManifestIndexError(f"duplicate bundle row {row_id!r}")
        row_by_id[row_id] = raw_row
    expected_rows = list(location.expected_row_ids)
    if len(expected_rows) != len(set(expected_rows)):
        raise OrchestrationManifestIndexError("expected row ids contain duplicates")
    missing = sorted(set(expected_rows) - set(row_by_id))
    extra = sorted(set(row_by_id) - set(expected_rows))
    if missing or extra:
        raise OrchestrationManifestIndexError(
            f"exact run-set row mismatch for {run_set_id!r}: missing={missing!r}, extra={extra!r}"
        )
    uri_rows = set(location.manifest_uris)
    if uri_rows != set(expected_rows):
        raise OrchestrationManifestIndexError(
            "manifest URI rows must exactly match expected rows: "
            f"missing={sorted(set(expected_rows) - uri_rows)!r}, "
            f"extra={sorted(uri_rows - set(expected_rows))!r}"
        )

    certificate_row_ids = set(certificate.rows)
    if certificate_row_ids != set(expected_rows):
        raise OrchestrationManifestIndexError(
            "certificate rows must exactly match expected rows: "
            f"missing={sorted(set(expected_rows) - certificate_row_ids)!r}, "
            f"extra={sorted(certificate_row_ids - set(expected_rows))!r}"
        )
    core_check_ids = {check_id for check_id, _check in build_core_check_registry().items()}
    refs: list[OrchestrationTrainingManifestRef] = []
    for row_id in expected_rows:
        checks = certificate.rows[row_id].checks
        check_ids = [check.check_id for check in checks]
        if not checks or len(check_ids) != len(set(check_ids)):
            raise OrchestrationManifestIndexError(
                f"certificate row {row_id!r} has empty or duplicate checks"
            )
        missing_checks = sorted(core_check_ids - set(check_ids))
        if missing_checks:
            raise OrchestrationManifestIndexError(
                f"certificate row {row_id!r} is missing core checks {missing_checks!r}"
            )
        if any(check.status != "pass" for check in checks):
            raise OrchestrationManifestIndexError(
                f"certificate row {row_id!r} contains a non-passing check"
            )
        source = run_set_dir / "collected" / row_id / "manifest.json"
        if not source.is_file():
            raise OrchestrationManifestIndexError(f"collected row {row_id!r} has no manifest")
        uri = location.manifest_uris[row_id]
        source_hash = _sha256_file(source)
        source_bytes = source.read_bytes()
        custody_ref = _artifact_ref_from_values(
            uri=uri,
            manifest_id="pending-source-validation",
            sha256=source_hash,
            size_bytes=len(source_bytes),
        )
        try:
            custody_bytes = manifest_provider.get_bytes(custody_ref)
        except Exception as exc:
            raise OrchestrationManifestIndexError(
                f"immutable manifest URI failed validation: {uri!r}: {exc}"
            ) from exc
        if custody_bytes != source_bytes:
            raise OrchestrationManifestIndexError(
                f"custody bytes differ from collected manifest for {run_set_id}/{row_id}"
            )
        manifest = load_manifest(source)
        if not isinstance(manifest, TrainingRunManifest):
            raise OrchestrationManifestIndexError("collected manifest is not TrainingRunManifest")
        planned_run_id = _planned_run_id(row_by_id[row_id], expected_row_id=row_id)
        _validate_orchestration_identity(
            manifest,
            run_set_id=run_set_id,
            row_id=row_id,
            planned_run_id=planned_run_id,
        )
        refs.append(
            OrchestrationTrainingManifestRef(
                id=manifest.id,
                uri=uri,
                metadata=OrchestrationTrainingManifestMetadata(
                    manifest_sha256=source_hash,
                    size_bytes=source.stat().st_size,
                    run_set_id=run_set_id,
                    row_id=row_id,
                    conformance_overall=certificate.overall,
                    certificate_sha256=certificate_sha256,
                    planned_run_id=planned_run_id,
                ),
            )
        )
    return tuple(refs)


def _validate_index_bytes(
    index: OrchestrationManifestIndexSpec,
    *,
    manifest_provider: ImmutableArtifactBlobProvider,
) -> tuple[bytes, ...]:
    resolved: list[bytes] = []
    seen_ids: dict[str, str] = {}
    for ref in index.refs:
        try:
            payload = manifest_provider.get_bytes(_artifact_ref(ref))
        except Exception as exc:
            raise OrchestrationManifestIndexError(
                f"immutable manifest URI failed validation: {ref.uri!r}: {exc}"
            ) from exc
        _validate_manifest_bytes(ref, payload)
        previous = seen_ids.get(ref.id)
        if previous is not None:
            detail = "different bytes" if previous != ref.metadata.manifest_sha256 else "duplicate"
            raise OrchestrationManifestIndexError(
                f"same TrainingRunManifest id has {detail}: {ref.id!r}"
            )
        seen_ids[ref.id] = ref.metadata.manifest_sha256
        resolved.append(payload)
    return tuple(resolved)


def _validate_manifest_path(ref: OrchestrationTrainingManifestRef, path: Path) -> None:
    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
        raise OrchestrationManifestIndexError(
            f"materialized manifest is not an independent regular file: {ref.id!r}"
        )
    payload = path.read_bytes()
    _validate_manifest_bytes(ref, payload)


def _validate_manifest_bytes(ref: OrchestrationTrainingManifestRef, payload: bytes) -> None:
    if len(payload) != ref.metadata.size_bytes:
        raise OrchestrationManifestIndexError(f"manifest size mismatch for {ref.id!r}")
    if hashlib.sha256(payload).hexdigest() != ref.metadata.manifest_sha256:
        raise OrchestrationManifestIndexError(f"manifest hash mismatch for {ref.id!r}")
    try:
        raw = json.loads(payload)
        manifest = TrainingRunManifest.model_validate(raw)
    except (ValueError, TypeError) as exc:
        raise OrchestrationManifestIndexError(
            f"referenced bytes are not TrainingRunManifest: {ref.id}"
        ) from exc
    _validate_orchestration_identity(
        manifest,
        run_set_id=ref.metadata.run_set_id,
        row_id=ref.metadata.row_id,
        planned_run_id=ref.metadata.planned_run_id,
    )
    if manifest.id != ref.id:
        raise OrchestrationManifestIndexError(f"manifest id mismatch for ref {ref.id!r}")


def _artifact_ref(ref: OrchestrationTrainingManifestRef) -> ArtifactRef:
    return _artifact_ref_from_values(
        uri=ref.uri,
        manifest_id=ref.id,
        sha256=ref.metadata.manifest_sha256,
        size_bytes=ref.metadata.size_bytes,
    )


def _artifact_ref_from_values(
    *,
    uri: str,
    manifest_id: str,
    sha256: str,
    size_bytes: int,
) -> ArtifactRef:
    """Construct the canonical Feedbax custody reference for manifest bytes."""

    return ArtifactRef(
        role="training_run_manifest",
        logical_name=manifest_id,
        artifact_id=uri,
        sha256=sha256,
        media_type="application/json",
        size_bytes=size_bytes,
        storage_backend="feedbax-local",
        uri=uri,
    )


def _validate_orchestration_identity(
    manifest: TrainingRunManifest,
    *,
    run_set_id: str,
    row_id: str,
    planned_run_id: str,
) -> None:
    if manifest.status != "completed":
        raise OrchestrationManifestIndexError(
            f"TrainingRunManifest {manifest.id!r} is not completed"
        )
    if manifest.id != planned_run_id:
        raise OrchestrationManifestIndexError("manifest id disagrees with planned_run_id")
    # Legacy M1 manifests predate an inline run_set_id. Their association is
    # established exactly by registration + bundle + certificate at index
    # construction; when the optional manifest field is present it must agree.
    if manifest.run_set_id is not None and manifest.run_set_id != run_set_id:
        raise OrchestrationManifestIndexError("manifest run_set_id disagrees with registration")
    provenance = manifest.metadata.get("training_row_provenance")
    if not isinstance(provenance, Mapping):
        raise OrchestrationManifestIndexError("manifest has no training_row_provenance")
    if provenance.get("row_id") != row_id:
        raise OrchestrationManifestIndexError("manifest row provenance disagrees with bundle")
    if provenance.get("planned_run_id") != planned_run_id:
        raise OrchestrationManifestIndexError(
            "manifest row provenance disagrees with planned_run_id"
        )


def _planned_run_id(row: Mapping[str, object], *, expected_row_id: str) -> str:
    execution = row.get("execution")
    if not isinstance(execution, Mapping):
        raise OrchestrationManifestIndexError("bundle row has no execution envelope")
    provenance = execution.get("row_provenance")
    if not isinstance(provenance, Mapping):
        raise OrchestrationManifestIndexError("bundle row has no row_provenance")
    provenance_row_id = _required_string(provenance, "row_id", "bundle row provenance")
    if provenance_row_id != expected_row_id:
        raise OrchestrationManifestIndexError(
            "bundle execution row_provenance.row_id disagrees with bundle row"
        )
    return _required_string(provenance, "planned_run_id", "bundle row provenance")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise OrchestrationManifestIndexError(f"missing {label}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OrchestrationManifestIndexError(f"{label} must be a JSON object")
    return data


def _required_string(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise OrchestrationManifestIndexError(f"{label} requires non-empty {key!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "INDEX_SCHEMA_ID",
    "INDEX_SCHEMA_VERSION",
    "MaterializedOrchestrationManifestIndex",
    "OrchestrationManifestIndexError",
    "OrchestrationManifestIndexSpec",
    "OrchestrationTrainingManifestMetadata",
    "OrchestrationTrainingManifestRef",
    "RegisteredRunSetLocation",
    "build_orchestration_manifest_index",
    "load_orchestration_manifest_index",
    "materialize_orchestration_manifest_index",
    "select_exact_training_manifest_refs",
    "write_orchestration_manifest_index",
]

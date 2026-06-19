"""Cleanup helpers for disposable raw rollout-array banks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "rlrmp.raw_rollout_cleanup.v1"


class CleanupPreconditionError(RuntimeError):
    """Raised when cleanup would be unsafe because durable artifacts are missing."""


def cleanup_raw_perturbation_rollouts(
    *,
    tracked_manifest_path: Path,
    repo_root: Path,
    bulk_dir: Path | None = None,
    tracked_summary_paths: Sequence[Path] | None = None,
    regeneration_spec_path: Path | None = None,
    manifest_out_path: Path | None = None,
    apply: bool = False,
    overwrite_manifest: bool = False,
) -> dict[str, Any]:
    """Plan or apply deletion of raw perturbation-response rollout arrays.

    The cleanup is deliberately narrow: it deletes only ``.npz`` files under a
    single perturbation-response bulk directory, and only after the tracked
    response manifest, tracked summary note, and regeneration spec all exist.

    Args:
        tracked_manifest_path: Tracked perturbation-response manifest under
            ``results/<issue>/notes/``.
        repo_root: Repository root used to resolve relative paths.
        bulk_dir: Bulk perturbation-response directory. If omitted, infer it
            from the manifest's ``bulk_detail_manifest`` or ``bulk_files`` refs.
        tracked_summary_paths: Required tracked summary/note files. If omitted,
            infer the conventional ``<manifest-stem-without-_manifest>.md``.
        regeneration_spec_path: Required regeneration spec. If omitted, infer it
            from the manifest's ``regeneration_spec`` field, falling back to the
            conventional ``<manifest-stem>_regeneration_spec.json`` path.
        manifest_out_path: Optional cleanup manifest destination. Apply mode
            defaults to ``results/<issue>/notes/raw_rollout_cleanup_<bulk>.json``.
        apply: Delete files when true. False is a dry-run plan.
        overwrite_manifest: Permit replacing an existing cleanup manifest.

    Returns:
        JSON-serializable cleanup manifest.

    Raises:
        CleanupPreconditionError: If a required durable artifact is missing, a
            target path is outside the requested bulk directory, or the cleanup
            manifest would be overwritten without permission.
    """

    repo_root = repo_root.resolve()
    manifest_path = _resolve_repo_path(tracked_manifest_path, repo_root=repo_root)
    manifest = _read_json(manifest_path)
    resolved_bulk_dir = _resolve_bulk_dir(manifest, bulk_dir=bulk_dir, repo_root=repo_root)
    summaries = _resolve_summary_paths(
        manifest_path,
        tracked_summary_paths=tracked_summary_paths,
        repo_root=repo_root,
    )
    regeneration_path = _resolve_regeneration_spec_path(
        manifest,
        manifest_path=manifest_path,
        regeneration_spec_path=regeneration_spec_path,
        repo_root=repo_root,
    )

    required_artifacts = [
        _artifact_record("tracked_manifest", manifest_path, repo_root=repo_root),
        *[
            _artifact_record("tracked_summary", summary_path, repo_root=repo_root)
            for summary_path in summaries
        ],
        _artifact_record("regeneration_spec", regeneration_path, repo_root=repo_root),
        _artifact_record("bulk_dir", resolved_bulk_dir, repo_root=repo_root),
    ]
    missing = [item for item in required_artifacts if not item["exists"]]
    if missing:
        missing_paths = ", ".join(str(item["path"]) for item in missing)
        raise CleanupPreconditionError(f"required cleanup preconditions are missing: {missing_paths}")

    manifest_npz_refs = list(_manifest_npz_refs(manifest, repo_root=repo_root))
    missing_manifest_references = [
        _repo_relative(path, repo_root=repo_root)
        for path in manifest_npz_refs
        if not path.exists()
    ]
    candidates = _candidate_npz_files(
        manifest,
        bulk_dir=resolved_bulk_dir,
        repo_root=repo_root,
    )
    would_delete = [
        _file_record(path, repo_root=repo_root)
        for path in candidates
        if path.exists()
    ]
    deleted: list[dict[str, Any]] = []
    deleted_bytes = 0

    if apply:
        for path in candidates:
            if not path.exists():
                continue
            record = _file_record(path, repo_root=repo_root)
            path.unlink()
            deleted.append(record)
            deleted_bytes += int(record["size_bytes"])

    manifest_out = _resolve_manifest_out_path(
        manifest_path=manifest_path,
        bulk_dir=resolved_bulk_dir,
        manifest_out_path=manifest_out_path,
        repo_root=repo_root,
        apply=apply,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cleanup_type": "perturbation_response_raw_rollout_arrays",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mode": "apply" if apply else "dry_run",
        "tracked_manifest": _repo_relative(manifest_path, repo_root=repo_root),
        "bulk_dir": _repo_relative(resolved_bulk_dir, repo_root=repo_root),
        "kept_preconditions": required_artifacts,
        "candidate_file_count": len(would_delete),
        "candidate_bytes": sum(int(item["size_bytes"]) for item in would_delete),
        "would_delete_files": would_delete,
        "deleted_file_count": len(deleted),
        "deleted_bytes": deleted_bytes,
        "deleted_files": deleted,
        "missing_manifest_references": missing_manifest_references,
    }
    if manifest_out is not None:
        payload["cleanup_manifest_path"] = _repo_relative(manifest_out, repo_root=repo_root)
        if manifest_out.exists() and not overwrite_manifest:
            raise CleanupPreconditionError(
                f"cleanup manifest already exists: {manifest_out}; pass overwrite to replace it"
            )
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def default_cleanup_manifest_path(
    *,
    tracked_manifest_path: Path,
    bulk_dir: Path,
    repo_root: Path,
) -> Path:
    """Return the conventional tracked deletion-manifest path for a cleanup."""

    repo_root = repo_root.resolve()
    manifest_path = _resolve_repo_path(tracked_manifest_path, repo_root=repo_root)
    notes_dir = manifest_path.parent
    return notes_dir / f"raw_rollout_cleanup_{bulk_dir.name}.json"


def _resolve_repo_path(path: Path | str, *, repo_root: Path) -> Path:
    raw = str(path)
    if raw.startswith("repo://"):
        raw = raw.removeprefix("repo://")
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CleanupPreconditionError(f"tracked manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CleanupPreconditionError(f"tracked manifest must be a JSON object: {path}")
    return payload


def _resolve_bulk_dir(
    manifest: Mapping[str, Any],
    *,
    bulk_dir: Path | None,
    repo_root: Path,
) -> Path:
    if bulk_dir is not None:
        return _resolve_repo_path(bulk_dir, repo_root=repo_root)
    detail = manifest.get("bulk_detail_manifest")
    if isinstance(detail, Mapping) and detail.get("path"):
        return _resolve_repo_path(str(detail["path"]), repo_root=repo_root).parent
    refs = list(_manifest_npz_refs(manifest, repo_root=repo_root))
    if refs:
        return Path(_common_path([path.parent for path in refs])).resolve()
    raise CleanupPreconditionError("bulk_dir is required when manifest has no bulk references")


def _resolve_summary_paths(
    manifest_path: Path,
    *,
    tracked_summary_paths: Sequence[Path] | None,
    repo_root: Path,
) -> list[Path]:
    if tracked_summary_paths:
        return [
            _resolve_repo_path(path, repo_root=repo_root)
            for path in tracked_summary_paths
        ]
    name = manifest_path.name
    if name.endswith("_manifest.json"):
        inferred = manifest_path.with_name(f"{name.removesuffix('_manifest.json')}.md")
    else:
        inferred = manifest_path.with_suffix(".md")
    return [inferred]


def _resolve_regeneration_spec_path(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    regeneration_spec_path: Path | None,
    repo_root: Path,
) -> Path:
    if regeneration_spec_path is not None:
        return _resolve_repo_path(regeneration_spec_path, repo_root=repo_root)
    raw = manifest.get("regeneration_spec")
    if isinstance(raw, str) and raw:
        return _resolve_repo_path(raw, repo_root=repo_root)
    return manifest_path.with_name(f"{manifest_path.stem}_regeneration_spec.json")


def _resolve_manifest_out_path(
    *,
    manifest_path: Path,
    bulk_dir: Path,
    manifest_out_path: Path | None,
    repo_root: Path,
    apply: bool,
) -> Path | None:
    if manifest_out_path is not None:
        return _resolve_repo_path(manifest_out_path, repo_root=repo_root)
    if apply:
        return default_cleanup_manifest_path(
            tracked_manifest_path=manifest_path,
            bulk_dir=bulk_dir,
            repo_root=repo_root,
        )
    return None


def _candidate_npz_files(
    manifest: Mapping[str, Any],
    *,
    bulk_dir: Path,
    repo_root: Path,
) -> list[Path]:
    bulk_dir = bulk_dir.resolve()
    paths = {path.resolve() for path in bulk_dir.glob("**/*.npz")}
    paths.update(path.resolve() for path in _manifest_npz_refs(manifest, repo_root=repo_root))
    candidates: list[Path] = []
    for path in sorted(paths):
        if not path.exists():
            continue
        if path.suffix != ".npz":
            continue
        if not _is_relative_to(path.resolve(), bulk_dir):
            raise CleanupPreconditionError(
                f"manifest references .npz outside cleanup bulk dir: {path}"
            )
        candidates.append(path)
    return candidates


def _manifest_npz_refs(payload: Any, *, repo_root: Path) -> Iterable[Path]:
    for value in _walk_manifest_strings(payload):
        if value.startswith("repo://"):
            value = value.removeprefix("repo://")
        if value.endswith(".npz"):
            yield _resolve_repo_path(value, repo_root=repo_root)


def _walk_manifest_strings(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, Mapping):
        for value in payload.values():
            yield from _walk_manifest_strings(value)
    elif isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        for value in payload:
            yield from _walk_manifest_strings(value)


def _artifact_record(role: str, path: Path, *, repo_root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": _repo_relative(path, repo_root=repo_root),
        "exists": path.exists(),
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
    }
    if path.is_file():
        record.update(_file_hash_record(path))
    elif path.is_dir():
        files = [child for child in path.rglob("*") if child.is_file()]
        record["file_count"] = len(files)
        record["total_bytes"] = sum(child.stat().st_size for child in files)
    return record


def _file_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": _repo_relative(path, repo_root=repo_root),
        **_file_hash_record(path),
    }


def _file_hash_record(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _common_path(paths: Sequence[Path]) -> str:
    parts = [str(path) for path in paths]
    if not parts:
        raise CleanupPreconditionError("cannot infer common path from no files")
    return os.path.commonpath(parts)

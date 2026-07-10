"""Feedbax regeneration provenance helpers for rlrmp analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.contracts.manifest import (
    ArtifactRef,
    EntrypointRef,
    FileHashRef,
    Provenance,
    REGENERATION_SPEC_SCHEMA_VERSION,
    RegenerationCommand,
    RegenerationSpec,
    TreeHashEntry,
    TreeHashRef,
)

from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.paths import REPO_ROOT, mkdir_p


SCHEMA_VERSION = REGENERATION_SPEC_SCHEMA_VERSION
DEFAULT_MAX_TREE_HASH_FILES = int(
    load_analysis_parameter_preset("diagnostic_provenance").parameters["max_tree_hash_files"]
)


def write_regeneration_spec(
    *,
    spec_path: Path,
    diagnostic_name: str,
    materializer: str,
    command: Sequence[str] | str | None,
    parameters: Mapping[str, Any] | None = None,
    inputs: Sequence[Mapping[str, Any] | Path | str] = (),
    outputs: Sequence[Mapping[str, Any] | Path | str] = (),
    source_files: Sequence[Path | str] = (),
    notes: Sequence[str] = (),
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Write a Feedbax regeneration spec for an RLRMP diagnostic artifact.

    Args:
        spec_path: Destination JSON path.
        diagnostic_name: Stable human-readable diagnostic identifier.
        materializer: Python function or script that produced the artifact.
        command: Re-run command, when known. A string is stored as a shell
            command; a sequence is stored as argv.
        parameters: JSON-compatible materializer parameters.
        inputs: Input path refs or already-built ref dictionaries.
        outputs: Output path refs or already-built ref dictionaries.
        source_files: Source files whose hashes should be recorded.
        notes: Short scope/limitation notes.
        repo_root: Repository root for relative paths and git metadata.

    Returns:
        The JSON-compatible spec payload.
    """

    repo_root = repo_root.resolve()
    git = git_provenance(repo_root)
    source_file_refs, source_tree_refs = _source_refs(source_files, repo_root=repo_root)
    spec = RegenerationSpec(
        command=_regeneration_command(command, materializer=materializer, repo_root=repo_root),
        parameters=_jsonify(parameters or {}),
        inputs=[_coerce_artifact_ref(item, role="input", repo_root=repo_root) for item in inputs],
        outputs=[
            _coerce_artifact_ref(item, role="output", repo_root=repo_root) for item in outputs
        ],
        source_files=source_file_refs,
        source_trees=source_tree_refs,
        provenance=Provenance(
            source_repo=str(repo_root),
            source_branch=git.get("branch"),
            source_commit=git.get("head"),
            dirty=bool(git.get("dirty")),
            entrypoint=EntrypointRef(kind="rlrmp-diagnostic-materializer", name=materializer),
            metadata={"status_porcelain": git.get("status_porcelain", [])},
        ),
        metadata={
            "diagnostic_name": diagnostic_name,
            "materializer": materializer,
            "notes": list(notes),
            "schema_boundary": "rlrmp diagnostic payload; Feedbax regeneration custody",
        },
    )
    payload = spec.model_dump(mode="json", exclude_none=True)
    mkdir_p(spec_path.parent)
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def path_ref(
    path: Path | str,
    *,
    role: str,
    repo_root: Path = REPO_ROOT,
    hash_files: bool = True,
    max_tree_hash_files: int = DEFAULT_MAX_TREE_HASH_FILES,
) -> dict[str, Any]:
    """Return a compact, JSON-compatible reference for a file or directory."""

    repo_root = repo_root.resolve()
    resolved = _resolve_path(path, repo_root=repo_root)
    ref: dict[str, Any] = {
        "role": role,
        "path": repo_relative(resolved, repo_root=repo_root),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        return ref
    if resolved.is_file():
        stat = resolved.stat()
        ref.update(
            {
                "kind": "file",
                "size_bytes": stat.st_size,
                "sha256": sha256_file(resolved) if hash_files else None,
            }
        )
        return ref
    if resolved.is_dir():
        summary = tree_summary(
            resolved,
            repo_root=repo_root,
            hash_files=hash_files,
            max_hash_files=max_tree_hash_files,
        )
        ref.update({"kind": "directory", **summary})
        return ref
    ref["kind"] = "other"
    return ref


def tree_summary(
    path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
    hash_files: bool = True,
    max_hash_files: int = DEFAULT_MAX_TREE_HASH_FILES,
) -> dict[str, Any]:
    """Return a deterministic directory summary without requiring full custody."""

    root = _resolve_path(path, repo_root=repo_root)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    total_bytes = sum(item.stat().st_size for item in files)
    sample = files[:max_hash_files]
    payload: dict[str, Any] = {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "hash_file_count": len(sample) if hash_files else 0,
        "hash_truncated": hash_files and len(files) > len(sample),
    }
    if hash_files:
        digest = hashlib.sha256()
        for item in sample:
            rel = item.relative_to(root).as_posix()
            item_hash = sha256_file(item)
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(item_hash.encode("ascii"))
            digest.update(b"\0")
        payload["sample_tree_sha256"] = digest.hexdigest()
    return payload


def sha256_file(path: Path | str) -> str:
    """Return a SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return git state relevant to diagnostic reproducibility."""

    repo_root = repo_root.resolve()
    branch = _git(["branch", "--show-current"], repo_root)
    head = _git(["rev-parse", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    return {
        "head": head or None,
        "branch": branch or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def repo_relative(path: Path | str, *, repo_root: Path = REPO_ROOT) -> str:
    """Return a repo-relative path when possible."""

    resolved = _resolve_path(path, repo_root=repo_root)
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _coerce_ref(
    item: Mapping[str, Any] | Path | str,
    *,
    role: str,
    repo_root: Path,
) -> dict[str, Any]:
    if isinstance(item, Mapping):
        payload = dict(item)
        if "path" in payload and "exists" not in payload:
            ref = path_ref(
                payload["path"], role=str(payload.get("role", role)), repo_root=repo_root
            )
            ref.update({key: _jsonify(value) for key, value in payload.items() if key != "path"})
            return ref
        return _jsonify(payload)
    return path_ref(item, role=role, repo_root=repo_root)


def _coerce_artifact_ref(
    item: Mapping[str, Any] | Path | str,
    *,
    role: str,
    repo_root: Path,
) -> ArtifactRef:
    ref = _coerce_ref(item, role=role, repo_root=repo_root)
    ref_role = str(ref.get("role", role))
    logical_name = str(ref.get("path") or ref.get("logical_name") or ref_role)
    metadata = {
        key: _jsonify(value)
        for key, value in ref.items()
        if key not in {"role", "path", "logical_name", "uri"}
    }
    return ArtifactRef(
        role=ref_role,
        logical_name=logical_name,
        uri=str(ref.get("uri") or f"repo://{logical_name}"),
        media_type=_media_type_for_path(logical_name),
        size_bytes=_optional_int(ref.get("size_bytes")),
        sha256=str(ref["sha256"]) if ref.get("sha256") is not None else None,
        metadata=metadata,
    )


def _source_refs(
    source_files: Sequence[Path | str],
    *,
    repo_root: Path,
) -> tuple[list[FileHashRef], list[TreeHashRef]]:
    file_refs: list[FileHashRef] = []
    tree_refs: list[TreeHashRef] = []
    for source_file in source_files:
        resolved = _resolve_path(source_file, repo_root=repo_root)
        if resolved.is_dir():
            tree_refs.append(_tree_hash_ref(resolved, repo_root=repo_root))
        elif resolved.exists():
            stat = resolved.stat()
            file_refs.append(
                FileHashRef(
                    path=repo_relative(resolved, repo_root=repo_root),
                    sha256=sha256_file(resolved),
                    size_bytes=stat.st_size,
                    role="source_file",
                )
            )
        else:
            file_refs.append(
                FileHashRef(
                    path=repo_relative(resolved, repo_root=repo_root),
                    sha256="missing",
                    size_bytes=0,
                    role="source_file",
                    metadata={"exists": False},
                )
            )
    return file_refs, tree_refs


def _tree_hash_ref(path: Path, *, repo_root: Path) -> TreeHashRef:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    entries: list[TreeHashEntry] = []
    for item in files:
        rel = item.relative_to(path).as_posix()
        stat = item.stat()
        item_hash = sha256_file(item)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\0")
        entries.append(
            TreeHashEntry(
                path=rel,
                sha256=item_hash,
                size_bytes=stat.st_size,
            )
        )
    return TreeHashRef(
        path=repo_relative(path, repo_root=repo_root),
        sha256=digest.hexdigest(),
        file_count=len(files),
        total_size_bytes=sum(entry.size_bytes for entry in entries),
        files=entries[:DEFAULT_MAX_TREE_HASH_FILES],
        role="source_tree",
        metadata={"hash_truncated": len(entries) > DEFAULT_MAX_TREE_HASH_FILES},
    )


def _resolve_path(path: Path | str, *, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.absolute()
    return (repo_root / candidate).absolute()


def _regeneration_command(
    command: Sequence[str] | str | None,
    *,
    materializer: str,
    repo_root: Path,
) -> RegenerationCommand:
    if command is None:
        return RegenerationCommand(
            argv=["unknown"],
            cwd=repo_relative(repo_root, repo_root=repo_root),
            metadata={
                "status": "not_recorded",
                "materializer": materializer,
            },
        )
    if isinstance(command, str):
        return RegenerationCommand(
            shell_command=command,
            cwd=repo_relative(repo_root, repo_root=repo_root),
            metadata={"materializer": materializer},
        )
    return RegenerationCommand(
        argv=[str(item) for item in command],
        cwd=repo_relative(repo_root, repo_root=repo_root),
        metadata={"materializer": materializer},
    )


def _media_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".json": "application/json",
        ".md": "text/markdown",
        ".html": "text/html",
        ".npz": "application/x-npz",
    }.get(suffix, "application/octet-stream")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _git(args: Sequence[str], repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


__all__ = [
    "SCHEMA_VERSION",
    "git_provenance",
    "path_ref",
    "repo_relative",
    "sha256_file",
    "tree_summary",
    "write_regeneration_spec",
]

"""Regeneration provenance helpers for rlrmp analysis artifacts.

This is intentionally lightweight and rlrmp-local. Feedbax GraphSpec/provider
manifests should eventually own this contract for graph-native analyses; until
then these specs keep current GRU diagnostics reproducible without issue-log
archaeology.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.spec_migrations import (
    DIAGNOSTIC_REGENERATION_SPEC_KIND,
    DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION,
    stamp_current_schema,
)


SCHEMA_VERSION = DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION
DEFAULT_MAX_TREE_HASH_FILES = 256


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
    """Write a JSON regeneration spec for a diagnostic artifact.

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
    payload = stamp_current_schema(
        DIAGNOSTIC_REGENERATION_SPEC_KIND,
        {
            "diagnostic_name": diagnostic_name,
            "materializer": materializer,
            "command": _command_payload(command),
            "parameters": _jsonify(parameters or {}),
            "inputs": [_coerce_ref(item, role="input", repo_root=repo_root) for item in inputs],
            "outputs": [
                _coerce_ref(item, role="output", repo_root=repo_root) for item in outputs
            ],
            "source_files": [
                path_ref(path, role="source_file", repo_root=repo_root) for path in source_files
            ],
            "git": git_provenance(repo_root),
            "notes": list(notes),
            "future_graphspec": {
                "status": "temporary_rlrmp_bridge",
                "expected_successor": (
                    "Feedbax-native analysis/regeneration manifests after GraphSpec "
                    "and provider-manifest migration"
                ),
            },
        },
    )
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
            ref = path_ref(payload["path"], role=str(payload.get("role", role)), repo_root=repo_root)
            ref.update({key: _jsonify(value) for key, value in payload.items() if key != "path"})
            return ref
        return _jsonify(payload)
    return path_ref(item, role=role, repo_root=repo_root)


def _resolve_path(path: Path | str, *, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.absolute()
    return (repo_root / candidate).absolute()


def _command_payload(command: Sequence[str] | str | None) -> dict[str, Any]:
    if command is None:
        return {"status": "not_recorded"}
    if isinstance(command, str):
        return {"type": "shell", "value": command}
    return {"type": "argv", "value": list(command)}


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

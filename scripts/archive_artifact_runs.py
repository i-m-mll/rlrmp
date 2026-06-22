from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TOOL_VERSION = "archive_artifact_runs.v1"
PROJECT = "rlrmp"
INDEX_PATH = Path("_artifacts") / ".archive_index.jsonl"
VOLUME_MARKER = "VOLUME_ID"


class ArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunRef:
    experiment: str
    run: str

    @property
    def canonical_path(self) -> Path:
        return Path("_artifacts") / self.experiment / "runs" / self.run


@dataclass(frozen=True)
class Inventory:
    bytes: int
    files: int
    manifest: dict[str, dict[str, str | int]]


@dataclass(frozen=True)
class ArchivePlan:
    run_ref: RunRef
    source_path: Path
    target_path: Path
    root_relative_path: Path
    local_symlink_path: Path
    volume_id: str
    inventory: Inventory
    reason: str

    def index_entry(self, *, status: str, timestamp_field: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "project": PROJECT,
            "kind": "cold_run_artifact_archive",
            "run_ref": {
                "experiment": self.run_ref.experiment,
                "run": self.run_ref.run,
                "canonical_path": self.run_ref.canonical_path.as_posix(),
            },
            "archive": {
                "volume_id": self.volume_id,
                "root_relative_path": self.root_relative_path.as_posix(),
                "local_symlink_path": self.local_symlink_path.as_posix(),
            },
            "inventory": {
                "bytes": self.inventory.bytes,
                "files": self.inventory.files,
            },
            "status": status,
            timestamp_field: utc_now(),
            "reason": self.reason,
            "tool_version": TOOL_VERSION,
        }

    def report(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "target_path": str(self.target_path),
            "bytes": self.inventory.bytes,
            "files": self.inventory.files,
            "index_entry": self.index_entry(status="planned", timestamp_field="planned_at"),
        }


def parse_run_ref(value: str) -> RunRef:
    parts = value.split("/")
    if len(parts) != 2:
        raise ArchiveError("run ref must be exactly <issue_hash>/<run_id>")
    experiment, run = parts
    if not is_plain_segment(experiment) or len(experiment) < 7:
        raise ArchiveError("issue hash must be a plain hash-like path segment")
    if not all(char in "0123456789abcdef" for char in experiment.lower()):
        raise ArchiveError("issue hash must be hexadecimal")
    if not is_plain_segment(run):
        raise ArchiveError("run id must be a single path segment")
    return RunRef(experiment=experiment, run=run)


def build_archive_plan(
    *,
    repo_root: Path,
    archive_root: Path,
    run_ref: RunRef,
    volume_id: str,
    reason: str,
    allow_index_update: bool = False,
) -> ArchivePlan:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    actual_volume_id = read_volume_id(archive_root)
    if actual_volume_id != volume_id:
        raise ArchiveError(
            f"archive root volume mismatch: expected {volume_id!r}, found {actual_volume_id!r}"
        )

    source_path = repo_root / run_ref.canonical_path
    if source_path.is_symlink():
        raise ArchiveError(f"source is already a symlink: {source_path}")
    if not source_path.exists():
        raise ArchiveError(f"source run directory does not exist: {source_path}")
    if not source_path.is_dir():
        raise ArchiveError(f"source run path is not a directory: {source_path}")

    root_relative_path = run_ref.canonical_path
    target_path = archive_root / root_relative_path
    inventory = inventory_directory(source_path)
    plan = ArchivePlan(
        run_ref=run_ref,
        source_path=source_path,
        target_path=target_path,
        root_relative_path=root_relative_path,
        local_symlink_path=run_ref.canonical_path,
        volume_id=volume_id,
        inventory=inventory,
        reason=reason,
    )
    require_no_conflicting_index(repo_root, plan, allow_index_update=allow_index_update)
    return plan


def apply_archive(
    *,
    repo_root: Path,
    archive_root: Path,
    run_ref: RunRef,
    volume_id: str,
    reason: str,
    allow_existing_target: bool = False,
    allow_index_update: bool = False,
) -> dict[str, Any]:
    plan = build_archive_plan(
        repo_root=repo_root,
        archive_root=archive_root,
        run_ref=run_ref,
        volume_id=volume_id,
        reason=reason,
        allow_index_update=allow_index_update,
    )
    if plan.target_path.exists() or plan.target_path.is_symlink():
        if not allow_existing_target:
            raise ArchiveError(f"archive target already exists: {plan.target_path}")
        target_inventory = inventory_directory(plan.target_path)
        if target_inventory.manifest != plan.inventory.manifest:
            raise ArchiveError(f"archive target inventory conflicts: {plan.target_path}")
    else:
        copy_verified_directory(plan.source_path, plan.target_path, plan.inventory)

    replace_directory_with_symlink(plan.source_path, plan.target_path)
    entry = plan.index_entry(status="archived", timestamp_field="archived_at")
    append_index_record(repo_root, entry)
    return {
        "status": "archived",
        "source_path": str(plan.source_path),
        "target_path": str(plan.target_path),
        "index_entry": entry,
    }


def verify_archive(
    *,
    repo_root: Path,
    archive_root: Path,
    run_ref: RunRef,
    volume_id: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    actual_volume_id = read_volume_id(archive_root)
    if actual_volume_id != volume_id:
        raise ArchiveError(
            f"archive root volume mismatch: expected {volume_id!r}, found {actual_volume_id!r}"
        )

    record = latest_index_record(repo_root, run_ref)
    if record is None:
        raise ArchiveError(f"no archive index record for {run_ref.experiment}/{run_ref.run}")
    archive = record["archive"]
    if archive["volume_id"] != volume_id:
        raise ArchiveError("latest index record points at a different archive volume")
    target_path = archive_root / archive["root_relative_path"]
    local_path = repo_root / record["run_ref"]["canonical_path"]
    target_inventory = inventory_directory(target_path)
    expected = record["inventory"]
    if target_inventory.bytes != expected["bytes"] or target_inventory.files != expected["files"]:
        raise ArchiveError("archive target inventory does not match the index")

    status = record["status"]
    if status == "archived":
        if not local_path.is_symlink():
            raise ArchiveError(f"local canonical path is not an archive symlink: {local_path}")
        if local_path.resolve() != target_path.resolve():
            raise ArchiveError("local archive symlink points at the wrong target")
    elif status == "restored":
        if local_path.is_symlink() or not local_path.is_dir():
            raise ArchiveError(f"restored local canonical path is not a directory: {local_path}")
        local_inventory = inventory_directory(local_path)
        if local_inventory.manifest != target_inventory.manifest:
            raise ArchiveError("restored local inventory does not match archive target")
    else:
        raise ArchiveError(f"latest index status is not verifiable: {status}")

    return {
        "status": "verified",
        "archive_status": status,
        "target_path": str(target_path),
        "bytes": target_inventory.bytes,
        "files": target_inventory.files,
    }


def restore_archive(
    *,
    repo_root: Path,
    archive_root: Path,
    run_ref: RunRef,
    volume_id: str,
    reason: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    actual_volume_id = read_volume_id(archive_root)
    if actual_volume_id != volume_id:
        raise ArchiveError(
            f"archive root volume mismatch: expected {volume_id!r}, found {actual_volume_id!r}"
        )
    record = latest_index_record(repo_root, run_ref)
    if record is None or record["status"] != "archived":
        raise ArchiveError("restore requires the latest index record to be archived")
    target_path = archive_root / record["archive"]["root_relative_path"]
    local_path = repo_root / record["run_ref"]["canonical_path"]
    if not local_path.is_symlink():
        raise ArchiveError(f"local canonical path is not an archive symlink: {local_path}")
    if local_path.resolve() != target_path.resolve():
        raise ArchiveError("local archive symlink points at the wrong target")

    target_inventory = inventory_directory(target_path)
    expected = record["inventory"]
    if target_inventory.bytes != expected["bytes"] or target_inventory.files != expected["files"]:
        raise ArchiveError("archive target inventory does not match the index")
    materialize_symlink(local_path, target_path, target_inventory)

    restored = dict(record)
    restored["status"] = "restored"
    restored["restored_at"] = utc_now()
    restored["reason"] = reason
    restored["tool_version"] = TOOL_VERSION
    restored.pop("archived_at", None)
    append_index_record(repo_root, restored)
    return {
        "status": "restored",
        "source_path": str(local_path),
        "target_path": str(target_path),
        "index_entry": restored,
    }


def read_volume_id(archive_root: Path) -> str:
    marker = archive_root / VOLUME_MARKER
    if not marker.is_file():
        raise ArchiveError(f"archive root is missing required {VOLUME_MARKER} marker: {marker}")
    volume_id = marker.read_text(encoding="utf-8").strip()
    if not volume_id:
        raise ArchiveError(f"archive root marker is empty: {marker}")
    return volume_id


def inventory_directory(path: Path) -> Inventory:
    if path.is_symlink():
        raise ArchiveError(f"refusing to inventory symlink path: {path}")
    if not path.is_dir():
        raise ArchiveError(f"cannot inventory missing/non-directory path: {path}")
    total_bytes = 0
    files = 0
    manifest: dict[str, dict[str, str | int]] = {}
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ArchiveError(f"refusing symlink inside run directory: {child}")
        if not child.is_file():
            continue
        relpath = child.relative_to(path).as_posix()
        digest = hashlib.sha256()
        size = 0
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        total_bytes += size
        files += 1
        manifest[relpath] = {"bytes": size, "sha256": digest.hexdigest()}
    return Inventory(bytes=total_bytes, files=files, manifest=manifest)


def copy_verified_directory(source: Path, target: Path, source_inventory: Inventory) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        shutil.copytree(source, temp_dir / target.name, symlinks=False)
        copied = temp_dir / target.name
        copied_inventory = inventory_directory(copied)
        if copied_inventory.manifest != source_inventory.manifest:
            raise ArchiveError("copied archive inventory does not match source")
        copied.rename(target)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def replace_directory_with_symlink(source: Path, target: Path) -> None:
    backup = source.parent / f".{source.name}.archive-backup"
    if backup.exists() or backup.is_symlink():
        raise ArchiveError(f"backup path already exists: {backup}")
    source.rename(backup)
    try:
        symlink_target = os.path.relpath(target, start=source.parent)
        source.symlink_to(symlink_target, target_is_directory=True)
        if not source.is_symlink() or source.resolve() != target.resolve():
            raise ArchiveError("created symlink failed verification")
    except Exception:
        if source.exists() or source.is_symlink():
            source.unlink()
        backup.rename(source)
        raise
    shutil.rmtree(backup)


def materialize_symlink(local_path: Path, target_path: Path, target_inventory: Inventory) -> None:
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{local_path.name}.restore-", dir=str(local_path.parent))
    )
    restored = temp_dir / local_path.name
    try:
        shutil.copytree(target_path, restored, symlinks=False)
        restored_inventory = inventory_directory(restored)
        if restored_inventory.manifest != target_inventory.manifest:
            raise ArchiveError("restored inventory does not match archive target")
        local_path.unlink()
        restored.rename(local_path)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def read_index_records(repo_root: Path) -> list[dict[str, Any]]:
    index_path = repo_root / INDEX_PATH
    if not index_path.exists():
        return []
    records = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ArchiveError(f"invalid archive index JSON on line {line_number}") from exc
    return records


def latest_index_record(repo_root: Path, run_ref: RunRef) -> dict[str, Any] | None:
    canonical = run_ref.canonical_path.as_posix()
    matches = [
        record
        for record in read_index_records(repo_root)
        if record.get("run_ref", {}).get("canonical_path") == canonical
    ]
    return matches[-1] if matches else None


def require_no_conflicting_index(
    repo_root: Path,
    plan: ArchivePlan,
    *,
    allow_index_update: bool,
) -> None:
    record = latest_index_record(repo_root, plan.run_ref)
    if record is None:
        return
    archive = record.get("archive", {})
    inventory = record.get("inventory", {})
    same_archive = (
        archive.get("volume_id") == plan.volume_id
        and archive.get("root_relative_path") == plan.root_relative_path.as_posix()
    )
    same_inventory = (
        inventory.get("bytes") == plan.inventory.bytes
        and inventory.get("files") == plan.inventory.files
    )
    if same_archive and same_inventory and allow_index_update:
        return
    raise ArchiveError(
        "archive index already has state for this run; pass --allow-index-update "
        "after checking that the existing state is intentionally superseded"
    )


def append_index_record(repo_root: Path, record: dict[str, Any]) -> None:
    index_path = repo_root / INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def is_plain_segment(value: str) -> bool:
    if not value or value in {".", ".."}:
        return False
    return "/" not in value and "\\" not in value and "\x00" not in value


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["plan", "apply", "verify", "restore"],
        help="archive operation to perform",
    )
    parser.add_argument("run_ref", help="run reference as <issue_hash>/<run_id>")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--allow-existing-target", action="store_true")
    parser.add_argument("--allow-index-update", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_ref = parse_run_ref(args.run_ref)
        if args.command == "plan":
            payload = build_archive_plan(
                repo_root=args.repo_root,
                archive_root=args.archive_root,
                run_ref=run_ref,
                volume_id=args.volume_id,
                reason=args.reason,
                allow_index_update=args.allow_index_update,
            ).report()
        elif args.command == "apply":
            payload = apply_archive(
                repo_root=args.repo_root,
                archive_root=args.archive_root,
                run_ref=run_ref,
                volume_id=args.volume_id,
                reason=args.reason,
                allow_existing_target=args.allow_existing_target,
                allow_index_update=args.allow_index_update,
            )
        elif args.command == "verify":
            payload = verify_archive(
                repo_root=args.repo_root,
                archive_root=args.archive_root,
                run_ref=run_ref,
                volume_id=args.volume_id,
            )
        else:
            payload = restore_archive(
                repo_root=args.repo_root,
                archive_root=args.archive_root,
                run_ref=run_ref,
                volume_id=args.volume_id,
                reason=args.reason,
            )
    except ArchiveError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

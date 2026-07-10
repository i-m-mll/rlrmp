from __future__ import annotations
from rlrmp.io import load_named_python_module

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "archive_artifact_runs.py"


def load_archive_module() -> ModuleType:
    return load_named_python_module('archive_artifact_runs', SCRIPT)


archive = load_archive_module()


def test_plan_reports_paths_inventory_and_planned_index_entry(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path)
    write_run(repo, "2ec5fa6", "cold__seed_0")

    plan = archive.build_archive_plan(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
        volume_id="fixture-volume",
        reason="cold run",
    )
    report = plan.report()

    assert report["source_path"] == str(repo / "_artifacts/2ec5fa6/runs/cold__seed_0")
    assert report["target_path"] == str(archive_root / "_artifacts/2ec5fa6/runs/cold__seed_0")
    assert report["bytes"] == len("alpha\n") + len("beta\n")
    assert report["files"] == 2
    entry = report["index_entry"]
    assert entry["status"] == "planned"
    assert entry["project"] == "rlrmp"
    assert entry["kind"] == "cold_run_artifact_archive"
    assert entry["run_ref"] == {
        "experiment": "2ec5fa6",
        "run": "cold__seed_0",
        "canonical_path": "_artifacts/2ec5fa6/runs/cold__seed_0",
    }
    assert entry["archive"]["volume_id"] == "fixture-volume"
    assert entry["archive"]["root_relative_path"] == "_artifacts/2ec5fa6/runs/cold__seed_0"
    assert entry["archive"]["local_symlink_path"] == "_artifacts/2ec5fa6/runs/cold__seed_0"
    assert entry["inventory"] == {"bytes": report["bytes"], "files": 2}
    assert "planned_at" in entry
    assert not (repo / "_artifacts/.archive_index.jsonl").exists()
    assert not (archive_root / "_artifacts").exists()


def test_wrong_volume_refuses_before_planning(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path, volume_id="actual-volume")
    write_run(repo, "2ec5fa6", "cold__seed_0")

    with pytest.raises(archive.ArchiveError, match="volume mismatch"):
        archive.build_archive_plan(
            repo_root=repo,
            archive_root=archive_root,
            run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
            volume_id="other-volume",
            reason="cold run",
        )


def test_apply_copies_then_replaces_local_run_with_symlink(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path)
    source = write_run(repo, "2ec5fa6", "cold__seed_0")

    result = archive.apply_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
        volume_id="fixture-volume",
        reason="cold run",
    )

    target = archive_root / "_artifacts/2ec5fa6/runs/cold__seed_0"
    assert result["status"] == "archived"
    assert source.is_symlink()
    assert source.resolve() == target.resolve()
    assert (target / "metrics.json").read_text(encoding="utf-8") == "alpha\n"
    assert (target / "nested/checkpoint.eqx").read_text(encoding="utf-8") == "beta\n"
    records = read_index(repo)
    assert len(records) == 1
    assert records[0]["status"] == "archived"
    assert records[0]["archive"]["volume_id"] == "fixture-volume"
    assert "archived_at" in records[0]


def test_existing_archive_target_conflict_requires_deliberate_flag(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path)
    write_run(repo, "2ec5fa6", "cold__seed_0")
    target = archive_root / "_artifacts/2ec5fa6/runs/cold__seed_0"
    target.mkdir(parents=True)
    (target / "metrics.json").write_text("different\n", encoding="utf-8")

    with pytest.raises(archive.ArchiveError, match="archive target already exists"):
        archive.apply_archive(
            repo_root=repo,
            archive_root=archive_root,
            run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
            volume_id="fixture-volume",
            reason="cold run",
        )

    with pytest.raises(archive.ArchiveError, match="inventory conflicts"):
        archive.apply_archive(
            repo_root=repo,
            archive_root=archive_root,
            run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
            volume_id="fixture-volume",
            reason="cold run",
            allow_existing_target=True,
        )


def test_index_conflict_refuses_until_deliberately_allowed(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path)
    write_run(repo, "2ec5fa6", "cold__seed_0")
    write_index(
        repo,
        {
            "schema_version": 1,
            "project": "rlrmp",
            "kind": "cold_run_artifact_archive",
            "run_ref": {
                "experiment": "2ec5fa6",
                "run": "cold__seed_0",
                "canonical_path": "_artifacts/2ec5fa6/runs/cold__seed_0",
            },
            "archive": {
                "volume_id": "other-volume",
                "root_relative_path": "_artifacts/2ec5fa6/runs/cold__seed_0",
                "local_symlink_path": "_artifacts/2ec5fa6/runs/cold__seed_0",
            },
            "inventory": {"bytes": 1, "files": 1},
            "status": "archived",
            "archived_at": "2026-06-21T00:00:00Z",
            "reason": "old",
            "tool_version": "archive_artifact_runs.v1",
        },
    )

    with pytest.raises(archive.ArchiveError, match="archive index already has state"):
        archive.build_archive_plan(
            repo_root=repo,
            archive_root=archive_root,
            run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
            volume_id="fixture-volume",
            reason="cold run",
        )

    with pytest.raises(archive.ArchiveError, match="archive index already has state"):
        archive.build_archive_plan(
            repo_root=repo,
            archive_root=archive_root,
            run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
            volume_id="fixture-volume",
            reason="cold run",
            allow_index_update=True,
        )


def test_verify_and_restore_materialize_archived_run(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path)
    source = write_run(repo, "2ec5fa6", "cold__seed_0")
    archive.apply_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
        volume_id="fixture-volume",
        reason="cold run",
    )

    verified = archive.verify_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
        volume_id="fixture-volume",
    )
    assert verified["status"] == "verified"
    assert verified["archive_status"] == "archived"

    restored = archive.restore_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
        volume_id="fixture-volume",
        reason="need local copy",
    )
    assert restored["status"] == "restored"
    assert source.is_dir()
    assert not source.is_symlink()
    assert (source / "metrics.json").read_text(encoding="utf-8") == "alpha\n"

    verified_restored = archive.verify_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
        volume_id="fixture-volume",
    )
    assert verified_restored["archive_status"] == "restored"
    assert [record["status"] for record in read_index(repo)] == ["archived", "restored"]


def test_internal_checkpoint_symlink_is_preserved(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path)
    source = write_run(repo, "b58592e", "checkpoint_symlink")
    checkpoint_dir = source / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "ckpt_0001.eqx").write_text("checkpoint\n", encoding="utf-8")
    (checkpoint_dir / "checkpoint_latest").symlink_to("ckpt_0001.eqx")

    plan = archive.build_archive_plan(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("b58592e/checkpoint_symlink"),
        volume_id="fixture-volume",
        reason="cold run",
    )
    symlink_entry = plan.inventory.manifest["checkpoints/checkpoint_latest"]
    assert symlink_entry == {"type": "symlink", "target": "ckpt_0001.eqx"}

    archive.apply_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("b58592e/checkpoint_symlink"),
        volume_id="fixture-volume",
        reason="cold run",
    )

    archived_link = (
        archive_root
        / "_artifacts"
        / "b58592e"
        / "runs"
        / "checkpoint_symlink"
        / "checkpoints"
        / "checkpoint_latest"
    )
    assert archived_link.is_symlink()
    assert archive.os.readlink(archived_link) == "ckpt_0001.eqx"
    verified = archive.verify_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("b58592e/checkpoint_symlink"),
        volume_id="fixture-volume",
    )
    assert verified["archive_status"] == "archived"

    archive.restore_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("b58592e/checkpoint_symlink"),
        volume_id="fixture-volume",
        reason="need local copy",
    )

    restored_link = source / "checkpoints" / "checkpoint_latest"
    assert restored_link.is_symlink()
    assert archive.os.readlink(restored_link) == "ckpt_0001.eqx"
    verified_restored = archive.verify_archive(
        repo_root=repo,
        archive_root=archive_root,
        run_ref=archive.parse_run_ref("b58592e/checkpoint_symlink"),
        volume_id="fixture-volume",
    )
    assert verified_restored["archive_status"] == "restored"


def test_refuses_noncanonical_run_refs_and_symlink_sources(tmp_path: Path) -> None:
    repo, archive_root = make_fixture_roots(tmp_path)
    target = write_run(repo, "2ec5fa6", "cold__seed_0")
    target.rename(repo / "_artifacts/2ec5fa6/runs/cold_target")
    target.symlink_to("cold_target", target_is_directory=True)

    with pytest.raises(archive.ArchiveError, match="exactly"):
        archive.parse_run_ref("2ec5fa6/runs/cold")
    with pytest.raises(archive.ArchiveError, match="single path segment"):
        archive.parse_run_ref("2ec5fa6/..")
    with pytest.raises(archive.ArchiveError, match="already a symlink"):
        archive.build_archive_plan(
            repo_root=repo,
            archive_root=archive_root,
            run_ref=archive.parse_run_ref("2ec5fa6/cold__seed_0"),
            volume_id="fixture-volume",
            reason="cold run",
        )


def make_fixture_roots(
    tmp_path: Path, *, volume_id: str = "fixture-volume"
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    archive_root = tmp_path / "archive"
    repo.mkdir()
    archive_root.mkdir()
    (archive_root / "VOLUME_ID").write_text(volume_id + "\n", encoding="utf-8")
    return repo, archive_root


def write_run(repo: Path, issue: str, run: str) -> Path:
    source = repo / "_artifacts" / issue / "runs" / run
    (source / "nested").mkdir(parents=True)
    (source / "metrics.json").write_text("alpha\n", encoding="utf-8")
    (source / "nested" / "checkpoint.eqx").write_text("beta\n", encoding="utf-8")
    return source


def read_index(repo: Path) -> list[dict[str, object]]:
    index_path = repo / "_artifacts/.archive_index.jsonl"
    return [
        archive.json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_index(repo: Path, record: dict[str, object]) -> None:
    index_path = repo / "_artifacts/.archive_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(archive.json.dumps(record) + "\n", encoding="utf-8")

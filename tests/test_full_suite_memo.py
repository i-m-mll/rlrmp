"""Canaries for the full-suite memo fingerprint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "full_suite.py"
SPEC = importlib.util.spec_from_file_location("full_suite", SCRIPT_PATH)
assert SPEC is not None
full_suite = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = full_suite
SPEC.loader.exec_module(full_suite)


def test_fingerprint_changes_when_feedbax_head_changes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    feedbax = _make_repo(tmp_path / "feedbax")
    _write_project_files(repo, feedbax)
    _commit_all(repo, "project files")

    first = full_suite.build_fingerprint(repo)
    assert first.ok, first.reason

    (feedbax / "feedbax.txt").write_text("changed\n", encoding="utf-8")
    _commit_all(feedbax, "feedbax change")

    second = full_suite.build_fingerprint(repo)
    assert second.ok, second.reason
    assert first.digest != second.digest
    assert first.payload["feedbax"]["head"] != second.payload["feedbax"]["head"]


def test_fingerprint_changes_when_uv_lock_changes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    feedbax = _make_repo(tmp_path / "feedbax")
    _write_project_files(repo, feedbax)
    _commit_all(repo, "project files")

    first = full_suite.build_fingerprint(repo)
    assert first.ok, first.reason

    (repo / "uv.lock").write_text("lock v2\n", encoding="utf-8")
    _commit_all(repo, "lock change")

    second = full_suite.build_fingerprint(repo)
    assert second.ok, second.reason
    assert first.digest != second.digest
    assert first.payload["uv_lock_sha256"] != second.payload["uv_lock_sha256"]


def test_fingerprint_uses_tree_not_commit_head(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    feedbax = _make_repo(tmp_path / "feedbax")
    _write_project_files(repo, feedbax)
    _commit_all(repo, "project files")

    first = full_suite.build_fingerprint(repo)
    assert first.ok, first.reason

    _git(repo, "commit", "--allow-empty", "-m", "metadata-only commit")

    second = full_suite.build_fingerprint(repo)
    assert second.ok, second.reason
    assert first.payload["rlrmp"]["tree"] == second.payload["rlrmp"]["tree"]
    assert first.digest == second.digest


def test_fingerprint_refuses_dirty_feedbax_checkout(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    feedbax = _make_repo(tmp_path / "feedbax")
    _write_project_files(repo, feedbax)
    _commit_all(repo, "project files")

    (feedbax / "initial.txt").write_text("dirty\n", encoding="utf-8")

    fingerprint = full_suite.build_fingerprint(repo)
    assert not fingerprint.ok
    assert fingerprint.digest is None
    assert "feedbax checkout has tracked or untracked changes" in fingerprint.reason


def test_untracked_dependency_docs_do_not_block_memo_recording(tmp_path: Path) -> None:
    memo_dir = tmp_path / "memo"
    repo = _make_repo(tmp_path / "repo")
    feedbax = _make_repo(tmp_path / "feedbax")
    _write_project_files(repo, feedbax)
    _commit_all(repo, "project files")

    docs_path = feedbax / "docs" / "foo.md"
    docs_path.parent.mkdir()
    docs_path.write_text("# local notes\n", encoding="utf-8")

    fingerprint = full_suite.build_fingerprint(repo)
    assert fingerprint.ok, fingerprint.reason
    full_suite.record_green(memo_dir, fingerprint, command=["pytest"])
    assert full_suite.memo_has_green(memo_dir, fingerprint)


def test_untracked_dependency_package_file_blocks_memo_recording(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    feedbax = _make_repo(tmp_path / "feedbax")
    _write_project_files(repo, feedbax)
    _commit_all(repo, "project files")

    package_path = feedbax / "feedbax" / "foo.py"
    package_path.parent.mkdir()
    package_path.write_text("VALUE = 1\n", encoding="utf-8")

    fingerprint = full_suite.build_fingerprint(repo)
    assert not fingerprint.ok
    assert fingerprint.digest is None
    assert "feedbax checkout has tracked or untracked changes" in fingerprint.reason


def test_memo_records_and_requires_exact_fingerprint(tmp_path: Path) -> None:
    memo_dir = tmp_path / "memo"
    first = full_suite.Fingerprint(ok=True, payload={"value": 1}, digest="abc")
    second = full_suite.Fingerprint(ok=True, payload={"value": 2}, digest="def")

    assert not full_suite.memo_has_green(memo_dir, first)
    full_suite.record_green(memo_dir, first, command=["pytest"])

    assert full_suite.memo_has_green(memo_dir, first)
    assert not full_suite.memo_has_green(memo_dir, second)


def _make_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Test User")
    (path / "initial.txt").write_text("initial\n", encoding="utf-8")
    _commit_all(path, "initial")
    return path


def _write_project_files(repo: Path, feedbax: Path) -> None:
    (repo / "pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.uv.sources.feedbax]",
                f'path = "{feedbax.as_posix()}"',
                "editable = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "uv.lock").write_text("lock v1\n", encoding="utf-8")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE)

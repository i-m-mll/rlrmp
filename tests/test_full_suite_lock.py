"""Focused tests for the cross-repository full-suite lock protocol."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "full_suite.py"
SPEC = importlib.util.spec_from_file_location("full_suite_lock_test_module", SCRIPT_PATH)
assert SPEC is not None
full_suite = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = full_suite
SPEC.loader.exec_module(full_suite)


HOLDER_CODE = """
import importlib.util
import os
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("shared_full_suite", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with module.FullSuiteLock(
    Path(sys.argv[2]),
    repo_root=Path(sys.argv[3]),
    repository="feedbax",
    command=["feedbax", "scripts/full_suite.sh"],
):
    print("READY", flush=True)
    if sys.argv[4] == "crash":
        os._exit(23)
    sys.stdin.readline()
"""


def test_lock_directory_default_and_override(tmp_path: Path) -> None:
    default = full_suite.full_suite_lock_dir({})
    assert default.parent == Path(full_suite.tempfile.gettempdir())
    assert default.name == f"full-suite-lock-{os.getuid()}"
    assert full_suite.full_suite_lock_path({}) == default / "full-suite.lock"

    override = tmp_path / "shared"
    assert full_suite.full_suite_lock_dir({"FULL_SUITE_LOCK_DIR": str(override)}) == override
    assert (
        full_suite.full_suite_lock_path({"FULL_SUITE_LOCK_DIR": str(override)})
        == override / "full-suite.lock"
    )


def test_lock_acquisition_writes_protocol_metadata(tmp_path: Path) -> None:
    lock_path = tmp_path / "locks" / "full-suite.lock"
    worktree = tmp_path / "rlrmp"
    command = ["scripts/full_suite.sh", "--force"]

    with full_suite.FullSuiteLock(lock_path, repo_root=worktree, command=command):
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        assert metadata["schema_version"] == 1
        assert metadata["protocol_version"] == 1
        assert metadata["repository"] == "rlrmp"
        assert metadata["pid"] == os.getpid()
        assert metadata["worktree"] == str(worktree)
        assert metadata["command"] == command
        assert metadata["host"]
        assert metadata["started_at"]


def test_main_fails_fast_with_cross_repository_holder(tmp_path: Path) -> None:
    lock_dir = tmp_path / "shared-lock"
    lock_path = lock_dir / "full-suite.lock"
    holder = _start_holder(lock_path, tmp_path / "feedbax")
    try:
        env = {**os.environ, "FULL_SUITE_LOCK_DIR": str(lock_dir)}
        blocked = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--no-memo"],
            cwd=SCRIPT_PATH.parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert blocked.returncode == 75
        assert "Full suite already running" in blocked.stderr
        assert "repository=feedbax" in blocked.stderr
        assert f"worktree={tmp_path / 'feedbax'}" in blocked.stderr
        assert "pid=" in blocked.stderr
        assert "started_at=" in blocked.stderr
    finally:
        holder.communicate(input="release\n", timeout=10)


def test_crashed_holder_releases_kernel_lock_despite_stale_metadata(tmp_path: Path) -> None:
    lock_path = tmp_path / "shared-lock" / "full-suite.lock"
    holder = _start_holder(lock_path, tmp_path / "feedbax", mode="crash")
    assert holder.wait(timeout=10) == 23

    stale = json.loads(lock_path.read_text(encoding="utf-8"))
    assert stale["repository"] == "feedbax"
    with full_suite.FullSuiteLock(lock_path, repo_root=tmp_path / "rlrmp"):
        current = json.loads(lock_path.read_text(encoding="utf-8"))
        assert current["repository"] == "rlrmp"
        assert current["pid"] == os.getpid()


def _start_holder(lock_path: Path, worktree: Path, *, mode: str = "wait") -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER_CODE, str(SCRIPT_PATH), str(lock_path), str(worktree), mode],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    return process

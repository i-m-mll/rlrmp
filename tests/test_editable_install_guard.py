"""Negative-canary coverage for the stale-editable-install guard.

`scripts/dev_tests.sh` and `scripts/full_suite.sh` both refuse to run when the
rlrmp package the shared venv would import does not resolve to the invoking
worktree's `src/` (issue 81e4588: the shared `.venv`'s editable install can
silently point at a stale, deleted worktree). These tests stage a copy of each
real wrapper script into an isolated fake repo and substitute a fake `uv` on
PATH so the guard's resolution result is controllable, without needing a real
Python/JAX environment.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPTS = ("dev_tests.sh", "full_suite.sh")


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _write_fake_uv(bin_dir: Path) -> Path:
    """A fake `uv` that intercepts only the guard's resolution invocation.

    The guard's `python -c ...` source contains the literal `rlrmp.__file__`;
    the final `exec uv run --no-sync python <wrapper>.py "$@"` call does not,
    so grepping for that substring distinguishes the two invocations without
    needing a real rlrmp install. `FAKE_RESOLVED_RLRMP_PATH` (set by the
    test) stands in for the path a real `import rlrmp` would resolve to.
    """
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if printf '%s\\n' "$*" | grep -q "rlrmp.__file__"; then
    last_arg="${@: -1}"
    expected="$last_arg/rlrmp"
    resolved="${FAKE_RESOLVED_RLRMP_PATH:-$expected}"
    if [[ "$resolved" != "$expected" ]]; then
        printf 'resolved=%s expected=%s\\n' "$resolved" "$expected"
        exit 1
    fi
    exit 0
fi
echo "FAKE_UV: ran wrapped script: $*"
exit 0
""",
        encoding="utf-8",
    )
    _make_executable(fake_uv)
    return fake_uv


def _stage_wrapper(tmp_path: Path, script_name: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    shutil.copy(REPO_ROOT / "scripts" / script_name, repo / "scripts" / script_name)
    _make_executable(repo / "scripts" / script_name)
    return repo


def _run(
    repo: Path, script_name: str, bin_dir: Path, *, resolved_path: str | None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    if resolved_path is not None:
        env["FAKE_RESOLVED_RLRMP_PATH"] = resolved_path
    else:
        env.pop("FAKE_RESOLVED_RLRMP_PATH", None)
    return subprocess.run(
        ["bash", str(repo / "scripts" / script_name)],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


@pytest.mark.parametrize("script_name", WRAPPER_SCRIPTS)
def test_guard_fails_loudly_on_stale_editable_install(tmp_path: Path, script_name: str) -> None:
    repo = _stage_wrapper(tmp_path, script_name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)

    stale_path = str(tmp_path / "worktrees" / "some-stale-worktree" / "src" / "rlrmp")
    result = _run(repo, script_name, bin_dir, resolved_path=stale_path)

    assert result.returncode != 0
    assert "stale editable install" in result.stdout
    assert stale_path in result.stdout
    assert str(repo / "src" / "rlrmp") in result.stdout
    # The guard must trip *before* the wrapped script (dev_tests.py /
    # full_suite.py) is ever invoked.
    assert "FAKE_UV: ran wrapped script" not in result.stdout


@pytest.mark.parametrize("script_name", WRAPPER_SCRIPTS)
def test_guard_passes_when_editable_install_matches(tmp_path: Path, script_name: str) -> None:
    repo = _stage_wrapper(tmp_path, script_name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_uv(bin_dir)

    matching_path = str(repo / "src" / "rlrmp")
    result = _run(repo, script_name, bin_dir, resolved_path=matching_path)

    assert result.returncode == 0
    assert "stale editable install" not in result.stdout
    assert "FAKE_UV: ran wrapped script" in result.stdout

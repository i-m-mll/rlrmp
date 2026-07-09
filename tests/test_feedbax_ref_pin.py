"""Feedbax pin-drift guard: ci/feedbax-ref.toml must match the editable checkout.

Local editable installs run rlrmp against the feedbax repo root; CI runs
against the rev pinned in ci/feedbax-ref.toml. This test fails when the two
diverge, so pin bumps happen in the same wave as the feedbax change (see
AGENTS.md "Integration verification bar"; precedent: issue 7766182). In CI and
other non-git installs the check skips — CI checks out the pin by construction.
The pinned rev must also be reachable from the last-fetched feedbax
origin/develop ref so local CI cannot bless an unpublished feedbax SHA.
"""

from __future__ import annotations

import os
import subprocess
import tomllib
import warnings
from collections.abc import Sequence
from pathlib import Path

import pytest

import feedbax

_PIN_FILE = Path(__file__).resolve().parents[1] / "ci" / "feedbax-ref.toml"
_FEEDBAX_REMOTE_REF = "refs/remotes/origin/develop"
_ALLOW_UNPUBLISHED_ENV = "RLRMP_ALLOW_UNPUBLISHED_FEEDBAX_PIN"


def _feedbax_git_root() -> Path | None:
    package_dir = Path(feedbax.__file__).resolve().parent
    for candidate in (package_dir, *package_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(
    root: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _git_stdout(root: Path, args: Sequence[str]) -> str:
    return _run_git(root, args).stdout.strip()


def _feedbax_origin_develop_rev(root: Path) -> str:
    try:
        return _git_stdout(root, ["rev-parse", "--verify", _FEEDBAX_REMOTE_REF])
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"Feedbax checkout at {root} has no last-fetched {_FEEDBAX_REMOTE_REF} ref. "
            "Fetch feedbax origin/develop before running the rlrmp suite so the "
            "ci/feedbax-ref.toml pin can be checked for publish reachability."
        ) from exc


def _warn_unpublished_pin_allowed(root: Path, pinned: str, origin_develop: str) -> None:
    warnings.warn(
        f"WARNING: {_ALLOW_UNPUBLISHED_ENV}=1 is set, so rlrmp is allowing an "
        f"unpublished feedbax pin for deliberate local-only iteration. "
        f"ci/feedbax-ref.toml pins {pinned}, but feedbax {_FEEDBAX_REMOTE_REF} at "
        f"{root} is only {origin_develop}. Push feedbax develop and fetch origin/develop "
        "before relying on CI reproducibility.",
        RuntimeWarning,
        stacklevel=2,
    )


def _assert_feedbax_pin_reachable_from_origin_develop(
    root: Path,
    pinned: str,
    *,
    allow_unpublished: bool,
) -> None:
    origin_develop = _feedbax_origin_develop_rev(root)
    reachability = _run_git(
        root,
        ["merge-base", "--is-ancestor", pinned, _FEEDBAX_REMOTE_REF],
        check=False,
    )
    if reachability.returncode == 0:
        return
    if reachability.returncode != 1:
        reachability.check_returncode()
    if allow_unpublished:
        _warn_unpublished_pin_allowed(root, pinned, origin_develop)
        return
    raise AssertionError(
        f"ci/feedbax-ref.toml pins feedbax at {pinned}, but that commit is not "
        f"reachable from the last-fetched feedbax {_FEEDBAX_REMOTE_REF} ({origin_develop}) "
        f"in checkout {root}. Push feedbax develop if the pin is local-only, or fetch "
        "feedbax origin/develop if the commit is already published."
    )


def test_feedbax_pin_reachability_accepts_origin_ancestor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text
        calls.append(tuple(cmd[3:]))
        if cmd[3:] == ["rev-parse", "--verify", _FEEDBAX_REMOTE_REF]:
            return subprocess.CompletedProcess(cmd, 0, stdout="origin-tip\n", stderr="")
        if cmd[3:] == ["merge-base", "--is-ancestor", "pin", _FEEDBAX_REMOTE_REF]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    _assert_feedbax_pin_reachable_from_origin_develop(
        Path("/feedbax"),
        "pin",
        allow_unpublished=False,
    )

    assert calls == [
        ("rev-parse", "--verify", _FEEDBAX_REMOTE_REF),
        ("merge-base", "--is-ancestor", "pin", _FEEDBAX_REMOTE_REF),
    ]


def test_feedbax_pin_reachability_fails_when_origin_is_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text
        if cmd[3:] == ["rev-parse", "--verify", _FEEDBAX_REMOTE_REF]:
            return subprocess.CompletedProcess(cmd, 0, stdout="origin-tip\n", stderr="")
        if cmd[3:] == ["merge-base", "--is-ancestor", "pin", _FEEDBAX_REMOTE_REF]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AssertionError, match="Push feedbax develop"):
        _assert_feedbax_pin_reachable_from_origin_develop(
            Path("/feedbax"),
            "pin",
            allow_unpublished=False,
        )


def test_feedbax_pin_reachability_escape_hatch_warns_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text
        if cmd[3:] == ["rev-parse", "--verify", _FEEDBAX_REMOTE_REF]:
            return subprocess.CompletedProcess(cmd, 0, stdout="origin-tip\n", stderr="")
        if cmd[3:] == ["merge-base", "--is-ancestor", "pin", _FEEDBAX_REMOTE_REF]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.warns(RuntimeWarning, match=_ALLOW_UNPUBLISHED_ENV):
        _assert_feedbax_pin_reachable_from_origin_develop(
            Path("/feedbax"),
            "pin",
            allow_unpublished=True,
        )


def test_feedbax_pin_reachability_fails_when_origin_ref_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text
        if cmd[3:] == ["rev-parse", "--verify", _FEEDBAX_REMOTE_REF]:
            raise subprocess.CalledProcessError(1, cmd, stderr="missing")
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AssertionError, match="Fetch feedbax origin/develop"):
        _feedbax_origin_develop_rev(Path("/feedbax"))


def test_feedbax_ref_pin_matches_editable_checkout() -> None:
    root = _feedbax_git_root()
    if root is None:
        pytest.skip(
            "feedbax is not installed from a git checkout; CI pins the rev by construction"
        )
    pinned = tomllib.loads(_PIN_FILE.read_text())["rev"]
    head = _git_stdout(root, ["rev-parse", "HEAD"])
    assert head == pinned, (
        f"ci/feedbax-ref.toml pins feedbax at {pinned}, but the editable checkout at "
        f"{root} is at {head}. Bump the pin in the same wave as the feedbax change and "
        "rerun the full suite (AGENTS.md 'Integration verification bar'; issue 7766182)."
    )
    _assert_feedbax_pin_reachable_from_origin_develop(
        root,
        pinned,
        allow_unpublished=os.environ.get(_ALLOW_UNPUBLISHED_ENV) == "1",
    )

"""Feedbax pin-drift guard: ci/feedbax-ref.toml must match the editable checkout.

Local editable installs run rlrmp against the feedbax repo root; CI runs
against the rev pinned in ci/feedbax-ref.toml. This test fails when the two
diverge, so pin bumps happen in the same wave as the feedbax change (see
AGENTS.md "Integration verification bar"; precedent: issue 7766182). In CI and
other non-git installs the check skips — CI checks out the pin by construction.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

import feedbax

_PIN_FILE = Path(__file__).resolve().parents[1] / "ci" / "feedbax-ref.toml"


def _feedbax_git_root() -> Path | None:
    package_dir = Path(feedbax.__file__).resolve().parent
    for candidate in (package_dir, *package_dir.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def test_feedbax_ref_pin_matches_editable_checkout() -> None:
    root = _feedbax_git_root()
    if root is None:
        pytest.skip(
            "feedbax is not installed from a git checkout; CI pins the rev by construction"
        )
    pinned = tomllib.loads(_PIN_FILE.read_text())["rev"]
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == pinned, (
        f"ci/feedbax-ref.toml pins feedbax at {pinned}, but the editable checkout at "
        f"{root} is at {head}. Bump the pin in the same wave as the feedbax change and "
        "rerun the full suite (AGENTS.md 'Integration verification bar'; issue 7766182)."
    )

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

from pathlib import Path

import pytest

import feedbax
from feedbax.testing.version_pin import check_version_pin

_PIN_FILE = Path(__file__).resolve().parents[1] / "ci" / "feedbax-ref.toml"
_FEEDBAX_REMOTE_REF = "refs/remotes/origin/develop"
_ALLOW_UNPUBLISHED_ENV = "RLRMP_ALLOW_UNPUBLISHED_FEEDBAX_PIN"


def test_feedbax_ref_pin_matches_editable_checkout() -> None:
    report = check_version_pin(
        package_name="feedbax",
        pin_file=_PIN_FILE,
        package_path=Path(feedbax.__file__),
        remote_ref=_FEEDBAX_REMOTE_REF,
        escape_hatch_env=_ALLOW_UNPUBLISHED_ENV,
    )
    if report.skipped:
        pytest.skip(report.skip_reason)

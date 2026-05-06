"""Sanity checks for the role-based artifact gitignore policy.

Bug: 58415d2 — repo structure migration to the mirror-tree (`results/` for
specs and narratives, `_artifacts/` for bulk outputs) layout. These tests
encode the user-visible promise of the new `.gitignore`: artifacts are
tracked or ignored based on their ROLE (file kind), not on the directory
NAME they happen to live under. New cloud providers (`runpod/`, `modal/`,
`coreweave/`) and new experiment phases must slot in without any new
ignore rules.

The tests shell out to `git check-ignore -q`, which is the authoritative
oracle for whether a path would be tracked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_ignored(path: str) -> bool:
    """Return True iff git would ignore `path` (relative to repo root).

    Uses `git check-ignore -q`: exit 0 means path matches an ignore rule,
    exit 1 means it does not. Note that paths need not exist on disk;
    `check-ignore` operates purely on patterns.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore failed for {path!r}: rc={result.returncode}"
        )
    return result.returncode == 0


def assert_committable(path: str) -> None:
    assert not _is_ignored(path), (
        f"{path!r} should be committable under the role-based policy "
        f"but is currently ignored."
    )


def assert_ignored(path: str) -> None:
    assert _is_ignored(path), (
        f"{path!r} should be ignored under the role-based policy "
        f"but would currently be tracked."
    )


# --- Specs and narratives under `results/` are tracked --------------------

@pytest.mark.parametrize(
    "path",
    [
        "results/part2_5/README.md",
        "results/part2_5/synthesis_review.md",
        "results/part2_5/runs/baseline__standard_12k/run.json",
        "results/part2_5/runs/baseline__standard_12k/notes.md",
        "results/part2_5/figures/peak_velocity/spec.json",
        "results/part2_5/figures/peak_velocity/figure.json",
        "results/part2_5/figures/peak_velocity/figure.png",
        # Legacy stub configs:
        "results/centerout_apt_pert1/config.json",
        # Typora markdown sidecar PNGs stay tracked:
        "results/1_general.assets/some_image.png",
        "results/2_general.assets/file-20241126113220236.png",
    ],
)
def test_committable(path: str) -> None:
    assert_committable(path)


# --- Bulk artifacts and unknown kinds under `results/` are ignored --------

@pytest.mark.parametrize(
    "path",
    [
        # Heavy training outputs anywhere under results/:
        "results/part2_5/runs/baseline__standard_12k/adversarial_model.eqx",
        "results/part2_5/runs/baseline__standard_12k/adversarial_losses.npz",
        "results/part2_5/runs/baseline__standard_12k/train.log",
        # Cloud-provider directory names (the whole point of the migration):
        "results/part2_5/runpod/baseline/standard_12k/adversarial_losses.npz",
        "results/part2_5/runpod/baseline/standard_12k/checkpoint.eqx",
        "results/part2_5/modal/some_run/checkpoint.eqx",
        "results/part2_5/coreweave/run/x.eqx",
        # Heavy figure renders:
        "results/part2_5/figures/big/figure.html",
        # Unknown file kinds in unknown subdirs:
        "results/part2_5/some_random_dir/data.pkl",
        "results/part2_5/something/random.png",
        # Everything under _artifacts/ except the README:
        "_artifacts/part2_5/runs/baseline/checkpoints/x.eqx",
        "_artifacts/part2_5/runs/baseline/run.json",
        "_artifacts/part2_5/figures/big/figure.html",
    ],
)
def test_ignored(path: str) -> None:
    assert_ignored(path)


# --- The _artifacts/README is the one tracked file under _artifacts/ ------

def test_artifacts_readme_is_tracked() -> None:
    assert_committable("_artifacts/README.md")


# --- Every currently-committed file under results/ stays committable ------

def test_no_committed_file_is_now_ignored() -> None:
    """Regression guard: the migration must not retroactively ignore anything
    already tracked under `results/`. If this fails, the gitignore needs another
    whitelist line."""
    listing = subprocess.run(
        ["git", "ls-files", "results/"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    broken = [
        line for line in listing.stdout.splitlines()
        if line and _is_ignored(line)
    ]
    assert not broken, (
        f"{len(broken)} currently-committed file(s) became ignored after the "
        f"migration:\n  " + "\n  ".join(broken[:20])
    )

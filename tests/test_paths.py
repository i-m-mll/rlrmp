"""Sanity checks for the role-based artifact gitignore policy and path helpers.

Bug: 58415d2 — repo structure migration to the mirror-tree (`results/` for
specs and narratives, `_artifacts/` for bulk outputs) layout. These tests
encode the user-visible promise of the new `.gitignore`: artifacts are
tracked or ignored based on their ROLE (file kind), not on the directory
NAME they happen to live under. New cloud providers (`runpod/`, `modal/`,
`coreweave/`) and new experiment phases must slot in without any new
ignore rules.

Bug: fd64bb4 — Phase 2 path-helper module. The helper tests verify that
``rlrmp.paths`` returns the correct absolute paths for the mirror-tree layout
and that the two sides of the mirror are structurally consistent.

The gitignore tests shell out to `git check-ignore -q`, which is the
authoritative oracle for whether a path would be tracked.
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


# ---------------------------------------------------------------------------
# rlrmp.paths helper assertions
# ---------------------------------------------------------------------------

from rlrmp.paths import (
    REPO_ROOT as PATHS_REPO_ROOT,
    figure_artifact_dir,
    figure_spec_dir,
    run_artifact_dir,
    run_spec_dir,
)


class TestRunSpecDir:
    """``run_spec_dir`` returns absolute paths inside ``results/``."""

    def test_correct_path(self) -> None:
        path = run_spec_dir("foo", "bar")
        assert path == PATHS_REPO_ROOT / "results" / "foo" / "runs" / "bar"

    def test_is_absolute(self) -> None:
        assert run_spec_dir("foo", "bar").is_absolute()

    def test_is_inside_results(self) -> None:
        path = run_spec_dir("foo", "bar")
        assert str(path).startswith(str(PATHS_REPO_ROOT / "results"))


class TestRunArtifactDir:
    """``run_artifact_dir`` returns absolute paths inside ``_artifacts/``."""

    def test_correct_path(self) -> None:
        path = run_artifact_dir("foo", "bar")
        assert path == PATHS_REPO_ROOT / "_artifacts" / "foo" / "runs" / "bar"

    def test_is_absolute(self) -> None:
        assert run_artifact_dir("foo", "bar").is_absolute()

    def test_is_inside_artifacts(self) -> None:
        path = run_artifact_dir("foo", "bar")
        assert str(path).startswith(str(PATHS_REPO_ROOT / "_artifacts"))


class TestRunDirMirrorInvariant:
    """The spec and artifact directories for the same run are mirror siblings."""

    def test_same_run_segment(self) -> None:
        """The final path component (run identifier) must be identical."""
        spec = run_spec_dir("foo", "bar")
        artifact = run_artifact_dir("foo", "bar")
        assert spec.name == artifact.name == "bar"

    def test_same_exp_segment(self) -> None:
        """The experiment slug appears in the same relative position in both."""
        spec = run_spec_dir("part2_5", "baseline__standard_12k")
        artifact = run_artifact_dir("part2_5", "baseline__standard_12k")
        # spec:     .../results/part2_5/runs/baseline__standard_12k
        # artifact: .../_artifacts/part2_5/runs/baseline__standard_12k
        assert spec.parts[-3] == artifact.parts[-3] == "part2_5"
        assert spec.parts[-2] == artifact.parts[-2] == "runs"

    def test_disjoint_top_level(self) -> None:
        """The two sides of the mirror must NOT share the same top-level dir."""
        spec = run_spec_dir("foo", "bar")
        artifact = run_artifact_dir("foo", "bar")
        assert spec.parts[-4] != artifact.parts[-4]


class TestFigureSpecDir:
    """``figure_spec_dir`` returns absolute paths inside ``results/``."""

    def test_correct_path(self) -> None:
        path = figure_spec_dir("foo", "my_figure")
        assert path == PATHS_REPO_ROOT / "results" / "foo" / "figures" / "my_figure"

    def test_is_absolute(self) -> None:
        assert figure_spec_dir("foo", "my_figure").is_absolute()

    def test_is_inside_results(self) -> None:
        path = figure_spec_dir("foo", "my_figure")
        assert str(path).startswith(str(PATHS_REPO_ROOT / "results"))


class TestFigureArtifactDir:
    """``figure_artifact_dir`` returns absolute paths inside ``_artifacts/``."""

    def test_correct_path(self) -> None:
        path = figure_artifact_dir("foo", "my_figure")
        assert path == PATHS_REPO_ROOT / "_artifacts" / "foo" / "figures" / "my_figure"

    def test_is_absolute(self) -> None:
        assert figure_artifact_dir("foo", "my_figure").is_absolute()

    def test_is_inside_artifacts(self) -> None:
        path = figure_artifact_dir("foo", "my_figure")
        assert str(path).startswith(str(PATHS_REPO_ROOT / "_artifacts"))


class TestFigureDirMirrorInvariant:
    """The spec and artifact directories for the same figure are mirror siblings."""

    def test_same_figure_segment(self) -> None:
        spec = figure_spec_dir("foo", "myfig")
        artifact = figure_artifact_dir("foo", "myfig")
        assert spec.name == artifact.name == "myfig"

    def test_same_figures_subdirectory(self) -> None:
        spec = figure_spec_dir("part2_5", "peak_velocity")
        artifact = figure_artifact_dir("part2_5", "peak_velocity")
        assert spec.parts[-2] == artifact.parts[-2] == "figures"

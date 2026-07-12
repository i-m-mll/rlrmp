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

Bug: 0077b42 — Phase 2 completion. Train scripts split the tracked
``run.json`` spec from the bulk-artifact outputs. Tests below verify that
``derive_spec_dir`` maps an ``_artifacts/`` artifact path to the corresponding
``results/`` spec path under the mirror invariant.

The gitignore tests shell out to `git check-ignore -q`, which is the
authoritative oracle for whether a path would be tracked.
"""

from __future__ import annotations
import subprocess
from pathlib import Path

import pytest

from rlrmp.train.run_spec_authoring import derive_spec_dir, derive_spec_path

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
        raise RuntimeError(f"git check-ignore failed for {path!r}: rc={result.returncode}")
    return result.returncode == 0


def assert_committable(path: str) -> None:
    assert not _is_ignored(path), (
        f"{path!r} should be committable under the role-based policy but is currently ignored."
    )


def assert_ignored(path: str) -> None:
    assert _is_ignored(path), (
        f"{path!r} should be ignored under the role-based policy but would currently be tracked."
    )


# --- Specs and narratives under `results/` are tracked --------------------


@pytest.mark.parametrize(
    "path",
    [
        "results/2ef67ca/README.md",
        "results/b557d4e/synthesis_review.md",
        "results/part2_5/runs/baseline__standard_12k.json",
        "results/part2_5/runs/baseline__standard_12k/notes.md",
        "results/part2_5/figures/peak_velocity/spec.json",
        "results/part2_5/figures/peak_velocity/figure.json",
        "results/part2_5/figures/peak_velocity/figure.png",
        # Local navigation symlinks to ignored figure renders stay tracked:
        "results/part2_5/figures/big/figure.html",
        # Legacy stub configs:
        "results/2ef67ca/models/centerout_apt_pert1/config.json",
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
        # Unknown file kinds in unknown subdirs:
        "results/part2_5/some_random_dir/data.pkl",
        "results/part2_5/something/random.png",
        # Legacy markdown sidecar asset directories are bulk artifacts:
        "results/1_general.assets/some_image.png",
        "results/2_general.assets/file-20241126113220236.png",
        "results/2_training-methods.assets/file-20250620091043895.png",
    ],
)
def test_ignored(path: str) -> None:
    assert_ignored(path)


# --- config.json depth-restriction contract (Bug: 3577dee) ----------------
# The broad `!results/**/config.json` whitelist was depth-blind and caused
# bulk per-run configs from cloud providers to be tracked. It is now replaced
# with three depth-specific patterns. These tests encode that contract.


@pytest.mark.parametrize(
    "path",
    [
        # depth 2: results/<exp>/config.json — these are the top-level experiment stubs
        "results/2ef67ca/models/centerout_apt_pert1/config.json",
        "results/2ef67ca/config.json",
        # depth 3: results/<exp>/<subdir>/config.json — legacy running_cost_nn1e6 layout
        "results/2ef67ca/running_cost_nn1e6/config.json",
        # depth 4 via models/: results/<exp>/models/<run>/config.json
        "results/2ef67ca/models/baseline_standard/config.json",
        "results/2ef67ca/models/minimax_test2/config.json",
    ],
)
def test_legacy_config_json_committable(path: str) -> None:
    """Legacy config.json stubs at known depths remain committable."""
    assert_committable(path)


@pytest.mark.parametrize(
    "path",
    [
        # depth 4, non-models subdir — the exact pattern that bit us (runpod)
        "results/part2_5/modal/some_run/config.json",
        "results/part2_5/coreweave/run/config.json",
        # depth 5 — the actual runpod layout that caused the Phase 2 auth-merge issue
        "results/part2_5/runpod/baseline/standard_12k/config.json",
        # arbitrary deep nesting
        "results/foo/bar/baz/config.json",
    ],
)
def test_bulk_config_json_ignored(path: str) -> None:
    """config.json files from bulk cloud-provider trees are ignored, not tracked.

    This is the regression guard for the depth-blind `!results/**/config.json`
    whitelist that allowed runpod bulk configs to slip into the index.
    """
    assert_ignored(path)


# --- _artifacts/ is ignored as the bulk-output tree -----------------------


def test_artifacts_tree_is_ignored() -> None:
    """The bulk artifact tree itself is ignored.

    Worktrees normally expose ``_artifacts`` as a symlink to the repo-root
    shared artifact directory. Asking ``git check-ignore`` about nested paths
    through that symlink fails with "beyond a symbolic link", so the durable
    policy assertion is on the tree root.
    """

    assert_ignored("_artifacts")


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
    broken = [line for line in listing.stdout.splitlines() if line and _is_ignored(line)]
    assert not broken, (
        f"{len(broken)} currently-committed file(s) became ignored after the "
        f"migration:\n  " + "\n  ".join(broken[:20])
    )


# ---------------------------------------------------------------------------
# rlrmp.paths helper assertions
# ---------------------------------------------------------------------------

from rlrmp.paths import (  # noqa: E402
    REPO_ROOT as PATHS_REPO_ROOT,
    figure_artifact_dir,
    figure_spec_dir,
    flat_run_spec_path,
    resolve_run_artifact_path,
    run_artifact_dir,
    run_spec_dir,
    run_spec_path,
    run_spec_sidecar_dir,
)


class TestRunSpecDir:
    """``run_spec_dir`` returns the optional tracked run sidecar directory."""

    def test_correct_path(self) -> None:
        path = run_spec_dir("foo", "bar")
        assert path == PATHS_REPO_ROOT / "results" / "foo" / "runs" / "bar"

    def test_is_absolute(self) -> None:
        assert run_spec_dir("foo", "bar").is_absolute()

    def test_is_inside_results(self) -> None:
        path = run_spec_dir("foo", "bar")
        assert str(path).startswith(str(PATHS_REPO_ROOT / "results"))


class TestRunSpecSidecarDir:
    """``run_spec_sidecar_dir`` names the optional sidecar directory explicitly."""

    def test_matches_legacy_helper(self) -> None:
        assert run_spec_sidecar_dir("foo", "bar") == run_spec_dir("foo", "bar")

    def test_is_distinct_from_flat_recipe_path(self, tmp_path: Path) -> None:
        recipe = run_spec_path("foo", "bar", repo_root=tmp_path)
        sidecar = tmp_path / "results" / "foo" / "runs" / "bar"

        assert recipe == tmp_path / "results" / "foo" / "runs" / "bar.json"
        assert sidecar != recipe


class TestRunSpecPath:
    """``run_spec_path`` always resolves the canonical flat spec."""

    def test_returns_existing_flat_spec(self, tmp_path: Path) -> None:
        flat = tmp_path / "results" / "abc1234" / "runs" / "run_a.json"
        flat.parent.mkdir(parents=True)
        flat.write_text("{}", encoding="utf-8")

        assert run_spec_path("abc1234", "run_a", repo_root=tmp_path) == flat

    def test_missing_path_points_to_flat_convention(self, tmp_path: Path) -> None:
        assert (
            run_spec_path("abc1234", "run_a", repo_root=tmp_path)
            == tmp_path / "results" / "abc1234" / "runs" / "run_a.json"
        )

    def test_for_write_matches_reader_path(self, tmp_path: Path) -> None:
        flat = tmp_path / "results" / "abc1234" / "runs" / "run_a.json"
        assert run_spec_path("abc1234", "run_a", repo_root=tmp_path) == flat
        assert run_spec_path("abc1234", "run_a", repo_root=tmp_path, for_write=True) == flat
        assert flat_run_spec_path("abc1234", "run_a", repo_root=tmp_path) == flat


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


class TestResolveRunArtifactPath:
    """``resolve_run_artifact_path`` supports legacy and post-run layouts."""

    def test_prefers_direct_existing_path(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "run_a"
        direct = artifact_dir / "trained_model.eqx"
        post_run = artifact_dir / "artifacts" / "trained_model.eqx"
        direct.parent.mkdir(parents=True)
        post_run.parent.mkdir(parents=True)
        direct.write_text("legacy", encoding="utf-8")
        post_run.write_text("post-run", encoding="utf-8")

        assert resolve_run_artifact_path(artifact_dir, "trained_model.eqx") == direct

    def test_uses_post_run_artifacts_child(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "run_a"
        post_run = artifact_dir / "artifacts" / "training_history.eqx"
        post_run.parent.mkdir(parents=True)
        post_run.write_text("post-run", encoding="utf-8")

        assert resolve_run_artifact_path(artifact_dir, "training_history.eqx") == post_run

    def test_uses_existing_nested_legacy_path(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "run_a"
        nested = artifact_dir / "run_a" / "checkpoints" / "checkpoint_0000100"
        nested.parent.mkdir(parents=True)
        nested.write_text("checkpoint", encoding="utf-8")

        assert (
            resolve_run_artifact_path(
                artifact_dir,
                "checkpoints",
                "checkpoint_0000100",
            )
            == nested
        )

    def test_missing_path_points_to_direct_convention(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "run_a"

        assert (
            resolve_run_artifact_path(artifact_dir, "trained_model.eqx")
            == artifact_dir / "trained_model.eqx"
        )


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


# ---------------------------------------------------------------------------
# Run-spec authoring path helpers (Bug: 0077b42)
# ---------------------------------------------------------------------------

class TestDerivedSpecDirMirrorInvariant:
    """``derive_spec_dir`` honours the mirror invariant for ``_artifacts/`` paths.

    For an artifact path under ``_artifacts/<exp>/runs/<run>/`` the helper
    must produce ``results/<exp>/runs/<run>/`` — the same path returned by
    ``rlrmp.paths.run_spec_dir(exp, run)`` for the same ``(exp, run)``.
    """

    def test_minimax_run_artifact_to_run_spec(self) -> None:
        artifact = run_artifact_dir("minimax", "seed_0")
        spec = derive_spec_dir(artifact)
        assert spec == run_spec_dir("minimax", "seed_0")


class TestDerivedSpecPathFlat:
    """``derive_spec_path`` returns the canonical FLAT run-recipe file path.

    The recipe lives at ``results/<exp>/runs/<run>.json`` and its optional
    sidecars live in the sibling ``results/<exp>/runs/<run>/`` directory.
    """

    def test_minimax_artifact_to_flat_recipe(self) -> None:
        artifact = run_artifact_dir("minimax", "seed_0")
        spec = derive_spec_path(artifact)
        assert spec == run_spec_path("minimax", "seed_0")
        assert spec.name == "seed_0.json"

    def test_relative_artifact_path_resolves_to_flat(self) -> None:
        spec = derive_spec_path(Path("_artifacts/minimax/runs/foo"))
        assert spec == run_spec_path("minimax", "foo")

    def test_flat_recipe_is_not_nested_run_json(self) -> None:
        artifact = run_artifact_dir("minimax", "seed_0")
        spec = derive_spec_path(artifact)
        nested = run_spec_dir("minimax", "seed_0") / "run.json"
        assert spec != nested

class TestDerivedSpecPathFallback:
    """Out-of-tree paths map to a flat ``<output_dir>_spec.json`` sibling file.

    The ``_spec`` suffix (inherited from ``derive_spec_dir``) keeps the flat
    recipe file from colliding with an out-of-tree artifact directory of the
    same name. The recipe is still a flat ``.json`` file, never nested.
    """

    def test_minimax_out_of_tree_path_uses_flat_sibling(self, tmp_path: Path) -> None:
        out = tmp_path / "minirun"
        out.mkdir()
        spec = derive_spec_path(out)
        assert spec.name == "minirun_spec.json"
        assert spec.parent == out.parent
        assert spec.suffix == ".json"


class TestDerivedSpecDirFallback:
    """When ``output_dir`` is outside ``_artifacts/``, fall back to a sibling."""

    def test_minimax_out_of_tree_path_uses_sibling_spec(self, tmp_path: Path) -> None:
        out = tmp_path / "minirun"
        out.mkdir()
        spec = derive_spec_dir(out)
        assert spec.name == "minirun_spec"
        assert spec.parent == out.parent

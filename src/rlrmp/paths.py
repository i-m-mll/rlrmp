"""Runtime path-resolver helpers for the mirror-tree artifact layout.

The repo separates artifacts by ROLE (see CLAUDE.md §Experiment Artifacts):

- ``results/<exp>/`` — tracked specs and narratives.
- ``_artifacts/<exp>/`` — gitignored bulk outputs (mirrors ``results/``).

These helpers resolve the two sides of the mirror so scripts can refer to
experiment / run / figure identifiers rather than hard-coding directory strings.

All functions return absolute ``pathlib.Path`` instances derived from the
repository root (detected once at import time via ``__file__``).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

#: Absolute path to the repository root directory.
#: Derived from the location of this file: ``src/rlrmp/paths.py`` is two
#: levels below ``src/`` which is one level below the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spec_root() -> Path:
    """Return the ``results/`` directory (tracked specs and narratives)."""
    return REPO_ROOT / "results"


def _artifact_root() -> Path:
    """Return the ``_artifacts/`` directory (gitignored bulk outputs)."""
    return REPO_ROOT / "_artifacts"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_spec_dir(exp: str, run: str) -> Path:
    """Return the optional tracked sidecar directory for a training run.

    New run recipes use the flat ``run_spec_path(exp, run)`` convention:
    ``results/<exp>/runs/<run>.json``. This directory is for lightweight
    tracked sidecars that accrue after the recipe exists.

    Args:
        exp: Experiment slug (e.g. ``"part2_5"``).
        run: Run identifier (e.g. ``"baseline__standard_12k"``).

    Returns:
        Absolute path: ``<repo_root>/results/<exp>/runs/<run>/``.
    """
    return _spec_root() / exp / "runs" / run


def run_spec_sidecar_dir(exp: str, run: str) -> Path:
    """Return the optional tracked sidecar directory for ``exp/run``.

    This is a clearer alias for ``run_spec_dir`` for new code. The older helper
    name remains available for callers that predate the explicit sidecar name.
    """

    return run_spec_dir(exp, run)


def flat_run_spec_path(exp: str, run: str, *, repo_root: Path | None = None) -> Path:
    """Return the canonical flat run-recipe file for ``exp/run``.

    This always constructs ``results/<exp>/runs/<run>.json`` directly. Writers
    and readers use the same canonical recipe path.

    Args:
        exp: Experiment slug (e.g. ``"part2_5"``).
        run: Run identifier (e.g. ``"baseline__standard_12k"``).
        repo_root: Optional root override (mainly for tests).

    Returns:
        Absolute path: ``<repo_root>/results/<exp>/runs/<run>.json``.
    """

    root = REPO_ROOT if repo_root is None else repo_root
    return root / "results" / exp / "runs" / f"{run}.json"


def run_spec_path(
    exp: str,
    run: str,
    *,
    repo_root: Path | None = None,
    for_write: bool = False,
) -> Path:
    """Return the canonical flat tracked run-recipe file for ``exp/run``.

    ``for_write`` remains accepted for API compatibility, but readers and
    writers now resolve to the same flat path. Use ``run_spec_sidecar_dir`` for
    optional tracked files that belong to the run but are not the recipe.
    """

    return flat_run_spec_path(exp, run, repo_root=repo_root)


def run_artifact_dir(exp: str, run: str) -> Path:
    """Return the artifact directory for a training run.

    This directory is gitignored and holds bulk outputs such as ``.eqx``
    checkpoints, ``.npz`` arrays, training logs, and optimizer state.

    Args:
        exp: Experiment slug (e.g. ``"part2_5"``).
        run: Run identifier (e.g. ``"baseline__standard_12k"``).

    Returns:
        Absolute path: ``<repo_root>/_artifacts/<exp>/runs/<run>/``.
    """
    return _artifact_root() / exp / "runs" / run


def portable_repo_path(path: Path | str, *, repo_root: Path | None = None) -> str:
    """Return a non-absolute path string for portable spec records.

    Paths under the active checkout are made repo-relative. Canonical rlrmp
    mirror-tree paths under ``results/`` or ``_artifacts/`` are also normalized
    by their marker, which keeps authored specs stable across worktree roots.
    Other absolute paths are made relative to the checkout root instead of
    leaking machine-specific prefixes into binding-hashed spec payloads.
    """

    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()

    root = REPO_ROOT if repo_root is None else repo_root
    for base in (root, root.resolve(strict=False)):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            pass

    resolved = candidate.resolve(strict=False)
    try:
        return resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        pass

    for marker in ("results", "_artifacts"):
        if marker in candidate.parts:
            marker_index = candidate.parts.index(marker)
            return Path(*candidate.parts[marker_index:]).as_posix()

    return Path(os.path.relpath(candidate, root)).as_posix()


def resolve_run_artifact_path(artifact_dir: Path, *parts: str) -> Path:
    """Return a run artifact path across historical and post-run layouts.

    Older GRU materializers wrote files directly under
    ``_artifacts/<exp>/runs/<run>/``. The post-run sync wrapper now preserves
    provider output shape, with model/history/checkpoint payloads under an
    ``artifacts/`` child. Some legacy runs also contain a nested
    ``<run>/<file>`` payload. This resolver returns the first existing candidate
    while keeping the direct path as the default for clearer missing-file errors.
    """

    direct_path = artifact_dir.joinpath(*parts)
    candidates = (
        direct_path,
        artifact_dir / "artifacts" / Path(*parts),
        artifact_dir / artifact_dir.name / Path(*parts),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return direct_path


def figure_spec_dir(exp: str, fig: str) -> Path:
    """Return the spec directory for a figure.

    This directory is tracked in git and holds ``spec.json`` and optionally
    ``figure.json`` (if below the 2 MB threshold) and ``figure.png`` thumbnail.

    Args:
        exp: Experiment slug (e.g. ``"part2_5"``).
        fig: Figure identifier (e.g. ``"peak_velocity_by_sisu"``).

    Returns:
        Absolute path: ``<repo_root>/results/<exp>/figures/<fig>/``.
    """
    return _spec_root() / exp / "figures" / fig


def figure_artifact_dir(exp: str, fig: str) -> Path:
    """Return the artifact directory for a figure.

    This directory is gitignored and holds heavy renders such as full-DPI
    ``.png`` files, interactive ``.html`` exports, and animation ``.mp4`` files.

    Args:
        exp: Experiment slug (e.g. ``"part2_5"``).
        fig: Figure identifier (e.g. ``"peak_velocity_by_sisu"``).

    Returns:
        Absolute path: ``<repo_root>/_artifacts/<exp>/figures/<fig>/``.
    """
    return _artifact_root() / exp / "figures" / fig


def mkdir_p(path: Path) -> Path:
    """Create ``path`` and all parent directories if they do not exist.

    A convenience wrapper around ``Path.mkdir(parents=True, exist_ok=True)``
    that returns the path, allowing inline use:

    .. code-block:: python

        out_dir = mkdir_p(run_artifact_dir("part2_5", "baseline__standard_12k"))

    Args:
        path: Directory path to create.

    Returns:
        The same ``path`` after creation.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

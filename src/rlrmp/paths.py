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
    """Return the spec directory for a training run.

    This directory is tracked in git and holds lightweight spec files such as
    ``run.json`` and optional ``notes.md``.

    Args:
        exp: Experiment slug (e.g. ``"part2_5"``).
        run: Run identifier (e.g. ``"baseline__standard_12k"``).

    Returns:
        Absolute path: ``<repo_root>/results/<exp>/runs/<run>/``.
    """
    return _spec_root() / exp / "runs" / run


def run_spec_path(exp: str, run: str, *, repo_root: Path | None = None) -> Path:
    """Return the tracked run-spec file for ``exp/run``.

    New post-run artifacts use the flat ``results/<exp>/runs/<run>.json``
    convention. Older runs may still use ``results/<exp>/runs/<run>/run.json``.
    When neither path exists, return the flat path so new callers fail or write
    against the current convention.
    """

    root = REPO_ROOT if repo_root is None else repo_root
    flat_path = root / "results" / exp / "runs" / f"{run}.json"
    legacy_path = root / "results" / exp / "runs" / run / "run.json"
    if flat_path.exists():
        return flat_path
    if legacy_path.exists():
        return legacy_path
    return flat_path


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

"""Smoke tests for `rlrmp.viz.profile_grids`.

Guards the shared-y-axis policy for profile-comparison subplot grids
(Bug: 06f7faf refinement).
"""

from __future__ import annotations

import pytest

import rlrmp.viz as viz
from rlrmp.viz.profile_grids import profile_comparison_grid


def test_viz_public_surface_is_pinned():
    """Only the live shared profile grid helper is exported from ``rlrmp.viz``."""

    assert viz.__all__ == ["profile_comparison_grid"]
    assert viz.profile_comparison_grid is profile_comparison_grid
    assert not hasattr(viz, "visualize_loss_structure")


def test_default_shares_yaxes_all():
    """Default behaviour must set shared_yaxes='all' (plotly anchors every
    non-anchor yaxis to a common yaxis via the ``matches`` property)."""
    fig = profile_comparison_grid(n_panels=4, subplot_titles=["a", "b", "c", "d"])
    layout = fig.layout
    matches = []
    for i in range(1, 5):
        key = "yaxis" if i == 1 else f"yaxis{i}"
        ax = layout[key]
        matches.append(ax.matches)
    non_none = [m for m in matches if m is not None]
    # Shared y across n_panels=4 → at least 3 axes link to a common anchor.
    assert len(non_none) >= 3, f"shared_yaxes='all' should link >=3 axes; got {matches}"
    assert len(set(non_none)) == 1, (
        f"all linked axes should share one anchor; got {matches}"
    )


def test_can_disable_shared_yaxes():
    """shared_yaxes=False must not set the matches property."""
    fig = profile_comparison_grid(
        n_panels=3, subplot_titles=["a", "b", "c"], shared_yaxes=False
    )
    layout = fig.layout
    for i in range(2, 4):
        ax = layout[f"yaxis{i}"]
        assert ax.matches is None


def test_subplot_titles_length_validation():
    """Mismatched subplot_titles length must raise."""
    with pytest.raises(ValueError, match="subplot_titles length"):
        profile_comparison_grid(n_panels=3, subplot_titles=["a", "b"])


def test_multi_column_layout_derives_rows():
    """When cols > 1, rows defaults to ceil(n_panels / cols)."""
    fig = profile_comparison_grid(
        n_panels=5, subplot_titles=["a", "b", "c", "d", "e"], cols=2
    )
    # With 5 panels and 2 cols, we expect 3 rows. Plotly creates yaxis1..yaxis6
    # but only 5 are visibly used; matches='y' on the others is harmless.
    layout = fig.layout
    # Confirm we have at least 5 yaxis entries.
    assert hasattr(layout, "yaxis5")

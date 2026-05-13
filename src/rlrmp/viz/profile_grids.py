"""Shared layout helpers for profile-comparison subplot grids.

Profile-comparison figures (forward-velocity-per-cell, hold-drift-per-cell,
their per-replicate variants) render one subplot per condition with the same
x-axis semantics across panels. To make panels visually comparable they MUST
share a y-axis scale. This module centralises that policy so the analysis
scripts do not need to set ``shared_yaxes`` per call site.

Bug: 06f7faf — codified after the go-cue alignment fix landed and the
re-rendered profile figures revealed that each panel was on its own y-scale,
hiding cross-cell differences that the panel layout was meant to expose.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from plotly.subplots import make_subplots
from plotly.graph_objects import Figure


SharedYMode = Literal["all", "rows", "columns", False]


def profile_comparison_grid(
    n_panels: int,
    *,
    subplot_titles: Sequence[str] | None = None,
    rows: int | None = None,
    cols: int = 1,
    shared_xaxes: bool = True,
    shared_yaxes: SharedYMode = "all",
    vertical_spacing: float = 0.06,
    horizontal_spacing: float | None = None,
    **make_subplots_kwargs,
) -> Figure:
    """Build a profile-comparison subplot grid with shared y-axes by default.

    Use this helper for every multi-panel profile figure where the panels
    plot the same kind of quantity (forward velocity, hold drift, etc.) across
    different conditions and the reader is meant to compare panels by visual
    overlay. Shared y-axes are enforced at this layer so analysis scripts do
    not have to remember to set them.

    Args:
        n_panels: Total number of panels.
        subplot_titles: One title per panel. Length must equal ``n_panels``.
        rows: Override the row count. Defaults to ``n_panels`` (single-column
            stack), or to ``ceil(n_panels / cols)`` if ``cols > 1``.
        cols: Number of columns. Default 1 (single-column stack — matches the
            current forward-velocity / hold-drift layout).
        shared_xaxes: Forwarded to ``plotly.subplots.make_subplots``. Default
            ``True`` (the panels share time on the x-axis).
        shared_yaxes: Forwarded to ``plotly.subplots.make_subplots``.
            **Default ``"all"`` — every panel shares one y-scale.** Pass
            ``"rows"`` for shared-per-row, ``"columns"`` for shared-per-column,
            or ``False`` to disable.
        vertical_spacing: Forwarded to ``make_subplots``. Default 0.06.
        horizontal_spacing: Forwarded to ``make_subplots`` (``None`` falls
            back to plotly's default).
        **make_subplots_kwargs: Any other ``make_subplots`` keyword arguments
            (e.g. ``specs``, ``column_widths``).

    Returns:
        A ``plotly.graph_objects.Figure`` with the shared-y-axis policy applied.
    """
    if rows is None:
        rows = max(1, (n_panels + cols - 1) // cols)
    if subplot_titles is not None and len(subplot_titles) != n_panels:
        raise ValueError(
            f"subplot_titles length ({len(subplot_titles)}) must equal n_panels "
            f"({n_panels})"
        )
    kwargs: dict[str, Any] = dict(
        rows=rows,
        cols=cols,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
    )
    if subplot_titles is not None:
        kwargs["subplot_titles"] = list(subplot_titles)
    if horizontal_spacing is not None:
        kwargs["horizontal_spacing"] = horizontal_spacing
    kwargs.update(make_subplots_kwargs)
    return make_subplots(**kwargs)

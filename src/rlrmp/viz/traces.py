"""Reusable Plotly trace assembly primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from rlrmp.viz.colors import hex_to_rgba


def add_band_trace(
    fig: go.Figure,
    *,
    x: np.ndarray,
    mean: np.ndarray,
    spread: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    color: str,
    name: str,
    row: int | None = None,
    col: int | None = None,
    legendgroup: str | None = None,
    showlegend: bool = True,
    fill_alpha: float = 0.16,
    line_width: float = 2.4,
    line_dash: str | None = None,
    opacity: float | None = None,
    band_label: str | None = None,
) -> None:
    """Add a mean line and a symmetric or explicit uncertainty band."""

    if spread is not None:
        if lower is not None or upper is not None:
            raise ValueError("pass spread or lower/upper, not both")
        lower = mean - spread
        upper = mean + spread
    if lower is None or upper is None:
        raise ValueError("spread or both lower and upper are required")
    location = {} if row is None or col is None else {"row": row, "col": col}
    group = legendgroup or name
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, fill_alpha),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name=band_label or f"{name} band",
            legendgroup=group,
            showlegend=False,
        ),
        **location,
    )
    add_line(
        fig,
        x=x,
        y=mean,
        color=color,
        name=name,
        row=row,
        col=col,
        legendgroup=group,
        showlegend=showlegend,
        width=line_width,
        dash=line_dash,
        opacity=opacity,
    )


def add_line(
    fig: go.Figure,
    *,
    x: Any,
    y: Any,
    color: str,
    name: str,
    row: int | None = None,
    col: int | None = None,
    legendgroup: str | None = None,
    showlegend: bool = True,
    width: float = 2.0,
    dash: str | None = None,
    opacity: float | None = None,
) -> None:
    """Add one consistently styled line to a figure or subplot."""

    location = {} if row is None or col is None else {"row": row, "col": col}
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            legendgroup=legendgroup or name,
            showlegend=showlegend,
            line={"color": color, "dash": dash, "width": width},
            opacity=opacity,
        ),
        **location,
    )

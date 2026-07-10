"""Reusable Plotly trace assembly primitives."""

from __future__ import annotations

from collections.abc import Callable
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
    std: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    color: str,
    name: str,
    row: int | None = None,
    col: int | None = 1,
    legendgroup: str | None = None,
    showlegend: bool = True,
    fill_alpha: float = 0.16,
    line_width: float = 2.4,
    line_dash: str | None = None,
    opacity: float | None = None,
    band_label: str | None = None,
    band_fill_color: str | None = None,
    band_line_color: str = "rgba(0,0,0,0)",
    band_showlegend: bool = False,
    band_mode: str | None = None,
) -> None:
    """Add a mean line and a symmetric or explicit uncertainty band."""

    if std is not None:
        if spread is not None:
            raise ValueError("pass spread or std, not both")
        spread = std
    if spread is not None:
        if lower is not None or upper is not None:
            raise ValueError("pass spread or lower/upper, not both")
        lower = mean - spread
        upper = mean + spread
    if lower is None or upper is None:
        raise ValueError("spread or both lower and upper are required")
    location = {} if row is None or col is None else {"row": row, "col": col}
    group = legendgroup or name
    band_kwargs: dict[str, Any] = {}
    if band_mode is not None:
        band_kwargs["mode"] = band_mode
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=band_fill_color or hex_to_rgba(color, fill_alpha),
            line={"color": band_line_color},
            hoverinfo="skip",
            name=band_label or f"{name} band",
            legendgroup=group,
            showlegend=band_showlegend,
            **band_kwargs,
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


def add_reference_trace(
    fig: go.Figure,
    *,
    reference: Any,
    row: int,
    showlegend: bool = True,
    fallback_color: str = "#111827",
) -> None:
    """Add a reference object's forward-velocity mean and standard-deviation band."""

    add_band_trace(
        fig,
        x=np.asarray(reference.time_s),
        mean=np.asarray(reference.forward_velocity),
        spread=np.asarray(reference.forward_velocity_std),
        row=row,
        col=1,
        color=reference.line_color or fallback_color,
        name=reference.label,
        legendgroup=f"reference-{reference.observation_channel}",
        showlegend=showlegend,
        fill_alpha=0.08,
        line_width=2.2,
        line_dash=reference.line_dash,
        band_label=f"{reference.label} mean +/- 1 SD",
    )


def add_profile_line(
    fig: go.Figure,
    profile: np.ndarray,
    *,
    dt: float = 0.01,
    row: int,
    col: int,
    name: str,
    color: str,
    dash: str,
    showlegend: bool,
    width: float = 2.1,
) -> None:
    """Reduce an optional sample axis and add a time-indexed profile line."""

    line_profile = np.asarray(profile, dtype=np.float64)
    if line_profile.ndim == 2:
        line_profile = np.nanmean(line_profile, axis=0)
    if line_profile.ndim != 1:
        raise ValueError(f"expected a 1D profile line, got shape {line_profile.shape}")
    add_line(
        fig,
        x=np.arange(line_profile.shape[0], dtype=np.float64) * dt,
        y=line_profile,
        row=row,
        col=col,
        name=name,
        color=color,
        dash=dash,
        showlegend=showlegend,
        width=width,
    )


def add_sample_band(
    fig: go.Figure,
    samples: np.ndarray,
    *,
    reducer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    row: int,
    col: int,
    name: str,
    color: str,
    showlegend: bool,
    dt: float = 0.01,
    fill_alpha: float = 0.12,
) -> None:
    """Reduce sample profiles and add a central band plus mean line."""

    mean, low, high = reducer(samples)
    time = np.arange(mean.shape[0], dtype=np.float64) * dt
    fig.add_trace(
        go.Scatter(
            x=time,
            y=high,
            mode="lines",
            line={"color": "rgba(0,0,0,0)", "width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=low,
            mode="lines",
            fill="tonexty",
            fillcolor=hex_to_rgba(color, fill_alpha),
            line={"color": "rgba(0,0,0,0)", "width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=name,
        ),
        row=row,
        col=col,
    )
    add_line(
        fig,
        x=time,
        y=mean,
        row=row,
        col=col,
        name=name,
        color=color,
        dash="solid",
        showlegend=showlegend,
        width=2.1,
    )


def add_reduced_sample_trace(
    fig: go.Figure,
    samples: np.ndarray,
    *,
    reducer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    row: int,
    col: int,
    name: str,
    legendgroup: str,
    color: str,
    band_fill_color: str,
    dash: str,
    width: float,
    opacity: float,
    showlegend: bool,
    dt: float = 0.01,
    hovertemplate: str | None = None,
) -> None:
    """Add a reduced sample trace with an optional central interval."""

    mean, low, high = reducer(samples)
    time = np.arange(mean.shape[0], dtype=np.float64) * dt
    if samples.shape[0] > 1:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=high,
                mode="lines",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=low,
                mode="lines",
                fill="tonexty",
                fillcolor=band_fill_color,
                line={"color": "rgba(0,0,0,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )
    line_kwargs: dict[str, Any] = {}
    if hovertemplate is not None:
        line_kwargs.update(
            customdata=np.full(mean.shape, samples.shape[0]),
            hovertemplate=hovertemplate,
        )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=mean,
            mode="lines",
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            line={"color": color, "dash": dash, "width": width},
            opacity=opacity,
            **line_kwargs,
        ),
        row=row,
        col=col,
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

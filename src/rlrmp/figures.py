"""RLRMP declarative figure registrations and adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from feedbax.contracts.figures import FigurePiece, FigureSpec, FigureTemplate, SlotSpec
from feedbax.contracts.manifest import ArtifactRef, StrictModel
from feedbax.contracts.selection import ManifestPredicate
from feedbax.plot.constructors import (
    PanelContent,
    get_figure_constructor,
    register_figure_constructor,
    register_figure_piece,
    register_figure_template,
)

STANDARD_MATRIX_PAYLOAD_ROLE = "standard_matrix_payload"


class StandardMatrixFigureParams(StrictModel):
    """Layout defaults for standard-matrix custom figure adapters."""

    vertical_spacing: float = 0.04
    width: int = 780
    row_height: int = 420
    min_height: int = 420
    profile_key: str | None = None
    output: str | None = None
    title: str | None = None


class EffectorTrajectoryFigureParams(StrictModel):
    """Layout defaults for 2D effector trajectory figures."""

    width_base: int = 100
    width_per_panel: int = 300
    height: int = 400


def register_rlrmp_figure_surfaces(*, replace: bool = True) -> None:
    """Register RLRMP's declarative figure templates, pieces, and adapters."""
    register_figure_constructor(
        "rlrmp.standard_matrix_figure",
        tier="custom_figure",
        constructor=_standard_matrix_figure,
        params_model=StandardMatrixFigureParams,
        description="RLRMP standard-matrix figure adapter over Feedbax trace constructors.",
        replace=replace,
    )
    register_figure_constructor(
        "rlrmp.effector_trajectories_2d",
        tier="custom_figure",
        constructor=_effector_trajectories_2d_figure,
        params_model=EffectorTrajectoryFigureParams,
        description="RLRMP 2D effector trajectory figure adapter.",
        replace=replace,
    )
    register_figure_template(
        FigureTemplate(
            name="rlrmp.profile_comparison",
            description="Velocity or hold-drift profile comparison grid with shared y-axes.",
            assembler="rlrmp.standard_matrix_figure",
            assembler_params={
                "vertical_spacing": 0.04,
                "width": 780,
                "row_height": 420,
                "min_height": 420,
            },
            slots=[
                SlotSpec(
                    name="profiles",
                    constructor="feedbax.profile_band",
                    multiplicity="per_facet",
                ),
                SlotSpec(
                    name="baseline",
                    constructor="feedbax.profile_band",
                    required=False,
                    multiplicity="many",
                ),
            ],
            facet_by=["condition"],
            facet_target="panels",
            metadata={"ported_from": "rlrmp.viz.profile_comparison_grid"},
        ),
        replace=replace,
    )
    register_figure_template(
        FigureTemplate(
            name="rlrmp.effector_trajectories_2d",
            description="2D effector trajectories, one panel per variable.",
            assembler="rlrmp.effector_trajectories_2d",
            slots=[
                SlotSpec(
                    name="trajectories",
                    constructor="feedbax.trajectory_2d",
                    multiplicity="per_facet",
                ),
                SlotSpec(
                    name="endpoints",
                    constructor="feedbax.endpoint_markers",
                    required=False,
                    multiplicity="per_facet",
                ),
            ],
            facet_by=["variable"],
            facet_target="panels",
            metadata={"ported_from": "feedbax.analysis.effector"},
        ),
        replace=replace,
    )
    register_figure_template(
        FigureTemplate(
            name="rlrmp.loss_history",
            description="Training or validation loss history comparison.",
            assembler="rlrmp.standard_matrix_figure",
            metadata={"port_status": "registered for follow-up constructor split"},
        ),
        replace=replace,
    )
    register_figure_piece(
        FigurePiece(
            name="rlrmp.lqg_baseline_rollout",
            description="Analytical LQG baseline rollout trace for profile comparisons.",
            manifest_predicate=ManifestPredicate(
                manifest_kind="EvaluationRunManifest",
                metadata_equals={"rlrmp_piece": "lqg_baseline_rollout"},
            ),
            data_path="payload",
            label="LQG (analytical)",
            constructor="feedbax.profile_band",
            style={"color": "#111827", "line_dash": "dash"},
        ),
        replace=replace,
    )


def standard_matrix_profile_spec(
    *,
    name: str,
    output: str,
    profile_key: str,
    title: str,
    figure_routing: Mapping[str, Any] | None = None,
) -> FigureSpec:
    """Build a standard-matrix profile ``FigureSpec``."""
    return FigureSpec(
        name=name,
        template="rlrmp.profile_comparison",
        assembler_params={
            "output": output,
            "profile_key": profile_key,
            "title": title,
        },
        figure_routing=dict(figure_routing or {}),
        metadata={
            "output": output,
            "profile_key": profile_key,
            "title": title,
        },
    )


def standard_matrix_output_spec(
    *,
    name: str,
    output: str,
    title: str,
    figure_routing: Mapping[str, Any] | None = None,
) -> FigureSpec:
    """Build a non-profile standard-matrix ``FigureSpec``."""
    return FigureSpec(
        name=name,
        assembler="rlrmp.standard_matrix_figure",
        assembler_params={"output": output, "title": title},
        figure_routing=dict(figure_routing or {}),
        metadata={"output": output, "title": title},
    )


def standard_matrix_payload(cells: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON payload consumed by standard-matrix figure specs."""
    return {
        "schema_id": "rlrmp.figure_data.standard_matrix",
        "schema_version": "rlrmp.figure_data.standard_matrix.v1",
        "cells": _jsonable(cells),
        "params": _jsonable(params),
    }


def _standard_matrix_figure(context: Mapping[str, Any], params: StrictModel) -> go.Figure:
    p = StandardMatrixFigureParams.model_validate(params.model_dump())
    payload = _standard_matrix_payload_from_context(context)
    cells = payload["cells"]
    output = p.output or payload.get("params", {}).get("standard_matrix_output")
    if output in {"forward_velocity_profiles", "hold_drift_profiles"}:
        profile_key = p.profile_key or (
            "forward_velocity" if output == "forward_velocity_profiles" else "hold_drift"
        )
        return _profile_comparison_figure(
            cells,
            profile_key=profile_key,
            params=p,
            references=payload.get("references", []),
        )
    if output == "peak_velocity_distributions":
        return _peak_velocity_figure(cells, title=p.title)
    if output == "summary_metrics":
        return _summary_metrics_figure(cells, payload.get("params", {}), title=p.title)
    if output == "training_loss":
        return _training_loss_figure(cells, title=p.title)
    if output == "training_loss_per_term":
        return _training_loss_per_term_figure(cells, title=p.title)
    if output == "rmse_ratio_comparison":
        return _rmse_ratio_figure(cells, title=p.title)
    raise ValueError(f"Unknown standard-matrix figure output {output!r}")


def _profile_comparison_figure(
    cells: Sequence[Mapping[str, Any]],
    *,
    profile_key: str,
    params: StandardMatrixFigureParams,
    references: Sequence[Mapping[str, Any]] = (),
) -> go.Figure:
    profile_constructor = get_figure_constructor("feedbax.profile_band", tier="trace")
    panel_constructor = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    figure_constructor = get_figure_constructor("feedbax.grid_figure", tier="figure")
    panels = []
    for index, cell in enumerate(cells, start=1):
        traces = []
        for series in _profile_series(cell, profile_key, references):
            profile = _mapping(series.get("profile", {}))
            x = _series(profile.get("time", cell.get("time")), default_len=_profile_len(profile))
            mean = _series(profile.get("mean", profile.get("value", [])))
            if not mean:
                continue
            data = {
                "x": x,
                "y": [mean],
                "mean": mean,
                "upper": _series(profile.get("upper")),
                "lower": _series(profile.get("lower")),
                "label": str(series.get("label") or cell["display_name"]),
                "color": _plotly_rgb(series.get("color") or cell.get("color")),
            }
            traces.extend(profile_constructor.callable(data, profile_constructor.params({})))
        panels.append(
            PanelContent(
                name=str(cell["run_id"]),
                traces=tuple(traces),
                title=str(cell["display_name"]),
                row=index,
                col=1,
                axes_labels={"x": "Time", "y": profile_key.replace("_", " ").title()},
            )
        )
    panel_params = panel_constructor.params(
        {
            "shared_xaxes": True,
            "shared_yaxes": "all",
            "vertical_spacing": _subplot_vertical_spacing(len(panels), params.vertical_spacing),
        }
    )
    fig = panel_constructor.callable(panels or [PanelContent(name=profile_key)], panel_params)
    return figure_constructor.callable(
        fig,
        panels,
        figure_constructor.params(
            {
                "width": params.width,
                "height": max(params.min_height, params.row_height * max(1, len(panels))),
                "title": params.title or profile_key.replace("_", " ").title(),
            }
        ),
    )


def _effector_trajectories_2d_figure(context: Mapping[str, Any], params: StrictModel) -> go.Figure:
    p = EffectorTrajectoryFigureParams.model_validate(params.model_dump())
    payload = _context_params(context)
    variables = payload.get("variables") or payload.get("trajectories") or {}
    if not isinstance(variables, Mapping):
        raise ValueError("effector trajectory figure metadata requires a mapping of variables")
    trajectory_constructor = get_figure_constructor("feedbax.trajectory_2d", tier="trace")
    endpoint_constructor = get_figure_constructor("feedbax.endpoint_markers", tier="trace")
    panel_constructor = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    figure_constructor = get_figure_constructor("feedbax.trajectories_2d_row", tier="figure")
    panels = []
    for index, (name, value) in enumerate(variables.items(), start=1):
        variable = _mapping(value)
        traces = list(
            trajectory_constructor.callable(
                {"trajectories": variable.get("trajectories", value), "label": name},
                trajectory_constructor.params(variable.get("style", {})),
            )
        )
        if variable.get("endpoints") is not None:
            traces.extend(
                endpoint_constructor.callable(
                    {"endpoints": variable["endpoints"]},
                    endpoint_constructor.params(variable.get("endpoint_style", {})),
                )
            )
        panels.append(
            PanelContent(
                name=str(name),
                traces=tuple(traces),
                title=str(name),
                row=1,
                col=index,
                axes_labels={"x": "x", "y": "y"},
            )
        )
    fig = panel_constructor.callable(
        panels or [PanelContent(name="trajectory")],
        panel_constructor.params({"shared_xaxes": False, "shared_yaxes": False}),
    )
    return figure_constructor.callable(
        fig,
        panels,
        figure_constructor.params(
            {
                "width_base": p.width_base,
                "width_per_panel": p.width_per_panel,
                "height": p.height,
            }
        ),
    )


def _standard_matrix_payload_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    for manifest in _context_manifests(context):
        for artifact in manifest.get("artifacts", []):
            if artifact.get("role") != STANDARD_MATRIX_PAYLOAD_ROLE:
                continue
            uri = artifact.get("uri")
            if uri is None:
                continue
            return json.loads(Path(uri).read_text(encoding="utf-8"))
    payload = _context_params(context)
    if "cells" in payload:
        return payload
    raise ValueError("No standard-matrix payload artifact found in figure context")


def _profile_series(
    cell: Mapping[str, Any],
    profile_key: str,
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    profile = _mapping(cell.get(profile_key, {}))
    if isinstance(profile.get("series"), Sequence) and not isinstance(
        profile.get("series"),
        (str, bytes, bytearray),
    ):
        output = [dict(_mapping(series)) for series in profile["series"]]
    else:
        output = [
            {
                "label": str(cell["display_name"]),
                "color": cell.get("color"),
                "profile": profile,
            }
        ]
    for reference in references:
        output.append(dict(reference))
    return output


def _context_manifests(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    item = context.get("manifests")
    payload = getattr(item, "payload", None)
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict):
        return [payload]
    item = context.get("manifest")
    payload = getattr(item, "payload", None)
    return [payload] if isinstance(payload, dict) else []


def _context_params(context: Mapping[str, Any]) -> dict[str, Any]:
    item = context.get("params")
    payload = getattr(item, "payload", None)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _peak_velocity_figure(cells: Sequence[Mapping[str, Any]], *, title: str | None) -> go.Figure:
    fig = go.Figure()
    for cell in cells:
        values = _series(cell.get("peak_velocity", []))
        fig.add_trace(
            go.Box(
                y=values,
                name=str(cell["display_name"]),
                marker={"color": cell.get("color")} if cell.get("color") else None,
            )
        )
    fig.update_layout(template="plotly_white", title=title or "Peak Velocity Distributions")
    fig.update_yaxes(title_text="Peak velocity")
    return fig


def _summary_metrics_figure(
    cells: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    *,
    title: str | None,
) -> go.Figure:
    metric_order = list(params.get("metric_order") or _observed_metric_order(cells))
    fig = go.Figure()
    for metric in metric_order:
        values = [_metric_value(cell, metric) for cell in cells]
        fig.add_trace(
            go.Bar(
                x=[str(cell["display_name"]) for cell in cells],
                y=values,
                name=metric.replace("_", " ").title(),
            )
        )
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title=title or "Summary Metrics",
        xaxis_title="Cell",
    )
    return fig


def _training_loss_figure(cells: Sequence[Mapping[str, Any]], *, title: str | None) -> go.Figure:
    fig = go.Figure()
    for cell in cells:
        loss = _mapping(cell.get("training_loss", {}))
        values = _series(loss.get("loss", loss.get("value", [])))
        steps = _series(loss.get("step", loss.get("steps")), default_len=len(values))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines",
                name=str(cell["display_name"]),
                line={"color": cell.get("color")} if cell.get("color") else None,
            )
        )
    fig.update_layout(template="plotly_white", title=title or "Training Loss", xaxis_title="Step")
    fig.update_yaxes(title_text="Loss")
    return fig


def _training_loss_per_term_figure(
    cells: Sequence[Mapping[str, Any]],
    *,
    title: str | None,
) -> go.Figure:
    fig = go.Figure()
    for cell in cells:
        loss = _mapping(cell.get("training_loss_per_term", {}))
        terms = _mapping(loss.get("terms", {}))
        for term, raw_values in terms.items():
            values = _series(raw_values)
            steps = _series(loss.get("step", loss.get("steps")), default_len=len(values))
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    mode="lines",
                    name=f"{cell['display_name']} {term}",
                )
            )
    fig.update_layout(
        template="plotly_white",
        title=title or "Training Loss Per Term",
        xaxis_title="Step",
    )
    fig.update_yaxes(title_text="Loss")
    return fig


def _rmse_ratio_figure(cells: Sequence[Mapping[str, Any]], *, title: str | None) -> go.Figure:
    fig = go.Figure()
    plotted = False
    for cell in cells:
        value = cell.get("rmse_ratio", _metric_value(cell, "rmse_ratio"))
        if value is None:
            continue
        plotted = True
        fig.add_trace(
            go.Bar(
                x=[str(cell["display_name"])],
                y=[float(value)],
                name=str(cell["display_name"]),
                marker={"color": cell.get("color")} if cell.get("color") else None,
            )
        )
    if not plotted:
        fig.add_annotation(text="No matched-control RMSE ratio data", showarrow=False)
    fig.update_layout(template="plotly_white", title=title or "RMSE Ratio Comparison")
    fig.update_yaxes(title_text="RMSE ratio")
    return fig


def _observed_metric_order(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for cell in cells:
        for metric in _mapping(cell.get("summary_metrics", {})):
            if metric not in seen:
                seen.append(str(metric))
    return seen


def _metric_value(cell: Mapping[str, Any], metric: str) -> float | None:
    metrics = _mapping(cell.get("summary_metrics", {}))
    if metric in metrics:
        return float(metrics[metric])
    if metric in cell and isinstance(cell[metric], int | float):
        return float(cell[metric])
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _series(value: Any, *, default_len: int | None = None) -> list[float]:
    if value is None:
        if default_len is None:
            return []
        return [float(i) for i in range(default_len)]
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return [float(array)]
    return [float(item) for item in array.reshape(-1)]


def _plotly_rgb(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("#") or len(value) != 7:
        return value if isinstance(value, str) else None
    red = int(value[1:3], 16)
    green = int(value[3:5], 16)
    blue = int(value[5:7], 16)
    return f"rgb({red},{green},{blue})"


def _profile_len(profile: Mapping[str, Any]) -> int | None:
    for key in ("mean", "value", "lower", "upper"):
        if key in profile:
            return len(_series(profile[key]))
    return None


def _subplot_vertical_spacing(n_rows: int, preferred: float) -> float:
    if n_rows <= 1:
        return preferred
    max_spacing = 1.0 / float(n_rows - 1)
    return min(preferred, max_spacing * 0.8)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(child) for child in value]
    return value


def figure_render_path(manifest_artifacts: Sequence[ArtifactRef], *, preferred_suffix: str) -> Path:
    """Return the custody render path with a preferred suffix."""
    for artifact in manifest_artifacts:
        if artifact.role == "figure_render" and artifact.uri and artifact.uri.endswith(preferred_suffix):
            return Path(artifact.uri)
    for artifact in manifest_artifacts:
        if artifact.role == "figure_render" and artifact.uri:
            return Path(artifact.uri)
    raise ValueError("Figure manifest did not record a render artifact")

"""RLRMP declarative figure registrations and payload adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from feedbax.contracts.figures import (
    FigurePiece,
    FigureSpec,
    FigureTemplate,
    SlotSpec,
    TraceBinding,
)
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
STANDARD_MATRIX_PAYLOAD_SCHEMA_ID = "rlrmp.figure_data.standard_matrix"
STANDARD_MATRIX_PAYLOAD_SCHEMA_VERSION = "rlrmp.figure_data.standard_matrix.v1"


class FigureTraceParams(StrictModel):
    """Shared defaults for RLRMP trace-group constructors."""

    opacity: float = 1.0


class RlrmpGridParams(StrictModel):
    """Combined panel/layout params accepted by the native figure executor."""

    panel_constructor: str = "rlrmp.comparison_grid"
    shared_xaxes: bool | str = False
    shared_yaxes: bool | str = False
    horizontal_spacing: float | None = None
    vertical_spacing: float | None = None
    width: int | None = None
    height: int | None = None
    title: str | None = None
    legend_tracegroupgap: int | None = None


def register_rlrmp_figure_surfaces(*, replace: bool = True) -> None:
    """Register RLRMP templates, pieces, and compositional trace constructors."""
    register_figure_constructor(
        "rlrmp.comparison_grid",
        tier="panel",
        constructor=_comparison_grid,
        params_model=RlrmpGridParams,
        description="Comparison grid accepting the executor's combined native layout params.",
        version="v1",
        replace=replace,
    )
    register_figure_constructor(
        "rlrmp.profile_grid",
        tier="panel",
        constructor=_profile_grid,
        params_model=RlrmpGridParams,
        description="Profile grid with aligned time and shared value axes.",
        version="v1",
        replace=replace,
    )
    for key, constructor, description in (
        (
            "rlrmp.profile_series",
            _profile_series_traces,
            "One or more profile bands from a standard-matrix cell.",
        ),
        ("rlrmp.box_values", _box_values, "One distribution box trace."),
        ("rlrmp.bar_values", _bar_values, "One categorical bar trace."),
        ("rlrmp.line_series", _line_series, "One or more line-history traces."),
        (
            "rlrmp.loss_history_curves",
            _loss_history_curves,
            "Feedbax loss-history traces carried through declarative custody.",
        ),
        (
            "rlrmp.response_norm_bands",
            _response_norm_bands,
            "Mean and SEM response-norm bands with optional max and LQG curves.",
        ),
        (
            "rlrmp.response_norm_bars",
            _response_norm_bars,
            "Response-norm comparison bars derived from payload-owned model rows.",
        ),
    ):
        register_figure_constructor(
            key,
            tier="trace",
            constructor=constructor,
            params_model=FigureTraceParams,
            description=description,
            version="v1",
            replace=replace,
        )

    for template in (
        FigureTemplate(
            name="rlrmp.profile_comparison",
            description="Velocity or hold-drift profile comparison with shared y-axes.",
            assembler="feedbax.grid_figure",
            assembler_params={"panel_constructor": "rlrmp.profile_grid"},
            slots=[
                SlotSpec(
                    name="profiles",
                    constructor="rlrmp.profile_series",
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
        FigureTemplate(
            name="rlrmp.distribution_comparison",
            description="One distribution panel per standard-matrix condition.",
            assembler="feedbax.grid_figure",
            assembler_params={"panel_constructor": "rlrmp.comparison_grid"},
            slots=[
                SlotSpec(
                    name="values",
                    constructor="rlrmp.box_values",
                    multiplicity="per_facet",
                )
            ],
            facet_by=["condition"],
            facet_target="panels",
        ),
        FigureTemplate(
            name="rlrmp.metric_comparison",
            description="One categorical bar panel per standard-matrix metric.",
            assembler="feedbax.grid_figure",
            assembler_params={"panel_constructor": "rlrmp.comparison_grid"},
            slots=[
                SlotSpec(
                    name="values",
                    constructor="rlrmp.bar_values",
                    multiplicity="per_facet",
                )
            ],
            facet_by=["metric"],
            facet_target="panels",
        ),
        FigureTemplate(
            name="rlrmp.history_comparison",
            description="One line-history panel per standard-matrix condition.",
            assembler="feedbax.grid_figure",
            assembler_params={"panel_constructor": "rlrmp.comparison_grid"},
            slots=[
                SlotSpec(
                    name="series",
                    constructor="rlrmp.line_series",
                    multiplicity="per_facet",
                )
            ],
            facet_by=["condition"],
            facet_target="panels",
        ),
        FigureTemplate(
            name="rlrmp.effector_trajectories_2d",
            description="2D effector trajectories, one panel per variable.",
            assembler="feedbax.trajectories_2d_row",
            assembler_params={"panel_constructor": "rlrmp.comparison_grid"},
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
        FigureTemplate(
            name="rlrmp.response_norm_comparison",
            description=(
                "Response-norm comparison with intrinsic metric/condition facets and "
                "payload-bound model/row collections."
            ),
            assembler="feedbax.grid_figure",
            assembler_params={"panel_constructor": "rlrmp.profile_grid"},
            slots=[
                SlotSpec(
                    name="norm_bands",
                    constructor="rlrmp.response_norm_bands",
                    multiplicity="per_facet",
                ),
                SlotSpec(
                    name="norm_bars",
                    constructor="rlrmp.response_norm_bars",
                    required=False,
                    multiplicity="per_facet",
                ),
            ],
            facet_by=["metric", "condition_class"],
            facet_target="panels",
            metadata={
                "intrinsic_facets": ["metric", "condition_class"],
                "data_bound_collections": ["model", "comparison_row"],
            },
        ),
        FigureTemplate(
            name="rlrmp.loss_history",
            description="Training or validation loss-history comparison.",
            assembler="feedbax.grid_figure",
            assembler_params={"panel_constructor": "rlrmp.comparison_grid"},
            slots=[
                SlotSpec(
                    name="curves",
                    constructor="rlrmp.loss_history_curves",
                    multiplicity="per_facet",
                )
            ],
            facet_by=["context"],
            facet_target="panels",
        ),
    ):
        register_figure_template(template, replace=replace)

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
            style={"color": "rgb(17,24,39)", "line_dash": "dash"},
        ),
        replace=replace,
    )

    # Imported here to keep the core template module independent of tracked
    # stock payload definitions during module initialization.
    from rlrmp.profile_payloads import register_profile_stock_pieces

    register_profile_stock_pieces(replace=replace)


def standard_matrix_profile_spec(
    *,
    name: str,
    output: str,
    profile_key: str,
    title: str,
    figure_routing: Mapping[str, Any] | None = None,
    payload_item: str = "manifest",
    payload_path: str = "metadata.figure_payload",
) -> FigureSpec:
    """Build a native slot/facet profile ``FigureSpec``."""
    prefix = f"{payload_path}." if payload_path else ""
    facet_path = f"{prefix}facets.{output}"
    baseline_path = f"{prefix}baseline"
    return FigureSpec(
        name=name,
        template="rlrmp.profile_comparison",
        slot_bindings={
            "profiles": TraceBinding(
                name="profiles",
                constructor="rlrmp.profile_series",
                required=True,
                data={
                    "cell": {"item": "facet_values", "path": "condition"},
                    "profile_key": profile_key,
                },
            ),
            "baseline": TraceBinding(
                name="lqg-baseline",
                constructor="feedbax.profile_band",
                piece="rlrmp.lqg_baseline_rollout",
                required=False,
                data={
                    "x": {"item": payload_item, "path": f"{baseline_path}.profile.time"},
                    "y": {"item": payload_item, "path": f"{baseline_path}.profile.y"},
                    "mean": {"item": payload_item, "path": f"{baseline_path}.profile.mean"},
                    "upper": {"item": payload_item, "path": f"{baseline_path}.profile.upper"},
                    "lower": {"item": payload_item, "path": f"{baseline_path}.profile.lower"},
                    "label": {"item": payload_item, "path": f"{baseline_path}.label"},
                    "color": {"item": payload_item, "path": f"{baseline_path}.color"},
                },
            ),
        },
        panels=[
            {
                "name": "profile",
                "title": {"item": "condition"},
                "axes_labels": {"x": "Time", "y": title},
            }
        ],
        facet_bindings={
            "condition": {"item": payload_item, "path": facet_path},
        },
        figure_routing=dict(figure_routing or {}),
        metadata={"output": output, "profile_key": profile_key, "title": title},
    )


def standard_matrix_output_spec(
    *,
    name: str,
    output: str,
    title: str,
    figure_routing: Mapping[str, Any] | None = None,
) -> FigureSpec:
    """Build a native non-profile standard-matrix ``FigureSpec``."""
    definitions = {
        "peak_velocity_distributions": (
            "rlrmp.distribution_comparison",
            "condition",
            "values",
            "rlrmp.box_values",
        ),
        "summary_metrics": (
            "rlrmp.metric_comparison",
            "metric",
            "values",
            "rlrmp.bar_values",
        ),
        "rmse_ratio_comparison": (
            "rlrmp.metric_comparison",
            "metric",
            "values",
            "rlrmp.bar_values",
        ),
        "training_loss": (
            "rlrmp.history_comparison",
            "condition",
            "series",
            "rlrmp.line_series",
        ),
        "training_loss_per_term": (
            "rlrmp.history_comparison",
            "condition",
            "series",
            "rlrmp.line_series",
        ),
    }
    try:
        template, facet, slot, constructor = definitions[output]
    except KeyError as exc:
        raise ValueError(f"Unknown standard-matrix figure output {output!r}") from exc
    payload_path = f"metadata.figure_payload.facets.{output}"
    return FigureSpec(
        name=name,
        template=template,
        slot_bindings={
            slot: TraceBinding(
                name=slot,
                constructor=constructor,
                required=True,
                data={"payload": {"item": "facet_values", "path": facet}},
            )
        },
        panels=[{"name": "panel", "title": {"item": facet}}],
        facet_bindings={facet: {"item": "manifest", "path": payload_path}},
        figure_routing=dict(figure_routing or {}),
        metadata={"output": output, "title": title},
    )


def effector_trajectory_spec(
    *,
    name: str,
    variables: Mapping[str, Any],
    figure_routing: Mapping[str, Any] | None = None,
) -> FigureSpec:
    """Build a native slot/facet effector-trajectory spec."""
    return FigureSpec(
        name=name,
        template="rlrmp.effector_trajectories_2d",
        slot_bindings={
            "trajectories": TraceBinding(
                name="trajectories",
                constructor="feedbax.trajectory_2d",
                data={
                    "trajectories": {
                        "item": "facet_values",
                        "path": "variable.trajectories",
                    },
                    "label": {"item": "variable"},
                },
            ),
            "endpoints": TraceBinding(
                name="endpoints",
                constructor="feedbax.endpoint_markers",
                required=False,
                data={
                    "trajectories": {
                        "item": "facet_values",
                        "path": "variable.trajectories",
                    }
                },
            ),
        },
        panels=[
            {
                "name": "trajectory",
                "title": {"item": "variable"},
                "axes_labels": {"x": "x", "y": "y"},
            }
        ],
        facet_bindings={"variable": {"item": "params", "path": "variables"}},
        figure_routing=dict(figure_routing or {}),
        metadata={"variables": _jsonable(variables)},
    )


def loss_history_spec(
    *,
    name: str,
    context: str,
    traces: Sequence[Mapping[str, Any]],
    figure_routing: Mapping[str, Any] | None = None,
) -> FigureSpec:
    """Build a functional native loss-history spec from Plotly trace records."""
    return FigureSpec(
        name=name,
        template="rlrmp.loss_history",
        slot_bindings={
            "curves": TraceBinding(
                name="loss-curves",
                constructor="rlrmp.loss_history_curves",
                data={"payload": {"item": "facet_values", "path": "context"}},
            )
        },
        panels=[{"name": "loss", "title": {"item": "context"}}],
        facet_bindings={"context": {"item": "params", "path": "contexts"}},
        figure_routing=dict(figure_routing or {}),
        metadata={
            "schema_id": "rlrmp.figure_data.loss_history",
            "schema_version": "rlrmp.figure_data.loss_history.v1",
            "contexts": {context: {"traces": _jsonable(traces)}},
        },
    )


def response_norm_comparison_spec(
    *,
    name: str,
    figure_routing: Mapping[str, Any] | None = None,
) -> FigureSpec:
    """Build a response-norm figure whose panel count follows intrinsic facets."""
    return FigureSpec(
        name=name,
        template="rlrmp.response_norm_comparison",
        slot_bindings={
            "norm_bands": TraceBinding(
                name="response-norm-bands",
                constructor="rlrmp.response_norm_bands",
                data={
                    "payload": {
                        "item": "manifest",
                        "path": "metadata.figure_payload",
                    },
                    "metric": {"item": "metric"},
                    "condition_class": {"item": "condition_class"},
                },
            ),
            "norm_bars": TraceBinding(
                name="response-norm-bars",
                constructor="rlrmp.response_norm_bars",
                required=False,
                data={
                    "payload": {
                        "item": "manifest",
                        "path": "metadata.figure_payload",
                    },
                    "metric": {"item": "metric"},
                    "condition_class": {"item": "condition_class"},
                },
            ),
        },
        panels=[
            {
                "name": "response_norm",
                "title": {"item": "metric"},
                "axes_labels": {"x": "Time (s)", "y": "Response norm"},
            }
        ],
        facet_bindings={
            "metric": {
                "item": "manifest",
                "path": "metadata.figure_payload.intrinsic_axes.metric",
            },
            "condition_class": {
                "item": "manifest",
                "path": "metadata.figure_payload.intrinsic_axes.condition_class",
            },
        },
        figure_routing=dict(figure_routing or {}),
        metadata={
            "schema_id": "rlrmp.figure_data.response_norm_comparison",
            "schema_version": "rlrmp.figure_data.response_norm_comparison.v1",
        },
    )


def standard_matrix_payload(
    cells: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Return schema-bearing facet payloads consumed by standard-matrix specs."""
    normalized_cells = [dict(_mapping(cell)) for cell in cells]
    cell_map = {str(cell["run_id"]): _jsonable(cell) for cell in normalized_cells}
    metric_order = list(params.get("metric_order") or _observed_metric_order(normalized_cells))
    references = params.get("references", ())
    baseline = params.get("baseline")
    if baseline is None and isinstance(references, Sequence) and references:
        baseline = references[0]
    baseline_payload = _normalize_baseline(baseline)
    return {
        "schema_id": STANDARD_MATRIX_PAYLOAD_SCHEMA_ID,
        "schema_version": STANDARD_MATRIX_PAYLOAD_SCHEMA_VERSION,
        "cells": _jsonable(normalized_cells),
        "params": _jsonable(params),
        "baseline": baseline_payload,
        "facets": {
            "forward_velocity_profiles": cell_map,
            "hold_drift_profiles": cell_map,
            "peak_velocity_distributions": _peak_velocity_facets(normalized_cells),
            "summary_metrics": _summary_metric_facets(normalized_cells, metric_order),
            "rmse_ratio_comparison": _rmse_ratio_facets(normalized_cells),
            "training_loss": _history_facets(normalized_cells, per_term=False),
            "training_loss_per_term": _history_facets(normalized_cells, per_term=True),
        },
    }


def _profile_series_traces(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    _ = FigureTraceParams.model_validate(params.model_dump())
    cell = _mapping(data.get("cell"))
    profile_key = str(data.get("profile_key", "forward_velocity"))
    profile_constructor = get_figure_constructor("feedbax.profile_band", tier="trace")
    traces: list[Any] = []
    for series in _profile_series(cell, profile_key):
        profile = _mapping(series.get("profile"))
        mean = _series(profile.get("mean", profile.get("value")))
        if not mean:
            continue
        traces.extend(
            profile_constructor.callable(
                {
                    "x": _series(profile.get("time"), default_len=len(mean)),
                    "y": [mean],
                    "mean": mean,
                    "upper": _series(profile.get("upper")) or mean,
                    "lower": _series(profile.get("lower")) or mean,
                    "label": str(series.get("label") or cell.get("display_name", "Profile")),
                    "color": _plotly_rgb(series.get("color") or cell.get("color")),
                },
                profile_constructor.params(),
            )
        )
    return traces


def _comparison_grid(
    panels: Sequence[PanelContent],
    params: StrictModel,
) -> go.Figure:
    p = RlrmpGridParams.model_validate(params.model_dump())
    constructor = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    return constructor.callable(
        panels,
        constructor.params(
            {
                "shared_xaxes": p.shared_xaxes,
                "shared_yaxes": p.shared_yaxes,
                "horizontal_spacing": p.horizontal_spacing,
                "vertical_spacing": p.vertical_spacing,
            }
        ),
    )


def _profile_grid(
    panels: Sequence[PanelContent],
    params: StrictModel,
) -> go.Figure:
    p = RlrmpGridParams.model_validate(params.model_dump())
    constructor = get_figure_constructor("feedbax.comparison_grid", tier="panel")
    return constructor.callable(
        panels,
        constructor.params(
            {
                "shared_xaxes": True,
                "shared_yaxes": "all",
                "horizontal_spacing": p.horizontal_spacing,
                "vertical_spacing": p.vertical_spacing,
            }
        ),
    )


def _box_values(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = FigureTraceParams.model_validate(params.model_dump())
    payload = _mapping(data.get("payload"))
    return [
        go.Box(
            y=_series(payload.get("values")),
            name=str(payload.get("label", "Values")),
            marker={"color": payload.get("color")} if payload.get("color") else None,
            opacity=p.opacity,
        )
    ]


def _bar_values(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = FigureTraceParams.model_validate(params.model_dump())
    payload = _mapping(data.get("payload"))
    return [
        go.Bar(
            x=list(payload.get("labels", [])),
            y=_series(payload.get("values")),
            name=str(payload.get("label", "Values")),
            opacity=p.opacity,
        )
    ]


def _line_series(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = FigureTraceParams.model_validate(params.model_dump())
    payload = _mapping(data.get("payload"))
    series = _mapping(payload.get("series"))
    traces = []
    for label, values in series.items():
        y = _series(values)
        traces.append(
            go.Scatter(
                x=_series(payload.get("x"), default_len=len(y)),
                y=y,
                mode="lines",
                name=str(label),
                opacity=p.opacity,
                line={"color": payload.get("color")} if payload.get("color") else None,
            )
        )
    return traces


def _loss_history_curves(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = FigureTraceParams.model_validate(params.model_dump())
    payload = _mapping(data.get("payload"))
    figure = go.Figure(data=list(payload.get("traces", [])))
    for trace in figure.data:
        trace.opacity = p.opacity
    return list(figure.data)


def _response_norm_bands(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = FigureTraceParams.model_validate(params.model_dump())
    payload = _response_norm_facet(data)
    traces: list[Any] = []
    for curve in payload.get("curves", ()):
        curve = _mapping(curve)
        mean = _series(curve.get("mean"))
        if not mean:
            continue
        time = _series(curve.get("time"), default_len=len(mean))
        sem = _series(curve.get("sem"))
        color = curve.get("color") or "rgb(31,119,180)"
        label = str(curve.get("label", curve.get("model_id", "Model")))
        if sem and len(sem) == len(mean):
            upper = [value + error for value, error in zip(mean, sem, strict=True)]
            lower = [value - error for value, error in zip(mean, sem, strict=True)]
            traces.extend(
                get_figure_constructor("feedbax.profile_band", tier="trace").callable(
                    {
                        "x": time,
                        "y": [mean],
                        "mean": mean,
                        "upper": upper,
                        "lower": lower,
                        "label": label,
                        "color": color,
                    },
                    get_figure_constructor("feedbax.profile_band", tier="trace").params(),
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=time,
                    y=mean,
                    mode="lines",
                    name=label,
                    opacity=p.opacity,
                    line={
                        "color": color,
                        "dash": "dash" if curve.get("is_lqg") else "solid",
                    },
                )
            )
        maximum = _series(curve.get("max"))
        if maximum:
            traces.append(
                go.Scatter(
                    x=time,
                    y=maximum,
                    mode="lines",
                    name=f"{label} max",
                    opacity=p.opacity,
                    line={"color": color, "dash": "dot"},
                )
            )
    return traces


def _response_norm_bars(data: Mapping[str, Any], params: StrictModel) -> Sequence[Any]:
    p = FigureTraceParams.model_validate(params.model_dump())
    payload = _response_norm_facet(data)
    labels: list[str] = []
    values: list[float] = []
    for curve in payload.get("curves", ()):
        curve = _mapping(curve)
        mean = _series(curve.get("mean"))
        if mean:
            labels.append(str(curve.get("label", curve.get("model_id", "Model"))))
            values.append(max(mean))
    return [go.Bar(x=labels, y=values, name="Peak mean norm", opacity=p.opacity)] if labels else []


def _response_norm_facet(data: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _mapping(data.get("payload"))
    metric = str(data.get("metric", ""))
    condition_class = str(data.get("condition_class", ""))
    return _mapping(_mapping(_mapping(payload.get("facets")).get(metric)).get(condition_class))


def _profile_series(cell: Mapping[str, Any], profile_key: str) -> list[dict[str, Any]]:
    profile = _mapping(cell.get(profile_key))
    raw_series = profile.get("series")
    if isinstance(raw_series, Sequence) and not isinstance(raw_series, (str, bytes)):
        return [dict(_mapping(series)) for series in raw_series]
    return [
        {
            "label": str(cell.get("display_name", cell.get("run_id", "Profile"))),
            "color": cell.get("color"),
            "profile": profile,
        }
    ]


def _peak_velocity_facets(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(cell["run_id"]): {
            "label": str(cell.get("display_name", cell["run_id"])),
            "color": cell.get("color"),
            "values": cell.get(
                "peak_velocity",
                _mapping(cell.get("summary_metrics")).get("peak_velocity", []),
            ),
        }
        for cell in cells
    }


def _summary_metric_facets(
    cells: Sequence[Mapping[str, Any]],
    metric_order: Sequence[str],
) -> dict[str, Any]:
    facets = {}
    for metric in metric_order:
        entries = [
            (str(cell.get("display_name", cell["run_id"])), _metric_value(cell, metric))
            for cell in cells
        ]
        entries = [(label, value) for label, value in entries if value is not None]
        facets[str(metric)] = {
            "label": str(metric).replace("_", " ").title(),
            "labels": [label for label, _value in entries],
            "values": [value for _label, value in entries],
        }
    return facets


def _rmse_ratio_facets(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = [
        (str(cell.get("display_name", cell["run_id"])), _metric_value(cell, "rmse_ratio"))
        for cell in cells
    ]
    entries = [(label, value) for label, value in entries if value is not None]
    return {
        "rmse_ratio": {
            "label": "RMSE Ratio",
            "labels": [label for label, _value in entries],
            "values": [value for _label, value in entries],
        }
    }


def _history_facets(
    cells: Sequence[Mapping[str, Any]],
    *,
    per_term: bool,
) -> dict[str, Any]:
    facets = {}
    key = "training_loss_per_term" if per_term else "training_loss"
    for cell in cells:
        history = _mapping(cell.get(key))
        if per_term:
            series = _mapping(history.get("terms"))
        else:
            series = {"Loss": history.get("loss", history.get("value", []))}
        facets[str(cell["run_id"])] = {
            "x": history.get("step", history.get("steps")),
            "series": _jsonable(series),
            "color": cell.get("color"),
        }
    return facets


def _observed_metric_order(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for cell in cells:
        for metric in _mapping(cell.get("summary_metrics")):
            if metric not in seen:
                seen.append(str(metric))
    return seen


def _normalize_baseline(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    baseline = dict(_mapping(_jsonable(value)))
    profile = dict(_mapping(baseline.get("profile")))
    mean = profile.get("mean")
    if "y" not in profile and mean is not None:
        profile["y"] = [mean]
    if mean is not None:
        profile.setdefault("upper", mean)
        profile.setdefault("lower", mean)
    if "time" not in profile and "x" in profile:
        profile["time"] = profile["x"]
    baseline["profile"] = profile
    baseline["color"] = _plotly_rgb(baseline.get("color"))
    return baseline


def _metric_value(cell: Mapping[str, Any], metric: str) -> float | None:
    metrics = _mapping(cell.get("summary_metrics"))
    if metric in metrics:
        return float(metrics[metric])
    if metric in cell and isinstance(cell[metric], int | float):
        return float(cell[metric])
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _series(value: Any, *, default_len: int | None = None) -> list[float]:
    if value is None:
        return [] if default_len is None else [float(index) for index in range(default_len)]
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return [float(array)]
    return [float(item) for item in array.reshape(-1)]


def _plotly_rgb(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("#") or len(value) != 7:
        return value if isinstance(value, str) else None
    return f"rgb({int(value[1:3], 16)},{int(value[3:5], 16)},{int(value[5:7], 16)})"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(child) for child in value]
    return value


def figure_render_path(
    manifest_artifacts: Sequence[ArtifactRef],
    *,
    preferred_suffix: str,
) -> Path:
    """Return the custody render path with a preferred suffix."""
    for artifact in manifest_artifacts:
        if artifact.role == "figure_render" and artifact.uri and artifact.uri.endswith(
            preferred_suffix
        ):
            return Path(artifact.uri)
    for artifact in manifest_artifacts:
        if artifact.role == "figure_render" and artifact.uri:
            return Path(artifact.uri)
    raise ValueError("Figure manifest did not record a render artifact")

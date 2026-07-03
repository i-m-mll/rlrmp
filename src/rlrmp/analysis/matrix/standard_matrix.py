"""Feedbax-native standard matrix analysis bundle support."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe
from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput
from feedbax.analysis.specs import register_analysis_recipe
from feedbax.contracts.manifest import EvaluationRunSpec
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace

from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.spec_migrations import (
    STANDARD_MATRIX_EVAL_PARAMS_KIND,
    accept_rlrmp_spec_payload,
)
from rlrmp.viz import profile_comparison_grid

STANDARD_MATRIX_ANALYSIS_TYPE = "rlrmp.standard_matrix"
STANDARD_MATRIX_EVALUATION_TYPE = "rlrmp.standard_matrix_evaluation"

STANDARD_MATRIX_OUTPUTS = (
    "forward_velocity_profiles",
    "hold_drift_profiles",
    "peak_velocity_distributions",
    "summary_metrics",
    "training_loss",
    "training_loss_per_term",
    "rmse_ratio_comparison",
    "notes",
)


class StandardMatrixAnalysis(AbstractAnalysis):
    """Render one standard matrix output from materialized matrix state."""

    output: str = ""

    def __post_init__(self):
        super().__post_init__()
        if self.output not in STANDARD_MATRIX_OUTPUTS:
            raise ValueError(f"Unknown standard matrix output {self.output!r}")

    def compute(self, data: AnalysisInputData, **kwargs):
        cells = _normalise_cells(data.states, params=data.extras.get("params", {}))
        if self.output == "notes":
            return _notes_markdown(cells, data.extras.get("params", {}))
        return {"cells": cells, "params": data.extras.get("params", {})}

    def make_figs(self, data: AnalysisInputData, *, result, **kwargs):
        if self.output == "notes":
            return None
        return {self.output: _figure_for_output(self.output, result["cells"], result["params"])}

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result,
        **kwargs,
    ):
        if self.output != "notes":
            return None

        params = data.extras.get("params", {})
        marker = str(params.get("note_marker", "standard_matrix"))
        note_path = _notes_path(context, params)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        update_marked_section(note_path, marker, str(result).rstrip() + "\n")
        artifact = context.record_artifact(
            note_path,
            role="analysis_notes",
            logical_name=f"standard_matrix/{note_path.name}",
            media_type="text/markdown",
            metadata={"marker": marker},
        )
        return {"markdown": result, "artifact_refs": {"notes": artifact}}

    def _params_to_save(self, hps, *, result, **kwargs):
        params = result.get("params", {}) if isinstance(result, Mapping) else {}
        return {
            "standard_matrix_output": self.output,
            **_figure_params(params),
        }


def register_standard_matrix_recipes(*, replace: bool = True) -> None:
    """Register rlrmp's standard matrix analysis and evaluation recipes."""
    register_analysis_recipe(STANDARD_MATRIX_ANALYSIS_TYPE, standard_matrix_recipe, replace=replace)
    register_evaluation_recipe(
        STANDARD_MATRIX_EVALUATION_TYPE,
        standard_matrix_evaluation_recipe,
        replace=replace,
    )


def standard_matrix_evaluation_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Produce standard-matrix evaluation states from refs or explicit legacy payloads."""
    params = _accept_standard_matrix_params(run_spec.params)
    legacy_payload_mode = params.get("legacy_payload_mode") is True
    payload = params.get("matrix_payload", params.get("states"))
    if payload is not None and not legacy_payload_mode:
        raise ValueError(
            "standard-matrix matrix_payload/states params require legacy_payload_mode=true"
        )
    if payload is None:
        payload = {
            "cells": [
                {
                    "run_id": ref.id,
                    "label": ref.metadata.get("label", ref.id),
                    "display_name": ref.metadata.get("display_name", ref.id),
                    "color": ref.metadata.get("color"),
                }
                for ref in run_spec.inputs
            ]
        }
    elif not legacy_payload_mode:
        raise ValueError("standard-matrix legacy payload mode must be explicit")
    cells = _normalise_cells(payload, params=params)
    return EvaluationRecipeResult(
        states={"cells": cells},
        summary_metrics={"standard_matrix_cells": len(cells)},
        metadata={
            "standard_matrix": True,
            "cell_count": len(cells),
            "legacy_payload_mode": legacy_payload_mode,
            "params_schema_id": params.get("schema_id"),
            "params_schema_version": params.get("schema_version"),
        },
    )


def _accept_standard_matrix_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if "schema_version" not in params and "schema_id" not in params:
        return dict(params)
    result = accept_rlrmp_spec_payload(STANDARD_MATRIX_EVAL_PARAMS_KIND, params)
    return dict(result.payload)


def standard_matrix_recipe(
    spec,
    _root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Build Feedbax analyses for one standard matrix bundle expansion."""
    params = dict(spec.params)
    states = {
        "cells": _cells_from_inputs(inputs, params=params),
    }
    analyses = {
        output: StandardMatrixAnalysis(output=output, variant="standard_matrix")
        for output in STANDARD_MATRIX_OUTPUTS
    }
    data = AnalysisInputData(
        models={},
        tasks={},
        states=states,
        hps={"standard_matrix": TreeNamespace(task=TreeNamespace(eval_n=len(states["cells"])))},
        extras={"params": params},
    )
    return AnalysisRecipeResult(analyses=analyses, data=data)


def _cells_from_inputs(
    inputs: Sequence[ResolvedAnalysisInput],
    *,
    params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for resolved in inputs:
        raw_states = resolved.states
        if raw_states is None:
            raw_states = {}
        source_cells = _normalise_cells(raw_states, params=params)
        for index, cell in enumerate(source_cells):
            cells.append(
                _merge_cell_metadata(
                    cell,
                    params=params,
                    fallback_run_id=resolved.ref.id,
                    fallback_index=index,
                    manifest_metadata=(
                        resolved.manifest.metadata if resolved.manifest is not None else {}
                    ),
                )
            )
    return cells


def _normalise_cells(value: Any, *, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and "cells" in value:
        raw_cells = value["cells"]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_cells = value
    elif isinstance(value, Mapping):
        raw_cells = [value]
    else:
        raise TypeError("standard matrix states must be a mapping or sequence of mappings")

    cells = []
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, Mapping):
            raise TypeError(f"standard matrix cell {index} must be a mapping")
        cells.append(_merge_cell_metadata(dict(raw_cell), params=params, fallback_index=index))
    return cells


def _merge_cell_metadata(
    cell: dict[str, Any],
    *,
    params: Mapping[str, Any],
    fallback_run_id: str | None = None,
    fallback_index: int = 0,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = str(cell.get("run_id") or fallback_run_id or f"cell_{fallback_index}")
    label = str(cell.get("label") or _manifest_metadata_value(manifest_metadata, "cell") or run_id)
    metadata = _cell_metadata(params).get(label, {})
    if run_id in _cell_metadata(params):
        metadata = {**metadata, **_cell_metadata(params)[run_id]}
    return {
        **cell,
        "run_id": run_id,
        "label": label,
        "display_name": str(
            cell.get("display_name")
            or metadata.get("display_name")
            or _manifest_metadata_value(manifest_metadata, "display_name")
            or label
        ),
        "color": cell.get("color") or metadata.get("color"),
    }


def _cell_metadata(params: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    metadata = params.get("cell_metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _manifest_metadata_value(metadata: Mapping[str, Any] | None, key: str) -> Any:
    return metadata.get(key) if metadata is not None else None


def _figure_params(params: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key in ("figure_routing", "presentation", "metric_order", "profile_key"):
        if key in params:
            output[key] = params[key]
    return output


def _figure_for_output(output: str, cells: Sequence[Mapping[str, Any]], params: Mapping[str, Any]):
    if output == "forward_velocity_profiles":
        return _profile_figure(
            cells, profile_key=str(params.get("profile_key", "forward_velocity"))
        )
    if output == "hold_drift_profiles":
        return _profile_figure(cells, profile_key=str(params.get("profile_key", "hold_drift")))
    if output == "peak_velocity_distributions":
        return _peak_velocity_figure(cells)
    if output == "summary_metrics":
        return _summary_metrics_figure(cells, params=params)
    if output == "training_loss":
        return _training_loss_figure(cells)
    if output == "training_loss_per_term":
        return _training_loss_per_term_figure(cells)
    if output == "rmse_ratio_comparison":
        return _rmse_ratio_figure(cells)
    raise ValueError(f"Unknown standard matrix output {output!r}")


def _profile_figure(cells: Sequence[Mapping[str, Any]], *, profile_key: str) -> go.Figure:
    fig = profile_comparison_grid(
        max(1, len(cells)),
        subplot_titles=[str(cell["display_name"]) for cell in cells] or [profile_key],
        vertical_spacing=0.04,
    )
    for row, cell in enumerate(cells, start=1):
        profile = _mapping(cell.get(profile_key, {}))
        x = _series(profile.get("time", cell.get("time")), default_len=_profile_len(profile))
        mean = _series(profile.get("mean", profile.get("value", [])))
        color = cell.get("color")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                name=str(cell["display_name"]),
                line={"color": color} if color else None,
            ),
            row=row,
            col=1,
        )
        lower = profile.get("lower")
        upper = profile.get("upper")
        if lower is not None and upper is not None:
            upper_values = _series(upper)
            lower_values = _series(lower)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=upper_values,
                    mode="lines",
                    line={"width": 0, "color": color} if color else {"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower_values,
                    mode="lines",
                    fill="tonexty",
                    line={"width": 0, "color": color} if color else {"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
    fig.update_layout(template="plotly_white", title=profile_key.replace("_", " ").title())
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text=profile_key.replace("_", " ").title())
    return fig


def _peak_velocity_figure(cells: Sequence[Mapping[str, Any]]) -> go.Figure:
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
    fig.update_layout(template="plotly_white", title="Peak Velocity Distributions")
    fig.update_yaxes(title_text="Peak velocity")
    return fig


def _summary_metrics_figure(
    cells: Sequence[Mapping[str, Any]],
    *,
    params: Mapping[str, Any],
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
        title="Summary Metrics",
        xaxis_title="Cell",
    )
    return fig


def _training_loss_figure(cells: Sequence[Mapping[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for cell in cells:
        loss = _mapping(cell.get("training_loss", {}))
        steps = _series(
            loss.get("step", loss.get("steps")), default_len=len(_series(loss.get("loss", [])))
        )
        values = _series(loss.get("loss", loss.get("value", [])))
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines",
                name=str(cell["display_name"]),
                line={"color": cell.get("color")} if cell.get("color") else None,
            )
        )
    fig.update_layout(template="plotly_white", title="Training Loss", xaxis_title="Step")
    fig.update_yaxes(title_text="Loss")
    return fig


def _training_loss_per_term_figure(cells: Sequence[Mapping[str, Any]]) -> go.Figure:
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
    fig.update_layout(template="plotly_white", title="Training Loss Per Term", xaxis_title="Step")
    fig.update_yaxes(title_text="Loss")
    return fig


def _rmse_ratio_figure(cells: Sequence[Mapping[str, Any]]) -> go.Figure:
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
    fig.update_layout(template="plotly_white", title="RMSE Ratio Comparison")
    fig.update_yaxes(title_text="RMSE ratio")
    return fig


def _notes_markdown(cells: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> str:
    metric_order = list(params.get("metric_order") or _observed_metric_order(cells))
    lines = ["## Standard Matrix Summary", "", "| Cell | " + " | ".join(metric_order) + " |"]
    lines.append("|---|" + "|".join("---" for _ in metric_order) + "|")
    for cell in cells:
        values = [_format_metric(_metric_value(cell, metric)) for metric in metric_order]
        lines.append(f"| {cell['display_name']} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _notes_path(context: AnalysisRunContext, params: Mapping[str, Any]) -> Path:
    raw = params.get("notes_path")
    if raw is not None:
        return Path(raw)
    experiment = _figure_routing(params).get("experiment", "standard_matrix")
    return REPO_ROOT / "results" / str(experiment) / "notes" / "matrix_results.md"


def _figure_routing(params: Mapping[str, Any]) -> Mapping[str, Any]:
    routing = params.get("figure_routing", {})
    return routing if isinstance(routing, Mapping) else {}


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


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


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


def _profile_len(profile: Mapping[str, Any]) -> int | None:
    for key in ("mean", "value", "lower", "upper"):
        if key in profile:
            return len(_series(profile[key]))
    return None


def dump_standard_matrix_payload(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON payload useful for standard-matrix evaluation specs."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

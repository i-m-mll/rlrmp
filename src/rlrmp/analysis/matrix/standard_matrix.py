"""Feedbax-native standard matrix analysis bundle support."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe
from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput
from feedbax.analysis.specs import register_analysis_recipe
from feedbax.contracts.manifest import EvaluationRunSpec
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.params_models import register_params_model
from rlrmp.runtime.spec_migrations import (
    STANDARD_MATRIX_EVAL_PARAMS_KIND,
    accept_rlrmp_spec_payload,
)
from rlrmp.figures import STANDARD_MATRIX_PAYLOAD_ROLE, standard_matrix_payload

STANDARD_MATRIX_ANALYSIS_TYPE = "rlrmp.standard_matrix"
STANDARD_MATRIX_EVALUATION_TYPE = "rlrmp.standard_matrix_evaluation"

STANDARD_MATRIX_OUTPUTS = (
    "figure_payload",
    "forward_velocity_profiles",
    "hold_drift_profiles",
    "peak_velocity_distributions",
    "summary_metrics",
    "training_loss",
    "training_loss_per_term",
    "rmse_ratio_comparison",
    "notes",
)


class StandardMatrixEvalParams(BaseModel):
    """Params for the standard-matrix evaluation recipe."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str | None = None
    schema_version: str | None = None
    matrix_payload: Any | None = None
    states: Any | None = None
    legacy_payload_mode: bool = False
    cell_metadata: dict[str, Any] = Field(default_factory=dict)
    profile_key: str | None = None
    metric_order: list[str] | None = None
    notes_path: str | None = None
    note_marker: str | None = None
    figure_routing: dict[str, Any] = Field(default_factory=dict)


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

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result,
        **kwargs,
    ):
        if self.output == "figure_payload":
            payload = standard_matrix_payload(result["cells"], result["params"])
            context.metadata = {
                **dict(context.metadata or {}),
                "figure_payload": payload,
            }
            artifact = context.record_json_artifact(
                payload,
                role=STANDARD_MATRIX_PAYLOAD_ROLE,
                logical_name="standard_matrix/payload.json",
                metadata={"standard_matrix": True},
            )
            return {"payload": result, "artifact_refs": {"payload": artifact}}

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
    register_params_model(
        STANDARD_MATRIX_EVALUATION_TYPE,
        StandardMatrixEvalParams,
        replace=replace,
    )
    register_analysis_recipe(
        STANDARD_MATRIX_ANALYSIS_TYPE,
        standard_matrix_recipe,
        replace=replace,
    )
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


def dump_standard_matrix_payload(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON payload useful for standard-matrix evaluation specs."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

"""Robustness-margin sidecar estimators for soft-constraint rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np

from feedbax.analysis import StagedExecutionContext
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput
from feedbax.analysis.specs import register_analysis_recipe
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.runtime.params_models import register_params_model


ROBUSTNESS_MARGIN_ANALYSIS_TYPE = "rlrmp.robustness_margin_sidecar"
ROBUSTNESS_MARGIN_SCHEMA_VERSION = "rlrmp.robustness_margin_sidecar.v1"
DEFAULT_HEADLINE_QUANTILE = 0.9
DEFAULT_THRESHOLD_MULTIPLIER = 10.0

EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE = {
    ROBUSTNESS_MARGIN_ANALYSIS_TYPE: ("robustness_margin_rows",),
}


class RobustnessMarginParams(BaseModel):
    """Schema-bearing params for the robustness-margin sidecar analysis."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str | None = None
    schema_version: str | None = None
    issue_id: str = "1ec6ae5"
    scope: str = "soft_constraint_robustness_margin"
    headline_quantile: float = Field(default=DEFAULT_HEADLINE_QUANTILE, gt=0.0, le=1.0)
    threshold_multiplier: float = Field(default=DEFAULT_THRESHOLD_MULTIPLIER, gt=0.0)
    row_keys: list[str] = Field(
        default_factory=lambda: ["robustness_margin_rows", "margin_inputs", "rows"]
    )


class RobustnessMarginSidecarAnalysis(AbstractAnalysis):
    """Compute robustness-margin rows and record the sidecar through Feedbax custody."""

    def compute(self, data: AnalysisInputData, **_kwargs: Any) -> dict[str, Any]:
        params = dict(data.extras.get("params", {}))
        rows = [
            build_margin_row(
                row,
                headline_quantile=float(params["headline_quantile"]),
                threshold_multiplier=float(
                    row.get("threshold_multiplier", params["threshold_multiplier"])
                ),
            )
            for row in data.states["rows"]
        ]
        return build_robustness_margin_sidecar(
            rows=rows,
            params=params,
            input_refs=data.extras.get("input_refs", ()),
        )

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            **dict(result),
            "declarative_analysis": {
                "analysis_manifest_id": context.manifest_id,
                "artifact_custody": "feedbax.AnalysisRunManifest",
            },
        }
        artifact = context.record_json_artifact(
            payload,
            role="rlrmp-robustness-margin-sidecar",
            logical_name="robustness_margin_sidecar.json",
            metadata={
                "schema_version": ROBUSTNESS_MARGIN_SCHEMA_VERSION,
                "schema_boundary": "rlrmp-owned robustness-margin sidecar payload",
                "issue": str(data.extras.get("params", {}).get("issue_id", "1ec6ae5")),
            },
        )
        return {**payload, "artifact_refs": {"sidecar": artifact}}


def register_robustness_margin_recipes(*, replace: bool = True) -> None:
    """Register the robustness-margin analysis recipe and params schema."""

    register_params_model(
        ROBUSTNESS_MARGIN_ANALYSIS_TYPE,
        RobustnessMarginParams,
        replace=True,
    )
    register_analysis_recipe(
        ROBUSTNESS_MARGIN_ANALYSIS_TYPE,
        robustness_margin_recipe,
        replace=replace,
    )


def robustness_margin_recipe(
    spec: Any,
    _root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
    _execution_context: StagedExecutionContext,
) -> AnalysisRecipeResult:
    """Build the Feedbax analysis node from upstream evaluation diagnostics."""

    params = RobustnessMarginParams.model_validate(dict(spec.params)).model_dump()
    rows = _rows_from_inputs(inputs, row_keys=tuple(params["row_keys"]))
    analysis = RobustnessMarginSidecarAnalysis(
        variant="robustness_margin_sidecar",
        cache_result=True,
    )
    return AnalysisRecipeResult(
        analyses={"robustness_margin_sidecar": analysis},
        data=AnalysisInputData(
            models={},
            tasks={},
            states={"rows": rows},
            hps={
                "robustness_margin": TreeNamespace(
                    task=TreeNamespace(eval_n=len(rows)),
                )
            },
            extras={
                "params": params,
                "input_refs": [
                    {
                        "kind": resolved.ref.kind,
                        "id": resolved.ref.id,
                        "role": resolved.ref.role,
                    }
                    for resolved in inputs
                ],
            },
        ),
    )


def build_robustness_margin_sidecar(
    *,
    rows: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any] | None = None,
    input_refs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the schema-stamped robustness-margin sidecar payload."""

    params = dict(params or {})
    finite_rows = [row for row in rows if _is_positive_finite(row.get("headline_lambda_margin"))]
    margins = [float(row["headline_lambda_margin"]) for row in finite_rows]
    return {
        "schema_version": ROBUSTNESS_MARGIN_SCHEMA_VERSION,
        "issue": str(params.get("issue_id", "1ec6ae5")),
        "scope": str(params.get("scope", "soft_constraint_robustness_margin")),
        "estimator_contract": {
            "small_signal": "top generalized Hessian eigenvalue around declared trajectories",
            "large_signal": "declared adversary probe price bisection; no internal rollout reruns",
            "headline_quantile": float(params.get("headline_quantile", DEFAULT_HEADLINE_QUANTILE)),
            "threshold_multiplier": float(
                params.get("threshold_multiplier", DEFAULT_THRESHOLD_MULTIPLIER)
            ),
        },
        "inputs": list(input_refs),
        "summary": {
            "row_count": len(rows),
            "finite_margin_count": len(finite_rows),
            "min_lambda_margin": min(margins) if margins else None,
            "median_lambda_margin": float(np.median(margins)) if margins else None,
            "prediction_check": margin_prediction_check(rows),
        },
        "rows": [dict(row) for row in rows],
    }


def build_margin_row(
    row: Mapping[str, Any],
    *,
    headline_quantile: float = DEFAULT_HEADLINE_QUANTILE,
    threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
) -> dict[str, Any]:
    """Estimate both breaking prices and lambda margins for one row."""

    normalized = dict(row)
    lambda_value = _first_present_float(
        normalized,
        ("lambda", "lambda_value", "energy_lambda", "converged_lambda"),
    )
    setpoint = _first_present_float(
        normalized,
        ("setpoint", "damage_setpoint", "target_damage", "objective_setpoint"),
    )
    small_signal = _small_signal_estimate(normalized, headline_quantile=headline_quantile)
    large_signal = _large_signal_estimate(
        normalized,
        setpoint=setpoint,
        headline_quantile=headline_quantile,
        threshold_multiplier=threshold_multiplier,
    )
    headline = _headline_breaking_price(small_signal, large_signal)
    margin = (
        float(lambda_value) / float(headline["price"])
        if lambda_value is not None and _is_positive_finite(headline.get("price"))
        else None
    )
    normalized.update(
        {
            "schema_version": ROBUSTNESS_MARGIN_SCHEMA_VERSION,
            "lambda_value": lambda_value,
            "damage_setpoint": setpoint,
            "small_signal": small_signal,
            "large_signal": large_signal,
            "headline_breaking_price": headline,
            "headline_lambda_margin": margin,
            "status": _row_status(lambda_value, small_signal, large_signal, margin),
        }
    )
    return normalized


def hessian_breaking_price(
    hessian: Sequence[Sequence[float]],
    *,
    energy_metric: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Estimate the small-signal breaking price from an explicit Hessian.

    The price is the largest generalized eigenvalue of the symmetrized loss
    Hessian with respect to the disturbance-energy metric.
    """

    h = _as_square_matrix(hessian, name="hessian")
    h = 0.5 * (h + h.T)
    if energy_metric is None:
        metric = np.eye(h.shape[0])
    else:
        metric = _as_square_matrix(energy_metric, name="energy_metric")
        metric = 0.5 * (metric + metric.T)
    factor = np.linalg.cholesky(metric)
    inv_factor = np.linalg.inv(factor)
    normalized_hessian = inv_factor @ h @ inv_factor.T
    normalized_hessian = 0.5 * (normalized_hessian + normalized_hessian.T)
    eigvals, eigvecs = np.linalg.eigh(normalized_hessian)
    index = int(np.argmax(eigvals))
    value = float(eigvals[index])
    vector = np.asarray(inv_factor.T @ eigvecs[:, index], dtype=float)
    norm = math.sqrt(float(vector.T @ metric @ vector))
    if norm > 0.0:
        vector = vector / norm
    return {
        "status": "available",
        "breaking_price": value,
        "top_eigenvalue": value,
        "worst_case_waveform": vector.tolist(),
    }


def price_bisection_breaking_price(
    probes: Sequence[Mapping[str, Any]],
    *,
    setpoint: float,
    threshold_multiplier: float = DEFAULT_THRESHOLD_MULTIPLIER,
) -> dict[str, Any]:
    """Estimate the price knee from monotone adversary probe observations."""

    if setpoint <= 0.0:
        raise ValueError("setpoint must be positive for price-bisection estimation")
    threshold = float(setpoint) * float(threshold_multiplier)
    observations = sorted(
        (
            {
                "price": _required_float(probe, ("price", "lambda", "lambda_value")),
                "damage": _required_float(probe, ("damage", "measured_damage", "loss")),
            }
            for probe in probes
        ),
        key=lambda item: item["price"],
    )
    if not observations:
        return {"status": "missing", "reason": "no probes", "threshold": threshold}

    unsafe = [item for item in observations if item["damage"] > threshold]
    safe = [item for item in observations if item["damage"] <= threshold]
    if not unsafe:
        return {
            "status": "right_censored",
            "breaking_price": observations[0]["price"],
            "threshold": threshold,
            "probes": observations,
        }
    if not safe:
        return {
            "status": "left_censored",
            "breaking_price": observations[-1]["price"],
            "threshold": threshold,
            "probes": observations,
        }

    lower = max(unsafe, key=lambda item: item["price"])
    upper_candidates = [item for item in safe if item["price"] >= lower["price"]]
    upper = min(upper_candidates, key=lambda item: item["price"]) if upper_candidates else safe[0]
    estimate = _log_linear_crossing(
        lower["price"],
        lower["damage"],
        upper["price"],
        upper["damage"],
        threshold,
    )
    return {
        "status": "available",
        "breaking_price": estimate,
        "threshold": threshold,
        "bracket": {"unsafe": lower, "safe": upper},
        "probes": observations,
    }


def margin_prediction_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check whether margins decrease as setpoint grows for finite row pairs."""

    pairs = [
        (float(row["damage_setpoint"]), float(row["headline_lambda_margin"]))
        for row in rows
        if _is_positive_finite(row.get("damage_setpoint"))
        and _is_positive_finite(row.get("headline_lambda_margin"))
    ]
    if len(pairs) < 2:
        return {"status": "not_applicable", "reason": "fewer than two finite setpoint rows"}
    ordered = sorted(pairs)
    nonincreasing = all(
        right_margin <= left_margin + 1e-12
        for (_, left_margin), (_, right_margin) in zip(ordered, ordered[1:])
    )
    return {
        "status": "pass" if nonincreasing else "fail",
        "ordered_setpoint_margin": [
            {"setpoint": setpoint, "lambda_margin": margin} for setpoint, margin in ordered
        ],
    }


def _rows_from_inputs(
    inputs: Sequence[ResolvedAnalysisInput],
    *,
    row_keys: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resolved in inputs:
        for row in _rows_from_payload(resolved.states, row_keys=row_keys):
            row.setdefault("source_manifest_id", resolved.ref.id)
            rows.append(row)
    if not rows:
        raise ValueError(
            "robustness-margin analysis requires upstream evaluation states with "
            f"one of these row keys: {', '.join(row_keys)}"
        )
    return rows


def _rows_from_payload(payload: Any, *, row_keys: Sequence[str]) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        for key in row_keys:
            if key in payload:
                return _coerce_rows(payload[key])
        if "runs" in payload and isinstance(payload["runs"], Mapping):
            rows = []
            for run_id, run_payload in payload["runs"].items():
                for row in _rows_from_payload(run_payload, row_keys=row_keys):
                    row.setdefault("run_id", run_id)
                    rows.append(row)
            return rows
        if _looks_like_margin_row(payload):
            return [dict(payload)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return _coerce_rows(payload)
    return []


def _coerce_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("robustness-margin rows must be a mapping or sequence of mappings")
    rows = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"robustness-margin row {index} must be a mapping")
        rows.append(dict(item))
    return rows


def _looks_like_margin_row(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "hessian",
            "hessian_trials",
            "price_probes",
            "price_probe_trials",
            "probe_results",
        )
    )


def _small_signal_estimate(
    row: Mapping[str, Any],
    *,
    headline_quantile: float,
) -> dict[str, Any]:
    hessian_trials = row.get("hessian_trials")
    if hessian_trials is not None:
        estimates = [
            _trial_hessian_estimate(trial, row=row)
            for trial in _coerce_trial_mappings(hessian_trials, item_name="hessian trial")
        ]
        prices = [
            float(estimate["breaking_price"])
            for estimate in estimates
            if _is_positive_finite(estimate.get("breaking_price"))
        ]
        if not prices:
            return {"status": "missing", "reason": "hessian trials have no finite prices"}
        headline = float(np.quantile(prices, headline_quantile))
        worst = max(estimates, key=lambda estimate: float(estimate.get("breaking_price", 0.0)))
        return {
            "status": "available",
            "breaking_price": headline,
            "headline_quantile": headline_quantile,
            "per_trial": estimates,
            "worst_case_waveform": worst.get("worst_case_waveform"),
        }

    hessian = row.get("hessian")
    if hessian is None:
        return {"status": "missing", "reason": "row has no hessian"}
    estimate = hessian_breaking_price(hessian, energy_metric=row.get("energy_metric"))
    shape = row.get("disturbance_shape")
    if shape is not None and estimate.get("worst_case_waveform") is not None:
        estimate["worst_case_waveform_shape"] = list(shape)
        estimate["worst_case_waveform"] = (
            np.asarray(estimate["worst_case_waveform"]).reshape(tuple(shape)).tolist()
        )
    return estimate


def _large_signal_estimate(
    row: Mapping[str, Any],
    *,
    setpoint: float | None,
    headline_quantile: float,
    threshold_multiplier: float,
) -> dict[str, Any]:
    probe_trials = row.get("price_probe_trials")
    if probe_trials is not None:
        if setpoint is None:
            return {"status": "missing", "reason": "row has no damage setpoint"}
        estimates = [
            _trial_probe_estimate(
                trial,
                setpoint=float(setpoint),
                threshold_multiplier=threshold_multiplier,
            )
            for trial in _coerce_trial_mappings(probe_trials, item_name="price probe trial")
        ]
        prices = [
            float(estimate["breaking_price"])
            for estimate in estimates
            if _is_positive_finite(estimate.get("breaking_price"))
        ]
        if not prices:
            return {"status": "missing", "reason": "price probe trials have no finite prices"}
        return {
            "status": "available",
            "breaking_price": float(np.quantile(prices, headline_quantile)),
            "headline_quantile": headline_quantile,
            "threshold": estimates[0].get("threshold") if estimates else None,
            "per_trial": estimates,
        }

    probes = row.get("price_probes", row.get("probe_results"))
    if probes is None:
        return {"status": "missing", "reason": "row has no price probes"}
    if setpoint is None:
        return {"status": "missing", "reason": "row has no damage setpoint"}
    return price_bisection_breaking_price(
        probes,
        setpoint=float(setpoint),
        threshold_multiplier=threshold_multiplier,
    )


def _trial_hessian_estimate(
    trial: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    hessian = trial.get("hessian", trial.get("value", trial.get("matrix")))
    if hessian is None:
        raise ValueError("hessian trial missing hessian/matrix value")
    estimate = hessian_breaking_price(
        hessian,
        energy_metric=trial.get("energy_metric", row.get("energy_metric")),
    )
    if "trial_id" in trial:
        estimate["trial_id"] = trial["trial_id"]
    return estimate


def _trial_probe_estimate(
    trial: Mapping[str, Any],
    *,
    setpoint: float,
    threshold_multiplier: float,
) -> dict[str, Any]:
    probes = trial.get("price_probes", trial.get("probes", trial.get("probe_results")))
    if probes is None:
        raise ValueError("price probe trial missing probes")
    estimate = price_bisection_breaking_price(
        probes,
        setpoint=setpoint,
        threshold_multiplier=threshold_multiplier,
    )
    if "trial_id" in trial:
        estimate["trial_id"] = trial["trial_id"]
    return estimate


def _coerce_trial_mappings(value: Any, *, item_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{item_name}s must be a sequence of mappings")
    trials = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            trials.append(dict(item))
        else:
            trials.append({"trial_id": index, "value": item})
    return trials


def _headline_breaking_price(
    small_signal: Mapping[str, Any],
    large_signal: Mapping[str, Any],
) -> dict[str, Any]:
    if _is_positive_finite(large_signal.get("breaking_price")):
        return {
            "source": "large_signal",
            "price": float(large_signal["breaking_price"]),
        }
    if _is_positive_finite(small_signal.get("breaking_price")):
        return {
            "source": "small_signal",
            "price": float(small_signal["breaking_price"]),
        }
    return {"source": "missing", "price": None}


def _row_status(
    lambda_value: float | None,
    small_signal: Mapping[str, Any],
    large_signal: Mapping[str, Any],
    margin: float | None,
) -> str:
    if lambda_value is None:
        return "missing_lambda"
    if small_signal.get("status") == "missing" and large_signal.get("status") == "missing":
        return "missing_estimators"
    if margin is None:
        return "missing_margin"
    if margin < 1.0:
        return "below_breaking_price"
    return "available"


def _log_linear_crossing(
    lower_price: float,
    lower_damage: float,
    upper_price: float,
    upper_damage: float,
    threshold: float,
) -> float:
    if lower_price <= 0.0 or upper_price <= 0.0:
        return 0.5 * (lower_price + upper_price)
    if lower_damage <= 0.0 or upper_damage <= 0.0 or lower_damage == upper_damage:
        return math.sqrt(lower_price * upper_price)
    fraction = (math.log(threshold) - math.log(lower_damage)) / (
        math.log(upper_damage) - math.log(lower_damage)
    )
    fraction = min(1.0, max(0.0, fraction))
    return float(
        math.exp(math.log(lower_price) + fraction * (math.log(upper_price) - math.log(lower_price)))
    )


def _as_square_matrix(value: Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return array


def _first_present_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])
    return None


def _required_float(row: Mapping[str, Any], keys: Sequence[str]) -> float:
    value = _first_present_float(row, keys)
    if value is None:
        raise ValueError(f"probe missing one of: {', '.join(keys)}")
    return value


def _is_positive_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


__all__ = [
    "DEFAULT_HEADLINE_QUANTILE",
    "DEFAULT_THRESHOLD_MULTIPLIER",
    "EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE",
    "ROBUSTNESS_MARGIN_ANALYSIS_TYPE",
    "ROBUSTNESS_MARGIN_SCHEMA_VERSION",
    "RobustnessMarginParams",
    "build_margin_row",
    "build_robustness_margin_sidecar",
    "hessian_breaking_price",
    "margin_prediction_check",
    "price_bisection_breaking_price",
    "register_robustness_margin_recipes",
    "robustness_margin_recipe",
]

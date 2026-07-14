"""Worst-case epsilon optimization and cached-rollout audit summaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from feedbax.analysis import StagedExecutionContext
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    ResolvedAnalysisInput,
    register_analysis_recipe,
)
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.analysis.math.cs_game_card import TARGET_POS, build_canonical_game
from rlrmp.data_products.broad_epsilon import load_broad_epsilon_anchors
from rlrmp.eval.recipes import WORST_CASE_EPSILON_EVALUATION_TYPE
from rlrmp.train.cs_perturbation_training import BROAD_EPSILON_REFERENCE_REACH_M

WORST_CASE_EPSILON_ANALYSIS_TYPE = "rlrmp.analysis.worst_case_epsilon"
WORST_CASE_EPSILON_PARAMS_SCHEMA = "rlrmp.analysis.worst_case_epsilon.params.v1"
ObjectiveFn = Callable[[Any], Any]
EpsilonOptimizerBackend = Literal["serial", "staged"]


class WorstCaseEpsilonAnalysisParams(BaseModel):
    """Governed optimizer-summary parameters."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str = WORST_CASE_EPSILON_PARAMS_SCHEMA
    schema_version: str = "v1"
    run_ids: tuple[str, ...] = ()
    budget_level: str | None = None
    budget_scale: float | None = Field(default=None, gt=0)
    n_steps: int = Field(default=12, ge=0)
    n_restarts: int = Field(default=3, ge=0)
    step_size: float | None = Field(default=None, gt=0)
    backend: EpsilonOptimizerBackend = "serial"


@dataclass(frozen=True)
class EpsilonOptimizationResult:
    """Projected-ascent result for one epsilon sequence."""

    epsilon: np.ndarray
    objective: float
    initial_objective: float
    final_objective: float
    energy: float
    l2_norm: float
    restart_index: int
    history: tuple[dict[str, float | int], ...]
    restart_summaries: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class FullQrfRolloutCostContext:
    """Candidate-invariant full-Q/R/Qf rollout-cost arrays."""

    initial_states: Any
    target_pos: Any
    q: Any
    r: Any
    q_f: Any


class WorstCaseEpsilonSummaryAnalysis(AbstractAnalysis):
    """Summarize optimizer and endpoint records from cached evaluation states."""

    def compute(self, data: AnalysisInputData, **_kwargs):
        params = dict(data.extras["params"])
        rows = list(data.states["rows"])
        return {
            "schema_version": "rlrmp.analysis.worst_case_epsilon.v1",
            "params": params,
            "rows": rows,
            "summary": {
                "n_rows": len(rows),
                "n_evaluated": sum(row.get("status", "evaluated") == "evaluated" for row in rows),
            },
        }


def register_worst_case_epsilon_recipe(*, replace: bool = True) -> None:
    """Register the cached-evaluation analysis recipe."""

    register_analysis_recipe(
        WORST_CASE_EPSILON_ANALYSIS_TYPE,
        worst_case_epsilon_recipe,
        replace=replace,
    )


def worst_case_epsilon_recipe(
    spec,
    _root,
    inputs: Sequence[ResolvedAnalysisInput],
    _execution_context: StagedExecutionContext,
) -> AnalysisRecipeResult:
    """Build a summary analysis without rerunning model rollouts."""

    params = WorstCaseEpsilonAnalysisParams.model_validate(spec.params)
    rows: list[Any] = []
    for resolved in inputs:
        states = resolved.states
        if isinstance(states, Mapping):
            payload = states.get("worst_case_rollouts", states.get("rows", ()))
            rows.extend(payload if isinstance(payload, Sequence) else (payload,))
    return AnalysisRecipeResult(
        analyses={"summary": WorstCaseEpsilonSummaryAnalysis(variant="worst_case_epsilon")},
        data=AnalysisInputData(
            models={},
            tasks={},
            states={"rows": rows},
            hps={"worst_case_epsilon": TreeNamespace(task=TreeNamespace(eval_n=len(rows)))},
            extras={"params": params.model_dump(mode="json", exclude_none=True)},
        ),
    )


worst_case_epsilon_recipe.EVAL_DEPENDENCIES = (WORST_CASE_EPSILON_EVALUATION_TYPE,)


def project_l2_ball(epsilon: Any, radius: float) -> Any:
    """Project epsilon onto a closed flattened L2 ball."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    eps = jnp.asarray(epsilon)
    norm = jnp.linalg.norm(eps)
    return eps * jnp.minimum(1.0, jnp.asarray(radius, dtype=eps.dtype) / (norm + 1e-30))


def epsilon_energy(epsilon: Any) -> float:
    """Return flattened epsilon energy."""

    return float(np.sum(np.square(np.asarray(epsilon, dtype=np.float64))))


def frozen_batch_adversary_audit_report(
    *,
    selected_epsilon: Any,
    selected_objective: float,
    zero_objective: float,
    safety_cap_l2_radius: float | None = None,
    batch_size: int | None = None,
    reference_batch_size: int | None = None,
    cap_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Summarize fixed-fixture optimizer and endpoint parity fields."""

    eps = np.asarray(selected_epsilon, dtype=np.float64)
    if eps.ndim < 2:
        raise ValueError("selected_epsilon must include time and epsilon dimensions")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive when provided")
    if reference_batch_size is not None and reference_batch_size < 1:
        raise ValueError("reference_batch_size must be positive when provided")
    if safety_cap_l2_radius is not None and safety_cap_l2_radius < 0:
        raise ValueError("safety_cap_l2_radius must be non-negative when provided")
    norms = _candidate_flattened_l2_norms(eps)
    energy = epsilon_energy(eps)
    gain = float(selected_objective) - float(zero_objective)
    scaling = {
        "batch_size": batch_size,
        "reference_batch_size": reference_batch_size,
        "objective_reduction": "caller_supplied_objective_values",
        "accepted_objective_gain_per_batch_item": None if batch_size is None else gain / batch_size,
        "selected_epsilon_energy_per_batch_item": None
        if batch_size is None
        else energy / batch_size,
    }
    if batch_size is not None and reference_batch_size is not None:
        scaling["reference_scaled_objective_gain"] = gain * reference_batch_size / batch_size
    return {
        "selected_epsilon_energy": energy,
        "selected_epsilon_l2_norm": float(np.linalg.norm(eps)),
        "selected_epsilon_l2_norm_max_per_sample": float(np.max(norms)) if norms.size else 0.0,
        "accepted_objective_gain_over_zero": gain,
        "cap_bound_fraction": None
        if safety_cap_l2_radius is None
        else float(np.mean(norms >= safety_cap_l2_radius * (1.0 - cap_tolerance))),
        "cap_tolerance": float(cap_tolerance),
        "nan_overflow_status": _nonfinite_status(eps, selected_objective, zero_objective),
        "batch_size_scaling": scaling,
    }


def declared_epsilon_l2_radius(
    run_spec: Mapping[str, Any],
    *,
    reach_length_m: float | None = None,
    budget_level_override: str | None = None,
    budget_scale_override: float | None = None,
) -> float:
    """Resolve the declared epsilon budget without experiment-specific defaults."""

    if budget_level_override is not None:
        anchors = load_broad_epsilon_anchors()
        if budget_level_override not in anchors:
            raise ValueError(f"Unknown budget_level_override {budget_level_override!r}")
        radius = float(anchors[budget_level_override]["closed_loop_epsilon_l2_15cm"])
        radius *= 1.0 if budget_scale_override is None else budget_scale_override
        if reach_length_m is not None:
            radius *= reach_length_m / BROAD_EPSILON_REFERENCE_REACH_M
        return radius
    hps = run_spec.get("hps", {})
    config = hps.get("broad_epsilon_training", {}) if isinstance(hps, Mapping) else {}
    pgd = hps.get("broad_epsilon_pgd_training", {}) if isinstance(hps, Mapping) else {}
    if not bool(config.get("enabled", False)) and bool(pgd.get("enabled", False)):
        config = pgd
    contract = config.get("budget_contract", {})
    schedule = config.get("budget_schedule", {})
    raw = (
        schedule.get("max_l2_radius_15cm")
        or contract.get("active_max_l2_radius_15cm")
        or contract.get("effective_l2_radius_15cm")
        or contract.get("closed_loop_epsilon_l2_15cm")
    )
    if raw is None:
        raise ValueError("run spec lacks a broad-epsilon L2 radius")
    radius = float(raw) * float(config.get("budget_scale", 1.0) or 1.0)
    if bool(config.get("reach_length_scaling", False)) and reach_length_m is not None:
        radius *= reach_length_m / float(contract.get("reference_reach_m", 0.15) or 0.15)
    return radius


def optimize_epsilon_sequence(
    objective: ObjectiveFn,
    *,
    shape: tuple[int, int],
    radius: float,
    n_steps: int,
    n_restarts: int,
    step_size: float,
    seed: int = 0,
    initial_candidates: Sequence[Any] = (),
    backend: EpsilonOptimizerBackend = "serial",
) -> EpsilonOptimizationResult:
    """Run deterministic projected ascent with best-incumbent retention."""

    if n_steps < 0 or n_restarts < 0 or radius < 0:
        raise ValueError("n_steps, n_restarts, and radius must be non-negative")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if backend not in ("serial", "staged"):
        raise ValueError(f"unknown epsilon optimizer backend {backend!r}")
    starts = [
        project_l2_ball(jnp.asarray(value, dtype=jnp.float64), radius)
        for value in initial_candidates
    ]
    n_random = max(0, n_restarts - len(starts))
    starts.extend(
        project_l2_ball(jax.random.normal(key, shape, dtype=jnp.float64), radius)
        for key in jax.random.split(jax.random.PRNGKey(seed), n_random)
    )
    if not starts:
        starts.append(jnp.zeros(shape, dtype=jnp.float64))
    return _optimize_starts(
        objective, tuple(starts), radius=radius, n_steps=n_steps, step_size=step_size
    )


def _optimize_starts(
    objective: ObjectiveFn,
    starts: Sequence[Any],
    *,
    radius: float,
    n_steps: int,
    step_size: float,
) -> EpsilonOptimizationResult:
    results = []
    for restart_index, start in enumerate(starts):
        epsilon = start
        initial = float(objective(epsilon))
        best_epsilon = epsilon
        best_value = initial
        history = [{"step": 0, "objective": initial, "epsilon_l2": float(jnp.linalg.norm(epsilon))}]
        for step in range(1, n_steps + 1):
            value, gradient = jax.value_and_grad(objective)(epsilon)
            del value
            direction = gradient / (jnp.linalg.norm(gradient) + 1e-30)
            epsilon = project_l2_ball(epsilon + step_size * direction, radius)
            current = float(objective(epsilon))
            if current > best_value:
                best_value = current
                best_epsilon = epsilon
            history.append(
                {
                    "step": step,
                    "objective": current,
                    "best_objective": best_value,
                    "epsilon_l2": float(jnp.linalg.norm(epsilon)),
                    "gradient_l2": float(jnp.linalg.norm(gradient)),
                }
            )
        results.append(
            (restart_index, initial, float(objective(epsilon)), best_value, best_epsilon, history)
        )
    winner = max(results, key=lambda row: row[3])
    best = np.asarray(winner[4], dtype=np.float64)
    summaries = tuple(
        {
            "restart_index": index,
            "initial_objective": initial,
            "final_objective": final,
            "best_objective": best_value,
            "best_epsilon_l2": float(jnp.linalg.norm(best_epsilon)),
        }
        for index, initial, final, best_value, best_epsilon, _ in results
    )
    return EpsilonOptimizationResult(
        epsilon=best,
        objective=winner[3],
        initial_objective=winner[1],
        final_objective=winner[2],
        energy=epsilon_energy(best),
        l2_norm=float(np.linalg.norm(best)),
        restart_index=winner[0],
        history=tuple(winner[5]),
        restart_summaries=summaries,
    )


def _full_qrf_rollout_cost_context(
    *,
    initial_states: Any,
    target_pos: Any = TARGET_POS,
) -> FullQrfRolloutCostContext:
    _, schedule = build_canonical_game()
    return FullQrfRolloutCostContext(
        initial_states=jnp.asarray(initial_states),
        target_pos=jnp.asarray(target_pos),
        q=jnp.asarray(schedule.Q),
        r=jnp.asarray(schedule.R),
        q_f=jnp.asarray(schedule.Q_f),
    )


def _jax_full_qrf_rollout_cost(
    *,
    states: Any,
    commands: Any,
    context: FullQrfRolloutCostContext | None = None,
    initial_states: Any | None = None,
    target_pos: Any = TARGET_POS,
) -> dict[str, Any]:
    if context is None:
        if initial_states is None:
            raise ValueError("initial_states is required when context is absent")
        context = _full_qrf_rollout_cost_context(
            initial_states=initial_states,
            target_pos=target_pos,
        )
    states = jnp.asarray(states)
    commands = jnp.asarray(commands)
    goal_states = states.at[..., :2].add(-context.target_pos)
    stage = jnp.einsum("...ti,tij,...tj->...t", goal_states, context.q, goal_states)
    control = jnp.einsum("...ti,tij,...tj->...t", commands, context.r, commands)
    terminal = jnp.einsum(
        "...i,ij,...j->...",
        goal_states[..., -1, :],
        context.q_f,
        goal_states[..., -1, :],
    )
    stage_state = jnp.sum(stage, axis=-1)
    control_total = jnp.sum(control, axis=-1)
    return {
        "stage_state": stage_state,
        "control": control_total,
        "terminal": terminal,
        "total": stage_state + control_total + terminal,
    }


def _candidate_flattened_l2_norms(epsilon: np.ndarray) -> np.ndarray:
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim <= 2:
        return np.asarray([np.linalg.norm(eps)], dtype=np.float64)
    return np.linalg.norm(eps.reshape((-1, eps.shape[-2] * eps.shape[-1])), axis=-1)


def _nonfinite_status(epsilon: np.ndarray, selected: float, zero: float) -> dict[str, Any]:
    scalars = np.asarray([selected, zero], dtype=np.float64)
    return {
        "status": "nonfinite"
        if np.any(~np.isfinite(epsilon)) or np.any(~np.isfinite(scalars))
        else "finite",
        "epsilon_has_nan": bool(np.any(np.isnan(epsilon))),
        "epsilon_has_inf": bool(np.any(np.isinf(epsilon))),
        "objective_has_nan": bool(np.any(np.isnan(scalars))),
        "objective_has_inf": bool(np.any(np.isinf(scalars))),
        "epsilon_max_abs": float(np.nanmax(np.abs(epsilon))) if epsilon.size else 0.0,
    }


__all__ = [
    "EpsilonOptimizationResult",
    "WorstCaseEpsilonAnalysisParams",
    "declared_epsilon_l2_radius",
    "epsilon_energy",
    "frozen_batch_adversary_audit_report",
    "optimize_epsilon_sequence",
    "project_l2_ball",
    "register_worst_case_epsilon_recipe",
]

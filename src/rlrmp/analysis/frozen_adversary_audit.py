"""Reusable geometry helpers for frozen finite-adversary audits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from rlrmp.train.closed_loop_finite_adversary import AFFINE_POLICY, LINEAR_NO_BIAS_POLICY

PER_TRIAL_LINEAR_NO_BIAS_POLICY = "per_trial_linear_no_bias"

FrozenFinitePolicyClass = Literal[
    "linear_no_bias",
    "affine",
    "per_trial_linear_no_bias",
]
EnergyReduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class PseudoinverseQuadraticSummary:
    """Summary of ``gradient.T @ G^+ @ gradient`` on a block metric."""

    value: float
    rank: int
    nullity: int
    cutoff: float
    ridge: float
    max_retained_eigenvalue: float | None
    min_retained_eigenvalue: float | None
    condition_number: float | None
    method: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "value": _json_float(self.value),
            "rank": int(self.rank),
            "nullity": int(self.nullity),
            "cutoff": _json_float(self.cutoff),
            "ridge": _json_float(self.ridge),
            "max_retained_eigenvalue": _json_float_or_none(self.max_retained_eigenvalue),
            "min_retained_eigenvalue": _json_float_or_none(self.min_retained_eigenvalue),
            "condition_number": _json_float_or_none(self.condition_number),
            "method": self.method,
        }


@dataclass(frozen=True)
class GradientPressureSummary:
    """Scale of the zero-start gradient in the realized-epsilon metric."""

    pressure_scale: float
    radius: float
    quadratic_form: PseudoinverseQuadraticSummary

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "pressure_scale": _json_float(self.pressure_scale),
            "radius": _json_float(self.radius),
            "quadratic_form": self.quadratic_form.to_json(),
            "formula": "sqrt(g^T G^+ g) / (2r)",
        }


@dataclass(frozen=True)
class GeneralizedCurvatureSummary:
    """Generalized curvature scale for ``lambda_star = 0.5 * lambda_max(H, G)``."""

    lambda_star: float
    max_generalized_eigenvalue: float
    status: Literal["finite", "infinite", "empty_support"]
    rank: int
    nullity: int
    cutoff: float
    null_curvature_max: float | None
    null_cross_norm: float | None
    condition_number: float | None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "lambda_star": _json_float(self.lambda_star),
            "max_generalized_eigenvalue": _json_float(self.max_generalized_eigenvalue),
            "status": self.status,
            "rank": int(self.rank),
            "nullity": int(self.nullity),
            "cutoff": _json_float(self.cutoff),
            "null_curvature_max": _json_float_or_none(self.null_curvature_max),
            "null_cross_norm": _json_float_or_none(self.null_cross_norm),
            "condition_number": _json_float_or_none(self.condition_number),
            "singular_metric_handling": (
                "finite support is whitened by G; positive curvature or H/G cross terms "
                "in the G-null space are reported as infinite"
            ),
        }


def realized_epsilon_energy(
    epsilon: Float[Array, "batch time epsilon_dim"],
    *,
    time_mask: Any | None = None,
    trial_weights: Any | None = None,
    reduction: EnergyReduction = "mean",
) -> Float[Array, ""] | Float[Array, " batch"]:
    """Return epsilon L2 energy with batch-mean semantics by default.

    The energy is ``sum_t,d epsilon[b, t, d]^2`` per trial. The default returns
    the mean across trials, matching the frozen-audit convention used to guard
    against accidental batch-size scaling.
    """

    epsilon_array = jnp.asarray(epsilon)
    if epsilon_array.ndim != 3:
        raise ValueError(
            f"epsilon must have shape (batch, time, epsilon_dim); got {epsilon_array.shape}"
        )
    mask = _time_mask(time_mask, horizon=epsilon_array.shape[1], dtype=epsilon_array.dtype)
    per_trial = jnp.sum(jnp.square(epsilon_array) * mask[None, :, None], axis=(-2, -1))
    if reduction == "none":
        return per_trial
    if reduction == "sum":
        return jnp.sum(per_trial)
    if reduction != "mean":
        raise ValueError(f"unknown energy reduction {reduction!r}")
    if trial_weights is None:
        return jnp.mean(per_trial)
    weights = _trial_weights(trial_weights, batch=epsilon_array.shape[0], dtype=epsilon_array.dtype)
    return jnp.sum(per_trial * weights)


def finite_policy_design_features(
    features: Float[Array, "batch time feature_dim"],
    policy_class: FrozenFinitePolicyClass,
) -> Float[Array, "batch time feature_dim_or_augmented"]:
    """Return metric design features for a finite policy family."""

    feature_array = _batch_features(features)
    if policy_class == AFFINE_POLICY:
        ones = jnp.ones((*feature_array.shape[:-1], 1), dtype=feature_array.dtype)
        return jnp.concatenate([feature_array, ones], axis=-1)
    if policy_class in (LINEAR_NO_BIAS_POLICY, PER_TRIAL_LINEAR_NO_BIAS_POLICY):
        return feature_array
    raise ValueError(f"unknown finite policy class {policy_class!r}")


def finite_policy_epsilon_from_parameters(
    features: Float[Array, "batch time feature_dim"],
    gains: Float[Array, "... time epsilon_dim feature_dim"],
    *,
    bias: Float[Array, "time epsilon_dim"] | None = None,
    policy_class: FrozenFinitePolicyClass = LINEAR_NO_BIAS_POLICY,
) -> Float[Array, "batch time epsilon_dim"]:
    """Evaluate a shared or per-trial finite epsilon policy on frozen features."""

    feature_array = _batch_features(features)
    gain_array = jnp.asarray(gains)
    if policy_class == PER_TRIAL_LINEAR_NO_BIAS_POLICY:
        expected = (
            feature_array.shape[0],
            feature_array.shape[1],
            gain_array.shape[-2],
            feature_array.shape[2],
        )
        if gain_array.shape != expected:
            raise ValueError(f"per-trial gains must have shape {expected}; got {gain_array.shape}")
        if bias is not None:
            raise ValueError("per_trial_linear_no_bias does not accept a bias")
        return jnp.einsum("btf,btef->bte", feature_array, gain_array)

    expected = (feature_array.shape[1], gain_array.shape[-2], feature_array.shape[2])
    if gain_array.ndim != 3 or gain_array.shape != expected:
        raise ValueError(f"shared gains must have shape {expected}; got {gain_array.shape}")
    epsilon = jnp.einsum("btf,tef->bte", feature_array, gain_array)
    if policy_class == AFFINE_POLICY:
        if bias is None:
            raise ValueError("affine policy requires a bias")
        bias_array = jnp.asarray(bias, dtype=epsilon.dtype)
        if bias_array.shape != gain_array.shape[:2]:
            raise ValueError(f"bias must have shape {gain_array.shape[:2]}; got {bias_array.shape}")
        return epsilon + bias_array[None, :, :]
    if policy_class != LINEAR_NO_BIAS_POLICY:
        raise ValueError(f"unknown finite policy class {policy_class!r}")
    if bias is not None:
        raise ValueError("linear_no_bias does not accept a bias")
    return epsilon


def shared_policy_parameter_matrix(
    gains: Float[Array, "time epsilon_dim feature_dim"],
    *,
    bias: Float[Array, "time epsilon_dim"] | None = None,
    policy_class: Literal["linear_no_bias", "affine"] = LINEAR_NO_BIAS_POLICY,
) -> Float[Array, "time epsilon_dim feature_dim_or_augmented"]:
    """Return shared finite-policy parameters in the same basis as metric blocks."""

    gain_array = jnp.asarray(gains)
    if gain_array.ndim != 3:
        raise ValueError(
            f"gains must have shape (time, epsilon_dim, feature_dim); got {gain_array.shape}"
        )
    if policy_class == AFFINE_POLICY:
        if bias is None:
            raise ValueError("affine policy requires a bias")
        bias_array = jnp.asarray(bias, dtype=gain_array.dtype)
        if bias_array.shape != gain_array.shape[:2]:
            raise ValueError(f"bias must have shape {gain_array.shape[:2]}; got {bias_array.shape}")
        return jnp.concatenate([gain_array, bias_array[..., None]], axis=-1)
    if policy_class != LINEAR_NO_BIAS_POLICY:
        raise ValueError(f"unknown shared finite policy class {policy_class!r}")
    if bias is not None:
        raise ValueError("linear_no_bias does not accept a bias")
    return gain_array


def shared_policy_energy_metric_blocks(
    features: Float[Array, "batch time feature_dim"],
    *,
    policy_class: Literal["linear_no_bias", "affine"] = LINEAR_NO_BIAS_POLICY,
    time_mask: Any | None = None,
    trial_weights: Any | None = None,
) -> Float[Array, "time feature_dim_or_augmented feature_dim_or_augmented"]:
    """Return per-time Gram blocks for a shared finite epsilon policy.

    For parameters ``theta[t, epsilon_dim, feature_dim]``, energy is
    ``sum_t,e theta[t, e].T @ G[t] @ theta[t, e]``.
    """

    design = finite_policy_design_features(features, policy_class)
    mask = _time_mask(time_mask, horizon=design.shape[1], dtype=design.dtype)
    weights = _trial_weights(trial_weights, batch=design.shape[0], dtype=design.dtype)
    block_weights = weights[:, None] * mask[None, :]
    return jnp.einsum("btf,btg,bt->tfg", design, design, block_weights)


def per_trial_linear_energy_metric_blocks(
    features: Float[Array, "batch time feature_dim"],
    *,
    time_mask: Any | None = None,
    trial_weights: Any | None = None,
) -> Float[Array, "batch time feature_dim feature_dim"]:
    """Return per-trial/time Gram blocks for unshared no-bias linear policies."""

    feature_array = _batch_features(features)
    mask = _time_mask(time_mask, horizon=feature_array.shape[1], dtype=feature_array.dtype)
    weights = _trial_weights(trial_weights, batch=feature_array.shape[0], dtype=feature_array.dtype)
    block_weights = weights[:, None] * mask[None, :]
    return jnp.einsum("btf,btg,bt->btfg", feature_array, feature_array, block_weights)


def energy_from_shared_metric_blocks(
    parameters: Float[Array, "time epsilon_dim feature_dim_or_augmented"],
    metric_blocks: Float[Array, "time feature_dim_or_augmented feature_dim_or_augmented"],
) -> Float[Array, ""]:
    """Evaluate shared-policy energy from realized-epsilon metric blocks."""

    theta = jnp.asarray(parameters)
    blocks = jnp.asarray(metric_blocks, dtype=theta.dtype)
    if theta.ndim != 3 or blocks.ndim != 3:
        raise ValueError("parameters and metric_blocks must both be rank-3 arrays")
    if blocks.shape[0] != theta.shape[0] or blocks.shape[1:] != (theta.shape[2], theta.shape[2]):
        raise ValueError(
            "metric blocks must have shape (time, parameter_dim, parameter_dim); "
            f"got blocks={blocks.shape}, parameters={theta.shape}"
        )
    return jnp.einsum("tef,tfg,teg->", theta, blocks, theta)


def energy_from_per_trial_metric_blocks(
    gains: Float[Array, "batch time epsilon_dim feature_dim"],
    metric_blocks: Float[Array, "batch time feature_dim feature_dim"],
) -> Float[Array, ""]:
    """Evaluate unshared per-trial linear energy from metric blocks."""

    gain_array = jnp.asarray(gains)
    blocks = jnp.asarray(metric_blocks, dtype=gain_array.dtype)
    if gain_array.ndim != 4 or blocks.ndim != 4:
        raise ValueError("gains and metric_blocks must both be rank-4 arrays")
    if blocks.shape[:2] != gain_array.shape[:2] or blocks.shape[2:] != (
        gain_array.shape[3],
        gain_array.shape[3],
    ):
        raise ValueError(
            "per-trial metric blocks must have shape (batch, time, feature_dim, feature_dim); "
            f"got blocks={blocks.shape}, gains={gain_array.shape}"
        )
    return jnp.einsum("btef,btfg,bteg->", gain_array, blocks, gain_array)


def dense_shared_metric_from_blocks(
    metric_blocks: Float[Array, "time feature_dim feature_dim"],
    *,
    epsilon_dim: int,
) -> Float[Array, "flat_params flat_params"]:
    """Materialize a dense block metric for small frozen curvature audits."""

    blocks = np.asarray(metric_blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError(
            f"metric_blocks must have shape (time, feature_dim, feature_dim); got {blocks.shape}"
        )
    epsilon_dim = int(epsilon_dim)
    if epsilon_dim < 1:
        raise ValueError("epsilon_dim must be positive")
    horizon, parameter_dim, _ = blocks.shape
    dense = np.zeros((horizon * epsilon_dim * parameter_dim,) * 2, dtype=blocks.dtype)
    for time_index in range(horizon):
        for epsilon_index in range(epsilon_dim):
            offset = (time_index * epsilon_dim + epsilon_index) * parameter_dim
            sl = slice(offset, offset + parameter_dim)
            dense[sl, sl] = blocks[time_index]
    return jnp.asarray(dense)


def pseudoinverse_metric_quadratic_form(
    gradient: Float[Array, "time epsilon_dim feature_dim_or_augmented"],
    metric_blocks: Float[Array, "time feature_dim_or_augmented feature_dim_or_augmented"],
    *,
    rcond: float = 1e-10,
    ridge: float = 0.0,
) -> PseudoinverseQuadraticSummary:
    """Compute ``g.T @ G^+ @ g`` without materializing repeated epsilon blocks."""

    grad = np.asarray(gradient, dtype=np.float64)
    blocks = np.asarray(metric_blocks, dtype=np.float64)
    if grad.ndim != 3 or blocks.ndim != 3:
        raise ValueError("gradient and metric_blocks must both be rank-3 arrays")
    if blocks.shape[0] != grad.shape[0] or blocks.shape[1:] != (grad.shape[2], grad.shape[2]):
        raise ValueError(
            "metric blocks must have shape (time, parameter_dim, parameter_dim); "
            f"got blocks={blocks.shape}, gradient={grad.shape}"
        )
    rcond = float(rcond)
    ridge = float(ridge)
    if rcond < 0.0:
        raise ValueError("rcond must be nonnegative")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")

    value = 0.0
    total_rank = 0
    total_nullity = 0
    max_cutoff = 0.0
    retained: list[float] = []
    method = "ridge" if ridge else "pseudoinverse"
    for time_index, block in enumerate(blocks):
        evals, evecs = np.linalg.eigh(_symmetrize(block))
        max_eval = float(np.max(np.abs(evals))) if evals.size else 0.0
        cutoff = rcond * max_eval
        max_cutoff = max(max_cutoff, cutoff)
        keep = evals > cutoff
        retained.extend(float(v) for v in evals[keep])
        total_rank += int(np.count_nonzero(keep)) * grad.shape[1]
        total_nullity += int(np.count_nonzero(~keep)) * grad.shape[1]

        coords = grad[time_index] @ evecs
        if ridge:
            denom = np.maximum(evals, 0.0) + ridge
            value += float(np.sum(np.square(coords) / denom[None, :]))
        elif np.any(keep):
            value += float(np.sum(np.square(coords[:, keep]) / evals[keep][None, :]))

    max_retained = max(retained) if retained else None
    min_retained = min(retained) if retained else None
    condition = None
    if max_retained is not None and min_retained is not None and min_retained > 0.0:
        condition = max_retained / min_retained
    return PseudoinverseQuadraticSummary(
        value=float(value),
        rank=int(total_rank),
        nullity=int(total_nullity),
        cutoff=float(max_cutoff),
        ridge=float(ridge),
        max_retained_eigenvalue=max_retained,
        min_retained_eigenvalue=min_retained,
        condition_number=condition,
        method=method,
    )


def gradient_pressure_scale(
    gradient: Float[Array, "time epsilon_dim feature_dim_or_augmented"],
    metric_blocks: Float[Array, "time feature_dim_or_augmented feature_dim_or_augmented"],
    *,
    radius: float,
    rcond: float = 1e-10,
    ridge: float = 0.0,
) -> GradientPressureSummary:
    """Return ``sqrt(g.T @ G^+ @ g) / (2r)`` for a soft-energy radius ``r``."""

    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    q_summary = pseudoinverse_metric_quadratic_form(
        gradient,
        metric_blocks,
        rcond=rcond,
        ridge=ridge,
    )
    pressure = math.sqrt(max(q_summary.value, 0.0)) / (2.0 * radius)
    return GradientPressureSummary(
        pressure_scale=float(pressure),
        radius=radius,
        quadratic_form=q_summary,
    )


def generalized_curvature_lambda_star(
    hessian: Float[Array, "flat_params flat_params"],
    metric: Float[Array, "flat_params flat_params"],
    *,
    rcond: float = 1e-10,
    null_tol: float = 1e-9,
) -> GeneralizedCurvatureSummary:
    """Estimate ``0.5 * lambda_max(H, G)`` with explicit singular-G handling."""

    hessian_array = np.asarray(hessian, dtype=np.float64)
    metric_array = np.asarray(metric, dtype=np.float64)
    if hessian_array.ndim != 2 or hessian_array.shape[0] != hessian_array.shape[1]:
        raise ValueError(f"hessian must be square; got {hessian_array.shape}")
    if metric_array.shape != hessian_array.shape:
        raise ValueError(
            f"metric shape {metric_array.shape} does not match Hessian {hessian_array.shape}"
        )
    hessian_array = _symmetrize(hessian_array)
    metric_array = _symmetrize(metric_array)
    evals, evecs = np.linalg.eigh(metric_array)
    max_eval = float(np.max(np.abs(evals))) if evals.size else 0.0
    cutoff = float(rcond) * max_eval
    support = evals > cutoff
    rank = int(np.count_nonzero(support))
    nullity = int(evals.size - rank)
    condition = None
    if rank:
        retained = evals[support]
        condition = float(np.max(retained) / np.min(retained))

    null_curvature_max = None
    null_cross_norm = None
    if nullity:
        null_basis = evecs[:, ~support]
        null_h = _symmetrize(null_basis.T @ hessian_array @ null_basis)
        null_curvature_max = float(np.max(np.linalg.eigvalsh(null_h)))
        if rank:
            support_basis = evecs[:, support]
            null_cross_norm = float(np.linalg.norm(null_basis.T @ hessian_array @ support_basis))
        else:
            null_cross_norm = 0.0
        if null_curvature_max > null_tol or null_cross_norm > null_tol:
            return GeneralizedCurvatureSummary(
                lambda_star=math.inf,
                max_generalized_eigenvalue=math.inf,
                status="infinite",
                rank=rank,
                nullity=nullity,
                cutoff=cutoff,
                null_curvature_max=null_curvature_max,
                null_cross_norm=null_cross_norm,
                condition_number=condition,
            )

    if not rank:
        return GeneralizedCurvatureSummary(
            lambda_star=0.0,
            max_generalized_eigenvalue=0.0,
            status="empty_support",
            rank=rank,
            nullity=nullity,
            cutoff=cutoff,
            null_curvature_max=null_curvature_max,
            null_cross_norm=null_cross_norm,
            condition_number=condition,
        )

    support_basis = evecs[:, support]
    whitened = support_basis / np.sqrt(evals[support])[None, :]
    projected = _symmetrize(whitened.T @ hessian_array @ whitened)
    max_generalized = float(np.max(np.linalg.eigvalsh(projected)))
    return GeneralizedCurvatureSummary(
        lambda_star=0.5 * max_generalized,
        max_generalized_eigenvalue=max_generalized,
        status="finite",
        rank=rank,
        nullity=nullity,
        cutoff=cutoff,
        null_curvature_max=null_curvature_max,
        null_cross_norm=null_cross_norm,
        condition_number=condition,
    )


def summarize_active_broad_epsilon_optimizer(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the actually active broad-epsilon optimizer lane in a run spec."""

    hps = _mapping(run_spec.get("hps"))
    broad = _mapping(hps.get("broad_epsilon_pgd_training"))
    policy = _mapping(hps.get("policy_adversary_training"))
    policy_payload = _mapping(policy.get("policy"))
    inner = _mapping(broad.get("inner_maximizer"))
    objective = _mapping(broad.get("objective"))
    safety_cap = _mapping(broad.get("safety_cap"))
    policy_inner = _mapping(policy.get("inner_optimizer"))
    broad_enabled = bool(broad.get("enabled", False))
    policy_enabled = bool(policy.get("enabled", False))

    warnings: list[str] = []
    active_lane = None
    active_method = None
    if broad_enabled:
        active_lane = "broad_epsilon_pgd_training.inner_maximizer"
        active_method = inner.get("method")
        active_mechanism = broad.get("adversary_mechanism")
    elif policy_enabled:
        active_lane = "policy_adversary_training.inner_optimizer"
        active_method = policy_inner.get("method")
        active_mechanism = policy.get("policy_class") or policy_payload.get("kind")
    else:
        active_mechanism = None
        warnings.append("no finite/broad epsilon optimizer lane is enabled")

    if broad_enabled and policy_inner.get("method") is not None and not policy_enabled:
        warnings.append(
            "policy_adversary_training.inner_optimizer is inactive metadata; "
            "do not treat it as the launched finite optimizer"
        )
    if broad_enabled and active_method != "adam":
        warnings.append(
            "active broad-epsilon finite path is not Adam; compare against intended Adam separately"
        )
    if broad_enabled and broad.get("adversary_mechanism") in {LINEAR_NO_BIAS_POLICY, AFFINE_POLICY}:
        mechanism = _mapping(broad.get("mechanism"))
        if not mechanism.get("no_fake_open_loop_replay", False):
            warnings.append("finite mechanism metadata does not explicitly forbid open-loop replay")
    if policy_enabled and active_method != "adam":
        warnings.append("active policy-adversary finite path is not Adam")
    if (
        policy_enabled
        and active_mechanism in {LINEAR_NO_BIAS_POLICY, AFFINE_POLICY}
        and policy_payload.get("closed_loop_semantics_status") == "not_live_rollout_hook"
    ):
        warnings.append(
            "finite policy uses Adam over finite parameters, but current training integration "
            "materializes static epsilon from clean-rollout features instead of a live rollout hook"
        )

    return {
        "active_lane": active_lane,
        "active_method": active_method,
        "active_enabled": bool(active_lane),
        "active_mechanism": active_mechanism,
        "active_mode": broad.get("mode") if broad_enabled else policy.get("row_mode"),
        "active_policy_evaluation_semantics": policy_payload.get("evaluation_semantics"),
        "active_policy_closed_loop_semantics_status": policy_payload.get(
            "closed_loop_semantics_status"
        ),
        "active_initialization": inner.get("initialization"),
        "active_n_steps": inner.get("n_steps"),
        "active_step_size_fraction_of_l2_radius": inner.get("step_size_fraction_of_l2_radius"),
        "active_projection": inner.get("projection"),
        "objective_kind": objective.get("kind"),
        "lambda": objective.get("lambda"),
        "safety_cap_enabled": safety_cap.get("enabled"),
        "safety_cap_l2_radius_15cm": safety_cap.get("l2_radius_15cm"),
        "policy_adversary_training_enabled": policy_enabled,
        "inactive_policy_adam_metadata": {
            "present": bool(policy_inner),
            "mode": policy.get("mode"),
            "enabled": policy_enabled,
            "method": policy_inner.get("method"),
            "learning_rate": policy_inner.get("learning_rate"),
            "n_ascent_steps_per_controller_step": policy_inner.get(
                "n_ascent_steps_per_controller_step"
            ),
            "weights_persist_across_batches": policy_inner.get("weights_persist_across_batches"),
        },
        "warnings": warnings,
    }


def _batch_features(features: Any) -> Float[Array, "batch time feature_dim"]:
    feature_array = jnp.asarray(features)
    if feature_array.ndim != 3:
        raise ValueError(
            f"features must have shape (batch, time, feature_dim); got {feature_array.shape}"
        )
    return feature_array


def _time_mask(
    time_mask: Any | None,
    *,
    horizon: int,
    dtype: Any,
) -> Float[Array, " time"]:
    if time_mask is None:
        return jnp.ones((horizon,), dtype=dtype)
    mask = jnp.asarray(time_mask, dtype=dtype)
    if mask.shape != (horizon,):
        raise ValueError(f"time_mask must have shape ({horizon},); got {mask.shape}")
    return mask


def _trial_weights(
    trial_weights: Any | None,
    *,
    batch: int,
    dtype: Any,
) -> Float[Array, " batch"]:
    if trial_weights is None:
        return jnp.ones((batch,), dtype=dtype) / float(batch)
    weights = jnp.asarray(trial_weights, dtype=dtype)
    if weights.shape != (batch,):
        raise ValueError(f"trial_weights must have shape ({batch},); got {weights.shape}")
    return weights / jnp.sum(weights)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _json_float(value: float) -> float | str:
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def _json_float_or_none(value: float | None) -> float | str | None:
    if value is None:
        return None
    return _json_float(value)

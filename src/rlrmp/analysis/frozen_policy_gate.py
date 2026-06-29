"""Reusable row schemas and helpers for regenerated frozen-policy gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.frozen_adversary_audit import (
    AFFINE_POLICY,
    LINEAR_NO_BIAS_POLICY,
    pseudoinverse_metric_quadratic_form,
    realized_epsilon_energy,
    shared_policy_energy_metric_blocks,
)

DIRECT_EPSILON_MECHANISM = "direct_epsilon"
FrozenGateMechanism = Literal["direct_epsilon", "linear_no_bias", "affine"]


@dataclass(frozen=True)
class FrozenBatchDescriptor:
    """Reproducibility descriptor for a regenerated frozen audit batch."""

    source_issue: str
    source_run: str
    checkpoint_path: str
    checkpoint_batches: int
    checkpoint_metadata_sha256: str
    run_spec_sha256: str
    replicate_index: int
    batch_size: int
    batch_index: int
    root_key: list[int]
    key_trials: list[int]
    key_model: list[int]
    key_optimizer: list[int]
    task_distribution: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "source_issue": self.source_issue,
            "source_run": self.source_run,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_batches": int(self.checkpoint_batches),
            "checkpoint_metadata_sha256": self.checkpoint_metadata_sha256,
            "run_spec_sha256": self.run_spec_sha256,
            "replicate_index": int(self.replicate_index),
            "batch_size": int(self.batch_size),
            "batch_index": int(self.batch_index),
            "root_key": [int(v) for v in self.root_key],
            "key_trials": [int(v) for v in self.key_trials],
            "key_model": [int(v) for v in self.key_model],
            "key_optimizer": [int(v) for v in self.key_optimizer],
            "task_distribution": self.task_distribution,
        }


@dataclass(frozen=True)
class FrozenOptimizerConfig:
    """JSON-facing description of the frozen inner optimizer."""

    method: str
    learning_rate: float
    n_steps: int
    selected_by: str = "best_finite_objective"

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "learning_rate": _json_float(self.learning_rate),
            "n_steps": int(self.n_steps),
            "selected_by": self.selected_by,
        }


@dataclass(frozen=True)
class FrozenAuditRow:
    """JSON-facing row for one mechanism on one frozen batch."""

    mechanism: FrozenGateMechanism
    lambda_input: float
    objective_at_zero: float
    task_loss_at_zero: float
    gradient_norm: float
    gradient_pressure_scale: float
    metric_geometry: dict[str, Any]
    curvature: dict[str, Any]
    optimizer: dict[str, Any]
    selected_energy: float
    selected_objective: float
    accepted_objective_gain: float
    cap_behavior: dict[str, Any]
    nonfinite: dict[str, bool]
    batch_size_invariance: dict[str, Any]
    tensor_artifact: str

    def to_json(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "lambda_input": _json_float(self.lambda_input),
            "objective_at_zero": _json_float(self.objective_at_zero),
            "task_loss_at_zero": _json_float(self.task_loss_at_zero),
            "gradient_norm": _json_float(self.gradient_norm),
            "gradient_pressure_scale": _json_float(self.gradient_pressure_scale),
            "metric_geometry": self.metric_geometry,
            "curvature": self.curvature,
            "optimizer": self.optimizer,
            "selected_energy": _json_float(self.selected_energy),
            "selected_objective": _json_float(self.selected_objective),
            "accepted_objective_gain": _json_float(self.accepted_objective_gain),
            "cap_behavior": self.cap_behavior,
            "nonfinite": self.nonfinite,
            "batch_size_invariance": self.batch_size_invariance,
            "tensor_artifact": self.tensor_artifact,
        }


def sha256_json(payload: Any) -> str:
    """Return a stable SHA256 digest for a JSON-compatible payload."""

    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_epsilon_invariance(
    epsilon: Any,
    *,
    time_mask: Any | None = None,
) -> dict[str, Any]:
    """Return mean/sum reduction invariance for a selected epsilon tensor."""

    eps = jnp.asarray(epsilon)
    duplicated = jnp.concatenate([eps, eps], axis=0)
    mean_original = realized_epsilon_energy(eps, time_mask=time_mask, reduction="mean")
    mean_duplicated = realized_epsilon_energy(duplicated, time_mask=time_mask, reduction="mean")
    sum_original = realized_epsilon_energy(eps, time_mask=time_mask, reduction="sum")
    sum_duplicated = realized_epsilon_energy(duplicated, time_mask=time_mask, reduction="sum")
    mean_original_f = float(np.asarray(mean_original))
    mean_duplicated_f = float(np.asarray(mean_duplicated))
    sum_original_f = float(np.asarray(sum_original))
    sum_duplicated_f = float(np.asarray(sum_duplicated))
    return {
        "mean_reduction_original": _json_float(mean_original_f),
        "mean_reduction_duplicated": _json_float(mean_duplicated_f),
        "mean_reduction_invariant": bool(np.isclose(mean_original_f, mean_duplicated_f)),
        "sum_reduction_original": _json_float(sum_original_f),
        "sum_reduction_duplicated": _json_float(sum_duplicated_f),
        "sum_reduction_ratio": _json_float(
            sum_duplicated_f / sum_original_f if sum_original_f else math.nan
        ),
        "interpretation": (
            "mean-reduced energy is the gate quantity; sum reduction doubles under "
            "batch duplication and is recorded only as a scaling check"
        ),
    }


def metric_geometry_summary(
    mechanism: FrozenGateMechanism,
    *,
    features: Any,
    epsilon_dim: int,
    gradient: Any,
    radius: float,
    time_mask: Any | None = None,
) -> tuple[dict[str, Any], float]:
    """Return metric geometry and gradient-pressure summaries for one mechanism."""

    grad = jnp.asarray(gradient)
    if mechanism == DIRECT_EPSILON_MECHANISM:
        batch = int(grad.shape[0])
        quadratic = float(batch * np.sum(np.square(np.asarray(grad, dtype=np.float64))))
        pressure = math.sqrt(max(quadratic, 0.0)) / (2.0 * float(radius))
        return (
            {
                "metric": "per-trial direct epsilon mean-energy metric",
                "rank": int(np.size(grad)),
                "nullity": 0,
                "condition_number": 1.0,
                "quadratic_form_g_Gplus_g": _json_float(quadratic),
                "assumption": (
                    "Direct epsilon pressure treats the per-trial trust-region radius "
                    "as a shared energy radius for the mean-reduced batch metric."
                ),
            },
            float(pressure),
        )

    policy_class = AFFINE_POLICY if mechanism == AFFINE_POLICY else LINEAR_NO_BIAS_POLICY
    blocks = shared_policy_energy_metric_blocks(
        features,
        policy_class=policy_class,
        time_mask=time_mask,
    )
    q_summary = pseudoinverse_metric_quadratic_form(grad, blocks)
    pressure = math.sqrt(max(q_summary.value, 0.0)) / (2.0 * float(radius))
    return (
        {
            "metric": "shared finite-policy realized-epsilon mean-energy metric",
            "rank": int(q_summary.rank),
            "nullity": int(q_summary.nullity),
            "condition_number": _json_float_or_none(q_summary.condition_number),
            "quadratic_form": q_summary.to_json(),
        },
        float(pressure),
    )


def directional_curvature_summary(
    *,
    hvp_directional_ratio: float,
    n_hvp: int,
    method: str,
) -> dict[str, Any]:
    """Return the standard approximate curvature metadata."""

    ratio = float(hvp_directional_ratio)
    return {
        "method": method,
        "hvp_directional_ratio": _json_float(ratio),
        "lambda_star_directional": _json_float(0.5 * ratio),
        "n_hvp": int(n_hvp),
        "status": "directional_approximation",
        "assumption": (
            "This is a gradient-direction HVP diagnostic, not a dense generalized "
            "eigen solve. Trust it as a local scale check; use Lanczos/dense support "
            "eigensolve if this value gates a final training lambda."
        ),
    }


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

"""Black-box controller-local policy diagnostics.

The helpers in this module operate on policy callables of the form
``policy(blocks) -> action`` where ``blocks`` is a mapping from named,
controller-visible input blocks to JAX arrays. They do not require a plant,
closed-loop rollout, recurrent state materializer, or Feedbax graph adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

POLICY_DIAGNOSTICS_SCHEMA_VERSION = 1
AVAILABLE = "available"
NOT_APPLICABLE = "not_applicable"


BlockValues = Mapping[str, Array]
PolicyCallable = Callable[[BlockValues], Array]


def _matricize_flat_input_block(block: Array, flat_input: Array) -> Array:
    """Reshape a dense Jacobian block with trailing flat input axes into 2D."""
    in_dim = int(np.asarray(flat_input).size)
    return jnp.asarray(block).reshape((-1, in_dim))


@dataclass(frozen=True)
class PolicyInputBlock:
    """A present controller-visible input block.

    Args:
        name: Stable block name, such as ``"feedback"``, ``"sisu"``, or
            ``"context"``.
        shape: Array shape for this block before flattening.
        role: Semantic role used by downstream summaries.
        status: Availability status. Present blocks normally use
            ``"available"``.
        interpretation: Optional human-readable interpretation of the block.
        metadata: Extra JSON-like metadata for downstream diagnostics.
    """

    name: str
    shape: tuple[int, ...]
    role: str = "input"
    status: str = AVAILABLE
    interpretation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PolicyInputBlock.name must be non-empty.")
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"Block {self.name!r} shape cannot contain negative dimensions.")

    @property
    def size(self) -> int:
        """Flattened block width."""
        return int(np.prod(self.shape, dtype=np.int64))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-like schema payload for this block."""
        payload: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "shape": list(self.shape),
            "size": self.size,
        }
        if self.interpretation is not None:
            payload["interpretation"] = self.interpretation
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class PolicyAbsentInputBlock:
    """A named input block that is intentionally absent for this row."""

    name: str
    role: str
    reason: str
    status: str = NOT_APPLICABLE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PolicyAbsentInputBlock.name must be non-empty.")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-like schema payload for this absent block."""
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyInputSchema:
    """Explicit schema for controller-visible policy input blocks."""

    blocks: tuple[PolicyInputBlock, ...]
    absent_blocks: tuple[PolicyAbsentInputBlock, ...] = ()
    schema_version: int = POLICY_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "absent_blocks", tuple(self.absent_blocks))
        names = [block.name for block in self.blocks]
        absent_names = [block.name for block in self.absent_blocks]
        if len(names) != len(set(names)):
            raise ValueError("PolicyInputSchema blocks must have unique names.")
        if set(names) & set(absent_names):
            overlap = sorted(set(names) & set(absent_names))
            raise ValueError(f"Blocks cannot be both present and absent: {overlap}.")
        if len(absent_names) != len(set(absent_names)):
            raise ValueError("PolicyInputSchema absent blocks must have unique names.")

    @classmethod
    def from_values(
        cls,
        values: BlockValues,
        *,
        roles: Mapping[str, str] | None = None,
        interpretations: Mapping[str, str] | None = None,
        absent_blocks: Sequence[PolicyAbsentInputBlock] = (),
    ) -> "PolicyInputSchema":
        """Build a schema from concrete input block arrays.

        Args:
            values: Mapping of present input blocks.
            roles: Optional semantic roles keyed by block name.
            interpretations: Optional interpretations keyed by block name.
            absent_blocks: Explicit not-applicable blocks for task families
                where a named channel is intentionally unavailable.

        Returns:
            A schema with one present block per ``values`` entry.
        """
        role_map = dict(roles or {})
        interpretation_map = dict(interpretations or {})
        blocks = tuple(
            PolicyInputBlock(
                name=name,
                shape=tuple(jnp.asarray(value).shape),
                role=role_map.get(name, name),
                interpretation=interpretation_map.get(name),
            )
            for name, value in values.items()
        )
        return cls(blocks=blocks, absent_blocks=tuple(absent_blocks))

    @property
    def block_names(self) -> tuple[str, ...]:
        """Present block names in flattening order."""
        return tuple(block.name for block in self.blocks)

    @property
    def size(self) -> int:
        """Total flattened width across present blocks."""
        return sum(block.size for block in self.blocks)

    def block(self, name: str) -> PolicyInputBlock:
        """Return a present block spec by name."""
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"Unknown present policy input block: {name!r}.")

    def block_slice(self, name: str) -> slice:
        """Return the flattened slice occupied by ``name``."""
        offset = 0
        for block in self.blocks:
            stop = offset + block.size
            if block.name == name:
                return slice(offset, stop)
            offset = stop
        raise KeyError(f"Unknown present policy input block: {name!r}.")

    def flatten(self, values: BlockValues) -> Array:
        """Flatten present input blocks into one vector using schema order."""
        flattened = []
        for block in self.blocks:
            if block.name not in values:
                raise KeyError(f"Missing policy input block: {block.name!r}.")
            value = jnp.asarray(values[block.name])
            if tuple(value.shape) != block.shape:
                raise ValueError(
                    f"Block {block.name!r} has shape {tuple(value.shape)}, "
                    f"expected {block.shape}."
                )
            flattened.append(value.reshape(-1))
        if not flattened:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate(flattened, axis=0)

    def unflatten(self, flat: Array) -> dict[str, Array]:
        """Unflatten a vector into a block mapping using this schema."""
        flat = jnp.asarray(flat)
        if flat.shape != (self.size,):
            raise ValueError(f"Flat input has shape {flat.shape}, expected ({self.size},).")
        values: dict[str, Array] = {}
        for block in self.blocks:
            block_range = self.block_slice(block.name)
            values[block.name] = flat[block_range].reshape(block.shape)
        return values

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-like schema payload."""
        return {
            "schema_version": self.schema_version,
            "blocks": [block.to_json() for block in self.blocks],
            "absent_blocks": [block.to_json() for block in self.absent_blocks],
            "flat_size": self.size,
        }


@dataclass(frozen=True)
class PolicyJacobian:
    """Dense local policy Jacobian and block views."""

    schema: PolicyInputSchema
    output_shape: tuple[int, ...]
    full: Array
    by_block: Mapping[str, Array]

    @property
    def output_size(self) -> int:
        """Flattened action/output width."""
        return int(np.prod(self.output_shape, dtype=np.int64))

    def block(self, name: str) -> Array:
        """Return the dense output-by-block Jacobian for ``name``."""
        return self.by_block[name]

    def to_summary(self) -> dict[str, Any]:
        """Return JSON-like matrix-shape metadata without dumping matrix values."""
        return {
            "schema": self.schema.to_json(),
            "output_shape": list(self.output_shape),
            "output_size": self.output_size,
            "jacobian_shape": list(self.full.shape),
            "blocks": {
                name: {"shape": list(matrix.shape)}
                for name, matrix in self.by_block.items()
            },
        }


@dataclass(frozen=True)
class PolicyFiniteDifferenceValidation:
    """Finite-difference comparison for a local policy Jacobian."""

    analytic: Array
    finite_difference: Array
    max_abs_error: float
    relative_fro_error: float
    passed: bool

    def to_summary(self) -> dict[str, Any]:
        """Return a compact JSON-like validation summary."""
        return {
            "max_abs_error": self.max_abs_error,
            "relative_fro_error": self.relative_fro_error,
            "passed": self.passed,
        }


def policy_jacobian(
    policy: PolicyCallable,
    values: BlockValues,
    *,
    schema: PolicyInputSchema | None = None,
    wrt_blocks: Sequence[str] | None = None,
) -> PolicyJacobian:
    """Compute the dense local Jacobian of a black-box policy.

    Args:
        policy: Callable receiving a mapping of named input blocks and returning
            the controller output/action array.
        values: Concrete input block values at the linearization point.
        schema: Optional explicit schema. When omitted, one is inferred from
            ``values`` with default roles equal to block names.
        wrt_blocks: Optional subset of present blocks to expose in
            ``PolicyJacobian.by_block``. The full matrix always covers all
            present blocks.

    Returns:
        Dense policy Jacobian with shape ``(output_size, flat_input_size)`` and
        per-block matrix views.
    """
    schema = schema or PolicyInputSchema.from_values(values)
    flat0 = schema.flatten(values)
    output0 = jnp.asarray(policy(schema.unflatten(flat0)))
    output_shape = tuple(int(dim) for dim in output0.shape)

    def policy_from_flat(flat: Array) -> Array:
        return jnp.asarray(policy(schema.unflatten(flat)))

    raw_jacobian = jax.jacobian(policy_from_flat)(flat0)
    full = _matricize_flat_input_block(raw_jacobian, flat0)
    block_names = tuple(wrt_blocks) if wrt_blocks is not None else schema.block_names
    by_block = {
        name: full[:, schema.block_slice(name)]
        for name in block_names
    }
    return PolicyJacobian(
        schema=schema,
        output_shape=output_shape,
        full=full,
        by_block=by_block,
    )


def policy_block_jacobian(
    policy: PolicyCallable,
    values: BlockValues,
    block_name: str,
    *,
    schema: PolicyInputSchema | None = None,
) -> Array:
    """Compute a local policy Jacobian with respect to one input block only."""
    schema = schema or PolicyInputSchema.from_values(values)
    block = schema.block(block_name)
    fixed_values = {
        name: jnp.asarray(value)
        for name, value in values.items()
    }
    if block_name not in fixed_values:
        raise KeyError(f"Missing policy input block: {block_name!r}.")
    flat0 = fixed_values[block_name].reshape(-1)

    def policy_from_block(flat_block: Array) -> Array:
        block_values = dict(fixed_values)
        block_values[block_name] = flat_block.reshape(block.shape)
        return jnp.asarray(policy(block_values))

    raw_jacobian = jax.jacobian(policy_from_block)(flat0)
    return _matricize_flat_input_block(raw_jacobian, flat0)


def finite_difference_jacobian(
    policy: PolicyCallable,
    values: BlockValues,
    *,
    schema: PolicyInputSchema | None = None,
    epsilon: float = 1e-3,
    batch_size: int | None = 128,
) -> Array:
    """Central finite-difference estimate of the dense policy Jacobian."""
    schema = schema or PolicyInputSchema.from_values(values)
    flat0 = schema.flatten(values)
    output0 = jnp.asarray(policy(schema.unflatten(flat0))).reshape(-1)
    if flat0.size == 0:
        return jnp.zeros((output0.size, 0), dtype=output0.dtype)

    def output_from_flat(flat: Array) -> Array:
        return jnp.asarray(policy(schema.unflatten(flat))).reshape(-1)

    def column(index: Array) -> Array:
        direction = jax.nn.one_hot(index, flat0.size, dtype=flat0.dtype)
        step = epsilon * direction
        return (output_from_flat(flat0 + step) - output_from_flat(flat0 - step)) / (
            2.0 * epsilon
        )

    indices = jnp.arange(flat0.size, dtype=jnp.int32)
    return jax.lax.map(column, indices, batch_size=batch_size).T


def validate_policy_jacobian(
    policy: PolicyCallable,
    values: BlockValues,
    *,
    schema: PolicyInputSchema | None = None,
    epsilon: float = 1e-3,
    finite_difference_batch_size: int | None = 128,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> PolicyFiniteDifferenceValidation:
    """Compare autodiff and finite-difference local policy Jacobians."""
    schema = schema or PolicyInputSchema.from_values(values)
    analytic = policy_jacobian(policy, values, schema=schema).full
    finite = finite_difference_jacobian(
        policy,
        values,
        schema=schema,
        epsilon=epsilon,
        batch_size=finite_difference_batch_size,
    )
    error = analytic - finite
    max_abs_error = float(jnp.max(jnp.abs(error))) if error.size else 0.0
    denom = jnp.maximum(jnp.linalg.norm(finite), jnp.asarray(1e-12, dtype=finite.dtype))
    relative_fro_error = float(jnp.linalg.norm(error) / denom)
    passed = bool(max_abs_error <= atol + rtol * float(jnp.max(jnp.abs(finite))))
    return PolicyFiniteDifferenceValidation(
        analytic=analytic,
        finite_difference=finite,
        max_abs_error=max_abs_error,
        relative_fro_error=relative_fro_error,
        passed=passed,
    )


def singular_value_summary(matrix: Array, *, tolerance: float = 1e-10) -> dict[str, Any]:
    """Summarize singular values for a dense local policy map."""
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return {
            "status": NOT_APPLICABLE,
            "reason": "matrix has zero output or input width",
            "shape": list(matrix.shape),
        }
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    min_sv = float(jnp.min(singular_values))
    max_sv = float(jnp.max(singular_values))
    return {
        "status": AVAILABLE,
        "shape": list(matrix.shape),
        "singular_values": [float(value) for value in singular_values],
        "spectral_norm": max_sv,
        "min_singular_value": min_sv,
        "frobenius_norm": float(jnp.linalg.norm(matrix)),
        "rank": int(jnp.sum(singular_values > tolerance)),
        "condition_number": float(max_sv / min_sv) if min_sv > tolerance else np.inf,
    }


def directional_gain_summary(
    matrix: Array,
    directions: Array | None = None,
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Summarize output/input norm gains along selected input directions.

    Args:
        matrix: Dense map with shape ``(output_size, input_size)``.
        directions: Optional directions with shape ``(n_directions, input_size)``.
            When omitted, coordinate directions are used.
        epsilon: Norm floor for identifying zero directions.

    Returns:
        JSON-like gain summary.
    """
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}.")
    input_size = int(matrix.shape[1])
    if input_size == 0:
        return {
            "status": NOT_APPLICABLE,
            "reason": "matrix has zero input width",
            "shape": list(matrix.shape),
        }
    if directions is None:
        direction_matrix = jnp.eye(input_size, dtype=matrix.dtype)
        basis = "coordinate"
    else:
        direction_matrix = jnp.asarray(directions, dtype=matrix.dtype)
        basis = "provided"
        if direction_matrix.ndim == 1:
            direction_matrix = direction_matrix[None, :]
        if direction_matrix.shape[1] != input_size:
            raise ValueError(
                f"directions have width {direction_matrix.shape[1]}, expected {input_size}."
            )

    output = direction_matrix @ matrix.T
    input_norm = jnp.linalg.norm(direction_matrix, axis=1)
    output_norm = jnp.linalg.norm(output, axis=1)
    valid = input_norm > epsilon
    gains = jnp.where(valid, output_norm / jnp.maximum(input_norm, epsilon), jnp.nan)
    valid_count = int(jnp.sum(valid))
    if valid_count == 0:
        return {
            "status": NOT_APPLICABLE,
            "reason": "all provided directions have zero norm",
            "shape": list(matrix.shape),
            "basis": basis,
        }
    max_gain = jnp.max(jnp.where(valid, gains, -jnp.inf))
    min_gain = jnp.min(jnp.where(valid, gains, jnp.inf))
    mean_gain = jnp.sum(jnp.where(valid, gains, 0.0)) / valid_count
    return {
        "status": AVAILABLE,
        "shape": list(matrix.shape),
        "basis": basis,
        "gains": [float(value) for value in gains],
        "max_gain": float(max_gain),
        "mean_gain": float(mean_gain),
        "min_gain": float(min_gain),
        "n_directions": int(direction_matrix.shape[0]),
        "n_valid_directions": valid_count,
    }


def feedback_jacobian_sisu_modulation(
    policy: PolicyCallable,
    values: BlockValues,
    *,
    sisu_values: Array,
    schema: PolicyInputSchema | None = None,
    feedback_block: str = "feedback",
    sisu_block: str = "sisu",
) -> dict[str, Any]:
    """Summarize how ``du/dfeedback`` changes across SISU values."""
    schema = schema or PolicyInputSchema.from_values(values)
    schema.block(feedback_block)
    sisu_spec = schema.block(sisu_block)
    levels = jnp.asarray(sisu_values)
    if levels.ndim == 0:
        levels = levels[None]

    def jacobian_at_level(level: Array) -> Array:
        level_values = dict(values)
        level_values[sisu_block] = jnp.broadcast_to(level, sisu_spec.shape)
        return policy_block_jacobian(
            policy,
            level_values,
            feedback_block,
            schema=schema,
        )

    stacked = jax.vmap(jacobian_at_level)(levels)
    delta_from_reference = stacked - stacked[0]
    endpoint_delta = stacked[-1] - stacked[0]
    payload: dict[str, Any] = {
        "status": AVAILABLE,
        "feedback_block": feedback_block,
        "sisu_block": sisu_block,
        "sisu_values": [float(value) for value in levels.reshape(-1)],
        "jacobians": stacked,
        "delta_from_reference": delta_from_reference,
        "endpoint_delta_norm": float(jnp.linalg.norm(endpoint_delta)),
    }
    if levels.size >= 2:
        span = float(levels[-1] - levels[0])
        payload["endpoint_slope"] = endpoint_delta / span if abs(span) > 1e-12 else None
    return payload


def signed_pair_odd_even_summary(
    positive: Array,
    negative: Array,
    *,
    amplitude: float = 1.0,
    baseline: Array | None = None,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Summarize odd and even responses from a signed perturbation pair.

    ``positive`` and ``negative`` may be raw outputs. When ``baseline`` is
    supplied, the signed-pair decomposition is applied to deviations from that
    baseline. Without a baseline, the arrays are treated as already-baselined
    perturbation responses.
    """
    positive = jnp.asarray(positive)
    negative = jnp.asarray(negative)
    if positive.shape != negative.shape:
        raise ValueError(
            f"positive and negative must have matching shapes, got "
            f"{positive.shape} and {negative.shape}."
        )
    if baseline is not None:
        baseline = jnp.asarray(baseline)
        if baseline.shape != positive.shape:
            raise ValueError(
                f"baseline must have shape {positive.shape}, got {baseline.shape}."
            )
        positive = positive - baseline
        negative = negative - baseline

    odd = 0.5 * (positive - negative)
    even = 0.5 * (positive + negative)
    odd_norm = float(jnp.linalg.norm(odd))
    even_norm = float(jnp.linalg.norm(even))
    amplitude_abs = max(abs(float(amplitude)), epsilon)
    return {
        "status": AVAILABLE,
        "response_shape": list(positive.shape),
        "amplitude": float(amplitude),
        "odd_response": odd,
        "even_nonlinear_residual": even,
        "odd_norm": odd_norm,
        "even_nonlinear_residual_norm": even_norm,
        "even_to_odd_norm_ratio": even_norm / max(odd_norm, epsilon),
        "curvature_like_even_norm": even_norm / (amplitude_abs * amplitude_abs),
    }


__all__ = [
    "AVAILABLE",
    "NOT_APPLICABLE",
    "POLICY_DIAGNOSTICS_SCHEMA_VERSION",
    "PolicyAbsentInputBlock",
    "PolicyFiniteDifferenceValidation",
    "PolicyInputBlock",
    "PolicyInputSchema",
    "PolicyJacobian",
    "directional_gain_summary",
    "feedback_jacobian_sisu_modulation",
    "finite_difference_jacobian",
    "policy_block_jacobian",
    "policy_jacobian",
    "signed_pair_odd_even_summary",
    "singular_value_summary",
    "validate_policy_jacobian",
]

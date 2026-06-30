"""Finite closed-loop adversary policies for full-state epsilon channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

FiniteAdversaryPolicyClass = Literal["linear_no_bias", "affine"]

LINEAR_NO_BIAS_POLICY: FiniteAdversaryPolicyClass = "linear_no_bias"
AFFINE_POLICY: FiniteAdversaryPolicyClass = "affine"
TARGET_CENTERED_FULL_STATE_FEATURES = "target_centered_full_state"
FINITE_POLICY_GAINS_INPUT = "epsilon_policy_gains"
FINITE_POLICY_BIAS_INPUT = "epsilon_policy_bias"


@dataclass(frozen=True)
class FiniteAdversaryPolicyMetadata:
    """Metadata defining a shared time-varying finite adversary policy.

    Shapes:
        gains: ``(time, epsilon_dim, feature_dim)``
        bias: ``(time, epsilon_dim)`` for affine policies only
        live_features: ``(..., time, feature_dim)``
        epsilon: ``(..., time, epsilon_dim)``
    """

    policy_class: FiniteAdversaryPolicyClass
    horizon: int
    feature_dim: int
    epsilon_dim: int
    feature_basis: str = TARGET_CENTERED_FULL_STATE_FEATURES
    live_feature_source: str = "live_perturbed_rollout_state"
    shared_across_trials_in_batch: bool = True
    time_varying: bool = True
    centered_features: bool = True
    has_bias: bool = False

    def __post_init__(self) -> None:
        if self.policy_class not in (LINEAR_NO_BIAS_POLICY, AFFINE_POLICY):
            raise ValueError(f"unknown finite adversary policy class {self.policy_class!r}")
        if int(self.horizon) < 1:
            raise ValueError("horizon must be positive")
        if int(self.feature_dim) < 1:
            raise ValueError("feature_dim must be positive")
        if int(self.epsilon_dim) < 1:
            raise ValueError("epsilon_dim must be positive")
        if self.policy_class == LINEAR_NO_BIAS_POLICY and self.has_bias:
            raise ValueError("linear_no_bias policy metadata cannot declare a bias")
        if self.policy_class == AFFINE_POLICY and not self.has_bias:
            raise ValueError("affine policy metadata must declare a bias")

    @property
    def gain_shape(self) -> tuple[int, int, int]:
        """Return the finite-horizon gain tensor shape."""

        return (int(self.horizon), int(self.epsilon_dim), int(self.feature_dim))

    @property
    def bias_shape(self) -> tuple[int, int] | None:
        """Return the finite-horizon bias shape when this policy has a bias."""

        if not self.has_bias:
            return None
        return (int(self.horizon), int(self.epsilon_dim))

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable policy metadata for run specs and audits."""

        return {
            "policy_class": self.policy_class,
            "horizon": int(self.horizon),
            "feature_dim": int(self.feature_dim),
            "epsilon_dim": int(self.epsilon_dim),
            "feature_basis": self.feature_basis,
            "live_feature_source": self.live_feature_source,
            "shared_across_trials_in_batch": bool(self.shared_across_trials_in_batch),
            "time_varying": bool(self.time_varying),
            "centered_features": bool(self.centered_features),
            "has_bias": bool(self.has_bias),
            "gain_shape": list(self.gain_shape),
            "bias_shape": None if self.bias_shape is None else list(self.bias_shape),
            "zero_feature_behavior": (
                "zero_epsilon" if self.policy_class == LINEAR_NO_BIAS_POLICY else "bias_epsilon"
            ),
            "semantics": (
                "epsilon_t is evaluated from live perturbed rollout features at time t; "
                "the finite policy parameters are shared across every trial in the batch."
            ),
        }


class FiniteLinearNoBiasPolicy(eqx.Module):
    """Shared time-varying linear policy with no open-loop bias term."""

    gains: Float[Array, "time epsilon_dim feature_dim"]
    metadata: FiniteAdversaryPolicyMetadata = eqx.field(static=True)

    def __init__(
        self,
        gains: Any,
        *,
        feature_basis: str = TARGET_CENTERED_FULL_STATE_FEATURES,
    ) -> None:
        gain_array = jnp.asarray(gains)
        if gain_array.ndim != 3:
            raise ValueError(
                f"gains must have shape (time, epsilon_dim, feature_dim); got {gain_array.shape}"
            )
        self.gains = gain_array
        self.metadata = FiniteAdversaryPolicyMetadata(
            policy_class=LINEAR_NO_BIAS_POLICY,
            horizon=int(gain_array.shape[0]),
            epsilon_dim=int(gain_array.shape[1]),
            feature_dim=int(gain_array.shape[2]),
            feature_basis=feature_basis,
            has_bias=False,
        )

    def __call__(self, live_features: Any) -> Float[Array, "... time epsilon_dim"]:
        """Evaluate epsilon from live perturbed rollout features."""

        features = _validate_live_features(live_features, self.metadata)
        return jnp.einsum("...tf,tef->...te", features, self.gains)


class FiniteAffinePolicy(eqx.Module):
    """Shared time-varying affine policy with an explicit open-loop bias."""

    gains: Float[Array, "time epsilon_dim feature_dim"]
    bias: Float[Array, "time epsilon_dim"]
    metadata: FiniteAdversaryPolicyMetadata = eqx.field(static=True)

    def __init__(
        self,
        gains: Any,
        bias: Any,
        *,
        feature_basis: str = TARGET_CENTERED_FULL_STATE_FEATURES,
    ) -> None:
        gain_array = jnp.asarray(gains)
        bias_array = jnp.asarray(bias, dtype=gain_array.dtype)
        if gain_array.ndim != 3:
            raise ValueError(
                f"gains must have shape (time, epsilon_dim, feature_dim); got {gain_array.shape}"
            )
        if bias_array.shape != gain_array.shape[:2]:
            raise ValueError(
                "bias must have shape (time, epsilon_dim); "
                f"got {bias_array.shape}, expected {gain_array.shape[:2]}"
            )
        self.gains = gain_array
        self.bias = bias_array
        self.metadata = FiniteAdversaryPolicyMetadata(
            policy_class=AFFINE_POLICY,
            horizon=int(gain_array.shape[0]),
            epsilon_dim=int(gain_array.shape[1]),
            feature_dim=int(gain_array.shape[2]),
            feature_basis=feature_basis,
            has_bias=True,
        )

    def __call__(self, live_features: Any) -> Float[Array, "... time epsilon_dim"]:
        """Evaluate epsilon from live perturbed rollout features plus bias."""

        features = _validate_live_features(live_features, self.metadata)
        epsilon = jnp.einsum("...tf,tef->...te", features, self.gains)
        return epsilon + self.bias


def zero_finite_linear_no_bias_policy(
    *,
    horizon: int,
    feature_dim: int,
    epsilon_dim: int,
    dtype: Any = jnp.float32,
) -> FiniteLinearNoBiasPolicy:
    """Return a zero-initialized no-bias finite policy."""

    gains = jnp.zeros((int(horizon), int(epsilon_dim), int(feature_dim)), dtype=dtype)
    return FiniteLinearNoBiasPolicy(gains)


def zero_finite_affine_policy(
    *,
    horizon: int,
    feature_dim: int,
    epsilon_dim: int,
    dtype: Any = jnp.float32,
) -> FiniteAffinePolicy:
    """Return a zero-initialized affine finite policy."""

    gains = jnp.zeros((int(horizon), int(epsilon_dim), int(feature_dim)), dtype=dtype)
    bias = jnp.zeros((int(horizon), int(epsilon_dim)), dtype=dtype)
    return FiniteAffinePolicy(gains, bias)


def target_centered_full_state_features(
    mechanics_vector: Any,
    *,
    target_position: Any,
    physical_block_size: int = 8,
) -> Float[Array, "... state_dim"]:
    """Return target/error-centered features from a live mechanics state.

    The C&S mechanics vector is treated as one or more physical state blocks. In
    each block, the first two coordinates are position-like coordinates and are
    shifted by ``target_position``. All other coordinates are already
    target-centered dynamics/features and pass through unchanged.
    """

    values = jnp.asarray(mechanics_vector)
    target = jnp.asarray(target_position, dtype=values.dtype)
    block_size = int(physical_block_size)
    if block_size < 2:
        raise ValueError(f"physical_block_size must be at least 2, got {block_size}")
    if values.shape[-1] % block_size != 0:
        if block_size == 8 and values.shape[-1] % 6 == 0:
            block_size = 6
        else:
            raise ValueError(
                f"expected state dimension divisible by {block_size}, got {values.shape[-1]}"
            )
    if target.shape[-1] != 2:
        raise ValueError(f"target_position must end in dimension 2, got {target.shape}")
    reshaped = values.reshape((*values.shape[:-1], values.shape[-1] // block_size, block_size))
    target = target[..., None, :]
    while target.ndim < len(reshaped.shape):
        target = jnp.expand_dims(target, axis=-3)
    target = jnp.broadcast_to(target, (*reshaped.shape[:-1], 2))
    centered = reshaped.at[..., 0:2].add(-target)
    return centered.reshape(values.shape)


def finite_policy_step_epsilon(
    mechanics_vector: Any,
    *,
    target_position: Any,
    gain_t: Any,
    bias_t: Any | None = None,
    physical_block_size: int = 8,
) -> Float[Array, "... epsilon_dim"]:
    """Evaluate one finite epsilon-policy step from live mechanics state."""

    features = target_centered_full_state_features(
        mechanics_vector,
        target_position=target_position,
        physical_block_size=physical_block_size,
    )
    gains = jnp.asarray(gain_t, dtype=features.dtype)
    if gains.ndim == 2:
        epsilon = jnp.einsum("...f,ef->...e", features, gains)
    elif gains.ndim >= 3:
        epsilon = jnp.einsum("...f,...ef->...e", features, gains)
    else:
        raise ValueError(f"gain_t must end with (epsilon_dim, feature_dim), got {gains.shape}")
    if bias_t is None:
        return epsilon
    bias = jnp.asarray(bias_t, dtype=epsilon.dtype)
    return epsilon + bias


def finite_policy_contract(metadata: FiniteAdversaryPolicyMetadata) -> dict[str, Any]:
    """Return the stable finite-policy contract used by run specs and audits."""

    return {
        "kind": "closed_loop_finite_time_varying_epsilon_policy",
        "metadata": metadata.to_json(),
        "scientific_constraint": "soft_energy_penalty_or_audit_cap_not_hard_projection",
        "batch_sharing": "one_policy_instance_shared_across_batch_trials",
        "closed_loop_semantics": (
            "call the policy during rollout with live perturbed state/error features; "
            "do not materialize epsilon from a clean rollout and replay it as open loop"
        ),
    }


def _validate_live_features(
    live_features: Any,
    metadata: FiniteAdversaryPolicyMetadata,
) -> Float[Array, "... time feature_dim"]:
    features = jnp.asarray(live_features)
    if features.ndim < 2:
        raise ValueError("live_features must have at least time and feature dimensions")
    expected = (int(metadata.horizon), int(metadata.feature_dim))
    if features.shape[-2:] != expected:
        raise ValueError(f"live_features must end with shape {expected}; got {features.shape[-2:]}")
    return features

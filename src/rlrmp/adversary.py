"""Parameterized adversary for minimax training.

The adversary generates perturbation force profiles as a sum of Gaussian
bumps in time, with learnable timing, amplitude, and direction. This is
the simplest adversary that captures temporal structure — a stepping stone
to neural adversaries.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


class GaussianBumpAdversary(eqx.Module):
    """Generates perturbation as sum of K Gaussian force bumps.

    Each bump has learnable: timing (center), width (sigma), amplitude, direction.
    The adversary produces SISU-independent force profiles; SISU gating is handled
    externally by the task (PAI-ASF's ``scale * field`` term).

    Parameters are stored as raw (unconstrained) values; widths and amplitudes
    are passed through softplus during the forward pass to ensure positivity.

    Args:
        n_bumps: Number of Gaussian bumps to sum.
        n_timesteps: Length of force profile.
        n_force_dims: Dimensionality of force (2 for 2D reaching).
        force_max: Maximum force magnitude per timestep (clip).
        dt: Timestep size in seconds.
    """

    # Learnable parameters (raw / unconstrained)
    bump_centers: Float[Array, " n_bumps"]  # in [0, n_timesteps * dt]
    bump_widths_raw: Float[Array, " n_bumps"]  # softplus → positive sigma
    bump_amplitudes_raw: Float[Array, " n_bumps"]  # softplus → positive amplitude
    bump_directions: Float[Array, "n_bumps n_force_dims"]  # unit vectors

    # Fixed metadata
    n_timesteps: int = eqx.field(static=True)
    n_force_dims: int = eqx.field(static=True)
    force_max: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(
        self,
        n_bumps: int = 3,
        n_timesteps: int = 130,
        n_force_dims: int = 2,
        force_max: float = 1.0,
        dt: float = 0.01,
        *,
        key: Array,
    ):
        keys = jr.split(key, 2)

        trial_duration = n_timesteps * dt

        # Centers uniformly in [20%, 80%] of trial duration
        self.bump_centers = jr.uniform(
            keys[0], (n_bumps,), minval=0.2 * trial_duration, maxval=0.8 * trial_duration
        )

        # Raw widths: initialise so softplus gives ~50 ms
        # softplus(x) ≈ x for large x; we want softplus(x) ≈ 0.05 → x ≈ 0.05
        # Use explicit dtype=float32 to avoid weak_type=True from Python scalar
        # initialization, which causes JIT recompilation after the first optimizer
        # update strips weak_type from the leaves.
        self.bump_widths_raw = jnp.array([0.05] * n_bumps, dtype=jnp.float32)

        # Raw amplitudes: initialise so softplus gives ~0.1 (small perturbation)
        self.bump_amplitudes_raw = jnp.array([0.1] * n_bumps, dtype=jnp.float32)

        # Directions: random unit vectors
        raw_dirs = jr.normal(keys[1], (n_bumps, n_force_dims))
        self.bump_directions = raw_dirs / (jnp.linalg.norm(raw_dirs, axis=-1, keepdims=True) + 1e-8)

        self.n_timesteps = n_timesteps
        self.n_force_dims = n_force_dims
        self.force_max = force_max
        self.dt = dt

    def __call__(self) -> Float[Array, "n_timesteps n_force_dims"]:
        """Generate force profile.

        Returns:
            forces: (n_timesteps, n_force_dims) perturbation force profile.
        """
        t = jnp.arange(self.n_timesteps) * self.dt  # (T,)

        # Positive widths and amplitudes via softplus
        widths = jax.nn.softplus(self.bump_widths_raw)  # (K,)
        amps = jax.nn.softplus(self.bump_amplitudes_raw)  # (K,)

        # Gaussian envelope for each bump: (T, K)
        diff = t[:, None] - self.bump_centers[None, :]  # (T, K)
        envelopes = jnp.exp(-0.5 * (diff / (widths[None, :] + 1e-4)) ** 2)  # (T, K)

        # Weighted sum over bumps: (T, d)
        forces = jnp.einsum("tk,k,kd->td", envelopes, amps, self.bump_directions)

        # Clip per-timestep force magnitude to force_max
        norms = jnp.linalg.norm(forces, axis=-1, keepdims=True)  # (T, 1)
        forces = jnp.where(
            norms > self.force_max,
            forces * self.force_max / (norms + 1e-8),
            forces,
        )

        return forces

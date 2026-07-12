"""Parameterized adversaries for minimax training.

Two flavours, both ``eqx.Module`` so they slot into the registered minimax
method's vmap/JIT machinery interchangeably:

- ``GaussianBumpAdversary`` (flavour-(a) input-instance): produces a
  per-trial **force profile** via a sum of learnable Gaussian bumps. The
  resulting force is injected into the trial-spec disturbance channel.

- ``LinearDynamicsAdversary`` (flavour-(b) model-class): produces a
  ``ΔA`` matrix mapping ``[pos, vel]`` to a velocity-row dynamics
  perturbation. Constrained to a Frobenius ball ``||ΔA||_F ≤ η_max · SISU``
  via PGD with Frobenius projection. The matrix parameterises the feedbax
  ``DynamicsMatrixPerturb`` intervenor at each adversarial inner step.

Both adversaries' learnable parameters are JAX arrays directly on the
``eqx.Module`` so that ``eqx.filter`` and ``optax.adam`` can update them
with the standard pattern. Bug: c723082.
"""

from typing import Optional

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


def _frobenius_project(M: Array, radius: Array) -> Array:
    """Project ``M`` onto the Frobenius ball of given ``radius``.

    Computes ``min(||M||_F, radius)`` and rescales ``M`` to that norm. Both
    ``radius`` and ``||M||_F`` may be scalar JAX traced values.

    Args:
        M: Matrix to project.
        radius: Non-negative scalar Frobenius-norm bound.

    Returns:
        Matrix with ``||M_proj||_F ≤ radius``, with sign and shape preserved.
    """
    norm = jnp.linalg.norm(M)
    # Avoid division-by-zero; if norm == 0, the matrix is already zero.
    scale = jnp.minimum(1.0, radius / (norm + 1e-12))
    return scale * M


class LinearDynamicsAdversary(eqx.Module):
    """Model-class ``ΔA`` adversary with Frobenius-ball-bounded entries.

    Parameterises an additive perturbation to the velocity-row dynamics:
    ``dot v += ΔA @ [pos, vel]``. Wired into the plant via feedbax's
    ``DynamicsMatrixPerturb`` intervenor. Constrained to
    ``||ΔA||_F ≤ η(SISU) = eta_max · SISU`` and trained with projected
    gradient ascent over ``n_inner_steps`` per outer batch.

    The constraint applies *uniformly* in time across the rollout: the
    same ``ΔA`` is applied at every timestep within a trial.

    Args:
        n_state: State dimension (``2 * n_dim``: ``[pos, vel]``).
        n_dim: Spatial dimension (number of velocity rows; default 2).
        eta_max: Maximum Frobenius norm at SISU=1.
        n_inner_steps: PGD steps per outer batch (default 5).
        learning_rate: PGD step size (default 1e-2).
        key: PRNG key for initialising ``ΔA`` to small random values.
    """

    delta_A: Float[Array, "n_dim n_state"]
    eta_max: float = eqx.field(static=True)
    n_inner_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    n_dim: int = eqx.field(static=True)
    n_state: int = eqx.field(static=True)

    def __init__(
        self,
        n_state: int = 4,
        n_dim: int = 2,
        eta_max: float = 0.1,
        n_inner_steps: int = 5,
        learning_rate: float = 1e-2,
        *,
        key: Optional[Array] = None,
    ):
        # Initialise to a tiny random matrix so the first projection is
        # well-defined and gradients have non-trivial signal at SISU=0
        # (where the ball collapses to {0}).
        if key is None:
            key = jr.PRNGKey(0)
        # Small init scale so the initial projection is essentially clamping
        # to within eta_max at SISU=1.
        init_scale = min(1e-3, 0.1 * eta_max)
        self.delta_A = init_scale * jr.normal(key, (n_dim, n_state), dtype=jnp.float32)
        self.eta_max = float(eta_max)
        self.n_inner_steps = int(n_inner_steps)
        self.learning_rate = float(learning_rate)
        self.n_dim = int(n_dim)
        self.n_state = int(n_state)

    def project(self) -> "LinearDynamicsAdversary":
        """Return a copy with ``delta_A`` projected to the Frobenius ball.

        Constrains ``||delta_A||_F ≤ eta_max``. SISU gating is applied
        downstream via the intervenor's ``params.scale = SISU`` field, so
        the projection itself is SISU-independent — the *budget* of the
        adversary at full SISU=1 is ``eta_max``.

        Returns:
            A new ``LinearDynamicsAdversary`` whose ``delta_A`` satisfies
            ``||delta_A||_F ≤ eta_max``.
        """
        new_delta_A = _frobenius_project(self.delta_A, jnp.asarray(self.eta_max))
        return eqx.tree_at(lambda a: a.delta_A, self, new_delta_A)

    def frobenius_norm(self) -> Array:
        """Frobenius norm of the current ``delta_A`` (scalar)."""
        return jnp.linalg.norm(self.delta_A)

    def __call__(self) -> Float[Array, "n_dim n_state"]:
        """Return the current ``ΔA`` matrix.

        Provided for API symmetry with ``GaussianBumpAdversary.__call__``,
        which returns its produced perturbation.
        """
        return self.delta_A

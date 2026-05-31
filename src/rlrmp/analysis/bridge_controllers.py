"""Small controller substrates for analytical bridge comparisons.

The bridge program needs two intermediate controller families before GRU
interpretation: constrained time-varying gains and explicitly linear
recurrences.  This module keeps both substrates independent of training code and
emits the shared :class:`BridgeRolloutBatch` arrays from ``bridge_contracts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Float

from rlrmp.analysis.bridge_contracts import BridgeRolloutBatch


class LinearPlantLike(Protocol):
    """Minimal discrete-time plant surface consumed by bridge rollouts."""

    A: Float[np.ndarray, "state state"]
    B: Float[np.ndarray, "state action"]
    Bw: Float[np.ndarray, "state disturbance"]


@dataclass(frozen=True)
class GainProjection:
    """Least-squares projection of an unconstrained gain sequence.

    Attributes:
        theta: Basis coefficients, shape ``(basis, action, input)``.
        reconstructed_gains: Projected gains, shape ``(horizon, action, input)``.
        residual_norm: Frobenius norm of the projection residual.
        relative_residual: Residual norm divided by the unconstrained gain norm.
        rank: Numerical rank reported by ``numpy.linalg.lstsq``.
        singular_values: Singular values of the time basis.
    """

    theta: Float[np.ndarray, "basis action input"]
    reconstructed_gains: Float[np.ndarray, "horizon action input"]
    residual_norm: float
    relative_residual: float
    rank: int
    singular_values: Float[np.ndarray, " basis"]


@dataclass(frozen=True)
class TimeConstrainedGainParameterization:
    """Linear time-basis parameterization for output-feedback gains.

    ``basis[t, b]`` maps coefficient block ``theta[b]`` into the gain at time
    ``t``.  Gains follow the usual bridge convention ``u_t = -K_t input_t``;
    this class only parameterizes ``K_t`` and does not roll a plant itself.
    """

    basis: Float[np.ndarray, "horizon basis"]
    action_dim: int
    input_dim: int

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError("basis must have shape (horizon, basis)")
        if basis.shape[0] <= 0 or basis.shape[1] <= 0:
            raise ValueError("basis must have positive horizon and basis dimensions")
        if self.action_dim <= 0 or self.input_dim <= 0:
            raise ValueError("action_dim and input_dim must be positive")
        object.__setattr__(self, "basis", basis)

    @property
    def horizon(self) -> int:
        """Number of time steps represented by the basis."""

        return int(self.basis.shape[0])

    @property
    def n_basis(self) -> int:
        """Number of time-basis coefficients."""

        return int(self.basis.shape[1])

    @classmethod
    def constant(
        cls, *, horizon: int, action_dim: int, input_dim: int
    ) -> "TimeConstrainedGainParameterization":
        """Return a time-invariant gain parameterization."""

        return cls(
            basis=np.ones((horizon, 1), dtype=np.float64),
            action_dim=action_dim,
            input_dim=input_dim,
        )

    @classmethod
    def piecewise_constant(
        cls,
        *,
        segment_ids: Float[np.ndarray, " horizon"],
        action_dim: int,
        input_dim: int,
    ) -> "TimeConstrainedGainParameterization":
        """Return a piecewise-constant parameterization from integer segment IDs."""

        segments = np.asarray(segment_ids)
        if segments.ndim != 1:
            raise ValueError("segment_ids must have shape (horizon,)")
        if segments.shape[0] == 0:
            raise ValueError("segment_ids must be nonempty")
        unique = tuple(dict.fromkeys(int(segment) for segment in segments))
        index = {segment: idx for idx, segment in enumerate(unique)}
        basis = np.zeros((segments.shape[0], len(unique)), dtype=np.float64)
        for t, segment in enumerate(segments):
            basis[t, index[int(segment)]] = 1.0
        return cls(basis=basis, action_dim=action_dim, input_dim=input_dim)

    @classmethod
    def cubic_bspline(
        cls,
        *,
        horizon: int,
        n_basis: int,
        action_dim: int,
        input_dim: int,
    ) -> "TimeConstrainedGainParameterization":
        """Return a compact smooth cubic time basis.

        The basis uses cardinal cubic B-spline weights evaluated at uniformly
        spaced centers over the discrete horizon.  Each row is normalized after
        evaluation, so the basis is nonnegative and forms a partition of unity
        over time.  When ``n_basis == horizon`` this produces a square local
        basis that is numerically full rank for ordinary horizons such as
        ``60``, so projections can reconstruct any gain sequence.
        """

        basis = _cubic_bspline_time_basis(horizon=horizon, n_basis=n_basis)
        return cls(basis=basis, action_dim=action_dim, input_dim=input_dim)

    def gains_from_theta(
        self,
        theta: Float[np.ndarray, "basis action input"],
    ) -> Float[np.ndarray, "horizon action input"]:
        """Expand basis coefficients into a time-varying gain sequence."""

        coefficients = np.asarray(theta, dtype=np.float64)
        expected = (self.n_basis, self.action_dim, self.input_dim)
        if coefficients.shape != expected:
            raise ValueError(f"theta must have shape {expected}; got {coefficients.shape}")
        return np.einsum("tb,bai->tai", self.basis, coefficients)

    def project_gains(
        self,
        gains: Float[np.ndarray, "horizon action input"],
        *,
        rcond: float | None = None,
    ) -> GainProjection:
        """Project unconstrained time-varying gains onto the time basis."""

        gain_array = np.asarray(gains, dtype=np.float64)
        expected = (self.horizon, self.action_dim, self.input_dim)
        if gain_array.shape != expected:
            raise ValueError(f"gains must have shape {expected}; got {gain_array.shape}")

        flat = gain_array.reshape((self.horizon, self.action_dim * self.input_dim))
        theta_flat, _residuals, rank, singular_values = np.linalg.lstsq(
            self.basis,
            flat,
            rcond=rcond,
        )
        theta = theta_flat.reshape((self.n_basis, self.action_dim, self.input_dim))
        reconstructed = self.gains_from_theta(theta)
        residual_norm = float(np.linalg.norm(gain_array - reconstructed))
        gain_norm = float(np.linalg.norm(gain_array))
        relative_residual = residual_norm / max(gain_norm, np.finfo(np.float64).eps)
        return GainProjection(
            theta=theta,
            reconstructed_gains=reconstructed,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            rank=int(rank),
            singular_values=np.asarray(singular_values, dtype=np.float64),
        )


def _cubic_bspline_time_basis(*, horizon: int, n_basis: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_basis <= 0:
        raise ValueError("n_basis must be positive")
    if n_basis == 1:
        return np.ones((horizon, 1), dtype=np.float64)

    times = np.arange(horizon, dtype=np.float64)
    centers = np.linspace(0.0, float(horizon - 1), n_basis, dtype=np.float64)
    spacing = centers[1] - centers[0]
    scaled_distance = (times[:, None] - centers[None, :]) / spacing
    basis = _cardinal_cubic_bspline(scaled_distance)
    row_sums = np.sum(basis, axis=1, keepdims=True)
    if np.any(row_sums <= np.finfo(np.float64).eps):
        raise ValueError("cubic basis construction produced an empty row")
    return basis / row_sums


def _cardinal_cubic_bspline(x: np.ndarray) -> np.ndarray:
    distance = np.abs(np.asarray(x, dtype=np.float64))
    values = np.zeros_like(distance)

    inner = distance < 1.0
    values[inner] = (
        4.0 - 6.0 * distance[inner] ** 2 + 3.0 * distance[inner] ** 3
    ) / 6.0

    outer = (distance >= 1.0) & (distance < 2.0)
    values[outer] = (2.0 - distance[outer]) ** 3 / 6.0
    return values


@dataclass(frozen=True)
class LinearRecurrentController:
    """Linear recurrent bridge controller.

    The recurrence is

    ``h_{t+1} = A_h h_t + B_y y_t``
    ``u_t = C_h h_t + D_y y_t``

    where ``y_t`` is the rollout observation and ``u_t`` is applied directly to
    the plant.  ``D_y`` defaults to zero so pure hidden-readout controllers stay
    explicit.
    """

    recurrent_weights: Float[np.ndarray, "hidden hidden"]
    observation_weights: Float[np.ndarray, "hidden observation"]
    readout_weights: Float[np.ndarray, "action hidden"]
    feedthrough_weights: Float[np.ndarray, "action observation"] | None = None
    initial_hidden: Float[np.ndarray, " hidden"] | None = None

    def __post_init__(self) -> None:
        recurrent = np.asarray(self.recurrent_weights, dtype=np.float64)
        observation = np.asarray(self.observation_weights, dtype=np.float64)
        readout = np.asarray(self.readout_weights, dtype=np.float64)
        if recurrent.ndim != 2 or recurrent.shape[0] != recurrent.shape[1]:
            raise ValueError("recurrent_weights must have shape (hidden, hidden)")
        if observation.ndim != 2 or observation.shape[0] != recurrent.shape[0]:
            raise ValueError("observation_weights must have shape (hidden, observation)")
        if readout.ndim != 2 or readout.shape[1] != recurrent.shape[0]:
            raise ValueError("readout_weights must have shape (action, hidden)")
        if self.feedthrough_weights is None:
            feedthrough = np.zeros((readout.shape[0], observation.shape[1]), dtype=np.float64)
        else:
            feedthrough = np.asarray(self.feedthrough_weights, dtype=np.float64)
            expected = (readout.shape[0], observation.shape[1])
            if feedthrough.shape != expected:
                raise ValueError(f"feedthrough_weights must have shape {expected}")
        if self.initial_hidden is None:
            initial_hidden = np.zeros((recurrent.shape[0],), dtype=np.float64)
        else:
            initial_hidden = np.asarray(self.initial_hidden, dtype=np.float64)
            if initial_hidden.shape != (recurrent.shape[0],):
                raise ValueError("initial_hidden must have shape (hidden,)")

        object.__setattr__(self, "recurrent_weights", recurrent)
        object.__setattr__(self, "observation_weights", observation)
        object.__setattr__(self, "readout_weights", readout)
        object.__setattr__(self, "feedthrough_weights", feedthrough)
        object.__setattr__(self, "initial_hidden", initial_hidden)

    @property
    def hidden_dim(self) -> int:
        """Hidden-state dimension."""

        return int(self.recurrent_weights.shape[0])

    @property
    def observation_dim(self) -> int:
        """Observation dimension expected by the recurrence."""

        return int(self.observation_weights.shape[1])

    @property
    def action_dim(self) -> int:
        """Action dimension emitted by the readout."""

        return int(self.readout_weights.shape[0])

    def next_hidden(
        self,
        hidden: Float[np.ndarray, "... hidden"],
        observation: Float[np.ndarray, "... observation"],
    ) -> Float[np.ndarray, "... hidden"]:
        """Return the next hidden state for one or more batch elements."""

        return hidden @ self.recurrent_weights.T + observation @ self.observation_weights.T

    def action(
        self,
        hidden: Float[np.ndarray, "... hidden"],
        observation: Float[np.ndarray, "... observation"],
    ) -> Float[np.ndarray, "... action"]:
        """Return the controller action for one or more batch elements."""

        assert self.feedthrough_weights is not None
        return hidden @ self.readout_weights.T + observation @ self.feedthrough_weights.T

    def stability_diagnostics(self) -> dict[str, float]:
        """Return recurrence-only stability diagnostics."""

        return {"recurrent_spectral_radius": recurrent_spectral_radius(self.recurrent_weights)}


def recurrent_spectral_radius(matrix: Float[np.ndarray, "hidden hidden"]) -> float:
    """Return the maximum absolute eigenvalue of a recurrent matrix."""

    recurrent = np.asarray(matrix, dtype=np.float64)
    if recurrent.ndim != 2 or recurrent.shape[0] != recurrent.shape[1]:
        raise ValueError("matrix must have shape (hidden, hidden)")
    if recurrent.shape[0] == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(recurrent))))


def hidden_growth_diagnostics(
    hidden_states: Float[np.ndarray, "batch horizon_plus_one hidden"],
    *,
    floor: float = 1e-12,
) -> dict[str, float]:
    """Summarize hidden-state norm growth in a rollout batch."""

    hidden = np.asarray(hidden_states, dtype=np.float64)
    if hidden.ndim != 3:
        raise ValueError("hidden_states must have shape (batch, horizon + 1, hidden)")
    norms = np.linalg.norm(hidden, axis=-1)
    initial = float(np.mean(norms[:, 0]))
    final = float(np.mean(norms[:, -1]))
    maximum = float(np.max(norms))
    denominator = max(initial, floor)
    return {
        "hidden_initial_mean_norm": initial,
        "hidden_final_mean_norm": final,
        "hidden_max_norm": maximum,
        "hidden_final_to_initial": final / denominator,
        "hidden_max_to_initial": maximum / denominator,
    }


def rollout_linear_recurrent_controller(
    controller: LinearRecurrentController,
    plant: LinearPlantLike,
    x0: np.ndarray,
    *,
    horizon: int | None = None,
    observation_matrix: Float[np.ndarray, "observation state"] | None = None,
    disturbances: np.ndarray | None = None,
    initial_hidden: np.ndarray | None = None,
) -> BridgeRolloutBatch:
    """Roll a linear recurrent controller through a discrete-time linear plant.

    Args:
        controller: Linear recurrence and action readout.
        plant: Object exposing ``A``, ``B``, and ``Bw`` arrays.
        x0: Initial plant state, shape ``(state,)`` or ``(batch, state)``.
        horizon: Number of rollout steps. Required when ``disturbances`` is absent.
        observation_matrix: Matrix mapping plant state to controller observation.
            Defaults to the full state identity.
        disturbances: Optional disturbance sequence, shape ``(T, m_w)`` or
            ``(batch, T, m_w)``.
        initial_hidden: Optional hidden initial condition, shape ``(hidden,)`` or
            ``(batch, hidden)``. Defaults to ``controller.initial_hidden``.

    Returns:
        A ``BridgeRolloutBatch`` with plant states, observations, actions, and
        hidden states populated.
    """

    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    Bw = np.asarray(plant.Bw, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("plant.A must have shape (state, state)")
    if B.shape != (A.shape[0], controller.action_dim):
        raise ValueError(
            f"plant.B must have shape {(A.shape[0], controller.action_dim)}; got {B.shape}"
        )
    if Bw.ndim != 2 or Bw.shape[0] != A.shape[0]:
        raise ValueError("plant.Bw must have shape (state, disturbance)")

    states = _as_batch(np.asarray(x0, dtype=np.float64), width=A.shape[0], name="x0")
    batch_size = states.shape[0]
    observations_matrix = (
        np.eye(A.shape[0], dtype=np.float64)
        if observation_matrix is None
        else np.asarray(observation_matrix, dtype=np.float64)
    )
    if observations_matrix.shape != (controller.observation_dim, A.shape[0]):
        raise ValueError(
            "observation_matrix must have shape "
            f"{(controller.observation_dim, A.shape[0])}; got {observations_matrix.shape}"
        )

    disturbances_batch, horizon_value = _normalize_disturbances(
        disturbances,
        batch_size=batch_size,
        disturbance_dim=Bw.shape[1],
        horizon=horizon,
    )
    hidden = _initial_hidden_batch(controller, initial_hidden, batch_size)

    plant_states = [states]
    hidden_states = [hidden]
    observations = []
    actions = []
    for t in range(horizon_value):
        y_t = states @ observations_matrix.T
        u_t = controller.action(hidden, y_t)
        states = states @ A.T + u_t @ B.T + disturbances_batch[:, t, :] @ Bw.T
        hidden = controller.next_hidden(hidden, y_t)
        observations.append(y_t)
        actions.append(u_t)
        plant_states.append(states)
        hidden_states.append(hidden)

    plant_state_array = np.stack(plant_states, axis=1)
    hidden_state_array = np.stack(hidden_states, axis=1)
    diagnostics = controller.stability_diagnostics() | hidden_growth_diagnostics(hidden_state_array)
    return BridgeRolloutBatch(
        plant_states=plant_state_array,
        actions=np.stack(actions, axis=1),
        observations=np.stack(observations, axis=1),
        hidden_states=hidden_state_array,
        metadata={"controller": "linear_recurrence", "diagnostics": diagnostics},
    )


def _as_batch(array: np.ndarray, *, width: int, name: str) -> np.ndarray:
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (batch, {width})")
    return array


def _initial_hidden_batch(
    controller: LinearRecurrentController,
    initial_hidden: np.ndarray | None,
    batch_size: int,
) -> np.ndarray:
    hidden = controller.initial_hidden if initial_hidden is None else np.asarray(initial_hidden)
    assert hidden is not None
    return np.broadcast_to(
        _as_batch(np.asarray(hidden, dtype=np.float64), width=controller.hidden_dim, name="hidden"),
        (batch_size, controller.hidden_dim),
    ).copy()


def _normalize_disturbances(
    disturbances: np.ndarray | None,
    *,
    batch_size: int,
    disturbance_dim: int,
    horizon: int | None,
) -> tuple[np.ndarray, int]:
    if disturbances is None:
        if horizon is None:
            raise ValueError("horizon is required when disturbances are absent")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        return np.zeros((batch_size, horizon, disturbance_dim), dtype=np.float64), horizon

    values = np.asarray(disturbances, dtype=np.float64)
    if values.ndim == 2:
        values = np.broadcast_to(values[None, :, :], (batch_size, values.shape[0], values.shape[1]))
    if values.ndim != 3 or values.shape[0] != batch_size or values.shape[2] != disturbance_dim:
        raise ValueError(
            "disturbances must have shape "
            f"(horizon, {disturbance_dim}) or (batch, horizon, {disturbance_dim})"
        )
    if horizon is not None and horizon != values.shape[1]:
        raise ValueError("horizon does not match disturbances")
    if values.shape[1] <= 0:
        raise ValueError("horizon must be positive")
    return values.copy(), int(values.shape[1])


__all__ = [
    "GainProjection",
    "LinearRecurrentController",
    "TimeConstrainedGainParameterization",
    "hidden_growth_diagnostics",
    "recurrent_spectral_radius",
    "rollout_linear_recurrent_controller",
]

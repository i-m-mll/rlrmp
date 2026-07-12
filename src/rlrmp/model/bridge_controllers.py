"""Small controller substrates for analytical bridge comparisons.

The bridge program needs two intermediate controller families before GRU
interpretation: constrained time-varying gains and explicitly linear
recurrences.  This module keeps both substrates independent of training code and
emits the shared :class:`BridgeRolloutBatch` arrays from ``bridge_results``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Float

from rlrmp.analysis.bridge_results import BridgeRolloutBatch


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


def clamped_bspline_time_basis(
    *, horizon: int, n_basis: int, degree: int | None = None
) -> Float[np.ndarray, "horizon basis"]:
    """Return a clamped B-spline phase basis over ``tau=t/(T-1)``."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_basis <= 0:
        raise ValueError("n_basis must be positive")
    spline_degree = min(3, n_basis - 1) if degree is None else int(degree)
    if spline_degree < 0 or spline_degree >= n_basis:
        raise ValueError("degree must satisfy 0 <= degree < n_basis")
    if n_basis == 1:
        return np.ones((horizon, 1), dtype=np.float64)

    tau = (
        np.zeros((1,), dtype=np.float64)
        if horizon == 1
        else np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    )
    n_interior = n_basis - spline_degree - 1
    interior = (
        np.linspace(0.0, 1.0, n_interior + 2, dtype=np.float64)[1:-1]
        if n_interior > 0
        else np.array([], dtype=np.float64)
    )
    knots = np.concatenate(
        [
            np.zeros((spline_degree + 1,), dtype=np.float64),
            interior,
            np.ones((spline_degree + 1,), dtype=np.float64),
        ]
    )
    basis = np.column_stack(
        [_bspline_basis_function(tau, i, spline_degree, knots) for i in range(n_basis)]
    )
    basis[tau == 1.0, :] = 0.0
    basis[tau == 1.0, -1] = 1.0
    row_sums = np.sum(basis, axis=1, keepdims=True)
    if np.any(row_sums <= np.finfo(np.float64).eps):
        raise ValueError("clamped B-spline construction produced an empty row")
    return basis / row_sums


def _bspline_basis_function(
    tau: np.ndarray,
    index: int,
    degree: int,
    knots: np.ndarray,
) -> np.ndarray:
    if degree == 0:
        return ((knots[index] <= tau) & (tau < knots[index + 1])).astype(np.float64)
    left_denominator = knots[index + degree] - knots[index]
    right_denominator = knots[index + degree + 1] - knots[index + 1]
    left = np.zeros_like(tau, dtype=np.float64)
    right = np.zeros_like(tau, dtype=np.float64)
    if left_denominator > 0.0:
        left = (
            (tau - knots[index])
            / left_denominator
            * _bspline_basis_function(tau, index, degree - 1, knots)
        )
    if right_denominator > 0.0:
        right = (
            (knots[index + degree + 1] - tau)
            / right_denominator
            * _bspline_basis_function(tau, index + 1, degree - 1, knots)
        )
    return left + right


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
    values[inner] = (4.0 - 6.0 * distance[inner] ** 2 + 3.0 * distance[inner] ** 3) / 6.0

    outer = (distance >= 1.0) & (distance < 2.0)
    values[outer] = (2.0 - distance[outer]) ** 3 / 6.0
    return values


@dataclass(frozen=True)
class MatrixBasisProjection:
    """Least-squares projection of a time-varying matrix sequence."""

    theta: Float[np.ndarray, "basis rows cols"]
    reconstructed: Float[np.ndarray, "horizon rows cols"]
    residual_norm: float
    relative_residual: float
    rank: int
    singular_values: Float[np.ndarray, " basis"]


def project_matrix_sequence_to_basis(
    matrix_sequence: Float[np.ndarray, "horizon rows cols"],
    basis: Float[np.ndarray, "horizon basis"],
    *,
    rcond: float | None = None,
) -> MatrixBasisProjection:
    """Least-squares project a time-varying matrix sequence onto ``basis``."""

    sequence = np.asarray(matrix_sequence, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError("matrix_sequence must have shape (horizon, rows, cols)")
    if basis_array.ndim != 2 or basis_array.shape[0] != sequence.shape[0]:
        raise ValueError("basis must have shape (horizon, basis)")
    flat = sequence.reshape((sequence.shape[0], sequence.shape[1] * sequence.shape[2]))
    theta_flat, _residuals, rank, singular_values = np.linalg.lstsq(
        basis_array,
        flat,
        rcond=rcond,
    )
    theta = theta_flat.reshape((basis_array.shape[1], sequence.shape[1], sequence.shape[2]))
    reconstructed = np.einsum("tb,bij->tij", basis_array, theta)
    residual_norm = float(np.linalg.norm(sequence - reconstructed))
    sequence_norm = float(np.linalg.norm(sequence))
    return MatrixBasisProjection(
        theta=theta,
        reconstructed=reconstructed,
        residual_norm=residual_norm,
        relative_residual=residual_norm / max(sequence_norm, np.finfo(np.float64).eps),
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=np.float64),
    )


@dataclass(frozen=True)
class PhaseModulatedLinearRecurrentController:
    """Linear recurrence whose dynamics/readout matrices vary with trial phase."""

    basis: Float[np.ndarray, "horizon basis"]
    recurrent_coefficients: Float[np.ndarray, "basis hidden hidden"]
    observation_coefficients: Float[np.ndarray, "basis hidden observation"]
    previous_action_coefficients: Float[np.ndarray, "basis hidden action"]
    hidden_bias_coefficients: Float[np.ndarray, "basis hidden"]
    readout_coefficients: Float[np.ndarray, "basis action hidden"]
    feedthrough_coefficients: Float[np.ndarray, "basis action observation"]
    action_bias_coefficients: Float[np.ndarray, "basis action"]
    initial_hidden: Float[np.ndarray, " hidden"] | None = None

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=np.float64)
        recurrent = np.asarray(self.recurrent_coefficients, dtype=np.float64)
        observation = np.asarray(self.observation_coefficients, dtype=np.float64)
        previous_action = np.asarray(self.previous_action_coefficients, dtype=np.float64)
        hidden_bias = np.asarray(self.hidden_bias_coefficients, dtype=np.float64)
        readout = np.asarray(self.readout_coefficients, dtype=np.float64)
        feedthrough = np.asarray(self.feedthrough_coefficients, dtype=np.float64)
        action_bias = np.asarray(self.action_bias_coefficients, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] <= 0 or basis.shape[1] <= 0:
            raise ValueError("basis must have shape (horizon, basis)")
        n_basis = basis.shape[1]
        if recurrent.ndim != 3 or recurrent.shape[0] != n_basis:
            raise ValueError("recurrent_coefficients must have shape (basis, hidden, hidden)")
        hidden_dim = recurrent.shape[1]
        if recurrent.shape[2] != hidden_dim:
            raise ValueError("recurrent coefficients must be square in hidden dimensions")
        if observation.ndim != 3 or observation.shape[:2] != (n_basis, hidden_dim):
            raise ValueError("observation_coefficients must have shape (basis, hidden, obs)")
        observation_dim = observation.shape[2]
        if readout.ndim != 3 or readout.shape[0] != n_basis or readout.shape[2] != hidden_dim:
            raise ValueError("readout_coefficients must have shape (basis, action, hidden)")
        action_dim = readout.shape[1]
        if previous_action.shape != (n_basis, hidden_dim, action_dim):
            raise ValueError("previous_action_coefficients has incompatible shape")
        if hidden_bias.shape != (n_basis, hidden_dim):
            raise ValueError("hidden_bias_coefficients has incompatible shape")
        if feedthrough.shape != (n_basis, action_dim, observation_dim):
            raise ValueError("feedthrough_coefficients has incompatible shape")
        if action_bias.shape != (n_basis, action_dim):
            raise ValueError("action_bias_coefficients has incompatible shape")
        if self.initial_hidden is None:
            initial_hidden = np.zeros((hidden_dim,), dtype=np.float64)
        else:
            initial_hidden = np.asarray(self.initial_hidden, dtype=np.float64)
            if initial_hidden.shape != (hidden_dim,):
                raise ValueError("initial_hidden must have shape (hidden,)")

        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "recurrent_coefficients", recurrent)
        object.__setattr__(self, "observation_coefficients", observation)
        object.__setattr__(self, "previous_action_coefficients", previous_action)
        object.__setattr__(self, "hidden_bias_coefficients", hidden_bias)
        object.__setattr__(self, "readout_coefficients", readout)
        object.__setattr__(self, "feedthrough_coefficients", feedthrough)
        object.__setattr__(self, "action_bias_coefficients", action_bias)
        object.__setattr__(self, "initial_hidden", initial_hidden)

    @property
    def horizon(self) -> int:
        """Number of rollout steps represented by the phase basis."""

        return int(self.basis.shape[0])

    @property
    def hidden_dim(self) -> int:
        """Hidden-state dimension."""

        return int(self.recurrent_coefficients.shape[1])

    @property
    def observation_dim(self) -> int:
        """Observation dimension."""

        return int(self.observation_coefficients.shape[2])

    @property
    def action_dim(self) -> int:
        """Action dimension."""

        return int(self.readout_coefficients.shape[1])

    def matrix_sequence(self, coefficients: np.ndarray) -> np.ndarray:
        """Expand ``(basis, ...)`` coefficients into ``(time, ...)`` arrays."""

        return np.einsum("tb,b...->t...", self.basis, np.asarray(coefficients))

    def matrices_at(self, time_index: int) -> dict[str, np.ndarray]:
        """Return all modulated matrices at one time index."""

        if time_index < 0 or time_index >= self.horizon:
            raise IndexError("time_index out of range")
        weights = self.basis[time_index]
        return {
            "A_h": np.einsum("b,bij->ij", weights, self.recurrent_coefficients),
            "B_y": np.einsum("b,bij->ij", weights, self.observation_coefficients),
            "B_u": np.einsum("b,bij->ij", weights, self.previous_action_coefficients),
            "b_h": np.einsum("b,bi->i", weights, self.hidden_bias_coefficients),
            "C_h": np.einsum("b,bij->ij", weights, self.readout_coefficients),
            "D_y": np.einsum("b,bij->ij", weights, self.feedthrough_coefficients),
            "c": np.einsum("b,bi->i", weights, self.action_bias_coefficients),
        }

    def action(
        self,
        time_index: int,
        hidden: Float[np.ndarray, "... hidden"],
        observation: Float[np.ndarray, "... observation"],
    ) -> Float[np.ndarray, "... action"]:
        """Return the action at one time index."""

        matrices = self.matrices_at(time_index)
        return hidden @ matrices["C_h"].T + observation @ matrices["D_y"].T + matrices["c"]

    def next_hidden(
        self,
        time_index: int,
        hidden: Float[np.ndarray, "... hidden"],
        observation: Float[np.ndarray, "... observation"],
        previous_action: Float[np.ndarray, "... action"] | None = None,
    ) -> Float[np.ndarray, "... hidden"]:
        """Return the next hidden state at one time index."""

        matrices = self.matrices_at(time_index)
        previous = (
            _zero_like_batch(hidden, self.action_dim)
            if previous_action is None
            else previous_action
        )
        return (
            hidden @ matrices["A_h"].T
            + observation @ matrices["B_y"].T
            + previous @ matrices["B_u"].T
            + matrices["b_h"]
        )

    def stability_diagnostics(self) -> dict[str, float]:
        """Return spectral diagnostics for modulated recurrent matrices."""

        recurrent = self.matrix_sequence(self.recurrent_coefficients)
        radii = np.asarray([recurrent_spectral_radius(matrix) for matrix in recurrent])
        return {
            "basis_rank": int(self.basis.shape[1]),
            "hidden_dim": self.hidden_dim,
            "observation_dim": self.observation_dim,
            "recurrent_spectral_radius_mean": float(np.mean(radii)),
            "recurrent_spectral_radius_max": float(np.max(radii)),
        }


@dataclass(frozen=True)
class LinearRecurrentController:
    """Linear recurrent bridge controller.

    The recurrence is

    ``h_{t+1} = A_h h_t + B_y y_t + B_u u_{t-1} + B_phi phi_t + b``
    ``u_t = C_h h_t + D_y y_t + D_phi phi_t + c``

    where ``y_t`` is the rollout observation, ``phi_t`` is an optional phase/time
    feature vector, and ``u_t`` is applied directly to the plant.  The previous
    action and phase terms default to zero, preserving the older
    observation-only recurrence.
    """

    recurrent_weights: Float[np.ndarray, "hidden hidden"]
    observation_weights: Float[np.ndarray, "hidden observation"]
    readout_weights: Float[np.ndarray, "action hidden"]
    feedthrough_weights: Float[np.ndarray, "action observation"] | None = None
    previous_action_weights: Float[np.ndarray, "hidden action"] | None = None
    phase_weights: Float[np.ndarray, "hidden phase"] | None = None
    hidden_bias: Float[np.ndarray, " hidden"] | None = None
    readout_phase_weights: Float[np.ndarray, "action phase"] | None = None
    action_bias: Float[np.ndarray, " action"] | None = None
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
        action_dim = readout.shape[0]
        hidden_dim = recurrent.shape[0]
        if self.previous_action_weights is None:
            previous_action = np.zeros((hidden_dim, action_dim), dtype=np.float64)
        else:
            previous_action = np.asarray(self.previous_action_weights, dtype=np.float64)
            expected = (hidden_dim, action_dim)
            if previous_action.shape != expected:
                raise ValueError(f"previous_action_weights must have shape {expected}")
        phase_dim = 0
        if self.phase_weights is None:
            phase = np.zeros((hidden_dim, 0), dtype=np.float64)
        else:
            phase = np.asarray(self.phase_weights, dtype=np.float64)
            if phase.ndim != 2 or phase.shape[0] != hidden_dim:
                raise ValueError("phase_weights must have shape (hidden, phase)")
            phase_dim = int(phase.shape[1])
        if self.hidden_bias is None:
            hidden_bias = np.zeros((hidden_dim,), dtype=np.float64)
        else:
            hidden_bias = np.asarray(self.hidden_bias, dtype=np.float64)
            if hidden_bias.shape != (hidden_dim,):
                raise ValueError("hidden_bias must have shape (hidden,)")
        if self.readout_phase_weights is None:
            readout_phase = np.zeros((action_dim, phase_dim), dtype=np.float64)
        else:
            readout_phase = np.asarray(self.readout_phase_weights, dtype=np.float64)
            if phase_dim == 0:
                phase_dim = int(readout_phase.shape[1])
                phase = np.zeros((hidden_dim, phase_dim), dtype=np.float64)
            expected = (action_dim, phase_dim)
            if readout_phase.shape != expected:
                raise ValueError(f"readout_phase_weights must have shape {expected}")
        if self.action_bias is None:
            action_bias = np.zeros((action_dim,), dtype=np.float64)
        else:
            action_bias = np.asarray(self.action_bias, dtype=np.float64)
            if action_bias.shape != (action_dim,):
                raise ValueError("action_bias must have shape (action,)")
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
        object.__setattr__(self, "previous_action_weights", previous_action)
        object.__setattr__(self, "phase_weights", phase)
        object.__setattr__(self, "hidden_bias", hidden_bias)
        object.__setattr__(self, "readout_phase_weights", readout_phase)
        object.__setattr__(self, "action_bias", action_bias)
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

    @property
    def phase_dim(self) -> int:
        """Phase/time feature dimension expected by the recurrence."""

        assert self.phase_weights is not None
        return int(self.phase_weights.shape[1])

    def next_hidden(
        self,
        hidden: Float[np.ndarray, "... hidden"],
        observation: Float[np.ndarray, "... observation"],
        previous_action: Float[np.ndarray, "... action"] | None = None,
        phase: Float[np.ndarray, "... phase"] | None = None,
    ) -> Float[np.ndarray, "... hidden"]:
        """Return the next hidden state for one or more batch elements."""

        assert self.previous_action_weights is not None
        assert self.phase_weights is not None
        assert self.hidden_bias is not None
        previous = (
            _zero_like_batch(hidden, self.action_dim)
            if previous_action is None
            else previous_action
        )
        phase_input = _zero_like_batch(hidden, self.phase_dim) if phase is None else phase
        return (
            hidden @ self.recurrent_weights.T
            + observation @ self.observation_weights.T
            + previous @ self.previous_action_weights.T
            + phase_input @ self.phase_weights.T
            + self.hidden_bias
        )

    def action(
        self,
        hidden: Float[np.ndarray, "... hidden"],
        observation: Float[np.ndarray, "... observation"],
        phase: Float[np.ndarray, "... phase"] | None = None,
    ) -> Float[np.ndarray, "... action"]:
        """Return the controller action for one or more batch elements."""

        assert self.feedthrough_weights is not None
        assert self.readout_phase_weights is not None
        assert self.action_bias is not None
        phase_input = _zero_like_batch(hidden, self.phase_dim) if phase is None else phase
        return (
            hidden @ self.readout_weights.T
            + observation @ self.feedthrough_weights.T
            + phase_input @ self.readout_phase_weights.T
            + self.action_bias
        )

    def stability_diagnostics(self) -> dict[str, float]:
        """Return recurrence-only stability diagnostics."""

        assert self.previous_action_weights is not None
        assert self.phase_weights is not None
        assert self.hidden_bias is not None
        assert self.readout_phase_weights is not None
        assert self.action_bias is not None
        return {
            "recurrent_spectral_radius": recurrent_spectral_radius(self.recurrent_weights),
            "hidden_dim": self.hidden_dim,
            "observation_dim": self.observation_dim,
            "phase_dim": self.phase_dim,
            "previous_action_weight_norm": float(np.linalg.norm(self.previous_action_weights)),
            "phase_weight_norm": float(np.linalg.norm(self.phase_weights)),
            "readout_phase_weight_norm": float(np.linalg.norm(self.readout_phase_weights)),
            "hidden_bias_norm": float(np.linalg.norm(self.hidden_bias)),
            "action_bias_norm": float(np.linalg.norm(self.action_bias)),
        }


def _zero_like_batch(reference: np.ndarray, width: int) -> np.ndarray:
    return np.zeros((*reference.shape[:-1], width), dtype=np.asarray(reference).dtype)


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
    previous_action = np.zeros((batch_size, controller.action_dim), dtype=np.float64)

    plant_states = [states]
    hidden_states = [hidden]
    observations = []
    actions = []
    for t in range(horizon_value):
        y_t = states @ observations_matrix.T
        u_t = controller.action(hidden, y_t)
        states = states @ A.T + u_t @ B.T + disturbances_batch[:, t, :] @ Bw.T
        hidden = controller.next_hidden(hidden, y_t, previous_action)
        previous_action = u_t
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


def rollout_phase_modulated_linear_recurrent_controller(
    controller: PhaseModulatedLinearRecurrentController,
    plant: LinearPlantLike,
    x0: np.ndarray,
    *,
    observation_matrix: Float[np.ndarray, "observation state"] | None = None,
    disturbances: np.ndarray | None = None,
    initial_hidden: np.ndarray | None = None,
) -> BridgeRolloutBatch:
    """Roll a phase-modulated linear recurrent controller through a plant."""

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

    disturbances_batch, _horizon = _normalize_disturbances(
        disturbances,
        batch_size=batch_size,
        disturbance_dim=Bw.shape[1],
        horizon=controller.horizon,
    )
    hidden_source = controller.initial_hidden if initial_hidden is None else initial_hidden
    hidden = _as_batch(
        np.asarray(hidden_source, dtype=np.float64), width=controller.hidden_dim, name="hidden"
    )
    hidden = np.broadcast_to(hidden, (batch_size, controller.hidden_dim)).copy()
    previous_action = np.zeros((batch_size, controller.action_dim), dtype=np.float64)

    plant_states = [states]
    hidden_states = [hidden]
    observations = []
    actions = []
    for t in range(controller.horizon):
        y_t = states @ observations_matrix.T
        u_t = controller.action(t, hidden, y_t)
        states = states @ A.T + u_t @ B.T + disturbances_batch[:, t, :] @ Bw.T
        hidden = controller.next_hidden(t, hidden, y_t, previous_action)
        previous_action = u_t
        observations.append(y_t)
        actions.append(u_t)
        plant_states.append(states)
        hidden_states.append(hidden)

    hidden_array = np.stack(hidden_states, axis=1)
    diagnostics = controller.stability_diagnostics() | hidden_growth_diagnostics(hidden_array)
    diagnostics["phase_modulates_matrices"] = True
    return BridgeRolloutBatch(
        plant_states=np.stack(plant_states, axis=1),
        actions=np.stack(actions, axis=1),
        observations=np.stack(observations, axis=1),
        hidden_states=hidden_array,
        metadata={"controller": "phase_modulated_linear_recurrence", "diagnostics": diagnostics},
    )


__all__ = [
    "GainProjection",
    "LinearRecurrentController",
    "MatrixBasisProjection",
    "PhaseModulatedLinearRecurrentController",
    "TimeConstrainedGainParameterization",
    "clamped_bspline_time_basis",
    "hidden_growth_diagnostics",
    "project_matrix_sequence_to_basis",
    "recurrent_spectral_radius",
    "rollout_linear_recurrent_controller",
    "rollout_phase_modulated_linear_recurrent_controller",
]

"""C&S released-code stochastic forward-simulation lane.

The deterministic output-feedback lane in :mod:`rlrmp.analysis.math.output_feedback`
is an analytical scaffold. Crevecoeur & Scott's released MATLAB code also runs
Monte Carlo forward simulations with sampled sensory noise, motor/process
noise, and signal-dependent control noise. This module keeps that stochastic
contract separate so deterministic Phase 0/1/3 results remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from rlrmp.analysis.math import require_jax_x64
from rlrmp.analysis.math.cs_game_card import TARGET_POS
from rlrmp.analysis.math.hinf_riccati import CostSchedule, PlantLinearization, RiccatiSolution
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    kalman_estimator_gains,
    measurement_covariance,
    process_covariance,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)

EXTLQG_MATLAB_FUNCTION_CHAIN = ("extLQG", "computeOFC", "computeExtKalman")


@dataclass(frozen=True)
class CSReleasedStochasticNoiseConfig:
    """Canonical scalar settings for the released C&S stochastic noise contract.

    These are the game-contract noise settings shared by Phase 1, Phase 3, and
    GRU-facing run specs. Monte Carlo trial counts and diagnostic process-noise
    sweeps are intentionally kept outside this contract.
    """

    motor_covariance_scale: float = 1e-10
    process_covariance_scale: float = 1.0
    signal_dependent_scale: float = 0.02

    def summary(self) -> dict[str, float | str]:
        """Return a JSON-serializable description of the canonical settings."""

        return {
            "contract": "cs_released_stochastic_v1",
            "additive_motor_covariance": "input_image_state_covariance_B_B_T",
            "signal_dependent_motor_noise": "input_image_state_tensor_Csdn",
            "process_noise": "separate_output_feedback_process_covariance",
            "motor_covariance_scale": self.motor_covariance_scale,
            "process_covariance_scale": self.process_covariance_scale,
            "signal_dependent_scale": self.signal_dependent_scale,
        }


DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG = CSReleasedStochasticNoiseConfig()


@dataclass(frozen=True)
class CSNoiseCovariances:
    """Noise covariance contract for released-code C&S forward simulation.

    Attributes:
        sensory: Observation covariance for the delayed physical measurement,
            shape ``(n_obs, n_obs)``.
        motor: Additive state-space motor covariance, shape ``(n, n)``. This is
            the input-image covariance ``B @ B.T * scale`` rather than an
            isotropic full-state covariance; it mirrors C&S motor noise as an
            input-channel perturbation added to the state through the plant
            input image.
        process: Additive state/process covariance after plant propagation,
            shape ``(n, n)``.
        signal_dependent_state: C&S ``Csdn`` tensor, shape ``(n, m_u, m_u)``.
            The sampled term is
            ``sum_j standard_j * signal_dependent_state[:, :, j] @ u`` and is
            added directly to the state update.
    """

    sensory: Float[Array, "n_obs n_obs"]
    motor: Float[Array, "n n"]
    process: Float[Array, "n n"]
    signal_dependent_state: Float[Array, "n m_u m_u"]


@dataclass(frozen=True)
class CSForwardNoiseDraws:
    """Concrete sampled noise sequences for one stochastic forward pass."""

    sensory: Float[Array, "T n_obs"]
    motor: Float[Array, "T n"]
    process: Float[Array, "T n"]
    signal_dependent_standard: Float[Array, "T m_u"]


@dataclass(frozen=True)
class FixedStepPerturbation:
    """Optional fixed-step state perturbation hook.

    The perturbation is added after the stochastic plant update for the matching
    step. ``step=None`` or a zero vector is the default no-op.
    """

    step: int | None = None
    value: Float[Array, " n"] | None = None


@dataclass(frozen=True)
class CSStochasticRollout:
    """Output-feedback rollout under sampled released-code-style noise."""

    x: Float[Array, "T_plus_1 n"]
    x_hat: Float[Array, "T_plus_1 n"]
    y_clean: Float[Array, "T n_obs"]
    y: Float[Array, "T n_obs"]
    u_command: Float[Array, "T m_u"]
    u_applied: Float[Array, "T m_u"]
    motor_noise: Float[Array, "T n"]
    signal_dependent_standard: Float[Array, "T m_u"]
    signal_dependent_noise: Float[Array, "T n"]
    process_noise: Float[Array, "T n"]
    sensory_noise: Float[Array, "T n_obs"]
    adversary_epsilon: Float[Array, "T m_w"]
    perturbations: Float[Array, "T n"]
    peak_forward_velocity: float
    peak_forward_velocity_idx: int
    terminal_position_error: float
    control_effort: float


@dataclass(frozen=True)
class CSExtLQGComparatorPath:
    """Explicit path for the C&S extLQG comparator implementation.

    This is intentionally a path object, not a claim that the current local
    arrays are a complete MATLAB extLQG port. The function chain and expected
    arrays are represented in one place so the eventual port can replace the
    provisional LQR/Kalman arrays without changing stochastic rollout callers.
    """

    function_chain: tuple[str, ...]
    controller_gains: Float[Array, "T m_u n"]
    estimator_gains: Float[Array, "T n n_obs"]
    state_covariances: Float[Array, "T_plus_1 n n"]
    noise_covariances: CSNoiseCovariances
    parity_status: str
    n_iterations: int = 0
    expected_cost: float | None = None
    deterministic_initial_state_cost: float | None = None
    initial_covariance_trace_cost: float | None = None
    accumulated_noise_scalar_cost: float | None = None


@dataclass(frozen=True)
class SharedNoiseComparison:
    """LQG and robust stochastic rollouts evaluated on identical noise draws."""

    draws: CSForwardNoiseDraws
    lqg: CSStochasticRollout
    robust: CSStochasticRollout


def zero_noise_covariances(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> CSNoiseCovariances:
    """Return an all-zero stochastic-noise contract with C&S-compatible shapes."""

    return CSNoiseCovariances(
        sensory=jnp.zeros((config.n_phys, config.n_phys), dtype=jnp.float64),
        motor=jnp.zeros((plant.n, plant.n), dtype=jnp.float64),
        process=jnp.zeros((plant.n, plant.n), dtype=jnp.float64),
        signal_dependent_state=jnp.zeros((plant.n, plant.m_u, plant.m_u), dtype=jnp.float64),
    )


def default_cs_noise_covariances(
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
    *,
    noise_config: CSReleasedStochasticNoiseConfig = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    motor_covariance_scale: float | None = None,
    process_covariance_scale: float | None = None,
    signal_dependent_scale: float | None = None,
) -> CSNoiseCovariances:
    """Return the local C&S noise covariance contract.

    The defaults are the canonical released-stochastic C&S settings. Pass
    explicit scale overrides for diagnostics, such as process-noise sweeps.
    """

    motor_scale = (
        noise_config.motor_covariance_scale
        if motor_covariance_scale is None
        else motor_covariance_scale
    )
    process_scale = (
        noise_config.process_covariance_scale
        if process_covariance_scale is None
        else process_covariance_scale
    )
    sdn_scale = (
        noise_config.signal_dependent_scale
        if signal_dependent_scale is None
        else signal_dependent_scale
    )
    process = process_covariance(plant, config) * jnp.asarray(process_scale, dtype=jnp.float64)
    return CSNoiseCovariances(
        sensory=measurement_covariance(plant, config),
        motor=cs_additive_motor_state_covariance(plant, scale=motor_scale),
        process=process,
        signal_dependent_state=cs_signal_dependent_state_tensor(
            plant,
            scale=sdn_scale,
        ),
    )


def cs_additive_motor_state_covariance(
    plant: PlantLinearization,
    *,
    scale: float,
) -> Float[Array, "n n"]:
    """Return additive C&S motor covariance in the plant-input image."""

    scale_arr = jnp.asarray(scale, dtype=jnp.float64)
    return scale_arr * (plant.B.astype(jnp.float64) @ plant.B.astype(jnp.float64).T)


def cs_signal_dependent_state_tensor(
    plant: PlantLinearization,
    *,
    scale: float = 0.1,
) -> Float[Array, "n m_u m_u"]:
    """Return C&S ``Csdn`` with ``Csdn[:, i, i] = scale * B[:, i]``."""

    tensor = jnp.zeros((plant.n, plant.m_u, plant.m_u), dtype=jnp.float64)
    for i in range(plant.m_u):
        tensor = tensor.at[:, i, i].set(jnp.asarray(scale, dtype=jnp.float64) * plant.B[:, i])
    return tensor


def sample_forward_noise_draws(
    key: Array,
    *,
    T: int,
    covariances: CSNoiseCovariances,
) -> CSForwardNoiseDraws:
    """Sample seeded C&S forward-noise draws.

    Passing the returned object to several rollout arms enforces common random
    numbers for LQG-vs-robust comparisons.
    """

    keys = jr.split(key, 4)
    return CSForwardNoiseDraws(
        sensory=_sample_zero_mean_gaussian(keys[0], T, covariances.sensory),
        motor=_sample_zero_mean_gaussian(keys[1], T, covariances.motor),
        process=_sample_zero_mean_gaussian(keys[2], T, covariances.process),
        signal_dependent_standard=jr.normal(
            keys[3],
            (T, covariances.signal_dependent_state.shape[2]),
            dtype=jnp.float64,
        ),
    )


def zero_forward_noise_draws(
    *,
    T: int,
    plant: PlantLinearization,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> CSForwardNoiseDraws:
    """Return zero-valued forward-noise draws with stochastic-lane shapes."""

    return CSForwardNoiseDraws(
        sensory=jnp.zeros((T, config.n_phys), dtype=jnp.float64),
        motor=jnp.zeros((T, plant.n), dtype=jnp.float64),
        process=jnp.zeros((T, plant.n), dtype=jnp.float64),
        signal_dependent_standard=jnp.zeros((T, plant.m_u), dtype=jnp.float64),
    )


def build_extlqg_comparator_path(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    covariances: CSNoiseCovariances,
    schedule: CostSchedule | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> CSExtLQGComparatorPath:
    """Build the explicit extLQG comparator path scaffold.

    The shape contract matches the C&S released-code chain
    ``extLQG -> computeOFC -> computeExtKalman``. When ``schedule`` is omitted,
    this returns the historical scaffold: caller-supplied controller gains plus
    the local delayed Kalman estimator. When ``schedule`` is supplied, this runs
    the C&S fixed-point iteration and replaces the supplied controller gains
    with the extLQG gains.
    """

    require_jax_x64("C&S released-code ExtLQG comparator")
    if schedule is not None:
        result = solve_extlqg_fixed_point(
            plant,
            schedule,
            covariances,
            initial_estimator_gains=None,
            config=config,
        )
        return result

    estimator_gains = kalman_estimator_gains(plant, controller_gains, config)
    state_covariances = _kalman_state_covariance_sequence(
        plant,
        controller_gains,
        estimator_gains,
        config,
    )
    return CSExtLQGComparatorPath(
        function_chain=EXTLQG_MATLAB_FUNCTION_CHAIN,
        controller_gains=controller_gains.astype(jnp.float64),
        estimator_gains=estimator_gains,
        state_covariances=state_covariances,
        noise_covariances=covariances,
        parity_status=(
            "scaffold: preserves extLQG/computeOFC/computeExtKalman call "
            "surface; full MATLAB fixed-point iteration still replaces gains"
        ),
    )


def solve_extlqg_fixed_point(
    plant: PlantLinearization,
    schedule: CostSchedule,
    covariances: CSNoiseCovariances,
    *,
    initial_estimator_gains: Float[Array, "T n n_obs"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
    tol: float = 1e-14,
    max_iter: int = 100,
) -> CSExtLQGComparatorPath:
    """Port the released C&S ``extLQG`` fixed-point comparator.

    The implementation follows ``extLQG.m``: alternate the backward
    ``computeOFC`` control recursion and the forward ``computeExtKalman``
    estimator recursion until expected cost stops changing. The additive
    state-noise covariance is the sum of the forward-simulation motor and
    process covariances; this keeps the local split noise contract compatible
    with C&S's single ``oXi`` argument.
    """

    T = schedule.T
    H = delayed_observation_matrix(plant, config)
    estimator_gains = (
        jnp.zeros((T, plant.n, H.shape[0]), dtype=jnp.float64)
        if initial_estimator_gains is None
        else initial_estimator_gains.astype(jnp.float64)
    )
    state_noise = covariances.motor + covariances.process
    initial_covariance = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance,
        dtype=jnp.float64,
    )
    current = 1.0e6
    expected_cost = current
    deterministic_cost = 0.0
    initial_covariance_trace_cost = 0.0
    accumulated_noise_scalar_cost = 0.0
    controller_gains = jnp.zeros((T, plant.m_u, plant.n), dtype=jnp.float64)
    state_covariances = jnp.repeat(initial_covariance[None, :, :], T + 1, axis=0)

    for iteration in range(1, max_iter + 1):
        controller_gains, sx0, se0, scalar_cost = _compute_ofc(
            plant,
            schedule,
            estimator_gains,
            H,
            covariances.signal_dependent_state,
            state_noise,
            covariances.sensory,
        )
        estimator_gains, state_covariances = _compute_ext_kalman(
            plant,
            H,
            controller_gains,
            covariances.signal_dependent_state,
            state_noise,
            covariances.sensory,
            initial_covariance,
            initial_covariance,
        )
        x0 = _default_output_feedback_initial_state(plant, config)
        deterministic_cost = float(x0 @ sx0 @ x0)
        initial_covariance_trace_cost = float(jnp.trace((sx0 + se0) @ initial_covariance))
        accumulated_noise_scalar_cost = float(scalar_cost)
        expected_cost = (
            deterministic_cost
            + initial_covariance_trace_cost
            + accumulated_noise_scalar_cost
        )
        relative_change = abs(current - expected_cost) / max(abs(expected_cost), 1e-300)
        current = expected_cost
        if relative_change <= tol:
            break

    return CSExtLQGComparatorPath(
        function_chain=EXTLQG_MATLAB_FUNCTION_CHAIN,
        controller_gains=controller_gains,
        estimator_gains=estimator_gains,
        state_covariances=state_covariances,
        noise_covariances=covariances,
        parity_status="fixed_point: local port of extLQG/computeOFC/computeExtKalman",
        n_iterations=iteration,
        expected_cost=expected_cost,
        deterministic_initial_state_cost=deterministic_cost,
        initial_covariance_trace_cost=initial_covariance_trace_cost,
        accumulated_noise_scalar_cost=accumulated_noise_scalar_cost,
    )


def _compute_ofc(
    plant: PlantLinearization,
    schedule: CostSchedule,
    estimator_gains: Float[Array, "T n n_obs"],
    H: Float[Array, "n_obs n"],
    signal_dependent_state: Float[Array, "n m_u m_u"],
    state_noise: Float[Array, "n n"],
    sensory_noise: Float[Array, "n_obs n_obs"],
) -> tuple[
    Float[Array, "T m_u n"],
    Float[Array, "n n"],
    Float[Array, "n n"],
    Float[Array, ""],
]:
    """C&S ``computeOFC`` backward recursion."""

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    curr_sx = schedule.Q_f.astype(jnp.float64)
    curr_se = jnp.zeros_like(curr_sx)
    scalar = jnp.asarray(0.0, dtype=jnp.float64)
    gains_rev = []
    R_terminal = schedule.R[-1].astype(jnp.float64)
    for t in range(schedule.T - 1, -1, -1):
        value_sum = curr_sx + curr_se
        sdn = jnp.einsum(
            "nuj,nm,mdj->ud",
            signal_dependent_state,
            value_sum,
            signal_dependent_state,
        )
        K_est = estimator_gains[t]
        statedn = jnp.zeros_like(curr_sx)
        L_t = jnp.linalg.solve(R_terminal + B.T @ curr_sx @ B + sdn, B.T @ curr_sx @ A)
        prev_sx = curr_sx
        prev_se = curr_se
        curr_sx = schedule.Q[t].astype(jnp.float64) + A.T @ curr_sx @ (A - B @ L_t) + statedn
        curr_se = A.T @ prev_sx @ B @ L_t + (A - K_est @ H).T @ prev_se @ (A - K_est @ H)
        scalar = scalar + jnp.trace(
            prev_sx @ state_noise + prev_se @ (state_noise + K_est @ sensory_noise @ K_est.T)
        )
        gains_rev.append(L_t)
    return (
        jnp.stack(list(reversed(gains_rev)), axis=0),
        0.5 * (curr_sx + curr_sx.T),
        0.5 * (curr_se + curr_se.T),
        scalar,
    )


def _compute_ext_kalman(
    plant: PlantLinearization,
    H: Float[Array, "n_obs n"],
    controller_gains: Float[Array, "T m_u n"],
    signal_dependent_state: Float[Array, "n m_u m_u"],
    state_noise: Float[Array, "n n"],
    sensory_noise: Float[Array, "n_obs n_obs"],
    initial_xhat_covariance: Float[Array, "n n"],
    initial_error_covariance: Float[Array, "n n"],
) -> tuple[Float[Array, "T n n_obs"], Float[Array, "T_plus_1 n n"]]:
    """C&S ``computeExtKalman`` forward recursion."""

    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    sigma_e = initial_error_covariance.astype(jnp.float64)
    sigma_x = initial_xhat_covariance.astype(jnp.float64)
    sigma_ex = jnp.zeros_like(sigma_e)
    gains = []
    state_covariances = [sigma_e]
    for L_t in controller_gains:
        s_temp = sigma_e + sigma_x + sigma_ex + sigma_ex.T
        statedn = jnp.zeros((H.shape[0], H.shape[0]), dtype=jnp.float64)
        sdn = jnp.einsum(
            "nuj,ua,ab,vb,mvj->nm",
            signal_dependent_state,
            L_t,
            sigma_x,
            L_t,
            signal_dependent_state,
        )
        innovation_cov = H @ sigma_e @ H.T + sensory_noise + statedn
        K_est = A @ sigma_e @ H.T @ jnp.linalg.inv(innovation_cov)
        gains.append(K_est)

        sigma_e_prev = sigma_e
        sigma_e = state_noise + (A - K_est @ H) @ sigma_e @ A.T + sdn
        term = (A - B @ L_t) @ sigma_ex @ H.T @ K_est.T
        sigma_x = (
            K_est @ H @ sigma_e_prev @ A.T
            + (A - B @ L_t) @ sigma_x @ (A - B @ L_t).T
            + term
            + term.T
        )
        sigma_ex = (A - B @ L_t) @ sigma_ex @ (A - K_est @ H).T
        sigma_e = 0.5 * (sigma_e + sigma_e.T)
        sigma_x = 0.5 * (sigma_x + sigma_x.T)
        state_covariances.append(sigma_e)
        _ = s_temp
    return jnp.stack(gains, axis=0), jnp.stack(state_covariances, axis=0)


def _default_output_feedback_initial_state(
    plant: PlantLinearization,
    config: OutputFeedbackConfig,
) -> Array:
    """Avoid an import cycle with ``output_feedback.make_cs_output_feedback_initial_state``."""

    x_phys = jnp.zeros((config.n_phys,), dtype=jnp.float64)
    pos_lo, pos_hi = plant.pos_slice
    x_phys = x_phys.at[pos_lo:pos_hi].set((-TARGET_POS).astype(jnp.float64))
    return jnp.tile(x_phys, config.delay_steps + 1)


def simulate_lqg_released_forward(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    *,
    draws: CSForwardNoiseDraws | None = None,
    covariances: CSNoiseCovariances | None = None,
    estimator_gains: Float[Array, "T n n_obs"] | None = None,
    adversary_epsilon: Float[Array, "T m_w"] | None = None,
    perturbation: FixedStepPerturbation = FixedStepPerturbation(),
    clean_observation_offset: Float[Array, "T n_obs"] | None = None,
    sensory_feedback_offset: Float[Array, "T n_obs"] | None = None,
    command_input_offset: Float[Array, "T m_u"] | None = None,
    initial_estimator_state: Float[Array, " n"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> CSStochasticRollout:
    """Simulate the released-code stochastic lane for the LQG comparator arm."""

    T = int(controller_gains.shape[0])
    covs = zero_noise_covariances(plant, config) if covariances is None else covariances
    noise = zero_forward_noise_draws(T=T, plant=plant, config=config) if draws is None else draws
    gains = (
        kalman_estimator_gains(plant, controller_gains, config)
        if estimator_gains is None
        else estimator_gains
    )

    def estimator_update(t, x_t, xhat_t, y_t, u_command):
        return (
            plant.A @ xhat_t
            + plant.B @ u_command
            + gains[t] @ (y_t - delayed_observation_matrix(plant, config) @ xhat_t)
        )

    return _simulate_released_forward(
        plant,
        controller_gains,
        x0,
        noise,
        covs,
        estimator_update,
        adversary_epsilon=adversary_epsilon,
        perturbation=perturbation,
        clean_observation_offset=clean_observation_offset,
        sensory_feedback_offset=sensory_feedback_offset,
        command_input_offset=command_input_offset,
        initial_estimator_state=initial_estimator_state,
        config=config,
    )


def simulate_robust_released_forward(
    plant: PlantLinearization,
    schedule,
    solution: RiccatiSolution,
    x0: Float[Array, " n"],
    *,
    draws: CSForwardNoiseDraws | None = None,
    covariances: CSNoiseCovariances | None = None,
    gains: Float[Array, "T m_u n"] | None = None,
    adversary_epsilon: Float[Array, "T m_w"] | None = None,
    perturbation: FixedStepPerturbation = FixedStepPerturbation(),
    command_input_offset: Float[Array, "T m_u"] | None = None,
    initial_estimator_state: Float[Array, " n"] | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> CSStochasticRollout:
    """Simulate the released-code stochastic lane for the robust arm."""

    covs = zero_noise_covariances(plant, config) if covariances is None else covariances
    noise = (
        zero_forward_noise_draws(T=schedule.T, plant=plant, config=config)
        if draws is None
        else draws
    )
    estimator_covariances = robust_estimator_covariances(
        plant,
        schedule,
        solution.gamma,
        config,
    )
    controller_gains = (
        robust_output_feedback_gains(plant, schedule, solution, estimator_covariances, config)
        if gains is None
        else gains
    )
    H = delayed_observation_matrix(plant, config)
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)

    def estimator_update(t, x_t, xhat_t, y_t, u_command):
        del x_t
        Sigma = estimator_covariances[t]
        precision = jnp.linalg.inv(Sigma) + H.T @ H - inv_gamma2 * schedule.Q[t]
        middle = jnp.linalg.inv(precision)
        innovation = y_t - H @ xhat_t
        correction = inv_gamma2 * schedule.Q[t] @ xhat_t + H.T @ innovation
        return plant.A @ xhat_t + plant.B @ u_command + plant.A @ middle @ correction

    return _simulate_released_forward(
        plant,
        controller_gains,
        x0,
        noise,
        covs,
        estimator_update,
        adversary_epsilon=adversary_epsilon,
        perturbation=perturbation,
        command_input_offset=command_input_offset,
        initial_estimator_state=initial_estimator_state,
        config=config,
    )


def simulate_shared_noise_lqg_vs_robust(
    key: Array,
    *,
    plant: PlantLinearization,
    schedule,
    lqg_gains: Float[Array, "T m_u n"],
    robust_solution: RiccatiSolution,
    x0: Float[Array, " n"],
    covariances: CSNoiseCovariances,
    robust_gains: Float[Array, "T m_u n"] | None = None,
    perturbation: FixedStepPerturbation = FixedStepPerturbation(),
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> SharedNoiseComparison:
    """Evaluate LQG and robust arms on identical sampled noise draws."""

    draws = sample_forward_noise_draws(key, T=schedule.T, covariances=covariances)
    lqg = simulate_lqg_released_forward(
        plant,
        lqg_gains,
        x0,
        draws=draws,
        covariances=covariances,
        perturbation=perturbation,
        config=config,
    )
    robust = simulate_robust_released_forward(
        plant,
        schedule,
        robust_solution,
        x0,
        draws=draws,
        covariances=covariances,
        gains=robust_gains,
        perturbation=perturbation,
        config=config,
    )
    return SharedNoiseComparison(draws=draws, lqg=lqg, robust=robust)


def _simulate_released_forward(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    x0: Float[Array, " n"],
    draws: CSForwardNoiseDraws,
    covariances: CSNoiseCovariances,
    estimator_update,
    *,
    adversary_epsilon: Float[Array, "T m_w"] | None,
    perturbation: FixedStepPerturbation,
    clean_observation_offset: Float[Array, "T n_obs"] | None = None,
    sensory_feedback_offset: Float[Array, "T n_obs"] | None = None,
    command_input_offset: Float[Array, "T m_u"] | None = None,
    initial_estimator_state: Float[Array, " n"] | None = None,
    config: OutputFeedbackConfig,
) -> CSStochasticRollout:
    """Shared released-code stochastic forward simulation core."""

    T = int(controller_gains.shape[0])
    H = delayed_observation_matrix(plant, config)
    eps = (
        jnp.zeros((T, plant.m_w), dtype=jnp.float64)
        if adversary_epsilon is None
        else adversary_epsilon.astype(jnp.float64)
    )
    clean_offset = _observation_offset_sequence(
        T,
        H.shape[0],
        clean_observation_offset,
        label="clean_observation_offset",
    )
    feedback_offset = _observation_offset_sequence(
        T,
        H.shape[0],
        sensory_feedback_offset,
        label="sensory_feedback_offset",
    )
    command_offset = _command_input_offset_sequence(
        T,
        plant.m_u,
        command_input_offset,
        label="command_input_offset",
    )
    perturb = _perturbation_sequence(T, plant.n, perturbation)
    x_seq = [x0.astype(jnp.float64)]
    xhat0 = (
        x0.astype(jnp.float64)
        if initial_estimator_state is None
        else initial_estimator_state.astype(jnp.float64)
    )
    if xhat0.shape != (plant.n,):
        raise ValueError(f"initial_estimator_state shape must be ({plant.n},); got {xhat0.shape}.")
    xhat_seq = [xhat0]
    y_clean_seq = []
    y_seq = []
    u_command_seq = []
    u_applied_seq = []
    motor_seq = []
    sdn_seq = []
    process_seq = []
    sensory_seq = []

    for t in range(T):
        x_t = x_seq[-1]
        xhat_t = xhat_seq[-1]
        y_clean = H @ x_t + clean_offset[t]
        sensory = draws.sensory[t]
        y_t = y_clean + sensory + feedback_offset[t]
        u_command = -controller_gains[t] @ xhat_t
        motor = draws.motor[t]
        signal_dependent = jnp.einsum(
            "j,nmj,m->n",
            draws.signal_dependent_standard[t],
            covariances.signal_dependent_state,
            u_command,
        )
        u_applied = u_command + command_offset[t]
        process = draws.process[t]
        xhat_next = estimator_update(t, x_t, xhat_t, y_t, u_command)
        x_next = (
            plant.A @ x_t
            + plant.B @ u_applied
            + plant.Bw @ eps[t]
            + motor
            + signal_dependent
            + process
            + perturb[t]
        )
        y_clean_seq.append(y_clean)
        y_seq.append(y_t)
        u_command_seq.append(u_command)
        u_applied_seq.append(u_applied)
        motor_seq.append(motor)
        sdn_seq.append(signal_dependent)
        process_seq.append(process)
        sensory_seq.append(sensory + feedback_offset[t])
        x_seq.append(x_next)
        xhat_seq.append(xhat_next)

    x = jnp.stack(x_seq, axis=0)
    u_applied = jnp.stack(u_applied_seq, axis=0)
    peak, peak_idx, terminal, effort = _summary_fields(plant, x, u_applied)
    return CSStochasticRollout(
        x=x,
        x_hat=jnp.stack(xhat_seq, axis=0),
        y_clean=jnp.stack(y_clean_seq, axis=0),
        y=jnp.stack(y_seq, axis=0),
        u_command=jnp.stack(u_command_seq, axis=0),
        u_applied=u_applied,
        motor_noise=jnp.stack(motor_seq, axis=0),
        signal_dependent_standard=draws.signal_dependent_standard,
        signal_dependent_noise=jnp.stack(sdn_seq, axis=0),
        process_noise=jnp.stack(process_seq, axis=0),
        sensory_noise=jnp.stack(sensory_seq, axis=0),
        adversary_epsilon=eps,
        perturbations=perturb,
        peak_forward_velocity=peak,
        peak_forward_velocity_idx=peak_idx,
        terminal_position_error=terminal,
        control_effort=effort,
    )


def _kalman_state_covariance_sequence(
    plant: PlantLinearization,
    controller_gains: Float[Array, "T m_u n"],
    estimator_gains: Float[Array, "T n n_obs"],
    config: OutputFeedbackConfig,
) -> Float[Array, "T_plus_1 n n"]:
    """Return Kalman covariance arrays for the extLQG comparator path."""

    del controller_gains
    H = delayed_observation_matrix(plant, config)
    Q_proc = process_covariance(plant, config)
    Sigma = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance,
        dtype=jnp.float64,
    )
    covariances = [Sigma]
    for gain in estimator_gains:
        Sigma = (plant.A - gain @ H) @ Sigma @ plant.A.T + Q_proc
        Sigma = 0.5 * (Sigma + Sigma.T)
        covariances.append(Sigma)
    return jnp.stack(covariances, axis=0)


def _sample_zero_mean_gaussian(
    key: Array,
    T: int,
    covariance: Float[Array, "n n"],
) -> Float[Array, "T n"]:
    """Sample a zero-mean Gaussian sequence, accepting singular zero covariances."""

    covariance = covariance.astype(jnp.float64)
    dim = int(covariance.shape[0])
    cov_sym = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = jnp.linalg.eigh(cov_sym)
    scale = eigvecs @ jnp.diag(jnp.sqrt(jnp.clip(eigvals, min=0.0)))
    standard = jr.normal(key, (T, dim), dtype=jnp.float64)
    return standard @ scale.T


def _perturbation_sequence(
    T: int,
    n: int,
    perturbation: FixedStepPerturbation,
) -> Float[Array, "T n"]:
    """Materialize a fixed-step perturbation as a dense time sequence."""

    perturb = jnp.zeros((T, n), dtype=jnp.float64)
    if perturbation.step is None:
        return perturb
    if perturbation.step < 0 or perturbation.step >= T:
        raise ValueError(f"perturbation step must be in [0, {T}); got {perturbation.step}.")
    value = (
        jnp.zeros((n,), dtype=jnp.float64)
        if perturbation.value is None
        else perturbation.value.astype(jnp.float64)
    )
    if value.shape != (n,):
        raise ValueError(f"perturbation value shape must be ({n},); got {value.shape}.")
    return perturb.at[perturbation.step].set(value)


def _observation_offset_sequence(
    T: int,
    n_obs: int,
    offset: Float[Array, "T n_obs"] | None,
    *,
    label: str,
) -> Float[Array, "T n_obs"]:
    """Return a validated dense observation-channel offset sequence."""

    if offset is None:
        return jnp.zeros((T, n_obs), dtype=jnp.float64)
    values = offset.astype(jnp.float64)
    if values.shape != (T, n_obs):
        raise ValueError(f"{label} shape must be ({T}, {n_obs}); got {values.shape}.")
    return values


def _command_input_offset_sequence(
    T: int,
    m_u: int,
    offset: Float[Array, "T m_u"] | None,
    *,
    label: str,
) -> Float[Array, "T m_u"]:
    """Return a validated dense command-input offset sequence."""

    if offset is None:
        return jnp.zeros((T, m_u), dtype=jnp.float64)
    values = offset.astype(jnp.float64)
    if values.shape != (T, m_u):
        raise ValueError(f"{label} shape must be ({T}, {m_u}); got {values.shape}.")
    return values


def _summary_fields(
    plant: PlantLinearization,
    x: Float[Array, "T_plus_1 n"],
    u: Float[Array, "T m_u"],
) -> tuple[float, int, float, float]:
    """Return C&S scalar rollout summary fields."""

    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos_abs = pos + TARGET_POS[None, :]
    terminal = jnp.linalg.norm(pos_abs[-1] - TARGET_POS)
    return (
        float(jnp.max(forward)),
        int(jnp.argmax(forward)),
        float(terminal),
        float(jnp.sum(jnp.linalg.norm(u, axis=-1) ** 2) * plant.dt),
    )

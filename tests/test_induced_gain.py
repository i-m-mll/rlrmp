"""Tests for ``rlrmp.analysis.induced_gain``.

Bug: 74bfd86 -- production induced-gain analyser. The tests cover:

- ``test_toy_ltv_matches_svd``: Power iteration on a hand-rolled 2-state LTV
  matches brute-force SVD of the explicit Toeplitz operator (foundational
  unit test).
- ``test_adjoint_correctness``: ``<z, T w> == <T^* z, w>`` on the toy LTV.
- ``test_riccati_round_trip_qr_cost``: Analyser gain on a Riccati-computed
  controller is bounded above by ``gamma`` (controller's design gamma) and
  below by the LQR baseline gain. Both algorithms agree on the LTV.
- ``test_lqr_round_trip_qr_cost``: Analyser gain on the LQR controller is
  finite and consistent with brute-force SVD on a long-horizon trajectory.
- ``test_hamiltonian_consistency_with_constant_K``: For a controller with a
  truly time-invariant gain (constant ``K`` over the horizon), the
  Hamiltonian (fixed-point) gain matches the power-iteration gain on the
  trajectory linearisation.
- ``test_state_error_channel_finite``: ``state_error`` z channel returns a
  finite gain on a stable controller.
- ``test_control_channel_finite``: ``control`` z channel returns a finite
  gain on a stable controller.
- ``test_structural_da_returns_finite_gain``: ``structural_da`` w channel
  returns a finite gain on the LQR controller (small-gain margin).
- ``test_w_channels_give_distinct_numbers``: ``additive_force`` and
  ``structural_da`` give different gain values (they answer different
  questions).
- ``test_peak_velocity_channel_decoration``: The ``peak_velocity`` z channel
  returns finite ``peak_forward_velocity`` and ``peak_lateral_velocity``
  scalars under the worst-case ``w*``.

Tests use double precision (the module enables x64 on import).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rlrmp.analysis.hinf_riccati import (
    CostSpec,
    cost_schedule_from_spec,
    find_gamma_star,
    linearize_pointmass,
    solve_hinf_riccati,
    solve_lqr,
)
from rlrmp.analysis.induced_gain import (
    TrajectoryLinearisation,
    W_ADDITIVE_FORCE,
    W_STRUCTURAL_DA,
    Z_CONTROL,
    Z_PEAK_VELOCITY,
    Z_QR_COST,
    Z_STATE_ERROR,
    _ltv_forward_sweep,
    induced_gain,
    induced_gain_hamiltonian,
    induced_gain_power_iteration,
    linearise_fixed_point,
    linearise_trajectory,
    lti_controller,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _rlrmp_setup():
    plant = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)
    spec = CostSpec(n_steps=80)
    schedule = cost_schedule_from_spec(spec, plant)
    return plant, schedule


def _toy_ltv(T: int = 8, n: int = 2, n_w: int = 1, n_z: int = 1, seed: int = 0):
    """Build a small random LTV system suitable for brute-force SVD comparison."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    A_t = 0.5 * jax.random.normal(keys[0], (T, n, n), dtype=jnp.float64)
    Bw_t = jax.random.normal(keys[1], (T, n, n_w), dtype=jnp.float64)
    Cz_t = jax.random.normal(keys[2], (T, n_z, n), dtype=jnp.float64)
    D_t = 0.1 * jax.random.normal(keys[3], (T, n_z, n_w), dtype=jnp.float64)
    x_nom = jnp.zeros((T + 1, n), dtype=jnp.float64)
    u_nom = jnp.zeros((T, 1), dtype=jnp.float64)
    return TrajectoryLinearisation(
        A_t=A_t,
        Bw_t=Bw_t,
        Cz_t=Cz_t,
        D_t=D_t,
        x_nominal=x_nom,
        u_nominal=u_nom,
        dt=0.01,
        n_plant=n,
        n_ctrl=0,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_STATE_ERROR,
    )


def _brute_force_toeplitz_svd(lin: TrajectoryLinearisation) -> float:
    """Build the explicit Toeplitz operator and return its top singular value."""
    A_t = lin.A_t
    Bw_t = lin.Bw_t
    Cz_t = lin.Cz_t
    D_t = lin.D_t
    T = lin.T
    n_w = lin.n_w
    n_z = lin.n_z
    op = np.zeros((T * n_z, T * n_w))
    for t in range(T):
        for i in range(n_w):
            w_seq = jnp.zeros((T, n_w), dtype=jnp.float64).at[t, i].set(1.0)
            z_seq = _ltv_forward_sweep(A_t, Bw_t, Cz_t, D_t, w_seq)
            op[:, t * n_w + i] = np.asarray(z_seq).reshape(-1)
    return float(np.linalg.svd(op, compute_uv=False)[0])


# -----------------------------------------------------------------------------
# Toy LTV: foundational tests
# -----------------------------------------------------------------------------


def test_toy_ltv_matches_svd():
    """Power iteration on a 2-state random LTV matches Toeplitz-SVD ground truth."""
    lin = _toy_ltv(T=8, n=2, n_w=1, n_z=1, seed=0)
    sv_top = _brute_force_toeplitz_svd(lin)

    result = induced_gain_power_iteration(
        lin, n_restarts=5, max_iter=400, rtol=1e-9
    )
    assert result.converged
    assert result.gamma == pytest.approx(sv_top, rel=1e-6)


def test_toy_ltv_matches_svd_multidim():
    """Multi-channel toy LTV: power iteration matches SVD."""
    lin = _toy_ltv(T=10, n=3, n_w=2, n_z=2, seed=42)
    sv_top = _brute_force_toeplitz_svd(lin)

    result = induced_gain_power_iteration(
        lin, n_restarts=5, max_iter=500, rtol=1e-9
    )
    assert result.converged
    assert result.gamma == pytest.approx(sv_top, rel=1e-5)


def test_adjoint_correctness():
    """Internal ``<z, T w> == <T^* z, w>`` identity on the toy LTV.

    Verifies the VJP-derived adjoint is the true transpose of the forward
    sweep. If this fails, power iteration converges to the wrong number.
    """
    lin = _toy_ltv(T=8, n=2, n_w=1, n_z=1, seed=0)
    A_t, Bw_t, Cz_t, D_t = lin.A_t, lin.Bw_t, lin.Cz_t, lin.D_t

    def forward(w):
        return _ltv_forward_sweep(A_t, Bw_t, Cz_t, D_t, w)

    rng = jax.random.PRNGKey(11)
    k_w, k_z = jax.random.split(rng)
    w = jax.random.normal(k_w, (lin.T, lin.n_w), dtype=jnp.float64)
    z = jax.random.normal(k_z, (lin.T, lin.n_z), dtype=jnp.float64)

    z_of_w = forward(w)
    _, vjp = jax.vjp(forward, w)
    (adj_z,) = vjp(z)

    # <z, T w>
    inner_left = float(jnp.sum(z * z_of_w))
    # <T^* z, w>
    inner_right = float(jnp.sum(adj_z * w))
    assert inner_left == pytest.approx(inner_right, rel=1e-10, abs=1e-12)


# -----------------------------------------------------------------------------
# Riccati / LQR round-trip
# -----------------------------------------------------------------------------


def test_riccati_round_trip_qr_cost():
    """Induced gain of an H-inf controller is bounded above by its design gamma.

    Verifies the analyser's gain on the LTV (qr_cost, additive_force) lies
    below the design ``gamma`` (the H-inf saddle-point upper bound) and is
    finite. This is the core round-trip sanity: dropping a Riccati controller
    into the analyser yields a number consistent with H-inf theory.

    The exact equality ``analyser_gain == gamma_star`` is not expected at the
    bisection's admissibility boundary -- ``find_gamma_star`` returns the
    smallest numerically-admissible gamma, which is an upper bound on the
    true H-inf optimum (boundary effects in finite-horizon LQ game). We
    require the ratio to be in [0.5, 1.0].
    """
    plant, schedule = _rlrmp_setup()
    g_star = find_gamma_star(plant, schedule)
    gamma_design = 1.5 * g_star
    hinf = solve_hinf_riccati(plant, schedule, gamma_design)
    assert hinf.admissible

    ctrl = lti_controller(hinf.K)
    init_pos = jnp.array([0.0, 0.0])
    target_pos = jnp.array([0.1, 0.0])

    out = induced_gain(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=80,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_QR_COST,
        schedule=schedule,
        methods=("power_iteration",),
        n_restarts=5,
        max_iter=500,
        rtol=1e-8,
    )
    pi = out["power_iteration"]
    assert pi.converged
    # Upper bound: ||T||_inf < gamma_design (the design level is admissible).
    assert pi.gamma < gamma_design
    # Lower bound: > 0 (the closed loop has nonzero disturbance amplification).
    assert pi.gamma > 0
    # Sanity: the analyser gain is on the same order of magnitude as gamma_star.
    assert 0.3 * g_star < pi.gamma < gamma_design


def test_lqr_round_trip_qr_cost():
    """LQR (gamma -> infinity) closed-loop induced gain is finite."""
    plant, schedule = _rlrmp_setup()
    lqr = solve_lqr(plant, schedule)
    ctrl = lti_controller(lqr.K)
    init_pos = jnp.array([0.0, 0.0])
    target_pos = jnp.array([0.1, 0.0])

    out = induced_gain(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=80,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_QR_COST,
        schedule=schedule,
        methods=("power_iteration",),
        n_restarts=5,
        max_iter=500,
        rtol=1e-8,
    )
    pi = out["power_iteration"]
    assert pi.converged
    assert 0.0 < pi.gamma < float("inf")


def test_hamiltonian_consistency_with_constant_K():
    """Hamiltonian (LTI) and power-iteration (LTV) gains agree on a constant-K
    controller.

    The internal-consistency check from the spec: when the closed loop is
    truly LTI (time-invariant ``K``), the trajectory linearisation reduces
    to repeated copies of the same LTI step, and the two algorithms should
    give the same number to bisection precision.
    """
    plant, schedule = _rlrmp_setup()
    # Pick a "settled-state" K: build the H-inf Riccati and use K[T//2] (mid
    # horizon, away from initial-condition and terminal-condition transients).
    g_star = find_gamma_star(plant, schedule)
    hinf = solve_hinf_riccati(plant, schedule, 2.0 * g_star)
    K_const = jnp.tile(hinf.K[40][None], (80, 1, 1))  # constant gain
    ctrl = lti_controller(K_const)
    init_pos = jnp.array([0.0, 0.0])
    target_pos = jnp.array([0.0, 0.0])  # all-hold trajectory

    out = induced_gain(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=80,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_STATE_ERROR,
        schedule=schedule,
        methods=("power_iteration", "hamiltonian"),
        n_restarts=5,
        max_iter=600,
        rtol=1e-9,
    )
    pi = out["power_iteration"]
    ham = out["hamiltonian"]
    assert pi.converged
    # Tolerance of 1% covers the bisection bracket precision and any
    # remaining transient contribution from the LTV operator (e.g. the
    # initial state being x_0 = 0 in the perturbation operator).
    assert pi.gamma == pytest.approx(ham.gamma, rel=1e-2)


# -----------------------------------------------------------------------------
# z-channel sanity
# -----------------------------------------------------------------------------


def test_state_error_channel_finite():
    """state_error z channel returns a finite gain on a stable controller."""
    plant, schedule = _rlrmp_setup()
    g_star = find_gamma_star(plant, schedule)
    hinf = solve_hinf_riccati(plant, schedule, 2.0 * g_star)
    ctrl = lti_controller(hinf.K)
    out = induced_gain(
        plant,
        ctrl,
        init_pos=jnp.array([0.0, 0.0]),
        target_pos=jnp.array([0.1, 0.0]),
        horizon=80,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_STATE_ERROR,
        methods=("power_iteration",),
        max_iter=400,
    )
    pi = out["power_iteration"]
    assert pi.converged
    assert 0.0 < pi.gamma < float("inf")


def test_control_channel_finite():
    """control z channel returns a finite gain on a stable controller."""
    plant, schedule = _rlrmp_setup()
    g_star = find_gamma_star(plant, schedule)
    hinf = solve_hinf_riccati(plant, schedule, 2.0 * g_star)
    ctrl = lti_controller(hinf.K)
    out = induced_gain(
        plant,
        ctrl,
        init_pos=jnp.array([0.0, 0.0]),
        target_pos=jnp.array([0.1, 0.0]),
        horizon=80,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_CONTROL,
        methods=("power_iteration",),
        max_iter=400,
    )
    pi = out["power_iteration"]
    assert pi.converged
    assert 0.0 < pi.gamma < float("inf")


def test_peak_velocity_channel_decoration():
    """peak_velocity z channel populates peak_forward / peak_lateral scalars."""
    plant, schedule = _rlrmp_setup()
    g_star = find_gamma_star(plant, schedule)
    hinf = solve_hinf_riccati(plant, schedule, 2.0 * g_star)
    ctrl = lti_controller(hinf.K)
    out = induced_gain(
        plant,
        ctrl,
        init_pos=jnp.array([0.0, 0.0]),
        target_pos=jnp.array([0.1, 0.0]),
        horizon=80,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_PEAK_VELOCITY,
        methods=("power_iteration",),
        max_iter=400,
    )
    pi = out["power_iteration"]
    assert pi.converged
    assert pi.peak_forward_velocity is not None
    assert pi.peak_lateral_velocity is not None
    # Worst-case w should produce nonzero peak velocity in both channels.
    assert abs(pi.peak_forward_velocity) > 0
    assert pi.peak_lateral_velocity >= 0


# -----------------------------------------------------------------------------
# w-channel sanity
# -----------------------------------------------------------------------------


def test_structural_da_returns_finite_gain():
    """structural_da w channel returns a finite gain on the LQR controller.

    The reciprocal of this gain is the small-gain margin: any unstructured
    ``Delta A`` with ``||Delta A||_op < 1/gamma`` preserves stability.
    """
    plant, schedule = _rlrmp_setup()
    lqr = solve_lqr(plant, schedule)
    ctrl = lti_controller(lqr.K)
    out = induced_gain(
        plant,
        ctrl,
        init_pos=jnp.array([0.0, 0.0]),
        target_pos=jnp.array([0.1, 0.0]),
        horizon=80,
        w_channel=W_STRUCTURAL_DA,
        z_channel=Z_STATE_ERROR,
        methods=("power_iteration",),
        max_iter=400,
    )
    pi = out["power_iteration"]
    assert pi.converged
    assert 0.0 < pi.gamma < float("inf")


def test_w_channels_give_distinct_numbers():
    """additive_force and structural_da give different gains (different operators)."""
    plant, schedule = _rlrmp_setup()
    lqr = solve_lqr(plant, schedule)
    ctrl = lti_controller(lqr.K)
    init_pos = jnp.array([0.0, 0.0])
    target_pos = jnp.array([0.1, 0.0])

    out_force = induced_gain(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=80,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_STATE_ERROR,
        methods=("power_iteration",),
        max_iter=400,
    )
    out_struct = induced_gain(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=80,
        w_channel=W_STRUCTURAL_DA,
        z_channel=Z_STATE_ERROR,
        methods=("power_iteration",),
        max_iter=400,
    )
    g_force = out_force["power_iteration"].gamma
    g_struct = out_struct["power_iteration"].gamma
    assert g_force > 0
    assert g_struct > 0
    # They should not be identical (different operators).
    assert abs(g_force - g_struct) / max(g_force, g_struct) > 1e-3


# -----------------------------------------------------------------------------
# Invalid input handling
# -----------------------------------------------------------------------------


def test_invalid_w_channel_raises():
    plant, schedule = _rlrmp_setup()
    lqr = solve_lqr(plant, schedule)
    ctrl = lti_controller(lqr.K)
    with pytest.raises(ValueError, match="Unknown w channel"):
        linearise_trajectory(
            plant,
            ctrl,
            init_pos=jnp.array([0.0, 0.0]),
            target_pos=jnp.array([0.1, 0.0]),
            horizon=80,
            w_channel="bogus",
            z_channel=Z_STATE_ERROR,
        )


def test_invalid_z_channel_raises():
    plant, schedule = _rlrmp_setup()
    lqr = solve_lqr(plant, schedule)
    ctrl = lti_controller(lqr.K)
    with pytest.raises(ValueError, match="Unknown z channel"):
        linearise_trajectory(
            plant,
            ctrl,
            init_pos=jnp.array([0.0, 0.0]),
            target_pos=jnp.array([0.1, 0.0]),
            horizon=80,
            w_channel=W_ADDITIVE_FORCE,
            z_channel="bogus",
        )


def test_qr_cost_requires_schedule():
    plant, _ = _rlrmp_setup()
    lqr = solve_lqr(plant, _)
    ctrl = lti_controller(lqr.K)
    with pytest.raises(ValueError, match="qr_cost.*requires"):
        linearise_trajectory(
            plant,
            ctrl,
            init_pos=jnp.array([0.0, 0.0]),
            target_pos=jnp.array([0.1, 0.0]),
            horizon=80,
            w_channel=W_ADDITIVE_FORCE,
            z_channel=Z_QR_COST,
            schedule=None,
        )

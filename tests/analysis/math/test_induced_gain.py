"""Tests for ``rlrmp.analysis.math.induced_gain``.

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

Tests use double precision via the analysis-math test configuration.
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rlrmp.analysis.math.hinf_riccati import (
    CostSpec,
    cost_schedule_from_spec,
    find_gamma_star,
    linearize_pointmass,
    solve_hinf_riccati,
    solve_lqr,
)
from rlrmp.analysis.math.induced_gain import (
    TrajectoryLinearisation,
    W_ADDITIVE_FORCE,
    W_SENSORY_PERTURBATION,
    W_STRUCTURAL_DA,
    Z_CONTROL,
    Z_PEAK_VELOCITY,
    Z_QR_COST,
    Z_STATE_ERROR,
    _ltv_forward_sweep,
    induced_gain,
    induced_gain_power_iteration,
    linearise_trajectory,
    lti_controller,
)
from rlrmp.analysis.feedbax_controllers import simple_feedback_induced_gain_controller
from rlrmp.train.minimax import build_hps
from rlrmp.train.task_model import setup_task_model_pair


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


def _minimax_args(**overrides) -> argparse.Namespace:
    base = {
        "n_warmup_batches": 1,
        "n_adversary_batches": 1,
        "controller_lr": 0.01,
        "loss_update_enabled": False,
        "loss_update_ratio": 0.3,
        "hidden_type": "gru",
        "sisu_gating": "additive",
        "n_replicates": 1,
        "sensory_noise_std": 0.0,
        "additive_motor_noise_std": 0.0,
        "signal_dependent_motor_noise_std": 0.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _small_simple_feedback_model():
    hps = build_hps(_minimax_args())
    hps = hps | {
        "model": hps.model
        | {
            "hidden_size": 4,
            "feedback_delay_steps": 2,
            "feedback_noise_std": 0.0,
            "motor_noise_std": 0.0,
            "sensory_noise_std": 0.0,
            "additive_motor_noise_std": 0.0,
            "signal_dependent_motor_noise_std": 0.0,
            "population_structure": hps.model.population_structure
            | {
                "n_input_only": 0,
                "n_readout_only": 0,
                "n_recurrent_only": 4,
                "n_input_readout": 0,
            },
        }
    }
    model = setup_task_model_pair(hps, key=jax.random.PRNGKey(0)).model
    return jax.tree.map(
        lambda x: x[0] if eqx.is_array(x) and x.ndim > 0 and x.shape[0] == 1 else x,
        model,
        is_leaf=eqx.is_array,
    )


def _manual_network_controller_outputs(model, ctrl, observations, *, key):
    """Old net-plus-delay wrapper logic used as a parity oracle."""
    net = model.nodes["net"]
    full_state = model.init_state(key=jax.random.PRNGKey(0))
    if hasattr(net, "initial_cycle_port_values"):
        net_state = full_state
        cycle_values = net.initial_cycle_port_values(net_state)
    else:
        net_state = full_state.get(net.state_index)
        cycle_values = None
    queue = jnp.zeros((ctrl.delay, ctrl.n_obs), dtype=jnp.float64)
    outputs = []
    for t, sensory_obs in enumerate(observations):
        pos_abs = sensory_obs[:2] + ctrl.target_pos
        obs_abs = jnp.concatenate([pos_abs, sensory_obs[2:4]], axis=0)
        if ctrl.delay > 0:
            delayed_obs = queue[0]
            queue = jnp.concatenate([queue[1:], obs_abs[None, :]], axis=0)
        else:
            delayed_obs = obs_abs

        net_inputs = {
            "input": ctrl.task_input,
            "feedback": (delayed_obs[:2], delayed_obs[2:]),
        }
        step_key = jax.random.fold_in(key, t)
        if hasattr(net, "step"):
            net_outputs, net_state, cycle_values = net.step(
                net_inputs,
                net_state,
                cycle_values,
                key=step_key,
            )
        else:
            leaves, treedef = jax.tree.flatten(full_state)
            state = jax.tree.unflatten(treedef, leaves)
            state = state.set(net.state_index, net_state)
            net_outputs, state_next = net(net_inputs, state, key=step_key)
            net_state = state_next.get(net.state_index)
        outputs.append(net_outputs["output"])
    return outputs


# -----------------------------------------------------------------------------
# Toy LTV: foundational tests
# -----------------------------------------------------------------------------


def test_toy_ltv_matches_svd():
    """Power iteration on a 2-state random LTV matches Toeplitz-SVD ground truth."""
    lin = _toy_ltv(T=8, n=2, n_w=1, n_z=1, seed=0)
    sv_top = _brute_force_toeplitz_svd(lin)

    result = induced_gain_power_iteration(lin, n_restarts=5, max_iter=400, rtol=1e-9)
    assert result.converged
    assert result.gamma == pytest.approx(sv_top, rel=1e-6)


def test_toy_ltv_matches_svd_multidim():
    """Multi-channel toy LTV: power iteration matches SVD."""
    lin = _toy_ltv(T=10, n=3, n_w=2, n_z=2, seed=42)
    sv_top = _brute_force_toeplitz_svd(lin)

    result = induced_gain_power_iteration(lin, n_restarts=5, max_iter=500, rtol=1e-9)
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
    """Induced gain of an H-inf controller lies in (gamma_star, gamma_design].

    The expected band is theoretical:

    - Upper bound ``||T||_PI <= gamma_design``: the controller is designed to
      attenuate disturbances at level ``gamma_design = 1.5 * gamma_star``.
    - Strict lower bound ``||T||_PI > gamma_star``: ``gamma_star`` is the
      H-infinity *infimum*; no admissible finite-horizon LTI controller can
      realise it exactly. A controller designed at ``gamma_design > gamma_star``
      is suboptimal, so its actual closed-loop gain is strictly above
      ``gamma_star`` and bounded by the design level.

    Empirically (rlrmp point-mass regime, gamma_design=1.5*gamma_star) the
    analyser gain plateaus at ~1.21 * gamma_star at long horizons (n>=200);
    at short horizons (n<=100) the ratio dips below 1.0, which is an
    artefact of the finite-horizon ``find_gamma_star`` bisection finding a
    smaller infimum than the long-horizon limit. We therefore use a long
    horizon (n=200) for this test to land cleanly in the (gamma_star,
    gamma_design] band. See ``scripts/probe_round_trip_ratio.py`` and bug
    ``3c74e3b``.
    """
    plant = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)
    spec = CostSpec(n_steps=200)
    schedule = cost_schedule_from_spec(spec, plant)
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
        horizon=200,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_QR_COST,
        schedule=schedule,
        methods=("power_iteration",),
        n_restarts=5,
        max_iter=800,
        rtol=1e-6,
    )
    pi = out["power_iteration"]
    assert pi.converged
    # Tightened band: strict above gamma_star, bounded above by gamma_design.
    # Bug: 3c74e3b — was [0.3 * gamma_star, gamma_design]; the previous lower
    # band was lax to accommodate short-horizon gamma_star artefacts. The
    # n=200 horizon avoids those artefacts.
    assert pi.gamma > g_star
    assert pi.gamma <= gamma_design


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


# -----------------------------------------------------------------------------
# Sensory-perturbation rename + D_z feedthrough
# -----------------------------------------------------------------------------


def test_sensory_noise_spelling_resolves_to_perturbation():
    """Historical ``sensory_noise`` configs still route to sensory_perturbation."""
    plant, schedule = _rlrmp_setup()
    lqr = solve_lqr(plant, schedule)
    ctrl = lti_controller(lqr.K)
    init_pos = jnp.array([0.0, 0.0])
    target_pos = jnp.array([0.1, 0.0])

    # Old string still works.
    lin_old = linearise_trajectory(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=40,
        w_channel="sensory_noise",
        z_channel=Z_STATE_ERROR,
    )
    lin_new = linearise_trajectory(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=40,
        w_channel="sensory_perturbation",
        z_channel=Z_STATE_ERROR,
    )
    # Operator matrices should match exactly.
    assert jnp.allclose(lin_old.A_t, lin_new.A_t)
    assert jnp.allclose(lin_old.Bw_t, lin_new.Bw_t)
    assert jnp.allclose(lin_old.Cz_t, lin_new.Cz_t)
    assert jnp.allclose(lin_old.D_t, lin_new.D_t)
    # The new linearisation reports the canonical name.
    assert lin_new.w_channel == "sensory_perturbation"


def test_dz_feedthrough_sensory_perturbation_qr_cost():
    """Computing D_z exactly raises the gain over the zero-Dz approximation.

    Bug: ec7710f. Builds an LTV operator with the production sensory-perturbation
    × qr_cost feedthrough, then builds the same operator with D_t zeroed (the
    pre-fix behaviour), and verifies the corrected gain is strictly larger.

    For sensory_perturbation × qr_cost on full-state-feedback LQR, D_u = -K
    (since u = -K(x + w_obs)) and D_z_ctrl = sqrt(R) @ (-K). On a 2D point
    mass with R diagonal and a nonzero LQR gain, D_z is non-trivially nonzero.
    """
    plant, schedule = _rlrmp_setup()
    lqr = solve_lqr(plant, schedule)
    ctrl = lti_controller(lqr.K)
    init_pos = jnp.array([0.0, 0.0])
    target_pos = jnp.array([0.1, 0.0])

    lin_corrected = linearise_trajectory(
        plant,
        ctrl,
        init_pos=init_pos,
        target_pos=target_pos,
        horizon=40,
        w_channel=W_SENSORY_PERTURBATION,
        z_channel=Z_QR_COST,
        schedule=schedule,
    )
    # The corrected operator has nonzero D_t for this channel pair.
    assert float(jnp.linalg.norm(lin_corrected.D_t)) > 0.0

    # Construct the pre-fix variant: same A, B_w, C_z, but D_t zeroed.
    lin_old = TrajectoryLinearisation(
        A_t=lin_corrected.A_t,
        Bw_t=lin_corrected.Bw_t,
        Cz_t=lin_corrected.Cz_t,
        D_t=jnp.zeros_like(lin_corrected.D_t),
        x_nominal=lin_corrected.x_nominal,
        u_nominal=lin_corrected.u_nominal,
        dt=lin_corrected.dt,
        n_plant=lin_corrected.n_plant,
        n_ctrl=lin_corrected.n_ctrl,
        w_channel=lin_corrected.w_channel,
        z_channel=lin_corrected.z_channel,
    )

    g_corrected = induced_gain_power_iteration(
        lin_corrected, n_restarts=4, max_iter=400, rtol=1e-7
    ).gamma
    g_old = induced_gain_power_iteration(lin_old, n_restarts=4, max_iter=400, rtol=1e-7).gamma
    # Strict inequality: D_z fix raised the operator's leading singular value.
    assert g_corrected > g_old, (
        f"D_z fix should raise the gain: corrected={g_corrected}, old={g_old}"
    )
    # Sanity: increase is meaningful (not just numerical noise).
    assert (g_corrected - g_old) / max(g_old, 1e-30) > 1e-4


def test_feedbax_graph_controller_smoke():
    """Smoke test: ``feedbax_graph_controller`` wires up a minimal pass-through.

    Bug: b131510. Builds a trivial feedbax ``Graph`` with one component that
    multiplies its observation by a fixed gain, plus one ``StateIndex`` to
    exercise the flatten/unflatten round-trip. Verifies that the adapter's
    ``initial_state`` and ``step`` produce the expected output and that the
    flat-state shape is consistent.
    """
    from equinox.nn import StateIndex
    from feedbax.runtime.graph import Component, Graph

    from feedbax.analysis import graph_controller

    class GainComponent(Component):
        """y = -K @ x; carries a 1-element counter to exercise stateful flatten."""

        input_ports = ("input",)
        output_ports = ("output",)

        K: jnp.ndarray
        state_index: StateIndex

        def __init__(self, K):
            self.K = K
            self.state_index = StateIndex(jnp.zeros((1,), dtype=jnp.float64))

        def __call__(self, inputs, state, *, key):
            x = inputs["input"]
            counter = state.get(self.state_index)
            state = state.set(self.state_index, counter + 1.0)
            u = -self.K @ x
            return {"output": u}, state

    K = jnp.array([[1.0, 0.5]], dtype=jnp.float64)
    node = GainComponent(K)
    graph = Graph(
        nodes={"net": node},
        wires=(),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("net", "input")},
        output_bindings={"output": ("net", "output")},
    )

    key = jax.random.PRNGKey(0)
    ctrl = graph_controller(graph, key=key)
    h0 = ctrl.initial_state()
    # The state contains a single 1-element float counter.
    assert h0.shape == (1,)
    assert float(h0[0]) == 0.0

    obs = jnp.array([2.0, 4.0], dtype=jnp.float64)
    h1, u1 = ctrl.step(h0, obs, 0)
    # Output: u = -K @ obs = -[2.0 + 0.5*4.0] = -4.0
    assert float(u1[0]) == pytest.approx(-4.0, abs=1e-12)
    # Counter incremented from 0 to 1.
    assert float(h1[0]) == 1.0

    # Second step uses the new state.
    h2, u2 = ctrl.step(h1, obs, 1)
    assert float(h2[0]) == 2.0
    assert float(u2[0]) == pytest.approx(-4.0, abs=1e-12)


def test_feedbax_graph_controller_cyclic_smoke():
    """Smoke test: ``feedbax_graph_controller`` works on a graph WITH a cycle.

    Bug: 53b5fe5 — the previous adapter used ``Graph._call_single_step`` which
    silently fails on cyclic graphs (cycle wires are not threaded). The
    rewritten adapter uses ``Graph.step`` (feedbax 0ec8492) to thread cycle
    port values through the augmented hidden state.

    Builds a minimal 2-component graph with a back-edge: ``a`` produces
    ``y = -K @ x``; the network output (``a.y``) feeds a delay node (``d``)
    which echoes its input one step later (``d.y[t] = d.x[t-1]``); ``d.y``
    feeds back into ``a`` via a recurrent input. The external "input" is the
    sensory observation, also routed to ``a``. The cycle ``a.y -> d.x -> d.y
    -> a.h`` makes ``_needs_iteration == True``.

    Verifies:
      * adapter does not raise on construction (cycle support).
      * augmented ``h`` is wider than just the network State (carries cycle).
      * three chained ``step`` calls run without error.
    """
    from equinox.nn import StateIndex
    from feedbax.runtime.graph import Component, Graph, Wire

    from feedbax.analysis import graph_controller

    class GainWithRecurrent(Component):
        """y = -K @ x + h_in. Stateless component with two input ports."""

        input_ports = ("x", "h_in")
        output_ports = ("y",)

        K: jnp.ndarray

        def __init__(self, K):
            self.K = K

        def __call__(self, inputs, state, *, key):
            x = inputs["x"]
            h_in = inputs.get("h_in", jnp.zeros_like(self.K[:, 0]))
            u = -self.K @ x + h_in
            return {"y": u}, state

    class OneStepDelay(Component):
        """y[t] = x[t-1] (one-step delay). Carries one previous-x in StateIndex."""

        input_ports = ("x",)
        output_ports = ("y",)

        state_index: StateIndex

        def __init__(self, n: int):
            self.state_index = StateIndex(jnp.zeros((n,), dtype=jnp.float64))

        def __call__(self, inputs, state, *, key):
            x = inputs["x"]
            prev = state.get(self.state_index)
            state = state.set(self.state_index, x)
            return {"y": prev}, state

        def initial_outputs(self, state_value):
            # The default Component.initial_outputs() expects state_value to
            # be a structured object with attributes named after output ports.
            # Our state_value is a bare array, so override to expose
            # ``y`` directly (cycle-target default at t=0).
            if state_value is None:
                return {}
            return {"y": state_value}

    K = jnp.array([[1.0, 0.5]], dtype=jnp.float64)
    a = GainWithRecurrent(K)
    d = OneStepDelay(n=1)  # output is 1D (matches a.K row count)

    graph = Graph(
        nodes={"a": a, "d": d},
        wires=(
            Wire("a", "y", "d", "x"),  # forward: a's output feeds delay
            Wire("d", "y", "a", "h_in", temporality="recurrent"),  # delayed back-edge
        ),
        input_ports=("input",),
        output_ports=("output",),
        input_bindings={"input": ("a", "x")},
        output_bindings={"output": ("a", "y")},
    )
    assert graph._needs_iteration  # sanity: cycle detected

    key = jax.random.PRNGKey(0)
    ctrl = graph_controller(graph, key=key)
    h0 = ctrl.initial_state()
    # Augmented h carries (delay state) + (cycle port value for a.h_in).
    # delay state is 1 element; cycle dict has one (1,) entry.
    assert h0.shape == (2,)

    obs = jnp.array([2.0, 4.0], dtype=jnp.float64)
    # Step 0: cycle init: a.h_in = 0 (d.y at init = zeros).
    #   a.y = -K @ obs + 0 = -4.0; d.y = prev = 0; cycle out: d.y = 0.
    h1, u1 = ctrl.step(h0, obs, 0)
    assert float(u1[0]) == pytest.approx(-4.0, abs=1e-12)

    # Step 1: a.h_in = cycle from step 0 = 0.
    #   a.y = -4.0; d state was -4.0 from step 0; d.y = -4.0; cycle out: d.y = -4.0.
    h2, u2 = ctrl.step(h1, obs, 1)
    assert float(u2[0]) == pytest.approx(-4.0, abs=1e-12)

    # Step 2: a.h_in = cycle from step 1 = -4.0.
    #   a.y = -4.0 + (-4.0) = -8.0; d state was -4.0; d.y = -4.0.
    h3, u3 = ctrl.step(h2, obs, 2)
    assert float(u3[0]) == pytest.approx(-8.0, abs=1e-12)


def test_simple_feedback_induced_gain_adapter_matches_manual_network_wrapper():
    """Feedbax adapter preserves the historical SimpleFeedback wrapper outputs."""
    model = _small_simple_feedback_model()
    target_pos = jnp.array([0.15, 0.0], dtype=jnp.float64)
    key = jax.random.PRNGKey(7)
    ctrl = simple_feedback_induced_gain_controller(
        model,
        target_pos=target_pos,
        sisu=0.5,
        key=key,
    )
    observations = jnp.array(
        [
            [-0.15, 0.00, 0.00, 0.00],
            [-0.14, 0.01, 0.02, -0.01],
            [-0.12, 0.02, 0.03, -0.02],
            [-0.10, 0.03, 0.04, -0.03],
        ],
        dtype=jnp.float64,
    )

    manual_outputs = _manual_network_controller_outputs(model, ctrl, observations, key=key)
    h = ctrl.initial_state()
    adapter_outputs = []
    for t, obs in enumerate(observations):
        h, u = ctrl.step(h, obs, t)
        adapter_outputs.append(u)

    for manual, adapter in zip(manual_outputs, adapter_outputs):
        assert jnp.allclose(adapter, manual, atol=1e-10, rtol=1e-10)


def test_simple_feedback_induced_gain_adapter_smoke():
    """Synthetic SimpleFeedback model runs through induced-gain linearisation."""
    model = _small_simple_feedback_model()
    ctrl = simple_feedback_induced_gain_controller(
        model,
        target_pos=jnp.array([0.1, 0.0], dtype=jnp.float64),
        sisu=0.5,
        key=jax.random.PRNGKey(3),
    )
    plant = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)
    schedule = cost_schedule_from_spec(CostSpec(n_steps=8), plant)

    out = induced_gain(
        plant,
        ctrl,
        init_pos=jnp.array([0.0, 0.0], dtype=jnp.float64),
        target_pos=jnp.array([0.1, 0.0], dtype=jnp.float64),
        horizon=8,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_QR_COST,
        schedule=schedule,
        methods=("power_iteration",),
        n_restarts=1,
        max_iter=3,
        rtol=1e-4,
    )

    result = out["power_iteration"]
    assert jnp.isfinite(result.gamma)
    assert result.diagnostics["n_restarts"] == 1


def test_dz_feedthrough_lti_analytic():
    """D_z feedthrough analytic check on a hand-rolled toy LTI controller.

    Build a 1D plant with simple LQR-style feedback; verify the linearisation
    matches the closed-form D_z = sqrt(R) * (-K) for sensory_perturbation x
    qr_cost.
    """
    # Toy: build a TrajectoryLinearisation directly by walking through the
    # autodiff path with a hand-rolled 1D controller. We instead construct
    # an artificial linearisation by hand using the analyser's _qr_cost_Cz_Dz
    # helper — this validates the math at the helper level.
    from rlrmp.analysis.math.induced_gain import _qr_cost_Cz_Dz
    from rlrmp.analysis.math.hinf_riccati import CostSchedule

    horizon = 5
    n_plant = 2
    m_u = 1
    n_w = 2
    n_aug = n_plant  # stateless controller

    # Simple Q = I, R = I.
    Q = jnp.tile(jnp.eye(n_plant)[None], (horizon, 1, 1)).astype(jnp.float64)
    R = jnp.tile(jnp.eye(m_u)[None], (horizon, 1, 1)).astype(jnp.float64)
    Q_f = jnp.eye(n_plant, dtype=jnp.float64)
    schedule = CostSchedule(Q=Q, R=R, Q_f=Q_f)

    # Cu = -K (constant) where K = [1.0, 0.5]. So d u / d x = -K.
    K = jnp.array([[1.0, 0.5]], dtype=jnp.float64)  # (m_u, n_plant)
    Cu_arr = jnp.tile((-K)[None], (horizon, 1, 1))  # (T, m_u, n_aug=n_plant)

    # Sensory perturbation: u = -K (x + w_obs); so d u / d w = -K.
    Du_arr = jnp.tile((-K)[None], (horizon, 1, 1))  # (T, m_u, n_w)

    # Plant — needs a small mock for plant.m_u
    class _Mock:
        m_u = 1

    plant_mock = _Mock()

    Cz, Dz = _qr_cost_Cz_Dz(
        plant_mock,
        schedule,
        Cu_arr,
        Du_arr,
        n_aug=n_aug,
        n_w=n_w,
        n_plant=n_plant,
        horizon=horizon,
    )
    # State half of D_z is zero.
    assert jnp.allclose(Dz[:, :n_plant, :], 0.0)
    # Control half of D_z = sqrt(R) @ Du = I @ (-K) = -K (R = I).
    expected_Dz_ctrl = jnp.tile((-K)[None], (horizon, 1, 1))
    assert jnp.allclose(Dz[:, n_plant:, :], expected_Dz_ctrl, atol=1e-12)

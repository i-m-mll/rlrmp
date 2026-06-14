"""Diagnose round-trip-ratio probe anomalies — focused, fast version.

Bug: 74bfd86 -- analyser-side investigation of probe anomalies.

Three focused checks (all at n=200 unless noted):
  A. PI convergence vs rtol: does loosening rtol from 1e-9 to 1e-6 (matching
     the test) eliminate ``conv: NO``?
  B. Terminal-Q_f hypothesis: append sqrt(Q_f) @ x_T to the LTV operator
     and re-evaluate. If gamma_PI then approaches gamma_design (1.5 gamma*),
     the gap is an operator-vs-Riccati definition mismatch.
  C. Q_f-scale: gamma* with Q_f=0 vs Q_f=baseline vs Q_f=100x. Does gamma*
     itself depend strongly on Q_f? (If yes, Q_f-scale anomaly in the probe
     is a finite-horizon Riccati artifact, not an analyser artifact.)

Results (rlrmp regime, n=200, captured 2026-05-07):

  A. rtol sweep:
     rtol=1e-3: 136 iters, conv=yes, gamma_PI=0.01643
     rtol=1e-4: 354 iters, conv=yes, gamma_PI=0.01650
     rtol=1e-6: 890 iters, conv=yes, gamma_PI=0.01651
     rtol=1e-9: cap (4800), conv=NO  (ratio still ~1.20, accurate)
     -> ``conv: NO`` in the original probe was a tolerance artefact: 1e-9
        is unattainable inside max_iter=600 because the leading-singular-
        value gap is small. The estimate is accurate well before the
        consecutive-iter gate trips.

  B. terminal-Q_f at qf_scale=1.0:
     gamma_PI (no Qf)  = 0.016512, ratio = 1.2010
     gamma_PI (with Qf)= 0.016529, ratio = 1.2022
     -> The 1.21 plateau is NOT explained by the operator dropping Q_f.
        Adding the terminal block changes gamma_PI by < 0.1%. The plateau
        is just the suboptimality margin of an H-inf controller designed
        at 1.5 * gamma_star (sufficient bound, not tight).

  C. gamma_star vs qf_scale:
     qf=0       0.013721
     qf=0.01    0.013721
     qf=0.1    0.013724
     qf=1.0    0.013749   <- canonical
     qf=10     0.024640   <- 1.8x
     qf=100    0.060197   <- 4.4x
     -> gamma_star is essentially Q_f-independent for qf <= 1.0, but
        terminal-Q_f admissibility starts to dominate at qf >= 10. The
        ratio collapse at qf=100 in the original probe IS an operator-vs-
        synthesis Q_f-mismatch, but it only matters in this large-Q_f
        regime (which the canonical rlrmp setup does not visit).
"""

from __future__ import annotations

import sys
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rlrmp.analysis.math.hinf_riccati import (
    CostSchedule,
    CostSpec,
    cost_schedule_from_spec,
    find_gamma_star,
    linearize_pointmass,
    solve_hinf_riccati,
)
from rlrmp.analysis.math.induced_gain import (
    W_ADDITIVE_FORCE,
    Z_QR_COST,
    induced_gain_power_iteration,
    linearise_trajectory,
    lti_controller,
)

PLANT = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)
INIT_POS = jnp.array([0.0, 0.0], dtype=jnp.float64)
TARGET_POS = jnp.array([0.1, 0.0], dtype=jnp.float64)


def _make_schedule(n_steps, qf_scale=1.0, zero_qf=False):
    spec = CostSpec(n_steps=n_steps, go_step=min(30, n_steps // 4))
    sched = cost_schedule_from_spec(spec, PLANT)
    if zero_qf:
        sched = CostSchedule(Q=sched.Q, R=sched.R,
                             Q_f=jnp.zeros_like(sched.Q_f))
    elif qf_scale != 1.0:
        sched = CostSchedule(Q=sched.Q, R=sched.R, Q_f=qf_scale * sched.Q_f)
    return sched


def _build_lin(n, qf_scale=1.0, zero_qf=False):
    sched = _make_schedule(n, qf_scale=qf_scale, zero_qf=zero_qf)
    gstar = find_gamma_star(PLANT, sched, tol=1e-4)
    print(f"  gamma_star = {gstar:.6f}", flush=True)
    gdes = 1.5 * gstar
    ric = solve_hinf_riccati(PLANT, sched, gdes)
    assert ric.admissible
    ctrl = lti_controller(ric.K)
    lin = linearise_trajectory(
        PLANT, ctrl,
        init_pos=INIT_POS, target_pos=TARGET_POS, horizon=n,
        w_channel=W_ADDITIVE_FORCE, z_channel=Z_QR_COST, schedule=sched,
    )
    return sched, gstar, gdes, lin


# --- A. rtol sweep at n=200 -------------------------------------------------


def investigate_A():
    print("\n" + "=" * 70, flush=True)
    print("A. PI convergence vs rtol (n=200, qf_scale=1.0)", flush=True)
    print("=" * 70, flush=True)
    sched, gstar, gdes, lin = _build_lin(200)
    print(f"  gamma_design = {gdes:.6f}", flush=True)
    print(f"  {'rtol':>10s} {'gamma_PI':>10s} {'ratio':>8s} "
          f"{'converged':>10s} {'iters':>6s}", flush=True)
    for rtol in [1e-3, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9]:
        res = induced_gain_power_iteration(
            lin, n_restarts=8, max_iter=600, rtol=rtol,
            return_trajectory=False,
        )
        ratio = res.gamma / gstar
        print(f"  {rtol:>10.0e} {res.gamma:>10.5f} {ratio:>8.4f} "
              f"{'yes' if res.converged else 'NO':>10s} "
              f"{res.iterations:>6d}", flush=True)


# --- B. Terminal-Q_f hypothesis at n=200 ------------------------------------


def pi_with_terminal_Qf(lin, schedule, *, n_restarts=8, max_iter=600,
                       rtol=1e-7, seed=0):
    A_t, Bw_t, Cz_t, D_t = lin.A_t, lin.Bw_t, lin.Cz_t, lin.D_t
    T = lin.T
    n_w = lin.n_w
    n_aug = lin.n_aug
    n_plant = lin.n_plant
    n_ctrl = lin.n_ctrl

    Q_f = jnp.asarray(schedule.Q_f, dtype=jnp.float64)
    eigvals, eigvecs = jnp.linalg.eigh(0.5 * (Q_f + Q_f.T))
    eigvals = jnp.maximum(eigvals, 0.0)
    Qf_sqrt = eigvecs @ (jnp.sqrt(eigvals)[:, None] * eigvecs.T)
    Cz_term = jnp.concatenate(
        [Qf_sqrt, jnp.zeros((n_plant, n_ctrl), dtype=jnp.float64)],
        axis=1,
    )

    def forward(w_seq):
        x0 = jnp.zeros((n_aug,), dtype=jnp.float64)

        def body(x, inputs):
            A, Bw, Cz, D, w = inputs
            z = Cz @ x + D @ w
            x_next = A @ x + Bw @ w
            return x_next, z

        x_T, z_seq = jax.lax.scan(body, x0, (A_t, Bw_t, Cz_t, D_t, w_seq))
        z_T = Cz_term @ x_T
        return z_seq, z_T

    w_dummy = jnp.zeros((T, n_w), dtype=jnp.float64)
    _, vjp_fn = jax.vjp(forward, w_dummy)

    def adjoint_pair(z_seq, z_T):
        out = vjp_fn((z_seq, z_T))
        return out[0] if isinstance(out, (tuple, list)) else out

    rng = jax.random.PRNGKey(int(seed))
    best_gamma = 0.0
    converged_any = False

    for r in range(n_restarts):
        rng, sub = jax.random.split(rng)
        w = jax.random.normal(sub, (T, n_w), dtype=jnp.float64)
        w = w / (jnp.linalg.norm(w) + 1e-30)
        prev_gamma = 0.0
        consec = 0
        gamma_est = 0.0
        converged = False
        for it in range(max_iter):
            z_seq, z_T = forward(w)
            z_n = float(jnp.sqrt(jnp.sum(z_seq * z_seq) + jnp.sum(z_T * z_T)))
            w_n = float(jnp.linalg.norm(w))
            if w_n < 1e-30:
                break
            gamma_est = z_n / w_n
            w_new = adjoint_pair(z_seq, z_T)
            w_new_n = float(jnp.linalg.norm(w_new))
            if w_new_n < 1e-30:
                break
            w = w_new / w_new_n
            if prev_gamma > 0:
                rel = abs(gamma_est - prev_gamma) / max(prev_gamma, 1e-30)
                if rel < rtol:
                    consec += 1
                else:
                    consec = 0
                if consec >= 2:
                    converged = True
                    break
            prev_gamma = gamma_est
        if gamma_est > best_gamma:
            best_gamma = gamma_est
        converged_any = converged_any or converged
    return best_gamma, converged_any


def investigate_B():
    print("\n" + "=" * 70, flush=True)
    print("B. Terminal-Q_f hypothesis (n=200)", flush=True)
    print("   Compare gamma_PI without Q_f vs with terminal sqrt(Q_f) x_T",
          flush=True)
    print("=" * 70, flush=True)
    sched, gstar, gdes, lin = _build_lin(200)
    res = induced_gain_power_iteration(
        lin, n_restarts=8, max_iter=600, rtol=1e-7, return_trajectory=False,
    )
    print(f"  gamma_design     = {gdes:.6f}", flush=True)
    print(f"  gamma_PI (no Qf) = {res.gamma:.6f}  ratio = "
          f"{res.gamma/gstar:.4f}", flush=True)
    g_with, conv = pi_with_terminal_Qf(
        lin, sched, n_restarts=8, max_iter=600, rtol=1e-7,
    )
    print(f"  gamma_PI (+ Qf)  = {g_with:.6f}  ratio = {g_with/gstar:.4f}  "
          f"conv={conv}", flush=True)
    print(f"  Note: gamma_design / gamma_star = 1.5 (exact by construction).",
          flush=True)


# --- C. Q_f-scale dependency of gamma* --------------------------------------


def investigate_C():
    print("\n" + "=" * 70, flush=True)
    print("C. Q_f-scale dependency of gamma* (n=200)", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'qf_scale':>10s} {'gamma_star':>12s}", flush=True)
    sched_zero = _make_schedule(200, zero_qf=True)
    g_zero = find_gamma_star(PLANT, sched_zero, tol=1e-4)
    print(f"  {'0 (zero)':>10s} {g_zero:>12.6f}", flush=True)
    for qf in [0.01, 0.1, 1.0, 10.0, 100.0]:
        sched = _make_schedule(200, qf_scale=qf)
        gstar = find_gamma_star(PLANT, sched, tol=1e-4)
        print(f"  {qf:>10.4f} {gstar:>12.6f}", flush=True)


def main():
    # A is fast (mainly converging); B is slow (PI on extended operator hits
    # max_iter cap because the gap shrinks with the extra terminal block);
    # C is fast.
    investigate_A()
    investigate_B()
    investigate_C()


if __name__ == "__main__":
    main()

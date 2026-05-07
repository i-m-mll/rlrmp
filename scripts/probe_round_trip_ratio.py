"""Probe: finite-horizon round-trip ratio ||T||_PI / gamma* for an H-inf Riccati controller.

Bug: 74bfd86 -- follow-up diagnostic on the induced-gain analyser.

Three sweeps:
  1. Horizon sweep: n_steps in [50, 100, 200, 400, 800], fixed Q_f scale = 1.0, tol = 1e-4.
  2. Q_f scale sweep: horizon = 200, Q_f scale in [0.01, 0.1, 1.0, 10.0, 100.0], tol = 1e-4.
  3. Bisection tolerance sweep: horizon = 200, Q_f scale = 1.0, tol in [1e-2, 1e-3, 1e-4, 1e-5].
  + LQR baseline: same horizon sweep with the LQR (gamma -> inf) controller.

Plant: rlrmp regime (mass=1.0, damping=10.0, tau=0.05, dt=0.01).
Cost schedule: CostSpec with n_steps=horizon (same shape as _rlrmp_setup in tests).
Controller design: gamma_design = 1.5 * gamma_star (H-inf), or gamma -> inf (LQR).
Analyser: induced_gain_power_iteration on qr_cost x additive_force.
"""

from __future__ import annotations

import sys
import jax
import jax.numpy as jnp

# x64 is already enabled by the modules on import; belt-and-suspenders here.
jax.config.update("jax_enable_x64", True)

from rlrmp.analysis.hinf_riccati import (
    CostSchedule,
    CostSpec,
    cost_schedule_from_spec,
    find_gamma_star,
    linearize_pointmass,
    solve_hinf_riccati,
    solve_lqr,
)
from rlrmp.analysis.induced_gain import (
    W_ADDITIVE_FORCE,
    Z_QR_COST,
    induced_gain_power_iteration,
    linearise_trajectory,
    lti_controller,
)

# Fixed plant (rlrmp regime, matching test_induced_gain._rlrmp_setup and
# test_hinf_riccati._rlrmp_plant).
PLANT = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)

# Canonical reach geometry (0 -> 0.1 m along x).
INIT_POS = jnp.array([0.0, 0.0], dtype=jnp.float64)
TARGET_POS = jnp.array([0.1, 0.0], dtype=jnp.float64)

# Power-iteration settings (tight to get a reliable estimate).
PI_KWARGS = dict(n_restarts=8, max_iter=600, rtol=1e-9)


def _make_schedule(n_steps: int, qf_scale: float = 1.0) -> CostSchedule:
    """Build a CostSpec-based schedule for ``n_steps`` stages.

    go_step is fixed at 30 (well within the shorter horizons).  For very short
    horizons (n_steps <= 50) the late period may not start, so Q is mostly
    zero; the terminal Q_f is what drives the Riccati.

    ``qf_scale`` multiplies Q_f (the terminal cost matrix) but leaves Q, R
    unchanged.
    """
    spec = CostSpec(n_steps=n_steps, go_step=min(30, n_steps // 4))
    sched = cost_schedule_from_spec(spec, PLANT)
    if qf_scale != 1.0:
        sched = CostSchedule(
            Q=sched.Q,
            R=sched.R,
            Q_f=qf_scale * sched.Q_f,
        )
    return sched


def _run_one(
    n_steps: int,
    qf_scale: float = 1.0,
    tol: float = 1e-4,
    gamma_factor: float = 1.5,
    use_lqr: bool = False,
) -> dict:
    """Run one (gamma*, analyser_gamma) measurement.

    Returns a dict with keys: n_steps, qf_scale, tol, gamma_star,
    gamma_design, analyser_gamma, ratio, converged.
    """
    sched = _make_schedule(n_steps, qf_scale=qf_scale)

    gamma_star = find_gamma_star(PLANT, sched, tol=tol)

    if use_lqr:
        riccati = solve_lqr(PLANT, sched)
        gamma_design = float("inf")
    else:
        gamma_design = gamma_factor * gamma_star
        riccati = solve_hinf_riccati(PLANT, sched, gamma_design)
        assert riccati.admissible, (
            f"Riccati inadmissible at gamma_design={gamma_design:.4f} "
            f"(gamma_star={gamma_star:.4f}, n_steps={n_steps}, qf_scale={qf_scale})"
        )

    ctrl = lti_controller(riccati.K)

    lin = linearise_trajectory(
        PLANT,
        ctrl,
        init_pos=INIT_POS,
        target_pos=TARGET_POS,
        horizon=n_steps,
        w_channel=W_ADDITIVE_FORCE,
        z_channel=Z_QR_COST,
        schedule=sched,
    )

    result = induced_gain_power_iteration(lin, **PI_KWARGS)

    ratio = result.gamma / gamma_star if (gamma_star > 0 and not use_lqr) else float("nan")

    return {
        "n_steps": n_steps,
        "qf_scale": qf_scale,
        "tol": tol,
        "gamma_star": gamma_star,
        "gamma_design": gamma_design,
        "analyser_gamma": result.gamma,
        "ratio": ratio,
        "converged": result.converged,
    }


def _fmt_row(*args, widths):
    parts = []
    for a, w in zip(args, widths):
        if isinstance(a, float):
            parts.append(f"{a:>{w}.5f}")
        elif isinstance(a, bool):
            parts.append(f"{'yes' if a else 'NO':>{w}}")
        elif isinstance(a, str):
            parts.append(f"{a:>{w}}")
        else:
            parts.append(f"{str(a):>{w}}")
    return "  ".join(parts)


def main():
    print("=" * 70)
    print("Round-trip ratio probe: ||T||_PI / gamma*")
    print(f"Plant: mass=1.0, damping=10.0, tau=0.05, dt=0.01")
    print(f"Controller: H-inf at gamma_design = 1.5 * gamma*")
    print(f"Analyser: power_iteration, qr_cost x additive_force")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Sweep 1: horizon
    # -----------------------------------------------------------------------
    horizons = [50, 100, 200, 400, 800]
    print("\n--- Sweep 1: Horizon (Q_f scale=1.0, tol=1e-4) ---")
    cols = ["n_steps", "gamma*", "gamma_PI", "ratio", "conv"]
    widths = [8, 10, 10, 8, 6]
    print(_fmt_row(*cols, widths=widths))
    print("-" * 50)
    horizon_results = []
    for n in horizons:
        r = _run_one(n_steps=n, qf_scale=1.0, tol=1e-4)
        horizon_results.append(r)
        print(_fmt_row(r["n_steps"], r["gamma_star"], r["analyser_gamma"],
                       r["ratio"], r["converged"], widths=widths))
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Sweep 2: Q_f scale
    # -----------------------------------------------------------------------
    qf_scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_steps_qf = 200
    print(f"\n--- Sweep 2: Q_f scale (horizon={n_steps_qf}, tol=1e-4) ---")
    cols2 = ["qf_scale", "gamma*", "gamma_PI", "ratio", "conv"]
    widths2 = [10, 10, 10, 8, 6]
    print(_fmt_row(*cols2, widths=widths2))
    print("-" * 55)
    qf_results = []
    for qf in qf_scales:
        r = _run_one(n_steps=n_steps_qf, qf_scale=qf, tol=1e-4)
        qf_results.append(r)
        print(_fmt_row(r["qf_scale"], r["gamma_star"], r["analyser_gamma"],
                       r["ratio"], r["converged"], widths=widths2))
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Sweep 3: bisection tolerance
    # -----------------------------------------------------------------------
    tols = [1e-2, 1e-3, 1e-4, 1e-5]
    n_steps_tol = 200
    print(f"\n--- Sweep 3: Bisection tol (horizon={n_steps_tol}, Q_f scale=1.0) ---")
    cols3 = ["tol", "gamma*", "gamma_PI", "ratio", "conv"]
    widths3 = [8, 10, 10, 8, 6]
    print(_fmt_row(*cols3, widths=widths3))
    print("-" * 50)
    tol_results = []
    for tol in tols:
        r = _run_one(n_steps=n_steps_tol, qf_scale=1.0, tol=tol)
        tol_results.append(r)
        print(_fmt_row(r["tol"], r["gamma_star"], r["analyser_gamma"],
                       r["ratio"], r["converged"], widths=widths3))
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # LQR baseline: horizon sweep
    # -----------------------------------------------------------------------
    print("\n--- LQR baseline: Horizon sweep (gamma -> inf, Q_f scale=1.0) ---")
    cols4 = ["n_steps", "gamma*", "gamma_PI_LQR", "conv"]
    widths4 = [8, 10, 14, 6]
    print(_fmt_row(*cols4, widths=widths4))
    print("-" * 45)
    for n in horizons:
        r = _run_one(n_steps=n, qf_scale=1.0, tol=1e-4, use_lqr=True)
        print(_fmt_row(r["n_steps"], r["gamma_star"], r["analyser_gamma"],
                       r["converged"], widths=widths4))
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"Horizon sweep: ratio at n=50: {horizon_results[0]['ratio']:.4f}  "
          f"n=100: {horizon_results[1]['ratio']:.4f}  "
          f"n=200: {horizon_results[2]['ratio']:.4f}  "
          f"n=400: {horizon_results[3]['ratio']:.4f}  "
          f"n=800: {horizon_results[4]['ratio']:.4f}")
    print(f"Q_f scale sweep: ratio at 0.01: {qf_results[0]['ratio']:.4f}  "
          f"0.1: {qf_results[1]['ratio']:.4f}  "
          f"1.0: {qf_results[2]['ratio']:.4f}  "
          f"10.0: {qf_results[3]['ratio']:.4f}  "
          f"100.0: {qf_results[4]['ratio']:.4f}")
    print(f"Tol sweep: ratio at 1e-2: {tol_results[0]['ratio']:.4f}  "
          f"1e-3: {tol_results[1]['ratio']:.4f}  "
          f"1e-4: {tol_results[2]['ratio']:.4f}  "
          f"1e-5: {tol_results[3]['ratio']:.4f}")

    # Verdict
    ratios_horizon = [r["ratio"] for r in horizon_results]
    delta_horizon = ratios_horizon[-1] - ratios_horizon[0]
    print(f"\nHorizon delta (n=800 minus n=50): {delta_horizon:+.4f}")
    ratios_qf = [r["ratio"] for r in qf_results]
    delta_qf = ratios_qf[-1] - ratios_qf[0]
    print(f"Q_f delta (scale=100 minus scale=0.01): {delta_qf:+.4f}")
    ratios_tol = [r["ratio"] for r in tol_results]
    delta_tol = ratios_tol[-1] - ratios_tol[0]
    print(f"Tol delta (tol=1e-5 minus tol=1e-2): {delta_tol:+.4f}")


if __name__ == "__main__":
    main()

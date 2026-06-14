"""Now that Bw=I_6 produces Δv>0 on C&S regime, sweep secondary variables to
quantify their contribution and see if magnitude approaches C&S's reported scale.

C&S quote: ~15% higher cost, "faster movement velocities toward target", Fig 1e
shows visually ~10-15% higher peak forward velocity. With gamma very close to gamma*,
inflation grows.

Variables to sweep:
1. alpha_1 (cost ramp coefficient) — C&S sweeps an order of magnitude
2. gamma_factor near boundary (1.01, 1.05, 1.1)
3. n_steps (reach duration: 60, 80, 100)
4. Geometry (15cm reach, target along y vs x — should be invariant for isotropic plant)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from rlrmp.analysis.math.hinf_riccati import (
    compute_velocity_inflation,
    cs_eq15_cost_schedule,
    find_gamma_star,
    linearize_pointmass,
)


def lift_bw(plant):
    n = plant.n
    return eqx.tree_at(
        lambda p: (p.Bw_c, p.Bw),
        plant,
        (jnp.eye(n, dtype=jnp.float64), jnp.eye(n, dtype=jnp.float64)),
    )


def main():
    init_pos = jnp.array([0.0, 0.0], dtype=jnp.float64)
    target_pos = jnp.array([0.15, 0.0], dtype=jnp.float64)

    plant = lift_bw(linearize_pointmass(mass=1.0, damping=0.1, tau=0.06, dt=0.01))

    print("=== C&S regime, Bw=I_6 ===")
    print("\nSweep alpha_1 (cost coefficient): factor=1.5")
    for a1 in [0.1, 0.3, 1.0, 3.0, 10.0]:
        schedule = cs_eq15_cost_schedule(n_steps=80, alpha_1=a1)
        gs = find_gamma_star(plant, schedule)
        res = compute_velocity_inflation(
            plant, schedule, init_pos=init_pos, target_pos=target_pos,
            gamma_factor=1.5, gamma_star=gs,
        )
        print(f"  alpha_1={a1:.1f}  gamma*={gs:.4f}  Δv={res.delta_v_percent:+.4f}%  LQR_fwd={res.lqr_peak_forward_velocity:.4f}  HINF_fwd={res.hinf_peak_forward_velocity:.4f}")

    print("\nSweep gamma_factor near boundary (alpha_1=1.0):")
    schedule = cs_eq15_cost_schedule(n_steps=80, alpha_1=1.0)
    gs = find_gamma_star(plant, schedule, tol=1e-6)
    print(f"  gamma_star (high-precision) = {gs:.6f}")
    for f in [1.001, 1.005, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0]:
        try:
            res = compute_velocity_inflation(
                plant, schedule, init_pos=init_pos, target_pos=target_pos,
                gamma_factor=f, gamma_star=gs,
            )
            print(f"  factor={f:.3f}  Δv={res.delta_v_percent:+.4f}%  HINF_fwd={res.hinf_peak_forward_velocity:.4f}")
        except Exception as e:
            print(f"  factor={f:.3f}  INADMISSIBLE: {e}")

    print("\nSweep n_steps:")
    for ns in [60, 80, 100, 120]:
        schedule = cs_eq15_cost_schedule(n_steps=ns, alpha_1=1.0)
        gs = find_gamma_star(plant, schedule)
        res = compute_velocity_inflation(
            plant, schedule, init_pos=init_pos, target_pos=target_pos,
            gamma_factor=1.5, gamma_star=gs,
        )
        print(f"  n_steps={ns}  gamma*={gs:.4f}  Δv={res.delta_v_percent:+.4f}%  LQR_fwd={res.lqr_peak_forward_velocity:.4f}  HINF_fwd={res.hinf_peak_forward_velocity:.4f}")

    # Also test on rlrmp regime
    print("\n=== rlrmp regime (k=10), Bw=I_6 ===")
    plant_rlrmp = lift_bw(linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01))
    schedule = cs_eq15_cost_schedule(n_steps=80, alpha_1=1.0)
    gs = find_gamma_star(plant_rlrmp, schedule)
    print(f"  gamma_star = {gs:.4f}")
    for f in [1.05, 1.2, 1.5, 2.0]:
        target_rlrmp = jnp.array([0.5, 0.0], dtype=jnp.float64)  # rlrmp uses 50cm reaches
        res = compute_velocity_inflation(
            plant_rlrmp, schedule, init_pos=init_pos, target_pos=target_rlrmp,
            gamma_factor=f, gamma_star=gs,
        )
        print(f"  factor={f:.2f}  Δv={res.delta_v_percent:+.4f}%  LQR_fwd={res.lqr_peak_forward_velocity:.4f}  HINF_fwd={res.hinf_peak_forward_velocity:.4f}")


if __name__ == "__main__":
    main()

"""Reproduce baseline numbers for the xfailed test_cs_faithful_qr_velocity_inflation.

Prints exact diagnostics: gamma*, gamma_eval, Δv (forward + lateral), peak velocities.
"""

from __future__ import annotations

import jax.numpy as jnp

from rlrmp.analysis.math.hinf_riccati import (
    compute_velocity_inflation,
    cs_eq15_cost_schedule,
    find_gamma_star,
    linearize_pointmass,
)


def main():
    plant = linearize_pointmass(mass=1.0, damping=0.1, tau=0.06, dt=0.01)
    schedule = cs_eq15_cost_schedule(n_steps=80, alpha_1=1.0)

    init_pos = jnp.array([0.0, 0.0], dtype=jnp.float64)
    target_pos = jnp.array([0.15, 0.0], dtype=jnp.float64)

    gamma_star = find_gamma_star(plant, schedule)
    print(f"gamma_star = {gamma_star:.6f}")

    for factor in [1.05, 1.2, 1.5, 2.0, 5.0]:
        res = compute_velocity_inflation(
            plant, schedule, init_pos=init_pos, target_pos=target_pos,
            gamma_factor=factor, gamma_star=gamma_star,
        )
        print(
            f"factor={factor:.2f}  gamma_eval={res.gamma_evaluated:.4f}  "
            f"Δv_fwd={res.delta_v_percent:+.4f}%  Δv_lat={res.delta_v_lateral_percent:+.4f}%  "
            f"LQR_fwd={res.lqr_peak_forward_velocity:.4f}  HINF_fwd={res.hinf_peak_forward_velocity:.4f}"
        )

    print(f"\nPlant Bw shape: {plant.Bw.shape}")
    print(f"Plant Bw_c (continuous):\n{plant.Bw_c}")
    print(f"Plant Bw (discrete):\n{plant.Bw}")
    print(f"\nB_w col norms (continuous): {jnp.linalg.norm(plant.Bw_c, axis=0)}")


if __name__ == "__main__":
    main()

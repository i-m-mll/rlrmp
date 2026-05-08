"""Test the leading hypothesis: C&S's H∞ Riccati uses B_w = I_6 (full-state additive ε),
not the 2D velocity-channel B_w that the rlrmp implementation uses (which matches the
*physical curl-field injection point*, not the H∞ design B_w).

Per C&S Eq 13: x_{t+1} = A_d x_t + B_d u_t + ε_t with ε_t ∈ R^6 (one per state coord).
Per C&S Eq 11: J(x,u,ε) = J(x,u) - γ²·ε^T·ε.

So in the H∞ Riccati ε is a free 6-vector, i.e. B_w_design = I_6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from rlrmp.analysis.hinf_riccati import (
    compute_velocity_inflation,
    cs_eq15_cost_schedule,
    find_gamma_star,
    linearize_pointmass,
    PlantLinearization,
    simulate_closed_loop,
    solve_lqr,
    solve_hinf_riccati,
    make_reach_initial_state,
)


def lift_bw_to_full_state(plant: PlantLinearization) -> PlantLinearization:
    """Replace plant's Bw / Bw_c with identity-style full-state additive disturbance.

    Per C&S Eq 13: ε_t enters every state coordinate (not just velocity). Bw_d = I_6.
    For the continuous-time Bw_c, use I as well (consistent with C&S's
    continuous-discrete equivalence after Euler/ZOH discretisation).
    """
    n = plant.n
    Bw_c_new = jnp.eye(n, dtype=jnp.float64)
    Bw_d_new = jnp.eye(n, dtype=jnp.float64)
    return eqx.tree_at(
        lambda p: (p.Bw_c, p.Bw),
        plant,
        (Bw_c_new, Bw_d_new),
    )


def main():
    plant_orig = linearize_pointmass(mass=1.0, damping=0.1, tau=0.06, dt=0.01)
    plant_full = lift_bw_to_full_state(plant_orig)
    schedule = cs_eq15_cost_schedule(n_steps=80, alpha_1=1.0)

    init_pos = jnp.array([0.0, 0.0], dtype=jnp.float64)
    target_pos = jnp.array([0.15, 0.0], dtype=jnp.float64)

    print("=== Original Bw (2D velocity-channel only) ===")
    gs_orig = find_gamma_star(plant_orig, schedule)
    print(f"gamma_star = {gs_orig:.4f}")
    for f in [1.05, 1.2, 1.5, 2.0]:
        res = compute_velocity_inflation(
            plant_orig, schedule, init_pos=init_pos, target_pos=target_pos,
            gamma_factor=f, gamma_star=gs_orig,
        )
        print(f"  factor={f:.2f}  Δv_fwd={res.delta_v_percent:+.4f}%  HINF_fwd={res.hinf_peak_forward_velocity:.4f}")

    print("\n=== Full-state Bw (I_6, matching C&S Eq 13 ε formulation) ===")
    gs_full = find_gamma_star(plant_full, schedule)
    print(f"gamma_star = {gs_full:.4f}")
    for f in [1.05, 1.2, 1.5, 2.0, 5.0, 10.0]:
        try:
            res = compute_velocity_inflation(
                plant_full, schedule, init_pos=init_pos, target_pos=target_pos,
                gamma_factor=f, gamma_star=gs_full,
            )
            print(f"  factor={f:.2f}  Δv_fwd={res.delta_v_percent:+.4f}%  HINF_fwd={res.hinf_peak_forward_velocity:.4f}  LQR_fwd={res.lqr_peak_forward_velocity:.4f}")
        except Exception as e:
            print(f"  factor={f:.2f}  FAILED: {e}")


if __name__ == "__main__":
    main()

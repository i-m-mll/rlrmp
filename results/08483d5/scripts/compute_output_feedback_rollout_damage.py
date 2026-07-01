from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    TARGET_POS,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    CSForwardNoiseDraws,
    CSStochasticRollout,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    zero_forward_noise_draws,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star, solve_hinf_riccati
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    make_cs_output_feedback_initial_state,
    output_feedback_cost,
    robust_estimator_covariances,
    robust_estimator_fixed_adversary_policy,
    robust_output_feedback_gains,
    rollout_with_robust_estimator,
    rollout_with_robust_estimator_policy,
)


jax.config.update("jax_enable_x64", True)

OUT_DIR = Path("/tmp/rlrmp_adaptive_bounds_20260630_205501")
JSON_PATH = OUT_DIR / "rollout_damage_estimate.json"
MD_PATH = OUT_DIR / "rollout_damage_estimate.md"
TEACHER_PACKAGE = Path("_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz")
TEACHER_MANIFEST = Path(
    "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers_manifest.json"
)
STOCHASTIC_SEED = 376023


def _float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _cost_from_arrays(schedule, x, u, epsilon, gamma: float | None) -> dict[str, float | None]:
    state_terms = jnp.einsum("ti,tij,tj->t", x[:-1], schedule.Q, x[:-1])
    control_terms = jnp.einsum("ti,tij,tj->t", u, schedule.R, u)
    terminal = x[-1] @ schedule.Q_f @ x[-1]
    state_stage = _float(jnp.sum(state_terms))
    control_stage = _float(jnp.sum(control_terms))
    terminal_state = _float(terminal)
    total = state_stage + control_stage + terminal_state
    disturbance_energy = _float(jnp.sum(epsilon**2))
    h_inf_objective = None
    if gamma is not None:
        h_inf_objective = total - float(gamma * gamma) * disturbance_energy
    return {
        "state_stage": state_stage,
        "control_stage": control_stage,
        "terminal_state": terminal_state,
        "total_without_disturbance_penalty": total,
        "disturbance_energy": disturbance_energy,
        "h_infinity_objective": h_inf_objective,
    }


def _rollout_summary(plant, x, u, epsilon, schedule, gamma: float | None) -> dict[str, Any]:
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos_abs = pos + TARGET_POS[None, :]
    terminal_error = jnp.linalg.norm(pos_abs[-1] - TARGET_POS)
    cost = _cost_from_arrays(schedule, x, u, epsilon, gamma)
    return {
        "cost": cost,
        "peak_forward_velocity_m_s": _float(jnp.max(forward)),
        "peak_forward_velocity_idx": int(jnp.argmax(forward)),
        "terminal_position_error_m": _float(terminal_error),
        "control_effort_sum_u2": _float(jnp.sum(u**2)),
        "disturbance_l2": float(np.sqrt(max(cost["disturbance_energy"], 0.0))),
    }


def _deterministic_result(plant, schedule, solution, x0, gains, policy, config) -> dict[str, Any]:
    clean = rollout_with_robust_estimator(
        plant,
        schedule,
        solution,
        x0,
        gains=gains,
        config=config,
    )
    adversarial = rollout_with_robust_estimator_policy(
        plant,
        schedule,
        solution,
        x0,
        policy,
        gains=gains,
        config=config,
    )
    clean_cost = output_feedback_cost(schedule, clean, gamma=solution.gamma)
    adversarial_cost = output_feedback_cost(schedule, adversarial, gamma=solution.gamma)
    clean_summary = _rollout_summary(plant, clean.x, clean.u, clean.epsilon, schedule, solution.gamma)
    adversarial_summary = _rollout_summary(
        plant,
        adversarial.x,
        adversarial.u,
        adversarial.epsilon,
        schedule,
        solution.gamma,
    )
    # Use the dataclass cost as the source of truth for deterministic repo rollouts.
    clean_summary["cost"] = clean_cost.__dict__
    adversarial_summary["cost"] = adversarial_cost.__dict__
    return {
        "mode": "deterministic_noise_off",
        "noise_available": False,
        "noise_paired": "not_applicable",
        "clean": clean_summary,
        "adversarial": adversarial_summary,
        "paired_damage": (
            adversarial_cost.total_without_disturbance_penalty
            - clean_cost.total_without_disturbance_penalty
        ),
        "paired_h_infinity_objective_delta": (
            adversarial_cost.h_infinity_objective - clean_cost.h_infinity_objective
        ),
        "max_abs_epsilon": _float(jnp.max(jnp.abs(adversarial.epsilon))),
        "epsilon_shape": list(adversarial.epsilon.shape),
    }


def _stochastic_policy_rollout(
    plant,
    schedule,
    solution,
    x0,
    draws: CSForwardNoiseDraws,
    covariances,
    gains,
    policy,
    *,
    adversarial: bool,
    config: OutputFeedbackConfig,
) -> CSStochasticRollout:
    T = int(gains.shape[0])
    H = delayed_observation_matrix(plant, config)
    estimator_covariances = robust_estimator_covariances(plant, schedule, solution.gamma, config)
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)

    x_seq = [x0.astype(jnp.float64)]
    xhat_seq = [x0.astype(jnp.float64)]
    y_clean_seq = []
    y_seq = []
    u_command_seq = []
    u_applied_seq = []
    motor_seq = []
    sdn_seq = []
    process_seq = []
    sensory_seq = []
    eps_seq = []
    zero_eps = jnp.zeros((plant.m_w,), dtype=jnp.float64)

    for t in range(T):
        x_t = x_seq[-1]
        xhat_t = xhat_seq[-1]
        Sigma = estimator_covariances[t]
        precision = jnp.linalg.inv(Sigma) + H.T @ H - inv_gamma2 * schedule.Q[t]
        middle = jnp.linalg.inv(precision)
        y_clean = H @ x_t
        sensory = draws.sensory[t]
        y_t = y_clean + sensory
        u_command = -gains[t] @ xhat_t
        eps_t = policy[t] @ jnp.concatenate([x_t, xhat_t], axis=0) if adversarial else zero_eps
        motor = draws.motor[t]
        signal_dependent = jnp.einsum(
            "j,nmj,m->n",
            draws.signal_dependent_standard[t],
            covariances.signal_dependent_state,
            u_command,
        )
        process = draws.process[t]
        innovation = y_t - H @ xhat_t
        correction = inv_gamma2 * schedule.Q[t] @ xhat_t + H.T @ innovation
        xhat_next = plant.A @ xhat_t + plant.B @ u_command + plant.A @ middle @ correction
        x_next = (
            plant.A @ x_t
            + plant.B @ u_command
            + plant.Bw @ eps_t
            + motor
            + signal_dependent
            + process
        )
        y_clean_seq.append(y_clean)
        y_seq.append(y_t)
        u_command_seq.append(u_command)
        u_applied_seq.append(u_command)
        motor_seq.append(motor)
        sdn_seq.append(signal_dependent)
        process_seq.append(process)
        sensory_seq.append(sensory)
        eps_seq.append(eps_t)
        x_seq.append(x_next)
        xhat_seq.append(xhat_next)

    x = jnp.stack(x_seq, axis=0)
    u_applied = jnp.stack(u_applied_seq, axis=0)
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    pos_abs = pos + TARGET_POS[None, :]
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
        adversary_epsilon=jnp.stack(eps_seq, axis=0),
        perturbations=jnp.zeros((T, plant.n), dtype=jnp.float64),
        peak_forward_velocity=_float(jnp.max(forward)),
        peak_forward_velocity_idx=int(jnp.argmax(forward)),
        terminal_position_error=_float(jnp.linalg.norm(pos_abs[-1] - TARGET_POS)),
        control_effort=_float(jnp.sum(u_applied**2)),
    )


def _stochastic_result(plant, schedule, solution, x0, gains, policy, config) -> dict[str, Any]:
    covariances = default_cs_noise_covariances(plant, config)
    draws = sample_forward_noise_draws(
        jr.PRNGKey(STOCHASTIC_SEED),
        T=schedule.T,
        covariances=covariances,
    )
    clean = _stochastic_policy_rollout(
        plant,
        schedule,
        solution,
        x0,
        draws,
        covariances,
        gains,
        policy,
        adversarial=False,
        config=config,
    )
    adversarial = _stochastic_policy_rollout(
        plant,
        schedule,
        solution,
        x0,
        draws,
        covariances,
        gains,
        policy,
        adversarial=True,
        config=config,
    )
    clean_summary = _rollout_summary(
        plant,
        clean.x,
        clean.u_applied,
        clean.adversary_epsilon,
        schedule,
        solution.gamma,
    )
    adversarial_summary = _rollout_summary(
        plant,
        adversarial.x,
        adversarial.u_applied,
        adversarial.adversary_epsilon,
        schedule,
        solution.gamma,
    )
    noise_energy = {
        "sensory": _float(jnp.sum(draws.sensory**2)),
        "motor_state": _float(jnp.sum(draws.motor**2)),
        "process_state": _float(jnp.sum(draws.process**2)),
        "signal_dependent_standard": _float(jnp.sum(draws.signal_dependent_standard**2)),
        "clean_signal_dependent_state": _float(jnp.sum(clean.signal_dependent_noise**2)),
        "adversarial_signal_dependent_state": _float(
            jnp.sum(adversarial.signal_dependent_noise**2)
        ),
    }
    return {
        "mode": "nominal_noise_paired",
        "noise_available": True,
        "noise_paired": True,
        "noise_seed": STOCHASTIC_SEED,
        "noise_covariance_contract": {
            "sensory_shape": list(covariances.sensory.shape),
            "motor_shape": list(covariances.motor.shape),
            "process_shape": list(covariances.process.shape),
            "signal_dependent_state_shape": list(covariances.signal_dependent_state.shape),
        },
        "noise_energy": noise_energy,
        "clean": clean_summary,
        "adversarial": adversarial_summary,
        "paired_damage": (
            adversarial_summary["cost"]["total_without_disturbance_penalty"]
            - clean_summary["cost"]["total_without_disturbance_penalty"]
        ),
        "paired_h_infinity_objective_delta": (
            adversarial_summary["cost"]["h_infinity_objective"]
            - clean_summary["cost"]["h_infinity_objective"]
        ),
        "max_abs_epsilon": _float(jnp.max(jnp.abs(adversarial.adversary_epsilon))),
        "epsilon_shape": list(adversarial.adversary_epsilon.shape),
    }


def _verification_against_teacher_package(plant, schedule, solution, x0, gains, covs) -> dict[str, Any]:
    if not TEACHER_PACKAGE.exists():
        return {"status": "missing", "path": str(TEACHER_PACKAGE)}
    checks = {}
    with np.load(TEACHER_PACKAGE, allow_pickle=False) as data:
        expected = {
            "plant_A": np.asarray(plant.A),
            "plant_B": np.asarray(plant.B),
            "plant_Bw": np.asarray(plant.Bw),
            "schedule_Q": np.asarray(schedule.Q),
            "schedule_R": np.asarray(schedule.R),
            "schedule_Q_f": np.asarray(schedule.Q_f),
            "x0": np.asarray(x0),
            "hinf_controller_gains": np.asarray(gains),
            "hinf_estimator_covariances": np.asarray(covs),
            "hinf_P": np.asarray(solution.P),
            "hinf_gamma": np.asarray(solution.gamma),
            "gamma_star": np.asarray(find_gamma_star(plant, schedule)),
            "observation_matrix": np.asarray(delayed_observation_matrix(plant, OutputFeedbackConfig(n_phys=6))),
        }
        for key, value in expected.items():
            observed = np.asarray(data[key])
            checks[key] = {
                "max_abs_error": float(np.max(np.abs(observed - value))),
                "allclose": bool(np.allclose(observed, value, rtol=1e-9, atol=1e-9)),
            }
    return {
        "status": "checked",
        "path": str(TEACHER_PACKAGE),
        "manifest": str(TEACHER_MANIFEST),
        "checks": checks,
    }


def _markdown(payload: dict[str, Any]) -> str:
    det = payload["rollouts"]["deterministic_noise_off"]
    sto = payload["rollouts"]["nominal_noise_paired"]
    lines = [
        "# 6D output-feedback H-infinity rollout damage estimate",
        "",
        "## Source",
        "",
        f"- Teacher package: `{payload['source']['teacher_package']}`",
        f"- Teacher manifest: `{payload['source']['teacher_manifest']}`",
        "- Live functions: `build_no_integrator_game`, `find_gamma_star`, "
        "`solve_hinf_riccati`, `robust_estimator_covariances`, "
        "`robust_output_feedback_gains`, `robust_estimator_fixed_adversary_policy`, "
        "`rollout_with_robust_estimator`, `rollout_with_robust_estimator_policy`.",
        "",
        "## Contract",
        "",
        f"- State basis: {payload['contract']['state_basis']}",
        f"- Horizon: {payload['contract']['horizon_steps']} steps at "
        f"{payload['contract']['dt_s']} s.",
        f"- Target convention: {payload['contract']['target_convention']}",
        f"- Q/R/Qf: {payload['contract']['cost_contract']}",
        f"- Gamma ratio: {payload['contract']['gamma_factor']} "
        f"(gamma_star={payload['contract']['gamma_star']:.12g}, "
        f"gamma={payload['contract']['gamma']:.12g}).",
        f"- F application: {payload['contract']['adversary_application']}",
        "",
        "## Results",
        "",
        "| Mode | Clean cost | Adversarial cost | Paired damage | Disturbance energy | H-inf objective delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in (det, sto):
        clean = item["clean"]["cost"]
        adv = item["adversarial"]["cost"]
        lines.append(
            f"| {item['mode']} | "
            f"{clean['total_without_disturbance_penalty']:.12g} | "
            f"{adv['total_without_disturbance_penalty']:.12g} | "
            f"{item['paired_damage']:.12g} | "
            f"{adv['disturbance_energy']:.12g} | "
            f"{item['paired_h_infinity_objective_delta']:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Cost Components",
            "",
            "| Mode | Condition | State stage | Control stage | Terminal | Peak forward velocity | Terminal error |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in (det, sto):
        for condition in ("clean", "adversarial"):
            summary = item[condition]
            cost = summary["cost"]
            lines.append(
                f"| {item['mode']} | {condition} | "
                f"{cost['state_stage']:.12g} | "
                f"{cost['control_stage']:.12g} | "
                f"{cost['terminal_state']:.12g} | "
                f"{summary['peak_forward_velocity_m_s']:.12g} | "
                f"{summary['terminal_position_error_m']:.12g} |"
            )
    lines.extend(
        [
            "",
            "## Noise",
            "",
            "- Deterministic rollout: noise off.",
            f"- Stochastic rollout: nominal C&S released-code noise terms available and paired "
            f"with seed `{sto['noise_seed']}`; clean and adversarial conditions use the same "
            "sensory, motor, process, and signal-dependent standard draws.",
            "",
            "## Verification",
            "",
            f"- Teacher package verification status: `{payload['teacher_package_verification']['status']}`.",
        ]
    )
    failed = [
        key
        for key, item in payload["teacher_package_verification"].get("checks", {}).items()
        if not item["allclose"]
    ]
    if failed:
        lines.append(f"- Package mismatch keys: `{', '.join(failed)}`.")
    else:
        lines.append("- Recomputed live arrays match the stored teacher package at rtol/atol 1e-9.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = OutputFeedbackConfig(n_phys=6)
    plant, schedule = build_no_integrator_game()
    gamma_star = find_gamma_star(plant, schedule)
    gamma_factor = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    gamma = gamma_factor * gamma_star
    solution = solve_hinf_riccati(plant, schedule, gamma)
    if not solution.admissible:
        raise RuntimeError(f"gamma={gamma} is not admissible")
    x0 = make_cs_output_feedback_initial_state(plant, config)
    covs = robust_estimator_covariances(plant, schedule, solution.gamma, config)
    gains = robust_output_feedback_gains(plant, schedule, solution, covs, config)
    policy = robust_estimator_fixed_adversary_policy(
        plant,
        schedule,
        solution,
        gains,
        covs,
        config,
    )

    deterministic = _deterministic_result(plant, schedule, solution, x0, gains, policy, config)
    stochastic = _stochastic_result(plant, schedule, solution, x0, gains, policy, config)
    verification = _verification_against_teacher_package(plant, schedule, solution, x0, gains, covs)

    payload = {
        "schema_version": "rlrmp.local_analytical_damage.v1",
        "source": {
            "teacher_package": str(TEACHER_PACKAGE),
            "teacher_manifest": str(TEACHER_MANIFEST),
            "materializer": "results/376d023/scripts/materialize_6d_analytical_velocity_profiles.py",
            "controller_reference": "6D output-feedback H-infinity analytical teacher",
        },
        "contract": {
            "state_basis": (
                "36D delay-augmented state = 6 delayed physical blocks of "
                "[x, y, vx, vy, force_x, force_y]; no disturbance-integrator coordinates; "
                "observation is the oldest delayed 6D physical block."
            ),
            "horizon_steps": int(schedule.T),
            "dt_s": float(plant.dt),
            "target_convention": (
                "goal-centered C&S state with INIT_POS=[0,0], TARGET_POS=[0.15,0]; "
                "x0 repeats [-0.15, 0, 0, 0, 0, 0] across all six delay blocks."
            ),
            "cost_contract": (
                "C&S Eq. 15 no-integrator 6D schedule: Q shape "
                f"{tuple(schedule.Q.shape)}, R shape {tuple(schedule.R.shape)}, "
                f"Qf shape {tuple(schedule.Q_f.shape)}."
            ),
            "gamma_factor": float(gamma_factor),
            "gamma_star": float(gamma_star),
            "gamma": float(solution.gamma),
            "adversary_application": (
                "Matched optimal fixed policy F_t from "
                "robust_estimator_fixed_adversary_policy; applied each step as "
                "epsilon_t = F_t @ concat([x_t, xhat_t]) through plant.Bw. "
                "The clean condition uses the same controller/estimator with epsilon_t=0."
            ),
            "policy_shape": list(policy.shape),
            "controller_gain_shape": list(gains.shape),
            "estimator_covariance_shape": list(covs.shape),
            "plant_shapes": {
                "A": list(plant.A.shape),
                "B": list(plant.B.shape),
                "Bw": list(plant.Bw.shape),
                "observation_matrix": list(delayed_observation_matrix(plant, config).shape),
            },
        },
        "rollouts": {
            "deterministic_noise_off": deterministic,
            "nominal_noise_paired": stochastic,
        },
        "teacher_package_verification": verification,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    MD_PATH.write_text(_markdown(payload))
    print(json.dumps({"json": str(JSON_PATH), "markdown": str(MD_PATH)}, indent=2))


if __name__ == "__main__":
    main()

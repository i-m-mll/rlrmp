"""Shared numerical and orchestration helpers for the 08483d5 analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.analysis.math.cs_game_card import TARGET_POS, build_no_integrator_game
from rlrmp.analysis.math.cs_released_simulation import CSForwardNoiseDraws, CSStochasticRollout
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    robust_estimator_covariances,
)
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.train.task_model import setup_task_model_pair


DamageComputation = Mapping[str, Any]


def compute_damage_row(
    *,
    experiment: str,
    run_id: str,
    label: str,
    repo_root: Path,
    n_trials: int,
    n_steps: int,
    seed: int,
) -> DamageComputation:
    """Compute one canonical frozen-batch direct-epsilon PGD damage row."""

    run = resolve_run_inputs(
        experiment=experiment, run_ids=(run_id,), labels=(label,), repo_root=repo_root
    )[0]
    run_spec = run.run_spec
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", seed))))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=experiment,
        run_id=run_id,
        run_spec=run_spec,
        checkpoint_selection_mode="sparse_history",
        repo_root=repo_root,
    )
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_trials)
    base_epsilon = jnp.asarray(trial_specs.inputs["epsilon"], dtype=jnp.float64)
    radius = declared_active_radius(run_spec, trial_specs)
    keys = jr.split(jr.PRNGKey(seed), n_replicates)
    context = full_qrf_cost_context(
        initial_states=jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=jnp.float64),
        target_pos=trial_target_position(trial_specs),
    )

    def objective(delta: jnp.ndarray) -> jnp.ndarray:
        candidate = with_epsilon_delta(trial_specs, delta)
        costs = rollout_costs(
            model=model,
            task=pair.task,
            trial_specs=candidate,
            n_replicates=n_replicates,
            keys=keys,
            context=context,
        )
        return jnp.mean(costs["total"])

    best_delta, _best_objective, history = projected_gradient_ascent(
        objective,
        jnp.zeros_like(base_epsilon),
        radius=radius,
        step_radius=radius * 0.25,
        n_steps=n_steps,
        normalize=normalize_per_trial,
        project=project_per_trial_l2,
        flattened_norm=flattened_per_trial_norm,
    )
    condition_specs = {
        "clean": trial_specs,
        "adversarial": with_epsilon_delta(trial_specs, best_delta),
    }
    condition_costs = {
        name: summarize_costs(
            rollout_costs(
                model=model,
                task=pair.task,
                trial_specs=specs,
                n_replicates=n_replicates,
                keys=keys,
                context=context,
            )
        )
        for name, specs in condition_specs.items()
    }
    clean_costs = condition_costs["clean"]
    adversarial_costs = condition_costs["adversarial"]
    return {
        "run": run,
        "run_spec": run_spec,
        "checkpoint_selection": checkpoint_selection,
        "trial_specs": trial_specs,
        "n_replicates": n_replicates,
        "base_epsilon": base_epsilon,
        "best_delta": best_delta,
        "radius": radius,
        "history": history,
        "clean_costs": clean_costs,
        "adversarial_costs": adversarial_costs,
        "damage": delta_summary(adversarial_costs, clean_costs),
    }


def damage_payload(computation: DamageComputation) -> dict[str, Any]:
    radius = computation["radius"]
    return {
        "adversary": {
            "mechanism": "direct_epsilon_open_loop_sequence_per_trial",
            "optimizer": "projected_gradient_ascent",
            "n_steps": len(computation["history"]) - 1,
            "step_size_fraction_of_l2_radius": 0.25,
            "l2_radius_per_trial": radius,
            "history": computation["history"],
            "selected_epsilon": summarize_epsilon(computation["best_delta"], radius=radius),
            "base_epsilon": summarize_epsilon(computation["base_epsilon"], radius=None),
        },
        "costs": {
            "clean": computation["clean_costs"],
            "adversarial": computation["adversarial_costs"],
            "paired_damage": computation["damage"],
        },
    }


def stochastic_policy_rollout(
    plant: Any,
    schedule: Any,
    solution: Any,
    x0: Any,
    draws: CSForwardNoiseDraws,
    covariances: Any,
    gains: Any,
    policy: Any,
    *,
    adversarial: bool,
    config: OutputFeedbackConfig,
) -> CSStochasticRollout:
    """Roll out one robust-estimator policy under fixed stochastic draws."""

    horizon = int(gains.shape[0])
    observation = delayed_observation_matrix(plant, config)
    estimator_covariances = robust_estimator_covariances(
        plant, schedule, solution.gamma, config
    )
    inv_gamma2 = 1.0 / (solution.gamma * solution.gamma)

    x_seq = [x0.astype(jnp.float64)]
    xhat_seq = [x0.astype(jnp.float64)]
    y_clean_seq = []
    y_seq = []
    u_command_seq = []
    motor_seq = []
    sdn_seq = []
    process_seq = []
    sensory_seq = []
    eps_seq = []
    zero_eps = jnp.zeros((plant.m_w,), dtype=jnp.float64)

    for step in range(horizon):
        x_t = x_seq[-1]
        xhat_t = xhat_seq[-1]
        sigma = estimator_covariances[step]
        precision = (
            jnp.linalg.inv(sigma)
            + observation.T @ observation
            - inv_gamma2 * schedule.Q[step]
        )
        middle = jnp.linalg.inv(precision)
        y_clean = observation @ x_t
        sensory = draws.sensory[step]
        y_t = y_clean + sensory
        u_command = -gains[step] @ xhat_t
        eps_t = (
            policy[step] @ jnp.concatenate([x_t, xhat_t], axis=0)
            if adversarial
            else zero_eps
        )
        motor = draws.motor[step]
        signal_dependent = jnp.einsum(
            "j,nmj,m->n",
            draws.signal_dependent_standard[step],
            covariances.signal_dependent_state,
            u_command,
        )
        process = draws.process[step]
        innovation = y_t - observation @ xhat_t
        correction = inv_gamma2 * schedule.Q[step] @ xhat_t + observation.T @ innovation
        xhat_next = (
            plant.A @ xhat_t
            + plant.B @ u_command
            + plant.A @ middle @ correction
        )
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
        motor_seq.append(motor)
        sdn_seq.append(signal_dependent)
        process_seq.append(process)
        sensory_seq.append(sensory)
        eps_seq.append(eps_t)
        x_seq.append(x_next)
        xhat_seq.append(xhat_next)

    x = jnp.stack(x_seq, axis=0)
    u_applied = jnp.stack(u_command_seq, axis=0)
    vel = x[:, plant.vel_slice[0] : plant.vel_slice[1]]
    forward = vel @ jnp.array([1.0, 0.0], dtype=jnp.float64)
    pos = x[:, plant.pos_slice[0] : plant.pos_slice[1]]
    pos_abs = pos + TARGET_POS[None, :]
    return CSStochasticRollout(
        x=x,
        x_hat=jnp.stack(xhat_seq, axis=0),
        y_clean=jnp.stack(y_clean_seq, axis=0),
        y=jnp.stack(y_seq, axis=0),
        u_command=u_applied,
        u_applied=u_applied,
        motor_noise=jnp.stack(motor_seq, axis=0),
        signal_dependent_standard=draws.signal_dependent_standard,
        signal_dependent_noise=jnp.stack(sdn_seq, axis=0),
        process_noise=jnp.stack(process_seq, axis=0),
        sensory_noise=jnp.stack(sensory_seq, axis=0),
        adversary_epsilon=jnp.stack(eps_seq, axis=0),
        perturbations=jnp.zeros((horizon, plant.n), dtype=jnp.float64),
        peak_forward_velocity=_float(jnp.max(forward)),
        peak_forward_velocity_idx=int(jnp.argmax(forward)),
        terminal_position_error=_float(jnp.linalg.norm(pos_abs[-1] - TARGET_POS)),
        control_effort=_float(jnp.sum(u_applied**2)),
    )


def _float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def projected_gradient_ascent(
    objective: Any,
    initial_delta: Any,
    *,
    radius: float,
    step_radius: float,
    n_steps: int,
    normalize: Any,
    project: Any,
    flattened_norm: Any,
) -> tuple[Any, Any, list[dict[str, float | int]]]:
    """Maximize a frozen-batch objective with per-trial projected gradients."""

    value_and_grad = jax.value_and_grad(objective)
    delta = initial_delta
    best_delta = delta
    best_objective = objective(delta)
    history: list[dict[str, float | int]] = [
        {
            "step": 0,
            "objective": float(best_objective),
            "best_objective": float(best_objective),
            "epsilon_l2_mean": 0.0,
            "epsilon_l2_max": 0.0,
        }
    ]
    for step in range(1, n_steps + 1):
        value, grad = value_and_grad(delta)
        proposal = project(delta + normalize(grad) * step_radius, radius)
        proposal_objective = objective(proposal)
        improved = proposal_objective > best_objective
        best_delta = jnp.where(improved, proposal, best_delta)
        best_objective = jnp.where(improved, proposal_objective, best_objective)
        delta = proposal
        norms = flattened_norm(proposal)
        history.append(
            {
                "step": step,
                "objective": float(proposal_objective),
                "best_objective": float(best_objective),
                "pre_step_objective": float(value),
                "epsilon_l2_mean": float(jnp.mean(norms)),
                "epsilon_l2_max": float(jnp.max(norms)),
                "gradient_l2_mean": float(jnp.mean(flattened_norm(grad))),
            }
        )
    return best_delta, best_objective, history


def declared_active_radius(run_spec: Mapping[str, Any], trial_specs: Any) -> float:
    pgd = run_spec["hps"]["broad_epsilon_pgd_training"]
    contract = pgd["budget_contract"]
    radius_15cm = float(contract["active_max_l2_radius_15cm"])
    if not bool(pgd.get("reach_length_scaling", False)):
        return radius_15cm
    target = np.asarray(trial_target_position(trial_specs), dtype=np.float64)
    initial = np.asarray(trial_specs.inits["mechanics.vector"][..., :2], dtype=np.float64)
    reach = np.linalg.norm(target - initial, axis=-1)
    if not np.allclose(reach, reach[0], rtol=1e-10, atol=1e-12):
        raise ValueError("this sanity script expects one fixed reach length")
    reference = float(contract.get("reference_reach_m", 0.15) or 0.15)
    return radius_15cm * float(reach[0]) / reference


def trial_target_position(trial_specs: Any) -> jnp.ndarray:
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    target = jnp.asarray(target_spec.value, dtype=jnp.float64)
    if target.ndim >= 3:
        return target[:, -1, :]
    if target.ndim == 2:
        return target
    raise ValueError(f"unexpected target shape: {target.shape}")


def with_epsilon_delta(trial_specs: Any, delta: jnp.ndarray) -> Any:
    inputs = dict(trial_specs.inputs)
    base = jnp.asarray(inputs["epsilon"], dtype=delta.dtype)
    inputs["epsilon"] = base + delta
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def rollout_costs(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    keys: Any,
    context: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: bool(
            eqx.is_array(leaf)
            and getattr(leaf, "ndim", 0) >= 1
            and int(getattr(leaf, "shape", (0,))[0]) == n_replicates
        ),
    )
    batch_size = int(trial_specs.inputs["epsilon"].shape[0])

    def eval_one(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(replicate_model, trial_specs, jr.split(key, batch_size))

    states = eqx.filter_vmap(eval_one, in_axes=(0, 0))(model_arrays, keys)
    return full_qrf_cost(
        states=jnp.asarray(states.mechanics.vector, dtype=jnp.float64),
        commands=jnp.asarray(states.net.output, dtype=jnp.float64),
        context=context,
    )


def full_qrf_cost_context(
    *, initial_states: jnp.ndarray, target_pos: jnp.ndarray
) -> dict[str, Any]:
    _plant, schedule = build_no_integrator_game()
    return {
        "initial_states": initial_states,
        "target_pos": target_pos,
        "q": jnp.asarray(schedule.Q, dtype=jnp.float64),
        "r": jnp.asarray(schedule.R, dtype=jnp.float64),
        "q_f": jnp.asarray(schedule.Q_f, dtype=jnp.float64),
        "physical_dim": 6,
    }


def full_qrf_cost(
    *,
    states: jnp.ndarray,
    commands: jnp.ndarray,
    context: Mapping[str, Any],
) -> dict[str, jnp.ndarray]:
    initial = jnp.asarray(context["initial_states"], dtype=jnp.float64)
    initial = jnp.broadcast_to(initial, (*states.shape[:-2], states.shape[-1]))
    x_pre = jnp.concatenate([initial[..., None, :], states[..., :-1, :]], axis=-2)
    x_pre = goal_centered_vectors(
        x_pre,
        target_pos=jnp.asarray(context["target_pos"], dtype=jnp.float64),
        physical_dim=int(context["physical_dim"]),
    )
    x_terminal = goal_centered_vectors(
        states[..., -1, :],
        target_pos=jnp.asarray(context["target_pos"], dtype=jnp.float64),
        physical_dim=int(context["physical_dim"]),
    )
    state_terms = jnp.einsum("...ti,tij,...tj->...t", x_pre, context["q"], x_pre)
    control_terms = jnp.einsum("...ti,tij,...tj->...t", commands, context["r"], commands)
    terminal_terms = jnp.einsum(
        "...i,ij,...j->...", x_terminal, context["q_f"], x_terminal
    )
    return {
        "total": jnp.sum(state_terms, axis=-1)
        + jnp.sum(control_terms, axis=-1)
        + terminal_terms,
        "stage_state": jnp.sum(state_terms, axis=-1),
        "control": jnp.sum(control_terms, axis=-1),
        "terminal": terminal_terms,
    }


def goal_centered_vectors(
    values: jnp.ndarray,
    *,
    target_pos: jnp.ndarray,
    physical_dim: int,
) -> jnp.ndarray:
    blocks = values.shape[-1] // physical_dim
    reshaped = values.reshape((*values.shape[:-1], blocks, physical_dim))
    if target_pos.ndim == 2:
        if values.ndim == 2:
            target = target_pos[:, None, :]
        elif values.ndim == 3:
            target = target_pos[None, :, None, :]
        elif values.ndim == 4:
            target = target_pos[None, :, None, None, :]
        else:
            raise ValueError(f"unsupported values ndim {values.ndim}")
    elif target_pos.ndim == 1:
        target = target_pos
    else:
        raise ValueError(f"unsupported target ndim {target_pos.ndim}")
    centered = reshaped.at[..., 0:2].add(-target)
    return centered.reshape(values.shape)


def flattened_per_trial_norm(delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(delta.reshape((delta.shape[0], -1)), axis=-1)


def normalize_per_trial(grad: jnp.ndarray) -> jnp.ndarray:
    norms = flattened_per_trial_norm(grad)
    return grad / (norms.reshape((grad.shape[0], 1, 1)) + 1e-30)


def project_per_trial_l2(delta: jnp.ndarray, radius: float) -> jnp.ndarray:
    norms = flattened_per_trial_norm(delta)
    scale = jnp.minimum(1.0, jnp.asarray(radius, dtype=delta.dtype) / (norms + 1e-30))
    return delta * scale.reshape((delta.shape[0], 1, 1))


def summarize_costs(costs: Mapping[str, jnp.ndarray]) -> dict[str, dict[str, Any]]:
    return {key: stats(np.asarray(value, dtype=np.float64)) for key, value in costs.items()}


def stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def delta_summary(
    adversarial: Mapping[str, Mapping[str, Any]],
    clean: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "mean": float(adversarial[key]["mean"]) - float(clean[key]["mean"]),
            "note": "mean(adversarial cost) - mean(clean cost)",
        }
        for key in ("total", "stage_state", "control", "terminal")
    }


def summarize_epsilon(epsilon: Any, *, radius: float | None) -> dict[str, Any]:
    eps = np.asarray(epsilon, dtype=np.float64)
    norms = np.linalg.norm(eps.reshape((eps.shape[0], -1)), axis=-1)
    out = {
        "shape": list(eps.shape),
        "energy_total": float(np.sum(np.square(eps))),
        "energy_mean_per_trial": float(np.mean(np.sum(np.square(eps), axis=(1, 2)))),
        "l2_norm_mean_per_trial": float(np.mean(norms)),
        "l2_norm_max_per_trial": float(np.max(norms)),
        "max_abs": float(np.max(np.abs(eps))) if eps.size else 0.0,
        "boundary_fraction": None,
    }
    if radius is not None:
        out["budget_l2_radius_per_trial"] = float(radius)
        out["budget_energy_per_trial"] = float(radius * radius)
        out["boundary_fraction"] = float(np.mean(norms >= radius * (1.0 - 1e-4)))
    return out


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

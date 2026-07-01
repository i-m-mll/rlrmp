from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.analysis.math.cs_game_card import TARGET_POS, build_no_integrator_game
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.paths import REPO_ROOT
from rlrmp.train.task_model import setup_task_model_pair


ISSUE = "c92ebd8"
RUN_ID = "moderate_pgd_ofb1p4"
TARGET_DAMAGE = 6131.6906765
N_TRIALS = 64
N_STEPS = 10
SEED = 42
OUTPUT_DIR = REPO_ROOT / "results" / "08483d5" / "notes"
OUTPUT_JSON = OUTPUT_DIR / "gru_pgd_damage_sanity.json"
OUTPUT_MD = OUTPUT_DIR / "gru_pgd_damage_sanity.md"


def main() -> None:
    run = resolve_run_inputs(
        experiment=ISSUE,
        run_ids=(RUN_ID,),
        labels=("c92ebd8 moderate_pgd_ofb1p4",),
        repo_root=REPO_ROOT,
    )[0]
    run_spec = run.run_spec
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", SEED))))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=ISSUE,
        run_id=RUN_ID,
        run_spec=run_spec,
        checkpoint_selection_mode="sparse_history",
        repo_root=REPO_ROOT,
    )

    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, N_TRIALS)
    base_epsilon = jnp.asarray(trial_specs.inputs["epsilon"], dtype=jnp.float64)
    horizon = int(base_epsilon.shape[-2])
    epsilon_dim = int(base_epsilon.shape[-1])
    radius = declared_active_radius(run_spec, trial_specs)
    step_radius = radius * 0.25
    keys = jr.split(jr.PRNGKey(SEED), n_replicates)

    cost_context = full_qrf_cost_context(
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
            context=cost_context,
        )
        return jnp.mean(costs["total"])

    value_and_grad = jax.value_and_grad(objective)
    delta = jnp.zeros_like(base_epsilon)
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
    for step in range(1, N_STEPS + 1):
        value, grad = value_and_grad(delta)
        proposal = project_per_trial_l2(
            delta + normalize_per_trial(grad) * step_radius,
            radius,
        )
        proposal_objective = objective(proposal)
        improved = proposal_objective > best_objective
        best_delta = jnp.where(improved, proposal, best_delta)
        best_objective = jnp.where(improved, proposal_objective, best_objective)
        delta = proposal
        norms = flattened_per_trial_norm(proposal)
        history.append(
            {
                "step": step,
                "objective": float(proposal_objective),
                "best_objective": float(best_objective),
                "pre_step_objective": float(value),
                "epsilon_l2_mean": float(jnp.mean(norms)),
                "epsilon_l2_max": float(jnp.max(norms)),
                "gradient_l2_mean": float(jnp.mean(flattened_per_trial_norm(grad))),
            }
        )

    clean_specs = trial_specs
    adversarial_specs = with_epsilon_delta(trial_specs, best_delta)
    clean_costs = summarize_costs(
        rollout_costs(
            model=model,
            task=pair.task,
            trial_specs=clean_specs,
            n_replicates=n_replicates,
            keys=keys,
            context=cost_context,
        )
    )
    adversarial_costs = summarize_costs(
        rollout_costs(
            model=model,
            task=pair.task,
            trial_specs=adversarial_specs,
            n_replicates=n_replicates,
            keys=keys,
            context=cost_context,
        )
    )
    damage = delta_summary(adversarial_costs, clean_costs)
    epsilon_summary = summarize_epsilon(best_delta, radius=radius)
    base_epsilon_summary = summarize_epsilon(base_epsilon, radius=None)

    result = {
        "schema_version": "rlrmp.08483d5_gru_damage_sanity.v1",
        "selected_model": {
            "issue": ISSUE,
            "run_id": RUN_ID,
            "run_spec_path": str(run.run_spec_path.relative_to(REPO_ROOT)),
            "artifact_dir": str(run.artifact_dir.relative_to(REPO_ROOT)),
            "row_rationale": (
                "6D no-integrator H0 no-hold c92 row trained with the gamma 1.4 "
                "output-feedback rollout PGD budget; existing diagnostics show "
                "reach-context attenuation but not a clean across-task robustness win."
            ),
            "checkpoint_policy": "validation_selected_per_replicate",
            "checkpoint_selection": [
                selection.to_json(repo_root=REPO_ROOT) for selection in checkpoint_selection
            ],
        },
        "batch": {
            "n_trials": N_TRIALS,
            "n_replicates": n_replicates,
            "target_position_m": np.asarray(trial_target_position(trial_specs)).tolist(),
            "initial_position_m": np.asarray(
                trial_specs.inits["mechanics.vector"][..., :2], dtype=np.float64
            ).tolist(),
            "construction": (
                "repeat_single_validation_trial(pair.task.validation_trials, 64), "
                "which repeats the fixed +x 15 cm nominal validation reach."
            ),
            "seed": SEED,
            "paired_rollout_keys": True,
        },
        "adversary": {
            "mechanism": "direct_epsilon_open_loop_sequence_per_trial",
            "optimizer": "projected_gradient_ascent",
            "n_steps": N_STEPS,
            "step_size_fraction_of_l2_radius": 0.25,
            "l2_radius_per_trial": radius,
            "epsilon_dim": epsilon_dim,
            "horizon": horizon,
            "budget_source": run_spec["hps"]["broad_epsilon_pgd_training"][
                "budget_contract"
            ]["budget_source"],
            "history": history,
            "selected_epsilon": epsilon_summary,
            "base_epsilon": base_epsilon_summary,
        },
        "costs": {
            "lens": "full no-integrator C&S Q/R/Q_f task cost; no disturbance penalty",
            "clean": clean_costs,
            "adversarial": adversarial_costs,
            "paired_damage": damage,
            "target_damage_reference": {
                "source": "Mandible issue 08483d5 first no-launch output-feedback check",
                "paired_nominal_noise_damage": TARGET_DAMAGE,
            },
            "comparison_to_target": {
                "damage_minus_target": damage["total"]["mean"] - TARGET_DAMAGE,
                "damage_over_target": damage["total"]["mean"] / TARGET_DAMAGE,
            },
        },
        "uncertainties": [
            "This is a small local frozen-batch diagnostic, not a training run.",
            "The adversary is the training-style per-trial direct-epsilon PGD inner loop, not a stored training adversary.",
            "The selected c92 row has empirical reach-context attenuation but existing notes do not claim a clean H-infinity certificate or across-task robustness win.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
    OUTPUT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(json_ready(result["costs"]), indent=2, sort_keys=True))


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


def full_qrf_cost_context(*, initial_states: jnp.ndarray, target_pos: jnp.ndarray) -> dict[str, Any]:
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
    terminal_terms = jnp.einsum("...i,ij,...j->...", x_terminal, context["q_f"], x_terminal)
    return {
        "total": jnp.sum(state_terms, axis=-1) + jnp.sum(control_terms, axis=-1) + terminal_terms,
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
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def render_markdown(result: Mapping[str, Any]) -> str:
    costs = result["costs"]
    selected = result["selected_model"]
    adv = result["adversary"]
    comparison = costs["comparison_to_target"]
    return "\n".join(
        [
            "# GRU Damage Sanity Check",
            "",
            "## Selected Row",
            "",
            f"- Issue/run: `{selected['issue']}` / `{selected['run_id']}`.",
            f"- Run spec: `{selected['run_spec_path']}`.",
            f"- Artifact dir: `{selected['artifact_dir']}`.",
            f"- Checkpoint policy: `{selected['checkpoint_policy']}`.",
            f"- Rationale: {selected['row_rationale']}",
            "",
            "Existing tracked evidence: `results/c92ebd8/notes/output_feedback_budget_diagnostics.md` "
            "identifies this as the gamma 1.4 output-feedback-budget PGD row, with "
            "active L2 radius 0.004545011406169036 and reach-context attenuation, "
            "but explicitly says this is not a clean across-task robustness improvement "
            "or formal H-infinity evidence.",
            "",
            "## Frozen-Batch Diagnostic",
            "",
            f"- Batch: {result['batch']['n_trials']} repeated fixed +x 15 cm nominal validation trials "
            f"across {result['batch']['n_replicates']} GRU replicates.",
            f"- Paired rollout seed: `{result['batch']['seed']}`; clean and adversarial costs use the same keys.",
            f"- Adversary: `{adv['optimizer']}` over per-trial direct 6D epsilon sequences, "
            f"{adv['n_steps']} steps, step size {adv['step_size_fraction_of_l2_radius']} x radius.",
            f"- Radius: `{adv['l2_radius_per_trial']:.16g}` per trial; selected epsilon mean norm "
            f"`{adv['selected_epsilon']['l2_norm_mean_per_trial']:.8g}`, max norm "
            f"`{adv['selected_epsilon']['l2_norm_max_per_trial']:.8g}`; boundary fraction "
            f"`{adv['selected_epsilon']['boundary_fraction']:.3f}`.",
            f"- Selected adversarial delta energy: mean per trial "
            f"`{adv['selected_epsilon']['energy_mean_per_trial']:.8g}`, total across batch "
            f"`{adv['selected_epsilon']['energy_total']:.8g}`.",
            f"- Base nominal epsilon mean norm: `{adv['base_epsilon']['l2_norm_mean_per_trial']:.8g}`.",
            "",
            "## Costs",
            "",
            "| quantity | clean | adversarial | paired damage |",
            "|---|---:|---:|---:|",
            f"| total | {costs['clean']['total']['mean']:.8g} | "
            f"{costs['adversarial']['total']['mean']:.8g} | "
            f"{costs['paired_damage']['total']['mean']:.8g} |",
            f"| stage state | {costs['clean']['stage_state']['mean']:.8g} | "
            f"{costs['adversarial']['stage_state']['mean']:.8g} | "
            f"{costs['paired_damage']['stage_state']['mean']:.8g} |",
            f"| control | {costs['clean']['control']['mean']:.8g} | "
            f"{costs['adversarial']['control']['mean']:.8g} | "
            f"{costs['paired_damage']['control']['mean']:.8g} |",
            f"| terminal | {costs['clean']['terminal']['mean']:.8g} | "
            f"{costs['adversarial']['terminal']['mean']:.8g} | "
            f"{costs['paired_damage']['terminal']['mean']:.8g} |",
            "",
            "Costs are full no-integrator C&S Q/R/Q_f task costs; the disturbance penalty is not subtracted.",
            "",
            "## Comparison To 08483d5 Target",
            "",
            f"- Reference paired-noise output-feedback damage: `{TARGET_DAMAGE:.8g}`.",
            f"- GRU paired damage / reference: `{comparison['damage_over_target']:.4g}`.",
            f"- Difference: `{comparison['damage_minus_target']:.8g}`.",
            "",
            "Interpretation: same order of magnitude if the ratio is near 1; clearly smaller if far below 1.",
            "",
            "## Uncertainties",
            "",
            *[f"- {item}" for item in result["uncertainties"]],
            "",
            "## Machine Output",
            "",
            f"- JSON: `{OUTPUT_JSON}`",
            f"- Script: `{OUTPUT_DIR / 'gru_damage_sanity.py'}`",
            "",
        ]
    )


if __name__ == "__main__":
    main()

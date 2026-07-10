"""Evaluation-layer execution for SISU-conditioned spectrum profiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    ReplicateCheckpointSelection,
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.eval.kinematics import initial_effector_velocity
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.run_spec_access import require_run_dt, require_run_seed
from rlrmp.train.task_model import setup_task_model_pair


@dataclass(frozen=True)
class SisuCurve:
    """Velocity and endpoint summary for one run at one SISU condition."""

    sisu: float
    time_s: np.ndarray
    mean_forward_velocity_m_s: np.ndarray
    std_forward_velocity_m_s: np.ndarray
    replicate_mean_forward_velocity_m_s: np.ndarray
    endpoint_error_by_replicate_m: np.ndarray
    peak_velocity_by_replicate_m_s: np.ndarray
    final_position_by_replicate_m: np.ndarray

    @property
    def endpoint_error_mean_m(self) -> float:
        """Mean terminal endpoint error over replicates."""
        return float(np.mean(self.endpoint_error_by_replicate_m))

    @property
    def peak_velocity_mean_m_s(self) -> float:
        """Mean peak speed over replicates."""
        return float(np.mean(self.peak_velocity_by_replicate_m_s))

    @property
    def final_position_mean_m(self) -> list[float]:
        """Mean final position over replicates."""
        return [float(value) for value in np.mean(self.final_position_by_replicate_m, axis=0)]


@dataclass(frozen=True)
class RunSisuProfile:
    """SISU-conditioned velocity profiles for one trained run."""

    run_id: str
    label: str
    input_key: str
    target_final_position_m: list[float]
    validation_input_unique: list[float]
    validation_epsilon_l2_mean: float
    checkpoint_selection: tuple[ReplicateCheckpointSelection, ...]
    curves: tuple[SisuCurve, ...]


def resolve_sisu_input_key(trial_specs: Any) -> str:
    """Return the trial input key that carries SISU for these runs."""

    inputs = getattr(trial_specs, "inputs", {})
    for key in ("sisu", "input"):
        if key in inputs:
            return key
    raise ValueError("SISU-conditioned trials require an 'sisu' or 'input' input.")


def set_sisu_condition(trial_specs: Any, sisu: float, *, input_key: str | None = None) -> Any:
    """Return trial specs with the SISU scalar set to ``sisu``."""

    key = input_key or resolve_sisu_input_key(trial_specs)
    current = jnp.asarray(trial_specs.inputs[key])
    updated = eqx.tree_at(
        lambda t: t.inputs[key],
        trial_specs,
        jnp.full_like(current, float(sisu)),
    )
    if key == "sisu" and "input" in updated.inputs:
        controller_input = jnp.asarray(updated.inputs["input"])
        if controller_input.ndim >= 3 and controller_input.shape[-1] >= 2:
            sisu_column = jnp.full_like(controller_input[..., 1], float(sisu))
            updated = eqx.tree_at(
                lambda t: t.inputs["input"],
                updated,
                controller_input.at[..., 1].set(sisu_column),
            )
    return updated


def zero_disturbance_payload(trial_specs: Any, *, input_key: str = "epsilon") -> Any:
    """Return trial specs with the broad-epsilon payload zeroed."""

    if input_key not in trial_specs.inputs:
        return trial_specs
    current = jnp.asarray(trial_specs.inputs[input_key])
    return eqx.tree_at(lambda t: t.inputs[input_key], trial_specs, jnp.zeros_like(current))


def evaluate_sisu_profiles(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str],
    sisu_levels: Sequence[float],
    n_rollout_trials: int,
    use_validation_selected_checkpoints: bool = True,
    repo_root: Path = REPO_ROOT,
) -> tuple[RunSisuProfile, ...]:
    """Evaluate SISU profiles as the registered evaluation recipe's executor."""

    runs = resolve_run_inputs(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    profiles: list[RunSisuProfile] = []
    for run in runs:
        hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
        n_replicates = int(hps.model.n_replicates)
        seed = require_run_seed(run.run_spec, source=run.run_spec_path)
        pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
        if use_validation_selected_checkpoints:
            model, selections = load_validation_selected_checkpoint_model(
                experiment=experiment,
                run_id=run.run_id,
                run_spec=run.run_spec,
                repo_root=repo_root,
            )
        else:
            model, _hyperparameters = load_with_hyperparameters(
                run.artifact_dir / "trained_model.eqx",
                setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
            )
            selections = []

        base_trials = repeat_single_validation_trial(
            pair.task.validation_trials,
            n_rollout_trials,
        )
        input_key = resolve_sisu_input_key(base_trials)
        base_trials = zero_disturbance_payload(base_trials)
        target = _target_final_position(base_trials)
        initial_velocity = initial_effector_velocity(base_trials)
        model_arrays, model_other = eqx.partition(
            model,
            lambda leaf: _is_replicate_array(leaf, n_replicates),
        )
        dt = require_run_dt(run.run_spec, hps, source=run.run_spec_path)
        curves: list[SisuCurve] = []
        for sisu in sisu_levels:
            trials = set_sisu_condition(base_trials, float(sisu), input_key=input_key)

            def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
                replicate_model = eqx.combine(model_array_leaves, model_other)
                return pair.task.eval_trials(
                    replicate_model,
                    trials,
                    jr.split(key, n_rollout_trials),
                )

            states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
                model_arrays,
                jr.split(jr.PRNGKey(0), n_replicates),
            )
            position = np.asarray(states.mechanics.effector.pos, dtype=np.float64)
            velocity = np.asarray(states.mechanics.effector.vel, dtype=np.float64)
            initial_velocity_array = np.asarray(initial_velocity, dtype=np.float64)
            initial_velocity_array = np.broadcast_to(
                initial_velocity_array[None, :, None, :],
                (n_replicates, n_rollout_trials, 1, initial_velocity_array.shape[-1]),
            )
            velocity_with_initial = np.concatenate([initial_velocity_array, velocity], axis=2)
            forward = velocity_with_initial[..., 0]
            pooled_forward = forward.reshape(n_replicates * n_rollout_trials, forward.shape[-1])
            endpoint_error = np.linalg.norm(
                position[:, :, -1, :] - target[None, :, :], axis=-1
            )
            speed = np.linalg.norm(velocity, axis=-1)
            peak_speed = np.max(speed, axis=-1)
            curves.append(
                SisuCurve(
                    sisu=float(sisu),
                    time_s=np.arange(pooled_forward.shape[-1], dtype=np.float64) * dt,
                    mean_forward_velocity_m_s=np.mean(pooled_forward, axis=0),
                    std_forward_velocity_m_s=np.std(pooled_forward, axis=0),
                    replicate_mean_forward_velocity_m_s=np.mean(forward, axis=1),
                    endpoint_error_by_replicate_m=np.mean(endpoint_error, axis=1),
                    peak_velocity_by_replicate_m_s=np.mean(peak_speed, axis=1),
                    final_position_by_replicate_m=np.mean(position[:, :, -1, :], axis=1),
                )
            )

        validation_input = np.asarray(pair.task.validation_trials.inputs[input_key])
        validation_epsilon = np.asarray(pair.task.validation_trials.inputs["epsilon"])
        profiles.append(
            RunSisuProfile(
                run_id=run.run_id,
                label=run.label,
                input_key=input_key,
                target_final_position_m=[float(value) for value in np.mean(target, axis=0)],
                validation_input_unique=sorted(
                    float(value) for value in np.unique(validation_input)
                ),
                validation_epsilon_l2_mean=float(
                    np.mean(
                        np.linalg.norm(
                            validation_epsilon.reshape(validation_epsilon.shape[0], -1),
                            axis=1,
                        )
                    )
                ),
                checkpoint_selection=tuple(selections),
                curves=tuple(curves),
            )
        )
    return tuple(profiles)


def _target_final_position(trial_specs: Any) -> np.ndarray:
    """Return final target position for each trial, shape ``(trials, 2)``."""

    if "effector_target" in trial_specs.inputs:
        return np.asarray(trial_specs.inputs["effector_target"].pos[..., -1, :], dtype=np.float64)
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    return np.asarray(target_spec.value[..., -1, :], dtype=np.float64)


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


__all__ = [
    "RunSisuProfile",
    "SisuCurve",
    "evaluate_sisu_profiles",
    "resolve_sisu_input_key",
    "set_sisu_condition",
    "zero_disturbance_payload",
]

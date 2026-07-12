"""Canonical stochastic and analytical forward-velocity evaluation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax_cookbook import load_with_hyperparameters
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.analysis.gru_standard_certificate import normalize_gru_hps
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
)
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    make_cs_output_feedback_initial_state,
    position_velocity_observation_config,
)
from rlrmp.eval.checkpoint_selection import (
    ReplicateCheckpointSelection,
    load_validation_selected_checkpoint_model,
)
from rlrmp.eval.evaluation_diagnostics import DEFAULT_N_ROLLOUT_TRIALS
from rlrmp.eval.trial_inputs import (
    EvaluationRunInputs,
    initial_effector_velocity,
    repeat_single_validation_trial,
    resolve_evaluation_run_inputs,
)
from rlrmp.paths import REPO_ROOT, resolve_run_artifact_path
from rlrmp.runtime.run_spec_access import require_run_dt, require_run_seed
from rlrmp.train.task_model import setup_task_model_pair

RunFigureInputs = EvaluationRunInputs
resolve_run_inputs = resolve_evaluation_run_inputs
REFERENCE_LABEL = "C&S extLQG/output-feedback 8D"
REFERENCE_4D_LABEL = "C&S extLQG/output-feedback 4D pos+vel"


@dataclass(frozen=True)
class VelocityProfile:
    """Pooled stochastic forward-velocity profile for one run."""

    run_id: str
    label: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_replicates: int
    n_rollout_trials_per_replicate: int
    replicate_mean: np.ndarray | None = None
    replicate_std: np.ndarray | None = None
    checkpoint_selection: tuple[ReplicateCheckpointSelection, ...] = ()

    @property
    def n_pooled_samples(self) -> int:
        """Return the number of replicate/trial samples pooled into the band."""

        return self.n_replicates * self.n_rollout_trials_per_replicate


@dataclass(frozen=True)
class ReferenceProfile:
    """Analytical output-feedback reference profile."""

    label: str
    observation_channel: str
    observation_dim: int
    observed_physical_indices: tuple[int, ...]
    time_s: np.ndarray
    forward_velocity: np.ndarray
    forward_velocity_std: np.ndarray
    n_samples: int
    peak_forward_velocity_m_s: float
    time_of_peak_forward_velocity_s: float
    terminal_position_error_m: float
    gamma_factor: float
    parity_status: str
    line_color: str
    line_dash: str


def evaluate_stochastic_forward_velocity_profile(
    run: RunFigureInputs,
    *,
    n_rollout_trials: int,
    use_validation_selected_checkpoints: bool = False,
    experiment: str = "",
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> VelocityProfile:
    """Evaluate one trained GRU under repeated stochastic validation trials."""

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")
    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = require_run_seed(run.run_spec, source=run.run_spec_path)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    if use_validation_selected_checkpoints:
        model, checkpoint_selection = load_validation_selected_checkpoint_model(
            experiment=experiment,
            run_id=run.run_id,
            run_spec=run.run_spec,
            preferred_manifest_path=preferred_checkpoint_manifest_path,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if preferred_checkpoint_manifest_path is not None
                else "sparse_history"
            ),
            repo_root=repo_root,
        )
    else:
        model, _hyperparameters = load_with_hyperparameters(
            resolve_run_artifact_path(run.artifact_dir, "trained_model.eqx"),
            setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
        )
        checkpoint_selection = []
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
    initial_velocity = initial_effector_velocity(trial_specs)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, n_rollout_trials),
        )
        return jnp.concatenate(
            [initial_velocity[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )

    velocity = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    velocity_np = np.asarray(velocity, dtype=np.float64)
    forward = velocity_np[..., 0]
    pooled = forward.reshape(n_replicates * n_rollout_trials, forward.shape[-1])
    dt = require_run_dt(run.run_spec, hps, source=run.run_spec_path)
    return VelocityProfile(
        run_id=run.run_id,
        label=run.label,
        time_s=np.arange(pooled.shape[-1], dtype=np.float64) * dt,
        mean=np.mean(pooled, axis=0),
        std=np.std(pooled, axis=0),
        n_replicates=n_replicates,
        n_rollout_trials_per_replicate=n_rollout_trials,
        replicate_mean=np.mean(forward, axis=1),
        replicate_std=np.std(forward, axis=1),
        checkpoint_selection=tuple(checkpoint_selection),
    )


def cs_output_feedback_reference_profiles(
    *,
    n_samples: int = DEFAULT_N_ROLLOUT_TRIALS,
    key: Any = jr.PRNGKey(0),
) -> tuple[ReferenceProfile, ...]:
    """Return stochastic analytical C&S output-feedback references."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    config_8d = OutputFeedbackConfig()
    config_4d = position_velocity_observation_config(reference.plant, config_8d)
    keys = jr.split(key, 2)
    return (
        cs_output_feedback_reference_profile(
            reference=reference,
            config=config_8d,
            label=REFERENCE_LABEL,
            observation_channel="oldest_delayed_physical_block_full_8d",
            n_samples=n_samples,
            key=keys[0],
            line_color="#111827",
            line_dash="dash",
        ),
        cs_output_feedback_reference_profile(
            reference=reference,
            config=config_4d,
            label=REFERENCE_4D_LABEL,
            observation_channel="oldest_delayed_position_velocity_4d",
            n_samples=n_samples,
            key=keys[1],
            line_color="#f97316",
            line_dash="dot",
        ),
    )


def cs_output_feedback_reference_profile(
    *,
    reference: Any | None = None,
    config: OutputFeedbackConfig = OutputFeedbackConfig(),
    label: str = REFERENCE_LABEL,
    observation_channel: str = "oldest_delayed_physical_block_full_8d",
    n_samples: int = DEFAULT_N_ROLLOUT_TRIALS,
    key: Any = jr.PRNGKey(0),
    line_color: str = "#111827",
    line_dash: str = "dash",
) -> ReferenceProfile:
    """Return one stochastic analytical C&S output-feedback velocity profile."""

    reference = reference or materialize_reference(
        gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,)
    )
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    covariances = default_cs_noise_covariances(reference.plant, config)
    comparator = build_extlqg_comparator_path(
        reference.plant,
        reference.lqr_solution.K,
        covariances,
        schedule=reference.schedule,
        config=config,
    )
    rollouts = [
        simulate_lqg_released_forward(
            reference.plant,
            comparator.controller_gains,
            x0,
            draws=sample_forward_noise_draws(
                sample_key,
                T=reference.schedule.T,
                covariances=covariances,
            ),
            covariances=covariances,
            estimator_gains=comparator.estimator_gains,
            config=config,
        )
        for sample_key in jr.split(key, n_samples)
    ]
    x = np.stack([np.asarray(rollout.x, dtype=np.float64) for rollout in rollouts], axis=0)
    vel_lo, _vel_hi = reference.plant.vel_slice
    dt = float(reference.plant.dt)
    forward = x[:, :, vel_lo]
    mean_forward = np.mean(forward, axis=0)
    observation_matrix = delayed_observation_matrix(reference.plant, config)
    observed_indices = (
        tuple(range(config.n_phys))
        if config.observed_physical_indices is None
        else tuple(config.observed_physical_indices)
    )
    peak_idx = int(np.argmax(mean_forward))
    return ReferenceProfile(
        label=label,
        observation_channel=observation_channel,
        observation_dim=int(observation_matrix.shape[0]),
        observed_physical_indices=observed_indices,
        time_s=np.arange(mean_forward.shape[0], dtype=np.float64) * dt,
        forward_velocity=mean_forward,
        forward_velocity_std=np.std(forward, axis=0),
        n_samples=n_samples,
        peak_forward_velocity_m_s=float(mean_forward[peak_idx]),
        time_of_peak_forward_velocity_s=float(peak_idx * dt),
        terminal_position_error_m=float(
            np.mean([rollout.terminal_position_error for rollout in rollouts])
        ),
        gamma_factor=OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        parity_status=comparator.parity_status,
        line_color=line_color,
        line_dash=line_dash,
    )


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim > 0 and leaf.shape[0] == n_replicates


__all__ = [
    "DEFAULT_N_ROLLOUT_TRIALS",
    "ReferenceProfile",
    "RunFigureInputs",
    "VelocityProfile",
    "cs_output_feedback_reference_profile",
    "cs_output_feedback_reference_profiles",
    "evaluate_stochastic_forward_velocity_profile",
    "resolve_run_inputs",
]

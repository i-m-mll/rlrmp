"""Temporary standard figures for C&S GRU pilot runs.

This module is deliberately small and artifact-oriented. It exists to regenerate
the current GRU pilot figures until Feedbax Studio owns this presentation path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from jax_cookbook import load_with_hyperparameters
from feedbax.objectives.loss import TermTree
from feedbax.plot import loss_history_compare
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from plotly.subplots import make_subplots

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
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    ReplicateCheckpointSelection,
    load_validation_selected_checkpoint_model,
    materialize_validation_selected_checkpoint_manifest,
)
from rlrmp.paths import REPO_ROOT, mkdir_p, run_spec_path
from rlrmp.train.task_model import setup_task_model_pair

DEFAULT_FIGURE_SUBDIR = "tmp_figures/gru_pilot"
DEFAULT_N_ROLLOUT_TRIALS = 64
LOSS_TERMS_MODE = "union"
REFERENCE_LABEL = "C&S extLQG/output-feedback 8D"
REFERENCE_4D_LABEL = "C&S extLQG/output-feedback 4D pos+vel"


@dataclass(frozen=True)
class RunFigureInputs:
    """Resolved local inputs for one GRU pilot run."""

    run_id: str
    label: str
    run_spec_path: Path
    artifact_dir: Path
    run_spec: dict[str, Any]


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
        """Number of replicate/trial samples pooled into the band."""

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


def materialize_gru_pilot_figures(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    output_dir: Path | None = None,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    include_reference: bool = True,
    use_validation_selected_checkpoints: bool = False,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Write loss and velocity figures for listed GRU pilot runs.

    Args:
        experiment: Experiment/issue directory under ``results`` and
            ``_artifacts``.
        run_ids: Run directory names under ``runs``.
        labels: Optional display labels, one per run ID.
        output_dir: Destination for ignored figure files. Defaults to
            ``_artifacts/<experiment>/tmp_figures/gru_pilot``.
        n_rollout_trials: Number of stochastic repeats per replicate for the
            fixed validation reach.
        include_reference: Whether to overlay the analytical output-feedback
            reference on velocity panels.
        use_validation_selected_checkpoints: If true, assemble each run's model
            from the recoverable validation-selected checkpoint for each
            replicate before evaluating velocity profiles.
        repo_root: Repository root for tests and local overrides.

    Returns:
        JSON-compatible summary also written to ``figure_summary.json``.
    """

    runs = resolve_run_inputs(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    output_dir = output_dir or default_output_dir(experiment, repo_root=repo_root)
    mkdir_p(output_dir)
    selection_manifest = (
        materialize_validation_selected_checkpoint_manifest(
            experiment=experiment,
            run_ids=run_ids,
            preferred_manifest_path=preferred_checkpoint_manifest_path,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if preferred_checkpoint_manifest_path is not None
                else "sparse_history"
            ),
            repo_root=repo_root,
        )
        if use_validation_selected_checkpoints
        else None
    )

    histories = {
        run.label: load_gru_training_history(run.run_spec, run.artifact_dir / "training_history.eqx")
        for run in runs
    }
    loss_files = write_loss_figures(histories, output_dir=output_dir)

    velocity_profiles = [
        evaluate_stochastic_forward_velocity_profile(
            run,
            n_rollout_trials=n_rollout_trials,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
            experiment=experiment,
            repo_root=repo_root,
        )
        for run in runs
    ]
    reference_n_samples = max(profile.n_pooled_samples for profile in velocity_profiles)
    references = (
        cs_output_feedback_reference_profiles(n_samples=reference_n_samples)
        if include_reference
        else ()
    )
    velocity_file = write_velocity_figure(
        velocity_profiles,
        output_dir=output_dir,
        references=references,
    )
    replicate_velocity_file = write_velocity_by_replicate_figure(
        velocity_profiles,
        output_dir=output_dir,
        references=references,
    )
    alias_file = output_dir / "forward_velocity_profiles_stochastic_with_extlqg.html"
    if include_reference:
        alias_file.write_text(velocity_file.read_text(encoding="utf-8"), encoding="utf-8")

    summary = build_figure_summary(
        experiment=experiment,
        runs=runs,
        loss_files=loss_files,
        velocity_file=velocity_file,
        replicate_velocity_file=replicate_velocity_file,
        alias_file=alias_file if include_reference else None,
        velocity_profiles=velocity_profiles,
        references=references,
        selection_manifest=selection_manifest,
        checkpoint_policy=(
            "validation_selected_per_replicate"
            if use_validation_selected_checkpoints
            else "final_checkpoint"
        ),
    )
    summary_path = output_dir / "figure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def resolve_run_inputs(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    repo_root: Path = REPO_ROOT,
) -> list[RunFigureInputs]:
    """Resolve run specs and artifact directories for CLI inputs."""

    if not run_ids:
        raise ValueError("At least one run ID is required")
    labels = tuple(labels or tuple(default_label(run_id) for run_id in run_ids))
    if len(labels) != len(run_ids):
        raise ValueError("--label must be passed once per --run-id when provided")

    runs: list[RunFigureInputs] = []
    for run_id, label in zip(run_ids, labels, strict=True):
        resolved_run_spec_path = run_spec_path(experiment, run_id, repo_root=repo_root)
        artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
        if not resolved_run_spec_path.exists():
            raise FileNotFoundError(f"Missing run spec: {resolved_run_spec_path}")
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")
        runs.append(
            RunFigureInputs(
                run_id=run_id,
                label=label,
                run_spec_path=resolved_run_spec_path,
                artifact_dir=artifact_dir,
                run_spec=json.loads(resolved_run_spec_path.read_text(encoding="utf-8")),
            )
        )
    return runs


def default_output_dir(experiment: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Return the ignored default figure directory for an experiment."""

    return repo_root / "_artifacts" / experiment / DEFAULT_FIGURE_SUBDIR


def default_label(run_id: str) -> str:
    """Return a compact display label from a run ID."""

    return run_id.split("__")[-1]


def write_loss_figures(
    histories: Mapping[str, Any],
    *,
    output_dir: Path,
) -> list[Path]:
    """Write Feedbax-native training and validation loss comparison figures."""

    files: list[Path] = []
    for context, filename in (
        ("training", "loss_training.html"),
        ("validation", "loss_validation.html"),
    ):
        fig = loss_history_compare(
            histories,
            loss_context=context,
            terms=LOSS_TERMS_MODE,
            n_cols=2,
            layout_kws={"title": f"GRU pilot {context} loss"},
        )
        path = output_dir / filename
        fig.write_html(path)
        files.append(path)
    return files


def load_gru_training_history(run_spec: Mapping[str, Any], path: Path) -> SimpleNamespace:
    """Load a GRU pilot training history from the current temporary stream format.

    The 30f2313 histories were saved without a metadata skeleton. Reconstructing
    the small ``TaskTrainerHistory`` surface needed by ``loss_history_compare`` is
    sufficient here and avoids broad training-loader changes.
    """

    term_labels = active_loss_term_labels(run_spec)
    with path.open("rb") as stream:
        header = stream.readline()
        if header.strip() != b"null":
            raise ValueError(f"Expected null history metadata header in {path}")
        arrays = _read_history_arrays(stream)
    if len(arrays) < 7 or (len(arrays) - 1) % 2:
        raise ValueError(f"Unexpected GRU history array count {len(arrays)} in {path}")
    arrays_per_loss_tree = (len(arrays) - 1) // 2
    loss = _loss_tree_from_arrays(arrays[:arrays_per_loss_tree], term_labels)
    loss_validation = _loss_tree_from_arrays(
        arrays[arrays_per_loss_tree : 2 * arrays_per_loss_tree],
        term_labels,
    )
    learning_rate = arrays[-1]
    return SimpleNamespace(
        loss=loss,
        loss_validation=loss_validation,
        learning_rate=jnp.asarray(learning_rate),
    )


def active_loss_term_labels(run_spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return active loss labels in Feedbax's serialized term order."""

    loss_objective = str(run_spec.get("loss_objective") or "")
    if loss_objective == "full_analytical_qrf":
        return ("full_analytical_qrf",)

    weights = run_spec.get("hps", {}).get("loss", {}).get("weights", {})
    candidate_order = (
        "effector_pos_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "effector_vel_running",
        "effector_hold_pos",
        "effector_hold_vel",
        "effector_pos_mid",
        "effector_vel_mid",
        "effector_pos_late",
        "effector_vel_late",
        "effector_final_vel",
        "goal_hit_in_window",
        "nn_hidden",
        "nn_hidden_derivative",
        "nn_output_jerk",
        "nn_output_pre_go",
        "nn_hidden_derivative_pre_go",
        "fix_readout_norm",
        "mechanics_force_filter",
        "nn_output",
    )
    active = tuple(label for label in candidate_order if float(weights.get(label, 0.0) or 0.0) != 0.0)
    if not active:
        raise ValueError("Run spec has no active loss terms")
    return active


def evaluate_stochastic_forward_velocity_profile(
    run: RunFigureInputs,
    *,
    n_rollout_trials: int,
    use_validation_selected_checkpoints: bool = False,
    experiment: str = "",
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> VelocityProfile:
    """Evaluate one trained GRU under repeated stochastic fixed validation trials."""

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
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
            run.artifact_dir / "trained_model.eqx",
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
        # Task.eval_trials strips the prepended initial history sample. Reinsert
        # it here so plotted GRU rollouts share the analytical reference time
        # convention: sample 0 is the true trial initial state.
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
    dt = float(run.run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
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


def initial_effector_velocity(trial_specs: Any) -> jnp.ndarray:
    """Return the trial initial effector velocity array, shape ``(trials, 2)``."""

    for init_state in trial_specs.inits.values():
        velocity = getattr(init_state, "vel", None)
        if velocity is not None:
            return velocity
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 4:
            return jnp.asarray(init_state)[..., 2:4]
    raise ValueError("Trial spec does not include an effector velocity initial state")


def trial_effector_target_position(trial_specs: Any) -> np.ndarray:
    """Return target positions from legacy or delayed trial input layouts."""

    inputs = getattr(trial_specs, "inputs", {})
    target = inputs.get("effector_target") if isinstance(inputs, Mapping) else None
    if target is None and isinstance(inputs, Mapping):
        delayed_inputs = inputs.get("task")
        target = getattr(delayed_inputs, "effector_target", None)
    position = getattr(target, "pos", None)
    if position is None:
        raise KeyError("could not locate effector_target.pos in trial inputs")
    return np.asarray(position, dtype=np.float64)


def repeat_single_validation_trial(trial_specs: Any, n_trials: int) -> Any:
    """Repeat the first validation trial along its leading trial axis.

    Fixed-target rows normally expose one validation trial, but target-relative
    multi-target rows expose a structured validation bank. Plotting and rollout
    diagnostics use ``n_trials`` stochastic samples, so every leading trial-axis
    array must be sliced to a representative trial before repeating.
    """

    source_trials = _trial_count_from_spec(trial_specs)

    def repeat_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] == source_trials:
            return jnp.repeat(leaf[:1], n_trials, axis=0)
        return leaf

    return jt.map(repeat_leaf, trial_specs)


def _trial_count_from_spec(trial_specs: Any) -> int:
    for target_spec in getattr(trial_specs, "targets", {}).values():
        value = getattr(target_spec, "value", None)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    for init in getattr(trial_specs, "inits", {}).values():
        shape = getattr(init, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    inputs = getattr(trial_specs, "inputs", None)
    if isinstance(inputs, dict):
        values = inputs.values()
    else:
        values = (inputs,)
    for value in values:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    return 1


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
    sample_keys = jr.split(key, n_samples)
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
        for sample_key in sample_keys
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


def _subplot_vertical_spacing(n_rows: int, preferred: float = 0.02) -> float:
    """Return a Plotly-valid vertical spacing for a one-column subplot stack."""

    if n_rows <= 1:
        return preferred
    max_spacing = 1.0 / float(n_rows - 1)
    return min(preferred, max_spacing * 0.8)


def write_velocity_figure(
    profiles: Sequence[VelocityProfile],
    *,
    output_dir: Path,
    references: Sequence[ReferenceProfile] = (),
) -> Path:
    """Write the stochastic forward-velocity profile figure."""

    if not profiles:
        raise ValueError("At least one velocity profile is required")
    fig = make_subplots(
        rows=len(profiles),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[profile.label for profile in profiles],
        vertical_spacing=_subplot_vertical_spacing(len(profiles)),
    )
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c")
    for idx, profile in enumerate(profiles, start=1):
        color = colors[(idx - 1) % len(colors)]
        upper = profile.mean + profile.std
        lower = profile.mean - profile.std
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.18),
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name=f"{profile.label} mean +/- 1 SD",
                showlegend=idx == 1,
            ),
            row=idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=profile.time_s,
                y=profile.mean,
                mode="lines",
                line={"color": color, "width": 2},
                name=profile.label,
                showlegend=True,
            ),
            row=idx,
            col=1,
        )
        for reference in references:
            upper = reference.forward_velocity + reference.forward_velocity_std
            lower = reference.forward_velocity - reference.forward_velocity_std
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([reference.time_s, reference.time_s[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=_rgba(reference.line_color, 0.10),
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    name=f"{reference.label} mean +/- 1 SD",
                    showlegend=idx == 1,
                ),
                row=idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=reference.time_s,
                    y=reference.forward_velocity,
                    mode="lines",
                    line={
                        "color": reference.line_color,
                        "width": 2,
                        "dash": reference.line_dash,
                    },
                    name=reference.label,
                    showlegend=idx == 1,
                ),
                row=idx,
                col=1,
            )
    fig.update_layout(
        title="GRU pilot stochastic forward velocity",
        width=780,
        height=max(420, 420 * len(profiles)),
        margin={"l": 70, "r": 20, "t": 60, "b": 60},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s)", row=len(profiles), col=1)
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True)
    path = output_dir / "forward_velocity_profiles_stochastic.html"
    fig.write_html(path)
    return path


def write_velocity_by_replicate_figure(
    profiles: Sequence[VelocityProfile],
    *,
    output_dir: Path,
    references: Sequence[ReferenceProfile] = (),
) -> Path:
    """Write stochastic forward velocity by replicate, with trial-wise bands."""

    if not profiles:
        raise ValueError("At least one velocity profile is required")
    fig = make_subplots(
        rows=len(profiles),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[profile.label for profile in profiles],
        vertical_spacing=_subplot_vertical_spacing(len(profiles)),
    )
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be123c")
    for row_idx, profile in enumerate(profiles, start=1):
        if profile.replicate_mean is None or profile.replicate_std is None:
            raise ValueError(f"Missing replicate-resolved statistics for {profile.run_id}")
        for rep_idx in range(profile.n_replicates):
            color = colors[rep_idx % len(colors)]
            mean = profile.replicate_mean[rep_idx]
            std = profile.replicate_std[rep_idx]
            upper = mean + std
            lower = mean - std
            legendgroup = f"replicate-{rep_idx}"
            name = f"replicate {rep_idx}"
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.12),
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    legendgroup=legendgroup,
                    name=f"{name} mean +/- 1 SD",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=profile.time_s,
                    y=mean,
                    mode="lines",
                    line={"color": color, "width": 1.8},
                    legendgroup=legendgroup,
                    name=name,
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )
        for reference in references:
            upper = reference.forward_velocity + reference.forward_velocity_std
            lower = reference.forward_velocity - reference.forward_velocity_std
            legendgroup = f"reference-{reference.observation_channel}"
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([reference.time_s, reference.time_s[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=_rgba(reference.line_color, 0.08),
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    legendgroup=legendgroup,
                    name=f"{reference.label} mean +/- 1 SD",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=reference.time_s,
                    y=reference.forward_velocity,
                    mode="lines",
                    line={
                        "color": reference.line_color,
                        "width": 2.2,
                        "dash": reference.line_dash,
                    },
                    legendgroup=legendgroup,
                    name=reference.label,
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )
    fig.update_layout(
        title="GRU pilot stochastic forward velocity by replicate",
        width=820,
        height=max(420, 440 * len(profiles)),
        margin={"l": 70, "r": 20, "t": 60, "b": 60},
        hovermode="x unified",
        legend={"groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", row=len(profiles), col=1)
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True)
    path = output_dir / "forward_velocity_profiles_by_replicate_stochastic_with_extlqg.html"
    fig.write_html(path)
    return path


def build_figure_summary(
    *,
    experiment: str,
    runs: Sequence[RunFigureInputs],
    loss_files: Sequence[Path],
    velocity_file: Path,
    alias_file: Path | None,
    velocity_profiles: Sequence[VelocityProfile],
    replicate_velocity_file: Path | None = None,
    references: Sequence[ReferenceProfile] = (),
    selection_manifest: dict[str, Any] | None = None,
    checkpoint_policy: str = "final_checkpoint",
) -> dict[str, Any]:
    """Build the JSON sidecar summary for generated figures."""

    run_map = {run.label: run.run_id for run in runs}
    velocity_profiles_summary: dict[str, Any] = {
        profile.label: {
            "run_id": profile.run_id,
            "n_replicates": profile.n_replicates,
            "n_rollout_trials_per_replicate": profile.n_rollout_trials_per_replicate,
            "n_pooled_samples": profile.n_pooled_samples,
            "n_time_steps": int(profile.mean.shape[0]),
            "peak_mean_forward_velocity_m_s": float(np.max(profile.mean)),
            "time_of_peak_mean_forward_velocity_s": float(profile.time_s[int(np.argmax(profile.mean))]),
            "replicates": _replicate_velocity_summaries(profile),
            "checkpoint_selection": [
                selection.to_json() for selection in profile.checkpoint_selection
            ],
        }
        for profile in velocity_profiles
    }
    velocity_summary: dict[str, Any] = {
        "file": velocity_file.name,
        "implementation": (
            "Feedbax fixed validation trial repeated under stochastic runtime, "
            "compared to C&S analytical output-feedback rollout"
        ),
        "error_band": (
            "GRU mean +/- 1 SD over pooled stochastic rollout trials across replicates; "
            "analytical references mean +/- 1 SD over stochastic C&S rollouts"
        ),
        "summaries": velocity_profiles_summary,
    }
    if replicate_velocity_file is not None:
        velocity_summary["replicate_file"] = replicate_velocity_file.name
        velocity_summary["replicate_error_band"] = (
            "GRU mean +/- 1 SD over stochastic rollout trials within each replicate"
        )
    if alias_file is not None:
        velocity_summary["alias_file"] = alias_file.name
    if references:
        velocity_summary["references"] = {
            reference.label: {
                "controller": "analytical_lqr_kalman_output_feedback",
                "display_label": reference.label,
                "observation_channel": reference.observation_channel,
                "observation_dim": reference.observation_dim,
                "observed_physical_indices": list(reference.observed_physical_indices),
                "gamma_factor_recorded_for_certificate": reference.gamma_factor,
                "n_stochastic_samples": reference.n_samples,
                "parity_status": reference.parity_status,
                "n_time_steps": int(reference.forward_velocity.shape[0]),
                "peak_forward_velocity_m_s": reference.peak_forward_velocity_m_s,
                "time_of_peak_forward_velocity_s": reference.time_of_peak_forward_velocity_s,
                "terminal_position_error_m": reference.terminal_position_error_m,
            }
            for reference in references
        }

    summary = {
        "issue": experiment,
        "checkpoint_policy": checkpoint_policy,
        "runs": run_map,
        "loss_plots": {
            "implementation": "feedbax.plot.loss_history_compare",
            "terms": LOSS_TERMS_MODE,
            "error_band": "mean +/- 1 SD over replicates",
            "files": [path.name for path in loss_files],
        },
        "velocity_profiles": velocity_summary,
    }
    if selection_manifest is not None:
        summary["checkpoint_selection"] = selection_manifest
    return summary


def _read_history_arrays(stream: Any) -> list[np.ndarray]:
    """Read all NumPy arrays from a simple Feedbax history stream."""

    arrays: list[np.ndarray] = []
    while True:
        try:
            arrays.append(np.load(stream, allow_pickle=False))
        except (EOFError, ValueError):
            return arrays


def _loss_tree_from_arrays(arrays: Sequence[np.ndarray], term_labels: Sequence[str]) -> TermTree:
    """Build a loss tree from serialized value/weight pairs plus branch weight."""

    if len(arrays) < 3 or not len(arrays) % 2:
        raise ValueError(f"Expected value/weight pairs plus branch weight, got {len(arrays)}")
    n_leaves = (len(arrays) - 1) // 2
    labels = _history_term_labels(term_labels, n_leaves)
    children: dict[str, TermTree] = {}
    for idx, label in enumerate(labels):
        value = arrays[2 * idx]
        weight = _scalar_weight(arrays[2 * idx + 1])
        children[label] = TermTree.leaf(label, jnp.asarray(value), weight=weight)
    branch_weight = _scalar_weight(arrays[-1])
    return TermTree.branch("reach_loss", children, weight=branch_weight)


def _history_term_labels(term_labels: Sequence[str], n_leaves: int) -> tuple[str, ...]:
    """Return labels matching the serialized loss-tree leaf count."""

    labels = tuple(term_labels)
    if len(labels) == n_leaves:
        return labels
    if len(labels) == 1:
        return tuple(f"{labels[0]}_component_{idx}" for idx in range(n_leaves))
    return tuple(labels[:n_leaves]) + tuple(
        f"loss_component_{idx}" for idx in range(len(labels), n_leaves)
    )


def _read_loss_tree(stream: Any, term_labels: Sequence[str]) -> TermTree:
    children: dict[str, TermTree] = {}
    for label in term_labels:
        value = np.load(stream, allow_pickle=False)
        weight = _scalar_weight(np.load(stream, allow_pickle=False))
        children[label] = TermTree.leaf(label, jnp.asarray(value), weight=weight)
    branch_weight = _scalar_weight(np.load(stream, allow_pickle=False))
    return TermTree.branch("reach_loss", children, weight=branch_weight)


def _scalar_weight(value: np.ndarray) -> float:
    """Return a scalar plotting weight from Feedbax history weight records."""

    array = np.asarray(value)
    if array.size == 1:
        return float(array.reshape(()))
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return 0.0
    first = float(nonzero.reshape(-1)[0])
    if np.allclose(nonzero, first):
        return first
    return float(np.mean(nonzero))


def _replicate_velocity_summaries(profile: VelocityProfile) -> list[dict[str, float | int]]:
    if profile.replicate_mean is None or profile.replicate_std is None:
        return []
    summaries: list[dict[str, float | int]] = []
    for idx in range(profile.n_replicates):
        peak_idx = int(np.argmax(profile.replicate_mean[idx]))
        summaries.append(
            {
                "replicate": idx,
                "peak_mean_forward_velocity_m_s": float(profile.replicate_mean[idx, peak_idx]),
                "time_of_peak_mean_forward_velocity_s": float(profile.time_s[peak_idx]),
                "trial_sd_at_peak_m_s": float(profile.replicate_std[idx, peak_idx]),
            }
        )
    return summaries


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


def _rgba(hex_color: str, alpha: float) -> str:
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return f"rgba({red},{green},{blue},{alpha})"


__all__ = [
    "DEFAULT_FIGURE_SUBDIR",
    "DEFAULT_N_ROLLOUT_TRIALS",
    "REFERENCE_LABEL",
    "REFERENCE_4D_LABEL",
    "RunFigureInputs",
    "VelocityProfile",
    "active_loss_term_labels",
    "build_figure_summary",
    "cs_output_feedback_reference_profiles",
    "cs_output_feedback_reference_profile",
    "default_label",
    "default_output_dir",
    "evaluate_stochastic_forward_velocity_profile",
    "load_gru_training_history",
    "materialize_gru_pilot_figures",
    "repeat_single_validation_trial",
    "resolve_run_inputs",
    "write_loss_figures",
    "write_velocity_figure",
    "write_velocity_by_replicate_figure",
]

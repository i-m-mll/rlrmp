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
from feedbax._io import load_with_hyperparameters
from feedbax.loss import TermTree
from feedbax.plot import loss_history_compare
from feedbax.types import TreeNamespace, dict_to_namespace
from plotly.subplots import make_subplots

from rlrmp.analysis.cs_game_card import OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
from rlrmp.analysis.cs_game_card import materialize_reference
from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    make_cs_output_feedback_initial_state,
    position_velocity_observation_config,
    rollout_with_kalman_estimator,
)
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


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
    peak_forward_velocity_m_s: float
    time_of_peak_forward_velocity_s: float
    terminal_position_error_m: float
    gamma_factor: float
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

    histories = {
        run.label: load_gru_training_history(run.run_spec, run.artifact_dir / "training_history.eqx")
        for run in runs
    }
    loss_files = write_loss_figures(histories, output_dir=output_dir)

    velocity_profiles = [
        evaluate_stochastic_forward_velocity_profile(
            run,
            n_rollout_trials=n_rollout_trials,
        )
        for run in runs
    ]
    references = cs_output_feedback_reference_profiles() if include_reference else ()
    velocity_file = write_velocity_figure(
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
        alias_file=alias_file if include_reference else None,
        velocity_profiles=velocity_profiles,
        references=references,
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
        run_spec_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
        artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
        if not run_spec_path.exists():
            raise FileNotFoundError(f"Missing run spec: {run_spec_path}")
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")
        runs.append(
            RunFigureInputs(
                run_id=run_id,
                label=label,
                run_spec_path=run_spec_path,
                artifact_dir=artifact_dir,
                run_spec=json.loads(run_spec_path.read_text(encoding="utf-8")),
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
        loss = _read_loss_tree(stream, term_labels)
        loss_validation = _read_loss_tree(stream, term_labels)
        learning_rate = np.load(stream, allow_pickle=False)
    return SimpleNamespace(
        loss=loss,
        loss_validation=loss_validation,
        learning_rate=jnp.asarray(learning_rate),
    )


def active_loss_term_labels(run_spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return active loss labels in Feedbax's serialized term order."""

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
) -> VelocityProfile:
    """Evaluate one trained GRU under repeated stochastic fixed validation trials."""

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, _hyperparameters = load_with_hyperparameters(
        run.artifact_dir / "trained_model.eqx",
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
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
        return states.mechanics.effector.vel

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
    )


def repeat_single_validation_trial(trial_specs: Any, n_trials: int) -> Any:
    """Repeat a one-trial validation spec along its leading trial axis."""

    def repeat_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] == 1:
            return jnp.repeat(leaf, n_trials, axis=0)
        return leaf

    return jt.map(repeat_leaf, trial_specs)


def cs_output_feedback_reference_profiles() -> tuple[ReferenceProfile, ...]:
    """Return analytical C&S output-feedback references for 8D and 4D observations."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    config_8d = OutputFeedbackConfig()
    config_4d = position_velocity_observation_config(reference.plant, config_8d)
    return (
        cs_output_feedback_reference_profile(
            reference=reference,
            config=config_8d,
            label=REFERENCE_LABEL,
            observation_channel="oldest_delayed_physical_block_full_8d",
            line_color="#111827",
            line_dash="dash",
        ),
        cs_output_feedback_reference_profile(
            reference=reference,
            config=config_4d,
            label=REFERENCE_4D_LABEL,
            observation_channel="oldest_delayed_position_velocity_4d",
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
    line_color: str = "#111827",
    line_dash: str = "dash",
) -> ReferenceProfile:
    """Return one analytical C&S output-feedback forward-velocity profile."""

    reference = reference or materialize_reference(
        gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,)
    )
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    rollout = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        x0,
        config=config,
    )
    x = np.asarray(rollout.x, dtype=np.float64)
    vel_lo, _vel_hi = reference.plant.vel_slice
    dt = float(reference.plant.dt)
    observation_matrix = delayed_observation_matrix(reference.plant, config)
    observed_indices = (
        tuple(range(config.n_phys))
        if config.observed_physical_indices is None
        else tuple(config.observed_physical_indices)
    )
    return ReferenceProfile(
        label=label,
        observation_channel=observation_channel,
        observation_dim=int(observation_matrix.shape[0]),
        observed_physical_indices=observed_indices,
        time_s=np.arange(x.shape[0], dtype=np.float64) * dt,
        forward_velocity=x[:, vel_lo],
        peak_forward_velocity_m_s=float(rollout.peak_forward_velocity),
        time_of_peak_forward_velocity_s=float(rollout.peak_forward_velocity_idx * dt),
        terminal_position_error_m=float(rollout.terminal_position_error),
        gamma_factor=OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        line_color=line_color,
        line_dash=line_dash,
    )


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
        rows=1,
        cols=len(profiles),
        shared_yaxes=True,
        subplot_titles=[profile.label for profile in profiles],
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
            row=1,
            col=idx,
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
            row=1,
            col=idx,
        )
        for reference in references:
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
                row=1,
                col=idx,
            )
    fig.update_layout(
        title="GRU pilot stochastic forward velocity",
        width=max(520, 430 * len(profiles)),
        height=420,
        margin={"l": 70, "r": 20, "t": 60, "b": 60},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True)
    path = output_dir / "forward_velocity_profiles_stochastic.html"
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
    references: Sequence[ReferenceProfile] = (),
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
            "analytical trace has no band"
        ),
        "summaries": velocity_profiles_summary,
    }
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
                "n_time_steps": int(reference.forward_velocity.shape[0]),
                "peak_forward_velocity_m_s": reference.peak_forward_velocity_m_s,
                "time_of_peak_forward_velocity_s": reference.time_of_peak_forward_velocity_s,
                "terminal_position_error_m": reference.terminal_position_error_m,
            }
            for reference in references
        }

    return {
        "issue": experiment,
        "runs": run_map,
        "loss_plots": {
            "implementation": "feedbax.plot.loss_history_compare",
            "terms": LOSS_TERMS_MODE,
            "error_band": "mean +/- 1 SD over replicates",
            "files": [path.name for path in loss_files],
        },
        "velocity_profiles": velocity_summary,
    }


def _read_loss_tree(stream: Any, term_labels: Sequence[str]) -> TermTree:
    children: dict[str, TermTree] = {}
    for label in term_labels:
        value = np.load(stream, allow_pickle=False)
        weight = float(np.load(stream, allow_pickle=False))
        children[label] = TermTree.leaf(label, jnp.asarray(value), weight=weight)
    branch_weight = float(np.load(stream, allow_pickle=False))
    return TermTree.branch("reach_loss", children, weight=branch_weight)


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
]

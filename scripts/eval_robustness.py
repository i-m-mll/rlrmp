"""Robustness evaluation and publication-quality figure generation for Part 2.5.

Loads trained models from results/2ef67ca/models/, evaluates under fixed gust
perturbations at multiple amplitudes and SISU levels, and produces Plotly figures.

Figures produced:
    fig1_aligned_trajectories.html  — 2D positions aligned to reach direction
    fig2_lateral_velocity.html      — Lateral velocity profiles over time
    fig3_lateral_force.html         — Lateral force profiles over time
    fig4_forward_velocity_no_pert.html — Forward velocity without perturbation
    fig5_endpoint_error_violin.html — Endpoint error vs SISU (violin plots)
    fig6_all_loss_curves.html       — Loss curves for all conditions (grid of subplots)

Usage:
    python scripts/eval_robustness.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

WORKTREE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKTREE / "scripts"))

from train_part2_5 import build_hps  # noqa: E402
from eval_part2_5_figures import (  # noqa: E402
    eval_ensemble_on_trials,
    compute_kinematics,
    set_sisu,
    N_REPLICATES,
)
from feedbax._io import load_with_hyperparameters  # noqa: E402
from feedbax.plot import save_figure  # noqa: E402  # Bug: f485c26 — project-config routing
from feedbax.train import init_task_trainer_history  # noqa: E402
from rlrmp.modules.training.part2 import setup_task_model_pair  # noqa: E402
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

RESULTS_BASE = WORKTREE / "results" / "2ef67ca"  # legacy Part 2.5 archive (Bug: f485c26)
MODELS_BASE = RESULTS_BASE / "models"
FIGURES_DIR = RESULTS_BASE / "figures"

# Conditions to evaluate: (display_name, model_dir_name)
CONDITIONS = [
    ("Standard (pert_std=1)", "running_cost_standard"),
    ("Baseline (no pert training)", "baseline_no_pert"),
    ("APT (lr=0.001, pert_std=1)", "apt_lr001"),
    ("APT (lr=0.01, pert_std=2)", "apt_pert2"),
]

# Eval parameters
SISU_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
PERT_AMPLITUDES = [0.0, 0.5, 1.0, 2.0]
EVAL_SISU = 0.5        # Fixed SISU for trajectory and profile figures
EVAL_PERT_AMP = 5.0    # Fixed perturbation amplitude for profile figures

# Colors: color = training method (blue=standard, gray=baseline, red=APT variants).
# All lines are solid — dash/dot distinction removed.
CONDITION_COLORS = {
    "Standard (pert_std=1)": "#1f77b4",         # blue
    "Baseline (no pert training)": "#888888",    # gray
    "APT (lr=0.001, pert_std=1)": "#d62728",     # red
    "APT (lr=0.01, pert_std=2)": "#e07a5f",      # salmon
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_condition(display_name: str, model_dir_name: str):
    """Load task, model, and training history for a single condition.

    Returns:
        Tuple (task, model, history, n_batches) or None if not found.
    """
    cond_dir = MODELS_BASE / model_dir_name
    config_path = cond_dir / "config.json"
    if not config_path.exists():
        print(f"  {display_name}: skipped — not found at {cond_dir}")
        return None

    with open(config_path) as f:
        config = json.load(f)

    args = argparse.Namespace(**config)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    n_batches = config.get("n_batches", 10000)
    loss_func = pair.task.loss_func
    history_skeleton = init_task_trainer_history(
        loss_func=loss_func,
        n_batches=n_batches,
        n_replicates=N_REPLICATES,
        ensembled=True,
    )
    history_path = cond_dir / "train_history.eqx"
    with open(history_path, "rb") as f:
        f.readline()  # skip hyperparameters line
        # The file was saved with only multi-element JAX arrays (training curves), not the
        # scalar loss weight leaves. Build a skeleton with float leaves promoted to 0-d
        # JAX arrays so the tree structure matches the saved file, then use a filter_spec
        # that only reads from file for multi-element arrays (skipping scalars).
        def _float_to_array(x):
            if isinstance(x, (float, int)) and not isinstance(x, bool):
                return jnp.array(x)
            return x

        history_skeleton_arr = jt.map(_float_to_array, history_skeleton)

        def _multi_array_filter_spec(file, x):
            """Load from file only for multi-element arrays; keep default for scalars."""
            if isinstance(x, jax.Array) and x.size > 1:
                return jnp.load(file)
            elif isinstance(x, np.ndarray) and x.size > 1:
                return np.load(file)
            else:
                return x  # keep skeleton default, don't advance file pointer

        history = eqx.tree_deserialise_leaves(
            f, history_skeleton_arr, filter_spec=_multi_array_filter_spec
        )

    trained_model, _ = load_with_hyperparameters(
        cond_dir / "trained_model.eqx",
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    return pair.task, trained_model, history, n_batches


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------


def get_reach_direction(trial_specs):
    """Compute per-trial reach direction vectors.

    Returns:
        direction_unit: (n_trials, 2) unit vectors pointing from start to goal.
        init_pos: (n_trials, 2) starting positions at go cue.
        goal_pos: (n_trials, 2) goal positions.
    """
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,)
    # targets value shape: (n_trials, n_steps, 2)
    target_key = list(trial_specs.targets.keys())[0]
    goal_pos = trial_specs.targets[target_key].value[:, -1, :]  # (n_trials, 2)

    # We need starting position at go cue — use inits effector pos
    init_pos_fixed = trial_specs.inits["mechanics.effector"].pos  # (n_trials, 2)

    direction = goal_pos - init_pos_fixed  # (n_trials, 2)
    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction_unit = direction / np.maximum(direction_norm, 1e-12)
    return direction_unit, init_pos_fixed, goal_pos


def project_to_aligned(arr, direction_unit):
    """Project 2D array (..., n_trials, n_steps, 2) into (parallel, lateral) components.

    Args:
        arr: Array of shape (..., n_trials, n_steps, 2).
        direction_unit: (n_trials, 2) unit reach direction vectors.

    Returns:
        parallel: (..., n_trials, n_steps) component along reach direction.
        lateral: (..., n_trials, n_steps) component perpendicular to reach.
    """
    # direction_unit: (n_trials, 2) -> broadcast to (..., n_trials, 1, 2)
    du = direction_unit[..., np.newaxis, :]  # (n_trials, 1, 2)
    # parallel: dot product with direction
    parallel = np.sum(arr * du, axis=-1)  # (..., n_trials, n_steps)
    # lateral: cross product (2D: x*dy - y*dx)
    # du shape: (n_trials, 1, 2), arr shape: (..., n_trials, n_steps, 2)
    lateral = arr[..., 0] * du[..., 1] - arr[..., 1] * du[..., 0]
    # lateral sign: positive = left of direction
    lateral = lateral  # (..., n_trials, n_steps)
    return parallel, lateral


def align_positions(pos, direction_unit, init_pos):
    """Subtract origin (init_pos) then project.

    Args:
        pos: (n_rep, n_trials, n_steps, 2)
        direction_unit: (n_trials, 2)
        init_pos: (n_trials, 2) — origin to subtract.

    Returns:
        parallel, lateral: each (n_rep, n_trials, n_steps).
    """
    pos_centered = pos - init_pos[np.newaxis, :, np.newaxis, :]
    return project_to_aligned(pos_centered, direction_unit)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_fixed_pert(task, model, sisu: float, pert_amp: float, *, key, ref_task=None):
    """Evaluate model at a fixed SISU and perturbation amplitude.

    When pert_amp > 0, trial specs (including gust amplitudes) are drawn from
    ref_task if provided, so that models trained with pert_std=0 are still
    evaluated under meaningful perturbations.  For pert_amp=0 the model's own
    task is always used.

    Args:
        task: The model's own task (used for unperturbed eval and model runner).
        model: The trained model to evaluate.
        sisu: SISU level to set.
        pert_amp: Perturbation scale factor to apply.
        key: JAX PRNGKey.
        ref_task: Reference task whose validation_trials supply gust amplitudes
            for perturbed evaluations.  Required when pert_amp > 0 and the
            model's own task has pert_std=0.  If None, falls back to task.

    Returns:
        states: model states, shape (n_rep, n_trials, n_steps, ...).
        trial_specs: the modified trial specs used.
    """
    # Use ref_task's trials for perturbed conditions so that models trained
    # without perturbations (pert_std=0) still receive non-zero gusts.
    source_task = (ref_task if (ref_task is not None and pert_amp > 0) else task)
    val_trials = source_task.validation_trials
    # Set SISU
    trial_specs = set_sisu(val_trials, sisu)
    # Set perturbation amplitude (scale)
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.full(trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape, pert_amp),
    )
    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return states, trial_specs


def get_go_idx(trial_specs):
    """Return (n_trials,) array of go-cue step indices."""
    return np.array(trial_specs.timeline.epoch_bounds[:, 2])


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def add_mean_std_band(
    fig,
    x,
    data,
    color,
    name,
    showlegend=True,
    *,
    row=None,
    col=None,
    legendgroup=None,
):
    """Add a mean line + ±1 std band to a Plotly figure.

    All lines are drawn solid; dash/dot distinction is not used.

    Args:
        data: (n_samples, n_timepoints) — mean and std computed over axis 0.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    ub = mean + std
    lb = mean - std

    rgba_fill = _hex_to_rgba(color, alpha=0.25)

    kwargs = dict(row=row, col=col) if row is not None else {}

    fig.add_trace(
        go.Scatter(
            x=x,
            y=ub,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=legendgroup or name,
        ),
        **kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lb,
            mode="lines",
            fill="tonexty",
            fillcolor=rgba_fill,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=legendgroup or name,
        ),
        **kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            name=name,
            line=dict(color=color, dash="solid", width=2),
            showlegend=showlegend,
            legendgroup=legendgroup or name,
        ),
        **kwargs,
    )


def _hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
    """Convert hex color string to rgba string."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Gust timing constants (must match disturbance.py get_fixed_gust_fn defaults)
_GUST_START_PROP = 0.1
_GUST_END_PROP = 0.75


def get_gust_window_seconds(trial_specs, dt: float = 0.01) -> tuple[float, float] | None:
    """Return the gust window (t_start, t_end) in seconds relative to the go cue.

    The gust is active from go_cue + start_prop*move_len to go_cue + end_prop*move_len
    (see disturbance.py::get_fixed_gust_fn). Returns None if timeline has no epochs.

    Args:
        trial_specs: Trial specs with timeline.epoch_bounds.
        dt: Timestep duration in seconds.

    Returns:
        (t_start, t_end) in seconds measured from go cue, or None.
    """
    if trial_specs.timeline is None or trial_specs.timeline.epoch_bounds is None:
        return None
    if not trial_specs.timeline.has_epochs:
        return None

    epoch_bounds = np.array(trial_specs.timeline.epoch_bounds)  # (n_trials, E+1)
    epoch_names = trial_specs.timeline.epoch_names

    if "movement" not in epoch_names:
        return None

    move_idx = list(epoch_names).index("movement")
    # epoch_bounds columns: [epoch0_start, epoch1_start, ..., epochN_start, end]
    # move epoch: [epoch_bounds[:, move_idx], epoch_bounds[:, move_idx+1])
    move_start = np.mean(epoch_bounds[:, move_idx])
    move_end = np.mean(epoch_bounds[:, move_idx + 1])
    move_len = move_end - move_start

    gust_start_step = move_start + int(np.floor(move_len * _GUST_START_PROP))
    gust_end_step = min(move_start + int(np.floor(move_len * _GUST_END_PROP)), move_end)

    # Convert from absolute step to seconds relative to go cue (= move_start)
    t_start = (gust_start_step - move_start) * dt
    t_end = (gust_end_step - move_start) * dt
    return float(t_start), float(t_end)


def add_gust_vrect(fig, trial_specs, *, row=None, col=None, dt: float = 0.01) -> None:
    """Add a light-grey vrect showing when the gust perturbation is active.

    The x-axis is assumed to be time from go cue (seconds). Does nothing if gust
    timing cannot be determined from trial_specs.

    Args:
        fig: Plotly Figure to annotate.
        trial_specs: Trial specs containing timeline with epoch info.
        row: Subplot row (1-indexed). Pass None for a non-subplot figure.
        col: Subplot col (1-indexed). Pass None for a non-subplot figure.
        dt: Timestep duration in seconds.
    """
    window = get_gust_window_seconds(trial_specs, dt=dt)
    if window is None:
        return

    t_start, t_end = window
    vrect_kwargs = dict(
        x0=t_start,
        x1=t_end,
        fillcolor="lightgrey",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="gust",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="grey",
    )
    if row is not None:
        vrect_kwargs["row"] = row
        vrect_kwargs["col"] = col
    fig.add_vrect(**vrect_kwargs)


# ---------------------------------------------------------------------------
# Figure 1: Aligned trajectories under perturbation
# ---------------------------------------------------------------------------


def make_fig1_aligned_trajectories(loaded: dict, ref_task=None) -> go.Figure:
    """2D position plots aligned to reach direction, one subplot per condition.

    Two rows: without perturbation (top), with perturbation EVAL_PERT_AMP (bottom).

    Args:
        loaded: Dict of condition data.
        ref_task: Reference task used for perturbed evaluations (see eval_fixed_pert).
    """
    n_conds = len(loaded)
    n_rows = 2
    subplot_titles = []
    for pert_label in ["No perturbation", f"Perturbation amp={EVAL_PERT_AMP}"]:
        for name in loaded:
            subplot_titles.append(f"{name}<br>({pert_label})")

    fig = make_subplots(
        rows=n_rows,
        cols=n_conds,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for col, (cond_name, d) in enumerate(loaded.items(), 1):
        color = CONDITION_COLORS[cond_name]
        task = d["task"]
        model = d["model"]

        for row, pert_amp in enumerate([0.0, EVAL_PERT_AMP], 1):
            key = jr.PRNGKey(7 + col * 100 + row * 10)
            states, trial_specs = eval_fixed_pert(
                task, model, EVAL_SISU, pert_amp, key=key, ref_task=ref_task
            )

            pos = np.array(states.mechanics.effector.pos)  # (n_rep, n_trials, n_steps, 2)
            direction_unit, init_pos, goal_pos = get_reach_direction(trial_specs)

            par_pos, lat_pos = align_positions(pos, direction_unit, init_pos)

            n_rep, n_trials, n_steps = par_pos.shape
            go_idx_arr = get_go_idx(trial_specs)

            # Plot one replicate, all trials
            rep_idx = 0
            for trial_idx in range(n_trials):
                go_step = go_idx_arr[trial_idx]
                x = par_pos[rep_idx, trial_idx, go_step:]
                y = lat_pos[rep_idx, trial_idx, go_step:]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.5,
                        showlegend=(col == 1 and row == 1 and trial_idx == 0),
                        name=cond_name,
                        legendgroup=cond_name,
                    ),
                    row=row,
                    col=col,
                )

            # Mark goal
            avg_goal_par = np.mean(
                np.sum(goal_pos * direction_unit, axis=-1)
                - np.sum(init_pos * direction_unit, axis=-1)
            )
            fig.add_trace(
                go.Scatter(
                    x=[avg_goal_par],
                    y=[0.0],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="Aligned reach trajectories by training condition",
        height=600,
        width=260 * n_conds,
        legend_title="Condition",
        font_size=11,
    )

    # Enforce square aspect ratio on every subplot so lateral deviations are
    # visually proportional to the forward distance.
    # scaleanchor uses axis *reference* format (x, x2, x3, ...) not layout key names.
    for col in range(1, n_conds + 1):
        for row in range(1, n_rows + 1):
            axis_idx = (row - 1) * n_conds + col
            y_axis_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
            # axis reference: "x" for the first axis, "x2", "x3", ... for the rest
            x_ref = "x" if axis_idx == 1 else f"x{axis_idx}"
            fig.update_layout({
                y_axis_key: dict(scaleanchor=x_ref, scaleratio=1),
            })

    # Axis labels only on edge subplots
    for col in range(1, n_conds + 1):
        for row in range(1, n_rows + 1):
            axis_idx = (row - 1) * n_conds + col
            x_axis_key = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
            y_axis_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
            if row == n_rows:
                fig.update_layout({x_axis_key: dict(title_text="Parallel (m)")})
            if col == 1:
                fig.update_layout({y_axis_key: dict(title_text="Lateral (m)")})

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Lateral velocity profiles
# ---------------------------------------------------------------------------


def make_fig2_lateral_velocity(loaded: dict, ref_task=None) -> go.Figure:
    """Time-series of lateral velocity under fixed eval perturbation, aligned to go cue.

    Mean ± 1 std across trials × replicates, one line per condition.
    A grey vrect shows when the gust perturbation is active.

    Args:
        loaded: Dict of condition data.
        ref_task: Reference task used for perturbed evaluations (see eval_fixed_pert).
    """
    fig = go.Figure()

    last_trial_specs = None
    for cond_name, d in loaded.items():
        color = CONDITION_COLORS[cond_name]
        task = d["task"]
        model = d["model"]
        key = jr.PRNGKey(42)

        states, trial_specs = eval_fixed_pert(
            task, model, EVAL_SISU, EVAL_PERT_AMP, key=key, ref_task=ref_task
        )
        last_trial_specs = trial_specs
        vel = np.array(states.mechanics.effector.vel)  # (n_rep, n_trials, n_steps, 2)
        direction_unit, _, _ = get_reach_direction(trial_specs)
        go_idx_arr = get_go_idx(trial_specs)

        _, lat_vel = project_to_aligned(vel, direction_unit)  # (n_rep, n_trials, n_steps)

        # Align to go cue: trim to earliest common window after go cue
        go_step_min = int(go_idx_arr.min())
        n_steps = lat_vel.shape[-1]
        post_go_len = n_steps - go_step_min

        # Gather post-go traces: for each (rep, trial) clip from go_idx onward
        traces = []
        for rep_idx in range(lat_vel.shape[0]):
            for trial_idx in range(lat_vel.shape[1]):
                go_step = go_idx_arr[trial_idx]
                trace = lat_vel[rep_idx, trial_idx, go_step:]
                # Pad or trim to common length
                trace = trace[:post_go_len]
                if len(trace) < post_go_len:
                    trace = np.pad(trace, (0, post_go_len - len(trace)), constant_values=np.nan)
                traces.append(trace)

        traces = np.array(traces)  # (n_rep * n_trials, post_go_len)
        t = np.arange(post_go_len) * 0.01  # in seconds (dt=0.01)

        add_mean_std_band(
            fig,
            x=t,
            data=traces,
            color=color,
            name=cond_name,
            showlegend=True,
            legendgroup=cond_name,
        )

    # Add gust perturbation timing indicator (vrect) below all data traces
    if last_trial_specs is not None:
        add_gust_vrect(fig, last_trial_specs)

    fig.add_hline(y=0, line_dash="dot", line_color="lightgray", line_width=1)
    fig.update_layout(
        title=f"Lateral velocity profiles under fixed perturbation (SISU={EVAL_SISU}, amp={EVAL_PERT_AMP})",
        xaxis_title="Time from go cue (s)",
        yaxis_title="Lateral velocity (m/s)",
        height=450,
        width=700,
        legend_title="Condition",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Lateral force profiles
# ---------------------------------------------------------------------------


def make_fig3_lateral_force(loaded: dict, ref_task=None) -> go.Figure:
    """Time-series of lateral force output under fixed eval perturbation.

    Same format as Fig 2 but for force_filter.output lateral component.
    A grey vrect shows when the gust perturbation is active.

    Args:
        loaded: Dict of condition data.
        ref_task: Reference task used for perturbed evaluations (see eval_fixed_pert).
    """
    fig = go.Figure()

    last_trial_specs = None
    for cond_name, d in loaded.items():
        color = CONDITION_COLORS[cond_name]
        task = d["task"]
        model = d["model"]
        key = jr.PRNGKey(43)

        states, trial_specs = eval_fixed_pert(
            task, model, EVAL_SISU, EVAL_PERT_AMP, key=key, ref_task=ref_task
        )
        last_trial_specs = trial_specs
        force = np.array(states.force_filter.output)  # (n_rep, n_trials, n_steps, 2)
        direction_unit, _, _ = get_reach_direction(trial_specs)
        go_idx_arr = get_go_idx(trial_specs)

        _, lat_force = project_to_aligned(force, direction_unit)  # (n_rep, n_trials, n_steps)

        go_step_min = int(go_idx_arr.min())
        n_steps = lat_force.shape[-1]
        post_go_len = n_steps - go_step_min

        traces = []
        for rep_idx in range(lat_force.shape[0]):
            for trial_idx in range(lat_force.shape[1]):
                go_step = go_idx_arr[trial_idx]
                trace = lat_force[rep_idx, trial_idx, go_step:][:post_go_len]
                if len(trace) < post_go_len:
                    trace = np.pad(trace, (0, post_go_len - len(trace)), constant_values=np.nan)
                traces.append(trace)

        traces = np.array(traces)
        t = np.arange(post_go_len) * 0.01

        add_mean_std_band(
            fig,
            x=t,
            data=traces,
            color=color,
            name=cond_name,
            showlegend=True,
            legendgroup=cond_name,
        )

    # Add gust perturbation timing indicator (vrect) below all data traces
    if last_trial_specs is not None:
        add_gust_vrect(fig, last_trial_specs)

    fig.add_hline(y=0, line_dash="dot", line_color="lightgray", line_width=1)
    fig.update_layout(
        title=f"Lateral force profiles under fixed perturbation (SISU={EVAL_SISU}, amp={EVAL_PERT_AMP})",
        xaxis_title="Time from go cue (s)",
        yaxis_title="Lateral force (N)",
        height=450,
        width=700,
        legend_title="Condition",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Forward velocity WITHOUT perturbation
# ---------------------------------------------------------------------------


def make_fig4_forward_velocity_no_pert(loaded: dict) -> go.Figure:
    """Forward (parallel) velocity profiles without perturbation, showing peak differences."""
    fig = go.Figure()

    for cond_name, d in loaded.items():
        color = CONDITION_COLORS[cond_name]
        task = d["task"]
        model = d["model"]
        key = jr.PRNGKey(44)

        states, trial_specs = eval_fixed_pert(task, model, EVAL_SISU, 0.0, key=key)
        vel = np.array(states.mechanics.effector.vel)  # (n_rep, n_trials, n_steps, 2)
        direction_unit, _, _ = get_reach_direction(trial_specs)
        go_idx_arr = get_go_idx(trial_specs)

        par_vel, _ = project_to_aligned(vel, direction_unit)  # (n_rep, n_trials, n_steps)

        go_step_min = int(go_idx_arr.min())
        n_steps = par_vel.shape[-1]
        post_go_len = n_steps - go_step_min

        traces = []
        for rep_idx in range(par_vel.shape[0]):
            for trial_idx in range(par_vel.shape[1]):
                go_step = go_idx_arr[trial_idx]
                trace = par_vel[rep_idx, trial_idx, go_step:][:post_go_len]
                if len(trace) < post_go_len:
                    trace = np.pad(trace, (0, post_go_len - len(trace)), constant_values=np.nan)
                traces.append(trace)

        traces = np.array(traces)
        t = np.arange(post_go_len) * 0.01

        add_mean_std_band(
            fig,
            x=t,
            data=traces,
            color=color,
            name=cond_name,
            showlegend=True,
            legendgroup=cond_name,
        )

    fig.add_hline(y=0, line_dash="dot", line_color="lightgray", line_width=1)
    fig.update_layout(
        title=f"Forward velocity profiles — no perturbation (SISU={EVAL_SISU})",
        xaxis_title="Time from go cue (s)",
        yaxis_title="Forward velocity (m/s)",
        height=450,
        width=700,
        legend_title="Condition",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Endpoint error vs SISU (violin plots)
# ---------------------------------------------------------------------------


def make_fig5_endpoint_error_violin(loaded: dict, ref_task=None) -> go.Figure:
    """Violin plots of endpoint error by SISU level, grouped by condition.

    Uses go.Violin with violinmode='group'. Each SISU level is a categorical x-axis
    tick; conditions are shown side-by-side at each SISU level.

    Uses fixed perturbation amplitude EVAL_PERT_AMP.

    Args:
        loaded: Dict of condition data.
        ref_task: Reference task used for perturbed evaluations (see eval_fixed_pert).
    """
    fig = go.Figure()

    # Use string labels for the x-axis so Plotly groups the violins automatically
    sisu_labels = [str(s) for s in SISU_LEVELS]

    for cond_idx, (cond_name, d) in enumerate(loaded.items()):
        color = CONDITION_COLORS[cond_name]
        task = d["task"]
        model = d["model"]

        # Collect all endpoint errors for this condition, one entry per SISU level
        all_x: list[str] = []
        all_y: list[float] = []

        for sisu_idx, sisu_val in enumerate(SISU_LEVELS):
            key = jr.PRNGKey(55 + sisu_idx * 13 + cond_idx * 7)
            states, trial_specs = eval_fixed_pert(
                task, model, sisu_val, EVAL_PERT_AMP, key=key, ref_task=ref_task
            )

            metrics = compute_kinematics(states, trial_specs)
            endpoint_err = np.array(metrics["endpoint_error"]).ravel()  # (n_rep * n_trials,)

            sisu_label = sisu_labels[sisu_idx]
            all_x.extend([sisu_label] * len(endpoint_err))
            all_y.extend(endpoint_err.tolist())

        fig.add_trace(
            go.Violin(
                x=all_x,
                y=all_y,
                name=cond_name,
                legendgroup=cond_name,
                showlegend=True,
                box_visible=True,
                meanline_visible=True,
                points="all",
                pointpos=0,
                marker=dict(size=3, opacity=0.35),
                fillcolor=_hex_to_rgba(color, alpha=0.4),
                line_color=color,
            )
        )

    fig.update_layout(
        title=f"Endpoint error vs SISU (perturbation amp={EVAL_PERT_AMP})",
        xaxis_title="SISU",
        yaxis_title="Endpoint error (m)",
        violinmode="group",
        height=500,
        width=850,
        legend_title="Condition",
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Loss curves using feedbax.plot.loss_history
# ---------------------------------------------------------------------------


def make_fig6_all_loss_curves(loaded: dict) -> go.Figure:
    """All loss curves in a single figure with one subplot per condition.

    Arranged in a grid: up to 2 columns, as many rows as needed. Each subplot
    contains the loss-history traces for one condition.

    Uses feedbax.plot.loss_history style: log-log with error bands per term.
    """
    from feedbax.plot import loss_history as fb_loss_history

    cond_names = list(loaded.keys())
    n_conds = len(cond_names)
    n_cols = min(2, n_conds)
    n_rows = (n_conds + n_cols - 1) // n_cols

    # Build per-condition figures first, then transplant their traces into subplots
    subplot_titles = cond_names
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    for idx, (cond_name, d) in enumerate(loaded.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        history = d["history"]

        # Build a standalone figure via feedbax helper, then copy traces into grid
        sub_fig = fb_loss_history(
            history.loss,
            loss_context="training",
        )
        for trace in sub_fig.data:
            # Prevent duplicate legend entries across subplots
            trace.showlegend = (idx == 0)
            fig.add_trace(trace, row=row, col=col)

        # Mirror log-axis settings from the standalone figure
        axis_idx = (row - 1) * n_cols + col
        x_key = f"xaxis{axis_idx}" if axis_idx > 1 else "xaxis"
        y_key = f"yaxis{axis_idx}" if axis_idx > 1 else "yaxis"
        if sub_fig.layout.xaxis.type:
            fig.update_layout({x_key: dict(type=sub_fig.layout.xaxis.type)})
        if sub_fig.layout.yaxis.type:
            fig.update_layout({y_key: dict(type=sub_fig.layout.yaxis.type)})

    fig.update_layout(
        title="Training loss curves — all conditions",
        height=350 * n_rows,
        width=700 * n_cols,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading conditions...")
    loaded: dict = {}
    for display_name, model_dir_name in CONDITIONS:
        print(f"  {display_name} ({model_dir_name})...")
        result = load_condition(display_name, model_dir_name)
        if result is None:
            continue
        task, model, history, n_batches = result
        loaded[display_name] = dict(task=task, model=model, history=history, n_batches=n_batches)
        print(f"    Loaded OK — {len(loaded)} conditions so far")

    if not loaded:
        print("No conditions loaded — aborting.")
        return

    print(f"\nLoaded {len(loaded)} condition(s): {list(loaded)}\n")

    # Load the reference task from running_cost_standard (pert_std=1).  This
    # task is used for ALL perturbed evaluations so that models trained without
    # perturbations (pert_std=0, e.g. baseline_no_pert) still receive non-zero
    # gusts when we scale by pert_amp > 0.
    REF_CONDITION_NAME = "Standard (pert_std=1)"
    ref_task = loaded[REF_CONDITION_NAME]["task"] if REF_CONDITION_NAME in loaded else None
    if ref_task is None:
        print(
            f"WARNING: reference condition '{REF_CONDITION_NAME}' not loaded; "
            "perturbed evaluations will fall back to each model's own task."
        )

    # All figures use the feedbax helper to also write a tracked spec.json.
    # Bug: 0077b42 — Phase 2 completion: figure-spec wiring.
    base_inputs = [
        {"path": str(MODELS_BASE / model_dir / "config.json")}
        for _, model_dir in CONDITIONS
        if (MODELS_BASE / model_dir / "config.json").exists()
    ]
    common_kwargs = {
        "conditions": [name for name, _ in CONDITIONS],
        "sisu_levels": list(SISU_LEVELS),
        "pert_amplitudes": list(PERT_AMPLITUDES),
        "eval_sisu": EVAL_SISU,
        "eval_pert_amp": EVAL_PERT_AMP,
    }

    def _save(fig, name: str, transform_name: str) -> None:
        spec = {
            "figure_kind": transform_name,
            "inputs": base_inputs,
            "transform": [{"name": transform_name, "kwargs": {}}],
            "plot_kwargs": common_kwargs,
        }
        # Bug: f485c26 — migrated to project-config routing under issue 2ef67ca.
        out = save_figure(
            fig=fig, spec=spec,
            package="rlrmp", experiment="2ef67ca", topic=name,
            extra_packages=["rlrmp"],
        )
        print(f"  Saved render+spec: {out['render_path']}")

    # --- Figure 1: Aligned trajectories ---
    print("Fig 1: Aligned trajectories...")
    _save(
        make_fig1_aligned_trajectories(loaded, ref_task=ref_task),
        "fig1_aligned_trajectories", "make_fig1_aligned_trajectories",
    )

    # --- Figure 2: Lateral velocity profiles ---
    print("Fig 2: Lateral velocity profiles...")
    _save(
        make_fig2_lateral_velocity(loaded, ref_task=ref_task),
        "fig2_lateral_velocity", "make_fig2_lateral_velocity",
    )

    # --- Figure 3: Lateral force profiles ---
    print("Fig 3: Lateral force profiles...")
    _save(
        make_fig3_lateral_force(loaded, ref_task=ref_task),
        "fig3_lateral_force", "make_fig3_lateral_force",
    )

    # --- Figure 4: Forward velocity (no pert) ---
    # No perturbation: ref_task not needed, each model uses its own task.
    print("Fig 4: Forward velocity (no perturbation)...")
    _save(
        make_fig4_forward_velocity_no_pert(loaded),
        "fig4_forward_velocity_no_pert", "make_fig4_forward_velocity_no_pert",
    )

    # --- Figure 5: Endpoint error violin ---
    print("Fig 5: Endpoint error violin plots...")
    _save(
        make_fig5_endpoint_error_violin(loaded, ref_task=ref_task),
        "fig5_endpoint_error_violin", "make_fig5_endpoint_error_violin",
    )

    # --- Figure 6: Loss curves (all conditions in one figure) ---
    print("Fig 6: Loss curves (all conditions)...")
    _save(
        make_fig6_all_loss_curves(loaded),
        "fig6_all_loss_curves", "make_fig6_all_loss_curves",
    )

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

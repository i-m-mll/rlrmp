"""Standard analysis for the b399efc 7-cell movement-ramp matrix.

Produces six HTML topic figures via ``feedbax.plot.save_figure`` plus a
metric-table block written into ``results/b399efc/notes/matrix_results.md``.

Topics emitted:
  - ``forward_velocity_profiles``       — per-cell pooled-trial-mean forward
    velocity profile, go-cue-aligned.
  - ``hold_drift_profiles``             — per-cell pooled-trial-mean pre-go
    forward-position drift, go-cue-aligned.
  - ``peak_velocity_distributions``     — per-cell per-trial peak forward
    velocity distribution (violin).
  - ``summary_metrics``                 — 2×2 bar panel of within-cell
    vel-RMSE, peak velocity, time-to-peak after go, and hold drift.
  - ``training_loss``                   — per-cell mean training-loss curve
    (log-log) with replicate SD band.
  - ``training_loss_per_term``          — per-cell, per-term weighted training
    loss curves.

A ``variance_analysis`` auto-section is written into
``notes/matrix_results.md`` carrying the headline metric tables (within-cell
vel-RMSE, peak velocity, time-to-peak after go, hold drift, training-loss
final). Existing hand-edited preamble + Interpretation prose is preserved by
``update_marked_section``.

Library policy (Bug: 8404108):
  - Generic eval primitives are imported from ``rlrmp.eval.*``.
  - Trial-alignment helpers from ``rlrmp.analysis.trial_alignment``.
  - Notes section management from ``rlrmp.io``.
  - Shared y-axis profile grid from ``rlrmp.viz.profile_grids`` (Plotting
    Conventions in CLAUDE.md).
  - The per-trial forward-vel / forward-pos projections + hold drift remain
    inline because the standard ``rlrmp.eval.kinematics.compute_kinematics``
    helper returns endpoint_error + lateral_deviation rather than the
    profiles + go-cue-relative time-to-peak we need here.

Usage (from a worktree of rlrmp):
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync \\
        python results/b399efc/scripts/analyse_movement_ramp_matrix.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feedbax.plot import save_figure  # Bug: f485c26, feedbax 67bf476
from feedbax.train import init_task_trainer_history, TaskTrainerHistory

from rlrmp.paths import REPO_ROOT  # Bug: 8404108 — was __file__-relative
from rlrmp.io import update_marked_section
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.minimax import build_hps
from rlrmp.eval.ensemble import N_REPLICATES, eval_ensemble_on_trials
from rlrmp.eval.minimax_io import load_config, load_model
from rlrmp.analysis.trial_alignment import (
    align_trials,
    pooled_trial_mean_with_band,
    replicate_mean_curves,
)
from rlrmp.viz import profile_comparison_grid


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

EXPERIMENT = "b399efc"
N_WARMUP_BATCHES = 12000
PRE_GO_WINDOW_STEPS = 20  # 200 ms at dt=0.01

CELL_VARIANTS = [
    "movement_ramp__linear",
    "movement_ramp__cosine",
    "movement_ramp__power2",
    "movement_ramp__power4",
    "movement_ramp__power6",
    "movement_ramp__power6_prego5",
    "movement_ramp__power6_dur80",
]

CELL_DISPLAY_NAMES = {
    "movement_ramp__linear":         "linear (dur=60, prego=1)",
    "movement_ramp__cosine":         "cosine (dur=60, prego=1)",
    "movement_ramp__power2":         "power² (dur=60, prego=1)",
    "movement_ramp__power4":         "power⁴ (dur=60, prego=1)",
    "movement_ramp__power6":         "power⁶ (dur=60, prego=1)",
    "movement_ramp__power6_prego5":  "power⁶ (dur=60, prego=5)",
    "movement_ramp__power6_dur80":   "power⁶ (dur=80, prego=1)",
}

# 7 visually distinct colours, mapped per-cell so figure ↔ cell colour pairings
# stay stable across figures.
CELL_COLORS = {
    "movement_ramp__linear":         "#1f77b4",  # blue
    "movement_ramp__cosine":         "#ff7f0e",  # orange
    "movement_ramp__power2":         "#2ca02c",  # green
    "movement_ramp__power4":         "#d62728",  # red
    "movement_ramp__power6":         "#9467bd",  # purple
    "movement_ramp__power6_prego5":  "#8c564b",  # brown
    "movement_ramp__power6_dur80":   "#17becf",  # cyan
}

# Per-cell CLI arg overrides used to rebuild the hyperparameter namespace from
# the on-disk run.json. The matrix is uniform except for `movement_ramp_*`
# fields and `nn_output_pre_go`. Shared defaults are filled in by
# ``_make_args_namespace``.
CELL_EXTRA_ARGS: dict[str, dict] = {
    "movement_ramp__linear": {
        "movement_ramp_shape": "linear",
        "movement_ramp_power": 2.0,
        "movement_ramp_duration_steps": 60,
        "nn_output_pre_go": 1.0,
    },
    "movement_ramp__cosine": {
        "movement_ramp_shape": "cosine",
        "movement_ramp_power": 2.0,
        "movement_ramp_duration_steps": 60,
        "nn_output_pre_go": 1.0,
    },
    "movement_ramp__power2": {
        "movement_ramp_shape": "power",
        "movement_ramp_power": 2.0,
        "movement_ramp_duration_steps": 60,
        "nn_output_pre_go": 1.0,
    },
    "movement_ramp__power4": {
        "movement_ramp_shape": "power",
        "movement_ramp_power": 4.0,
        "movement_ramp_duration_steps": 60,
        "nn_output_pre_go": 1.0,
    },
    "movement_ramp__power6": {
        "movement_ramp_shape": "power",
        "movement_ramp_power": 6.0,
        "movement_ramp_duration_steps": 60,
        "nn_output_pre_go": 1.0,
    },
    "movement_ramp__power6_prego5": {
        "movement_ramp_shape": "power",
        "movement_ramp_power": 6.0,
        "movement_ramp_duration_steps": 60,
        "nn_output_pre_go": 5.0,
    },
    "movement_ramp__power6_dur80": {
        "movement_ramp_shape": "power",
        "movement_ramp_power": 6.0,
        "movement_ramp_duration_steps": 80,
        "nn_output_pre_go": 1.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _color_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _make_args_namespace(label: str) -> argparse.Namespace:
    """Reconstruct the argparse namespace used to train cell ``label``.

    Shared defaults mirror the b399efc RUN_PLAN; per-cell overrides
    (``movement_ramp_*`` + ``nn_output_pre_go``) come from ``CELL_EXTRA_ARGS``.
    """
    defaults = dict(
        n_warmup_batches=N_WARMUP_BATCHES,
        n_adversary_batches=0,
        batch_size=250,
        n_replicates=N_REPLICATES,
        seed=42,
        hidden_type="gru",
        sisu_gating="additive",
        loss_update_enabled=False,
        loss_update_ratio=0.5,
        # Shared loss-weight defaults for b399efc (Bug: b399efc — see RUN_PLAN).
        effector_pos_running=1.0,
        effector_hold_pos=0.0,
        effector_hold_vel=0.0,
        effector_pos_late_weight=0.0,
        effector_vel_late=0.0,
        effector_final_vel=0.0,
        effector_pos_late_final_scale=2.0,
        effector_pos_late_start_step=80,
        p_catch_trial=0.5,
        nn_output=1e-5,
        nn_hidden=1e-5,
        nn_output_jerk=0.0,
        nn_hidden_derivative=0.001,
        nn_hidden_derivative_pre_go=0.0,
        # Schedule shape: movement_ramp for running cost, flat for hold.
        effector_pos_running_schedule="movement_ramp",
        effector_hold_pos_schedule="flat",
        position_powerlaw_power=6.0,
        controller_lr=1e-4,
    )
    defaults.update(CELL_EXTRA_ARGS[label])
    return argparse.Namespace(**defaults)


def _count_replicates_in_model(model, *, expected: int = N_REPLICATES) -> int:
    """Confirm replicate-axis size by checking that ``expected`` shows up.

    The first array-leaf in a model often has a small leading axis that is NOT
    the ensemble dim (e.g. a per-trial 2-vector). Using the expected replicate
    count and verifying multiple leaves match is more robust than guessing from
    the first leaf.

    Returns the most common leading-axis size that matches ``expected``; falls
    back to 1 if no batched leaves are found.
    """
    import jax.tree as jt
    matches = 0
    total = 0
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim >= 1:
            total += 1
            if leaf.shape[0] == expected:
                matches += 1
    if matches > 0:
        return expected
    return 1


# ---------------------------------------------------------------------------
# Model / history loading
# ---------------------------------------------------------------------------


def load_cell_model(label: str, artifact_base: Path):
    """Load warmup_model.eqx for ``label`` using the minimax IO loader.

    For b399efc cells, ``n_adversary_batches=0`` so the canonical final model
    is the warmup model.

    Returns:
        (model, task, hps, n_reps)
    """
    cell_dir = artifact_base / label
    warmup_path = cell_dir / "warmup_model.eqx"
    if not warmup_path.exists():
        raise FileNotFoundError(f"warmup_model.eqx not found in {cell_dir}")

    args = _make_args_namespace(label)
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(args.seed))
    task = pair.task

    # b399efc run dirs lack config.json (the prior post-training agent did not
    # persist it). ``load_model`` only uses ``config`` for forward-compat;
    # passing an empty dict is safe per the docstring.
    cfg: dict = {}
    try:
        cfg = load_config(cell_dir)
    except FileNotFoundError:
        cfg = {}

    model = load_model(cell_dir, "warmup_model.eqx", hps, cfg)
    if model is None:  # defensive — load_model returns None only on missing file
        raise FileNotFoundError(f"load_model returned None for {warmup_path}")

    n_reps = _count_replicates_in_model(model)
    return model, task, hps, n_reps


def load_cell_history(label: str, artifact_base: Path, hps) -> TaskTrainerHistory | None:
    """Load warmup_history.eqx for ``label`` by building the proper skeleton.

    Returns ``None`` if the history file is absent or fails to deserialise.
    """
    cell_dir = artifact_base / label
    history_path = cell_dir / "warmup_history.eqx"
    if not history_path.exists():
        return None

    pair = setup_task_model_pair(hps, key=jr.PRNGKey(_make_args_namespace(label).seed))
    skeleton = init_task_trainer_history(
        loss_func=pair.task.loss_func,
        n_batches=N_WARMUP_BATCHES,
        n_replicates=N_REPLICATES,
        ensembled=True,
    )
    try:
        with open(history_path, "rb") as f:
            f.readline()  # skip JSON-hyperparameters header line
            history = eqx.tree_deserialise_leaves(f, skeleton)
        return history
    except Exception as e:
        print(f"  WARNING: failed to load history for {label}: {e}")
        return None


# ---------------------------------------------------------------------------
# Trial building + evaluation
# ---------------------------------------------------------------------------


def build_clean_trials(task, *, sisu: float = 0.5, eval_key: jax.Array | None = None):
    """Construct zero-perturbation validation trials with a fixed SISU value.

    Uses the task's bound ``validation_trials`` (per the part2 module wiring;
    ``setup_task_model_pair`` returns the centerout-delayed-reach validation
    set with `eval_n_directions=8`).
    """
    val = task.validation_trials
    n_trials = val.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val,
        jnp.zeros((n_trials,)),
    )
    trial_specs = eqx.tree_at(
        lambda t: t.inputs["sisu"],
        trial_specs,
        jnp.full((n_trials,), sisu),
    )
    return trial_specs


# ---------------------------------------------------------------------------
# Kinematics (per-trial forward-axis projections + scalar summaries)
# ---------------------------------------------------------------------------


def compute_kinematics_per_replicate(states, trial_specs) -> dict[str, np.ndarray]:
    """Compute per-replicate kinematic metrics for the matrix analyses.

    Returns:
        Dict with the following keys:
          - ``forward_vel_profile``      : (n_rep, n_trials, n_steps)
          - ``pos_forward_profile``      : (n_rep, n_trials, n_steps)
          - ``peak_forward_velocity``    : (n_rep, n_trials) — max post-go
            velocity along the reach axis.
          - ``time_to_peak_after_go``    : (n_rep, n_trials) — steps after go
            cue at which the peak forward velocity is reached.
          - ``hold_drift_mm``            : (n_rep, n_trials) — max forward
            displacement (mm) over the full pre-go period.
          - ``go_idx``                   : (n_trials,) — per-trial go-cue index.
    """
    pos = states.mechanics.effector.pos  # (n_rep, n_trials, n_steps, 2)
    vel = states.mechanics.effector.vel

    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value  # (n_trials, n_steps, 2)
    goal = goal_seq[:, -1, :]

    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,)

    n_rep, n_trials, n_steps, _ = pos.shape
    t_arr = jnp.arange(n_steps)
    after_go = t_arr[None, None, :] >= go_idx[None, :, None]
    before_go = t_arr[None, None, :] < go_idx[None, :, None]

    def _pos_at_go(pos_rep, go_arr):
        return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_arr)

    pos_at_go = jax.vmap(_pos_at_go, in_axes=(0, None))(pos, go_idx)
    direction = goal[None, :, :] - pos_at_go
    d_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    d_unit = direction / jnp.maximum(d_norm, 1e-12)

    v_fwd = jnp.sum(vel * d_unit[:, :, None, :], axis=-1)  # signed projection
    v_fwd_post_go = jnp.where(after_go, v_fwd, 0.0)
    peak_fwd = jnp.max(v_fwd_post_go, axis=-1)
    abs_argmax = jnp.argmax(v_fwd_post_go, axis=-1)
    time_to_peak_after_go = jnp.maximum(abs_argmax - go_idx[None, :], 0)

    pos_at_start = pos[:, :, 0, :]
    pos_rel = pos - pos_at_start[:, :, None, :]
    pos_fwd = jnp.sum(pos_rel * d_unit[:, :, None, :], axis=-1)

    pos_fwd_pre_go = jnp.where(before_go, pos_fwd, -jnp.inf)
    hold_drift_m = jnp.max(pos_fwd_pre_go, axis=-1)
    hold_drift_m = jnp.where(jnp.isinf(hold_drift_m), 0.0, hold_drift_m)
    hold_drift_mm = hold_drift_m * 1000.0

    return {
        "forward_vel_profile": np.array(v_fwd),
        "pos_forward_profile": np.array(pos_fwd),
        "peak_forward_velocity": np.array(peak_fwd),
        "time_to_peak_after_go": np.array(time_to_peak_after_go),
        "hold_drift_mm": np.array(hold_drift_mm),
        "go_idx": np.array(go_idx),
    }


def _within_cell_mean_pairwise_rmse(profiles: np.ndarray) -> float:
    """Mean pairwise RMSE between all distinct replicate pairs (nan-tolerant)."""
    n_rep = profiles.shape[0]
    rmse_vals = []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            diff = profiles[i] - profiles[j]
            rmse_vals.append(float(np.sqrt(np.nanmean(diff ** 2))))
    return float(np.mean(rmse_vals)) if rmse_vals else float("nan")


def compute_cell_stats(km: dict[str, np.ndarray]) -> dict:
    """Aggregate per-replicate kinematic outputs into per-cell summaries."""
    peak_vel_per_rep = km["peak_forward_velocity"].mean(axis=-1)
    hold_drift_per_rep = km["hold_drift_mm"].mean(axis=-1)
    ttp_per_rep = km["time_to_peak_after_go"].mean(axis=-1)

    # Within-cell pairwise velocity-RMSE on go-aligned per-replicate curves.
    # Bug: 06f7faf — align then trim before reducing across replicates.
    aligned_vel, _center = align_trials(km["forward_vel_profile"], km["go_idx"])
    vel_profiles, _sl = replicate_mean_curves(aligned_vel)  # (n_rep, n_kept_steps)
    within_rmse_vel = _within_cell_mean_pairwise_rmse(vel_profiles)

    def _scalar_sd(arr: np.ndarray) -> float:
        return float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return {
        "peak_vel_per_rep": peak_vel_per_rep.tolist(),
        "hold_drift_per_rep": hold_drift_per_rep.tolist(),
        "time_to_peak_per_rep": ttp_per_rep.tolist(),
        "mean_peak_velocity": float(peak_vel_per_rep.mean()),
        "sd_peak_velocity": _scalar_sd(peak_vel_per_rep),
        "mean_hold_drift_mm": float(hold_drift_per_rep.mean()),
        "sd_hold_drift_mm": _scalar_sd(hold_drift_per_rep),
        "mean_time_to_peak_steps": float(ttp_per_rep.mean()),
        "sd_time_to_peak_steps": _scalar_sd(ttp_per_rep),
        "within_cell_vel_rmse": within_rmse_vel,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_forward_velocity_profile_figure(
    cell_kms: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Per-cell pooled-trial-mean forward velocity profile, go-cue-aligned."""
    labels_present = [l for l in CELL_VARIANTS if l in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()

    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        vertical_spacing=0.025,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        v_fwd = km["forward_vel_profile"]
        go_idx = km["go_idx"]
        color = CELL_COLORS[label]

        aligned_v, center = align_trials(v_fwd, go_idx)
        mean, lower, upper, sl = pooled_trial_mean_with_band(aligned_v, band="sd")
        t = ((np.arange(aligned_v.shape[-1]) - center) * dt)[sl]

        # SD band (upper invisible, lower filled to upper)
        fig.add_trace(go.Scatter(
            x=t, y=upper, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=lower, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor=_color_rgba(color, 0.25),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=mean, mode="lines",
            line=dict(color=color, width=2),
            name=CELL_DISPLAY_NAMES[label], showlegend=False,
        ), row=row, col=1)

        fig.add_vline(
            x=0.0,
            line=dict(color="black", dash="dash", width=1),
            row=row, col=1,
        )

    fig.update_layout(
        title=(
            "Forward velocity profiles (go-cue-aligned, pooled trial mean ± SD) — "
            "7-cell movement-ramp matrix (b399efc)"
        ),
        width=1000,
        height=170 * n_cells + 100,
        margin=dict(l=70, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text="Fwd vel (m/s)", row=row, col=1)

    return fig


def make_hold_drift_figure(
    cell_kms: dict[str, dict],
    dt: float = 0.01,
) -> go.Figure:
    """Per-cell pooled-trial-mean pre-go forward-position drift."""
    labels_present = [l for l in CELL_VARIANTS if l in cell_kms]
    n_cells = len(labels_present)
    if n_cells == 0:
        return go.Figure()

    fig = profile_comparison_grid(
        n_panels=n_cells,
        subplot_titles=[CELL_DISPLAY_NAMES[l] for l in labels_present],
        vertical_spacing=0.025,
    )

    for row, label in enumerate(labels_present, start=1):
        km = cell_kms[label]
        pos_fwd = km["pos_forward_profile"]
        go_idx = km["go_idx"]
        color = CELL_COLORS[label]

        aligned_pos, center = align_trials(pos_fwd, go_idx)
        mean_m, lower_m, upper_m, sl = pooled_trial_mean_with_band(aligned_pos, band="sd")
        mean = mean_m * 1000.0
        lower = lower_m * 1000.0
        upper = upper_m * 1000.0

        t_rel = ((np.arange(aligned_pos.shape[-1]) - center) * dt)[sl]
        keep = t_rel <= 0.0
        t_pre = t_rel[keep]
        mean = mean[keep]
        lower = lower[keep]
        upper = upper[keep]

        fig.add_trace(go.Scatter(
            x=t_pre, y=upper, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t_pre, y=lower, mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty", fillcolor=_color_rgba(color, 0.25),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t_pre, y=mean, mode="lines",
            line=dict(color=color, width=2),
            name=CELL_DISPLAY_NAMES[label], showlegend=False,
        ), row=row, col=1)

        fig.add_hline(y=0, line=dict(color="grey", dash="dot", width=1), row=row, col=1)

    fig.update_layout(
        title=(
            "Pre-go forward position drift (go-cue-aligned, pooled trial mean ± SD) — "
            "7-cell movement-ramp matrix (b399efc)"
        ),
        width=1000,
        height=170 * n_cells + 100,
        margin=dict(l=70, r=60, t=80, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=n_cells, col=1)
    for row in range(1, n_cells + 1):
        fig.update_yaxes(title_text="Fwd pos (mm)", row=row, col=1)

    return fig


def make_peak_velocity_figure(cell_stats: dict[str, dict]) -> go.Figure:
    """Per-cell per-replicate peak forward velocity (violin + points)."""
    fig = go.Figure()
    for label in CELL_VARIANTS:
        if label not in cell_stats:
            continue
        pvs = cell_stats[label]["peak_vel_per_rep"]
        color = CELL_COLORS[label]
        fig.add_trace(go.Violin(
            y=pvs,
            name=CELL_DISPLAY_NAMES[label],
            box_visible=True,
            meanline_visible=True,
            points="all",
            jitter=0.3,
            pointpos=-1.5,
            line_color=color,
            fillcolor=_color_rgba(color, 0.35),
            marker=dict(color=color, size=8),
            showlegend=False,
        ))

    fig.update_layout(
        title=(
            "Peak forward velocity per replicate — "
            "7-cell movement-ramp matrix (b399efc)"
        ),
        yaxis_title="Peak forward velocity (m/s)",
        xaxis_title="Cell",
        width=1200,
        height=550,
        margin=dict(l=70, r=40, t=80, b=180),
    )
    fig.update_xaxes(tickangle=-25)
    return fig


def make_summary_metrics_figure(cell_stats: dict[str, dict]) -> go.Figure:
    """2×2 panel of headline scalar metrics per cell.

    Panels:
      (1,1) Within-cell pairwise velocity-RMSE (m/s)
      (1,2) Peak forward velocity (m/s, mean ± SD)
      (2,1) Time-to-peak after go cue (steps, mean ± SD)
      (2,2) Hold-period peak drift (mm, mean ± SD)
    """
    labels_present = [l for l in CELL_VARIANTS if l in cell_stats]
    display_names = [CELL_DISPLAY_NAMES[l] for l in labels_present]
    colors = [CELL_COLORS[l] for l in labels_present]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Within-cell pairwise velocity-RMSE (m/s)",
            "Peak forward velocity (m/s, mean ± SD)",
            "Time-to-peak after go (steps, mean ± SD)",
            "Hold-period peak drift (mm, mean ± SD)",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    vel_rmse = [cell_stats[l]["within_cell_vel_rmse"] for l in labels_present]
    peak_vel = [cell_stats[l]["mean_peak_velocity"] for l in labels_present]
    peak_vel_sd = [cell_stats[l]["sd_peak_velocity"] for l in labels_present]
    ttp = [cell_stats[l]["mean_time_to_peak_steps"] for l in labels_present]
    ttp_sd = [cell_stats[l]["sd_time_to_peak_steps"] for l in labels_present]
    hold = [cell_stats[l]["mean_hold_drift_mm"] for l in labels_present]
    hold_sd = [cell_stats[l]["sd_hold_drift_mm"] for l in labels_present]

    fig.add_trace(go.Bar(
        x=display_names, y=vel_rmse, marker_color=colors,
        text=[f"{v:.4f}" for v in vel_rmse], textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=display_names, y=peak_vel,
        error_y=dict(type="data", array=peak_vel_sd),
        marker_color=colors,
        text=[f"{v:.3f}" for v in peak_vel], textposition="outside",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=display_names, y=ttp,
        error_y=dict(type="data", array=ttp_sd),
        marker_color=colors,
        text=[f"{v:.1f}" for v in ttp], textposition="outside",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=display_names, y=hold,
        error_y=dict(type="data", array=hold_sd),
        marker_color=colors,
        text=[f"{v:.2f}" for v in hold], textposition="outside",
    ), row=2, col=2)

    fig.update_layout(
        title=(
            "Summary metrics per cell — 7-cell movement-ramp matrix (b399efc)<br>"
            "<sup>All quantities absolute (not ratios). Within-cell vel-RMSE on go-aligned per-rep curves.</sup>"
        ),
        width=1500,
        height=950,
        showlegend=False,
        margin=dict(l=70, r=40, t=100, b=200),
    )
    fig.update_yaxes(title_text="Vel-RMSE (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Peak vel (m/s)", row=1, col=2)
    fig.update_yaxes(title_text="Time-to-peak (steps)", row=2, col=1)
    fig.update_yaxes(title_text="Hold drift (mm)", row=2, col=2)
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(tickangle=-30, row=r, col=c)

    return fig


def _add_band_traces(
    fig: go.Figure,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    display_name: str,
    legendgroup: str,
    *,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
) -> None:
    """Add a (mean, mean+std, mean-std) band trio. Used for log-y plots."""
    upper = mean + std
    lower = np.maximum(mean - std, 1e-15)
    kw: dict = {}
    if row is not None and col is not None:
        kw = dict(row=row, col=col)

    fig.add_trace(go.Scatter(
        name=display_name, x=x, y=mean, mode="lines",
        line=dict(color=color, width=2),
        legendgroup=legendgroup, showlegend=show_legend,
    ), **kw)
    fig.add_trace(go.Scatter(
        name=f"{display_name}+sd", x=x, y=upper, mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
        legendgroup=legendgroup,
    ), **kw)
    fig.add_trace(go.Scatter(
        name=f"{display_name}-sd", x=x, y=lower, mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor=_color_rgba(color, 0.2),
        hoverinfo="skip", showlegend=False,
        legendgroup=legendgroup,
    ), **kw)


def make_training_loss_figure(histories: dict[str, TaskTrainerHistory]) -> tuple[go.Figure, dict]:
    """Total weighted training-loss curve per cell, mean ± SD over replicates."""
    fig = go.Figure()
    end_stats: dict[str, dict] = {}

    for label in CELL_VARIANTS:
        if label not in histories:
            continue
        total = np.array(histories[label].loss.aggregate(leaf_fn=lambda x: x))
        n_batches = total.shape[0]
        x = np.arange(n_batches)
        mean = total.mean(axis=1)
        std = total.std(axis=1)

        end_stats[label] = {
            "final_mean": float(mean[-1]),
            "final_std": float(std[-1]),
            "mean_last_100": float(mean[-100:].mean()),
            "std_last_100": float(std[-100:].mean()),
            "n_replicates": int(total.shape[1]),
        }

        _add_band_traces(
            fig, x, mean, std,
            color=CELL_COLORS[label],
            display_name=CELL_DISPLAY_NAMES[label],
            legendgroup=label,
        )

    fig.update_layout(
        title="Total weighted training loss vs batch — 7-cell movement-ramp matrix (b399efc)",
        xaxis_title="Training batch",
        yaxis_title="Total weighted loss",
        width=1000,
        height=500,
        legend=dict(orientation="v", x=1.0, xanchor="left"),
        yaxis_type="log",
        xaxis_type="log",
        margin=dict(l=70, r=240, t=70, b=60),
        hovermode="x",
    )
    return fig, end_stats


def make_training_loss_per_term_figure(histories: dict[str, TaskTrainerHistory]) -> go.Figure:
    """Per-term weighted training-loss decomposition for all cells."""
    all_term_keys: set[str] = set()
    term_data: dict[str, dict[str, np.ndarray]] = {}
    for label, history in histories.items():
        flat = history.loss.flatten(apply_weights=True)
        term_data[label] = {k: np.array(v) for k, v in flat.items()}
        all_term_keys.update(term_data[label].keys())

    term_keys = sorted(all_term_keys)
    n_terms = len(term_keys)
    if n_terms == 0:
        return go.Figure()

    n_cols = min(4, n_terms)
    n_rows = (n_terms + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=term_keys,
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.07,
    )

    for term_idx, term_key in enumerate(term_keys):
        row = term_idx // n_cols + 1
        col = term_idx % n_cols + 1
        for label in CELL_VARIANTS:
            if label not in term_data or term_key not in term_data[label]:
                continue
            term_vals = term_data[label][term_key]  # (n_batches, n_replicates)
            n_batches = term_vals.shape[0]
            x = np.arange(n_batches)
            mean = term_vals.mean(axis=1)
            mean = np.where(mean > 0, mean, 1e-30)  # log floor
            color = CELL_COLORS[label]
            show_legend = (term_idx == 0)
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines",
                name=CELL_DISPLAY_NAMES[label],
                line=dict(color=color, width=1.5),
                legendgroup=label,
                showlegend=show_legend,
            ), row=row, col=col)

    fig.update_layout(
        title="Per-term weighted training loss vs batch — 7-cell movement-ramp matrix (b399efc)",
        width=320 * n_cols + 240,
        height=290 * n_rows + 120,
        margin=dict(l=60, r=240, t=100, b=60),
    )
    for i in range(1, n_terms + 1):
        r = (i - 1) // n_cols + 1
        c = (i - 1) % n_cols + 1
        fig.update_yaxes(type="log", row=r, col=c)
        fig.update_xaxes(type="log", row=r, col=c)

    return fig


# ---------------------------------------------------------------------------
# Notes-table emission
# ---------------------------------------------------------------------------


def _format_metric_tables(
    cell_stats: dict[str, dict],
    train_loss_end_stats: dict[str, dict],
) -> str:
    """Markdown block (tables only) for the ``variance_analysis`` auto-section.

    No prose — the matrix_results.md file already carries the human-written
    Interpretation. The auto-section is reserved for the headline metric
    tables so re-running the script does not stomp on hand-edited content.
    """
    lines: list[str] = [
        f"## Headline metrics (auto-generated, b399efc)",
        "",
        "All quantities computed on `warmup_model.eqx` for each cell, evaluated on "
        "8-direction center-out validation trials at SISU=0.5 with zero perturbation.",
        "",
        "### Headline scalar metrics",
        "",
        "| Cell | Within-cell vel-RMSE (m/s) | Peak vel (m/s) | Time-to-peak (steps) | Hold drift (mm) | Training loss (final) |",
        "|------|---:|---:|---:|---:|---:|",
    ]
    for label in CELL_VARIANTS:
        if label not in cell_stats:
            lines.append(f"| {CELL_DISPLAY_NAMES[label]} | n/a | n/a | n/a | n/a | n/a |")
            continue
        s = cell_stats[label]
        loss = train_loss_end_stats.get(label, {})
        loss_str = (
            f"{loss['final_mean']:.3e} ± {loss['final_std']:.2e}"
            if loss else "n/a"
        )
        lines.append(
            f"| {CELL_DISPLAY_NAMES[label]} "
            f"| {s['within_cell_vel_rmse']:.4f} "
            f"| {s['mean_peak_velocity']:.3f} ± {s['sd_peak_velocity']:.3f} "
            f"| {s['mean_time_to_peak_steps']:.1f} ± {s['sd_time_to_peak_steps']:.1f} "
            f"| {s['mean_hold_drift_mm']:.2f} ± {s['sd_hold_drift_mm']:.2f} "
            f"| {loss_str} |"
        )

    lines += [
        "",
        "### Per-replicate peak forward velocity",
        "",
        "| Cell | Rep 0 | Rep 1 | Rep 2 | Rep 3 | Rep 4 |",
        "|------|---:|---:|---:|---:|---:|",
    ]
    for label in CELL_VARIANTS:
        if label not in cell_stats:
            continue
        pvs = cell_stats[label]["peak_vel_per_rep"]
        cells_str = " | ".join(f"{v:.3f}" for v in pvs)
        # Pad to 5 columns if fewer reps present.
        if len(pvs) < 5:
            cells_str += " | " + " | ".join(["n/a"] * (5 - len(pvs)))
        lines.append(f"| {CELL_DISPLAY_NAMES[label]} | {cells_str} |")

    lines += [
        "",
        "### Per-replicate hold drift (mm)",
        "",
        "| Cell | Rep 0 | Rep 1 | Rep 2 | Rep 3 | Rep 4 |",
        "|------|---:|---:|---:|---:|---:|",
    ]
    for label in CELL_VARIANTS:
        if label not in cell_stats:
            continue
        hd = cell_stats[label]["hold_drift_per_rep"]
        cells_str = " | ".join(f"{v:.3f}" for v in hd)
        if len(hd) < 5:
            cells_str += " | " + " | ".join(["n/a"] * (5 - len(hd)))
        lines.append(f"| {CELL_DISPLAY_NAMES[label]} | {cells_str} |")

    lines += [
        "",
        "### Figures",
        "",
        f"- `results/{EXPERIMENT}/figures/forward_velocity_profiles/` — forward velocity per cell, go-cue-aligned",
        f"- `results/{EXPERIMENT}/figures/hold_drift_profiles/` — pre-go forward position per cell, go-cue-aligned",
        f"- `results/{EXPERIMENT}/figures/peak_velocity_distributions/` — per-replicate peak velocity (violin)",
        f"- `results/{EXPERIMENT}/figures/summary_metrics/` — 2×2 scalar-metric bar panel",
        f"- `results/{EXPERIMENT}/figures/training_loss/` — total weighted training loss per cell",
        f"- `results/{EXPERIMENT}/figures/training_loss_per_term/` — per-term decomposition per cell",
        "",
        "HTML renders in `_artifacts/b399efc/figures/<topic>/figure.html`.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=None,
        help="Base directory containing cell subdirs (default: <repo>/_artifacts/b399efc/runs)",
    )
    parser.add_argument("--sisu", type=float, default=0.5)
    parser.add_argument("--eval-seed", type=int, default=42)
    args = parser.parse_args()

    artifact_base = args.artifact_base or (REPO_ROOT / "_artifacts" / EXPERIMENT / "runs")
    results_base = REPO_ROOT / "results" / EXPERIMENT

    print(f"Artifact base: {artifact_base}")
    print(f"Results base:  {results_base}")

    notes_dir = results_base / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load + evaluate each cell
    # -----------------------------------------------------------------------
    cell_stats: dict[str, dict] = {}
    cell_kms: dict[str, dict] = {}
    histories: dict[str, TaskTrainerHistory] = {}
    input_artifacts: list[dict] = []

    for label in CELL_VARIANTS:
        print(f"\n[{label}] Loading model ...", flush=True)
        try:
            model, task, hps, n_reps = load_cell_model(label, artifact_base)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        except Exception as e:
            import traceback
            print(f"  FAILED loading model: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        print(f"  Loaded. n_replicates={n_reps}")
        warmup_path = artifact_base / label / "warmup_model.eqx"
        input_artifacts.append({"path": str(warmup_path), "role": f"warmup_model:{label}"})

        # Eval clean trials
        try:
            trial_specs = build_clean_trials(task, sisu=args.sisu)
            n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
            print(f"  Evaluating {n_reps} replicates on {n_trials} trials ...", flush=True)
            states = eval_ensemble_on_trials(
                task, model, trial_specs,
                key=jr.PRNGKey(args.eval_seed),
                n_replicates=n_reps,
            )
            print("  Eval OK. Computing kinematics ...", flush=True)
            km = compute_kinematics_per_replicate(states, trial_specs)
            stats = compute_cell_stats(km)
            cell_kms[label] = km
            cell_stats[label] = stats
            print(
                f"  peak_vel={stats['mean_peak_velocity']:.4f} ± {stats['sd_peak_velocity']:.4f} m/s  "
                f"hold_drift={stats['mean_hold_drift_mm']:.2f} ± {stats['sd_hold_drift_mm']:.2f} mm  "
                f"vel_rmse_within={stats['within_cell_vel_rmse']:.4f} m/s"
            )
        except Exception as e:
            import traceback
            print(f"  FAILED eval/kinematics: {type(e).__name__}: {e}")
            traceback.print_exc()

        # Load training history (independent of eval — proceed even if eval failed)
        history = load_cell_history(label, artifact_base, hps)
        if history is not None:
            histories[label] = history
            history_path = artifact_base / label / "warmup_history.eqx"
            input_artifacts.append({"path": str(history_path), "role": f"warmup_history:{label}"})

    if not cell_stats and not histories:
        print("\nNo cells loaded — aborting.")
        return

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\n--- Building figures ---")

    spec_base = {
        "experiment": EXPERIMENT,
        "inputs": input_artifacts,
    }

    # 1. forward_velocity_profiles
    if cell_kms:
        fig_fv = make_forward_velocity_profile_figure(cell_kms)
        spec_fv = {
            **spec_base,
            "figure_kind": "forward_velocity_profile_time_series_go_aligned",
            "transform": [
                {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
                {"name": "forward_velocity_projection_onto_reach_axis", "kwargs": {}},
                {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
                {"name": "trim_to_full_support", "kwargs": {"min_coverage": 1.0}},
                {"name": "pooled_trial_mean_with_band", "kwargs": {"band": "sd"}},
            ],
            "plot_kwargs": {
                "cells": CELL_VARIANTS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
                "dt": 0.01,
                "alignment": "go_cue_per_trial",
                "shared_yaxes": "all",
            },
        }
        out = save_figure(
            fig=fig_fv, spec=spec_fv,
            package="rlrmp", experiment=EXPERIMENT, topic="forward_velocity_profiles",
            extra_packages=["rlrmp"],
        )
        print(f"  forward_velocity_profiles: {out['render_path']}")

    # 2. hold_drift_profiles
    if cell_kms:
        fig_hd = make_hold_drift_figure(cell_kms)
        spec_hd = {
            **spec_base,
            "figure_kind": "hold_drift_profile_pre_go_position_go_aligned",
            "transform": [
                {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
                {"name": "forward_position_projection_onto_reach_axis", "kwargs": {}},
                {"name": "align_trials_to_go_cue", "kwargs": {"pad": "nan"}},
                {"name": "trim_to_full_support", "kwargs": {"min_coverage": 1.0}},
                {"name": "pooled_trial_mean_with_band", "kwargs": {"band": "sd"}},
                {"name": "clip_to_pre_go_window", "kwargs": {}},
            ],
            "plot_kwargs": {
                "cells": CELL_VARIANTS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
                "dt": 0.01,
                "alignment": "go_cue_per_trial",
                "shared_yaxes": "all",
            },
        }
        out = save_figure(
            fig=fig_hd, spec=spec_hd,
            package="rlrmp", experiment=EXPERIMENT, topic="hold_drift_profiles",
            extra_packages=["rlrmp"],
        )
        print(f"  hold_drift_profiles: {out['render_path']}")

    # 3. peak_velocity_distributions
    if cell_stats:
        fig_pv = make_peak_velocity_figure(cell_stats)
        spec_pv = {
            **spec_base,
            "figure_kind": "peak_velocity_distributions_violin",
            "transform": [
                {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
                {"name": "compute_peak_forward_velocity", "kwargs": {}},
                {"name": "mean_over_trials_per_replicate", "kwargs": {}},
            ],
            "plot_kwargs": {
                "cells": CELL_VARIANTS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
            },
            "cell_stats": {
                label: {
                    "mean_peak_velocity": s["mean_peak_velocity"],
                    "sd_peak_velocity": s["sd_peak_velocity"],
                    "peak_vel_per_rep": s["peak_vel_per_rep"],
                }
                for label, s in cell_stats.items()
            },
        }
        out = save_figure(
            fig=fig_pv, spec=spec_pv,
            package="rlrmp", experiment=EXPERIMENT, topic="peak_velocity_distributions",
            extra_packages=["rlrmp"],
        )
        print(f"  peak_velocity_distributions: {out['render_path']}")

    # 4. summary_metrics
    if cell_stats:
        fig_sm = make_summary_metrics_figure(cell_stats)
        spec_sm = {
            **spec_base,
            "figure_kind": "summary_metrics_bar_panel",
            "transform": [
                {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
                {"name": "compute_cell_stats", "kwargs": {}},
            ],
            "plot_kwargs": {
                "cells": CELL_VARIANTS,
                "n_replicates": N_REPLICATES,
                "sisu": args.sisu,
                "pert_scale": 0.0,
                "metrics": [
                    "within_cell_vel_rmse",
                    "mean_peak_velocity",
                    "mean_time_to_peak_steps",
                    "mean_hold_drift_mm",
                ],
            },
            "cell_stats": {
                label: {
                    "within_cell_vel_rmse": s["within_cell_vel_rmse"],
                    "mean_peak_velocity": s["mean_peak_velocity"],
                    "sd_peak_velocity": s["sd_peak_velocity"],
                    "mean_time_to_peak_steps": s["mean_time_to_peak_steps"],
                    "sd_time_to_peak_steps": s["sd_time_to_peak_steps"],
                    "mean_hold_drift_mm": s["mean_hold_drift_mm"],
                    "sd_hold_drift_mm": s["sd_hold_drift_mm"],
                }
                for label, s in cell_stats.items()
            },
        }
        out = save_figure(
            fig=fig_sm, spec=spec_sm,
            package="rlrmp", experiment=EXPERIMENT, topic="summary_metrics",
            extra_packages=["rlrmp"],
        )
        print(f"  summary_metrics: {out['render_path']}")

    # 5. training_loss
    train_loss_end_stats: dict[str, dict] = {}
    if histories:
        fig_tl, train_loss_end_stats = make_training_loss_figure(histories)
        spec_tl = {
            **spec_base,
            "figure_kind": "training_loss_total_multiline_errorbands",
            "transform": [
                {"name": "load_warmup_history_fbx", "kwargs": {"header_lines": 1}},
                {"name": "TermTree.aggregate", "kwargs": {"leaf_fn": "identity"}},
                {"name": "mean_std_across_replicates", "kwargs": {"axis": 1}},
            ],
            "plot_kwargs": {
                "cells": CELL_VARIANTS,
                "n_replicates": N_REPLICATES,
                "n_warmup_batches": N_WARMUP_BATCHES,
                "error_band": "mean ± 1sd",
                "yaxis_type": "log",
                "xaxis_type": "log",
            },
            "end_of_training_stats": train_loss_end_stats,
        }
        out = save_figure(
            fig=fig_tl, spec=spec_tl,
            package="rlrmp", experiment=EXPERIMENT, topic="training_loss",
            extra_packages=["rlrmp"],
        )
        print(f"  training_loss: {out['render_path']}")

    # 6. training_loss_per_term
    if histories:
        fig_pt = make_training_loss_per_term_figure(histories)
        spec_pt = {
            **spec_base,
            "figure_kind": "training_loss_per_term_multiline",
            "transform": [
                {"name": "load_warmup_history_fbx", "kwargs": {"header_lines": 1}},
                {"name": "TermTree.flatten", "kwargs": {"apply_weights": True}},
                {"name": "mean_across_replicates", "kwargs": {"axis": 1}},
            ],
            "plot_kwargs": {
                "cells": CELL_VARIANTS,
                "n_replicates": N_REPLICATES,
                "n_warmup_batches": N_WARMUP_BATCHES,
            },
        }
        out = save_figure(
            fig=fig_pt, spec=spec_pt,
            package="rlrmp", experiment=EXPERIMENT, topic="training_loss_per_term",
            extra_packages=["rlrmp"],
        )
        print(f"  training_loss_per_term: {out['render_path']}")

    # -----------------------------------------------------------------------
    # Notes file — auto-generated metric tables
    # -----------------------------------------------------------------------
    notes_path = notes_dir / "matrix_results.md"
    block = _format_metric_tables(cell_stats, train_loss_end_stats)
    update_marked_section(notes_path, "variance_analysis", block)
    print(f"\nUpdated notes section: {notes_path} (marker=variance_analysis)")

    # Save per-cell stats JSON for downstream/follow-up scripts.
    stats_json_path = notes_dir / "matrix_results_data.json"
    json_data = {
        "experiment": EXPERIMENT,
        "sisu": args.sisu,
        "cells": cell_stats,
        "training_loss_end_stats": train_loss_end_stats,
    }
    with open(stats_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved stats JSON: {stats_json_path}")

    # -----------------------------------------------------------------------
    # Summary print
    # -----------------------------------------------------------------------
    print("\n=== b399efc matrix summary ===")
    header = (
        f"{'Cell':36s} {'Vel-RMSE (m/s)':>15} {'Peak vel (m/s)':>15} "
        f"{'TTP (steps)':>12} {'Hold drift (mm)':>18}"
    )
    print(header)
    print("-" * len(header))
    for label in CELL_VARIANTS:
        if label not in cell_stats:
            print(f"  {CELL_DISPLAY_NAMES[label]:34s} SKIPPED")
            continue
        s = cell_stats[label]
        print(
            f"  {CELL_DISPLAY_NAMES[label]:34s} "
            f"{s['within_cell_vel_rmse']:>15.4f} "
            f"{s['mean_peak_velocity']:>15.3f} "
            f"{s['mean_time_to_peak_steps']:>12.1f} "
            f"{s['mean_hold_drift_mm']:>18.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

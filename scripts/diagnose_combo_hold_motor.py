"""Diagnostic: hold-period motor output in the combo cell.

Loads the combo cell (gru__jerk_motor_smooth_combo, adversarial_model.eqx),
runs inference on catch and go trials, and inspects per-step motor output
magnitude during hold vs movement epochs.

Key question (from _artifacts/scratchpad/residual_anticipation_proposal.md §5):
Is the 4.6 mm residual hold-period drift in the combo cell mechanically driven
(non-zero motor commands escaping the pre-go penalty) or residual plant mechanics
(near-zero commands but effector still drifts)?

Output:
  - HTML figure: time vs motor output magnitude, one trace per replicate,
    vertical line at go cue.
    Path: _artifacts/2bc95fd/figures/combo_hold_motor_diagnostic/figure.html
  - Spec: results/2bc95fd/figures/combo_hold_motor_diagnostic/spec.json
  - Notes: results/2bc95fd/notes/combo_hold_motor_diagnostic.md

Usage (from repo root):
    /path/to/.venv/bin/python scripts/diagnose_combo_hold_motor.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feedbax._io import load_with_hyperparameters
from feedbax.plot import save_figure  # Bug: f485c26, feedbax 67bf476

from train_minimax import build_hps
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.modules.training.part2 import setup_task_model_pair


COMBO_LABEL = "gru__jerk_motor_smooth_combo"
N_REPLICATES = 5
N_WARMUP_BATCHES = 12000

COMBO_EXTRA_ARGS = {"nn_output_pre_go": 1e-2, "nn_hidden_derivative": 1e2}


def _color_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


REP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
]


def _make_args_namespace():
    defaults = dict(
        n_warmup_batches=N_WARMUP_BATCHES,
        n_adversary_batches=0,
        batch_size=250,
        n_replicates=N_REPLICATES,
        nn_output_jerk=1e5,
        seed=42,
        hidden_type="gru",
        sisu_gating="additive",
        loss_update_enabled=False,
        loss_update_ratio=0.5,
        effector_pos_running=1.0,
        effector_pos_late_weight=0.5,
        effector_vel_late=0.1,
        effector_final_vel=0.0,
        effector_pos_late_final_scale=2.0,
        effector_pos_late_start_step=80,
        nn_hidden_derivative=0.0,
        nn_output_pre_go=0.0,
        nn_hidden_derivative_pre_go=0.0,
        controller_lr=1e-4,
    )
    defaults.update(COMBO_EXTRA_ARGS)
    return argparse.Namespace(**defaults)


def load_combo_model(artifact_base: Path):
    """Load adversarial_model.eqx for the combo cell."""
    cell_dir = artifact_base / COMBO_LABEL
    eqx_path = cell_dir / "adversarial_model.eqx"
    if not eqx_path.exists():
        raise FileNotFoundError(f"adversarial_model.eqx not found: {eqx_path}")

    args = _make_args_namespace()
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(42))
    task = pair.task

    model, _ = load_with_hyperparameters(
        eqx_path,
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )
    return model, task, eqx_path


def _count_replicates(model) -> int:
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim >= 3:
            return int(leaf.shape[0])
    for leaf in jt.leaves(model):
        if eqx.is_array(leaf) and leaf.ndim == 2:
            return 1
    return 1


def eval_ensemble(task, model, trial_specs, *, key: jax.Array, n_replicates: int):
    """Rollout ensembled model. Returns states (n_rep, n_trials, n_steps, ...)."""
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    def _is_rep(x):
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_replicates

    arrays, other = eqx.partition(model, _is_rep)

    def _eval_one(arr, oth, rep_key):
        m = eqx.combine(arr, oth)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(m, trial_specs, keys)

    rep_keys = jr.split(key, n_replicates)
    return eqx.filter_vmap(_eval_one, in_axes=(0, None, 0))(arrays, other, rep_keys)


def build_trials(task, *, sisu: float, pert_scale: float = 0.0):
    """Build trial specs at given sisu and pert_scale."""
    val = task.validation_trials
    n_trials = val.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val,
        jnp.full((n_trials,), pert_scale),
    )
    trial_specs = eqx.tree_at(
        lambda t: t.inputs["sisu"],
        trial_specs,
        jnp.full((n_trials,), sisu),
    )
    return trial_specs


def analyse_motor_during_epochs(states, trial_specs) -> dict:
    """Extract per-step motor output magnitude and summarise hold vs movement.

    Args:
        states: (n_rep, n_trials, n_steps, ...) — ensembled eval states.
        trial_specs: TaskTrialSpec.

    Returns dict with:
        - "motor_magnitude": (n_rep, n_trials, n_steps) — L2 norm of efferent output
        - "go_idx": (n_trials,) int step index of go cue
        - "hold_motor_mean": (n_rep, n_trials) — mean magnitude during hold epoch
        - "hold_motor_max": (n_rep, n_trials) — max magnitude during hold epoch
        - "move_motor_mean": (n_rep, n_trials) — mean magnitude during movement epoch
        - "hold_to_move_ratio": (n_rep, n_trials) — hold_mean / move_mean
    """
    # efferent output: (n_rep, n_trials, n_steps, 2)
    motor = states.efferent.output
    motor_mag = jnp.linalg.norm(motor, axis=-1)  # (n_rep, n_trials, n_steps)

    # epoch_bounds: go cue index
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,)
    n_rep, n_trials, n_steps = motor_mag.shape
    t_arr = jnp.arange(n_steps)
    before_go = t_arr[None, None, :] < go_idx[None, :, None]   # (1, n_trials, n_steps)
    after_go = t_arr[None, None, :] >= go_idx[None, :, None]   # (1, n_trials, n_steps)

    # Hold-period stats
    hold_mag = jnp.where(before_go, motor_mag, 0.0)
    hold_count = before_go.astype(jnp.float32).sum(axis=-1)   # (1, n_trials)
    hold_count = jnp.broadcast_to(hold_count, (n_rep, n_trials))
    hold_motor_mean = hold_mag.sum(axis=-1) / jnp.maximum(hold_count, 1.0)
    hold_motor_max = jnp.where(before_go, motor_mag, -jnp.inf).max(axis=-1)
    hold_motor_max = jnp.where(jnp.isinf(hold_motor_max), 0.0, hold_motor_max)

    # Movement-period stats
    move_mag = jnp.where(after_go, motor_mag, 0.0)
    move_count = after_go.astype(jnp.float32).sum(axis=-1)
    move_count = jnp.broadcast_to(move_count, (n_rep, n_trials))
    move_motor_mean = move_mag.sum(axis=-1) / jnp.maximum(move_count, 1.0)

    hold_to_move = hold_motor_mean / jnp.maximum(move_motor_mean, 1e-12)

    return {
        "motor_magnitude": np.array(motor_mag),       # (n_rep, n_trials, n_steps)
        "go_idx": np.array(go_idx),                    # (n_trials,)
        "hold_motor_mean": np.array(hold_motor_mean),  # (n_rep, n_trials)
        "hold_motor_max": np.array(hold_motor_max),    # (n_rep, n_trials)
        "move_motor_mean": np.array(move_motor_mean),  # (n_rep, n_trials)
        "hold_to_move_ratio": np.array(hold_to_move),  # (n_rep, n_trials)
    }


def make_motor_magnitude_figure(
    motor_data: dict,
    dt: float = 0.01,
    n_trial_examples: int = 4,
) -> go.Figure:
    """Time vs motor output magnitude, one trace per replicate.

    Shows a selection of trials (both catch and go). Vertical dashed line at go cue.
    """
    motor_mag = motor_data["motor_magnitude"]     # (n_rep, n_trials, n_steps)
    go_idx = motor_data["go_idx"]                 # (n_trials,)
    n_rep, n_trials, n_steps = motor_mag.shape
    t = np.arange(n_steps) * dt

    # Select a few trials to display; use the first n_trial_examples
    trial_idx_show = np.arange(min(n_trial_examples, n_trials))
    n_show = len(trial_idx_show)

    fig = make_subplots(
        rows=n_show, cols=1,
        subplot_titles=[f"Trial {i}" for i in trial_idx_show],
        shared_xaxes=True,
        vertical_spacing=0.07,
    )

    for row_idx, trial_i in enumerate(trial_idx_show, start=1):
        go_t = float(go_idx[trial_i]) * dt

        for rep in range(n_rep):
            color = REP_COLORS[rep % len(REP_COLORS)]
            fig.add_trace(go.Scatter(
                x=t,
                y=motor_mag[rep, trial_i, :],
                mode="lines",
                name=f"Rep {rep}",
                line=dict(color=_color_rgba(color, 0.8), width=1.5),
                showlegend=(row_idx == 1),
                legendgroup=f"rep{rep}",
            ), row=row_idx, col=1)

        fig.add_vline(
            x=go_t,
            line=dict(color="black", dash="dash", width=1.5),
            row=row_idx,
            col=1,
            annotation_text="go cue",
        )

    fig.update_layout(
        title=(
            "Combo cell motor output magnitude over time — hold vs movement<br>"
            "<sup>Each panel = one trial; traces = 5 replicates. "
            "Dashed line = go cue.</sup>"
        ),
        width=900,
        height=200 * n_show + 100,
        margin=dict(l=70, r=60, t=100, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (s)", row=n_show, col=1)
    for row_idx in range(1, n_show + 1):
        fig.update_yaxes(title_text="‖motor‖ (N)", row=row_idx, col=1)

    return fig


def write_diagnostic_note(
    motor_data: dict,
    notes_path: Path,
    dt: float = 0.01,
):
    """Write a short verdict note."""
    hold_mean = motor_data["hold_motor_mean"]    # (n_rep, n_trials)
    hold_max = motor_data["hold_motor_max"]
    move_mean = motor_data["move_motor_mean"]
    h2m = motor_data["hold_to_move_ratio"]

    # Mean over all replicates and trials
    hold_mean_all = float(hold_mean.mean())
    hold_max_all = float(hold_max.mean())
    move_mean_all = float(move_mean.mean())
    h2m_all = float(h2m.mean())

    # Per-replicate (mean over trials)
    hold_mean_rep = hold_mean.mean(axis=1)   # (n_rep,)
    move_mean_rep = move_mean.mean(axis=1)
    h2m_rep = h2m.mean(axis=1)

    # Verdict
    if h2m_all > 0.05:
        verdict = (
            "**MOTOR COMMANDS NON-ZERO DURING HOLD.** "
            f"Hold-period motor magnitude is {h2m_all:.1%} of movement-period magnitude "
            f"(mean ‖motor‖ during hold = {hold_mean_all:.4f} N, "
            f"max = {hold_max_all:.4f} N). "
            "Drift is likely motor-command-driven — the nn_output_pre_go penalty is "
            "reducing but not eliminating pre-go motor activity. "
            "Increasing nn_output_pre_go weight is the targeted lever."
        )
    else:
        verdict = (
            "**MOTOR COMMANDS NEAR-ZERO DURING HOLD.** "
            f"Hold-period motor magnitude is {h2m_all:.1%} of movement-period magnitude "
            f"(mean ‖motor‖ during hold = {hold_mean_all:.4f} N, "
            f"max = {hold_max_all:.4f} N). "
            "Drift is likely mechanical (residual plant dynamics), NOT motor-command-driven. "
            "Increasing nn_output_pre_go would have limited effect; "
            "increasing effector_hold_pos/vel weights is the appropriate lever."
        )

    lines = [
        "# Diagnostic: Combo Cell Hold-Period Motor Outputs",
        "",
        "## Purpose",
        "",
        "Is the 4.6 mm residual hold-period drift in the combo cell",
        "(`gru__jerk_motor_smooth_combo`) driven by non-zero motor commands",
        "during the hold epoch, or by residual plant mechanics?",
        "",
        "See `_artifacts/scratchpad/residual_anticipation_proposal.md` §5 for framing.",
        "",
        "## Method",
        "",
        "- Loaded adversarial_model.eqx for combo cell (5 replicates).",
        "- Ran inference on 8 validation trials at SISU=0.5, pert_scale=0.",
        "- Extracted `states.efferent.output` (motor commands) per step.",
        "- Computed ‖motor‖ = L2 norm of 2D force command per step.",
        "- Hold epoch = steps before go cue; movement epoch = steps from go cue onward.",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean ‖motor‖ during hold (all reps, all trials) | {hold_mean_all:.4f} N |",
        f"| Mean max ‖motor‖ during hold (all reps, all trials) | {hold_max_all:.4f} N |",
        f"| Mean ‖motor‖ during movement (all reps, all trials) | {move_mean_all:.4f} N |",
        f"| Hold / movement mean ratio | {h2m_all:.4f} ({h2m_all:.1%}) |",
        "",
        "Per-replicate (mean over 8 trials):",
        "",
        "| Replicate | Hold mean (N) | Move mean (N) | Ratio |",
        "|-----------|--------------|--------------|-------|",
    ]
    for rep in range(len(hold_mean_rep)):
        lines.append(
            f"| Rep {rep} "
            f"| {hold_mean_rep[rep]:.4f} "
            f"| {move_mean_rep[rep]:.4f} "
            f"| {h2m_rep[rep]:.4f} ({h2m_rep[rep]:.1%}) |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Figure",
        "",
        "See `figures/combo_hold_motor_diagnostic/figure.html` for per-trial time traces.",
    ]

    notes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notes_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "hold_mean_all": hold_mean_all,
        "hold_max_all": hold_max_all,
        "move_mean_all": move_mean_all,
        "h2m_all": h2m_all,
        "verdict_type": "nonzero" if h2m_all > 0.05 else "near_zero",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=None,
    )
    parser.add_argument("--sisu", type=float, default=0.5)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--n-trial-examples",
        type=int,
        default=4,
        help="Number of example trials to show in the figure",
    )
    args = parser.parse_args()

    artifact_base = args.artifact_base or (REPO_ROOT / "_artifacts" / "2bc95fd")
    results_base = REPO_ROOT / "results" / "2bc95fd"

    print(f"Artifact base: {artifact_base}")
    print(f"Results base:  {results_base}")

    # -----------------------------------------------------------------------
    # Load combo cell
    # -----------------------------------------------------------------------
    print(f"\nLoading combo cell ({COMBO_LABEL}) ...")
    model, task, eqx_path = load_combo_model(artifact_base)
    n_reps = _count_replicates(model)
    print(f"  Loaded. n_replicates={n_reps}")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    trial_specs = build_trials(task, sisu=args.sisu, pert_scale=0.0)
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    print(f"  Evaluating on {n_trials} trials ...")
    states = eval_ensemble(task, model, trial_specs, key=jr.PRNGKey(args.eval_seed), n_replicates=n_reps)
    print("  Eval OK.")

    # -----------------------------------------------------------------------
    # Analyse motor outputs
    # -----------------------------------------------------------------------
    print("\nAnalysing hold-period motor outputs ...")
    motor_data = analyse_motor_during_epochs(states, trial_specs)

    hold_mean = float(motor_data["hold_motor_mean"].mean())
    hold_max = float(motor_data["hold_motor_max"].mean())
    move_mean = float(motor_data["move_motor_mean"].mean())
    h2m = float(motor_data["hold_to_move_ratio"].mean())

    print(f"  Hold mean ‖motor‖:  {hold_mean:.4f} N")
    print(f"  Hold max  ‖motor‖:  {hold_max:.4f} N")
    print(f"  Move mean ‖motor‖:  {move_mean:.4f} N")
    print(f"  Hold/Move ratio:    {h2m:.4f} ({h2m:.1%})")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    print("\nBuilding figure ...")
    fig = make_motor_magnitude_figure(
        motor_data,
        dt=0.01,
        n_trial_examples=args.n_trial_examples,
    )

    (artifact_base / "figures" / "combo_hold_motor_diagnostic").mkdir(parents=True, exist_ok=True)
    (results_base / "figures" / "combo_hold_motor_diagnostic").mkdir(parents=True, exist_ok=True)

    spec = {
        "figure_kind": "motor_magnitude_time_series_hold_diagnostic",
        "experiment": "anti_anticipation_loss_shape_6cell",
        "cell": COMBO_LABEL,
        "inputs": [{"path": str(eqx_path), "role": "adversarial_model:combo"}],
        "transform": [
            {"name": "eval_ensemble", "kwargs": {"sisu": args.sisu, "pert_scale": 0.0}},
            {"name": "efferent_output_l2_norm", "kwargs": {}},
        ],
        "plot_kwargs": {
            "sisu": args.sisu,
            "pert_scale": 0.0,
            "n_trial_examples": args.n_trial_examples,
            "dt": 0.01,
        },
        "summary": {
            "hold_motor_mean_N": hold_mean,
            "hold_motor_max_N": hold_max,
            "move_motor_mean_N": move_mean,
            "hold_to_move_ratio": h2m,
        },
    }

    fig_out = save_figure(
        fig=fig, spec=spec,
        package="rlrmp", experiment="2bc95fd", topic="combo_hold_motor_diagnostic",
        extra_packages=["rlrmp"],
    )
    print(f"  Spec:   {fig_out['spec_path']}")
    print(f"  Render: {fig_out['render_path']}")

    # -----------------------------------------------------------------------
    # Write notes
    # -----------------------------------------------------------------------
    notes_path = results_base / "notes" / "combo_hold_motor_diagnostic.md"
    summary = write_diagnostic_note(motor_data, notes_path, dt=0.01)
    print(f"\nSaved notes: {notes_path}")
    print(f"Verdict: {summary['verdict_type']}")
    print("\nDone.")


if __name__ == "__main__":
    main()

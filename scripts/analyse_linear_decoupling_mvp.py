"""Linear-controller decoupling acid test — MVP analysis (Bug: 410d7ac).

Loads the two warmup-trained linear-controller models (regulator + tracker)
alongside the f47abb1 ``lit__post_nojerk`` GRU baseline, evaluates each at a
sweep of plant-disturbance scales, and reports the velocity-inflation signature:

    Δv(scale) = (peak_fwd_vel(scale) - peak_fwd_vel(0)) / peak_fwd_vel(0)

Hypothesis (Bug: 410d7ac): the **regulator** parameterisation gives Δv > 0
(velocity inflates under disturbance) while the **tracker** parameterisation
gives Δv ≈ 0 because the independent ``u_ff(t)`` channel decouples open-loop
drive from closed-loop stiffness.

Outputs:
  - stdout report with per-model (Δv mean ± SD across replicates)
  - JSON summary at ``results/410d7ac/notes/delta_v_summary.json``
  - HTML figure at ``_artifacts/410d7ac/figures/delta_v_signature/figure.html``
    via ``feedbax.plot.save_figure``.

Usage (from feature worktree):
    JAX_PLATFORMS=cpu uv run python scripts/analyse_linear_decoupling_mvp.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

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

from feedbax._io import load_with_hyperparameters
from feedbax.plot import save_figure

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.modules.training.part2 import setup_task_model_pair
from train_minimax import build_hps


EXPERIMENT = "410d7ac"
N_REPLICATES = 5
PERT_SCALES = (0.0, 0.5, 1.0, 1.5)
SEED_EVAL = 42


# ---------------------------------------------------------------------------
# Configuration of the runs to compare
# ---------------------------------------------------------------------------


def _linear_args(hidden_type: str) -> argparse.Namespace:
    """Reconstruct the CLI namespace used to train the linear MVP runs."""
    return argparse.Namespace(
        method="pai-asf",
        hidden_type=hidden_type,
        sisu_gating="additive",
        n_warmup_batches=1000,
        n_adversary_batches=0,
        batch_size=64,
        n_replicates=N_REPLICATES,
        controller_lr=5e-3,
        seed=42,
        adversary_type="linear_dynamics",
        # loss flags matching the lit__post_nojerk shape
        effector_hold_pos=1.0,
        effector_hold_vel=0.0,
        effector_pos_running=1.0,
        effector_pos_late_weight=0.0,
        effector_pos_late_final_scale=2.0,
        effector_pos_late_start_step=80,
        effector_vel_late=0.0,
        effector_final_vel=0.0,
        nn_output=1e-5,
        nn_hidden=0.0,
        nn_output_jerk=0.0,
        nn_hidden_derivative=0.0,
        nn_output_pre_go=0.0,
        nn_hidden_derivative_pre_go=0.0,
        effector_pos_running_schedule="powerlaw",
        effector_hold_pos_schedule="flat",
        position_powerlaw_power=6.0,
        p_catch_trial=0.5,
        loss_update_enabled=False,
        loss_update_ratio=0.5,
    )


def _baseline_args() -> argparse.Namespace:
    """CLI namespace for the f47abb1 lit__post_nojerk GRU baseline."""
    return argparse.Namespace(
        method="pai-asf",
        hidden_type="gru",
        sisu_gating="additive",
        n_warmup_batches=12000,
        n_adversary_batches=0,
        batch_size=250,
        n_replicates=N_REPLICATES,
        controller_lr=1e-4,
        seed=42,
        adversary_type="linear_dynamics",
        effector_hold_pos=1.0,
        effector_hold_vel=0.0,
        effector_pos_running=1.0,
        effector_pos_late_weight=0.0,
        effector_pos_late_final_scale=2.0,
        effector_pos_late_start_step=80,
        effector_vel_late=0.0,
        effector_final_vel=0.0,
        nn_output=1e-5,
        nn_hidden=1e-5,
        nn_output_jerk=0.0,
        nn_hidden_derivative=0.001,
        nn_output_pre_go=0.0,
        nn_hidden_derivative_pre_go=0.0,
        effector_pos_running_schedule="powerlaw",
        effector_hold_pos_schedule="flat",
        position_powerlaw_power=6.0,
        p_catch_trial=0.5,
        loss_update_enabled=False,
        loss_update_ratio=0.5,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(eqx_path: Path, args: argparse.Namespace):
    """Load a model by reconstructing the skeleton via setup_task_model_pair."""
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(args.seed))
    model, _ = load_with_hyperparameters(
        eqx_path,
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )
    return model, pair.task


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def make_trials(task, pert_scale: float, sisu: float = 0.5):
    """Validation trials with the disturbance scale and SISU pinned."""
    val = task.validation_trials
    n_trials = val.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    trials = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val,
        jnp.full((n_trials,), pert_scale),
    )
    # Pin SISU iff the input dict carries it (it does — added by part2).
    if "sisu" in trials.inputs:
        trials = eqx.tree_at(
            lambda t: t.inputs["sisu"],
            trials,
            jnp.full((n_trials,), sisu),
        )
    return trials


def eval_ensemble(task, model, trial_specs, *, key, n_replicates: int):
    """Evaluate ensembled model. Returns states with leading (n_rep, n_trials)."""
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


def peak_forward_velocity(states, trial_specs) -> np.ndarray:
    """Compute peak forward (along reach-axis) velocity per (replicate, trial).

    Returns ``(n_rep, n_trials)`` array of peak forward speeds (m/s).
    """
    pos = states.mechanics.effector.pos  # (n_rep, n_trials, n_steps, 2)
    vel = states.mechanics.effector.vel  # same shape

    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value  # (n_trials, n_steps, 2)
    goal = goal_seq[:, -1, :]  # (n_trials, 2)

    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,)
    n_rep, n_trials, n_steps, _ = pos.shape

    # Init pos at go cue, per replicate per trial
    def _gather_init(pos_rep, go_idx_arr):
        # pos_rep: (n_trials, n_steps, 2); go_idx_arr: (n_trials,)
        return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_idx_arr)

    init_pos = jax.vmap(_gather_init, in_axes=(0, None))(pos, go_idx)  # (n_rep, n_trials, 2)
    direction = goal[None, :, :] - init_pos  # (n_rep, n_trials, 2)
    direction_unit = direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)

    # Forward velocity projection: dot(vel_t, direction_unit) per step
    fwd_vel = jnp.sum(vel * direction_unit[:, :, None, :], axis=-1)  # (n_rep, n_trials, n_steps)
    # Mask to post-go window
    t_idx = jnp.arange(n_steps)
    after_go = t_idx[None, None, :] >= go_idx[None, :, None]
    fwd_vel = jnp.where(after_go, fwd_vel, -jnp.inf)
    return np.asarray(jnp.max(fwd_vel, axis=-1))  # (n_rep, n_trials)


# ---------------------------------------------------------------------------
# Top-level: sweep + report
# ---------------------------------------------------------------------------


def sweep_model(
    label: str,
    eqx_path: Path,
    args: argparse.Namespace,
    pert_scales: tuple[float, ...] = PERT_SCALES,
) -> dict[str, np.ndarray]:
    print(f"\n[{label}] loading {eqx_path}")
    model, task = load_model(eqx_path, args)

    per_scale_peak_vel = {}
    key = jr.PRNGKey(SEED_EVAL)
    for s in pert_scales:
        trials = make_trials(task, pert_scale=float(s), sisu=0.5)
        key, k = jr.split(key)
        states = eval_ensemble(task, model, trials, key=k, n_replicates=N_REPLICATES)
        peak = peak_forward_velocity(states, trials)  # (n_rep, n_trials)
        per_scale_peak_vel[float(s)] = peak
        print(f"  pert_scale={s:.2f}  peak_vel mean={peak.mean():.4f}  sd={peak.std():.4f}")
    return per_scale_peak_vel


def delta_v(per_scale_peak_vel: dict[float, np.ndarray]) -> dict[float, dict[str, float]]:
    """Compute Δv = (peak(s) - peak(0)) / peak(0) per scale."""
    baseline = per_scale_peak_vel[0.0]  # (n_rep, n_trials)
    out = {}
    for s, peak in per_scale_peak_vel.items():
        if s == 0.0:
            continue
        # Per-replicate Δv: trial-mean(peak) / trial-mean(baseline) - 1
        # Use replicate-level means so SD across replicates is meaningful.
        peak_rep = peak.mean(axis=-1)         # (n_rep,)
        base_rep = baseline.mean(axis=-1)     # (n_rep,)
        dv_rep = (peak_rep - base_rep) / np.maximum(base_rep, 1e-12)
        out[s] = {
            "mean": float(dv_rep.mean()),
            "sd": float(dv_rep.std()),
            "per_rep": dv_rep.tolist(),
        }
    return out


def main():
    artifact_base_410 = REPO_ROOT / "_artifacts" / EXPERIMENT / "runs"
    artifact_base_f47 = REPO_ROOT / "_artifacts" / "f47abb1" / "runs"

    runs = {
        "linear_regulator": (
            artifact_base_410 / "linear_regulator" / "warmup_model.eqx",
            _linear_args("linear"),
        ),
        "linear_tracker": (
            artifact_base_410 / "linear_tracker" / "warmup_model.eqx",
            _linear_args("linear_tracker"),
        ),
        "gru_baseline_lit_post_nojerk": (
            artifact_base_f47 / "lit__post_nojerk" / "warmup_model.eqx",
            _baseline_args(),
        ),
    }

    results: dict[str, dict] = {}
    for label, (eqx_path, args) in runs.items():
        if not eqx_path.exists():
            print(f"[{label}] WARN: artifact {eqx_path} missing — skipping")
            continue
        per_scale = sweep_model(label, eqx_path, args)
        dv = delta_v(per_scale)
        results[label] = {
            "peak_velocity": {
                str(s): {
                    "mean": float(p.mean()),
                    "sd": float(p.std()),
                    "rep_means": p.mean(axis=-1).tolist(),
                }
                for s, p in per_scale.items()
            },
            "delta_v": {str(s): v for s, v in dv.items()},
        }

    # Report
    print("\n" + "=" * 80)
    print("Δv signature (peak forward velocity inflation under disturbance)")
    print("Convention: Δv > 0  => velocity inflates under perturbation (regulator-like)")
    print("            Δv ≈ 0  => decoupled feedforward/feedback (tracker-like)")
    print("=" * 80)
    header = f"  {'model':>32} | " + " | ".join(
        f"{f'Δv@{s:.1f}':>14}" for s in PERT_SCALES if s > 0.0
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, res in results.items():
        dv = res["delta_v"]
        cells = " | ".join(
            f"{dv[str(s)]['mean']:+.3f} ± {dv[str(s)]['sd']:.3f}"
            for s in PERT_SCALES if s > 0.0
        )
        print(f"  {label:>32} | {cells}")

    # Save JSON summary
    notes_dir = REPO_ROOT / "results" / EXPERIMENT / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    summary_path = notes_dir / "delta_v_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved summary JSON: {summary_path}")

    # Build figure
    fig = go.Figure()
    color_map = {
        "linear_regulator": "#d62728",  # red — predicted Δv > 0
        "linear_tracker": "#2ca02c",     # green — predicted Δv ≈ 0
        "gru_baseline_lit_post_nojerk": "#1f77b4",  # blue — reference
    }
    scales_pos = [s for s in PERT_SCALES if s > 0.0]
    for label, res in results.items():
        means = [res["delta_v"][str(s)]["mean"] for s in scales_pos]
        sds = [res["delta_v"][str(s)]["sd"] for s in scales_pos]
        fig.add_trace(
            go.Scatter(
                x=scales_pos, y=means,
                error_y=dict(type="data", array=sds),
                mode="lines+markers",
                name=label,
                line=dict(color=color_map.get(label, None), width=2),
                marker=dict(size=10),
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title="Velocity-inflation signature Δv under LinearDynamics disturbance — 410d7ac MVP",
        xaxis_title="Disturbance scale (relative units)",
        yaxis_title="Δv = (peak_vel(scale) − peak_vel(0)) / peak_vel(0)",
        legend_title="Architecture",
        template="plotly_white",
    )
    spec = {
        "experiment": EXPERIMENT,
        "topic": "delta_v_signature",
        "pert_scales": list(PERT_SCALES),
        "n_replicates": N_REPLICATES,
        "seed_eval": SEED_EVAL,
        "runs": list(results.keys()),
        "description": (
            "Δv per disturbance scale per architecture, mean ± SD across "
            f"{N_REPLICATES} replicates."
        ),
    }
    save_figure(
        fig=fig, spec=spec,
        package="rlrmp",
        experiment=EXPERIMENT,
        topic="delta_v_signature",
        extra_packages=["rlrmp"],
    )
    print(f"\nFigure saved (see results/{EXPERIMENT}/figures/delta_v_signature/)")
    return results


if __name__ == "__main__":
    main()

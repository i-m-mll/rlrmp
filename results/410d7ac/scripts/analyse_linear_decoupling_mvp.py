# TODO: relocate to results/410d7ac/scripts/ — per CLAUDE.md script-placement convention
"""Linear-controller decoupling acid test — corrected MVP analysis (Bug: 410d7ac).

Bug: 06f7faf — the primary Δv metric is unaffected (it uses per-trial after-go
masking before computing peak), but the velocity-profile figure is now plotted
in go-aligned coordinates so the visual onset of motion lines up with t=0.


This is the **corrected** version of the linear regulator-vs-tracker decoupling
acid test. The prior version (commit 20ae797) trained warmup-only models and
measured test-time perturbation response, which is NOT Δv. See the retraction
comment on issue 410d7ac.

What Δv actually means
----------------------
Δv is the **peak forward velocity inflation between an adversarially-trained
model and the warmup-only baseline of the SAME architecture**, signed and
projected on the reach axis:

    Δv_arch = (peak_v(arch_adversarial) - peak_v(arch_baseline)) / peak_v(arch_baseline)

This is a training-method comparison, not a test-time perturbation response.
Both models are evaluated under the *same* conditions; the difference between
them lies in whether adversarial training was applied. This matches Crevecoeur
& Scott's signature for robust control: H-infinity controllers (adversarially
optimal) inflate peak forward velocity relative to LQR controllers (baseline-
optimal).

Discriminator prediction (d448c9d)
----------------------------------
- ``Δv_regulator > 0`` — the regulator parameterisation has no decoupled
  feedforward channel, so adversarial training inflates peak velocity.
- ``Δv_tracker ≈ 0`` — the tracker's independent ``u_ff(t)`` channel can
  absorb the disturbance threat without changing the forward motion plan, so
  adversarial training leaves peak velocity untouched.

Four trained models are required for this comparison; this script loads all
four and computes Δv for each architecture pair.

Inputs (run directories)
------------------------
- ``_artifacts/410d7ac/runs/linear_regulator__baseline/warmup_model.eqx``
- ``_artifacts/410d7ac/runs/linear_regulator__adversarial/adversarial_model.eqx``
- ``_artifacts/410d7ac/runs/linear_tracker__baseline/warmup_model.eqx``
- ``_artifacts/410d7ac/runs/linear_tracker__adversarial/adversarial_model.eqx``

Outputs
-------
- stdout report with Δv per architecture (mean ± SEM across replicates)
- JSON summary at ``results/410d7ac/notes/delta_v_summary.json``
- HTML figure at ``_artifacts/410d7ac/figures/delta_v_signature/figure.html``
  via ``feedbax.plot.save_figure`` (auto-mirrored into ``results/...``).

Usage (from feature worktree):
    JAX_PLATFORMS=cpu uv run python scripts/analyse_linear_decoupling_mvp.py
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
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go
from feedbax import load_with_hyperparameters
from feedbax.plot import save_figure
from plotly.subplots import make_subplots

from rlrmp.analysis.trial_alignment import align_trials, replicate_mean_curves
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.intervention_compat import (
    swap_plant_intervenor_to_dynamics_matrix,
    swap_task_intervention_to_dynamics_matrix,
)
from rlrmp.paths import REPO_ROOT  # Bug: 8404108 — was __file__-relative
from rlrmp.train.minimax import build_hps
from rlrmp.train.task_model import setup_task_model_pair

EXPERIMENT = "410d7ac"
N_REPLICATES = 5
SEED_EVAL = 42

# Evaluation conditions. Δv is computed at pert_scale=0 (the C&S-style analytical
# baseline: the controller has been *designed* against a disturbance, but is
# evaluated on a clean reach). pert_scale=1.0 is included as a secondary check.
EVAL_PERT_SCALES = (0.0, 1.0)
HEADLINE_PERT_SCALE = 0.0


# ---------------------------------------------------------------------------
# Configuration of the 4 runs to compare
# ---------------------------------------------------------------------------


def _common_linear_args() -> dict:
    """Shared CLI namespace fields for both regulator + tracker variants."""
    return dict(
        method="pai-asf",
        sisu_gating="additive",
        n_warmup_batches=1000,
        batch_size=64,
        n_replicates=N_REPLICATES,
        controller_lr=5e-3,
        seed=42,
        adversary_type="linear_dynamics",
        # adversarial-phase hyperparameters (used only for namespace completeness;
        # setup_task_model_pair does not consult these for model construction)
        n_adversary_steps=5,
        adversary_lr=3e-4,
        linear_dynamics_eta_max=0.1,
        linear_dynamics_pgd_steps=5,
        linear_dynamics_lr=1e-2,
        n_bumps=3,
        force_max=1.0,
        n_adversaries=1,
        adv_batch_size=None,
        # loss flags
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


def _linear_args(hidden_type: str, n_adversary_batches: int) -> argparse.Namespace:
    """Reconstruct the CLI namespace used to train a linear-controller run."""
    kw = _common_linear_args()
    kw["hidden_type"] = hidden_type
    kw["n_adversary_batches"] = n_adversary_batches
    return argparse.Namespace(**kw)


# Four-run config table.
# Each entry: (label, eqx_relative_path, args)
# eqx_relative_path is relative to _artifacts/<experiment>/runs/<dir>/.
RUNS = {
    "linear_regulator__baseline": (
        "linear_regulator__baseline",
        "warmup_model.eqx",
        _linear_args("linear", n_adversary_batches=0),
    ),
    "linear_regulator__adversarial": (
        "linear_regulator__adversarial",
        "adversarial_model.eqx",
        _linear_args("linear", n_adversary_batches=500),
    ),
    "linear_tracker__baseline": (
        "linear_tracker__baseline",
        "warmup_model.eqx",
        _linear_args("linear_tracker", n_adversary_batches=0),
    ),
    "linear_tracker__adversarial": (
        "linear_tracker__adversarial",
        "adversarial_model.eqx",
        _linear_args("linear_tracker", n_adversary_batches=500),
    ),
}

ARCH_PAIRS = (
    ("linear_regulator", "linear_regulator__baseline", "linear_regulator__adversarial"),
    ("linear_tracker", "linear_tracker__baseline", "linear_tracker__adversarial"),
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(eqx_path: Path, args: argparse.Namespace):
    """Load a model by reconstructing the skeleton via setup_task_model_pair.

    For adversarial-phase models (those saved as ``adversarial_model.eqx``),
    the saved PyTree carries a DynamicsMatrixPerturb intervenor (swapped in
    during training). The default skeleton produced by
    ``setup_task_model_pair`` has the warmup-phase FixedField intervenor.
    Apply the same swap to the skeleton before deserialising so the
    structures match. Bug: 410d7ac.
    """
    hps = build_hps(args)
    is_adversarial = (
        args.n_adversary_batches > 0 and eqx_path.name == "adversarial_model.eqx"
    )

    def _setup_func(key, **kwargs):
        m = setup_task_model_pair(hps, key=key).model
        if is_adversarial:
            m = jt.map(
                lambda x: swap_plant_intervenor_to_dynamics_matrix(
                    x, PLANT_INTERVENOR_LABEL, mass=hps.model.effector_mass,
                ),
                m,
                is_leaf=lambda x: x is not None and hasattr(x, "nodes")
                                  and hasattr(x, "input_ports"),
            )
        return m

    model, _ = load_with_hyperparameters(eqx_path, setup_func=_setup_func)

    pair = setup_task_model_pair(hps, key=jr.PRNGKey(args.seed))
    task = pair.task
    if is_adversarial:
        task = swap_task_intervention_to_dynamics_matrix(task, PLANT_INTERVENOR_LABEL)
    return model, task


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def make_trials(task, pert_scale: float, sisu: float = 0.5):
    """Validation trials with disturbance scale and SISU pinned."""
    val = task.validation_trials
    n_trials = val.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    trials = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val,
        jnp.full((n_trials,), pert_scale),
    )
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
    """Per-trial peak forward velocity projected on the reach axis.

    Returns ``(n_rep, n_trials)`` array of peak forward speeds.
    """
    pos = states.mechanics.effector.pos
    vel = states.mechanics.effector.vel

    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value
    goal = goal_seq[:, -1, :]

    go_idx = trial_specs.timeline.epoch_bounds[:, 2]
    n_rep, n_trials, n_steps, _ = pos.shape

    def _gather_init(pos_rep, go_idx_arr):
        return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_idx_arr)

    init_pos = jax.vmap(_gather_init, in_axes=(0, None))(pos, go_idx)
    direction = goal[None, :, :] - init_pos
    direction_unit = direction / jnp.maximum(
        jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-12
    )

    fwd_vel = jnp.sum(vel * direction_unit[:, :, None, :], axis=-1)
    t_idx = jnp.arange(n_steps)
    after_go = t_idx[None, None, :] >= go_idx[None, :, None]
    fwd_vel_masked = jnp.where(after_go, fwd_vel, -jnp.inf)
    peak = jnp.max(fwd_vel_masked, axis=-1)
    return np.asarray(peak), np.asarray(fwd_vel), np.asarray(after_go)


def forward_velocity_profile(states, trial_specs) -> np.ndarray:
    """Per-step forward velocity (n_rep, n_trials, n_steps)."""
    peak, fwd_vel, _ = peak_forward_velocity(states, trial_specs)
    return fwd_vel


def u_ff_diagnostic(model, args: argparse.Namespace) -> dict | None:
    """If model is a LinearTrackerController ensemble, return |u_ff| stats.

    Reads ``model.nodes["net"].u_ff`` directly. The ensembled model has
    a leading ``(n_replicates,)`` axis on the array.
    """
    if args.hidden_type != "linear_tracker":
        return None
    try:
        u_ff_arr = np.asarray(model.nodes["net"].u_ff)
    except (AttributeError, KeyError):
        return None
    return {
        "u_ff_shape": list(u_ff_arr.shape),
        "u_ff_abs_max": float(np.abs(u_ff_arr).max()),
        "u_ff_l2_mean": float(np.sqrt((u_ff_arr ** 2).sum(axis=-1)).mean()),
        "u_ff_per_rep_abs_max": [
            float(np.abs(u_ff_arr[i]).max()) for i in range(u_ff_arr.shape[0])
        ],
    }


def K_diagnostic(model) -> dict | None:
    """Return |K| stats for both LinearController and LinearTrackerController."""
    try:
        K_arr = np.asarray(model.nodes["net"].K)
    except (AttributeError, KeyError):
        return None
    return {
        "K_shape": list(K_arr.shape),
        "K_frobenius_mean": float(np.sqrt((K_arr ** 2).sum(axis=(-1, -2))).mean()),
        "K_abs_max": float(np.abs(K_arr).max()),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def evaluate_one_run(label: str, dir_name: str, eqx_filename: str,
                     args: argparse.Namespace) -> dict:
    """Load a single trained model and compute peak fwd vel at each pert scale."""
    artifact_dir = REPO_ROOT / "_artifacts" / EXPERIMENT / "runs" / dir_name
    eqx_path = artifact_dir / eqx_filename
    if not eqx_path.exists():
        print(f"[{label}] MISSING artifact: {eqx_path}")
        return {"missing": True, "eqx_path": str(eqx_path)}

    print(f"\n[{label}] loading {eqx_path}")
    model, task = load_model(eqx_path, args)

    out: dict = {
        "eqx_path": str(eqx_path),
        "peak_velocity_per_scale": {},
        "fwd_velocity_profile_per_scale": {},
    }
    key = jr.PRNGKey(SEED_EVAL)
    for s in EVAL_PERT_SCALES:
        trials = make_trials(task, pert_scale=float(s), sisu=0.5)
        key, k = jr.split(key)
        states = eval_ensemble(task, model, trials, key=k, n_replicates=N_REPLICATES)
        peak, fwd_vel, after_go = peak_forward_velocity(states, trials)
        out["peak_velocity_per_scale"][float(s)] = {
            "per_rep_trial": peak,  # (n_rep, n_trials)
            "rep_means": peak.mean(axis=-1),  # (n_rep,)
            "overall_mean": float(peak.mean()),
            "overall_sd": float(peak.std()),
        }
        # Trial-mean profile per replicate. For plotting only.
        # Bug: 06f7faf — align per-trial profiles to each trial's go cue BEFORE
        # the trial-axis collapse. The primary Δv metric (peak above) is
        # unaffected because it uses per-trial after_go masking before max.
        go_idx = np.asarray(trials.timeline.epoch_bounds[:, 2])
        aligned_fv, center = align_trials(np.asarray(fwd_vel), go_idx)
        # trim=False because the downstream plot computes its own time axis
        # from a stored `center` and the array's column count; the trim slice
        # would have to be persisted alongside to apply consistently.
        out["fwd_velocity_profile_per_scale"][float(s)] = replicate_mean_curves(
            aligned_fv, trim=False
        )  # (n_rep, n_aligned_steps)
        out.setdefault("go_align_center_per_scale", {})[float(s)] = int(center)
        print(f"  pert_scale={s:.2f}  peak_vel mean={peak.mean():.4f}  "
              f"sd_across_reps={peak.mean(axis=-1).std():.4f}")

    u_ff_diag = u_ff_diagnostic(model, args)
    if u_ff_diag is not None:
        out["u_ff_diagnostic"] = u_ff_diag
        print(f"  |u_ff|_max = {u_ff_diag['u_ff_abs_max']:.4f}  "
              f"|u_ff|_L2 mean = {u_ff_diag['u_ff_l2_mean']:.4f}")
    k_diag = K_diagnostic(model)
    if k_diag is not None:
        out["K_diagnostic"] = k_diag
        print(f"  |K|_frob_mean = {k_diag['K_frobenius_mean']:.4f}  "
              f"|K|_max = {k_diag['K_abs_max']:.4f}")
    return out


def compute_delta_v(baseline_result: dict, adversarial_result: dict) -> dict:
    """Compute Δv at each pert scale.

    Returns a dict keyed by pert_scale, with mean / SEM across replicates.
    Uses per-replicate trial-mean peak velocities (n_rep,) as the unit of
    variation.
    """
    out = {}
    for s in EVAL_PERT_SCALES:
        peak_b = baseline_result["peak_velocity_per_scale"][float(s)]["rep_means"]
        peak_a = adversarial_result["peak_velocity_per_scale"][float(s)]["rep_means"]
        # Per-replicate Δv (paired by replicate index — same seed implies same
        # warmup initialisation, so the pairing is meaningful).
        dv_rep = (peak_a - peak_b) / np.maximum(peak_b, 1e-12)
        out[float(s)] = {
            "mean": float(dv_rep.mean()),
            "sem": float(dv_rep.std(ddof=1) / np.sqrt(len(dv_rep))) if len(dv_rep) > 1 else 0.0,
            "sd": float(dv_rep.std(ddof=1)) if len(dv_rep) > 1 else 0.0,
            "per_rep": dv_rep.tolist(),
            "peak_v_baseline_mean": float(peak_b.mean()),
            "peak_v_adversarial_mean": float(peak_a.mean()),
        }
    return out


def main():
    # 1. Evaluate all four runs
    results: dict = {}
    for label, (dir_name, eqx_filename, args) in RUNS.items():
        results[label] = evaluate_one_run(label, dir_name, eqx_filename, args)

    # 2. Compute Δv per architecture pair
    delta_v_by_arch: dict = {}
    for arch, baseline_label, adv_label in ARCH_PAIRS:
        if results[baseline_label].get("missing") or results[adv_label].get("missing"):
            print(f"\n[{arch}] skipped — missing artifact(s)")
            continue
        dv = compute_delta_v(results[baseline_label], results[adv_label])
        delta_v_by_arch[arch] = dv

    # 3. Report
    print("\n" + "=" * 80)
    print("Δv signature — peak forward velocity inflation, adversarial vs baseline")
    print("Definition: Δv = (peak_v(adv) - peak_v(base)) / peak_v(base)")
    print("Discriminator: regulator → Δv > 0;  tracker → Δv ≈ 0")
    print("=" * 80)

    header = f"  {'architecture':>20} | " + " | ".join(
        f"{f'Δv @ pert={s:.1f}':>18}" for s in EVAL_PERT_SCALES
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arch, dv in delta_v_by_arch.items():
        cells = " | ".join(
            f"{dv[float(s)]['mean']:+.4f} ± {dv[float(s)]['sem']:.4f} (SEM)"
            for s in EVAL_PERT_SCALES
        )
        print(f"  {arch:>20} | {cells}")
    print()

    # 4. u_ff diagnostic
    print("u_ff utilization (tracker only):")
    for label in ("linear_tracker__baseline", "linear_tracker__adversarial"):
        diag = results[label].get("u_ff_diagnostic") if label in results else None
        if diag:
            print(f"  {label:>28}: |u_ff|_max={diag['u_ff_abs_max']:.4f}, "
                  f"|u_ff|_L2 mean={diag['u_ff_l2_mean']:.4f}")

    # 5. Save JSON summary
    summary = {
        "experiment": EXPERIMENT,
        "definition": "Δv = (peak_v(adversarial) - peak_v(baseline)) / peak_v(baseline)",
        "eval_pert_scales": list(EVAL_PERT_SCALES),
        "headline_pert_scale": HEADLINE_PERT_SCALE,
        "n_replicates": N_REPLICATES,
        "seed_eval": SEED_EVAL,
        "discriminator_prediction": {
            "regulator": "Δv > 0",
            "tracker": "Δv ≈ 0",
        },
        "per_run": {
            label: {
                "eqx_path": r.get("eqx_path"),
                "peak_velocity": {
                    str(s): {
                        "overall_mean": r["peak_velocity_per_scale"][s]["overall_mean"],
                        "overall_sd": r["peak_velocity_per_scale"][s]["overall_sd"],
                        "rep_means": r["peak_velocity_per_scale"][s]["rep_means"].tolist(),
                    }
                    for s in EVAL_PERT_SCALES
                    if not r.get("missing")
                } if not r.get("missing") else None,
                "u_ff_diagnostic": r.get("u_ff_diagnostic"),
                "missing": r.get("missing", False),
            }
            for label, r in results.items()
        },
        "delta_v_by_arch": {
            arch: {str(s): v for s, v in dv.items()}
            for arch, dv in delta_v_by_arch.items()
        },
    }
    notes_dir = REPO_ROOT / "results" / EXPERIMENT / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    summary_path = notes_dir / "delta_v_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved summary JSON: {summary_path}")

    # 6. Figure: side-by-side velocity profiles (regulator vs tracker, baseline vs adv)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Linear regulator", "Linear tracker"),
        shared_yaxes=True,
    )
    color_map = {"baseline": "#1f77b4", "adversarial": "#d62728"}
    pert_idx_for_plot = HEADLINE_PERT_SCALE
    for col, (arch, base_label, adv_label) in enumerate(ARCH_PAIRS, start=1):
        for role, label in (("baseline", base_label), ("adversarial", adv_label)):
            r = results.get(label, {})
            if r.get("missing"):
                continue
            profile = r["fwd_velocity_profile_per_scale"].get(pert_idx_for_plot)
            if profile is None:
                continue
            mean = np.nanmean(profile, axis=0)
            sem = np.nanstd(profile, axis=0, ddof=1) / np.sqrt(profile.shape[0])
            # Bug: 06f7faf — center x-axis on the go cue (t=0 at go).
            center = r.get("go_align_center_per_scale", {}).get(pert_idx_for_plot, 0)
            x = np.arange(mean.shape[0]) - center
            fig.add_trace(
                go.Scatter(
                    x=x, y=mean, mode="lines",
                    line=dict(color=color_map[role], width=2),
                    name=f"{arch} {role}",
                    legendgroup=role,
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                    fill="toself",
                    fillcolor=color_map[role],
                    opacity=0.15,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=role,
                ),
                row=1, col=col,
            )
    fig.update_layout(
        title=(f"Forward velocity profile — adversarial vs baseline "
               f"(at pert_scale={HEADLINE_PERT_SCALE}) — 410d7ac corrected MVP"),
        template="plotly_white",
        legend_title="Training method",
    )
    fig.update_xaxes(title_text="time step (go cue at 0)", row=1, col=1)
    fig.update_xaxes(title_text="time step (go cue at 0)", row=1, col=2)
    fig.update_yaxes(title_text="forward velocity (toward target)", row=1, col=1)

    # Annotate Δv on each subplot
    for col, (arch, _, _) in enumerate(ARCH_PAIRS, start=1):
        if arch in delta_v_by_arch:
            dv0 = delta_v_by_arch[arch][HEADLINE_PERT_SCALE]
            axis_suffix = "" if col == 1 else str(col)
            fig.add_annotation(
                xref=f"x{axis_suffix} domain",
                yref=f"y{axis_suffix} domain",
                x=0.95, y=0.95, xanchor="right", yanchor="top",
                text=f"Δv @ {HEADLINE_PERT_SCALE} = {dv0['mean']:+.3f} ± {dv0['sem']:.3f}",
                showarrow=False,
                font=dict(size=12),
            )

    spec = {
        "experiment": EXPERIMENT,
        "topic": "delta_v_signature",
        "eval_pert_scales": list(EVAL_PERT_SCALES),
        "headline_pert_scale": HEADLINE_PERT_SCALE,
        "n_replicates": N_REPLICATES,
        "seed_eval": SEED_EVAL,
        "runs": list(RUNS.keys()),
        "description": (
            "Corrected 4-model Δv comparison: forward-velocity profile of "
            "baseline (warmup-only) vs adversarial (warmup + 500 adversarial "
            "batches) for both LinearController (regulator) and "
            "LinearTrackerController. Δv = (peak_v(adv) - peak_v(base)) / "
            "peak_v(base), evaluated at pert_scale=0 (the controller's "
            "designed-against threat is the only structural difference)."
        ),
    }
    save_figure(
        fig=fig, spec=spec,
        package="rlrmp",
        experiment=EXPERIMENT,
        topic="delta_v_signature",
        extra_packages=["rlrmp"],
    )
    print(f"\nFigure saved to results/{EXPERIMENT}/figures/delta_v_signature/")

    return summary


if __name__ == "__main__":
    main()

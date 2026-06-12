"""Empirical peak-forward-velocity Δv on flavor-(b) trained controllers vs baseline.

Bug: c723082 — fills the "operating-point sensitivity" gap flagged in
``results/72fb8d9/synthesis.md`` §8.0.1 / §5.3. Empirical analogue of
the analytical Δv comparison: rolls out trained controllers on a clean
(no-perturbation) canonical reach and measures peak forward velocity, then
compares flavor-(b) trained controllers against the no-perturbation GRU
baseline.

Δv definition (matches ``rlrmp.analysis.hinf_riccati.compute_velocity_inflation``,
issue ``f90bf74``)::

    delta_v_percent =
        (peak_forward_velocity_target - peak_forward_velocity_baseline)
        / peak_forward_velocity_baseline * 100

where ``peak_forward_velocity = max_t (vel_t · unit(target - init))`` —
signed projection of velocity onto the reach axis (positive = toward target).

Pipeline mirrors ``scripts/run_induced_gain_flavor_b.py`` for loading; uses
``feedbax.task.eval_trials`` for the actual rollout (no synthetic w-channel,
unlike the induced-gain analyser).

Usage:
    uv run python scripts/run_peak_velocity_flavor_b.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np


from feedbax._io import load_with_hyperparameters

from rlrmp.intervention_compat import (
    swap_plant_intervenor_to_dynamics_matrix,
    swap_task_intervention_to_dynamics_matrix,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.paths import mkdir_p, run_artifact_dir, run_spec_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Group registry
# =============================================================================


@dataclass(frozen=True)
class FlavorBGroupSpec:
    name: str
    eta_max: float
    seed: int
    run_dir_subpath: str  # e.g. "eta0.03__seed_0"


def build_flavor_b_groups() -> tuple[FlavorBGroupSpec, ...]:
    out = []
    for eta in (0.03, 0.10, 0.30):
        for seed in (0, 1, 2):
            sub = f"eta{eta:.2f}__seed_{seed}"
            out.append(
                FlavorBGroupSpec(
                    name=f"flavor_b_eta{eta:.2f}__seed_{seed}",
                    eta_max=eta,
                    seed=seed,
                    run_dir_subpath=sub,
                )
            )
    return tuple(out)


FLAVOR_B_GROUPS: tuple[FlavorBGroupSpec, ...] = build_flavor_b_groups()


# Baselines (single-replicate). ``baseline_standard_12k`` is the canonical
# no-perturbation GRU baseline (12000 warmup batches, n_adversary_batches=0).
# ``vanilla_single``/``vanilla_pop5`` are vanilla_RNN (different architecture);
# included for context but architecturally not directly comparable to flavor-B.
BASELINES = {
    "baseline_standard_12k": {
        "artifact_subpath": "baseline/standard_12k",
        "model_filename": "adversarial_model.eqx",  # n_adv=0 means this is just warmup-end
        "kind": "minimax_no_adv",
        "hidden_type": "gru",
    },
    "vanilla_single": {
        "artifact_subpath": "vanilla_single",
        "model_filename": "warmup_model.eqx",  # adversarial phase didn't run
        "kind": "warmup_only",
        "hidden_type": "vanilla_rnn",
    },
}


_BUILD_HPS_DEFAULTS = {
    "n_warmup_batches": 2000,
    "n_adversary_batches": 5000,
    "controller_lr": 0.0001,
    "loss_update_enabled": False,
    "loss_update_ratio": 0.5,
    "hidden_type": "gru",
    "sisu_gating": "additive",
}


# =============================================================================
# Loading helpers
# =============================================================================


def _select_replicate(model, replicate_idx: int):
    """Index a specific replicate from the leading n_replicates axis."""
    net = model.nodes["net"]
    cell = net.hidden
    if hasattr(cell, "weight_hh"):
        ref = cell.weight_hh
    elif hasattr(cell, "_cell") and hasattr(cell._cell, "weight_hh"):
        ref = cell._cell.weight_hh
    else:
        for leaf in jt.leaves(net):
            if hasattr(leaf, "ndim") and leaf.ndim >= 2:
                ref = leaf
                break
        else:
            return model
    n_replicates = int(ref.shape[0])
    if n_replicates <= 1:
        return _squeeze_replicate_axis(model)

    def _pick(x):
        if hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == n_replicates:
            return x[replicate_idx]
        return x

    return jt.map(_pick, model, is_leaf=eqx.is_array)


def _squeeze_replicate_axis(model):
    return jt.map(
        lambda x: x[0] if (hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == 1) else x,
        model,
        is_leaf=eqx.is_array,
    )


def _count_replicates(model) -> int:
    net = model.nodes["net"]
    cell = net.hidden
    if hasattr(cell, "weight_hh"):
        ref = cell.weight_hh
    elif hasattr(cell, "_cell") and hasattr(cell._cell, "weight_hh"):
        ref = cell._cell.weight_hh
    else:
        for leaf in jt.leaves(net):
            if hasattr(leaf, "ndim") and leaf.ndim >= 2:
                ref = leaf
                break
        else:
            return 1
    if ref.ndim <= 2:
        return 1
    return int(ref.shape[0])


def _build_hps_from_run_json(run_json: dict):
    from rlrmp.train.minimax import build_hps as build_hps_fn

    cli = run_json.get("cli_args", {})
    schedule = run_json.get("training_schedule", {})
    ctrl = run_json.get("controller_params", {})

    cfg = {**_BUILD_HPS_DEFAULTS}
    cfg["n_warmup_batches"] = int(cli.get("n-warmup-batches",
                                          schedule.get("n_warmup_batches", cfg["n_warmup_batches"])))
    cfg["n_adversary_batches"] = int(cli.get("n-adversary-batches",
                                             schedule.get("n_adversary_batches", cfg["n_adversary_batches"])))
    cfg["controller_lr"] = float(ctrl.get("controller_lr", cfg["controller_lr"]))
    cfg["loss_update_enabled"] = bool(schedule.get("loss_update_enabled", cfg["loss_update_enabled"]))
    cfg["hidden_type"] = ctrl.get("hidden_type", cfg["hidden_type"])
    cfg["sisu_gating"] = schedule.get("sisu_gating", cfg["sisu_gating"])
    args_ns = argparse.Namespace(**cfg)
    return build_hps_fn(args_ns)


def _build_hps_from_baseline_config(config: dict):
    """Build hps for a baseline config.json (train_minimax.py format)."""
    from rlrmp.train.minimax import build_hps as build_hps_fn

    cfg_filtered = {
        k: v for k, v in config.items()
        if k not in ("git", "output_dir", "warmup_model", "jax_cache_dir",
                     "jax_explain_cache_misses", "checkpoint", "checkpoint_every",
                     "resume", "streaming_loss")
    }
    # Older configs (vanilla_single, pre-additive-sisu) lack newer fields.
    # Fill in defaults to avoid AttributeError when build_hps reads args.foo.
    cfg_filtered.setdefault("sisu_gating", "additive")
    cfg_filtered.setdefault("hidden_type", "vanilla_rnn")
    cfg_filtered.setdefault("loss_update_enabled", False)
    cfg_filtered.setdefault("loss_update_ratio", 0.5)
    args_ns = argparse.Namespace(**cfg_filtered)
    return build_hps_fn(args_ns)


def load_flavor_b_model(group: FlavorBGroupSpec, flavor_b_root: Path,
                         run_json_root: Path) -> tuple[object, dict, int]:
    """Load the ensembled adversarial_model.eqx for one flavor-B config."""
    run_dir = flavor_b_root / group.run_dir_subpath
    eqx_path = run_dir / "adversarial_model.eqx"
    if not eqx_path.exists():
        raise FileNotFoundError(f"adversarial_model.eqx not found in {run_dir}")

    run_json_path = run_json_root / f"{group.name}" / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(f"run.json not found at {run_json_path}")
    run_json_data = json.loads(run_json_path.read_text())

    hps = _build_hps_from_run_json(run_json_data)
    mass = float(hps.model.effector_mass)

    def _setup(key, **kwargs):
        warmup_model = setup_task_model_pair(hps, key=key).model
        adv_model = jt.map(
            lambda m: swap_plant_intervenor_to_dynamics_matrix(
                m, PLANT_INTERVENOR_LABEL, mass=mass,
            ),
            warmup_model,
            is_leaf=lambda x: x is not None and hasattr(x, "nodes")
                              and hasattr(x, "input_ports"),
        )
        return adv_model

    model, _ = load_with_hyperparameters(eqx_path, setup_func=_setup)
    n_reps = _count_replicates(model)
    return model, run_json_data, n_reps


def load_baseline_model(name: str, baseline_root: Path) -> tuple[object, dict, int, object]:
    """Load a baseline model (single replicate). Returns (model, config, n_reps, task)."""
    info = BASELINES[name]
    run_dir = baseline_root / info["artifact_subpath"]
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with open(config_path) as f:
        config = json.load(f)
    hps = _build_hps_from_baseline_config(config)

    eqx_path = run_dir / info["model_filename"]
    if not eqx_path.exists():
        raise FileNotFoundError(eqx_path)

    def _setup(key, **kwargs):
        return setup_task_model_pair(hps, key=key).model

    model, _ = load_with_hyperparameters(eqx_path, setup_func=_setup)
    n_reps = _count_replicates(model)
    if n_reps == 1:
        model = _squeeze_replicate_axis(model)

    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    return model, config, n_reps, pair.task


# =============================================================================
# Trial-spec setup: canonical 15 cm forward reach
# =============================================================================


def _set_sisu_and_pert(val_trials, sisu_val: float, pert_scale: float = 0.0):
    """Set SISU level and perturbation scale on validation trials."""
    n_trials = val_trials.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    new_trials = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val_trials,
        jnp.full((n_trials,), pert_scale),
    )
    new_trials = eqx.tree_at(
        lambda t: t.inputs["sisu"],
        new_trials,
        jnp.full((n_trials,), sisu_val),
    )
    return new_trials


def _build_canonical_reach_trial_specs(task, *, init_pos, target_pos,
                                        sisu: float, pert_scale: float):
    """Set SISU and pert_scale on validation trials.

    Uses the existing center-out validation trials (8 reaches at 0.5m). Per-trial
    peak forward velocity is computed against each trial's own reach axis, so
    averaging over the 8 trials gives an isotropic Δv estimate.

    Note: ``init_pos``/``target_pos`` are kept in the call signature for
    compatibility with the run.json spec (canonical 15 cm reach is the
    induced-gain analyser's setting on the linearised plant); for the actual
    trained-controller rollout we use the validation trials' geometry, since
    overriding only the target without updating task inputs and init states
    creates inconsistent trials. Bug: c723082.
    """
    val_trials = task.validation_trials
    return _set_sisu_and_pert(val_trials, sisu_val=sisu, pert_scale=pert_scale)


# =============================================================================
# Rollout + peak forward velocity computation
# =============================================================================


def eval_ensemble_on_trials(task, model, trial_specs, *, key, n_replicates: int):
    """Rollout `model` on trial_specs. Handles both ensembled (n_rep > 1) and
    single-replicate models. Returns states with leading replicate dim if
    n_replicates > 1, else without."""
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    if n_replicates <= 1:
        keys = jr.split(key, n_trials)
        return task.eval_trials(model, trial_specs, keys)

    def _is_batched(x):
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_replicates

    models_arrays, models_other = eqx.partition(model, _is_batched)

    def _eval_one(model_arrays, model_other, rep_key):
        rep_model = eqx.combine(model_arrays, model_other)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(rep_model, trial_specs, keys)

    rep_keys = jr.split(key, n_replicates)
    states = eqx.filter_vmap(
        _eval_one,
        in_axes=(0, None, 0),
    )(models_arrays, models_other, rep_keys)
    return states


def compute_peak_forward_velocity(states, trial_specs, *, has_rep_dim: bool):
    """Compute peak forward velocity (signed projection onto reach axis).

    Returns:
        Dict with "peak_forward_velocity" (per-replicate, per-trial),
        "peak_lateral_velocity", "peak_speed".
        If has_rep_dim, arrays are (n_rep, n_trials); else (n_trials,).
    """
    pos = states.mechanics.effector.pos  # (..., n_trials, n_steps, 2)
    vel = states.mechanics.effector.vel

    target_key = list(trial_specs.targets.keys())[0]
    goal_seq = trial_specs.targets[target_key].value  # (n_trials, n_steps, 2)
    goal = goal_seq[:, -1, :]  # (n_trials, 2)
    go_idx = trial_specs.timeline.epoch_bounds[:, 2]  # (n_trials,)

    if has_rep_dim:
        n_rep, n_trials, n_steps, _ = pos.shape
        t = jnp.arange(n_steps)
        after_go = t[None, None, :] >= go_idx[None, :, None]  # (1, n_trials, n_steps)

        def _get_init_pos_rep(pos_rep):
            return jax.vmap(lambda p, idx: p[idx])(pos_rep, go_idx)
        init_pos = jax.vmap(_get_init_pos_rep)(pos)  # (n_rep, n_trials, 2)

        direction = goal[None, :, :] - init_pos  # (n_rep, n_trials, 2)
        dnorm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        dunit = direction / jnp.maximum(dnorm, 1e-12)  # (n_rep, n_trials, 2)

        v_forward_t = jnp.sum(vel * dunit[:, :, None, :], axis=-1)  # (n_rep, n_trials, n_steps)
        # zero out before go-cue
        v_forward_t = jnp.where(after_go, v_forward_t, 0.0)
        peak_forward = jnp.max(v_forward_t, axis=-1)

        # Lateral component
        v_along = v_forward_t[..., None] * dunit[:, :, None, :]
        v_lateral = vel - v_along
        v_lateral_mag = jnp.linalg.norm(v_lateral, axis=-1)
        v_lateral_mag = jnp.where(after_go, v_lateral_mag, 0.0)
        peak_lateral = jnp.max(v_lateral_mag, axis=-1)

        speed = jnp.linalg.norm(vel, axis=-1)
        speed = jnp.where(after_go, speed, 0.0)
        peak_speed = jnp.max(speed, axis=-1)
    else:
        n_trials, n_steps, _ = pos.shape
        t = jnp.arange(n_steps)
        after_go = t[None, :] >= go_idx[:, None]

        init_pos = jax.vmap(lambda p, idx: p[idx])(pos, go_idx)  # (n_trials, 2)
        direction = goal - init_pos
        dnorm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        dunit = direction / jnp.maximum(dnorm, 1e-12)

        v_forward_t = jnp.sum(vel * dunit[:, None, :], axis=-1)  # (n_trials, n_steps)
        v_forward_t = jnp.where(after_go, v_forward_t, 0.0)
        peak_forward = jnp.max(v_forward_t, axis=-1)

        v_along = v_forward_t[..., None] * dunit[:, None, :]
        v_lateral = vel - v_along
        v_lateral_mag = jnp.linalg.norm(v_lateral, axis=-1)
        v_lateral_mag = jnp.where(after_go, v_lateral_mag, 0.0)
        peak_lateral = jnp.max(v_lateral_mag, axis=-1)

        speed = jnp.linalg.norm(vel, axis=-1)
        speed = jnp.where(after_go, speed, 0.0)
        peak_speed = jnp.max(speed, axis=-1)

    return {
        "peak_forward_velocity": np.array(peak_forward),
        "peak_lateral_velocity": np.array(peak_lateral),
        "peak_speed": np.array(peak_speed),
    }


# =============================================================================
# Per-config evaluation
# =============================================================================


def eval_one_flavor_b_group(
    group: FlavorBGroupSpec,
    *,
    flavor_b_root: Path,
    run_json_root: Path,
    init_pos: np.ndarray,
    target_pos: np.ndarray,
    sisu: float,
    pert_scale: float,
    eval_seed: int,
) -> dict:
    """Roll out a single flavor-B config (5 internal replicates) and compute
    per-replicate peak forward velocity."""
    t0 = time.time()
    model, run_json_data, n_reps = load_flavor_b_model(group, flavor_b_root, run_json_root)
    hps = _build_hps_from_run_json(run_json_data)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    # Swap task's intervention-spec params to DynamicsMatrixPerturbParams so
    # validation_trials.intervene[label] matches the model's intervenor type
    # (mirrors train_minimax.py line 792, Bug: c723082).
    task = swap_task_intervention_to_dynamics_matrix(pair.task, PLANT_INTERVENOR_LABEL)

    trial_specs = _build_canonical_reach_trial_specs(
        task, init_pos=init_pos, target_pos=target_pos,
        sisu=sisu, pert_scale=pert_scale,
    )
    states = eval_ensemble_on_trials(
        task, model, trial_specs,
        key=jr.PRNGKey(eval_seed), n_replicates=n_reps,
    )
    metrics = compute_peak_forward_velocity(states, trial_specs, has_rep_dim=(n_reps > 1))

    # per-replicate, per-trial → per-replicate (mean over trials)
    pv = metrics["peak_forward_velocity"]  # (n_rep, n_trials) or (n_trials,)
    pl = metrics["peak_lateral_velocity"]
    ps = metrics["peak_speed"]
    if pv.ndim == 2:
        rep_pv = pv.mean(axis=-1).tolist()
        rep_pl = pl.mean(axis=-1).tolist()
        rep_ps = ps.mean(axis=-1).tolist()
    else:
        rep_pv = [float(pv.mean())]
        rep_pl = [float(pl.mean())]
        rep_ps = [float(ps.mean())]

    return {
        "group": group.name,
        "eta_max": group.eta_max,
        "seed": group.seed,
        "n_replicates": int(n_reps),
        "per_replicate_peak_forward_velocity": rep_pv,
        "per_replicate_peak_lateral_velocity": rep_pl,
        "per_replicate_peak_speed": rep_ps,
        "wall_time_s": time.time() - t0,
    }


def eval_baseline(
    name: str,
    *,
    baseline_root: Path,
    init_pos: np.ndarray,
    target_pos: np.ndarray,
    sisu: float,
    pert_scale: float,
    eval_seed: int,
) -> dict:
    """Roll out a baseline (single-replicate) and compute peak forward velocity."""
    t0 = time.time()
    info = BASELINES[name]
    model, config, n_reps, task = load_baseline_model(name, baseline_root)
    trial_specs = _build_canonical_reach_trial_specs(
        task, init_pos=init_pos, target_pos=target_pos,
        sisu=sisu, pert_scale=pert_scale,
    )
    states = eval_ensemble_on_trials(
        task, model, trial_specs,
        key=jr.PRNGKey(eval_seed), n_replicates=n_reps,
    )
    has_rep = n_reps > 1
    metrics = compute_peak_forward_velocity(states, trial_specs, has_rep_dim=has_rep)
    pv = metrics["peak_forward_velocity"]
    pl = metrics["peak_lateral_velocity"]
    ps = metrics["peak_speed"]
    if pv.ndim == 2:
        rep_pv = pv.mean(axis=-1).tolist()
        rep_pl = pl.mean(axis=-1).tolist()
        rep_ps = ps.mean(axis=-1).tolist()
    else:
        rep_pv = [float(pv.mean())]
        rep_pl = [float(pl.mean())]
        rep_ps = [float(ps.mean())]
    return {
        "group": name,
        "kind": "baseline",
        "n_replicates": int(n_reps),
        "hidden_type": info["hidden_type"],
        "model_filename": info["model_filename"],
        "per_replicate_peak_forward_velocity": rep_pv,
        "per_replicate_peak_lateral_velocity": rep_pl,
        "per_replicate_peak_speed": rep_ps,
        "wall_time_s": time.time() - t0,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_flavor_b_root = (
        "/Users/mll/Main/10 Projects/10 PhD/rlrmp/_artifacts/c723082/runs"
    )
    default_run_json_root = (
        "/Users/mll/Main/10 Projects/10 PhD/rlrmp/results/c723082/runs"
    )
    default_baseline_root = (
        "/Users/mll/Main/10 Projects/10 PhD/rlrmp/_artifacts/e81f491/runs"
    )
    parser.add_argument("--flavor-b-root", type=str, default=default_flavor_b_root)
    parser.add_argument("--run-json-root", type=str, default=default_run_json_root)
    parser.add_argument("--baseline-root", type=str, default=default_baseline_root)
    parser.add_argument("--init-x", type=float, default=0.0)
    parser.add_argument("--init-y", type=float, default=0.0)
    parser.add_argument("--target-x", type=float, default=0.15)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--sisu", type=float, default=0.5)
    parser.add_argument("--pert-scale", type=float, default=0.0,
                        help="Plant disturbance scale (0 = clean reach).")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--limit-baselines", action="store_true",
                        help="Only run baseline_standard_12k (skip vanilla_single).")
    parser.add_argument("--exp", type=str, default="part2_5")
    parser.add_argument("--run", type=str, default="peak_velocity_flavor_b")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    flavor_b_root = Path(args.flavor_b_root)
    run_json_root = Path(args.run_json_root)
    baseline_root = Path(args.baseline_root)
    if not flavor_b_root.exists():
        raise FileNotFoundError(flavor_b_root)
    if not run_json_root.exists():
        raise FileNotFoundError(run_json_root)
    if not baseline_root.exists():
        raise FileNotFoundError(baseline_root)

    init_pos = np.array([args.init_x, args.init_y], dtype=np.float32)
    target_pos = np.array([args.target_x, args.target_y], dtype=np.float32)

    artifact_root = mkdir_p(run_artifact_dir(args.exp, args.run))
    spec_dir = mkdir_p(run_spec_dir(args.exp, args.run))

    # ---------- Baselines ----------
    baseline_results: list[dict] = []
    baselines_to_run = ["baseline_standard_12k"]
    if not args.limit_baselines:
        baselines_to_run.append("vanilla_single")

    for bname in baselines_to_run:
        logger.info("Baseline %s ...", bname)
        try:
            res = eval_baseline(
                bname,
                baseline_root=baseline_root,
                init_pos=init_pos, target_pos=target_pos,
                sisu=args.sisu, pert_scale=args.pert_scale,
                eval_seed=args.eval_seed,
            )
            logger.info("  %s: peak_forward_v=%s",
                        bname, res["per_replicate_peak_forward_velocity"])
            baseline_results.append(res)
        except Exception as e:
            logger.error("  %s FAILED: %s: %s", bname, type(e).__name__, e)
            baseline_results.append({
                "group": bname, "kind": "baseline",
                "_error": f"{type(e).__name__}: {e}",
            })

    # ---------- Flavor-B groups ----------
    groups_to_run = FLAVOR_B_GROUPS
    if args.limit_groups is not None:
        groups_to_run = groups_to_run[: args.limit_groups]

    flavor_b_results: list[dict] = []
    for gi, group in enumerate(groups_to_run):
        logger.info("[%d/%d] Flavor-B %s ...", gi + 1, len(groups_to_run), group.name)
        try:
            res = eval_one_flavor_b_group(
                group,
                flavor_b_root=flavor_b_root,
                run_json_root=run_json_root,
                init_pos=init_pos, target_pos=target_pos,
                sisu=args.sisu, pert_scale=args.pert_scale,
                eval_seed=args.eval_seed,
            )
            logger.info("  peak_forward_v=%s", res["per_replicate_peak_forward_velocity"])
            flavor_b_results.append(res)
        except Exception as e:
            logger.error("  FAILED: %s: %s", type(e).__name__, e)
            import traceback; traceback.print_exc()
            flavor_b_results.append({
                "group": group.name,
                "_error": f"{type(e).__name__}: {e}",
            })

    # ---------- Aggregate ----------
    # Δv vs baseline_standard_12k. Baseline aggregation: mean across all baseline
    # replicates (5 vmapped reps × per-replicate mean over 8 center-out trials),
    # i.e. one scalar baseline peak_forward_velocity.
    bsl = next((b for b in baseline_results
                if b.get("group") == "baseline_standard_12k" and "_error" not in b),
               None)
    if bsl is not None:
        bsl_per_rep = np.array(bsl["per_replicate_peak_forward_velocity"])
        bsl_pv_mean = float(bsl_per_rep.mean())
        bsl_pv_sd = float(bsl_per_rep.std(ddof=1)) if bsl_per_rep.size > 1 else 0.0
    else:
        bsl_pv_mean = None
        bsl_pv_sd = None

    # Per-(eta, seed) aggregates first.
    aggregates_per_eta_seed: list[dict] = []
    for r in flavor_b_results:
        if "_error" in r:
            continue
        pvs = np.array(r["per_replicate_peak_forward_velocity"])
        row = {
            "eta_max": r["eta_max"],
            "seed": r["seed"],
            "n_replicates": len(pvs),
            "peak_forward_velocity_mean": float(pvs.mean()),
            "peak_forward_velocity_sd": float(pvs.std(ddof=1)) if len(pvs) > 1 else 0.0,
        }
        if bsl_pv_mean is not None and bsl_pv_mean > 0:
            dvp = (pvs - bsl_pv_mean) / bsl_pv_mean * 100.0
            row["delta_v_percent_mean"] = float(dvp.mean())
            row["delta_v_percent_sd"] = float(dvp.std(ddof=1)) if len(pvs) > 1 else 0.0
        aggregates_per_eta_seed.append(row)

    aggregates_per_eta: dict[float, dict] = {}
    for eta in (0.03, 0.10, 0.30):
        eta_pvs = []
        for r in flavor_b_results:
            if r.get("eta_max") == eta and "_error" not in r:
                eta_pvs.extend(r["per_replicate_peak_forward_velocity"])
        if eta_pvs:
            mean = float(np.mean(eta_pvs))
            sd = float(np.std(eta_pvs, ddof=1)) if len(eta_pvs) > 1 else 0.0
            agg = {
                "eta_max": eta,
                "n": len(eta_pvs),
                "mean_peak_forward_velocity": mean,
                "sd_peak_forward_velocity": sd,
                "all_values": eta_pvs,
            }
            if bsl_pv_mean is not None and bsl_pv_mean > 0:
                delta_v_pct = (np.array(eta_pvs) - bsl_pv_mean) / bsl_pv_mean * 100.0
                agg["delta_v_percent_mean"] = float(delta_v_pct.mean())
                agg["delta_v_percent_sd"] = float(delta_v_pct.std(ddof=1)) if len(eta_pvs) > 1 else 0.0
                agg["delta_v_percent_per_replicate"] = delta_v_pct.tolist()
                agg["baseline_peak_forward_velocity"] = float(bsl_pv_mean)
                agg["baseline_peak_forward_velocity_sd"] = float(bsl_pv_sd) if bsl_pv_sd is not None else None
            aggregates_per_eta[eta] = agg

    summary = {
        "experiment": args.exp,
        "run": args.run,
        "tracking_issue": "c723082",
        "delta_v_definition": (
            "delta_v_percent = (pv_target - pv_baseline) / pv_baseline * 100, "
            "where pv = peak forward velocity (signed projection of velocity onto "
            "unit(target - init) reach axis); matches "
            "rlrmp.analysis.hinf_riccati.compute_velocity_inflation (issue f90bf74)."
        ),
        "baseline_for_delta_v": "baseline_standard_12k",
        "baseline_peak_forward_velocity_mean": bsl_pv_mean,
        "baseline_peak_forward_velocity_sd": bsl_pv_sd,
        "init_pos": list(init_pos.tolist()),
        "target_pos": list(target_pos.tolist()),
        "sisu": float(args.sisu),
        "pert_scale": float(args.pert_scale),
        "eval_seed": int(args.eval_seed),
        "baselines": baseline_results,
        "flavor_b_groups": flavor_b_results,
        "flavor_b_aggregates_per_eta_seed": aggregates_per_eta_seed,
        "flavor_b_aggregates_per_eta": {f"{k:.2f}": v for k, v in aggregates_per_eta.items()},
    }

    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary.json to %s", artifact_root / "summary.json")

    run_json = {
        "experiment": args.exp,
        "run": args.run,
        "spec_kind": "peak_velocity_analysis",
        "subject": "flavor_b_trained_controllers",
        "tracking_issue": "c723082",
        "metric": "peak_forward_velocity",
        "delta_v_baseline": "baseline_standard_12k",
        "init_pos": list(init_pos.tolist()),
        "target_pos": list(target_pos.tolist()),
        "sisu": float(args.sisu),
        "pert_scale": float(args.pert_scale),
        "eval_seed": int(args.eval_seed),
        "input_artifacts": {
            "flavor_b_root": str(flavor_b_root),
            "run_json_root": str(run_json_root),
            "baseline_root": str(baseline_root),
        },
    }
    (spec_dir / "run.json").write_text(json.dumps(run_json, indent=2))
    logger.info("Saved run.json to %s", spec_dir / "run.json")

    # Log headline
    logger.info("=" * 60)
    logger.info("HEADLINE Δv (vs baseline_standard_12k mean=%.4f m/s, SD=%.4f):",
                bsl_pv_mean if bsl_pv_mean is not None else float("nan"),
                bsl_pv_sd if bsl_pv_sd is not None else float("nan"))
    for eta, agg in aggregates_per_eta.items():
        if "delta_v_percent_mean" in agg:
            logger.info("  eta_max=%.2f: Δv = %+.2f%% ± %.2f%% (n=%d, peak_v=%.4f m/s)",
                        eta, agg["delta_v_percent_mean"],
                        agg["delta_v_percent_sd"], agg["n"],
                        agg["mean_peak_forward_velocity"])

    logger.info("Done.")


if __name__ == "__main__":
    main()

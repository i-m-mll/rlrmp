"""Induced-gain analyser on flavor-(b) (LinearDynamicsAdversary) trained checkpoints.

Bug: 74bfd86 — second cross-method run, focused on the 9 flavor-(b) trained
configurations from issue ``c723082`` (eta_max in {0.03, 0.10, 0.30} x seed
in {0, 1, 2}, with 5 internal vmap replicates per config = 45 controllers).

Mirrors ``scripts/run_induced_gain_part2_5.py``'s structure (loader, network-only
adapter, channel set, output schema). Differences:

- **Per-replicate analysis**: each of the 5 replicates is analysed and saved
  individually. ``summary.json`` rolls per-replicate values up to per-config
  medians + MAD, applying the §5.2 hygiene rule (flag gamma > 10x from group
  median).
- **Canonical post-probe parameters**: ``rtol=1e-6``, ``n=200`` horizon,
  ``Q_f=1.0`` (cost_schedule_from_spec default). These are the values
  established as trustworthy after the recent probe diagnosis (commit
  ``855c33fd`` merged via ``d193d9e`` on main).
- **Group registry built from flavor-(b) sweep**: 9 configs of the form
  ``flavor_b_eta{X}__seed_{Y}``.

Pre-registered headline metric: ``gamma_sd x qr_cost`` — induced gain on the
``structural_da`` w channel against the cost-matched z. Auxiliary channels
(additive_force, sensory_perturbation x qr_cost) are also reported for
completeness.

Usage:
    uv run python scripts/run_induced_gain_flavor_b.py
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

from rlrmp.analysis.hinf_riccati import (
    CostSpec,
    PlantLinearization,
    cost_schedule_from_spec,
    find_gamma_star,
    linearize_pointmass,
)
from rlrmp.analysis.induced_gain import (
    Controller,
    W_ADDITIVE_FORCE,
    W_SENSORY_PERTURBATION,
    W_STRUCTURAL_DA,
    Z_QR_COST,
    induced_gain,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p, run_artifact_dir, run_spec_dir
from rlrmp.intervention_compat import swap_plant_intervenor_to_dynamics_matrix
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL

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


def build_groups() -> tuple[FlavorBGroupSpec, ...]:
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


GROUPS: tuple[FlavorBGroupSpec, ...] = build_groups()


# Defaults for any train_minimax.build_hps fields not surfaced in run.json's
# cli_args. Match the train_minimax.py defaults at the time of the run.
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


def _cast_to_float64(tree):
    def _cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float64)
        return x
    return jt.map(_cast, tree, is_leaf=eqx.is_array)


def _cast_to_float32(tree):
    def _cast(x):
        if hasattr(x, "dtype"):
            if jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.float32)
            if jnp.issubdtype(x.dtype, jnp.signedinteger):
                if x.dtype == jnp.int64:
                    return x.astype(jnp.int32)
        return x
    return jt.map(_cast, tree, is_leaf=eqx.is_array)


def _build_hps_from_run_json(run_json: dict):
    """Construct train_minimax build_hps namespace from flavor-B run.json."""
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


def load_flavor_b_model(
    group: FlavorBGroupSpec,
    flavor_b_root: Path,
    run_json_root: Path,
) -> tuple[object, dict, int]:
    """Load the ensembled (n_reps=5) adversarial_model.eqx for one flavor-B config.

    Returns ``(model_ensembled, run_json, n_reps)``.
    """
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
        # Build the warmup-phase template, then swap the plant intervenor to
        # DynamicsMatrixPerturb to match what was saved during the adversarial
        # phase (mirrors train_minimax.py lines 773-790, Bug: c723082).
        warmup_model = setup_task_model_pair(hps, key=key).model
        adv_model = jt.map(
            lambda m: swap_plant_intervenor_to_dynamics_matrix(
                m, PLANT_INTERVENOR_LABEL, mass=mass,
            ),
            warmup_model,
            is_leaf=lambda x: x is not None and hasattr(x, "nodes")
                              and hasattr(x, "input_ports"),
        )
        return _cast_to_float32(adv_model)

    model, _ = load_with_hyperparameters(eqx_path, setup_func=_setup)

    net = model.nodes["net"]
    cell = net.hidden
    if hasattr(cell, "weight_hh"):
        ref = cell.weight_hh
    else:
        ref = cell._cell.weight_hh
    n_reps = int(ref.shape[0]) if ref.ndim > 2 else 1

    return model, run_json_data, n_reps


# =============================================================================
# Network-only controller adapter
# =============================================================================


@dataclass(frozen=True)
class _NetworkController:
    net: object
    state_template: object
    net_state_index: object
    net_state_template: object
    task_input: jnp.ndarray
    n_obs: int
    delay: int
    h0_flat: jnp.ndarray
    net_treedef: object
    net_leaf_shapes: tuple
    net_leaf_sizes: tuple
    target_pos: jnp.ndarray
    key: jnp.ndarray

    def initial_state(self):
        return self.h0_flat

    def _split(self, h):
        net_size = sum(self.net_leaf_sizes)
        net_part = h[:net_size]
        queue_part = h[net_size:]
        leaves = []
        offset = 0
        for shape, size in zip(self.net_leaf_shapes, self.net_leaf_sizes):
            leaves.append(net_part[offset:offset + size].reshape(shape))
            offset += size
        net_state = jt.unflatten(self.net_treedef, leaves)
        if self.delay > 0:
            queue = queue_part.reshape(self.delay, self.n_obs)
        else:
            queue = jnp.zeros((0, self.n_obs), dtype=jnp.float64)
        return net_state, queue

    def _join(self, net_state, queue):
        leaves = jt.leaves(net_state)
        flat_parts = [jnp.asarray(leaf, dtype=jnp.float64).reshape(-1) for leaf in leaves]
        net_part = jnp.concatenate(flat_parts, axis=0) if flat_parts else jnp.zeros((0,), dtype=jnp.float64)
        queue_flat = queue.reshape(-1).astype(jnp.float64)
        return jnp.concatenate([net_part, queue_flat], axis=0)

    def step(self, h, sensory_obs, t):
        pos_gc = sensory_obs[:2]
        vel = sensory_obs[2:4]
        pos_abs = pos_gc + self.target_pos
        obs_abs = jnp.concatenate([pos_abs, vel], axis=0)

        net_state, queue = self._split(h)

        if self.delay > 0:
            delayed_obs = queue[0]
            new_queue = jnp.concatenate([queue[1:], obs_abs[None, :]], axis=0)
        else:
            delayed_obs = obs_abs
            new_queue = queue

        state_leaves, state_treedef = jt.flatten(self.state_template)
        state = jt.unflatten(state_treedef, state_leaves)
        state = state.set(self.net_state_index, net_state)

        key_t = jax.random.fold_in(self.key, t)
        net_inputs = {
            "input": self.task_input,
            "feedback": (delayed_obs[:2], delayed_obs[2:]),
        }
        outputs, state_next = self.net(net_inputs, state, key=key_t)
        u = outputs["output"]

        net_state_next = state_next.get(self.net_state_index)
        h_next = self._join(net_state_next, new_queue)
        return h_next, u


def build_network_controller(
    model,
    *,
    target_pos: jnp.ndarray,
    sisu: float = 0.5,
    key: jnp.ndarray = jr.PRNGKey(0),
) -> Controller:
    net = model.nodes["net"]

    full_state = model.init_state(key=jr.PRNGKey(0))
    net_state_template = full_state.get(net.state_index)

    target_pos_arr = jnp.asarray(target_pos, dtype=jnp.float32)
    task_input = jnp.concatenate(
        [
            target_pos_arr,
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.array([float(sisu)], dtype=jnp.float32),
        ],
        axis=0,
    )
    expected_task_size = int(net.input_size) - 4
    if int(task_input.shape[0]) != expected_task_size:
        raise ValueError(
            f"Constructed task_input has dim {task_input.shape[0]} but "
            f"net expects {expected_task_size} (= {net.input_size} - 4 fb)."
        )

    fb_node = model.nodes.get("feedback")
    delay = 0
    n_obs = 4
    if fb_node is not None:
        channels = fb_node.channels
        if hasattr(channels, "delay"):
            delay = int(channels.delay)
        else:
            ch_leaves = jt.leaves(channels, is_leaf=lambda x: hasattr(x, "delay"))
            if ch_leaves and hasattr(ch_leaves[0], "delay"):
                delay = int(ch_leaves[0].delay)

    leaves = jt.leaves(net_state_template)
    leaf_shapes = tuple(tuple(jnp.asarray(l).shape) for l in leaves)
    leaf_sizes = tuple(int(jnp.asarray(l).size) for l in leaves)
    treedef = jt.structure(net_state_template)
    net_flat = jnp.concatenate(
        [jnp.asarray(l, dtype=jnp.float64).reshape(-1) for l in leaves], axis=0
    )
    queue_flat = jnp.zeros((delay * n_obs,), dtype=jnp.float64)
    h0 = jnp.concatenate([net_flat, queue_flat], axis=0)

    return _NetworkController(
        net=net,
        state_template=full_state,
        net_state_index=net.state_index,
        net_state_template=net_state_template,
        task_input=task_input,
        n_obs=n_obs,
        delay=delay,
        h0_flat=h0,
        net_treedef=treedef,
        net_leaf_shapes=leaf_shapes,
        net_leaf_sizes=leaf_sizes,
        target_pos=target_pos_arr.astype(jnp.float64),
        key=key,
    )


# =============================================================================
# Main analyser
# =============================================================================


def _build_plant_and_schedule(horizon: int) -> tuple[PlantLinearization, object]:
    plant = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)
    spec = CostSpec(n_steps=horizon)
    schedule = cost_schedule_from_spec(spec, plant)
    return plant, schedule


def analyse_replicate(
    ctrl: Controller,
    *,
    plant: PlantLinearization,
    schedule,
    init_pos: jnp.ndarray,
    target_pos: jnp.ndarray,
    horizon: int,
    channels: tuple[tuple[str, str], ...],
    seed: int,
    g_star: float,
    n_restarts: int,
    max_iter: int,
    rtol: float,
) -> dict:
    out: dict = {}
    for w_ch, z_ch in channels:
        ch_label = f"{w_ch}__{z_ch}"
        t0 = time.time()
        try:
            res = induced_gain(
                plant,
                ctrl,
                init_pos=init_pos,
                target_pos=target_pos,
                horizon=horizon,
                w_channel=w_ch,
                z_channel=z_ch,
                schedule=schedule,
                methods=("power_iteration",),
                n_restarts=n_restarts,
                max_iter=max_iter,
                rtol=rtol,
                seed=seed,
            )
            pi = res["power_iteration"]
            out[ch_label] = {
                "gamma": float(pi.gamma),
                "converged": bool(pi.converged),
                "iterations": int(pi.iterations),
                "ratio_to_gstar": float(pi.gamma / g_star) if g_star > 0 else float("nan"),
                "time_s": time.time() - t0,
            }
        except Exception as e:
            out[ch_label] = {
                "error": f"{type(e).__name__}: {e}",
                "time_s": time.time() - t0,
            }
    return out


def _median_mad(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)])
    if arr.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def _hygiene_flag(values: list[float], threshold: float = 10.0) -> list[bool]:
    arr = np.asarray(values)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return [bool(not f) for f in finite]
    med = float(np.median(arr[finite]))
    flags = []
    for v in arr:
        if not np.isfinite(v) or med == 0:
            flags.append(not np.isfinite(v))
            continue
        ratio = max(v / med, med / v)
        flags.append(bool(ratio > threshold))
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_flavor_b_root = (
        "/Users/mll/Main/10 Projects/10 PhD/rlrmp/_artifacts/c723082/runs"
    )
    default_run_json_root = (
        "/Users/mll/Main/10 Projects/10 PhD/rlrmp/worktrees/"
        "feature__part2_5-flavor-b-runs/results/part2_5/runs"
    )
    parser.add_argument("--flavor-b-root", type=str, default=default_flavor_b_root)
    parser.add_argument("--run-json-root", type=str, default=default_run_json_root)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--init-x", type=float, default=0.0)
    parser.add_argument("--init-y", type=float, default=0.0)
    parser.add_argument("--target-x", type=float, default=0.15)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--sisu", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--limit-replicates", type=int, default=None)
    parser.add_argument("--exp", type=str, default="part2_5")
    parser.add_argument("--run", type=str, default="induced_gain_flavor_b")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    flavor_b_root = Path(args.flavor_b_root)
    run_json_root = Path(args.run_json_root)
    if not flavor_b_root.exists():
        raise FileNotFoundError(flavor_b_root)
    if not run_json_root.exists():
        raise FileNotFoundError(run_json_root)

    init_pos = jnp.array([args.init_x, args.init_y], dtype=jnp.float64)
    target_pos = jnp.array([args.target_x, args.target_y], dtype=jnp.float64)

    plant, schedule = _build_plant_and_schedule(args.horizon)
    logger.info("Built plant (n=%d) and cost schedule (T=%d).", plant.n, schedule.T)

    logger.info("Computing Riccati gamma_star baseline...")
    t0 = time.time()
    g_star = find_gamma_star(plant, schedule)
    logger.info("gamma_star = %.6f  (%.1fs)", g_star, time.time() - t0)

    channels = (
        (W_STRUCTURAL_DA, Z_QR_COST),       # HEADLINE
        (W_ADDITIVE_FORCE, Z_QR_COST),
        (W_SENSORY_PERTURBATION, Z_QR_COST),
    )

    groups_to_run = GROUPS
    if args.limit_groups is not None:
        groups_to_run = groups_to_run[: args.limit_groups]

    artifact_root = mkdir_p(run_artifact_dir(args.exp, args.run))
    spec_dir = mkdir_p(run_spec_dir(args.exp, args.run))

    summary_per_group: list[dict] = []

    for gi, group in enumerate(groups_to_run):
        logger.info("[%d/%d] Loading group %s", gi + 1, len(groups_to_run), group.name)
        group_artifact_dir = mkdir_p(artifact_root / group.name)
        try:
            ensembled_model, run_json_data, n_reps = load_flavor_b_model(
                group, flavor_b_root, run_json_root,
            )
        except Exception as e:
            logger.error("  load failed: %s", e)
            err_row = {"group": group.name, "_error": f"load failed: {type(e).__name__}: {e}"}
            (group_artifact_dir / "gains.json").write_text(json.dumps(err_row, indent=2))
            continue

        n_reps_run = n_reps if args.limit_replicates is None else min(n_reps, args.limit_replicates)
        logger.info("  n_replicates=%d (analysing %d)", n_reps, n_reps_run)

        per_replicate: list[dict] = []
        for r_idx in range(n_reps_run):
            t_rep = time.time()
            try:
                single_rep_model = _select_replicate(ensembled_model, r_idx)
                single_rep_model = _cast_to_float64(single_rep_model)
                ctrl = build_network_controller(
                    single_rep_model,
                    target_pos=target_pos,
                    sisu=args.sisu,
                    key=jr.PRNGKey(args.seed + r_idx),
                )
            except Exception as e:
                logger.warning("    rep %d adapter build failed: %s", r_idx, e)
                per_replicate.append({"replicate": r_idx,
                                      "_error": f"adapter: {type(e).__name__}: {e}"})
                continue
            res = analyse_replicate(
                ctrl,
                plant=plant,
                schedule=schedule,
                init_pos=init_pos,
                target_pos=target_pos,
                horizon=args.horizon,
                channels=channels,
                seed=args.seed,
                g_star=g_star,
                n_restarts=args.n_restarts,
                max_iter=args.max_iter,
                rtol=args.rtol,
            )
            res["replicate"] = r_idx
            res["wall_time_s"] = time.time() - t_rep
            per_replicate.append(res)
            sd = res.get(f"{W_STRUCTURAL_DA}__{Z_QR_COST}", {})
            af = res.get(f"{W_ADDITIVE_FORCE}__{Z_QR_COST}", {})
            sp = res.get(f"{W_SENSORY_PERTURBATION}__{Z_QR_COST}", {})
            logger.info(
                "    rep %d: g_sd=%s g_af=%s g_sp=%s (%.1fs)",
                r_idx,
                f"{sd['gamma']:.4f}" if "gamma" in sd else "ERR",
                f"{af['gamma']:.4f}" if "gamma" in af else "ERR",
                f"{sp['gamma']:.4f}" if "gamma" in sp else "ERR",
                res["wall_time_s"],
            )

        per_group_summary = {
            "group": group.name,
            "eta_max": group.eta_max,
            "seed": group.seed,
            "n_replicates_run": n_reps_run,
            "n_replicates_total": n_reps,
            "gamma_star_riccati": float(g_star),
        }
        for w_ch, z_ch in channels:
            ch_label = f"{w_ch}__{z_ch}"
            gammas = []
            converged_flags = []
            for r in per_replicate:
                cell = r.get(ch_label, {})
                if "gamma" in cell:
                    gammas.append(cell["gamma"])
                    converged_flags.append(bool(cell.get("converged", False)))
                else:
                    gammas.append(float("nan"))
                    converged_flags.append(False)
            med, mad = _median_mad(gammas)
            flags = _hygiene_flag(gammas)
            per_group_summary[ch_label] = {
                "gammas": [float(g) for g in gammas],
                "converged": converged_flags,
                "median": med,
                "mad": mad,
                "ratio_to_gstar_median": float(med / g_star) if g_star > 0 and np.isfinite(med) else float("nan"),
                "outlier_flags": flags,
                "n_excluded_outliers": int(sum(flags)),
            }

        group_record = {
            "group": group.name,
            "eta_max": group.eta_max,
            "seed": group.seed,
            "n_replicates": n_reps,
            "channels": [{"w": w, "z": z} for w, z in channels],
            "gamma_star_riccati": float(g_star),
            "replicates": per_replicate,
            "summary": per_group_summary,
        }
        (group_artifact_dir / "gains.json").write_text(json.dumps(group_record, indent=2))
        logger.info("  saved %s/gains.json", group_artifact_dir)
        summary_per_group.append(per_group_summary)

    summary_doc = {
        "experiment": args.exp,
        "run": args.run,
        "groups": summary_per_group,
        "channels_order": [f"{w}__{z}" for w, z in channels],
        "headline_channel": f"{W_STRUCTURAL_DA}__{Z_QR_COST}",
        "gamma_star_riccati": float(g_star),
        "horizon": int(args.horizon),
        "rtol": float(args.rtol),
        "max_iter": int(args.max_iter),
        "n_restarts": int(args.n_restarts),
    }
    (artifact_root / "summary.json").write_text(json.dumps(summary_doc, indent=2))
    logger.info("Saved summary.json to %s", artifact_root / "summary.json")

    run_json = {
        "experiment": args.exp,
        "run": args.run,
        "spec_kind": "induced_gain_analysis",
        "subject": "flavor_b_trained_controllers",
        "channels": [{"w": w, "z": z} for w, z in channels],
        "headline_channel": f"{W_STRUCTURAL_DA}__{Z_QR_COST}",
        "horizon": int(args.horizon),
        "plant": {
            "kind": "linearize_pointmass",
            "mass": 1.0, "damping": 10.0, "tau": 0.05, "dt": 0.01,
        },
        "schedule": {"kind": "cost_schedule_from_spec", "n_steps": int(args.horizon)},
        "init_pos": [float(args.init_x), float(args.init_y)],
        "target_pos": [float(args.target_x), float(args.target_y)],
        "sisu": float(args.sisu),
        "n_restarts": int(args.n_restarts),
        "max_iter": int(args.max_iter),
        "rtol": float(args.rtol),
        "seed": int(args.seed),
        "groups_run": [g.name for g in groups_to_run],
        "input_artifacts": {
            "flavor_b_root": str(flavor_b_root),
            "run_json_root": str(run_json_root),
        },
        "gamma_star_riccati": float(g_star),
        "analyser_version": {
            "rlrmp_branch": "feature/induced-gain-flavor-b",
            "tracking_issue": "74bfd86",
            "input_training_issue": "c723082",
            "post_probe_canonical_params": True,
        },
        "hygiene_rule": "outlier flag if gamma > 10x from group median or < median / 10",
    }
    (spec_dir / "run.json").write_text(json.dumps(run_json, indent=2))
    logger.info("Saved run.json to %s", spec_dir / "run.json")

    logger.info("Done.")


if __name__ == "__main__":
    main()

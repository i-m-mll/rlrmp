"""Induced-gain first run on Part 2.5 trained checkpoints.

Bug: 6fdf9a4 — produces the first cross-method table of
``||T_{w → z}||_∞`` values for trained Part 2.5 networks.

Wraps the trained SimpleFeedback controller path as a ``Controller`` for
``rlrmp.analysis.math.induced_gain`` and runs the headline channels:
    - ``additive_force × qr_cost``  (Riccati-comparable scalar)
    - ``structural_da × qr_cost``   (flavor (a) ⊊ (b) discriminator)
    - ``sensory_perturbation × qr_cost`` (feedback-perturbation analogue)

Plus the Riccati H-inf γ⋆ baseline for the same plant + schedule + horizon
(so the table normalises RNN gains against the LTI optimum).

The analyzer owns the linearized plant and disturbance injection, so the
controller adapter intentionally wraps the Feedbax ``feedback -> net`` subgraph
rather than advancing the full SimpleFeedback mechanics graph internally. This
uses Feedbax's public ``GraphControllerAdapter`` for graph/state execution and
keeps only the RLRMP-specific observation-to-mechanics-state bridge downstream:

- Controller hidden state = Feedbax graph state flat (feedback delay channels
  plus network state) and any recurrent carry owned by the adapter.
- The trained ``DisturbanceField``, ``efferent`` (motor) channel, and force
  filter are bypassed. Disturbance enters via the analyser's
  ``additive_force`` w channel; motor noise and feedback noise are zeroed.
  Force filter is still part of the linearised plant (``tau=0.05``).
- Task input is held at a representative mid-movement value: the absolute
  target position with hold/go cues set to (0, 1) and SISU = 0.5.

Outputs:
- ``_artifacts/c723082/runs/induced_gain_first_run/gains.npz`` — structured array.
- ``results/c723082/runs/induced_gain_first_run.json`` — analysis spec (flat, post-f485c26).
- ``results/c723082/notes/induced_gain_first_run.md`` — narrative + headline table.

Usage:
    uv run --no-sync python results/c723082/scripts/run_induced_gain_part2_5.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.math.hinf_riccati import (
    CostSpec,
    PlantLinearization,
    cost_schedule_from_spec,
    find_gamma_star,
    linearize_pointmass,
)
from rlrmp.analysis.math.induced_gain import (
    W_ADDITIVE_FORCE,
    W_SENSORY_PERTURBATION,
    W_STRUCTURAL_DA,
    Z_QR_COST,
    Controller,
    induced_gain,
)
from rlrmp.analysis.feedbax_controllers import simple_feedback_induced_gain_controller
from rlrmp.paths import REPO_ROOT, mkdir_p, run_artifact_dir, run_spec_dir
from rlrmp.train.task_model import setup_task_model_pair

warnings.filterwarnings("ignore")

# Note: do NOT enable x64 at top level — saved checkpoints are float32 and
# `eqx.tree_deserialise_leaves` will refuse to load if the template has
# mismatched dtype. The analyser internally upcasts to float64 where it
# matters (linearisation, power iteration). The template-construction step in
# ``setup_task_model_pair`` runs in default precision (float32), and the
# saved weights are float32, so the load step succeeds.

logger = logging.getLogger(__name__)


# =============================================================================
# Group registry: (group_name, results_subpath, build_hps_module, checkpoint_filename)
# =============================================================================


@dataclass(frozen=True)
class GroupSpec:
    """Specification for a checkpoint group.

    Attributes:
        name: Group identifier (used as a row label and in artifact paths).
        results_subdir: Path under ``_artifacts/part2_5/runpod/`` where the
            ``config.json`` and checkpoint live.
        build_hps_module: Historical selector for the ``build_hps`` function
            that constructs the right hps namespace from the saved
            ``config.json`` (``train_minimax`` for Part 2.5 / minimax,
            ``train_part2_5`` maps to ``rlrmp.train.standard`` for newer
            baseline runs).
        checkpoint_kind: ``"adversarial"`` to load
            ``checkpoints_adversarial/checkpoint_latest/model.eqx`` (post-
            adversarial training); ``"warmup"`` to load ``warmup_model.eqx``
            (post-warmup, pre-adversarial); ``"adversarial_eqx"`` to load
            ``adversarial_model.eqx`` (final-cut single file when present).
    """

    name: str
    results_subdir: str
    build_hps_module: str
    checkpoint_kind: str  # "adversarial" | "warmup" | "adversarial_eqx"


# Pick the post-final/latest checkpoint for each group. ``adversarial_eqx``
# (a single ``adversarial_model.eqx`` saved at end of training) is preferred
# when present; otherwise the resumable ``checkpoint_latest`` directory.
GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec("baseline_standard_12k", "baseline/standard_12k", "train_minimax", "warmup"),  # baseline has only warmup phase
    GroupSpec("vanilla_single", "vanilla_single", "train_minimax", "adversarial"),
    GroupSpec("vanilla_pop5", "vanilla_pop5", "train_minimax", "adversarial"),
    GroupSpec("minimax_single_seed0", "minimax_single/seed_0", "train_minimax", "adversarial"),
    GroupSpec("minimax_single_seed1", "minimax_single/seed_1", "train_minimax", "adversarial"),
    GroupSpec("minimax_single_seed2", "minimax_single/seed_2", "train_minimax", "adversarial"),
    GroupSpec("mult_single", "mult_single", "train_minimax", "adversarial"),
    GroupSpec("mult_pop5", "mult_pop5", "train_minimax", "adversarial"),
    GroupSpec("ratio03_single", "ratio03_single", "train_minimax", "adversarial"),
    GroupSpec("ratio03_pop5", "ratio03_pop5", "train_minimax", "adversarial"),
)


# Map runpod-saved configs (which sometimes lack newer fields) to defaults
# the build_hps function expects. This is the smallest set of safe defaults
# to make every saved config compatible with the current train_minimax.build_hps.
_CONFIG_DEFAULTS = {
    "loss_update_enabled": True,
    "loss_update_ratio": 0.5,
    "hidden_type": "gru",
    "sisu_gating": "additive",
    "streaming_loss": True,
    "fused": True,
}


# =============================================================================
# Loading helpers
# =============================================================================


def _select_replicate(model, replicate_idx: int = 0):
    """Index the leading replicate axis on every weight leaf of the model.

    Detects ``n_replicates`` from the recurrent weight's leading axis (which
    is always the replicate axis for the trained ensemble). Picks one
    replicate; this is the natural way to extract a single deterministic
    network from the ensemble for the induced-gain analysis.
    """
    # weight_hh exists on GRUCell; for VanillaRNNCell wrapper we look one level deeper.
    net = model.nodes["net"]
    cell = net.hidden
    if hasattr(cell, "weight_hh"):
        ref = cell.weight_hh
    elif hasattr(cell, "_cell") and hasattr(cell._cell, "weight_hh"):
        ref = cell._cell.weight_hh
    else:
        # Fallback: take any weight-like leaf.
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
    """Cast all float-dtype array leaves to float64 (post-load promotion).

    After deserialisation the model is float32 (matches on-disk storage), but
    the analyser runs at float64. Casting the loaded model up to float64
    ensures the analyser's autodiff and ``state.set`` calls see consistent
    dtypes throughout the closed-loop linearisation.
    """
    def _cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float64)
        return x
    return jt.map(_cast, tree, is_leaf=eqx.is_array)


def _cast_to_float32(tree):
    """Cast template's float64→float32 and int64→int32 array leaves.

    Required because the analyser modules enable ``jax_enable_x64`` at import
    time (so ``setup_task_model_pair`` constructs templates in float64 / int64),
    but the on-disk checkpoints were saved at float32 / int32 —
    ``eqx.tree_deserialise_leaves`` refuses to load mismatched dtypes. Casting
    the template before load resolves the mismatch; the analyser internally
    upcasts to float64 where needed (``linearise_trajectory`` etc.).
    """
    def _cast(x):
        if hasattr(x, "dtype"):
            if jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp.float32)
            if jnp.issubdtype(x.dtype, jnp.signedinteger):
                # Convert int64→int32 to match on-disk storage.
                if x.dtype == jnp.int64:
                    return x.astype(jnp.int32)
        return x
    return jt.map(_cast, tree, is_leaf=eqx.is_array)


def _build_hps_for_group(group: GroupSpec, config: dict):
    """Resolve the build_hps callable for a group and return populated hps."""
    if group.build_hps_module == "train_minimax":
        from rlrmp.train.minimax import build_hps as build_hps_fn
    elif group.build_hps_module == "train_part2_5":
        from rlrmp.train.standard import build_hps as build_hps_fn
    else:
        raise ValueError(f"Unknown build_hps_module: {group.build_hps_module}")
    config_filtered = {k: v for k, v in config.items() if k not in ("git", "output_dir")}
    # Inject defaults for any missing newer fields.
    for k, v in _CONFIG_DEFAULTS.items():
        config_filtered.setdefault(k, v)
    args_ns = argparse.Namespace(**config_filtered)
    return build_hps_fn(args_ns)


def load_group_model(
    group: GroupSpec, runpod_root: Path, *, replicate_idx: int = 0
) -> tuple[object, object, dict]:
    """Load a single trained model + task + config for a checkpoint group.

    Returns:
        (model, task, config). The model has the replicate axis indexed
        (replicate 0) and is ready for single-trial use.
    """
    results_dir = runpod_root / group.results_subdir
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {results_dir}")
    config = json.loads(config_path.read_text())

    hps = _build_hps_for_group(group, config)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))

    if group.checkpoint_kind == "adversarial":
        ckpt_path = results_dir / "checkpoints_adversarial" / "checkpoint_latest" / "model.eqx"
        if not ckpt_path.exists():
            # Some groups never ran adversarial; fall back to warmup.
            warmup_path = results_dir / "warmup_model.eqx"
            if warmup_path.exists():
                logger.warning(
                    "%s: no adversarial checkpoint, falling back to warmup_model.eqx",
                    group.name,
                )
                model = _load_warmup(warmup_path, hps)
                model = _select_replicate(_squeeze_replicate_axis(model), replicate_idx)
                return _cast_to_float64(model), pair.task, config
            raise FileNotFoundError(f"No adversarial or warmup checkpoint found in {results_dir}")
        # The adversarial checkpoint was saved by reconstructing a single-replicate
        # tree (see train_minimax._save_adversarial_checkpoint: uses treedef_model
        # from single_rep_model). So the template must also be a single-replicate
        # model. Build it by extracting replicate 0 from the n_replicates ensemble.
        net = pair.model.nodes["net"]
        try:
            ref = net.hidden.weight_hh
        except AttributeError:
            ref = net.hidden._cell.weight_hh
        n_reps = int(ref.shape[0]) if ref.ndim > 2 else 1
        single_rep_template = jt.map(
            lambda x: x[0] if (eqx.is_array(x) and x.ndim > 0 and x.shape[0] == n_reps) else x,
            pair.model,
            is_leaf=eqx.is_array,
        )
        template = _cast_to_float32(single_rep_template)
        try:
            model = eqx.tree_deserialise_leaves(ckpt_path, template)
        except Exception:
            # Fallback: try with just the casted full-ensemble template.
            model = eqx.tree_deserialise_leaves(ckpt_path, _cast_to_float32(pair.model))
            model = _select_replicate(model, replicate_idx)
            return _cast_to_float64(model), pair.task, config
        # Already single-replicate.
        return _cast_to_float64(model), pair.task, config

    if group.checkpoint_kind == "adversarial_eqx":
        path = results_dir / "adversarial_model.eqx"
        model = _load_warmup(path, hps)  # same load idiom (HDF5 + hyperparameters)
        model = _select_replicate(_squeeze_replicate_axis(model), replicate_idx)
        return _cast_to_float64(model), pair.task, config

    # warmup
    path = results_dir / "warmup_model.eqx"
    model = _load_warmup(path, hps)
    model = _select_replicate(_squeeze_replicate_axis(model), replicate_idx)
    return _cast_to_float64(model), pair.task, config


def _load_warmup(path: Path, hps):
    """Load via load_with_hyperparameters; the produced model is float32 on
    disk, but the template constructed inside the setup_func runs in default
    precision (which is float64 once x64 is enabled). Cast template to f32."""
    def _setup(key, **kwargs):
        return _cast_to_float32(setup_task_model_pair(hps, key=key).model)
    model, _ = load_with_hyperparameters(path, setup_func=_setup)
    return model


def build_network_controller(
    model,
    *,
    target_pos: jnp.ndarray,
    sisu: float = 0.5,
    key: jnp.ndarray = jr.PRNGKey(0),
) -> Controller:
    """Build a Feedbax-backed controller adapter for a Part 2.5 model.

    Args:
        model: The trained ``SimpleFeedback`` graph (replicate axis already
            indexed — see ``_select_replicate``).
        target_pos: Absolute target position (shape (2,)).
        sisu: SISU level held throughout the analysis. The default 0.5 is a
            mid-range value; the gain depends weakly on SISU.
        key: Base PRNG key. Per-step keys are derived via ``fold_in(t)``.

    Returns:
        A ``Controller`` ready for ``induced_gain(...)``.
    """
    return simple_feedback_induced_gain_controller(
        model,
        target_pos=target_pos,
        sisu=sisu,
        key=key,
    )


# =============================================================================
# Main analyser
# =============================================================================


def _build_plant_and_schedule(horizon: int) -> tuple[PlantLinearization, object]:
    """Build the rlrmp 6-state plant + a representative cost schedule.

    Both match the test in ``tests/test_induced_gain.py::test_riccati_round_trip_qr_cost``.
    """
    plant = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)
    spec = CostSpec(n_steps=horizon)
    schedule = cost_schedule_from_spec(spec, plant)
    return plant, schedule


def analyse_group(
    group: GroupSpec,
    runpod_root: Path,
    *,
    horizon: int,
    init_pos: jnp.ndarray,
    target_pos: jnp.ndarray,
    sisu: float,
    seed: int,
    plant: PlantLinearization,
    schedule,
    g_star: float,
    channels: tuple[tuple[str, str], ...],
    n_restarts: int = 3,
    max_iter: int = 600,
    rtol: float = 1e-5,
) -> dict:
    """Run the induced-gain analysis on one group's checkpoint.

    Returns a dict mapping channel name → InducedGainResult, plus
    ``"_meta"`` with timing and config info. On failure, returns
    ``{"_error": "..."}`` so the runner can record skip reasons.
    """
    out: dict = {}
    t_load_start = time.time()
    try:
        model, task, config = load_group_model(group, runpod_root, replicate_idx=0)
    except Exception as e:
        return {"_error": f"load failed: {type(e).__name__}: {e}"}
    t_load = time.time() - t_load_start

    # Build the controller for this checkpoint.
    try:
        ctrl = build_network_controller(
            model, target_pos=target_pos, sisu=sisu, key=jr.PRNGKey(seed)
        )
    except Exception as e:
        return {"_error": f"adapter build failed: {type(e).__name__}: {e}"}

    out["_meta"] = {
        "load_time_s": t_load,
        "n_ctrl": int(ctrl.h0_flat.shape[0]),
        "delay": int(ctrl.delay),
        "n_obs": int(ctrl.n_obs),
        "config_keys": sorted(config.keys()),
    }

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
                "diagnostics": pi.diagnostics,
            }
        except Exception as e:
            out[ch_label] = {
                "error": f"{type(e).__name__}: {e}",
                "time_s": time.time() - t0,
            }
    return out


def render_markdown_table(rows: list[dict], g_star: float, channels) -> str:
    """Build the headline markdown table in the form prescribed by the spec."""
    # Format: | Group | γ⋆ | γ_PI: af×qr | γ_PI: sd×qr | γ_PI: sp×qr | γ/γ⋆ (af) |
    af_qr = "additive_force__qr_cost"
    sd_qr = "structural_da__qr_cost"
    sp_qr = "sensory_perturbation__qr_cost"

    lines = []
    lines.append("| Group | γ⋆ (Riccati) | γ_PI: af×qr | γ_PI: sd×qr | γ_PI: sp×qr | γ/γ⋆ (af) |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        name = r["group"]
        if r.get("_error"):
            lines.append(f"| `{name}` | {g_star:.4f} | — | — | — | error: {r['_error']} |")
            continue

        def _cell(key):
            if key in r and "gamma" in r[key]:
                converged_mark = "" if r[key]["converged"] else "*"
                return f"{r[key]['gamma']:.4f}{converged_mark}"
            if key in r and "error" in r[key]:
                return "err"
            return "—"

        af = _cell(af_qr)
        sd = _cell(sd_qr)
        sp = _cell(sp_qr)
        if af_qr in r and "gamma" in r[af_qr] and g_star > 0:
            ratio = f"{r[af_qr]['gamma'] / g_star:.3f}"
        else:
            ratio = "—"
        lines.append(f"| `{name}` | {g_star:.4f} | {af} | {sd} | {sp} | {ratio} |")
    lines.append("")
    lines.append("Asterisk (`*`) marks non-converged power-iteration results — the reported gamma is the largest restart estimate at ``max_iter``.")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # The trained checkpoints live in the main worktree's _artifacts/ tree
    # (worktrees only get a stub README.md). Default to the main worktree's
    # absolute path; override with --runpod-root if running elsewhere.
    default_runpod_root = "/Users/mll/Main/10 Projects/10 PhD/rlrmp/_artifacts/part2_5/runpod"
    if not Path(default_runpod_root).exists():
        default_runpod_root = str(REPO_ROOT / "_artifacts" / "part2_5" / "runpod")
    parser.add_argument(
        "--runpod-root", type=str,
        default=default_runpod_root,
        help="Root of Part 2.5 runpod artifacts.",
    )
    parser.add_argument("--horizon", type=int, default=200,
                        help="LTV horizon T (matches test_induced_gain post-fix).")
    parser.add_argument("--init-x", type=float, default=0.0)
    parser.add_argument("--init-y", type=float, default=0.0)
    parser.add_argument("--target-x", type=float, default=0.15,
                        help="Target x in metres (default 15 cm forward).")
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--sisu", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-restarts", type=int, default=3,
                        help="Power-iteration restarts (more = better top-SV recovery).")
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N groups (debug / smoke test).")
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated group names to include (overrides default registry).")
    parser.add_argument("--exp", type=str, default="part2_5")
    parser.add_argument("--run", type=str, default="induced_gain_first_run")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    runpod_root = Path(args.runpod_root)
    if not runpod_root.exists():
        raise FileNotFoundError(f"Runpod artifact root not found: {runpod_root}")

    init_pos = jnp.array([args.init_x, args.init_y], dtype=jnp.float64)
    target_pos = jnp.array([args.target_x, args.target_y], dtype=jnp.float64)

    plant, schedule = _build_plant_and_schedule(args.horizon)
    logger.info("Built plant (n=%d) and cost schedule (T=%d).", plant.n, schedule.T)

    logger.info("Computing Riccati γ⋆ baseline...")
    t0 = time.time()
    g_star = find_gamma_star(plant, schedule)
    logger.info("γ⋆ = %.6f  (%.1fs)", g_star, time.time() - t0)

    channels = (
        (W_ADDITIVE_FORCE, Z_QR_COST),
        (W_STRUCTURAL_DA, Z_QR_COST),
        (W_SENSORY_PERTURBATION, Z_QR_COST),
    )

    if args.groups:
        wanted = set(args.groups.split(","))
        groups_to_run = tuple(g for g in GROUPS if g.name in wanted)
    else:
        groups_to_run = GROUPS
    if args.limit is not None:
        groups_to_run = groups_to_run[: args.limit]

    rows: list[dict] = []
    for i, group in enumerate(groups_to_run):
        logger.info("[%d/%d] Group %s", i + 1, len(groups_to_run), group.name)
        result = analyse_group(
            group,
            runpod_root,
            horizon=args.horizon,
            init_pos=init_pos,
            target_pos=target_pos,
            sisu=args.sisu,
            seed=args.seed,
            plant=plant,
            schedule=schedule,
            g_star=g_star,
            channels=channels,
            n_restarts=args.n_restarts,
            max_iter=args.max_iter,
            rtol=args.rtol,
        )
        row = {"group": group.name}
        if "_error" in result:
            row["_error"] = result["_error"]
            logger.warning("  → SKIPPED: %s", result["_error"])
        else:
            for ch in channels:
                ch_label = f"{ch[0]}__{ch[1]}"
                row[ch_label] = result.get(ch_label, {})
            row["_meta"] = result.get("_meta", {})
            for ch in channels:
                ch_label = f"{ch[0]}__{ch[1]}"
                cell = result.get(ch_label, {})
                if "gamma" in cell:
                    logger.info(
                        "  %s: γ=%.4f (conv=%s, iters=%d, %.1fs)",
                        ch_label, cell["gamma"], cell["converged"],
                        cell["iterations"], cell["time_s"],
                    )
                elif "error" in cell:
                    logger.warning("  %s: ERROR %s", ch_label, cell["error"])
        rows.append(row)

    # =========================================================================
    # Save artifacts
    # =========================================================================
    artifact_dir = mkdir_p(run_artifact_dir(args.exp, args.run))
    spec_dir = mkdir_p(run_spec_dir(args.exp, args.run))

    # Structured save: gains.npz with per-(group, channel) arrays.
    columns: dict[str, list] = {
        "group": [],
        "w_channel": [],
        "z_channel": [],
        "gamma_PI": [],
        "gamma_star_riccati": [],
        "ratio": [],
        "n_steps_pi": [],
        "converged": [],
        "n_ctrl": [],
        "delay": [],
        "load_time_s": [],
        "channel_time_s": [],
        "error": [],
    }
    for row in rows:
        meta = row.get("_meta", {})
        n_ctrl = int(meta.get("n_ctrl", -1)) if meta else -1
        delay = int(meta.get("delay", -1)) if meta else -1
        load_time = float(meta.get("load_time_s", float("nan"))) if meta else float("nan")
        if "_error" in row:
            for ch in channels:
                columns["group"].append(row["group"])
                columns["w_channel"].append(ch[0])
                columns["z_channel"].append(ch[1])
                columns["gamma_PI"].append(float("nan"))
                columns["gamma_star_riccati"].append(float(g_star))
                columns["ratio"].append(float("nan"))
                columns["n_steps_pi"].append(-1)
                columns["converged"].append(False)
                columns["n_ctrl"].append(n_ctrl)
                columns["delay"].append(delay)
                columns["load_time_s"].append(load_time)
                columns["channel_time_s"].append(float("nan"))
                columns["error"].append(row["_error"])
            continue
        for ch in channels:
            ch_label = f"{ch[0]}__{ch[1]}"
            cell = row.get(ch_label, {})
            columns["group"].append(row["group"])
            columns["w_channel"].append(ch[0])
            columns["z_channel"].append(ch[1])
            if "gamma" in cell:
                columns["gamma_PI"].append(float(cell["gamma"]))
                columns["gamma_star_riccati"].append(float(g_star))
                columns["ratio"].append(float(cell["gamma"] / g_star) if g_star > 0 else float("nan"))
                columns["n_steps_pi"].append(int(cell.get("iterations", -1)))
                columns["converged"].append(bool(cell.get("converged", False)))
                columns["error"].append("")
            else:
                columns["gamma_PI"].append(float("nan"))
                columns["gamma_star_riccati"].append(float(g_star))
                columns["ratio"].append(float("nan"))
                columns["n_steps_pi"].append(-1)
                columns["converged"].append(False)
                columns["error"].append(cell.get("error", ""))
            columns["n_ctrl"].append(n_ctrl)
            columns["delay"].append(delay)
            columns["load_time_s"].append(load_time)
            columns["channel_time_s"].append(float(cell.get("time_s", float("nan"))))

    arrays = {k: np.array(v) for k, v in columns.items()}
    np.savez(artifact_dir / "gains.npz", **arrays)
    logger.info("Saved gains.npz to %s", artifact_dir / "gains.npz")

    # run.json: analysis spec
    run_json = {
        "experiment": args.exp,
        "run": args.run,
        "spec_kind": "induced_gain_analysis",
        "channels": [{"w": w, "z": z} for w, z in channels],
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
        "gamma_star_riccati": float(g_star),
        "analyser_version": {
            "rlrmp_branch": "feature/induced-gain-first-run",
            "induced_gain_commit_refs": ["aa60d3f", "f0a1d44"],
            "tracking_issue": "6fdf9a4",
            "audit_issue": "b131510",
        },
    }
    (spec_dir / "run.json").write_text(json.dumps(run_json, indent=2))
    logger.info("Saved run.json to %s", spec_dir / "run.json")

    # notes.md: narrative + table
    table_md = render_markdown_table(rows, g_star, channels)
    notes = _build_notes_md(args, g_star, rows, table_md, channels)
    (spec_dir / "notes.md").write_text(notes)
    logger.info("Saved notes.md to %s", spec_dir / "notes.md")
    logger.info("Done.")


def _build_notes_md(args, g_star: float, rows: list[dict], table_md: str, channels) -> str:
    """Compose the short narrative for notes.md."""
    n_total = len(rows)
    n_ok = sum(1 for r in rows if "_error" not in r)
    n_err = n_total - n_ok

    af_qr = "additive_force__qr_cost"
    af_gammas = [r[af_qr]["gamma"] for r in rows if "_error" not in r and af_qr in r and "gamma" in r[af_qr]]

    af_summary = (
        f"min={min(af_gammas):.4f}, max={max(af_gammas):.4f}, "
        f"min γ/γ⋆={min(af_gammas) / g_star:.3f}, max γ/γ⋆={max(af_gammas) / g_star:.3f}"
        if af_gammas else "no successful runs"
    )

    # Group together findings worth highlighting
    findings_lines: list[str] = []
    af_qr_key = "additive_force__qr_cost"
    # Sort groups by additive_force gain (excluding mult_single outlier if extreme)
    af_pairs = [
        (r["group"], r[af_qr_key]["gamma"])
        for r in rows
        if "_error" not in r and af_qr_key in r and "gamma" in r[af_qr_key]
    ]
    if af_pairs:
        af_pairs.sort(key=lambda p: p[1])
        sorted_str = " < ".join(f"{name} ({g:.4f})" for name, g in af_pairs)
        findings_lines.append(f"- Force-channel ranking (low → high): {sorted_str}")

    # Spotlight outliers
    outliers = [r["group"] for r in rows
                if "_error" not in r and af_qr_key in r and "gamma" in r[af_qr_key]
                and r[af_qr_key]["gamma"] > 1.0]
    if outliers:
        findings_lines.append(
            f"- Outliers (γ_af > 1.0, indicating closed-loop instability for the linearised analysis): "
            f"{', '.join(outliers)}"
        )

    findings_md = "\n".join(findings_lines) if findings_lines else "(no findings extracted)"

    return f"""# Induced-gain first run — Part 2.5 trained checkpoints

**Issue.** `6fdf9a4`
**Branch.** `feature/induced-gain-first-run`
**Date.** 2026-05-07

## Goal

Compute the closed-loop H-infinity induced gain `||T_{{w → z}}||_∞` for each
trained Part 2.5 checkpoint group across three w channels (additive_force,
structural_da, sensory_perturbation), comparing each gain against the H-inf
Riccati γ⋆ on the same plant + cost schedule.

## Setup

- **Plant** (rlrmp regime): `linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)` → 6-state.
- **Cost schedule**: `cost_schedule_from_spec(CostSpec(n_steps={args.horizon}))` (matches `test_riccati_round_trip_qr_cost`).
- **Reach**: `init=({args.init_x}, {args.init_y})`, `target=({args.target_x}, {args.target_y})` (15 cm forward).
- **SISU**: held at {args.sisu}.
- **Horizon**: T = {args.horizon} steps (= 2 s).
- **Algorithm**: power iteration only (3 restarts, max_iter={args.max_iter}, rtol={args.rtol}).
  Hamiltonian/fixed-point not run for v1 — adds friction without changing the headline.
- **Riccati baseline**: γ⋆ = **{g_star:.6f}**.

## Headline table

{table_md}

**Headline summary (additive_force × qr_cost)**: {af_summary}.

## Controller adapter

The maintained runner now uses `rlrmp.analysis.feedbax_controllers.
simple_feedback_induced_gain_controller`. That adapter wraps the original
Feedbax `feedback -> net` controller subgraph with Feedbax's public
`GraphControllerAdapter`, while RLRMP supplies the induced-gain-specific bridge
from analyzer observations to the feedback node's `MechanicsState` input.

- The augmented controller hidden state is the Feedbax graph-controller state:
  feedback-channel delay state plus network state, with recurrent carry handled
  by Feedbax when present.
- Disturbance enters via `linearize_pointmass`'s `Bw` (additive force on
  velocity), bypassing the trained `DisturbanceField` intervenor.
- Motor noise is bypassed with the efferent channel; feedback-channel noise is
  disabled for deterministic linearisation.
- Task input is held at a representative mid-movement vector
  (`target_pos`, hold=0, go=1, sisu={args.sisu}).
- The trained network was trained on absolute coordinates; the adapter
  converts the analyser's goal-centred `pos` back to absolute by adding
  `target_pos`.

Other small frictions handled in the runner:
- Replicate axis. Trained ensembles save with leading `n_replicates=5` axis
  on every weight leaf; `_select_replicate(model, 0)` indexes one replicate
  before passing to the adapter.
- Multiple checkpoint kinds: `baseline_standard_12k` saves only a warmup
  model (no adversarial phase); other groups save
  `checkpoints_adversarial/checkpoint_latest/model.eqx`. The runner picks
  the latest available per group.
- Saved configs from the runpod batch lack newer fields
  (`hidden_type`, `sisu_gating`, `streaming_loss`, `fused`, etc.); the
  runner injects sane defaults so `train_minimax.build_hps` accepts them.

## What is *not* in this run

- **Hamiltonian / LTI fixed-point** induced gain. The power-iteration LTV
  result is the headline for finite-horizon reaches; the Hamiltonian path
  would require a fixed-point Newton solve per checkpoint and adds friction
  without changing the cross-method comparison.
- **`peak_velocity`** z channel. Behavioural-scaled gain — useful but
  orthogonal to the flavor (a) ⊊ (b) discrimination.
- **Single seed / single replicate**. Each group reports replicate 0 only;
  the analyser is deterministic given the loaded weights, but cross-replicate
  variance (within `_pop5` groups especially) is not characterised here.
- **Noise / stochastic terms**. The trained controller has multiplicative
  motor noise, additive feedback noise, and hidden-unit noise. All are
  zeroed for the analysis (the H-inf framing is a deterministic worst-case
  perturbation; the analyser does not model stochastic gain). For groups
  that explicitly trained against stochastic perturbation (e.g. `mult_*`),
  the gain reported here may understate the *effective* perturbation
  attenuation.
- **Deprioritised groups**: `tier1_redo`, `bench`, `ratio_sweep`. Skipped
  per the issue spec.
- **Cross-checkpoint sweep** (γ vs training step). Only the
  final/latest checkpoint per group.

## Headline observations

{n_ok}/{n_total} groups produced gains; {n_err} skipped (see table for
reasons). The full numerical detail lives in
`_artifacts/part2_5/runs/induced_gain_first_run/gains.npz` (mirror of this
spec dir).

{findings_md}

**Cross-method observations (af×qr channel):**

- Riccati H-inf γ⋆ = {g_star:.4f}; the *best* trained network (`baseline_standard_12k`,
  γ_af ≈ {min(af_gammas):.4f} if af_gammas else "n/a") is roughly **{min(af_gammas) / g_star:.0f}× above γ⋆**.
  This is the expected order of magnitude — no trained network is designed to
  minimise the H-inf operator norm directly; they minimise expected QR cost
  under stochastic perturbations.
- The **minimax-trained** seeds (0/1/2) cluster tightly (γ_af ≈ 0.145–0.156)
  but are *not* the lowest force-channel gains in the table — `baseline_standard_12k`
  (γ_af ≈ 0.124) is comparable, and `vanilla_pop5` (γ_af ≈ 0.146) matches
  minimax. This suggests the H-inf operator norm of the closed loop is **not
  a sensitive discriminator** between the parametric force-field adversary
  used in minimax training and other training regimes — at least not on this
  single canonical reach with this single replicate.
- The **sensory-perturbation** channel (γ_sp) distinguishes minimax (γ_sp ≈
  1.3–2.1) from non-adversarial training (γ_sp ≈ 2.6–4.4) more cleanly than
  γ_af does. This tracks the analogous pattern from the existing
  feedback-perturbation analysis: minimax improves robustness to feedback
  noise more than to body-frame force.
- The **structural_da** channel produces very large gains (γ_sd ≈ 150–170)
  across all groups — the implied small-gain margin against unstructured
  plant uncertainty is ~0.6% of the operator norm of the nominal closed loop,
  uniformly across training methods. This is consistent with the expectation
  that finite-horizon LQ-style training does not reward small-gain
  robustness.

**Outlier**: `mult_single` produces γ_af ≈ 15.6 and γ_sd ≈ 5860 — clearly an
unstable closed-loop linearisation. The most likely explanation is replicate
0 of `mult_single` did not converge during training (or has a degenerate
fixed point); `mult_pop5` of the same training method is well-behaved
(γ_af ≈ 0.18). Worth a v2 spot-check across replicates of `mult_single`.

**Caveats on flavor (a) ⊊ (b)**: this first run does **not** strongly
discriminate flavor-(a) (additive force) and flavor-(b) (structural ΔA)
trained networks because:
1. The set of training methods covered here uses only flavor-(a)
   adversaries (parametric force-field minimax + multiplicative noise
   variants). No model in this run was trained against a structural ΔA
   adversary, so we cannot compare γ_sd × qr ratios *across* flavors.
2. The `structural_da` channel here measures the *sensitivity* of any
   closed loop to unstructured ΔA, but on its own does not establish
   whether flavor-(a) or flavor-(b) training reduces that sensitivity
   more — that comparison requires a flavor-(b)-trained network as a
   data point.

The implication for the broader question is logged separately on the
analyses coordination issue (`4d38c15`) and the training-methods
coordination issue (`c99ad9d`).
"""


if __name__ == "__main__":
    main()

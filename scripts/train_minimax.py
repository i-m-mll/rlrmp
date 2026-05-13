"""Minimax adversarial training script for the RLRMP project.

Phase 1 (warm-start): Train the controller normally with random gust perturbations.
Phase 2 (adversarial): Alternate between adversary gradient ascent and controller gradient descent.
Supports fused mode (--fused, default) which compiles the K adversary steps + 1 controller
step into a single JIT call via lax.fori_loop, and decomposed mode (--no-fused) which uses
K×2 + 1 separate JIT calls per batch.

The adversary (GaussianBumpAdversary) generates SISU-conditional force profiles that
replace random gusts during adversarial training.

Population-based mode (--n-adversaries K) creates K independent adversaries that
rotate each batch (adversary index = batch_idx % K), providing diverse perturbation
pressure. When K=1, behavior is identical to the original single-adversary mode.

Usage:
    uv run python scripts/train_minimax.py --n-warmup-batches 2000 --n-adversary-batches 8000
    uv run python scripts/train_minimax.py --n-warmup-batches 20 --n-adversary-batches 30 \
        --output-dir /tmp/minimax_smoke
    uv run python scripts/train_minimax.py --n-adversaries 5 --n-adversary-batches 10000 \
        --output-dir results/pop_adversary
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
import optax
from feedbax._io import load_with_hyperparameters, save as fbx_save
from feedbax.graph import init_state_from_component
from feedbax.iterate import run_component
from feedbax.misc import BatchInfo
from feedbax.streaming_loss import make_streaming_loss_fn
from feedbax.task import (
    _extract_timeseries_params,
    _infer_n_steps,
    _merge_intervene_inputs,
    _prepare_inputs,
    _safe_state_set,
    _set_state_by_path,
    _where_key_to_path,
)
from feedbax.training.train import (
    TaskTrainer,
    make_delayed_cosine_schedule,
    train_pair,
)
from feedbax.types import TreeNamespace, dict_to_namespace
from feedbax.intervene import TimeSeriesParam

from rlrmp.adversarial_training import (
    _inject_adversary_delta_A,
    _inject_adversary_forces,
)
from rlrmp.adversary import GaussianBumpAdversary, LinearDynamicsAdversary
from rlrmp.intervention_compat import (
    swap_plant_intervenor_to_dynamics_matrix,
    swap_task_intervention_to_dynamics_matrix,
)
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p

logger = logging.getLogger(__name__)


def _adversary_update(adversary_optimizer, adversary, dL_dforces, adv_opt_st):
    """Update adversary parameters via VJP through adversary().

    Performs gradient ascent: adversary maximises loss, so we negate dL_dforces
    before computing the VJP (equivalent to ascending the loss gradient).

    Args:
        adversary_optimizer: Optax optimizer for the adversary (static).
        adversary: Current GaussianBumpAdversary.
        dL_dforces: Gradient of loss w.r.t. forces, shape (batch_size, T, d).
        adv_opt_st: Current adversary optimizer state.

    Returns:
        Tuple of (updated_adversary, updated_opt_state).
    """
    # forces = adversary() broadcast to (batch_size, T, d)
    # We need dL/d(adversary_params) = dL/dforces * dforces/d(adversary_params)
    # Use jax.linear_util / vjp directly through the broadcast.
    def _forces_fn(a):
        force_profile = a()  # (T, d)
        return jnp.broadcast_to(force_profile, dL_dforces.shape)

    # VJP: dL/d(params) via chain rule through the broadcast
    _, vjp_fn = jax.vjp(lambda a: eqx.filter(_forces_fn(a), eqx.is_array), adversary)
    # Negate for gradient ascent
    neg_dL_dforces = jt.map(lambda g: -g, dL_dforces)
    (param_grads,) = vjp_fn(neg_dL_dforces)

    updates, new_opt_st = adversary_optimizer.update(
        eqx.filter(param_grads, eqx.is_array),
        adv_opt_st,
        eqx.filter(adversary, eqx.is_array),
    )
    new_adversary = eqx.apply_updates(adversary, updates)
    return new_adversary, new_opt_st


# ---------------------------------------------------------------------------
# Spec-dir / artifact-dir helpers
# ---------------------------------------------------------------------------

def derive_spec_dir(output_dir: Path) -> Path:
    """Derive the run spec directory from the run artifact directory.

    Applies the mirror invariant ``run_artifact_dir(exp, run)`` ↔
    ``run_spec_dir(exp, run)``: paths under ``<repo>/_artifacts/...`` are
    re-rooted under ``<repo>/results/...``. Paths outside the
    ``_artifacts/`` tree fall back to a sibling ``<output_dir>_spec``.

    Args:
        output_dir: Absolute or relative path to the bulk-artifact directory
            (typically under ``_artifacts/<exp>/runs/<run>/``).

    Returns:
        Absolute path to the corresponding spec directory.
    """
    out = Path(output_dir).resolve()
    artifact_root = (REPO_ROOT / "_artifacts").resolve()
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = out.relative_to(artifact_root)
        return spec_root / rel
    except ValueError:
        return out.parent / (out.name + "_spec")


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _get_git_metadata() -> dict:
    """Capture git info for reproducibility."""
    meta = {}
    try:
        import rlrmp
        meta["rlrmp_version"] = getattr(rlrmp, "__version__", "unknown")
    except ImportError:
        pass
    for cmd, key in [
        (["git", "rev-parse", "HEAD"], "rlrmp_commit"),
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "rlrmp_branch"),
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                meta[key] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    try:
        import jax
        meta["jax_version"] = jax.__version__
    except ImportError:
        pass
    try:
        import feedbax
        meta["feedbax_version"] = getattr(feedbax, "__version__", "unknown")
    except ImportError:
        pass
    return meta


def _collect_gpu_info() -> dict:
    """Capture GPU/device info for run-config reproducibility.

    Records JAX-visible device kinds, count, and (best-effort) per-device
    total memory in GiB via ``nvidia-smi``. Never raises — failures are
    caught and surfaced as ``nvidia_smi_error`` / ``device_memory_gb_total =
    None`` so the surrounding training run is not blocked. Bug: c723082.

    Returns:
        Dict with keys ``device_kinds``, ``device_count``, optionally
        ``device_memory_gb_total`` and ``nvidia_smi_error``. Safe to embed
        directly in the run-config JSON written to disk.
    """
    info: dict = {}
    try:
        devices = jax.devices()
        info["device_kinds"] = [d.device_kind for d in devices]
        info["device_count"] = len(devices)
    except Exception as e:
        info["device_kinds"] = None
        info["device_count"] = 0
        info["jax_devices_error"] = str(e)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        mem_csv = result.stdout.strip().split("\n")
        info["device_memory_gb_total"] = [float(m) / 1024.0 for m in mem_csv if m]
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        info["device_memory_gb_total"] = None
        info["nvidia_smi_error"] = str(e)
    except Exception as e:  # pragma: no cover — defensive
        info["device_memory_gb_total"] = None
        info["nvidia_smi_error"] = f"unexpected: {e}"
    return info


def _configure_jax_runtime(args: argparse.Namespace) -> None:
    """Configure JAX runtime options that must be set before first compile."""
    cache_dir = args.jax_cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_path))
        logger.info("Using JAX compilation cache dir: %s", cache_path)

    if args.jax_explain_cache_misses:
        jax.config.update("jax_explain_cache_misses", True)
        logger.info("Enabled jax_explain_cache_misses for cache diagnostics")


# ---------------------------------------------------------------------------
# Hyperparameter construction
# ---------------------------------------------------------------------------

def _resolve_hidden_type(hidden_type_str: str, dt: float):
    """Map a CLI hidden-type string to the corresponding recurrent cell class/partial.

    Args:
        hidden_type_str: One of "gru" or "vanilla_rnn".
        dt: Simulation timestep.

    Returns:
        A class or partial-applied constructor compatible with point_mass_nn's
        ``hidden_type`` parameter (i.e. callable as ``hidden_type(input_size,
        hidden_size, use_bias=..., key=...)``).
    """
    if hidden_type_str == "gru":
        return eqx.nn.GRUCell
    elif hidden_type_str == "vanilla_rnn":
        from functools import partial as _partial
        from rlrmp.models import VanillaRNNCell
        # tau=0.1 s (100 ms) => alpha=dt/tau=0.1 at dt=0.01 — matches cortical-neuron
        # time constant in motor-control RNN literature (Yang 2019, Sussillo 2015).
        return _partial(VanillaRNNCell, dt=dt, tau=0.1)
    elif hidden_type_str in ("linear", "linear_tracker"):
        # Sentinel string forwarded to setup_task_model_pair, which dispatches to
        # create_point_mass_linear_ensemble. Linear controllers have no recurrent
        # cell — they replace SimpleStagedNetwork entirely. Bug: 410d7ac.
        return hidden_type_str
    else:
        raise ValueError(f"Unknown hidden_type: {hidden_type_str!r}")


def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Construct hyperparameters for task/model setup.

    Uses the same task config as train_part2_5.py (running_cost loss mode),
    so the two scripts produce comparable models.
    """
    dt = 0.01
    hps_dict = {
        "method": "pai-asf",
        "dt": dt,
        # n_batches_condition drives setup_task_model_pair's loss schedule;
        # set to total training length so late-ramp terms are calibrated correctly.
        "n_batches_condition": args.n_warmup_batches + args.n_adversary_batches,
        "n_batches_baseline": 0,
        "batch_size": getattr(args, "batch_size", 250),
        "learning_rate_0": args.controller_lr,
        "n_scaleup_batches": 0,
        "constant_lr_iterations": 0,
        "cosine_annealing_alpha": 1.0,
        "weight_decay": 0.0,
        "state_reset_iterations": [],
        "intervention_scaleup_batches": [0, 0],
        "model": {
            "n_replicates": getattr(args, "n_replicates", 5),
            "effector_mass": 1.0,
            "hidden_size": 180,
            "feedback_delay_steps": 5,
            "feedback_noise_std": 0.01,
            "motor_noise_std": 0.01,
            "damping": 10.0,
            "tau_rise": 0.05,
            "population_structure": {
                "n_input_only": 60,
                "n_readout_only": 60,
                "n_recurrent_only": 60,
                "n_input_readout": 0,
            },
        },
        "task": {
            "type": "center_out_delayed_reach",
            "n_steps": 140,
            "workspace": [[-1.0, -1.0], [1.0, 1.0]],
            "eval_grid_n": 1,
            "eval_n_directions": 8,
            "eval_reach_length": 0.5,
            # Drop pure-hold to 0 steps; target-on now 100-300 ms (10-30 steps
            # at dt=0.01 s), matching Shahbazi 2025 §4.2. Bug: 2bc95fd
            "epoch_len_ranges": [[0, 1], [10, 30]],
            "target_on_epochs": [1, 2],
            "hold_epochs": [0, 1],
            "move_epochs": [2],
            "p_catch_trial": getattr(args, "p_catch_trial", 0.5),
        },
        "pert": {
            "type": "gusts",
            # Warm-start uses pert_std=1.0 (normal gusts).
            "std": 1.0,
            "duration_mean": 8,
            "n_expected": 3,
        },
        "loss": {
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": getattr(args, "effector_pos_running", 1.0),
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": getattr(args, "effector_pos_late_weight", 0.5),
                "effector_vel_late": getattr(args, "effector_vel_late", 0.1),
                "effector_hold_pos": getattr(args, "effector_hold_pos", 10.0),
                "effector_hold_vel": getattr(args, "effector_hold_vel", 10.0),
                # Terminal-step velocity penalty (historical simple_reach_loss
                # shape). Fires only at t=T; strong "come-to-rest" signal.
                # Default 0.0 = disabled (preserves baseline behaviour).
                # Activate via --effector-final-vel 1.0. Bug: 2bc95fd
                "effector_final_vel": getattr(args, "effector_final_vel", 0.0),
                "nn_output": getattr(args, "nn_output", 1e-5),
                "nn_hidden": getattr(args, "nn_hidden", 1e-5),
                # Compositional ||h_t - h_{t-1}||² hidden-state smoothness
                # term, off-by-default. Enable via --nn-hidden-derivative
                # (e.g. 1e-3 per Shahbazi et al. 2025 Eq. 1). Bug: efc4d68
                "nn_hidden_derivative": getattr(args, "nn_hidden_derivative", 0.0),
                # Compositional ||v_{t+1} - 2 v_t + v_{t-1}||² output-jerk
                # term, off-by-default. Enable via --nn-output-jerk
                # (e.g. 1e5 per Shahbazi et al. 2025 Eq. 1). Bug: efc4d68
                # (feedbax 7e1d257)
                "nn_output_jerk": getattr(args, "nn_output_jerk", 0.0),
                # Pre-go controller-output penalty (epochs 0+1, before the go
                # cue). Wraps the standard nn_output squared-L2 term in
                # EpochMaskedLoss; off-by-default. Enable via
                # --nn-output-pre-go (suggested 1e-2 ≈ 1000x the post-aggregated
                # nn_output weight). Bug: efc4d68 (feedbax 50507a9)
                "nn_output_pre_go": getattr(args, "nn_output_pre_go", 0.0),
                # Pre-go hidden-state-derivative penalty (epochs 0+1).
                # Companion to the motor-pre-go term — included so the
                # "suppress preparation too" comparator is one flag away.
                # Off-by-default. Bug: efc4d68 (feedbax 50507a9)
                "nn_hidden_derivative_pre_go": getattr(
                    args, "nn_hidden_derivative_pre_go", 0.0
                ),
            },
            "effector_pos_late": {
                "start_step_after_go": getattr(
                    args, "effector_pos_late_start_step", 80
                ),
                "final_scale_factor": getattr(
                    args, "effector_pos_late_final_scale", 2.0
                ),
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            # Power-law schedule: "flat" (default) or "powerlaw" ((t/T-1)^power).
            # Bug: 2e1a6ad
            "effector_pos_running_schedule": getattr(
                args, "effector_pos_running_schedule", "flat"
            ),
            "effector_hold_pos_schedule": getattr(
                args, "effector_hold_pos_schedule", "flat"
            ),
            "position_powerlaw_power": getattr(args, "position_powerlaw_power", 6.0),
            "movement_ramp_shape": getattr(args, "movement_ramp_shape", "linear"),
            "movement_ramp_duration_steps": getattr(
                args, "movement_ramp_duration_steps", 60
            ),
            "movement_ramp_power": getattr(args, "movement_ramp_power", 2.0),
        },
        "loss_update": {
            "enabled": args.loss_update_enabled,
            "target_ratio": args.loss_update_ratio,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
        "where": {
            0: ["nodes.net.hidden", "nodes.net.readout"],
        },
        # hidden_type is a callable (class or partial), not serialisable to JSON.
        # It is resolved here from the CLI string and stored directly in the namespace.
        "hidden_type": _resolve_hidden_type(args.hidden_type, dt),
        "sisu_gating": args.sisu_gating,
    }
    return dict_to_namespace(hps_dict, to_type=TreeNamespace)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR_NAME = "checkpoints_adversarial"
_CHECKPOINT_SUBDIR = "checkpoint_latest"


def _save_adversarial_checkpoint(
    checkpoint_dir: Path,
    flat_model: list,
    treedef_model,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state,
    batch_idx: int,
    adv_losses: list,
    ctrl_losses: list,
    adv_indices: list,
) -> None:
    """Save adversarial training state to a checkpoint directory.

    Writes atomically: assembles state in a temp dir then renames it over
    the previous checkpoint so a preempted write never leaves corrupt state.

    The model is serialized with ``eqx.tree_serialise_leaves`` (template needed
    at load time is the current ``adv_model`` reconstructed via ``treedef_model``).
    Adversaries and optimizer states are similarly serialized.

    Args:
        checkpoint_dir: Parent directory; ``checkpoint_latest/`` is created inside.
        flat_model: Flat list of model array leaves (from ``jtu.tree_flatten``).
        treedef_model: PyTree treedef for reconstructing the model.
        adversaries: List of ``GaussianBumpAdversary`` instances.
        adv_opt_states: List of adversary optimizer states (one per adversary).
        ctrl_opt_state: Controller optimizer state.
        batch_idx: Index of the batch that was just completed.
        adv_losses: Adversary loss history (up to and including ``batch_idx``).
        ctrl_losses: Controller loss history.
        adv_indices: Active adversary index history.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / _CHECKPOINT_SUBDIR

    # Write into a sibling temp dir for atomic replacement.
    tmp_dir = checkpoint_dir / (f"_{_CHECKPOINT_SUBDIR}_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    try:
        # 1. Model: reconstruct → serialize leaves
        model = jtu.tree_unflatten(treedef_model, flat_model)
        eqx.tree_serialise_leaves(tmp_dir / "model.eqx", model)

        # 2. Adversaries: serialize each
        for i, adv in enumerate(adversaries):
            eqx.tree_serialise_leaves(tmp_dir / f"adversary_{i}.eqx", adv)

        # 3. Optimizer states: serialize
        eqx.tree_serialise_leaves(tmp_dir / "ctrl_opt_state.eqx", ctrl_opt_state)
        for i, opt_st in enumerate(adv_opt_states):
            eqx.tree_serialise_leaves(tmp_dir / f"adv_opt_state_{i}.eqx", opt_st)

        # 4. Scalar progress + loss histories
        meta = {
            "batch_idx": batch_idx,
            "n_adversaries": len(adversaries),
            "adv_losses": adv_losses,
            "ctrl_losses": ctrl_losses,
            "adv_indices": adv_indices,
        }
        with open(tmp_dir / "meta.json", "w") as fh:
            json.dump(meta, fh)

        # Atomic rename
        if target.exists():
            shutil.rmtree(target)
        tmp_dir.rename(target)

    except Exception:
        # Leave the tmp dir for debugging; do not corrupt the previous checkpoint.
        logger.exception("Checkpoint save failed — previous checkpoint (if any) is intact")
        raise


def _load_adversarial_checkpoint(
    checkpoint_dir: Path,
    model_template,
    adversaries_template: list,
    adv_opt_states_template: list,
    ctrl_opt_state_template,
    treedef_model,
):
    """Load adversarial training state from ``checkpoint_latest/``.

    Args:
        checkpoint_dir: Parent directory containing ``checkpoint_latest/``.
        model_template: Model object with the correct PyTree structure (template).
        adversaries_template: List of ``GaussianBumpAdversary`` instances with the
            correct structure (used as deserialization templates).
        adv_opt_states_template: List of adversary optimizer states (templates).
        ctrl_opt_state_template: Controller optimizer state (template).
        treedef_model: PyTree treedef for the model (used to re-flatten the result).

    Returns:
        Tuple of ``(flat_model, adversaries, adv_opt_states, ctrl_opt_state,
        resume_batch_idx, adv_losses, ctrl_losses, adv_indices)``.
    """
    target = checkpoint_dir / _CHECKPOINT_SUBDIR
    if not target.exists():
        raise FileNotFoundError(f"No checkpoint found at {target}")

    # 1. Model — deserialize onto the template; then flatten for the training loop
    model = eqx.tree_deserialise_leaves(target / "model.eqx", model_template)
    flat_model = jtu.tree_flatten(model)[0]

    # 2. Adversaries
    adversaries = []
    for i, tmpl in enumerate(adversaries_template):
        adv = eqx.tree_deserialise_leaves(target / f"adversary_{i}.eqx", tmpl)
        adversaries.append(adv)

    # 3. Optimizer states
    ctrl_opt_state = eqx.tree_deserialise_leaves(
        target / "ctrl_opt_state.eqx", ctrl_opt_state_template
    )
    adv_opt_states = []
    for i, tmpl in enumerate(adv_opt_states_template):
        opt_st = eqx.tree_deserialise_leaves(target / f"adv_opt_state_{i}.eqx", tmpl)
        adv_opt_states.append(opt_st)

    # 4. Meta
    with open(target / "meta.json") as fh:
        meta = json.load(fh)

    return (
        flat_model,
        adversaries,
        adv_opt_states,
        ctrl_opt_state,
        meta["batch_idx"],
        meta["adv_losses"],
        meta["ctrl_losses"],
        meta["adv_indices"],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_trainable(model):
    """Return the trainable leaves of the model.

    Default (SimpleStagedNetwork): (net.hidden, net.readout). Linear-controller
    MVP variants (Bug: 410d7ac) carry their parameters as ``K`` (and ``u_ff``
    for the tracker) directly on the net Component — branch on Module class
    name to keep this single function compatible with both code paths.
    """
    net = model.nodes["net"]
    cls_name = type(net).__name__
    if cls_name == "LinearController":
        return (net.K,)
    if cls_name == "LinearTrackerController":
        return (net.K, net.u_ff)
    return (net.hidden, net.readout)


def _trainable_where(model):
    """Return the ``eqx.tree_at``-compatible selector lambda for trainable leaves.

    Mirrors ``_get_trainable`` but returns a ``where`` function (mapping a model
    to its trainable subtrees) instead of the subtrees themselves. Required by
    ``eqx.tree_at`` to splice updated parameters back into the model PyTree.
    Bug: 410d7ac — the linear-controller MVP variants do not have
    ``net.hidden`` / ``net.readout`` attributes, so the default selector
    triggers an AttributeError during the adversarial controller step. This
    helper centralises the architecture branch.
    """
    net = model.nodes["net"]
    cls_name = type(net).__name__
    if cls_name == "LinearController":
        return lambda m: (m.nodes["net"].K,)
    if cls_name == "LinearTrackerController":
        return lambda m: (m.nodes["net"].K, m.nodes["net"].u_ff)
    return lambda m: (m.nodes["net"].hidden, m.nodes["net"].readout)


def _eval_trials_streaming(task, model, trial_specs, keys, loss_func):
    """Evaluate trials with streaming loss — no trajectory stored.

    Mirrors ``task.eval_trials`` but passes a ``streaming_loss_fn`` to
    ``run_component`` so that loss is accumulated inside the scan body.
    Returns the mean scalar loss across the batch (no state history).

    Args:
        task: The task instance (provides intervention_state_indices, etc.).
        model: The model to evaluate.
        trial_specs: Batched trial specifications (leading batch dim).
        keys: Per-trial PRNG keys, shape ``(batch_size,)``.
        loss_func: The loss function (``AbstractLoss``); used to build the
            per-step streaming closure via ``make_streaming_loss_fn``.

    Returns:
        Scalar loss averaged over the batch.
    """
    # Bug: 3ef9c25 — streaming loss integration for memory-efficient training
    def eval_single(trial_spec, key):
        key_init, key_run = jr.split(key)
        init_state = init_state_from_component(model)

        for where_substate, init_substate in trial_spec.inits.items():
            path = _where_key_to_path(where_substate)
            init_state = _set_state_by_path(model, init_state, path, init_substate)

        # Apply intervention params (same logic as task.eval_trials)
        intervene_inputs = {}
        if trial_spec.intervene:
            indices = model.intervention_state_indices()
            for label, params in trial_spec.intervene.items():
                if label not in indices:
                    raise ValueError(f"Unknown intervention label '{label}'")
                idx = indices[label]
                current = init_state.get(idx)

                def _merge_leaf(p, c):
                    if isinstance(p, TimeSeriesParam):
                        return c
                    if p is None:
                        return c
                    return p

                merged = jt.map(
                    _merge_leaf,
                    params, current,
                    is_leaf=lambda x: x is None or isinstance(x, TimeSeriesParam),
                )
                init_state = _safe_state_set(init_state, idx, merged)
                tv_params = _extract_timeseries_params(params, current)
                if tv_params is not None:
                    intervene_inputs[label] = tv_params

        init_state = model.state_consistency_update(init_state)

        inputs = _prepare_inputs(model, trial_spec.inputs)
        if intervene_inputs:
            inputs = _merge_intervene_inputs(inputs, intervene_inputs)
        n_steps = _infer_n_steps(inputs, trial_spec.timeline)

        # Build per-step streaming loss closure (single trial, no batch dim)
        streaming_fn = make_streaming_loss_fn(loss_func, trial_spec, model, n_steps)

        checkpoint = getattr(model, 'checkpoint', False)
        _outputs, _final_state, total_loss = run_component(
            model,
            inputs,
            init_state,
            key=key_run,
            n_steps=n_steps,
            streaming_loss_fn=streaming_fn,
        )
        return total_loss

    per_trial_losses = eqx.filter_vmap(eval_single)(trial_specs, keys)
    return per_trial_losses.mean()


def _make_where_train(sisu_gating: str = "additive"):
    """Return the where_train dict for the controller optimizer."""
    def where_train_fn(model):
        net = model.nodes["net"]
        cls_name = type(net).__name__
        # Linear-controller MVP variants (Bug: 410d7ac) carry their parameters
        # directly on the net Component as K (+ u_ff for the tracker).
        if cls_name == "LinearController":
            return (net.K,)
        if cls_name == "LinearTrackerController":
            return (net.K, net.u_ff)
        params = [net.hidden, net.readout]
        # Include sisu_alpha in trainable params when using multiplicative gating
        if sisu_gating == "multiplicative" and net.sisu_alpha is not None:
            params.append(net.sisu_alpha)
        return tuple(params)
    return {0: where_train_fn}


def run_training(args: argparse.Namespace) -> None:
    """Run minimax adversarial training."""
    _configure_jax_runtime(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spec dir holds tracked recipe/run.json; artifact dir holds bulk outputs.
    # When --spec-dir is unset, derive it from --output-dir via the mirror
    # invariant (paths.run_artifact_dir(exp, run) ↔ paths.run_spec_dir(exp, run)).
    # Bug: 0077b42
    spec_dir = (
        Path(args.spec_dir) if args.spec_dir is not None
        else derive_spec_dir(output_dir)
    )
    mkdir_p(spec_dir)

    config_dict = {
        **vars(args),
        "git": _get_git_metadata(),
        "gpu_info": _collect_gpu_info(),
    }
    spec_path = spec_dir / "run.json"
    with open(spec_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info("Saved run spec to %s", spec_path)

    hps = build_hps(args)

    key = jr.PRNGKey(args.seed)
    key_init, key_warmup, key_adv = jr.split(key, 3)

    # -----------------------------------------------------------------------
    # Task / model setup
    # -----------------------------------------------------------------------
    logger.info(
        "Setting up task-model pair (hidden_type=%s, sisu_gating=%s)",
        args.hidden_type, args.sisu_gating,
    )
    pair = setup_task_model_pair(hps, key=key_init)
    task = pair.task
    loss_func = task.loss_func

    # Build a loss computation closure that abstracts over standard vs
    # streaming evaluation.  Streaming mode accumulates loss inside the scan
    # body, avoiding storage of the full state trajectory.  Bug: 3ef9c25
    use_streaming_loss = args.streaming_loss
    if use_streaming_loss:
        logger.info("Using STREAMING loss (no trajectory storage)")

        def _compute_loss(model, trial_specs, keys):
            return _eval_trials_streaming(task, model, trial_specs, keys, loss_func)
    else:
        def _compute_loss(model, trial_specs, keys):
            states = task.eval_trials(model, trial_specs, keys)
            return loss_func(states, trial_specs, model).total.mean()

    where_train = _make_where_train(sisu_gating=args.sisu_gating)

    # -----------------------------------------------------------------------
    # Phase 1 — warm-start (or load pre-trained model)
    # -----------------------------------------------------------------------
    warmup_model_path = output_dir / "warmup_model.eqx"

    warmup_model = None
    warmup_history = None

    # When resuming, load an already-saved warmup_model.eqx using
    # load_with_hyperparameters and this script's build_hps, since
    # warmup_model.eqx was saved with fbx_save (HDF5 format), not eqx's
    # native format. Using eqx.tree_deserialise_leaves would fail with
    # a TreePathError.
    if args.resume and warmup_model_path.exists():
        logger.info(
            "--resume: loading warmup_model.eqx from %s — skipping phase 1.",
            warmup_model_path,
        )

        def _resume_setup_func(key=jr.PRNGKey(0), **stored_hps):
            """Reconstruct model from stored config for resume."""
            # Filter out non-hps keys that are stored in config.json but not
            # expected by build_hps
            for k in ("git", "output_dir", "checkpoint_every", "resume"):
                stored_hps.pop(k, None)
            resume_args = argparse.Namespace(**stored_hps)
            resume_hps = build_hps(resume_args)
            return setup_task_model_pair(resume_hps, key=key).model

        warmup_model, _ = load_with_hyperparameters(
            warmup_model_path, setup_func=_resume_setup_func
        )

    if warmup_model is None and args.warmup_model is not None:
        from feedbax._io import load as fbx_load
        logger.info("Loading pre-trained warm-start model from %s", args.warmup_model)

        def _model_setup_func(key=jr.PRNGKey(0), **stored_hps):
            """Reconstruct a model from stored hyperparameters."""
            # The stored hps come from train_part2_5.py, not train_minimax.py,
            # so use its build_hps function
            from train_part2_5 import build_hps as build_hps_standard
            stored_hps.pop("git", None)
            stored_hps.pop("output_dir", None)
            stored_args = argparse.Namespace(**stored_hps)
            stored_hps_obj = build_hps_standard(stored_args)
            return setup_task_model_pair(stored_hps_obj, key=key).model

        warmup_model = fbx_load(args.warmup_model, setup_func=_model_setup_func)
        logger.info("Loaded warm-start model (skipping phase 1).")

    if warmup_model is None:
        logger.info("Phase 1: warm-start for %d batches (controller_lr=%g)",
                    args.n_warmup_batches, args.controller_lr)

        warmup_schedule = make_delayed_cosine_schedule(
            args.controller_lr,
            constant_steps=0,
            total_steps=args.n_warmup_batches,
        )
        warmup_optimizer = optax.inject_hyperparams(
            partial(optax.adamw, weight_decay=0.0)
        )(learning_rate=warmup_schedule)

        chkpt_dir = output_dir / "checkpoints_warmup"
        chkpt_dir.mkdir(parents=True, exist_ok=True)
        warmup_trainer = TaskTrainer(
            optimizer=warmup_optimizer,
            checkpointing=True,
            chkpt_dir=chkpt_dir,
        )

        warmup_model, warmup_history = train_pair(
            warmup_trainer,
            pair,
            n_batches=args.n_warmup_batches,
            key=key_warmup,
            ensembled=True,
            loss_func=loss_func,
            where_train=where_train,
            batch_size=hps.batch_size,
            log_step=max(1, args.n_warmup_batches // 20),
        )
        logger.info("Warm-start complete.")

    # Enable jax.checkpoint on the model's scan body to trade compute for VRAM.
    # Must be done before flattening the model for the adversarial phase.
    # The checkpoint field is static so it doesn't affect array shapes.
    if args.checkpoint:
        try:
            object.__setattr__(warmup_model, "checkpoint", True)
            logger.info("Enabled jax.checkpoint on model scan body")
        except Exception:
            logger.warning("Could not enable checkpoint on model — flag has no effect")

    # Save warm-started model (skip if we loaded it from this path via --resume)
    fbx_save(warmup_model_path, warmup_model, hyperparameters=config_dict)
    logger.info("Saved warm-start model to %s", warmup_model_path)

    # -----------------------------------------------------------------------
    # Optional: swap plant intervenor for the adversarial phase.
    # When --adversary-type linear_dynamics, replace the warmup-phase
    # FixedField/CurlField intervenor with DynamicsMatrixPerturb so the
    # adversary can drive ΔA·x perturbations through the same disturbance
    # channel. Bug: c723082.
    # -----------------------------------------------------------------------
    use_linear_dynamics = (args.adversary_type == "linear_dynamics")
    if use_linear_dynamics:
        from rlrmp.disturbance import PLANT_INTERVENOR_LABEL as _PLABEL
        logger.info(
            "Swapping plant intervenor at label %r to DynamicsMatrixPerturb "
            "(eta_max=%g, pgd_steps=%d, pgd_lr=%g) for adversarial phase",
            _PLABEL, args.linear_dynamics_eta_max,
            args.linear_dynamics_pgd_steps, args.linear_dynamics_lr,
        )
        # Swap on the ensembled model (jt.map over the ensemble pytree).
        warmup_model = jt.map(
            lambda m: swap_plant_intervenor_to_dynamics_matrix(
                m, _PLABEL, mass=hps.model.effector_mass,
            ),
            warmup_model,
            is_leaf=lambda x: x is not None and hasattr(x, "nodes")
                              and hasattr(x, "input_ports"),
        )
        # Swap on the task so trial_specs.intervene[label] has the right type.
        task = swap_task_intervention_to_dynamics_matrix(task, _PLABEL)

    # -----------------------------------------------------------------------
    # Phase 2 — adversarial training
    # -----------------------------------------------------------------------
    n_adversaries = args.n_adversaries
    n_reps = hps.model.n_replicates
    logger.info(
        "Phase 2: adversarial training for %d batches "
        "(adversary_type=%s, n_replicates=%d vmapped, n_adversaries=%d, "
        "n_adversary_steps=%d, adversary_lr=%g, controller_lr=%g, "
        "loss_update_enabled=%s, loss_update_ratio=%g)",
        args.n_adversary_batches, args.adversary_type, n_reps, n_adversaries,
        args.n_adversary_steps, args.adversary_lr, args.controller_lr,
        args.loss_update_enabled, args.loss_update_ratio,
    )

    # Create adversary population (K independent adversaries with different seeds).
    # Each adversary is vmapped across n_reps replicates so every replicate gets
    # its own independent adversary parameters.
    n_timesteps = hps.task.n_steps - 1  # feedbax uses n_steps-1 as the sim length

    def _make_adversary_population(n_adversaries: int) -> list:
        """Create K adversaries, each vmapped over n_reps replicates.

        Dispatches on ``args.adversary_type``:
        - ``gaussian_bump``: ``GaussianBumpAdversary`` (force-profile).
        - ``linear_dynamics``: ``LinearDynamicsAdversary`` (ΔA·x).
        """
        pop = []
        for i in range(n_adversaries):
            # Each replicate within adversary i gets a unique key
            rep_keys = jr.split(jr.PRNGKey(7 + i), n_reps)
            if args.adversary_type == "gaussian_bump":
                adv_vmapped = eqx.filter_vmap(
                    lambda k: GaussianBumpAdversary(
                        n_bumps=args.n_bumps,
                        n_timesteps=n_timesteps,
                        n_force_dims=2,
                        force_max=args.force_max,
                        dt=hps.dt,
                        key=k,
                    )
                )(rep_keys)
            elif args.adversary_type == "linear_dynamics":
                adv_vmapped = eqx.filter_vmap(
                    lambda k: LinearDynamicsAdversary(
                        n_state=4,
                        n_dim=2,
                        eta_max=args.linear_dynamics_eta_max,
                        n_inner_steps=args.n_adversary_steps,
                        learning_rate=args.linear_dynamics_lr,
                        key=k,
                    )
                )(rep_keys)
            else:
                raise ValueError(f"Unknown adversary_type: {args.adversary_type}")
            pop.append(adv_vmapped)
        return pop

    adversaries = _make_adversary_population(n_adversaries)

    # Adversary optimizers (one per adversary population member).
    # Init on a single-replicate adversary, then stack n_reps copies so ALL
    # state arrays (including step counters) carry a leading (n_reps,) axis.
    # The linear_dynamics adversary uses its own learning_rate (passed via
    # CLI), distinct from the GaussianBump adversary's --adversary-lr.
    adv_lr = (
        args.linear_dynamics_lr if args.adversary_type == "linear_dynamics"
        else args.adversary_lr
    )
    adversary_optimizer = optax.adam(adv_lr)

    def _init_vmapped_opt_state(vmapped_adv):
        """Init optimizer on one replicate's adversary, stack for all reps."""
        single_adv = jt.map(
            lambda x: x[0] if (eqx.is_array(x) and x.ndim > 0) else x,
            vmapped_adv, is_leaf=eqx.is_array,
        )
        single_st = adversary_optimizer.init(eqx.filter(single_adv, eqx.is_array))
        return jt.map(
            lambda x: jnp.stack([x] * n_reps) if eqx.is_array(x) else x,
            single_st, is_leaf=eqx.is_array,
        )

    adv_opt_states = [_init_vmapped_opt_state(adv) for adv in adversaries]

    # Controller optimizer (constant LR for the adversarial phase).
    # We train only the recurrent net weights (hidden + readout), same as TaskTrainer.
    ctrl_optimizer = optax.adamw(args.controller_lr, weight_decay=0.0)

    # The model from train_pair with ensembled=True has a leading replicate axis
    # on MOST array leaves: shape (n_reps, ...). Some arrays are shared across
    # replicates (no leading axis). We separate these for vmapping.
    #
    # Pre-flatten strategy (Bug: d6cc111): feedbax models store JAX arrays as
    # static PyTree metadata, so passing the model directly to filter_jit causes
    # recompilation after every update. We pass only flat arrays (all dynamic)
    # and close over the treedef (fixed).
    #
    # For vmapping: we split arrays into "per-replicate" (leading n_reps axis)
    # and "shared" (no replicate axis). The vmapped function receives per-rep
    # arrays; shared arrays are closed over. Both are recombined inside the
    # function via treedef_model.unflatten.

    def _has_rep_axis(x):
        return eqx.is_array(x) and x.ndim > 0 and x.shape[0] == n_reps

    # Extract single replicate for treedef (structure is same across replicates)
    single_rep_model = jt.map(
        lambda x: x[0] if _has_rep_axis(x) else x,
        warmup_model,
        is_leaf=eqx.is_array,
    )
    flat_single_rep, treedef_model = jtu.tree_flatten(single_rep_model)

    # Build masks: which flat leaves are per-replicate vs shared
    flat_ensembled = jtu.tree_flatten(warmup_model)[0]
    is_per_rep = [_has_rep_axis(x) for x in flat_ensembled]

    # Split into per-replicate arrays (vmapped) and shared arrays (closed over)
    per_rep_arrays = [x for x, pr in zip(flat_ensembled, is_per_rep) if pr]
    shared_leaves = [x for x, pr in zip(flat_ensembled, is_per_rep) if not pr]

    def _recombine_flat(per_rep_list, shared_list):
        """Recombine per-rep and shared leaves into a full flat list."""
        result = []
        pr_idx, sh_idx = 0, 0
        for is_pr in is_per_rep:
            if is_pr:
                result.append(per_rep_list[pr_idx])
                pr_idx += 1
            else:
                result.append(shared_list[sh_idx])
                sh_idx += 1
        return result

    def _split_flat(flat_list):
        """Split a full flat list into per-rep and shared portions."""
        per_rep = [x for x, pr in zip(flat_list, is_per_rep) if pr]
        shared = [x for x, pr in zip(flat_list, is_per_rep) if not pr]
        return per_rep, shared

    # The "flat_model" for the training loop is just the per-replicate arrays.
    # Shared arrays are closed over inside the vmapped functions.
    flat_model = per_rep_arrays

    # Also keep the full ensembled treedef for final model reconstruction
    _, treedef_ensembled = jtu.tree_flatten(warmup_model)

    # Initialise controller optimizer state on a single replicate, then stack
    # n_reps copies so ALL state arrays (including step counters) carry a leading
    # (n_reps,) axis for the vmapped training loop.
    single_rep_ctrl_state = ctrl_optimizer.init(
        eqx.filter(_get_trainable(single_rep_model), eqx.is_array)
    )
    ctrl_opt_state = jt.map(
        lambda x: jnp.stack([x] * n_reps) if eqx.is_array(x) else x,
        single_rep_ctrl_state,
        is_leaf=eqx.is_array,
    )

    # ---------------------------------------------------------------------------
    # JIT-compiled training steps — defined as closures over task, loss_func,
    # ctrl_optimizer, and treedef_model, all of which are fixed for the entire
    # adversarial phase. The model is passed as a flat list of arrays (flat_model)
    # so that eqx.filter_jit sees only dynamic leaves in its argument, never
    # static metadata that changes with each update. Bug: d6cc111
    #
    # All functions operate on a SINGLE replicate internally. They are wrapped
    # with eqx.filter_vmap at the call site so each replicate trains its own
    # model + adversary independently, sharing only trial_specs and PRNG keys.
    # ---------------------------------------------------------------------------

    # Adversarial phase batch size (may differ from warmup to reduce XLA compile time)
    adv_batch_size = args.adv_batch_size if args.adv_batch_size is not None else hps.batch_size
    logger.info("Adversarial phase batch size: %d", adv_batch_size)

    n_adversary_steps = args.n_adversary_steps

    def _unflatten_model(per_rep_flat):
        """Reconstruct a single-replicate model from per-rep arrays + shared leaves.

        Merges the dynamic per-replicate arrays with the static shared leaves
        (closed over), then unflattens using treedef_model.
        """
        full_flat = _recombine_flat(per_rep_flat, shared_leaves)
        return jtu.tree_unflatten(treedef_model, full_flat)

    def _reflatten_model(model):
        """Extract only the per-replicate arrays from a single-replicate model."""
        full_flat = jtu.tree_flatten(model)[0]
        per_rep, _ = _split_flat(full_flat)
        return per_rep

    def _single_rep_loss_and_force_grad(per_rep_flat, adversary, trial_specs, keys):
        """Compute loss and gradient w.r.t. force array for a single replicate.

        Args:
            per_rep_flat: Per-replicate model arrays for ONE replicate.
            adversary: GaussianBumpAdversary for this replicate.
            trial_specs: Batched trial specifications (shared across replicates).
            keys: Per-trial PRNG keys, shape (batch_size,).

        Returns:
            Tuple of (loss_scalar, dL_dforces) where dL_dforces has shape
            (batch_size, T, d).
        """
        model = _unflatten_model(per_rep_flat)
        model_sg = jt.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            model, is_leaf=eqx.is_array,
        )

        force_profile = adversary()  # (T, d)
        forces = jnp.broadcast_to(force_profile, (adv_batch_size, *force_profile.shape))

        def _loss_fn(f):
            ts = _inject_adversary_forces(trial_specs, f)
            return _compute_loss(model_sg, ts, keys)

        return jax.value_and_grad(_loss_fn)(forces)

    def _single_rep_controller_step(per_rep_flat, ctrl_opt_st, adversary,
                                    trial_specs, keys):
        """Single gradient-descent step on the controller for one replicate.

        Args:
            per_rep_flat: Per-replicate model arrays for ONE replicate.
            ctrl_opt_st: Controller optimizer state for this replicate.
            adversary: GaussianBumpAdversary for this replicate.
            trial_specs: Batched trial specifications (shared across replicates).
            keys: Per-trial PRNG keys, shape (batch_size,).

        Returns:
            Tuple of (per_rep_flat_updated, updated_opt_state, loss_scalar).
        """
        model = _unflatten_model(per_rep_flat)

        force_profile = adversary()  # (T, d)
        forces = jnp.broadcast_to(force_profile, (adv_batch_size, *force_profile.shape))
        adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

        def _ctrl_loss(m):
            return _compute_loss(m, adv_trial_specs, keys)

        loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)

        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, new_opt_st = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            _trainable_where(model),
            model,
            updated_trainable,
        )
        return _reflatten_model(new_model), new_opt_st, loss_val

    # ---------------------------------------------------------------------------
    # Fused adversary batch — single JIT call replaces K×2 + 1 round-trips.
    # Uses lax.fori_loop for the inner adversary ascent steps, then performs a
    # single controller descent step. Closes over the same fixed objects as the
    # decomposed functions above. Bug: d6cc111
    # ---------------------------------------------------------------------------

    def _single_rep_fused_batch(per_rep_flat, adversary, adv_opt_st, ctrl_opt_st,
                                trial_specs, keys):
        """Fused adversary inner loop + controller step for a single replicate.

        Args:
            per_rep_flat: Per-replicate model arrays for ONE replicate.
            adversary: GaussianBumpAdversary for this replicate.
            adv_opt_st: Adversary optimizer state for this replicate.
            ctrl_opt_st: Controller optimizer state for this replicate.
            trial_specs: Batched trial specifications (shared across replicates).
            keys: Per-trial PRNG keys, shape (batch_size,).

        Returns:
            Tuple of (per_rep_flat_new, adversary_new, adv_opt_st_new,
            ctrl_opt_st_new, adv_loss, ctrl_loss).
        """
        model = _unflatten_model(per_rep_flat)
        model_sg = jt.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            model, is_leaf=eqx.is_array,
        )

        # --- Inner adversary loop (K ascent steps) via lax.fori_loop ---
        def _adv_body(i, carry):
            adv, opt_st, _last_loss = carry

            force_profile = adv()  # (T, d)
            forces = jnp.broadcast_to(
                force_profile, (adv_batch_size, *force_profile.shape)
            )

            def _loss_fn(f):
                ts = _inject_adversary_forces(trial_specs, f)
                return _compute_loss(model_sg, ts, keys)

            loss_val, dL_dforces = jax.value_and_grad(_loss_fn)(forces)

            def _forces_fn(a):
                fp = a()
                return jnp.broadcast_to(fp, dL_dforces.shape)

            _, vjp_fn = jax.vjp(
                lambda a: eqx.filter(_forces_fn(a), eqx.is_array), adv
            )
            neg_dL = jt.map(lambda g: -g, dL_dforces)
            (param_grads,) = vjp_fn(neg_dL)

            updates, new_opt_st = adversary_optimizer.update(
                eqx.filter(param_grads, eqx.is_array),
                opt_st,
                eqx.filter(adv, eqx.is_array),
            )
            new_adv = eqx.apply_updates(adv, updates)
            return new_adv, new_opt_st, loss_val

        init_carry = (adversary, adv_opt_st, jnp.float32(0.0))
        adversary_new, adv_opt_st_new, adv_loss = jax.lax.fori_loop(
            0, n_adversary_steps, _adv_body, init_carry
        )

        # --- Controller descent step (1 step) ---
        force_profile = adversary_new()
        forces = jnp.broadcast_to(
            force_profile, (adv_batch_size, *force_profile.shape)
        )
        adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

        def _ctrl_loss(m):
            return _compute_loss(m, adv_trial_specs, keys)

        ctrl_loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)

        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, ctrl_opt_st_new = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            _trainable_where(model),
            model,
            updated_trainable,
        )

        return (_reflatten_model(new_model), adversary_new, adv_opt_st_new,
                ctrl_opt_st_new, adv_loss, ctrl_loss_val)

    def _single_rep_fused_batch_linear_dynamics(
        per_rep_flat, adversary, adv_opt_st, ctrl_opt_st,
        trial_specs, keys,
    ):
        """Fused inner-loop + controller step for ``LinearDynamicsAdversary``.

        Mirrors ``_single_rep_fused_batch`` but injects the adversary's
        ``ΔA`` matrix into trial_specs via ``_inject_adversary_delta_A``,
        and applies a Frobenius-ball projection after each PGD step. Bug:
        c723082.
        """
        model = _unflatten_model(per_rep_flat)
        model_sg = jt.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            model, is_leaf=eqx.is_array,
        )

        def _adv_body(i, carry):
            adv, opt_st, _last_loss = carry

            # Loss as a function of ``adv.delta_A`` directly.
            def _loss_fn(a):
                ts = _inject_adversary_delta_A(
                    trial_specs, a.delta_A, adv_batch_size,
                )
                return _compute_loss(model_sg, ts, keys)

            loss_val, grads = eqx.filter_value_and_grad(_loss_fn)(adv)
            # Negate for gradient ascent
            neg_grads = jt.map(
                lambda g: -g if eqx.is_array(g) else g, grads,
            )
            updates, new_opt_st = adversary_optimizer.update(
                eqx.filter(neg_grads, eqx.is_array),
                opt_st,
                eqx.filter(adv, eqx.is_array),
            )
            new_adv = eqx.apply_updates(adv, updates)
            # Project to Frobenius ball (||delta_A||_F ≤ eta_max)
            new_adv = new_adv.project()
            return new_adv, new_opt_st, loss_val

        init_carry = (adversary, adv_opt_st, jnp.float32(0.0))
        adversary_new, adv_opt_st_new, adv_loss = jax.lax.fori_loop(
            0, n_adversary_steps, _adv_body, init_carry,
        )

        # --- Controller descent step (1 step) ---
        adv_trial_specs = _inject_adversary_delta_A(
            trial_specs, adversary_new.delta_A, adv_batch_size,
        )

        def _ctrl_loss(m):
            return _compute_loss(m, adv_trial_specs, keys)

        ctrl_loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)
        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, ctrl_opt_st_new = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            _trainable_where(model),
            model,
            updated_trainable,
        )

        return (_reflatten_model(new_model), adversary_new, adv_opt_st_new,
                ctrl_opt_st_new, adv_loss, ctrl_loss_val)

    # ---------------------------------------------------------------------------
    # Vmapped + JIT wrappers: vmap over replicate axis (0) for model, adversary,
    # and optimizer states; trial_specs and keys are broadcast (shared).
    # ---------------------------------------------------------------------------

    @eqx.filter_jit
    def _vmapped_fused_batch(flat_model, adversary, adv_opt_st, ctrl_opt_st,
                             trial_specs, keys):
        """Fused adversary batch vmapped over replicates.

        flat_model, adversary, adv_opt_st, ctrl_opt_st have leading (n_reps,)
        on array leaves and are vmapped. trial_specs and keys are shared
        across replicates (closed over via lambda).

        Args:
            flat_model: Flat list of arrays, each (n_reps, ...).
            adversary: Vmapped GaussianBumpAdversary (arrays have leading n_reps).
            adv_opt_st: Vmapped adversary optimizer state.
            ctrl_opt_st: Vmapped controller optimizer state.
            trial_specs: Batched trial specs (shared across replicates).
            keys: Per-trial PRNG keys (shared across replicates).

        Returns:
            Same as _single_rep_fused_batch, with leading (n_reps,) on arrays.
        """
        # Close over trial_specs/keys so they are NOT vmapped; only the
        # per-replicate state (model, adversary, opt states) is vmapped.
        return eqx.filter_vmap(
            lambda fm, adv, aos, cos: _single_rep_fused_batch(
                fm, adv, aos, cos, trial_specs, keys
            )
        )(flat_model, adversary, adv_opt_st, ctrl_opt_st)

    @eqx.filter_jit
    def _vmapped_fused_batch_linear_dynamics(
        flat_model, adversary, adv_opt_st, ctrl_opt_st, trial_specs, keys,
    ):
        """Linear-dynamics fused adversary batch vmapped over replicates."""
        return eqx.filter_vmap(
            lambda fm, adv, aos, cos: _single_rep_fused_batch_linear_dynamics(
                fm, adv, aos, cos, trial_specs, keys
            )
        )(flat_model, adversary, adv_opt_st, ctrl_opt_st)

    @eqx.filter_jit
    def _vmapped_loss_and_force_grad(flat_model, adversary, trial_specs, keys):
        """Loss and force grad vmapped over replicates."""
        return eqx.filter_vmap(
            lambda fm, adv: _single_rep_loss_and_force_grad(
                fm, adv, trial_specs, keys
            )
        )(flat_model, adversary)

    @eqx.filter_jit
    def _vmapped_controller_step(flat_model, ctrl_opt_st, adversary, trial_specs,
                                 keys):
        """Controller step vmapped over replicates."""
        return eqx.filter_vmap(
            lambda fm, cos, adv: _single_rep_controller_step(
                fm, cos, adv, trial_specs, keys
            )
        )(flat_model, ctrl_opt_st, adversary)

    @eqx.filter_jit
    def _vmapped_adversary_update(adversary, dL_dforces, adv_opt_st):
        """Adversary update vmapped over replicates (decomposed mode only)."""
        return eqx.filter_vmap(
            partial(_adversary_update, adversary_optimizer)
        )(adversary, dL_dforces, adv_opt_st)

    # -----------------------------------------------------------------------
    # Resume from checkpoint (if requested)
    # -----------------------------------------------------------------------
    adv_checkpoint_dir = output_dir / _CHECKPOINT_DIR_NAME
    start_batch_idx = 0
    adv_losses = []
    ctrl_losses = []
    adv_indices = []  # track which adversary was active each batch

    if args.resume:
        ckpt_path = adv_checkpoint_dir / _CHECKPOINT_SUBDIR
        if ckpt_path.exists():
            logger.info("--resume: loading adversarial checkpoint from %s", ckpt_path)
            (
                full_flat_loaded,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                last_completed_batch,
                adv_losses,
                ctrl_losses,
                adv_indices,
            ) = _load_adversarial_checkpoint(
                adv_checkpoint_dir,
                warmup_model,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                treedef_ensembled,
            )
            # Extract only the per-replicate arrays for the training loop
            flat_model, _ = _split_flat(full_flat_loaded)
            start_batch_idx = last_completed_batch + 1
            logger.info(
                "Resuming adversarial training from batch %d/%d",
                start_batch_idx, args.n_adversary_batches,
            )
        else:
            logger.warning(
                "--resume was set but no checkpoint found at %s — starting from scratch.",
                ckpt_path,
            )

    # Periodic logging interval
    log_step = max(1, args.n_adversary_batches // 20)

    use_fused = args.fused
    if use_fused:
        logger.info(
            "Using FUSED adversary batch (single JIT call with lax.fori_loop "
            "for %d inner adversary steps + 1 controller step)",
            n_adversary_steps,
        )
    else:
        logger.info(
            "Using DECOMPOSED adversary batch (%d×2 + 1 = %d separate JIT "
            "calls per batch)",
            n_adversary_steps, 2 * n_adversary_steps + 1,
        )

    for batch_idx in range(start_batch_idx, args.n_adversary_batches):
        batch_key, key_adv = jr.split(key_adv)
        trial_keys = jr.split(batch_key, adv_batch_size)

        # Sample trial specs with intervenor params (needed for SISU/scale values).
        # BatchInfo fields must be JAX arrays (not Python ints) so that
        # filter_jit on get_train_trial_with_intervenor_params treats them as
        # dynamic traced values.  Python ints are static in eqx.Module and
        # would cause recompilation every batch, leaking ~0.5 GB/min of host
        # memory from the accumulated compilation cache.  Bug: d6cc111
        batch_info = BatchInfo(
            size=jnp.int32(adv_batch_size),
            current=jnp.int32(batch_idx),
            total=jnp.int32(args.n_adversary_batches),
        )
        trial_specs = jax.vmap(
            lambda key: task.get_train_trial_with_intervenor_params(key, batch_info)
        )(trial_keys)

        # task.eval_trials calls int(timeline.n_steps) which fails on traced arrays.
        # All trials have the same n_steps=140 (the fixed trial length); materialize
        # it as a concrete Python int so the call succeeds inside filter_grad.
        trial_specs = eqx.tree_at(
            lambda ts: ts.timeline.n_steps,
            trial_specs,
            int(hps.task.n_steps),
        )

        # Select active adversary (deterministic rotation for equal usage).
        # Each adversary in the population is vmapped across n_reps replicates.
        adv_idx = batch_idx % n_adversaries
        adversary = adversaries[adv_idx]
        adv_opt_state = adv_opt_states[adv_idx]
        adv_indices.append(adv_idx)

        if use_fused:
            # --- Single fused JIT call: K adversary steps + 1 controller step ---
            # Vmapped over replicates: each replicate trains independently.
            # Dispatch on adversary type (Bug: c723082).
            if use_linear_dynamics:
                fused_call = _vmapped_fused_batch_linear_dynamics
            else:
                fused_call = _vmapped_fused_batch
            (flat_model, adversary, adv_opt_state, ctrl_opt_state,
             adv_loss_vals, ctrl_loss_vals) = fused_call(
                flat_model, adversary, adv_opt_state, ctrl_opt_state,
                trial_specs, trial_keys,
            )
        else:
            # --- Decomposed: K×2 + 1 separate JIT calls per batch ---
            # Vmapped over replicates. Decomposed mode is only supported for
            # the gaussian_bump adversary; linear_dynamics requires the fused
            # path (its inner step uses ``filter_value_and_grad`` over the
            # adversary directly, not a separate force-grad VJP).
            if use_linear_dynamics:
                raise ValueError(
                    "--no-fused is not supported with --adversary-type "
                    "linear_dynamics; pass --fused (default)."
                )
            adv_loss_vals = jnp.zeros(n_reps)
            for _ in range(args.n_adversary_steps):
                adv_loss_vals, dL_dforces = _vmapped_loss_and_force_grad(
                    flat_model, adversary, trial_specs, trial_keys,
                )
                adversary, adv_opt_state = _vmapped_adversary_update(
                    adversary, dL_dforces, adv_opt_state,
                )

            # Controller update (1 descent step)
            flat_model, ctrl_opt_state, ctrl_loss_vals = _vmapped_controller_step(
                flat_model, ctrl_opt_state, adversary, trial_specs, trial_keys,
            )

        # Write back updated adversary and optimizer state
        adversaries[adv_idx] = adversary
        adv_opt_states[adv_idx] = adv_opt_state

        # Losses are (n_reps,) arrays; store per-replicate means for history
        adv_loss_mean = float(jnp.mean(adv_loss_vals))
        ctrl_loss_mean = float(jnp.mean(ctrl_loss_vals))
        adv_losses.append(adv_loss_mean)
        ctrl_losses.append(ctrl_loss_mean)

        if batch_idx % log_step == 0 or batch_idx == args.n_adversary_batches - 1:
            adv_label = f" [adv {adv_idx}]" if n_adversaries > 1 else ""
            ctrl_std = float(jnp.std(ctrl_loss_vals))
            adv_std = float(jnp.std(adv_loss_vals))
            logger.info(
                "Adversarial batch %d/%d%s — ctrl_loss=%.4g +/- %.4g, "
                "adv_loss=%.4g +/- %.4g  (n_reps=%d)",
                batch_idx, args.n_adversary_batches, adv_label,
                ctrl_loss_mean, ctrl_std, adv_loss_mean, adv_std, n_reps,
            )

        # Periodic checkpoint (save after batch_idx is complete, not before)
        checkpoint_every = args.checkpoint_every
        if checkpoint_every > 0 and (batch_idx + 1) % checkpoint_every == 0:
            logger.info(
                "Saving adversarial checkpoint at batch %d → %s",
                batch_idx, adv_checkpoint_dir / _CHECKPOINT_SUBDIR,
            )
            _save_adversarial_checkpoint(
                adv_checkpoint_dir,
                _recombine_flat(flat_model, shared_leaves),
                treedef_ensembled,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                batch_idx,
                adv_losses,
                ctrl_losses,
                adv_indices,
            )

    logger.info("Adversarial training complete.")

    # Reconstruct the ensembled model from per-replicate + shared arrays.
    # flat_model contains only per-replicate arrays (with leading n_reps axis);
    # shared_leaves are the arrays/values without the replicate axis.
    full_flat = _recombine_flat(flat_model, shared_leaves)
    adv_model = jtu.tree_unflatten(treedef_ensembled, full_flat)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # Final adversarially-trained model (ensembled: arrays have leading n_reps axis).
    # Bug: a517040 — skip when n_adversary_batches=0: the adversarial phase did not
    # run, and the saved PyTree's adversary state does not match the local skeleton
    # produced by `setup_task_model_pair` at load time, breaking deserialization.
    # Downstream loaders fall back to `warmup_model.eqx` (the correct final model).
    if args.n_adversary_batches > 0:
        final_model_path = output_dir / "adversarial_model.eqx"
        fbx_save(final_model_path, adv_model, hyperparameters=config_dict)
        logger.info(
            "Saved adversarial model (n_reps=%d ensembled) to %s", n_reps, final_model_path
        )
    else:
        logger.info(
            "Skipping adversarial_model.eqx save (n_adversary_batches=0); "
            "warmup_model.eqx is the canonical final model for this run."
        )

    # Training histories (warmup from TaskTrainer; adversarial phase as numpy arrays)
    if warmup_history is not None:
        fbx_save(output_dir / "warmup_history.eqx", warmup_history)
    loss_data = {
        "ctrl_losses": np.array(ctrl_losses),
        "adv_losses": np.array(adv_losses),
        "adv_indices": np.array(adv_indices),
    }
    np.savez(output_dir / "adversarial_losses.npz", **loss_data)
    logger.info("Saved adversarial loss curves to %s", output_dir / "adversarial_losses.npz")

    # Final adversary/adversaries (each is vmapped across n_reps replicates)
    log_fn = (
        _log_linear_dynamics_adversary if use_linear_dynamics
        else _log_adversary_force_profiles
    )
    if n_adversaries == 1:
        # Single adversary population: save with original filename for backward compat
        fbx_save(output_dir / "trained_adversary.eqx", adversaries[0])
        logger.info(
            "Saved trained adversary (n_reps=%d) to %s",
            n_reps, output_dir / "trained_adversary.eqx",
        )
        log_fn(adversaries[0], output_dir, n_reps=n_reps)
    else:
        adv_dir = output_dir / "adversaries"
        adv_dir.mkdir(parents=True, exist_ok=True)
        for i, adv in enumerate(adversaries):
            adv_path = adv_dir / f"adversary_{i}.eqx"
            fbx_save(adv_path, adv)
            logger.info("Saved adversary %d to %s", i, adv_path)
            log_fn(adv, output_dir, suffix=f"_adv{i}", n_reps=n_reps)
        logger.info("Saved %d adversaries (each n_reps=%d) to %s",
                     n_adversaries, n_reps, adv_dir)

    logger.info("All results saved to %s", output_dir)


def _log_adversary_force_profiles(
    adversary: GaussianBumpAdversary,
    output_dir: Path,
    suffix: str = "",
    n_reps: int = 1,
) -> None:
    """Log adversary force profiles (SISU-independent) to a numpy archive.

    When the adversary is vmapped across replicates, generates and saves
    per-replicate force profiles with shape (n_reps, T, 2).

    Args:
        adversary: Trained GaussianBumpAdversary (possibly vmapped with leading
            n_reps axis on array leaves).
        output_dir: Directory to write the archive into.
        suffix: Optional suffix for the output filename (e.g. "_adv0").
        n_reps: Number of replicates (for logging).
    """
    # Generate force profiles: vmapped adversary produces (n_reps, T, 2)
    forces = eqx.filter_vmap(lambda a: a())(adversary)  # (n_reps, T, 2)
    forces_np = np.array(forces)
    per_rep_norms = np.linalg.norm(forces_np.reshape(n_reps, -1), axis=-1)
    logger.info(
        "Adversary%s force profile norms: mean=%.4g +/- %.4g (n_reps=%d)",
        suffix, per_rep_norms.mean(), per_rep_norms.std(), n_reps,
    )

    filename = f"adversary_force_profiles{suffix}.npz"
    np.savez(output_dir / filename, forces=forces_np)
    logger.info("Saved adversary force profiles to %s", output_dir / filename)


def _log_linear_dynamics_adversary(
    adversary: LinearDynamicsAdversary,
    output_dir: Path,
    suffix: str = "",
    n_reps: int = 1,
) -> None:
    """Log per-replicate ``ΔA`` matrices to a numpy archive.

    Counterpart to ``_log_adversary_force_profiles`` for the
    ``LinearDynamicsAdversary`` flavour. Bug: c723082.

    Args:
        adversary: Trained ``LinearDynamicsAdversary`` (vmapped over
            replicates: ``delta_A`` has shape ``(n_reps, n_dim, n_state)``).
        output_dir: Directory to write the archive into.
        suffix: Optional filename suffix.
        n_reps: Number of replicates (for logging).
    """
    deltas = eqx.filter_vmap(lambda a: a.delta_A)(adversary)
    deltas_np = np.array(deltas)
    norms = np.linalg.norm(deltas_np.reshape(n_reps, -1), axis=-1)
    logger.info(
        "Adversary%s ΔA Frobenius norms: mean=%.4g +/- %.4g (n_reps=%d)",
        suffix, norms.mean(), norms.std(), n_reps,
    )
    filename = f"adversary_delta_A{suffix}.npz"
    np.savez(output_dir / filename, delta_A=deltas_np)
    logger.info("Saved adversary ΔA matrices to %s", output_dir / filename)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimax adversarial training for RLRMP reaching controllers."
    )
    parser.add_argument(
        "--n-warmup-batches", type=int, default=2000,
        help="Number of warm-start batches before adversarial phase (default: 2000).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=250,
        help="Per-batch trial count for warmup phase (default: 250).",
    )
    parser.add_argument(
        "--n-replicates", type=int, default=5,
        help="Number of vmapped controller replicates in the ensemble (default: 5).",
    )
    parser.add_argument(
        "--n-adversary-batches", type=int, default=8000,
        help="Number of adversarial training batches (default: 8000).",
    )
    parser.add_argument(
        "--n-adversary-steps", type=int, default=5,
        help="Inner adversary gradient-ascent steps per controller step (default: 5).",
    )
    parser.add_argument(
        "--adversary-lr", type=float, default=3e-4,
        help="Adversary learning rate (TTUR: should be 3-10x controller LR; default: 3e-4).",
    )
    parser.add_argument(
        "--controller-lr", type=float, default=1e-4,
        help="Controller learning rate during adversarial phase (default: 1e-4).",
    )
    parser.add_argument(
        "--adversary-type", type=str, default="gaussian_bump",
        choices=["gaussian_bump", "linear_dynamics"],
        help=(
            "Adversary class for the inner-loop maximisation. "
            "'gaussian_bump' (default, flavour-(a) input-instance): produces "
            "a per-trial force profile via a sum of learnable Gaussian bumps "
            "injected through FixedField. 'linear_dynamics' (flavour-(b) "
            "model-class): produces a ΔA matrix mapping [pos, vel] to a "
            "velocity-row dynamics perturbation, constrained to a Frobenius "
            "ball ||ΔA||_F ≤ eta_max via PGD; injected through "
            "DynamicsMatrixPerturb. Bug: c723082."
        ),
    )
    parser.add_argument(
        "--linear-dynamics-eta-max", type=float, default=0.1,
        help=(
            "Frobenius-norm budget for LinearDynamicsAdversary at SISU=1 "
            "(default: 0.1). SISU gating multiplies the resulting "
            "perturbation force via the intervenor's `scale` field. Only "
            "used with --adversary-type linear_dynamics."
        ),
    )
    parser.add_argument(
        "--linear-dynamics-pgd-steps", type=int, default=5,
        help=(
            "Number of inner PGD ascent steps for LinearDynamicsAdversary "
            "(default: 5). Currently mirrors --n-adversary-steps; reserved "
            "for future divergence."
        ),
    )
    parser.add_argument(
        "--linear-dynamics-lr", type=float, default=1e-2,
        help=(
            "Learning rate for the LinearDynamicsAdversary's PGD step "
            "(default: 1e-2). Larger than the GaussianBump default since "
            "ΔA values are O(eta_max) ≈ 0.1, vs O(1) for force amplitudes."
        ),
    )
    parser.add_argument(
        "--n-bumps", type=int, default=3,
        help="Number of Gaussian bumps in the adversary (default: 3).",
    )
    parser.add_argument(
        "--force-max", type=float, default=1.0,
        help="Maximum adversary force magnitude per timestep (default: 1.0).",
    )
    parser.add_argument(
        "--n-adversaries", type=int, default=1,
        help=(
            "Number of adversaries in the population (default: 1 = single adversary). "
            "When K > 1, adversaries rotate each batch (index = batch_idx %% K), "
            "providing diverse perturbation strategies."
        ),
    )
    parser.add_argument(
        "--adv-batch-size", type=int, default=None,
        help=(
            "Batch size for adversarial phase (default: same as warmup batch size). "
            "Smaller values (e.g. 64) dramatically reduce XLA compilation time."
        ),
    )
    parser.add_argument(
        "--warmup-model", type=str, default=None,
        help="Path to a pre-trained model to use as warm-start (skips phase 1).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="_artifacts/minimax/minimax_test",
        help=(
            "Output directory for bulk artifacts (checkpoints, .eqx, .npz, logs). "
            "Default mirrors the role-based layout: _artifacts/<exp>/runs/<run>/. "
            "Use rlrmp.paths.run_artifact_dir(exp, run) to construct this path "
            "programmatically. Write run.json to the sibling spec directory "
            "results/<exp>/runs/<run>/ via rlrmp.paths.run_spec_dir(exp, run)."
        ),
    )
    parser.add_argument(
        "--spec-dir", type=str, default=None,
        help=(
            "Spec directory for the tracked run.json recipe (default: derived "
            "from --output-dir via the mirror invariant, mapping "
            "_artifacts/<exp>/runs/<run>/ -> results/<exp>/runs/<run>/). "
            "Use rlrmp.paths.run_spec_dir(exp, run) to construct this path "
            "programmatically. Bug: 0077b42."
        ),
    )
    parser.add_argument(
        "--jax-cache-dir", type=str, default=None,
        help=(
            "Persistent JAX compilation cache directory. If omitted, uses "
            "JAX_COMPILATION_CACHE_DIR from the environment when set."
        ),
    )
    parser.add_argument(
        "--jax-explain-cache-misses", action="store_true",
        help="Enable JAX cache-miss diagnostics for debugging recompilation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for JAX PRNG (default: 42). Use different seeds for independent replicates.",
    )
    parser.add_argument(
        "--checkpoint", action="store_true",
        help=(
            "Enable jax.checkpoint on the model's scan body to reduce peak VRAM at ~22%% "
            "extra compute cost. Requires feedbax Graph to have a checkpoint field "
            "(available on the target GPU pod); silently no-ops otherwise."
        ),
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=500,
        help=(
            "Save adversarial training checkpoint every N batches (default: 500). "
            "Set 0 to disable. Checkpoints are written to "
            "<output-dir>/checkpoints_adversarial/checkpoint_latest/ and overwritten "
            "each time (only one checkpoint is kept)."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume adversarial training from the latest checkpoint in "
            "<output-dir>/checkpoints_adversarial/checkpoint_latest/. "
            "Skips phase 1 (warm-start) if warmup_model.eqx already exists in "
            "<output-dir>."
        ),
    )
    parser.add_argument(
        "--loss-update-enabled", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Enable adaptive loss update to drive control cost toward a target ratio "
            "of goal-state cost (default: False)."
        ),
    )
    parser.add_argument(
        "--loss-update-ratio", type=float, default=0.5,
        help=(
            "Target ratio of control cost to goal-state cost for adaptive loss update "
            "(default: 0.5). Only used when --loss-update-enabled is set."
        ),
    )
    parser.add_argument(
        "--fused", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Fuse the K adversary steps + controller step into a single JIT "
            "call using lax.fori_loop (default: True). Use --no-fused to fall "
            "back to the decomposed approach (K×2 + 1 separate JIT calls per "
            "batch) for debugging or comparison."
        ),
    )
    parser.add_argument(
        "--streaming-loss", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Accumulate loss inside the simulation scan body instead of storing "
            "the full state trajectory (default: False). Eliminates trajectory "
            "memory at the cost of requiring all loss terms to support per-step "
            "evaluation (cross-timestep losses like EffectorStraightPathLoss "
            "are not supported). Use --no-streaming-loss to disable."
        ),
    )
    parser.add_argument(
        "--hidden-type", type=str, default="gru",
        choices=["gru", "vanilla_rnn", "linear", "linear_tracker"],
        help=(
            "Controller architecture. RNNs: 'gru' (default, GRUCell with gating) "
            "or 'vanilla_rnn' (LeakyRNNCell with tau=0.1 s => alpha=0.1, no gating, "
            "diagnostic for the gating-laziness hypothesis). Linear-controller MVP "
            "variants (Bug: 410d7ac): 'linear' is a pure LTV regulator "
            "u_t=-K_t·e_t on the target-relative error state; 'linear_tracker' adds "
            "an independent LTV feedforward channel u_ff(t) for the decoupling acid "
            "test. With either linear variant, nn_hidden_size, population_structure "
            "config, and the SISU-gating mode are ignored on the model side."
        ),
    )
    parser.add_argument(
        "--nn-hidden-derivative", type=float, default=0.0,
        help=(
            "Weight on the compositional hidden-state smoothness term "
            "mean(||h_t - h_{t-1}||²) (default: 0.0 = disabled, baseline "
            "behaviour). Set to 1e-3 to mirror Shahbazi et al. 2025 Eq. 1. "
            "Bug: efc4d68."
        ),
    )
    parser.add_argument(
        "--nn-output-jerk", type=float, default=0.0,
        help=(
            "Weight on the compositional output-jerk term "
            "mean(||v_{t+1} - 2 v_t + v_{t-1}||²) on effector velocity "
            "(default: 0.0 = disabled, baseline behaviour). Set to 1e5 to "
            "mirror Shahbazi et al. 2025 Eq. 1. Bug: efc4d68 (feedbax 7e1d257)."
        ),
    )
    parser.add_argument(
        "--nn-output-pre-go", type=float, default=0.0,
        help=(
            "Weight on the pre-go controller-output penalty: "
            "EpochMaskedLoss wrapping the squared-L2 controller force, "
            "active during epochs 0+1 (hold + target_on, before the go cue) "
            "and zero afterwards. Default 0.0 = disabled. Suggested initial "
            "weight 1e-2 (≈ 1000x the post-aggregated nn_output weight) to "
            "strongly penalise pre-go anticipatory motor output without "
            "affecting post-go reach dynamics. Bug: efc4d68 (feedbax 50507a9)."
        ),
    )
    parser.add_argument(
        "--nn-hidden-derivative-pre-go", type=float, default=0.0,
        help=(
            "Weight on the pre-go hidden-state-derivative penalty: "
            "EpochMaskedLoss wrapping mean(||h_t - h_{t-1}||²), active "
            "during epochs 0+1 (before the go cue) and zero afterwards. "
            "Default 0.0 = disabled. Companion to --nn-output-pre-go for "
            "the 'suppress preparation too' comparator. "
            "Bug: efc4d68 (feedbax 50507a9)."
        ),
    )
    parser.add_argument(
        "--sisu-gating", type=str, default="additive",
        choices=["additive", "multiplicative"],
        help=(
            "How SISU enters the network: additive (default, concatenated with input) "
            "or multiplicative (post-hidden gain modulation h*(1+alpha*sisu), where "
            "alpha is a learned per-unit vector). Multiplicative forces SISU to act "
            "as neuromodulatory gain control rather than an ignorable input channel."
        ),
    )
    # ---------------------------------------------------------------------------
    # Loss-shape flags: restore historical simple_reach_loss structure.
    # These expose the loss terms that existed before the running/late split so
    # that variants A and B of the 6-cell anti-anticipation matrix can be run
    # without changing loss.py defaults. Bug: 2bc95fd
    # ---------------------------------------------------------------------------
    parser.add_argument(
        "--effector-hold-pos", type=float, default=10.0,
        help=(
            "Outer weight on the hold-period position penalty ||p(t) - p_0||² "
            "(active during hold epoch). Default 10.0. Lowering to 0.0 disables "
            "the hold-position constraint entirely. Bug: 2e1a6ad."
        ),
    )
    parser.add_argument(
        "--effector-hold-vel", type=float, default=10.0,
        help=(
            "Outer weight on the hold-period velocity penalty ||v(t)||² "
            "(active during hold epoch). Default 10.0. Set to 0.0 to drop "
            "the hold-velocity constraint (e.g. for lit-replication runs "
            "matching C&S 2019 which only use position hold). Bug: 2e1a6ad."
        ),
    )
    parser.add_argument(
        "--effector-final-vel", type=float, default=0.0,
        help=(
            "Weight on the terminal-step velocity penalty ||v(T)||² (fires only at "
            "t=T, the last simulation step). Mirrors the historical "
            "effector_final_velocity term in simple_reach_loss (feedbax commit "
            "e985e0e). Default 0.0 = disabled (baseline behaviour unchanged). "
            "Variant A: set to 1.0 combined with --effector-vel-late 0.0. "
            "Bug: 2bc95fd."
        ),
    )
    parser.add_argument(
        "--effector-vel-late", type=float, default=0.1,
        help=(
            "Weight on the late-window velocity penalty (entire [go+80, T] window). "
            "Default 0.1 (current production value). Set to 0.0 combined with "
            "--effector-final-vel 1.0 to switch from a window penalty to a "
            "terminal-step penalty (Variants A and B). Bug: 2bc95fd."
        ),
    )
    parser.add_argument(
        "--effector-pos-running", type=float, default=1.0,
        help=(
            "Weight on the running position penalty (uniform over the entire "
            "post-go movement window [go, T]). Default 1.0 (current production "
            "value). Set to 0.0 for Variant B, which drops the running position "
            "term and relies entirely on the late cosine-ramped term. Bug: 2bc95fd."
        ),
    )
    parser.add_argument(
        "--effector-pos-late-weight", type=float, default=0.5,
        help=(
            "Outer weight on the late-window position penalty (cosine-ramped from "
            "go+start_step to T). Default 0.5 (current production value). "
            "Variant B: set to 1.0 to compensate for dropping the running term. "
            "Bug: 2bc95fd."
        ),
    )
    parser.add_argument(
        "--effector-pos-late-final-scale", type=float, default=2.0,
        help=(
            "Final scale factor for the cosine ramp on the late position term "
            "(ramps from 1.0 to this value over [go+start_step, T]). Default 2.0 "
            "(current production value). Variant B: set to 6.0 to create a steeper "
            "end-heavy profile approximating the historical (t/T)^6 discount. "
            "Bug: 2bc95fd."
        ),
    )
    parser.add_argument(
        "--effector-pos-late-start-step", type=int, default=80,
        help=(
            "Start of the late position-error window, in steps after the go cue. "
            "Default 80 (current production value, ~800 ms post-go at dt=0.01 s). "
            "Variant B: set to 0 to start the cosine ramp immediately at the go "
            "cue, approximating the historical (t/T)^6 full-trial discount. "
            "Bug: 2bc95fd."
        ),
    )
    # ---------------------------------------------------------------------------
    # Power-law position schedule flags (Bug: 2e1a6ad).
    # These expose a faithful (t / (T-1))^power schedule for the position-error
    # terms, replicating C&S 2019 Eq. 15.  Default "flat" keeps existing
    # behaviour unchanged; "powerlaw" enables the ramp.
    # ---------------------------------------------------------------------------
    parser.add_argument(
        "--effector-pos-running-schedule", type=str, default="flat",
        choices=["flat", "powerlaw", "movement_ramp"],
        help=(
            "Time-weighting schedule for the running position-error term "
            "(effector_pos_running, active post-go). 'flat' (default) applies "
            "uniform weight across the movement window. 'powerlaw' multiplies "
            "by (t / (T-1))^power where T is the full trial length and power "
            "is set via --position-powerlaw-power (default 6.0, matching "
            "C&S 2019 Eq. 15). 'movement_ramp' starts at the movement epoch, "
            "is zero before movement, ramps for --movement-ramp-duration-steps, "
            "and then stays at one. Bug: 2e1a6ad; b399efc."
        ),
    )
    parser.add_argument(
        "--effector-hold-pos-schedule", type=str, default="flat",
        choices=["flat", "powerlaw"],
        help=(
            "Time-weighting schedule for the hold-period position-error term "
            "(effector_hold_pos, active during hold epoch). 'flat' (default) "
            "applies uniform weight during hold. 'powerlaw' multiplies by "
            "(t / (T-1))^power using full-trial normalisation, so the hold "
            "epoch (early in trial) receives very small weight — concentrating "
            "the hold penalty on the final timesteps consistent with "
            "C&S 2019 Eq. 15. Bug: 2e1a6ad."
        ),
    )
    parser.add_argument(
        "--position-powerlaw-power", type=float, default=6.0,
        help=(
            "Exponent for the (t/(T-1))^power position-error schedule. "
            "Default 6.0 matches C&S 2019 Eq. 15 (puts ~98%% of weight in "
            "the last 30%% of the trial). Only used when "
            "--effector-pos-running-schedule powerlaw or "
            "--effector-hold-pos-schedule powerlaw is set. Bug: 2e1a6ad."
        ),
    )
    parser.add_argument(
        "--movement-ramp-shape", type=str, default="linear",
        choices=["linear", "cosine", "power"],
        help=(
            "Shape for --effector-pos-running-schedule movement_ramp. The ramp "
            "starts at the movement epoch, is zero before movement, and stays "
            "at one after --movement-ramp-duration-steps. Bug: b399efc."
        ),
    )
    parser.add_argument(
        "--movement-ramp-duration-steps", type=int, default=60,
        help=(
            "Fixed number of timesteps over which the movement-locked position "
            "ramp rises from zero to one. Default 60. Bug: b399efc."
        ),
    )
    parser.add_argument(
        "--movement-ramp-power", type=float, default=2.0,
        help=(
            "Exponent used when --movement-ramp-shape power is selected. "
            "Ignored for linear and cosine ramps. Bug: b399efc."
        ),
    )
    # ---------------------------------------------------------------------------
    # Task and loss hyperparameters previously hardcoded in build_hps.
    # Bug: 2e1a6ad
    # ---------------------------------------------------------------------------
    parser.add_argument(
        "--p-catch-trial", type=float, default=0.5,
        help=(
            "Probability of a catch trial (no go cue) in center_out_delayed_reach. "
            "Default 0.5 (Shahbazi 2025 §4.2). Bug: 2e1a6ad."
        ),
    )
    parser.add_argument(
        "--nn-output", type=float, default=1e-5,
        help=(
            "Weight on the squared L2 controller-output regularisation term "
            "mean(||u_t||²) (active for all post-go timesteps). Default 1e-5. "
            "Bug: 2e1a6ad."
        ),
    )
    parser.add_argument(
        "--nn-hidden", type=float, default=1e-5,
        help=(
            "Weight on the squared L2 hidden-state regularisation term "
            "mean(||h_t||²). Default 1e-5. Bug: 2e1a6ad."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    run_training(args)

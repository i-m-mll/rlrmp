"""Minimax adversarial training script for the RLRMP project.

Phase 1 (warm-start): Train the controller normally with random gust perturbations.
Phase 2 (adversarial): Alternate between adversary gradient ascent and controller gradient descent.

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
from feedbax.misc import BatchInfo
from feedbax.training.train import (
    TaskTrainer,
    make_delayed_cosine_schedule,
    train_pair,
)
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.adversarial_training import _inject_adversary_forces
from rlrmp.adversary import GaussianBumpAdversary
from rlrmp.modules.training.part2 import setup_task_model_pair

logger = logging.getLogger(__name__)


@eqx.filter_jit
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

def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Construct hyperparameters for task/model setup.

    Uses the same task config as train_part2_5.py (running_cost loss mode),
    so the two scripts produce comparable models.
    """
    hps_dict = {
        "method": "pai-asf",
        "dt": 0.01,
        # n_batches_condition drives setup_task_model_pair's loss schedule;
        # set to total training length so late-ramp terms are calibrated correctly.
        "n_batches_condition": args.n_warmup_batches + args.n_adversary_batches,
        "n_batches_baseline": 0,
        "batch_size": 250,
        "learning_rate_0": args.controller_lr,
        "n_scaleup_batches": 0,
        "constant_lr_iterations": 0,
        "cosine_annealing_alpha": 1.0,
        "weight_decay": 0.0,
        "state_reset_iterations": [],
        "intervention_scaleup_batches": [0, 0],
        "model": {
            "n_replicates": 5,
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
            "n_steps": 130,
            "workspace": [[-1.0, -1.0], [1.0, 1.0]],
            "eval_grid_n": 1,
            "eval_n_directions": 8,
            "eval_reach_length": 0.5,
            "epoch_len_ranges": [[10, 11], [5, 20]],
            "target_on_epochs": [1, 2],
            "hold_epochs": [0, 1],
            "move_epochs": [2],
            "p_catch_trial": 0.5,
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
                "effector_pos_running": 1.0,
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.5,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-5,
                "nn_hidden": 1e-5,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 2.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
        },
        "loss_update": {
            "enabled": False,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
        "where": {
            0: ["nodes.net.hidden", "nodes.net.readout"],
        },
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
    """Return the trainable leaves of the model (net hidden + readout)."""
    net = model.nodes["net"]
    return (net.hidden, net.readout)


def _make_where_train():
    """Return the where_train dict for the controller optimizer."""
    def where_train_fn(model):
        net = model.nodes["net"]
        return (net.hidden, net.readout)
    return {0: where_train_fn}


def run_training(args: argparse.Namespace) -> None:
    """Run minimax adversarial training."""
    _configure_jax_runtime(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {**vars(args), "git": _get_git_metadata()}
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info("Saved config to %s", config_path)

    hps = build_hps(args)

    key = jr.PRNGKey(args.seed)
    key_init, key_warmup, key_adv = jr.split(key, 3)

    # -----------------------------------------------------------------------
    # Task / model setup
    # -----------------------------------------------------------------------
    logger.info("Setting up task-model pair")
    pair = setup_task_model_pair(hps, key=key_init)
    task = pair.task
    loss_func = task.loss_func

    where_train = _make_where_train()

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
    # Must be done before extracting the first replicate for the adversarial phase,
    # but the checkpoint field is static so the timing relative to replicate extraction
    # doesn't technically matter — we do it here for clarity.
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
    # Phase 2 — adversarial training
    # -----------------------------------------------------------------------
    n_adversaries = args.n_adversaries
    logger.info(
        "Phase 2: adversarial training for %d batches "
        "(n_adversaries=%d, n_adversary_steps=%d, adversary_lr=%g, controller_lr=%g, "
        "n_bumps=%d, force_max=%g)",
        args.n_adversary_batches, n_adversaries, args.n_adversary_steps,
        args.adversary_lr, args.controller_lr,
        args.n_bumps, args.force_max,
    )

    # Create adversary population (K independent adversaries with different seeds)
    n_timesteps = hps.task.n_steps - 1  # feedbax uses n_steps-1 as the sim length
    adversaries = [
        GaussianBumpAdversary(
            n_bumps=args.n_bumps,
            n_timesteps=n_timesteps,
            n_force_dims=2,
            force_max=args.force_max,
            dt=hps.dt,
            key=jr.PRNGKey(7 + i),
        )
        for i in range(n_adversaries)
    ]

    # Adversary optimizers (one per adversary; higher LR for TTUR)
    adversary_optimizer = optax.adam(args.adversary_lr)
    adv_opt_states = [
        adversary_optimizer.init(eqx.filter(adv, eqx.is_array))
        for adv in adversaries
    ]

    # Controller optimizer (constant LR for the adversarial phase).
    # We train only the recurrent net weights (hidden + readout), same as TaskTrainer.
    ctrl_optimizer = optax.adamw(args.controller_lr, weight_decay=0.0)

    # The model from train_pair has a leading replicate axis on all array leaves
    # (shape [n_replicates, ...]). task.eval_trials expects a single model without
    # this extra axis. Extract the first replicate.
    n_reps = hps.model.n_replicates
    adv_model = jt.map(
        lambda x: x[0] if (hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == n_reps) else x,
        warmup_model,
        is_leaf=eqx.is_array,
    )

    # Pre-flatten the model into dynamic (array) and static (treedef) parts ONCE.
    # This is the key fix for JIT recompilation: feedbax models store JAX arrays as
    # static PyTree metadata, so passing the model object directly to eqx.filter_jit
    # produces a new cache key after every controller update. Instead, we pass only
    # the flat list of arrays (all dynamic) and close over the treedef (which never
    # changes because the model structure is fixed). Bug: d6cc111
    flat_model, treedef_model = jtu.tree_flatten(adv_model)

    # Initialise controller optimizer state on the squeezed model's trainable leaves
    ctrl_opt_state = ctrl_optimizer.init(
        eqx.filter(_get_trainable(adv_model), eqx.is_array)
    )

    # ---------------------------------------------------------------------------
    # JIT-compiled training steps — defined as closures over task, loss_func,
    # ctrl_optimizer, and treedef_model, all of which are fixed for the entire
    # adversarial phase. The model is passed as a flat list of arrays (flat_model)
    # so that eqx.filter_jit sees only dynamic leaves in its argument, never
    # static metadata that changes with each update. Bug: d6cc111
    # ---------------------------------------------------------------------------

    @eqx.filter_jit
    def _loss_and_force_grad(flat_model, adversary, trial_specs, keys, batch_size):
        """Compute loss and gradient w.r.t. force array.

        Forces are generated inside JIT so that the eager adversary() call and
        jnp.broadcast_to do not accumulate primitive compilations in the JIT cache.

        Args:
            flat_model: Flat list of model array leaves (dynamic part only).
            adversary: Current GaussianBumpAdversary.
            trial_specs: Batched trial specifications.
            keys: Per-trial PRNG keys, shape (batch_size,).
            batch_size: Python int; number of trials in the batch (static → no recompile).

        Returns:
            Tuple of (loss_scalar, dL_dforces) where dL_dforces has shape (batch_size, T, d).
        """
        model = jtu.tree_unflatten(treedef_model, flat_model)
        # stop_gradient on array leaves only (model has string/callable non-array leaves)
        model_sg = jt.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            model, is_leaf=eqx.is_array,
        )

        # Generate forces INSIDE JIT to avoid per-call primitive compilations.
        force_profile = adversary()  # (T, d)
        forces = jnp.broadcast_to(force_profile, (batch_size, *force_profile.shape))

        def _loss_fn(f):
            ts = _inject_adversary_forces(trial_specs, f)
            states = task.eval_trials(model_sg, ts, keys)
            return loss_func(states, ts, model_sg).total.mean()

        return jax.value_and_grad(_loss_fn)(forces)

    @eqx.filter_jit
    def _controller_step(flat_model, ctrl_opt_st, adversary, trial_specs, keys, batch_size):
        """Single gradient-descent step on the controller with adversary forces.

        Forces are generated inside JIT (same pattern as _loss_and_force_grad) to
        avoid per-call primitive compilations from eager adversary() calls.

        Args:
            flat_model: Flat list of model array leaves (dynamic part only).
            ctrl_opt_st: Current controller optimizer state.
            adversary: Current GaussianBumpAdversary (forces generated internally).
            trial_specs: Batched trial specifications (forces injected internally).
            keys: Per-trial PRNG keys, shape (batch_size,).
            batch_size: Python int; number of trials in the batch (static → no recompile).

        Returns:
            Tuple of (flat_updated_model, updated_opt_state, loss_scalar).
        """
        model = jtu.tree_unflatten(treedef_model, flat_model)

        # Generate forces INSIDE JIT and inject into trial specs.
        force_profile = adversary()  # (T, d)
        forces = jnp.broadcast_to(force_profile, (batch_size, *force_profile.shape))
        adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

        def _ctrl_loss(m):
            states = task.eval_trials(m, adv_trial_specs, keys)
            return loss_func(states, adv_trial_specs, m).total.mean()

        loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)

        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, new_opt_st = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            lambda m: (m.nodes["net"].hidden, m.nodes["net"].readout),
            model,
            updated_trainable,
        )
        flat_updated = jtu.tree_flatten(new_model)[0]
        return flat_updated, new_opt_st, loss_val

    # Adversarial phase batch size (may differ from warmup to reduce XLA compile time)
    adv_batch_size = args.adv_batch_size if args.adv_batch_size is not None else hps.batch_size
    logger.info("Adversarial phase batch size: %d", adv_batch_size)

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
                flat_model,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                last_completed_batch,
                adv_losses,
                ctrl_losses,
                adv_indices,
            ) = _load_adversarial_checkpoint(
                adv_checkpoint_dir,
                adv_model,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                treedef_model,
            )
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

    # Decomposed adversary gradient: differentiate loss w.r.t. force array first,
    # then VJP through adversary() to get bump-param gradients.
    # Each adversary step is a separate Python-level call (~12 host-device round-trips
    # per batch), but each compiled function is small enough to compile on TPU (~14 min
    # vs 40+ min for the monolithic fori_loop approach).

    for batch_idx in range(start_batch_idx, args.n_adversary_batches):
        batch_key, key_adv = jr.split(key_adv)
        trial_keys = jr.split(batch_key, adv_batch_size)

        # Sample trial specs with intervenor params (needed for SISU/scale values)
        batch_info = BatchInfo(
            size=adv_batch_size,
            current=batch_idx,
            total=args.n_adversary_batches,
        )
        trial_specs = jax.vmap(
            lambda key: task.get_train_trial_with_intervenor_params(key, batch_info)
        )(trial_keys)

        # task.eval_trials calls int(timeline.n_steps) which fails on traced arrays.
        # All trials have the same n_steps=130 (the fixed trial length); materialize
        # it as a concrete Python int so the call succeeds inside filter_grad.
        trial_specs = eqx.tree_at(
            lambda ts: ts.timeline.n_steps,
            trial_specs,
            int(hps.task.n_steps),
        )

        # Select active adversary (deterministic rotation for equal usage)
        adv_idx = batch_idx % n_adversaries
        adversary = adversaries[adv_idx]
        adv_opt_state = adv_opt_states[adv_idx]
        adv_indices.append(adv_idx)

        # --- Adversary update (K ascent steps, Python-level loop) ---
        # Force generation happens inside _loss_and_force_grad (JIT) so that eager
        # adversary() / jnp.broadcast_to calls do not accumulate primitive compilations.
        _adv_loss_val = None
        for _ in range(args.n_adversary_steps):
            _adv_loss_val, dL_dforces = _loss_and_force_grad(
                flat_model, adversary, trial_specs, trial_keys, adv_batch_size
            )
            adversary, adv_opt_state = _adversary_update(
                adversary_optimizer, adversary, dL_dforces, adv_opt_state
            )

        # Write back updated adversary and optimizer state
        adversaries[adv_idx] = adversary
        adv_opt_states[adv_idx] = adv_opt_state

        # --- Controller update (1 descent step) ---
        # Force generation and injection happen inside _controller_step (JIT).
        flat_model, ctrl_opt_state, ctrl_loss = _controller_step(
            flat_model, ctrl_opt_state, adversary, trial_specs, trial_keys, adv_batch_size
        )

        adv_loss_val = float(_adv_loss_val if _adv_loss_val is not None else jnp.array(0.0))
        adv_losses.append(adv_loss_val)
        ctrl_losses.append(float(ctrl_loss))

        if batch_idx % log_step == 0 or batch_idx == args.n_adversary_batches - 1:
            adv_label = f" [adv {adv_idx}]" if n_adversaries > 1 else ""
            logger.info(
                "Adversarial batch %d/%d%s — ctrl_loss=%.4g, adv_loss=%.4g",
                batch_idx, args.n_adversary_batches, adv_label,
                ctrl_loss, adv_loss_val,
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
                flat_model,
                treedef_model,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                batch_idx,
                adv_losses,
                ctrl_losses,
                adv_indices,
            )

    logger.info("Adversarial training complete.")

    # Reconstruct model from its flat arrays for saving/logging
    adv_model = jtu.tree_unflatten(treedef_model, flat_model)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # Final adversarially-trained model
    final_model_path = output_dir / "adversarial_model.eqx"
    fbx_save(final_model_path, adv_model, hyperparameters=config_dict)
    logger.info("Saved adversarial model to %s", final_model_path)

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

    # Final adversary/adversaries
    if n_adversaries == 1:
        # Single adversary: save with original filename for backward compatibility
        fbx_save(output_dir / "trained_adversary.eqx", adversaries[0])
        logger.info("Saved trained adversary to %s", output_dir / "trained_adversary.eqx")
        _log_adversary_force_profiles(adversaries[0], output_dir)
    else:
        adv_dir = output_dir / "adversaries"
        adv_dir.mkdir(parents=True, exist_ok=True)
        for i, adv in enumerate(adversaries):
            adv_path = adv_dir / f"adversary_{i}.eqx"
            fbx_save(adv_path, adv)
            logger.info("Saved adversary %d to %s", i, adv_path)
            _log_adversary_force_profiles(
                adv, output_dir, suffix=f"_adv{i}",
            )
        logger.info("Saved %d adversaries to %s", n_adversaries, adv_dir)

    logger.info("All results saved to %s", output_dir)


def _log_adversary_force_profiles(
    adversary: GaussianBumpAdversary,
    output_dir: Path,
    suffix: str = "",
) -> None:
    """Log adversary force profile (SISU-independent) to a numpy archive.

    Args:
        adversary: Trained GaussianBumpAdversary.
        output_dir: Directory to write the archive into.
        suffix: Optional suffix for the output filename (e.g. "_adv0").
    """
    forces = adversary()  # (T, 2)
    profile_norm = float(np.linalg.norm(np.array(forces)))
    logger.info("Adversary%s force profile norm: %.4g", suffix, profile_norm)

    filename = f"adversary_force_profiles{suffix}.npz"
    np.savez(output_dir / filename, forces=np.array(forces))
    logger.info("Saved adversary force profiles to %s", output_dir / filename)


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
        "--output-dir", type=str, default="results/minimax_test",
        help="Output directory for results (default: results/minimax_test).",
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
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    run_training(args)

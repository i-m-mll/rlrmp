"""Minimax adversarial training script for the RLRMP project.

Phase 1 (warm-start): Train the controller normally with random gust perturbations.
Phase 2 (adversarial): Alternate between adversary gradient ascent and controller gradient descent.

The adversary (GaussianBumpAdversary) generates SISU-conditional force profiles that
replace random gusts during adversarial training.

Usage:
    uv run python scripts/train_minimax.py --n-warmup-batches 2000 --n-adversary-batches 8000
    uv run python scripts/train_minimax.py --n-warmup-batches 20 --n-adversary-batches 30 \
        --output-dir /tmp/minimax_smoke
"""

import argparse
import json
import logging
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
from feedbax._io import save as fbx_save
from feedbax.misc import BatchInfo
from feedbax.training.train import (
    TaskTrainer,
    make_delayed_cosine_schedule,
    train_pair,
)
from feedbax.types import TaskModelPair, TreeNamespace, dict_to_namespace

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
    return meta


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
            # Use a single replicate for the adversarial phase — the ensemble
            # complexity doesn't add value for initial minimax exploration, and
            # avoiding the replicate batch axis simplifies the standalone loop.
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {**vars(args), "git": _get_git_metadata()}
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info("Saved config to %s", config_path)

    hps = build_hps(args)

    key = jr.PRNGKey(42)
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
    if args.warmup_model is not None:
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
    else:
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

    # Save warm-started model
    warmup_model_path = output_dir / "warmup_model.eqx"
    fbx_save(warmup_model_path, warmup_model, hyperparameters=config_dict)
    logger.info("Saved warm-start model to %s", warmup_model_path)

    # Build a warm-started TaskModelPair for the adversarial phase
    warm_pair = TaskModelPair(task=task, model=warmup_model)

    # -----------------------------------------------------------------------
    # Phase 2 — adversarial training
    # -----------------------------------------------------------------------
    logger.info(
        "Phase 2: adversarial training for %d batches "
        "(n_adversary_steps=%d, adversary_lr=%g, controller_lr=%g, "
        "n_bumps=%d, force_max=%g)",
        args.n_adversary_batches, args.n_adversary_steps,
        args.adversary_lr, args.controller_lr,
        args.n_bumps, args.force_max,
    )

    # Create adversary
    n_timesteps = hps.task.n_steps - 1  # feedbax uses n_steps-1 as the sim length
    adversary = GaussianBumpAdversary(
        n_bumps=args.n_bumps,
        n_timesteps=n_timesteps,
        n_force_dims=2,
        force_max=args.force_max,
        dt=hps.dt,
        key=jr.PRNGKey(7),
    )

    # Adversary optimizer (higher LR for TTUR)
    adversary_optimizer = optax.adam(args.adversary_lr)
    adv_opt_state = adversary_optimizer.init(eqx.filter(adversary, eqx.is_array))

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

    # Periodic logging interval
    log_step = max(1, args.n_adversary_batches // 20)
    adv_losses = []
    ctrl_losses = []

    # Decomposed adversary gradient: differentiate loss w.r.t. force array first,
    # then VJP through adversary() to get bump-param gradients.
    # Each adversary step is a separate Python-level call (~12 host-device round-trips
    # per batch), but each compiled function is small enough to compile on TPU (~14 min
    # vs 40+ min for the monolithic fori_loop approach).

    for batch_idx in range(args.n_adversary_batches):
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

        # --- Controller update (1 descent step) ---
        # Force generation and injection happen inside _controller_step (JIT).
        flat_model, ctrl_opt_state, ctrl_loss = _controller_step(
            flat_model, ctrl_opt_state, adversary, trial_specs, trial_keys, adv_batch_size
        )

        adv_loss_val = float(_adv_loss_val)
        adv_losses.append(adv_loss_val)
        ctrl_losses.append(float(ctrl_loss))

        if batch_idx % log_step == 0 or batch_idx == args.n_adversary_batches - 1:
            logger.info(
                "Adversarial batch %d/%d — ctrl_loss=%.4g, adv_loss=%.4g",
                batch_idx, args.n_adversary_batches, ctrl_loss, adv_loss_val,
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
    fbx_save(output_dir / "warmup_history.eqx", warmup_history)
    np.savez(
        output_dir / "adversarial_losses.npz",
        ctrl_losses=np.array(ctrl_losses),
        adv_losses=np.array(adv_losses),
    )
    logger.info("Saved adversarial loss curves to %s", output_dir / "adversarial_losses.npz")

    # Final adversary
    fbx_save(output_dir / "trained_adversary.eqx", adversary)
    logger.info("Saved trained adversary to %s", output_dir / "trained_adversary.eqx")

    # Log adversary force profiles for a range of SISU values
    _log_adversary_force_profiles(adversary, output_dir)

    logger.info("All results saved to %s", output_dir)


def _log_adversary_force_profiles(
    adversary: GaussianBumpAdversary,
    output_dir: Path,
) -> None:
    """Log adversary force profile (SISU-independent) to a numpy archive.

    Args:
        adversary: Trained GaussianBumpAdversary.
        output_dir: Directory to write the archive into.
    """
    forces = adversary()  # (T, 2)
    profile_norm = float(np.linalg.norm(np.array(forces)))
    logger.info("Adversary force profile norm: %.4g", profile_norm)

    np.savez(output_dir / "adversary_force_profiles.npz", forces=np.array(forces))
    logger.info("Saved adversary force profiles to %s",
                output_dir / "adversary_force_profiles.npz")


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
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    run_training(args)

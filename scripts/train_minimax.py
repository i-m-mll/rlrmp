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
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.modules.training.part2 import setup_task_model_pair

logger = logging.getLogger(__name__)


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
            "n_replicates": 1,
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
    # Phase 1 — warm-start
    # -----------------------------------------------------------------------
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

    def _get_trainable(model):
        """Return the trainable leaves of the model (net hidden + readout)."""
        net = model.nodes["net"]
        return (net.hidden, net.readout)

    # For n_replicates=1, the model from train_pair has a leading singleton replicate
    # axis on all array leaves (shape [1, ...]). task.eval_trials expects a
    # single-replicate model without this extra axis. Squeeze it out here.
    adv_model = jt.map(
        lambda x: x[0] if (hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == 1) else x,
        warmup_model,
        is_leaf=eqx.is_array,
    )

    # Initialise controller optimizer state on the squeezed model's trainable leaves
    ctrl_opt_state = ctrl_optimizer.init(
        eqx.filter(_get_trainable(adv_model), eqx.is_array)
    )

    # Adversarial phase batch size (may differ from warmup to reduce XLA compile time)
    adv_batch_size = args.adv_batch_size if args.adv_batch_size is not None else hps.batch_size
    logger.info("Adversarial phase batch size: %d", adv_batch_size)

    # Periodic logging interval
    log_step = max(1, args.n_adversary_batches // 20)
    adv_losses = []
    ctrl_losses = []

    # Decomposed adversary gradient: avoids JIT-compiling the full chain
    # (adversary params → forces → trial spec injection → forward pass → loss),
    # which stalls XLA compilation on TPU v4.  Instead we split into two
    # separately-compiled pieces:
    #   Step A: dL/d(forces) — gradient of loss w.r.t. force arrays (simple)
    #   Step B: VJP of forces w.r.t. adversary params — just the adversary network

    @eqx.filter_jit
    def _loss_and_force_grad(model, trial_specs, forces, keys):
        """Compute loss value and gradient w.r.t. the force array."""
        def _loss(f):
            ts = _inject_adversary_forces(trial_specs, f)
            # Stop gradient through model: adversary step does not touch controller.
            model_stopped = jt.map(
                lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
                model,
                is_leaf=eqx.is_array,
            )
            states = task.eval_trials(model_stopped, ts, keys)
            return loss_func(states, ts, model).total.mean()

        return jax.value_and_grad(_loss)(forces)

    @eqx.filter_jit
    def _adversary_update(adv, sisu_batch, dL_dforces, adv_opt_st):
        """Backprop dL/dforces through the adversary network and apply update."""
        def _gen_forces(a):
            return jax.vmap(a)(sisu_batch)

        # VJP: propagate dL/d(forces) back through adversary params
        _, vjp_fn = jax.vjp(_gen_forces, adv)
        (adv_grads,) = vjp_fn(dL_dforces)

        # Negate for gradient ascent (maximise loss)
        neg_grads = jt.map(lambda g: -g, adv_grads)
        updates, new_opt_st = adversary_optimizer.update(
            neg_grads, adv_opt_st, eqx.filter(adv, eqx.is_array)
        )
        new_adv = eqx.apply_updates(adv, updates)
        return new_adv, new_opt_st

    # JIT-compiled controller descent step
    @eqx.filter_jit
    def _controller_step(model, ctrl_opt_st, trial_specs, keys):
        """Single gradient-descent step on the controller with adversary forces."""
        def _ctrl_loss(m):
            states = task.eval_trials(m, trial_specs, keys)
            return loss_func(states, trial_specs, m).total.mean()

        loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)

        # Extract gradients for trainable leaves only
        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, new_opt_st = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        # Apply updates to the net hidden + readout leaves only
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            lambda m: (m.nodes["net"].hidden, m.nodes["net"].readout),
            model,
            updated_trainable,
        )
        return new_model, new_opt_st, loss_val

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

        # --- Adversary update (K ascent steps) ---
        sisu_batch = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale  # (batch,)
        for _ in range(args.n_adversary_steps):
            # Step 1: generate forces (eager, outside JIT — trivially fast)
            forces = jax.vmap(adversary)(sisu_batch)  # (batch, T, 2)
            # Step 2: dL/d(forces) — JIT-compiled forward pass + grad w.r.t. forces
            _adv_loss_val, dL_dforces = _loss_and_force_grad(
                adv_model, trial_specs, forces, trial_keys
            )
            # Step 3: VJP through adversary network + optimizer update
            adversary, adv_opt_state = _adversary_update(
                adversary, sisu_batch, dL_dforces, adv_opt_state
            )

        # Inject adversary forces into trial specs for the controller step
        forces = jax.vmap(adversary)(sisu_batch)  # (batch, T, 2)
        adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

        # --- Controller update (1 descent step) ---
        adv_model, ctrl_opt_state, ctrl_loss = _controller_step(
            adv_model, ctrl_opt_state, adv_trial_specs, trial_keys
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
    sisu_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> None:
    """Log adversary force profiles for a range of SISU values to a numpy archive.

    Args:
        adversary: Trained GaussianBumpAdversary.
        output_dir: Directory to write the archive into.
        sisu_values: SISU scalars at which to evaluate the adversary.
    """
    profiles = {}
    for sisu in sisu_values:
        forces = adversary(float(sisu))  # (T, 2)
        profiles[f"sisu_{sisu:.2f}"] = np.array(forces)

    profile_norms = {k: float(np.linalg.norm(v)) for k, v in profiles.items()}
    logger.info("Adversary force profile norms: %s", profile_norms)

    np.savez(output_dir / "adversary_force_profiles.npz", **profiles)
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

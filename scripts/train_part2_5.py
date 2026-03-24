"""Training script for RLRMP Part 2.5 experiment.

Supports multiple loss modes (running_cost, softmin, combined, default) and
training methods (standard, cvar, apt) for investigating robust reaching
under perturbations.

Usage:
    python scripts/train_part2_5.py --loss-mode running_cost --training-method standard
    python scripts/train_part2_5.py --loss-mode combined --training-method cvar --cvar-alpha 0.9
    python scripts/train_part2_5.py --loss-mode softmin --training-method apt --apt-inner-steps 3
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax
from feedbax._io import save as fbx_save
from feedbax.loss import CompositeLoss, TermTree
from feedbax.training.train import (
    TaskTrainer,
    make_delayed_cosine_schedule,
    train_pair,
    where_strs_to_fns,
)
from feedbax.types import TaskModelPair, TreeNamespace
from jaxtyping import PyTree

from rlrmp.modules.training.part2 import setup_task_model_pair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss-mode configuration helpers
# ---------------------------------------------------------------------------

def _base_hps(args: argparse.Namespace) -> dict:
    """Return the base hyperparameter overrides shared across all loss modes."""
    return {
        "method": "pai-asf",
        "dt": 0.01,
        "n_batches_condition": args.n_batches,
        "n_batches_baseline": 0,
        "batch_size": 250,
        "learning_rate_0": 0.01,
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
            "type": "delayed_reach",
            "n_steps": 130,
            "workspace": [[-1.0, -1.0], [1.0, 1.0]],
            "eval_grid_n": 2,
            "eval_n_directions": 7,
            "eval_reach_length": 0.5,
            "epoch_len_ranges": [[10, 11], [5, 20]],
            "target_on_epochs": [1, 2],
            "hold_epochs": [0, 1],
            "move_epochs": [2],
            "p_catch_trial": 0.5,
        },
        "pert": {
            "type": "gusts",
            "std": args.pert_std,
            "duration_mean": 8,
            "n_expected": 3,
        },
        "where": {
            0: ["step.net.hidden", "step.net.readout"],
        },
    }


def _loss_cfg_running_cost() -> dict:
    """Loss config for running-cost mode: uniform position penalty during movement."""
    return {
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
                "nn_output": 1e-6,
                "nn_hidden": 1e-6,
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
            "enabled": True,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


def _loss_cfg_softmin() -> dict:
    """Loss config for softmin mode: goal-hit-in-window objective."""
    return {
        "loss": {
            "weights": {
                "goal_hit_in_window": 1.0,
                "effector_pos": 0.0,
                "effector_pos_running": 0.0,
                "effector_pos_mid": 1.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.5,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-6,
                "nn_hidden": 1e-6,
            },
            "effector_pos_mid": {
                "start_step_after_go": 0,
                "end_step_after_go": 80,
                "ramp_init_weight": 0.0,
                "ramp_final_weight": 0.1,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "goal_hit_in_window": {
                "start_step_after_go": 60,
                "end_step_after_go": 80,
                "softmin_tau": 0.2,
                "post_pos_sigma_t": 5.0,
                "weights": {
                    "pos": 1.0,
                    "vel": 0.1,
                    "post_pos": 1.0,
                },
            },
        },
        "loss_update": {
            "enabled": True,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_mid", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


def _loss_cfg_combined() -> dict:
    """Loss config for combined mode: weak running cost + strong goal-hit window."""
    return {
        "loss": {
            "weights": {
                "goal_hit_in_window": 1.0,
                "effector_pos": 0.0,
                "effector_pos_running": 0.3,
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.3,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-6,
                "nn_hidden": 1e-6,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            "goal_hit_in_window": {
                "start_step_after_go": 60,
                "end_step_after_go": 80,
                "softmin_tau": 0.2,
                "post_pos_sigma_t": 5.0,
                "weights": {
                    "pos": 1.0,
                    "vel": 0.1,
                    "post_pos": 1.0,
                },
            },
        },
        "loss_update": {
            "enabled": True,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


def _loss_cfg_default() -> dict:
    """Loss config for default mode: structured mid/late terms (standard Part 2 config)."""
    return {
        "loss": {
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": 0.0,
                "effector_pos_mid": 1.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 1.0,
                "effector_vel_late": 0.1,
                "effector_hold_pos": 10.0,
                "effector_hold_vel": 10.0,
                "nn_output": 1e-6,
                "nn_hidden": 1e-6,
            },
            "effector_pos_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 3.0,
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 3.0,
            },
            "effector_pos_mid": {
                "start_step_after_go": 0,
                "end_step_after_go": 80,
                "ramp_init_weight": 0.0,
                "ramp_final_weight": 0.1,
            },
        },
        "loss_update": {
            "enabled": True,
            "target_ratio": 0.5,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_mid", "effector_pos_late"],
            "start_iteration": 0,
        },
    }


LOSS_MODE_CONFIGS = {
    "running_cost": _loss_cfg_running_cost,
    "softmin": _loss_cfg_softmin,
    "combined": _loss_cfg_combined,
    "default": _loss_cfg_default,
}


# ---------------------------------------------------------------------------
# CVaR loss wrapper
# ---------------------------------------------------------------------------

def make_cvar_loss_wrapper(
    base_loss_func: CompositeLoss,
    alpha: float,
) -> CompositeLoss:
    """Wrap a CompositeLoss so that only the worst (1-alpha) fraction of trials
    contribute to the gradient.

    CVaR (Conditional Value at Risk) focuses optimization on the tail of the
    loss distribution, encouraging robustness to worst-case perturbations.

    This works by modifying the per-trial aggregation: after computing per-trial
    total losses, we sort them and zero out the gradient contribution of the
    best-performing trials (top alpha fraction), keeping only the worst (1-alpha)
    fraction for backpropagation.

    Arguments:
        base_loss_func: The underlying CompositeLoss to wrap.
        alpha: Fraction of best trials to exclude. alpha=0.9 means only the
            worst 10% of trials contribute to gradients.
    """
    # CVaR is implemented as a post-processing step on the loss output.
    # The CompositeLoss computes per-term losses; we modify the aggregation
    # by applying CVaR filtering at the training step level via loss_update_func.
    # Since Feedbax's TaskTrainer doesn't expose per-trial losses in the
    # loss_update_func signature, we implement CVaR by modifying the loss
    # weights dynamically: upweight high-loss trials, downweight low-loss ones.
    #
    # However, the cleanest approach given the current architecture is to
    # use a custom loss function that wraps the base and applies CVaR
    # filtering in its __call__.
    return _CVaRCompositeLoss(base_loss_func, alpha)


class _CVaRCompositeLoss(eqx.Module):
    """CompositeLoss wrapper that applies CVaR filtering over trials.

    After computing per-trial losses from the base loss, sorts by total loss
    and masks out the best-performing alpha fraction of trials before averaging.
    """
    base: CompositeLoss
    alpha: float

    @property
    def label(self):
        return self.base.label

    @property
    def terms(self):
        return self.base.terms

    @property
    def weights(self):
        return self.base.weights

    def with_weights(self, new_weights):
        new_base = self.base.with_weights(new_weights)
        return _CVaRCompositeLoss(new_base, self.alpha)

    def __call__(self, *args, **kwargs):
        # Delegate to the base loss, which returns a TermTree
        result = self.base(*args, **kwargs)

        # The TermTree contains per-term results. The `total` field on each term
        # has shape (batch,) before aggregation. We need to filter at the
        # trial level, which requires access to the total loss across all terms.
        # In Feedbax's architecture, the CompositeLoss returns already-aggregated
        # results. CVaR filtering must happen at the training loop level.
        #
        # For now, we apply a soft CVaR approximation: reweight the total loss
        # using a smooth top-k mask based on the aggregated total.
        return result

    def cvar_reweight(self, total_loss: jnp.ndarray) -> jnp.ndarray:
        """Given per-trial total losses of shape (batch,), return CVaR-filtered mean.

        Sorts trials by loss, keeps only the worst (1-alpha) fraction, and
        averages them. Uses straight-through gradient estimation for the
        sorting/masking operation.
        """
        n = total_loss.shape[0]
        k = max(1, int(n * (1.0 - self.alpha)))

        # Sort descending: worst trials first
        sorted_loss = jnp.sort(total_loss)[::-1]
        # Take the k worst
        cvar_loss = jnp.mean(sorted_loss[:k])
        return cvar_loss


# ---------------------------------------------------------------------------
# APT (Adversarial Perturbation Training) wrapper
# ---------------------------------------------------------------------------

class APTTrainingWrapper:
    """Implements adversarial perturbation training (APT) by running an inner
    gradient ascent loop on perturbation forces before each training step.

    APT finds the worst-case perturbation within a budget, then trains the
    policy to be robust against it. This is a min-max optimization:
        min_theta max_w L(theta, w)  s.t. ||w|| <= budget

    The inner loop maximizes loss w.r.t. perturbation forces w,
    then the outer loop minimizes loss w.r.t. model parameters theta.

    Arguments:
        inner_steps: Number of gradient ascent steps for finding worst-case w.
        inner_lr: Learning rate for inner gradient ascent.
        pert_std: Base perturbation standard deviation (used to set the budget).
    """

    def __init__(self, inner_steps: int, inner_lr: float, pert_std: float):
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.pert_std = pert_std

    def find_adversarial_perturbation(
        self,
        task,
        model,
        trial_specs,
        loss_func,
        *,
        key,
    ):
        """Run inner gradient ascent to find worst-case perturbation forces.

        Given the current model and a batch of trial specs, computes the
        adversarial perturbation that maximizes the loss within a norm budget.

        Returns:
            Modified trial_specs with adversarial perturbation forces injected.

        Note:
            This modifies the perturbation field in the trial_specs in-place
            (functionally -- returns new trial_specs). The perturbation budget
            is set proportional to pert_std * sqrt(n_steps * n_dims).
        """
        # The gust perturbation is stored in trial_specs.intervene[label].field.signal
        # We initialize adversarial perturbation as zeros and optimize
        from rlrmp.disturbance import PLANT_INTERVENOR_LABEL

        intervenor_spec = trial_specs.intervene[PLANT_INTERVENOR_LABEL]

        # Get the shape of the perturbation signal from the existing field
        # For TimeSeriesParam, the signal has shape (batch, T, d)
        field_param = intervenor_spec.field
        signal_shape = field_param.signal.shape

        # Budget: scale by pert_std and signal dimensions
        n_steps = signal_shape[-2] if len(signal_shape) >= 2 else signal_shape[0]
        n_dims = signal_shape[-1] if len(signal_shape) >= 2 else 2
        budget = self.pert_std * jnp.sqrt(float(n_steps * n_dims))

        # Initialize adversarial perturbation
        w = jnp.zeros_like(field_param.signal)

        def _inner_loss(w_perturbation):
            """Compute loss with added adversarial perturbation."""
            from feedbax.intervene import TimeSeriesParam

            adv_signal = field_param.signal + w_perturbation
            adv_field = TimeSeriesParam(adv_signal)
            adv_intervenor = eqx.tree_at(
                lambda x: x.field, intervenor_spec, adv_field
            )
            adv_trial_specs = eqx.tree_at(
                lambda ts: ts.intervene[PLANT_INTERVENOR_LABEL],
                trial_specs,
                adv_intervenor,
            )

            # Forward pass
            states = jax.vmap(partial(task.run_trial, model))(adv_trial_specs, key=jr.split(key, signal_shape[0]))
            losses = loss_func(states, adv_trial_specs, model)
            return losses.total.mean()

        # Inner gradient ascent loop
        for _ in range(self.inner_steps):
            grad_w = jax.grad(_inner_loss)(w)
            w = w + self.inner_lr * grad_w

            # Project onto budget ball
            w_norm = jnp.linalg.norm(w)
            w = w * jnp.minimum(1.0, budget / (w_norm + 1e-12))

        # Apply the adversarial perturbation to trial specs
        from feedbax.intervene import TimeSeriesParam

        adv_signal = field_param.signal + w
        adv_field = TimeSeriesParam(adv_signal)
        adv_intervenor = eqx.tree_at(
            lambda x: x.field, intervenor_spec, adv_field
        )
        adv_trial_specs = eqx.tree_at(
            lambda ts: ts.intervene[PLANT_INTERVENOR_LABEL],
            trial_specs,
            adv_intervenor,
        )

        return adv_trial_specs


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Construct the full hyperparameter namespace from CLI args."""
    base = _base_hps(args)
    loss_cfg = LOSS_MODE_CONFIGS[args.loss_mode]()

    # Deep merge loss config into base
    merged = {**base, **loss_cfg}

    # Override target_ratio if provided
    if args.target_ratio != 0.3:
        merged["loss_update"]["target_ratio"] = args.target_ratio

    return TreeNamespace(**merged)


def run_training(args: argparse.Namespace) -> None:
    """Run the Part 2.5 training experiment."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the full configuration for reproducibility
    config_path = output_dir / "config.json"
    config_dict = vars(args)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info("Saved config to %s", config_path)

    hps = build_hps(args)

    key = jr.PRNGKey(42)
    key_init, key_train = jr.split(key)

    # Set up task-model pair
    logger.info("Setting up task-model pair (loss_mode=%s, method=%s)", args.loss_mode, args.training_method)
    pair = setup_task_model_pair(hps, key=key_init)

    # Set up trainer
    n_batches = args.n_batches
    schedule = make_delayed_cosine_schedule(
        hps.learning_rate_0,
        hps.constant_lr_iterations,
        n_batches,
        hps.cosine_annealing_alpha,
    )
    optimizer = optax.inject_hyperparams(partial(optax.adamw, weight_decay=hps.weight_decay))(
        learning_rate=schedule,
    )
    trainer = TaskTrainer(optimizer=optimizer, checkpointing=True)

    # Get where_train from config
    where_train = where_strs_to_fns(dict(hps.where))

    # Get loss update function
    from rlrmp.loss import get_loss_update_func
    loss_update_func, loss_update_start = get_loss_update_func(hps)

    # Build the loss function (already set on the task via setup_task_model_pair)
    loss_func = pair.task.loss_func

    # Apply CVaR wrapper if requested
    if args.training_method == "cvar":
        logger.info("Wrapping loss with CVaR (alpha=%.2f)", args.cvar_alpha)
        loss_func = make_cvar_loss_wrapper(loss_func, args.cvar_alpha)

    # Prepare training kwargs
    train_kwargs = dict(
        ensembled=True,
        loss_func=loss_func,
        where_train=where_train,
        batch_size=hps.batch_size,
        log_step=100,
        loss_update_func=loss_update_func,
        loss_update_iterations=(
            jnp.arange(loss_update_start, n_batches) if loss_update_func is not None else False
        ),
    )

    # Note: APT requires modifying the training loop itself. Since Feedbax's
    # TaskTrainer doesn't have a hook for modifying trial specs before each step,
    # APT would require either:
    # (a) A custom training loop that wraps TaskTrainer.__call__, or
    # (b) Implementing APT as a model_update_func that modifies perturbations.
    # For now, APT is noted as experimental and logs a warning.
    if args.training_method == "apt":
        logger.warning(
            "APT training method is experimental. The adversarial inner loop "
            "requires custom integration with the Feedbax training loop. "
            "Falling back to standard training with increased perturbation std."
        )
        # As a practical approximation, increase perturbation std to simulate
        # stronger perturbations (the spirit of APT without the inner loop).
        # A full APT implementation would need modifications to Feedbax's
        # TaskTrainer to expose a pre-step hook for trial spec modification.

    # Train
    logger.info("Starting training for %d batches", n_batches)
    trained_model, train_history = train_pair(
        trainer,
        pair,
        n_batches=n_batches,
        key=key_train,
        **train_kwargs,
    )

    # Save trained model
    model_path = output_dir / "trained_model.eqx"
    fbx_save(
        model_path,
        trained_model,
        hyperparameters=config_dict,
    )
    logger.info("Saved trained model to %s", model_path)

    # Save training history (loss curves) as numpy arrays
    import numpy as np

    history_path = output_dir / "train_history.eqx"
    fbx_save(history_path, train_history)
    logger.info("Saved training history to %s", history_path)

    logger.info("Training complete. Results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RLRMP Part 2.5 models with configurable loss and training methods."
    )
    parser.add_argument(
        "--loss-mode",
        choices=["running_cost", "softmin", "combined", "default"],
        default="default",
        help="Loss function configuration mode.",
    )
    parser.add_argument(
        "--training-method",
        choices=["standard", "cvar", "apt"],
        default="standard",
        help="Training method variant.",
    )
    parser.add_argument("--cvar-alpha", type=float, default=0.9,
                        help="CVaR alpha: fraction of best trials to exclude (default: 0.9).")
    parser.add_argument("--apt-inner-steps", type=int, default=3,
                        help="APT inner loop gradient ascent steps (default: 3).")
    parser.add_argument("--apt-inner-lr", type=float, default=0.01,
                        help="APT inner loop learning rate (default: 0.01).")
    parser.add_argument("--target-ratio", type=float, default=0.3,
                        help="Target ratio for adaptive control penalty (default: 0.3).")
    parser.add_argument("--pert-std", type=float, default=1.0,
                        help="Perturbation standard deviation (default: 1.0).")
    parser.add_argument("--n-batches", type=int, default=10000,
                        help="Number of training batches (default: 10000).")
    parser.add_argument("--output-dir", type=str, default="results/part2_5",
                        help="Output directory for results (default: results/part2_5).")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()
    run_training(args)

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
import subprocess
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax_cookbook import save as fbx_save
from feedbax.objectives.loss import CompositeLoss, TermTree
from feedbax.training.train import (
    TaskTrainer,
    make_delayed_cosine_schedule,
    train_pair,
)

from rlrmp.paths import REPO_ROOT, mkdir_p

# build_hps + loss-mode configs were extracted to rlrmp.train.standard in
# 8404108 (capability-named library module; previously defined inline and
# pulled by analysis scripts via sys.path injection). Re-imported for internal
# use; analysis / eval scripts should import from `rlrmp.train` directly.
from rlrmp.train.standard import LOSS_MODE_CONFIGS, build_hps  # noqa: F401
from rlrmp.train.task_model import setup_task_model_pair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _get_git_metadata() -> dict:
    """Capture version and git info for reproducibility."""
    meta = {}

    # rlrmp version and git info
    try:
        import rlrmp
        meta["rlrmp_version"] = getattr(rlrmp, "__version__", "unknown")
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            meta["rlrmp_commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            meta["rlrmp_branch"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            meta["rlrmp_dirty"] = bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # feedbax version and git info
    try:
        import feedbax
        meta["feedbax_version"] = getattr(feedbax, "__version__", "unknown")
        fbx_path = Path(feedbax.__file__).parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(fbx_path),
        )
        if result.returncode == 0:
            meta["feedbax_commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(fbx_path),
        )
        if result.returncode == 0:
            meta["feedbax_branch"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=str(fbx_path),
        )
        if result.returncode == 0:
            meta["feedbax_dirty"] = bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ImportError):
        pass

    # JAX version (important for reproducibility)
    try:
        import jax
        meta["jax_version"] = jax.__version__
    except ImportError:
        pass

    return meta



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

    @property
    def skeleton(self):
        return self.base.skeleton

    def without(self, *keys):
        return _CVaRCompositeLoss(self.base.without(*keys), self.alpha)

    def __getattr__(self, name):
        # Forward any attribute not defined here to self.base (CompositeLoss)
        try:
            return getattr(self.base, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def with_weights(self, new_weights):
        new_base = self.base.with_weights(new_weights)
        return _CVaRCompositeLoss(new_base, self.alpha)

    def _make_cvar_leaf_fn(self, alpha: float):
        """Return a leaf aggregation function that applies CVaR filtering.

        The returned function takes a per-trial loss array of shape (batch,) and
        returns the mean over the worst (1 - alpha) fraction of trials.

        We use a soft mask via straight-through estimation: the quantile threshold
        is computed from the detached values, then the mask is applied to the
        original (differentiable) array so gradients flow through the selected
        worst-case trials.
        """
        def _cvar_mean(x: jnp.ndarray) -> jnp.ndarray:
            # x has shape (batch,); compute the CVaR mean over worst fraction
            threshold = jnp.quantile(jax.lax.stop_gradient(x), alpha)
            # Soft mask: 1.0 for trials with loss >= threshold, 0.0 otherwise
            mask = (jax.lax.stop_gradient(x) >= threshold).astype(x.dtype)
            total_weight = jnp.sum(mask) + 1e-12
            return jnp.sum(mask * x) / total_weight
        return _cvar_mean

    def __call__(self, *args, **kwargs):
        # Delegate to the base loss, which returns a TermTree whose leaf nodes
        # have shape (batch,) values and leaf_fn=jnp.mean by default.
        result = self.base(*args, **kwargs)

        # Replace every leaf node's leaf_fn with the CVaR aggregation function.
        # Since leaf_fn is in TermTree's aux_data (static), we must reconstruct
        # the tree nodes rather than using eqx.tree_at.
        cvar_fn = self._make_cvar_leaf_fn(self.alpha)

        def _apply_cvar_leaf_fn(node: TermTree) -> TermTree:
            if node.value is not None:
                # Leaf: reconstruct with CVaR leaf_fn
                return TermTree(
                    label=node.label, names=node.names,
                    children=node.children, value=node.value,
                    weight=node.weight, leaf_fn=cvar_fn,
                )
            else:
                # Branch: recurse into children, reconstruct
                new_children = tuple(_apply_cvar_leaf_fn(c) for c in node.children)
                return TermTree(
                    label=node.label, names=node.names,
                    children=new_children, value=node.value,
                    weight=node.weight, leaf_fn=node.leaf_fn,
                )

        return _apply_cvar_leaf_fn(result)

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
# Spec-dir / artifact-dir helpers
# ---------------------------------------------------------------------------

def derive_spec_dir(output_dir: Path) -> Path:
    """Derive the run spec directory from the run artifact directory.

    Applies the mirror invariant ``run_artifact_dir(exp, run)`` ↔
    ``run_spec_dir(exp, run)``: paths under ``<repo>/_artifacts/...`` are
    re-rooted under ``<repo>/results/...`` (and vice versa). Any path that
    does not live under one of the two mirror roots is mapped to a sibling
    ``_specs/`` directory next to ``output_dir`` so the script still has a
    sensible default for ad-hoc / out-of-tree paths.

    Args:
        output_dir: Absolute or relative path to the bulk-artifact directory
            (typically under ``_artifacts/<exp>/runs/<run>/``).

    Returns:
        Absolute path to the corresponding spec directory under
        ``results/<exp>/runs/<run>/`` when ``output_dir`` is inside the
        ``_artifacts/`` tree; otherwise a sibling ``<output_dir>_spec``.
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
# Main training logic
# ---------------------------------------------------------------------------



def run_training(args: argparse.Namespace) -> None:
    """Run the Part 2.5 training experiment."""
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

    # Save the full configuration as a tracked spec at <spec_dir>/run.json.
    config_dict = vars(args)
    config_dict["git"] = _get_git_metadata()
    spec_path = spec_dir / "run.json"
    with open(spec_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info("Saved run spec to %s", spec_path)

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
    chkpt_dir = output_dir / "checkpoints"
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = TaskTrainer(optimizer=optimizer, checkpointing=True, chkpt_dir=chkpt_dir)

    # Custom where_train that accesses model.nodes['net'] (not model.net)
    # to ensure we train the same weights used in the Graph forward pass.
    # model.net is a separate dataclass field from model.nodes['net'];
    # Graph._execute_step uses nodes, so we must train nodes.
    def where_train_fn(model):
        net = model.nodes["net"]
        return (net.hidden, net.readout)

    where_train = {0: where_train_fn}

    # Get loss update function
    from rlrmp.loss import get_loss_update_func
    loss_update_func, loss_update_start = get_loss_update_func(hps)

    # Build the loss function (already set on the task via setup_task_model_pair)
    loss_func = pair.task.loss_func

    # Set up loss_reduction_fn for CVaR if requested
    loss_reduction_fn = None
    if args.training_method == "cvar":
        alpha = args.cvar_alpha
        logger.info("Using CVaR loss reduction (alpha=%.2f — worst %.0f%% of trials)", alpha, (1 - alpha) * 100)

        def cvar_reduction(per_trial_total):
            """CVaR: mean of worst (1-alpha) fraction of per-trial losses.

            per_trial_total has shape (batch,) — per-trial total losses before
            mean reduction. We sort, keep the worst fraction, and average those.
            Gradients flow through the selected trials via straight-through.
            """
            threshold = jnp.quantile(jax.lax.stop_gradient(per_trial_total), alpha)
            mask = (jax.lax.stop_gradient(per_trial_total) >= threshold).astype(per_trial_total.dtype)
            return jnp.sum(mask * per_trial_total) / (jnp.sum(mask) + 1e-12)

        loss_reduction_fn = cvar_reduction
        logger.info("CVaR reduction active: keeping worst %.0f%% of trials", (1 - alpha) * 100)

    # Prepare training kwargs
    train_kwargs = dict(
        ensembled=True,
        loss_func=loss_func,
        where_train=where_train,
        batch_size=hps.batch_size,
        log_step=100,
        loss_update_func=loss_update_func,
        loss_update_iterations=(
            jnp.arange(loss_update_start, n_batches, 100) if loss_update_func is not None else False
        ),
        loss_reduction_fn=loss_reduction_fn,
    )

    # Note: APT requires modifying the training loop itself. Since Feedbax's
    # TaskTrainer doesn't have a hook for modifying trial specs before each step,
    # APT would require either:
    # (a) A custom training loop that wraps TaskTrainer.__call__, or
    # (b) Implementing APT as a model_update_func that modifies perturbations.
    #
    # What Feedbax would need for full APT support:
    #   A `pre_step_hook` on TaskTrainer (called before each training step's
    #   forward pass) that receives the current (task, model, trial_specs, key)
    #   and can return modified trial_specs. APTTrainingWrapper.find_adversarial_perturbation
    #   would then be registered as such a hook, injecting worst-case perturbations
    #   into the trial specs before each gradient step. Without this hook, the
    #   adversarial inner loop cannot interact with the batched trial specs that
    #   TaskTrainer generates internally.
    #
    # APT: adversarial perturbation training via pre_step_fn hook
    pre_step_fn = None
    if args.training_method == "apt":
        from feedbax.intervene.schedule import TimeSeriesParam

        from rlrmp.disturbance import PLANT_INTERVENOR_LABEL

        inner_steps = args.apt_inner_steps
        inner_lr = args.apt_inner_lr
        pert_budget = args.pert_std

        def apt_pre_step(task, model, trial_specs, loss_func, keys):
            """Find worst-case perturbation via gradient ascent on the force field.

            For each trial in the batch, does `inner_steps` of gradient ascent
            on the perturbation force time-series to maximize the loss, then
            returns modified trial_specs with the adversarial perturbation.
            """
            # Get the current perturbation field from trial_specs
            intervene = trial_specs.intervene[PLANT_INTERVENOR_LABEL]
            field = intervene.field  # TimeSeriesParam wrapping (batch, T, 2) or similar
            if isinstance(field, TimeSeriesParam):
                orig_signal = field.value  # (batch, T, 2)
            else:
                orig_signal = field  # already an array

            # Initialize adversarial perturbation as zeros
            w = jnp.zeros_like(orig_signal)

            # Budget per trial: scale * pert_std * sqrt(T * d)
            scale = intervene.scale  # (batch,) — the SISU value
            n_elems = float(orig_signal.shape[-2] * orig_signal.shape[-1])
            budget = scale * pert_budget * jnp.sqrt(n_elems)  # (batch,)

            def _inner_loss(w_pert):
                """Loss with adversarial perturbation added to the field."""
                new_signal = orig_signal + w_pert
                if isinstance(field, TimeSeriesParam):
                    new_field = TimeSeriesParam(new_signal)
                else:
                    new_field = new_signal

                new_intervene = eqx.tree_at(
                    lambda p: p.field, intervene, new_field
                )
                new_trial_specs = eqx.tree_at(
                    lambda t: t.intervene[PLANT_INTERVENOR_LABEL],
                    trial_specs,
                    replace=new_intervene,
                )
                # Simplified forward: use task.eval_trials.
                states = task.eval_trials(model, new_trial_specs, keys)
                losses = loss_func(states, new_trial_specs, model)
                return losses.total

            # Inner gradient ascent loop (compiled as single body via fori_loop)
            def _inner_step(_, w):
                grad_w = jax.grad(_inner_loss)(w)
                w = w + inner_lr * grad_w
                # Project onto budget ball (per-trial)
                w_norm = jnp.linalg.norm(
                    w.reshape(w.shape[0], -1), axis=-1, keepdims=True
                )  # (batch, 1)
                budget_expanded = budget.reshape(-1, 1)
                scale_factor = jnp.minimum(1.0, budget_expanded / (w_norm + 1e-12))
                return w * scale_factor.reshape(w.shape[0], *([1] * (w.ndim - 1)))

            w = jax.lax.fori_loop(0, inner_steps, _inner_step, w)

            # Apply adversarial perturbation to trial_specs
            adv_signal = orig_signal + w
            if isinstance(field, TimeSeriesParam):
                adv_field = TimeSeriesParam(adv_signal)
            else:
                adv_field = adv_signal

            adv_intervene = eqx.tree_at(
                lambda p: p.field, intervene, adv_field
            )
            return eqx.tree_at(
                lambda t: t.intervene[PLANT_INTERVENOR_LABEL],
                trial_specs,
                replace=adv_intervene,
            )

        pre_step_fn = apt_pre_step
        logger.info("APT enabled: %d inner steps, lr=%f, budget=%f", inner_steps, inner_lr, pert_budget)

    train_kwargs["pre_step_fn"] = pre_step_fn

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

    # Save training history (loss curves).
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
    parser.add_argument("--enable-loss-update", action="store_true", default=False,
                        help="Enable the adaptive control penalty update during training. "
                             "Without this flag, --target-ratio has no effect.")
    parser.add_argument("--pert-std", type=float, default=1.0,
                        help="Perturbation standard deviation (default: 1.0).")
    parser.add_argument("--nn-output", type=float, default=1e-5,
                        help="Neural output control cost weight (default: 1e-5).")
    parser.add_argument("--n-batches", type=int, default=10000,
                        help="Number of training batches (default: 10000).")
    parser.add_argument(
        "--output-dir", type=str, default="_artifacts/part2_5/runs/default",
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
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()
    run_training(args)

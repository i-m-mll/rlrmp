"""Minimax adversarial training for reaching controllers.

Alternates between:
1. Adversary update: maximize controller loss by adjusting perturbation parameters.
2. Controller update: minimize loss against the current adversary's perturbation.

The adversary (GaussianBumpAdversary) generates a per-trial force profile from
the trial's SISU value. This replaces the random gust perturbations during training.

Integration with Feedbax TaskTrainer:
    The adversary update is implemented as a ``pre_step_fn`` hook, which is called
    by TaskTrainer before each controller gradient step. The hook:
        1. Extracts per-trial SISU values from the trial specs.
        2. Runs ``n_adversary_steps`` gradient-ascent steps on the adversary
           parameters to maximise the controller loss.
        3. Injects the resulting force profile back into the trial specs, replacing
           the existing perturbation field.
    The updated trial specs are then used for the normal controller gradient step.

Usage example::

    from rlrmp.adversary import GaussianBumpAdversary
    from rlrmp.adversarial_training import make_adversary_pre_step_fn
    import jax.random as jr
    import optax

    adversary = GaussianBumpAdversary(n_bumps=3, n_timesteps=130, key=jr.PRNGKey(1))
    adversary_optimizer = optax.adam(1e-3)
    adv_opt_state = adversary_optimizer.init(eqx.filter(adversary, eqx.is_array))

    pre_step_fn, get_adversary = make_adversary_pre_step_fn(
        adversary=adversary,
        adv_opt_state=adv_opt_state,
        adversary_optimizer=adversary_optimizer,
        n_adversary_steps=5,
    )

    # Pass pre_step_fn to TaskTrainer.__call__
"""

from collections.abc import Callable
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import optax
from feedbax import TaskTrialSpec
from feedbax.intervene import TimeSeriesParam
from feedbax.types import TaskModelPair
from jaxtyping import Array, Float, PyTree

from rlrmp.adversary import GaussianBumpAdversary
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL


def _inject_adversary_forces(
    trial_specs: TaskTrialSpec,
    forces: Float[Array, "batch n_timesteps n_force_dims"],
) -> TaskTrialSpec:
    """Return new trial_specs with the adversary force profile substituted in.

    Replaces the ``field`` of the plant intervenor with a TimeSeriesParam
    wrapping the supplied ``forces`` array.

    Args:
        trial_specs: Batched trial specifications (batch dimension outermost).
        forces: Adversary force profile, shape (batch, T, d).

    Returns:
        Modified trial_specs with forces injected as the perturbation field.
    """
    new_field = TimeSeriesParam(forces)
    new_intervene = eqx.tree_at(
        lambda spec: spec.field,
        trial_specs.intervene[PLANT_INTERVENOR_LABEL],
        new_field,
    )
    return eqx.tree_at(
        lambda ts: ts.intervene[PLANT_INTERVENOR_LABEL],
        trial_specs,
        new_intervene,
    )


def _inject_adversary_delta_A(
    trial_specs: TaskTrialSpec,
    delta_A: Float[Array, "n_dim n_state"],
    batch_size: int,
) -> TaskTrialSpec:
    """Return new trial_specs with the adversary ``ΔA`` matrix substituted in.

    Broadcasts a single ``delta_A`` matrix across the batch by stacking. The
    plant intervenor at ``PLANT_INTERVENOR_LABEL`` must already be a
    ``DynamicsMatrixPerturb`` (its params type is ``DynamicsMatrixPerturbParams``)
    — typically wired up by ``swap_plant_intervenor_to_dynamics_matrix`` before
    the adversarial phase begins. Bug: c723082.

    Args:
        trial_specs: Batched trial specifications.
        delta_A: ``ΔA`` matrix, shape ``(n_dim, n_state)``.
        batch_size: Number of trials in the batch.

    Returns:
        Modified trial_specs with the ``delta_A`` injected on every trial.
    """
    delta_A_batched = jnp.broadcast_to(
        delta_A[None, ...], (batch_size,) + delta_A.shape
    )
    new_intervene = eqx.tree_at(
        lambda spec: spec.delta_A,
        trial_specs.intervene[PLANT_INTERVENOR_LABEL],
        delta_A_batched,
    )
    return eqx.tree_at(
        lambda ts: ts.intervene[PLANT_INTERVENOR_LABEL],
        trial_specs,
        new_intervene,
    )


def _adversary_loss(
    adversary: GaussianBumpAdversary,
    task,
    model,
    trial_specs: TaskTrialSpec,
    loss_func,
    keys,
) -> Float[Array, ""]:
    """Scalar loss for the adversary — the mean controller loss over the batch.

    Generates a force profile from the adversary for each trial (conditioned on
    the trial's SISU value), injects it into the trial specs, runs the controller
    forward pass, and returns the mean batch loss (which the adversary maximises).

    Args:
        adversary: Current adversary parameters.
        task: Feedbax task (used to run trials).
        model: Current controller model (held fixed during adversary update).
        trial_specs: Batched trial specifications.
        loss_func: Loss function.
        keys: PRNG keys for the forward pass, shape (batch,).

    Returns:
        Scalar mean controller loss.
    """
    # Infer batch size from the SISU scale array (used only for shape, not value).
    batch_size = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    # Generate force profile once, then broadcast across batch: (batch, T, d).
    # SISU gating is handled by the task (PAI-ASF scale * field); the adversary
    # produces a single SISU-independent profile.
    force_profile = adversary()  # (T, d)
    forces = jnp.broadcast_to(force_profile, (batch_size,) + force_profile.shape)

    # Inject forces into trial specs
    adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

    # Forward pass (model is fixed — stop gradient to prevent adversary from
    # affecting controller parameters during the inner loop).
    # Use eqx.filter + jt.map to stop-gradient only the array leaves, since
    # jax.lax.stop_gradient cannot handle non-JAX types (strings, ints, etc.)
    # that live in the model pytree as static metadata.
    model_stopped = jt.map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
        model,
        is_leaf=eqx.is_array,
    )
    states = task.eval_trials(model_stopped, adv_trial_specs, keys)
    losses = loss_func(states, adv_trial_specs, model)

    return losses.total.mean()


def adversarial_train_step(
    controller_pair: TaskModelPair,
    adversary: GaussianBumpAdversary,
    adversary_opt_state: PyTree,
    adversary_optimizer: optax.GradientTransformation,
    batch_key: Array,
    batch_size: int,
    n_adversary_steps: int = 5,
):
    """One outer training step with adversarial perturbation.

    Runs ``n_adversary_steps`` gradient-ascent steps on the adversary
    (maximising the controller loss), then returns the updated adversary and
    optimizer state. The caller is responsible for the subsequent controller
    gradient step (via the normal TaskTrainer machinery).

    This function is designed for standalone use / debugging. For integration
    with TaskTrainer, use :func:`make_adversary_pre_step_fn` instead.

    Args:
        controller_pair: TaskModelPair containing the task and current model.
        adversary: GaussianBumpAdversary to update.
        adversary_opt_state: optax optimizer state for the adversary.
        adversary_optimizer: optax optimizer for the adversary (ascent via negated grad).
        batch_key: PRNG key for the forward pass.
        batch_size: Number of trials per batch.
        n_adversary_steps: Number of inner gradient-ascent steps.

    Returns:
        Tuple of (updated_adversary, updated_adversary_opt_state, adversary_loss).
    """
    task = controller_pair.task
    model = controller_pair.model
    loss_func = task.loss_func

    trial_keys = jax.random.split(batch_key, batch_size)
    trial_specs = jax.vmap(task.get_trial_spec)(trial_keys)

    def _loss_for_ascent(adv: GaussianBumpAdversary) -> Float[Array, ""]:
        return _adversary_loss(adv, task, model, trial_specs, loss_func, trial_keys)

    # Inner gradient-ascent loop: maximise controller loss w.r.t. adversary params
    for _ in range(n_adversary_steps):
        # Gradient of the loss w.r.t. adversary parameters (learnable arrays only)
        grads = eqx.filter_grad(_loss_for_ascent)(adversary)

        # Negate gradients: we maximise the loss (gradient ascent)
        neg_grads = jt.map(lambda g: -g, grads)

        updates, adversary_opt_state = adversary_optimizer.update(
            neg_grads, adversary_opt_state, eqx.filter(adversary, eqx.is_array)
        )
        adversary = eqx.apply_updates(adversary, updates)

    adversary_loss = _loss_for_ascent(adversary)

    return adversary, adversary_opt_state, adversary_loss


def make_adversary_pre_step_fn(
    adversary: GaussianBumpAdversary,
    adv_opt_state: PyTree,
    adversary_optimizer: optax.GradientTransformation,
    n_adversary_steps: int = 5,
) -> tuple[Callable, Callable]:
    """Create a ``pre_step_fn`` for TaskTrainer that updates the adversary in-place.

    Returns a closure over mutable Python-level state (the current adversary and
    optimizer state). Each call to the returned ``pre_step_fn`` runs
    ``n_adversary_steps`` gradient-ascent steps on the adversary, then injects
    the resulting force profiles into the trial specs for the controller step.

    Args:
        adversary: Initial GaussianBumpAdversary.
        adv_opt_state: Initial optax optimizer state for the adversary.
        adversary_optimizer: optax optimizer for the adversary.
        n_adversary_steps: Number of gradient-ascent steps per controller step.

    Returns:
        Tuple of ``(pre_step_fn, get_adversary)`` where:
            - ``pre_step_fn(task, model, trial_specs, loss_func, keys)`` is
              the hook to pass to TaskTrainer, returning modified trial_specs.
            - ``get_adversary()`` returns the current adversary (after training).

    Example::

        pre_step_fn, get_adversary = make_adversary_pre_step_fn(
            adversary, adv_opt_state, adversary_optimizer, n_adversary_steps=5
        )
        trainer(task, model, n_batches=1000, pre_step_fn=pre_step_fn, ...)
        trained_adversary = get_adversary()
    """
    # Mutable state lives in a list so the closure can rebind it
    state = [adversary, adv_opt_state]

    def pre_step_fn(task, model, trial_specs: TaskTrialSpec, loss_func, keys):
        """Update adversary, then inject its forces into trial_specs.

        Called by TaskTrainer before each controller gradient step.
        """
        adv, opt_st = state

        def _loss_for_ascent(a: GaussianBumpAdversary) -> Float[Array, ""]:
            return _adversary_loss(a, task, model, trial_specs, loss_func, keys)

        # Gradient-ascent loop (runs eagerly; jit applied by TaskTrainer outside)
        for _ in range(n_adversary_steps):
            grads = eqx.filter_grad(_loss_for_ascent)(adv)
            # Negate: we want ascent (maximise loss)
            neg_grads = jt.map(lambda g: -g, grads)
            updates, opt_st = adversary_optimizer.update(
                neg_grads, opt_st, eqx.filter(adv, eqx.is_array)
            )
            adv = eqx.apply_updates(adv, updates)

        # Commit updated state
        state[0] = adv
        state[1] = opt_st

        # Generate force profile with the updated adversary; broadcast across batch.
        batch_size = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
        force_profile = adv()  # (T, d)
        forces = jnp.broadcast_to(force_profile, (batch_size,) + force_profile.shape)

        # Replace the perturbation field in trial_specs with the adversary forces
        return _inject_adversary_forces(trial_specs, forces)

    def get_adversary() -> GaussianBumpAdversary:
        """Return the current adversary (useful after training to inspect it)."""
        return state[0]

    return pre_step_fn, get_adversary

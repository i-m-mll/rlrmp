"""Compatibility helpers for minimax adversary intervention swaps."""
import equinox as eqx
import jax.numpy as jnp
from feedbax.graph import Wire
from feedbax.intervene import (
    DynamicsMatrixPerturb,
    DynamicsMatrixPerturbParams,
)
from jaxtyping import PyTree


def swap_plant_intervenor_to_dynamics_matrix(
    model: PyTree,
    label: str,
    *,
    mass: float = 1.0,
    n_dim: int = 2,
    n_state: int = 4,
) -> PyTree:
    """Replace a plant intervenor (FixedField/CurlField) with DynamicsMatrixPerturb.

    Used by ``train_minimax.py`` when ``--adversary-type linear_dynamics`` is
    set: the model graph is initially wired with the warm-start ``FixedField``
    (or similar) intervenor at ``label``, and we swap in a
    ``DynamicsMatrixPerturb`` for the adversarial phase. The label is
    preserved so existing scheduling / trial-spec routing continues to work
    after the swap.

    The swapped intervenor is constructed with ``active=False`` and a zero
    ``delta_A``; the training loop drives both via per-batch param overrides.

    Args:
        model: The model graph (e.g. SimpleFeedback) with an existing plant
            intervenor at ``label``.
        label: The intervenor's node name in the graph.
        mass: Effector mass for the ``DynamicsMatrixPerturb`` constructor.
        n_dim: Number of velocity rows in ``delta_A`` (default 2 for 2D
            reaches).
        n_state: Number of state columns in ``delta_A`` (default 4 for
            ``[pos, vel]`` with 2D reaches).

    Returns:
        Modified model graph with the swapped intervenor.
    """
    new_intervenor = DynamicsMatrixPerturb(
        params=DynamicsMatrixPerturbParams(
            active=False,
            delta_A=jnp.zeros((n_dim, n_state), dtype=jnp.float32),
        ),
        label=label,
        mass=mass,
    )
    # Swap the node in the graph by replacing the existing component at `label`.
    model = eqx.tree_at(
        lambda g: g.nodes[label],
        model,
        new_intervenor,
        is_leaf=lambda x: x is None,
    )
    # Ensure an effector wire exists into the swapped intervenor. The original
    # FixedField/AddNoise wiring did not include `mechanics:effector`, but
    # DynamicsMatrixPerturb requires it. We add it idempotently.
    needed_wire = Wire("mechanics", "effector", label, "effector", temporality="recurrent")
    existing = list(getattr(model, "wires", ()) or ())
    if needed_wire not in existing:
        model = model.add_wire(needed_wire)
    return model


def swap_task_intervention_to_dynamics_matrix(
    task: PyTree,
    label: str,
    *,
    n_dim: int = 2,
    n_state: int = 4,
) -> PyTree:
    """Swap a task's intervention-spec params to ``DynamicsMatrixPerturbParams``.

    Counterpart to ``swap_plant_intervenor_to_dynamics_matrix`` for the model
    side. Preserves the existing ``scale`` and ``active`` callables (which
    encode SISU sampling and bernoulli-active masks) while replacing the
    perturbation-shape fields with a zero ``delta_A`` of the requested shape.
    Bug: c723082.

    The training loop substitutes per-batch ``delta_A`` values via
    ``_inject_adversary_delta_A``; this helper just installs the right
    placeholder so trial_specs end up with a sensibly-shaped
    ``DynamicsMatrixPerturbParams`` after evaluation.

    Args:
        task: The task instance (e.g. ``DelayedReaches``) with an
            ``intervention_specs.training[label]`` of any params type.
        label: The intervenor label whose params to swap.
        n_dim: Number of velocity rows in ``delta_A``.
        n_state: Number of state columns in ``delta_A``.

    Returns:
        Modified task with swapped intervention-spec params.
    """
    from feedbax.intervene import InterventionSpec  # local to avoid cycle

    def _scheduled_dynamics_matrix_params(**values):
        placeholders = {
            "scale": 1.0,
            "active": False,
        }
        init_values = {
            key: placeholders[key] if key in placeholders and callable(value) else value
            for key, value in values.items()
        }
        params = DynamicsMatrixPerturbParams(**init_values)
        for key, value in values.items():
            if callable(value):
                object.__setattr__(params, key, value)
        return params

    def _swap_one(specs):
        if label not in specs:
            return specs
        old_spec = specs[label]
        old_params = old_spec.params
        # Reuse scale/active so SISU/bernoulli routing carries over.
        scale = getattr(old_params, "scale", 1.0)
        active = getattr(old_params, "active", True)
        new_params = _scheduled_dynamics_matrix_params(
            scale=scale,
            active=active,
            delta_A=jnp.zeros((n_dim, n_state), dtype=jnp.float32),
        )
        return {
            **specs,
            label: InterventionSpec(
                params=new_params,
                default_active=old_spec.default_active,
            ),
        }

    task = eqx.tree_at(
        lambda t: t.intervention_specs.training,
        task,
        _swap_one(task.intervention_specs.training),
        is_leaf=lambda x: x is None,
    )
    if hasattr(task.intervention_specs, "validation"):
        task = eqx.tree_at(
            lambda t: t.intervention_specs.validation,
            task,
            _swap_one(task.intervention_specs.validation),
            is_leaf=lambda x: x is None,
        )
    return task

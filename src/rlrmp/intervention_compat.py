"""Intervention compatibility helpers for eager-models architecture.

This module provides helper functions for creating and adding interventions
to models using the new graph-based API.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from feedbax.graph import Wire
from feedbax.intervene import (
    CurlField,
    CurlFieldParams,
    DynamicsMatrixPerturb,
    DynamicsMatrixPerturbParams,
    FixedField,
    FixedFieldParams,
    AddNoise,
    AddNoiseParams,
)
from jax_cookbook import is_module
from jaxtyping import PyTree


# Mapping from disturbance type names to intervention classes and param classes
PLANT_INTERVENTION_CLASSES = {
    "curl": (CurlField, CurlFieldParams),
    "constant": (FixedField, FixedFieldParams),
    "gusts": (FixedField, FixedFieldParams),
    "noise": (AddNoise, AddNoiseParams),
    # Bug: c723082 — model-class ΔA·x adversary, force-channel embedding
    "dynamics_matrix": (DynamicsMatrixPerturb, DynamicsMatrixPerturbParams),
}


def create_plant_intervention(
    intervention_type: str,
    label: str,
    *,
    active: bool = False,
    **params,
) -> CurlField | FixedField | AddNoise | DynamicsMatrixPerturb:
    """Create a plant intervention component with the given parameters.

    Args:
        intervention_type: Type of intervention ("curl", "constant", "gusts", "noise")
        label: Label for the intervention (used for parameter scheduling)
        active: Whether the intervention is active by default
        **params: Additional parameters passed to the params class

    Returns:
        An intervention component ready to be inserted into a model graph.
    """
    if intervention_type not in PLANT_INTERVENTION_CLASSES:
        raise ValueError(f"Unknown intervention type: {intervention_type}")

    component_class, params_class = PLANT_INTERVENTION_CLASSES[intervention_type]
    intervention_params = params_class(active=active, **params)
    return component_class(params=intervention_params, label=label)


def add_plant_intervention(
    model: PyTree,
    intervention: CurlField | FixedField | AddNoise | DynamicsMatrixPerturb,
    node_name: str,
) -> PyTree:
    """Add a plant intervention to a model by inserting it into the graph.

    For SimpleFeedback models, this inserts the intervention between the
    efferent channel (or force filter) and the mechanics node.

    Args:
        model: The model (SimpleFeedback or similar) to modify.
        intervention: The intervention component to insert.
        node_name: Name for the intervention node in the graph.

    Returns:
        Modified model with the intervention inserted.
    """
    # Determine where to insert based on model structure
    # For SimpleFeedback: insert between efferent->mechanics or force_filter->mechanics
    if hasattr(model, 'force_lp') and model.force_lp is not None:
        # Insert between force_filter and mechanics
        source_node = "force_filter"
        source_port = "output"
    else:
        # Insert between efferent and mechanics
        source_node = "efferent"
        source_port = "output"

    target_node = "mechanics"
    target_port = "force"

    # CurlField and DynamicsMatrixPerturb both need an `effector` input in
    # addition to `force`; FixedField/AddNoise need only `force`.
    needs_effector = isinstance(intervention, (CurlField, DynamicsMatrixPerturb))
    model = model.insert_between(
        node_name,
        intervention,
        source_node,
        source_port,
        target_node,
        target_port,
        input_port="force",
        output_port="force",
    )
    if needs_effector:
        model = model.add_wire(Wire("mechanics", "effector", node_name, "effector"))

    # Add an input binding so that time-varying intervention params can be
    # passed as model inputs keyed by "intervene:{label}".  The training
    # loop (grad_wrap_abstract_loss) and eval_single extract TimeSeriesParam
    # leaves from trial_spec.intervene and pass them under this key.
    intervene_port = f"intervene:{intervention.label}"
    model = eqx.tree_at(
        lambda g: g.input_ports,
        model,
        model.input_ports + (intervene_port,),
    )
    model = eqx.tree_at(
        lambda g: g.input_bindings,
        model,
        {**model.input_bindings, intervene_port: (node_name, "params_override")},
    )

    return model


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


def add_plant_intervention_to_ensemble(
    models: PyTree,
    intervention_type: str,
    label: str,
    *,
    active: bool = False,
    **params,
) -> PyTree:
    """Add a plant intervention to an ensemble of models.

    Args:
        models: PyTree of models to modify.
        intervention_type: Type of intervention ("curl", "constant", "gusts", "noise")
        label: Label for the intervention (used for parameter scheduling)
        active: Whether the intervention is active by default
        **params: Additional parameters passed to the params class

    Returns:
        Modified ensemble with interventions inserted.
    """

    def _add_to_model(model):
        intervention = create_plant_intervention(
            intervention_type, label, active=active, **params
        )
        return add_plant_intervention(model, intervention, label)

    return jt.map(_add_to_model, models, is_leaf=is_module)

"""Compatibility helpers for minimax adversary intervention bindings."""

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
from feedbax.intervene import (
    DynamicsMatrixPerturb,
    DynamicsMatrixPerturbParams,
    InterventionSpec,
)
from feedbax.runtime.graph import Wire
from jaxtyping import PyTree

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL


LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET = {
    "role": "component_parameter",
    "source_data_id": "linear_dynamics_adversary_params",
    "target_node_id": PLANT_INTERVENOR_LABEL,
    "target_port": "params_override",
    "task_parameter_label": PLANT_INTERVENOR_LABEL,
    "temporal_support": "trajectory",
}


def require_exactly_one_intervenor_for_dynamics_matrix_swap(
    model: PyTree,
    label: str,
) -> PyTree:
    """Validate that a legacy linear-dynamics checkpoint swap is well-scoped."""

    nodes = getattr(model, "nodes", None)
    if not isinstance(nodes, Mapping):
        raise ValueError(
            "DynamicsMatrixPerturb swap requires a graph-like model with a nodes mapping"
        )

    matches = sorted(
        {
            str(name)
            for name, node in nodes.items()
            if name == label or getattr(node, "label", None) == label
        }
    )
    if len(matches) != 1:
        raise ValueError(
            "DynamicsMatrixPerturb swap requires exactly one intervenor node "
            f"matching label {label!r}; found {len(matches)}: {matches}"
        )

    node = nodes[matches[0]]
    if isinstance(node, DynamicsMatrixPerturb):
        raise ValueError(
            "DynamicsMatrixPerturb swap would be applied twice to intervenor node "
            f"{matches[0]!r}"
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
    """Replace a legacy plant intervenor with ``DynamicsMatrixPerturb``.

    Historical linear-dynamics minimax checkpoints were saved after swapping the
    warmup plant intervenor to this component. Older analysis scripts still need
    the same template reconstruction before deserializing those checkpoint
    PyTrees; current spec-first training binds the component parameter target
    declaratively instead.
    """

    require_exactly_one_intervenor_for_dynamics_matrix_swap(model, label)
    new_intervenor = DynamicsMatrixPerturb(
        params=DynamicsMatrixPerturbParams(
            active=False,
            delta_A=jnp.zeros((n_dim, n_state), dtype=jnp.float32),
        ),
        label=label,
        mass=mass,
    )
    model = eqx.tree_at(
        lambda graph: graph.nodes[label],
        model,
        new_intervenor,
        is_leaf=lambda value: value is None,
    )
    needed_wire = Wire(
        "mechanics",
        "effector",
        label,
        "effector",
        temporality="recurrent",
    )
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
    """Swap legacy task intervention params to ``DynamicsMatrixPerturbParams``."""

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
        new_params = _scheduled_dynamics_matrix_params(
            scale=getattr(old_params, "scale", 1.0),
            active=getattr(old_params, "active", True),
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
        lambda current: current.intervention_specs.training,
        task,
        _swap_one(task.intervention_specs.training),
        is_leaf=lambda value: value is None,
    )
    if hasattr(task.intervention_specs, "validation"):
        task = eqx.tree_at(
            lambda current: current.intervention_specs.validation,
            task,
            _swap_one(task.intervention_specs.validation),
            is_leaf=lambda value: value is None,
        )
    return task


__all__ = [
    "LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET",
    "require_exactly_one_intervenor_for_dynamics_matrix_swap",
    "swap_plant_intervenor_to_dynamics_matrix",
    "swap_task_intervention_to_dynamics_matrix",
]

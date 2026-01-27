"""Intervention compatibility helpers for eager-models architecture.

This module provides helper functions for creating and adding interventions
to models using the new graph-based API.
"""

from typing import Optional

import equinox as eqx
import jax.tree as jt
from feedbax.graph import Wire
from feedbax.intervene import (
    CurlField,
    CurlFieldParams,
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
}


def create_plant_intervention(
    intervention_type: str,
    label: str,
    *,
    active: bool = False,
    **params,
) -> CurlField | FixedField | AddNoise:
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
    intervention: CurlField | FixedField | AddNoise,
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

    # For CurlField, we also need to wire the effector input
    if isinstance(intervention, CurlField):
        # CurlField needs both force and effector inputs
        # First insert between force source and mechanics
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
        # Add wire from mechanics effector output to the intervention
        model = model.add_wire(Wire("mechanics", "effector", node_name, "effector"))
    else:
        # FixedField only needs force input
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

    return model


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

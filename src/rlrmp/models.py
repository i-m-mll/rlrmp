"""Centralized model creation for RLRMP experiments.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from typing import Any, Optional

import jax.random as jr
from feedbax.nn import PopulationStructure
from feedbax.xabdeef.models import point_mass_nn
from feedbax_experiments.types import TreeNamespace
from jax_cookbook.tree import get_ensemble
from jaxtyping import PRNGKeyArray


def _get_or_default(obj: Any, attr: str, default: Any) -> Any:
    """Get attribute value, returning default if attribute is missing or None."""
    value = getattr(obj, attr, default)
    return default if value is None else value


def create_point_mass_nn_ensemble(
    hps: TreeNamespace,
    task,
    n_extra_inputs: int = 0,
    population_structure: Optional[PopulationStructure] = None,
    *,
    key: PRNGKeyArray,
):
    """Create an ensemble of point-mass controlled by neural networks.

    This centralizes the creation of `point_mass_nn` models for RLRMP experiments,
    ensuring consistent parameterization across different training modules.

    Arguments:
        hps: Hyperparameters namespace containing model configuration.
        task: The task the models will be trained to perform.
        n_extra_inputs: Number of additional input channels beyond task/feedback inputs.
            For example, SISU (sensory indication of stimulus uncertainty) adds 1 extra input.
        population_structure: Optional population structure defining connectivity patterns
            for hidden units (input-only, readout-only, recurrent-only, input-readout).
            If None and hps.model contains population_structure config, it will be
            parsed from the config.
        key: Random key for model initialization.

    Returns:
        An ensemble of models as a PyTree.
    """
    # Parse population structure from config if not explicitly provided
    if population_structure is None and hasattr(hps.model, 'population_structure'):
        pop_config = hps.model.population_structure
        key_pop, key = jr.split(key)
        population_structure = PopulationStructure.create(
            hidden_size=hps.model.hidden_size,
            n_input_only=_get_or_default(pop_config, 'n_input_only', 0),
            n_readout_only=_get_or_default(pop_config, 'n_readout_only', 0),
            n_recurrent_only=_get_or_default(pop_config, 'n_recurrent_only', 0),
            n_input_readout=_get_or_default(pop_config, 'n_input_readout', 0),
            assignment_fn=None,  # TODO: support custom assignment functions from config
            key=key_pop,
        )

    return get_ensemble(
        point_mass_nn,
        task,
        n_extra_inputs=n_extra_inputs,
        n=hps.model.n_replicates,
        dt=hps.dt,
        mass=hps.model.effector_mass,
        damping=hps.model.damping,
        hidden_size=hps.model.hidden_size,
        n_steps=hps.task.n_steps,
        feedback_delay_steps=hps.model.feedback_delay_steps,
        feedback_noise_std=hps.model.feedback_noise_std,
        motor_noise_std=hps.model.motor_noise_std,
        tau_rise=hps.model.tau_rise,
        tau_decay=hps.model.tau_rise,  # Note: using tau_rise for both
        population_structure=population_structure,
        key=key,
    )

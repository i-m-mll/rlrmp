"""Centralized model creation for RLRMP experiments.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from functools import partial
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.nn import LeakyRNNCell, PopulationStructure
from feedbax.xabdeef.models import point_mass_nn
from feedbax.types import TreeNamespace
from jax_cookbook.tree import get_ensemble
from jaxtyping import Array, PRNGKeyArray


class VanillaRNNCell(eqx.Module):
    """Wrapper around `LeakyRNNCell` with a 2-arg `__call__(input, state)` interface.

    `SimpleStagedNetwork` calls the hidden cell as `hidden(input, state)` (2 positional
    args).  `LeakyRNNCell.__call__` requires a third `key` argument (for optional
    noise injection), which makes it incompatible.  This wrapper absorbs the `key`
    requirement and always passes a dummy key (safe when `use_noise=False`).

    Args:
        input_size: Number of input features.
        hidden_size: Number of hidden units.
        dt: Simulation timestep. Setting ``tau=dt`` gives ``alpha=1.0`` (pure vanilla RNN).
        use_bias: Whether to include a bias term.
        key: PRNG key for weight initialisation.
    """

    _cell: LeakyRNNCell

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        dt: float = 0.01,
        tau: float | None = None,
        key: PRNGKeyArray,
    ):
        if tau is None:
            tau = dt  # alpha=1.0 → pure vanilla RNN (no leaky integration)
        self._cell = LeakyRNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            use_bias=use_bias,
            use_noise=False,
            dt=dt,
            tau=tau,
            key=key,
        )

    def __call__(self, input: Array, state: Array) -> Array:
        """Forward pass compatible with SimpleStagedNetwork's 2-arg call convention."""
        dummy_key = jr.PRNGKey(0)
        return self._cell(input, state, dummy_key)

    @property
    def input_size(self) -> int:
        return self._cell.input_size

    @property
    def hidden_size(self) -> int:
        return self._cell.hidden_size


def _get_or_default(obj: Any, attr: str, default: Any) -> Any:
    """Get attribute value, returning default if attribute is missing or None."""
    value = getattr(obj, attr, default)
    return default if value is None else value


def create_point_mass_nn_ensemble(
    hps: TreeNamespace,
    task,
    n_extra_inputs: int = 0,
    population_structure: Optional[PopulationStructure] = None,
    hidden_type: Optional[type] = None,
    sisu_gating: str = "additive",
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
        hidden_type: The recurrent cell class to use. Defaults to `eqx.nn.GRUCell`.
            Pass e.g. `functools.partial(feedbax.nn.LeakyRNNCell, dt=hps.dt)` for a
            vanilla leaky RNN.
        key: Random key for model initialization.

    Returns:
        An ensemble of models as a PyTree.
    """
    if hidden_type is None:
        hidden_type = eqx.nn.GRUCell
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
        hidden_type=hidden_type,
        sisu_gating=sisu_gating,
        key=key,
    )

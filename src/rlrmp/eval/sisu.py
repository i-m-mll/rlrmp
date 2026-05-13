"""SISU-level manipulation for evaluation trials.

SISU = Subject-Imposed Sensorimotor Uncertainty. It is the conditioning
scalar on the plant-side intervenor (``PLANT_INTERVENOR_LABEL``) and on the
network's input. Changing SISU at evaluation time is a common manoeuvre
across all training methods.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL

__all__ = ["set_sisu"]


def set_sisu(val_trials, sisu_val: float):
    """Return a copy of ``val_trials`` with a fixed SISU level.

    Replaces both the plant-intervenor scale and the network's ``"sisu"`` input
    with a constant ``sisu_val`` across all trials. Pure with respect to
    ``val_trials`` (no in-place mutation).

    Args:
        val_trials: ``TaskTrialSpec`` (or compatible PyTree) whose ``.intervene``
            mapping contains an entry at ``PLANT_INTERVENOR_LABEL`` with a
            ``.scale`` array of shape ``(n_trials,)`` and whose ``.inputs``
            mapping contains a ``"sisu"`` entry of the same shape.
        sisu_val: Scalar SISU level to broadcast across all trials.

    Returns:
        A new trial-spec PyTree with both fields set to ``sisu_val``.
    """
    n_trials = val_trials.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]
    new_trials = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        val_trials,
        jnp.full((n_trials,), sisu_val),
    )
    new_trials = eqx.tree_at(
        lambda t: t.inputs["sisu"],
        new_trials,
        jnp.full((n_trials,), sisu_val),
    )
    return new_trials

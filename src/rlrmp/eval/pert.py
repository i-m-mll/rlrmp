"""Perturbation-scaled evaluation helpers.

Compositions of :func:`rlrmp.eval.ensemble.eval_ensemble_on_trials` and
:func:`rlrmp.eval.kinematics.compute_kinematics` that fix the plant-intervenor
perturbation scale at a chosen value before evaluating.

Bug: 8404108 — extracted from ``scripts/eval_minimax.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.eval.ensemble import eval_ensemble_on_trials
from rlrmp.eval.kinematics import compute_kinematics
from rlrmp.eval.sisu import set_sisu

__all__ = ["eval_at_pert0", "eval_at_pert_scale"]


def eval_at_pert0(task, model, sisu: float, *, key) -> dict[str, np.ndarray]:
    """Evaluate ``model`` with ``pert_scale=0`` at a given ``sisu`` level.

    Convenience wrapper that zeros out the plant-intervenor perturbation
    scale and runs an ensemble evaluation. Useful for measuring the unperturbed
    velocity / endpoint profile of a model.

    Args:
        task: Task whose ``validation_trials`` are used as the trial source.
        model: Ensembled model PyTree.
        sisu: SISU level to evaluate at.
        key: PRNG key.

    Returns:
        Kinematics dict from :func:`compute_kinematics`.
    """
    val_trials = task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)
    pert_shape = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.zeros(pert_shape),
    )
    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)


def eval_at_pert_scale(
    task, model, sisu: float, pert_scale: float, *, key,
) -> dict[str, np.ndarray]:
    """Evaluate ``model`` at a given ``pert_scale`` and ``sisu`` level.

    Sets the plant-intervenor perturbation scale to ``pert_scale`` across all
    trials, then runs an ensemble evaluation. Useful for sweeping the
    perturbation magnitude axis at a fixed SISU.

    Args:
        task: Task whose ``validation_trials`` are used as the trial source.
        model: Ensembled model PyTree.
        sisu: SISU level to evaluate at.
        pert_scale: Constant perturbation scale to apply across all trials.
        key: PRNG key.

    Returns:
        Kinematics dict from :func:`compute_kinematics`.
    """
    val_trials = task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)
    pert_shape = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape
    trial_specs = eqx.tree_at(
        lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
        trial_specs,
        jnp.full(pert_shape, pert_scale),
    )
    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)

"""Ensemble (replicate-vmapped) trial evaluation.

Bug: 8404108 — extracted from ``scripts/eval_part2_5_figures.py`` and made
training-method-agnostic.
"""

from __future__ import annotations

import equinox as eqx
import jax.random as jr

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL

__all__ = ["N_REPLICATES", "eval_ensemble_on_trials"]


#: Default number of replicates across the rlrmp project.
#:
#: Kept here as a module-level constant for backwards compatibility with the
#: legacy ``eval_part2_5_figures.py`` API. New code should prefer passing
#: ``n_replicates=`` explicitly to :func:`eval_ensemble_on_trials`.
N_REPLICATES: int = 5


def eval_ensemble_on_trials(task, model, trial_specs, *, key, n_replicates: int = N_REPLICATES):
    """Evaluate ``n_replicates`` models on the given ``trial_specs``.

    Mirrors feedbax's ``_eval_ensemble`` partitioning strategy: model leaves
    that carry the replicate dimension (i.e. arrays whose leading axis has
    length ``n_replicates``) are vmapped over; everything else is held fixed
    via :func:`equinox.partition` / :func:`equinox.combine`.

    Args:
        task: The task object whose ``eval_trials`` is called per replicate.
        model: An ensembled model (replicate-batched along leading axis on the
            array leaves; ``StateIndex.init.field`` and similar non-batched
            leaves are handled automatically).
        trial_specs: ``TaskTrialSpec`` for the trials to evaluate.
        key: PRNGKey, split into ``n_replicates`` sub-keys.
        n_replicates: Replicate-axis size. Default :data:`N_REPLICATES`.

    Returns:
        States PyTree with leading replicate dimension:
        ``(n_replicates, n_trials, n_steps, ...)``.
    """
    n_trials = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape[0]

    def _is_batched_array(x):
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_replicates

    models_arrays, models_other = eqx.partition(model, _is_batched_array)

    def eval_one_replicate(model_arrays, model_other, rep_key):
        rep_model = eqx.combine(model_arrays, model_other)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(rep_model, trial_specs, keys)

    rep_keys = jr.split(key, n_replicates)
    states = eqx.filter_vmap(
        eval_one_replicate,
        in_axes=(0, None, 0),
    )(models_arrays, models_other, rep_keys)
    return states

from collections.abc import Callable, Sequence
from typing import Optional

import jax.numpy as jnp
import jax.tree as jt
from feedbax_experiments.analysis.pca import PCAResults
from feedbax_experiments.tree_utils import ldict_level_to_top
from feedbax_experiments.types import LDict, TreeNamespace
from jax_cookbook import is_module
from jaxtyping import Array


def get_state_pcs(pca_results: PCAResults, states):
    return jt.map(
        lambda pca_results_by_std, states_by_std: LDict.of("train__pert__std")(
            {
                std: pca_results_by_std[std].batch_transform(states_by_std[std])
                for std in pca_results_by_std
            }
        ),
        pca_results,
        # Ensure level is above the `PCAResults` level.
        ldict_level_to_top("train__pert__std", states, is_leaf=is_module),
        is_leaf=LDict.is_of("train__pert__std"),
    )


def segment_epochs(
    states,
    *,
    hps_common,
    where_idxs: Callable[[TreeNamespace], Sequence[int]],
    labels: Optional[Sequence[str]] = None,
    time_axis: int = -2,
):
    idxs = where_idxs(hps_common)

    if labels is None:
        labels = [f"epoch{i}" for i in range(len(idxs) - 1)]

    return jt.map(
        lambda states: jt.map(
            lambda arr: LDict.of("epoch")(zip(labels, jnp.split(arr, idxs, axis=time_axis))),
            states,
        ),
        states,
        is_leaf=is_module,
    )

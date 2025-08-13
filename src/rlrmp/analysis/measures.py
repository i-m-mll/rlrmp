from collections.abc import Callable, Iterable, Sequence
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional, Dict, Any, TypeAlias

import equinox as eqx
from equinox import filter_vmap as vmap
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PyTree

from jax_cookbook import is_type, compose
import numpy as np

from rlrmp.plot_utils import get_label_str
from rlrmp.types import AnalysisInputData, TreeNamespace, LDictConstructor
from rlrmp.analysis.aligned import AlignedVars, Direction, Measure, ResponseVar
from rlrmp.analysis.analysis import (
    AbstractAnalysis, 
    AbstractAnalysisPorts, 
    DefaultFigParamNamespace, 
    FigParamNamespace, 
    InputOf, 
    SinglePort,
)
from rlrmp.misc import lohi
from rlrmp.plot import get_measure_replicate_comparisons, get_violins
from rlrmp.tree_utils import (
    ldict_label_only_func,
    ldict_level_to_bottom, 
    prefix_expand,
    subdict, 
    tree_level_labels,
    tree_level_types, 
    tree_subset_ldict_level,
)
from rlrmp.types import LDict


# Common transformations

    
    
frob = lambda x: jnp.linalg.norm(x, axis=(-1, -2), ord='fro')

    
def output_corr(
    activities: Float[Array, "evals replicates conditions time hidden"], 
    weights: Float[Array, "replicates outputs hidden"],
):
    # center the activities in time
    activities = activities - jnp.mean(activities, axis=-2, keepdims=True)
    
    def corr(x, w):
        z = jnp.dot(x, w.T)
        return frob(z) / (frob(w) * frob(x))

    corrs = vmap(
        # Vmap over evals and reach conditions (activities only)
        vmap(vmap(corr, in_axes=(0, None)), in_axes=(0, None)), 
        # Vmap over replicates (appears in both activities and weights)
        in_axes=(1, 0),
    )(activities, weights)
    
    # Return the replicate axis to the same position as in `activities`
    return jnp.moveaxis(corrs, 0, 1)


#! TODO
# measure_ranges = {
#     key: (
#             jnp.nanmin(measure_data_stacked),
#             jnp.nanmax(measure_data_stacked),   
#     )
#     for key, measure_data_stacked in {
#         key: jnp.stack(jt.leaves(measure_data))
#         for key, measure_data in all_measure_values.items()
#     }.items()
# }



class ApplyFuncs(AbstractAnalysis[SinglePort[PyTree]]):
    """Apply a PyTree of callables to a PyTree of data.

    The functions are stored in the instance field `funcs` and are applied to
    the leaves of the `data` PyTree (customizable via `is_data_leaf`).

    - If `funcs` is not structurally aligned with `data`, set
      `expand_funcs_to_data=True` to prefix-expand functions across the
      structure of `data` (useful when one set of functions is broadcast over
      many conditions).
    - By default, callable/equinox-`Measure` leaves are treated as function
      leaves; `Responses` or arrays are treated as data leaves.
    """

    Ports = SinglePort
    inputs: SinglePort[PyTree] = eqx.field(
        default_factory=SinglePort[PyTree], converter=SinglePort[PyTree].converter
    )

    funcs: PyTree[Callable] = eqx.field(kw_only=True)  # required at runtime
    is_leaf: Optional[Callable[[Any], bool]] = None

    def _apply_func(self, func, subdata):
        return jt.map(lambda leaf: func(leaf), subdata, is_leaf=self.is_leaf)

    def compute(self, data: AnalysisInputData, *, input: PyTree, **kwargs):
        return jt.map(
            lambda func: jt.map(
                lambda x: func(x),
                input,
                is_leaf=self.is_leaf,
            ),
            self.funcs,
            is_leaf=callable,
        )


class ViolinsPorts(AbstractAnalysisPorts):
    """Input ports for `Violins` analysis.

    - `data`: required PyTree to plot; expects the innermost two LDict levels
      at figure time to represent [legend groups, x-axis bins]. Use fig-ops
      like `.map_figs_at_level` to slice deeper trees.
    - `data_split`: optional matching PyTree for split violins.
    """
    input: InputOf[PyTree]
    input_split: InputOf[PyTree] | None = None


class Violins(AbstractAnalysis[ViolinsPorts]):
    Ports = ViolinsPorts
    inputs: ViolinsPorts = eqx.field(
        default_factory=ViolinsPorts, converter=ViolinsPorts.converter
    )

    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        violinmode="overlay",
        zero_hline=False,
        arr_axis_labels=None,
        legend_title=None,
        xaxis_title=None,
        yaxis_title=None,  # Often provided per-slice via fig-ops
    )

    #? TODO: How can we map figures over multiple levels? 
    #? e.g. over measure and also over pert__amp, for `part2.plant_perts`?
    def make_figs(
        self, data: AnalysisInputData, *, result, colors, input: PyTree, input_split=None, **kwargs
    ):
        # Determine the two innermost LDict levels for grouping and x-axis
        level_types = tree_level_types(input)
        level_labels = [ldict_label_only_func(node_type) for node_type in level_types]
        
        if (
            len(level_labels) < 2 
            or any(not isinstance(t, LDictConstructor) for t in level_types[-2:])
        ):
            raise ValueError("Violins expects at least two inner LDict levels to determine group/x axes.")

        group_label, x_label = level_labels[-2:]

        # Prepare figure params
        legend_title = self.fig_params.legend_title or get_label_str(group_label)
        xaxis_title = self.fig_params.xaxis_title or get_label_str(x_label)

        def _make_fig(node, node_split=None):
            return get_violins(
                node,
                data_split=node_split,
                split_mode='whole' if node_split is None else 'split',
                legend_title=legend_title,
                violinmode=self.fig_params.violinmode or 'overlay',
                arr_axis_labels=self.fig_params.arr_axis_labels,
                zero_hline=bool(self.fig_params.zero_hline),
                yaxis_title=self.fig_params.yaxis_title or "Value",
                xaxis_title=xaxis_title,
                colors=colors[group_label].dark,
            )

        # Map over any outer levels; at the group level we build one figure
        if input_split is None:
            figs = jt.map(_make_fig, input, is_leaf=LDict.is_of(group_label))
        else:
            figs = jt.map(_make_fig, input, input_split, is_leaf=LDict.is_of(group_label))

        return figs
    
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, input, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(input)[0].shape)),
        )
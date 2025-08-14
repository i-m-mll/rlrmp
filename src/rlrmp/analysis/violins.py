
import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

import numpy as np

from rlrmp.plot_utils import get_label_str
from rlrmp.types import AnalysisInputData, TreeNamespace, LDictConstructor
from rlrmp.analysis.analysis import (
    AbstractAnalysis, 
    AbstractAnalysisPorts, 
    DefaultFigParamNamespace, 
    FigParamNamespace, 
    InputOf,
)
from rlrmp.plot import get_violins
from rlrmp.tree_utils import (
    ldict_label_only_func,
    tree_level_types, 
)
from rlrmp.types import LDict
    
    
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
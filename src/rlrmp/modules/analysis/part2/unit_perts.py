from ast import In
from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import Any, Dict, Optional, Sequence
from typing import Literal as L

import equinox as eqx
import feedbax.plot as fbp
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from equinox import Module
from feedbax.intervene import (
    NetworkConstantInput,
    TimeSeriesParam,
    add_intervenors,
    schedule_intervenor,
)
from feedbax.task import AbstractTask
from feedbax_experiments.analysis import CallWithDeps
from feedbax_experiments.analysis.aligned import (
    DEFAULT_VARSET,
    VAR_LEVEL_LABEL,
    AlignedVars,
    get_trivial_reach_directions,
)
from feedbax_experiments.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    Data,
    DummyNode,
    FigIterCtx,
    IdentityNode,
    InputOf,
)
from feedbax_experiments.analysis.disturbance import (
    PLANT_INTERVENOR_LABEL,
    PLANT_PERT_FNS,
    get_pert_amp_vmap_eval_fn,
)
from feedbax_experiments.analysis.effector import EffectorTrajectories
from feedbax_experiments.analysis.grad import Jacobians
from feedbax_experiments.analysis.network import UnitPreferences
from feedbax_experiments.analysis.pca import StatesPCA
from feedbax_experiments.analysis.plot import ScatterPlots
from feedbax_experiments.analysis.profiles import Profiles
from feedbax_experiments.analysis.regression import Regression, RegressionResults
from feedbax_experiments.analysis.state_utils import (
    get_best_replicate,
    get_constant_task_input_fn,
    get_segment_trials_fn,
    get_symmetric_accel_decel_epochs,
    vmap_eval_ensemble,
)
from feedbax_experiments.analysis.violins import Violins
from feedbax_experiments.colors import ColorscaleSpec
from feedbax_experiments.config import PLOTLY_CONFIG
from feedbax_experiments.constants import POS_ENDPOINTS_ALIGNED
from feedbax_experiments.plot import add_endpoint_traces, get_violins, set_axes_bounds_equal
from feedbax_experiments.tree_utils import (
    ldict_level_keys,
    move_ldict_level_above,
    subdict,
    tree_level_labels,
)
from feedbax_experiments.types import (
    AnalysisInputData,
    LDict,
    TreeNamespace,
    VarSpec,
)
from jax_cookbook import MultiVmapAxes, is_module, is_none, is_type
from jax_cookbook.misc import deep_merge, split_by
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

from rlrmp.constants import RNN_INPUT_CHANNEL_SIZES
from rlrmp.fb_response import InstantFBResponse
from rlrmp.misc import split_rnn_input_channels
from rlrmp.transforms import get_state_pcs, segment_epochs
from rlrmp.types import RNNCellArgs, RNNInputChannels

COLOR_FNS = dict(
    sisu=ColorscaleSpec(
        sequence_fn=lambda hps: hps.sisu,
        colorscale="thermal",
    ),
    stim_amp=ColorscaleSpec(
        sequence_fn=lambda hps: hps.pert.unit.amp,
        colorscale="viridis",
    ),
    pert__amp=ColorscaleSpec(
        sequence_fn=lambda hps: hps.pert.plant.amp,
        colorscale="viridis",
    ),
)


UNIT_STIM_INTERVENOR_LABEL = "UnitStim"

SCALE_UNIT_STIM_BY_READOUT_VECTOR_LENGTH = False

PLANT_PERT_LABELS = {0: "no curl", 1: "curl"}
PLANT_PERT_STYLES = dict(line_dash={0: "dot", 1: "solid"})

SISU_LABELS = {0: -2, 1: 0, 2: 2}
SISU_STYLES = dict(line_dash={0: "dot", 1: "dash", 2: "solid"})

N_PCA = 10
#! Long enough to get some stim effects, not so long that our PCs
#! are influenced by huge variance of some trials after unit stim?
PCA_START_STEP = 0
PCA_END_STEP = 60

aligned_varset = subdict(DEFAULT_VARSET, ("pos", "vel"))


unit_idxs_profiles_plot = jnp.array([2, 83, 95, 97, 43, 48])  # jnp.arange(8)


def unit_stim(hps):
    idxs = slice(
        hps.pert.unit.start_step,
        hps.pert.unit.start_step + hps.pert.unit.duration,
    )
    trial_mask = jnp.zeros((hps.task.n_steps - 1,), bool).at[idxs].set(True)

    return NetworkConstantInput.with_params(
        # active=True,
        active=TimeSeriesParam(trial_mask),
    )


def schedule_unit_stim(*, tasks, models, hps):
    tasks, models = jtree.unzip(
        jt.map(
            lambda task, model, hps: schedule_intervenor(
                task,
                model,
                lambda model: model.step.net,
                # unit_stim(unit_idx, hps=hps),
                unit_stim(hps),
                default_active=False,
                stage_name=None,  # None -> before RNN forward pass; 'hidden' -> after
                label=UNIT_STIM_INTERVENOR_LABEL,
            ),
            tasks,
            models,
            hps,
            is_leaf=is_module,
        )
    )
    return tasks, models


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FNS[hps.pert.plant.type](hps)
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.plant.type}")

    pert_amps = hps.pert.plant.amp

    # Tasks with varying plant perturbation amplitude
    tasks_by_amp, _ = jtree.unzip(
        jt.map(  # over disturbance amplitudes
            lambda pert_amp: schedule_intervenor(  # (implicitly) over train stds
                task_base,
                jt.leaves(models_base, is_leaf=is_module)[0],
                lambda model: model.step.mechanics,
                disturbance(pert_amp),
                label=PLANT_INTERVENOR_LABEL,
                default_active=False,
            ),
            LDict.of("pert__amp")(
                dict(zip(pert_amps, pert_amps)),
            ),
        )
    )

    # Add plant perturbation module (placeholder with amp 0.0) to all loaded models
    models_by_std = jt.map(
        lambda models: add_intervenors(
            models,
            lambda model: model.step.mechanics,
            # The first key is the model stage where to insert the disturbance field;
            # `None` means prior to the first stage.
            # The field parameters will come from the task, so use an amplitude 0.0 placeholder.
            {None: {PLANT_INTERVENOR_LABEL: disturbance(0.0)}},
        ),
        models_base,
        is_leaf=is_module,
    )

    # Also vary tasks by SISU
    tasks = LDict.of("sisu")(
        {
            sisu: jt.map(
                lambda task: task.add_input(
                    name="sisu",
                    input_fn=get_constant_task_input_fn(
                        sisu,
                        hps.task.n_steps - 1,
                        task.n_validation_trials,
                    ),
                ),
                tasks_by_amp,
                is_leaf=is_module,
            )
            for sisu in hps.sisu
        }
    )

    # The outer levels of `models` have to match those of `tasks`
    models, hps = jtree.unzip(jt.map(lambda _: (models_by_std, hps), tasks, is_leaf=is_module))

    # Schedule unit stim for a placeholder unit (0)
    tasks, models = schedule_unit_stim(tasks=tasks, models=models, hps=hps)

    return tasks, models, hps, None


def task_with_scaled_unit_stim(model, task, unit_idx, stim_amp_base, hidden_size, intervenor_label):
    """Scale the magnitude of unit stim based on the length of the unit's readout vector."""
    if SCALE_UNIT_STIM_BY_READOUT_VECTOR_LENGTH:
        readout_vector_length = jnp.linalg.norm(model.step.net.readout.weight[..., unit_idx])
        stim_amp = stim_amp_base / readout_vector_length
    else:
        stim_amp = stim_amp_base

    return eqx.tree_at(
        lambda task: (
            task.intervention_specs.validation[intervenor_label].intervenor.params.scale,
            task.intervention_specs.validation[intervenor_label].intervenor.params.unit_spec,
        ),
        task,
        (
            stim_amp,
            jnp.full(hidden_size, jnp.nan).at[unit_idx].set(1.0),
        ),
        is_leaf=is_none,
    )


def get_task_eval_fn(task_base, hps, unit_idx, stim_amp_base):
    def task_eval_fn(model, key):
        task = task_with_scaled_unit_stim(
            model,
            task_base,
            unit_idx,
            stim_amp_base,
            hps.train.model.hidden_size,
            UNIT_STIM_INTERVENOR_LABEL,
        )
        return task.eval(model, key=key)

    return task_eval_fn


def eval_fn(key_eval, hps, models, task):
    states = eqx.filter_vmap(
        lambda stim_amp: eqx.filter_vmap(
            lambda unit_idx: eqx.filter_vmap(
                lambda key_eval: eqx.filter_vmap(get_task_eval_fn(task, hps, unit_idx, stim_amp))(
                    models, jr.split(key_eval, hps.train.model.n_replicates)
                )
            )(jr.split(key_eval, hps.eval_n))
        )(jnp.arange(hps.train.model.hidden_size))
    )(jnp.array(hps.pert.unit.amp))

    return states


MEASURE_KEYS = ()


# def get_unit_stim_origins_directions(task, models, hps):
#     origins = task.validation_trials.inits["mechanics.effector"].pos
#     directions = jnp.broadcast_to(jnp.array([1., 0.]), origins.shape)
#     return origins, directions


def get_impulse_vrect_kws(hps):
    return dict(
        x0=hps.pert.unit.start_step,
        x1=hps.pert.unit.start_step + hps.pert.unit.duration,
        fillcolor="grey",
        opacity=0.2,
        line_width=0,
        name="Perturbation",
    )


def transform_profile_vars(states_by_var, keepdims=True):
    return LDict.of("var")(
        dict(
            deviation=jnp.linalg.norm(states_by_var["pos"], axis=-1, keepdims=keepdims),
            angle=jnp.arctan2(states_by_var["pos"][..., 1], states_by_var["pos"][..., 0])[
                ..., None
            ],
            speed=jnp.linalg.norm(states_by_var["vel"], axis=-1, keepdims=keepdims),
        )
    )


# def max_deviation_after_stim(states_by_var, *, hps_common, **kwargs):
#     deviation = jnp.linalg.norm(states_by_var["pos"], axis=-1)
#     pert_end = hps_common.pert.unit.start_step + hps_common.pert.unit.duration
#     ts = jnp.arange(pert_end, hps_common.task.n_steps)
#     return jnp.max(deviation[..., ts], axis=-1)


def get_response_vars(states_by_var, *, hps_common):
    """Response variables for regression analysis."""
    vars = transform_profile_vars(states_by_var, keepdims=False)
    pert_end = hps_common.pert.unit.start_step + hps_common.pert.unit.duration
    ts = jnp.arange(pert_end, pert_end + 30)
    return LDict.of("response_var")(
        dict(
            max_deviation=jnp.max(vars["deviation"][..., ts], axis=-1),
            max_speed=jnp.max(vars["speed"][..., ts], axis=-1),
            angle=vars["angle"],
        )
    )


class UnitStimRegressionFiguresPorts(AbstractAnalysisPorts):
    """Input ports for UnitStimRegressionFigures analysis."""

    regression_results: Optional[InputOf[RegressionResults]] = None
    unit_fb_gains: Optional[InputOf[Any]] = None


class UnitStimRegressionFigures(AbstractAnalysis[UnitStimRegressionFiguresPorts]):
    Ports = UnitStimRegressionFiguresPorts
    inputs: UnitStimRegressionFiguresPorts = eqx.field(
        default_factory=UnitStimRegressionFiguresPorts,
        converter=UnitStimRegressionFiguresPorts.converter,
    )

    fig_params: Mapping = MappingProxyType(
        dict(
            mode="markers",
            hovertemplate="unit %{pointNumber}: (%{x}, %{y})",
            layout=dict(
                xaxis_title="Response regressor weight",
                yaxis_title="Feedback gain regressor weight",
            ),
        )
    )

    variant: Optional[str] = "full"

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        regression_results,
        unit_fb_gains,
        replicate_info,
        **kwargs,
    ) -> PyTree[go.Figure]:
        coefs_response = regression_results[0].weight

        coefs_gain = unit_fb_gains[0].weight

        feature_labels = regression_results[1]
        assert unit_fb_gains[1] == feature_labels

        kwargs = dict(self.fig_params)
        layout_kws = kwargs.pop("layout", {})

        figs = LDict.of("regressor")(
            {
                feature_label: go.Figure(
                    data=go.Scatter(
                        x=coefs_gain[:, 0, i],
                        y=coefs_response[:, 0, i],
                        **kwargs,
                    ),
                    layout=layout_kws,
                )
                for i, feature_label in enumerate(feature_labels)
            }
        )

        for feature_label, fig in figs.items():
            fig.update_layout(
                title=self.fig_params["layout"]["title"].replace("{feature_label}", feature_label)
            )

        # ## Plot single regressor weights against each other
        # fig = go.Figure(
        #     layout=dict(
        #         xaxis_title="SISU regression weight",
        #         yaxis_title="Curl amp. regression weight",
        #     )
        # )

        # figs[(feature_labels[1], feature_labels[2])] = fig

        return figs


def jacobian_input_channel_norms(jacobians: Array, *, axis=-1) -> RNNInputChannels[Array]:
    jacobians_split = split_rnn_input_channels(jacobians, axis=axis)
    return jt.map(lambda arr: jnp.linalg.norm(arr, axis=-1), jacobians_split)


rnn_fns = Data.models(where=lambda model: model.step.net.hidden)
# these have batch shape (evals, replicates, reach conditions, time)
rnn_inputs = Data.states(where=lambda states: states.net.input)
rnn_states = Data.states(where=lambda states: states.net.hidden)


DEPENDENCIES = {
    "aligned_vars_trivial": AlignedVars(
        varset=aligned_varset,
        # Bypass alignment; keep aligned with x-y axes
        directions_fn=get_trivial_reach_directions,
    ),
    "hidden_states_pca": (
        StatesPCA(
            n_components=N_PCA,
            where_states=lambda states: states.net.hidden,
            aggregate_over_labels=("pert__amp", "sisu"),
        )
        .after_transform(partial(get_best_replicate, axis=3))
        .after_getitem_at_level("task_variant", "full")
        .after_indexing(
            -2,
            #! TODO: Improve `CallWithDeps` so it works with functions that map over leaves,
            #! like `jtree.take` does in `after_indexing`.
            np.arange(PCA_START_STEP, PCA_END_STEP),
            axis_label="timestep",
        )
    ),
    # "jac-u-ss": (
    #     Jacobians(
    #         inputs=Jacobians.Ports(
    #             fns=rnn_fns,
    #             fn_args=RNNCellArgs(rnn_inputs, rnn_states),
    #         ),
    #         argnums=0,  # with respect to inputs only
    #     )
    #     .after_getitem_at_level("task_variant", "full")
    #     # There's only one reach condition atm
    #     .after_indexing(-3, 0, axis_label="condition", dependency_name="fn_args")
    #     # Computation based on steady-state period; stim vars not relevant.
    #     .after_indexing(1, 0, axis_label="stim_unit_idx", dependency_name="fn_args")
    #     .after_indexing(0, 0, axis_label="stim_amp", dependency_name="fn_args")
    #     # Shape is now: (evals, replicates, time, input/state)
    #     # We'll average the Jacobian over several steady-state timesteps
    #     .after_indexing(
    #         -2,
    #         #! For now, hardcode the steps. But should base on `hps`, generally.
    #         lambda shape: jnp.arange(19, 29),
    #         axis_label="timestep",
    #         dependency_name="fn_args",
    #     )
    #     .vmap(
    #         in_axes={
    #             "fns": MultiVmapAxes(None, 0, None),
    #             "fn_args": MultiVmapAxes(0, 1, 2),
    #         }
    #     )
    #     # Take the mean over the sampled timesteps
    #     .then_transform_result(fn=lambda tree: jt.map(lambda arr: jnp.mean(arr, axis=-3), tree))
    #     # Split the Jacobian by input channel, and take the Euclidean norm for each channel
    #     .then_transform_result(jacobian_input_channel_norms, is_leaf=None)
    #     # Only keep results for the inputs; the state Jacobians weren't calculated anyway
    #     # result: RNNCellArgs; result.state is None
    #     .then_transform_result(lambda result: result.input)
    #     # .estimate_memory()
    # ),
    "unit_stim_response_vars": (
        IdentityNode(
            inputs=IdentityNode.Ports(
                input="aligned_vars_trivial",
            ),
        )
        .after_transform(partial(get_best_replicate, axis=3))
        .after_getitem_at_level("task_variant", "full")
        #! Only compute regression for stim condition
        .after_indexing(0, 1, axis_label="stim_amp")
        # Compute the response variables of interest
        .after_transform(get_response_vars, level="var", dependency_names="input")  #
        .after_rearrange_levels(
            ["response_var", "train__pert__std", "sisu", "pert__amp"],
            dependency_name="input",
        )
        # e.g. positions to deviations
        # .after_transform(max_deviation_after_stim, level="var", dependency_names="regressor_tree")
    ),
    # "unit_stim_regression": (
    #     Regression(
    #         inputs=Regression.Ports(
    #             regressor_tree="unit_stim_response_vars",
    #         ),
    #     )
    #     # Compute distinct regressions for each response variable, and for each train std
    #     .map_compute(is_leaf=LDict.is_of("sisu"), dependency_names="regressor_tree")
    #     #! Compute distinct regressions (in parallel) over stim units
    #     .vmap(in_axes={"regressor_tree": 0})
    # ),
    # "jac-u-ss_regression_prep": (
    #     IdentityNode(
    #         inputs=IdentityNode.Ports(
    #             input="jac-u-ss",
    #         ),
    #     )
    #     .after_transform(partial(get_best_replicate, axis=1))
    #     .after_rearrange_levels(
    #         ["train__pert__std", ...],
    #         dependency_name="input",
    #         # is_leaf=is_type(RNNInputChannels),
    #     )
    #     #! TEMP: Move `RNNInputChannels` above `LDict.of("sisu")` so we can map input channels separately
    #     .after_transform(
    #         lambda tree: jt.map(
    #             lambda subtree: jt.transpose(
    #                 jt.structure(subtree, is_leaf=is_type(RNNInputChannels)),
    #                 None,
    #                 subtree,
    #             ),
    #             tree,
    #             is_leaf=LDict.is_of("sisu"),
    #         ),
    #         dependency_names="input",
    #     )
    # ),
    # "jac-u-ss_regression": (
    #     Regression(
    #         inputs=Regression.Ports(
    #             regressor_tree="jac-u-ss_regression_prep",
    #         ),
    #     )
    #     # Compute distinct regressions for each input channel
    #     .map_compute(is_leaf=LDict.is_of("sisu"), dependency_names="regressor_tree")
    #     #! Compute distinct regressions (in parallel) over stim units
    #     .vmap(in_axes={"regressor_tree": 1})
    # ),
    # "unit_fb_gains": (
    #     InstantFBResponse(
    #         n_directions=24,
    #         fb_pert_amp=0.5,
    #     )
    #     # There's only one reach condition atm
    #     .after_indexing(-3, 0, axis_label="condition", dependency_name="data.states")
    #     # Computation based on steady-state period; stim vars not relevant.
    #     .after_indexing(1, 0, axis_label="stim_unit_idx", dependency_name="data.states")
    #     .after_indexing(0, 0, axis_label="stim_amp", dependency_name="data.states")
    #     .vmap(
    #         in_axes={"rnn_cells": MultiVmapAxes(0, None), "data.states": MultiVmapAxes(0, 0)}
    #     )  # replicates
    # ),
}


def dashed_fig_params_fn(fig_params, ctx: FigIterCtx):
    return fig_params | dict(
        scatter_kws=dict(
            line_dash=SISU_STYLES["line_dash"][ctx.idx],
            legendgroup=SISU_LABELS[ctx.idx],
            legendgrouptitle_text=f"SISU: {SISU_LABELS[ctx.idx]}",
        ),
    )


def regression_fig_params_fn(fig_params, ctx: FigIterCtx):
    if ctx.level == "response_var":
        fig_params = deep_merge(
            fig_params,
            dict(
                layout=dict(
                    title=fig_params["layout"]["yaxis_title"].replace(
                        "{feature_label}", str(ctx.key)
                    ),
                )
            ),
        )
    elif ctx.level == "train__pert__std":
        fig_params = deep_merge(
            fig_params,
            dict(
                layout=dict(
                    title=fig_params["layout"]["title"].replace("{train__pert__std}", str(ctx.key)),
                )
            ),
        )
    elif ctx.level == "input_channel":
        fig_params = deep_merge(
            fig_params,
            dict(
                layout=dict(
                    title=fig_params["layout"]["yaxis_title"].replace(
                        "{input_channel}", str(ctx.key)
                    ),
                )
            ),
        )
    return fig_params


def response_violin_params_fn(fig_params, ctx: FigIterCtx):
    return fig_params | dict(
        yaxis_title=ctx.key,
    )


# PyTree structure: [sisu, pert__amp, train__pert__std]
# Array batch shape: [stim_amp, stim_unit_idx, eval, replicate, condition=1]
ANALYSES = {
    # "JJ": IdentityNode(
    #     inputs=IdentityNode.Ports(
    #         input="jac-u-ss_regression_prep",
    #     ),
    #     tmp_label="1",
    # ),
    # "RR": IdentityNode(
    #     inputs=IdentityNode.Ports(
    #         input="unit_stim_response_vars",
    #     ),
    #     tmp_label="2",
    # ),
    "plot--unit-stim_profiles": (
        Profiles(
            variant="full",
            vrect_kws_fn=get_impulse_vrect_kws,
            coord_labels=None,
            inputs=Profiles.Ports(
                vars="aligned_vars_trivial",
            ),
            agg_mode=dict(deviation="standard", angle="circular", speed="standard"),
        )
        .after_transform(partial(get_best_replicate, axis=3))
        #! Only make figures for a few stim units
        .after_indexing(1, unit_idxs_profiles_plot, axis_label="unit_stim_idx")
        .after_transform(
            transform_profile_vars, level="var", dependency_names="vars"
        )  # e.g. positions to deviations
        .after_unstacking(1, "unit_stim_idx", above_level="pert__amp", dependency_name="vars")
        #! Only make figures for unit stim condition
        # .after_indexing(0, 1, axis_label="stim_amp")
        .after_unstacking(0, "stim_amp", above_level="pert__amp", dependency_name="vars")
        # Plot pert amp. on same figure
        .after_rearrange_levels(
            [..., "train__pert__std", "var", "pert__amp"], dependency_name="vars"
        )
        # .after_subdict_at_level("pert__amp", [0.0, 8.0], dependency_name="vars")
        .after_subdict_at_level("sisu", [-2, 0, 2], dependency_name="vars")
        .combine_figs_by_level(  # Also plot SISU on same figure, with different line styles
            level="sisu",
            fig_params_fn=dashed_fig_params_fn,
        )
        .with_fig_params(
            layout_kws=dict(
                width=500,
                height=350,
            ),
        )
    ),
    #! Compare distributions (over units) of response variables, with varying SiSU
    "plot--stim_response_dists": (
        Violins(
            inputs=Violins.Ports(
                input="unit_stim_response_vars",
            ),
        )
        .with_fig_params(violinmode="group")
        .after_rearrange_levels([..., "sisu", "train__pert__std"], dependency_name="input")
        .map_figs_at_level(
            "response_var",
            fig_params_fn=response_violin_params_fn,
            dependency_name="input",
        )
    ),
    # "plot--unit-stim_hidden_pc": (
    #     ScatterPlots(
    #         inputs=ScatterPlots.Ports(
    #             input=rnn_states,
    #         ),
    #         subplot_level="train__pert__std",
    #         colorscale_key="sisu",
    #     )
    #     .with_fig_params(
    #         axes_labels=fbp.AxesLabels("PC1", "PC2"),
    #         layout_kws=dict(width=800, height=500),
    #     )
    #     .after_getitem_at_level("task_variant", "full", dependency_name="input")
    #     # Take a subset of perturbed units
    #     .after_indexing(1, np.arange(10), axis_label="unit_stim_idx", dependency_name="input")
    #     .after_unstacking(1, "unit_stim_idx", dependency_name="input")
    #     # Expects SISU in the legend
    #     .after_stacking("sisu", dependency_name="input")
    #     # Now shape: (sisu, stim_amp, eval, replicate, condn=1, timestep, state)
    #     .after_transform(
    #         # Pull in the PCA results and use them to transform the hidden states
    #         CallWithDeps("hidden_states_pca")(get_state_pcs),
    #         dependency_names="input",
    #     )
    #     #! Select top 2/3 PCs
    #     .after_indexing(-1, jnp.arange(2), axis_label="pc", dependency_name="input")
    #     .map_figs_at_level("pert__amp", dependency_name="input")
    # ),
    # "plot--unit-jac-u-ss": (
    #     Violins(
    #         inputs=Violins.Ports(
    #             input="jac-u-ss",
    #         ),
    #     )
    # ),
    # "plot--unit-agg-fb-gains": (
    #     Violins(
    #         inputs=Violins.Ports(
    #             input="unit_fb_gains",
    #         ),
    #     )
    #     .after_transform(get_best_replicate)
    #     .after_rearrange_levels([..., "sisu", "train__pert__std"], dependency_name="input")
    # ),
    # "unit_stim_regression_figures": (
    #     UnitStimRegressionFigures(
    #         inputs=UnitStimRegressionFigures.Ports(
    #             regression_results="unit_stim_regression",
    #             unit_fb_gains="jac-u-ss_regression",
    #         ),
    #     )
    #     .with_fig_params(
    #         layout=dict(
    #             xaxis_title="{input_channel} input channel gain coef.",
    #             yaxis_title="{response_var} response coef.",
    #             title="Regressor: {feature_label}; Train pert. std.: {train__pert__std}",
    #         )
    #     )
    #     .after_map(
    #         lambda ntuple: LDict.of("input_channel")(ntuple._asdict()),
    #         is_leaf=is_type(RNNInputChannels),
    #         dependency_name="unit_fb_gains",
    #     )
    #     .map_make_figs_to_output(
    #         ["response_var", "train__pert__std", "input_channel"],
    #         dependency_names=["regression_results", "unit_fb_gains"],
    #         is_leaf=is_type(RegressionResults),
    #         # fig_params_fn=regression_fig_params_fn,
    #     )
    # ),
    #! TODO: Replace with `get_aligned_trajectories_node`
    # "aligned_effector_trajectories": (
    #     AlignedEffectorTrajectories(
    #         variant="full",
    #         custom_inputs={
    #             "aligned_vars": "aligned_vars_trivial",
    #         },
    #     )
    #     .after_transform(partial(get_best_replicate, axis=3))
    # ),
    # "unit_preferences": (
    #     # Result shape: [stim_amp, unit_stim_idx, unit_idx, feature]
    #     UnitPreferences(
    #         variant="full",
    #         feature_fn=lambda task, states: states.efferent.output,
    #     )
    #     .after_transform(partial(get_best_replicate, axis=3))
    #     .after_transform(
    #         # get_segment_trials_fn(get_symmetric_accel_decel_epochs),
    #         partial(
    #             segment_epochs,
    #             where_idxs=lambda hps: (
    #                 hps.pert.unit.start_step,
    #                 hps.pert.unit.start_step + hps.pert.unit.duration,
    #             ),
    #             labels=("pre", "peri", "post"),
    #         ),
    #         dependency_names="states",
    #     )
    #     .vmap(
    #         in_axes={"data.states": MultiVmapAxes(0, 1)}
    #     )  # Compute preferences separately for stim vs. nostim, and for each stim unit
    # ),
}

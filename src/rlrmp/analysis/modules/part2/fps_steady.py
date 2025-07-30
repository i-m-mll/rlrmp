"""
Analysis of fixed point (FP) structure of the network at steady state.

Includes eigendecomposition of the linearization (Jacobian).

Steady state FPs are the "goal-goal" FPs; i.e. the point mass is at the goal, so the
network will stabilize on some hidden state that outputs a constant force that does not change the
position of the point mass on average.

This contrasts with non-steady-state FPs, which correspond to network outputs which should
cause the point mass to move, and thus the network's feedback input (and thus FP) to
change.
"""

from collections.abc import Callable, Sequence
from functools import partial, wraps
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Optional

from arrow import get
import equinox as eqx
from equinox import Module
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
import numpy as np

from feedbax.task import AbstractTask
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rlrmp.analysis import AbstractAnalysis, AnalysisInputData
from rlrmp.analysis.analysis import _DummyAnalysis, LiteralInput, Data, DefaultFigParamNamespace, ExpandTo, FigParamNamespace, Transformed
from rlrmp.analysis.fp_finder import FPFilteredResults, take_top_fps
from rlrmp.analysis.fps import FixedPoints
from rlrmp.analysis.grad import Jacobians, Hessians
from rlrmp.analysis.pca import StatesPCA
from rlrmp.misc import get_constant_input_fn
from rlrmp.analysis.state_utils import get_best_model_replicate, vmap_eval_ensemble
from rlrmp.tree_utils import take_replicate
from rlrmp.types import TreeNamespace
from rlrmp.types import LDict
from rlrmp.analysis.execution import AnalysisModuleTransformSpec
from rlrmp.analysis.fps_tmp import (
    FPsInPCSpace,
)


"""Specify any additional colorscales needed for this analysis. 
These will be included in the `colors` kwarg passed to `AbstractAnalysis` methods
"""
COLOR_FUNCS: dict[str, Callable[[TreeNamespace], Sequence]] = dict(
)


def setup_eval_tasks_and_models(
    task_base: Module,
    models_base: LDict[float, Module],
    hps: TreeNamespace
):
    """Define a task where the point mass is already at the goal.

    This is similar to `feedback_perts`, but without an impulse.
    """
    all_tasks, all_models, all_hps = jtree.unzip(
        LDict.of('sisu')({
            sisu: (
                task_base.add_input(
                    name="sisu",
                    input_fn=get_constant_input_fn(
                        sisu, hps.model.n_steps, task_base.n_validation_trials,
                    ),
                ),
                models_base,  
                hps | dict(sisu=sisu),
            )
            for sisu in hps.sisu
        })
    )
    # Provides any additional data needed for the analysis
    extras = SimpleNamespace()  
    
    return all_tasks, all_models, all_hps, extras


# Unlike `feedback_perts`, we don't need to vmap over impulse amplitude 
eval_func: Callable = vmap_eval_ensemble



N_PCA = 50
START_STEP = 0
END_STEP = 100
STRIDE_FP_CANDIDATES = 16


#! TODO: Refactor out the common parts, e.g. meets criteria, top fps, etc.
def process_fps(all_fps: PyTree[FPFilteredResults]):
    """Only keep FPs/replicates that meet criteria."""
    
    n_fps_meeting_criteria = jt.map(
        lambda fps: fps.counts['meets_all_criteria'],
        all_fps,
        is_leaf=is_type(FPFilteredResults),
    )

    # satisfactory_replicates = jt.map(
    #     lambda n_matching_fps_by_sisu: jnp.all(
    #         jnp.stack(jt.leaves(n_matching_fps_by_sisu), axis=0),
    #         axis=0,
    #     ),
    #     n_fps_meeting_criteria,
    #     is_leaf=LDict.is_of('sisu'),
    # )

    all_top_fps = take_top_fps(all_fps, n_keep=6)

    # Average over the top fixed points, to get a single one for each included replicate and
    # control input.
    fps_final = jt.map(
        lambda top_fps_by_sisu: jt.map(
            lambda fps: jnp.nanmean(fps, axis=-2),
            top_fps_by_sisu,
            is_leaf=is_type(FPFilteredResults),
        ),
        all_top_fps,
        is_leaf=LDict.is_of('sisu'),
    )

    return TreeNamespace(
        fps=fps_final,
        all_top_fps=all_top_fps,
        n_fps_meeting_criteria=n_fps_meeting_criteria,
        # satisfactory_replicates=satisfactory_replicates,
    )


def get_ss_rnn_input(sisu: float, pos: Float[Array, "2"]):
    vel = jnp.zeros_like(pos)
    return jnp.array([sisu, *pos, *vel, *pos, *vel])


def get_ss_rnn_func(rnn_cell: Callable[[Array, Array], Array]):
    def rnn_func(sisu, pos, h):
        input_star = get_ss_rnn_input(sisu, pos)
        return rnn_cell(input_star, h)

    return rnn_func
    
    
def reshape_candidates(states: Array) -> Array:
    """Reshape the initial FP candidates to (positions, candidates, state)."""
    positions_first = jnp.moveaxis(states, 1, 0)
    return jnp.reshape(
        positions_first, 
        (positions_first.shape[0], -1, positions_first.shape[-1]),
    )


"""Apply global transformations to reduce computation.
Since this module frequently uses get_best_replicate, we apply it globally
before evaluation to save computational resources."""
TRANSFORMS = AnalysisModuleTransformSpec(
    pre_setup=dict(models=get_best_model_replicate),
    # `get_best_model_replicate` leaves in the singleton replicate axis -- so, remove it 
    post_eval=dict(
        states=lambda states: jt.map(lambda x: x[0], states),
        models=partial(take_replicate, 0),
    )
)


ss_rnn_funcs = Data.models(where=lambda model: get_ss_rnn_func(model.step.net.hidden))
sisu = Data.hps(where=lambda hps: hps.sisu, is_leaf=is_type(TreeNamespace))
positions = Data.tasks(where=lambda task: (
    task.validation_trials.targets["mechanics.effector.pos"].value[:, -1]
))
steady_state_fps = Transformed(
    "steady_state_fp_results",
    lambda all_fp_results: jt.map(
        lambda fp_results: fp_results.fps,
        all_fp_results,
        is_leaf=is_type(FPFilteredResults),
    ),
)


# Model PyTree structure: ['sisu', 'train__pert__std']
# State batch shape: (eval, replicate, condition)
DEPENDENCIES = {
    "states_pca": (
        StatesPCA(n_components=N_PCA, where_states=lambda states: states.net.hidden)
        .after_indexing(-2, np.arange(START_STEP, END_STEP), axis_label="timestep")
    ),
    "steady_state_fp_results": (
        FixedPoints(
            inputs=FixedPoints.Ports(
                funcs=ss_rnn_funcs,
                func_args=ExpandTo.map(  
                    "funcs", 
                    (sisu, positions), 
                    is_leaf_prefix=(is_type(TreeNamespace), is_type(AbstractTask)),
                ),
                # We can reshape_candidates here since get_best_replicate is applied globally
                candidates=Data.states(
                    # FP initial conditions <- full hidden state trajectories
                    where=lambda states: reshape_candidates(states.net.hidden),
                ),
            ),
        )
        # over the steady state workspace positions & corresponding candidates
        .vmap(in_axes={'func_args': (None, 0), 'candidates': 0})
        .then_transform_result(process_fps)
    ),
    "steady_state_jacobians": (
        Jacobians(
            inputs=Jacobians.Ports(
                funcs=ss_rnn_funcs,  
                func_args=ExpandTo.map(
                    "funcs",  # signature: (sisu, pos, h)
                    (sisu, positions, steady_state_fps),
                    is_leaf=is_module,
                    is_leaf_prefix=(is_type(TreeNamespace), is_module, None),
                ),
            ),
        )
        # over the steady state workspace positions
        .vmap(in_axes={'func_args': (None, None, None)})
    ),
    "steady_state_hessians": (
        Hessians(
            diag_only=True,  # Only xx & uu, no xu & ux
            inputs=Hessians.Ports(
                funcs=ss_rnn_funcs,  
                func_args=ExpandTo.map(
                    "funcs",  # signature: (sisu, pos, h)
                    (sisu, positions, steady_state_fps),
                    is_leaf=is_module,
                    is_leaf_prefix=(is_type(TreeNamespace), is_module, None),
                ),
            ),
        )
        # over the steady state workspace positions
        .vmap(in_axes={'func_args': (None, 0, None)})
    ),
}




# State PyTree structure: ['sisu', 'train__pert__std']
# Array batch shape: (evals, replicates, reach conditions)
ANALYSES = {
    # "fps_in_pc_space": (
    #     FPsInPCSpace(
    #         custom_inputs=dict(
    #             fps_results="steady_state_fps",
    #             pca_results="states_pca",
                
    #         )
    #     )
    # ),
    "temporary": _DummyAnalysis()
}


#! Get readout weights in PC space, for plotting
# all_readout_weights = exclude_nan_replicates(jt.map(
#     lambda model: model.step.net.readout.weight,
#     all_models,
#     is_leaf=is_module,
# ))

# all_readout_weights_pc = batch_transform(all_readout_weights)


"""
If we decide to aggregate over multiple replicates rather than taking only the best one,
it will be necessary to exclude the NaN replicates prior to performing sklearn PCA.
The following are the original functions I used for that purpose.
"""
def exclude_nan_replicates(tree, replicate_info, exclude_underperformers_by='best_total_loss'):
    return jt.map(
        take_replicate,
        replicate_info['included_replicates'][exclude_underperformers_by],
        tree,
    )


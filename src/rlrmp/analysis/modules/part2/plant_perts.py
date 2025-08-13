from functools import partial 

import jax.tree as jt

from feedbax.intervene import add_intervenors, schedule_intervenor
from jax_cookbook import is_module
import jax_cookbook.tree as jtree

from rlrmp.analysis.aligned import ALL_MEASURES, MEASURE_LABELS, VAR_LEVEL_LABEL, AlignedEffectorTrajectories, AlignedVars
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from rlrmp.analysis.measures import ApplyFuncs, Violins
from rlrmp.analysis.profiles import Profiles
from rlrmp.analysis.state_utils import get_best_replicate, get_constant_task_input_fn, vmap_eval_ensemble
from rlrmp.colors import ColorscaleSpec
from rlrmp.plot import set_axes_bounds_equal
from rlrmp.tree_utils import ldict_level_to_bottom, move_ldict_level_above, subdict
from rlrmp.types import (
    LDict,
)


COLOR_FUNCS = dict(
    sisu=ColorscaleSpec(
        sequence_func=lambda hps: hps.sisu,
        colorscale="thermal",
    ),
)


eval_func = vmap_eval_ensemble


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.type}")
    
    pert_amps = hps.pert.amp
    
    # Tasks with varying plant perturbation amplitude 
    tasks_by_amp, _ = jtree.unzip(jt.map( # over disturbance amplitudes
        lambda pert_amp: schedule_intervenor(  # (implicitly) over train stds
            task_base, jt.leaves(models_base, is_leaf=is_module)[0],
            lambda model: model.step.mechanics,
            disturbance(pert_amp),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        ),
        LDict.of("pert__amp")(
            dict(zip(pert_amps, pert_amps)),
        )
    ))
    
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
    tasks = LDict.of('sisu')({
        sisu: jt.map(
            lambda task: task.add_input(
                name="sisu",
                input_fn=get_constant_task_input_fn(
                    sisu, 
                    hps.model.n_steps - 1, 
                    task.n_validation_trials,
                ),
            ),
            tasks_by_amp,
            is_leaf=is_module,
        )
        for sisu in hps.sisu
    })
    
    # The outer levels of `models` have to match those of `tasks`
    models, hps = jtree.unzip(jt.map(
        lambda _: (models_by_std, hps), tasks, is_leaf=is_module
    ))
    
    return tasks, models, hps, None


MEASURE_KEYS = (
    "max_parallel_vel_forward",
    # "max_lateral_vel_signed",
    # "max_lateral_vel_left",
    # "max_lateral_vel_right",  # -2
    "largest_lateral_distance",
    # "max_lateral_distance_left",
    # "sum_lateral_distance",
    "sum_lateral_distance_abs",
    "end_position_error",
    # "end_velocity_error",  # -1
    "max_parallel_force_forward",
    # "sum_parallel_force",  # -2
    # "max_lateral_force_right",  # -1
    "sum_lateral_force_abs",
    "max_net_force",
    "sum_net_force",
)

MEASURE_FUNCS = subdict(ALL_MEASURES, MEASURE_KEYS)

DEPENDENCIES = {
    "measures": (
        ApplyFuncs(
            funcs=MEASURE_FUNCS,
            inputs=ApplyFuncs.Ports(input=AlignedVars()),
            is_leaf=LDict.is_of(VAR_LEVEL_LABEL),
        )
        # Discard the varset; only keep the aligned vars
        .after_transform(lambda result, varset: result, dependency_names="input")
    )
}


def measure_violin_params_fn(fig_params, i, item):
    return fig_params | dict(
        yaxis_title=MEASURE_LABELS[item],
    )

measures_base = (
     Violins(inputs=Violins.Ports(input="measures"))
    .map_figs_at_level(
        "measure", 
        dependency_name="input", 
        fig_params_fn=measure_violin_params_fn,
    )
)


# State PyTree structure: ['sisu', 'pert__amp', 'train__pert__std']
ANALYSES = {
    # "effector_trajectories_by_condition": (
    #     # By condition, all evals for the best replicate only
    #     EffectorTrajectories(
    #         colorscale_axis=1, 
    #         colorscale_key="reach_condition",
    #     )
    #     .after_transform(get_best_replicate)  # By default has `axis=1` for replicates
    # ),
    "aligned_trajectories_by_sisu": (
        AlignedEffectorTrajectories(colorscale_key="sisu")
        .after_stacking('sisu')
        .map_figs_at_level("train__pert__std")
        .then_transform_figs(
            partial(
                set_axes_bounds_equal, 
                padding_factor=0.1,
                trace_selector=lambda trace: trace.showlegend is True,
            ),
        )
    ),
    "aligned_trajectories_by_train_std": (
        AlignedEffectorTrajectories(colorscale_key="train__pert__std")
        .after_stacking("train__pert__std")
        .map_figs_at_level('sisu')
    ),
    "profiles_by_train_std": (
        Profiles()
        .after_transform(get_best_replicate)
        .after_transform(
            lambda tree, **kws: ldict_level_to_bottom("train__pert__std", tree),
            dependency_names="vars",
        )
    ),
    "profiles_by_sisu": (
        Profiles()
        .after_transform(get_best_replicate)
        .after_transform(
            lambda tree, **kws: ldict_level_to_bottom("sisu", tree),
            dependency_names="vars",
        )
    ),
    # "measures_by_pert_amp": Measures(measure_keys=MEASURE_KEYS).map_figs_at_level("pert__amp"),
    # "measures_by_train_std": Measures(measure_keys=MEASURE_KEYS).map_figs_at_level("train__pert__std"),
    # "measures_train_std_by_pert_amp": Measures(measure_keys=MEASURE_KEYS).after_level_to_top("train__pert__std").map_figs_at_level("pert__amp"),
    "plot--measures_by_pert_amp": (
        Violins(inputs=Violins.Ports(input="measures"))
        .map_figs_at_level("pert__amp")
    ),
    "plot--measures_by_train_std": (
        Violins(inputs=Violins.Ports(input="measures"))
        .map_figs_at_level("train__pert__std")
    ),
    "plot--measures_train_std_by_pert_amp": (
        Violins(inputs=Violins.Ports(input="measures"))
        .after_level_to_top("train__pert__std")
        .map_figs_at_level("pert__amp")
    ),
}
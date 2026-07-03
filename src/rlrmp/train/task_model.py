"""Task/model construction for rlrmp training and checkpoint reload paths."""

import warnings
from collections.abc import Callable, Mapping
from typing import Literal as L
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax import AbstractTask, TaskTrialSpec, WhereDict
from feedbax.intervene import (
    CurlFieldParams,
    DynamicsMatrixPerturbParams,
    FixedFieldParams,
    schedule_intervenor,
)
from rlrmp.misc import get_field_amplitude, vector_with_gaussian_length
from feedbax.models.networks import PopulationStructure
from feedbax.runtime.state import CartesianState
from feedbax.training.train import always_active, bernoulli_active
from feedbax.config.namespace import TreeNamespace
from feedbax.training.types import TaskModelPair
from jax_cookbook import LDict
from jaxtyping import PRNGKeyArray

from rlrmp.analysis.math.cs_game_card import build_canonical_game, build_no_integrator_game
from rlrmp.analysis.math.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
)
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig, process_covariance
from rlrmp.model.cs_lss_gru import (
    CS_DEFAULT_TRAINABLE_DTYPE,
    CS_FEEDBACK_DIM,
    CS_H0_CONTEXT_INPUT,
    CS_PHYSICAL_STATE_DIM,
    CS_PROPRIOCEPTIVE_FEEDBACK_DIM,
    CS_REDUCED_PHYSICAL_STATE_DIM,
    build_cs_lss_gru_graph,
)
from rlrmp.train.closed_loop_finite_adversary import (
    AFFINE_POLICY,
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
)
from rlrmp.disturbance import (
    PLANT_INTERVENOR_LABEL,
    get_gusts_fn,
)
from rlrmp.loss import get_reach_loss
from rlrmp.model import (
    LINEAR_HIDDEN_TYPES,
    create_point_mass_linear_ensemble,
    create_point_mass_nn_ensemble,
)
from rlrmp.model.feedbax_graph import POINT_MASS_TARGET_POSITION_INPUT
from rlrmp.task import TASK_TYPES
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_FINITE_POLICY_MECHANISMS,
    BroadFullStateEpsilonTrainingTaskAdapter,
    FixedTargetPerturbationTrainingTaskAdapter,
    TargetRelativeMultiTargetTrainingTaskAdapter,
    config_from_broad_epsilon_pgd_hps,
    config_from_broad_epsilon_hps,
    config_from_hps,
    config_from_target_hps,
    install_perturbation_training_graph_adapters,
)

TrainingMethodLabel: TypeAlias = L["bcs", "dai", "pai-asf", "pai-n", "nominal-cs-gru"]
PlantBackendLabel: TypeAlias = L["cs_lss", "legacy_causal_simplefeedback"]

CS_LSS_PLANT_BACKEND = "cs_lss"
LEGACY_CAUSAL_PLANT_BACKEND = "legacy_causal_simplefeedback"
LEGACY_CAUSAL_BACKEND_WARNING = (
    "Using legacy causal SimpleFeedback backend for nominal-cs-gru. This backend "
    "splits the C&S plant into a point-mass mechanics node plus a first-order "
    "force filter, which creates the known same-step force-filter-to-mechanics "
    "timing problem. Use plant_backend='cs_lss' for the exact C&S LinearStateSpace "
    "plant."
)


P_PERTURBED = LDict.of("train__method")(
    {
        "nominal-cs-gru": 0.0,
        "bcs": 0.5,
        "dai": 1.0,
        "pai-asf": 1.0,
    }
)

# Define whether the disturbance is active on each trial
disturbance_active: LDict[str, Callable] = LDict.of("train__method")(
    {
        "nominal-cs-gru": always_active,
        "bcs": bernoulli_active,
        "dai": bernoulli_active,  # or always_active?
        "pai-asf": always_active,  # or bernoulli_active? and let hps control it
    }
)


def scaled_sampler(sample_fn, scale=1.0):
    def _fn(trial_spec, batch_info, key):
        return scale * sample_fn(key)

    return _fn


# Separate this def by training method so that we can multiply by `field_std` in the "pai-asf" case,
# without it affecting the SISU. That is, in all three cases `field_std` is a factor of
# the actual field strength, but in `"bcs"` and `"dai"` it is multiplied by the
# `scale` parameter, which is not seen by the network in those cases; and in `"pai-asf"` it is
# multiplied by the `field` parameter, which is not seen by the network in that case.
# (See the definition of `SCALE_FNS` below.)
disturbance_extra_params = LDict.of("train__method")(
    {
        "nominal-cs-gru": {
            "gusts": get_gusts_fn,
            "constant": lambda hps: dict(field=scaled_sampler(vector_with_gaussian_length)),
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal)),
        },
        "bcs": {
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal)),
            "constant": lambda hps: dict(field=scaled_sampler(vector_with_gaussian_length)),
        },
        "dai": {
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal)),
            "constant": lambda hps: dict(field=scaled_sampler(vector_with_gaussian_length)),
        },
        "pai-asf": {
            "curl": lambda hps: dict(amplitude=scaled_sampler(jr.normal, hps.pert.std)),
            "constant": lambda hps: dict(
                field=scaled_sampler(vector_with_gaussian_length, hps.pert.std)
            ),
            "gusts": get_gusts_fn,
            "dynamics_matrix": lambda hps: {},
        },
    }
)


# Define how the network's SISU will be determined from the trial specs, to which it is then added
SISU_FNS = LDict.of("train__method")(
    {
        "nominal-cs-gru": lambda trial_specs, key: jnp.zeros(
            (
                (trial_specs.timeline.epoch_bounds.shape[0], trial_specs.timeline.n_steps)
                if trial_specs.timeline.epoch_bounds is not None
                and trial_specs.timeline.epoch_bounds.ndim > 1
                else (trial_specs.timeline.n_steps,)
            ),
            dtype=jnp.float32,
        ),
        "bcs": lambda trial_specs, key: trial_specs.intervene[PLANT_INTERVENOR_LABEL].active.astype(
            jnp.float32
        ),
        "dai": lambda trial_specs, key: jnp.asarray(
            get_field_amplitude(trial_specs.intervene[PLANT_INTERVENOR_LABEL]),
            dtype=jnp.float32,
        ),
        "pai-asf": lambda trial_specs, key: jnp.asarray(
            trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale,
            dtype=jnp.float32,
        ),
    }
)


"""Either scale the field strength by a constant std, or sample the std for each trial.

Note that in the `"pai-asf"` case the actual field amplitude is still scaled by `field_std`,
but this is done in `disturbance_extra_params` so that the magnitude of the SISU
is the same on average between the `"dai"` and `"pai-asf"` methods.
"""
SCALE_FNS = LDict.of("train__method")(
    {
        "nominal-cs-gru": lambda field_std: field_std,
        "bcs": lambda field_std: field_std,
        "dai": lambda field_std: field_std,
        "pai-asf": lambda field_std: (
            lambda trial_spec, _, key: jr.uniform(key, (), minval=0, maxval=1)
        ),
    }
)


def get_disturbance_params(hps: TreeNamespace):
    """Build disturbance params for the given hyperparameters.

    Args:
        hps: Hyperparameters including method, pert.type, pert.std

    Returns:
        Appropriate params object (CurlFieldParams, FixedFieldParams, etc.)
    """
    pert_type = hps.pert.type
    method = hps.method

    extra_params = disturbance_extra_params[method][pert_type](hps)
    scale = SCALE_FNS[method](hps.pert.std)
    active = disturbance_active[method](P_PERTURBED[method])

    if pert_type == "curl":
        return _scheduled_intervention_params(
            CurlFieldParams,
            scale=scale,
            active=active,
            **extra_params,
        )
    elif pert_type == "constant":
        return _scheduled_intervention_params(
            FixedFieldParams,
            scale=scale,
            active=active,
            **extra_params,
        )
    elif pert_type == "gusts":
        return _scheduled_intervention_params(
            FixedFieldParams,
            scale=scale,
            active=active,
            **extra_params,
        )
    elif pert_type == "dynamics_matrix":
        return _scheduled_intervention_params(
            DynamicsMatrixPerturbParams,
            scale=scale,
            active=False,
            delta_A=jnp.zeros((2, 4), dtype=jnp.float32),
            **extra_params,
        )
    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")


def _scheduled_intervention_params(params_type, **values):
    """Construct intervention params while preserving callable schedules."""

    placeholders = {
        "scale": 1.0,
        "active": False,
        "amplitude": 1.0,
    }
    init_values = {
        key: placeholders[key] if key in placeholders and callable(value) else value
        for key, value in values.items()
    }
    params = params_type(**init_values)
    for key, value in values.items():
        if callable(value):
            object.__setattr__(params, key, value)
    return params


def build_task_base(hps: TreeNamespace):
    """Build the unscheduled base task used for main point-mass model sizing."""

    task_type = hps.task.type
    hps_task = {k: v for k, v in hps.task.omitting_attrs("eval_n", "type").items() if v is not None}
    if task_type in {"simple_reach", "fixed_simple_reach"}:
        delayed_only_keys = {
            "epoch_len_ranges",
            "target_on_epochs",
            "hold_epochs",
            "move_epochs",
            "p_catch_trial",
            "train_endpoint_mode",
            "preset",
            "n_control_stages",
            "target_visible_from_start",
            "go_cue_event_name",
            "catch_metadata_policy",
        }
        hps_task = {k: v for k, v in hps_task.items() if k not in delayed_only_keys}
    return TASK_TYPES[task_type](loss_func=get_reach_loss(hps), **hps_task)


def setup_task_model_pair(
    hps: TreeNamespace = TreeNamespace(),
    *,
    key: PRNGKeyArray,
    **kwargs,
):
    """Returns a skeleton PyTree for reloading trained models."""
    hps = hps | kwargs

    # TODO: Implement scale-up for this experiment
    scaleup_batches = hps.intervention_scaleup_batches
    n_batches_scaleup = scaleup_batches[1] - scaleup_batches[0]
    if n_batches_scaleup > 0:

        def batch_scale_up(batch_start, n_batches, batch_info, x):
            progress = jax.nn.relu(batch_info.current - batch_start) / n_batches
            progress = jnp.minimum(progress, 1.0)
            scale = 0.5 * (1 - jnp.cos(progress * jnp.pi))
            return x * scale
    else:

        def batch_scale_up(batch_start, n_batches, batch_info, x):
            return x

    task_base = build_task_base(hps)

    # Resolve hidden_type from hps if present; default (None) falls back to GRUCell
    hidden_type = getattr(hps, "hidden_type", None)
    # Resolve SISU gating mode; default "additive" preserves existing behavior
    sisu_gating = getattr(hps, "sisu_gating", "additive")
    plant_backend = getattr(hps.model, "plant_backend", LEGACY_CAUSAL_PLANT_BACKEND)

    if hps.method == "nominal-cs-gru" and plant_backend == CS_LSS_PLANT_BACKEND:
        if isinstance(hidden_type, str) and hidden_type in LINEAR_HIDDEN_TYPES:
            raise ValueError("The C&S LSS nominal GRU backend requires a recurrent cell type.")
        target_training = config_from_target_hps(
            getattr(hps, "target_relative_multitarget", TreeNamespace(enabled=False))
        )
        no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
        physical_state_dim = (
            CS_REDUCED_PHYSICAL_STATE_DIM if no_integrator_state else CS_PHYSICAL_STATE_DIM
        )
        runtime_dtype = jnp.dtype(
            getattr(hps.model, "trainable_dtype", None) or CS_DEFAULT_TRAINABLE_DTYPE
        )
        delayed_reach = _cs_delayed_reach_enabled(hps)
        sisu_conditioned_pgd = _sisu_conditioned_pgd_budget_enabled(hps)
        finite_epsilon_policy = _finite_epsilon_policy_mechanism(hps)
        task = _add_cs_lss_task_inputs(
            _CsLssTaskAdapter(
                task_base,
                physical_state_dim=physical_state_dim,
                dtype=runtime_dtype,
            ),
            target_relative=target_training.enabled,
            go_cue_input=target_training.enabled and delayed_reach,
            scalar_input=sisu_conditioned_pgd,
            scalar_input_name=_sisu_conditioned_pgd_budget_input_name(hps),
            scalar_input_fn=_sisu_conditioned_pgd_budget_input_fn(hps)
            if sisu_conditioned_pgd
            else None,
            finite_epsilon_policy=finite_epsilon_policy,
            initial_hidden_encoder=bool(getattr(hps.model, "initial_hidden_encoder", False)),
            force_filter_feedback=target_training.force_filter_feedback,
            physical_state_dim=physical_state_dim,
            dtype=runtime_dtype,
        )
        models = _create_cs_lss_gru_ensemble(
            hps,
            hidden_type=hidden_type,
            sisu_gating=sisu_gating,
            key=key,
        )
        if target_training.enabled:
            task = TargetRelativeMultiTargetTrainingTaskAdapter(task, target_training)
        broad_epsilon_training = config_from_broad_epsilon_hps(
            getattr(hps, "broad_epsilon_training", TreeNamespace(enabled=False))
        )
        if broad_epsilon_training.enabled:
            task = BroadFullStateEpsilonTrainingTaskAdapter(task, broad_epsilon_training)
        perturbation_training = config_from_hps(
            getattr(hps, "perturbation_training", TreeNamespace(enabled=False))
        )
        if perturbation_training.enabled:
            models = install_perturbation_training_graph_adapters(
                models,
                force_filter_feedback=target_training.force_filter_feedback,
            )
            task = FixedTargetPerturbationTrainingTaskAdapter(
                task,
                perturbation_training,
            )
        return TaskModelPair(task, models)

    if hps.method == "nominal-cs-gru" and plant_backend == LEGACY_CAUSAL_PLANT_BACKEND:
        warnings.warn(
            LEGACY_CAUSAL_BACKEND_WARNING,
            RuntimeWarning,
            stacklevel=2,
        )
    elif hps.method == "nominal-cs-gru":
        raise ValueError(
            f"Unknown nominal-cs-gru plant_backend {plant_backend!r}; expected "
            f"{CS_LSS_PLANT_BACKEND!r} or {LEGACY_CAUSAL_PLANT_BACKEND!r}."
        )

    # Dispatch: linear-controller MVP variants (Bug: 410d7ac) bypass
    # ``create_point_mass_nn_ensemble`` entirely because they replace
    # ``SimpleStagedNetwork`` with a purpose-built ``Component``. Detected via
    # the sentinel strings ``"linear"`` / ``"linear_tracker"``; for these,
    # hidden_type is a str (not a class), and SISU is still threaded through
    # the task input pipeline (controller ignores it) so n_extra_inputs is 0.
    if isinstance(hidden_type, str) and hidden_type in LINEAR_HIDDEN_TYPES:
        models_base = create_point_mass_linear_ensemble(
            hps,
            task_base,
            controller_type=hidden_type,
            key=key,
        )
    else:
        models_base = create_point_mass_nn_ensemble(
            hps,
            task_base,
            n_extra_inputs=1,  # for SISU (even when multiplicative, task still provides it)
            hidden_type=hidden_type,
            sisu_gating=sisu_gating,
            key=key,
        )

    models = models_base

    # Add SISU input to task
    try:
        task = task_base.add_input(
            name="sisu",
            input_fn=SISU_FNS[hps.method],
        )
    except AttributeError:
        raise ValueError("No training method label assigned to hps_train.method")
    if isinstance(hidden_type, str) and hidden_type in LINEAR_HIDDEN_TYPES:
        task = task.add_input(
            name=POINT_MASS_TARGET_POSITION_INPUT,
            input_fn=_point_mass_target_position_input,
        )

    # Build disturbance params for scheduling
    disturbance_params = get_disturbance_params(hps)

    # Schedule the intervention params on the task
    task, models = schedule_intervenor(
        task,
        models,
        label=PLANT_INTERVENOR_LABEL,
        intervenor_params=disturbance_params,
        default_active=False,
    )

    return TaskModelPair(task, models)


class _CsLssTaskAdapter(AbstractTask):
    """Rewrite SimpleReaches initial state for a C&S LSS graph.

    Feedbax ``SimpleReaches`` initializes ``mechanics.effector`` because the
    legacy point-mass mechanics retains a ``MechanicsState``. The C&S LSS graph
    retains ``mechanics.vector`` and exposes ``mechanics.effector`` only as a
    semantic state view. This adapter leaves targets, inputs, timelines, and
    loss functions unchanged while translating the initial Cartesian effector
    state into the 48D LSS vector.
    """

    task: object
    physical_state_dim: int = eqx.field(default=CS_PHYSICAL_STATE_DIM, static=True)
    dtype: jnp.dtype = eqx.field(default=jnp.dtype(CS_DEFAULT_TRAINABLE_DTYPE), static=True)

    def __getattr__(self, name: str):
        return getattr(self.task, name)

    @property
    def loss_func(self):
        return self.task.loss_func

    @property
    def n_steps(self) -> int:
        return int(self.task.n_steps)

    @property
    def seed_validation(self) -> int:
        return int(self.task.seed_validation)

    @property
    def intervention_specs(self):
        return self.task.intervention_specs

    @property
    def input_dependencies(self):
        return self.task.input_dependencies

    def add_input(
        self,
        name: str,
        input_fn: Callable[[TaskTrialSpec, PRNGKeyArray], object],
        exist_ok: bool = True,
    ):
        return _CsLssTaskAdapter(
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
            physical_state_dim=self.physical_state_dim,
            dtype=self.dtype,
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self._rewrite_trial(self.task.get_train_trial(key, batch_info))

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        return self._rewrite_trial(
            self.task.get_train_trial_with_intervenor_params(key, batch_info)
        )

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        return self._rewrite_trial(self.task.get_validation_trials(key))

    @property
    def n_validation_trials(self) -> int:
        return int(self.task.n_validation_trials)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return self._rewrite_trial(self.task.validation_trials)

    def _rewrite_trial(self, trial_spec: TaskTrialSpec) -> TaskTrialSpec:
        effector_init = trial_spec.inits["mechanics.effector"]
        lss_vector = _effector_init_to_lss_vector(
            effector_init,
            physical_state_dim=self.physical_state_dim,
            dtype=self.dtype,
        )
        inputs = _broadcast_finite_policy_inputs_to_lss_batch(
            trial_spec.inputs,
            batch_shape=lss_vector.shape[:-1],
        )
        return TaskTrialSpec(
            inits=WhereDict({"mechanics.vector": lss_vector}),
            inputs=inputs,
            targets=trial_spec.targets,
            intervene=trial_spec.intervene,
            extra=trial_spec.extra,
            timeline=trial_spec.timeline,
        )


def _effector_init_to_lss_vector(
    effector_init: CartesianState,
    *,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
    dtype=jnp.float32,
) -> jax.Array:
    pos = jnp.asarray(effector_init.pos, dtype=dtype)
    vel = jnp.asarray(effector_init.vel, dtype=dtype)
    force = jnp.asarray(effector_init.force, dtype=dtype)
    if int(physical_state_dim) < CS_REDUCED_PHYSICAL_STATE_DIM:
        raise ValueError(f"physical_state_dim must be >= 6; got {physical_state_dim}.")
    batch_shape = jnp.broadcast_shapes(pos.shape[:-1], vel.shape[:-1], force.shape[:-1])
    vector = jnp.zeros(
        (*batch_shape, 6 * int(physical_state_dim)),
        dtype=dtype,
    )
    return (
        vector.at[..., 0:2]
        .set(jnp.broadcast_to(pos, (*batch_shape, 2)))
        .at[..., 2:4]
        .set(jnp.broadcast_to(vel, (*batch_shape, 2)))
        .at[..., 4:6]
        .set(jnp.broadcast_to(force, (*batch_shape, 2)))
    )


def _add_cs_lss_task_inputs(
    task: _CsLssTaskAdapter,
    *,
    target_relative: bool = False,
    go_cue_input: bool = False,
    scalar_input: bool = False,
    scalar_input_name: str = "input",
    scalar_input_fn: Callable | None = None,
    finite_epsilon_policy: str | None = None,
    initial_hidden_encoder: bool = False,
    force_filter_feedback: bool = False,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
    dtype=jnp.float32,
) -> _CsLssTaskAdapter:
    if go_cue_input and scalar_input:
        scalar_fn = scalar_input_fn or SISU_FNS["nominal-cs-gru"]
        task = task.add_input(
            name=scalar_input_name,
            input_fn=scalar_fn,
        )
        task = task.add_input(
            name="input",
            input_fn=_cs_delayed_go_cue_with_scalar_input(scalar_fn),
        )
    elif go_cue_input:
        task = task.add_input(
            name="input",
            input_fn=_cs_delayed_go_cue_input,
        )
    elif scalar_input:
        task = task.add_input(
            name=scalar_input_name,
            input_fn=scalar_input_fn or SISU_FNS["nominal-cs-gru"],
        )
    elif not go_cue_input and not target_relative:
        task = task.add_input(
            name="input",
            input_fn=SISU_FNS["nominal-cs-gru"],
        )
    task = task.add_input(
        name="epsilon",
        input_fn=lambda trial_spec, key: _sample_cs_lss_process_epsilon(
            trial_spec,
            key,
            physical_state_dim=physical_state_dim,
            dtype=dtype,
        ),
    )
    if initial_hidden_encoder:
        context_dim = (
            CS_PROPRIOCEPTIVE_FEEDBACK_DIM if force_filter_feedback else CS_FEEDBACK_DIM
        )
        task = task.add_input(
            name=CS_H0_CONTEXT_INPUT,
            input_fn=lambda trial_spec, key: _zero_h0_context_input(
                trial_spec,
                context_dim=context_dim,
                dtype=dtype,
            ),
        )
    if finite_epsilon_policy is None:
        return task
    state_dim = 6 * int(physical_state_dim)
    task = task.add_input(
        name=FINITE_POLICY_GAINS_INPUT,
        input_fn=lambda trial_spec, key: _zero_finite_policy_gains_input(
            trial_spec,
            epsilon_dim=int(physical_state_dim),
            feature_dim=state_dim,
            dtype=dtype,
        ),
    )
    if str(finite_epsilon_policy) == AFFINE_POLICY:
        task = task.add_input(
            name=FINITE_POLICY_BIAS_INPUT,
            input_fn=lambda trial_spec, key: _zero_finite_policy_bias_input(
                trial_spec,
                epsilon_dim=int(physical_state_dim),
                dtype=dtype,
            ),
        )
    return task


def _zero_h0_context_input(
    trial_spec: TaskTrialSpec,
    *,
    context_dim: int,
    dtype=jnp.float32,
) -> jax.Array:
    target = trial_spec.targets["mechanics.effector.pos"].value
    batch_shape = target.shape[:-2] if target.ndim >= 3 else ()
    return jnp.zeros((*batch_shape, int(context_dim)), dtype=dtype)


def _cs_delayed_reach_enabled(hps: TreeNamespace) -> bool:
    delayed_contract = getattr(hps, "delayed_reach", TreeNamespace())
    if bool(getattr(delayed_contract, "enabled", False)):
        return True
    task = getattr(hps, "task", TreeNamespace())
    task_type = str(getattr(task, "type", ""))
    return task_type == "cs_delayed_center_out_reach" or (
        task_type == "delayed_reach" and str(getattr(task, "preset", "")) == "delayed_center_out"
    )


def _sisu_conditioned_pgd_budget_enabled(hps: TreeNamespace) -> bool:
    config = _sisu_conditioned_pgd_budget_config(hps)
    return config is not None


def _sisu_conditioned_pgd_budget_config(hps: TreeNamespace) -> TreeNamespace | None:
    config = getattr(hps, "broad_epsilon_pgd_training", TreeNamespace(enabled=False))
    if not bool(getattr(config, "enabled", False)):
        return None
    schedule = getattr(config, "budget_schedule", None)
    mode = getattr(schedule, "mode", schedule)
    if str(mode) != "sisu_energy_fraction":
        return None
    return config


def _sisu_conditioned_pgd_budget_input_fn(hps: TreeNamespace) -> Callable | None:
    config = _sisu_conditioned_pgd_budget_config(hps)
    if config is None:
        return None
    schedule = config.budget_schedule
    levels = tuple(float(value) for value in schedule.levels)
    probabilities = tuple(float(value) for value in schedule.probabilities)

    def input_fn(trial_spec, key):
        epoch_bounds = trial_spec.timeline.epoch_bounds
        n_steps = int(trial_spec.timeline.n_steps)
        if epoch_bounds is not None and getattr(epoch_bounds, "ndim", 0) > 1:
            batch_shape = (int(epoch_bounds.shape[0]),)
        else:
            batch_shape = ()
        sampled = jr.choice(
            key,
            jnp.asarray(levels, dtype=jnp.float32),
            shape=batch_shape,
            p=jnp.asarray(probabilities, dtype=jnp.float32),
        )
        if batch_shape:
            return jnp.broadcast_to(sampled[:, None], (*batch_shape, n_steps))
        return jnp.broadcast_to(sampled, (n_steps,))

    return input_fn


def _sisu_conditioned_pgd_budget_input_name(hps: TreeNamespace) -> str:
    config = _sisu_conditioned_pgd_budget_config(hps)
    if config is None:
        return "input"
    schedule = config.budget_schedule
    conditioning = getattr(schedule, "conditioning_scalar", TreeNamespace())
    input_name = str(
        getattr(conditioning, "input_key", getattr(config, "sisu_condition_input", "auto"))
    )
    if input_name == "auto":
        return "sisu" if _cs_delayed_reach_enabled(hps) else "input"
    return input_name


def _cs_delayed_go_cue_input(
    trial_spec: TaskTrialSpec,
    key: PRNGKeyArray,
) -> jax.Array:
    """Return a scalar go cue where 0 is prep/hold and 1 is movement."""

    del key
    inputs = trial_spec.inputs
    task_inputs = inputs.get("task", inputs) if isinstance(inputs, Mapping) else inputs
    if not hasattr(task_inputs, "hold"):
        raise ValueError("Delayed C&S go-cue input requires task inputs with a hold signal.")
    hold = jnp.asarray(task_inputs.hold, dtype=jnp.float32)
    go = 1.0 - hold
    return go[..., 0] if go.ndim > 0 and go.shape[-1] == 1 else go


def _point_mass_target_position_input(
    trial_spec: TaskTrialSpec,
    key: PRNGKeyArray,
) -> jax.Array:
    """Expose the native Feedbax task-data binding for point-mass target position."""

    del key
    inputs = trial_spec.inputs
    task_inputs = inputs.get("effector_target") if isinstance(inputs, Mapping) else inputs
    if task_inputs is None or not hasattr(task_inputs, "pos"):
        raise ValueError("Point-mass target-position binding requires inputs.effector_target.pos")
    return jnp.asarray(task_inputs.pos)


def _cs_delayed_go_cue_with_scalar_input(scalar_input_fn: Callable) -> Callable:
    def input_fn(trial_spec: TaskTrialSpec, key: PRNGKeyArray) -> jax.Array:
        go = jnp.asarray(_cs_delayed_go_cue_input(trial_spec, key), dtype=jnp.float32)
        scalar = jnp.asarray(scalar_input_fn(trial_spec, key), dtype=go.dtype)
        if scalar.ndim >= 1 and scalar.shape[-1] == go.shape[-1] + 1:
            scalar = scalar[..., :-1]
        if scalar.shape != go.shape:
            scalar = jnp.broadcast_to(scalar, go.shape)
        return jnp.stack([go, scalar], axis=-1)

    return input_fn


def _sample_cs_lss_process_epsilon(
    trial_spec: TaskTrialSpec,
    key: PRNGKeyArray,
    *,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
    dtype=jnp.float32,
) -> jax.Array:
    """Sample the temporary physical-process epsilon bridge for C&S LSS GRUs.

    ``LinearStateSpace.B_w`` injects a physical epsilon into the current
    physical block. Sensory and motor noise are handled by the LSS graph's
    causal ``Channel`` nodes; this function owns only the process/load noise
    source.
    """

    target = trial_spec.targets["mechanics.effector.pos"].value
    batch_shape = target.shape[:-2] if target.ndim >= 3 else ()
    n_steps = int(target.shape[-2])
    factor = _cs_lss_process_epsilon_factor(physical_state_dim=physical_state_dim, dtype=dtype)
    epsilon_dim = int(factor.shape[0])
    draws = jr.normal(key, (*batch_shape, n_steps, epsilon_dim), dtype=dtype)
    return draws @ factor.T


def _zero_finite_policy_gains_input(
    trial_spec: TaskTrialSpec,
    *,
    epsilon_dim: int,
    feature_dim: int,
    dtype=jnp.float32,
) -> jax.Array:
    target = trial_spec.targets["mechanics.effector.pos"].value
    batch_shape = _trial_batch_shape_from_init(trial_spec)
    n_steps = int(target.shape[-2])
    return jnp.zeros((*batch_shape, n_steps, int(epsilon_dim), int(feature_dim)), dtype=dtype)


def _zero_finite_policy_bias_input(
    trial_spec: TaskTrialSpec,
    *,
    epsilon_dim: int,
    dtype=jnp.float32,
) -> jax.Array:
    target = trial_spec.targets["mechanics.effector.pos"].value
    batch_shape = _trial_batch_shape_from_init(trial_spec)
    n_steps = int(target.shape[-2])
    return jnp.zeros((*batch_shape, n_steps, int(epsilon_dim)), dtype=dtype)


def _trial_batch_shape_from_init(trial_spec: TaskTrialSpec) -> tuple[int, ...]:
    if "mechanics.vector" in trial_spec.inits:
        return jnp.asarray(trial_spec.inits["mechanics.vector"]).shape[:-1]
    effector = trial_spec.inits["mechanics.effector"]
    return jnp.asarray(effector.pos).shape[:-1]


def _broadcast_finite_policy_inputs_to_lss_batch(
    inputs: Mapping,
    *,
    batch_shape: tuple[int, ...],
) -> Mapping:
    if not batch_shape or FINITE_POLICY_GAINS_INPUT not in inputs:
        return inputs
    updated = dict(inputs)
    for name in (FINITE_POLICY_GAINS_INPUT, FINITE_POLICY_BIAS_INPUT):
        if name not in updated:
            continue
        values = jnp.asarray(updated[name])
        suffix_rank = 3 if name == FINITE_POLICY_GAINS_INPUT else 2
        suffix = values.shape[-suffix_rank:]
        prefix = values.shape[:-suffix_rank]
        if prefix == batch_shape:
            continue
        if prefix == ():
            updated[name] = jnp.broadcast_to(values, (*batch_shape, *suffix))
            continue
        try:
            updated[name] = jnp.broadcast_to(values, (*batch_shape, *suffix))
        except ValueError as exc:
            raise ValueError(
                f"Cannot broadcast finite policy input {name!r} from batch prefix "
                f"{prefix} to C&S LSS batch shape {batch_shape}."
            ) from exc
    return WhereDict(updated) if isinstance(inputs, WhereDict) else updated


def _cs_lss_process_epsilon_factor(
    *,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
    dtype=jnp.float32,
) -> jax.Array:
    """Return a square-root factor for the physical process epsilon covariance."""

    if int(physical_state_dim) == CS_REDUCED_PHYSICAL_STATE_DIM:
        plant, _schedule = build_no_integrator_game()
    elif int(physical_state_dim) == CS_PHYSICAL_STATE_DIM:
        plant, _schedule = build_canonical_game()
    else:
        raise ValueError(f"Unsupported C&S LSS physical_state_dim {physical_state_dim}.")
    process_cov = process_covariance(
        plant,
        OutputFeedbackConfig(n_phys=int(physical_state_dim)),
    ) * jnp.asarray(
        DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.process_covariance_scale,
        dtype=jnp.float64,
    )
    physical_cov = process_cov[: int(physical_state_dim), : int(physical_state_dim)]
    eigvals, eigvecs = jnp.linalg.eigh(0.5 * (physical_cov + physical_cov.T))
    factor = eigvecs @ jnp.diag(jnp.sqrt(jnp.clip(eigvals, min=0.0)))
    return factor.astype(dtype)


def _create_cs_lss_gru_ensemble(
    hps: TreeNamespace,
    *,
    hidden_type,
    sisu_gating: str,
    key: PRNGKeyArray,
):
    if hidden_type is None:
        hidden_type = eqx.nn.GRUCell
    pop_config = hps.model.population_structure
    key_pop, key_models = jr.split(key)
    population_structure = PopulationStructure.create(
        hidden_size=hps.model.hidden_size,
        n_input_only=getattr(pop_config, "n_input_only", 0) or 0,
        n_readout_only=getattr(pop_config, "n_readout_only", 0) or 0,
        n_recurrent_only=getattr(pop_config, "n_recurrent_only", 0) or 0,
        n_input_readout=getattr(pop_config, "n_input_readout", 0) or 0,
        assignment_fn=None,
        key=key_pop,
    )
    keys = jr.split(key_models, int(hps.model.n_replicates))

    def build_one(key_one):
        target_training = config_from_target_hps(
            getattr(hps, "target_relative_multitarget", TreeNamespace(enabled=False))
        )
        no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
        finite_epsilon_policy = _finite_epsilon_policy_mechanism(hps)
        delayed_reach = _cs_delayed_reach_enabled(hps)
        scalar_input_count = int(delayed_reach or not target_training.enabled)
        if _sisu_conditioned_pgd_budget_enabled(hps):
            sisu_input_name = _sisu_conditioned_pgd_budget_input_name(hps)
            if not (sisu_input_name == "input" and scalar_input_count > 0):
                scalar_input_count += 1
        return build_cs_lss_gru_graph(
            hidden_size=int(hps.model.hidden_size),
            input_size=scalar_input_count,
            hidden_type=hidden_type,
            population_structure=population_structure,
            sisu_gating=sisu_gating,
            sensory_noise_std=float(hps.model.sensory_noise_std),
            additive_motor_noise_std=float(hps.model.additive_motor_noise_std),
            signal_dependent_motor_noise_std=float(hps.model.signal_dependent_motor_noise_std),
            bind_epsilon_input=True,
            finite_epsilon_policy=finite_epsilon_policy,
            target_relative_feedback=target_training.enabled,
            force_filter_feedback=target_training.force_filter_feedback,
            initial_hidden_encoder=bool(getattr(hps.model, "initial_hidden_encoder", False)),
            no_integrator_state=no_integrator_state,
            trainable_dtype=getattr(hps.model, "trainable_dtype", None),
            population_mask_mode=getattr(hps.model, "population_mask_mode", None),
            key=key_one,
        )

    def stack_leaves(*leaves):
        first = leaves[0]
        if isinstance(first, eqx.nn.StateIndex):
            init = jax.tree.map(stack_leaves, *(leaf.init for leaf in leaves))
            return eqx.nn.StateIndex(init)
        return jnp.stack(leaves) if all(eqx.is_array(leaf) for leaf in leaves) else first

    models = [build_one(key_one) for key_one in keys]
    return jax.tree.map(
        stack_leaves,
        *models,
        is_leaf=lambda leaf: isinstance(leaf, eqx.nn.StateIndex),
    )


def _finite_epsilon_policy_mechanism(hps: TreeNamespace) -> str | None:
    cfg = config_from_broad_epsilon_pgd_hps(
        getattr(hps, "broad_epsilon_pgd_training", TreeNamespace(enabled=False))
    )
    if not cfg.enabled:
        return None
    mechanism = str(cfg.adversary_mechanism)
    if mechanism in BROAD_EPSILON_PGD_FINITE_POLICY_MECHANISMS:
        return mechanism
    return None

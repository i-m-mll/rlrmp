from collections.abc import Callable
from collections.abc import Mapping
from typing import Literal as L
from typing import TypeAlias
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax._mapping import WhereDict
from feedbax.intervene import (
    CurlFieldParams,
    FixedFieldParams,
    schedule_intervenor,
)
from feedbax.misc import get_field_amplitude, vector_with_gaussian_length
from feedbax.nn import PopulationStructure
from feedbax.state import CartesianState
from feedbax.task import AbstractTask, TaskTrialSpec
from feedbax.training.train import always_active, bernoulli_active
from feedbax.types import LDict, TaskModelPair, TreeNamespace
from jaxtyping import PRNGKeyArray

from rlrmp.analysis.cs_game_card import build_canonical_game, build_no_integrator_game
from rlrmp.analysis.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
)
from rlrmp.analysis.output_feedback import OutputFeedbackConfig, process_covariance
from rlrmp.cs_lss_gru import (
    CS_PHYSICAL_STATE_DIM,
    CS_REDUCED_PHYSICAL_STATE_DIM,
    build_cs_lss_gru_graph,
)
from rlrmp.disturbance import (
    PLANT_INTERVENOR_LABEL,
)
from rlrmp.disturbances import get_gusts_fn
from rlrmp.intervention_compat import add_plant_intervention_to_ensemble
from rlrmp.loss import get_reach_loss
from rlrmp.models import (
    LINEAR_HIDDEN_TYPES,
    create_point_mass_linear_ensemble,
    create_point_mass_nn_ensemble,
)
from rlrmp.stochastic_runtime import (
    apply_stochastic_runtime_to_ensemble,
    stochastic_runtime_config_from_model,
)
from rlrmp.train.cs_perturbation_training import (
    BroadFullStateEpsilonTrainingTaskAdapter,
    FixedTargetPerturbationTrainingTaskAdapter,
    TargetRelativeMultiTargetTrainingTaskAdapter,
    config_from_broad_epsilon_hps,
    config_from_hps,
    config_from_target_hps,
    install_perturbation_training_graph_adapters,
)
from rlrmp.task import TASK_TYPES

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
            dtype=float,
        ),
        "bcs": lambda trial_specs, key: trial_specs.intervene[PLANT_INTERVENOR_LABEL].active.astype(
            float
        ),
        "dai": lambda trial_specs, key: get_field_amplitude(
            trial_specs.intervene[PLANT_INTERVENOR_LABEL]
        ),
        "pai-asf": lambda trial_specs, key: trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale,
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
        return CurlFieldParams(scale=scale, active=active, **extra_params)
    elif pert_type == "constant":
        return FixedFieldParams(scale=scale, active=active, **extra_params)
    elif pert_type == "gusts":
        return FixedFieldParams(scale=scale, active=active, **extra_params)
    else:
        raise ValueError(f"Unknown perturbation type: {pert_type}")


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
        }
        hps_task = {k: v for k, v in hps_task.items() if k not in delayed_only_keys}

    task_base = TASK_TYPES[task_type](loss_func=get_reach_loss(hps), **hps_task)

    # Resolve hidden_type from hps if present; default (None) falls back to GRUCell
    hidden_type = getattr(hps, 'hidden_type', None)
    # Resolve SISU gating mode; default "additive" preserves existing behavior
    sisu_gating = getattr(hps, 'sisu_gating', 'additive')
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
        delayed_reach = _cs_delayed_reach_enabled(hps)
        task = _add_cs_lss_task_inputs(
            _CsLssTaskAdapter(task_base, physical_state_dim=physical_state_dim),
            target_relative=target_training.enabled,
            go_cue_input=target_training.enabled and delayed_reach,
            physical_state_dim=physical_state_dim,
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
            models = install_perturbation_training_graph_adapters(models)
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
        # Create base models with extra input for SISU
        models_base = create_point_mass_nn_ensemble(
            hps,
            task_base,
            n_extra_inputs=1,  # for SISU (even when multiplicative, task still provides it)
            hidden_type=hidden_type,
            sisu_gating=sisu_gating,
            key=key,
        )

    # Insert intervention components into models via graph surgery
    models = add_plant_intervention_to_ensemble(
        models_base,
        hps.pert.type,
        PLANT_INTERVENOR_LABEL,
        active=False,  # Default to inactive; schedule_intervenor will control activation
    )
    models = apply_stochastic_runtime_to_ensemble(
        models,
        stochastic_runtime_config_from_model(hps.model),
    )

    # Add SISU input to task
    try:
        task = task_base.add_input(
            name="sisu",
            input_fn=SISU_FNS[hps.method],
        )
    except AttributeError:
        raise ValueError("No training method label assigned to hps_train.method")

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
        )
        return TaskTrialSpec(
            inits=WhereDict({"mechanics.vector": lss_vector}),
            inputs=trial_spec.inputs,
            targets=trial_spec.targets,
            intervene=trial_spec.intervene,
            extra=trial_spec.extra,
            timeline=trial_spec.timeline,
        )


def _effector_init_to_lss_vector(
    effector_init: CartesianState,
    *,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
) -> jax.Array:
    pos = jnp.asarray(effector_init.pos)
    vel = jnp.asarray(effector_init.vel)
    force = jnp.asarray(effector_init.force)
    if int(physical_state_dim) < CS_REDUCED_PHYSICAL_STATE_DIM:
        raise ValueError(f"physical_state_dim must be >= 6; got {physical_state_dim}.")
    batch_shape = jnp.broadcast_shapes(pos.shape[:-1], vel.shape[:-1], force.shape[:-1])
    vector = jnp.zeros(
        (*batch_shape, 6 * int(physical_state_dim)),
        dtype=jnp.result_type(pos, vel, force, float),
    )
    return vector.at[..., 0:2].set(jnp.broadcast_to(pos, (*batch_shape, 2))).at[
        ..., 2:4
    ].set(jnp.broadcast_to(vel, (*batch_shape, 2))).at[..., 4:6].set(
        jnp.broadcast_to(force, (*batch_shape, 2))
    )


def _add_cs_lss_task_inputs(
    task: _CsLssTaskAdapter,
    *,
    target_relative: bool = False,
    go_cue_input: bool = False,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
) -> _CsLssTaskAdapter:
    if go_cue_input:
        task = task.add_input(
            name="input",
            input_fn=_cs_delayed_go_cue_input,
        )
    elif not target_relative:
        task = task.add_input(
            name="input",
            input_fn=SISU_FNS["nominal-cs-gru"],
        )
    return task.add_input(
        name="epsilon",
        input_fn=lambda trial_spec, key: _sample_cs_lss_process_epsilon(
            trial_spec,
            key,
            physical_state_dim=physical_state_dim,
        ),
    )


def _cs_delayed_reach_enabled(hps: TreeNamespace) -> bool:
    return str(getattr(getattr(hps, "task", TreeNamespace()), "type", "")) == (
        "cs_delayed_center_out_reach"
    )


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


def _sample_cs_lss_process_epsilon(
    trial_spec: TaskTrialSpec,
    key: PRNGKeyArray,
    *,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
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
    factor = _cs_lss_process_epsilon_factor(physical_state_dim=physical_state_dim)
    epsilon_dim = int(factor.shape[0])
    draws = jr.normal(key, (*batch_shape, n_steps, epsilon_dim), dtype=jnp.float64)
    return draws @ factor.T


def _cs_lss_process_epsilon_factor(
    *,
    physical_state_dim: int = CS_PHYSICAL_STATE_DIM,
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
    return eigvecs @ jnp.diag(jnp.sqrt(jnp.clip(eigvals, min=0.0)))


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
        delayed_reach = _cs_delayed_reach_enabled(hps)
        return build_cs_lss_gru_graph(
            hidden_size=int(hps.model.hidden_size),
            input_size=1 if (delayed_reach or not target_training.enabled) else 0,
            hidden_type=hidden_type,
            population_structure=population_structure,
            sisu_gating=sisu_gating,
            sensory_noise_std=float(hps.model.sensory_noise_std),
            additive_motor_noise_std=float(hps.model.additive_motor_noise_std),
            signal_dependent_motor_noise_std=float(hps.model.signal_dependent_motor_noise_std),
            bind_epsilon_input=True,
            target_relative_feedback=target_training.enabled,
            force_filter_feedback=target_training.force_filter_feedback,
            initial_hidden_encoder=bool(getattr(hps.model, "initial_hidden_encoder", False)),
            no_integrator_state=no_integrator_state,
            key=key_one,
        )

    return eqx.filter_vmap(build_one)(keys)

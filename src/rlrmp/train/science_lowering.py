"""Registered lowering of authored training science into run-spec metadata."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
from feedbax import LowererRegistration, OrderedLowererRegistry
from feedbax.config.namespace import TreeNamespace

from rlrmp.analysis.math.cs_game_card import build_canonical_game, build_no_integrator_game
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.train.broad_epsilon_training import (
    lower_broad_epsilon_science_mode,
    lower_pgd_science_mode,
)
from rlrmp.train.config_materialization import (
    CS_REGULARIZED_NN_HIDDEN,
    CS_STAGE_COUNT,
    DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
)
from rlrmp.train.cs_perturbation_training import lower_target_relative_science_mode
from rlrmp.train.delayed_reach import lower_science_mode as lower_delayed_reach_mode
from rlrmp.train.executor.checkpoints import _plain
from rlrmp.train.fixed_target_perturbation_training import (
    lower_science_mode as lower_perturbation_mode,
)
from rlrmp.train.policy_adversary_native import lower_science_mode as lower_policy_mode
from rlrmp.train.science_vocabulary import (
    FidelityProfile,
    FULL_QRF_FIDELITY,
    PARTIAL_FIDELITY,
    PARTIAL_FORCE_FIDELITY,
    ScienceMode,
)
from rlrmp.train.task_model import CS_LSS_PLANT_BACKEND


@dataclass(frozen=True)
class LoweredMode:
    """Composed mode string and contributing capability registrations."""

    training_mode: str
    lowerer_ids: tuple[str, ...]


@dataclass(frozen=True)
class LoweredScience:
    """Combined compatibility view for callers needing mode and objective."""

    training_mode: str
    loss_spec: dict[str, Any]
    fidelity_status: dict[str, Any]
    lowerer_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObjectiveProfile:
    objective: str
    loss_builder: Callable[[TreeNamespace], dict[str, Any]]
    fidelity: FidelityProfile


def _enabled(hps: TreeNamespace, field: str) -> bool:
    return bool(getattr(hps, field).enabled)


def _time_indexing(hps: TreeNamespace) -> tuple[dict[str, Any], str]:
    if not _enabled(hps, "delayed_reach"):
        return (
            {
                "stage_schedule": "trial_age_full_simple_reach",
                "canonical_movement_horizon_steps": CS_STAGE_COUNT,
            },
            "((t + 1) / T)^6",
        )
    tail = str(hps.loss.delayed_movement_cost_tail_mode)
    return (
        {
            "stage_schedule": "movement_age_from_go_cue",
            "movement_epoch_source": "trial_specs.timeline.epoch_bounds[-2:]",
            "prep_target_directed_movement_loss": "zero",
            "canonical_movement_horizon_steps": CS_STAGE_COUNT,
            "cost_tail_mode": tail,
            "post_horizon_tail": "hold_terminal_running_qr_weights_flat_to_trial_end"
            if tail == DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON
            else "zero_weight_after_canonical_horizon",
        },
        "((movement_age + 1) / 60)^6, capped at 1",
    )


def delayed_pre_go_auxiliary_terms(hps: TreeNamespace) -> dict[str, Any]:
    """Return provenance for delayed-reach preparation-epoch auxiliary terms."""

    weights = hps.loss.weights
    terms = {
        "delayed_pre_go_force_filter_hold": {
            "scale": float(weights.delayed_pre_go_force_filter_hold),
            "state_key": "states.mechanics.vector delay blocks[..., 4:6]",
            "target": "zero_force_filter_state",
        },
        "delayed_pre_go_start_pos_hold": {
            "scale": float(weights.delayed_pre_go_start_pos_hold),
            "state_key": "states.mechanics.effector.pos",
            "target": "trial_specs.inits['mechanics.vector'][..., :2]",
            "norm": str(getattr(hps.loss, "delayed_pre_go_start_pos_hold_norm", "l2")),
        },
        "delayed_pre_go_zero_vel_hold": {
            "scale": float(weights.delayed_pre_go_zero_vel_hold),
            "state_key": "states.mechanics.effector.vel",
            "target": "zero_velocity",
        },
    }
    delayed = _enabled(hps, "delayed_reach")
    return {
        "scope": "prep_epoch_only" if delayed else "inactive",
        "epoch_indices": [0] if delayed else [],
        "movement_window_qrf_comparator": "unchanged",
        "terms": terms,
        "active_terms": {name: meta for name, meta in terms.items() if meta["scale"] != 0.0},
    }


def _common_loss(hps: TreeNamespace, objective: str) -> dict[str, Any]:
    indexing, fact_t = _time_indexing(hps)
    weights = hps.loss.weights
    return {
        "weights": _plain(weights),
        "delayed_pre_go_auxiliary_terms": delayed_pre_go_auxiliary_terms(hps),
        "delayed_reach": _plain(hps.delayed_reach),
        "effector_pos_late": _plain(hps.loss.effector_pos_late),
        "effector_vel_late": _plain(hps.loss.effector_vel_late),
        "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
        "objective_profile": objective,
        "active_cs_terms": {
            "stage_position": {
                "term": "effector_pos_running",
                "scale": float(weights.effector_pos_running),
                "fact_t": fact_t,
            },
            "stage_velocity": {
                "term": "effector_vel_running",
                "scale": float(weights.effector_vel_running),
                "fact_t": fact_t,
            },
            "control": {
                "term": "nn_output",
                "scale": float(weights.nn_output),
                "equivalent_R": "I_2 on efferent output",
            },
            "terminal_position": {
                "term": "effector_terminal_pos",
                "scale": float(weights.effector_terminal_pos),
            },
            "terminal_velocity": {
                "term": "effector_terminal_vel",
                "scale": float(weights.effector_terminal_vel),
            },
        },
        "hidden_regularizer": {
            "term": "nn_hidden",
            "scale": float(weights.nn_hidden),
            "exact_fidelity_default": 0.0,
            "regularized_pair_scale": CS_REGULARIZED_NN_HIDDEN,
        },
        "simple_reach_position_loss_contract": (
            "effector_pos_running compares mechanics.effector.pos to the SimpleReaches "
            "same-coordinate target sequence over every transition, using the configured "
            "C&S Eq. 15 power-law discount when requested."
        ),
        "effector_hold_pos_schedule": str(hps.loss.effector_hold_pos_schedule),
        "position_powerlaw_power": float(hps.loss.position_powerlaw_power),
        "movement_ramp_shape": str(hps.loss.movement_ramp_shape),
        "movement_ramp_duration_steps": int(hps.loss.movement_ramp_duration_steps),
        "movement_ramp_power": float(hps.loss.movement_ramp_power),
        "time_indexing": indexing,
    }


def _partial_loss(hps: TreeNamespace) -> dict[str, Any]:
    payload = _common_loss(hps, CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE)
    payload.update(
        force_filter_state_cost="not_available",
        force_filter_state_cost_note=(
            "No force/filter-state quadratic term is synthesized because this nominal "
            "Feedbax loss path has no clean C&S physical force/integrator state target "
            "exposed through the task state contract."
        ),
    )
    return payload


def _partial_force_filter_loss(hps: TreeNamespace) -> dict[str, Any]:
    payload = _common_loss(hps, CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE)
    payload.update(
        objective_kind="partial_feedbax_ablation",
        hypothesis=(
            "historical partial position/velocity terms plus intended-command control cost "
            "and LSS force/filter state cost"
        ),
        force_filter_state_cost="included_as_partial_ablation_running_term",
        disturbance_integrator_state_cost="omitted_in_this_ablation",
    )
    payload["active_cs_terms"]["control"].update(
        state_key="states.net.output",
        equivalent_R="I_2 on intended controller command before noise",
    )
    payload["active_cs_terms"]["force_filter"] = {
        "term": "mechanics_force_filter",
        "state_key": "states.mechanics.vector delay blocks[..., 4:6]",
        "scale": float(hps.loss.weights.mechanics_force_filter),
        "basis": "force/filter coordinates from every 8D physical delay block",
    }
    return payload


def _full_qrf_loss(hps: TreeNamespace) -> dict[str, Any]:
    delayed = _enabled(hps, "delayed_reach")
    no_integrator = bool(hps.model.no_integrator_state)
    _plant, schedule = build_no_integrator_game() if no_integrator else build_canonical_game()
    physical_dim = 6 if no_integrator else 8
    indexing, _fact_t = _time_indexing(hps)
    normalization = _plain(
        getattr(hps.loss, "delayed_trial_type_normalization", {"enabled": False})
    )
    return {
        "weights": _plain(hps.loss.weights),
        "delayed_pre_go_auxiliary_terms": delayed_pre_go_auxiliary_terms(hps),
        "delayed_trial_type_normalization": normalization,
        "delayed_reach": _plain(hps.delayed_reach),
        "objective_profile": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        "objective_kind": "finite_horizon_quadratic",
        "grouped_reduction_implementation": "rlrmp_bridge_pending_feedbax_69d8d76"
        if normalization["enabled"]
        else "not_enabled",
        "source_module": "rlrmp.analysis.math.cs_game_card.build_no_integrator_game"
        if no_integrator
        else "rlrmp.analysis.math.cs_game_card.build_canonical_game",
        "comparator_variant": "no_integrator_state" if no_integrator else None,
        "state_basis": {
            "state_key": "states.mechanics.vector",
            "dimension": int(schedule.Q.shape[-1]),
            "physical_block_size": physical_dim,
            "delay_blocks": int(schedule.Q.shape[-1] // physical_dim),
            "coordinate_transform": (
                "absolute Feedbax position entries are converted to target-centred "
                "analytical coordinates before applying Q_t and Q_f"
            ),
        },
        "time_indexing": {
            "running_state": "state before each movement command from sampled go cue"
            if delayed
            else "trial init plus rollout states[:-1], paired with commands",
            "terminal_state": (
                "final rollout state after the variable post-horizon tail"
                if str(hps.loss.delayed_movement_cost_tail_mode)
                == DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON
                else "state after 60 movement commands from sampled go cue"
            )
            if delayed
            else "rollout states[-1]",
            "horizon_steps": int(schedule.T),
            **indexing,
        },
        "matrix_shapes": {
            "Q": list(schedule.Q.shape),
            "R": list(schedule.R.shape),
            "Q_f": list(schedule.Q_f.shape),
        },
        "active_cs_terms": {
            "state_running_q": {
                "term": "mechanics.vector^T Q_t mechanics.vector",
                "source": "canonical delay-augmented C&S schedule.Q",
                "initial_diag_first_block": [float(x) for x in jnp.diag(schedule.Q[0])[:8]],
            },
            "control_r": {
                "term": "net.output^T R_t net.output",
                "source": (
                    "canonical C&S schedule.R on intended controller command before "
                    "efferent/motor-channel noise"
                ),
                "diag": [float(x) for x in jnp.diag(schedule.R[0])],
            },
            "terminal_q_f": {
                "term": "mechanics.vector_T^T Q_f mechanics.vector_T",
                "source": "canonical delay-augmented C&S schedule.Q_f",
                "diag_first_block": [float(x) for x in jnp.diag(schedule.Q_f)[:8]],
            },
        },
        "force_filter_state_cost": "included_via_Q_entries_4_5_each_delay_block",
        "disturbance_integrator_state_cost": "included_via_Q_entries_6_7_each_delay_block",
        "hidden_regularizer": {
            "term": "not_in_full_analytical_qrf_loss",
            "configured_weight": float(hps.loss.weights.nn_hidden),
        },
    }


def _fidelity(hps: TreeNamespace, profile: FidelityProfile) -> dict[str, Any]:
    nn_hidden = float(hps.loss.weights.nn_hidden)
    backend, objective = (
        str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)),
        profile.objective,
    )
    exact_lss, allow_regularizer = (
        backend == CS_LSS_PLANT_BACKEND,
        profile.allow_regularizer,
    )
    extra = (
        []
        if not allow_regularizer or nn_hidden == 0.0
        else [
            {
                "term": "nn_hidden",
                "scale": nn_hidden,
                "status": "auxiliary_regularizer_not_in_analytical_objective",
            }
        ]
    )
    payload = {
        "objective": "cs_fidelity_stochastic_rollout",
        "loss_objective": objective,
        "exact_fidelity": False,
        "exact_objective_terms": exact_lss if profile.exact_from_backend else False,
        "exact_objective_terms_scope": profile.exact_scope,
        "objective_fidelity": {
            "implemented_terms": list(profile.implemented),
            "omitted_terms": ([] if exact_lss else list(profile.omitted))
            if profile.exact_from_backend
            else list(profile.omitted),
            "extra_terms": extra,
            "selection_policy": profile.selection_policy,
        },
        "exact_stochastic_rollout": False,
        "exact_stochastic_noise_sources": exact_lss,
        "exact_plant_matrices": exact_lss,
        "plant_backend": backend,
        "temporary_stochastic_bridge": (
            "temporary RLRMP LSS wrapper implements sensory Channel, additive and "
            "signal-dependent motor Channel, and sampled physical-process "
            "mechanics.epsilon; future Feedbax acausal/ODE plant support should subsume "
            "this wrapper"
        )
        if exact_lss
        else None,
        "stochastic_preset": str(hps.model.stochastic_preset),
        "stochastic_projection": (
            "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, and "
            "plant/load force noise channels without feeding the 48D delay-augmented "
            "analytical state to the GRU."
        ),
        "regularized_pair": allow_regularizer and nn_hidden != 0.0,
        "regularizer": "nn_hidden" if allow_regularizer and nn_hidden != 0.0 else "none",
        "nn_hidden": nn_hidden,
        "certificate_lens": "input_output_map_certificate",
        "same_coordinate_gain_certificate": False,
    }
    if profile.analytical_delay_flag:
        payload["analytical_delay_augmented_state_input"] = False
    return payload


def _select(profile: ObjectiveProfile, hps: TreeNamespace) -> ObjectiveProfile | None:
    return profile if str(hps.loss.objective) == profile.objective else None


_PROFILES = (
    ObjectiveProfile(
        CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        _full_qrf_loss,
        FULL_QRF_FIDELITY,
    ),
    ObjectiveProfile(
        CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
        _partial_force_filter_loss,
        PARTIAL_FORCE_FIDELITY,
    ),
    ObjectiveProfile(
        CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
        _partial_loss,
        PARTIAL_FIDELITY,
    ),
)
MODE_LOWERERS = OrderedLowererRegistry[TreeNamespace, ScienceMode](
    [
        LowererRegistration(
            "delayed_reach", 0, "rlrmp.train.delayed_reach", lower_delayed_reach_mode
        ),
        LowererRegistration(
            "target_relative",
            10,
            "rlrmp.train.cs_perturbation_training",
            lower_target_relative_science_mode,
        ),
        LowererRegistration(
            "broad_epsilon",
            20,
            "rlrmp.train.broad_epsilon_training",
            lower_broad_epsilon_science_mode,
        ),
        LowererRegistration(
            "broad_epsilon_pgd", 21, "rlrmp.train.broad_epsilon_training", lower_pgd_science_mode
        ),
        LowererRegistration(
            "policy_adversary", 22, "rlrmp.train.policy_adversary_native", lower_policy_mode
        ),
        LowererRegistration(
            "perturbation",
            30,
            "rlrmp.train.fixed_target_perturbation_training",
            lower_perturbation_mode,
        ),
    ]
)
OBJECTIVE_LOWERERS = OrderedLowererRegistry[TreeNamespace, ObjectiveProfile](
    [
        LowererRegistration(
            f"objective.{name}", 100, f"rlrmp.loss.{name}", partial(_select, profile)
        )
        for name, profile in zip(
            ("full_qrf", "partial_force_filter", "partial"), _PROFILES, strict=True
        )
    ]
)


def lower_training_mode(hps: TreeNamespace) -> LoweredMode:
    """Compose only cheap capability mode callbacks; never build objective payloads."""

    contributions = MODE_LOWERERS.lower(hps)
    return LoweredMode(
        "+".join(item.fragment.value for item in contributions) or ScienceMode.NOMINAL.value,
        tuple(item.lowerer_id for item in contributions),
    )


def _objective_profile(hps: TreeNamespace) -> tuple[str, ObjectiveProfile]:
    contributions = OBJECTIVE_LOWERERS.lower(hps)
    if len(contributions) != 1:
        raise ValueError(
            f"training science must lower exactly one objective; found {len(contributions)}"
        )
    item = contributions[0]
    return item.lowerer_id, item.fragment


def lower_training_loss(hps: TreeNamespace) -> dict[str, Any]:
    """Build only the selected objective's loss payload."""

    _lowerer_id, profile = _objective_profile(hps)
    return profile.loss_builder(hps)


def lower_training_fidelity(hps: TreeNamespace) -> dict[str, Any]:
    """Build only the selected objective's fidelity payload."""

    _lowerer_id, profile = _objective_profile(hps)
    return _fidelity(hps, profile.fidelity)


def lower_training_science(hps: TreeNamespace) -> LoweredScience:
    """Return the combined view for diagnostics and semantic-equivalence tests."""

    mode, (lowerer_id, profile) = lower_training_mode(hps), _objective_profile(hps)
    return LoweredScience(
        mode.training_mode,
        profile.loss_builder(hps),
        _fidelity(hps, profile.fidelity),
        (*mode.lowerer_ids, lowerer_id),
    )


__all__ = [
    "LoweredMode",
    "LoweredScience",
    "MODE_LOWERERS",
    "OBJECTIVE_LOWERERS",
    "delayed_pre_go_auxiliary_terms",
    "lower_training_fidelity",
    "lower_training_loss",
    "lower_training_mode",
    "lower_training_science",
]

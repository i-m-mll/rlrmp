"""Typed vocabulary emitted by training-science lowerers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)


class ScienceMode(StrEnum):
    """Stable run-spec identifiers for composed training capabilities."""

    NOMINAL = "nominal"
    PERTURBATION = "fixed_target_perturbation_randomized"
    PERTURBATION_CALIBRATED = "fixed_target_perturbation_calibrated_timing"
    PERTURBATION_GENERALIZED = "fixed_target_perturbation_generalized"
    TARGET_RELATIVE = "target_relative_multitarget_static"
    TARGET_RELATIVE_H0 = "target_relative_multitarget_static_h0"
    BROAD_EPSILON = "broad_full_state_epsilon_l2"
    BROAD_EPSILON_PGD = "broad_full_state_epsilon_pgd_l2"
    POLICY_ADVERSARY = "broad_full_state_epsilon_policy_l2"
    DELAYED_REACH = "delayed_reach_target_visible_go_cue"


class AdaptiveEpsilonControllerMode(StrEnum):
    """Authored adaptive-epsilon controller update strategies."""

    LOSS_BLEND = "loss_blend"
    EPSILON_SCALED_OUTER = "epsilon_scaled_outer_training"


@dataclass(frozen=True)
class FidelityProfile:
    """Declarative fidelity vocabulary for one authored objective."""

    objective: str
    exact_scope: str
    implemented: tuple[str, ...]
    omitted: tuple[Any, ...]
    selection_policy: str
    exact_from_backend: bool = False
    allow_regularizer: bool = True
    analytical_delay_flag: bool = True


_SHARED_PARTIAL_TERMS = (
    "running_position_cs_eq15_power6",
    "terminal_position",
    "running_velocity_cs_eq15_power6",
    "terminal_velocity",
)

PARTIAL_FIDELITY = FidelityProfile(
    objective=CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    exact_scope=(
        "false because force/filter-state and disturbance-integrator state costs from "
        "the analytical C&S schedule are omitted from the current Feedbax GRU loss contract"
    ),
    implemented=(*_SHARED_PARTIAL_TERMS, "command_quadratic_nn_output"),
    omitted=(
        {
            "term": "force_filter_state_cost",
            "analytical_role": "unit-weight force/filter state cost in the C&S 8D schedule",
            "status": "not_synthesized_in_feedbax_gru_loss",
        },
        {
            "term": "disturbance_integrator_state_cost",
            "analytical_role": (
                "unit-weight disturbance-integrator state cost in the C&S 8D schedule"
            ),
            "status": "not_synthesized_in_feedbax_gru_loss",
        },
    ),
    selection_policy=(
        "rollout validation loss only; analytical action and I/O metrics are audit-only"
    ),
)

PARTIAL_FORCE_FIDELITY = FidelityProfile(
    objective=CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    exact_scope=(
        "ablation only: old partial position/velocity terms are kept, control is moved to "
        "intended net.output, and running force/filter state cost is added; this is not the "
        "full Q/R/Q_f objective"
    ),
    implemented=(
        *_SHARED_PARTIAL_TERMS,
        "intended_command_quadratic_net_output",
        "running_force_filter_state_cost",
    ),
    omitted=(
        {
            "term": "disturbance_integrator_state_cost",
            "analytical_role": (
                "unit-weight disturbance-integrator state cost in the C&S 8D schedule"
            ),
            "status": "intentionally_omitted_for_force_filter_ablation",
        },
        {
            "term": "terminal_force_filter_and_integrator_Q_f",
            "analytical_role": "terminal full-state Q_f costs",
            "status": "not_synthesized_in_partial_ablation",
        },
    ),
    selection_policy=(
        "rollout validation loss only; analytical action and I/O metrics are audit-only"
    ),
)

FULL_QRF_FIDELITY = FidelityProfile(
    objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    exact_scope=(
        "true for the implemented training scalar when plant_backend='cs_lss': the loss "
        "evaluates canonical C&S delay-augmented Q_t, R_t, and Q_f on the exposed "
        "LinearStateSpace state and command history"
    ),
    implemented=(
        "delay_augmented_state_running_Q_t",
        "command_running_R_t",
        "delay_augmented_terminal_Q_f",
    ),
    omitted=("cs_lss_state_unavailable",),
    selection_policy=(
        "rollout validation loss uses the same full analytical Q/R/Q_f training scalar; "
        "analytical action and I/O metrics remain audit-only"
    ),
    exact_from_backend=True,
    allow_regularizer=False,
    analytical_delay_flag=False,
)


__all__ = [
    "AdaptiveEpsilonControllerMode",
    "FidelityProfile",
    "FULL_QRF_FIDELITY",
    "PARTIAL_FIDELITY",
    "PARTIAL_FORCE_FIDELITY",
    "ScienceMode",
]

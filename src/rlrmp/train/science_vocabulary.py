"""Typed vocabulary emitted by training-science lowerers."""

from __future__ import annotations

from enum import StrEnum


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


__all__ = ["AdaptiveEpsilonControllerMode", "ScienceMode"]

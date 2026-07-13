"""Unified, registered C&S training configuration family.

This module owns the authoring schemas consumed by generated CLIs and typed
run-matrix validation. Runtime trainers import these models; they do not
reconstruct parallel hyperparameter schemas.
"""
# ruff: noqa: F401

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
)
from jaxtyping import PRNGKeyArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlrmp.data_products.broad_epsilon import (
    load_broad_epsilon_anchors,
    load_pgd_radius_source,
)
from rlrmp.data_products.calibration import (
    CALIBRATION_PRODUCT_RELPATH,
    CALIBRATION_PRODUCT_ROLE,
    load_open_loop_calibration,
    load_perturbation_calibration_defaults,
)
from rlrmp.model.feedbax_channel_adapters import (
    additive_channel_provenance,
)
from rlrmp.model.feedback_descriptors import (
    COMPONENT_FORCE_FILTER,
    resolve_controller_feedback_view,
)
from rlrmp.model.cs_lss_contracts import (
    FINITE_EPSILON_POLICY_GRAPH_COMPONENT,
    FINITE_EPSILON_POLICY_NODE_LABEL,
)
from rlrmp.runtime.params_models import register_params_model
from rlrmp.train.closed_loop_finite_adversary import (
    AFFINE_POLICY,
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
    LINEAR_NO_BIAS_POLICY,
)
from rlrmp.train.science_vocabulary import (
    AdaptiveEpsilonControllerMode,
    ScienceMode,
)
from rlrmp.train.training_presets import load_training_presets, training_preset_value
from rlrmp.train.training_payload_migrations import migrate_frozen_rendered_training_payload


class CsPerturbationTrainingConfig(BaseModel):
    """Shared strict base for C&S perturbation training config models."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    def to_json(self) -> dict[str, Any]:
        """Return run-spec metadata for this config."""

        return self.to_hps_dict()

    @classmethod
    def from_payload(cls, payload: Any) -> Any:
        """Validate the canonical authored config embedded in a runtime payload."""

        return validate_training_config(cls, payload)


FIXED_TARGET_PERTURBATION_PARAMS_REF = "rlrmp.train.fixed_target_perturbation_training.v1"

TARGET_RELATIVE_MULTITARGET_PARAMS_REF = "rlrmp.train.target_relative_multitarget.v1"

BROAD_EPSILON_PARAMS_REF = "rlrmp.train.broad_full_state_epsilon.v1"

BROAD_EPSILON_PGD_PARAMS_REF = "rlrmp.train.broad_full_state_epsilon_pgd.v1"

POLICY_ADVERSARY_PARAMS_REF = "rlrmp.train.broad_full_state_epsilon_policy.v1"

POLICY_ADVERSARY_MEMORYLESS_MLP = "memoryless_mlp"

POLICY_ADVERSARY_PLAIN_MODE = "plain"

POLICY_ADVERSARY_ENERGY_MODE = "energy"

POLICY_ADVERSARY_POLICY_CLASSES = (
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    LINEAR_NO_BIAS_POLICY,
    AFFINE_POLICY,
)

BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE = "fixed"

BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE = "sisu_energy_fraction"

BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE = "hard_l2"

BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE = "soft_energy"

BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM = "direct_epsilon"

BROAD_EPSILON_PGD_MECHANISMS: tuple[str, ...] = (
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    LINEAR_NO_BIAS_POLICY,
    AFFINE_POLICY,
)

BROAD_EPSILON_PGD_FINITE_POLICY_MECHANISMS: tuple[str, ...] = (
    LINEAR_NO_BIAS_POLICY,
    AFFINE_POLICY,
)

BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT = "projected_gradient_ascent"

BROAD_EPSILON_PGD_ADAM = "adam"

BROAD_EPSILON_PGD_INNER_OPTIMIZER_METHODS: tuple[str, ...] = (
    BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
    BROAD_EPSILON_PGD_ADAM,
)

_TRAINING_AUTHORING_PRESETS = load_training_presets()
_PGD_SISU_PRESET = _TRAINING_AUTHORING_PRESETS["pgd_sisu"]
_TARGET_SUPPORT_PRESET = _TRAINING_AUTHORING_PRESETS["target_support"]

DEFAULT_PGD_SISU_LEVELS: tuple[float, ...] = tuple(_PGD_SISU_PRESET["levels"])

DEFAULT_PGD_SISU_EXACT_ZERO_MASS = float(_PGD_SISU_PRESET["exact_zero_mass"])

HISTORICAL_020A65B_PGD_RADIUS_15CM = float(
    load_pgd_radius_source("effective_020a65b_pgd_training_radius")["l2_radius_15cm"]
)

MILD_COMBINED_FAMILIES: tuple["PerturbationBin", ...] = (
    "initial_position",
    "command_input",
)

AMPLITUDE_LEVELS: tuple[float, ...] = tuple(
    training_preset_value("shared", "amplitude_levels")
)

ORIGINAL_TARGET_ANCHOR_M: tuple[float, float] = tuple(
    _TARGET_SUPPORT_PRESET["original_target_anchor_m"]
)

DEFAULT_SEEN_TARGET_DIRECTIONS_DEG: tuple[float, ...] = tuple(
    _TARGET_SUPPORT_PRESET["seen_directions_deg"]
)

DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG: tuple[float, ...] = tuple(
    _TARGET_SUPPORT_PRESET["held_out_directions_deg"]
)

DEFAULT_SEEN_TARGET_AMPLITUDES_M: tuple[float, ...] = tuple(
    _TARGET_SUPPORT_PRESET["seen_amplitudes_m"]
)

DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M: tuple[float, ...] = tuple(
    _TARGET_SUPPORT_PRESET["held_out_amplitudes_m"]
)

TARGET_SUPPORT_PROFILE_020A65B = str(
    training_preset_value("shared", "legacy_target_support_profile")
)

TARGET_SUPPORT_PROFILE_CONST_DENSE_ALL = "const_dense_all"

TARGET_SUPPORT_PROFILE_CONST_SPARSE8 = "const_sparse8"

TARGET_SUPPORT_PROFILE_CONST_BAND8 = "const_band8"

TARGET_SUPPORT_PROFILE_CONST_BAND16 = "const_band16"

TARGET_SUPPORT_PROFILE_CONST_BAND36 = "const_band36"

DEFAULT_TARGET_SUPPORT_PROFILE = str(_TARGET_SUPPORT_PRESET["default_profile"])

TARGET_SUPPORT_PROFILES: tuple[str, ...] = (
    TARGET_SUPPORT_PROFILE_020A65B,
    TARGET_SUPPORT_PROFILE_CONST_DENSE_ALL,
    TARGET_SUPPORT_PROFILE_CONST_SPARSE8,
    TARGET_SUPPORT_PROFILE_CONST_BAND8,
    TARGET_SUPPORT_PROFILE_CONST_BAND16,
    TARGET_SUPPORT_PROFILE_CONST_BAND36,
)

TARGET_SUPPORT_CONST_REACH_M = float(_TARGET_SUPPORT_PRESET["constant_reach_m"])

TARGET_SUPPORT_DENSE_N_DIRECTIONS = int(_TARGET_SUPPORT_PRESET["dense_n_directions"])

TARGET_SUPPORT_SPARSE_N_DIRECTIONS = int(_TARGET_SUPPORT_PRESET["sparse_n_directions"])

TARGET_SUPPORT_BAND_CENTERS_DEG: tuple[float, ...] = tuple(
    _TARGET_SUPPORT_PRESET["band_centers_deg"]
)

TARGET_SUPPORT_BAND8_HELD_OUT_DIRECTIONS = int(
    _TARGET_SUPPORT_PRESET["held_out_counts"][TARGET_SUPPORT_PROFILE_CONST_BAND8]
)

TARGET_SUPPORT_BAND16_HELD_OUT_DIRECTIONS = int(
    _TARGET_SUPPORT_PRESET["held_out_counts"][TARGET_SUPPORT_PROFILE_CONST_BAND16]
)

TARGET_SUPPORT_BAND36_HELD_OUT_DIRECTIONS = int(
    _TARGET_SUPPORT_PRESET["held_out_counts"][TARGET_SUPPORT_PROFILE_CONST_BAND36]
)

PerturbationBin = Literal[
    "nominal",
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "target_aligned_lateral_load",
    "delayed_observation",
    "mild_combined",
]

TrainingCalibrationRegime = Literal[
    "open_loop_all",
    "closed_loop_sensory",
    "closed_loop_sensory_command_lateral",
]

OPEN_LOOP_ALL_CALIBRATION_REGIME: TrainingCalibrationRegime = "open_loop_all"

CLOSED_LOOP_SENSORY_CALIBRATION_REGIME: TrainingCalibrationRegime = "closed_loop_sensory"

CLOSED_LOOP_SENSORY_COMMAND_LATERAL_CALIBRATION_REGIME: TrainingCalibrationRegime = (
    "closed_loop_sensory_command_lateral"
)

TRAINING_CALIBRATION_REGIMES: tuple[TrainingCalibrationRegime, ...] = (
    OPEN_LOOP_ALL_CALIBRATION_REGIME,
    CLOSED_LOOP_SENSORY_CALIBRATION_REGIME,
    CLOSED_LOOP_SENSORY_COMMAND_LATERAL_CALIBRATION_REGIME,
)

TRAINING_CALIBRATION_CLOSED_LOOP_FAMILIES: dict[TrainingCalibrationRegime, tuple[str, ...]] = {
    OPEN_LOOP_ALL_CALIBRATION_REGIME: (),
    CLOSED_LOOP_SENSORY_CALIBRATION_REGIME: ("sensory_feedback_offset",),
    CLOSED_LOOP_SENSORY_COMMAND_LATERAL_CALIBRATION_REGIME: (
        "sensory_feedback_offset",
        "command_input_pulse",
        "target_aligned_lateral_command_load_pulse",
    ),
}

INACTIVE_LEGACY_PERTURBATION_BINS: tuple[PerturbationBin, ...] = ("delayed_observation",)

VALIDATION_BINS: tuple[PerturbationBin, ...] = (
    "nominal",
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "mild_combined",
)

SINGLE_FAMILY_BINS: tuple[PerturbationBin, ...] = (
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
)

TARGET_ALIGNED_LATERAL_LOAD_BIN: PerturbationBin = "target_aligned_lateral_load"

GRAPH_CHANNEL_BINS: tuple[PerturbationBin, ...] = (
    "command_input",
    "sensory_feedback",
)

PLANT_TIMED_BINS: tuple[PerturbationBin, ...] = ("process_epsilon", "command_input")

CONTROLLER_VISIBLE_TIMED_BINS: tuple[PerturbationBin, ...] = ("sensory_feedback",)

REACH_RELATIVE_LEVELS: dict[str, float] = {
    level.name: float(level.fraction_of_reach)
    for level in load_perturbation_calibration_defaults().reach_relative_levels
}

TRAINING_REACH_RELATIVE_LEVELS: tuple[str, ...] = ("small", "moderate")

EVAL_ONLY_REACH_RELATIVE_LEVELS: tuple[str, ...] = ("stress",)

PROCESS_EPSILON_COMPONENT_FAMILIES: tuple[str, ...] = (
    "process_epsilon_position_xy",
    "process_epsilon_position_xy",
    "process_epsilon_velocity_xy",
    "process_epsilon_velocity_xy",
    "process_epsilon_force_state_xy",
    "process_epsilon_force_state_xy",
    "process_epsilon_integrator_xy",
    "process_epsilon_integrator_xy",
)

TIMING_LABELS_PLANT = tuple(
    bin_.label for bin_ in load_perturbation_calibration_defaults().plant_timing_bins
)

TIMING_LABELS_CONTROLLER_VISIBLE = tuple(
    bin_.label for bin_ in load_perturbation_calibration_defaults().controller_visible_timing_bins
)

BROAD_EPSILON_DIM = 8

BROAD_EPSILON_REFERENCE_REACH_M = float(
    training_preset_value("shared", "broad_epsilon_reference_reach_m")
)

class BroadFullStateEpsilonTrainingConfig(CsPerturbationTrainingConfig):
    """Random full-state epsilon training lane for the C&S analytical game."""

    enabled: bool = training_preset_value("BroadFullStateEpsilonTrainingConfig", "enabled")
    level: str = training_preset_value("BroadFullStateEpsilonTrainingConfig", "level")
    budget_scale: float = training_preset_value("BroadFullStateEpsilonTrainingConfig", "budget_scale")
    reach_length_scaling: bool = training_preset_value("BroadFullStateEpsilonTrainingConfig", "reach_length_scaling")
    nominal_reach_length_m: float = training_preset_value("BroadFullStateEpsilonTrainingConfig", "nominal_reach_length_m")
    movement_epoch_only: bool = training_preset_value("BroadFullStateEpsilonTrainingConfig", "movement_epoch_only")
    epsilon_dim: int = training_preset_value("BroadFullStateEpsilonTrainingConfig", "epsilon_dim")

    @model_validator(mode="after")
    def _validate_config(self) -> "BroadFullStateEpsilonTrainingConfig":
        anchors = load_broad_epsilon_anchors()
        if self.level not in anchors:
            levels = ", ".join(anchors.keys())
            raise ValueError(
                f"Unknown broad-epsilon level {self.level!r}; expected one of {levels}."
            )
        if float(self.budget_scale) <= 0.0:
            raise ValueError("broad epsilon budget_scale must be positive.")
        if float(self.nominal_reach_length_m) <= 0.0:
            raise ValueError("broad epsilon nominal_reach_length_m must be positive.")
        if int(self.epsilon_dim) < 1:
            raise ValueError("broad epsilon epsilon_dim must be positive.")
        return self

    @property
    def level_contract(self) -> dict[str, Any]:
        """Return the immutable analytical budget anchor for this level."""

        return dict(load_broad_epsilon_anchors()[self.level])

    @property
    def reference_l2_radius(self) -> float:
        """Return the 15 cm reference L2 radius after the explicit budget scale."""

        return float(self.level_contract["closed_loop_epsilon_l2_15cm"]) * float(self.budget_scale)

    def to_hps_dict(self) -> dict[str, Any]:
        """Return TreeNamespace-compatible broad-epsilon training metadata."""

        contract = self.level_contract
        return {
            "config": self.model_dump(mode="python"),
            "enabled": self.enabled,
            "mode": ScienceMode.BROAD_EPSILON if self.enabled else "disabled",
            "level": self.level,
            "budget_scale": float(self.budget_scale),
            "reach_length_scaling": bool(self.reach_length_scaling),
            "nominal_reach_length_m": float(self.nominal_reach_length_m),
            "movement_epoch_only": bool(self.movement_epoch_only),
            "epsilon_dim": int(self.epsilon_dim),
            "epsilon_channel": {
                "state_basis": _broad_epsilon_state_basis(int(self.epsilon_dim)),
                "shape": ["batch", "time", int(self.epsilon_dim)],
                "injection": (
                    f"B_w[:{int(self.epsilon_dim)}, :] = I_{int(self.epsilon_dim)}; "
                    f"B_w[{int(self.epsilon_dim)}:, :] = 0"
                ),
                "lag_history_direct_write": False,
                "dt_scaling": "none",
            },
            "sampling": {
                "distribution": "iid_standard_normal",
                "randomized_axes": ["trial", "time", "component"],
                "projection": (
                    "per_trial_flattened_movement_time_component_l2_sphere"
                    if self.movement_epoch_only
                    else "per_trial_flattened_time_component_l2_sphere"
                ),
                "shared_across_batch": False,
            },
            "time_mask": _epsilon_time_mask_contract(self.movement_epoch_only),
            "budget_contract": {
                **contract,
                "reference_reach_m": BROAD_EPSILON_REFERENCE_REACH_M,
                "effective_l2_radius_15cm": self.reference_l2_radius,
                "reach_length_scaling_note": (
                    "Reach scaling is an explicit multi-target normalization choice; "
                    "the original analytical game card reports the 15 cm budget."
                ),
            },
        }


class PgdFullStateEpsilonTrainingConfig(BroadFullStateEpsilonTrainingConfig):
    """Training-time PGD lane on the C&S full-state epsilon channel."""

    enabled: bool = training_preset_value("PgdFullStateEpsilonTrainingConfig", "enabled")
    adversary_mechanism: str = training_preset_value("PgdFullStateEpsilonTrainingConfig", "adversary_mechanism")
    level: str = training_preset_value("PgdFullStateEpsilonTrainingConfig", "level")
    budget_scale: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "budget_scale")
    reach_length_scaling: bool = training_preset_value("PgdFullStateEpsilonTrainingConfig", "reach_length_scaling")
    nominal_reach_length_m: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "nominal_reach_length_m")
    n_steps: int = training_preset_value("PgdFullStateEpsilonTrainingConfig", "n_steps")
    step_size_fraction: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "step_size_fraction")
    inner_optimizer_method: str = training_preset_value("PgdFullStateEpsilonTrainingConfig", "inner_optimizer_method")
    adam_learning_rate: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "adam_learning_rate")
    adam_b1: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "adam_b1")
    adam_b2: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "adam_b2")
    adam_eps: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "adam_eps")
    init: str = training_preset_value("PgdFullStateEpsilonTrainingConfig", "init")
    movement_epoch_only: bool = training_preset_value("PgdFullStateEpsilonTrainingConfig", "movement_epoch_only")
    epsilon_dim: int = training_preset_value("PgdFullStateEpsilonTrainingConfig", "epsilon_dim")
    budget_schedule: str = training_preset_value("PgdFullStateEpsilonTrainingConfig", "budget_schedule")
    sisu_levels: tuple[float, ...] = training_preset_value("PgdFullStateEpsilonTrainingConfig", "sisu_levels")
    sisu_exact_zero_mass: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "sisu_exact_zero_mass")
    sisu_condition_input: str = training_preset_value("PgdFullStateEpsilonTrainingConfig", "sisu_condition_input")
    sisu_max_l2_radius_15cm: float | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "sisu_max_l2_radius_15cm")
    sisu_max_radius_source: str | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "sisu_max_radius_source")
    fixed_l2_radius_15cm: float | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "fixed_l2_radius_15cm")
    fixed_radius_source: str | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "fixed_radius_source")
    objective_kind: str = training_preset_value("PgdFullStateEpsilonTrainingConfig", "objective_kind")
    energy_gamma_star: float | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "energy_gamma_star")
    energy_gamma_factor: float | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "energy_gamma_factor")
    energy_gamma: float | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "energy_gamma")
    energy_penalty_scale: float = training_preset_value("PgdFullStateEpsilonTrainingConfig", "energy_penalty_scale")
    energy_lambda: float | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "energy_lambda")
    safety_cap_l2_radius_15cm: float | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "safety_cap_l2_radius_15cm")
    safety_cap_source: str | None = training_preset_value("PgdFullStateEpsilonTrainingConfig", "safety_cap_source")

    @model_validator(mode="after")
    def _validate_config(self) -> "PgdFullStateEpsilonTrainingConfig":
        if self.adversary_mechanism not in BROAD_EPSILON_PGD_MECHANISMS:
            mechanisms = ", ".join(BROAD_EPSILON_PGD_MECHANISMS)
            raise ValueError(
                f"Unknown PGD adversary mechanism {self.adversary_mechanism!r}; "
                f"expected one of {mechanisms}."
            )
        anchors = load_broad_epsilon_anchors()
        if self.level not in anchors:
            levels = ", ".join(anchors.keys())
            raise ValueError(
                f"Unknown PGD broad-epsilon level {self.level!r}; expected one of {levels}."
            )
        if float(self.budget_scale) <= 0.0:
            raise ValueError("PGD broad epsilon budget_scale must be positive.")
        if float(self.nominal_reach_length_m) <= 0.0:
            raise ValueError("PGD broad epsilon nominal_reach_length_m must be positive.")
        if int(self.n_steps) < 1:
            raise ValueError("PGD broad epsilon n_steps must be positive.")
        if float(self.step_size_fraction) <= 0.0:
            raise ValueError("PGD broad epsilon step_size_fraction must be positive.")
        if self.inner_optimizer_method not in BROAD_EPSILON_PGD_INNER_OPTIMIZER_METHODS:
            methods = ", ".join(BROAD_EPSILON_PGD_INNER_OPTIMIZER_METHODS)
            raise ValueError(f"PGD broad epsilon inner_optimizer_method must be one of {methods}.")
        if float(self.adam_learning_rate) <= 0.0:
            raise ValueError("PGD broad epsilon Adam learning rate must be positive.")
        if not 0.0 <= float(self.adam_b1) < 1.0:
            raise ValueError("PGD broad epsilon Adam b1 must be in [0, 1).")
        if not 0.0 <= float(self.adam_b2) < 1.0:
            raise ValueError("PGD broad epsilon Adam b2 must be in [0, 1).")
        if float(self.adam_eps) <= 0.0:
            raise ValueError("PGD broad epsilon Adam eps must be positive.")
        if int(self.epsilon_dim) < 1:
            raise ValueError("PGD broad epsilon epsilon_dim must be positive.")
        if self.init != "zero":
            raise ValueError("Only zero-initialized PGD broad epsilon is currently supported.")
        if self.budget_schedule not in (
            BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE,
            BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
        ):
            raise ValueError(
                "PGD broad epsilon budget_schedule must be 'fixed' or 'sisu_energy_fraction'."
            )
        if self.budget_schedule == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE:
            _sisu_level_probabilities(self.sisu_levels, self.sisu_exact_zero_mass)
            if self.sisu_condition_input not in ("auto", "input", "sisu"):
                raise ValueError("SISU PGD budget condition input must be auto, input, or sisu.")
            if self.sisu_max_l2_radius_15cm is not None and self.sisu_max_l2_radius_15cm <= 0.0:
                raise ValueError("SISU PGD max L2 radius must be positive when provided.")
            if self.sisu_max_l2_radius_15cm is not None and self.sisu_max_radius_source is None:
                raise ValueError("SISU PGD max L2 radius requires explicit provenance.")
            if self.fixed_l2_radius_15cm is not None:
                raise ValueError("fixed PGD L2 radius is only valid for the fixed budget schedule.")
        if self.fixed_l2_radius_15cm is not None and self.fixed_l2_radius_15cm <= 0.0:
            raise ValueError("fixed PGD L2 radius must be positive when provided.")
        if self.fixed_l2_radius_15cm is not None and self.fixed_radius_source is None:
            raise ValueError("fixed PGD L2 radius requires explicit provenance.")
        if self.objective_kind not in (
            BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
            BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        ):
            raise ValueError("PGD objective_kind must be 'hard_l2' or 'soft_energy'.")
        if self.energy_penalty_scale <= 0.0:
            raise ValueError("PGD soft-energy penalty scale must be positive.")
        if self.energy_gamma_star is not None and self.energy_gamma_star <= 0.0:
            raise ValueError("PGD soft-energy gamma_star must be positive when provided.")
        if self.energy_gamma_factor is not None and self.energy_gamma_factor <= 0.0:
            raise ValueError("PGD soft-energy gamma_factor must be positive when provided.")
        if self.energy_gamma is not None and self.energy_gamma <= 0.0:
            raise ValueError("PGD soft-energy gamma must be positive when provided.")
        if self.energy_lambda is not None and self.energy_lambda <= 0.0:
            raise ValueError("PGD soft-energy lambda must be positive when provided.")
        if self.safety_cap_l2_radius_15cm is not None and self.safety_cap_l2_radius_15cm <= 0.0:
            raise ValueError("PGD soft-energy safety cap radius must be positive when provided.")
        if self.safety_cap_l2_radius_15cm is not None and self.safety_cap_source is None:
            raise ValueError("PGD soft-energy safety cap radius requires explicit provenance.")
        if self.objective_kind == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE:
            if self.soft_energy_lambda is None:
                raise ValueError(
                    "PGD soft-energy objective requires energy_lambda or gamma metadata."
                )
            if (
                self.safety_cap_l2_radius_15cm is None
                and self.adversary_mechanism != BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
            ):
                raise ValueError(
                    "Finite-policy PGD soft-energy objectives require an explicit "
                    "safety-cap radius."
                )
        return self

    @property
    def level_contract(self) -> dict[str, Any]:
        """Return the immutable analytical budget anchor for this level."""

        return dict(load_broad_epsilon_anchors()[self.level])

    @property
    def reference_l2_radius(self) -> float:
        """Return the 15 cm reference L2 radius after the explicit budget scale."""

        if self.fixed_l2_radius_15cm is not None:
            return float(self.fixed_l2_radius_15cm)
        return float(self.level_contract["closed_loop_epsilon_l2_15cm"]) * float(self.budget_scale)

    @property
    def sisu_max_l2_radius(self) -> float:
        """Return the 15 cm max radius for SISU-conditioned PGD budgets."""

        if self.sisu_max_l2_radius_15cm is not None:
            return float(self.sisu_max_l2_radius_15cm)
        return self.reference_l2_radius

    @property
    def soft_energy_gamma(self) -> float | None:
        """Return the soft-energy gamma value when enough metadata is available."""

        if self.energy_gamma is not None:
            return float(self.energy_gamma)
        if self.energy_gamma_star is not None and self.energy_gamma_factor is not None:
            return float(self.energy_gamma_star) * float(self.energy_gamma_factor)
        return None

    @property
    def soft_energy_lambda(self) -> float | None:
        """Return the soft-energy penalty lambda."""

        if self.energy_lambda is not None:
            return float(self.energy_lambda)
        gamma = self.soft_energy_gamma
        if gamma is None:
            return None
        return float(self.energy_penalty_scale) * float(gamma) ** 2

    @property
    def safety_cap_l2_radius(self) -> float:
        """Return the 15 cm trust-region cap used for soft-energy stabilization."""

        if self.safety_cap_l2_radius_15cm is None:
            raise ValueError("PGD soft-energy safety-cap radius must be explicit.")
        return float(self.safety_cap_l2_radius_15cm)

    def to_hps_dict(self) -> dict[str, Any]:
        """Return TreeNamespace-compatible PGD broad-epsilon training metadata."""

        contract = self.level_contract
        budget_schedule = pgd_budget_schedule_contract(self)
        return {
            "config": self.model_dump(mode="python"),
            "enabled": self.enabled,
            "mode": ScienceMode.BROAD_EPSILON_PGD if self.enabled else "disabled",
            "adversary_mechanism": self.adversary_mechanism,
            "level": self.level,
            "budget_scale": float(self.budget_scale),
            "reach_length_scaling": bool(self.reach_length_scaling),
            "nominal_reach_length_m": float(self.nominal_reach_length_m),
            "movement_epoch_only": bool(self.movement_epoch_only),
            "epsilon_dim": int(self.epsilon_dim),
            "budget_schedule": budget_schedule,
            "objective": pgd_objective_contract(self),
            "mechanism": pgd_adversary_mechanism_contract(self),
            "epsilon_channel": {
                "state_basis": _broad_epsilon_state_basis(int(self.epsilon_dim)),
                "shape": ["batch", "time", int(self.epsilon_dim)],
                "injection": (
                    f"B_w[:{int(self.epsilon_dim)}, :] = I_{int(self.epsilon_dim)}; "
                    f"B_w[{int(self.epsilon_dim)}:, :] = 0"
                ),
                "lag_history_direct_write": False,
                "dt_scaling": "none",
            },
            "inner_maximizer": {
                "method": self.inner_optimizer_method,
                "n_steps": int(self.n_steps),
                "step_size_fraction_of_l2_radius": float(self.step_size_fraction),
                "step_size_reference": (
                    "absolute_normalized_gradient_step"
                    if _pgd_cap_free_direct_soft_energy(self)
                    else "fraction_of_active_l2_radius"
                ),
                "learning_rate": float(self.adam_learning_rate),
                "adam": {
                    "learning_rate": float(self.adam_learning_rate),
                    "b1": float(self.adam_b1),
                    "b2": float(self.adam_b2),
                    "eps": float(self.adam_eps),
                },
                "initialization": self.init,
                "projection": _pgd_inner_projection_contract(self),
                "time_mask": _epsilon_time_mask_contract(self.movement_epoch_only),
                "differentiated_through_outer_update": False,
            },
            "safety_cap": pgd_safety_cap_contract(self),
            "time_mask": _epsilon_time_mask_contract(self.movement_epoch_only),
            "budget_contract": {
                **contract,
                "reference_reach_m": BROAD_EPSILON_REFERENCE_REACH_M,
                "effective_l2_radius_15cm": (
                    None if _pgd_cap_free_direct_soft_energy(self) else self.reference_l2_radius
                ),
                "active_max_l2_radius_15cm": _pgd_active_max_l2_radius_15cm(self),
                "radius_bound_mode": _pgd_uses_projection_radius(self),
                "reach_length_scaling_note": (
                    "Reach scaling is an explicit multi-target normalization choice; "
                    "the original analytical game card reports the 15 cm budget."
                ),
                "budget_source": (
                    None if _pgd_cap_free_direct_soft_energy(self) else _pgd_budget_source(self)
                ),
                "scientific_constraint": (
                    (
                        "soft_energy_penalty_cap_free"
                        if _pgd_cap_free_direct_soft_energy(self)
                        else "soft_energy_penalty"
                    )
                    if self.objective_kind == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
                    else "hard_l2_projection"
                ),
            },
        }


class PolicyFullStateEpsilonTrainingConfig(CsPerturbationTrainingConfig):
    """Learned policy lane on the C&S full-state epsilon channel."""

    enabled: bool = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "enabled")
    policy_class: str = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "policy_class")
    mode: str = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "mode")
    width: int = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "width")
    depth: int = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "depth")
    n_steps: int = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "n_steps")
    learning_rate: float = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "learning_rate")
    energy_penalty_gamma: float = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "energy_penalty_gamma")
    reference_l2_radius_15cm: float | None = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "reference_l2_radius_15cm")
    reach_length_scaling: bool = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "reach_length_scaling")
    nominal_reach_length_m: float = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "nominal_reach_length_m")
    movement_epoch_only: bool = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "movement_epoch_only")
    epsilon_dim: int = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "epsilon_dim")
    state_feature_dim: int = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "state_feature_dim")
    budget_source: str | None = training_preset_value("PolicyFullStateEpsilonTrainingConfig", "budget_source")

    @model_validator(mode="after")
    def _validate_config(self) -> "PolicyFullStateEpsilonTrainingConfig":
        if self.policy_class not in POLICY_ADVERSARY_POLICY_CLASSES:
            raise ValueError(
                "Policy adversary policy_class must be one of: "
                f"{', '.join(POLICY_ADVERSARY_POLICY_CLASSES)}."
            )
        if self.mode not in (POLICY_ADVERSARY_PLAIN_MODE, POLICY_ADVERSARY_ENERGY_MODE):
            raise ValueError("Policy adversary mode must be 'plain' or 'energy'.")
        if int(self.width) < 1:
            raise ValueError("Policy adversary width must be positive.")
        if int(self.depth) < 0:
            raise ValueError("Policy adversary depth must be non-negative.")
        if int(self.n_steps) < 1:
            raise ValueError("Policy adversary n_steps must be positive.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("Policy adversary learning_rate must be positive.")
        if float(self.energy_penalty_gamma) < 0.0:
            raise ValueError("Policy adversary energy_penalty_gamma must be non-negative.")
        if self.reference_l2_radius_15cm is not None and self.reference_l2_radius_15cm <= 0.0:
            raise ValueError("Policy adversary reference_l2_radius_15cm must be positive.")
        if self.budget_source is not None and not self.budget_source.strip():
            raise ValueError("Policy adversary budget_source must be non-empty when provided.")
        if self.enabled:
            if self.reference_l2_radius_15cm is None:
                raise ValueError(
                    "Policy adversary training requires explicit reference_l2_radius_15cm."
                )
            if self.budget_source is None:
                raise ValueError("Policy adversary training requires explicit budget_source.")
        if float(self.nominal_reach_length_m) <= 0.0:
            raise ValueError("Policy adversary nominal_reach_length_m must be positive.")
        if int(self.epsilon_dim) < 1:
            raise ValueError("Policy adversary epsilon_dim must be positive.")
        if int(self.state_feature_dim) < 1:
            raise ValueError("Policy adversary state_feature_dim must be positive.")
        return self

    @property
    def reference_l2_radius(self) -> float:
        """Return the active 15 cm reference L2 radius."""

        if self.reference_l2_radius_15cm is None:
            raise ValueError("Policy adversary reference_l2_radius_15cm must be explicit.")
        return float(self.reference_l2_radius_15cm)

    def to_hps_dict(self) -> dict[str, Any]:
        """Return TreeNamespace-compatible policy-adversary metadata."""

        policy = self._policy_metadata()
        reference_l2_radius_15cm = (
            None if self.reference_l2_radius_15cm is None else float(self.reference_l2_radius_15cm)
        )
        budget_source = (
            None
            if self.budget_source is None
            else {
                "key": self.budget_source,
                **load_pgd_radius_source(self.budget_source),
            }
        )
        return {
            "config": self.model_dump(mode="python"),
            "enabled": self.enabled,
            "mode": ScienceMode.POLICY_ADVERSARY if self.enabled else "disabled",
            "row_mode": self.mode,
            "policy_class": self.policy_class,
            "policy": policy,
            "inner_optimizer": {
                "method": "adam",
                "n_ascent_steps_per_controller_step": int(self.n_steps),
                "learning_rate": float(self.learning_rate),
                "weights_persist_across_batches": True,
            },
            "objective": {
                "plain": "maximize controller loss under hard projected epsilon",
                "energy": (
                    "maximize controller loss minus energy_penalty_gamma * epsilon_energy; "
                    "H-infinity-style stabilizer only, not a formal certificate"
                ),
                "active": self.mode,
                "energy_penalty_gamma": float(self.energy_penalty_gamma),
                "formal_certificate": False,
            },
            "epsilon_channel": {
                "state_basis": _broad_epsilon_state_basis(int(self.epsilon_dim)),
                "shape": ["batch", "time", int(self.epsilon_dim)],
                "injection": (
                    f"B_w[:{int(self.epsilon_dim)}, :] = I_{int(self.epsilon_dim)}; "
                    f"B_w[{int(self.epsilon_dim)}:, :] = 0"
                ),
                "lag_history_direct_write": False,
                "dt_scaling": "none",
            },
            "projection": (
                "per_trial_flattened_movement_time_component_l2_ball"
                if self.movement_epoch_only
                else "per_trial_flattened_time_component_l2_ball"
            ),
            "time_mask": _epsilon_time_mask_contract(self.movement_epoch_only),
            "budget_contract": {
                "reference_reach_m": BROAD_EPSILON_REFERENCE_REACH_M,
                "effective_l2_radius_15cm": reference_l2_radius_15cm,
                "active_max_l2_radius_15cm": reference_l2_radius_15cm,
                "budget_source": budget_source,
                "reach_length_scaling": bool(self.reach_length_scaling),
                "reach_length_scaling_note": (
                    "The 15 cm effective PGD radius is scaled by sampled reach length "
                    "when target-relative multi-target training is active."
                ),
            },
            "diagnostics": {
                "epsilon_norm_radius_ratio": True,
                "epsilon_energy": True,
                "projection_boundary_fraction": True,
                "adversary_objective_components": True,
                "controller_loss": True,
                "stabilizer_term": self.mode == POLICY_ADVERSARY_ENERGY_MODE,
            },
        }

    def _policy_metadata(self) -> dict[str, Any]:
        if self.policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP:
            return {
                "kind": POLICY_ADVERSARY_MEMORYLESS_MLP,
                "state_feature_key": "clean_rollout.states.mechanics.vector",
                "closed_loop_finite_policy": False,
                "live_rollout_hook": False,
                "materialization": "legacy_clean_rollout_open_loop_epsilon_sequence",
                "state_feature_dim": int(self.state_feature_dim),
                "width": int(self.width),
                "depth": int(self.depth),
                "output_dim": int(self.epsilon_dim),
                "shared_across_replicates": True,
            }
        return {
            "kind": self.policy_class,
            "parameterization": "shared_time_varying_finite_policy",
            "state_feature_key": (
                "target_centered(clean_rollout.states.mechanics.vector, "
                "trial_specs.targets['mechanics.effector.pos'])"
            ),
            "state_feature_dim": int(self.state_feature_dim),
            "output_dim": int(self.epsilon_dim),
            "shared_across_replicates": True,
            "shared_across_trials_in_batch": True,
            "time_varying": True,
            "has_bias": self.policy_class == AFFINE_POLICY,
            "initialization": "zero",
            "evaluation_semantics": "static_epsilon_materialized_from_clean_rollout_pre_step",
            "closed_loop_semantics_status": "not_live_rollout_hook",
            "closed_loop_semantics_note": (
                "Current Feedbax pre-step integration materializes epsilon before rollout. "
                "The optimizer is Adam over finite-policy parameters, but the rollout does "
                "not call the finite policy on perturbed live states at each time step."
            ),
        }

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable policy-adversary metadata."""

        return self.to_hps_dict()


class MemorylessFullStateEpsilonPolicy(eqx.Module):
    """Small MLP mapping a single state feature vector to one epsilon vector."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        *,
        state_feature_dim: int,
        epsilon_dim: int,
        width: int = 64,
        depth: int = 2,
        key: PRNGKeyArray,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=int(state_feature_dim),
            out_size=int(epsilon_dim),
            width_size=int(width),
            depth=int(depth),
            activation=jax.nn.tanh,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, state_features: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(state_features)


def pgd_budget_schedule_contract(
    config: PgdFullStateEpsilonTrainingConfig,
) -> dict[str, Any]:
    """Return the active PGD budget-schedule metadata."""

    if config.budget_schedule == BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE:
        return {
            "mode": BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE,
            "conditioned": False,
            "mapping_rule": "epsilon_l2_radius = fixed_l2_radius",
        }
    levels = tuple(float(level) for level in config.sisu_levels)
    probabilities = _sisu_level_probabilities(levels, float(config.sisu_exact_zero_mass))
    return {
        "mode": BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
        "conditioned": True,
        "conditioning_scalar": {
            "name": "SISU",
            "input_key": config.sisu_condition_input,
            "input_key_resolution": (
                "auto resolves to trial_specs.inputs['sisu'] when present, otherwise "
                "trial_specs.inputs['input']"
            ),
        },
        "levels": list(levels),
        "probabilities": list(probabilities),
        "exact_zero_mass": float(config.sisu_exact_zero_mass),
        "remaining_nonzero_mass": float(1.0 - config.sisu_exact_zero_mass),
        "nonzero_mass_policy": "uniform_split_over_nonzero_levels",
        "mapping_rule": "epsilon_l2_radius = max_l2_radius_15cm * sqrt(SISU)",
        "mapping": {
            "sisu_interpretation": "energy_fraction",
            "radius_fraction": "sqrt(SISU)",
            "zero_sisu_radius_fraction": 0.0,
            "unit_sisu_radius_fraction": 1.0,
        },
        "max_l2_radius_15cm": config.sisu_max_l2_radius,
        "max_radius_source": _pgd_sisu_max_radius_source(config),
        "controller_internal_mutation": False,
        "teacher_or_distillation": "not_used",
    }


def pgd_objective_contract(
    config: PgdFullStateEpsilonTrainingConfig,
) -> dict[str, Any]:
    """Return metadata for the active PGD inner objective."""

    if config.objective_kind == BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE:
        return {
            "kind": BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
            "formula": "maximize task_loss under hard per-trial L2 projection",
            "hard_l2_projection_is_scientific_constraint": True,
            "soft_energy_penalty": False,
        }
    gamma = config.soft_energy_gamma
    return {
        "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        "formula": "maximize task_loss - lambda * epsilon_energy",
        "hard_l2_projection_is_scientific_constraint": False,
        "soft_energy_penalty": True,
        "gamma_star": config.energy_gamma_star,
        "gamma_factor": config.energy_gamma_factor,
        "gamma": gamma,
        "penalty_scale_c": float(config.energy_penalty_scale),
        "lambda": config.soft_energy_lambda,
        "lambda_mapping": "lambda = c * gamma^2 unless lambda is explicitly supplied",
        "formal_certificate": False,
    }


def pgd_adversary_mechanism_contract(
    config: PgdFullStateEpsilonTrainingConfig,
) -> dict[str, Any]:
    """Return metadata for the selected broad-epsilon PGD adversary mechanism."""

    mechanism = str(config.adversary_mechanism)
    if mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM:
        return {
            "name": mechanism,
            "kind": "open_loop_direct_epsilon_sequence",
            "implementation_status": "implemented",
            "matches_legacy_default": True,
            "semantics": (
                "The inner maximizer writes a full T x epsilon_dim epsilon sequence into "
                "TaskTrialSpec.inputs['epsilon'] before controller rollout."
            ),
        }
    return {
        "name": mechanism,
        "kind": "closed_loop_finite_time_varying_epsilon_policy",
        "implementation_status": "implemented",
        "policy_class": mechanism,
        "supported_objectives": [BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE],
        "runtime_inputs": {
            "base_epsilon": "TaskTrialSpec.inputs['epsilon']",
            "gains": f"TaskTrialSpec.inputs[{FINITE_POLICY_GAINS_INPUT!r}]",
            "bias": (
                f"TaskTrialSpec.inputs[{FINITE_POLICY_BIAS_INPUT!r}]"
                if mechanism == AFFINE_POLICY
                else None
            ),
        },
        "required_policy_contract": {
            "feature_basis": "target_centered_full_state",
            "live_feature_source": "live_perturbed_rollout_state",
            "feature_source_detail": "pre_mechanics_state",
            "time_varying": True,
            "shared_across_trials_in_batch": True,
            "has_bias": mechanism == AFFINE_POLICY,
        },
        "live_evaluation": {
            "implementation": "graph_component",
            "component": FINITE_EPSILON_POLICY_GRAPH_COMPONENT,
            "component_label": FINITE_EPSILON_POLICY_NODE_LABEL,
            "hook": None,
            "input_keys": [
                "epsilon",
                FINITE_POLICY_GAINS_INPUT,
                *([FINITE_POLICY_BIAS_INPUT] if mechanism == AFFINE_POLICY else []),
            ],
            "time_indexing": "policy row t is evaluated before mechanics step t",
            "target_centering": True,
            "static_clean_rollout_materialization": False,
        },
        "graph_component": FINITE_EPSILON_POLICY_GRAPH_COMPONENT,
        "graph_component_label": FINITE_EPSILON_POLICY_NODE_LABEL,
        "no_fake_open_loop_replay": True,
    }


def pgd_safety_cap_contract(
    config: PgdFullStateEpsilonTrainingConfig,
) -> dict[str, Any]:
    """Return metadata for the optional soft-PGD trust-region cap."""

    if config.objective_kind != BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE:
        return {
            "enabled": False,
            "role": "not_used_for_hard_l2_objective",
        }
    if config.safety_cap_l2_radius_15cm is None:
        return {
            "enabled": False,
            "role": "cap_free_soft_energy_no_trust_region",
            "hard_budget_scientific_constraint": False,
            "activation_diagnostics": {
                "epsilon_norm_cap_ratio": False,
                "cap_boundary_fraction": False,
            },
        }
    return {
        "enabled": True,
        "role": "numerical_stabilization_trust_region_only",
        "hard_budget_scientific_constraint": False,
        "l2_radius_15cm": config.safety_cap_l2_radius,
        "reach_length_scaling": bool(config.reach_length_scaling),
        "source": _pgd_safety_cap_source(config),
        "activation_diagnostics": {
            "epsilon_norm_cap_ratio": True,
            "cap_boundary_fraction": True,
        },
    }


def _sisu_level_probabilities(
    levels: tuple[float, ...],
    exact_zero_mass: float,
) -> tuple[float, ...]:
    """Return the exact-zero plus uniform nonzero SISU level probabilities."""

    if not levels:
        raise ValueError("SISU PGD budget schedule requires at least one level.")
    zero_count = sum(np.isclose(level, 0.0) for level in levels)
    if zero_count != 1:
        raise ValueError("SISU PGD budget levels must include exactly one zero level.")
    nonzero_levels = [level for level in levels if not np.isclose(level, 0.0)]
    if not nonzero_levels:
        raise ValueError("SISU PGD budget schedule requires at least one nonzero level.")
    if any(level < 0.0 or level > 1.0 for level in levels):
        raise ValueError("SISU PGD budget levels must lie in [0, 1].")
    if not 0.0 < exact_zero_mass < 1.0:
        raise ValueError("SISU exact-zero mass must lie in (0, 1).")
    nonzero_probability = (1.0 - float(exact_zero_mass)) / len(nonzero_levels)
    return tuple(
        float(exact_zero_mass) if np.isclose(level, 0.0) else float(nonzero_probability)
        for level in levels
    )


def _pgd_cap_free_direct_soft_energy(config: PgdFullStateEpsilonTrainingConfig) -> bool:
    return (
        config.adversary_mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
        and config.objective_kind == BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE
        and config.safety_cap_l2_radius_15cm is None
    )


def _pgd_uses_projection_radius(config: PgdFullStateEpsilonTrainingConfig) -> bool:
    return not _pgd_cap_free_direct_soft_energy(config)


def _pgd_inner_projection_contract(config: PgdFullStateEpsilonTrainingConfig) -> str:
    if _pgd_cap_free_direct_soft_energy(config):
        return "none_cap_free_direct_soft_energy"
    if config.movement_epoch_only:
        return "per_trial_flattened_movement_time_component_l2_ball"
    return "per_trial_flattened_time_component_l2_ball"


def _pgd_active_max_l2_radius_15cm(config: PgdFullStateEpsilonTrainingConfig) -> float | None:
    if _pgd_cap_free_direct_soft_energy(config):
        return None
    if config.budget_schedule == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE:
        return config.sisu_max_l2_radius
    return config.reference_l2_radius


def _pgd_budget_source(config: PgdFullStateEpsilonTrainingConfig) -> dict[str, Any]:
    if config.budget_schedule == BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE:
        return _pgd_sisu_max_radius_source(config)
    return _pgd_fixed_radius_source(config)


def _pgd_sisu_max_radius_source(
    config: PgdFullStateEpsilonTrainingConfig,
) -> dict[str, Any]:
    source_key = config.sisu_max_radius_source
    if source_key is not None:
        return {
            "key": source_key,
            **load_pgd_radius_source(source_key),
        }
    return {
        "key": f"analytical_broad_epsilon_level:{config.level}",
        "source_kind": "analytical_broad_epsilon_anchor",
        "source_issue": config.level_contract.get("source_issue"),
        "source_note": config.level_contract.get("source_note"),
        "gamma_factor": config.level_contract.get("gamma_factor"),
        "gamma_equivalent_analytical_anchor": True,
        "description": f"{config.level} analytical broad-epsilon radius after budget_scale",
    }


def _pgd_fixed_radius_source(
    config: PgdFullStateEpsilonTrainingConfig,
) -> dict[str, Any]:
    """Return provenance for a fixed PGD L2 radius."""

    source_key = config.fixed_radius_source
    if source_key is not None:
        return {
            "key": source_key,
            **load_pgd_radius_source(source_key),
        }
    return {
        "key": f"analytical_broad_epsilon_level:{config.level}",
        "source_kind": "analytical_broad_epsilon_anchor",
        "source_issue": config.level_contract.get("source_issue"),
        "source_note": config.level_contract.get("source_note"),
        "gamma_factor": config.level_contract.get("gamma_factor"),
        "gamma_equivalent_analytical_anchor": True,
        "description": f"{config.level} analytical broad-epsilon radius after budget_scale",
    }


def _pgd_safety_cap_source(
    config: PgdFullStateEpsilonTrainingConfig,
) -> dict[str, Any]:
    """Return provenance for a soft-PGD stabilization cap."""

    source_key = config.safety_cap_source
    if source_key is not None:
        return {
            "key": source_key,
            **load_pgd_radius_source(source_key),
        }
    return {
        "key": "caller_declared_soft_pgd_safety_cap",
        "source_kind": "caller_declared",
        "gamma_equivalent_analytical_anchor": False,
        "description": "explicit soft-PGD trust-region cap",
    }


GRAPH_ADAPTER_SPECS: dict[PerturbationBin, AdditiveGraphChannelAdapterSpec] = {
    "command_input": AdditiveGraphChannelAdapterSpec(
        label="command_input",
        input_key="perturbation_training.command_input",
        target=AdditiveGraphChannelTargetSpec(
            kind="edge",
            source_node="efferent",
            source_port="output",
            target_node="mechanics",
            target_port="force",
        ),
        payload_shape=[2],
        payload_dtype="float32",
        provenance_role="perturbation_training_input",
        metadata={
            "graphspec_mapping": (
                "named additive command_input channel on efferent.output -> mechanics.force"
            )
        },
    ),
    "sensory_feedback": AdditiveGraphChannelAdapterSpec(
        label="sensory_feedback",
        input_key="perturbation_training.sensory_feedback",
        target=AdditiveGraphChannelTargetSpec(
            kind="edge",
            source_node="sensory",
            source_port="output",
            target_node="net",
            target_port="feedback",
        ),
        payload_shape=[4],
        payload_dtype="float32",
        provenance_role="perturbation_training_input",
        metadata={
            "graphspec_mapping": (
                "named additive sensory_feedback channel after sensory noise before net.feedback"
            )
        },
    ),
    "delayed_observation": AdditiveGraphChannelAdapterSpec(
        label="delayed_observation",
        input_key="perturbation_training.delayed_observation",
        target=AdditiveGraphChannelTargetSpec(
            kind="edge",
            source_node="feedback",
            source_port="feedback",
            target_node="sensory",
            target_port="input",
        ),
        payload_shape=[4],
        payload_dtype="float32",
        provenance_role="perturbation_training_input",
        metadata={
            "graphspec_mapping": (
                "named additive delayed_observation channel before sensory.input noise"
            )
        },
    ),
}


def graph_adapter_specs(
    *,
    force_filter_feedback: bool = False,
) -> dict[PerturbationBin, AdditiveGraphChannelAdapterSpec]:
    """Return graph-channel adapter specs for the controller feedback width."""

    if not force_filter_feedback:
        return GRAPH_ADAPTER_SPECS
    return {
        **GRAPH_ADAPTER_SPECS,
        "sensory_feedback": _widen_controller_visible_adapter(
            GRAPH_ADAPTER_SPECS["sensory_feedback"]
        ),
        "delayed_observation": _widen_controller_visible_adapter(
            GRAPH_ADAPTER_SPECS["delayed_observation"]
        ),
    }


def active_graph_adapter_specs(
    *,
    force_filter_feedback: bool = False,
) -> dict[PerturbationBin, AdditiveGraphChannelAdapterSpec]:
    """Return graph adapters for active final-bank perturbation families."""

    specs = graph_adapter_specs(force_filter_feedback=force_filter_feedback)
    return {bin_name: specs[bin_name] for bin_name in GRAPH_CHANNEL_BINS}


def _widen_controller_visible_adapter(
    spec: AdditiveGraphChannelAdapterSpec,
) -> AdditiveGraphChannelAdapterSpec:
    force_filter = resolve_controller_feedback_view(
        None,
        feedback_dim=6,
        source="cs_perturbation_training_widened_adapter",
    ).component(COMPONENT_FORCE_FILTER)
    return spec.model_copy(
        update={
            "payload_shape": [6],
            "metadata": {
                **dict(spec.metadata),
                "force_filter_feedback_payload": "widened_to_controller_feedback_dim",
                "active_calibrated_components": 4,
                "inactive_force_filter_components": list(force_filter.absolute_indices),
            },
        }
    )


class FixedTargetPerturbationTrainingConfig(CsPerturbationTrainingConfig):
    """Mixture and amplitudes for fixed-target C&S GRU perturbation training."""

    enabled: bool = training_preset_value("FixedTargetPerturbationTrainingConfig", "enabled")
    nominal_fraction: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "nominal_fraction")
    single_fraction: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "single_fraction")
    combined_fraction: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "combined_fraction")
    combined_amplitude_scale: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "combined_amplitude_scale")
    initial_position_offset_m: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "initial_position_offset_m")
    initial_velocity_offset_m_s: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "initial_velocity_offset_m_s")
    process_epsilon_scale: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "process_epsilon_scale")
    command_input_pulse_n: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "command_input_pulse_n")
    sensory_feedback_offset_m: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "sensory_feedback_offset_m")
    delayed_observation_offset_m: float = training_preset_value("FixedTargetPerturbationTrainingConfig", "delayed_observation_offset_m")
    pulse_start_step: int = training_preset_value("FixedTargetPerturbationTrainingConfig", "pulse_start_step")
    pulse_duration_steps: int = training_preset_value("FixedTargetPerturbationTrainingConfig", "pulse_duration_steps")
    calibrated_timing: bool = training_preset_value("FixedTargetPerturbationTrainingConfig", "calibrated_timing")
    movement_age_timing: bool = training_preset_value("FixedTargetPerturbationTrainingConfig", "movement_age_timing")
    physical_level: str = training_preset_value("FixedTargetPerturbationTrainingConfig", "physical_level")
    force_filter_feedback: bool = training_preset_value("FixedTargetPerturbationTrainingConfig", "force_filter_feedback")
    calibration_regime: TrainingCalibrationRegime = training_preset_value("FixedTargetPerturbationTrainingConfig", "calibration_regime")
    closed_loop_calibration_table_path: str | None = training_preset_value("FixedTargetPerturbationTrainingConfig", "closed_loop_calibration_table_path")

    @model_validator(mode="after")
    def _validate_config(self) -> "FixedTargetPerturbationTrainingConfig":
        total = self.nominal_fraction + self.single_fraction + self.combined_fraction
        if not np.isclose(total, 1.0):
            raise ValueError(f"Perturbation-training fractions must sum to 1.0; got {total:.6g}.")
        if not 0.40 <= self.nominal_fraction <= 0.50:
            raise ValueError("Nominal perturbation-training fraction must be 40-50%.")
        if not 0.40 <= self.single_fraction <= 0.50:
            raise ValueError("Single-family perturbation-training fraction must be 40-50%.")
        if not 0.05 <= self.combined_fraction <= 0.15:
            raise ValueError("Mild-combined perturbation-training fraction must be 5-15%.")
        if self.combined_amplitude_scale <= 0.0 or self.combined_amplitude_scale > 1.0:
            raise ValueError("Combined perturbation amplitude scale must be in (0, 1].")
        if self.physical_level not in REACH_RELATIVE_LEVELS:
            levels = ", ".join(REACH_RELATIVE_LEVELS)
            raise ValueError(
                f"Unknown perturbation physical level {self.physical_level!r}; "
                f"expected one of {levels}."
            )
        if self.movement_age_timing and not self.calibrated_timing:
            raise ValueError("Movement-age perturbation timing requires calibrated_timing.")
        if self.calibration_regime not in TRAINING_CALIBRATION_REGIMES:
            regimes = ", ".join(TRAINING_CALIBRATION_REGIMES)
            raise ValueError(
                f"Unknown perturbation calibration regime {self.calibration_regime!r}; "
                f"expected one of {regimes}."
            )
        if self.calibration_regime != OPEN_LOOP_ALL_CALIBRATION_REGIME:
            if not self.calibrated_timing:
                raise ValueError("Mixed calibration regimes require calibrated_timing.")
            if not self.closed_loop_calibration_table_path:
                raise ValueError(
                    "Mixed calibration regimes require closed_loop_calibration_table_path."
                )
            _load_closed_loop_calibration_table(str(self.closed_loop_calibration_table_path))
        return self

    @property
    def mode(self) -> str:
        """Return the declared training-mode identifier."""

        if not self.enabled:
            return "nominal"
        if self.calibrated_timing:
            return ScienceMode.PERTURBATION_CALIBRATED
        return ScienceMode.PERTURBATION

    def to_hps_dict(self) -> dict[str, Any]:
        """Return the TreeNamespace-compatible config payload."""

        return {
            "config": self.model_dump(mode="python"),
            "enabled": self.enabled,
            "mode": self.mode,
            "sampling": {
                "kind": (
                    "prng_driven_calibrated_timing"
                    if self.calibrated_timing
                    else "prng_driven_fixed_target"
                ),
                "uses_supplied_key": True,
                "randomized_fields": [
                    "mixture_membership",
                    "single_family",
                    "sign",
                    "axis_or_component",
                    "timing_bin" if self.calibrated_timing else "pulse_start",
                    *(
                        ["movement_start_index"]
                        if self.calibrated_timing and self.movement_age_timing
                        else []
                    ),
                    "physical_level" if self.calibrated_timing else "amplitude_level",
                ],
                "amplitude_levels": (
                    [self.physical_level] if self.calibrated_timing else list(AMPLITUDE_LEVELS)
                ),
                "mild_combined_families": list(MILD_COMBINED_FAMILIES),
            },
            "mixture_semantics": perturbation_training_mixture_semantics(self),
            "nominal_fraction": self.nominal_fraction,
            "single_fraction": self.single_fraction,
            "combined_fraction": self.combined_fraction,
            "combined_amplitude_scale": self.combined_amplitude_scale,
            "calibrated_timing": self.calibrated_timing,
            "movement_age_timing": self.movement_age_timing,
            "physical_level": self.physical_level,
            "force_filter_feedback": self.force_filter_feedback,
            "calibration_regime": self.calibration_regime,
            "closed_loop_calibration_table_path": self.closed_loop_calibration_table_path,
            "calibration_sources": calibration_regime_manifest(self),
            "graph_adapter_payloads": {
                bin_name: {
                    "input_key": spec.input_key,
                    "payload_shape": list(spec.payload_shape or []),
                    "active_calibrated_components": spec.metadata.get(
                        "active_calibrated_components",
                        spec.payload_shape[-1] if spec.payload_shape else None,
                    ),
                }
                for bin_name, spec in active_graph_adapter_specs(
                    force_filter_feedback=self.force_filter_feedback
                ).items()
            },
            "physical_level_fraction_of_reach": REACH_RELATIVE_LEVELS[self.physical_level],
            "training_physical_levels": list(TRAINING_REACH_RELATIVE_LEVELS),
            "eval_only_physical_levels": list(EVAL_ONLY_REACH_RELATIVE_LEVELS),
            "single_family_bins": list(_active_single_family_bins(self)),
            "validation_bins": list(VALIDATION_BINS),
            "inactive_legacy_bins": {
                "bins": list(INACTIVE_LEGACY_PERTURBATION_BINS),
                "reason": (
                    "delayed_observation offsets are redundant with sensory_feedback "
                    "offsets in the current sensory stage and are not sampled or "
                    "validated in the active final perturbation bank"
                ),
                "adapter_support": "preserved_for_legacy_manifests",
            },
            "families": {
                "initial_position": {
                    "channel": "initial_state",
                    "family": "initial_position_offset",
                    "amplitude": self.initial_position_offset_m,
                    "units": "m",
                    "calibration_source": "open_loop",
                },
                "initial_velocity": {
                    "channel": "initial_state",
                    "family": "initial_velocity_offset",
                    "amplitude": self.initial_velocity_offset_m_s,
                    "units": "m/s",
                    "calibration_source": "open_loop",
                },
                "process_epsilon": {
                    "channel": "process_epsilon",
                    "family": "process_epsilon_pulse",
                    "amplitude": self.process_epsilon_scale,
                    "units": "epsilon",
                    "calibration_source": "open_loop",
                },
                "command_input": {
                    "channel": "command_input",
                    "family": "command_input_pulse",
                    "amplitude": self.command_input_pulse_n,
                    "units": "N",
                    "calibration_source": (
                        "closed_loop"
                        if _calibration_uses_closed_loop(self, "command_input_pulse")
                        else "open_loop"
                    ),
                },
                "sensory_feedback": {
                    "channel": "sensory_feedback",
                    "family": "sensory_feedback_offset",
                    "amplitude": self.sensory_feedback_offset_m,
                    "units": "m_or_m_s_channel_units",
                    "calibration_source": (
                        "closed_loop"
                        if _calibration_uses_closed_loop(self, "sensory_feedback_offset")
                        else "open_loop"
                    ),
                },
                **(
                    {
                        "target_aligned_lateral_load": {
                            "channel": "command_input",
                            "family": "target_aligned_lateral_command_load_pulse",
                            "amplitude": "closed_loop_table_by_timing_and_physical_level",
                            "units": "N",
                            "direction": "perpendicular_to_trial_target_direction",
                            "calibration_source": "closed_loop",
                        }
                    }
                    if _calibration_uses_closed_loop(
                        self,
                        "target_aligned_lateral_command_load_pulse",
                    )
                    else {}
                ),
            },
            "pulse": {
                "start_step": self.pulse_start_step,
                "duration_steps": self.pulse_duration_steps,
            },
            "timing_basis": calibrated_timing_basis_manifest(self),
            "timing_bins": calibrated_timing_bins_manifest(self.movement_age_timing),
            "calibrated_amplitude_policy": calibrated_amplitude_policy_manifest(self),
            "target_stream": {
                "status": "not_applicable",
                "reason": "fixed-target C&S GRU does not consume a target-position stream",
            },
            "controller_internal_mutation": False,
        }

    def to_json(self) -> dict[str, Any]:
        """Return run-spec metadata."""

        payload = self.to_hps_dict()
        payload["graph_adapter_inputs"] = {
            bin_name: additive_channel_provenance(
                active_graph_adapter_specs(force_filter_feedback=self.force_filter_feedback)[
                    bin_name
                ],
                adapter="feedbax.additive_channel_adapter",
            )
            for bin_name in GRAPH_CHANNEL_BINS
        }
        return payload


class TargetRelativeMultiTargetTrainingConfig(CsPerturbationTrainingConfig):
    """Structured static-target distribution for target-relative GRU training."""

    enabled: bool = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "enabled")
    force_filter_feedback: bool = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "force_filter_feedback")
    target_support_profile: str = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "target_support_profile")
    seen_directions_deg: tuple[float, ...] = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "seen_directions_deg")
    held_out_directions_deg: tuple[float, ...] = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "held_out_directions_deg")
    seen_amplitudes_m: tuple[float, ...] = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "seen_amplitudes_m")
    held_out_amplitudes_m: tuple[float, ...] = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "held_out_amplitudes_m")
    original_target_anchor_m: tuple[float, float] = training_preset_value("TargetRelativeMultiTargetTrainingConfig", "original_target_anchor_m")
    support_metadata: Any = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_config(self) -> "TargetRelativeMultiTargetTrainingConfig":
        object.__setattr__(
            self,
            "support_metadata",
            _freeze_target_support_metadata(self.support_metadata),
        )
        if self.target_support_profile not in TARGET_SUPPORT_PROFILES:
            profiles = ", ".join(TARGET_SUPPORT_PROFILES)
            raise ValueError(
                f"Unknown target_support_profile {self.target_support_profile!r}; "
                f"expected one of {profiles}."
            )
        if len(self.original_target_anchor_m) != 2:
            raise ValueError("original_target_anchor_m must be a 2D target.")
        if not self.seen_directions_deg:
            raise ValueError("At least one seen target direction is required.")
        if not self.seen_amplitudes_m:
            raise ValueError("At least one seen target amplitude is required.")
        if bool(self.held_out_directions_deg) != bool(self.held_out_amplitudes_m):
            raise ValueError(
                "Held-out directions and amplitudes must both be non-empty or both be empty."
            )
        if any(float(value) <= 0.0 for value in self.seen_amplitudes_m):
            raise ValueError("Seen target amplitudes must be positive.")
        if any(float(value) <= 0.0 for value in self.held_out_amplitudes_m):
            raise ValueError("Held-out target amplitudes must be positive.")
        seen = _target_tuples(self.seen_directions_deg, self.seen_amplitudes_m)
        held_out = _target_tuples(self.held_out_directions_deg, self.held_out_amplitudes_m)
        if tuple(float(x) for x in self.original_target_anchor_m) not in seen:
            raise ValueError(
                "The structured seen target distribution must include the original "
                "15 cm forward target anchor."
            )
        overlap = set(seen).intersection(set(held_out))
        if overlap:
            raise ValueError(f"Seen and held-out target sets overlap: {sorted(overlap)!r}")
        return self

    @property
    def seen_targets_m(self) -> tuple[tuple[float, float], ...]:
        """Return train/seen target positions in metres."""

        return _target_tuples(self.seen_directions_deg, self.seen_amplitudes_m)

    @property
    def held_out_targets_m(self) -> tuple[tuple[float, float], ...]:
        """Return held-out validation target positions in metres."""

        return _target_tuples(self.held_out_directions_deg, self.held_out_amplitudes_m)

    @property
    def validation_targets_m(self) -> tuple[tuple[float, float], ...]:
        """Return validation targets, including the original anchor."""

        return _dedupe_targets(
            (
                tuple(float(x) for x in self.original_target_anchor_m),
                *self.seen_targets_m,
                *self.held_out_targets_m,
            )
        )

    def to_hps_dict(self) -> dict[str, Any]:
        """Return the TreeNamespace-compatible config payload."""

        return {
            "config": self.model_dump(mode="python"),
            "enabled": self.enabled,
            "mode": ScienceMode.TARGET_RELATIVE if self.enabled else "disabled",
            "force_filter_feedback": self.force_filter_feedback,
            "force_filter_feedback_contract": force_filter_feedback_manifest(
                self.force_filter_feedback
            ),
            "input_contract": target_relative_input_contract(
                force_filter_feedback=self.force_filter_feedback
            ),
            "target_distribution": {
                "kind": "structured_static_targets",
                "target_support_profile": self.target_support_profile,
                "support_metadata": _target_support_metadata_to_json(self.support_metadata),
                "original_target_anchor_m": list(self.original_target_anchor_m),
                "seen_directions_deg": list(self.seen_directions_deg),
                "seen_amplitudes_m": list(self.seen_amplitudes_m),
                "held_out_directions_deg": list(self.held_out_directions_deg),
                "held_out_amplitudes_m": list(self.held_out_amplitudes_m),
                "seen_targets_m": [list(row) for row in self.seen_targets_m],
                "held_out_targets_m": [list(row) for row in self.held_out_targets_m],
                "validation_targets_m": [list(row) for row in self.validation_targets_m],
            },
            "validation_bins": target_relative_validation_bins(self),
            "perturbation_mixture_emphasis": target_relative_perturbation_emphasis(),
            "moving_targets": False,
            "teacher_or_jacobian_supervision": False,
            "command_port_comparator": "diagnostic_only",
        }

    def to_json(self) -> dict[str, Any]:
        """Return run-spec metadata."""

        return self.to_hps_dict()


def register_perturbation_training_params_models(*, replace: bool = True) -> None:
    """Register C&S perturbation training config models for run-matrix validation."""

    register_params_model(
        FIXED_TARGET_PERTURBATION_PARAMS_REF,
        FixedTargetPerturbationTrainingConfig,
        replace=replace,
    )
    register_params_model(
        TARGET_RELATIVE_MULTITARGET_PARAMS_REF,
        TargetRelativeMultiTargetTrainingConfig,
        replace=replace,
    )
    register_params_model(
        BROAD_EPSILON_PARAMS_REF, BroadFullStateEpsilonTrainingConfig, replace=replace
    )
    register_params_model(
        BROAD_EPSILON_PGD_PARAMS_REF,
        PgdFullStateEpsilonTrainingConfig,
        replace=replace,
    )
    register_params_model(
        POLICY_ADVERSARY_PARAMS_REF,
        PolicyFullStateEpsilonTrainingConfig,
        replace=replace,
    )


def calibration_regime_manifest(
    config: FixedTargetPerturbationTrainingConfig,
) -> dict[str, Any]:
    """Return the open-loop/closed-loop calibration-source selector contract."""

    closed_loop_families = TRAINING_CALIBRATION_CLOSED_LOOP_FAMILIES[config.calibration_regime]
    all_families = (
        "initial_position_offset",
        "initial_velocity_offset",
        "process_epsilon_pulse",
        "command_input_pulse",
        "sensory_feedback_offset",
        "target_aligned_lateral_command_load_pulse",
    )
    return {
        "schema_version": "rlrmp.cs_perturbation_training_calibration_regime.v1",
        "regime": config.calibration_regime,
        "closed_loop_calibration_table_path": config.closed_loop_calibration_table_path,
        "closed_loop_families": list(closed_loop_families),
        "open_loop_families": [
            family for family in all_families if family not in closed_loop_families
        ],
        "target_aligned_lateral_load_training": (
            "enabled_as_single_family_bin"
            if _calibration_uses_closed_loop(
                config,
                "target_aligned_lateral_command_load_pulse",
            )
            else "not_sampled"
        ),
    }


def calibrated_timing_basis_manifest(
    config: FixedTargetPerturbationTrainingConfig,
) -> dict[str, Any]:
    """Return the calibrated timing basis used by starts in run specs."""

    if not config.movement_age_timing:
        return {
            "mode": "absolute_trial_time",
            "start_index_basis": "trial_start_index",
            "trial_start_index": 0,
            "epoch_source": None,
            "start_time_indices_are": "absolute_trial_indices",
        }
    return {
        "mode": "movement_age",
        "start_index_basis": "movement_start_index",
        "epoch_source": "trial_specs.timeline.epoch_bounds[-2]",
        "undelayed_equivalence": (
            "For one-epoch immediate reaches, epoch_bounds[-2] resolves to 0, so "
            "movement-age starts match the historical absolute starts."
        ),
        "start_time_indices_are": "movement_start_relative_offsets",
        "prep_support": "zero_for_positive_movement_start",
        "movement_onset_state_offsets": {
            "requested_semantics": (
                "initial_position and initial_velocity become movement-onset state-offset "
                "diagnostics in movement-age mode"
            ),
            "implementation": (
                "TaskTrialSpec has trial-start inits but no arbitrary state-set event; "
                "delayed movement-onset offsets are represented as one-step process "
                "epsilon impulses on the matching mechanics.vector position/velocity "
                "component at movement_start."
            ),
            "direct_state_mutation": False,
        },
    }


def calibrated_timing_bins_manifest(movement_age_timing: bool = False) -> dict[str, Any]:
    """Return the calibrated timing-bin contract for training/run specs."""

    defaults = load_perturbation_calibration_defaults()
    plant_timing_bins = defaults.plant_timing_bins
    controller_visible_timing_bins = defaults.controller_visible_timing_bins
    plant_bins = [bin_.to_json() for bin_ in plant_timing_bins]
    visible_bins = [bin_.to_json() for bin_ in controller_visible_timing_bins]
    start_time_kind = (
        "movement_start_relative_offsets" if movement_age_timing else "absolute_trial_indices"
    )
    return {
        "schema_version": "rlrmp.cs_perturbation_calibrated_timing_bins.v1",
        "sampling": "uniform_per_active_family_trial",
        "timing_basis": "movement_age" if movement_age_timing else "absolute_trial_time",
        "start_time_indices_are": start_time_kind,
        "pulse_duration_steps": 5,
        "initial_state": {
            "families": ["initial_position", "initial_velocity"],
            "start_time_index": 0,
            "duration_steps": 1,
            "timing_role": (
                "movement_onset_process_epsilon_impulse"
                if movement_age_timing
                else "initial_condition_not_pulse"
            ),
        },
        "plant_side": {
            "families": list(PLANT_TIMED_BINS),
            "bins": plant_bins,
        },
        "controller_visible": {
            "families": list(CONTROLLER_VISIBLE_TIMED_BINS),
            "bins": visible_bins,
            "inactive_legacy_families": {
                "families": list(INACTIVE_LEGACY_PERTURBATION_BINS),
                "reason": (
                    "delayed_observation offsets duplicate sensory_feedback offsets in "
                    "the current sensory stage and are excluded from active training bins"
                ),
            },
        },
        "family_timing_bins": {
            "initial_position": {
                "start_time_indices": [0],
                "duration_steps": 1,
                "timing_set": "initial_state",
                "start_time_indices_are": start_time_kind,
            },
            "initial_velocity": {
                "start_time_indices": [0],
                "duration_steps": 1,
                "timing_set": "initial_state",
                "start_time_indices_are": start_time_kind,
            },
            **{
                family: {
                    "start_time_indices": [
                        int(bin_.start_time_index) for bin_ in plant_timing_bins
                    ],
                    "duration_steps": 5,
                    "timing_set": "plant_side",
                    "start_time_indices_are": start_time_kind,
                }
                for family in PLANT_TIMED_BINS
            },
            **{
                family: {
                    "start_time_indices": [
                        int(bin_.start_time_index) for bin_ in controller_visible_timing_bins
                    ],
                    "duration_steps": 5,
                    "timing_set": "controller_visible",
                    "start_time_indices_are": start_time_kind,
                }
                for family in CONTROLLER_VISIBLE_TIMED_BINS
            },
        },
    }


def calibrated_level_manifest(
    config: FixedTargetPerturbationTrainingConfig,
) -> dict[str, Any]:
    """Return small/moderate/stress physical-level semantics."""

    levels = {
        name: {
            "fraction_of_reach": fraction,
            "training_role": (
                "training_row" if name in TRAINING_REACH_RELATIVE_LEVELS else "evaluation_only"
            ),
        }
        for name, fraction in REACH_RELATIVE_LEVELS.items()
    }
    return {
        "schema_version": "rlrmp.cs_perturbation_reach_relative_levels.v1",
        "active_level": config.physical_level,
        "active_fraction_of_reach": REACH_RELATIVE_LEVELS[config.physical_level],
        "levels": levels,
        "training_levels": list(TRAINING_REACH_RELATIVE_LEVELS),
        "eval_only_levels": list(EVAL_ONLY_REACH_RELATIVE_LEVELS),
        "amplitude_wiring_status": ("wired_in_sampler_when_calibrated_timing_true"),
    }


def calibrated_amplitude_policy_manifest(
    config: FixedTargetPerturbationTrainingConfig,
) -> dict[str, Any]:
    """Return the calibrated amplitude rule consumed by the training sampler."""

    calibration = load_open_loop_calibration()
    data_product = {
        "role": CALIBRATION_PRODUCT_ROLE,
        "product_schema_id": "rlrmp.perturbation_open_loop_calibration",
        "product_schema_version": "rlrmp.perturbation_open_loop_calibration.v2",
        "product_identity_hash": calibration.product_identity_hash,
        "product_path": CALIBRATION_PRODUCT_RELPATH,
    }
    # The open-loop unit-sensitivity table and controller-visible velocity scale are
    # a governed data product consumed at runtime; artifact_dependency names the
    # runtime dependency and must not claim "none_at_runtime" while calibration is
    # consumed (issue ea6ccb4). A distinct closed-loop calibration table, when used,
    # is still reported as its own path.
    if config.calibration_regime != OPEN_LOOP_ALL_CALIBRATION_REGIME:
        artifact_dependency = config.closed_loop_calibration_table_path
    else:
        artifact_dependency = CALIBRATION_PRODUCT_RELPATH
    return {
        "schema_version": "rlrmp.cs_perturbation_calibrated_amplitude_policy.v1",
        "active": bool(config.calibrated_timing),
        "active_level": config.physical_level,
        "active_fraction_of_reach": REACH_RELATIVE_LEVELS[config.physical_level],
        "plant_side_rule": (
            "amount = reach_length_m * level_fraction / "
            "open_loop_peak_delta_x_per_unit[family,timing]"
        ),
        "initial_state_rule": (
            "position amount = reach_length_m * level_fraction; velocity amount = "
            "reach_length_m * level_fraction / initial_velocity_peak_delta_x_per_unit"
        ),
        "controller_visible_rule": (
            "position components are native reach_length_m * level_fraction offsets; "
            "velocity components are native nominal_peak_speed_m_s * level_fraction offsets"
        ),
        "controller_visible_velocity_scale_m_s": (
            calibration.controller_visible_velocity_scale_m_s
        ),
        "open_loop_peak_delta_x_per_unit": calibration.peak_delta_x_per_unit,
        "data_product": data_product,
        "calibration_regime": calibration_regime_manifest(config),
        "command_input_training_direction_policy": {
            "distribution": "uniform_random_2d_direction",
            "amplitude_interpretation": "calibrated value is the 2D vector norm",
            "applies_to": "randomized training sampler only",
            "deterministic_bank_rows": (
                "validation and diagnostic banks may still enumerate cardinal or "
                "target-aligned lateral directions for interpretability"
            ),
        },
        "amplitude_level_randomization": (
            "disabled in calibrated_timing mode; the declared physical_level fixes the "
            "effect-size target"
        ),
        "artifact_dependency": artifact_dependency,
    }


def perturbation_training_mixture_semantics(
    config: FixedTargetPerturbationTrainingConfig,
) -> dict[str, Any]:
    """Return explicit fixed-target perturbation-training sampling semantics."""

    return {
        "schema_version": "rlrmp.cs_perturbation_training_mixture_semantics.v2",
        "mode": config.mode,
        "experimental_factor_note": (
            "Perturbation uncertainty level is an experimental factor distinct from "
            "physical perturbation amplitude. Broader randomized families, signs, "
            "components, timings, or mixtures can induce robustness rather than "
            "only testing ordinary feedback control."
        ),
        "calibration_note": (
            "Calibrated timing mode samples timing bins uniformly by family. "
            "Small/moderate/stress physical-effect levels are reach-relative; "
            "calibrated training consumes the checked-in unit-sensitivity table "
            "to set plant-side amplitudes per family, timing bin, and reach length."
        ),
        "calibrated_levels": calibrated_level_manifest(config),
        "calibrated_amplitude_policy": calibrated_amplitude_policy_manifest(config),
        "timing_basis": calibrated_timing_basis_manifest(config),
        "timing_bins": calibrated_timing_bins_manifest(config.movement_age_timing),
        "membership": {
            "unit": "per training trial",
            "nominal_fraction": float(config.nominal_fraction),
            "single_family_fraction": float(config.single_fraction),
            "mild_combined_fraction": float(config.combined_fraction),
            "nominal": "no explicit perturbation beyond ordinary stochastic runtime",
            "single_family": (
                "one family sampled uniformly from single_family_bins, then "
                "family-specific component/sign/timing/level randomization is applied"
            ),
            "mild_combined": (
                "initial_position and command_input are both active, scaled by "
                "combined_amplitude_scale; the other families are inactive"
            ),
        },
        "single_family_bins": list(_active_single_family_bins(config)),
        "mild_combined_families": list(MILD_COMBINED_FAMILIES),
        "inactive_legacy_bins": list(INACTIVE_LEGACY_PERTURBATION_BINS),
        "amplitude_levels": list(AMPLITUDE_LEVELS),
        "families": {
            "initial_position": {
                "base_amplitude": float(config.initial_position_offset_m),
                "units": "m",
                "emission": (
                    "offset one random mechanics.vector position component among x/y "
                    "at movement_start using a one-step process-epsilon impulse by "
                    "sign * reach_length * physical_level_fraction"
                    if config.movement_age_timing
                    else "offset one random mechanics.vector position component among x/y "
                    "at t=0 by sign * reach_length * physical_level_fraction"
                    if config.calibrated_timing
                    else (
                        "offset one random mechanics.vector position component among x/y "
                        "at t=0 by sign * amplitude_level * base_amplitude"
                    )
                ),
                "randomized": [
                    "axis",
                    "sign",
                    "physical_level" if config.calibrated_timing else "amplitude_level",
                ],
            },
            "initial_velocity": {
                "base_amplitude": float(config.initial_velocity_offset_m_s),
                "units": "m/s",
                "emission": (
                    "offset one random mechanics.vector velocity component among x/y "
                    "at movement_start using a one-step process-epsilon impulse by "
                    "sign * reach_length * physical_level_fraction / "
                    "initial_velocity_peak_delta_x_per_unit"
                    if config.movement_age_timing
                    else "offset one random mechanics.vector velocity component among x/y "
                    "at t=0 by sign * reach_length * physical_level_fraction / "
                    "initial_velocity_peak_delta_x_per_unit"
                    if config.calibrated_timing
                    else (
                        "offset one random mechanics.vector velocity component among x/y "
                        "at t=0 by sign * amplitude_level * base_amplitude"
                    )
                ),
                "randomized": [
                    "axis",
                    "sign",
                    "physical_level" if config.calibrated_timing else "amplitude_level",
                ],
            },
            "process_epsilon": {
                "base_amplitude": float(config.process_epsilon_scale),
                "units": "epsilon",
                "emission": (
                    "add a duration-limited pulse to one random epsilon component over "
                    "a calibrated timing bin"
                    if config.calibrated_timing
                    else (
                        "add a duration-limited pulse to one random epsilon component "
                        "over a random start time"
                    )
                ),
                "randomized": [
                    "epsilon_component",
                    "timing_bin" if config.calibrated_timing else "start_time",
                    "sign",
                    "physical_level" if config.calibrated_timing else "amplitude_level",
                ],
                "duration_steps": int(config.pulse_duration_steps),
            },
            "command_input": {
                "base_amplitude": float(config.command_input_pulse_n),
                "units": "N",
                "emission": (
                    "add a duration-limited pulse in one uniform random 2D command "
                    "direction over a calibrated timing bin; calibrated amount is "
                    "the vector norm"
                    if config.calibrated_timing
                    else (
                        "add a duration-limited pulse in one uniform random 2D command "
                        "direction over a random start time; base amount is the vector norm"
                    )
                ),
                "randomized": [
                    "command_direction",
                    "timing_bin" if config.calibrated_timing else "start_time",
                    "physical_level" if config.calibrated_timing else "amplitude_level",
                ],
                "duration_steps": int(config.pulse_duration_steps),
            },
            "sensory_feedback": {
                "base_amplitude": float(config.sensory_feedback_offset_m),
                "units": "m_or_m_s_channel_units",
                "emission": (
                    "add an offset pulse on one random 4D sensory-feedback component; "
                    "calibrated timing mode uses controller-visible 5-step bins"
                    if config.calibrated_timing
                    else (
                        "add an offset pulse on one random 4D sensory-feedback "
                        "component; current training uses full-trial duration"
                    )
                ),
                "randomized": [
                    "feedback_component",
                    "timing_bin" if config.calibrated_timing else "start_time",
                    "sign",
                    "physical_level" if config.calibrated_timing else "amplitude_level",
                ],
                "duration_steps": (
                    int(config.pulse_duration_steps) if config.calibrated_timing else "full_trial"
                ),
            },
            **(
                {
                    "target_aligned_lateral_load": {
                        "base_amplitude": "closed_loop_table_by_timing_and_physical_level",
                        "units": "N",
                        "emission": (
                            "add a target-relative lateral command-channel pulse over a "
                            "calibrated timing bin; direction is perpendicular to the "
                            "trial reach vector and sign is randomized"
                        ),
                        "randomized": [
                            "timing_bin",
                            "sign",
                            "physical_level",
                            "target_direction",
                        ],
                        "duration_steps": int(config.pulse_duration_steps),
                    }
                }
                if _calibration_uses_closed_loop(
                    config,
                    "target_aligned_lateral_command_load_pulse",
                )
                else {}
            ),
        },
        "validation_difference": (
            "Validation bins are deterministic family-separated probes, not a replay "
            "of the full training mixture. They expose separate nominal, "
            "single-family, and mild-combined bins for checkpoint selection/reporting."
        ),
    }


def _epsilon_time_mask_contract(movement_epoch_only: bool) -> dict[str, Any]:
    """Return run-spec metadata for broad-epsilon time support."""

    if not movement_epoch_only:
        return {
            "mode": "full_trial",
            "movement_epoch_only": False,
        }
    return {
        "mode": "movement_epoch_only",
        "movement_epoch_only": True,
        "epoch_source": "trial_specs.timeline.epoch_bounds[-2:]",
        "prep_support": "zero",
        "projection_support": "movement_epoch_time_component_axes_only",
    }


def _broad_epsilon_state_basis(epsilon_dim: int) -> str:
    state_dim = int(epsilon_dim) * 6
    return f"C&S {state_dim}D delay-augmented LinearStateSpace state"


def target_relative_validation_bins(
    config: TargetRelativeMultiTargetTrainingConfig,
) -> list[dict[str, Any]]:
    """Return target validation-bin rows."""

    anchor = tuple(float(x) for x in config.original_target_anchor_m)
    seen = config.seen_targets_m
    held_out = config.held_out_targets_m
    validation_targets = config.validation_targets_m
    seen_held_out_targets = _dedupe_targets((*seen, *held_out))
    return [
        {
            "bin": "original_target_nominal",
            "target_role": "anchor",
            "targets_m": [list(anchor)],
        },
        {
            "bin": "seen_multitarget_nominal",
            "target_role": "seen_training_support",
            "targets_m": [list(row) for row in seen],
        },
        {
            "bin": "held_out_multitarget_nominal",
            "target_role": "held_out_validation_support",
            "targets_m": [list(row) for row in held_out],
        },
        {
            "bin": "initial_position_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["initial_position"],
        },
        {
            "bin": "initial_velocity_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["initial_velocity"],
        },
        {
            "bin": "sensory_feedback_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["sensory_feedback"],
        },
        {
            "bin": "process_load_epsilon",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["process_epsilon"],
        },
        {
            "bin": "mild_combined",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["initial_position", "sensory_feedback", "process_epsilon"],
        },
        {
            "bin": "command_input_diagnostic",
            "target_role": "diagnostic_only",
            "targets_m": [list(row) for row in validation_targets],
            "families": ["command_input"],
            "checkpoint_selection": "excluded_unless_comparator_defined",
        },
    ]


def target_relative_input_contract(*, force_filter_feedback: bool = False) -> dict[str, Any]:
    """Return the documented controller-visible target-relative sign contract."""

    sign_convention = [
        "target_x - delayed_x",
        "target_y - delayed_y",
        "-delayed_vx",
        "-delayed_vy",
    ]
    if force_filter_feedback:
        sign_convention.extend(["delayed_force_filter_x", "delayed_force_filter_y"])
    return {
        "controller_feedback_basis": (
            "target_relative_delayed_feedback_plus_force_filter"
            if force_filter_feedback
            else "target_relative_delayed_feedback"
        ),
        "static_target_input": "known_immediately_not_visually_delayed",
        "sign_convention": sign_convention,
        "shape": [6 if force_filter_feedback else 4],
        "force_filter_feedback": force_filter_feedback_manifest(force_filter_feedback),
        "moving_targets": "out_of_scope",
    }


def force_filter_feedback_manifest(enabled: bool) -> dict[str, Any]:
    """Return metadata for the optional force/filter proprioceptive channel."""

    return {
        "enabled": bool(enabled),
        "added_coordinates": (
            ["delayed_force_filter_x", "delayed_force_filter_y"] if enabled else []
        ),
        "source": "states.mechanics.vector delayed physical block indices 44:46",
        "disturbance_integrators_exposed": False,
        "rationale": (
            "Expose a proprioception-like delayed force/filter state without exposing "
            "the C&S disturbance integrators."
            if enabled
            else "Baseline target-relative feedback exposes delayed position error and velocity."
        ),
    }


def target_relative_perturbation_emphasis() -> dict[str, Any]:
    """Return the target-relative perturbation-mixture design intent."""

    return {
        "nominal_multitarget_fraction": [0.50, 0.70],
        "initial_position_velocity_fraction": [0.10, 0.20],
        "sensory_feedback_fraction": [0.10, 0.20],
        "inactive_legacy_families": list(INACTIVE_LEGACY_PERTURBATION_BINS),
        "process_or_load_fraction": [0.05, 0.15],
        "command_input": "optional_diagnostic_only",
    }


def _active_single_family_bins(
    config: FixedTargetPerturbationTrainingConfig,
) -> tuple[PerturbationBin, ...]:
    if _calibration_uses_closed_loop(config, "target_aligned_lateral_command_load_pulse"):
        return (*SINGLE_FAMILY_BINS, TARGET_ALIGNED_LATERAL_LOAD_BIN)
    return SINGLE_FAMILY_BINS


def _calibration_uses_closed_loop(
    config: FixedTargetPerturbationTrainingConfig,
    family: str,
) -> bool:
    return family in TRAINING_CALIBRATION_CLOSED_LOOP_FAMILIES[config.calibration_regime]


def _load_closed_loop_calibration_table(path: str) -> dict[str, Any]:
    table_path = Path(path).expanduser()
    if not table_path.is_absolute():
        table_path = Path.cwd() / table_path
    try:
        payload = json.loads(table_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Closed-loop calibration table not found: {path}") from exc
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Closed-loop calibration table {path!r} has no rows list.")
    return payload


def _target_tuples(
    directions_deg: tuple[float, ...],
    amplitudes_m: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    rows: list[tuple[float, float]] = []
    for amplitude in amplitudes_m:
        for direction in directions_deg:
            radians = np.deg2rad(float(direction))
            rows.append(
                (
                    round(float(amplitude) * float(np.cos(radians)), 12),
                    round(float(amplitude) * float(np.sin(radians)), 12),
                )
            )
    return _dedupe_targets(tuple(rows))


def _freeze_target_support_metadata(value: Any) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, list | tuple) and all(_is_metadata_pair(item) for item in value):
        return tuple((str(key), _freeze_metadata_value(item_value)) for key, item_value in value)
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_metadata_value(item_value)) for key, item_value in value.items()
            )
        )
    if hasattr(value, "__dict__"):
        return tuple(
            sorted(
                (str(key), _freeze_metadata_value(item_value))
                for key, item_value in vars(value).items()
                if not key.startswith("_")
            )
        )
    raise TypeError(f"Unsupported target support metadata type: {type(value).__name__}")


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping) or hasattr(value, "__dict__"):
        return _freeze_target_support_metadata(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata_value(item) for item in value)
    return value


def _target_support_metadata_to_json(value: Any) -> dict[str, Any]:
    frozen = _freeze_target_support_metadata(value)
    return {key: _metadata_value_to_json(item_value) for key, item_value in frozen}


def _metadata_value_to_json(value: Any) -> Any:
    if value and isinstance(value, tuple) and all(_is_metadata_pair(item) for item in value):
        return {key: _metadata_value_to_json(item_value) for key, item_value in value}
    if isinstance(value, tuple):
        return [_metadata_value_to_json(item) for item in value]
    return value


def _is_metadata_pair(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) == 2 and isinstance(value[0], str)


def _dedupe_targets(targets: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    seen: set[tuple[float, float]] = set()
    rows: list[tuple[float, float]] = []
    for target in targets:
        key = (round(float(target[0]), 12), round(float(target[1]), 12))
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return tuple(rows)


CS_POSITION_SCALE = 1e6

CS_VELOCITY_SCALE = 1e5

CS_CONTROL_SCALE = 1.0

DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW = "canonical_window"

MINIMAX_PARAMS_REF = "rlrmp/minimax/v1"

GUIDED_DISTILLATION_PARAMS_REF = "rlrmp/guided_distillation/v1"

CLOSED_LOOP_DISTILLATION_PARAMS_REF = "rlrmp/closed_loop_distillation/v1"


class GuidedDistillationConfig(BaseModel):
    """Unified authoring config for guided distillation CLI/spec surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str | None = training_preset_value("GuidedDistillationConfig", "run_id")
    run_spec: str | None = training_preset_value("GuidedDistillationConfig", "run_spec")
    run_spec_output: str | None = training_preset_value("GuidedDistillationConfig", "run_spec_output")
    output_dir: str | None = training_preset_value("GuidedDistillationConfig", "output_dir")
    teacher_package: str | None = training_preset_value("GuidedDistillationConfig", "teacher_package")
    teacher_manifest: str | None = training_preset_value("GuidedDistillationConfig", "teacher_manifest")
    teacher_gains_key: str | None = training_preset_value("GuidedDistillationConfig", "teacher_gains_key")
    clean_action_weight: float = training_preset_value("GuidedDistillationConfig", "clean_action_weight")
    perturbation_response_weight: float = training_preset_value("GuidedDistillationConfig", "perturbation_response_weight")
    input_output_jvp_weight: float = training_preset_value("GuidedDistillationConfig", "input_output_jvp_weight")
    rollout_anchor_weight: float = training_preset_value("GuidedDistillationConfig", "rollout_anchor_weight")
    n_jvp_directions: int = Field(training_preset_value("GuidedDistillationConfig", "n_jvp_directions"), gt=0)
    n_batches: int = Field(training_preset_value("GuidedDistillationConfig", "n_batches"), gt=0)
    batch_size: int = Field(training_preset_value("GuidedDistillationConfig", "batch_size"), gt=0)
    n_replicates: int = Field(training_preset_value("GuidedDistillationConfig", "n_replicates"), gt=0)
    hidden_size: int = Field(training_preset_value("GuidedDistillationConfig", "hidden_size"), gt=0)
    horizon: int = Field(training_preset_value("GuidedDistillationConfig", "horizon"), gt=0)
    seed: int = training_preset_value("GuidedDistillationConfig", "seed")
    controller_lr: float = Field(training_preset_value("GuidedDistillationConfig", "controller_lr"), gt=0.0)
    lr_warmup_batches: int = Field(training_preset_value("GuidedDistillationConfig", "lr_warmup_batches"), ge=0)
    lr_warmup_init_fraction: float = Field(training_preset_value("GuidedDistillationConfig", "lr_warmup_init_fraction"), ge=0.0)
    lr_cosine_alpha: float = Field(training_preset_value("GuidedDistillationConfig", "lr_cosine_alpha"), ge=0.0)
    gradient_clip_norm: float = Field(training_preset_value("GuidedDistillationConfig", "gradient_clip_norm"), gt=0.0)
    trainable_dtype: str = training_preset_value("GuidedDistillationConfig", "trainable_dtype")
    population_mask_mode: str = training_preset_value("GuidedDistillationConfig", "population_mask_mode")
    log_step: int = Field(training_preset_value("GuidedDistillationConfig", "log_step"), gt=0)
    checkpoint: bool = training_preset_value("GuidedDistillationConfig", "checkpoint")
    checkpoint_interval_batches: int = Field(training_preset_value("GuidedDistillationConfig", "checkpoint_interval_batches"), gt=0)
    stop_after_batches: int | None = Field(training_preset_value("GuidedDistillationConfig", "stop_after_batches"), gt=0)
    smoke_loss: bool = training_preset_value("GuidedDistillationConfig", "smoke_loss")
    smoke_train: bool = training_preset_value("GuidedDistillationConfig", "smoke_train")
    full_train: bool = training_preset_value("GuidedDistillationConfig", "full_train")
    resume: bool = training_preset_value("GuidedDistillationConfig", "resume")
    dry_run: bool = training_preset_value("GuidedDistillationConfig", "dry_run")


class ClosedLoopDistillationConfig(BaseModel):
    """Unified authoring config for closed-loop distillation CLI/spec surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str | None = training_preset_value("ClosedLoopDistillationConfig", "run_id")
    run_spec: Path | None = training_preset_value("ClosedLoopDistillationConfig", "run_spec")
    run_spec_output: Path | None = training_preset_value("ClosedLoopDistillationConfig", "run_spec_output")
    output_dir: str | None = training_preset_value("ClosedLoopDistillationConfig", "output_dir")
    teacher_package: str | None = training_preset_value("ClosedLoopDistillationConfig", "teacher_package")
    teacher_gains_key: str | None = training_preset_value("ClosedLoopDistillationConfig", "teacher_gains_key")
    horizon: int = Field(training_preset_value("ClosedLoopDistillationConfig", "horizon"), gt=0)
    seed: int = training_preset_value("ClosedLoopDistillationConfig", "seed")
    n_replicates: int = Field(training_preset_value("ClosedLoopDistillationConfig", "n_replicates"), gt=0)
    hidden_size: int = Field(training_preset_value("ClosedLoopDistillationConfig", "hidden_size"), gt=0)
    batch_size: int = Field(training_preset_value("ClosedLoopDistillationConfig", "batch_size"), gt=0)
    n_batches: int = Field(training_preset_value("ClosedLoopDistillationConfig", "n_batches"), gt=0)
    controller_lr: float = Field(training_preset_value("ClosedLoopDistillationConfig", "controller_lr"), gt=0.0)
    lr_warmup_batches: int = Field(training_preset_value("ClosedLoopDistillationConfig", "lr_warmup_batches"), ge=0)
    lr_cosine_alpha: float = Field(training_preset_value("ClosedLoopDistillationConfig", "lr_cosine_alpha"), ge=0.0)
    gradient_clip_norm: float = Field(training_preset_value("ClosedLoopDistillationConfig", "gradient_clip_norm"), gt=0.0)
    checkpoint_interval_batches: int = Field(training_preset_value("ClosedLoopDistillationConfig", "checkpoint_interval_batches"), gt=0)
    trainable_dtype: str = training_preset_value("ClosedLoopDistillationConfig", "trainable_dtype")
    kinematics_trajectory_weight: float = training_preset_value("ClosedLoopDistillationConfig", "kinematics_trajectory_weight")
    velocity_weight: float = training_preset_value("ClosedLoopDistillationConfig", "velocity_weight")
    endpoint_weight: float = training_preset_value("ClosedLoopDistillationConfig", "endpoint_weight")
    settling_weight: float = training_preset_value("ClosedLoopDistillationConfig", "settling_weight")
    action_force_weight: float = training_preset_value("ClosedLoopDistillationConfig", "action_force_weight")
    perturbation_response_weight: float = training_preset_value("ClosedLoopDistillationConfig", "perturbation_response_weight")
    input_output_jvp_weight: float = training_preset_value("ClosedLoopDistillationConfig", "input_output_jvp_weight")
    task_rollout_loss_weight: float = training_preset_value("ClosedLoopDistillationConfig", "task_rollout_loss_weight")
    write_run_spec: bool = training_preset_value("ClosedLoopDistillationConfig", "write_run_spec")
    dry_run: bool = training_preset_value("ClosedLoopDistillationConfig", "dry_run")
    smoke_preflight: bool = training_preset_value("ClosedLoopDistillationConfig", "smoke_preflight")
    smoke_train: bool = training_preset_value("ClosedLoopDistillationConfig", "smoke_train")
    smoke_n_batches: int = Field(training_preset_value("ClosedLoopDistillationConfig", "smoke_n_batches"), gt=0)
    smoke_batch_size: int = Field(training_preset_value("ClosedLoopDistillationConfig", "smoke_batch_size"), gt=0)
    full_train: bool = training_preset_value("ClosedLoopDistillationConfig", "full_train")
    confirm_full_train: bool = training_preset_value("ClosedLoopDistillationConfig", "confirm_full_train")
    resume: bool = training_preset_value("ClosedLoopDistillationConfig", "resume")
    user_confirmed: bool = training_preset_value("ClosedLoopDistillationConfig", "user_confirmed")


class MinimaxConfig(BaseModel):
    """Flat minimax method config whose fields own all authoring defaults."""

    model_config = ConfigDict(extra="forbid")

    n_warmup_batches: int = Field(training_preset_value("MinimaxConfig", "n_warmup_batches"), ge=0)
    n_adversary_batches: int = Field(training_preset_value("MinimaxConfig", "n_adversary_batches"), ge=0)
    n_adversary_steps: int = Field(training_preset_value("MinimaxConfig", "n_adversary_steps"), gt=0)
    batch_size: int = Field(training_preset_value("MinimaxConfig", "batch_size"), gt=0)
    adv_batch_size: int | None = training_preset_value("MinimaxConfig", "adv_batch_size")
    n_replicates: int = Field(training_preset_value("MinimaxConfig", "n_replicates"), gt=0)
    seed: int = training_preset_value("MinimaxConfig", "seed")

    controller_lr: float = training_preset_value("MinimaxConfig", "controller_lr")
    adversary_lr: float = training_preset_value("MinimaxConfig", "adversary_lr")
    loss_update_enabled: bool = training_preset_value("MinimaxConfig", "loss_update_enabled")
    loss_update_ratio: float = training_preset_value("MinimaxConfig", "loss_update_ratio")

    adversary_type: Literal["gaussian_bump", "linear_dynamics"] = training_preset_value("MinimaxConfig", "adversary_type")
    n_adversaries: int = Field(training_preset_value("MinimaxConfig", "n_adversaries"), gt=0)
    n_bumps: int = training_preset_value("MinimaxConfig", "n_bumps")
    force_max: float = training_preset_value("MinimaxConfig", "force_max")
    linear_dynamics_eta_max: float = training_preset_value("MinimaxConfig", "linear_dynamics_eta_max")
    linear_dynamics_pgd_steps: int = training_preset_value("MinimaxConfig", "linear_dynamics_pgd_steps")
    linear_dynamics_lr: float = training_preset_value("MinimaxConfig", "linear_dynamics_lr")

    hidden_type: Literal["gru", "vanilla_rnn", "linear", "linear_tracker"] = training_preset_value("MinimaxConfig", "hidden_type")
    sisu_gating: Literal["additive", "multiplicative"] = training_preset_value("MinimaxConfig", "sisu_gating")

    nn_output: float = training_preset_value("MinimaxConfig", "nn_output")
    nn_hidden: float = training_preset_value("MinimaxConfig", "nn_hidden")
    nn_hidden_derivative: float = training_preset_value("MinimaxConfig", "nn_hidden_derivative")
    nn_output_jerk: float = training_preset_value("MinimaxConfig", "nn_output_jerk")
    nn_output_pre_go: float = training_preset_value("MinimaxConfig", "nn_output_pre_go")
    nn_hidden_derivative_pre_go: float = training_preset_value("MinimaxConfig", "nn_hidden_derivative_pre_go")
    effector_hold_pos: float = training_preset_value("MinimaxConfig", "effector_hold_pos")
    effector_hold_vel: float = training_preset_value("MinimaxConfig", "effector_hold_vel")
    effector_final_vel: float = training_preset_value("MinimaxConfig", "effector_final_vel")
    effector_vel_late: float = training_preset_value("MinimaxConfig", "effector_vel_late")
    effector_pos_running: float = training_preset_value("MinimaxConfig", "effector_pos_running")
    effector_pos_late_weight: float = training_preset_value("MinimaxConfig", "effector_pos_late_weight")
    effector_pos_late_final_scale: float = training_preset_value("MinimaxConfig", "effector_pos_late_final_scale")
    effector_pos_late_start_step: int = training_preset_value("MinimaxConfig", "effector_pos_late_start_step")

    effector_pos_running_schedule: Literal["flat", "powerlaw", "movement_ramp"] = training_preset_value("MinimaxConfig", "effector_pos_running_schedule")
    effector_hold_pos_schedule: Literal["flat", "powerlaw"] = training_preset_value("MinimaxConfig", "effector_hold_pos_schedule")
    position_powerlaw_power: float = training_preset_value("MinimaxConfig", "position_powerlaw_power")
    movement_ramp_shape: Literal["linear", "cosine", "power"] = training_preset_value("MinimaxConfig", "movement_ramp_shape")
    movement_ramp_duration_steps: int = training_preset_value("MinimaxConfig", "movement_ramp_duration_steps")
    movement_ramp_power: float = training_preset_value("MinimaxConfig", "movement_ramp_power")

    p_catch_trial: float = training_preset_value("MinimaxConfig", "p_catch_trial")

    warmup_model: str | None = training_preset_value("MinimaxConfig", "warmup_model")
    output_dir: str
    spec_dir: str | None = training_preset_value("MinimaxConfig", "spec_dir")
    jax_cache_dir: str | None = training_preset_value("MinimaxConfig", "jax_cache_dir")
    jax_explain_cache_misses: bool = training_preset_value("MinimaxConfig", "jax_explain_cache_misses")
    allow_x64: bool = training_preset_value("MinimaxConfig", "allow_x64")
    checkpoint: bool = training_preset_value("MinimaxConfig", "checkpoint")
    checkpoint_every: int = training_preset_value("MinimaxConfig", "checkpoint_every")
    resume: bool = training_preset_value("MinimaxConfig", "resume")
    allow_fresh_start: bool = training_preset_value("MinimaxConfig", "allow_fresh_start")
    fused: bool = training_preset_value("MinimaxConfig", "fused")
    streaming_loss: bool = training_preset_value("MinimaxConfig", "streaming_loss")

    @model_validator(mode="after")
    def _validate_config(self) -> "MinimaxConfig":
        if self.adversary_type == "linear_dynamics" and not self.fused:
            raise ValueError("linear_dynamics minimax requires fused execution")
        return self


def register_native_training_params_models(*, replace: bool = True) -> None:
    """Register native trainer configs for typed run-matrix authoring."""

    register_params_model(MINIMAX_PARAMS_REF, MinimaxConfig, replace=replace)
    register_params_model(
        GUIDED_DISTILLATION_PARAMS_REF,
        GuidedDistillationConfig,
        replace=replace,
    )
    register_params_model(
        CLOSED_LOOP_DISTILLATION_PARAMS_REF,
        ClosedLoopDistillationConfig,
        replace=replace,
    )


register_native_training_params_models()


class CsNominalGruConfig(BaseModel):
    """Flat nominal-GRU trainer config whose fields own all authoring defaults."""

    model_config = ConfigDict(extra="forbid")

    run_spec: str | None = training_preset_value("CsNominalGruConfig", "run_spec")
    compact_run_spec: bool = Field(training_preset_value("CsNominalGruConfig", "compact_run_spec"), exclude=True)
    output_dir: str
    spec_dir: str | None = training_preset_value("CsNominalGruConfig", "spec_dir")
    issue: str
    seed: int = training_preset_value("CsNominalGruConfig", "seed")
    controller_architecture: Literal[
        "gru",
        "time_constrained_free_gain",
        "linear_recurrence",
        "static_linear",
    ] = training_preset_value("CsNominalGruConfig", "controller_architecture")

    n_train_batches: int = Field(training_preset_value("CsNominalGruConfig", "n_train_batches"), ge=0)
    batch_size: int = Field(training_preset_value("CsNominalGruConfig", "batch_size"), gt=0)
    controller_lr: float = Field(training_preset_value("CsNominalGruConfig", "controller_lr"), gt=0.0)
    lr_warmup_batches: int = Field(training_preset_value("CsNominalGruConfig", "lr_warmup_batches"), ge=0)
    lr_warmup_init_fraction: float = Field(training_preset_value("CsNominalGruConfig", "lr_warmup_init_fraction"), ge=0.0)
    lr_cosine_alpha: float = Field(training_preset_value("CsNominalGruConfig", "lr_cosine_alpha"), ge=0.0)
    gradient_clip_norm: float | None = training_preset_value("CsNominalGruConfig", "gradient_clip_norm")
    n_replicates: int = Field(training_preset_value("CsNominalGruConfig", "n_replicates"), gt=0)
    hidden_size: int = Field(training_preset_value("CsNominalGruConfig", "hidden_size"), gt=0)

    plant_backend: Literal["cs_lss", "legacy_causal_simplefeedback"] = training_preset_value("CsNominalGruConfig", "plant_backend")
    no_integrator_state: bool = training_preset_value("CsNominalGruConfig", "no_integrator_state")
    stochastic_preset: Literal["cs2019-rollout"] = training_preset_value("CsNominalGruConfig", "stochastic_preset")
    target_m: float = training_preset_value("CsNominalGruConfig", "target_m")
    n_input_only: int = Field(training_preset_value("CsNominalGruConfig", "n_input_only"), ge=0)
    n_readout_only: int = Field(training_preset_value("CsNominalGruConfig", "n_readout_only"), ge=0)
    n_recurrent_only: int = Field(training_preset_value("CsNominalGruConfig", "n_recurrent_only"), ge=0)

    effector_pos_running: float = training_preset_value("CsNominalGruConfig", "effector_pos_running")
    effector_vel_running: float = training_preset_value("CsNominalGruConfig", "effector_vel_running")
    effector_terminal_pos: float = training_preset_value("CsNominalGruConfig", "effector_terminal_pos")
    effector_terminal_vel: float = training_preset_value("CsNominalGruConfig", "effector_terminal_vel")
    effector_final_vel: float = training_preset_value("CsNominalGruConfig", "effector_final_vel")
    nn_output: float = training_preset_value("CsNominalGruConfig", "nn_output")
    nn_output_jerk: float = training_preset_value("CsNominalGruConfig", "nn_output_jerk")
    nn_output_pre_go: float | None = training_preset_value("CsNominalGruConfig", "nn_output_pre_go")
    delayed_pre_go_force_filter_hold: float = training_preset_value("CsNominalGruConfig", "delayed_pre_go_force_filter_hold")
    delayed_pre_go_start_pos_hold: float = training_preset_value("CsNominalGruConfig", "delayed_pre_go_start_pos_hold")
    delayed_pre_go_start_pos_hold_norm: Literal["l2", "l1"] = training_preset_value("CsNominalGruConfig", "delayed_pre_go_start_pos_hold_norm")
    delayed_pre_go_zero_vel_hold: float = training_preset_value("CsNominalGruConfig", "delayed_pre_go_zero_vel_hold")
    loss_objective: Literal[
        "partial_feedbax_terms",
        "partial_net_output_force_filter",
        "full_analytical_qrf",
    ] = training_preset_value("CsNominalGruConfig", "loss_objective")
    regularized_fidelity: bool = training_preset_value("CsNominalGruConfig", "regularized_fidelity")

    perturbation_training: bool | None = training_preset_value("CsNominalGruConfig", "perturbation_training")
    perturbation_nominal_fraction: float = training_preset_value("CsNominalGruConfig", "perturbation_nominal_fraction")
    perturbation_single_fraction: float = training_preset_value("CsNominalGruConfig", "perturbation_single_fraction")
    perturbation_combined_fraction: float = training_preset_value("CsNominalGruConfig", "perturbation_combined_fraction")
    perturbation_combined_amplitude_scale: float = training_preset_value("CsNominalGruConfig", "perturbation_combined_amplitude_scale")
    perturbation_initial_position_offset_m: float = training_preset_value("CsNominalGruConfig", "perturbation_initial_position_offset_m")
    perturbation_initial_velocity_offset_m_s: float = training_preset_value("CsNominalGruConfig", "perturbation_initial_velocity_offset_m_s")
    perturbation_process_epsilon_scale: float = training_preset_value("CsNominalGruConfig", "perturbation_process_epsilon_scale")
    perturbation_command_input_pulse_n: float = training_preset_value("CsNominalGruConfig", "perturbation_command_input_pulse_n")
    perturbation_sensory_feedback_offset_m: float = training_preset_value("CsNominalGruConfig", "perturbation_sensory_feedback_offset_m")
    perturbation_delayed_observation_offset_m: float = training_preset_value("CsNominalGruConfig", "perturbation_delayed_observation_offset_m")
    perturbation_pulse_start_step: int = training_preset_value("CsNominalGruConfig", "perturbation_pulse_start_step")
    perturbation_pulse_duration_steps: int = training_preset_value("CsNominalGruConfig", "perturbation_pulse_duration_steps")
    perturbation_calibrated_timing: bool | None = training_preset_value("CsNominalGruConfig", "perturbation_calibrated_timing")
    perturbation_movement_age_timing: bool | None = training_preset_value("CsNominalGruConfig", "perturbation_movement_age_timing")
    perturbation_physical_level: Literal["small", "moderate", "stress"] | None = training_preset_value("CsNominalGruConfig", "perturbation_physical_level")
    perturbation_calibration_regime: Literal[
        "open_loop_all",
        "closed_loop_sensory",
        "closed_loop_sensory_command_lateral",
    ] = training_preset_value("CsNominalGruConfig", "perturbation_calibration_regime")
    perturbation_closed_loop_calibration_table: str | None = training_preset_value("CsNominalGruConfig", "perturbation_closed_loop_calibration_table")

    target_relative_multitarget: bool = training_preset_value("CsNominalGruConfig", "target_relative_multitarget")
    target_support_profile: Literal[
        "old_020a65b",
        "const_dense_all",
        "const_sparse8",
        "const_band8",
        "const_band16",
        "const_band36",
    ] = training_preset_value("CsNominalGruConfig", "target_support_profile")
    delayed_reach: bool = training_preset_value("CsNominalGruConfig", "delayed_reach")
    delayed_reach_go_cue_min_step: int = training_preset_value("CsNominalGruConfig", "delayed_reach_go_cue_min_step")
    delayed_reach_go_cue_max_step: int = training_preset_value("CsNominalGruConfig", "delayed_reach_go_cue_max_step")
    delayed_reach_p_catch_trial: float = training_preset_value("CsNominalGruConfig", "delayed_reach_p_catch_trial")
    delayed_movement_cost_tail_mode: Literal[
        "canonical_window",
        "flat_after_canonical_horizon",
    ] = training_preset_value("CsNominalGruConfig", "delayed_movement_cost_tail_mode")
    delayed_reach_trial_type_normalized_loss: bool = training_preset_value("CsNominalGruConfig", "delayed_reach_trial_type_normalized_loss")
    delayed_reach_no_catch_qrf_weight: float = training_preset_value("CsNominalGruConfig", "delayed_reach_no_catch_qrf_weight")
    delayed_reach_catch_qrf_weight: float = training_preset_value("CsNominalGruConfig", "delayed_reach_catch_qrf_weight")
    force_filter_feedback: bool | None = training_preset_value("CsNominalGruConfig", "force_filter_feedback")

    broad_epsilon_training: bool = training_preset_value("CsNominalGruConfig", "broad_epsilon_training")
    broad_epsilon_pgd_training: bool = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_training")
    broad_epsilon_level: Literal["moderate", "strong"] = training_preset_value("CsNominalGruConfig", "broad_epsilon_level")
    broad_epsilon_budget_scale: float = training_preset_value("CsNominalGruConfig", "broad_epsilon_budget_scale")
    broad_epsilon_pgd_fixed_radius_15cm: float | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_fixed_radius_15cm")
    broad_epsilon_pgd_fixed_radius_source: str | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_fixed_radius_source")
    broad_epsilon_reach_scaling: bool = training_preset_value("CsNominalGruConfig", "broad_epsilon_reach_scaling")
    broad_epsilon_pgd_steps: int = Field(training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_steps"), gt=0)
    broad_epsilon_pgd_step_size_fraction: float = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_step_size_fraction")
    broad_epsilon_pgd_inner_optimizer_method: Literal[
        "projected_gradient_ascent",
        "adam",
    ] = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_inner_optimizer_method")
    broad_epsilon_pgd_adam_lr: float = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_adam_lr")
    broad_epsilon_pgd_adam_b1: float = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_adam_b1")
    broad_epsilon_pgd_adam_b2: float = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_adam_b2")
    broad_epsilon_pgd_adam_eps: float = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_adam_eps")
    broad_epsilon_pgd_mechanism: Literal[
        "direct_epsilon",
        "linear_no_bias",
        "affine",
    ] = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_mechanism")
    broad_epsilon_pgd_objective: Literal["hard_l2", "soft_energy"] = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_objective")
    broad_epsilon_pgd_energy_gamma_star: float | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_energy_gamma_star")
    broad_epsilon_pgd_energy_gamma_factor: float | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_energy_gamma_factor")
    broad_epsilon_pgd_energy_gamma: float | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_energy_gamma")
    broad_epsilon_pgd_energy_penalty_scale: float = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_energy_penalty_scale")
    broad_epsilon_pgd_energy_lambda: float | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_energy_lambda")
    broad_epsilon_pgd_safety_cap_15cm: float | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_safety_cap_15cm")
    broad_epsilon_pgd_safety_cap_source: str | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_safety_cap_source")
    broad_epsilon_pgd_budget_schedule: Literal["fixed", "sisu_energy_fraction"] = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_budget_schedule")
    broad_epsilon_pgd_sisu_condition_input: Literal["auto", "input", "sisu"] = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_sisu_condition_input")
    broad_epsilon_pgd_sisu_max_radius: float | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_sisu_max_radius")
    broad_epsilon_pgd_sisu_max_radius_source: str | None = training_preset_value("CsNominalGruConfig", "broad_epsilon_pgd_sisu_max_radius_source")

    adaptive_epsilon_curriculum: bool = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_curriculum")
    adaptive_epsilon_controller_training_mode: Literal[
        "loss_blend",
        "epsilon_scaled_outer_training",
    ] = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_controller_training_mode")
    adaptive_epsilon_damage_start: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_damage_start")
    adaptive_epsilon_damage_peak: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_damage_peak")
    adaptive_epsilon_damage_final: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_damage_final")
    adaptive_epsilon_damage_ramp_batches: int = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_damage_ramp_batches")
    adaptive_epsilon_damage_anneal_batches: int = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_damage_anneal_batches")
    adaptive_epsilon_update_interval_batches: int = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_update_interval_batches")
    adaptive_epsilon_ema_alpha: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_ema_alpha")
    adaptive_epsilon_eta: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_eta")
    adaptive_epsilon_deadband_frac: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_deadband_frac")
    adaptive_epsilon_hysteresis_frac: float | None = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_hysteresis_frac")
    adaptive_epsilon_freeze_during_application_ramp: bool = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_freeze_during_application_ramp")
    adaptive_epsilon_gain_normalization: bool = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_gain_normalization")
    adaptive_epsilon_gain_ema_alpha: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_gain_ema_alpha")
    adaptive_epsilon_gain_min: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_gain_min")
    adaptive_epsilon_gain_max: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_gain_max")
    adaptive_epsilon_lambda_min: float | None = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_lambda_min")
    adaptive_epsilon_lambda_max: float | None = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_lambda_max")
    adaptive_epsilon_max_log_step: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_max_log_step")
    adaptive_epsilon_outer_weight_start: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_outer_weight_start")
    adaptive_epsilon_outer_weight_final: float = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_outer_weight_final")
    adaptive_epsilon_outer_weight_ramp_batches: int = training_preset_value("CsNominalGruConfig", "adaptive_epsilon_outer_weight_ramp_batches")

    policy_adversary_training: bool = training_preset_value("CsNominalGruConfig", "policy_adversary_training")
    policy_adversary_policy_class: Literal["memoryless_mlp", "linear_no_bias", "affine"] = (
        training_preset_value("CsNominalGruConfig", "policy_adversary_policy_class")
    )
    policy_adversary_mode: Literal["plain", "energy"] = training_preset_value("CsNominalGruConfig", "policy_adversary_mode")
    policy_adversary_width: int = Field(training_preset_value("CsNominalGruConfig", "policy_adversary_width"), gt=0)
    policy_adversary_depth: int = Field(training_preset_value("CsNominalGruConfig", "policy_adversary_depth"), ge=0)
    policy_adversary_steps: int = Field(training_preset_value("CsNominalGruConfig", "policy_adversary_steps"), gt=0)
    policy_adversary_lr: float = training_preset_value("CsNominalGruConfig", "policy_adversary_lr")
    policy_adversary_energy_gamma: float = training_preset_value("CsNominalGruConfig", "policy_adversary_energy_gamma")
    policy_adversary_radius_15cm: float | None = training_preset_value("CsNominalGruConfig", "policy_adversary_radius_15cm")
    policy_adversary_radius_source: str | None = training_preset_value("CsNominalGruConfig", "policy_adversary_radius_source")

    initial_hidden_encoder: bool = training_preset_value("CsNominalGruConfig", "initial_hidden_encoder")
    smoke: bool = training_preset_value("CsNominalGruConfig", "smoke")
    full_train: bool = training_preset_value("CsNominalGruConfig", "full_train")
    resume: bool = training_preset_value("CsNominalGruConfig", "resume")
    allow_fresh_start: bool = training_preset_value("CsNominalGruConfig", "allow_fresh_start")
    stop_after_batches: int | None = training_preset_value("CsNominalGruConfig", "stop_after_batches")
    training_diagnostics: bool = training_preset_value("CsNominalGruConfig", "training_diagnostics")
    checkpoint_interval_batches: int = training_preset_value("CsNominalGruConfig", "checkpoint_interval_batches")
    log_step: int = training_preset_value("CsNominalGruConfig", "log_step")
    disable_progress: bool = training_preset_value("CsNominalGruConfig", "disable_progress")
    quiet_progress: bool = training_preset_value("CsNominalGruConfig", "quiet_progress")
    allow_x64: bool = training_preset_value("CsNominalGruConfig", "allow_x64")
    dry_run: bool = training_preset_value("CsNominalGruConfig", "dry_run")

    @model_validator(mode="after")
    def _validate_config(self) -> "CsNominalGruConfig":
        if self.delayed_reach_go_cue_min_step < 0:
            raise ValueError("delayed_reach_go_cue_min_step must be nonnegative")
        if self.delayed_reach_go_cue_max_step < self.delayed_reach_go_cue_min_step:
            raise ValueError(
                "delayed_reach_go_cue_max_step must be >= delayed_reach_go_cue_min_step"
            )
        if not 0.0 <= self.delayed_reach_p_catch_trial <= 1.0:
            raise ValueError("delayed_reach_p_catch_trial must be between 0 and 1")
        if self.n_input_only + self.n_readout_only + self.n_recurrent_only > self.hidden_size:
            raise ValueError("population subgroup counts must not exceed hidden_size")
        return self


def validate_training_config(config_type: type[BaseModel], payload: Any) -> Any:
    """Validate current authoring or read a frozen pre-canonical runtime payload."""

    if isinstance(payload, config_type):
        return payload
    raw = payload if isinstance(payload, Mapping) else vars(payload)
    canonical = raw.get("config")
    if canonical is None:
        extra_keys = set(raw).difference(config_type.model_fields)
        if extra_keys:
            canonical = migrate_frozen_rendered_training_payload(
                config_type.__name__,
                raw,
                field_names=frozenset(config_type.model_fields),
            )
            if canonical is None:
                canonical = raw
        else:
            canonical = raw
    if not isinstance(canonical, Mapping):
        canonical = vars(canonical)
    return config_type.model_validate(canonical)


def target_relative_target_support_config(
    *,
    profile: str = DEFAULT_TARGET_SUPPORT_PROFILE,
    enabled: bool = False,
    force_filter_feedback: bool = False,
) -> TargetRelativeMultiTargetTrainingConfig:
    """Return a target-relative config for a named finite target-support profile."""

    profile = str(profile)
    if profile == TARGET_SUPPORT_PROFILE_020A65B:
        return TargetRelativeMultiTargetTrainingConfig(
            enabled=enabled,
            force_filter_feedback=force_filter_feedback,
            target_support_profile=profile,
            support_metadata={
                "role": "exact_020a65b_no_pgd_h0_replay_target_support",
                "reach_length_policy": "mixed_seen_0p10_0p15_held_out_0p12_0p18",
                "direction_policy": "six_seen_spokes_plus_diagonal_held_out_validation",
            },
        )

    if profile == TARGET_SUPPORT_PROFILE_CONST_DENSE_ALL:
        return TargetRelativeMultiTargetTrainingConfig(
            enabled=enabled,
            force_filter_feedback=force_filter_feedback,
            target_support_profile=profile,
            seen_directions_deg=_uniform_directions_deg(TARGET_SUPPORT_DENSE_N_DIRECTIONS),
            held_out_directions_deg=(),
            seen_amplitudes_m=(TARGET_SUPPORT_CONST_REACH_M,),
            held_out_amplitudes_m=(),
            support_metadata={
                "role": "fixed_reach_dense_all_angle_training",
                "constant_reach_m": TARGET_SUPPORT_CONST_REACH_M,
                "dense_n_directions": TARGET_SUPPORT_DENSE_N_DIRECTIONS,
                "held_out_policy": "none",
            },
        )

    if profile == TARGET_SUPPORT_PROFILE_CONST_SPARSE8:
        seen = _uniform_directions_deg(TARGET_SUPPORT_SPARSE_N_DIRECTIONS)
        held_out = _directions_not_in(
            _uniform_directions_deg(TARGET_SUPPORT_DENSE_N_DIRECTIONS),
            seen,
        )
        return TargetRelativeMultiTargetTrainingConfig(
            enabled=enabled,
            force_filter_feedback=force_filter_feedback,
            target_support_profile=profile,
            seen_directions_deg=seen,
            held_out_directions_deg=held_out,
            seen_amplitudes_m=(TARGET_SUPPORT_CONST_REACH_M,),
            held_out_amplitudes_m=(TARGET_SUPPORT_CONST_REACH_M,),
            support_metadata={
                "role": "fixed_reach_sparse_8_direction_training_dense_validation",
                "constant_reach_m": TARGET_SUPPORT_CONST_REACH_M,
                "seen_n_directions": TARGET_SUPPORT_SPARSE_N_DIRECTIONS,
                "validation_grid_n_directions": TARGET_SUPPORT_DENSE_N_DIRECTIONS,
            },
        )

    band_counts = {
        TARGET_SUPPORT_PROFILE_CONST_BAND8: TARGET_SUPPORT_BAND8_HELD_OUT_DIRECTIONS,
        TARGET_SUPPORT_PROFILE_CONST_BAND16: TARGET_SUPPORT_BAND16_HELD_OUT_DIRECTIONS,
        TARGET_SUPPORT_PROFILE_CONST_BAND36: TARGET_SUPPORT_BAND36_HELD_OUT_DIRECTIONS,
    }
    if profile in band_counts:
        held_out_count = band_counts[profile]
        seen, held_out, directions_per_band = _split_directions_by_held_out_band_count(
            n_directions=TARGET_SUPPORT_DENSE_N_DIRECTIONS,
            centers_deg=TARGET_SUPPORT_BAND_CENTERS_DEG,
            held_out_count=held_out_count,
        )
        return TargetRelativeMultiTargetTrainingConfig(
            enabled=enabled,
            force_filter_feedback=force_filter_feedback,
            target_support_profile=profile,
            seen_directions_deg=seen,
            held_out_directions_deg=held_out,
            seen_amplitudes_m=(TARGET_SUPPORT_CONST_REACH_M,),
            held_out_amplitudes_m=(TARGET_SUPPORT_CONST_REACH_M,),
            support_metadata={
                "role": "fixed_reach_dense_training_with_held_out_angular_bands",
                "constant_reach_m": TARGET_SUPPORT_CONST_REACH_M,
                "validation_grid_n_directions": TARGET_SUPPORT_DENSE_N_DIRECTIONS,
                "held_out_direction_count": held_out_count,
                "held_out_band_count": len(TARGET_SUPPORT_BAND_CENTERS_DEG),
                "held_out_band_centers_deg": list(TARGET_SUPPORT_BAND_CENTERS_DEG),
                "held_out_directions_per_band": directions_per_band,
                "direction_grid_spacing_deg": 360.0 / TARGET_SUPPORT_DENSE_N_DIRECTIONS,
            },
        )

    profiles = ", ".join(TARGET_SUPPORT_PROFILES)
    raise ValueError(f"Unknown target support profile {profile!r}; expected one of {profiles}.")


def _uniform_directions_deg(n_directions: int) -> tuple[float, ...]:
    if int(n_directions) <= 0:
        raise ValueError("n_directions must be positive.")
    return tuple(round(360.0 * index / int(n_directions), 12) for index in range(n_directions))


def _directions_not_in(
    directions_deg: tuple[float, ...],
    excluded_deg: tuple[float, ...],
) -> tuple[float, ...]:
    excluded = {_angle_key(direction) for direction in excluded_deg}
    return tuple(direction for direction in directions_deg if _angle_key(direction) not in excluded)


def _split_directions_by_held_out_band_count(
    *,
    n_directions: int,
    centers_deg: tuple[float, ...],
    held_out_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    if not centers_deg:
        raise ValueError("At least one held-out band center is required.")
    directions = _uniform_directions_deg(n_directions)
    held_out_count = int(held_out_count)
    if not (0 < held_out_count < len(directions)):
        raise ValueError("held_out_count must be between zero and n_directions.")
    if held_out_count % len(centers_deg):
        raise ValueError("held_out_count must divide evenly across held-out band centers.")

    directions_per_band = held_out_count // len(centers_deg)
    held_out_keys: set[float] = set()
    for center in centers_deg:
        selected_for_center = 0
        candidates = sorted(
            directions,
            key=lambda direction: (
                _circular_abs_delta_deg(direction, center),
                _circular_signed_delta_deg(direction, center),
            ),
        )
        for direction in candidates:
            key = _angle_key(direction)
            if key in held_out_keys:
                continue
            held_out_keys.add(key)
            selected_for_center += 1
            if selected_for_center >= directions_per_band:
                break

    if len(held_out_keys) != held_out_count:
        raise ValueError(
            f"Held-out band split produced {len(held_out_keys)} directions, "
            f"expected {held_out_count}."
        )
    held_out = tuple(
        direction for direction in directions if _angle_key(direction) in held_out_keys
    )
    seen = _directions_not_in(directions, held_out)
    if not seen or not held_out:
        raise ValueError("Held-out band split must produce non-empty seen and held-out sets.")
    return seen, held_out, directions_per_band


def _circular_abs_delta_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def _circular_signed_delta_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def _angle_key(angle_deg: float) -> float:
    return round(float(angle_deg) % 360.0, 12)


# The runtime module historically re-exported constants and models from this
# family. Include private contract helpers required by that internal surface.
__all__ = [name for name in globals() if not name.startswith("__")]

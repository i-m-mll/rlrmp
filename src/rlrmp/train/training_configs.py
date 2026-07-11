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
        """Validate a flat config or normalize a historical serialized payload."""

        return validate_training_config(cls, payload)


PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_randomized"

CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_calibrated_timing"

LEGACY_PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_generalized"

TARGET_RELATIVE_MULTITARGET_TRAINING_MODE = "target_relative_multitarget_static"

TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE = "target_relative_multitarget_static_h0"

BROAD_EPSILON_TRAINING_MODE = "broad_full_state_epsilon_l2"

BROAD_EPSILON_PGD_TRAINING_MODE = "broad_full_state_epsilon_pgd_l2"

POLICY_ADVERSARY_TRAINING_MODE = "broad_full_state_epsilon_policy_l2"

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

DEFAULT_PGD_SISU_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

DEFAULT_PGD_SISU_EXACT_ZERO_MASS = 0.30

RAW_STRONG_GAMMA_1P05_RADIUS_15CM = 0.0023284905801002004

HISTORICAL_020A65B_PGD_RADIUS_15CM = 0.004545500088363065

MILD_COMBINED_FAMILIES: tuple["PerturbationBin", ...] = (
    "initial_position",
    "command_input",
)

AMPLITUDE_LEVELS: tuple[float, ...] = (0.5, 1.0)

ORIGINAL_TARGET_ANCHOR_M: tuple[float, float] = (0.15, 0.0)

DEFAULT_SEEN_TARGET_DIRECTIONS_DEG: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)

DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG: tuple[float, ...] = (30.0, 150.0, 210.0, 330.0)

DEFAULT_SEEN_TARGET_AMPLITUDES_M: tuple[float, ...] = (0.10, 0.15)

DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M: tuple[float, ...] = (0.12, 0.18)

TARGET_SUPPORT_PROFILE_020A65B = "old_020a65b"

TARGET_SUPPORT_PROFILE_CONST_DENSE_ALL = "const_dense_all"

TARGET_SUPPORT_PROFILE_CONST_SPARSE8 = "const_sparse8"

TARGET_SUPPORT_PROFILE_CONST_BAND8 = "const_band8"

TARGET_SUPPORT_PROFILE_CONST_BAND16 = "const_band16"

TARGET_SUPPORT_PROFILE_CONST_BAND36 = "const_band36"

DEFAULT_TARGET_SUPPORT_PROFILE = TARGET_SUPPORT_PROFILE_CONST_BAND16

TARGET_SUPPORT_PROFILES: tuple[str, ...] = (
    TARGET_SUPPORT_PROFILE_020A65B,
    TARGET_SUPPORT_PROFILE_CONST_DENSE_ALL,
    TARGET_SUPPORT_PROFILE_CONST_SPARSE8,
    TARGET_SUPPORT_PROFILE_CONST_BAND8,
    TARGET_SUPPORT_PROFILE_CONST_BAND16,
    TARGET_SUPPORT_PROFILE_CONST_BAND36,
)

TARGET_SUPPORT_CONST_REACH_M = 0.15

TARGET_SUPPORT_DENSE_N_DIRECTIONS = 72

TARGET_SUPPORT_SPARSE_N_DIRECTIONS = 8

TARGET_SUPPORT_BAND_CENTERS_DEG: tuple[float, ...] = (45.0, 135.0, 225.0, 315.0)

TARGET_SUPPORT_BAND8_HELD_OUT_DIRECTIONS = 8

TARGET_SUPPORT_BAND16_HELD_OUT_DIRECTIONS = 16

TARGET_SUPPORT_BAND36_HELD_OUT_DIRECTIONS = 36

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

BROAD_EPSILON_REFERENCE_REACH_M = 0.15

PGD_SISU_MAX_RADIUS_SOURCES: dict[str, dict[str, Any]] = {
    "raw_strong_gamma_1p05_radius": {
        "source_kind": "raw_analytical_gamma_anchor",
        "source_issue": "a7dad8a",
        "source_note": "results/a7dad8a/notes/adversary_equivalence_manifest.json",
        "gamma_factor": 1.05,
        "gamma_equivalent_analytical_anchor": True,
        "description": "raw strong gamma-1.05 analytical radius",
    },
    "effective_020a65b_pgd_training_radius": {
        "source_kind": "historical_replay_effective_pgd_training_radius",
        "source_issue": "020a65b",
        "source_note": "020a65b broad-epsilon PGD local training contract",
        "gamma_equivalent_analytical_anchor": False,
        "description": (
            "historical 020a65b PGD replay radius; not a current default or "
            "new gamma-equivalent analytical anchor"
        ),
    },
    "ofb_6d_no_integrator_gamma_1p4_rollout_radius": {
        "source_kind": "output_feedback_rollout_budget",
        "source_issue": "c92ebd8",
        "source_note": "6D no-integrator output-feedback robust-estimator rollout",
        "gamma_factor": 1.4,
        "gamma_star": 9166.831285473823,
        "gamma": 12833.563799663352,
        "epsilon_dim": 6,
        "disturbance_energy": 2.0657128682206633e-05,
        "gamma_equivalent_analytical_anchor": True,
        "description": (
            "6D no-integrator C&S output-feedback H-infinity rollout L2 radius for gamma_factor=1.4"
        ),
    },
    "ofb_6d_no_integrator_gamma_1p05_rollout_radius": {
        "source_kind": "output_feedback_rollout_budget",
        "source_issue": "c92ebd8",
        "source_note": "6D no-integrator output-feedback robust-estimator rollout",
        "gamma_factor": 1.05,
        "gamma_star": 9166.831285473823,
        "gamma": 9625.172849747514,
        "epsilon_dim": 6,
        "disturbance_energy": 3.0671655167860113e-06,
        "gamma_equivalent_analytical_anchor": True,
        "description": (
            "6D no-integrator C&S output-feedback H-infinity rollout L2 radius "
            "for gamma_factor=1.05"
        ),
    },
}


class BroadFullStateEpsilonTrainingConfig(CsPerturbationTrainingConfig):
    """Random full-state epsilon training lane for the C&S analytical game."""

    enabled: bool = False
    level: str = "moderate"
    budget_scale: float = 1.0
    reach_length_scaling: bool = True
    nominal_reach_length_m: float = BROAD_EPSILON_REFERENCE_REACH_M
    movement_epoch_only: bool = False
    epsilon_dim: int = BROAD_EPSILON_DIM

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
            "enabled": self.enabled,
            "mode": BROAD_EPSILON_TRAINING_MODE if self.enabled else "disabled",
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

    enabled: bool = False
    adversary_mechanism: str = BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM
    level: str = "moderate"
    budget_scale: float = 1.0
    reach_length_scaling: bool = True
    nominal_reach_length_m: float = BROAD_EPSILON_REFERENCE_REACH_M
    n_steps: int = 3
    step_size_fraction: float = 0.25
    inner_optimizer_method: str = BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT
    adam_learning_rate: float = 3e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    init: str = "zero"
    movement_epoch_only: bool = False
    epsilon_dim: int = BROAD_EPSILON_DIM
    budget_schedule: str = BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE
    sisu_levels: tuple[float, ...] = DEFAULT_PGD_SISU_LEVELS
    sisu_exact_zero_mass: float = DEFAULT_PGD_SISU_EXACT_ZERO_MASS
    sisu_condition_input: str = "auto"
    sisu_max_l2_radius_15cm: float | None = None
    sisu_max_radius_source: str | None = None
    fixed_l2_radius_15cm: float | None = None
    fixed_radius_source: str | None = None
    objective_kind: str = BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE
    energy_gamma_star: float | None = None
    energy_gamma_factor: float | None = None
    energy_gamma: float | None = None
    energy_penalty_scale: float = 1.0
    energy_lambda: float | None = None
    safety_cap_l2_radius_15cm: float | None = None
    safety_cap_source: str | None = None

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
            "enabled": self.enabled,
            "mode": BROAD_EPSILON_PGD_TRAINING_MODE if self.enabled else "disabled",
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

    enabled: bool = False
    policy_class: str = POLICY_ADVERSARY_MEMORYLESS_MLP
    mode: str = POLICY_ADVERSARY_PLAIN_MODE
    width: int = 64
    depth: int = 2
    n_steps: int = 5
    learning_rate: float = 3e-4
    energy_penalty_gamma: float = 1.0
    reference_l2_radius_15cm: float | None = None
    reach_length_scaling: bool = True
    nominal_reach_length_m: float = BROAD_EPSILON_REFERENCE_REACH_M
    movement_epoch_only: bool = False
    epsilon_dim: int = BROAD_EPSILON_DIM
    state_feature_dim: int = BROAD_EPSILON_DIM * 6
    budget_source: str | None = None

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
                **PGD_SISU_MAX_RADIUS_SOURCES.get(
                    self.budget_source,
                    {
                        "source_kind": "caller_declared",
                        "gamma_equivalent_analytical_anchor": False,
                        "description": self.budget_source,
                    },
                ),
            }
        )
        return {
            "enabled": self.enabled,
            "mode": POLICY_ADVERSARY_TRAINING_MODE if self.enabled else "disabled",
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
            **PGD_SISU_MAX_RADIUS_SOURCES.get(
                source_key,
                {
                    "source_kind": "caller_declared",
                    "gamma_equivalent_analytical_anchor": False,
                    "description": source_key,
                },
            ),
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
            **PGD_SISU_MAX_RADIUS_SOURCES.get(
                source_key,
                {
                    "source_kind": "caller_declared",
                    "gamma_equivalent_analytical_anchor": False,
                    "description": source_key,
                },
            ),
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
            **PGD_SISU_MAX_RADIUS_SOURCES.get(
                source_key,
                {
                    "source_kind": "caller_declared",
                    "gamma_equivalent_analytical_anchor": False,
                    "description": source_key,
                },
            ),
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

    enabled: bool = False
    nominal_fraction: float = 0.45
    single_fraction: float = 0.45
    combined_fraction: float = 0.10
    combined_amplitude_scale: float = 0.5
    initial_position_offset_m: float = 0.01
    initial_velocity_offset_m_s: float = 0.05
    process_epsilon_scale: float = 0.01
    command_input_pulse_n: float = 1.0
    sensory_feedback_offset_m: float = 0.01
    delayed_observation_offset_m: float = 0.01
    pulse_start_step: int = 20
    pulse_duration_steps: int = 5
    calibrated_timing: bool = False
    movement_age_timing: bool = False
    physical_level: str = "moderate"
    force_filter_feedback: bool = False
    calibration_regime: TrainingCalibrationRegime = OPEN_LOOP_ALL_CALIBRATION_REGIME
    closed_loop_calibration_table_path: str | None = None

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
            return CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE
        return PERTURBATION_TRAINING_MODE

    def to_hps_dict(self) -> dict[str, Any]:
        """Return the TreeNamespace-compatible config payload."""

        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "legacy_mode": (LEGACY_PERTURBATION_TRAINING_MODE if self.enabled else None),
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

    enabled: bool = False
    force_filter_feedback: bool = False
    target_support_profile: str = TARGET_SUPPORT_PROFILE_020A65B
    seen_directions_deg: tuple[float, ...] = DEFAULT_SEEN_TARGET_DIRECTIONS_DEG
    held_out_directions_deg: tuple[float, ...] = DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG
    seen_amplitudes_m: tuple[float, ...] = DEFAULT_SEEN_TARGET_AMPLITUDES_M
    held_out_amplitudes_m: tuple[float, ...] = DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M
    original_target_anchor_m: tuple[float, float] = ORIGINAL_TARGET_ANCHOR_M
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
            "enabled": self.enabled,
            "mode": (TARGET_RELATIVE_MULTITARGET_TRAINING_MODE if self.enabled else "disabled"),
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
    if isinstance(value, tuple) and all(_is_metadata_pair(item) for item in value):
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
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)


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


ISSUE_ID = "30f2313"

CS_POSITION_SCALE = 1e6

CS_VELOCITY_SCALE = 1e5

CS_CONTROL_SCALE = 1.0

DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW = "canonical_window"

ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND = "loss_blend"

MINIMAX_PARAMS_REF = "rlrmp/minimax/v1"

GUIDED_DISTILLATION_PARAMS_REF = "rlrmp/guided_distillation/v1"

CLOSED_LOOP_DISTILLATION_PARAMS_REF = "rlrmp/closed_loop_distillation/v1"


class GuidedDistillationConfig(BaseModel):
    """Unified authoring config for guided distillation CLI/spec surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = "h0_extlqg_6d_standard_graph_distillation"
    run_spec: str | None = None
    run_spec_output: str | None = None
    output_dir: str | None = None
    teacher_package: str = "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz"
    teacher_manifest: str = (
        "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers_manifest.json"
    )
    teacher_gains_key: str = "extlqg_controller_gains"
    clean_action_weight: float = 1.0
    perturbation_response_weight: float = 1.0
    input_output_jvp_weight: float = 0.25
    rollout_anchor_weight: float = 0.25
    n_jvp_directions: int = Field(16, gt=0)
    n_batches: int = Field(12000, gt=0)
    batch_size: int = Field(64, gt=0)
    n_replicates: int = Field(5, gt=0)
    hidden_size: int = Field(180, gt=0)
    horizon: int = Field(60, gt=0)
    seed: int = 0
    controller_lr: float = Field(3e-3, gt=0.0)
    lr_warmup_batches: int = Field(500, ge=0)
    lr_warmup_init_fraction: float = Field(0.1, ge=0.0)
    lr_cosine_alpha: float = Field(0.01, ge=0.0)
    gradient_clip_norm: float = Field(5.0, gt=0.0)
    trainable_dtype: str = "float32"
    population_mask_mode: str = "plain_all_ones"
    log_step: int = Field(10, gt=0)
    checkpoint: bool = True
    checkpoint_interval_batches: int = Field(500, gt=0)
    stop_after_batches: int | None = Field(None, gt=0)
    smoke_loss: bool = False
    smoke_train: bool = False
    full_train: bool = False
    resume: bool = False
    dry_run: bool = False


class ClosedLoopDistillationConfig(BaseModel):
    """Unified authoring config for closed-loop distillation CLI/spec surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = "h0_extlqg_6d_closed_loop_distillation"
    run_spec: Path | None = None
    run_spec_output: Path = Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json")
    output_dir: str = "_artifacts/a378b34/runs/h0_extlqg_6d_closed_loop_distillation"
    teacher_package: str = "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz"
    teacher_gains_key: str = "extlqg_controller_gains"
    horizon: int = Field(60, gt=0)
    seed: int = 0
    n_replicates: int = Field(5, gt=0)
    hidden_size: int = Field(180, gt=0)
    batch_size: int = Field(64, gt=0)
    n_batches: int = Field(12000, gt=0)
    controller_lr: float = Field(3e-3, gt=0.0)
    lr_warmup_batches: int = Field(500, ge=0)
    lr_cosine_alpha: float = Field(0.01, ge=0.0)
    gradient_clip_norm: float = Field(5.0, gt=0.0)
    checkpoint_interval_batches: int = Field(500, gt=0)
    trainable_dtype: str = "float32"
    kinematics_trajectory_weight: float = 1.0
    velocity_weight: float = 1.0
    endpoint_weight: float = 0.0
    settling_weight: float = 0.0
    action_force_weight: float = 1.0
    perturbation_response_weight: float = 1.0
    input_output_jvp_weight: float = 0.25
    task_rollout_loss_weight: float = 0.0
    write_run_spec: bool = False
    dry_run: bool = False
    smoke_preflight: bool = False
    smoke_train: bool = False
    smoke_n_batches: int = Field(1, gt=0)
    smoke_batch_size: int = Field(1, gt=0)
    full_train: bool = False
    confirm_full_train: bool = False
    resume: bool = False
    user_confirmed: bool = False


class MinimaxConfig(BaseModel):
    """Flat minimax method config whose fields own all authoring defaults."""

    model_config = ConfigDict(extra="forbid")

    n_warmup_batches: int = Field(2000, ge=0)
    n_adversary_batches: int = Field(8000, ge=0)
    n_adversary_steps: int = Field(5, gt=0)
    batch_size: int = Field(250, gt=0)
    adv_batch_size: int | None = None
    n_replicates: int = Field(5, gt=0)
    seed: int = 42

    controller_lr: float = 1e-4
    adversary_lr: float = 3e-4
    loss_update_enabled: bool = False
    loss_update_ratio: float = 0.5

    adversary_type: Literal["gaussian_bump", "linear_dynamics"] = "gaussian_bump"
    n_adversaries: int = Field(1, gt=0)
    n_bumps: int = 3
    force_max: float = 1.0
    linear_dynamics_eta_max: float = 0.1
    linear_dynamics_pgd_steps: int = 5
    linear_dynamics_lr: float = 1e-2

    hidden_type: Literal["gru", "vanilla_rnn", "linear", "linear_tracker"] = "gru"
    sisu_gating: Literal["additive", "multiplicative"] = "additive"

    nn_output: float = 1e-5
    nn_hidden: float = 1e-5
    nn_hidden_derivative: float = 0.0
    nn_output_jerk: float = 0.0
    nn_output_pre_go: float = 0.0
    nn_hidden_derivative_pre_go: float = 0.0
    effector_hold_pos: float = 10.0
    effector_hold_vel: float = 10.0
    effector_final_vel: float = 0.0
    effector_vel_late: float = 0.1
    effector_pos_running: float = 1.0
    effector_pos_late_weight: float = 0.5
    effector_pos_late_final_scale: float = 2.0
    effector_pos_late_start_step: int = 80

    effector_pos_running_schedule: Literal["flat", "powerlaw", "movement_ramp"] = "flat"
    effector_hold_pos_schedule: Literal["flat", "powerlaw"] = "flat"
    position_powerlaw_power: float = 6.0
    movement_ramp_shape: Literal["linear", "cosine", "power"] = "linear"
    movement_ramp_duration_steps: int = 60
    movement_ramp_power: float = 2.0

    p_catch_trial: float = 0.5

    warmup_model: str | None = None
    output_dir: str = "_artifacts/minimax/minimax_test"
    spec_dir: str | None = None
    jax_cache_dir: str | None = None
    jax_explain_cache_misses: bool = False
    allow_x64: bool = False
    checkpoint: bool = False
    checkpoint_every: int = 500
    resume: bool = False
    allow_fresh_start: bool = False
    fused: bool = True
    streaming_loss: bool = False

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

    run_spec: str | None = None
    compact_run_spec: bool = Field(False, exclude=True)
    output_dir: str = Field(f"_artifacts/{ISSUE_ID}/runs/cs_stochastic_gru__no_hidden_penalty")
    spec_dir: str | None = None
    issue: str = ISSUE_ID
    seed: int = 42

    n_train_batches: int = Field(12000, ge=0)
    batch_size: int = Field(250, gt=0)
    controller_lr: float = Field(1e-2, gt=0.0)
    lr_warmup_batches: int = Field(0, ge=0)
    lr_warmup_init_fraction: float = Field(0.1, ge=0.0)
    lr_cosine_alpha: float = Field(1.0, ge=0.0)
    gradient_clip_norm: float | None = None
    n_replicates: int = Field(5, gt=0)
    hidden_size: int = Field(180, gt=0)

    plant_backend: Literal["cs_lss", "legacy_causal_simplefeedback"] = "cs_lss"
    no_integrator_state: bool = False
    stochastic_preset: Literal["cs2019-rollout"] = "cs2019-rollout"
    target_m: float = 0.15
    n_input_only: int = Field(0, ge=0)
    n_readout_only: int = Field(0, ge=0)
    n_recurrent_only: int = Field(0, ge=0)

    effector_pos_running: float = CS_POSITION_SCALE
    effector_vel_running: float = CS_VELOCITY_SCALE
    effector_terminal_pos: float = CS_POSITION_SCALE
    effector_terminal_vel: float = CS_VELOCITY_SCALE
    effector_final_vel: float = 0.0
    nn_output: float = CS_CONTROL_SCALE
    nn_output_jerk: float = 0.0
    nn_output_pre_go: float | None = None
    delayed_pre_go_force_filter_hold: float = 0.0
    delayed_pre_go_start_pos_hold: float = 0.0
    delayed_pre_go_start_pos_hold_norm: Literal["l2", "l1"] = "l2"
    delayed_pre_go_zero_vel_hold: float = 0.0
    loss_objective: Literal[
        "partial_feedbax_terms",
        "partial_net_output_force_filter",
        "full_analytical_qrf",
    ] = "partial_feedbax_terms"
    regularized_fidelity: bool = False

    perturbation_training: bool | None = None
    perturbation_nominal_fraction: float = 0.45
    perturbation_single_fraction: float = 0.45
    perturbation_combined_fraction: float = 0.10
    perturbation_combined_amplitude_scale: float = 0.5
    perturbation_initial_position_offset_m: float = 0.01
    perturbation_initial_velocity_offset_m_s: float = 0.05
    perturbation_process_epsilon_scale: float = 0.01
    perturbation_command_input_pulse_n: float = 1.0
    perturbation_sensory_feedback_offset_m: float = 0.01
    perturbation_delayed_observation_offset_m: float = 0.01
    perturbation_pulse_start_step: int = 20
    perturbation_pulse_duration_steps: int = 5
    perturbation_calibrated_timing: bool | None = None
    perturbation_movement_age_timing: bool | None = None
    perturbation_physical_level: Literal["small", "moderate", "stress"] | None = None
    perturbation_calibration_regime: Literal[
        "open_loop_all",
        "closed_loop_sensory",
        "closed_loop_sensory_command_lateral",
    ] = "open_loop_all"
    perturbation_closed_loop_calibration_table: str | None = None

    target_relative_multitarget: bool = False
    target_support_profile: Literal[
        "old_020a65b",
        "const_dense_all",
        "const_sparse8",
        "const_band8",
        "const_band16",
        "const_band36",
    ] = DEFAULT_TARGET_SUPPORT_PROFILE
    delayed_reach: bool = False
    delayed_reach_go_cue_min_step: int = 10
    delayed_reach_go_cue_max_step: int = 30
    delayed_reach_p_catch_trial: float = 0.5
    delayed_movement_cost_tail_mode: Literal[
        "canonical_window",
        "flat_after_canonical_horizon",
    ] = DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW
    delayed_reach_trial_type_normalized_loss: bool = False
    delayed_reach_no_catch_qrf_weight: float = 1.0
    delayed_reach_catch_qrf_weight: float = 1.0
    force_filter_feedback: bool | None = None

    broad_epsilon_training: bool = False
    broad_epsilon_pgd_training: bool = False
    broad_epsilon_level: Literal["moderate", "strong"] = "moderate"
    broad_epsilon_budget_scale: float = 1.0
    broad_epsilon_pgd_fixed_radius_15cm: float | None = None
    broad_epsilon_pgd_fixed_radius_source: str | None = None
    broad_epsilon_reach_scaling: bool = True
    broad_epsilon_pgd_steps: int = Field(3, gt=0)
    broad_epsilon_pgd_step_size_fraction: float = 0.25
    broad_epsilon_pgd_inner_optimizer_method: Literal[
        "projected_gradient_ascent",
        "adam",
    ] = "projected_gradient_ascent"
    broad_epsilon_pgd_adam_lr: float = 3e-4
    broad_epsilon_pgd_adam_b1: float = 0.9
    broad_epsilon_pgd_adam_b2: float = 0.999
    broad_epsilon_pgd_adam_eps: float = 1e-8
    broad_epsilon_pgd_mechanism: Literal[
        "direct_epsilon",
        "linear_no_bias",
        "affine",
    ] = "direct_epsilon"
    broad_epsilon_pgd_objective: Literal["hard_l2", "soft_energy"] = "hard_l2"
    broad_epsilon_pgd_energy_gamma_star: float | None = None
    broad_epsilon_pgd_energy_gamma_factor: float | None = None
    broad_epsilon_pgd_energy_gamma: float | None = None
    broad_epsilon_pgd_energy_penalty_scale: float = 1.0
    broad_epsilon_pgd_energy_lambda: float | None = None
    broad_epsilon_pgd_safety_cap_15cm: float | None = None
    broad_epsilon_pgd_safety_cap_source: str | None = None
    broad_epsilon_pgd_budget_schedule: Literal["fixed", "sisu_energy_fraction"] = "fixed"
    broad_epsilon_pgd_sisu_condition_input: Literal["auto", "input", "sisu"] = "auto"
    broad_epsilon_pgd_sisu_max_radius: float | None = None
    broad_epsilon_pgd_sisu_max_radius_source: str | None = None

    adaptive_epsilon_curriculum: bool = False
    adaptive_epsilon_controller_training_mode: Literal[
        "loss_blend",
        "epsilon_scaled_outer_training",
    ] = ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND
    adaptive_epsilon_damage_start: float = 0.0
    adaptive_epsilon_damage_peak: float = 3500.0
    adaptive_epsilon_damage_final: float = 1000.0
    adaptive_epsilon_damage_ramp_batches: int = 2500
    adaptive_epsilon_damage_anneal_batches: int = 5000
    adaptive_epsilon_update_interval_batches: int = 50
    adaptive_epsilon_ema_alpha: float = 0.1
    adaptive_epsilon_eta: float = 0.1
    adaptive_epsilon_deadband_frac: float = 0.10
    adaptive_epsilon_hysteresis_frac: float | None = None
    adaptive_epsilon_freeze_during_application_ramp: bool = False
    adaptive_epsilon_gain_normalization: bool = False
    adaptive_epsilon_gain_ema_alpha: float = 0.2
    adaptive_epsilon_gain_min: float = 0.25
    adaptive_epsilon_gain_max: float = 8.0
    adaptive_epsilon_lambda_min: float | None = None
    adaptive_epsilon_lambda_max: float | None = None
    adaptive_epsilon_max_log_step: float = 0.25
    adaptive_epsilon_outer_weight_start: float = 0.0
    adaptive_epsilon_outer_weight_final: float = 1.0
    adaptive_epsilon_outer_weight_ramp_batches: int = 2500

    policy_adversary_training: bool = False
    policy_adversary_policy_class: Literal["memoryless_mlp", "linear_no_bias", "affine"] = (
        "memoryless_mlp"
    )
    policy_adversary_mode: Literal["plain", "energy"] = "plain"
    policy_adversary_width: int = Field(64, gt=0)
    policy_adversary_depth: int = Field(2, ge=0)
    policy_adversary_steps: int = Field(5, gt=0)
    policy_adversary_lr: float = 3e-4
    policy_adversary_energy_gamma: float = 1.0
    policy_adversary_radius_15cm: float | None = None
    policy_adversary_radius_source: str | None = None

    initial_hidden_encoder: bool = False
    smoke: bool = False
    full_train: bool = False
    resume: bool = False
    allow_fresh_start: bool = False
    stop_after_batches: int | None = None
    training_diagnostics: bool = True
    checkpoint_interval_batches: int = 500
    log_step: int = 100
    disable_progress: bool = False
    quiet_progress: bool = True
    allow_x64: bool = False
    dry_run: bool = False

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


def _normalize_fixed_target_payload(config: Any) -> FixedTargetPerturbationTrainingConfig:
    """Normalize historical fixed-target training metadata."""

    return FixedTargetPerturbationTrainingConfig(
        enabled=bool(getattr(config, "enabled", False)),
        nominal_fraction=float(getattr(config, "nominal_fraction", 0.45)),
        single_fraction=float(getattr(config, "single_fraction", 0.45)),
        combined_fraction=float(getattr(config, "combined_fraction", 0.10)),
        combined_amplitude_scale=float(getattr(config, "combined_amplitude_scale", 0.5)),
        initial_position_offset_m=float(getattr(config, "initial_position_offset_m", 0.01)),
        initial_velocity_offset_m_s=float(getattr(config, "initial_velocity_offset_m_s", 0.05)),
        process_epsilon_scale=float(getattr(config, "process_epsilon_scale", 0.01)),
        command_input_pulse_n=float(getattr(config, "command_input_pulse_n", 1.0)),
        sensory_feedback_offset_m=float(getattr(config, "sensory_feedback_offset_m", 0.01)),
        delayed_observation_offset_m=float(getattr(config, "delayed_observation_offset_m", 0.01)),
        pulse_start_step=int(getattr(config, "pulse_start_step", 20)),
        pulse_duration_steps=int(getattr(config, "pulse_duration_steps", 5)),
        calibrated_timing=bool(getattr(config, "calibrated_timing", False)),
        movement_age_timing=bool(getattr(config, "movement_age_timing", False)),
        physical_level=str(getattr(config, "physical_level", "moderate")),
        force_filter_feedback=bool(getattr(config, "force_filter_feedback", False)),
        calibration_regime=str(
            getattr(config, "calibration_regime", OPEN_LOOP_ALL_CALIBRATION_REGIME)
        ),
        closed_loop_calibration_table_path=getattr(
            config,
            "closed_loop_calibration_table_path",
            None,
        ),
    )


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


def _normalize_target_payload(config: Any) -> TargetRelativeMultiTargetTrainingConfig:
    """Normalize an hps target-distribution payload to a dataclass."""

    enabled = bool(_payload_get(config, "enabled", False))
    force_filter_feedback = _payload_get(config, "force_filter_feedback", False)
    if not isinstance(force_filter_feedback, bool):
        force_filter_feedback = bool(_payload_get(force_filter_feedback, "enabled", False))
    target_distribution = _payload_get(config, "target_distribution", None)

    profile = str(
        _first_payload_value(
            (config, "target_support_profile"),
            (target_distribution, "target_support_profile"),
            default=DEFAULT_TARGET_SUPPORT_PROFILE,
        )
    )
    default_config = target_relative_target_support_config(
        profile=profile,
        enabled=enabled,
        force_filter_feedback=bool(force_filter_feedback),
    )
    return TargetRelativeMultiTargetTrainingConfig(
        enabled=enabled,
        force_filter_feedback=bool(force_filter_feedback),
        target_support_profile=profile,
        seen_directions_deg=tuple(
            float(x)
            for x in _first_payload_value(
                (config, "seen_directions_deg"),
                (target_distribution, "seen_directions_deg"),
                default=default_config.seen_directions_deg,
            )
        ),
        held_out_directions_deg=tuple(
            float(x)
            for x in _first_payload_value(
                (config, "held_out_directions_deg"),
                (target_distribution, "held_out_directions_deg"),
                default=default_config.held_out_directions_deg,
            )
        ),
        seen_amplitudes_m=tuple(
            float(x)
            for x in _first_payload_value(
                (config, "seen_amplitudes_m"),
                (target_distribution, "seen_amplitudes_m"),
                default=default_config.seen_amplitudes_m,
            )
        ),
        held_out_amplitudes_m=tuple(
            float(x)
            for x in _first_payload_value(
                (config, "held_out_amplitudes_m"),
                (target_distribution, "held_out_amplitudes_m"),
                default=default_config.held_out_amplitudes_m,
            )
        ),
        original_target_anchor_m=tuple(
            float(x)
            for x in _first_payload_value(
                (config, "original_target_anchor_m"),
                (target_distribution, "original_target_anchor_m"),
                default=default_config.original_target_anchor_m,
            )
        ),
        support_metadata=_first_payload_value(
            (config, "support_metadata"),
            (target_distribution, "support_metadata"),
            default=default_config.support_metadata,
        ),
    )


def _normalize_broad_epsilon_payload(config: Any) -> BroadFullStateEpsilonTrainingConfig:
    """Normalize an hps broad-epsilon payload to a dataclass."""

    return BroadFullStateEpsilonTrainingConfig(
        enabled=bool(getattr(config, "enabled", False)),
        level=str(getattr(config, "level", "moderate")),
        budget_scale=float(getattr(config, "budget_scale", 1.0)),
        reach_length_scaling=bool(getattr(config, "reach_length_scaling", True)),
        nominal_reach_length_m=float(
            getattr(config, "nominal_reach_length_m", BROAD_EPSILON_REFERENCE_REACH_M)
        ),
        movement_epoch_only=bool(getattr(config, "movement_epoch_only", False)),
        epsilon_dim=int(getattr(config, "epsilon_dim", BROAD_EPSILON_DIM)),
    )


def _normalize_broad_epsilon_pgd_payload(config: Any) -> PgdFullStateEpsilonTrainingConfig:
    """Normalize an hps PGD broad-epsilon payload to a dataclass."""

    inner = _payload_get(config, "inner_maximizer", None)
    mechanism = _payload_get(config, "mechanism", None)
    schedule = _payload_get(config, "budget_schedule", None)
    objective = _payload_get(config, "objective", None)
    safety_cap = _payload_get(config, "safety_cap", None)
    budget_schedule = str(
        _first_payload_value(
            (schedule, "mode"),
            (config, "budget_schedule_mode"),
            (config, "budget_schedule"),
            default=BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE,
        )
    )
    budget_contract = _payload_get(config, "budget_contract", None)
    budget_source = _payload_get(budget_contract, "budget_source", None)
    fixed_l2_radius_15cm = (
        None
        if budget_schedule != BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE
        else _optional_float(
            _first_payload_value(
                (budget_contract, "effective_l2_radius_15cm"),
                (config, "fixed_l2_radius_15cm"),
                default=None,
            )
        )
    )
    fixed_radius_source = (
        None
        if budget_schedule != BROAD_EPSILON_PGD_FIXED_BUDGET_SCHEDULE
        else _optional_str(
            _first_payload_value(
                (budget_source, "key"),
                (config, "fixed_radius_source"),
                default=None,
            )
        )
    )
    adam = _payload_get(inner, "adam", None)
    return PgdFullStateEpsilonTrainingConfig(
        enabled=bool(_payload_get(config, "enabled", False)),
        adversary_mechanism=str(
            _first_payload_value(
                (config, "adversary_mechanism"),
                (mechanism, "name"),
                (mechanism, "policy_class"),
                default=BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
            )
        ),
        level=str(_payload_get(config, "level", "moderate")),
        budget_scale=float(_payload_get(config, "budget_scale", 1.0)),
        reach_length_scaling=bool(_payload_get(config, "reach_length_scaling", True)),
        nominal_reach_length_m=float(
            _payload_get(config, "nominal_reach_length_m", BROAD_EPSILON_REFERENCE_REACH_M)
        ),
        n_steps=int(_first_payload_value((inner, "n_steps"), (config, "n_steps"), default=3)),
        step_size_fraction=float(
            _first_payload_value(
                (inner, "step_size_fraction_of_l2_radius"),
                (inner, "step_size_fraction"),
                (config, "step_size_fraction_of_l2_radius"),
                (config, "step_size_fraction"),
                default=0.25,
            )
        ),
        inner_optimizer_method=str(
            _first_payload_value(
                (inner, "method"),
                (config, "inner_optimizer_method"),
                default=BROAD_EPSILON_PGD_PROJECTED_GRADIENT_ASCENT,
            )
        ),
        adam_learning_rate=float(
            _first_payload_value(
                (adam, "learning_rate"),
                (inner, "learning_rate"),
                (config, "adam_learning_rate"),
                default=3e-4,
            )
        ),
        adam_b1=float(
            _first_payload_value(
                (adam, "b1"),
                (config, "adam_b1"),
                default=0.9,
            )
        ),
        adam_b2=float(
            _first_payload_value(
                (adam, "b2"),
                (config, "adam_b2"),
                default=0.999,
            )
        ),
        adam_eps=float(
            _first_payload_value(
                (adam, "eps"),
                (config, "adam_eps"),
                default=1e-8,
            )
        ),
        init=str(
            _first_payload_value(
                (inner, "initialization"),
                (inner, "init"),
                (config, "initialization"),
                (config, "init"),
                default="zero",
            )
        ),
        movement_epoch_only=bool(_payload_get(config, "movement_epoch_only", False)),
        epsilon_dim=int(_payload_get(config, "epsilon_dim", BROAD_EPSILON_DIM)),
        budget_schedule=budget_schedule,
        sisu_levels=tuple(
            float(x)
            for x in _first_payload_value(
                (schedule, "levels"),
                (config, "sisu_levels"),
                default=DEFAULT_PGD_SISU_LEVELS,
            )
        ),
        sisu_exact_zero_mass=float(
            _first_payload_value(
                (schedule, "exact_zero_mass"),
                (config, "sisu_exact_zero_mass"),
                default=DEFAULT_PGD_SISU_EXACT_ZERO_MASS,
            )
        ),
        sisu_condition_input=str(
            _first_payload_value(
                (schedule, "conditioning_input"),
                (_payload_get(schedule, "conditioning_scalar", None), "input_key"),
                (config, "sisu_condition_input"),
                default="auto",
            )
        ),
        sisu_max_l2_radius_15cm=_optional_float(
            _first_payload_value(
                (schedule, "max_l2_radius_15cm"),
                (config, "sisu_max_l2_radius_15cm"),
                default=None,
            )
        ),
        sisu_max_radius_source=_optional_str(
            _first_payload_value(
                (schedule, "max_radius_source_key"),
                (_payload_get(schedule, "max_radius_source", None), "key"),
                (config, "sisu_max_radius_source"),
                default=None,
            )
        ),
        fixed_l2_radius_15cm=fixed_l2_radius_15cm,
        fixed_radius_source=fixed_radius_source,
        objective_kind=str(
            _first_payload_value(
                (objective, "kind"),
                (config, "objective_kind"),
                default=BROAD_EPSILON_PGD_HARD_L2_OBJECTIVE,
            )
        ),
        energy_gamma_star=_optional_float(
            _first_payload_value(
                (objective, "gamma_star"),
                (config, "energy_gamma_star"),
                default=None,
            )
        ),
        energy_gamma_factor=_optional_float(
            _first_payload_value(
                (objective, "gamma_factor"),
                (config, "energy_gamma_factor"),
                default=None,
            )
        ),
        energy_gamma=_optional_float(
            _first_payload_value(
                (objective, "gamma"),
                (config, "energy_gamma"),
                default=None,
            )
        ),
        energy_penalty_scale=float(
            _first_payload_value(
                (objective, "penalty_scale_c"),
                (config, "energy_penalty_scale"),
                default=1.0,
            )
        ),
        energy_lambda=_optional_float(
            _first_payload_value(
                (objective, "lambda"),
                (config, "energy_lambda"),
                default=None,
            )
        ),
        safety_cap_l2_radius_15cm=_optional_float(
            _first_payload_value(
                (safety_cap, "l2_radius_15cm"),
                (config, "safety_cap_l2_radius_15cm"),
                default=None,
            )
        ),
        safety_cap_source=_optional_str(
            _first_payload_value(
                (_payload_get(safety_cap, "source", None), "key"),
                (config, "safety_cap_source"),
                default=None,
            )
        ),
    )


def _normalize_policy_adversary_payload(config: Any) -> PolicyFullStateEpsilonTrainingConfig:
    """Normalize an hps policy-adversary payload to a dataclass."""

    policy = _payload_get(config, "policy", None)
    optimizer = _payload_get(config, "inner_optimizer", None)
    objective = _payload_get(config, "objective", None)
    budget = _payload_get(config, "budget_contract", None)
    budget_source = _payload_get(budget, "budget_source", None)
    policy_kind = str(
        _first_payload_value(
            (config, "policy_class"),
            (policy, "kind"),
            default=POLICY_ADVERSARY_MEMORYLESS_MLP,
        )
    )
    if policy_kind == "closed_loop_finite_time_varying_epsilon_policy":
        metadata = _payload_get(policy, "metadata", None)
        policy_kind = str(_payload_get(metadata, "policy_class", LINEAR_NO_BIAS_POLICY))
    return PolicyFullStateEpsilonTrainingConfig(
        enabled=bool(_payload_get(config, "enabled", False)),
        policy_class=policy_kind,
        mode=str(
            _first_payload_value(
                (config, "row_mode"),
                (objective, "active"),
                (config, "mode"),
                default=POLICY_ADVERSARY_PLAIN_MODE,
            )
        ),
        width=int(_first_payload_value((policy, "width"), (config, "width"), default=64)),
        depth=int(_first_payload_value((policy, "depth"), (config, "depth"), default=2)),
        n_steps=int(
            _first_payload_value(
                (optimizer, "n_ascent_steps_per_controller_step"),
                (config, "n_steps"),
                default=5,
            )
        ),
        learning_rate=float(
            _first_payload_value(
                (optimizer, "learning_rate"),
                (config, "learning_rate"),
                default=3e-4,
            )
        ),
        energy_penalty_gamma=float(
            _first_payload_value(
                (objective, "energy_penalty_gamma"),
                (config, "energy_penalty_gamma"),
                default=1.0,
            )
        ),
        reference_l2_radius_15cm=_optional_float(
            _first_payload_value(
                (budget, "effective_l2_radius_15cm"),
                (budget, "active_max_l2_radius_15cm"),
                (config, "reference_l2_radius_15cm"),
                default=None,
            )
        ),
        reach_length_scaling=bool(_payload_get(config, "reach_length_scaling", True)),
        nominal_reach_length_m=float(
            _payload_get(config, "nominal_reach_length_m", BROAD_EPSILON_REFERENCE_REACH_M)
        ),
        movement_epoch_only=bool(_payload_get(config, "movement_epoch_only", False)),
        epsilon_dim=int(
            _first_payload_value(
                (policy, "output_dim"),
                (config, "epsilon_dim"),
                default=BROAD_EPSILON_DIM,
            )
        ),
        state_feature_dim=int(
            _first_payload_value(
                (policy, "state_feature_dim"),
                (config, "state_feature_dim"),
                default=BROAD_EPSILON_DIM * 6,
            )
        ),
        budget_source=_optional_str(
            _first_payload_value(
                (budget_source, "key"),
                (config, "budget_source"),
                default=None,
            )
        ),
    )


_HISTORICAL_PAYLOAD_NORMALIZERS = {
    FixedTargetPerturbationTrainingConfig: _normalize_fixed_target_payload,
    TargetRelativeMultiTargetTrainingConfig: _normalize_target_payload,
    BroadFullStateEpsilonTrainingConfig: _normalize_broad_epsilon_payload,
    PgdFullStateEpsilonTrainingConfig: _normalize_broad_epsilon_pgd_payload,
    PolicyFullStateEpsilonTrainingConfig: _normalize_policy_adversary_payload,
}


def validate_training_config(config_type: type[BaseModel], payload: Any) -> Any:
    """Validate one canonical training config, normalizing historical metadata once."""

    if isinstance(payload, config_type):
        return payload
    normalizer = _HISTORICAL_PAYLOAD_NORMALIZERS.get(config_type)
    if normalizer is None:
        raw = payload if isinstance(payload, Mapping) else vars(payload)
        return config_type.model_validate(raw)
    return normalizer(payload)


_MISSING = object()


def _payload_get(payload: Any, name: str, default: Any = _MISSING) -> Any:
    if payload is None:
        return default
    if isinstance(payload, dict):
        if name in payload:
            return payload[name]
        return default
    return getattr(payload, name, default)


def _first_payload_value(*candidates: tuple[Any, str], default: Any) -> Any:
    for payload, name in candidates:
        value = _payload_get(payload, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


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

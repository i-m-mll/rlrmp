"""Fixed-target perturbation-generalized training config for C&S GRU runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax import AbstractTask, TaskTrialSpec, WhereDict
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
)
from jaxtyping import PRNGKeyArray

from rlrmp.analysis.pipelines.gru_perturbation_calibration import (
    DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS,
    DEFAULT_CONTROLLER_VISIBLE_VELOCITY_SCALE_M_S,
    DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT,
    DEFAULT_PLANT_TIMING_BINS,
    DEFAULT_REACH_RELATIVE_LEVELS,
)
from rlrmp.model.feedbax_channel_adapters import (
    additive_channel_payload_dim,
    additive_channel_provenance,
    materialize_additive_channel_adapters_on_graph,
)

PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_randomized"
CALIBRATED_TIMING_PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_calibrated_timing"
LEGACY_PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_generalized"
TARGET_RELATIVE_MULTITARGET_TRAINING_MODE = "target_relative_multitarget_static"
TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE = "target_relative_multitarget_static_h0"
BROAD_EPSILON_TRAINING_MODE = "broad_full_state_epsilon_l2"
BROAD_EPSILON_PGD_TRAINING_MODE = "broad_full_state_epsilon_pgd_l2"
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

PerturbationBin = Literal[
    "nominal",
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "delayed_observation",
    "mild_combined",
]

VALIDATION_BINS: tuple[PerturbationBin, ...] = (
    "nominal",
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "delayed_observation",
    "mild_combined",
)

SINGLE_FAMILY_BINS: tuple[PerturbationBin, ...] = (
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "delayed_observation",
)

GRAPH_CHANNEL_BINS: tuple[PerturbationBin, ...] = (
    "command_input",
    "sensory_feedback",
    "delayed_observation",
)
PLANT_TIMED_BINS: tuple[PerturbationBin, ...] = ("process_epsilon", "command_input")
CONTROLLER_VISIBLE_TIMED_BINS: tuple[PerturbationBin, ...] = (
    "sensory_feedback",
    "delayed_observation",
)
REACH_RELATIVE_LEVELS: dict[str, float] = {
    level.name: float(level.fraction_of_reach) for level in DEFAULT_REACH_RELATIVE_LEVELS
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
TIMING_LABELS_PLANT = tuple(bin_.label for bin_ in DEFAULT_PLANT_TIMING_BINS)
BROAD_EPSILON_DIM = 8
BROAD_EPSILON_REFERENCE_REACH_M = 0.15
BROAD_EPSILON_LEVELS: dict[str, dict[str, Any]] = {
    "moderate": {
        "gamma_factor": 1.4,
        "closed_loop_epsilon_energy_15cm": 1.518885046213267e-06,
        "closed_loop_epsilon_l2_15cm": 0.0012324305441740995,
        "delta_v_percent": 4.041729916548296,
        "source_issue": "cb98e58",
        "source_note": "results/cb98e58/notes/analytical_game_card_manifest.json",
    },
    "strong": {
        "gamma_factor": 1.05,
        "closed_loop_epsilon_energy_15cm": 5.421868381615368e-06,
        "closed_loop_epsilon_l2_15cm": 0.0023284905801002004,
        "delta_v_percent": 7.460371202249536,
        "source_issue": "a7dad8a",
        "source_note": "results/a7dad8a/notes/adversary_equivalence_manifest.json",
    },
}


@dataclass(frozen=True)
class BroadFullStateEpsilonTrainingConfig:
    """Random full-state epsilon training lane for the C&S analytical game."""

    enabled: bool = False
    level: str = "moderate"
    budget_scale: float = 1.0
    reach_length_scaling: bool = True
    nominal_reach_length_m: float = BROAD_EPSILON_REFERENCE_REACH_M
    movement_epoch_only: bool = False
    epsilon_dim: int = BROAD_EPSILON_DIM

    def __post_init__(self) -> None:
        if self.level not in BROAD_EPSILON_LEVELS:
            levels = ", ".join(BROAD_EPSILON_LEVELS)
            raise ValueError(
                f"Unknown broad-epsilon level {self.level!r}; expected one of {levels}."
            )
        if float(self.budget_scale) <= 0.0:
            raise ValueError("broad epsilon budget_scale must be positive.")
        if float(self.nominal_reach_length_m) <= 0.0:
            raise ValueError("broad epsilon nominal_reach_length_m must be positive.")
        if int(self.epsilon_dim) < 1:
            raise ValueError("broad epsilon epsilon_dim must be positive.")

    @property
    def level_contract(self) -> dict[str, Any]:
        """Return the immutable analytical budget anchor for this level."""

        return dict(BROAD_EPSILON_LEVELS[self.level])

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

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable broad-epsilon metadata."""

        return self.to_hps_dict()


@dataclass(frozen=True)
class PgdFullStateEpsilonTrainingConfig:
    """Training-time PGD lane on the C&S full-state epsilon channel."""

    enabled: bool = False
    level: str = "moderate"
    budget_scale: float = 1.0
    reach_length_scaling: bool = True
    nominal_reach_length_m: float = BROAD_EPSILON_REFERENCE_REACH_M
    n_steps: int = 3
    step_size_fraction: float = 0.25
    init: str = "zero"
    movement_epoch_only: bool = False
    epsilon_dim: int = BROAD_EPSILON_DIM

    def __post_init__(self) -> None:
        if self.level not in BROAD_EPSILON_LEVELS:
            levels = ", ".join(BROAD_EPSILON_LEVELS)
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
        if int(self.epsilon_dim) < 1:
            raise ValueError("PGD broad epsilon epsilon_dim must be positive.")
        if self.init != "zero":
            raise ValueError("Only zero-initialized PGD broad epsilon is currently supported.")

    @property
    def level_contract(self) -> dict[str, Any]:
        """Return the immutable analytical budget anchor for this level."""

        return dict(BROAD_EPSILON_LEVELS[self.level])

    @property
    def reference_l2_radius(self) -> float:
        """Return the 15 cm reference L2 radius after the explicit budget scale."""

        return float(self.level_contract["closed_loop_epsilon_l2_15cm"]) * float(self.budget_scale)

    def to_hps_dict(self) -> dict[str, Any]:
        """Return TreeNamespace-compatible PGD broad-epsilon training metadata."""

        contract = self.level_contract
        return {
            "enabled": self.enabled,
            "mode": BROAD_EPSILON_PGD_TRAINING_MODE if self.enabled else "disabled",
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
            "inner_maximizer": {
                "method": "projected_gradient_ascent",
                "n_steps": int(self.n_steps),
                "step_size_fraction_of_l2_radius": float(self.step_size_fraction),
                "initialization": self.init,
                "projection": (
                    "per_trial_flattened_movement_time_component_l2_ball"
                    if self.movement_epoch_only
                    else "per_trial_flattened_time_component_l2_ball"
                ),
                "time_mask": _epsilon_time_mask_contract(self.movement_epoch_only),
                "differentiated_through_outer_update": False,
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

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable PGD broad-epsilon metadata."""

        return self.to_hps_dict()


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


def _widen_controller_visible_adapter(
    spec: AdditiveGraphChannelAdapterSpec,
) -> AdditiveGraphChannelAdapterSpec:
    return spec.model_copy(
        update={
            "payload_shape": [6],
            "metadata": {
                **dict(spec.metadata),
                "force_filter_feedback_payload": "widened_to_controller_feedback_dim",
                "active_calibrated_components": 4,
                "inactive_force_filter_components": [4, 5],
            },
        }
    )


@dataclass(frozen=True)
class FixedTargetPerturbationTrainingConfig:
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

    def __post_init__(self) -> None:
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
            "graph_adapter_payloads": {
                bin_name: {
                    "input_key": spec.input_key,
                    "payload_shape": list(spec.payload_shape or []),
                    "active_calibrated_components": spec.metadata.get(
                        "active_calibrated_components",
                        spec.payload_shape[-1] if spec.payload_shape else None,
                    ),
                }
                for bin_name, spec in graph_adapter_specs(
                    force_filter_feedback=self.force_filter_feedback
                ).items()
            },
            "physical_level_fraction_of_reach": REACH_RELATIVE_LEVELS[self.physical_level],
            "training_physical_levels": list(TRAINING_REACH_RELATIVE_LEVELS),
            "eval_only_physical_levels": list(EVAL_ONLY_REACH_RELATIVE_LEVELS),
            "single_family_bins": list(SINGLE_FAMILY_BINS),
            "validation_bins": list(VALIDATION_BINS),
            "families": {
                "initial_position": {
                    "channel": "initial_state",
                    "family": "initial_position_offset",
                    "amplitude": self.initial_position_offset_m,
                    "units": "m",
                },
                "initial_velocity": {
                    "channel": "initial_state",
                    "family": "initial_velocity_offset",
                    "amplitude": self.initial_velocity_offset_m_s,
                    "units": "m/s",
                },
                "process_epsilon": {
                    "channel": "process_epsilon",
                    "family": "process_epsilon_pulse",
                    "amplitude": self.process_epsilon_scale,
                    "units": "epsilon",
                },
                "command_input": {
                    "channel": "command_input",
                    "family": "command_input_pulse",
                    "amplitude": self.command_input_pulse_n,
                    "units": "N",
                },
                "sensory_feedback": {
                    "channel": "sensory_feedback",
                    "family": "sensory_feedback_offset",
                    "amplitude": self.sensory_feedback_offset_m,
                    "units": "m_or_m_s_channel_units",
                },
                "delayed_observation": {
                    "channel": "delayed_observation",
                    "family": "delayed_observation_offset",
                    "amplitude": self.delayed_observation_offset_m,
                    "units": "m_or_m_s_channel_units",
                },
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
                graph_adapter_specs(force_filter_feedback=self.force_filter_feedback)[bin_name],
                adapter="feedbax.additive_channel_adapter",
            )
            for bin_name in GRAPH_CHANNEL_BINS
        }
        return payload


@dataclass(frozen=True)
class TargetRelativeMultiTargetTrainingConfig:
    """Structured static-target distribution for target-relative GRU training."""

    enabled: bool = False
    force_filter_feedback: bool = False
    seen_directions_deg: tuple[float, ...] = DEFAULT_SEEN_TARGET_DIRECTIONS_DEG
    held_out_directions_deg: tuple[float, ...] = DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG
    seen_amplitudes_m: tuple[float, ...] = DEFAULT_SEEN_TARGET_AMPLITUDES_M
    held_out_amplitudes_m: tuple[float, ...] = DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M
    original_target_anchor_m: tuple[float, float] = ORIGINAL_TARGET_ANCHOR_M

    def __post_init__(self) -> None:
        if len(self.original_target_anchor_m) != 2:
            raise ValueError("original_target_anchor_m must be a 2D target.")
        if not self.seen_directions_deg:
            raise ValueError("At least one seen target direction is required.")
        if not self.held_out_directions_deg:
            raise ValueError("At least one held-out target direction is required.")
        if not self.seen_amplitudes_m:
            raise ValueError("At least one seen target amplitude is required.")
        if not self.held_out_amplitudes_m:
            raise ValueError("At least one held-out target amplitude is required.")
        if any(float(value) <= 0.0 for value in self.seen_amplitudes_m):
            raise ValueError("Seen target amplitudes must be positive.")
        if any(float(value) <= 0.0 for value in self.held_out_amplitudes_m):
            raise ValueError("Held-out target amplitudes must be positive.")
        seen = _target_tuples(self.seen_directions_deg, self.seen_amplitudes_m)
        if tuple(float(x) for x in self.original_target_anchor_m) not in seen:
            raise ValueError(
                "The structured seen target distribution must include the original "
                "15 cm forward target anchor."
            )

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


class TargetRelativeMultiTargetTrainingTaskAdapter(AbstractTask):
    """Rewrite static target trials and expose controller-visible target input."""

    task: object
    config: Any

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

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.get_train_trial_with_intervenor_params(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        base = self.task.get_train_trial_with_intervenor_params(key, batch_info)
        return apply_training_target_distribution(base, self.config, key)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        del key
        return apply_validation_target_distribution(self.task.validation_trials, self.config)

    @property
    def n_validation_trials(self) -> int:
        return len(config_from_target_hps(self.config).validation_targets_m)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return apply_validation_target_distribution(self.task.validation_trials, self.config)


class BroadFullStateEpsilonTrainingTaskAdapter(AbstractTask):
    """Inject randomized full-state C&S epsilon after target sampling."""

    task: object
    config: Any

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

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.get_train_trial_with_intervenor_params(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        key_base, key_epsilon = jr.split(key)
        base = self.task.get_train_trial_with_intervenor_params(key_base, batch_info)
        return apply_broad_epsilon_training(base, self.config, key_epsilon)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        return self.task.get_validation_trials(key)

    @property
    def n_validation_trials(self) -> int:
        return int(self.task.n_validation_trials)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return self.task.validation_trials


class FixedTargetPerturbationTrainingTaskAdapter(AbstractTask):
    """Apply fixed-target perturbation mixture and validation bins to a task."""

    task: object
    config: Any
    validation_bin: str | None = None

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

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.task.get_train_trial(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        key_trial, key_pert = jr.split(key)
        base = self.task.get_train_trial_with_intervenor_params(key_trial, batch_info)
        return apply_training_perturbation_mixture(base, self.config, key_pert, batch_info)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        base = self.task.get_validation_trials(key)
        return apply_validation_bin(base, self.config, self.validation_bin or "nominal")

    @property
    def n_validation_trials(self) -> int:
        return int(self.task.n_validation_trials)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return apply_validation_bin(
            self.task.validation_trials,
            self.config,
            self.validation_bin or "nominal",
        )


def config_from_hps(config: Any) -> FixedTargetPerturbationTrainingConfig:
    """Normalize an hps training-distribution payload to a dataclass."""

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
    )


def calibrated_timing_basis_manifest(config: FixedTargetPerturbationTrainingConfig) -> dict[str, Any]:
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

    plant_bins = [bin_.to_json() for bin_ in DEFAULT_PLANT_TIMING_BINS]
    visible_bins = [bin_.to_json() for bin_ in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS]
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
            "delayed_observation_semantics": (
                "offset to clean delayed measurement before sensory noise, not literal "
                "extra temporal delay"
            ),
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
                        int(bin_.start_time_index) for bin_ in DEFAULT_PLANT_TIMING_BINS
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
                        int(bin_.start_time_index)
                        for bin_ in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS
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
        "controller_visible_velocity_scale_m_s": DEFAULT_CONTROLLER_VISIBLE_VELOCITY_SCALE_M_S,
        "open_loop_peak_delta_x_per_unit": DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT,
        "amplitude_level_randomization": (
            "disabled in calibrated_timing mode; the declared physical_level fixes the "
            "effect-size target"
        ),
        "artifact_dependency": "none_at_runtime",
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
        "single_family_bins": list(SINGLE_FAMILY_BINS),
        "mild_combined_families": list(MILD_COMBINED_FAMILIES),
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
                    "add a duration-limited pulse to one random command-channel "
                    "component over a calibrated timing bin"
                    if config.calibrated_timing
                    else (
                        "add a duration-limited pulse to one random command-channel "
                        "component over a random start time"
                    )
                ),
                "randomized": [
                    "axis",
                    "timing_bin" if config.calibrated_timing else "start_time",
                    "sign",
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
            "delayed_observation": {
                "base_amplitude": float(config.delayed_observation_offset_m),
                "units": "m_or_m_s_channel_units",
                "emission": (
                    "add an offset pulse on one random 4D delayed-observation component; "
                    "calibrated timing mode uses controller-visible 5-step bins. "
                    "This is an offset to clean delayed measurement before sensory "
                    "noise, not literal extra temporal delay"
                    if config.calibrated_timing
                    else (
                        "add an offset pulse on one random 4D delayed-observation "
                        "component; current training uses full-trial duration"
                    )
                ),
                "randomized": [
                    "observation_component",
                    "timing_bin" if config.calibrated_timing else "start_time",
                    "sign",
                    "physical_level" if config.calibrated_timing else "amplitude_level",
                ],
                "duration_steps": (
                    int(config.pulse_duration_steps) if config.calibrated_timing else "full_trial"
                ),
            },
        },
        "validation_difference": (
            "Validation bins are deterministic family-separated probes, not a replay "
            "of the full training mixture. They expose separate nominal, "
            "single-family, and mild-combined bins for checkpoint selection/reporting."
        ),
    }


def config_from_target_hps(config: Any) -> TargetRelativeMultiTargetTrainingConfig:
    """Normalize an hps target-distribution payload to a dataclass."""

    force_filter_feedback = getattr(config, "force_filter_feedback", False)
    if not isinstance(force_filter_feedback, bool):
        force_filter_feedback = bool(getattr(force_filter_feedback, "enabled", False))
    return TargetRelativeMultiTargetTrainingConfig(
        enabled=bool(getattr(config, "enabled", False)),
        force_filter_feedback=bool(force_filter_feedback),
        seen_directions_deg=tuple(
            float(x)
            for x in getattr(
                config,
                "seen_directions_deg",
                DEFAULT_SEEN_TARGET_DIRECTIONS_DEG,
            )
        ),
        held_out_directions_deg=tuple(
            float(x)
            for x in getattr(
                config,
                "held_out_directions_deg",
                DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG,
            )
        ),
        seen_amplitudes_m=tuple(
            float(x)
            for x in getattr(
                config,
                "seen_amplitudes_m",
                DEFAULT_SEEN_TARGET_AMPLITUDES_M,
            )
        ),
        held_out_amplitudes_m=tuple(
            float(x)
            for x in getattr(
                config,
                "held_out_amplitudes_m",
                DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M,
            )
        ),
        original_target_anchor_m=tuple(
            float(x)
            for x in getattr(
                config,
                "original_target_anchor_m",
                ORIGINAL_TARGET_ANCHOR_M,
            )
        ),
    )


def config_from_broad_epsilon_hps(config: Any) -> BroadFullStateEpsilonTrainingConfig:
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


def config_from_broad_epsilon_pgd_hps(config: Any) -> PgdFullStateEpsilonTrainingConfig:
    """Normalize an hps PGD broad-epsilon payload to a dataclass."""

    inner = _payload_get(config, "inner_maximizer", None)
    return PgdFullStateEpsilonTrainingConfig(
        enabled=bool(_payload_get(config, "enabled", False)),
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
    )


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


def make_broad_epsilon_pgd_pre_step(config: Any) -> Callable | None:
    """Return a Feedbax pre-step hook for training-time broad-epsilon PGD."""

    cfg = config_from_broad_epsilon_pgd_hps(config)
    if not cfg.enabled:
        return None

    def pre_step_fn(task, model, trial_specs, loss_func, keys_model):
        specs, _ = run_broad_epsilon_pgd_inner_maximizer(
            task,
            model,
            trial_specs,
            loss_func,
            keys_model,
            cfg,
            return_diagnostics=False,
        )
        return specs

    return pre_step_fn


def run_broad_epsilon_pgd_inner_maximizer(
    task: Any,
    model: Any,
    trial_specs: TaskTrialSpec,
    loss_func: Any,
    keys_model: Any,
    config: Any,
    *,
    return_diagnostics: bool = False,
) -> tuple[TaskTrialSpec, dict[str, jnp.ndarray]]:
    """Run the PGD inner maximizer and optionally return compact scalar diagnostics."""

    cfg = config_from_broad_epsilon_pgd_hps(config)
    specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=cfg.epsilon_dim)
    base_epsilon = jnp.asarray(specs.inputs["epsilon"])
    radius = _broad_epsilon_l2_radius(specs, cfg).astype(base_epsilon.dtype)
    time_mask = _epsilon_time_mask(specs, base_epsilon, cfg.movement_epoch_only)
    delta = jnp.zeros_like(base_epsilon)

    def objective(delta_candidate):
        candidate = _set_input(specs, "epsilon", base_epsilon + delta_candidate * time_mask)
        candidate_states = task.eval_trials(model, candidate, keys_model)
        return loss_func(candidate_states, candidate, model).total

    def objective_and_grad(delta_candidate):
        return jax.value_and_grad(objective)(delta_candidate)

    zero_delta = jnp.zeros_like(base_epsilon)
    objective_initial, grad_initial = objective_and_grad(zero_delta)
    grad_initial = grad_initial * time_mask
    step_radius = _expand_radius(
        radius * jnp.asarray(cfg.step_size_fraction, dtype=base_epsilon.dtype),
        base_epsilon.ndim,
    )

    def proposal_from_gradient(delta_current, grad_current):
        step = _normalize_flattened_per_trial(grad_current) * step_radius
        return _project_flattened_per_trial_l2_ball(
            (delta_current + step) * time_mask,
            radius,
        )

    def select_best(best_delta, best_objective, candidate_delta, candidate_objective):
        improved = candidate_objective > best_objective
        best_delta = jnp.where(
            _expand_bool_like(improved, candidate_delta),
            candidate_delta,
            best_delta,
        )
        best_objective = jnp.where(improved, candidate_objective, best_objective)
        return best_delta, best_objective

    def body(_, state):
        delta_current, _current_objective, grad_current, best_delta, best_objective = state
        proposal = proposal_from_gradient(delta_current, grad_current)
        proposal_objective, proposal_grad = objective_and_grad(proposal)
        proposal_grad = proposal_grad * time_mask
        best_delta, best_objective = select_best(
            best_delta,
            best_objective,
            proposal,
            proposal_objective,
        )
        return proposal, proposal_objective, proposal_grad, best_delta, best_objective

    delta_current = zero_delta
    current_objective = objective_initial
    grad_current = grad_initial
    best_delta = zero_delta
    objective_best = objective_initial
    if int(cfg.n_steps) > 1:
        delta_current, current_objective, grad_current, best_delta, objective_best = (
            jax.lax.fori_loop(
                0,
                int(cfg.n_steps) - 1,
                body,
                (delta_current, current_objective, grad_current, best_delta, objective_best),
            )
        )

    final_delta = proposal_from_gradient(delta_current, grad_current)
    objective_final_endpoint = objective(final_delta)
    best_delta, objective_best = select_best(
        best_delta,
        objective_best,
        final_delta,
        objective_final_endpoint,
    )
    delta = jax.lax.stop_gradient(best_delta * time_mask)
    updated = _set_input(specs, "epsilon", base_epsilon + delta)
    if not return_diagnostics:
        return updated, {}

    objective_selected = objective_best
    delta_norm = _flattened_per_trial_norm(delta).astype(radius.dtype)
    ratio = delta_norm / jnp.maximum(radius, jnp.asarray(1e-12, dtype=radius.dtype))
    boundary = ratio >= jnp.asarray(1.0 - 1e-4, dtype=ratio.dtype)
    diagnostics = {
        "radius_mean": jnp.mean(radius),
        "radius_max": jnp.max(radius),
        "epsilon_norm_mean": jnp.mean(delta_norm),
        "epsilon_norm_max": jnp.max(delta_norm),
        "epsilon_norm_radius_ratio_mean": jnp.mean(ratio),
        "epsilon_norm_radius_ratio_max": jnp.max(ratio),
        "inner_objective_before": jnp.asarray(objective_initial),
        "inner_objective_after": jnp.asarray(objective_selected),
        "inner_objective_improvement": jnp.asarray(objective_selected - objective_initial),
        "inner_objective_best": jnp.asarray(objective_best),
        "inner_objective_final_endpoint": jnp.asarray(objective_final_endpoint),
        "inner_objective_final_endpoint_gap": jnp.asarray(
            objective_best - objective_final_endpoint
        ),
        "boundary_fraction": jnp.mean(boundary.astype(radius.dtype)),
        "n_steps": jnp.asarray(cfg.n_steps, dtype=jnp.float32),
        "step_size_fraction_of_l2_radius": jnp.asarray(
            cfg.step_size_fraction,
            dtype=jnp.float32,
        ),
    }
    return updated, diagnostics


def _expand_bool_like(mask: jnp.ndarray | bool, values: jnp.ndarray) -> jnp.ndarray:
    mask_array = jnp.asarray(mask)
    while mask_array.ndim < values.ndim:
        mask_array = jnp.expand_dims(mask_array, axis=-1)
    return mask_array


def _ensure_broad_epsilon_input(
    trial_specs: TaskTrialSpec,
    *,
    epsilon_dim: int = BROAD_EPSILON_DIM,
) -> TaskTrialSpec:
    """Ensure an epsilon input exists and is broadcast to the trial batch."""

    if "epsilon" not in trial_specs.inputs:
        zeros = jnp.zeros(
            (*_batch_shape(trial_specs), int(trial_specs.timeline.n_steps), int(epsilon_dim)),
            dtype=jnp.float32,
        )
        trial_specs = _set_input(trial_specs, "epsilon", zeros)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    if epsilon.shape[-1] != int(epsilon_dim):
        raise ValueError(
            f"PGD broad full-state epsilon expects a {int(epsilon_dim)}D process "
            "epsilon input; "
            f"got trailing dimension {epsilon.shape[-1]}."
        )
    batch_shape = _batch_shape(trial_specs)
    if batch_shape and epsilon.shape[: len(batch_shape)] != batch_shape:
        epsilon = jnp.broadcast_to(epsilon, (*batch_shape, *epsilon.shape[-2:]))
        trial_specs = _set_input(trial_specs, "epsilon", epsilon)
    return trial_specs


def _flattened_per_trial_norm(x: jnp.ndarray) -> jnp.ndarray:
    axes = tuple(range(max(x.ndim - 2, 0), x.ndim))
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axes))


def _expand_radius(radius: jnp.ndarray, ndim: int) -> jnp.ndarray:
    while radius.ndim < ndim:
        radius = jnp.expand_dims(radius, axis=-1)
    return radius


def _normalize_flattened_per_trial(x: jnp.ndarray) -> jnp.ndarray:
    norms = _expand_radius(_flattened_per_trial_norm(x), x.ndim)
    return x / jnp.maximum(norms, jnp.asarray(1e-12, dtype=x.dtype))


def _project_flattened_per_trial_l2_ball(
    x: jnp.ndarray,
    radius: jnp.ndarray,
) -> jnp.ndarray:
    radius_expanded = _expand_radius(radius.astype(x.dtype), x.ndim)
    norms = _expand_radius(_flattened_per_trial_norm(x).astype(x.dtype), x.ndim)
    scale = jnp.minimum(
        1.0,
        radius_expanded / jnp.maximum(norms, jnp.asarray(1e-12, dtype=x.dtype)),
    )
    return x * scale


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


def _epsilon_time_mask(
    trial_specs: TaskTrialSpec,
    epsilon: jnp.ndarray,
    movement_epoch_only: bool,
) -> jnp.ndarray:
    """Return a broadcastable epsilon mask over ``[..., time, component]``."""

    if not movement_epoch_only:
        return jnp.ones_like(epsilon)
    bounds = trial_specs.timeline.epoch_bounds
    if bounds is None:
        raise ValueError("movement-epoch broad-epsilon masking requires epoch bounds.")
    bounds = jnp.asarray(bounds)
    t = jnp.arange(epsilon.shape[-2], dtype=bounds.dtype)
    if bounds.ndim == 1:
        time_mask = (t >= bounds[-2]) & (t < bounds[-1])
    else:
        start = bounds[..., -2]
        end = bounds[..., -1]
        time_mask = (t >= jnp.expand_dims(start, -1)) & (t < jnp.expand_dims(end, -1))
    while time_mask.ndim < epsilon.ndim - 1:
        time_mask = jnp.expand_dims(time_mask, axis=0)
    return jnp.expand_dims(time_mask.astype(epsilon.dtype), axis=-1)


def install_perturbation_training_graph_adapters(
    model: Any,
    *,
    force_filter_feedback: bool = False,
) -> Any:
    """Install the fixed external additive channel adapters on a C&S GRU graph."""

    return materialize_additive_channel_adapters_on_graph(
        model,
        tuple(graph_adapter_specs(force_filter_feedback=force_filter_feedback).values()),
    )


def apply_training_target_distribution(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    """Apply one PRNG-driven static target draw from the seen target set."""

    cfg = config_from_target_hps(config)
    targets = jnp.asarray(cfg.seen_targets_m, dtype=jnp.float32)
    batch_shape = _batch_shape(trial_specs)
    index = jr.randint(key, batch_shape, 0, targets.shape[0])
    target = targets[index]
    return _with_static_target(trial_specs, target, metadata=None)


def apply_validation_target_distribution(
    trial_specs: TaskTrialSpec,
    config: Any,
) -> TaskTrialSpec:
    """Return validation trials covering original, seen, and held-out targets."""

    cfg = config_from_target_hps(config)
    targets = jnp.asarray(cfg.validation_targets_m, dtype=jnp.float32)
    trial_specs = _with_static_target(trial_specs, targets, metadata=None)
    extra = dict(trial_specs.extra or {})
    extra["target_relative_multitarget_bins"] = target_relative_validation_bins(cfg)
    extra["target_relative_input_contract"] = target_relative_input_contract(
        force_filter_feedback=cfg.force_filter_feedback
    )
    return TaskTrialSpec(
        inits=WhereDict(trial_specs.inits),
        inputs=trial_specs.inputs,
        targets=trial_specs.targets,
        intervene=trial_specs.intervene,
        timeline=trial_specs.timeline,
        extra=extra,
    )


def apply_broad_epsilon_training(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    """Add a per-trial L2-projected C&S epsilon sequence to training inputs."""

    cfg = config_from_broad_epsilon_hps(config)
    if not cfg.enabled:
        return trial_specs
    if "epsilon" not in trial_specs.inputs:
        zeros = jnp.zeros(
            (*_batch_shape(trial_specs), int(trial_specs.timeline.n_steps), int(cfg.epsilon_dim)),
            dtype=jnp.float32,
        )
        trial_specs = _set_input(trial_specs, "epsilon", zeros)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    if epsilon.shape[-1] != int(cfg.epsilon_dim):
        raise ValueError(
            f"Broad full-state epsilon expects a {int(cfg.epsilon_dim)}D process "
            "epsilon input; "
            f"got trailing dimension {epsilon.shape[-1]}."
        )
    batch_shape = _batch_shape(trial_specs)
    if batch_shape and epsilon.shape[: len(batch_shape)] != batch_shape:
        epsilon = jnp.broadcast_to(epsilon, (*batch_shape, *epsilon.shape[-2:]))
    time_mask = _epsilon_time_mask(trial_specs, epsilon, cfg.movement_epoch_only)
    draws = jr.normal(key, epsilon.shape, dtype=epsilon.dtype) * time_mask
    flat_axes = tuple(range(max(epsilon.ndim - 2, 0), epsilon.ndim))
    norms = jnp.sqrt(jnp.sum(jnp.square(draws), axis=flat_axes))
    radius = _broad_epsilon_l2_radius(trial_specs, cfg).astype(epsilon.dtype)
    while radius.ndim < epsilon.ndim:
        radius = jnp.expand_dims(radius, axis=-1)
    while norms.ndim < epsilon.ndim:
        norms = jnp.expand_dims(norms, axis=-1)
    broad = draws * (radius / jnp.maximum(norms, jnp.asarray(1e-12, dtype=epsilon.dtype)))
    return _set_input(trial_specs, "epsilon", epsilon + broad)


def apply_training_perturbation_mixture(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
    batch_info=None,
) -> TaskTrialSpec:
    """Apply one PRNG-driven fixed-target perturbation-training batch."""

    cfg = config_from_hps(config)
    specs = graph_adapter_specs(force_filter_feedback=cfg.force_filter_feedback)
    trial_specs = add_zero_graph_channel_inputs(
        trial_specs,
        force_filter_feedback=cfg.force_filter_feedback,
    )
    batch_shape = _batch_shape(trial_specs)
    (
        key_mix,
        key_family,
        key_pos,
        key_vel,
        key_process,
        key_command,
        key_sensory,
        key_delayed,
    ) = jr.split(key, 8)
    mixture = jr.uniform(key_mix, batch_shape)
    single_mask = (
        (mixture >= float(cfg.nominal_fraction))
        & (mixture < float(cfg.nominal_fraction + cfg.single_fraction))
    ).astype(jnp.float32)
    combined_mask = (mixture >= float(cfg.nominal_fraction + cfg.single_fraction)).astype(
        jnp.float32
    )
    family_index = jr.randint(key_family, batch_shape, 0, len(SINGLE_FAMILY_BINS))

    if cfg.calibrated_timing:
        if cfg.movement_age_timing:
            trial_specs = _add_movement_onset_state_offset_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_position",
                ),
                component_offset=0,
                n_components=2,
                active_mask=(
                    single_mask * _family_mask(family_index, "initial_position")
                    + combined_mask * float(cfg.combined_amplitude_scale)
                ),
                key=key_pos,
            )
            trial_specs = _add_movement_onset_state_offset_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_velocity",
                ),
                component_offset=2,
                n_components=2,
                active_mask=single_mask * _family_mask(family_index, "initial_velocity"),
                key=key_vel,
            )
        else:
            trial_specs = _offset_initial_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_position",
                ),
                component_offset=0,
                n_components=2,
                active_mask=(
                    single_mask * _family_mask(family_index, "initial_position")
                    + combined_mask * float(cfg.combined_amplitude_scale)
                ),
                randomize_amplitude_level=False,
                key=key_pos,
            )
            trial_specs = _offset_initial_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_velocity",
                ),
                component_offset=2,
                n_components=2,
                active_mask=single_mask * _family_mask(family_index, "initial_velocity"),
                randomize_amplitude_level=False,
                key=key_vel,
            )
        trial_specs = _add_process_epsilon_calibrated_random_pulse(
            trial_specs,
            cfg,
            active_mask=single_mask * _family_mask(family_index, "process_epsilon"),
            key=key_process,
        )
        trial_specs = _add_graph_channel_calibrated_random_pulse(
            trial_specs,
            cfg,
            specs["command_input"],
            active_mask=(
                single_mask * _family_mask(family_index, "command_input")
                + combined_mask * float(cfg.combined_amplitude_scale)
            ),
            key=key_command,
        )
        trial_specs = _add_graph_channel_calibrated_random_pulse(
            trial_specs,
            cfg,
            specs["sensory_feedback"],
            active_mask=single_mask * _family_mask(family_index, "sensory_feedback"),
            key=key_sensory,
        )
        trial_specs = _add_graph_channel_calibrated_random_pulse(
            trial_specs,
            cfg,
            specs["delayed_observation"],
            active_mask=single_mask * _family_mask(family_index, "delayed_observation"),
            key=key_delayed,
        )
    else:
        trial_specs = _offset_initial_random_components(
            trial_specs,
            base_amount=cfg.initial_position_offset_m,
            component_offset=0,
            n_components=2,
            active_mask=(
                single_mask * _family_mask(family_index, "initial_position")
                + combined_mask * float(cfg.combined_amplitude_scale)
            ),
            key=key_pos,
        )
        trial_specs = _offset_initial_random_components(
            trial_specs,
            base_amount=cfg.initial_velocity_offset_m_s,
            component_offset=2,
            n_components=2,
            active_mask=single_mask * _family_mask(family_index, "initial_velocity"),
            key=key_vel,
        )
        trial_specs = _add_process_epsilon_random_pulse(
            trial_specs,
            base_amount=cfg.process_epsilon_scale,
            active_mask=single_mask * _family_mask(family_index, "process_epsilon"),
            duration=cfg.pulse_duration_steps,
            key=key_process,
        )
        trial_specs = _add_graph_channel_random_pulse(
            trial_specs,
            specs["command_input"],
            base_amount=cfg.command_input_pulse_n,
            active_mask=(
                single_mask * _family_mask(family_index, "command_input")
                + combined_mask * float(cfg.combined_amplitude_scale)
            ),
            duration=cfg.pulse_duration_steps,
            key=key_command,
        )
        trial_specs = _add_graph_channel_random_pulse(
            trial_specs,
            specs["sensory_feedback"],
            base_amount=cfg.sensory_feedback_offset_m,
            active_mask=single_mask * _family_mask(family_index, "sensory_feedback"),
            duration=trial_specs.timeline.n_steps,
            key=key_sensory,
        )
        trial_specs = _add_graph_channel_random_pulse(
            trial_specs,
            specs["delayed_observation"],
            base_amount=cfg.delayed_observation_offset_m,
            active_mask=single_mask * _family_mask(family_index, "delayed_observation"),
            duration=trial_specs.timeline.n_steps,
            key=key_delayed,
        )
    # Train trials are produced inside Feedbax's vmap'd training step, so their
    # PyTree leaves must be JAX values. Keep string/list provenance in run specs
    # and validation sidecars rather than returning it through this dynamic path.
    return trial_specs


def apply_validation_bin(
    trial_specs: TaskTrialSpec,
    config: Any,
    bin_name: str,
) -> TaskTrialSpec:
    """Apply one deterministic validation perturbation bin."""

    cfg = (
        config_from_hps(config)
        if not isinstance(config, FixedTargetPerturbationTrainingConfig)
        else config
    )
    if bin_name == "nominal":
        return _with_perturbation_metadata(
            trial_specs,
            "nominal",
            force_filter_feedback=cfg.force_filter_feedback,
        )
    if bin_name == "mild_combined":
        trial_specs = _apply_single_bin(
            trial_specs,
            cfg,
            "initial_position",
            cfg.combined_amplitude_scale,
        )
        trial_specs = _apply_single_bin(
            trial_specs,
            cfg,
            "command_input",
            cfg.combined_amplitude_scale,
        )
        return _with_perturbation_metadata(
            trial_specs,
            "mild_combined",
            families=("initial_position", "command_input"),
            force_filter_feedback=cfg.force_filter_feedback,
        )
    if bin_name not in SINGLE_FAMILY_BINS:
        raise ValueError(f"Unknown perturbation validation bin {bin_name!r}.")
    return _apply_single_bin(trial_specs, cfg, bin_name, 1.0)


def validation_bin_manifest(config: Any) -> dict[str, Any]:
    """Return validation-bin metadata for run specs and sidecars."""

    cfg = (
        config_from_hps(config)
        if not isinstance(config, FixedTargetPerturbationTrainingConfig)
        else config
    )
    selection_role = (
        "aggregate rollout loss over predeclared held-out perturbation bins selects "
        "checkpoints; analytical action, I/O, and Jacobian metrics are audit-only"
        if cfg.enabled
        else "nominal rollout validation loss selects checkpoints"
    )
    validation_role = (
        "generalized_held_out_perturbation_rollout_loss"
        if cfg.enabled
        else "nominal_rollout_validation_loss"
    )
    return {
        "schema_version": "rlrmp.cs_fixed_target_perturbation_validation_bins.v1",
        "validation_role": validation_role,
        "selection_role": selection_role,
        "nominal_quality_role": (
            "nominal bin remains a reported quality sidecar/gate and is not an "
            "analytical-fidelity selector"
        ),
        "bins": [
            {
                "bin": bin_name,
                "families": _bin_families(bin_name),
                "target_stream_mutated": False,
            }
            for bin_name in VALIDATION_BINS
        ],
        "config": cfg.to_json(),
    }


def target_relative_validation_manifest(config: Any) -> dict[str, Any]:
    """Return target-relative validation-bin metadata for run specs."""

    cfg = config_from_target_hps(config)
    return {
        "schema_version": "rlrmp.cs_target_relative_multitarget_validation_bins.v1",
        "validation_role": "target_relative_multitarget_rollout_loss",
        "selection_role": (
            "rollout loss over original-anchor, seen-target, held-out-target, and "
            "perturbation-emphasis bins selects checkpoints; analytical action and "
            "I/O metrics remain audit-only"
        ),
        "target_centered_scoring": "trial_static_target",
        "bins": target_relative_validation_bins(cfg),
        "input_contract": target_relative_input_contract(
            force_filter_feedback=cfg.force_filter_feedback
        ),
        "config": cfg.to_json(),
    }


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
            "bin": "delayed_observation_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["delayed_observation"],
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
        "sensory_or_delayed_observation_fraction": [0.10, 0.20],
        "process_or_load_fraction": [0.05, 0.15],
        "command_input": "optional_diagnostic_only",
    }


def _apply_single_bin(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
    amplitude_scale: float,
) -> TaskTrialSpec:
    return _with_perturbation_metadata(
        _apply_single_bin_raw(trial_specs, config, bin_name, amplitude_scale),
        bin_name,
        force_filter_feedback=config.force_filter_feedback,
    )


def _apply_single_bin_raw(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
    amplitude_scale: float,
) -> TaskTrialSpec:
    if bin_name == "initial_position":
        amount = _single_bin_amount(
            trial_specs,
            config,
            "initial_position",
        ) * amplitude_scale
        if config.calibrated_timing and config.movement_age_timing:
            return _add_movement_onset_state_offset_pulse(
                trial_specs,
                component=1,
                amount=amount,
            )
        return _offset_initial_vector(trial_specs, axis=1, amount=amount)
    if bin_name == "initial_velocity":
        amount = _single_bin_amount(
            trial_specs,
            config,
            "initial_velocity",
        ) * amplitude_scale
        if config.calibrated_timing and config.movement_age_timing:
            return _add_movement_onset_state_offset_pulse(
                trial_specs,
                component=3,
                amount=amount,
            )
        return _offset_initial_vector(trial_specs, axis=3, amount=amount)
    if bin_name == "process_epsilon":
        return _add_process_epsilon_pulse(
            trial_specs,
            amount=_single_bin_amount(
                trial_specs,
                config,
                "process_epsilon",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "process_epsilon"),
            duration=config.pulse_duration_steps,
        )
    if bin_name == "command_input":
        return _add_graph_channel_pulse(
            trial_specs,
            GRAPH_ADAPTER_SPECS["command_input"],
            amount=_single_bin_amount(
                trial_specs,
                config,
                "command_input",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "command_input"),
            duration=config.pulse_duration_steps,
        )
    if bin_name == "sensory_feedback":
        specs = graph_adapter_specs(force_filter_feedback=config.force_filter_feedback)
        return _add_graph_channel_pulse(
            trial_specs,
            specs["sensory_feedback"],
            amount=_single_bin_amount(
                trial_specs,
                config,
                "sensory_feedback",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "sensory_feedback"),
            duration=(
                config.pulse_duration_steps
                if config.calibrated_timing
                else trial_specs.timeline.n_steps
            ),
        )
    if bin_name == "delayed_observation":
        specs = graph_adapter_specs(force_filter_feedback=config.force_filter_feedback)
        return _add_graph_channel_pulse(
            trial_specs,
            specs["delayed_observation"],
            amount=_single_bin_amount(
                trial_specs,
                config,
                "delayed_observation",
            )
            * amplitude_scale,
            start=_deterministic_validation_start(trial_specs, config, "delayed_observation"),
            duration=(
                config.pulse_duration_steps
                if config.calibrated_timing
                else trial_specs.timeline.n_steps
            ),
        )
    raise ValueError(f"Unsupported perturbation bin {bin_name!r}.")


def _single_bin_amount(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
) -> float | jnp.ndarray:
    if not config.calibrated_timing:
        if bin_name == "initial_position":
            return config.initial_position_offset_m
        if bin_name == "initial_velocity":
            return config.initial_velocity_offset_m_s
        if bin_name == "process_epsilon":
            return config.process_epsilon_scale
        if bin_name == "command_input":
            return config.command_input_pulse_n
        if bin_name == "sensory_feedback":
            return config.sensory_feedback_offset_m
        if bin_name == "delayed_observation":
            return config.delayed_observation_offset_m
    target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
    if bin_name == "initial_position":
        return target_peak_delta_x
    if bin_name == "initial_velocity":
        sensitivity = DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT["initial_velocity_offset"][
            "initial_condition"
        ]
        return target_peak_delta_x / sensitivity
    if bin_name == "process_epsilon":
        sensitivity = DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT["process_epsilon_force_state_xy"][
            TIMING_LABELS_PLANT[0]
        ]
        return target_peak_delta_x / sensitivity
    if bin_name == "command_input":
        sensitivity = DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT["command_input_pulse"][
            TIMING_LABELS_PLANT[0]
        ]
        return target_peak_delta_x / sensitivity
    if bin_name in {"sensory_feedback", "delayed_observation"}:
        return target_peak_delta_x
    raise ValueError(f"Unsupported perturbation bin {bin_name!r}.")


def _cycle_amplitude(
    index: jnp.ndarray,
    *,
    single_indices: tuple[int, ...],
    combined_indices: tuple[int, ...],
    cfg: FixedTargetPerturbationTrainingConfig,
) -> jnp.ndarray:
    single = jnp.zeros_like(index, dtype=jnp.float32)
    for value in single_indices:
        single = single + (index == value).astype(jnp.float32)
    combined = jnp.zeros_like(index, dtype=jnp.float32)
    for value in combined_indices:
        combined = combined + (index == value).astype(jnp.float32)
    return single.astype(jnp.float32) + combined.astype(jnp.float32) * float(
        cfg.combined_amplitude_scale
    )


def _family_mask(family_index: jnp.ndarray, bin_name: PerturbationBin) -> jnp.ndarray:
    return (family_index == SINGLE_FAMILY_BINS.index(bin_name)).astype(jnp.float32)


def _calibrated_initial_amount(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: Literal["initial_position", "initial_velocity"],
) -> jnp.ndarray:
    target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
    if bin_name == "initial_position":
        return target_peak_delta_x
    sensitivity = DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT["initial_velocity_offset"][
        "initial_condition"
    ]
    return target_peak_delta_x / jnp.asarray(sensitivity, dtype=jnp.float32)


def _target_peak_delta_x_m(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
) -> jnp.ndarray:
    reach_length = _trial_reach_length_m(trial_specs)
    return reach_length * jnp.asarray(
        REACH_RELATIVE_LEVELS[config.physical_level],
        dtype=jnp.float32,
    )


def _broad_epsilon_l2_radius(
    trial_specs: TaskTrialSpec,
    config: BroadFullStateEpsilonTrainingConfig,
) -> jnp.ndarray:
    """Return per-trial L2 radius for broad full-state epsilon sampling."""

    radius = jnp.asarray(config.reference_l2_radius, dtype=jnp.float32)
    if not config.reach_length_scaling:
        return jnp.broadcast_to(radius, _batch_shape(trial_specs))
    reach_length = _trial_reach_length_m(trial_specs)
    return radius * (
        reach_length / jnp.asarray(config.nominal_reach_length_m, dtype=reach_length.dtype)
    )


def _trial_reach_length_m(trial_specs: TaskTrialSpec) -> jnp.ndarray:
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    target = jnp.asarray(target_spec.value)
    if target.ndim >= 2:
        target_pos = target[..., -1, :]
    else:
        target_pos = target
    init_vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    init_pos = init_vector[..., :2]
    try:
        delta = target_pos - init_pos
    except TypeError:
        delta = target_pos
    return jnp.linalg.norm(delta, axis=-1)


def _calibrated_timing_indexed_amounts(
    *,
    family: str,
    timing_labels: tuple[str, ...],
    target_peak_delta_x: jnp.ndarray,
    dtype: Any,
) -> jnp.ndarray:
    sensitivities = jnp.asarray(
        [
            DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT[family][timing_label]
            for timing_label in timing_labels
        ],
        dtype=dtype,
    )
    return jnp.expand_dims(jnp.asarray(target_peak_delta_x, dtype=dtype), -1) / sensitivities


def _process_epsilon_sensitivity_table(dtype: Any) -> jnp.ndarray:
    rows = []
    for family in PROCESS_EPSILON_COMPONENT_FAMILIES:
        rows.append(
            [
                DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT[family][timing_label]
                for timing_label in TIMING_LABELS_PLANT
            ]
        )
    return jnp.asarray(rows, dtype=dtype)


def _offset_initial_vector(
    trial_specs: TaskTrialSpec,
    *,
    axis: int,
    amount: float,
) -> TaskTrialSpec:
    vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    updated = vector.at[..., axis].add(jnp.asarray(amount, dtype=vector.dtype))
    return eqx.tree_at(lambda ts: ts.inits["mechanics.vector"], trial_specs, updated)


def _offset_initial_random_components(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float | jnp.ndarray,
    component_offset: int,
    n_components: int,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
    randomize_amplitude_level: bool = True,
) -> TaskTrialSpec:
    key_component, key_sign, key_level = jr.split(key, 3)
    vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, n_components) + int(component_offset)
    sign = _random_sign(key_sign, batch_shape)
    level = (
        _random_amplitude_level(key_level, batch_shape)
        if randomize_amplitude_level
        else jnp.ones(batch_shape, dtype=jnp.float32)
    )
    amount = jnp.asarray(base_amount, dtype=vector.dtype) * sign * level * active_mask
    component_mask = jax.nn.one_hot(component, vector.shape[-1], dtype=vector.dtype)
    updated = vector + _expand_to_rank(amount, vector.ndim) * component_mask
    return eqx.tree_at(lambda ts: ts.inits["mechanics.vector"], trial_specs, updated)


def _add_movement_onset_state_offset_random_components(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float | jnp.ndarray,
    component_offset: int,
    n_components: int,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_sign = jr.split(key, 2)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, n_components) + int(component_offset)
    amount = jnp.asarray(base_amount, dtype=epsilon.dtype) * _random_sign(
        key_sign,
        batch_shape,
    ) * active_mask
    pulse = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=amount,
        duration=1,
        start=_movement_start_index(trial_specs, batch_shape=batch_shape),
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_movement_onset_state_offset_pulse(
    trial_specs: TaskTrialSpec,
    *,
    component: int,
    amount: float | jnp.ndarray,
) -> TaskTrialSpec:
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component_index = jnp.full(batch_shape, int(component), dtype=jnp.int32)
    pulse = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component_index,
        amount=jnp.asarray(amount, dtype=epsilon.dtype),
        duration=1,
        start=_movement_start_index(trial_specs, batch_shape=batch_shape),
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_process_epsilon_pulse(
    trial_specs: TaskTrialSpec,
    *,
    amount: float | jnp.ndarray,
    start: int | jnp.ndarray,
    duration: int,
) -> TaskTrialSpec:
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jnp.full(batch_shape, 5, dtype=jnp.int32)
    updated = epsilon + _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=jnp.asarray(amount, dtype=epsilon.dtype),
        duration=duration,
        start=jnp.asarray(start, dtype=jnp.int32),
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, updated)


def _add_process_epsilon_random_pulse(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float,
    active_mask: jnp.ndarray,
    duration: int,
    starts: tuple[int, ...] | None = None,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_start, key_sign, key_level = jr.split(key, 4)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, epsilon.shape[-1])
    amount = (
        jnp.asarray(base_amount, dtype=epsilon.dtype)
        * _random_sign(key_sign, batch_shape)
        * _random_amplitude_level(key_level, batch_shape)
        * active_mask
    )
    pulse = _random_pulse_tensor(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=amount,
        duration=duration,
        starts=starts,
        key=key_start,
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_process_epsilon_calibrated_random_pulse(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    *,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_start, key_sign = jr.split(key, 3)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, epsilon.shape[-1])
    start, start_index = _sample_pulse_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        duration=config.pulse_duration_steps,
        starts=_plant_timing_starts(),
        timing_basis=_calibrated_timing_basis(trial_specs, config, batch_shape=batch_shape),
        key=key_start,
    )
    target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
    sensitivity = _process_epsilon_sensitivity_table(epsilon.dtype)[component, start_index]
    amount = (
        jnp.asarray(target_peak_delta_x, dtype=epsilon.dtype)
        / sensitivity
        * _random_sign(key_sign, batch_shape)
        * active_mask
    )
    pulse = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=amount,
        duration=config.pulse_duration_steps,
        start=start,
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_graph_channel_pulse(
    trial_specs: TaskTrialSpec,
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    amount: float | jnp.ndarray,
    start: int | jnp.ndarray,
    duration: int,
) -> TaskTrialSpec:
    payload = _zero_graph_payload(trial_specs, spec)
    batch_shape = _batch_shape(trial_specs)
    component = jnp.zeros(batch_shape, dtype=jnp.int32)
    updated = payload + _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        width=payload.shape[-1],
        component=component,
        amount=jnp.asarray(amount, dtype=payload.dtype),
        duration=duration,
        start=jnp.asarray(start, dtype=jnp.int32),
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _add_graph_channel_random_pulse(
    trial_specs: TaskTrialSpec,
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    base_amount: float,
    active_mask: jnp.ndarray,
    duration: int,
    starts: tuple[int, ...] | None = None,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_start, key_sign, key_level = jr.split(key, 4)
    payload = _zero_graph_payload(trial_specs, spec)
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, payload.shape[-1])
    amount = (
        jnp.asarray(base_amount, dtype=payload.dtype)
        * _random_sign(key_sign, batch_shape)
        * _random_amplitude_level(key_level, batch_shape)
        * active_mask
    )
    updated = payload + _random_pulse_tensor(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        width=payload.shape[-1],
        component=component,
        amount=amount,
        duration=duration,
        starts=starts,
        key=key_start,
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _add_graph_channel_calibrated_random_pulse(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_start, key_sign = jr.split(key, 3)
    payload = _zero_graph_payload(trial_specs, spec)
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(
        key_component,
        batch_shape,
        0,
        _randomized_payload_width(spec, config, payload.shape[-1]),
    )
    starts = (
        _plant_timing_starts()
        if spec.label == GRAPH_ADAPTER_SPECS["command_input"].label
        else _controller_visible_timing_starts()
    )
    start, start_index = _sample_pulse_start(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        duration=config.pulse_duration_steps,
        starts=starts,
        timing_basis=_calibrated_timing_basis(trial_specs, config, batch_shape=batch_shape),
        key=key_start,
    )
    if spec.label == GRAPH_ADAPTER_SPECS["command_input"].label:
        target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
        amount_by_timing = _calibrated_timing_indexed_amounts(
            family="command_input_pulse",
            timing_labels=TIMING_LABELS_PLANT,
            target_peak_delta_x=target_peak_delta_x,
            dtype=payload.dtype,
        )
        amount = jnp.take_along_axis(
            amount_by_timing,
            jnp.expand_dims(start_index, axis=-1),
            axis=-1,
        )[..., 0]
    else:
        amount_by_component = _controller_visible_component_amounts(
            trial_specs,
            config,
            dtype=payload.dtype,
        )
        amount = jnp.take_along_axis(
            amount_by_component,
            jnp.expand_dims(component, axis=-1),
            axis=-1,
        )[..., 0]
    amount = amount * _random_sign(key_sign, batch_shape) * active_mask
    updated = payload + _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        width=payload.shape[-1],
        component=component,
        amount=amount,
        duration=config.pulse_duration_steps,
        start=start,
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _controller_visible_component_amounts(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    *,
    dtype: Any,
) -> jnp.ndarray:
    position_amount = _target_peak_delta_x_m(trial_specs, config)
    velocity_amount = jnp.asarray(
        DEFAULT_CONTROLLER_VISIBLE_VELOCITY_SCALE_M_S
        * REACH_RELATIVE_LEVELS[config.physical_level],
        dtype=dtype,
    )
    components = [
        jnp.asarray(position_amount, dtype=dtype),
        jnp.asarray(position_amount, dtype=dtype),
        jnp.broadcast_to(velocity_amount, jnp.shape(position_amount)),
        jnp.broadcast_to(velocity_amount, jnp.shape(position_amount)),
    ]
    if config.force_filter_feedback:
        zero = jnp.zeros_like(jnp.asarray(position_amount, dtype=dtype))
        components.extend([zero, zero])
    return jnp.stack(components, axis=-1)


def _randomized_payload_width(
    spec: AdditiveGraphChannelAdapterSpec,
    config: FixedTargetPerturbationTrainingConfig,
    payload_width: int,
) -> int:
    if config.force_filter_feedback and spec.label in {
        GRAPH_ADAPTER_SPECS["sensory_feedback"].label,
        GRAPH_ADAPTER_SPECS["delayed_observation"].label,
    }:
        return min(4, int(payload_width))
    return int(payload_width)


def _random_pulse_tensor(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    width: int,
    component: jnp.ndarray,
    amount: jnp.ndarray,
    duration: int,
    starts: tuple[int, ...] | None,
    key: PRNGKeyArray,
    dtype: Any,
) -> jnp.ndarray:
    start, _ = _sample_pulse_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        duration=duration,
        starts=starts,
        key=key,
    )
    return _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        width=width,
        component=component,
        amount=amount,
        duration=duration,
        start=start,
        dtype=dtype,
    )


def _sample_pulse_start(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    duration: int,
    starts: tuple[int, ...] | None,
    key: PRNGKeyArray,
    timing_basis: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if starts is None:
        max_start = max(1, int(n_steps) - int(duration) + 1)
        start_index = jr.randint(key, batch_shape, 0, max_start)
        return start_index, start_index
    valid_starts = tuple(
        int(start)
        for start in starts
        if 0 <= int(start) < int(n_steps) and int(start) + int(duration) <= int(n_steps)
    )
    if not valid_starts:
        raise ValueError("At least one calibrated timing-bin start must fit the trial.")
    if valid_starts != starts:
        raise ValueError(
            "Calibrated timing mode requires all declared timing bins to fit the trial."
        )
    start_values = jnp.asarray(valid_starts, dtype=jnp.int32)
    start_index = jr.randint(key, batch_shape, 0, len(valid_starts))
    start = start_values[start_index]
    if timing_basis is not None:
        start = start + jnp.asarray(timing_basis, dtype=jnp.int32)
    return start, start_index


def _pulse_tensor_from_start(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    width: int,
    component: jnp.ndarray,
    amount: jnp.ndarray,
    duration: int,
    start: jnp.ndarray,
    dtype: Any,
) -> jnp.ndarray:
    time = jnp.arange(int(n_steps))
    time_mask = (
        (time >= jnp.expand_dims(start, axis=-1))
        & (time < jnp.expand_dims(start, axis=-1) + int(duration))
    ).astype(dtype)
    component_mask = jax.nn.one_hot(component, int(width), dtype=dtype)
    return (
        _expand_to_rank(amount, len(batch_shape) + 2)
        * jnp.expand_dims(time_mask, axis=-1)
        * jnp.expand_dims(component_mask, axis=-2)
    )


def _plant_timing_starts() -> tuple[int, ...]:
    return tuple(int(bin_.start_time_index) for bin_ in DEFAULT_PLANT_TIMING_BINS)


def _controller_visible_timing_starts() -> tuple[int, ...]:
    return tuple(int(bin_.start_time_index) for bin_ in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS)


def _calibrated_timing_basis(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    *,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray | None:
    if not config.movement_age_timing:
        return None
    return _movement_start_index(trial_specs, batch_shape=batch_shape)


def _movement_start_index(
    trial_specs: TaskTrialSpec,
    *,
    batch_shape: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    bounds = trial_specs.timeline.epoch_bounds
    if batch_shape is None:
        batch_shape = _batch_shape(trial_specs)
    if bounds is None:
        return jnp.zeros(batch_shape, dtype=jnp.int32)
    bounds = jnp.asarray(bounds)
    if bounds.ndim == 1:
        start = jnp.asarray(bounds[-2], dtype=jnp.int32)
        return jnp.broadcast_to(start, batch_shape)
    start = jnp.asarray(bounds[..., -2], dtype=jnp.int32)
    if batch_shape and start.shape != batch_shape:
        start = jnp.broadcast_to(start, batch_shape)
    return start


def _deterministic_validation_start(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
) -> int | jnp.ndarray:
    if not config.calibrated_timing:
        return 0 if bin_name in CONTROLLER_VISIBLE_TIMED_BINS else config.pulse_start_step
    if bin_name in PLANT_TIMED_BINS:
        start = _plant_timing_starts()[0]
    elif bin_name in CONTROLLER_VISIBLE_TIMED_BINS:
        start = _controller_visible_timing_starts()[0]
    else:
        start = 0
    if not config.movement_age_timing:
        return start
    return _movement_start_index(trial_specs) + jnp.asarray(start, dtype=jnp.int32)


def _random_sign(key: PRNGKeyArray, shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.where(jr.bernoulli(key, 0.5, shape), 1.0, -1.0).astype(jnp.float32)


def _random_amplitude_level(key: PRNGKeyArray, shape: tuple[int, ...]) -> jnp.ndarray:
    index = jr.randint(key, shape, 0, len(AMPLITUDE_LEVELS))
    return jnp.asarray(AMPLITUDE_LEVELS, dtype=jnp.float32)[index]


def _expand_to_rank(value: jnp.ndarray, rank: int) -> jnp.ndarray:
    expanded = jnp.asarray(value)
    while expanded.ndim < rank:
        expanded = jnp.expand_dims(expanded, axis=-1)
    return expanded


def _zero_graph_payload(
    trial_specs: TaskTrialSpec,
    spec: AdditiveGraphChannelAdapterSpec,
) -> jnp.ndarray:
    batch_shape = _batch_shape(trial_specs)
    n_steps = int(trial_specs.timeline.n_steps)
    return jnp.zeros(
        (*batch_shape, n_steps, additive_channel_payload_dim(spec)),
        dtype=jnp.float32,
    )


def add_zero_graph_channel_inputs(
    trial_specs: TaskTrialSpec,
    *,
    force_filter_feedback: bool = False,
) -> TaskTrialSpec:
    """Ensure all graph adapter payload inputs exist with zero values."""

    for spec in graph_adapter_specs(force_filter_feedback=force_filter_feedback).values():
        if spec.input_key not in trial_specs.inputs:
            trial_specs = _set_input(
                trial_specs,
                spec.input_key,
                _zero_graph_payload(trial_specs, spec),
            )
    return trial_specs


def _set_input(trial_specs: TaskTrialSpec, key: str, value: Any) -> TaskTrialSpec:
    inputs = dict(trial_specs.inputs)
    inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _with_static_target(
    trial_specs: TaskTrialSpec,
    target: jnp.ndarray,
    *,
    metadata: dict[str, Any] | None,
) -> TaskTrialSpec:
    target_array = jnp.asarray(target)
    n_steps = int(trial_specs.timeline.n_steps)
    batch_shape = target_array.shape[:-1]
    target_sequence = jnp.broadcast_to(
        jnp.expand_dims(target_array, axis=-2),
        (*batch_shape, n_steps, 2),
    )
    loss_target_sequence = _catch_preserving_loss_target_sequence(
        trial_specs,
        target_sequence=target_sequence,
        batch_shape=batch_shape,
        n_steps=n_steps,
    )
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    updated_target_spec = eqx.tree_at(
        lambda spec: spec.value,
        target_spec,
        loss_target_sequence,
    )
    updated_target_spec = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        updated_target_spec,
    )
    targets = dict(trial_specs.targets)
    targets["mechanics.effector.pos"] = updated_target_spec
    inits = {
        key: _broadcast_trial_array(value, batch_shape)
        for key, value in dict(trial_specs.inits).items()
    }
    inputs = dict(trial_specs.inputs)
    if "effector_target" in inputs and hasattr(inputs["effector_target"], "pos"):
        inputs["effector_target"] = eqx.tree_at(
            lambda state: state.pos,
            inputs["effector_target"],
            loss_target_sequence,
        )
        inputs["effector_target"] = jax.tree.map(
            lambda leaf: _broadcast_trial_array(leaf, batch_shape),
            inputs["effector_target"],
        )
    if "task" in inputs and hasattr(inputs["task"], "effector_target"):
        task_inputs = inputs["task"]
        if hasattr(task_inputs.effector_target, "pos"):
            task_inputs = eqx.tree_at(
                lambda task: task.effector_target.pos,
                task_inputs,
                loss_target_sequence,
            )
            task_inputs = jax.tree.map(
                lambda leaf: _broadcast_trial_array(leaf, batch_shape),
                task_inputs,
            )
            inputs["task"] = task_inputs
    inputs["target"] = target_sequence
    inputs = {
        key: (
            value
            if key in {"target", "effector_target", "task"}
            else _broadcast_trial_array(value, batch_shape)
        )
        for key, value in inputs.items()
    }
    timeline = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        trial_specs.timeline,
    )
    intervene = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        trial_specs.intervene,
    )
    extra = _broadcast_trial_extra(trial_specs.extra, batch_shape)
    if metadata is not None:
        extra = {**dict(extra or {}), **metadata}
    return TaskTrialSpec(
        inits=WhereDict(inits),
        inputs=inputs,
        targets=WhereDict(targets),
        intervene=intervene,
        timeline=timeline,
        extra=extra,
    )


def _catch_preserving_loss_target_sequence(
    trial_specs: TaskTrialSpec,
    *,
    target_sequence: jnp.ndarray,
    batch_shape: tuple[int, ...],
    n_steps: int,
) -> jnp.ndarray:
    """Return scored target sequence, preserving no-go catch trials if present."""

    catch_mask = _catch_mask_from_go_input(trial_specs, batch_shape)
    if catch_mask is None:
        return target_sequence
    init_sequence = _initial_position_sequence(
        trial_specs,
        batch_shape=batch_shape,
        n_steps=n_steps,
        dtype=target_sequence.dtype,
    )
    return jnp.where(
        _expand_to_rank(catch_mask, target_sequence.ndim),
        init_sequence,
        target_sequence,
    )


def _catch_mask_from_go_input(
    trial_specs: TaskTrialSpec,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray | None:
    """Return per-trial catch mask from a delayed go-cue input, if available."""

    go_input = dict(trial_specs.inputs).get("input")
    if go_input is None:
        return None
    go = jnp.asarray(go_input)
    if go.ndim == 0:
        return None
    if go.ndim >= 2 and go.shape[-1] == 1:
        any_go = jnp.any(go > 0.5, axis=-2)
        any_go = jnp.squeeze(any_go, axis=-1)
    else:
        any_go = jnp.any(go > 0.5, axis=-1)
    catch_mask = jnp.logical_not(any_go)
    if batch_shape:
        catch_mask = jnp.broadcast_to(catch_mask, batch_shape)
    return catch_mask


def _initial_position_sequence(
    trial_specs: TaskTrialSpec,
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    dtype: Any,
) -> jnp.ndarray:
    """Return the initial effector position broadcast as a time sequence."""

    init_pos = None
    for value in dict(trial_specs.inits).values():
        pos = getattr(value, "pos", None)
        if pos is not None:
            init_pos = jnp.asarray(pos, dtype=dtype)
            break
        if eqx.is_array(value):
            array = jnp.asarray(value, dtype=dtype)
            if array.ndim >= 1 and array.shape[-1] >= 2:
                init_pos = array[..., :2]
                break
    if init_pos is None:
        raise ValueError("Catch-preserving target replacement requires an initial position.")
    init_pos = _broadcast_trial_array(init_pos, batch_shape)
    return jnp.broadcast_to(
        jnp.expand_dims(jnp.asarray(init_pos, dtype=dtype), axis=-2),
        (*batch_shape, int(n_steps), 2),
    )


def _broadcast_trial_array(value: Any, batch_shape: tuple[int, ...]) -> Any:
    if not batch_shape:
        return value
    if not eqx.is_array(value):
        return value
    array = jnp.asarray(value)
    if array.ndim == 0:
        return value
    if array.shape[: len(batch_shape)] == batch_shape:
        return value
    tail = array.shape[-1:] if array.ndim <= 2 else array.shape[-2:]
    try:
        return jnp.broadcast_to(array, (*batch_shape, *tail))
    except ValueError:
        return value


def _broadcast_trial_extra(
    extra: Mapping[str, Any] | None,
    batch_shape: tuple[int, ...],
) -> dict[str, Any] | None:
    """Broadcast array-valued TaskTrialSpec metadata to a rewritten trial bank."""

    if extra is None:
        return None
    if not batch_shape:
        return dict(extra)
    result: dict[str, Any] = {}
    for key, value in dict(extra).items():
        if not eqx.is_array(value):
            result[key] = value
            continue
        array = jnp.asarray(value)
        if array.shape[: len(batch_shape)] == batch_shape:
            result[key] = value
        elif array.ndim <= 1 and array.size == 1:
            result[key] = jnp.broadcast_to(jnp.reshape(array, ()), batch_shape)
        else:
            result[key] = _broadcast_trial_array(value, batch_shape)
    return result


def _with_perturbation_metadata(
    trial_specs: TaskTrialSpec,
    bin_name: str,
    *,
    families: tuple[str, ...] | None = None,
    force_filter_feedback: bool = False,
) -> TaskTrialSpec:
    trial_specs = add_zero_graph_channel_inputs(
        trial_specs,
        force_filter_feedback=force_filter_feedback,
    )
    extra = dict(trial_specs.extra or {})
    extra["perturbation_training_bin"] = bin_name
    extra["perturbation_training_families"] = list(families or _bin_families(bin_name))
    return TaskTrialSpec(
        inits=WhereDict(trial_specs.inits),
        inputs=trial_specs.inputs,
        targets=trial_specs.targets,
        intervene=trial_specs.intervene,
        timeline=trial_specs.timeline,
        extra=extra,
    )


def _batch_shape(trial_specs: TaskTrialSpec) -> tuple[int, ...]:
    target = trial_specs.targets["mechanics.effector.pos"].value
    return tuple(target.shape[:-2]) if target.ndim >= 3 else ()


def _bin_families(bin_name: str) -> tuple[str, ...]:
    if bin_name == "nominal":
        return ()
    if bin_name == "mild_combined":
        return MILD_COMBINED_FAMILIES
    return (bin_name,)


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


def planned_fixed_target_perturbation_rows(
    *,
    experiment: str = "aacb9ed",
) -> list[dict[str, Any]]:
    """Return the first two planned local perturbation-generalized run rows."""

    rows = []
    for lr_label, lr in (("lr1e-3", 1e-3), ("lr3e-3", 3e-3)):
        run = f"fixed_target_random_perturb_fullqrf_warmcos__{lr_label}_clip5_b64"
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "controller_lr": lr,
                "batch_size": 64,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "loss_objective": "full_analytical_qrf",
                "lr_schedule": "warmup_cosine",
                "perturbation_training": PERTURBATION_TRAINING_MODE,
                "checkpoint_selection": "generalized_held_out_perturbation_validation",
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_cs_nominal_gru.py",
                    "--issue",
                    "aacb9ed",
                    "--output-dir",
                    f"_artifacts/{experiment}/runs/{run}",
                    "--n-train-batches",
                    "12000",
                    "--batch-size",
                    "64",
                    "--controller-lr",
                    str(lr),
                    "--gradient-clip-norm",
                    "5",
                    "--lr-warmup-batches",
                    "500",
                    "--lr-warmup-init-fraction",
                    "0.1",
                    "--lr-cosine-alpha",
                    "0.01",
                    "--n-replicates",
                    "5",
                    "--loss-objective",
                    "full_analytical_qrf",
                    "--perturbation-training",
                    "--full-train",
                    "--resume",
                ],
            }
        )
    return rows


def planned_target_relative_multitarget_rows(
    *,
    experiment: str = "ba82f3d",
) -> list[dict[str, Any]]:
    """Return the planned target-relative smoke/main run rows."""

    rows = [
        {
            "experiment": experiment,
            "run": "target_relative_multitarget_fullqrf_smoke",
            "controller_lr": 1e-3,
            "batch_size": 2,
            "gradient_clip_norm": 5.0,
            "n_replicates": 1,
            "loss_objective": "full_analytical_qrf",
            "row_kind": "smoke",
            "training": TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
            "command": [
                "uv",
                "run",
                "python",
                "scripts/train_cs_nominal_gru.py",
                "--issue",
                "ba82f3d",
                "--output-dir",
                f"/tmp/{experiment}_target_relative_smoke",
                "--target-relative-multitarget",
                "--perturbation-training",
                "--loss-objective",
                "full_analytical_qrf",
                "--controller-lr",
                "0.001",
                "--gradient-clip-norm",
                "5",
                "--smoke",
                "--full-train",
                "--resume",
            ],
        }
    ]
    for lr_label, lr in (("lr1e-3", 1e-3), ("lr3e-3", 3e-3)):
        run = f"target_relative_multitarget_fullqrf_warmcos__{lr_label}_clip5_b64"
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "controller_lr": lr,
                "batch_size": 64,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "loss_objective": "full_analytical_qrf",
                "lr_schedule": "warmup_cosine",
                "row_kind": "main",
                "training": TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
                "checkpoint_selection": "target_relative_multitarget_rollout_validation",
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_cs_nominal_gru.py",
                    "--issue",
                    "ba82f3d",
                    "--output-dir",
                    f"_artifacts/{experiment}/runs/{run}",
                    "--n-train-batches",
                    "12000",
                    "--batch-size",
                    "64",
                    "--controller-lr",
                    str(lr),
                    "--gradient-clip-norm",
                    "5",
                    "--lr-warmup-batches",
                    "500",
                    "--lr-warmup-init-fraction",
                    "0.1",
                    "--lr-cosine-alpha",
                    "0.01",
                    "--n-replicates",
                    "5",
                    "--loss-objective",
                    "full_analytical_qrf",
                    "--target-relative-multitarget",
                    "--perturbation-training",
                    "--full-train",
                    "--resume",
                ],
            }
        )
    return rows


def planned_target_relative_multitarget_h0_rows(
    *,
    experiment: str = "643f101",
) -> list[dict[str, Any]]:
    """Return the planned H0 target-relative smoke/main run rows."""

    rows = [
        {
            "experiment": experiment,
            "run": "target_relative_multitarget_h0_fullqrf_smoke",
            "controller_lr": 1e-3,
            "batch_size": 2,
            "gradient_clip_norm": 5.0,
            "n_replicates": 1,
            "loss_objective": "full_analytical_qrf",
            "row_kind": "smoke",
            "training": TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
            "initial_hidden_encoder": "zero_affine_target_relative_feedback",
            "command": [
                "uv",
                "run",
                "python",
                "scripts/train_cs_nominal_gru.py",
                "--issue",
                "643f101",
                "--output-dir",
                f"/tmp/{experiment}_target_relative_h0_smoke",
                "--target-relative-multitarget",
                "--initial-hidden-encoder",
                "--perturbation-training",
                "--loss-objective",
                "full_analytical_qrf",
                "--controller-lr",
                "0.001",
                "--gradient-clip-norm",
                "5",
                "--smoke",
                "--full-train",
                "--resume",
            ],
        }
    ]
    for lr_label, lr in (("lr1e-3", 1e-3), ("lr3e-3", 3e-3)):
        run = f"target_relative_multitarget_h0_fullqrf_warmcos__{lr_label}_clip5_b64"
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "controller_lr": lr,
                "batch_size": 64,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "n_train_batches": 12000,
                "loss_objective": "full_analytical_qrf",
                "lr_schedule": "warmup_cosine",
                "row_kind": "main",
                "training": TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
                "initial_hidden_encoder": "zero_affine_target_relative_feedback",
                "training_diagnostics": "default_enabled",
                "checkpoint_selection": "target_relative_multitarget_rollout_validation",
                "comparison_rows": [
                    (
                        "results/ba82f3d/runs/"
                        f"target_relative_multitarget_fullqrf_warmcos__{lr_label}_clip5_b64"
                    )
                ],
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_cs_nominal_gru.py",
                    "--issue",
                    "643f101",
                    "--output-dir",
                    f"_artifacts/{experiment}/runs/{run}",
                    "--n-train-batches",
                    "12000",
                    "--batch-size",
                    "64",
                    "--controller-lr",
                    str(lr),
                    "--gradient-clip-norm",
                    "5",
                    "--lr-warmup-batches",
                    "500",
                    "--lr-warmup-init-fraction",
                    "0.1",
                    "--lr-cosine-alpha",
                    "0.01",
                    "--n-replicates",
                    "5",
                    "--loss-objective",
                    "full_analytical_qrf",
                    "--target-relative-multitarget",
                    "--initial-hidden-encoder",
                    "--perturbation-training",
                    "--full-train",
                    "--resume",
                ],
            }
        )
    return rows


def planned_020a65b_h0_pgd_rows(
    *,
    experiment: str = "020a65b",
) -> list[dict[str, Any]]:
    """Return the two local H0 replication rows for the 020a65b PGD lane."""

    common_command = [
        "env",
        "JAX_PLATFORM_NAME=cpu",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_cs_nominal_gru.py",
        "--issue",
        "020a65b",
        "--n-train-batches",
        "12000",
        "--batch-size",
        "64",
        "--controller-lr",
        "0.003",
        "--gradient-clip-norm",
        "5",
        "--lr-warmup-batches",
        "500",
        "--lr-warmup-init-fraction",
        "0.1",
        "--lr-cosine-alpha",
        "0.01",
        "--n-replicates",
        "5",
        "--loss-objective",
        "full_analytical_qrf",
        "--target-relative-multitarget",
        "--initial-hidden-encoder",
        "--force-filter-feedback",
        "--perturbation-training",
        "--perturbation-calibrated-timing",
        "--perturbation-physical-level",
        "small",
    ]

    rows = []
    for pgd_enabled in (False, True):
        pgd_label = "pgd_ofb" if pgd_enabled else "no_pgd"
        run = (
            "target_relative_multitarget_h0_fullqrf_warmcos__"
            f"proprio_cal_small_{pgd_label}_lr3e-3_clip5_b64"
        )
        command = [
            *common_command,
            "--output-dir",
            f"_artifacts/{experiment}/runs/{run}",
        ]
        if pgd_enabled:
            command.extend(
                [
                    "--broad-epsilon-pgd-training",
                    "--broad-epsilon-level",
                    "moderate",
                    "--broad-epsilon-budget-scale",
                    "3.688240371719434",
                    "--broad-epsilon-pgd-steps",
                    "10",
                    "--broad-epsilon-pgd-step-size-fraction",
                    "0.25",
                ]
            )
        full_resume_command = [*command, "--full-train", "--resume"]
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "controller_lr": 3e-3,
                "batch_size": 64,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "n_train_batches": 12000,
                "stop_after_batches": 1000,
                "loss_objective": "full_analytical_qrf",
                "lr_schedule": "warmup_cosine",
                "row_kind": "checkpoint_gate",
                "local_device": "cpu",
                "training": TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
                "force_filter_feedback": True,
                "perturbation_training": "fixed_target_perturbation_calibrated_timing",
                "perturbation_physical_level": "small",
                "initial_hidden_encoder": "zero_affine_target_relative_feedback_plus_force_filter",
                "broad_epsilon_pgd_training": pgd_enabled,
                "broad_epsilon_level": "moderate" if pgd_enabled else None,
                "broad_epsilon_budget_scale": 3.688240371719434 if pgd_enabled else None,
                "broad_epsilon_pgd_steps": 10 if pgd_enabled else None,
                "broad_epsilon_pgd_step_size_fraction": 0.25 if pgd_enabled else None,
                "checkpoint_selection": "target_relative_multitarget_rollout_validation",
                "full_training_contract_command": full_resume_command,
                "command": [
                    *full_resume_command,
                    "--stop-after-batches",
                    "1000",
                ],
            }
        )
    return rows

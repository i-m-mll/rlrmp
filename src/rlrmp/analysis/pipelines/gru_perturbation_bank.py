"""Controller-independent perturbation-response bank for C&S GRU diagnostics."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
)
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.analysis.perturbation_rows import PerturbationChannel, PerturbationSpec
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    TARGET_POS,
    build_canonical_game,
    build_no_integrator_game,
    materialize_reference,
)
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.math.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    simulate_lqg_released_forward,
    simulate_robust_released_forward,
    zero_forward_noise_draws,
    zero_noise_covariances,
)
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    CheckpointSelectionMode,
    load_materialized_fixed_bank_manifest,
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines._selected_eval_rollouts import SelectedEvalRolloutProduct
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import RolloutEvaluation
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    RunFigureInputs,
    repeat_single_validation_trial,
)
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.model.feedbax_channel_adapters import (
    additive_channel_payload_dim,
    additive_channel_provenance,
    find_materialized_additive_channel_adapter,
    materialize_additive_channel_adapter_on_graph,
)
from rlrmp.model.feedback_descriptors import (
    COMPONENT_FORCE_FILTER,
    COMPONENT_POSITION,
    COMPONENT_VELOCITY,
    DESCRIPTOR_PAYLOAD_KEY,
    controller_feedback_axis_index,
    resolve_controller_feedback_view,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.runtime.run_spec_access import require_run_seed


SCHEMA_VERSION = "rlrmp.gru_perturbation_bank.v3"
DEFAULT_BANK_ID = "cs_standard_perturbation_response_v3"
CALIBRATED_BANK_ID = "cs_calibrated_perturbation_response_v3"
DEFAULT_OUTPUT_FILENAME = "gru_perturbation_response_fullqrf_validation_selected_manifest.json"
DEFAULT_NOTE_FILENAME = "gru_perturbation_response_fullqrf_validation_selected.md"
DEFAULT_BULK_SUBDIR = "perturbation_response/gru_fullqrf_validation_selected"
DEFAULT_SOURCE_EXPERIMENT = "5f70333"
DEFAULT_RESULT_EXPERIMENT = "3992394"
DEFAULT_RUN_IDS = (
    "lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64",
    "lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64",
)

PerturbationStatus = Literal["evaluated", "blocked", "not_implemented", "not_applicable"]

GRAPH_ADAPTER_INPUT_PREFIX = "perturbation.channel"
PerturbationEvaluationBackend = Literal["serial"]


@dataclass(frozen=True)
class AdapterResult:
    """Result of applying one perturbation to a TaskTrialSpec."""

    status: PerturbationStatus
    trial_specs: Any
    model: Any | None = None
    reason: str | None = None
    adapter_provenance: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """Return JSON status metadata without the TaskTrialSpec payload."""

        return {
            "status": self.status,
            "reason": self.reason,
            "adapter_provenance": dict(self.adapter_provenance or {}),
        }


def default_cs_perturbation_bank(
    *,
    mode: Literal["raw", "calibrated"] = "raw",
    calibration_level: str | Sequence[str] | None = None,
    calibration_reach: str | float | None = None,
    feedback_scale_manifest: Mapping[str, Any] | None = None,
    feedback_scale_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Return the JSON-serializable default C&S perturbation-response bank."""

    if mode == "calibrated":
        return default_cs_calibrated_perturbation_bank(
            calibration_level=calibration_level,
            calibration_reach=calibration_reach,
            feedback_scale_manifest=feedback_scale_manifest,
            feedback_scale_manifest_path=feedback_scale_manifest_path,
        )
    if mode != "raw":
        raise ValueError(f"unsupported perturbation bank mode {mode!r}")

    from rlrmp.analysis.pipelines.gru_perturbation_calibration import (
        DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS,
        DEFAULT_PLANT_TIMING_BINS,
    )

    perturbations: list[PerturbationSpec] = []
    for family, units, amplitude in (
        ("initial_position_offset", "m", 0.01),
        ("initial_velocity_offset", "m/s", 0.05),
    ):
        for axis in ("x", "y"):
            for sign in (-1, 1):
                perturbations.append(
                    PerturbationSpec(
                        perturbation_id=f"{family}__{axis}_{_sign_label(sign)}",
                        channel="initial_state",
                        family=family,
                        amplitude=amplitude,
                        units=units,
                        axis=axis,
                        basis="plant_cartesian_xy",
                        sign=sign,
                        timing={"epoch": "initial_condition", "time_index": 0},
                        adapter="task_trial_spec.inits",
                        description=(
                            "Offset the external task initial effector "
                            f"{'position' if 'position' in family else 'velocity'}."
                        ),
                        initial_position_case=(
                            "D_current_state_immediately_visible"
                            if family == "initial_position_offset"
                            else None
                        ),
                        calibration_role="raw_default_unscaled_effect_size",
                        timing_bin="initial_condition",
                    )
                )
    process_epsilon_components = (
        ("position", "x", 0, "position_x"),
        ("position", "y", 1, "position_y"),
        ("velocity", "x", 2, "velocity_x"),
        ("velocity", "y", 3, "velocity_y"),
        ("force_state", "x", 4, "force_state_x"),
        ("force_state", "y", 5, "force_state_y"),
        ("integrator", "x", 6, "integrator_x"),
        ("integrator", "y", 7, "integrator_y"),
    )
    for timing_bin in DEFAULT_PLANT_TIMING_BINS:
        start = int(timing_bin.start_time_index)
        duration = int(timing_bin.duration_steps)
        for axis in ("x", "y"):
            for sign in (-1, 1):
                perturbations.append(
                    PerturbationSpec(
                        perturbation_id=(
                            "command_input_pulse__"
                            f"{timing_bin.label}_t{start}_{axis}_{_sign_label(sign)}"
                        ),
                        channel="command_input",
                        family="command_input_pulse",
                        amplitude=1.0,
                        units="N",
                        axis=axis,
                        basis="command_cartesian_force_xy",
                        sign=sign,
                        timing={
                            "epoch": "movement_indexed",
                            "start_time_index": start,
                            "duration_steps": duration,
                            "timing_bin": timing_bin.label,
                            "timing_bin_role": timing_bin.role,
                        },
                        adapter="feedbax.additive_channel_adapter.command_input",
                        description=(
                            "Add a pulse at the post-controller command port that feeds "
                            "mechanics.force. This is not an external load-force row."
                        ),
                        calibration_role="raw_default_requires_same_bank_calibration",
                        timing_bin=timing_bin.label,
                    )
                )
        for sign in (-1, 1):
            perturbations.append(
                PerturbationSpec(
                    perturbation_id=(
                        "target_aligned_lateral_command_load_pulse__"
                        f"{timing_bin.label}_t{start}_{_sign_label(sign)}"
                    ),
                    channel="command_input",
                    family="target_aligned_lateral_command_load_pulse",
                    amplitude=1.0,
                    units="N",
                    axis="y",
                    basis="target_aligned_radial_tangential_force",
                    sign=sign,
                    timing={
                        "epoch": "movement_indexed",
                        "start_time_index": start,
                        "duration_steps": duration,
                        "timing_bin": timing_bin.label,
                        "timing_bin_role": timing_bin.role,
                    },
                    adapter="feedbax.additive_channel_adapter.command_input",
                    description=(
                        "Human-protocol-like lateral mechanical load pulse. In the "
                        "canonical +x reach this is the plant-input y axis; response "
                        "summaries report target-relative radial/tangential components."
                    ),
                    calibration_role="raw_default_requires_same_bank_calibration",
                    timing_bin=timing_bin.label,
                    semantic_family="human_protocol_like_lateral_mechanical_load",
                    channel_provenance={
                        "information_structure": "external_load_after_controller_command",
                        "target_relative_axis_role": "tangential",
                        "target_relative_basis": "canonical_plus_x_reach",
                        "closest_graph_compatible_equivalent": (
                            "post-controller command-input offset on "
                            "efferent.output -> mechanics.force"
                        ),
                    },
                )
            )
        for component_family, axis, epsilon_index, epsilon_component in process_epsilon_components:
            for sign in (-1, 1):
                perturbations.append(
                    PerturbationSpec(
                        perturbation_id=(
                            "process_epsilon_pulse__"
                            f"{epsilon_component}__{timing_bin.label}_t{start}_"
                            f"{_sign_label(sign)}"
                        ),
                        channel="process_epsilon",
                        family=f"process_epsilon_{component_family}_xy",
                        amplitude=0.01,
                        units="epsilon",
                        axis=axis,
                        basis="cs_lss_process_epsilon_current_physical_block",
                        sign=sign,
                        timing={
                            "epoch": "movement_indexed",
                            "start_time_index": start,
                            "duration_steps": duration,
                            "timing_bin": timing_bin.label,
                            "timing_bin_role": timing_bin.role,
                        },
                        adapter="task_trial_spec.inputs['epsilon']",
                        description=(
                            "Add a pulse on the C&S LSS mechanics.epsilon input, which "
                            "is injected through the plant B_w process channel. The "
                            f"component is {epsilon_component} at epsilon index "
                            f"{epsilon_index}."
                        ),
                        epsilon_component=epsilon_component,
                        epsilon_index=epsilon_index,
                        calibration_role="raw_coordinate_not_scale_normalized",
                        timing_bin=timing_bin.label,
                        semantic_family=(
                            "human_protocol_like_lateral_process_load"
                            if epsilon_component == "force_state_y"
                            else None
                        ),
                        channel_provenance=(
                            {
                                "target_relative_axis_role": "tangential",
                                "target_relative_basis": "canonical_plus_x_reach",
                                "closest_graph_compatible_equivalent": (
                                    "process epsilon pulse on the force-state y "
                                    "component of the current physical block"
                                ),
                            }
                            if epsilon_component == "force_state_y"
                            else None
                        ),
                    )
                )
    for timing_bin in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS:
        start = int(timing_bin.start_time_index)
        duration = int(timing_bin.duration_steps)
        for channel, family, basis, description, semantic_family, provenance in (
            (
                "sensory_feedback",
                "sensory_feedback_offset",
                "sensory_feedback_named_channel",
                "Offset the external post-noise sensory channel between "
                "sensory.output and net.feedback.",
                None,
                {
                    "information_structure": "post_noise_controller_visible_feedback",
                    "insertion_point": "sensory.output -> net.feedback",
                },
            ),
        ):
            for component in resolve_controller_feedback_view(
                None,
                feedback_dim=6,
                source="default_gru_perturbation_bank",
            ).iter_components():
                amplitude = _default_feedback_amplitude(component.component_id)
                for axis in component.axes:
                    for sign in (-1, 1):
                        axis_role = _target_relative_axis_role(axis)
                        feedback_index = _controller_visible_feedback_index(
                            component.component_id,
                            axis,
                        )
                        perturbations.append(
                            PerturbationSpec(
                                perturbation_id=(
                                    f"{family}__{component.component_id}__"
                                    f"{timing_bin.label}_t{start}_{axis}_{_sign_label(sign)}"
                                ),
                                channel=channel,
                                family=family,
                                amplitude=amplitude,
                                units=component.units or "model_feedback_units",
                                axis=axis,
                                basis=basis,
                                sign=sign,
                                timing={
                                    "epoch": "controller_visible",
                                    "start_time_index": start,
                                    "duration_steps": duration,
                                    "timing_bin": timing_bin.label,
                                    "timing_bin_role": timing_bin.role,
                                },
                                adapter=f"feedbax.additive_channel_adapter.{channel}",
                                description=(
                                    f"{description} This row is a signed "
                                    f"target-relative {axis_role} {component.component_id} "
                                    "false-feedback probe in the canonical +x reach."
                                ),
                                calibration_role="raw_default_requires_same_bank_calibration",
                                timing_bin=timing_bin.label,
                                semantic_family=semantic_family or "false_feedback_offset",
                                channel_provenance={
                                    **provenance,
                                    "feedback_quantity": component.component_id,
                                    "descriptor_id": component.descriptor_id,
                                    "feedback_payload_index": feedback_index,
                                    "force_filter_feedback_only": (
                                        component.component_id == COMPONENT_FORCE_FILTER
                                    ),
                                    "target_relative_axis_role": axis_role,
                                    "target_relative_basis": "canonical_plus_x_reach",
                                    "false_feedback_probe": True,
                                },
                            )
                        )
    perturbations.append(
        PerturbationSpec(
            perturbation_id="target_stream_jump__x_pos",
            channel="target_stream",
            family="target_stream_jump",
            amplitude=0.01,
            units="m",
            axis="x",
            basis="target_cartesian_xy",
            sign=1,
            timing={"epoch": "adapter_defined"},
            adapter="not_applicable_current_fixed_target_checkpoint",
            description="blocked because current C&S GRU input is scalar SISU, not a target stream",
            calibration_role="raw_default_requires_same_bank_calibration",
            timing_bin="not_applicable",
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "bank_id": DEFAULT_BANK_ID,
        "controller_independence": (
            "Perturbations are declared on task, plant, sensory, observation, or "
            "target interfaces. GRU hidden state, readout state, and controller "
            "input tensors are not edited directly."
        ),
        "legacy_migration": {
            "v2": (
                "v3 preserves v2 row identity and status fields while adding "
                "response-shape summaries, target-relative alignment summaries, "
                "initial-position information-contract metadata, and calibration "
                "metadata hooks."
            ),
            "plant_force": (
                "Deprecated v1 channel name. The C&S LSS graph path is "
                "net.output -> efferent -> mechanics.force, with the force/filter "
                "state inside mechanics, so the former plant_force_pulse rows are "
                "command_input_pulse rows in v2. True process rows use "
                "process_epsilon_pulse through mechanics.epsilon / B_w."
            ),
        },
        "graphspec_alignment": {
            "named_channels": [
                "initial_state",
                "command_input",
                "process_epsilon",
                "sensory_feedback",
                "target_stream",
            ],
            "adapter_contract": (
                "Graph-channel and process-epsilon rows record Feedbax additive "
                "channel adapter specs that can be materialized onto the current "
                "eager graph or serialized as GraphSpec named-channel adapters."
            ),
            "feedbax_additive_channel_adapters": {
                "command_input_pulse": (
                    "Feedbax additive edge adapter on efferent.output -> mechanics.force "
                    "with the row payload supplied from trial_specs.inputs."
                ),
                "process_epsilon_pulse": (
                    "Feedbax additive input adapter targeting mechanics.epsilon / B_w; "
                    "the base epsilon input is preserved and the perturbation payload is "
                    "supplied as a separate named input. Rows declare epsilon_component "
                    "and epsilon_index over the canonical current physical block "
                    "[px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]."
                ),
                "sensory_feedback_offset": (
                    "Feedbax additive edge adapter on sensory.output -> net.feedback, "
                    "representing sensory_feedback after sensory noise and "
                    "before the controller feedback port."
                ),
                "target_stream_jump": (
                    "Deferred: current fixed-target C&S GRU checkpoints do not consume "
                    "a target-position input stream."
                ),
            },
        },
        "initial_position_information_contracts": _initial_position_contract_manifest(),
        "target_relative_alignment": {
            "default": "radial_tangential_when_target_and_current_geometry_available",
            "basis": (
                "Per-time radial axis points from current base hand position to the "
                "current target position; tangential is the signed 2D orthogonal axis."
            ),
            "missing_inputs_status": "not_available",
        },
        "calibration_metadata_hooks": {
            "status": "declared_unbound",
            "coordinating_issue": "1ad3c16",
            "policy": (
                "Raw amplitudes are not interpreted as cross-coordinate normalized "
                "effect sizes. Same-bank open-loop, extLQG, and GRU calibration "
                "metadata can be attached here once parent coordination materializes "
                "the calibrated amplitude sets."
            ),
            "fields": [
                "calibrated_amplitude_set_id",
                "physical_units",
                "effect_size_reference",
                "open_loop_command_replay_response",
                "extlqg_same_bank_response",
                "gru_same_bank_response",
            ],
        },
        "timing_bin_conventions": {
            "plant_side": [timing_bin.to_json() for timing_bin in DEFAULT_PLANT_TIMING_BINS],
            "controller_visible": [
                timing_bin.to_json() for timing_bin in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS
            ],
            "policy": (
                "Initial-condition rows stay at t=0. Plant-side command_input and "
                "process_epsilon rows use early/mid/late bins. Controller-visible "
                "sensory and pre-noise delayed-measurement offsets use "
                "early_visible/mid_visible/late_visible bins."
            ),
        },
        "signed_pairing_rule": "signed_axis_pairs; aggregate absolute and signed responses",
        "perturbations": [spec.to_json() for spec in perturbations],
    }


def default_cs_calibrated_perturbation_bank(
    *,
    calibration_level: str | Sequence[str] | None = None,
    calibration_reach: str | float | None = None,
    feedback_scale_manifest: Mapping[str, Any] | None = None,
    feedback_scale_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Return a reach-relative calibrated C&S perturbation-response bank."""

    from rlrmp.analysis.pipelines.gru_perturbation_calibration import (
        DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS,
        DEFAULT_PLANT_TIMING_BINS,
        DEFAULT_REACH_CALIBRATION_POINTS,
        DEFAULT_REACH_RELATIVE_LEVELS,
        calibrated_amplitude_from_unit_sensitivity,
    )
    from rlrmp.data_products.calibration import load_open_loop_calibration

    calibration = load_open_loop_calibration()
    controller_visible_velocity_scale_m_s = calibration.controller_visible_velocity_scale_m_s

    reach = _select_reach_calibration_point(
        calibration_reach,
        reach_points=DEFAULT_REACH_CALIBRATION_POINTS,
    )
    levels = _select_reach_relative_levels(
        calibration_level,
        levels=DEFAULT_REACH_RELATIVE_LEVELS,
    )
    feedback_scale_manifest = _load_feedback_scale_manifest(
        feedback_scale_manifest,
        feedback_scale_manifest_path,
    )
    position_scale = _controller_feedback_component_scale(
        feedback_scale_manifest,
        "position",
        required=False,
    )
    velocity_scale = _controller_feedback_component_scale(
        feedback_scale_manifest,
        "velocity",
        required=False,
    )
    force_filter_scale = _controller_feedback_component_scale(
        feedback_scale_manifest,
        "force_filter",
        required=True,
    )
    perturbations: list[PerturbationSpec] = []

    def provenance(
        *,
        level: Any,
        calibration_role: str,
        open_loop_peak_dx_per_unit_m: float | None = None,
        open_loop_auc_dx_per_unit_m_s: float | None = None,
        target_open_loop_peak_dx_m: float | None = None,
        target_open_loop_auc_dx_m_s: float | None = None,
        native_unit_rule: str | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "calibration_mode": "reach_relative_peak_delta_x",
            "calibration_role": calibration_role,
            "level_name": level.name,
            "level_fraction_of_reach": float(level.fraction_of_reach),
            "reach_label": reach.label,
            "reach_length_m": float(reach.reach_length_m),
        }
        if open_loop_peak_dx_per_unit_m is not None:
            row["open_loop_peak_dx_per_unit_m"] = float(open_loop_peak_dx_per_unit_m)
        if open_loop_auc_dx_per_unit_m_s is not None:
            row["open_loop_auc_dx_per_unit_m_s"] = float(open_loop_auc_dx_per_unit_m_s)
        if target_open_loop_peak_dx_m is not None:
            row["target_open_loop_peak_dx_m"] = float(target_open_loop_peak_dx_m)
        if target_open_loop_auc_dx_m_s is not None:
            row["target_open_loop_auc_dx_m_s"] = float(target_open_loop_auc_dx_m_s)
        if native_unit_rule is not None:
            row["native_unit_rule"] = native_unit_rule
        return row

    for level in levels:
        target_peak = float(reach.reach_length_m) * float(level.fraction_of_reach)
        for family, units in (
            ("initial_position_offset", "m"),
            ("initial_velocity_offset", "m/s"),
        ):
            sensitivity = calibration[family]["initial_condition"]
            amplitude = calibrated_amplitude_from_unit_sensitivity(
                target_peak_delta_x_m=target_peak,
                peak_delta_x_per_unit_m=float(sensitivity),
            )
            for axis in ("x", "y"):
                for sign in (-1, 1):
                    perturbations.append(
                        PerturbationSpec(
                            perturbation_id=(f"{family}__{level.name}__{axis}_{_sign_label(sign)}"),
                            channel="initial_state",
                            family=family,
                            amplitude=amplitude,
                            units=units,
                            axis=axis,
                            basis="plant_cartesian_xy",
                            sign=sign,
                            timing={"epoch": "initial_condition", "time_index": 0},
                            adapter="task_trial_spec.inits",
                            description=(
                                "Reach-relative calibrated initial effector "
                                f"{'position' if 'position' in family else 'velocity'} offset."
                            ),
                            initial_position_case=(
                                "D_current_state_immediately_visible"
                                if family == "initial_position_offset"
                                else None
                            ),
                            timing_bin="initial_condition",
                            calibration_provenance=provenance(
                                level=level,
                                calibration_role="reach_relative_calibrated_open_loop",
                                open_loop_peak_dx_per_unit_m=float(sensitivity),
                                target_open_loop_peak_dx_m=target_peak,
                            ),
                        )
                    )

        for timing_bin in DEFAULT_PLANT_TIMING_BINS:
            start = int(timing_bin.start_time_index)
            duration = int(timing_bin.duration_steps)
            for family, channel, units, basis, adapter, extra in (
                (
                    "command_input_pulse",
                    "command_input",
                    "N",
                    "command_cartesian_force_xy",
                    "feedbax.additive_channel_adapter.command_input",
                    {},
                ),
                (
                    "target_aligned_lateral_command_load_pulse",
                    "command_input",
                    "N",
                    "target_aligned_radial_tangential_force",
                    "feedbax.additive_channel_adapter.command_input",
                    {"target_relative_axis_role": "tangential", "axis_filter": ("y",)},
                ),
                (
                    "process_epsilon_position_xy",
                    "process_epsilon",
                    "epsilon",
                    "cs_lss_process_epsilon_current_physical_block",
                    "task_trial_spec.inputs['epsilon']",
                    {"epsilon_index_offset": 0, "epsilon_component_prefix": "position"},
                ),
                (
                    "process_epsilon_velocity_xy",
                    "process_epsilon",
                    "epsilon",
                    "cs_lss_process_epsilon_current_physical_block",
                    "task_trial_spec.inputs['epsilon']",
                    {"epsilon_index_offset": 2, "epsilon_component_prefix": "velocity"},
                ),
                (
                    "process_epsilon_force_state_xy",
                    "process_epsilon",
                    "epsilon",
                    "cs_lss_process_epsilon_current_physical_block",
                    "task_trial_spec.inputs['epsilon']",
                    {"epsilon_index_offset": 4, "epsilon_component_prefix": "force_state"},
                ),
                (
                    "process_epsilon_integrator_xy",
                    "process_epsilon",
                    "epsilon",
                    "cs_lss_process_epsilon_current_physical_block",
                    "task_trial_spec.inputs['epsilon']",
                    {"epsilon_index_offset": 6, "epsilon_component_prefix": "integrator"},
                ),
            ):
                sensitivity_family = (
                    "command_input_pulse"
                    if family == "target_aligned_lateral_command_load_pulse"
                    else family
                )
                sensitivity = float(calibration[sensitivity_family][timing_bin.label])
                amplitude = calibrated_amplitude_from_unit_sensitivity(
                    target_peak_delta_x_m=target_peak,
                    peak_delta_x_per_unit_m=sensitivity,
                )
                for axis in ("x", "y"):
                    if axis not in extra.get("axis_filter", ("x", "y")):
                        continue
                    axis_index = _axis_index(axis)
                    for sign in (-1, 1):
                        epsilon_index = None
                        epsilon_component = None
                        if channel == "process_epsilon":
                            epsilon_index = int(extra["epsilon_index_offset"]) + axis_index
                            epsilon_component = f"{extra['epsilon_component_prefix']}_{axis}"
                        perturbations.append(
                            PerturbationSpec(
                                perturbation_id=(
                                    f"{family}__{level.name}__{timing_bin.label}_t{start}_"
                                    f"{axis}_{_sign_label(sign)}"
                                ),
                                channel=channel,  # type: ignore[arg-type]
                                family=family,
                                amplitude=amplitude,
                                units=units,
                                axis=axis,
                                basis=basis,
                                sign=sign,
                                timing={
                                    "epoch": "movement_indexed",
                                    "start_time_index": start,
                                    "duration_steps": duration,
                                    "timing_bin": timing_bin.label,
                                    "timing_bin_role": timing_bin.role,
                                },
                                adapter=adapter,
                                description=(
                                    "Reach-relative calibrated plant-side perturbation "
                                    "using open-loop peak delta-x sensitivity."
                                ),
                                epsilon_component=epsilon_component,
                                epsilon_index=epsilon_index,
                                timing_bin=timing_bin.label,
                                semantic_family=(
                                    "human_protocol_like_lateral_mechanical_load"
                                    if family == "target_aligned_lateral_command_load_pulse"
                                    else (
                                        "human_protocol_like_lateral_process_load"
                                        if epsilon_component == "force_state_y"
                                        else None
                                    )
                                ),
                                channel_provenance=(
                                    {
                                        "information_structure": (
                                            "external_load_after_controller_command"
                                            if channel == "command_input"
                                            else "process_epsilon_current_physical_block"
                                        ),
                                        "target_relative_axis_role": "tangential",
                                        "target_relative_basis": "canonical_plus_x_reach",
                                        "closest_graph_compatible_equivalent": (
                                            "post-controller command-input offset on "
                                            "efferent.output -> mechanics.force"
                                            if channel == "command_input"
                                            else "process epsilon pulse on force-state y"
                                        ),
                                    }
                                    if (
                                        family == "target_aligned_lateral_command_load_pulse"
                                        or epsilon_component == "force_state_y"
                                    )
                                    else None
                                ),
                                calibration_provenance=provenance(
                                    level=level,
                                    calibration_role="reach_relative_calibrated_open_loop",
                                    open_loop_peak_dx_per_unit_m=sensitivity,
                                    target_open_loop_peak_dx_m=target_peak,
                                ),
                            )
                        )

        for timing_bin in DEFAULT_CONTROLLER_VISIBLE_TIMING_BINS:
            start = int(timing_bin.start_time_index)
            duration = int(timing_bin.duration_steps)
            for channel, family, basis, semantic_family, channel_provenance in (
                (
                    "sensory_feedback",
                    "sensory_feedback_offset",
                    "sensory_feedback_named_channel",
                    None,
                    {
                        "information_structure": "post_noise_controller_visible_feedback",
                        "insertion_point": "sensory.output -> net.feedback",
                    },
                ),
            ):
                for axis in ("x", "y"):
                    for sign in (-1, 1):
                        axis_role = _target_relative_axis_role(axis)
                        perturbations.append(
                            PerturbationSpec(
                                perturbation_id=(
                                    f"{family}__position_{level.name}__"
                                    f"{timing_bin.label}_t{start}_{axis}_{_sign_label(sign)}"
                                ),
                                channel=channel,  # type: ignore[arg-type]
                                family=family,
                                amplitude=(
                                    float(
                                        position_scale["reference_scale"]
                                        if position_scale is not None
                                        else reach.reach_length_m
                                    )
                                    * float(level.fraction_of_reach)
                                ),
                                units="m",
                                axis=axis,
                                basis=basis,
                                sign=sign,
                                timing={
                                    "epoch": "controller_visible",
                                    "start_time_index": start,
                                    "duration_steps": duration,
                                    "timing_bin": timing_bin.label,
                                    "timing_bin_role": timing_bin.role,
                                },
                                adapter=f"feedbax.additive_channel_adapter.{channel}",
                                description=(
                                    "Native controller-visible position offset scaled "
                                    "as a fraction of reach length."
                                ),
                                timing_bin=timing_bin.label,
                                channel_provenance=channel_provenance,
                                semantic_family=semantic_family or "false_feedback_offset",
                                calibration_provenance=provenance(
                                    level=level,
                                    calibration_role="reach_relative_calibrated_native_units",
                                    target_open_loop_peak_dx_m=target_peak,
                                    native_unit_rule=(
                                        "position_offset_m = reference_position_scale_m "
                                        "* level_fraction_of_reach"
                                    ),
                                )
                                | {
                                    "reference_position_scale_m": float(
                                        position_scale["reference_scale"]
                                        if position_scale is not None
                                        else reach.reach_length_m
                                    ),
                                    "controller_feedback_scale": (
                                        None
                                        if position_scale is None
                                        else _feedback_scale_provenance(position_scale)
                                    ),
                                    "feedback_quantity": "position",
                                    "feedback_payload_index": (
                                        _controller_visible_feedback_index("position", axis)
                                    ),
                                    "target_relative_axis_role": axis_role,
                                    "target_relative_basis": "canonical_plus_x_reach",
                                    "false_feedback_probe": True,
                                },
                            )
                        )
                velocity_amplitude = float(
                    velocity_scale["reference_scale"]
                    if velocity_scale is not None
                    else controller_visible_velocity_scale_m_s
                ) * float(level.fraction_of_reach)
                for axis in ("vx", "vy"):
                    for sign in (-1, 1):
                        axis_role = _target_relative_axis_role(axis)
                        perturbations.append(
                            PerturbationSpec(
                                perturbation_id=(
                                    f"{family}__velocity_{level.name}__"
                                    f"{timing_bin.label}_t{start}_{axis}_{_sign_label(sign)}"
                                ),
                                channel=channel,  # type: ignore[arg-type]
                                family=family,
                                amplitude=velocity_amplitude,
                                units="m/s",
                                axis=axis,
                                basis=basis,
                                sign=sign,
                                timing={
                                    "epoch": "controller_visible",
                                    "start_time_index": start,
                                    "duration_steps": duration,
                                    "timing_bin": timing_bin.label,
                                    "timing_bin_role": timing_bin.role,
                                },
                                adapter=f"feedbax.additive_channel_adapter.{channel}",
                                description=(
                                    "Native controller-visible velocity offset scaled "
                                    "as a fraction of nominal peak speed."
                                ),
                                timing_bin=timing_bin.label,
                                semantic_family=semantic_family or "false_feedback_offset",
                                channel_provenance=channel_provenance,
                                calibration_provenance=provenance(
                                    level=level,
                                    calibration_role="reach_relative_calibrated_native_units",
                                    native_unit_rule=(
                                        "velocity_offset_m_s = nominal_peak_speed_m_s "
                                        "* level_fraction_of_reach"
                                    ),
                                )
                                | {
                                    "nominal_peak_speed_m_s": float(
                                        velocity_scale["reference_scale"]
                                        if velocity_scale is not None
                                        else controller_visible_velocity_scale_m_s
                                    ),
                                    "controller_feedback_scale": (
                                        None
                                        if velocity_scale is None
                                        else _feedback_scale_provenance(velocity_scale)
                                    ),
                                    "feedback_quantity": "velocity",
                                    "feedback_payload_index": (
                                        _controller_visible_feedback_index("velocity", axis)
                                    ),
                                    "target_relative_axis_role": axis_role,
                                    "target_relative_basis": "canonical_plus_x_reach",
                                    "false_feedback_probe": True,
                                },
                            )
                        )
                force_filter_amplitude = float(force_filter_scale["reference_scale"]) * float(
                    level.fraction_of_reach
                )
                for axis in ("x", "y"):
                    for sign in (-1, 1):
                        axis_role = _target_relative_axis_role(axis)
                        perturbations.append(
                            PerturbationSpec(
                                perturbation_id=(
                                    f"{family}__force_filter_{level.name}__"
                                    f"{timing_bin.label}_t{start}_{axis}_{_sign_label(sign)}"
                                ),
                                channel=channel,  # type: ignore[arg-type]
                                family=family,
                                amplitude=force_filter_amplitude,
                                units="N",
                                axis=axis,
                                basis=basis,
                                sign=sign,
                                timing={
                                    "epoch": "controller_visible",
                                    "start_time_index": start,
                                    "duration_steps": duration,
                                    "timing_bin": timing_bin.label,
                                    "timing_bin_role": timing_bin.role,
                                },
                                adapter=f"feedbax.additive_channel_adapter.{channel}",
                                description=(
                                    "Native controller-visible force/filter feedback "
                                    "offset in model force units. This row applies only "
                                    "when force_filter_feedback widens the feedback "
                                    "payload to 6D."
                                ),
                                timing_bin=timing_bin.label,
                                semantic_family=semantic_family or "false_feedback_offset",
                                channel_provenance=channel_provenance,
                                calibration_provenance=provenance(
                                    level=level,
                                    calibration_role="reach_relative_calibrated_native_units",
                                    native_unit_rule=(
                                        "force_filter_offset_N = reference_force_filter_scale_N "
                                        "* level_fraction_of_reach"
                                    ),
                                )
                                | {
                                    "reference_force_filter_scale_N": float(
                                        force_filter_scale["reference_scale"]
                                    ),
                                    "controller_feedback_scale": _feedback_scale_provenance(
                                        force_filter_scale,
                                    ),
                                    "feedback_quantity": "force_filter",
                                    "feedback_payload_index": (
                                        _controller_visible_feedback_index(
                                            "force_filter",
                                            axis,
                                        )
                                    ),
                                    "force_filter_feedback_only": True,
                                    "target_relative_axis_role": axis_role,
                                    "target_relative_basis": "canonical_plus_x_reach",
                                    "false_feedback_probe": True,
                                },
                            )
                        )

    perturbations.append(
        PerturbationSpec(
            perturbation_id="target_stream_jump__calibrated_not_applicable",
            channel="target_stream",
            family="target_stream_jump",
            amplitude=0.0,
            units="m",
            axis="x",
            basis="target_cartesian_xy",
            sign=1,
            timing={"epoch": "adapter_defined"},
            adapter="not_applicable_current_fixed_target_checkpoint",
            description="blocked because current C&S GRU input is scalar SISU, not a target stream",
            timing_bin="not_applicable",
            calibration_provenance={
                "calibration_mode": "reach_relative_peak_delta_x",
                "calibration_role": "reach_relative_calibrated_not_applicable",
                "reach_label": reach.label,
                "reach_length_m": float(reach.reach_length_m),
            },
        )
    )

    bank = default_cs_perturbation_bank()
    bank.update(
        {
            "bank_id": CALIBRATED_BANK_ID,
            "calibration_metadata_hooks": {
                "status": "bound_to_reach_relative_defaults",
                "coordinating_issue": "1ad3c16",
                "calibration_mode": "reach_relative_peak_delta_x",
                "reach_label": reach.label,
                "reach_length_m": float(reach.reach_length_m),
                "level_definitions": [level.to_json() for level in levels],
                "controller_feedback_scale_manifest": (
                    None
                    if feedback_scale_manifest_path is None
                    else _repo_relative(feedback_scale_manifest_path, repo_root=REPO_ROOT)
                ),
                "controller_feedback_scales": {
                    "position": (
                        None
                        if position_scale is None
                        else _feedback_scale_provenance(position_scale)
                    ),
                    "velocity": (
                        None
                        if velocity_scale is None
                        else _feedback_scale_provenance(velocity_scale)
                    ),
                    "force_filter": _feedback_scale_provenance(force_filter_scale),
                },
                "source": (
                    "governed open-loop calibration data product "
                    "(rlrmp.perturbation_open_loop_calibration.v2 at "
                    "results/ea6ccb4/data_products/perturbation_open_loop_calibration.json), "
                    "native conventions, and nominal-rollout controller feedback scale manifest"
                ),
            },
            "perturbations": [spec.to_json() for spec in perturbations],
        }
    )
    return bank


def _load_feedback_scale_manifest(
    manifest: Mapping[str, Any] | None,
    manifest_path: Path | None,
) -> Mapping[str, Any] | None:
    if manifest is not None and manifest_path is not None:
        raise ValueError(
            "Pass either feedback_scale_manifest or feedback_scale_manifest_path, not both"
        )
    if manifest is not None:
        return manifest
    if manifest_path is None:
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _controller_feedback_component_scale(
    manifest: Mapping[str, Any] | None,
    component: str,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if manifest is None:
        if required:
            raise ValueError(
                "calibrated force/filter feedback rows require a controller-feedback "
                "scale manifest; pass feedback_scale_manifest_path generated by "
                "gru_evaluation_diagnostics"
            )
        return None

    scale_entries = _controller_feedback_scale_entries(manifest)
    component_entries: list[dict[str, Any]] = []
    for run_id, entry in scale_entries:
        descriptor_payload = entry.get(DESCRIPTOR_PAYLOAD_KEY)
        descriptor_view = resolve_controller_feedback_view(
            descriptor_payload if isinstance(descriptor_payload, Mapping) else None,
            feedback_dim=int(
                entry.get("feedback_dim", 6 if component == COMPONENT_FORCE_FILTER else 4)
            ),
            source="gru_perturbation_bank.feedback_scale_manifest",
        )
        descriptor_id = descriptor_view.component(component).descriptor_id
        components = entry.get("components", {})
        if not isinstance(components, Mapping) or component not in components:
            continue
        payload = components[component]
        if not isinstance(payload, Mapping):
            continue
        reference_scale = payload.get("reference_scale", payload.get("p95_norm"))
        if reference_scale is None:
            continue
        component_entries.append(
            {
                "run_id": run_id,
                "descriptor_id": payload.get("descriptor_id", descriptor_id),
                "reference_scale": float(reference_scale),
                "reference_scale_statistic": payload.get(
                    "reference_scale_statistic",
                    entry.get("statistic", "p95_norm"),
                ),
                "units": payload.get("units"),
                "feedback_basis": entry.get("feedback_basis"),
                "feedback_dim": entry.get("feedback_dim"),
                "feedback_basis_indices": payload.get("feedback_basis_indices"),
                "gru_input_indices": payload.get("gru_input_indices"),
            }
        )

    if not component_entries:
        if required:
            raise ValueError(
                f"controller-feedback scale manifest does not include required "
                f"{component!r} scale data"
            )
        return None

    reference_scale = float(np.mean([entry["reference_scale"] for entry in component_entries]))
    return {
        "component": component,
        "descriptor_id": component_entries[0].get("descriptor_id"),
        "reference_scale": reference_scale,
        "aggregation": "mean_reference_scale_across_manifest_runs",
        "runs": component_entries,
    }


def _controller_feedback_scale_entries(
    manifest: Mapping[str, Any],
) -> list[tuple[str | None, Mapping[str, Any]]]:
    if manifest.get("schema_version") == "rlrmp.controller_feedback_scales.v1":
        return [
            (str(manifest.get("run_id")) if manifest.get("run_id") is not None else None, manifest)
        ]

    runs = manifest.get("runs")
    if isinstance(runs, Mapping):
        entries: list[tuple[str | None, Mapping[str, Any]]] = []
        for run_id, run_payload in runs.items():
            if not isinstance(run_payload, Mapping):
                continue
            scales = run_payload.get("controller_feedback_scales")
            if isinstance(scales, Mapping) and scales.get("status", "available") == "available":
                entries.append((str(run_id), scales))
        return entries

    scales = manifest.get("controller_feedback_scales")
    if isinstance(scales, Mapping):
        return [(str(scales.get("run_id")) if scales.get("run_id") is not None else None, scales)]
    return []


def _feedback_scale_provenance(scale: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "component": scale["component"],
        "descriptor_id": scale.get("descriptor_id"),
        "reference_scale": float(scale["reference_scale"]),
        "aggregation": scale["aggregation"],
        "runs": list(scale["runs"]),
    }


def _validated_perturbation_row(
    perturbation: PerturbationSpec | Mapping[str, Any],
) -> dict[str, Any]:
    spec = (
        perturbation
        if isinstance(perturbation, PerturbationSpec)
        else PerturbationSpec.from_mapping(perturbation)
    )
    spec.validate()
    return spec.to_json()


def apply_perturbation_to_trial_specs(
    trial_specs: Any,
    perturbation: PerturbationSpec | Mapping[str, Any],
    *,
    model: Any | None = None,
    plant_intervenor_label: str = PLANT_INTERVENOR_LABEL,
) -> AdapterResult:
    """Apply one perturbation row to external TaskTrialSpec interfaces."""

    perturbation = _validated_perturbation_row(perturbation)
    channel = str(perturbation["channel"])
    if channel == "initial_state":
        return _apply_initial_state_perturbation(trial_specs, perturbation)
    if channel == "plant_force":
        return _apply_legacy_plant_force_pulse(
            trial_specs,
            perturbation,
            plant_intervenor_label=plant_intervenor_label,
        )
    if channel == "command_input":
        return _apply_command_input_pulse(
            trial_specs,
            perturbation,
            model=model,
            plant_intervenor_label=plant_intervenor_label,
        )
    if channel == "process_epsilon":
        return _apply_process_epsilon_pulse(
            trial_specs,
            perturbation,
            model=model,
        )
    if channel == "sensory_feedback":
        return _apply_named_graph_channel_offset(
            trial_specs,
            perturbation,
            model=model,
            adapter_spec=_graph_adapter_spec(
                perturbation,
                label_prefix="sensory_feedback",
                source_node="sensory",
                source_port="output",
                target_node="net",
                target_port="feedback",
                graphspec_mapping=(
                    "named additive sensory_feedback channel after sensory noise and "
                    "before net.feedback"
                ),
            ),
        )
    if channel == "delayed_observation":
        return _apply_named_graph_channel_offset(
            trial_specs,
            perturbation,
            model=model,
            adapter_spec=_graph_adapter_spec(
                perturbation,
                label_prefix="delayed_observation",
                source_node="feedback",
                source_port="feedback",
                target_node="sensory",
                target_port="input",
                graphspec_mapping=(
                    "named additive pre_noise_delayed_measurement channel before "
                    "sensory.input noise; legacy delayed_observation channel remains "
                    "a compatibility alias"
                ),
            ),
        )
    if channel == "target_stream":
        return AdapterResult(
            status="not_applicable",
            trial_specs=trial_specs,
            model=model,
            reason=(
                "target_stream is deferred: current fixed-target C&S GRU validation "
                "checkpoints do not consume a target-position input stream"
            ),
            adapter_provenance={
                "adapter": "not_applicable_current_fixed_target_checkpoint",
                "graphspec_mapping": "target_stream named graph input when models consume it",
                "controller_input_mutated": False,
            },
        )
    return AdapterResult(
        status="blocked",
        trial_specs=trial_specs,
        model=model,
        reason=f"unknown perturbation channel {channel!r}",
    )


def _initial_position_contract_manifest() -> dict[str, Any]:
    """Return explicit information-contract cases for initial-position offsets."""

    return {
        "status": "declared",
        "default_current_adapter_case": "D_current_state_immediately_visible",
        "cases": {
            "A_target_changed_hand_start_nominal": {
                "status": "not_available",
                "reason": (
                    "Current fixed-target checkpoints do not consume a controller-visible "
                    "target stream, so a target-only initial-position contract cannot be "
                    "applied without changing the model input contract."
                ),
                "future_graphspec_mapping": "task target transform with nominal hand start",
            },
            "B_hand_start_and_delay_history_changed_consistently": {
                "status": "not_available",
                "reason": (
                    "Current eager validation specs do not expose a separate delayed "
                    "history/buffer initializer that can be shifted consistently with "
                    "the hand start."
                ),
                "future_graphspec_mapping": (
                    "task-data initial hand transform plus delayed-observation buffer "
                    "initializer transform"
                ),
            },
            "C_hand_start_changed_history_nominal_until_delay_elapses": {
                "status": "not_available",
                "reason": (
                    "Blind-delay stress rows require explicit delayed-history buffer "
                    "control so the nominal history can be preserved until the delay "
                    "horizon elapses."
                ),
                "future_graphspec_mapping": (
                    "task-data initial hand transform with delayed-observation buffer "
                    "held at nominal values"
                ),
            },
            "D_current_state_immediately_visible": {
                "status": "implemented",
                "current_adapter": "task_trial_spec.inits",
                "row_family": "initial_position_offset",
                "reason": (
                    "The current adapter shifts the exposed initial effector/current "
                    "mechanics state. It is the feasible eager contract for current "
                    "validation-selected checkpoints."
                ),
                "future_graphspec_mapping": (
                    "task-data initial hand transform with immediate current-state visibility"
                ),
            },
        },
    }


def materialize_gru_perturbation_response(
    *,
    source_experiment: str = DEFAULT_SOURCE_EXPERIMENT,
    result_experiment: str = DEFAULT_RESULT_EXPERIMENT,
    run_ids: Sequence[str] = DEFAULT_RUN_IDS,
    labels: Sequence[str] | None = None,
    n_rollout_trials: int = 8,
    evaluate: bool = True,
    write_bulk_arrays: bool = True,
    output_path: Path | None = None,
    note_path: Path | None = None,
    bulk_dir: Path | None = None,
    regeneration_spec_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    bank_mode: Literal["raw", "calibrated"] = "raw",
    calibration_level: str | Sequence[str] | None = None,
    calibration_reach: str | float | None = None,
    feedback_scale_manifest_path: Path | None = None,
    extlqg_physical_dim: Literal[6, 8] = 8,
    preferred_checkpoint_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
) -> dict[str, Any]:
    """Compatibility adapter for the retired monolithic bank materializer.

    The durable file-writing entry point now routes through the P1b spec path:
    a perturbation-response evaluation manifest, per-family leaf payloads, and
    the aggregate legacy bank shape. Legacy path arguments are accepted so
    existing callers fail less abruptly, but this adapter does not write tracked
    JSON, Markdown notes, regeneration specs, or per-row NPZ bulk arrays.
    """

    eval_manifest, eval_manifest_path, eval_states = _execute_perturbation_bank_eval_adapter(
        source_experiment=source_experiment,
        result_experiment=result_experiment,
        run_ids=run_ids,
        labels=labels,
        n_rollout_trials=n_rollout_trials,
        evaluate=evaluate,
        bank_mode=bank_mode,
        calibration_level=calibration_level,
        calibration_reach=calibration_reach,
        feedback_scale_manifest_path=feedback_scale_manifest_path,
        extlqg_physical_dim=extlqg_physical_dim,
        preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=checkpoint_selection_mode,
        repo_root=repo_root,
    )
    manifest = _aggregate_perturbation_eval_adapter(
        eval_states,
        eval_manifest=eval_manifest,
        eval_manifest_path=eval_manifest_path,
        issue=result_experiment,
        source_experiment=source_experiment,
        bank_mode=bank_mode,
    )
    manifest["bank_summary"] = _adapter_bank_summary(manifest.get("bank", {}))
    manifest["compatibility_adapter"] = {
        "materializer": (
            "rlrmp.analysis.pipelines.gru_perturbation_bank."
            "materialize_gru_perturbation_response"
        ),
        "route": "feedbax_evaluation_manifest_to_perturbation_class_leaf_aggregate",
        "evaluation_manifest_id": eval_manifest.id,
        "evaluation_manifest_path": str(eval_manifest_path),
        "legacy_output_paths_ignored": {
            "output_path": None if output_path is None else str(output_path),
            "note_path": None if note_path is None else str(note_path),
            "bulk_dir": None if bulk_dir is None else str(bulk_dir),
            "regeneration_spec_path": (
                None if regeneration_spec_path is None else str(regeneration_spec_path)
            ),
        },
        "write_bulk_arrays_requested": bool(write_bulk_arrays),
        "write_bulk_arrays_effective": False,
    }
    return manifest


def _execute_perturbation_bank_eval_adapter(
    *,
    source_experiment: str,
    result_experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    n_rollout_trials: int,
    evaluate: bool,
    bank_mode: Literal["raw", "calibrated"],
    calibration_level: str | Sequence[str] | None,
    calibration_reach: str | float | None,
    feedback_scale_manifest_path: Path | None,
    extlqg_physical_dim: Literal[6, 8],
    preferred_checkpoint_manifest_path: Path | None,
    checkpoint_selection_mode: CheckpointSelectionMode,
    repo_root: Path,
) -> tuple[Any, Path, Mapping[str, Any]]:
    from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef

    from rlrmp import ensure_rlrmp_recipes_registered
    from rlrmp.eval.recipes import (
        PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
        perturbation_response_bank_recipe,
    )
    from rlrmp.runtime.spec_migrations import (
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
        stamp_current_schema,
    )

    ensure_rlrmp_recipes_registered()
    params: dict[str, Any] = {
        "source_experiment": source_experiment,
        "run_ids": list(run_ids),
        "labels": None if labels is None else list(labels),
        "n_rollout_trials": int(n_rollout_trials),
        "bank_mode": bank_mode,
        "calibration_level": calibration_level,
        "calibration_reach": calibration_reach,
        "feedback_scale_manifest_path": (
            None if feedback_scale_manifest_path is None else str(feedback_scale_manifest_path)
        ),
        "extlqg_physical_dim": int(extlqg_physical_dim),
        "preferred_checkpoint_manifest_path": (
            None
            if preferred_checkpoint_manifest_path is None
            else str(preferred_checkpoint_manifest_path)
        ),
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "repo_root": str(repo_root),
        "write_bulk_arrays": False,
    }
    params = {key: value for key, value in params.items() if value is not None}
    if not evaluate:
        params.update(
            {
                "checkpoint_bank_ref": {
                    "kind": "CheckpointSelectionManifest",
                    "id": f"{source_experiment}_legacy_adapter_no_evaluate",
                },
                "perturbation_battery": default_cs_perturbation_bank(
                    mode=bank_mode,
                    calibration_level=calibration_level,
                    calibration_reach=calibration_reach,
                    feedback_scale_manifest_path=feedback_scale_manifest_path,
                ),
                "response_tensors": {"runs": {}},
                "legacy_payload_mode": True,
            }
        )
    spec = EvaluationRunSpec(
        evaluation_type=PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
        training_run_ids=list(run_ids),
        inputs=[
            ParentRef(kind="TrainingRunManifest", id=str(run_id), role="training_run")
            for run_id in run_ids
        ],
        params=stamp_current_schema(PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND, params),
    )
    result = perturbation_response_bank_recipe(
        spec,
        repo_root,
        repo_root / "cache" / "states" / "legacy_perturbation_adapter.pkl",
    )
    states = result.states
    if not isinstance(states, Mapping):
        raise TypeError("perturbation-response evaluation states must be a mapping")
    manifest = SimpleNamespace(
        id=str(result.metadata.get("caching_identity", {}).get("manifest_id")),
        metadata=result.metadata,
        summary_metrics=result.summary_metrics,
    )
    manifest_path = repo_root / "manifests" / "evaluation_runs" / f"{manifest.id}.json"
    return manifest, manifest_path, states


def _aggregate_perturbation_eval_adapter(
    states: Mapping[str, Any],
    *,
    eval_manifest: Any,
    eval_manifest_path: Path,
    issue: str,
    source_experiment: str,
    bank_mode: Literal["raw", "calibrated"],
) -> dict[str, Any]:
    from feedbax.contracts.manifest import ParentRef

    from rlrmp.analysis import declarative_materialization as dm

    class_index_map = states.get("class_index_map")
    if not isinstance(class_index_map, Mapping):
        raise ValueError("perturbation-response evaluation states lack class_index_map")
    resolved_eval = SimpleNamespace(
        ref=ParentRef(
            kind="EvaluationRunManifest",
            id=eval_manifest.id,
            role="evaluation_run",
            uri=str(eval_manifest_path),
        ),
        manifest=eval_manifest,
        path=eval_manifest_path,
        states=states,
    )
    leaf_products = [
        dm._perturbation_class_response_payload(
            states,
            {"family": str(family)},
            evaluation_input=resolved_eval,
        )
        for family in class_index_map
    ]
    return dm._aggregate_perturbation_class_products(
        leaf_products,
        {
            "issue": issue,
            "source_experiment": source_experiment,
            "bank_mode": bank_mode,
        },
    )


def _adapter_bank_summary(bank: Any) -> dict[str, Any]:
    if not isinstance(bank, Mapping):
        return {"n_perturbations": 0, "families": [], "channels": [], "timing_bins": []}
    rows = bank.get("perturbations", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    return {
        "bank_id": bank.get("bank_id"),
        "schema_version": bank.get("schema_version"),
        "n_perturbations": len(rows),
        "families": sorted(
            {
                str(row.get("family"))
                for row in rows
                if isinstance(row, Mapping) and row.get("family") is not None
            }
        ),
        "channels": sorted(
            {
                str(row.get("channel"))
                for row in rows
                if isinstance(row, Mapping) and row.get("channel") is not None
            }
        ),
        "timing_bins": sorted(
            {
                str(row.get("timing_bin"))
                for row in rows
                if isinstance(row, Mapping) and row.get("timing_bin") is not None
            }
        ),
    }


def _regeneration_spec_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_regeneration_spec.json")


def _slim_perturbation_response_manifest(
    manifest: Mapping[str, Any],
    *,
    detail_manifest_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    """Remove per-row response payloads from the tracked response manifest."""

    slim = dict(manifest)
    bank = slim.pop("bank", None)
    if detail_manifest_path is not None:
        if isinstance(bank, Mapping):
            slim["bank_summary"] = _slim_perturbation_bank_summary(
                bank,
                detail_manifest_path=detail_manifest_path,
                repo_root=repo_root,
            )
        slim["bulk_detail_manifest"] = {
            "path": _repo_relative(detail_manifest_path, repo_root=repo_root),
            "format": "json",
            "contains": "full per-run perturbation rows and row-level metric summaries",
        }
    slim_runs: dict[str, Any] = {}
    for run_id, run_payload in dict(manifest.get("runs", {})).items():
        run = dict(run_payload)
        perturbations = run.pop("perturbations", [])
        robust_summary = run.pop("robust_response_summary", None)
        bulk_files = run.pop("bulk_files", None)
        run["n_perturbation_rows"] = (
            len(perturbations) if isinstance(perturbations, Sequence) else 0
        )
        if isinstance(bulk_files, Mapping):
            run["bulk_files_count"] = len(bulk_files)
        if detail_manifest_path is not None:
            run["perturbation_rows_detail_manifest"] = _repo_relative(
                detail_manifest_path,
                repo_root=repo_root,
            )
            if robust_summary is not None:
                run["robust_response_summary_detail_manifest"] = _repo_relative(
                    detail_manifest_path,
                    repo_root=repo_root,
                )
            if isinstance(bulk_files, Mapping):
                run["bulk_files_detail_manifest"] = _repo_relative(
                    detail_manifest_path,
                    repo_root=repo_root,
                )
        if isinstance(robust_summary, Mapping):
            run["robust_response_summary_status"] = robust_summary.get("status", "available")
        slim_runs[str(run_id)] = run
    slim["runs"] = slim_runs
    return slim


def _slim_perturbation_bank_summary(
    bank: Mapping[str, Any],
    *,
    detail_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return tracked bank metadata without the full perturbation row list."""

    perturbations = bank.get("perturbations")
    if not isinstance(perturbations, Sequence):
        perturbations = []
    families = sorted(
        {
            str(row.get("family"))
            for row in perturbations
            if isinstance(row, Mapping) and row.get("family") is not None
        }
    )
    channels = sorted(
        {
            str(row.get("channel"))
            for row in perturbations
            if isinstance(row, Mapping) and row.get("channel") is not None
        }
    )
    timing_bins = sorted(
        {
            str(row.get("timing_bin"))
            for row in perturbations
            if isinstance(row, Mapping) and row.get("timing_bin") is not None
        }
    )
    return {
        "bank_id": bank.get("bank_id"),
        "schema_version": bank.get("schema_version"),
        "n_perturbations": len(perturbations),
        "families": families,
        "channels": channels,
        "timing_bins": timing_bins,
        "detail_manifest": _repo_relative(detail_manifest_path, repo_root=repo_root),
        "detail_contains": "full perturbation bank including per-row perturbation definitions",
    }


def _effective_checkpoint_policy_from_manifest(
    experiment: str,
    *,
    preferred_checkpoint_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
    repo_root: Path = REPO_ROOT,
) -> str:
    """Return the checkpoint policy represented by an optional preferred manifest."""

    effective_selection_mode = checkpoint_selection_mode
    if (
        effective_selection_mode == "sparse_history"
        and preferred_checkpoint_manifest_path is not None
    ):
        effective_selection_mode = "fixed_bank_manifest"
    if effective_selection_mode == "sparse_history":
        return "validation_selected_per_replicate"
    if effective_selection_mode != "fixed_bank_manifest":
        raise ValueError(f"unsupported checkpoint selection mode {checkpoint_selection_mode!r}")
    manifest = load_materialized_fixed_bank_manifest(
        experiment=experiment,
        manifest_path=preferred_checkpoint_manifest_path,
        repo_root=repo_root,
    )
    if manifest is not None:
        return str(manifest.get("checkpoint_policy") or "fixed_bank_rescored_per_replicate")
    raise ValueError(
        "checkpoint_selection_mode='fixed_bank_manifest' requires a materialized "
        "fixed-bank checkpoint manifest"
    )


def evaluate_run_perturbation_bank(
    run: RunFigureInputs,
    *,
    source_experiment: str,
    bank: Mapping[str, Any],
    n_rollout_trials: int,
    write_bulk_arrays: bool,
    bulk_dir: Path,
    evaluation_backend: PerturbationEvaluationBackend = "serial",
    trial_spec_transform: Callable[[Any], Any] | None = None,
    extlqg_physical_dim: Literal[6, 8] = 8,
    preferred_checkpoint_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate one validation-selected GRU run on a perturbation bank."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = require_run_seed(run.run_spec, source=run.run_spec_path)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=source_experiment,
        run_id=run.run_id,
        run_spec=run.run_spec,
        preferred_manifest_path=preferred_checkpoint_manifest_path,
        checkpoint_selection_mode=checkpoint_selection_mode,
        repo_root=repo_root,
    )
    base_trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
    if trial_spec_transform is not None:
        base_trial_specs = trial_spec_transform(base_trial_specs)
    nominal_base_evaluation = _evaluate_model_rollout_product(
        model=model,
        task=pair.task,
        trial_specs=base_trial_specs,
        n_replicates=n_replicates,
        seed=0,
    )
    nominal_base_cost = full_qrf_cost_summary(nominal_base_evaluation, base_trial_specs)
    extlqg_context = _build_extlqg_comparator_context(physical_dim=extlqg_physical_dim)
    robust_context = _build_robust_output_feedback_comparator_context()
    rows = []
    bulk_files: dict[str, str] = {}
    if evaluation_backend != "serial":
        raise ValueError(
            f"unsupported perturbation evaluation backend {evaluation_backend!r}; expected 'serial'"
        )

    for perturbation in bank["perturbations"]:
        adapter = apply_perturbation_to_trial_specs(
            base_trial_specs,
            perturbation,
            model=model,
        )
        if adapter.status != "evaluated":
            rows.append(
                {
                    "perturbation_id": perturbation["perturbation_id"],
                    "channel": perturbation["channel"],
                    "family": perturbation.get("family"),
                    "axis": perturbation.get("axis"),
                    "sign": perturbation.get("sign"),
                    "amplitude": perturbation.get("amplitude"),
                    "timing_bin": perturbation.get("timing_bin"),
                    "semantic_family": perturbation.get("semantic_family"),
                    "timing": perturbation.get("timing"),
                    "perturbation": dict(perturbation),
                    "status": adapter.status,
                    "reason": adapter.reason,
                    "adapter": adapter.to_json(),
                    "extlqg_comparator": extlqg_comparator_status(
                        perturbation,
                        status="not_applicable",
                    ),
                    "robust_output_feedback_comparator": (
                        robust_output_feedback_comparator_status(
                            perturbation,
                            status="not_applicable",
                        )
                    ),
                }
            )
            continue

        row_base_model, row_base_trial_specs = _paired_base_for_adapter(
            model=model,
            base_trial_specs=base_trial_specs,
            adapter=adapter,
        )
        if row_base_model is model and row_base_trial_specs is base_trial_specs:
            base_evaluation = nominal_base_evaluation
            base_cost = nominal_base_cost
        else:
            base_evaluation = _evaluate_model_rollout_product(
                model=row_base_model,
                task=pair.task,
                trial_specs=row_base_trial_specs,
                n_replicates=n_replicates,
                seed=0,
            )
            base_cost = full_qrf_cost_summary(base_evaluation, row_base_trial_specs)
        perturbed_evaluation = _evaluate_model_rollout_product(
            model=adapter.model if adapter.model is not None else model,
            task=pair.task,
            trial_specs=adapter.trial_specs,
            n_replicates=n_replicates,
            seed=0,
        )
        perturbed_cost = full_qrf_cost_summary(perturbed_evaluation, adapter.trial_specs)
        rows.append(
            _evaluated_perturbation_row(
                perturbation=perturbation,
                adapter=adapter,
                base_evaluation=base_evaluation,
                perturbed_evaluation=perturbed_evaluation,
                base_cost=base_cost,
                perturbed_cost=perturbed_cost,
                extlqg_context=extlqg_context,
                robust_context=robust_context,
                write_bulk_arrays=write_bulk_arrays,
                bulk_dir=bulk_dir / run.run_id,
                repo_root=repo_root,
                bulk_files=bulk_files,
            )
        )

    return {
        "label": run.label,
        "run_spec_path": _repo_relative(run.run_spec_path, repo_root=repo_root),
        "artifact_dir": _repo_relative(run.artifact_dir, repo_root=repo_root),
        "checkpoint_selection": [
            selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
        ],
        "n_replicates": int(base_evaluation.command.shape[0]),
        "n_rollout_trials_per_replicate": int(base_evaluation.command.shape[1]),
        "n_time_steps": int(base_evaluation.command.shape[2]),
        "dt_s": float(base_evaluation.dt),
        "status_counts": _status_counts(rows),
        "robust_response_summary": summarize_perturbation_bank(rows),
        "perturbations": rows,
        "bulk_files": bulk_files,
    }


def _paired_base_for_adapter(
    *,
    model: Any,
    base_trial_specs: Any,
    adapter: AdapterResult,
) -> tuple[Any, Any]:
    """Use the same graph topology and zero payload for graph-adapter base rows.

    Temporary graph adapters add an extra component node. Evaluating the base
    trajectory on the unmodified graph can change stochastic key paths relative
    to the perturbed trajectory before a timed payload begins. Therefore the
    paired base path uses the adapter-modified graph with an all-zero copy of
    the same external input payload.
    """

    provenance = adapter.adapter_provenance or {}
    if provenance.get("requires_zero_payload_base") is True and adapter.model is not None:
        input_key = provenance.get("input_key")
        if isinstance(input_key, str) and input_key in getattr(adapter.trial_specs, "inputs", {}):
            if input_key in getattr(base_trial_specs, "inputs", {}):
                return adapter.model, base_trial_specs
            payload = jnp.zeros_like(jnp.asarray(adapter.trial_specs.inputs[input_key]))
            return adapter.model, _add_trial_input(base_trial_specs, input_key, payload)
        return adapter.model, base_trial_specs
    return model, base_trial_specs


def _evaluated_perturbation_row(
    *,
    perturbation: Mapping[str, Any],
    adapter: AdapterResult,
    base_evaluation: RolloutEvaluation,
    perturbed_evaluation: RolloutEvaluation,
    base_cost: Mapping[str, Any],
    perturbed_cost: Mapping[str, Any],
    extlqg_context: Mapping[str, Any],
    robust_context: Mapping[str, Any],
    write_bulk_arrays: bool,
    bulk_dir: Path,
    repo_root: Path,
    bulk_files: dict[str, str],
) -> dict[str, Any]:
    """Build the manifest row for one evaluated perturbation."""

    metrics = summarize_perturbation_response(
        base_evaluation,
        perturbed_evaluation,
        base_full_qrf_cost=base_cost,
        perturbed_full_qrf_cost=perturbed_cost,
    )
    metrics = _with_attenuation_metrics(metrics, perturbation)
    no_op_guard = _command_input_no_op_guard(
        perturbation=perturbation,
        adapter=adapter,
        metrics=metrics,
    )
    if no_op_guard is not None and no_op_guard["status"] == "blocked":
        return {
            "perturbation_id": perturbation["perturbation_id"],
            "channel": perturbation["channel"],
            "family": perturbation.get("family"),
            "axis": perturbation.get("axis"),
            "sign": perturbation.get("sign"),
            "amplitude": perturbation.get("amplitude"),
            "timing_bin": perturbation.get("timing_bin"),
            "semantic_family": perturbation.get("semantic_family"),
            "timing": perturbation.get("timing"),
            "perturbation": dict(perturbation),
            "status": "blocked",
            "reason": no_op_guard["reason"],
            "adapter": adapter.to_json(),
            "metrics": metrics,
            "evaluation_guard": no_op_guard,
            "extlqg_comparator": extlqg_comparator_status(
                perturbation,
                status="not_applicable",
            ),
            "robust_output_feedback_comparator": (
                robust_output_feedback_comparator_status(
                    perturbation,
                    status="not_applicable",
                )
            ),
        }
    extlqg_comparator = evaluate_extlqg_perturbation_comparator(
        perturbation,
        context=extlqg_context,
        gru_metrics=metrics,
    )
    robust_comparator = evaluate_robust_output_feedback_perturbation_comparator(
        perturbation,
        context=robust_context,
        gru_metrics=metrics,
    )
    bulk_file = None
    if write_bulk_arrays:
        bulk_file = _write_perturbation_bulk_arrays(
            base_evaluation,
            perturbed_evaluation,
            bulk_dir=bulk_dir,
            perturbation_id=str(perturbation["perturbation_id"]),
        )
        bulk_files[str(perturbation["perturbation_id"])] = _repo_relative(
            bulk_file,
            repo_root=repo_root,
        )
    return {
        "perturbation_id": perturbation["perturbation_id"],
        "channel": perturbation["channel"],
        "family": perturbation.get("family"),
        "axis": perturbation.get("axis"),
        "sign": perturbation.get("sign"),
        "amplitude": perturbation.get("amplitude"),
        "timing_bin": perturbation.get("timing_bin"),
        "semantic_family": perturbation.get("semantic_family"),
        "timing": perturbation.get("timing"),
        "perturbation": dict(perturbation),
        "status": "evaluated",
        "adapter": adapter.to_json(),
        "metrics": metrics,
        "evaluation_guard": no_op_guard
        or {"status": "passed", "guard": "command_input_nonzero_payload_nonzero_effect"},
        "extlqg_comparator": extlqg_comparator,
        "robust_output_feedback_comparator": robust_comparator,
        "bulk_arrays": None
        if bulk_file is None
        else {
            "path": _repo_relative(bulk_file, repo_root=repo_root),
            "format": "np.savez_compressed",
            "arrays": [
                "delta_action",
                "delta_gru_input",
                "delta_position",
                "delta_velocity",
                "base_position",
                "perturbed_position",
                "base_velocity",
                "perturbed_velocity",
                "base_action",
                "perturbed_action",
                "base_gru_input",
                "perturbed_gru_input",
            ],
        },
    }


def summarize_perturbation_response(
    base: RolloutEvaluation,
    perturbed: RolloutEvaluation,
    *,
    base_full_qrf_cost: Mapping[str, Any] | None = None,
    perturbed_full_qrf_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute paired perturbation-response metrics."""

    base_command = jnp.asarray(base.command, dtype=jnp.float64)
    base_input = jnp.asarray(base.gru_input, dtype=jnp.float64)
    base_position = jnp.asarray(base.position, dtype=jnp.float64)
    base_velocity = jnp.asarray(base.velocity, dtype=jnp.float64)
    base_target = jnp.asarray(base.target_position, dtype=jnp.float64)
    perturbed_command = jnp.asarray(perturbed.command, dtype=jnp.float64)
    perturbed_input = jnp.asarray(perturbed.gru_input, dtype=jnp.float64)
    perturbed_position = jnp.asarray(perturbed.position, dtype=jnp.float64)
    perturbed_velocity = jnp.asarray(perturbed.velocity, dtype=jnp.float64)
    perturbed_target = jnp.asarray(perturbed.target_position, dtype=jnp.float64)

    delta_action = perturbed_command - base_command
    delta_input = perturbed_input - base_input
    delta_position = perturbed_position - base_position
    delta_velocity = perturbed_velocity - base_velocity
    delta_position_norm = jnp.linalg.norm(delta_position, axis=-1)
    delta_velocity_norm = jnp.linalg.norm(delta_velocity, axis=-1)
    delta_state_norm = jnp.linalg.norm(
        jnp.concatenate([delta_position, delta_velocity], axis=-1),
        axis=-1,
    )
    delta_action_norm = jnp.linalg.norm(delta_action, axis=-1)
    endpoint_recovery = jnp.linalg.norm(
        perturbed_position[:, :, -1, :] - perturbed_target[None, :, -1, :],
        axis=-1,
    )
    base_endpoint = jnp.linalg.norm(
        base_position[:, :, -1, :] - base_target[None, :, -1, :],
        axis=-1,
    )
    terminal_speed = jnp.linalg.norm(perturbed_velocity[:, :, -1, :], axis=-1)
    base_terminal_speed = jnp.linalg.norm(base_velocity[:, :, -1, :], axis=-1)
    metrics = {
        "delta_action_norm": _summary_stats(delta_action_norm),
        "delta_position_trajectory_norm_m": _summary_stats(delta_position_norm),
        "delta_velocity_trajectory_norm_m_s": _summary_stats(delta_velocity_norm),
        "delta_state_trajectory_norm": _summary_stats(delta_state_norm),
        "delta_position_response_m": _response_magnitude_summary(
            delta_position_norm,
            dt=float(base.dt),
            value_label="delta_position_norm_m",
        ),
        "delta_state_response": _response_magnitude_summary(
            delta_state_norm,
            dt=float(base.dt),
            value_label="delta_state_norm",
        ),
        "delta_action_response": _response_magnitude_summary(
            delta_action_norm,
            dt=float(base.dt),
            value_label="delta_action_norm",
        ),
        "response_shape": _response_shape_summary(
            delta_position_norm,
            dt=float(base.dt),
            value_label="delta_position_norm_m",
        ),
        "target_relative_alignment": _target_relative_alignment_summary(
            base=base,
            delta_position=delta_position,
            delta_action=delta_action,
        ),
        "endpoint_error_m": _summary_stats(endpoint_recovery),
        "delta_endpoint_error_m": _summary_stats(endpoint_recovery - base_endpoint),
        "terminal_speed_m_s": _summary_stats(terminal_speed),
        "delta_terminal_speed_m_s": _summary_stats(terminal_speed - base_terminal_speed),
        "controller_io_response": _controller_io_response_summary(
            delta_input=delta_input,
            delta_action=delta_action,
        ),
    }
    if base_full_qrf_cost is None or perturbed_full_qrf_cost is None:
        metrics["extra_full_qrf_cost"] = {
            "status": "not_available",
            "reason": "full-Q/R/Q_f cost summaries were not provided for this row",
        }
    else:
        metrics["extra_full_qrf_cost"] = delta_full_qrf_cost_summary(
            base_full_qrf_cost,
            perturbed_full_qrf_cost,
        )
    return metrics


def _command_input_no_op_guard(
    *,
    perturbation: Mapping[str, Any],
    adapter: AdapterResult,
    metrics: Mapping[str, Any],
    tolerance: float = 1e-12,
) -> dict[str, Any] | None:
    """Block command-input rows whose nonzero payload had no measured effect."""

    if perturbation.get("channel") != "command_input":
        return None
    provenance = dict(adapter.adapter_provenance or {})
    input_key = provenance.get("input_key")
    if input_key is None or input_key not in getattr(adapter.trial_specs, "inputs", {}):
        return {
            "status": "blocked",
            "guard": "command_input_payload_missing",
            "reason": (
                "command_input perturbation adapter was evaluated but did not expose "
                "the declared external graph input payload"
            ),
            "input_key": input_key,
            "adapter_provenance": provenance,
        }
    payload = np.asarray(adapter.trial_specs.inputs[input_key], dtype=np.float64)
    payload_abs_max = float(np.max(np.abs(payload))) if payload.size else 0.0
    duration = int(_row_timing(perturbation).get("duration_steps", 1) or 1)
    if payload_abs_max <= tolerance or duration <= 0:
        return {"status": "not_applicable", "reason": "zero_command_input_payload"}

    state_max = _metric_summary_value(metrics, "delta_state_response.max", "max")
    position_max = _metric_summary_value(metrics, "delta_position_response_m.max", "max")
    action_max = _metric_summary_value(metrics, "delta_action_response.max", "max")
    input_max = _metric_summary_value(metrics, "controller_io_response.delta_input_norm", "max")
    cost_abs_max = _cost_delta_abs_max(metrics)
    observed = {
        "state_response_max": state_max,
        "position_response_max_m": position_max,
        "action_response_max": action_max,
        "input_response_max": input_max,
        "full_qrf_delta_abs_max": cost_abs_max,
    }
    finite_values = [
        value for value in observed.values() if value is not None and np.isfinite(value)
    ]
    if finite_values and max(abs(value) for value in finite_values) > tolerance:
        return {
            "status": "passed",
            "guard": "command_input_nonzero_payload_nonzero_effect",
            "tolerance": tolerance,
            "input_key": input_key,
            "payload_abs_max": payload_abs_max,
            "observed": observed,
        }
    if bool(perturbation.get("allow_zero_graph_effect", False)):
        return {
            "status": "allowed",
            "guard": "command_input_nonzero_payload_zero_effect_allowed",
            "reason": (
                "nonzero command_input payload produced all-zero paired response, "
                "but the row explicitly set allow_zero_graph_effect"
            ),
            "tolerance": tolerance,
            "input_key": input_key,
            "payload_abs_max": payload_abs_max,
            "duration_steps": duration,
            "adapter_provenance": provenance,
            "observed": observed,
        }
    return {
        "status": "blocked",
        "guard": "command_input_nonzero_payload_zero_effect",
        "reason": (
            "nonzero command_input payload produced all-zero paired response; "
            "this indicates an adapter/materialization failure rather than a "
            "valid controller response"
        ),
        "tolerance": tolerance,
        "input_key": input_key,
        "payload_abs_max": payload_abs_max,
        "duration_steps": duration,
        "adapter_provenance": provenance,
        "observed": observed,
    }


def _metric_summary_value(
    metrics: Mapping[str, Any],
    dotted_key: str,
    summary_key: str,
) -> float | None:
    current: Any = metrics
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, Mapping) or summary_key not in current:
        return None
    value = current[summary_key]
    if value is None:
        return None
    return float(value)


def _cost_delta_abs_max(metrics: Mapping[str, Any]) -> float | None:
    current: Any = metrics
    for key in ("extra_full_qrf_cost", "delta_cost", "total"):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    values = [
        float(current[key])
        for key in ("min", "max", "mean")
        if key in current and current[key] is not None
    ]
    if not values:
        return None
    return max(abs(value) for value in values)


def score_full_qrf_rollout_cost(
    *,
    states: Any,
    commands: Any,
    initial_states: Any,
    target_pos: Any = TARGET_POS,
) -> dict[str, Any]:
    """Score realized full analytical Q/R/Q_f costs with per-term arrays.

    Args:
        states: Rollout mechanics vectors with shape ``(..., T, 48)`` for the
            canonical 8D delayed state, or ``(..., T, 36)`` for the 6D
            no-integrator delayed state.
        commands: Controller commands with shape ``(..., T, 2)``.
        initial_states: Initial mechanics vectors broadcastable to the rollout
            state dimension.
        target_pos: Absolute target position subtracted from every physical
            delay block's x/y position coordinates before applying Q/Q_f.

    Returns:
        JSON-compatible arrays for total, stage-state, control, and terminal
        costs. Arrays retain all leading dimensions so replicate variability is
        not hidden.
    """

    state_array = jnp.asarray(states, dtype=jnp.float64)
    command_array = jnp.asarray(commands, dtype=jnp.float64)
    initial_array = jnp.asarray(initial_states, dtype=jnp.float64)
    _plant, schedule, physical_dim, schedule_source = _full_qrf_game_for_state_dim(
        int(state_array.shape[-1])
    )
    if state_array.shape[-1] != schedule.Q.shape[-1]:
        raise ValueError(
            f"Full-Q/R/Q_f scorer expected state dim {schedule.Q.shape[-1]}, "
            f"got {state_array.shape[-1]}."
        )
    if command_array.shape[-1] != schedule.R.shape[-1]:
        raise ValueError(
            f"Full-Q/R/Q_f scorer expected command dim {schedule.R.shape[-1]}, "
            f"got {command_array.shape[-1]}."
        )
    horizon = int(schedule.T)
    if state_array.shape[-2] != horizon:
        raise ValueError(
            f"Full-Q/R/Q_f scorer expected {horizon} rollout states, got {state_array.shape[-2]}."
        )
    if command_array.shape[-2] != horizon:
        raise ValueError(
            f"Full-Q/R/Q_f scorer expected {horizon} commands, got {command_array.shape[-2]}."
        )
    initial_array = jnp.broadcast_to(
        initial_array,
        (*state_array.shape[:-2], state_array.shape[-1]),
    )
    x_pre = jnp.concatenate([initial_array[..., None, :], state_array[..., :-1, :]], axis=-2)
    x_pre = _goal_centered_vectors(x_pre, target_pos=target_pos, physical_dim=physical_dim)
    x_terminal = _goal_centered_vectors(
        state_array[..., -1, :],
        target_pos=target_pos,
        physical_dim=physical_dim,
    )
    q = jnp.asarray(schedule.Q, dtype=jnp.float64)
    r = jnp.asarray(schedule.R, dtype=jnp.float64)
    q_f = jnp.asarray(schedule.Q_f, dtype=jnp.float64)
    state_terms = jnp.einsum("...ti,tij,...tj->...t", x_pre, q, x_pre)
    control_terms = jnp.einsum("...ti,tij,...tj->...t", command_array, r, command_array)
    terminal_terms = jnp.einsum("...i,ij,...j->...", x_terminal, q_f, x_terminal)
    stage_state = jnp.sum(state_terms, axis=-1)
    control = jnp.sum(control_terms, axis=-1)
    total = stage_state + control + terminal_terms
    return {
        "status": "available",
        "lens": "realized_deterministic_rollout_full_qrf",
        "basis": {
            "state_key": "states.mechanics.vector",
            "command_key": "states.net.output",
            "state_transform": "subtract TARGET_POS from each physical delay block x/y",
            "physical_state_dim": physical_dim,
            "schedule_source": schedule_source,
        },
        "total": total,
        "stage_state": stage_state,
        "control": control,
        "terminal": terminal_terms,
        "timewise_stage_state": state_terms,
        "timewise_control": control_terms,
    }


def full_qrf_cost_summary(
    evaluation: RolloutEvaluation,
    trial_specs: Any,
) -> dict[str, Any]:
    """Return JSON-compatible full-Q/R/Q_f cost summary for one GRU rollout."""

    mechanics_vector = getattr(evaluation, "mechanics_vector", None)
    if mechanics_vector is None:
        return {
            "status": "not_available",
            "reason": "RolloutEvaluation does not carry states.mechanics.vector.",
        }
    if "mechanics.vector" not in trial_specs.inits:
        return {
            "status": "not_available",
            "reason": "trial_specs.inits lacks mechanics.vector initial state.",
        }
    mechanics_vector, commands, initial_states, window_metadata = _full_qrf_window_inputs(
        mechanics_vector,
        evaluation.command,
        trial_specs,
    )
    scored = score_full_qrf_rollout_cost(
        states=mechanics_vector,
        commands=commands,
        initial_states=initial_states,
    )
    scored["basis"]["time_window"] = window_metadata
    return _cost_arrays_to_summary(scored)


def _full_qrf_window_inputs(
    states: Any,
    commands: Any,
    trial_specs: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Return C&S full-QRF movement-window arrays for immediate or delayed trials."""

    _plant, schedule = build_canonical_game()
    horizon = int(schedule.T)
    state_array = jnp.asarray(states, dtype=jnp.float64)
    command_array = jnp.asarray(commands, dtype=jnp.float64)
    initial_array = jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=jnp.float64)
    if state_array.shape[-2] == horizon and command_array.shape[-2] == horizon:
        return (
            state_array,
            command_array,
            initial_array,
            {"basis": "full_rollout_matches_canonical_horizon", "start": 0, "stop": horizon},
        )
    if state_array.shape[-2] < horizon or command_array.shape[-2] < horizon:
        return (
            state_array,
            command_array,
            initial_array,
            {
                "basis": "shorter_than_canonical_horizon",
                "state_steps": int(state_array.shape[-2]),
                "command_steps": int(command_array.shape[-2]),
                "canonical_horizon": horizon,
            },
        )

    start = _constant_movement_start(trial_specs)
    if start is None or start + horizon > state_array.shape[-2]:
        start = int(state_array.shape[-2] - horizon)
        basis = "trailing_canonical_movement_window"
    else:
        basis = "timeline_epoch_bounds_movement_window"
    stop = int(start + horizon)
    window_initial = initial_array if start == 0 else state_array[..., start - 1, :]
    return (
        state_array[..., start:stop, :],
        command_array[..., start:stop, :],
        window_initial,
        {"basis": basis, "start": int(start), "stop": stop, "canonical_horizon": horizon},
    )


def _constant_movement_start(trial_specs: Any) -> int | None:
    """Return a unique delayed movement start from timeline metadata, if present."""

    timeline = getattr(trial_specs, "timeline", None)
    epoch_bounds = getattr(timeline, "epoch_bounds", None)
    if epoch_bounds is None:
        return None
    bounds = np.asarray(epoch_bounds)
    if bounds.ndim < 2 or bounds.shape[-1] < 2:
        return None
    starts = np.unique(bounds[..., _movement_start_bound_column(bounds)])
    if starts.size != 1:
        return None
    return int(starts[0])


def _movement_start_indices(trial_specs: Any, *, batch_size: int) -> np.ndarray | None:
    """Return per-trial movement starts from timeline metadata, if available."""

    timeline = getattr(trial_specs, "timeline", None)
    epoch_bounds = getattr(timeline, "epoch_bounds", None)
    if epoch_bounds is None:
        return None
    bounds = np.asarray(epoch_bounds)
    if bounds.ndim < 2 or bounds.shape[-1] < 2:
        return None
    starts = np.asarray(
        bounds[..., _movement_start_bound_column(bounds)],
        dtype=np.int64,
    ).reshape(-1)
    if starts.size == 1:
        return np.full((batch_size,), int(starts[0]), dtype=np.int64)
    if starts.size != batch_size:
        raise ValueError(
            "movement-indexed perturbation timing requires scalar or per-trial "
            f"movement starts; got {starts.size} starts for batch size {batch_size}"
        )
    return starts


def _movement_start_bound_column(bounds: np.ndarray) -> int:
    """Return the epoch-bound column that denotes movement start.

    A one-epoch movement-only timeline is encoded as ``[0, T]``; its movement
    start is column 0, not the terminal bound. Older delayed/full-trial specs
    keep their movement start in column 1.
    """

    return 0 if bounds.shape[-1] == 2 else 1


def _movement_start_source(trial_specs: Any) -> str:
    timeline = getattr(trial_specs, "timeline", None)
    epoch_bounds = getattr(timeline, "epoch_bounds", None)
    if epoch_bounds is None:
        return "absent_timeline_assumed_zero_for_immediate_reach"
    bounds = np.asarray(epoch_bounds)
    column = _movement_start_bound_column(bounds)
    return f"trial_specs.timeline.epoch_bounds[..., {column}]"


def _is_movement_indexed_timing(perturbation: Mapping[str, Any]) -> bool:
    timing = perturbation.get("timing", {})
    return isinstance(timing, Mapping) and timing.get("epoch") == "movement_indexed"


def _movement_aligned_start_indices(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve absolute pulse starts for trial-time and movement-indexed rows."""

    timing = perturbation.get("timing", {})
    relative_start = int(timing.get("start_time_index", 0)) if isinstance(timing, Mapping) else 0
    if not _is_movement_indexed_timing(perturbation):
        return (
            np.full((batch_size,), relative_start, dtype=np.int64),
            {
                "timing_basis": "absolute_trial_time",
                "movement_start_aligned": False,
            },
        )
    movement_starts = _movement_start_indices(trial_specs, batch_size=batch_size)
    if movement_starts is None:
        movement_starts = np.zeros((batch_size,), dtype=np.int64)
        movement_start_source = "absent_timeline_assumed_zero_for_immediate_reach"
    else:
        movement_start_source = _movement_start_source(trial_specs)
    return (
        movement_starts + relative_start,
        {
            "timing_basis": "movement_age",
            "movement_start_aligned": True,
            "movement_start_indices": [int(start) for start in movement_starts.tolist()],
            "movement_start_source": movement_start_source,
        },
    )


def _validate_timed_pulse_indices(
    start_indices: np.ndarray,
    duration: int,
    *,
    n_time: int,
) -> str | None:
    if duration < 1:
        return f"duration={duration} must be positive"
    if np.any(start_indices < 0) or np.any(start_indices + duration > n_time):
        return f"start_indices={start_indices.tolist()}, duration={duration}, n_time={n_time}"
    return None


def delta_full_qrf_cost_summary(
    base: Mapping[str, Any],
    perturbed: Mapping[str, Any],
) -> dict[str, Any]:
    """Return paired base/perturbed/delta full-Q/R/Q_f cost summaries."""

    if base.get("status") != "available" or perturbed.get("status") != "available":
        return {
            "status": "not_available",
            "base": dict(base),
            "perturbed": dict(perturbed),
            "reason": "base and perturbed full-Q/R/Q_f summaries must both be available",
        }
    deltas: dict[str, Any] = {}
    for key in ("total", "stage_state", "control", "terminal"):
        base_values = np.asarray(base[key]["values"], dtype=np.float64)
        perturbed_values = np.asarray(perturbed[key]["values"], dtype=np.float64)
        deltas[key] = _summary_stats(perturbed_values - base_values)
    return {
        "status": "available",
        "lens": "paired_realized_deterministic_rollout_full_qrf",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "base_cost": _cost_summary_public(base),
        "perturbed_cost": _cost_summary_public(perturbed),
        "delta_cost": deltas,
    }


def evaluate_extlqg_perturbation_comparator(
    perturbation: PerturbationSpec | Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    gru_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the deterministic extLQG comparator for one perturbation row."""

    perturbation = _validated_perturbation_row(perturbation)
    channel = str(perturbation["channel"])
    supported_channels = {
        "command_input",
        "initial_state",
        "process_epsilon",
        "sensory_feedback",
        "delayed_observation",
    }
    if channel not in supported_channels:
        return extlqg_comparator_status(perturbation, status="not_applicable")
    inapplicable_reason = _extlqg_inapplicable_reason(perturbation, context=context)
    if inapplicable_reason is not None:
        return extlqg_comparator_status(
            perturbation,
            status="not_applicable",
            reason=inapplicable_reason,
        )
    required_context_keys = (
        "base_evaluation",
        "base_initial_state",
        "parity_status",
        "n_iterations",
    )
    missing_context = [key for key in required_context_keys if key not in context]
    if missing_context:
        return {
            "status": "blocked",
            "lens": "deterministic_extlqg_same_declared_perturbation",
            "reason": ("requires extLQG comparator context keys: " + ", ".join(missing_context)),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }
    try:
        base = context["base_evaluation"]
        base_initial_state = np.asarray(context["base_initial_state"], dtype=np.float64)
        (
            perturbed_evaluation,
            perturbed_initial_state,
            adapter_provenance,
        ) = _simulate_extlqg_perturbed(
            perturbation,
            context=context,
        )
        base_cost = _extlqg_cost_summary(base, base_initial_state)
        perturbed_cost = _extlqg_cost_summary(perturbed_evaluation, perturbed_initial_state)
        response_metrics = summarize_perturbation_response(
            base,
            perturbed_evaluation,
            base_full_qrf_cost=base_cost,
            perturbed_full_qrf_cost=perturbed_cost,
        )
        return {
            "status": "available",
            "lens": "deterministic_extlqg_same_declared_perturbation",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "parity_status": str(context["parity_status"]),
            "n_iterations": int(context["n_iterations"]),
            "analytical_adapter": adapter_provenance,
            "reference_response_metrics": response_metrics,
            "gru_vs_extlqg": compare_response_metric_summaries(gru_metrics, response_metrics),
        }
    except (ValueError, KeyError) as exc:
        return {
            "status": "blocked",
            "lens": "deterministic_extlqg_same_declared_perturbation",
            "reason": str(exc),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }


def _extlqg_inapplicable_reason(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> str | None:
    """Return why a row has no analytical port in the selected extLQG basis."""

    channel = str(perturbation["channel"])
    if channel == "process_epsilon":
        plant = context.get("plant")
        if plant is None:
            return None
        epsilon_dim = int(getattr(plant, "m_w"))
        epsilon_index_raw = perturbation.get("epsilon_index")
        epsilon_index = (
            _axis_index(str(perturbation["axis"]))
            if epsilon_index_raw is None
            else int(epsilon_index_raw)
        )
        if epsilon_index < 0 or epsilon_index >= epsilon_dim:
            return (
                f"process_epsilon row addresses epsilon_index {epsilon_index}, "
                f"but selected extLQG comparator exposes {epsilon_dim} process "
                "disturbance dimensions"
            )
    if channel in {"sensory_feedback", "delayed_observation"}:
        config = context.get("config")
        if config is None:
            return None
        observation_dim = int(getattr(config, "n_phys"))
        observation_index = _graph_channel_payload_index(perturbation)
        if observation_index < 0 or observation_index >= observation_dim:
            return (
                f"{channel} row addresses payload index {observation_index}, "
                f"but selected extLQG comparator exposes {observation_dim} "
                "controller-visible physical dimensions"
            )
    return None


def extlqg_comparator_status(
    perturbation: Mapping[str, Any],
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return a structured not-applicable comparator status for a row."""

    channel = str(perturbation["channel"])
    reasons = {
        "sensory_feedback": (
            "sensory_feedback rows are supported as post-noise measurement-channel "
            "offsets when evaluated by the extLQG comparator"
        ),
        "delayed_observation": (
            "delayed_observation rows are supported as clean pre-noise delayed-"
            "measurement offsets when evaluated by the extLQG comparator"
        ),
        "target_stream": (
            "target_stream rows are deferred for current fixed-target checkpoints "
            "without a controller-visible target input stream"
        ),
    }
    return {
        "status": status,
        "lens": "deterministic_extlqg_same_declared_perturbation",
        "reason": (
            reason
            or reasons.get(
                channel,
                "analytical comparator is only defined for evaluated rows with "
                "supported external analytical adapters",
            )
        ),
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
    }


def evaluate_robust_output_feedback_perturbation_comparator(
    perturbation: PerturbationSpec | Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    gru_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the robust output-feedback analytical response for one bank row."""

    perturbation = _validated_perturbation_row(perturbation)
    channel = str(perturbation["channel"])
    supported_channels = {"initial_state", "command_input", "process_epsilon"}
    if channel not in supported_channels:
        return robust_output_feedback_comparator_status(
            perturbation,
            status="not_applicable",
        )
    required_context_keys = (
        "base_evaluation",
        "base_initial_state",
        "plant",
        "schedule",
        "config",
        "solution",
        "gains",
    )
    missing_context = [key for key in required_context_keys if key not in context]
    if missing_context:
        return {
            "status": "blocked",
            "lens": "deterministic_output_feedback_robust_same_declared_perturbation",
            "reason": (
                "requires robust output-feedback comparator context keys: "
                + ", ".join(missing_context)
            ),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }
    try:
        base = context["base_evaluation"]
        base_initial_state = np.asarray(context["base_initial_state"], dtype=np.float64)
        (
            perturbed_evaluation,
            perturbed_initial_state,
            adapter_provenance,
        ) = _simulate_robust_output_feedback_perturbed(
            perturbation,
            context=context,
        )
        base_cost = _extlqg_cost_summary(base, base_initial_state)
        perturbed_cost = _extlqg_cost_summary(perturbed_evaluation, perturbed_initial_state)
        response_metrics = summarize_perturbation_response(
            base,
            perturbed_evaluation,
            base_full_qrf_cost=base_cost,
            perturbed_full_qrf_cost=perturbed_cost,
        )
        response_metrics = _with_attenuation_metrics(response_metrics, perturbation)
        return {
            "status": "available",
            "lens": "deterministic_output_feedback_robust_same_declared_perturbation",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "gamma_factor": float(context["gamma_factor"]),
            "gamma": float(context["gamma"]),
            "analytical_adapter": adapter_provenance,
            "reference_response_metrics": response_metrics,
            "gru_vs_robust_analytical": compare_response_metric_summaries(
                gru_metrics,
                response_metrics,
            ),
        }
    except (KeyError, ValueError, RuntimeError) as exc:
        return {
            "status": "blocked",
            "lens": "deterministic_output_feedback_robust_same_declared_perturbation",
            "reason": str(exc),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }


def robust_output_feedback_comparator_status(
    perturbation: Mapping[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    """Return robust analytical comparator status metadata for unsupported rows."""

    channel = str(perturbation["channel"])
    reasons = {
        "sensory_feedback": (
            "robust output-feedback released-forward replay does not yet expose "
            "post-noise measurement-offset ports; extLQG carries this row today"
        ),
        "delayed_observation": (
            "robust output-feedback released-forward replay does not yet expose "
            "clean delayed-measurement offset ports; extLQG carries this row today"
        ),
        "target_stream": (
            "fixed-target checkpoints do not expose a target stream, and the robust "
            "analytical comparator has no target-stream intervention"
        ),
    }
    return {
        "status": status,
        "lens": "deterministic_output_feedback_robust_same_declared_perturbation",
        "reason": reasons.get(
            channel,
            f"channel {channel!r} is not part of the robust output-feedback comparator",
        ),
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
    }


def compare_response_metric_summaries(
    gru_metrics: Mapping[str, Any],
    extlqg_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare GRU and extLQG perturbation metrics using their summary means."""

    comparisons: dict[str, Any] = {}
    for key in (
        "delta_action_norm",
        "delta_position_trajectory_norm_m",
        "delta_velocity_trajectory_norm_m_s",
        "delta_state_trajectory_norm",
        "delta_position_response_m.max",
        "delta_position_response_m.auc",
        "delta_state_response.max",
        "delta_state_response.auc",
        "delta_action_response.max",
        "delta_action_response.auc",
        "response_shape.peak_time_s",
        "response_shape.recovery_time_s",
        "target_relative_alignment.delta_position.abs_radial_component",
        "target_relative_alignment.delta_position.abs_tangential_component",
        "delta_endpoint_error_m",
        "delta_terminal_speed_m_s",
        "controller_io_response.delta_input_norm",
        "controller_io_response.action_per_input_gain",
    ):
        gru_mean = _metric_mean(gru_metrics, key)
        ext_mean = _metric_mean(extlqg_metrics, key)
        comparisons[key] = _scalar_delta_ratio(gru_mean, ext_mean)
    gru_cost = _metric_mean(gru_metrics.get("extra_full_qrf_cost", {}), "delta_cost.total")
    ext_cost = _metric_mean(extlqg_metrics.get("extra_full_qrf_cost", {}), "delta_cost.total")
    comparisons["extra_full_qrf_delta_total"] = _scalar_delta_ratio(gru_cost, ext_cost)
    return comparisons


def summarize_perturbation_bank(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluated perturbation rows into robust bank-level diagnostics."""

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    return {
        "status": "available" if evaluated else "not_available",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "class_summary": _class_summary_by_group(rows),
        "timing_cell_summary": _class_summary_by_group(rows, include_timing_bin=True),
        "ratio_of_means": _ratio_of_means_by_group(evaluated),
        "ratio_of_means_by_timing": _ratio_of_means_by_group(
            evaluated,
            include_timing_bin=True,
        ),
        "signed_pair_response": _signed_pair_response_summary(evaluated),
        "controller_io_response": _controller_io_bank_summary(evaluated),
        "denominator_guard": {
            "epsilon": 1e-12,
            "inflated_ratio_threshold": 10.0,
            "policy": (
                "Ratios whose denominator is at or below epsilon are suppressed; "
                "ratios above the inflated threshold retain raw numerator and "
                "denominator means."
            ),
        },
    }


def render_perturbation_response_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for tracked notes."""

    lines = [
        "# GRU perturbation-response bank",
        "",
        f"Issue: `{manifest['issue']}`. Source experiment: `{manifest['source_experiment']}`.",
        "",
        "The bank is controller-independent: it perturbs external task, command-port, "
        "process, sensory, observation, or target interfaces and does not mutate GRU "
        "internals.",
        "",
        manifest.get("semantics_correction", ""),
        "",
        "## Bank",
        "",
        "| Channel | Count |",
        "|---|---:|",
    ]
    channel_counts: dict[str, int] = {}
    for perturbation in manifest["bank"]["perturbations"]:
        channel_counts[perturbation["channel"]] = channel_counts.get(perturbation["channel"], 0) + 1
    lines.extend(f"| `{channel}` | {count} |" for channel, count in sorted(channel_counts.items()))
    lines.extend(["", "| Family | Count |", "|---|---:|"])
    family_counts: dict[str, int] = {}
    for perturbation in manifest["bank"]["perturbations"]:
        family_counts[perturbation["family"]] = family_counts.get(perturbation["family"], 0) + 1
    lines.extend(f"| `{family}` | {count} |" for family, count in sorted(family_counts.items()))
    lines.extend(["", "## Evaluation", ""])
    if not manifest["runs"]:
        lines.append("No checkpoint rollouts were evaluated in this materialization.")
    for run_id, run in manifest["runs"].items():
        counts = run["status_counts"]
        robust_summary = run.get("robust_response_summary", {})
        lines.extend(
            [
                f"### `{run_id}`",
                "",
                f"- Evaluated: {counts.get('evaluated', 0)}",
                f"- Blocked: {counts.get('blocked', 0)}",
                f"- Not implemented: {counts.get('not_implemented', 0)}",
                f"- Not applicable: {counts.get('not_applicable', 0)}",
                f"- Rollout trials per replicate: {run['n_rollout_trials_per_replicate']}",
                f"- Robust summaries: {robust_summary.get('status', 'not_available')}",
                "",
            ]
        )
        class_summary = robust_summary.get("class_summary", {})
        if class_summary.get("status") == "available":
            lines.extend(
                [
                    "#### Class-Binned Summary",
                    "",
                    "| Class | Rows | Status | Amplitudes | Mean delta action | "
                    "Max delta x | AUC delta x | Max delta state | AUC delta state | "
                    "Max delta u | AUC delta u | Peak time | Recovery time | "
                    "Mean endpoint delta | Mean terminal-speed delta | "
                    "Mean full-Q/R/Q_f delta cost | "
                    "GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | "
                    "Warnings / not applicable |",
                    "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for class_key, class_row in class_summary.get("groups", {}).items():
                metrics = class_row.get("metrics", {})
                ratio = class_row.get("gru_extlqg_delta_cost_ratio", {})
                robust_ratio = class_row.get("gru_robust_analytical_delta_cost_ratio", {})
                lines.append(
                    "| "
                    f"`{class_key}` | "
                    f"{class_row.get('n_rows', 0)} | "
                    f"{_format_status_counts(class_row.get('status_counts', {}))} | "
                    f"{_format_amplitudes(class_row.get('amplitudes', []))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_action_norm'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_position_response_m.max'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_position_response_m.auc'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_state_response.max'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_state_response.auc'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_action_response.max'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_action_response.auc'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'response_shape.peak_time_s'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'response_shape.recovery_time_s'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_endpoint_error_m'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_terminal_speed_m_s'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'extra_full_qrf_delta_cost_total'))} | "
                    f"{_format_optional_float(ratio.get('ratio_of_means'))} | "
                    f"{_format_optional_float(robust_ratio.get('ratio_of_means'))} | "
                    f"{_format_class_notes(class_row)} |"
                )
            lines.append("")
        timing_cell_summary = robust_summary.get("timing_cell_summary", {})
        if timing_cell_summary.get("status") == "available":
            lines.extend(
                [
                    "#### Timing-Cell Summary",
                    "",
                    "| Cell | Rows | Status | Amplitudes | Mean delta action | "
                    "Max delta x | AUC delta x | Mean full-Q/R/Q_f delta cost | "
                    "GRU/extLQG delta-cost ratio | GRU/robust delta-cost ratio | "
                    "Warnings / not applicable |",
                    "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for cell_key, cell_row in timing_cell_summary.get("groups", {}).items():
                metrics = cell_row.get("metrics", {})
                ratio = cell_row.get("gru_extlqg_delta_cost_ratio", {})
                robust_ratio = cell_row.get("gru_robust_analytical_delta_cost_ratio", {})
                lines.append(
                    "| "
                    f"`{cell_key}` | "
                    f"{cell_row.get('n_rows', 0)} | "
                    f"{_format_status_counts(cell_row.get('status_counts', {}))} | "
                    f"{_format_amplitudes(cell_row.get('amplitudes', []))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_action_norm'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_position_response_m.max'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'delta_position_response_m.auc'))} | "
                    f"{_format_optional_float(_class_metric_mean(metrics, 'extra_full_qrf_delta_cost_total'))} | "
                    f"{_format_optional_float(ratio.get('ratio_of_means'))} | "
                    f"{_format_optional_float(robust_ratio.get('ratio_of_means'))} | "
                    f"{_format_class_notes(cell_row)} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Residuals",
            "",
            f"- ExtLQG comparator: {manifest['extlqg_comparator']['status']} - "
            f"{manifest['extlqg_comparator']['reason']}",
            f"- Full-Q/R/Q_f perturbation cost: {manifest['full_qrf_cost']['status']} - "
            f"{manifest['full_qrf_cost']['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def _apply_initial_state_perturbation(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
) -> AdapterResult:
    family = str(perturbation["family"])
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    axis_index = _axis_index(str(perturbation["axis"]))
    if family not in {"initial_position_offset", "initial_velocity_offset"}:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=f"unsupported initial-state family {family!r}",
        )
    if _uses_movement_onset_initial_state_impulse(trial_specs, perturbation):
        return _apply_movement_onset_initial_state_process_impulse(trial_specs, perturbation)
    for init_key, init_state in trial_specs.inits.items():
        if hasattr(init_state, "pos") and family == "initial_position_offset":
            updated = _offset_array_axis(init_state.pos, axis_index, amount)
            new_state = eqx.tree_at(lambda state: state.pos, init_state, updated)
            return AdapterResult(
                status="evaluated",
                trial_specs=eqx.tree_at(lambda ts: ts.inits[init_key], trial_specs, new_state),
                adapter_provenance={"adapter": "trial_specs.inits.*.pos", "axis_index": axis_index},
            )
        if hasattr(init_state, "vel") and family == "initial_velocity_offset":
            updated = _offset_array_axis(init_state.vel, axis_index, amount)
            new_state = eqx.tree_at(lambda state: state.vel, init_state, updated)
            return AdapterResult(
                status="evaluated",
                trial_specs=eqx.tree_at(lambda ts: ts.inits[init_key], trial_specs, new_state),
                adapter_provenance={"adapter": "trial_specs.inits.*.vel", "axis_index": axis_index},
            )
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 4:
            start = 0 if family == "initial_position_offset" else 2
            vector_axis = start + axis_index
            updated = _offset_array_axis(init_state, vector_axis, amount)
            return AdapterResult(
                status="evaluated",
                trial_specs=eqx.tree_at(lambda ts: ts.inits[init_key], trial_specs, updated),
                adapter_provenance={
                    "adapter": "trial_specs.inits.*[pos_vel_vector]",
                    "axis_index": axis_index,
                    "vector_axis": vector_axis,
                },
            )
    return AdapterResult(
        status="blocked",
        trial_specs=trial_specs,
        reason="trial_specs.inits does not expose compatible effector position/velocity state",
    )


def _uses_movement_onset_initial_state_impulse(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
) -> bool:
    """Return whether delayed initial rows should use movement-onset process impulses."""

    if str(perturbation.get("family")) not in {
        "initial_position_offset",
        "initial_velocity_offset",
    }:
        return False
    if not _is_movement_indexed_timing(perturbation):
        batch_size = _infer_batch_size(trial_specs)
        try:
            movement_starts = _movement_start_indices(trial_specs, batch_size=batch_size)
        except ValueError:
            return True
        return movement_starts is not None and bool(np.any(movement_starts > 0))
    return True


def _apply_movement_onset_initial_state_process_impulse(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
) -> AdapterResult:
    """Represent delayed initial-position/velocity rows as movement-onset epsilon impulses."""

    if "epsilon" not in trial_specs.inputs:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=(
                "delayed movement-onset initial-state eval rows require an epsilon "
                "input so the offset can mirror movement-age training semantics"
            ),
        )
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    if epsilon.ndim < 3:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=f"epsilon input must have shape (batch, time, dim); got {epsilon.shape}",
        )
    family = str(perturbation["family"])
    axis_index = _axis_index(str(perturbation["axis"]))
    epsilon_index = axis_index if family == "initial_position_offset" else 2 + axis_index
    if epsilon_index < 0 or epsilon.shape[-1] <= epsilon_index:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=(
                f"epsilon input has dimension {epsilon.shape[-1]}, cannot address "
                f"movement-onset initial-state epsilon index {epsilon_index}"
            ),
        )
    batch_size = _infer_batch_size(trial_specs)
    try:
        movement_starts = _movement_start_indices(trial_specs, batch_size=batch_size)
    except ValueError as exc:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=str(exc),
        )
    if movement_starts is None:
        movement_starts = np.zeros((batch_size,), dtype=np.int64)
        movement_start_source = "absent_timeline_assumed_zero_for_immediate_reach"
    else:
        movement_start_source = _movement_start_source(trial_specs)
    timing_error = _validate_timed_pulse_indices(
        movement_starts,
        1,
        n_time=int(epsilon.shape[-2]),
    )
    if timing_error is not None:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=(
                "movement-onset initial-state process impulse timing is outside "
                f"epsilon time axis: {timing_error}"
            ),
        )
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    payload = jnp.zeros_like(epsilon)
    for batch_index, start in enumerate(movement_starts):
        payload = payload.at[batch_index, int(start), epsilon_index].add(amount)
    updated_trial_specs = _add_trial_input(trial_specs, "epsilon", epsilon + payload)
    return AdapterResult(
        status="evaluated",
        trial_specs=updated_trial_specs,
        adapter_provenance={
            "adapter": "trial_specs.inputs.epsilon",
            "movement_start_aligned": True,
            "timing_basis": "movement_age",
            "movement_start_indices": [int(start) for start in movement_starts.tolist()],
            "movement_start_source": movement_start_source,
            "start_time_index": int(movement_starts[0]),
            "absolute_start_time_indices": [int(start) for start in movement_starts.tolist()],
            "relative_start_time_index": 0,
            "duration_steps": 1,
            "epsilon_index": epsilon_index,
            "process_channel": "LinearStateSpace.B_w",
            "movement_onset_initial_state_semantics": (
                "TaskTrialSpec inits are trial-start only; delayed movement-age "
                "initial-position/velocity eval rows mirror training by using a "
                "one-step process-epsilon impulse at movement onset."
            ),
        },
    )


def _apply_legacy_plant_force_pulse(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
    *,
    plant_intervenor_label: str,
) -> AdapterResult:
    migrated = dict(perturbation)
    migrated["channel"] = "command_input"
    migrated["family"] = "command_input_pulse"
    result = _apply_command_input_pulse(
        trial_specs,
        migrated,
        model=None,
        plant_intervenor_label=plant_intervenor_label,
    )
    provenance = dict(result.adapter_provenance or {})
    provenance["deprecated_channel"] = "plant_force"
    provenance["migration"] = "plant_force_pulse -> command_input_pulse"
    return AdapterResult(
        status=result.status,
        trial_specs=result.trial_specs,
        reason=result.reason,
        adapter_provenance=provenance,
    )


def _apply_command_input_pulse(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
    *,
    model: Any | None,
    plant_intervenor_label: str,
) -> AdapterResult:
    del plant_intervenor_label
    return _apply_named_graph_channel_offset(
        trial_specs,
        perturbation,
        model=model,
        adapter_spec=_graph_adapter_spec(
            perturbation,
            label_prefix="command_input",
            source_node="efferent",
            source_port="output",
            target_node="mechanics",
            target_port="force",
            graphspec_mapping=(
                "named additive command_input channel on efferent.output -> mechanics.force"
            ),
        ),
    )


def _apply_named_graph_channel_offset(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
    *,
    model: Any | None,
    adapter_spec: AdditiveGraphChannelAdapterSpec,
) -> AdapterResult:
    """Add a time-varying graph-channel offset payload for one perturbation row."""

    effective_spec = (
        find_materialized_additive_channel_adapter(model, adapter_spec)
        if model is not None
        else None
    ) or adapter_spec
    batch_size = _infer_batch_size(trial_specs)
    timing = perturbation["timing"]
    duration = int(timing.get("duration_steps", 1))
    relative_start = int(timing.get("start_time_index", 0))
    try:
        start_indices, timing_provenance = _movement_aligned_start_indices(
            trial_specs,
            perturbation,
            batch_size=batch_size,
        )
    except ValueError as exc:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            model=model,
            reason=str(exc),
        )
    existing_payload = getattr(trial_specs, "inputs", {}).get(effective_spec.input_key)
    existing_payload_shape = np.shape(existing_payload)
    n_time = (
        int(existing_payload_shape[-2])
        if existing_payload is not None and len(existing_payload_shape) >= 2
        else _infer_trial_n_time(trial_specs, int(np.max(start_indices)) + duration)
    )
    declared_payload_dim = additive_channel_payload_dim(effective_spec)
    payload_dim = (
        int(existing_payload_shape[-1])
        if existing_payload is not None and len(existing_payload_shape) >= 1
        else declared_payload_dim
    )
    active_calibrated_components = _active_graph_channel_components(
        effective_spec,
        payload_dim=payload_dim,
    )
    if _is_force_filter_feedback_row(perturbation) and payload_dim >= 6:
        active_calibrated_components = payload_dim
    payload = np.zeros(
        (batch_size, n_time, payload_dim),
        dtype=np.float32,
    )
    axis_index = _graph_channel_payload_index(perturbation)
    if axis_index >= active_calibrated_components:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            model=model,
            reason=(
                f"Perturbation axis index {axis_index} exceeds additive-channel payload "
                f"active calibrated width {active_calibrated_components} for input "
                f"{effective_spec.input_key!r}."
            ),
        )
    timing_error = _validate_timed_pulse_indices(start_indices, duration, n_time=n_time)
    if timing_error is not None:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            model=model,
            reason=timing_error,
        )
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    for batch_index, start in enumerate(start_indices):
        payload[batch_index, int(start) : int(start) + duration, axis_index] = amount
    payload_array = jnp.asarray(payload)
    if existing_payload is not None:
        payload_array = jnp.asarray(existing_payload) + payload_array
    updated_trial_specs = _add_trial_input(
        trial_specs,
        effective_spec.input_key,
        payload_array,
    )
    updated_model = None
    provenance = {
        **additive_channel_provenance(
            effective_spec,
            adapter="feedbax.additive_channel_adapter",
        ),
        "start_time_index": int(start_indices[0]),
        "relative_start_time_index": relative_start,
        "absolute_start_time_indices": [int(start) for start in start_indices.tolist()],
        "duration_steps": duration,
        "axis_index": axis_index,
        "declared_payload_dim": declared_payload_dim,
        "effective_payload_dim": payload_dim,
        "active_calibrated_components": active_calibrated_components,
        "requires_zero_payload_base": True,
        **timing_provenance,
    }
    if payload_dim != declared_payload_dim:
        provenance["payload_shape_source"] = "existing_trial_input"
    else:
        provenance["payload_shape_source"] = "adapter_spec"
    if payload_dim > active_calibrated_components:
        provenance["inactive_force_filter_components"] = list(
            range(active_calibrated_components, payload_dim)
        )
    target = effective_spec.target
    if target.target_node == "mechanics" and target.target_port == "force":
        provenance["external_load_force"] = False
    if effective_spec is not adapter_spec:
        updated_model = model
        provenance["graph_inserted"] = False
        provenance["graph_adapter_reused"] = True
        provenance["diagnostic_requested_label"] = adapter_spec.label
        provenance["diagnostic_requested_input_key"] = adapter_spec.input_key
    elif model is not None:
        try:
            updated_model = materialize_additive_channel_adapter_on_graph(model, effective_spec)
        except ValueError as exc:
            return AdapterResult(
                status="blocked",
                trial_specs=trial_specs,
                model=model,
                reason=str(exc),
                adapter_provenance=provenance,
            )
        provenance["graph_inserted"] = True
        provenance["graph_adapter_reused"] = False
    else:
        provenance["graph_inserted"] = False
        provenance["graph_adapter_reused"] = False
        provenance["graph_insertion_requires_model"] = True
    return AdapterResult(
        status="evaluated",
        trial_specs=updated_trial_specs,
        model=updated_model,
        adapter_provenance=provenance,
    )


def _apply_process_epsilon_pulse(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
    *,
    model: Any | None,
) -> AdapterResult:
    if "epsilon" not in trial_specs.inputs:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=(
                "trial_specs.inputs lacks 'epsilon'; process_epsilon_pulse requires a "
                "model input bound to mechanics.epsilon / B_w"
            ),
            adapter_provenance={
                "adapter": "feedbax.additive_channel_adapter",
                "target_kind": "input",
                "target_node": "mechanics",
                "target_port": "epsilon",
            },
        )
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    if epsilon.ndim < 3:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=f"epsilon input must have shape (batch, time, dim); got {epsilon.shape}",
        )
    epsilon_index_raw = perturbation.get("epsilon_index")
    if epsilon_index_raw is None:
        epsilon_index = _axis_index(str(perturbation["axis"]))
    else:
        epsilon_index = int(epsilon_index_raw)
    if epsilon_index < 0 or epsilon.shape[-1] <= epsilon_index:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=(
                f"epsilon input has dimension {epsilon.shape[-1]}, cannot address "
                f"epsilon index {epsilon_index}"
            ),
        )
    timing = perturbation["timing"]
    duration = int(timing.get("duration_steps", 1))
    relative_start = int(timing.get("start_time_index", 0))
    batch_size = _infer_batch_size(trial_specs)
    try:
        start_indices, timing_provenance = _movement_aligned_start_indices(
            trial_specs,
            perturbation,
            batch_size=batch_size,
        )
    except ValueError as exc:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            model=model,
            reason=str(exc),
        )
    timing_error = _validate_timed_pulse_indices(
        start_indices,
        duration,
        n_time=int(epsilon.shape[-2]),
    )
    if timing_error is not None:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=(f"process_epsilon_pulse timing is outside epsilon time axis: {timing_error}"),
        )
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    adapter_spec = _process_epsilon_adapter_spec(
        perturbation,
        payload_dim=int(epsilon.shape[-1]),
    )
    payload = jnp.zeros_like(epsilon)
    for batch_index, start in enumerate(start_indices):
        payload = payload.at[batch_index, int(start) : int(start) + duration, epsilon_index].add(
            amount
        )
    if adapter_spec.input_key in trial_specs.inputs:
        payload = jnp.asarray(trial_specs.inputs[adapter_spec.input_key]) + payload
    updated_trial_specs = _add_trial_input(trial_specs, adapter_spec.input_key, payload)
    updated_model = None
    graph_inserted = False
    if model is not None:
        try:
            updated_model = materialize_additive_channel_adapter_on_graph(model, adapter_spec)
        except ValueError as exc:
            return AdapterResult(
                status="blocked",
                trial_specs=trial_specs,
                model=model,
                reason=str(exc),
                adapter_provenance=additive_channel_provenance(
                    adapter_spec,
                    adapter="feedbax.additive_channel_adapter",
                ),
            )
        graph_inserted = True
    return AdapterResult(
        status="evaluated",
        trial_specs=updated_trial_specs,
        model=updated_model,
        adapter_provenance={
            **additive_channel_provenance(
                adapter_spec,
                adapter="feedbax.additive_channel_adapter",
            ),
            "epsilon_component": perturbation.get("epsilon_component"),
            "epsilon_index": epsilon_index,
            "start_time_index": int(start_indices[0]),
            "relative_start_time_index": relative_start,
            "absolute_start_time_indices": [int(start) for start in start_indices.tolist()],
            "duration_steps": duration,
            "process_channel": "LinearStateSpace.B_w",
            "requires_zero_payload_base": True,
            "graph_inserted": graph_inserted,
            "graph_insertion_requires_model": model is None,
            **timing_provenance,
        },
    )


def _graph_adapter_spec(
    perturbation: Mapping[str, Any],
    *,
    label_prefix: str,
    source_node: str,
    source_port: str,
    target_node: str,
    target_port: str,
    graphspec_mapping: str,
) -> AdditiveGraphChannelAdapterSpec:
    perturbation_id = str(perturbation["perturbation_id"])
    stable_id = _stable_label(perturbation_id)
    payload_dim = 4 if target_node in {"net", "sensory"} else 2
    return AdditiveGraphChannelAdapterSpec(
        label=label_prefix,
        input_key=f"{GRAPH_ADAPTER_INPUT_PREFIX}.{label_prefix}.{stable_id}",
        adapter_node=f"{label_prefix}_{stable_id}_additive",
        payload_shape=[payload_dim],
        payload_dtype="float32",
        provenance_role="perturbation_response_input",
        metadata={
            "perturbation_id": perturbation_id,
            "graphspec_mapping": graphspec_mapping,
            "active_calibrated_components": payload_dim,
        },
        target=AdditiveGraphChannelTargetSpec(
            kind="edge",
            source_node=source_node,
            source_port=source_port,
            target_node=target_node,
            target_port=target_port,
        ),
    )


def _active_graph_channel_components(
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    payload_dim: int,
) -> int:
    active = spec.metadata.get("active_calibrated_components")
    if active is not None:
        return min(int(active), payload_dim)
    if spec.label in {"sensory_feedback", "delayed_observation"}:
        return min(4, payload_dim)
    return payload_dim


def _graph_channel_payload_index(perturbation: Mapping[str, Any]) -> int:
    channel_provenance = perturbation.get("channel_provenance")
    if isinstance(channel_provenance, Mapping):
        feedback_index = channel_provenance.get("feedback_payload_index")
        if feedback_index is not None:
            return int(feedback_index)
    feedback_index = perturbation.get("feedback_payload_index")
    if feedback_index is not None:
        return int(feedback_index)
    return _axis_index(str(perturbation["axis"]))


def _is_force_filter_feedback_row(perturbation: Mapping[str, Any]) -> bool:
    if perturbation.get("force_filter_feedback_only") is True:
        return True
    if perturbation.get("feedback_quantity") == "force_filter":
        return True
    channel_provenance = perturbation.get("channel_provenance")
    if isinstance(channel_provenance, Mapping):
        return (
            channel_provenance.get("force_filter_feedback_only") is True
            or channel_provenance.get("feedback_quantity") == "force_filter"
        )
    return False


def _extlqg_observation_sign_multiplier(
    perturbation: Mapping[str, Any],
    *,
    observation_index: int,
) -> int:
    """Map controller-visible GRU feedback signs into raw extLQG observation signs."""

    if _is_force_filter_feedback_row(perturbation):
        return 1

    feedback_quantity = perturbation.get("feedback_quantity")
    channel_provenance = perturbation.get("channel_provenance")
    if feedback_quantity is None and isinstance(channel_provenance, Mapping):
        feedback_quantity = channel_provenance.get("feedback_quantity")

    axis = str(perturbation.get("axis", ""))
    is_controller_visible_state_feedback = (
        feedback_quantity in {"position", "velocity"}
        or axis in {"x", "y", "vx", "vy"}
        or observation_index < 4
    )
    if is_controller_visible_state_feedback:
        return -1
    return 1


def _controller_visible_feedback_index(feedback_quantity: str, axis: str) -> int:
    return controller_feedback_axis_index(feedback_quantity, axis, feedback_dim=6)


def _default_feedback_amplitude(component_id: str) -> float:
    if component_id == COMPONENT_POSITION:
        return 0.01
    if component_id == COMPONENT_VELOCITY:
        return 0.05
    if component_id == COMPONENT_FORCE_FILTER:
        return 0.1
    raise ValueError(f"Unsupported controller-visible feedback quantity {component_id!r}")


def _process_epsilon_adapter_spec(
    perturbation: Mapping[str, Any],
    *,
    payload_dim: int,
) -> AdditiveGraphChannelAdapterSpec:
    perturbation_id = str(perturbation.get("perturbation_id") or "process_epsilon_pulse")
    stable_id = _stable_label(perturbation_id)
    return AdditiveGraphChannelAdapterSpec(
        label="process_epsilon",
        input_key=f"{GRAPH_ADAPTER_INPUT_PREFIX}.process_epsilon.{stable_id}",
        adapter_node=f"process_epsilon_{stable_id}_additive",
        payload_shape=[int(payload_dim)],
        payload_dtype="float32",
        provenance_role="process_disturbance",
        metadata={
            "perturbation_id": perturbation_id,
            "graphspec_mapping": "named process_epsilon channel into mechanics.epsilon / B_w",
        },
        target=AdditiveGraphChannelTargetSpec(
            kind="input",
            target_node="mechanics",
            target_port="epsilon",
        ),
    )


def _add_trial_input(trial_specs: Any, key: str, value: Any) -> Any:
    inputs = dict(trial_specs.inputs)
    inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _evaluate_model_on_trial_specs(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> RolloutEvaluation:
    product = _evaluate_model_rollout_product(
        model=model,
        task=task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=seed,
    )
    return product.to_rollout_evaluation(RolloutEvaluation)


def _evaluate_model_rollout_product(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> SelectedEvalRolloutProduct:
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, _infer_batch_size(trial_specs)),
        )

    states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(seed), n_replicates),
    )
    return SelectedEvalRolloutProduct.from_states(
        states,
        trial_specs,
        dt=0.01,
        include_mechanics_vector=True,
    )


def _build_extlqg_comparator_context(
    *,
    physical_dim: Literal[6, 8] = 8,
) -> dict[str, Any]:
    """Build deterministic analytical comparator context for perturbation rows."""

    if physical_dim == 8:
        plant, schedule = build_canonical_game()
        game_source = "rlrmp.analysis.math.cs_game_card.build_canonical_game"
    elif physical_dim == 6:
        plant, schedule = build_no_integrator_game()
        game_source = "rlrmp.analysis.math.cs_game_card.build_no_integrator_game"
    else:
        raise ValueError(f"unsupported extLQG physical_dim {physical_dim}; expected 6 or 8")
    config = OutputFeedbackConfig(n_phys=int(physical_dim))
    covariances = default_cs_noise_covariances(plant, config)
    comparator = build_extlqg_comparator_path(
        plant,
        jnp.zeros((schedule.T, plant.m_u, plant.n), dtype=jnp.float64),
        covariances,
        schedule=schedule,
        config=config,
    )
    x0 = make_cs_output_feedback_initial_state(plant, config)
    base_rollout = simulate_lqg_released_forward(
        plant,
        comparator.controller_gains,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        estimator_gains=comparator.estimator_gains,
        config=config,
    )
    return {
        "plant": plant,
        "schedule": schedule,
        "config": config,
        "comparator": comparator,
        "physical_dim": int(physical_dim),
        "game_source": game_source,
        "base_initial_state": np.asarray(x0, dtype=np.float64),
        "base_evaluation": _evaluation_from_extlqg_rollout(base_rollout, initial_state=x0),
        "parity_status": comparator.parity_status,
        "n_iterations": comparator.n_iterations,
    }


def _build_robust_output_feedback_comparator_context() -> dict[str, Any]:
    """Build deterministic robust output-feedback comparator context."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    gamma_ref = reference.gamma_references[0]
    plant = reference.plant
    schedule = reference.schedule
    config = OutputFeedbackConfig()
    covariances = robust_estimator_covariances(
        plant,
        schedule,
        gamma_ref.solution.gamma,
        config,
    )
    gains = robust_output_feedback_gains(
        plant,
        schedule,
        gamma_ref.solution,
        covariances,
        config,
    )
    x0 = make_cs_output_feedback_initial_state(plant, config)
    base_rollout = simulate_robust_released_forward(
        plant,
        schedule,
        gamma_ref.solution,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        gains=gains,
        config=config,
    )
    return {
        "plant": plant,
        "schedule": schedule,
        "config": config,
        "solution": gamma_ref.solution,
        "gains": gains,
        "gamma_factor": gamma_ref.factor,
        "gamma": gamma_ref.gamma,
        "base_initial_state": np.asarray(x0, dtype=np.float64),
        "base_evaluation": _evaluation_from_extlqg_rollout(base_rollout, initial_state=x0),
    }


def _simulate_extlqg_perturbed(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> tuple[RolloutEvaluation, np.ndarray, dict[str, Any]]:
    """Simulate one deterministic analytical perturbation row."""

    plant = context["plant"]
    schedule = context["schedule"]
    config = context["config"]
    comparator = context["comparator"]
    x0 = jnp.asarray(context["base_initial_state"], dtype=jnp.float64)
    adversary_epsilon = None
    clean_observation_offset = None
    sensory_feedback_offset = None
    command_input_offset = None
    initial_estimator_state = None
    adapter_provenance: dict[str, Any]
    if perturbation["channel"] == "initial_state":
        x0 = _perturbed_extlqg_initial_state(x0, perturbation)
        initial_estimator_state = jnp.asarray(context["base_initial_state"], dtype=jnp.float64)
        adapter_provenance = {
            "adapter": "analytical_initial_state_offset_with_nominal_estimator",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
            "information_structure": (
                "plant initial state is perturbed while estimator/controller "
                "initial state remains nominal, matching delayed GRU visibility"
            ),
        }
    elif perturbation["channel"] == "command_input":
        command_input_offset = _extlqg_command_input_offset(
            perturbation,
            schedule.T,
            plant.m_u,
        )
        adapter_provenance = {
            "adapter": "analytical_command_input_offset",
            "insertion_point": "plant.B @ (u_command + command_input_offset)",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
            "information_structure": "external actuator/plant-input pulse after controller command",
        }
    elif perturbation["channel"] == "process_epsilon":
        adversary_epsilon = _extlqg_process_epsilon(perturbation, schedule.T, plant.m_w)
        adapter_provenance = {
            "adapter": "analytical_process_epsilon_sequence",
            "process_channel": "PlantLinearization.Bw",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
        }
    elif perturbation["channel"] == "sensory_feedback":
        sensory_feedback_offset = _extlqg_observation_offset(
            perturbation,
            horizon=schedule.T,
            observation_dim=config.n_phys,
        )
        adapter_provenance = {
            "adapter": "analytical_post_noise_measurement_offset",
            "insertion_point": "y_clean + sensory_noise -> estimator innovation",
            "information_structure": "post_noise_feedback_channel",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
        }
    elif perturbation["channel"] == "delayed_observation":
        clean_observation_offset = _extlqg_observation_offset(
            perturbation,
            horizon=schedule.T,
            observation_dim=config.n_phys,
        )
        adapter_provenance = {
            "adapter": "analytical_clean_delayed_measurement_offset",
            "insertion_point": "H @ x_t -> sensory measurement before noise",
            "information_structure": "pre_noise_delayed_observation_channel",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
        }
    else:
        raise ValueError(f"extLQG comparator does not support channel {perturbation['channel']!r}")
    rollout = simulate_lqg_released_forward(
        plant,
        comparator.controller_gains,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        estimator_gains=comparator.estimator_gains,
        adversary_epsilon=adversary_epsilon,
        clean_observation_offset=clean_observation_offset,
        sensory_feedback_offset=sensory_feedback_offset,
        command_input_offset=command_input_offset,
        initial_estimator_state=initial_estimator_state,
        config=config,
    )
    return (
        _evaluation_from_extlqg_rollout(rollout, initial_state=x0),
        np.asarray(x0),
        adapter_provenance,
    )


def _simulate_robust_output_feedback_perturbed(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> tuple[RolloutEvaluation, np.ndarray, dict[str, Any]]:
    """Simulate one deterministic robust output-feedback perturbation row."""

    plant = context["plant"]
    schedule = context["schedule"]
    config = context["config"]
    solution = context["solution"]
    gains = context["gains"]
    x0 = jnp.asarray(context["base_initial_state"], dtype=jnp.float64)
    adversary_epsilon = None
    command_input_offset = None
    initial_estimator_state = None
    if perturbation["channel"] == "initial_state":
        x0 = _perturbed_extlqg_initial_state(x0, perturbation)
        initial_estimator_state = jnp.asarray(context["base_initial_state"], dtype=jnp.float64)
        adapter_provenance = {
            "adapter": "robust_analytical_initial_state_offset_with_nominal_estimator",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
            "information_structure": (
                "plant initial state is perturbed while estimator/controller "
                "initial state remains nominal, matching delayed GRU visibility"
            ),
        }
    elif perturbation["channel"] == "command_input":
        command_input_offset = _extlqg_command_input_offset(
            perturbation,
            schedule.T,
            plant.m_u,
        )
        adapter_provenance = {
            "adapter": "robust_analytical_command_input_offset",
            "insertion_point": "plant.B @ (u_command + command_input_offset)",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
            "information_structure": "external actuator/plant-input pulse after controller command",
        }
    elif perturbation["channel"] == "process_epsilon":
        adversary_epsilon = _extlqg_process_epsilon(perturbation, schedule.T, plant.m_w)
        adapter_provenance = {
            "adapter": "robust_analytical_process_epsilon_sequence",
            "process_channel": "PlantLinearization.Bw",
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
        }
    else:
        raise ValueError(
            "robust output-feedback comparator does not support channel "
            f"{perturbation['channel']!r}"
        )
    rollout = simulate_robust_released_forward(
        plant,
        schedule,
        solution,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        gains=gains,
        adversary_epsilon=adversary_epsilon,
        command_input_offset=command_input_offset,
        initial_estimator_state=initial_estimator_state,
        config=config,
    )
    return (
        _evaluation_from_extlqg_rollout(rollout, initial_state=x0),
        np.asarray(x0),
        adapter_provenance,
    )


def _evaluation_from_extlqg_rollout(
    rollout: Any,
    *,
    initial_state: Any,
) -> RolloutEvaluation:
    """Convert one analytical rollout to the GRU perturbation metric schema."""

    x = np.asarray(rollout.x, dtype=np.float64)
    command = np.asarray(rollout.u_command, dtype=np.float64)
    target = np.asarray(TARGET_POS, dtype=np.float64)
    position = x[1:, 0:2] + target[None, :]
    velocity = x[1:, 2:4]
    evaluation = RolloutEvaluation(
        position=position[None, None, :, :],
        velocity=velocity[None, None, :, :],
        command=command[None, None, :, :],
        hidden=np.zeros((1, 1, command.shape[0], 0), dtype=np.float64),
        gru_input=np.zeros((1, 1, command.shape[0], 0), dtype=np.float64),
        initial_position=np.asarray(initial_state, dtype=np.float64)[None, 0:2] + target[None, :],
        initial_velocity=np.asarray(initial_state, dtype=np.float64)[None, 2:4],
        target_position=np.broadcast_to(target, (1, command.shape[0], 2)),
        dt=0.01,
    )
    object.__setattr__(evaluation, "mechanics_vector", x[1:][None, None, :, :])
    return evaluation


def _perturbed_extlqg_initial_state(x0: Any, perturbation: Mapping[str, Any]) -> jnp.ndarray:
    """Apply an initial-state perturbation in the analytical state basis."""

    x = jnp.asarray(x0, dtype=jnp.float64)
    family = str(perturbation["family"])
    if family not in {"initial_position_offset", "initial_velocity_offset"}:
        raise ValueError(f"unsupported analytical initial-state family {family!r}")
    start = 0 if family == "initial_position_offset" else 2
    index = start + _axis_index(str(perturbation["axis"]))
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    return x.at[index].add(amount)


def _extlqg_process_epsilon(
    perturbation: Mapping[str, Any],
    horizon: int,
    epsilon_dim: int,
) -> jnp.ndarray:
    """Return an analytical process-epsilon pulse sequence."""

    epsilon_index_raw = perturbation.get("epsilon_index")
    epsilon_index = (
        _axis_index(str(perturbation["axis"]))
        if epsilon_index_raw is None
        else int(epsilon_index_raw)
    )
    if epsilon_index < 0 or epsilon_index >= epsilon_dim:
        raise ValueError(f"epsilon_index {epsilon_index} outside analytical dim {epsilon_dim}")
    timing = perturbation["timing"]
    start = int(timing.get("start_time_index", 0))
    duration = int(timing.get("duration_steps", 1))
    if start < 0 or duration < 1 or start + duration > horizon:
        raise ValueError(
            f"process_epsilon timing outside analytical horizon: {start=}, {duration=}, {horizon=}"
        )
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    epsilon = jnp.zeros((horizon, epsilon_dim), dtype=jnp.float64)
    return epsilon.at[start : start + duration, epsilon_index].set(amount)


def _extlqg_command_input_offset(
    perturbation: Mapping[str, Any],
    horizon: int,
    action_dim: int,
) -> jnp.ndarray:
    """Return an analytical external command-input pulse sequence."""

    action_index = _axis_index(str(perturbation["axis"]))
    if action_index < 0 or action_index >= action_dim:
        raise ValueError(f"command axis index {action_index} outside action dim {action_dim}")
    timing = perturbation["timing"]
    start = int(timing.get("start_time_index", 0))
    duration = int(timing.get("duration_steps", 1))
    if start < 0 or duration < 1 or start + duration > horizon:
        raise ValueError(
            f"command-input timing outside analytical horizon: {start=}, {duration=}, {horizon=}"
        )
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    offset = jnp.zeros((horizon, action_dim), dtype=jnp.float64)
    return offset.at[start : start + duration, action_index].set(amount)


def _extlqg_observation_offset(
    perturbation: Mapping[str, Any],
    *,
    horizon: int,
    observation_dim: int,
) -> jnp.ndarray:
    """Return a timed analytical observation-channel offset sequence."""

    family = str(perturbation["family"])
    if family not in {"sensory_feedback_offset", "delayed_observation_offset"}:
        raise ValueError(f"unsupported analytical observation-offset family {family!r}")
    observation_index = _graph_channel_payload_index(perturbation)
    if observation_index < 0 or observation_index >= observation_dim:
        raise ValueError(
            f"observation axis index {observation_index} outside analytical dim {observation_dim}"
        )
    timing = perturbation["timing"]
    start = int(timing.get("start_time_index", 0))
    duration = int(timing.get("duration_steps", 1))
    if start < 0 or duration < 1 or start + duration > horizon:
        raise ValueError(
            f"observation-offset timing outside analytical horizon: {start=}, "
            f"{duration=}, {horizon=}"
        )
    amount = (
        float(perturbation["amplitude"])
        * int(perturbation["sign"])
        * _extlqg_observation_sign_multiplier(
            perturbation,
            observation_index=observation_index,
        )
    )
    offset = jnp.zeros((horizon, observation_dim), dtype=jnp.float64)
    return offset.at[start : start + duration, observation_index].set(amount)


def _extlqg_cost_summary(evaluation: RolloutEvaluation, initial_state: Any) -> dict[str, Any]:
    """Return full-Q/R/Q_f summary for one analytical rollout."""

    mechanics_vector = getattr(evaluation, "mechanics_vector")
    state_dim = int(np.asarray(mechanics_vector).shape[-1])
    if state_dim not in {36, 48}:
        return {
            "status": "not_available",
            "reason": (
                "full-Q/R/Q_f scorer is defined on the 8D canonical delayed "
                "state basis (48 states) and the 6D no-integrator delayed state "
                f"basis (36 states), but selected analytical rollout has {state_dim} states"
            ),
        }
    try:
        scored = score_full_qrf_rollout_cost(
            states=mechanics_vector,
            commands=evaluation.command,
            initial_states=np.asarray(initial_state, dtype=np.float64)[None, None, :],
            target_pos=np.zeros((2,), dtype=np.float64),
        )
    except ValueError as exc:
        return {
            "status": "not_available",
            "reason": str(exc),
        }
    summary = _cost_arrays_to_summary(scored)
    summary["basis"]["state_transform"] = "analytical extLQG states are already target-centered"
    return summary


def _write_perturbation_bulk_arrays(
    base: RolloutEvaluation,
    perturbed: RolloutEvaluation,
    *,
    bulk_dir: Path,
    perturbation_id: str,
) -> Path:
    # Kept for direct evaluate_run_perturbation_bank and benchmark callers that
    # still request legacy per-row NPZ arrays; the public materializer adapter no
    # longer routes through this writer.
    mkdir_p(bulk_dir)
    path = bulk_dir / f"{perturbation_id}.npz"
    np.savez_compressed(
        path,
        delta_action=np.asarray(perturbed.command - base.command, dtype=np.float64),
        delta_gru_input=np.asarray(perturbed.gru_input - base.gru_input, dtype=np.float64),
        delta_position=np.asarray(perturbed.position - base.position, dtype=np.float64),
        delta_velocity=np.asarray(perturbed.velocity - base.velocity, dtype=np.float64),
        base_position=np.asarray(base.position, dtype=np.float64),
        perturbed_position=np.asarray(perturbed.position, dtype=np.float64),
        base_velocity=np.asarray(base.velocity, dtype=np.float64),
        perturbed_velocity=np.asarray(perturbed.velocity, dtype=np.float64),
        base_action=np.asarray(base.command, dtype=np.float64),
        perturbed_action=np.asarray(perturbed.command, dtype=np.float64),
        base_gru_input=np.asarray(base.gru_input, dtype=np.float64),
        perturbed_gru_input=np.asarray(perturbed.gru_input, dtype=np.float64),
    )
    return path


def _full_qrf_game_for_state_dim(state_dim: int) -> tuple[Any, Any, int, str]:
    """Return the C&S Q/R/Q_f schedule matching a delayed rollout state basis."""

    if state_dim == 48:
        plant, schedule = build_canonical_game()
        return (
            plant,
            schedule,
            8,
            "rlrmp.analysis.math.cs_game_card.build_canonical_game",
        )
    if state_dim == 36:
        plant, schedule = build_no_integrator_game()
        return (
            plant,
            schedule,
            6,
            "rlrmp.analysis.math.cs_game_card.build_no_integrator_game",
        )
    raise ValueError(
        "Full-Q/R/Q_f scorer expected canonical 48-state or no-integrator "
        f"36-state delayed rollout basis, got {state_dim}."
    )


def _goal_centered_vectors(values: Any, *, target_pos: Any, physical_dim: int = 8) -> Any:
    """Subtract target position from every physical block's x/y entries."""

    result = jnp.asarray(values, dtype=jnp.float64)
    target = jnp.asarray(target_pos, dtype=jnp.float64)
    if target.shape != (2,):
        raise ValueError(f"target_pos must have shape (2,), got {target.shape}")
    if result.shape[-1] % physical_dim != 0:
        raise ValueError(f"state dimension {result.shape[-1]} is not divisible by {physical_dim}")
    for start in range(0, result.shape[-1], physical_dim):
        result = result.at[..., start : start + 2].add(-target)
    return result


def _cost_arrays_to_summary(scored: Mapping[str, Any]) -> dict[str, Any]:
    """Convert scorer arrays to compact summaries while retaining values."""

    return {
        "status": str(scored["status"]),
        "lens": str(scored["lens"]),
        "basis": dict(scored["basis"]),
        "total": _summary_with_values(scored["total"]),
        "stage_state": _summary_with_values(scored["stage_state"]),
        "control": _summary_with_values(scored["control"]),
        "terminal": _summary_with_values(scored["terminal"]),
        "timewise_stage_state": _summary_stats(scored["timewise_stage_state"]),
        "timewise_control": _summary_stats(scored["timewise_control"]),
    }


def _summary_with_values(values: Any) -> dict[str, Any]:
    """Return summary statistics plus JSON-compatible leading values."""

    array = np.asarray(values, dtype=np.float64)
    return {
        **_summary_stats(array),
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _controller_io_response_summary(
    *,
    delta_input: Any,
    delta_action: Any,
) -> dict[str, Any]:
    """Summarize perturbation-induced controller input/output response."""

    input_array = np.asarray(delta_input, dtype=np.float64)
    action_array = np.asarray(delta_action, dtype=np.float64)
    if input_array.shape[-1] == 0:
        return {
            "status": "not_available",
            "reason": "controller input vector has zero width for this comparator arm",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }
    input_norm = np.linalg.norm(input_array, axis=-1)
    action_norm = np.linalg.norm(action_array, axis=-1)
    gain = action_norm / np.maximum(input_norm, 1e-12)
    return {
        "status": "available",
        "lens": "perturbation_controller_input_to_action_response",
        "input_key": "states.net.input",
        "output_key": "states.net.output",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "delta_input_norm": _summary_stats(input_norm),
        "action_per_input_gain": _summary_stats(gain),
    }


def _response_magnitude_summary(
    norm: Any,
    *,
    dt: float,
    value_label: str,
) -> dict[str, Any]:
    """Summarize max and time-integrated trajectory response magnitudes."""

    array = np.asarray(norm, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] == 0:
        return {
            "status": "not_available",
            "reason": f"{value_label} has no time axis",
        }
    return {
        "status": "available",
        "value": value_label,
        "dt_s": float(dt),
        "max": _summary_stats(np.max(array, axis=-1)),
        "auc": _summary_stats(np.sum(array, axis=-1) * float(dt)),
    }


def _response_shape_summary(
    norm: Any,
    *,
    dt: float,
    value_label: str,
    recovery_fraction: float = 0.1,
) -> dict[str, Any]:
    """Summarize peak and recovery timing for a nonnegative response trace."""

    array = np.asarray(norm, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] == 0:
        return {
            "status": "not_available",
            "reason": f"{value_label} has no time axis",
        }
    flat = array.reshape((-1, array.shape[-1]))
    peak_indices = np.argmax(flat, axis=-1)
    peak_values = np.take_along_axis(flat, peak_indices[:, None], axis=-1)[:, 0]
    recovery_times = []
    unrecovered = 0
    for trace, peak_index, peak_value in zip(flat, peak_indices, peak_values, strict=True):
        if peak_value <= 0.0:
            recovery_times.append(0.0)
            continue
        threshold = recovery_fraction * peak_value
        candidates = np.nonzero(trace[peak_index:] <= threshold)[0]
        if candidates.size == 0:
            unrecovered += 1
            continue
        recovery_times.append(float((peak_index + int(candidates[0])) * dt))
    return {
        "status": "available",
        "value": value_label,
        "dt_s": float(dt),
        "recovery_fraction_of_peak": float(recovery_fraction),
        "peak_time_s": _summary_stats(peak_indices * float(dt)),
        "peak_value": _summary_stats(peak_values),
        "recovery_time_s": _summary_stats_or_not_available(recovery_times),
        "n_unrecovered": int(unrecovered),
    }


def _target_relative_alignment_summary(
    *,
    base: RolloutEvaluation,
    delta_position: Any,
    delta_action: Any,
) -> dict[str, Any]:
    """Decompose response vectors into radial/tangential target-relative axes."""

    try:
        base_position = np.asarray(base.position, dtype=np.float64)
        target = np.asarray(base.target_position, dtype=np.float64)
        delta_pos = np.asarray(delta_position, dtype=np.float64)
        delta_act = np.asarray(delta_action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return {"status": "not_available", "reason": str(exc)}
    if base_position.shape[-1] != 2 or target.shape[-1] != 2:
        return {
            "status": "not_available",
            "reason": "target-relative alignment requires 2D position and target geometry",
        }
    try:
        target = np.broadcast_to(target[None, ...], base_position.shape)
    except ValueError as exc:
        return {
            "status": "not_available",
            "reason": f"target geometry could not broadcast to rollout positions: {exc}",
        }
    radial = target - base_position
    radial_norm = np.linalg.norm(radial, axis=-1, keepdims=True)
    valid = radial_norm[..., 0] > 1e-12
    if not np.any(valid):
        return {
            "status": "not_available",
            "reason": "no nonzero target-relative radial axis is available",
        }
    radial_unit = radial / np.maximum(radial_norm, 1e-12)
    tangential_unit = np.stack(
        [-radial_unit[..., 1], radial_unit[..., 0]],
        axis=-1,
    )
    result = {
        "status": "available",
        "basis": "target_relative_radial_tangential",
        "radial_axis": "target_position - base_position at each time step",
        "valid_fraction": float(np.mean(valid)),
        "delta_position": _radial_tangential_component_summary(
            delta_pos,
            radial_unit=radial_unit,
            tangential_unit=tangential_unit,
            valid=valid,
        ),
    }
    if delta_act.shape == delta_pos.shape:
        result["delta_action"] = _radial_tangential_component_summary(
            delta_act,
            radial_unit=radial_unit,
            tangential_unit=tangential_unit,
            valid=valid,
        )
    else:
        result["delta_action"] = {
            "status": "not_available",
            "reason": (
                "delta action shape does not match target-relative position geometry: "
                f"{delta_act.shape} vs {delta_pos.shape}"
            ),
        }
    return result


def _radial_tangential_component_summary(
    values: np.ndarray,
    *,
    radial_unit: np.ndarray,
    tangential_unit: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    radial_component = np.sum(values * radial_unit, axis=-1)
    tangential_component = np.sum(values * tangential_unit, axis=-1)
    magnitude = np.linalg.norm(values, axis=-1)
    alignment = radial_component / np.maximum(magnitude, 1e-12)
    return {
        "status": "available",
        "radial_component": _summary_stats(radial_component[valid]),
        "tangential_component": _summary_stats(tangential_component[valid]),
        "abs_radial_component": _summary_stats(np.abs(radial_component[valid])),
        "abs_tangential_component": _summary_stats(np.abs(tangential_component[valid])),
        "radial_alignment_cosine": _summary_stats(alignment[valid]),
    }


def _cost_summary_public(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Drop large paired value arrays from nested public cost summaries."""

    result = {
        "status": summary.get("status"),
        "lens": summary.get("lens"),
        "basis": dict(summary.get("basis", {})),
    }
    for key in ("total", "stage_state", "control", "terminal"):
        term = dict(summary[key])
        term.pop("values", None)
        result[key] = term
    return result


_ROBUST_RATIO_METRICS = (
    "delta_action_norm",
    "delta_position_trajectory_norm_m",
    "delta_velocity_trajectory_norm_m_s",
    "delta_state_trajectory_norm",
    "delta_endpoint_error_m",
    "delta_terminal_speed_m_s",
    "controller_io_response.delta_input_norm",
    "controller_io_response.action_per_input_gain",
    "extra_full_qrf_cost.delta_cost.total",
)

_SIGNED_PAIR_METRICS = (
    "delta_endpoint_error_m",
    "delta_terminal_speed_m_s",
    "extra_full_qrf_cost.delta_cost.total",
    "controller_io_response.delta_input_norm",
    "controller_io_response.action_per_input_gain",
)


_CLASS_SUMMARY_METRICS = (
    "delta_action_norm",
    "delta_position_trajectory_norm_m",
    "delta_velocity_trajectory_norm_m_s",
    "delta_state_trajectory_norm",
    "delta_position_response_m.max",
    "delta_position_response_m.auc",
    "delta_state_response.max",
    "delta_state_response.auc",
    "delta_action_response.max",
    "delta_action_response.auc",
    "response_shape.peak_time_s",
    "response_shape.recovery_time_s",
    "target_relative_alignment.delta_position.abs_radial_component",
    "target_relative_alignment.delta_position.abs_tangential_component",
    "delta_endpoint_error_m",
    "delta_terminal_speed_m_s",
    "attenuation_metrics.closed_loop_peak_dx_over_open_loop_peak_dx",
    "attenuation_metrics.closed_loop_auc_dx_over_open_loop_auc_dx",
    "attenuation_metrics.endpoint_delta_over_reach_length",
    "attenuation_metrics.auc_du_over_open_loop_peak_dx",
)


def _class_summary_by_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_timing_bin: bool = False,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str | None], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["channel"]),
            str(_row_family(row)),
            _row_timing_bin(row) if include_timing_bin else None,
        )
        grouped.setdefault(key, []).append(row)
    groups = {
        _class_group_key(channel, family, timing_bin): _class_group_summary(
            channel,
            family,
            group_rows,
            timing_bin=timing_bin,
        )
        for (channel, family, timing_bin), group_rows in sorted(grouped.items())
    }
    return {
        "status": "available" if groups else "not_available",
        "grouping": "channel/family/timing_bin" if include_timing_bin else "channel/family",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "groups": groups,
    }


def _class_group_summary(
    channel: str,
    family: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    timing_bin: str | None = None,
) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    metrics = {
        metric: _summary_stats_or_not_available(_available_metric_means(evaluated, metric))
        for metric in _CLASS_SUMMARY_METRICS
    }
    metrics["extra_full_qrf_delta_cost_total"] = _summary_stats_or_not_available(
        _available_metric_means(evaluated, "extra_full_qrf_cost.delta_cost.total")
    )
    comparator_rows = [
        row for row in evaluated if row.get("extlqg_comparator", {}).get("status") == "available"
    ]
    cost_ratio = _ratio_of_means(
        _available_metric_means(evaluated, "extra_full_qrf_cost.delta_cost.total"),
        _available_extlqg_metric_means(
            comparator_rows,
            "extra_full_qrf_cost.delta_cost.total",
        ),
    )
    if cost_ratio["status"] == "not_available":
        cost_ratio["reason"] = (
            "no meaningful extLQG full-Q/R/Q_f denominator for this channel/family"
        )
    robust_comparator_rows = [
        row
        for row in evaluated
        if row.get("robust_output_feedback_comparator", {}).get("status") == "available"
    ]
    robust_cost_ratio = _ratio_of_means(
        _available_metric_means(evaluated, "extra_full_qrf_cost.delta_cost.total"),
        _available_robust_output_feedback_metric_means(
            robust_comparator_rows,
            "extra_full_qrf_cost.delta_cost.total",
        ),
    )
    if robust_cost_ratio["status"] == "not_available":
        robust_cost_ratio["reason"] = (
            "no meaningful robust analytical full-Q/R/Q_f denominator for this channel/family"
        )
    return {
        "channel": channel,
        "family": family,
        "timing_bin": timing_bin,
        "n_rows": len(rows),
        "status_counts": _status_counts(rows),
        "amplitudes": sorted({float(_row_amplitude(row)) for row in rows}),
        "metrics": metrics,
        "gru_extlqg_delta_cost_ratio": cost_ratio,
        "gru_robust_analytical_delta_cost_ratio": robust_cost_ratio,
        "not_applicable_reasons": _reason_counts(
            row for row in rows if row.get("status") == "not_applicable"
        ),
        "extlqg_not_applicable_reasons": _reason_counts(
            row.get("extlqg_comparator", {})
            for row in rows
            if row.get("extlqg_comparator", {}).get("status") == "not_applicable"
        ),
        "robust_analytical_not_applicable_reasons": _reason_counts(
            row.get("robust_output_feedback_comparator", {})
            for row in rows
            if row.get("robust_output_feedback_comparator", {}).get("status") == "not_applicable"
        ),
        "denominator_warnings": _ratio_warnings(cost_ratio) + _ratio_warnings(robust_cost_ratio),
    }


def _available_extlqg_metric_means(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[float]:
    return [
        value
        for row in rows
        if (
            value := _metric_mean(
                row.get("extlqg_comparator", {}).get("reference_response_metrics", {}),
                metric,
            )
        )
        is not None
    ]


def _available_robust_output_feedback_metric_means(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[float]:
    return [
        value
        for row in rows
        if (
            value := _metric_mean(
                row.get("robust_output_feedback_comparator", {}).get(
                    "reference_response_metrics",
                    {},
                ),
                metric,
            )
        )
        is not None
    ]


def _summary_stats_or_not_available(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"status": "not_available", "count": 0, "mean": None}
    summary = _summary_stats(values)
    summary["status"] = "available"
    return summary


def _with_attenuation_metrics(
    metrics: Mapping[str, Any],
    perturbation: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach calibrated attenuation ratios when row provenance supplies denominators."""

    result = dict(metrics)
    open_loop_peak = _optional_float(perturbation.get("target_open_loop_peak_dx_m"))
    open_loop_auc = _optional_float(perturbation.get("target_open_loop_auc_dx_m_s"))
    if open_loop_auc is None:
        amplitude = _optional_float(perturbation.get("amplitude"))
        auc_per_unit = _optional_float(perturbation.get("open_loop_auc_dx_per_unit_m_s"))
        if amplitude is not None and auc_per_unit is not None:
            open_loop_auc = abs(amplitude) * auc_per_unit
    reach_length = _optional_float(perturbation.get("reach_length_m"))
    attenuation = {
        "closed_loop_peak_dx_over_open_loop_peak_dx": _scalar_ratio(
            _metric_mean(metrics, "delta_position_response_m.max"),
            open_loop_peak,
        ),
        "closed_loop_auc_dx_over_open_loop_auc_dx": _scalar_ratio(
            _metric_mean(metrics, "delta_position_response_m.auc"),
            open_loop_auc,
        ),
        "endpoint_delta_over_reach_length": _scalar_ratio(
            _metric_mean(metrics, "delta_endpoint_error_m"),
            reach_length,
        ),
        "auc_du_over_open_loop_peak_dx": _scalar_ratio(
            _metric_mean(metrics, "delta_action_response.auc"),
            open_loop_peak,
        ),
    }
    if any(value["status"] != "not_available" for value in attenuation.values()):
        result["attenuation_metrics"] = attenuation
    return result


def _scalar_ratio(
    numerator: float | None,
    denominator: float | None,
    *,
    denominator_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Return a JSON-safe scalar ratio with the bank denominator guard policy."""

    if numerator is None or denominator is None:
        return {
            "status": "not_available",
            "numerator": numerator,
            "denominator": denominator,
        }
    if abs(denominator) <= denominator_epsilon:
        return {
            "status": "denominator_guarded",
            "numerator": float(numerator),
            "denominator": float(denominator),
            "ratio": None,
            "denominator_epsilon": denominator_epsilon,
        }
    return {
        "status": "available",
        "numerator": float(numerator),
        "denominator": float(denominator),
        "ratio": float(numerator / denominator),
        "mean": float(numerator / denominator),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _reason_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason", "unspecified"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _ratio_warnings(ratio: Mapping[str, Any]) -> list[str]:
    warnings = []
    if ratio.get("status") == "denominator_guarded":
        warnings.append("denominator_guarded")
    if ratio.get("inflated_ratio") is True:
        warnings.append("inflated_ratio")
    return warnings


def _ratio_of_means_by_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_timing_bin: bool = False,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str | None], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["channel"]),
            str(_row_family(row)),
            _row_timing_bin(row) if include_timing_bin else None,
        )
        grouped.setdefault(key, []).append(row)
    return {
        _class_group_key(channel, family, timing_bin): _ratio_group_summary(group_rows)
        for (channel, family, timing_bin), group_rows in sorted(grouped.items())
    }


def _ratio_group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "metrics": {},
    }
    for metric in _ROBUST_RATIO_METRICS:
        numerators = [
            value
            for row in rows
            if (value := _metric_mean(row.get("metrics", {}), metric)) is not None
        ]
        denominators = [
            value
            for row in rows
            if (
                value := _metric_mean(
                    row.get("extlqg_comparator", {}).get("reference_response_metrics", {}),
                    metric,
                )
            )
            is not None
        ]
        summary["metrics"][metric] = _ratio_of_means(numerators, denominators)
    return summary


def _ratio_of_means(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    denominator_epsilon: float = 1e-12,
    inflated_threshold: float = 10.0,
) -> dict[str, Any]:
    if not numerators or not denominators:
        return {
            "status": "not_available",
            "n_numerator": len(numerators),
            "n_denominator": len(denominators),
        }
    numerator_mean = float(np.mean(np.asarray(numerators, dtype=np.float64)))
    denominator_mean = float(np.mean(np.asarray(denominators, dtype=np.float64)))
    if abs(denominator_mean) <= denominator_epsilon:
        return {
            "status": "denominator_guarded",
            "numerator_mean": numerator_mean,
            "denominator_mean": denominator_mean,
            "ratio_of_means": None,
            "denominator_epsilon": denominator_epsilon,
        }
    ratio = numerator_mean / denominator_mean
    payload = {
        "status": "available",
        "numerator_mean": numerator_mean,
        "denominator_mean": denominator_mean,
        "ratio_of_means": float(ratio),
        "n_numerator": len(numerators),
        "n_denominator": len(denominators),
    }
    if abs(ratio) >= inflated_threshold:
        payload["inflated_ratio"] = True
        payload["inflated_ratio_threshold"] = inflated_threshold
        payload["raw_numerator_values"] = [float(value) for value in numerators]
        payload["raw_denominator_values"] = [float(value) for value in denominators]
    return payload


def _signed_pair_response_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = _signed_pairs(rows)
    if not pairs:
        return {
            "status": "not_available",
            "reason": "no evaluated +/- signed perturbation pairs were available",
        }
    pair_summaries = []
    aggregate_by_metric: dict[str, dict[str, list[float]]] = {
        metric: {
            "odd_response": [],
            "even_nonlinear_residual": [],
            "curvature_like_symmetric_response": [],
        }
        for metric in _SIGNED_PAIR_METRICS
    }
    for key, pair in pairs.items():
        pair_metrics: dict[str, Any] = {}
        amplitude = _pair_amplitude(pair)
        for metric in _SIGNED_PAIR_METRICS:
            positive = _metric_mean(pair[1].get("metrics", {}), metric)
            negative = _metric_mean(pair[-1].get("metrics", {}), metric)
            if positive is None or negative is None:
                pair_metrics[metric] = {
                    "status": "not_available",
                    "positive_mean": positive,
                    "negative_mean": negative,
                }
                continue
            odd = 0.5 * (positive - negative)
            even = 0.5 * (positive + negative)
            curvature = even / max(amplitude * amplitude, 1e-12)
            pair_metrics[metric] = {
                "status": "available",
                "positive_mean": float(positive),
                "negative_mean": float(negative),
                "odd_response": float(odd),
                "even_nonlinear_residual": float(even),
                "curvature_like_symmetric_response": float(curvature),
                "amplitude": float(amplitude),
            }
            aggregate_by_metric[metric]["odd_response"].append(float(odd))
            aggregate_by_metric[metric]["even_nonlinear_residual"].append(float(even))
            aggregate_by_metric[metric]["curvature_like_symmetric_response"].append(
                float(curvature)
            )
        pair_summaries.append(
            {
                "pair_key": {
                    "channel": key[0],
                    "family": key[1],
                    "axis": key[2],
                    "timing": key[3],
                },
                "positive_perturbation_id": pair[1]["perturbation_id"],
                "negative_perturbation_id": pair[-1]["perturbation_id"],
                "metrics": pair_metrics,
            }
        )
    return {
        "status": "available",
        "n_pairs": len(pair_summaries),
        "pairing_rule": "channel/family/axis/timing/amplitude +/- pairs",
        "pairs": pair_summaries,
        "aggregate": {
            metric: {key: _summary_stats(values) for key, values in metric_values.items() if values}
            for metric, metric_values in aggregate_by_metric.items()
        },
    }


def _controller_io_bank_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [
        row
        for row in rows
        if row.get("metrics", {}).get("controller_io_response", {}).get("status") == "available"
    ]
    if not available:
        return {
            "status": "not_available",
            "reason": "no evaluated rows carried controller I/O response metrics",
        }
    return {
        "status": "available",
        "n_rows": len(available),
        "delta_input_norm": _summary_stats(
            _available_metric_means(
                available,
                "controller_io_response.delta_input_norm",
            )
        ),
        "action_per_input_gain": _summary_stats(
            _available_metric_means(
                available,
                "controller_io_response.action_per_input_gain",
            )
        ),
    }


def _available_metric_means(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    return [
        value for row in rows if (value := _metric_mean(row.get("metrics", {}), metric)) is not None
    ]


def _signed_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str, float], dict[int, Mapping[str, Any]]]:
    pairs: dict[tuple[str, str, str, str, float], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        sign = _row_sign(row)
        if sign not in {-1, 1}:
            continue
        key = (
            str(row["channel"]),
            str(_row_family(row)),
            str(_row_axis(row)),
            json.dumps(_row_timing(row), sort_keys=True),
            float(_row_amplitude(row)),
        )
        pairs.setdefault(key, {})[sign] = row
    return {key: pair for key, pair in pairs.items() if -1 in pair and 1 in pair}


def _pair_amplitude(pair: Mapping[int, Mapping[str, Any]]) -> float:
    return max(abs(_row_amplitude(pair[1])), abs(_row_amplitude(pair[-1])), 1e-12)


def _row_family(row: Mapping[str, Any]) -> str:
    return str(row.get("family") or _row_spec(row).get("family") or "unknown")


def _row_timing_bin(row: Mapping[str, Any]) -> str:
    spec = _row_spec(row)
    timing = _row_timing(row)
    return str(
        row.get("timing_bin")
        or spec.get("timing_bin")
        or timing.get("timing_bin")
        or timing.get("calibration_timing_bin")
        or timing.get("epoch")
        or "unspecified"
    )


def _class_group_key(channel: str, family: str, timing_bin: str | None) -> str:
    key = f"{channel}/{family}"
    if timing_bin is not None:
        key = f"{key}/{timing_bin}"
    return key


def _row_axis(row: Mapping[str, Any]) -> str:
    return str(row.get("axis") or _row_spec(row).get("axis") or "unknown")


def _row_sign(row: Mapping[str, Any]) -> int | None:
    value = row.get("sign", _row_spec(row).get("sign"))
    return None if value is None else int(value)


def _row_amplitude(row: Mapping[str, Any]) -> float:
    value = row.get("amplitude", _row_spec(row).get("amplitude", 1.0))
    return float(value)


def _row_timing(row: Mapping[str, Any]) -> Mapping[str, Any]:
    timing = row.get("timing", _row_spec(row).get("timing", {}))
    return timing if isinstance(timing, Mapping) else {}


def _row_spec(row: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = row.get("perturbation")
    return spec if isinstance(spec, Mapping) else row


def _metric_mean(metrics: Mapping[str, Any], dotted_key: str) -> float | None:
    """Read a nested metric summary mean using dot-separated keys."""

    current: Any = metrics
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, Mapping) or "mean" not in current:
        return None
    value = current["mean"]
    if value is None:
        return None
    return float(value)


def _scalar_delta_ratio(value: float | None, reference: float | None) -> dict[str, Any]:
    """Return a JSON-safe scalar delta and ratio with denominator guard."""

    if value is None or reference is None:
        return {"status": "not_available", "value": value, "reference": reference}
    denominator = abs(reference)
    ratio = None if denominator <= 1e-12 else float(value / reference)
    return {
        "status": "available",
        "gru_mean": float(value),
        "extlqg_mean": float(reference),
        "delta_mean": float(value - reference),
        "ratio_to_extlqg": ratio,
    }


def _summary_stats(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    flat = array.reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
    }


def _nested_metric_max(metrics: Mapping[str, Any], outer: str, inner: str) -> float:
    value = metrics.get(outer, {})
    if isinstance(value, Mapping):
        value = value.get(inner, {})
    if isinstance(value, Mapping):
        value = value.get("max", np.nan)
    return float(value)


def _summary_mean(metrics: Mapping[str, Any], key: str) -> float | None:
    summary = metrics.get(key)
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("mean")
    if value is None:
        return None
    return float(value)


def _class_metric_mean(metrics: Mapping[str, Any], key: str) -> float | None:
    """Read a class-summary metric stored either flat or nested."""

    value = _summary_mean(metrics, key)
    return _metric_mean(metrics, key) if value is None else value


def _format_status_counts(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _format_amplitudes(values: Sequence[Any]) -> str:
    if not values:
        return "NA"
    return ", ".join(_format_optional_float(float(value)) for value in values)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.6g}"


def _format_class_notes(class_row: Mapping[str, Any]) -> str:
    notes = list(class_row.get("denominator_warnings", []))
    ratio = class_row.get("gru_extlqg_delta_cost_ratio", {})
    if isinstance(ratio, Mapping) and ratio.get("status") == "not_available":
        reason = ratio.get("reason")
        if reason is not None:
            notes.append(str(reason))
    robust_ratio = class_row.get("gru_robust_analytical_delta_cost_ratio", {})
    if isinstance(robust_ratio, Mapping) and robust_ratio.get("status") == "not_available":
        reason = robust_ratio.get("reason")
        if reason is not None:
            notes.append(str(reason))
    for key in (
        "not_applicable_reasons",
        "extlqg_not_applicable_reasons",
        "robust_analytical_not_applicable_reasons",
    ):
        reasons = class_row.get(key, {})
        if isinstance(reasons, Mapping):
            notes.extend(f"{reason} ({count})" for reason, count in sorted(reasons.items()))
    return "; ".join(notes) if notes else "none"


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _initial_effector_position(trial_specs: Any) -> jnp.ndarray:
    for init_state in trial_specs.inits.values():
        position = getattr(init_state, "pos", None)
        if position is not None:
            return position
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            return jnp.asarray(init_state)[..., 0:2]
    raise ValueError("Trial spec does not include an effector position initial state")


def _offset_array_axis(values: Any, axis_index: int, amount: float) -> jnp.ndarray:
    array = jnp.asarray(values)
    offset = jnp.zeros_like(array)
    return array + offset.at[..., axis_index].set(amount)


def _axis_index(axis: str) -> int:
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "vx":
        return 2
    if axis == "vy":
        return 3
    raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y', 'vx', or 'vy'")


def _target_relative_axis_role(axis: str) -> str:
    """Return canonical +x reach radial/tangential role for position/velocity axes."""

    if axis in {"x", "vx"}:
        return "radial"
    if axis in {"y", "vy"}:
        return "tangential"
    raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y', 'vx', or 'vy'")


def _infer_batch_size(trial_specs: Any) -> int:
    for init_state in trial_specs.inits.values():
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        position = getattr(init_state, "pos", None)
        if position is not None:
            return int(position.shape[0])
    target = trial_specs.inputs.get("effector_target")
    if target is None:
        delayed_inputs = trial_specs.inputs.get("task")
        target = getattr(delayed_inputs, "effector_target", None)
    if target is not None and hasattr(target, "pos"):
        return int(target.pos.shape[0])
    raise ValueError("Unable to infer trial batch size")


def _infer_trial_n_time(trial_specs: Any, minimum: int) -> int:
    target = trial_specs.inputs.get("effector_target")
    if target is None:
        delayed_inputs = trial_specs.inputs.get("task")
        target = getattr(delayed_inputs, "effector_target", None)
    if target is not None and hasattr(target, "pos"):
        return max(int(target.pos.shape[-2]), minimum)
    for value in trial_specs.inputs.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2:
            return max(int(shape[-2]), minimum)
    return minimum


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


def _sign_label(sign: int) -> str:
    return "pos" if sign > 0 else "neg"


def _stable_label(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _select_reach_calibration_point(
    calibration_reach: str | float | None,
    *,
    reach_points: Sequence[Any],
) -> Any:
    if calibration_reach is None:
        return next(reach for reach in reach_points if reach.label == "seen_train_anchor_0p15")
    if isinstance(calibration_reach, str):
        for reach in reach_points:
            if reach.label == calibration_reach:
                return reach
        try:
            reach_length = float(calibration_reach)
        except ValueError as exc:
            raise ValueError(f"unknown calibration reach {calibration_reach!r}") from exc
    else:
        reach_length = float(calibration_reach)
    from rlrmp.analysis.pipelines.gru_perturbation_calibration import ReachCalibrationPoint

    return ReachCalibrationPoint(
        label=f"fixed_{reach_length:g}m",
        split="fixed/user",
        reach_length_m=reach_length,
        role="user_selected_fixed_reach_length",
    )


def _select_reach_relative_levels(
    calibration_level: str | Sequence[str] | None,
    *,
    levels: Sequence[Any],
) -> tuple[Any, ...]:
    if calibration_level is None:
        return tuple(levels)
    names = (calibration_level,) if isinstance(calibration_level, str) else tuple(calibration_level)
    by_name = {level.name: level for level in levels}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"unknown calibration level(s): {', '.join(missing)}")
    return tuple(by_name[name] for name in names)


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def write_default_bank(path: Path) -> None:
    """Retired raw writer retained only as an import-compatible stub."""

    raise RuntimeError(
        "write_default_bank is retired; use default_cs_perturbation_bank() and "
        "record the payload through Feedbax custody instead."
    )


__all__ = [
    "CALIBRATED_BANK_ID",
    "DEFAULT_BANK_ID",
    "DEFAULT_BULK_SUBDIR",
    "DEFAULT_OUTPUT_FILENAME",
    "DEFAULT_RUN_IDS",
    "DEFAULT_SOURCE_EXPERIMENT",
    "GRAPH_ADAPTER_INPUT_PREFIX",
    "SCHEMA_VERSION",
    "AdapterResult",
    "PerturbationChannel",
    "PerturbationSpec",
    "apply_perturbation_to_trial_specs",
    "compare_response_metric_summaries",
    "default_cs_calibrated_perturbation_bank",
    "default_cs_perturbation_bank",
    "delta_full_qrf_cost_summary",
    "evaluate_extlqg_perturbation_comparator",
    "evaluate_robust_output_feedback_perturbation_comparator",
    "evaluate_run_perturbation_bank",
    "extlqg_comparator_status",
    "full_qrf_cost_summary",
    "materialize_gru_perturbation_response",
    "render_perturbation_response_markdown",
    "robust_output_feedback_comparator_status",
    "score_full_qrf_rollout_cost",
    "summarize_perturbation_bank",
    "summarize_perturbation_response",
    "write_default_bank",
]

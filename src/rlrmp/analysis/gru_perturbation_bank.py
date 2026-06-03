"""Controller-independent perturbation-response bank for C&S GRU diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.graph import Component, Wire
from feedbax.types import TreeNamespace, dict_to_namespace
from jaxtyping import PRNGKeyArray, PyTree

from rlrmp.analysis.cs_game_card import TARGET_POS, build_canonical_game
from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.cs_released_simulation import (
    FixedStepPerturbation,
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    simulate_lqg_released_forward,
    zero_forward_noise_draws,
    zero_noise_covariances,
)
from rlrmp.analysis.gru_checkpoint_selection import load_validation_selected_checkpoint_model
from rlrmp.analysis.gru_evaluation_diagnostics import RolloutEvaluation
from rlrmp.analysis.gru_pilot_figures import (
    RunFigureInputs,
    initial_effector_velocity,
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


SCHEMA_VERSION = "rlrmp.gru_perturbation_bank.v2"
DEFAULT_BANK_ID = "cs_standard_perturbation_response_v2"
DEFAULT_OUTPUT_FILENAME = "gru_perturbation_response_fullqrf_validation_selected_manifest.json"
DEFAULT_NOTE_FILENAME = "gru_perturbation_response_fullqrf_validation_selected.md"
DEFAULT_BULK_SUBDIR = "perturbation_response/gru_fullqrf_validation_selected"
DEFAULT_SOURCE_EXPERIMENT = "5f70333"
DEFAULT_RESULT_EXPERIMENT = "3992394"
DEFAULT_RUN_IDS = (
    "lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64",
    "lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64",
)

PerturbationChannel = Literal[
    "initial_state",
    "command_input",
    "process_epsilon",
    "sensory_feedback",
    "delayed_observation",
    "target_stream",
]
PerturbationStatus = Literal["evaluated", "blocked", "not_implemented", "not_applicable"]

GRAPH_ADAPTER_INPUT_PREFIX = "perturbation_adapter"


@dataclass(frozen=True)
class GraphAdapterSpec:
    """Temporary pre-GraphSpec additive graph insertion contract."""

    label: str
    input_key: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    input_port: str
    output_port: str
    future_graphspec_mapping: str

    @property
    def insertion_point(self) -> str:
        """Return the source-to-target wire represented by this adapter."""

        return (
            f"{self.source_node}.{self.source_port} -> "
            f"{self.target_node}.{self.target_port}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable adapter provenance."""

        return {
            "adapter": "temporary_external_additive_graph_channel",
            "label": self.label,
            "input_key": self.input_key,
            "insertion_point": self.insertion_point,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
            "temporary_pre_graphspec": True,
            "future_graphspec_mapping": self.future_graphspec_mapping,
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
        }


class AdditiveGraphChannelAdapter(Component):
    """Add an external time-varying offset to a graph edge payload."""

    input_ports = ("signal", "offset")
    output_ports = ("signal",)

    label: str = eqx.field(static=True)

    def __init__(self, *, label: str):
        self.label = str(label)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        return {"signal": inputs["signal"] + inputs["offset"]}, state


@dataclass(frozen=True)
class PerturbationSpec:
    """Declarative perturbation row in the standard C&S bank."""

    perturbation_id: str
    channel: PerturbationChannel
    family: str
    amplitude: float
    units: str
    axis: str
    basis: str
    sign: int
    timing: Mapping[str, Any]
    adapter: str
    description: str
    epsilon_component: str | None = None
    epsilon_index: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable perturbation specification."""

        row = {
            "perturbation_id": self.perturbation_id,
            "channel": self.channel,
            "family": self.family,
            "amplitude": float(self.amplitude),
            "units": self.units,
            "axis": self.axis,
            "basis": self.basis,
            "sign": int(self.sign),
            "timing": dict(self.timing),
            "adapter": self.adapter,
            "description": self.description,
        }
        if self.epsilon_component is not None:
            row["epsilon_component"] = self.epsilon_component
        if self.epsilon_index is not None:
            row["epsilon_index"] = int(self.epsilon_index)
        return row


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


def default_cs_perturbation_bank() -> dict[str, Any]:
    """Return the JSON-serializable default C&S perturbation-response bank."""

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
    for start in (20, 40, 50):
        for axis in ("x", "y"):
            for sign in (-1, 1):
                perturbations.append(
                    PerturbationSpec(
                        perturbation_id=(
                            f"command_input_pulse__t{start}_{axis}_{_sign_label(sign)}"
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
                            "duration_steps": 5,
                        },
                        adapter="temporary_external_graph_adapter.command_input",
                        description=(
                            "Add a pulse at the post-controller command port that feeds "
                            "mechanics.force. This is not an external load-force row."
                        ),
                    )
                )
        for component_family, axis, epsilon_index, epsilon_component in process_epsilon_components:
            for sign in (-1, 1):
                perturbations.append(
                    PerturbationSpec(
                        perturbation_id=(
                            "process_epsilon_pulse__"
                            f"{epsilon_component}__t{start}_{_sign_label(sign)}"
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
                            "duration_steps": 5,
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
                    )
                )
    blocked_specs = (
        (
            "sensory_feedback_offset__x_pos",
            "sensory_feedback",
            "sensory_feedback_offset",
            "m",
            0.01,
            "x",
            "sensory_feedback_named_channel",
            "Offset the external sensory channel between sensory.output and net.feedback.",
        ),
        (
            "delayed_observation_offset__x_pos",
            "delayed_observation",
            "delayed_observation_offset",
            "m",
            0.01,
            "x",
            "observation_history_named_channel",
            "Offset the clean delayed-observation channel before sensory noise.",
        ),
        (
            "target_stream_jump__x_pos",
            "target_stream",
            "target_stream_jump",
            "m",
            0.01,
            "x",
            "target_cartesian_xy",
            "blocked because current C&S GRU input is scalar SISU, not a target stream",
        ),
    )
    for row in blocked_specs:
        perturbations.append(
            PerturbationSpec(
                perturbation_id=row[0],
                channel=row[1],
                family=row[2],
                amplitude=row[4],
                units=row[3],
                axis=row[5],
                basis=row[6],
                sign=1,
                timing={"epoch": "adapter_defined"},
                adapter=(
                    "not_applicable_current_fixed_target_checkpoint"
                    if row[1] == "target_stream"
                    else f"temporary_external_graph_adapter.{row[1]}"
                ),
                description=row[7],
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
                "delayed_observation",
                "target_stream",
            ],
            "adapter_contract": (
                "Each row records the current eager adapter and remains portable "
                "to future GraphSpec named-channel adapters."
            ),
            "temporary_eager_adapters": {
                "command_input_pulse": (
                    "Temporary eager path inserts an external additive graph component "
                    "on efferent.output -> mechanics.force and binds the row payload "
                    "from trial_specs.inputs. Future GraphSpec insertion point: "
                    "named additive command_input channel on that same edge."
                ),
                "process_epsilon_pulse": (
                    "Current eager path edits trial_specs.inputs['epsilon'] only when "
                    "the model exposes an epsilon input bound to mechanics.epsilon. "
                    "Future GraphSpec insertion point: named process channel into "
                    "LinearStateSpace.epsilon / B_w. Rows declare epsilon_component "
                    "and epsilon_index over the canonical current physical block "
                    "[px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]."
                ),
                "sensory_feedback_offset": (
                    "Temporary eager path inserts an external additive graph component "
                    "on sensory.output -> net.feedback. Future GraphSpec mapping: "
                    "named additive sensory_feedback channel after sensory noise and "
                    "before the controller feedback port."
                ),
                "delayed_observation_offset": (
                    "Temporary eager path inserts an external additive graph component "
                    "on feedback.feedback -> sensory.input. Future GraphSpec mapping: "
                    "named additive delayed_observation channel before sensory noise."
                ),
                "target_stream_jump": (
                    "Deferred: current fixed-target C&S GRU checkpoints do not consume "
                    "a target-position input stream."
                ),
            },
        },
        "signed_pairing_rule": "signed_axis_pairs; aggregate absolute and signed responses",
        "perturbations": [spec.to_json() for spec in perturbations],
    }


def apply_perturbation_to_trial_specs(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
    *,
    model: Any | None = None,
    plant_intervenor_label: str = PLANT_INTERVENOR_LABEL,
) -> AdapterResult:
    """Apply one perturbation row to external TaskTrialSpec interfaces."""

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
        return _apply_process_epsilon_pulse(trial_specs, perturbation)
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
                future_graphspec_mapping=(
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
                future_graphspec_mapping=(
                    "named additive delayed_observation channel before sensory.input "
                    "noise"
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
                "temporary_pre_graphspec": False,
                "future_graphspec_mapping": "target_stream named graph input when models consume it",
                "controller_input_mutated": False,
            },
        )
    return AdapterResult(
        status="blocked",
        trial_specs=trial_specs,
        model=model,
        reason=f"unknown perturbation channel {channel!r}",
    )


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
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize the standard C&S perturbation-response bank and GRU responses."""

    bank = default_cs_perturbation_bank()
    output_path = output_path or (
        repo_root / "results" / result_experiment / "notes" / DEFAULT_OUTPUT_FILENAME
    )
    note_path = note_path or (
        repo_root / "results" / result_experiment / "notes" / DEFAULT_NOTE_FILENAME
    )
    bulk_dir = bulk_dir or repo_root / "_artifacts" / result_experiment / DEFAULT_BULK_SUBDIR
    mkdir_p(output_path.parent)
    if write_bulk_arrays and evaluate:
        mkdir_p(bulk_dir)

    run_summaries: dict[str, Any] = {}
    if evaluate:
        runs = resolve_run_inputs(
            experiment=source_experiment,
            run_ids=run_ids,
            labels=labels,
            repo_root=repo_root,
        )
        for run in runs:
            run_summaries[run.run_id] = evaluate_run_perturbation_bank(
                run,
                source_experiment=source_experiment,
                bank=bank,
                n_rollout_trials=n_rollout_trials,
                write_bulk_arrays=write_bulk_arrays,
                bulk_dir=bulk_dir,
                repo_root=repo_root,
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "source_experiment": source_experiment,
        "checkpoint_policy": "validation_selected_per_replicate",
        "scope": "controller_independent_perturbation_response",
        "semantics_correction": (
            "v2 splits the former plant_force rows into command_input_pulse "
            "(post-controller command-port perturbations) and process_epsilon_pulse "
            "(mechanics.epsilon / B_w process perturbations). Process-epsilon "
            "rows span the canonical current physical block [px, py, vx, vy, "
            "fx, fy, eps_x_int, eps_y_int]."
        ),
        "bank": bank,
        "extlqg_comparator": {
            "status": "available_for_initial_state_and_process_epsilon",
            "reason": (
                "Deterministic extLQG response rows are evaluated for perturbations "
                "with clean analytical interfaces: initial_state and process_epsilon. "
                "Command-input, sensory-feedback, and delayed-observation GRU rows "
                "are evaluated through temporary external graph adapters, but do not "
                "yet have matching analytical extLQG channel adapters; target-stream "
                "is deferred for current fixed-target checkpoints."
            ),
            "checkpoint_selection_role": "audit_only_not_used_for_selection",
        },
        "full_qrf_cost": {
            "status": "available",
            "lens": "realized_deterministic_rollout_full_qrf",
            "reason": (
                "Costs are rescored post hoc from states.mechanics.vector and "
                "states.net.output using the canonical C&S Q_t/R_t/Q_f schedule. "
                "They are audit-only perturbation diagnostics and are not used for "
                "checkpoint selection."
            ),
        },
        "runs": run_summaries,
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    note_path.write_text(render_perturbation_response_markdown(manifest), encoding="utf-8")
    return manifest


def evaluate_run_perturbation_bank(
    run: RunFigureInputs,
    *,
    source_experiment: str,
    bank: Mapping[str, Any],
    n_rollout_trials: int,
    write_bulk_arrays: bool,
    bulk_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate one validation-selected GRU run on a perturbation bank."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=source_experiment,
        run_id=run.run_id,
        run_spec=run.run_spec,
        repo_root=repo_root,
    )
    base_trial_specs = repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)
    base_evaluation = _evaluate_model_on_trial_specs(
        model=model,
        task=pair.task,
        trial_specs=base_trial_specs,
        n_replicates=n_replicates,
        seed=0,
    )
    base_cost = full_qrf_cost_summary(base_evaluation, base_trial_specs)
    extlqg_context = _build_extlqg_comparator_context()
    rows = []
    bulk_files: dict[str, str] = {}
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
                    "status": adapter.status,
                    "reason": adapter.reason,
                    "adapter": adapter.to_json(),
                    "extlqg_comparator": extlqg_comparator_status(
                        perturbation,
                        status="not_applicable",
                    ),
                }
            )
            continue
        perturbed_evaluation = _evaluate_model_on_trial_specs(
            model=adapter.model if adapter.model is not None else model,
            task=pair.task,
            trial_specs=adapter.trial_specs,
            n_replicates=n_replicates,
            seed=0,
        )
        perturbed_cost = full_qrf_cost_summary(perturbed_evaluation, adapter.trial_specs)
        metrics = summarize_perturbation_response(
            base_evaluation,
            perturbed_evaluation,
            base_full_qrf_cost=base_cost,
            perturbed_full_qrf_cost=perturbed_cost,
        )
        extlqg_comparator = evaluate_extlqg_perturbation_comparator(
            perturbation,
            context=extlqg_context,
            gru_metrics=metrics,
        )
        bulk_file = None
        if write_bulk_arrays:
            bulk_file = _write_perturbation_bulk_arrays(
                base_evaluation,
                perturbed_evaluation,
                bulk_dir=bulk_dir / run.run_id,
                perturbation_id=str(perturbation["perturbation_id"]),
            )
            bulk_files[str(perturbation["perturbation_id"])] = _repo_relative(
                bulk_file,
                repo_root=repo_root,
            )
        rows.append(
            {
                "perturbation_id": perturbation["perturbation_id"],
                "channel": perturbation["channel"],
                "status": "evaluated",
                "adapter": adapter.to_json(),
                "metrics": metrics,
                "extlqg_comparator": extlqg_comparator,
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
                        "base_gru_input",
                        "perturbed_gru_input",
                    ],
                },
            }
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
        "perturbations": rows,
        "bulk_files": bulk_files,
    }


def summarize_perturbation_response(
    base: RolloutEvaluation,
    perturbed: RolloutEvaluation,
    *,
    base_full_qrf_cost: Mapping[str, Any] | None = None,
    perturbed_full_qrf_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute paired perturbation-response metrics."""

    delta_action = perturbed.command - base.command
    delta_input = perturbed.gru_input - base.gru_input
    delta_position = perturbed.position - base.position
    delta_velocity = perturbed.velocity - base.velocity
    endpoint_recovery = np.linalg.norm(
        perturbed.position[:, :, -1, :] - perturbed.target_position[None, :, -1, :],
        axis=-1,
    )
    base_endpoint = np.linalg.norm(
        base.position[:, :, -1, :] - base.target_position[None, :, -1, :],
        axis=-1,
    )
    terminal_speed = np.linalg.norm(perturbed.velocity[:, :, -1, :], axis=-1)
    base_terminal_speed = np.linalg.norm(base.velocity[:, :, -1, :], axis=-1)
    metrics = {
        "delta_action_norm": _summary_stats(np.linalg.norm(delta_action, axis=-1)),
        "delta_position_trajectory_norm_m": _summary_stats(np.linalg.norm(delta_position, axis=-1)),
        "delta_velocity_trajectory_norm_m_s": _summary_stats(
            np.linalg.norm(delta_velocity, axis=-1)
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


def score_full_qrf_rollout_cost(
    *,
    states: Any,
    commands: Any,
    initial_states: Any,
    target_pos: Any = TARGET_POS,
) -> dict[str, Any]:
    """Score realized full analytical Q/R/Q_f costs with per-term arrays.

    Args:
        states: Rollout mechanics vectors with shape ``(..., T, 48)``.
        commands: Controller commands with shape ``(..., T, 2)``.
        initial_states: Initial mechanics vectors broadcastable to ``(..., 48)``.
        target_pos: Absolute target position subtracted from every physical
            delay block's x/y position coordinates before applying Q/Q_f.

    Returns:
        JSON-compatible arrays for total, stage-state, control, and terminal
        costs. Arrays retain all leading dimensions so replicate variability is
        not hidden.
    """

    _plant, schedule = build_canonical_game()
    state_array = np.asarray(states, dtype=np.float64)
    command_array = np.asarray(commands, dtype=np.float64)
    initial_array = np.asarray(initial_states, dtype=np.float64)
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
            f"Full-Q/R/Q_f scorer expected {horizon} rollout states, "
            f"got {state_array.shape[-2]}."
        )
    if command_array.shape[-2] != horizon:
        raise ValueError(
            f"Full-Q/R/Q_f scorer expected {horizon} commands, got {command_array.shape[-2]}."
        )
    initial_array = np.broadcast_to(initial_array, (*state_array.shape[:-2], state_array.shape[-1]))
    x_pre = np.concatenate([initial_array[..., None, :], state_array[..., :-1, :]], axis=-2)
    x_pre = _goal_centered_vectors(x_pre, target_pos=target_pos)
    x_terminal = _goal_centered_vectors(state_array[..., -1, :], target_pos=target_pos)
    q = np.asarray(schedule.Q, dtype=np.float64)
    r = np.asarray(schedule.R, dtype=np.float64)
    q_f = np.asarray(schedule.Q_f, dtype=np.float64)
    state_terms = np.einsum("...ti,tij,...tj->...t", x_pre, q, x_pre)
    control_terms = np.einsum("...ti,tij,...tj->...t", command_array, r, command_array)
    terminal_terms = np.einsum("...i,ij,...j->...", x_terminal, q_f, x_terminal)
    stage_state = np.sum(state_terms, axis=-1)
    control = np.sum(control_terms, axis=-1)
    total = stage_state + control + terminal_terms
    return {
        "status": "available",
        "lens": "realized_deterministic_rollout_full_qrf",
        "basis": {
            "state_key": "states.mechanics.vector",
            "command_key": "states.net.output",
            "state_transform": "subtract TARGET_POS from each physical delay block x/y",
            "schedule_source": "rlrmp.analysis.cs_game_card.build_canonical_game",
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
    scored = score_full_qrf_rollout_cost(
        states=mechanics_vector,
        commands=evaluation.command,
        initial_states=trial_specs.inits["mechanics.vector"],
    )
    return _cost_arrays_to_summary(scored)


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
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    gru_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the deterministic extLQG comparator for one perturbation row."""

    channel = str(perturbation["channel"])
    if channel not in {"initial_state", "process_epsilon"}:
        return extlqg_comparator_status(perturbation, status="not_applicable")
    try:
        base = context["base_evaluation"]
        base_initial_state = np.asarray(context["base_initial_state"], dtype=np.float64)
        perturbed_evaluation, perturbed_initial_state = _simulate_extlqg_perturbed(
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


def extlqg_comparator_status(
    perturbation: Mapping[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    """Return a structured not-applicable comparator status for a row."""

    channel = str(perturbation["channel"])
    reasons = {
        "command_input": (
            "command_input rows require the graph/channel adapter lane to define a "
            "matching analytical command-port intervention"
        ),
        "sensory_feedback": (
            "sensory_feedback rows require a matching extLQG measurement-channel "
            "adapter rather than controller-internal mutation"
        ),
        "delayed_observation": (
            "delayed_observation rows require a clean pre-noise observation-history "
            "adapter in both GRU and analytical paths"
        ),
        "target_stream": (
            "target_stream rows are deferred for current fixed-target checkpoints "
            "without a controller-visible target input stream"
        ),
    }
    return {
        "status": status,
        "lens": "deterministic_extlqg_same_declared_perturbation",
        "reason": reasons.get(
            channel,
            "analytical comparator is only defined for evaluated initial_state and process_epsilon rows",
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
        lines.extend(
            [
                f"### `{run_id}`",
                "",
                f"- Evaluated: {counts.get('evaluated', 0)}",
                f"- Blocked: {counts.get('blocked', 0)}",
                f"- Not implemented: {counts.get('not_implemented', 0)}",
                f"- Not applicable: {counts.get('not_applicable', 0)}",
                f"- Rollout trials per replicate: {run['n_rollout_trials_per_replicate']}",
                "",
            ]
        )
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
            future_graphspec_mapping=(
                "named additive command_input channel on efferent.output -> "
                "mechanics.force"
            ),
        ),
    )


def _apply_named_graph_channel_offset(
    trial_specs: Any,
    perturbation: Mapping[str, Any],
    *,
    model: Any | None,
    adapter_spec: GraphAdapterSpec,
) -> AdapterResult:
    """Add a time-varying graph-channel offset payload for one perturbation row."""

    batch_size = _infer_batch_size(trial_specs)
    timing = perturbation["timing"]
    start = int(timing.get("start_time_index", 0))
    duration = int(timing.get("duration_steps", 1))
    n_time = _infer_trial_n_time(trial_specs, start + duration)
    payload = np.zeros((batch_size, n_time, _adapter_payload_dim(adapter_spec)), dtype=np.float32)
    axis_index = _axis_index(str(perturbation["axis"]))
    payload[:, start : start + duration, axis_index] = (
        float(perturbation["amplitude"]) * int(perturbation["sign"])
    )
    updated_trial_specs = _add_trial_input(
        trial_specs,
        adapter_spec.input_key,
        jnp.asarray(payload),
    )
    updated_model = None
    provenance = {
        **adapter_spec.to_json(),
        "start_time_index": start,
        "duration_steps": duration,
        "axis_index": axis_index,
    }
    if adapter_spec.target_node == "mechanics" and adapter_spec.target_port == "force":
        provenance["external_load_force"] = False
    if model is not None:
        try:
            updated_model = insert_additive_graph_channel_adapter(model, adapter_spec)
        except ValueError as exc:
            return AdapterResult(
                status="blocked",
                trial_specs=trial_specs,
                model=model,
                reason=str(exc),
                adapter_provenance=provenance,
            )
        provenance["graph_inserted"] = True
    else:
        provenance["graph_inserted"] = False
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
                "adapter": "trial_specs.inputs['epsilon']",
                "future_graphspec_insertion_point": "mechanics.epsilon",
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
    start = int(timing.get("start_time_index", 0))
    duration = int(timing.get("duration_steps", 1))
    if start < 0 or duration < 1 or start + duration > epsilon.shape[-2]:
        return AdapterResult(
            status="blocked",
            trial_specs=trial_specs,
            reason=(
                "process_epsilon_pulse timing is outside epsilon time axis: "
                f"start={start}, duration={duration}, n_time={epsilon.shape[-2]}"
            ),
        )
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    updated = epsilon.at[..., start : start + duration, epsilon_index].add(amount)
    return AdapterResult(
        status="evaluated",
        trial_specs=eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, updated),
        adapter_provenance={
            "adapter": "trial_specs.inputs['epsilon']",
            "epsilon_component": perturbation.get("epsilon_component"),
            "epsilon_index": epsilon_index,
            "start_time_index": start,
            "duration_steps": duration,
            "future_graphspec_insertion_point": "mechanics.epsilon",
            "process_channel": "LinearStateSpace.B_w",
            "temporary_eager_adapter": True,
        },
    )


def insert_additive_graph_channel_adapter(model: Any, adapter_spec: GraphAdapterSpec) -> Any:
    """Insert a temporary external additive adapter and bind its input payload."""

    if adapter_spec.label in getattr(model, "nodes", {}):
        return model
    old_wire = Wire(
        adapter_spec.source_node,
        adapter_spec.source_port,
        adapter_spec.target_node,
        adapter_spec.target_port,
    )
    graph = model.remove_wire(old_wire)
    graph = graph.add_node(
        adapter_spec.label,
        AdditiveGraphChannelAdapter(label=adapter_spec.label),
    )
    graph = graph.add_wire(
        Wire(
            adapter_spec.source_node,
            adapter_spec.source_port,
            adapter_spec.label,
            adapter_spec.input_port,
        )
    )
    graph = graph.add_wire(
        Wire(
            adapter_spec.label,
            adapter_spec.output_port,
            adapter_spec.target_node,
            adapter_spec.target_port,
        )
    )
    graph = eqx.tree_at(
        lambda g: g.input_ports,
        graph,
        (*graph.input_ports, adapter_spec.input_key),
    )
    graph = eqx.tree_at(
        lambda g: g.input_bindings,
        graph,
        {**graph.input_bindings, adapter_spec.input_key: (adapter_spec.label, "offset")},
    )
    return graph


def _graph_adapter_spec(
    perturbation: Mapping[str, Any],
    *,
    label_prefix: str,
    source_node: str,
    source_port: str,
    target_node: str,
    target_port: str,
    future_graphspec_mapping: str,
) -> GraphAdapterSpec:
    perturbation_id = str(perturbation["perturbation_id"])
    label = f"{GRAPH_ADAPTER_INPUT_PREFIX}_{label_prefix}_{_stable_label(perturbation_id)}"
    input_key = f"{GRAPH_ADAPTER_INPUT_PREFIX}:{perturbation_id}"
    return GraphAdapterSpec(
        label=label,
        input_key=input_key,
        source_node=source_node,
        source_port=source_port,
        target_node=target_node,
        target_port=target_port,
        input_port="signal",
        output_port="signal",
        future_graphspec_mapping=future_graphspec_mapping,
    )


def _add_trial_input(trial_specs: Any, key: str, value: Any) -> Any:
    inputs = dict(trial_specs.inputs)
    inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _adapter_payload_dim(adapter_spec: GraphAdapterSpec) -> int:
    if adapter_spec.target_node == "net" and adapter_spec.target_port == "feedback":
        return 4
    if adapter_spec.target_node == "sensory" and adapter_spec.target_port == "input":
        return 4
    if adapter_spec.target_node == "mechanics" and adapter_spec.target_port == "force":
        return 2
    raise ValueError(f"Unsupported graph adapter insertion point {adapter_spec.insertion_point!r}")


def _evaluate_model_on_trial_specs(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> RolloutEvaluation:
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
    target_position = np.asarray(trial_specs.inputs["effector_target"].pos, dtype=np.float64)
    evaluation = RolloutEvaluation(
        position=np.asarray(states.mechanics.effector.pos, dtype=np.float64),
        velocity=np.asarray(states.mechanics.effector.vel, dtype=np.float64),
        command=np.asarray(states.net.output, dtype=np.float64),
        hidden=np.asarray(states.net.hidden, dtype=np.float64),
        gru_input=np.asarray(states.net.input, dtype=np.float64),
        initial_position=np.asarray(_initial_effector_position(trial_specs), dtype=np.float64),
        initial_velocity=np.asarray(initial_effector_velocity(trial_specs), dtype=np.float64),
        target_position=target_position,
        dt=0.01,
    )
    object.__setattr__(
        evaluation,
        "mechanics_vector",
        np.asarray(states.mechanics.vector, dtype=np.float64),
    )
    return evaluation


def _build_extlqg_comparator_context() -> dict[str, Any]:
    """Build deterministic analytical comparator context for perturbation rows."""

    plant, schedule = build_canonical_game()
    config = OutputFeedbackConfig()
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
        "base_initial_state": np.asarray(x0, dtype=np.float64),
        "base_evaluation": _evaluation_from_extlqg_rollout(base_rollout, initial_state=x0),
        "parity_status": comparator.parity_status,
        "n_iterations": comparator.n_iterations,
    }


def _simulate_extlqg_perturbed(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> tuple[RolloutEvaluation, np.ndarray]:
    """Simulate one deterministic analytical perturbation row."""

    plant = context["plant"]
    schedule = context["schedule"]
    config = context["config"]
    comparator = context["comparator"]
    x0 = jnp.asarray(context["base_initial_state"], dtype=jnp.float64)
    adversary_epsilon = None
    if perturbation["channel"] == "initial_state":
        x0 = _perturbed_extlqg_initial_state(x0, perturbation)
    elif perturbation["channel"] == "process_epsilon":
        adversary_epsilon = _extlqg_process_epsilon(perturbation, schedule.T, plant.m_w)
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
        config=config,
    )
    return _evaluation_from_extlqg_rollout(rollout, initial_state=x0), np.asarray(x0)


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
            f"process_epsilon timing outside analytical horizon: {start=}, "
            f"{duration=}, {horizon=}"
        )
    amount = float(perturbation["amplitude"]) * int(perturbation["sign"])
    epsilon = jnp.zeros((horizon, epsilon_dim), dtype=jnp.float64)
    return epsilon.at[start : start + duration, epsilon_index].set(amount)


def _extlqg_cost_summary(evaluation: RolloutEvaluation, initial_state: Any) -> dict[str, Any]:
    """Return full-Q/R/Q_f summary for one analytical rollout."""

    scored = score_full_qrf_rollout_cost(
        states=getattr(evaluation, "mechanics_vector"),
        commands=evaluation.command,
        initial_states=np.asarray(initial_state, dtype=np.float64)[None, None, :],
        target_pos=np.zeros((2,), dtype=np.float64),
    )
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
    mkdir_p(bulk_dir)
    path = bulk_dir / f"{perturbation_id}.npz"
    np.savez_compressed(
        path,
        delta_action=perturbed.command - base.command,
        delta_gru_input=perturbed.gru_input - base.gru_input,
        delta_position=perturbed.position - base.position,
        delta_velocity=perturbed.velocity - base.velocity,
        base_position=base.position,
        perturbed_position=perturbed.position,
        base_velocity=base.velocity,
        perturbed_velocity=perturbed.velocity,
        base_action=base.command,
        perturbed_action=perturbed.command,
        base_gru_input=base.gru_input,
        perturbed_gru_input=perturbed.gru_input,
    )
    return path


def _goal_centered_vectors(values: Any, *, target_pos: Any) -> np.ndarray:
    """Subtract target position from every 8D physical block's x/y entries."""

    result = np.array(values, dtype=np.float64, copy=True)
    target = np.asarray(target_pos, dtype=np.float64)
    if target.shape != (2,):
        raise ValueError(f"target_pos must have shape (2,), got {target.shape}")
    if result.shape[-1] % 8 != 0:
        raise ValueError(f"state dimension {result.shape[-1]} is not divisible by 8")
    for start in range(0, result.shape[-1], 8):
        result[..., start : start + 2] -= target
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
    raise ValueError(f"Unsupported axis {axis!r}; expected 'x' or 'y'")


def _infer_batch_size(trial_specs: Any) -> int:
    for init_state in trial_specs.inits.values():
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
        position = getattr(init_state, "pos", None)
        if position is not None:
            return int(position.shape[0])
    target = trial_specs.inputs.get("effector_target")
    if target is not None and hasattr(target, "pos"):
        return int(target.pos.shape[0])
    raise ValueError("Unable to infer trial batch size")


def _infer_trial_n_time(trial_specs: Any, minimum: int) -> int:
    target = trial_specs.inputs.get("effector_target")
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


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def write_default_bank(path: Path) -> None:
    """Write the default bank schema to a JSON file."""

    mkdir_p(path.parent)
    path.write_text(
        json.dumps(default_cs_perturbation_bank(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_BANK_ID",
    "DEFAULT_BULK_SUBDIR",
    "DEFAULT_OUTPUT_FILENAME",
    "DEFAULT_RUN_IDS",
    "DEFAULT_SOURCE_EXPERIMENT",
    "GRAPH_ADAPTER_INPUT_PREFIX",
    "SCHEMA_VERSION",
    "AdditiveGraphChannelAdapter",
    "AdapterResult",
    "GraphAdapterSpec",
    "PerturbationSpec",
    "apply_perturbation_to_trial_specs",
    "compare_response_metric_summaries",
    "default_cs_perturbation_bank",
    "delta_full_qrf_cost_summary",
    "evaluate_extlqg_perturbation_comparator",
    "evaluate_run_perturbation_bank",
    "extlqg_comparator_status",
    "full_qrf_cost_summary",
    "insert_additive_graph_channel_adapter",
    "materialize_gru_perturbation_response",
    "render_perturbation_response_markdown",
    "score_full_qrf_rollout_cost",
    "summarize_perturbation_response",
    "write_default_bank",
]

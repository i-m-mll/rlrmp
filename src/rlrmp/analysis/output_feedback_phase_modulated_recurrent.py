"""Phase-modulated linear recurrent output-feedback bridge for issue d6d25d6."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.bridge_certificates import (
    DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH,
    OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    build_standard_certificate_components,
)
from rlrmp.analysis.bridge_contracts import (
    BridgeRolloutBatch,
    BridgeRunManifest,
    BridgeRunSpec,
    make_bridge_run_id,
)
from rlrmp.analysis.bridge_controllers import (
    MatrixBasisProjection,
    PhaseModulatedLinearRecurrentController,
    clamped_bspline_time_basis,
    hidden_growth_diagnostics,
    project_matrix_sequence_to_basis,
)
from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    kalman_estimator_gains,
    make_cs_output_feedback_initial_state,
    output_feedback_cost,
    process_covariance,
    rollout_with_kalman_estimator,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE_ID = "d6d25d6"
UMBRELLA_ID = "1fabee8"
RELATED_RECURRENT_ISSUE_ID = "5e55f69"
IO_MAP_CERTIFICATE_ISSUE_ID = "007087e"

NOTE_PATH = REPO_ROOT / "results" / ISSUE_ID / "notes" / "phase_modulated_recurrent.md"
MANIFEST_PATH = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "phase_modulated_recurrent_manifest.json"
)
ARTIFACT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "phase_modulated_recurrent"
    / "phase_modulated_recurrent.npz"
)

LEGACY_AUDIT_RANKS = (3, 5, 8)
PROJECTION_SWEEP_RANKS = (12, 20, 30, 60)
SPLINE_RANKS = LEGACY_AUDIT_RANKS + PROJECTION_SWEEP_RANKS
DEFAULT_PROCESS_IO_DISTURBANCE_SCALE = 0.02
DEFAULT_MEASUREMENT_IO_DISTURBANCE_SCALE = 0.02
DEFAULT_REWARD_TRAIN_STEPS = 80
DEFAULT_REWARD_LEARNING_RATE = 3e-3
DEFAULT_REWARD_STABILITY_PENALTY = 1e-3


@dataclass(frozen=True)
class OracleRecurrentReference:
    """Finite-horizon Kalman recurrence realizing the analytical reference."""

    basis_state: str
    observation_matrix: np.ndarray
    recurrent_matrices: np.ndarray
    observation_matrices: np.ndarray
    previous_action_matrices: np.ndarray
    hidden_biases: np.ndarray
    readout_matrices: np.ndarray
    feedthrough_matrices: np.ndarray
    action_biases: np.ndarray
    initial_hidden: np.ndarray


@dataclass(frozen=True)
class PhaseModulatedCondition:
    """One d6d25d6 materialized row."""

    label: str
    row_family: str
    rank: int
    training_distribution: str
    evaluation_lens: str
    coverage_family: str | None = None
    coverage_modes: int | None = None
    coverage_scale: float = 0.0
    disturbance_scale: float = 0.0
    measurement_scale: float = 0.0
    n_train_steps: int = DEFAULT_REWARD_TRAIN_STEPS
    learning_rate: float = DEFAULT_REWARD_LEARNING_RATE
    stability_penalty: float = DEFAULT_REWARD_STABILITY_PENALTY
    seed: int = 0

    @property
    def run_id(self) -> str:
        """Stable row identifier."""

        return make_bridge_run_id("phase_modulated_recurrent", self.label)


def default_conditions(*, include_reward: bool = True) -> tuple[PhaseModulatedCondition, ...]:
    """Return the planned exact-oracle, projected-oracle, and reward rows."""

    exact_oracle = (
        PhaseModulatedCondition(
            label="pm_linrec_exact_oracle_nominal_replay",
            row_family="exact_oracle_sanity",
            rank=60,
            training_distribution="nominal",
            evaluation_lens="nominal_clean",
        ),
        PhaseModulatedCondition(
            label="pm_linrec_exact_oracle_process_io",
            row_family="exact_oracle_sanity",
            rank=60,
            training_distribution="process_io_probe",
            evaluation_lens="process_io",
            disturbance_scale=DEFAULT_PROCESS_IO_DISTURBANCE_SCALE,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_exact_oracle_process_measurement_io",
            row_family="exact_oracle_sanity",
            rank=60,
            training_distribution="process_measurement_io_probe",
            evaluation_lens="process_measurement_io",
            disturbance_scale=DEFAULT_PROCESS_IO_DISTURBANCE_SCALE,
            measurement_scale=DEFAULT_MEASUREMENT_IO_DISTURBANCE_SCALE,
        ),
    )
    legacy_projection = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_legacy_projected_oracle_nominal_replay",
            row_family="legacy_projected_oracle_replay",
            rank=rank,
            training_distribution="nominal",
            evaluation_lens="nominal_clean",
        )
        for rank in LEGACY_AUDIT_RANKS
    )
    projected_nominal = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_projected_oracle_nominal_replay",
            row_family="projected_oracle_replay",
            rank=rank,
            training_distribution="nominal",
            evaluation_lens="nominal_clean",
        )
        for rank in PROJECTION_SWEEP_RANKS
    )
    projected_process = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_projected_oracle_process_io_eval",
            row_family="projected_oracle_io_eval",
            rank=rank,
            training_distribution="process_io_probe",
            evaluation_lens="process_io",
            disturbance_scale=DEFAULT_PROCESS_IO_DISTURBANCE_SCALE,
        )
        for rank in PROJECTION_SWEEP_RANKS
    )
    projected_process_measurement = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_projected_oracle_process_measurement_io_eval",
            row_family="projected_oracle_io_eval",
            rank=rank,
            training_distribution="process_measurement_io_probe",
            evaluation_lens="process_measurement_io",
            disturbance_scale=DEFAULT_PROCESS_IO_DISTURBANCE_SCALE,
            measurement_scale=DEFAULT_MEASUREMENT_IO_DISTURBANCE_SCALE,
        )
        for rank in PROJECTION_SWEEP_RANKS
    )
    state_coverage = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_projected_oracle_state_coverage_eigen_m4_s0p3_eval",
            row_family="projected_oracle_state_coverage_eval",
            rank=rank,
            training_distribution="state_coverage_eigen",
            evaluation_lens="state_coverage_eigen_m4_s0.3",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=0.3,
        )
        for rank in (LEGACY_AUDIT_RANKS + (12,))
    )
    reward = (
        PhaseModulatedCondition(
            label="pm_linrec_r12_clean_scratch_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="nominal",
            evaluation_lens="nominal_clean",
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_state_coverage_eigen_m1_s0p3_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="state_coverage_eigen",
            evaluation_lens="state_coverage_eigen_m1_s0.3",
            coverage_family="state_eigenspectrum",
            coverage_modes=1,
            coverage_scale=0.3,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_state_coverage_eigen_m4_s0p3_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="state_coverage_eigen",
            evaluation_lens="state_coverage_eigen_m4_s0.3",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=0.3,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_state_coverage_eigen_m4_s1_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="state_coverage_eigen",
            evaluation_lens="state_coverage_eigen_m4_s1",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=1.0,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_observer_error_svd_m1_s0p3_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="observer_error",
            evaluation_lens="observer_error_svd_m1_s0.3",
            coverage_family="observer_error_state",
            coverage_modes=1,
            coverage_scale=0.3,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_mixed_process_observer_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="mixed",
            evaluation_lens="mixed_process_observer",
            coverage_family="mixed_deviation",
            coverage_modes=4,
            coverage_scale=0.3,
            disturbance_scale=0.02,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_projected_oracle_nominal_then_reward",
            row_family="projection_warm_start_then_reward_lens",
            rank=12,
            training_distribution="nominal",
            evaluation_lens="nominal_clean",
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_projected_oracle_state_coverage_eigen_m4_then_reward",
            row_family="projection_warm_start_then_reward_lens",
            rank=12,
            training_distribution="state_coverage_eigen",
            evaluation_lens="state_coverage_eigen_m4_s0.3",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=0.3,
        ),
    )
    return (
        exact_oracle
        + legacy_projection
        + projected_nominal
        + projected_process
        + projected_process_measurement
        + state_coverage
        + (reward if include_reward else ())
    )


def build_oracle_recurrent_reference(
    *,
    plant: Any,
    gains: np.ndarray,
    output_config: OutputFeedbackConfig,
    initial_hidden: np.ndarray,
) -> OracleRecurrentReference:
    """Construct the time-varying Kalman recurrence that realizes the reference."""

    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    K = np.asarray(gains, dtype=np.float64)
    L = np.asarray(kalman_estimator_gains(plant, jnp.asarray(K), output_config), dtype=np.float64)
    horizon = K.shape[0]
    hidden_dim = A.shape[0]
    action_dim = B.shape[1]
    recurrent = np.empty((horizon, hidden_dim, hidden_dim), dtype=np.float64)
    observation = np.empty((horizon, hidden_dim, H.shape[0]), dtype=np.float64)
    readout = np.empty((horizon, action_dim, hidden_dim), dtype=np.float64)
    for t in range(horizon):
        recurrent[t] = A - B @ K[t] - L[t] @ H
        observation[t] = L[t]
        readout[t] = -K[t]
    return OracleRecurrentReference(
        basis_state="kalman_estimate_hidden_state",
        observation_matrix=H,
        recurrent_matrices=recurrent,
        observation_matrices=observation,
        previous_action_matrices=np.zeros((horizon, hidden_dim, action_dim), dtype=np.float64),
        hidden_biases=np.zeros((horizon, hidden_dim), dtype=np.float64),
        readout_matrices=readout,
        feedthrough_matrices=np.zeros((horizon, action_dim, H.shape[0]), dtype=np.float64),
        action_biases=np.zeros((horizon, action_dim), dtype=np.float64),
        initial_hidden=np.asarray(initial_hidden, dtype=np.float64),
    )


def phase_basis(*, horizon: int, rank: int) -> np.ndarray:
    """Return the d6d25d6 clamped spline phase basis for one rank."""

    return clamped_bspline_time_basis(horizon=horizon, n_basis=rank, degree=min(3, rank - 1))


def project_oracle_reference(
    reference: OracleRecurrentReference,
    *,
    basis: np.ndarray,
) -> tuple[PhaseModulatedLinearRecurrentController, dict[str, MatrixBasisProjection]]:
    """Project all oracle recurrent matrices onto one phase basis."""

    recurrent = project_matrix_sequence_to_basis(reference.recurrent_matrices, basis)
    observation = project_matrix_sequence_to_basis(reference.observation_matrices, basis)
    previous_action = project_matrix_sequence_to_basis(reference.previous_action_matrices, basis)
    hidden_bias = project_matrix_sequence_to_basis(reference.hidden_biases[:, :, None], basis)
    readout = project_matrix_sequence_to_basis(reference.readout_matrices, basis)
    feedthrough = project_matrix_sequence_to_basis(reference.feedthrough_matrices, basis)
    action_bias = project_matrix_sequence_to_basis(reference.action_biases[:, :, None], basis)
    controller = PhaseModulatedLinearRecurrentController(
        basis=basis,
        recurrent_coefficients=recurrent.theta,
        observation_coefficients=observation.theta,
        previous_action_coefficients=previous_action.theta,
        hidden_bias_coefficients=hidden_bias.theta[:, :, 0],
        readout_coefficients=readout.theta,
        feedthrough_coefficients=feedthrough.theta,
        action_bias_coefficients=action_bias.theta[:, :, 0],
        initial_hidden=reference.initial_hidden,
    )
    return controller, {
        "A_h": recurrent,
        "B_y": observation,
        "B_u": previous_action,
        "b_h": hidden_bias,
        "C_h": readout,
        "D_y": feedthrough,
        "c": action_bias,
    }


def exact_oracle_controller(
    reference: OracleRecurrentReference,
) -> PhaseModulatedLinearRecurrentController:
    """Represent the finite-horizon oracle recurrence without spline projection."""

    horizon = reference.recurrent_matrices.shape[0]
    return PhaseModulatedLinearRecurrentController(
        basis=np.eye(horizon, dtype=np.float64),
        recurrent_coefficients=reference.recurrent_matrices,
        observation_coefficients=reference.observation_matrices,
        previous_action_coefficients=reference.previous_action_matrices,
        hidden_bias_coefficients=reference.hidden_biases,
        readout_coefficients=reference.readout_matrices,
        feedthrough_coefficients=reference.feedthrough_matrices,
        action_bias_coefficients=reference.action_biases,
        initial_hidden=reference.initial_hidden,
    )


def materialize(
    *,
    include_reward: bool = True,
    conditions: tuple[PhaseModulatedCondition, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Materialize phase-modulated recurrent rows."""

    start = time.perf_counter()
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    plant = reference.plant
    schedule = reference.schedule
    output_config = OutputFeedbackConfig()
    gains = np.asarray(reference.lqr_solution.K, dtype=np.float64)
    x0 = np.asarray(make_cs_output_feedback_initial_state(plant, output_config), dtype=np.float64)
    clean_reference = rollout_with_kalman_estimator(plant, jnp.asarray(gains), jnp.asarray(x0))
    clean_reference_cost = float(
        output_feedback_cost(schedule, clean_reference).total_without_disturbance_penalty
    )
    oracle = build_oracle_recurrent_reference(
        plant=plant,
        gains=gains,
        output_config=output_config,
        initial_hidden=np.asarray(clean_reference.x_hat[0], dtype=np.float64),
    )
    exact_controller = exact_oracle_controller(oracle)
    retained = conditions or default_conditions(include_reward=include_reward)
    projected_ranks = sorted(
        {condition.rank for condition in retained if condition.row_family != "exact_oracle_sanity"}
    )
    bases = {rank: phase_basis(horizon=gains.shape[0], rank=rank) for rank in projected_ranks}
    projected = {
        rank: project_oracle_reference(oracle, basis=bases[rank]) for rank in projected_ranks
    }
    exact_projections = project_oracle_reference(
        oracle,
        basis=np.eye(gains.shape[0], dtype=np.float64),
    )[1]
    arrays: dict[str, np.ndarray] = {
        "reference_clean_x": np.asarray(clean_reference.x),
        "reference_clean_x_hat": np.asarray(clean_reference.x_hat),
        "reference_clean_u": np.asarray(clean_reference.u),
        "exact_time_identity_basis": np.eye(gains.shape[0], dtype=np.float64),
        "oracle_A_h": oracle.recurrent_matrices,
        "oracle_B_y": oracle.observation_matrices,
        "oracle_C_h": oracle.readout_matrices,
    }
    for rank, basis in bases.items():
        arrays[f"clamped_bspline_r{rank}_basis"] = basis

    rows: list[BridgeRunManifest] = []
    for condition in retained:
        training = _training_batch_for_condition(
            condition,
            plant=plant,
            gains=gains,
            x0=x0,
            observation_dim=oracle.observation_matrix.shape[0],
            output_config=output_config,
        )
        if condition.row_family == "exact_oracle_sanity":
            projected_controller, projections = exact_controller, exact_projections
        else:
            projected_controller, projections = projected[condition.rank]
        controller, fit_metadata = _controller_for_condition(
            condition,
            exact_controller=exact_controller,
            projected_controller=projected_controller,
            plant=plant,
            schedule=schedule,
            training=training,
            observation_matrix=oracle.observation_matrix,
        )
        rollout = _rollout_phase_modulated_recurrent_condition(
            controller,
            plant,
            training["x0"],
            observation_matrix=oracle.observation_matrix,
            disturbances=training["disturbances"],
            measurement_disturbances=training["measurement_disturbances"],
            initial_hidden=training["xhat0"],
        )
        reference_batch = _reference_output_feedback_batch(
            plant=plant,
            gains=gains,
            x0=training["x0"],
            xhat0=training["xhat0"],
            output_config=output_config,
            disturbances=training["disturbances"],
            measurement_disturbances=training["measurement_disturbances"],
        )
        row = _manifest_for_condition(
            condition=condition,
            rollout=rollout,
            reference_batch=reference_batch,
            schedule=schedule,
            clean_reference_cost=clean_reference_cost,
            projections=projections,
            fit_metadata=fit_metadata,
            response_maps=_response_maps_for_controller_pair(
                candidate=controller,
                reference=exact_controller,
                plant=plant,
                observation_matrix=oracle.observation_matrix,
            ),
        )
        rows.append(row)
        prefix = condition.run_id
        arrays[f"{prefix}__plant_states"] = np.asarray(rollout.plant_states)
        arrays[f"{prefix}__hidden_states"] = np.asarray(rollout.hidden_states)
        arrays[f"{prefix}__observations"] = np.asarray(rollout.observations)
        arrays[f"{prefix}__actions"] = np.asarray(rollout.actions)
        arrays[f"{prefix}__reference_actions"] = reference_batch["u"]
        arrays[f"{prefix}__reference_observations"] = reference_batch["y"]

    component_counts: Counter[str] = Counter()
    for row in rows:
        for component in row.certificate_components:
            component_counts[f"{component.name}:{component.status}"] += 1
    summary = {
        "format": "rlrmp.output_feedback_phase_modulated_recurrent.v2",
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "source_issues": {
            "additive_phase_recurrent": RELATED_RECURRENT_ISSUE_ID,
            "io_map_certificate": IO_MAP_CERTIFICATE_ISSUE_ID,
        },
        "scope": (
            "Oracle Kalman recurrent reference plus clamped-spline phase-modulated "
            "linear recurrence. The spline basis modulates A/B/C/D matrices over "
            "tau=t/(T-1), not additive phase offsets."
        ),
        "non_goals": (
            "No GRU training, no supervised imitation optimization rows, no broad "
            "robust-epsilon arm, and no claim that projected-oracle diagnostic rows "
            "are bridge passes."
        ),
        "runtime_seconds": time.perf_counter() - start,
        "diagnostics": {
            "basis_family": "clamped_b_spline_partition_of_unity",
            "basis_ranks": projected_ranks,
            "basis_degrees": {str(rank): min(3, rank - 1) for rank in projected_ranks},
            "component_status_counts": dict(sorted(component_counts.items())),
            "retained_rows": [row.spec.run_id for row in rows],
            "audit": {
                "exact_process_eigen_label_disposition": (
                    "The prior exact_process_eigen rows were state-trajectory covariance "
                    "coverage directions, not process-eigen disturbance sequences. They are "
                    "retained under state_coverage_eigen labels."
                ),
                "projection_sweep_ranks": list(PROJECTION_SWEEP_RANKS),
                "exact_oracle_sanity_rows": [
                    row.spec.run_id
                    for row in rows
                    if row.metrics["row_family"] == "exact_oracle_sanity"
                ],
                "supervised_optimization_rows": 0,
            },
        },
        "rows": [row.to_json_dict() for row in rows],
        "result": _result_text(rows),
    }
    return summary, arrays


def write_outputs(
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    note_path: Path = NOTE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    artifact_path: Path = ARTIFACT_PATH,
) -> None:
    """Write tracked note/manifest and ignored bulk arrays."""

    mkdir_p(note_path.parent)
    mkdir_p(manifest_path.parent)
    mkdir_p(artifact_path.parent)
    results_dir = mkdir_p(REPO_ROOT / "results" / ISSUE_ID)
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Phase-modulated linear recurrent output-feedback bridge. See "
            "`notes/phase_modulated_recurrent.md`.\n",
            encoding="utf-8",
        )
    np.savez_compressed(artifact_path, **arrays)
    summary["tracked_note"] = _repo_relative(note_path)
    summary["tracked_manifest"] = _repo_relative(manifest_path)
    summary["artifact_npz"] = _repo_relative(artifact_path)
    summary["artifact_npz_keys"] = sorted(arrays)
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked result note."""

    table = [
        (
            "| row | family | lens | verdict | objective ratio | action mismatch | "
            "matrix residual | obs I/O | proc I/O | io-map |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        metrics = row["metrics"]
        response = metrics["response_map_mismatch"]
        table.append(
            "| "
            f"{row['spec']['run_id']} | "
            f"{metrics['row_family']} | "
            f"{metrics['evaluation_lens']} | "
            f"{metrics['verdict']} | "
            f"{metrics['objective_ratio_to_clean_reference']:.8g} | "
            f"{metrics['state_weighted_action_mismatch']:.8g} | "
            f"{metrics['projection']['combined_relative_residual']:.8g} | "
            f"{response['observation_to_action']:.8g} | "
            f"{response['disturbance_to_action']:.8g} | "
            f"{metrics['io_map_certificate']['status']} |"
        )
    component_rows = [
        f"- `{key}`: {value}"
        for key, value in summary["diagnostics"]["component_status_counts"].items()
    ]
    return f"""# Phase-Modulated Linear Recurrent Output-Feedback Bridge

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.

Scope: {summary["scope"]}

Non-goals: {summary["non_goals"]}

Runtime: `{summary.get("runtime_seconds", 0.0):.2f}` seconds.

Verdict: {summary["result"]}

Audit note: {summary["diagnostics"]["audit"]["exact_process_eigen_label_disposition"]}

## Rows

{"\n".join(table)}

## Certificate Boundary

These rows are diagnostics, not bridge passes. The recurrent controller is an
augmented-linear system over `z_t = [x_t; h_t]`, and action/visited-state
components are reported through the existing augmented recurrent adapter. The
formal I/O-map certificate is consumed through the standard certificate
component builder; projected-oracle rows remain diagnostic even when those
components are available. No supervised imitation rows are materialized here.

{"\n".join(component_rows)}

## Interpretation

The exact-oracle rows replay the finite-horizon Kalman recurrent reference under
nominal, process I/O, and process+measurement I/O probes. Ranked rows then
project each time-varying recurrent/readout matrix onto a clamped B-spline
partition of unity. State-coverage eigen rows preserve the old coverage
semantics explicitly: they perturb initial state/estimator coverage directions
from a state-trajectory covariance, not process-eigen disturbances. Reward rows
optimize trainable phase-modulated recurrent coefficients against the true
quadratic rollout objective on their retained training distributions; projection
warm-start rows use the projected oracle controller before reward fine-tuning.
"""


def _controller_for_condition(
    condition: PhaseModulatedCondition,
    *,
    exact_controller: PhaseModulatedLinearRecurrentController,
    projected_controller: PhaseModulatedLinearRecurrentController,
    plant: Any,
    schedule: Any,
    training: dict[str, np.ndarray],
    observation_matrix: np.ndarray,
) -> tuple[PhaseModulatedLinearRecurrentController, dict[str, Any]]:
    if condition.row_family == "exact_oracle_sanity":
        return exact_controller, {
            "fit_method": "exact_oracle_replay",
            "is_reward_trained": False,
            "initialization": "analytical_lqr_kalman_oracle_recurrence",
        }

    projected_diagnostics = {
        "legacy_projected_oracle_replay",
        "projected_oracle_replay",
        "projected_oracle_io_eval",
        "projected_oracle_state_coverage_eval",
    }
    if condition.row_family in projected_diagnostics:
        return projected_controller, {
            "fit_method": "oracle_projection_clamped_bspline",
            "is_reward_trained": False,
            "initialization": "oracle_matrix_projection",
        }

    if condition.row_family == "projection_warm_start_then_reward_lens":
        initial = _params_from_controller(projected_controller)
        initialization = "oracle_matrix_projection_warm_start"
    else:
        initial = _scratch_params_like_controller(projected_controller, seed=condition.seed)
        initialization = "scratch_random_stable"

    fitted, fit_metadata = _fit_phase_modulated_reward_params(
        initial,
        condition=condition,
        plant=plant,
        schedule=schedule,
        training=training,
        observation_matrix=observation_matrix,
        basis=projected_controller.basis,
    )
    controller = _controller_from_params(
        fitted,
        basis=projected_controller.basis,
        initial_hidden=projected_controller.initial_hidden,
    )
    fit_metadata.update(
        {
            "fit_method": "adam_phase_modulated_reward_rollout",
            "is_reward_trained": True,
            "initialization": initialization,
            "n_train_steps": condition.n_train_steps,
            "learning_rate": condition.learning_rate,
            "stability_penalty": condition.stability_penalty,
        }
    )
    return controller, fit_metadata


def _params_from_controller(
    controller: PhaseModulatedLinearRecurrentController,
) -> dict[str, np.ndarray]:
    return {
        "A_h": np.asarray(controller.recurrent_coefficients, dtype=np.float64),
        "B_y": np.asarray(controller.observation_coefficients, dtype=np.float64),
        "B_u": np.asarray(controller.previous_action_coefficients, dtype=np.float64),
        "b_h": np.asarray(controller.hidden_bias_coefficients, dtype=np.float64),
        "C_h": np.asarray(controller.readout_coefficients, dtype=np.float64),
        "D_y": np.asarray(controller.feedthrough_coefficients, dtype=np.float64),
        "c": np.asarray(controller.action_bias_coefficients, dtype=np.float64),
    }


def _scratch_params_like_controller(
    controller: PhaseModulatedLinearRecurrentController,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_basis = controller.basis.shape[1]
    hidden_dim = controller.hidden_dim
    observation_dim = controller.observation_dim
    action_dim = controller.action_dim
    recurrent = np.broadcast_to(
        0.72 * np.eye(hidden_dim, dtype=np.float64),
        (n_basis, hidden_dim, hidden_dim),
    ).copy()
    recurrent += 0.01 * rng.normal(size=recurrent.shape)
    return {
        "A_h": recurrent,
        "B_y": 0.03 * rng.normal(size=(n_basis, hidden_dim, observation_dim)),
        "B_u": 0.02 * rng.normal(size=(n_basis, hidden_dim, action_dim)),
        "b_h": np.zeros((n_basis, hidden_dim), dtype=np.float64),
        "C_h": 0.03 * rng.normal(size=(n_basis, action_dim, hidden_dim)),
        "D_y": 0.01 * rng.normal(size=(n_basis, action_dim, observation_dim)),
        "c": np.zeros((n_basis, action_dim), dtype=np.float64),
    }


def _controller_from_params(
    params: dict[str, np.ndarray],
    *,
    basis: np.ndarray,
    initial_hidden: np.ndarray | None,
) -> PhaseModulatedLinearRecurrentController:
    return PhaseModulatedLinearRecurrentController(
        basis=basis,
        recurrent_coefficients=params["A_h"],
        observation_coefficients=params["B_y"],
        previous_action_coefficients=params["B_u"],
        hidden_bias_coefficients=params["b_h"],
        readout_coefficients=params["C_h"],
        feedthrough_coefficients=params["D_y"],
        action_bias_coefficients=params["c"],
        initial_hidden=initial_hidden,
    )


def _fit_phase_modulated_reward_params(
    params: dict[str, np.ndarray],
    *,
    condition: PhaseModulatedCondition,
    plant: Any,
    schedule: Any,
    training: dict[str, np.ndarray],
    observation_matrix: np.ndarray,
    basis: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    constants = {
        "A": jnp.asarray(plant.A, dtype=jnp.float64),
        "B": jnp.asarray(plant.B, dtype=jnp.float64),
        "Bw": jnp.asarray(plant.Bw, dtype=jnp.float64),
        "H": jnp.asarray(observation_matrix, dtype=jnp.float64),
        "basis": jnp.asarray(basis, dtype=jnp.float64),
        "x0": jnp.asarray(training["x0"], dtype=jnp.float64),
        "xhat0": jnp.asarray(training["xhat0"], dtype=jnp.float64),
        "disturbances": jnp.asarray(training["disturbances"], dtype=jnp.float64),
        "measurement_disturbances": jnp.asarray(
            training["measurement_disturbances"],
            dtype=jnp.float64,
        ),
        "Q": jnp.asarray(schedule.Q, dtype=jnp.float64),
        "R": jnp.asarray(schedule.R, dtype=jnp.float64),
        "Q_f": jnp.asarray(schedule.Q_f, dtype=jnp.float64),
    }

    def loss_fn(jax_params: dict[str, jax.Array]) -> jax.Array:
        rollout = _jax_phase_modulated_rollout(jax_params, constants)
        cost = _jax_quadratic_cost(
            states=rollout["states"],
            actions=rollout["actions"],
            q=constants["Q"],
            r=constants["R"],
            q_f=constants["Q_f"],
        )
        return cost + _jax_phase_stability_penalty(
            jax_params,
            basis=constants["basis"],
            scale=condition.stability_penalty,
        )

    fitted, history = _adam_minimize(
        params,
        loss_fn,
        n_steps=condition.n_train_steps,
        learning_rate=condition.learning_rate,
    )
    return fitted, {
        "reward_initial_loss": history["initial_loss"],
        "reward_final_loss": history["final_loss"],
        "reward_last_loss": history["last_loss"],
        "reward_best_loss": history["best_loss"],
        "reward_loss_improvement": history["initial_loss"] - history["final_loss"],
    }


def _jax_phase_modulated_rollout(
    params: dict[str, jax.Array],
    constants: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    x = constants["x0"]
    hidden = constants["xhat0"]
    previous_action = jnp.zeros((x.shape[0], params["C_h"].shape[1]), dtype=x.dtype)
    states = [x]
    hidden_states = [hidden]
    actions = []
    for t in range(constants["basis"].shape[0]):
        weights = constants["basis"][t]
        a_h = jnp.einsum("b,bij->ij", weights, params["A_h"])
        b_y = jnp.einsum("b,bij->ij", weights, params["B_y"])
        b_u = jnp.einsum("b,bij->ij", weights, params["B_u"])
        b_h = jnp.einsum("b,bi->i", weights, params["b_h"])
        c_h = jnp.einsum("b,bij->ij", weights, params["C_h"])
        d_y = jnp.einsum("b,bij->ij", weights, params["D_y"])
        c = jnp.einsum("b,bi->i", weights, params["c"])
        y_t = x @ constants["H"].T + constants["measurement_disturbances"][:, t, :]
        u_t = hidden @ c_h.T + y_t @ d_y.T + c
        x = (
            x @ constants["A"].T
            + u_t @ constants["B"].T
            + constants["disturbances"][:, t, :] @ constants["Bw"].T
        )
        hidden = hidden @ a_h.T + y_t @ b_y.T + previous_action @ b_u.T + b_h
        previous_action = u_t
        states.append(x)
        hidden_states.append(hidden)
        actions.append(u_t)
    return {
        "states": jnp.stack(states, axis=1),
        "hidden_states": jnp.stack(hidden_states, axis=1),
        "actions": jnp.stack(actions, axis=1),
    }


def _jax_quadratic_cost(
    *,
    states: jax.Array,
    actions: jax.Array,
    q: jax.Array,
    r: jax.Array,
    q_f: jax.Array,
) -> jax.Array:
    state_terms = jnp.einsum("bti,tij,btj->bt", states[:, :-1, :], q, states[:, :-1, :])
    control_terms = jnp.einsum("bti,tij,btj->bt", actions, r, actions)
    terminal = jnp.einsum("bi,ij,bj->b", states[:, -1, :], q_f, states[:, -1, :])
    return jnp.mean(jnp.sum(state_terms + control_terms, axis=1) + terminal)


def _jax_phase_stability_penalty(
    params: dict[str, jax.Array],
    *,
    basis: jax.Array,
    scale: float,
) -> jax.Array:
    if scale <= 0.0:
        return jnp.asarray(0.0)
    recurrent = jnp.einsum("tb,bij->tij", basis, params["A_h"])
    row_norms = jnp.linalg.norm(recurrent, axis=2)
    return scale * jnp.mean(jnp.square(jnp.maximum(row_norms - 0.98, 0.0)))


def _adam_minimize(
    params: dict[str, np.ndarray],
    loss_fn: Any,
    *,
    n_steps: int,
    learning_rate: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    jax_params = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in params.items()}
    if n_steps <= 0:
        loss = float(loss_fn(jax_params))
        return params, {
            "initial_loss": loss,
            "final_loss": loss,
            "last_loss": loss,
            "best_loss": loss,
        }
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    initial_loss = float(jax.jit(loss_fn)(jax_params))
    best_loss = initial_loss
    best_params = jax_params
    m = jax.tree.map(jnp.zeros_like, jax_params)
    v = jax.tree.map(jnp.zeros_like, jax_params)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    loss = initial_loss
    for step in range(1, n_steps + 1):
        loss_value, grads = value_and_grad(jax_params)
        loss = float(loss_value)
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_params = jax_params
        m = jax.tree.map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, m, grads)
        v = jax.tree.map(lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i * g_i), v, grads)
        m_hat = jax.tree.map(lambda m_i: m_i / (1.0 - beta1**step), m)
        v_hat = jax.tree.map(lambda v_i: v_i / (1.0 - beta2**step), v)
        jax_params = jax.tree.map(
            lambda p_i, m_i, v_i: p_i - learning_rate * m_i / (jnp.sqrt(v_i) + eps),
            jax_params,
            m_hat,
            v_hat,
        )
    return (
        {key: np.asarray(value, dtype=np.float64) for key, value in best_params.items()},
        {
            "initial_loss": initial_loss,
            "final_loss": best_loss,
            "last_loss": loss,
            "best_loss": best_loss,
        },
    )


def _rollout_phase_modulated_recurrent_condition(
    controller: PhaseModulatedLinearRecurrentController,
    plant: Any,
    x0: np.ndarray,
    *,
    observation_matrix: np.ndarray,
    disturbances: np.ndarray,
    measurement_disturbances: np.ndarray,
    initial_hidden: np.ndarray,
) -> BridgeRolloutBatch:
    """Roll a condition, including explicit measurement-noise probes."""

    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    Bw = np.asarray(plant.Bw, dtype=np.float64)
    H = np.asarray(observation_matrix, dtype=np.float64)
    states = np.asarray(x0, dtype=np.float64)
    hidden = np.asarray(initial_hidden, dtype=np.float64)
    disturbances_array = np.asarray(disturbances, dtype=np.float64)
    measurement_array = np.asarray(measurement_disturbances, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    if hidden.ndim == 1:
        hidden = hidden[None, :]
    batch_size = states.shape[0]
    if hidden.shape[0] == 1 and batch_size != 1:
        hidden = np.broadcast_to(hidden, (batch_size, hidden.shape[1])).copy()
    if disturbances_array.shape != (batch_size, controller.horizon, Bw.shape[1]):
        raise ValueError("disturbances must have shape (batch, horizon, disturbance)")
    if measurement_array.shape != (batch_size, controller.horizon, H.shape[0]):
        raise ValueError("measurement_disturbances must have shape (batch, horizon, observation)")

    previous_action = np.zeros((batch_size, controller.action_dim), dtype=np.float64)
    plant_states = [states]
    hidden_states = [hidden]
    observations = []
    actions = []
    for t in range(controller.horizon):
        y_t = states @ H.T + measurement_array[:, t, :]
        u_t = controller.action(t, hidden, y_t)
        states = states @ A.T + u_t @ B.T + disturbances_array[:, t, :] @ Bw.T
        hidden = controller.next_hidden(t, hidden, y_t, previous_action)
        previous_action = u_t
        observations.append(y_t)
        actions.append(u_t)
        plant_states.append(states)
        hidden_states.append(hidden)

    hidden_array = np.stack(hidden_states, axis=1)
    diagnostics = controller.stability_diagnostics() | hidden_growth_diagnostics(hidden_array)
    diagnostics["phase_modulates_matrices"] = True
    diagnostics["measurement_disturbance_scale"] = float(np.max(np.abs(measurement_array)))
    diagnostics["process_disturbance_scale"] = float(np.max(np.abs(disturbances_array)))
    return BridgeRolloutBatch(
        plant_states=np.stack(plant_states, axis=1),
        actions=np.stack(actions, axis=1),
        observations=np.stack(observations, axis=1),
        hidden_states=hidden_array,
        metadata={"controller": "phase_modulated_linear_recurrence", "diagnostics": diagnostics},
    )


def _response_maps_for_controller_pair(
    *,
    candidate: PhaseModulatedLinearRecurrentController,
    reference: PhaseModulatedLinearRecurrentController,
    plant: Any,
    observation_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "candidate_observation_to_action": _observation_to_action_response_map(candidate),
        "reference_observation_to_action": _observation_to_action_response_map(reference),
        "candidate_disturbance_to_action": _disturbance_to_action_response_map(
            candidate,
            plant=plant,
            observation_matrix=observation_matrix,
        ),
        "reference_disturbance_to_action": _disturbance_to_action_response_map(
            reference,
            plant=plant,
            observation_matrix=observation_matrix,
        ),
        "candidate_disturbance_to_state": _disturbance_to_state_response_map(
            candidate,
            plant=plant,
            observation_matrix=observation_matrix,
        ),
        "reference_disturbance_to_state": _disturbance_to_state_response_map(
            reference,
            plant=plant,
            observation_matrix=observation_matrix,
        ),
    }


def _observation_to_action_response_map(
    controller: PhaseModulatedLinearRecurrentController,
) -> np.ndarray:
    horizon = controller.horizon
    obs_dim = controller.observation_dim
    action_dim = controller.action_dim
    hidden_dim = controller.hidden_dim
    input_dim = horizon * obs_dim
    response = np.zeros((horizon, action_dim, input_dim), dtype=np.float64)
    hidden_response = np.zeros((hidden_dim, input_dim), dtype=np.float64)
    previous_action_response = np.zeros((action_dim, input_dim), dtype=np.float64)
    for t in range(horizon):
        matrices = controller.matrices_at(t)
        observation_response = np.zeros((obs_dim, input_dim), dtype=np.float64)
        observation_response[:, t * obs_dim : (t + 1) * obs_dim] = np.eye(obs_dim)
        action_response = matrices["C_h"] @ hidden_response + matrices["D_y"] @ observation_response
        response[t] = action_response
        hidden_response = (
            matrices["A_h"] @ hidden_response
            + matrices["B_y"] @ observation_response
            + matrices["B_u"] @ previous_action_response
        )
        previous_action_response = action_response
    return response


def _disturbance_to_action_response_map(
    controller: PhaseModulatedLinearRecurrentController,
    *,
    plant: Any,
    observation_matrix: np.ndarray,
) -> np.ndarray:
    horizon = controller.horizon
    state_dim = int(plant.A.shape[0])
    disturbance_dim = int(plant.Bw.shape[1])
    action_dim = controller.action_dim
    input_dim = horizon * disturbance_dim
    response = np.zeros((horizon, action_dim, input_dim), dtype=np.float64)
    state_response = np.zeros((state_dim, input_dim), dtype=np.float64)
    hidden_response = np.zeros((controller.hidden_dim, input_dim), dtype=np.float64)
    previous_action_response = np.zeros((action_dim, input_dim), dtype=np.float64)
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    Bw = np.asarray(plant.Bw, dtype=np.float64)
    H = np.asarray(observation_matrix, dtype=np.float64)
    for t in range(horizon):
        matrices = controller.matrices_at(t)
        observation_response = H @ state_response
        action_response = matrices["C_h"] @ hidden_response + matrices["D_y"] @ observation_response
        response[t] = action_response
        disturbance_injection = np.zeros((disturbance_dim, input_dim), dtype=np.float64)
        disturbance_injection[:, t * disturbance_dim : (t + 1) * disturbance_dim] = np.eye(
            disturbance_dim
        )
        state_response = A @ state_response + B @ action_response + Bw @ disturbance_injection
        hidden_response = (
            matrices["A_h"] @ hidden_response
            + matrices["B_y"] @ observation_response
            + matrices["B_u"] @ previous_action_response
        )
        previous_action_response = action_response
    return response


def _disturbance_to_state_response_map(
    controller: PhaseModulatedLinearRecurrentController,
    *,
    plant: Any,
    observation_matrix: np.ndarray,
) -> np.ndarray:
    horizon = controller.horizon
    state_dim = int(plant.A.shape[0])
    disturbance_dim = int(plant.Bw.shape[1])
    input_dim = horizon * disturbance_dim
    response = np.zeros((horizon + 1, state_dim, input_dim), dtype=np.float64)
    state_response = np.zeros((state_dim, input_dim), dtype=np.float64)
    hidden_response = np.zeros((controller.hidden_dim, input_dim), dtype=np.float64)
    previous_action_response = np.zeros((controller.action_dim, input_dim), dtype=np.float64)
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    Bw = np.asarray(plant.Bw, dtype=np.float64)
    H = np.asarray(observation_matrix, dtype=np.float64)
    for t in range(horizon):
        matrices = controller.matrices_at(t)
        observation_response = H @ state_response
        action_response = matrices["C_h"] @ hidden_response + matrices["D_y"] @ observation_response
        disturbance_injection = np.zeros((disturbance_dim, input_dim), dtype=np.float64)
        disturbance_injection[:, t * disturbance_dim : (t + 1) * disturbance_dim] = np.eye(
            disturbance_dim
        )
        state_response = A @ state_response + B @ action_response + Bw @ disturbance_injection
        hidden_response = (
            matrices["A_h"] @ hidden_response
            + matrices["B_y"] @ observation_response
            + matrices["B_u"] @ previous_action_response
        )
        previous_action_response = action_response
        response[t + 1] = state_response
    return response


def _manifest_for_condition(
    *,
    condition: PhaseModulatedCondition,
    rollout: Any,
    reference_batch: dict[str, np.ndarray],
    schedule: Any,
    clean_reference_cost: float,
    projections: dict[str, MatrixBasisProjection],
    fit_metadata: dict[str, Any],
    response_maps: dict[str, np.ndarray],
) -> BridgeRunManifest:
    candidate_actions = np.asarray(rollout.actions)
    reference_actions = reference_batch["u"]
    candidate_cost = _batch_quadratic_cost(schedule, rollout.plant_states, candidate_actions)
    reference_cost = _batch_quadratic_cost(schedule, reference_batch["x"], reference_actions)
    augmented_states = np.concatenate([rollout.plant_states, rollout.hidden_states], axis=-1)
    components = build_standard_certificate_components(
        architecture="linear_recurrence",
        certificate_mode="augmented_linear",
        augmented_states=augmented_states,
        candidate_actions=candidate_actions,
        reference_actions=reference_actions,
        recurrence_diagnostics=rollout.metadata["diagnostics"],
        candidate_observation_to_action_map=response_maps["candidate_observation_to_action"],
        reference_observation_to_action_map=response_maps["reference_observation_to_action"],
        observation_history_covariance=np.eye(
            response_maps["candidate_observation_to_action"].shape[-1],
            dtype=np.float64,
        ),
        candidate_disturbance_to_action_map=response_maps["candidate_disturbance_to_action"],
        reference_disturbance_to_action_map=response_maps["reference_disturbance_to_action"],
        candidate_disturbance_to_state_map=response_maps["candidate_disturbance_to_state"],
        reference_disturbance_to_state_map=response_maps["reference_disturbance_to_state"],
        disturbance_history_covariance=np.eye(
            response_maps["candidate_disturbance_to_action"].shape[-1],
            dtype=np.float64,
        ),
        optimizer_metadata={
            **fit_metadata,
            "rank": condition.rank,
            "row_family": condition.row_family,
        },
        state_label="plant_hidden_augmented_state",
        action_label="control",
    )
    by_name = {component.name: component for component in components}
    action_summary = by_name[STATE_WEIGHTED_ACTION_MISMATCH].summary
    observation_response_summary = by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH].summary
    disturbance_action_summary = by_name[DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH].summary
    disturbance_state_summary = by_name[DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH].summary
    projection_summary = _projection_summary(projections)
    metrics = {
        "row_family": condition.row_family,
        "evaluation_lens": condition.evaluation_lens,
        "candidate_cost": candidate_cost,
        "reference_lens_cost": reference_cost,
        "clean_reference_cost": clean_reference_cost,
        "objective_ratio_to_clean_reference": candidate_cost / max(clean_reference_cost, 1e-12),
        "objective_ratio_to_lens_reference": candidate_cost / max(reference_cost, 1e-12),
        "state_weighted_action_mismatch": action_summary["mismatch_ratio_mean"],
        "aggregate_action_energy_mismatch": action_summary["aggregate_mismatch_ratio"],
        "response_map_mismatch": {
            "observation_to_action": observation_response_summary["aggregate_mismatch_ratio"],
            "disturbance_to_action": disturbance_action_summary["aggregate_mismatch_ratio"],
            "disturbance_to_state": disturbance_state_summary["aggregate_mismatch_ratio"],
            "max_aggregate_mismatch": max(
                observation_response_summary["aggregate_mismatch_ratio"],
                disturbance_action_summary["aggregate_mismatch_ratio"],
                disturbance_state_summary["aggregate_mismatch_ratio"],
            ),
        },
        "projection": projection_summary,
        "recurrence_diagnostics": rollout.metadata["diagnostics"],
        "optimizer": fit_metadata,
        "io_map_certificate": {
            "status": "standard_components_available",
            "owner_issue": IO_MAP_CERTIFICATE_ISSUE_ID,
            "reason": (
                "The row consumes the integrated standard certificate component builder "
                "for linear recurrence response maps and augmented rollout diagnostics."
            ),
        },
        "verdict": _row_verdict(condition, projection_summary, action_summary),
    }
    spec = BridgeRunSpec(
        issue_id=ISSUE_ID,
        run_id=condition.run_id,
        objective=(
            "reward_rollout" if fit_metadata.get("is_reward_trained", False) else "diagnostic"
        ),
        architecture="linear_recurrence",
        controller_label=condition.label,
        optimizer_label=str(fit_metadata["fit_method"]),
        training_distribution=condition.training_distribution,  # type: ignore[arg-type]
        evaluation_lane="deterministic"
        if fit_metadata.get("is_reward_trained", False)
        else "diagnostic",
        reference_controller="analytical_lqr_kalman_oracle_recurrence",
        parameters={
            "rank": condition.rank,
            "basis": (
                "time_identity"
                if condition.row_family == "exact_oracle_sanity"
                else "clamped_b_spline"
            ),
            "degree": (
                0 if condition.row_family == "exact_oracle_sanity" else min(3, condition.rank - 1)
            ),
            "row_family": condition.row_family,
            "evaluation_lens": condition.evaluation_lens,
            "coverage_family": condition.coverage_family,
            "coverage_modes": condition.coverage_modes,
            "coverage_scale": condition.coverage_scale,
            "disturbance_scale": condition.disturbance_scale,
            "measurement_scale": condition.measurement_scale,
            "n_train_steps": (
                condition.n_train_steps if fit_metadata.get("is_reward_trained", False) else 0
            ),
            "learning_rate": (
                condition.learning_rate if fit_metadata.get("is_reward_trained", False) else None
            ),
            "stability_penalty": (
                condition.stability_penalty
                if fit_metadata.get("is_reward_trained", False)
                else None
            ),
            "initialization": fit_metadata.get("initialization"),
        },
        notes=(
            "Phase-modulated linear recurrence uses spline coefficients for "
            "time-varying A/B/C/D matrices. Exact-oracle rows replay the analytical "
            "Kalman recurrence; projected-oracle rows are representational diagnostics "
            "without supervised optimization. Reward rows optimize these coefficients "
            "against the retained rollout objective."
        ),
    )
    return BridgeRunManifest(
        spec=spec,
        status=metrics["verdict"],
        arrays=rollout.array_specs(),
        metrics=metrics,
        certificate_components=components,
    )


def _projection_summary(projections: dict[str, MatrixBasisProjection]) -> dict[str, Any]:
    weighted_num = 0.0
    weighted_den = 0.0
    matrices: dict[str, Any] = {}
    for name, projection in projections.items():
        residual = projection.residual_norm
        denominator = residual / max(projection.relative_residual, np.finfo(np.float64).eps)
        weighted_num += residual**2
        weighted_den += denominator**2
        matrices[name] = {
            "relative_residual": projection.relative_residual,
            "residual_norm": projection.residual_norm,
            "rank": projection.rank,
        }
    return {
        "combined_relative_residual": float(
            np.sqrt(weighted_num) / max(np.sqrt(weighted_den), np.finfo(np.float64).eps)
        ),
        "matrices": matrices,
    }


def _row_verdict(
    condition: PhaseModulatedCondition,
    projection: dict[str, Any],
    action_summary: dict[str, Any],
) -> str:
    if condition.row_family == "exact_oracle_sanity":
        if (
            action_summary["aggregate_mismatch_ratio"] < 1e-18
            and projection["combined_relative_residual"] < 1e-12
        ):
            return "exact_oracle_sanity_pass"
        return "exact_oracle_sanity_fail"
    if condition.row_family == "legacy_projected_oracle_replay":
        return "legacy_projected_oracle_replay_diagnostic"
    if condition.row_family == "projected_oracle_replay":
        return "projected_oracle_replay_diagnostic"
    if condition.row_family == "projected_oracle_io_eval":
        return "projected_oracle_io_diagnostic"
    if condition.row_family == "projected_oracle_state_coverage_eval":
        return "state_coverage_projection_diagnostic"
    if action_summary["mismatch_ratio_mean"] < 0.1:
        return "reward_trained_reference_equivalent"
    if (
        projection["combined_relative_residual"] < 0.05
        and action_summary["mismatch_ratio_mean"] < 0.1
    ):
        return "reward_trained_projection_close"
    return "reward_trained_non_equivalent"


def _training_batch_for_condition(
    condition: PhaseModulatedCondition,
    *,
    plant: Any,
    gains: np.ndarray,
    x0: np.ndarray,
    observation_dim: int,
    output_config: OutputFeedbackConfig,
) -> dict[str, np.ndarray]:
    x0_batch, xhat0_batch = _coverage_initial_states(condition, plant=plant, x0=x0)
    disturbances = _disturbances_for_condition(
        condition,
        batch_size=x0_batch.shape[0],
        horizon=gains.shape[0],
        disturbance_dim=int(plant.Bw.shape[1]),
    )
    measurement_disturbances = _measurement_disturbances_for_condition(
        condition,
        batch_size=x0_batch.shape[0],
        horizon=gains.shape[0],
        observation_dim=observation_dim,
    )
    return {
        "x0": x0_batch,
        "xhat0": xhat0_batch,
        "disturbances": disturbances,
        "measurement_disturbances": measurement_disturbances,
    }


def _coverage_initial_states(
    condition: PhaseModulatedCondition,
    *,
    plant: Any,
    x0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_x = np.asarray(x0, dtype=np.float64)
    if condition.coverage_family is None:
        return base_x[None, :], base_x[None, :]

    modes = int(condition.coverage_modes or 1)
    scale = float(condition.coverage_scale)
    directions = _state_eigen_directions(plant=plant, x0=base_x, modes=modes)
    offsets = [np.zeros_like(base_x)]
    offsets.extend(scale * direction for direction in directions)
    offsets.extend(-scale * direction for direction in directions)
    offsets_array = np.stack(offsets, axis=0)
    if condition.coverage_family == "observer_error_state":
        return np.broadcast_to(base_x, offsets_array.shape).copy(), base_x[None, :] + offsets_array
    if condition.coverage_family == "mixed_deviation":
        observer_x = np.broadcast_to(base_x, offsets_array.shape).copy()
        observer_xhat = base_x[None, :] + offsets_array
        state_x = base_x[None, :] + offsets_array
        return (
            np.concatenate([state_x, observer_x], axis=0),
            np.concatenate([state_x.copy(), observer_xhat], axis=0),
        )
    return base_x[None, :] + offsets_array, base_x[None, :] + offsets_array


def _state_eigen_directions(*, plant: Any, x0: np.ndarray, modes: int) -> np.ndarray:
    A = np.asarray(plant.A, dtype=np.float64)
    trajectory = [x0]
    state = x0
    for _ in range(16):
        state = state @ A.T
        trajectory.append(state)
    cov = np.asarray(trajectory).T @ np.asarray(trajectory)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigvals)[::-1]
    directions = eigvecs[:, order[:modes]].T
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.maximum(norms, 1e-12)


def _disturbances_for_condition(
    condition: PhaseModulatedCondition,
    *,
    batch_size: int,
    horizon: int,
    disturbance_dim: int,
) -> np.ndarray:
    disturbances = np.zeros((batch_size, horizon, disturbance_dim), dtype=np.float64)
    if condition.disturbance_scale <= 0.0 or disturbance_dim == 0:
        return disturbances
    tau = np.linspace(0.0, np.pi, horizon, dtype=np.float64)
    disturbances[:, :, 0] = condition.disturbance_scale * np.sin(tau)[None, :]
    return disturbances


def _measurement_disturbances_for_condition(
    condition: PhaseModulatedCondition,
    *,
    batch_size: int,
    horizon: int,
    observation_dim: int,
) -> np.ndarray:
    measurement = np.zeros((batch_size, horizon, observation_dim), dtype=np.float64)
    if condition.measurement_scale <= 0.0 or observation_dim == 0:
        return measurement
    tau = np.linspace(0.0, 2.0 * np.pi, horizon, dtype=np.float64)
    for index in range(observation_dim):
        phase = index * np.pi / max(observation_dim, 1)
        measurement[:, :, index] = condition.measurement_scale * np.cos(tau + phase)[None, :]
    return measurement


def _reference_output_feedback_batch(
    *,
    plant: Any,
    gains: np.ndarray,
    x0: np.ndarray,
    xhat0: np.ndarray,
    output_config: OutputFeedbackConfig,
    disturbances: np.ndarray,
    measurement_disturbances: np.ndarray,
) -> dict[str, np.ndarray]:
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    Bw = np.asarray(plant.Bw, dtype=np.float64)
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    K = np.asarray(gains, dtype=np.float64)
    L = np.asarray(kalman_estimator_gains(plant, jnp.asarray(K), output_config), dtype=np.float64)
    process = np.asarray(process_covariance(plant, output_config), dtype=np.float64)
    sigma = np.eye(A.shape[0]) * output_config.estimator_initial_covariance
    x = np.asarray(x0, dtype=np.float64).copy()
    xhat = np.asarray(xhat0, dtype=np.float64).copy()
    x_seq = [x]
    xhat_seq = [xhat]
    y_seq = []
    u_seq = []
    for t in range(K.shape[0]):
        y_t = x @ H.T + measurement_disturbances[:, t, :]
        u_t = -xhat @ K[t].T
        xhat = xhat @ (A - B @ K[t] - L[t] @ H).T + y_t @ L[t].T
        x = x @ A.T + u_t @ B.T + disturbances[:, t, :] @ Bw.T
        sigma = (A - L[t] @ H) @ sigma @ A.T + process
        sigma = 0.5 * (sigma + sigma.T)
        u_seq.append(u_t)
        y_seq.append(y_t)
        x_seq.append(x)
        xhat_seq.append(xhat)
    return {
        "x": np.stack(x_seq, axis=1),
        "x_hat": np.stack(xhat_seq, axis=1),
        "y": np.stack(y_seq, axis=1),
        "u": np.stack(u_seq, axis=1),
    }


def _batch_quadratic_cost(schedule: Any, states: np.ndarray, actions: np.ndarray) -> float:
    x = np.asarray(states, dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64)
    Q = np.asarray(schedule.Q, dtype=np.float64)
    R = np.asarray(schedule.R, dtype=np.float64)
    Q_f = np.asarray(schedule.Q_f, dtype=np.float64)
    state_terms = np.einsum("bti,tij,btj->bt", x[:, :-1, :], Q, x[:, :-1, :])
    control_terms = np.einsum("bti,tij,btj->bt", u, R, u)
    terminal = np.einsum("bi,ij,bj->b", x[:, -1, :], Q_f, x[:, -1, :])
    return float(np.mean(np.sum(state_terms + control_terms, axis=1) + terminal))


def _result_text(rows: list[BridgeRunManifest]) -> str:
    exact_rows = [row for row in rows if row.metrics["row_family"] == "exact_oracle_sanity"]
    r12 = [
        row
        for row in rows
        if row.spec.parameters.get("rank") == 12
        and row.metrics["row_family"] == "projected_oracle_replay"
    ]
    if not r12:
        return "Phase-modulated recurrent rows were materialized with pending I/O-map certificates."
    mismatch = r12[0].metrics["state_weighted_action_mismatch"]
    residual = r12[0].metrics["projection"]["combined_relative_residual"]
    exact_response = (
        max(row.metrics["response_map_mismatch"]["max_aggregate_mismatch"] for row in exact_rows)
        if exact_rows
        else float("nan")
    )
    reward_rows = [
        row
        for row in rows
        if row.metrics["row_family"] in {"reward_lens", "projection_warm_start_then_reward_lens"}
        and row.metrics["optimizer"].get("is_reward_trained", False)
    ]
    return (
        "The exact oracle and clamped-spline projected-oracle rows were materialized. "
        f"Exact-oracle sanity rows have max aggregate response-map mismatch "
        f"{exact_response:.4g}. The r=12 projected-oracle nominal replay row has "
        f"action mismatch {mismatch:.4g} and combined matrix residual {residual:.4g}. "
        f"{len(reward_rows)} r=12 reward rows were optimized with bounded Adam "
        "over phase-modulated recurrent coefficients."
    )


def _repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


__all__ = [
    "ARTIFACT_PATH",
    "ISSUE_ID",
    "MANIFEST_PATH",
    "NOTE_PATH",
    "OracleRecurrentReference",
    "PhaseModulatedCondition",
    "build_oracle_recurrent_reference",
    "default_conditions",
    "exact_oracle_controller",
    "materialize",
    "phase_basis",
    "project_oracle_reference",
    "render_markdown",
    "write_outputs",
]

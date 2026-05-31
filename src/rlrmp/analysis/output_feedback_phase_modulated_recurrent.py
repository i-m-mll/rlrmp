"""Phase-modulated linear recurrent output-feedback bridge for issue d6d25d6."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.bridge_certificates import (
    DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH,
    OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    build_standard_certificate_components,
)
from rlrmp.analysis.bridge_contracts import BridgeRunManifest, BridgeRunSpec, make_bridge_run_id
from rlrmp.analysis.bridge_controllers import (
    MatrixBasisProjection,
    PhaseModulatedLinearRecurrentController,
    clamped_bspline_time_basis,
    project_matrix_sequence_to_basis,
    rollout_phase_modulated_linear_recurrent_controller,
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

SPLINE_RANKS = (3, 5, 8, 12)


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

    @property
    def run_id(self) -> str:
        """Stable row identifier."""

        return make_bridge_run_id("phase_modulated_recurrent", self.label)


def default_conditions(*, include_reward: bool = True) -> tuple[PhaseModulatedCondition, ...]:
    """Return the planned representation, imitation, I/O-map, and r=12 rows."""

    projection = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_oracle_matrix_projection",
            row_family="oracle_matrix_projection",
            rank=rank,
            training_distribution="none",
            evaluation_lens="oracle_matrix_projection",
        )
        for rank in SPLINE_RANKS
    )
    imitation = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_action_imitation_nominal",
            row_family="action_imitation",
            rank=rank,
            training_distribution="nominal",
            evaluation_lens="nominal_clean",
        )
        for rank in SPLINE_RANKS
    )
    io_map = tuple(
        PhaseModulatedCondition(
            label=f"pm_linrec_r{rank}_io_map_imitation_exact_process_eigen_m4",
            row_family="io_map_imitation",
            rank=rank,
            training_distribution="eigenspectrum_state",
            evaluation_lens="exact_process_eigen_m4",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=0.3,
        )
        for rank in SPLINE_RANKS
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
            label="pm_linrec_r12_exact_process_eigen_m1_s0p3_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="eigenspectrum_state",
            evaluation_lens="exact_process_eigen_m1_s0.3",
            coverage_family="state_eigenspectrum",
            coverage_modes=1,
            coverage_scale=0.3,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_exact_process_eigen_m4_s0p3_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="eigenspectrum_state",
            evaluation_lens="exact_process_eigen_m4_s0.3",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=0.3,
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_exact_process_eigen_m4_s1_reward",
            row_family="reward_lens",
            rank=12,
            training_distribution="eigenspectrum_state",
            evaluation_lens="exact_process_eigen_m4_s1",
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
            label="pm_linrec_r12_imitation_nominal_then_reward",
            row_family="imitation_then_reward_lens",
            rank=12,
            training_distribution="nominal",
            evaluation_lens="nominal_clean",
        ),
        PhaseModulatedCondition(
            label="pm_linrec_r12_imitation_exact_eigen_m4_then_reward",
            row_family="imitation_then_reward_lens",
            rank=12,
            training_distribution="eigenspectrum_state",
            evaluation_lens="exact_process_eigen_m4_s0.3",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=0.3,
        ),
    )
    return projection + imitation + io_map + (reward if include_reward else ())


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
    retained = conditions or default_conditions(include_reward=include_reward)
    bases = {rank: phase_basis(horizon=gains.shape[0], rank=rank) for rank in SPLINE_RANKS}
    projected = {rank: project_oracle_reference(oracle, basis=bases[rank]) for rank in SPLINE_RANKS}
    arrays: dict[str, np.ndarray] = {
        "reference_clean_x": np.asarray(clean_reference.x),
        "reference_clean_x_hat": np.asarray(clean_reference.x_hat),
        "reference_clean_u": np.asarray(clean_reference.u),
        "oracle_A_h": oracle.recurrent_matrices,
        "oracle_B_y": oracle.observation_matrices,
        "oracle_C_h": oracle.readout_matrices,
    }
    for rank, basis in bases.items():
        arrays[f"clamped_bspline_r{rank}_basis"] = basis

    rows: list[BridgeRunManifest] = []
    for condition in retained:
        controller, projections = projected[condition.rank]
        training = _training_batch_for_condition(
            condition,
            plant=plant,
            gains=gains,
            x0=x0,
            output_config=output_config,
        )
        rollout = rollout_phase_modulated_linear_recurrent_controller(
            controller,
            plant,
            training["x0"],
            observation_matrix=oracle.observation_matrix,
            disturbances=training["disturbances"],
            initial_hidden=training["xhat0"],
        )
        reference_batch = _reference_output_feedback_batch(
            plant=plant,
            gains=gains,
            x0=training["x0"],
            xhat0=training["xhat0"],
            output_config=output_config,
            disturbances=training["disturbances"],
        )
        row = _manifest_for_condition(
            condition=condition,
            controller=controller,
            oracle=oracle,
            plant=plant,
            rollout=rollout,
            reference_batch=reference_batch,
            schedule=schedule,
            clean_reference_cost=clean_reference_cost,
            projections=projections,
        )
        rows.append(row)
        prefix = condition.run_id
        arrays[f"{prefix}__plant_states"] = np.asarray(rollout.plant_states)
        arrays[f"{prefix}__hidden_states"] = np.asarray(rollout.hidden_states)
        arrays[f"{prefix}__actions"] = np.asarray(rollout.actions)
        arrays[f"{prefix}__reference_actions"] = reference_batch["u"]

    component_counts: Counter[str] = Counter()
    for row in rows:
        for component in row.certificate_components:
            component_counts[f"{component.name}:{component.status}"] += 1
    summary = {
        "format": "rlrmp.output_feedback_phase_modulated_recurrent.v1",
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
            "No GRU training, no broad robust-epsilon arm, and no claim that "
            "supervised/imitation/reward-lens diagnostic rows are bridge passes."
        ),
        "runtime_seconds": time.perf_counter() - start,
        "diagnostics": {
            "basis_family": "clamped_b_spline_partition_of_unity",
            "basis_ranks": list(SPLINE_RANKS),
            "basis_degrees": {str(rank): min(3, rank - 1) for rank in SPLINE_RANKS},
            "component_status_counts": dict(sorted(component_counts.items())),
            "retained_rows": [row.spec.run_id for row in rows],
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
        "| row | family | lens | verdict | objective ratio | action mismatch | matrix residual | io-map |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        metrics = row["metrics"]
        table.append(
            "| "
            f"{row['spec']['run_id']} | "
            f"{metrics['row_family']} | "
            f"{metrics['evaluation_lens']} | "
            f"{metrics['verdict']} | "
            f"{metrics['objective_ratio_to_clean_reference']:.8g} | "
            f"{metrics['state_weighted_action_mismatch']:.8g} | "
            f"{metrics['projection']['combined_relative_residual']:.8g} | "
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

## Rows

{"\n".join(table)}

## Certificate Boundary

These rows are diagnostics, not bridge passes. The recurrent controller is an
augmented-linear system over `z_t = [x_t; h_t]`, and action/visited-state
components are reported through the existing augmented recurrent adapter. The
I/O-map certificate from `{IO_MAP_CERTIFICATE_ISSUE_ID}` compares finite-horizon
observation-to-action, disturbance-to-action, and disturbance-to-state response
maps against the oracle recurrent reference.

{"\n".join(component_rows)}

## Interpretation

The oracle row constructs the exact finite-horizon Kalman recurrent reference.
Ranked rows then project each time-varying recurrent/readout matrix onto a
clamped B-spline partition of unity. Reward-lens rows reuse the same projected
oracle initialization under retained evaluation distributions; they diagnose
objective behavior and do not claim scratch reward optimization or bridge pass
status.
"""


def _manifest_for_condition(
    *,
    condition: PhaseModulatedCondition,
    controller: PhaseModulatedLinearRecurrentController,
    oracle: OracleRecurrentReference,
    plant: Any,
    rollout: Any,
    reference_batch: dict[str, np.ndarray],
    schedule: Any,
    clean_reference_cost: float,
    projections: dict[str, MatrixBasisProjection],
) -> BridgeRunManifest:
    candidate_actions = np.asarray(rollout.actions)
    reference_actions = reference_batch["u"]
    candidate_cost = _batch_quadratic_cost(schedule, rollout.plant_states, candidate_actions)
    reference_cost = _batch_quadratic_cost(schedule, reference_batch["x"], reference_actions)
    augmented_states = np.concatenate([rollout.plant_states, rollout.hidden_states], axis=-1)
    candidate_maps = _response_maps_for_phase_modulated_controller(
        controller=controller,
        plant=plant,
        observation_matrix=oracle.observation_matrix,
    )
    reference_maps = _response_maps_for_matrix_sequences(
        plant=plant,
        observation_matrix=oracle.observation_matrix,
        recurrent=oracle.recurrent_matrices,
        observation=oracle.observation_matrices,
        previous_action=oracle.previous_action_matrices,
        readout=oracle.readout_matrices,
        feedthrough=oracle.feedthrough_matrices,
    )
    components = build_standard_certificate_components(
        architecture="linear_recurrence",
        certificate_mode="augmented_linear",
        augmented_states=augmented_states,
        candidate_actions=candidate_actions,
        reference_actions=reference_actions,
        recurrence_diagnostics=rollout.metadata["diagnostics"],
        optimizer_metadata={
            "fit_method": "oracle_matrix_projection_to_clamped_spline_basis",
            "rank": condition.rank,
            "row_family": condition.row_family,
        },
        state_label="plant_hidden_augmented_state",
        action_label="control",
        candidate_observation_to_action_map=candidate_maps["observation_to_action"],
        reference_observation_to_action_map=reference_maps["observation_to_action"],
        candidate_disturbance_to_action_map=candidate_maps["disturbance_to_action"],
        reference_disturbance_to_action_map=reference_maps["disturbance_to_action"],
        candidate_disturbance_to_state_map=candidate_maps["disturbance_to_state"],
        reference_disturbance_to_state_map=reference_maps["disturbance_to_state"],
    )
    by_name = {component.name: component for component in components}
    action_summary = by_name[STATE_WEIGHTED_ACTION_MISMATCH].summary
    io_map_summary = _io_map_certificate_summary(by_name)
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
        "projection": projection_summary,
        "recurrence_diagnostics": rollout.metadata["diagnostics"],
        "io_map_certificate": io_map_summary,
        "verdict": _row_verdict(condition, projection_summary, action_summary, io_map_summary),
    }
    spec = BridgeRunSpec(
        issue_id=ISSUE_ID,
        run_id=condition.run_id,
        objective="diagnostic",
        architecture="linear_recurrence",
        controller_label=condition.label,
        optimizer_label="oracle_projection_clamped_bspline",
        training_distribution=condition.training_distribution,  # type: ignore[arg-type]
        evaluation_lane="diagnostic",
        reference_controller="analytical_lqr_kalman_oracle_recurrence",
        parameters={
            "rank": condition.rank,
            "basis": "clamped_b_spline",
            "degree": min(3, condition.rank - 1),
            "row_family": condition.row_family,
            "evaluation_lens": condition.evaluation_lens,
            "coverage_family": condition.coverage_family,
            "coverage_modes": condition.coverage_modes,
            "coverage_scale": condition.coverage_scale,
            "disturbance_scale": condition.disturbance_scale,
        },
        notes=(
            "Phase-modulated linear recurrence uses spline coefficients for "
            "time-varying A/B/C/D matrices. I/O-map certificate components "
            "compare external response maps to the oracle recurrent reference; "
            "imitation/projection rows remain diagnostic rather than bridge passes."
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
    io_map_summary: dict[str, Any],
) -> str:
    if condition.row_family == "oracle_matrix_projection":
        return "representation_diagnostic"
    if condition.row_family in {"action_imitation", "io_map_imitation"}:
        return "imitation_diagnostic"
    if (
        projection["combined_relative_residual"] < 0.05
        and action_summary["mismatch_ratio_mean"] < 0.1
        and io_map_summary["max_relative_mismatch"] < 0.1
    ):
        return "reward_lens_projection_close_io_map_near_reference"
    return "reward_lens_non_equivalent"


def _io_map_certificate_summary(
    by_name: dict[str, Any],
) -> dict[str, Any]:
    names = (
        OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
        DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH,
        DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH,
    )
    components = {name: by_name[name] for name in names}
    available = [component for component in components.values() if component.status == "available"]
    ratios = [
        float(component.summary["aggregate_mismatch_ratio"])
        for component in available
        if "aggregate_mismatch_ratio" in component.summary
    ]
    return {
        "status": "available" if len(available) == len(names) else "missing",
        "owner_issue": IO_MAP_CERTIFICATE_ISSUE_ID,
        "max_relative_mismatch": max(ratios) if ratios else float("nan"),
        "components": {
            name: {
                "status": component.status,
                "aggregate_mismatch_ratio": component.summary.get("aggregate_mismatch_ratio"),
                "mismatch_ratio_mean": component.summary.get("mismatch_ratio_mean"),
                "covariance_weighted_mismatch_ratio_mean": component.summary.get(
                    "covariance_weighted_mismatch_ratio_mean"
                ),
            }
            for name, component in components.items()
        },
    }


def _response_maps_for_phase_modulated_controller(
    *,
    controller: PhaseModulatedLinearRecurrentController,
    plant: Any,
    observation_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    return _response_maps_for_matrix_sequences(
        plant=plant,
        observation_matrix=observation_matrix,
        recurrent=controller.matrix_sequence(controller.recurrent_coefficients),
        observation=controller.matrix_sequence(controller.observation_coefficients),
        previous_action=controller.matrix_sequence(controller.previous_action_coefficients),
        readout=controller.matrix_sequence(controller.readout_coefficients),
        feedthrough=controller.matrix_sequence(controller.feedthrough_coefficients),
    )


def _response_maps_for_matrix_sequences(
    *,
    plant: Any,
    observation_matrix: np.ndarray,
    recurrent: np.ndarray,
    observation: np.ndarray,
    previous_action: np.ndarray,
    readout: np.ndarray,
    feedthrough: np.ndarray,
) -> dict[str, np.ndarray]:
    horizon = int(recurrent.shape[0])
    observation_dim = int(observation.shape[2])
    disturbance_dim = int(plant.Bw.shape[1])
    action_dim = int(readout.shape[1])
    state_dim = int(plant.A.shape[0])
    obs_inputs = horizon * observation_dim
    disturbance_inputs = horizon * disturbance_dim

    obs_to_action = np.zeros((horizon, action_dim, obs_inputs), dtype=np.float64)
    for input_index in range(obs_inputs):
        observations = np.zeros((horizon, observation_dim), dtype=np.float64)
        observations.reshape(-1)[input_index] = 1.0
        obs_to_action[:, :, input_index] = _linear_recurrence_actions_from_observations(
            recurrent=recurrent,
            observation=observation,
            previous_action=previous_action,
            readout=readout,
            feedthrough=feedthrough,
            observations=observations,
        )

    disturbance_to_action = np.zeros(
        (horizon, action_dim, disturbance_inputs),
        dtype=np.float64,
    )
    disturbance_to_state = np.zeros(
        (horizon + 1, state_dim, disturbance_inputs),
        dtype=np.float64,
    )
    for input_index in range(disturbance_inputs):
        disturbances = np.zeros((horizon, disturbance_dim), dtype=np.float64)
        disturbances.reshape(-1)[input_index] = 1.0
        x, u = _linear_closed_loop_response_from_disturbances(
            plant=plant,
            observation_matrix=observation_matrix,
            recurrent=recurrent,
            observation=observation,
            previous_action=previous_action,
            readout=readout,
            feedthrough=feedthrough,
            disturbances=disturbances,
        )
        disturbance_to_action[:, :, input_index] = u
        disturbance_to_state[:, :, input_index] = x
    return {
        "observation_to_action": obs_to_action,
        "disturbance_to_action": disturbance_to_action,
        "disturbance_to_state": disturbance_to_state,
    }


def _linear_recurrence_actions_from_observations(
    *,
    recurrent: np.ndarray,
    observation: np.ndarray,
    previous_action: np.ndarray,
    readout: np.ndarray,
    feedthrough: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    horizon = recurrent.shape[0]
    hidden = np.zeros((recurrent.shape[1],), dtype=np.float64)
    prev_u = np.zeros((readout.shape[1],), dtype=np.float64)
    actions = []
    for t in range(horizon):
        y_t = observations[t]
        u_t = readout[t] @ hidden + feedthrough[t] @ y_t
        hidden = recurrent[t] @ hidden + observation[t] @ y_t + previous_action[t] @ prev_u
        prev_u = u_t
        actions.append(u_t)
    return np.stack(actions, axis=0)


def _linear_closed_loop_response_from_disturbances(
    *,
    plant: Any,
    observation_matrix: np.ndarray,
    recurrent: np.ndarray,
    observation: np.ndarray,
    previous_action: np.ndarray,
    readout: np.ndarray,
    feedthrough: np.ndarray,
    disturbances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    Bw = np.asarray(plant.Bw, dtype=np.float64)
    H = np.asarray(observation_matrix, dtype=np.float64)
    horizon = recurrent.shape[0]
    x = np.zeros((A.shape[0],), dtype=np.float64)
    hidden = np.zeros((recurrent.shape[1],), dtype=np.float64)
    prev_u = np.zeros((readout.shape[1],), dtype=np.float64)
    states = [x.copy()]
    actions = []
    for t in range(horizon):
        y_t = H @ x
        u_t = readout[t] @ hidden + feedthrough[t] @ y_t
        hidden = recurrent[t] @ hidden + observation[t] @ y_t + previous_action[t] @ prev_u
        x = A @ x + B @ u_t + Bw @ disturbances[t]
        prev_u = u_t
        actions.append(u_t)
        states.append(x.copy())
    return np.stack(states, axis=0), np.stack(actions, axis=0)


def _training_batch_for_condition(
    condition: PhaseModulatedCondition,
    *,
    plant: Any,
    gains: np.ndarray,
    x0: np.ndarray,
    output_config: OutputFeedbackConfig,
) -> dict[str, np.ndarray]:
    x0_batch, xhat0_batch = _coverage_initial_states(condition, plant=plant, x0=x0)
    disturbances = _disturbances_for_condition(
        condition,
        batch_size=x0_batch.shape[0],
        horizon=gains.shape[0],
        disturbance_dim=int(plant.Bw.shape[1]),
    )
    return {"x0": x0_batch, "xhat0": xhat0_batch, "disturbances": disturbances}


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


def _reference_output_feedback_batch(
    *,
    plant: Any,
    gains: np.ndarray,
    x0: np.ndarray,
    xhat0: np.ndarray,
    output_config: OutputFeedbackConfig,
    disturbances: np.ndarray,
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
    u_seq = []
    for t in range(K.shape[0]):
        y_t = x @ H.T
        u_t = -xhat @ K[t].T
        xhat = xhat @ (A - B @ K[t] - L[t] @ H).T + y_t @ L[t].T
        x = x @ A.T + u_t @ B.T + disturbances[:, t, :] @ Bw.T
        sigma = (A - L[t] @ H) @ sigma @ A.T + process
        sigma = 0.5 * (sigma + sigma.T)
        u_seq.append(u_t)
        x_seq.append(x)
        xhat_seq.append(xhat)
    return {
        "x": np.stack(x_seq, axis=1),
        "x_hat": np.stack(xhat_seq, axis=1),
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
    r12 = [
        row
        for row in rows
        if row.spec.parameters.get("rank") == 12 and row.metrics["row_family"] == "action_imitation"
    ]
    if not r12:
        return "Phase-modulated recurrent rows were materialized with pending I/O-map certificates."
    mismatch = r12[0].metrics["state_weighted_action_mismatch"]
    residual = r12[0].metrics["projection"]["combined_relative_residual"]
    return (
        "The oracle recurrent reference and clamped-spline projections were "
        f"materialized. The r=12 nominal imitation row has action mismatch "
        f"{mismatch:.4g} and combined matrix residual {residual:.4g}. I/O-map "
        "certificate components are available, but these rows remain diagnostic "
        "rather than bridge passes."
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
    "materialize",
    "phase_basis",
    "project_oracle_reference",
    "render_markdown",
    "write_outputs",
]

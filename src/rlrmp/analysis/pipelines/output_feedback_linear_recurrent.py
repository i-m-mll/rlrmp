"""Phase-aware linear recurrent output-feedback bridge for issue 5e55f69.

This module keeps the GRU-facing bridge deliberately auditable: the controller
is a linear recurrence driven by delayed observations plus explicit phase/time
features. It does not claim a formal static-gain certificate. Instead, rows use
the augmented-linear recurrent certificate mode over plant plus hidden state
when those arrays are available.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from rlrmp.analysis.pipelines.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
    build_standard_certificate_components,
)
from rlrmp.analysis.pipelines.bridge_contracts import (
    BridgeRolloutBatch,
    BridgeRunManifest,
    BridgeRunSpec,
    make_bridge_run_id,
)
from rlrmp.analysis.pipelines.bridge_controllers import (
    LinearRecurrentController,
    hidden_growth_diagnostics,
)
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.pipelines.failure_decomposition import classify_failure
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    kalman_estimator_gains,
    make_cs_output_feedback_initial_state,
    output_feedback_cost,
    process_covariance,
    rollout_with_kalman_estimator,
)
from rlrmp.io import write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE_ID = "5e55f69"
UMBRELLA_ID = "43e8728"
SUBSTRATE_ISSUE_ID = "4ded904"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
FAILURE_DECOMPOSITION_ISSUE_ID = "c45adde"

NOTE_PATH = REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_linear_recurrent.md"
MANIFEST_PATH = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_linear_recurrent_manifest.json"
)
ARTIFACT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "output_feedback_linear_recurrent"
    / "output_feedback_linear_recurrent.npz"
)

PHASE_FEATURE_NAMES = ("phase_bias", "phase_tau", "phase_tau_squared")
FORMAL_STATIC_GAIN_COMPONENTS = (
    CLOSED_LOOP_TRANSITION_MISMATCH,
    VALUE_POLICY_GAP,
    BELLMAN_HESSIAN_RESIDUAL,
)
DEFAULT_HIDDEN_DIM = 48
DEFAULT_REWARD_STEPS = 80
DEFAULT_IMITATION_STEPS = 120
DEFAULT_FINE_TUNE_STEPS = 50
LENS_KIND_CATALOG = {
    "nominal_clean": "clean nominal output-feedback rollout from the canonical initial state",
    "true_riccati_epsilon": (
        "analytical H-infinity Riccati epsilon trajectory from the exact output-feedback audit"
    ),
    "sinusoidal_process_disturbance": (
        "hand-authored sinusoid injected into the first process-disturbance coordinate"
    ),
    "state_covariance_modes": (
        "eigenvectors of the open-loop A^k x0 covariance used as signed initial-state offsets"
    ),
    "exact_audit_eigen_disturbance_modes": (
        "disturbance modes from exact-audit epsilon trajectories or their covariance"
    ),
    "observer_error_svd_modes": ("left singular modes of the disturbance-to-observer-error map"),
    "observer_estimate_state_covariance_modes": (
        "state-covariance directions applied only to the observer estimate"
    ),
    "mixed_sinusoidal_process_and_state_covariance_modes": (
        "local sinusoidal process disturbance plus state-covariance offset coverage"
    ),
}


@dataclass(frozen=True)
class LinearRecurrentCondition:
    """One retained phase-aware linear recurrent bridge row."""

    label: str
    training_distribution: str
    initialization: str
    objective: str
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    coverage_family: str | None = None
    coverage_modes: int | None = None
    coverage_scale: float | None = None
    coverage_weight: float | None = None
    disturbance_scale: float = 0.0
    seed: int = 0
    n_train_steps: int = DEFAULT_REWARD_STEPS
    imitation_steps: int = DEFAULT_IMITATION_STEPS
    fine_tune_steps: int = 0
    learning_rate: float = 3e-3
    stability_penalty: float = 1e-3
    ridge: float = 1e-5

    @property
    def run_id(self) -> str:
        """Stable run identifier for manifests and array keys."""

        return make_bridge_run_id("linear_recurrent", self.label)


def lens_metadata_for_condition(condition: LinearRecurrentCondition) -> dict[str, Any]:
    """Return explicit local lens metadata without changing historical row IDs."""

    if condition.training_distribution == "riccati_epsilon" and condition.disturbance_scale > 0.0:
        return {
            "lens_kind": "sinusoidal_process_disturbance",
            "lens_source": "first_process_coordinate_sinusoid",
            "lens_deprecated_alias": "riccati_epsilon",
            "lens_notes": (
                "Historical row ID and training_distribution are preserved, but this row "
                "uses a sinusoidal process disturbance, not the exact Riccati epsilon."
            ),
        }
    if condition.coverage_family == "state_eigenspectrum":
        return {
            "lens_kind": "state_covariance_modes",
            "lens_source": "open_loop_A_power_x0_covariance_eigenvectors",
            "lens_deprecated_alias": "state_eigenspectrum",
            "lens_notes": (
                "Directions come from eigenvectors of the open-loop A^k x0 covariance; "
                "they are not exact-audit eigen disturbance modes."
            ),
        }
    if condition.coverage_family == "observer_error_state":
        return {
            "lens_kind": "observer_estimate_state_covariance_modes",
            "lens_source": "open_loop_A_power_x0_covariance_offsets_on_xhat",
            "lens_deprecated_alias": "observer_error",
            "lens_notes": (
                "Offsets are applied to the observer estimate along state-covariance "
                "directions; observer-error SVD modes are a separate lens kind."
            ),
        }
    if condition.coverage_family == "mixed_deviation":
        return {
            "lens_kind": "mixed_sinusoidal_process_and_state_covariance_modes",
            "lens_source": (
                "first_process_coordinate_sinusoid_plus_open_loop_A_power_x0_"
                "covariance_eigenvectors"
            ),
            "lens_deprecated_alias": "mixed_deviation",
            "lens_notes": (
                "Mixed rows combine the local sinusoidal process disturbance with "
                "state-covariance initial-state and observer-estimate offsets."
            ),
        }
    return {
        "lens_kind": "nominal_clean",
        "lens_source": "canonical_output_feedback_initial_state",
        "lens_deprecated_alias": None,
        "lens_notes": "No coverage or disturbance lens is applied.",
    }


def default_conditions(*, include_coverage: bool = True) -> tuple[LinearRecurrentCondition, ...]:
    """Return the planned trainable recurrent bridge rows."""

    conditions = (
        LinearRecurrentCondition(
            label="linrec_clean_scratch_baseline",
            training_distribution="clean_nominal",
            initialization="scratch",
            objective="reward_rollout",
        ),
        LinearRecurrentCondition(
            label="linrec_riccati_eps_scratch",
            training_distribution="riccati_epsilon",
            initialization="scratch",
            objective="reward_rollout",
            disturbance_scale=0.03,
        ),
        LinearRecurrentCondition(
            label="linrec_state_eig_scratch",
            training_distribution="state_eigenspectrum",
            initialization="scratch",
            objective="reward_rollout",
            coverage_family="state_eigenspectrum",
            coverage_modes=4,
            coverage_scale=1.0,
            coverage_weight=0.1,
        ),
        LinearRecurrentCondition(
            label="linrec_observer_error_scratch",
            training_distribution="observer_error",
            initialization="scratch",
            objective="reward_rollout",
            coverage_family="observer_error_state",
            coverage_modes=1,
            coverage_scale=0.3,
            coverage_weight=0.1,
        ),
        LinearRecurrentCondition(
            label="linrec_mixed_scratch",
            training_distribution="mixed_deviation",
            initialization="scratch",
            objective="reward_rollout",
            coverage_family="mixed_deviation",
            coverage_modes=4,
            coverage_scale=1.0,
            coverage_weight=0.1,
            disturbance_scale=0.02,
        ),
        LinearRecurrentCondition(
            label="linrec_imitation_nominal",
            training_distribution="nominal",
            initialization="reference_action_imitation",
            objective="imitation_diagnostic",
            n_train_steps=0,
            fine_tune_steps=0,
        ),
        LinearRecurrentCondition(
            label="linrec_imitation_mixed_then_rollout",
            training_distribution="mixed_deviation",
            initialization="reference_action_imitation_then_rollout",
            objective="imitation_then_reward_rollout_scaffold",
            coverage_family="mixed_deviation",
            coverage_modes=4,
            coverage_scale=1.0,
            coverage_weight=0.1,
            disturbance_scale=0.02,
            n_train_steps=0,
            fine_tune_steps=DEFAULT_FINE_TUNE_STEPS,
        ),
    )
    return conditions if include_coverage else conditions[:2]


def phase_time_features(horizon: int) -> Float[np.ndarray, "horizon phase"]:
    """Return the explicit phase/time input used by retained recurrent rows."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon == 1:
        tau = np.zeros((1,), dtype=np.float64)
    else:
        tau = np.linspace(0.0, 1.0, horizon, dtype=np.float64)
    return np.column_stack([np.ones_like(tau), tau, tau**2])


def rollout_phase_aware_linear_recurrent(
    *,
    controller: LinearRecurrentController,
    plant: Any,
    x0: np.ndarray,
    horizon: int,
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
    phase_features: np.ndarray | None = None,
    disturbances: np.ndarray | None = None,
) -> BridgeRolloutBatch:
    """Roll a phase-aware recurrence through the delayed-observation plant."""

    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    Bw = np.asarray(plant.Bw, dtype=np.float64)
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    phase = phase_time_features(horizon) if phase_features is None else np.asarray(phase_features)
    if phase.shape != (horizon, len(PHASE_FEATURE_NAMES)):
        raise ValueError(
            "phase_features must have shape "
            f"{(horizon, len(PHASE_FEATURE_NAMES))}; got {phase.shape}"
        )
    if controller.observation_dim != H.shape[0]:
        raise ValueError("controller observation dimension does not match delayed observation")
    if controller.phase_dim != phase.shape[1]:
        raise ValueError("controller phase dimension does not match phase features")

    states = _as_batch(np.asarray(x0, dtype=np.float64), width=A.shape[0], name="x0")
    batch_size = states.shape[0]
    eps = _normalize_disturbances(
        disturbances,
        batch_size=batch_size,
        disturbance_dim=Bw.shape[1],
        horizon=horizon,
    )
    hidden = np.broadcast_to(controller.initial_hidden, (batch_size, controller.hidden_dim)).copy()
    previous_action = np.zeros((batch_size, controller.action_dim), dtype=np.float64)

    plant_states = [states]
    hidden_states = [hidden]
    observations = []
    actions = []
    for t in range(horizon):
        delayed = states @ H.T
        phase_t = np.broadcast_to(phase[t], (batch_size, phase.shape[1]))
        u_t = controller.action(hidden, delayed, phase_t)
        states = states @ A.T + u_t @ B.T + eps[:, t, :] @ Bw.T
        hidden = controller.next_hidden(hidden, delayed, previous_action, phase_t)
        previous_action = u_t
        observations.append(np.concatenate([delayed, phase_t], axis=-1))
        actions.append(u_t)
        plant_states.append(states)
        hidden_states.append(hidden)

    hidden_array = np.stack(hidden_states, axis=1)
    diagnostics = {
        "phase_time_input_used": True,
        "phase_time_feature_names": list(PHASE_FEATURE_NAMES),
        "phase_time_input_dim": int(phase.shape[1]),
        "delayed_observation_dim": int(H.shape[0]),
        **controller.stability_diagnostics(),
        **hidden_growth_diagnostics(hidden_array),
    }
    return BridgeRolloutBatch(
        plant_states=np.stack(plant_states, axis=1),
        actions=np.stack(actions, axis=1),
        observations=np.stack(observations, axis=1),
        hidden_states=hidden_array,
        metadata={"controller": "phase_aware_linear_recurrence", "diagnostics": diagnostics},
    )


def materialize(
    *,
    include_coverage: bool = True,
    conditions: tuple[LinearRecurrentCondition, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Materialize retained phase-aware linear recurrent rows."""

    start = time.perf_counter()
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    output_config = OutputFeedbackConfig()
    plant = reference.plant
    schedule = reference.schedule
    K_ref = np.asarray(reference.lqr_solution.K, dtype=np.float64)
    x0 = np.asarray(make_cs_output_feedback_initial_state(plant, output_config), dtype=np.float64)
    reference_clean = rollout_with_kalman_estimator(plant, jnp.asarray(K_ref), jnp.asarray(x0))
    reference_clean_cost = output_feedback_cost(schedule, reference_clean)
    phase = phase_time_features(K_ref.shape[0])
    retained_conditions = conditions or default_conditions(include_coverage=include_coverage)

    rows: list[BridgeRunManifest] = []
    failure_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "reference_clean_x": np.asarray(reference_clean.x),
        "reference_clean_x_hat": np.asarray(reference_clean.x_hat),
        "reference_clean_u": np.asarray(reference_clean.u),
        "phase_time_features": phase,
    }
    ablation = _phase_ablation(reference_clean, plant, K_ref, output_config, phase)

    for condition in retained_conditions:
        training = _training_batch_for_condition(
            condition,
            plant=plant,
            K_ref=K_ref,
            x0=x0,
            schedule=schedule,
            output_config=output_config,
        )
        controller, fit_metadata = _controller_for_condition(
            condition,
            plant=plant,
            reference_training=training,
            phase=phase,
            output_config=output_config,
        )
        rollout = rollout_phase_aware_linear_recurrent(
            controller=controller,
            plant=plant,
            x0=x0,
            horizon=K_ref.shape[0],
            output_config=output_config,
            phase_features=phase,
        )
        candidate_cost = _quadratic_cost(schedule, rollout.plant_states[0], rollout.actions[0])
        row = _manifest_for_condition(
            condition=condition,
            rollout=rollout,
            reference_clean=reference_clean,
            reference_clean_cost=float(reference_clean_cost.total_without_disturbance_penalty),
            candidate_cost=candidate_cost,
            fit_metadata=fit_metadata,
        )
        rows.append(row)
        failure_rows.append(
            _failure_row(
                manifest=row,
                candidate_cost=candidate_cost,
                reference_cost=float(reference_clean_cost.total_without_disturbance_penalty),
            )
        )
        prefix = condition.run_id
        arrays[f"{prefix}__plant_states"] = np.asarray(rollout.plant_states)
        arrays[f"{prefix}__actions"] = np.asarray(rollout.actions)
        arrays[f"{prefix}__observations"] = np.asarray(rollout.observations)
        arrays[f"{prefix}__hidden_states"] = np.asarray(rollout.hidden_states)
        arrays[f"{prefix}__reference_actions"] = np.asarray(reference_clean.u)[None, :, :]
        arrays[f"{prefix}__training_x0"] = training["x0"]
        arrays[f"{prefix}__training_xhat0"] = training["xhat0"]
        arrays[f"{prefix}__training_disturbances"] = training["disturbances"]
        arrays[f"{prefix}__A_h"] = np.asarray(controller.recurrent_weights)
        arrays[f"{prefix}__B_y"] = np.asarray(controller.observation_weights)
        arrays[f"{prefix}__B_u"] = np.asarray(controller.previous_action_weights)
        arrays[f"{prefix}__B_phi"] = np.asarray(controller.phase_weights)
        arrays[f"{prefix}__b"] = np.asarray(controller.hidden_bias)
        arrays[f"{prefix}__C_h"] = np.asarray(controller.readout_weights)
        arrays[f"{prefix}__D_y"] = np.asarray(controller.feedthrough_weights)
        arrays[f"{prefix}__D_phi"] = np.asarray(controller.readout_phase_weights)
        arrays[f"{prefix}__c"] = np.asarray(controller.action_bias)

    component_counts: Counter[str] = Counter()
    for row in rows:
        for component in row.certificate_components:
            component_counts[f"{component.name}:{component.status}"] += 1

    summary = {
        "format": "rlrmp.output_feedback_linear_recurrent.v2",
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "source_issues": {
            "substrate": SUBSTRATE_ISSUE_ID,
            "standard_certificate": STANDARD_CERTIFICATE_ISSUE_ID,
            "failure_decomposition": FAILURE_DECOMPOSITION_ISSUE_ID,
        },
        "scope": (
            "Trainable phase-aware linear recurrent output-feedback rows using "
            "delayed observations, previous actions, and explicit polynomial "
            "phase/time inputs."
        ),
        "non_goals": (
            "No GRU training, robust/H-infinity training arm, formal game-card "
            "change, or affine tracker implementation."
        ),
        "runtime_seconds": time.perf_counter() - start,
        "diagnostics": {
            "phase_time_feature_names": list(PHASE_FEATURE_NAMES),
            "phase_ablation": ablation,
            "lens_kind_catalog": LENS_KIND_CATALOG,
            "component_status_counts": dict(sorted(component_counts.items())),
            "retained_rows": [row.spec.run_id for row in rows],
        },
        "rows": [row.to_json_dict() for row in rows],
        "failure_decomposition": {
            "schema": "recurrence-compatible c45adde subset",
            "rows": failure_rows,
            "classification_counts": dict(
                sorted(
                    Counter(row["classification"]["classification"] for row in failure_rows).items()
                )
            ),
        },
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
    """Write the tracked note/manifest and ignored bulk arrays."""

    mkdir_p(note_path.parent)
    mkdir_p(manifest_path.parent)
    mkdir_p(artifact_path.parent)
    results_dir = mkdir_p(REPO_ROOT / "results" / ISSUE_ID)
    readme = results_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Phase-aware linear recurrent output-feedback bridge. See "
            "`notes/output_feedback_linear_recurrent.md`.\n",
            encoding="utf-8",
        )
    np.savez_compressed(artifact_path, **arrays)
    summary["tracked_note"] = _repo_relative(note_path)
    summary["tracked_manifest"] = _repo_relative(manifest_path)
    summary["artifact_npz"] = _repo_relative(artifact_path)
    summary["artifact_npz_keys"] = sorted(arrays)
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    write_compact_json(manifest_path, summary)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked result note."""

    rows = [
        "| row | status | train dist | lens kind | lens source | objective ratio | "
        "action mismatch | spectral radius | hidden max | failure |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        metrics = row["metrics"]
        recurrence = metrics["recurrence_diagnostics"]
        lens = row["spec"]["parameters"].get("lens_metadata", {})
        failure = next(
            item
            for item in summary["failure_decomposition"]["rows"]
            if item["run_id"] == row["spec"]["run_id"]
        )
        rows.append(
            "| "
            f"{row['spec']['run_id']} | {row['status']} | "
            f"{row['spec']['training_distribution']} | "
            f"{lens.get('lens_kind', 'unknown')} | "
            f"{lens.get('lens_source', 'unknown')} | "
            f"{metrics['objective_ratio_to_reference']:.8g} | "
            f"{metrics['state_weighted_action_mismatch']:.8g} | "
            f"{recurrence['recurrent_spectral_radius']:.8g} | "
            f"{recurrence['hidden_max_norm']:.8g} | "
            f"{failure['classification']['classification']} |"
        )
    component_rows = [
        f"- `{key}`: {value}"
        for key, value in summary["diagnostics"]["component_status_counts"].items()
    ]
    ablation = summary["diagnostics"]["phase_ablation"]
    return f"""# Phase-Aware Linear Recurrent Output-Feedback Bridge

Issue: `{summary["issue"]}`. Umbrella: `{summary["umbrella"]}`.

Scope: {summary["scope"]}

Non-goals: {summary["non_goals"]}

Runtime: `{summary.get("runtime_seconds", 0.0):.2f}` seconds.

Verdict: {summary["result"]}

## Retained Rows

{"\n".join(rows)}

Historical row IDs and `training_distribution` values are preserved. The
`lens kind` and `lens source` columns are the authoritative interpretation for
these retained rows; deprecated aliases are kept only as compatibility labels.

## Lens Catalog

{"\n".join(f"- `{key}`: {value}" for key, value in summary["diagnostics"]["lens_kind_catalog"].items())}

## Certificate Boundary

Linear recurrent rows use the augmented-linear certificate mode over
`z_t = [x_t; h_t]` when plant and hidden states are available. Action mismatch,
visited-subspace diagnostics, optimizer metadata, and recurrence diagnostics
are therefore reported on the augmented state rather than through a static gain
certificate.

{"\n".join(component_rows)}

Transition/value/Bellman rows are explicit `missing` components in this pass
when a same-coordinate reference recurrent realization is unavailable. That is
different from a pass and different from the old static-gain `not_applicable`
boundary.

## Failure Diagnostics

The failure rows use a recurrence-compatible subset of `c45adde`: clean
objective ratio, state-weighted action mismatch, recurrence diagnostics, and
the standard failure classifier where its inputs are meaningful. Gain-subspace
decomposition is `not_applicable` for these retained recurrent rows.

## Phase/Time Input

Features: `{summary["diagnostics"]["phase_time_feature_names"]}`.
No-phase replay ablation training RMSE: `{ablation["no_phase_training_action_rmse"]:.8g}`;
phase-aware training RMSE: `{ablation["phase_training_action_rmse"]:.8g}`.
"""


def _controller_for_condition(
    condition: LinearRecurrentCondition,
    *,
    plant: Any,
    reference_training: dict[str, np.ndarray],
    phase: np.ndarray,
    output_config: OutputFeedbackConfig,
) -> tuple[LinearRecurrentController, dict[str, Any]]:
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    observation_dim = H.shape[0]
    action_dim = int(plant.B.shape[1])
    params = _initial_trainable_params(
        hidden_dim=condition.hidden_dim,
        observation_dim=observation_dim,
        action_dim=action_dim,
        phase_dim=phase.shape[1],
        seed=condition.seed,
    )
    fit_metadata: dict[str, Any] = {
        "fit_method": _fit_method_label(condition),
        "hidden_dim": condition.hidden_dim,
        "learning_rate": condition.learning_rate,
        "stability_penalty": condition.stability_penalty,
        "ridge": condition.ridge,
    }
    if condition.objective.startswith("imitation"):
        params, imitation_metadata = _fit_imitation_params(
            params,
            condition=condition,
            plant=plant,
            reference_training=reference_training,
            phase=phase,
            output_config=output_config,
        )
        fit_metadata.update(imitation_metadata)
    if condition.objective == "reward_rollout" or condition.fine_tune_steps > 0:
        steps = (
            condition.n_train_steps
            if condition.objective == "reward_rollout"
            else condition.fine_tune_steps
        )
        params, reward_metadata = _fit_reward_params(
            params,
            condition=condition,
            plant=plant,
            training=reference_training,
            phase=phase,
            output_config=output_config,
            n_steps=steps,
        )
        fit_metadata.update(reward_metadata)
    controller = _controller_from_params(params)
    fit_metadata.update(_parameter_norms(controller))
    return controller, fit_metadata


def _fit_method_label(condition: LinearRecurrentCondition) -> str:
    if condition.objective == "reward_rollout":
        return "adam_reward_rollout_from_scratch"
    if condition.objective == "imitation_diagnostic":
        return "adam_reference_action_imitation_diagnostic"
    return "adam_reference_action_imitation_then_reward_rollout"


def _initial_trainable_params(
    *,
    hidden_dim: int,
    observation_dim: int,
    action_dim: int,
    phase_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    recurrent = 0.72 * np.eye(hidden_dim, dtype=np.float64)
    recurrent += 0.01 * rng.normal(size=(hidden_dim, hidden_dim))
    return {
        "A_h": recurrent,
        "B_y": 0.03 * rng.normal(size=(hidden_dim, observation_dim)),
        "B_u": 0.02 * rng.normal(size=(hidden_dim, action_dim)),
        "B_phi": 0.03 * rng.normal(size=(hidden_dim, phase_dim)),
        "b_h": np.zeros((hidden_dim,), dtype=np.float64),
        "C_h": 0.03 * rng.normal(size=(action_dim, hidden_dim)),
        "D_y": 0.01 * rng.normal(size=(action_dim, observation_dim)),
        "D_phi": 0.01 * rng.normal(size=(action_dim, phase_dim)),
        "c": np.zeros((action_dim,), dtype=np.float64),
    }


def _controller_from_params(params: dict[str, np.ndarray]) -> LinearRecurrentController:
    return LinearRecurrentController(
        recurrent_weights=np.asarray(params["A_h"]),
        observation_weights=np.asarray(params["B_y"]),
        previous_action_weights=np.asarray(params["B_u"]),
        phase_weights=np.asarray(params["B_phi"]),
        hidden_bias=np.asarray(params["b_h"]),
        readout_weights=np.asarray(params["C_h"]),
        feedthrough_weights=np.asarray(params["D_y"]),
        readout_phase_weights=np.asarray(params["D_phi"]),
        action_bias=np.asarray(params["c"]),
    )


def _fit_imitation_params(
    params: dict[str, np.ndarray],
    *,
    condition: LinearRecurrentCondition,
    plant: Any,
    reference_training: dict[str, np.ndarray],
    phase: np.ndarray,
    output_config: OutputFeedbackConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    constants = _jax_training_constants(
        plant=plant,
        training=reference_training,
        phase=phase,
        output_config=output_config,
    )

    def loss_fn(jax_params: dict[str, jax.Array]) -> jax.Array:
        rollout = _jax_rollout(jax_params, constants)
        imitation = jnp.mean((rollout["actions"] - constants["reference_actions"]) ** 2)
        return imitation + _jax_stability_penalty(jax_params, condition.stability_penalty)

    fitted, history = _adam_minimize(
        params,
        loss_fn,
        n_steps=condition.imitation_steps,
        learning_rate=condition.learning_rate,
    )
    return fitted, {
        "imitation_steps": condition.imitation_steps,
        "imitation_initial_loss": history["initial_loss"],
        "imitation_final_loss": history["final_loss"],
        "training_action_rmse": float(np.sqrt(max(history["final_loss"], 0.0))),
    }


def _fit_reward_params(
    params: dict[str, np.ndarray],
    *,
    condition: LinearRecurrentCondition,
    plant: Any,
    training: dict[str, np.ndarray],
    phase: np.ndarray,
    output_config: OutputFeedbackConfig,
    n_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    constants = _jax_training_constants(
        plant=plant,
        training=training,
        phase=phase,
        output_config=output_config,
    )

    def loss_fn(jax_params: dict[str, jax.Array]) -> jax.Array:
        rollout = _jax_rollout(jax_params, constants)
        cost = _jax_quadratic_cost(
            states=rollout["states"],
            actions=rollout["actions"],
            q=constants["Q"],
            r=constants["R"],
            q_f=constants["Q_f"],
        )
        return cost + _jax_stability_penalty(jax_params, condition.stability_penalty)

    fitted, history = _adam_minimize(
        params,
        loss_fn,
        n_steps=n_steps,
        learning_rate=condition.learning_rate,
    )
    return fitted, {
        "reward_rollout_steps": n_steps,
        "reward_initial_loss": history["initial_loss"],
        "reward_final_loss": history["final_loss"],
    }


def _jax_training_constants(
    *,
    plant: Any,
    training: dict[str, np.ndarray],
    phase: np.ndarray,
    output_config: OutputFeedbackConfig,
) -> dict[str, jax.Array]:
    return {
        "A": jnp.asarray(plant.A, dtype=jnp.float64),
        "B": jnp.asarray(plant.B, dtype=jnp.float64),
        "Bw": jnp.asarray(plant.Bw, dtype=jnp.float64),
        "H": jnp.asarray(delayed_observation_matrix(plant, output_config), dtype=jnp.float64),
        "x0": jnp.asarray(training["x0"], dtype=jnp.float64),
        "disturbances": jnp.asarray(training["disturbances"], dtype=jnp.float64),
        "reference_actions": jnp.asarray(training["u"], dtype=jnp.float64),
        "phase": jnp.asarray(phase, dtype=jnp.float64),
        "Q": jnp.asarray(training["Q"], dtype=jnp.float64),
        "R": jnp.asarray(training["R"], dtype=jnp.float64),
        "Q_f": jnp.asarray(training["Q_f"], dtype=jnp.float64),
    }


def _jax_rollout(
    params: dict[str, jax.Array],
    constants: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    x = constants["x0"]
    batch_size = x.shape[0]
    hidden = jnp.zeros((batch_size, params["A_h"].shape[0]), dtype=x.dtype)
    previous_action = jnp.zeros((batch_size, params["C_h"].shape[0]), dtype=x.dtype)
    states = [x]
    hidden_states = [hidden]
    actions = []
    for t in range(constants["phase"].shape[0]):
        y_t = x @ constants["H"].T
        phi_t = jnp.broadcast_to(constants["phase"][t], (batch_size, constants["phase"].shape[1]))
        u_t = (
            hidden @ params["C_h"].T
            + y_t @ params["D_y"].T
            + phi_t @ params["D_phi"].T
            + params["c"]
        )
        x = (
            x @ constants["A"].T
            + u_t @ constants["B"].T
            + constants["disturbances"][:, t, :] @ constants["Bw"].T
        )
        hidden = (
            hidden @ params["A_h"].T
            + y_t @ params["B_y"].T
            + previous_action @ params["B_u"].T
            + phi_t @ params["B_phi"].T
            + params["b_h"]
        )
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


def _jax_stability_penalty(params: dict[str, jax.Array], scale: float) -> jax.Array:
    if scale <= 0.0:
        return jnp.asarray(0.0)
    row_norms = jnp.linalg.norm(params["A_h"], axis=1)
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
        return params, {"initial_loss": loss, "final_loss": loss}
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    initial_loss = float(jax.jit(loss_fn)(jax_params))
    m = jax.tree.map(jnp.zeros_like, jax_params)
    v = jax.tree.map(jnp.zeros_like, jax_params)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    loss = initial_loss
    for step in range(1, n_steps + 1):
        loss_value, grads = value_and_grad(jax_params)
        loss = float(loss_value)
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
        {key: np.asarray(value, dtype=np.float64) for key, value in jax_params.items()},
        {"initial_loss": initial_loss, "final_loss": loss},
    )


def _parameter_norms(controller: LinearRecurrentController) -> dict[str, float]:
    assert controller.previous_action_weights is not None
    assert controller.phase_weights is not None
    assert controller.hidden_bias is not None
    assert controller.readout_phase_weights is not None
    assert controller.action_bias is not None
    return {
        "A_h_norm": float(np.linalg.norm(controller.recurrent_weights)),
        "B_y_norm": float(np.linalg.norm(controller.observation_weights)),
        "B_u_norm": float(np.linalg.norm(controller.previous_action_weights)),
        "B_phi_norm": float(np.linalg.norm(controller.phase_weights)),
        "b_norm": float(np.linalg.norm(controller.hidden_bias)),
        "C_h_norm": float(np.linalg.norm(controller.readout_weights)),
        "D_y_norm": float(np.linalg.norm(controller.feedthrough_weights)),
        "D_phi_norm": float(np.linalg.norm(controller.readout_phase_weights)),
        "c_norm": float(np.linalg.norm(controller.action_bias)),
    }


def _training_batch_for_condition(
    condition: LinearRecurrentCondition,
    *,
    plant: Any,
    K_ref: np.ndarray,
    x0: np.ndarray,
    schedule: Any | None = None,
    output_config: OutputFeedbackConfig,
) -> dict[str, np.ndarray]:
    x0_batch, xhat0_batch = _coverage_initial_states(condition, plant=plant, x0=x0)
    batch = _reference_output_feedback_batch(
        plant=plant,
        K=K_ref,
        x0=x0_batch,
        xhat0=xhat0_batch,
        output_config=output_config,
    )
    horizon = K_ref.shape[0]
    disturbances = _disturbances_for_condition(
        condition,
        batch_size=x0_batch.shape[0],
        horizon=horizon,
        disturbance_dim=int(plant.Bw.shape[1]),
    )
    batch["disturbances"] = disturbances
    if schedule is not None:
        batch["Q"] = np.asarray(schedule.Q, dtype=np.float64)
        batch["R"] = np.asarray(schedule.R, dtype=np.float64)
        batch["Q_f"] = np.asarray(schedule.Q_f, dtype=np.float64)
    return batch


def _disturbances_for_condition(
    condition: LinearRecurrentCondition,
    *,
    batch_size: int,
    horizon: int,
    disturbance_dim: int,
) -> np.ndarray:
    disturbances = np.zeros((batch_size, horizon, disturbance_dim), dtype=np.float64)
    if condition.disturbance_scale <= 0.0 or disturbance_dim == 0:
        return disturbances
    tau = np.linspace(0.0, np.pi, horizon, dtype=np.float64)
    base = condition.disturbance_scale * np.sin(tau)
    disturbances[:, :, 0] = base[None, :]
    return disturbances


def _coverage_initial_states(
    condition: LinearRecurrentCondition,
    *,
    plant: Any,
    x0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_x = np.asarray(x0, dtype=np.float64)
    if condition.coverage_family is None:
        return base_x[None, :], base_x[None, :]

    modes = int(condition.coverage_modes or 1)
    scale = float(condition.coverage_scale or 0.0)
    directions = _state_eigen_directions(plant=plant, x0=base_x, modes=modes)
    offsets = [np.zeros_like(base_x)]
    offsets.extend(scale * direction for direction in directions)
    offsets.extend(-scale * direction for direction in directions)
    offsets_array = np.stack(offsets, axis=0)
    if condition.coverage_family == "observer_error_state":
        return (
            np.broadcast_to(base_x, offsets_array.shape).copy(),
            base_x[None, :] + offsets_array,
        )
    if condition.coverage_family == "mixed_deviation":
        observer_x = np.broadcast_to(base_x, offsets_array.shape).copy()
        observer_xhat = base_x[None, :] + offsets_array
        state_x = base_x[None, :] + offsets_array
        state_xhat = state_x.copy()
        return (
            np.concatenate([state_x, observer_x], axis=0),
            np.concatenate([state_xhat, observer_xhat], axis=0),
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


def _reference_output_feedback_batch(
    *,
    plant: Any,
    K: np.ndarray,
    x0: np.ndarray,
    xhat0: np.ndarray,
    output_config: OutputFeedbackConfig,
) -> dict[str, np.ndarray]:
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64)
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    gains = np.asarray(kalman_estimator_gains(plant, jnp.asarray(K), output_config))
    Sigma = np.asarray(
        np.eye(A.shape[0]) * output_config.estimator_initial_covariance,
        dtype=np.float64,
    )
    process = np.asarray(process_covariance(plant, output_config), dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()
    xhat = np.asarray(xhat0, dtype=np.float64).copy()
    batch_size = x.shape[0]
    x_seq = [x]
    xhat_seq = [xhat]
    y_seq = []
    u_seq = []
    for t in range(K.shape[0]):
        y_t = x @ H.T
        u_t = -xhat @ K[t].T
        xhat = xhat @ (A - B @ K[t] - gains[t] @ H).T + y_t @ gains[t].T
        x = x @ A.T + u_t @ B.T
        Sigma = (A - gains[t] @ H) @ Sigma @ A.T + process
        Sigma = 0.5 * (Sigma + Sigma.T)
        y_seq.append(y_t)
        u_seq.append(u_t)
        x_seq.append(x)
        xhat_seq.append(xhat)
    return {
        "x0": np.asarray(x0, dtype=np.float64),
        "xhat0": np.asarray(xhat0, dtype=np.float64),
        "x": np.stack(x_seq, axis=1),
        "x_hat": np.stack(xhat_seq, axis=1),
        "y": np.stack(y_seq, axis=1),
        "u": np.stack(u_seq, axis=1),
        "batch_size": np.asarray(batch_size),
    }


def _manifest_for_condition(
    *,
    condition: LinearRecurrentCondition,
    rollout: Any,
    reference_clean: Any,
    reference_clean_cost: float,
    candidate_cost: float,
    fit_metadata: dict[str, Any],
) -> BridgeRunManifest:
    candidate_actions = np.asarray(rollout.actions)
    reference_actions = np.asarray(reference_clean.u)[None, :, :]
    recurrence_diagnostics = dict(rollout.metadata["diagnostics"])
    recurrence_diagnostics.update(fit_metadata)
    hidden_states = np.asarray(rollout.hidden_states)
    augmented_states = np.concatenate([np.asarray(rollout.plant_states), hidden_states], axis=-1)
    components = build_standard_certificate_components(
        architecture="linear_recurrence",
        certificate_mode="augmented_linear",
        states=np.asarray(rollout.plant_states),
        augmented_states=augmented_states,
        candidate_actions=candidate_actions,
        reference_actions=reference_actions,
        optimizer_metadata=fit_metadata,
        recurrence_diagnostics=recurrence_diagnostics,
        state_label="plant_hidden_augmented_state",
        action_label="control",
    )
    by_name = {component.name: component for component in components}
    action_summary = by_name[STATE_WEIGHTED_ACTION_MISMATCH].summary
    mismatch = action_summary["mismatch_ratio_mean"]
    metrics = {
        "candidate_clean_cost": candidate_cost,
        "reference_clean_cost": reference_clean_cost,
        "objective_ratio_to_reference": candidate_cost / max(reference_clean_cost, 1e-12),
        "state_weighted_action_mismatch": mismatch,
        "aggregate_action_energy_mismatch": action_summary["aggregate_mismatch_ratio"],
        "recurrence_diagnostics": recurrence_diagnostics,
        "augmented_state_certificate": {
            "status": "augmented_linear_mode",
            "state": "plant_hidden_augmented_state",
            "component_statuses": {
                name: by_name[name].status for name in FORMAL_STATIC_GAIN_COMPONENTS
            },
            "reason": (
                "Action and visited-state components use z_t=[x_t; h_t]. "
                "Transition/value/Bellman components are explicit missing rows "
                "until a reference recurrent realization in the same augmented "
                "coordinates is available."
            ),
        },
    }
    lens_metadata = lens_metadata_for_condition(condition)
    spec = BridgeRunSpec(
        issue_id=ISSUE_ID,
        run_id=condition.run_id,
        objective=condition.objective,
        architecture="linear_recurrence",
        controller_label=condition.label,
        optimizer_label=fit_metadata["fit_method"],
        training_distribution=condition.training_distribution,  # type: ignore[arg-type]
        evaluation_lane="deterministic",
        reference_controller="analytical_lqr_kalman",
        seed=condition.seed,
        parameters={
            "initialization": condition.initialization,
            "lens_metadata": lens_metadata,
            "coverage_family": condition.coverage_family,
            "coverage_modes": condition.coverage_modes,
            "coverage_scale": condition.coverage_scale,
            "coverage_weight": condition.coverage_weight,
            "phase_time_feature_names": list(PHASE_FEATURE_NAMES),
            "hidden_dim": condition.hidden_dim,
            "disturbance_scale": condition.disturbance_scale,
            "n_train_steps": condition.n_train_steps,
            "imitation_steps": condition.imitation_steps,
            "fine_tune_steps": condition.fine_tune_steps,
            "learning_rate": condition.learning_rate,
            "stability_penalty": condition.stability_penalty,
        },
        notes=(
            "Trainable linear recurrence evaluated on clean output-feedback "
            "rollout. Certificate rows use augmented-linear mode over plant "
            "plus hidden state; transition/value/Bellman components remain "
            "explicit missing rows when same-coordinate reference recurrent "
            "sensitivities are unavailable."
        ),
    )
    return BridgeRunManifest(
        spec=spec,
        status="trainable_recurrence_augmented_certificate",
        arrays=rollout.array_specs(),
        metrics=metrics,
        certificate_components=components,
    )


def _failure_row(
    *,
    manifest: BridgeRunManifest,
    candidate_cost: float,
    reference_cost: float,
) -> dict[str, Any]:
    components = {component.name: component for component in manifest.certificate_components}
    action_summary = components[STATE_WEIGHTED_ACTION_MISMATCH].summary
    mismatch = action_summary.get("mismatch_ratio_mean")
    objective_ratio = candidate_cost / max(reference_cost, 1e-12)
    classification = classify_failure(
        objective_ratio=objective_ratio,
        learned_gradient_norm=None,
        reference_gradient_norm=None,
        certificate_mismatch_ratio=mismatch,
        subspace_decomposition=None,
    )
    return {
        "run_id": manifest.spec.run_id,
        "schema": "recurrence-compatible c45adde subset",
        "objective": {
            "learned_objective": candidate_cost,
            "reference_objective": reference_cost,
            "learned_to_reference_objective_ratio": objective_ratio,
            "learned_gradient_norm": None,
            "reference_gradient_norm": None,
            "source": "clean_closed_loop_quadratic_cost",
        },
        "certificate": {
            "state_weighted_action_mismatch": mismatch,
            "aggregate_action_energy_mismatch": action_summary.get("aggregate_mismatch_ratio"),
            "formal_static_gain_components": {
                name: components[name].status for name in FORMAL_STATIC_GAIN_COMPONENTS
            },
        },
        "subspace_decomposition": {
            "status": "not_applicable",
            "reason": "linear recurrent controller has no time-local static gain delta",
        },
        "classification": classification,
    }


def _phase_ablation(
    reference_clean: Any,
    plant: Any,
    K_ref: np.ndarray,
    output_config: OutputFeedbackConfig,
    phase: np.ndarray,
) -> dict[str, float]:
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    no_phase_obs = np.asarray(reference_clean.x)[:-1] @ H.T
    phase_obs = np.concatenate([no_phase_obs, phase], axis=-1)
    targets = np.asarray(reference_clean.u)
    no_phase_weights = _ridge_readout(no_phase_obs, targets, ridge=1e-6)
    phase_weights = _ridge_readout(phase_obs, targets, ridge=1e-6)
    no_phase_rmse = float(np.sqrt(np.mean((no_phase_obs @ no_phase_weights.T - targets) ** 2)))
    phase_rmse = float(np.sqrt(np.mean((phase_obs @ phase_weights.T - targets) ** 2)))
    return {
        "phase_training_action_rmse": phase_rmse,
        "no_phase_training_action_rmse": no_phase_rmse,
        "reference_gain_time_variation_norm": float(np.linalg.norm(np.diff(K_ref, axis=0))),
    }


def _ridge_readout(features: np.ndarray, targets: np.ndarray, *, ridge: float) -> np.ndarray:
    gram = features.T @ features
    rhs = features.T @ targets
    weights = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    return weights.T


def _quadratic_cost(schedule: Any, states: np.ndarray, actions: np.ndarray) -> float:
    x = np.asarray(states, dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64)
    Q = np.asarray(schedule.Q, dtype=np.float64)
    R = np.asarray(schedule.R, dtype=np.float64)
    Q_f = np.asarray(schedule.Q_f, dtype=np.float64)
    state_terms = np.einsum("ti,tij,tj->t", x[:-1], Q, x[:-1])
    control_terms = np.einsum("ti,tij,tj->t", u, R, u)
    terminal = float(x[-1] @ Q_f @ x[-1])
    return float(np.sum(state_terms) + np.sum(control_terms) + terminal)


def _result_text(rows: list[BridgeRunManifest]) -> str:
    by_id = {row.spec.run_id: row for row in rows}
    clean = by_id.get(make_bridge_run_id("linear_recurrent", "linrec_clean_scratch_baseline"))
    riccati = by_id.get(make_bridge_run_id("linear_recurrent", "linrec_riccati_eps_scratch"))
    imitation = by_id.get(make_bridge_run_id("linear_recurrent", "linrec_imitation_nominal"))
    if clean is None or riccati is None:
        return (
            "Retained trainable rows were materialized; no clean/Riccati comparison was available."
        )
    clean_ratio = clean.metrics["objective_ratio_to_reference"]
    riccati_ratio = riccati.metrics["objective_ratio_to_reference"]
    imitation_ratio = (
        None if imitation is None else imitation.metrics["objective_ratio_to_reference"]
    )
    if clean_ratio <= 1.5 and riccati_ratio <= 2.0:
        return (
            "The d_h=48 trainable linear recurrence shows nominal bridge "
            f"recovery (clean ratio {clean_ratio:.4g}, historical "
            f"`riccati_epsilon` alias/sinusoidal-process ratio {riccati_ratio:.4g}). "
            "The augmented-state certificate must still be inspected before this can "
            "be treated as a formal recurrent certificate pass."
        )
    scaffold = (
        "" if imitation_ratio is None else f"; imitation diagnostic ratio {imitation_ratio:.4g}"
    )
    return (
        "The d_h=48 trainable linear recurrence rows were materialized, but "
        "nominal bridge recovery is not established by the scratch reward rows "
        f"(clean ratio {clean_ratio:.4g}, historical `riccati_epsilon` "
        f"alias/sinusoidal-process ratio {riccati_ratio:.4g}"
        f"{scaffold})."
    )


def _normalize_disturbances(
    disturbances: np.ndarray | None,
    *,
    batch_size: int,
    disturbance_dim: int,
    horizon: int,
) -> np.ndarray:
    if disturbances is None:
        return np.zeros((batch_size, horizon, disturbance_dim), dtype=np.float64)
    values = np.asarray(disturbances, dtype=np.float64)
    if values.ndim == 2:
        values = np.broadcast_to(values[None, :, :], (batch_size, values.shape[0], values.shape[1]))
    if values.shape != (batch_size, horizon, disturbance_dim):
        raise ValueError(
            "disturbances must have shape "
            f"({horizon}, {disturbance_dim}) or {(batch_size, horizon, disturbance_dim)}"
        )
    return values.copy()


def _as_batch(array: np.ndarray, *, width: int, name: str) -> np.ndarray:
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (batch, {width})")
    return array


def _repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


__all__ = [
    "ARTIFACT_PATH",
    "ISSUE_ID",
    "MANIFEST_PATH",
    "NOTE_PATH",
    "LENS_KIND_CATALOG",
    "LinearRecurrentCondition",
    "default_conditions",
    "lens_metadata_for_condition",
    "materialize",
    "phase_time_features",
    "render_markdown",
    "rollout_phase_aware_linear_recurrent",
    "write_outputs",
]

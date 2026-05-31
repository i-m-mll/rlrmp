"""Phase-aware linear recurrent output-feedback bridge for issue 5e55f69.

This module keeps the GRU-facing bridge deliberately auditable: the controller
is a linear recurrence driven by delayed observations plus explicit phase/time
features.  It does not claim a formal static-gain certificate.  Instead, rows
carry standard action-mismatch and visited-subspace diagnostics, with formal
static-gain components marked ``not_applicable`` by the shared certificate
adapter.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
    build_standard_certificate_components,
)
from rlrmp.analysis.bridge_contracts import (
    BridgeRolloutBatch,
    BridgeRunManifest,
    BridgeRunSpec,
    make_bridge_run_id,
)
from rlrmp.analysis.bridge_controllers import (
    LinearRecurrentController,
    hidden_growth_diagnostics,
)
from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.failure_decomposition import classify_failure
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


ISSUE_ID = "5e55f69"
UMBRELLA_ID = "43e8728"
SUBSTRATE_ISSUE_ID = "4ded904"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
FAILURE_DECOMPOSITION_ISSUE_ID = "c45adde"

NOTE_PATH = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_linear_recurrent.md"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_linear_recurrent_manifest.json"
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


@dataclass(frozen=True)
class LinearRecurrentCondition:
    """One retained phase-aware linear recurrent bridge row."""

    label: str
    training_distribution: str
    initialization: str
    fit_reference_actions: bool
    coverage_family: str | None = None
    coverage_modes: int | None = None
    coverage_scale: float | None = None
    coverage_weight: float | None = None
    seed: int = 0
    recurrent_decay: float = 0.85
    ridge: float = 1e-6

    @property
    def run_id(self) -> str:
        """Stable run identifier for manifests and array keys."""

        return make_bridge_run_id("linear_recurrent", self.label)


def default_conditions(*, include_coverage: bool = True) -> tuple[LinearRecurrentCondition, ...]:
    """Return the planned no-coverage and selected coverage rows."""

    conditions = [
        LinearRecurrentCondition(
            label="no_coverage__scratch_seed_0",
            training_distribution="none",
            initialization="scratch_seed_0",
            fit_reference_actions=False,
        ),
        LinearRecurrentCondition(
            label="no_coverage__reference_replay",
            training_distribution="nominal",
            initialization="least_squares_reference_replay",
            fit_reference_actions=True,
        ),
    ]
    if include_coverage:
        conditions.extend(
            [
                LinearRecurrentCondition(
                    label="state_eigenspectrum_m4_s1_w0p1__reference_replay",
                    training_distribution="eigenspectrum_state",
                    initialization="least_squares_reference_replay",
                    fit_reference_actions=True,
                    coverage_family="state_eigenspectrum",
                    coverage_modes=4,
                    coverage_scale=1.0,
                    coverage_weight=0.1,
                ),
                LinearRecurrentCondition(
                    label="state_eigenspectrum_m4_s3_w0p1__reference_replay",
                    training_distribution="eigenspectrum_state",
                    initialization="least_squares_reference_replay",
                    fit_reference_actions=True,
                    coverage_family="state_eigenspectrum",
                    coverage_modes=4,
                    coverage_scale=3.0,
                    coverage_weight=0.1,
                ),
                LinearRecurrentCondition(
                    label="observer_error_state_m1_s0p3_w0p1__reference_replay",
                    training_distribution="observer_error",
                    initialization="least_squares_reference_replay",
                    fit_reference_actions=True,
                    coverage_family="observer_error_state",
                    coverage_modes=1,
                    coverage_scale=0.3,
                    coverage_weight=0.1,
                ),
            ]
        )
    return tuple(conditions)


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
    if controller.observation_dim != H.shape[0] + phase.shape[1]:
        raise ValueError("controller observation dimension does not match delayed obs + phase")

    states = _as_batch(np.asarray(x0, dtype=np.float64), width=A.shape[0], name="x0")
    batch_size = states.shape[0]
    eps = _normalize_disturbances(
        disturbances,
        batch_size=batch_size,
        disturbance_dim=Bw.shape[1],
        horizon=horizon,
    )
    hidden = np.broadcast_to(controller.initial_hidden, (batch_size, controller.hidden_dim)).copy()

    plant_states = [states]
    hidden_states = [hidden]
    observations = []
    actions = []
    for t in range(horizon):
        delayed = states @ H.T
        phase_t = np.broadcast_to(phase[t], (batch_size, phase.shape[1]))
        y_t = np.concatenate([delayed, phase_t], axis=-1)
        u_t = controller.action(hidden, y_t)
        states = states @ A.T + u_t @ B.T + eps[:, t, :] @ Bw.T
        hidden = controller.next_hidden(hidden, y_t)
        observations.append(y_t)
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

    component_counts: Counter[str] = Counter()
    for row in rows:
        for component in row.certificate_components:
            component_counts[f"{component.name}:{component.status}"] += 1

    summary = {
        "format": "rlrmp.output_feedback_linear_recurrent.v1",
        "issue": ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "source_issues": {
            "substrate": SUBSTRATE_ISSUE_ID,
            "standard_certificate": STANDARD_CERTIFICATE_ISSUE_ID,
            "failure_decomposition": FAILURE_DECOMPOSITION_ISSUE_ID,
        },
        "scope": (
            "Phase-aware linear recurrent output-feedback rows using delayed "
            "observations plus explicit polynomial phase/time inputs."
        ),
        "non_goals": (
            "No GRU training, robust/H-infinity training arm, formal game-card "
            "change, or affine tracker implementation."
        ),
        "runtime_seconds": time.perf_counter() - start,
        "diagnostics": {
            "phase_time_feature_names": list(PHASE_FEATURE_NAMES),
            "phase_ablation": ablation,
            "component_status_counts": dict(sorted(component_counts.items())),
            "retained_rows": [row.spec.run_id for row in rows],
        },
        "rows": [row.to_json_dict() for row in rows],
        "failure_decomposition": {
            "schema": "recurrence-compatible c45adde subset",
            "rows": failure_rows,
            "classification_counts": dict(
                sorted(
                    Counter(
                        row["classification"]["classification"] for row in failure_rows
                    ).items()
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
    manifest_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the tracked result note."""

    rows = [
        "| row | status | train dist | objective ratio | action mismatch | "
        "spectral radius | hidden max | failure |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        metrics = row["metrics"]
        recurrence = metrics["recurrence_diagnostics"]
        failure = next(
            item
            for item in summary["failure_decomposition"]["rows"]
            if item["run_id"] == row["spec"]["run_id"]
        )
        rows.append(
            "| "
            f"{row['spec']['run_id']} | {row['status']} | "
            f"{row['spec']['training_distribution']} | "
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

## Certificate Boundary

Formal static-gain components are not silently treated as passes for the
linear recurrence. They are explicit `not_applicable` rows because the
controller has recurrent hidden state and no global static gain over the
certificate state.

{"\n".join(component_rows)}

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
    observation_dim = H.shape[0] + phase.shape[1]
    action_dim = int(plant.B.shape[1])
    recurrent = condition.recurrent_decay * np.eye(observation_dim, dtype=np.float64)
    observation = (1.0 - condition.recurrent_decay) * np.eye(observation_dim, dtype=np.float64)
    base = LinearRecurrentController(
        recurrent_weights=recurrent,
        observation_weights=observation,
        readout_weights=np.zeros((action_dim, observation_dim), dtype=np.float64),
    )
    if not condition.fit_reference_actions:
        rng = np.random.default_rng(condition.seed)
        readout = 0.01 * rng.normal(size=(action_dim, observation_dim))
        feedthrough = 0.01 * rng.normal(size=(action_dim, observation_dim))
        controller = LinearRecurrentController(
            recurrent_weights=recurrent,
            observation_weights=observation,
            readout_weights=readout,
            feedthrough_weights=feedthrough,
        )
        return controller, {
            "fit_method": "scratch_random_linear_readout",
            "ridge": None,
            "training_action_rmse": None,
            "phase_feedthrough_norm": float(np.linalg.norm(feedthrough[:, -phase.shape[1] :])),
            "phase_observation_weight_norm": float(
                np.linalg.norm(observation[:, -phase.shape[1] :])
            ),
        }

    features, targets = _teacher_forced_features(
        base,
        plant=plant,
        reference_training=reference_training,
        phase=phase,
        output_config=output_config,
    )
    weights = _ridge_readout(features, targets, ridge=condition.ridge)
    hidden_dim = base.hidden_dim
    readout = weights[:, :hidden_dim]
    feedthrough = weights[:, hidden_dim:]
    prediction = features @ weights.T
    rmse = float(np.sqrt(np.mean((prediction - targets) ** 2)))
    controller = LinearRecurrentController(
        recurrent_weights=recurrent,
        observation_weights=observation,
        readout_weights=readout,
        feedthrough_weights=feedthrough,
    )
    return controller, {
        "fit_method": "teacher_forced_least_squares_reference_replay",
        "ridge": condition.ridge,
        "training_action_rmse": rmse,
        "phase_feedthrough_norm": float(np.linalg.norm(feedthrough[:, -phase.shape[1] :])),
        "phase_observation_weight_norm": float(np.linalg.norm(observation[:, -phase.shape[1] :])),
    }


def _teacher_forced_features(
    controller: LinearRecurrentController,
    *,
    plant: Any,
    reference_training: dict[str, np.ndarray],
    phase: np.ndarray,
    output_config: OutputFeedbackConfig,
) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    x = reference_training["x"]
    u = reference_training["u"]
    batch_size, horizon, _action_dim = u.shape
    hidden = np.broadcast_to(controller.initial_hidden, (batch_size, controller.hidden_dim)).copy()
    features = []
    targets = []
    for t in range(horizon):
        delayed = x[:, t, :] @ H.T
        y_t = np.concatenate(
            [delayed, np.broadcast_to(phase[t], (batch_size, phase.shape[1]))],
            axis=-1,
        )
        features.append(np.concatenate([hidden, y_t], axis=-1))
        targets.append(u[:, t, :])
        hidden = controller.next_hidden(hidden, y_t)
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _ridge_readout(features: np.ndarray, targets: np.ndarray, *, ridge: float) -> np.ndarray:
    gram = features.T @ features
    rhs = features.T @ targets
    weights = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    return weights.T


def _training_batch_for_condition(
    condition: LinearRecurrentCondition,
    *,
    plant: Any,
    K_ref: np.ndarray,
    x0: np.ndarray,
    output_config: OutputFeedbackConfig,
) -> dict[str, np.ndarray]:
    x0_batch, xhat0_batch = _coverage_initial_states(condition, plant=plant, x0=x0)
    return _reference_output_feedback_batch(
        plant=plant,
        K=K_ref,
        x0=x0_batch,
        xhat0=xhat0_batch,
        output_config=output_config,
    )


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
    components = build_standard_certificate_components(
        architecture="linear_recurrence",
        states=np.asarray(rollout.plant_states),
        candidate_actions=candidate_actions,
        reference_actions=reference_actions,
        optimizer_metadata=fit_metadata,
        recurrence_diagnostics=recurrence_diagnostics,
        state_label="clean_delayed_output_feedback_state",
        action_label="control",
    )
    by_name = {component.name: component for component in components}
    mismatch = by_name[STATE_WEIGHTED_ACTION_MISMATCH].summary["mismatch_ratio_mean"]
    metrics = {
        "candidate_clean_cost": candidate_cost,
        "reference_clean_cost": reference_clean_cost,
        "objective_ratio_to_reference": candidate_cost / max(reference_clean_cost, 1e-12),
        "state_weighted_action_mismatch": mismatch,
        "recurrence_diagnostics": recurrence_diagnostics,
        "formal_static_gain_certificate_boundary": {
            name: by_name[name].status for name in FORMAL_STATIC_GAIN_COMPONENTS
        },
    }
    spec = BridgeRunSpec(
        issue_id=ISSUE_ID,
        run_id=condition.run_id,
        objective="diagnostic",
        architecture="linear_recurrence",
        controller_label=condition.label,
        optimizer_label=fit_metadata["fit_method"],
        training_distribution=condition.training_distribution,  # type: ignore[arg-type]
        evaluation_lane="deterministic",
        reference_controller="analytical_lqr_kalman",
        seed=condition.seed,
        parameters={
            "initialization": condition.initialization,
            "coverage_family": condition.coverage_family,
            "coverage_modes": condition.coverage_modes,
            "coverage_scale": condition.coverage_scale,
            "coverage_weight": condition.coverage_weight,
            "phase_time_feature_names": list(PHASE_FEATURE_NAMES),
            "recurrent_decay": condition.recurrent_decay,
        },
        notes=(
            "Linear recurrence evaluated against clean output-feedback LQR "
            "reference actions. Formal static-gain certificate components are "
            "explicitly not applicable."
        ),
    )
    return BridgeRunManifest(
        spec=spec,
        status="recurrence_audit_not_formal_static_gain",
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
    mismatch = components[STATE_WEIGHTED_ACTION_MISMATCH].summary.get("mismatch_ratio_mean")
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
    training = {
        "x": np.asarray(reference_clean.x)[None, :, :],
        "u": np.asarray(reference_clean.u)[None, :, :],
    }
    H = np.asarray(delayed_observation_matrix(plant, output_config), dtype=np.float64)
    full_condition = LinearRecurrentCondition(
        label="ablation_phase",
        training_distribution="nominal",
        initialization="least_squares_reference_replay",
        fit_reference_actions=True,
    )
    controller, full_meta = _controller_for_condition(
        full_condition,
        plant=plant,
        reference_training=training | {"x0": training["x"][:, 0], "xhat0": training["x"][:, 0]},
        phase=phase,
        output_config=output_config,
    )
    del controller
    no_phase_obs = np.asarray(reference_clean.x)[:-1] @ H.T
    targets = np.asarray(reference_clean.u)
    weights = _ridge_readout(no_phase_obs, targets, ridge=1e-6)
    no_phase_rmse = float(np.sqrt(np.mean((no_phase_obs @ weights.T - targets) ** 2)))
    return {
        "phase_training_action_rmse": float(full_meta["training_action_rmse"] or 0.0),
        "no_phase_training_action_rmse": no_phase_rmse,
        "reference_gain_time_variation_norm": float(np.linalg.norm(np.diff(K_ref, axis=0))),
    }


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
    scratch = by_id.get(make_bridge_run_id("linear_recurrent", "no_coverage__scratch_seed_0"))
    replay = by_id.get(make_bridge_run_id("linear_recurrent", "no_coverage__reference_replay"))
    if scratch is None or replay is None:
        return "Retained rows were materialized; no scratch/reference comparison was available."
    scratch_ratio = scratch.metrics["objective_ratio_to_reference"]
    replay_ratio = replay.metrics["objective_ratio_to_reference"]
    if replay_ratio < scratch_ratio:
        return (
            "The phase-aware least-squares reference-replay recurrence improves "
            f"the no-coverage clean objective ratio versus scratch "
            f"({replay_ratio:.4g} vs {scratch_ratio:.4g}), but remains an "
            "audit row rather than a formal static-gain certificate pass."
        )
    return (
        "The phase-aware recurrence rows were materialized, but no-coverage "
        f"reference replay did not improve clean objective ratio over scratch "
        f"({replay_ratio:.4g} vs {scratch_ratio:.4g})."
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
    "LinearRecurrentCondition",
    "default_conditions",
    "materialize",
    "phase_time_features",
    "render_markdown",
    "rollout_phase_aware_linear_recurrent",
    "write_outputs",
]

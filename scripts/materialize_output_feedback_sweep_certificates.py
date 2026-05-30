"""Apply bridge standard-certificate rows to 7a459bb sweep summaries.

The recent coverage/noise sweep manifests intentionally keep tracked outputs
small.  This materializer uses the saved no-coverage arrays where available and
records explicit missing component rows where compact sweep manifests only
preserve scalar summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    OPTIMIZER_METADATA,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
    VISITED_SUBSPACE_DIAGNOSTICS,
    build_standard_certificate_components,
    missing_component,
)
from rlrmp.analysis.bridge_contracts import (
    BridgeCertificateComponent,
    BridgeRunManifest,
    BridgeRunSpec,
    make_bridge_run_id,
)
from rlrmp.analysis.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    kalman_estimator_joint_matrices,
    rollout_with_kalman_estimator,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE_ID = "7a459bb"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
SOURCE_ARTIFACT = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "output_feedback_rollout_recovery"
    / "output_feedback_rollout_recovery.npz"
)
SOURCE_MANIFESTS = {
    "no_coverage": REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_rollout_recovery_manifest.json",
    "initial_state": REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_initial_state_variability_sweep_manifest.json",
    "process_noise": REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_process_noise_sweep_manifest.json",
    "eigenspectrum": REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_eigenspectrum_coverage_sweep_manifest.json",
}
NOTE_PATH = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_sweep_standard_certificates.md"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_sweep_standard_certificates_manifest.json"
)

BEHAVIORAL_ACTION_SIDECAR = "behavioral_action_sidecar"
DETERMINISTIC_AUDIT_SIDECAR = "deterministic_exact_l2_and_gamma_sidecar"
GAIN_DIAGNOSTIC_SIDECAR = "gain_diagnostic_sidecar"
ROLL_OUT_BEHAVIOR_SIDECAR = "rollout_behavior_sidecar"

STANDARD_COMPONENTS = (
    STATE_WEIGHTED_ACTION_MISMATCH,
    VISITED_SUBSPACE_DIAGNOSTICS,
    OPTIMIZER_METADATA,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    VALUE_POLICY_GAP,
    BELLMAN_HESSIAN_RESIDUAL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize()
    write_result(result, note_path=args.note_output, manifest_path=args.manifest_output)
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")


def materialize() -> dict[str, Any]:
    """Return the deterministic partial-certificate application bundle."""

    source_manifests = {name: _read_json(path) for name, path in SOURCE_MANIFESTS.items()}
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    output_config = OutputFeedbackConfig()
    with np.load(SOURCE_ARTIFACT) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    _ensure_reference_under_epsilon_arrays(arrays, reference, output_config)

    rows: list[BridgeRunManifest] = []
    rows.extend(
        _saved_no_coverage_rows(source_manifests["no_coverage"], arrays, reference, output_config)
    )
    rows.extend(_initial_state_rows(source_manifests["initial_state"]))
    rows.extend(_process_noise_rows(source_manifests["process_noise"]))
    rows.extend(_eigenspectrum_rows(source_manifests["eigenspectrum"]))

    row_dicts = [row.to_json_dict() for row in rows]
    status_counts = Counter(row.status for row in rows)
    component_counts: Counter[str] = Counter()
    for row in rows:
        for component in row.certificate_components:
            component_counts[f"{component.name}:{component.status}"] += 1

    return {
        "format": "rlrmp.output_feedback_sweep_standard_certificates.v1",
        "issue": ISSUE_ID,
        "standard_certificate_issue": STANDARD_CERTIFICATE_ISSUE_ID,
        "source_manifests": {name: _repo_relative(path) for name, path in SOURCE_MANIFESTS.items()},
        "source_artifacts": {"no_coverage_arrays": _repo_relative(SOURCE_ARTIFACT)},
        "summary": {
            "n_rows": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "component_status_counts": dict(sorted(component_counts.items())),
            "full_standard_certificate_rows": [
                row.spec.run_id for row in rows if row.status == "full_standard_certificate"
            ],
            "partial_or_sidecar_rows": [
                row.spec.run_id for row in rows if row.status != "full_standard_certificate"
            ],
        },
        "result": (
            "Full standard-certificate components are available only for the saved "
            "no-coverage/reference rows backed by the rollout-recovery NPZ; those "
            "rows are now evaluated on both nominal-clean and Riccati-epsilon "
            "state lenses. The "
            "recent compact sweep manifests provide deterministic behavioral, "
            "optimizer, gain-diagnostic, exact-L2/gamma, and coverage-rank sidecars "
            "where those summaries were saved, but they do not include enough raw "
            "gains, rollout state/action arrays, covariances, or value matrices to "
            "recompute every standard component."
        ),
        "rows": row_dicts,
    }


def _saved_no_coverage_rows(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
) -> list[BridgeRunManifest]:
    fit_by_label = {fit["label"]: fit for fit in manifest["fits"]}
    selected = (
        ("analytical_lqr_reference", None),
        ("strong_optimizer_whitened__scratch", fit_by_label["strong_optimizer_whitened__scratch"]),
        (
            "strong_optimizer_whitened__bellman_init",
            fit_by_label["strong_optimizer_whitened__bellman_init"],
        ),
    )
    rows = []
    evaluation_lenses = (
        ("nominal_clean", "clean", "no_coverage_nominal_clean"),
        ("riccati_epsilon_response", "under_eps", "no_coverage_riccati_epsilon_response"),
    )
    for label, fit in selected:
        is_reference = fit is None
        for lens_label, array_suffix, evaluation_distribution in evaluation_lenses:
            array_prefix = f"lqr_{array_suffix}" if is_reference else f"{label}_{array_suffix}"
            components = list(
                _full_standard_components(
                    arrays=arrays,
                    reference=reference,
                    output_config=output_config,
                    array_prefix=array_prefix,
                    candidate_gain=np.asarray(reference.lqr_solution.K)
                    if is_reference
                    else arrays[f"{label}_K"],
                    optimizer_metadata=None if is_reference else _optimizer_metadata(fit),
                    state_label=f"{evaluation_distribution}_coupled_state",
                    action_state_label=f"{evaluation_distribution}_estimated_state",
                    architecture="reference" if is_reference else "time_constrained_free_gain",
                )
            )
            if is_reference:
                components = _replace_component(
                    components,
                    BridgeCertificateComponent.not_applicable(
                        OPTIMIZER_METADATA,
                        "analytical LQR reference has no optimizer metadata",
                    ),
                )
            components.extend(_diagnostic_sidecars(fit=fit, reference_manifest=manifest))
            spec = BridgeRunSpec(
                issue_id=ISSUE_ID,
                run_id=make_bridge_run_id("no_coverage", label, lens_label),
                objective="optimal",
                architecture="reference" if is_reference else "time_constrained_free_gain",
                controller_label=label,
                optimizer_label="analytical"
                if is_reference
                else "lbfgsb_strong_optimizer_whitened",
                training_distribution="none" if is_reference else "nominal",
                evaluation_lane="deterministic",
                reference_controller="analytical_lqr_kalman",
                gamma_factor=manifest["diagnostics"]["gamma_factor"],
                parameters={
                    "evaluation_distribution": evaluation_distribution,
                    "distribution_family": "no-coverage/reference",
                    "source_manifest": _repo_relative(SOURCE_MANIFESTS["no_coverage"]),
                    "source_artifact": _repo_relative(SOURCE_ARTIFACT),
                },
                notes=(
                    "Full standard bundle computed from saved gains and reconstructed "
                    f"{lens_label} rollout arrays."
                ),
            )
            rows.append(
                BridgeRunManifest(
                    spec=spec,
                    status="full_standard_certificate",
                    metrics=_no_coverage_metrics(label, fit, manifest)
                    | {"certificate_evaluation_lens": lens_label},
                    artifacts={"source_npz": _repo_relative(SOURCE_ARTIFACT)},
                    certificate_components=tuple(components),
                )
            )
    return rows


def _ensure_reference_under_epsilon_arrays(
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
) -> None:
    if "lqr_under_eps_x" in arrays:
        return
    rollout = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        jnp.asarray(arrays["lqr_clean_x"][0], dtype=jnp.float64),
        jnp.asarray(arrays["riccati_epsilon"], dtype=jnp.float64),
        output_config,
    )
    arrays["lqr_under_eps_x"] = np.asarray(rollout.x)
    arrays["lqr_under_eps_x_hat"] = np.asarray(rollout.x_hat)


def _initial_state_rows(manifest: dict[str, Any]) -> list[BridgeRunManifest]:
    rows = []
    for cell in manifest["cells"]:
        fit = cell["fits"][0]
        components = _summary_fit_components(
            fit=fit,
            available_visited=_visited_summary_component(
                state_label="initial_state_training_ensemble",
                source="summary_manifest.initial_state_ensemble",
                diagnostics=cell["diagnostics"]["initial_state_ensemble"],
                extra={
                    "scale_factor": cell["scale_factor"],
                    "basis_scale": cell["basis_scale"],
                    "random_state_scale": cell["random_state_scale"],
                    "n_random_states": cell["n_random_states"],
                    "reach_weight": cell["reach_weight"],
                },
            ),
            missing_reason=(
                "initial-state sweep manifest stores scalar summaries only; fitted gains "
                "and rollout state/action arrays were not saved for this cell"
            ),
        )
        spec = BridgeRunSpec(
            issue_id=ISSUE_ID,
            run_id=make_bridge_run_id("initial_state_coverage", cell["label"], fit["label"]),
            objective="optimal",
            architecture="time_constrained_free_gain",
            controller_label=fit["label"],
            optimizer_label="lbfgsb_strong_optimizer_whitened",
            training_distribution="synthetic_initial_state",
            evaluation_lane="deterministic",
            reference_controller="analytical_lqr_kalman",
            gamma_factor=fit.get("condition", {}).get("gamma_factor"),
            parameters={
                "evaluation_distribution": "deterministic_clean_rollout_summary",
                "distribution_family": "initial-state coverage",
                "scale_factor": cell["scale_factor"],
                "basis_scale": cell["basis_scale"],
                "random_state_scale": cell["random_state_scale"],
                "source_manifest": _repo_relative(SOURCE_MANIFESTS["initial_state"]),
            },
            notes="Partial certificate row from compact initial-state coverage manifest.",
        )
        rows.append(
            BridgeRunManifest(
                spec=spec,
                status="partial_summary_certificate",
                metrics=_fit_metrics(fit),
                artifacts={"source_manifest": _repo_relative(SOURCE_MANIFESTS["initial_state"])},
                certificate_components=components,
            )
        )
    return rows


def _process_noise_rows(manifest: dict[str, Any]) -> list[BridgeRunManifest]:
    rows = []
    for cell in manifest["cells"]:
        for evaluation in cell["evaluations"]:
            is_reference = evaluation["label"] == "analytical_lqr_reference"
            components = _process_summary_components(
                evaluation=evaluation,
                missing_reason=(
                    "process-noise sweep manifest stores Monte Carlo summary statistics "
                    "but not sampled stochastic state/action trajectories or covariances"
                ),
                is_reference=is_reference,
            )
            spec = BridgeRunSpec(
                issue_id=ISSUE_ID,
                run_id=make_bridge_run_id(
                    "process_noise_stochastic",
                    cell["label"],
                    evaluation["label"],
                ),
                objective="optimal",
                architecture="reference" if is_reference else "time_constrained_free_gain",
                controller_label=evaluation["label"],
                optimizer_label="analytical"
                if is_reference
                else evaluation.get("source", "unknown"),
                training_distribution="none" if is_reference else "nominal",
                evaluation_lane="released_stochastic",
                reference_controller="analytical_lqr_kalman",
                seed=cell["monte_carlo"]["seed"],
                gamma_factor=None,
                parameters={
                    "evaluation_distribution": "process-noise stochastic",
                    "distribution_family": "process-noise stochastic",
                    "process_covariance_scale": cell["process_covariance_scale"],
                    "n_trials": cell["monte_carlo"]["n_trials"],
                    "motor_covariance_scale": cell["monte_carlo"]["motor_covariance_scale"],
                    "signal_dependent_scale": cell["monte_carlo"]["signal_dependent_scale"],
                    "source_manifest": _repo_relative(SOURCE_MANIFESTS["process_noise"]),
                },
                notes=(
                    "Sidecar-only stochastic evaluation row; formal standard components "
                    "need sampled trajectories/covariances that this manifest does not save."
                ),
            )
            rows.append(
                BridgeRunManifest(
                    spec=spec,
                    status="sidecar_only_missing_inputs",
                    metrics=_process_metrics(evaluation),
                    artifacts={
                        "source_manifest": _repo_relative(SOURCE_MANIFESTS["process_noise"])
                    },
                    certificate_components=components,
                )
            )
    return rows


def _eigenspectrum_rows(manifest: dict[str, Any]) -> list[BridgeRunManifest]:
    rows = []
    coverage_by_label = manifest["diagnostics"]["eigenspectrum_coverage"]
    for fit in manifest["fits"]:
        coverage = fit["condition"]["eigenspectrum_coverage"]
        condition_label = fit["condition"]["label"]
        family = f"eigenspectrum {coverage['objective']} coverage"
        components = _summary_fit_components(
            fit=fit,
            available_visited=_visited_summary_component(
                state_label=f"eigenspectrum_{coverage['objective']}_coverage_estimated_state",
                source="summary_manifest.eigenspectrum_coverage.xhat_coverage",
                diagnostics=coverage_by_label[condition_label]["xhat_coverage"],
                extra={
                    "objective": coverage["objective"],
                    "n_modes": coverage["n_modes"],
                    "scale": coverage["scale"],
                    "weight": coverage["weight"],
                    "n_trajectories": coverage_by_label[condition_label]["n_trajectories"],
                    "n_state_samples_for_diagnostics": coverage_by_label[condition_label][
                        "n_state_samples_for_diagnostics"
                    ],
                },
            ),
            missing_reason=(
                "eigenspectrum sweep manifest stores scalar summaries and coverage-rank "
                "diagnostics only; fitted gains and rollout state/action arrays were not "
                "saved for this cell"
            ),
        )
        spec = BridgeRunSpec(
            issue_id=ISSUE_ID,
            run_id=make_bridge_run_id("eigenspectrum", coverage["objective"], fit["label"]),
            objective="optimal",
            architecture="time_constrained_free_gain",
            controller_label=fit["label"],
            optimizer_label="lbfgsb_strong_optimizer_whitened",
            training_distribution=f"eigenspectrum_{coverage['objective']}",
            evaluation_lane="deterministic",
            reference_controller="analytical_lqr_kalman",
            gamma_factor=manifest["diagnostics"]["gamma_factor"],
            parameters={
                "evaluation_distribution": family,
                "distribution_family": family,
                "n_modes": coverage["n_modes"],
                "scale": coverage["scale"],
                "weight": coverage["weight"],
                "source_manifest": _repo_relative(SOURCE_MANIFESTS["eigenspectrum"]),
            },
            notes=f"Partial certificate row from compact {family} manifest.",
        )
        rows.append(
            BridgeRunManifest(
                spec=spec,
                status="partial_summary_certificate",
                metrics=_fit_metrics(fit),
                artifacts={"source_manifest": _repo_relative(SOURCE_MANIFESTS["eigenspectrum"])},
                certificate_components=components,
            )
        )
    return rows


def _full_standard_components(
    *,
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
    array_prefix: str,
    candidate_gain: np.ndarray,
    optimizer_metadata: dict[str, Any] | None,
    state_label: str,
    action_state_label: str,
    architecture: str,
) -> tuple[BridgeCertificateComponent, ...]:
    plant = reference.plant
    schedule = reference.schedule
    reference_gain = np.asarray(reference.lqr_solution.K)
    x = _saved_array(arrays, array_prefix, "x")
    x_hat = _saved_array(arrays, array_prefix, "x_hat")
    coupled_states = np.concatenate([x, x_hat], axis=-1)[None, :, :]
    action_states = x_hat[None, :, :]
    candidate_transition = np.asarray(
        kalman_estimator_joint_matrices(plant, jnp.asarray(candidate_gain), output_config)[0]
    )
    reference_transition = np.asarray(
        kalman_estimator_joint_matrices(plant, jnp.asarray(reference_gain), output_config)[0]
    )
    candidate_values = _policy_value_matrices(schedule, candidate_gain, candidate_transition)
    reference_values = _policy_value_matrices(schedule, reference_gain, reference_transition)
    bellman_hessian = _bellman_hessian(schedule, plant, reference.lqr_solution.P)
    return build_standard_certificate_components(
        architecture=architecture,  # type: ignore[arg-type]
        states=coupled_states,
        action_states=action_states,
        candidate_gain=candidate_gain,
        reference_gain=reference_gain,
        action_weight=np.asarray(schedule.R),
        candidate_transition=candidate_transition,
        reference_transition=reference_transition,
        candidate_value_matrices=candidate_values,
        reference_value_matrices=reference_values,
        bellman_hessian=bellman_hessian,
        optimizer_metadata=optimizer_metadata,
        state_label=state_label,
        action_state_label=action_state_label,
        action_label="control",
    )


def _saved_array(arrays: dict[str, np.ndarray], prefix: str, suffix: str) -> np.ndarray:
    key = f"{prefix}_{suffix}" if prefix != "lqr_clean" else f"lqr_clean_{suffix}"
    return arrays[key]


def _joint_cost_matrices(schedule: Any, gains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(schedule.Q.shape[-1])
    zeros = np.zeros((n, n), dtype=float)
    stage = []
    for q_t, r_t, k_t in zip(schedule.Q, schedule.R, gains, strict=True):
        state_block = np.asarray(q_t, dtype=float)
        control_block = (
            np.asarray(k_t, dtype=float).T
            @ np.asarray(r_t, dtype=float)
            @ np.asarray(k_t, dtype=float)
        )
        stage.append(np.block([[state_block, zeros], [zeros, control_block]]))
    terminal = np.block([[np.asarray(schedule.Q_f, dtype=float), zeros], [zeros, zeros]])
    return np.asarray(stage), np.asarray(terminal)


def _policy_value_matrices(
    schedule: Any,
    gains: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    stage, terminal = _joint_cost_matrices(schedule, gains)
    values = [terminal]
    next_value = terminal
    for a_t, q_t in zip(transition[::-1], stage[::-1], strict=True):
        next_value = q_t + a_t.T @ next_value @ a_t
        next_value = 0.5 * (next_value + next_value.T)
        values.append(next_value)
    return np.asarray(list(reversed(values)))


def _bellman_hessian(schedule: Any, plant: Any, p_values: np.ndarray) -> np.ndarray:
    p_next = np.asarray(p_values[1:], dtype=float)
    b = np.asarray(plant.B, dtype=float)
    return np.asarray(schedule.R, dtype=float) + np.einsum("iu,tij,jv->tuv", b, p_next, b)


def _summary_fit_components(
    *,
    fit: dict[str, Any],
    available_visited: BridgeCertificateComponent,
    missing_reason: str,
) -> tuple[BridgeCertificateComponent, ...]:
    components = [
        missing_component(STATE_WEIGHTED_ACTION_MISMATCH, missing_reason),
        available_visited,
        BridgeCertificateComponent.available(OPTIMIZER_METADATA, **_optimizer_metadata(fit)),
        missing_component(CLOSED_LOOP_TRANSITION_MISMATCH, missing_reason),
        missing_component(VALUE_POLICY_GAP, missing_reason),
        missing_component(BELLMAN_HESSIAN_RESIDUAL, missing_reason),
    ]
    components.extend(_diagnostic_sidecars(fit=fit, reference_manifest=None))
    return tuple(components)


def _process_summary_components(
    *,
    evaluation: dict[str, Any],
    missing_reason: str,
    is_reference: bool,
) -> tuple[BridgeCertificateComponent, ...]:
    optimizer = (
        BridgeCertificateComponent.not_applicable(
            OPTIMIZER_METADATA,
            "analytical reference row has no optimizer metadata",
        )
        if is_reference
        else missing_component(
            OPTIMIZER_METADATA,
            "process-noise evaluation manifest does not carry source optimizer metadata",
        )
    )
    gain = evaluation.get("deterministic_gain_relative_error")
    gain_component = (
        BridgeCertificateComponent.available(
            GAIN_DIAGNOSTIC_SIDECAR,
            gain_relative_error=gain,
            diagnostic_only=True,
            gate="not_used_as_certificate_gate",
        )
        if gain is not None
        else missing_component(
            GAIN_DIAGNOSTIC_SIDECAR,
            "source evaluation did not include a deterministic gain-relative-error scalar",
        )
    )
    return (
        missing_component(STATE_WEIGHTED_ACTION_MISMATCH, missing_reason),
        missing_component(VISITED_SUBSPACE_DIAGNOSTICS, missing_reason),
        optimizer,
        missing_component(CLOSED_LOOP_TRANSITION_MISMATCH, missing_reason),
        missing_component(VALUE_POLICY_GAP, missing_reason),
        missing_component(BELLMAN_HESSIAN_RESIDUAL, missing_reason),
        BridgeCertificateComponent.available(
            BEHAVIORAL_ACTION_SIDECAR,
            cost_ratio_to_reference_mean=evaluation.get("cost_ratio_to_reference_mean"),
            action_mismatch_to_reference_mean=evaluation.get("action_mismatch_to_reference_mean"),
            action_mismatch_to_reference_std=evaluation.get("action_mismatch_to_reference_std"),
            evaluation_distribution="process-noise stochastic",
        ),
        BridgeCertificateComponent.available(
            DETERMINISTIC_AUDIT_SIDECAR,
            exact_l2_cost_ratio_to_lqr=evaluation.get("deterministic_exact_l2_cost_ratio_to_lqr"),
            exact_l2_cost_ratio_to_hinf=evaluation.get("deterministic_exact_l2_cost_ratio_to_hinf"),
            lambda_over_gamma_squared=evaluation.get("deterministic_lambda_over_gamma_squared"),
            gamma_penalized_feasible=evaluation.get("deterministic_gamma_penalized_feasible"),
        ),
        gain_component,
        BridgeCertificateComponent.available(
            ROLL_OUT_BEHAVIOR_SIDECAR,
            peak_forward_velocity_mean=evaluation.get("peak_forward_velocity_mean"),
            terminal_error_mean=evaluation.get("terminal_error_mean"),
            control_effort_mean=evaluation.get("control_effort_mean"),
        ),
    )


def _diagnostic_sidecars(
    *,
    fit: dict[str, Any] | None,
    reference_manifest: dict[str, Any] | None,
) -> list[BridgeCertificateComponent]:
    if fit is None:
        assert reference_manifest is not None
        diagnostics = reference_manifest["diagnostics"]
        return [
            BridgeCertificateComponent.available(
                BEHAVIORAL_ACTION_SIDECAR,
                clean_action_mismatch_ratio=0.0,
                under_epsilon_action_mismatch_ratio=0.0,
                clean_cost=diagnostics["lqr_clean"]["cost"],
                under_epsilon_cost=diagnostics["lqr_under_riccati_epsilon"]["cost"],
                evaluation_distribution="no_coverage/reference",
            ),
            BridgeCertificateComponent.available(
                DETERMINISTIC_AUDIT_SIDECAR,
                exact_l2_cost_ratio_to_lqr=1.0,
                exact_l2_cost_ratio_to_hinf=(
                    diagnostics["analytical_exact_audits"]["lqr"]["cost"]
                    / diagnostics["analytical_exact_audits"]["hinf"]["cost"]
                ),
                lambda_over_gamma_squared=diagnostics["analytical_exact_audits"]["lqr"][
                    "lambda_over_gamma_squared"
                ],
                gamma_penalized_feasible=diagnostics["analytical_exact_audits"]["lqr"][
                    "gamma_penalized_feasible"
                ],
            ),
            BridgeCertificateComponent.available(
                GAIN_DIAGNOSTIC_SIDECAR,
                gain_relative_error=0.0,
                diagnostic_only=True,
                gate="not_used_as_certificate_gate",
            ),
            BridgeCertificateComponent.available(
                ROLL_OUT_BEHAVIOR_SIDECAR,
                peak_forward_velocity=diagnostics["lqr_clean"]["rollout"]["peak_forward_velocity"],
                terminal_position_error_m=diagnostics["lqr_clean"]["rollout"][
                    "terminal_position_error_m"
                ],
                control_effort=diagnostics["lqr_clean"]["rollout"]["control_effort"],
            ),
        ]
    return [
        BridgeCertificateComponent.available(
            BEHAVIORAL_ACTION_SIDECAR,
            clean_action_mismatch_ratio=fit.get("clean_action_mismatch_ratio"),
            under_epsilon_action_mismatch_ratio=fit.get("under_epsilon_action_mismatch_ratio"),
            clean_cost=fit.get("clean_cost"),
            under_epsilon_cost_ratio_to_lqr=fit.get("under_epsilon_cost_ratio_to_lqr"),
            evaluation_distribution="deterministic clean plus Riccati-epsilon sidecar",
        ),
        BridgeCertificateComponent.available(
            DETERMINISTIC_AUDIT_SIDECAR,
            exact_l2_cost_ratio_to_lqr=fit.get("exact_l2_cost_ratio_to_lqr"),
            exact_l2_cost_ratio_to_hinf=fit.get("exact_l2_cost_ratio_to_hinf"),
            lambda_over_gamma_squared=fit.get("gamma_penalized_lambda_over_gamma_squared"),
            gamma_penalized_feasible=fit.get("gamma_penalized_feasible"),
        ),
        BridgeCertificateComponent.available(
            GAIN_DIAGNOSTIC_SIDECAR,
            gain_relative_error=fit.get("gain_relative_error"),
            diagnostic_only=True,
            gate="not_used_as_certificate_gate",
        ),
        BridgeCertificateComponent.available(
            ROLL_OUT_BEHAVIOR_SIDECAR,
            peak_forward_velocity=fit.get("clean_rollout", {}).get("peak_forward_velocity"),
            terminal_position_error_m=fit.get("clean_rollout", {}).get("terminal_position_error_m"),
            control_effort=fit.get("clean_rollout", {}).get("control_effort"),
        ),
    ]


def _visited_summary_component(
    *,
    state_label: str,
    source: str,
    diagnostics: dict[str, Any],
    extra: dict[str, Any],
) -> BridgeCertificateComponent:
    return BridgeCertificateComponent.available(
        VISITED_SUBSPACE_DIAGNOSTICS,
        state_label=state_label,
        source=source,
        trace=diagnostics.get("trace"),
        effective_rank_entropy=diagnostics.get("effective_rank_entropy"),
        effective_rank_participation=diagnostics.get("effective_rank_participation"),
        numerical_rank=diagnostics.get("numerical_rank"),
        condition_number=diagnostics.get("condition_number"),
        limitation="summary-only rank diagnostics; no gain-error decomposition without saved gains",
        **extra,
    )


def _replace_component(
    components: list[BridgeCertificateComponent],
    replacement: BridgeCertificateComponent,
) -> list[BridgeCertificateComponent]:
    return [
        replacement if component.name == replacement.name else component for component in components
    ]


def _optimizer_metadata(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_status": fit.get("optimizer_status"),
        "optimizer_success": fit.get("optimizer_success"),
        "n_iterations": fit.get("n_iterations"),
        "n_function_evaluations": fit.get("n_function_evaluations"),
        "gradient_norm_initial": fit.get("gradient_norm_initial"),
        "gradient_norm_final": fit.get("gradient_norm_final"),
        "projected_gradient_norm_final": fit.get("projected_gradient_norm_final"),
    }


def _no_coverage_metrics(
    label: str,
    fit: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if fit is not None:
        return _fit_metrics(fit)
    diagnostics = manifest["diagnostics"]
    return {
        "label": label,
        "clean_cost": diagnostics["lqr_clean"]["cost"],
        "clean_action_mismatch_ratio": 0.0,
        "gain_relative_error": 0.0,
        "exact_l2_cost_ratio_to_lqr": 1.0,
        "gamma_penalized_lambda_over_gamma_squared": diagnostics["analytical_exact_audits"]["lqr"][
            "lambda_over_gamma_squared"
        ],
        "clean_rollout": diagnostics["lqr_clean"]["rollout"],
    }


def _fit_metrics(fit: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "label",
        "objective_ratio_to_reference",
        "gain_relative_error",
        "clean_action_mismatch_ratio",
        "under_epsilon_action_mismatch_ratio",
        "exact_l2_cost_ratio_to_lqr",
        "exact_l2_cost_ratio_to_hinf",
        "gamma_penalized_lambda_over_gamma_squared",
        "gamma_penalized_feasible",
        "n_iterations",
        "optimizer_success",
        "clean_rollout",
    )
    return {key: fit.get(key) for key in keys}


def _process_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "label",
        "source",
        "cost_ratio_to_reference_mean",
        "action_mismatch_to_reference_mean",
        "deterministic_gain_relative_error",
        "deterministic_exact_l2_cost_ratio_to_lqr",
        "deterministic_lambda_over_gamma_squared",
        "peak_forward_velocity_mean",
        "terminal_error_mean",
    )
    return {key: evaluation.get(key) for key in keys}


def write_result(
    result: dict[str, Any],
    *,
    note_path: Path = NOTE_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    mkdir_p(note_path.parent)
    note_path.write_text(render_markdown(result), encoding="utf-8")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(result: dict[str, Any]) -> str:
    rows = result["rows"]
    category_rows = _category_rows(rows)
    key_rows = _key_rows(rows)
    return f"""# Output-Feedback Sweep Standard Certificates

Issue: `{ISSUE_ID}`. Standard certificate cross-reference: `{STANDARD_CERTIFICATE_ISSUE_ID}`.

This materialization applies the bridge standard-certificate row contract to the
recent output-feedback coverage/noise sweep outputs. It is intentionally a
partial certificate report: the saved no-coverage/reference artifact contains
gains plus nominal-clean/Riccati-epsilon rollout arrays, while the newer sweep
manifests mostly preserve small scalar summaries.

Result: {result["result"]}

Raw gain recovery is reported only in `gain_diagnostic_sidecar` rows and is not
used as the certificate gate.

## Source Inputs

- No-coverage/reference manifest: `{result["source_manifests"]["no_coverage"]}`
- Saved no-coverage arrays: `{result["source_artifacts"]["no_coverage_arrays"]}`
- Initial-state coverage manifest: `{result["source_manifests"]["initial_state"]}`
- Process-noise stochastic manifest: `{result["source_manifests"]["process_noise"]}`
- Eigenspectrum coverage manifest: `{result["source_manifests"]["eigenspectrum"]}`

## Availability by Distribution

| distribution family | rows | standard state/action | transition/value/Bellman | available sidecars |
|---|---:|---|---|---|
{category_rows}

## Key Rows

| run | status | distribution | objective ratio | gain sidecar | action mismatch | exact L2 sidecar | lambda/gamma^2 |
|---|---|---|---:|---:|---:|---:|---:|
{key_rows}

## Missing Components

- Initial-state coverage rows lack saved fitted gains and rollout state/action
  arrays for each scale, so formal state-weighted action mismatch, closed-loop
  transition mismatch, value-policy gap, and Bellman-Hessian residual are
  marked `missing`. The tracked manifest still records the saved training
  ensemble effective-rank diagnostics and deterministic behavioral/exact-audit
  sidecars.
- Process-noise stochastic rows are evaluation summaries. They expose stochastic
  cost/action sidecars and deterministic exact-L2/gamma sidecars, but not the
  sampled state/action trajectories or covariances required for the formal
  standard components.
- Eigenspectrum trajectory/state coverage rows expose coverage-induced xhat
  rank diagnostics by objective/mode/scale, plus deterministic behavioral and
  exact-audit sidecars. They do not include per-row fitted gains or rollout
  arrays, so the formal linear components are marked `missing`.

## Verdict

The standard certificate application does not rescue the bridge. The only rows
with full component availability are the saved no-coverage/reference rows on
nominal-clean and Riccati-epsilon evaluation lenses. The recent initial-state,
process-noise, and eigenspectrum rows remain partial/sidecar-only from current
tracked artifacts, and their saved sidecars continue to show behaviorally close
but certificate-poor from-scratch recovery.
"""


def _category_rows(rows: list[dict[str, Any]]) -> str:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = row["spec"]["parameters"]["distribution_family"]
        by_family.setdefault(family, []).append(row)

    lines = []
    for family in sorted(by_family):
        family_rows = by_family[family]
        status = _family_status(family_rows)
        sidecars = sorted(
            {
                component["name"]
                for row in family_rows
                for component in row["certificate_components"]
                if component["name"].endswith("_sidecar") and component["status"] == "available"
            }
        )
        lines.append(
            "| "
            f"{family} | "
            f"{len(family_rows)} | "
            f"{status[STATE_WEIGHTED_ACTION_MISMATCH]} | "
            f"{status[CLOSED_LOOP_TRANSITION_MISMATCH]}/"
            f"{status[VALUE_POLICY_GAP]}/"
            f"{status[BELLMAN_HESSIAN_RESIDUAL]} | "
            f"{', '.join(sidecars) if sidecars else 'none'} |"
        )
    return "\n".join(lines)


def _family_status(rows: list[dict[str, Any]]) -> dict[str, str]:
    status: dict[str, str] = {}
    for component_name in STANDARD_COMPONENTS:
        statuses = {
            component["status"]
            for row in rows
            for component in row["certificate_components"]
            if component["name"] == component_name
        }
        if statuses == {"available"}:
            status[component_name] = "available"
        elif "available" in statuses:
            status[component_name] = "mixed"
        elif statuses == {"not_applicable"}:
            status[component_name] = "not_applicable"
        else:
            status[component_name] = "missing"
    return status


def _key_rows(rows: list[dict[str, Any]]) -> str:
    selected = [
        row
        for row in rows
        if row["spec"]["controller_label"]
        in {
            "analytical_lqr_reference",
            "strong_optimizer_whitened__scratch",
            "strong_optimizer_whitened__bellman_init",
            "strong_optimizer_whitened_eigen_state_m4_s3__scratch",
            "strong_optimizer_whitened_eigen_trajectory_m1_s0.3__scratch",
        }
        or row["spec"]["parameters"].get("process_covariance_scale") in {0.0, 3.0}
        and row["spec"]["controller_label"] == "strong_optimizer_whitened__scratch"
    ]
    lines = []
    for row in selected:
        metrics = row["metrics"]
        exact_l2 = _first_present(
            metrics.get("exact_l2_cost_ratio_to_lqr"),
            metrics.get("deterministic_exact_l2_cost_ratio_to_lqr"),
        )
        lambda_ratio = _first_present(
            metrics.get("gamma_penalized_lambda_over_gamma_squared"),
            metrics.get("deterministic_lambda_over_gamma_squared"),
        )
        action = _first_present(
            _component_summary(row, STATE_WEIGHTED_ACTION_MISMATCH, "mismatch_ratio_mean"),
            metrics.get("under_epsilon_action_mismatch_ratio")
            if metrics.get("certificate_evaluation_lens") == "riccati_epsilon_response"
            else None,
            metrics.get("clean_action_mismatch_ratio"),
            metrics.get("action_mismatch_to_reference_mean"),
        )
        objective = _first_present(
            metrics.get("objective_ratio_to_reference"),
            metrics.get("cost_ratio_to_reference_mean"),
        )
        gain = _first_present(
            metrics.get("gain_relative_error"),
            metrics.get("deterministic_gain_relative_error"),
        )
        lines.append(
            "| "
            f"{row['spec']['run_id']} | "
            f"{row['status']} | "
            f"{row['spec']['parameters']['distribution_family']} | "
            f"{_fmt(objective)} | "
            f"{_fmt(gain)} | "
            f"{_fmt(action)} | "
            f"{_fmt(exact_l2)} | "
            f"{_fmt(lambda_ratio)} |"
        )
    return "\n".join(lines)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _component_summary(row: dict[str, Any], name: str, summary_key: str) -> Any:
    for component in row["certificate_components"]:
        if component["name"] == name and component["status"] == "available":
            return component["summary"].get(summary_key)
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()

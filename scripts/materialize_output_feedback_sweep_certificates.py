"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Apply bridge standard-certificate rows to 7a459bb sweep outputs.

The tracked sweep manifests are intentionally compact, so this materializer
reruns the deterministic sweep cells and the released-stochastic process-noise
evaluation when needed to recover the gains, trajectories, and covariances used
by the standard certificate. Tracked outputs stay small; regenerated bulk arrays
remain in ignored artifacts or in memory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.io import load_named_python_module
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
from rlrmp.analysis.bridge_results import (
    BridgeCertificateComponent,
    BridgeAnalysisResult,
    BridgeRunSpec,
    make_bridge_run_id,
)
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    CSReleasedStochasticNoiseConfig,
)
from rlrmp.analysis.math.linear_round_trip import LinearOptimizationConfig
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    kalman_estimator_joint_matrices,
    rollout_with_kalman_estimator,
)
from rlrmp.eval.output_feedback_rollout_recovery import (
    STRONG_OPTIMIZER_WHITENED,
    eigenspectrum_coverage_conditions,
    execute_governed_output_feedback_rollout_recovery,
)
from rlrmp.paths import REPO_ROOT, mkdir_p, portable_repo_path as _repo_relative


_LEGACY_PHASE3 = load_named_python_module(
    "rlrmp_legacy_cs_stochastic_phase3_consumer",
    REPO_ROOT / "legacy" / "analysis_pipelines" / "cs_stochastic_phase3.py",
)
Phase3StochasticConfig = _LEGACY_PHASE3.Phase3StochasticConfig
process_noise_sweep_summary = _LEGACY_PHASE3.process_noise_sweep_summary
run_phase3_process_noise_sweep = _LEGACY_PHASE3.run_phase3_process_noise_sweep


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
ROLLOUT_RECOVERY_RUNS_ROOT = (
    REPO_ROOT / "_artifacts" / ISSUE_ID / "output_feedback_sweep_certificates" / "feedbax_runs"
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

    rows: list[BridgeAnalysisResult] = []
    rows.extend(
        _saved_no_coverage_rows(source_manifests["no_coverage"], arrays, reference, output_config)
    )
    rows.extend(_initial_state_rows(source_manifests["initial_state"], reference, output_config))
    rows.extend(
        _process_noise_rows(
            source_manifests["process_noise"],
            source_manifests["no_coverage"],
            reference,
            output_config,
        )
    )
    rows.extend(_eigenspectrum_rows(source_manifests["eigenspectrum"], reference, output_config))

    row_dicts = [row.to_payload() for row in rows]
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
            "Full standard-certificate components are now available for the "
            "no-coverage/reference rows and for rerun deterministic initial-state "
            "and eigenspectrum coverage rows on nominal-clean and Riccati-epsilon "
            "evaluation lenses. Released-stochastic process-noise rows are rerun "
            "with common random numbers so state/action, transition, value-gap, "
            "Bellman-Hessian, visited-subspace, behavioral, exact-L2/gamma, and "
            "gain-diagnostic fields are all explicit when defined."
        ),
        "rows": row_dicts,
    }


def _saved_no_coverage_rows(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
) -> list[BridgeAnalysisResult]:
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
                BridgeAnalysisResult(
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


def _initial_state_rows(
    manifest: dict[str, Any],
    reference: Any,
    output_config: OutputFeedbackConfig,
) -> list[BridgeAnalysisResult]:
    rows = []
    base_config = LinearOptimizationConfig(**manifest["base_training_config"])
    for cell in manifest["cells"]:
        training_config = replace(
            base_config,
            basis_scale=cell["basis_scale"],
            random_state_scale=cell["random_state_scale"],
        )
        product = execute_governed_output_feedback_rollout_recovery(
            conditions=(STRONG_OPTIMIZER_WHITENED,),
            training_config=training_config,
            output_config=output_config,
            root=ROLLOUT_RECOVERY_RUNS_ROOT,
            issue_id=ISSUE_ID,
        )
        summary = product.summary
        fit = summary["fits"][0]
        rows.extend(
            _deterministic_fit_rows(
                fit=fit,
                arrays=product.arrays,
                reference=reference,
                output_config=output_config,
                family="initial-state coverage",
                run_parts=("initial_state_coverage", cell["label"], fit["label"]),
                training_distribution="synthetic_initial_state",
                source_manifest=SOURCE_MANIFESTS["initial_state"],
                extra_parameters={
                    "scale_factor": cell["scale_factor"],
                    "basis_scale": cell["basis_scale"],
                    "random_state_scale": cell["random_state_scale"],
                    "n_random_states": cell["n_random_states"],
                    "reach_weight": cell["reach_weight"],
                },
                notes=(
                    "Full standard bundle recomputed by rerunning the initial-state "
                    "coverage cell from the tracked sweep specification."
                ),
            )
        )
    return rows


def _process_noise_rows(
    manifest: dict[str, Any],
    no_coverage_manifest: dict[str, Any],
    reference: Any,
    output_config: OutputFeedbackConfig,
) -> list[BridgeAnalysisResult]:
    rows = []
    base_monte_carlo = manifest["base_monte_carlo"]
    base_noise_contract = manifest.get("base_noise_contract", base_monte_carlo)
    config = Phase3StochasticConfig(
        n_trials=base_monte_carlo["n_trials"],
        seed=base_monte_carlo["seed"],
        noise_config=CSReleasedStochasticNoiseConfig(
            motor_covariance_scale=base_noise_contract.get(
                "motor_covariance_scale",
                DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.motor_covariance_scale,
            ),
            process_covariance_scale=base_noise_contract.get(
                "process_covariance_scale",
                DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.process_covariance_scale,
            ),
            signal_dependent_scale=base_noise_contract.get(
                "signal_dependent_scale",
                DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG.signal_dependent_scale,
            ),
        ),
    )
    process_result = run_phase3_process_noise_sweep(
        config=config,
        process_covariance_scales=tuple(manifest["process_covariance_scales"]),
        output_config=output_config,
    )
    summary = process_noise_sweep_summary(process_result)
    optimizer_by_label = _optimizer_metadata_by_label(no_coverage_manifest)
    for cell, summary_cell in zip(process_result.cells, summary["cells"], strict=True):
        evaluation_by_label = {row["label"]: row for row in summary_cell["evaluations"]}
        for evaluation in cell.result.evaluations:
            label = evaluation.spec.label
            row_summary = evaluation_by_label[label]
            key = _safe_key(label)
            is_reference = label == "analytical_lqr_reference"
            raw_optimizer = optimizer_by_label.get(label)
            optimizer_metadata = None if is_reference else raw_optimizer
            optimizer_not_applicable = None
            if is_reference:
                optimizer_not_applicable = "analytical LQR reference has no optimizer metadata"
            elif optimizer_metadata is None:
                optimizer_not_applicable = (
                    "source controller is an initial-controller artifact without a "
                    "separate stochastic optimizer run"
                )
            components = list(
                _full_standard_components_from_trajectories(
                    x=cell.result.arrays[f"{key}_x"],
                    x_hat=cell.result.arrays[f"{key}_x_hat"],
                    reference=reference,
                    output_config=output_config,
                    candidate_gain=np.asarray(evaluation.spec.K),
                    optimizer_metadata=optimizer_metadata,
                    optimizer_not_applicable_reason=optimizer_not_applicable,
                    state_label=(
                        f"process_noise_{cell.process_covariance_scale:g}_stochastic_coupled_state"
                    ),
                    action_state_label=(
                        f"process_noise_{cell.process_covariance_scale:g}_stochastic_"
                        "estimated_state"
                    ),
                    architecture="reference" if is_reference else "time_constrained_free_gain",
                )
            )
            components.extend(_process_diagnostic_sidecars(row_summary))
            spec = BridgeRunSpec(
                issue_id=ISSUE_ID,
                run_id=make_bridge_run_id(
                    "process_noise_stochastic",
                    cell.label,
                    label,
                ),
                objective="optimal",
                architecture="reference" if is_reference else "time_constrained_free_gain",
                controller_label=label,
                optimizer_label="analytical" if is_reference else evaluation.spec.source,
                training_distribution="none" if is_reference else "nominal",
                evaluation_lane="released_stochastic",
                reference_controller="analytical_lqr_kalman",
                seed=cell.result.config.seed,
                gamma_factor=None,
                parameters={
                    "evaluation_distribution": "process-noise stochastic",
                    "distribution_family": "process-noise stochastic",
                    "process_covariance_scale": cell.process_covariance_scale,
                    "n_trials": cell.result.config.n_trials,
                    "motor_covariance_scale": cell.result.config.motor_covariance_scale,
                    "signal_dependent_scale": cell.result.config.signal_dependent_scale,
                    "source_manifest": _repo_relative(SOURCE_MANIFESTS["process_noise"]),
                },
                notes=(
                    "Full standard bundle recomputed on common-random-number released-"
                    "stochastic trajectories for this process-noise scale."
                ),
            )
            rows.append(
                BridgeAnalysisResult(
                    spec=spec,
                    status="full_standard_certificate",
                    metrics=_process_metrics(row_summary),
                    artifacts={
                        "source_manifest": _repo_relative(SOURCE_MANIFESTS["process_noise"])
                    },
                    certificate_components=components,
                )
            )
    return rows


def _eigenspectrum_rows(
    manifest: dict[str, Any],
    reference: Any,
    output_config: OutputFeedbackConfig,
) -> list[BridgeAnalysisResult]:
    rows = []
    product = execute_governed_output_feedback_rollout_recovery(
        conditions=eigenspectrum_coverage_conditions(),
        training_config=LinearOptimizationConfig(),
        output_config=output_config,
        root=ROLLOUT_RECOVERY_RUNS_ROOT,
        issue_id=ISSUE_ID,
    )
    summary = product.summary
    for fit in summary["fits"]:
        coverage = fit["condition"]["eigenspectrum_coverage"]
        family = f"eigenspectrum {coverage['objective']} coverage"
        rows.extend(
            _deterministic_fit_rows(
                fit=fit,
                arrays=product.arrays,
                reference=reference,
                output_config=output_config,
                family=family,
                run_parts=("eigenspectrum", coverage["objective"], fit["label"]),
                training_distribution=f"eigenspectrum_{coverage['objective']}",
                source_manifest=SOURCE_MANIFESTS["eigenspectrum"],
                extra_parameters={
                    "n_modes": coverage["n_modes"],
                    "scale": coverage["scale"],
                    "weight": coverage["weight"],
                },
                notes=(
                    "Full standard bundle recomputed by rerunning the eigenspectrum "
                    f"{coverage['objective']} coverage cell from the tracked sweep "
                    "specification."
                ),
            )
        )
    return rows


def _deterministic_fit_rows(
    *,
    fit: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
    family: str,
    run_parts: tuple[str, ...],
    training_distribution: str,
    source_manifest: Path,
    extra_parameters: dict[str, Any],
    notes: str,
    issue_id: str = ISSUE_ID,
    array_prefix: str | None = None,
    architecture: str = "time_constrained_free_gain",
    optimizer_label: str = "lbfgsb_strong_optimizer_whitened",
    gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    reference_controller: str = "analytical_lqr_kalman",
    row_metrics: dict[str, Any] | None = None,
) -> list[BridgeAnalysisResult]:
    rows = []
    prefix = array_prefix or fit["label"]
    evaluation_lenses = (
        ("nominal_clean", "clean", f"{family} nominal-clean"),
        ("riccati_epsilon_response", "under_eps", f"{family} Riccati-epsilon response"),
    )
    for lens_label, array_suffix, evaluation_distribution in evaluation_lenses:
        components = list(
            _full_standard_components(
                arrays=arrays,
                reference=reference,
                output_config=output_config,
                array_prefix=f"{prefix}_{array_suffix}",
                candidate_gain=np.asarray(arrays[f"{prefix}_K"]),
                optimizer_metadata=_optimizer_metadata(fit),
                state_label=f"{evaluation_distribution}_coupled_state",
                action_state_label=f"{evaluation_distribution}_estimated_state",
                architecture=architecture,
            )
        )
        components.extend(_diagnostic_sidecars(fit=fit, reference_manifest=None))
        spec = BridgeRunSpec(
            issue_id=issue_id,
            run_id=make_bridge_run_id(*run_parts, lens_label),
            objective="optimal",
            architecture=architecture,  # type: ignore[arg-type]
            controller_label=fit["label"],
            optimizer_label=optimizer_label,
            training_distribution=training_distribution,
            evaluation_lane="deterministic",
            reference_controller=reference_controller,
            gamma_factor=gamma_factor,
            parameters={
                "evaluation_distribution": evaluation_distribution,
                "distribution_family": family,
                "source_manifest": _repo_relative(source_manifest),
            }
            | extra_parameters,
            notes=notes,
        )
        rows.append(
            BridgeAnalysisResult(
                spec=spec,
                status="full_standard_certificate",
                metrics=_fit_metrics(fit)
                | (row_metrics or {})
                | {"certificate_evaluation_lens": lens_label},
                artifacts={"source_manifest": _repo_relative(source_manifest)},
                certificate_components=tuple(components),
            )
        )
    return rows


def deterministic_standard_rows_from_manifest_entries(
    *,
    entries: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
    issue_id: str,
    source_manifest: Path,
    default_family: str,
    default_training_distribution: str = "mixed",
    default_optimizer_label: str = "lbfgsb_strong_optimizer_whitened",
    default_architecture: str = "time_constrained_free_gain",
    default_gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    default_reference_controller: str = "analytical_lqr_kalman",
    default_notes: str = "Full standard certificate computed from saved deterministic arrays.",
) -> list[dict[str, Any]]:
    """Build standard-certificate rows from saved deterministic row descriptors.

    This is the adapter for follow-on materializers such as the 87edaae smooth
    time-basis sweep: callers keep their own manifest schema, but provide one
    descriptor per fitted row plus the NPZ array prefix containing
    ``<prefix>_K``, ``<prefix>_clean_x``, ``<prefix>_clean_x_hat``,
    ``<prefix>_under_eps_x``, and ``<prefix>_under_eps_x_hat``.
    """

    rows: list[dict[str, Any]] = []
    for entry in entries:
        fit = entry.get("fit", entry)
        label = fit["label"]
        run_parts = tuple(
            entry.get("run_parts", (entry.get("source_group", "saved"), label))
        )
        row_manifests = _deterministic_fit_rows(
            fit=fit,
            arrays=arrays,
            reference=reference,
            output_config=output_config,
            family=entry.get("family", default_family),
            run_parts=run_parts,
            training_distribution=entry.get(
                "training_distribution",
                default_training_distribution,
            ),
            source_manifest=source_manifest,
            extra_parameters=entry.get("parameters", entry.get("row_parameters", {})),
            notes=entry.get("notes", default_notes),
            issue_id=issue_id,
            array_prefix=entry.get("array_prefix", label),
            architecture=entry.get("architecture", default_architecture),
            optimizer_label=entry.get("optimizer_label", default_optimizer_label),
            gamma_factor=entry.get("gamma_factor", default_gamma_factor),
            reference_controller=entry.get(
                "reference_controller",
                default_reference_controller,
            ),
            row_metrics=entry.get("metrics"),
        )
        rows.extend(row.to_payload() for row in row_manifests)
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
    x: np.ndarray | None = None,
    x_hat: np.ndarray | None = None,
) -> tuple[BridgeCertificateComponent, ...]:
    plant = reference.plant
    schedule = reference.schedule
    reference_gain = np.asarray(reference.lqr_solution.K)
    if x is None:
        x = _saved_array(arrays, array_prefix, "x")
    if x_hat is None:
        x_hat = _saved_array(arrays, array_prefix, "x_hat")
    coupled = np.concatenate([x, x_hat], axis=-1)
    coupled_states = coupled[None, :, :] if coupled.ndim == 2 else coupled
    action_states = x_hat[None, :, :] if x_hat.ndim == 2 else x_hat
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


def _full_standard_components_from_trajectories(
    *,
    x: np.ndarray,
    x_hat: np.ndarray,
    reference: Any,
    output_config: OutputFeedbackConfig,
    candidate_gain: np.ndarray,
    optimizer_metadata: dict[str, Any] | None,
    optimizer_not_applicable_reason: str | None = None,
    state_label: str,
    action_state_label: str,
    architecture: str,
) -> tuple[BridgeCertificateComponent, ...]:
    components = list(
        _full_standard_components(
            arrays={},
            reference=reference,
            output_config=output_config,
            array_prefix="",
            candidate_gain=candidate_gain,
            optimizer_metadata=optimizer_metadata,
            state_label=state_label,
            action_state_label=action_state_label,
            architecture=architecture,
            x=x,
            x_hat=x_hat,
        )
    )
    if optimizer_not_applicable_reason is not None:
        components = _replace_component(
            components,
            BridgeCertificateComponent.not_applicable(
                OPTIMIZER_METADATA,
                optimizer_not_applicable_reason,
            ),
        )
    return tuple(components)


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


def _process_diagnostic_sidecars(evaluation: dict[str, Any]) -> list[BridgeCertificateComponent]:
    return [
        BridgeCertificateComponent.available(
            BEHAVIORAL_ACTION_SIDECAR,
            cost_ratio_to_reference_mean=evaluation.get("cost_ratio_to_reference_mean"),
            cost_ratio_to_reference_std=evaluation.get("cost_ratio_to_reference_std"),
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
        BridgeCertificateComponent.available(
            GAIN_DIAGNOSTIC_SIDECAR,
            gain_relative_error=evaluation.get("deterministic_gain_relative_error"),
            diagnostic_only=True,
            gate="not_used_as_certificate_gate",
        ),
        BridgeCertificateComponent.available(
            ROLL_OUT_BEHAVIOR_SIDECAR,
            peak_forward_velocity_mean=evaluation.get("peak_forward_velocity_mean"),
            terminal_error_mean=evaluation.get("terminal_error_mean"),
            control_effort_mean=evaluation.get("control_effort_mean"),
        ),
    ]


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


def _optimizer_metadata_by_label(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {fit["label"]: _optimizer_metadata(fit) for fit in manifest.get("fits", [])}


def _safe_key(label: str) -> str:
    return (
        label.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("-", "_")
        .replace("+", "plus")
    )


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
recent output-feedback coverage/noise sweep outputs. It reruns deterministic
sweep cells and released-stochastic process-noise evaluations as needed so the
standard certificate is computed from fitted gains, trajectories, and
covariances rather than inferred from scalar summaries.

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

| run | status | distribution | objective ratio | gain sidecar<sup>2</sup> | action mismatch<sup>1</sup> | transition mismatch | value gap | Bellman residual<sup>1</sup> | exact L2 sidecar | lambda/gamma^2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{key_rows}

<sup>1</sup> State-weighted action mismatch and Bellman-Hessian residual can
match exactly when the Bellman action Hessian is a scalar multiple of the action
cost geometry on that row. In that case they are the same evidence expressed
through two certificate views; they diverge when downstream value geometry
weights action directions differently.

<sup>2</sup> Gain mismatch is a diagnostic sidecar, not the bridge gate. The
gate is disturbance-relevant same-game behavior under the standard certificate
components.

## Computation Notes

- The initial-state and eigenspectrum rows are not inferred from the compact
  tracked manifests. They are rerun from the tracked sweep specifications so
  fitted gains, nominal-clean rollouts, and Riccati-epsilon rollouts are
  available for the full certificate.
- Process-noise stochastic rows are rerun with the tracked common-random-number
  settings so sampled state/estimate/action trajectories are available for the
  same component bundle.
- Evaluation lenses are not training axes. Nominal-clean, Riccati-epsilon, and
  process-noise stochastic rows describe where the finished controller is
  evaluated; they do not by themselves mean the controller was trained with a
  robust objective or coverage distribution.

## Verdict

The standard certificate application does not rescue the bridge. All rows in
this materialization now have full component availability where the linear
output-feedback quantities are defined. The rerun initial-state, process-noise,
and eigenspectrum rows continue to show behaviorally close but certificate-poor
from-scratch recovery.
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
        transition = _component_summary(
            row,
            CLOSED_LOOP_TRANSITION_MISMATCH,
            "mismatch_ratio_mean",
        )
        value_gap = _component_summary(row, VALUE_POLICY_GAP, "gap_ratio_mean")
        bellman = _component_summary(row, BELLMAN_HESSIAN_RESIDUAL, "residual_ratio_mean")
        lines.append(
            "| "
            f"{row['spec']['run_id']} | "
            f"{row['status']} | "
            f"{row['spec']['parameters']['distribution_family']} | "
            f"{_fmt(objective)} | "
            f"{_fmt(gain)} | "
            f"{_fmt(action)} | "
            f"{_fmt(transition)} | "
            f"{_fmt(value_gap)} | "
            f"{_fmt(bellman)} | "
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


if __name__ == "__main__":
    main()

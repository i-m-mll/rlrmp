"""Materialize output-feedback bridge failure-decomposition diagnostics.

This companion to the standard certificate uses saved rollout-recovery arrays
when available.  It does not retrain controllers for the no-coverage key rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.bridge_certificates import (
    STATE_WEIGHTED_ACTION_MISMATCH,
    state_weighted_action_mismatch_component,
)
from rlrmp.analysis.math.cs_game_card import materialize_reference
from rlrmp.analysis.pipelines.failure_decomposition import (
    classify_failure,
    covariances_from_states,
    gain_error_subspace_decomposition,
    interpolation_curve,
    objective_gradient_summary,
)
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig, output_feedback_clean_objective
from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
    STRONG_OPTIMIZER_WHITENED,
    _training_ensemble,
    eigenspectrum_coverage_conditions,
    result_summary as rollout_result_summary,
    run_output_feedback_rollout_recovery,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


jax.config.update("jax_enable_x64", True)

ISSUE_ID = "c45adde"
SOURCE_ISSUE_ID = "7a459bb"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
UMBRELLA_ID = "43e8728"
SOURCE_ARTIFACT = (
    REPO_ROOT
    / "_artifacts"
    / SOURCE_ISSUE_ID
    / "output_feedback_rollout_recovery"
    / "output_feedback_rollout_recovery.npz"
)
SOURCE_ROLLOUT_MANIFEST = (
    REPO_ROOT
    / "results"
    / SOURCE_ISSUE_ID
    / "notes"
    / "output_feedback_rollout_recovery_manifest.json"
)
SOURCE_STANDARD_MANIFEST = (
    REPO_ROOT
    / "results"
    / SOURCE_ISSUE_ID
    / "notes"
    / "output_feedback_sweep_standard_certificates_manifest.json"
)
SOURCE_OBSERVER_MANIFEST = (
    REPO_ROOT
    / "results"
    / SOURCE_ISSUE_ID
    / "notes"
    / "output_feedback_observer_error_coverage_manifest.json"
)
SOURCE_OBSERVER_ARTIFACT = (
    REPO_ROOT
    / "_artifacts"
    / SOURCE_ISSUE_ID
    / "output_feedback_observer_error_coverage"
    / "output_feedback_observer_error_coverage.npz"
)
COVERAGE_CACHE_DIR = (
    REPO_ROOT / "_artifacts" / SOURCE_ISSUE_ID / "output_feedback_failure_decomposition"
)
COVERAGE_ARRAY_CACHE = COVERAGE_CACHE_DIR / "deterministic_coverage_arrays.npz"
COVERAGE_SUMMARY_CACHE = COVERAGE_CACHE_DIR / "deterministic_coverage_summary.json"
NOTE_PATH = (
    REPO_ROOT / "results" / SOURCE_ISSUE_ID / "notes" / "output_feedback_failure_decomposition.md"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "results"
    / SOURCE_ISSUE_ID
    / "notes"
    / "output_feedback_failure_decomposition_manifest.json"
)

KEY_LABELS = (
    "strong_optimizer_whitened__scratch",
    "strong_optimizer_whitened__bellman_init",
)
EVALUATION_LENSES = (
    ("nominal_clean", "clean"),
    ("riccati_epsilon_response", "under_eps"),
)
INTERPOLATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, default=SOURCE_ARTIFACT)
    parser.add_argument("--rollout-manifest", type=Path, default=SOURCE_ROLLOUT_MANIFEST)
    parser.add_argument("--standard-manifest", type=Path, default=SOURCE_STANDARD_MANIFEST)
    parser.add_argument("--observer-manifest", type=Path, default=SOURCE_OBSERVER_MANIFEST)
    parser.add_argument("--observer-artifact", type=Path, default=SOURCE_OBSERVER_ARTIFACT)
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize(
        source_artifact=args.source_artifact,
        rollout_manifest_path=args.rollout_manifest,
        standard_manifest_path=args.standard_manifest,
        observer_manifest_path=args.observer_manifest,
        observer_artifact=args.observer_artifact,
    )
    write_result(result, note_path=args.note_output, manifest_path=args.manifest_output)
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")


def materialize(
    *,
    source_artifact: Path = SOURCE_ARTIFACT,
    rollout_manifest_path: Path = SOURCE_ROLLOUT_MANIFEST,
    standard_manifest_path: Path = SOURCE_STANDARD_MANIFEST,
    observer_manifest_path: Path = SOURCE_OBSERVER_MANIFEST,
    observer_artifact: Path = SOURCE_OBSERVER_ARTIFACT,
) -> dict[str, Any]:
    """Return the failure-decomposition bundle for current output-feedback rows."""

    rollout_manifest = _read_json(rollout_manifest_path)
    standard_manifest = _read_json(standard_manifest_path)
    observer_manifest = _read_json(observer_manifest_path)
    with np.load(source_artifact) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(observer_artifact) as archive:
        observer_arrays = {name: np.asarray(archive[name]) for name in archive.files}

    reference = materialize_reference()
    output_config = OutputFeedbackConfig(**rollout_manifest["diagnostics"]["output_config"])
    training_config = LinearTrainingConfig(**rollout_manifest["diagnostics"]["training_config"])
    training_states, training_weights = _training_ensemble(
        reference.plant,
        training_config,
        output_config,
    )
    objective = _make_objective_bundle(
        reference=reference,
        output_config=output_config,
        arrays=arrays,
        training_states=training_states,
        training_weights=training_weights,
    )
    standard_by_id = {row["spec"]["run_id"]: row for row in standard_manifest["rows"]}
    standard_by_id.update(
        {row["spec"]["run_id"]: row for row in observer_manifest["standard_certificate"]["rows"]}
    )
    fit_by_label = {fit["label"]: fit for fit in rollout_manifest["fits"]}
    rows = []
    for label in KEY_LABELS:
        rows.extend(
            _label_rows(
                label=label,
                arrays=arrays,
                standard_by_id=standard_by_id,
                fit=fit_by_label[label],
                objective=objective,
            )
        )
    coverage_summaries, coverage_arrays = _load_or_run_coverage_cache(
        output_config=output_config,
    )
    rows.extend(
        _initial_state_rows_from_cache(
            summaries=coverage_summaries["initial_state"],
            arrays=coverage_arrays,
            standard_by_id=standard_by_id,
        )
    )
    rows.extend(
        _rollout_summary_rows(
            summary=coverage_summaries["eigenspectrum"],
            arrays=coverage_arrays,
            standard_by_id=standard_by_id,
            run_id_prefix=lambda fit: (
                "eigenspectrum",
                fit["condition"]["eigenspectrum_coverage"]["objective"],
                fit["label"],
            ),
            array_prefix=lambda fit: f"eigenspectrum__{fit['label']}",
            source_group="eigenspectrum_coverage",
        )
    )
    rows.extend(
        _rollout_summary_rows(
            summary=observer_manifest["rollout_summary"],
            arrays=observer_arrays,
            standard_by_id=standard_by_id,
            run_id_prefix=lambda fit: (
                "observer_error",
                fit["condition"]["observer_error_coverage"]["objective"],
                fit["label"],
            ),
            array_prefix=lambda fit: fit["label"],
            source_group="observer_error_coverage",
        )
    )

    classifications = Counter(row["classification"]["classification"] for row in rows)
    return {
        "format": "rlrmp.output_feedback_failure_decomposition.v1",
        "issue": ISSUE_ID,
        "source_issue": SOURCE_ISSUE_ID,
        "standard_certificate_issue": STANDARD_CERTIFICATE_ISSUE_ID,
        "umbrella": UMBRELLA_ID,
        "source_manifests": {
            "rollout_recovery": _repo_relative(rollout_manifest_path),
            "standard_certificate": _repo_relative(standard_manifest_path),
            "observer_error_coverage": _repo_relative(observer_manifest_path),
        },
        "source_artifacts": {
            "saved_rollout_arrays": _repo_relative(source_artifact),
            "observer_error_arrays": _repo_relative(observer_artifact),
            "deterministic_coverage_arrays": _repo_relative(COVERAGE_ARRAY_CACHE),
            "deterministic_coverage_summary": _repo_relative(COVERAGE_SUMMARY_CACHE),
        },
        "scope": (
            "Current output-feedback no-coverage, initial-state coverage, "
            "eigenspectrum coverage, and observer-error coverage rows. Saved "
            "arrays are reused where available; deterministic coverage reruns are "
            "cached under ignored artifacts so future applications do not lose "
            "the controller and rollout arrays required for the decomposition."
        ),
        "summary": {
            "n_rows": len(rows),
            "n_controller_labels": len({row["controller_label"] for row in rows}),
            "classification_counts": dict(sorted(classifications.items())),
            "source_groups": sorted({row["source_group"] for row in rows}),
            "evaluation_lenses": [lens for lens, _suffix in EVALUATION_LENSES],
        },
        "rows": rows,
    }


def _load_or_run_coverage_cache(
    *,
    output_config: OutputFeedbackConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if COVERAGE_SUMMARY_CACHE.exists() and COVERAGE_ARRAY_CACHE.exists():
        with np.load(COVERAGE_ARRAY_CACHE) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        _normalize_cached_common_arrays(arrays)
        return _read_json(COVERAGE_SUMMARY_CACHE), arrays

    initial_manifest = _read_json(
        REPO_ROOT
        / "results"
        / SOURCE_ISSUE_ID
        / "notes"
        / "output_feedback_initial_state_variability_sweep_manifest.json"
    )
    base_config = LinearTrainingConfig(**initial_manifest["base_training_config"])
    summaries: dict[str, Any] = {"initial_state": [], "eigenspectrum": None}
    arrays: dict[str, np.ndarray] = {}

    for cell in initial_manifest["cells"]:
        training_config = replace(
            base_config,
            basis_scale=cell["basis_scale"],
            random_state_scale=cell["random_state_scale"],
        )
        result = run_output_feedback_rollout_recovery(
            conditions=(STRONG_OPTIMIZER_WHITENED,),
            training_config=training_config,
            output_config=output_config,
        )
        summary = rollout_result_summary(result)
        summaries["initial_state"].append({"cell": cell, "summary": summary})
        prefix = f"initial_state_coverage__{cell['label']}"
        for key, value in result.arrays.items():
            arrays[f"{prefix}__{key}"] = value

    eigen_result = run_output_feedback_rollout_recovery(
        conditions=eigenspectrum_coverage_conditions(),
        training_config=LinearTrainingConfig(),
        output_config=output_config,
    )
    summaries["eigenspectrum"] = rollout_result_summary(eigen_result)
    for key, value in eigen_result.arrays.items():
        arrays[f"eigenspectrum__{key}"] = value
    _normalize_cached_common_arrays(arrays)

    mkdir_p(COVERAGE_CACHE_DIR)
    np.savez_compressed(COVERAGE_ARRAY_CACHE, **arrays)
    COVERAGE_SUMMARY_CACHE.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summaries, arrays


def _normalize_cached_common_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Expose common reference arrays independent of the cached sweep prefix."""

    for key in ("lqr_reference_K",):
        if key in arrays:
            continue
        for candidate, value in arrays.items():
            if candidate.endswith(f"__{key}"):
                arrays[key] = value
                break


def _make_objective_bundle(
    *,
    reference: Any,
    output_config: OutputFeedbackConfig,
    arrays: dict[str, np.ndarray],
    training_states: jnp.ndarray,
    training_weights: jnp.ndarray,
) -> dict[str, Any]:
    plant = reference.plant
    schedule = reference.schedule
    reference_gain = jnp.asarray(arrays["lqr_reference_K"], dtype=jnp.float64)
    state_scales = jnp.asarray(arrays["state_scales"], dtype=jnp.float64)
    shape = reference_gain.shape

    def to_theta(gains: jnp.ndarray) -> jnp.ndarray:
        return gains * state_scales[None, None, :]

    def to_gain(theta: jnp.ndarray) -> jnp.ndarray:
        return theta / state_scales[None, None, :]

    def objective_theta(theta_flat: jnp.ndarray) -> jnp.ndarray:
        theta = theta_flat.reshape(shape)
        gains = to_gain(theta)
        return output_feedback_clean_objective(
            plant,
            schedule,
            gains,
            training_states,
            training_weights,
            output_config,
        )

    value_and_grad = jax.value_and_grad(objective_theta)

    def objective_np(theta_flat: np.ndarray) -> float:
        return float(objective_theta(jnp.asarray(theta_flat, dtype=jnp.float64)))

    def gradient_np(theta_flat: np.ndarray) -> np.ndarray:
        _value, grad = value_and_grad(jnp.asarray(theta_flat, dtype=jnp.float64))
        return np.asarray(grad, dtype=float)

    return {
        "reference_gain": np.asarray(reference_gain),
        "reference_theta_flat": np.asarray(to_theta(reference_gain)).reshape(-1),
        "to_theta_flat": lambda gains: np.asarray(to_theta(jnp.asarray(gains))).reshape(-1),
        "to_gain": lambda theta_flat: np.asarray(
            to_gain(jnp.asarray(theta_flat, dtype=jnp.float64).reshape(shape))
        ),
        "objective_np": objective_np,
        "gradient_np": gradient_np,
        "schedule": schedule,
    }


def _label_rows(
    *,
    label: str,
    arrays: dict[str, np.ndarray],
    standard_by_id: dict[str, dict[str, Any]],
    fit: dict[str, Any],
    objective: dict[str, Any],
) -> list[dict[str, Any]]:
    learned_gain = arrays[f"{label}_K"]
    reference_gain = objective["reference_gain"]
    learned_theta = objective["to_theta_flat"](learned_gain)
    reference_theta = objective["reference_theta_flat"]
    objective_summary = objective_gradient_summary(
        learned=learned_theta,
        reference=reference_theta,
        objective_fn=objective["objective_np"],
        gradient_fn=objective["gradient_np"],
        projected_gradient_fn=objective["gradient_np"],
    )
    rows = []
    for lens, suffix in EVALUATION_LENSES:
        run_id = f"no_coverage__{label}__{lens}"
        standard_row = standard_by_id[run_id]
        x_hat = arrays[f"{label}_{suffix}_x_hat"]
        states = x_hat[None, :, :]
        decomposition = gain_error_subspace_decomposition(
            gain_delta=learned_gain - reference_gain,
            state_covariances=covariances_from_states(states),
        )
        interpolation = interpolation_curve(
            learned=learned_gain,
            reference=reference_gain,
            metric_fns={
                "training_objective": lambda gains: _objective_for_gain(gains, objective),
                "state_weighted_action_mismatch": lambda gains: _action_mismatch(
                    gains=gains,
                    reference_gain=reference_gain,
                    x_hat=x_hat,
                    action_weight=np.asarray(objective["schedule"].R),
                ),
            },
            alphas=INTERPOLATION_ALPHAS,
        )
        objective_reference = objective_summary["reference_objective"]
        for record in interpolation:
            record["training_objective_ratio_to_reference"] = (
                record["training_objective"] / objective_reference
            )
        certificate_mismatch = _component_summary(
            standard_row,
            STATE_WEIGHTED_ACTION_MISMATCH,
            "mismatch_ratio_mean",
        )
        classification = classify_failure(
            objective_ratio=objective_summary["learned_to_reference_objective_ratio"],
            learned_gradient_norm=objective_summary["learned_projected_gradient_norm"],
            reference_gradient_norm=objective_summary["reference_projected_gradient_norm"],
            certificate_mismatch_ratio=certificate_mismatch,
            subspace_decomposition=decomposition,
        )
        rows.append(
            {
                "run_id": run_id,
                "source_group": "no_coverage",
                "controller_label": label,
                "evaluation_lens": lens,
                "row_parameters": {},
                "source_standard_status": standard_row["status"],
                "source_standard_distribution": standard_row["spec"]["parameters"][
                    "distribution_family"
                ],
                "objective": objective_summary,
                "source_optimizer": {
                    key: fit.get(key)
                    for key in (
                        "optimizer_status",
                        "optimizer_success",
                        "n_iterations",
                        "n_function_evaluations",
                    )
                },
                "certificate": {
                    "state_weighted_action_mismatch": certificate_mismatch,
                    "bellman_hessian_residual": _component_summary(
                        standard_row,
                        "bellman_hessian_residual",
                        "residual_ratio_mean",
                    ),
                    "closed_loop_transition_mismatch": _component_summary(
                        standard_row,
                        "closed_loop_transition_mismatch",
                        "mismatch_ratio_mean",
                    ),
                    "value_gap": _component_summary(
                        standard_row,
                        "value_policy_gap",
                        "gap_ratio_mean",
                    ),
                },
                "gain_error_decomposition": decomposition,
                "interpolation": interpolation,
                "classification": classification,
            }
        )
    return rows


def _initial_state_rows_from_cache(
    *,
    summaries: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    standard_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for entry in summaries:
        cell = entry["cell"]
        rows.extend(
            _rollout_summary_rows(
                summary=entry["summary"],
                arrays=arrays,
                standard_by_id=standard_by_id,
                run_id_prefix=lambda fit, cell=cell: (
                    "initial_state_coverage",
                    cell["label"],
                    fit["label"],
                ),
                array_prefix=lambda fit, cell=cell: (
                    f"initial_state_coverage__{cell['label']}__{fit['label']}"
                ),
                source_group="initial_state_coverage",
                row_parameters={
                    "scale_factor": cell["scale_factor"],
                    "basis_scale": cell["basis_scale"],
                    "random_state_scale": cell["random_state_scale"],
                },
            )
        )
    return rows


def _rollout_summary_rows(
    *,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    standard_by_id: dict[str, dict[str, Any]],
    run_id_prefix: Any,
    array_prefix: Any,
    source_group: str,
    row_parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    reference_gain = arrays["lqr_reference_K"]
    action_weight = np.asarray(materialize_reference().schedule.R)
    rows = []
    for fit in summary["fits"]:
        label = fit["label"]
        prefix = array_prefix(fit)
        learned_gain = arrays[f"{prefix}_K"]
        objective_summary = _fit_objective_summary(fit)
        for lens, suffix in EVALUATION_LENSES:
            run_id = "__".join((*run_id_prefix(fit), lens))
            standard_row = standard_by_id[run_id]
            x_hat = arrays[f"{prefix}_{suffix}_x_hat"]
            decomposition = gain_error_subspace_decomposition(
                gain_delta=learned_gain - reference_gain,
                state_covariances=covariances_from_states(x_hat[None, :, :]),
            )
            interpolation = interpolation_curve(
                learned=learned_gain,
                reference=reference_gain,
                metric_fns={
                    "state_weighted_action_mismatch": lambda gains, x_hat=x_hat: _action_mismatch(
                        gains=gains,
                        reference_gain=reference_gain,
                        x_hat=x_hat,
                        action_weight=action_weight,
                    ),
                },
                alphas=INTERPOLATION_ALPHAS,
            )
            certificate_mismatch = _component_summary(
                standard_row,
                STATE_WEIGHTED_ACTION_MISMATCH,
                "mismatch_ratio_mean",
            )
            classification = classify_failure(
                objective_ratio=objective_summary["learned_to_reference_objective_ratio"],
                learned_gradient_norm=objective_summary["learned_projected_gradient_norm"],
                reference_gradient_norm=objective_summary["reference_projected_gradient_norm"],
                certificate_mismatch_ratio=certificate_mismatch,
                subspace_decomposition=decomposition,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "source_group": source_group,
                    "controller_label": label,
                    "evaluation_lens": lens,
                    "row_parameters": row_parameters
                    or _coverage_parameters(fit.get("condition", {})),
                    "source_standard_status": standard_row["status"],
                    "source_standard_distribution": standard_row["spec"]["parameters"][
                        "distribution_family"
                    ],
                    "objective": objective_summary,
                    "source_optimizer": {
                        key: fit.get(key)
                        for key in (
                            "optimizer_status",
                            "optimizer_success",
                            "n_iterations",
                            "n_function_evaluations",
                        )
                    },
                    "certificate": {
                        "state_weighted_action_mismatch": certificate_mismatch,
                        "bellman_hessian_residual": _component_summary(
                            standard_row,
                            "bellman_hessian_residual",
                            "residual_ratio_mean",
                        ),
                        "closed_loop_transition_mismatch": _component_summary(
                            standard_row,
                            "closed_loop_transition_mismatch",
                            "mismatch_ratio_mean",
                        ),
                        "value_gap": _component_summary(
                            standard_row,
                            "value_policy_gap",
                            "gap_ratio_mean",
                        ),
                    },
                    "gain_error_decomposition": decomposition,
                    "interpolation": interpolation,
                    "classification": classification,
                }
            )
    return rows


def failure_rows_from_manifest_entries(
    *,
    entries: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    standard_rows: dict[str, Any] | list[dict[str, Any]],
    default_source_group: str,
) -> list[dict[str, Any]]:
    """Build failure-decomposition rows from saved deterministic row descriptors.

    The entries use the same descriptor shape as
    ``deterministic_standard_rows_from_manifest_entries`` in the standard
    certificate materializer. Each descriptor supplies a fit summary, run-id
    prefix, and NPZ array prefix; this adapter joins those rows back to their
    standard-certificate rows and emits the existing c45adde-compatible
    failure-decomposition schema.
    """

    standard_by_id = _standard_rows_by_id(standard_rows)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        fit = entry.get("fit", entry)
        label = fit["label"]
        run_parts = tuple(
            entry.get("run_parts", (entry.get("source_group", default_source_group), label))
        )
        array_prefix = entry.get("array_prefix", label)
        source_group = entry.get("source_group", default_source_group)
        rows.extend(
            _rollout_summary_rows(
                summary={"fits": [fit]},
                arrays=arrays,
                standard_by_id=standard_by_id,
                run_id_prefix=lambda _fit, run_parts=run_parts: run_parts,
                array_prefix=lambda _fit, array_prefix=array_prefix: array_prefix,
                source_group=source_group,
                row_parameters=entry.get("row_parameters", entry.get("parameters", {})),
            )
        )
    return rows


def _standard_rows_by_id(
    standard_rows: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(standard_rows, dict):
        if "standard_certificate" in standard_rows:
            rows = standard_rows["standard_certificate"]["rows"]
        elif "rows" in standard_rows:
            rows = standard_rows["rows"]
        else:
            rows = list(standard_rows.values())
    else:
        rows = standard_rows
    return {row["spec"]["run_id"]: row for row in rows}


def _fit_objective_summary(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "learned_objective": fit.get("objective_final"),
        "reference_objective": fit.get("objective_reference"),
        "learned_to_reference_objective_ratio": fit.get("objective_ratio_to_reference"),
        "learned_gradient_norm": fit.get("gradient_norm_final"),
        "reference_gradient_norm": None,
        "learned_projected_gradient_norm": fit.get("projected_gradient_norm_final")
        if fit.get("projected_gradient_norm_final") is not None
        else fit.get("gradient_norm_final"),
        "reference_projected_gradient_norm": None,
        "source": "rollout_fit_summary",
    }


def _coverage_parameters(condition: dict[str, Any]) -> dict[str, Any]:
    return (
        condition.get("eigenspectrum_coverage")
        or condition.get("observer_error_coverage")
        or {}
    )


def _objective_for_gain(gains: np.ndarray, objective: dict[str, Any]) -> float:
    theta = objective["to_theta_flat"](gains)
    return float(objective["objective_np"](theta))


def _action_mismatch(
    *,
    gains: np.ndarray,
    reference_gain: np.ndarray,
    x_hat: np.ndarray,
    action_weight: np.ndarray,
) -> float:
    component = state_weighted_action_mismatch_component(
        states=x_hat[None, :, :],
        candidate_gain=gains,
        reference_gain=reference_gain,
        action_weight=action_weight,
        state_label="evaluation_estimated_state",
        action_label="control",
    )
    return float(component.summary["mismatch_ratio_mean"])


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
    return f"""# Output-Feedback Failure Decomposition

Issue: `{ISSUE_ID}`. Source bridge issue: `{SOURCE_ISSUE_ID}`. Standard certificate
cross-reference: `{STANDARD_CERTIFICATE_ISSUE_ID}`.

This materialization is the standard failure-decomposition companion to the
bridge standard certificate. It answers why a row failed after the certificate
has answered whether the learned controller is equivalent to the analytical
reference. It does not change the bridge gate.

Scope: {result["scope"]}

## Source Inputs

- Rollout-recovery manifest: `{result["source_manifests"]["rollout_recovery"]}`
- Standard-certificate manifest: `{result["source_manifests"]["standard_certificate"]}`
- Observer-error manifest: `{result["source_manifests"]["observer_error_coverage"]}`
- Saved no-coverage arrays: `{result["source_artifacts"]["saved_rollout_arrays"]}`
- Observer-error arrays: `{result["source_artifacts"]["observer_error_arrays"]}`
- Deterministic coverage array cache: `{result["source_artifacts"]["deterministic_coverage_arrays"]}`

## Key Rows

| run | class | objective ratio | learned proj-grad<sup>1</sup> | reference proj-grad<sup>1</sup> | action mismatch | Bellman residual | strong visited error | weak/unvisited error<sup>2</sup> | best interp alpha | best interp objective ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{_table_rows(result["rows"])}

<sup>1</sup> Objective and projected-gradient diagnostics are evaluated in each
row's training objective and optimizer parameterization. For no-coverage rows,
the materializer recomputes learned/reference gradients directly from the
saved objective. For coverage rows, the final optimizer gradient is taken from
the saved or cached fit summary because the optimizer closure itself is not
stored in tracked results.

<sup>2</sup> The visited/weakly visited decomposition projects gain error
through the evaluation-lens estimated-state covariance used by the standard
state-weighted action mismatch. Weak/unvisited gain error is explanatory only;
it is not the certificate gate.

## Definitions

- `under_identification`: the training objective is already near the reference,
  but certificate mismatch remains and gain error lies mostly in weakly visited
  or unvisited state directions.
- `optimizer_basin`: the learned point is still objectively worse than the
  reference and/or has a large projected gradient in the optimizer
  parameterization.
- `objective_mismatch`: the learned point is stationary under the training
  objective while the analytical reference is not.
- `mixed`: more than one of the preceding signals is active.

## Interpretation

The from-scratch key rows remain optimizer-basin failures under this diagnostic:
their learned controllers are not stationary under the objective that trained
them, and the standard-certificate mismatches remain large. Coverage changes
the training distribution, but it does not rescue this free time-varying
architecture. The Bellman-initialized no-coverage key row remains the sanity
check: objective, gradient, and certificate residuals are all near the
reference.
"""


def _table_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        objective_records = [
            record
            for record in row["interpolation"]
            if "training_objective_ratio_to_reference" in record
        ]
        best = (
            min(objective_records, key=lambda record: record["training_objective_ratio_to_reference"])
            if objective_records
            else None
        )
        lines.append(
            "| "
            f"{row['run_id']} | "
            f"{row['classification']['classification']} | "
            f"{_fmt(row['objective']['learned_to_reference_objective_ratio'])} | "
            f"{_fmt(row['objective']['learned_projected_gradient_norm'])} | "
            f"{_fmt(row['objective']['reference_projected_gradient_norm'])} | "
            f"{_fmt(row['certificate']['state_weighted_action_mismatch'])} | "
            f"{_fmt(row['certificate']['bellman_hessian_residual'])} | "
            f"{_fmt(row['gain_error_decomposition']['strong_fraction_mean'])} | "
            f"{_fmt(row['gain_error_decomposition']['weak_or_unvisited_fraction_mean'])} | "
            f"{_fmt(None if best is None else best['alpha'])} | "
            f"{_fmt(None if best is None else best['training_objective_ratio_to_reference'])} |"
        )
    return "\n".join(lines)


def _component_summary(row: dict[str, Any], name: str, summary_key: str) -> Any:
    for component in row["certificate_components"]:
        if component["name"] == name:
            return component.get("summary", {}).get(summary_key)
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

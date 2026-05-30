"""Materialize output-feedback bridge failure-decomposition diagnostics.

This companion to the standard certificate uses saved rollout-recovery arrays
when available.  It does not retrain controllers for the no-coverage key rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.bridge_certificates import (
    STATE_WEIGHTED_ACTION_MISMATCH,
    state_weighted_action_mismatch_component,
)
from rlrmp.analysis.cs_game_card import materialize_reference
from rlrmp.analysis.failure_decomposition import (
    classify_failure,
    covariances_from_states,
    gain_error_subspace_decomposition,
    interpolation_curve,
    objective_gradient_summary,
)
from rlrmp.analysis.linear_round_trip import LinearTrainingConfig
from rlrmp.analysis.output_feedback import OutputFeedbackConfig, output_feedback_clean_objective
from rlrmp.analysis.output_feedback_rollout_recovery import _training_ensemble
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
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize(
        source_artifact=args.source_artifact,
        rollout_manifest_path=args.rollout_manifest,
        standard_manifest_path=args.standard_manifest,
    )
    write_result(result, note_path=args.note_output, manifest_path=args.manifest_output)
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")


def materialize(
    *,
    source_artifact: Path = SOURCE_ARTIFACT,
    rollout_manifest_path: Path = SOURCE_ROLLOUT_MANIFEST,
    standard_manifest_path: Path = SOURCE_STANDARD_MANIFEST,
) -> dict[str, Any]:
    """Return the failure-decomposition bundle for saved no-coverage key rows."""

    rollout_manifest = _read_json(rollout_manifest_path)
    standard_manifest = _read_json(standard_manifest_path)
    with np.load(source_artifact) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

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
        },
        "source_artifacts": {"saved_rollout_arrays": _repo_relative(source_artifact)},
        "scope": (
            "Saved no-coverage output-feedback key rows. The materializer reuses "
            "stored gains and rollout arrays, evaluates the same clean training "
            "objective at learned/reference gains, and projects gain error through "
            "the standard-certificate evaluation state distributions."
        ),
        "summary": {
            "n_rows": len(rows),
            "n_controller_labels": len(KEY_LABELS),
            "classification_counts": dict(sorted(classifications.items())),
            "labels": list(KEY_LABELS),
            "evaluation_lenses": [lens for lens, _suffix in EVALUATION_LENSES],
        },
        "rows": rows,
    }


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
                "controller_label": label,
                "evaluation_lens": lens,
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
- Saved no-coverage arrays: `{result["source_artifacts"]["saved_rollout_arrays"]}`

## Key Rows

| run | class | objective ratio | learned proj-grad<sup>1</sup> | reference proj-grad<sup>1</sup> | action mismatch | Bellman residual | strong visited error | weak/unvisited error<sup>2</sup> | best interp alpha | best interp objective ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{_table_rows(result["rows"])}

<sup>1</sup> Objective and projected-gradient diagnostics are evaluated in the
same whitened L-BFGS-B theta parameterization used by the saved
`strong_optimizer_whitened` no-coverage fits. There are no active box
constraints, so projected-gradient norm equals gradient norm for these rows.

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

The no-coverage from-scratch key row is an optimizer-basin failure under this
diagnostic: moving along the straight line toward the analytical reference
reduces the clean training objective, and the learned projected gradient remains
large. The Bellman-initialized key row is not a substantive failure: objective,
gradient, and certificate residuals are all near the reference.
"""


def _table_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        best = min(
            row["interpolation"],
            key=lambda record: record["training_objective_ratio_to_reference"],
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
            f"{_fmt(best['alpha'])} | "
            f"{_fmt(best['training_objective_ratio_to_reference'])} |"
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

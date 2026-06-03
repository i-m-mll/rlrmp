"""Objective-comparator sidecars for GRU/full-QRF analytical comparisons."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlrmp.paths import REPO_ROOT


SCHEMA_VERSION = "rlrmp.objective_comparator_sidecar.v2"
DEFAULT_MONTE_CARLO_STATUS = {
    "status": "not_implemented",
    "reason": (
        "same-noise-bank extLQG-vs-GRU realized comparison was not materialized; "
        "the available tracked source only contains validation-selected GRU "
        "realized full-QRF scalars and the analytical extLQG expected-cost "
        "decomposition"
    ),
}


@dataclass(frozen=True)
class ExtLQGCostDecomposition:
    """C&S extLQG expected-cost components under the full-QRF objective lens."""

    deterministic_initial_state: float
    initial_covariance_trace: float
    accumulated_noise_scalar: float
    provenance: str
    total_expected_cost: float | None = None

    @property
    def component_sum(self) -> float:
        """Return the sum of deterministic, covariance, and noise terms."""

        return (
            self.deterministic_initial_state
            + self.initial_covariance_trace
            + self.accumulated_noise_scalar
        )

    @property
    def expected_cost(self) -> float:
        """Return the declared total expected cost, or the component sum."""

        if self.total_expected_cost is None:
            return self.component_sum
        return self.total_expected_cost

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable decomposition record."""

        return {
            "lens": "extlqg_covariance_inclusive_expected_cost",
            "deterministic_initial_state": self.deterministic_initial_state,
            "initial_covariance_trace": self.initial_covariance_trace,
            "accumulated_noise_scalar": self.accumulated_noise_scalar,
            "component_sum": self.component_sum,
            "total_expected_cost": self.expected_cost,
            "component_sum_delta": self.expected_cost - self.component_sum,
            "comparable_scalar": self.deterministic_initial_state,
            "comparable_scalar_lens": "extlqg_deterministic_initial_state_full_qrf",
            "provenance": self.provenance,
        }


def build_objective_comparator_sidecar(
    *,
    issue: str,
    source_manifest: str,
    checkpoint_selection: Mapping[str, Any],
    extlqg: ExtLQGCostDecomposition,
    scope: str,
    generated_by: str,
    same_noise_bank_monte_carlo: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON sidecar from validation-selected checkpoint records."""

    runs = checkpoint_selection.get("runs")
    if not isinstance(runs, Mapping):
        raise ValueError("checkpoint_selection must contain a mapping at key 'runs'")

    sidecar_rows = [
        _build_run_row(run_id=str(run_id), selections=_expect_sequence(selections), extlqg=extlqg)
        for run_id, selections in sorted(runs.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": issue,
        "scope": scope,
        "source_manifest": source_manifest,
        "generated_by": generated_by,
        "checkpoint_policy": {
            "label": "validation_selected_per_replicate",
            "source": checkpoint_selection.get("schema_version"),
            "selection_policy": checkpoint_selection.get("selection_policy"),
            "caveat": (
                "Checkpoint selection is inherited from the validation-selected "
                "GRU manifest. Analytical action, I/O, and extLQG comparator "
                "metrics are audit-only and are not used for checkpoint selection."
            ),
        },
        "objective_lenses": {
            "gru_validation_selected_realized_full_qrf": {
                "kind": "realized_validation_objective",
                "definition": (
                    "sum_t x_t^T Q_t x_t + u_t^T R_t u_t + x_T^T Q_f x_T "
                    "using states.mechanics.vector for x and states.net.output for u"
                ),
            },
            "extlqg_deterministic_initial_state_full_qrf": {
                "kind": "deterministic_analytical_term",
                "definition": "x0^T Sx0 x0 from the extLQG/computeOFC recursion",
            },
            "extlqg_covariance_inclusive_expected_cost": {
                "kind": "expected_cost",
                "definition": (
                    "deterministic initial-state term plus initial covariance trace "
                    "plus accumulated process/sensory/motor noise scalar"
                ),
            },
        },
        "extlqg_decomposition": extlqg.to_json(),
        "same_noise_bank_monte_carlo": dict(
            same_noise_bank_monte_carlo or DEFAULT_MONTE_CARLO_STATUS
        ),
        "rows": sidecar_rows,
        "caveats": [
            (
                "The apples-to-apples scalar for the available GRU validation "
                "records is the deterministic extLQG term, not the "
                "covariance-inclusive expected cost."
            ),
            "This sidecar is diagnostic only and is not a standard-certificate gate.",
            (
                "GRU values are validation-selected realized full-QRF scalars; "
                "same-noise-bank extLQG realized values require separate Monte "
                "Carlo materialization."
            ),
        ],
    }


def render_objective_comparator_markdown(sidecar: Mapping[str, Any]) -> str:
    """Render a compact Markdown companion for an objective-comparator sidecar."""

    decomposition = _expect_mapping(sidecar["extlqg_decomposition"])
    rows = _expect_sequence(sidecar["rows"])
    lines = [
        "# Full-QRF objective comparator sidecar",
        "",
        f"Schema: `{sidecar['schema_version']}`.",
        "",
        f"Scope: {sidecar['scope']}.",
        "",
        "This is an objective-lens diagnostic, not a standard-certificate gate.",
        "",
        "## extLQG decomposition",
        "",
        "| component | value | lens |",
        "|---|---:|---|",
        (
            "| deterministic initial-state term | "
            f"{_fmt(decomposition['deterministic_initial_state'])} | comparable to "
            "realized/validation full-QRF values |"
        ),
        (
            "| initial covariance trace term | "
            f"{_fmt(decomposition['initial_covariance_trace'])} | expected-cost sidecar only |"
        ),
        (
            "| accumulated noise scalar | "
            f"{_fmt(decomposition['accumulated_noise_scalar'])} | expected-cost sidecar only |"
        ),
        (
            "| total expected cost | "
            f"{_fmt(decomposition['total_expected_cost'])} | not directly comparable to GRU "
            "validation values |"
        ),
        "",
        "## GRU comparison",
        "",
        (
            "| run | mean selected validation | deterministic extLQG | "
            "selected/deterministic | total expected cost | selected/total |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        row_map = _expect_mapping(row)
        lines.append(
            "| "
            f"`{row_map['run_id']}` | "
            f"{_fmt(row_map['gru_mean_selected_validation_full_qrf'])} | "
            f"{_fmt(row_map['extlqg_deterministic_full_qrf'])} | "
            f"{_fmt(row_map['selected_to_extlqg_deterministic_ratio'])} | "
            f"{_fmt(row_map['extlqg_total_expected_cost'])} | "
            f"{_fmt(row_map['selected_to_extlqg_total_ratio_not_apples_to_apples'])} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            (
                "- `selected/total` is retained only as a labeled non-apples-to-apples "
                "diagnostic for continuity with the provisional sidecar."
            ),
        ]
    )
    for caveat in _expect_sequence(sidecar["caveats"]):
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            "Same-noise-bank Monte Carlo: "
            f"`{_expect_mapping(sidecar['same_noise_bank_monte_carlo'])['status']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_objective_comparator_sidecar(
    sidecar: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write JSON and Markdown sidecar artifacts."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_objective_comparator_markdown(sidecar), encoding="utf-8")


def compute_default_extlqg_cost_decomposition() -> ExtLQGCostDecomposition:
    """Compute the canonical C&S extLQG expected-cost decomposition."""

    import jax.numpy as jnp

    from rlrmp.analysis.cs_game_card import (
        OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        materialize_reference,
    )
    from rlrmp.analysis.cs_released_simulation import (
        _compute_ext_kalman,
        _compute_ofc,
        _default_output_feedback_initial_state,
        default_cs_noise_covariances,
    )
    from rlrmp.analysis.output_feedback import (
        delayed_observation_matrix,
        position_velocity_observation_config,
    )

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    plant = reference.plant
    schedule = reference.schedule
    config = position_velocity_observation_config(plant)
    covariances = default_cs_noise_covariances(plant, config)
    h_matrix = delayed_observation_matrix(plant, config)
    state_noise = covariances.motor + covariances.process
    initial_covariance = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance,
        dtype=jnp.float64,
    )
    estimator_gains = jnp.zeros((schedule.T, plant.n, h_matrix.shape[0]), dtype=jnp.float64)
    current = 1.0e6
    deterministic = 0.0
    initial_trace = 0.0
    scalar = 0.0
    expected = current
    iteration = 0
    for iteration in range(1, 101):
        controller_gains, sx0, se0, scalar_cost = _compute_ofc(
            plant,
            schedule,
            estimator_gains,
            h_matrix,
            covariances.signal_dependent_state,
            state_noise,
            covariances.sensory,
        )
        estimator_gains, _state_covariances = _compute_ext_kalman(
            plant,
            h_matrix,
            controller_gains,
            covariances.signal_dependent_state,
            state_noise,
            covariances.sensory,
            initial_covariance,
            initial_covariance,
        )
        x0 = _default_output_feedback_initial_state(plant, config)
        deterministic = float(x0 @ sx0 @ x0)
        initial_trace = float(jnp.trace((sx0 + se0) @ initial_covariance))
        scalar = float(scalar_cost)
        expected = deterministic + initial_trace + scalar
        relative_change = abs(current - expected) / max(abs(expected), 1e-300)
        current = expected
        if relative_change <= 1e-14:
            break

    return ExtLQGCostDecomposition(
        deterministic_initial_state=deterministic,
        initial_covariance_trace=initial_trace,
        accumulated_noise_scalar=scalar,
        total_expected_cost=expected,
        provenance=(
            "canonical C&S extLQG fixed-point decomposition from "
            "materialize_reference(output_feedback_certificate_gamma_factor), "
            f"{iteration} iterations"
        ),
    )


def materialize_gru_objective_comparator_sidecar(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    checkpoint_policy: str,
    use_validation_selected_checkpoints: bool,
    checkpoint_manifest: Mapping[str, Any] | None,
    checkpoint_manifest_path: Path | None,
    standard_manifest_path: Path,
    output_path: Path,
    note_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize the GRU objective-comparator sidecar for a post-run bundle."""

    del labels
    if not use_validation_selected_checkpoints:
        return {
            "status": "skipped",
            "reason": "objective_comparator_requires_validation_selected_checkpoints",
        }
    if checkpoint_policy != "validation_selected_per_replicate":
        return {
            "status": "skipped",
            "reason": "unsupported_checkpoint_policy",
            "checkpoint_policy": checkpoint_policy,
        }
    if checkpoint_manifest is None:
        if checkpoint_manifest_path is None:
            raise ValueError("checkpoint_manifest or checkpoint_manifest_path is required")
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))

    extlqg = compute_default_extlqg_cost_decomposition()
    sidecar = build_objective_comparator_sidecar(
        issue=experiment,
        source_manifest=_repo_relative(standard_manifest_path, repo_root=repo_root),
        checkpoint_selection=checkpoint_manifest,
        extlqg=extlqg,
        scope=(
            "validation-selected checkpoints for C&S GRU runs: "
            + ", ".join(str(run_id) for run_id in run_ids)
        ),
        generated_by="rlrmp.analysis.objective_comparator.materialize_gru_objective_comparator_sidecar",
    )
    write_objective_comparator_sidecar(
        sidecar,
        json_path=output_path,
        markdown_path=note_path,
    )
    return {
        "status": "materialized",
        "schema_version": sidecar["schema_version"],
        "n_rows": len(sidecar["rows"]),
        "extlqg_deterministic_full_qrf": extlqg.deterministic_initial_state,
        "extlqg_total_expected_cost": extlqg.expected_cost,
    }


def _build_run_row(
    *,
    run_id: str,
    selections: Sequence[Any],
    extlqg: ExtLQGCostDecomposition,
) -> dict[str, Any]:
    selected = [_float_from_selection(item, "scoring_validation_objective") for item in selections]
    best_logged = [
        _float_from_selection(item, "best_logged_validation_objective") for item in selections
    ]
    mean_selected = sum(selected) / len(selected)
    mean_best_logged = sum(best_logged) / len(best_logged)
    return {
        "run_id": run_id,
        "checkpoint_policy": "validation_selected_per_replicate",
        "n_replicates": len(selections),
        "gru_realized_lens": "gru_validation_selected_realized_full_qrf",
        "extlqg_comparable_lens": "extlqg_deterministic_initial_state_full_qrf",
        "gru_mean_selected_validation_full_qrf": mean_selected,
        "gru_mean_best_logged_validation_full_qrf": mean_best_logged,
        "extlqg_deterministic_full_qrf": extlqg.deterministic_initial_state,
        "selected_to_extlqg_deterministic_ratio": (
            mean_selected / extlqg.deterministic_initial_state
        ),
        "best_logged_to_extlqg_deterministic_ratio": (
            mean_best_logged / extlqg.deterministic_initial_state
        ),
        "extlqg_total_expected_cost": extlqg.expected_cost,
        "selected_to_extlqg_total_ratio_not_apples_to_apples": (
            mean_selected / extlqg.expected_cost
        ),
        "selected_checkpoints": list(selections),
    }


def _float_from_selection(selection: Any, key: str) -> float:
    selection_map = _expect_mapping(selection)
    return float(selection_map[key])


def _expect_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping, found {type(value).__name__}")
    return value


def _expect_sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected sequence, found {type(value).__name__}")
    return value


def _fmt(value: Any) -> str:
    return f"{float(value):.8g}"


def _load_checkpoint_selection(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    checkpoint_selection = manifest.get("checkpoint_selection", manifest)
    return _expect_mapping(checkpoint_selection)


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for materializing a sidecar from tracked manifests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--generated-by", default="python -m rlrmp.analysis.objective_comparator")
    parser.add_argument("--extlqg-deterministic", required=True, type=float)
    parser.add_argument("--extlqg-initial-covariance", required=True, type=float)
    parser.add_argument("--extlqg-accumulated-noise", required=True, type=float)
    parser.add_argument("--extlqg-total", required=True, type=float)
    parser.add_argument("--extlqg-provenance", required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sidecar = build_objective_comparator_sidecar(
        issue=args.issue,
        source_manifest=str(args.manifest),
        checkpoint_selection=_load_checkpoint_selection(manifest),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=args.extlqg_deterministic,
            initial_covariance_trace=args.extlqg_initial_covariance,
            accumulated_noise_scalar=args.extlqg_accumulated_noise,
            total_expected_cost=args.extlqg_total,
            provenance=args.extlqg_provenance,
        ),
        scope=args.scope,
        generated_by=args.generated_by,
    )
    write_objective_comparator_sidecar(
        sidecar,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )


if __name__ == "__main__":
    main()

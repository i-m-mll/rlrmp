"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Materialize observer-error coverage rows for output-feedback rollout recovery."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
    observer_error_coverage_conditions,
    result_summary as rollout_result_summary,
    run_output_feedback_rollout_recovery,
)
from rlrmp.analysis.pipelines.standard_certificate_materialization import (
    deterministic_output_feedback_rows,
)
from rlrmp.paths import REPO_ROOT, mkdir_p, portable_repo_path


ISSUE_ID = "3becdec"
PARENT_ISSUE_ID = "7a459bb"
UMBRELLA_ID = "43e8728"
NOTE_PATH = (
    REPO_ROOT / "results" / PARENT_ISSUE_ID / "notes" / "output_feedback_observer_error_coverage.md"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "results"
    / PARENT_ISSUE_ID
    / "notes"
    / "output_feedback_observer_error_coverage_manifest.json"
)
ARTIFACT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / PARENT_ISSUE_ID
    / "output_feedback_observer_error_coverage"
    / "output_feedback_observer_error_coverage.npz"
)
NO_COVERAGE_MANIFEST = (
    REPO_ROOT
    / "results"
    / PARENT_ISSUE_ID
    / "notes"
    / "output_feedback_rollout_recovery_manifest.json"
)
EIGENSPECTRUM_MANIFEST = (
    REPO_ROOT
    / "results"
    / PARENT_ISSUE_ID
    / "notes"
    / "output_feedback_eigenspectrum_coverage_sweep_manifest.json"
)
STANDARD_CERTIFICATE_MANIFEST = (
    REPO_ROOT
    / "results"
    / PARENT_ISSUE_ID
    / "notes"
    / "output_feedback_sweep_standard_certificates_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--artifact-output", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, arrays = materialize()
    write_result(
        result,
        arrays=arrays,
        note_path=args.note_output,
        manifest_path=args.manifest_output,
        artifact_path=args.artifact_output,
    )
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")
    print(f"Wrote {args.artifact_output}")


def materialize() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the observer-error grid and return tracked manifest content."""

    start = time.perf_counter()
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    output_config = OutputFeedbackConfig()
    conditions = observer_error_coverage_conditions(
        objectives=("trajectory", "state"),
        modes=(1,),
        scales=(0.3, 1.0),
        weight=0.1,
    )
    result = run_output_feedback_rollout_recovery(
        conditions=conditions,
        training_config=LinearTrainingConfig(),
        output_config=output_config,
    )
    summary = rollout_result_summary(result)
    standard_rows = []
    for fit in summary["fits"]:
        coverage = fit["condition"]["observer_error_coverage"]
        standard_rows.extend(
            deterministic_output_feedback_rows(
                fit=fit,
                arrays=result.arrays,
                reference=reference,
                output_config=output_config,
                family=f"observer-error {coverage['objective']} coverage",
                run_parts=("observer_error", coverage["objective"], fit["label"]),
                training_distribution=f"observer_error_{coverage['objective']}",
                source_manifest=MANIFEST_PATH,
                extra_parameters={
                    "n_modes": coverage["n_modes"],
                    "scale": coverage["scale"],
                    "weight": coverage["weight"],
                    "coverage_reference": coverage["reference"],
                },
                notes=(
                    "Full standard bundle computed from the deterministic "
                    f"observer-error {coverage['objective']} coverage row."
                ),
                issue_id=PARENT_ISSUE_ID,
            )
        )

    standard_row_dicts = [row.to_json_dict() for row in standard_rows]
    status_counts = Counter(row.status for row in standard_rows)
    return (
        {
            "format": "rlrmp.output_feedback_observer_error_coverage.v1",
            "issue": ISSUE_ID,
            "parent_issue": PARENT_ISSUE_ID,
            "umbrella": UMBRELLA_ID,
            "source_manifests": {
                "no_coverage": _repo_relative(NO_COVERAGE_MANIFEST),
                "eigenspectrum": _repo_relative(EIGENSPECTRUM_MANIFEST),
                "prior_standard_certificates": _repo_relative(STANDARD_CERTIFICATE_MANIFEST),
            },
            "artifact_npz": _repo_relative(ARTIFACT_PATH),
            "runtime_seconds": time.perf_counter() - start,
            "grid": {
                "objectives": ["trajectory", "state"],
                "modes": [1],
                "scales": [0.3, 1.0],
                "weight": 0.1,
                "optimizer": "strong_optimizer_whitened",
                "initialization": "scratch",
            },
            "interpretation": (
                "Observer-error coverage uses leading singular disturbance directions "
                "of the analytical LQR disturbance-to-observer-error map. Trajectory "
                "rows train on signed full-trial disturbance samples; state rows train "
                "on the time-indexed (x, xhat) states induced by those samples."
            ),
            "comparison": _comparison_rows(summary),
            "rollout_summary": summary,
            "standard_certificate": {
                "status_counts": dict(sorted(status_counts.items())),
                "full_standard_certificate_rows": [
                    row.spec.run_id
                    for row in standard_rows
                    if row.status == "full_standard_certificate"
                ],
                "rows": standard_row_dicts,
            },
            "verdict": _verdict(summary),
        },
        result.arrays,
    )


def write_result(
    result: dict[str, Any],
    *,
    arrays: dict[str, np.ndarray],
    note_path: Path = NOTE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    artifact_path: Path = ARTIFACT_PATH,
) -> None:
    mkdir_p(note_path.parent)
    mkdir_p(artifact_path.parent)
    result = dict(result)
    result["tracked_note"] = _repo_relative(note_path)
    result["tracked_manifest"] = _repo_relative(manifest_path)
    result["artifact_npz"] = _repo_relative(artifact_path)
    result["artifact_npz_keys"] = sorted(arrays)
    np.savez_compressed(artifact_path, **arrays)
    note_path.write_text(render_markdown(result), encoding="utf-8")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(result: dict[str, Any]) -> str:
    observer_rows = _observer_rows(result["rollout_summary"]["fits"])
    comparison_rows = _comparison_table(result["comparison"])
    certificate_count = len(result["standard_certificate"]["full_standard_certificate_rows"])
    return f"""# Observer-Error Coverage for Output-Feedback Rollout Recovery

Issue: `{ISSUE_ID}`. Parent: `{PARENT_ISSUE_ID}`. Umbrella: `{UMBRELLA_ID}`.

This materialization tests observer-error coverage as the remaining small
coverage-style diagnostic before moving away from coverage/noise changes. The
task and cost are unchanged; coverage only changes the training distribution.
All rows use `strong_optimizer_whitened` from scratch.

Method: {result["interpretation"]}

Runtime: `{result["runtime_seconds"]:.2f}` seconds.

Artifacts:

- Manifest: `{result["tracked_manifest"]}`
- Arrays: `{result["artifact_npz"]}`

## Observer-Error Grid

| objective | modes | scale | iters | objective ratio | gain rel err | exact L2 ratio | lambda/gamma^2 |
|---|---:|---:|---:|---:|---:|---:|---:|
{observer_rows}

## Comparison

| source | objective | modes | scale | objective ratio | gain rel err | exact L2 ratio | lambda/gamma^2 |
|---|---|---:|---:|---:|---:|---:|---:|
{comparison_rows}

## Standard Certificate Coverage

All `{certificate_count}` observer-error evaluation rows have
`full_standard_certificate` status, covering nominal-clean and Riccati-epsilon
evaluation lenses for each trained controller.

## Verdict

{result["verdict"]}
"""


def _observer_rows(fits: list[dict[str, Any]]) -> str:
    lines = []
    for fit in fits:
        coverage = fit["condition"]["observer_error_coverage"]
        lines.append(
            "| "
            f"{coverage['objective']} | "
            f"{coverage['n_modes']} | "
            f"{coverage['scale']:g} | "
            f"{fit['n_iterations']} | "
            f"{_fmt(fit['objective_ratio_to_reference'])} | "
            f"{_fmt(fit['gain_relative_error'])} | "
            f"{_fmt(fit['exact_l2_cost_ratio_to_lqr'])} | "
            f"{_fmt(fit['gamma_penalized_lambda_over_gamma_squared'])} |"
        )
    return "\n".join(lines)


def _comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    no_coverage = _read_json(NO_COVERAGE_MANIFEST)
    eigenspectrum = _read_json(EIGENSPECTRUM_MANIFEST)
    baseline = _fit_by_label(no_coverage["fits"], "strong_optimizer_whitened__scratch")
    rows = [_comparison_row("no coverage", "none", baseline)]
    for objective in ("trajectory", "state"):
        rows.append(
            _comparison_row(
                "eigenspectrum best exact-L2",
                objective,
                _best_fit(
                    eigenspectrum["fits"],
                    coverage_key="eigenspectrum_coverage",
                    objective=objective,
                    metric="exact_l2_cost_ratio_to_lqr",
                ),
            )
        )
        rows.append(
            _comparison_row(
                "observer-error best exact-L2",
                objective,
                _best_fit(
                    summary["fits"],
                    coverage_key="observer_error_coverage",
                    objective=objective,
                    metric="exact_l2_cost_ratio_to_lqr",
                ),
            )
        )
    return rows


def _comparison_row(source: str, objective: str, fit: dict[str, Any]) -> dict[str, Any]:
    coverage = (
        fit["condition"].get("observer_error_coverage")
        or fit["condition"].get("eigenspectrum_coverage")
        or {}
    )
    return {
        "source": source,
        "objective": objective,
        "modes": coverage.get("n_modes"),
        "scale": coverage.get("scale"),
        "objective_ratio_to_reference": fit.get("objective_ratio_to_reference"),
        "gain_relative_error": fit.get("gain_relative_error"),
        "exact_l2_cost_ratio_to_lqr": fit.get("exact_l2_cost_ratio_to_lqr"),
        "gamma_penalized_lambda_over_gamma_squared": fit.get(
            "gamma_penalized_lambda_over_gamma_squared"
        ),
    }


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        "| "
        f"{row['source']} | "
        f"{row['objective']} | "
        f"{_fmt(row['modes'])} | "
        f"{_fmt(row['scale'])} | "
        f"{_fmt(row['objective_ratio_to_reference'])} | "
        f"{_fmt(row['gain_relative_error'])} | "
        f"{_fmt(row['exact_l2_cost_ratio_to_lqr'])} | "
        f"{_fmt(row['gamma_penalized_lambda_over_gamma_squared'])} |"
        for row in rows
    )


def _verdict(summary: dict[str, Any]) -> str:
    fits = summary["fits"]
    best_gain = min(fits, key=lambda fit: fit["gain_relative_error"])
    best_exact = min(fits, key=lambda fit: fit["exact_l2_cost_ratio_to_lqr"])
    lines = [
        f"Best observer-error gain error is `{best_gain['gain_relative_error']:.6g}` "
        f"({best_gain['label']}).",
        f"Best observer-error exact-L2 ratio is `{best_exact['exact_l2_cost_ratio_to_lqr']:.6g}` "
        f"({best_exact['label']}).",
    ]
    if best_gain["gain_relative_error"] < 1e-2:
        lines.append("Observer-error coverage rescues from-scratch gain recovery.")
    elif best_exact["exact_l2_cost_ratio_to_lqr"] < 1.05:
        lines.append(
            "Observer-error coverage improves the disturbance sidecar but does not "
            "recover the analytical gain."
        )
    else:
        lines.append(
            "Observer-error coverage does not rescue the free time-varying "
            "output-feedback rollout bridge in this small grid."
        )
    return "\n".join(lines)


def _fit_by_label(fits: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for fit in fits:
        if fit["label"] == label:
            return fit
    raise KeyError(label)


def _best_fit(
    fits: list[dict[str, Any]],
    *,
    coverage_key: str,
    objective: str,
    metric: str,
) -> dict[str, Any]:
    candidates = [
        fit for fit in fits if fit["condition"].get(coverage_key, {}).get("objective") == objective
    ]
    if not candidates:
        raise ValueError(f"No {coverage_key} fits for objective={objective!r}.")
    return min(candidates, key=lambda fit: fit[metric])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path) -> str:
    return portable_repo_path(path, repo_root=REPO_ROOT)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()

"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Materialize the 87edaae smooth time-basis output-feedback bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

import materialize_output_feedback_failure_decomposition as failure
import materialize_output_feedback_sweep_certificates as certificates
from rlrmp.analysis.math.cs_game_card import materialize_reference
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.analysis.pipelines.output_feedback_time_constrained import (
    ISSUE_ID,
    SPLINE_RANKS,
    TimeBasisCondition,
    r12_observer_error_state_coverage_conditions,
    r12_state_eigenspectrum_coverage_conditions,
    r20_observer_error_state_coverage_conditions,
    r20_state_eigenspectrum_coverage_conditions,
    render_markdown,
    timed_run,
    write_basic_outputs,
)
from rlrmp.io import write_compact_json
from rlrmp.paths import REPO_ROOT, mkdir_p


NOTE_PATH = REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_time_constrained.md"
MANIFEST_PATH = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_time_constrained_manifest.json"
)
ARTIFACT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "output_feedback_time_constrained"
    / "output_feedback_time_constrained.npz"
)
R20_COVERAGE_NOTE_PATH = (
    REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_time_constrained_r20_coverage.md"
)
R20_COVERAGE_MANIFEST_PATH = (
    REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_time_constrained_r20_coverage_manifest.json"
)
R20_COVERAGE_ARTIFACT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "output_feedback_time_constrained_r20_coverage"
    / "output_feedback_time_constrained_r20_coverage.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=str, default=",".join(str(rank) for rank in SPLINE_RANKS))
    parser.add_argument("--fit-ranks", type=str, default="")
    parser.add_argument("--adamw-lrs", type=str, default="0.003,0.01")
    parser.add_argument("--lbfgsb-maxiter", type=int, default=2000)
    parser.add_argument("--adamw-steps", type=int, default=5000)
    parser.add_argument("--polish-maxiter", type=int, default=1000)
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--artifact-output", type=Path, default=ARTIFACT_PATH)
    parser.add_argument(
        "--include-r12-coverage",
        action="store_true",
        help="Include r=12 state eigenspectrum and observer-error state coverage rows.",
    )
    parser.add_argument("--r12-coverage-rank", type=int, default=12)
    parser.add_argument("--r12-state-eigenspectrum-modes", type=str, default="1,4")
    parser.add_argument("--r12-state-eigenspectrum-scales", type=str, default="0.3,1,3")
    parser.add_argument("--r12-observer-error-modes", type=str, default="1")
    parser.add_argument("--r12-observer-error-scales", type=str, default="0.3,1")
    parser.add_argument("--r12-coverage-weight", type=float, default=0.1)
    parser.add_argument(
        "--include-r20-coverage",
        action="store_true",
        help="Include the focused r=20 state eigenspectrum and observer-error coverage rows.",
    )
    parser.add_argument("--r20-coverage-rank", type=int, default=20)
    parser.add_argument("--r20-state-eigenspectrum-modes", type=str, default="4")
    parser.add_argument("--r20-state-eigenspectrum-scales", type=str, default="1,3")
    parser.add_argument("--r20-observer-error-modes", type=str, default="1")
    parser.add_argument("--r20-observer-error-scales", type=str, default="0.3")
    parser.add_argument("--r20-coverage-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.include_r12_coverage and args.include_r20_coverage:
        raise ValueError("Choose at most one focused coverage materialization path.")
    ranks = _parse_ints(args.ranks)
    fit_ranks = _parse_ints(args.fit_ranks) if args.fit_ranks else None
    note_path = args.note_output
    manifest_path = args.manifest_output
    artifact_path = args.artifact_output
    if args.include_r20_coverage:
        if args.ranks == ",".join(str(rank) for rank in SPLINE_RANKS):
            ranks = (args.r20_coverage_rank,)
        if not args.fit_ranks:
            fit_ranks = (args.r20_coverage_rank,)
        if note_path == NOTE_PATH:
            note_path = R20_COVERAGE_NOTE_PATH
        if manifest_path == MANIFEST_PATH:
            manifest_path = R20_COVERAGE_MANIFEST_PATH
        if artifact_path == ARTIFACT_PATH:
            artifact_path = R20_COVERAGE_ARTIFACT_PATH
    summary, arrays = materialize(
        ranks=ranks,
        fit_ranks=fit_ranks,
        adamw_lrs=_parse_floats(args.adamw_lrs),
        lbfgsb_maxiter=args.lbfgsb_maxiter,
        adamw_steps=args.adamw_steps,
        polish_maxiter=args.polish_maxiter,
        manifest_path=manifest_path,
        include_r12_coverage=args.include_r12_coverage,
        r12_coverage_rank=args.r12_coverage_rank,
        r12_state_eigenspectrum_modes=_parse_ints(args.r12_state_eigenspectrum_modes),
        r12_state_eigenspectrum_scales=_parse_floats(args.r12_state_eigenspectrum_scales),
        r12_observer_error_modes=_parse_ints(args.r12_observer_error_modes),
        r12_observer_error_scales=_parse_floats(args.r12_observer_error_scales),
        r12_coverage_weight=args.r12_coverage_weight,
        include_r20_coverage=args.include_r20_coverage,
        r20_coverage_rank=args.r20_coverage_rank,
        r20_state_eigenspectrum_modes=_parse_ints(args.r20_state_eigenspectrum_modes),
        r20_state_eigenspectrum_scales=_parse_floats(args.r20_state_eigenspectrum_scales),
        r20_observer_error_modes=_parse_ints(args.r20_observer_error_modes),
        r20_observer_error_scales=_parse_floats(args.r20_observer_error_scales),
        r20_coverage_weight=args.r20_coverage_weight,
    )
    write_result(
        summary,
        arrays=arrays,
        note_path=note_path,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
    )
    print(f"Wrote {note_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {artifact_path}")


def materialize(
    *,
    ranks: tuple[int, ...] = SPLINE_RANKS,
    fit_ranks: tuple[int, ...] | None = None,
    adamw_lrs: tuple[float, ...] = (3e-3, 1e-2),
    lbfgsb_maxiter: int = 2000,
    adamw_steps: int = 5000,
    polish_maxiter: int = 1000,
    manifest_path: Path = MANIFEST_PATH,
    include_r12_coverage: bool = False,
    r12_coverage_rank: int = 12,
    r12_state_eigenspectrum_modes: tuple[int, ...] = (1, 4),
    r12_state_eigenspectrum_scales: tuple[float, ...] = (0.3, 1.0, 3.0),
    r12_observer_error_modes: tuple[int, ...] = (1,),
    r12_observer_error_scales: tuple[float, ...] = (0.3, 1.0),
    r12_coverage_weight: float = 0.1,
    include_r20_coverage: bool = False,
    r20_coverage_rank: int = 20,
    r20_state_eigenspectrum_modes: tuple[int, ...] = (4,),
    r20_state_eigenspectrum_scales: tuple[float, ...] = (1.0, 3.0),
    r20_observer_error_modes: tuple[int, ...] = (1,),
    r20_observer_error_scales: tuple[float, ...] = (0.3,),
    r20_coverage_weight: float = 0.1,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run training plus standard-certificate/failure adapters."""

    if include_r12_coverage and include_r20_coverage:
        raise ValueError("Choose at most one focused coverage materialization path.")
    coverage_conditions = _r12_coverage_conditions(
        include=include_r12_coverage,
        rank=r12_coverage_rank,
        learning_rate=adamw_lrs[-1],
        adamw_steps=adamw_steps,
        polish_maxiter=polish_maxiter,
        state_eigenspectrum_modes=r12_state_eigenspectrum_modes,
        state_eigenspectrum_scales=r12_state_eigenspectrum_scales,
        observer_error_modes=r12_observer_error_modes,
        observer_error_scales=r12_observer_error_scales,
        weight=r12_coverage_weight,
    )
    coverage_conditions += _r20_coverage_conditions(
        include=include_r20_coverage,
        rank=r20_coverage_rank,
        learning_rate=adamw_lrs[-1],
        adamw_steps=adamw_steps,
        polish_maxiter=polish_maxiter,
        state_eigenspectrum_modes=r20_state_eigenspectrum_modes,
        state_eigenspectrum_scales=r20_state_eigenspectrum_scales,
        observer_error_modes=r20_observer_error_modes,
        observer_error_scales=r20_observer_error_scales,
        weight=r20_coverage_weight,
    )
    if coverage_conditions:
        coverage_ranks = tuple(condition.rank for condition in coverage_conditions)
        ranks = tuple(dict.fromkeys((*ranks, *coverage_ranks)))
    summary, arrays = timed_run(
        ranks=ranks,
        fit_ranks=fit_ranks,
        adamw_lrs=adamw_lrs,
        lbfgsb_maxiter=lbfgsb_maxiter,
        adamw_steps=adamw_steps,
        polish_maxiter=polish_maxiter,
        coverage_conditions=coverage_conditions,
    )
    if include_r12_coverage:
        summary["scope"] = (
            "Focused r=12 state-coverage follow-up for the smooth spline "
            "time-basis output-feedback bridge. Coverage rows test whether "
            "state-eigenspectrum or observer-error state coverage changes "
            "scratch discovery relative to the no-coverage r=12 baseline."
        )
        summary["non_goals"] = (
            "No trajectory eigenspectrum coverage, broader rank sweep, GRU, "
            "linear recurrence, robust training variants, or direct "
            "teacher-cloning claims."
        )
    if include_r20_coverage:
        summary["scope"] = (
            "Focused r=20 state-coverage closure for the smooth spline "
            "time-basis output-feedback bridge. Coverage rows are restricted "
            "to state-eigenspectrum m=4 at scales 1 and 3 plus observer-error "
            "state m=1 at scale 0.3, all with weight 0.1."
        )
        summary["non_goals"] = (
            "No broader rank sweep, trajectory eigenspectrum coverage, affine "
            "tracker, recurrent controller, GRU, robust training variants, or "
            "direct teacher-cloning claims."
        )
    entries = _row_entries(summary)
    reference = materialize_reference()
    output_config = OutputFeedbackConfig()
    standard_rows = certificates.deterministic_standard_rows_from_manifest_entries(
        entries=entries,
        arrays=arrays,
        reference=reference,
        output_config=output_config,
        issue_id=ISSUE_ID,
        source_manifest=manifest_path,
        default_family="smooth spline time-basis",
        default_training_distribution="mixed",
    )
    failure_rows = failure.failure_rows_from_manifest_entries(
        entries=entries,
        arrays=arrays,
        standard_rows={"standard_certificate": {"rows": standard_rows}},
        default_source_group="smooth_spline_time_basis",
    )
    summary["standard_certificate"] = {
        "rows": standard_rows,
        "n_rows": len(standard_rows),
        "status_counts": _counts(row["status"] for row in standard_rows),
    }
    summary["failure_decomposition"] = {
        "rows": failure_rows,
        "n_rows": len(failure_rows),
        "classification_counts": _counts(
            row["classification"]["classification"] for row in failure_rows
        ),
    }
    return summary, arrays


def write_result(
    summary: dict[str, Any],
    *,
    arrays: dict[str, np.ndarray],
    note_path: Path,
    manifest_path: Path,
    artifact_path: Path,
) -> None:
    write_basic_outputs(
        summary=summary,
        arrays=arrays,
        note_path=note_path,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
    )
    # write_basic_outputs writes before these adapter fields existed in older
    # callers; rewrite here so the final manifest always includes them.
    mkdir_p(note_path.parent)
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    write_compact_json(manifest_path, summary)


def _row_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for fit in summary["fits"]:
        condition = fit.get("condition", {})
        rank = condition.get("rank")
        label = fit["label"]
        coverage = _coverage_metadata(condition)
        if coverage is None and _is_trajectory_eigenspectrum_coverage(condition):
            continue
        basis_family = summary["diagnostics"]["basis_family"]
        parameters = {
            "rank": rank,
            "basis_family": basis_family,
            "initialization": fit.get("initialization"),
        }
        metrics = {
            "rank": rank,
            "basis_family": basis_family,
        }
        source_group = "smooth_spline_time_basis"
        family = f"smooth spline time-basis rank {rank}"
        training_distribution = "nominal"
        run_parts = ("smooth_spline_time_basis", f"rank_{rank}", label)
        notes = "Full standard certificate computed from saved deterministic arrays."
        if coverage is not None:
            parameters |= {
                "coverage_family": coverage["family"],
                "coverage_objective": coverage["objective"],
                "coverage_modes": coverage["modes"],
                "coverage_scale": coverage["scale"],
                "coverage_weight": coverage["weight"],
                "coverage_reference": coverage.get("reference"),
            }
            metrics |= {
                "coverage_family": coverage["family"],
                "coverage_objective": coverage["objective"],
                "coverage_modes": coverage["modes"],
                "coverage_scale": coverage["scale"],
                "coverage_weight": coverage["weight"],
            }
            source_group = coverage["family"]
            family = f"{coverage['family'].replace('_', ' ')} {coverage['objective']} coverage"
            training_distribution = f"{coverage['family']}_{coverage['objective']}"
            run_parts = (
                "smooth_spline_time_basis",
                f"rank_{rank}",
                coverage["family"],
                coverage["objective"],
                label,
            )
            notes = (
                "Full standard certificate computed from saved deterministic arrays "
                f"for the r={rank} state-coverage follow-up."
            )
        entries.append(
            {
                "fit": fit,
                "array_prefix": label,
                "run_parts": run_parts,
                "source_group": source_group,
                "family": family,
                "training_distribution": training_distribution,
                "optimizer_label": _optimizer_label(fit),
                "parameters": parameters,
                "metrics": metrics,
                "notes": notes,
            }
        )
    return entries


def _r12_coverage_conditions(
    *,
    include: bool,
    rank: int,
    learning_rate: float,
    adamw_steps: int,
    polish_maxiter: int,
    state_eigenspectrum_modes: tuple[int, ...],
    state_eigenspectrum_scales: tuple[float, ...],
    observer_error_modes: tuple[int, ...],
    observer_error_scales: tuple[float, ...],
    weight: float,
) -> tuple[TimeBasisCondition, ...]:
    if not include:
        return ()
    if rank != 12:
        raise ValueError("r12 coverage materialization only supports rank 12.")
    common_kwargs = {
        "weight": weight,
        "maxiter": adamw_steps,
        "learning_rate": learning_rate,
        "polish_maxiter": polish_maxiter,
    }
    return (
        r12_state_eigenspectrum_coverage_conditions(
            modes=state_eigenspectrum_modes,
            scales=state_eigenspectrum_scales,
            **common_kwargs,
        )
        + r12_observer_error_state_coverage_conditions(
            modes=observer_error_modes,
            scales=observer_error_scales,
            **common_kwargs,
        )
    )


def _r20_coverage_conditions(
    *,
    include: bool,
    rank: int,
    learning_rate: float,
    adamw_steps: int,
    polish_maxiter: int,
    state_eigenspectrum_modes: tuple[int, ...],
    state_eigenspectrum_scales: tuple[float, ...],
    observer_error_modes: tuple[int, ...],
    observer_error_scales: tuple[float, ...],
    weight: float,
) -> tuple[TimeBasisCondition, ...]:
    if not include:
        return ()
    if rank != 20:
        raise ValueError("r20 coverage materialization only supports rank 20.")
    common_kwargs = {
        "weight": weight,
        "maxiter": adamw_steps,
        "learning_rate": learning_rate,
        "polish_maxiter": polish_maxiter,
    }
    return (
        r20_state_eigenspectrum_coverage_conditions(
            modes=state_eigenspectrum_modes,
            scales=state_eigenspectrum_scales,
            **common_kwargs,
        )
        + r20_observer_error_state_coverage_conditions(
            modes=observer_error_modes,
            scales=observer_error_scales,
            **common_kwargs,
        )
    )


def _coverage_metadata(condition: dict[str, Any]) -> dict[str, Any] | None:
    for key, family in (
        ("state_eigenspectrum_coverage", "state_eigenspectrum"),
        ("eigenspectrum_state_coverage", "state_eigenspectrum"),
        ("eigenspectrum_coverage", "state_eigenspectrum"),
        ("observer_error_state_coverage", "observer_error"),
        ("observer_error_coverage", "observer_error"),
    ):
        coverage = condition.get(key)
        if coverage is None:
            continue
        objective = coverage.get("objective", "state")
        if family == "state_eigenspectrum" and objective == "trajectory":
            return None
        return {
            "family": family,
            "objective": objective,
            "modes": coverage.get("n_modes", coverage.get("modes")),
            "scale": coverage.get("scale"),
            "weight": coverage.get("weight"),
            "reference": coverage.get("reference"),
        }
    return None


def _is_trajectory_eigenspectrum_coverage(condition: dict[str, Any]) -> bool:
    for key in (
        "state_eigenspectrum_coverage",
        "eigenspectrum_state_coverage",
        "eigenspectrum_coverage",
    ):
        coverage = condition.get(key)
        if coverage is not None and coverage.get("objective") == "trajectory":
            return True
    return False


def _optimizer_label(fit: dict[str, Any]) -> str:
    condition = fit.get("condition", {})
    optimizer = condition.get("optimizer", "unknown")
    if optimizer.startswith("adamw"):
        return f"{optimizer}_lr_{condition.get('learning_rate')}"
    return "lbfgsb_spline_whitened"


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

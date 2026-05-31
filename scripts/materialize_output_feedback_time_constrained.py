"""Materialize the 87edaae smooth time-basis output-feedback bridge."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import materialize_output_feedback_failure_decomposition as failure
import materialize_output_feedback_sweep_certificates as certificates
import rlrmp.analysis.output_feedback_time_constrained as time_constrained
from rlrmp.analysis.cs_game_card import materialize_reference
from rlrmp.analysis.output_feedback import OutputFeedbackConfig
from rlrmp.analysis.output_feedback_time_constrained import (
    ISSUE_ID,
    SPLINE_RANKS,
    render_markdown,
    timed_run,
    write_basic_outputs,
)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, arrays = materialize(
        ranks=_parse_ints(args.ranks),
        fit_ranks=_parse_ints(args.fit_ranks) if args.fit_ranks else None,
        adamw_lrs=_parse_floats(args.adamw_lrs),
        lbfgsb_maxiter=args.lbfgsb_maxiter,
        adamw_steps=args.adamw_steps,
        polish_maxiter=args.polish_maxiter,
        manifest_path=args.manifest_output,
        include_r12_coverage=args.include_r12_coverage,
        r12_coverage_rank=args.r12_coverage_rank,
        r12_state_eigenspectrum_modes=_parse_ints(args.r12_state_eigenspectrum_modes),
        r12_state_eigenspectrum_scales=_parse_floats(args.r12_state_eigenspectrum_scales),
        r12_observer_error_modes=_parse_ints(args.r12_observer_error_modes),
        r12_observer_error_scales=_parse_floats(args.r12_observer_error_scales),
        r12_coverage_weight=args.r12_coverage_weight,
    )
    write_result(
        summary,
        arrays=arrays,
        note_path=args.note_output,
        manifest_path=args.manifest_output,
        artifact_path=args.artifact_output,
    )
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")
    print(f"Wrote {args.artifact_output}")


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
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run training plus standard-certificate/failure adapters."""

    summary, arrays = _timed_run(
        {
            "ranks": ranks,
            "fit_ranks": fit_ranks,
            "adamw_lrs": adamw_lrs,
            "lbfgsb_maxiter": lbfgsb_maxiter,
            "adamw_steps": adamw_steps,
            "polish_maxiter": polish_maxiter,
        },
        include_r12_coverage=include_r12_coverage,
        coverage_kwargs={
            "include_r12_coverage": True,
            "r12_coverage_rank": r12_coverage_rank,
            "r12_state_eigenspectrum_modes": r12_state_eigenspectrum_modes,
            "r12_state_eigenspectrum_scales": r12_state_eigenspectrum_scales,
            "r12_observer_error_modes": r12_observer_error_modes,
            "r12_observer_error_scales": r12_observer_error_scales,
            "r12_coverage_weight": r12_coverage_weight,
            "coverage_rank": r12_coverage_rank,
            "state_eigenspectrum_modes": r12_state_eigenspectrum_modes,
            "state_eigenspectrum_scales": r12_state_eigenspectrum_scales,
            "observer_error_modes": r12_observer_error_modes,
            "observer_error_scales": r12_observer_error_scales,
            "coverage_weight": r12_coverage_weight,
        },
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
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
                "for the r=12 state-coverage follow-up."
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


def _timed_run(
    base_kwargs: dict[str, Any],
    *,
    include_r12_coverage: bool,
    coverage_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not include_r12_coverage:
        return timed_run(**base_kwargs)

    helper_conditions = _r12_coverage_conditions(base_kwargs, coverage_kwargs)
    if helper_conditions is not None:
        run_kwargs = base_kwargs.copy()
        rank = int(coverage_kwargs["r12_coverage_rank"])
        run_kwargs["ranks"] = tuple(dict.fromkeys((*run_kwargs["ranks"], rank)))
        run_kwargs["coverage_conditions"] = helper_conditions
        return timed_run(**run_kwargs)

    kwargs = base_kwargs | coverage_kwargs
    for hook_name in (
        "timed_run_with_r12_coverage",
        "timed_run_r12_coverage",
        "timed_run_with_coverage",
    ):
        hook = getattr(time_constrained, hook_name, None)
        if callable(hook):
            return hook(**_accepted_kwargs(hook, kwargs))
    if _accepts_keyword(timed_run, "include_r12_coverage"):
        return timed_run(**_accepted_kwargs(timed_run, kwargs))
    raise RuntimeError(
        "--include-r12-coverage requires the analysis module to expose a coverage-aware "
        "timed_run hook: timed_run_with_r12_coverage, timed_run_r12_coverage, "
        "timed_run_with_coverage, or timed_run(..., include_r12_coverage=...)."
    )


def _r12_coverage_conditions(
    base_kwargs: dict[str, Any],
    coverage_kwargs: dict[str, Any],
) -> tuple[Any, ...] | None:
    state_hook = getattr(time_constrained, "r12_state_eigenspectrum_coverage_conditions", None)
    observer_hook = getattr(time_constrained, "r12_observer_error_state_coverage_conditions", None)
    if not callable(state_hook) or not callable(observer_hook):
        return None
    learning_rate = base_kwargs["adamw_lrs"][-1]
    common_kwargs = {
        "weight": coverage_kwargs["r12_coverage_weight"],
        "maxiter": base_kwargs["adamw_steps"],
        "learning_rate": learning_rate,
        "polish_maxiter": base_kwargs["polish_maxiter"],
    }
    state = state_hook(
        **_accepted_kwargs(
            state_hook,
            common_kwargs
            | {
                "modes": coverage_kwargs["r12_state_eigenspectrum_modes"],
                "scales": coverage_kwargs["r12_state_eigenspectrum_scales"],
            },
        )
    )
    observer = observer_hook(
        **_accepted_kwargs(
            observer_hook,
            common_kwargs
            | {
                "modes": coverage_kwargs["r12_observer_error_modes"],
                "scales": coverage_kwargs["r12_observer_error_scales"],
            },
        )
    )
    return tuple(state) + tuple(observer)


def _accepted_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _accepts_keyword(func: Any, keyword: str) -> bool:
    signature = inspect.signature(func)
    return keyword in signature.parameters or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
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

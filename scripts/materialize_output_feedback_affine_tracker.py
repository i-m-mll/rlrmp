"""LEGACY (frozen 2026-07-03, issue 64d5f13).

This materializer is not contract-native: it predates the feedbax recipe,
bundle, and manifest contracts. It may not run without deliberate realignment.
Do not copy it as a pattern for new analyses. The port-or-delete decision is
deferred to the report-stage era (feedbax 132f98c) / publication.

Materialize the 50c260d affine tracker output-feedback bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import materialize_output_feedback_failure_decomposition as failure
from rlrmp.analysis.pipelines.output_feedback_affine_tracker import (
    ISSUE_ID,
    DEFAULT_SPLINE_RANK,
    render_markdown,
    timed_run,
    write_basic_outputs,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


NOTE_PATH = REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_affine_tracker.md"
MANIFEST_PATH = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_affine_tracker_manifest.json"
)
ARTIFACT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "output_feedback_affine_tracker"
    / "output_feedback_affine_tracker.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--no-selected-coverage", action="store_true")
    parser.add_argument("--spline-rank", type=int, default=DEFAULT_SPLINE_RANK)
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--artifact-output", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, arrays = materialize(
        maxiter=args.maxiter,
        include_selected_coverage=not args.no_selected_coverage,
        manifest_path=args.manifest_output,
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
    maxiter: int = 200,
    include_selected_coverage: bool = True,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run affine tracker rows plus failure-decomposition adapter."""

    summary, arrays = timed_run(
        maxiter=maxiter,
        include_selected_coverage=include_selected_coverage,
        manifest_path=manifest_path,
    )
    entries = _row_entries(summary)
    failure_rows = failure.failure_rows_from_manifest_entries(
        entries=entries,
        arrays=arrays,
        standard_rows={"standard_certificate": summary["standard_certificate"]},
        default_source_group="affine_tracker",
    )
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
    mkdir_p(note_path.parent)
    note_path.write_text(render_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for fit in summary["fits"]:
        condition = fit["condition"]
        entries.append(
            {
                "fit": fit,
                "array_prefix": fit["label"],
                "run_parts": ("affine_tracker", fit["label"]),
                "source_group": condition.get("training_distribution", "affine_tracker"),
                "parameters": {
                    "controller_family": "affine_tracker",
                    "row_kind": condition["row_kind"],
                    "train_feedforward": condition["train_feedforward"],
                    "train_gain": condition["train_gain"],
                    "gain_basis_rank": condition["gain_basis_rank"],
                    "coverage": condition.get("eigenspectrum_coverage")
                    or condition.get("observer_error_coverage"),
                },
            }
        )
    return entries


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

"""Materialize GRU observation-action map-error decomposition sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.gru_map_error_decomposition import (
    DEFAULT_LABEL,
    DEFAULT_STANDARD_MANIFEST,
    SOURCE_ISSUE_ID,
    materialize_gru_map_error_decomposition,
    write_map_error_decomposition_result,
)
from rlrmp.paths import REPO_ROOT


def main() -> None:
    """Run the map-error decomposition materializer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=SOURCE_ISSUE_ID)
    parser.add_argument(
        "--standard-manifest",
        type=Path,
        default=DEFAULT_STANDARD_MANIFEST,
        help="Existing GRU standard-certificate manifest with source run IDs.",
    )
    parser.add_argument("--run-id", action="append", help="Source run ID. May repeat.")
    parser.add_argument(
        "--final-checkpoints",
        action="store_true",
        help="Use final checkpoints instead of validation-selected checkpoints.",
    )
    parser.add_argument(
        "--alignment-basis",
        choices=("raw_cartesian", "static_reach_aligned", "auto_static_reach_aligned"),
        default="raw_cartesian",
        help=(
            "Map-decomposition basis. Use static_reach_aligned for fixed/static targets; "
            "raw_cartesian preserves the historical decomposition."
        ),
    )
    parser.add_argument(
        "--reference-feedback-basis",
        choices=("auto", "raw_delayed_position_velocity"),
        default="auto",
        help=(
            "Analytical reference feedback basis. With auto, target-relative GRU rows "
            "convert the raw delayed position/velocity reference into the controller-visible "
            "target-relative sign convention before comparison."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / SOURCE_ISSUE_ID
            / "notes"
            / f"gru_map_error_decomposition_{DEFAULT_LABEL}.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / SOURCE_ISSUE_ID
            / "notes"
            / f"gru_map_error_decomposition_{DEFAULT_LABEL}.md"
        ),
    )
    args = parser.parse_args()

    result = materialize_gru_map_error_decomposition(
        standard_manifest_path=args.standard_manifest,
        experiment=args.experiment,
        run_ids=tuple(args.run_id) if args.run_id else None,
        use_validation_selected_checkpoints=not args.final_checkpoints,
        alignment_basis=args.alignment_basis,
        reference_feedback_basis=args.reference_feedback_basis,
        top_k=args.top_k,
    )
    write_map_error_decomposition_result(
        result,
        markdown_path=args.markdown_output,
        json_path=args.json_output,
    )
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()

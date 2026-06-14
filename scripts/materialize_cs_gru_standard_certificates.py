"""Materialize standard-certificate rows for C&S GRU pilots."""

from __future__ import annotations

import argparse
from pathlib import Path

from feedbax.analysis.specs import execute_analysis_run_spec
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import (
    MATERIALIZER_ISSUE_ID,
    MANIFEST_PATH,
    NOTE_PATH,
    RUN_IDS,
    SOURCE_ISSUE_ID,
)
from rlrmp.analysis.declarative_materialization import (
    gru_standard_certificate_spec,
    register_certificate_analysis_recipes,
)


def main() -> None:
    """Run the C&S GRU standard-certificate materializer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=SOURCE_ISSUE_ID)
    parser.add_argument(
        "--run-id",
        action="append",
        help="Run identifier to materialize. May be passed more than once.",
    )
    parser.add_argument(
        "--note-output",
        type=Path,
        default=None,
        help=(
            "Tracked markdown note path. Defaults to "
            "results/<experiment>/notes/gru_standard_certificates.md"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=(
            "Tracked JSON manifest path. Defaults to "
            "results/<experiment>/notes/gru_standard_certificates_manifest.json"
        ),
    )
    parser.add_argument(
        "--validation-selected-checkpoints",
        action="store_true",
        help="Use validation-selected per-replicate checkpoints instead of final checkpoints.",
    )
    parser.add_argument(
        "--feedbax-runs-root",
        type=Path,
        default=None,
        help="Feedbax manifest/artifact root. Defaults to Feedbax's manifest root.",
    )
    args = parser.parse_args()

    run_ids = tuple(args.run_id) if args.run_id else RUN_IDS
    note_output = args.note_output or (
        NOTE_PATH
        if args.experiment == SOURCE_ISSUE_ID
        else Path("results") / args.experiment / "notes" / "gru_standard_certificates.md"
    )
    manifest_output = args.manifest_output or (
        MANIFEST_PATH
        if args.experiment == SOURCE_ISSUE_ID
        else Path("results") / args.experiment / "notes" / "gru_standard_certificates_manifest.json"
    )
    register_certificate_analysis_recipes(replace=True)
    spec = gru_standard_certificate_spec(
        run_ids=run_ids,
        experiment=args.experiment,
        materializer_issue_id=MATERIALIZER_ISSUE_ID,
        use_validation_selected_checkpoints=args.validation_selected_checkpoints,
        note_output=note_output,
        manifest_output=manifest_output,
    )
    manifest, feedbax_manifest_path = execute_analysis_run_spec(
        spec,
        root=args.feedbax_runs_root,
        issues=[MATERIALIZER_ISSUE_ID],
    )
    print(f"Wrote {note_output}")
    print(f"Wrote {manifest_output}")
    print(f"Wrote {feedbax_manifest_path}")
    print(f"Feedbax analysis manifest: {manifest.id}")


if __name__ == "__main__":
    main()

"""Materialize standard-certificate rows for the 30f2313 C&S GRU pilots."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.cs_gru_standard_materialization import (
    MANIFEST_PATH,
    NOTE_PATH,
    RUN_IDS,
    materialize_gru_standard_result,
    write_gru_standard_result,
)


def main() -> None:
    """Run the C&S GRU standard-certificate materializer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        action="append",
        choices=RUN_IDS,
        help="Run identifier to materialize. May be passed more than once.",
    )
    parser.add_argument(
        "--note-output",
        type=Path,
        default=NOTE_PATH,
        help=f"Tracked markdown note path. Default: {NOTE_PATH}",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Tracked JSON manifest path. Default: {MANIFEST_PATH}",
    )
    args = parser.parse_args()

    run_ids = tuple(args.run_id) if args.run_id else RUN_IDS
    result = materialize_gru_standard_result(run_ids=run_ids)
    write_gru_standard_result(
        result,
        note_path=args.note_output,
        manifest_path=args.manifest_output,
    )
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")


if __name__ == "__main__":
    main()

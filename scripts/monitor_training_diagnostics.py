#!/usr/bin/env python
"""Summarize live rlrmp training diagnostics for local or RunPod runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.training_diagnostics import (
    render_text,
    summarize_output_dir,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print latest training diagnostics from one or more output directories.",
    )
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    summaries = [summarize_output_dir(path) for path in args.output_dirs]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        for summary in summaries:
            print(render_text(summary))


if __name__ == "__main__":
    main()

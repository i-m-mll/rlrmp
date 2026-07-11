"""Build supported Feedbax CLI row commands from governed RLRMP run specs."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.paths import REPO_ROOT
from rlrmp.train.feedbax_cli_rows import build_feedbax_cli_rows_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_feedbax_cli_rows_manifest(args.source, args.output, repo_root=REPO_ROOT)


if __name__ == "__main__":
    main()

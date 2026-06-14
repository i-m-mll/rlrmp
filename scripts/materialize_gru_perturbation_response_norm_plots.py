#!/usr/bin/env python
"""Materialize GRU perturbation-response norm Plotly figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.gru_perturbation_response_norm_plots import (
    DEFAULT_ASSET_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_NOTE_PATH,
    DEFAULT_REGENERATION_SPEC_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SOURCE_MANIFEST,
    materialize_response_norm_plots,
)
from rlrmp.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--asset-dir", type=Path, default=Path(DEFAULT_ASSET_DIR))
    parser.add_argument("--note-path", type=Path, default=Path(DEFAULT_NOTE_PATH))
    parser.add_argument("--manifest-path", type=Path, default=Path(DEFAULT_MANIFEST_PATH))
    parser.add_argument(
        "--regeneration-spec-path",
        type=Path,
        default=Path(DEFAULT_REGENERATION_SPEC_PATH),
    )
    parser.add_argument(
        "--run-id-contains",
        action="append",
        default=[],
        help="Only include runs whose run id contains this substring. May be repeated.",
    )
    parser.add_argument(
        "--no-extlqg",
        action="store_true",
        help="Skip deterministic extLQG curve reconstruction.",
    )
    args = parser.parse_args()

    manifest = materialize_response_norm_plots(
        source_manifest_path=args.source_manifest,
        results_dir=args.results_dir,
        asset_dir=args.asset_dir,
        note_path=args.note_path,
        manifest_path=args.manifest_path,
        regeneration_spec_path=args.regeneration_spec_path,
        repo_root=REPO_ROOT,
        reconstruct_extlqg=not args.no_extlqg,
        run_id_contains=tuple(args.run_id_contains),
    )
    print(
        "Wrote "
        f"{manifest['figure_count']} perturbation-response norm figure(s) to "
        f"{manifest.get('asset_dir', DEFAULT_ASSET_DIR)}."
    )


if __name__ == "__main__":
    main()

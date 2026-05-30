"""Read bridge manifests and write compact comparison summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.bridge_aggregation import (
    render_bridge_summary_markdown,
    summarize_bridge_manifests,
    write_bridge_summary,
    write_bridge_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="BridgeRunManifest JSON files to validate and aggregate.",
    )
    parser.add_argument("--json-output", type=Path, help="Path for the summary JSON.")
    parser.add_argument("--markdown-output", type=Path, help="Path for the Markdown table.")
    parser.add_argument(
        "--required-artifact",
        action="append",
        default=[],
        help="Artifact label that must be present. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--required-certificate",
        action="append",
        default=[],
        help="Certificate component name that must be present. Repeat for multiple names.",
    )
    parser.add_argument(
        "--allow-missing-certificate-components",
        action="store_true",
        help="Allow required certificate components with status 'missing'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_bridge_manifests(
        args.manifests,
        required_artifact_labels=args.required_artifact,
        required_certificate_labels=args.required_certificate,
        allow_missing_certificate_components=args.allow_missing_certificate_components,
    )
    if args.json_output is not None:
        write_bridge_summary(summary, args.json_output)
        print(f"Wrote {args.json_output}")
    if args.markdown_output is not None:
        write_bridge_summary_markdown(summary["rows"], args.markdown_output)
        print(f"Wrote {args.markdown_output}")
    if args.json_output is None and args.markdown_output is None:
        print(render_bridge_summary_markdown(summary["rows"]), end="")


if __name__ == "__main__":
    main()

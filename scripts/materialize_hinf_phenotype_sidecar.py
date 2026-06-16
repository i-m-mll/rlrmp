"""Materialize the GRU H-infinity phenotype sidecar from existing diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.hinf_phenotype_sidecar import (
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_MARKDOWN,
    DEFAULT_REGENERATION_SPEC,
    DEFAULT_SCOPE,
    build_hinf_phenotype_sidecar,
    load_hinf_phenotype_sources,
    write_hinf_phenotype_sidecar,
)
from rlrmp.paths import REPO_ROOT


def main() -> None:
    """Run the sidecar materializer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--standard-certificate", type=Path)
    parser.add_argument("--objective-comparator", type=Path)
    parser.add_argument("--perturbation-response", type=Path)
    parser.add_argument("--feedback-ablation", type=Path)
    parser.add_argument("--map-error-decomposition", type=Path)
    parser.add_argument("--evaluation-diagnostics", type=Path)
    parser.add_argument("--induced-gain", type=Path)
    parser.add_argument("--exact-audit", type=Path)
    parser.add_argument("--broad-epsilon-attribution", type=Path)
    parser.add_argument(
        "--paired-run",
        action="append",
        type=_paired_run_arg,
        default=[],
        metavar="BASELINE=ROBUST",
        help=("Explicit interpretive baseline-to-robust run pair. Repeat for multiple pairs."),
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument(
        "--regeneration-spec-path",
        type=Path,
        default=DEFAULT_REGENERATION_SPEC,
    )
    args = parser.parse_args()

    sources = load_hinf_phenotype_sources(
        {
            "standard_certificate": args.standard_certificate,
            "objective_comparator": args.objective_comparator,
            "perturbation_response": args.perturbation_response,
            "feedback_ablation": args.feedback_ablation,
            "map_error_decomposition": args.map_error_decomposition,
            "evaluation_diagnostics": args.evaluation_diagnostics,
            "induced_gain": args.induced_gain,
            "exact_audit": args.exact_audit,
            "broad_epsilon_attribution": args.broad_epsilon_attribution,
        }
    )
    sidecar = build_hinf_phenotype_sidecar(
        sources=sources,
        scope=args.scope,
        paired_run_ids=dict(args.paired_run),
    )
    write_hinf_phenotype_sidecar(
        sidecar,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        regeneration_spec_path=args.regeneration_spec_path,
        repo_root=REPO_ROOT,
    )
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")


def _paired_run_arg(value: str) -> tuple[str, str]:
    baseline, separator, robust = value.partition("=")
    if not separator or not baseline or not robust:
        raise argparse.ArgumentTypeError(
            "--paired-run must use BASELINE=ROBUST with both run IDs present"
        )
    return baseline, robust


if __name__ == "__main__":
    main()

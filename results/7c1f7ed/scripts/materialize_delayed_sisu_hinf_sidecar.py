"""Materialize the delayed SISU H-infinity phenotype sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlrmp.analysis.pipelines.hinf_phenotype_sidecar import (
    build_hinf_phenotype_sidecar,
    load_hinf_phenotype_sources,
    write_hinf_phenotype_sidecar,
)
from rlrmp.paths import REPO_ROOT


DEFAULT_SCOPE = "delayed_sisu_two_rows_final"
DEFAULT_OUTPUT_STEM = "hinf_phenotype_sidecar_delayed_sisu_final"
DEFAULT_STANDARD_CERTIFICATE = (
    REPO_ROOT
    / "results"
    / "7c1f7ed"
    / "notes"
    / "gru_standard_certificates_delayed_sisu_final_manifest.json"
)
DEFAULT_EVALUATION_DIAGNOSTICS = (
    REPO_ROOT
    / "results"
    / "7c1f7ed"
    / "notes"
    / "gru_evaluation_diagnostics_delayed_sisu_final.json"
)
DEFAULT_PERTURBATION_RESPONSE = (
    REPO_ROOT
    / "results"
    / "7c1f7ed"
    / "notes"
    / "delayed_sisu_perturbation_class_comparison.json"
)


def main() -> None:
    """Run the delayed SISU H-infinity phenotype sidecar materializer."""

    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_base = repo_root / "results" / args.issue / "notes" / args.output_stem
    sources = load_hinf_phenotype_sources(
        {
            "standard_certificate": args.standard_certificate,
            "evaluation_diagnostics": args.evaluation_diagnostics,
            "perturbation_response": args.perturbation_response,
            "objective_comparator": None,
            "feedback_ablation": None,
            "map_error_decomposition": None,
            "induced_gain": None,
            "exact_audit": None,
            "broad_epsilon_attribution": None,
        },
        repo_root=repo_root,
    )
    sidecar = build_hinf_phenotype_sidecar(
        sources=sources,
        issue=args.issue,
        scope=args.scope,
    )
    sidecar["delayed_contract_caveat"] = {
        "formal_hinf_equivalence": "not_claimed",
        "blocker": (
            "Current standard certificate reports 6D delayed feedback/force-filter "
            "GraphSpec versus 8D output-feedback analytical reference response-map "
            "mismatch, so this sidecar is interpretive phenotype evidence only."
        ),
    }
    write_hinf_phenotype_sidecar(
        sidecar,
        json_path=output_base.with_suffix(".json"),
        markdown_path=output_base.with_suffix(".md"),
        regeneration_spec_path=output_base.with_name(
            f"{args.output_stem}_regeneration_spec.json"
        ),
        repo_root=repo_root,
    )
    print(output_base.with_suffix(".md"))
    print(output_base.with_suffix(".json"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--issue", default="7c1f7ed")
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--standard-certificate", type=Path, default=DEFAULT_STANDARD_CERTIFICATE)
    parser.add_argument("--evaluation-diagnostics", type=Path, default=DEFAULT_EVALUATION_DIAGNOSTICS)
    parser.add_argument("--perturbation-response", type=Path, default=DEFAULT_PERTURBATION_RESPONSE)
    return parser.parse_args()


if __name__ == "__main__":
    main()

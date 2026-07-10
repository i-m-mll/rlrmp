"""Materialize the special SISU-conditioned e4800d6 analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rlrmp
from feedbax.analysis.evaluation import execute_evaluation_run_spec
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.plugins import EXPERIMENT_REGISTRY

from rlrmp.analysis.declarative_materialization import (
    register_certificate_analysis_recipes,
    sisu_spectrum_evaluation_spec,
    sisu_spectrum_spec,
)
from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import (
    DEFAULT_N_ROLLOUT_TRIALS,
    DEFAULT_SISU_LEVELS,
    DEFAULT_TOPIC,
    SISU_SPECTRUM_COMPACT_ARRAYS_ROLE,
    SISU_SPECTRUM_MANIFEST_ROLE,
    SISU_SPECTRUM_NOTE_ROLE,
)
from rlrmp.paths import REPO_ROOT


EXPERIMENT = "e4800d6"
ISSUE_ID = "dc96336"
TOPIC = DEFAULT_TOPIC
DEFAULT_RUN_IDS = (
    "cs_gru_h0_sisu_spectrum__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64",
    "cs_gru_h0_sisu_spectrum__effective_020a65b_pgd_radius_lr3e-3_clip5_b64",
)
DEFAULT_LABELS = (
    "A: raw strong gamma=1.05 radius",
    "B: effective 020a65b PGD radius",
)


def main() -> None:
    """Run the special SISU materialization."""

    args = parse_args()
    if "rlrmp" not in EXPERIMENT_REGISTRY.get_package_names():
        rlrmp.register_experiment_package(EXPERIMENT_REGISTRY)
    repo_root = args.repo_root.resolve()
    manifest_root = args.feedbax_runs_root or (
        repo_root / "_artifacts" / args.experiment / args.output_stem / "feedbax_runs"
    )
    note_output = repo_root / "results" / args.experiment / "notes" / f"{args.output_stem}.md"
    register_certificate_analysis_recipes(replace=True)
    evaluation_manifest, evaluation_manifest_path = execute_evaluation_run_spec(
        sisu_spectrum_evaluation_spec(
            experiment=args.experiment,
            run_ids=args.run_ids,
            labels=args.labels,
            topic=args.topic,
            sisu_levels=tuple(args.sisu_levels),
            n_rollout_trials=args.n_rollout_trials,
            reference_samples=args.reference_samples,
            output_stem=args.output_stem,
            note_output=note_output,
        ),
        root=manifest_root,
    )
    analysis_manifest, analysis_manifest_path = execute_analysis_run_spec(
        sisu_spectrum_spec(
            evaluation_manifest_id=evaluation_manifest.id,
            evaluation_manifest_uri=evaluation_manifest_path,
        ),
        root=manifest_root,
        issues=[ISSUE_ID, args.experiment],
        fig_dump_path=repo_root / "_artifacts" / args.experiment / "figures" / args.topic,
        fig_dump_formats=("html",),
    )

    manifest_ref = _artifact_for_role(analysis_manifest, SISU_SPECTRUM_MANIFEST_ROLE)
    note_ref = _artifact_for_role(analysis_manifest, SISU_SPECTRUM_NOTE_ROLE)
    compact_ref = _artifact_for_role(analysis_manifest, SISU_SPECTRUM_COMPACT_ARRAYS_ROLE)
    figure_refs = _artifacts_for_role(analysis_manifest, "figure")

    print(
        json.dumps(
            {
                "evaluation_manifest": str(evaluation_manifest_path),
                "analysis_manifest": str(analysis_manifest_path),
                "manifest": manifest_ref.uri,
                "note": note_ref.uri,
                "figures": [artifact.uri for artifact in figure_refs],
                "compact_arrays": compact_ref.uri,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--feedbax-runs-root", type=Path, default=None)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--output-stem", default="sisu_spectrum_special")
    parser.add_argument("--run-id", dest="run_ids", action="append", default=None)
    parser.add_argument("--label", dest="labels", action="append", default=None)
    parser.add_argument(
        "--sisu-level",
        dest="sisu_levels",
        action="append",
        type=float,
        default=None,
    )
    parser.add_argument("--n-rollout-trials", type=int, default=DEFAULT_N_ROLLOUT_TRIALS)
    parser.add_argument("--reference-samples", type=int, default=128)
    args = parser.parse_args()
    args.run_ids = args.run_ids or list(DEFAULT_RUN_IDS)
    args.labels = args.labels or list(DEFAULT_LABELS)
    args.sisu_levels = args.sisu_levels or list(DEFAULT_SISU_LEVELS)
    if len(args.run_ids) != len(args.labels):
        raise SystemExit("--run-id and --label must be passed the same number of times")
    return args


def _artifact_for_role(manifest, role: str):
    matches = _artifacts_for_role(manifest, role)
    if len(matches) != 1:
        raise ValueError(f"Expected one {role!r} artifact, found {len(matches)}")
    return matches[0]


def _artifacts_for_role(manifest, role: str):
    matches = [artifact for artifact in manifest.artifacts if artifact.role == role]
    if not matches:
        raise ValueError(f"Expected at least one {role!r} artifact")
    return matches


if __name__ == "__main__":
    main()

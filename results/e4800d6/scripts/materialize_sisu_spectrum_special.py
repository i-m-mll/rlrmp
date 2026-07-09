"""Materialize the special SISU-conditioned e4800d6 analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rlrmp
from feedbax.plugins import EXPERIMENT_REGISTRY
from feedbax.plot import save_figure

from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import (
    DEFAULT_N_ROLLOUT_TRIALS,
    DEFAULT_SISU_LEVELS,
    DEFAULT_TOPIC,
    analytical_reference_curves,
    build_manifest,
    build_velocity_profile_figure,
    evaluate_sisu_profiles,
    render_markdown,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT


EXPERIMENT = "e4800d6"
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
    profiles = evaluate_sisu_profiles(
        experiment=args.experiment,
        run_ids=tuple(args.run_ids),
        labels=tuple(args.labels),
        sisu_levels=tuple(args.sisu_levels),
        n_rollout_trials=args.n_rollout_trials,
        repo_root=repo_root,
    )
    references = analytical_reference_curves(
        n_samples=max(
            len(profiles) * args.n_rollout_trials,
            args.reference_samples,
        )
    )
    fig = build_velocity_profile_figure(profiles, references)

    spec = {
        "schema_id": "rlrmp.figure_spec.sisu_spectrum_velocity_profiles.v1",
        "experiment": args.experiment,
        "topic": args.topic,
        "run_ids": list(args.run_ids),
        "labels": list(args.labels),
        "sisu_levels": list(args.sisu_levels),
        "n_rollout_trials_per_replicate": args.n_rollout_trials,
        "reference_samples": max(
            len(profiles) * args.n_rollout_trials,
            args.reference_samples,
        ),
        "checkpoint_policy": "validation_selected_per_replicate",
        "input_contract": (
            "SISU is carried by trial_specs.inputs['input'] for these runs; "
            "epsilon is zeroed for the nominal velocity-profile comparison."
        ),
        "interpretation": (
            "Discovery-trained robustness, not teacher/distillation and not "
            "formal H-infinity equivalence."
        ),
    }
    save_figure(
        fig=fig,
        spec=spec,
        package="rlrmp",
        experiment=args.experiment,
        topic=args.topic,
        extra_packages=["rlrmp"],
    )

    compact_npz = (
        repo_root
        / "_artifacts"
        / args.experiment
        / args.output_stem
        / "sisu_velocity_profile_curves.npz"
    )
    write_compact_arrays(profiles=profiles, references=references, path=compact_npz)

    figure_spec = repo_root / "results" / args.experiment / "figures" / args.topic / "spec.json"
    figure_html = repo_root / "_artifacts" / args.experiment / "figures" / args.topic / "figure.html"
    manifest = build_manifest(
        experiment=args.experiment,
        topic=args.topic,
        profiles=profiles,
        references=references,
        compact_npz_path=compact_npz,
        figure_spec_path=figure_spec,
        figure_html_path=figure_html,
        repo_root=repo_root,
        sisu_levels=tuple(args.sisu_levels),
        n_rollout_trials=args.n_rollout_trials,
    )
    notes_dir = repo_root / "results" / args.experiment / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = notes_dir / f"{args.output_stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    note_path = notes_dir / f"{args.output_stem}.md"
    write_note(note_path, manifest)

    print(
        json.dumps(
            {
                "manifest": str(manifest_path.relative_to(repo_root)),
                "note": str(note_path.relative_to(repo_root)),
                "figure_spec": str(figure_spec.relative_to(repo_root)),
                "figure_html": str(figure_html.relative_to(repo_root)),
                "compact_arrays": str(compact_npz.relative_to(repo_root)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
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


def write_compact_arrays(
    *,
    profiles: Sequence[Any],
    references: Sequence[Any],
    path: Path,
) -> None:
    """Write compact regenerable velocity profile arrays for this experiment."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for profile_idx, profile in enumerate(profiles):
        prefix = f"run_{profile_idx}"
        arrays[f"{prefix}_run_id"] = np.asarray(profile.run_id)
        for curve in profile.curves:
            sisu_tag = str(curve.sisu).replace(".", "p")
            arrays[f"{prefix}_sisu_{sisu_tag}_time_s"] = curve.time_s
            arrays[f"{prefix}_sisu_{sisu_tag}_mean_forward_velocity_m_s"] = (
                curve.mean_forward_velocity_m_s
            )
            arrays[f"{prefix}_sisu_{sisu_tag}_std_forward_velocity_m_s"] = (
                curve.std_forward_velocity_m_s
            )
            arrays[f"{prefix}_sisu_{sisu_tag}_replicate_mean_forward_velocity_m_s"] = (
                curve.replicate_mean_forward_velocity_m_s
            )
    for reference_idx, reference in enumerate(references):
        prefix = f"reference_{reference_idx}"
        arrays[f"{prefix}_label"] = np.asarray(reference.label)
        arrays[f"{prefix}_time_s"] = reference.time_s
        arrays[f"{prefix}_forward_velocity_m_s"] = reference.forward_velocity_m_s
        arrays[f"{prefix}_std_forward_velocity_m_s"] = reference.std_forward_velocity_m_s
    np.savez_compressed(path, **arrays)


def write_note(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write or update the SISU special Markdown note for this experiment."""

    update_marked_section(path, "sisu_spectrum_special", render_markdown(manifest))


if __name__ == "__main__":
    main()

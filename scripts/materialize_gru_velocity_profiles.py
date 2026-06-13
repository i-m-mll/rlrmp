"""Materialize GRU stochastic forward-velocity profiles only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.analysis.gru_checkpoint_selection import (
    materialize_validation_selected_checkpoint_manifest,
)
from rlrmp.analysis.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    cs_output_feedback_reference_profiles,
    evaluate_stochastic_forward_velocity_profile,
    resolve_run_inputs,
    write_velocity_figure,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    labels = tuple(args.label or args.run_id)
    output_dir = args.output_dir or (
        REPO_ROOT / "_artifacts" / args.experiment / "figures" / args.topic
    )
    mkdir_p(output_dir)
    runs = resolve_run_inputs(
        experiment=args.experiment,
        run_ids=tuple(args.run_id),
        labels=labels,
    )
    selection_manifest = (
        materialize_validation_selected_checkpoint_manifest(
            experiment=args.experiment,
            run_ids=tuple(args.run_id),
            preferred_manifest_path=args.preferred_checkpoint_manifest,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if args.preferred_checkpoint_manifest is not None
                else "sparse_history"
            ),
        )
        if args.validation_selected_checkpoints
        else None
    )
    if args.delayed_eval_bank == "both":
        summaries = {}
        for bank_kind in ("no_catch", "catch"):
            summaries[bank_kind] = _materialize_profiles_for_bank(
                args=args,
                runs=runs,
                labels=labels,
                output_dir=output_dir / bank_kind,
                selection_manifest=selection_manifest,
                delayed_eval_bank_kind=bank_kind,
                include_reference=args.include_reference and bank_kind == "no_catch",
            )
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return
    delayed_eval_bank_kind = (
        None if args.delayed_eval_bank == "none" else args.delayed_eval_bank
    )
    summary = _materialize_profiles_for_bank(
        args=args,
        runs=runs,
        labels=labels,
        output_dir=output_dir,
        selection_manifest=selection_manifest,
        delayed_eval_bank_kind=delayed_eval_bank_kind,
        include_reference=args.include_reference,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _materialize_profiles_for_bank(
    *,
    args: argparse.Namespace,
    runs: list[Any],
    labels: tuple[str, ...],
    output_dir: Path,
    selection_manifest: dict[str, Any] | None,
    delayed_eval_bank_kind: str | None,
    include_reference: bool,
) -> dict[str, Any]:
    """Materialize velocity profiles for one delayed eval-bank lens."""

    mkdir_p(output_dir)
    profiles = [
        evaluate_stochastic_forward_velocity_profile(
            run,
            n_rollout_trials=args.n_rollout_trials,
            use_validation_selected_checkpoints=args.validation_selected_checkpoints,
            experiment=args.experiment,
            preferred_checkpoint_manifest_path=args.preferred_checkpoint_manifest,
            pre_go_context_steps=args.pre_go_context_steps,
            delayed_eval_bank_kind=delayed_eval_bank_kind,
        )
        for run in runs
    ]
    references = (
        cs_output_feedback_reference_profiles(
            n_samples=max(profile.n_pooled_samples for profile in profiles)
        )
        if include_reference
        else ()
    )
    velocity_file = write_velocity_figure(profiles, output_dir=output_dir, references=references)
    summary = _summary_payload(
        experiment=args.experiment,
        run_ids=tuple(args.run_id),
        labels=labels,
        velocity_file=velocity_file,
        profiles=profiles,
        selection_manifest=selection_manifest,
        references=references,
        n_rollout_trials=args.n_rollout_trials,
        pre_go_context_steps=args.pre_go_context_steps,
        delayed_eval_bank_kind=delayed_eval_bank_kind,
    )
    summary_path = output_dir / "velocity_profile_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--topic", default="delayed_velocity_profiles")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-rollout-trials", type=int, default=DEFAULT_N_ROLLOUT_TRIALS)
    parser.add_argument("--validation-selected-checkpoints", action="store_true")
    parser.add_argument("--preferred-checkpoint-manifest", type=Path)
    parser.add_argument("--pre-go-context-steps", type=int)
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help=(
            "Overlay stochastic analytical extLQG references. With "
            "--delayed-eval-bank both, references are written only for no_catch."
        ),
    )
    parser.add_argument(
        "--delayed-eval-bank",
        choices=("none", "no_catch", "catch", "both"),
        default="none",
        help=(
            "Use a fixed delayed-reach evaluation bank instead of the run's sparse "
            "validation_trials. 'both' writes separate no_catch/ and catch/ outputs."
        ),
    )
    return parser


def _summary_payload(
    *,
    experiment: str,
    run_ids: tuple[str, ...],
    labels: tuple[str, ...],
    velocity_file: Path,
    profiles: list[Any],
    selection_manifest: dict[str, Any] | None,
    references: tuple[Any, ...],
    n_rollout_trials: int,
    pre_go_context_steps: int | None,
    delayed_eval_bank_kind: str | None,
) -> dict[str, Any]:
    """Return a compact JSON sidecar for the velocity-only figure."""

    return {
        "schema_version": "rlrmp.gru_velocity_profiles.v1",
        "issue": experiment,
        "runs": dict(zip(labels, run_ids, strict=True)),
        "figure": str(velocity_file),
        "n_rollout_trials": int(n_rollout_trials),
        "pre_go_context_steps": pre_go_context_steps,
        "delayed_eval_bank_kind": delayed_eval_bank_kind,
        "checkpoint_policy": (
            "validation_selected_per_replicate"
            if selection_manifest is not None
            else "final_checkpoint"
        ),
        "error_band": (
            "mean +/- 1 SD over pooled stochastic rollout trials across replicates; "
            "replicates are not averaged before the band is computed"
        ),
        "references": {
            reference.label: {
                "controller": "analytical_lqr_kalman_output_feedback",
                "display_label": reference.label,
                "observation_channel": reference.observation_channel,
                "observation_dim": int(reference.observation_dim),
                "observed_physical_indices": list(reference.observed_physical_indices),
                "gamma_factor_recorded_for_certificate": float(reference.gamma_factor),
                "n_stochastic_samples": int(reference.n_samples),
                "parity_status": reference.parity_status,
                "n_time_steps": int(reference.forward_velocity.shape[0]),
                "peak_forward_velocity_m_s": float(reference.peak_forward_velocity_m_s),
                "time_of_peak_forward_velocity_s": float(
                    reference.time_of_peak_forward_velocity_s
                ),
                "terminal_position_error_m": float(reference.terminal_position_error_m),
            }
            for reference in references
        },
        "profiles": {
            profile.label: {
                "run_id": profile.run_id,
                "time_basis": profile.time_basis,
                "alignment": profile.alignment,
                "n_replicates": int(profile.n_replicates),
                "n_rollout_trials_per_replicate": int(
                    profile.n_rollout_trials_per_replicate
                ),
                "n_pooled_samples": int(profile.n_pooled_samples),
                "n_time_steps": int(profile.mean.shape[0]),
                "peak_mean_forward_velocity_m_s": float(np.max(profile.mean)),
                "time_of_peak_mean_forward_velocity_s": float(
                    profile.time_s[int(np.argmax(profile.mean))]
                ),
                "checkpoint_selection": [
                    selection.to_json(repo_root=REPO_ROOT)
                    for selection in profile.checkpoint_selection
                ],
            }
            for profile in profiles
        },
        "checkpoint_selection": selection_manifest,
    }


if __name__ == "__main__":
    main()

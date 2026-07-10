"""Materialize delayed_movement_bank fixed-bank velocity profiles.

This issue-local script evaluates the final delayed_movement_bank checkpoint on
the corrected delayed-reach no-catch and catch banks: go cues 10..30, 20 uniform
center-out directions, 0.15 m reach length, and go-cue-aligned velocity windows.
"""

from __future__ import annotations
from rlrmp.eval import ensemble as _eval_ensemble
from rlrmp.eval.ensemble import (
    evaluate_velocity_profile as canonical_evaluate_velocity_profile,
    make_delayed_eval_bank as canonical_make_delayed_eval_bank,
)
from rlrmp.viz.colors import hex_to_rgba
from rlrmp.viz.traces import (
    add_band_trace as canonical_add_band_trace,
    add_reference_trace as canonical_add_reference_trace,
)

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from feedbax import TaskTrialSpec

from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    DEFAULT_DELAYED_REACH_GO_CUE_STEPS,
    DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import cs_output_feedback_reference_profiles
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.viz import profile_comparison_grid


canonical_movement_horizon = _eval_ensemble._canonical_movement_horizon
infer_trial_count = _eval_ensemble._infer_trial_count
infer_trial_n_time = _eval_ensemble._infer_trial_n_time
like_existing_array = _eval_ensemble._like_existing_array
like_existing_time_input = _eval_ensemble._like_existing_time_input
reach_direction = _eval_ensemble._reach_direction
remap_source_trials = _eval_ensemble._remap_source_trials
target_position_sequence = _eval_ensemble._target_position_sequence
update_cartesian_position = _eval_ensemble._update_cartesian_position
update_delayed_task_inputs = _eval_ensemble._update_delayed_task_inputs
update_initial_positions = _eval_ensemble._update_initial_positions
update_inputs = _eval_ensemble._update_inputs
update_targets = _eval_ensemble._update_targets


EXPERIMENT = "6c36536"
RUN_ID = "delayed_movement_bank"
LABEL = "delayed_movement_bank"
TOPIC = "delayed_movement_bank_velocity_profiles"
PRE_GO_CONTEXT_STEPS = 10
REFERENCE_COLOR = "#111827"
BankKind = Literal["no_catch", "catch"]


@dataclass(frozen=True)
class RunInputs:
    """Resolved local files for one run."""

    run_id: str
    label: str
    run_spec_path: Path
    artifact_dir: Path
    run_spec: dict[str, Any]


@dataclass(frozen=True)
class DelayedEvalBank:
    """A concrete delayed evaluation bank plus JSON metadata."""

    trial_specs: TaskTrialSpec
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VelocityProfile:
    """Go-cue-aligned target-radial velocity profile."""

    run_id: str
    label: str
    bank_kind: BankKind
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    replicate_mean: np.ndarray
    replicate_std: np.ndarray
    n_replicates: int
    n_trials_per_replicate: int
    alignment: dict[str, Any]
    evaluation_bank: dict[str, Any]

    @property
    def n_pooled_samples(self) -> int:
        """Return replicate x trial sample count."""

        return int(self.n_replicates * self.n_trials_per_replicate)


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    run = resolve_run_inputs(
        experiment=args.experiment,
        run_id=args.run_id,
        label=args.label,
        repo_root=args.repo_root,
    )
    output_root = args.output_dir or (
        args.repo_root / "_artifacts" / args.experiment / "figures" / args.topic
    )
    summaries: dict[str, Any] = {}
    for bank_kind in ("no_catch", "catch"):
        output_dir = output_root / bank_kind
        mkdir_p(output_dir)
        profile = evaluate_velocity_profile(
            run,
            bank_kind=bank_kind,
            go_cue_steps=tuple(args.go_cue_step),
            direction_count=args.direction_count,
            reach_length_m=args.reach_length_m,
            pre_go_context_steps=args.pre_go_context_steps,
        )
        include_reference = bool(args.include_reference and bank_kind == "no_catch")
        references = (
            cs_output_feedback_reference_profiles(n_samples=profile.n_pooled_samples)
            if include_reference
            else ()
        )
        pooled_file = write_velocity_figure(
            profile,
            output_dir=output_dir,
            references=references,
        )
        replicate_file = write_velocity_by_replicate_figure(
            profile,
            output_dir=output_dir,
            references=references,
        )
        summary = build_summary(
            run=run,
            profile=profile,
            pooled_file=pooled_file,
            replicate_file=replicate_file,
            references=references,
            output_dir=output_dir,
            direction_split_status=(
                "not_materialized: prior good/bad direction split was a "
                "diagnostic grouping for earlier no-PGD rows, not a natural "
                "movement-bank row grouping; by-replicate profiles were "
                "materialized as the cheap directly analogous variant"
            ),
        )
        summary_path = output_dir / "velocity_profile_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries[bank_kind] = summary
    print(json.dumps(summaries, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--go-cue-step",
        type=int,
        action="append",
        default=list(DEFAULT_DELAYED_REACH_GO_CUE_STEPS),
    )
    parser.add_argument(
        "--direction-count",
        type=int,
        default=DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    )
    parser.add_argument(
        "--reach-length-m",
        type=float,
        default=DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    )
    parser.add_argument("--pre-go-context-steps", type=int, default=PRE_GO_CONTEXT_STEPS)
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay extLQG references on the no-catch profile.",
    )
    return parser


def resolve_run_inputs(
    *,
    experiment: str,
    run_id: str,
    label: str,
    repo_root: Path,
) -> RunInputs:
    """Resolve the flat or legacy directory-form run spec for one run."""

    flat_path = repo_root / "results" / experiment / "runs" / f"{run_id}.json"
    legacy_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    if flat_path.exists():
        run_spec_path = flat_path
    elif legacy_path.exists():
        run_spec_path = legacy_path
    else:
        raise FileNotFoundError(f"Missing run spec for {experiment}/{run_id}")
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")
    return RunInputs(
        run_id=run_id,
        label=label,
        run_spec_path=run_spec_path,
        artifact_dir=artifact_dir,
        run_spec=json.loads(run_spec_path.read_text(encoding="utf-8")),
    )


def evaluate_velocity_profile(
    run: RunInputs,
    *,
    bank_kind: BankKind,
    go_cue_steps: Sequence[int],
    direction_count: int,
    reach_length_m: float,
    pre_go_context_steps: int,
) -> VelocityProfile:
    """Evaluate via the canonical delayed-profile implementation."""

    return canonical_evaluate_velocity_profile(
        run,
        bank_kind=bank_kind,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        reach_length_m=reach_length_m,
        pre_go_context_steps=pre_go_context_steps,
    )


def make_delayed_eval_bank(
    trial_specs: TaskTrialSpec,
    *,
    bank_kind: BankKind,
    go_cue_steps: Iterable[int],
    direction_count: int,
    reach_length_m: float,
    movement_horizon_steps: int,
) -> DelayedEvalBank:
    """Build via the canonical delayed-bank implementation."""

    return canonical_make_delayed_eval_bank(
        trial_specs,
        bank_kind=bank_kind,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        reach_length_m=reach_length_m,
        movement_horizon_steps=movement_horizon_steps,
    )


def write_velocity_figure(
    profile: VelocityProfile,
    *,
    output_dir: Path,
    references: Sequence[Any],
) -> Path:
    """Write the pooled fixed-bank velocity profile."""

    fig = profile_comparison_grid(
        n_panels=1,
        subplot_titles=[f"{profile.label} ({profile.bank_kind})"],
        vertical_spacing=0.04,
    )
    add_band_trace(
        fig,
        x=profile.time_s,
        mean=profile.mean,
        std=profile.std,
        row=1,
        color="#2563eb",
        name=profile.label,
        legendgroup="gru",
        showlegend=True,
    )
    for reference in references:
        add_reference_trace(fig, reference=reference, row=1)
    fig.add_vline(x=0.0, line={"color": "black", "dash": "dash", "width": 1}, row=1, col=1)
    fig.update_layout(
        title=f"Delayed movement-bank target-radial velocity ({profile.bank_kind})",
        width=900,
        height=520,
        margin={"l": 72, "r": 24, "t": 76, "b": 72},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=1, col=1)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)
    path = output_dir / "forward_velocity_profiles_stochastic.html"
    fig.write_html(path)
    return path


def write_velocity_by_replicate_figure(
    profile: VelocityProfile,
    *,
    output_dir: Path,
    references: Sequence[Any],
) -> Path:
    """Write replicate-resolved fixed-bank velocity profiles."""

    fig = profile_comparison_grid(
        n_panels=1,
        subplot_titles=[f"{profile.label} by replicate ({profile.bank_kind})"],
        vertical_spacing=0.04,
    )
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#be123c")
    for rep_idx in range(profile.n_replicates):
        add_band_trace(
            fig,
            x=profile.time_s,
            mean=profile.replicate_mean[rep_idx],
            std=profile.replicate_std[rep_idx],
            row=1,
            color=colors[rep_idx % len(colors)],
            name=f"replicate {rep_idx}",
            legendgroup=f"replicate-{rep_idx}",
            showlegend=True,
            fill_alpha=0.10,
            line_width=1.8,
        )
    for reference in references:
        add_reference_trace(fig, reference=reference, row=1)
    fig.add_vline(x=0.0, line={"color": "black", "dash": "dash", "width": 1}, row=1, col=1)
    fig.update_layout(
        title=f"Delayed movement-bank target-radial velocity by replicate ({profile.bank_kind})",
        width=940,
        height=560,
        margin={"l": 72, "r": 24, "t": 76, "b": 76},
        hovermode="x unified",
        legend={"groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=1, col=1)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)
    path = output_dir / "forward_velocity_profiles_by_replicate_stochastic.html"
    fig.write_html(path)
    return path


add_band_trace = canonical_add_band_trace


add_reference_trace = canonical_add_reference_trace


def build_summary(
    *,
    run: RunInputs,
    profile: VelocityProfile,
    pooled_file: Path,
    replicate_file: Path,
    references: Sequence[Any],
    output_dir: Path,
    direction_split_status: str,
) -> dict[str, Any]:
    """Return JSON-compatible sidecar metadata."""

    peak_idx = int(np.nanargmax(profile.mean))
    return {
        "schema_version": "rlrmp.delayed_movement_bank_velocity_profiles.v1",
        "issue": EXPERIMENT,
        "run_id": run.run_id,
        "run_label": run.label,
        "run_spec": repo_relative(run.run_spec_path),
        "artifact_dir": repo_relative(run.artifact_dir),
        "output_dir": repo_relative(output_dir),
        "bank_kind": profile.bank_kind,
        "checkpoint_policy": "final_checkpoint",
        "checkpoint_source": repo_relative(run.artifact_dir / "trained_model.eqx"),
        "figure": pooled_file.name,
        "replicate_figure": replicate_file.name,
        "projection": "target-radial velocity: dot(effector velocity, unit(target - initial_position))",
        "error_band": ("mean +/- 1 SD over pooled replicate x fixed-bank go-cue/direction trials"),
        "direction_split": {"status": direction_split_status},
        "alignment": profile.alignment,
        "evaluation_bank": profile.evaluation_bank,
        "profile": {
            "n_replicates": int(profile.n_replicates),
            "n_trials_per_replicate": int(profile.n_trials_per_replicate),
            "n_pooled_samples": int(profile.n_pooled_samples),
            "n_time_steps": int(profile.mean.shape[0]),
            "peak_mean_forward_velocity_m_s": float(profile.mean[peak_idx]),
            "time_of_peak_mean_forward_velocity_s": float(profile.time_s[peak_idx]),
            "mean_forward_velocity_min_m_s": float(np.nanmin(profile.mean)),
            "mean_forward_velocity_max_m_s": float(np.nanmax(profile.mean)),
            "time_start_s": float(profile.time_s[0]),
            "time_stop_s": float(profile.time_s[-1]),
            "finite": bool(
                np.isfinite(profile.mean).all()
                and np.isfinite(profile.std).all()
                and np.isfinite(profile.replicate_mean).all()
                and np.isfinite(profile.replicate_std).all()
            ),
        },
        "replicates": [
            replicate_summary(profile, rep_idx) for rep_idx in range(profile.n_replicates)
        ],
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
                "time_of_peak_forward_velocity_s": float(reference.time_of_peak_forward_velocity_s),
                "terminal_position_error_m": float(reference.terminal_position_error_m),
            }
            for reference in references
        },
    }


def replicate_summary(profile: VelocityProfile, rep_idx: int) -> dict[str, float | int]:
    """Return one replicate profile summary."""

    peak_idx = int(np.nanargmax(profile.replicate_mean[rep_idx]))
    return {
        "replicate": int(rep_idx),
        "peak_mean_forward_velocity_m_s": float(profile.replicate_mean[rep_idx, peak_idx]),
        "time_of_peak_mean_forward_velocity_s": float(profile.time_s[peak_idx]),
        "trial_sd_at_peak_m_s": float(profile.replicate_std[rep_idx, peak_idx]),
    }


rgba = hex_to_rgba


def repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    """Return repo-relative path text when possible."""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

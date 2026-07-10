"""Materialize six-row delayed timing / pre-go hold velocity profiles.

This issue-local script evaluates final checkpoints on the corrected
delayed-reach no-catch and catch banks: go cues 10..30, 20 uniform center-out
directions, 0.15 m reach length, and go-cue-aligned target-radial velocity
windows. It is adapted from the one-run delayed_movement_bank materializer under
``results/6c36536/scripts/`` so the delayed fixed-bank semantics stay identical
while the presentation becomes a six-row comparison.
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
from rlrmp.viz.figures import (
    write_velocity_by_replicate_figure as canonical_write_velocity_by_replicate_figure,
    write_velocity_figure as canonical_write_velocity_figure,
)

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
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
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.viz import profile_comparison_grid


canonical_movement_horizon = _eval_ensemble._canonical_movement_horizon
infer_trial_count = _eval_ensemble._infer_trial_count
infer_trial_n_time = _eval_ensemble._infer_trial_n_time
fixed_bank_projection_direction = _eval_ensemble.fixed_bank_projection_direction
like_existing_array = _eval_ensemble._like_existing_array
like_existing_time_input = _eval_ensemble._like_existing_time_input
reach_direction = _eval_ensemble._reach_direction
remap_source_trials = _eval_ensemble._remap_source_trials
target_position_sequence = _eval_ensemble._target_position_sequence
target_radial_projection_direction = _eval_ensemble._target_radial_projection_direction
update_cartesian_position = _eval_ensemble._update_cartesian_position
update_controller_input = _eval_ensemble._update_controller_input
update_delayed_task_inputs = _eval_ensemble._update_delayed_task_inputs
update_initial_positions = _eval_ensemble._update_initial_positions
update_inputs = _eval_ensemble._update_inputs
update_targets = _eval_ensemble._update_targets


RESULT_EXPERIMENT = "40e1911"
TOPIC = "delayed_timing_hold_lane_velocity_profiles"
PRE_GO_CONTEXT_STEPS = 10
REFERENCE_COLOR = "#111827"
BankKind = Literal["no_catch", "catch"]
DEFAULT_RUN_REFS = (
    "6c36536/baseline__delayed_repeat=baseline delayed repeat",
    "bf71d86/timing__fixed_go10=fixed go 10",
    "bf71d86/timing__fixed_go20=fixed go 20",
    "bf71d86/timing__go10_15=go 10-15",
    "ef9c882/hold__force_filter=force-filter hold",
    "ef9c882/hold__start_pos_zero_vel=start-pos zero-vel hold",
)


@dataclass(frozen=True)
class RunInputs:
    """Resolved local files for one run."""

    experiment: str
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

    experiment: str
    run_id: str
    label: str
    run_spec_path: Path
    artifact_dir: Path
    bank_kind: BankKind
    sisu_level: float | None
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
    run_refs = tuple(args.run_ref or DEFAULT_RUN_REFS)
    runs = [
        resolve_run_inputs(ref=parse_run_ref(run_ref), repo_root=args.repo_root)
        for run_ref in run_refs
    ]
    output_root = args.output_dir or (
        args.repo_root / "_artifacts" / args.result_experiment / "figures" / args.topic
    )
    tracked_root = args.repo_root / "results" / args.result_experiment / "figures" / args.topic
    notes_path = args.repo_root / "results" / args.result_experiment / "notes" / f"{args.topic}.md"
    mkdir_p(output_root)
    mkdir_p(tracked_root)
    mkdir_p(notes_path.parent)
    summaries: dict[str, Any] = {}
    for bank_kind in ("no_catch", "catch"):
        output_dir = output_root / bank_kind
        mkdir_p(output_dir)
        sisu_levels = tuple(args.sisu_level or (None,))
        profiles = [
            evaluate_velocity_profile(
                run,
                bank_kind=bank_kind,
                go_cue_steps=(
                    run_go_cue_steps(run.run_spec)
                    if args.go_cue_step is None
                    else tuple(args.go_cue_step)
                ),
                direction_count=args.direction_count,
                reach_length_m=args.reach_length_m,
                pre_go_context_steps=args.pre_go_context_steps,
                sisu_level=sisu_level,
            )
            for run in runs
            for sisu_level in sisu_levels
        ]
        include_reference = bool(args.include_reference and bank_kind == "no_catch")
        references = (
            cs_output_feedback_reference_profiles(
                n_samples=max(profile.n_pooled_samples for profile in profiles)
            )
            if include_reference
            else ()
        )
        pooled_file = write_velocity_figure(
            profiles,
            output_dir=output_dir,
            references=references,
        )
        replicate_file = write_velocity_by_replicate_figure(
            profiles,
            output_dir=output_dir,
            references=references,
        )
        summary = build_bank_summary(
            profiles=profiles,
            pooled_file=pooled_file,
            replicate_file=replicate_file,
            references=references,
            output_dir=output_dir,
            issue=args.result_experiment,
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
    manifest = build_manifest(
        runs=runs,
        summaries=summaries,
        output_root=output_root,
        tracked_root=tracked_root,
        args=args,
    )
    manifest_path = tracked_root / "manifest.json"
    spec_path = tracked_root / "spec.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spec_path.write_text(
        json.dumps(build_figure_spec(runs=runs, args=args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_marked_section(
        notes_path,
        "delayed_timing_hold_velocity_profiles",
        render_notes(manifest) + "\n",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-ref",
        action="append",
        help=(
            "Run reference as EXPERIMENT/RUN_ID=label. May be passed multiple times; "
            "defaults to the six completed delayed timing / pre-go hold rows."
        ),
    )
    parser.add_argument("--result-experiment", default=RESULT_EXPERIMENT)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--go-cue-step",
        type=int,
        action="append",
        default=None,
        help=(
            "Override delayed-bank go-cue steps for every row. Defaults to each "
            "run spec's declared delayed_reach.go_cue_sampling range."
        ),
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
    parser.add_argument(
        "--sisu-level",
        type=float,
        action="append",
        default=None,
        help=(
            "Evaluate each run at this SISU level. Repeat to compare levels. "
            "When omitted, preserve each run's validation-bank SISU values."
        ),
    )
    return parser


@dataclass(frozen=True)
class RunRef:
    """Parsed CLI run reference."""

    experiment: str
    run_id: str
    label: str


def parse_run_ref(value: str) -> RunRef:
    """Parse ``EXPERIMENT/RUN_ID=label`` into a run reference."""

    path_part, sep, label = value.partition("=")
    experiment, slash, run_id = path_part.partition("/")
    if not sep or not slash or not experiment or not run_id or not label:
        raise ValueError(f"run ref must be EXPERIMENT/RUN_ID=label; got {value!r}")
    return RunRef(experiment=experiment, run_id=run_id, label=label)


def resolve_run_inputs(
    *,
    ref: RunRef,
    repo_root: Path,
) -> RunInputs:
    """Resolve the flat or legacy directory-form run spec for one run."""

    flat_path = repo_root / "results" / ref.experiment / "runs" / f"{ref.run_id}.json"
    legacy_path = repo_root / "results" / ref.experiment / "runs" / ref.run_id / "run.json"
    if flat_path.exists():
        run_spec_path = flat_path
    elif legacy_path.exists():
        run_spec_path = legacy_path
    else:
        raise FileNotFoundError(f"Missing run spec for {ref.experiment}/{ref.run_id}")
    artifact_dir = repo_root / "_artifacts" / ref.experiment / "runs" / ref.run_id
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")
    return RunInputs(
        experiment=ref.experiment,
        run_id=ref.run_id,
        label=ref.label,
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
    sisu_level: float | None = None,
) -> VelocityProfile:
    """Evaluate via the canonical delayed-profile implementation."""

    return canonical_evaluate_velocity_profile(
        run,
        bank_kind=bank_kind,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        reach_length_m=reach_length_m,
        pre_go_context_steps=pre_go_context_steps,
        sisu_level=sisu_level,
        include_sisu_metadata=True,
    )


def make_delayed_eval_bank(
    trial_specs: TaskTrialSpec,
    *,
    bank_kind: BankKind,
    go_cue_steps: Iterable[int],
    direction_count: int,
    reach_length_m: float,
    movement_horizon_steps: int,
    sisu_level: float | None = None,
) -> DelayedEvalBank:
    """Build via the canonical delayed-bank implementation."""

    return canonical_make_delayed_eval_bank(
        trial_specs,
        bank_kind=bank_kind,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        reach_length_m=reach_length_m,
        movement_horizon_steps=movement_horizon_steps,
        sisu_level=sisu_level,
        include_sisu_metadata=True,
    )


def run_go_cue_steps(run_spec: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the declared delayed go-cue steps for one run spec."""

    sampling = run_spec.get("delayed_reach", {}).get("go_cue_sampling", {})
    min_step = sampling.get("min_step_inclusive")
    max_step = sampling.get("max_step_inclusive")
    if min_step is not None and max_step is not None:
        return tuple(range(int(min_step), int(max_step) + 1))
    return tuple(DEFAULT_DELAYED_REACH_GO_CUE_STEPS)


def write_velocity_figure(
    profiles: Sequence[VelocityProfile],
    *,
    output_dir: Path,
    references: Sequence[Any],
) -> Path:
    grouped = group_sisu_profiles(profiles)
    if grouped is not None:
        return write_sisu_velocity_figure(
            grouped, bank_kind=profiles[0].bank_kind, output_dir=output_dir, references=references
        )
    return canonical_write_velocity_figure(
        profiles, output_dir=output_dir, references=references,
        title="Delayed timing / pre-go hold target-radial velocity ({bank_kind})"
    )


def group_sisu_profiles(
    profiles: Sequence[VelocityProfile],
) -> list[tuple[VelocityProfile, list[VelocityProfile]]] | None:
    """Group explicit-SISU profiles by source run, preserving first-seen order."""

    if not profiles or all(profile.sisu_level is None for profile in profiles):
        return None
    grouped: dict[tuple[str, str], list[VelocityProfile]] = {}
    representatives: dict[tuple[str, str], VelocityProfile] = {}
    for profile in profiles:
        key = (profile.experiment, profile.run_id)
        grouped.setdefault(key, []).append(profile)
        representatives.setdefault(key, profile)
    return [(representatives[key], grouped[key]) for key in grouped]


def write_sisu_velocity_figure(
    grouped_profiles: Sequence[tuple[VelocityProfile, Sequence[VelocityProfile]]],
    *,
    bank_kind: BankKind,
    output_dir: Path,
    references: Sequence[Any],
) -> Path:
    """Write a SISU comparison figure with one panel per trained row."""

    fig = profile_comparison_grid(
        n_panels=len(grouped_profiles),
        subplot_titles=[profile.label for profile, _group in grouped_profiles],
        vertical_spacing=0.025,
    )
    for row, (_representative, group) in enumerate(grouped_profiles, start=1):
        for reference in references:
            add_reference_trace(fig, reference=reference, row=row, showlegend=row == 1)
        for profile in sorted(
            group, key=lambda item: -1.0 if item.sisu_level is None else item.sisu_level
        ):
            label = profile.label if profile.sisu_level is None else f"SISU={profile.sisu_level:g}"
            legendgroup = (
                f"sisu-{profile.sisu_level:g}"
                if profile.sisu_level is not None
                else f"run-{profile.experiment}-{profile.run_id}"
            )
            add_band_trace(
                fig,
                x=profile.time_s,
                mean=profile.mean,
                std=profile.std,
                row=row,
                color=sisu_color(profile.sisu_level),
                name=label,
                legendgroup=legendgroup,
                showlegend=row == 1,
            )
        fig.add_vline(x=0.0, line={"color": "black", "dash": "dash", "width": 1}, row=row, col=1)
    fig.update_layout(
        title=f"Delayed SISU target-radial velocity comparison ({bank_kind})",
        width=980,
        height=max(540, 320 * len(grouped_profiles)),
        margin={"l": 72, "r": 24, "t": 76, "b": 72},
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=len(grouped_profiles), col=1)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)
    fig.update_yaxes(matches="y")
    path = output_dir / "forward_velocity_profiles_stochastic.html"
    fig.write_html(path)
    return path


def sisu_color(sisu_level: float | None) -> str:
    """Return a stable color for SISU traces."""

    if sisu_level is None:
        return "#2563eb"
    if float(sisu_level) <= 0.0:
        return "#2563eb"
    if float(sisu_level) >= 1.0:
        return "#dc2626"
    return "#7c3aed"


def write_velocity_by_replicate_figure(
    profiles: Sequence[VelocityProfile],
    *,
    output_dir: Path,
    references: Sequence[Any],
) -> Path:
    return canonical_write_velocity_by_replicate_figure(
        profiles, output_dir=output_dir, references=references,
        title="Delayed timing / pre-go hold target-radial velocity by replicate ({bank_kind})"
    )


add_band_trace = canonical_add_band_trace


add_reference_trace = canonical_add_reference_trace


def build_bank_summary(
    *,
    profiles: Sequence[VelocityProfile],
    pooled_file: Path,
    replicate_file: Path,
    references: Sequence[Any],
    output_dir: Path,
    issue: str,
    direction_split_status: str,
) -> dict[str, Any]:
    """Return JSON-compatible sidecar metadata."""

    return {
        "schema_version": "rlrmp.delayed_timing_hold_velocity_profiles.v1",
        "issue": issue,
        "output_dir": repo_relative(output_dir),
        "bank_kind": profiles[0].bank_kind if profiles else None,
        "checkpoint_policy": "final_checkpoint",
        "figure": pooled_file.name,
        "replicate_figure": replicate_file.name,
        "projection": (
            "target-radial velocity: dot(effector velocity, "
            "unit(intended_visible_target - initial_position))"
        ),
        "error_band": ("mean +/- 1 SD over pooled replicate x fixed-bank go-cue/direction trials"),
        "reference_overlay_contract": (
            "No-catch panels include existing output-feedback extLQG analytical "
            "references when requested. Output-feedback H-infinity traces are not "
            "materialized for delayed fixed-bank/catch views because the current "
            "delayed GRU certificate contract is blocked by the 6D delayed "
            "feedback/force-filter GraphSpec versus 8D analytical reference "
            "response-map mismatch."
        ),
        "direction_split": {"status": direction_split_status},
        "profiles": [profile_summary(profile) for profile in profiles],
        "references": {reference.label: reference_summary(reference) for reference in references},
    }


def profile_summary(profile: VelocityProfile) -> dict[str, Any]:
    """Return JSON-compatible summary for one run profile."""

    peak_idx = int(np.nanargmax(profile.mean))
    return {
        "experiment": profile.experiment,
        "run_id": profile.run_id,
        "run_label": profile.label,
        "sisu_level": profile.sisu_level,
        "run_spec": repo_relative(profile.run_spec_path),
        "artifact_dir": repo_relative(profile.artifact_dir),
        "checkpoint_source": repo_relative(profile.artifact_dir / "trained_model.eqx"),
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
    }


def reference_summary(reference: Any) -> dict[str, Any]:
    """Return JSON-compatible summary for one analytical reference."""

    return {
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


def build_manifest(
    *,
    runs: Sequence[RunInputs],
    summaries: Mapping[str, Any],
    output_root: Path,
    tracked_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return the tracked cross-lane figure manifest."""

    return {
        "schema_version": "rlrmp.delayed_timing_hold_velocity_manifest.v1",
        "issue": args.result_experiment,
        "topic": args.topic,
        "source": "results/40e1911/scripts/materialize_delayed_timing_hold_lane_velocity_profiles.py",
        "lineage": (
            "Adapted from results/6c36536/scripts/"
            "materialize_delayed_movement_bank_velocity_profiles.py"
        ),
        "output_root": repo_relative(output_root),
        "tracked_root": repo_relative(tracked_root),
        "checkpoint_policy": "final_checkpoint",
        "bank_kinds": sorted(summaries.keys()),
        "run_refs": [
            {
                "experiment": run.experiment,
                "run_id": run.run_id,
                "label": run.label,
                "run_spec": repo_relative(run.run_spec_path),
                "artifact_dir": repo_relative(run.artifact_dir),
            }
            for run in runs
        ],
        "figures": {
            bank_kind: {
                "aggregate": f"{bank_kind}/forward_velocity_profiles_stochastic.html",
                "by_replicate": (
                    f"{bank_kind}/forward_velocity_profiles_by_replicate_stochastic.html"
                ),
                "summary": f"{bank_kind}/velocity_profile_summary.json",
            }
            for bank_kind in summaries
        },
        "summaries": dict(summaries),
    }


def build_figure_spec(*, runs: Sequence[RunInputs], args: argparse.Namespace) -> dict[str, Any]:
    """Return a tracked regeneration spec for the cross-lane figures."""

    return {
        "schema_version": "rlrmp.figure_spec.delayed_timing_hold_velocity.v1",
        "figure_family": "delayed_fixed_bank_target_radial_velocity_profiles",
        "materializer": (
            "results/40e1911/scripts/materialize_delayed_timing_hold_lane_velocity_profiles.py"
        ),
        "result_experiment": args.result_experiment,
        "topic": args.topic,
        "run_refs": [
            {
                "experiment": run.experiment,
                "run_id": run.run_id,
                "label": run.label,
            }
            for run in runs
        ],
        "parameters": {
            "go_cue_steps": (
                "per_run_declared_range"
                if args.go_cue_step is None
                else [int(step) for step in args.go_cue_step]
            ),
            "sisu_levels": (
                "preserve_validation_bank_values"
                if args.sisu_level is None
                else [float(level) for level in args.sisu_level]
            ),
            "direction_count": int(args.direction_count),
            "reach_length_m": float(args.reach_length_m),
            "pre_go_context_steps": int(args.pre_go_context_steps),
            "include_reference": bool(args.include_reference),
            "reference_overlay_contract": (
                "no_catch overlays use existing output-feedback extLQG reference "
                "profiles. Output-feedback H-infinity overlay is not materialized "
                "for delayed fixed-bank/catch views because the current delayed "
                "GRU standard-certificate contract is blocked by a 6D delayed "
                "feedback/force-filter GraphSpec versus 8D analytical reference "
                "response-map mismatch."
            ),
            "checkpoint_policy": "final_checkpoint",
            "shared_yaxes": "all",
        },
        "outputs": {
            "catch": f"_artifacts/{args.result_experiment}/figures/{args.topic}/catch",
            "no_catch": f"_artifacts/{args.result_experiment}/figures/{args.topic}/no_catch",
        },
    }


def render_notes(manifest: Mapping[str, Any]) -> str:
    """Render a compact Markdown note for the generated figures."""

    lines = [
        "## Delayed timing / pre-go hold velocity profiles",
        "",
        (
            "Generated "
            f"{len(manifest.get('run_refs', []))}-row fixed delayed-bank "
            "target-radial velocity profiles."
        ),
        "",
        (
            "Reference overlay contract: no-catch panels include the existing "
            "output-feedback extLQG analytical references when requested. "
            "Output-feedback H-infinity traces are not overlaid for delayed "
            "fixed-bank/catch views because the current delayed certificate "
            "contract is blocked by the 6D delayed feedback/force-filter versus "
            "8D analytical response-map mismatch."
        ),
        "",
        "| Bank | Aggregate HTML | By-replicate HTML |",
        "|---|---|---|",
    ]
    figures = manifest.get("figures", {})
    output_root = str(manifest.get("output_root", ""))
    for bank_kind in ("no_catch", "catch"):
        row = figures.get(bank_kind, {})
        lines.append(
            f"| `{bank_kind}` | `{output_root}/{row.get('aggregate')}` | "
            f"`{output_root}/{row.get('by_replicate')}` |"
        )
    lines.extend(
        [
            "",
            "Rows:",
        ]
    )
    for run in manifest.get("run_refs", []):
        lines.append(f"- `{run['experiment']}/{run['run_id']}` - {run['label']}")
    return "\n".join(lines)


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

"""Materialize nominal velocity profiles for the a378b34 distillation run."""

from __future__ import annotations
from rlrmp.viz.traces import add_band_trace as canonical_add_band_trace

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.random as jr
import numpy as np
import plotly.graph_objects as go

import rlrmp.analysis  # noqa: F401 - registers analysis/task surfaces used by setup.
from rlrmp.io import update_marked_section
from rlrmp.train.distillation_native.closed_loop_kernel import (
    ExtLQGClosedLoopReference,
    _initial_vector,
    _target_position,
    _training_hps_from_spec,
)
from rlrmp.train.task_model import setup_task_model_pair


ISSUE = "a378b34"
RUN_ID = "h0_extlqg_6d_closed_loop_distillation"
TOPIC = "nominal_velocity_profile_comparison"
DEFAULT_RUN_SPEC = Path(f"results/{ISSUE}/runs/{RUN_ID}.json")
DEFAULT_RUN_ARTIFACT_DIR = Path(f"_artifacts/{ISSUE}/runs/{RUN_ID}")
DEFAULT_FIGURE_DIR = Path(f"_artifacts/{ISSUE}/figures/{TOPIC}")
DEFAULT_SPEC_DIR = Path(f"results/{ISSUE}/figures/{TOPIC}")
DEFAULT_NOTES = Path(f"results/{ISSUE}/notes/{RUN_ID}.md")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _nominal_trial_count(trial_specs: Any) -> int:
    target_spec = trial_specs.targets.get("mechanics.effector.pos", None)
    if target_spec is not None and hasattr(target_spec, "value"):
        return int(target_spec.value.shape[0])
    init = trial_specs.inits.get("mechanics.vector", None)
    if init is not None:
        return int(init.shape[0])
    raise ValueError("Could not infer nominal trial count from targets or mechanics.vector init.")


def _evaluate_ensemble(task: Any, model: Any, trial_specs: Any, *, key: Any, n_replicates: int):
    """Evaluate a replicate-batched Feedbax graph on trials without requiring intervenors."""

    n_trials = _nominal_trial_count(trial_specs)

    def _is_batched_array(x: Any) -> bool:
        return eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_replicates

    model_arrays, model_other = eqx.partition(model, _is_batched_array)

    def eval_one_replicate(model_array_leaves: Any, model_static: Any, rep_key: Any):
        rep_model = eqx.combine(model_array_leaves, model_static)
        keys = jr.split(rep_key, n_trials)
        return task.eval_trials(rep_model, trial_specs, keys)

    return eqx.filter_vmap(eval_one_replicate, in_axes=(0, None, 0))(
        model_arrays,
        model_other,
        jr.split(key, n_replicates),
    )


def _trial_directions(trial_specs: Any, states: Any) -> np.ndarray:
    initial = np.asarray(_initial_vector(trial_specs, states))
    target = np.asarray(_target_position(trial_specs, states))
    if initial.ndim == 3:
        initial = initial[0]
    if target.ndim == 3:
        target = target[0]
    initial = initial[:, 0:2]
    direction = target - initial
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    return direction / np.maximum(norm, 1e-12)


def _project_forward_velocity(velocity: np.ndarray, directions: np.ndarray) -> np.ndarray:
    if velocity.ndim == 4:
        return np.einsum("rnti,ni->rnt", velocity, directions)
    if velocity.ndim == 3:
        return np.einsum("nti,ni->nt", velocity, directions)
    raise ValueError(f"Expected velocity with 3 or 4 dims, got shape {velocity.shape}.")


def _mean_sem(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = samples.reshape((-1, samples.shape[-1]))
    mean = np.nanmean(flat, axis=0)
    if flat.shape[0] <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.nanstd(flat, axis=0, ddof=1) / np.sqrt(flat.shape[0])
    return mean, sem


def _add_profile(
    fig: go.Figure,
    *,
    time_s: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Add one mean profile and SEM band."""
    canonical_add_band_trace(
        fig,
        x=time_s,
        mean=mean,
        spread=sem,
        color=color,
        name=label,
        band_fill_color=color.replace("1)", "0.16)"),
        band_label=f"{label} SEM",
        line_width=2.4,
        row=None,
        col=None,
    )


def _write_profile_csv(path: Path, rows: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows)
    n = len(rows[keys[0]])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([float(rows[key][i]) for key in keys])


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    spec = _read_json(args.run_spec)
    n_replicates = int(spec["student_contract"]["n_replicates"])
    hps = _training_hps_from_spec(spec)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(spec.get("seed", 0))))
    model_template = pair.model
    model_path = args.run_artifact_dir / "trained_model.eqx"
    trained_model = eqx.tree_deserialise_leaves(model_path, model_template)

    trial_specs = pair.task.validation_trials
    states = _evaluate_ensemble(
        pair.task,
        trained_model,
        trial_specs,
        key=jr.PRNGKey(int(args.eval_seed)),
        n_replicates=n_replicates,
    )
    reference = ExtLQGClosedLoopReference.from_package(
        spec["teacher_contract"]["teacher_package"],
        teacher_gains_key=spec["teacher_contract"]["teacher_gains_key"],
    )
    teacher = reference.rollout(
        initial_vector=_initial_vector(trial_specs, states),
        target_pos=_target_position(trial_specs, states),
        n_steps=int(states.mechanics.effector.vel.shape[-2]),
    )

    directions = _trial_directions(trial_specs, states)
    student_forward = _project_forward_velocity(
        np.asarray(states.mechanics.effector.vel), directions
    )
    teacher_forward = _project_forward_velocity(np.asarray(teacher["velocity"]), directions)
    student_mean, student_sem = _mean_sem(student_forward)
    teacher_mean, teacher_sem = _mean_sem(teacher_forward[0])
    dt = float(getattr(pair.task, "dt", 0.01))
    time_s = np.arange(student_mean.shape[0], dtype=float) * dt

    fig = go.Figure()
    _add_profile(
        fig,
        time_s=time_s,
        mean=teacher_mean,
        sem=teacher_sem,
        label="6D extLQG",
        color="rgba(32, 92, 142, 1)",
    )
    _add_profile(
        fig,
        time_s=time_s,
        mean=student_mean,
        sem=student_sem,
        label="Distilled h0 GRU",
        color="rgba(188, 82, 44, 1)",
    )
    fig.update_layout(
        title="Nominal target-radial velocity: distilled h0 GRU vs 6D extLQG",
        width=980,
        height=560,
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 72, "r": 30, "t": 78, "b": 70},
        legend={"orientation": "h", "x": 0.02, "y": 1.06},
    )
    fig.update_xaxes(title_text="Time from trial start (s)")
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.figure_dir / "nominal_forward_velocity_profiles.html"
    png_path = args.figure_dir / "nominal_forward_velocity_profiles.png"
    csv_path = args.figure_dir / "nominal_forward_velocity_profiles.csv"
    summary_path = args.figure_dir / "nominal_forward_velocity_profiles_summary.json"
    fig.write_html(html_path)
    png_error = None
    try:
        fig.write_image(png_path, scale=2)
    except Exception as exc:  # pragma: no cover - depends on optional kaleido binary.
        png_error = f"{type(exc).__name__}: {exc}"
    _write_profile_csv(
        csv_path,
        {
            "time_s": time_s,
            "extlqg_mean_m_s": teacher_mean,
            "extlqg_sem_m_s": teacher_sem,
            "distilled_mean_m_s": student_mean,
            "distilled_sem_m_s": student_sem,
            "distilled_minus_extlqg_mean_m_s": student_mean - teacher_mean,
        },
    )

    error = student_forward - teacher_forward
    rmse_by_sample = np.sqrt(np.nanmean(np.square(error), axis=-1))
    summary = {
        "schema_version": "rlrmp.a378b34.nominal_velocity_profile_comparison.v1",
        "issue": ISSUE,
        "run_id": spec["run_id"],
        "source_run_spec": str(args.run_spec),
        "trained_model": str(model_path),
        "teacher_package": spec["teacher_contract"]["teacher_package"],
        "n_replicates": n_replicates,
        "n_nominal_trials": _nominal_trial_count(trial_specs),
        "n_time_steps": int(student_mean.shape[0]),
        "dt_s": dt,
        "band": "SEM across replicate-trial profiles for student; SEM across trial profiles for teacher",
        "profile": "target_radial_velocity",
        "student_peak_mean_m_s": float(np.nanmax(student_mean)),
        "student_peak_time_s": float(time_s[int(np.nanargmax(student_mean))]),
        "extlqg_peak_mean_m_s": float(np.nanmax(teacher_mean)),
        "extlqg_peak_time_s": float(time_s[int(np.nanargmax(teacher_mean))]),
        "mean_profile_rmse_m_s": float(np.sqrt(np.nanmean(np.square(student_mean - teacher_mean)))),
        "sample_profile_rmse_mean_m_s": float(np.nanmean(rmse_by_sample)),
        "sample_profile_rmse_sem_m_s": float(
            np.nanstd(rmse_by_sample, ddof=1) / np.sqrt(rmse_by_sample.size)
        ),
        "html": str(html_path),
        "png": str(png_path) if png_error is None else None,
        "png_error": png_error,
        "csv": str(csv_path),
    }
    _write_json(summary_path, summary)

    spec_payload = {
        "schema_version": "rlrmp.figure_spec.v1",
        "issue": ISSUE,
        "topic": TOPIC,
        "run_id": spec["run_id"],
        "figure_family": "nominal_target_radial_velocity_profiles",
        "tracked_outputs": {
            "spec": str(args.figure_spec_dir / "spec.json"),
            "note": str(args.notes_path),
        },
        "bulk_outputs": {
            "html": str(html_path),
            "png": str(png_path) if png_error is None else None,
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
        "source_script": "results/a378b34/scripts/materialize_nominal_velocity_profile_comparison.py",
        "comparison": {
            "student": "distilled standard h0 GRU closed-loop rollout",
            "reference": "6D extLQG analytical closed-loop rollout",
            "trial_set": "nominal validation trials",
            "band": summary["band"],
        },
    }
    _write_json(args.figure_spec_dir / "spec.json", spec_payload)

    note = "\n".join(
        [
            "## Nominal Velocity Profile",
            "",
            f"- Figure: `{html_path}`",
            f"- Profile CSV: `{csv_path}`",
            f"- Summary: `{summary_path}`",
            f"- Trials: `{summary['n_nominal_trials']}` nominal validation trials, "
            f"`{n_replicates}` student replicates.",
            "- Bands: SEM across replicate-trial profiles for the distilled model; "
            "SEM across nominal trial profiles for the extLQG reference.",
            f"- Mean-profile RMSE: `{summary['mean_profile_rmse_m_s']:.6g} m/s`.",
            f"- Student peak mean velocity: `{summary['student_peak_mean_m_s']:.6g} m/s` "
            f"at `{summary['student_peak_time_s']:.6g} s`.",
            f"- extLQG peak mean velocity: `{summary['extlqg_peak_mean_m_s']:.6g} m/s` "
            f"at `{summary['extlqg_peak_time_s']:.6g} s`.",
            "",
        ]
    )
    if png_error is None:
        note = note.replace(
            f"- Profile CSV: `{csv_path}`",
            f"- Static PNG: `{png_path}`\n- Profile CSV: `{csv_path}`",
        )
    else:
        note = note.replace(
            f"- Profile CSV: `{csv_path}`",
            "- Static PNG: not emitted because the local Kaleido executable is unavailable.\n"
            f"- Profile CSV: `{csv_path}`",
        )
    update_marked_section(args.notes_path, TOPIC, note)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, default=DEFAULT_RUN_SPEC)
    parser.add_argument("--run-artifact-dir", type=Path, default=DEFAULT_RUN_ARTIFACT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--figure-spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--notes-path", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--eval-seed", type=int, default=1001)
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = materialize(_build_parser().parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Materialize post-run analysis for issue 1ab1fef."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from jax_cookbook import load_with_hyperparameters

import rlrmp.analysis  # noqa: F401 - registers analysis/task surfaces used by setup.
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    initial_effector_velocity,
    repeat_single_validation_trial,
    trial_effector_target_position,
)
from rlrmp.io import update_marked_section
from rlrmp.train.cs_nominal_gru import build_hps, build_parser, resolve_run_spec_args
from rlrmp.train.task_model import setup_task_model_pair


jax.config.update("jax_enable_x64", True)

ISSUE = "1ab1fef"
RUN_ID = "epsilon_scaled_short_3500to1000"
COMPARISON_ISSUE = "91a090c"
N_ROLLOUT_TRIALS = 64
EVAL_SEED = 0
CHECKOUT_ROOT = Path(__file__).resolve().parents[3]

RUN_SPEC = CHECKOUT_ROOT / "results" / ISSUE / "runs" / f"{RUN_ID}.json"
RUN_ARTIFACT_DIR = CHECKOUT_ROOT / "_artifacts" / ISSUE / "runs" / RUN_ID
COMPARISON_VELOCITY_SUMMARY = (
    CHECKOUT_ROOT / "_artifacts" / COMPARISON_ISSUE / "figures" / "nominal_velocity_profiles" / "summary.json"
)
COMPARISON_VELOCITY_CSV = (
    CHECKOUT_ROOT / "_artifacts" / COMPARISON_ISSUE / "figures" / "nominal_velocity_profiles" / "profiles.csv"
)
COMPARISON_DAMAGE_SUMMARY = (
    CHECKOUT_ROOT / "_artifacts" / COMPARISON_ISSUE / "figures" / "adaptive_damage_lambda" / "summary.json"
)

NOTES_PATH = CHECKOUT_ROOT / "results" / ISSUE / "notes" / "post_run_analysis.md"
VELOCITY_TOPIC = "nominal_velocity_profiles"
DAMAGE_TOPIC = "adaptive_damage_lambda"
VELOCITY_TRACKED_DIR = CHECKOUT_ROOT / "results" / ISSUE / "figures" / VELOCITY_TOPIC
DAMAGE_TRACKED_DIR = CHECKOUT_ROOT / "results" / ISSUE / "figures" / DAMAGE_TOPIC
VELOCITY_BULK_DIR = CHECKOUT_ROOT / "_artifacts" / ISSUE / "figures" / VELOCITY_TOPIC
DAMAGE_BULK_DIR = CHECKOUT_ROOT / "_artifacts" / ISSUE / "figures" / DAMAGE_TOPIC


@dataclass(frozen=True)
class Profile:
    """One forward-velocity profile."""

    row: str
    trace: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    source: str


def _repo_ref(path: Path) -> str:
    absolute = path if path.is_absolute() else CHECKOUT_ROOT / path
    return str(absolute.relative_to(CHECKOUT_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _symlink_to_bulk(tracked_dir: Path, bulk_html: Path) -> Path:
    tracked_dir.mkdir(parents=True, exist_ok=True)
    link_path = tracked_dir / "figure.html"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(os.path.relpath(bulk_html, start=tracked_dir))
    return link_path


def _mean_std(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = samples.reshape((-1, samples.shape[-1]))
    return np.mean(flat, axis=0), np.std(flat, axis=0)


def _load_run_args() -> argparse.Namespace:
    parser = build_parser()
    values = vars(parser.parse_args([])).copy()
    values["run_spec"] = str(RUN_SPEC)
    return resolve_run_spec_args(argparse.Namespace(**values), parser=parser)


def evaluate_new_velocity_profile() -> tuple[Profile, dict[str, Any]]:
    """Evaluate the epsilon-scaled model on repeated nominal validation trials."""

    run_spec = _read_json(RUN_SPEC)
    args = _load_run_args()
    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(args.seed)))
    model, _hyperparameters = load_with_hyperparameters(
        RUN_ARTIFACT_DIR / "trained_model.eqx",
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )

    n_replicates = int(hps.model.n_replicates)
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, N_ROLLOUT_TRIALS)
    initial_velocity = initial_effector_velocity(trial_specs)
    target_position = trial_effector_target_position(trial_specs)

    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates,
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> tuple[Any, Any]:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, N_ROLLOUT_TRIALS),
        )
        velocity = jnp.concatenate(
            [initial_velocity[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )
        return velocity, states.mechanics.effector.pos

    velocity, position = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(EVAL_SEED), n_replicates),
    )
    velocity_np = np.asarray(velocity, dtype=np.float64)
    position_np = np.asarray(position, dtype=np.float64)
    target_np = np.asarray(target_position, dtype=np.float64)
    forward = velocity_np[..., 0]
    mean, std = _mean_std(forward)
    dt = float(run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    time_s = np.arange(mean.shape[0], dtype=np.float64) * dt
    endpoint_error = np.linalg.norm(position_np[:, :, -1, :] - target_np[None, :, -1, :], axis=-1)
    peak_idx = int(np.argmax(mean))

    profile = Profile(
        row=RUN_ID,
        trace="checkpoint 16500",
        time_s=time_s,
        mean=mean,
        std=std,
        source=_repo_ref(RUN_ARTIFACT_DIR / "trained_model.eqx"),
    )
    summary = {
        "row": RUN_ID,
        "trace": "checkpoint 16500",
        "checkpoint": _repo_ref(RUN_ARTIFACT_DIR / "checkpoints" / "checkpoint_0016500"),
        "n_samples": int(n_replicates * N_ROLLOUT_TRIALS),
        "n_replicates": n_replicates,
        "n_rollout_trials_per_replicate": N_ROLLOUT_TRIALS,
        "peak_mean_forward_velocity_m_s": float(mean[peak_idx]),
        "time_of_peak_mean_forward_velocity_s": float(time_s[peak_idx]),
        "mean_terminal_position_error_m": float(np.mean(endpoint_error)),
        "endpoint_error_spread_m": float(np.std(endpoint_error)),
        "run_spec_path": _repo_ref(RUN_SPEC),
        "artifact_dir": _repo_ref(RUN_ARTIFACT_DIR),
        "evaluation_seed": EVAL_SEED,
    }
    return profile, summary


def load_comparison_profiles() -> list[Profile]:
    """Load selected 91a090c profile traces plus analytical comparators."""

    wanted = {
        ("short_3500to1000", "6D analytical extLQG nominal"),
        ("short_3500to1000", "6D output-feedback H-infinity nominal"),
        ("short_3500to1000", "checkpoint 15500"),
        ("medium_3500to1000", "checkpoint 17000"),
        ("medium_3500to1000", "checkpoint 19000"),
    }
    grouped: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    with COMPARISON_VELOCITY_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (row["row"], row["trace"])
            if key in wanted:
                grouped.setdefault(key, []).append(
                    (
                        float(row["time_s"]),
                        float(row["mean_forward_velocity_m_s"]),
                        float(row["std_forward_velocity_m_s"]),
                    )
                )
    profiles: list[Profile] = []
    for row_name, trace in sorted(grouped):
        rows = sorted(grouped[(row_name, trace)], key=lambda item: item[0])
        profiles.append(
            Profile(
                row=row_name,
                trace=trace,
                time_s=np.asarray([item[0] for item in rows], dtype=np.float64),
                mean=np.asarray([item[1] for item in rows], dtype=np.float64),
                std=np.asarray([item[2] for item in rows], dtype=np.float64),
                source=_repo_ref(COMPARISON_VELOCITY_CSV),
            )
        )
    return profiles


def write_velocity_outputs(new_profile: Profile, new_summary: dict[str, Any]) -> dict[str, Any]:
    """Write velocity figure, spec, CSV, and summary."""

    comparison_summary = _read_json(COMPARISON_VELOCITY_SUMMARY)
    comparison_profiles = load_comparison_profiles()
    profiles = comparison_profiles + [new_profile]
    VELOCITY_BULK_DIR.mkdir(parents=True, exist_ok=True)
    VELOCITY_TRACKED_DIR.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    colors = {
        "6D analytical extLQG nominal": "#334155",
        "6D output-feedback H-infinity nominal": "#f97316",
        "checkpoint 15500": "#2563eb",
        "checkpoint 17000": "#059669",
        "checkpoint 19000": "#10b981",
        "checkpoint 16500": "#dc2626",
    }
    dashes = {
        "6D analytical extLQG nominal": "dash",
        "6D output-feedback H-infinity nominal": "dot",
        "checkpoint 15500": "solid",
        "checkpoint 17000": "solid",
        "checkpoint 19000": "dashdot",
        "checkpoint 16500": "solid",
    }
    for profile in profiles:
        label = f"{profile.row}: {profile.trace}"
        color = colors.get(profile.trace, "#111827")
        fig.add_trace(
            go.Scatter(
                x=profile.time_s,
                y=profile.mean,
                mode="lines",
                name=label,
                line={"color": color, "width": 2.4, "dash": dashes.get(profile.trace, "solid")},
            )
        )
    fig.update_layout(
        title="Nominal forward velocity: epsilon-scaled short vs adaptive rows",
        template="plotly_white",
        width=1040,
        height=600,
        hovermode="x unified",
        margin={"l": 70, "r": 30, "t": 80, "b": 70},
        legend={"orientation": "h", "x": 0.0, "y": -0.18},
    )
    fig.update_xaxes(title_text="Time from trial start (s)")
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True)

    html_path = VELOCITY_BULK_DIR / "figure.html"
    csv_path = VELOCITY_BULK_DIR / "profiles.csv"
    summary_path = VELOCITY_BULK_DIR / "summary.json"
    fig.write_html(html_path)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["row", "trace", "time_s", "mean_forward_velocity_m_s", "std_forward_velocity_m_s"]
        )
        for profile in profiles:
            for time_s, mean, std in zip(profile.time_s, profile.mean, profile.std, strict=True):
                writer.writerow([profile.row, profile.trace, time_s, mean, std])

    tracked_link = _symlink_to_bulk(VELOCITY_TRACKED_DIR, html_path)
    peak_table = list(comparison_summary["peak_velocity_table"]) + [new_summary]
    summary = {
        "schema_version": "rlrmp.1ab1fef.nominal_velocity_profiles.v1",
        "issue": ISSUE,
        "topic": VELOCITY_TOPIC,
        "run_id": RUN_ID,
        "materializer": _repo_ref(Path(__file__)),
        "new_profile": new_summary,
        "comparison_source": {
            "issue": COMPARISON_ISSUE,
            "summary": _repo_ref(COMPARISON_VELOCITY_SUMMARY),
            "profiles_csv": _repo_ref(COMPARISON_VELOCITY_CSV),
            "analytical_comparator_policy": (
                "Reused 6D analytical extLQG and output-feedback H-infinity profiles "
                "from the 91a090c materialization because they are independent of this run."
            ),
        },
        "comparison_trace_policy": {
            "plotted_91a090c_traces": [
                "short_3500to1000 checkpoint 15500",
                "medium_3500to1000 checkpoint 17000",
                "medium_3500to1000 checkpoint 19000",
            ],
            "new_trace": "epsilon_scaled_short_3500to1000 checkpoint 16500",
        },
        "peak_velocity_table": peak_table,
        "outputs": {
            "html": _repo_ref(html_path),
            "csv": _repo_ref(csv_path),
            "summary": _repo_ref(summary_path),
            "spec": _repo_ref(VELOCITY_TRACKED_DIR / "spec.json"),
            "spec_symlink": _repo_ref(tracked_link),
        },
        "limitations": [
            "The 91a090c short row overran its intended 15500-batch stop; the primary plotted short comparison is checkpoint 15500.",
            "The 91a090c medium row was manually stopped at 19000; checkpoint 17000 is the near-intended comparison and checkpoint 19000 is a later sidecar.",
            "The analytical comparator profiles are reused from 91a090c and are not regenerated here.",
        ],
    }
    spec = {
        "schema_version": "rlrmp.figure_spec.v1",
        "figure_kind": "nominal_forward_velocity_overlay",
        "issue": ISSUE,
        "topic": VELOCITY_TOPIC,
        "inputs": [
            _repo_ref(RUN_SPEC),
            _repo_ref(RUN_ARTIFACT_DIR / "trained_model.eqx"),
            _repo_ref(COMPARISON_VELOCITY_SUMMARY),
            _repo_ref(COMPARISON_VELOCITY_CSV),
        ],
        "transform": [
            {
                "name": "evaluate_epsilon_scaled_final_checkpoint",
                "kwargs": {"n_rollout_trials": N_ROLLOUT_TRIALS, "seed": EVAL_SEED},
            },
            {"name": "reuse_91a090c_comparison_profiles", "kwargs": {}},
        ],
        "outputs": summary["outputs"],
    }
    _write_json(summary_path, summary)
    _write_json(VELOCITY_TRACKED_DIR / "spec.json", spec)
    return summary


def _series_stats(values: np.ndarray, x: np.ndarray, *, near_x: float) -> dict[str, Any]:
    if values.ndim == 2:
        series = np.nanmean(values, axis=1)
        replica_count = int(values.shape[1])
    else:
        series = values
        replica_count = None
    nearest = int(np.nanargmin(np.abs(x - near_x)))
    out: dict[str, Any] = {
        "min": float(np.nanmin(series)),
        "max": float(np.nanmax(series)),
        "range": float(np.nanmax(series) - np.nanmin(series)),
        "final_record": float(series[-1]),
        "near_intended_endpoint": float(series[nearest]),
    }
    if replica_count is not None:
        out["replica_count"] = replica_count
    return out


def summarize_new_damage() -> dict[str, Any]:
    diagnostics_path = RUN_ARTIFACT_DIR / "training_diagnostics.npz"
    diagnostics_json_path = RUN_ARTIFACT_DIR / "training_diagnostics.json"
    training_summary = _read_json(RUN_ARTIFACT_DIR / "training_summary.json")
    run_spec = _read_json(RUN_SPEC)
    with np.load(diagnostics_path) as data:
        x = np.asarray(data["adaptive_epsilon_global_batch"], dtype=np.float64)
        target = np.asarray(data["adaptive_epsilon_target_damage"], dtype=np.float64)
        lam = np.asarray(data["adaptive_epsilon_lambda_value"], dtype=np.float64)
        epsilon_scale = np.asarray(data["adaptive_epsilon_epsilon_scale_used"], dtype=np.float64)
        full_damage = np.asarray(
            data["adaptive_epsilon_training_batch_full_strength_damage_raw"],
            dtype=np.float64,
        )
        applied_damage = np.asarray(
            data["adaptive_epsilon_training_batch_applied_scaled_damage_raw"],
            dtype=np.float64,
        )
        damage_ema = np.asarray(data["adaptive_epsilon_damage_ema"], dtype=np.float64)
        cap_free = np.asarray(data["adaptive_epsilon_inner_cap_free_soft_energy"], dtype=bool)
        projection = np.asarray(data["adaptive_epsilon_inner_projection_active"], dtype=bool)
        safety_cap = np.asarray(data["adaptive_epsilon_inner_safety_cap_enabled"], dtype=bool)
    intended_final = float(run_spec["n_train_batches"])
    return {
        "label": "epsilon scaled short 3500 to 1000",
        "run_spec_path": _repo_ref(RUN_SPEC),
        "diagnostics_path": _repo_ref(diagnostics_path),
        "diagnostics_json_path": _repo_ref(diagnostics_json_path),
        "completed_batches": int(training_summary["completed_batches"]),
        "first_global_batch": float(x[0]),
        "last_global_batch": float(x[-1]),
        "intended_final_global_batch": int(intended_final),
        "nearest_intended_global_batch": float(x[int(np.nanargmin(np.abs(x - intended_final)))]),
        "record_count": int(x.shape[0]),
        "x_source": "adaptive_epsilon_global_batch",
        "damage_schedule": run_spec["hps"]["adaptive_epsilon_curriculum"]["damage_schedule"],
        "target_damage": _series_stats(target, x, near_x=intended_final),
        "adaptive_lambda": _series_stats(lam, x, near_x=intended_final),
        "epsilon_scale": _series_stats(epsilon_scale, x, near_x=intended_final),
        "full_strength_damage_mean": _series_stats(full_damage, x, near_x=intended_final),
        "applied_scaled_damage_mean": _series_stats(applied_damage, x, near_x=intended_final),
        "damage_ema": _series_stats(damage_ema, x, near_x=intended_final),
        "cap_free_soft_energy_all_records": bool(np.all(cap_free)),
        "projection_active_any_record": bool(np.any(projection)),
        "safety_cap_enabled_any_record": bool(np.any(safety_cap)),
        "controller_training_mode": run_spec["hps"]["adaptive_epsilon_curriculum"][
            "controller_training_mode"
        ],
        "status_note": "completed the intended 16500-batch stop with a 1000-batch target hold.",
    }


def write_damage_outputs(new_damage: dict[str, Any]) -> dict[str, Any]:
    comparison_damage = _read_json(COMPARISON_DAMAGE_SUMMARY)
    DAMAGE_BULK_DIR.mkdir(parents=True, exist_ok=True)
    DAMAGE_TRACKED_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(RUN_ARTIFACT_DIR / "training_diagnostics.npz") as data:
        new_x = np.asarray(data["adaptive_epsilon_global_batch"], dtype=np.float64)
        new_series = {
            "target_damage": np.asarray(data["adaptive_epsilon_target_damage"], dtype=np.float64),
            "lambda": np.asarray(data["adaptive_epsilon_lambda_value"], dtype=np.float64),
            "epsilon_scale": np.nanmean(
                np.asarray(data["adaptive_epsilon_epsilon_scale_used"], dtype=np.float64),
                axis=1,
            ),
            "full_strength_damage": np.nanmean(
                np.asarray(
                    data["adaptive_epsilon_training_batch_full_strength_damage_raw"],
                    dtype=np.float64,
                ),
                axis=1,
            ),
            "applied_scaled_damage": np.nanmean(
                np.asarray(
                    data["adaptive_epsilon_training_batch_applied_scaled_damage_raw"],
                    dtype=np.float64,
                ),
                axis=1,
            ),
        }

    csv_path = DAMAGE_BULK_DIR / "series.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "series", "global_batch", "value"])
        for name, values in new_series.items():
            for x_value, y_value in zip(new_x, values, strict=True):
                writer.writerow([RUN_ID, name, x_value, y_value])

    fig = go.Figure()
    damage_colors = {
        "target_damage": "#64748b",
        "full_strength_damage": "#dc2626",
        "applied_scaled_damage": "#2563eb",
    }
    for name in ["target_damage", "full_strength_damage", "applied_scaled_damage"]:
        fig.add_trace(
            go.Scatter(
                x=new_x,
                y=new_series[name],
                mode="lines",
                name=name.replace("_", " "),
                line={"color": damage_colors[name], "width": 2.4},
                yaxis="y",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=new_x,
            y=new_series["lambda"],
            mode="lines",
            name="lambda",
            line={"color": "#7c3aed", "width": 2.1},
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=new_x,
            y=new_series["epsilon_scale"],
            mode="lines",
            name="epsilon scale",
            line={"color": "#111827", "width": 2.1, "dash": "dot"},
            yaxis="y3",
        )
    )
    fig.update_layout(
        title="Epsilon-scaled curriculum diagnostics",
        template="plotly_white",
        width=1040,
        height=620,
        hovermode="x unified",
        margin={"l": 74, "r": 86, "t": 80, "b": 70},
        xaxis={"title": "Global batch"},
        yaxis={"title": "Damage / target"},
        yaxis2={
            "title": "lambda",
            "overlaying": "y",
            "side": "right",
            "type": "log",
            "showgrid": False,
        },
        yaxis3={
            "title": "epsilon scale",
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "position": 0.94,
            "range": [0.0, 1.05],
            "showgrid": False,
        },
        legend={"orientation": "h", "x": 0.0, "y": -0.18},
    )
    html_path = DAMAGE_BULK_DIR / "figure.html"
    summary_path = DAMAGE_BULK_DIR / "summary.json"
    fig.write_html(html_path)
    tracked_link = _symlink_to_bulk(DAMAGE_TRACKED_DIR, html_path)

    rows = dict(comparison_damage["rows"])
    rows[RUN_ID] = new_damage
    summary = {
        "schema_version": "rlrmp.1ab1fef.adaptive_damage_lambda.v1",
        "issue": ISSUE,
        "topic": DAMAGE_TOPIC,
        "run_id": RUN_ID,
        "materializer": _repo_ref(Path(__file__)),
        "rows": rows,
        "comparison_source": {
            "issue": COMPARISON_ISSUE,
            "summary": _repo_ref(COMPARISON_DAMAGE_SUMMARY),
        },
        "outputs": {
            "html": _repo_ref(html_path),
            "csv": _repo_ref(csv_path),
            "summary": _repo_ref(summary_path),
            "spec": _repo_ref(DAMAGE_TRACKED_DIR / "spec.json"),
            "spec_symlink": _repo_ref(tracked_link),
        },
        "limitations": [
            "The plot renders the new epsilon-scaled row only; 91a090c row summaries are included for table comparison.",
            "Applied-scaled damage is the controller-training exposure; full-strength damage is the diagnostic threat at scale 1.",
        ],
    }
    spec = {
        "schema_version": "rlrmp.figure_spec.v1",
        "figure_kind": "adaptive_damage_lambda_epsilon_scale",
        "issue": ISSUE,
        "topic": DAMAGE_TOPIC,
        "inputs": [
            _repo_ref(RUN_SPEC),
            _repo_ref(RUN_ARTIFACT_DIR / "training_diagnostics.npz"),
            _repo_ref(COMPARISON_DAMAGE_SUMMARY),
        ],
        "transform": [
            {"name": "summarize_epsilon_scaled_training_diagnostics", "kwargs": {}},
            {"name": "reuse_91a090c_damage_lambda_summary", "kwargs": {}},
        ],
        "outputs": summary["outputs"],
    }
    _write_json(summary_path, summary)
    _write_json(DAMAGE_TRACKED_DIR / "spec.json", spec)
    return summary


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def write_note(velocity_summary: dict[str, Any], damage_summary: dict[str, Any]) -> None:
    peak_rows = [
        row
        for row in velocity_summary["peak_velocity_table"]
        if (
            (row["row"] == "short_3500to1000" and row["trace"] in {"checkpoint 15500"})
            or (row["row"] == "medium_3500to1000" and row["trace"] in {"checkpoint 17000", "checkpoint 19000"})
            or row["row"] == RUN_ID
            or row["trace"] in {"6D analytical extLQG nominal", "6D output-feedback H-infinity nominal"}
        )
    ]
    # Keep one copy of each analytical comparator.
    seen: set[tuple[str, str]] = set()
    filtered_peak_rows = []
    for row in peak_rows:
        key = (row["row"] if "checkpoint" in row["trace"] else "analytical", row["trace"])
        if key in seen:
            continue
        seen.add(key)
        filtered_peak_rows.append(row)

    lines = [
        "# Post-run analysis: epsilon-scaled short row",
        "",
        "This note compares `epsilon_scaled_short_3500to1000` against [issue:91a090c] "
        "`short_3500to1000` and `medium_3500to1000`, with the 6D analytical extLQG "
        "and output-feedback H-infinity nominal comparators reused from the 91a090c "
        "velocity-profile materialization.",
        "",
        "## Headline",
        "",
    ]
    new_peak = velocity_summary["new_profile"]["peak_mean_forward_velocity_m_s"]
    hinf_peak = next(
        row["peak_mean_forward_velocity_m_s"]
        for row in velocity_summary["peak_velocity_table"]
        if row["trace"] == "6D output-feedback H-infinity nominal"
    )
    short_peak = next(
        row["peak_mean_forward_velocity_m_s"]
        for row in velocity_summary["peak_velocity_table"]
        if row["row"] == "short_3500to1000" and row["trace"] == "checkpoint 15500"
    )
    medium_peak = next(
        row["peak_mean_forward_velocity_m_s"]
        for row in velocity_summary["peak_velocity_table"]
        if row["row"] == "medium_3500to1000" and row["trace"] == "checkpoint 17000"
    )
    new_damage = damage_summary["rows"][RUN_ID]
    lines.extend(
        [
            "- The epsilon-scaled row reaches a nominal peak forward velocity of "
            f"`{new_peak:.6f} m/s` at "
            f"`{velocity_summary['new_profile']['time_of_peak_mean_forward_velocity_s']:.2f} s`, "
            f"nearly matching the 91a090c short intended checkpoint (`{short_peak:.6f}`) "
            f"and medium near-intended checkpoint (`{medium_peak:.6f}`).",
            "- Its nominal peak is also close to the reused output-feedback H-infinity "
            f"comparator (`{hinf_peak:.6f} m/s`), while remaining above the reused "
            "extLQG nominal comparator.",
            "- Damage control at the end of the run is near, but above, the 1000 target: "
            f"full-strength mean damage `{new_damage['full_strength_damage_mean']['final_record']:.3f}`, "
            f"applied-scaled mean damage `{new_damage['applied_scaled_damage_mean']['final_record']:.3f}`, "
            f"and EMA `{new_damage['damage_ema']['final_record']:.3f}`.",
            "- The controller-training exposure was epsilon-scaled: applied damage starts near zero "
            "when the scale is zero, then converges to the full-strength diagnostic once "
            "epsilon scale reaches one.",
            "",
            "## Nominal Velocity And Quality",
            "",
            "| Row / trace | Peak velocity (m/s) | Peak time (s) | Mean terminal error (m) | Endpoint spread (m) | Samples |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in filtered_peak_rows:
        lines.append(
            "| "
            f"{row['row']} / {row['trace']} | "
            f"{_fmt(row.get('peak_mean_forward_velocity_m_s'), 6)} | "
            f"{_fmt(row.get('time_of_peak_mean_forward_velocity_s'), 4)} | "
            f"{_fmt(row.get('mean_terminal_position_error_m'), 6)} | "
            f"{_fmt(row.get('endpoint_error_spread_m'), 6)} | "
            f"{row.get('n_samples', 'NA')} |"
        )

    rows = damage_summary["rows"]
    lines.extend(
        [
            "",
            "## Damage, Lambda, And Epsilon Scale",
            "",
            "| Row | Completed batches | Target near endpoint | Damage mean near endpoint | Damage mean final | Lambda final | Epsilon scale final | Note |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row_name in ["short_3500to1000", "medium_3500to1000", RUN_ID]:
        row = rows[row_name]
        damage_key = "full_strength_damage_mean" if row_name == RUN_ID else "damage_mean"
        epsilon_final = row.get("epsilon_scale", {}).get("final_record")
        lines.append(
            "| "
            f"{row_name} | {row['completed_batches']} | "
            f"{_fmt(row['target_damage']['near_intended_endpoint'], 6)} | "
            f"{_fmt(row[damage_key]['near_intended_endpoint'], 6)} | "
            f"{_fmt(row[damage_key]['final_record'], 6)} | "
            f"{_fmt(row['adaptive_lambda']['final_record'], 6)} | "
            f"{_fmt(epsilon_final, 4)} | "
            f"{row.get('status_note', '')} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Velocity figure spec: `{velocity_summary['outputs']['spec']}`",
            f"- Velocity render: `{velocity_summary['outputs']['html']}`",
            f"- Velocity summary: `{velocity_summary['outputs']['summary']}`",
            f"- Damage/lambda figure spec: `{damage_summary['outputs']['spec']}`",
            f"- Damage/lambda render: `{damage_summary['outputs']['html']}`",
            f"- Damage/lambda summary: `{damage_summary['outputs']['summary']}`",
            "",
            "## Caveats",
            "",
            "- The 91a090c short row overran its intended stop; this comparison uses checkpoint "
            "15500 as the primary short-row endpoint.",
            "- The 91a090c medium row was manually stopped at 19000; checkpoint 17000 is the "
            "near-intended comparator and checkpoint 19000 is retained as a later sidecar.",
            "- This analysis does not claim a GRU standard-certificate pass. The analytical "
            "curves are nominal behavioral comparators only.",
            "",
        ]
    )
    update_marked_section(NOTES_PATH, "post_run_analysis", "\n".join(lines) + "\n")


def main() -> None:
    if not RUN_SPEC.exists():
        raise FileNotFoundError(RUN_SPEC)
    if not (RUN_ARTIFACT_DIR / "trained_model.eqx").exists():
        raise FileNotFoundError(RUN_ARTIFACT_DIR / "trained_model.eqx")
    if not COMPARISON_VELOCITY_SUMMARY.exists():
        raise FileNotFoundError(COMPARISON_VELOCITY_SUMMARY)
    if not COMPARISON_DAMAGE_SUMMARY.exists():
        raise FileNotFoundError(COMPARISON_DAMAGE_SUMMARY)

    new_profile, new_velocity_summary = evaluate_new_velocity_profile()
    velocity_summary = write_velocity_outputs(new_profile, new_velocity_summary)
    new_damage = summarize_new_damage()
    damage_summary = write_damage_outputs(new_damage)
    write_note(velocity_summary, damage_summary)
    print(
        json.dumps(
            {
                "velocity_summary": velocity_summary["outputs"]["summary"],
                "damage_summary": damage_summary["outputs"]["summary"],
                "notes": _repo_ref(NOTES_PATH),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

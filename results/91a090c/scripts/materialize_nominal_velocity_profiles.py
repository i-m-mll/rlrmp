"""Materialize nominal velocity profiles for 91a090c wave-1 checkpoints."""

from __future__ import annotations
from rlrmp.viz.figures import materialize_analytical_profiles
from rlrmp.eval.kinematics import initial_effector_velocity
from rlrmp.io import json_ready
from rlrmp.paths import portable_repo_path

# ruff: noqa: E402

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
import rlrmp
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.plot import save_figure

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    repeat_single_validation_trial,
)
from rlrmp.io import update_marked_section
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.viz import profile_comparison_grid


ISSUE_ID = "91a090c"
TOPIC = "nominal_velocity_profiles"
N_ROLLOUT_TRIALS = DEFAULT_N_ROLLOUT_TRIALS
ANALYTICAL_SEED = 376023
GRU_ROLLOUT_SEED = 0
REPO_ROOT = Path(__file__).resolve().parents[3]
NOTES_PATH = REPO_ROOT / "results" / ISSUE_ID / "notes" / f"{TOPIC}.md"


@dataclass(frozen=True)
class CheckpointSpec:
    """One requested checkpoint trace."""

    row: str
    checkpoint: str
    label: str
    color: str

    @property
    def run_spec_path(self) -> Path:
        """Return the tracked run spec path."""

        return REPO_ROOT / "results" / ISSUE_ID / "runs" / f"{self.row}.json"

    @property
    def checkpoint_dir(self) -> Path:
        """Return the checkpoint artifact directory."""

        return (
            REPO_ROOT
            / "_artifacts"
            / ISSUE_ID
            / "runs"
            / self.row
            / "checkpoints"
            / self.checkpoint
        )


@dataclass(frozen=True)
class Profile:
    """One plotted nominal forward-velocity profile."""

    row: str
    label: str
    kind: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    source: str
    line_color: str
    line_dash: str = "solid"
    checkpoint: str | None = None
    run_spec: str | None = None
    terminal_position_error_m: float | None = None
    endpoint_spread_m: float | None = None
    parity_status: str | None = None

    @property
    def peak_forward_velocity_m_s(self) -> float:
        """Return peak mean forward velocity."""

        return float(np.nanmax(self.mean))

    @property
    def time_of_peak_forward_velocity_s(self) -> float:
        """Return time of peak mean forward velocity."""

        return float(self.time_s[int(np.nanargmax(self.mean))])


CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        row="short_3500to1000",
        checkpoint="checkpoint_0013000",
        label="checkpoint 13000",
        color="#2563eb",
    ),
    CheckpointSpec(
        row="short_3500to1000",
        checkpoint="checkpoint_0015500",
        label="checkpoint 15500",
        color="#16a34a",
    ),
    CheckpointSpec(
        row="short_3500to1000",
        checkpoint="checkpoint_0017500",
        label="checkpoint 17500",
        color="#f97316",
    ),
    CheckpointSpec(
        row="medium_3500to1000",
        checkpoint="checkpoint_0013500",
        label="checkpoint 13500",
        color="#2563eb",
    ),
    CheckpointSpec(
        row="medium_3500to1000",
        checkpoint="checkpoint_0017000",
        label="checkpoint 17000",
        color="#16a34a",
    ),
    CheckpointSpec(
        row="medium_3500to1000",
        checkpoint="checkpoint_0019000",
        label="checkpoint 19000",
        color="#f97316",
    ),
)

CHECKPOINT_LEGEND: dict[str, tuple[str, str]] = {
    "checkpoint 13000": (
        "checkpoint_peak_target",
        "peak-target checkpoint (13000 / 13500)",
    ),
    "checkpoint 13500": (
        "checkpoint_peak_target",
        "peak-target checkpoint (13000 / 13500)",
    ),
    "checkpoint 15500": (
        "checkpoint_end_anneal",
        "near anneal-end checkpoint (15500 / 17000)",
    ),
    "checkpoint 17000": (
        "checkpoint_end_anneal",
        "near anneal-end checkpoint (15500 / 17000)",
    ),
    "checkpoint 17500": (
        "checkpoint_later_hold",
        "later hold checkpoint (17500 / 19000)",
    ),
    "checkpoint 19000": (
        "checkpoint_later_hold",
        "later hold checkpoint (17500 / 19000)",
    ),
}


def main() -> None:
    """Write the figure, sidecars, and notes."""

    require_current_worktree_import()
    checkpoint_profiles = tuple(
        evaluate_gru_checkpoint_profile(spec) for spec in CHECKPOINTS
    )
    analytical_profiles = materialize_analytical_profiles(
        n_samples=max(profile.n_samples for profile in checkpoint_profiles)
    )
    profiles_by_row = {
        row: (*analytical_profiles_for_row(row, analytical_profiles), *row_checkpoint_profiles)
        for row, row_checkpoint_profiles in group_profiles_by_row(checkpoint_profiles).items()
    }
    fig = make_plotly_figure(profiles_by_row)
    summary = build_summary(profiles_by_row)
    spec = build_figure_spec(summary)
    saved = save_figure(
        fig=fig,
        spec=spec,
        package="rlrmp",
        experiment=ISSUE_ID,
        topic=TOPIC,
        extra_packages=["rlrmp"],
    )
    render_dir = require_path(saved["render_path"]).parent
    csv_path = write_profile_csv(profiles_by_row, render_dir=render_dir)
    summary_path = render_dir / "summary.json"
    summary["outputs"] = {
        "html": repo_ref(require_path(saved["render_path"])),
        "csv": repo_ref(csv_path),
        "summary": repo_ref(summary_path),
        "spec": repo_ref(require_path(saved["spec_path"])),
        "spec_symlink": (
            repo_ref(require_path(saved["symlink_path"])) if saved.get("symlink_path") else None
        ),
        "notes": repo_ref(NOTES_PATH),
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n")
    write_notes(summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


def evaluate_gru_checkpoint_profile(spec: CheckpointSpec) -> Profile:
    """Evaluate one GRU checkpoint on the fixed nominal validation reach."""

    run_spec = json.loads(spec.run_spec_path.read_text(encoding="utf-8"))
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model = eqx.tree_deserialise_leaves(spec.checkpoint_dir / "model.eqx", pair.model)
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, N_ROLLOUT_TRIALS)
    initial_velocity = initial_effector_velocity(trial_specs)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, N_ROLLOUT_TRIALS),
        )

    states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(GRU_ROLLOUT_SEED), n_replicates),
    )
    velocity = jnp.concatenate(
        [
            jnp.broadcast_to(
                initial_velocity[None, :, None, :],
                (n_replicates, N_ROLLOUT_TRIALS, 1, initial_velocity.shape[-1]),
            ),
            states.mechanics.effector.vel,
        ],
        axis=2,
    )
    forward = np.asarray(velocity[..., 0], dtype=np.float64)
    pooled_forward = forward.reshape(n_replicates * N_ROLLOUT_TRIALS, forward.shape[-1])
    endpoint_error = endpoint_errors_m(trial_specs, states)
    dt = float(run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    return Profile(
        row=spec.row,
        label=spec.label,
        kind="gru_adaptive_curriculum_checkpoint",
        time_s=np.arange(pooled_forward.shape[-1], dtype=np.float64) * dt,
        mean=np.mean(pooled_forward, axis=0),
        std=np.std(pooled_forward, axis=0),
        n_samples=int(pooled_forward.shape[0]),
        source="gru_checkpoint_nominal_rollout",
        checkpoint=repo_ref(spec.checkpoint_dir),
        run_spec=repo_ref(spec.run_spec_path),
        line_color=spec.color,
        terminal_position_error_m=float(np.mean(endpoint_error)),
        endpoint_spread_m=float(np.std(endpoint_error)),
    )




def make_plotly_figure(profiles_by_row: dict[str, tuple[Profile, ...]]) -> go.Figure:
    """Build the requested two-subplot line-only figure."""

    rows = ("short_3500to1000", "medium_3500to1000")
    fig = profile_comparison_grid(
        n_panels=2,
        rows=1,
        cols=2,
        subplot_titles=rows,
        horizontal_spacing=0.08,
    )
    for col, row_name in enumerate(rows, start=1):
        for profile in profiles_by_row[row_name]:
            fig.add_trace(
                go.Scatter(
                    x=profile.time_s,
                    y=profile.mean,
                    mode="lines",
                    line={
                        "color": profile.line_color,
                        "width": 2.6,
                        "dash": profile.line_dash,
                    },
                    name=plot_legend_label(profile),
                    legendgroup=plot_legend_group(profile),
                    showlegend=(col == 1),
                    hovertemplate=(
                        "time=%{x:.3f}s<br>"
                        "mean forward velocity=%{y:.6f} m/s"
                        f"<extra>{profile.label}</extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="Time (s)", range=[0.0, 0.6], row=1, col=col)
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True, row=1, col=1)
    fig.update_layout(
        title="91a090c wave-1 nominal forward velocity profiles",
        template="plotly_white",
        width=1160,
        height=560,
        margin={"l": 72, "r": 28, "t": 86, "b": 64},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "x": 0.0},
    )
    return fig


def plot_legend_group(profile: Profile) -> str:
    """Return the legend group shared by corresponding traces across subplots."""

    if profile.label in CHECKPOINT_LEGEND:
        group, _label = CHECKPOINT_LEGEND[profile.label]
        return group
    return profile.kind


def plot_legend_label(profile: Profile) -> str:
    """Return the legend label shown once for the corresponding subplot pair."""

    if profile.label in CHECKPOINT_LEGEND:
        _group, label = CHECKPOINT_LEGEND[profile.label]
        return label
    return profile.label


def group_profiles_by_row(profiles: tuple[Profile, ...]) -> dict[str, tuple[Profile, ...]]:
    """Group checkpoint profiles by row in requested order."""

    return {
        "short_3500to1000": tuple(
            profile for profile in profiles if profile.row == "short_3500to1000"
        ),
        "medium_3500to1000": tuple(
            profile for profile in profiles if profile.row == "medium_3500to1000"
        ),
    }


def analytical_profiles_for_row(
    row: str,
    analytical_profiles: tuple[Profile, Profile],
) -> tuple[Profile, Profile]:
    """Return analytical profiles stamped with the subplot row name."""

    return tuple(
        Profile(
            row=row,
            label=profile.label,
            kind=profile.kind,
            time_s=profile.time_s,
            mean=profile.mean,
            std=profile.std,
            n_samples=profile.n_samples,
            source=profile.source,
            line_color=profile.line_color,
            line_dash=profile.line_dash,
            checkpoint=profile.checkpoint,
            run_spec=profile.run_spec,
            terminal_position_error_m=profile.terminal_position_error_m,
            endpoint_spread_m=profile.endpoint_spread_m,
            parity_status=profile.parity_status,
        )
        for profile in analytical_profiles
    )


def write_profile_csv(
    profiles_by_row: dict[str, tuple[Profile, ...]],
    *,
    render_dir: Path,
) -> Path:
    """Write a long CSV sidecar for plotted mean and standard deviation arrays."""

    path = render_dir / "profiles.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "row",
                "trace",
                "time_s",
                "mean_forward_velocity_m_s",
                "std_forward_velocity_m_s",
            ],
        )
        writer.writeheader()
        for row_profiles in profiles_by_row.values():
            for profile in row_profiles:
                for time_s, mean, std in zip(
                    profile.time_s,
                    profile.mean,
                    profile.std,
                    strict=True,
                ):
                    writer.writerow(
                        {
                            "row": profile.row,
                            "trace": profile.label,
                            "time_s": f"{time_s:.12g}",
                            "mean_forward_velocity_m_s": f"{mean:.12g}",
                            "std_forward_velocity_m_s": f"{std:.12g}",
                        }
                    )
    return path


def build_summary(profiles_by_row: dict[str, tuple[Profile, ...]]) -> dict[str, Any]:
    """Return the JSON-compatible figure summary without output paths."""

    table = [
        {
            "row": profile.row,
            "trace": profile.label,
            "checkpoint": profile.checkpoint,
            "peak_mean_forward_velocity_m_s": profile.peak_forward_velocity_m_s,
            "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_forward_velocity_s,
            "mean_terminal_position_error_m": profile.terminal_position_error_m,
            "endpoint_error_spread_m": profile.endpoint_spread_m,
            "n_samples": profile.n_samples,
        }
        for row_profiles in profiles_by_row.values()
        for profile in row_profiles
    ]
    return {
        "schema_version": "rlrmp.91a090c.nominal_velocity_profiles.v1",
        "issue": ISSUE_ID,
        "topic": TOPIC,
        "materializer": repo_ref(Path(__file__).resolve()),
        "timestamp": datetime.now(UTC).isoformat(),
        "plot_contract": {
            "quantity": "target-axis forward velocity",
            "condition": "nominal fixed +x validation reach",
            "visual_traces_per_subplot": 5,
            "visual_trace_policy": "line-only; no bands, baseline, 8D, or other analytical traces",
            "gru_rollout_trials_per_replicate": N_ROLLOUT_TRIALS,
            "gru_rollout_seed": GRU_ROLLOUT_SEED,
            "analytical_seed": ANALYTICAL_SEED,
            "grid_helper": "rlrmp.viz.profile_comparison_grid",
            "shared_yaxes": "all",
        },
        "peak_velocity_table": table,
        "profiles": {
            f"{profile.row}/{profile.label}": {
                "kind": profile.kind,
                "source": profile.source,
                "checkpoint": profile.checkpoint,
                "run_spec": profile.run_spec,
                "n_samples": profile.n_samples,
                "peak_mean_forward_velocity_m_s": profile.peak_forward_velocity_m_s,
                "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_forward_velocity_s,
                "mean_terminal_position_error_m": profile.terminal_position_error_m,
                "endpoint_error_spread_m": profile.endpoint_spread_m,
                "parity_status": profile.parity_status,
            }
            for row_profiles in profiles_by_row.values()
            for profile in row_profiles
        },
        "limitations": [
            "The short row overran its intended 15500-batch stop; this figure uses only the requested checkpoints, not checkpoint_0019500.",
            "The medium row was stopped at checkpoint_0019000 and has no training_summary.json; checkpoint traces are evaluated directly.",
            "The analytical profiles are 6D no-integrator output-feedback comparators under common random draws; they are not trained model checkpoints.",
        ],
        "no_training_or_remote_gpu": True,
    }


def build_figure_spec(summary: dict[str, Any]) -> dict[str, Any]:
    """Build the tracked Feedbax figure spec."""

    checkpoint_inputs = [
        {
            "role": "checkpoint_model",
            "row": spec.row,
            "checkpoint": spec.checkpoint,
            "path": repo_ref(spec.checkpoint_dir / "model.eqx"),
        }
        for spec in CHECKPOINTS
    ]
    checkpoint_inputs.extend(
        {
            "role": "checkpoint_metadata",
            "row": spec.row,
            "checkpoint": spec.checkpoint,
            "path": repo_ref(spec.checkpoint_dir / "metadata.json"),
        }
        for spec in CHECKPOINTS
    )
    return {
        "schema_version": "rlrmp.91a090c.figure_spec.v1",
        "issue": ISSUE_ID,
        "topic": TOPIC,
        "materializer": summary["materializer"],
        "inputs": [
            {
                "role": "run_spec",
                "row": "short_3500to1000",
                "path": "results/91a090c/runs/short_3500to1000.json",
            },
            {
                "role": "run_spec",
                "row": "medium_3500to1000",
                "path": "results/91a090c/runs/medium_3500to1000.json",
            },
            *checkpoint_inputs,
        ],
        "analytical_provenance": {
            "analytical_builders": [
                "rlrmp.analysis.math.cs_game_card.build_no_integrator_game",
                "rlrmp.analysis.math.cs_released_simulation",
                "rlrmp.analysis.math.output_feedback",
            ],
            "included": [
                "6D analytical extLQG nominal",
                "6D output-feedback H-infinity nominal",
            ],
            "excluded": ["baseline", "8D analytical traces", "other analytical models"],
        },
        "plot_contract": summary["plot_contract"],
        "peak_velocity_table": summary["peak_velocity_table"],
        "no_training_or_remote_gpu": True,
    }


def write_notes(summary: dict[str, Any]) -> None:
    """Update the tracked Markdown summary note."""

    lines = [
        "## Nominal Velocity Profiles",
        "",
        f"- Figure: `{summary['outputs']['html']}`",
        f"- Figure spec: `{summary['outputs']['spec']}`",
        f"- Profile CSV: `{summary['outputs']['csv']}`",
        f"- Summary JSON: `{summary['outputs']['summary']}`",
        "- Figure contents: two subplots, each with only the 6D extLQG line, "
        "the 6D output-feedback H-infinity line, and the three requested row checkpoints.",
        "",
        "| Row | Trace/checkpoint | Peak mean forward velocity (m/s) | Time to peak (s) | Mean terminal position error (m) | Endpoint spread (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["peak_velocity_table"]:
        checkpoint = item["checkpoint"]
        trace = item["trace"] if checkpoint is None else Path(checkpoint).name
        lines.append(
            "| "
            f"{item['row']} | "
            f"{trace} | "
            f"{item['peak_mean_forward_velocity_m_s']:.6g} | "
            f"{item['time_of_peak_mean_forward_velocity_s']:.6g} | "
            f"{format_optional_float(item['mean_terminal_position_error_m'])} | "
            f"{format_optional_float(item['endpoint_error_spread_m'])} |"
        )
    lines.extend(
        [
            "",
            "Caveats: the short row is summarized at the requested intermediate checkpoints, "
            "not the overrun endpoint; the medium row was user-stopped at checkpoint 19000 "
            "and has no `training_summary.json`, so all checkpoint metrics here come from "
            "direct nominal rollouts.",
            "",
        ]
    )
    update_marked_section(NOTES_PATH, TOPIC, "\n".join(lines))


def endpoint_errors_m(trial_specs: Any, states: Any) -> np.ndarray:
    """Return terminal position error for each replicate and nominal rollout."""

    target = np.asarray(final_goal_position(trial_specs), dtype=np.float64)
    final_pos = np.asarray(states.mechanics.effector.pos[..., -1, :], dtype=np.float64)
    return np.linalg.norm(final_pos - target[None, :, :], axis=-1)


def final_goal_position(trial_specs: Any) -> jnp.ndarray:
    """Return final target position for each repeated nominal trial."""

    target = trial_specs.targets["mechanics.effector.pos"].value
    return target[:, -1, :]




def is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    """Return true when an array's leading axis indexes model replicates."""

    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


repo_ref = portable_repo_path


def require_path(path: Any) -> Path:
    """Return a non-null Path from Feedbax save output."""

    if path is None:
        raise ValueError("Expected Feedbax save_figure to return a path")
    return Path(path)


def require_current_worktree_import() -> None:
    """Fail before writes if imports resolve to another checkout."""

    package_path = Path(rlrmp.__file__).resolve()
    if not package_path.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            "rlrmp is imported from a different checkout. "
            f"Expected under {REPO_ROOT}, got {package_path}. "
            "Run with PYTHONPATH=src uv run --no-sync python ..."
        )


def format_optional_float(value: float | None) -> str:
    """Format optional floats for Markdown tables."""

    if value is None:
        return "n/a"
    return f"{value:.6g}"


json_ready = json_ready


if __name__ == "__main__":
    main()

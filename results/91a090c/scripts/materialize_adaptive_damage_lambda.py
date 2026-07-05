"""Materialize adaptive damage and lambda traces for issue 91a090c wave 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from feedbax.plot import save_figure
from plotly.subplots import make_subplots

from rlrmp.io import update_marked_section
from rlrmp.paths import mkdir_p


ISSUE = "91a090c"
TOPIC = "adaptive_damage_lambda"
DAMAGE_KEY = "adaptive_epsilon_adaptive_update_damage_raw"
DAMAGE_EMA_KEY = "adaptive_epsilon_damage_ema"
LAMBDA_KEY = "adaptive_epsilon_lambda_value"
GLOBAL_BATCH_KEY = "adaptive_epsilon_global_batch"
TARGET_DAMAGE_KEY = "adaptive_epsilon_target_damage"
RUNS = {
    "short_3500to1000": {
        "label": "short 3500 to 1000",
        "color": "#2563eb",
        "intended_final_global_batch": 15500,
        "status_note": "overran beyond intended 15500; data extend to 19499/19500.",
    },
    "medium_3500to1000": {
        "label": "medium 3500 to 1000",
        "color": "#0f766e",
        "intended_final_global_batch": 17250,
        "status_note": "continued past intended 17250 and was stopped at 19000.",
    },
}


REPO_ROOT = Path.cwd().resolve()


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "range": float(np.nanmax(finite) - np.nanmin(finite)),
    }


def _nearest_index(x: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(x, dtype=np.float64) - target)))


def _mean_trace(values: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr, 1
    return np.nanmean(arr, axis=1), int(arr.shape[1])


def _load_run(run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    diagnostics_path = REPO_ROOT / "_artifacts" / ISSUE / "runs" / run_id / "training_diagnostics.npz"
    diagnostics_json_path = diagnostics_path.with_suffix(".json")
    spec_path = REPO_ROOT / "results" / ISSUE / "runs" / f"{run_id}.json"

    data = np.load(diagnostics_path)
    missing = [
        key
        for key in (DAMAGE_KEY, DAMAGE_EMA_KEY, LAMBDA_KEY, TARGET_DAMAGE_KEY)
        if key not in data.files
    ]
    if missing:
        raise KeyError(f"{diagnostics_path} is missing required diagnostics: {missing}")

    if GLOBAL_BATCH_KEY in data.files:
        x = np.asarray(data[GLOBAL_BATCH_KEY], dtype=np.float64)
        x_source = GLOBAL_BATCH_KEY
    else:
        # Fallback for older sidecars: align the adaptive arrays to their final
        # recorded training batches, preserving global-batch semantics.
        batch_index = np.asarray(data["batch_index"], dtype=np.float64)
        n = np.asarray(data[DAMAGE_KEY]).shape[0]
        x = batch_index[-n:]
        x_source = "batch_index_tail_reconstruction"

    damage_mean, damage_replica_count = _mean_trace(data[DAMAGE_KEY])
    damage_ema, _ema_replica_count = _mean_trace(data[DAMAGE_EMA_KEY])
    lambda_values, _lambda_replica_count = _mean_trace(data[LAMBDA_KEY])
    target_damage, _target_replica_count = _mean_trace(data[TARGET_DAMAGE_KEY])

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    diagnostics_meta = json.loads(diagnostics_json_path.read_text(encoding="utf-8"))
    damage_schedule = spec["hps"]["adaptive_epsilon_curriculum"]["damage_schedule"]
    intended_idx = _nearest_index(x, config["intended_final_global_batch"])
    final_idx = len(x) - 1

    return {
        "run_id": run_id,
        "label": config["label"],
        "color": config["color"],
        "x": x,
        "x_source": x_source,
        "damage_mean": damage_mean,
        "damage_ema": damage_ema,
        "lambda_values": lambda_values,
        "target_damage": target_damage,
        "diagnostics_path": diagnostics_path,
        "diagnostics_json_path": diagnostics_json_path,
        "spec_path": spec_path,
        "completed_batches": int(diagnostics_meta.get("completed_batches", len(data["batch_index"]))),
        "record_count": int(len(x)),
        "damage_replica_count": damage_replica_count,
        "damage_schedule": damage_schedule,
        "n_batches_condition": int(spec["hps"]["n_batches_condition"]),
        "intended_final_global_batch": int(config["intended_final_global_batch"]),
        "intended_idx": intended_idx,
        "final_idx": final_idx,
        "status_note": config["status_note"],
    }


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _run_summary(row: dict[str, Any]) -> dict[str, Any]:
    intended_idx = row["intended_idx"]
    final_idx = row["final_idx"]
    damage = row["damage_mean"]
    lambda_values = row["lambda_values"]
    damage_ema = row["damage_ema"]
    target_damage = row["target_damage"]
    x = row["x"]
    return {
        "label": row["label"],
        "diagnostics_path": _repo_rel(row["diagnostics_path"]),
        "diagnostics_json_path": _repo_rel(row["diagnostics_json_path"]),
        "run_spec_path": _repo_rel(row["spec_path"]),
        "completed_batches": row["completed_batches"],
        "record_count": row["record_count"],
        "x_source": row["x_source"],
        "first_global_batch": float(x[0]),
        "last_global_batch": float(x[-1]),
        "intended_final_global_batch": row["intended_final_global_batch"],
        "nearest_intended_global_batch": float(x[intended_idx]),
        "extends_past_intended_endpoint": bool(x[-1] > row["intended_final_global_batch"]),
        "damage_replica_count": row["damage_replica_count"],
        "damage_mean": {
            **_finite_summary(damage),
            "near_intended_endpoint": float(damage[intended_idx]),
            "final_record": float(damage[final_idx]),
        },
        "damage_ema": {
            **_finite_summary(damage_ema),
            "near_intended_endpoint": float(damage_ema[intended_idx]),
            "final_record": float(damage_ema[final_idx]),
        },
        "target_damage": {
            **_finite_summary(target_damage),
            "near_intended_endpoint": float(target_damage[intended_idx]),
            "final_record": float(target_damage[final_idx]),
        },
        "adaptive_lambda": {
            **_finite_summary(lambda_values),
            "near_intended_endpoint": float(lambda_values[intended_idx]),
            "final_record": float(lambda_values[final_idx]),
        },
        "damage_schedule": row["damage_schedule"],
        "n_batches_condition": row["n_batches_condition"],
        "status_note": row["status_note"],
    }


def _build_figure(rows: list[dict[str, Any]]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for row in rows:
        schedule = row["damage_schedule"]
        schedule_start = float(row["x"][0])
        ramp_end = schedule_start + float(schedule["ramp_batches"])
        anneal_end = ramp_end + float(schedule["anneal_batches"])
        fig.add_trace(
            go.Scatter(
                x=row["x"],
                y=row["damage_mean"],
                mode="lines",
                name=f"{row['label']} damage",
                line={"color": row["color"], "width": 2.2},
                hovertemplate="batch=%{x:.0f}<br>damage=%{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=row["x"],
                y=row["damage_ema"],
                mode="lines",
                name=f"{row['label']} damage EMA",
                line={"color": row["color"], "width": 1.7, "dash": "dashdot"},
                opacity=0.82,
                hovertemplate="batch=%{x:.0f}<br>damage EMA=%{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=row["x"],
                y=row["target_damage"],
                mode="lines",
                name=f"{row['label']} target damage",
                line={"color": row["color"], "width": 1.7, "dash": "dot"},
                opacity=0.7,
                hovertemplate="batch=%{x:.0f}<br>target damage=%{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=row["x"],
                y=row["lambda_values"],
                mode="lines",
                name=f"{row['label']} lambda (post-update)",
                line={"color": row["color"], "width": 2.0, "dash": "dash"},
                hovertemplate=(
                    "batch=%{x:.0f}<br>post-update lambda=%{y:.6g}<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        for x_boundary in (ramp_end, anneal_end):
            fig.add_vline(
                x=x_boundary,
                line={"color": row["color"], "width": 1.0, "dash": "dot"},
                opacity=0.35,
            )

    fig.update_xaxes(title_text="global batch")
    fig.update_yaxes(title_text="damage diagnostic / target", secondary_y=False)
    fig.update_yaxes(title_text="adaptive lambda (post-update)", secondary_y=True, type="log")
    fig.update_layout(
        template="plotly_white",
        title="Adaptive damage and lambda over wave-1 training",
        legend_title_text="series",
        width=1050,
        height=620,
        hovermode="x unified",
        margin={"l": 80, "r": 90, "t": 80, "b": 70},
    )
    return fig


def _format_number(value: float) -> str:
    if abs(value) >= 1e5 or (0 < abs(value) < 1e-3):
        return f"{value:.6g}"
    return f"{value:.3f}"


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "## Adaptive damage and lambda",
        "",
        "Actual damage uses `adaptive_epsilon_adaptive_update_damage_raw`, averaged "
        "across the five replicate columns at each adaptive diagnostics sample. "
        "The plotted target uses `adaptive_epsilon_target_damage`, the smoother "
        "feedback signal uses `adaptive_epsilon_damage_ema`, and adaptive lambda "
        "uses the post-update `adaptive_epsilon_lambda_value`. The figure uses "
        "the recorded `adaptive_epsilon_global_batch` x-axis.",
        "",
        "| row | records | batch span | intended endpoint | damage near intended | damage final | lambda near intended | lambda final | damage range | lambda range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id, row in summary["rows"].items():
        damage = row["damage_mean"]
        lam = row["adaptive_lambda"]
        lines.append(
            "| "
            + " | ".join(
                [
                    run_id,
                    str(row["record_count"]),
                    f"{row['first_global_batch']:.0f}-{row['last_global_batch']:.0f}",
                    f"{row['intended_final_global_batch']}",
                    _format_number(damage["near_intended_endpoint"]),
                    _format_number(damage["final_record"]),
                    _format_number(lam["near_intended_endpoint"]),
                    _format_number(lam["final_record"]),
                    _format_number(damage["range"]),
                    _format_number(lam["range"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Overrun context: short overran beyond intended 15500 and records extend "
            "to global batch 19499 (19500 completed batches); medium continued past "
            "its intended anneal endpoint and was stopped at 19000 completed batches.",
            "",
            "No scientific verdict is inferred here beyond the plotted diagnostics.",
            "",
            f"Figure: `results/{ISSUE}/figures/{TOPIC}/figure.html`",
            f"Summary JSON: `results/{ISSUE}/figures/{TOPIC}/summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [_load_run(run_id, config) for run_id, config in RUNS.items()]
    fig = _build_figure(rows)

    spec = {
        "figure_kind": "adaptive_damage_lambda",
        "issue": ISSUE,
        "inputs": [
            {"path": _repo_rel(row["diagnostics_path"])}
            for row in rows
        ]
        + [{"path": _repo_rel(row["diagnostics_json_path"])} for row in rows]
        + [{"path": _repo_rel(row["spec_path"])} for row in rows],
        "transform": [
            {
                "name": "adaptive_damage_lambda_trace",
                "kwargs": {
                    "damage_key": DAMAGE_KEY,
                    "damage_ema_key": DAMAGE_EMA_KEY,
                    "lambda_key": LAMBDA_KEY,
                    "target_damage_key": TARGET_DAMAGE_KEY,
                    "x_key": GLOBAL_BATCH_KEY,
                    "damage_reduction": "replicate_mean",
                },
            }
        ],
        "plot_kwargs": {
            "secondary_y": True,
            "primary_y": "actual damage, damage EMA, target damage",
            "secondary_y_axis": "adaptive lambda (post-update)",
            "secondary_y_type": "log",
            "rows": list(RUNS),
        },
    }
    saved = save_figure(
        fig=fig,
        spec=spec,
        package="rlrmp",
        experiment=ISSUE,
        topic=TOPIC,
        extra_packages=["rlrmp"],
    )

    tracked_dir = REPO_ROOT / "results" / ISSUE / "figures" / TOPIC
    artifact_dir = REPO_ROOT / "_artifacts" / ISSUE / "figures" / TOPIC
    mkdir_p(tracked_dir)
    mkdir_p(artifact_dir)
    summary = {
        "schema_version": "rlrmp.91a090c.adaptive_damage_lambda.v1",
        "issue": ISSUE,
        "topic": TOPIC,
        "saved": {key: _repo_rel(path) if path is not None else None for key, path in saved.items()},
        "rows": {row["run_id"]: _run_summary(row) for row in rows},
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (tracked_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (artifact_dir / "summary.json").write_text(summary_text, encoding="utf-8")

    notes_path = REPO_ROOT / "results" / ISSUE / "notes" / f"{TOPIC}.md"
    update_marked_section(notes_path, TOPIC, _markdown_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

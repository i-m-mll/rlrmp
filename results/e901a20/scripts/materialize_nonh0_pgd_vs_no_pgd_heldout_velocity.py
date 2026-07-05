"""Materialize non-H0 PGD vs no-PGD held-out velocity profiles."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import feedbax
import numpy as np
import plotly.graph_objects as go

from rlrmp.analysis.pipelines.gru_pilot_figures import cs_output_feedback_reference_profiles
from rlrmp.io import update_marked_section
from rlrmp.paths import figure_artifact_dir, figure_spec_dir, mkdir_p, run_spec_path
from rlrmp.viz import profile_comparison_grid


EXPERIMENT = "e901a20"
TOPIC = "nonh0_pgd_vs_no_pgd_heldout_velocity"
MARKER = TOPIC
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
NONH0_HELPER_PATH = (
    REPO_ROOT / "results/e901a20/scripts/materialize_nonh0_no_pgd_extlqg_velocity.py"
)
RUN_EXPERIMENT = "020a65b"
NO_PGD_RUN_ID = (
    "target_relative_multitarget_fullqrf_warmcos__"
    "proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
PGD_RUN_ID = (
    "target_relative_multitarget_fullqrf_warmcos__"
    "proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64"
)
REFERENCE_REACH_M = 0.15
N_REFERENCE_SAMPLES = 320


def repo_relative(path: Path) -> str:
    """Return a path relative to the repository root."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.absolute().relative_to(REPO_ROOT.absolute()))


def load_nonh0_materializer() -> Any:
    """Load the existing non-H0 no-PGD materializer helper."""

    spec = importlib.util.spec_from_file_location("e901a20_nonh0_materializer", NONH0_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {NONH0_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_profiles(materializer: Any, ref: Any) -> dict[str, Any]:
    """Evaluate seen, held-out, and all validation target profiles for one row."""

    helper = materializer.load_helper()
    return {
        "seen": helper.evaluate_validation_selected_stochastic_values(
            ref,
            first_target_only=False,
            raw_x_velocity=False,
            length_normalize=True,
            target_condition_mask_fn=helper.non_held_out_direction_length_mask,
        ),
        "held_out": helper.evaluate_validation_selected_stochastic_values(
            ref,
            first_target_only=False,
            raw_x_velocity=False,
            length_normalize=True,
            target_condition_mask_fn=helper.held_out_target_mask,
        ),
        "all": helper.evaluate_validation_selected_stochastic_values(
            ref,
            first_target_only=False,
            raw_x_velocity=False,
            length_normalize=True,
        ),
    }


def row_metrics(
    *,
    materializer: Any,
    row_key: str,
    row_label: str,
    split_key: str,
    split_label: str,
    profile: Any,
    reference_8d: np.ndarray,
    reference_4d: np.ndarray,
    seen_mean_band: float | None = None,
) -> dict[str, Any]:
    """Summarize one row/split profile."""

    metrics = materializer.profile_metrics(
        label=split_label,
        profile=profile,
        reference_8d=reference_8d,
        reference_4d=reference_4d,
        seen_mean_band=seen_mean_band,
    )
    return {
        "row": row_key,
        "row_label": row_label,
        **metrics,
    }


def add_profile_trace(
    fig: go.Figure,
    profile: Any,
    *,
    row: int,
    label: str,
    color: str,
    dash: str | None = None,
    showlegend: bool,
) -> None:
    """Add a mean profile and one-SD band to one subplot row."""

    upper = profile.mean + profile.std
    lower = profile.mean - profile.std
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.12),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name=f"{label} +/- 1 SD",
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    line: dict[str, Any] = {"color": color, "width": 2.4}
    if dash is not None:
        line["dash"] = dash
    fig.add_trace(
        go.Scatter(
            x=profile.time_s,
            y=profile.mean,
            mode="lines",
            line=line,
            name=label,
            legendgroup=label,
            showlegend=showlegend,
        ),
        row=row,
        col=1,
    )


def add_reference_trace(
    fig: go.Figure,
    *,
    row: int,
    time_s: np.ndarray,
    mean: np.ndarray,
    label: str,
    color: str,
    showlegend: bool,
) -> None:
    """Add the normalized extLQG reference line to one subplot row."""

    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=mean,
            mode="lines",
            line={"color": color, "width": 2.0, "dash": "dash"},
            name=label,
            legendgroup=label,
            showlegend=showlegend,
        ),
        row=row,
        col=1,
    )


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to Plotly rgba."""

    color = color.lstrip("#")
    return f"rgba({int(color[0:2], 16)},{int(color[2:4], 16)},{int(color[4:6], 16)},{alpha})"


def write_outputs(
    *,
    materializer: Any,
    profiles_by_row: dict[str, dict[str, Any]],
    row_refs: dict[str, Any],
    references: tuple[Any, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write tracked and bulk outputs."""

    figure_dir = mkdir_p(figure_artifact_dir(EXPERIMENT, TOPIC))
    spec_dir = mkdir_p(figure_spec_dir(EXPERIMENT, TOPIC))
    notes_dir = mkdir_p(REPO_ROOT / "results" / EXPERIMENT / "notes")

    ref8, ref4 = references
    ref8_mean = np.asarray(ref8.forward_velocity, dtype=np.float64) / REFERENCE_REACH_M
    ref4_mean = np.asarray(ref4.forward_velocity, dtype=np.float64) / REFERENCE_REACH_M

    split_order = [
        ("seen", "Seen direction + seen length"),
        ("held_out", "Held-out direction/length"),
        ("all", "All validation targets"),
    ]
    fig = profile_comparison_grid(
        len(split_order),
        subplot_titles=[title for _key, title in split_order],
        vertical_spacing=0.065,
    )
    for panel_idx, (split_key, _title) in enumerate(split_order, start=1):
        add_profile_trace(
            fig,
            profiles_by_row["no_pgd"][split_key],
            row=panel_idx,
            label="non-H0 no-PGD GRU",
            color="#2563eb",
            showlegend=panel_idx == 1,
        )
        add_profile_trace(
            fig,
            profiles_by_row["pgd"][split_key],
            row=panel_idx,
            label="non-H0 PGD GRU",
            color="#dc2626",
            dash="solid",
            showlegend=panel_idx == 1,
        )
        add_reference_trace(
            fig,
            row=panel_idx,
            time_s=np.asarray(ref8.time_s, dtype=np.float64),
            mean=ref8_mean,
            label="extLQG 8D / 0.15 m",
            color="#111827",
            showlegend=panel_idx == 1,
        )
    fig.update_layout(
        title="Non-H0 PGD vs no-PGD GRU: seen/held-out validation velocity",
        width=980,
        height=900,
        margin={"l": 76, "r": 24, "t": 86, "b": 76},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.11, "x": 0.0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", zeroline=False, row=3, col=1)
    fig.update_yaxes(title_text="Target-radial velocity / reach length (1/s)", zeroline=True)

    html_path = figure_dir / f"{TOPIC}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    data_path = figure_dir / f"{TOPIC}.npz"
    npz_payload: dict[str, Any] = {
        "extlqg_8d_time_s": ref8.time_s,
        "extlqg_8d_mean_1_s": ref8_mean,
        "extlqg_8d_std_1_s": np.asarray(ref8.forward_velocity_std, dtype=np.float64)
        / REFERENCE_REACH_M,
        "extlqg_4d_time_s": ref4.time_s,
        "extlqg_4d_mean_1_s": ref4_mean,
        "extlqg_4d_std_1_s": np.asarray(ref4.forward_velocity_std, dtype=np.float64)
        / REFERENCE_REACH_M,
    }
    for row_key, split_profiles in profiles_by_row.items():
        for split_key, profile in split_profiles.items():
            npz_payload[f"{row_key}__{split_key}__time_s"] = profile.time_s
            npz_payload[f"{row_key}__{split_key}__mean"] = profile.mean
            npz_payload[f"{row_key}__{split_key}__std"] = profile.std
    np.savez_compressed(data_path, **npz_payload)

    csv_path = figure_dir / f"{TOPIC}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    feedbax_package_file = Path(feedbax.__file__).resolve()
    runtime_provenance = {
        "feedbax_package_file": str(feedbax_package_file),
        "feedbax_runtime": "current Python import path",
        "feedbax_commit": git_short_commit(feedbax_package_file.parents[1]),
        "legacy_feedbax_3add27d7_check": (
            "same convention as the non-H0 no-PGD diagnostic: exact Feedbax 3add27d7 "
            "was not used because that old runtime blocked before evaluating the current "
            "non-H0 graph path"
        ),
        "local_compatibility_patches": [
            "cast non-H0 SimpleStagedNetwork checkpoint floating leaves to float64",
            "set loaded SimpleStagedNetwork dtype/static initial state to float64",
            "cast nominal trial specs to float64",
            "pad zero sensory/delayed perturbation inputs from 4D to 6D for nominal-clean force/filter feedback graph",
        ],
    }
    manifest = {
        "schema_version": f"rlrmp.{EXPERIMENT}.{TOPIC}.v1",
        "figure": repo_relative(html_path),
        "data": repo_relative(data_path),
        "summary_csv": repo_relative(csv_path),
        "evaluation_lens": "validation_selected_stochastic_target_radial_length_normalized",
        "source_runs": {
            row_key: {
                "experiment": ref.experiment,
                "run_id": ref.run_id,
                "label": ref.label,
            }
            for row_key, ref in row_refs.items()
        },
        "reference": {
            "primary": "C&S extLQG/output-feedback 8D divided by 0.15 m",
            "sidecar": "C&S extLQG/output-feedback 4D pos+vel divided by 0.15 m",
            "n_reference_samples": N_REFERENCE_SAMPLES,
        },
        "split_definition": (
            "seen uses validation targets with non-held-out directions and non-held-out "
            "reach lengths; held-out uses held_out_targets_m; all uses all nominal "
            "validation targets"
        ),
        "velocity_definition": (
            "GRU curves use validation-selected checkpoints, 64 stochastic repeats per "
            "target condition per replicate, and target-radial effector velocity divided "
            "by reach length. extLQG references are stochastic forward velocity for the "
            "15 cm released C&S reference divided by 0.15 m."
        ),
        "band_definition": (
            "GRU one SD over pooled replicate x target-condition x stochastic repeats; "
            "extLQG one SD over stochastic reference rollouts"
        ),
        "checkpoint_policy": "validation_selected_per_replicate_sparse_history",
        "runtime_provenance": runtime_provenance,
        "rows": rows,
        "selected_checkpoints": {
            row_key: [
                selection.to_json() for selection in split_profiles["all"].selected_checkpoints
            ]
            for row_key, split_profiles in profiles_by_row.items()
        },
    }
    manifest_path = figure_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figure_link = spec_dir / "figure.html"
    if figure_link.exists() or figure_link.is_symlink():
        figure_link.unlink()
    figure_link.symlink_to(os.path.relpath(html_path, start=figure_link.parent))
    spec = {
        "schema_version": "rlrmp.figure_spec.v1",
        "topic": TOPIC,
        "source_script": repo_relative(Path(__file__)),
        "manifest": repo_relative(manifest_path),
        "figure": repo_relative(html_path),
        "figure_link": repo_relative(figure_link),
        "data": repo_relative(data_path),
        "summary_csv": repo_relative(csv_path),
        "inputs": [
            {
                "run_spec": repo_relative(run_spec_path(ref.experiment, ref.run_id)),
                "artifact_dir": f"_artifacts/{ref.experiment}/runs/{ref.run_id}",
            }
            for ref in row_refs.values()
        ],
        **{
            key: manifest[key]
            for key in (
                "evaluation_lens",
                "source_runs",
                "reference",
                "split_definition",
                "velocity_definition",
                "band_definition",
                "checkpoint_policy",
                "runtime_provenance",
                "rows",
            )
        },
    }
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    note_path = notes_dir / f"{TOPIC}.md"
    update_marked_section(
        note_path,
        MARKER,
        build_note(manifest, html_path, data_path, csv_path, manifest_path),
    )
    manifest["note"] = repo_relative(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_note(
    manifest: dict[str, Any],
    html_path: Path,
    data_path: Path,
    csv_path: Path,
    manifest_path: Path,
) -> str:
    """Build the tracked Markdown note body."""

    rows = manifest["rows"]
    table_lines = [
        "| Row | Split | Peak (1/s) | Late mean (1/s) | Mean band (1/s) | Band ratio | RMSE vs 8D extLQG (1/s) | Shape corr vs 8D |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = row["band_ratio_vs_seen"]
        table_lines.append(
            f"| `{row['row']}` | `{row['split']}` | "
            f"{row['peak_mean_velocity_1_s']:.4f} | "
            f"{row['late_tail_mean_last_10_1_s']:.4f} | "
            f"{row['mean_band_1_s']:.4f} | "
            f"{'--' if ratio is None else f'{ratio:.2f}'} | "
            f"{row['rmse_vs_extlqg_8d_1_s']:.4f} | "
            f"{row['shape_corr_vs_extlqg_8d']:.4f} |"
        )
    pgd_seen = next(row for row in rows if row["row"] == "pgd" and row["split"].startswith("seen"))
    pgd_held = next(row for row in rows if row["row"] == "pgd" and row["split"].startswith("held-out"))
    no_pgd_seen = next(
        row for row in rows if row["row"] == "no_pgd" and row["split"].startswith("seen")
    )
    no_pgd_held = next(
        row for row in rows if row["row"] == "no_pgd" and row["split"].startswith("held-out")
    )
    return "\n".join(
        [
            "## Non-H0 PGD vs no-PGD held-out velocity",
            "",
            "Source rows:",
            f"- no-PGD: `{manifest['source_runs']['no_pgd']['run_id']}`",
            f"- PGD: `{manifest['source_runs']['pgd']['run_id']}`",
            "",
            manifest["velocity_definition"],
            "",
            "Result: the non-H0 PGD row largely removes the held-out direction/length "
            "degradation visible in the paired non-H0 no-PGD row. The no-PGD held-out "
            f"peak drops by {no_pgd_seen['peak_mean_velocity_1_s'] - no_pgd_held['peak_mean_velocity_1_s']:.4f} "
            "1/s and its RMSE vs 8D extLQG rises from "
            f"{no_pgd_seen['rmse_vs_extlqg_8d_1_s']:.4f} to "
            f"{no_pgd_held['rmse_vs_extlqg_8d_1_s']:.4f} 1/s. The PGD held-out peak "
            f"differs from its seen peak by {pgd_seen['peak_mean_velocity_1_s'] - pgd_held['peak_mean_velocity_1_s']:.4f} "
            "1/s, with RMSE moving from "
            f"{pgd_seen['rmse_vs_extlqg_8d_1_s']:.4f} to "
            f"{pgd_held['rmse_vs_extlqg_8d_1_s']:.4f} 1/s.",
            "",
            "Runtime provenance: generated with current Feedbax commit "
            f"`{manifest['runtime_provenance']['feedbax_commit']}` plus the same scoped "
            "non-H0 dtype/shape compatibility patches used by the no-PGD diagnostic.",
            "",
            *table_lines,
            "",
            f"- Figure: `{repo_relative(html_path)}`",
            f"- Data: `{repo_relative(data_path)}`",
            f"- Summary CSV: `{repo_relative(csv_path)}`",
            f"- Manifest: `{repo_relative(manifest_path)}`",
            "",
        ]
    )


def main() -> None:
    """Materialize the non-H0 PGD vs no-PGD held-out diagnostic."""

    materializer = load_nonh0_materializer()
    helper = materializer.load_helper()
    row_refs = {
        "no_pgd": helper.RunRef(RUN_EXPERIMENT, NO_PGD_RUN_ID, "020a65b no-PGD non-H0 GRU", "#2563eb"),
        "pgd": helper.RunRef(RUN_EXPERIMENT, PGD_RUN_ID, "020a65b PGD non-H0 GRU", "#dc2626"),
    }
    profiles_by_row = {
        row_key: evaluate_profiles(materializer, ref) for row_key, ref in row_refs.items()
    }
    references = cs_output_feedback_reference_profiles(n_samples=N_REFERENCE_SAMPLES)
    ref8_mean = np.asarray(references[0].forward_velocity, dtype=np.float64) / REFERENCE_REACH_M
    ref4_mean = np.asarray(references[1].forward_velocity, dtype=np.float64) / REFERENCE_REACH_M

    rows: list[dict[str, Any]] = []
    split_labels = {
        "seen": "seen direction + seen length",
        "held_out": "held-out direction/length",
        "all": "all validation targets",
    }
    for row_key, row_profiles in profiles_by_row.items():
        seen = row_metrics(
            materializer=materializer,
            row_key=row_key,
            row_label=row_refs[row_key].label,
            split_key="seen",
            split_label=split_labels["seen"],
            profile=row_profiles["seen"],
            reference_8d=ref8_mean,
            reference_4d=ref4_mean,
        )
        rows.append(seen)
        for split_key in ("held_out", "all"):
            rows.append(
                row_metrics(
                    materializer=materializer,
                    row_key=row_key,
                    row_label=row_refs[row_key].label,
                    split_key=split_key,
                    split_label=split_labels[split_key],
                    profile=row_profiles[split_key],
                    reference_8d=ref8_mean,
                    reference_4d=ref4_mean,
                    seen_mean_band=seen["mean_band_1_s"],
                )
            )

    manifest = write_outputs(
        materializer=materializer,
        profiles_by_row=profiles_by_row,
        row_refs=row_refs,
        references=references,
        rows=rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def git_short_commit(path: Path) -> str | None:
    """Return the git short commit for ``path`` when available."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()

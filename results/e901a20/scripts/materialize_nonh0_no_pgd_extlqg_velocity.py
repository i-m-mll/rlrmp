"""Materialize non-H0 no-PGD velocity profiles against extLQG."""

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
import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import plotly.graph_objects as go

from rlrmp.analysis.pipelines.gru_pilot_figures import cs_output_feedback_reference_profiles
from rlrmp.io import update_marked_section
from rlrmp.paths import (
    figure_artifact_dir,
    figure_spec_dir,
    mkdir_p,
    run_spec_path,
)


EXPERIMENT = "e901a20"
TOPIC = "nonh0_no_pgd_extlqg_heldout_velocity"
MARKER = TOPIC
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
SOURCE_HELPER = (
    REPO_ROOT / "results/e901a20/scripts/materialize_nominal_velocity_profile_comparison.py"
)
RUN_EXPERIMENT = "020a65b"
RUN_ID = "target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
RUN_LABEL = "020a65b no-PGD non-H0 GRU"
REFERENCE_REACH_M = 0.15
N_REFERENCE_SAMPLES = 320


def repo_relative(path: Path) -> str:
    """Return a path relative to the repository root."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.absolute().relative_to(REPO_ROOT.absolute()))


def load_helper() -> Any:
    """Load the sibling e901a20 materializer as a module."""

    spec = importlib.util.spec_from_file_location("e901a20_velocity_helper", SOURCE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {SOURCE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_checkpoint_model_compatible = load_nonh0_checkpoint_model_compatible(module)
    module.nominalize_trial_specs = nominalize_trial_specs_float64(module.nominalize_trial_specs)
    return module


def load_nonh0_checkpoint_model_compatible(helper: Any) -> Any:
    """Return a checkpoint loader compatible with non-H0 SimpleStagedNetwork rows."""

    def load(path: Path, hps: Any, seed: int) -> Any:
        template = helper.checkpoint_model_template(hps, seed)

        def cast_floating_leaf(leaf: Any) -> Any:
            if eqx.is_array(leaf) and np.issubdtype(leaf.dtype, np.floating):
                return leaf.astype(jnp.float64)
            return leaf

        model = eqx.tree_deserialise_leaves(path, jt.map(cast_floating_leaf, template))
        net = model.nodes["net"]
        if getattr(net, "dtype", None) is not jnp.float64:
            object.__setattr__(net, "dtype", jnp.float64)
        if hasattr(net, "_initial_state"):
            initial_state = jt.map(cast_floating_leaf, net._initial_state)
            object.__setattr__(net, "_initial_state", initial_state)
        return model

    return load


def nominalize_trial_specs_float64(nominalize: Any) -> Any:
    """Return a nominal-trial helper that matches float64 legacy checkpoint state."""

    def wrapper(trial_specs: Any) -> Any:
        specs = nominalize(trial_specs)
        inputs = dict(getattr(specs, "inputs", {}))
        changed = False
        for key in (
            "perturbation_training.delayed_observation",
            "perturbation_training.sensory_feedback",
        ):
            value = inputs.get(key)
            if eqx.is_array(value) and value.shape[-1] == 4:
                pad_width = [(0, 0)] * value.ndim
                pad_width[-1] = (0, 2)
                inputs[key] = jnp.pad(value, tuple(pad_width))
                changed = True
        if changed:
            specs = eqx.tree_at(lambda t: t.inputs, specs, inputs)

        def cast_floating_leaf(leaf: Any) -> Any:
            if eqx.is_array(leaf) and np.issubdtype(leaf.dtype, np.floating):
                return leaf.astype(jnp.float64)
            return leaf

        return jt.map(cast_floating_leaf, specs)

    return wrapper


def profile_metrics(
    *,
    label: str,
    profile: Any,
    reference_8d: np.ndarray,
    reference_4d: np.ndarray,
    seen_mean_band: float | None = None,
) -> dict[str, Any]:
    """Summarize one profile against normalized extLQG reference curves."""

    n_common = min(len(profile.mean), len(reference_8d), len(reference_4d))
    mean = np.asarray(profile.mean[:n_common], dtype=np.float64)
    ref8 = np.asarray(reference_8d[:n_common], dtype=np.float64)
    ref4 = np.asarray(reference_4d[:n_common], dtype=np.float64)
    peak_idx = int(np.nanargmax(mean))
    mean_band = float(np.nanmean(profile.std[:n_common]))
    late_window = mean[-10:] if mean.shape[0] >= 10 else mean
    corr = float(np.corrcoef(mean, ref8)[0, 1])
    return {
        "split": label,
        "n_target_conditions": int(profile.n_target_conditions),
        "n_pooled_profiles": int(profile.n_replicates * profile.n_target_conditions * profile.n_rollout_repeats),
        "target_distance_min_m": float(profile.target_distance_min_m),
        "target_distance_max_m": float(profile.target_distance_max_m),
        "peak_mean_velocity_1_s": float(mean[peak_idx]),
        "time_of_peak_s": float(profile.time_s[peak_idx]),
        "late_tail_mean_last_10_1_s": float(np.nanmean(late_window)),
        "late_tail_min_last_10_1_s": float(np.nanmin(late_window)),
        "mean_band_1_s": mean_band,
        "band_ratio_vs_seen": (
            None if seen_mean_band is None else float(mean_band / seen_mean_band)
        ),
        "rmse_vs_extlqg_8d_1_s": float(np.sqrt(np.nanmean(np.square(mean - ref8)))),
        "rmse_vs_extlqg_4d_1_s": float(np.sqrt(np.nanmean(np.square(mean - ref4)))),
        "shape_corr_vs_extlqg_8d": corr,
    }


def add_profile_trace(fig: go.Figure, profile: Any, *, label: str, color: str) -> None:
    """Add a mean profile plus one-SD band."""

    upper = profile.mean + profile.std
    lower = profile.mean - profile.std
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.13),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name=f"{label} +/- 1 SD",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile.time_s,
            y=profile.mean,
            mode="lines",
            line={"color": color, "width": 2.5},
            name=label,
        )
    )


def add_reference_trace(
    fig: go.Figure,
    *,
    time_s: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    dash: str,
) -> None:
    """Add a normalized extLQG reference trace."""

    upper = mean + std
    lower = mean - std
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_s, time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.08),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name=f"{label} +/- 1 SD",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=mean,
            mode="lines",
            line={"color": color, "width": 2.2, "dash": dash},
            name=label,
        )
    )


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to Plotly rgba."""

    color = color.lstrip("#")
    return f"rgba({int(color[0:2], 16)},{int(color[2:4], 16)},{int(color[4:6], 16)},{alpha})"


def write_outputs(
    *,
    profiles: dict[str, Any],
    references: tuple[Any, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write tracked and bulk outputs."""

    figure_dir = mkdir_p(figure_artifact_dir(EXPERIMENT, TOPIC))
    spec_dir = mkdir_p(figure_spec_dir(EXPERIMENT, TOPIC))
    notes_dir = mkdir_p(REPO_ROOT / "results" / EXPERIMENT / "notes")

    ref8, ref4 = references
    ref8_mean = np.asarray(ref8.forward_velocity, dtype=np.float64) / REFERENCE_REACH_M
    ref8_std = np.asarray(ref8.forward_velocity_std, dtype=np.float64) / REFERENCE_REACH_M
    ref4_mean = np.asarray(ref4.forward_velocity, dtype=np.float64) / REFERENCE_REACH_M
    ref4_std = np.asarray(ref4.forward_velocity_std, dtype=np.float64) / REFERENCE_REACH_M

    fig = go.Figure()
    add_profile_trace(fig, profiles["seen"], label="GRU seen direction + seen length", color="#2563eb")
    add_profile_trace(fig, profiles["held_out"], label="GRU held-out direction/length", color="#dc2626")
    add_profile_trace(fig, profiles["all"], label="GRU all validation targets", color="#64748b")
    add_reference_trace(
        fig,
        time_s=np.asarray(ref8.time_s, dtype=np.float64),
        mean=ref8_mean,
        std=ref8_std,
        label="extLQG 8D output-feedback / 0.15 m",
        color="#111827",
        dash="dash",
    )
    add_reference_trace(
        fig,
        time_s=np.asarray(ref4.time_s, dtype=np.float64),
        mean=ref4_mean,
        std=ref4_std,
        label="extLQG 4D pos+vel / 0.15 m",
        color="#f97316",
        dash="dot",
    )
    fig.update_layout(
        title="Non-H0 no-PGD GRU: seen/held-out validation velocity vs extLQG",
        width=980,
        height=620,
        margin={"l": 76, "r": 24, "t": 76, "b": 76},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.24, "x": 0.0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", zeroline=False)
    fig.update_yaxes(title_text="Target-radial velocity / reach length (1/s)", zeroline=True)

    html_path = figure_dir / f"{TOPIC}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    data_path = figure_dir / f"{TOPIC}.npz"
    np.savez_compressed(
        data_path,
        **{f"{name}__time_s": profile.time_s for name, profile in profiles.items()},
        **{f"{name}__mean": profile.mean for name, profile in profiles.items()},
        **{f"{name}__std": profile.std for name, profile in profiles.items()},
        extlqg_8d_time_s=ref8.time_s,
        extlqg_8d_mean_1_s=ref8_mean,
        extlqg_8d_std_1_s=ref8_std,
        extlqg_4d_time_s=ref4.time_s,
        extlqg_4d_mean_1_s=ref4_mean,
        extlqg_4d_std_1_s=ref4_std,
    )
    csv_path = figure_dir / f"{TOPIC}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    feedbax_commit = os.environ.get("RLRMP_COMPAT_FEEDBAX_COMMIT")
    feedbax_package_file = Path(feedbax.__file__).resolve()
    current_feedbax_commit = git_short_commit(feedbax_package_file.parents[1])
    runtime_provenance = {
        "feedbax_package_file": str(feedbax_package_file),
        "feedbax_runtime": "feedbax git archive" if feedbax_commit else "current Python import path",
        "feedbax_commit": feedbax_commit,
        "current_feedbax_commit": current_feedbax_commit,
        "legacy_feedbax_3add27d7_check": (
            "attempted separately; blocked before evaluation because current RLRMP graph "
            "construction passes dtype to the older Feedbax SimpleStagedNetwork constructor"
        ),
        "local_compatibility_patches": [
            "cast non-H0 SimpleStagedNetwork checkpoint floating leaves to float64",
            "set loaded SimpleStagedNetwork dtype/static initial state to float64",
            "cast nominal trial specs to float64",
            "pad zero sensory/delayed perturbation inputs from 4D to 6D for nominal-clean force/filter feedback graph",
        ],
        "runtime_note": (
            "This non-H0 row was generated with current Feedbax plus scoped compatibility "
            "patches for legacy checkpoint/trial dtypes and nominal zero-input shapes. It "
            "is not labeled as the pre-1e1c94f5 old-compatible runtime used for the H0 "
            "MaskedLinear figures."
        ),
    }
    manifest = {
        "schema_version": f"rlrmp.{EXPERIMENT}.{TOPIC}.v1",
        "figure": repo_relative(html_path),
        "data": repo_relative(data_path),
        "summary_csv": repo_relative(csv_path),
        "evaluation_lens": (
            "validation_selected_stochastic_target_radial_length_normalized"
        ),
        "source_run": {
            "experiment": RUN_EXPERIMENT,
            "run_id": RUN_ID,
            "label": RUN_LABEL,
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
        "selected_checkpoints": [
            selection.to_json() for selection in profiles["all"].selected_checkpoints
        ],
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
                "run_spec": repo_relative(run_spec_path(RUN_EXPERIMENT, RUN_ID)),
                "artifact_dir": f"_artifacts/{RUN_EXPERIMENT}/runs/{RUN_ID}",
            }
        ],
        **{
            key: manifest[key]
            for key in (
                "evaluation_lens",
                "source_run",
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

    table_lines = [
        "| Split | Peak (1/s) | Late mean (1/s) | Mean band (1/s) | Band ratio | RMSE vs 8D extLQG (1/s) | Shape corr vs 8D |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = row["band_ratio_vs_seen"]
        table_lines.append(
            f"| `{row['split']}` | {row['peak_mean_velocity_1_s']:.4f} | "
            f"{row['late_tail_mean_last_10_1_s']:.4f} | {row['mean_band_1_s']:.4f} | "
            f"{'--' if ratio is None else f'{ratio:.2f}'} | "
            f"{row['rmse_vs_extlqg_8d_1_s']:.4f} | "
            f"{row['shape_corr_vs_extlqg_8d']:.4f} |"
        )
    note = "\n".join(
        [
            "## Non-H0 no-PGD held-out velocity vs extLQG",
            "",
            f"Source row: `{RUN_ID}`. This is the non-H0 020a65b no-PGD GRU row.",
            "",
            manifest["velocity_definition"],
            "",
            "The 8D extLQG output-feedback curve is the primary reference; the 4D "
            "pos+vel reference is included as a sidecar because the GRU observes 6D "
            "target-relative force/filter feedback rather than exactly either "
            "analytical observation channel.",
            "",
            "Runtime provenance: generated with "
            f"`{runtime_provenance['feedbax_runtime']}`"
            + (
                f" at compatibility commit `{runtime_provenance['feedbax_commit']}`."
                if runtime_provenance["feedbax_commit"]
                else "."
            ),
            "",
            "Compatibility caveat: this was generated with current Feedbax commit "
            f"`{runtime_provenance['current_feedbax_commit']}` plus scoped dtype/shape "
            "patches for the non-H0 legacy checkpoint. The exact Feedbax `3add27d7` "
            "runtime used for the H0 MaskedLinear old-compatible figures was attempted "
            "separately but blocked before evaluation because the current RLRMP graph "
            "builder passes `dtype` to the older `SimpleStagedNetwork` constructor.",
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
    note_path = notes_dir / f"{TOPIC}.md"
    update_marked_section(note_path, MARKER, note)
    manifest["note"] = repo_relative(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Materialize the non-H0 no-PGD diagnostic."""

    helper = load_helper()
    run_ref = helper.RunRef(RUN_EXPERIMENT, RUN_ID, RUN_LABEL, "#2563eb")
    profiles = {
        "all": helper.evaluate_validation_selected_stochastic_values(
            run_ref,
            first_target_only=False,
            raw_x_velocity=False,
            length_normalize=True,
        ),
        "seen": helper.evaluate_validation_selected_stochastic_values(
            run_ref,
            first_target_only=False,
            raw_x_velocity=False,
            length_normalize=True,
            target_condition_mask_fn=helper.non_held_out_direction_length_mask,
        ),
        "held_out": helper.evaluate_validation_selected_stochastic_values(
            run_ref,
            first_target_only=False,
            raw_x_velocity=False,
            length_normalize=True,
            target_condition_mask_fn=helper.held_out_target_mask,
        ),
    }
    references = cs_output_feedback_reference_profiles(n_samples=N_REFERENCE_SAMPLES)
    ref8_mean = np.asarray(references[0].forward_velocity, dtype=np.float64) / REFERENCE_REACH_M
    ref4_mean = np.asarray(references[1].forward_velocity, dtype=np.float64) / REFERENCE_REACH_M
    seen_row = profile_metrics(
        label="seen direction + seen length",
        profile=profiles["seen"],
        reference_8d=ref8_mean,
        reference_4d=ref4_mean,
    )
    rows = [
        seen_row,
        profile_metrics(
            label="held-out direction/length",
            profile=profiles["held_out"],
            reference_8d=ref8_mean,
            reference_4d=ref4_mean,
            seen_mean_band=seen_row["mean_band_1_s"],
        ),
        profile_metrics(
            label="all validation targets",
            profile=profiles["all"],
            reference_8d=ref8_mean,
            reference_4d=ref4_mean,
            seen_mean_band=seen_row["mean_band_1_s"],
        ),
    ]
    manifest = write_outputs(profiles=profiles, references=references, rows=rows)
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

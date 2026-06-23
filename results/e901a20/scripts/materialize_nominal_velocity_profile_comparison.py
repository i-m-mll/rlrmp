"""Materialize nominal velocity profiles for the e901a20 policy-adversary run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.io import update_marked_section
from rlrmp.paths import (
    REPO_ROOT,
    figure_artifact_dir,
    figure_spec_dir,
    mkdir_p,
    resolve_run_artifact_path,
    run_spec_path,
)
from rlrmp.train.task_model import setup_task_model_pair


EXPERIMENT = "e901a20"
TOPIC = "nominal_velocity_profile_comparison"
NOMINAL_MARKER = "nominal_velocity_profile_comparison"


@dataclass(frozen=True)
class RunRef:
    """One run to include in the nominal velocity comparison."""

    experiment: str
    run_id: str
    label: str
    color: str


@dataclass(frozen=True)
class VelocityProfile:
    """Nominal target-radial velocity profile for one trained run."""

    run: RunRef
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_replicates: int
    n_trials: int
    peak_mean_forward_velocity_m_s: float
    time_of_peak_mean_forward_velocity_s: float


RUNS: tuple[RunRef, ...] = (
    RunRef(
        "020a65b",
        "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64",
        "020a65b no-PGD H0",
        "#64748b",
    ),
    RunRef(
        "020a65b",
        "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64",
        "020a65b PGD H0",
        "#dc2626",
    ),
    RunRef(
        "e901a20",
        "h0_policy_adversary__plain",
        "Policy adversary plain",
        "#2563eb",
    ),
    RunRef(
        "e901a20",
        "h0_policy_adversary__energy",
        "Policy adversary energy",
        "#059669",
    ),
)


def repo_relative(path: Path) -> str:
    """Return a repo-relative path string."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.absolute().relative_to(REPO_ROOT.absolute()))


def load_run_spec(ref: RunRef) -> dict[str, Any]:
    """Load one tracked run spec."""

    path = run_spec_path(ref.experiment, ref.run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing run spec for {ref.run_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_dir(ref: RunRef) -> Path:
    """Return one run artifact directory."""

    path = REPO_ROOT / "_artifacts" / ref.experiment / "runs" / ref.run_id
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact dir for {ref.run_id}: {path}")
    return path


def nominalize_trial_specs(trial_specs: Any) -> Any:
    """Return validation specs with explicit perturbation inputs zeroed."""

    specs = trial_specs
    if PLANT_INTERVENOR_LABEL in specs.intervene:
        scale = specs.intervene[PLANT_INTERVENOR_LABEL].scale
        specs = eqx.tree_at(
            lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
            specs,
            jnp.zeros_like(scale),
        )
    zero_input_keys = tuple(
        key
        for key, value in specs.inputs.items()
        if eqx.is_array(value)
        and (
            key == "epsilon"
            or key.startswith("perturbation_training.")
            or key.endswith("_perturbation")
        )
    )
    if zero_input_keys:
        inputs = dict(specs.inputs)
        for key in zero_input_keys:
            inputs[key] = jnp.zeros_like(inputs[key])
        specs = eqx.tree_at(lambda t: t.inputs, specs, inputs)
    return specs


def initial_effector_field(trial_specs: Any, field: str) -> jnp.ndarray:
    """Return an initial effector field from trial initial conditions."""

    for init_state in trial_specs.inits.values():
        if eqx.is_array(init_state):
            if field == "pos":
                return init_state[..., :2]
            if field == "vel":
                return init_state[..., 2:4]
        value = getattr(init_state, field, None)
        if value is not None:
            return value
        vector = getattr(init_state, "vector", None)
        if vector is not None:
            if field == "pos":
                return vector[..., :2]
            if field == "vel":
                return vector[..., 2:4]
    raise ValueError(f"Could not find initial effector {field!r} in trial specs")


def final_goal_position(trial_specs: Any) -> jnp.ndarray:
    """Return final target position for each nominal trial."""

    if not trial_specs.targets:
        raise ValueError("Trial specs do not declare targets")
    target = next(iter(trial_specs.targets.values())).value
    return target[:, -1, :]


def load_trained_model(ref: RunRef, hps: TreeNamespace, seed: int) -> Any:
    """Load a trained model with the matching template."""

    path = resolve_run_artifact_path(artifact_dir(ref), "trained_model.eqx")
    if not path.exists():
        raise FileNotFoundError(f"Missing trained model for {ref.run_id}: {path}")
    model, _hyperparameters = load_with_hyperparameters(
        path,
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )
    return model


def evaluate_profile(ref: RunRef) -> VelocityProfile:
    """Evaluate a trained run on nominal validation trials."""

    run_spec = load_run_spec(ref)
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model = load_trained_model(ref, hps, seed)
    trial_specs = nominalize_trial_specs(pair.task.validation_trials)
    n_trials = int(next(iter(trial_specs.targets.values())).value.shape[0])
    init_pos = initial_effector_field(trial_specs, "pos")
    init_vel = initial_effector_field(trial_specs, "vel")
    goal = final_goal_position(trial_specs)
    direction = goal - init_pos
    direction_unit = direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)

    def is_replicate_array(leaf: Any) -> bool:
        return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates

    model_arrays, model_other = eqx.partition(model, is_replicate_array)

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> jnp.ndarray:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(replicate_model, trial_specs, jr.split(key, n_trials))
        velocity = jnp.concatenate(
            [init_vel[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )
        return jnp.sum(velocity * direction_unit[:, None, :], axis=-1)

    forward_velocity = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    values = np.asarray(forward_velocity, dtype=np.float64)
    pooled = values.reshape(n_replicates * n_trials, values.shape[-1])
    mean = np.mean(pooled, axis=0)
    std = np.std(pooled, axis=0)
    dt = float(run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    time_s = np.arange(mean.shape[0], dtype=np.float64) * dt
    peak_idx = int(np.nanargmax(mean))
    return VelocityProfile(
        run=ref,
        time_s=time_s,
        mean=mean,
        std=std,
        n_replicates=n_replicates,
        n_trials=n_trials,
        peak_mean_forward_velocity_m_s=float(mean[peak_idx]),
        time_of_peak_mean_forward_velocity_s=float(time_s[peak_idx]),
    )


def add_band_trace(fig: go.Figure, profile: VelocityProfile) -> None:
    """Add a mean velocity trace with a one-standard-deviation band."""

    upper = profile.mean + profile.std
    lower = profile.mean - profile.std
    color = profile.run.color
    legend_group = f"{profile.run.experiment}::{profile.run.run_id}"
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.13),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            legendgroup=legend_group,
            name=profile.run.label,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile.time_s,
            y=profile.mean,
            mode="lines",
            line={"color": color, "width": 2.5},
            legendgroup=legend_group,
            name=profile.run.label,
        )
    )


def hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to a Plotly rgba color."""

    color = color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def write_outputs(profiles: list[VelocityProfile]) -> dict[str, Any]:
    """Write figure, data, manifest, and note outputs."""

    figure_dir = mkdir_p(figure_artifact_dir(EXPERIMENT, TOPIC))
    spec_dir = mkdir_p(figure_spec_dir(EXPERIMENT, TOPIC))
    notes_dir = mkdir_p(REPO_ROOT / "results" / EXPERIMENT / "notes")

    fig = go.Figure()
    for profile in profiles:
        add_band_trace(fig, profile)
    fig.update_layout(
        title="Nominal target-radial velocity profiles",
        width=960,
        height=560,
        margin={"l": 72, "r": 24, "t": 72, "b": 68},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.22, "x": 0.0, "groupclick": "togglegroup"},
    )
    fig.update_xaxes(title_text="Time (s)", zeroline=False)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)

    html_path = figure_dir / "nominal_forward_velocity_profiles.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    data_path = figure_dir / "nominal_forward_velocity_profiles.npz"
    np.savez_compressed(
        data_path,
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__time_s": profile.time_s
            for profile in profiles
        },
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__mean": profile.mean
            for profile in profiles
        },
        **{
            f"{profile.run.experiment}__{profile.run.run_id}__std": profile.std
            for profile in profiles
        },
    )

    rows = [
        {
            "experiment": profile.run.experiment,
            "run_id": profile.run.run_id,
            "label": profile.run.label,
            "n_replicates": profile.n_replicates,
            "n_trials": profile.n_trials,
            "n_pooled_profiles": profile.n_replicates * profile.n_trials,
            "peak_mean_forward_velocity_m_s": profile.peak_mean_forward_velocity_m_s,
            "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_mean_forward_velocity_s,
        }
        for profile in profiles
    ]
    manifest = {
        "schema_version": "rlrmp.e901a20.nominal_velocity_profile_comparison.v1",
        "figure": repo_relative(html_path),
        "data": repo_relative(data_path),
        "evaluation_lens": "nominal_clean_validation_trials",
        "velocity_definition": "effector velocity projected onto target direction per trial",
        "runs": rows,
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
        "evaluation_lens": manifest["evaluation_lens"],
        "velocity_definition": manifest["velocity_definition"],
        "runs": rows,
        "inputs": [
            {
                "run_spec": repo_relative(run_spec_path(ref.experiment, ref.run_id)),
                "trained_model": repo_relative(
                    resolve_run_artifact_path(artifact_dir(ref), "trained_model.eqx")
                ),
            }
            for ref in RUNS
        ],
    }
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table_lines = [
        "| Row | Peak mean forward velocity (m/s) | Time of peak (s) | Pooled profiles |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| `{row['label']}` | {row['peak_mean_forward_velocity_m_s']:.4f} | "
            f"{row['time_of_peak_mean_forward_velocity_s']:.3f} | "
            f"{row['n_pooled_profiles']} |"
        )
    note = "\n".join(
        [
            "## Nominal velocity profile comparison",
            "",
            "Nominal-clean validation trials with perturbation inputs zeroed. Curves show "
            "target-radial velocity pooled over replicates and validation trials; bands are one "
            "standard deviation over the pooled profiles.",
            "",
            *table_lines,
            "",
            f"- Figure: `{repo_relative(html_path)}`",
            f"- Data: `{repo_relative(data_path)}`",
            f"- Manifest: `{repo_relative(manifest_path)}`",
            "",
        ]
    )
    note_path = notes_dir / f"{TOPIC}.md"
    update_marked_section(note_path, NOMINAL_MARKER, note)
    manifest["note"] = repo_relative(note_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Evaluate all rows and materialize outputs."""

    profiles = [evaluate_profile(ref) for ref in RUNS]
    manifest = write_outputs(profiles)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

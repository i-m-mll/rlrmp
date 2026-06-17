"""SISU-conditioned diagnostics for the e4800d6 GRU spectrum rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from jax_cookbook import load_with_hyperparameters

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.cs_released_simulation import (
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_robust_released_forward,
)
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
)
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    ReplicateCheckpointSelection,
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    _is_replicate_array,
    cs_output_feedback_reference_profile,
    initial_effector_velocity,
    repeat_single_validation_trial,
    resolve_run_inputs,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.viz import profile_comparison_grid


EXPERIMENT = "e4800d6"
TOPIC = "sisu_spectrum_velocity_profiles"
DEFAULT_RUN_IDS = (
    "cs_gru_h0_sisu_spectrum__raw_strong_gamma_1p05_radius_lr3e-3_clip5_b64",
    "cs_gru_h0_sisu_spectrum__effective_020a65b_pgd_radius_lr3e-3_clip5_b64",
)
DEFAULT_LABELS = (
    "A: raw strong gamma=1.05 radius",
    "B: effective 020a65b PGD radius",
)
DEFAULT_SISU_LEVELS = (0.0, 0.5, 1.0)
DEFAULT_N_ROLLOUT_TRIALS = 64
CHECKPOINT_POLICY = "validation_selected_per_replicate"

LOW_SISU_ENDPOINT_REACH_THRESHOLD_M = 0.05
LOW_SISU_PEAK_SPEED_THRESHOLD_M_S = 0.2


@dataclass(frozen=True)
class SisuCurve:
    """Velocity and endpoint summary for one run at one SISU condition."""

    sisu: float
    time_s: np.ndarray
    mean_forward_velocity_m_s: np.ndarray
    std_forward_velocity_m_s: np.ndarray
    replicate_mean_forward_velocity_m_s: np.ndarray
    endpoint_error_by_replicate_m: np.ndarray
    peak_velocity_by_replicate_m_s: np.ndarray
    final_position_by_replicate_m: np.ndarray

    @property
    def endpoint_error_mean_m(self) -> float:
        """Mean terminal endpoint error over replicates."""
        return float(np.mean(self.endpoint_error_by_replicate_m))

    @property
    def peak_velocity_mean_m_s(self) -> float:
        """Mean peak speed over replicates."""
        return float(np.mean(self.peak_velocity_by_replicate_m_s))

    @property
    def final_position_mean_m(self) -> list[float]:
        """Mean final position over replicates."""
        return [
            float(value)
            for value in np.mean(self.final_position_by_replicate_m, axis=0)
        ]


@dataclass(frozen=True)
class RunSisuProfile:
    """SISU-conditioned velocity profiles for one trained run."""

    run_id: str
    label: str
    input_key: str
    target_final_position_m: list[float]
    validation_input_unique: list[float]
    validation_epsilon_l2_mean: float
    checkpoint_selection: tuple[ReplicateCheckpointSelection, ...]
    curves: tuple[SisuCurve, ...]


@dataclass(frozen=True)
class ReferenceCurve:
    """Analytical reference velocity curve."""

    label: str
    time_s: np.ndarray
    forward_velocity_m_s: np.ndarray
    std_forward_velocity_m_s: np.ndarray
    line_color: str
    line_dash: str
    controller: str
    gamma_factor: float | None = None
    gamma: float | None = None
    n_samples: int = 0


def resolve_sisu_input_key(trial_specs: Any) -> str:
    """Return the trial input key that carries SISU for these runs."""

    inputs = getattr(trial_specs, "inputs", {})
    for key in ("sisu", "input"):
        if key in inputs:
            return key
    raise ValueError("SISU-conditioned trials require an 'sisu' or 'input' input.")


def set_sisu_condition(trial_specs: Any, sisu: float, *, input_key: str | None = None) -> Any:
    """Return trial specs with the SISU scalar set to ``sisu``."""

    key = input_key or resolve_sisu_input_key(trial_specs)
    current = jnp.asarray(trial_specs.inputs[key])
    return eqx.tree_at(
        lambda t: t.inputs[key],
        trial_specs,
        jnp.full_like(current, float(sisu)),
    )


def zero_disturbance_payload(trial_specs: Any, *, input_key: str = "epsilon") -> Any:
    """Return trial specs with the broad-epsilon payload zeroed."""

    if input_key not in trial_specs.inputs:
        return trial_specs
    current = jnp.asarray(trial_specs.inputs[input_key])
    return eqx.tree_at(lambda t: t.inputs[input_key], trial_specs, jnp.zeros_like(current))


def robustification_comparison(curves: Sequence[SisuCurve]) -> dict[str, float]:
    """Compare SISU=1 against SISU=0 within one trained network."""

    by_sisu = {float(curve.sisu): curve for curve in curves}
    if 0.0 not in by_sisu or 1.0 not in by_sisu:
        return {"status": "missing_sisu_endpoint"}  # type: ignore[return-value]
    low = by_sisu[0.0]
    high = by_sisu[1.0]
    endpoint_delta = low.endpoint_error_mean_m - high.endpoint_error_mean_m
    peak_delta = high.peak_velocity_mean_m_s - low.peak_velocity_mean_m_s
    return {
        "sisu_1_endpoint_error_mean_m": high.endpoint_error_mean_m,
        "sisu_0_endpoint_error_mean_m": low.endpoint_error_mean_m,
        "endpoint_error_delta_0_minus_1_m": float(endpoint_delta),
        "endpoint_error_ratio_1_over_0": float(
            high.endpoint_error_mean_m / max(low.endpoint_error_mean_m, 1e-12)
        ),
        "sisu_1_peak_velocity_mean_m_s": high.peak_velocity_mean_m_s,
        "sisu_0_peak_velocity_mean_m_s": low.peak_velocity_mean_m_s,
        "peak_velocity_delta_1_minus_0_m_s": float(peak_delta),
        "peak_velocity_ratio_1_over_0": float(
            high.peak_velocity_mean_m_s / max(low.peak_velocity_mean_m_s, 1e-12)
        ),
    }


def evaluate_sisu_profiles(
    *,
    experiment: str = EXPERIMENT,
    run_ids: Sequence[str] = DEFAULT_RUN_IDS,
    labels: Sequence[str] = DEFAULT_LABELS,
    sisu_levels: Sequence[float] = DEFAULT_SISU_LEVELS,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    use_validation_selected_checkpoints: bool = True,
    repo_root: Path = REPO_ROOT,
) -> tuple[RunSisuProfile, ...]:
    """Evaluate SISU velocity profiles for the requested run rows."""

    runs = resolve_run_inputs(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    profiles: list[RunSisuProfile] = []
    for run in runs:
        hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
        n_replicates = int(hps.model.n_replicates)
        seed = int(run.run_spec.get("seed", 42))
        pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
        if use_validation_selected_checkpoints:
            model, selections = load_validation_selected_checkpoint_model(
                experiment=experiment,
                run_id=run.run_id,
                run_spec=run.run_spec,
                repo_root=repo_root,
            )
        else:
            model, _hyperparameters = load_with_hyperparameters(
                run.artifact_dir / "trained_model.eqx",
                setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
            )
            selections = []

        base_trials = repeat_single_validation_trial(
            pair.task.validation_trials,
            n_rollout_trials,
        )
        input_key = resolve_sisu_input_key(base_trials)
        base_trials = zero_disturbance_payload(base_trials)
        target = _target_final_position(base_trials)
        initial_velocity = initial_effector_velocity(base_trials)
        model_arrays, model_other = eqx.partition(
            model,
            lambda leaf: _is_replicate_array(leaf, n_replicates),
        )
        dt = float(run.run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
        curves: list[SisuCurve] = []
        for sisu in sisu_levels:
            trials = set_sisu_condition(base_trials, float(sisu), input_key=input_key)

            def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
                replicate_model = eqx.combine(model_array_leaves, model_other)
                return pair.task.eval_trials(
                    replicate_model,
                    trials,
                    jr.split(key, n_rollout_trials),
                )

            states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
                model_arrays,
                jr.split(jr.PRNGKey(0), n_replicates),
            )
            position = np.asarray(states.mechanics.effector.pos, dtype=np.float64)
            velocity = np.asarray(states.mechanics.effector.vel, dtype=np.float64)
            initial_velocity_array = np.asarray(initial_velocity, dtype=np.float64)
            initial_velocity_array = np.broadcast_to(
                initial_velocity_array[None, :, None, :],
                (n_replicates, n_rollout_trials, 1, initial_velocity_array.shape[-1]),
            )
            velocity_with_initial = np.concatenate([initial_velocity_array, velocity], axis=2)
            forward = velocity_with_initial[..., 0]
            pooled_forward = forward.reshape(n_replicates * n_rollout_trials, forward.shape[-1])
            endpoint_error = np.linalg.norm(position[:, :, -1, :] - target[None, :, :], axis=-1)
            speed = np.linalg.norm(velocity, axis=-1)
            peak_speed = np.max(speed, axis=-1)
            curves.append(
                SisuCurve(
                    sisu=float(sisu),
                    time_s=np.arange(pooled_forward.shape[-1], dtype=np.float64) * dt,
                    mean_forward_velocity_m_s=np.mean(pooled_forward, axis=0),
                    std_forward_velocity_m_s=np.std(pooled_forward, axis=0),
                    replicate_mean_forward_velocity_m_s=np.mean(forward, axis=1),
                    endpoint_error_by_replicate_m=np.mean(endpoint_error, axis=1),
                    peak_velocity_by_replicate_m_s=np.mean(peak_speed, axis=1),
                    final_position_by_replicate_m=np.mean(position[:, :, -1, :], axis=1),
                )
            )

        validation_input = np.asarray(pair.task.validation_trials.inputs[input_key])
        validation_epsilon = np.asarray(pair.task.validation_trials.inputs["epsilon"])
        profiles.append(
            RunSisuProfile(
                run_id=run.run_id,
                label=run.label,
                input_key=input_key,
                target_final_position_m=[float(value) for value in np.mean(target, axis=0)],
                validation_input_unique=sorted(float(value) for value in np.unique(validation_input)),
                validation_epsilon_l2_mean=float(
                    np.mean(
                        np.linalg.norm(
                            validation_epsilon.reshape(validation_epsilon.shape[0], -1),
                            axis=1,
                        )
                    )
                ),
                checkpoint_selection=tuple(selections),
                curves=tuple(curves),
            )
        )
    return tuple(profiles)


def analytical_reference_curves(
    *,
    n_samples: int,
    key: Any = jr.PRNGKey(0),
) -> tuple[ReferenceCurve, ...]:
    """Return extLQG and analytical H-infinity reference velocity curves."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    config = OutputFeedbackConfig()
    ext = cs_output_feedback_reference_profile(
        reference=reference,
        config=config,
        label="extLQG analytical reference",
        n_samples=n_samples,
        key=key,
        line_color="#111827",
        line_dash="dash",
    )
    gamma_ref = reference.gamma_references[0]
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    covariances = default_cs_noise_covariances(reference.plant, config)
    rollouts = [
        simulate_robust_released_forward(
            reference.plant,
            reference.schedule,
            gamma_ref.solution,
            x0,
            draws=sample_forward_noise_draws(
                sample_key,
                T=reference.schedule.T,
                covariances=covariances,
            ),
            covariances=covariances,
            config=config,
        )
        for sample_key in jr.split(jr.fold_in(key, 1), n_samples)
    ]
    x = np.stack([np.asarray(rollout.x, dtype=np.float64) for rollout in rollouts], axis=0)
    vel_lo, _vel_hi = reference.plant.vel_slice
    forward = x[:, :, vel_lo]
    hinf = ReferenceCurve(
        label="H-infinity analytical reference",
        time_s=np.arange(forward.shape[1], dtype=np.float64) * float(reference.plant.dt),
        forward_velocity_m_s=np.mean(forward, axis=0),
        std_forward_velocity_m_s=np.std(forward, axis=0),
        line_color="#dc2626",
        line_dash="dot",
        controller="analytical_hinf_output_feedback",
        gamma_factor=float(gamma_ref.factor),
        gamma=float(gamma_ref.gamma),
        n_samples=int(n_samples),
    )
    return (
        ReferenceCurve(
            label=ext.label,
            time_s=ext.time_s,
            forward_velocity_m_s=ext.forward_velocity,
            std_forward_velocity_m_s=ext.forward_velocity_std,
            line_color=ext.line_color,
            line_dash=ext.line_dash,
            controller="analytical_extlqg_output_feedback",
            gamma_factor=float(ext.gamma_factor),
            n_samples=int(ext.n_samples),
        ),
        hinf,
    )


def build_velocity_profile_figure(
    profiles: Sequence[RunSisuProfile],
    references: Sequence[ReferenceCurve],
) -> go.Figure:
    """Build the two-panel SISU velocity profile figure."""

    fig = profile_comparison_grid(
        n_panels=len(profiles),
        subplot_titles=[profile.label for profile in profiles],
        vertical_spacing=0.08,
    )
    for row_idx, profile in enumerate(profiles, start=1):
        for reference in references:
            fig.add_trace(
                go.Scatter(
                    x=reference.time_s,
                    y=reference.forward_velocity_m_s,
                    mode="lines",
                    line={
                        "color": reference.line_color,
                        "width": 2.0,
                        "dash": reference.line_dash,
                    },
                    name=reference.label,
                    legendgroup=f"ref-{reference.label}",
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )
        for curve in profile.curves:
            fig.add_trace(
                go.Scatter(
                    x=curve.time_s,
                    y=curve.mean_forward_velocity_m_s,
                    mode="lines",
                    line={"color": _sisu_color(curve.sisu), "width": 2.6},
                    name=f"SISU={curve.sisu:g}",
                    legendgroup=f"sisu-{curve.sisu:g}",
                    showlegend=row_idx == 1,
                    customdata=np.column_stack(
                        [
                            np.full_like(curve.time_s, curve.sisu, dtype=np.float64),
                            curve.std_forward_velocity_m_s,
                        ]
                    ),
                    hovertemplate=(
                        "time=%{x:.3f}s<br>"
                        "velocity=%{y:.4f}m/s<br>"
                        "SISU=%{customdata[0]:.2f}<br>"
                        "SD=%{customdata[1]:.4f}m/s<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "color": [0.0],
                "colorbar": {"title": "SISU", "len": 0.55},
                "showscale": True,
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=(
            "SISU-conditioned nominal forward velocity profiles "
            "(validation-selected GRU checkpoints)"
        ),
        width=920,
        height=max(700, 390 * len(profiles)),
        margin={"l": 74, "r": 110, "t": 82, "b": 64},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    fig.update_xaxes(title_text="Time (s)", row=len(profiles), col=1)
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True)
    fig.update_yaxes(matches="y")
    return fig


def build_manifest(
    *,
    experiment: str,
    topic: str,
    profiles: Sequence[RunSisuProfile],
    references: Sequence[ReferenceCurve],
    compact_npz_path: Path,
    figure_spec_path: Path,
    figure_html_path: Path,
    repo_root: Path,
    sisu_levels: Sequence[float] = DEFAULT_SISU_LEVELS,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
) -> dict[str, Any]:
    """Build the JSON summary manifest."""

    low_sisu_behavior = summarize_low_sisu_behavior(profiles)
    return {
        "schema_id": "rlrmp.sisu_spectrum_special.v2",
        "issue": experiment,
        "topic": topic,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "interpretation": (
            "Discovery-trained robustness behavior. Not teacher/distillation behavior "
            "and not a formal H-infinity equivalence claim."
        ),
        "verified_low_sisu_behavior": low_sisu_behavior,
        "inputs": {
            "sisu_levels": list(sisu_levels),
            "n_rollout_trials_per_replicate": int(n_rollout_trials),
            "nominal_profile_epsilon_policy": "trial_specs.inputs['epsilon'] zeroed",
            "low_sisu_endpoint_reach_threshold_m": LOW_SISU_ENDPOINT_REACH_THRESHOLD_M,
            "low_sisu_peak_speed_threshold_m_s": LOW_SISU_PEAK_SPEED_THRESHOLD_M_S,
        },
        "outputs": {
            "compact_arrays": str(compact_npz_path.relative_to(repo_root)),
            "figure_spec": str(figure_spec_path.relative_to(repo_root)),
            "figure_html": str(figure_html_path.relative_to(repo_root)),
        },
        "references": {
            reference.label: {
                "controller": reference.controller,
                "gamma_factor": reference.gamma_factor,
                "gamma": reference.gamma,
                "n_samples": reference.n_samples,
                "n_time_steps": int(reference.forward_velocity_m_s.shape[0]),
                "peak_forward_velocity_m_s": float(np.max(reference.forward_velocity_m_s)),
            }
            for reference in references
        },
        "runs": {
            profile.run_id: {
                "label": profile.label,
                "input_key": profile.input_key,
                "target_final_position_m": profile.target_final_position_m,
                "validation_input_unique": profile.validation_input_unique,
                "validation_epsilon_l2_mean": profile.validation_epsilon_l2_mean,
                "checkpoint_selection": [
                    selection.to_json() for selection in profile.checkpoint_selection
                ],
                "curves": {
                    f"sisu_{curve.sisu:g}": curve_summary(curve)
                    for curve in profile.curves
                },
                "within_network_robustification_sisu_1_vs_0": robustification_comparison(
                    profile.curves
                ),
            }
            for profile in profiles
        },
    }


def summarize_low_sisu_behavior(profiles: Sequence[RunSisuProfile]) -> str:
    """Summarize whether SISU 0.0 and 0.5 show non-reaching behavior."""

    failures: list[str] = []
    for profile in profiles:
        by_sisu = {float(curve.sisu): curve for curve in profile.curves}
        for sisu in (0.0, 0.5):
            curve = by_sisu.get(sisu)
            if curve is None:
                failures.append(f"{profile.label} SISU={sisu:g} missing")
                continue
            if curve.endpoint_error_mean_m > LOW_SISU_ENDPOINT_REACH_THRESHOLD_M:
                failures.append(
                    f"{profile.label} SISU={sisu:g} endpoint "
                    f"{curve.endpoint_error_mean_m:.4f}m"
                )
            if curve.peak_velocity_mean_m_s < LOW_SISU_PEAK_SPEED_THRESHOLD_M_S:
                failures.append(
                    f"{profile.label} SISU={sisu:g} peak "
                    f"{curve.peak_velocity_mean_m_s:.4f}m/s"
                )
    if failures:
        return (
            "Low-SISU reaching check did not fully pass by the configured thresholds: "
            + "; ".join(failures)
            + ". Inspect the targetfix figure and per-SISU metrics."
        )
    return (
        "Low-SISU reaching check passed: SISU 0.0 and 0.5 have endpoint errors "
        f"<= {LOW_SISU_ENDPOINT_REACH_THRESHOLD_M:.3f} m and peak speeds >= "
        f"{LOW_SISU_PEAK_SPEED_THRESHOLD_M_S:.3f} m/s in both targetfix rows."
    )


def curve_summary(curve: SisuCurve) -> dict[str, Any]:
    """Return JSON-compatible scalar metrics for a SISU curve."""

    peak_idx = int(np.argmax(curve.mean_forward_velocity_m_s))
    return {
        "endpoint_error_mean_m": curve.endpoint_error_mean_m,
        "endpoint_error_by_replicate_m": [
            float(value) for value in curve.endpoint_error_by_replicate_m
        ],
        "peak_velocity_mean_m_s": curve.peak_velocity_mean_m_s,
        "peak_velocity_by_replicate_m_s": [
            float(value) for value in curve.peak_velocity_by_replicate_m_s
        ],
        "mean_forward_velocity_peak_m_s": float(curve.mean_forward_velocity_m_s[peak_idx]),
        "mean_forward_velocity_peak_time_s": float(curve.time_s[peak_idx]),
        "final_position_mean_m": curve.final_position_mean_m,
        "final_position_by_replicate_m": [
            [float(value) for value in row] for row in curve.final_position_by_replicate_m
        ],
    }


def write_compact_arrays(
    *,
    profiles: Sequence[RunSisuProfile],
    references: Sequence[ReferenceCurve],
    path: Path,
) -> None:
    """Write compact regenerable velocity profile arrays."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for profile_idx, profile in enumerate(profiles):
        prefix = f"run_{profile_idx}"
        arrays[f"{prefix}_run_id"] = np.asarray(profile.run_id)
        for curve in profile.curves:
            sisu_tag = str(curve.sisu).replace(".", "p")
            arrays[f"{prefix}_sisu_{sisu_tag}_time_s"] = curve.time_s
            arrays[f"{prefix}_sisu_{sisu_tag}_mean_forward_velocity_m_s"] = (
                curve.mean_forward_velocity_m_s
            )
            arrays[f"{prefix}_sisu_{sisu_tag}_std_forward_velocity_m_s"] = (
                curve.std_forward_velocity_m_s
            )
            arrays[f"{prefix}_sisu_{sisu_tag}_replicate_mean_forward_velocity_m_s"] = (
                curve.replicate_mean_forward_velocity_m_s
            )
    for reference_idx, reference in enumerate(references):
        prefix = f"reference_{reference_idx}"
        arrays[f"{prefix}_label"] = np.asarray(reference.label)
        arrays[f"{prefix}_time_s"] = reference.time_s
        arrays[f"{prefix}_forward_velocity_m_s"] = reference.forward_velocity_m_s
        arrays[f"{prefix}_std_forward_velocity_m_s"] = reference.std_forward_velocity_m_s
    np.savez_compressed(path, **arrays)


def write_note(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write or update the SISU special Markdown note."""

    update_marked_section(path, "sisu_spectrum_special", render_markdown(manifest))


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render the special SISU note from the manifest."""

    lines = [
        "# SISU Spectrum Special Analysis",
        "",
        (
            "This is a SISU-conditioned post-run analysis for the two e4800d6 H0 "
            "spectrum rows. It is discovery-trained robustness evidence, not "
            "teacher/distillation behavior and not formal H-infinity equivalence."
        ),
        "",
        f"**Low-SISU check:** {manifest['verified_low_sisu_behavior']}",
        "",
        "## Velocity Profiles",
        "",
        f"- Figure spec: `{manifest['outputs']['figure_spec']}`",
        f"- Figure render: `{manifest['outputs']['figure_html']}`",
        f"- Compact arrays: `{manifest['outputs']['compact_arrays']}`",
        "",
        "## Within-Network SISU=1 vs SISU=0 Comparison",
        "",
        "| row | SISU=0 endpoint (m) | SISU=1 endpoint (m) | endpoint ratio 1/0 | SISU=0 peak (m/s) | SISU=1 peak (m/s) | peak ratio 1/0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id, row in manifest["runs"].items():
        comparison = row["within_network_robustification_sisu_1_vs_0"]
        lines.append(
            "| "
            f"{row['label']} | "
            f"{comparison['sisu_0_endpoint_error_mean_m']:.6f} | "
            f"{comparison['sisu_1_endpoint_error_mean_m']:.6f} | "
            f"{comparison['endpoint_error_ratio_1_over_0']:.5f} | "
            f"{comparison['sisu_0_peak_velocity_mean_m_s']:.6f} | "
            f"{comparison['sisu_1_peak_velocity_mean_m_s']:.6f} | "
            f"{comparison['peak_velocity_ratio_1_over_0']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Input Contract",
            "",
            (
                "Both rows use the SISU scalar on `trial_specs.inputs['input']`; "
                "the materialized validation bank has `input = 1.0` by default "
                "and no separate `sisu` key. The special profile materializer "
                "therefore changes `input` to 0.0, 0.5, and 1.0 and zeroes "
                "`epsilon` for the nominal profile comparison."
            ),
            "",
            "## Per-SISU Metrics",
            "",
            "| row | SISU | endpoint error mean (m) | peak velocity mean (m/s) | final position mean (m) |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for _run_id, row in manifest["runs"].items():
        for curve_key in ("sisu_0", "sisu_0.5", "sisu_1"):
            curve = row["curves"].get(curve_key)
            if curve is None:
                continue
            sisu = curve_key.removeprefix("sisu_")
            lines.append(
                "| "
                f"{row['label']} | {sisu} | "
                f"{curve['endpoint_error_mean_m']:.6f} | "
                f"{curve['peak_velocity_mean_m_s']:.6f} | "
                f"{_fmt_pair(curve['final_position_mean_m'])} |"
            )
    return "\n".join(lines) + "\n"


def _target_final_position(trial_specs: Any) -> np.ndarray:
    """Return final target position for each trial, shape ``(trials, 2)``."""

    if "effector_target" in trial_specs.inputs:
        return np.asarray(trial_specs.inputs["effector_target"].pos[..., -1, :], dtype=np.float64)
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    return np.asarray(target_spec.value[..., -1, :], dtype=np.float64)


def _fmt_pair(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def _sisu_color(sisu: float) -> str:
    """Sample the continuous Viridis scale for a SISU value in [0, 1]."""

    value = min(max(float(sisu), 0.0), 1.0)
    return str(sample_colorscale("Viridis", [value])[0])


__all__ = [
    "CHECKPOINT_POLICY",
    "DEFAULT_LABELS",
    "DEFAULT_N_ROLLOUT_TRIALS",
    "DEFAULT_RUN_IDS",
    "DEFAULT_SISU_LEVELS",
    "EXPERIMENT",
    "TOPIC",
    "SisuCurve",
    "RunSisuProfile",
    "ReferenceCurve",
    "analytical_reference_curves",
    "build_manifest",
    "build_velocity_profile_figure",
    "curve_summary",
    "evaluate_sisu_profiles",
    "render_markdown",
    "resolve_sisu_input_key",
    "robustification_comparison",
    "set_sisu_condition",
    "summarize_low_sisu_behavior",
    "write_compact_arrays",
    "write_note",
    "zero_disturbance_payload",
]

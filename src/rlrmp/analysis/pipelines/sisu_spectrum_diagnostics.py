"""SISU-conditioned diagnostics for GRU spectrum rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe
from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput
from feedbax.analysis.specs import register_analysis_recipe
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from feedbax.contracts.manifest import EvaluationRunSpec
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.analysis.data_products import load_analysis_parameter_preset
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
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    cs_output_feedback_reference_profile,
)
from rlrmp.eval import sisu_spectrum as sisu_eval
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT
from rlrmp.viz import profile_comparison_grid


SISU_SPECTRUM_ANALYSIS_TYPE = "rlrmp.sisu_spectrum"
SISU_SPECTRUM_EVALUATION_TYPE = "rlrmp.sisu_spectrum_evaluation"
SISU_SPECTRUM_STATES_SCHEMA = "rlrmp.sisu_spectrum.states.v1"
DEFAULT_TOPIC = "sisu_spectrum_velocity_profiles"
_ANALYSIS_PRESET = load_analysis_parameter_preset("sisu_spectrum_diagnostics").parameters
DEFAULT_SISU_LEVELS = tuple(_ANALYSIS_PRESET["sisu_levels"])
DEFAULT_N_ROLLOUT_TRIALS = int(_ANALYSIS_PRESET["n_rollout_trials"])
DEFAULT_REFERENCE_SAMPLES = int(_ANALYSIS_PRESET["reference_samples"])
CHECKPOINT_POLICY = "validation_selected_per_replicate"

LOW_SISU_ENDPOINT_REACH_THRESHOLD_M = 0.05
LOW_SISU_PEAK_SPEED_THRESHOLD_M_S = 0.2


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


class SisuSpectrumEvaluationParams(BaseModel):
    """Params for the SISU-spectrum evaluation recipe."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str | None = None
    schema_version: str | None = None
    experiment: str
    run_ids: list[str] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    topic: str = DEFAULT_TOPIC
    sisu_levels: list[float] = Field(default_factory=lambda: list(DEFAULT_SISU_LEVELS))
    n_rollout_trials: int = Field(DEFAULT_N_ROLLOUT_TRIALS, ge=1)
    reference_samples: int = Field(DEFAULT_REFERENCE_SAMPLES, ge=1)
    use_validation_selected_checkpoints: bool = True
    output_stem: str = "sisu_spectrum_special"
    note_marker: str = "sisu_spectrum_special"
    note_output: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if len(self.run_ids) != len(self.labels):
            raise ValueError("run_ids and labels must have the same length")


class SisuSpectrumAnalysis(AbstractAnalysis):
    """Render SISU-spectrum profile figures or markdown from evaluation states."""

    output: str = "velocity_profiles"

    def __post_init__(self):
        super().__post_init__()
        if self.output not in {"velocity_profiles", "notes"}:
            raise ValueError(f"Unknown SISU spectrum output {self.output!r}")

    def compute(self, data: AnalysisInputData, **kwargs):
        states = _sisu_states(data.states)
        if self.output == "notes":
            return render_markdown(states["manifest"])
        return states

    def make_figs(self, data: AnalysisInputData, *, result, **kwargs):
        if self.output == "notes":
            return None
        return {
            "sisu_spectrum_velocity_profiles": build_velocity_profile_figure(
                result["profiles"],
                result["references"],
            )
        }

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result,
        **kwargs,
    ):
        states = _sisu_states(data.states)
        manifest = states["manifest"]
        refs: dict[str, Any] = {}
        if self.output == "notes":
            params = _params_from_states(states)
            note_path = _note_output_path(context, params)
            note_path.parent.mkdir(parents=True, exist_ok=True)
            update_marked_section(note_path, params.note_marker, str(result).rstrip() + "\n")
            refs["notes"] = context.record_artifact(
                note_path,
                role="analysis_notes",
                logical_name=f"sisu_spectrum/{note_path.name}",
                media_type="text/markdown",
                metadata={"marker": params.note_marker},
            )
        refs["manifest"] = context.record_json_artifact(
            manifest,
            role="rlrmp-sisu-spectrum-manifest",
            logical_name="sisu_spectrum/manifest.json",
            metadata={"states_schema": SISU_SPECTRUM_STATES_SCHEMA},
        )
        return {
            "manifest": manifest,
            "artifact_refs": refs,
        }

    def _params_to_save(self, hps, *, result, **kwargs):
        states = _sisu_states(result) if isinstance(result, Mapping) else {}
        params = _params_from_states(states) if states else None
        return {
            "sisu_spectrum_output": self.output,
            "topic": None if params is None else params.topic,
        }


def register_sisu_spectrum_recipes(*, replace: bool = True) -> None:
    """Register SISU-spectrum evaluation and analysis recipes."""

    register_evaluation_recipe(
        SISU_SPECTRUM_EVALUATION_TYPE,
        sisu_spectrum_evaluation_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        SISU_SPECTRUM_ANALYSIS_TYPE,
        sisu_spectrum_recipe,
        replace=replace,
    )


def sisu_spectrum_evaluation_spec_params(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str],
    topic: str = DEFAULT_TOPIC,
    sisu_levels: Sequence[float] = DEFAULT_SISU_LEVELS,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    reference_samples: int = DEFAULT_REFERENCE_SAMPLES,
    use_validation_selected_checkpoints: bool = True,
    output_stem: str = "sisu_spectrum_special",
    note_marker: str = "sisu_spectrum_special",
    note_output: Path | str | None = None,
) -> dict[str, Any]:
    """Return validated JSON params for a SISU-spectrum evaluation run."""

    return SisuSpectrumEvaluationParams(
        experiment=experiment,
        run_ids=[str(run_id) for run_id in run_ids],
        labels=[str(label) for label in labels],
        topic=topic,
        sisu_levels=[float(value) for value in sisu_levels],
        n_rollout_trials=n_rollout_trials,
        reference_samples=reference_samples,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        output_stem=output_stem,
        note_marker=note_marker,
        note_output=None if note_output is None else str(note_output),
    ).model_dump(mode="json", exclude_none=True)


def sisu_spectrum_evaluation_recipe(
    run_spec: EvaluationRunSpec,
    _root: Path,
    _states_path: Path,
) -> EvaluationRecipeResult:
    """Evaluate SISU velocity profiles for an evaluation manifest."""

    params = SisuSpectrumEvaluationParams.model_validate(run_spec.params)
    profiles = sisu_eval.evaluate_sisu_profiles(
        experiment=params.experiment,
        run_ids=tuple(params.run_ids),
        labels=tuple(params.labels),
        sisu_levels=tuple(params.sisu_levels),
        n_rollout_trials=params.n_rollout_trials,
        use_validation_selected_checkpoints=params.use_validation_selected_checkpoints,
        repo_root=REPO_ROOT,
    )
    references = analytical_reference_curves(
        n_samples=max(len(profiles) * params.n_rollout_trials, params.reference_samples)
    )
    manifest = build_manifest(
        experiment=params.experiment,
        topic=params.topic,
        profiles=profiles,
        references=references,
        compact_npz_path=REPO_ROOT
        / "_artifacts"
        / params.experiment
        / params.output_stem
        / "sisu_velocity_profile_curves.npz",
        figure_spec_path=REPO_ROOT
        / "results"
        / params.experiment
        / "figures"
        / params.topic
        / "spec.json",
        figure_html_path=REPO_ROOT
        / "_artifacts"
        / params.experiment
        / "figures"
        / params.topic
        / "figure.html",
        repo_root=REPO_ROOT,
        sisu_levels=tuple(params.sisu_levels),
        n_rollout_trials=params.n_rollout_trials,
    )
    return EvaluationRecipeResult(
        states={
            "profiles": profiles,
            "references": references,
            "manifest": manifest,
            "params": params.model_dump(mode="json", exclude_none=True),
        },
        summary_metrics={
            "sisu_spectrum_runs": len(profiles),
            "sisu_spectrum_levels": len(params.sisu_levels),
        },
        metadata={
            "states_schema": SISU_SPECTRUM_STATES_SCHEMA,
            "topic": params.topic,
            "checkpoint_policy": CHECKPOINT_POLICY,
        },
    )


def sisu_spectrum_recipe(
    spec,
    _root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Build SISU-spectrum analyses from evaluation-manifest input states."""

    states = _states_from_inputs(inputs)
    data = AnalysisInputData(
        models={},
        tasks={},
        states=states,
        hps={"sisu_spectrum": TreeNamespace(task=TreeNamespace(eval_n=len(states["profiles"])))},
        extras={"params": states.get("params", {})},
    )
    return AnalysisRecipeResult(
        analyses={
            "velocity_profiles": SisuSpectrumAnalysis(
                output="velocity_profiles",
                variant="sisu_spectrum",
            ),
            "notes": SisuSpectrumAnalysis(output="notes", variant="sisu_spectrum"),
        },
        data=data,
    )


sisu_spectrum_recipe.EVAL_DEPENDENCIES = (SISU_SPECTRUM_EVALUATION_TYPE,)


def robustification_comparison(curves: Sequence[sisu_eval.SisuCurve]) -> dict[str, float]:
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
    profiles: Sequence[sisu_eval.RunSisuProfile],
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
    profiles: Sequence[sisu_eval.RunSisuProfile],
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
                    f"sisu_{curve.sisu:g}": curve_summary(curve) for curve in profile.curves
                },
                "within_network_robustification_sisu_1_vs_0": robustification_comparison(
                    profile.curves
                ),
            }
            for profile in profiles
        },
    }


def summarize_low_sisu_behavior(profiles: Sequence[sisu_eval.RunSisuProfile]) -> str:
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
                    f"{profile.label} SISU={sisu:g} endpoint {curve.endpoint_error_mean_m:.4f}m"
                )
            if curve.peak_velocity_mean_m_s < LOW_SISU_PEAK_SPEED_THRESHOLD_M_S:
                failures.append(
                    f"{profile.label} SISU={sisu:g} peak {curve.peak_velocity_mean_m_s:.4f}m/s"
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


def curve_summary(curve: sisu_eval.SisuCurve) -> dict[str, Any]:
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


def _states_from_inputs(inputs: Sequence[ResolvedAnalysisInput]) -> dict[str, Any]:
    if not inputs:
        raise ValueError("SISU spectrum analysis requires an evaluation manifest input")
    if len(inputs) != 1:
        raise ValueError("SISU spectrum analysis expects exactly one evaluation input")
    states = inputs[0].states
    if states is None:
        raise ValueError("SISU spectrum evaluation input has no cached states")
    return _sisu_states(states)


def _sisu_states(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("SISU spectrum states must be a mapping")
    required = {"profiles", "references", "manifest"}
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"SISU spectrum states missing required keys: {missing}")
    return dict(value)


def _params_from_states(states: Mapping[str, Any]) -> SisuSpectrumEvaluationParams:
    params = states.get("params")
    if isinstance(params, SisuSpectrumEvaluationParams):
        return params
    if not isinstance(params, Mapping):
        manifest = states.get("manifest", {})
        if isinstance(manifest, Mapping):
            params = {
                "experiment": manifest.get("issue", "unknown"),
                "run_ids": list((manifest.get("runs") or {}).keys()),
                "labels": [
                    str(row.get("label", run_id))
                    for run_id, row in (manifest.get("runs") or {}).items()
                    if isinstance(row, Mapping)
                ],
                "topic": manifest.get("topic", DEFAULT_TOPIC),
            }
        else:
            params = {}
    return SisuSpectrumEvaluationParams.model_validate(params)


def _note_output_path(
    _context: AnalysisRunContext,
    params: SisuSpectrumEvaluationParams,
) -> Path:
    if params.note_output is not None:
        path = Path(params.note_output)
        return path if path.is_absolute() else REPO_ROOT / path
    return REPO_ROOT / "results" / params.experiment / "notes" / f"{params.output_stem}.md"


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render the special SISU note from the manifest."""

    lines = [
        "# SISU Spectrum Special Analysis",
        "",
        (
            "This is a SISU-conditioned post-run analysis for the requested "
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


def _fmt_pair(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def _sisu_color(sisu: float) -> str:
    """Sample the continuous Viridis scale for a SISU value in [0, 1]."""

    value = min(max(float(sisu), 0.0), 1.0)
    return str(sample_colorscale("Viridis", [value])[0])


__all__ = [
    "CHECKPOINT_POLICY",
    "DEFAULT_N_ROLLOUT_TRIALS",
    "DEFAULT_REFERENCE_SAMPLES",
    "DEFAULT_SISU_LEVELS",
    "DEFAULT_TOPIC",
    "SISU_SPECTRUM_ANALYSIS_TYPE",
    "SISU_SPECTRUM_EVALUATION_TYPE",
    "SISU_SPECTRUM_STATES_SCHEMA",
    "SisuSpectrumAnalysis",
    "SisuSpectrumEvaluationParams",
    "ReferenceCurve",
    "analytical_reference_curves",
    "build_manifest",
    "build_velocity_profile_figure",
    "curve_summary",
    "register_sisu_spectrum_recipes",
    "render_markdown",
    "robustification_comparison",
    "sisu_spectrum_evaluation_recipe",
    "sisu_spectrum_evaluation_spec_params",
    "sisu_spectrum_recipe",
    "summarize_low_sisu_behavior",
]

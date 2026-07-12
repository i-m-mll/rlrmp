"""SISU-conditioned diagnostics for GRU spectrum rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax.random as jr
import numpy as np
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
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
    simulate_robust_released_forward,
)
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
)
from rlrmp.eval import sisu_spectrum as sisu_eval
from rlrmp.paths import REPO_ROOT


SISU_SPECTRUM_ANALYSIS_TYPE = "rlrmp.sisu_spectrum"
SISU_SPECTRUM_EVALUATION_TYPE = "rlrmp.sisu_spectrum_evaluation"
SISU_ROBUSTIFICATION_ANALYSIS_TYPE = "rlrmp.sisu_robustification_comparison"
SISU_SPECTRUM_STATES_SCHEMA = "rlrmp.sisu_spectrum.states.v1"
SISU_SPECTRUM_MANIFEST_SCHEMA = "rlrmp.sisu_spectrum_special.v3"
SISU_SPECTRUM_EVALUATION_PARAMS_SCHEMA = "rlrmp.sisu_spectrum.evaluation_params"
SISU_SPECTRUM_ANALYSIS_PARAMS_SCHEMA = "rlrmp.sisu_spectrum.analysis_params"
SISU_SPECTRUM_MANIFEST_ROLE = "rlrmp-sisu-spectrum-manifest"
SISU_PERTURBATION_COMPARISON_SCHEMA = "rlrmp.sisu_perturbation_class_comparison.v1"
DEFAULT_TOPIC = "sisu_spectrum_velocity_profiles"
_ANALYSIS_PRESET = load_analysis_parameter_preset("sisu_spectrum_diagnostics").parameters
DEFAULT_SISU_LEVELS = tuple(_ANALYSIS_PRESET["sisu_levels"])
DEFAULT_N_ROLLOUT_TRIALS = int(_ANALYSIS_PRESET["n_rollout_trials"])
DEFAULT_REFERENCE_SAMPLES = int(_ANALYSIS_PRESET["reference_samples"])
CHECKPOINT_POLICY = "validation_selected_per_replicate"

LOW_SISU_ENDPOINT_REACH_THRESHOLD_M = 0.05
LOW_SISU_PEAK_SPEED_THRESHOLD_M_S = 0.2

METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("delta_action_norm", "mean_delta_action", "response_magnitude"),
    ("delta_position_response_m.max", "max_delta_x_m", "response_magnitude"),
    ("delta_position_response_m.auc", "auc_delta_x_m_s", "response_magnitude"),
    ("delta_endpoint_error_m", "mean_endpoint_delta_m", "signed_endpoint_delta"),
    (
        "delta_terminal_speed_m_s",
        "mean_terminal_speed_delta_m_s",
        "signed_endpoint_delta",
    ),
    ("extra_full_qrf_delta_cost_total", "mean_full_qrf_delta_cost", "cost_delta"),
)


@dataclass(frozen=True)
class ReferenceCurve:
    """Analytical reference velocity curve."""

    label: str
    time_s: np.ndarray
    forward_velocity_m_s: np.ndarray
    std_forward_velocity_m_s: np.ndarray
    controller: str
    gamma_factor: float | None = None
    gamma: float | None = None
    n_samples: int = 0


class SisuSpectrumEvaluationParams(BaseModel):
    """Params for the SISU-spectrum evaluation recipe."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["rlrmp.sisu_spectrum.evaluation_params"]
    schema_version: Literal["v2"]
    experiment: str
    run_ids: list[str] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    topic: str = DEFAULT_TOPIC
    sisu_levels: list[float] = Field(default_factory=lambda: list(DEFAULT_SISU_LEVELS))
    n_rollout_trials: int = Field(DEFAULT_N_ROLLOUT_TRIALS, ge=1)
    reference_samples: int = Field(DEFAULT_REFERENCE_SAMPLES, ge=1)
    use_validation_selected_checkpoints: bool = True

    def model_post_init(self, __context: Any) -> None:
        if len(self.run_ids) != len(self.labels):
            raise ValueError("run_ids and labels must have the same length")


class SisuSpectrumAnalysisParams(BaseModel):
    """Versioned params that identify SISU analysis custody semantics."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["rlrmp.sisu_spectrum.analysis_params"]
    schema_version: Literal["v1"]
    manifest_schema: Literal["rlrmp.sisu_spectrum_special.v3"]


class SisuRobustificationAnalysisParams(BaseModel):
    """Params for grouped SISU=1 versus SISU=0 perturbation comparisons."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["rlrmp.sisu_robustification.analysis_params"]
    schema_version: Literal["v1"]
    low_sisu: float = 0.0
    high_sisu: float = 1.0


class SisuSpectrumAnalysis(AbstractAnalysis):
    """Project cached SISU states into figure-ready structured science."""

    def compute(self, data: AnalysisInputData, **_kwargs):
        states = _sisu_states(data.states)
        return {
            "schema_id": SISU_SPECTRUM_MANIFEST_SCHEMA,
            "summary": states["manifest"],
            "profiles": profile_payload(states["profiles"]),
            "references": reference_payload(states["references"]),
        }

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result,
        **_kwargs,
    ):
        artifact = context.record_json_artifact(
            result,
            role=SISU_SPECTRUM_MANIFEST_ROLE,
            logical_name="sisu_spectrum/structured_analysis.json",
            metadata={"states_schema": SISU_SPECTRUM_STATES_SCHEMA},
        )
        return {**result, "artifact_refs": {"structured_analysis": artifact}}


class SisuRobustificationAnalysis(AbstractAnalysis):
    """Emit grouped robustification science from paired cached summaries."""

    def compute(self, data: AnalysisInputData, **_kwargs):
        run_summaries = data.states["run_summaries"]
        return build_perturbation_comparison(run_summaries)

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result,
        **_kwargs,
    ):
        artifact = context.record_json_artifact(
            result,
            role="rlrmp-sisu-robustification-comparison",
            logical_name="sisu_spectrum/robustification_comparison.json",
        )
        return {**result, "artifact_refs": {"structured_analysis": artifact}}


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
    register_analysis_recipe(
        SISU_ROBUSTIFICATION_ANALYSIS_TYPE,
        sisu_robustification_recipe,
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
) -> dict[str, Any]:
    """Return validated JSON params for a SISU-spectrum evaluation run."""

    return SisuSpectrumEvaluationParams(
        schema_id=SISU_SPECTRUM_EVALUATION_PARAMS_SCHEMA,
        schema_version="v2",
        experiment=experiment,
        run_ids=[str(run_id) for run_id in run_ids],
        labels=[str(label) for label in labels],
        topic=topic,
        sisu_levels=[float(value) for value in sisu_levels],
        n_rollout_trials=n_rollout_trials,
        reference_samples=reference_samples,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
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

    SisuSpectrumAnalysisParams.model_validate(spec.params)
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
                variant="sisu_spectrum",
            ),
        },
        data=data,
    )


sisu_spectrum_recipe.EVAL_DEPENDENCIES = (SISU_SPECTRUM_EVALUATION_TYPE,)


def sisu_robustification_recipe(
    spec,
    _root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Build the grouped comparison from paired cached evaluation summaries."""

    params = SisuRobustificationAnalysisParams.model_validate(spec.params)
    run_summaries = paired_run_summaries(
        inputs,
        low_sisu=params.low_sisu,
        high_sisu=params.high_sisu,
    )
    return AnalysisRecipeResult(
        analyses={
            "robustification_comparison": SisuRobustificationAnalysis(
                variant="sisu_robustification_comparison",
            )
        },
        data=AnalysisInputData(
            models={},
            tasks={},
            states={"run_summaries": run_summaries},
            hps={"sisu_robustification": TreeNamespace(task=TreeNamespace(eval_n=len(inputs)))},
            extras={"params": params.model_dump(mode="json")},
        ),
    )


sisu_robustification_recipe.EVAL_DEPENDENCIES = ("rlrmp.eval.perturbation_response_bank",)


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
    x0 = make_cs_output_feedback_initial_state(reference.plant, config)
    covariances = default_cs_noise_covariances(reference.plant, config)
    ext_comparator = build_extlqg_comparator_path(
        reference.plant,
        reference.lqr_solution.K,
        covariances,
        schedule=reference.schedule,
        config=config,
    )
    ext_rollouts = [
        simulate_lqg_released_forward(
            reference.plant,
            ext_comparator.controller_gains,
            x0,
            draws=sample_forward_noise_draws(
                sample_key,
                T=reference.schedule.T,
                covariances=covariances,
            ),
            covariances=covariances,
            estimator_gains=ext_comparator.estimator_gains,
            config=config,
        )
        for sample_key in jr.split(key, n_samples)
    ]
    ext_x = np.stack([np.asarray(rollout.x, dtype=np.float64) for rollout in ext_rollouts], axis=0)
    vel_lo, _vel_hi = reference.plant.vel_slice
    ext_forward = ext_x[:, :, vel_lo]
    gamma_ref = reference.gamma_references[0]
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
        controller="analytical_hinf_output_feedback",
        gamma_factor=float(gamma_ref.factor),
        gamma=float(gamma_ref.gamma),
        n_samples=int(n_samples),
    )
    return (
        ReferenceCurve(
            label="extLQG analytical reference",
            time_s=np.arange(ext_forward.shape[1], dtype=np.float64) * float(reference.plant.dt),
            forward_velocity_m_s=np.mean(ext_forward, axis=0),
            std_forward_velocity_m_s=np.std(ext_forward, axis=0),
            controller="analytical_extlqg_output_feedback",
            gamma_factor=float(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR),
            n_samples=int(n_samples),
        ),
        hinf,
    )


def build_manifest(
    *,
    experiment: str,
    topic: str,
    profiles: Sequence[sisu_eval.RunSisuProfile],
    references: Sequence[ReferenceCurve],
    sisu_levels: Sequence[float] = DEFAULT_SISU_LEVELS,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
) -> dict[str, Any]:
    """Build the JSON summary manifest."""

    low_sisu_behavior = summarize_low_sisu_behavior(profiles)
    return {
        "schema_id": SISU_SPECTRUM_MANIFEST_SCHEMA,
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
        "output_contract": {
            "role": SISU_SPECTRUM_MANIFEST_ROLE,
            "shape": "structured_profile_and_reference_payload",
            "rendering_owner": "rlrmp.report.sisu_spectrum_figure_stage",
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


def profile_payload(profiles: Sequence[sisu_eval.RunSisuProfile]) -> list[dict[str, Any]]:
    """Return JSON-safe profile curves for downstream report/figure recipes."""

    return [
        {
            "run_id": profile.run_id,
            "label": profile.label,
            "curves": [
                {
                    "sisu": float(curve.sisu),
                    "time_s": np.asarray(curve.time_s).tolist(),
                    "mean_forward_velocity_m_s": np.asarray(
                        curve.mean_forward_velocity_m_s
                    ).tolist(),
                    "std_forward_velocity_m_s": np.asarray(curve.std_forward_velocity_m_s).tolist(),
                    "replicate_mean_forward_velocity_m_s": np.asarray(
                        curve.replicate_mean_forward_velocity_m_s
                    ).tolist(),
                    **curve_summary(curve),
                }
                for curve in profile.curves
            ],
        }
        for profile in profiles
    ]


def reference_payload(references: Sequence[ReferenceCurve]) -> list[dict[str, Any]]:
    """Return JSON-safe analytical reference curves for downstream rendering."""

    return [
        {
            "label": reference.label,
            "controller": reference.controller,
            "gamma_factor": reference.gamma_factor,
            "gamma": reference.gamma,
            "n_samples": reference.n_samples,
            "time_s": np.asarray(reference.time_s).tolist(),
            "forward_velocity_m_s": np.asarray(reference.forward_velocity_m_s).tolist(),
            "std_forward_velocity_m_s": np.asarray(reference.std_forward_velocity_m_s).tolist(),
        }
        for reference in references
    ]


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


def paired_run_summaries(
    inputs: Sequence[ResolvedAnalysisInput],
    *,
    low_sisu: float = 0.0,
    high_sisu: float = 1.0,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Pair cached perturbation summaries by run and declared SISU condition."""

    paired: dict[str, dict[str, Mapping[str, Any]]] = {}
    expected = {float(low_sisu): "sisu_0", float(high_sisu): "sisu_1"}
    for resolved in inputs:
        states = resolved.states
        if not isinstance(states, Mapping):
            raise ValueError("SISU comparison inputs require cached mapping states")
        level = float(states["sisu_level"])
        if level not in expected:
            raise ValueError(f"unexpected SISU comparison level {level:g}")
        runs = states.get("runs")
        if not isinstance(runs, Mapping):
            raise ValueError("SISU comparison states require a runs mapping")
        for run_id, summary in runs.items():
            if not isinstance(summary, Mapping):
                raise ValueError(f"SISU summary for {run_id!r} must be a mapping")
            paired.setdefault(str(run_id), {})[expected[level]] = summary
    for run_id, levels in paired.items():
        missing = sorted({"sisu_0", "sisu_1"}.difference(levels))
        if missing:
            raise ValueError(f"SISU comparison run {run_id!r} missing {missing}")
    return paired


def build_perturbation_comparison(
    run_summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the canonical grouped robustification comparison payload."""

    runs: dict[str, Any] = {}
    for run_id, levels in run_summaries.items():
        low = levels["sisu_0"]
        high = levels["sisu_1"]
        low_response = _response_summary(low)
        high_response = _response_summary(high)
        classes = compare_summary_groups(
            low_response["class_summary"]["groups"],
            high_response["class_summary"]["groups"],
        )
        timings = compare_summary_groups(
            low_response["timing_cell_summary"]["groups"],
            high_response["timing_cell_summary"]["groups"],
        )
        runs[run_id] = {
            "label": high.get("label", low.get("label", run_id)),
            "class_comparison": classes,
            "timing_cell_comparison": timings,
            "headline": summarize_headline(classes),
        }
    return {
        "schema_id": SISU_PERTURBATION_COMPARISON_SCHEMA,
        "sisu_levels": [0.0, 1.0],
        "comparison": "sisu_1_over_sisu_0",
        "runs": runs,
    }


def compare_summary_groups(
    low_groups: Mapping[str, Mapping[str, Any]],
    high_groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare SISU=1 and SISU=0 grouped perturbation summaries."""

    return {
        key: compare_group(key, low_groups.get(key, {}), high_groups.get(key, {}))
        for key in sorted(set(low_groups) | set(high_groups))
    }


def compare_group(
    group_key: str,
    low: Mapping[str, Any],
    high: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one perturbation class or timing-cell group."""

    low_metrics = _mapping(low.get("metrics"))
    high_metrics = _mapping(high.get("metrics"))
    metrics = {}
    for metric_key, slug, role in METRIC_SPECS:
        low_value = metric_mean(low_metrics, metric_key)
        high_value = metric_mean(high_metrics, metric_key)
        metrics[slug] = {
            "metric_key": metric_key,
            "role": role,
            "sisu_0": low_value,
            "sisu_1": high_value,
            "delta_1_minus_0": _delta(high_value, low_value),
            "ratio_1_over_0": _ratio(high_value, low_value),
        }
    return {
        "group": group_key,
        "rows_sisu_0": low.get("n_rows"),
        "rows_sisu_1": high.get("n_rows"),
        "status_counts_sisu_0": low.get("status_counts", {}),
        "status_counts_sisu_1": high.get("status_counts", {}),
        "metrics": metrics,
    }


def metric_mean(metrics: Mapping[str, Any], dotted_key: str) -> float | None:
    """Read a mean from flat or nested grouped-summary metrics."""

    direct = metrics.get(dotted_key)
    if isinstance(direct, Mapping) and direct.get("mean") is not None:
        return float(direct["mean"])
    current: Any = metrics
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if not isinstance(current, Mapping) or current.get("mean") is None:
        return None
    return float(current["mean"])


def summarize_headline(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize lower/equal/higher ratio counts for headline metrics."""

    return {
        key: _ratio_counts(comparisons, key)
        for key in (
            "mean_full_qrf_delta_cost",
            "max_delta_x_m",
            "mean_delta_action",
        )
    }


def _ratio_counts(
    comparisons: Mapping[str, Mapping[str, Any]],
    metric_slug: str,
) -> dict[str, int]:
    counts = {"improved": 0, "equal": 0, "worse": 0, "not_available": 0}
    for row in comparisons.values():
        ratio = _mapping(_mapping(row.get("metrics")).get(metric_slug)).get("ratio_1_over_0")
        if ratio is None or not np.isfinite(float(ratio)):
            counts["not_available"] += 1
        elif np.isclose(float(ratio), 1.0):
            counts["equal"] += 1
        elif float(ratio) < 1.0:
            counts["improved"] += 1
        else:
            counts["worse"] += 1
    return counts


def _response_summary(value: Mapping[str, Any]) -> Mapping[str, Any]:
    response = value.get("robust_response_summary", value)
    if not isinstance(response, Mapping):
        raise ValueError("cached SISU state has no robust response summary")
    return response


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _delta(high: float | None, low: float | None) -> float | None:
    if high is None or low is None:
        return None
    return float(high - low)


def _ratio(high: float | None, low: float | None) -> float | None:
    if high is None or low is None or abs(low) <= 1e-12:
        return None
    return float(high / low)


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


__all__ = [
    "CHECKPOINT_POLICY",
    "DEFAULT_N_ROLLOUT_TRIALS",
    "DEFAULT_REFERENCE_SAMPLES",
    "DEFAULT_SISU_LEVELS",
    "DEFAULT_TOPIC",
    "SISU_ROBUSTIFICATION_ANALYSIS_TYPE",
    "SISU_SPECTRUM_ANALYSIS_TYPE",
    "SISU_SPECTRUM_ANALYSIS_PARAMS_SCHEMA",
    "SISU_SPECTRUM_EVALUATION_TYPE",
    "SISU_SPECTRUM_EVALUATION_PARAMS_SCHEMA",
    "SISU_SPECTRUM_MANIFEST_ROLE",
    "SISU_SPECTRUM_MANIFEST_SCHEMA",
    "SISU_SPECTRUM_STATES_SCHEMA",
    "SisuRobustificationAnalysis",
    "SisuRobustificationAnalysisParams",
    "SisuSpectrumAnalysis",
    "SisuSpectrumAnalysisParams",
    "SisuSpectrumEvaluationParams",
    "ReferenceCurve",
    "analytical_reference_curves",
    "build_perturbation_comparison",
    "build_manifest",
    "compare_group",
    "compare_summary_groups",
    "curve_summary",
    "metric_mean",
    "paired_run_summaries",
    "profile_payload",
    "reference_payload",
    "register_sisu_spectrum_recipes",
    "robustification_comparison",
    "sisu_robustification_recipe",
    "sisu_spectrum_evaluation_recipe",
    "sisu_spectrum_evaluation_spec_params",
    "sisu_spectrum_recipe",
    "summarize_low_sisu_behavior",
    "summarize_headline",
]

"""Registered feedback-ablation evaluation and cached-state science."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    ResolvedAnalysisInput,
)
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.contracts.manifest import AnalysisRunSpec, EvaluationRunSpec, ParentRef
from feedbax.runtime.graph import Component, Wire
from jaxtyping import PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.analysis.math.summary_stats import summary_stats as _summary_stats
from rlrmp.eval.checkpoint_selection import (
    available_checkpoint_batches,
    checkpoint_path_for_batches,
    load_materialized_fixed_bank_manifest,
    load_validation_selected_checkpoint_model,
)
from rlrmp.eval.gru_diagnostics import RolloutEvaluation
from rlrmp.eval.model_slots import ModelSlotProjection
from rlrmp.eval.rollout_states import CachedEvaluationStates
from rlrmp.eval.perturbation_bank import (
    apply_perturbation_to_trial_specs,
    default_cs_perturbation_bank,
    delta_full_qrf_cost_summary,
    full_qrf_cost_summary,
)
from rlrmp.eval.replicates import is_replicate_array
from rlrmp.eval.trial_inputs import EvaluationRunInputs, repeat_single_validation_trial
from rlrmp.analysis.gru_standard_certificate import normalize_gru_hps
from rlrmp.model.feedback_descriptors import (
    COMPONENT_POSITION,
    COMPONENT_VELOCITY,
    resolve_controller_feedback_view,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.run_spec_access import require_run_seed
from rlrmp.runtime.spec_migrations import (
    FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
    stamp_current_schema,
)


FEEDBACK_ABLATION_ANALYSIS_TYPE = "rlrmp.feedback_ablation"
FEEDBACK_ABLATION_EVALUATION_TYPE = "rlrmp.eval.feedback_ablation"
SCHEMA_VERSION = "rlrmp.gru_feedback_ablation.v1"
FEEDBACK_SELECTION_SCHEMA_VERSION = "rlrmp.gru_feedback_checkpoint_selection_audit.v1"
FEEDBACK_AUDIT_SELECTION_ROLE = "audit_only_not_primary_selection"
DEFAULT_RESULT_EXPERIMENT = "57ab156"
DEFAULT_SCOPE = "fixed_target_random_perturb_validation_selected"
DEFAULT_RUN_IDS = (
    "fixed_target_random_perturb_fullqrf_warmcos__lr1e-3_clip5_b64",
    "fixed_target_random_perturb_fullqrf_warmcos__lr3e-3_clip5_b64",
)
DEFAULT_OUTPUT_FILENAME = (
    "gru_feedback_ablation_fixed_target_random_perturb_validation_selected_manifest.json"
)

AblationMode = Literal[
    "normal",
    "frozen_nominal_observation_tape",
    "zeroed_perturbation_observation_deviation",
    "shuffled_observation_history",
    "lagged_observation_history",
    "position_only_observation",
    "velocity_only_observation",
]

EvaluationBin = Literal[
    "nominal",
    "initial_state",
    "process_epsilon",
    "sensory_feedback",
    "delayed_observation",
]

OBSERVATION_ABLATION_INPUT_PREFIX = "feedback_ablation"
OBSERVATION_DIM = 4
NOMINAL_ENDPOINT_WARN_M = 0.08
NOMINAL_TERMINAL_SPEED_WARN_M_S = 0.35
COMMAND_RATIO_WARN = 2.0
COMMAND_RATIO_FAIL = 4.0
DEFAULT_ABLATION_MODES: tuple[AblationMode, ...] = (
    "normal",
    "frozen_nominal_observation_tape",
    "zeroed_perturbation_observation_deviation",
    "shuffled_observation_history",
    "lagged_observation_history",
    "position_only_observation",
    "velocity_only_observation",
)
DEFAULT_EVALUATION_BINS: dict[EvaluationBin, str | None] = {
    "nominal": None,
    "initial_state": "initial_position_offset__x_pos",
    "process_epsilon": "process_epsilon_pulse__force_state_x__mid_t15_pos",
    "sensory_feedback": "sensory_feedback_offset__position__mid_visible_t20_x_pos",
    "delayed_observation": "delayed_observation_offset__position__mid_visible_t20_x_pos",
}


@dataclass(frozen=True)
class DetailedRolloutEvaluation:
    """Rollout product plus arrays used by feedback-ablation reducers."""

    rollout: CachedEvaluationStates
    feedback: Any
    mechanics_vector: Any


def evaluate_model_on_trial_specs(
    *, model: Any, task: Any, trial_specs: Any, n_replicates: int, seed: int
) -> DetailedRolloutEvaluation:
    """Execute one feedback-ablation evaluation at the eval-layer boundary."""

    model_arrays, model_other = eqx.partition(
        model, lambda leaf: is_replicate_array(leaf, n_replicates)
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, _infer_batch_size(trial_specs)),
        )

    if _has_replicate_specific_trial_inputs(trial_specs, n_replicates):
        keys = jr.split(jr.PRNGKey(seed), n_replicates)
        states_by_replicate = []
        for replicate in range(n_replicates):
            replicate_model = eqx.combine(
                _select_replicate_tree(model_arrays, replicate, n_replicates), model_other
            )
            replicate_trial_specs = _select_replicate_trial_inputs(
                trial_specs, replicate, n_replicates
            )
            states_by_replicate.append(
                task.eval_trials(
                    replicate_model,
                    replicate_trial_specs,
                    jr.split(keys[replicate], _infer_batch_size(replicate_trial_specs)),
                )
            )
        states = jt.map(lambda *xs: jnp.stack(xs, axis=0), *states_by_replicate)
    else:
        states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
            model_arrays, jr.split(jr.PRNGKey(seed), n_replicates)
        )
    rollout = CachedEvaluationStates.from_states(
        states,
        trial_specs,
        dt=0.01,
        include_mechanics_vector=True,
        include_feedback=True,
    )
    return DetailedRolloutEvaluation(
        rollout=rollout,
        feedback=rollout.feedback,
        mechanics_vector=rollout.mechanics_vector,
    )


def evaluate_feedback_ablation_runs(
    params: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Execute spec-declared feedback ablations for registered evaluation."""

    from rlrmp.eval.trial_inputs import resolve_evaluation_run_inputs

    source_experiment = str(params["source_experiment"])
    run_ids = tuple(str(run_id) for run_id in params["run_ids"])
    labels_value = params.get("labels")
    labels = None if labels_value is None else tuple(str(label) for label in labels_value)
    bank_value = params.get("bank")
    bank = (
        dict(bank_value)
        if isinstance(bank_value, Mapping)
        else default_cs_perturbation_bank(
            mode=str(params["bank_mode"]),
            calibration_level=params.get("calibration_level"),
            calibration_reach=params.get("calibration_reach"),
            feedback_scale_manifest_path=_optional_repo_path(
                params.get("feedback_scale_manifest_path"), repo_root=repo_root
            ),
        )
    )
    bins_value = params.get("evaluation_bins")
    evaluation_bins = (
        {str(key): value for key, value in bins_value.items()}
        if isinstance(bins_value, Mapping)
        else selected_feedback_ablation_bins_for_bank(
            bank, preferred_level=str(params["feedback_selection_level"])
        )
    )
    run_inputs = resolve_evaluation_run_inputs(
        experiment=source_experiment,
        run_ids=run_ids,
        labels=labels,
        repo_root=repo_root,
    )
    runs = {
        run.run_id: _execute_feedback_ablation_run(
            run,
            source_experiment=source_experiment,
            n_rollout_trials=int(params["n_rollout_trials"]),
            include_checkpoint_rescore=bool(params["include_checkpoint_rescore"]),
            bank=bank,
            evaluation_bins=evaluation_bins,
            preferred_checkpoint_manifest_path=_optional_repo_path(
                params.get("preferred_checkpoint_manifest_path"), repo_root=repo_root
            ),
            repo_root=repo_root,
        )
        for run in run_inputs
    }
    return {
        "source_experiment": source_experiment,
        "run_ids": list(run_ids),
        "labels": None if labels is None else list(labels),
        "scope": str(params["scope"]),
        "bank_mode": str(params["bank_mode"]),
        "bank": {
            "bank_id": bank.get("bank_id"),
            "calibration_metadata_hooks": bank.get("calibration_metadata_hooks"),
            "n_perturbations": len(bank.get("perturbations", ())),
        },
        "evaluation_bins": evaluation_bins,
        "ablation_modes": list(default_ablation_modes()),
        "runs": runs,
    }


def evaluate_projected_feedback_ablation_run(
    projection: ModelSlotProjection,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute feedback ablation from one exact native model projection."""

    bank_value = params.get("bank")
    bank = (
        dict(bank_value)
        if isinstance(bank_value, Mapping)
        else default_cs_perturbation_bank(
            mode=str(params["bank_mode"]),
            calibration_level=params.get("calibration_level"),
            calibration_reach=params.get("calibration_reach"),
            feedback_scale_manifest_path=None,
        )
    )
    bins_value = params.get("evaluation_bins")
    evaluation_bins = (
        {str(key): value for key, value in bins_value.items()}
        if isinstance(bins_value, Mapping)
        else selected_feedback_ablation_bins_for_bank(
            bank, preferred_level=str(params["feedback_selection_level"])
        )
    )
    labels = params.get("labels")
    label = (
        str(labels[0]) if isinstance(labels, Sequence) and labels else projection.provenance.run_id
    )
    run = SimpleNamespace(label=label)
    result = _execute_feedback_ablation_run(
        run,
        source_experiment=None,
        n_rollout_trials=int(params["n_rollout_trials"]),
        include_checkpoint_rescore=bool(params["include_checkpoint_rescore"]),
        bank=bank,
        evaluation_bins=evaluation_bins,
        model_projection=projection,
    )
    return {
        "source": "exact_training_manifest_parent",
        "run_ids": [projection.provenance.training_manifest_id],
        "labels": [label],
        "scope": str(params["scope"]),
        "bank_mode": str(params["bank_mode"]),
        "bank": {
            "bank_id": bank.get("bank_id"),
            "calibration_metadata_hooks": bank.get("calibration_metadata_hooks"),
            "n_perturbations": len(bank.get("perturbations", ())),
        },
        "evaluation_bins": evaluation_bins,
        "ablation_modes": list(default_ablation_modes()),
        "runs": {projection.provenance.training_manifest_id: result},
    }


def _required_source_experiment(params: FeedbackAblationAnalysisParams) -> str:
    source = params.source_experiment or params.experiment
    if source is None:
        raise ValueError("feedback-ablation analysis requires source_experiment input")
    return source


def _optional_repo_path(value: Any, *, repo_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo_root / path


class FeedbackAblationAnalysisParams(BaseModel):
    """Params for the registered feedback-ablation scoring analysis."""

    model_config = ConfigDict(extra="allow")

    schema_id: str | None = None
    schema_version: str | None = SCHEMA_VERSION
    experiment: str | None = None
    scope: str = DEFAULT_SCOPE
    source_experiment: str | None = None
    result_experiment: str | None = None
    output_tag: str = "feedback_ablation"
    run_ids: list[str] = Field(default_factory=list)
    labels: list[str] | None = None
    ablation_modes: list[AblationMode] = Field(default_factory=lambda: list(DEFAULT_ABLATION_MODES))
    evaluation_bins: dict[EvaluationBin, str | None] = Field(
        default_factory=lambda: dict(DEFAULT_EVALUATION_BINS)
    )
    bank: dict[str, Any] | None = None
    bank_mode: str = "params"
    feedback_selection_level: str = "small"
    checkpoint_policy: str = "evaluation_manifest"
    include_checkpoint_rescore: bool = False


class FeedbackAblationAnalysis(AbstractAnalysis):
    """Aggregate cached feedback-ablation evaluation states into a custody payload."""

    params: dict[str, Any] = eqx.field(default_factory=dict, static=True)
    evaluation_refs: tuple[dict[str, Any], ...] = eqx.field(default_factory=tuple, static=True)
    analysis_type: str = eqx.field(default=FEEDBACK_ABLATION_ANALYSIS_TYPE, static=True)
    artifact_role: str = eqx.field(
        default="rlrmp-gru-feedback-ablation-manifest",
        static=True,
    )
    logical_name: str = eqx.field(
        default="feedback_ablation/feedback_ablation_manifest.json",
        static=True,
    )
    report_role: str = eqx.field(
        default="rlrmp-gru-feedback-ablation-report-render",
        static=True,
    )
    report_logical_name: str = eqx.field(
        default="feedback_ablation/feedback_ablation_report.json",
        static=True,
    )

    def compute(self, data: AnalysisInputData, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        params = FeedbackAblationAnalysisParams.model_validate(self.params)
        runs = _feedback_ablation_runs_from_states(data.states, params=params)
        manifest = {
            "schema_version": params.schema_version or SCHEMA_VERSION,
            "analysis_type": self.analysis_type,
            "issue": params.result_experiment or params.experiment or DEFAULT_RESULT_EXPERIMENT,
            "source_experiment": _required_source_experiment(params),
            "scope": params.scope,
            "checkpoint_policy": params.checkpoint_policy,
            "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
            "ablation_modes": list(params.ablation_modes),
            "evaluation_bins": dict(params.evaluation_bins),
            "bank_mode": params.bank_mode,
            "bank": params.bank,
            "feedback_selection_level": params.feedback_selection_level,
            "evaluation_manifest_dependencies": list(self.evaluation_refs),
            "runs": runs,
        }
        manifest["status_counts"] = _manifest_status_counts(runs)
        manifest["feedback_checkpoint_selection_audit"] = feedback_checkpoint_selection_audit(
            manifest
        )
        return manifest

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        del data, kwargs
        context.record_json_artifact(
            dict(result),
            role=self.artifact_role,
            logical_name=self.logical_name,
            metadata={
                "schema_boundary": "rlrmp-owned feedback-ablation scoring payload",
                "analysis_type": self.analysis_type,
                "evaluation_manifest_dependencies": list(self.evaluation_refs),
            },
        )
        markdown = render_feedback_ablation_markdown(result)
        context.record_json_artifact(
            {
                "schema_version": "rlrmp.gru_feedback_ablation_report.v1",
                "analysis_type": self.analysis_type,
                "markdown": markdown,
            },
            role=self.report_role,
            logical_name=self.report_logical_name,
            metadata={"rendered_from": self.artifact_role},
        )
        return result


def feedback_ablation_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Build the registered feedback-ablation analysis from evaluation manifests."""

    params = FeedbackAblationAnalysisParams.model_validate(spec.params)
    evaluation_inputs = [
        resolved
        for resolved in inputs
        if resolved.ref.kind == "EvaluationRunManifest" or resolved.states is not None
    ]
    if not evaluation_inputs:
        raise ValueError("feedback-ablation analysis requires EvaluationRunManifest input")
    params_payload = params.model_dump(mode="json", exclude_none=True)
    first_evaluation_states = evaluation_inputs[0].states
    if isinstance(first_evaluation_states, Mapping):
        for key in (
            "source_experiment",
            "scope",
            "run_ids",
            "labels",
            "ablation_modes",
            "evaluation_bins",
            "bank_mode",
            "bank",
        ):
            value = first_evaluation_states.get(key)
            if value is not None:
                params_payload[key] = value
    params = FeedbackAblationAnalysisParams.model_validate(params_payload)
    states = {
        "evaluations": [
            {
                "ref": _parent_ref_json(resolved.ref),
                "manifest_id": None if resolved.manifest is None else resolved.manifest.id,
                "states": resolved.states or {},
            }
            for resolved in evaluation_inputs
        ]
    }
    evaluation_refs = tuple(
        {
            "kind": resolved.ref.kind,
            "id": resolved.ref.id,
            "role": resolved.ref.role,
            "uri": resolved.ref.uri,
            "manifest_id": None if resolved.manifest is None else resolved.manifest.id,
        }
        for resolved in evaluation_inputs
    )
    analysis = FeedbackAblationAnalysis(
        variant="feedback_ablation",
        params=params.model_dump(mode="json", exclude_none=True),
        evaluation_refs=evaluation_refs,
        analysis_type=spec.analysis_type,
        **_feedback_ablation_artifact_roles(spec.analysis_type),
    )
    data = AnalysisInputData(
        models={},
        tasks={},
        states=states,
        hps={"feedback_ablation": TreeNamespace(n_evaluations=len(evaluation_inputs))},
        extras={"params": params.model_dump(mode="json", exclude_none=True)},
    )
    return AnalysisRecipeResult(analyses={"feedback_ablation": analysis}, data=data)


def feedback_ablation_spec(
    *,
    evaluation_manifest_id: str,
    evaluation_manifest_uri: Path | str | None = None,
    params: Mapping[str, Any] | None = None,
) -> AnalysisRunSpec:
    """Return an analysis spec for cached feedback-ablation evaluation states."""

    ref = ParentRef(
        kind="EvaluationRunManifest",
        id=evaluation_manifest_id,
        role="evaluation_run",
        uri=None if evaluation_manifest_uri is None else str(evaluation_manifest_uri),
    )
    return AnalysisRunSpec(
        analysis_type=FEEDBACK_ABLATION_ANALYSIS_TYPE,
        inputs=[ref],
        params=dict(params or {}),
    )


def feedback_ablation_evaluation_spec(
    *,
    training_run_ref: ParentRef | None = None,
    checkpoint_custody_root: Path | str | None = None,
    source_experiment: str | None = None,
    run_ids: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    scope: str = DEFAULT_SCOPE,
    n_rollout_trials: int = 4,
    include_checkpoint_rescore: bool = True,
    bank_mode: str = "raw",
    calibration_level: str | Sequence[str] | None = None,
    calibration_reach: str | float | None = None,
    feedback_selection_level: str = "small",
    feedback_scale_manifest_path: Path | str | None = None,
    preferred_checkpoint_manifest_path: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
    bank: Mapping[str, Any] | None = None,
    evaluation_bins: Mapping[str, str | None] | None = None,
) -> EvaluationRunSpec:
    """Return one unambiguous native or legacy feedback-ablation evaluation spec."""

    has_native_authority = training_run_ref is not None
    has_legacy_authority = source_experiment is not None or run_ids is not None
    if has_native_authority and has_legacy_authority:
        raise ValueError(
            "feedback-ablation spec cannot mix training_run_ref with legacy "
            "source_experiment/run_ids selectors"
        )
    if has_native_authority:
        if (
            training_run_ref.kind != "TrainingRunManifest"
            or training_run_ref.role != "training_run"
        ):
            raise ValueError(
                "native feedback-ablation training_run_ref must be a "
                "TrainingRunManifest/training_run ParentRef"
            )
        if checkpoint_custody_root is None:
            raise ValueError("native feedback-ablation spec requires checkpoint_custody_root")
        if preferred_checkpoint_manifest_path is not None:
            raise ValueError(
                "native feedback-ablation spec cannot declare legacy "
                "preferred_checkpoint_manifest_path authority"
            )
        custody_root = Path(checkpoint_custody_root).expanduser()
        if not custody_root.is_absolute():
            raise ValueError("native feedback-ablation checkpoint_custody_root must be absolute")
        selected_run_ids: list[str] = []
        inputs = [training_run_ref]
    else:
        if checkpoint_custody_root is not None:
            raise ValueError("legacy feedback-ablation spec cannot declare checkpoint_custody_root")
        selected_run_ids = [str(run_id) for run_id in (run_ids or DEFAULT_RUN_IDS)]
        if not source_experiment or not selected_run_ids:
            raise ValueError(
                "legacy feedback-ablation spec requires source_experiment and non-empty run_ids"
            )
        inputs = []

    params = {
        "labels": None if labels is None else [str(label) for label in labels],
        "scope": scope,
        "n_rollout_trials": n_rollout_trials,
        "include_checkpoint_rescore": include_checkpoint_rescore,
        "bank_mode": bank_mode,
        "calibration_level": calibration_level,
        "calibration_reach": calibration_reach,
        "feedback_selection_level": feedback_selection_level,
        "feedback_scale_manifest_path": _optional_spec_path(
            feedback_scale_manifest_path,
            repo_root=repo_root,
        ),
        "preferred_checkpoint_manifest_path": _optional_spec_path(
            preferred_checkpoint_manifest_path,
            repo_root=repo_root,
        ),
        "repo_root": str(repo_root),
        "bank": None if bank is None else dict(bank),
        "evaluation_bins": (
            None
            if evaluation_bins is None
            else {str(key): value for key, value in evaluation_bins.items()}
        ),
    }
    if has_native_authority:
        params["checkpoint_custody_root"] = str(custody_root)
    else:
        params["source_experiment"] = source_experiment
        params["run_ids"] = selected_run_ids
    return EvaluationRunSpec(
        evaluation_type=FEEDBACK_ABLATION_EVALUATION_TYPE,
        training_run_ids=selected_run_ids,
        inputs=inputs,
        params=stamp_current_schema(FEEDBACK_ABLATION_EVAL_PARAMS_KIND, params),
    )


def _optional_spec_path(value: Path | str | None, *, repo_root: Path) -> str | None:
    if value is None:
        return None
    return _repo_relative(_absolute_repo_path(value, repo_root=repo_root), repo_root=repo_root)


def _absolute_repo_path(value: Path | str, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


feedback_ablation_recipe.EVAL_DEPENDENCIES = ("rlrmp.eval.feedback_ablation",)


def _feedback_ablation_artifact_roles(analysis_type: str) -> dict[str, str]:
    if analysis_type == "rlrmp.feedback_quality.feedback_ablation":
        return {
            "artifact_role": "rlrmp-feedback-quality-feedback-ablation-status",
            "logical_name": "feedback_quality/feedback_ablation_status.json",
            "report_role": "rlrmp-feedback-quality-feedback-ablation-note",
            "report_logical_name": "feedback_quality/feedback_ablation_report.json",
        }
    return {
        "artifact_role": "rlrmp-gru-feedback-ablation-manifest",
        "logical_name": "feedback_ablation/feedback_ablation_manifest.json",
        "report_role": "rlrmp-gru-feedback-ablation-report-render",
        "report_logical_name": "feedback_ablation/feedback_ablation_report.json",
    }


@dataclass(frozen=True)
class ObservationAblationSpec:
    """Graph-level delayed-feedback ablation configuration."""

    mode: AblationMode
    label: str
    input_key: str | None = None
    mask: tuple[float, ...] | None = None

    @property
    def requires_payload(self) -> bool:
        """Return whether this ablation consumes an external observation tape."""

        return self.input_key is not None

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable ablation provenance."""

        return {
            "mode": self.mode,
            "label": self.label,
            "input_key": self.input_key,
            "mask": None if self.mask is None else list(self.mask),
            "intervention_point": "sensory.output -> net.feedback",
            "controller_weights_mutated": False,
            "checkpoint_selection_role": "selected_checkpoint_per_replicate",
            "analytical_metrics_role": "audit_only_not_used_for_checkpoint_selection",
        }


class ObservationOverride(Component):
    """Replace a delayed-feedback graph signal with an external tape."""

    input_ports = ("signal", "replacement")
    output_ports = ("feedback",)

    label: str = eqx.field(static=True)

    def __init__(self, *, label: str):
        self.label = str(label)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        return {"feedback": inputs["replacement"]}, state


class ObservationMask(Component):
    """Apply a fixed mask to the delayed-feedback graph signal."""

    input_ports = ("signal",)
    output_ports = ("feedback",)

    label: str = eqx.field(static=True)
    mask: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, *, label: str, mask: Sequence[float]):
        self.label = str(label)
        self.mask = tuple(float(value) for value in mask)
        if len(self.mask) != OBSERVATION_DIM:
            raise ValueError(f"observation mask must have {OBSERVATION_DIM} entries")

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        signal = jnp.asarray(inputs["signal"])
        mask = jnp.asarray(self.mask, dtype=signal.dtype)
        if mask.shape[-1] < signal.shape[-1]:
            mask = jnp.pad(mask, [(0, signal.shape[-1] - mask.shape[-1])])
        elif mask.shape[-1] > signal.shape[-1]:
            mask = mask[: signal.shape[-1]]
        return {
            "feedback": signal * mask,
        }, state


def default_ablation_modes() -> tuple[AblationMode, ...]:
    """Return the standard feedback-ablation mode order."""

    return tuple(FeedbackAblationAnalysisParams().ablation_modes)


def selected_feedback_ablation_bins() -> dict[EvaluationBin, str | None]:
    """Return representative perturbation rows for the standard evaluation bins."""

    return dict(FeedbackAblationAnalysisParams().evaluation_bins)


def _parent_ref_json(ref: ParentRef) -> dict[str, Any]:
    return ref.model_dump(mode="json", exclude_none=True)


def _feedback_ablation_runs_from_states(
    states: Any,
    *,
    params: FeedbackAblationAnalysisParams,
) -> dict[str, Any]:
    """Normalize cached evaluation states into the historical manifest run map."""

    runs: dict[str, Any] = {}
    evaluations = []
    if isinstance(states, Mapping):
        raw_evaluations = states.get("evaluations")
        if isinstance(raw_evaluations, Sequence) and not isinstance(raw_evaluations, (str, bytes)):
            evaluations = list(raw_evaluations)
        else:
            evaluations = [{"states": states, "ref": None, "manifest_id": None}]
    for index, item in enumerate(evaluations):
        if not isinstance(item, Mapping):
            continue
        eval_states = item.get("states")
        if not isinstance(eval_states, Mapping):
            eval_states = {}
        ref = item.get("ref") if isinstance(item.get("ref"), Mapping) else {}
        evaluated_runs = eval_states.get("runs")
        if isinstance(evaluated_runs, Mapping):
            for run_id, run_states in evaluated_runs.items():
                if not isinstance(run_states, Mapping):
                    continue
                runs[str(run_id)] = _feedback_ablation_run_payload(
                    run_states,
                    ref=ref,
                    manifest_id=item.get("manifest_id"),
                    params=params,
                )
            continue
        run_id = _feedback_ablation_run_id(
            eval_states,
            ref=ref,
            index=index,
            params=params,
        )
        run_payload = _feedback_ablation_run_payload(
            eval_states,
            ref=ref,
            manifest_id=item.get("manifest_id"),
            params=params,
        )
        runs[run_id] = run_payload
    return runs


def _feedback_ablation_run_id(
    states: Mapping[str, Any],
    *,
    ref: Mapping[str, Any],
    index: int,
    params: FeedbackAblationAnalysisParams,
) -> str:
    if index < len(params.run_ids):
        return str(params.run_ids[index])
    metadata = ref.get("metadata") if isinstance(ref.get("metadata"), Mapping) else {}
    for value in (
        states.get("run_id"),
        states.get("training_run_id"),
        metadata.get("run_id"),
        ref.get("id"),
    ):
        if value:
            return str(value)
    return f"feedback_ablation_eval_{index}"


def _feedback_ablation_run_payload(
    states: Mapping[str, Any],
    *,
    ref: Mapping[str, Any],
    manifest_id: Any,
    params: FeedbackAblationAnalysisParams,
) -> dict[str, Any]:
    rows = _feedback_ablation_rows_from_states(states, params=params)
    checkpoint_rescore = states.get("feedback_checkpoint_rescore")
    if not isinstance(checkpoint_rescore, Mapping):
        checkpoint_rescore = {
            "status": "skipped",
            "reason": "not_present_in_evaluation_manifest",
            "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
        }
    label = states.get("label")
    metadata = ref.get("metadata") if isinstance(ref.get("metadata"), Mapping) else {}
    if label is None:
        label = metadata.get("label") or ref.get("id") or manifest_id
    return {
        "label": None if label is None else str(label),
        "evaluation_manifest_id": manifest_id or ref.get("id"),
        "evaluation_manifest_ref": dict(ref),
        "n_replicates": states.get("n_replicates"),
        "n_rollout_trials_per_replicate": states.get("n_rollout_trials_per_replicate"),
        "n_time_steps": states.get("n_time_steps"),
        "dt_s": states.get("dt_s"),
        "status_counts": _status_counts(rows),
        "interpretation": interpret_run_feedback_ablation(rows),
        "normalized_feedback_use": summarize_normalized_feedback_use(rows),
        "feedback_pass_audit": summarize_feedback_pass_audit(rows),
        "feedback_checkpoint_rescore": dict(checkpoint_rescore),
        "ablation_modes": list(params.ablation_modes),
        "evaluation_bins": dict(params.evaluation_bins),
        "ablations": rows,
    }


def _feedback_ablation_rows_from_states(
    states: Mapping[str, Any],
    *,
    params: FeedbackAblationAnalysisParams,
) -> list[dict[str, Any]]:
    rows = states.get("ablations", states.get("rows"))
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    rollout_pairs = states.get("rollout_pairs")
    if isinstance(rollout_pairs, Sequence) and not isinstance(rollout_pairs, (str, bytes)):
        return [_feedback_ablation_row_from_pair(pair, params=params) for pair in rollout_pairs]
    return []


def _feedback_ablation_row_from_pair(
    pair: Any,
    *,
    params: FeedbackAblationAnalysisParams,
) -> dict[str, Any]:
    if isinstance(pair, Mapping):
        row = dict(pair)
    else:
        row = {"payload": pair}
    if "bin" not in row:
        row["bin"] = (
            row["evaluation_bin"] if "evaluation_bin" in row else next(iter(params.evaluation_bins))
        )
    if "mode" not in row:
        row["mode"] = row["ablation_mode"] if "ablation_mode" in row else params.ablation_modes[0]
    if "status" not in row:
        row["status"] = "available" if row.get("metrics") else "not_available"
    row.setdefault("metrics", row.get("metrics", {}))
    return row


def _manifest_status_counts(runs: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs.values():
        if not isinstance(run, Mapping):
            continue
        for status, count in dict(run.get("status_counts", {})).items():
            counts[str(status)] = counts.get(str(status), 0) + int(count)
    return counts


def selected_feedback_ablation_bins_for_bank(
    bank: Mapping[str, Any],
    *,
    preferred_level: str = "small",
) -> dict[EvaluationBin, str | None]:
    """Return representative feedback bins for the supplied perturbation bank."""

    rows = {
        str(row["perturbation_id"]): row
        for row in bank.get("perturbations", ())
        if isinstance(row, Mapping)
    }
    if str(bank.get("bank_id", "")).startswith("cs_standard_perturbation_response"):
        return {
            bin_id: perturbation_id if perturbation_id in rows else None
            for bin_id, perturbation_id in selected_feedback_ablation_bins().items()
        }
    return {
        "nominal": None,
        "initial_state": _select_representative_perturbation_id(
            rows,
            family="initial_position_offset",
            level_name=preferred_level,
            timing_bin="initial_condition",
        ),
        "process_epsilon": _select_representative_perturbation_id(
            rows,
            family="process_epsilon_force_state_xy",
            level_name=preferred_level,
            timing_bin="mid",
        ),
        "sensory_feedback": _select_representative_perturbation_id(
            rows,
            family="sensory_feedback_offset",
            level_name=preferred_level,
            timing_bin="mid_visible",
            units="m",
        ),
        "delayed_observation": _select_representative_perturbation_id(
            rows,
            family="delayed_observation_offset",
            level_name=preferred_level,
            timing_bin="mid_visible",
            units="m",
        ),
    }


def _select_representative_perturbation_id(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    family: str,
    level_name: str,
    timing_bin: str,
    units: str | None = None,
) -> str | None:
    """Select a deterministic x/positive row for one calibrated feedback bin."""

    candidates = [
        row
        for row in rows.values()
        if str(row.get("family")) == family
        and str(row.get("timing_bin")) == timing_bin
        and (units is None or str(row.get("units")) == units)
        and (
            "level_name" not in row
            or row.get("level_name") is None
            or str(row.get("level_name")) == level_name
        )
    ]
    if not candidates:
        return None

    def sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
        axis = str(row.get("axis", ""))
        sign = int(row.get("sign", 0) or 0)
        axis_priority = 0 if axis == "x" else 1 if axis in {"vx", "y", "vy"} else 2
        sign_priority = 0 if sign > 0 else 1
        return axis_priority, sign_priority, str(row.get("perturbation_id", ""))

    return str(min(candidates, key=sort_key)["perturbation_id"])


def build_observation_tape(
    mode: AblationMode,
    *,
    bin_feedback: Any,
    nominal_feedback: Any,
) -> Any | None:
    """Return the external delayed-feedback tape for a payload-based ablation."""

    feedback = jnp.asarray(bin_feedback, dtype=jnp.float64)
    nominal = jnp.asarray(nominal_feedback, dtype=jnp.float64)
    if feedback.shape != nominal.shape:
        raise ValueError(
            "bin_feedback and nominal_feedback must have the same shape; "
            f"got {feedback.shape} and {nominal.shape}"
        )
    if mode == "normal":
        return None
    if mode in {
        "frozen_nominal_observation_tape",
        "zeroed_perturbation_observation_deviation",
    }:
        return nominal
    if mode == "shuffled_observation_history":
        if feedback.shape[1] < 2:
            return jnp.array(feedback, copy=True)
        return jnp.roll(feedback, shift=1, axis=1)
    if mode == "lagged_observation_history":
        first = feedback[:, :, :1, :]
        return jnp.concatenate([first, feedback[:, :, :-1, :]], axis=2)
    if mode in {"position_only_observation", "velocity_only_observation"}:
        return None
    raise ValueError(f"unsupported feedback ablation mode {mode!r}")


def build_observation_ablation_spec(
    mode: AblationMode,
    *,
    bin_id: str,
) -> ObservationAblationSpec:
    """Return graph-adapter metadata for one ablation/bin pair."""

    safe_bin = bin_id.replace("/", "_").replace(":", "_")
    label = f"feedback_ablation_{mode}_{safe_bin}"
    if mode == "position_only_observation":
        return ObservationAblationSpec(
            mode=mode,
            label=label,
            mask=_feedback_component_mask(COMPONENT_POSITION),
        )
    if mode == "velocity_only_observation":
        return ObservationAblationSpec(
            mode=mode,
            label=label,
            mask=_feedback_component_mask(COMPONENT_VELOCITY),
        )
    if mode == "normal":
        return ObservationAblationSpec(mode=mode, label=label)
    return ObservationAblationSpec(
        mode=mode,
        label=label,
        input_key=f"{OBSERVATION_ABLATION_INPUT_PREFIX}:{mode}:{safe_bin}",
    )


def _feedback_component_mask(
    component_id: str, *, feedback_dim: int = OBSERVATION_DIM
) -> tuple[float, ...]:
    descriptor_view = resolve_controller_feedback_view(
        None,
        feedback_dim=feedback_dim,
        source="gru_feedback_ablation_mask",
    )
    component = descriptor_view.component(component_id)
    mask = [0.0] * feedback_dim
    for index in range(component.slice.start, component.slice.stop, component.slice.step):
        mask[index] = 1.0
    return tuple(mask)


def insert_observation_ablation(model: Any, spec: ObservationAblationSpec) -> Any:
    """Insert a feedback-ablation node on ``sensory.output -> net.feedback``."""

    if spec.mode == "normal":
        return model
    if spec.label in getattr(model, "nodes", {}):
        return model
    old_wire = Wire("sensory", "output", "net", "feedback")
    graph = model.remove_wire(old_wire)
    node: Component
    if spec.requires_payload:
        node = ObservationOverride(label=spec.label)
    else:
        if spec.mask is None:
            raise ValueError(f"ablation mode {spec.mode!r} has no payload or mask")
        node = ObservationMask(label=spec.label, mask=spec.mask)
    graph = graph.add_node(spec.label, node)
    graph = graph.add_wire(Wire("sensory", "output", spec.label, "signal"))
    graph = graph.add_wire(Wire(spec.label, "feedback", "net", "feedback"))
    if spec.requires_payload:
        graph = eqx.tree_at(
            lambda g: g.input_ports,
            graph,
            (*graph.input_ports, spec.input_key),
        )
        graph = eqx.tree_at(
            lambda g: g.input_bindings,
            graph,
            {**graph.input_bindings, spec.input_key: (spec.label, "replacement")},
        )
    return graph


def summarize_ablation_delta(
    *,
    baseline: DetailedRolloutEvaluation,
    ablated: DetailedRolloutEvaluation,
    baseline_cost: Mapping[str, Any],
    ablated_cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Return compact per-ablation deltas against a normal-observation baseline."""

    return {
        "baseline_action_norm": _summary_stats(np.linalg.norm(baseline.rollout.command, axis=-1)),
        "ablated_action_norm": _summary_stats(np.linalg.norm(ablated.rollout.command, axis=-1)),
        "delta_action_norm": _summary_stats(
            np.linalg.norm(ablated.rollout.command - baseline.rollout.command, axis=-1)
        ),
        "delta_observation_norm": _summary_stats(
            np.linalg.norm(ablated.feedback - baseline.feedback, axis=-1)
        ),
        "delta_endpoint_error_m": _summary_stats(
            _endpoint_error(ablated.rollout) - _endpoint_error(baseline.rollout)
        ),
        "delta_terminal_speed_m_s": _summary_stats(
            _terminal_speed(ablated.rollout) - _terminal_speed(baseline.rollout)
        ),
        "baseline_endpoint_error_m": _summary_stats(_endpoint_error(baseline.rollout)),
        "baseline_terminal_speed_m_s": _summary_stats(_terminal_speed(baseline.rollout)),
        "baseline_full_qrf_cost": baseline_cost,
        "ablated_full_qrf_cost": ablated_cost,
        "rollout_full_qrf": delta_full_qrf_cost_summary(baseline_cost, ablated_cost),
        "validation_objective": {
            "status": "substitute",
            "substitute": "paired_realized_deterministic_rollout_full_qrf.total",
            "reason": (
                "checkpoint validation objective is a selection scalar in the saved "
                "manifest; post-hoc ablation rollouts are scored with the realized "
                "full-Q/R/Q_f substitute on the same rollout bank"
            ),
        },
    }


def summarize_normalized_feedback_use(
    rows: Sequence[Mapping[str, Any]],
    *,
    open_loop_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return normalized feedback-use indices from evaluated ablation rows.

    The returned indices are audit metrics. Missing denominators and unavailable
    open-loop references are represented explicitly rather than silently
    converted to zero.
    """

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    warnings: list[str] = []
    ablation_dependence = _ablation_dependence_index(evaluated, warnings=warnings)
    perturbation_rescue = _perturbation_rescue_index(evaluated, warnings=warnings)
    correction = _correction_index_vs_open_loop(
        evaluated,
        open_loop_reference=open_loop_reference,
        warnings=warnings,
    )
    available_scores = [
        metric.get("value")
        for metric in (ablation_dependence, perturbation_rescue, correction)
        if metric.get("status") == "available"
    ]
    score = (
        float(np.nanmean(np.asarray(available_scores, dtype=np.float64)))
        if available_scores
        else None
    )
    return {
        "status": "available" if available_scores else "not_available",
        "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
        "score": score,
        "score_components": [
            name
            for name, metric in (
                ("ablation_dependence_index", ablation_dependence),
                ("perturbation_rescue_index", perturbation_rescue),
                ("correction_index_vs_open_loop", correction),
            )
            if metric.get("status") == "available"
        ],
        "ablation_dependence_index": ablation_dependence,
        "perturbation_rescue_index": perturbation_rescue,
        "correction_index_vs_open_loop": correction,
        "warnings": warnings,
    }


def summarize_feedback_pass_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the baseline feedback-quality lens without changing selection.

    This pass criterion is an audit surface. It combines the existing feedback
    ablation/dependence lens with explicit nominal-quality, calibrated small
    perturbation, sensory/delayed stability, and command-reasonableness evidence.
    Missing evidence is reported as ``not_available`` and can make the aggregate
    result inconclusive, but is never silently converted to a passing zero.
    """

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    warnings: list[str] = []
    nominal = _run_nominal_quality_gate(evaluated, warnings=warnings)
    dependence = _run_feedback_dependence_component(evaluated, warnings=warnings)
    attenuation = _run_small_perturbation_component(evaluated, warnings=warnings)
    stability = _run_sensory_delayed_stability_component(evaluated, warnings=warnings)
    command = _run_command_reasonableness_component(evaluated, warnings=warnings)
    components = {
        "nominal_quality_gate": nominal,
        "feedback_ablation_dependence": dependence,
        "small_calibrated_perturbation_attenuation_readiness": attenuation,
        "sensory_delayed_stability": stability,
        "command_energy_reasonableness": command,
    }
    statuses = [component["status"] for component in components.values()]
    if "fail" in statuses:
        status = "fail"
    elif "not_available" in statuses:
        status = "inconclusive"
    elif "warn" in statuses:
        status = "warn"
    else:
        status = "pass"
    return {
        "status": status,
        "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
        "criterion_role": "audit_reporting_only",
        "components": components,
        "warnings": warnings,
    }


def feedback_checkpoint_selection_audit(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an audit-only feedback-selected checkpoint sidecar.

    Prefer a per-replicate checkpoint rescore when it has been materialized.
    Older manifests without that pass fall back to the legacy run-level summary
    so consumers get an explicit status rather than silent primary-selection
    changes.
    """

    runs = manifest.get("runs", {})
    if not isinstance(runs, Mapping) or not runs:
        return {
            "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
            "status": "not_available",
            "reason": "no_feedback_ablation_runs_available",
            "selection_use": FEEDBACK_AUDIT_SELECTION_ROLE,
            "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
        }
    checkpoint_audits = {
        str(run_id): run.get("feedback_checkpoint_rescore")
        for run_id, run in runs.items()
        if isinstance(run, Mapping)
    }
    materialized = {
        run_id: audit
        for run_id, audit in checkpoint_audits.items()
        if isinstance(audit, Mapping) and audit.get("status") == "materialized"
    }
    if materialized:
        return {
            "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
            "status": "materialized",
            "selection_use": FEEDBACK_AUDIT_SELECTION_ROLE,
            "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
            "feedback_selection_policy": (
                "per-replicate minimum family-balanced signed-pair-aware "
                "absolute/excess feedback perturbation score"
            ),
            "candidate_granularity": "checkpoint_batch_per_replicate",
            "runs": materialized,
            "note": (
                "Feedback-selected checkpoints are reported for audit only. The primary "
                "materialization path still loads validation-selected checkpoints."
            ),
        }

    candidates: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_id, run in runs.items():
        if not isinstance(run, Mapping):
            missing.append(str(run_id))
            continue
        normalized = run.get("normalized_feedback_use", {})
        score = normalized.get("score") if isinstance(normalized, Mapping) else None
        score_float = _coerce_reference_float(score)
        if score_float is None:
            missing.append(str(run_id))
            continue
        candidates.append(
            {
                "run_id": str(run_id),
                "label": run.get("label"),
                "feedback_score": score_float,
                "score_components": list(normalized.get("score_components", ())),
                "checkpoint_selection": run.get("checkpoint_selection", []),
            }
        )
    if missing:
        return {
            "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
            "status": "not_available",
            "reason": "feedback_scores_missing_for_some_candidates",
            "missing_candidates": missing,
            "n_candidates": len(runs),
            "n_scored_candidates": len(candidates),
            "selection_use": FEEDBACK_AUDIT_SELECTION_ROLE,
            "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
        }
    selected = max(candidates, key=lambda candidate: candidate["feedback_score"])
    return {
        "schema_version": FEEDBACK_SELECTION_SCHEMA_VERSION,
        "status": "available",
        "selection_use": FEEDBACK_AUDIT_SELECTION_ROLE,
        "primary_checkpoint_policy": manifest.get("checkpoint_policy"),
        "feedback_selection_policy": "max_normalized_feedback_use_score",
        "candidate_granularity": "run_legacy_fallback",
        "selected_candidate": selected,
        "candidates": sorted(
            candidates,
            key=lambda candidate: candidate["feedback_score"],
            reverse=True,
        ),
        "note": (
            "Feedback-selected candidates are reported for audit only. The primary "
            "materialization path still loads validation-selected checkpoints."
        ),
    }


def interpret_run_feedback_ablation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify the run-level ablation pattern."""

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    if not evaluated:
        return {
            "label": "inconclusive",
            "reason": "no ablation rows were evaluated",
        }

    action_means = {
        (str(row["bin"]), str(row["mode"])): _metric_mean(row, "delta_action_norm")
        for row in evaluated
    }
    perturb_sensitive = [
        value
        for (bin_id, mode), value in action_means.items()
        if bin_id != "nominal"
        and mode
        in {
            "frozen_nominal_observation_tape",
            "zeroed_perturbation_observation_deviation",
            "lagged_observation_history",
        }
        and value is not None
    ]
    channel_position = [
        value
        for (_bin_id, mode), value in action_means.items()
        if mode == "position_only_observation" and value is not None
    ]
    channel_velocity = [
        value
        for (_bin_id, mode), value in action_means.items()
        if mode == "velocity_only_observation" and value is not None
    ]
    max_feedback_delta = max(perturb_sensitive) if perturb_sensitive else 0.0
    max_channel_delta = (
        max(channel_position + channel_velocity) if channel_position or channel_velocity else 0.0
    )
    if max_feedback_delta >= 1e-2:
        return {
            "label": "feedback_sensitive",
            "reason": "observation tape ablations changed actions above the diagnostic threshold",
            "max_feedback_delta_action_norm_mean": max_feedback_delta,
            "max_channel_delta_action_norm_mean": max_channel_delta,
        }
    if max_channel_delta >= 1e-2:
        return {
            "label": "channel_sensitive",
            "reason": "position-only or velocity-only masks changed actions above the threshold",
            "max_feedback_delta_action_norm_mean": max_feedback_delta,
            "max_channel_delta_action_norm_mean": max_channel_delta,
        }
    if evaluated:
        return {
            "label": "motor_tape_like",
            "reason": "all evaluated feedback ablations produced small action changes",
            "max_feedback_delta_action_norm_mean": max_feedback_delta,
            "max_channel_delta_action_norm_mean": max_channel_delta,
        }
    return {"label": "inconclusive", "reason": "insufficient evaluated ablation rows"}


def _effective_checkpoint_policy_from_manifest(
    experiment: str,
    *,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Return the checkpoint policy represented by an optional preferred manifest."""

    manifest = load_materialized_fixed_bank_manifest(
        manifest_path=preferred_checkpoint_manifest_path,
    )
    if manifest is not None:
        return str(
            manifest.metadata.get("checkpoint_policy") or "fixed_bank_rescored_per_replicate"
        )
    return "validation_selected_per_replicate"


def _execute_feedback_ablation_run(
    run: EvaluationRunInputs,
    *,
    source_experiment: str | None,
    n_rollout_trials: int,
    include_checkpoint_rescore: bool = True,
    bank: Mapping[str, Any] | None = None,
    evaluation_bins: Mapping[str, str | None] | None = None,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    model_projection: ModelSlotProjection | None = None,
) -> dict[str, Any]:
    """Evaluate one run for the registered feedback-ablation recipe."""

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")
    if model_projection is None:
        if source_experiment is None:
            raise ValueError("legacy feedback-ablation evaluation requires source_experiment")
        hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
        n_replicates = int(hps.model.n_replicates)
        seed = require_run_seed(run.run_spec, source=run.run_spec_path)
        pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
        model, checkpoint_selection = load_validation_selected_checkpoint_model(
            experiment=source_experiment,
            run_id=run.run_id,
            run_spec=run.run_spec,
            preferred_manifest_path=preferred_checkpoint_manifest_path,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if preferred_checkpoint_manifest_path is not None
                else "sparse_history"
            ),
            repo_root=repo_root,
        )
        task = pair.task
        checkpoint_provenance = [
            selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
        ]
    else:
        model = model_projection.model
        task = model_projection.task
        n_replicates = model_projection.n_replicates
        checkpoint_selection = ()
        checkpoint_provenance = [
            {
                "selection_mode": "exact_native_transaction",
                **asdict(model_projection.provenance),
            }
        ]
    base_trial_specs = repeat_single_validation_trial(task.validation_trials, n_rollout_trials)
    bank = bank or default_cs_perturbation_bank()
    perturbations = {str(row["perturbation_id"]): row for row in bank["perturbations"]}
    evaluation_bins = evaluation_bins or selected_feedback_ablation_bins_for_bank(bank)
    nominal = evaluate_model_on_trial_specs(
        model=model,
        task=task,
        trial_specs=base_trial_specs,
        n_replicates=n_replicates,
        seed=0,
    )
    rows: list[dict[str, Any]] = []
    for bin_id, perturbation_id in evaluation_bins.items():
        trial_specs = base_trial_specs
        bin_model = model
        adapter_json: dict[str, Any] | None = None
        if perturbation_id is not None:
            perturbation = perturbations[perturbation_id]
            adapter = apply_perturbation_to_trial_specs(
                base_trial_specs,
                perturbation,
                model=model,
            )
            adapter_json = adapter.to_json()
            if adapter.status != "evaluated":
                rows.extend(
                    _not_available_rows(
                        bin_id=bin_id,
                        perturbation_id=perturbation_id,
                        reason=adapter.reason or f"{bin_id} perturbation adapter was not evaluated",
                        adapter=adapter_json,
                    )
                )
                continue
            trial_specs = adapter.trial_specs
            bin_model = adapter.model if adapter.model is not None else model
        if perturbation_id is None and bin_model is model and trial_specs is base_trial_specs:
            baseline = nominal
        else:
            baseline = evaluate_model_on_trial_specs(
                model=bin_model,
                task=task,
                trial_specs=trial_specs,
                n_replicates=n_replicates,
                seed=0,
            )
        baseline_cost = full_qrf_cost_summary(baseline.rollout, trial_specs)
        for mode in default_ablation_modes():
            spec = build_observation_ablation_spec(mode, bin_id=bin_id)
            if mode == "normal":
                rows.append(
                    {
                        "bin": bin_id,
                        "mode": mode,
                        "perturbation_id": perturbation_id,
                        "status": "evaluated",
                        "ablation": spec.to_json(),
                        "adapter": adapter_json,
                        "metrics": summarize_ablation_delta(
                            baseline=baseline,
                            ablated=baseline,
                            baseline_cost=baseline_cost,
                            ablated_cost=baseline_cost,
                        ),
                    }
                )
                continue
            tape = build_observation_tape(
                mode,
                bin_feedback=baseline.feedback,
                nominal_feedback=nominal.feedback,
            )
            ablated_trial_specs = trial_specs
            if tape is not None:
                ablated_trial_specs = _add_trial_input(
                    trial_specs,
                    str(spec.input_key),
                    jnp.asarray(tape, dtype=jnp.float64),
                )
            try:
                ablated_model = insert_observation_ablation(bin_model, spec)
            except ValueError as exc:
                rows.append(
                    {
                        "bin": bin_id,
                        "mode": mode,
                        "perturbation_id": perturbation_id,
                        "status": "not_available",
                        "reason": str(exc),
                        "ablation": spec.to_json(),
                        "adapter": adapter_json,
                    }
                )
                continue
            ablated = evaluate_model_on_trial_specs(
                model=ablated_model,
                task=task,
                trial_specs=ablated_trial_specs,
                n_replicates=n_replicates,
                seed=0,
            )
            ablated_cost = full_qrf_cost_summary(ablated.rollout, ablated_trial_specs)
            rows.append(
                {
                    "bin": bin_id,
                    "mode": mode,
                    "perturbation_id": perturbation_id,
                    "status": "evaluated",
                    "ablation": spec.to_json(),
                    "adapter": adapter_json,
                    "metrics": summarize_ablation_delta(
                        baseline=baseline,
                        ablated=ablated,
                        baseline_cost=baseline_cost,
                        ablated_cost=ablated_cost,
                    ),
                }
            )
    run_result = {
        "label": run.label,
        "checkpoint_selection": checkpoint_provenance,
        "n_replicates": n_replicates,
        "n_rollout_trials_per_replicate": n_rollout_trials,
        "n_time_steps": int(nominal.rollout.command.shape[2]),
        "dt_s": float(nominal.rollout.dt),
        "status_counts": _status_counts(rows),
        "interpretation": interpret_run_feedback_ablation(rows),
        "normalized_feedback_use": summarize_normalized_feedback_use(rows),
        "feedback_pass_audit": summarize_feedback_pass_audit(rows),
        "ablations": rows,
    }
    if model_projection is None:
        run_result["run_spec_path"] = _repo_relative(run.run_spec_path, repo_root=repo_root)
        run_result["artifact_dir"] = _repo_relative(run.artifact_dir, repo_root=repo_root)
    else:
        run_result["training_run_id"] = model_projection.provenance.training_manifest_id
    if include_checkpoint_rescore:
        run_result["feedback_checkpoint_rescore"] = feedback_checkpoint_rescore_audit_for_run(
            run=run,
            source_experiment=source_experiment,
            task=task,
            base_trial_specs=base_trial_specs,
            perturbations=perturbations,
            evaluation_bins=evaluation_bins,
            validation_checkpoint_selection=checkpoint_selection,
            n_replicates=n_replicates,
            repo_root=repo_root,
            checkpoint_candidates=(model_projection,) if model_projection is not None else None,
            model_template=model,
        )
    else:
        run_result["feedback_checkpoint_rescore"] = {
            "status": "skipped",
            "reason": "disabled_by_materializer",
            "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
        }
    return run_result


def feedback_checkpoint_rescore_audit_for_run(
    *,
    run: EvaluationRunInputs,
    source_experiment: str,
    task: Any,
    base_trial_specs: Any,
    perturbations: Mapping[str, Mapping[str, Any]],
    evaluation_bins: Mapping[str, str | None],
    validation_checkpoint_selection: Sequence[Any],
    n_replicates: int,
    repo_root: Path = REPO_ROOT,
    checkpoint_candidates: Sequence[ModelSlotProjection] | None = None,
    model_template: Any | None = None,
) -> dict[str, Any]:
    """Score every durable checkpoint on a declared feedback perturbation bank."""

    del source_experiment
    scoring_rows = _feedback_scoring_perturbation_rows(
        perturbations,
        evaluation_bins=evaluation_bins,
    )
    scored_rows: list[dict[str, Any]] = []
    if checkpoint_candidates is not None:
        for candidate in checkpoint_candidates:
            scored_rows.append(
                _score_feedback_checkpoint_batch(
                    model=candidate.model,
                    task=candidate.task,
                    trial_specs=base_trial_specs,
                    scoring_rows=scoring_rows,
                    checkpoint_batch=candidate.provenance.completed_batches,
                    checkpoint_identity=asdict(candidate.provenance),
                    n_replicates=n_replicates,
                    repo_root=repo_root,
                )
            )
    else:
        checkpoint_batches = available_checkpoint_batches(run.artifact_dir)
        if not checkpoint_batches:
            return {
                "status": "not_available",
                "reason": f"no numbered checkpoints under {run.artifact_dir / 'checkpoints'}",
                "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
            }
        for checkpoint_batch in checkpoint_batches:
            checkpoint_path = checkpoint_path_for_batches(run.artifact_dir, checkpoint_batch)
            if model_template is None:
                raise ValueError("legacy feedback checkpoint rescore requires model_template")
            model = eqx.tree_deserialise_leaves(checkpoint_path / "model.eqx", model_template)
            scored_rows.append(
                _score_feedback_checkpoint_batch(
                    model=model,
                    task=task,
                    trial_specs=base_trial_specs,
                    scoring_rows=scoring_rows,
                    checkpoint_batch=checkpoint_batch,
                    checkpoint_identity={
                        "legacy_checkpoint_path": _repo_relative(
                            checkpoint_path, repo_root=repo_root
                        )
                    },
                    n_replicates=n_replicates,
                    repo_root=repo_root,
                )
            )

    selected = []
    validation_by_replicate = {
        int(selection.replicate): selection for selection in validation_checkpoint_selection
    }
    for replicate in range(n_replicates):
        candidates = [
            row
            for row in scored_rows
            if row["per_replicate_scores"][replicate]["status"] == "available"
        ]
        if not candidates:
            selected.append(
                {
                    "replicate": replicate,
                    "status": "not_available",
                    "reason": "no checkpoint had an available feedback-bank score",
                }
            )
            continue
        best = min(
            candidates,
            key=lambda row: row["per_replicate_scores"][replicate]["score"],
        )
        validation_selection = validation_by_replicate.get(replicate)
        validation_batch = (
            None if validation_selection is None else validation_selection.checkpoint_batches
        )
        selected.append(
            {
                "replicate": replicate,
                "status": "available",
                "feedback_selected_checkpoint_batches": int(best["checkpoint_batches"]),
                "feedback_score": float(best["per_replicate_scores"][replicate]["score"]),
                "feedback_score_components": best["per_replicate_scores"][replicate][
                    "score_components"
                ],
                "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
                "validation_selected_checkpoint_batches": validation_batch,
                "feedback_minus_validation_batches": (
                    None
                    if validation_batch is None
                    else int(best["checkpoint_batches"]) - int(validation_batch)
                ),
                "n_available_feedback_bins": int(
                    best["per_replicate_scores"][replicate]["n_available_bins"]
                ),
            }
        )

    return {
        "status": "materialized",
        "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
        "selection_policy": (
            "per-replicate argmin of family-balanced signed-pair-aware "
            "absolute/excess perturbation response score with nominal-quality gates "
            "and available command-energy/smoothness/oscillation penalties"
        ),
        "selection_leakage_guard": (
            "feedback-selected checkpoints are reported only in this sidecar; "
            "they become the effective loaded checkpoints only when materialized "
            "through an explicit fixed-bank checkpoint manifest"
        ),
        "feedback_bank_bins": [
            bin_id
            for bin_id, perturbation_id in evaluation_bins.items()
            if perturbation_id is not None
        ],
        "feedback_scoring_policy": _feedback_scoring_policy_json(),
        "n_checkpoint_candidates": len(scored_rows),
        "checkpoint_scores": scored_rows,
        "feedback_selected_checkpoints": selected,
        "source_checkpoint_policy": "validation_selected_per_replicate",
    }


def _score_feedback_checkpoint_batch(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    scoring_rows: Sequence[Mapping[str, Any]],
    checkpoint_batch: int,
    checkpoint_identity: Mapping[str, Any],
    n_replicates: int,
    repo_root: Path,
) -> dict[str, Any]:
    """Score one checkpoint on normal-controller feedback perturbation bins."""

    baseline = evaluate_model_on_trial_specs(
        model=model,
        task=task,
        trial_specs=trial_specs,
        n_replicates=n_replicates,
        seed=0,
    )
    baseline_cost = full_qrf_cost_summary(baseline.rollout, trial_specs)
    nominal_gate = _nominal_quality_gate_by_replicate(
        baseline=baseline,
        baseline_cost=baseline_cost,
        n_replicates=n_replicates,
    )
    bin_scores = []
    for scoring_row in scoring_rows:
        bin_id = str(scoring_row["bin"])
        perturbation = scoring_row["perturbation"]
        perturbation_id = str(perturbation["perturbation_id"])
        adapter = apply_perturbation_to_trial_specs(
            trial_specs,
            perturbation,
            model=model,
        )
        if adapter.status != "evaluated":
            bin_scores.append(
                {
                    "bin": bin_id,
                    "perturbation_id": perturbation_id,
                    "family": scoring_row.get("family"),
                    "signed_pair_key": scoring_row.get("signed_pair_key"),
                    "sign": scoring_row.get("sign"),
                    "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
                    "status": adapter.status,
                    "reason": adapter.reason,
                }
            )
            continue
        perturbed = evaluate_model_on_trial_specs(
            model=adapter.model if adapter.model is not None else model,
            task=task,
            trial_specs=adapter.trial_specs,
            n_replicates=n_replicates,
            seed=0,
        )
        perturbed_cost = full_qrf_cost_summary(perturbed.rollout, adapter.trial_specs)
        delta_cost = delta_full_qrf_cost_summary(baseline_cost, perturbed_cost)
        values = _per_replicate_cost_delta_values(
            baseline_cost,
            perturbed_cost,
            n_replicates=n_replicates,
        )
        command_metrics = _per_replicate_command_penalty_metrics(
            baseline_command=baseline.rollout.command,
            perturbed_command=perturbed.rollout.command,
            n_replicates=n_replicates,
        )
        replicate_scores = _per_replicate_feedback_response_scores(
            values,
            command_metrics=command_metrics,
        )
        bin_scores.append(
            {
                "bin": bin_id,
                "perturbation_id": perturbation_id,
                "family": scoring_row.get("family"),
                "signed_pair_key": scoring_row.get("signed_pair_key"),
                "sign": scoring_row.get("sign"),
                "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
                "status": delta_cost.get("status", "not_available"),
                "mean_delta_full_qrf_cost": _summary_mean(
                    delta_cost.get("delta_cost", {}).get("total", {})
                ),
                "per_replicate_mean_delta_full_qrf_cost": values,
                "per_replicate_feedback_response_score": replicate_scores,
                "command_penalty_metrics": command_metrics,
            }
        )
    per_replicate_scores = []
    for replicate in range(n_replicates):
        per_replicate_scores.append(
            _aggregate_feedback_checkpoint_score(
                bin_scores,
                replicate=replicate,
                nominal_gate=nominal_gate[replicate],
            )
        )
    return {
        "checkpoint_batches": int(checkpoint_batch),
        "checkpoint_identity": dict(checkpoint_identity),
        "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
        "nominal_quality_gate": nominal_gate,
        "per_replicate_scores": per_replicate_scores,
        "bin_scores": bin_scores,
    }


def _feedback_scoring_policy_json() -> dict[str, Any]:
    return {
        "selection_role": FEEDBACK_AUDIT_SELECTION_ROLE,
        "primary_policy_invariance": (
            "feedback audit is not a hidden selector; the effective checkpoint policy "
            "is whichever manifest the materializer is explicitly given"
        ),
        "score_orientation": "lower_is_better",
        "response_cost": (
            "absolute paired full-Q/R/Q_f perturbation delta; excess positive delta is "
            "also reported so negative signed improvements cannot win by sign alone"
        ),
        "family_balance": "mean over families after within-family row/pair aggregation",
        "signed_pair_handling": (
            "rows with available +/- counterparts are averaged by signed_pair_key before "
            "family balancing; missing counterparts emit warnings"
        ),
        "nominal_quality_gate": {
            "endpoint_warn_m": NOMINAL_ENDPOINT_WARN_M,
            "terminal_speed_warn_m_s": NOMINAL_TERMINAL_SPEED_WARN_M_S,
            "finite_full_qrf_cost_required": True,
        },
        "penalties": {
            "command_energy_ratio": "available",
            "command_smoothness_ratio": "available when command horizon has >=2 steps",
            "command_oscillation_ratio": "available when command horizon has >=3 steps",
            "warn_ratio": COMMAND_RATIO_WARN,
            "fail_ratio": COMMAND_RATIO_FAIL,
        },
    }


def _feedback_scoring_perturbation_rows(
    perturbations: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_bins: Mapping[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    evaluation_bins = evaluation_bins or selected_feedback_ablation_bins()
    for bin_id, perturbation_id in evaluation_bins.items():
        if perturbation_id is None or perturbation_id not in perturbations:
            continue
        perturbation = perturbations[perturbation_id]
        for row in (perturbation, _opposite_signed_perturbation(perturbation, perturbations)):
            if row is None:
                continue
            row_id = str(row["perturbation_id"])
            if row_id in seen:
                continue
            seen.add(row_id)
            rows.append(
                {
                    "bin": bin_id,
                    "perturbation": row,
                    "family": _feedback_family(row),
                    "signed_pair_key": _signed_pair_key(row),
                    "sign": _coerce_int(row.get("sign")),
                }
            )
    return rows


def _opposite_signed_perturbation(
    perturbation: Mapping[str, Any],
    perturbations: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    sign = _coerce_int(perturbation.get("sign"))
    if sign not in {-1, 1}:
        return None
    key = _signed_pair_key(perturbation)
    for candidate in perturbations.values():
        if candidate is perturbation:
            continue
        if _signed_pair_key(candidate) == key and _coerce_int(candidate.get("sign")) == -sign:
            return candidate
    return None


def _signed_pair_key(perturbation: Mapping[str, Any]) -> str:
    timing = perturbation.get("timing", {})
    timing_items = tuple(sorted(timing.items())) if isinstance(timing, Mapping) else ()
    return json.dumps(
        {
            "channel": perturbation.get("channel"),
            "family": perturbation.get("family"),
            "axis": perturbation.get("axis"),
            "basis": perturbation.get("basis"),
            "timing": timing_items,
            "epsilon_component": perturbation.get("epsilon_component"),
            "epsilon_index": perturbation.get("epsilon_index"),
            "initial_position_case": perturbation.get("initial_position_case"),
        },
        sort_keys=True,
        default=str,
    )


def _feedback_family(perturbation: Mapping[str, Any]) -> str:
    return str(
        perturbation.get("semantic_family")
        or perturbation.get("family")
        or perturbation.get("channel")
        or "unknown_family"
    )


def _nominal_quality_gate_by_replicate(
    *,
    baseline: DetailedRolloutEvaluation,
    baseline_cost: Mapping[str, Any],
    n_replicates: int,
) -> list[dict[str, Any]]:
    endpoints = _per_replicate_reduce(_endpoint_error(baseline.rollout), n_replicates)
    speeds = _per_replicate_reduce(_terminal_speed(baseline.rollout), n_replicates)
    costs = _per_replicate_cost_values(baseline_cost, n_replicates=n_replicates)
    gates = []
    for replicate in range(n_replicates):
        warnings = []
        endpoint = endpoints[replicate]
        speed = speeds[replicate]
        cost = costs[replicate]
        if endpoint is None or speed is None or cost is None:
            gates.append(
                {
                    "replicate": replicate,
                    "status": "fail",
                    "reason": "nominal_quality_metrics_not_available",
                    "warnings": ["nominal quality metrics not available"],
                }
            )
            continue
        if not np.isfinite(cost):
            warnings.append("nominal full-Q/R/Q_f cost is nonfinite")
        if endpoint > NOMINAL_ENDPOINT_WARN_M:
            warnings.append("nominal endpoint error exceeds warning threshold")
        if speed > NOMINAL_TERMINAL_SPEED_WARN_M_S:
            warnings.append("nominal terminal speed exceeds warning threshold")
        gates.append(
            {
                "replicate": replicate,
                "status": "fail" if not np.isfinite(cost) else ("warn" if warnings else "pass"),
                "endpoint_error_m": endpoint,
                "terminal_speed_m_s": speed,
                "full_qrf_cost": cost,
                "thresholds": {
                    "endpoint_warn_m": NOMINAL_ENDPOINT_WARN_M,
                    "terminal_speed_warn_m_s": NOMINAL_TERMINAL_SPEED_WARN_M_S,
                },
                "warnings": warnings,
            }
        )
    return gates


def _per_replicate_command_penalty_metrics(
    *,
    baseline_command: Any,
    perturbed_command: Any,
    n_replicates: int,
) -> list[dict[str, Any]]:
    baseline = jnp.asarray(baseline_command, dtype=jnp.float64)
    perturbed = jnp.asarray(perturbed_command, dtype=jnp.float64)
    if baseline.shape != perturbed.shape or baseline.shape[:1] != (n_replicates,):
        return [
            {
                "replicate": replicate,
                "status": "not_available",
                "reason": "command arrays unavailable or shape-mismatched",
            }
            for replicate in range(n_replicates)
        ]
    results = []
    for replicate in range(n_replicates):
        base = baseline[replicate]
        pert = perturbed[replicate]
        metrics: dict[str, Any] = {"replicate": replicate, "status": "available"}
        metrics["command_energy_ratio"] = _safe_ratio(
            float(jnp.mean(jnp.sum(jnp.square(pert), axis=-1))),
            float(jnp.mean(jnp.sum(jnp.square(base), axis=-1))),
        )
        if base.shape[-2] >= 2:
            metrics["command_smoothness_ratio"] = _safe_ratio(
                float(jnp.mean(jnp.sum(jnp.square(jnp.diff(pert, axis=-2)), axis=-1))),
                float(jnp.mean(jnp.sum(jnp.square(jnp.diff(base, axis=-2)), axis=-1))),
            )
        else:
            metrics["command_smoothness_ratio"] = None
            metrics["status"] = "partial"
            metrics.setdefault("warnings", []).append("command smoothness unavailable")
        if base.shape[-2] >= 3:
            metrics["command_oscillation_ratio"] = _safe_ratio(
                _oscillation_rate(pert),
                _oscillation_rate(base),
            )
        else:
            metrics["command_oscillation_ratio"] = None
            metrics["status"] = "partial"
            metrics.setdefault("warnings", []).append("command oscillation unavailable")
        results.append(metrics)
    return results


def _command_penalty(
    metrics: Mapping[str, Any],
    *,
    warnings: list[str],
) -> float:
    if metrics.get("status") == "not_available":
        warnings.append(f"command penalties not available: {metrics.get('reason')}")
        return 0.0
    penalty = 0.0
    for key in (
        "command_energy_ratio",
        "command_smoothness_ratio",
        "command_oscillation_ratio",
    ):
        ratio = _coerce_reference_float(metrics.get(key))
        if ratio is None:
            warnings.append(f"{key} not available")
            continue
        if ratio >= COMMAND_RATIO_FAIL:
            penalty += ratio
            warnings.append(f"{key} exceeds fail ratio {COMMAND_RATIO_FAIL:g}")
        elif ratio >= COMMAND_RATIO_WARN:
            penalty += 0.25 * ratio
            warnings.append(f"{key} exceeds warn ratio {COMMAND_RATIO_WARN:g}")
    return float(penalty)


def _per_replicate_feedback_response_scores(
    delta_values: Sequence[float | None],
    *,
    command_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for replicate, value in enumerate(delta_values):
        warnings: list[str] = []
        if value is None or not np.isfinite(float(value)):
            scores.append(
                {
                    "replicate": replicate,
                    "status": "not_available",
                    "reason": "paired full-Q/R/Q_f delta unavailable",
                    "warnings": warnings,
                }
            )
            continue
        absolute = abs(float(value))
        excess = max(float(value), 0.0)
        command_penalty = _command_penalty(command_metrics[replicate], warnings=warnings)
        score = absolute + command_penalty
        scores.append(
            {
                "replicate": replicate,
                "status": "available",
                "score": float(score),
                "absolute_delta_full_qrf_cost": float(absolute),
                "excess_delta_full_qrf_cost": float(excess),
                "raw_signed_delta_full_qrf_cost": float(value),
                "command_penalty": command_penalty,
                "warnings": warnings,
            }
        )
    return scores


def _aggregate_feedback_checkpoint_score(
    bin_scores: Sequence[Mapping[str, Any]],
    *,
    replicate: int,
    nominal_gate: Mapping[str, Any],
) -> dict[str, Any]:
    warnings = list(nominal_gate.get("warnings", ()))
    if nominal_gate.get("status") == "fail":
        return {
            "replicate": replicate,
            "status": "not_available",
            "reason": "nominal_quality_gate_failed",
            "nominal_quality_gate": nominal_gate,
            "n_available_bins": 0,
            "warnings": warnings,
        }

    entries = []
    available_pair_keys: dict[str, set[int]] = {}
    for row in bin_scores:
        per_rep = row.get("per_replicate_feedback_response_score", ())
        if replicate >= len(per_rep) or not isinstance(per_rep[replicate], Mapping):
            continue
        score = per_rep[replicate]
        if score.get("status") != "available":
            continue
        sign = _coerce_int(row.get("sign"))
        pair_key = str(row.get("signed_pair_key"))
        if sign in {-1, 1}:
            available_pair_keys.setdefault(pair_key, set()).add(sign)
        entries.append(
            {
                "family": str(row.get("family", "unknown_family")),
                "signed_pair_key": pair_key,
                "sign": sign,
                "bin": row.get("bin"),
                "perturbation_id": row.get("perturbation_id"),
                "score": float(score["score"]),
                "score_detail": score,
            }
        )
    if not entries:
        return {
            "replicate": replicate,
            "status": "not_available",
            "reason": "no evaluated feedback bins",
            "nominal_quality_gate": nominal_gate,
            "n_available_bins": 0,
            "warnings": warnings,
        }

    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    singles: list[dict[str, Any]] = []
    for entry in entries:
        sign = entry["sign"]
        pair_key = entry["signed_pair_key"]
        if sign in {-1, 1} and available_pair_keys.get(pair_key) == {-1, 1}:
            pair_groups.setdefault((entry["family"], pair_key), []).append(entry)
        else:
            if sign in {-1, 1}:
                warnings.append(
                    "signed-pair counterpart not available for "
                    f"{entry['perturbation_id']}; scored as unpaired absolute response"
                )
            singles.append(entry)

    family_values: dict[str, list[float]] = {}
    score_components: list[dict[str, Any]] = []
    for (family, pair_key), members in pair_groups.items():
        value = float(np.mean([member["score"] for member in members]))
        family_values.setdefault(family, []).append(value)
        score_components.append(
            {
                "type": "signed_pair",
                "family": family,
                "signed_pair_key": pair_key,
                "score": value,
                "members": [member["perturbation_id"] for member in members],
            }
        )
    for entry in singles:
        family = entry["family"]
        family_values.setdefault(family, []).append(entry["score"])
        score_components.append(
            {
                "type": "unpaired_row",
                "family": family,
                "score": entry["score"],
                "perturbation_id": entry["perturbation_id"],
            }
        )
    family_scores = {
        family: float(np.mean(np.asarray(values, dtype=np.float64)))
        for family, values in family_values.items()
    }
    return {
        "replicate": replicate,
        "status": "available",
        "score": float(np.mean(np.asarray(list(family_scores.values()), dtype=np.float64))),
        "score_orientation": "lower_is_better",
        "score_components": score_components,
        "family_scores": family_scores,
        "nominal_quality_gate": nominal_gate,
        "n_available_bins": len(entries),
        "warnings": warnings,
    }


def _per_replicate_cost_delta_values(
    base_cost: Mapping[str, Any],
    perturbed_cost: Mapping[str, Any],
    *,
    n_replicates: int,
) -> list[float | None]:
    """Return mean total delta cost per replicate from a paired cost summary."""

    if base_cost.get("status") != "available" or perturbed_cost.get("status") != "available":
        return [None for _ in range(n_replicates)]
    base_values = np.asarray(base_cost.get("total", {}).get("values"), dtype=np.float64)
    perturbed_values = np.asarray(
        perturbed_cost.get("total", {}).get("values"),
        dtype=np.float64,
    )
    if (
        base_values.shape[:1] != (n_replicates,)
        or perturbed_values.shape[:1] != (n_replicates,)
        or base_values.shape != perturbed_values.shape
    ):
        return [None for _ in range(n_replicates)]
    values = perturbed_values - base_values
    if values.ndim == 1:
        return [float(value) for value in values]
    reduced = np.mean(values.reshape((n_replicates, -1)), axis=1)
    return [float(value) for value in reduced]


def _per_replicate_cost_values(
    cost: Mapping[str, Any],
    *,
    n_replicates: int,
) -> list[float | None]:
    if cost.get("status") != "available":
        return [None for _ in range(n_replicates)]
    values = np.asarray(cost.get("total", {}).get("values"), dtype=np.float64)
    return _per_replicate_reduce(values, n_replicates)


def _per_replicate_reduce(values: Any, n_replicates: int) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[:1] != (n_replicates,):
        return [None for _ in range(n_replicates)]
    if array.ndim == 1:
        return [float(value) for value in array]
    reduced = np.mean(array.reshape((n_replicates, -1)), axis=1)
    return [float(value) for value in reduced]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return None
    return float(numerator / abs(denominator))


def _oscillation_rate(command: np.ndarray) -> float:
    diffs = np.diff(command, axis=-2)
    if diffs.shape[-2] < 2:
        return 0.0
    signs = np.sign(diffs)
    changes = signs[..., 1:, :] * signs[..., :-1, :] < 0.0
    return float(np.mean(changes))


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render_feedback_ablation_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact Markdown sidecar for a feedback-ablation manifest."""

    lines = [
        "# GRU Feedback Ablation Diagnostic",
        "",
        f"- Issue: `{manifest['issue']}`",
        f"- Source experiment: `{manifest['source_experiment']}`",
        f"- Scope: `{manifest['scope']}`",
        f"- Checkpoint policy: `{manifest['checkpoint_policy']}`",
        "",
        "## Interpretation",
        "",
        "| Run | Label | Reason |",
        "|---|---|---|",
    ]
    for run_id, run in manifest["runs"].items():
        interpretation = run.get("interpretation", {})
        lines.append(
            f"| `{run_id}` | `{interpretation.get('label', 'inconclusive')}` | "
            f"{interpretation.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Ablation Deltas",
            "",
            "| Run | Bin | Mode | Status | dAction mean | dFull-QRF mean | "
            "dEndpoint mean | dTerminal speed mean |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for run_id, run in manifest["runs"].items():
        for row in run.get("ablations", []):
            metrics = row.get("metrics", {})
            full_qrf = metrics.get("rollout_full_qrf", {})
            delta_cost = full_qrf.get("delta_cost", {}) if isinstance(full_qrf, Mapping) else {}
            lines.append(
                f"| `{run_id}` | `{row.get('bin')}` | `{row.get('mode')}` | "
                f"{row.get('status')} | "
                f"{_format_float(_summary_mean(metrics.get('delta_action_norm')))} | "
                f"{_format_float(_summary_mean(delta_cost.get('total')))} | "
                f"{_format_float(_summary_mean(metrics.get('delta_endpoint_error_m')))} | "
                f"{_format_float(_summary_mean(metrics.get('delta_terminal_speed_m_s')))} |"
            )
    lines.extend(
        [
            "",
            "## Normalized Feedback-Use Indices",
            "",
            "| Run | Status | Score | Ablation dependence | Perturbation rescue | "
            "Correction vs open-loop | Warnings |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for run_id, run in manifest["runs"].items():
        normalized = run.get("normalized_feedback_use", {})
        lines.append(
            f"| `{run_id}` | {normalized.get('status', 'not_available')} | "
            f"{_format_float(normalized.get('score'))} | "
            f"{_format_float(_index_value(normalized, 'ablation_dependence_index'))} | "
            f"{_format_float(_index_value(normalized, 'perturbation_rescue_index'))} | "
            f"{_format_float(_index_value(normalized, 'correction_index_vs_open_loop'))} | "
            f"{'; '.join(normalized.get('warnings', ())) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Feedback Pass Audit",
            "",
            "| Run | Overall | Nominal gate | Dependence | Small perturbation | "
            "Sensory/delayed | Command | Warnings |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for run_id, run in manifest["runs"].items():
        audit_pass = run.get("feedback_pass_audit", {})
        components = audit_pass.get("components", {}) if isinstance(audit_pass, Mapping) else {}
        perturbation_status = _component_status(
            components,
            "small_calibrated_perturbation_attenuation_readiness",
        )
        lines.append(
            f"| `{run_id}` | `{audit_pass.get('status', 'inconclusive')}` | "
            f"`{_component_status(components, 'nominal_quality_gate')}` | "
            f"`{_component_status(components, 'feedback_ablation_dependence')}` | "
            f"`{perturbation_status}` | "
            f"`{_component_status(components, 'sensory_delayed_stability')}` | "
            f"`{_component_status(components, 'command_energy_reasonableness')}` | "
            f"{'; '.join(audit_pass.get('warnings', ())) or 'none'} |"
        )
    audit = manifest.get("feedback_checkpoint_selection_audit", {})
    lines.extend(
        [
            "",
            "## Feedback-Selected Checkpoint Audit",
            "",
            f"- Status: `{audit.get('status', 'not_available')}`",
            f"- Selection use: `{audit.get('selection_use', FEEDBACK_AUDIT_SELECTION_ROLE)}`",
            "- Primary checkpoint policy: "
            f"`{audit.get('primary_checkpoint_policy', manifest.get('checkpoint_policy'))}`",
        ]
    )
    if isinstance(audit, Mapping) and audit.get("status") == "available":
        selected = audit.get("selected_candidate", {})
        if isinstance(selected, Mapping):
            lines.append(f"- Feedback-selected candidate: `{selected.get('run_id')}`")
    elif isinstance(audit, Mapping) and audit.get("status") == "materialized":
        lines.extend(
            [
                "",
                "| Run | Replicate | Validation checkpoint | Feedback checkpoint | "
                "Feedback - validation | Feedback score | Bins |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id, audit_run in audit.get("runs", {}).items():
            if not isinstance(audit_run, Mapping):
                continue
            for selection in audit_run.get("feedback_selected_checkpoints", ()):
                if not isinstance(selection, Mapping):
                    continue
                lines.append(
                    f"| `{run_id}` | {selection.get('replicate')} | "
                    f"{selection.get('validation_selected_checkpoint_batches')} | "
                    f"{selection.get('feedback_selected_checkpoint_batches')} | "
                    f"{selection.get('feedback_minus_validation_batches')} | "
                    f"{_format_float(selection.get('feedback_score'))} | "
                    f"{selection.get('n_available_feedback_bins', 0)} |"
                )
    elif isinstance(audit, Mapping) and audit.get("reason"):
        lines.append(f"- Reason: {audit.get('reason')}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `normal` is the per-bin baseline; all other rows are paired deltas against it.",
            "- `frozen_nominal_observation_tape` and "
            "`zeroed_perturbation_observation_deviation` are separate diagnostic lenses "
            "that share the nominal feedback tape for perturbed bins.",
            "- Validation-selected checkpoints are used for model loading. Analytical "
            "action/I/O metrics, where present in adjacent diagnostics, remain audit-only.",
            "",
        ]
    )
    return "\n".join(lines)


def _ablation_dependence_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    candidates: list[float] = []
    for row in rows:
        if row.get("mode") == "normal":
            continue
        metrics = row.get("metrics", {})
        delta_action = _summary_mean(metrics.get("delta_action_norm"))
        baseline_action = _summary_mean(metrics.get("baseline_action_norm"))
        if delta_action is None:
            continue
        if baseline_action is None or abs(baseline_action) <= 1e-12:
            warnings.append(
                "ablation_dependence_index denominator unavailable or near zero for "
                f"{row.get('bin')}:{row.get('mode')}"
            )
            continue
        candidates.append(float(delta_action) / abs(float(baseline_action)))
    if not candidates:
        return {
            "status": "not_available",
            "reason": "no evaluated ablation rows with nonzero baseline action denominator",
        }
    return {
        "status": "available",
        "value": float(max(candidates)),
        "normalization": "max_delta_action_norm_mean_over_baseline_action_norm_mean",
    }


def _perturbation_rescue_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    candidates: list[float] = []
    for row in rows:
        if row.get("bin") == "nominal" or row.get("mode") not in {
            "frozen_nominal_observation_tape",
            "zeroed_perturbation_observation_deviation",
        }:
            continue
        rollout_full_qrf = row.get("metrics", {}).get("rollout_full_qrf", {})
        if (
            not isinstance(rollout_full_qrf, Mapping)
            or rollout_full_qrf.get("status") != "available"
        ):
            warnings.append(
                "perturbation_rescue_index full-Q/R/Q_f denominator unavailable for "
                f"{row.get('bin')}:{row.get('mode')}"
            )
            continue
        delta_total = _summary_mean(rollout_full_qrf.get("delta_cost", {}).get("total", {}))
        perturbed_total = _summary_mean(rollout_full_qrf.get("perturbed_cost", {}).get("total", {}))
        if delta_total is None:
            continue
        if perturbed_total is None or abs(perturbed_total) <= 1e-12:
            warnings.append(
                "perturbation_rescue_index denominator unavailable or near zero for "
                f"{row.get('bin')}:{row.get('mode')}"
            )
            continue
        candidates.append(float(delta_total) / abs(float(perturbed_total)))
    if not candidates:
        return {
            "status": "not_available",
            "reason": "no perturbation rows with available paired full-Q/R/Q_f costs",
        }
    return {
        "status": "available",
        "value": float(max(candidates)),
        "normalization": "max_ablation_extra_full_qrf_cost_over_ablated_full_qrf_cost",
    }


def _correction_index_vs_open_loop(
    rows: Sequence[Mapping[str, Any]],
    *,
    open_loop_reference: Mapping[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    if open_loop_reference is None:
        warnings.append("correction_index_vs_open_loop not available: open-loop data not supplied")
        return {
            "status": "not_available",
            "reason": "open_loop_reference_not_supplied",
        }
    values = []
    for row in rows:
        if row.get("mode") != "normal" or row.get("bin") == "nominal":
            continue
        bin_id = str(row.get("bin"))
        open_loop_delta = _coerce_reference_float(open_loop_reference.get(bin_id))
        if open_loop_delta is None or abs(open_loop_delta) <= 1e-12:
            warnings.append(
                f"correction_index_vs_open_loop denominator unavailable or near zero for {bin_id}"
            )
            continue
        closed_loop_delta = _summary_mean(
            row.get("metrics", {})
            .get("rollout_full_qrf", {})
            .get("delta_cost", {})
            .get("total", {})
        )
        if closed_loop_delta is None:
            continue
        values.append(
            (float(open_loop_delta) - float(closed_loop_delta)) / abs(float(open_loop_delta))
        )
    if not values:
        return {
            "status": "not_available",
            "reason": "no perturbation rows had both closed-loop and open-loop cost deltas",
        }
    return {
        "status": "available",
        "value": float(np.nanmean(np.asarray(values, dtype=np.float64))),
        "normalization": "mean_open_loop_minus_closed_loop_delta_cost_over_open_loop_delta_cost",
    }


def _run_nominal_quality_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    nominal = next(
        (row for row in rows if row.get("bin") == "nominal" and row.get("mode") == "normal"),
        None,
    )
    if nominal is None:
        warnings.append("nominal quality gate not available: nominal normal row missing")
        return {"status": "not_available", "reason": "nominal_normal_row_missing"}
    metrics = nominal.get("metrics", {})
    endpoint = _summary_mean(metrics.get("baseline_endpoint_error_m"))
    terminal_speed = _summary_mean(metrics.get("baseline_terminal_speed_m_s"))
    total_cost = _summary_mean(metrics.get("baseline_full_qrf_cost", {}).get("total", {}))
    if endpoint is None or terminal_speed is None or total_cost is None:
        warnings.append("nominal quality gate not available: nominal quality metrics missing")
        return {"status": "not_available", "reason": "nominal_quality_metrics_missing"}
    failures = []
    if not np.isfinite(total_cost):
        failures.append("nonfinite_full_qrf_cost")
    if endpoint > NOMINAL_ENDPOINT_WARN_M:
        failures.append("endpoint_error_above_warn_threshold")
    if terminal_speed > NOMINAL_TERMINAL_SPEED_WARN_M_S:
        failures.append("terminal_speed_above_warn_threshold")
    status = "warn" if failures else "pass"
    return {
        "status": status,
        "endpoint_error_m": endpoint,
        "terminal_speed_m_s": terminal_speed,
        "full_qrf_cost": total_cost,
        "thresholds": {
            "endpoint_warn_m": NOMINAL_ENDPOINT_WARN_M,
            "terminal_speed_warn_m_s": NOMINAL_TERMINAL_SPEED_WARN_M_S,
        },
        "warnings": failures,
    }


def _run_feedback_dependence_component(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    metric = _ablation_dependence_index(rows, warnings=warnings)
    if metric.get("status") != "available":
        return {"status": "not_available", "reason": metric.get("reason")}
    value = float(metric["value"])
    return {
        "status": "pass" if value >= 1e-2 else "warn",
        "ablation_dependence_index": value,
        "threshold": 1e-2,
    }


def _run_small_perturbation_component(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    values = []
    for row in rows:
        if row.get("bin") == "nominal" or row.get("mode") != "normal":
            continue
        value = _summary_mean(
            row.get("metrics", {}).get("baseline_full_qrf_cost", {}).get("total", {})
        )
        if value is not None:
            values.append(float(value))
    if not values:
        warnings.append(
            "small calibrated perturbation attenuation readiness not available: "
            "normal perturbation cost metrics missing"
        )
        return {"status": "not_available", "reason": "normal_perturbation_costs_missing"}
    finite = [value for value in values if np.isfinite(value)]
    if len(finite) != len(values):
        return {"status": "fail", "reason": "nonfinite_perturbation_cost"}
    return {
        "status": "pass",
        "n_perturbation_bins": len(finite),
        "mean_normal_perturbed_full_qrf_cost": float(np.mean(np.asarray(finite))),
    }


def _run_sensory_delayed_stability_component(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    bins = {"sensory_feedback", "delayed_observation"}
    values = []
    for row in rows:
        if row.get("bin") not in bins or row.get("mode") != "normal":
            continue
        metrics = row.get("metrics", {})
        endpoint = _summary_mean(metrics.get("baseline_endpoint_error_m"))
        terminal_speed = _summary_mean(metrics.get("baseline_terminal_speed_m_s"))
        if endpoint is not None and terminal_speed is not None:
            values.append((str(row.get("bin")), endpoint, terminal_speed))
    if not values:
        warnings.append("sensory/delayed stability not available: normal rows missing")
        return {"status": "not_available", "reason": "sensory_delayed_normal_rows_missing"}
    unstable = [
        bin_id
        for bin_id, endpoint, terminal_speed in values
        if endpoint > NOMINAL_ENDPOINT_WARN_M or terminal_speed > NOMINAL_TERMINAL_SPEED_WARN_M_S
    ]
    return {
        "status": "warn" if unstable else "pass",
        "evaluated_bins": [bin_id for bin_id, _endpoint, _speed in values],
        "unstable_bins": unstable,
    }


def _run_command_reasonableness_component(
    rows: Sequence[Mapping[str, Any]],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    values = []
    for row in rows:
        if row.get("mode") != "normal":
            continue
        baseline_action = _summary_mean(row.get("metrics", {}).get("baseline_action_norm"))
        if baseline_action is not None:
            values.append(float(baseline_action))
    if not values:
        warnings.append("command energy reasonableness not available: action metrics missing")
        return {"status": "not_available", "reason": "normal_action_metrics_missing"}
    if any(not np.isfinite(value) for value in values):
        return {"status": "fail", "reason": "nonfinite_action_norm"}
    ratio = max(values) / max(float(np.median(np.asarray(values))), 1e-12)
    if ratio >= COMMAND_RATIO_FAIL:
        status = "fail"
    elif ratio >= COMMAND_RATIO_WARN:
        status = "warn"
    else:
        status = "pass"
    return {
        "status": status,
        "max_to_median_action_norm_ratio": float(ratio),
        "warn_ratio": COMMAND_RATIO_WARN,
        "fail_ratio": COMMAND_RATIO_FAIL,
    }


def _coerce_reference_float(value: Any) -> float | None:
    if isinstance(value, Mapping):
        if "delta_cost" in value:
            return _coerce_reference_float(value["delta_cost"])
        if "total" in value:
            return _coerce_reference_float(value["total"])
        if "mean" in value:
            return _coerce_reference_float(value["mean"])
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _index_value(normalized: Mapping[str, Any], key: str) -> float | None:
    metric = normalized.get(key, {})
    if not isinstance(metric, Mapping):
        return None
    value = metric.get("value")
    if value is None:
        return None
    return float(value)


def _not_available_rows(
    *,
    bin_id: str,
    perturbation_id: str | None,
    reason: str,
    adapter: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "bin": bin_id,
            "mode": mode,
            "perturbation_id": perturbation_id,
            "status": "not_available",
            "reason": reason,
            "ablation": build_observation_ablation_spec(mode, bin_id=bin_id).to_json(),
            "adapter": None if adapter is None else dict(adapter),
        }
        for mode in default_ablation_modes()
    ]


def _add_trial_input(trial_specs: Any, key: str, value: Any) -> Any:
    return eqx.tree_at(
        lambda ts: ts.inputs,
        trial_specs,
        {**trial_specs.inputs, key: value},
    )


def _endpoint_error(evaluation: RolloutEvaluation) -> np.ndarray:
    endpoint = evaluation.position[:, :, -1, :]
    target = evaluation.target_position[:, -1, :]
    return np.linalg.norm(endpoint - target[None, :, :], axis=-1)


def _terminal_speed(evaluation: RolloutEvaluation) -> np.ndarray:
    return np.linalg.norm(evaluation.velocity[:, :, -1, :], axis=-1)


def _metric_mean(row: Mapping[str, Any], metric: str) -> float | None:
    return _summary_mean(row.get("metrics", {}).get(metric))


def _summary_mean(summary: Any) -> float | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("mean")
    if value is None:
        return None
    return float(value)


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _format_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


def _component_status(components: Mapping[str, Any], key: str) -> str:
    component = components.get(key, {})
    if not isinstance(component, Mapping):
        return "not_available"
    return str(component.get("status", "not_available"))


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _has_replicate_specific_trial_inputs(trial_specs: Any, n_replicates: int) -> bool:
    return any(
        key.startswith("feedback_ablation:") and getattr(value, "shape", ())[:1] == (n_replicates,)
        for key, value in trial_specs.inputs.items()
    )


def _select_replicate_trial_inputs(trial_specs: Any, replicate: int, n_replicates: int) -> Any:
    inputs = {}
    for key, value in trial_specs.inputs.items():
        if key.startswith("feedback_ablation:") and getattr(value, "shape", ())[:1] == (
            n_replicates,
        ):
            inputs[key] = value[replicate]
        else:
            inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _select_replicate_tree(tree: Any, replicate: int, n_replicates: int) -> Any:
    return jt.map(
        lambda leaf: leaf[replicate] if is_replicate_array(leaf, n_replicates) else leaf,
        tree,
    )


def _infer_batch_size(trial_specs: Any) -> int:
    for values in (trial_specs.inputs.values(), trial_specs.inits.values()):
        for value in values:
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) >= 1:
                return int(shape[0])
            pos = getattr(value, "pos", None)
            if pos is not None:
                return int(pos.shape[0])
    raise ValueError("could not infer trial batch size")


__all__ = [
    "AblationMode",
    "DEFAULT_OUTPUT_FILENAME",
    "DEFAULT_RESULT_EXPERIMENT",
    "DEFAULT_RUN_IDS",
    "DEFAULT_SCOPE",
    "DetailedRolloutEvaluation",
    "FEEDBACK_ABLATION_ANALYSIS_TYPE",
    "FEEDBACK_AUDIT_SELECTION_ROLE",
    "FeedbackAblationAnalysis",
    "FeedbackAblationAnalysisParams",
    "ObservationAblationSpec",
    "SCHEMA_VERSION",
    "build_observation_ablation_spec",
    "build_observation_tape",
    "default_ablation_modes",
    "evaluate_feedback_ablation_runs",
    "evaluate_model_on_trial_specs",
    "feedback_ablation_evaluation_spec",
    "feedback_ablation_recipe",
    "feedback_ablation_spec",
    "feedback_checkpoint_selection_audit",
    "insert_observation_ablation",
    "interpret_run_feedback_ablation",
    "render_feedback_ablation_markdown",
    "selected_feedback_ablation_bins",
    "summarize_ablation_delta",
    "summarize_feedback_pass_audit",
    "summarize_normalized_feedback_use",
]

"""Feedbax declarative recipes for rlrmp certificate materializers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from feedbax.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts
from feedbax.analysis.context import AnalysisArtifactFile, AnalysisRunContext
from feedbax.analysis.materialization import (
    AnalysisArtifactGroup,
    ExistingAnalysisArtifact,
    MaterializationResult,
    materialization_metadata,
)
from feedbax.analysis.specs import AnalysisRecipeResult, register_analysis_recipe
from feedbax.contracts.expressions import (
    AllOf,
    AnyOf,
    Compare,
    ContextItem,
    Expr,
    ExpressionContext,
    Not,
    canonical_expression_json,
    evaluate_expr,
)
from feedbax.contracts.manifest import AnalysisRunSpec, ParentRef
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from pydantic import BaseModel, ConfigDict, Field

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import (
    MATERIALIZER_ISSUE_ID,
    RUN_IDS,
    SOURCE_ISSUE_ID,
    materialize_gru_standard_result_from_evaluation_states,
    materialize_gru_standard_result,
    write_gru_standard_result,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import (
    DEFAULT_JACOBIAN_TIMEPOINTS,
    DEFAULT_N_ROLLOUT_TRIALS,
    DEFAULT_OUTPUT_FILENAME,
    materialize_gru_evaluation_diagnostics,
)
from rlrmp.analysis.pipelines.gru_postrun_materialization import (
    DEFAULT_OUTPUT_TAG,
    materialize_gru_postrun_analysis,
    materialize_optional_feedback_ablation,
    materialize_optional_objective_comparator,
    materialize_optional_perturbation_response,
    plan_gru_postrun_materialization,
)
from rlrmp.analysis.pipelines.hinf_phenotype_sidecar import (
    DEFAULT_SCOPE as DEFAULT_HINF_PHENOTYPE_SCOPE,
    build_hinf_phenotype_sidecar,
    load_hinf_phenotype_sources,
    write_hinf_phenotype_sidecar,
)
from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
    ISSUE_ID as OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ISSUE_ID,
    write_outputs as write_output_feedback_rollout_recovery_outputs,
)
from rlrmp.analysis.math.rerun_metadata import DEFAULT_DISCRETIZATION, DEFAULT_LANE
from rlrmp.eval.recipes import (
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
)
from rlrmp.eval.policy_diagnostics import (
    PolicyAbsentInputBlock,
    PolicyInputSchema,
    directional_gain_summary,
    feedback_jacobian_sisu_modulation,
    policy_jacobian,
    signed_pair_odd_even_summary,
    singular_value_summary,
    validate_policy_jacobian,
)
from rlrmp.eval.recurrent_jacobians import compute_recurrent_jacobian_bank
from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.params_models import register_params_model
from rlrmp.runtime.spec_migrations import (
    GRU_PERTURBATION_BANK_SCHEMA_VERSION,
    PERTURBATION_CLASS_RESPONSE_SCHEMA_ID,
    PERTURBATION_CLASS_RESPONSE_SCHEMA_VERSION,
)


GRU_STANDARD_ANALYSIS_TYPE = "rlrmp.certificate.gru_standard"
GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE = "rlrmp.diagnostic.gru_evaluation"
GRU_POSTRUN_ANALYSIS_TYPE = "rlrmp.gru_postrun"
PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE = "rlrmp.perturbation_class_response"
PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE = "rlrmp.perturbation_bank_aggregate"
POLICY_DIAGNOSTICS_ANALYSIS_TYPE = "rlrmp.diagnostic.policy_local"
RECURRENT_JACOBIAN_ANALYSIS_TYPE = "rlrmp.diagnostic.recurrent_jacobian"
FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE = "rlrmp.feedback_quality_lens"
ROBUSTNESS_PHENOTYPE_ANALYSIS_TYPE = "rlrmp.robustness_phenotype"
OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE = (
    "rlrmp.output_feedback_bridge.rollout_recovery"
)
BRIDGE_STANDARD_ANALYSIS_TYPE = GRU_STANDARD_ANALYSIS_TYPE

ROBUSTNESS_PHENOTYPE_ISSUE_ID = "769aea6"

ROBUSTNESS_PHENOTYPE_SOURCE_ROLES = {
    "rlrmp-bridge-standard-certificate": "standard_certificate",
    "rlrmp-bridge-standard-certificate-manifest": "standard_certificate",
    "rlrmp-gru-standard-certificate-manifest": "standard_certificate",
    "rlrmp-gru-objective-comparator-manifest": "objective_comparator",
    "rlrmp-gru-perturbation-response-manifest": "perturbation_response",
    "rlrmp-gru-feedback-ablation-manifest": "feedback_ablation",
    "rlrmp-gru-map-decomposition-manifest": "map_error_decomposition",
    "rlrmp-gru-evaluation-diagnostics-manifest": "evaluation_diagnostics",
    "rlrmp-gru-worst-case-epsilon-audit-manifest": "worst_case_epsilon_audit",
    "rlrmp-gru-broad-epsilon-attribution-manifest": "broad_epsilon_attribution",
    "rlrmp-induced-gain-manifest": "induced_gain",
    "rlrmp-exact-audit-manifest": "exact_audit",
}

EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE = {
    GRU_STANDARD_ANALYSIS_TYPE: (CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,),
    GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE: ("evaluation_run",),
    GRU_POSTRUN_ANALYSIS_TYPE: ("evaluation_run",),
    PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE: (
        PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
    ),
    PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE: ("analysis_run",),
    POLICY_DIAGNOSTICS_ANALYSIS_TYPE: ("evaluation_run",),
    RECURRENT_JACOBIAN_ANALYSIS_TYPE: ("evaluation_run",),
    FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE: ("analysis_run",),
    ROBUSTNESS_PHENOTYPE_ANALYSIS_TYPE: ("evaluation_run",),
}

FEEDBACK_QUALITY_COMPONENT_NAMES = (
    "evaluation_diagnostics",
    "objective_comparator",
    "perturbation_response",
    "feedback_ablation",
    "response_norm_plots",
    "perturbation_calibration",
)
FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES = {
    name: f"rlrmp.feedback_quality.{name}" for name in FEEDBACK_QUALITY_COMPONENT_NAMES
}
FEEDBACK_QUALITY_COMPONENT_STATUS_ROLES = {
    name: f"rlrmp-feedback-quality-{name.replace('_', '-')}-status"
    for name in FEEDBACK_QUALITY_COMPONENT_NAMES
}

EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE.update(
    {
        FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["evaluation_diagnostics"]: (
            "evaluation_run",
            "params.materialize_evaluation_diagnostics",
        ),
        FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["objective_comparator"]: (
            "evaluation_run",
            "params.materialize_objective_comparator",
        ),
        FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["perturbation_response"]: (
            "evaluation_run",
            "params.materialize_perturbation_response",
        ),
        FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["feedback_ablation"]: (
            "evaluation_run",
            "params.materialize_feedback_ablation",
        ),
        FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["response_norm_plots"]: (
            "rlrmp-feedback-quality-perturbation-response-manifest",
        ),
        FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["perturbation_calibration"]: (
            "params.materialize_perturbation_calibration",
        ),
    }
)


FeedbackQualityMaterializer = Callable[
    [AnalysisRunContext, Mapping[str, Any], str, tuple[str, ...], Any | None, Path],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class FeedbackQualityComponentRegistration:
    """Registered feedback-quality component leaf metadata."""

    name: str
    materializer: FeedbackQualityMaterializer
    artifact_role: str
    logical_name: str
    live_materializer: str
    gating_label: str
    gating_expr: Expr


@dataclass(frozen=True)
class FeedbackQualityGatingDecision:
    """Evaluated feedback-quality component gate state."""

    included: bool
    not_applicable: bool
    eligible: bool
    should_materialize: bool


@dataclass(frozen=True)
class _AnalysisEvaluationInput:
    """Minimal resolved evaluation input reconstructed from AnalysisInputData."""

    states: Mapping[str, Any] | None
    path: Path | None
    ref: ParentRef | None
    manifest: Any | None = None


class PolicyDiagnosticsAnalysisParams(BaseModel):
    """Params for controller-local policy diagnostic-bank analysis."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str | None = None
    schema_version: str | None = None
    source_key: str = "policy_diagnostics"
    row_ids: list[str] | None = None
    include_arrays: bool = False
    include_finite_difference: bool = False
    finite_difference_epsilon: float = Field(1e-3, gt=0.0)
    finite_difference_atol: float = Field(1e-4, ge=0.0)
    finite_difference_rtol: float = Field(1e-3, ge=0.0)
    finite_difference_batch_size: int | None = Field(128, ge=1)
    feedback_block: str = "feedback"
    sisu_block: str = "sisu"
    sisu_values: list[float] | None = None


class RecurrentJacobianAnalysisParams(BaseModel):
    """Params for staged recurrent Jacobian diagnostic-bank analysis."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str | None = None
    schema_version: str | None = None
    source_key: str = "recurrent_jacobians"
    row_ids: list[str] | None = None
    include_arrays: bool = False
    include_finite_difference: bool = False
    finite_difference_epsilon: float = Field(1e-4, gt=0.0)
    finite_difference_batch_size: int | None = Field(128, ge=1)


class RLRMPManifestAnalysis(AbstractAnalysis):
    """Context-aware rlrmp analysis node that records Feedbax-owned artifacts."""

    materializer: Callable[[AnalysisRunContext, AnalysisInputData], Any | MaterializationResult] = (
        eqx.field(kw_only=True, static=True)
    )
    artifact_role: str = eqx.field(kw_only=True, static=True)
    logical_name: str = eqx.field(kw_only=True, static=True)
    schema_boundary: str | None = eqx.field(default=None, static=True)
    metadata: dict[str, Any] = eqx.field(default_factory=dict, static=True)

    @property
    def _field_params(self) -> dict[str, Any]:
        return {
            "artifact_role": self.artifact_role,
            "logical_name": self.logical_name,
            "schema_boundary": self.schema_boundary,
            "metadata": self.metadata,
        }

    def compute(self, data: AnalysisInputData, **kwargs: Any) -> dict[str, Any]:
        del data, kwargs
        return {
            "status": "pending_context_artifact_emission",
            "artifact_role": self.artifact_role,
            "logical_name": self.logical_name,
        }

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        del result, kwargs
        materialized = self.materializer(context, data)
        if not isinstance(materialized, MaterializationResult):
            materialized = MaterializationResult(payload=materialized)
        payload = materialized.payload
        metadata = {**self.metadata, **materialized.payload_metadata}
        if self.schema_boundary is not None:
            metadata.setdefault("schema_boundary", self.schema_boundary)

        context.record_artifact_refs_from_value(payload)
        payload_ref = context.record_json_artifact(
            payload,
            role=self.artifact_role,
            logical_name=self.logical_name,
            metadata=metadata,
        )
        context.record_artifact_refs([payload_ref, *materialized.artifact_refs])
        for artifact in materialized.existing_artifacts:
            context.record_artifact(
                artifact.path,
                role=artifact.role,
                logical_name=artifact.logical_name,
                media_type=artifact.media_type,
                metadata=artifact.metadata,
                group_id=artifact.group_id,
                group_role=artifact.group_role,
                group_metadata=artifact.group_metadata,
            )
        for group in materialized.artifact_groups:
            context.record_artifact_group(
                group_id=group.group_id,
                members=group.members,
                metadata=group.metadata,
            )
        context.record_regeneration_specs(materialized.regeneration_specs)
        return payload


class PerturbationClassResponseAnalysis(AbstractAnalysis):
    """Slice one perturbation family from a shared perturbation-bank eval payload."""

    params: dict[str, Any] = eqx.field(kw_only=True, static=True)
    evaluation_input: Any = eqx.field(kw_only=True, static=True)

    def compute(self, data: AnalysisInputData, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        states = _resolved_input_states(self.evaluation_input)
        if states is None and isinstance(data.states, Mapping):
            maybe_states = data.states.get("evaluation")
            states = maybe_states if isinstance(maybe_states, Mapping) else None
        if states is None:
            raise ValueError("perturbation class response requires evaluation states")
        return _perturbation_class_response_payload(
            states,
            self.params,
            evaluation_input=self.evaluation_input,
        )

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del data, kwargs
        payload = dict(result)
        family = str(payload["family"])
        context.record_json_artifact(
            payload,
            role="rlrmp-perturbation-class-response",
            logical_name=f"perturbation_class_response/{family}.json",
            metadata={
                "schema_boundary": "rlrmp-owned perturbation class response payload",
                "family": family,
                "evaluation_manifest_id": payload["evaluation_manifest"]["id"],
            },
        )
        return payload


class PerturbationBankAggregateAnalysis(AbstractAnalysis):
    """Aggregate perturbation-family leaf products into the legacy bank payload."""

    params: dict[str, Any] = eqx.field(kw_only=True, static=True)
    leaf_products: tuple[dict[str, Any], ...] = eqx.field(kw_only=True, static=True)

    def compute(self, data: AnalysisInputData, **kwargs: Any) -> dict[str, Any]:
        del data, kwargs
        return _aggregate_perturbation_class_products(self.leaf_products, self.params)

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del data, kwargs
        payload = dict(result)
        context.record_json_artifact(
            payload,
            role="rlrmp-gru-perturbation-response-manifest",
            logical_name="gru_perturbation_response_aggregate.json",
            metadata={
                "schema_boundary": "rlrmp-owned GRU perturbation bank payload",
                "schema_version": GRU_PERTURBATION_BANK_SCHEMA_VERSION,
                "source": "perturbation_class_response_leaves",
            },
        )
        return payload


class FeedbackQualityLensPorts(AbstractAnalysisPorts):
    """Dependency ports for the feedback-quality summary node."""

    evaluation_diagnostics: Any = None
    objective_comparator: Any = None
    perturbation_response: Any = None
    feedback_ablation: Any = None
    response_norm_plots: Any = None
    perturbation_calibration: Any = None


class FeedbackQualitySummaryAnalysis(AbstractAnalysis[FeedbackQualityLensPorts]):
    """Aggregate feedback-quality component nodes into the lens payload."""

    Ports = FeedbackQualityLensPorts
    inputs: FeedbackQualityLensPorts = eqx.field(
        default_factory=FeedbackQualityLensPorts,
        converter=FeedbackQualityLensPorts.converter,
    )
    component_outputs: dict[str, dict[str, Any]] = eqx.field(kw_only=True, static=True)
    params: dict[str, Any] = eqx.field(kw_only=True, static=True)
    experiment: str = eqx.field(kw_only=True, static=True)
    run_ids: tuple[str, ...] = eqx.field(kw_only=True, static=True)
    output_tag: str = eqx.field(kw_only=True, static=True)
    plan_checkpoint_policy: str = eqx.field(kw_only=True, static=True)
    plan_checkpoint_selection_source: str = eqx.field(kw_only=True, static=True)

    def compute(
        self,
        data: AnalysisInputData,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del data, kwargs
        outputs = {name: dict(self.component_outputs[name]) for name in FEEDBACK_QUALITY_COMPONENT_NAMES}
        return {
            "schema_id": "rlrmp.feedback_quality_lens",
            "schema_version": "rlrmp.feedback_quality_lens.v1",
            "issue": str(self.params.get("issue", "af77a06")),
            "scope": "feedback_control_quality_diagnostics",
            "experiment": self.experiment,
            "run_ids": list(self.run_ids),
            "labels": _optional_str_sequence(self.params.get("labels")),
            "output_tag": self.output_tag,
            "checkpoint_policy": self.plan_checkpoint_policy,
            "checkpoint_selection_source": self.plan_checkpoint_selection_source,
            "selection_leakage_guard": {
                "status": "audit_only",
                "primary_checkpoint_selection": self.plan_checkpoint_selection_source,
                "feedback_quality_components": sorted(outputs),
                "note": (
                    "Feedback-control quality diagnostics are audit sidecars; they do "
                    "not silently replace the explicitly selected checkpoints."
                ),
            },
            "outputs": outputs,
            "bundle_contract": {
                "primary": "feedbax_analysis_bundle",
                "bundle": "rlrmp/feedback_quality_lens",
                "scientific_schema_owner": "rlrmp",
                "artifact_custody": "feedbax.AnalysisRunManifest",
                "component_execution": "component_abstract_analysis_nodes",
            },
        }

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del data, kwargs
        payload = {
            **dict(result),
            "bundle_contract": {
                **dict(result.get("bundle_contract", {})),
                "analysis_manifest_id": context.manifest_id,
            },
            "declarative_analysis": _declarative_metadata(context),
        }
        context.record_json_artifact(
            payload,
            role="rlrmp-feedback-quality-lens",
            logical_name="feedback_quality_lens.json",
            metadata={"schema_boundary": "rlrmp-owned feedback-control quality lens payload"},
        )
        return payload


class RobustnessPhenotypeAnalysis(AbstractAnalysis):
    """Build and record the robustness phenotype sidecar through Feedbax custody."""

    params: dict[str, Any] = eqx.field(kw_only=True, static=True)
    resolved_inputs: tuple[Any, ...] = eqx.field(default=(), kw_only=True, static=True)

    def compute(self, data: AnalysisInputData, **kwargs: Any) -> dict[str, Any]:
        del data, kwargs
        repo_root = _repo_root_from_params(self.params)
        source_paths = _robustness_phenotype_source_paths(
            self.params,
            self.resolved_inputs,
            repo_root=repo_root,
        )
        sources = load_hinf_phenotype_sources(source_paths, repo_root=repo_root)
        return build_hinf_phenotype_sidecar(
            sources=sources,
            issue=str(self.params.get("issue_id", ROBUSTNESS_PHENOTYPE_ISSUE_ID)),
            scope=str(self.params.get("scope", DEFAULT_HINF_PHENOTYPE_SCOPE)),
            generated_by="rlrmp.analysis.declarative_materialization.robustness_phenotype",
        )

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del data, kwargs
        payload = {
            **dict(result),
            "declarative_analysis": _declarative_metadata(context),
            "bundle_contract": {
                "primary": "feedbax_analysis_bundle",
                "bundle": "rlrmp/robustness_phenotype",
                "analysis_manifest_id": context.manifest_id,
                "schema_owner": "rlrmp",
                "formal_claim_policy": "conservative_no_upgrade_without_formal_inputs",
                "artifact_custody": "feedbax.AnalysisRunManifest",
            },
        }
        repo_root = _repo_root_from_params(self.params)
        json_path = _optional_path(self.params.get("output_json"), repo_root=repo_root) or (
            context.results_cache_dir / "hinf_phenotype_sidecar.json"
        )
        markdown_path = _optional_path(self.params.get("output_markdown"), repo_root=repo_root) or (
            context.results_cache_dir / "hinf_phenotype_sidecar.md"
        )
        regeneration_spec_path = _optional_path(
            self.params.get("regeneration_spec_path"),
            repo_root=repo_root,
        )
        write_hinf_phenotype_sidecar(
            payload,
            json_path=json_path,
            markdown_path=markdown_path,
            regeneration_spec_path=regeneration_spec_path,
            repo_root=repo_root,
        )
        payload = _read_json_payload(json_path)
        context.record_json_artifact(
            payload,
            role="rlrmp-robustness-phenotype-sidecar",
            logical_name="hinf_phenotype_sidecar.json",
            metadata={"schema_boundary": "rlrmp-owned H-infinity phenotype sidecar payload"},
        )
        for artifact in (
            _existing_file(
                json_path,
                role="rlrmp-robustness-phenotype-sidecar-json",
                logical_name=_legacy_logical_name(json_path, repo_root),
            ),
            _existing_file(
                markdown_path,
                role="rlrmp-robustness-phenotype-sidecar-note",
                logical_name=_legacy_logical_name(markdown_path, repo_root),
            ),
            _existing_file(
                regeneration_spec_path,
                role="rlrmp-robustness-phenotype-regeneration-spec",
                logical_name=_legacy_logical_name(regeneration_spec_path, repo_root),
            )
            if regeneration_spec_path is not None
            else None,
        ):
            if artifact is None:
                continue
            context.record_artifact(
                artifact.path,
                role=artifact.role,
                logical_name=artifact.logical_name,
                media_type=artifact.media_type,
                metadata=artifact.metadata,
                group_id=artifact.group_id,
                group_role=artifact.group_role,
                group_metadata=artifact.group_metadata,
            )
        return payload


def register_certificate_analysis_recipes(*, replace: bool = False) -> None:
    """Register rlrmp certificate/diagnostic analysis recipes with Feedbax."""

    register_params_model(
        POLICY_DIAGNOSTICS_ANALYSIS_TYPE,
        PolicyDiagnosticsAnalysisParams,
        replace=True,
    )
    register_params_model(
        RECURRENT_JACOBIAN_ANALYSIS_TYPE,
        RecurrentJacobianAnalysisParams,
        replace=True,
    )
    register_analysis_recipe(
        GRU_STANDARD_ANALYSIS_TYPE,
        gru_standard_certificate_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE,
        gru_evaluation_diagnostics_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        GRU_POSTRUN_ANALYSIS_TYPE,
        gru_postrun_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE,
        perturbation_class_response_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE,
        perturbation_bank_aggregate_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        POLICY_DIAGNOSTICS_ANALYSIS_TYPE,
        policy_diagnostics_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        RECURRENT_JACOBIAN_ANALYSIS_TYPE,
        recurrent_jacobian_recipe,
        replace=replace,
    )
    for registration in _feedback_quality_component_registrations().values():
        register_analysis_recipe(
            FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES[registration.name],
            _feedback_quality_component_recipe(registration.name),
            replace=replace,
        )
    register_analysis_recipe(
        FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE,
        feedback_quality_lens_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        ROBUSTNESS_PHENOTYPE_ANALYSIS_TYPE,
        robustness_phenotype_recipe,
        replace=replace,
    )
    register_analysis_recipe(
        OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE,
        output_feedback_rollout_recovery_recipe,
        replace=replace,
    )


def register_declarative_materialization_recipes(*, replace: bool = False) -> None:
    """Register all RLRMP declarative materialization recipes."""

    register_certificate_analysis_recipes(replace=replace)


def gru_standard_certificate_spec(
    *,
    run_ids: Sequence[str] = RUN_IDS,
    experiment: str = SOURCE_ISSUE_ID,
    materializer_issue_id: str = MATERIALIZER_ISSUE_ID,
    load_models: bool = True,
    use_validation_selected_checkpoints: bool = False,
    preferred_checkpoint_manifest_path: Path | str | None = None,
    note_output: Path | str | None = None,
    manifest_output: Path | str | None = None,
    regeneration_spec_path: Path | str | None = None,
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for the GRU standard-certificate materializer."""

    params = {
        "run_ids": list(run_ids),
        "experiment": experiment,
        "materializer_issue_id": materializer_issue_id,
        "load_models": load_models,
        "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
    }
    _set_optional_path_param(
        params, "preferred_checkpoint_manifest_path", preferred_checkpoint_manifest_path
    )
    _set_optional_path_param(params, "note_output", note_output)
    _set_optional_path_param(params, "manifest_output", manifest_output)
    _set_optional_path_param(params, "regeneration_spec_path", regeneration_spec_path)
    _set_optional_path_param(params, "repo_root", repo_root)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=GRU_STANDARD_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def gru_evaluation_diagnostics_spec(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    output_path: Path | str | None = None,
    bulk_dir: Path | str | None = None,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    use_validation_selected_checkpoints: bool = True,
    preferred_checkpoint_manifest_path: Path | str | None = None,
    jacobian_timepoints: Sequence[str] = DEFAULT_JACOBIAN_TIMEPOINTS,
    write_bulk_arrays: bool = True,
    regeneration_spec_path: Path | str | None = None,
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for GRU rollout diagnostics."""

    params = {
        "experiment": experiment,
        "run_ids": list(run_ids),
        "n_rollout_trials": n_rollout_trials,
        "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
        "jacobian_timepoints": list(jacobian_timepoints),
        "write_bulk_arrays": write_bulk_arrays,
    }
    if labels is not None:
        params["labels"] = list(labels)
    _set_optional_path_param(params, "output_path", output_path)
    _set_optional_path_param(params, "bulk_dir", bulk_dir)
    _set_optional_path_param(
        params, "preferred_checkpoint_manifest_path", preferred_checkpoint_manifest_path
    )
    _set_optional_path_param(params, "regeneration_spec_path", regeneration_spec_path)
    _set_optional_path_param(params, "repo_root", repo_root)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def gru_postrun_spec(
    *,
    experiment: str,
    run_ids: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    output_tag: str = DEFAULT_OUTPUT_TAG,
    use_validation_selected_checkpoints: bool = True,
    include_reference: bool = True,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    include_objective_comparator: bool = True,
    include_map_decomposition: bool = True,
    include_perturbation_response: bool = True,
    include_feedback_ablation: bool = True,
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for the complete GRU post-run bundle."""

    params: dict[str, Any] = {
        "experiment": experiment,
        "output_tag": output_tag,
        "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
        "include_reference": include_reference,
        "n_rollout_trials": n_rollout_trials,
        "include_objective_comparator": include_objective_comparator,
        "include_map_decomposition": include_map_decomposition,
        "include_perturbation_response": include_perturbation_response,
        "include_feedback_ablation": include_feedback_ablation,
    }
    if run_ids is not None:
        params["run_ids"] = list(run_ids)
    if labels is not None:
        params["labels"] = list(labels)
    _set_optional_path_param(params, "repo_root", repo_root)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=GRU_POSTRUN_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def perturbation_class_response_spec(
    *,
    family: str,
    row_ids: Sequence[str] | None = None,
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    expected_calibration_identity: Mapping[str, Any] | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for one perturbation-class response leaf."""

    params: dict[str, Any] = {"family": family}
    if row_ids is not None:
        params["row_ids"] = list(row_ids)
    if expected_calibration_identity is not None:
        params["expected_calibration_identity"] = dict(expected_calibration_identity)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def perturbation_bank_aggregate_spec(
    *,
    leaf_manifest_refs: Sequence[ParentRef] = (),
    issue: str | None = None,
    source_experiment: str | None = None,
    bank_mode: str | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for aggregating perturbation-class leaves."""

    params: dict[str, Any] = {}
    if issue is not None:
        params["issue"] = issue
    if source_experiment is not None:
        params["source_experiment"] = source_experiment
    if bank_mode is not None:
        params["bank_mode"] = bank_mode
    return AnalysisRunSpec(
        analysis_type=PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE,
        inputs=list(leaf_manifest_refs),
        params=params,
    )


def policy_diagnostics_spec(
    *,
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    row_ids: Sequence[str] | None = None,
    include_arrays: bool = False,
    include_finite_difference: bool = False,
    sisu_values: Sequence[float] | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for controller-local policy diagnostics."""

    params = PolicyDiagnosticsAnalysisParams(
        row_ids=None if row_ids is None else [str(row_id) for row_id in row_ids],
        include_arrays=include_arrays,
        include_finite_difference=include_finite_difference,
        sisu_values=None if sisu_values is None else [float(value) for value in sisu_values],
    ).model_dump(mode="json", exclude_none=True)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=POLICY_DIAGNOSTICS_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def recurrent_jacobian_spec(
    *,
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    row_ids: Sequence[str] | None = None,
    include_arrays: bool = False,
    include_finite_difference: bool = False,
) -> AnalysisRunSpec:
    """Return declarative spec data for staged recurrent Jacobian diagnostics."""

    params = RecurrentJacobianAnalysisParams(
        row_ids=None if row_ids is None else [str(row_id) for row_id in row_ids],
        include_arrays=include_arrays,
        include_finite_difference=include_finite_difference,
    ).model_dump(mode="json", exclude_none=True)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=RECURRENT_JACOBIAN_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def feedback_quality_lens_spec(
    *,
    experiment: str | None = None,
    run_ids: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    output_tag: str = DEFAULT_OUTPUT_TAG,
    use_validation_selected_checkpoints: bool = True,
    include_evaluation_diagnostics: bool = True,
    include_objective_comparator: bool = True,
    include_perturbation_response: bool = True,
    include_feedback_ablation: bool = True,
    include_response_norm_plots: bool = True,
    include_perturbation_calibration: bool = True,
    not_applicable_components: Sequence[str] = (),
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for the feedback-control quality lens."""

    params: dict[str, Any] = {
        "output_tag": output_tag,
        "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
        "include_evaluation_diagnostics": include_evaluation_diagnostics,
        "include_objective_comparator": include_objective_comparator,
        "include_perturbation_response": include_perturbation_response,
        "include_feedback_ablation": include_feedback_ablation,
        "include_response_norm_plots": include_response_norm_plots,
        "include_perturbation_calibration": include_perturbation_calibration,
        "not_applicable_components": list(not_applicable_components),
    }
    if experiment is not None:
        params["experiment"] = experiment
    if run_ids is not None:
        params["run_ids"] = list(run_ids)
    if labels is not None:
        params["labels"] = list(labels)
    _set_optional_path_param(params, "repo_root", repo_root)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def output_feedback_rollout_recovery_spec(
    *,
    issue_id: str = OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ISSUE_ID,
    discretization: str | None = None,
    lane: str | None = None,
    note_output: Path | str | None = None,
    manifest_output: Path | str | None = None,
    artifact_output: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for the output-feedback rollout-recovery diagnostic."""

    params = {
        "issue_id": issue_id,
    }
    if discretization is not None:
        params["discretization"] = discretization
    if lane is not None:
        params["lane"] = lane
    _set_optional_path_param(params, "note_output", note_output)
    _set_optional_path_param(params, "manifest_output", manifest_output)
    _set_optional_path_param(params, "artifact_output", artifact_output)
    _set_optional_path_param(params, "repo_root", repo_root)
    return AnalysisRunSpec(
        analysis_type=OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE,
        params=params,
    )


def robustness_phenotype_spec(
    *,
    source_paths: Mapping[str, Path | str | None] | None = None,
    issue_id: str = ROBUSTNESS_PHENOTYPE_ISSUE_ID,
    scope: str = DEFAULT_HINF_PHENOTYPE_SCOPE,
    output_json: Path | str | None = None,
    output_markdown: Path | str | None = None,
    regeneration_spec_path: Path | str | None = None,
    evaluation_manifest_id: str | None = None,
    evaluation_manifest_uri: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> AnalysisRunSpec:
    """Return declarative spec data for the robustness phenotype sidecar."""

    params: dict[str, Any] = {
        "issue_id": issue_id,
        "scope": scope,
    }
    if source_paths is not None:
        params["source_paths"] = {
            str(name): None if path is None else str(path)
            for name, path in source_paths.items()
        }
    _set_optional_path_param(params, "output_json", output_json)
    _set_optional_path_param(params, "output_markdown", output_markdown)
    _set_optional_path_param(params, "regeneration_spec_path", regeneration_spec_path)
    _set_optional_path_param(params, "repo_root", repo_root)
    inputs = _evaluation_parent_refs(
        evaluation_manifest_id=evaluation_manifest_id,
        evaluation_manifest_uri=evaluation_manifest_uri,
    )
    return AnalysisRunSpec(
        analysis_type=ROBUSTNESS_PHENOTYPE_ANALYSIS_TYPE,
        inputs=inputs,
        params=params,
    )


def gru_standard_certificate_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative GRU standard-certificate recipe."""

    params = dict(spec.params)
    evaluation_input = _primary_evaluation_input(inputs)
    analysis = RLRMPManifestAnalysis(
        materializer=lambda context, data: _materialize_gru_standard(
            context,
            params,
            evaluation_input=_evaluation_input_from_analysis_data(data),
        ),
        artifact_role="rlrmp-bridge-standard-certificate",
        logical_name="gru_standard_certificates.json",
        schema_boundary="rlrmp-owned BridgeRunManifest/certificate payload",
    )
    return AnalysisRecipeResult(
        analyses={"gru_standard_certificate": analysis},
        data=_analysis_data_from_evaluation_input(evaluation_input),
    )


def gru_evaluation_diagnostics_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative GRU rollout-diagnostics recipe."""

    params = dict(spec.params)
    evaluation_input = _primary_evaluation_input(inputs)
    analysis = RLRMPManifestAnalysis(
        materializer=lambda context, data: _materialize_gru_evaluation_diagnostics(
            context,
            params,
            evaluation_input=_evaluation_input_from_analysis_data(data),
        ),
        artifact_role="rlrmp-gru-evaluation-diagnostics",
        logical_name="gru_evaluation_diagnostics.json",
        schema_boundary="rlrmp-owned GRU diagnostic payload",
    )
    return AnalysisRecipeResult(
        analyses={"gru_evaluation_diagnostics": analysis},
        data=_analysis_data_from_evaluation_input(evaluation_input),
    )


def gru_postrun_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative complete GRU post-run materialization recipe."""

    params = dict(spec.params)
    resolved_run_ids = _run_ids_from_params_or_inputs(params, inputs)
    experiment = _experiment_from_params_or_inputs(params, inputs)
    evaluation_input = _primary_evaluation_input(inputs)
    analysis = RLRMPManifestAnalysis(
        materializer=lambda context, data: _materialize_gru_postrun(
            context,
            params,
            experiment=experiment,
            run_ids=resolved_run_ids,
            evaluation_input=_evaluation_input_from_analysis_data(data),
        ),
        artifact_role="rlrmp-gru-postrun-manifest",
        logical_name="gru_postrun_materialization.json",
        schema_boundary="rlrmp-owned GRU post-run diagnostic bundle payload",
    )
    return AnalysisRecipeResult(
        analyses={"gru_postrun_materialization": analysis},
        data=_analysis_data_from_evaluation_input(evaluation_input),
    )


def perturbation_class_response_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build one perturbation-family response analysis from shared eval states."""

    params = dict(spec.params)
    evaluation_input = _primary_evaluation_input(inputs)
    if evaluation_input is None:
        raise ValueError("perturbation class response requires an EvaluationRunManifest input")
    analysis = PerturbationClassResponseAnalysis(
        params=params,
        evaluation_input=evaluation_input,
    )
    family = str(params.get("family", "perturbation_class"))
    return AnalysisRecipeResult(
        analyses={family: analysis},
        data=_analysis_data_from_evaluation_input(evaluation_input),
    )


def perturbation_bank_aggregate_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build an aggregate perturbation bank from class-response leaf products."""

    params = dict(spec.params)
    leaf_products = tuple(_load_perturbation_class_product(resolved) for resolved in inputs)
    analysis = PerturbationBankAggregateAnalysis(
        params=params,
        leaf_products=leaf_products,
    )
    return AnalysisRecipeResult(
        analyses={"perturbation_bank_aggregate": analysis},
        data=_empty_analysis_data(),
    )


def policy_diagnostics_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build controller-local policy diagnostic-bank analysis from eval states."""

    params = PolicyDiagnosticsAnalysisParams.model_validate(spec.params)
    evaluation_input = _primary_evaluation_input(inputs)
    if evaluation_input is None:
        raise ValueError("policy diagnostics analysis requires an EvaluationRunManifest input")
    analysis = RLRMPManifestAnalysis(
        materializer=lambda context, data: _materialize_policy_diagnostics(
            context,
            params,
            evaluation_input=_evaluation_input_from_analysis_data(data),
        ),
        artifact_role="rlrmp-policy-diagnostics-bank",
        logical_name="policy_diagnostics_bank.json",
        schema_boundary="rlrmp-owned controller-local policy diagnostics payload",
    )
    return AnalysisRecipeResult(
        analyses={"policy_diagnostics_bank": analysis},
        data=_analysis_data_from_evaluation_input(evaluation_input),
    )


def recurrent_jacobian_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build staged recurrent Jacobian diagnostic-bank analysis from eval states."""

    params = RecurrentJacobianAnalysisParams.model_validate(spec.params)
    evaluation_input = _primary_evaluation_input(inputs)
    if evaluation_input is None:
        raise ValueError("recurrent Jacobian analysis requires an EvaluationRunManifest input")
    analysis = RLRMPManifestAnalysis(
        materializer=lambda context, data: _materialize_recurrent_jacobians(
            context,
            params,
            evaluation_input=_evaluation_input_from_analysis_data(data),
        ),
        artifact_role="rlrmp-recurrent-jacobian-bank",
        logical_name="recurrent_jacobian_bank.json",
        schema_boundary="rlrmp-owned staged recurrent Jacobian diagnostics payload",
    )
    return AnalysisRecipeResult(
        analyses={"recurrent_jacobian_bank": analysis},
        data=_analysis_data_from_evaluation_input(evaluation_input),
    )


def _feedback_quality_component_recipe(
    component_name: str,
) -> Callable[[AnalysisRunSpec, Path, Sequence[Any]], AnalysisRecipeResult]:
    """Return the registered recipe for one feedback-quality component leaf."""

    def _recipe(
        spec: AnalysisRunSpec,
        _root: Path,
        inputs: Sequence[Any],
    ) -> AnalysisRecipeResult:
        params = dict(spec.params)
        resolved_run_ids = _feedback_quality_run_ids_from_params_or_inputs(params, inputs)
        experiment = _experiment_from_params_or_inputs(params, inputs)
        repo_root = _repo_root_from_params(params)
        evaluation_input = _primary_evaluation_input(inputs)
        registration = _feedback_quality_component_registrations()[component_name]
        analysis = RLRMPManifestAnalysis(
            materializer=lambda context, data: _materialize_feedback_quality_component(
                context,
                data,
                registration=registration,
                params=params,
                experiment=experiment,
                run_ids=resolved_run_ids,
                evaluation_input=_evaluation_input_from_analysis_data(data),
                repo_root=repo_root,
            ),
            artifact_role=registration.artifact_role,
            logical_name=registration.logical_name,
            schema_boundary="rlrmp-owned feedback-quality component status payload",
            metadata={
                "component": registration.name,
                "live_materializer": registration.live_materializer,
                "gating_label": registration.gating_label,
                "gating_expr": registration.gating_expr.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "gating_expr_canonical": canonical_expression_json(registration.gating_expr),
            },
        )
        return AnalysisRecipeResult(
            analyses={registration.name: analysis},
            data=_analysis_data_from_evaluation_input(evaluation_input),
        )

    _recipe.__name__ = f"feedback_quality_{component_name}_recipe"
    return _recipe


def output_feedback_rollout_recovery_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    _inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative output-feedback rollout-recovery recipe."""

    params = dict(spec.params)
    analysis = RLRMPManifestAnalysis(
        materializer=lambda context, data: _materialize_output_feedback_rollout_recovery(
            context,
            params,
        ),
        artifact_role="rlrmp-output-feedback-rollout-recovery",
        logical_name="output_feedback_rollout_recovery.json",
        schema_boundary=(
            "rlrmp-owned output-feedback bridge diagnostic payload; analytical "
            "rollouts stay analysis-internal per e1ad278 Q2"
        ),
    )
    return AnalysisRecipeResult(
        analyses={"output_feedback_rollout_recovery": analysis},
        data=_empty_analysis_data(),
    )


def feedback_quality_lens_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative feedback-control quality lens recipe."""

    params = dict(spec.params)
    resolved_run_ids = _feedback_quality_run_ids_from_params_or_inputs(params, inputs)
    experiment = _experiment_from_params_or_inputs(params, inputs)
    repo_root = _repo_root_from_params(params)
    output_tag = str(params.get("output_tag", DEFAULT_OUTPUT_TAG))
    plan = plan_gru_postrun_materialization(
        experiment=experiment,
        run_ids=tuple(resolved_run_ids),
        output_tag=output_tag,
        use_validation_selected_checkpoints=bool(
            params.get("use_validation_selected_checkpoints", True)
        ),
        fixed_bank_rescore_manifest_path=_optional_path(
            params.get("fixed_bank_rescore_manifest_path"),
            repo_root=repo_root,
        ),
        repo_root=repo_root,
    )
    component_outputs = _feedback_quality_component_outputs_from_inputs(inputs)
    summary = FeedbackQualitySummaryAnalysis(
        component_outputs=component_outputs,
        params=params,
        experiment=experiment,
        run_ids=tuple(resolved_run_ids),
        output_tag=output_tag,
        plan_checkpoint_policy=plan.checkpoint_policy,
        plan_checkpoint_selection_source=plan.checkpoint_selection_source,
    )
    return AnalysisRecipeResult(
        analyses={"feedback_quality_lens": summary},
        data=_empty_analysis_data(),
    )


def robustness_phenotype_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative robustness phenotype sidecar recipe."""

    params = dict(spec.params)
    analysis = RobustnessPhenotypeAnalysis(
        params=params,
        resolved_inputs=tuple(inputs),
    )
    return AnalysisRecipeResult(
        analyses={"robustness_phenotype": analysis},
        data=_analysis_data_from_evaluation_input(_primary_evaluation_input(inputs)),
    )


def _materialize_gru_standard(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    *,
    evaluation_input: Any | None = None,
) -> MaterializationResult:
    run_ids = tuple(str(run_id) for run_id in params.get("run_ids", RUN_IDS))
    experiment = str(params.get("experiment", SOURCE_ISSUE_ID))
    repo_root = _repo_root_from_params(params)
    evaluation_states = _resolved_input_states(evaluation_input)
    if evaluation_states is not None:
        result = materialize_gru_standard_result_from_evaluation_states(
            evaluation_states,
            run_ids=run_ids,
            experiment=experiment,
            materializer_issue_id=str(
                params.get("materializer_issue_id", MATERIALIZER_ISSUE_ID)
            ),
            repo_root=repo_root,
        )
    else:
        result = materialize_gru_standard_result(
            run_ids=run_ids,
            load_models=bool(params.get("load_models", True)),
            experiment=experiment,
            materializer_issue_id=str(
                params.get("materializer_issue_id", MATERIALIZER_ISSUE_ID)
            ),
            use_validation_selected_checkpoints=bool(
                params.get("use_validation_selected_checkpoints", False)
            ),
            preferred_checkpoint_manifest_path=_optional_path(
                params.get("preferred_checkpoint_manifest_path"),
                repo_root=repo_root,
            ),
            repo_root=repo_root,
        )
    note_path = _optional_path(params.get("note_output"), repo_root=repo_root)
    manifest_path = _optional_path(params.get("manifest_output"), repo_root=repo_root)
    existing_artifacts: list[ExistingAnalysisArtifact] = []
    if note_path is not None or manifest_path is not None:
        manifest_output = manifest_path or _default_output_path(
            context,
            "gru_standard_certificates_manifest.json",
        )
        actual_note_path = note_path or manifest_output.with_suffix(".md")
        write_gru_standard_result(
            result,
            note_path=actual_note_path,
            manifest_path=manifest_output,
            regeneration_spec_path=_optional_path(
                params.get("regeneration_spec_path"),
                repo_root=repo_root,
            ),
            repo_root=repo_root,
        )
        result = {
            **_read_json_payload(manifest_output),
            "declarative_analysis": _declarative_metadata(context),
        }
        existing_artifacts.extend(
            artifact
            for artifact in (
                _existing_file(
                    manifest_output,
                    role="rlrmp-bridge-standard-certificate-manifest",
                    logical_name="legacy/gru_standard_certificates_manifest.json",
                ),
                _existing_file(
                    actual_note_path,
                    role="rlrmp-bridge-standard-certificate-note",
                    logical_name="legacy/gru_standard_certificates.md",
                ),
            )
            if artifact is not None
        )
    else:
        result = {
            **result,
            "declarative_analysis": _declarative_metadata(context),
        }
    if evaluation_input is not None:
        result["evaluation_manifest_dependency"] = _evaluation_dependency_metadata(
            evaluation_input
        )
    return MaterializationResult(
        payload=result,
        existing_artifacts=tuple(existing_artifacts),
    )


def _materialize_gru_evaluation_diagnostics(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    *,
    evaluation_input: Any | None = None,
) -> MaterializationResult:
    if "experiment" not in params:
        raise ValueError("GRU evaluation diagnostics recipe requires params.experiment")
    if "run_ids" not in params:
        raise ValueError("GRU evaluation diagnostics recipe requires params.run_ids")
    repo_root = _repo_root_from_params(params)
    output_path = _optional_path(params.get("output_path"), repo_root=repo_root) or (
        context.results_cache_dir / DEFAULT_OUTPUT_FILENAME
    )
    bulk_dir = _optional_path(params.get("bulk_dir"), repo_root=repo_root) or (
        context.results_cache_dir / "bulk"
    )
    manifest = materialize_gru_evaluation_diagnostics(
        experiment=str(params["experiment"]),
        run_ids=[str(run_id) for run_id in params["run_ids"]],
        labels=_optional_str_sequence(params.get("labels")),
        output_path=output_path,
        bulk_dir=bulk_dir,
        n_rollout_trials=int(params.get("n_rollout_trials", DEFAULT_N_ROLLOUT_TRIALS)),
        use_validation_selected_checkpoints=bool(
            params.get("use_validation_selected_checkpoints", True)
        ),
        preferred_checkpoint_manifest_path=_optional_path(
            params.get("preferred_checkpoint_manifest_path"),
            repo_root=repo_root,
        ),
        jacobian_timepoints=tuple(
            str(item) for item in params.get("jacobian_timepoints", DEFAULT_JACOBIAN_TIMEPOINTS)
        ),
        write_bulk_arrays=bool(params.get("write_bulk_arrays", True)),
        regeneration_spec_path=_optional_path(
            params.get("regeneration_spec_path"),
            repo_root=repo_root,
        ),
        evaluation_manifest_path=_resolved_input_path(evaluation_input),
        evaluation_states=_resolved_input_states(evaluation_input),
        repo_root=repo_root,
    )
    existing = _existing_file(
        output_path,
        role="rlrmp-gru-evaluation-diagnostics-manifest",
        logical_name="legacy/gru_evaluation_diagnostics.json",
    )
    artifact_groups = _bulk_artifact_groups(
        manifest,
        group_id="gru_evaluation_diagnostics_bulk",
        repo_root=repo_root,
    )
    return MaterializationResult(
        payload={
            **manifest,
            "declarative_analysis": _declarative_metadata(context),
        },
        existing_artifacts=() if existing is None else (existing,),
        artifact_groups=artifact_groups,
    )


def _materialize_gru_postrun(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    *,
    experiment: str,
    run_ids: Sequence[str],
    evaluation_input: Any | None = None,
) -> MaterializationResult:
    if not run_ids:
        raise ValueError("GRU post-run recipe requires at least one run ID")

    repo_root = _repo_root_from_params(params)
    output_tag = str(params.get("output_tag", DEFAULT_OUTPUT_TAG))
    manifest = materialize_gru_postrun_analysis(
        experiment=experiment,
        run_ids=tuple(run_ids),
        labels=_optional_str_sequence(params.get("labels")),
        output_tag=output_tag,
        use_validation_selected_checkpoints=bool(
            params.get("use_validation_selected_checkpoints", True)
        ),
        fixed_bank_rescore_manifest_path=_optional_path(
            params.get("fixed_bank_rescore_manifest_path"),
            repo_root=repo_root,
        ),
        include_reference=bool(params.get("include_reference", True)),
        n_rollout_trials=int(params.get("n_rollout_trials", DEFAULT_N_ROLLOUT_TRIALS)),
        materializer_issue_id=str(params.get("materializer_issue_id", "103db99")),
        include_objective_comparator=bool(params.get("include_objective_comparator", True)),
        include_map_decomposition=bool(params.get("include_map_decomposition", True)),
        include_perturbation_response=bool(params.get("include_perturbation_response", True)),
        include_feedback_ablation=bool(params.get("include_feedback_ablation", True)),
        perturbation_bank_mode=str(params.get("perturbation_bank_mode", "raw")),
        perturbation_calibration_level=params.get("perturbation_calibration_level"),
        perturbation_calibration_reach=params.get("perturbation_calibration_reach"),
        feedback_selection_level=str(params.get("feedback_selection_level", "small")),
        evaluation_manifest_path=_resolved_input_path(evaluation_input),
        evaluation_states=_resolved_input_states(evaluation_input),
        repo_root=repo_root,
    )
    plan = plan_gru_postrun_materialization(
        experiment=experiment,
        run_ids=tuple(run_ids),
        output_tag=output_tag,
        use_validation_selected_checkpoints=bool(
            params.get("use_validation_selected_checkpoints", True)
        ),
        fixed_bank_rescore_manifest_path=_optional_path(
            params.get("fixed_bank_rescore_manifest_path"),
            repo_root=repo_root,
        ),
        repo_root=repo_root,
    )
    payload = {
        **manifest,
        "declarative_analysis": _declarative_metadata(context),
        "bundle_contract": {
            "primary": "feedbax_analysis_bundle",
            "bundle": "rlrmp/gru_postrun",
            "analysis_manifest_id": context.manifest_id,
            "legacy_regeneration_spec_role": "compatibility",
        },
    }
    return MaterializationResult(
        payload=payload,
        existing_artifacts=tuple(_postrun_existing_artifacts(manifest, plan, repo_root=repo_root)),
    )


def _materialize_policy_diagnostics(
    context: AnalysisRunContext,
    params: PolicyDiagnosticsAnalysisParams,
    *,
    evaluation_input: Any,
) -> MaterializationResult:
    states = _required_evaluation_states(evaluation_input, analysis_name="policy diagnostics")
    rows = _selected_diagnostic_rows(
        states,
        source_key=params.source_key,
        row_ids=params.row_ids,
        analysis_name="policy diagnostics",
    )
    payload = {
        "schema_id": "rlrmp.policy_diagnostics_bank",
        "schema_version": "rlrmp.policy_diagnostics_bank.v1",
        "analysis_type": POLICY_DIAGNOSTICS_ANALYSIS_TYPE,
        "source_key": params.source_key,
        "evaluation_manifest_dependency": _evaluation_dependency_metadata(evaluation_input),
        "adapter": {
            "input_contract": "evaluation_manifest_cached_local_linear_policy_rows",
            "rollout_policy": "analysis_consumes_cached_evaluation_states_never_reruns_rollouts",
            "kernel_module": "rlrmp.eval.policy_diagnostics",
        },
        "rows": [
            _policy_diagnostic_row_payload(row, params=params, row_index=index)
            for index, row in enumerate(rows)
        ],
        "declarative_analysis": _declarative_metadata(context),
    }
    return MaterializationResult(
        payload=_json_ready(payload),
        payload_metadata={
            "analysis_type": POLICY_DIAGNOSTICS_ANALYSIS_TYPE,
            "source_key": params.source_key,
        },
    )


def _materialize_recurrent_jacobians(
    context: AnalysisRunContext,
    params: RecurrentJacobianAnalysisParams,
    *,
    evaluation_input: Any,
) -> MaterializationResult:
    states = _required_evaluation_states(evaluation_input, analysis_name="recurrent Jacobian")
    rows = _selected_diagnostic_rows(
        states,
        source_key=params.source_key,
        row_ids=params.row_ids,
        analysis_name="recurrent Jacobian",
    )
    payload = {
        "schema_id": "rlrmp.recurrent_jacobian_bank",
        "schema_version": "rlrmp.recurrent_jacobian_bank.v1",
        "analysis_type": RECURRENT_JACOBIAN_ANALYSIS_TYPE,
        "source_key": params.source_key,
        "evaluation_manifest_dependency": _evaluation_dependency_metadata(evaluation_input),
        "adapter": {
            "input_contract": "evaluation_manifest_cached_staged_recurrent_linearization_rows",
            "rollout_policy": "analysis_consumes_cached_evaluation_states_never_reruns_rollouts",
            "kernel_module": "rlrmp.eval.recurrent_jacobians",
        },
        "rows": [
            _recurrent_jacobian_row_payload(row, params=params, row_index=index)
            for index, row in enumerate(rows)
        ],
        "declarative_analysis": _declarative_metadata(context),
    }
    return MaterializationResult(
        payload=_json_ready(payload),
        payload_metadata={
            "analysis_type": RECURRENT_JACOBIAN_ANALYSIS_TYPE,
            "source_key": params.source_key,
        },
    )


def _required_evaluation_states(
    evaluation_input: Any,
    *,
    analysis_name: str,
) -> Mapping[str, Any]:
    states = _resolved_input_states(evaluation_input)
    if states is None:
        raise ValueError(f"{analysis_name} analysis requires evaluation states")
    return states


def _selected_diagnostic_rows(
    states: Mapping[str, Any],
    *,
    source_key: str,
    row_ids: Sequence[str] | None,
    analysis_name: str,
) -> tuple[Mapping[str, Any], ...]:
    source = states.get(source_key)
    if not isinstance(source, Mapping):
        raise ValueError(
            f"{analysis_name} evaluation states lack mapping payload {source_key!r}"
        )
    raw_rows = source.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError(f"{analysis_name} payload {source_key!r} must contain rows")
    rows = tuple(row for row in raw_rows if isinstance(row, Mapping))
    if len(rows) != len(raw_rows):
        raise TypeError(f"{analysis_name} rows must all be mappings")
    if row_ids is None:
        if not rows:
            raise ValueError(f"{analysis_name} payload {source_key!r} contains no rows")
        return rows

    wanted = {str(row_id) for row_id in row_ids}
    selected = tuple(row for row in rows if _row_id(row, row_index=-1) in wanted)
    found = {_row_id(row, row_index=-1) for row in selected}
    missing = sorted(wanted.difference(found))
    if missing:
        available = sorted(_row_id(row, row_index=index) for index, row in enumerate(rows))
        raise ValueError(
            f"{analysis_name} requested missing row_ids {missing}; available rows: {available}"
        )
    return selected


def _policy_diagnostic_row_payload(
    row: Mapping[str, Any],
    *,
    params: PolicyDiagnosticsAnalysisParams,
    row_index: int,
) -> dict[str, Any]:
    row_id = _row_id(row, row_index=row_index)
    values = _policy_block_values(row)
    schema = PolicyInputSchema.from_values(
        values,
        roles=_string_mapping(row.get("roles")),
        interpretations=_string_mapping(row.get("interpretations")),
        absent_blocks=_policy_absent_blocks(row.get("absent_blocks")),
    )
    action = _required_array(row, "action").reshape(-1)
    full_map = _policy_linear_map(row, schema=schema, output_size=action.size)
    flat0 = schema.flatten(values)
    action_shape = tuple(int(dim) for dim in _required_array(row, "action").shape)

    def local_policy(blocks: Mapping[str, Any]) -> jnp.ndarray:
        flat = schema.flatten(blocks)
        output = action + full_map @ (flat - flat0)
        return output.reshape(action_shape)

    jacobian = policy_jacobian(local_policy, values, schema=schema)
    block_summaries = {
        name: {
            "singular_values": singular_value_summary(matrix),
            "directional_gains": directional_gain_summary(
                matrix,
                _optional_direction_matrix(row, name),
            ),
        }
        for name, matrix in jacobian.by_block.items()
    }
    finite_difference = (
        validate_policy_jacobian(
            local_policy,
            values,
            schema=schema,
            epsilon=params.finite_difference_epsilon,
            finite_difference_batch_size=params.finite_difference_batch_size,
            atol=params.finite_difference_atol,
            rtol=params.finite_difference_rtol,
        ).to_summary()
        if params.include_finite_difference
        else {"status": "skipped"}
    )
    sisu_values = params.sisu_values
    if sisu_values is None and isinstance(row.get("sisu_values"), Sequence):
        sisu_values = [float(value) for value in row["sisu_values"]]
    sisu_modulation = (
        feedback_jacobian_sisu_modulation(
            local_policy,
            values,
            sisu_values=jnp.asarray(sisu_values),
            schema=schema,
            feedback_block=params.feedback_block,
            sisu_block=params.sisu_block,
        )
        if sisu_values is not None
        and params.feedback_block in schema.block_names
        and params.sisu_block in schema.block_names
        else {"status": "not_applicable", "reason": "sisu_values_or_blocks_absent"}
    )
    payload: dict[str, Any] = {
        "row_id": row_id,
        "metadata": dict(row.get("metadata", {})) if isinstance(row.get("metadata"), Mapping) else {},
        "policy_point": {
            "input_schema": schema.to_json(),
            "action_shape": list(action_shape),
        },
        "jacobian": jacobian.to_summary(),
        "block_summaries": block_summaries,
        "finite_difference": finite_difference,
        "sisu_modulation": sisu_modulation,
        "signed_pairs": _policy_signed_pair_payloads(row),
    }
    if params.include_arrays:
        payload["arrays"] = {
            "jacobian": jacobian.full,
            "blocks": dict(jacobian.by_block),
        }
    return payload


def _recurrent_jacobian_row_payload(
    row: Mapping[str, Any],
    *,
    params: RecurrentJacobianAnalysisParams,
    row_index: int,
) -> dict[str, Any]:
    row_id = _row_id(row, row_index=row_index)
    blocks = _required_mapping(row, "blocks")
    h_pre = _required_array(row, "h_pre")
    feedback = _required_array(row, "feedback")
    sisu = _required_array(row, "sisu")
    context = _optional_array(row.get("context"))
    h_post = _required_array(row, "h_post").reshape(-1)
    action = _required_array(row, "u").reshape(-1)
    A = _required_block_matrix(blocks, "A")
    B_y = _required_block_matrix(blocks, "B_y")
    B_s = _required_block_matrix(blocks, "B_s")
    B_c = _optional_array(blocks.get("B_c"))
    W = _required_block_matrix(blocks, "W")
    h0 = h_pre.reshape(-1)
    y0 = feedback.reshape(-1)
    s0 = sisu.reshape(-1)
    c0 = None if context is None else context.reshape(-1)

    def staged_update(
        h_value: jnp.ndarray,
        feedback_value: jnp.ndarray,
        sisu_value: jnp.ndarray,
        context_value: jnp.ndarray | None,
    ) -> jnp.ndarray:
        delta = A @ (jnp.ravel(h_value) - h0)
        delta = delta + B_y @ (jnp.ravel(feedback_value) - y0)
        delta = delta + B_s @ (jnp.ravel(sisu_value) - s0)
        if B_c is not None and c0 is not None and context_value is not None:
            delta = delta + B_c @ (jnp.ravel(context_value) - c0)
        return (h_post + delta).reshape(_required_array(row, "h_post").shape)

    def readout(h_value: jnp.ndarray) -> jnp.ndarray:
        return action + W @ (jnp.ravel(h_value) - h_post)

    bank = compute_recurrent_jacobian_bank(
        staged_update=staged_update,
        readout=readout,
        h_pre=h_pre,
        feedback=feedback,
        sisu=sisu,
        context=context,
        finite_difference=params.include_finite_difference,
        finite_difference_epsilon=params.finite_difference_epsilon,
        finite_difference_batch_size=params.finite_difference_batch_size,
    )
    payload = bank.as_dict(include_arrays=params.include_arrays)
    payload["row_id"] = row_id
    row_metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata"), Mapping) else {}
    payload["metadata"] = {
        **row_metadata,
        **dict(payload.get("metadata", {})),
    }
    return payload


def _row_id(row: Mapping[str, Any], *, row_index: int) -> str:
    value = row.get("row_id", row.get("id"))
    if value is not None:
        return str(value)
    if row_index < 0:
        return "<missing-row-id>"
    return f"row_{row_index}"


def _policy_block_values(row: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
    raw_blocks = row.get("blocks")
    if raw_blocks is None and isinstance(row.get("policy_point"), Mapping):
        raw_blocks = row["policy_point"].get("blocks")
    if not isinstance(raw_blocks, Mapping):
        raise ValueError("policy diagnostic row requires blocks mapping")
    return {str(name): jnp.asarray(value) for name, value in raw_blocks.items()}


def _policy_absent_blocks(value: Any) -> tuple[PolicyAbsentInputBlock, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("absent_blocks must be a sequence of mappings")
    blocks = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("absent_blocks entries must be mappings")
        blocks.append(
            PolicyAbsentInputBlock(
                name=str(item["name"]),
                role=str(item["role"]),
                reason=str(item["reason"]),
            )
        )
    return tuple(blocks)


def _policy_linear_map(
    row: Mapping[str, Any],
    *,
    schema: PolicyInputSchema,
    output_size: int,
) -> jnp.ndarray:
    if "linear_map" in row:
        matrix = jnp.asarray(row["linear_map"])
        return _validate_matrix_shape(matrix, rows=output_size, cols=schema.size, name="linear_map")
    block_maps = row.get("block_maps")
    if block_maps is None:
        block_maps = row.get("linear_maps")
    if not isinstance(block_maps, Mapping):
        raise ValueError("policy diagnostic row requires linear_map or block_maps")
    matrices = []
    for block in schema.blocks:
        if block.name not in block_maps:
            raise ValueError(f"policy diagnostic row lacks block map {block.name!r}")
        matrices.append(
            _validate_matrix_shape(
                jnp.asarray(block_maps[block.name]),
                rows=output_size,
                cols=block.size,
                name=f"block_maps.{block.name}",
            )
        )
    return jnp.concatenate(matrices, axis=1) if matrices else jnp.zeros((output_size, 0))


def _policy_signed_pair_payloads(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_pairs = row.get("signed_pairs", ())
    if raw_pairs is None:
        return []
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise TypeError("signed_pairs must be a sequence of mappings")
    pairs = []
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, Mapping):
            raise TypeError("signed_pairs entries must be mappings")
        payload = signed_pair_odd_even_summary(
            _required_array(pair, "positive_response"),
            _required_array(pair, "negative_response"),
            baseline=_optional_array(pair.get("baseline_response")),
        )
        payload["pair_id"] = str(pair.get("pair_id", f"pair_{index}"))
        pairs.append(payload)
    return pairs


def _optional_direction_matrix(row: Mapping[str, Any], block_name: str) -> jnp.ndarray | None:
    directions = row.get("directions")
    if not isinstance(directions, Mapping) or block_name not in directions:
        return None
    return jnp.asarray(directions[block_name])


def _required_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"diagnostic row requires mapping {key!r}")
    return value


def _required_array(row: Mapping[str, Any], key: str) -> jnp.ndarray:
    if key not in row:
        raise ValueError(f"diagnostic row requires array {key!r}")
    return jnp.asarray(row[key])


def _optional_array(value: Any) -> jnp.ndarray | None:
    return None if value is None else jnp.asarray(value)


def _required_block_matrix(blocks: Mapping[str, Any], key: str) -> jnp.ndarray:
    if key not in blocks:
        raise ValueError(f"recurrent diagnostic row requires block {key!r}")
    matrix = jnp.asarray(blocks[key])
    if matrix.ndim != 2:
        raise ValueError(f"recurrent block {key!r} must be 2D, got shape {matrix.shape}")
    return matrix


def _validate_matrix_shape(
    matrix: jnp.ndarray,
    *,
    rows: int,
    cols: int,
    name: str,
) -> jnp.ndarray:
    if matrix.shape != (rows, cols):
        raise ValueError(f"{name} has shape {matrix.shape}, expected {(rows, cols)}")
    return matrix


def _string_mapping(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if (
        hasattr(value, "tolist")
        and hasattr(value, "shape")
        and not isinstance(value, (str, bytes))
    ):
        return _json_ready(value.tolist())
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


_COMPONENT_STATUS_UNAVAILABLE_EXPR = Compare(
    item="component_status",
    path="status",
    op="eq",
    value="unavailable",
)


def _feedback_quality_component_gate_expr(name: str) -> Expr:
    return AllOf(
        exprs=[
            _feedback_quality_include_flag_expr(name),
            Not(expr=_feedback_quality_not_applicable_expr(name)),
        ]
    )


def _feedback_quality_include_flag_expr(name: str) -> Expr:
    return AnyOf(
        exprs=[
            Not(expr=Compare(item="params", path=f"include_{name}", op="exists")),
            Compare(item="params", path=f"include_{name}", op="eq", value=True),
        ]
    )


def _feedback_quality_not_applicable_expr(name: str) -> Expr:
    return AllOf(
        exprs=[
            Compare(item="params", path="not_applicable_components", op="exists"),
            Compare(
                item="params",
                path="not_applicable_components",
                op="contains",
                value=name,
            ),
        ]
    )


def _feedback_quality_gating_decision(
    registration: FeedbackQualityComponentRegistration,
    *,
    params: Mapping[str, Any],
    component_status: Mapping[str, Any],
) -> FeedbackQualityGatingDecision:
    params_payload = dict(params)
    normalized_not_applicable = _optional_str_sequence(
        params_payload.get("not_applicable_components")
    )
    if normalized_not_applicable is not None:
        params_payload["not_applicable_components"] = normalized_not_applicable
    ctx = ExpressionContext(
        items={
            "params": ContextItem(kind="params", payload=params_payload),
            "component_status": ContextItem(
                kind="component_status",
                payload=dict(component_status),
            ),
        }
    )
    included = evaluate_expr(_feedback_quality_include_flag_expr(registration.name), ctx)
    not_applicable = evaluate_expr(_feedback_quality_not_applicable_expr(registration.name), ctx)
    eligible = evaluate_expr(registration.gating_expr, ctx)
    should_materialize = evaluate_expr(
        AllOf(exprs=[registration.gating_expr, _COMPONENT_STATUS_UNAVAILABLE_EXPR]),
        ctx,
    )
    return FeedbackQualityGatingDecision(
        included=included,
        not_applicable=not_applicable,
        eligible=eligible,
        should_materialize=should_materialize,
    )


def _feedback_quality_component_registrations() -> dict[str, FeedbackQualityComponentRegistration]:
    return {
        "evaluation_diagnostics": FeedbackQualityComponentRegistration(
            name="evaluation_diagnostics",
            materializer=_materialize_feedback_quality_evaluation_diagnostics,
            artifact_role=FEEDBACK_QUALITY_COMPONENT_STATUS_ROLES["evaluation_diagnostics"],
            logical_name="feedback_quality/evaluation_diagnostics_status.json",
            live_materializer=(
                "rlrmp.analysis.pipelines.gru_evaluation_diagnostics."
                "materialize_gru_evaluation_diagnostics"
            ),
            gating_label="include flag and applicable component",
            gating_expr=_feedback_quality_component_gate_expr("evaluation_diagnostics"),
        ),
        "objective_comparator": FeedbackQualityComponentRegistration(
            name="objective_comparator",
            materializer=_materialize_feedback_quality_objective_comparator,
            artifact_role=FEEDBACK_QUALITY_COMPONENT_STATUS_ROLES["objective_comparator"],
            logical_name="feedback_quality/objective_comparator_status.json",
            live_materializer=(
                "rlrmp.analysis.pipelines.objective_comparator."
                "materialize_gru_objective_comparator_sidecar"
            ),
            gating_label="include flag and applicable component",
            gating_expr=_feedback_quality_component_gate_expr("objective_comparator"),
        ),
        "perturbation_response": FeedbackQualityComponentRegistration(
            name="perturbation_response",
            materializer=_materialize_feedback_quality_perturbation_response,
            artifact_role=FEEDBACK_QUALITY_COMPONENT_STATUS_ROLES["perturbation_response"],
            logical_name="feedback_quality/perturbation_response_status.json",
            live_materializer=(
                "rlrmp.analysis.pipelines.gru_perturbation_bank."
                "materialize_gru_perturbation_response"
            ),
            gating_label="include flag and applicable component",
            gating_expr=_feedback_quality_component_gate_expr("perturbation_response"),
        ),
        "feedback_ablation": FeedbackQualityComponentRegistration(
            name="feedback_ablation",
            materializer=_materialize_feedback_quality_feedback_ablation,
            artifact_role=FEEDBACK_QUALITY_COMPONENT_STATUS_ROLES["feedback_ablation"],
            logical_name="feedback_quality/feedback_ablation_status.json",
            live_materializer=(
                "rlrmp.analysis.pipelines.gru_feedback_ablation."
                "materialize_gru_feedback_ablation"
            ),
            gating_label="include flag and applicable component",
            gating_expr=_feedback_quality_component_gate_expr("feedback_ablation"),
        ),
        "response_norm_plots": FeedbackQualityComponentRegistration(
            name="response_norm_plots",
            materializer=_materialize_feedback_quality_response_norm_plots,
            artifact_role=FEEDBACK_QUALITY_COMPONENT_STATUS_ROLES["response_norm_plots"],
            logical_name="feedback_quality/response_norm_plots_status.json",
            live_materializer=(
                "rlrmp.analysis.pipelines.gru_perturbation_response_norm_plots."
                "materialize_response_norm_plots"
            ),
            gating_label="include flag and applicable component",
            gating_expr=_feedback_quality_component_gate_expr("response_norm_plots"),
        ),
        "perturbation_calibration": FeedbackQualityComponentRegistration(
            name="perturbation_calibration",
            materializer=_materialize_feedback_quality_perturbation_calibration,
            artifact_role=FEEDBACK_QUALITY_COMPONENT_STATUS_ROLES["perturbation_calibration"],
            logical_name="feedback_quality/perturbation_calibration_status.json",
            live_materializer=(
                "rlrmp.analysis.pipelines.gru_perturbation_calibration."
                "materialize_perturbation_open_loop_calibration"
            ),
            gating_label="include flag and applicable component",
            gating_expr=_feedback_quality_component_gate_expr("perturbation_calibration"),
        ),
    }


def _materialize_feedback_quality_component(
    context: AnalysisRunContext,
    data: AnalysisInputData,
    *,
    registration: FeedbackQualityComponentRegistration,
    params: Mapping[str, Any],
    experiment: str,
    run_ids: Sequence[str],
    evaluation_input: Any | None,
    repo_root: Path,
) -> MaterializationResult:
    del data
    if not run_ids:
        raise ValueError("Feedback-quality component recipe requires at least one run ID")
    plan = _feedback_quality_plan(
        params,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
    )
    component = _feedback_quality_components(plan, params=params, repo_root=repo_root)[
        registration.name
    ]
    output, existing, groups = _feedback_quality_component_output(
        registration,
        component,
        params=params,
        repo_root=repo_root,
    )
    materialization_mode = "live_analysis_node"
    if _feedback_quality_gating_decision(
        registration,
        params=params,
        component_status=output,
    ).should_materialize:
        raw_output = registration.materializer(
            context,
            params,
            experiment,
            tuple(run_ids),
            evaluation_input,
            repo_root,
        )
        refreshed, existing, groups = _feedback_quality_component_output(
            registration,
            component,
            params=params,
            repo_root=repo_root,
        )
        output = _feedback_quality_live_output(
            refreshed,
            raw_output,
            name=registration.name,
        )
        materialization_mode = "live_analysis_node"

    output.setdefault("component_node", registration.name)
    output.setdefault("materialization_mode", materialization_mode)
    output.setdefault("live_materializer", registration.live_materializer)
    return MaterializationResult(
        payload=output,
        existing_artifacts=existing,
        artifact_groups=groups,
        payload_metadata={"component": registration.name},
    )


def _feedback_quality_live_output(
    refreshed: Mapping[str, Any],
    raw_output: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    if refreshed.get("status") == "materialized":
        return dict(refreshed)
    if raw_output.get("status") == "skipped":
        return {
            **dict(refreshed),
            "status": "skipped",
            "reason": raw_output.get("reason", f"{name}_materializer_skipped"),
            "detail": dict(raw_output),
        }
    return dict(refreshed)


def _materialize_feedback_quality_evaluation_diagnostics(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    experiment: str,
    run_ids: tuple[str, ...],
    evaluation_input: Any | None,
    repo_root: Path,
) -> Mapping[str, Any]:
    plan = _feedback_quality_plan(
        params,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
    )
    return _materialize_gru_evaluation_diagnostics(
        context,
        {
            **dict(params),
            "experiment": experiment,
            "run_ids": list(run_ids),
            "output_path": str(plan.evaluation_manifest_path),
            "bulk_dir": str(plan.evaluation_bulk_dir),
        },
        evaluation_input=evaluation_input,
    ).payload


def _materialize_feedback_quality_objective_comparator(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    experiment: str,
    run_ids: tuple[str, ...],
    evaluation_input: Any | None,
    repo_root: Path,
) -> Mapping[str, Any]:
    del context, evaluation_input
    plan = _feedback_quality_plan(
        params,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
    )
    return materialize_optional_objective_comparator(
        experiment=experiment,
        run_ids=run_ids,
        labels=_optional_str_sequence(params.get("labels")),
        checkpoint_policy=plan.checkpoint_policy,
        use_validation_selected_checkpoints=bool(
            params.get("use_validation_selected_checkpoints", True)
        ),
        checkpoint_manifest=None,
        checkpoint_manifest_path=plan.checkpoint_manifest_path,
        standard_manifest_path=plan.standard_manifest_path,
        output_path=plan.objective_comparator_json_path,
        note_path=plan.objective_comparator_note_path,
        repo_root=repo_root,
    )


def _materialize_feedback_quality_perturbation_response(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    experiment: str,
    run_ids: tuple[str, ...],
    evaluation_input: Any | None,
    repo_root: Path,
) -> Mapping[str, Any]:
    del context, evaluation_input
    plan = _feedback_quality_plan(
        params,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
    )
    return materialize_optional_perturbation_response(
        experiment=experiment,
        run_ids=run_ids,
        labels=_optional_str_sequence(params.get("labels")),
        n_rollout_trials=int(params.get("n_rollout_trials", DEFAULT_N_ROLLOUT_TRIALS)),
        output_path=plan.perturbation_response_json_path,
        note_path=plan.perturbation_response_note_path,
        bulk_dir=plan.perturbation_response_bulk_dir,
        calibration_level=params.get("perturbation_calibration_level"),
        calibration_reach=params.get("perturbation_calibration_reach"),
        preferred_checkpoint_manifest_path=plan.checkpoint_manifest_path,
        write_bulk_arrays=bool(params.get("write_perturbation_bulk_arrays", False)),
        repo_root=repo_root,
    )


def _materialize_feedback_quality_feedback_ablation(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    experiment: str,
    run_ids: tuple[str, ...],
    evaluation_input: Any | None,
    repo_root: Path,
) -> Mapping[str, Any]:
    del context, evaluation_input
    plan = _feedback_quality_plan(
        params,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
    )
    return materialize_optional_feedback_ablation(
        experiment=experiment,
        run_ids=run_ids,
        labels=_optional_str_sequence(params.get("labels")),
        n_rollout_trials=int(params.get("n_rollout_trials", DEFAULT_N_ROLLOUT_TRIALS)),
        output_path=plan.feedback_ablation_json_path,
        note_path=plan.feedback_ablation_note_path,
        calibration_level=params.get("perturbation_calibration_level"),
        calibration_reach=params.get("perturbation_calibration_reach"),
        feedback_selection_level=str(params.get("feedback_selection_level", "small")),
        preferred_checkpoint_manifest_path=plan.checkpoint_manifest_path,
        repo_root=repo_root,
    )


def _materialize_feedback_quality_response_norm_plots(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    experiment: str,
    run_ids: tuple[str, ...],
    evaluation_input: Any | None,
    repo_root: Path,
) -> Mapping[str, Any]:
    del context, evaluation_input
    plan = _feedback_quality_plan(
        params,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
    )
    component = _feedback_quality_components(plan, params=params, repo_root=repo_root)[
        "response_norm_plots"
    ]
    from rlrmp.analysis.pipelines.gru_perturbation_response_norm_plots import (
        materialize_response_norm_plots,
    )

    figure_dir = component["groups"][0][0]
    return materialize_response_norm_plots(
        source_manifest_path=plan.perturbation_response_json_path,
        results_dir=figure_dir,
        asset_dir=figure_dir / "_assets",
        note_path=component["notes"][0],
        manifest_path=component["tracked"][0],
        repo_root=repo_root,
    )


def _materialize_feedback_quality_perturbation_calibration(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    experiment: str,
    run_ids: tuple[str, ...],
    evaluation_input: Any | None,
    repo_root: Path,
) -> Mapping[str, Any]:
    del context, evaluation_input
    plan = _feedback_quality_plan(
        params,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
    )
    component = _feedback_quality_components(plan, params=params, repo_root=repo_root)[
        "perturbation_calibration"
    ]
    from rlrmp.analysis.pipelines.gru_perturbation_calibration import (
        materialize_perturbation_open_loop_calibration,
    )

    return materialize_perturbation_open_loop_calibration(
        result_experiment=experiment,
        output_path=component["tracked"][0],
        note_path=component["notes"][0],
        repo_root=repo_root,
    )


def _feedback_quality_plan(
    params: Mapping[str, Any],
    *,
    experiment: str,
    run_ids: Sequence[str],
    repo_root: Path,
) -> Any:
    return plan_gru_postrun_materialization(
        experiment=experiment,
        run_ids=tuple(run_ids),
        output_tag=str(params.get("output_tag", DEFAULT_OUTPUT_TAG)),
        use_validation_selected_checkpoints=bool(
            params.get("use_validation_selected_checkpoints", True)
        ),
        fixed_bank_rescore_manifest_path=_optional_path(
            params.get("fixed_bank_rescore_manifest_path"),
            repo_root=repo_root,
        ),
        repo_root=repo_root,
    )


def _feedback_quality_component_outputs_from_inputs(
    inputs: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    by_role = {
        registration.artifact_role: registration
        for registration in _feedback_quality_component_registrations().values()
    }
    for resolved in inputs:
        manifest = getattr(resolved, "manifest", None)
        if manifest is None:
            continue
        for artifact in getattr(manifest, "artifacts", ()):
            registration = by_role.get(artifact.role)
            if registration is None or artifact.logical_name != registration.logical_name:
                continue
            if artifact.uri is None:
                raise ValueError(
                    f"Feedback-quality component {registration.name!r} lacks artifact URI"
                )
            outputs[registration.name] = _read_json_payload(Path(artifact.uri))
    missing = sorted(set(FEEDBACK_QUALITY_COMPONENT_NAMES).difference(outputs))
    if missing:
        raise ValueError(f"Feedback-quality aggregate missing component payloads: {missing}")
    return outputs


def _materialize_output_feedback_rollout_recovery(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
) -> MaterializationResult:
    repo_root = _repo_root_from_params(params)
    issue_id = str(params.get("issue_id", OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ISSUE_ID))
    note_path = _optional_path(params.get("note_output"), repo_root=repo_root) or (
        context.results_cache_dir / "output_feedback_rollout_recovery.md"
    )
    manifest_path = _optional_path(params.get("manifest_output"), repo_root=repo_root) or (
        context.results_cache_dir / "output_feedback_rollout_recovery_manifest.json"
    )
    artifact_path = _optional_path(params.get("artifact_output"), repo_root=repo_root) or (
        context.results_cache_dir / "bulk" / "output_feedback_rollout_recovery.npz"
    )
    summary = write_output_feedback_rollout_recovery_outputs(
        issue_id=issue_id,
        discretization=str(params.get("discretization", DEFAULT_DISCRETIZATION)),
        lane=str(params.get("lane", DEFAULT_LANE)),
        note_path=note_path,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        repo_root=repo_root,
    )
    existing_artifacts = tuple(
        artifact
        for artifact in (
            _existing_file(
                manifest_path,
                role="rlrmp-output-feedback-rollout-recovery-manifest",
                logical_name="legacy/output_feedback_rollout_recovery_manifest.json",
            ),
            _existing_file(
                note_path,
                role="rlrmp-output-feedback-rollout-recovery-note",
                logical_name="legacy/output_feedback_rollout_recovery.md",
            ),
        )
        if artifact is not None
    )
    artifact_groups = _single_file_artifact_group(
        artifact_path,
        role="rlrmp-output-feedback-rollout-recovery-bulk",
        logical_name="bulk/output_feedback_rollout_recovery.npz",
        group_id="output_feedback_rollout_recovery_bulk",
        member_role="rollout_recovery_arrays",
        repo_root=repo_root,
    )
    return MaterializationResult(
        payload={
            **summary,
            "declarative_analysis": _declarative_metadata(context),
            "evaluation_dependency_policy": {
                "status": "not_applicable",
                "reason": (
                    "Analytical output-feedback rollouts take fitted gains, not "
                    "model artifacts, and remain analysis-internal per e1ad278 Q2."
                ),
            },
        },
        existing_artifacts=existing_artifacts,
        artifact_groups=artifact_groups,
    )


def _materialize_robustness_phenotype(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    inputs: Sequence[Any],
) -> MaterializationResult:
    repo_root = _repo_root_from_params(params)
    source_paths = _robustness_phenotype_source_paths(params, inputs, repo_root=repo_root)
    sources = load_hinf_phenotype_sources(source_paths, repo_root=repo_root)
    sidecar = build_hinf_phenotype_sidecar(
        sources=sources,
        issue=str(params.get("issue_id", ROBUSTNESS_PHENOTYPE_ISSUE_ID)),
        scope=str(params.get("scope", DEFAULT_HINF_PHENOTYPE_SCOPE)),
        generated_by="rlrmp.analysis.declarative_materialization.robustness_phenotype",
    )

    json_path = _optional_path(params.get("output_json"), repo_root=repo_root) or (
        context.results_cache_dir / "hinf_phenotype_sidecar.json"
    )
    markdown_path = _optional_path(params.get("output_markdown"), repo_root=repo_root) or (
        context.results_cache_dir / "hinf_phenotype_sidecar.md"
    )
    regeneration_spec_path = _optional_path(
        params.get("regeneration_spec_path"),
        repo_root=repo_root,
    )
    write_hinf_phenotype_sidecar(
        sidecar,
        json_path=json_path,
        markdown_path=markdown_path,
        regeneration_spec_path=regeneration_spec_path,
        repo_root=repo_root,
    )
    payload = {
        **_read_json_payload(json_path),
        "declarative_analysis": _declarative_metadata(context),
        "bundle_contract": {
            "primary": "feedbax_analysis_bundle",
            "bundle": "rlrmp/robustness_phenotype",
            "analysis_manifest_id": context.manifest_id,
            "schema_owner": "rlrmp",
            "formal_claim_policy": "conservative_no_upgrade_without_formal_inputs",
        },
    }
    existing_artifacts = tuple(
        artifact
        for artifact in (
            _existing_file(
                json_path,
                role="rlrmp-robustness-phenotype-sidecar-json",
                logical_name=_legacy_logical_name(json_path, repo_root),
            ),
            _existing_file(
                markdown_path,
                role="rlrmp-robustness-phenotype-sidecar-note",
                logical_name=_legacy_logical_name(markdown_path, repo_root),
            ),
            _existing_file(
                regeneration_spec_path,
                role="rlrmp-robustness-phenotype-regeneration-spec",
                logical_name=_legacy_logical_name(regeneration_spec_path, repo_root),
            )
            if regeneration_spec_path is not None
            else None,
        )
        if artifact is not None
    )
    return MaterializationResult(
        payload=payload,
        existing_artifacts=existing_artifacts,
    )


def _robustness_phenotype_source_paths(
    params: Mapping[str, Any],
    inputs: Sequence[Any],
    *,
    repo_root: Path,
) -> dict[str, Path | str | None]:
    source_paths = {
        str(name): None if value is None else _optional_path(value, repo_root=repo_root)
        for name, value in (params.get("source_paths") or {}).items()
    }
    for resolved in inputs:
        manifest = getattr(resolved, "manifest", None)
        if manifest is None:
            continue
        for artifact in getattr(manifest, "artifacts", ()):
            source_name = ROBUSTNESS_PHENOTYPE_SOURCE_ROLES.get(artifact.role)
            if source_name is None or source_name in source_paths:
                continue
            if artifact.uri is not None:
                source_paths[source_name] = _optional_path(artifact.uri, repo_root=repo_root)
    return source_paths


def _perturbation_class_response_payload(
    states: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    evaluation_input: Any,
) -> dict[str, Any]:
    family = str(params.get("family", ""))
    if not family:
        raise ValueError("perturbation class response params must include family")
    if states.get("evaluation_type") != PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE:
        raise ValueError(
            "perturbation class response requires "
            f"{PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE!r} eval states"
        )
    class_index_map = states.get("class_index_map")
    if not isinstance(class_index_map, Mapping):
        raise ValueError("perturbation response eval states lack class_index_map")
    if family not in class_index_map:
        available = sorted(str(item) for item in class_index_map)
        raise ValueError(
            "perturbation class response requested family "
            f"{family!r}, but the evaluation manifest contains families {available}"
        )
    class_entry = class_index_map[family]
    if not isinstance(class_entry, Mapping):
        raise TypeError(f"class_index_map entry for {family!r} must be a mapping")

    perturbation_ids = _selected_perturbation_ids(class_entry, params, family=family)
    row_index_by_id = _row_index_by_perturbation_id(class_entry, perturbation_ids)
    bank = _slice_perturbation_bank(
        states.get("perturbation_battery"),
        perturbation_ids,
        row_index_by_id=row_index_by_id,
    )
    runs = _slice_perturbation_runs(
        states.get("response_tensors"),
        perturbation_ids,
        row_index_by_id=row_index_by_id,
    )
    calibration_identities = _calibration_identities_for_leaf(states, params, family=family)
    eval_dependency = _evaluation_dependency_metadata(evaluation_input)
    eval_id = (
        eval_dependency.get("manifest_id")
        or states.get("evaluation_manifest_id")
        or getattr(getattr(evaluation_input, "ref", None), "id", None)
    )
    return {
        "schema_id": PERTURBATION_CLASS_RESPONSE_SCHEMA_ID,
        "schema_version": PERTURBATION_CLASS_RESPONSE_SCHEMA_VERSION,
        "family": family,
        "row_ids": list(perturbation_ids),
        "row_indices": [row_index_by_id[row_id] for row_id in perturbation_ids],
        "row_index_by_perturbation_id": row_index_by_id,
        "class_index_entry": dict(class_entry),
        "calibration_identity": calibration_identities,
        "evaluation_manifest": {
            **eval_dependency,
            "id": eval_id,
        },
        "bank": bank,
        "runs": runs,
        "aggregate_base": _aggregate_base_from_eval_states(states),
    }


def _selected_perturbation_ids(
    class_entry: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    family: str,
) -> tuple[str, ...]:
    available = tuple(str(row_id) for row_id in class_entry.get("perturbation_ids", ()))
    requested = params.get("row_ids")
    if requested is None:
        return available
    if isinstance(requested, str):
        selected = (requested,)
    elif isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)):
        selected = tuple(str(row_id) for row_id in requested)
    else:
        raise TypeError("perturbation class response row_ids must be a string or sequence")
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(
            f"perturbation class response family {family!r} requested unavailable "
            f"row_ids {missing}; available row_ids: {sorted(available)}"
        )
    return selected


def _row_index_by_perturbation_id(
    class_entry: Mapping[str, Any],
    perturbation_ids: Sequence[str],
) -> dict[str, int]:
    ids = [str(row_id) for row_id in class_entry.get("perturbation_ids", ())]
    indices = [int(index) for index in class_entry.get("row_indices", ())]
    mapping = dict(zip(ids, indices, strict=True))
    return {row_id: mapping[row_id] for row_id in perturbation_ids}


def _slice_perturbation_bank(
    raw_bank: Any,
    perturbation_ids: Sequence[str],
    *,
    row_index_by_id: Mapping[str, int],
) -> dict[str, Any]:
    if not isinstance(raw_bank, Mapping):
        raise TypeError("perturbation response eval states lack perturbation_battery")
    bank = dict(raw_bank)
    rows = raw_bank.get("perturbations", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("perturbation_battery.perturbations must be a sequence")
    by_id = {
        str(row.get("perturbation_id")): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }
    bank["perturbations"] = [
        {**by_id[row_id], "class_response_row_index": row_index_by_id[row_id]}
        for row_id in perturbation_ids
    ]
    return bank


def _slice_perturbation_runs(
    response_tensors: Any,
    perturbation_ids: Sequence[str],
    *,
    row_index_by_id: Mapping[str, int],
) -> dict[str, Any]:
    if not isinstance(response_tensors, Mapping):
        raise TypeError("perturbation response eval states lack response_tensors")
    runs = response_tensors.get("runs", {})
    if not isinstance(runs, Mapping):
        raise TypeError("response_tensors.runs must be a mapping")
    return {
        str(run_id): _slice_perturbation_run(
            run_payload,
            perturbation_ids,
            row_index_by_id=row_index_by_id,
        )
        for run_id, run_payload in runs.items()
        if isinstance(run_payload, Mapping)
    }


def _slice_perturbation_run(
    run_payload: Mapping[str, Any],
    perturbation_ids: Sequence[str],
    *,
    row_index_by_id: Mapping[str, int],
) -> dict[str, Any]:
    selected = set(perturbation_ids)
    run = dict(run_payload)
    rows = [
        dict(row)
        for row in run_payload.get("perturbations", ())
        if isinstance(row, Mapping) and str(row.get("perturbation_id")) in selected
    ]
    rows.sort(key=lambda row: row_index_by_id[str(row["perturbation_id"])])
    for row in rows:
        row["class_response_row_index"] = row_index_by_id[str(row["perturbation_id"])]
    run["perturbations"] = rows
    run["status_counts"] = _perturbation_status_counts(rows)
    run["robust_response_summary"] = _summarize_perturbation_rows(rows)
    bulk_files = run_payload.get("bulk_files")
    if isinstance(bulk_files, Mapping):
        run["bulk_files"] = {
            str(row_id): bulk_files[row_id]
            for row_id in perturbation_ids
            if row_id in bulk_files
        }
    return run


def _calibration_identities_for_leaf(
    states: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    family: str,
) -> list[dict[str, Any]]:
    identities = states.get("consumed_data_identities", [])
    if isinstance(identities, Mapping):
        recorded = [dict(identities)]
    elif isinstance(identities, Sequence) and not isinstance(identities, (str, bytes)):
        recorded = [dict(item) for item in identities if isinstance(item, Mapping)]
    else:
        recorded = []
    expected = params.get("expected_calibration_identity", params.get("calibration_identity"))
    if expected is None:
        return recorded
    if not isinstance(expected, Mapping):
        raise TypeError("expected_calibration_identity must be a mapping")
    if not any(_identity_matches(identity, expected) for identity in recorded):
        raise ValueError(
            f"perturbation class response family {family!r} expected calibration "
            f"identity {dict(expected)!r}, but eval manifest recorded {recorded!r}"
        )
    return recorded


def _identity_matches(recorded: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(recorded.get(key) == value for key, value in expected.items())


def _aggregate_base_from_eval_states(states: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": GRU_PERTURBATION_BANK_SCHEMA_VERSION,
        "issue": None,
        "source_experiment": None,
        "checkpoint_policy": {
            "status": "from_evaluation_manifest",
            "evaluation_manifest_id": states.get("evaluation_manifest_id"),
        },
        "scope": "controller_independent_perturbation_response",
        "regeneration_spec": None,
        "bank_mode": states.get("production_mode"),
        "feedback_scale_manifest": None,
        "semantics_correction": (
            "Aggregate reconstructed from perturbation-class response leaves over the "
            "shared perturbation-response evaluation manifest."
        ),
        "extlqg_comparator": {
            "status": "preserved_in_row_metrics",
            "checkpoint_selection_role": "audit_only_not_used_for_selection",
        },
        "robust_output_feedback_comparator": {
            "status": "preserved_in_row_metrics",
            "checkpoint_selection_role": "audit_only_not_used_for_selection",
        },
        "full_qrf_cost": {
            "status": "preserved_in_row_metrics",
            "lens": "realized_deterministic_rollout_full_qrf",
        },
    }


def _load_perturbation_class_product(resolved: Any) -> dict[str, Any]:
    manifest = getattr(resolved, "manifest", None)
    if manifest is None:
        raise ValueError("perturbation bank aggregate requires analysis manifest inputs")
    for artifact in getattr(manifest, "artifacts", ()):
        if artifact.role != "rlrmp-perturbation-class-response":
            continue
        if artifact.uri is None:
            raise ValueError("perturbation class response artifact lacks uri")
        payload = _read_json_payload(Path(artifact.uri))
        if payload.get("schema_id") != PERTURBATION_CLASS_RESPONSE_SCHEMA_ID:
            raise ValueError(
                "perturbation class response artifact has wrong schema_id "
                f"{payload.get('schema_id')!r}"
            )
        return payload
    raise ValueError(
        f"analysis manifest {getattr(manifest, 'id', '<unknown>')!r} lacks "
        "rlrmp-perturbation-class-response artifact"
    )


def _aggregate_perturbation_class_products(
    leaf_products: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    if not leaf_products:
        raise ValueError("perturbation bank aggregate requires at least one leaf product")
    eval_ids = {
        str(product.get("evaluation_manifest", {}).get("id"))
        for product in leaf_products
        if isinstance(product.get("evaluation_manifest"), Mapping)
    }
    if len(eval_ids) != 1:
        raise ValueError(
            "perturbation bank aggregate requires all leaves to share one evaluation "
            f"manifest; got {sorted(eval_ids)}"
        )
    base = dict(leaf_products[0].get("aggregate_base", {}))
    base["schema_version"] = GRU_PERTURBATION_BANK_SCHEMA_VERSION
    for key in ("issue", "source_experiment", "bank_mode"):
        if params.get(key) is not None:
            base[key] = params[key]
    rows = _aggregate_bank_rows(leaf_products)
    bank = dict(leaf_products[0].get("bank", {}))
    bank["perturbations"] = rows
    return {
        **base,
        "bank": bank,
        "runs": _aggregate_leaf_runs(leaf_products),
    }


def _aggregate_bank_rows(leaf_products: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in leaf_products:
        bank = product.get("bank", {})
        if not isinstance(bank, Mapping):
            continue
        for row in bank.get("perturbations", ()):
            if isinstance(row, Mapping):
                rows.append(dict(row))
    rows.sort(key=lambda row: int(row.get("class_response_row_index", 0)))
    for row in rows:
        row.pop("class_response_row_index", None)
    return rows


def _aggregate_leaf_runs(leaf_products: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    for product in leaf_products:
        run_payloads = product.get("runs", {})
        if not isinstance(run_payloads, Mapping):
            continue
        for run_id, run_payload in run_payloads.items():
            if not isinstance(run_payload, Mapping):
                continue
            run_key = str(run_id)
            if run_key not in runs:
                run = dict(run_payload)
                rows = list(run_payload.get("perturbations", ()))
                runs[run_key] = run
            else:
                run = runs[run_key]
                rows = list(run.get("perturbations", ()))
                rows.extend(run_payload.get("perturbations", ()))
            run["perturbations"] = [
                dict(row) for row in rows if isinstance(row, Mapping)
            ]
            run["bulk_files"] = {
                **dict(run.get("bulk_files", {}) or {}),
                **dict(run_payload.get("bulk_files", {}) or {}),
            }
    for run in runs.values():
        rows = list(run.get("perturbations", ()))
        rows.sort(key=lambda row: int(row.get("class_response_row_index", 0)))
        for row in rows:
            row.pop("class_response_row_index", None)
        run["perturbations"] = rows
        run["status_counts"] = _perturbation_status_counts(rows)
        run["robust_response_summary"] = _summarize_perturbation_rows(rows)
    return runs


def _perturbation_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _summarize_perturbation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from rlrmp.analysis.pipelines.gru_perturbation_bank import summarize_perturbation_bank

    return summarize_perturbation_bank(rows)


def _empty_analysis_data() -> AnalysisInputData:
    return AnalysisInputData(
        models={},
        tasks={},
        states={},
        hps={},
        extras=TreeNamespace(),
    )


def _analysis_data_from_evaluation_input(evaluation_input: Any | None) -> AnalysisInputData:
    if evaluation_input is None:
        return _empty_analysis_data()
    ref = getattr(evaluation_input, "ref", None)
    path = _resolved_input_path(evaluation_input)
    return AnalysisInputData(
        models={},
        tasks={},
        states={
            "evaluation": _resolved_input_states(evaluation_input),
            "evaluation_manifest_id": getattr(ref, "id", None),
            "evaluation_manifest_path": None if path is None else str(path),
        },
        hps={},
        extras=TreeNamespace(),
    )


def _evaluation_input_from_analysis_data(data: AnalysisInputData) -> _AnalysisEvaluationInput | None:
    states = data.states.get("evaluation")
    if states is None:
        return None
    manifest_id = data.states.get("evaluation_manifest_id")
    path = data.states.get("evaluation_manifest_path")
    return _AnalysisEvaluationInput(
        states=states if isinstance(states, Mapping) else None,
        path=Path(path) if path not in (None, "") else None,
        ref=ParentRef(kind="EvaluationRunManifest", id=str(manifest_id))
        if manifest_id not in (None, "")
        else None,
    )


def _primary_evaluation_input(inputs: Sequence[Any]) -> Any | None:
    for resolved in inputs:
        ref = getattr(resolved, "ref", None)
        if getattr(ref, "kind", None) == "EvaluationRunManifest":
            return resolved
    return None


def _resolved_input_path(resolved: Any | None) -> Path | None:
    path = getattr(resolved, "path", None)
    return Path(path) if path is not None else None


def _resolved_input_states(resolved: Any | None) -> Mapping[str, Any] | None:
    states = getattr(resolved, "states", None)
    return states if isinstance(states, Mapping) else None


def _evaluation_dependency_metadata(resolved: Any) -> dict[str, Any]:
    ref = getattr(resolved, "ref", None)
    manifest = getattr(resolved, "manifest", None)
    states = _resolved_input_states(resolved) or {}
    return {
        "manifest_id": getattr(ref, "id", None),
        "evaluation_type": states.get(
            "evaluation_type",
            getattr(getattr(manifest, "evaluation_spec", None), "inline", {}).get(
                "evaluation_type"
            )
            if getattr(manifest, "evaluation_spec", None) is not None
            else None,
        ),
        "path": None if _resolved_input_path(resolved) is None else str(_resolved_input_path(resolved)),
        "product_role": states.get("product_role"),
    }


def _run_ids_from_params_or_inputs(
    params: Mapping[str, Any],
    inputs: Sequence[Any],
) -> tuple[str, ...]:
    if params.get("run_ids") is not None:
        return tuple(str(run_id) for run_id in params["run_ids"])
    run_ids: list[str] = []
    for resolved in inputs:
        ref = getattr(resolved, "ref", None)
        if ref is None:
            continue
        if getattr(ref, "kind", None) == "EvaluationRunManifest":
            manifest = getattr(resolved, "manifest", None)
            for parent in getattr(manifest, "input_training_runs", ()):
                parent_id = getattr(parent, "id", None)
                if parent_id is not None:
                    run_ids.append(str(parent_id))
            continue
        run_id = getattr(ref, "metadata", {}).get("run_id") or getattr(ref, "id", None)
        if run_id is not None:
            run_ids.append(str(run_id))
    return tuple(run_ids)


def _feedback_quality_run_ids_from_params_or_inputs(
    params: Mapping[str, Any],
    inputs: Sequence[Any],
) -> tuple[str, ...]:
    if params.get("run_ids") is not None:
        return tuple(str(run_id) for run_id in params["run_ids"])
    run_ids: list[str] = []
    for resolved in inputs:
        ref = getattr(resolved, "ref", None)
        if ref is None:
            continue
        if getattr(ref, "kind", None) == "EvaluationRunManifest":
            manifest = getattr(resolved, "manifest", None)
            for parent in getattr(manifest, "input_training_runs", ()):
                parent_id = getattr(parent, "id", None)
                if parent_id is not None:
                    run_ids.append(str(parent_id))
        elif getattr(ref, "kind", None) == "TrainingRunManifest":
            run_ids.append(str(getattr(ref, "id", "")))
    return tuple(run_id for run_id in run_ids if run_id)


def _experiment_from_params_or_inputs(
    params: Mapping[str, Any],
    inputs: Sequence[Any],
) -> str:
    if params.get("experiment") is not None:
        return str(params["experiment"])
    metadata_key = str(params.get("experiment_metadata_key", "rlrmp_experiment"))
    for resolved in inputs:
        metadata = {}
        manifest = getattr(resolved, "manifest", None)
        if manifest is not None:
            metadata = getattr(manifest, "metadata", {})
        if not metadata:
            ref = getattr(resolved, "ref", None)
            metadata = getattr(ref, "metadata", {}) if ref is not None else {}
        if metadata_key in metadata:
            return str(metadata[metadata_key])
    raise ValueError(
        f"GRU post-run recipe requires params.experiment or input metadata {metadata_key!r}"
    )


def _repo_root_from_params(params: Mapping[str, Any]) -> Path:
    value = params.get("repo_root")
    return Path(value).expanduser() if value is not None else REPO_ROOT


def _optional_path(value: Any, *, repo_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _optional_str_sequence(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(item) for item in value]


def _default_output_path(context: AnalysisRunContext, filename: str) -> Path:
    context.results_cache_dir.mkdir(parents=True, exist_ok=True)
    return context.results_cache_dir / filename


def _existing_file(
    path: Path,
    *,
    role: str,
    logical_name: str,
) -> ExistingAnalysisArtifact | None:
    if not path.exists():
        return None
    return ExistingAnalysisArtifact(path=path, role=role, logical_name=logical_name)


def _postrun_existing_artifacts(
    manifest: Mapping[str, Any],
    plan: Any,
    *,
    repo_root: Path,
) -> tuple[ExistingAnalysisArtifact, ...]:
    artifacts: list[ExistingAnalysisArtifact] = []
    direct_paths = (
        ("rlrmp-gru-checkpoint-selection-manifest", plan.checkpoint_manifest_path),
        ("rlrmp-gru-standard-certificate-note", plan.standard_note_path),
        ("rlrmp-gru-standard-certificate-manifest", plan.standard_manifest_path),
        ("rlrmp-gru-evaluation-diagnostics-manifest", plan.evaluation_manifest_path),
        ("rlrmp-gru-pilot-figure-summary", plan.figure_output_dir / "figure_summary.json"),
        ("rlrmp-gru-postrun-legacy-regeneration-spec", plan.postrun_regeneration_spec_path),
    )
    for role, path in direct_paths:
        if path is None:
            continue
        artifact = _existing_file(
            path, role=role, logical_name=_legacy_logical_name(path, repo_root)
        )
        if artifact is not None:
            artifacts.append(artifact)

    output_roles = {
        "objective_comparator": (
            "rlrmp-gru-objective-comparator-manifest",
            "rlrmp-gru-objective-comparator-note",
        ),
        "map_decomposition": (
            "rlrmp-gru-map-decomposition-manifest",
            "rlrmp-gru-map-decomposition-note",
        ),
        "perturbation_response": (
            "rlrmp-gru-perturbation-response-manifest",
            "rlrmp-gru-perturbation-response-note",
        ),
        "feedback_ablation": (
            "rlrmp-gru-feedback-ablation-manifest",
            "rlrmp-gru-feedback-ablation-note",
        ),
    }
    outputs = manifest.get("outputs", {})
    for output_name, (json_role, note_role) in output_roles.items():
        output = outputs.get(output_name) if isinstance(outputs, Mapping) else None
        if not isinstance(output, Mapping):
            continue
        for key, role in (("json_path", json_role), ("note_path", note_role)):
            path = _optional_path(output.get(key), repo_root=repo_root)
            if path is None:
                continue
            artifact = _existing_file(
                path,
                role=role,
                logical_name=_legacy_logical_name(path, repo_root),
            )
            if artifact is not None:
                artifacts.append(artifact)
    return tuple(artifacts)


def _legacy_logical_name(path: Path, repo_root: Path) -> str:
    return f"legacy/{_repo_relative(path, repo_root=repo_root)}"


def _feedback_quality_components(
    plan: Any,
    *,
    params: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    notes_dir = repo_root / "results" / plan.experiment / "notes"
    artifact_dir = repo_root / "_artifacts" / plan.experiment
    response_norm_topic = str(
        params.get("response_norm_plots_topic", f"perturbation_response_norms_{plan.output_tag}")
    )
    return {
        "evaluation_diagnostics": {
            "schema_kind": "RLRMPGRUEvaluationDiagnosticsManifest",
            "tracked": (
                _optional_path(
                    params.get("evaluation_diagnostics_manifest_path"),
                    repo_root=repo_root,
                )
                or plan.evaluation_manifest_path,
                "rlrmp-feedback-quality-evaluation-diagnostics-manifest",
            ),
            "groups": (
                (
                    _optional_path(
                        params.get("evaluation_diagnostics_bulk_dir"),
                        repo_root=repo_root,
                    )
                    or plan.evaluation_bulk_dir,
                    "rlrmp-feedback-quality-evaluation-diagnostics-bulk",
                    "feedback_quality_evaluation_diagnostics_bulk",
                    "rollout_arrays",
                ),
            ),
        },
        "objective_comparator": {
            "schema_kind": "RLRMPObjectiveComparatorSidecar",
            "tracked": (
                _optional_path(
                    params.get("objective_comparator_manifest_path"),
                    repo_root=repo_root,
                )
                or plan.objective_comparator_json_path,
                "rlrmp-feedback-quality-objective-comparator-manifest",
            ),
            "notes": (
                _optional_path(params.get("objective_comparator_note_path"), repo_root=repo_root)
                or plan.objective_comparator_note_path,
                "rlrmp-feedback-quality-objective-comparator-note",
            ),
        },
        "perturbation_response": {
            "schema_kind": "RLRMPGRUPerturbationBank",
            "tracked": (
                _optional_path(
                    params.get("perturbation_response_manifest_path"),
                    repo_root=repo_root,
                )
                or plan.perturbation_response_json_path,
                "rlrmp-feedback-quality-perturbation-response-manifest",
            ),
            "notes": (
                _optional_path(params.get("perturbation_response_note_path"), repo_root=repo_root)
                or plan.perturbation_response_note_path,
                "rlrmp-feedback-quality-perturbation-response-note",
            ),
            "groups": (
                (
                    _optional_path(
                        params.get("perturbation_response_bulk_dir"), repo_root=repo_root
                    )
                    or plan.perturbation_response_bulk_dir,
                    "rlrmp-feedback-quality-perturbation-response-bulk",
                    "feedback_quality_perturbation_response_bulk",
                    "perturbation_arrays",
                ),
            ),
        },
        "feedback_ablation": {
            "schema_kind": "RLRMPGRUFeedbackAblation",
            "tracked": (
                _optional_path(params.get("feedback_ablation_manifest_path"), repo_root=repo_root)
                or plan.feedback_ablation_json_path,
                "rlrmp-feedback-quality-feedback-ablation-manifest",
            ),
            "notes": (
                _optional_path(params.get("feedback_ablation_note_path"), repo_root=repo_root)
                or plan.feedback_ablation_note_path,
                "rlrmp-feedback-quality-feedback-ablation-note",
            ),
        },
        "response_norm_plots": {
            "schema_kind": "RLRMPGRUPerturbationResponseNormPlots",
            "tracked": (
                _optional_path(params.get("response_norm_plots_manifest_path"), repo_root=repo_root)
                or notes_dir
                / f"gru_perturbation_response_norm_plots_{plan.output_tag}_manifest.json",
                "rlrmp-feedback-quality-response-norm-plots-manifest",
            ),
            "notes": (
                _optional_path(params.get("response_norm_plots_note_path"), repo_root=repo_root)
                or notes_dir / f"gru_perturbation_response_norm_plots_{plan.output_tag}.md",
                "rlrmp-feedback-quality-response-norm-plots-note",
            ),
            "groups": (
                (
                    _optional_path(
                        params.get("response_norm_plots_figure_dir"), repo_root=repo_root
                    )
                    or artifact_dir / "figures" / response_norm_topic,
                    "rlrmp-feedback-quality-response-norm-figure",
                    "feedback_quality_response_norm_figures",
                    "plotly_html",
                ),
            ),
        },
        "perturbation_calibration": {
            "schema_kind": "RLRMPPerturbationOpenLoopCalibration",
            "tracked": (
                _optional_path(
                    params.get("perturbation_calibration_manifest_path"),
                    repo_root=repo_root,
                )
                or artifact_dir
                / "perturbation_open_loop_calibration"
                / "perturbation_open_loop_calibration.json",
                "rlrmp-feedback-quality-perturbation-calibration-manifest",
            ),
            "notes": (
                _optional_path(
                    params.get("perturbation_calibration_note_path"), repo_root=repo_root
                )
                or notes_dir / "perturbation_open_loop_calibration.md",
                "rlrmp-feedback-quality-perturbation-calibration-note",
            ),
            "groups": (
                (
                    _optional_path(
                        params.get("perturbation_calibration_bulk_dir"),
                        repo_root=repo_root,
                    )
                    or artifact_dir / "perturbation_open_loop_calibration",
                    "rlrmp-feedback-quality-perturbation-calibration-bulk",
                    "feedback_quality_perturbation_calibration_bulk",
                    "calibration_payload",
                ),
            ),
        },
    }


def _feedback_quality_component_output(
    registration: FeedbackQualityComponentRegistration,
    component: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], tuple[ExistingAnalysisArtifact, ...], tuple[AnalysisArtifactGroup, ...]]:
    paths = _component_paths(component)
    payload = {
        "status": "unavailable",
        "schema_kind": component["schema_kind"],
        "paths": {key: _repo_relative(path, repo_root=repo_root) for key, path in paths.items()},
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
    }
    gate = _feedback_quality_gating_decision(
        registration,
        params=params,
        component_status=payload,
    )
    if not gate.included:
        payload["status"] = "skipped"
        payload["reason"] = "component disabled by feedback-quality lens params"
        return payload, (), ()
    if gate.not_applicable:
        payload["status"] = "not_applicable"
        payload["reason"] = "component is not meaningful for this manifest set"
        return payload, (), ()

    existing: list[ExistingAnalysisArtifact] = []
    for key, value in component.items():
        if key not in {"tracked", "notes"}:
            continue
        path, role = value
        artifact = _existing_file(
            path,
            role=role,
            logical_name=_legacy_logical_name(path, repo_root),
        )
        if artifact is not None:
            existing.append(artifact)

    groups: list[AnalysisArtifactGroup] = []
    for directory, role, group_id, member_role in component.get("groups", ()):
        groups.extend(
            _directory_artifact_group(
                directory,
                role=role,
                group_id=group_id,
                member_role=member_role,
                repo_root=repo_root,
            )
        )

    if existing or groups:
        payload["status"] = "materialized"
        payload["artifact_roles"] = [artifact.role for artifact in existing] + [
            member.role for group in groups for member in group.members
        ]
        payload["artifact_group_ids"] = [group.group_id for group in groups]
    else:
        payload["reason"] = f"{registration.name} outputs were not found at configured paths"
    return payload, tuple(existing), tuple(groups)


def _component_paths(component: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, value in component.items():
        if key in {"tracked", "notes"}:
            paths[key] = value[0]
        elif key == "groups":
            for index, group in enumerate(value):
                paths[f"group_{index}"] = group[0]
    return paths


def _bulk_artifact_groups(
    manifest: Mapping[str, Any],
    *,
    group_id: str,
    repo_root: Path,
) -> tuple[AnalysisArtifactGroup, ...]:
    members: list[AnalysisArtifactFile] = []
    for run_id, run_payload in manifest.get("runs", {}).items():
        if not isinstance(run_payload, Mapping):
            continue
        bulk_arrays = run_payload.get("bulk_arrays")
        if not isinstance(bulk_arrays, Mapping):
            continue
        raw_path = bulk_arrays.get("path")
        if raw_path is None:
            continue
        path = _optional_path(raw_path, repo_root=repo_root)
        if path is None or not path.exists():
            continue
        members.append(
            AnalysisArtifactFile(
                path=path,
                role="rlrmp-gru-evaluation-diagnostics-bulk",
                logical_name=f"bulk/{run_id}.npz",
                metadata={"run_id": str(run_id)},
                group_role="rollout_arrays",
            )
        )
    if not members:
        return ()
    return (
        AnalysisArtifactGroup(
            group_id=group_id,
            members=tuple(members),
            metadata={"schema_boundary": "rlrmp-owned GRU diagnostic payload"},
        ),
    )


def _single_file_artifact_group(
    path: Path,
    *,
    role: str,
    logical_name: str,
    group_id: str,
    member_role: str,
    repo_root: Path,
) -> tuple[AnalysisArtifactGroup, ...]:
    if not path.exists():
        return ()
    return (
        AnalysisArtifactGroup(
            group_id=group_id,
            members=(
                AnalysisArtifactFile(
                    path=path,
                    role=role,
                    logical_name=logical_name,
                    metadata={"repo_relative_path": _repo_relative(path, repo_root=repo_root)},
                    group_role=member_role,
                ),
            ),
            metadata={"schema_boundary": "rlrmp-owned output-feedback bridge diagnostic payload"},
        ),
    )


def _directory_artifact_group(
    path: Path,
    *,
    role: str,
    group_id: str,
    member_role: str,
    repo_root: Path,
) -> tuple[AnalysisArtifactGroup, ...]:
    if not path.exists():
        return ()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        return ()
    members = tuple(
        AnalysisArtifactFile(
            path=item,
            role=role,
            logical_name=_repo_relative(item, repo_root=repo_root),
            metadata={"repo_relative_path": _repo_relative(item, repo_root=repo_root)},
            group_role=member_role,
        )
        for item in files
    )
    return (
        AnalysisArtifactGroup(
            group_id=group_id,
            members=members,
            metadata={"schema_boundary": "rlrmp-owned feedback-quality diagnostic payload"},
        ),
    )


def _read_json_payload(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _declarative_metadata(context: AnalysisRunContext) -> dict[str, Any]:
    return materialization_metadata(context, schema_owner="rlrmp")


def _set_optional_path_param(
    params: dict[str, Any],
    key: str,
    value: Path | str | None,
) -> None:
    if value is not None:
        params[key] = str(value)


def _evaluation_parent_refs(
    *,
    evaluation_manifest_id: str | None,
    evaluation_manifest_uri: Path | str | None,
) -> list[Any]:
    if evaluation_manifest_id is None:
        return []
    return [
        ParentRef(
            kind="EvaluationRunManifest",
            id=evaluation_manifest_id,
            role="evaluation_run",
            uri=None if evaluation_manifest_uri is None else str(evaluation_manifest_uri),
        )
    ]


__all__ = [
    "BRIDGE_STANDARD_ANALYSIS_TYPE",
    "FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES",
    "FEEDBACK_QUALITY_COMPONENT_NAMES",
    "FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE",
    "GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE",
    "GRU_POSTRUN_ANALYSIS_TYPE",
    "GRU_STANDARD_ANALYSIS_TYPE",
    "OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE",
    "PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE",
    "PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE",
    "ROBUSTNESS_PHENOTYPE_ANALYSIS_TYPE",
    "feedback_quality_lens_recipe",
    "feedback_quality_lens_spec",
    "gru_evaluation_diagnostics_spec",
    "gru_evaluation_diagnostics_recipe",
    "gru_postrun_recipe",
    "gru_postrun_spec",
    "gru_standard_certificate_spec",
    "gru_standard_certificate_recipe",
    "output_feedback_rollout_recovery_recipe",
    "output_feedback_rollout_recovery_spec",
    "perturbation_bank_aggregate_recipe",
    "perturbation_bank_aggregate_spec",
    "perturbation_class_response_recipe",
    "perturbation_class_response_spec",
    "register_certificate_analysis_recipes",
    "register_declarative_materialization_recipes",
    "robustness_phenotype_recipe",
    "robustness_phenotype_spec",
]

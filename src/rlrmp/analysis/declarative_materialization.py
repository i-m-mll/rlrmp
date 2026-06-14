"""Feedbax declarative recipes for rlrmp certificate materializers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from feedbax.analysis.context import AnalysisArtifactFile, AnalysisRunContext
from feedbax.analysis.materialization import (
    AnalysisArtifactGroup,
    ContextMaterializer,
    ExistingAnalysisArtifact,
    MaterializationResult,
    materialization_metadata,
)
from feedbax.analysis.specs import AnalysisRecipeResult, register_analysis_recipe
from feedbax.manifest import AnalysisRunSpec
from feedbax.types import AnalysisInputData, TreeNamespace

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import (
    MATERIALIZER_ISSUE_ID,
    RUN_IDS,
    SOURCE_ISSUE_ID,
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
    plan_gru_postrun_materialization,
)
from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
    ISSUE_ID as OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ISSUE_ID,
    write_outputs as write_output_feedback_rollout_recovery_outputs,
)
from rlrmp.analysis.rerun_metadata import DEFAULT_DISCRETIZATION, DEFAULT_LANE
from rlrmp.paths import REPO_ROOT


GRU_STANDARD_ANALYSIS_TYPE = "rlrmp.certificate.gru_standard"
GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE = "rlrmp.diagnostic.gru_evaluation"
GRU_POSTRUN_ANALYSIS_TYPE = "rlrmp.gru_postrun"
FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE = "rlrmp.feedback_quality_lens"
OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE = "rlrmp.output_feedback_bridge.rollout_recovery"
BRIDGE_STANDARD_ANALYSIS_TYPE = GRU_STANDARD_ANALYSIS_TYPE


def register_certificate_analysis_recipes(*, replace: bool = False) -> None:
    """Register rlrmp certificate/diagnostic analysis recipes with Feedbax."""

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
        FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE,
        feedback_quality_lens_recipe,
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
    return AnalysisRunSpec(
        analysis_type=GRU_STANDARD_ANALYSIS_TYPE,
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
    return AnalysisRunSpec(
        analysis_type=GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE,
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
    return AnalysisRunSpec(
        analysis_type=GRU_POSTRUN_ANALYSIS_TYPE,
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
    return AnalysisRunSpec(
        analysis_type=FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE,
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


def gru_standard_certificate_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    _inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative GRU standard-certificate recipe."""

    params = dict(spec.params)
    analysis = ContextMaterializer(
        materializer=lambda context: _materialize_gru_standard(context, params),
        artifact_role="rlrmp-bridge-standard-certificate",
        logical_name="gru_standard_certificates.json",
        schema_boundary="rlrmp-owned BridgeRunManifest/certificate payload",
    )
    return AnalysisRecipeResult(
        analyses={"gru_standard_certificate": analysis},
        data=_empty_analysis_data(),
    )


def gru_evaluation_diagnostics_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    _inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative GRU rollout-diagnostics recipe."""

    params = dict(spec.params)
    analysis = ContextMaterializer(
        materializer=lambda context: _materialize_gru_evaluation_diagnostics(context, params),
        artifact_role="rlrmp-gru-evaluation-diagnostics",
        logical_name="gru_evaluation_diagnostics.json",
        schema_boundary="rlrmp-owned GRU diagnostic payload",
    )
    return AnalysisRecipeResult(
        analyses={"gru_evaluation_diagnostics": analysis},
        data=_empty_analysis_data(),
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
    analysis = ContextMaterializer(
        materializer=lambda context: _materialize_gru_postrun(
            context,
            params,
            experiment=experiment,
            run_ids=resolved_run_ids,
        ),
        artifact_role="rlrmp-gru-postrun-manifest",
        logical_name="gru_postrun_materialization.json",
        schema_boundary="rlrmp-owned GRU post-run diagnostic bundle payload",
    )
    return AnalysisRecipeResult(
        analyses={"gru_postrun_materialization": analysis},
        data=_empty_analysis_data(),
    )


def output_feedback_rollout_recovery_recipe(
    spec: AnalysisRunSpec,
    _root: Path,
    _inputs: Sequence[Any],
) -> AnalysisRecipeResult:
    """Build the declarative output-feedback rollout-recovery recipe."""

    params = dict(spec.params)
    analysis = ContextMaterializer(
        materializer=lambda context: _materialize_output_feedback_rollout_recovery(
            context,
            params,
        ),
        artifact_role="rlrmp-output-feedback-rollout-recovery",
        logical_name="output_feedback_rollout_recovery.json",
        schema_boundary="rlrmp-owned output-feedback bridge diagnostic payload",
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
    resolved_run_ids = _run_ids_from_params_or_inputs(params, inputs)
    experiment = _experiment_from_params_or_inputs(params, inputs)
    analysis = ContextMaterializer(
        materializer=lambda context: _materialize_feedback_quality_lens(
            context,
            params,
            experiment=experiment,
            run_ids=resolved_run_ids,
        ),
        artifact_role="rlrmp-feedback-quality-lens",
        logical_name="feedback_quality_lens.json",
        schema_boundary="rlrmp-owned feedback-control quality lens payload",
    )
    return AnalysisRecipeResult(
        analyses={"feedback_quality_lens": analysis},
        data=_empty_analysis_data(),
    )


def _materialize_gru_standard(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
) -> MaterializationResult:
    run_ids = tuple(str(run_id) for run_id in params.get("run_ids", RUN_IDS))
    experiment = str(params.get("experiment", SOURCE_ISSUE_ID))
    repo_root = _repo_root_from_params(params)
    result = materialize_gru_standard_result(
        run_ids=run_ids,
        load_models=bool(params.get("load_models", True)),
        experiment=experiment,
        materializer_issue_id=str(params.get("materializer_issue_id", MATERIALIZER_ISSUE_ID)),
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
    return MaterializationResult(
        payload=result,
        existing_artifacts=tuple(existing_artifacts),
    )


def _materialize_gru_evaluation_diagnostics(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
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


def _materialize_feedback_quality_lens(
    context: AnalysisRunContext,
    params: Mapping[str, Any],
    *,
    experiment: str,
    run_ids: Sequence[str],
) -> MaterializationResult:
    if not run_ids:
        raise ValueError("Feedback-quality lens recipe requires at least one run ID")

    repo_root = _repo_root_from_params(params)
    output_tag = str(params.get("output_tag", DEFAULT_OUTPUT_TAG))
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
    not_applicable = set(_optional_str_sequence(params.get("not_applicable_components")) or [])
    components = _feedback_quality_components(plan, params=params, repo_root=repo_root)

    outputs: dict[str, dict[str, Any]] = {}
    existing_artifacts: list[ExistingAnalysisArtifact] = []
    artifact_groups: list[AnalysisArtifactGroup] = []
    for name, component in components.items():
        output, existing, groups = _feedback_quality_component_output(
            name,
            component,
            include=bool(params.get(f"include_{name}", True)),
            not_applicable=name in not_applicable,
            repo_root=repo_root,
        )
        outputs[name] = output
        existing_artifacts.extend(existing)
        artifact_groups.extend(groups)

    payload = {
        "schema_id": "rlrmp.feedback_quality_lens",
        "schema_version": "rlrmp.feedback_quality_lens.v1",
        "issue": str(params.get("issue", "af77a06")),
        "scope": "feedback_control_quality_diagnostics",
        "experiment": experiment,
        "run_ids": list(run_ids),
        "labels": _optional_str_sequence(params.get("labels")),
        "output_tag": output_tag,
        "checkpoint_policy": plan.checkpoint_policy,
        "checkpoint_selection_source": plan.checkpoint_selection_source,
        "selection_leakage_guard": {
            "status": "audit_only",
            "primary_checkpoint_selection": plan.checkpoint_selection_source,
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
            "analysis_manifest_id": context.manifest_id,
            "scientific_schema_owner": "rlrmp",
            "artifact_custody": "feedbax.AnalysisRunManifest",
        },
        "declarative_analysis": _declarative_metadata(context),
    }
    return MaterializationResult(
        payload=payload,
        existing_artifacts=tuple(existing_artifacts),
        artifact_groups=tuple(artifact_groups),
    )


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
        },
        existing_artifacts=existing_artifacts,
        artifact_groups=artifact_groups,
    )


def _empty_analysis_data() -> AnalysisInputData:
    return AnalysisInputData(
        models={},
        tasks={},
        states={},
        hps={},
        extras=TreeNamespace(),
    )


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
        run_id = getattr(ref, "metadata", {}).get("run_id") or getattr(ref, "id", None)
        if run_id is not None:
            run_ids.append(str(run_id))
    return tuple(run_ids)


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
    name: str,
    component: Mapping[str, Any],
    *,
    include: bool,
    not_applicable: bool,
    repo_root: Path,
) -> tuple[dict[str, Any], tuple[ExistingAnalysisArtifact, ...], tuple[AnalysisArtifactGroup, ...]]:
    paths = _component_paths(component)
    payload = {
        "status": "unavailable",
        "schema_kind": component["schema_kind"],
        "paths": {key: _repo_relative(path, repo_root=repo_root) for key, path in paths.items()},
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
    }
    if not include:
        payload["status"] = "skipped"
        payload["reason"] = "component disabled by feedback-quality lens params"
        return payload, (), ()
    if not_applicable:
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
        payload["reason"] = f"{name} outputs were not found at configured paths"
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


__all__ = [
    "BRIDGE_STANDARD_ANALYSIS_TYPE",
    "FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE",
    "GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE",
    "GRU_POSTRUN_ANALYSIS_TYPE",
    "GRU_STANDARD_ANALYSIS_TYPE",
    "OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE",
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
    "register_certificate_analysis_recipes",
    "register_declarative_materialization_recipes",
]

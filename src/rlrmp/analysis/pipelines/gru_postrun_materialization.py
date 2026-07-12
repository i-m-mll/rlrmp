"""One-command post-run materialization for C&S GRU runs."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rlrmp.analysis.gru_standard_certificate import (
    MATERIALIZER_ISSUE_ID,
    materialize_gru_standard_result,
)
from rlrmp.analysis.pipelines.diagnostic_provenance import write_regeneration_spec
from rlrmp.eval.checkpoint_selection import (
    build_validation_checkpoint_selection_manifest,
    checkpoint_selection_rows,
    load_materialized_fixed_bank_manifest,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    materialize_gru_pilot_figures,
)
from rlrmp.paths import REPO_ROOT, mkdir_p, run_spec_path


SCHEMA_VERSION = "rlrmp.gru_postrun_materialization.v1"
VALIDATION_SELECTED_CHECKPOINT_POLICY = "validation_selected_per_replicate"
FINAL_CHECKPOINT_POLICY = "final_checkpoint"
DEFAULT_OUTPUT_TAG = "validation_selected"


@dataclass(frozen=True)
class GruPostrunMaterializationPlan:
    """Concrete output paths for a GRU post-run materialization."""

    experiment: str
    run_ids: tuple[str, ...]
    output_tag: str
    checkpoint_policy: str
    checkpoint_selection_source: str
    notes_dir: Path
    checkpoint_manifest_path: Path | None
    fixed_bank_rescore_manifest_path: Path | None
    standard_note_path: Path
    standard_manifest_path: Path
    evaluation_manifest_path: Path
    evaluation_bulk_dir: Path
    figure_output_dir: Path
    objective_comparator_json_path: Path
    objective_comparator_note_path: Path
    map_decomposition_json_path: Path
    map_decomposition_note_path: Path
    perturbation_response_json_path: Path
    perturbation_response_note_path: Path
    perturbation_response_bulk_dir: Path
    feedback_ablation_json_path: Path
    feedback_ablation_note_path: Path
    postrun_manifest_path: Path
    postrun_regeneration_spec_path: Path

    def to_json(self, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
        """Return a JSON-compatible plan with repo-relative paths."""

        payload = asdict(self)
        for key, value in tuple(payload.items()):
            if isinstance(value, Path):
                payload[key] = _repo_relative(value, repo_root=repo_root)
            elif value is None:
                payload[key] = None
        payload["run_ids"] = list(self.run_ids)
        return payload


def plan_gru_postrun_materialization(
    *,
    experiment: str,
    run_ids: Sequence[str],
    output_tag: str = DEFAULT_OUTPUT_TAG,
    use_validation_selected_checkpoints: bool = True,
    fixed_bank_rescore_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> GruPostrunMaterializationPlan:
    """Return the tracked and ignored output paths for a post-run materialization."""

    if not run_ids:
        raise ValueError("At least one run ID is required")
    checkpoint_policy = checkpoint_policy_name(use_validation_selected_checkpoints)
    notes_dir = repo_root / "results" / experiment / "notes"
    artifact_dir = repo_root / "_artifacts" / experiment
    fixed_bank_manifest = None
    fixed_bank_available = False
    if use_validation_selected_checkpoints:
        fixed_bank_manifest = load_materialized_fixed_bank_manifest(
            manifest_path=fixed_bank_rescore_manifest_path,
        )
        fixed_bank_available = fixed_bank_manifest is not None and all(
            run_id in checkpoint_selection_rows(fixed_bank_manifest) for run_id in run_ids
        )
    effective_checkpoint_policy = (
        str(
            fixed_bank_manifest.metadata.get("checkpoint_policy")
            or "fixed_bank_rescored_per_replicate"
        )
        if fixed_bank_available and fixed_bank_manifest is not None
        else checkpoint_policy
    )
    return GruPostrunMaterializationPlan(
        experiment=experiment,
        run_ids=tuple(run_ids),
        output_tag=output_tag,
        checkpoint_policy=effective_checkpoint_policy,
        checkpoint_selection_source=(
            "fixed_bank_rescore" if fixed_bank_available else checkpoint_policy
        ),
        notes_dir=notes_dir,
        checkpoint_manifest_path=None,
        fixed_bank_rescore_manifest_path=(
            fixed_bank_rescore_manifest_path if use_validation_selected_checkpoints else None
        ),
        standard_note_path=notes_dir / f"gru_standard_certificates_{output_tag}.md",
        standard_manifest_path=notes_dir / f"gru_standard_certificates_{output_tag}_manifest.json",
        evaluation_manifest_path=notes_dir / f"gru_evaluation_diagnostics_{output_tag}.json",
        evaluation_bulk_dir=artifact_dir / "evaluation_diagnostics" / f"gru_{output_tag}",
        figure_output_dir=artifact_dir / "figures" / f"gru_postrun_{output_tag}",
        objective_comparator_json_path=notes_dir / f"objective_comparator_{output_tag}.json",
        objective_comparator_note_path=notes_dir / f"objective_comparator_{output_tag}.md",
        map_decomposition_json_path=notes_dir / f"gru_map_error_decomposition_{output_tag}.json",
        map_decomposition_note_path=notes_dir / f"gru_map_error_decomposition_{output_tag}.md",
        perturbation_response_json_path=(
            notes_dir / f"gru_perturbation_response_{output_tag}_manifest.json"
        ),
        perturbation_response_note_path=notes_dir / f"gru_perturbation_response_{output_tag}.md",
        perturbation_response_bulk_dir=(
            artifact_dir / "perturbation_response" / f"gru_{output_tag}"
        ),
        feedback_ablation_json_path=notes_dir / f"gru_feedback_ablation_{output_tag}_manifest.json",
        feedback_ablation_note_path=notes_dir / f"gru_feedback_ablation_{output_tag}.md",
        postrun_manifest_path=notes_dir / f"gru_postrun_materialization_{output_tag}.json",
        postrun_regeneration_spec_path=(
            notes_dir / f"gru_postrun_materialization_{output_tag}_regeneration_spec.json"
        ),
    )


def materialize_gru_postrun_analysis(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    output_tag: str = DEFAULT_OUTPUT_TAG,
    use_validation_selected_checkpoints: bool = True,
    fixed_bank_rescore_manifest_path: Path | None = None,
    include_reference: bool = True,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    materializer_issue_id: str = MATERIALIZER_ISSUE_ID,
    include_objective_comparator: bool = True,
    include_map_decomposition: bool = True,
    include_perturbation_response: bool = True,
    include_feedback_ablation: bool = True,
    perturbation_bank_mode: str = "raw",
    perturbation_calibration_level: str | Sequence[str] | None = None,
    perturbation_calibration_reach: str | float | None = None,
    write_perturbation_bulk_arrays: bool = False,
    feedback_selection_level: str = "small",
    evaluation_manifest_path: Path | None = None,
    evaluation_states: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize the standard post-run GRU analysis bundle.

    The default checkpoint policy is validation-selected per replicate. Standard
    certificate, I/O map, covariance-weighted map, and objective-comparator
    values remain audit sidecars and are never used for checkpoint selection.
    """

    if write_perturbation_bulk_arrays:
        raise ValueError(
            "direct perturbation bulk-array writes are retired; run the registered "
            "evaluation and analysis bundle to obtain Feedbax-custody artifacts"
        )
    if evaluation_states is None:
        raise ValueError(
            "GRU post-run analysis requires cached states from an EvaluationRunManifest; "
            "it may not rerun diagnostic rollouts"
        )
    run_ids = tuple(run_ids)
    plan = plan_gru_postrun_materialization(
        experiment=experiment,
        run_ids=run_ids,
        output_tag=output_tag,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        fixed_bank_rescore_manifest_path=fixed_bank_rescore_manifest_path,
        repo_root=repo_root,
    )
    mkdir_p(plan.notes_dir)
    effective_checkpoint_manifest_path = (
        plan.fixed_bank_rescore_manifest_path
        if plan.checkpoint_selection_source == "fixed_bank_rescore"
        else None
    )

    checkpoint_manifest: dict[str, Any] | None = None
    if use_validation_selected_checkpoints:
        checkpoint_manifest = build_validation_checkpoint_selection_manifest(
            experiment=experiment,
            run_ids=run_ids,
            preferred_manifest_path=effective_checkpoint_manifest_path,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if effective_checkpoint_manifest_path is not None
                else "sparse_history"
            ),
            repo_root=repo_root,
        ).model_dump(mode="json", exclude_none=True)

    standard_result = materialize_gru_standard_result(
        evaluation_states,
        run_ids=run_ids,
        experiment=experiment,
        materializer_issue_id=materializer_issue_id,
        repo_root=repo_root,
    )

    evaluation_manifest = dict(evaluation_states)

    figure_summary = materialize_gru_pilot_figures(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        output_dir=plan.figure_output_dir,
        n_rollout_trials=n_rollout_trials,
        include_reference=include_reference,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        preferred_checkpoint_manifest_path=effective_checkpoint_manifest_path,
        repo_root=repo_root,
    )

    objective_comparator = (
        materialize_optional_objective_comparator(
            experiment=experiment,
            run_ids=run_ids,
            labels=labels,
            checkpoint_policy=plan.checkpoint_policy,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_manifest_path=plan.checkpoint_manifest_path,
            standard_manifest_path=plan.standard_manifest_path,
            output_path=plan.objective_comparator_json_path,
            note_path=plan.objective_comparator_note_path,
            repo_root=repo_root,
        )
        if include_objective_comparator
        else {"status": "skipped", "reason": "disabled_by_cli"}
    )
    map_decomposition = (
        materialize_optional_map_error_decomposition(
            experiment=experiment,
            run_ids=run_ids,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            standard_manifest_path=plan.standard_manifest_path,
            preferred_checkpoint_manifest_path=effective_checkpoint_manifest_path,
            output_path=plan.map_decomposition_json_path,
            note_path=plan.map_decomposition_note_path,
            repo_root=repo_root,
        )
        if include_map_decomposition
        else {"status": "skipped", "reason": "disabled_by_cli"}
    )
    split_stress_objective_comparator = split_stress_objective_comparator_status(
        objective_comparator
    )
    perturbation_response = (
        materialize_optional_perturbation_response(
            experiment=experiment,
            run_ids=run_ids,
            labels=labels,
            n_rollout_trials=n_rollout_trials,
            bank_mode=perturbation_bank_mode,
            calibration_level=perturbation_calibration_level,
            calibration_reach=perturbation_calibration_reach,
            feedback_scale_manifest_path=evaluation_manifest_path,
            preferred_checkpoint_manifest_path=effective_checkpoint_manifest_path,
            repo_root=repo_root,
        )
        if include_perturbation_response
        else {"status": "skipped", "reason": "disabled_by_cli"}
    )
    feedback_ablation = (
        materialize_optional_feedback_ablation(
            experiment=experiment,
            run_ids=run_ids,
            labels=labels,
            n_rollout_trials=n_rollout_trials,
            output_path=plan.feedback_ablation_json_path,
            note_path=plan.feedback_ablation_note_path,
            bank_mode=perturbation_bank_mode,
            calibration_level=perturbation_calibration_level,
            calibration_reach=perturbation_calibration_reach,
            feedback_selection_level=feedback_selection_level,
            feedback_scale_manifest_path=evaluation_manifest_path,
            preferred_checkpoint_manifest_path=effective_checkpoint_manifest_path,
            regeneration_spec_path=_regeneration_spec_path(plan.feedback_ablation_json_path),
            repo_root=repo_root,
        )
        if include_feedback_ablation
        else {"status": "skipped", "reason": "disabled_by_cli"}
    )
    feedback_checkpoint_selection = feedback_checkpoint_selection_status(feedback_ablation)
    regeneration_specs = _postrun_regeneration_specs(plan, repo_root=repo_root)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": experiment,
        "run_ids": list(run_ids),
        "labels": None if labels is None else list(labels),
        "checkpoint_policy": plan.checkpoint_policy,
        "checkpoint_selection_source": plan.checkpoint_selection_source,
        "selection_leakage_guard": {
            "status": "audit_only",
            "audit_only_metrics": [
                "state_weighted_action_mismatch",
                "clean_action_mismatch",
                "observation_history_to_action_map_mismatch",
                "covariance_weighted_map_mismatch",
                "map_error_decomposition",
                "extlqg_objective_comparator",
                "split_stress_bank_objective_comparator",
                "perturbation_response_bank",
                "feedback_ablation",
                "feedback_selected_checkpoint_audit",
            ],
            "note": _selection_leakage_guard_note(plan),
        },
        "primary_run_contract": {
            "type": "feedbax_analysis_bundle",
            "bundle": "rlrmp/gru_postrun",
            "legacy_regeneration_spec": "compatibility_only",
            "evaluation_manifest_dependency": (
                None
                if evaluation_manifest_path is None
                else _repo_relative(evaluation_manifest_path, repo_root=repo_root)
            ),
        },
        "plan": plan.to_json(repo_root=repo_root),
        "regeneration_specs": regeneration_specs,
        "perturbation_bank": {
            "mode": perturbation_bank_mode,
            "calibration_level": (
                None
                if perturbation_calibration_level is None
                else list(perturbation_calibration_level)
                if not isinstance(perturbation_calibration_level, str)
                else perturbation_calibration_level
            ),
            "calibration_reach": perturbation_calibration_reach,
            "feedback_selection_level": feedback_selection_level,
            "write_perturbation_bulk_arrays": bool(write_perturbation_bulk_arrays),
        },
        "outputs": {
            "checkpoint_manifest": (
                None
                if plan.checkpoint_manifest_path is None
                else _repo_relative(plan.checkpoint_manifest_path, repo_root=repo_root)
            ),
            "fixed_bank_rescore_manifest": (
                None
                if plan.fixed_bank_rescore_manifest_path is None
                else fixed_bank_rescore_manifest_status(
                    plan.fixed_bank_rescore_manifest_path,
                    repo_root=repo_root,
                )
            ),
            "standard_certificate_note": _repo_relative(
                plan.standard_note_path,
                repo_root=repo_root,
            ),
            "standard_certificate_manifest": _repo_relative(
                plan.standard_manifest_path,
                repo_root=repo_root,
            ),
            "evaluation_run_manifest": (
                None
                if evaluation_manifest_path is None
                else _repo_relative(evaluation_manifest_path, repo_root=repo_root)
            ),
            "figure_output_dir": _repo_relative(plan.figure_output_dir, repo_root=repo_root),
            "figure_summary": _repo_relative(
                plan.figure_output_dir / "figure_summary.json",
                repo_root=repo_root,
            ),
            "objective_comparator": objective_comparator,
            "split_stress_objective_comparator": split_stress_objective_comparator,
            "map_decomposition": map_decomposition,
            "perturbation_response": perturbation_response,
            "feedback_ablation": feedback_ablation,
            "feedback_checkpoint_selection": feedback_checkpoint_selection,
        },
        "summaries": {
            "standard_certificate": standard_result.get("summary", {}),
            "evaluation_manifest_id": evaluation_manifest.get("evaluation_manifest_id"),
            "evaluation_product_role": evaluation_manifest.get("product_role"),
            "figure_summary_keys": sorted(figure_summary.keys()),
        },
    }
    plan.postrun_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_postrun_auxiliary_regeneration_specs(
        plan=plan,
        run_ids=run_ids,
        labels=labels,
        include_reference=include_reference,
        n_rollout_trials=n_rollout_trials,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        effective_checkpoint_manifest_path=effective_checkpoint_manifest_path,
        include_objective_comparator=include_objective_comparator,
        include_map_decomposition=include_map_decomposition,
        include_perturbation_response=include_perturbation_response,
        include_feedback_ablation=include_feedback_ablation,
        perturbation_bank_mode=perturbation_bank_mode,
        perturbation_calibration_level=perturbation_calibration_level,
        perturbation_calibration_reach=perturbation_calibration_reach,
        write_perturbation_bulk_arrays=write_perturbation_bulk_arrays,
        feedback_selection_level=feedback_selection_level,
        evaluation_manifest_path=evaluation_manifest_path,
        repo_root=repo_root,
    )
    return manifest


def materialize_optional_feedback_ablation(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    n_rollout_trials: int,
    output_path: Path,
    note_path: Path,
    bank_mode: str = "raw",
    calibration_level: str | Sequence[str] | None = None,
    calibration_reach: str | float | None = None,
    feedback_selection_level: str = "small",
    feedback_scale_manifest_path: Path | None = None,
    preferred_checkpoint_manifest_path: Path | None = None,
    regeneration_spec_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the optional feedback-ablation analysis through Feedbax manifests."""

    from rlrmp.analysis.pipelines.gru_feedback_ablation import (
        execute_feedback_ablation_pipeline,
    )

    try:
        execution = execute_feedback_ablation_pipeline(
            source_experiment=experiment,
            result_experiment=experiment,
            scope="postrun_feedback_ablation",
            run_ids=tuple(run_ids),
            labels=None if labels is None else tuple(labels),
            n_rollout_trials=n_rollout_trials,
            bank_mode=bank_mode,
            calibration_level=calibration_level,
            calibration_reach=calibration_reach,
            feedback_selection_level=feedback_selection_level,
            feedback_scale_manifest_path=feedback_scale_manifest_path,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
            repo_root=repo_root,
            feedbax_runs_root=repo_root / "_artifacts" / experiment / "feedbax_runs",
            issues=(experiment,),
        )
    except (FileNotFoundError, ValueError, KeyError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "feedback_ablation_inputs_unavailable",
            "detail": str(exc),
            "json_path": _repo_relative(output_path, repo_root=repo_root),
            "note_path": _repo_relative(note_path, repo_root=repo_root),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }

    result = execution.payload
    runs = result.get("runs", {}) if isinstance(result, dict) else {}
    audit = (
        result.get("feedback_checkpoint_selection_audit", {}) if isinstance(result, dict) else {}
    )
    return {
        "status": "materialized",
        "json_path": str(execution.analysis_manifest_path),
        "note_path": None,
        "bulk_detail_manifest": None,
        "regeneration_spec": None,
        "evaluation_manifest_id": execution.evaluation_manifest.id,
        "analysis_manifest_id": execution.analysis_manifest.id,
        "custody_route": "EvaluationRunManifest->AnalysisRunManifest",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "result": {
            "schema_version": result.get("schema_version") if isinstance(result, dict) else None,
            "n_runs": len(runs),
            "checkpoint_policy": (
                result.get("checkpoint_policy") if isinstance(result, dict) else None
            ),
            "feedback_checkpoint_selection_audit_status": (
                audit.get("status") if isinstance(audit, dict) else None
            ),
        },
        "feedback_checkpoint_selection_audit": audit,
    }


def _postrun_regeneration_specs(
    plan: GruPostrunMaterializationPlan,
    *,
    repo_root: Path,
) -> dict[str, str]:
    """Return the tracked regeneration-spec index for a post-run bundle."""

    return {
        "postrun": _repo_relative(plan.postrun_regeneration_spec_path, repo_root=repo_root),
        "standard_certificate": _repo_relative(
            _regeneration_spec_path(plan.standard_manifest_path),
            repo_root=repo_root,
        ),
        "pilot_figures": _repo_relative(
            plan.notes_dir / f"gru_pilot_figures_{plan.output_tag}_regeneration_spec.json",
            repo_root=repo_root,
        ),
        "objective_comparator": _repo_relative(
            _regeneration_spec_path(plan.objective_comparator_json_path),
            repo_root=repo_root,
        ),
        "map_decomposition": _repo_relative(
            _regeneration_spec_path(plan.map_decomposition_json_path),
            repo_root=repo_root,
        ),
        "perturbation_response": _repo_relative(
            _regeneration_spec_path(plan.perturbation_response_json_path),
            repo_root=repo_root,
        ),
        "feedback_ablation": _repo_relative(
            _regeneration_spec_path(plan.feedback_ablation_json_path),
            repo_root=repo_root,
        ),
    }


def _write_postrun_auxiliary_regeneration_specs(
    *,
    plan: GruPostrunMaterializationPlan,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    include_reference: bool,
    n_rollout_trials: int,
    use_validation_selected_checkpoints: bool,
    effective_checkpoint_manifest_path: Path | None,
    include_objective_comparator: bool,
    include_map_decomposition: bool,
    include_perturbation_response: bool,
    include_feedback_ablation: bool,
    perturbation_bank_mode: str,
    perturbation_calibration_level: str | Sequence[str] | None,
    perturbation_calibration_reach: str | float | None,
    write_perturbation_bulk_arrays: bool,
    feedback_selection_level: str,
    evaluation_manifest_path: Path | None,
    repo_root: Path,
) -> None:
    """Write postrun-owned specs for optional hooks without native spec support."""

    run_inputs = _run_input_refs(plan.experiment, run_ids, repo_root=repo_root)
    evaluation_inputs = (
        []
        if evaluation_manifest_path is None
        else [{"role": "evaluation_run_manifest", "path": evaluation_manifest_path}]
    )
    checkpoint_inputs = (
        []
        if effective_checkpoint_manifest_path is None
        else [{"role": "checkpoint_manifest", "path": effective_checkpoint_manifest_path}]
    )
    write_regeneration_spec(
        spec_path=plan.notes_dir / f"gru_pilot_figures_{plan.output_tag}_regeneration_spec.json",
        diagnostic_name="gru_pilot_figures",
        materializer="rlrmp.analysis.pipelines.gru_pilot_figures.materialize_gru_pilot_figures",
        command=None,
        parameters={
            "experiment": plan.experiment,
            "run_ids": list(run_ids),
            "labels": None if labels is None else list(labels),
            "n_rollout_trials": n_rollout_trials,
            "include_reference": include_reference,
            "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
            "preferred_checkpoint_manifest_path": (
                None
                if effective_checkpoint_manifest_path is None
                else _repo_relative(effective_checkpoint_manifest_path, repo_root=repo_root)
            ),
        },
        inputs=run_inputs + checkpoint_inputs,
        outputs=[
            {"role": "pilot_figure_dir", "path": plan.figure_output_dir},
            {
                "role": "pilot_figure_summary",
                "path": plan.figure_output_dir / "figure_summary.json",
            },
        ],
        source_files=["src/rlrmp/analysis/pipelines/gru_pilot_figures.py"],
        notes=["Postrun-owned regeneration spec for pilot loss/velocity figures."],
        repo_root=repo_root,
    )
    if include_objective_comparator:
        write_regeneration_spec(
            spec_path=_regeneration_spec_path(plan.objective_comparator_json_path),
            diagnostic_name="gru_objective_comparator",
            materializer="rlrmp.analysis.pipelines.objective_comparator.materialize_gru_objective_comparator_sidecar",
            command=None,
            parameters={
                "experiment": plan.experiment,
                "run_ids": list(run_ids),
                "labels": None if labels is None else list(labels),
                "checkpoint_policy": plan.checkpoint_policy,
                "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
            },
            inputs=run_inputs
            + evaluation_inputs
            + checkpoint_inputs
            + [{"role": "standard_certificate_manifest", "path": plan.standard_manifest_path}],
            outputs=[
                {
                    "role": "objective_comparator_manifest",
                    "path": plan.objective_comparator_json_path,
                },
                {"role": "objective_comparator_note", "path": plan.objective_comparator_note_path},
            ],
            source_files=[
                "src/rlrmp/analysis/pipelines/objective_comparator.py",
                "src/rlrmp/analysis/math/cs_released_simulation.py",
            ],
            notes=["Postrun-owned regeneration spec for objective-comparator sidecar."],
            repo_root=repo_root,
        )
    if include_map_decomposition:
        write_regeneration_spec(
            spec_path=_regeneration_spec_path(plan.map_decomposition_json_path),
            diagnostic_name="gru_map_error_decomposition",
            materializer="rlrmp.analysis.pipelines.gru_map_error_decomposition.materialize_gru_map_error_decomposition",
            command=None,
            parameters={
                "experiment": plan.experiment,
                "run_ids": list(run_ids),
                "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
            },
            inputs=run_inputs
            + evaluation_inputs
            + checkpoint_inputs
            + [{"role": "standard_certificate_manifest", "path": plan.standard_manifest_path}],
            outputs=[
                {"role": "map_decomposition_manifest", "path": plan.map_decomposition_json_path},
                {"role": "map_decomposition_note", "path": plan.map_decomposition_note_path},
            ],
            source_files=["src/rlrmp/analysis/pipelines/gru_map_error_decomposition.py"],
            notes=["Postrun-owned regeneration spec for target-relative map-error decomposition."],
            repo_root=repo_root,
        )
    write_regeneration_spec(
        spec_path=plan.postrun_regeneration_spec_path,
        diagnostic_name="gru_postrun_materialization_bundle",
        materializer="rlrmp.analysis.pipelines.gru_postrun_materialization.materialize_gru_postrun_analysis",
        command=None,
        parameters={
            "experiment": plan.experiment,
            "run_ids": list(run_ids),
            "labels": None if labels is None else list(labels),
            "output_tag": plan.output_tag,
            "use_validation_selected_checkpoints": use_validation_selected_checkpoints,
            "include_reference": include_reference,
            "n_rollout_trials": n_rollout_trials,
            "include_objective_comparator": include_objective_comparator,
            "include_map_decomposition": include_map_decomposition,
            "include_perturbation_response": include_perturbation_response,
            "include_feedback_ablation": include_feedback_ablation,
            "perturbation_bank_mode": perturbation_bank_mode,
            "perturbation_calibration_level": perturbation_calibration_level,
            "perturbation_calibration_reach": perturbation_calibration_reach,
            "write_perturbation_bulk_arrays": bool(write_perturbation_bulk_arrays),
            "feedback_selection_level": feedback_selection_level,
            "effective_checkpoint_manifest_path": (
                None
                if effective_checkpoint_manifest_path is None
                else _repo_relative(effective_checkpoint_manifest_path, repo_root=repo_root)
            ),
        },
        inputs=run_inputs + evaluation_inputs + checkpoint_inputs,
        outputs=[
            {"role": "postrun_manifest", "path": plan.postrun_manifest_path},
            {"role": "standard_certificate_manifest", "path": plan.standard_manifest_path},
            {"role": "pilot_figure_dir", "path": plan.figure_output_dir},
            {"role": "objective_comparator_manifest", "path": plan.objective_comparator_json_path},
            {"role": "map_decomposition_manifest", "path": plan.map_decomposition_json_path},
            {
                "role": "perturbation_response_manifest",
                "path": plan.perturbation_response_json_path,
            },
            {"role": "feedback_ablation_manifest", "path": plan.feedback_ablation_json_path},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/gru_postrun_materialization.py",
            "src/rlrmp/analysis/gru_standard_certificate.py",
            "src/rlrmp/eval/evaluation_diagnostics.py",
            "src/rlrmp/eval/gru_diagnostics.py",
            "src/rlrmp/analysis/pipelines/gru_pilot_figures.py",
            "src/rlrmp/eval/perturbation_bank.py",
            "src/rlrmp/analysis/pipelines/gru_feedback_ablation.py",
        ],
        notes=[
            "Compatibility index for active GRU postrun diagnostics.",
            "The primary run contract is the Feedbax analysis bundle "
            "`rlrmp/gru_postrun` when bundle manifests are available.",
        ],
        repo_root=repo_root,
    )


def _run_input_refs(
    experiment: str,
    run_ids: Sequence[str],
    *,
    repo_root: Path,
) -> list[dict[str, Path | str]]:
    refs: list[dict[str, Path | str]] = []
    for run_id in run_ids:
        refs.append(
            {
                "role": "run_spec",
                "path": run_spec_path(experiment, run_id, repo_root=repo_root),
            }
        )
        refs.append(
            {
                "role": "run_artifact_dir",
                "path": repo_root / "_artifacts" / experiment / "runs" / run_id,
            }
        )
    return refs


def _regeneration_spec_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_regeneration_spec.json")


def feedback_checkpoint_selection_status(
    feedback_ablation: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the post-run manifest entry for feedback-selected checkpoints."""

    status = feedback_ablation.get("status", "unknown")
    payload = {
        "status": status,
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "selection_use": "audit_only_not_primary_checkpoint_selection",
        "source_sidecar": "feedback_ablation",
    }
    for key in ("json_path", "note_path"):
        if key in feedback_ablation:
            payload[key] = feedback_ablation[key]
    if status != "materialized":
        return payload | {
            "reason": feedback_ablation.get("reason", "feedback_ablation_not_materialized")
        }
    audit = feedback_ablation.get("feedback_checkpoint_selection_audit", {})
    if not isinstance(audit, Mapping):
        return payload | {"reason": "feedback_selection_audit_missing"}
    payload["status"] = audit.get("status", "not_available")
    payload["schema_version"] = audit.get("schema_version")
    payload["primary_checkpoint_policy"] = audit.get("primary_checkpoint_policy")
    if audit.get("reason") is not None:
        payload["reason"] = audit.get("reason")
    if audit.get("selected_candidate") is not None:
        payload["selected_candidate"] = audit.get("selected_candidate")
    return payload


def materialize_optional_perturbation_response(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    n_rollout_trials: int,
    bank_mode: str = "raw",
    calibration_level: str | Sequence[str] | None = None,
    calibration_reach: str | float | None = None,
    feedback_scale_manifest_path: Path | None = None,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return the registered evaluation-matrix request for perturbation responses."""

    from rlrmp.analysis.perturbation_bank import perturbation_bank_matrix_payload

    try:
        matrix = perturbation_bank_matrix_payload(
            source_experiment=experiment,
            run_ids=run_ids,
            labels=labels,
            n_rollout_trials=n_rollout_trials,
            bank_mode=bank_mode,
            calibration_level=calibration_level,
            calibration_reach=calibration_reach,
            feedback_scale_manifest_path=feedback_scale_manifest_path,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
        )
    except (FileNotFoundError, ValueError, KeyError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "perturbation_response_inputs_unavailable",
            "detail": str(exc),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }

    return {
        "status": "registered_evaluation_matrix",
        "custody": "feedbax_evaluation_manifests",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "evaluation_matrix": matrix,
        "next_action": "execute_evaluation_run_matrix",
    }


def materialize_optional_map_error_decomposition(
    *,
    experiment: str,
    run_ids: Sequence[str],
    use_validation_selected_checkpoints: bool,
    standard_manifest_path: Path,
    preferred_checkpoint_manifest_path: Path | None,
    output_path: Path,
    note_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Call the optional map-error decomposition sidecar where inputs are available."""

    try:
        module = importlib.import_module("rlrmp.analysis.pipelines.gru_map_error_decomposition")
        materializer = getattr(module, "materialize_gru_map_error_decomposition")
        writer = getattr(module, "write_map_error_decomposition_result")
    except (ImportError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "optional_map_decomposition_unavailable",
            "detail": str(exc),
            "expected_hook": (
                "rlrmp.analysis.pipelines.gru_map_error_decomposition."
                "materialize_gru_map_error_decomposition"
            ),
        }

    try:
        result = materializer(
            standard_manifest_path=standard_manifest_path,
            experiment=experiment,
            run_ids=tuple(run_ids),
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
            repo_root=repo_root,
        )
    except (FileNotFoundError, ValueError, KeyError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "map_decomposition_inputs_unavailable",
            "detail": str(exc),
            "json_path": _repo_relative(output_path, repo_root=repo_root),
            "note_path": _repo_relative(note_path, repo_root=repo_root),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }

    writer(result, json_path=output_path, markdown_path=note_path)
    return {
        "status": "materialized",
        "json_path": _repo_relative(output_path, repo_root=repo_root),
        "note_path": _repo_relative(note_path, repo_root=repo_root),
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "result": {
            "schema_version": result.get("format"),
            "n_rows": len(result.get("rows", ())),
            "checkpoint_policy": result.get("checkpoint_policy"),
        },
    }


def materialize_optional_objective_comparator(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    checkpoint_policy: str,
    use_validation_selected_checkpoints: bool,
    checkpoint_manifest: dict[str, Any] | None,
    checkpoint_manifest_path: Path | None,
    standard_manifest_path: Path,
    output_path: Path,
    note_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Call the optional objective-comparator sidecar when that module exists."""

    try:
        module = importlib.import_module("rlrmp.analysis.pipelines.objective_comparator")
        materializer = getattr(module, "materialize_gru_objective_comparator_sidecar")
    except (ImportError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "optional_comparator_unavailable",
            "detail": str(exc),
            "expected_hook": (
                "rlrmp.analysis.pipelines.objective_comparator."
                "materialize_gru_objective_comparator_sidecar"
            ),
        }

    result = materializer(
        experiment=experiment,
        run_ids=tuple(run_ids),
        labels=None if labels is None else tuple(labels),
        checkpoint_policy=checkpoint_policy,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_manifest_path=checkpoint_manifest_path,
        standard_manifest_path=standard_manifest_path,
        output_path=output_path,
        note_path=note_path,
        repo_root=repo_root,
    )
    result_status = result.get("status") if isinstance(result, dict) else None
    if result_status not in (None, "materialized"):
        return {
            "status": result_status,
            "json_path": _repo_relative(output_path, repo_root=repo_root),
            "note_path": _repo_relative(note_path, repo_root=repo_root),
            "result": result,
        }
    return {
        "status": "materialized",
        "json_path": _repo_relative(output_path, repo_root=repo_root),
        "note_path": _repo_relative(note_path, repo_root=repo_root),
        "result": result,
    }


def split_stress_objective_comparator_status(
    objective_comparator: dict[str, Any],
) -> dict[str, Any]:
    """Return the post-run manifest entry for the split stress-bank comparator."""

    status = objective_comparator.get("status", "unknown")
    payload = {
        "status": status,
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "source_sidecar": "objective_comparator",
    }
    for key in ("json_path", "note_path"):
        if key in objective_comparator:
            payload[key] = objective_comparator[key]
    if status != "materialized":
        result = objective_comparator.get("result", {})
        result_reason = result.get("reason") if isinstance(result, dict) else None
        return payload | {
            "reason": (
                objective_comparator.get("reason")
                or result_reason
                or "objective_comparator_not_materialized"
            )
        }
    result = objective_comparator.get("result", {})
    if isinstance(result, dict):
        split_status = result.get("standard_split_bank_comparator_status")
        if split_status is not None:
            payload["standard_split_bank_comparator_status"] = split_status
        schema_version = result.get("schema_version")
        if schema_version is not None:
            payload["schema_version"] = schema_version
    return payload


def checkpoint_policy_name(use_validation_selected_checkpoints: bool) -> str:
    """Return the manifest checkpoint-policy label for a GRU materialization."""

    return (
        VALIDATION_SELECTED_CHECKPOINT_POLICY
        if use_validation_selected_checkpoints
        else FINAL_CHECKPOINT_POLICY
    )


def _selection_leakage_guard_note(plan: GruPostrunMaterializationPlan) -> str:
    if plan.fixed_bank_rescore_manifest_path is not None:
        return (
            "This materialization explicitly loads the supplied fixed-bank checkpoint "
            "manifest. Certificate, I/O map, covariance-weighted map, "
            "map-decomposition, objective-comparator, split-stress-bank, "
            "perturbation-response, and feedback-ablation values remain audit "
            "sidecars and are not silently used to choose checkpoints."
        )
    return (
        "Checkpoint selection uses rollout validation objective only. Certificate, "
        "I/O map, covariance-weighted map, map-decomposition, objective-comparator, "
        "split-stress-bank, perturbation-response, and feedback-ablation values are "
        "audit sidecars and do not replace validation-selected checkpoint loading."
    )


def fixed_bank_rescore_manifest_status(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return postrun provenance for a fixed-bank rescore manifest path."""

    status: dict[str, Any] = {"path": _repo_relative(path, repo_root=repo_root)}
    if not path.exists():
        return status | {"status": "missing", "selection_use": "sparse_history_fallback"}
    manifest = load_materialized_fixed_bank_manifest(manifest_path=path)
    if manifest is None:
        return status | {"status": "failed", "selection_use": "sparse_history_fallback"}
    return status | {
        "status": "materialized",
        "schema_version": manifest.schema_version,
        "selection_use": "fixed_bank_rescore",
        "validation_bank": manifest.bank.metadata.get("validation_bank"),
    }


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_OUTPUT_TAG",
    "FINAL_CHECKPOINT_POLICY",
    "GruPostrunMaterializationPlan",
    "SCHEMA_VERSION",
    "VALIDATION_SELECTED_CHECKPOINT_POLICY",
    "checkpoint_policy_name",
    "fixed_bank_rescore_manifest_status",
    "feedback_checkpoint_selection_status",
    "materialize_gru_postrun_analysis",
    "materialize_optional_feedback_ablation",
    "materialize_optional_map_error_decomposition",
    "materialize_optional_objective_comparator",
    "materialize_optional_perturbation_response",
    "plan_gru_postrun_materialization",
    "split_stress_objective_comparator_status",
]

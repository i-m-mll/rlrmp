"""One-command post-run materialization for C&S GRU runs."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rlrmp.analysis.cs_gru_standard_materialization import (
    MATERIALIZER_ISSUE_ID,
    materialize_gru_standard_result,
    write_gru_standard_result,
)
from rlrmp.analysis.gru_checkpoint_selection import (
    fixed_bank_manifest_path,
    load_materialized_fixed_bank_manifest,
    materialize_validation_selected_checkpoint_manifest,
)
from rlrmp.analysis.gru_evaluation_diagnostics import (
    materialize_gru_evaluation_diagnostics,
)
from rlrmp.analysis.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    materialize_gru_pilot_figures,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


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
    postrun_manifest_path: Path

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
    fixed_bank_rescore_manifest_path = (
        fixed_bank_rescore_manifest_path
        if fixed_bank_rescore_manifest_path is not None
        else fixed_bank_manifest_path(experiment, repo_root=repo_root)
    )
    fixed_bank_available = False
    if use_validation_selected_checkpoints:
        fixed_bank_manifest = load_materialized_fixed_bank_manifest(
            experiment=experiment,
            repo_root=repo_root,
            manifest_path=fixed_bank_rescore_manifest_path,
        )
        fixed_bank_available = (
            fixed_bank_manifest is not None
            and all(run_id in fixed_bank_manifest.get("runs", {}) for run_id in run_ids)
        )
    return GruPostrunMaterializationPlan(
        experiment=experiment,
        run_ids=tuple(run_ids),
        output_tag=output_tag,
        checkpoint_policy=checkpoint_policy,
        checkpoint_selection_source=(
            "fixed_bank_rescore" if fixed_bank_available else checkpoint_policy
        ),
        notes_dir=notes_dir,
        checkpoint_manifest_path=(
            notes_dir / "validation_selected_checkpoints.json"
            if use_validation_selected_checkpoints
            else None
        ),
        fixed_bank_rescore_manifest_path=(
            fixed_bank_rescore_manifest_path if use_validation_selected_checkpoints else None
        ),
        standard_note_path=notes_dir / f"gru_standard_certificates_{output_tag}.md",
        standard_manifest_path=notes_dir
        / f"gru_standard_certificates_{output_tag}_manifest.json",
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
        postrun_manifest_path=notes_dir / f"gru_postrun_materialization_{output_tag}.json",
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
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize the standard post-run GRU analysis bundle.

    The default checkpoint policy is validation-selected per replicate. Standard
    certificate, I/O map, covariance-weighted map, and objective-comparator
    values remain audit sidecars and are never used for checkpoint selection.
    """

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

    checkpoint_manifest: dict[str, Any] | None = None
    if plan.checkpoint_manifest_path is not None:
        checkpoint_manifest = materialize_validation_selected_checkpoint_manifest(
            experiment=experiment,
            run_ids=run_ids,
            output_path=plan.checkpoint_manifest_path,
            preferred_manifest_path=plan.fixed_bank_rescore_manifest_path,
            repo_root=repo_root,
        )

    standard_result = materialize_gru_standard_result(
        run_ids=run_ids,
        experiment=experiment,
        materializer_issue_id=materializer_issue_id,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        repo_root=repo_root,
    )
    write_gru_standard_result(
        standard_result,
        note_path=plan.standard_note_path,
        manifest_path=plan.standard_manifest_path,
    )

    evaluation_manifest = materialize_gru_evaluation_diagnostics(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        output_path=plan.evaluation_manifest_path,
        bulk_dir=plan.evaluation_bulk_dir,
        n_rollout_trials=n_rollout_trials,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
        repo_root=repo_root,
    )

    figure_summary = materialize_gru_pilot_figures(
        experiment=experiment,
        run_ids=run_ids,
        labels=labels,
        output_dir=plan.figure_output_dir,
        n_rollout_trials=n_rollout_trials,
        include_reference=include_reference,
        use_validation_selected_checkpoints=use_validation_selected_checkpoints,
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
            output_path=plan.perturbation_response_json_path,
            note_path=plan.perturbation_response_note_path,
            bulk_dir=plan.perturbation_response_bulk_dir,
            repo_root=repo_root,
        )
        if include_perturbation_response
        else {"status": "skipped", "reason": "disabled_by_cli"}
    )

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
            ],
            "note": (
                "Checkpoint selection uses rollout validation objective only. "
                "Certificate, I/O map, covariance-weighted map, map-decomposition, "
                "objective-comparator, split-stress-bank, and perturbation-response "
                "values are audit sidecars."
            ),
        },
        "plan": plan.to_json(repo_root=repo_root),
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
            "evaluation_diagnostics_manifest": _repo_relative(
                plan.evaluation_manifest_path,
                repo_root=repo_root,
            ),
            "evaluation_bulk_dir": _repo_relative(plan.evaluation_bulk_dir, repo_root=repo_root),
            "figure_output_dir": _repo_relative(plan.figure_output_dir, repo_root=repo_root),
            "figure_summary": _repo_relative(
                plan.figure_output_dir / "figure_summary.json",
                repo_root=repo_root,
            ),
            "objective_comparator": objective_comparator,
            "split_stress_objective_comparator": split_stress_objective_comparator,
            "map_decomposition": map_decomposition,
            "perturbation_response": perturbation_response,
        },
        "summaries": {
            "standard_certificate": standard_result.get("summary", {}),
            "evaluation_diagnostics_schema": evaluation_manifest.get("schema_version"),
            "figure_summary_keys": sorted(figure_summary.keys()),
        },
    }
    plan.postrun_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_optional_perturbation_response(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    n_rollout_trials: int,
    output_path: Path,
    note_path: Path,
    bulk_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Call the optional perturbation-response bank materializer."""

    try:
        module = importlib.import_module("rlrmp.analysis.gru_perturbation_bank")
        materializer = getattr(module, "materialize_gru_perturbation_response")
    except (ImportError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "optional_perturbation_response_unavailable",
            "detail": str(exc),
            "expected_hook": (
                "rlrmp.analysis.gru_perturbation_bank."
                "materialize_gru_perturbation_response"
            ),
        }

    try:
        result = materializer(
            source_experiment=experiment,
            result_experiment=experiment,
            run_ids=tuple(run_ids),
            labels=None if labels is None else tuple(labels),
            n_rollout_trials=n_rollout_trials,
            evaluate=True,
            write_bulk_arrays=True,
            output_path=output_path,
            note_path=note_path,
            bulk_dir=bulk_dir,
            repo_root=repo_root,
        )
    except (FileNotFoundError, ValueError, KeyError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "perturbation_response_inputs_unavailable",
            "detail": str(exc),
            "json_path": _repo_relative(output_path, repo_root=repo_root),
            "note_path": _repo_relative(note_path, repo_root=repo_root),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }

    runs = result.get("runs", {}) if isinstance(result, dict) else {}
    bank = result.get("bank", {}) if isinstance(result, dict) else {}
    perturbations = bank.get("perturbations", ()) if isinstance(bank, dict) else ()
    return {
        "status": "materialized",
        "json_path": _repo_relative(output_path, repo_root=repo_root),
        "note_path": _repo_relative(note_path, repo_root=repo_root),
        "bulk_dir": _repo_relative(bulk_dir, repo_root=repo_root),
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "result": {
            "schema_version": result.get("schema_version") if isinstance(result, dict) else None,
            "n_runs": len(runs),
            "n_perturbations": len(perturbations),
            "checkpoint_policy": (
                result.get("checkpoint_policy") if isinstance(result, dict) else None
            ),
        },
    }


def materialize_optional_map_error_decomposition(
    *,
    experiment: str,
    run_ids: Sequence[str],
    use_validation_selected_checkpoints: bool,
    standard_manifest_path: Path,
    output_path: Path,
    note_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Call the optional map-error decomposition sidecar where inputs are available."""

    try:
        module = importlib.import_module("rlrmp.analysis.gru_map_error_decomposition")
        materializer = getattr(module, "materialize_gru_map_error_decomposition")
        writer = getattr(module, "write_map_error_decomposition_result")
    except (ImportError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "optional_map_decomposition_unavailable",
            "detail": str(exc),
            "expected_hook": (
                "rlrmp.analysis.gru_map_error_decomposition."
                "materialize_gru_map_error_decomposition"
            ),
        }

    try:
        result = materializer(
            standard_manifest_path=standard_manifest_path,
            experiment=experiment,
            run_ids=tuple(run_ids),
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
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
        module = importlib.import_module("rlrmp.analysis.objective_comparator")
        materializer = getattr(module, "materialize_gru_objective_comparator_sidecar")
    except (ImportError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "optional_comparator_unavailable",
            "detail": str(exc),
            "expected_hook": (
                "rlrmp.analysis.objective_comparator."
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


def fixed_bank_rescore_manifest_status(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return postrun provenance for a fixed-bank rescore manifest path."""

    status: dict[str, Any] = {"path": _repo_relative(path, repo_root=repo_root)}
    if not path.exists():
        return status | {"status": "missing", "selection_use": "sparse_history_fallback"}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    materialization_status = str(manifest.get("materialization_status", "unknown"))
    return status | {
        "status": materialization_status,
        "schema_version": manifest.get("schema_version"),
        "selection_use": (
            "fixed_bank_rescore"
            if materialization_status == "materialized"
            else "sparse_history_fallback"
        ),
        "validation_bank": manifest.get("validation_bank"),
        "not_materialized_reason": manifest.get("not_materialized_reason"),
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
    "materialize_gru_postrun_analysis",
    "materialize_optional_map_error_decomposition",
    "materialize_optional_objective_comparator",
    "materialize_optional_perturbation_response",
    "plan_gru_postrun_materialization",
    "split_stress_objective_comparator_status",
]

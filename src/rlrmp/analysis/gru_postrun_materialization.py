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
    notes_dir: Path
    checkpoint_manifest_path: Path | None
    standard_note_path: Path
    standard_manifest_path: Path
    evaluation_manifest_path: Path
    evaluation_bulk_dir: Path
    figure_output_dir: Path
    objective_comparator_json_path: Path
    objective_comparator_note_path: Path
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
    repo_root: Path = REPO_ROOT,
) -> GruPostrunMaterializationPlan:
    """Return the tracked and ignored output paths for a post-run materialization."""

    if not run_ids:
        raise ValueError("At least one run ID is required")
    checkpoint_policy = checkpoint_policy_name(use_validation_selected_checkpoints)
    notes_dir = repo_root / "results" / experiment / "notes"
    artifact_dir = repo_root / "_artifacts" / experiment
    return GruPostrunMaterializationPlan(
        experiment=experiment,
        run_ids=tuple(run_ids),
        output_tag=output_tag,
        checkpoint_policy=checkpoint_policy,
        notes_dir=notes_dir,
        checkpoint_manifest_path=(
            notes_dir / "validation_selected_checkpoints.json"
            if use_validation_selected_checkpoints
            else None
        ),
        standard_note_path=notes_dir / f"gru_standard_certificates_{output_tag}.md",
        standard_manifest_path=notes_dir
        / f"gru_standard_certificates_{output_tag}_manifest.json",
        evaluation_manifest_path=notes_dir / f"gru_evaluation_diagnostics_{output_tag}.json",
        evaluation_bulk_dir=artifact_dir / "evaluation_diagnostics" / f"gru_{output_tag}",
        figure_output_dir=artifact_dir / "figures" / f"gru_postrun_{output_tag}",
        objective_comparator_json_path=notes_dir / f"objective_comparator_{output_tag}.json",
        objective_comparator_note_path=notes_dir / f"objective_comparator_{output_tag}.md",
        postrun_manifest_path=notes_dir / f"gru_postrun_materialization_{output_tag}.json",
    )


def materialize_gru_postrun_analysis(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    output_tag: str = DEFAULT_OUTPUT_TAG,
    use_validation_selected_checkpoints: bool = True,
    include_reference: bool = True,
    n_rollout_trials: int = DEFAULT_N_ROLLOUT_TRIALS,
    materializer_issue_id: str = MATERIALIZER_ISSUE_ID,
    include_objective_comparator: bool = True,
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
        repo_root=repo_root,
    )
    mkdir_p(plan.notes_dir)

    checkpoint_manifest: dict[str, Any] | None = None
    if plan.checkpoint_manifest_path is not None:
        checkpoint_manifest = materialize_validation_selected_checkpoint_manifest(
            experiment=experiment,
            run_ids=run_ids,
            output_path=plan.checkpoint_manifest_path,
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

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": experiment,
        "run_ids": list(run_ids),
        "checkpoint_policy": plan.checkpoint_policy,
        "selection_leakage_guard": {
            "status": "audit_only",
            "audit_only_metrics": [
                "state_weighted_action_mismatch",
                "clean_action_mismatch",
                "observation_history_to_action_map_mismatch",
                "covariance_weighted_map_mismatch",
                "extlqg_objective_comparator",
            ],
            "note": (
                "Checkpoint selection uses rollout validation objective only. "
                "Certificate, I/O map, covariance-weighted map, and objective "
                "comparator values are audit sidecars."
            ),
        },
        "plan": plan.to_json(repo_root=repo_root),
        "outputs": {
            "checkpoint_manifest": (
                None
                if plan.checkpoint_manifest_path is None
                else _repo_relative(plan.checkpoint_manifest_path, repo_root=repo_root)
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
        module = importlib.import_module("rlrmp.analysis.gru_objective_comparator")
        materializer = getattr(module, "materialize_gru_objective_comparator_sidecar")
    except (ImportError, AttributeError) as exc:
        return {
            "status": "skipped",
            "reason": "optional_comparator_unavailable",
            "detail": str(exc),
            "expected_hook": (
                "rlrmp.analysis.gru_objective_comparator."
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
    return {
        "status": "materialized",
        "json_path": _repo_relative(output_path, repo_root=repo_root),
        "note_path": _repo_relative(note_path, repo_root=repo_root),
        "result": result,
    }


def checkpoint_policy_name(use_validation_selected_checkpoints: bool) -> str:
    """Return the manifest checkpoint-policy label for a GRU materialization."""

    return (
        VALIDATION_SELECTED_CHECKPOINT_POLICY
        if use_validation_selected_checkpoints
        else FINAL_CHECKPOINT_POLICY
    )


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
    "materialize_gru_postrun_analysis",
    "materialize_optional_objective_comparator",
    "plan_gru_postrun_materialization",
]

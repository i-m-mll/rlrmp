"""Typed perturbation-matrix science over manifest-canonical evaluation rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from feedbax.analysis import EvaluationRunMatrixSpec, materialize_evaluation_run_matrix
from feedbax.contracts import MaterializedMatrixRow

from rlrmp.paths import REPO_ROOT


DEFAULT_MATRIX_SPEC = (
    REPO_ROOT / "src" / "rlrmp" / "config" / "evaluation_matrices" / "cs_perturbation_bank.json"
)


def load_perturbation_bank_matrix(
    path: Path | str = DEFAULT_MATRIX_SPEC,
) -> EvaluationRunMatrixSpec:
    """Load the governed perturbation-bank evaluation conditions."""

    return EvaluationRunMatrixSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def materialize_perturbation_bank_rows(
    matrix: EvaluationRunMatrixSpec | Mapping[str, Any] | None = None,
    *,
    repo_root: Path | str = REPO_ROOT,
) -> list[MaterializedMatrixRow[Any]]:
    """Resolve typed bank conditions without executing evaluation rollouts."""

    resolved = load_perturbation_bank_matrix() if matrix is None else matrix
    return materialize_evaluation_run_matrix(resolved, repo_root=repo_root)


def perturbation_bank_matrix_payload(
    *,
    source_experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    n_rollout_trials: int,
    bank_mode: str,
    calibration_level: str | Sequence[str] | None,
    calibration_reach: str | float | None,
    feedback_scale_manifest_path: Path | None,
    preferred_checkpoint_manifest_path: Path | None,
) -> dict[str, Any]:
    """Return a caller-specialized matrix spec for registered evaluation execution."""

    matrix = load_perturbation_bank_matrix().model_dump(mode="json", exclude_none=True)
    params = matrix["base"]["params"]
    params.update(
        {
            "source_experiment": source_experiment,
            "run_ids": list(run_ids),
            "labels": None if labels is None else list(labels),
            "n_rollout_trials": int(n_rollout_trials),
            "bank_mode": bank_mode,
            "calibration_level": calibration_level,
            "calibration_reach": calibration_reach,
            "feedback_scale_manifest_path": (
                None if feedback_scale_manifest_path is None else str(feedback_scale_manifest_path)
            ),
            "preferred_checkpoint_manifest_path": (
                None
                if preferred_checkpoint_manifest_path is None
                else str(preferred_checkpoint_manifest_path)
            ),
            "checkpoint_selection_mode": (
                "fixed_bank_manifest"
                if preferred_checkpoint_manifest_path is not None
                else "sparse_history"
            ),
        }
    )
    matrix["base"]["training_run_ids"] = list(run_ids)
    matrix["base"]["inputs"] = [
        {"kind": "TrainingRunManifest", "id": run_id, "role": "training_run"} for run_id in run_ids
    ]
    return json.loads(json.dumps(matrix))


def archived_parity_projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Project archived row metrics onto the compact strangler parity contract."""

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    return {
        "row_count": len(rows),
        "evaluated_count": len(evaluated),
        "families": sorted({str(row["family"]) for row in rows}),
        "mean_delta_action_norm": float(
            np.mean([float(row["delta_action_norm"]) for row in evaluated])
        ),
        "mean_delta_position_max": float(
            np.mean([float(row["delta_position_max"]) for row in evaluated])
        ),
        "mean_delta_cost_total": float(
            np.mean([float(row["delta_cost_total"]) for row in evaluated])
        ),
    }

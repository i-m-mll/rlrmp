"""Materialize GRU observation-action map-error decomposition sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.analysis.cs_gru_standard_materialization import (
    cs_output_feedback_observation_action_map,
    evaluate_gru_clean_actions,
)
from rlrmp.analysis.gru_map_error_decomposition import (
    FORMAT_VERSION,
    decompose_gru_map_error,
    write_map_error_decomposition_result,
)
from rlrmp.paths import REPO_ROOT

ISSUE_ID = "ddf7f43"
SOURCE_ISSUE_ID = "aacb9ed"
DEFAULT_LABEL = "fixed_target_random_perturb_validation_selected"
DEFAULT_STANDARD_MANIFEST = (
    REPO_ROOT
    / "results"
    / SOURCE_ISSUE_ID
    / "notes"
    / f"gru_standard_certificates_{DEFAULT_LABEL}_manifest.json"
)


def materialize_gru_map_error_decomposition(
    *,
    standard_manifest_path: Path = DEFAULT_STANDARD_MANIFEST,
    experiment: str = SOURCE_ISSUE_ID,
    run_ids: tuple[str, ...] | None = None,
    use_validation_selected_checkpoints: bool = True,
    top_k: int = 5,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Recompute response maps and return compact decomposition rows."""

    manifest = _read_json(standard_manifest_path)
    selected_run_ids = run_ids or _source_run_ids_from_standard_manifest(manifest)
    reference_map, reference_metadata = cs_output_feedback_observation_action_map()
    rows = []
    for run_id in selected_run_ids:
        run_spec = _read_json(repo_root / "results" / experiment / "runs" / run_id / "run.json")
        _actions, candidate_map, evaluation_metadata = evaluate_gru_clean_actions(
            run_id,
            run_spec=run_spec,
            experiment=experiment,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            repo_root=repo_root,
        )
        covariance = evaluation_metadata.pop("_observation_history_covariance_array", None)
        covariance_metadata = evaluation_metadata.get("observation_history_covariance")
        reference_batch = np.broadcast_to(reference_map[None, :, :, :], candidate_map.shape)
        decomposition = decompose_gru_map_error(
            candidate_map=candidate_map,
            reference_map=reference_batch,
            observation_dim=int(reference_metadata["observation_dim"]),
            input_covariance=covariance,
            input_covariance_metadata=covariance_metadata,
            top_k=top_k,
        )
        rows.append(
            {
                "run_id": f"{run_id}__nominal_clean",
                "source_run_id": run_id,
                "standard_certificate_row": _find_standard_row(manifest, run_id),
                "reference_metadata": reference_metadata,
                "evaluation_metadata": evaluation_metadata,
                "decomposition": decomposition,
            }
        )
    return {
        "format": FORMAT_VERSION,
        "issue": ISSUE_ID,
        "source_issue": experiment,
        "source_standard_manifest": _repo_relative(standard_manifest_path, repo_root=repo_root),
        "checkpoint_policy": (
            "validation_selected_per_replicate"
            if use_validation_selected_checkpoints
            else "final_checkpoint"
        ),
        "rows": rows,
    }


def main() -> None:
    """Run the map-error decomposition materializer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=SOURCE_ISSUE_ID)
    parser.add_argument(
        "--standard-manifest",
        type=Path,
        default=DEFAULT_STANDARD_MANIFEST,
        help="Existing GRU standard-certificate manifest with source run IDs.",
    )
    parser.add_argument("--run-id", action="append", help="Source run ID. May repeat.")
    parser.add_argument(
        "--final-checkpoints",
        action="store_true",
        help="Use final checkpoints instead of validation-selected checkpoints.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / SOURCE_ISSUE_ID
            / "notes"
            / f"gru_map_error_decomposition_{DEFAULT_LABEL}.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / SOURCE_ISSUE_ID
            / "notes"
            / f"gru_map_error_decomposition_{DEFAULT_LABEL}.md"
        ),
    )
    args = parser.parse_args()

    result = materialize_gru_map_error_decomposition(
        standard_manifest_path=args.standard_manifest,
        experiment=args.experiment,
        run_ids=tuple(args.run_id) if args.run_id else None,
        use_validation_selected_checkpoints=not args.final_checkpoints,
        top_k=args.top_k,
    )
    write_map_error_decomposition_result(
        result,
        markdown_path=args.markdown_output,
        json_path=args.json_output,
    )
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")


def _source_run_ids_from_standard_manifest(manifest: dict[str, Any]) -> tuple[str, ...]:
    run_ids = []
    for row in manifest.get("rows", ()):
        source_run_id = row.get("spec", {}).get("parameters", {}).get("source_run_id")
        if source_run_id:
            run_ids.append(source_run_id)
    if not run_ids:
        raise ValueError("standard manifest does not contain spec.parameters.source_run_id rows")
    return tuple(run_ids)


def _find_standard_row(manifest: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    for row in manifest.get("rows", ()):
        if row.get("spec", {}).get("parameters", {}).get("source_run_id") == run_id:
            spec = row.get("spec", {})
            return {
                "run_id": spec.get("run_id"),
                "status": row.get("status"),
                "observation_history_to_action_map_mismatch": _component_summary(
                    row,
                    "observation_history_to_action_map_mismatch",
                ),
            }
    return None


def _component_summary(row: dict[str, Any], component_name: str) -> dict[str, Any] | None:
    for component in row.get("certificate_components", ()):
        if component.get("name") == component_name:
            return {
                "status": component.get("status"),
                "summary": component.get("summary"),
                "reason": component.get("reason"),
            }
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

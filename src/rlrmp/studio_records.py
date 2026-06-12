"""Materialize Feedbax Studio records from rlrmp run manifests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from feedbax.analysis.evaluation import EvaluationRecipeResult, register_evaluation_recipe
from feedbax.contracts.graph import (
    GraphMetadata,
    GraphSpec,
    StudioArtifactRef,
    StudioCollectionRef,
    StudioManifestRef,
    StudioTaskBindingSpec,
    StudioValidationState,
    StudioWorkspaceSpec,
    build_default_studio_workspace,
)
from feedbax.manifest import (
    ArtifactRef,
    ParentRef,
    SpecPayload,
    TrainingRunManifest,
    load_manifest,
    safe_manifest_key,
)
from feedbax.studio_execution import (
    StudioPipelineMaterializationRequest,
    StudioPipelineMaterializationResult,
    materialize_studio_pipeline,
)

from rlrmp.analysis.matrix import STANDARD_MATRIX_ANALYSIS_TYPE, register_standard_matrix_recipes

DEFAULT_MANIFEST_ROOT = Path("_artifacts/feedbax_runs")
DEFAULT_REQUESTED_OUTPUTS = ("summary_metrics",)
STUDIO_DEFAULT_EVALUATION_TYPE = "studio_default_eval"


@dataclass(frozen=True)
class StudioRecordsMaterializationResult:
    """rlrmp wrapper result around Feedbax Studio pipeline materialization."""

    workspace_path: Path
    workspace: StudioWorkspaceSpec
    feedbax_result: StudioPipelineMaterializationResult | None
    selected_manifest_paths: tuple[Path, ...]

    @property
    def manifest_paths(self) -> dict[str, str]:
        """Return Feedbax materialized manifest paths keyed by stage id."""
        if self.feedbax_result is None:
            return {}
        return dict(self.feedbax_result.manifest_paths)

    @property
    def stage_ids(self) -> list[str]:
        """Return Feedbax materialized stage ids."""
        if self.feedbax_result is None:
            return []
        return list(self.feedbax_result.stage_ids)


def discover_training_run_manifests(root: Path | str) -> list[Path]:
    """Return candidate TrainingRunManifest paths under a Feedbax manifest root."""
    root_path = Path(root)
    return sorted((root_path / "manifests" / "training_runs").glob("*.json"))


def load_completed_training_manifests(
    root: Path | str,
    *,
    manifest_paths: Sequence[Path | str] | None = None,
    run_set_id: str | None = None,
    run_ids: Sequence[str] = (),
) -> list[tuple[Path, TrainingRunManifest]]:
    """Load completed training manifests selected from a manifest root."""
    selected_paths = (
        [Path(path) for path in manifest_paths]
        if manifest_paths is not None
        else discover_training_run_manifests(root)
    )
    wanted_run_ids = set(run_ids)
    manifests: list[tuple[Path, TrainingRunManifest]] = []
    for path in selected_paths:
        manifest = load_manifest(path)
        if not isinstance(manifest, TrainingRunManifest):
            continue
        if manifest.status != "completed":
            continue
        if run_set_id is not None and manifest.run_set_id != run_set_id:
            continue
        if (
            wanted_run_ids
            and manifest.id not in wanted_run_ids
            and manifest.job_id not in wanted_run_ids
        ):
            continue
        manifests.append((path, manifest))
    if not manifests:
        raise FileNotFoundError(
            "No completed TrainingRunManifest records matched the requested selection"
        )
    return manifests


def register_rlrmp_studio_recipes(*, replace: bool = True) -> None:
    """Register lightweight recipes used by Feedbax's Studio materializer.

    Feedbax currently asks Studio pipeline materialization for a
    ``studio_default_eval`` evaluation. rlrmp supplies a manifest-summary recipe
    for that key, then reuses the existing standard-matrix analysis recipe for
    a browsable figure-producing analysis record.
    """
    register_evaluation_recipe(
        STUDIO_DEFAULT_EVALUATION_TYPE,
        _studio_default_eval_recipe,
        replace=replace,
    )
    register_standard_matrix_recipes(replace=replace)


def build_studio_workspace_from_training_manifests(
    manifests: Sequence[tuple[Path, TrainingRunManifest]],
    *,
    label: str | None = None,
    analysis_type: str = STANDARD_MATRIX_ANALYSIS_TYPE,
    requested_outputs: Sequence[str] = DEFAULT_REQUESTED_OUTPUTS,
) -> StudioWorkspaceSpec:
    """Build a Studio workspace seeded with completed rlrmp training manifests."""
    if not manifests:
        raise ValueError("At least one completed training manifest is required")

    graph = _graph_from_training_manifest(manifests[0][1])
    workspace = build_default_studio_workspace(
        label=label or _default_workspace_label(manifests),
        graph=graph,
    )
    train_stage = _stage_by_kind(workspace, "train")
    train_scenario = workspace.scenarios[train_stage.scenario_id or ""]
    _copy_training_scenario_payload(train_scenario, manifests[0][1])

    training_refs = [_training_manifest_ref(path, manifest) for path, manifest in manifests]
    graph_refs = [ref for _path, manifest in manifests for ref in _graph_manifest_refs(manifest)]
    artifact_refs = [
        _artifact_ref(artifact, manifest_id=manifest.id)
        for _path, manifest in manifests
        for artifact in manifest.artifacts
    ]

    train_stage.status = "completed"
    train_stage.validation = StudioValidationState(
        valid=True,
        metadata={
            "materialized_by": "rlrmp.studio_records",
            "input_training_runs": [manifest.id for _path, manifest in manifests],
        },
    )
    train_stage.manifest_refs = _dedupe_manifest_refs([*training_refs, *graph_refs])
    train_stage.artifact_refs = _dedupe_artifact_refs(artifact_refs)
    train_stage.output_collections = [
        StudioCollectionRef(
            id="collection:training-runs",
            kind="training_runs",
            label="Training runs",
            source_stage_id=train_stage.id,
            item_refs=training_refs,
            facets={
                "run_set_ids": sorted(
                    {
                        manifest.run_set_id
                        for _path, manifest in manifests
                        if manifest.run_set_id is not None
                    }
                ),
            },
        )
    ]
    _replace_stage(workspace, train_stage)

    analysis_stage = _stage_by_kind(workspace, "analysis")
    analysis_scenario = workspace.scenarios[analysis_stage.scenario_id or ""]
    analysis_scenario.analysis_spec = {
        "analysis_type": analysis_type,
        "requested_outputs": list(requested_outputs),
        "input_requirements": [],
        "source": "rlrmp_training_manifest_import",
    }
    workspace.scenarios[analysis_scenario.id] = analysis_scenario

    workspace.manifest_refs = _dedupe_manifest_refs([*training_refs, *graph_refs])
    workspace.artifact_refs = _dedupe_artifact_refs(artifact_refs)
    workspace.collections = list(train_stage.output_collections)
    workspace.metadata = {
        **workspace.metadata,
        "source": "rlrmp_training_manifest_import",
        "training_manifest_ids": [manifest.id for _path, manifest in manifests],
        "training_manifest_paths": [str(path) for path, _manifest in manifests],
    }
    return workspace


def materialize_studio_records(
    *,
    manifest_root: Path | str = DEFAULT_MANIFEST_ROOT,
    manifest_paths: Sequence[Path | str] | None = None,
    run_set_id: str | None = None,
    run_ids: Sequence[str] = (),
    job_id: str | None = None,
    workspace_label: str | None = None,
    analysis_type: str = STANDARD_MATRIX_ANALYSIS_TYPE,
    requested_outputs: Sequence[str] = DEFAULT_REQUESTED_OUTPUTS,
    stages: Sequence[str] = ("eval", "analysis", "report"),
    issues: Sequence[str] = ("10b38d7",),
    register_recipes: bool = True,
    dry_run: bool = False,
) -> StudioRecordsMaterializationResult:
    """Materialize Studio records from completed rlrmp Feedbax manifests."""
    root_path = Path(manifest_root)
    selected = load_completed_training_manifests(
        root_path,
        manifest_paths=manifest_paths,
        run_set_id=run_set_id,
        run_ids=run_ids,
    )
    workspace = build_studio_workspace_from_training_manifests(
        selected,
        label=workspace_label,
        analysis_type=analysis_type,
        requested_outputs=requested_outputs,
    )
    materialization_job_id = job_id or _default_job_id(selected)
    workspace_path = _workspace_path(root_path, materialization_job_id)

    feedbax_result: StudioPipelineMaterializationResult | None = None
    if not dry_run:
        if register_recipes:
            register_rlrmp_studio_recipes()
        feedbax_result = materialize_studio_pipeline(
            StudioPipelineMaterializationRequest(
                workspace=workspace,
                stages=list(stages),  # type: ignore[arg-type]
                job_id=materialization_job_id,
                root=str(root_path),
                issues=list(issues),
                metadata={
                    "source": "rlrmp.studio_records",
                    "training_manifest_ids": [manifest.id for _path, manifest in selected],
                },
            )
        )
        workspace = feedbax_result.workspace
        _write_workspace(workspace_path, workspace)

    return StudioRecordsMaterializationResult(
        workspace_path=workspace_path,
        workspace=workspace,
        feedbax_result=feedbax_result,
        selected_manifest_paths=tuple(path for path, _manifest in selected),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Studio record materialization."""
    parser = argparse.ArgumentParser(
        description="Materialize Feedbax Studio records from rlrmp training manifests."
    )
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--manifest", action="append", type=Path, dest="manifest_paths")
    parser.add_argument("--run-set-id")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--job-id")
    parser.add_argument("--workspace-label")
    parser.add_argument("--analysis-type", default=STANDARD_MATRIX_ANALYSIS_TYPE)
    parser.add_argument(
        "--requested-output",
        action="append",
        dest="requested_outputs",
        default=[],
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("eval", "analysis", "report"),
        dest="stages",
        default=[],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-register-recipes", action="store_true")
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args(argv)

    result = materialize_studio_records(
        manifest_root=args.manifest_root,
        manifest_paths=args.manifest_paths,
        run_set_id=args.run_set_id,
        run_ids=args.run_id,
        job_id=args.job_id,
        workspace_label=args.workspace_label,
        analysis_type=args.analysis_type,
        requested_outputs=args.requested_outputs or DEFAULT_REQUESTED_OUTPUTS,
        stages=args.stages or ("eval", "analysis", "report"),
        register_recipes=not args.no_register_recipes,
        dry_run=args.dry_run,
    )
    summary = {
        "workspace_path": str(result.workspace_path),
        "selected_manifest_paths": [str(path) for path in result.selected_manifest_paths],
        "stage_ids": result.stage_ids,
        "manifest_paths": result.manifest_paths,
        "dry_run": args.dry_run,
    }
    if args.output_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Studio workspace: {summary['workspace_path']}")
        print(f"Selected training manifests: {len(result.selected_manifest_paths)}")
        for stage_id, path in result.manifest_paths.items():
            print(f"{stage_id}: {path}")
    return 0


def _studio_default_eval_recipe(spec, _root: Path, _states_path: Path) -> EvaluationRecipeResult:
    cells = []
    for ref in spec.inputs:
        metrics = ref.metadata.get("summary_metrics", {})
        cells.append(
            {
                "run_id": ref.id,
                "label": ref.metadata.get("job_id") or ref.metadata.get("run_set_id") or ref.id,
                "display_name": (
                    ref.metadata.get("display_name") or ref.metadata.get("job_id") or ref.id
                ),
                "summary_metrics": metrics if isinstance(metrics, dict) else {},
                "manifest_uri": ref.uri,
                "artifact_roles": ref.metadata.get("artifact_roles", []),
            }
        )
    return EvaluationRecipeResult(
        states={"cells": cells},
        summary_metrics={"rlrmp_training_runs": len(cells)},
        metadata={"standard_matrix": True, "source": "rlrmp.studio_records"},
    )


def _graph_from_training_manifest(manifest: TrainingRunManifest) -> GraphSpec:
    graph_spec = manifest.graph_spec
    if isinstance(graph_spec, SpecPayload):
        return GraphSpec.model_validate(graph_spec.inline)
    return GraphSpec(
        metadata=GraphMetadata(
            name=manifest.job_id or manifest.id,
            created_at=manifest.created_at.isoformat(),
            updated_at=manifest.created_at.isoformat(),
            tags=["rlrmp", "manifest-import"],
        )
    )


def _copy_training_scenario_payload(scenario, manifest: TrainingRunManifest) -> None:
    if isinstance(manifest.training_spec, SpecPayload):
        scenario.training_spec = manifest.training_spec.inline
    if isinstance(manifest.task_spec, SpecPayload):
        scenario.task_spec = manifest.task_spec.inline
    if isinstance(manifest.task_binding_spec, SpecPayload):
        scenario.task_binding_spec = StudioTaskBindingSpec.model_validate(
            manifest.task_binding_spec.inline
        )


def _training_manifest_ref(path: Path, manifest: TrainingRunManifest) -> StudioManifestRef:
    return StudioManifestRef(
        kind="TrainingRunManifest",
        id=manifest.id,
        role="training_run",
        uri=str(path),
        metadata={
            "job_id": manifest.job_id,
            "run_set_id": manifest.run_set_id,
            "status": manifest.status,
            "summary_metrics": manifest.summary_metrics,
            "artifact_roles": [artifact.role for artifact in manifest.artifacts],
        },
    )


def _graph_manifest_refs(manifest: TrainingRunManifest) -> list[StudioManifestRef]:
    graph_spec = manifest.graph_spec
    if isinstance(graph_spec, ParentRef):
        return [
            StudioManifestRef(
                kind=graph_spec.kind,
                id=graph_spec.id,
                role=graph_spec.role or "model_graph",
                uri=graph_spec.uri,
                metadata=graph_spec.metadata,
            )
        ]
    if isinstance(graph_spec, SpecPayload):
        return [
            StudioManifestRef(
                kind=graph_spec.kind,
                id=graph_spec.sha256 or f"{manifest.id}:graph_spec",
                role="model_graph",
                uri=graph_spec.ref,
                metadata={
                    **graph_spec.metadata,
                    "training_run_manifest_id": manifest.id,
                    "inline": graph_spec.ref is None,
                },
            )
        ]
    return []


def _artifact_ref(artifact: ArtifactRef, *, manifest_id: str) -> StudioArtifactRef:
    return StudioArtifactRef(
        kind="TrainingRunArtifact",
        id=artifact.artifact_id or f"{manifest_id}:{artifact.role}:{artifact.logical_name}",
        role=artifact.role,
        uri=artifact.uri,
        media_type=artifact.media_type,
        metadata={
            **artifact.metadata,
            "training_run_manifest_id": manifest_id,
            "logical_name": artifact.logical_name,
            "sha256": artifact.sha256,
            "storage_backend": artifact.storage_backend,
        },
    )


def _stage_by_kind(workspace: StudioWorkspaceSpec, kind: str):
    return next(stage for stage in workspace.stages if stage.kind == kind)


def _replace_stage(workspace: StudioWorkspaceSpec, updated) -> None:
    workspace.stages = [updated if stage.id == updated.id else stage for stage in workspace.stages]


def _dedupe_manifest_refs(refs: Sequence[StudioManifestRef]) -> list[StudioManifestRef]:
    deduped: dict[tuple[str, str, str | None], StudioManifestRef] = {}
    for ref in refs:
        deduped[(ref.kind, ref.id, ref.role)] = ref
    return list(deduped.values())


def _dedupe_artifact_refs(refs: Sequence[StudioArtifactRef]) -> list[StudioArtifactRef]:
    deduped: dict[tuple[str, str, str | None], StudioArtifactRef] = {}
    for ref in refs:
        deduped[(ref.kind, ref.id, ref.role)] = ref
    return list(deduped.values())


def _default_workspace_label(manifests: Sequence[tuple[Path, TrainingRunManifest]]) -> str:
    run_set_ids = sorted(
        {manifest.run_set_id for _path, manifest in manifests if manifest.run_set_id is not None}
    )
    if len(run_set_ids) == 1:
        return f"rlrmp {run_set_ids[0]}"
    return f"rlrmp Studio import ({len(manifests)} runs)"


def _default_job_id(manifests: Sequence[tuple[Path, TrainingRunManifest]]) -> str:
    first = manifests[0][1].run_set_id or manifests[0][1].job_id or manifests[0][1].id
    return f"rlrmp-studio-{safe_manifest_key(first)}"


def _workspace_path(root: Path, job_id: str) -> Path:
    return root / "studio_workspaces" / f"{safe_manifest_key(job_id)}.json"


def _write_workspace(path: Path, workspace: StudioWorkspaceSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        workspace.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

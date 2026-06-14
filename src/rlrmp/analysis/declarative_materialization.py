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
from rlrmp.paths import REPO_ROOT


GRU_STANDARD_ANALYSIS_TYPE = "rlrmp.certificate.gru_standard"
GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE = "rlrmp.diagnostic.gru_evaluation"
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


def _empty_analysis_data() -> AnalysisInputData:
    return AnalysisInputData(
        models={},
        tasks={},
        states={},
        hps={},
        extras=TreeNamespace(),
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


def _read_json_payload(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


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
    "GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE",
    "GRU_STANDARD_ANALYSIS_TYPE",
    "gru_evaluation_diagnostics_spec",
    "gru_evaluation_diagnostics_recipe",
    "gru_standard_certificate_spec",
    "gru_standard_certificate_recipe",
    "register_certificate_analysis_recipes",
]

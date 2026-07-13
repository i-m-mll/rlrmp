"""Feedbax-backed training diagnostics summary bundle support."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.specs import AnalysisRecipeResult, ResolvedAnalysisInput
from feedbax.analysis.specs import register_analysis_recipe
from feedbax.contracts.manifest import ArtifactRef, TrainingRunManifest
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from feedbax.training import TrainingDiagnostics

from rlrmp.io import read_json, update_marked_section
from rlrmp.mappings import as_mapping as _mapping
from rlrmp.paths import REPO_ROOT

TRAINING_DIAGNOSTICS_ANALYSIS_TYPE = "rlrmp.training_diagnostics_summary"
TRAINING_DIAGNOSTICS_SCHEMA_VERSION = "rlrmp.training_diagnostics_summary.v1"
NATIVE_DIAGNOSTICS_SCHEMA_ID = "feedbax.manifest.training_diagnostics"
NATIVE_DIAGNOSTICS_SCHEMA_VERSION = "feedbax.manifest.training_diagnostics.v1"

DEFAULT_METRICS = (
    "train_loss__total",
    "validation_loss__total",
    "pgd_broad_epsilon_inner_objective_improvement",
    "pgd_broad_epsilon_inner_objective_best",
    "pgd_broad_epsilon_inner_objective_final_endpoint",
    "pgd_broad_epsilon_inner_objective_final_endpoint_gap",
    "pgd_broad_epsilon_epsilon_norm_radius_ratio_mean",
    "pgd_broad_epsilon_boundary_fraction",
)


class TrainingDiagnosticsSummaryAnalysis(AbstractAnalysis):
    """Emit compact JSON and Markdown summaries for training diagnostics sidecars."""

    def compute(self, data: AnalysisInputData, **_kwargs):
        return {
            "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
            "summaries": list(data.states["summaries"]),
            "params": dict(data.extras.get("params", {})),
        }

    def emit_artifacts(
        self,
        context: AnalysisRunContext,
        data: AnalysisInputData,
        *,
        result,
        **_kwargs,
    ):
        params = dict(data.extras.get("params", {}))
        json_path, markdown_path = _output_paths(params)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        json_path.write_text(_json_dumps(result), encoding="utf-8")
        marker = str(params.get("note_marker", "training_diagnostics_summary"))
        update_marked_section(markdown_path, marker, render_markdown(result["summaries"]))

        json_ref = context.record_artifact(
            json_path,
            role="training_diagnostics_summary_json",
            logical_name=f"training_diagnostics/{json_path.name}",
            media_type="application/json",
            metadata=_artifact_metadata(params),
        )
        markdown_ref = context.record_artifact(
            markdown_path,
            role="analysis_notes",
            logical_name=f"training_diagnostics/{markdown_path.name}",
            media_type="text/markdown",
            metadata={"marker": marker, **_artifact_metadata(params)},
        )
        return {
            **result,
            "artifact_refs": {
                "json": json_ref,
                "markdown": markdown_ref,
            },
        }


def register_training_diagnostics_recipes(*, replace: bool = True) -> None:
    """Register rlrmp's training diagnostics analysis recipe."""
    register_analysis_recipe(
        TRAINING_DIAGNOSTICS_ANALYSIS_TYPE,
        training_diagnostics_recipe,
        replace=replace,
    )


def training_diagnostics_recipe(
    spec,
    root: Path,
    inputs: Sequence[ResolvedAnalysisInput],
) -> AnalysisRecipeResult:
    """Build a manifest-backed training diagnostics summary analysis."""
    params = dict(spec.params)
    summaries = [
        _summary_for_input(resolved, root=root)
        for resolved in inputs
    ]
    analysis = TrainingDiagnosticsSummaryAnalysis(
        variant="training_diagnostics_summary",
        cache_result=True,
    )
    return AnalysisRecipeResult(
        analyses={"summary": analysis},
        data=AnalysisInputData(
            models={},
            tasks={},
            states={"summaries": summaries},
            hps={"training_diagnostics": TreeNamespace(task=TreeNamespace(eval_n=len(summaries)))},
            extras={"params": params},
        ),
    )


def summarize_output_dir(output_dir: Path) -> dict[str, Any]:
    """Summarize live training diagnostics from one training output directory."""
    output_dir = output_dir.expanduser()
    diagnostics_path = output_dir / "training_diagnostics.npz"
    manifest_path = output_dir / "training_diagnostics.json"
    summary_path = output_dir / "training_summary.json"
    return summarize_training_diagnostics(
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        output_dir=output_dir,
    )


def summarize_training_diagnostics(
    *,
    diagnostics_path: Path,
    manifest_path: Path | None = None,
    summary_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Summarize one ``training_diagnostics.npz`` sidecar and companion JSON files."""
    diagnostics_path = diagnostics_path.expanduser()
    output_dir = (output_dir or diagnostics_path.parent).expanduser()
    manifest_path = manifest_path.expanduser() if manifest_path is not None else None
    summary_path = summary_path.expanduser() if summary_path is not None else None
    result: dict[str, Any] = {
        "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "exists": output_dir.exists(),
        "diagnostics_npz": str(diagnostics_path),
        "ok": True,
        "alerts": [],
        "latest": {},
    }
    manifest = (
        _read_legacy_companion(manifest_path, label="training diagnostics manifest")
        if manifest_path is not None and manifest_path.exists()
        else None
    )
    training_summary = (
        _read_legacy_companion(summary_path, label="training summary")
        if summary_path is not None and summary_path.exists()
        else None
    )
    if manifest is not None:
        result["manifest_completed_batches"] = manifest.get("completed_batches")
    if training_summary is not None:
        result["summary_completed_batches"] = training_summary.get("completed_batches")
        result["stopped_early_for_checkpoint_gate"] = training_summary.get(
            "stopped_early_for_checkpoint_gate"
        )
    if not diagnostics_path.exists():
        result["ok"] = False
        result["alerts"].append("missing training_diagnostics.npz")
        return result

    with np.load(diagnostics_path) as diagnostics:
        keys = set(diagnostics.files)
        pgd_sampled = (
            np.asarray(diagnostics["pgd_broad_epsilon_diagnostic_sampled"], dtype=bool)
            if "pgd_broad_epsilon_diagnostic_sampled" in keys
            else None
        )
        if "batch_index" in keys and diagnostics["batch_index"].size:
            result["latest_batch_index"] = _json_scalar(diagnostics["batch_index"][-1])
            result["n_logged_batches"] = int(diagnostics["batch_index"].shape[0])
        else:
            result["ok"] = False
            result["alerts"].append("missing or empty batch_index")
        for metric in DEFAULT_METRICS:
            if metric in keys and diagnostics[metric].size:
                result["latest"][metric] = _latest_stats(diagnostics[metric])
        for name in sorted(keys):
            array = diagnostics[name]
            checked = _finite_check_view(name, array, pgd_sampled)
            if (
                checked.size
                and np.issubdtype(checked.dtype, np.number)
                and not np.isfinite(checked).all()
            ):
                result["ok"] = False
                result["alerts"].append(f"nonfinite values in {name}")

    _add_metric_alerts(result)
    return result


def _read_legacy_companion(path: Path, *, label: str) -> Mapping[str, Any]:
    """Read one explicitly role-dispatched legacy JSON companion."""
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Legacy {label} must contain a JSON object: {path}")
    completed_batches = payload.get("completed_batches")
    if completed_batches is not None and (
        isinstance(completed_batches, bool)
        or not isinstance(completed_batches, int)
        or completed_batches < 0
    ):
        raise ValueError(
            f"Legacy {label} completed_batches must be a non-negative integer: {path}"
        )
    stopped = payload.get("stopped_early_for_checkpoint_gate")
    if stopped is not None and not isinstance(stopped, bool):
        raise ValueError(
            f"Legacy {label} stopped_early_for_checkpoint_gate must be boolean: {path}"
        )
    return payload


def render_text(summary: dict[str, Any]) -> str:
    """Render one diagnostics summary as compact plain text."""
    lines = [
        f"{summary['output_dir']}",
        f"  ok: {summary['ok']}",
        f"  latest_batch_index: {summary.get('latest_batch_index', 'n/a')}",
    ]
    for alert in summary.get("alerts", []):
        lines.append(f"  alert: {alert}")
    for name, stats in summary.get("latest", {}).items():
        if "value" in stats:
            value = stats["value"]
            lines.append(f"  {name}: {value}")
        else:
            lines.append(
                f"  {name}: mean={stats['mean']:.6g} "
                f"min={stats['min']:.6g} max={stats['max']:.6g} "
                f"finite={stats['finite']}"
            )
    return "\n".join(lines)


def render_markdown(summaries: Sequence[Mapping[str, Any]]) -> str:
    """Render diagnostics summaries as a marked-section Markdown table."""
    lines = [
        "## Training Diagnostics Summary",
        "",
        "| Run | OK | Latest batch | Alerts | Train loss | Validation loss |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for summary in summaries:
        latest = _mapping(summary.get("latest", {}))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary.get("run_id") or Path(str(summary["output_dir"])).name),
                    "yes" if summary.get("ok") else "no",
                    _format_optional(summary.get("latest_batch_index")),
                    "; ".join(str(alert) for alert in summary.get("alerts", [])) or "",
                    _format_latest_metric(latest, "train_loss__total"),
                    _format_latest_metric(latest, "validation_loss__total"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _summary_for_input(
    resolved: ResolvedAnalysisInput,
    *,
    root: Path,
) -> dict[str, Any]:
    manifest = resolved.manifest
    if not isinstance(manifest, TrainingRunManifest):
        raise TypeError(
            f"Expected TrainingRunManifest {resolved.ref.id!r}, got "
            f"{type(manifest).__name__}"
        )

    native_artifact = _unique_artifact(manifest, roles=("training_diagnostics",))
    if native_artifact is not None:
        summary = _summarize_native_diagnostics(
            manifest=manifest,
            manifest_path=resolved.path,
            artifact=native_artifact,
            root=root,
        )
        return summary

    diagnostics_artifact = _unique_artifact(
        manifest,
        roles=("training_diagnostics_npz",),
    )
    if diagnostics_artifact is None:
        raise ValueError(
            f"Training manifest {resolved.ref.id!r} has no typed training diagnostics "
            "artifact or explicit legacy training_diagnostics_npz artifact"
        )
    diagnostics_path = _artifact_path(diagnostics_artifact, root=root)
    summary_path = _companion_path(
        resolved,
        root=root,
        roles=("training_summary", "training_summary_json"),
        default=None,
    )
    manifest_path = _companion_path(
        resolved,
        root=root,
        roles=("training_diagnostics_manifest", "training_diagnostics_json"),
        default=None,
    )
    summary = summarize_training_diagnostics(
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        output_dir=diagnostics_path.parent,
    )
    summary["source_format"] = "legacy_npz"
    summary["run_id"] = resolved.ref.id
    summary["job_id"] = manifest.job_id
    summary["training_manifest_id"] = manifest.id
    summary["training_manifest_status"] = manifest.status
    summary["training_manifest_path"] = str(resolved.path) if resolved.path else None
    summary["diagnostics_artifact"] = _artifact_identity(diagnostics_artifact)
    return summary


def _summarize_native_diagnostics(
    *,
    manifest: TrainingRunManifest,
    manifest_path: Path | None,
    artifact: ArtifactRef,
    root: Path,
) -> dict[str, Any]:
    """Validate and summarize a native Feedbax ``TrainingDiagnostics`` artifact."""
    schema_id = artifact.metadata.get("schema_id")
    schema_version = artifact.metadata.get("schema_version")
    if artifact.media_type != "application/json":
        raise ValueError(
            f"Native training diagnostics artifact for {manifest.id!r} has unsupported "
            f"media_type {artifact.media_type!r}; expected 'application/json'"
        )
    if schema_id != NATIVE_DIAGNOSTICS_SCHEMA_ID:
        raise ValueError(
            f"Native training diagnostics artifact for {manifest.id!r} has unsupported "
            f"schema_id {schema_id!r}; expected {NATIVE_DIAGNOSTICS_SCHEMA_ID!r}"
        )
    if schema_version != NATIVE_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError(
            f"Native training diagnostics artifact for {manifest.id!r} has unsupported "
            f"schema_version {schema_version!r}; expected "
            f"{NATIVE_DIAGNOSTICS_SCHEMA_VERSION!r}"
        )

    diagnostics_path = _artifact_path(artifact, root=root)
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"Native training diagnostics artifact does not exist: {diagnostics_path}"
        )
    payload = diagnostics_path.read_bytes()
    if artifact.size_bytes is None:
        raise ValueError(
            f"Native training diagnostics artifact for {manifest.id!r} is missing "
            "required size_bytes"
        )
    if artifact.size_bytes != len(payload):
        raise ValueError(
            f"Native training diagnostics size mismatch for {manifest.id!r}: "
            f"artifact={artifact.size_bytes}, actual={len(payload)}"
        )
    payload_sha256 = sha256(payload).hexdigest()
    if artifact.sha256 is None:
        raise ValueError(
            f"Native training diagnostics artifact for {manifest.id!r} is missing "
            "required sha256"
        )
    if artifact.sha256 != payload_sha256:
        raise ValueError(
            f"Native training diagnostics sha256 mismatch for {manifest.id!r}: "
            f"artifact={artifact.sha256!r}, actual={payload_sha256!r}"
        )
    diagnostics = TrainingDiagnostics.model_validate_json(payload)
    _validate_native_diagnostics_binding(manifest, diagnostics)

    return {
        "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        "source_format": "feedbax_training_diagnostics",
        "output_dir": str(diagnostics_path.parent),
        "exists": True,
        "ok": True,
        "alerts": [],
        "latest": {},
        "latest_batch_index": (
            diagnostics.completed_batches - 1 if diagnostics.completed_batches > 0 else None
        ),
        "training_manifest_id": manifest.id,
        "job_id": manifest.job_id,
        "run_id": diagnostics.run_id,
        "training_manifest_status": manifest.status,
        "training_manifest_path": str(manifest_path) if manifest_path else None,
        "diagnostics_artifact": _artifact_identity(artifact),
        "training_diagnostics": diagnostics.model_dump(mode="json", exclude_none=True),
    }


def _validate_native_diagnostics_binding(
    manifest: TrainingRunManifest,
    diagnostics: TrainingDiagnostics,
) -> None:
    if diagnostics.manifest_id != manifest.id:
        raise ValueError(
            "Native training diagnostics manifest_id does not match its parent manifest: "
            f"diagnostics={diagnostics.manifest_id!r}, manifest={manifest.id!r}"
        )
    if manifest.job_id is None or diagnostics.run_id != manifest.job_id:
        raise ValueError(
            "Native training diagnostics run_id does not match parent manifest job_id: "
            f"diagnostics={diagnostics.run_id!r}, manifest={manifest.job_id!r}"
        )
    if diagnostics.terminal_status != manifest.status:
        raise ValueError(
            "Native training diagnostics terminal_status does not match parent manifest "
            f"status: diagnostics={diagnostics.terminal_status!r}, "
            f"manifest={manifest.status!r}"
        )
    if manifest.completed_batches is None:
        raise ValueError(
            "Native training diagnostics require parent manifest completed_batches"
        )
    if diagnostics.completed_batches != manifest.completed_batches:
        raise ValueError(
            "Native training diagnostics completed_batches does not match parent manifest: "
            f"diagnostics={diagnostics.completed_batches}, "
            f"manifest={manifest.completed_batches}"
        )


def _artifact_identity(artifact: ArtifactRef) -> dict[str, Any]:
    return artifact.model_dump(mode="json", exclude_none=True)


def _companion_path(
    resolved: ResolvedAnalysisInput,
    *,
    root: Path,
    roles: Sequence[str],
    default: Path | None,
) -> Path | None:
    artifact = _unique_artifact(resolved.manifest, roles=roles)
    if artifact is None:
        return default
    return _artifact_path(artifact, root=root)


def _unique_artifact(manifest: Any, *, roles: Sequence[str]) -> ArtifactRef | None:
    artifacts = getattr(manifest, "artifacts", []) if manifest is not None else []
    role_set = set(roles)
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, ArtifactRef) and artifact.role in role_set
    ]
    if len(matches) > 1:
        raise ValueError(
            "Training manifest has ambiguous artifacts for roles "
            f"{sorted(role_set)!r}: found {len(matches)} matches"
        )
    return matches[0] if matches else None


def _artifact_path(artifact: ArtifactRef, *, root: Path) -> Path:
    for candidate in (artifact.uri, artifact.logical_name):
        if candidate:
            return _resolve_repo_path(candidate, root=root)
    raise ValueError(f"Artifact {artifact.role!r} has no path-like URI or logical_name")


def _resolve_repo_path(value: Any, *, root: Path | None = None) -> Path:
    text = str(value)
    if text.startswith("repo://"):
        return REPO_ROOT / text.removeprefix("repo://")
    if text.startswith("file://"):
        return Path(text.removeprefix("file://"))
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    if root is not None and (root / path).exists():
        return root / path
    return REPO_ROOT / path


def _output_paths(params: Mapping[str, Any]) -> tuple[Path, Path]:
    experiment = _experiment(params)
    topic = str(params.get("topic", "training_diagnostics_summary"))
    notes_dir = REPO_ROOT / "results" / experiment / "notes"
    return notes_dir / f"{topic}.json", notes_dir / f"{topic}.md"


def _experiment(params: Mapping[str, Any]) -> str:
    if params.get("experiment"):
        return str(params["experiment"])
    figure_routing = params.get("figure_routing")
    if isinstance(figure_routing, Mapping) and figure_routing.get("experiment"):
        return str(figure_routing["experiment"])
    return "training_diagnostics"


def _artifact_metadata(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        "experiment": _experiment(params),
        "topic": str(params.get("topic", "training_diagnostics_summary")),
    }


def _finite_check_view(
    name: str,
    array: np.ndarray,
    pgd_sampled: np.ndarray | None,
) -> np.ndarray:
    if (
        pgd_sampled is not None
        and name.startswith("pgd_broad_epsilon_")
        and name != "pgd_broad_epsilon_diagnostic_sampled"
        and array.shape[:1] == pgd_sampled.shape[:1]
    ):
        return array[pgd_sampled]
    return array


def _latest_stats(array: np.ndarray) -> dict[str, Any]:
    latest = np.asarray(array[-1])
    if latest.dtype == np.bool_:
        return {"value": bool(np.all(latest))}
    if latest.ndim == 0:
        return {"value": _json_scalar(latest)}
    finite = np.isfinite(latest)
    return {
        "shape": list(latest.shape),
        "mean": _json_scalar(np.nanmean(latest)),
        "min": _json_scalar(np.nanmin(latest)),
        "max": _json_scalar(np.nanmax(latest)),
        "finite": bool(finite.all()),
    }


def _add_metric_alerts(result: dict[str, Any]) -> None:
    latest = result.get("latest", {})
    ratio = latest.get("pgd_broad_epsilon_epsilon_norm_radius_ratio_mean", {})
    ratio_max = ratio.get("max", ratio.get("value"))
    if ratio_max is not None and ratio_max > 1.0001:
        result["ok"] = False
        result["alerts"].append(f"PGD epsilon radius ratio exceeds 1: {ratio_max:.6g}")
    gap = latest.get("pgd_broad_epsilon_inner_objective_final_endpoint_gap", {})
    gap_min = gap.get("min", gap.get("value"))
    if gap_min is not None and gap_min < -1e-6:
        result["ok"] = False
        result["alerts"].append(f"PGD best/final gap is negative: {gap_min:.6g}")


def _json_scalar(value: Any) -> float | int | bool:
    scalar = np.asarray(value).item()
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    return scalar


def _format_latest_metric(latest: Mapping[str, Any], name: str) -> str:
    stats = _mapping(latest.get(name))
    if "value" in stats:
        return _format_optional(stats["value"])
    if "mean" in stats:
        return _format_optional(stats["mean"])
    return ""


def _format_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"

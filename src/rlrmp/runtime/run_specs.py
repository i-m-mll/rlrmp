"""Run-spec validation helpers for tracked RLRMP training recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feedbax.contracts.expressions import (
    AllOf,
    AnyOf,
    Compare,
    ContextItem,
    ExpressionContext,
    evaluate_expr,
)
from feedbax.contracts.manifest import TrainingRunManifest, load_manifest

from rlrmp.paths import REPO_ROOT, flat_run_spec_path, portable_repo_path, run_spec_path
from rlrmp.runtime.spec_migrations import (
    RUN_SPEC_KIND,
    accept_rlrmp_spec_payload,
    ensure_rlrmp_spec_families,
)
from rlrmp.train.science_vocabulary import ScienceMode
from rlrmp.train.minimax_native import (
    validate_minimax_run_spec,
    validate_minimax_run_spec_file,
)


NOMINAL_GRU_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "game_card",
        "task_timing",
        "model_summary",
        "training_summary",
        "loss_summary",
        "provenance",
        "feedbax_graph",
    }
)
NOMINAL_GRU_LOSS_OBJECTIVES = frozenset(
    {
        "partial_feedbax_terms",
        "partial_net_output_force_filter",
        "full_analytical_qrf",
    }
)
NOMINAL_GRU_SCIENCE_MODES = frozenset(mode.value for mode in ScienceMode)
NOMINAL_GRU_REQUIRED_PROVENANCE_KEYS = frozenset(
    {
        "git",
        "dependencies",
        "modal",
        "gpu",
    }
)
FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS = frozenset(
    {
        "graph_spec_path",
        "manifest_path",
    }
)
CS_LSS_PLANT_BACKEND = "cs_lss"
CS_LSS_REQUIRED_MECHANICS_TYPE = "LinearStateSpace"
# CS-LSS feedback component types accepted by run-spec validation. The clean
# target is the Feedbax-native ``StateFeedbackSelector``; the branded
# ``RLRMPCsLss*`` feedback IDs are accepted ONLY so archived CS-LSS run specs
# still validate. They are ComponentMigration source_types (see
# src/rlrmp/model/cs_lss_gru.py) and are pinned as retired IDs by the 7811e47
# confinement scan (ci/retired-component-id-confinement.toml) and the 9728133
# ratchet — do not add new branded feedback IDs here for active runs.
CS_LSS_FEEDBACK_COMPONENT_TYPES = frozenset(
    {
        "StateFeedbackSelector",
        "RLRMPCsLssDelayedPositionVelocityFeedback",
        "RLRMPCsLssTargetRelativeDelayedFeedback",
        "RLRMPCsLssTargetRelativeDelayedProprioceptiveFeedback",
    }
)

_CS_LSS_PLANT_BACKEND_EXPR = AnyOf(
    exprs=[
        AllOf(
            exprs=[
                Compare(item="run_spec", path=path, op="exists"),
                Compare(item="run_spec", path=path, op="eq", value=CS_LSS_PLANT_BACKEND),
            ]
        )
        for path in (
            "model_summary.plant_backend",
            "training_summary.plant_backend",
            "fidelity_status.plant_backend",
            "hps.model.plant_backend",
        )
    ]
    + [
        AllOf(
            exprs=[
                Compare(
                    item="run_spec",
                    path="model_summary.exact_cs_linear_state_space",
                    op="exists",
                ),
                Compare(
                    item="run_spec",
                    path="model_summary.exact_cs_linear_state_space",
                    op="eq",
                    value=True,
                ),
            ]
        )
    ]
)
# Legacy point-mass graph node types accepted by run-spec validation. The clean
# targets are the Feedbax-native ``PointMass``/``FirstOrderFilter`` (and native
# ``FeedbackChannels``); the branded ``RLRMPPointMass``/``RLRMPFeedbackChannels``
# entries are accepted ONLY for archived-artifact validation and are retired IDs
# under the 7811e47 confinement scan and the 9728133 ratchet.
LEGACY_POINT_MASS_GRAPH_TYPES = frozenset(
    {
        "FirstOrderFilter",
        "PointMass",
        "RLRMPFeedbackChannels",
        "RLRMPPointMass",
    }
)


class RunSpecValidationError(ValueError):
    """Raised when a tracked run spec is missing required metadata."""


def resolve_run_record(
    exp: str,
    run: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Resolve a governed RLRMP run record from its canonical TrainingRunManifest.

    New-format runs are authoritative through
    ``TrainingRunManifest.training_spec`` with ``SpecPayload(kind="RLRMPRunSpec")``.
    The tracked ``results/<exp>/runs/<run>.json`` file remains a convenient
    recipe/ref target, but this resolver does not read it for run details.
    """

    ensure_rlrmp_spec_families()
    manifest_path, manifest = _resolve_training_manifest(exp, run, repo_root=repo_root)
    training_spec = manifest.training_spec
    if training_spec is None:
        raise RunSpecValidationError(
            f"TrainingRunManifest has no training_spec for {exp}/{run}: {manifest_path}"
        )
    if training_spec.kind != RUN_SPEC_KIND:
        raise RunSpecValidationError(
            f"TrainingRunManifest training_spec kind must be {RUN_SPEC_KIND!r}; "
            f"found {training_spec.kind!r} in {manifest_path}"
        )
    result = accept_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        training_spec.inline,
        source_version=training_spec.schema_version,
        path=f"{manifest_path}:training_spec.inline",
    )
    payload = dict(result.payload)
    validate_nominal_gru_run_spec(
        payload,
        spec_dir=flat_run_spec_path(exp, run, repo_root=repo_root).with_suffix(""),
        require_graph_sidecars=False,
    )
    return payload


def validate_nominal_gru_run_spec(
    run_spec: dict[str, Any],
    *,
    spec_dir: Path,
    require_graph_sidecars: bool = True,
) -> None:
    """Validate the C&S-fidelity GRU run metadata contract.

    Args:
        run_spec: Decoded ``run.json`` payload.
        spec_dir: Directory containing the ``run.json`` file and graph sidecars.
        require_graph_sidecars: When false, validate pointer metadata without
            requiring adjacent sidecar files. This is used only for replaying a
            historical flat run spec into a fresh output/spec directory.

    Raises:
        RunSpecValidationError: If the run spec is missing top-level metadata,
            provenance groups, graph pointers, or adjacent graph sidecar files.
    """

    missing_top_level = _missing_keys(run_spec, NOMINAL_GRU_REQUIRED_TOP_LEVEL_KEYS)
    if missing_top_level:
        raise RunSpecValidationError(
            "nominal GRU run spec is missing required top-level metadata keys: "
            + ", ".join(missing_top_level)
        )

    model_summary = _mapping(run_spec, "model_summary")
    controller_kind = model_summary.get("controller_kind")
    if controller_kind != "gru":
        raise RunSpecValidationError(
            f"nominal GRU run spec must declare model_summary.controller_kind='gru'; "
            f"found {controller_kind!r}"
        )

    training_summary = _mapping(run_spec, "training_summary")
    training_mode = training_summary.get("training_mode")
    training_modes = str(training_mode).split("+") if training_mode is not None else []
    if not training_modes or any(mode not in NOMINAL_GRU_SCIENCE_MODES for mode in training_modes):
        raise RunSpecValidationError(
            "nominal GRU run spec must declare training_summary.training_mode as one of "
            f"{sorted(NOMINAL_GRU_SCIENCE_MODES)} or a '+'-joined composite; "
            f"found {training_mode!r}"
        )

    loss_objective = run_spec.get("loss_objective")
    if loss_objective not in NOMINAL_GRU_LOSS_OBJECTIVES:
        raise RunSpecValidationError(
            "nominal GRU run spec must declare loss_objective as one of "
            f"{sorted(NOMINAL_GRU_LOSS_OBJECTIVES)}; found {loss_objective!r}"
        )
    loss_summary = _mapping(run_spec, "loss_summary")
    loss_profile = loss_summary.get("objective_profile")
    if loss_profile != loss_objective:
        raise RunSpecValidationError(
            "nominal GRU run spec loss_summary.objective_profile must match "
            f"loss_objective; found {loss_profile!r} vs {loss_objective!r}"
        )

    missing_provenance = _missing_keys(
        _mapping(run_spec, "provenance"),
        NOMINAL_GRU_REQUIRED_PROVENANCE_KEYS,
    )
    if missing_provenance:
        raise RunSpecValidationError(
            "nominal GRU run spec is missing required provenance groups: "
            + ", ".join(missing_provenance)
        )

    graph_metadata = _mapping(run_spec, "feedbax_graph")
    missing_graph_pointers = _missing_keys(
        graph_metadata,
        FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS,
    )
    if missing_graph_pointers:
        raise RunSpecValidationError(
            "nominal GRU run spec is missing Feedbax graph pointer keys: "
            + ", ".join(missing_graph_pointers)
        )

    graph_spec_sidecar: Path | None = None
    for key in sorted(FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS):
        pointer = graph_metadata[key]
        if pointer is None and key == "graph_spec_path":
            status = graph_metadata.get("graph_export_status")
            if status == "unavailable":
                continue
            raise RunSpecValidationError(
                "nominal GRU run spec has no Feedbax graph sidecar but does not "
                "declare feedbax_graph.graph_export_status='unavailable'"
            )
        sidecar = spec_dir / str(pointer)
        if not require_graph_sidecars:
            continue
        if not sidecar.is_file():
            raise RunSpecValidationError(
                f"nominal GRU run spec points to missing Feedbax graph sidecar: {sidecar}"
            )
        if key == "graph_spec_path":
            graph_spec_sidecar = sidecar

    if graph_spec_sidecar is not None and _is_cs_lss_run_spec(run_spec):
        _validate_cs_lss_graph_spec_sidecar(graph_spec_sidecar)


def validate_nominal_gru_run_spec_file(run_spec_path: Path | str) -> None:
    """Load and validate a C&S-fidelity GRU recipe and its graph sidecars.

    Flat recipes use ``<recipe>.json`` for the tracked payload and the sibling
    ``<recipe>/`` directory for graph sidecars.
    """

    path = Path(run_spec_path)
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise RunSpecValidationError("nominal GRU run spec file must contain a JSON object")
    from rlrmp.runtime.training_run_specs import hydrate_compact_run_spec_envelope

    validate_nominal_gru_run_spec(
        hydrate_compact_run_spec_envelope(raw_payload),
        spec_dir=run_spec_sidecar_dir(path),
    )


def run_spec_sidecar_dir(run_spec_path: Path) -> Path:
    """Return the contract-owned graph-sidecar directory for one recipe path."""

    return run_spec_path.parent / run_spec_path.stem


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise RunSpecValidationError(f"nominal GRU run spec key {key!r} must be an object")
    return value


def _is_cs_lss_run_spec(run_spec: dict[str, Any]) -> bool:
    return evaluate_expr(
        _CS_LSS_PLANT_BACKEND_EXPR,
        ExpressionContext(
            items={"run_spec": ContextItem(kind="run_spec", payload=run_spec)}
        ),
    )


def _validate_cs_lss_graph_spec_sidecar(graph_spec_path: Path) -> None:
    try:
        payload = json.loads(graph_spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunSpecValidationError(
            f"CS-LSS Feedbax graph sidecar is not valid JSON: {graph_spec_path}"
        ) from exc

    nodes = payload.get("nodes")
    if not isinstance(nodes, dict):
        raise RunSpecValidationError(
            f"CS-LSS Feedbax graph sidecar must contain an object 'nodes': {graph_spec_path}"
        )

    node_types = {
        str(node_id): node.get("type") for node_id, node in nodes.items() if isinstance(node, dict)
    }
    mechanics_type = node_types.get("mechanics")
    feedback_type = node_types.get("feedback")
    legacy_types = sorted(
        node_type
        for node_type in set(node_types.values())
        if node_type in LEGACY_POINT_MASS_GRAPH_TYPES
    )
    if mechanics_type != CS_LSS_REQUIRED_MECHANICS_TYPE:
        legacy_note = (
            f"; stale legacy graph types present: {', '.join(legacy_types)}" if legacy_types else ""
        )
        raise RunSpecValidationError(
            "CS-LSS Feedbax graph sidecar must declare mechanics node type "
            f"{CS_LSS_REQUIRED_MECHANICS_TYPE!r}; found {mechanics_type!r}{legacy_note}"
        )
    if feedback_type not in CS_LSS_FEEDBACK_COMPONENT_TYPES:
        raise RunSpecValidationError(
            "CS-LSS Feedbax graph sidecar must declare a delayed position/velocity "
            f"feedback selector; found feedback node type {feedback_type!r}"
        )


def _resolve_training_manifest(
    exp: str,
    run: str,
    *,
    repo_root: Path,
) -> tuple[Path, TrainingRunManifest]:
    rel_flat = portable_repo_path(
        flat_run_spec_path(exp, run, repo_root=repo_root), repo_root=repo_root
    )
    rel_existing = portable_repo_path(
        run_spec_path(exp, run, repo_root=repo_root), repo_root=repo_root
    )
    matches: list[tuple[Path, TrainingRunManifest]] = []
    for path in _iter_training_manifest_paths(exp, run, repo_root=repo_root):
        loaded = load_manifest(path)
        if not isinstance(loaded, TrainingRunManifest):
            continue
        if _manifest_matches(loaded, run=run, rel_paths={rel_flat, rel_existing}):
            matches.append((path, loaded))
    if not matches:
        raise RunSpecValidationError(
            "TrainingRunManifest run record not_found for "
            f"{exp}/{run}; legacy archive-only specs are not canonical run records"
        )
    by_id: dict[str, tuple[Path, TrainingRunManifest, dict[str, Any]]] = {}
    for path, manifest in matches:
        dumped = manifest.model_dump(mode="json", exclude_none=True)
        previous = by_id.get(manifest.id)
        if previous is not None and previous[2] != dumped:
            raise RunSpecValidationError(
                f"same TrainingRunManifest id has different content for {exp}/{run}: "
                f"{previous[0]} and {path}"
            )
        by_id[manifest.id] = (path, manifest, dumped)
    distinct = [(path, manifest) for path, manifest, _dumped in by_id.values()]
    if len(distinct) > 1:
        selected = _select_preferred_training_manifest(distinct)
        if selected is not None:
            return selected
        raise RunSpecValidationError(
            f"multiple TrainingRunManifest records match {exp}/{run}: "
            + ", ".join(str(path) for path, _manifest in distinct)
        )
    return distinct[0]


def _iter_training_manifest_paths(exp: str, run: str, *, repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (
        repo_root / "_artifacts" / "feedbax_runs" / "manifests" / "training_runs",
        repo_root / "results" / exp / "manifests" / "training_runs",
    ):
        if root.is_dir():
            paths.extend(sorted(root.glob("*.json")))
    for name in (
        "training_run_manifest.json",
        "feedbax_training_run_manifest.json",
        "model.training_run.manifest.json",
    ):
        candidate = repo_root / "_artifacts" / exp / "runs" / run / name
        if candidate.is_file():
            paths.append(candidate)
    return sorted(dict.fromkeys(paths))


def _manifest_matches(
    manifest: TrainingRunManifest,
    *,
    run: str,
    rel_paths: set[str],
) -> bool:
    refs = set()
    if manifest.training_spec is not None and manifest.training_spec.ref:
        refs.add(manifest.training_spec.ref)
    for artifact in manifest.artifacts:
        if artifact.uri:
            refs.add(artifact.uri)
        original_uri = artifact.metadata.get("original_uri")
        if isinstance(original_uri, str):
            refs.add(original_uri)
    return (
        bool(refs & rel_paths)
        or _run_label_matches(manifest.job_id, run=run)
        or _run_label_matches(manifest.id, run=run)
        or manifest.id.endswith(run)
    )


def _run_label_matches(value: str | None, *, run: str) -> bool:
    if value is None:
        return False
    text = str(value)
    tail = text.rsplit(":", maxsplit=1)[-1].rsplit("/", maxsplit=1)[-1]
    return tail == run or tail.startswith(f"{run}-")


def _select_preferred_training_manifest(
    manifests: list[tuple[Path, TrainingRunManifest]],
) -> tuple[Path, TrainingRunManifest] | None:
    materialized = [
        item for item in manifests if not _is_spec_authoring_placeholder_manifest(item[1])
    ]
    if len(materialized) == 1:
        return materialized[0]

    completed = [
        item
        for item in materialized
        if str(item[1].status).lower() == "completed"
        and not _has_planned_only_summary(item[1])
    ]
    if len(completed) == 1:
        return completed[0]
    return None


def _is_spec_authoring_placeholder_manifest(manifest: TrainingRunManifest) -> bool:
    metadata = getattr(manifest.provenance, "metadata", None) or {}
    return (
        isinstance(metadata, dict)
        and metadata.get("producer") == "rlrmp.train.cs_nominal_gru.write_run_spec"
        and _has_planned_only_summary(manifest)
    )


def _has_planned_only_summary(manifest: TrainingRunManifest) -> bool:
    metrics = manifest.summary_metrics or {}
    return set(metrics) == {"planned_batches"}


def _missing_keys(mapping: dict[str, Any], required_keys: frozenset[str]) -> list[str]:
    return sorted(key for key in required_keys if key not in mapping)


__all__ = [
    "CS_LSS_FEEDBACK_COMPONENT_TYPES",
    "CS_LSS_PLANT_BACKEND",
    "CS_LSS_REQUIRED_MECHANICS_TYPE",
    "FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS",
    "LEGACY_POINT_MASS_GRAPH_TYPES",
    "NOMINAL_GRU_LOSS_OBJECTIVES",
    "NOMINAL_GRU_REQUIRED_PROVENANCE_KEYS",
    "NOMINAL_GRU_REQUIRED_TOP_LEVEL_KEYS",
    "NOMINAL_GRU_SCIENCE_MODES",
    "RunSpecValidationError",
    "run_spec_sidecar_dir",
    "resolve_run_record",
    "validate_nominal_gru_run_spec",
    "validate_nominal_gru_run_spec_file",
    "validate_minimax_run_spec",
    "validate_minimax_run_spec_file",
]

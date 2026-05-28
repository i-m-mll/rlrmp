"""RLRMP migration helpers for legacy Feedbax model artifacts.

This module owns the RLRMP-specific side of issue ``b41c940``. Feedbax owns the
general array-store and model-artifact schemas; RLRMP owns reconstruction of its
known legacy ``.eqx`` files and the mapping from historical run specs to those
schemas.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import equinox as eqx
import jax.tree_util as jtu
import numpy as np
from feedbax.artifact_schema import (
    ArrayStoreValidationError,
    array_store_ref,
    read_npz_array_store,
    validate_role_address,
    write_npz_array_store,
)
from feedbax.manifest import (
    ArtifactValidationRecord,
    ModelArtifactManifest,
    ParentRef,
    Provenance,
    SpecPayload,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

from rlrmp.eval.minimax_io import load_model
from rlrmp.feedbax_graph import (
    SCHEMA_VERSION as RLRMP_GRAPH_SCHEMA_VERSION,
    build_rlrmp_feedbax_graph_bundle,
    graph_spec_payload,
)
from rlrmp.train.minimax import build_hps


MIGRATION_SCHEMA_VERSION = "rlrmp.feedbax_artifact_migration.v1"
B_SET_ISSUES = ("efc4d68", "2bc95fd", "f47abb1", "3702f54", "b399efc")
DEFAULT_OUTPUT_ISSUE = "b41c940"

LEGACY_EXECUTION_BACKEND = "rlrmp.legacy_simple_feedback_compat"
CANONICAL_MODEL_FILENAME = "warmup_model.eqx"
MIGRATED_MODEL_ARRAY_STORE = "model.arrays.npz"
MIGRATION_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)


_DEFAULT_MINIMAX_ARGS: dict[str, Any] = {
    "n_warmup_batches": 12000,
    "batch_size": 250,
    "n_replicates": 5,
    "n_adversary_batches": 0,
    "n_adversary_steps": 5,
    "adversary_lr": 3e-4,
    "controller_lr": 1e-4,
    "adversary_type": "linear_dynamics",
    "linear_dynamics_eta_max": 0.1,
    "linear_dynamics_pgd_steps": 5,
    "linear_dynamics_lr": 1e-2,
    "n_bumps": 3,
    "force_max": 1.0,
    "n_adversaries": 1,
    "adv_batch_size": None,
    "warmup_model": None,
    "output_dir": None,
    "spec_dir": None,
    "jax_cache_dir": None,
    "jax_explain_cache_misses": False,
    "seed": 42,
    "checkpoint": True,
    "checkpoint_every": 1000,
    "resume": False,
    "loss_update_enabled": False,
    "loss_update_ratio": 0.5,
    "fused": True,
    "streaming_loss": False,
    "hidden_type": "gru",
    "nn_hidden_derivative": 0.0,
    "nn_output_jerk": 0.0,
    "nn_output_pre_go": 0.0,
    "nn_hidden_derivative_pre_go": 0.0,
    "sisu_gating": "additive",
    "effector_hold_pos": 10.0,
    "effector_hold_vel": 10.0,
    "effector_final_vel": 0.0,
    "effector_vel_late": 0.1,
    "effector_pos_running": 1.0,
    "effector_pos_late_weight": 0.5,
    "effector_pos_late_final_scale": 2.0,
    "effector_pos_late_start_step": 80,
    "effector_pos_running_schedule": "flat",
    "effector_hold_pos_schedule": "flat",
    "position_powerlaw_power": 6.0,
    "movement_ramp_shape": "linear",
    "movement_ramp_duration_steps": 60,
    "movement_ramp_power": 2.0,
    "p_catch_trial": 0.5,
    "nn_output": 1e-5,
    "nn_hidden": 1e-5,
}


@dataclass(frozen=True)
class LegacyRunArtifact:
    """Resolved legacy run inputs for one migration unit."""

    issue_id: str
    run_label: str
    run_spec_path: Path
    artifact_dir: Path
    model_path: Path
    run_spec: dict[str, Any]

    @property
    def artifact_id(self) -> str:
        return f"{self.issue_id}/{self.run_label}"


@dataclass(frozen=True)
class MigratedRunArtifact:
    """Paths and validation data produced by one migration."""

    legacy: LegacyRunArtifact
    tracked_dir: Path
    bulk_dir: Path
    graph_spec_path: Path
    manifest_path: Path
    array_store_path: Path
    array_count: int
    total_nbytes: int
    validation_status: Literal["passed", "failed"]


def discover_b_set_runs(repo_root: Path | str = Path(".")) -> list[LegacyRunArtifact]:
    """Discover selected B-set runs that have a canonical warmup checkpoint."""

    root = Path(repo_root)
    runs: list[LegacyRunArtifact] = []
    for issue_id in B_SET_ISSUES:
        for run_spec_path in _iter_run_specs(root, issue_id):
            run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
            run_label = _run_label(run_spec_path)
            artifact_dir = _resolve_artifact_dir(root, issue_id, run_label, run_spec)
            model_path = artifact_dir / CANONICAL_MODEL_FILENAME
            if model_path.exists():
                runs.append(
                    LegacyRunArtifact(
                        issue_id=issue_id,
                        run_label=run_label,
                        run_spec_path=run_spec_path,
                        artifact_dir=artifact_dir,
                        model_path=model_path,
                        run_spec=run_spec,
                    )
                )
    return sorted(runs, key=lambda run: (run.issue_id, run.run_label))


def minimax_args_from_run_spec(run_spec: dict[str, Any]) -> argparse.Namespace:
    """Build a minimax argparse namespace from historical run-spec JSON."""

    values = dict(_DEFAULT_MINIMAX_ARGS)
    for key, value in _normalized_cli_flags(run_spec).items():
        if key in values:
            values[key] = value
    for key, value in run_spec.items():
        if key in values and key not in {"cli_flags", "feedbax_graph"}:
            values[key] = value
    return argparse.Namespace(**values)


def migrate_legacy_run(
    legacy: LegacyRunArtifact,
    *,
    repo_root: Path | str = Path("."),
    tracked_root: Path | str | None = None,
    bulk_root: Path | str | None = None,
) -> MigratedRunArtifact:
    """Migrate one legacy RLRMP run to Feedbax model-artifact records."""

    root = Path(repo_root)
    tracked_base = Path(tracked_root) if tracked_root is not None else root / "results"
    bulk_base = Path(bulk_root) if bulk_root is not None else root / "_artifacts"
    tracked_dir = (
        tracked_base / DEFAULT_OUTPUT_ISSUE / "migrated" / legacy.issue_id / legacy.run_label
    )
    bulk_dir = bulk_base / DEFAULT_OUTPUT_ISSUE / "migrated" / legacy.issue_id / legacy.run_label
    tracked_dir.mkdir(parents=True, exist_ok=True)
    bulk_dir.mkdir(parents=True, exist_ok=True)

    args = minimax_args_from_run_spec(legacy.run_spec)
    hps = build_hps(args)
    bundle = build_rlrmp_feedbax_graph_bundle(hps)
    graph_payload = graph_spec_payload(bundle.graph_spec)
    graph_spec_path = tracked_dir / "model.graph.json"
    graph_spec_path.write_text(_json_dumps(graph_payload), encoding="utf-8")

    model = load_model(legacy.artifact_dir, CANONICAL_MODEL_FILENAME, hps, legacy.run_spec)
    if model is None:
        raise FileNotFoundError(legacy.model_path)

    arrays = extract_role_addressed_arrays(model, root_role="model")
    array_store_path = bulk_dir / MIGRATED_MODEL_ARRAY_STORE
    array_payload = write_npz_array_store(
        array_store_path,
        arrays,
        store_role="params",
        graph_spec_ref=str(graph_spec_path.relative_to(root)),
        graph_spec_sha256=sha256_file(graph_spec_path),
        metadata={
            "migration_schema_version": MIGRATION_SCHEMA_VERSION,
            "source_checkpoint": str(legacy.model_path.relative_to(root)),
            "source_run_spec": str(legacy.run_spec_path.relative_to(root)),
            "legacy_execution_backend": LEGACY_EXECUTION_BACKEND,
            "array_role_policy": "jax.tree_util path over eqx.is_array leaves",
        },
    )
    validation = validate_array_store_roundtrip(array_store_path, arrays)
    total_nbytes = int(sum(array.nbytes for array in arrays.values()))

    manifest = ModelArtifactManifest(
        id=f"rlrmp-model-artifact:{legacy.issue_id}/{legacy.run_label}/model",
        created_at=MIGRATION_TIMESTAMP,
        status="completed" if validation.status == "passed" else "failed",
        graph_spec=SpecPayload(
            kind="GraphSpec",
            inline=graph_payload,
            ref=str(graph_spec_path.relative_to(root)),
            sha256=sha256_bytes(canonical_json_bytes(graph_payload)),
            metadata={
                "rlrmp_graph_schema_version": RLRMP_GRAPH_SCHEMA_VERSION,
                "execution_backend": LEGACY_EXECUTION_BACKEND,
            },
        ),
        parameter_store=array_store_ref(
            array_store_path,
            array_payload,
            logical_name=MIGRATED_MODEL_ARRAY_STORE,
            artifact_id=f"rlrmp-array-store:{legacy.issue_id}/{legacy.run_label}/model",
        ),
        provenance=Provenance(
            issues=[DEFAULT_OUTPUT_ISSUE, legacy.issue_id],
            parents=[
                ParentRef(
                    kind="legacy_run_spec",
                    id=f"rlrmp-run-spec:{legacy.issue_id}/{legacy.run_label}",
                    uri=str(legacy.run_spec_path.relative_to(root)),
                ),
                ParentRef(
                    kind="legacy_checkpoint",
                    id=f"rlrmp-eqx:{legacy.issue_id}/{legacy.run_label}/warmup",
                    uri=str(legacy.model_path.relative_to(root)),
                    metadata={
                        "format": "feedbax._io.save",
                        "canonical_model_filename": CANONICAL_MODEL_FILENAME,
                    },
                ),
            ],
            metadata={
                "migration_schema_version": MIGRATION_SCHEMA_VERSION,
                "run_label": legacy.run_label,
                "source_issue": legacy.issue_id,
            },
        ),
        validation_records=[validation],
        metadata={
            "source_run_spec_digest": sha256_file(legacy.run_spec_path),
            "source_checkpoint_digest": sha256_file(legacy.model_path),
            "array_count": len(arrays),
            "total_array_nbytes": total_nbytes,
            "checkpoint_policy": (
                "warmup_model.eqx is the canonical legacy source checkpoint for "
                "this migration. The migrated model artifact is model.graph.json "
                "plus model.arrays.npz; adversary/intervention artifacts remain "
                "provenance because n_adversary_batches=0 for the B-set runs."
            ),
        },
    )
    manifest_path = tracked_dir / "model.artifact.manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True) + "\n")
    summary_path = tracked_dir / "migration_summary.json"
    summary_path.write_text(
        _json_dumps(
            {
                "schema_version": MIGRATION_SCHEMA_VERSION,
                "source_issue": legacy.issue_id,
                "run_label": legacy.run_label,
                "source_run_spec": str(legacy.run_spec_path.relative_to(root)),
                "source_checkpoint": str(legacy.model_path.relative_to(root)),
                "model_artifact_manifest": str(manifest_path.relative_to(root)),
                "array_store": str(array_store_path.relative_to(root)),
                "array_count": len(arrays),
                "total_array_nbytes": total_nbytes,
                "validation": validation.model_dump(mode="json", exclude_none=True),
            }
        ),
        encoding="utf-8",
    )

    return MigratedRunArtifact(
        legacy=legacy,
        tracked_dir=tracked_dir,
        bulk_dir=bulk_dir,
        graph_spec_path=graph_spec_path,
        manifest_path=manifest_path,
        array_store_path=array_store_path,
        array_count=len(arrays),
        total_nbytes=total_nbytes,
        validation_status=validation.status,
    )


def extract_role_addressed_arrays(
    tree: Any,
    *,
    root_role: str = "model",
) -> dict[str, np.ndarray]:
    """Extract array leaves under deterministic semantic role addresses."""

    arrays: dict[str, np.ndarray] = {}
    for path, leaf in jtu.tree_leaves_with_path(eqx.filter(tree, eqx.is_array)):
        if leaf is None:
            continue
        role = ".".join([root_role, *(_path_part(part) for part in path)])
        role = validate_role_address(role)
        if role in arrays:
            raise ArrayStoreValidationError(f"Duplicate array role address: {role}")
        arrays[role] = np.asarray(leaf)
    if not arrays:
        raise ArrayStoreValidationError("No array leaves found in artifact.")
    return arrays


def validate_array_store_roundtrip(
    path: Path | str,
    expected_arrays: dict[str, np.ndarray],
) -> ArtifactValidationRecord:
    """Validate that a written Feedbax array store preserves roles and values."""

    loaded = read_npz_array_store(path)
    missing = sorted(set(expected_arrays) - set(loaded.arrays))
    unexpected = sorted(set(loaded.arrays) - set(expected_arrays))
    mismatched: list[str] = []
    for role, expected in expected_arrays.items():
        actual = loaded.arrays.get(role)
        if actual is not None and not np.array_equal(np.asarray(expected), np.asarray(actual)):
            mismatched.append(role)
    status: Literal["passed", "failed"] = (
        "passed" if not missing and not unexpected and not mismatched else "failed"
    )
    return ArtifactValidationRecord(
        name="array_store_roundtrip",
        status=status,
        checked_at=MIGRATION_TIMESTAMP,
        schema_version=loaded.payload.schema_version,
        details={
            "array_count": len(expected_arrays),
            "missing_roles": missing,
            "unexpected_roles": unexpected,
            "mismatched_roles": mismatched,
        },
    )


def _iter_run_specs(root: Path, issue_id: str) -> Iterable[Path]:
    runs_dir = root / "results" / issue_id / "runs"
    if not runs_dir.exists():
        return []
    return sorted(
        [
            *runs_dir.glob("*.json"),
            *runs_dir.glob("*/run.json"),
        ]
    )


def _run_label(run_spec_path: Path) -> str:
    if run_spec_path.name == "run.json":
        return run_spec_path.parent.name
    return run_spec_path.stem


def _resolve_artifact_dir(
    root: Path,
    issue_id: str,
    run_label: str,
    run_spec: dict[str, Any],
) -> Path:
    candidates: list[Path] = []
    output_dir = _normalized_cli_flags(run_spec).get("output_dir") or run_spec.get("output_dir")
    if output_dir:
        path = Path(str(output_dir))
        if not path.is_absolute():
            candidates.append(root / path)
        candidates.append(root / "_artifacts" / issue_id / "runs" / run_label)
        candidates.append(root / "_artifacts" / issue_id / run_label)
    else:
        candidates.extend(
            [
                root / "_artifacts" / issue_id / "runs" / run_label,
                root / "_artifacts" / issue_id / run_label,
            ]
        )
    for candidate in candidates:
        if (candidate / CANONICAL_MODEL_FILENAME).exists():
            return candidate
    return candidates[0]


def _normalized_cli_flags(run_spec: dict[str, Any]) -> dict[str, Any]:
    flags = run_spec.get("cli_flags") or {}
    normalized: dict[str, Any] = {}
    for key, value in flags.items():
        normalized_key = str(key).lstrip("-").replace("-", "_")
        if normalized_key.startswith("no_") and value is True:
            normalized[normalized_key[3:]] = False
        else:
            normalized[normalized_key] = value
    return normalized


def _path_part(part: Any) -> str:
    if hasattr(part, "name"):
        raw = str(part.name)
    elif hasattr(part, "key"):
        raw = str(part.key)
    elif hasattr(part, "idx"):
        raw = f"{int(part.idx):04d}"
    else:
        raw = str(part)
    return _sanitize_role_part(raw)


def _sanitize_role_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.:/@+-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unnamed"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"

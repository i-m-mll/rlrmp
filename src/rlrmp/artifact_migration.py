"""Load RLRMP models migrated to Feedbax model-artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np
from feedbax.contracts.artifact_schema import (
    ARRAY_STORE_SCHEMA_VERSION,
    METADATA_KEY,
    ArrayStore,
    ArrayStorePayload,
    ArrayStoreValidationError,
    validate_role_coverage,
)
from feedbax.contracts.graphs.materialization import (
    materialize_array_store,
    materialize_model_artifact,
)
from feedbax.contracts.expressions import (
    Compare,
    ContextItem,
    ExpressionContext,
    ExpressionPathMissing,
    ExpressionSelectAmbiguous,
    Select,
    ValueQuery,
    evaluate_query,
)
from feedbax.contracts.manifest import ModelArtifactManifest, ParentRef, sha256_file

from rlrmp.eval.minimax_io import (
    make_legacy_minimax_model_template,
    normalize_loaded_minimax_runtime,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.minimax import build_hps


LEGACY_EXECUTION_BACKEND = "rlrmp.legacy_simple_feedback_compat"
_LEGACY_RLRMP_GRAPH_SCHEMA_VERSION = "rlrmp.feedbax_graph.v1"
_FEEDBAX_LEGACY_GRAPH_SCHEMA_VERSION = "1.0.0"
_LEGACY_ARRAY_STORE_SCHEMA_VERSION = "feedbax.array_store.v1"


# Frozen fallback defaults for reconstructing historical (pre-migration) run
# specs (issue b41c940). This dict is pinned to the CLI defaults in effect
# when those runs launched and must NOT be kept in sync with live
# MINIMAX_CONFIG_DEFAULTS in rlrmp.train.minimax. The two are allowed to
# diverge on keys whose live defaults changed after those runs were recorded.
# See tests/test_artifact_migration.py for the allowlisted drift guard.
_DEFAULT_MINIMAX_ARGS: dict[str, Any] = {
    "n_warmup_batches": 12000,
    "batch_size": 250,
    "n_replicates": 5,
    "n_adversary_batches": 0,
    "n_adversary_steps": 5,
    "adversary_lr": 3e-4,
    "controller_lr": 1e-4,
    "adversary_type": "gaussian_bump",
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
    "allow_x64": False,
    "seed": 42,
    "checkpoint": True,
    "checkpoint_every": 1000,
    "resume": False,
    "allow_fresh_start": False,
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


def load_migrated_model_artifact(
    manifest_path: Path | str,
    *,
    repo_root: Path | str = Path("."),
    key=None,
) -> Any:
    """Load an RLRMP migrated model artifact from its Feedbax manifest.

    The artifacts produced for ``b41c940`` declare the execution backend as
    ``rlrmp.legacy_simple_feedback_compat``. Rehydration therefore uses the
    legacy run spec recorded in the manifest to rebuild the executable template,
    then fills that template from the Feedbax role-addressed array store.
    """

    root = Path(repo_root)
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    manifest_data = json.loads(path.read_text(encoding="utf-8"))
    legacy_array_store = _uses_legacy_array_store_schema(manifest_data)
    manifest = ModelArtifactManifest.model_validate(
        _normalized_legacy_manifest_payload(manifest_data)
    )

    backend = (manifest.graph_spec.metadata or {}).get("execution_backend")
    if backend != LEGACY_EXECUTION_BACKEND:
        raise ValueError(
            f"Unsupported migrated artifact execution backend {backend!r}; "
            f"expected {LEGACY_EXECUTION_BACKEND!r}."
        )

    run_spec_parent = _manifest_parent(manifest, "legacy_run_spec")
    run_spec_path = root / run_spec_parent.uri
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    hps = build_hps(minimax_args_from_run_spec(run_spec))

    template_key = jr.PRNGKey(0) if key is None else key
    if legacy_array_store:
        template = make_legacy_minimax_model_template(hps, key=template_key)
    else:
        template = setup_task_model_pair(hps, key=template_key).model

    if manifest.parameter_store is None:
        raise ValueError(f"Manifest {path} has no parameter_store.")
    if legacy_array_store:
        model = _materialize_legacy_array_store(
            manifest,
            template,
            root=root,
            root_role="model",
        )
    else:
        model = materialize_model_artifact(path, template, root=root, root_role="model")
    return normalize_loaded_minimax_runtime(model, hps, key=jr.PRNGKey(0))


def _normalized_legacy_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    graph_spec = normalized.get("graph_spec")
    if isinstance(graph_spec, dict):
        metadata = graph_spec.get("metadata") or {}
        if (
            metadata.get("rlrmp_graph_schema_version") == _LEGACY_RLRMP_GRAPH_SCHEMA_VERSION
            and graph_spec.get("schema_version") is None
        ):
            graph_spec["schema_version"] = _FEEDBAX_LEGACY_GRAPH_SCHEMA_VERSION

    parameter_store = normalized.get("parameter_store")
    if (
        isinstance(parameter_store, dict)
        and parameter_store.get("schema_version") == _LEGACY_ARRAY_STORE_SCHEMA_VERSION
    ):
        parameter_store["schema_version"] = ARRAY_STORE_SCHEMA_VERSION
    return normalized


def _uses_legacy_array_store_schema(payload: dict[str, Any]) -> bool:
    parameter_store = payload.get("parameter_store")
    return (
        isinstance(parameter_store, dict)
        and parameter_store.get("schema_version") == _LEGACY_ARRAY_STORE_SCHEMA_VERSION
    )


def _materialize_legacy_array_store(
    manifest: ModelArtifactManifest,
    template: Any,
    *,
    root: Path,
    root_role: str,
) -> Any:
    store_ref = manifest.parameter_store
    if store_ref is None or store_ref.uri is None:
        raise ArrayStoreValidationError(f"Model artifact {manifest.id!r} has no array store URI.")

    store_path = Path(store_ref.uri)
    if not store_path.is_absolute():
        store_path = root / store_path
    if store_ref.sha256 and sha256_file(store_path) != store_ref.sha256:
        raise ArrayStoreValidationError(f"Array store digest mismatch for {store_path}.")

    store = _read_legacy_npz_array_store(store_path)
    if store.payload.store_role != store_ref.role:
        raise ArrayStoreValidationError(
            f"Array store role mismatch: manifest={store_ref.role!r}, "
            f"store={store.payload.store_role!r}"
        )
    return materialize_array_store(template, store, root_role=root_role)


def _read_legacy_npz_array_store(path: Path) -> ArrayStore:
    with np.load(path, allow_pickle=False) as npz:
        if METADATA_KEY not in npz.files:
            raise ArrayStoreValidationError(
                f"NPZ array store is missing metadata member {METADATA_KEY!r}."
            )
        metadata = json.loads(npz[METADATA_KEY].tobytes().decode("utf-8"))
        if metadata.get("schema_version") == _LEGACY_ARRAY_STORE_SCHEMA_VERSION:
            metadata["schema_version"] = ARRAY_STORE_SCHEMA_VERSION
        payload = ArrayStorePayload.model_validate(metadata)

        arrays: dict[str, np.ndarray] = {}
        seen_storage_keys: set[str] = set()
        for record in payload.arrays:
            if record.storage_key not in npz.files:
                raise ArrayStoreValidationError(
                    f"Array role {record.role!r} is missing storage key {record.storage_key!r}."
                )
            array = np.asarray(npz[record.storage_key])
            seen_storage_keys.add(record.storage_key)
            if str(array.dtype) != record.dtype or tuple(array.shape) != tuple(record.shape):
                raise ArrayStoreValidationError(
                    f"Array role {record.role!r} metadata mismatch: "
                    f"expected dtype={record.dtype}, shape={record.shape}; "
                    f"found dtype={array.dtype}, shape={tuple(array.shape)}."
                )
            digest = hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()
            if digest != record.sha256:
                raise ArrayStoreValidationError(f"Array role {record.role!r} digest mismatch.")
            arrays[record.role] = array

        extra_keys = sorted(set(npz.files) - seen_storage_keys - {METADATA_KEY})
        if extra_keys:
            raise ArrayStoreValidationError(f"NPZ array store has unknown members: {extra_keys}")

    validate_role_coverage(payload.roles)
    return ArrayStore(payload, arrays)


def _manifest_parent(manifest: ModelArtifactManifest, kind: str) -> ParentRef:
    query = ValueQuery(
        item="manifest",
        path="provenance.parents",
        select=Select(
            where=Compare(item="entry", path="kind", op="eq", value=kind),
        ),
    )
    try:
        return evaluate_query(
            query,
            ExpressionContext(
                items={
                    "manifest": ContextItem(kind="ModelArtifactManifest", payload=manifest),
                }
            ),
        )
    except (ExpressionSelectAmbiguous, ExpressionPathMissing) as exc:
        matches = [parent for parent in manifest.provenance.parents if parent.kind == kind]
        raise ValueError(
            f"Expected exactly one {kind!r} parent in manifest {manifest.id}; found {len(matches)}."
        ) from exc


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

"""RLRMP adapter for Feedbax training-run matrix checkpoint forks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.checkpoints import CheckpointForkBarrierMapping
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
)
from feedbax.training.run_matrix import (
    ForkParityError,
    fork_matrix_checkpoints,
    materialize_run_matrix,
)

from rlrmp.runtime.lr_continuation import RlrmpLrContinuationReporter
from rlrmp.runtime.adaptive_checkpoint_adapter import NominalToAdaptiveSlotAdapter
from rlrmp.runtime.training_run_specs import (
    register_rlrmp_cs_supervised_method,
    register_rlrmp_distillation_methods,
)
from rlrmp.train.adaptive_epsilon_native import (
    ensure_adaptive_epsilon_training_method_registered,
)
from rlrmp.train.minimax_native import (
    ensure_minimax_training_method_registered,
)


_TASK_IDENTITY_METADATA_KEY = "rlrmp_task_identity"
_TASK_IDENTITY_SUBTREES = ("game_card", "perturbation_training")
_TASK_SOURCE_RUN_SPEC_REF_METADATA_KEY = "rlrmp_source_run_spec_ref"
_TASK_ROW_SPEC_PARAMS_PATH = "task.params"
_RATIO_SETPOINT_METADATA_KEY = "ratio_setpoint"
_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH = Path(
    "results/5cd9feb/runs/stage2_lambda_update_contract.json"
)
_ADAPTIVE_EPSILON_METHOD_REF = "rlrmp/adaptive_epsilon_curriculum/v1"


@dataclass(frozen=True)
class ForkTarget:
    """One target row checkpoint root for a matrix fork."""

    row_id: str
    checkpoint_root: Path


def register_rlrmp_training_methods() -> None:
    """Register RLRMP Feedbax training methods in this process."""

    ensure_adaptive_epsilon_training_method_registered()
    ensure_minimax_training_method_registered()
    register_rlrmp_cs_supervised_method()
    register_rlrmp_distillation_methods()


def parse_target(value: str) -> ForkTarget:
    """Parse ``ROW=CHECKPOINT_ROOT`` into a fork target."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("target must be ROW=CHECKPOINT_ROOT")
    row_id, checkpoint_root = value.split("=", 1)
    if not row_id:
        raise argparse.ArgumentTypeError("target row id must not be empty")
    if not checkpoint_root:
        raise argparse.ArgumentTypeError("target checkpoint root must not be empty")
    return ForkTarget(row_id=row_id, checkpoint_root=Path(checkpoint_root))


def load_matrix(path: Path) -> TrainingRunMatrixSpec:
    """Load and validate one ``TrainingRunMatrixSpec`` document."""

    return TrainingRunMatrixSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


def fork_checkpoints_with_parity(
    *,
    matrix_path: Path,
    source_checkpoint_root: Path,
    targets: Sequence[ForkTarget],
    parity_output_path: Path,
    repo_root: Path | None = None,
    skip_fork: bool = False,
) -> dict[str, Any]:
    """Materialize a matrix, fork row checkpoints, and write Feedbax parity JSON."""

    if not targets:
        raise ValueError("at least one fork target is required")
    register_rlrmp_training_methods()
    matrix = load_matrix(matrix_path)
    resolved_repo_root = Path.cwd() if repo_root is None else repo_root
    materialized = materialize_run_matrix(
        matrix,
        repo_root=resolved_repo_root,
        method_registry=DEFAULT_TRAINING_METHOD_REGISTRY,
    )
    _validate_fork_prelaunch_contracts(matrix, materialized, repo_root=resolved_repo_root)
    ratio_setpoint = _ratio_setpoint_prelaunch_report(matrix)
    target_roots = {target.row_id: target.checkpoint_root for target in targets}
    adaptive_contracts = _adaptive_continuation_fork_contracts(materialized)
    reporter = RlrmpLrContinuationReporter(source_checkpoint_root=source_checkpoint_root)
    feedbax_matrix = matrix
    if adaptive_contracts and matrix.fork is not None:
        # Feedbax's byte-parity table treats declared target-only slots as a
        # topology mismatch. RLRMP composes and enforces topology-aware parity
        # from custody provenance immediately below.
        feedbax_matrix = matrix.model_copy(
            update={"fork": matrix.fork.model_copy(update={"parity": "skip"})}
        )
    table = fork_matrix_checkpoints(
        feedbax_matrix,
        materialized,
        source_checkpoint_root=source_checkpoint_root,
        target_checkpoint_roots=target_roots,
        parity_output_path=parity_output_path,
        target_slot_templates={row_id: value[0].adaptive_initial_slots for row_id, value in adaptive_contracts.items()},
        row_continuation_slot_templates={row_id: value[0].continuation_slot_templates() for row_id, value in adaptive_contracts.items()},
        row_target_slot_transforms={row_id: value[0].transform for row_id, value in adaptive_contracts.items()},
        row_target_transform_metadata={row_id: value[0].transform_metadata for row_id, value in adaptive_contracts.items()},
        row_target_transformed_slots={row_id: value[0].target_transformed_slots for row_id, value in adaptive_contracts.items()},
        row_target_only_slots={row_id: value[0].target_only_slots for row_id, value in adaptive_contracts.items()},
        row_barrier_mappings={row_id: value[1] for row_id, value in adaptive_contracts.items()},
        skip_fork=skip_fork,
        lr_reporter=reporter,
        tool_version="rlrmp.checkpoint_fork_gate.v2",
    )
    if adaptive_contracts:
        table = _compose_adaptive_fork_parity(
            table,
            matrix=matrix,
            materialized=materialized,
            target_roots=target_roots,
            adaptive_contracts=adaptive_contracts,
        )
    if ratio_setpoint is not None:
        table["ratio_setpoint"] = ratio_setpoint
    parity_output_path.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if table.get("ok") is not True:
        raise ForkParityError("composed checkpoint fork parity is not green")
    return table


def _compose_adaptive_fork_parity(
    feedbax_table: Mapping[str, Any],
    *,
    matrix: TrainingRunMatrixSpec,
    materialized: Any,
    target_roots: Mapping[str, Path],
    adaptive_contracts: Mapping[
        str, tuple[NominalToAdaptiveSlotAdapter, CheckpointForkBarrierMapping]
    ],
) -> dict[str, Any]:
    """Compose topology-aware parity from custody provenance and LR evidence."""

    lr_rows = [
        dict(row)
        for row in feedbax_table.get("rows", [])
        if isinstance(row, Mapping) and row.get("kind") == "lr_continuation"
    ]
    lr_row_ids = {str(row.get("row_id")) for row in lr_rows}
    expected_row_ids = set(adaptive_contracts)
    if lr_row_ids != expected_row_ids:
        raise ForkParityError(
            "adaptive fork parity LR row coverage mismatch; "
            f"expected={sorted(expected_row_ids)!r} actual={sorted(lr_row_ids)!r}"
        )
    planned_ids = {row.row_id: row.planned_run_id for row in materialized.rows}
    expected_slots = tuple(matrix.fork.expected_slots if matrix.fork is not None else ())
    parity_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row_id, (adapter, _mapping) in adaptive_contracts.items():
        manifest = _read_latest_manifest(target_roots[row_id])
        provenance = manifest.get("fork_provenance")
        records = provenance.get("slots") if isinstance(provenance, Mapping) else None
        by_slot = {
            str(record.get("slot")): record
            for record in records or []
            if isinstance(record, Mapping) and isinstance(record.get("slot"), str)
        }
        for slot in expected_slots:
            record = by_slot.get(slot, {})
            source_sha = record.get("source_sha256")
            target_sha = record.get("target_sha256")
            transformed = slot in adapter.target_transformed_slots
            transform = record.get("transform")
            transform_identity = (
                transform.get("identity") if isinstance(transform, Mapping) else None
            )
            ok = (
                isinstance(source_sha, str)
                and isinstance(target_sha, str)
                and (
                    transform_identity == adapter.transform_metadata["identity"]
                    if transformed
                    else source_sha == target_sha
                )
            )
            if not ok:
                mismatches.append(f"row={row_id} slot={slot}")
            parity_rows.append(
                {
                    "kind": "slot_parity",
                    "row_id": row_id,
                    "planned_run_id": planned_ids[row_id],
                    "transaction_id": manifest.get("transaction_id"),
                    "slot": slot,
                    "source_digest": source_sha,
                    "target_digest": target_sha,
                    "comparison": "declared_transform" if transformed else "byte_identical",
                    "transform_identity": transform_identity,
                    "ok": ok,
                }
            )
    table = dict(feedbax_table)
    table["rows"] = [*parity_rows, *lr_rows]
    table["ok"] = not mismatches
    if mismatches:
        raise ForkParityError("; ".join(mismatches))
    return table


def _read_latest_manifest(root: Path) -> dict[str, Any]:
    latest_path = root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    relative_path = latest.get("manifest_relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ForkParityError(f"checkpoint latest pointer lacks manifest path: {latest_path}")
    manifest = json.loads((root / relative_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ForkParityError(f"checkpoint manifest is not an object: {root / relative_path}")
    return manifest


def _adaptive_continuation_fork_contracts(
    materialized: Any,
) -> dict[str, tuple[NominalToAdaptiveSlotAdapter, CheckpointForkBarrierMapping]]:
    """Build the declared target topology and C&S-to-adaptive barrier map."""

    import jax.random as jr
    from rlrmp.train.adaptive_epsilon_native import (
        AdaptiveEpsilonMethodPayload,
        AdaptiveEpsilonNativeRuntime,
        build_adaptive_epsilon_native_initial_slots,
    )
    from rlrmp.train.cs_nominal_gru import _config_namespace, build_hps

    contracts = {}
    for row in materialized.rows:
        spec = row.spec
        if spec is None or spec.checkpoint_progress.continuation is None:
            continue
        method_ref = spec.method_ref
        if f"{method_ref.package}/{method_ref.name}/{method_ref.version}" != _ADAPTIVE_EPSILON_METHOD_REF:
            continue
        payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
            method_ref, spec.method_payload, path=f"/rows/{row.row_id}/method_payload"
        )
        if not isinstance(payload, AdaptiveEpsilonMethodPayload):
            raise ForkParityError(f"adaptive adapter invalid payload row={row.row_id!r}")
        args = _config_namespace(payload.config)
        initial_slots, runtime = build_adaptive_epsilon_native_initial_slots(
            run_spec=spec,
            hps=build_hps(args),
            args=args,
            key=jr.PRNGKey(int(args.seed)),
            lr_continuation_mode=payload.lr_continuation_mode,
        )
        native = runtime.component("adaptive_epsilon")
        if not isinstance(native, AdaptiveEpsilonNativeRuntime):
            raise ForkParityError(f"adaptive adapter missing runtime row={row.row_id!r}")
        target_barrier = next(
            barrier
            for barrier in spec.worker_execution.method_contract.phase_program.checkpoint_barriers
            if barrier.name == "after_adaptive_epsilon_train_chunk"
        )
        request = spec.checkpoint_progress.continuation
        mapping = CheckpointForkBarrierMapping(
            source_barrier="after_train_chunk",
            target_barrier=target_barrier.name,
            target_coordinate=ProgressCoordinate(
                run_id=row.planned_run_id,
                phase=target_barrier.phase,
                program_step=request.source_completed_batches,
                completed_barrier=target_barrier.name,
            ),
            coordinate_mapping={
                "identity": "rlrmp.cs_supervised_to_adaptive_epsilon.v1",
                "parameters": {"program_step": "preserve_completed_training_batches"},
            },
        )
        contracts[row.row_id] = (
            NominalToAdaptiveSlotAdapter(
                model_template=native.model_template,
                optimizer_template=native.optimizer_template,
                adaptive_initial_slots=initial_slots,
            ),
            mapping,
        )
    return contracts


def _validate_fork_prelaunch_contracts(
    matrix: TrainingRunMatrixSpec,
    materialized: Any,
    *,
    repo_root: Path,
) -> None:
    """Fail before any fork write when task or LR continuation contracts drift."""

    source_identity = _task_identity_from_source_run_spec(matrix, repo_root=repo_root)
    source_hash = _canonical_task_identity_hash(source_identity)
    _assert_task_identity_label(
        label=_task_identity_label_from_mapping(
            matrix.metadata,
            path=f"matrix.metadata.{_TASK_IDENTITY_METADATA_KEY}",
        ),
        derived_hash=source_hash,
        path=f"matrix.metadata.{_TASK_IDENTITY_METADATA_KEY}",
    )
    row_metadata = {row.row_id: row.metadata for row in matrix.rows}
    for row in materialized.rows:
        target_identity = _task_identity_from_row_spec(row_id=row.row_id, row_spec=row.spec)
        target_hash = _canonical_task_identity_hash(target_identity)
        _assert_task_identity_label(
            label=_task_identity_label_from_mapping(
                row_metadata.get(row.row_id),
                path=(f"row={row.row_id!r}.metadata.{_TASK_IDENTITY_METADATA_KEY}"),
            ),
            derived_hash=target_hash,
            path=f"row={row.row_id!r}.metadata.{_TASK_IDENTITY_METADATA_KEY}",
        )
        _assert_matching_task_identity(
            row_id=row.row_id,
            source_identity=source_identity,
            target_identity=target_identity,
            source_hash=source_hash,
            target_hash=target_hash,
        )
        _assert_payload_lr_continuation_mode(
            row_id=row.row_id,
            row_spec=row.spec,
            declared_mode=matrix.fork.lr_continuation if matrix.fork is not None else None,
        )
        _assert_stage2_lambda_update_contract(
            row_id=row.row_id,
            row_spec=row.spec,
            repo_root=repo_root,
        )


def _task_identity_from_source_run_spec(
    matrix: TrainingRunMatrixSpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Read fork-task subtrees from the source run's tracked outer recipe."""

    source_ref = matrix.metadata.get(_TASK_SOURCE_RUN_SPEC_REF_METADATA_KEY)
    source_ref_path = f"matrix.metadata.{_TASK_SOURCE_RUN_SPEC_REF_METADATA_KEY}"
    if not isinstance(source_ref, str) or not source_ref.strip():
        raise ForkParityError(f"task identity gate missing {source_ref_path}")
    relative_source_path = Path(source_ref)
    if relative_source_path.is_absolute():
        raise ForkParityError(f"task identity gate requires repo-relative {source_ref_path}")
    source_path = (repo_root / relative_source_path).resolve()
    try:
        source_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ForkParityError(f"task identity gate rejects escaping {source_ref_path}") from exc
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ForkParityError(
            f"task identity gate missing source run spec {source_ref_path}={source_ref!r}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ForkParityError(
            f"task identity gate source run spec is not JSON {source_ref_path}={source_ref!r}"
        ) from exc
    if not isinstance(source, Mapping):
        raise ForkParityError(
            f"task identity gate source run spec must be an object {source_ref_path}={source_ref!r}"
        )
    return _task_identity_from_outer_run_spec(
        source,
        path=f"source_run_spec={source_ref!r}",
    )


def _task_identity_from_outer_run_spec(
    run_spec: Mapping[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    """Extract source task subtrees from a full or compact RLRMP run recipe."""

    canonical: Mapping[str, Any] = run_spec
    if not isinstance(canonical.get("game_card"), Mapping):
        extension = canonical.get("rlrmp_run_spec")
        if isinstance(extension, Mapping):
            canonical = extension
    training_distribution = canonical.get("training_distribution")
    if not isinstance(training_distribution, Mapping):
        raise ForkParityError(f"task identity gate missing {path}.training_distribution")
    return _task_identity_from_subtrees(
        game_card=canonical.get("game_card"),
        perturbation_training=training_distribution.get("perturbation_training"),
        path=path,
    )


def _task_identity_from_row_spec(*, row_id: str, row_spec: Any) -> dict[str, Any]:
    """Extract the task subtrees carried by one materialized row ``TrainingRunSpec``."""

    task = getattr(row_spec, "task", None)
    params = getattr(task, "params", None)
    path = f"row={row_id!r}.spec.{_TASK_ROW_SPEC_PARAMS_PATH}"
    if not isinstance(params, Mapping):
        raise ForkParityError(f"task identity gate missing {path}")
    return _task_identity_from_subtrees(
        game_card=params.get("game_card"),
        perturbation_training=params.get("perturbation_training"),
        path=path,
    )


def _task_identity_from_subtrees(
    *,
    game_card: Any,
    perturbation_training: Any,
    path: str,
) -> dict[str, Any]:
    """Validate and return the two governed task-identity subtrees."""

    values = {
        "game_card": game_card,
        "perturbation_training": perturbation_training,
    }
    for subtree, value in values.items():
        if not isinstance(value, Mapping):
            raise ForkParityError(f"task identity gate missing {path}.{subtree}")
    return dict(values)


def _canonical_task_identity_hash(identity: Mapping[str, Any]) -> str:
    """Return the canonical JSON SHA-256 for the two exact task subtrees."""

    try:
        payload = json.dumps(
            {
                subtree: identity[subtree]
                for subtree in _TASK_IDENTITY_SUBTREES
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (KeyError, TypeError, ValueError) as exc:
        raise ForkParityError("task identity gate cannot canonicalize task subtrees") from exc
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _task_identity_label_from_mapping(mapping: Any, *, path: str) -> str:
    if not isinstance(mapping, dict) or _TASK_IDENTITY_METADATA_KEY not in mapping:
        raise ForkParityError(f"task identity gate missing {path}")
    label = mapping[_TASK_IDENTITY_METADATA_KEY]
    if not isinstance(label, str) or not label:
        raise ForkParityError(f"task identity gate requires nonempty string label at {path}")
    return label


def _assert_task_identity_label(*, label: str, derived_hash: str, path: str) -> None:
    if label != derived_hash:
        raise ForkParityError(
            f"task identity label mismatch field={path}: label={label!r} derived={derived_hash!r}"
        )


def _assert_matching_task_identity(
    *,
    row_id: str,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    source_hash: str,
    target_hash: str,
) -> None:
    if source_hash == target_hash:
        return
    for subtree in _TASK_IDENTITY_SUBTREES:
        mismatch = _first_subtree_mismatch(
            source_identity[subtree],
            target_identity[subtree],
            path=subtree,
        )
        if mismatch is None:
            continue
        path, source_value, target_value = mismatch
        raise ForkParityError(
            f"task identity hash mismatch row={row_id!r} path={path!r}: "
            f"source_hash={source_hash!r} row_hash={target_hash!r} "
            f"source={source_value} target={target_value}"
        )
    raise ForkParityError(
        f"task identity hash mismatch row={row_id!r}: "
        f"source_hash={source_hash!r} row_hash={target_hash!r}"
    )


def _first_subtree_mismatch(
    source: Any,
    target: Any,
    *,
    path: str,
) -> tuple[str, str, str] | None:
    if isinstance(source, dict) and isinstance(target, dict):
        source_keys = set(source)
        target_keys = set(target)
        if source_keys != target_keys:
            return (
                path,
                _render_value({"keys": sorted(source_keys)}),
                _render_value({"keys": sorted(target_keys)}),
            )
        for key in sorted(source_keys):
            mismatch = _first_subtree_mismatch(
                source[key],
                target[key],
                path=f"{path}.{key}",
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(source, list) and isinstance(target, list):
        if len(source) != len(target):
            return path, _render_value(source), _render_value(target)
        for index, (source_item, target_item) in enumerate(zip(source, target, strict=True)):
            mismatch = _first_subtree_mismatch(
                source_item,
                target_item,
                path=f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    if type(source) is not type(target) or source != target:
        return path, _render_value(source), _render_value(target)
    return None


def _assert_payload_lr_continuation_mode(
    *,
    row_id: str,
    row_spec: Any,
    declared_mode: str | None,
) -> None:
    if row_spec is None:
        raise ForkParityError(f"LR continuation gate missing canonical row spec for row={row_id!r}")
    method_ref = row_spec.method_ref
    method_ref_string = f"{method_ref.package}/{method_ref.name}/{method_ref.version}"
    if method_ref_string != "rlrmp/adaptive_epsilon_curriculum/v1":
        return
    payload = row_spec.method_payload.payload
    payload_mode = payload.get("lr_continuation_mode") if isinstance(payload, dict) else None
    if payload_mode != declared_mode:
        rendered_payload = "<missing>" if payload_mode is None else repr(payload_mode)
        raise ForkParityError(
            "LR continuation mode mismatch "
            f"row={row_id!r}: declared={declared_mode!r} payload={rendered_payload}"
        )


def _assert_stage2_lambda_update_contract(
    *,
    row_id: str,
    row_spec: Any,
    repo_root: Path,
) -> None:
    """Require every adaptive row to preserve the tracked Stage-2 clamp retune."""

    method_ref = getattr(row_spec, "method_ref", None)
    method_ref_string = (
        f"{method_ref.package}/{method_ref.name}/{method_ref.version}"
        if method_ref is not None
        else "<missing>"
    )
    if method_ref_string != _ADAPTIVE_EPSILON_METHOD_REF:
        return

    contract = _load_stage2_lambda_update_contract(repo_root=repo_root)
    payload = getattr(getattr(row_spec, "method_payload", None), "payload", None)
    if not isinstance(payload, Mapping):
        raise ForkParityError(
            f"lambda-update gate missing row={row_id!r}.method_payload.payload"
        )
    actual_update = payload.get("lambda_update")
    if not isinstance(actual_update, Mapping):
        raise ForkParityError(
            f"lambda-update gate missing row={row_id!r}.method_payload.payload.lambda_update"
        )
    expected_update = contract["lambda_update"]
    for field, expected in expected_update.items():
        actual = actual_update.get(field)
        if actual != expected:
            rendered_actual = "<missing>" if field not in actual_update else repr(actual)
            raise ForkParityError(
                "lambda-update gate mismatch "
                f"row={row_id!r} field=method_payload.payload.lambda_update.{field}: "
                f"expected={expected!r} actual={rendered_actual}"
            )

    relative_contract = contract["lambda_min_relative_to_analytical_seed"]
    seed_field = relative_contract["config_field"]
    factor = relative_contract["factor"]
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ForkParityError(f"lambda-update gate missing row={row_id!r}.method_payload.payload.config")
    seed = _finite_positive_float(
        config.get(seed_field),
        path=f"row={row_id!r}.method_payload.payload.config.{seed_field}",
    )
    lambda_min = _finite_positive_float(
        actual_update.get("lambda_min"),
        path=f"row={row_id!r}.method_payload.payload.lambda_update.lambda_min",
    )
    expected_lambda_min = factor * seed
    if not math.isclose(lambda_min, expected_lambda_min, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ForkParityError(
            "lambda-update gate mismatch "
            f"row={row_id!r} field=method_payload.payload.lambda_update.lambda_min: "
            f"expected={factor:.12g} * {seed_field}={expected_lambda_min:.12g} "
            f"actual={lambda_min:.12g}"
        )


def _load_stage2_lambda_update_contract(*, repo_root: Path) -> dict[str, Any]:
    """Load the tracked clamp prescription used by the Stage-2 fork gate."""

    path = repo_root / _STAGE2_LAMBDA_UPDATE_CONTRACT_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ForkParityError(
            f"lambda-update gate missing tracked contract {_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ForkParityError(
            f"lambda-update gate invalid JSON contract {_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ForkParityError(
            f"lambda-update gate requires object contract {_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}"
        )
    expected_update = raw.get("lambda_update")
    relative_contract = raw.get("lambda_min_relative_to_analytical_seed")
    if not isinstance(expected_update, Mapping):
        raise ForkParityError(
            "lambda-update gate missing field "
            f"{_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}.lambda_update"
        )
    if not isinstance(relative_contract, Mapping):
        raise ForkParityError(
            "lambda-update gate missing field "
            f"{_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}.lambda_min_relative_to_analytical_seed"
        )
    required_update = {
        "interval_batches",
        "ema_alpha",
        "eta",
        "max_log_step",
        "deadband_frac",
        "freeze_during_application_ramp",
    }
    missing_update = sorted(required_update - set(expected_update))
    if missing_update:
        raise ForkParityError(
            "lambda-update gate contract missing fields "
            f"{_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}.lambda_update={missing_update!r}"
        )
    config_field = relative_contract.get("config_field")
    factor = relative_contract.get("factor")
    if not isinstance(config_field, str) or not config_field:
        raise ForkParityError(
            "lambda-update gate contract requires nonempty "
            f"{_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}.lambda_min_relative_to_analytical_seed.config_field"
        )
    factor = _finite_positive_float(
        factor,
        path=(
            f"{_STAGE2_LAMBDA_UPDATE_CONTRACT_PATH}."
            "lambda_min_relative_to_analytical_seed.factor"
        ),
    )
    return {
        "lambda_update": dict(expected_update),
        "lambda_min_relative_to_analytical_seed": {
            "config_field": config_field,
            "factor": factor,
        },
    }


def _finite_positive_float(value: Any, *, path: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ForkParityError(f"lambda-update gate requires positive numeric field={path}") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ForkParityError(f"lambda-update gate requires positive numeric field={path}")
    return numeric


def _ratio_setpoint_prelaunch_report(matrix: TrainingRunMatrixSpec) -> dict[str, Any] | None:
    """Return the required R-star derivation when a continuation matrix declares one."""

    raw = matrix.metadata.get(_RATIO_SETPOINT_METADATA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ForkParityError("ratio setpoint metadata must be an object")
    required = {
        "numerator",
        "numerator_convention",
        "denominator_window",
        "baseline_final_quarter_mean_clean_loss",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ForkParityError(f"ratio setpoint metadata missing fields {missing!r}")
    numerator_convention = raw["numerator_convention"]
    denominator_window = raw["denominator_window"]
    if numerator_convention != "excess" or denominator_window != "baseline_final_quarter":
        raise ForkParityError(
            "ratio setpoint metadata requires numerator_convention='excess' and "
            "denominator_window='baseline_final_quarter'"
        )
    try:
        numerator = float(raw["numerator"])
        denominator = float(raw["baseline_final_quarter_mean_clean_loss"])
    except (TypeError, ValueError) as exc:
        raise ForkParityError("ratio setpoint numerator and denominator must be numeric") from exc
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        raise ForkParityError(
            "ratio setpoint numerator and denominator must be finite; denominator > 0"
        )
    if numerator != 1024.0:
        raise ForkParityError(
            f"ratio setpoint metadata requires excess numerator=1024; got {numerator:.12g}"
        )
    raw_setpoint = numerator / denominator
    return {
        "numerator_convention": "excess",
        "denominator_window": "baseline_final_quarter",
        "numerator": numerator,
        "baseline_final_quarter_mean_clean_loss": denominator,
        "raw_ratio_setpoint": raw_setpoint,
        "rounded_ratio_setpoint_2sf": _round_to_significant_figures(raw_setpoint, figures=2),
    }


def _round_to_significant_figures(value: float, *, figures: int) -> float:
    if value == 0.0:
        return 0.0
    return round(value, figures - 1 - int(math.floor(math.log10(abs(value)))))


def format_ratio_setpoint_report(report: dict[str, Any]) -> str:
    """Format the reviewable R-star derivation emitted by the fork gate."""

    return (
        "RATIO_SETPOINT "
        f"numerator_convention={report['numerator_convention']} "
        f"denominator_window={report['denominator_window']} "
        f"numerator={report['numerator']:.12g} "
        "baseline_final_quarter_mean_clean_loss="
        f"{report['baseline_final_quarter_mean_clean_loss']:.12g} "
        f"raw_ratio_setpoint={report['raw_ratio_setpoint']:.12g} "
        f"rounded_ratio_setpoint_2sf={report['rounded_ratio_setpoint_2sf']:.12g}"
    )


def _render_value(value: Any) -> str:
    return repr(value)

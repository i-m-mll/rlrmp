"""RLRMP adapter for Feedbax training-run matrix checkpoint forks."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
)
from feedbax.training.run_matrix import (
    ForkParityError,
    fork_matrix_checkpoints,
    materialize_run_matrix,
)

from rlrmp.runtime.lr_continuation import RlrmpLrContinuationReporter
from rlrmp.runtime.training_run_specs import (
    register_rlrmp_cs_supervised_method,
    register_rlrmp_distillation_methods,
)
from rlrmp.train.minimax_native import (
    ensure_minimax_training_method_registered,
)


_TASK_IDENTITY_METADATA_KEY = "rlrmp_task_identity"
_TASK_IDENTITY_SUBTREES = ("game_card", "perturbation_training")
_RATIO_SETPOINT_METADATA_KEY = "ratio_setpoint"


@dataclass(frozen=True)
class ForkTarget:
    """One target row checkpoint root for a matrix fork."""

    row_id: str
    checkpoint_root: Path


def register_rlrmp_training_methods() -> None:
    """Register RLRMP Feedbax training methods in this process."""

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
    materialized = materialize_run_matrix(
        matrix,
        repo_root=Path.cwd() if repo_root is None else repo_root,
        method_registry=DEFAULT_TRAINING_METHOD_REGISTRY,
    )
    _validate_fork_prelaunch_contracts(matrix, materialized)
    ratio_setpoint = _ratio_setpoint_prelaunch_report(matrix)
    target_roots = {target.row_id: target.checkpoint_root for target in targets}
    reporter = RlrmpLrContinuationReporter(source_checkpoint_root=source_checkpoint_root)
    table = fork_matrix_checkpoints(
        matrix,
        materialized,
        source_checkpoint_root=source_checkpoint_root,
        target_checkpoint_roots=target_roots,
        parity_output_path=parity_output_path,
        skip_fork=skip_fork,
        lr_reporter=reporter,
        tool_version="rlrmp.checkpoint_fork_gate.v2",
    )
    if ratio_setpoint is not None:
        table["ratio_setpoint"] = ratio_setpoint
        parity_output_path.write_text(
            json.dumps(table, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return table


def _validate_fork_prelaunch_contracts(
    matrix: TrainingRunMatrixSpec,
    materialized: Any,
) -> None:
    """Fail before any fork write when task or LR continuation contracts drift."""

    source_identity = _task_identity_from_mapping(
        matrix.metadata,
        path=f"matrix.metadata.{_TASK_IDENTITY_METADATA_KEY}",
    )
    row_metadata = {row.row_id: row.metadata for row in matrix.rows}
    for row in materialized.rows:
        target_identity = _task_identity_from_mapping(
            row_metadata.get(row.row_id),
            path=(f"row={row.row_id!r}.metadata.{_TASK_IDENTITY_METADATA_KEY}"),
        )
        _assert_matching_task_identity(
            row_id=row.row_id,
            source_identity=source_identity,
            target_identity=target_identity,
        )
        _assert_payload_lr_continuation_mode(
            row_id=row.row_id,
            row_spec=row.spec,
            declared_mode=matrix.fork.lr_continuation if matrix.fork is not None else None,
        )


def _task_identity_from_mapping(mapping: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or _TASK_IDENTITY_METADATA_KEY not in mapping:
        raise ForkParityError(f"task identity gate missing {path}")
    identity = mapping[_TASK_IDENTITY_METADATA_KEY]
    if not isinstance(identity, dict):
        raise ForkParityError(f"task identity gate requires object at {path}")
    identity_keys = set(identity)
    expected_keys = set(_TASK_IDENTITY_SUBTREES)
    if identity_keys != expected_keys:
        missing = sorted(expected_keys - identity_keys)
        extra = sorted(identity_keys - expected_keys)
        raise ForkParityError(
            f"task identity gate requires exactly {_TASK_IDENTITY_SUBTREES!r} at {path}; "
            f"missing={missing!r} extra={extra!r}"
        )
    return identity


def _assert_matching_task_identity(
    *,
    row_id: str,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
) -> None:
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
            f"task identity mismatch row={row_id!r} path={path!r}: "
            f"source={source_value} target={target_value}"
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

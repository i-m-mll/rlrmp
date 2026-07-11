"""Prepare a runnable 12,000-to-12,200 C&S seam launch fork."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from feedbax.training.checkpoint_custody import (
    fork_checkpoint_transaction,
    load_latest_checkpoint,
)

from rlrmp.runtime.checkpoint_custody import cs_custody_training_spec
from rlrmp.train.cs_nominal_gru import build_parser
from rlrmp.train.executor.cs_supervised import (
    build_cs_supervised_native_initial_slots,
    build_run_spec_execution_context,
)
from rlrmp.train.executor.slots import OPTIMIZER
from rlrmp.train.resume_control import declare_cs_supervised_checkpoint_continuation

SOURCE_COMPLETED_BATCHES = 12_000
TARGET_TOTAL_BATCHES = 12_200
HISTORY_INDICES = (1, 2, 3, 30, 31, 32)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-spec", type=Path, required=True)
    parser.add_argument("--target-run-spec", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--target-checkpoint-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.target_checkpoint_root.exists():
        raise ValueError(f"launch-fork target must be new: {args.target_checkpoint_root}")

    run_parser = build_parser()
    source_context = build_run_spec_execution_context(
        run_parser.parse_args(["--run-spec", str(args.source_run_spec)]),
        parser=run_parser,
    )
    target_context = build_run_spec_execution_context(
        run_parser.parse_args(["--run-spec", str(args.target_run_spec)]),
        parser=run_parser,
    )
    source_spec = cs_custody_training_spec(source_context.run_spec)
    source_custody_slots = _read_custody_slots(args.source_checkpoint_root)
    source_program = source_spec.worker_execution.method_contract.phase_program
    source_loaded = load_latest_checkpoint(
        args.source_checkpoint_root,
        expected_run_spec=source_spec,
        expected_phase_program=source_program,
        expected_slots=source_custody_slots,
    )
    if source_loaded.manifest.completed_training_batches != SOURCE_COMPLETED_BATCHES:
        raise ValueError(
            "launch-fork source completed_training_batches mismatch: "
            f"expected={SOURCE_COMPLETED_BATCHES} "
            f"actual={source_loaded.manifest.completed_training_batches}"
        )
    if int(source_loaded.slots["completed_batches"]) != SOURCE_COMPLETED_BATCHES:
        raise ValueError("launch-fork source completed_batches slot is not 12,000")

    target_initial_slots, _runtime = build_cs_supervised_native_initial_slots(
        run_spec=target_context.run_spec,
        hps=target_context.hps,
        args=target_context.args,
        key=jax.random.PRNGKey(int(target_context.args.seed)),
    )
    expected_slots = dict(source_custody_slots)
    expected_slots[OPTIMIZER] = _extend_optimizer_histories(
        source_custody_slots[OPTIMIZER],
        target_initial_slots[OPTIMIZER],
    )
    target_spec = declare_cs_supervised_checkpoint_continuation(
        cs_custody_training_spec(target_context.run_spec),
        source_completed_batches=SOURCE_COMPLETED_BATCHES,
        target_total_batches=TARGET_TOTAL_BATCHES,
    )
    continuation = target_spec.checkpoint_progress.continuation
    if continuation is None:
        raise ValueError("target run spec lacks the required checkpoint continuation contract")

    result = fork_checkpoint_transaction(
        args.source_checkpoint_root,
        args.target_checkpoint_root,
        target_run_spec=target_spec,
        target_phase_program=target_spec.worker_execution.method_contract.phase_program,
        expected_slots=expected_slots,
        source_slot_transforms={OPTIMIZER: _optimizer_transform(expected_slots[OPTIMIZER])},
        source_transform_metadata={
            OPTIMIZER: {
                "identity": "rlrmp.baseline_custody_launch_fork.extend_histories.v1",
                "parameters": {
                    "source_completed_batches": SOURCE_COMPLETED_BATCHES,
                    "target_total_batches": TARGET_TOTAL_BATCHES,
                    "history_indices": list(HISTORY_INDICES),
                },
            }
        },
        metadata={
            "rlrmp_launch_fork": {
                "schema_version": 1,
                "source_completed_batches": SOURCE_COMPLETED_BATCHES,
                "target_total_batches": TARGET_TOTAL_BATCHES,
                "continuation_request": continuation.model_dump(mode="json", exclude_none=True),
            }
        },
    )
    loaded = load_latest_checkpoint(
        args.target_checkpoint_root,
        expected_run_spec=target_spec,
        expected_phase_program=target_spec.worker_execution.method_contract.phase_program,
        expected_slots=expected_slots,
    )
    _validate_launch_fork(loaded)
    print(
        json.dumps(
            {
                "transaction_id": result.manifest.transaction_id,
                "completed_training_batches": loaded.manifest.completed_training_batches,
                "target_total_batches": TARGET_TOTAL_BATCHES,
                "optimizer_history_horizon": int(loaded.slots[OPTIMIZER][1].shape[-1]),
                "program_step": loaded.manifest.completed_coordinate.program_step,
                "barrier_visit_ordinal": loaded.manifest.metadata.get("barrier_visit_ordinal"),
            },
            sort_keys=True,
        )
    )


def _read_custody_slots(checkpoint_root: Path) -> dict[str, object]:
    latest = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoint_root / latest["manifest_relative_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        slot["slot"]: pickle.loads((manifest_path.parent / slot["relative_path"]).read_bytes())
        for slot in manifest["slots"]
    }


def _extend_optimizer_histories(source_optimizer: object, target_optimizer: object) -> tuple:
    if not isinstance(source_optimizer, tuple) or not isinstance(target_optimizer, tuple):
        raise TypeError("C&S optimizer slots must be native tuple PyTrees")
    extended = list(source_optimizer)
    for index in HISTORY_INDICES:
        source = jnp.asarray(source_optimizer[index])
        target = jnp.asarray(target_optimizer[index])
        if source.shape[-1] != SOURCE_COMPLETED_BATCHES:
            raise ValueError(f"optimizer history {index} source horizon is not 12,000")
        if target.shape[-1] != TARGET_TOTAL_BATCHES:
            raise ValueError(f"optimizer history {index} target template horizon is not 12,200")
        if source.shape[:-1] != target.shape[:-1] or source.dtype != target.dtype:
            raise ValueError(f"optimizer history {index} target template ABI mismatch")
        extended[index] = jnp.concatenate(
            (source, target[..., SOURCE_COMPLETED_BATCHES:]),
            axis=-1,
        )
    return tuple(extended)


def _optimizer_transform(target_optimizer: object):
    def transform(slots: dict[str, object]) -> dict[str, object]:
        updated = dict(slots)
        updated[OPTIMIZER] = target_optimizer
        return updated

    return transform


def _validate_launch_fork(loaded: object) -> None:
    manifest = loaded.manifest
    slots = loaded.slots
    if manifest.completed_training_batches != SOURCE_COMPLETED_BATCHES:
        raise ValueError("launch fork must preserve completed_training_batches=12,000")
    if int(slots["completed_batches"]) != SOURCE_COMPLETED_BATCHES:
        raise ValueError("launch fork must preserve completed_batches slot=12,000")
    optimizer = slots[OPTIMIZER]
    if not isinstance(optimizer, tuple):
        raise TypeError("launch-fork optimizer slot must be a native tuple PyTree")
    for index in HISTORY_INDICES:
        if int(optimizer[index].shape[-1]) != TARGET_TOTAL_BATCHES:
            raise ValueError(f"launch-fork optimizer history {index} is not 12,200")
    marker = manifest.metadata.get("rlrmp_launch_fork", {})
    if marker.get("source_completed_batches") != SOURCE_COMPLETED_BATCHES:
        raise ValueError("launch-fork provenance lacks the source completed count")
    if marker.get("target_total_batches") != TARGET_TOTAL_BATCHES:
        raise ValueError("launch-fork provenance lacks the target total")


if __name__ == "__main__":
    main()

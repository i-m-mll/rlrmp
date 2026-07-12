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
from rlrmp.train.executor.cs_supervised import (
    build_execution_context_from_spec,
    build_cs_supervised_native_initial_slots,
)
from rlrmp.train.executor.slots import OPTIMIZER
from rlrmp.train.resume_control import declare_cs_supervised_checkpoint_continuation

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

    source_context = build_execution_context_from_spec(args.source_run_spec)
    target_context = build_execution_context_from_spec(args.target_run_spec)
    source_spec = cs_custody_training_spec(source_context.run_spec)
    source_custody_slots = _read_custody_slots(args.source_checkpoint_root)
    source_program = source_spec.worker_execution.method_contract.phase_program
    source_loaded = load_latest_checkpoint(
        args.source_checkpoint_root,
        expected_run_spec=source_spec,
        expected_phase_program=source_program,
        expected_slots=source_custody_slots,
    )
    source_completed_batches = int(source_loaded.manifest.completed_training_batches)
    if int(source_loaded.slots["completed_batches"]) != source_completed_batches:
        raise ValueError("launch-fork source progress slot disagrees with its manifest")

    target_initial_slots, _runtime = build_cs_supervised_native_initial_slots(
        run_spec=target_context.run_spec,
        hps=target_context.hps,
        args=target_context.args,
        key=jax.random.PRNGKey(int(target_context.args.seed)),
    )
    target_total_batches = int(target_context.args.n_train_batches)
    history_indices = _history_indices(
        source_custody_slots[OPTIMIZER],
        target_initial_slots[OPTIMIZER],
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
    )
    expected_slots = dict(source_custody_slots)
    expected_slots[OPTIMIZER] = _extend_optimizer_histories(
        source_custody_slots[OPTIMIZER],
        target_initial_slots[OPTIMIZER],
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
    )
    target_spec = declare_cs_supervised_checkpoint_continuation(
        cs_custody_training_spec(target_context.run_spec),
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
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
                    "source_completed_batches": source_completed_batches,
                    "target_total_batches": target_total_batches,
                    "history_indices": list(history_indices),
                },
            }
        },
        metadata={
            "rlrmp_launch_fork": {
                "schema_version": 1,
                "source_completed_batches": source_completed_batches,
                "target_total_batches": target_total_batches,
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
    _validate_launch_fork(
        loaded,
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
        history_indices=history_indices,
    )
    print(
        json.dumps(
            {
                "transaction_id": result.manifest.transaction_id,
                "completed_training_batches": loaded.manifest.completed_training_batches,
                "target_total_batches": target_total_batches,
                "optimizer_history_horizon": int(
                    loaded.slots[OPTIMIZER][history_indices[0]].shape[-1]
                ),
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


def _extend_optimizer_histories(
    source_optimizer: object,
    target_optimizer: object,
    *,
    source_completed_batches: int,
    target_total_batches: int,
) -> tuple:
    if not isinstance(source_optimizer, tuple) or not isinstance(target_optimizer, tuple):
        raise TypeError("C&S optimizer slots must be native tuple PyTrees")
    if len(source_optimizer) != len(target_optimizer):
        raise ValueError("C&S optimizer source and target tuple lengths differ")
    history_indices = _history_indices(
        source_optimizer,
        target_optimizer,
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
    )
    if not history_indices:
        raise ValueError("C&S optimizer has no compatible batch-history leaves")
    extended = list(source_optimizer)
    for index in history_indices:
        source = jnp.asarray(source_optimizer[index])
        target = jnp.asarray(target_optimizer[index])
        extended[index] = jnp.concatenate(
            (source, target[..., source.shape[-1] :]),
            axis=-1,
        )
    return tuple(extended)


def _history_indices(
    source_optimizer: object,
    target_optimizer: object,
    *,
    source_completed_batches: int,
    target_total_batches: int,
) -> tuple[int, ...]:
    if not isinstance(source_optimizer, tuple) or not isinstance(target_optimizer, tuple):
        raise TypeError("C&S optimizer slots must be native tuple PyTrees")
    indices = []
    for index, (source_leaf, target_leaf) in enumerate(zip(source_optimizer, target_optimizer)):
        source = jnp.asarray(source_leaf)
        target = jnp.asarray(target_leaf)
        if (
            source.ndim > 0
            and target.ndim == source.ndim
            and target.shape[:-1] == source.shape[:-1]
            and source.shape[-1] == source_completed_batches
            and target.shape[-1] == target_total_batches
            and target.dtype == source.dtype
        ):
            indices.append(index)
    return tuple(indices)


def _optimizer_transform(target_optimizer: object):
    def transform(slots: dict[str, object]) -> dict[str, object]:
        updated = dict(slots)
        updated[OPTIMIZER] = target_optimizer
        return updated

    return transform


def _validate_launch_fork(
    loaded: object,
    *,
    source_completed_batches: int,
    target_total_batches: int,
    history_indices: tuple[int, ...],
) -> None:
    manifest = loaded.manifest
    slots = loaded.slots
    if manifest.completed_training_batches != source_completed_batches:
        raise ValueError("launch fork must preserve source completed_training_batches")
    if int(slots["completed_batches"]) != source_completed_batches:
        raise ValueError("launch fork must preserve the source completed_batches slot")
    optimizer = slots[OPTIMIZER]
    if not isinstance(optimizer, tuple):
        raise TypeError("launch-fork optimizer slot must be a native tuple PyTree")
    for index in history_indices:
        if int(optimizer[index].shape[-1]) != target_total_batches:
            raise ValueError(f"launch-fork optimizer history {index} has the wrong horizon")
    marker = manifest.metadata.get("rlrmp_launch_fork", {})
    if marker.get("source_completed_batches") != source_completed_batches:
        raise ValueError("launch-fork provenance lacks the source completed count")
    if marker.get("target_total_batches") != target_total_batches:
        raise ValueError("launch-fork provenance lacks the target total")


if __name__ == "__main__":
    main()

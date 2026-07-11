"""Prove a repaired C&S custody source can fork from 12,000 to 12,200 batches."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--target-checkpoint-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.target_checkpoint_root.exists():
        raise ValueError(f"fork target must be new: {args.target_checkpoint_root}")
    run_parser = build_parser()
    context = build_run_spec_execution_context(
        run_parser.parse_args(["--run-spec", str(args.run_spec)]),
        parser=run_parser,
    )
    source_spec = cs_custody_training_spec(context.run_spec)
    source_slots, _runtime = build_cs_supervised_native_initial_slots(
        run_spec=context.run_spec,
        hps=context.hps,
        args=context.args,
        key=jax.random.PRNGKey(int(context.args.seed)),
    )
    source_custody_slots = _read_custody_slots(args.source_checkpoint_root)
    source_program = source_spec.worker_execution.method_contract.phase_program
    load_latest_checkpoint(
        args.source_checkpoint_root,
        expected_run_spec=source_spec,
        expected_phase_program=source_program,
        expected_slots=source_custody_slots,
    )
    expected_slots = dict(source_custody_slots)
    optimizer = source_custody_slots[OPTIMIZER]
    if not isinstance(optimizer, tuple):
        raise TypeError("C&S source optimizer slot must be a native tuple PyTree")
    target_optimizer = list(optimizer)
    for index in (1, 2, 3, 30, 31, 32):
        values = jnp.asarray(target_optimizer[index])
        target_optimizer[index] = jnp.pad(values, ((0, 0), (0, 200)))
    expected_slots[OPTIMIZER] = tuple(target_optimizer)
    expected_slots["completed_batches"] = jnp.asarray(12_200, dtype=jnp.int32)
    target_spec = declare_cs_supervised_checkpoint_continuation(
        source_spec,
        source_completed_batches=12_000,
        target_total_batches=12_200,
    )
    program = target_spec.worker_execution.method_contract.phase_program
    result = fork_checkpoint_transaction(
        args.source_checkpoint_root,
        args.target_checkpoint_root,
        target_run_spec=target_spec,
        target_phase_program=program,
        expected_slots=expected_slots,
        slot_transforms={"completed_batches": _set_target_completed_batches},
        transform_metadata={
            "completed_batches": {
                "identity": "rlrmp.baseline_custody_fork.set_completed_batches.v1",
                "parameters": {"source": 12_000, "target": 12_200},
            }
        },
        continuation_request=target_spec.checkpoint_progress.continuation,
    )
    loaded = load_latest_checkpoint(
        args.target_checkpoint_root,
        expected_run_spec=target_spec,
        expected_phase_program=program,
        expected_slots=expected_slots,
        continuation_request=target_spec.checkpoint_progress.continuation,
    )
    optimizer = loaded.slots[OPTIMIZER]
    if not isinstance(optimizer, tuple):
        raise TypeError("C&S custody optimizer slot must be a native tuple PyTree")
    print(
        json.dumps(
            {
                "transaction_id": result.manifest.transaction_id,
                "source_completed_training_batches": 12000,
                "target_completed_training_batches": loaded.manifest.completed_training_batches,
                "optimizer_history_horizon": int(optimizer[1].shape[-1]),
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


def _set_target_completed_batches(slots: dict[str, object]) -> dict[str, object]:
    updated = dict(slots)
    updated["completed_batches"] = jnp.asarray(12_200, dtype=jnp.int32)
    return updated


if __name__ == "__main__":
    main()

"""Prepare runnable batch-12,000 Stage-2 launch forks from a matrix spec."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
from feedbax.contracts.checkpoints import CheckpointForkBarrierMapping
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.training.checkpoint_custody import fork_checkpoint_transaction, load_latest_checkpoint

from rlrmp.runtime.checkpoint_custody import cs_custody_training_spec
from rlrmp.runtime.adaptive_checkpoint_adapter import NominalToAdaptiveSlotAdapter
from rlrmp.runtime.training_run_specs import feedbax_training_run_spec_from_payload
from rlrmp.train.adaptive_epsilon_native import (
    AdaptiveEpsilonNativeRuntime,
    _adaptive_state_from_slot,
    build_adaptive_epsilon_native_initial_slots,
)
from rlrmp.train.cs_nominal_gru import build_parser
from rlrmp.train.executor.cs_supervised import (
    _cs_supervised_native_run_id,
    build_run_spec_execution_context,
)
from rlrmp.train.executor.slots import OPTIMIZER
from rlrmp.train.resume_control import declare_cs_supervised_checkpoint_continuation

SOURCE_COMPLETED_BATCHES = 12_000
TARGET_TOTAL_BATCHES = 16_500
HISTORY_INDICES = (1, 2, 3, 30, 31, 32)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--source-run-spec", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    rows = matrix.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("matrix rows must be a non-empty list")
    run_parser = build_parser()
    source_context = build_run_spec_execution_context(
        run_parser.parse_args(["--run-spec", str(args.source_run_spec)]), parser=run_parser
    )
    source_spec = cs_custody_training_spec(source_context.run_spec)
    source_slots = _read_custody_slots(args.source_checkpoint_root)
    source_loaded = load_latest_checkpoint(
        args.source_checkpoint_root,
        expected_run_spec=source_spec,
        expected_phase_program=source_spec.worker_execution.method_contract.phase_program,
        expected_slots=source_slots,
    )
    _validate_source(source_loaded)

    reports = []
    for row in rows:
        row_id = row["row_id"]
        run_spec_path = args.matrix.parent / f"{row_id}.json"
        target_root = args.target_root / row_id / "checkpoints"
        if target_root.exists():
            if not args.replace:
                raise ValueError(f"launch-fork target must be new: {target_root}")
            shutil.rmtree(target_root)
        target_context = build_run_spec_execution_context(
            run_parser.parse_args(["--run-spec", str(run_spec_path)]), parser=run_parser
        )
        target_spec = feedbax_training_run_spec_from_payload(target_context.run_spec)
        target_initial_slots, runtime = build_adaptive_epsilon_native_initial_slots(
            run_spec=target_spec,
            hps=target_context.hps,
            args=target_context.args,
            key=jax.random.PRNGKey(int(target_context.args.seed)),
            schedule_start_batch=SOURCE_COMPLETED_BATCHES,
        )
        native = runtime.component("adaptive_epsilon")
        if not isinstance(native, AdaptiveEpsilonNativeRuntime):
            raise TypeError(f"adaptive runtime missing for Stage-2 row {row_id!r}")
        adapter = NominalToAdaptiveSlotAdapter(
            model_template=native.model_template,
            optimizer_template=native.optimizer_template,
            adaptive_initial_slots=target_initial_slots,
        )
        target_spec = declare_cs_supervised_checkpoint_continuation(
            target_spec,
            source_completed_batches=SOURCE_COMPLETED_BATCHES,
            target_total_batches=TARGET_TOTAL_BATCHES,
        )
        target_barrier = "after_adaptive_epsilon_train_chunk"
        barrier_mapping = CheckpointForkBarrierMapping(
            source_barrier="after_train_chunk",
            target_barrier=target_barrier,
            target_coordinate=ProgressCoordinate(
                run_id=_cs_supervised_native_run_id(target_context.args, run_spec_path),
                phase="adaptive_epsilon_train_chunk",
                program_step=source_loaded.manifest.completed_coordinate.program_step,
                completed_barrier=target_barrier,
            ),
            coordinate_mapping={"identity": "rlrmp.cs_supervised_to_adaptive_epsilon.v1",
                                "parameters": {"program_step":
                                               "preserve_completed_training_batches"}},
        )
        result = fork_checkpoint_transaction(
            args.source_checkpoint_root,
            target_root,
            target_run_spec=target_spec,
            target_phase_program=target_spec.worker_execution.method_contract.phase_program,
            expected_slots=target_initial_slots,
            barrier_mapping=barrier_mapping,
            continuation_slot_templates=adapter.continuation_slot_templates(),
            continuation_request=target_spec.checkpoint_progress.continuation,
            target_slot_transform=adapter.transform,
            target_transform_metadata=adapter.transform_metadata,
            target_transformed_slots=adapter.target_transformed_slots,
            target_only_slots=adapter.target_only_slots,
            metadata={"rlrmp_stage2_launch_fork": {
                "schema_version": 1, "matrix_row_id": row_id,
                "source_completed_batches": SOURCE_COMPLETED_BATCHES,
                "target_total_batches": TARGET_TOTAL_BATCHES,
            }},
        )
        loaded = load_latest_checkpoint(
            target_root,
            expected_run_spec=target_spec,
            expected_phase_program=target_spec.worker_execution.method_contract.phase_program,
            expected_slots=target_initial_slots,
        )
        validate_launch_fork(loaded, row_id=row_id)
        reports.append({"row_id": row_id, "transaction_id": result.manifest.transaction_id,
                        "completed_training_batches": loaded.manifest.completed_training_batches,
                        "optimizer_history_horizon": TARGET_TOTAL_BATCHES})
    print(json.dumps({"rows": reports}, sort_keys=True))


def _read_custody_slots(checkpoint_root: Path) -> dict[str, object]:
    latest = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoint_root / latest["manifest_relative_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {slot["slot"]: pickle.loads((manifest_path.parent / slot["relative_path"]).read_bytes())
            for slot in manifest["slots"]}


def _validate_source(loaded: object) -> None:
    if loaded.manifest.completed_training_batches != SOURCE_COMPLETED_BATCHES:
        raise ValueError("Stage-2 launch source manifest is not at batch 12,000")
    if int(loaded.slots["completed_batches"]) != SOURCE_COMPLETED_BATCHES:
        raise ValueError("Stage-2 launch source completed_batches slot is not 12,000")


def extend_optimizer_histories(source_optimizer: object, target_optimizer: object) -> tuple:
    if not isinstance(source_optimizer, tuple) or not isinstance(target_optimizer, tuple):
        raise TypeError("C&S optimizer slots must be native tuple PyTrees")
    extended = list(source_optimizer)
    for index in HISTORY_INDICES:
        source = jnp.asarray(source_optimizer[index])
        target = jnp.asarray(target_optimizer[index])
        if source.shape[-1] != SOURCE_COMPLETED_BATCHES:
            raise ValueError(f"optimizer history {index} source horizon is not 12,000")
        if target.shape[-1] != TARGET_TOTAL_BATCHES:
            raise ValueError(f"optimizer history {index} target template horizon is not 16,500")
        if source.shape[:-1] != target.shape[:-1] or source.dtype != target.dtype:
            raise ValueError(f"optimizer history {index} target template ABI mismatch")
        extended[index] = jnp.concatenate((source, target[..., SOURCE_COMPLETED_BATCHES:]), axis=-1)
    return tuple(extended)


def validate_launch_fork(loaded: object, *, row_id: str) -> None:
    if loaded.manifest.completed_training_batches != TARGET_TOTAL_BATCHES:
        raise ValueError("launch-fork manifest is not bound to the target total")
    if int(loaded.slots["completed_batches"]) != SOURCE_COMPLETED_BATCHES:
        raise ValueError("launch-fork runtime progress slot is not at the source total")
    adaptive_state = _adaptive_state_from_slot(loaded.slots["adaptive_epsilon_state"])
    if adaptive_state is None:
        raise ValueError("launch-fork adaptive state is missing")
    if adaptive_state.schedule_start_batch != SOURCE_COMPLETED_BATCHES:
        raise ValueError("launch-fork adaptive schedule does not start at the source total")
    optimizer = loaded.slots[OPTIMIZER]
    if not hasattr(optimizer, "payload"):
        raise TypeError("launch-fork optimizer is not an adaptive serialized slot")
    marker = loaded.manifest.metadata.get("rlrmp_stage2_launch_fork", {})
    if marker.get("matrix_row_id") != row_id:
        raise ValueError("launch-fork provenance has the wrong matrix row")
    if marker.get("source_completed_batches") != SOURCE_COMPLETED_BATCHES:
        raise ValueError("launch-fork provenance lacks the source completed count")
    if marker.get("target_total_batches") != TARGET_TOTAL_BATCHES:
        raise ValueError("launch-fork provenance lacks the target total")


if __name__ == "__main__":
    main()

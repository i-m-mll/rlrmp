"""RLRMP adoption tests for Feedbax checkpoint custody."""

from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from feedbax.contracts.training import (
    GraphTopologySourceSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
)

from rlrmp.runtime.checkpoint_custody import load_cs_checkpoint_transaction
from rlrmp.runtime.training_run_specs import FEEDBAX_TRAINING_RUN_SPEC_KEY
from rlrmp.train.cs_nominal_gru import (
    TrainingState,
    load_latest_checkpoint,
    save_training_checkpoint,
)
from rlrmp.train.minimax_native import (
    MINIMAX_METHOD_PAYLOAD_SCHEMA_ID,
    MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
    MinimaxConfig,
    MinimaxMethodPayload,
    minimax_effective_phase_spec,
    minimax_method_contract,
)


def _minimal_graph() -> dict[str, object]:
    return {
        "schema_id": "feedbax.spec.graph",
        "schema_version": "feedbax.spec.graph.v4",
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _standard_training_spec() -> TrainingRunSpec:
    return TrainingRunSpec(
        graph=GraphTopologySourceSpec(inline=_minimal_graph()),
        task=TaskSpec(type="toy", params={"n_steps": 2}),
        training_config=TrainingConfig(n_batches=4, batch_size=2),
        objective=ObjectiveSlotSpec(kind="external", payload={"loss": "toy"}),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        method_extensions={
            "metadata": {
                "runner": "rlrmp.tests.checkpoint_custody_adoption",
                "rlrmp_training_mode": "standard",
                "rlrmp_loss_objective": "test_objective",
                "adversarial_phase": "none",
            }
        },
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )


def _cs_run_spec() -> dict[str, object]:
    return {
        "schema_version": "rlrmp.test",
        "issue": "799fcb9",
        "training_summary": {
            "training_mode": "standard",
            "n_train_batches": 4,
            "batch_size": 2,
            "controller_lr": 0.001,
            "lr_schedule": "constant",
            "gradient_clip_norm": None,
        },
        "checkpointing": {"interval_batches": 2},
        "optimizer": {"name": "adam", "learning_rate": 0.001},
        "training_diagnostics": {},
        "hps": {},
        FEEDBAX_TRAINING_RUN_SPEC_KEY: _standard_training_spec().model_dump(
            mode="json",
            exclude_none=True,
        ),
    }


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        issue="799fcb9",
        n_train_batches=4,
        checkpoint_interval_batches=2,
        seed=11,
        stochastic_preset="test",
    )


def _minimax_training_spec() -> TrainingRunSpec:
    contract = minimax_method_contract()
    return TrainingRunSpec(
        graph=GraphTopologySourceSpec(inline=_minimal_graph()),
        task=TaskSpec(type="toy_minimax", params={"n_steps": 2}),
        training_config=TrainingConfig(n_batches=4, batch_size=2),
        objective=ObjectiveSlotSpec(kind="external", payload={"loss": "minimax"}),
        method_ref=MethodRefSpec(package="rlrmp", name="minimax", version="v1"),
        method_payload=MethodPayloadEnvelope(
            schema_id=MINIMAX_METHOD_PAYLOAD_SCHEMA_ID,
            schema_version=MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload=MinimaxMethodPayload(config=MinimaxConfig()).model_dump(mode="json"),
        ),
        worker_execution=WorkerExecutionSpec(
            method_contract=contract,
            effective_phase=minimax_effective_phase_spec(contract),
        ),
    )


def test_missing_custody_checkpoint_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(CheckpointIntegrityError, match="latest pointer is missing"):
        load_cs_checkpoint_transaction(
            tmp_path,
            run_spec=_cs_run_spec(),
            expected_slots={
                "model": jnp.asarray([1.0]),
                "optimizer": jnp.asarray([0.0]),
                "prng": jnp.asarray([0, 0], dtype=jnp.uint32),
                "completed_batches": jnp.asarray(0, dtype=jnp.int32),
            },
        )


def test_cs_checkpoint_custody_roundtrips_and_materializes_legacy_latest(
    tmp_path: Path,
) -> None:
    state = TrainingState(
        model=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        optimizer_state={"count": jnp.asarray(2, dtype=jnp.int32)},
        completed_batches=2,
        key=jnp.asarray([3, 4], dtype=jnp.uint32),
        history={"loss": jnp.asarray([0.5, 0.25], dtype=jnp.float32)},
    )
    checkpoint_root = tmp_path / "checkpoints"
    run_spec = _cs_run_spec()

    materialized = save_training_checkpoint(
        checkpoint_root,
        state,
        args=_args(),
        run_spec=run_spec,
    )
    loaded = load_latest_checkpoint(
        checkpoint_root,
        model_template=state.model,
        optimizer_state_template=state.optimizer_state,
        run_spec=run_spec,
    )

    assert (checkpoint_root / "latest.json").is_file()
    assert (checkpoint_root / "transactions").is_dir()
    assert materialized == checkpoint_root / "checkpoint_0000002"
    assert (checkpoint_root / "checkpoint_latest" / "model.eqx").is_file()
    provenance = json.loads((materialized / "provenance.json").read_text(encoding="utf-8"))
    latest_pointer = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == "rlrmp.legacy_checkpoint_provenance.v1"
    assert provenance["issue"] == "799fcb9"
    assert provenance["source_transaction_id"] == latest_pointer["transaction_id"]
    assert provenance["writer"] == (
        "rlrmp.train.cs_nominal_gru._save_training_checkpoint_materialization"
    )
    assert provenance["authoritative"] is False
    assert loaded.completed_batches == 2
    manifest = json.loads(
        (checkpoint_root / latest_pointer["manifest_relative_path"]).read_text(encoding="utf-8")
    )
    assert manifest["completed_coordinate"]["program_step"] == 1
    assert loaded.model.tolist() == [1.0, 2.0]
    assert loaded.optimizer_state["count"].tolist() == 2
    assert loaded.key.tolist() == [3, 4]
    assert loaded.history["loss"].tolist() == [0.5, 0.25]


def test_cs_checkpoint_materialization_clears_foreign_legacy_dirs(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    stale = checkpoint_root / "checkpoint_0000001"
    stale.mkdir(parents=True)
    (stale / "metadata.json").write_text(
        json.dumps({"issue": "foreign", "completed_batches": 1}),
        encoding="utf-8",
    )
    (stale / "model.eqx").write_text("foreign", encoding="utf-8")
    (checkpoint_root / "checkpoint_latest").symlink_to(stale.name)
    state = TrainingState(
        model=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        optimizer_state={"count": jnp.asarray(2, dtype=jnp.int32)},
        completed_batches=2,
        key=jnp.asarray([3, 4], dtype=jnp.uint32),
        history={"loss": jnp.asarray([0.5, 0.25], dtype=jnp.float32)},
    )

    materialized = save_training_checkpoint(
        checkpoint_root,
        state,
        args=_args(),
        run_spec=_cs_run_spec(),
    )

    assert not stale.exists()
    assert materialized == checkpoint_root / "checkpoint_0000002"
    assert (checkpoint_root / "checkpoint_latest").resolve() == materialized
    assert (
        not (checkpoint_root / "checkpoint_latest" / "metadata.json")
        .read_text(encoding="utf-8")
        .count("foreign")
    )


def test_cs_terminal_checkpoint_is_final_custody_transaction(tmp_path: Path) -> None:
    state = TrainingState(
        model=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        optimizer_state={"count": jnp.asarray(4, dtype=jnp.int32)},
        completed_batches=4,
        key=jnp.asarray([5, 6], dtype=jnp.uint32),
        history=None,
    )
    run_spec = _cs_run_spec()

    materialized = save_training_checkpoint(
        tmp_path,
        state,
        args=_args(),
        run_spec=run_spec,
    )
    provenance = json.loads((materialized / "provenance.json").read_text(encoding="utf-8"))
    latest_pointer = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / latest_pointer["manifest_relative_path"]).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "final"
    assert manifest["completed_coordinate"]["program_step"] == 2
    assert manifest["transaction_id"] == latest_pointer["transaction_id"]
    assert provenance["source_transaction_id"] == manifest["transaction_id"]


def test_cs_checkpoint_incompatible_model_abi_fails_closed(tmp_path: Path) -> None:
    state = TrainingState(
        model=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        optimizer_state=jnp.asarray([0.0], dtype=jnp.float32),
        completed_batches=2,
        key=jnp.asarray([3, 4], dtype=jnp.uint32),
        history=None,
    )
    run_spec = _cs_run_spec()
    save_training_checkpoint(tmp_path, state, args=_args(), run_spec=run_spec)

    with pytest.raises(CheckpointCompatibilityError, match="structural ABI mismatch"):
        load_latest_checkpoint(
            tmp_path,
            model_template=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
            optimizer_state_template=state.optimizer_state,
            run_spec=run_spec,
        )

"""RLRMP adoption tests for Feedbax checkpoint custody."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from feedbax.contracts.training import (
    GraphTopologySourceSpec,
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

from rlrmp.runtime.checkpoint_custody import (
    MINIMAX_WARMUP_BARRIER,
    load_cs_checkpoint_transaction,
    load_minimax_checkpoint_transaction,
)
from rlrmp.runtime.training_run_specs import FEEDBAX_TRAINING_RUN_SPEC_KEY
from rlrmp.train.cs_nominal_gru import (
    TrainingState,
    load_latest_checkpoint,
    save_training_checkpoint,
)
from rlrmp.train.minimax import (
    MINIMAX_CONFIG_DEFAULTS,
    _minimax_method_payload,
    minimax_effective_phase_spec,
    minimax_method_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_minimax_module():
    spec = importlib.util.spec_from_file_location(
        "train_minimax_checkpoint_custody_under_test",
        REPO_ROOT / "scripts" / "train_minimax.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_graph() -> dict[str, object]:
    return {
        "schema_id": "feedbax.spec.graph",
        "schema_version": "feedbax.spec.graph.v1",
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
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )


def _cs_run_spec() -> dict[str, object]:
    return {
        "schema_version": "rlrmp.test",
        "issue": "799fcb9",
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
        method_payload=_minimax_method_payload(
            dict(MINIMAX_CONFIG_DEFAULTS),
            output_dir=Path("bulk"),
            spec_dir=Path("spec"),
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
    assert loaded.completed_batches == 2
    assert loaded.model.tolist() == [1.0, 2.0]
    assert loaded.optimizer_state["count"].tolist() == 2
    assert loaded.key.tolist() == [3, 4]
    assert loaded.history["loss"].tolist() == [0.5, 0.25]


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

    with pytest.raises(CheckpointCompatibilityError, match="resume template"):
        load_latest_checkpoint(
            tmp_path,
            model_template=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
            optimizer_state_template=state.optimizer_state,
            run_spec=run_spec,
        )


def test_minimax_adversarial_checkpoint_custody_restores_rng_and_histories(
    tmp_path: Path,
) -> None:
    train_minimax = _load_train_minimax_module()
    training_spec = _minimax_training_spec()
    model = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    flat_model, treedef = jtu.tree_flatten(model)
    adversaries = [jnp.asarray([0.1, 0.2], dtype=jnp.float32)]
    adv_opt_states = [{"count": jnp.asarray(1, dtype=jnp.int32)}]
    ctrl_opt_state = {"count": jnp.asarray(2, dtype=jnp.int32)}

    train_minimax._save_adversarial_checkpoint(
        tmp_path,
        flat_model,
        treedef,
        adversaries,
        adv_opt_states,
        ctrl_opt_state,
        1,
        [0.9, 0.8],
        [1.9, 1.8],
        [0, 0],
        training_spec=training_spec,
        rng_key=jnp.asarray([5, 6], dtype=jnp.uint32),
    )
    loaded = train_minimax._load_adversarial_checkpoint(
        tmp_path,
        model,
        adversaries,
        adv_opt_states,
        ctrl_opt_state,
        treedef,
        training_spec=training_spec,
    )

    assert len(loaded) == 9
    assert loaded[4] == 1
    assert loaded[5] == [0.9, 0.8]
    assert loaded[6] == [1.9, 1.8]
    assert loaded[7] == [0, 0]
    assert loaded[8].tolist() == [5, 6]
    assert (tmp_path / "latest.json").is_file()
    assert (tmp_path / "checkpoint_latest" / "meta.json").is_file()


def test_minimax_warmup_boundary_checkpoint_resumes_at_adversarial_start(
    tmp_path: Path,
) -> None:
    train_minimax = _load_train_minimax_module()
    training_spec = _minimax_training_spec()
    model = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    adversaries = [jnp.asarray([0.1, 0.2], dtype=jnp.float32)]
    adv_opt_states = [{"count": jnp.asarray(1, dtype=jnp.int32)}]
    ctrl_opt_state = {"count": jnp.asarray(2, dtype=jnp.int32)}

    train_minimax._write_warmup_boundary_checkpoint(
        tmp_path,
        training_spec=training_spec,
        model=model,
        adversaries=adversaries,
        adv_opt_states=adv_opt_states,
        ctrl_opt_state=ctrl_opt_state,
        rng_key=jnp.asarray([7, 8], dtype=jnp.uint32),
        warmup_history={"loss": jnp.asarray([1.0], dtype=jnp.float32)},
    )
    loaded = load_minimax_checkpoint_transaction(
        tmp_path,
        training_spec=training_spec,
        expected_slots=train_minimax._minimax_expected_slots(
            model_template=model,
            adversaries_template=adversaries,
            adv_opt_states_template=adv_opt_states,
            ctrl_opt_state_template=ctrl_opt_state,
        ),
        expected_population_member_ids={"adversary_population": ["adversary_0"]},
    )

    assert loaded.manifest.barrier == MINIMAX_WARMUP_BARRIER
    assert int(loaded.slots["active_batch_index"]) == -1
    assert loaded.slots["rng"].tolist() == [7, 8]

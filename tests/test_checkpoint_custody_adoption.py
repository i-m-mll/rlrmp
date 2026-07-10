"""RLRMP adoption tests for Feedbax checkpoint custody."""

from __future__ import annotations
from rlrmp.io import load_named_python_module

import importlib.util
import json
import re
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
    MINIMAX_ADVERSARIAL_BARRIER,
    MINIMAX_WARMUP_BARRIER,
    has_custody_checkpoint,
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
    MinimaxConfig,
    _minimax_method_payload,
    minimax_effective_phase_spec,
    minimax_method_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_minimax_module():
    return load_named_python_module('train_minimax_checkpoint_custody_under_test', REPO_ROOT / 'scripts' / 'train_minimax.py')


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
        method_payload=_minimax_method_payload(
            MinimaxConfig().model_dump(mode="python"),
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
    assert not (checkpoint_root / "checkpoint_latest" / "metadata.json").read_text(
        encoding="utf-8"
    ).count("foreign")


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


def test_minimax_final_custody_transaction_is_content_addressed(tmp_path: Path) -> None:
    """The terminal minimax outputs resolve through the custody run record.

    Issue 7e71950: the final controller / adversary population / loss curves are
    routed onto a status="final" custody transaction that content-addresses each
    slot and publishes the latest pointer, rather than raw fbx_save/np.savez
    writes to output_dir.
    """
    train_minimax = _load_train_minimax_module()
    training_spec = _minimax_training_spec()
    model = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    adversaries = [jnp.asarray([0.1, 0.2], dtype=jnp.float32)]
    adv_opt_states = [{"count": jnp.asarray(1, dtype=jnp.int32)}]
    ctrl_opt_state = {"count": jnp.asarray(2, dtype=jnp.int32)}

    train_minimax._write_final_minimax_custody_transaction(
        tmp_path,
        training_spec=training_spec,
        model=model,
        adversaries=adversaries,
        adv_opt_states=adv_opt_states,
        ctrl_opt_state=ctrl_opt_state,
        rng_key=jnp.asarray([9, 10], dtype=jnp.uint32),
        batch_idx=3,
        adv_losses=[0.9, 0.8],
        ctrl_losses=[1.9, 1.8],
        adv_indices=[0, 0],
        warmup_history={"loss": jnp.asarray([1.0], dtype=jnp.float32)},
    )

    assert has_custody_checkpoint(tmp_path)
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

    assert loaded.manifest.status == "final"
    assert loaded.manifest.barrier == MINIMAX_ADVERSARIAL_BARRIER
    assert int(loaded.slots["active_batch_index"]) == 3
    assert loaded.slots["rng"].tolist() == [9, 10]
    # Every persisted slot is content-addressed (sha256) in the manifest.
    assert loaded.manifest.slots
    for slot in loaded.manifest.slots:
        assert re.fullmatch(r"[0-9a-f]{64}", slot.sha256)

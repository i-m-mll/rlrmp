from __future__ import annotations

import json
from pathlib import Path
import inspect

from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.training import (
    LossTermSpec,
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

from rlrmp.runtime.checkpoint_fork_gate import fork_checkpoints_with_parity, parse_target
from rlrmp.runtime.lr_continuation import RlrmpLrContinuationReporter


def _training_run_payload() -> dict[str, object]:
    spec = TrainingRunSpec(
        graph={"inline": {"nodes": {}, "wires": [], "input_ports": [], "output_ports": []}},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=2, batch_size=3, learning_rate=0.01),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )
    return spec.model_dump(mode="json")


def _write_matrix(path: Path) -> None:
    payload = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "fork gate adapter",
        "base": {"inline": _training_run_payload()},
        "fork": {
            "source_run_id": "feedbax-training-run:source",
            "lr_continuation": "continue",
            "expected_slots": ["model"],
        },
        "rows": [
            {
                "row_id": "lr_hi",
                "overrides": [
                    {"path": "training_config.learning_rate", "op": "replace", "value": 0.02}
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_latest(root: Path, *, transaction_id: str, digest: str) -> None:
    tx_dir = root / "transactions" / transaction_id
    tx_dir.mkdir(parents=True)
    manifest = {
        "transaction_id": transaction_id,
        "completed_training_batches": 5,
        "content_integrity_digest": {
            "slots": [{"slot": "model", "slot_root_sha256": digest}],
        },
    }
    (tx_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({"manifest_relative_path": f"transactions/{transaction_id}/manifest.json"}),
        encoding="utf-8",
    )


def test_parse_target_accepts_matrix_row_checkpoint_root() -> None:
    target = parse_target("lr_hi=/tmp/checkpoints")

    assert target.row_id == "lr_hi"
    assert target.checkpoint_root == Path("/tmp/checkpoints")


def test_lr_continuation_reporter_public_api_handles_restart_and_continue(
    tmp_path: Path,
) -> None:
    row_payload = _training_run_payload()
    row_spec = TrainingRunSpec.model_validate(row_payload)
    reporter = RlrmpLrContinuationReporter(source_checkpoint_root=tmp_path)

    continued = reporter.points(
        source_manifest={"completed_training_batches": 5},
        row_payload=row_payload,
        row_spec=row_spec,
        declared_mode="continue",
    )
    restarted = reporter.points(
        source_manifest={"completed_training_batches": 5},
        row_payload=row_payload,
        row_spec=row_spec,
        declared_mode="restart",
    )

    assert continued == [
        {
            "step": 5,
            "global_step": 5,
            "optimizer_count": 5,
            "lr": 0.01,
            "mode": "continue",
            "completed_batches": 5,
        }
    ]
    assert restarted[0] == {
        **continued[0],
        "step": 0,
        "global_step": 0,
        "optimizer_count": 0,
        "mode": "restart",
    }


def test_checkpoint_fork_gate_has_no_lr_reporter_implementation_residue() -> None:
    from rlrmp.runtime import checkpoint_fork_gate

    source = inspect.getsource(checkpoint_fork_gate)

    assert RlrmpLrContinuationReporter.__module__ == "rlrmp.runtime.lr_continuation"
    assert "class RlrmpLrContinuationReporter" not in source
    assert "def _learning_rate_at_step" not in source
    assert "def _adaptive_epsilon_lr_continuation_points" not in source


def test_fork_checkpoints_with_parity_delegates_matrix_skip_fork(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_latest(source, transaction_id="tx-source", digest="same")
    _write_latest(target, transaction_id="tx-target", digest="same")

    table = fork_checkpoints_with_parity(
        matrix_path=matrix_path,
        source_checkpoint_root=source,
        targets=[parse_target(f"lr_hi={target}")],
        parity_output_path=tmp_path / "parity.json",
        repo_root=tmp_path,
        skip_fork=True,
    )

    assert table["schema_version"] == "feedbax.run_matrix_fork_parity.v1"
    assert table["ok"] is True
    assert any(row["kind"] == "slot_parity" and row["ok"] for row in table["rows"])
    assert any(row["kind"] == "lr_continuation" and row["lr"] == 0.02 for row in table["rows"])

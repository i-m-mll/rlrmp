from __future__ import annotations

import json
from pathlib import Path
import inspect
from copy import deepcopy

import pytest

from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.training import (
    LossTermSpec,
    MethodPayloadEnvelope,
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

from rlrmp.runtime.checkpoint_fork_gate import (
    ForkParityError,
    fork_checkpoints_with_parity,
    parse_target,
)
from rlrmp.runtime.checkpoint_fork_gate import (
    _assert_payload_lr_continuation_mode,
    _assert_stage2_lambda_update_contract,
    _canonical_task_identity_hash,
    _ratio_setpoint_prelaunch_report,
    format_ratio_setpoint_report,
    load_matrix,
)
from rlrmp.runtime.lr_continuation import RlrmpLrContinuationReporter
from rlrmp.train.adaptive_epsilon_native import adaptive_epsilon_method_ref


def _training_run_payload(
    *,
    task_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    task_params: dict[str, object] = {"n_steps": 4}
    if task_identity is not None:
        task_params.update(deepcopy(task_identity))
    spec = TrainingRunSpec(
        graph={"inline": {"nodes": {}, "wires": [], "input_ports": [], "output_ports": []}},
        task=TaskSpec(type="ReachingTask", params=task_params),
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


def _task_identity() -> dict[str, object]:
    return {
        "game_card": {
            "dt": 0.01,
            "plant": {"state_dim": 6},
        },
        "perturbation_training": {
            "bank": {"identity": "open-loop-moderate"},
            "families": ["initial_position", "process_epsilon"],
        },
    }


def _write_matrix(
    path: Path,
    *,
    source_identity: dict[str, object] | None = None,
    target_identity: dict[str, object] | None = None,
    ratio_setpoint: dict[str, object] | None = None,
) -> None:
    source_identity = _task_identity() if source_identity is None else source_identity
    target_identity = deepcopy(source_identity) if target_identity is None else target_identity
    source_path = path.parent / "source_run.json"
    source_path.write_text(
        json.dumps(
            {
                "game_card": source_identity["game_card"],
                "training_distribution": {
                    "perturbation_training": source_identity["perturbation_training"],
                },
            }
        ),
        encoding="utf-8",
    )
    metadata: dict[str, object] = {
        "rlrmp_source_run_spec_ref": source_path.name,
        "rlrmp_task_identity": _canonical_task_identity_hash(source_identity),
    }
    if ratio_setpoint is not None:
        metadata["ratio_setpoint"] = ratio_setpoint
    payload = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "fork gate adapter",
        "base": {"inline": _training_run_payload(task_identity=source_identity)},
        "fork": {
            "source_run_id": "feedbax-training-run:source",
            "lr_continuation": "continue",
            "expected_slots": ["model"],
        },
        "metadata": metadata,
        "rows": [
            {
                "row_id": "lr_hi",
                "metadata": {
                    "rlrmp_task_identity": _canonical_task_identity_hash(target_identity),
                },
                "overrides": [
                    {"path": "training_config.learning_rate", "op": "replace", "value": 0.02},
                    *_task_identity_overrides(source_identity, target_identity),
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _task_identity_overrides(
    source: dict[str, object],
    target: dict[str, object],
) -> list[dict[str, object]]:
    """Return row patches that mutate actual materialized task-spec leaves."""

    overrides: list[dict[str, object]] = []

    def walk(source_value: object, target_value: object, *, path: str) -> None:
        if isinstance(source_value, dict) and isinstance(target_value, dict):
            assert set(source_value) == set(target_value)
            for key in sorted(source_value):
                walk(source_value[key], target_value[key], path=f"{path}.{key}")
            return
        if source_value != target_value:
            overrides.append({"path": path, "op": "replace", "value": target_value})

    for subtree in ("game_card", "perturbation_training"):
        walk(
            source[subtree],
            target[subtree],
            path=f"task.params.{subtree}",
        )
    return overrides


def _adaptive_epsilon_row_spec(*, payload_mode: str | None) -> TrainingRunSpec:
    spec = TrainingRunSpec.model_validate(_training_run_payload())
    payload = {"lr_continuation_mode": payload_mode} if payload_mode is not None else {}
    return spec.model_copy(
        update={
            "method_ref": adaptive_epsilon_method_ref(),
            "method_payload": MethodPayloadEnvelope(
                schema_id="rlrmp.test.adaptive_epsilon_payload",
                schema_version="v1",
                payload=payload,
            ),
        }
    )


def _stage2_lambda_update_payload(
    *,
    eta: float = 0.2,
    lambda_min: float = 0.002,
    omit_field: str | None = None,
) -> dict[str, object]:
    lambda_update: dict[str, object] = {
        "interval_batches": 50,
        "ema_alpha": 0.1,
        "eta": eta,
        "max_log_step": 0.15,
        "deadband_frac": 0.03,
        "freeze_during_application_ramp": False,
        "lambda_min": lambda_min,
    }
    if omit_field is not None:
        lambda_update.pop(omit_field)
    return {
        "config": {"broad_epsilon_pgd_energy_lambda": 2.0},
        "lambda_update": lambda_update,
    }


def _stage2_adaptive_epsilon_row_spec(
    *,
    eta: float = 0.2,
    lambda_min: float = 0.002,
    omit_field: str | None = None,
) -> TrainingRunSpec:
    spec = TrainingRunSpec.model_validate(_training_run_payload(task_identity=_task_identity()))
    return spec.model_copy(
        update={
            "method_ref": adaptive_epsilon_method_ref(),
            "method_payload": MethodPayloadEnvelope(
                schema_id="rlrmp.test.adaptive_epsilon_payload",
                schema_version="v1",
                payload=_stage2_lambda_update_payload(
                    eta=eta,
                    lambda_min=lambda_min,
                    omit_field=omit_field,
                ),
            ),
        }
    )


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


def test_lr_continuation_reporter_rejects_phase_coordinate_as_batch_total(
    tmp_path: Path,
) -> None:
    row_payload = _training_run_payload()
    row_spec = TrainingRunSpec.model_validate(row_payload)
    reporter = RlrmpLrContinuationReporter(source_checkpoint_root=tmp_path)

    with pytest.raises(
        ValueError,
        match="phase/global coordinates cannot supply continuation batch arithmetic",
    ):
        reporter.points(
            source_manifest={"completed_coordinate": {"phase_step": 24}},
            row_payload=row_payload,
            row_spec=row_spec,
            declared_mode="continue",
        )


def test_checkpoint_fork_gate_has_no_lr_reporter_implementation_residue() -> None:
    from rlrmp.runtime import checkpoint_fork_gate

    source = inspect.getsource(checkpoint_fork_gate)

    assert RlrmpLrContinuationReporter.__module__ == "rlrmp.runtime.lr_continuation"
    assert "class RlrmpLrContinuationReporter" not in source
    assert "def _learning_rate_at_step" not in source
    assert "def _adaptive_epsilon_lr_continuation_points" not in source


def test_task_identity_gate_rejects_real_row_game_card_leaf_with_derived_hash_labels(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    target_identity = _task_identity()
    target_identity["game_card"]["plant"]["state_dim"] = 8  # type: ignore[index]
    _write_matrix(matrix_path, target_identity=target_identity)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert matrix["metadata"]["rlrmp_task_identity"] == _canonical_task_identity_hash(
        _task_identity()
    )
    assert matrix["rows"][0]["metadata"]["rlrmp_task_identity"] == _canonical_task_identity_hash(
        target_identity
    )

    with pytest.raises(
        ForkParityError,
        match=(
            "task identity hash mismatch row='lr_hi' path='game_card.plant.state_dim': "
            ".*source=6 target=8"
        ),
    ):
        fork_checkpoints_with_parity(
            matrix_path=matrix_path,
            source_checkpoint_root=tmp_path / "source",
            targets=[parse_target(f"lr_hi={tmp_path / 'target'}")],
            parity_output_path=tmp_path / "parity.json",
            repo_root=tmp_path,
            skip_fork=True,
        )


def test_task_identity_gate_rejects_real_row_leaf_with_copied_matching_labels(
    tmp_path: Path,
) -> None:
    """Copied labels must not hide a mutation of the actual materialized row task."""

    matrix_path = tmp_path / "matrix.json"
    target_identity = _task_identity()
    target_identity["game_card"]["plant"]["state_dim"] = 8  # type: ignore[index]
    _write_matrix(matrix_path, target_identity=target_identity)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["rows"][0]["metadata"]["rlrmp_task_identity"] = payload["metadata"][
        "rlrmp_task_identity"
    ]
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ForkParityError,
        match=(
            "task identity label mismatch "
            "field=row='lr_hi'.metadata.rlrmp_task_identity: "
            "label='sha256:.*' derived='sha256:"
        ),
    ):
        fork_checkpoints_with_parity(
            matrix_path=matrix_path,
            source_checkpoint_root=tmp_path / "source",
            targets=[parse_target(f"lr_hi={tmp_path / 'target'}")],
            parity_output_path=tmp_path / "parity.json",
            repo_root=tmp_path,
            skip_fork=True,
        )


def test_task_identity_gate_rejects_perturbation_training_field_drift(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    target_identity = _task_identity()
    target_identity["perturbation_training"]["bank"]["identity"] = "different-bank"  # type: ignore[index]
    _write_matrix(matrix_path, target_identity=target_identity)

    with pytest.raises(
        ForkParityError,
        match=(
            "task identity hash mismatch row='lr_hi' "
            "path='perturbation_training.bank.identity': "
            ".*source='open-loop-moderate' target='different-bank'"
        ),
    ):
        fork_checkpoints_with_parity(
            matrix_path=matrix_path,
            source_checkpoint_root=tmp_path / "source",
            targets=[parse_target(f"lr_hi={tmp_path / 'target'}")],
            parity_output_path=tmp_path / "parity.json",
            repo_root=tmp_path,
            skip_fork=True,
        )


def test_task_identity_gate_rejects_missing_target_identity(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["rows"][0]["metadata"].pop("rlrmp_task_identity")
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ForkParityError,
        match="task identity gate missing row='lr_hi'.metadata.rlrmp_task_identity",
    ):
        fork_checkpoints_with_parity(
            matrix_path=matrix_path,
            source_checkpoint_root=tmp_path / "source",
            targets=[parse_target(f"lr_hi={tmp_path / 'target'}")],
            parity_output_path=tmp_path / "parity.json",
            repo_root=tmp_path,
            skip_fork=True,
        )


def test_task_identity_gate_rejects_stale_hash_label(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["rows"][0]["metadata"]["rlrmp_task_identity"] = "sha256:stale"
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ForkParityError,
        match=(
            "task identity label mismatch "
            "field=row='lr_hi'.metadata.rlrmp_task_identity: label='sha256:stale' derived='sha256:"
        ),
    ):
        fork_checkpoints_with_parity(
            matrix_path=matrix_path,
            source_checkpoint_root=tmp_path / "source",
            targets=[parse_target(f"lr_hi={tmp_path / 'target'}")],
            parity_output_path=tmp_path / "parity.json",
            repo_root=tmp_path,
            skip_fork=True,
        )


def test_task_identity_gate_rejects_missing_actual_row_task_subtree(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["base"]["inline"]["task"]["params"].pop("perturbation_training")
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ForkParityError,
        match="task identity gate missing row='lr_hi'.spec.task.params.perturbation_training",
    ):
        fork_checkpoints_with_parity(
            matrix_path=matrix_path,
            source_checkpoint_root=tmp_path / "source",
            targets=[parse_target(f"lr_hi={tmp_path / 'target'}")],
            parity_output_path=tmp_path / "parity.json",
            repo_root=tmp_path,
            skip_fork=True,
        )


def test_lr_continuation_gate_rejects_missing_payload_mode() -> None:
    with pytest.raises(
        ForkParityError,
        match="LR continuation mode mismatch row='restart_row': declared='restart' payload=<missing>",
    ):
        _assert_payload_lr_continuation_mode(
            row_id="restart_row",
            row_spec=_adaptive_epsilon_row_spec(payload_mode=None),
            declared_mode="restart",
        )


def test_lr_continuation_gate_rejects_payload_mode_mismatch() -> None:
    with pytest.raises(
        ForkParityError,
        match="LR continuation mode mismatch row='restart_row': declared='restart' payload='continue'",
    ):
        _assert_payload_lr_continuation_mode(
            row_id="restart_row",
            row_spec=_adaptive_epsilon_row_spec(payload_mode="continue"),
            declared_mode="restart",
        )


def test_lr_continuation_gate_accepts_declared_restart_payload() -> None:
    _assert_payload_lr_continuation_mode(
        row_id="restart_row",
        row_spec=_adaptive_epsilon_row_spec(payload_mode="restart"),
        declared_mode="restart",
    )


def test_stage2_lambda_update_gate_accepts_tracked_retune() -> None:
    _assert_stage2_lambda_update_contract(
        row_id="flat_3e-5",
        row_spec=_stage2_adaptive_epsilon_row_spec(),
        repo_root=Path(__file__).resolve().parents[1],
    )


def test_stage2_lambda_update_gate_rejects_retune_eta_drift() -> None:
    with pytest.raises(
        ForkParityError,
        match=(
            "lambda-update gate mismatch row='flat_3e-5' "
            "field=method_payload.payload.lambda_update.eta: expected=0.2 actual=0.1"
        ),
    ):
        _assert_stage2_lambda_update_contract(
            row_id="flat_3e-5",
            row_spec=_stage2_adaptive_epsilon_row_spec(eta=0.1),
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_stage2_lambda_update_gate_rejects_missing_retuned_field() -> None:
    with pytest.raises(
        ForkParityError,
        match=(
            "lambda-update gate mismatch row='flat_3e-5' "
            "field=method_payload.payload.lambda_update.max_log_step: "
            "expected=0.15 actual=<missing>"
        ),
    ):
        _assert_stage2_lambda_update_contract(
            row_id="flat_3e-5",
            row_spec=_stage2_adaptive_epsilon_row_spec(omit_field="max_log_step"),
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_stage2_lambda_update_gate_rejects_lambda_min_seed_ratio_drift() -> None:
    with pytest.raises(
        ForkParityError,
        match=(
            "lambda-update gate mismatch row='flat_3e-5' "
            "field=method_payload.payload.lambda_update.lambda_min: "
            "expected=0.001 \\* broad_epsilon_pgd_energy_lambda=0.002 actual=0.004"
        ),
    ):
        _assert_stage2_lambda_update_contract(
            row_id="flat_3e-5",
            row_spec=_stage2_adaptive_epsilon_row_spec(lambda_min=0.004),
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_ratio_setpoint_report_includes_required_derivation_components(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(
        matrix_path,
        ratio_setpoint={
            "numerator": 1024,
            "numerator_convention": "excess",
            "denominator_window": "baseline_final_quarter",
            "baseline_final_quarter_mean_clean_loss": 4444,
        },
    )

    report = _ratio_setpoint_prelaunch_report(load_matrix(matrix_path))

    assert report is not None
    assert report["rounded_ratio_setpoint_2sf"] == pytest.approx(0.23)
    rendered = format_ratio_setpoint_report(report)
    assert "numerator_convention=excess" in rendered
    assert "denominator_window=baseline_final_quarter" in rendered
    assert "numerator=1024" in rendered
    assert "baseline_final_quarter_mean_clean_loss=4444" in rendered
    assert "rounded_ratio_setpoint_2sf=0.23" in rendered


def test_ratio_setpoint_report_rejects_noncanonical_excess_numerator(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _write_matrix(
        matrix_path,
        ratio_setpoint={
            "numerator": 1000,
            "numerator_convention": "excess",
            "denominator_window": "baseline_final_quarter",
            "baseline_final_quarter_mean_clean_loss": 4444,
        },
    )

    with pytest.raises(
        ForkParityError,
        match="ratio setpoint metadata requires excess numerator=1024; got 1000",
    ):
        _ratio_setpoint_prelaunch_report(load_matrix(matrix_path))


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

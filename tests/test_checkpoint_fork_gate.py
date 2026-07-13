from __future__ import annotations

import ast
import json
from pathlib import Path
import inspect
import subprocess
import tomllib
from copy import deepcopy
from types import SimpleNamespace

import pytest

import feedbax
import jax.tree as jt
import rlrmp.runtime.checkpoint_fork_gate as checkpoint_fork_gate
from feedbax.contracts.checkpoints import BatchHistory, CheckpointForkBarrierMapping
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRunMatrixSpec,
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
from feedbax.training.run_matrix import fork_matrix_checkpoints
from feedbax.training.checkpoint_custody import _validate_program_step_units

from rlrmp.runtime.adaptive_checkpoint_adapter import NominalToAdaptiveSlotAdapter
from rlrmp.runtime.checkpoint_fork_gate import (
    ForkParityError,
    _adaptive_continuation_fork_contracts,
    _source_program_step_from_manifest,
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
from rlrmp.train.adaptive_epsilon_native import (
    adaptive_epsilon_method_ref,
    attach_adaptive_epsilon_checkpoint_continuation,
)
from rlrmp.train.cs_nominal_gru import (
    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CsNominalGruConfig,
    _config_namespace,
    write_run_spec,
)


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
        "base": {
            "kind": "inline",
            "inline": _training_run_payload(task_identity=source_identity),
        },
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


def _write_latest(
    root: Path,
    *,
    transaction_id: str,
    digest: str,
    completed_batches: int = 5,
) -> None:
    tx_dir = root / "transactions" / transaction_id
    tx_dir.mkdir(parents=True)
    manifest = {
        "transaction_id": transaction_id,
        "completed_training_batches": completed_batches,
        "completed_coordinate": {"program_step": completed_batches},
        "content_integrity_digest": {
            "slots": [{"slot": "model", "slot_root_sha256": digest}],
        },
    }
    (tx_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({"manifest_relative_path": f"transactions/{transaction_id}/manifest.json"}),
        encoding="utf-8",
    )


def _mock_source_ordinal(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ordinal: int,
    coordinate: int | None = None,
) -> None:
    manifest = SimpleNamespace(
        metadata={"barrier_visit_ordinal": ordinal},
        completed_coordinate=SimpleNamespace(
            program_step=ordinal if coordinate is None else coordinate
        ),
    )
    monkeypatch.setattr(
        checkpoint_fork_gate,
        "load_checkpoint_custody_documents",
        lambda _root: SimpleNamespace(manifest=SimpleNamespace(document=manifest)),
    )


def _write_adaptive_skip_fork_manifests(
    source_root: Path,
    target_root: Path,
    *,
    adapter: NominalToAdaptiveSlotAdapter,
) -> None:
    """Write honest topology/provenance evidence for the real skip-fork API test."""

    comparable_slots = ("model", "optimizer")
    source_blobs = {slot: f"source-{slot}-blob" for slot in comparable_slots}
    target_slots = (*comparable_slots, *adapter.target_only_slots)
    target_blobs = {slot: f"target-{slot}-blob" for slot in target_slots}
    source_manifest = {
        "transaction_id": "source",
        "completed_training_batches": 2,
        "completed_coordinate": {"program_step": 2},
        "slots": [{"slot": slot, "sha256": source_blobs[slot]} for slot in comparable_slots],
        "content_integrity_digest": {
            "slots": [
                {"slot": slot, "slot_root_sha256": f"source-{slot}-content"}
                for slot in comparable_slots
            ]
        },
    }
    provenance = []
    for slot in target_slots:
        metadata = {
            "stage": "target_post",
            "stages": [
                {
                    "stage": "target_post",
                    "identity": adapter.transform_metadata["identity"],
                    "parameters": adapter.transform_metadata["parameters"],
                    "metadata": {},
                }
            ],
        }
        if slot in adapter.target_only_slots:
            metadata["target_only_declaration"] = adapter.target_only_slots[slot]
        provenance.append(
            {
                "slot": slot,
                "source_sha256": source_blobs.get(slot),
                "target_sha256": target_blobs[slot],
                "transfer_mode": "serialized",
                "transform": {
                    "slot": slot,
                    "identity": adapter.transform_metadata["identity"],
                    "parameters": adapter.transform_metadata["parameters"],
                    "metadata": metadata,
                },
            }
        )
    target_manifest = {
        "transaction_id": "target",
        "completed_training_batches": 2,
        "completed_coordinate": {"program_step": 2},
        "slots": [{"slot": slot, "sha256": target_blobs[slot]} for slot in target_slots],
        "content_integrity_digest": {
            "slots": [
                {"slot": slot, "slot_root_sha256": f"target-{slot}-content"}
                for slot in target_slots
            ]
        },
        "fork_provenance": {"slots": provenance},
    }
    for root, manifest in ((source_root, source_manifest), (target_root, target_manifest)):
        tx_dir = root / "transactions" / manifest["transaction_id"]
        tx_dir.mkdir(parents=True)
        (tx_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "latest.json").write_text(
            json.dumps(
                {
                    "manifest_relative_path": f"transactions/{manifest['transaction_id']}/manifest.json"
                }
            ),
            encoding="utf-8",
        )


def _adaptive_continuation_spec(
    tmp_path: Path,
    *,
    source_completed_batches: int = 2,
    target_total_batches: int = 4,
    n_replicates: int = 1,
    checkpoint_interval_batches: int = 1,
) -> TrainingRunSpec:
    """Build a minimal real adaptive row through RLRMP's normal authoring path."""

    args = _config_namespace(
        CsNominalGruConfig(issue="test", output_dir="_artifacts/test/runs/test")
    )
    values = {
        "n_train_batches": target_total_batches,
        "batch_size": 1,
        "n_replicates": n_replicates,
        "hidden_size": 4,
        "dry_run": True,
        "full_train": True,
        "resume": True,
        "checkpoint_interval_batches": checkpoint_interval_batches,
        "controller_lr": 1e-3,
        "gradient_clip_norm": 5.0,
        "lr_warmup_batches": 1,
        "lr_warmup_init_fraction": 0.1,
        "lr_cosine_alpha": 0.01,
        "log_step": 1,
        "disable_progress": True,
        "quiet_progress": True,
        "target_relative_multitarget": True,
        "force_filter_feedback": True,
        "broad_epsilon_pgd_training": True,
        "broad_epsilon_pgd_steps": 1,
        "broad_epsilon_pgd_step_size_fraction": 0.5,
        "broad_epsilon_pgd_objective": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
        "broad_epsilon_pgd_energy_lambda": 1.0,
        "adaptive_epsilon_curriculum": True,
        "adaptive_epsilon_update_interval_batches": 1,
        "adaptive_epsilon_outer_weight_start": 0.25,
        "adaptive_epsilon_outer_weight_final": 0.25,
        "adaptive_epsilon_outer_weight_ramp_batches": 0,
        "adaptive_epsilon_controller_training_mode": (
            ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
        ),
        "loss_objective": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        "output_dir": str(tmp_path / "bulk"),
        "spec_dir": str(tmp_path / "spec"),
    }
    for key, value in values.items():
        setattr(args, key, value)
    payload = write_run_spec(args)["run_spec"]
    return attach_adaptive_epsilon_checkpoint_continuation(
        TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"]),
        source_completed_batches=source_completed_batches,
        target_total_batches=target_total_batches,
    )


def test_parse_target_accepts_matrix_row_checkpoint_root() -> None:
    target = parse_target("lr_hi=/tmp/checkpoints")

    assert target.row_id == "lr_hi"
    assert target.checkpoint_root == Path("/tmp/checkpoints")


def test_source_program_step_uses_manifest_ordinal_not_continuation_interval() -> None:
    manifest = {
        "metadata": {"barrier_visit_ordinal": 24},
        "completed_coordinate": {"program_step": 24},
    }

    assert _source_program_step_from_manifest(manifest) == 24
    assert _source_program_step_from_manifest(manifest) != 12_000 // 100


@pytest.mark.parametrize(
    "manifest",
    [
        {"metadata": {}, "completed_coordinate": {"program_step": 24}},
        {
            "metadata": {"barrier_visit_ordinal": 24},
            "completed_coordinate": {},
        },
    ],
)
def test_source_program_step_requires_both_manifest_ordinal_fields(
    manifest: dict[str, object],
) -> None:
    with pytest.raises(ForkParityError, match="requires non-negative integer"):
        _source_program_step_from_manifest(manifest)


def test_source_program_step_rejects_disagreeing_manifest_fields() -> None:
    with pytest.raises(ForkParityError, match="ordinal fields disagree"):
        _source_program_step_from_manifest(
            {
                "metadata": {"barrier_visit_ordinal": 24},
                "completed_coordinate": {"program_step": 25},
            }
        )


@pytest.mark.parametrize(
    ("checkpoint_interval_batches", "target_total_batches"),
    [(100, 12_200), (500, 16_500)],
)
def test_adaptive_fork_target_stays_in_source_manifest_ordinal_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_interval_batches: int,
    target_total_batches: int,
) -> None:
    """Changing continuation cadence never reinterprets the source seam."""
    target_spec = _adaptive_continuation_spec(
        tmp_path,
        source_completed_batches=12_000,
        target_total_batches=target_total_batches,
        checkpoint_interval_batches=checkpoint_interval_batches,
    )
    materialized = SimpleNamespace(
        rows=[
            SimpleNamespace(
                row_id="adaptive",
                planned_run_id="feedbax-training-run:adaptive",
                spec=target_spec,
            )
        ]
    )
    _mock_source_ordinal(monkeypatch, ordinal=24)

    _, mapping = _adaptive_continuation_fork_contracts(
        materialized, source_checkpoint_root=tmp_path / "source"
    )["adaptive"]

    assert mapping.target_coordinate is not None
    assert mapping.target_coordinate.program_step == 24
    _validate_program_step_units(
        mapping.target_coordinate,
        {"barrier_visit_ordinal": 24},
        context="test fork target",
    )
    if checkpoint_interval_batches == 100:
        assert mapping.target_coordinate.program_step != 12_000 // checkpoint_interval_batches


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


def test_adaptive_restart_lr_report_uses_target_optimizer_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row_spec = _adaptive_continuation_spec(tmp_path)
    source_root = tmp_path / "source"
    transaction = source_root / "transactions" / "tx-source"
    transaction.mkdir(parents=True)
    manifest = {
        "completed_training_batches": 2,
        "slots": [],
    }
    manifest_path = transaction / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_root.joinpath("latest.json").write_text(
        json.dumps({"manifest_relative_path": "transactions/tx-source/manifest.json"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rlrmp.runtime.lr_continuation._load_manifest_slot",
        lambda *_args, **_kwargs: pytest.fail("restart must not inspect nominal optimizer state"),
    )

    points = RlrmpLrContinuationReporter(source_checkpoint_root=source_root).points(
        source_manifest=manifest,
        row_payload=row_spec.model_dump(mode="json"),
        row_spec=row_spec,
        declared_mode="restart",
    )

    assert points
    assert {point["mode"] for point in points} == {"restart"}
    assert {point["optimizer_count_at_current_step"] for point in points} == {0}


def test_checkpoint_fork_gate_has_no_lr_reporter_implementation_residue() -> None:
    from rlrmp.runtime import checkpoint_fork_gate

    source = inspect.getsource(checkpoint_fork_gate)

    assert RlrmpLrContinuationReporter.__module__ == "rlrmp.runtime.lr_continuation"
    assert "class RlrmpLrContinuationReporter" not in source
    assert "def _learning_rate_at_step" not in source
    assert "def _adaptive_epsilon_lr_continuation_points" not in source


def test_checkpoint_fork_gate_registers_adaptive_epsilon_method(monkeypatch) -> None:
    from rlrmp.runtime import checkpoint_fork_gate

    calls: list[str] = []
    monkeypatch.setattr(
        checkpoint_fork_gate,
        "ensure_adaptive_epsilon_training_method_registered",
        lambda: calls.append("adaptive_epsilon"),
    )
    monkeypatch.setattr(
        checkpoint_fork_gate,
        "ensure_minimax_training_method_registered",
        lambda: calls.append("minimax"),
    )
    monkeypatch.setattr(
        checkpoint_fork_gate,
        "register_rlrmp_cs_supervised_method",
        lambda: calls.append("cs_supervised"),
    )
    monkeypatch.setattr(
        checkpoint_fork_gate,
        "register_rlrmp_distillation_methods",
        lambda: calls.append("distillation"),
    )

    checkpoint_fork_gate.register_rlrmp_training_methods()

    assert calls == ["adaptive_epsilon", "minimax", "cs_supervised", "distillation"]


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


def test_fork_gate_forward_api_guard_matches_pinned_feedbax_delivery() -> None:
    """The gate's real Feedbax call cannot advance beyond the tracked pin."""

    source_path = Path(__file__).resolve().parents[1] / "src/rlrmp/runtime/checkpoint_fork_gate.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fork_matrix_checkpoints"
    ]
    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    required_continuation_kwargs = {
        "target_slot_templates",
        "row_slot_transforms",
        "row_transform_metadata",
        "row_segment_history_templates",
        "row_target_slot_transforms",
        "row_target_transform_metadata",
        "row_target_transformed_slots",
        "row_target_only_slots",
        "row_barrier_mappings",
    }
    assert required_continuation_kwargs <= keyword_names

    available_parameters = set(inspect.signature(fork_matrix_checkpoints).parameters)
    assert keyword_names <= available_parameters, (
        "RLRMP checkpoint_fork_gate calls Feedbax fork_matrix_checkpoints with "
        f"parameters absent at the installed/pinned Feedbax revision: "
        f"{sorted(keyword_names - available_parameters)!r}"
    )
    assert CheckpointForkBarrierMapping.__module__ == "feedbax.contracts.checkpoints"

    package_dir = Path(feedbax.__file__).resolve().parent
    checkout_root = next(
        (
            candidate
            for candidate in (package_dir, *package_dir.parents)
            if (candidate / ".git").exists()
        ),
        None,
    )
    if checkout_root is not None:
        pin = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "ci/feedbax-ref.toml").read_text(
                encoding="utf-8"
            )
        )["rev"]
        head = subprocess.run(
            [
                "git",
                "-c",
                "core.optionalLocks=false",
                "-C",
                str(checkout_root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == pin, "forward API guard must inspect the revision pinned for RLRMP"


def test_adaptive_fork_contracts_call_real_pinned_feedbax_matrix_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise adaptive contracts through Feedbax's real matrix-fork function.

    This uses ``skip_fork`` only to avoid writing a full model checkpoint in a
    unit test.  It deliberately does not mock Feedbax's function or signature:
    the adaptive contracts are constructed by RLRMP and passed to the installed
    Feedbax matrix API exactly as the launch gate does.
    """

    target_spec = _adaptive_continuation_spec(tmp_path)
    row = SimpleNamespace(
        row_id="adaptive",
        planned_run_id="feedbax-training-run:adaptive",
        spec=target_spec,
        payload=target_spec.model_dump(mode="json"),
    )
    materialized = SimpleNamespace(matrix_spec_sha256="test-matrix", rows=[row])
    _mock_source_ordinal(monkeypatch, ordinal=2)
    contracts = _adaptive_continuation_fork_contracts(
        materialized, source_checkpoint_root=tmp_path / "source"
    )
    adapter, barrier_mapping = contracts["adaptive"]
    assert barrier_mapping.source_barrier == "after_train_chunk"
    assert barrier_mapping.target_barrier == "after_adaptive_epsilon_train_chunk"
    assert barrier_mapping.target_coordinate is not None
    assert barrier_mapping.target_coordinate.program_step == 2
    assert barrier_mapping.coordinate_mapping == {
        "identity": "rlrmp.cs_supervised_to_adaptive_epsilon.v1",
        "parameters": {
            "program_step": "source_manifest_completed_barrier_ordinal",
            "source_completed_training_batches": 2,
            "source_manifest_program_step": 2,
        },
    }

    matrix_path = tmp_path / "matrix.json"
    _write_matrix(matrix_path)
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_payload["fork"]["expected_slots"] = ["model", "optimizer"]
    matrix = TrainingRunMatrixSpec.model_validate(matrix_payload)
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_adaptive_skip_fork_manifests(source_root, target_root, adapter=adapter)

    table = fork_matrix_checkpoints(
        matrix,
        materialized,
        source_checkpoint_root=source_root,
        target_checkpoint_roots={"adaptive": target_root},
        parity_output_path=tmp_path / "parity.json",
        target_slot_templates={"adaptive": adapter.adaptive_initial_slots},
        row_segment_history_templates={"adaptive": adapter.continuation_slot_templates()},
        row_target_slot_transforms={"adaptive": adapter.transform},
        row_target_transform_metadata={"adaptive": adapter.transform_metadata},
        row_target_transformed_slots={"adaptive": adapter.target_transformed_slots},
        row_target_only_slots={"adaptive": adapter.target_only_slots},
        row_barrier_mappings={"adaptive": barrier_mapping},
        skip_fork=True,
    )

    assert table["ok"] is True
    assert (tmp_path / "parity.json").is_file()


def test_adaptive_fork_contract_uses_real_shaped_segment_local_optimizer_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 12k source fork targets 4.5k histories, never the 16.5k cumulative total."""

    target_spec = _adaptive_continuation_spec(
        tmp_path,
        source_completed_batches=12_000,
        target_total_batches=16_500,
        n_replicates=5,
    )
    row = SimpleNamespace(
        row_id="adaptive",
        planned_run_id="feedbax-training-run:adaptive",
        spec=target_spec,
    )

    _mock_source_ordinal(monkeypatch, ordinal=24)
    adapter, _ = _adaptive_continuation_fork_contracts(
        SimpleNamespace(rows=[row]), source_checkpoint_root=tmp_path / "source"
    )["adaptive"]
    optimizer_leaves = tuple(jt.leaves(adapter.optimizer_template))
    segment_template_leaves = adapter.continuation_slot_templates()["optimizer"]

    assert target_spec.checkpoint_progress.continuation is not None
    assert target_spec.checkpoint_progress.continuation.additional_batches == 4_500
    assert optimizer_leaves[1].shape == (5, 4_500)
    assert isinstance(segment_template_leaves[1], BatchHistory)
    assert segment_template_leaves[1].value.shape == (5, 4_500)
    assert all(getattr(leaf, "shape", None) != (5, 16_500) for leaf in optimizer_leaves)

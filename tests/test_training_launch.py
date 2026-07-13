"""Focused contract tests for the authored training launch frontend."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rlrmp.train import launch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _minimal_matrix() -> dict[str, Any]:
    return {
        "schema_id": "feedbax.spec.training_run_matrix",
        "schema_version": "feedbax.spec.training_run_matrix.v3",
        "name": "one-row launch",
        "base": {
            "kind": "resolved_output",
            "ref": "resolved.json",
            "resolved_root_hash": "0" * 64,
        },
        "rows": [{"row_id": "only"}],
    }


def _emitted_launch(
    tmp_path: Path,
    *,
    fork: bool = False,
    metadata: dict[str, Any] | None = None,
) -> launch.AuthoredLaunch:
    payload = _minimal_matrix()
    if fork:
        payload["fork"] = {"lr_continuation": "continue"}
    if metadata is not None:
        payload["metadata"] = metadata
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(payload), encoding="utf-8")
    sidecar = {
        "schema_id": payload["schema_id"],
        "schema_version": payload["schema_version"],
        "artifact_id": "test:matrix",
        "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "uri": str(matrix_path),
    }
    matrix_path.with_suffix(".json.artifact.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return launch.load_authored_training_intent(matrix_path, repo_root=tmp_path)


def test_launch_frontend_import_does_not_import_jax() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rlrmp.train.launch; assert 'jax' not in sys.modules",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_loader_accepts_only_strict_training_matrix(tmp_path: Path) -> None:
    valid_path = tmp_path / "matrix.json"
    valid_path.write_text(json.dumps(_minimal_matrix()), encoding="utf-8")
    loaded = launch.load_authored_training_intent(valid_path, repo_root=tmp_path)
    assert loaded.document.rows[0].row_id == "only"

    invalid = _minimal_matrix()
    invalid["scientific_override"] = 3
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError):
        launch.load_authored_training_intent(invalid_path, repo_root=tmp_path)

    nested_path = tmp_path / "nested.json"
    nested_path.write_text(json.dumps({"method_ref": "rlrmp/minimax/v1"}), encoding="utf-8")
    assert not launch.accepted_authored_document(nested_path, repo_root=tmp_path)


def test_row_selection_is_exact() -> None:
    rows = (
        launch.LaunchRow("a", "run-a", object()),
        launch.LaunchRow("b", "run-b", object()),
    )
    assert launch.select_launch_rows(rows, "b") == (rows[1],)
    assert launch.select_launch_rows(rows, None) == rows
    with pytest.raises(ValueError, match="unknown row selector.*a, b"):
        launch.select_launch_rows(rows, "missing")


def test_orchestrated_path_accepts_every_frontend_document(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_minimal_matrix()), encoding="utf-8")
    assert launch.accepted_authored_document(matrix_path, repo_root=tmp_path)
    loaded = launch.load_authored_training_intent(matrix_path, repo_root=tmp_path)
    assert loaded.document.schema_id == "feedbax.spec.training_run_matrix"
    assert launch.RowSelection(row_ids=["only"]).model_dump() == {"row_ids": ["only"]}


def test_fresh_orchestration_request_has_no_checkpoint_input(tmp_path: Path) -> None:
    request, _context, _registry = launch.build_orchestration_request(
        _emitted_launch(tmp_path), row=None, driver="local"
    )

    assert request.inputs == []
    assert request.metadata["row_selection"] == {"row_ids": []}


@pytest.mark.parametrize(
    "metadata",
    [
        {"source_checkpoint_root": "checkpoints"},
        {"source_checkpoint_transaction_id": "txn-1"},
        {"source_checkpoint_root": None},
        {
            "source_checkpoint_root": "checkpoints",
            "source_checkpoint_transaction_id": "txn-1",
        },
    ],
)
def test_fresh_orchestration_request_rejects_checkpoint_metadata(
    tmp_path: Path, metadata: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="fresh execute cannot declare source checkpoint metadata"):
        launch.build_orchestration_request(
            _emitted_launch(tmp_path, metadata=metadata), row=None, driver="local"
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"source_checkpoint_root": "checkpoints"},
        {"source_checkpoint_transaction_id": "txn-1"},
        {
            "source_checkpoint_root": None,
            "source_checkpoint_transaction_id": "txn-1",
        },
        {
            "source_checkpoint_root": "checkpoints",
            "source_checkpoint_transaction_id": None,
        },
        {
            "source_checkpoint_root": 3,
            "source_checkpoint_transaction_id": "txn-1",
        },
        {
            "source_checkpoint_root": "checkpoints",
            "source_checkpoint_transaction_id": 3,
        },
    ],
)
def test_fork_orchestration_request_requires_common_checkpoint_transaction(
    tmp_path: Path, metadata: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="requires one common source checkpoint transaction"):
        launch.build_orchestration_request(
            _emitted_launch(tmp_path, fork=True, metadata=metadata),
            row=None,
            driver="local",
        )


def test_fork_orchestration_request_requires_digest_bound_checkpoint_manifest(
    tmp_path: Path,
) -> None:
    transaction_id = "txn-1"
    manifest = tmp_path / "checkpoints" / "transactions" / transaction_id / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"transaction_id":"txn-1"}\n', encoding="utf-8")
    metadata = {
        "source_checkpoint_root": "checkpoints",
        "source_checkpoint_transaction_id": transaction_id,
    }
    authored = _emitted_launch(tmp_path, fork=True, metadata=metadata)

    request, context, _registry = launch.build_orchestration_request(
        authored, row="only", driver="local"
    )

    assert len(request.inputs) == 1
    declaration = request.inputs[0]
    assert declaration.role == "source_checkpoint"
    assert declaration.kind == "checkpoint_transaction"
    expected_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert f"manifest_sha256={expected_digest}" in declaration.locator
    assert context.input_resolver(declaration).digest.value == expected_digest

    manifest.write_text('{"transaction_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint transaction manifest digest mismatch"):
        context.input_resolver(declaration)


def test_fork_orchestration_request_rejects_missing_checkpoint_manifest(tmp_path: Path) -> None:
    authored = _emitted_launch(
        tmp_path,
        fork=True,
        metadata={
            "source_checkpoint_root": "checkpoints",
            "source_checkpoint_transaction_id": "txn-missing",
        },
    )

    with pytest.raises(ValueError, match="source checkpoint transaction is missing: txn-missing"):
        launch.build_orchestration_request(authored, row=None, driver="local")


def test_execute_controls_keep_fresh_and_fork_resume_semantics_distinct(tmp_path: Path) -> None:
    fresh = _emitted_launch(tmp_path)
    launch._validate_execute_controls(  # type: ignore[attr-defined]
        fresh, launch.LaunchRuntimeControls()
    )
    with pytest.raises(ValueError, match="must match the authored continuation envelope"):
        launch._validate_execute_controls(  # type: ignore[attr-defined]
            fresh, launch.LaunchRuntimeControls(resume=True)
        )

    forked = _emitted_launch(tmp_path, fork=True)
    with pytest.raises(ValueError, match="must match the authored continuation envelope"):
        launch._validate_execute_controls(  # type: ignore[attr-defined]
            forked, launch.LaunchRuntimeControls()
        )
    launch._validate_execute_controls(  # type: ignore[attr-defined]
        forked, launch.LaunchRuntimeControls(resume=True)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allow_fresh_start": True}, "requires resume"),
        ({"stop_after_batches": 0}, "must be positive"),
        ({"log_step": 0}, "must be positive"),
    ],
)
def test_runtime_controls_are_operational_and_fail_closed(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        launch.LaunchRuntimeControls(**kwargs)


def test_launch_evidence_records_lifecycle_controls() -> None:
    controls = launch.LaunchRuntimeControls(
        resume=True, allow_fresh_start=True, stop_after_batches=12, quiet_progress=True
    )
    evidence = launch.launch_evidence((launch.LaunchRow("only", "run-only", object()),), controls)
    assert evidence["runtime_controls"] == {
        "resume": True,
        "allow_fresh_start": True,
        "resume_policy": "resume_if_checkpoint_exists_else_fresh",
        "stop_after_batches": 12,
        "disable_progress": False,
        "quiet_progress": True,
        "log_step": 1,
        "manifest_root": None,
        "checkpoint_root": None,
    }


def test_verify_resume_prepares_executor_context_and_strictly_loads_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    continuation = object()
    phase_program = object()
    method_ref = SimpleNamespace(key="rlrmp/test/v1")
    fake_spec = SimpleNamespace(
        method_ref=method_ref,
        checkpoint_progress=SimpleNamespace(
            metadata={"checkpoint_dir": str(checkpoint_root)},
            continuation=continuation,
        ),
        artifacts=SimpleNamespace(artifact_root="unused"),
        worker_execution=SimpleNamespace(
            method_contract=SimpleNamespace(phase_program=phase_program)
        ),
    )
    row = launch.LaunchRow("only", "run-only", fake_spec)
    monkeypatch.setattr(launch, "compile_authored_training_intent", lambda _launch: (row,))
    prepared = launch._PreparedExecution(  # type: ignore[attr-defined]
        initial_slots={"model": object()},
        kernel_context={"runtime": object()},
        loss_service=object(),
        resume_slot_transform=object(),
    )
    prepare_calls: list[tuple[launch.LaunchRow, bool]] = []

    def prepare(item: launch.LaunchRow, *, resume: bool) -> object:
        prepare_calls.append((item, resume))
        return prepared

    monkeypatch.setattr(launch, "_prepare_execution", prepare)
    import feedbax.training.checkpoint_custody as custody

    load_calls: list[tuple[Path, dict[str, object]]] = []

    def load_checkpoint(root: Path, **kwargs: object) -> object:
        load_calls.append((root, kwargs))
        return SimpleNamespace(manifest=SimpleNamespace(transaction_id="txn-ok"))

    monkeypatch.setattr(custody, "load_latest_checkpoint", load_checkpoint)
    result = launch.verify_resume_authored_training_intent(object())  # type: ignore[arg-type]
    assert prepare_calls == [(row, True)]
    assert load_calls == [
        (
            checkpoint_root,
            {
                "expected_run_spec": fake_spec,
                "expected_phase_program": phase_program,
                "expected_slots": prepared.initial_slots,
                "resume_slot_transform": prepared.resume_slot_transform,
                "continuation_request": continuation,
                "allow_new_lineage_override": True,
            },
        )
    ]
    assert result == (
        {
            "row_id": "only",
            "checkpoint_root": str(checkpoint_root),
            "status": "valid",
            "transaction_id": "txn-ok",
        },
    )

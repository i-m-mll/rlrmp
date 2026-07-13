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
    matrix_path.with_suffix(".json.artifact.json").write_text(json.dumps(sidecar), encoding="utf-8")
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


def test_execute_controls_keep_same_row_and_fork_resume_semantics_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = _emitted_launch(tmp_path)
    assert (
        launch._validate_execute_controls(  # type: ignore[attr-defined]
            fresh, launch.LaunchRuntimeControls()
        )
        == {}
    )
    binding = {
        "only": {
            "checkpoint_root": str(tmp_path / "row"),
            "transaction_id": "tx-1",
            "manifest_sha256": "a" * 64,
        }
    }
    calls: list[tuple[object, str | None, Path | None]] = []

    def same_row(
        authored: object,
        *,
        row: str | None,
        checkpoint_root: Path | None,
    ) -> dict[str, dict[str, str]]:
        calls.append((authored, row, checkpoint_root))
        return binding

    monkeypatch.setattr(launch, "_operational_same_row_resume_bindings", same_row)
    assert (
        launch._validate_execute_controls(  # type: ignore[attr-defined]
            fresh,
            launch.LaunchRuntimeControls(resume=True),
            row="only",
        )
        == binding
    )
    assert calls == [(fresh, "only", None)]

    forked = _emitted_launch(tmp_path, fork=True)
    with pytest.raises(ValueError, match="must match the authored continuation envelope"):
        launch._validate_execute_controls(  # type: ignore[attr-defined]
            forked, launch.LaunchRuntimeControls()
        )
    assert (
        launch._validate_execute_controls(  # type: ignore[attr-defined]
            forked, launch.LaunchRuntimeControls(resume=True)
        )
        == {}
    )


def test_operational_same_row_resume_requires_one_exact_selected_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = _emitted_launch(tmp_path)
    rows = (
        launch.LaunchRow("a", "run-a", object()),
        launch.LaunchRow("b", "run-b", object()),
    )
    monkeypatch.setattr(launch, "compile_authored_training_intent", lambda _launch: rows)

    with pytest.raises(ValueError, match="requires exactly one selected row"):
        launch._operational_same_row_resume_bindings(  # type: ignore[attr-defined]
            authored,
            row=None,
            checkpoint_root=None,
        )


def test_operational_same_row_resume_pins_public_typed_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feedbax.training as feedbax_training
    import feedbax.training.checkpoint_custody as checkpoint_custody

    authored = _emitted_launch(tmp_path)
    checkpoint_root = tmp_path / "row-custody"
    run_id = "feedbax-training-run:exact-row"
    transaction_id = "tx-exact"
    digest = "a" * 64
    phase_program = object()
    run_spec = SimpleNamespace(
        method_ref=SimpleNamespace(key="rlrmp/cs_supervised/v1"),
        method_payload=SimpleNamespace(payload={"n_train_batches": 100}),
        artifacts=SimpleNamespace(artifact_root=str(checkpoint_root)),
        worker_execution=SimpleNamespace(
            method_contract=SimpleNamespace(phase_program=phase_program)
        ),
    )
    row = launch.LaunchRow("only", run_id, run_spec)
    prepared = launch._PreparedExecution(  # type: ignore[attr-defined]
        initial_slots={"model": object()},
        kernel_context=object(),
        loss_service=object(),
        resume_slot_transform=object(),
    )
    monkeypatch.setattr(launch, "compile_authored_training_intent", lambda _launch: (row,))
    monkeypatch.setattr(launch, "_prepare_execution", lambda *_args, **_kwargs: prepared)
    loaded = SimpleNamespace(
        manifest=SimpleNamespace(
            transaction_id=transaction_id,
            run_id=run_id,
            completed_training_batches=50,
        )
    )
    latest = SimpleNamespace(
        run_id=run_id,
        transaction_id=transaction_id,
        manifest_sha256=digest,
    )
    documents = SimpleNamespace(
        latest_pointer=SimpleNamespace(document=latest),
        manifest=SimpleNamespace(document=loaded.manifest),
    )
    load_calls: list[tuple[Path, dict[str, object]]] = []

    def load_latest(root: Path, **kwargs: object) -> object:
        load_calls.append((root, kwargs))
        return loaded

    monkeypatch.setattr(checkpoint_custody, "load_latest_checkpoint", load_latest)
    monkeypatch.setattr(
        feedbax_training,
        "load_checkpoint_custody_documents",
        lambda root: documents,
    )

    binding = launch._operational_same_row_resume_bindings(  # type: ignore[attr-defined]
        authored,
        row="only",
        checkpoint_root=None,
    )

    assert binding == {
        "only": {
            "checkpoint_root": str(checkpoint_root),
            "transaction_id": transaction_id,
            "manifest_sha256": digest,
        }
    }
    assert load_calls == [
        (
            checkpoint_root,
            {
                "expected_run_spec": run_spec,
                "expected_phase_program": phase_program,
                "expected_slots": prepared.initial_slots,
                "resume_slot_transform": prepared.resume_slot_transform,
                "continuation_request": None,
                "allow_new_lineage_override": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("run_id", "completed", "message"),
    [
        ("feedbax-training-run:other", 50, "run identity does not match"),
        ("feedbax-training-run:exact-row", 100, "already complete"),
    ],
)
def test_operational_same_row_resume_rejects_wrong_identity_or_complete_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
    completed: int,
    message: str,
) -> None:
    import feedbax.training as feedbax_training
    import feedbax.training.checkpoint_custody as checkpoint_custody

    authored = _emitted_launch(tmp_path)
    expected_run_id = "feedbax-training-run:exact-row"
    run_spec = SimpleNamespace(
        method_ref=SimpleNamespace(key="rlrmp/cs_supervised/v1"),
        method_payload=SimpleNamespace(payload={"n_train_batches": 100}),
        artifacts=SimpleNamespace(artifact_root=str(tmp_path / "custody")),
        worker_execution=SimpleNamespace(method_contract=SimpleNamespace(phase_program=object())),
    )
    row = launch.LaunchRow("only", expected_run_id, run_spec)
    prepared = launch._PreparedExecution(  # type: ignore[attr-defined]
        initial_slots={"model": object()},
        kernel_context=object(),
        loss_service=object(),
        resume_slot_transform=object(),
    )
    manifest = SimpleNamespace(
        transaction_id="tx-exact",
        run_id=run_id,
        completed_training_batches=completed,
    )
    monkeypatch.setattr(launch, "compile_authored_training_intent", lambda _launch: (row,))
    monkeypatch.setattr(launch, "_prepare_execution", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        checkpoint_custody,
        "load_latest_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=manifest),
    )
    monkeypatch.setattr(
        feedbax_training,
        "load_checkpoint_custody_documents",
        lambda _root: SimpleNamespace(
            latest_pointer=SimpleNamespace(
                document=SimpleNamespace(run_id=run_id, manifest_sha256="a" * 64)
            ),
            manifest=SimpleNamespace(document=manifest),
        ),
    )

    with pytest.raises(ValueError, match=message):
        launch._operational_same_row_resume_bindings(  # type: ignore[attr-defined]
            authored,
            row="only",
            checkpoint_root=None,
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


@pytest.mark.parametrize("stop_after_batches", [None, 50])
@pytest.mark.parametrize(
    ("row", "expected_row_ids"),
    [("selected", {"selected"}), (None, {"selected", "other"})],
)
def test_execute_hands_selected_operational_stop_to_typed_conformance_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_after_batches: int | None,
    row: str | None,
    expected_row_ids: set[str],
) -> None:
    from feedbax.orchestration import StageEngine

    authored = SimpleNamespace(
        document=SimpleNamespace(
            rows=(SimpleNamespace(row_id="selected"), SimpleNamespace(row_id="other")),
            fork=None,
            metadata={},
        ),
        repo_root=tmp_path,
    )
    request = object()
    context = object()
    registry = object()
    monkeypatch.setattr(
        launch,
        "_validate_execute_controls",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        launch,
        "build_orchestration_request",
        lambda *_args, **_kwargs: (request, context, registry),
    )
    monkeypatch.setattr(launch, "_run_fork_gate", lambda *_args, **_kwargs: None)

    captured: dict[str, Any] = {}

    class FakeEngine:
        def run(self) -> str:
            return "ran"

    def from_request(received: object, **kwargs: object) -> FakeEngine:
        captured["request"] = received
        captured.update(kwargs)
        return FakeEngine()

    monkeypatch.setattr(StageEngine, "from_request", from_request)

    result = launch.execute_authored_training_intent(
        authored,  # type: ignore[arg-type]
        row=row,
        controls=launch.LaunchRuntimeControls(stop_after_batches=stop_after_batches),
    )

    assert result == "ran"
    assert captured["request"] is request
    runtime_inputs = captured["row_conformance_inputs"]
    if stop_after_batches is None:
        assert runtime_inputs == {}
    else:
        assert set(runtime_inputs) == expected_row_ids
        for inputs in runtime_inputs.values():
            assert inputs.authorized_batch_stop.stop_after_batches == 50
            assert inputs.authorized_batch_stop.reason == "stop_after_batches"


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

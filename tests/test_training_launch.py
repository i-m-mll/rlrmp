"""Focused contract tests for the authored training launch frontend."""

from __future__ import annotations

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


def test_backend_swap_does_not_change_accepted_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_minimal_matrix()), encoding="utf-8")
    loaded = launch.load_authored_training_intent(matrix_path, repo_root=tmp_path)
    compiled = (launch.LaunchRow("only", "run-only", object()),)
    monkeypatch.setattr(launch, "compile_authored_training_intent", lambda _launch: compiled)

    class Interim:
        def execute(
            self, row: launch.LaunchRow, controls: launch.LaunchRuntimeControls
        ) -> str:
            del controls
            return f"executed:{row.row_id}"

    class Assemble:
        def execute(
            self, row: launch.LaunchRow, controls: launch.LaunchRuntimeControls
        ) -> str:
            del controls
            return f"submitted:{row.row_id}"

    accepted_before = launch.accepted_authored_document(matrix_path, repo_root=tmp_path)
    interim = launch.execute_authored_training_intent(loaded, backend=Interim())
    assemble = launch.execute_authored_training_intent(loaded, backend=Assemble())
    accepted_after = launch.accepted_authored_document(matrix_path, repo_root=tmp_path)

    assert accepted_before is accepted_after is True
    assert interim == ("executed:only",)
    assert assemble == ("submitted:only",)


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
    evidence = launch.launch_evidence(
        (launch.LaunchRow("only", "run-only", object()),), controls
    )
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
    monkeypatch.setattr(
        launch, "compile_authored_training_intent", lambda _launch: (row,)
    )
    prepared = launch._PreparedExecution(  # type: ignore[attr-defined]
        initial_slots={"model": object()},
        kernel_context={"runtime": object()},
        loss_service=object(),
        resume_slot_transform=object(),
    )
    backend = launch.TransitionalFeedbaxBackend()
    prepare_calls: list[tuple[launch.LaunchRow, bool]] = []

    def prepare(item: launch.LaunchRow, *, resume: bool) -> object:
        prepare_calls.append((item, resume))
        return prepared

    monkeypatch.setattr(backend, "_prepare", prepare)
    import feedbax.training.checkpoint_custody as custody

    load_calls: list[tuple[Path, dict[str, object]]] = []

    def load_checkpoint(root: Path, **kwargs: object) -> object:
        load_calls.append((root, kwargs))
        return SimpleNamespace(manifest=SimpleNamespace(transaction_id="txn-ok"))

    monkeypatch.setattr(custody, "load_latest_checkpoint", load_checkpoint)
    result = launch.verify_resume_authored_training_intent(  # type: ignore[arg-type]
        object(), backend=backend
    )
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

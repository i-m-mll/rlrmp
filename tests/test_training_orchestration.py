"""Focused Gate-1 contract tests for RLRMP orchestration bindings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyInputDeclaration,
    CompiledExecutionRow,
    CompiledRunSet,
)
from feedbax.orchestration.bundle import RowLaunchSpec
from feedbax.orchestration.conformance import (
    ConformanceRowArtifacts,
    check_checkpoint_cadence,
    check_completed_batches,
)

from rlrmp.train.orchestration_compiler import (
    COMPILER_ID,
    COMPILER_VERSION,
    RlrmpOrchestratedLaunchCompiler,
    _native_training_diagnostics,
    register_orchestrated_training_compiler,
)
from rlrmp.train.orchestration_inputs import (
    CheckpointTransactionInputResolver,
    checkpoint_transaction_locator,
)
from rlrmp.train.orchestration_capabilities import (
    SCHEDULED_CERTIFY_SKIP_REASON,
    SCHEDULED_PREFLIGHT_SKIP_REASON,
    scheduled_certify_capable,
    scheduled_preflight_capable,
)


def test_compiler_registration_and_runnable_execution_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"schema_id": "feedbax.spec.training_run", "schema_version": "v1"}
    stock = CompiledRunSet(
        rows=[
            CompiledExecutionRow(
                row_id="a",
                payload=payload,
                resolved_semantics={"row_id": "a"},
                launch=RowLaunchSpec(command=["incomplete"]),
            )
        ]
    )
    compile_calls: list[dict[str, object]] = []
    lowerer = object()

    def compile_matrix(*_args: object, **kwargs: object) -> CompiledRunSet:
        compile_calls.append(kwargs)
        return stock

    monkeypatch.setattr(
        "rlrmp.train.orchestration_compiler.compile_training_run_matrix", compile_matrix
    )
    monkeypatch.setattr(
        "rlrmp.train.orchestration_compiler.rlrmp_training_row_lowerer",
        lambda *_args, **_kwargs: lowerer,
    )
    request = SimpleNamespace(metadata={"row_selection": {"row_ids": ["a"]}})
    compiled = RlrmpOrchestratedLaunchCompiler().compile(
        request,
        authored={},
        run_set_id="set",
        context=SimpleNamespace(repo_root=Path.cwd()),
    )
    row = compiled.rows[0]
    assert row.launch.command == [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "rlrmp.train.orchestrated_row",
        "--packet",
        "{packet_path}",
    ]
    assert row.launch.collect == [
        "manifest.json",
        "training-diagnostics.json",
        "training_summary.json",
    ]
    assert "resume" not in row.payload
    assert row.payload is stock.rows[0].payload
    assert compile_calls[0]["row_lowerer"] is lowerer
    assert compile_calls[0]["row_validator"] is not None
    assert row.launch.metadata["native_training_diagnostics"]["metadata"] == {
        "row_id": "a",
        "run_set_id": "set",
    }
    registry = AssemblyCompilerRegistry()
    register_orchestrated_training_compiler(registry)
    assert COMPILER_ID == "rlrmp.orchestrated_launch"
    assert COMPILER_VERSION == "v1"


def test_checkpoint_input_resolver_validates_recorded_manifest(tmp_path: Path) -> None:
    transaction = "tx-abc"
    manifest = tmp_path / "transactions" / transaction / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"transaction_id":"tx-abc"}\n', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    declaration = AssemblyInputDeclaration(
        role="source_checkpoint",
        kind="checkpoint_transaction",
        locator=checkpoint_transaction_locator(
            tmp_path, transaction_id=transaction, manifest_sha256=digest
        ),
    )
    identity = CheckpointTransactionInputResolver()(declaration)
    assert identity.identifier == "checkpoint-transaction:tx-abc"
    assert identity.digest.value == digest

    bad = declaration.model_copy(
        update={
            "locator": checkpoint_transaction_locator(
                tmp_path, transaction_id=transaction, manifest_sha256="0" * 64
            )
        }
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        CheckpointTransactionInputResolver()(bad)
    with pytest.raises(ValueError, match="latest.json"):
        CheckpointTransactionInputResolver()(
            declaration.model_copy(update={"locator": "checkpoint-transaction://latest.json"})
        )


def test_locked_segment_projection_passes_completed_and_cadence_checks() -> None:
    diagnostics = {
        "completed_batches": 200,
        "checkpoint_coordinates": [100, 200],
        "segment_checkpoint_program_steps": [1, 2],
        "custody_checkpoint_program_steps": [25, 26],
        "absolute_completed_batches": [12100, 12200],
        "resume_context": {
            "schedule_origin_step": 12000,
            "current_step": 12000,
            "optimizer_count_at_current_step": 0,
        },
        "lr_trace": {"12000": 3e-5, "12050": 3e-5, "12100": 2e-5, "12200": 1e-5},
    }
    row = ConformanceRowArtifacts(
        row_id="continuation",
        bundle_row_spec={"n_batches": 200, "checkpoint_interval": 100},
        training_diagnostics=diagnostics,
    )
    assert check_completed_batches(row).status == "pass"
    assert check_checkpoint_cadence(row).status == "pass"
    assert min(map(int, diagnostics["lr_trace"])) >= 12000


def test_compiler_hands_resume_and_seed_context_to_native_diagnostics() -> None:
    authored = {
        "fork": {"lr_continuation": "restart"},
        "metadata": {"source_completed_training_batches": 12_000},
    }
    payload = {
        "training_config": {"n_batches": 12_200},
        "checkpoint_progress": {
            "checkpoint_interval": 100,
            "continuation": {
                "source_completed_batches": 12_000,
                "additional_batches": 200,
            },
        },
    }
    diagnostics = _native_training_diagnostics(
        authored,
        payload,
        run_set_id="run-set",
        row_id="continuation",
        seed=42,
    )

    assert diagnostics.seeds == [42]
    assert diagnostics.resume_context is not None
    assert diagnostics.resume_context.current_step == 12_000
    assert diagnostics.resume_context.schedule_origin_step == 12_000
    assert diagnostics.optimizer_build_context == diagnostics.resume_context
    assert diagnostics.metadata == {"run_set_id": "run-set", "row_id": "continuation"}


@pytest.mark.skipif(not scheduled_preflight_capable(), reason=SCHEDULED_PREFLIGHT_SKIP_REASON)
def test_lane0_discovers_controller_optimizer_and_metadata_contexts() -> None:
    assert scheduled_preflight_capable()


@pytest.mark.skipif(not scheduled_certify_capable(), reason=SCHEDULED_CERTIFY_SKIP_REASON)
def test_lane0_certify_discovers_controller_optimizer() -> None:
    assert scheduled_certify_capable()

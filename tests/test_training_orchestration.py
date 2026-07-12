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
    check_seeds,
)

from rlrmp.train.orchestration_compiler import (
    COMPILER_ID,
    COMPILER_VERSION,
    RlrmpOrchestratedLaunchCompiler,
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
from rlrmp.train.orchestrated_row import _training_diagnostics


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
    monkeypatch.setattr(
        "rlrmp.train.orchestration_compiler.compile_training_run_matrix",
        lambda *args, **kwargs: stock,
    )
    request = SimpleNamespace(metadata={"row_selection": {"row_ids": ["a"]}})
    compiled = RlrmpOrchestratedLaunchCompiler().compile(
        request, authored={}, run_set_id="set", context=object()
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


def test_real_entrypoint_diagnostics_pass_absolute_batches_and_scalar_seed_checks() -> None:
    payload = {
        "training_config": {"n_batches": 12_200},
        "checkpoint_progress": {
            "checkpoint_interval": 100,
            "continuation": {
                "source_completed_batches": 12_000,
                "additional_batches": 200,
            },
        },
        "metadata": {
            "seed": 42,
            "resume_context": {
                "schedule_origin_step": 12_000,
                "current_step": 12_000,
                "optimizer_count_at_current_step": 0,
            },
            "optimizer_build_context": {
                "schedule_origin_step": 12_000,
                "current_step": 12_000,
                "optimizer_count_at_current_step": 0,
            },
        },
    }
    writes = [
        SimpleNamespace(
            manifest=SimpleNamespace(
                completed_coordinate=SimpleNamespace(program_step=program_step)
            )
        )
        for program_step in (25, 26)
    ]
    diagnostics = _training_diagnostics(
        SimpleNamespace(payload=payload),
        SimpleNamespace(checkpoint_writes=writes),
    )
    row = ConformanceRowArtifacts(
        row_id="continuation",
        bundle_row_spec=payload,
        training_diagnostics=diagnostics,
    )

    assert diagnostics["completed_batches"] == 12_200
    assert diagnostics["segment_completed_batches"] == 200
    assert diagnostics["checkpoint_coordinates"] == [100, 200]
    assert diagnostics["absolute_completed_batches"] == [12_100, 12_200]
    assert diagnostics["segment_checkpoint_program_steps"] == [1, 2]
    assert diagnostics["custody_checkpoint_program_steps"] == [25, 26]
    assert diagnostics["seeds"] == 42
    assert "raw_checkpoint_program_steps" not in diagnostics
    assert check_completed_batches(row).status == "pass"
    assert check_seeds(row).status == "pass"


@pytest.mark.skipif(not scheduled_preflight_capable(), reason=SCHEDULED_PREFLIGHT_SKIP_REASON)
def test_lane0_discovers_controller_optimizer_and_metadata_contexts() -> None:
    assert scheduled_preflight_capable()


@pytest.mark.skipif(not scheduled_certify_capable(), reason=SCHEDULED_CERTIFY_SKIP_REASON)
def test_lane0_certify_discovers_controller_optimizer() -> None:
    assert scheduled_certify_capable()

"""Tests for fail-closed training resume launch controls."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import jax
import pytest
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.spec_storage import training_run_execution_hash
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    BudgetPolicy,
    EnvironmentDeclaration,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ResolvedSnapshotRef,
    RunBundle,
    RunRowSpec,
    RowLaunchSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.stages import run_preflight_checks
from feedbax.training.checkpoint_custody import (
    CheckpointIntegrityError,
    write_checkpoint_transaction,
)

from rlrmp.io import load_named_python_module
from rlrmp.train.executor.cs_supervised import (
    _cs_supervised_execution_registry,
    _adaptive_runtime_template_inputs,
    build_cs_supervised_native_initial_slots,
    build_execution_context_from_spec,
)
from rlrmp.train.executor import cs_supervised as cs_supervised_module
from rlrmp.train import minimax_resume as minimax_resume_module
from rlrmp.train.resume_control import (
    LAUNCH_CONTINUATION_PREFIX,
    LaunchContinuation,
    attach_cs_supervised_checkpoint_continuation,
    completed_batches_from_latest,
    declare_cs_supervised_checkpoint_continuation,
    emit_launch_continuation,
    resolve_launch_continuation,
)
from rlrmp.train.training_configs import MinimaxConfig
from rlrmp.runtime.training_run_specs import (
    build_feedbax_training_run_spec,
    feedbax_training_run_spec_from_payload,
    hydrate_compact_run_spec_envelope,
)
from rlrmp.runtime.checkpoint_custody import cs_custody_training_spec


def _baseline_recipe_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )


def _schedule_preflight_bundle(training_spec: TrainingRunSpec, tmp_path: Path) -> RunBundle:
    payload = training_spec.model_dump(mode="json", exclude_none=True)
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_path = tmp_path / "continued-row.json"
    payload_path.write_bytes(payload_bytes)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    resolved_root = "1" * 64
    return RunBundle(
        run_set_id="2026-07-13-ebd5d02-resume",
        driver="local",
        rows=[
            RunRowSpec(
                row_id="continued-cs-row",
                execution=ExecutionIdentityEnvelope(
                    payload=SchemaArtifactRef(
                        schema_id=payload["schema_id"],
                        schema_version=payload["schema_version"],
                        artifact_id="test:continued-cs-row:payload",
                        sha256=payload_hash,
                        uri=str(payload_path),
                    ),
                    authored_intent=AuthoredIntentRef(
                        schema_id="feedbax.spec.training_run_matrix",
                        schema_version="feedbax.spec.training_run_matrix.v3",
                        artifact_id="test:continued-cs-row:authored",
                        sha256="2" * 64,
                        intent_hash="3" * 64,
                    ),
                    resolved_snapshot=ResolvedSnapshotRef(
                        schema_id="feedbax.spec.training_run_resolved_semantics",
                        schema_version="feedbax.spec.training_run_resolved_semantics.v1",
                        artifact_id="test:continued-cs-row:resolved",
                        sha256="4" * 64,
                        root_hash=resolved_root,
                    ),
                    execution_capsule=ExecutionCapsuleRef(
                        schema_id="feedbax.manifest.training_run_execution_capsule",
                        schema_version="feedbax.manifest.training_run_execution_capsule.v2",
                        artifact_id="test:continued-cs-row:capsule",
                        sha256="5" * 64,
                        execution_hash=training_run_execution_hash(resolved_root, []),
                    ),
                    immutable_inputs=[],
                ),
                launch=RowLaunchSpec(command=["true"]),
            )
        ],
        environment=EnvironmentDeclaration(python_version="3.13"),
        budget=BudgetPolicy(max_wall_clock_seconds=10.0),
        orchestration_root=str(tmp_path / "orchestration"),
    )


@pytest.fixture(scope="module")
def custody_checkpoint_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize one valid custody source for typed-document canaries."""

    recipe_path = _baseline_recipe_path()
    context = build_execution_context_from_spec(recipe_path)
    source_spec = feedbax_training_run_spec_from_payload(context.run_spec)
    source_slots, _runtime = build_cs_supervised_native_initial_slots(
        run_spec=context.run_spec,
        hps=context.hps,
        args=context.args,
        key=jax.random.PRNGKey(int(context.args.seed)),
    )
    root = tmp_path_factory.mktemp("typed-custody") / "source"
    write_checkpoint_transaction(
        root,
        run_spec=source_spec,
        phase_program=source_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_train_chunk",
        coordinate=ProgressCoordinate(
            run_id="typed-custody-source",
            phase="train_chunk",
            program_step=24,
            completed_barrier="after_train_chunk",
        ),
        slots=source_slots,
        completed_training_batches=12_000,
    )
    return root


def _copy_custody_documents(source_root: Path, target_root: Path) -> tuple[Path, Path]:
    """Copy just the published pointer and manifest for reader-only mutation tests."""

    pointer_path = source_root / "latest.json"
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_relative_path = Path(pointer_payload["manifest_relative_path"])
    target_pointer = target_root / "latest.json"
    target_manifest = target_root / manifest_relative_path
    target_manifest.parent.mkdir(parents=True)
    shutil.copyfile(source_root / manifest_relative_path, target_manifest)
    target_pointer.parent.mkdir(parents=True, exist_ok=True)
    target_pointer.write_text(json.dumps(pointer_payload), encoding="utf-8")
    return target_pointer, target_manifest


def test_resume_without_latest_json_is_hard_error(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "row" / "checkpoints"

    with pytest.raises(FileNotFoundError, match=r"checkpoints/latest\.json"):
        resolve_launch_continuation(
            checkpoint_root=checkpoint_root,
            resume_requested=True,
            allow_fresh_start=False,
            stop_target_batches=12_500,
        )


def test_allow_fresh_start_override_emits_fresh_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    continuation = resolve_launch_continuation(
        checkpoint_root=tmp_path / "checkpoints",
        resume_requested=True,
        allow_fresh_start=True,
        stop_target_batches=500,
    )

    with caplog.at_level(logging.INFO, logger="rlrmp.tests.resume"):
        emit_launch_continuation(continuation, logger=logging.getLogger("rlrmp.tests.resume"))

    line = capsys.readouterr().out.strip()
    assert continuation.resume is False
    assert line.startswith(f"{LAUNCH_CONTINUATION_PREFIX} ")
    assert "resume_source=fresh-start-override" in line
    assert "completed_batches=0" in line
    assert "stop_target_batches=500" in line
    assert "continuation_batches=500" in line
    assert caplog.records[-1].message == line


def test_resume_uses_manifest_total_not_custody_order_coordinate(
    custody_checkpoint_root: Path,
) -> None:
    continuation = resolve_launch_continuation(
        checkpoint_root=custody_checkpoint_root,
        resume_requested=True,
        allow_fresh_start=False,
        stop_target_batches=12_200,
    )

    assert continuation.completed_batches == 12_000
    assert continuation.stop_target_batches == 12_200
    assert continuation.continuation_batches == 200


def test_resume_rejects_manifest_pointer_escaping_checkpoint_root(
    tmp_path: Path,
    custody_checkpoint_root: Path,
) -> None:
    latest_path, _manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    pointer["manifest_relative_path"] = "../outside/manifest.json"
    latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="escapes custody root"):
        completed_batches_from_latest(latest_path)


@pytest.mark.parametrize(
    ("document", "field", "value"),
    [
        ("latest", "schema_id", "rlrmp.invalid.latest"),
        ("latest", "schema_version", "rlrmp.invalid.latest.v1"),
        ("manifest", "schema_id", "rlrmp.invalid.manifest"),
        ("manifest", "schema_version", "rlrmp.invalid.manifest.v1"),
    ],
)
def test_resume_rejects_invalid_typed_custody_schema_identity_or_version(
    tmp_path: Path,
    custody_checkpoint_root: Path,
    document: str,
    field: str,
    value: str,
) -> None:
    latest_path, manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    path = latest_path if document == "latest" else manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="invalid|unsupported"):
        completed_batches_from_latest(latest_path)


def test_resume_rejects_pointer_manifest_batch_total_disagreement(
    tmp_path: Path,
    custody_checkpoint_root: Path,
) -> None:
    latest_path, _manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    pointer["completed_training_batches"] = 24
    latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="completed_training_batches disagrees"):
        completed_batches_from_latest(latest_path)


def test_resume_uses_feedbax_legacy_global_step_migration(
    tmp_path: Path,
    custody_checkpoint_root: Path,
) -> None:
    latest_path, _manifest_path = _copy_custody_documents(
        custody_checkpoint_root,
        tmp_path / "checkpoints",
    )
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    pointer["schema_version"] = "feedbax.manifest.training_checkpoint_latest_pointer.v2"
    coordinate = pointer["completed_coordinate"]
    coordinate["global_step"] = coordinate.pop("program_step")
    latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    assert completed_batches_from_latest(latest_path) == 12_000


def test_non_positive_continuation_is_hard_error(
    custody_checkpoint_root: Path,
) -> None:

    with pytest.raises(ValueError, match="non-positive launch continuation"):
        resolve_launch_continuation(
            checkpoint_root=custody_checkpoint_root,
            resume_requested=True,
            allow_fresh_start=False,
            stop_target_batches=12_000,
        )


def test_cs_supervised_continuation_declares_actual_baseline_diagnostic_paths() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(recipe_path.read_text(encoding="utf-8"))
    )
    continuation = LaunchContinuation(
        resume=True,
        resume_source="/tmp/checkpoints/latest.json",
        completed_batches=12_000,
        stop_target_batches=12_200,
        continuation_batches=200,
    )

    attached = attach_cs_supervised_checkpoint_continuation(training_spec, continuation)

    request = attached.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.additional_batches == 200
    assert request.target_total == 12_200


def test_cs_supervised_continuation_does_not_change_exact_parity_spec() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(recipe_path.read_text(encoding="utf-8"))
    )
    fresh = LaunchContinuation(
        resume=False,
        resume_source="fresh-start",
        completed_batches=0,
        stop_target_batches=12_000,
        continuation_batches=12_000,
    )

    assert attach_cs_supervised_checkpoint_continuation(training_spec, fresh) is training_spec


def test_target_bound_launch_fork_still_attaches_runnable_continuation() -> None:
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(_baseline_recipe_path().read_text(encoding="utf-8"))
    )
    launch_fork = LaunchContinuation(
        resume=True,
        resume_source="/tmp/launch-fork/latest.json",
        completed_batches=12_000,
        stop_target_batches=12_200,
        continuation_batches=200,
        source_target_batches=12_200,
    )

    attached = attach_cs_supervised_checkpoint_continuation(training_spec, launch_fork)

    request = attached.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.additional_batches == 200
    assert request.target_total == 12_200


def test_cs_supervised_resume_registry_uses_attached_custody_contract() -> None:
    recipe_path = Path(__file__).resolve().parents[1] / "results/cb3685a/runs/seam_probe.json"
    context = build_execution_context_from_spec(recipe_path)
    continuation = LaunchContinuation(
        resume=True,
        resume_source="/tmp/checkpoints/latest.json",
        completed_batches=12_000,
        stop_target_batches=12_200,
        continuation_batches=200,
    )
    training_spec = attach_cs_supervised_checkpoint_continuation(
        cs_custody_training_spec(context.run_spec),
        continuation,
    )

    registry = _cs_supervised_execution_registry(training_spec)
    execution_contract = registry.resolve(
        training_spec.method_ref, path="/method_ref"
    ).contract_factory()

    assert execution_contract == training_spec.worker_execution.method_contract
    request = training_spec.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.additional_batches == 200
    assert request.target_total == 12_200
    barrier = execution_contract.phase_program.checkpoint_barriers[0]
    assert {slot.slot for slot in barrier.slots} == {
        "model",
        "optimizer",
        "prng",
        "completed_batches",
        "history",
        "adversary_policy",
        "adversary_optimizer",
        "adaptive_epsilon_state",
        "checkpoint_metadata",
    }


@pytest.mark.parametrize("continuation_batches", [4_500, 1])
def test_adaptive_runtime_template_args_are_segment_local(
    continuation_batches: int,
) -> None:
    governed_args = Namespace(n_train_batches=16_500)
    governed_hps = Namespace(n_batches_condition=16_500)
    continuation = LaunchContinuation(
        resume=True,
        resume_source="/tmp/checkpoints/latest.json",
        completed_batches=12_000,
        stop_target_batches=12_000 + continuation_batches,
        continuation_batches=continuation_batches,
    )

    runtime_template_args, runtime_template_hps = _adaptive_runtime_template_inputs(
        governed_args,
        governed_hps,
        continuation,
    )

    assert governed_args.n_train_batches == 16_500
    assert governed_hps.n_batches_condition == 16_500
    assert runtime_template_args.n_train_batches == continuation_batches
    assert runtime_template_hps.n_batches_condition == continuation_batches


def test_stage2_authoring_declares_12000_to_16500_total_horizon() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "results/cb3685a/runs/harmonized_nominal_h0_const_band16_lr3e-3_clip5_b64.json"
    )
    training_spec = feedbax_training_run_spec_from_payload(
        json.loads(recipe_path.read_text(encoding="utf-8"))
    )

    attached = declare_cs_supervised_checkpoint_continuation(
        training_spec,
        source_completed_batches=12_000,
        target_total_batches=16_500,
    )

    request = attached.checkpoint_progress.continuation
    assert request is not None
    assert request.source_completed_batches == 12_000
    assert request.target_total == 16_500
    assert request.additional_batches == 4_500


def test_continued_cs_row_drops_fresh_context_and_preflight_fails_closed(
    tmp_path: Path,
) -> None:
    tracked = json.loads(_baseline_recipe_path().read_text(encoding="utf-8"))
    embedded = feedbax_training_run_spec_from_payload(tracked)
    run_spec = hydrate_compact_run_spec_envelope(tracked)
    assert embedded.graph.inline is not None
    training_spec = build_feedbax_training_run_spec(
        run_spec,
        graph_spec=GraphSpec.model_validate(embedded.graph.inline),
        output_dir=tmp_path / "artifacts",
        spec_dir=tmp_path / "spec",
    )
    assert training_spec.metadata["resume_context"]["current_step"] == 0
    assert training_spec.metadata["optimizer_build_context"]["current_step"] == 0

    attached = declare_cs_supervised_checkpoint_continuation(
        training_spec,
        source_completed_batches=12_000,
        target_total_batches=12_200,
    )

    assert "resume_context" not in attached.metadata
    assert "optimizer_build_context" not in attached.metadata
    checks = {
        check.name: check
        for check in run_preflight_checks(_schedule_preflight_bundle(attached, tmp_path))
    }
    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "fail"
    assert "resume_context missing" in (schedule_check.detail or "")
    assert schedule_check.observed == {"continued-cs-row": []}


@pytest.mark.parametrize("loader_error", [None, ValueError("checkpoint binding mismatch")])
def test_verify_resume_only_loads_strict_checkpoint_without_executor_steps(
    monkeypatch: pytest.MonkeyPatch,
    loader_error: Exception | None,
) -> None:
    recipe_path = _baseline_recipe_path()
    context = build_execution_context_from_spec(recipe_path, resume=True)
    continuation = LaunchContinuation(
        resume=True,
        resume_source="/tmp/checkpoints/latest.json",
        completed_batches=11_000,
        stop_target_batches=12_000,
        continuation_batches=1_000,
        source_target_batches=12_000,
    )
    monkeypatch.setattr(
        cs_supervised_module,
        "_resolve_full_train_launch_context",
        lambda resolved_context, **_kwargs: (resolved_context, True, continuation),
    )
    monkeypatch.setattr(
        cs_supervised_module,
        "build_cs_supervised_native_initial_slots",
        lambda **_kwargs: ({"model": object(), "completed_batches": 0}, object()),
    )
    monkeypatch.setattr(
        cs_supervised_module,
        "_cs_supervised_resume_slot_transform",
        lambda: None,
    )
    calls: list[dict[str, object]] = []

    def load_checkpoint(_root: Path, **kwargs: object) -> object:
        calls.append(kwargs)
        if loader_error is not None:
            raise loader_error
        return SimpleNamespace(manifest=SimpleNamespace(transaction_id="txn-ok"))

    monkeypatch.setattr(cs_supervised_module, "load_feedbax_checkpoint", load_checkpoint)

    if loader_error is not None:
        with pytest.raises(ValueError, match="checkpoint binding mismatch"):
            cs_supervised_module.verify_resume_from_context(context)
        return

    result = cs_supervised_module.verify_resume_from_context(context)
    assert result["verified_resume"] is True
    assert result["transaction_id"] == "txn-ok"
    assert len(calls) == 1
    assert "expected_run_spec" in calls[0]
    assert "expected_phase_program" in calls[0]
    assert "expected_slots" in calls[0]


def test_launch_interface_exposes_checkpoint_only_resume_gate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    launch_script = load_named_python_module(
        "launch_training_under_test",
        repo_root / "scripts/launch_training.py",
    )
    verify_args = launch_script.build_parser().parse_args(
        ["verify-resume", "matrix.json", "--row", "minimax"]
    )
    assert verify_args.command == "verify-resume"
    assert verify_args.row == "minimax"
    assert not hasattr(verify_args, "allow_fresh_start")

    execute_args = launch_script.build_parser().parse_args(
        ["execute", "matrix.json", "--resume", "--allow-fresh-start"]
    )
    assert execute_args.command == "execute"
    assert execute_args.resume is True
    assert execute_args.allow_fresh_start is True

    method = (repo_root / "src/rlrmp/train/minimax_native/method.py").read_text(encoding="utf-8")
    resume = (repo_root / "src/rlrmp/train/minimax_resume.py").read_text(encoding="utf-8")
    assert "load_latest_checkpoint(" not in method
    assert "load_latest_checkpoint(" in resume


def test_minimax_resume_verification_stays_strict_after_relocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    phase_program = object()
    training_spec = SimpleNamespace(
        worker_execution=SimpleNamespace(
            method_contract=SimpleNamespace(phase_program=phase_program)
        )
    )

    class TrainingRunSpecStub:
        @classmethod
        def model_validate(cls, value: object) -> object:
            assert value == {}
            return training_spec

    config = MinimaxConfig(
        output_dir=str(tmp_path / "minimax"),
        n_warmup_batches=4,
        n_adversary_batches=6,
    )
    continuation = LaunchContinuation(
        resume=True,
        resume_source=str(tmp_path / "latest.json"),
        completed_batches=4,
        stop_target_batches=10,
        continuation_batches=6,
        source_target_batches=10,
    )
    pointer = tmp_path / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "completed_coordinate": {
                    "completed_barrier": "after_warmup",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(minimax_resume_module, "TrainingRunSpec", TrainingRunSpecStub)
    monkeypatch.setattr(
        minimax_resume_module,
        "minimax_training_run_spec_to_config",
        lambda _spec: config.model_dump(mode="python"),
    )

    def resolve(**kwargs: object) -> LaunchContinuation:
        completed = kwargs["completed_batches_from_latest"]
        assert callable(completed)
        assert completed(pointer) == 4
        return continuation

    monkeypatch.setattr(minimax_resume_module, "resolve_launch_continuation", resolve)
    monkeypatch.setattr(minimax_resume_module, "emit_launch_continuation", lambda *_a, **_k: None)
    monkeypatch.setattr(minimax_resume_module, "build_hps", lambda _config: object())
    expected_slots = {"controller": object()}
    monkeypatch.setattr(
        minimax_resume_module,
        "build_minimax_native_initial_slots",
        lambda **_kwargs: (expected_slots, object()),
    )
    calls: list[tuple[Path, dict[str, object]]] = []

    def load_checkpoint(root: Path, **kwargs: object) -> object:
        calls.append((root, kwargs))
        return SimpleNamespace(manifest=SimpleNamespace(transaction_id="txn-minimax"))

    monkeypatch.setattr(minimax_resume_module, "load_latest_checkpoint", load_checkpoint)

    result = minimax_resume_module.verify_minimax_checkpoint_resume({})

    assert result["verified_resume"] is True
    assert result["transaction_id"] == "txn-minimax"
    assert result["completed_batches"] == 4
    assert result["continuation_batches"] == 6
    assert calls == [
        (
            Path(config.output_dir) / "checkpoints_adversarial",
            {
                "expected_run_spec": training_spec,
                "expected_phase_program": phase_program,
                "expected_slots": expected_slots,
            },
        )
    ]

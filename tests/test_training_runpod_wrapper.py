"""Transport-level contract tests for the RLRMP RunPod wrapper."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from feedbax.contracts.spec_storage import (
    training_run_execution_hash,
    training_spec_canonical_bytes,
)
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingIdentityAdapter,
)
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompiledExecutionRow,
    CompiledRunSet,
    CompilerIdentity,
    RunAssemblyRequest,
)
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    BudgetPolicy,
    EnvironmentDeclaration,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ImmutableInputDigest,
    ImmutableInputIdentity,
    ResolvedSnapshotRef,
    RowLaunchSpec,
    RunBundle,
    RunRowSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import build_core_check_registry
from feedbax.orchestration.drivers.runpod import CommandResult, RunPodDriverConfig
from feedbax.orchestration.stages import StageEngine
from feedbax.orchestration.state import RowState, RunSetState

from rlrmp.train.fixture_orchestration import (
    execute_fixture_packet,
    fixture_training_run_spec,
    register_fixture_method,
)
from rlrmp.train.orchestration_drivers import RlrmpRunPodDriver


class FakeRunPodTransport:
    """Filesystem-backed transport with no subprocess, network, or pod calls."""

    def __init__(self, remote_root: Path) -> None:
        self.remote_root = remote_root
        self.runpodctl_calls: list[tuple[str, ...]] = []
        self.ssh_commands: list[str] = []
        self.rsync_calls: list[tuple[str, str, bool, tuple[str, ...]]] = []
        self.bundle: RunBundle | None = None
        self._executed_rows: set[str] = set()

    def bind_bundle(self, bundle: RunBundle) -> None:
        """Bind the ASSEMBLE output used to simulate remote row execution."""
        self.bundle = bundle

    def runpodctl(self, *args: str) -> CommandResult:
        self.runpodctl_calls.append(args)
        return CommandResult(0, "{}")

    def image_exists(self, image: str) -> bool:
        return image == "runpod/pytorch:fixture"

    def ssh(self, command: str) -> CommandResult:
        self.ssh_commands.append(command)
        if "rlrmp.train.fixture_orchestration" in command:
            self._execute_launched_row(command)
        if command.startswith("cat ") and command.endswith(".pid'"):
            return CommandResult(0, "4321\n")
        if command.startswith("python - <<'PY'") and self.bundle is not None:
            rows = {
                row.row_id: {"status": "completed", "pid": 4321, "detail": None}
                for row in self.bundle.rows
            }
            return CommandResult(0, json.dumps({"gpu": "fixture", "rows": rows}))
        return CommandResult(0, "")

    def rsync(
        self,
        source: str,
        target: str,
        *,
        delete: bool = False,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        self.rsync_calls.append((source, target, delete, tuple(excludes)))
        source_path = self._path(source)
        target_path = self._path(target)
        if source_path.exists():
            self._copy(source_path, target_path, source.endswith("/"), target.endswith("/"))
        return CommandResult(0, "")

    def remote_path(self, path: str) -> Path:
        return self._path(path)

    def _execute_launched_row(self, command: str) -> None:
        if self.bundle is None:
            raise AssertionError("fake transport must bind the assembled bundle before launch")
        match = re.search(r"FEEDBAX_ROW_ID=([^ ]+)", command)
        if match is None:
            raise AssertionError("launch command omitted FEEDBAX_ROW_ID")
        row_id = match.group(1).strip("'\"")
        if row_id in self._executed_rows:
            return
        fingerprint_match = re.search(r"FEEDBAX_ENV_FINGERPRINT=([^ ]+)", command)
        if fingerprint_match is None:
            raise AssertionError("launch command omitted FEEDBAX_ENV_FINGERPRINT")
        fingerprint = fingerprint_match.group(1).strip("'\"")
        remote_run = f"/workspace/runs/{self.bundle.run_set_id}"
        row_dir = self.remote_path(f"{remote_run}/rows/{row_id}")
        event_dir = self.remote_path(f"{remote_run}/events")
        packet = row_dir / "launch-packet.json"
        environment = {
            "FEEDBAX_RUN_SET_ID": self.bundle.run_set_id,
            "FEEDBAX_ROW_ID": row_id,
            "FEEDBAX_RUN_EVENTS_DIR": str(event_dir),
            "FEEDBAX_ENV_FINGERPRINT": fingerprint,
            "FEEDBAX_ROW_DIR": str(row_dir),
        }
        with patch.dict(os.environ, environment, clear=False):
            execute_fixture_packet(packet)
        self._executed_rows.add(row_id)

    def _path(self, value: str) -> Path:
        stripped = value.rstrip("/")
        if stripped.startswith("/workspace"):
            return self.remote_root / stripped.removeprefix("/workspace/")
        return Path(stripped)

    @staticmethod
    def _copy(source: Path, target: Path, source_dir: bool, target_dir: bool) -> None:
        if source.is_dir():
            if source_dir and target_dir:
                target.mkdir(parents=True, exist_ok=True)
                for child in source.iterdir():
                    destination = target / child.name
                    if child.is_dir():
                        shutil.copytree(child, destination, dirs_exist_ok=True)
                    else:
                        shutil.copy2(child, destination)
            else:
                shutil.copytree(source, target, dirs_exist_ok=True)
            return
        destination = target / source.name if target_dir else target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


@dataclass(frozen=True)
class _FixtureCompiler:
    """Compile one tiny registered training row for the RunPod lifecycle."""

    def compile(
        self,
        request: RunAssemblyRequest,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del request, authored, run_set_id, context
        payload = fixture_training_run_spec().model_dump(mode="json", exclude_none=True)
        return CompiledRunSet(
            rows=[
                CompiledExecutionRow(
                    row_id="warm",
                    payload=payload,
                    resolved_semantics={"row_id": "warm", "training": payload},
                    launch=RowLaunchSpec(
                        command=[
                            "uv",
                            "run",
                            "--no-sync",
                            "python",
                            "-m",
                            "rlrmp.train.fixture_orchestration",
                            "--packet",
                            "{packet_path}",
                        ],
                        collect=[
                            "manifest.json",
                            "training-diagnostics.json",
                            "training_summary.json",
                        ],
                    ),
                )
            ]
        )


def test_stage_engine_runpod_wrapper_registers_with_all_core_checks(tmp_path: Path) -> None:
    register_fixture_method()
    run_set_id = "runpod-full-lifecycle"
    authored = {
        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
        "total_batches": 2,
        "training_config": {"fixture": "runpod-full-lifecycle-v1"},
    }
    authored_path = tmp_path / "full-lifecycle-authored.json"
    authored_path.write_bytes(training_spec_canonical_bytes(authored))
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            artifact_id=f"fixture:{run_set_id}:authored",
            sha256=_sha256(authored_path),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id="rlrmp.tests.runpod-full-lifecycle",
            compiler_version="rlrmp.tests.runpod-full-lifecycle.v1",
        ),
        driver="runpod",
        environment=EnvironmentDeclaration(
            python_version=sys.version.split()[0],
            image_id="runpod/pytorch:fixture",
        ),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(tmp_path),
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=request.compiler.compiler_id,
        compiler_version=request.compiler.compiler_version,
        compiler=_FixtureCompiler(),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    transport = FakeRunPodTransport(tmp_path / "full-lifecycle-remote")
    local_repo = tmp_path / "full-lifecycle-repo"
    local_repo.mkdir()

    def driver_factory(bundle: RunBundle) -> RlrmpRunPodDriver:
        transport.bind_bundle(bundle)
        return RlrmpRunPodDriver(
            config=RunPodDriverConfig(
                ssh_host="198.51.100.10",
                ssh_port=2222,
                image="runpod/pytorch:fixture",
                local_repos={"rlrmp": local_repo},
                remote_repos={"rlrmp": "/workspace/rlrmp"},
                remote_run_root="/workspace/runs",
                overlay_steps=(),
                auto_teardown=False,
            ),
            transport=transport,
        )

    state = StageEngine.from_request(
        request,
        context=AssemblyContext(custody_root=tmp_path / "full-lifecycle-custody"),
        registry=registry,
        driver_factory=driver_factory,
        run_set_id=run_set_id,
        conformance_registry=build_core_check_registry(),
        poll_interval_seconds=0.01,
    ).run()

    run_set_dir = tmp_path / run_set_id
    certificate = json.loads((run_set_dir / "conformance.json").read_text(encoding="utf-8"))
    checks = {entry["check_id"]: entry for entry in certificate["rows"]["warm"]["checks"]}
    assert state.stage("REGISTER").status == "completed"
    assert state.registration_payload is not None
    assert state.registration_payload["status"] == "completed"
    assert certificate["overall"] == "pass"
    assert set(checks) == {
        "checkpoint_cadence",
        "completed_batches",
        "environment_fingerprint",
        "events_terminal",
        "execution_identity",
        "lr_trace",
        "manifest_valid",
        "seeds",
    }
    assert all(entry["status"] == "pass" for entry in checks.values())
    assert all(entry["expected"] is not None for entry in checks.values())
    assert all(entry["observed"] is not None for entry in checks.values())
    assert (run_set_dir / "collected" / "warm" / "manifest.json").is_file()
    assert (run_set_dir / "collected" / "warm" / "training-diagnostics.json").is_file()
    assert (run_set_dir / "collected" / "warm" / "training_summary.json").is_file()
    assert (run_set_dir / "events" / "warm.events.jsonl").is_file()
    assert transport._executed_rows == {"warm"}
    assert transport.runpodctl_calls == []


def test_fake_runpod_transport_covers_packet_launch_and_collection(tmp_path: Path) -> None:
    bundle, fork_record, target_root = _bundle(tmp_path)
    transport = FakeRunPodTransport(tmp_path / "remote")
    local_repo = tmp_path / "repo"
    local_repo.mkdir()
    driver = RlrmpRunPodDriver(
        config=RunPodDriverConfig(
            ssh_host="198.51.100.10",
            ssh_port=2222,
            image="runpod/pytorch:fixture",
            local_repos={"rlrmp": local_repo},
            remote_repos={"rlrmp": "/workspace/rlrmp"},
            remote_run_root="/workspace/runs",
            overlay_steps=(),
            auto_teardown=False,
        ),
        transport=transport,
        resume=True,
        fork_record_path=fork_record,
        fork_record_sha256=_sha256(fork_record),
    )
    state = _state(bundle)

    assert all(check.status == "pass" for check in driver.preflight_checks(bundle))
    assert driver.provision(bundle, state)["provided_endpoint"] is True
    state = state.model_copy(update={"environment_fingerprint": driver.realize_env(bundle, state)})
    staged = driver.stage_inputs(bundle, state)
    launched = driver.launch_row(bundle, bundle.rows[0], state)

    remote_run = "/workspace/runs/run-set"
    remote_row = transport.remote_path(f"{remote_run}/rows/warm")
    remote_events = transport.remote_path(f"{remote_run}/events/warm.events.jsonl")
    remote_row.mkdir(parents=True, exist_ok=True)
    remote_events.parent.mkdir(parents=True, exist_ok=True)
    (remote_row / "manifest.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    (remote_row / "training-diagnostics.json").write_text(
        '{"completed_batches":1}\n', encoding="utf-8"
    )
    (remote_row / "training_summary.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    remote_events.write_text('{"event_type":"terminal","status":"completed"}\n', encoding="utf-8")

    collected = driver.collect(bundle, bundle.rows[0], state)
    teardown = driver.teardown(bundle, state)

    assert staged["rlrmp_packets"] == [
        {
            "row_id": "warm",
            "remote_packet": "/workspace/runs/run-set/rows/warm/launch-packet.json",
        }
    ]
    run_targets = {
        target
        for _source, target, _delete, _excludes in transport.rsync_calls
        if target.startswith(remote_run)
    }
    assert run_targets == {
        "/workspace/runs/run-set/inputs/warm/checkpoint/",
        "/workspace/runs/run-set/inputs/fork-gate.json",
        "/workspace/runs/run-set/rows/warm/identity/payload.json",
        "/workspace/runs/run-set/rows/warm/identity/authored_intent.json",
        "/workspace/runs/run-set/rows/warm/identity/resolved_snapshot.json",
        "/workspace/runs/run-set/rows/warm/identity/execution_capsule.json",
        "/workspace/runs/run-set/rows/warm/launch-packet.json",
    }
    packet_path = bundle.run_set_dir / "packets" / "warm" / "launch-packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["staged_checkpoint_root"] == ("/workspace/runs/run-set/inputs/warm/checkpoint")
    assert packet["fork_record_path"] == "/workspace/runs/run-set/inputs/fork-gate.json"
    assert all(
        ref["uri"].startswith("/workspace/runs/run-set/rows/warm/identity/")
        for ref in (
            packet["envelope"]["payload"],
            packet["envelope"]["authored_intent"],
            packet["envelope"]["resolved_snapshot"],
            packet["envelope"]["execution_capsule"],
        )
    )
    assert any(
        "sha256sum -c -" in command
        and "/inputs/warm/checkpoint/transactions/target-tx/manifest.json" in command
        and _sha256(target_root / "transactions" / "target-tx" / "manifest.json") in command
        for command in transport.ssh_commands
    )
    launch_command = launched["command"]
    assert "FEEDBAX_RUN_SET_ID=run-set" in launch_command
    assert "FEEDBAX_ROW_ID=warm" in launch_command
    assert "FEEDBAX_RUN_EVENTS_DIR=/workspace/runs/run-set/events" in launch_command
    assert "FEEDBAX_ENV_FINGERPRINT=" in launch_command
    assert "rlrmp.train.orchestrated_row" in launch_command
    assert "--packet" in launch_command and "{packet_path}" in launch_command

    certify_dir = bundle.run_set_dir / "collected" / "warm"
    assert {
        "manifest.json",
        "training-diagnostics.json",
        "training_summary.json",
    }.issubset(path.name for path in certify_dir.iterdir())
    canonical_events = bundle.run_set_dir / "events" / "warm.events.jsonl"
    assert canonical_events.is_file()
    assert collected["warm.events.jsonl"] == str(canonical_events)
    assert teardown["teardown"] == "skipped"
    assert transport.runpodctl_calls == []
    assert target_root.is_dir()


def _bundle(tmp_path: Path) -> tuple[RunBundle, Path, Path]:
    source_transaction = "source-tx"
    target_transaction = "target-tx"
    source_manifest = tmp_path / "source" / "transactions" / source_transaction / "manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(
        json.dumps({"transaction_id": source_transaction}) + "\n", encoding="utf-8"
    )
    target_root = tmp_path / "target"
    target_manifest = target_root / "transactions" / target_transaction / "manifest.json"
    target_manifest.parent.mkdir(parents=True)
    target_manifest.write_text(
        json.dumps(
            {
                "transaction_id": target_transaction,
                "segment_lineage": {"parent_transaction_id": source_transaction},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fork_record = tmp_path / "fork-gate.json"
    fork_record.write_text(
        json.dumps(
            {
                "source_input": {
                    "transaction_id": source_transaction,
                    "manifest_sha256": _sha256(source_manifest),
                },
                "targets": [
                    {
                        "row_id": "warm",
                        "checkpoint_root": str(target_root),
                        "transaction_id": target_transaction,
                        "manifest_sha256": _sha256(target_manifest),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    identity = ImmutableInputIdentity(
        role="source_checkpoint",
        kind="checkpoint_transaction",
        identifier=f"checkpoint-transaction:{source_transaction}",
        digest=ImmutableInputDigest(value=_sha256(source_manifest)),
    )
    payload = {
        "schema_id": "feedbax.spec.training_run",
        "schema_version": "feedbax.spec.training_run.v1",
    }
    payload_path = _write_identity(tmp_path, "payload", payload)
    authored_path = _write_identity(tmp_path, "authored", {"kind": "authored"})
    resolved_path = _write_identity(tmp_path, "resolved", {"kind": "resolved"})
    capsule_path = _write_identity(tmp_path, "capsule", {"kind": "capsule"})
    resolved_root = "2" * 64
    execution_hash = training_run_execution_hash(
        resolved_root, [identity.model_dump(mode="json", exclude_none=True)]
    )
    envelope = ExecutionIdentityEnvelope(
        payload=_schema_ref(
            payload_path,
            schema_id="feedbax.spec.training_run",
            schema_version="feedbax.spec.training_run.v1",
        ),
        authored_intent=AuthoredIntentRef(
            **_schema_ref(authored_path).model_dump(), intent_hash="1" * 64
        ),
        resolved_snapshot=ResolvedSnapshotRef(
            **_schema_ref(resolved_path).model_dump(), root_hash=resolved_root
        ),
        execution_capsule=ExecutionCapsuleRef(
            **_schema_ref(capsule_path).model_dump(), execution_hash=execution_hash
        ),
        immutable_inputs=[identity],
    )
    row = RunRowSpec(
        row_id="warm",
        execution=envelope,
        launch=RowLaunchSpec(
            command=[
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "rlrmp.train.orchestrated_row",
                "--packet",
                "{packet_path}",
            ],
            collect=[
                "rows/warm/manifest.json",
                "rows/warm/training-diagnostics.json",
                "rows/warm/training_summary.json",
            ],
        ),
    )
    return (
        RunBundle(
            run_set_id="run-set",
            driver="runpod",
            rows=[row],
            environment=EnvironmentDeclaration(image_id="runpod/pytorch:fixture"),
            budget=BudgetPolicy(max_wall_clock_seconds=30),
            orchestration_root=str(tmp_path / "orchestration"),
        ),
        fork_record,
        target_root,
    )


def _write_identity(tmp_path: Path, name: str, value: dict[str, Any]) -> Path:
    path = tmp_path / "identity" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(training_spec_canonical_bytes(value))
    return path


def _schema_ref(
    path: Path,
    *,
    schema_id: str = "fixture.identity",
    schema_version: str = "fixture.identity.v1",
) -> SchemaArtifactRef:
    digest = _sha256(path)
    return SchemaArtifactRef(
        schema_id=schema_id,
        schema_version=schema_version,
        artifact_id=f"artifact://sha256/{digest}",
        sha256=digest,
        uri=str(path),
    )


def _state(bundle: RunBundle) -> RunSetState:
    return RunSetState(
        run_set_id=bundle.run_set_id,
        rows={row.row_id: RowState() for row in bundle.rows},
        environment_fingerprint="fixture-fingerprint",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

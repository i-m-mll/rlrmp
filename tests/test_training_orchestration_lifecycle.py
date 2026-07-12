from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from feedbax.contracts.spec_storage import training_spec_canonical_bytes
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingIdentityAdapter,
)
from feedbax.contracts.training import TrainingRunSpec
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompiledExecutionRow,
    CompiledRunSet,
    CompilerIdentity,
    RunAssemblyRequest,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RowLaunchSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.conformance import build_core_check_registry
from feedbax.orchestration.drivers.local import LocalOrchestrationDriver
from feedbax.orchestration.events import RunEventReader
from feedbax.orchestration.stages import StageEngine
from feedbax.orchestration.state import RunSetStateStore

from rlrmp.train.fixture_orchestration import (
    fixture_training_run_spec,
    register_fixture_method,
)
from rlrmp.train.orchestration_capabilities import missing_scheduled_capability_reasons


COMPILER_ID = "rlrmp.tests.orchestration-lifecycle"
COMPILER_VERSION = "rlrmp.tests.orchestration-lifecycle.v1"
CORE_CHECK_IDS = {
    "checkpoint_cadence",
    "completed_batches",
    "environment_fingerprint",
    "events_terminal",
    "execution_identity",
    "lr_trace",
    "manifest_valid",
    "seeds",
}


@dataclass(frozen=True)
class _FixtureCompiler:
    row_ids: tuple[str, ...]
    scheduled: bool = False

    def compile(
        self,
        request: RunAssemblyRequest,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del request, authored, run_set_id, context
        rows = []
        for index, row_id in enumerate(self.row_ids):
            payload = _run_spec(seed=17 + index, scheduled=self.scheduled).model_dump(
                mode="json", exclude_none=True
            )
            rows.append(
                CompiledExecutionRow(
                    row_id=row_id,
                    payload=payload,
                    resolved_semantics={"row_id": row_id, "training": payload},
                    launch=RowLaunchSpec(
                        command=[
                            sys.executable,
                            "-m",
                            "rlrmp.train.fixture_orchestration",
                            "--packet",
                            "launch-packet.json",
                        ],
                        collect=[
                            "manifest.json",
                            "training-diagnostics.json",
                            "training_summary.json",
                            "checkpoints",
                        ],
                    ),
                )
            )
        return CompiledRunSet(rows=rows)


def _minimal_graph() -> dict[str, Any]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _run_spec(*, seed: int, scheduled: bool = False) -> TrainingRunSpec:
    return fixture_training_run_spec(seed=seed, scheduled=scheduled)


def _assembly_parts(
    tmp_path: Path,
    *,
    row_ids: tuple[str, ...],
    run_set_id: str,
    scheduled: bool = False,
) -> tuple[RunAssemblyRequest, AssemblyContext, AssemblyCompilerRegistry]:
    authored = {
        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
        "total_batches": 2,
        "training_config": {"fixture": "rlrmp-orchestration-lifecycle-v1"},
    }
    authored_bytes = training_spec_canonical_bytes(authored)
    authored_path = tmp_path / "authored.json"
    authored_path.write_bytes(authored_bytes)
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            artifact_id=f"fixture:{run_set_id}:authored",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=COMPILER_ID,
            compiler_version=COMPILER_VERSION,
        ),
        driver="local",
        environment=EnvironmentDeclaration(python_version=sys.version.split()[0]),
        launch_policy=LaunchPolicy(max_parallel_rows=2, warm_first=True),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(tmp_path),
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=COMPILER_ID,
        compiler_version=COMPILER_VERSION,
        compiler=_FixtureCompiler(row_ids, scheduled=scheduled),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    return request, AssemblyContext(custody_root=tmp_path / "custody"), registry


def _write_packets(bundle: Any, *, stop_after_batches: int | None) -> None:
    for row in bundle.rows:
        row_dir = bundle.run_set_dir / "rows" / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(Path(row.execution.payload.uri).read_text(encoding="utf-8"))
        packet = {
            "run_set_id": bundle.run_set_id,
            "row_id": row.row_id,
            "envelope": row.execution.model_dump(mode="json", exclude_none=True),
            "payload": payload,
            "stop_after_batches": stop_after_batches,
        }
        (row_dir / "launch-packet.json").write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _engine(
    tmp_path: Path,
    *,
    row_ids: tuple[str, ...],
    run_set_id: str,
    stop_after_batches: int | None = None,
    scheduled: bool = False,
) -> StageEngine:
    register_fixture_method()
    request, context, registry = _assembly_parts(
        tmp_path, row_ids=row_ids, run_set_id=run_set_id, scheduled=scheduled
    )

    def driver_factory(bundle: Any) -> LocalOrchestrationDriver:
        _write_packets(bundle, stop_after_batches=stop_after_batches)
        return LocalOrchestrationDriver(cwd=Path.cwd(), freeze_lines=("rlrmp==fixture",))

    return StageEngine.from_request(
        request,
        context=context,
        registry=registry,
        driver_factory=driver_factory,
        run_set_id=run_set_id,
        conformance_registry=build_core_check_registry(),
        poll_interval_seconds=0.01,
    )


def test_gate_2_two_row_local_lifecycle_registers_with_all_core_checks(
    tmp_path: Path,
) -> None:
    run_set_id = "gate-2-local-lifecycle"
    state = _engine(
        tmp_path,
        row_ids=("warm", "second"),
        run_set_id=run_set_id,
    ).run()
    run_set_dir = tmp_path / run_set_id
    certificate = json.loads((run_set_dir / "conformance.json").read_text(encoding="utf-8"))

    assert state.stage("REGISTER").status == "completed"
    assert state.registration_payload is not None
    assert state.registration_payload["status"] == "completed"
    assert certificate["overall"] == "pass"
    for row_id in ("warm", "second"):
        checks = {entry["check_id"]: entry for entry in certificate["rows"][row_id]["checks"]}
        assert set(checks) == CORE_CHECK_IDS
        assert all(entry["status"] == "pass" for entry in checks.values())
        assert all(entry["expected"] is not None for entry in checks.values())
        assert all(entry["observed"] is not None for entry in checks.values())

    warm_events = RunEventReader(run_set_dir / "events" / "warm.events.jsonl").read_all()
    ready = next(event for event in warm_events if event.type == "ready")
    second_started = float((run_set_dir / "sentinels" / "second.started").read_text())
    assert second_started >= ready.emitted_at_ms / 1000


def test_stop_after_batches_cannot_register_completed(tmp_path: Path) -> None:
    run_set_id = "gate-2-stop-after-batches"
    engine = _engine(
        tmp_path,
        row_ids=("stopped",),
        run_set_id=run_set_id,
        stop_after_batches=1,
    )

    with pytest.raises(ValueError, match="REGISTER cannot emit phase=completed"):
        engine.run()

    run_set_dir = tmp_path / run_set_id
    state = RunSetStateStore(run_set_dir / "state.json").load()
    certificate = json.loads((run_set_dir / "conformance.json").read_text(encoding="utf-8"))
    checks = {entry["check_id"]: entry for entry in certificate["rows"]["stopped"]["checks"]}
    assert state.rows["stopped"].status == "stopped"
    assert state.stage("REGISTER").status == "failed"
    assert state.registration_payload is not None
    assert state.registration_payload["status"] == "failed"
    assert certificate["overall"] == "fail"
    assert checks["completed_batches"]["status"] == "fail"
    assert checks["completed_batches"]["observed"] < checks["completed_batches"]["expected"]


_SCHEDULED_LIFECYCLE_MISSING = missing_scheduled_capability_reasons()


@pytest.mark.skipif(
    bool(_SCHEDULED_LIFECYCLE_MISSING),
    reason="; ".join(_SCHEDULED_LIFECYCLE_MISSING),
)
def test_scheduled_fixture_full_lifecycle_on_lane0(tmp_path: Path) -> None:
    run_set_id = "gate-2-scheduled-lane0"
    state = _engine(
        tmp_path,
        row_ids=("scheduled",),
        run_set_id=run_set_id,
        scheduled=True,
    ).run()
    certificate = json.loads(
        (tmp_path / run_set_id / "conformance.json").read_text(encoding="utf-8")
    )
    checks = certificate["rows"]["scheduled"]["checks"]
    assert state.registration_payload is not None
    assert state.registration_payload["status"] == "completed"
    assert certificate["overall"] == "pass"
    assert all(entry["status"] == "pass" for entry in checks)

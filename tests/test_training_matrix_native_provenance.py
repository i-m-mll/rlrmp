"""Focused identity and custody coverage for RLRMP row re-lowering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.contracts.spec_storage import training_spec_canonical_bytes, training_spec_sha256
from feedbax.orchestration import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    BudgetPolicy,
    CompilerIdentity,
    EnvironmentDeclaration,
    RunAssemblyRequest,
    SchemaArtifactRef,
    assemble_run_bundle,
)

from rlrmp.train.matrix_lowering import RlrmpTrainingAuthoringIntent
from rlrmp.train.orchestration_compiler import (
    COMPILER_ID,
    COMPILER_VERSION,
    register_orchestrated_training_compiler,
)
from rlrmp.train.orchestration_drivers import _packet_for_row
from rlrmp.train.training_configs import CsNominalGruConfig


def _authored_matrix(tmp_path: Path) -> tuple[dict[str, object], Path]:
    intent = RlrmpTrainingAuthoringIntent(
        config=CsNominalGruConfig(
            issue="5816bf0",
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            dry_run=True,
            smoke=True,
            target_relative_multitarget=True,
        )
    ).model_dump(mode="json", exclude_none=True)
    intent_path = tmp_path / "intent.json"
    intent_path.write_bytes(training_spec_canonical_bytes(intent) + b"\n")
    matrix: dict[str, object] = {
        "schema_id": "feedbax.spec.training_run_matrix",
        "schema_version": "feedbax.spec.training_run_matrix.v3",
        "name": "native provenance",
        "base": {
            "kind": "authored_intent",
            "ref": intent_path.name,
            "content_hash": training_spec_sha256(intent),
        },
        "rows": [{"row_id": "science-row", "seed": 13}],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_bytes = training_spec_canonical_bytes(matrix) + b"\n"
    matrix_path.write_bytes(matrix_bytes)
    return matrix, matrix_path


def test_assembly_custodies_lowered_payload_and_exact_row_provenance(
    tmp_path: Path,
) -> None:
    authored, matrix_path = _authored_matrix(tmp_path)
    authored_bytes = matrix_path.read_bytes()
    authored_ref = SchemaArtifactRef(
        schema_id=str(authored["schema_id"]),
        schema_version=str(authored["schema_version"]),
        artifact_id="matrix:native-provenance",
        sha256=hashlib.sha256(authored_bytes).hexdigest(),
        uri=str(matrix_path),
    )
    request = RunAssemblyRequest(
        authored=authored_ref,
        compiler=CompilerIdentity(
            compiler_id=COMPILER_ID,
            compiler_version=COMPILER_VERSION,
        ),
        driver="local",
        environment=EnvironmentDeclaration(python_version="3.13"),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(tmp_path / "orchestration"),
    )
    registry = AssemblyCompilerRegistry()
    register_orchestrated_training_compiler(registry)
    bundle = assemble_run_bundle(
        request,
        run_set_id="run-set-native-provenance",
        context=AssemblyContext(
            custody_root=tmp_path / "custody",
            repo_root=tmp_path,
            materializer_commit="a" * 40,
            dependency_lock_digest="b" * 64,
        ),
        registry=registry,
    )

    assert len(bundle.rows) == 1
    row = bundle.rows[0]
    provenance = row.execution.row_provenance
    assert provenance is not None
    assert provenance.row_id == row.row_id == "science-row"
    assert provenance.seed == 13
    assert provenance.planned_run_id.startswith("feedbax-training-run:")
    assert provenance.axis_coordinates["run_id"] == provenance.planned_run_id
    assert provenance.lowerer_identities[0].lowerer_id == ("rlrmp.train.cs_nominal_gru.authoring")
    assert row.execution.payload.sha256 == provenance.lowered_execution_payload_hash

    lowered = json.loads(Path(row.execution.payload.uri).read_text(encoding="utf-8"))
    assert lowered["schema_id"] == "feedbax.spec.training_run"
    assert "config" not in lowered
    resolved = decode_resolved_snapshot(
        json.loads(Path(row.execution.resolved_snapshot.uri).read_text(encoding="utf-8"))
    )
    assert resolved["payload"] == lowered
    assert resolved["planned_run_id"] == provenance.planned_run_id
    assert resolved["row_provenance"] == provenance.model_dump(mode="json", exclude_none=True)

    packet = _packet_for_row(
        bundle,
        row,
        row_dir=tmp_path / "row",
        resume=False,
        checkpoint_root=None,
        fork_record_path=None,
        fork_record_sha256=None,
        stop_after_batches=None,
    )
    assert packet.payload == lowered
    assert packet.envelope == row.execution
    assert packet.native_training_diagnostics.seeds == [13]
    assert packet.native_training_diagnostics.metadata == {
        "row_id": "science-row",
        "run_set_id": "run-set-native-provenance",
    }

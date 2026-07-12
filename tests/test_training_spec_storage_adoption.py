"""Focused coverage for RLRMP's three-layer training-spec storage path."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    training_run_intent_hash,
)
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.training.run_matrix import materialize_run_matrix

from rlrmp.runtime.checkpoint_fork_gate import load_matrix, register_rlrmp_training_methods
from rlrmp.runtime.spec_storage import (
    emit_rlrmp_training_run_spec_storage,
    migrate_inline_training_run_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_MIGRATION_MATRIX_COMMIT = "edfb3d358565393e58b79a6a26eccbaf406acde0"


def test_c6c5997_matrix_is_compact_and_resolves_from_exact_snapshot(
    tmp_path: Path,
) -> None:
    matrix_path = REPO_ROOT / "results/c6c5997/runs/matrix.json"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))

    assert matrix_path.read_text(encoding="utf-8").count("\n") < 500
    assert payload["base"]["kind"] == "resolved_output"
    historical_run = json.loads(
        (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(encoding="utf-8")
    )
    snapshot = build_resolved_semantics_snapshot(historical_run["feedbax_training_run_spec"])
    assert snapshot["root_hash"] == payload["base"]["resolved_root_hash"]

    temporary_matrix_path = tmp_path / "matrix.json"
    temporary_matrix_path.write_text(json.dumps(payload), encoding="utf-8")
    snapshot_path = tmp_path / payload["base"]["ref"]
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    matrix = load_matrix(temporary_matrix_path)
    register_rlrmp_training_methods()
    materialized = materialize_run_matrix(matrix, repo_root=tmp_path)
    assert [row.row_id for row in materialized.rows] == [
        "flat_3e-5-epsilon-ramp",
        "rewarm_3e-4-epsilon-ramp",
        "rewarm_3e-3-epsilon-ramp",
    ]
    controller_lrs = [
        row.payload["method_payload"]["payload"]["config"]["controller_lr"]
        for row in materialized.rows
    ]
    assert controller_lrs == [3e-5, 3e-4, 3e-3]


def test_rlrmp_emitter_writes_intent_snapshot_and_capsule(tmp_path: Path) -> None:
    legacy_run = json.loads(
        (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(encoding="utf-8")
    )
    base = legacy_run["feedbax_training_run_spec"]
    authored = TrainingRunMatrixSpec.model_validate(
        {
            "name": "storage test",
            "base": {"kind": "resolved_output", "ref": "base.json", "resolved_root_hash": "0" * 64},
            "rows": [{"row_id": "flat_3e-5-epsilon-ramp", "seed": 42}],
        }
    )
    # Replace the fixture with a canonical Feedbax snapshot and matching root.
    snapshot = build_resolved_semantics_snapshot(base)
    (tmp_path / "base.json").write_text(json.dumps(snapshot), encoding="utf-8")
    payload = authored.model_dump(mode="json", exclude_none=True)
    payload["base"]["resolved_root_hash"] = snapshot["root_hash"]
    result = emit_rlrmp_training_run_spec_storage(
        payload,
        repo_root=tmp_path,
        authored_path=tmp_path / "matrix.json",
        custody_root=tmp_path / "custody",
        materializer_commit="a" * 40,
        dependency_lock_path=REPO_ROOT / "uv.lock",
    )

    emitted = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))
    assert result.intent_hash == training_run_intent_hash(emitted)
    assert Path(result.snapshot_artifact.uri).is_file()
    assert Path(result.capsule_artifact.uri).is_file()
    assert result.capsule.resolved_root_hash == result.resolved_root_hash


def test_migration_preserves_exact_pre_migration_inline_semantics(tmp_path: Path) -> None:
    original = json.loads(
        subprocess.run(
            [
                "git",
                "show",
                f"{PRE_MIGRATION_MATRIX_COMMIT}:results/c6c5997/runs/matrix.json",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    original_inline = original["base"]["inline"]
    repo_root = tmp_path / "repo"
    authored_path = repo_root / "results/c6c5997/runs/matrix.json"
    authored_path.parent.mkdir(parents=True)
    authored_path.write_text(json.dumps(original), encoding="utf-8")
    lock_path = repo_root / "uv.lock"
    lock_path.write_bytes((REPO_ROOT / "uv.lock").read_bytes())

    result = migrate_inline_training_run_matrix(
        original,
        repo_root=repo_root,
        authored_path=authored_path,
        custody_root=repo_root / "_artifacts",
        materializer_commit="b" * 40,
        dependency_lock_path=lock_path,
    )

    compact = json.loads(authored_path.read_text(encoding="utf-8"))
    assert compact["base"]["kind"] == "resolved_output"
    base_snapshot = json.loads((repo_root / compact["base"]["ref"]).read_text(encoding="utf-8"))
    assert decode_resolved_snapshot(base_snapshot) == original_inline
    assert base_snapshot["root_hash"] == compact["base"]["resolved_root_hash"]

    resolved_snapshot = json.loads(Path(result.snapshot_artifact.uri).read_text())
    resolved = decode_resolved_snapshot(resolved_snapshot)
    assert resolved["rows"][0]["payload"] == original_inline
    assert result.capsule.resolved_root_hash == resolved_snapshot["root_hash"]

"""Focused coverage for RLRMP's three-layer training-spec storage path."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.migrations import migrate_graph_spec
from feedbax.contracts.spec_storage import (
    build_resolved_semantics_snapshot,
    training_spec_canonical_bytes,
    training_run_intent_hash,
    training_spec_sha256,
)
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY
from feedbax.contracts.resolved_snapshot_decoder import decode_resolved_snapshot
from feedbax.training.run_matrix import materialize_adapted_run_matrix, materialize_run_matrix

from rlrmp.runtime.checkpoint_fork_gate import load_matrix, register_rlrmp_training_methods
from rlrmp.runtime.spec_storage import (
    emit_rlrmp_training_run_spec_storage,
    migrate_inline_training_run_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_MIGRATION_MATRIX_COMMIT = "edfb3d358565393e58b79a6a26eccbaf406acde0"
PRE_3CD018B_COMPACTION_COMMIT = "bce3e4df18f2bdcc71febfabaa925a4c1c16a40f"
REQUIRE_LOCAL_3CD018B_CUSTODY = "RLRMP_REQUIRE_LOCAL_3CD018B_CUSTODY"


def _cold_emitter_environment(tmp_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "src"), environment.get("PYTHONPATH", ""))
    )
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    environment["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "jax-cache")
    environment["FEEDBAX_JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "jax-cache")
    return environment


def _run_emitter_cold(
    *args: str,
    cwd: Path,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/emit_training_run_matrix.py"), *args],
        cwd=cwd,
        env=_cold_emitter_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture(autouse=True)
def _isolate_training_method_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registrations = dict(DEFAULT_TRAINING_METHOD_REGISTRY._registrations)
    monkeypatch.setattr(DEFAULT_TRAINING_METHOD_REGISTRY, "_registrations", registrations)


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


def test_ef9c882_matrix_is_portable_and_resolves_from_tracked_authored_base(
    tmp_path: Path,
) -> None:
    matrix_path = REPO_ROOT / "results/ef9c882/runs/matrix.json"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    base_path = REPO_ROOT / payload["base"]["ref"]
    base = json.loads(base_path.read_text(encoding="utf-8"))

    assert payload["base"]["kind"] == "authored_intent"
    assert payload["base"]["pin_algorithm"] == "canonical_json_v1"
    assert payload["base"]["content_hash"] == training_spec_sha256(base)
    assert len(payload["rows"]) == 17
    assert not any("legacy_run_spec" in row["metadata"] for row in payload["rows"])

    temporary_matrix_path = tmp_path / "results/ef9c882/runs/matrix.json"
    temporary_matrix_path.parent.mkdir(parents=True)
    temporary_matrix_path.write_text(json.dumps(payload), encoding="utf-8")
    temporary_base_path = tmp_path / payload["base"]["ref"]
    temporary_base_path.write_text(json.dumps(base), encoding="utf-8")
    graph_path = REPO_ROOT / base["graph"]["ref"]
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert base["graph"]["metadata"]["pin_algorithm"] == "canonical_json_v1"
    assert base["graph"]["metadata"]["content_hash"] == training_spec_sha256(graph)
    temporary_graph_path = tmp_path / base["graph"]["ref"]
    temporary_graph_path.write_bytes(graph_path.read_bytes())

    matrix = load_matrix(temporary_matrix_path)
    register_rlrmp_training_methods()
    materialized = materialize_run_matrix(matrix, repo_root=tmp_path)
    assert [row.row_id for row in materialized.rows] == [
        row["row_id"] for row in payload["rows"]
    ]
    assert all(
        row.payload["artifacts"]["artifact_root"]
        == f"_artifacts/ef9c882/runs/{row.row_id}"
        for row in materialized.rows
    )


def test_3cd018b_frozen_rows_use_compact_matrices_and_exact_envelope_snapshots(
    tmp_path: Path,
) -> None:
    require_local_custody = os.environ.get(REQUIRE_LOCAL_3CD018B_CUSTODY) == "1"
    run_paths = sorted((REPO_ROOT / "results/3cd018b/runs").glob("*.json"))
    assert [path.stem for path in run_paths] == [
        "const1000",
        "const1750",
        "const250",
        "hold1750_to1000",
        "hold1750_to250",
        "hold3500_to1000",
        "hold3500_to250",
        "ramp3500_to1000",
    ]

    actual_custody_refs_verified = 0
    for matrix_path in run_paths:
        row_id = matrix_path.stem
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        original = json.loads(
            subprocess.run(
                [
                    "git",
                    "show",
                    (
                        f"{PRE_3CD018B_COMPACTION_COMMIT}:"
                        f"results/3cd018b/runs/{row_id}.json"
                    ),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        snapshot = build_resolved_semantics_snapshot(original)
        snapshot_digest = sha256(training_spec_canonical_bytes(snapshot)).hexdigest()

        assert matrix_path.stat().st_size < 16 * 1024
        assert payload["base"] == {
            "kind": "resolved_output",
            "payload_path": "feedbax_training_run_spec",
            "ref": (
                f"_artifacts/spec-storage/sha256/{snapshot_digest[:2]}/"
                f"{snapshot_digest}.json"
            ),
            "resolved_root_hash": snapshot["root_hash"],
            "symbolic_name": f"{row_id}.historical-envelope",
        }
        assert [row["row_id"] for row in payload["rows"]] == [row_id]

        actual_snapshot_path = REPO_ROOT / payload["base"]["ref"]
        if actual_snapshot_path.is_file():
            actual_snapshot_bytes = actual_snapshot_path.read_bytes()
            assert sha256(actual_snapshot_bytes).hexdigest() == actual_snapshot_path.stem
            actual_snapshot = json.loads(actual_snapshot_bytes)
            assert actual_snapshot == snapshot
            assert actual_snapshot["root_hash"] == payload["base"]["resolved_root_hash"]
            assert decode_resolved_snapshot(actual_snapshot) == original
            materialization_root = REPO_ROOT
            actual_custody_refs_verified += 1
        else:
            assert not require_local_custody, (
                f"required local custody snapshot is missing: {actual_snapshot_path}"
            )
            # Clean clones intentionally omit ignored local custody. Rebuild the
            # exact commit-pinned snapshot so the tracked contract remains
            # hermetic while closeout can require the real local refs above.
            reconstructed_path = tmp_path / payload["base"]["ref"]
            reconstructed_path.parent.mkdir(parents=True, exist_ok=True)
            reconstructed_path.write_bytes(training_spec_canonical_bytes(snapshot))
            materialization_root = tmp_path

        assert decode_resolved_snapshot(snapshot) == original
        matrix = load_matrix(matrix_path)
        resolved_rows: list[dict[str, object]] = []

        def validate_frozen_row(row: dict[str, object], actual_row_id: str) -> None:
            assert actual_row_id == row_id
            resolved_rows.append(row)
            return None

        materialized = materialize_adapted_run_matrix(
            matrix,
            repo_root=materialization_root,
            row_validator=validate_frozen_row,
        )
        assert [row.row_id for row in materialized.rows] == [row_id]
        expected_materialized = json.loads(json.dumps(original["feedbax_training_run_spec"]))
        expected_materialized["graph"]["inline"] = migrate_graph_spec(
            expected_materialized["graph"]["inline"],
            path="graph.inline",
        ).payload
        assert resolved_rows == [expected_materialized]
        assert resolved_rows[0]["method_payload"]["schema_version"] == (
            "rlrmp.spec.training_method.adaptive_epsilon_curriculum_payload.v1"
        )
        assert resolved_rows[0]["graph"]["inline"]["schema_version"] == (
            "feedbax.spec.graph.v4"
        )

    if require_local_custody:
        assert actual_custody_refs_verified == len(run_paths)


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


def test_rlrmp_emitter_help_cold_process_reaches_argument_parser(tmp_path: Path) -> None:
    completed = _run_emitter_cold("--help", cwd=tmp_path, tmp_path=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "usage: emit_training_run_matrix.py" in completed.stdout
    assert "Authored matrix JSON to emit." in completed.stdout
    assert "--output OUTPUT" in completed.stdout


def test_rlrmp_emitter_cold_process_writes_governed_storage(tmp_path: Path) -> None:
    legacy_run = json.loads(
        (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(encoding="utf-8")
    )
    snapshot = build_resolved_semantics_snapshot(legacy_run["feedbax_training_run_spec"])
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(snapshot), encoding="utf-8")
    source_path = tmp_path / "matrix.intent.json"
    source_path.write_text(
        json.dumps(
            TrainingRunMatrixSpec.model_validate(
                {
                    "name": "cold storage test",
                    "base": {
                        "kind": "resolved_output",
                        "ref": base_path.name,
                        "resolved_root_hash": snapshot["root_hash"],
                    },
                    "rows": [{"row_id": "flat-3e-5", "seed": 42}],
                }
            ).model_dump(mode="json", exclude_none=True)
        ),
        encoding="utf-8",
    )
    dependency_lock = tmp_path / "uv.lock"
    dependency_lock.write_bytes((REPO_ROOT / "uv.lock").read_bytes())
    output_path = tmp_path / "results/matrix.json"
    custody_root = tmp_path / "custody"

    completed = _run_emitter_cold(
        str(source_path),
        "--output",
        str(output_path),
        "--repo-root",
        str(tmp_path),
        "--custody-root",
        str(custody_root),
        "--dependency-lock",
        str(dependency_lock),
        "--materializer-commit",
        "a" * 40,
        cwd=tmp_path,
        tmp_path=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert output_path.is_file()
    assert output_path.with_suffix(".json.artifact.json").is_file()
    snapshot_artifact = Path(result["storage"]["snapshot_artifact"]["uri"])
    capsule_artifact = Path(result["storage"]["capsule_artifact"]["uri"])
    assert snapshot_artifact.is_file()
    assert capsule_artifact.is_file()
    assert snapshot_artifact.is_relative_to(custody_root)
    assert capsule_artifact.is_relative_to(custody_root)
    assert result["storage"]["capsule"]["materializer_commit"] == "a" * 40
    assert result["authored_artifact"]["uri"] == str(output_path)


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

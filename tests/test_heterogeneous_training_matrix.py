"""Compact authoring and governed-emission contracts for issue 427d0d8."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.spec_storage import training_spec_canonical_bytes
from feedbax.contracts.training import TrainingRunSpec

from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
from rlrmp.train.heterogeneous_training_matrix import (
    ARCHITECTURES,
    COMPACT_ROW_OVERRIDE_PATHS,
    DISTRIBUTIONS,
    author_training_run_matrix,
)
from rlrmp.train.training_configs import CsNominalGruConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/emit_heterogeneous_training_matrix.py"
RUNS_DIR = REPO_ROOT / "results/427d0d8/runs"
FORBIDDEN_EXPANDED_PATHS = {
    "graph",
    "task",
    "objective",
    "method_payload",
    "worker_execution",
}


def _base_intent(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    payload = {
        "schema_id": "rlrmp.spec.training_authoring_intent",
        "schema_version": "rlrmp.spec.training_authoring_intent.v1",
        "config": {
            "issue": "427d0d8",
            "output_dir": "_artifacts/427d0d8/runs/base",
            "spec_dir": "results/427d0d8/runs/base",
            "controller_architecture": "gru",
            "broad_epsilon_pgd_training": False,
        },
    }
    path = tmp_path / "results/427d0d8/runs/base.intent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(training_spec_canonical_bytes(payload) + b"\n")
    return payload, path


def _matrix(tmp_path: Path) -> TrainingRunMatrixSpec:
    payload, path = _base_intent(tmp_path)
    return author_training_run_matrix(
        payload,
        issue="427d0d8",
        base_ref=path,
        repo_root=tmp_path,
    )


def _load_emitter_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("emit_heterogeneous_training_matrix", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_is_small_native_v3_authored_intent(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path)
    payload = matrix.model_dump(mode="json", exclude_none=True)
    encoded = training_spec_canonical_bytes(payload)
    repeated = training_spec_canonical_bytes(
        _matrix(tmp_path).model_dump(mode="json", exclude_none=True)
    )

    assert matrix.schema_id == "feedbax.spec.training_run_matrix"
    assert matrix.schema_version == "feedbax.spec.training_run_matrix.v3"
    assert matrix.base.kind == "authored_intent"
    assert len(encoded) < 16 * 1024
    assert encoded == repeated
    assert matrix.metadata["expanded_payload_patching"] is False
    assert matrix.metadata["required_row_lowering_contract"] == (
        "rlrmp.heterogeneous_cs_architecture.v1"
    )
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_EXPANDED_PATHS:
        assert f'"{forbidden}"' not in serialized


def test_rows_patch_only_typed_architecture_distribution_and_routes(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path)

    assert [row.row_id for row in matrix.rows] == [
        f"{architecture}.{distribution}"
        for architecture in ARCHITECTURES
        for distribution in DISTRIBUTIONS
    ]
    for row in matrix.rows:
        overrides = {override.path: override.value for override in row.overrides}
        assert set(overrides) == COMPACT_ROW_OVERRIDE_PATHS
        assert not FORBIDDEN_EXPANDED_PATHS.intersection(overrides)
        architecture, distribution = row.row_id.split(".", maxsplit=1)
        assert overrides["config.controller_architecture"] == architecture
        assert overrides["config.broad_epsilon_pgd_training"] is (
            distribution == "broad_epsilon_pgd"
        )
        assert overrides["config.output_dir"] == f"_artifacts/427d0d8/runs/{row.row_id}"
        assert overrides["config.spec_dir"] == f"results/427d0d8/runs/{row.row_id}"


def test_matrix_authoring_rejects_a_base_without_architecture_contract(
    tmp_path: Path,
) -> None:
    payload, path = _base_intent(tmp_path)
    del payload["config"]["controller_architecture"]

    with pytest.raises(ValueError, match="config.controller_architecture"):
        author_training_run_matrix(
            payload,
            issue="427d0d8",
            base_ref=path,
            repo_root=tmp_path,
        )


def test_emitter_delegates_all_writes_to_three_layer_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_emitter_module()
    _payload, base_path = _base_intent(tmp_path)
    output_path = tmp_path / "results/427d0d8/runs/matrix.json"
    custody_root = tmp_path / "custody"
    dependency_lock = tmp_path / "uv.lock"
    dependency_lock.write_text("lock", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    sentinel = object()

    monkeypatch.setattr(module, "require_heterogeneous_row_lowering_contract", lambda: None)

    def fake_storage(matrix: TrainingRunMatrixSpec, **kwargs: Any) -> object:
        calls.append({"matrix": matrix, **kwargs})
        return sentinel

    monkeypatch.setattr(module, "emit_rlrmp_training_run_spec_storage", fake_storage)
    result = module.emit_heterogeneous_training_matrix(
        base_intent_path=base_path,
        output_path=output_path,
        issue="427d0d8",
        repo_root=tmp_path,
        custody_root=custody_root,
        dependency_lock_path=dependency_lock,
        materializer_commit="a" * 40,
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["authored_path"] == output_path
    assert calls[0]["custody_root"] == custody_root
    assert calls[0]["dependency_lock_path"] == dependency_lock
    assert calls[0]["materializer_commit"] == "a" * 40
    assert isinstance(calls[0]["matrix"], TrainingRunMatrixSpec)
    assert not output_path.exists()


def test_current_generic_lowerer_dependency_is_explicit() -> None:
    module = _load_emitter_module()
    supported = "controller_architecture" in CsNominalGruConfig.model_fields

    if supported:
        module.require_heterogeneous_row_lowering_contract()
    else:
        with pytest.raises(RuntimeError, match="blocked by 5816bf0 row lowering"):
            module.require_heterogeneous_row_lowering_contract()


def test_complete_content_pinned_bases_are_retained_with_disjoint_roots() -> None:
    register_rlrmp_training_methods()
    retained = sorted(RUNS_DIR.glob("*.training.json"))
    assert [path.name for path in retained] == sorted(
        f"{architecture}.{distribution}.training.json"
        for architecture in ARCHITECTURES
        for distribution in DISTRIBUTIONS
    )

    artifact_roots: set[str] = set()
    manifest_roots: set[str] = set()
    for path in retained:
        spec = TrainingRunSpec.model_validate_json(path.read_text(encoding="utf-8"))
        row_id = path.name.removesuffix(".training.json")
        assert spec.metadata["row_id"] == row_id
        assert spec.metadata["execution_start"] == "fresh"
        assert spec.checkpoint_progress.continuation is None
        assert spec.checkpoint_progress.resume_from is None
        artifact_roots.add(spec.artifacts.artifact_root)
        manifest_roots.add(spec.artifacts.manifest_root)

    assert len(artifact_roots) == len(retained)
    assert len(manifest_roots) == len(retained)

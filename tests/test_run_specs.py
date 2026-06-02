"""Run-spec validation tests for nominal GRU training recipes."""

from __future__ import annotations

import json

import pytest

from rlrmp.run_specs import (
    RunSpecValidationError,
    validate_nominal_gru_run_spec,
    validate_nominal_gru_run_spec_file,
)


def _valid_nominal_gru_run_spec() -> dict:
    return {
        "game_card": {
            "issue_id": "43e8728",
            "plant": "cs2019_point_mass",
        },
        "task_timing": {
            "dt": 0.01,
            "n_steps": 140,
            "epoch_len_ranges": [[0, 1], [10, 30]],
            "target_on_epochs": [1, 2],
            "hold_epochs": [0, 1],
            "move_epochs": [2],
        },
        "model_summary": {
            "architecture": "simple_staged_network",
            "controller_kind": "gru",
            "hidden_size": 180,
        },
        "training_summary": {
            "training_mode": "nominal",
            "n_warmup_batches": 3,
            "n_adversary_batches": 0,
            "loss_update_enabled": False,
        },
        "provenance": {
            "git": {"rlrmp_commit": "abc123"},
            "dependencies": {
                "rlrmp": "unknown",
                "feedbax": "unknown",
                "jax_cookbook": "unknown",
            },
            "modal": {"app": "rlrmp-gru-smoke"},
            "gpu": {"device_kinds": ["cpu"], "device_count": 1},
        },
        "feedbax_graph": {
            "schema_version": "rlrmp.feedbax_graph.v1",
            "graph_spec_path": "model.graph.json",
            "manifest_path": "model.graph.manifest.json",
            "graph_export_status": "available",
        },
    }


def test_nominal_gru_run_spec_requires_top_level_metadata(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec.pop("training_summary")

    with pytest.raises(RunSpecValidationError, match="training_summary"):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


def test_nominal_gru_run_spec_requires_provenance_groups(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec["provenance"].pop("modal")

    with pytest.raises(RunSpecValidationError, match="modal"):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


def test_nominal_gru_run_spec_requires_gru_nominal_summaries(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec["model_summary"]["controller_kind"] = "linear"

    with pytest.raises(RunSpecValidationError, match="controller_kind"):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)

    run_spec = _valid_nominal_gru_run_spec()
    run_spec["training_summary"]["training_mode"] = "minimax"

    with pytest.raises(RunSpecValidationError, match="training_mode"):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


def test_nominal_gru_run_spec_requires_adjacent_graph_sidecars(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    (tmp_path / "model.graph.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RunSpecValidationError, match="model.graph.manifest.json"):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)

    (tmp_path / "model.graph.manifest.json").write_text("{}", encoding="utf-8")
    validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


def test_nominal_gru_run_spec_allows_declared_unavailable_graph_export(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec["feedbax_graph"]["graph_spec_path"] = None
    run_spec["feedbax_graph"]["graph_export_status"] = "unavailable"
    (tmp_path / "model.graph.manifest.json").write_text("{}", encoding="utf-8")

    validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


def test_nominal_gru_run_spec_file_loads_json_and_checks_sidecars(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    (tmp_path / "model.graph.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.graph.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run.json").write_text(
        json.dumps(run_spec),
        encoding="utf-8",
    )

    validate_nominal_gru_run_spec_file(tmp_path / "run.json")

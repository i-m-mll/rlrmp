"""Run-spec validation tests for nominal GRU training recipes."""

from __future__ import annotations

import json

import pytest

from rlrmp.runtime.run_specs import (
    CS_LSS_PLANT_BACKEND,
    RunSpecValidationError,
    _is_cs_lss_run_spec,
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
        "loss_objective": "partial_feedbax_terms",
        "loss_summary": {
            "objective_profile": "partial_feedbax_terms",
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


def test_nominal_gru_run_spec_requires_consistent_loss_objective(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec.pop("loss_objective")

    with pytest.raises(RunSpecValidationError, match="loss_objective"):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)

    run_spec = _valid_nominal_gru_run_spec()
    run_spec["loss_objective"] = "full_analytical_qrf"

    with pytest.raises(RunSpecValidationError, match="objective_profile"):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)

    run_spec = _valid_nominal_gru_run_spec()
    run_spec["loss_objective"] = "partial_net_output_force_filter"
    run_spec["loss_summary"]["objective_profile"] = "partial_net_output_force_filter"
    run_spec["feedbax_graph"]["graph_spec_path"] = None
    run_spec["feedbax_graph"]["graph_export_status"] = "unavailable"
    (tmp_path / "model.graph.manifest.json").write_text("{}", encoding="utf-8")
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


def test_nominal_gru_run_spec_rejects_stale_cs_lss_graph_sidecar(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec["model_summary"]["plant_backend"] = CS_LSS_PLANT_BACKEND
    _write_graph_sidecars(
        tmp_path,
        {
            "nodes": {
                "feedback": {"type": "RLRMPFeedbackChannels"},
                "force_filter": {"type": "FirstOrderFilter"},
                "mechanics": {"type": "PointMass"},
            }
        },
    )

    with pytest.raises(
        RunSpecValidationError,
        match="LinearStateSpace.*FirstOrderFilter.*PointMass",
    ):
        validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


def test_nominal_gru_run_spec_accepts_verified_cs_lss_graph_sidecar(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec["model_summary"]["plant_backend"] = CS_LSS_PLANT_BACKEND
    _write_graph_sidecars(
        tmp_path,
        {
            "nodes": {
                "feedback": {"type": "StateFeedbackSelector"},
                "mechanics": {"type": "LinearStateSpace"},
            }
        },
    )

    validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


def test_nominal_gru_run_spec_accepts_legacy_verified_cs_lss_selector(tmp_path) -> None:
    run_spec = _valid_nominal_gru_run_spec()
    run_spec["model_summary"]["plant_backend"] = CS_LSS_PLANT_BACKEND
    _write_graph_sidecars(
        tmp_path,
        {
            "nodes": {
                "feedback": {"type": "RLRMPCsLssDelayedPositionVelocityFeedback"},
                "mechanics": {"type": "LinearStateSpace"},
            }
        },
    )

    validate_nominal_gru_run_spec(run_spec, spec_dir=tmp_path)


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"model_summary": {"plant_backend": CS_LSS_PLANT_BACKEND}}, True),
        ({"training_summary": {"plant_backend": CS_LSS_PLANT_BACKEND}}, True),
        ({"fidelity_status": {"plant_backend": CS_LSS_PLANT_BACKEND}}, True),
        ({"hps": {"model": {"plant_backend": CS_LSS_PLANT_BACKEND}}}, True),
        ({"model_summary": {"exact_cs_linear_state_space": True}}, True),
        ({"model_summary": {"plant_backend": "point_mass"}}, False),
        ({"model_summary": {"exact_cs_linear_state_space": False}}, False),
        ({}, False),
    ],
)
def test_cs_lss_run_spec_predicate_uses_declared_paths(
    patch: dict[str, object],
    expected: bool,
) -> None:
    assert _is_cs_lss_run_spec(patch) is expected


def test_flat_nominal_gru_run_spec_file_loads_sibling_sidecars(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    run_spec_path = runs_dir / "baseline.json"
    sidecar_dir = runs_dir / "baseline"
    runs_dir.mkdir()
    sidecar_dir.mkdir()
    _write_graph_sidecars(sidecar_dir, {})
    run_spec_path.write_text(json.dumps(_valid_nominal_gru_run_spec()), encoding="utf-8")

    validate_nominal_gru_run_spec_file(run_spec_path)


def test_flat_nominal_gru_run_spec_file_rejects_non_sibling_sidecars(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    run_spec_path = runs_dir / "baseline.json"
    runs_dir.mkdir()
    _write_graph_sidecars(runs_dir, {})
    run_spec_path.write_text(json.dumps(_valid_nominal_gru_run_spec()), encoding="utf-8")

    with pytest.raises(RunSpecValidationError, match=r"baseline/model\.graph\.json"):
        validate_nominal_gru_run_spec_file(run_spec_path)


def _write_graph_sidecars(tmp_path, graph_payload: dict) -> None:
    (tmp_path / "model.graph.json").write_text(
        json.dumps(graph_payload),
        encoding="utf-8",
    )
    (tmp_path / "model.graph.manifest.json").write_text("{}", encoding="utf-8")

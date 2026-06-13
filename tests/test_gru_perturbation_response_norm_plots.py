"""Tests for perturbation-response norm plot materialization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rlrmp.analysis.gru_perturbation_response_norm_plots as norm_plots
from rlrmp.analysis.gru_perturbation_response_norm_plots import (
    ExtlqgCurveCache,
    ResponseTimeWindow,
    aggregate_response_curves,
    materialize_response_norm_plots,
)


def _write_npz(path: Path, *, scale: float, sign: int = 1, n_time: int = 3) -> None:
    values = np.zeros((2, 2, n_time, 2), dtype=np.float64)
    base = np.arange(1, n_time + 1, dtype=np.float64)
    values[..., 0] = sign * scale * np.stack(
        [
            np.stack([base, base + 1.0], axis=0),
            np.stack([base + 2.0, base + 3.0], axis=0),
        ],
        axis=0,
    )
    np.savez_compressed(
        path,
        delta_position=values,
        delta_action=2.0 * values,
    )


def _row(path: Path, *, perturbation_id: str, sign: int) -> dict[str, object]:
    return {
        "perturbation_id": perturbation_id,
        "channel": "initial_state",
        "family": "initial_position_offset",
        "sign": sign,
        "timing_bin": "initial_condition",
        "status": "evaluated",
        "perturbation": {
            "perturbation_id": perturbation_id,
            "channel": "initial_state",
            "family": "initial_position_offset",
            "level_name": "small",
            "sign": sign,
            "timing_bin": "initial_condition",
        },
        "bulk_arrays": {"path": str(path)},
        "_bulk_path": str(path),
        "_severity": "small",
        "_timing_bin_normalized": "initial_condition",
    }


def _delayed_run_spec(path: Path, *, horizon: int = 3) -> None:
    path.write_text(
        json.dumps(
            {
                "task_timing": {
                    "delayed_reach": {"enabled": True},
                    "movement_window": {"cs_horizon_steps": horizon},
                }
            }
        )
    )


def _install_fake_extlqg(monkeypatch, *, n_time: int = 60) -> None:
    base = np.zeros((1, 1, n_time, 2), dtype=np.float64)
    perturbation = np.zeros((1, 1, n_time, 2), dtype=np.float64)
    perturbation[..., 0] = np.arange(1, n_time + 1, dtype=np.float64)

    def build_context() -> dict[str, object]:
        return {"base_evaluation": SimpleNamespace(position=base, command=base)}

    def simulate_perturbed(_perturbation: object, *, context: object) -> tuple[object, None, None]:
        del context
        return SimpleNamespace(position=perturbation, command=2.0 * perturbation), None, None

    monkeypatch.setattr(norm_plots, "_build_extlqg_comparator_context", build_context)
    monkeypatch.setattr(norm_plots, "_simulate_extlqg_perturbed", simulate_perturbed)


def test_aggregate_response_curves_uses_pooled_sem_and_unbanded_max(tmp_path: Path) -> None:
    pos = tmp_path / "pos.npz"
    neg = tmp_path / "neg.npz"
    _write_npz(pos, scale=1.0, sign=1)
    _write_npz(neg, scale=3.0, sign=-1)
    rows = [
        _row(pos, perturbation_id="initial_position_offset__small__x_pos", sign=1),
        _row(neg, perturbation_id="initial_position_offset__small__x_neg", sign=-1),
    ]

    mean_stats = aggregate_response_curves(
        rows,
        metric="delta_position",
        dt_s=0.01,
        repo_root=tmp_path,
        stat="mean_norm",
    )
    samples = []
    for scale in (1.0, 3.0):
        samples.append(
            scale
            * np.array(
                [
                    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                    [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
                ]
            )
        )
    pooled = np.stack(samples, axis=0).reshape((-1, 3))

    np.testing.assert_allclose(mean_stats.mean, np.mean(pooled, axis=0))
    np.testing.assert_allclose(
        mean_stats.sem,
        np.std(pooled, axis=0, ddof=1) / np.sqrt(float(pooled.shape[0])),
    )
    assert mean_stats.n_samples == 8

    max_stats = aggregate_response_curves(
        rows,
        metric="delta_position",
        dt_s=0.01,
        repo_root=tmp_path,
        stat="max_norm",
    )
    np.testing.assert_allclose(max_stats.mean, np.max(pooled, axis=0))
    assert max_stats.sem is None
    assert max_stats.n_samples == 8


def test_aggregate_response_curves_applies_delayed_time_window(tmp_path: Path) -> None:
    path = tmp_path / "response.npz"
    _write_npz(path, scale=1.0, n_time=8)
    window = ResponseTimeWindow(
        start_index=2,
        length=5,
        origin_index=4,
        time_basis="go_cue_aligned_canonical_movement_window",
        source="test",
        go_cue_index=4,
        pre_go_steps=2,
        movement_horizon_steps=3,
    )

    stats = aggregate_response_curves(
        [_row(path, perturbation_id="initial_position_offset__small__x_pos", sign=1)],
        metric="delta_position",
        dt_s=0.01,
        repo_root=tmp_path,
        stat="mean_norm",
        time_window=window,
    )

    np.testing.assert_allclose(stats.time_s, np.array([-0.02, -0.01, 0.0, 0.01, 0.02]))
    np.testing.assert_allclose(stats.mean, np.array([4.5, 5.5, 6.5, 7.5, 8.5]))


def test_materialize_response_norm_plots_writes_inventory_and_spec(tmp_path: Path) -> None:
    bulk_dir = tmp_path / "bulk" / "run"
    bulk_dir.mkdir(parents=True)
    pos = bulk_dir / "initial_position_offset__small__x_pos.npz"
    neg = bulk_dir / "initial_position_offset__small__x_neg.npz"
    _write_npz(pos, scale=1.0, sign=1)
    _write_npz(neg, scale=1.5, sign=-1)
    manifest = {
        "runs": {
            "run": {
                "label": "none_lr1e-3_clip5_b64",
                "dt_s": 0.01,
                "n_time_steps": 3,
                "bulk_files": {},
                "perturbations": [
                    _row(pos, perturbation_id=pos.stem, sign=1),
                    _row(neg, perturbation_id=neg.stem, sign=-1),
                ],
            }
        }
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = materialize_response_norm_plots(
        source_manifest_path=manifest_path,
        results_dir=tmp_path / "results",
        asset_dir=tmp_path / "assets",
        note_path=tmp_path / "note.md",
        manifest_path=tmp_path / "plot_manifest.json",
        repo_root=tmp_path,
        reconstruct_extlqg=False,
    )

    assert result["figure_count"] == 4
    assert result["aggregation_method"]["alignment"].startswith("Responses are sign-equalized")
    assert (tmp_path / "results" / "spec.json").exists()
    assert (tmp_path / "plot_manifest.json").exists()
    assert (tmp_path / "note.md").read_text().startswith("# GRU Perturbation-Response")
    for figure in result["figures"]:
        assert (tmp_path / figure["html_path"]).exists()


def test_materialize_response_norm_plots_parses_proprio_label_from_run_id(
    tmp_path: Path,
) -> None:
    bulk_dir = tmp_path / "bulk" / "run"
    bulk_dir.mkdir(parents=True)
    pos = bulk_dir / "initial_position_offset__small__x_pos.npz"
    _write_npz(pos, scale=1.0, sign=1)
    run_id = "target_relative_multitarget_fullqrf_warmcos__proprio_cal_small_lr1e-3_clip5_b64"
    manifest = {
        "runs": {
            run_id: {
                "label": "no_pgd_lr1e-3",
                "dt_s": 0.01,
                "n_time_steps": 3,
                "bulk_files": {},
                "perturbations": [
                    _row(pos, perturbation_id=pos.stem, sign=1),
                ],
            }
        }
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = materialize_response_norm_plots(
        source_manifest_path=manifest_path,
        results_dir=tmp_path / "results",
        asset_dir=tmp_path / "assets",
        note_path=tmp_path / "note.md",
        manifest_path=tmp_path / "plot_manifest.json",
        repo_root=tmp_path,
        reconstruct_extlqg=False,
        run_id_contains=("__proprio_",),
    )

    assert result["runs"][run_id]["learning_rate"] == "lr1e-3"
    assert result["runs"][run_id]["training_level"] == "small"


def test_materialize_response_norm_plots_aligns_delayed_run_to_fixed_go_cue(
    tmp_path: Path,
) -> None:
    bulk_dir = tmp_path / "bulk" / "run"
    bulk_dir.mkdir(parents=True)
    response = bulk_dir / "command_input_pulse__early_t5_x_pos.npz"
    _write_npz(response, scale=1.0, n_time=8)
    run_spec = tmp_path / "run.json"
    _delayed_run_spec(run_spec, horizon=3)
    row = _row(response, perturbation_id=response.stem, sign=1)
    row["channel"] = "target_stream"
    row["family"] = "command_input_pulse"
    row["level_name"] = None
    row["timing_bin"] = "early"
    assert isinstance(row["perturbation"], dict)
    row["perturbation"]["level_name"] = None
    row["perturbation"]["calibration_role"] = "raw_default_unscaled_effect_size"
    row["adapter"] = {
        "status": "evaluated",
        "adapter_provenance": {
            "effective_timing_mode": "go_cue_relative",
            "go_cue_index_min": 4,
            "go_cue_index_max": 4,
            "movement_horizon_steps": 3,
        },
    }
    run_id = "delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42"
    manifest = {
        "runs": {
            run_id: {
                "label": "delayed_8d_no_pgd_seed42",
                "dt_s": 0.01,
                "n_time_steps": 8,
                "run_spec_path": str(run_spec),
                "bulk_files": {},
                "perturbations": [row],
            }
        }
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = materialize_response_norm_plots(
        source_manifest_path=manifest_path,
        results_dir=tmp_path / "results",
        asset_dir=tmp_path / "assets",
        note_path=tmp_path / "note.md",
        manifest_path=tmp_path / "plot_manifest.json",
        repo_root=tmp_path,
        reconstruct_extlqg=False,
        run_id_contains=("delayed_8d_no_pgd",),
    )

    window = result["runs"][run_id]["time_window"]
    assert result["runs"][run_id]["training_level"] == "none"
    assert result["figure_count"] == 4
    assert window["time_basis"] == "go_cue_aligned_canonical_movement_window"
    assert window["start_index"] == 0
    assert window["length"] == 7
    assert window["origin_index"] == 4


def test_delayed_materialization_reenables_extlqg_movement_comparator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_fake_extlqg(monkeypatch)
    bulk_dir = tmp_path / "bulk" / "run"
    bulk_dir.mkdir(parents=True)
    response = bulk_dir / "command_input_pulse__early_t5_x_pos.npz"
    _write_npz(response, scale=1.0, n_time=70)
    run_spec = tmp_path / "run.json"
    _delayed_run_spec(run_spec, horizon=60)
    row = _row(response, perturbation_id=response.stem, sign=1)
    row["channel"] = "command_input"
    row["family"] = "command_input_pulse"
    row["level_name"] = None
    row["timing_bin"] = "early"
    assert isinstance(row["perturbation"], dict)
    row["perturbation"]["level_name"] = None
    row["perturbation"]["calibration_role"] = "raw_default_unscaled_effect_size"
    row["adapter"] = {
        "status": "evaluated",
        "adapter_provenance": {
            "effective_timing_mode": "go_cue_relative",
            "go_cue_index_min": 10,
            "go_cue_index_max": 10,
            "movement_horizon_steps": 60,
        },
    }
    run_id = "delayed_8d_no_pgd_lr3e-3_clip5_b64_seed42"
    manifest = {
        "runs": {
            run_id: {
                "label": "delayed_8d_no_pgd_seed42",
                "dt_s": 0.01,
                "n_time_steps": 70,
                "run_spec_path": str(run_spec),
                "bulk_files": {},
                "perturbations": [row],
            }
        }
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    result = materialize_response_norm_plots(
        source_manifest_path=manifest_path,
        results_dir=tmp_path / "results",
        asset_dir=tmp_path / "assets",
        note_path=tmp_path / "note.md",
        manifest_path=tmp_path / "plot_manifest.json",
        repo_root=tmp_path,
        reconstruct_extlqg=False,
        run_id_contains=("delayed_8d_no_pgd",),
    )

    policy = result["extlqg_trace_policy"]
    assert policy["status"] == "reconstructed"
    assert policy["requested_status"] == "disabled"
    assert policy["delayed_disable_override"] is True
    assert policy["time_basis"] == "movement_age_seconds"
    assert policy["movement_comparator_steps"] == 60
    assert "not a full delayed-task analytical reference" in policy["delayed_scope"]
    assert result["extlqg_trace_status"] == {"available": 8}
    assert {figure["extlqg_trace_count"] for figure in result["figures"]} == {2}

    window = result["runs"][run_id]["time_window"]
    assert window["start_index"] == 0
    assert window["length"] == 70
    assert window["origin_index"] == 10

    note = (tmp_path / "note.md").read_text()
    assert "movement-window comparators only" in note
    assert "not a full delayed-task analytical reference" in note
    assert "ExtLQG disable override" in note

    curve = ExtlqgCurveCache(enabled=True).group_curve(
        [row],
        metric="delta_position",
        dt_s=0.01,
    )
    assert curve.status == "available"
    assert curve.value.shape == (60,)
    np.testing.assert_allclose(curve.time_s[:3], np.array([0.0, 0.01, 0.02]))

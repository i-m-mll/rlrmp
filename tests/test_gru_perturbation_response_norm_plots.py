"""Tests for perturbation-response norm plot materialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rlrmp.analysis.gru_perturbation_response_norm_plots import (
    aggregate_response_curves,
    load_plot_inputs,
    materialize_response_norm_plots,
)


def _write_npz(path: Path, *, scale: float, sign: int = 1) -> None:
    values = np.zeros((2, 2, 3, 2), dtype=np.float64)
    values[..., 0] = sign * scale * np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
        ]
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
    regeneration_spec = tmp_path / result["regeneration_spec_path"]
    assert regeneration_spec.exists()
    regeneration = json.loads(regeneration_spec.read_text())
    assert regeneration["diagnostic_name"] == "gru_perturbation_response_norm_plots"
    assert any(item["role"] == "source_manifest" for item in regeneration["inputs"])
    assert any(item["role"] == "html_asset_directory" for item in regeneration["outputs"])
    assert (tmp_path / "note.md").read_text().startswith("# GRU Perturbation-Response")
    for figure in result["figures"]:
        assert (tmp_path / figure["html_path"]).exists()


def test_load_plot_inputs_reads_bulk_detail_manifest_for_slim_source(tmp_path: Path) -> None:
    bulk_dir = tmp_path / "bulk" / "run"
    bulk_dir.mkdir(parents=True)
    pos = bulk_dir / "initial_position_offset__small__x_pos.npz"
    _write_npz(pos, scale=1.0, sign=1)
    detail_manifest = {
        "runs": {
            "run": {
                "label": "none_lr1e-3_clip5_b64",
                "dt_s": 0.01,
                "n_time_steps": 3,
                "bulk_files": {},
                "perturbations": [_row(pos, perturbation_id=pos.stem, sign=1)],
            }
        }
    }
    detail_path = tmp_path / "_artifacts" / "detail.json"
    detail_path.parent.mkdir(parents=True)
    detail_path.write_text(json.dumps(detail_manifest))
    slim_manifest = {
        "bulk_detail_manifest": {"path": "_artifacts/detail.json"},
        "runs": {
            "run": {
                "label": "none_lr1e-3_clip5_b64",
                "dt_s": 0.01,
                "n_time_steps": 3,
                "bulk_files": {},
                "n_perturbation_rows": 1,
            }
        },
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(slim_manifest))

    inputs = load_plot_inputs(manifest_path, repo_root=tmp_path)

    assert len(inputs.rows["run"]) == 1
    assert inputs.rows["run"][0]["perturbation_id"] == pos.stem

"""Tests for nominal C&S-fidelity GRU run-spec preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlrmp.analysis.cs_game_card import OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir
from rlrmp.train.cs_nominal_gru import (
    build_graph_bundle,
    build_hps,
    build_parser,
    derive_spec_dir,
    write_run_spec,
)


def _args(**overrides) -> argparse.Namespace:
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_hps_uses_canonical_cs_nominal_task() -> None:
    hps = build_hps(_args())

    assert hps.method == "nominal-cs-gru"
    assert hps.dt == 0.01
    assert hps.task.type == "simple_reach"
    assert hps.task.n_steps == 60
    assert hps.task.eval_reach_length == 0.15
    assert hps.task.hold_epochs == []
    assert hps.task.p_catch_trial == 0.0
    assert hps.model.feedback_delay_steps == 5
    assert hps.model.feedback_noise_std == 0.0
    assert hps.loss.weights.effector_hold_pos == 0.0
    assert hps.loss.weights.effector_hold_vel == 0.0
    assert hps.pert.std == 0.0


def test_graph_bundle_records_nominal_provenance() -> None:
    hps = build_hps(_args(smoke=True))
    bundle = build_graph_bundle(hps)

    assert bundle.training_spec["nominal_only"] is True
    assert bundle.training_spec["adversarial_phase"] == "none"
    assert bundle.manifest["game_card_provenance"]["horizon_steps"] == 60
    assert bundle.manifest["game_card_provenance"]["target_distance_m"] == 0.15
    assert (
        bundle.manifest["game_card_provenance"]["output_feedback_certificate_gamma_factor"]
        == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    assert bundle.manifest["model_structure"]["controller_kind"] == "gru"
    assert bundle.graph_spec.nodes["net"].params["hidden_size"] == 4


def test_derive_spec_dir_preserves_artifact_results_mirror() -> None:
    artifact = run_artifact_dir("18ae684", "cs_nominal_gru__local_smoke")
    assert derive_spec_dir(artifact) == run_spec_dir("18ae684", "cs_nominal_gru__local_smoke")


def test_dry_run_does_not_write_files(tmp_path: Path) -> None:
    spec_dir = tmp_path / "spec"
    args = _args(output_dir=str(tmp_path / "artifacts"), spec_dir=str(spec_dir), dry_run=True)

    result = write_run_spec(args)

    assert "run_spec" in result
    assert result["run_spec"]["mode"] == "dry_run"
    assert result["run_spec"]["nominal_only"] is True
    assert not spec_dir.exists()


def test_write_run_spec_creates_only_lightweight_spec_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    args = _args(
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        smoke=True,
    )

    result = write_run_spec(args)

    run_path = Path(result["run_spec_path"])
    graph_path = Path(result["graph_spec_path"])
    manifest_path = Path(result["graph_manifest_path"])
    payload = json.loads(run_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert run_path == spec_dir / "run.json"
    assert graph_path == spec_dir / "model.graph.json"
    assert manifest_path == spec_dir / "model.graph.manifest.json"
    assert payload["schema_version"] == "rlrmp.cs_nominal_gru.v1"
    assert payload["model_structure"]["hidden_size"] == 4
    assert payload["game_card_provenance"]["plant"]["bw_shape"] == [48, 8]
    assert manifest["training_spec"]["nominal_only"] is True
    assert not output_dir.exists()
    assert REPO_ROOT not in output_dir.parents

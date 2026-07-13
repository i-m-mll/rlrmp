"""Golden/load coverage for the analysis data-in-code migration destinations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlrmp.analysis.data_products import (
    load_analysis_parameter_preset,
    load_first_run_baselines,
    registered_analysis_parameter_presets,
)
from rlrmp.analysis.data_products import parameter_presets
from rlrmp.data_products.registry import registered_data_product_identities
from rlrmp.paths import REPO_ROOT


def test_all_registered_analysis_presets_load_with_pinned_hashes() -> None:
    registrations = registered_analysis_parameter_presets()

    # Pilot-only parameters moved to the canonical evaluation-diagnostics preset.
    assert len(registrations) == 13
    for preset_id in registrations:
        preset = load_analysis_parameter_preset(preset_id)
        assert preset.preset_id == preset_id
        assert len(preset.content_sha256) == 64
        assert preset.parameters


def test_first_run_baselines_data_product_loads() -> None:
    baselines = load_first_run_baselines()

    assert len(baselines.rows) == 11
    assert baselines.rows[0]["group"] == "baseline_standard_12k"
    assert baselines.rows[0]["g_sd"] == 148.5024
    assert baselines.product_identity_hash == (
        "c881652ee47d4261ea13f90167fb3ea03a4a662a0d1c418bc81442251a4f4664"
    )
    assert "cross_method_first_run_baselines" in registered_data_product_identities()


def test_gamma_sweep_factors_load() -> None:
    values = load_analysis_parameter_preset("output_feedback_gamma_sweep").parameters

    assert values["gamma_sweep_factors"] == [
        1.001,
        1.005,
        1.01,
        1.02,
        1.05,
        1.1,
        1.2,
        1.25,
        1.3,
        1.32,
        1.33,
        1.34,
        1.345,
        1.35,
        1.4,
        1.45,
        1.5,
        2.0,
        3.0,
    ]


def test_state_component_slices_load() -> None:
    values = load_analysis_parameter_preset("objective_comparator").parameters

    assert values["state_component_slices"] == {
        "position": [0, 2],
        "velocity": [2, 4],
        "force_filter": [4, 6],
        "disturbance_integrator": [6, 8],
    }


def test_steady_state_bank_defaults_load() -> None:
    values = load_analysis_parameter_preset("gru_steady_state_perturbation_bank").parameters

    assert values["pre_go_steps"] == 10
    assert values["post_go_washin_steps"] == 30
    assert values["post_onset_figure_steps"] == 50
    assert values["force_filter_scale"] == 10.0


def test_adversary_equivalence_defaults_load() -> None:
    values = load_analysis_parameter_preset("adversary_equivalence").parameters

    assert values["open_loop_step_sweep"] == [50, 200, 800]
    assert values["open_loop_restarts"] == 8
    assert values["target_position_m"] == [0.15, 0.0]


def test_analysis_resolution_manifest_covers_exact_partition() -> None:
    path = REPO_ROOT / "results/6a298a1/notes/analysis_resolution_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["resolution_count"] == 43
    assert len(payload["resolutions"]) == 43
    assert all(
        {"route", "destination", "rationale"} <= resolution.keys()
        for resolution in payload["resolutions"].values()
    )
    assert sum("load_proof" in resolution for resolution in payload["resolutions"].values()) >= 6


@pytest.mark.parametrize(
    ("preset_id", "parameter_key"),
    (
        ("adversary_equivalence", "open_loop_restarts"),
        ("soft_lambda_pgd_contract", "budget_scale"),
    ),
)
def test_parameter_preset_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset_id: str,
    parameter_key: str,
) -> None:
    preset_dir = tmp_path / "analysis_presets"
    preset_dir.mkdir()
    source = REPO_ROOT / f"src/rlrmp/config/analysis_presets/{preset_id}.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["parameters"][parameter_key] = "tampered"
    (preset_dir / f"{preset_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.setattr(parameter_presets, "_CONFIG_ROOT", preset_dir)
    parameter_presets.load_analysis_parameter_preset.cache_clear()

    with pytest.raises(ValueError, match="stale content hash"):
        parameter_presets.load_analysis_parameter_preset(preset_id)

    parameter_presets.load_analysis_parameter_preset.cache_clear()

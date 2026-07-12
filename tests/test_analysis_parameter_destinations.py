"""Golden/load coverage for the analysis data-in-code migration destinations."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from rlrmp.analysis.data_products import (
    load_analysis_parameter_preset,
    load_first_run_baselines,
    registered_analysis_parameter_presets,
)
from rlrmp.analysis.data_products import parameter_presets
from rlrmp.data_products.registry import registered_data_product_identities
from rlrmp.paths import REPO_ROOT


DELAYED_SISU_DRIVER_SCRIPT = (
    REPO_ROOT / "results/7c1f7ed/scripts/materialize_delayed_sisu_velocity_profiles.py"
)
DELAYED_SISU_DRIVER_GOLDEN = {
    "schema_id": "rlrmp.delayed_sisu_velocity_profiles.driver_spec",
    "schema_version": "rlrmp.delayed_sisu_velocity_profiles.driver_spec.v1",
    "result_experiment": "7c1f7ed",
    "topic": "delayed_sisu_velocity_profiles",
    "sisu_levels": [0.0, 1.0],
    "run_refs": [
        (
            "7c1f7ed/delayed_sisu_spectrum__raw_strong_gamma_1p05_radius_"
            "lr1e-2_clip5_b64=raw strong gamma-1.05 delayed SISU"
        ),
        (
            "7c1f7ed/delayed_sisu_spectrum__effective_020a65b_pgd_radius_"
            "lr1e-2_clip5_b64=effective 020a65b PGD delayed SISU"
        ),
    ],
    "content_sha256": "69bd5ac3f9c39760457cf5d716c842a6f21e0c9bb275ad8dad24776dbda434c8",
}


def test_all_registered_analysis_presets_load_with_pinned_hashes() -> None:
    registrations = registered_analysis_parameter_presets()

    assert len(registrations) == 16
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


def test_training_widths_load() -> None:
    values = load_analysis_parameter_preset("gru_perturbation_response_norm_plots").parameters

    assert values["training_widths"] == {
        "none": 1.4,
        "small": 2.2,
        "moderate": 3.0,
        "stress": 3.8,
        "sisu_raw_strong_gamma_1p05": 2.2,
        "sisu_effective_020a65b_pgd": 3.2,
    }
    assert values["extlqg_width"] == 2.0


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


def test_delayed_sisu_driver_spec_loads() -> None:
    payload = runpy.run_path(str(DELAYED_SISU_DRIVER_SCRIPT))["load_driver_spec"]()

    assert payload == DELAYED_SISU_DRIVER_GOLDEN


@pytest.mark.parametrize("schema_field", ("schema_id", "schema_version"))
def test_delayed_sisu_driver_spec_rejects_unsupported_schema(
    tmp_path: Path,
    schema_field: str,
) -> None:
    payload = _copy_delayed_sisu_driver_golden()
    payload[schema_field] = f"unsupported.{schema_field}"
    payload["content_sha256"] = _semantic_hash(payload)

    with pytest.raises(ValueError, match=schema_field):
        _load_delayed_sisu_driver_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "tamper_case",
    ("result_experiment", "topic", "sisu_levels", "run_ref_0", "run_ref_1"),
)
def test_delayed_sisu_driver_spec_rejects_self_consistent_semantic_tamper(
    tmp_path: Path,
    tamper_case: str,
) -> None:
    payload = _copy_delayed_sisu_driver_golden()
    if tamper_case.startswith("run_ref_"):
        index = int(tamper_case.rsplit("_", 1)[1])
        payload["run_refs"][index] += "-tampered"
    else:
        payload[tamper_case] = {
            "result_experiment": "tampered-experiment",
            "topic": "tampered-topic",
            "sisu_levels": [0.0, 0.5, 1.0],
        }[tamper_case]
    # Prove the external pin rejects even a self-consistent payload whose embedded
    # hash was deliberately updated alongside the historical values.
    payload["content_sha256"] = _semantic_hash(payload)

    with pytest.raises(ValueError, match="content identity mismatch"):
        _load_delayed_sisu_driver_payload(tmp_path, payload)


def _copy_delayed_sisu_driver_golden() -> dict[str, Any]:
    return json.loads(json.dumps(DELAYED_SISU_DRIVER_GOLDEN))


def _load_delayed_sisu_driver_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = tmp_path / "delayed_sisu_velocity_profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    namespace = runpy.run_path(str(DELAYED_SISU_DRIVER_SCRIPT))
    load_driver_spec = namespace["load_driver_spec"]
    load_driver_spec.__globals__["DRIVER_SPEC_PATH"] = path
    return load_driver_spec()


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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

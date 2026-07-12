"""Load proofs for runtime defaults drained from Python literals (issue 6a298a1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from rlrmp.train import task_model as _task_model  # noqa: F401
from rlrmp import loss
from rlrmp import loss_presets
from rlrmp.cloud import modal_runner
from rlrmp.data_products import broad_epsilon, calibration
from rlrmp.eval import ensemble
from rlrmp.model import cs_lss_gru, feedbax_graph, presets as model_presets
from rlrmp.runtime import parameter_presets
from rlrmp.train.training_configs import CsNominalGruConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "470ffe0928712fcdbc7cbaf8f3042b5e919f8008"


def test_loss_defaults_load_from_registered_typed_preset() -> None:
    """Every migrated loss bundle retains its exact pre-migration values."""

    preset = loss_presets.load_loss_preset()

    assert loss.DEFAULT_TOP_WEIGHTS == preset.top_weights.model_dump()
    assert loss.DEFAULT_GOAL_HIT_SUBWEIGHTS == preset.goal_hit_subweights.model_dump()
    assert loss.DEFAULT_GOAL_HIT_PARAMS == preset.goal_hit_params.model_dump()
    assert loss.DEFAULT_EFFECTOR_POS_LATE_PARAMS == preset.effector_pos_late_params.model_dump()
    assert loss.DEFAULT_EFFECTOR_VEL_LATE_PARAMS == preset.effector_vel_late_params.model_dump()
    assert loss.DEFAULT_EFFECTOR_POS_MID_PARAMS == preset.effector_pos_mid_params.model_dump()
    assert loss.DEFAULT_EFFECTOR_VEL_MID_PARAMS == preset.effector_vel_mid_params.model_dump()
    assert loss.DEFAULT_TOP_WEIGHTS["nn_output"] == 1e-5
    assert loss.DEFAULT_GOAL_HIT_PARAMS["start_step_after_go"] == 60


def test_runtime_defaults_load_from_registered_typed_presets() -> None:
    modal = parameter_presets.load_runtime_preset(
        "rlrmp.modal_runner.default",
        parameter_presets.ModalRunnerPreset,
    )
    evaluation = parameter_presets.load_runtime_preset(
        "rlrmp.evaluation_ensemble.default",
        parameter_presets.EvaluationEnsemblePreset,
    )

    assert modal_runner.DEFAULT_TIMEOUT_SECONDS == modal.timeout_seconds == 60
    assert modal_runner.DEFAULT_N_TRAIN_BATCHES == modal.n_train_batches == 12_000
    assert modal_runner.DEFAULT_BATCH_SIZE == modal.batch_size == 250
    assert modal_runner.DEFAULT_N_REPLICATES == modal.n_replicates == 5
    assert modal_runner.DEFAULT_HIDDEN_SIZE == modal.hidden_size == 180
    assert CsNominalGruConfig.model_fields["checkpoint_interval_batches"].default == 500
    assert ensemble.N_REPLICATES == evaluation.n_replicates == 5


def test_model_and_graph_defaults_load_from_registered_typed_presets() -> None:
    cs_preset = model_presets.load_model_preset(
        "rlrmp.cs_lss_gru.default",
        model_presets.CsLssGruPreset,
    )
    graph_preset = model_presets.load_model_preset(
        "rlrmp.feedbax_graph.default",
        model_presets.FeedbaxGraphPreset,
    )

    assert cs_lss_gru.CS_DELAYED_POS_VEL_INDICES == tuple(cs_preset.delayed_pos_vel_indices)
    assert cs_lss_gru.CS_DELAYED_POS_VEL_FORCE_INDICES == tuple(
        cs_preset.delayed_pos_vel_force_indices
    )
    assert feedbax_graph.DEFAULT_GRAPH_COMPONENT_SEED == graph_preset.graph_component_seed == 0

    empty_population = SimpleNamespace()
    expected_population = {
        "hidden_size": 11,
        "n_input_only": 0,
        "n_readout_only": 0,
        "n_recurrent_only": 0,
        "n_input_readout": 0,
    }
    assert cs_lss_gru._population_structure_params(empty_population, 11) == expected_population
    assert (
        feedbax_graph._population_structure_params(
            SimpleNamespace(model=SimpleNamespace(hidden_size=11)),
            empty_population,
        )
        == expected_population
    )
    runtime_population = feedbax_graph._runtime_population_structure_params(
        empty_population,
        hidden_size=11,
    )
    assert {key: runtime_population[key] for key in expected_population} == expected_population

    migration = feedbax_graph._migrate_legacy_plant_process_force_noise_params({})
    assert migration == {
        "delay": 0,
        "noise_model": "additive_gaussian",
        "noise_std": 0.0,
        "add_noise": False,
        "noise_role": "plant_process_load",
        "noise_timing": "post_force_filter_pre_mechanics",
        "input_shape": [2],
    }


def test_governed_products_load_reference_reach_and_force_convention() -> None:
    broad_epsilon.load_broad_epsilon_anchors.cache_clear()
    calibration.load_open_loop_calibration.cache_clear()

    anchors = broad_epsilon.load_broad_epsilon_anchors()
    open_loop = calibration.load_open_loop_calibration()

    assert anchors.reference_reach_m == 0.15
    assert broad_epsilon.BROAD_EPSILON_REFERENCE_REACH_M == anchors.reference_reach_m
    assert anchors.product_identity_hash == broad_epsilon.BROAD_EPSILON_PRODUCT_IDENTITY_HASH
    assert open_loop.controller_visible_force_filter_scale_convention == {
        "value_n": 1.0,
        "description": (
            "native 1 N reference offset for force/filter controller-visible components; "
            "unit convention, not generated data"
        ),
    }
    assert open_loop.product_identity_hash == calibration.CALIBRATION_PRODUCT_IDENTITY_HASH


def test_all_destination_presets_are_registered() -> None:
    assert loss_presets.registered_loss_presets() == ("rlrmp.default_loss",)
    assert model_presets.registered_model_presets() == (
        "rlrmp.cs_lss_gru.default",
        "rlrmp.feedbax_graph.default",
    )
    assert parameter_presets.registered_runtime_presets() == (
        "rlrmp.evaluation_ensemble.default",
        "rlrmp.modal_runner.default",
    )


def test_runtime_resolution_manifest_is_scoped_to_historical_partition() -> None:
    baseline = set(
        json.loads(
            subprocess.run(
                ["git", "show", f"{SOURCE_COMMIT}:ci/data_in_code_baseline.json"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        )
    )
    owned_source_prefixes = (
        "src/rlrmp/cloud/modal_runner.py::",
        "src/rlrmp/data_products/broad_epsilon.py::",
        "src/rlrmp/data_products/calibration.py::",
        "src/rlrmp/eval/ensemble.py::",
        "src/rlrmp/loss.py::",
        "src/rlrmp/model/cs_lss_gru.py::",
        "src/rlrmp/model/feedbax_graph.py::",
    )
    excluded = {
        "src/rlrmp/cloud/modal_runner.py::DEFAULT_TRAIN_TIMEOUT_SECONDS::hp_constant",
        "src/rlrmp/cloud/modal_runner.py::NominalGruRunConfig::default_bundle",
    }
    expected = {key for key in baseline if key.startswith(owned_source_prefixes)} - excluded
    manifest = json.loads(
        (REPO_ROOT / "results/6a298a1/notes/runtime_resolution_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_id"] == "rlrmp.data_in_code_resolution_fragment"
    assert manifest["schema_version"] == "rlrmp.data_in_code_resolution_fragment.v1"
    assert manifest["resolution_count"] == len(manifest["resolutions"]) == 23
    assert set(manifest["resolutions"]) <= expected
    assert sum("load_proof" in resolution for resolution in manifest["resolutions"].values()) >= 5


_PRESET_CASES = (
    ("loss", "rlrmp.default_loss", "default.json", None),
    (
        "model",
        "rlrmp.cs_lss_gru.default",
        "cs_lss_gru.json",
        model_presets.CsLssGruPreset,
    ),
    (
        "model",
        "rlrmp.feedbax_graph.default",
        "feedbax_graph.json",
        model_presets.FeedbaxGraphPreset,
    ),
    (
        "runtime",
        "rlrmp.modal_runner.default",
        "modal_runner.json",
        parameter_presets.ModalRunnerPreset,
    ),
    (
        "runtime",
        "rlrmp.evaluation_ensemble.default",
        "evaluation_ensemble.json",
        parameter_presets.EvaluationEnsemblePreset,
    ),
)


@pytest.mark.parametrize(("kind", "preset_id", "filename", "model_type"), _PRESET_CASES)
def test_destination_preset_content_hash_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    preset_id: str,
    filename: str,
    model_type: type[Any] | None,
) -> None:
    module, source = _preset_module_and_source(kind, filename)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["content_sha256"] = "0" * 64
    (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "_CONFIG_ROOT", tmp_path)
    _clear_preset_cache(kind)

    with pytest.raises(ValueError, match="stale content hash"):
        _load_preset_case(kind, preset_id, model_type)


@pytest.mark.parametrize(("kind", "preset_id", "filename", "model_type"), _PRESET_CASES)
@pytest.mark.parametrize("schema_field", ("schema_id", "schema_version"))
def test_destination_preset_unsupported_schema_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    preset_id: str,
    filename: str,
    model_type: type[Any] | None,
    schema_field: str,
) -> None:
    module, source = _preset_module_and_source(kind, filename)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload[schema_field] = f"unsupported.{schema_field}"
    payload["content_sha256"] = _semantic_hash(payload)
    (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "_CONFIG_ROOT", tmp_path)
    _clear_preset_cache(kind)

    with pytest.raises(ValueError, match=f"unsupported {schema_field}"):
        _load_preset_case(kind, preset_id, model_type)


def _preset_module_and_source(kind: str, filename: str) -> tuple[Any, Path]:
    if kind == "loss":
        module = loss_presets
    elif kind == "model":
        module = model_presets
    else:
        module = parameter_presets
    return module, module._CONFIG_ROOT / filename


def _clear_preset_cache(kind: str) -> None:
    if kind == "loss":
        loss_presets.load_loss_preset.cache_clear()
    elif kind == "model":
        model_presets._load_model_preset.cache_clear()
    else:
        parameter_presets._load_runtime_preset.cache_clear()


def _load_preset_case(kind: str, preset_id: str, model_type: type[Any] | None) -> Any:
    if kind == "loss":
        return loss_presets.load_loss_preset(preset_id)
    assert model_type is not None
    if kind == "model":
        return model_presets.load_model_preset(preset_id, model_type)
    return parameter_presets.load_runtime_preset(preset_id, model_type)


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

"""C&S nominal-GRU identity contract tests."""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import pytest
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY
from pydantic import ValidationError
from rlrmp.analysis.math.cs_game_card import OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
from rlrmp.loss import CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE
from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir
import rlrmp.train.cs_nominal_gru as cs_nominal_gru
import rlrmp.train.executor.cs_supervised as cs_supervised_executor
from rlrmp.train.cs_nominal_gru import (
    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER,
    CsNominalGruConfig,
    DEFAULT_STOCHASTIC_PRESET,
    SCHEMA_VERSION,
    build_graph_bundle,
    build_training_run_graph_spec,
    build_hps,
    derive_spec_dir,
    derive_spec_path,
    write_run_spec,
)
from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    RLRMP_RUN_SPEC_PAYLOAD_KEY,
    assert_runtime_graph_matches_training_spec,
    hydrate_compact_run_spec_envelope,
)
from rlrmp.runtime.run_specs import validate_nominal_gru_run_spec_file
from rlrmp.data_products.broad_epsilon import load_pgd_radius_source
from rlrmp.train.executor.slots import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    CS_SUPERVISED_METHOD_REF,
    POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
)
from rlrmp.train.executor.cs_supervised import build_execution_context_from_spec
from rlrmp.train.run_spec_authoring import COMPACT_RUN_SPEC_KEY, MAX_TRACKED_RUN_SPEC_BYTES
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    POLICY_ADVERSARY_PLAIN_MODE,
    BroadFullStateEpsilonTrainingConfig,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    TargetRelativeMultiTargetTrainingConfig,
    FixedTargetPerturbationTrainingConfig,
)
from rlrmp.train.task_model import CS_LSS_PLANT_BACKEND

HISTORICAL_020A65B_PGD_RADIUS_15CM = float(
    load_pgd_radius_source("effective_020a65b_pgd_training_radius")["l2_radius_15cm"]
)


def _args(**overrides) -> argparse.Namespace:
    values = CsNominalGruConfig(
        issue="test",
        output_dir="_artifacts/test/runs/test",
    ).model_dump(mode="python")
    values.update(compact_run_spec=False, verify_resume_only=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def _remove_training_method_registration(
    monkeypatch: pytest.MonkeyPatch,
    method_ref: str,
) -> None:
    registrations = dict(DEFAULT_TRAINING_METHOD_REGISTRY._registrations)
    registrations.pop(method_ref, None)
    monkeypatch.setattr(DEFAULT_TRAINING_METHOD_REGISTRY, "_registrations", registrations)
    assert method_ref not in DEFAULT_TRAINING_METHOD_REGISTRY.available_keys()


def _absolute_string_leaves(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)] if Path(value).is_absolute() else []
    if isinstance(value, dict):
        leaves: list[tuple[str, str]] = []
        for key, child in value.items():
            leaves.extend(_absolute_string_leaves(child, path=f"{path}.{key}"))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, child in enumerate(value):
            leaves.extend(_absolute_string_leaves(child, path=f"{path}[{index}]"))
        return leaves
    return []


def _assert_no_absolute_string_leaves(value: Any) -> None:
    absolute = _absolute_string_leaves(value)
    assert absolute == []


def _cs_stochastic_gru_run_spec_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted(Path("results").rglob("*.json")):
        payload = hydrate_compact_run_spec_envelope(json.loads(path.read_text(encoding="utf-8")))
        schema_versions = {
            payload.get("schema_version"),
            payload.get("source_schema_version"),
        }
        if SCHEMA_VERSION not in schema_versions:
            continue
        if {"hps", "feedbax_graph", "training_script"}.issubset(payload):
            paths.append(path)
    return paths


def _native_method_run_spec_payload(tmp_path: Path, method_ref: str) -> dict[str, Any]:
    common = {
        "output_dir": str(tmp_path / method_ref.replace("/", "_") / "artifacts"),
        "spec_dir": str(tmp_path / method_ref.replace("/", "_") / "spec"),
        "dry_run": True,
    }
    if method_ref == CS_SUPERVISED_METHOD_REF:
        return write_run_spec(
            _args(
                **common,
                target_relative_multitarget=True,
                perturbation_training=True,
                perturbation_calibrated_timing=True,
                perturbation_physical_level="small",
            )
        )["run_spec"]
    if method_ref == ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF:
        return write_run_spec(
            _args(
                **common,
                target_relative_multitarget=True,
                broad_epsilon_pgd_training=True,
                broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                broad_epsilon_pgd_energy_lambda=2.5,
                adaptive_epsilon_curriculum=True,
                adaptive_epsilon_controller_training_mode=(
                    ADAPTIVE_EPSILON_TRAINING_MODE_EPSILON_SCALED_OUTER
                ),
                adaptive_epsilon_damage_peak=3500.0,
                adaptive_epsilon_damage_final=1000.0,
                adaptive_epsilon_damage_ramp_batches=1,
                adaptive_epsilon_damage_anneal_batches=2,
                adaptive_epsilon_update_interval_batches=3,
                adaptive_epsilon_ema_alpha=0.2,
                adaptive_epsilon_eta=0.3,
                adaptive_epsilon_deadband_frac=0.4,
                adaptive_epsilon_lambda_min=1e-9,
                adaptive_epsilon_max_log_step=0.5,
                adaptive_epsilon_outer_weight_ramp_batches=6,
            )
        )["run_spec"]
    if method_ref == POLICY_ADVERSARY_SUPERVISED_METHOD_REF:
        return write_run_spec(
            _args(
                **common,
                target_relative_multitarget=True,
                force_filter_feedback=True,
                initial_hidden_encoder=True,
                perturbation_training=True,
                perturbation_calibrated_timing=True,
                perturbation_physical_level="small",
                policy_adversary_training=True,
                policy_adversary_mode=POLICY_ADVERSARY_PLAIN_MODE,
                policy_adversary_steps=5,
                policy_adversary_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
                policy_adversary_radius_source="effective_020a65b_pgd_training_radius",
                n_train_batches=12000,
                stop_after_batches=1000,
                loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            )
        )["run_spec"]
    raise AssertionError(f"unsupported native method reference: {method_ref}")


def _compact_run_spec(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "compact_run_spec": True,
        "game_card": payload["game_card"],
        "training_distribution": {
            "perturbation_training": payload["training_distribution"]["perturbation_training"],
        },
        "feedbax_graph": payload["feedbax_graph"],
        FEEDBAX_TRAINING_RUN_SPEC_KEY: payload[FEEDBAX_TRAINING_RUN_SPEC_KEY],
        RLRMP_RUN_SPEC_PAYLOAD_KEY: payload[RLRMP_RUN_SPEC_PAYLOAD_KEY],
    }


def _write_flat_recipe_sidecars(recipe_path: Path, payload: dict[str, Any]) -> None:
    sidecar_dir = recipe_path.with_suffix("")
    sidecar_dir.mkdir(parents=True)
    graph = payload["feedbax_graph"]
    for key in ("graph_spec_path", "manifest_path"):
        pointer = graph[key]
        if pointer is None:
            continue
        sidecar_path = sidecar_dir / pointer
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_payload = (
            {
                "nodes": {
                    "mechanics": {"type": "LinearStateSpace"},
                    "feedback": {"type": "StateFeedbackSelector"},
                }
            }
            if key == "graph_spec_path"
            else {}
        )
        sidecar_path.write_text(json.dumps(sidecar_payload), encoding="utf-8")


def test_stochastic_preset_metadata_is_independent_of_jax_x64() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    code = """
import json
import sys

import jax

jax.config.update("jax_enable_x64", sys.argv[1] == "true")
from rlrmp.train.cs_nominal_gru import stochastic_preset

print(json.dumps(stochastic_preset("cs2019-rollout").summary(), sort_keys=True))
"""
    summaries = []
    for flag in ("false", "true"):
        completed = subprocess.run(
            [sys.executable, "-c", code, flag],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        summaries.append(json.loads(completed.stdout))

    assert summaries[0] == summaries[1]


def test_cs_nominal_gru_config_validates_tracked_cs_stochastic_gru_corpus() -> None:
    paths = _cs_stochastic_gru_run_spec_paths()
    clean_paths = []
    fail_closed: set[Path] = set()
    wave_1_paths = {
        Path("results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"),
        Path("results/c6c5997/runs/rewarm_3e-3-epsilon-ramp.json"),
        Path("results/c6c5997/runs/rewarm_3e-4-epsilon-ramp.json"),
        Path("results/cb3685a/runs/seam_probe.json"),
    }

    assert wave_1_paths.issubset(paths)
    # Issue b6b5502 retired 26 pre-three-layer CS-GRU recipes. Issue ee7a6f4
    # moved eight 3cd018b envelopes into TrainingRunMatrixSpec documents, and
    # dd7234e replaced 17 ef9c882 flat recipes with rows inside one compact
    # matrix. Their dedicated storage-adoption tests exhaustively decode and
    # materialize those rows. Pin this collector's remaining 23 historical or
    # current flat recipes plus four wave-1 compact flat recipes.
    assert len(paths) == 23 + len(wave_1_paths)
    for path in paths:
        payload = hydrate_compact_run_spec_envelope(json.loads(path.read_text(encoding="utf-8")))
        try:
            CsNominalGruConfig.model_validate(cs_nominal_gru._args_values_from_run_spec(payload))
        except ValidationError:
            fail_closed.add(path)
        else:
            clean_paths.append(path)

    assert len(clean_paths) == 23 + len(wave_1_paths)
    assert fail_closed == set()


def test_graph_bundle_records_nominal_provenance() -> None:
    hps = build_hps(_args(smoke=True))
    bundle = build_graph_bundle(hps)

    assert bundle.training_spec["nominal_only"] is True
    assert bundle.training_spec["plant_backend"] == CS_LSS_PLANT_BACKEND
    assert bundle.training_spec["adversarial_phase"] == "none"
    assert bundle.training_spec["certificate_lens"] == "input_output_map_certificate"
    assert bundle.manifest["game_card_provenance"]["horizon_steps"] == 60
    assert bundle.manifest["game_card_provenance"]["feedbax_task_n_steps"] == 61
    assert bundle.manifest["game_card_provenance"]["feedbax_control_cost_stages"] == 60
    assert bundle.manifest["game_card_provenance"]["init_pos_m"] == [0.0, 0.0]
    assert bundle.manifest["game_card_provenance"]["target_distance_m"] == 0.15
    assert (
        bundle.manifest["game_card_provenance"]["cost"]["feedbax_force_filter_state_cost"]
        == "not_available"
    )
    assert (
        bundle.manifest["game_card_provenance"]["output_feedback_certificate_gamma_factor"]
        == OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
    )
    assert bundle.manifest["model_structure"]["controller_kind"] == "gru"
    assert bundle.manifest["model_structure"]["plant_backend"] == CS_LSS_PLANT_BACKEND
    assert bundle.manifest["model_structure"]["exact_cs_linear_state_space"] is True
    assert bundle.manifest["model_structure"]["fixed_plant_parameters"] == [
        "nodes.mechanics.A",
        "nodes.mechanics.B",
        "nodes.mechanics.B_w",
    ]
    assert (
        bundle.manifest["model_structure"]["stochastic_runtime"]["state_diffusion"]
        == "mechanics.epsilon"
    )
    assert bundle.manifest["stochastic_preset"]["name"] == DEFAULT_STOCHASTIC_PRESET
    assert bundle.manifest["stochastic_preset"]["signal_dependent_motor_noise_std"] == 0.02
    assert (
        bundle.manifest["model_structure"]["plant_process"]["noise_timing"]
        == "mechanics.epsilon_sampled_task_input"
    )
    assert bundle.manifest["model_structure"]["plant_process"]["state_diffusion"] == (
        "mechanics.epsilon"
    )
    assert (
        "physical-process/load epsilon"
        in (bundle.manifest["model_structure"]["plant_process"]["epsilon_bridge"])
    )
    assert bundle.manifest["model_structure"]["population_structure"] == {
        "n_input_only": 0,
        "n_readout_only": 0,
        "n_recurrent_only": 0,
        "n_input_readout": 4,
    }
    assert bundle.manifest["model_structure"]["certificate_lens"] == "input_output_map_certificate"
    assert bundle.manifest["model_structure"]["analytical_delay_augmented_state_input"] is False
    assert bundle.graph_spec.nodes["net"].params["hidden_size"] == 4


def test_derive_spec_dir_preserves_artifact_results_mirror() -> None:
    artifact = run_artifact_dir("30f2313", "cs_stochastic_gru__no_hidden_penalty")
    assert derive_spec_dir(artifact) == run_spec_dir(
        "30f2313",
        "cs_stochastic_gru__no_hidden_penalty",
    )
    assert derive_spec_path(artifact) == (
        REPO_ROOT / "results" / "30f2313" / "runs" / "cs_stochastic_gru__no_hidden_penalty.json"
    )


@pytest.mark.parametrize(
    "compact_run_spec",
    (False, True),
    ids=("default_full_payload", "explicit_compact_envelope"),
)
def test_large_composed_run_spec_compaction_is_opt_in(
    tmp_path: Path,
    compact_run_spec: bool,
) -> None:
    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "spec"
    result = write_run_spec(
        _args(
            output_dir=str(output_dir),
            spec_dir=str(spec_dir),
            full_train=True,
            n_train_batches=12000,
            batch_size=64,
            controller_lr=3e-3,
            lr_warmup_batches=500,
            lr_warmup_init_fraction=0.1,
            lr_cosine_alpha=0.01,
            gradient_clip_norm=5.0,
            checkpoint_interval_batches=500,
            hidden_size=180,
            n_replicates=5,
            no_integrator_state=True,
            target_relative_multitarget=True,
            target_support_profile="const_band16",
            force_filter_feedback=True,
            initial_hidden_encoder=True,
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_physical_level="moderate",
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            compact_run_spec=compact_run_spec,
        )
    )

    run_path = Path(result["run_spec_path"])
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    authoritative = payload[RLRMP_RUN_SPEC_PAYLOAD_KEY]

    if not compact_run_spec:
        assert COMPACT_RUN_SPEC_KEY not in payload
        assert run_path.stat().st_size > MAX_TRACKED_RUN_SPEC_BYTES
        assert payload["hps"]["model"]["hidden_size"] == 180
        return

    assert run_path.stat().st_size <= MAX_TRACKED_RUN_SPEC_BYTES
    assert set(payload) == {
        COMPACT_RUN_SPEC_KEY,
        RLRMP_RUN_SPEC_PAYLOAD_KEY,
        FEEDBAX_TRAINING_RUN_SPEC_KEY,
        "game_card",
        "training_distribution",
        "feedbax_graph",
    }
    assert payload[COMPACT_RUN_SPEC_KEY] is True
    assert payload["game_card"] == authoritative["game_card"]
    assert payload["training_distribution"] == {
        "perturbation_training": authoritative["training_distribution"]["perturbation_training"],
    }
    assert payload["feedbax_graph"] == authoritative["feedbax_graph"]
    assert isinstance(payload[FEEDBAX_TRAINING_RUN_SPEC_KEY], dict)

    replay_args = build_execution_context_from_spec(run_path).args
    assert replay_args.hidden_size == 180
    assert replay_args.target_support_profile == "const_band16"
    assert replay_args.perturbation_calibrated_timing is True


def test_feedbax_training_run_spec_authoring_uses_portable_binding_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError):
        _assert_no_absolute_string_leaves(
            {"method_payload": {"payload": {"config": {"spec_dir": str(tmp_path / "spec")}}}}
        )

    def authored_payload(root: Path) -> dict[str, Any]:
        output_dir = root / "_artifacts" / "81e3d8d" / "runs" / "adaptive"
        spec_dir = root / "results" / "81e3d8d" / "runs" / "adaptive"
        result = write_run_spec(
            _args(
                output_dir=str(output_dir),
                spec_dir=str(spec_dir),
                issue="81e3d8d",
                smoke=True,
                dry_run=True,
                gradient_clip_norm=5.0,
                target_relative_multitarget=True,
                broad_epsilon_pgd_training=True,
                broad_epsilon_pgd_objective=BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
                broad_epsilon_pgd_energy_lambda=2.5,
                adaptive_epsilon_curriculum=True,
            )
        )
        return result["run_spec"][FEEDBAX_TRAINING_RUN_SPEC_KEY]

    first = authored_payload(tmp_path / "worktree_a")
    second = authored_payload(tmp_path / "worktree_b")
    method_payload = first["method_payload"]

    _assert_no_absolute_string_leaves(method_payload)
    assert first["artifacts"]["artifact_root"] == "_artifacts/81e3d8d/runs/adaptive"
    assert first["artifacts"]["metadata"]["tracked_spec_dir"] == ("results/81e3d8d/runs/adaptive")
    assert method_payload["payload"]["config"]["output_dir"] == ("_artifacts/81e3d8d/runs/adaptive")
    assert method_payload["payload"]["config"]["spec_dir"] == "results/81e3d8d/runs/adaptive"
    assert first["method_payload"] == second["method_payload"]


def test_training_run_spec_graph_guard_rejects_diverging_hps(tmp_path: Path) -> None:
    args = _args(
        output_dir=str(tmp_path / "bulk"),
        spec_dir=str(tmp_path / "spec"),
        smoke=True,
        dry_run=True,
        gradient_clip_norm=5.0,
    )
    payload = write_run_spec(args)["run_spec"]
    diverging_hps = build_hps(_args(hidden_size=5, n_replicates=1))

    with pytest.raises(ValueError, match="Serialized TrainingRunSpec graph"):
        assert_runtime_graph_matches_training_spec(
            payload,
            graph_spec=build_training_run_graph_spec(diverging_hps, seed=int(args.seed)),
        )


@pytest.mark.parametrize(
    "method_ref",
    (
        CS_SUPERVISED_METHOD_REF,
        ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
        POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
    ),
)
def test_generic_run_spec_loader_registers_native_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_ref: str,
) -> None:
    payload = _native_method_run_spec_payload(tmp_path, method_ref)
    recipe_path = tmp_path / "runs" / f"{method_ref.split('/')[1]}.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    _remove_training_method_registration(monkeypatch, method_ref)

    _path, loaded = cs_supervised_executor.load_validated_run_spec(recipe_path)

    assert loaded[FEEDBAX_TRAINING_RUN_SPEC_KEY]["method_ref"] == {
        "package": "rlrmp",
        "name": method_ref.split("/")[1],
        "version": "v1",
    }
    assert method_ref in DEFAULT_TRAINING_METHOD_REGISTRY.available_keys()


def test_generic_loader_resolves_flat_recipe_sibling_sidecars(tmp_path: Path) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    recipe_path = tmp_path / "runs" / "baseline.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_flat_recipe_sidecars(recipe_path, payload)

    _path, loaded = cs_supervised_executor.load_validated_run_spec(
        recipe_path,
        require_graph_sidecars=True,
    )

    assert loaded["feedbax_graph"] == payload["feedbax_graph"]


def test_public_flat_recipe_validator_hydrates_compact_envelope(tmp_path: Path) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    recipe_path = tmp_path / "runs" / "baseline.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(_compact_run_spec(payload)), encoding="utf-8")
    _write_flat_recipe_sidecars(recipe_path, payload)

    validate_nominal_gru_run_spec_file(recipe_path)


def test_executor_first_import_replays_compact_flat_recipe(tmp_path: Path) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    recipe_path = tmp_path / "runs" / "baseline.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(_compact_run_spec(payload)), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from rlrmp.train.executor.cs_supervised import load_validated_run_spec; "
                f"load_validated_run_spec({str(recipe_path)!r})"
            ),
        ],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_compact_run_spec_loader_hydrates_authoritative_extension(tmp_path: Path) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    compact = _compact_run_spec(payload)
    recipe_path = tmp_path / "runs" / "baseline.json"
    recipe_path.parent.mkdir()
    recipe_path.write_text(json.dumps(compact), encoding="utf-8")

    _path, hydrated = cs_supervised_executor.load_validated_run_spec(recipe_path)

    assert "compact_run_spec" not in hydrated
    assert hydrated["hps"] == payload["hps"]
    assert hydrated["game_card"] == payload["game_card"]
    assert hydrated["training_distribution"] == payload["training_distribution"]
    assert hydrated["feedbax_graph"] == payload["feedbax_graph"]
    assert hydrated[FEEDBAX_TRAINING_RUN_SPEC_KEY] == payload[FEEDBAX_TRAINING_RUN_SPEC_KEY]


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda payload: payload.__setitem__("compact_run_spec", False), "boolean true"),
        (lambda payload: payload.pop(RLRMP_RUN_SPEC_PAYLOAD_KEY), "rlrmp_run_spec"),
        (lambda payload: payload.__setitem__(RLRMP_RUN_SPEC_PAYLOAD_KEY, []), "rlrmp_run_spec"),
        (
            lambda payload: payload.__setitem__("game_card", {"issue_id": "mismatched-identity"}),
            "game_card",
        ),
    ),
)
def test_compact_run_spec_loader_rejects_missing_or_mismatched_extension(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    payload = _native_method_run_spec_payload(tmp_path, CS_SUPERVISED_METHOD_REF)
    compact = _compact_run_spec(payload)
    mutation(compact)

    with pytest.raises(ValueError, match=match):
        hydrate_compact_run_spec_envelope(compact)


@pytest.mark.parametrize(
    "config",
    (
        FixedTargetPerturbationTrainingConfig(enabled=True),
        TargetRelativeMultiTargetTrainingConfig(enabled=True),
        BroadFullStateEpsilonTrainingConfig(enabled=True),
        PgdFullStateEpsilonTrainingConfig(enabled=True),
        PolicyFullStateEpsilonTrainingConfig(
            enabled=True,
            reference_l2_radius_15cm=HISTORICAL_020A65B_PGD_RADIUS_15CM,
            budget_source="effective_020a65b_pgd_training_radius",
        ),
    ),
)
def test_frozen_rendered_training_payloads_migrate_without_authored_defaults(config) -> None:
    payload = config.to_hps_dict()
    payload.pop("config")

    migrated_payload = type(config).from_payload(payload).to_hps_dict()
    migrated_payload.pop("config")

    assert migrated_payload == payload


def test_stage2_adaptive_epsilon_specs_reject_cross_mirror_drift() -> None:
    recipe_path = REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

    cs_supervised_executor._validate_adaptive_epsilon_cross_mirrors(recipe)

    recipe["hps"]["adaptive_epsilon_curriculum"]["lambda_update"]["eta"] = 0.1
    with pytest.raises(
        ValueError,
        match=(
            "Adaptive-epsilon cross-mirror mismatch "
            "field=lambda_update.eta: hps=0.1 payload=0.2 config=0.2"
        ),
    ):
        cs_supervised_executor._validate_adaptive_epsilon_cross_mirrors(recipe)

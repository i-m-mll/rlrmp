"""Spec-first minimax training contracts."""

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from feedbax.contracts.training import TrainingRunSpec
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.model.feedbax_graph import build_rlrmp_feedbax_graph_bundle
from rlrmp.train.minimax import (
    MINIMAX_METHOD_REF,
    MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
    MinimaxConfig,
    MinimaxMethodPayload,
    _minimax_method_payload,
    build_hps,
    build_minimax_training_run_spec,
    legacy_cli_args_to_minimax_config,
    minimax_config_namespace,
    minimax_effective_phase_fingerprint,
    minimax_effective_phase_spec,
    minimax_method_contract,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
)
from rlrmp.train.task_model import build_task_base


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _payload_from_config(tmp_path: Path, config: dict) -> dict:
    args = minimax_config_namespace(config)
    hps = build_hps(args)
    graph_bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
        key=jr.PRNGKey(args.seed),
    )
    return build_minimax_training_run_spec(
        config,
        graph_spec=graph_bundle.graph_spec,
        output_dir=Path(args.output_dir),
        spec_dir=tmp_path / "results" / "54b0c2e" / "runs" / "spec",
        git={"commit": "test"},
        gpu_info={"available": False},
        feedbax_graph={"graph_spec_path": "graph_spec.json", "manifest_path": "manifest.json"},
    )


def _payload(tmp_path: Path, argv: list[str]) -> dict:
    config = legacy_cli_args_to_minimax_config(
        [
            "--n-warmup-batches",
            "1",
            "--n-adversary-batches",
            "0",
            "--batch-size",
            "1",
            "--n-replicates",
            "1",
            "--output-dir",
            str(tmp_path / "_artifacts" / "54b0c2e" / "runs" / "spec"),
            *argv,
        ]
    )
    return _payload_from_config(tmp_path, config)


def test_warmup_only_minimax_cli_round_trips_to_training_run_spec(tmp_path: Path) -> None:
    payload = _payload(tmp_path, [])

    validate_minimax_run_spec(payload, spec_dir=tmp_path)
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])

    assert spec.method_ref.key == MINIMAX_METHOD_REF
    assert spec.worker_execution.effective_phase.phase_program.initial_phase == "warmup"
    assert minimax_training_run_spec_to_config(spec)["n_adversary_batches"] == 0
    assert payload["rlrmp_run_spec"]["schema_version"] == "rlrmp.run_spec.v2"


def test_gaussian_bump_minimax_cli_round_trips_to_training_run_spec(tmp_path: Path) -> None:
    payload = _payload(
        tmp_path,
        ["--n-adversary-batches", "2", "--n-bumps", "2", "--force-max", "0.5"],
    )
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    config = minimax_training_run_spec_to_config(spec)

    assert config["adversary_type"] == "gaussian_bump"
    assert config["n_bumps"] == 2
    assert spec.method_payload.payload["adversarial"]["inner_direction"] == "maximize"


def test_linear_dynamics_minimax_cli_authors_declarative_dynamics_node(
    tmp_path: Path,
) -> None:
    payload = _payload(
        tmp_path,
        [
            "--adversary-type",
            "linear_dynamics",
            "--n-adversary-batches",
            "2",
            "--linear-dynamics-eta-max",
            "0.2",
        ],
    )
    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    graph = spec.graph.inline

    assert graph["nodes"][PLANT_INTERVENOR_LABEL]["type"] == "DynamicsMatrixPerturb"
    assert jnp.asarray(graph["nodes"][PLANT_INTERVENOR_LABEL]["params"]["delta_A"]).shape == (
        2,
        4,
    )
    assert spec.method_payload.payload["projection"] == {
        "target": "adversary_population[active_member].delta_A",
        "operator": "frobenius_ball",
        "radius": 0.2,
        "radius_source": "linear_dynamics_eta_max",
        "timing": "after_each_adversary_step",
        "phase_scope": "adversarial",
    }


def test_effective_phase_fingerprint_rejects_tampered_spec(tmp_path: Path) -> None:
    payload = _payload(tmp_path, [])
    spec = dict(payload["feedbax_training_run_spec"])
    spec["worker_execution"]["effective_phase"]["phase_program"]["metadata"] = {
        "phase_program_identity": "tampered"
    }
    payload["feedbax_training_run_spec"] = spec

    try:
        validate_minimax_run_spec(payload, spec_dir=tmp_path)
    except ValueError as exc:
        assert "effective-phase fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("tampered effective phase fingerprint was accepted")


def test_minimax_method_payload_matches_pre_refactor_golden_fixtures() -> None:
    fixtures = json.loads(
        Path("tests/fixtures/minimax_method_payload_golden.json").read_text(encoding="utf-8")
    )
    graph_payload = fixtures["graph_payload"]
    contract = minimax_method_contract()
    effective_phase = minimax_effective_phase_spec(contract)

    for case in fixtures["cases"].values():
        config = legacy_cli_args_to_minimax_config(case["argv"])
        envelope = _minimax_method_payload(
            config,
            output_dir=Path("_artifacts/minimax/minimax_test"),
            spec_dir=Path("results/5e5ba8b/runs/golden"),
        )
        envelope_dump = envelope.model_dump(mode="json", exclude_none=True)

        assert _canonical_json(envelope_dump) == case["method_payload_envelope_json"]
        assert (
            minimax_effective_phase_fingerprint(
                effective_phase=effective_phase,
                graph_payload=graph_payload,
                method_payload=envelope_dump,
            )
            == case["effective_phase_fingerprint"]
        )


def test_minimax_training_run_spec_file_round_trip_rebuilds_idempotently(
    tmp_path: Path,
) -> None:
    payload = _payload(
        tmp_path,
        ["--adversary-type", "linear_dynamics", "--linear-dynamics-eta-max", "0.2"],
    )
    spec_path = tmp_path / "run.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = minimax_training_run_spec_from_file(spec_path)
    rebuilt = _payload_from_config(tmp_path, minimax_training_run_spec_to_config(loaded))

    original_spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    rebuilt_spec = TrainingRunSpec.model_validate(rebuilt["feedbax_training_run_spec"])
    assert _canonical_json(original_spec.model_dump(mode="json", exclude_none=True)) == (
        _canonical_json(rebuilt_spec.model_dump(mode="json", exclude_none=True))
    )


def test_minimax_payload_schema_version_stays_v1() -> None:
    assert MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION == (
        "rlrmp.spec.training_method.minimax_payload.v1"
    )


def test_minimax_payload_rejects_derived_view_drift() -> None:
    envelope = _minimax_method_payload(
        MinimaxConfig().model_dump(mode="python"),
        output_dir=Path("_artifacts/minimax/minimax_test"),
        spec_dir=Path("results/5e5ba8b/runs/golden"),
    )
    payload = dict(envelope.payload)
    payload["warmup"] = {**payload["warmup"], "n_batches": 999}

    with pytest.raises(ValueError, match="derived views disagree"):
        MinimaxMethodPayload.model_validate(payload)


def test_minimax_cli_adapter_preserves_flag_forms_and_help() -> None:
    config = legacy_cli_args_to_minimax_config(
        [
            "--checkpoint",
            "--no-resume",
            "--n-bumps=2",
            "--force-max",
            "0.5",
            "--jax-explain-cache-misses=false",
        ]
    )

    assert config["checkpoint"] is True
    assert config["resume"] is False
    assert config["n_bumps"] == 2
    assert config["force_max"] == 0.5
    assert config["jax_explain_cache_misses"] is False

    with pytest.raises(ValueError, match="unknown minimax option"):
        legacy_cli_args_to_minimax_config(["--does-not-exist"])
    with pytest.raises(ValueError, match="only valid for boolean"):
        legacy_cli_args_to_minimax_config(["--no-batch-size"])

    with pytest.raises(SystemExit) as exc_info:
        legacy_cli_args_to_minimax_config(["--help"])
    help_text = str(exc_info.value)
    for name, field in MinimaxConfig.model_fields.items():
        assert f"--{name.replace('_', '-')} (default: {field.default!r})" in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["--adversary-type", "invalid"],
        ["--n-warmup-batches", "-1"],
        ["--n-replicates", "0"],
        ["--adversary-type", "linear_dynamics", "--no-fused"],
    ],
)
def test_minimax_config_validation_replaces_legacy_validator(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        legacy_cli_args_to_minimax_config(argv)

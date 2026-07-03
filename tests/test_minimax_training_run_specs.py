"""Spec-first minimax training contracts."""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from feedbax.contracts.training import TrainingRunSpec
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.model.feedbax_graph import build_rlrmp_feedbax_graph_bundle
from rlrmp.train.minimax import (
    MINIMAX_METHOD_REF,
    build_hps,
    build_minimax_training_run_spec,
    legacy_cli_args_to_minimax_config,
    minimax_config_namespace,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
)
from rlrmp.train.task_model import build_task_base


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

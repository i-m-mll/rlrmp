from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlrmp.cloud import modal_runner


def test_modal_training_command_ships_authored_document_and_row() -> None:
    document = Path("results/19d6acd/runs/matrix.json").resolve()
    config = modal_runner.NominalGruRunConfig(
        authored_document=str(document),
        row="cs_row",
        resume=True,
    )

    command = modal_runner.build_training_command(config, remote=True)

    assert command == [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/launch_training.py",
        "execute",
        "/workspace/rlrmp/results/19d6acd/runs/matrix.json",
        "--row",
        "cs_row",
        "--resume",
        "--log-step",
        "1",
    ]
    assert "train_cs_nominal_gru.py" not in command
    assert "execute-training-run-spec" not in command


def test_modal_training_command_requires_authored_document() -> None:
    with pytest.raises(ValueError, match="TrainingRunMatrixSpec"):
        modal_runner.build_training_command(modal_runner.NominalGruRunConfig(), remote=True)


def test_modal_paths_derive_experiment_from_authored_document() -> None:
    config = modal_runner.NominalGruRunConfig(
        authored_document="results/19d6acd/runs/matrix.json"
    )

    assert config.resolved_experiment() == "19d6acd"
    assert config.local_artifact_dir().relative_to(Path.cwd()) == (
        Path("_artifacts/19d6acd/runs") / config.run
    )


def test_modal_paths_require_explicit_or_authored_experiment() -> None:
    with pytest.raises(ValueError, match="experiment is required"):
        modal_runner.NominalGruRunConfig().local_artifact_dir()


def test_modal_document_must_be_part_of_embedded_repo(tmp_path: Path) -> None:
    config = modal_runner.NominalGruRunConfig(authored_document=str(tmp_path / "matrix.json"))

    with pytest.raises(ValueError, match="inside the rlrmp repository"):
        modal_runner.build_training_command(config, remote=True)


def test_modal_parser_exposes_operations_but_no_scientific_overlay() -> None:
    parser = modal_runner.build_parser()
    args = parser.parse_args(
        [
            "modal-run",
            "--document",
            "results/19d6acd/runs/matrix.json",
            "--row",
            "cs_row",
            "--gpu",
            "A100",
            "--timeout-seconds",
            "90",
            "--confirm-billable-launch",
        ]
    )
    config = modal_runner.make_config(args)

    assert config.authored_document == "results/19d6acd/runs/matrix.json"
    assert config.row == "cs_row"
    assert config.gpu == "A100"
    assert config.timeout_seconds == 90
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--extra-arg" not in option_strings
    assert "--batch-size" not in option_strings
    assert "--controller-lr" not in option_strings
    assert "--loss-objective" not in option_strings
    assert "--broad-epsilon-pgd-training" not in option_strings


def test_remote_payload_executes_launch_runner_without_recompiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = Path("results/19d6acd/runs/matrix.json").resolve()
    config = modal_runner.NominalGruRunConfig(
        authored_document=str(document),
        row="row_a",
        timeout_seconds=10,
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(modal_runner, "write_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        modal_runner,
        "run_subprocess",
        lambda command, **kwargs: calls.append(command) or 0,
    )

    result = modal_runner.execute_remote_payload(
        {"command_kind": "modal-run", "config": config.__dict__}
    )

    assert result == 0
    assert calls[0][4:7] == [
        "scripts/launch_training.py",
        "execute",
        "/workspace/rlrmp/results/19d6acd/runs/matrix.json",
    ]
    assert calls[0][-5:] == ["--row", "row_a", "--resume", "--log-step", "1"]


def test_billable_modal_run_requires_confirmation() -> None:
    args = modal_runner.build_parser().parse_args(
        ["modal-run", "--document", "results/19d6acd/runs/matrix.json"]
    )
    with pytest.raises(SystemExit, match="Refusing billable Modal training launch"):
        modal_runner.require_billable_launch_confirmation(args)


def test_packing_command_retains_only_operational_cli_controls() -> None:
    args = modal_runner.build_parser().parse_args(
        [
            "modal-packing-smoke",
            "--experiment",
            "packing-benchmark",
            "--n-workers",
            "2",
            "--packing-jax-platform",
            "cpu",
            "--packing-cpu-threads-per-worker",
            "1",
        ]
    )
    command = modal_runner.build_packing_benchmark_command(
        modal_runner.make_config(args), remote=True
    )

    assert command[command.index("--n-workers") + 1] == "2"
    assert command[command.index("--jax-platform") + 1] == "cpu"
    scenario = json.loads(command[command.index("--scenario-config-json") + 1])
    assert scenario["batch_size"] == modal_runner.DEFAULT_BATCH_SIZE


def test_modal_app_script_uses_feedbax_image_renderer_and_exact_payload() -> None:
    script = Path("scripts/modal_cs_nominal_gru.py").read_text(encoding="utf-8")

    assert "render_modal_app" in script
    assert "build_modal_image_execution_spec" in script
    assert "build_launcher_spec_bundle" not in script
    assert '"config": config.__dict__' in script
    assert "extra_args" not in script
    assert "add_local_dir" not in script


def test_modal_runner_has_no_config_to_spec_compatibility_surface() -> None:
    assert not hasattr(modal_runner, "build_launcher_spec_bundle")
    assert not hasattr(modal_runner, "spec_lock_payload")
    assert not hasattr(modal_runner, "materialize_training_run_spec")


def test_modal_image_spec_has_no_training_document_or_inline_spec() -> None:
    spec = modal_runner.build_modal_image_execution_spec()

    assert spec.kind == "custom"
    assert spec.command == "true"
    assert spec.training_run_spec is None
    assert {source.name for source in spec.repos} == {"rlrmp", "feedbax", "jax-cookbook"}

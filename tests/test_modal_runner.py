from __future__ import annotations

from rlrmp.modal_runner import (
    DEFAULT_RUN,
    MODAL_VOLUME_NAME,
    REGULARIZED_RUN,
    NominalGruRunConfig,
    build_packing_benchmark_command,
    build_remote_smoke_command,
    build_training_command,
    dry_run_payload,
)


def test_nominal_training_command_is_bounded_and_nominal_only() -> None:
    config = NominalGruRunConfig(
        experiment="30f2313",
        run="smoke__unit",
        n_train_batches=1,
        batch_size=2,
        n_replicates=1,
        hidden_size=4,
        resume=False,
    )

    command = build_training_command(config)

    assert command[:4] == ["uv", "run", "python", "scripts/train_cs_nominal_gru.py"]
    assert "--n-adversary-batches" not in command
    assert command[command.index("--n-train-batches") + 1] == "1"
    assert command[command.index("--hidden-size") + 1] == "4"
    assert command[command.index("--stochastic-preset") + 1] == "cs2019-rollout"
    assert command[command.index("--checkpoint-interval-batches") + 1] == "500"
    assert "--full-train" in command
    assert "--resume" not in command
    assert "--regularized-fidelity" not in command


def test_regularized_training_command_uses_hidden_penalty_switch() -> None:
    config = NominalGruRunConfig(run=REGULARIZED_RUN, regularized_fidelity=True)

    command = build_training_command(config)

    assert "--regularized-fidelity" in command
    assert "--nn-hidden" not in command


def test_remote_training_command_uses_no_sync_and_remote_paths() -> None:
    config = NominalGruRunConfig(experiment="30f2313", run=DEFAULT_RUN)

    command = build_training_command(config, remote=True)

    assert command[:5] == ["uv", "run", "--no-sync", "python", "scripts/train_cs_nominal_gru.py"]
    assert f"/vol/rlrmp-cs-stochastic-gru/_artifacts/30f2313/runs/{DEFAULT_RUN}" in command
    assert f"/vol/rlrmp-cs-stochastic-gru/results/30f2313/runs/{DEFAULT_RUN}" in command
    assert "--full-train" in command
    assert "--resume" in command


def test_pinned_mode_uses_configured_repo_dir() -> None:
    config = NominalGruRunConfig(
        experiment="30f2313",
        run=DEFAULT_RUN,
        mode="pinned",
        pinned_repo_dir="/opt/rlrmp",
    )

    command = build_training_command(config, remote=True)

    assert f"/vol/rlrmp-cs-stochastic-gru/_artifacts/30f2313/runs/{DEFAULT_RUN}" in command
    assert f"/vol/rlrmp-cs-stochastic-gru/results/30f2313/runs/{DEFAULT_RUN}" in command


def test_regularized_modal_command_selects_hidden_penalty_pair() -> None:
    command = build_training_command(
        NominalGruRunConfig(
            run=REGULARIZED_RUN,
            regularized_fidelity=True,
        ),
        remote=True,
    )

    assert "--regularized-fidelity" in command
    assert "cs_stochastic_gru__hidden_penalty" in command[command.index("--output-dir") + 1]
    assert "cs_stochastic_gru__hidden_penalty" in command[command.index("--spec-dir") + 1]


def test_dry_run_payload_exposes_no_warm_container_settings() -> None:
    payload = dry_run_payload(NominalGruRunConfig(gpu="A10G", timeout_seconds=90))

    assert payload["gpu"] == "A10G"
    assert payload["timeout_seconds"] == 90
    assert payload["stochastic_preset"] == "cs2019-rollout"
    assert payload["modal_volume_name"] == MODAL_VOLUME_NAME
    assert payload["warm_containers"] == 0
    assert payload["min_containers"] == 0
    assert payload["max_containers"] == 1
    assert payload["remote_smoke_command"] == build_remote_smoke_command()
    assert payload["remote_packing_benchmark_command"] == build_packing_benchmark_command(
        NominalGruRunConfig(gpu="A10G", timeout_seconds=90),
        remote=True,
    )
    planned = payload["planned_stochastic_runs"]
    assert planned["stochastic_no_hidden_penalty"]["run"] == DEFAULT_RUN
    assert planned["stochastic_no_hidden_penalty"]["nn_hidden"] == 0.0
    assert planned["stochastic_hidden_penalty"]["run"] == REGULARIZED_RUN
    assert planned["stochastic_hidden_penalty"]["nn_hidden"] == 1e-5
    assert (
        "--regularized-fidelity"
        in planned["stochastic_hidden_penalty"]["remote_training_command"]
    )
    pull = payload["modal_volume_pull_commands"]
    assert pull["artifacts"][:4] == ["modal", "volume", "get", MODAL_VOLUME_NAME]
    assert pull["artifacts"][4] == f"_artifacts/30f2313/runs/{DEFAULT_RUN}"
    assert pull["specs"][4] == f"results/30f2313/runs/{DEFAULT_RUN}"


def test_packing_benchmark_command_disables_sync_and_sets_worker_count() -> None:
    command = build_packing_benchmark_command(
        NominalGruRunConfig(
            run="packing_a10_n2",
            n_workers=2,
            burn_in_seconds=45,
            measure_seconds=60,
            warmup_batches=1,
            chunk_batches=5,
        ),
        remote=True,
    )

    assert command[:6] == [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "rlrmp.modal_packing_benchmark",
    ]
    assert command[command.index("--n-workers") + 1] == "2"
    assert command[command.index("--burn-in-seconds") + 1] == "45"
    assert command[command.index("--measure-seconds") + 1] == "60"
    assert command[command.index("--stochastic-preset") + 1] == "cs2019-rollout"
    assert "--nn-hidden" not in command
    assert "--regularized-fidelity" not in command

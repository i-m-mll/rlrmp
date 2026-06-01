from __future__ import annotations

from rlrmp.modal_runner import (
    NominalGruRunConfig,
    build_packing_benchmark_command,
    build_remote_smoke_command,
    build_training_command,
    dry_run_payload,
)


def test_nominal_training_command_is_bounded_and_nominal_only() -> None:
    config = NominalGruRunConfig(
        experiment="a1a8e39",
        run="smoke__unit",
        n_train_batches=1,
        batch_size=2,
        n_replicates=1,
    )

    command = build_training_command(config)

    assert command[:4] == ["uv", "run", "python", "scripts/train_cs_nominal_gru.py"]
    assert "--n-adversary-batches" not in command
    assert command[command.index("--n-train-batches") + 1] == "1"
    assert command[command.index("--hidden-size") + 1] == "4"


def test_remote_training_command_uses_no_sync_and_remote_paths() -> None:
    config = NominalGruRunConfig(experiment="a1a8e39", run="nominal_cs_gru__modal_prep")

    command = build_training_command(config, remote=True)

    assert command[:5] == ["uv", "run", "--no-sync", "python", "scripts/train_cs_nominal_gru.py"]
    assert "/workspace/rlrmp/_artifacts/a1a8e39/runs/nominal_cs_gru__modal_prep" in command
    assert "/workspace/rlrmp/results/a1a8e39/runs/nominal_cs_gru__modal_prep" in command


def test_pinned_mode_uses_configured_repo_dir() -> None:
    config = NominalGruRunConfig(
        experiment="a1a8e39",
        run="nominal_cs_gru__modal_prep",
        mode="pinned",
        pinned_repo_dir="/opt/rlrmp",
    )

    command = build_training_command(config, remote=True)

    assert "/opt/rlrmp/_artifacts/a1a8e39/runs/nominal_cs_gru__modal_prep" in command
    assert "/opt/rlrmp/results/a1a8e39/runs/nominal_cs_gru__modal_prep" in command


def test_dry_run_payload_exposes_no_warm_container_settings() -> None:
    payload = dry_run_payload(NominalGruRunConfig(gpu="A10G", timeout_seconds=90))

    assert payload["gpu"] == "A10G"
    assert payload["timeout_seconds"] == 90
    assert payload["warm_containers"] == 0
    assert payload["min_containers"] == 0
    assert payload["max_containers"] == 1
    assert payload["remote_smoke_command"] == build_remote_smoke_command()
    assert payload["remote_packing_benchmark_command"] == build_packing_benchmark_command(
        NominalGruRunConfig(gpu="A10G", timeout_seconds=90),
        remote=True,
    )


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
    assert "--nn-hidden" not in command
    assert "--regularized-fidelity" not in command

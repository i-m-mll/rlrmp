from __future__ import annotations

from rlrmp.modal_runner import (
    NominalGruRunConfig,
    build_remote_smoke_command,
    build_training_command,
    dry_run_payload,
)


def test_nominal_training_command_is_bounded_and_nominal_only() -> None:
    config = NominalGruRunConfig(
        experiment="18ae684",
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
    config = NominalGruRunConfig(experiment="18ae684", run="nominal_cs_gru__modal_prep")

    command = build_training_command(config, remote=True)

    assert command[:5] == ["uv", "run", "--no-sync", "python", "scripts/train_cs_nominal_gru.py"]
    assert "/workspace/rlrmp/_artifacts/18ae684/runs/nominal_cs_gru__modal_prep" in command
    assert "/workspace/rlrmp/results/18ae684/runs/nominal_cs_gru__modal_prep" in command


def test_pinned_mode_uses_configured_repo_dir() -> None:
    config = NominalGruRunConfig(
        experiment="18ae684",
        run="nominal_cs_gru__modal_prep",
        mode="pinned",
        pinned_repo_dir="/opt/rlrmp",
    )

    command = build_training_command(config, remote=True)

    assert "/opt/rlrmp/_artifacts/18ae684/runs/nominal_cs_gru__modal_prep" in command
    assert "/opt/rlrmp/results/18ae684/runs/nominal_cs_gru__modal_prep" in command


def test_dry_run_payload_exposes_no_warm_container_settings() -> None:
    payload = dry_run_payload(NominalGruRunConfig(gpu="A10G", timeout_seconds=90))

    assert payload["gpu"] == "A10G"
    assert payload["timeout_seconds"] == 90
    assert payload["warm_containers"] == 0
    assert payload["min_containers"] == 0
    assert payload["max_containers"] == 1
    assert payload["remote_smoke_command"] == build_remote_smoke_command()

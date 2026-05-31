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
        n_warmup_batches=1,
        batch_size=2,
        n_replicates=1,
    )

    command = build_training_command(config)

    assert command[:4] == ["uv", "run", "python", "scripts/train_minimax.py"]
    assert command[command.index("--n-adversary-batches") + 1] == "0"
    assert command[command.index("--hidden-type") + 1] == "gru"
    assert command[command.index("--effector-hold-pos") + 1] == "0.0"
    assert command[command.index("--effector-hold-vel") + 1] == "0.0"
    assert command[command.index("--effector-pos-running-schedule") + 1] == "movement_ramp"


def test_remote_training_command_uses_no_sync_and_remote_paths() -> None:
    config = NominalGruRunConfig(experiment="18ae684", run="nominal_cs_gru__modal_prep")

    command = build_training_command(config, remote=True)

    assert command[:5] == ["uv", "run", "--no-sync", "python", "scripts/train_minimax.py"]
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

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.modal_runner import (
    DEFAULT_RUN,
    DEFAULT_GPU,
    DEFAULT_TRAIN_TIMEOUT_SECONDS,
    MODAL_VOLUME_NAME,
    REGULARIZED_RUN,
    NominalGruRunConfig,
    activate_project_venv,
    build_parser,
    build_packing_benchmark_command,
    build_remote_smoke_command,
    build_training_command,
    collect_source_provenance,
    dry_run_payload,
    make_config,
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
    assert command[command.index("--issue") + 1] == "30f2313"
    assert command[command.index("--hidden-size") + 1] == "4"
    assert command[command.index("--stochastic-preset") + 1] == "cs2019-rollout"
    assert command[command.index("--checkpoint-interval-batches") + 1] == "500"
    assert "--full-train" in command
    assert "--resume" not in command
    assert "--regularized-fidelity" not in command


def test_training_command_passes_optimizer_grid_parameters() -> None:
    config = NominalGruRunConfig(
        controller_lr=3e-3,
        gradient_clip_norm=5.0,
    )

    command = build_training_command(config, remote=True)

    assert command[command.index("--controller-lr") + 1] == "0.003"
    assert command[command.index("--gradient-clip-norm") + 1] == "5.0"


def test_training_command_passes_loss_objective() -> None:
    config = NominalGruRunConfig(loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE)

    command = build_training_command(config, remote=True)

    assert command[command.index("--loss-objective") + 1] == "full_analytical_qrf"

    ablation = build_training_command(
        NominalGruRunConfig(loss_objective=CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE),
        remote=True,
    )

    assert ablation[ablation.index("--loss-objective") + 1] == "partial_net_output_force_filter"


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
    payload = dry_run_payload(
        NominalGruRunConfig(
            gpu="A10G",
            timeout_seconds=90,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        )
    )

    assert payload["gpu"] == "A10G"
    assert payload["timeout_seconds"] == 90
    assert payload["stochastic_preset"] == "cs2019-rollout"
    assert payload["loss_objective"] == "full_analytical_qrf"
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
        "--regularized-fidelity" in planned["stochastic_hidden_penalty"]["remote_training_command"]
    )
    pull = payload["modal_volume_pull_commands"]
    assert pull["artifacts"][:4] == ["modal", "volume", "get", MODAL_VOLUME_NAME]
    assert pull["artifacts"][4] == f"_artifacts/30f2313/runs/{DEFAULT_RUN}"
    assert pull["specs"][4] == f"results/30f2313/runs/{DEFAULT_RUN}"
    assert payload["modal_volume_sync_command"] == [
        "uv",
        "run",
        "python",
        "scripts/sync_modal_run_artifacts.py",
        "--issue",
        "30f2313",
        "--run",
        DEFAULT_RUN,
    ]


def test_modal_run_defaults_to_training_timeout() -> None:
    args = build_parser().parse_args(
        ["modal-run", "--loss-objective", "full_analytical_qrf"]
    )

    config = make_config(args)

    assert config.timeout_seconds == DEFAULT_TRAIN_TIMEOUT_SECONDS
    assert config.gpu == DEFAULT_GPU == "A10"
    assert config.loss_objective == "full_analytical_qrf"


def test_activate_project_venv_exposes_uv_site_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_dir = tmp_path / ".venv"
    site_packages = venv_dir / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    editable_src = tmp_path / "editable-src"
    editable_src.mkdir()
    (site_packages / "editable.pth").write_text(str(editable_src) + "\n")
    modal_deps = tmp_path / "__modal" / "deps"
    modal_deps.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin")
    original_path = list(sys.path)

    try:
        sys.path.insert(0, str(modal_deps))
        activated = activate_project_venv(venv_dir)

        assert activated == site_packages
        assert sys.path.index(str(site_packages)) < sys.path.index(str(modal_deps))
        assert sys.path.index(str(editable_src)) < sys.path.index(str(modal_deps))
        assert os.environ["VIRTUAL_ENV"] == str(venv_dir)
        assert os.environ["PATH"].split(os.pathsep)[0] == str(venv_dir / "bin")
    finally:
        sys.path[:] = original_path


def test_activate_project_venv_prefers_venv_package_over_modal_deps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_dir = tmp_path / ".venv"
    site_packages = venv_dir / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "typing_extensions.py").write_text("Sentinel = object()\n")
    modal_deps = tmp_path / "__modal" / "deps"
    modal_deps.mkdir(parents=True)
    (modal_deps / "typing_extensions.py").write_text("_Sentinel = object()\n")
    original_path = list(sys.path)
    old_module = sys.modules.pop("typing_extensions", None)

    try:
        sys.path.insert(0, str(modal_deps))
        shadow_module = importlib.import_module("typing_extensions")
        assert not hasattr(shadow_module, "Sentinel")
        assert Path(shadow_module.__file__).parent == modal_deps

        activate_project_venv(venv_dir)
        module = importlib.import_module("typing_extensions")

        assert hasattr(module, "Sentinel")
        assert Path(module.__file__).parent == site_packages
    finally:
        sys.path[:] = original_path
        sys.modules.pop("typing_extensions", None)
        if old_module is not None:
            sys.modules["typing_extensions"] = old_module


def test_collect_source_provenance_reports_commit_and_status() -> None:
    provenance = collect_source_provenance()

    assert provenance["commit"]
    assert "status_short" in provenance


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

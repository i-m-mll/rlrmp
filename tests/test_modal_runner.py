from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlrmp import packing_benchmark
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
    cs_nominal_gru_scenario_config,
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
        NominalGruRunConfig(
            gpu="A10G",
            timeout_seconds=90,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
        ),
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
        "rlrmp.packing_benchmark",
    ]
    assert command[command.index("--n-workers") + 1] == "2"
    assert command[command.index("--burn-in-seconds") + 1] == "45"
    assert command[command.index("--measure-seconds") + 1] == "60"
    assert command[command.index("--scenario") + 1] == "cs-nominal-gru"
    scenario_config = json.loads(command[command.index("--scenario-config-json") + 1])
    assert scenario_config["stochastic_preset"] == "cs2019-rollout"
    assert "--nn-hidden" not in command
    assert "--regularized-fidelity" not in command
    assert "--batch-size" not in command


def test_packing_benchmark_command_can_match_b8aa38e_proprio_contract() -> None:
    command = build_packing_benchmark_command(
        NominalGruRunConfig(
            experiment="b8aa38e",
            run="packing_titan_proprio_cal_stress_b64_n2",
            n_workers=2,
            batch_size=64,
            controller_lr=1e-3,
            lr_warmup_batches=500,
            lr_warmup_init_fraction=0.1,
            lr_cosine_alpha=0.01,
            gradient_clip_norm=5.0,
            loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            perturbation_training=True,
            perturbation_calibrated_timing=True,
            perturbation_physical_level="stress",
            schedule_total_batches=1000,
            training_diagnostics=True,
        ),
        remote=True,
    )

    scenario_config = json.loads(command[command.index("--scenario-config-json") + 1])
    assert scenario_config["batch_size"] == 64
    assert scenario_config["controller_lr"] == 0.001
    assert scenario_config["lr_warmup_batches"] == 500
    assert scenario_config["lr_cosine_alpha"] == 0.01
    assert scenario_config["gradient_clip_norm"] == 5.0
    assert scenario_config["loss_objective"] == "full_analytical_qrf"
    assert scenario_config["schedule_total_batches"] == 1000
    assert scenario_config["perturbation_physical_level"] == "stress"
    assert scenario_config["target_relative_multitarget"] is True
    assert scenario_config["force_filter_feedback"] is True
    assert scenario_config["perturbation_training"] is True
    assert scenario_config["perturbation_calibrated_timing"] is True
    assert scenario_config["training_diagnostics"] is True
    assert "--target-relative-multitarget" not in command
    assert "--perturbation-training" not in command


def test_packing_benchmark_command_passes_pgd_as_scenario_payload() -> None:
    command = build_packing_benchmark_command(
        NominalGruRunConfig(
            experiment="4d79e07",
            run="packing_4090_pgd_b64_n2",
            n_workers=2,
            batch_size=64,
            controller_lr=3e-3,
            gradient_clip_norm=5.0,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            broad_epsilon_pgd_training=True,
            broad_epsilon_pgd_level="strong",
            broad_epsilon_pgd_budget_scale=1.5,
            broad_epsilon_pgd_steps=8,
            broad_epsilon_pgd_step_size_fraction=0.5,
            broad_epsilon_pgd_seed=123,
        ),
        remote=True,
    )

    scenario_config = json.loads(command[command.index("--scenario-config-json") + 1])

    assert scenario_config["broad_epsilon_pgd_training"] is True
    assert scenario_config["broad_epsilon_pgd_level"] == "strong"
    assert scenario_config["broad_epsilon_pgd_budget_scale"] == 1.5
    assert scenario_config["broad_epsilon_pgd_steps"] == 8
    assert scenario_config["broad_epsilon_pgd_step_size_fraction"] == 0.5
    assert scenario_config["broad_epsilon_pgd_seed"] == 123
    assert "--broad-epsilon-pgd-training" not in command


def test_packing_benchmark_command_passes_opt_in_cpu_runtime_controls() -> None:
    command = build_packing_benchmark_command(
        NominalGruRunConfig(
            n_workers=4,
            packing_jax_platform="cpu",
            packing_cpu_threads_per_worker=1,
            packing_jax_compilation_cache_dir="/tmp/rlrmp-jax-cache",
            packing_jax_persistent_cache_min_compile_time_secs=0.0,
            packing_jax_persistent_cache_min_entry_size_bytes=-1,
        ),
        remote=False,
    )

    assert command[command.index("--jax-platform") + 1] == "cpu"
    assert command[command.index("--cpu-threads-per-worker") + 1] == "1"
    assert (
        command[command.index("--jax-compilation-cache-dir") + 1]
        == "/tmp/rlrmp-jax-cache"
    )
    assert command[command.index("--jax-persistent-cache-min-compile-time-secs") + 1] == "0.0"
    assert command[command.index("--jax-persistent-cache-min-entry-size-bytes") + 1] == "-1"
    assert command[command.index("--n-workers") + 1] == "4"


def test_packing_benchmark_command_leaves_gpu_defaults_unconstrained() -> None:
    command = build_packing_benchmark_command(NominalGruRunConfig(), remote=True)

    assert "--jax-platform" not in command
    assert "--cpu-threads-per-worker" not in command
    assert "--jax-compilation-cache-dir" not in command


def test_modal_runner_parser_builds_pgd_scenario_config() -> None:
    args = build_parser().parse_args(
        [
            "modal-packing-smoke",
            "--target-relative-multitarget",
            "--force-filter-feedback",
            "--broad-epsilon-pgd-training",
            "--broad-epsilon-pgd-level",
            "strong",
            "--broad-epsilon-pgd-budget-scale",
            "1.5",
            "--broad-epsilon-pgd-steps",
            "8",
            "--broad-epsilon-pgd-step-size-fraction",
            "0.5",
            "--broad-epsilon-pgd-seed",
            "123",
        ]
    )
    scenario_config = cs_nominal_gru_scenario_config(make_config(args))

    assert scenario_config["target_relative_multitarget"] is True
    assert scenario_config["force_filter_feedback"] is True
    assert scenario_config["broad_epsilon_pgd_training"] is True
    assert scenario_config["broad_epsilon_pgd_level"] == "strong"
    assert scenario_config["broad_epsilon_pgd_budget_scale"] == 1.5
    assert scenario_config["broad_epsilon_pgd_steps"] == 8
    assert scenario_config["broad_epsilon_pgd_step_size_fraction"] == 0.5
    assert scenario_config["broad_epsilon_pgd_seed"] == 123


def test_modal_runner_parser_builds_packing_runtime_config() -> None:
    args = build_parser().parse_args(
        [
            "modal-packing-smoke",
            "--packing-jax-platform",
            "cpu",
            "--packing-cpu-threads-per-worker",
            "2",
            "--packing-jax-compilation-cache-dir",
            "/cache/jax",
            "--packing-jax-persistent-cache-min-compile-time-secs",
            "0",
            "--packing-jax-persistent-cache-min-entry-size-bytes",
            "-1",
        ]
    )
    config = make_config(args)

    assert config.packing_jax_platform == "cpu"
    assert config.packing_cpu_threads_per_worker == 2
    assert config.packing_jax_compilation_cache_dir == "/cache/jax"
    assert config.packing_jax_persistent_cache_min_compile_time_secs == 0.0
    assert config.packing_jax_persistent_cache_min_entry_size_bytes == -1


def test_provider_neutral_packing_parser_is_scenario_driven() -> None:
    args = packing_benchmark.build_parser().parse_args(
        [
            "parent",
            "--output-dir",
            "/tmp/out",
            "--n-workers",
            "2",
            "--scenario",
            "custom.module:factory",
            "--scenario-config-json",
            '{"name": "row-a"}',
        ]
    )

    assert args.scenario == "custom.module:factory"
    assert args.scenario_config_json == '{"name": "row-a"}'
    assert not hasattr(args, "batch_size")


def test_packing_worker_env_forced_cpu_caps_threads_and_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "jax-cache"
    args = packing_benchmark.build_parser().parse_args(
        [
            "parent",
            "--output-dir",
            str(tmp_path / "out"),
            "--n-workers",
            "2",
            "--jax-platform",
            "cpu",
            "--cpu-threads-per-worker",
            "1",
            "--jax-compilation-cache-dir",
            str(cache_dir),
            "--jax-persistent-cache-min-compile-time-secs",
            "0",
            "--jax-persistent-cache-min-entry-size-bytes",
            "-1",
        ]
    )
    env = {}

    metadata = packing_benchmark._configure_worker_env(env, args)

    assert env["JAX_PLATFORM_NAME"] == "cpu"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["TF_NUM_INTRAOP_THREADS"] == "1"
    assert "intra_op_parallelism_threads=1" in env["XLA_FLAGS"]
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(cache_dir)
    assert env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0.0"
    assert env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "-1"
    assert cache_dir.is_dir()
    assert metadata["compilation_cache"]["expected_effect"] == "startup_compile_and_warmup_only"


def test_packing_worker_env_defaults_do_not_force_gpu_or_cpu_caps(tmp_path: Path) -> None:
    args = packing_benchmark.build_parser().parse_args(
        ["parent", "--output-dir", str(tmp_path / "out"), "--n-workers", "1"]
    )
    env = {}

    metadata = packing_benchmark._configure_worker_env(env, args)

    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "JAX_PLATFORM_NAME" not in env
    assert "JAX_PLATFORMS" not in env
    assert "OMP_NUM_THREADS" not in env
    assert "XLA_FLAGS" not in env
    assert metadata["compilation_cache"]["enabled"] is False


def test_packing_cs_nominal_gru_scenario_wires_pgd_pre_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlrmp.modules.training.part2 as part2
    import rlrmp.train.cs_nominal_gru as nominal
    import rlrmp.train.cs_perturbation_training as perturbation

    pgd_hps = SimpleNamespace(enabled=True)
    pre_step_fn = object()

    class Parser:
        def parse_args(self, args: list[str]) -> argparse.Namespace:
            assert args == []
            return argparse.Namespace()

    monkeypatch.setattr(nominal, "build_parser", lambda: Parser())
    monkeypatch.setattr(
        nominal,
        "build_hps",
        lambda args: SimpleNamespace(batch_size=2, broad_epsilon_pgd_training=pgd_hps),
    )
    monkeypatch.setattr(nominal, "_build_trainer", lambda hps: object())
    monkeypatch.setattr(
        perturbation,
        "make_broad_epsilon_pgd_pre_step",
        lambda config: pre_step_fn if config is pgd_hps else None,
    )
    monkeypatch.setattr(
        part2,
        "setup_task_model_pair",
        lambda hps, key: SimpleNamespace(task=object(), model=object()),
    )
    monkeypatch.setattr(packing_benchmark, "_cs_nominal_gru_metadata", lambda hps: {})

    runtime = packing_benchmark.build_cs_nominal_gru_scenario(
        {"broad_epsilon_pgd_training": True},
        seed=1,
    )

    assert runtime.pre_step_fn is pre_step_fn


def test_packing_cs_nominal_gru_argv_does_not_clobber_payload_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlrmp.modules.training.part2 as part2
    import rlrmp.train.cs_nominal_gru as nominal

    captured = {}

    monkeypatch.setattr(nominal, "_build_trainer", lambda hps: object())

    def fake_build_hps(args: argparse.Namespace) -> SimpleNamespace:
        captured.update(vars(args))
        return SimpleNamespace(batch_size=2, broad_epsilon_pgd_training=SimpleNamespace())

    monkeypatch.setattr(nominal, "build_hps", fake_build_hps)
    monkeypatch.setattr(
        part2,
        "setup_task_model_pair",
        lambda hps, key: SimpleNamespace(task=object(), model=object()),
    )
    monkeypatch.setattr(packing_benchmark, "_cs_nominal_gru_metadata", lambda hps: {})

    packing_benchmark.build_cs_nominal_gru_scenario(
        {
            "target_relative_multitarget": True,
            "force_filter_feedback": True,
            "loss_objective": "full_analytical_qrf",
            "argv": ["--delayed-reach"],
        },
        seed=1,
    )

    assert captured["delayed_reach"] is True
    assert captured["target_relative_multitarget"] is True
    assert captured["force_filter_feedback"] is True
    assert captured["loss_objective"] == "full_analytical_qrf"


def test_packing_timed_train_uses_scenario_runtime_interface() -> None:
    class FakeRuntime:
        metadata: dict[str, object] = {}

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def warmup(self, n_batches: int) -> int:
            return n_batches

        def train_chunk(self, model: int, n_batches: int) -> int:
            self.calls.append((model, n_batches))
            return model + n_batches

    runtime = FakeRuntime()

    summary = packing_benchmark._timed_train(
        runtime=runtime,
        model=0,
        seconds=0.0,
        chunk_batches=3,
    )

    assert runtime.calls == [(0, 3)]
    assert summary["model"] == 3
    assert summary["batches"] == 3


def test_packing_aggregate_reports_cpu_rss_fallback() -> None:
    workers = [
        {"status": "done", "measured": {"batches_per_second": 1.5}},
        {"status": "done", "measured": {"batches_per_second": 2.5}},
    ]
    samples = [
        {"kind": "rss", "total_rss_mib": 100.0, "max_worker_rss_mib": 60.0},
        {"kind": "rss", "total_rss_mib": 150.0, "max_worker_rss_mib": 80.0},
    ]

    summary = packing_benchmark._aggregate(workers, samples)

    assert summary["aggregate_batches_per_second"] == 4.0
    assert summary["max_total_rss_mib"] == 150.0
    assert summary["max_worker_rss_mib"] == 80.0

"""Modal launch helpers for stochastic C&S-fidelity GRU preparation.

This module keeps cloud execution behind explicit CLI choices. The default
path is a local dry-run that prints the command and provenance paths without
starting a Modal container.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import site
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir

APP_NAME = "rlrmp-cs-stochastic-gru"
DEFAULT_EXPERIMENT = "30f2313"
DEFAULT_RUN = "cs_stochastic_gru__no_hidden_penalty"
REGULARIZED_RUN = "cs_stochastic_gru__hidden_penalty"
DEFAULT_STOCHASTIC_PRESET = "cs2019-rollout"
CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE = "partial_feedbax_terms"
CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE = "partial_net_output_force_filter"
CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE = "full_analytical_qrf"
CS_LOSS_OBJECTIVES = (
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
)
DEFAULT_GPU = "A10"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_TRAIN_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_N_TRAIN_BATCHES = 12000
DEFAULT_BATCH_SIZE = 250
DEFAULT_N_REPLICATES = 5
DEFAULT_HIDDEN_SIZE = 180
DEFAULT_CHECKPOINT_INTERVAL_BATCHES = 500
MODAL_VOLUME_NAME = "rlrmp-cs-stochastic-gru"
MODAL_VOLUME_MOUNT = Path("/vol/rlrmp-cs-stochastic-gru")
REMOTE_REPO_DIR = Path("/workspace/rlrmp")
REMOTE_FEEDBAX_DIR = Path("/workspace/feedbax")
REMOTE_JAX_COOKBOOK_DIR = Path("/workspace/jax-cookbook")
REMOTE_VENV_DIR = REMOTE_REPO_DIR / ".venv"
LOCAL_FEEDBAX_DIR = Path("/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax")
LOCAL_JAX_COOKBOOK_DIR = Path("/Users/mll/Main/10 Projects/05 Utils/jax-cookbook")

Mode = Literal["source", "pinned"]
CommandKind = Literal["dry-run", "local-smoke", "modal-smoke", "modal-run", "modal-packing-smoke"]


@dataclass(frozen=True)
class NominalGruRunConfig:
    """Command-level configuration for the stochastic GRU run."""

    experiment: str = DEFAULT_EXPERIMENT
    run: str = DEFAULT_RUN
    n_train_batches: int = DEFAULT_N_TRAIN_BATCHES
    batch_size: int = DEFAULT_BATCH_SIZE
    n_replicates: int = DEFAULT_N_REPLICATES
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    seed: int = 42
    controller_lr: float = 1e-2
    lr_warmup_batches: int = 0
    lr_warmup_init_fraction: float = 0.1
    lr_cosine_alpha: float = 1.0
    gradient_clip_norm: float | None = None
    stochastic_preset: str = DEFAULT_STOCHASTIC_PRESET
    loss_objective: str = CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
    regularized_fidelity: bool = False
    checkpoint_interval_batches: int = DEFAULT_CHECKPOINT_INTERVAL_BATCHES
    resume: bool = True
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    gpu: str = DEFAULT_GPU
    mode: Mode = "source"
    pinned_repo_dir: str = str(REMOTE_REPO_DIR)
    extra_args: tuple[str, ...] = ()
    n_workers: int = 1
    stagger_seconds: float = 10.0
    burn_in_seconds: float = 60.0
    measure_seconds: float = 60.0
    warmup_batches: int = 2
    chunk_batches: int = 4
    ready_timeout_seconds: float = 900.0
    sample_seconds: float = 5.0
    packing_jax_platform: str | None = None
    packing_cpu_threads_per_worker: int = 1
    packing_jax_compilation_cache_dir: str | None = None
    packing_jax_persistent_cache_min_compile_time_secs: float | None = None
    packing_jax_persistent_cache_min_entry_size_bytes: int | None = None
    target_relative_multitarget: bool = False
    force_filter_feedback: bool = False
    perturbation_training: bool = False
    perturbation_calibrated_timing: bool = False
    perturbation_physical_level: str = "moderate"
    broad_epsilon_training: bool = False
    broad_epsilon_level: str = "moderate"
    broad_epsilon_budget_scale: float = 1.0
    broad_epsilon_reach_scaling: bool = True
    broad_epsilon_pgd_training: bool = False
    broad_epsilon_pgd_level: str = "moderate"
    broad_epsilon_pgd_budget_scale: float = 1.0
    broad_epsilon_pgd_steps: int = 3
    broad_epsilon_pgd_step_size_fraction: float = 0.25
    broad_epsilon_pgd_seed: int | None = None
    initial_hidden_encoder: bool = False
    training_diagnostics: bool = True
    schedule_total_batches: int = 1000

    def local_artifact_dir(self) -> Path:
        return run_artifact_dir(self.experiment, self.run)

    def local_spec_dir(self) -> Path:
        return run_spec_dir(self.experiment, self.run)

    def remote_artifact_dir(self) -> Path:
        return MODAL_VOLUME_MOUNT / "_artifacts" / self.experiment / "runs" / self.run

    def remote_spec_dir(self) -> Path:
        return MODAL_VOLUME_MOUNT / "results" / self.experiment / "runs" / self.run

    def remote_repo_dir(self) -> Path:
        if self.mode == "pinned":
            return Path(self.pinned_repo_dir)
        return REMOTE_REPO_DIR


def _append_arg(command: list[str], flag: str, value: str | int | float) -> None:
    command.extend([flag, str(value)])


def build_training_command(
    config: NominalGruRunConfig,
    *,
    remote: bool = False,
) -> list[str]:
    """Build the stochastic C&S-fidelity GRU training command.

    The command intentionally targets the dedicated C&S GRU trainer,
    not the older delayed-reach minimax trainer.
    """

    artifact_dir = config.remote_artifact_dir() if remote else config.local_artifact_dir()
    spec_dir = config.remote_spec_dir() if remote else config.local_spec_dir()
    command = [
        "uv",
        "run",
        "--no-sync" if remote else "",
        "python",
        "scripts/train_cs_nominal_gru.py",
    ]
    command = [part for part in command if part]
    _append_arg(command, "--n-train-batches", config.n_train_batches)
    _append_arg(command, "--issue", config.experiment)
    _append_arg(command, "--batch-size", config.batch_size)
    _append_arg(command, "--n-replicates", config.n_replicates)
    _append_arg(command, "--hidden-size", config.hidden_size)
    _append_arg(command, "--seed", config.seed)
    _append_arg(command, "--controller-lr", config.controller_lr)
    _append_arg(command, "--lr-warmup-batches", config.lr_warmup_batches)
    _append_arg(command, "--lr-warmup-init-fraction", config.lr_warmup_init_fraction)
    _append_arg(command, "--lr-cosine-alpha", config.lr_cosine_alpha)
    if config.gradient_clip_norm is not None:
        _append_arg(command, "--gradient-clip-norm", config.gradient_clip_norm)
    _append_arg(command, "--stochastic-preset", config.stochastic_preset)
    _append_arg(command, "--loss-objective", config.loss_objective)
    _append_arg(command, "--output-dir", artifact_dir)
    _append_arg(command, "--spec-dir", spec_dir)
    _append_arg(command, "--checkpoint-interval-batches", config.checkpoint_interval_batches)
    command.append("--full-train")
    if config.resume:
        command.append("--resume")
    if config.regularized_fidelity:
        command.append("--regularized-fidelity")
    command.extend(config.extra_args)
    return command


def build_training_script_args(
    config: NominalGruRunConfig,
    *,
    remote: bool = False,
) -> list[str]:
    """Return argv for ``scripts/train_cs_nominal_gru.py`` without uv/python."""

    command = build_training_command(config, remote=remote)
    script_index = command.index("scripts/train_cs_nominal_gru.py")
    return command[script_index + 1 :]


def cs_nominal_gru_scenario_config(config: NominalGruRunConfig) -> dict[str, Any]:
    """Return the C&S nominal GRU scenario payload for the packing benchmark."""

    payload: dict[str, Any] = {
        "batch_size": config.batch_size,
        "n_replicates": config.n_replicates,
        "hidden_size": config.hidden_size,
        "controller_lr": config.controller_lr,
        "lr_warmup_batches": config.lr_warmup_batches,
        "lr_warmup_init_fraction": config.lr_warmup_init_fraction,
        "lr_cosine_alpha": config.lr_cosine_alpha,
        "gradient_clip_norm": config.gradient_clip_norm,
        "plant_backend": "cs_lss",
        "stochastic_preset": config.stochastic_preset,
        "loss_objective": config.loss_objective,
        "regularized_fidelity": config.regularized_fidelity,
        "target_relative_multitarget": config.target_relative_multitarget,
        "force_filter_feedback": config.force_filter_feedback,
        "perturbation_training": config.perturbation_training,
        "perturbation_calibrated_timing": config.perturbation_calibrated_timing,
        "perturbation_physical_level": config.perturbation_physical_level,
        "broad_epsilon_training": config.broad_epsilon_training,
        "broad_epsilon_level": config.broad_epsilon_level,
        "broad_epsilon_budget_scale": config.broad_epsilon_budget_scale,
        "broad_epsilon_reach_scaling": config.broad_epsilon_reach_scaling,
        "broad_epsilon_pgd_training": config.broad_epsilon_pgd_training,
        "broad_epsilon_pgd_level": config.broad_epsilon_pgd_level,
        "broad_epsilon_pgd_budget_scale": config.broad_epsilon_pgd_budget_scale,
        "broad_epsilon_pgd_steps": config.broad_epsilon_pgd_steps,
        "broad_epsilon_pgd_step_size_fraction": config.broad_epsilon_pgd_step_size_fraction,
        "initial_hidden_encoder": config.initial_hidden_encoder,
        "training_diagnostics": config.training_diagnostics,
        "schedule_total_batches": config.schedule_total_batches,
    }
    if config.broad_epsilon_pgd_seed is not None:
        payload["broad_epsilon_pgd_seed"] = config.broad_epsilon_pgd_seed
    if config.extra_args:
        payload["argv"] = list(config.extra_args)
    return payload


def build_remote_smoke_command() -> list[str]:
    """Return a tiny command that exits immediately inside a Modal container."""

    code = (
        "import json, os, platform, shutil, subprocess, sys; "
        "payload={'python': sys.version, 'platform': platform.platform(), "
        "'modal_env': {k: v for k, v in os.environ.items() if k.startswith('MODAL_')}, "
        "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}; "
        "cmd=['nvidia-smi','--query-gpu=name,driver_version,memory.total',"
        "'--format=csv,noheader']; "
        "payload['nvidia_smi_available']=shutil.which('nvidia-smi') is not None; "
        "payload['nvidia_smi']=''; "
        "payload['nvidia_smi_error']=''; "
        "\ntry:\n"
        "    payload['nvidia_smi']=subprocess.run(cmd, capture_output=True, text=True, "
        "timeout=10, check=False).stdout.strip()\n"
        "except Exception as exc:\n"
        "    payload['nvidia_smi_error']=str(exc)\n"
        "print(json.dumps(payload, indent=2, sort_keys=True))"
    )
    return ["python", "-c", code]


def build_packing_benchmark_command(
    config: NominalGruRunConfig,
    *,
    remote: bool = False,
) -> list[str]:
    """Build the multi-process packing benchmark command."""

    artifact_dir = config.remote_artifact_dir() if remote else config.local_artifact_dir()
    scenario_config = cs_nominal_gru_scenario_config(config)
    command = [
        "uv",
        "run",
        "--no-sync" if remote else "",
        "python",
        "-m",
        "rlrmp.packing_benchmark",
        "parent",
    ]
    command = [part for part in command if part]
    for flag, value in [
        ("--output-dir", artifact_dir),
        ("--n-workers", config.n_workers),
        ("--stagger-seconds", config.stagger_seconds),
        ("--ready-timeout-seconds", config.ready_timeout_seconds),
        ("--burn-in-seconds", config.burn_in_seconds),
        ("--measure-seconds", config.measure_seconds),
        ("--warmup-batches", config.warmup_batches),
        ("--chunk-batches", config.chunk_batches),
        ("--scenario", "cs-nominal-gru"),
        ("--scenario-config-json", json.dumps(scenario_config, sort_keys=True)),
        ("--seed", config.seed),
        ("--sample-seconds", config.sample_seconds),
    ]:
        _append_arg(command, flag, value)
    if config.packing_jax_platform is not None:
        _append_arg(command, "--jax-platform", config.packing_jax_platform)
        _append_arg(command, "--cpu-threads-per-worker", config.packing_cpu_threads_per_worker)
    if config.packing_jax_compilation_cache_dir is not None:
        _append_arg(
            command,
            "--jax-compilation-cache-dir",
            config.packing_jax_compilation_cache_dir,
        )
    if config.packing_jax_persistent_cache_min_compile_time_secs is not None:
        _append_arg(
            command,
            "--jax-persistent-cache-min-compile-time-secs",
            config.packing_jax_persistent_cache_min_compile_time_secs,
        )
    if config.packing_jax_persistent_cache_min_entry_size_bytes is not None:
        _append_arg(
            command,
            "--jax-persistent-cache-min-entry-size-bytes",
            config.packing_jax_persistent_cache_min_entry_size_bytes,
        )
    return command


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def dry_run_payload(config: NominalGruRunConfig) -> dict[str, Any]:
    """Return the local and remote commands without executing cloud work."""

    return {
        "app_name": APP_NAME,
        "modal_volume_name": MODAL_VOLUME_NAME,
        "modal_volume_mount": str(MODAL_VOLUME_MOUNT),
        "mode": config.mode,
        "gpu": config.gpu,
        "timeout_seconds": config.timeout_seconds,
        "stochastic_preset": config.stochastic_preset,
        "loss_objective": config.loss_objective,
        "controller_lr": config.controller_lr,
        "gradient_clip_norm": config.gradient_clip_norm,
        "regularized_fidelity": config.regularized_fidelity,
        "warm_containers": 0,
        "min_containers": 0,
        "max_containers": 1,
        "local_training_command": build_training_command(config, remote=False),
        "local_training_shell": shell_join(build_training_command(config, remote=False)),
        "remote_training_command": build_training_command(config, remote=True),
        "remote_training_shell": shell_join(build_training_command(config, remote=True)),
        "remote_packing_benchmark_command": build_packing_benchmark_command(config, remote=True),
        "remote_packing_benchmark_shell": shell_join(
            build_packing_benchmark_command(config, remote=True)
        ),
        "remote_smoke_command": build_remote_smoke_command(),
        "remote_smoke_shell": shell_join(build_remote_smoke_command()),
        "local_spec_dir": str(config.local_spec_dir()),
        "local_artifact_dir": str(config.local_artifact_dir()),
        "remote_spec_dir": str(config.remote_spec_dir()),
        "remote_artifact_dir": str(config.remote_artifact_dir()),
        "modal_volume_pull_commands": modal_volume_pull_commands(config),
        "modal_volume_sync_command": modal_volume_sync_command(config),
        "planned_stochastic_runs": planned_stochastic_runs(config),
    }


def planned_stochastic_runs(config: NominalGruRunConfig) -> dict[str, dict[str, Any]]:
    """Return the two planned stochastic C&S GRU run commands."""

    base = {
        **config.__dict__,
        "run": DEFAULT_RUN,
        "regularized_fidelity": False,
        "extra_args": config.extra_args,
    }
    regularized = {
        **config.__dict__,
        "run": REGULARIZED_RUN,
        "regularized_fidelity": True,
        "extra_args": config.extra_args,
    }
    no_hidden_config = NominalGruRunConfig(**base)
    hidden_config = NominalGruRunConfig(**regularized)
    return {
        "stochastic_no_hidden_penalty": {
            "run": DEFAULT_RUN,
            "nn_hidden": 0.0,
            "local_training_command": build_training_command(no_hidden_config, remote=False),
            "remote_training_command": build_training_command(no_hidden_config, remote=True),
            "modal_volume_pull_commands": modal_volume_pull_commands(no_hidden_config),
            "modal_volume_sync_command": modal_volume_sync_command(no_hidden_config),
        },
        "stochastic_hidden_penalty": {
            "run": REGULARIZED_RUN,
            "nn_hidden": 1e-5,
            "local_training_command": build_training_command(hidden_config, remote=False),
            "remote_training_command": build_training_command(hidden_config, remote=True),
            "modal_volume_pull_commands": modal_volume_pull_commands(hidden_config),
            "modal_volume_sync_command": modal_volume_sync_command(hidden_config),
        },
    }


def modal_volume_pull_commands(config: NominalGruRunConfig) -> dict[str, list[str]]:
    """Return commands for pulling persisted Modal Volume outputs locally."""

    return {
        "artifacts": [
            "modal",
            "volume",
            "get",
            MODAL_VOLUME_NAME,
            _modal_volume_relative(config.remote_artifact_dir()),
            str(config.local_artifact_dir()),
        ],
        "specs": [
            "modal",
            "volume",
            "get",
            MODAL_VOLUME_NAME,
            _modal_volume_relative(config.remote_spec_dir()),
            str(config.local_spec_dir()),
        ],
    }


def modal_volume_sync_command(config: NominalGruRunConfig) -> list[str]:
    """Return the one-action local command for pulling and validating a run."""

    return [
        "uv",
        "run",
        "python",
        "scripts/sync_modal_run_artifacts.py",
        "--issue",
        config.experiment,
        "--run",
        config.run,
    ]


def _modal_volume_relative(path: Path) -> str:
    return str(path.relative_to(MODAL_VOLUME_MOUNT))


def activate_project_venv(venv_dir: Path = REMOTE_VENV_DIR) -> Path:
    """Expose a uv-created project venv to the current Python process.

    Modal invokes the function body with its system Python, while the image
    build installs rlrmp dependencies into ``/workspace/rlrmp/.venv``. The
    in-process trainer path therefore needs the venv's site-packages on
    ``sys.path`` before importing Equinox/Feedbax/JAX-dependent modules.
    """

    lib_dir = venv_dir / "lib"
    site_packages = sorted(lib_dir.glob("python*/site-packages"))
    if not site_packages:
        raise FileNotFoundError(f"No site-packages directory found under {lib_dir}")

    before = list(sys.path)
    site_path = str(site_packages[-1])
    site.addsitedir(site_path)
    activated_paths = [
        path for path in sys.path if path == site_path or (path not in before and path)
    ]
    sys.path[:] = [
        *activated_paths,
        *(path for path in sys.path if path not in set(activated_paths)),
    ]
    _evict_modal_bundled_modules(("typing_extensions",))

    bin_path = str(venv_dir / "bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_path not in path_parts:
        os.environ["PATH"] = os.pathsep.join([bin_path, *path_parts])
    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    return site_packages[-1]


def _evict_modal_bundled_modules(module_names: Sequence[str]) -> None:
    for module_name in module_names:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        parts = Path(module_file).parts
        if "__modal" in parts and "deps" in parts:
            sys.modules.pop(module_name, None)


def collect_provenance() -> dict[str, Any]:
    """Collect best-effort local or Modal environment provenance."""

    provenance: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "modal_env": {key: value for key, value in os.environ.items() if key.startswith("MODAL_")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for package in ("modal", "rlrmp", "feedbax", "jax_cookbook", "jax"):
        try:
            module = __import__(package)
            provenance[f"{package}_version"] = getattr(module, "__version__", "unknown")
            if package == "jax":
                provenance["jax_devices"] = [str(device) for device in module.devices()]
                provenance["jax_default_backend"] = module.default_backend()
        except Exception as exc:
            provenance[f"{package}_error"] = str(exc)

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        provenance["rlrmp_commit"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception as exc:
        provenance["rlrmp_commit_error"] = str(exc)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        provenance["nvidia_smi"] = result.stdout.strip()
        provenance["nvidia_smi_returncode"] = result.returncode
        provenance["nvidia_smi_stderr"] = result.stderr.strip()
    except Exception as exc:
        provenance["nvidia_smi_error"] = str(exc)
    return provenance


def collect_source_provenance(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Collect local source provenance before Modal source dirs are copied."""

    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }
    provenance: dict[str, Any] = {}
    for key, command in commands.items():
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            provenance[key] = result.stdout.strip() if result.returncode == 0 else None
            if result.returncode != 0:
                provenance[f"{key}_stderr"] = result.stderr.strip()
        except Exception as exc:
            provenance[f"{key}_error"] = str(exc)
    return provenance


def write_provenance(
    config: NominalGruRunConfig,
    *,
    remote: bool = False,
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write Modal/environment provenance next to the run spec and artifacts."""

    payload = collect_provenance()
    if source_provenance is not None:
        payload["source_provenance"] = source_provenance
    spec_dir = config.remote_spec_dir() if remote else config.local_spec_dir()
    artifact_dir = config.remote_artifact_dir() if remote else config.local_artifact_dir()
    spec_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "spec_provenance": str(spec_dir / "modal_environment.json"),
        "artifact_provenance": str(artifact_dir / "modal_environment.json"),
    }
    for path in paths.values():
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return paths


def run_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> int:
    """Run a bounded subprocess and return its exit code."""

    print(shell_join(command), flush=True)
    result = subprocess.run(list(command), timeout=timeout_seconds, cwd=cwd, check=False)
    return result.returncode


def patch_remote_editable_paths() -> None:
    """Patch exact local editable paths for source-mode execution on Modal."""

    replacements = {
        str(LOCAL_FEEDBAX_DIR): str(REMOTE_FEEDBAX_DIR),
        "../../../20 Feedbax/feedbax": str(REMOTE_FEEDBAX_DIR),
        str(LOCAL_JAX_COOKBOOK_DIR): str(REMOTE_JAX_COOKBOOK_DIR),
        "../../../../05 Utils/jax-cookbook": str(REMOTE_JAX_COOKBOOK_DIR),
        "../../../../../05 Utils/jax-cookbook": str(REMOTE_JAX_COOKBOOK_DIR),
    }
    files = [
        REMOTE_REPO_DIR / "pyproject.toml",
        REMOTE_REPO_DIR / "uv.lock",
        REMOTE_FEEDBAX_DIR / "pyproject.toml",
        REMOTE_FEEDBAX_DIR / "uv.lock",
    ]
    for file_path in files:
        if not file_path.exists():
            continue
        text = file_path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        file_path.write_text(text)


def execute_remote_payload(
    payload: dict[str, Any],
    *,
    volume_commit: Callable[[], None] | None = None,
) -> int:
    """Execute a Modal payload inside the remote container."""

    config = NominalGruRunConfig(**payload["config"])
    command_kind: CommandKind = payload["command_kind"]
    if config.mode == "source":
        patch_remote_editable_paths()
    write_provenance(
        config,
        remote=True,
        source_provenance=payload.get("source_provenance"),
    )
    if command_kind == "modal-smoke":
        return run_subprocess(build_remote_smoke_command(), timeout_seconds=config.timeout_seconds)
    if command_kind == "modal-run":
        from rlrmp.train.cs_nominal_gru import main as train_main

        return train_main(
            build_training_script_args(config, remote=True),
            volume_commit=volume_commit,
        )
    if command_kind == "modal-packing-smoke":
        return run_subprocess(
            build_packing_benchmark_command(config, remote=True),
            timeout_seconds=config.timeout_seconds,
            cwd=config.remote_repo_dir(),
        )
    raise ValueError(f"Remote execution does not support {command_kind!r}")


def make_config(args: argparse.Namespace) -> NominalGruRunConfig:
    timeout_seconds = int(args.timeout_seconds)
    if args.command == "modal-run" and timeout_seconds == DEFAULT_TIMEOUT_SECONDS:
        timeout_seconds = DEFAULT_TRAIN_TIMEOUT_SECONDS
    return NominalGruRunConfig(
        experiment=args.experiment,
        run=args.run,
        n_train_batches=args.n_train_batches,
        batch_size=args.batch_size,
        n_replicates=args.n_replicates,
        hidden_size=args.hidden_size,
        seed=args.seed,
        controller_lr=args.controller_lr,
        lr_warmup_batches=args.lr_warmup_batches,
        lr_warmup_init_fraction=args.lr_warmup_init_fraction,
        lr_cosine_alpha=args.lr_cosine_alpha,
        gradient_clip_norm=args.gradient_clip_norm,
        stochastic_preset=args.stochastic_preset,
        loss_objective=args.loss_objective,
        regularized_fidelity=args.regularized_fidelity,
        checkpoint_interval_batches=args.checkpoint_interval_batches,
        resume=args.resume,
        timeout_seconds=timeout_seconds,
        gpu=args.gpu,
        mode=args.mode,
        pinned_repo_dir=args.pinned_repo_dir,
        extra_args=tuple(args.extra_arg or ()),
        n_workers=args.n_workers,
        stagger_seconds=args.stagger_seconds,
        burn_in_seconds=args.burn_in_seconds,
        measure_seconds=args.measure_seconds,
        warmup_batches=args.warmup_batches,
        chunk_batches=args.chunk_batches,
        ready_timeout_seconds=args.ready_timeout_seconds,
        sample_seconds=args.sample_seconds,
        packing_jax_platform=args.packing_jax_platform,
        packing_cpu_threads_per_worker=args.packing_cpu_threads_per_worker,
        packing_jax_compilation_cache_dir=args.packing_jax_compilation_cache_dir,
        packing_jax_persistent_cache_min_compile_time_secs=(
            args.packing_jax_persistent_cache_min_compile_time_secs
        ),
        packing_jax_persistent_cache_min_entry_size_bytes=(
            args.packing_jax_persistent_cache_min_entry_size_bytes
        ),
        target_relative_multitarget=args.target_relative_multitarget,
        force_filter_feedback=args.force_filter_feedback,
        perturbation_training=args.perturbation_training,
        perturbation_calibrated_timing=args.perturbation_calibrated_timing,
        perturbation_physical_level=args.perturbation_physical_level,
        broad_epsilon_training=args.broad_epsilon_training,
        broad_epsilon_level=args.broad_epsilon_level,
        broad_epsilon_budget_scale=args.broad_epsilon_budget_scale,
        broad_epsilon_reach_scaling=args.broad_epsilon_reach_scaling,
        broad_epsilon_pgd_training=args.broad_epsilon_pgd_training,
        broad_epsilon_pgd_level=args.broad_epsilon_pgd_level,
        broad_epsilon_pgd_budget_scale=args.broad_epsilon_pgd_budget_scale,
        broad_epsilon_pgd_steps=args.broad_epsilon_pgd_steps,
        broad_epsilon_pgd_step_size_fraction=args.broad_epsilon_pgd_step_size_fraction,
        broad_epsilon_pgd_seed=args.broad_epsilon_pgd_seed,
        initial_hidden_encoder=args.initial_hidden_encoder,
        training_diagnostics=args.training_diagnostics,
        schedule_total_batches=args.schedule_total_batches,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or launch the stochastic C&S-fidelity GRU Modal runner."
    )
    parser.add_argument(
        "command",
        choices=["dry-run", "local-smoke", "modal-smoke", "modal-run", "modal-packing-smoke"],
        nargs="?",
        default="dry-run",
        help="dry-run prints commands; local-smoke runs only an immediate local smoke.",
    )
    parser.add_argument("--mode", choices=["source", "pinned"], default="source")
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--n-train-batches", type=int, default=DEFAULT_N_TRAIN_BATCHES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-lr", type=float, default=1e-2)
    parser.add_argument("--lr-warmup-batches", type=int, default=0)
    parser.add_argument("--lr-warmup-init-fraction", type=float, default=0.1)
    parser.add_argument("--lr-cosine-alpha", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--stochastic-preset", default=DEFAULT_STOCHASTIC_PRESET)
    parser.add_argument(
        "--loss-objective",
        choices=CS_LOSS_OBJECTIVES,
        default=CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
        help="Training objective passed through to scripts/train_cs_nominal_gru.py.",
    )
    parser.add_argument(
        "--regularized-fidelity",
        action="store_true",
        help="Run the paired stochastic condition with nn_hidden=1e-5.",
    )
    parser.add_argument(
        "--checkpoint-interval-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_BATCHES,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--stagger-seconds", type=float, default=10.0)
    parser.add_argument("--burn-in-seconds", type=float, default=60.0)
    parser.add_argument("--measure-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--chunk-batches", type=int, default=4)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument(
        "--packing-jax-platform",
        choices=["cpu", "gpu", "tpu"],
        default=None,
        help=(
            "Optional JAX backend override passed to the packing benchmark. Omit for "
            "provider defaults such as RunPod/Modal GPU; use cpu for local CPU packing."
        ),
    )
    parser.add_argument(
        "--packing-cpu-threads-per-worker",
        type=int,
        default=1,
        help=(
            "Common per-worker CPU thread cap passed to the packing benchmark when "
            "--packing-jax-platform=cpu."
        ),
    )
    parser.add_argument(
        "--packing-jax-compilation-cache-dir",
        default=None,
        help=(
            "Optional persistent JAX compilation-cache directory for packing workers. "
            "This affects startup compile/warmup metadata, not steady-state metrics."
        ),
    )
    parser.add_argument(
        "--packing-jax-persistent-cache-min-compile-time-secs",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--packing-jax-persistent-cache-min-entry-size-bytes",
        type=int,
        default=None,
    )
    parser.add_argument("--target-relative-multitarget", action="store_true")
    parser.add_argument("--force-filter-feedback", "--proprioceptive-feedback", action="store_true")
    parser.add_argument("--perturbation-training", action="store_true")
    parser.add_argument("--perturbation-calibrated-timing", action="store_true")
    parser.add_argument("--perturbation-physical-level", default="moderate")
    parser.add_argument("--broad-epsilon-training", action="store_true")
    parser.add_argument("--broad-epsilon-level", default="moderate")
    parser.add_argument("--broad-epsilon-budget-scale", type=float, default=1.0)
    parser.add_argument(
        "--broad-epsilon-reach-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--broad-epsilon-pgd-training", action="store_true")
    parser.add_argument("--broad-epsilon-pgd-level", default="moderate")
    parser.add_argument("--broad-epsilon-pgd-budget-scale", type=float, default=1.0)
    parser.add_argument("--broad-epsilon-pgd-steps", type=int, default=3)
    parser.add_argument("--broad-epsilon-pgd-step-size-fraction", type=float, default=0.25)
    parser.add_argument("--broad-epsilon-pgd-seed", type=int, default=None)
    parser.add_argument("--initial-hidden-encoder", "--h0-encoder", action="store_true")
    parser.add_argument(
        "--training-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--schedule-total-batches", type=int, default=1000)
    parser.add_argument("--pinned-repo-dir", default=str(REMOTE_REPO_DIR))
    parser.add_argument(
        "--extra-arg",
        action="append",
        help=(
            "Additional argument token appended to scripts/train_cs_nominal_gru.py; "
            "repeat per token."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = make_config(args)
    if args.command == "dry-run":
        print(json.dumps(dry_run_payload(config), indent=2, sort_keys=True))
        return 0
    if args.command == "local-smoke":
        print(json.dumps(collect_provenance(), indent=2, sort_keys=True))
        return 0
    if args.command in {"modal-smoke", "modal-run"}:
        raise SystemExit(
            "Use `modal run scripts/modal_cs_nominal_gru.py -- "
            f"{args.command} ...` for cloud execution."
        )
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

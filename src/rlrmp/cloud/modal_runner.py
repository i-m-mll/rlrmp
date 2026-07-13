"""Modal launch helpers for stochastic C&S-fidelity GRU preparation.

This module keeps cloud execution behind explicit CLI choices. The default
path is a local dry-run that prints the command and provenance paths without
starting a Modal container.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from feedbax.execution.container import (
    collect_environment_provenance,
    run_subprocess,
)
from feedbax.execution.models import (
    ExecutionSpec,
    ModalBackendConfig,
    RepoSource,
)

from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir
from rlrmp.runtime.parameter_presets import ModalRunnerPreset, load_runtime_preset

APP_NAME = "rlrmp-cs-stochastic-gru"
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
_RUNTIME_PRESET = load_runtime_preset("rlrmp.modal_runner.default", ModalRunnerPreset)
DEFAULT_TIMEOUT_SECONDS = _RUNTIME_PRESET.timeout_seconds
DEFAULT_TRAIN_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_N_TRAIN_BATCHES = _RUNTIME_PRESET.n_train_batches
DEFAULT_BATCH_SIZE = _RUNTIME_PRESET.batch_size
DEFAULT_N_REPLICATES = _RUNTIME_PRESET.n_replicates
DEFAULT_HIDDEN_SIZE = _RUNTIME_PRESET.hidden_size
MODAL_VOLUME_NAME = "rlrmp-cs-stochastic-gru"
MODAL_VOLUME_MOUNT = Path("/vol/rlrmp-cs-stochastic-gru")
REMOTE_REPO_DIR = Path("/workspace/rlrmp")
REMOTE_FEEDBAX_DIR = Path("/workspace/feedbax")
REMOTE_JAX_COOKBOOK_DIR = Path("/workspace/jax-cookbook")
REMOTE_VENV_DIR = REMOTE_REPO_DIR / ".venv"
LOCAL_FEEDBAX_DIR = Path("/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax")
LOCAL_JAX_COOKBOOK_DIR = Path("/Users/mll/Main/10 Projects/05 Utils/jax-cookbook")
LOCAL_EMBED_IGNORE_PARTS = [
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "_artifacts",
    "worktrees",
    "manuscript.assets",
    "TODO.assets",
]
LOCAL_EMBED_IGNORE_SUFFIXES = [".assets"]

Mode = Literal["source", "pinned"]
CommandKind = Literal["dry-run", "local-smoke", "modal-smoke", "modal-run", "modal-packing-smoke"]


@dataclass(frozen=True)
class NominalGruRunConfig:
    """Command-level configuration for the stochastic GRU run."""

    experiment: str | None = None
    run: str = DEFAULT_RUN
    authored_document: str | None = None
    row: str | None = None
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
    resume: bool = True
    allow_fresh_start: bool = False
    stop_after_batches: int | None = None
    disable_progress: bool = False
    quiet_progress: bool = False
    log_step: int = 1
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    gpu: str = DEFAULT_GPU
    mode: Mode = "source"
    pinned_repo_dir: str = str(REMOTE_REPO_DIR)
    # Packing scenarios still accept explicit library-level argv in tests and
    # programmatic callers. The Modal CLI deliberately exposes no equivalent.
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
    confirm_billable_launch: bool = False

    def resolved_experiment(self) -> str:
        """Return the explicit issue key or derive it from the authored matrix path."""

        if self.experiment:
            return self.experiment
        if self.authored_document:
            parts = Path(self.authored_document).parts
            try:
                results_index = parts.index("results")
            except ValueError:
                pass
            else:
                if len(parts) > results_index + 2 and parts[results_index + 1]:
                    return parts[results_index + 1]
        raise ValueError(
            "experiment is required unless --document has the form results/<issue>/runs/<file>"
        )

    def local_artifact_dir(self) -> Path:
        return run_artifact_dir(self.resolved_experiment(), self.run)

    def local_spec_dir(self) -> Path:
        return run_spec_dir(self.resolved_experiment(), self.run)

    def remote_artifact_dir(self) -> Path:
        return (
            MODAL_VOLUME_MOUNT
            / "_artifacts"
            / self.resolved_experiment()
            / "runs"
            / self.run
        )

    def remote_spec_dir(self) -> Path:
        return MODAL_VOLUME_MOUNT / "results" / self.resolved_experiment() / "runs" / self.run

    def remote_repo_dir(self) -> Path:
        if self.mode == "pinned":
            return Path(self.pinned_repo_dir)
        return REMOTE_REPO_DIR


def _append_arg(command: list[str], flag: str, value: str | int | float) -> None:
    command.extend([flag, str(value)])


def _execution_sources(config: NominalGruRunConfig) -> list[RepoSource]:
    if config.mode == "pinned":
        return [
            RepoSource(
                name="rlrmp",
                role="project",
                install_mode="github-ref",
                package="rlrmp",
                git_ref="HEAD",
                target_path=config.pinned_repo_dir,
                metadata={"source_mode": "pinned"},
            )
        ]
    return [
        RepoSource(
            name="rlrmp",
            role="project",
            install_mode="local-embed",
            package="rlrmp",
            local_path=str(REPO_ROOT),
            target_path=str(REMOTE_REPO_DIR),
            ignore_parts=LOCAL_EMBED_IGNORE_PARTS,
            ignore_suffixes=LOCAL_EMBED_IGNORE_SUFFIXES,
            extra_path_rewrites={
                "../20 Feedbax/feedbax": str(REMOTE_FEEDBAX_DIR),
                "../../../20 Feedbax/feedbax": str(REMOTE_FEEDBAX_DIR),
            },
        ),
        RepoSource(
            name="feedbax",
            role="dependency",
            install_mode="local-embed",
            package="feedbax",
            local_path=str(LOCAL_FEEDBAX_DIR),
            target_path=str(REMOTE_FEEDBAX_DIR),
            ignore_parts=LOCAL_EMBED_IGNORE_PARTS,
            ignore_suffixes=LOCAL_EMBED_IGNORE_SUFFIXES,
            extra_path_rewrites={
                "../../../20 Feedbax/feedbax": str(REMOTE_FEEDBAX_DIR),
            },
        ),
        RepoSource(
            name="jax-cookbook",
            role="dependency",
            install_mode="local-embed",
            package="jax-cookbook",
            local_path=str(LOCAL_JAX_COOKBOOK_DIR),
            target_path=str(REMOTE_JAX_COOKBOOK_DIR),
            ignore_parts=LOCAL_EMBED_IGNORE_PARTS,
            ignore_suffixes=LOCAL_EMBED_IGNORE_SUFFIXES,
            extra_path_rewrites={
                "../../../../05 Utils/jax-cookbook": str(REMOTE_JAX_COOKBOOK_DIR),
                "../../../../../05 Utils/jax-cookbook": str(REMOTE_JAX_COOKBOOK_DIR),
            },
        ),
    ]


def build_modal_image_execution_spec() -> ExecutionSpec:
    """Describe only the sources and environment needed to render the Modal image."""

    config = NominalGruRunConfig()
    return ExecutionSpec(
        kind="custom",
        job_id="rlrmp-authored-launch-image",
        backend="modal",
        command="true",
        repos=_execution_sources(config),
        primary_repo="rlrmp",
        env={
            "PYTHONPATH": (
                f"{REMOTE_REPO_DIR}/src:{REMOTE_FEEDBAX_DIR}:{REMOTE_JAX_COOKBOOK_DIR}"
            ),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        modal=ModalBackendConfig(
            app_name=APP_NAME,
            gpu=config.gpu,
            volume_name=MODAL_VOLUME_NAME,
            volume_mount_path=str(MODAL_VOLUME_MOUNT),
            timeout_seconds=config.timeout_seconds,
            max_containers=1,
            image_packages=[],
            extra_install_commands=['uv pip install -U "jax[cuda12]"'],
        ),
        metadata={"launcher": "rlrmp.cloud.modal_runner", "purpose": "image-only"},
    )


def build_training_command(
    config: NominalGruRunConfig,
    *,
    remote: bool = False,
) -> list[str]:
    """Build the authored-matrix launch command for local or Modal execution."""

    if config.authored_document is None:
        raise ValueError("training launch requires --document TrainingRunMatrixSpec.json")
    document = _document_path_for_backend(config.authored_document, remote=remote)
    command = [
        "uv",
        "run",
        *(["--no-sync"] if remote else []),
        "python",
        "scripts/launch_training.py",
        "execute",
        str(document),
    ]
    if config.row is not None:
        command.extend(["--row", config.row])
    if config.resume:
        command.append("--resume")
    if config.allow_fresh_start:
        command.append("--allow-fresh-start")
    if config.stop_after_batches is not None:
        _append_arg(command, "--stop-after-batches", config.stop_after_batches)
    if config.disable_progress:
        command.append("--disable-progress")
    if config.quiet_progress:
        command.append("--quiet-progress")
    _append_arg(command, "--log-step", config.log_step)
    return command


def _document_path_for_backend(document: str, *, remote: bool) -> Path:
    """Map a checked-in authored document to the embedded repository on Modal."""

    path = Path(document)
    if not remote:
        return path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Modal authored documents must live inside the rlrmp repository so the "
            "source bundle ships the exact reviewed bytes"
        ) from exc
    return REMOTE_REPO_DIR / relative


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
        "rlrmp.benchmarks.packing",
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

    local_command = build_training_command(config)
    remote_command = build_training_command(config, remote=True)
    return {
        "app_name": APP_NAME,
        "modal_volume_name": MODAL_VOLUME_NAME,
        "modal_volume_mount": str(MODAL_VOLUME_MOUNT),
        "mode": config.mode,
        "gpu": config.gpu,
        "timeout_seconds": config.timeout_seconds,
        "authored_document": config.authored_document,
        "row": config.row,
        "warm_containers": 0,
        "min_containers": 0,
        "max_containers": 1,
        "local_training_command": local_command,
        "local_training_shell": shell_join(local_command),
        "remote_training_command": remote_command,
        "remote_training_shell": shell_join(remote_command),
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
        config.resolved_experiment(),
        "--run",
        config.run,
    ]


def _modal_volume_relative(path: Path) -> str:
    return str(path.relative_to(MODAL_VOLUME_MOUNT))


def write_provenance(
    config: NominalGruRunConfig,
    *,
    remote: bool = False,
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write Modal/environment provenance next to the run spec and artifacts."""

    payload = collect_environment_provenance(
        packages=("modal", "rlrmp", "feedbax", "jax_cookbook", "jax"),
        repo_root=REPO_ROOT,
    )
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


def execute_remote_payload(
    payload: dict[str, Any],
    *,
    volume_commit: Callable[[], None] | None = None,
) -> int:
    """Execute a Modal payload inside the remote container."""

    config = NominalGruRunConfig(**payload["config"])
    command_kind: CommandKind = payload["command_kind"]
    write_provenance(
        config,
        remote=True,
        source_provenance=payload.get("source_provenance"),
    )
    if command_kind == "modal-smoke":
        return run_subprocess(build_remote_smoke_command(), timeout_seconds=config.timeout_seconds)
    if command_kind == "modal-run":
        return_code = run_subprocess(
            build_training_command(config, remote=True),
            timeout_seconds=config.timeout_seconds,
            cwd=config.remote_repo_dir(),
        )
        if volume_commit is not None:
            volume_commit()
        return return_code
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
        authored_document=args.document,
        row=args.row,
        n_train_batches=DEFAULT_N_TRAIN_BATCHES,
        batch_size=DEFAULT_BATCH_SIZE,
        n_replicates=DEFAULT_N_REPLICATES,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        resume=args.resume,
        allow_fresh_start=args.allow_fresh_start,
        stop_after_batches=args.stop_after_batches,
        disable_progress=args.disable_progress,
        quiet_progress=args.quiet_progress,
        log_step=args.log_step,
        timeout_seconds=timeout_seconds,
        gpu=args.gpu,
        mode=args.mode,
        pinned_repo_dir=args.pinned_repo_dir,
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
        confirm_billable_launch=args.confirm_billable_launch,
    )


def require_billable_launch_confirmation(args: argparse.Namespace) -> None:
    """Refuse billable cloud training launches without explicit confirmation."""

    if args.command == "modal-run" and not args.confirm_billable_launch:
        raise SystemExit(
            "Refusing billable Modal training launch without --confirm-billable-launch. "
            "Dry-run/spec-lock review is not launch authorization."
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
    parser.add_argument(
        "--confirm-billable-launch",
        action="store_true",
        help=(
            "Required for non-smoke Modal training launches after the run spec-lock "
            "has been explicitly approved."
        ),
    )
    parser.add_argument(
        "--experiment",
        help="Issue key for packing/provenance paths; inferred from --document when possible.",
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument(
        "--document",
        help="Authored TrainingRunMatrixSpec JSON shipped unchanged to the Modal pod.",
    )
    parser.add_argument("--row", help="Authored matrix row selector.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-fresh-start", action="store_true")
    parser.add_argument("--stop-after-batches", type=int)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--log-step", type=int, default=1)
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
    parser.add_argument("--pinned-repo-dir", default=str(REMOTE_REPO_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_billable_launch_confirmation(args)
    config = make_config(args)
    if args.command == "dry-run":
        print(json.dumps(dry_run_payload(config), indent=2, sort_keys=True))
        return 0
    if args.command == "local-smoke":
        print(
            json.dumps(
                collect_environment_provenance(
                    packages=("modal", "rlrmp", "feedbax", "jax_cookbook", "jax"),
                    repo_root=REPO_ROOT,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"modal-smoke", "modal-run"}:
        raise SystemExit(
            "Use `modal run scripts/modal_cs_nominal_gru.py -- "
            f"{args.command} ...` for cloud execution."
        )
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

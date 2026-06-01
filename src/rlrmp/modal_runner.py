"""Modal launch helpers for nominal C&S-fidelity GRU preparation.

This module keeps cloud execution behind explicit CLI choices. The default
path is a local dry-run that prints the command and provenance paths without
starting a Modal container.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from rlrmp.paths import REPO_ROOT, run_artifact_dir, run_spec_dir

APP_NAME = "rlrmp-cs-nominal-gru"
DEFAULT_EXPERIMENT = "a1a8e39"
DEFAULT_RUN = "nominal_cs_gru__modal_prep"
DEFAULT_GPU = "A10G"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_TRAIN_TIMEOUT_SECONDS = 24 * 60 * 60
REMOTE_REPO_DIR = Path("/workspace/rlrmp")
REMOTE_FEEDBAX_DIR = Path("/workspace/feedbax")
REMOTE_JAX_COOKBOOK_DIR = Path("/workspace/jax-cookbook")
LOCAL_FEEDBAX_DIR = Path("/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax")
LOCAL_JAX_COOKBOOK_DIR = Path("/Users/mll/Main/10 Projects/05 Utils/jax-cookbook")

Mode = Literal["source", "pinned"]
CommandKind = Literal["dry-run", "local-smoke", "modal-smoke", "modal-run", "modal-packing-smoke"]


@dataclass(frozen=True)
class NominalGruRunConfig:
    """Command-level configuration for the nominal GRU run."""

    experiment: str = DEFAULT_EXPERIMENT
    run: str = DEFAULT_RUN
    n_train_batches: int = 1
    batch_size: int = 4
    n_replicates: int = 1
    hidden_size: int = 4
    seed: int = 42
    controller_lr: float = 1e-2
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

    def local_artifact_dir(self) -> Path:
        return run_artifact_dir(self.experiment, self.run)

    def local_spec_dir(self) -> Path:
        return run_spec_dir(self.experiment, self.run)

    def remote_artifact_dir(self) -> Path:
        return self.remote_repo_dir() / "_artifacts" / self.experiment / "runs" / self.run

    def remote_spec_dir(self) -> Path:
        return self.remote_repo_dir() / "results" / self.experiment / "runs" / self.run

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
    """Build the nominal C&S-fidelity GRU training command.

    The command intentionally targets the dedicated C&S nominal GRU spec writer,
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
    _append_arg(command, "--batch-size", config.batch_size)
    _append_arg(command, "--n-replicates", config.n_replicates)
    _append_arg(command, "--hidden-size", config.hidden_size)
    _append_arg(command, "--seed", config.seed)
    _append_arg(command, "--output-dir", artifact_dir)
    _append_arg(command, "--spec-dir", spec_dir)
    command.extend(config.extra_args)
    return command


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
    command = [
        "uv",
        "run",
        "--no-sync" if remote else "",
        "python",
        "-m",
        "rlrmp.modal_packing_benchmark",
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
        ("--batch-size", config.batch_size),
        ("--n-replicates", config.n_replicates),
        ("--hidden-size", config.hidden_size),
        ("--controller-lr", config.controller_lr),
        ("--seed", config.seed),
        ("--sample-seconds", config.sample_seconds),
    ]:
        _append_arg(command, flag, value)
    return command


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def dry_run_payload(config: NominalGruRunConfig) -> dict[str, Any]:
    """Return the local and remote commands without executing cloud work."""

    return {
        "app_name": APP_NAME,
        "mode": config.mode,
        "gpu": config.gpu,
        "timeout_seconds": config.timeout_seconds,
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
    }


def collect_provenance() -> dict[str, Any]:
    """Collect best-effort local or Modal environment provenance."""

    provenance: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "modal_env": {
            key: value for key, value in os.environ.items() if key.startswith("MODAL_")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for package in ("modal", "rlrmp", "feedbax", "jax_cookbook", "jax"):
        try:
            module = __import__(package)
            provenance[f"{package}_version"] = getattr(module, "__version__", "unknown")
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


def write_provenance(config: NominalGruRunConfig, *, remote: bool = False) -> dict[str, str]:
    """Write Modal/environment provenance next to the run spec and artifacts."""

    payload = collect_provenance()
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


def execute_remote_payload(payload: dict[str, Any]) -> int:
    """Execute a Modal payload inside the remote container."""

    config = NominalGruRunConfig(**payload["config"])
    command_kind: CommandKind = payload["command_kind"]
    if config.mode == "source":
        patch_remote_editable_paths()
    write_provenance(config, remote=True)
    if command_kind == "modal-smoke":
        return run_subprocess(build_remote_smoke_command(), timeout_seconds=config.timeout_seconds)
    if command_kind == "modal-run":
        return run_subprocess(
            build_training_command(config, remote=True),
            timeout_seconds=config.timeout_seconds,
            cwd=config.remote_repo_dir(),
        )
    if command_kind == "modal-packing-smoke":
        return run_subprocess(
            build_packing_benchmark_command(config, remote=True),
            timeout_seconds=config.timeout_seconds,
            cwd=config.remote_repo_dir(),
        )
    raise ValueError(f"Remote execution does not support {command_kind!r}")


def make_config(args: argparse.Namespace) -> NominalGruRunConfig:
    return NominalGruRunConfig(
        experiment=args.experiment,
        run=args.run,
        n_train_batches=args.n_train_batches,
        batch_size=args.batch_size,
        n_replicates=args.n_replicates,
        hidden_size=args.hidden_size,
        seed=args.seed,
        controller_lr=args.controller_lr,
        timeout_seconds=args.timeout_seconds,
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
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or launch the nominal C&S-fidelity GRU Modal runner."
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
    parser.add_argument("--n-train-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-replicates", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-lr", type=float, default=1e-2)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--stagger-seconds", type=float, default=10.0)
    parser.add_argument("--burn-in-seconds", type=float, default=60.0)
    parser.add_argument("--measure-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--chunk-batches", type=int, default=4)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
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

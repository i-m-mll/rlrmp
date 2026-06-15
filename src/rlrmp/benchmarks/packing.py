"""Scenario-driven multi-process packing benchmark harness.

The benchmark owns process packing mechanics only: spawning workers, disabling
XLA preallocation, ready-barrier synchronization, burn-in, measured windows, and
aggregation. Training/model details enter through a scenario payload.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from rlrmp.model.trainable import staged_network_trainable_parts


PREALLOC_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
JAX_COMPILATION_CACHE_DIR_ENV = "JAX_COMPILATION_CACHE_DIR"
JAX_CACHE_MIN_COMPILE_TIME_ENV = "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
JAX_CACHE_MIN_ENTRY_SIZE_ENV = "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"
JAX_PLATFORM_ENV = "JAX_PLATFORM_NAME"
JAX_PLATFORMS_ENV = "JAX_PLATFORMS"
XLA_FLAGS_ENV = "XLA_FLAGS"
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
DEFAULT_SCENARIO = "cs-nominal-gru"


@dataclass(frozen=True)
class WorkerConfig:
    worker_index: int
    output_dir: str
    start_file: str
    seed: int
    warmup_batches: int
    burn_in_seconds: float
    measure_seconds: float
    chunk_batches: int
    scenario: str
    scenario_config: dict[str, Any]


class ScenarioRuntime(Protocol):
    metadata: dict[str, Any]

    def warmup(self, n_batches: int) -> Any:
        """Compile/warm the scenario workload and return a scenario-owned model handle."""

    def train_chunk(self, model: Any, n_batches: int) -> Any:
        """Run one measured training chunk and return the updated model handle."""


ScenarioFactory = Callable[[Mapping[str, Any], int], ScenarioRuntime]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded scenario-driven multi-process packing benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parent = subparsers.add_parser("parent")
    parent.add_argument("--output-dir", required=True)
    parent.add_argument("--n-workers", type=int, required=True)
    parent.add_argument("--stagger-seconds", type=float, default=10.0)
    parent.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    parent.add_argument("--burn-in-seconds", type=float, default=60.0)
    parent.add_argument("--measure-seconds", type=float, default=60.0)
    parent.add_argument("--warmup-batches", type=int, default=2)
    parent.add_argument("--chunk-batches", type=int, default=4)
    parent.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parent.add_argument(
        "--scenario-config-json",
        default="{}",
        help=(
            "JSON object consumed by the selected scenario. The benchmark harness "
            "does not interpret model or training fields inside this object."
        ),
    )
    parent.add_argument("--seed", type=int, default=42)
    parent.add_argument("--sample-seconds", type=float, default=5.0)
    parent.add_argument(
        "--jax-platform",
        choices=["cpu", "gpu", "tpu"],
        default=None,
        help=(
            "Optional JAX backend override for benchmark workers. Use 'cpu' for local "
            "CPU packing; omit for provider defaults such as RunPod/Modal GPU."
        ),
    )
    parent.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        default=1,
        help=(
            "Common BLAS/OpenMP/TF thread cap per worker when --jax-platform=cpu. "
            "Use 0 to leave thread env vars unchanged."
        ),
    )
    parent.add_argument(
        "--xla-cpu-thread-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --jax-platform=cpu, add conservative XLA CPU threading flags unless "
            "the corresponding XLA_FLAGS entries are already present."
        ),
    )
    parent.add_argument(
        "--jax-compilation-cache-dir",
        default=None,
        help=(
            "Optional persistent JAX compilation cache directory. This mainly affects "
            "startup compile/warmup time; steady-state measured windows are reported "
            "separately."
        ),
    )
    parent.add_argument(
        "--jax-persistent-cache-min-compile-time-secs",
        type=float,
        default=None,
        help="Optional JAX persistent-cache minimum compile-time threshold.",
    )
    parent.add_argument(
        "--jax-persistent-cache-min-entry-size-bytes",
        type=int,
        default=None,
        help="Optional JAX persistent-cache minimum entry-size threshold.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument("--config-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "parent":
        return run_parent(args)
    if args.command == "worker":
        return run_worker(WorkerConfig(**json.loads(args.config_json)))
    raise ValueError(f"Unknown command {args.command!r}")


def run_parent(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_file = output_dir / "start.json"
    if start_file.exists():
        start_file.unlink()

    scenario_config = _load_scenario_config(args.scenario_config_json)
    parent_config = vars(args).copy()
    parent_config["scenario_config"] = scenario_config
    parent_config.pop("scenario_config_json", None)
    parent_config["captured_at"] = _utc_now()
    parent_config["preallocation"] = "disabled"
    parent_config["python"] = sys.version

    env = os.environ.copy()
    runtime_env = _configure_worker_env(env, args)
    parent_config["runtime_env"] = runtime_env
    (output_dir / "parent_config.json").write_text(_json(parent_config), encoding="utf-8")

    procs: list[subprocess.Popen[str]] = []
    handles = []
    for worker_index in range(args.n_workers):
        worker_dir = output_dir / f"worker_{worker_index:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_config = WorkerConfig(
            worker_index=worker_index,
            output_dir=str(worker_dir),
            start_file=str(start_file),
            seed=int(args.seed) + worker_index,
            warmup_batches=int(args.warmup_batches),
            burn_in_seconds=float(args.burn_in_seconds),
            measure_seconds=float(args.measure_seconds),
            chunk_batches=int(args.chunk_batches),
            scenario=str(args.scenario),
            scenario_config=scenario_config,
        )
        command = [
            sys.executable,
            "-m",
            "rlrmp.benchmarks.packing",
            "worker",
            "--config-json",
            json.dumps(asdict(worker_config), sort_keys=True),
        ]
        stdout = (worker_dir / "stdout.log").open("w", encoding="utf-8")
        stderr = (worker_dir / "stderr.log").open("w", encoding="utf-8")
        handles.extend([stdout, stderr])
        procs.append(
            subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        )
        time.sleep(max(0.0, float(args.stagger_seconds)))

    try:
        ready = _wait_for_ready(output_dir, args.n_workers, args.ready_timeout_seconds, procs)
        (output_dir / "ready_summary.json").write_text(_json(ready), encoding="utf-8")
        start_payload = {
            "released_at": _utc_now(),
            "n_workers": args.n_workers,
            "burn_in_seconds": args.burn_in_seconds,
            "measure_seconds": args.measure_seconds,
        }
        start_file.write_text(_json(start_payload), encoding="utf-8")
        gpu_samples = _sample_until_done(
            output_dir=output_dir,
            procs=procs,
            sample_seconds=float(args.sample_seconds),
        )
        worker_summaries = _collect_worker_summaries(output_dir, args.n_workers)
        summary = {
            "captured_at": _utc_now(),
            "n_workers": args.n_workers,
            "scenario": str(args.scenario),
            "scenario_config": scenario_config,
            "preallocation_env": env.get(PREALLOC_ENV),
            "runtime_env": runtime_env,
            "ready": ready,
            "workers": worker_summaries,
            "gpu_samples": gpu_samples,
            "aggregate": _aggregate(worker_summaries, gpu_samples),
        }
        (output_dir / "summary.json").write_text(_json(summary), encoding="utf-8")
        print("RLRMP_PACKING_SUMMARY_START", flush=True)
        print(_json(summary), flush=True)
        print("RLRMP_PACKING_SUMMARY_END", flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 30.0
        for proc in procs:
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                proc.kill()
        for handle in handles:
            handle.close()

    failed = [proc.returncode for proc in procs if proc.returncode not in (0, None)]
    return 1 if failed else 0


def run_worker(config: WorkerConfig) -> int:
    os.environ[PREALLOC_ENV] = "false"

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    summary_path = output_dir / "summary.json"
    _write_json(status_path, {"status": "starting", "captured_at": _utc_now()})

    scenario_factory = _resolve_scenario_factory(config.scenario)
    runtime = scenario_factory(config.scenario_config, config.seed)

    _write_json(status_path, {"status": "compiling", "captured_at": _utc_now()})
    compile_start = time.monotonic()
    model = runtime.warmup(max(1, config.warmup_batches, config.chunk_batches))
    compile_seconds = time.monotonic() - compile_start
    _write_json(
        status_path,
        {
            "status": "ready",
            "captured_at": _utc_now(),
            "compile_and_warmup_seconds": compile_seconds,
            "runtime_env": _runtime_env_metadata(os.environ),
            "timing_phases": _timing_phase_metadata(compile_seconds),
            "config": asdict(config),
            "scenario_metadata": runtime.metadata,
        },
    )

    start_file = Path(config.start_file)
    while not start_file.exists():
        time.sleep(0.2)

    burn = _timed_train(
        runtime=runtime,
        model=model,
        seconds=config.burn_in_seconds,
        chunk_batches=config.chunk_batches,
    )
    measured = _timed_train(
        runtime=runtime,
        model=burn["model"],
        seconds=config.measure_seconds,
        chunk_batches=config.chunk_batches,
    )
    payload = {
        "status": "done",
        "captured_at": _utc_now(),
        "worker_index": config.worker_index,
        "preallocation_env": os.environ.get(PREALLOC_ENV),
        "compile_and_warmup_seconds": compile_seconds,
        "runtime_env": _runtime_env_metadata(os.environ),
        "timing_phases": _timing_phase_metadata(compile_seconds),
        "burn_in": _strip_model(burn),
        "measured": _strip_model(measured),
        "config": asdict(config),
        "scenario_metadata": runtime.metadata,
    }
    _write_json(summary_path, payload)
    _write_json(status_path, payload)
    return 0


def _configure_worker_env(
    env: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Configure subprocess env before workers import JAX."""

    env[PREALLOC_ENV] = "false"
    env.setdefault("PYTHONUNBUFFERED", "1")

    jax_platform = getattr(args, "jax_platform", None)
    if jax_platform:
        env[JAX_PLATFORM_ENV] = str(jax_platform)
        if str(jax_platform) == "cpu":
            env[JAX_PLATFORMS_ENV] = "cpu"

    cpu_threads = int(getattr(args, "cpu_threads_per_worker", 1))
    if jax_platform == "cpu" and cpu_threads > 0:
        thread_value = str(cpu_threads)
        for name in CPU_THREAD_ENV_VARS:
            env[name] = thread_value
        if bool(getattr(args, "xla_cpu_thread_flags", True)):
            env[XLA_FLAGS_ENV] = _with_xla_cpu_thread_flags(
                env.get(XLA_FLAGS_ENV, ""),
                cpu_threads=cpu_threads,
            )

    cache_dir = getattr(args, "jax_compilation_cache_dir", None)
    if cache_dir:
        cache_path = str(Path(str(cache_dir)).expanduser())
        if "://" not in cache_path:
            Path(cache_path).mkdir(parents=True, exist_ok=True)
        env[JAX_COMPILATION_CACHE_DIR_ENV] = cache_path

    min_compile_time = getattr(args, "jax_persistent_cache_min_compile_time_secs", None)
    if min_compile_time is not None:
        env[JAX_CACHE_MIN_COMPILE_TIME_ENV] = str(float(min_compile_time))

    min_entry_size = getattr(args, "jax_persistent_cache_min_entry_size_bytes", None)
    if min_entry_size is not None:
        env[JAX_CACHE_MIN_ENTRY_SIZE_ENV] = str(int(min_entry_size))

    return _runtime_env_metadata(env)


def _with_xla_cpu_thread_flags(existing: str, *, cpu_threads: int) -> str:
    parts = existing.split()
    if not any(part.startswith("--xla_cpu_multi_thread_eigen") for part in parts):
        enabled = "false" if cpu_threads == 1 else "true"
        parts.append(f"--xla_cpu_multi_thread_eigen={enabled}")
    if not any(part.startswith("intra_op_parallelism_threads=") for part in parts):
        parts.append(f"intra_op_parallelism_threads={cpu_threads}")
    return " ".join(parts)


def _runtime_env_metadata(env: Mapping[str, str]) -> dict[str, Any]:
    thread_env = {
        name: env[name]
        for name in CPU_THREAD_ENV_VARS
        if name in env
    }
    cache_dir = env.get(JAX_COMPILATION_CACHE_DIR_ENV)
    return {
        "preallocation_env": env.get(PREALLOC_ENV),
        "jax_platform": env.get(JAX_PLATFORM_ENV),
        "jax_platforms": env.get(JAX_PLATFORMS_ENV),
        "cpu_thread_env": thread_env,
        "xla_flags": env.get(XLA_FLAGS_ENV),
        "compilation_cache": {
            "enabled": cache_dir is not None,
            "dir": cache_dir,
            "min_compile_time_secs": env.get(JAX_CACHE_MIN_COMPILE_TIME_ENV),
            "min_entry_size_bytes": env.get(JAX_CACHE_MIN_ENTRY_SIZE_ENV),
            "expected_effect": "startup_compile_and_warmup_only",
        },
    }


def _timing_phase_metadata(compile_seconds: float) -> dict[str, Any]:
    return {
        "startup": {
            "compile_and_warmup_seconds": compile_seconds,
            "includes_persistent_cache_effects": True,
        },
        "steady_state": {
            "burn_in": "reported separately",
            "measured": "reported separately",
            "includes_startup_compile": False,
        },
    }


def build_cs_nominal_gru_scenario(
    config: Mapping[str, Any],
    seed: int,
) -> ScenarioRuntime:
    import argparse as _argparse

    import jax.random as jr

    from rlrmp.train.task_model import setup_task_model_pair
    from rlrmp.train.cs_nominal_gru import (
        _build_trainer,
        build_hps,
        build_parser as build_nominal_parser,
    )
    from rlrmp.train.cs_perturbation_training import make_broad_epsilon_pgd_pre_step

    nominal_args = build_nominal_parser().parse_args([])
    overrides = _cs_nominal_gru_overrides(config, seed)
    nominal_args = _argparse.Namespace(**{**vars(nominal_args), **overrides})
    hps = build_hps(nominal_args)

    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    return CsNominalGruRuntime(
        pair=pair,
        trainer=_build_trainer(hps),
        pre_step_fn=make_broad_epsilon_pgd_pre_step(hps.broad_epsilon_pgd_training),
        where_train=_make_cs_nominal_gru_where_train(),
        batch_size=int(hps.batch_size),
        key=jr.PRNGKey(seed + 1),
        metadata=_cs_nominal_gru_metadata(hps),
    )


@dataclass
class CsNominalGruRuntime:
    pair: Any
    trainer: Any
    pre_step_fn: Any
    where_train: Any
    batch_size: int
    key: Any
    metadata: dict[str, Any]

    def warmup(self, n_batches: int) -> Any:
        return self._train(
            pair=self.pair,
            n_batches=n_batches,
        )

    def train_chunk(self, model: Any, n_batches: int) -> Any:
        from feedbax.training.types import TaskModelPair

        return self._train(
            pair=TaskModelPair(self.pair.task, model),
            n_batches=n_batches,
        )

    def _train(self, *, pair: Any, n_batches: int) -> Any:
        import jax.random as jr
        from feedbax.training.train import train_pair

        self.key, key_chunk = jr.split(self.key)
        model, _history = train_pair(
            self.trainer,
            pair,
            n_batches=max(1, n_batches),
            key=key_chunk,
            ensembled=True,
            loss_func=pair.task.loss_func,
            pre_step_fn=self.pre_step_fn,
            where_train=self.where_train,
            batch_size=self.batch_size,
            log_step=max(1, n_batches),
            disable_progress=True,
            verbose_progress=False,
        )
        return model


def _cs_nominal_gru_overrides(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "seed": seed,
        "n_input_only": 0,
        "n_readout_only": 0,
        "n_recurrent_only": 0,
    }
    config_dict = dict(config)
    argv = config_dict.pop("argv", None)
    for key, value in config_dict.items():
        overrides[_normalize_cs_nominal_gru_key(str(key))] = value
    if argv is not None:
        from rlrmp.train.cs_nominal_gru import build_parser as build_nominal_parser

        parser = build_nominal_parser()
        defaults = parser.parse_args([])
        parsed = parser.parse_args([str(part) for part in argv])
        overrides.update(
            {
                name: value
                for name, value in vars(parsed).items()
                if value != getattr(defaults, name)
            }
        )

    if "schedule_total_batches" in overrides:
        overrides["n_train_batches"] = max(1, int(overrides.pop("schedule_total_batches")))
    elif "n_train_batches" in overrides:
        overrides["n_train_batches"] = max(1, int(overrides["n_train_batches"]))

    if "broad_epsilon_pgd_level" in overrides:
        overrides["broad_epsilon_level"] = overrides.pop("broad_epsilon_pgd_level")
    if "broad_epsilon_pgd_budget_scale" in overrides:
        overrides["broad_epsilon_budget_scale"] = overrides.pop(
            "broad_epsilon_pgd_budget_scale"
        )
    # The current C&S trainer derives PGD randomness from the training key.
    overrides.pop("broad_epsilon_pgd_seed", None)
    return overrides


def _normalize_cs_nominal_gru_key(key: str) -> str:
    return key.replace("-", "_")


def _make_cs_nominal_gru_where_train() -> dict[int, Any]:
    def where_train_fn(model: Any) -> tuple[Any, ...]:
        net = model.nodes["net"]
        return staged_network_trainable_parts(net)

    return {0: where_train_fn}


def _cs_nominal_gru_metadata(hps: Any) -> dict[str, Any]:
    broad_epsilon = getattr(hps, "broad_epsilon_training", None)
    broad_epsilon_pgd = getattr(hps, "broad_epsilon_pgd_training", None)
    initial_hidden_encoder = getattr(hps, "initial_hidden_encoder", None)
    pgd_inner = getattr(broad_epsilon_pgd, "inner_maximizer", None)
    return {
        "scenario": DEFAULT_SCENARIO,
        "batch_size": int(hps.batch_size),
        "controller_lr": float(hps.learning_rate_0),
        "lr_schedule": str(hps.lr_schedule),
        "lr_warmup_batches": int(hps.constant_lr_iterations),
        "lr_warmup_init_fraction": float(hps.warmup_init_fraction),
        "lr_cosine_alpha": float(hps.cosine_annealing_alpha),
        "gradient_clip_norm": (
            None if hps.gradient_clip_norm is None else float(hps.gradient_clip_norm)
        ),
        "plant_backend": str(hps.model.plant_backend),
        "stochastic_preset": str(hps.model.stochastic_preset),
        "loss_objective": str(hps.loss.objective),
        "n_replicates": int(hps.model.n_replicates),
        "hidden_size": int(hps.model.hidden_size),
        "target_relative_multitarget": bool(hps.target_relative_multitarget.enabled),
        "force_filter_feedback": bool(hps.target_relative_multitarget.force_filter_feedback),
        "perturbation_training": bool(hps.perturbation_training.enabled),
        "perturbation_calibrated_timing": bool(hps.perturbation_training.calibrated_timing),
        "perturbation_physical_level": str(hps.perturbation_training.physical_level),
        "broad_epsilon_training": bool(getattr(broad_epsilon, "enabled", False)),
        "broad_epsilon_pgd_training": bool(getattr(broad_epsilon_pgd, "enabled", False)),
        "broad_epsilon_level": str(getattr(broad_epsilon, "level", "not_applicable")),
        "broad_epsilon_pgd_level": str(getattr(broad_epsilon_pgd, "level", "not_applicable")),
        "broad_epsilon_pgd_steps": (
            None if pgd_inner is None else int(getattr(pgd_inner, "n_steps", 0))
        ),
        "broad_epsilon_pgd_step_size_fraction": (
            None
            if pgd_inner is None
            else float(getattr(pgd_inner, "step_size_fraction_of_l2_radius", 0.0))
        ),
        "initial_hidden_encoder": bool(getattr(initial_hidden_encoder, "enabled", False)),
        "training_diagnostics": bool(hps.training_diagnostics),
        "schedule_total_batches": int(hps.n_batches_condition),
    }


SCENARIO_FACTORIES: dict[str, ScenarioFactory] = {
    DEFAULT_SCENARIO: build_cs_nominal_gru_scenario,
}


def _resolve_scenario_factory(scenario: str) -> ScenarioFactory:
    if scenario in SCENARIO_FACTORIES:
        return SCENARIO_FACTORIES[scenario]
    if ":" not in scenario:
        choices = ", ".join(sorted(SCENARIO_FACTORIES))
        raise ValueError(
            f"Unknown scenario {scenario!r}. Use one of: {choices}, or an import path "
            "formatted as module:function."
        )
    module_name, function_name = scenario.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"Scenario factory {scenario!r} is not callable.")
    return factory


def _load_scenario_config(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise TypeError("--scenario-config-json must decode to a JSON object.")
    return payload


def _timed_train(
    *,
    runtime: ScenarioRuntime,
    model: Any,
    seconds: float,
    chunk_batches: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, seconds)
    total_batches = 0
    chunks = 0
    chunk_times: list[float] = []
    current_model = model
    while time.monotonic() < deadline or chunks == 0:
        chunk_start = time.monotonic()
        current_model = runtime.train_chunk(current_model, max(1, chunk_batches))
        elapsed = time.monotonic() - chunk_start
        chunk_times.append(elapsed)
        total_batches += max(1, chunk_batches)
        chunks += 1
    total_seconds = sum(chunk_times)
    return {
        "model": current_model,
        "chunks": chunks,
        "batches": total_batches,
        "seconds": total_seconds,
        "batches_per_second": total_batches / total_seconds if total_seconds > 0 else None,
        "chunk_seconds": chunk_times,
    }


def _wait_for_ready(
    output_dir: Path,
    n_workers: int,
    timeout_seconds: float,
    procs: Sequence[subprocess.Popen[str]],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ready: list[dict[str, Any]] = []
        for worker_index in range(n_workers):
            status_path = output_dir / f"worker_{worker_index:02d}" / "status.json"
            if not status_path.exists():
                continue
            try:
                payload = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ready":
                ready.append(payload)
        if len(ready) == n_workers:
            return {"status": "ready", "workers": ready}
        failed = [proc.returncode for proc in procs if proc.poll() not in (None, 0)]
        if failed:
            return {"status": "failed_before_ready", "returncodes": failed, "workers": ready}
        time.sleep(1.0)
    return {"status": "ready_timeout", "workers": ready}


def _sample_until_done(
    *,
    output_dir: Path,
    procs: Sequence[subprocess.Popen[str]],
    sample_seconds: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while any(proc.poll() is None for proc in procs):
        samples.append(_resource_sample(procs))
        time.sleep(max(1.0, sample_seconds))
    samples.append(_resource_sample(procs))
    (output_dir / "gpu_samples.json").write_text(_json(samples), encoding="utf-8")
    return samples


def _resource_sample(procs: Sequence[subprocess.Popen[str]]) -> dict[str, Any]:
    sample = _nvidia_smi_sample()
    if "error" not in sample and sample.get("returncode") == 0:
        sample["kind"] = "nvidia_smi"
        return sample
    rss = _rss_sample(procs)
    rss["nvidia_smi"] = sample
    return rss


def _nvidia_smi_sample() -> dict[str, Any]:
    query = (
        "timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,"
        "temperature.gpu"
    )
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return {"captured_at": _utc_now(), "error": str(exc)}
    return {
        "captured_at": _utc_now(),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _rss_sample(procs: Sequence[subprocess.Popen[str]]) -> dict[str, Any]:
    pids = [proc.pid for proc in procs if proc.poll() is None]
    per_process = []
    for pid in pids:
        rss_mib = _rss_mib(pid)
        per_process.append({"pid": pid, "rss_mib": rss_mib})
    rss_values = [item["rss_mib"] for item in per_process if item["rss_mib"] is not None]
    return {
        "captured_at": _utc_now(),
        "kind": "rss",
        "processes": per_process,
        "total_rss_mib": sum(rss_values) if rss_values else None,
        "max_worker_rss_mib": max(rss_values) if rss_values else None,
    }


def _rss_mib(pid: int) -> float | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    try:
        return float(text.splitlines()[0].strip()) / 1024.0
    except ValueError:
        return None


def _collect_worker_summaries(output_dir: Path, n_workers: int) -> list[dict[str, Any]]:
    summaries = []
    for worker_index in range(n_workers):
        summary_path = output_dir / f"worker_{worker_index:02d}" / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text()))
        else:
            summaries.append({"worker_index": worker_index, "status": "missing_summary"})
    return summaries


def _aggregate(
    worker_summaries: Sequence[dict[str, Any]],
    gpu_samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    measured = [w.get("measured", {}) for w in worker_summaries if w.get("status") == "done"]
    rates = [m["batches_per_second"] for m in measured if m.get("batches_per_second") is not None]
    memory_used = []
    total_rss = []
    max_worker_rss = []
    for sample in gpu_samples:
        if sample.get("total_rss_mib") is not None:
            total_rss.append(float(sample["total_rss_mib"]))
        if sample.get("max_worker_rss_mib") is not None:
            max_worker_rss.append(float(sample["max_worker_rss_mib"]))
        stdout = sample.get("stdout", "")
        if not stdout:
            continue
        first_line = stdout.splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) >= 4:
            try:
                memory_used.append(float(parts[3]))
            except ValueError:
                pass
    return {
        "completed_workers": len(rates),
        "aggregate_batches_per_second": sum(rates),
        "mean_worker_batches_per_second": sum(rates) / len(rates) if rates else None,
        "max_memory_used_mib": max(memory_used) if memory_used else None,
        "max_total_rss_mib": max(total_rss) if total_rss else None,
        "max_worker_rss_mib": max(max_worker_rss) if max_worker_rss else None,
    }


def _strip_model(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "model"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(_json(payload), encoding="utf-8")


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

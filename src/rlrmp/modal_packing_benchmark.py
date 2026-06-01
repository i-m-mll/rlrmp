"""Multi-process Modal packing benchmark for nominal C&S GRU training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PREALLOC_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"


@dataclass(frozen=True)
class WorkerConfig:
    worker_index: int
    output_dir: str
    start_file: str
    seed: int
    batch_size: int
    n_replicates: int
    hidden_size: int
    warmup_batches: int
    burn_in_seconds: float
    measure_seconds: float
    chunk_batches: int
    controller_lr: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded nominal C&S GRU multi-process packing benchmark."
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
    parent.add_argument("--batch-size", type=int, default=250)
    parent.add_argument("--n-replicates", type=int, default=5)
    parent.add_argument("--hidden-size", type=int, default=180)
    parent.add_argument("--controller-lr", type=float, default=1e-2)
    parent.add_argument("--seed", type=int, default=42)
    parent.add_argument("--sample-seconds", type=float, default=5.0)

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

    parent_config = vars(args).copy()
    parent_config["captured_at"] = _utc_now()
    parent_config["preallocation"] = "disabled"
    parent_config["python"] = sys.version
    (output_dir / "parent_config.json").write_text(_json(parent_config), encoding="utf-8")

    env = os.environ.copy()
    env[PREALLOC_ENV] = "false"
    env.setdefault("PYTHONUNBUFFERED", "1")

    procs: list[subprocess.Popen[str]] = []
    for worker_index in range(args.n_workers):
        worker_dir = output_dir / f"worker_{worker_index:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_config = WorkerConfig(
            worker_index=worker_index,
            output_dir=str(worker_dir),
            start_file=str(start_file),
            seed=int(args.seed) + worker_index,
            batch_size=int(args.batch_size),
            n_replicates=int(args.n_replicates),
            hidden_size=int(args.hidden_size),
            warmup_batches=int(args.warmup_batches),
            burn_in_seconds=float(args.burn_in_seconds),
            measure_seconds=float(args.measure_seconds),
            chunk_batches=int(args.chunk_batches),
            controller_lr=float(args.controller_lr),
        )
        command = [
            sys.executable,
            "-m",
            "rlrmp.modal_packing_benchmark",
            "worker",
            "--config-json",
            json.dumps(asdict(worker_config), sort_keys=True),
        ]
        stdout = (worker_dir / "stdout.log").open("w", encoding="utf-8")
        stderr = (worker_dir / "stderr.log").open("w", encoding="utf-8")
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
            "preallocation_env": env.get(PREALLOC_ENV),
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

    failed = [proc.returncode for proc in procs if proc.returncode not in (0, None)]
    return 1 if failed else 0


def run_worker(config: WorkerConfig) -> int:
    os.environ[PREALLOC_ENV] = "false"

    import argparse as _argparse
    from functools import partial

    import jax.random as jr
    import optax
    from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule, train_pair

    from rlrmp.train.cs_nominal_gru import build_hps, build_parser as build_nominal_parser
    from rlrmp.modules.training.part2 import setup_task_model_pair

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    summary_path = output_dir / "summary.json"
    _write_json(status_path, {"status": "starting", "captured_at": _utc_now()})

    nominal_args = build_nominal_parser().parse_args([])
    overrides = {
        "seed": config.seed,
        "batch_size": config.batch_size,
        "n_replicates": config.n_replicates,
        "hidden_size": config.hidden_size,
        "n_input_only": config.hidden_size // 3,
        "n_readout_only": config.hidden_size // 3,
        "n_recurrent_only": config.hidden_size - 2 * (config.hidden_size // 3),
        "controller_lr": config.controller_lr,
        "n_train_batches": max(1, config.warmup_batches + config.chunk_batches),
    }
    nominal_args = _argparse.Namespace(**{**vars(nominal_args), **overrides})
    hps = build_hps(nominal_args)

    key = jr.PRNGKey(config.seed)
    key_init, key_warmup, key_burn, key_measure = jr.split(key, 4)
    pair = setup_task_model_pair(hps, key=key_init)
    where_train = _make_where_train()

    def make_trainer(total_steps: int) -> TaskTrainer:
        schedule = make_delayed_cosine_schedule(
            config.controller_lr,
            constant_steps=0,
            total_steps=max(1, total_steps),
        )
        optimizer = optax.inject_hyperparams(partial(optax.adamw, weight_decay=0.0))(
            learning_rate=schedule
        )
        return TaskTrainer(optimizer=optimizer, checkpointing=False)

    _write_json(status_path, {"status": "compiling", "captured_at": _utc_now()})
    compile_start = time.monotonic()
    model, _history = train_pair(
        make_trainer(config.warmup_batches),
        pair,
        n_batches=config.warmup_batches,
        key=key_warmup,
        ensembled=True,
        loss_func=pair.task.loss_func,
        where_train=where_train,
        batch_size=config.batch_size,
        log_step=max(1, config.warmup_batches),
        disable_progress=True,
        verbose_progress=False,
    )
    compile_seconds = time.monotonic() - compile_start
    _write_json(
        status_path,
        {
            "status": "ready",
            "captured_at": _utc_now(),
            "compile_and_warmup_seconds": compile_seconds,
            "config": asdict(config),
        },
    )

    start_file = Path(config.start_file)
    while not start_file.exists():
        time.sleep(0.2)

    burn = _timed_train(
        trainer=make_trainer(max(1, config.chunk_batches)),
        pair=pair,
        model=model,
        seconds=config.burn_in_seconds,
        chunk_batches=config.chunk_batches,
        key=key_burn,
        where_train=where_train,
        batch_size=config.batch_size,
    )
    measured = _timed_train(
        trainer=make_trainer(max(1, config.chunk_batches)),
        pair=pair,
        model=burn["model"],
        seconds=config.measure_seconds,
        chunk_batches=config.chunk_batches,
        key=key_measure,
        where_train=where_train,
        batch_size=config.batch_size,
    )
    payload = {
        "status": "done",
        "captured_at": _utc_now(),
        "worker_index": config.worker_index,
        "preallocation_env": os.environ.get(PREALLOC_ENV),
        "compile_and_warmup_seconds": compile_seconds,
        "burn_in": _strip_model(burn),
        "measured": _strip_model(measured),
        "config": asdict(config),
    }
    _write_json(summary_path, payload)
    _write_json(status_path, payload)
    return 0


def _timed_train(
    *,
    trainer: Any,
    pair: Any,
    model: Any,
    seconds: float,
    chunk_batches: int,
    key: Any,
    where_train: Any,
    batch_size: int,
) -> dict[str, Any]:
    import jax.random as jr
    from feedbax.training.train import train_pair
    from feedbax.types import TaskModelPair

    deadline = time.monotonic() + max(0.0, seconds)
    total_batches = 0
    chunks = 0
    chunk_times: list[float] = []
    current_model = model
    while time.monotonic() < deadline or chunks == 0:
        key, key_chunk = jr.split(key)
        chunk_start = time.monotonic()
        current_model, _history = train_pair(
            trainer,
            TaskModelPair(pair.task, current_model),
            n_batches=max(1, chunk_batches),
            key=key_chunk,
            ensembled=True,
            loss_func=pair.task.loss_func,
            where_train=where_train,
            batch_size=batch_size,
            log_step=max(1, chunk_batches),
            disable_progress=True,
            verbose_progress=False,
        )
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


def _make_where_train() -> dict[int, Any]:
    def where_train_fn(model: Any) -> tuple[Any, ...]:
        net = model.nodes["net"]
        return (net.hidden, net.readout)

    return {0: where_train_fn}


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
        samples.append(_nvidia_smi_sample())
        time.sleep(max(1.0, sample_seconds))
    samples.append(_nvidia_smi_sample())
    (output_dir / "gpu_samples.json").write_text(_json(samples), encoding="utf-8")
    return samples


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
    for sample in gpu_samples:
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
    }


def _strip_model(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "model"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(_json(payload), encoding="utf-8")


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

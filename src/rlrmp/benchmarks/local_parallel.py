"""Local CPU parallel packing benchmark for nominal C&S GRU training.

This is a temporary measurement harness for issue ``3b2af27``. It mirrors the
Modal packing benchmark's parent/worker shape, but forces JAX onto CPU and
samples resident memory with local ``ps`` rather than ``nvidia-smi``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rlrmp.benchmarks._common import run_parent_processes
from rlrmp.model.trainable import staged_network_trainable_parts


PREALLOC_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
JAX_PLATFORM_ENV = "JAX_PLATFORM_NAME"


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
    stochastic_preset: str
    regularized_fidelity: bool
    plant_backend: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded local CPU multi-process C&S GRU packing benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parent = subparsers.add_parser("parent")
    parent.add_argument("--output-dir", required=True)
    parent.add_argument("--n-workers", type=int, required=True)
    parent.add_argument("--stagger-seconds", type=float, default=1.0)
    parent.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    parent.add_argument("--burn-in-seconds", type=float, default=30.0)
    parent.add_argument("--measure-seconds", type=float, default=60.0)
    parent.add_argument("--warmup-batches", type=int, default=1)
    parent.add_argument("--chunk-batches", type=int, default=1)
    parent.add_argument("--batch-size", type=int, default=32)
    parent.add_argument("--n-replicates", type=int, default=1)
    parent.add_argument("--hidden-size", type=int, default=32)
    parent.add_argument("--controller-lr", type=float, default=1e-2)
    parent.add_argument("--stochastic-preset", default="cs2019-rollout")
    parent.add_argument("--regularized-fidelity", action="store_true")
    parent.add_argument("--plant-backend", default="cs_lss")
    parent.add_argument("--seed", type=int, default=42)
    parent.add_argument("--sample-seconds", type=float, default=2.0)

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
    parent_config = vars(args).copy()
    parent_config["captured_at"] = _utc_now()
    parent_config["preallocation"] = "disabled"
    parent_config["jax_platform"] = "cpu"
    parent_config["python"] = sys.version
    env = os.environ.copy()
    env[PREALLOC_ENV] = "false"
    env[JAX_PLATFORM_ENV] = "cpu"
    env.setdefault("PYTHONUNBUFFERED", "1")

    def worker_config(worker_index: int, worker_dir: Path, start_file: Path) -> WorkerConfig:
        return WorkerConfig(
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
            stochastic_preset=str(args.stochastic_preset),
            regularized_fidelity=bool(args.regularized_fidelity),
            plant_backend=str(args.plant_backend),
        )

    return run_parent_processes(
        output_dir=output_dir,
        n_workers=int(args.n_workers),
        worker_module="rlrmp.benchmarks.local_parallel",
        worker_config=worker_config,
        env=env,
        parent_config=parent_config,
        start_payload={
            "released_at": _utc_now(),
            "n_workers": args.n_workers,
            "burn_in_seconds": args.burn_in_seconds,
            "measure_seconds": args.measure_seconds,
        },
        stagger_seconds=float(args.stagger_seconds),
        ready_timeout_seconds=float(args.ready_timeout_seconds),
        sample_seconds=float(args.sample_seconds),
        wait_for_ready=_wait_for_ready,
        sample_until_done=_sample_until_done,
        collect_worker_summaries=_collect_worker_summaries,
        aggregate=_aggregate,
        summary_fields={
            "captured_at": _utc_now(),
            "n_workers": args.n_workers,
            "preallocation_env": env.get(PREALLOC_ENV),
            "jax_platform": env.get(JAX_PLATFORM_ENV),
        },
        sample_field="memory_samples",
        summary_marker="RLRMP_LOCAL_PACKING_SUMMARY",
    )


def run_worker(config: WorkerConfig) -> int:
    os.environ[PREALLOC_ENV] = "false"
    os.environ[JAX_PLATFORM_ENV] = "cpu"
    raise RuntimeError(
        "local_parallel benchmark worker used the retired Feedbax trainer path; "
        "port this benchmark to the RLRMP native executor before running it."
    )


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
    raise RuntimeError(
        "local_parallel timed training used the retired Feedbax trainer path; "
        "port this benchmark to the RLRMP native executor before running it."
    )


def _make_where_train() -> dict[int, Any]:
    def where_train_fn(model: Any) -> tuple[Any, ...]:
        net = model.nodes["net"]
        return staged_network_trainable_parts(net)

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
        time.sleep(0.5)
    return {"status": "ready_timeout", "workers": ready}


def _sample_until_done(
    *,
    output_dir: Path,
    procs: Sequence[subprocess.Popen[str]],
    sample_seconds: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while any(proc.poll() is None for proc in procs):
        samples.append(_rss_sample(procs))
        time.sleep(max(0.5, sample_seconds))
    samples.append(_rss_sample(procs))
    (output_dir / "memory_samples.json").write_text(_json(samples), encoding="utf-8")
    return samples


def _rss_sample(procs: Sequence[subprocess.Popen[str]]) -> dict[str, Any]:
    pids = [proc.pid for proc in procs if proc.poll() is None]
    per_process = []
    for pid in pids:
        rss_mib = _rss_mib(pid)
        per_process.append({"pid": pid, "rss_mib": rss_mib})
    rss_values = [item["rss_mib"] for item in per_process if item["rss_mib"] is not None]
    return {
        "captured_at": _utc_now(),
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
    memory_samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    measured = [w.get("measured", {}) for w in worker_summaries if w.get("status") == "done"]
    rates = [m["batches_per_second"] for m in measured if m.get("batches_per_second") is not None]
    total_rss = [
        sample["total_rss_mib"]
        for sample in memory_samples
        if sample.get("total_rss_mib") is not None
    ]
    max_worker_rss = [
        sample["max_worker_rss_mib"]
        for sample in memory_samples
        if sample.get("max_worker_rss_mib") is not None
    ]
    return {
        "completed_workers": len(rates),
        "aggregate_batches_per_second": sum(rates),
        "mean_worker_batches_per_second": sum(rates) / len(rates) if rates else None,
        "max_total_rss_mib": max(total_rss) if total_rss else None,
        "max_worker_rss_mib": max(max_worker_rss) if max_worker_rss else None,
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

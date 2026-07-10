"""Shared process-parent mechanics for packing benchmarks."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


def run_parent_processes(
    *,
    output_dir: Path,
    n_workers: int,
    worker_module: str,
    worker_config: Callable[[int, Path, Path], Any],
    env: Mapping[str, str],
    parent_config: Mapping[str, Any],
    start_payload: Mapping[str, Any],
    stagger_seconds: float,
    ready_timeout_seconds: float,
    sample_seconds: float,
    wait_for_ready: Callable[
        [Path, int, float, Sequence[subprocess.Popen[str]]], dict[str, Any]
    ],
    sample_until_done: Callable[..., list[dict[str, Any]]],
    collect_worker_summaries: Callable[[Path, int], list[dict[str, Any]]],
    aggregate: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],
    summary_fields: Mapping[str, Any],
    sample_field: str,
    summary_marker: str,
) -> int:
    """Spawn benchmark workers, synchronize them, sample resources, and summarize."""

    output_dir.mkdir(parents=True, exist_ok=True)
    start_file = output_dir / "start.json"
    start_file.unlink(missing_ok=True)
    _write_json(output_dir / "parent_config.json", parent_config)

    procs: list[subprocess.Popen[str]] = []
    handles = []
    for worker_index in range(n_workers):
        worker_dir = output_dir / f"worker_{worker_index:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        config = worker_config(worker_index, worker_dir, start_file)
        command = [
            sys.executable,
            "-m",
            worker_module,
            "worker",
            "--config-json",
            json.dumps(asdict(config), sort_keys=True),
        ]
        stdout = (worker_dir / "stdout.log").open("w", encoding="utf-8")
        stderr = (worker_dir / "stderr.log").open("w", encoding="utf-8")
        handles.extend([stdout, stderr])
        procs.append(
            subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=dict(env),
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        )
        time.sleep(max(0.0, stagger_seconds))

    try:
        ready = wait_for_ready(output_dir, n_workers, ready_timeout_seconds, procs)
        _write_json(output_dir / "ready_summary.json", ready)
        _write_json(start_file, start_payload)
        samples = sample_until_done(
            output_dir=output_dir,
            procs=procs,
            sample_seconds=sample_seconds,
        )
        workers = collect_worker_summaries(output_dir, n_workers)
        summary = {
            **summary_fields,
            "ready": ready,
            "workers": workers,
            sample_field: samples,
            "aggregate": aggregate(workers, samples),
        }
        _write_json(output_dir / "summary.json", summary)
        print(f"{summary_marker}_START", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"{summary_marker}_END", flush=True)
    finally:
        _terminate_processes(procs)
        for handle in handles:
            handle.close()

    return int(any(proc.returncode not in (0, None) for proc in procs))


def _terminate_processes(procs: Sequence[subprocess.Popen[str]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 30.0
    for proc in procs:
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if proc.poll() is None:
            proc.kill()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

"""Tests for the local CPU GRU packing benchmark harness."""

from __future__ import annotations

from rlrmp.local_parallel_benchmark import _aggregate, build_parser


def test_parent_parser_defaults_to_cs_lss_cpu_contract() -> None:
    args = build_parser().parse_args(["parent", "--output-dir", "/tmp/out", "--n-workers", "2"])

    assert args.command == "parent"
    assert args.n_workers == 2
    assert args.plant_backend == "cs_lss"
    assert args.stagger_seconds == 1.0
    assert args.measure_seconds == 60.0


def test_aggregate_reports_rate_and_memory_scaling() -> None:
    workers = [
        {"status": "done", "measured": {"batches_per_second": 1.5}},
        {"status": "done", "measured": {"batches_per_second": 2.5}},
    ]
    memory_samples = [
        {"total_rss_mib": 100.0, "max_worker_rss_mib": 60.0},
        {"total_rss_mib": 150.0, "max_worker_rss_mib": 80.0},
    ]

    summary = _aggregate(workers, memory_samples)

    assert summary["completed_workers"] == 2
    assert summary["aggregate_batches_per_second"] == 4.0
    assert summary["mean_worker_batches_per_second"] == 2.0
    assert summary["max_total_rss_mib"] == 150.0
    assert summary["max_worker_rss_mib"] == 80.0

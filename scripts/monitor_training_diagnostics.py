#!/usr/bin/env python
"""Summarize live rlrmp training diagnostics for local or RunPod runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS = (
    "train_loss__total",
    "validation_loss__total",
    "pgd_broad_epsilon_inner_objective_improvement",
    "pgd_broad_epsilon_inner_objective_best",
    "pgd_broad_epsilon_inner_objective_final_endpoint",
    "pgd_broad_epsilon_inner_objective_final_endpoint_gap",
    "pgd_broad_epsilon_epsilon_norm_radius_ratio_mean",
    "pgd_broad_epsilon_boundary_fraction",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print latest training diagnostics from one or more output directories.",
    )
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    summaries = [summarize_output_dir(path) for path in args.output_dirs]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        for summary in summaries:
            print(render_text(summary))


def summarize_output_dir(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser()
    diagnostics_path = output_dir / "training_diagnostics.npz"
    manifest_path = output_dir / "training_diagnostics.json"
    summary_path = output_dir / "training_summary.json"
    result: dict[str, Any] = {
        "output_dir": str(output_dir),
        "exists": output_dir.exists(),
        "diagnostics_npz": str(diagnostics_path),
        "ok": True,
        "alerts": [],
        "latest": {},
    }
    manifest = _read_json(manifest_path)
    training_summary = _read_json(summary_path)
    if manifest is not None:
        result["manifest_completed_batches"] = manifest.get("completed_batches")
    if training_summary is not None:
        result["summary_completed_batches"] = training_summary.get("completed_batches")
        result["stopped_early_for_checkpoint_gate"] = training_summary.get(
            "stopped_early_for_checkpoint_gate"
        )
    _add_checkpoint_progress(output_dir, result)
    if not diagnostics_path.exists():
        result["ok"] = False
        if "checkpoint_completed_batches" in result or "latest_history_chunk_batch" in result:
            result["alerts"].append(
                "training_diagnostics.npz not written yet; "
                "checkpoint/history progress exists"
            )
        else:
            result["alerts"].append("missing training_diagnostics.npz")
        return result

    with np.load(diagnostics_path) as diagnostics:
        keys = set(diagnostics.files)
        pgd_sampled = (
            np.asarray(diagnostics["pgd_broad_epsilon_diagnostic_sampled"], dtype=bool)
            if "pgd_broad_epsilon_diagnostic_sampled" in keys
            else None
        )
        if "batch_index" in keys and diagnostics["batch_index"].size:
            result["latest_batch_index"] = _json_scalar(diagnostics["batch_index"][-1])
            result["n_logged_batches"] = int(diagnostics["batch_index"].shape[0])
        else:
            result["ok"] = False
            result["alerts"].append("missing or empty batch_index")
        for metric in DEFAULT_METRICS:
            if metric in keys and diagnostics[metric].size:
                result["latest"][metric] = _latest_stats(diagnostics[metric])
        for name in sorted(keys):
            array = diagnostics[name]
            checked = _finite_check_view(name, array, pgd_sampled)
            if (
                checked.size
                and np.issubdtype(checked.dtype, np.number)
                and not np.isfinite(checked).all()
            ):
                result["ok"] = False
                result["alerts"].append(f"nonfinite values in {name}")

    _add_metric_alerts(result)
    return result


def _add_checkpoint_progress(output_dir: Path, result: dict[str, Any]) -> None:
    checkpoint_index = _read_json(output_dir / "checkpoints" / "checkpoint_index.json")
    if checkpoint_index is not None:
        completed = checkpoint_index.get("completed_batches")
        if completed is not None:
            result["checkpoint_completed_batches"] = completed
        latest = checkpoint_index.get("latest")
        if latest is not None:
            result["latest_checkpoint"] = latest
    history_dir = output_dir / "history_chunks"
    if history_dir.exists():
        batches: list[int] = []
        for path in history_dir.glob("history_*.eqx"):
            try:
                batches.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        if batches:
            result["latest_history_chunk_batch"] = max(batches)
            result["n_history_chunks"] = len(batches)

def _finite_check_view(
    name: str,
    array: np.ndarray,
    pgd_sampled: np.ndarray | None,
) -> np.ndarray:
    if (
        pgd_sampled is not None
        and name.startswith("pgd_broad_epsilon_")
        and name != "pgd_broad_epsilon_diagnostic_sampled"
        and array.shape[:1] == pgd_sampled.shape[:1]
    ):
        return array[pgd_sampled]
    return array


def _latest_stats(array: np.ndarray) -> dict[str, Any]:
    latest = np.asarray(array[-1])
    if latest.dtype == np.bool_:
        return {"value": bool(np.all(latest))}
    if latest.ndim == 0:
        return {"value": _json_scalar(latest)}
    finite = np.isfinite(latest)
    return {
        "shape": list(latest.shape),
        "mean": _json_scalar(np.nanmean(latest)),
        "min": _json_scalar(np.nanmin(latest)),
        "max": _json_scalar(np.nanmax(latest)),
        "finite": bool(finite.all()),
    }


def _add_metric_alerts(result: dict[str, Any]) -> None:
    latest = result.get("latest", {})
    ratio = latest.get("pgd_broad_epsilon_epsilon_norm_radius_ratio_mean", {})
    ratio_max = ratio.get("max", ratio.get("value"))
    if ratio_max is not None and ratio_max > 1.0001:
        result["ok"] = False
        result["alerts"].append(f"PGD epsilon radius ratio exceeds 1: {ratio_max:.6g}")
    gap = latest.get("pgd_broad_epsilon_inner_objective_final_endpoint_gap", {})
    gap_min = gap.get("min", gap.get("value"))
    if gap_min is not None and gap_min < -1e-6:
        result["ok"] = False
        result["alerts"].append(f"PGD best/final gap is negative: {gap_min:.6g}")


def render_text(summary: dict[str, Any]) -> str:
    lines = [
        f"{summary['output_dir']}",
        f"  ok: {summary['ok']}",
        f"  latest_batch_index: {summary.get('latest_batch_index', 'n/a')}",
    ]
    if "checkpoint_completed_batches" in summary:
        lines.append(
            f"  checkpoint_completed_batches: {summary['checkpoint_completed_batches']}"
        )
    if "latest_history_chunk_batch" in summary:
        lines.append(
            f"  latest_history_chunk_batch: {summary['latest_history_chunk_batch']} "
            f"({summary.get('n_history_chunks', 0)} chunks)"
        )
    for alert in summary.get("alerts", []):
        lines.append(f"  alert: {alert}")
    for name, stats in summary.get("latest", {}).items():
        if "value" in stats:
            value = stats["value"]
            lines.append(f"  {name}: {value}")
        else:
            lines.append(
                f"  {name}: mean={stats['mean']:.6g} "
                f"min={stats['min']:.6g} max={stats['max']:.6g} "
                f"finite={stats['finite']}"
            )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_scalar(value: Any) -> float | int | bool:
    scalar = np.asarray(value).item()
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    return scalar


if __name__ == "__main__":
    main()

"""Materialize ae9f30f frozen finite-policy audit metadata for issue 3c5836c."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rlrmp.analysis.frozen_adversary_audit import summarize_active_broad_epsilon_optimizer
from rlrmp.paths import portable_repo_path

ISSUE_ID = "3c5836c"
SOURCE_ISSUE_ID = "ae9f30f"
SCHEMA_VERSION = "rlrmp.frozen_finite_policy_audit.v1"

ROW_IDS = (
    "direct_epsilon_b1p05",
    "direct_epsilon_b1p4",
    "linear_no_bias_b1p05",
    "linear_no_bias_b1p4",
)

DIAGNOSTIC_COMPLETED_BATCHES = (500, 1000, 3000, 6000, 12000)

PGD_METRICS = (
    "pgd_broad_epsilon_selected_objective_gain_over_zero",
    "pgd_broad_epsilon_epsilon_energy_mean",
    "pgd_broad_epsilon_epsilon_energy_max",
    "pgd_broad_epsilon_epsilon_norm_radius_ratio_mean",
    "pgd_broad_epsilon_epsilon_norm_radius_ratio_max",
    "pgd_broad_epsilon_cap_boundary_fraction",
    "pgd_broad_epsilon_safety_cap_boundary_fraction",
    "pgd_broad_epsilon_inner_objective_improvement",
    "pgd_broad_epsilon_inner_objective_after",
    "pgd_broad_epsilon_inner_objective_best",
    "pgd_broad_epsilon_inner_objective_final_endpoint_gap",
    "pgd_broad_epsilon_penalized_objective_zero",
    "pgd_broad_epsilon_penalized_objective_selected",
    "pgd_broad_epsilon_penalized_objective_final_endpoint",
    "pgd_broad_epsilon_energy_penalty_term_selected",
    "pgd_broad_epsilon_energy_penalty_term_final_endpoint",
    "pgd_broad_epsilon_raw_task_loss_zero",
    "pgd_broad_epsilon_raw_task_loss_selected",
    "pgd_broad_epsilon_raw_task_loss_final_endpoint",
)

REPLAY_BLOCKERS = (
    "Exact selected direct-epsilon tensors were not persisted, so exact same-batch "
    "direct-to-linear projection cannot be reconstructed from this cache.",
    "The saved run cache lacks a compact replay descriptor for each selected batch: "
    "raw trial batch, per-update PRNG subkey, pre-update checkpoint/model id, and "
    "trial target metadata are not all present together.",
    "The issue-linked branch used for this audit has reusable finite-policy primitives, "
    "but exact ae9f30f live finite-graph replay should wait until the run-producing "
    "finite graph integration is committed or otherwise made available.",
    "No ae9f30f affine finite-policy row artifacts were present locally, so affine "
    "lambda/gain-bias conclusions are not inferred from these rows.",
)

STABILIZATION_CAVEATS = (
    "Small stabilization-table values are not by themselves evidence of a unit bug: "
    "endpoint/reach is dimensionless endpoint delta divided by 0.15 m, and AUC dx is "
    "raw m*s even though older notes often displayed mm*s.",
    "The current ae9f30f stabilization note labels the baseline as "
    "no_pgd_h0_6d_const_band16, but the table source was the 020a65b calibrated H0/no-PGD "
    "artifact. The explicit prior const-band16 artifact appears to be 3244f1a/33b0dcb "
    "and is same-family but not numerically identical.",
    "Future stabilization tables should show mm and mm*s companion columns and blocked-row "
    "counts beside evaluated-row counts. The linear_no_bias_b1p05 plant/command comparison "
    "is fragile because only 54 rows were evaluated while 144 process-epsilon rows were "
    "blocked.",
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    result_root = repo_root / "results" / ISSUE_ID
    notes_dir = result_root / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    artifact_root = repo_root / "_artifacts" / SOURCE_ISSUE_ID
    rows = [
        summarize_row(repo_root=repo_root, artifact_root=artifact_root, row_id=row_id)
        for row_id in ROW_IDS
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_ID,
        "source_issue": SOURCE_ISSUE_ID,
        "artifact_root": _repo_rel(repo_root, artifact_root),
        "rows": rows,
        "confirmed_facts": confirmed_facts(rows),
        "unsupported_or_blocked": list(REPLAY_BLOCKERS),
        "stabilization_caveats": list(STABILIZATION_CAVEATS),
        "next_replay_instrumentation": [
            "selected epsilon arrays or compact replay representation",
            "checkpoint/pre-update model id for the frozen objective",
            "deterministic batch descriptor and per-update PRNG key/subkey",
            "trial targets and active optimizer config",
            "finite-policy parameters at selected, final-endpoint, and zero proposals",
        ],
    }

    json_path = notes_dir / "frozen_finite_policy_audit_manifest.json"
    markdown_path = notes_dir / "frozen_finite_policy_audit_metadata.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {_repo_rel(repo_root, json_path)}")
    print(f"wrote {_repo_rel(repo_root, markdown_path)}")


def summarize_row(*, repo_root: Path, artifact_root: Path, row_id: str) -> dict[str, Any]:
    run_spec_path = artifact_root / "runpod_wave1_remote" / "result_runs" / f"{row_id}.json"
    run_spec = read_json(run_spec_path)
    if isinstance(run_spec, Sequence) and not isinstance(run_spec, (str, bytes, bytearray)):
        if len(run_spec) != 1 or not isinstance(run_spec[0], Mapping):
            raise ValueError(f"unexpected run spec list shape in {run_spec_path}")
        run_spec = run_spec[0]
    if not isinstance(run_spec, Mapping):
        raise ValueError(f"run spec must be a JSON object in {run_spec_path}")
    artifact_dir = artifact_root / "runs" / row_id
    if not artifact_dir.exists():
        artifact_dir = artifact_root / "runpod_wave1_remote" / "artifact_runs" / row_id
    optimizer = summarize_active_broad_epsilon_optimizer(run_spec)
    checkpoints = summarize_checkpoints(repo_root=repo_root, artifact_dir=artifact_dir)
    training_summary = summarize_training_summary(repo_root=repo_root, artifact_dir=artifact_dir)
    diagnostics = summarize_training_diagnostics(repo_root=repo_root, artifact_dir=artifact_dir)
    sentinels = summarize_sentinels(repo_root=repo_root, artifact_dir=artifact_dir)
    logs = summarize_logs(repo_root=repo_root, artifact_dir=artifact_dir)

    status = infer_row_status(
        training_summary=training_summary,
        checkpoints=checkpoints,
        diagnostics=diagnostics,
        sentinels=sentinels,
    )
    return {
        "row_id": row_id,
        "status": status,
        "run_spec_path": _repo_rel(repo_root, run_spec_path),
        "artifact_dir": _repo_rel(repo_root, artifact_dir),
        "optimizer": optimizer,
        "training_summary": training_summary,
        "checkpoints": checkpoints,
        "training_diagnostics": diagnostics,
        "sentinels": sentinels,
        "logs": logs,
    }


def summarize_checkpoints(*, repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    checkpoint_dir = artifact_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return {
            "exists": False,
            "checkpoint_dir": _repo_rel(repo_root, checkpoint_dir),
            "completed_batches": [],
            "sample_metadata": [],
        }
    completed_batches: list[int] = []
    for path in sorted(checkpoint_dir.glob("checkpoint_[0-9]*")):
        match = re.search(r"checkpoint_(\d+)$", path.name)
        if match:
            completed_batches.append(int(match.group(1)))
    index_path = checkpoint_dir / "checkpoint_index.json"
    index = read_json(index_path) if index_path.exists() else None
    sample_batches = _sample_batches(completed_batches)
    sample_metadata = []
    for completed in sample_batches:
        metadata_path = checkpoint_dir / f"checkpoint_{completed:07d}" / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        sample_metadata.append(
            {
                "completed_batches": completed,
                "metadata_path": _repo_rel(repo_root, metadata_path),
                "next_prng_key": _mapping(metadata).get("next_prng_key"),
                "n_train_batches": _mapping(metadata).get("n_train_batches"),
                "checkpoint_interval_batches": _mapping(metadata).get(
                    "checkpoint_interval_batches"
                ),
            }
        )
    return {
        "exists": True,
        "checkpoint_dir": _repo_rel(repo_root, checkpoint_dir),
        "count": len(completed_batches),
        "completed_batches": completed_batches,
        "latest": _mapping(index).get("latest"),
        "index": index,
        "sample_metadata": sample_metadata,
    }


def summarize_training_summary(*, repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    summary_path = artifact_dir / "training_summary.json"
    if not summary_path.exists():
        return {"exists": False, "path": _repo_rel(repo_root, summary_path)}
    summary = _mapping(read_json(summary_path))
    return {
        "exists": True,
        "path": _repo_rel(repo_root, summary_path),
        "completed_batches": summary.get("completed_batches"),
        "n_train_batches": summary.get("n_train_batches"),
        "stopped_early_for_checkpoint_gate": summary.get("stopped_early_for_checkpoint_gate"),
        "training_batches_per_second": summary.get("training_batches_per_second"),
        "latest_checkpoint": summary.get("latest_checkpoint"),
        "final_model_path": summary.get("final_model_path"),
        "final_adversary_policy_path": summary.get("final_adversary_policy_path"),
    }


def summarize_training_diagnostics(*, repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    diagnostics_path = artifact_dir / "training_diagnostics.npz"
    manifest_path = artifact_dir / "training_diagnostics.json"
    if not diagnostics_path.exists():
        return {
            "exists": False,
            "npz_path": _repo_rel(repo_root, diagnostics_path),
            "manifest_path": _repo_rel(repo_root, manifest_path),
            "sample_rows": [],
        }
    sample_rows: list[dict[str, Any]] = []
    with np.load(diagnostics_path) as diagnostics:
        batch_index = np.asarray(diagnostics["batch_index"], dtype=np.int64)
        sample_completed = [
            completed
            for completed in DIAGNOSTIC_COMPLETED_BATCHES
            if completed - 1 in set(batch_index.tolist())
        ]
        for completed in sample_completed:
            index = int(np.nonzero(batch_index == completed - 1)[0][0])
            row: dict[str, Any] = {
                "completed_batches": completed,
                "batch_index": int(batch_index[index]),
            }
            for metric in PGD_METRICS:
                if metric in diagnostics:
                    row[metric] = array_summary_at(diagnostics[metric], index)
            sample_rows.append(row)
        latest = {}
        if len(batch_index):
            latest_index = int(len(batch_index) - 1)
            for metric in PGD_METRICS:
                if metric in diagnostics:
                    latest[metric] = array_summary_at(diagnostics[metric], latest_index)
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    return {
        "exists": True,
        "npz_path": _repo_rel(repo_root, diagnostics_path),
        "manifest_path": _repo_rel(repo_root, manifest_path),
        "manifest": manifest,
        "sample_rows": sample_rows,
        "latest": latest,
    }


def array_summary_at(array: np.ndarray, index: int) -> dict[str, Any]:
    values = np.asarray(array[index])
    if values.dtype == np.bool_:
        return {
            "any": bool(np.any(values)),
            "all": bool(np.all(values)),
            "true_fraction": float(np.mean(values.astype(np.float64))),
        }
    if values.shape == ():
        value = float(values)
        return {"value": value, "finite": bool(np.isfinite(value))}
    values = values.astype(np.float64)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "finite": bool(np.isfinite(values).all()),
    }


def summarize_sentinels(*, repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    sentinel_dir = artifact_dir / "sentinels"
    if not sentinel_dir.exists():
        return {"exists": False, "sentinel_dir": _repo_rel(repo_root, sentinel_dir), "files": []}
    files = []
    for path in sorted(sentinel_dir.glob("*")):
        content = (
            path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else ""
        )
        files.append(
            {
                "name": path.name,
                "path": _repo_rel(repo_root, path),
                "size_bytes": path.stat().st_size,
                "content_preview": content[:200],
            }
        )
    return {"exists": True, "sentinel_dir": _repo_rel(repo_root, sentinel_dir), "files": files}


def summarize_logs(*, repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    log_dir = artifact_dir / "logs"
    if not log_dir.exists():
        return {"exists": False, "log_dir": _repo_rel(repo_root, log_dir), "files": []}
    files = []
    for path in sorted(log_dir.glob("*.log")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        batch_lines = [line for line in lines if line.startswith("BATCH ")]
        files.append(
            {
                "name": path.name,
                "path": _repo_rel(repo_root, path),
                "size_bytes": path.stat().st_size,
                "n_lines": len(lines),
                "n_batch_lines": len(batch_lines),
                "last_batch_lines": batch_lines[-3:],
            }
        )
    return {"exists": True, "log_dir": _repo_rel(repo_root, log_dir), "files": files}


def infer_row_status(
    *,
    training_summary: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    sentinels: Mapping[str, Any],
) -> str:
    if training_summary.get("exists"):
        completed = training_summary.get("completed_batches")
        expected = training_summary.get("n_train_batches")
        if completed is not None and expected is not None and int(completed) >= int(expected):
            return (
                "completed_with_training_diagnostics" if diagnostics.get("exists") else "completed"
            )
        return "partial_with_training_summary"
    sentinel_names = {file_info.get("name") for file_info in sentinels.get("files", [])}
    if checkpoints.get("completed_batches") and {"linear_no_bias_b1p4.stopped"}.issubset(
        sentinel_names
    ):
        return "stopped_context_only"
    if checkpoints.get("completed_batches"):
        return "checkpoint_context_only"
    return "missing_local_artifacts"


def confirmed_facts(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    facts = [
        "The ae9f30f finite linear_no_bias rows used the broad-epsilon PGD lane, "
        "not the disabled policy_adversary_training Adam metadata.",
        "linear_no_bias_b1p4 has checkpoint context at 500 and 1000 batches but no "
        "completed training summary or training_diagnostics.npz in the local artifact cache.",
    ]
    for row in rows:
        if row.get("row_id") == "linear_no_bias_b1p05":
            diagnostics = _mapping(row.get("training_diagnostics"))
            latest = _mapping(diagnostics.get("latest"))
            sample_rows = list(diagnostics.get("sample_rows", []))
            gain = _metric_mean(latest, "pgd_broad_epsilon_selected_objective_gain_over_zero")
            energy = _metric_mean(latest, "pgd_broad_epsilon_epsilon_energy_mean")
            final_energy = _metric_mean(
                latest,
                "pgd_broad_epsilon_energy_penalty_term_final_endpoint",
            )
            early_energy = _sample_metric_mean(
                sample_rows,
                completed_batches=500,
                metric="pgd_broad_epsilon_epsilon_energy_mean",
            )
            mid_energy = _sample_metric_mean(
                sample_rows,
                completed_batches=3000,
                metric="pgd_broad_epsilon_epsilon_energy_mean",
            )
            if early_energy is not None and early_energy > 0.0 and mid_energy == 0.0:
                facts.append(
                    "The completed linear_no_bias_b1p05 row was not zero at every sampled "
                    "early checkpoint: selected energy was nonzero at 500 batches, but the "
                    "saved samples show zero selected energy by 3000 batches and at the final "
                    "batch."
                )
            if gain == 0.0 and energy == 0.0:
                facts.append(
                    "The completed linear_no_bias_b1p05 diagnostics selected zero epsilon at the "
                    "final logged batch: selected gain over zero and selected epsilon energy "
                    "are both zero in the saved scalar diagnostics."
                )
            if final_energy is not None and final_energy > 0.0:
                facts.append(
                    "The linear_no_bias_b1p05 final PGD endpoint was not necessarily zero; "
                    "the saved diagnostics show it was rejected in favor of the zero selected "
                    "candidate under the active lambda/objective."
                )
        if str(row.get("row_id", "")).startswith("direct_epsilon_"):
            diagnostics = _mapping(row.get("training_diagnostics"))
            latest = _mapping(diagnostics.get("latest"))
            gain = _metric_mean(latest, "pgd_broad_epsilon_selected_objective_gain_over_zero")
            energy = _metric_mean(latest, "pgd_broad_epsilon_epsilon_energy_mean")
            if gain is not None and energy is not None and gain > 0.0 and energy > 0.0:
                facts.append(
                    f"The {row['row_id']} direct-epsilon control retained nonzero selected "
                    "epsilon energy and positive selected objective gain at the final logged "
                    "batch."
                )
    return facts


def render_markdown(payload: Mapping[str, Any]) -> str:
    rows = list(payload["rows"])
    lines = [
        "# Frozen finite-policy audit metadata",
        "",
        "This is a local, saved-artifact audit for ae9f30f. It confirms optimizer "
        "provenance and selected PGD scalar behavior, and it records why exact "
        "live-graph frozen replay is not yet available from the saved cache.",
        "",
        "## Confirmed facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in payload["confirmed_facts"])
    lines.extend(
        [
            "",
            "## Optimizer provenance",
            "",
            "| Row | Status | Mechanism | Active lane | Method | Steps | Step frac | Lambda | "
            "Policy Adam active? |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        optimizer = _mapping(row.get("optimizer"))
        inactive_adam = _mapping(optimizer.get("inactive_policy_adam_metadata"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["row_id"]),
                    str(row["status"]),
                    _md(optimizer.get("active_mechanism")),
                    _md(optimizer.get("active_lane")),
                    _md(optimizer.get("active_method")),
                    _fmt(optimizer.get("active_n_steps")),
                    _fmt(optimizer.get("active_step_size_fraction_of_l2_radius")),
                    _fmt(optimizer.get("lambda")),
                    "yes" if inactive_adam.get("enabled") else "no",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Checkpoint and diagnostic coverage",
            "",
            "| Row | Checkpoints | Latest | Training summary | Diagnostics NPZ | Sentinels |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        checkpoints = _mapping(row.get("checkpoints"))
        summary = _mapping(row.get("training_summary"))
        diagnostics = _mapping(row.get("training_diagnostics"))
        sentinels = _mapping(row.get("sentinels"))
        sentinel_names = ", ".join(file_info["name"] for file_info in sentinels.get("files", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["row_id"]),
                    str(checkpoints.get("count", 0)),
                    _md(checkpoints.get("latest")),
                    "yes" if summary.get("exists") else "no",
                    "yes" if diagnostics.get("exists") else "no",
                    _md(sentinel_names or "none"),
                ]
            )
            + " |"
        )

    lines.extend(render_pgd_table(rows))
    lines.extend(
        [
            "",
            "## Unsupported or blocked in this cache",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["unsupported_or_blocked"])
    lines.extend(
        [
            "",
            "## Stabilization-table caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["stabilization_caveats"])
    lines.extend(
        [
            "",
            "## Next replay instrumentation",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["next_replay_instrumentation"])
    return "\n".join(lines) + "\n"


def render_pgd_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Selected PGD scalar diagnostics",
        "",
        "| Row | Completed batches | Selected gain mean | Selected energy mean | "
        "Radius ratio mean | Cap boundary mean | Final endpoint gap mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        diagnostics = _mapping(row.get("training_diagnostics"))
        if not diagnostics.get("exists"):
            lines.append(f"| {row['row_id']} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        for sample in diagnostics.get("sample_rows", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["row_id"]),
                        str(sample["completed_batches"]),
                        _fmt(
                            _metric_mean(
                                sample, "pgd_broad_epsilon_selected_objective_gain_over_zero"
                            )
                        ),
                        _fmt(_metric_mean(sample, "pgd_broad_epsilon_epsilon_energy_mean")),
                        _fmt(
                            _metric_mean(
                                sample,
                                "pgd_broad_epsilon_epsilon_norm_radius_ratio_mean",
                            )
                        ),
                        _fmt(_metric_mean(sample, "pgd_broad_epsilon_cap_boundary_fraction")),
                        _fmt(
                            _metric_mean(
                                sample,
                                "pgd_broad_epsilon_inner_objective_final_endpoint_gap",
                            )
                        ),
                    ]
                )
                + " |"
            )
    return lines


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_batches(completed_batches: Sequence[int]) -> list[int]:
    if not completed_batches:
        return []
    desired = [500, 1000, 3000, completed_batches[-1]]
    return sorted({batch for batch in desired if batch in completed_batches})


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _metric_mean(container: Mapping[str, Any], metric: str) -> float | None:
    summary = _mapping(container.get(metric))
    if "mean" in summary:
        return float(summary["mean"])
    if "value" in summary:
        return float(summary["value"])
    return None


def _sample_metric_mean(
    samples: Sequence[Mapping[str, Any]],
    *,
    completed_batches: int,
    metric: str,
) -> float | None:
    for sample in samples:
        if sample.get("completed_batches") == completed_batches:
            return _metric_mean(sample, metric)
    return None


def _repo_rel(repo_root: Path, path: Path) -> str:
    return portable_repo_path(path, repo_root=repo_root)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(numeric)
    if numeric == 0.0:
        return "0"
    if abs(numeric) >= 1e4 or abs(numeric) < 1e-3:
        return f"{numeric:.3e}"
    return f"{numeric:.6g}"


def _md(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value).replace("|", "\\|")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()

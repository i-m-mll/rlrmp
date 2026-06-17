"""Local benchmark for GRU post-run evaluation materializers.

The harness times each diagnostic bundle separately so production materializers
can stay independently runnable while shared speedups remain measurable.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import (
    MATERIALIZER_ISSUE_ID,
    materialize_gru_standard_result,
    write_gru_standard_result,
)
from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import (
    materialize_gru_evaluation_diagnostics,
)
from rlrmp.analysis.pipelines.gru_feedback_ablation import (
    evaluate_run_feedback_ablation,
    selected_feedback_ablation_bins_for_bank,
)
from rlrmp.analysis.pipelines.gru_map_error_decomposition import (
    materialize_gru_map_error_decomposition,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    default_cs_perturbation_bank,
    evaluate_run_perturbation_bank,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    materialize_gru_pilot_figures,
    resolve_run_inputs,
)
from rlrmp.analysis.pipelines.gru_worst_case_epsilon_audit import (
    audit_run_worst_case_epsilon,
)
from rlrmp.analysis.pipelines.objective_comparator import (
    materialize_gru_objective_comparator_sidecar,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


DEFAULT_ISSUE = "79d2d8b"
DEFAULT_SOURCE_EXPERIMENT = "020a65b"
DEFAULT_RUN_ID = (
    "target_relative_multitarget_h0_fullqrf_warmcos__"
    "proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
DEFAULT_ROW_FAMILIES = (
    "initial_position_offset",
    "command_input_pulse",
    "process_epsilon_force_state_xy",
    "sensory_feedback_offset",
    "delayed_observation_offset",
)


@dataclass(frozen=True)
class TimedBundle:
    """One bundle timing result."""

    bundle: str
    elapsed_s: float
    status: str
    summary: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle,
            "elapsed_s": self.elapsed_s,
            "status": self.status,
            "summary": dict(self.summary),
        }


def build_parser() -> argparse.ArgumentParser:
    """Return the benchmark CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--issue", default=DEFAULT_ISSUE)
    parser.add_argument("--step-label", default="baseline")
    parser.add_argument("--n-rollout-trials", type=int, default=1)
    parser.add_argument("--perturbation-rows", type=int, default=6)
    parser.add_argument(
        "--perturbation-evaluation-backend",
        choices=("serial",),
        default="serial",
        help=(
            "Backend passed to evaluate_run_perturbation_bank. "
            "Only serial is currently supported; the rejected union-graph attempt is "
            "documented in issue timing notes."
        ),
    )
    parser.add_argument("--feedback-bins", type=int, default=3)
    parser.add_argument("--worst-case-steps", type=int, default=1)
    parser.add_argument("--worst-case-restarts", type=int, default=1)
    parser.add_argument(
        "--worst-case-optimizer-backend",
        choices=("serial", "staged"),
        default="serial",
        help="Backend passed to the worst-case epsilon optimizer.",
    )
    parser.add_argument("--no-write-bulk-arrays", action="store_true")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write a timing JSON payload."""

    args = build_parser().parse_args(argv)
    result = run_benchmark(
        source_experiment=args.source_experiment,
        run_id=args.run_id,
        issue=args.issue,
        step_label=args.step_label,
        n_rollout_trials=args.n_rollout_trials,
        perturbation_rows=args.perturbation_rows,
        perturbation_evaluation_backend=args.perturbation_evaluation_backend,
        feedback_bins=args.feedback_bins,
        worst_case_steps=args.worst_case_steps,
        worst_case_restarts=args.worst_case_restarts,
        worst_case_optimizer_backend=args.worst_case_optimizer_backend,
        write_bulk_arrays=not args.no_write_bulk_arrays,
        output_path=args.output_path,
        scratch_dir=args.scratch_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_benchmark(
    *,
    source_experiment: str = DEFAULT_SOURCE_EXPERIMENT,
    run_id: str = DEFAULT_RUN_ID,
    issue: str = DEFAULT_ISSUE,
    step_label: str = "baseline",
    n_rollout_trials: int = 1,
    perturbation_rows: int = 6,
    perturbation_evaluation_backend: str = "serial",
    feedback_bins: int = 3,
    worst_case_steps: int = 1,
    worst_case_restarts: int = 1,
    worst_case_optimizer_backend: str = "serial",
    write_bulk_arrays: bool = True,
    output_path: Path | None = None,
    scratch_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run the local all-bundle benchmark and return JSON-ready timings."""

    if n_rollout_trials < 1:
        raise ValueError("n_rollout_trials must be at least 1")
    if perturbation_rows < 1:
        raise ValueError("perturbation_rows must be at least 1")
    if feedback_bins < 1:
        raise ValueError("feedback_bins must be at least 1")

    output_path = output_path or (
        repo_root / "results" / issue / "notes" / f"postrun_eval_timing_{step_label}.json"
    )
    scratch_dir = scratch_dir or repo_root / "_artifacts" / issue / "postrun_eval_benchmark"
    step_scratch = scratch_dir / step_label
    notes_scratch = step_scratch / "notes"
    mkdir_p(output_path.parent)
    mkdir_p(notes_scratch)
    mkdir_p(step_scratch)

    runs = resolve_run_inputs(
        experiment=source_experiment,
        run_ids=(run_id,),
        labels=None,
        repo_root=repo_root,
    )
    run = runs[0]
    bank = subset_perturbation_bank(
        default_cs_perturbation_bank(),
        max_rows=perturbation_rows,
    )
    feedback_evaluation_bins = dict(
        list(selected_feedback_ablation_bins_for_bank(bank).items())[:feedback_bins]
    )

    context: dict[str, Any] = {
        "source_experiment": source_experiment,
        "run_id": run_id,
        "n_rollout_trials": int(n_rollout_trials),
        "perturbation_rows": int(len(bank["perturbations"])),
        "perturbation_evaluation_backend": str(perturbation_evaluation_backend),
        "feedback_bins": int(len(feedback_evaluation_bins)),
        "write_bulk_arrays": bool(write_bulk_arrays),
        "worst_case_steps": int(worst_case_steps),
        "worst_case_restarts": int(worst_case_restarts),
        "worst_case_optimizer_backend": str(worst_case_optimizer_backend),
    }
    bundle_results: list[TimedBundle] = []
    total_start = time.perf_counter()

    standard_manifest_path = notes_scratch / "gru_standard_manifest.json"
    standard_note_path = notes_scratch / "gru_standard.md"
    standard_result_holder: dict[str, Any] = {}

    def run_standard() -> Mapping[str, Any]:
        result = materialize_gru_standard_result(
            run_ids=(run_id,),
            experiment=source_experiment,
            materializer_issue_id=MATERIALIZER_ISSUE_ID,
            use_validation_selected_checkpoints=False,
            repo_root=repo_root,
        )
        write_gru_standard_result(
            result,
            note_path=standard_note_path,
            manifest_path=standard_manifest_path,
            repo_root=repo_root,
        )
        standard_result_holder["result"] = result
        return {
            "rows": len(result.get("rows", ())),
            "checkpoint_policy": result.get("checkpoint_policy"),
            "output": _repo_relative(standard_manifest_path, repo_root=repo_root),
        }

    bundle_results.append(_time_bundle("standard_certificate", run_standard))

    def run_evaluation_diagnostics() -> Mapping[str, Any]:
        result = materialize_gru_evaluation_diagnostics(
            experiment=source_experiment,
            run_ids=(run_id,),
            labels=None,
            output_path=notes_scratch / "gru_evaluation_diagnostics.json",
            bulk_dir=step_scratch / "evaluation_diagnostics",
            n_rollout_trials=n_rollout_trials,
            use_validation_selected_checkpoints=True,
            write_bulk_arrays=write_bulk_arrays,
            repo_root=repo_root,
        )
        return _bundle_status_counts(result)

    bundle_results.append(_time_bundle("evaluation_diagnostics", run_evaluation_diagnostics))

    def run_pilot_figures() -> Mapping[str, Any]:
        result = materialize_gru_pilot_figures(
            experiment=source_experiment,
            run_ids=(run_id,),
            labels=None,
            output_dir=step_scratch / "figures",
            n_rollout_trials=n_rollout_trials,
            include_reference=True,
            use_validation_selected_checkpoints=True,
            repo_root=repo_root,
        )
        return {"figure_keys": sorted(result.keys())}

    bundle_results.append(_time_bundle("pilot_figures", run_pilot_figures))

    def run_objective_comparator() -> Mapping[str, Any]:
        result = materialize_gru_objective_comparator_sidecar(
            experiment=source_experiment,
            run_ids=(run_id,),
            labels=None,
            checkpoint_policy="validation_selected_per_replicate",
            use_validation_selected_checkpoints=True,
            checkpoint_manifest=None,
            checkpoint_manifest_path=(
                repo_root / "results" / source_experiment / "notes"
                / "validation_selected_checkpoints.json"
            ),
            standard_manifest_path=standard_manifest_path,
            output_path=notes_scratch / "objective_comparator.json",
            note_path=notes_scratch / "objective_comparator.md",
            repo_root=repo_root,
        )
        return {"status": result.get("status", "materialized"), "keys": sorted(result.keys())}

    bundle_results.append(_time_bundle("objective_comparator", run_objective_comparator))

    def run_map_decomposition() -> Mapping[str, Any]:
        result = materialize_gru_map_error_decomposition(
            standard_manifest_path=standard_manifest_path,
            experiment=source_experiment,
            run_ids=(run_id,),
            use_validation_selected_checkpoints=True,
            top_k=3,
            repo_root=repo_root,
        )
        (notes_scratch / "gru_map_error_decomposition.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"rows": len(result.get("rows", ())), "keys": sorted(result.keys())}

    bundle_results.append(_time_bundle("map_decomposition", run_map_decomposition))

    def run_perturbation_response() -> Mapping[str, Any]:
        result = evaluate_run_perturbation_bank(
            run,
            source_experiment=source_experiment,
            bank=bank,
            n_rollout_trials=n_rollout_trials,
            write_bulk_arrays=write_bulk_arrays,
            bulk_dir=step_scratch / "perturbation_response",
            evaluation_backend=perturbation_evaluation_backend,
            repo_root=repo_root,
        )
        return {
            "status_counts": result.get("status_counts", {}),
            "rows": len(result.get("perturbations", ())),
        }

    bundle_results.append(_time_bundle("perturbation_response", run_perturbation_response))

    def run_feedback_ablation() -> Mapping[str, Any]:
        result = evaluate_run_feedback_ablation(
            run,
            source_experiment=source_experiment,
            n_rollout_trials=n_rollout_trials,
            include_checkpoint_rescore=False,
            bank=bank,
            evaluation_bins=feedback_evaluation_bins,
            repo_root=repo_root,
        )
        return {
            "status_counts": result.get("status_counts", {}),
            "rows": len(result.get("rows", ())),
        }

    bundle_results.append(_time_bundle("feedback_ablation", run_feedback_ablation))

    def run_worst_case_epsilon() -> Mapping[str, Any]:
        result = audit_run_worst_case_epsilon(
            run,
            source_experiment=source_experiment,
            n_rollout_trials=n_rollout_trials,
            n_steps=worst_case_steps,
            n_restarts=worst_case_restarts,
            step_size=None,
            n_random_baselines=1,
            seed=0,
            budget_level_override="moderate",
            budget_scale_override=None,
            bulk_dir=step_scratch / "worst_case_epsilon",
            optimizer_backend=worst_case_optimizer_backend,
            repo_root=repo_root,
        )
        return {"status": result.get("status", "evaluated"), "keys": sorted(result.keys())}

    bundle_results.append(_time_bundle("worst_case_epsilon", run_worst_case_epsilon))

    total_elapsed = time.perf_counter() - total_start
    payload = {
        "schema_version": "rlrmp.postrun_eval_materialization_benchmark.v1",
        "issue": issue,
        "step_label": step_label,
        "captured_at_unix_s": time.time(),
        "environment": _environment(),
        "context": context,
        "bundles": [result.to_json() for result in bundle_results],
        "total_elapsed_s": total_elapsed,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def subset_perturbation_bank(
    bank: Mapping[str, Any],
    *,
    max_rows: int,
    families: Sequence[str] = DEFAULT_ROW_FAMILIES,
) -> dict[str, Any]:
    """Return a small deterministic bank slice spanning key adapter families."""

    selected: list[Mapping[str, Any]] = []
    rows = list(bank.get("perturbations", ()))
    for family in families:
        for row in rows:
            row_family = str(row.get("family", ""))
            if row_family != family and not row_family.startswith(f"{family}_"):
                continue
            if row.get("axis") not in {None, "x"}:
                continue
            if row.get("sign") not in {None, 1}:
                continue
            selected.append(dict(row))
            break
        if len(selected) >= max_rows:
            break
    if len(selected) < max_rows:
        seen = {row["perturbation_id"] for row in selected}
        for row in rows:
            if row["perturbation_id"] in seen:
                continue
            selected.append(dict(row))
            if len(selected) >= max_rows:
                break
    sliced = dict(bank)
    sliced["bank_id"] = f"benchmark_subset_of_{bank.get('bank_id', 'bank')}"
    sliced["perturbations"] = selected[:max_rows]
    sliced["benchmark_subset"] = {
        "max_rows": int(max_rows),
        "families": list(families),
        "selected_ids": [str(row["perturbation_id"]) for row in selected[:max_rows]],
    }
    return sliced


def _time_bundle(bundle: str, fn: Callable[[], Mapping[str, Any]]) -> TimedBundle:
    start = time.perf_counter()
    try:
        summary = dict(fn())
        status = str(summary.get("status", "ok"))
    except Exception as exc:  # pragma: no cover - kept visible in benchmark JSON.
        summary = {"error": type(exc).__name__, "detail": str(exc)}
        status = "error"
    return TimedBundle(
        bundle=bundle,
        elapsed_s=time.perf_counter() - start,
        status=status,
        summary=summary,
    )


def _bundle_status_counts(result: Mapping[str, Any]) -> dict[str, Any]:
    runs = result.get("runs", {})
    if not isinstance(runs, Mapping):
        return {"runs": 0}
    return {
        "runs": len(runs),
        "run_status_counts": {
            str(run_id): dict(run.get("status_counts", {}))
            for run_id, run in runs.items()
            if isinstance(run, Mapping)
        },
    }


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())

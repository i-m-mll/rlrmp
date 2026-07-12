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
import jax.tree as jt

from rlrmp.eval.feedback_ablation import selected_feedback_ablation_bins_for_bank
from rlrmp.paths import portable_repo_path as _repo_relative
from rlrmp.eval.perturbation_bank import (
    default_cs_perturbation_bank,
    evaluate_run_perturbation_bank,
)
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    materialize_gru_pilot_figures,
    resolve_run_inputs,
)
from rlrmp.analysis.pipelines.objective_comparator import (
    materialize_gru_objective_comparator_sidecar,
)
from rlrmp.eval.checkpoint_selection import build_validation_checkpoint_selection_manifest
from rlrmp.paths import REPO_ROOT, mkdir_p


DEFAULT_ISSUE = "79d2d8b"
DEFAULT_SOURCE_EXPERIMENT = "020a65b"
DEFAULT_RUN_ID = (
    "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
DEFAULT_ROW_FAMILIES = (
    "initial_position_offset",
    "command_input_pulse",
    "process_epsilon_force_state_xy",
    "sensory_feedback_offset",
    "delayed_observation_offset",
)
PROCESS_STARTUP_NOT_MEASURED = {
    "status": "not_measured",
    "reason": (
        "Python code inside this benchmark cannot observe interpreter startup "
        "or module import time before the module begins executing."
    ),
}
COMPILE_EXECUTION_SPLIT_NOT_MEASURED = {
    "status": "not_measured",
    "reason": (
        "This harness measures the first materializer call as a cold call. "
        "XLA compile, first execution, internal host transfers, Python work, "
        "and materializer-owned writes are not separable at this boundary."
    ),
}
WARM_REPLAY_DISABLED_REASON = (
    "Warm replay is disabled by default because it reruns every bundle and can "
    "roughly double benchmark cost. Pass --warm-replay to enable a second call "
    "with separate warm-replay scratch paths."
)
MODULE_IMPORT_COMPLETED_AT = time.perf_counter()


@dataclass(frozen=True)
class TimedBundle:
    """One bundle timing result."""

    bundle: str
    total_elapsed_s: float
    cold_call_elapsed_s: float
    cold_ready_block_s: float
    cold_ready_blocked_leaves: int
    cold_ready_block_note: str
    warm_call_elapsed_s: float | None
    warm_ready_block_s: float | None
    warm_ready_blocked_leaves: int | None
    warm_ready_block_note: str | None
    warm_replay_status: str
    warm_replay_note: str
    output_write_elapsed_s: float | None
    output_write_mode: str
    output_write_note: str
    summary_elapsed_s: float
    status: str
    summary: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle,
            "total_elapsed_s": self.total_elapsed_s,
            "cold_call_elapsed_s": self.cold_call_elapsed_s,
            "cold_call_definition": (
                "Elapsed time for the first bundle invocation. This may include "
                "Python setup inside the materializer, XLA compilation, first "
                "execution, internal host transfer/synchronization, and "
                "materializer-owned output writes. It is not pure compile or "
                "calculation time."
            ),
            "xla_compile_execution_split": dict(COMPILE_EXECUTION_SPLIT_NOT_MEASURED),
            "cold_ready_block_s": self.cold_ready_block_s,
            "cold_ready_blocked_leaves": self.cold_ready_blocked_leaves,
            "cold_ready_block_note": self.cold_ready_block_note,
            "warm_call_elapsed_s": self.warm_call_elapsed_s,
            "warm_ready_block_s": self.warm_ready_block_s,
            "warm_ready_blocked_leaves": self.warm_ready_blocked_leaves,
            "warm_ready_block_note": self.warm_ready_block_note,
            "warm_replay_status": self.warm_replay_status,
            "warm_replay_note": self.warm_replay_note,
            "output_write_elapsed_s": self.output_write_elapsed_s,
            "output_write_mode": self.output_write_mode,
            "output_write_note": self.output_write_note,
            "summary_elapsed_s": self.summary_elapsed_s,
            "status": self.status,
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class ReadyBlockTiming:
    """Post-call JAX readiness timing for a bundle result."""

    elapsed_s: float
    blocked_leaves: int
    note: str


@dataclass(frozen=True)
class OutputWriteTiming:
    """Timing and classification for benchmark-owned output writes."""

    elapsed_s: float | None
    mode: str
    note: str


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
    parser.add_argument(
        "--warm-replay",
        action="store_true",
        help=(
            "Run each bundle a second time with separate warm-replay scratch "
            "paths to measure a warm call. Disabled by default to avoid "
            "unexpectedly doubling benchmark cost."
        ),
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
        warm_replay=args.warm_replay,
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
    warm_replay: bool = False,
    write_bulk_arrays: bool = True,
    output_path: Path | None = None,
    scratch_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run the local all-bundle benchmark and return JSON-ready timings."""

    benchmark_entry_elapsed_s = time.perf_counter() - MODULE_IMPORT_COMPLETED_AT
    total_start = time.perf_counter()
    setup_start = time.perf_counter()

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
    warm_step_scratch = step_scratch / "_warm_replay"
    warm_notes_scratch = warm_step_scratch / "notes"
    mkdir_p(output_path.parent)
    mkdir_p(notes_scratch)
    mkdir_p(step_scratch)
    if warm_replay:
        mkdir_p(warm_notes_scratch)
        mkdir_p(warm_step_scratch)

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
    setup_elapsed_s = time.perf_counter() - setup_start

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
        "warm_replay": {
            "enabled": bool(warm_replay),
            "default_enabled": False,
            "reason_if_disabled": None if warm_replay else WARM_REPLAY_DISABLED_REASON,
            "scratch_path_policy": (
                "warm replay writes to a separate _warm_replay scratch tree"
                if warm_replay
                else "not_applicable"
            ),
        },
    }
    bundle_results: list[TimedBundle] = []

    standard_manifest_path = notes_scratch / "gru_standard_manifest.json"
    checkpoint_manifest = build_validation_checkpoint_selection_manifest(
        experiment=source_experiment,
        run_ids=(run_id,),
        repo_root=repo_root,
    )

    def run_pilot_figures() -> Mapping[str, Any]:
        return materialize_gru_pilot_figures(
            experiment=source_experiment,
            run_ids=(run_id,),
            labels=None,
            output_dir=step_scratch / "figures",
            n_rollout_trials=n_rollout_trials,
            include_reference=True,
            use_validation_selected_checkpoints=True,
            repo_root=repo_root,
        )

    bundle_results.append(
        _time_bundle(
            "pilot_figures",
            run_pilot_figures,
            summarize=lambda result: {"figure_keys": sorted(result.keys())},
            output_write_mode="included_in_cold_call_elapsed_s",
            output_write_note=(
                "materialize_gru_pilot_figures writes figure outputs inside the materializer call."
            ),
            warm_replay=warm_replay,
            warm_fn=lambda: materialize_gru_pilot_figures(
                experiment=source_experiment,
                run_ids=(run_id,),
                labels=None,
                output_dir=warm_step_scratch / "figures",
                n_rollout_trials=n_rollout_trials,
                include_reference=True,
                use_validation_selected_checkpoints=True,
                repo_root=repo_root,
            ),
        )
    )

    def run_objective_comparator() -> Mapping[str, Any]:
        return materialize_gru_objective_comparator_sidecar(
            experiment=source_experiment,
            run_ids=(run_id,),
            labels=None,
            checkpoint_policy="validation_selected_per_replicate",
            use_validation_selected_checkpoints=True,
            checkpoint_manifest=checkpoint_manifest,
            checkpoint_manifest_path=None,
            standard_manifest_path=standard_manifest_path,
            output_path=notes_scratch / "objective_comparator.json",
            note_path=notes_scratch / "objective_comparator.md",
            repo_root=repo_root,
        )

    bundle_results.append(
        _time_bundle(
            "objective_comparator",
            run_objective_comparator,
            summarize=lambda result: {
                "status": result.get("status", "materialized"),
                "keys": sorted(result.keys()),
            },
            output_write_mode="included_in_cold_call_elapsed_s",
            output_write_note=(
                "materialize_gru_objective_comparator_sidecar writes its JSON/note "
                "outputs inside the materializer call."
            ),
            warm_replay=warm_replay,
            warm_fn=lambda: materialize_gru_objective_comparator_sidecar(
                experiment=source_experiment,
                run_ids=(run_id,),
                labels=None,
                checkpoint_policy="validation_selected_per_replicate",
                use_validation_selected_checkpoints=True,
                checkpoint_manifest=checkpoint_manifest,
                checkpoint_manifest_path=None,
                standard_manifest_path=standard_manifest_path,
                output_path=warm_notes_scratch / "objective_comparator.json",
                note_path=warm_notes_scratch / "objective_comparator.md",
                repo_root=repo_root,
            ),
        )
    )

    def run_perturbation_response() -> Mapping[str, Any]:
        return evaluate_run_perturbation_bank(
            run,
            source_experiment=source_experiment,
            bank=bank,
            n_rollout_trials=n_rollout_trials,
            evaluation_backend=perturbation_evaluation_backend,
            repo_root=repo_root,
        )

    bundle_results.append(
        _time_bundle(
            "perturbation_response",
            run_perturbation_response,
            summarize=lambda result: {
                "status_counts": result.get("status_counts", {}),
                "rows": len(result.get("perturbations", ())),
            },
            output_write_mode="none",
            output_write_note="Perturbation evaluation emits no direct durable outputs.",
            warm_replay=warm_replay,
            warm_fn=lambda: evaluate_run_perturbation_bank(
                run,
                source_experiment=source_experiment,
                bank=bank,
                n_rollout_trials=n_rollout_trials,
                evaluation_backend=perturbation_evaluation_backend,
                repo_root=repo_root,
            ),
        )
    )

    payload = {
        "schema_version": "rlrmp.postrun_eval_materialization_benchmark.v2",
        "issue": issue,
        "step_label": step_label,
        "captured_at_unix_s": time.time(),
        "environment": _environment(),
        "phase_definitions": _phase_definitions(),
        "process_timing": {
            "process_startup_elapsed_s": dict(PROCESS_STARTUP_NOT_MEASURED),
            "module_import_to_benchmark_entry_elapsed_s": benchmark_entry_elapsed_s,
        },
        "setup_timing": {
            "elapsed_s": setup_elapsed_s,
            "definition": (
                "Path creation, run context resolution, perturbation-bank slicing, "
                "and feedback-bin selection before the first bundle starts."
            ),
        },
        "context": context,
        "bundles": [result.to_json() for result in bundle_results],
    }
    serialization_start = time.perf_counter()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(serialized, encoding="utf-8")
    report_serialization_elapsed_s = time.perf_counter() - serialization_start
    payload["report_serialization_timing"] = {
        "elapsed_s": report_serialization_elapsed_s,
        "output": _repo_relative(output_path, repo_root=repo_root),
        "definition": "JSON serialization plus writing the benchmark timing report.",
    }
    payload["total_elapsed_s"] = time.perf_counter() - total_start
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


def _time_bundle(
    bundle: str,
    fn: Callable[[], Mapping[str, Any]],
    *,
    summarize: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    output_writer: Callable[[Mapping[str, Any]], None] | None = None,
    output_write_mode: str | None = None,
    output_write_note: str | None = None,
    warm_replay: bool = False,
    warm_fn: Callable[[], Mapping[str, Any]] | None = None,
    warm_output_writer: Callable[[Mapping[str, Any]], None] | None = None,
) -> TimedBundle:
    start = time.perf_counter()
    cold_call_elapsed_s = 0.0
    ready_block = ReadyBlockTiming(
        elapsed_s=0.0,
        blocked_leaves=0,
        note="bundle raised before result readiness could be checked",
    )
    output_write = OutputWriteTiming(
        elapsed_s=None,
        mode=output_write_mode or "not_applicable",
        note=output_write_note or "no benchmark-owned output write for this bundle",
    )
    summary_elapsed_s = 0.0
    warm_call_elapsed_s: float | None = None
    warm_ready_block_s: float | None = None
    warm_ready_blocked_leaves: int | None = None
    warm_ready_block_note: str | None = None
    warm_replay_status = "disabled"
    warm_replay_note = WARM_REPLAY_DISABLED_REASON
    try:
        call_start = time.perf_counter()
        result = fn()
        cold_call_elapsed_s = time.perf_counter() - call_start
        ready_block = _block_until_ready(result)

        if output_writer is not None:
            output_start = time.perf_counter()
            output_writer(result)
            output_write = OutputWriteTiming(
                elapsed_s=time.perf_counter() - output_start,
                mode="separate_measured",
                note=output_write_note or "benchmark-owned output write measured separately",
            )

        summary_start = time.perf_counter()
        summary = dict(summarize(result) if summarize is not None else result)
        summary_elapsed_s = time.perf_counter() - summary_start
        status = str(summary.get("status", "ok"))

        if warm_replay:
            warm_replay_status = "ok"
            warm_replay_note = (
                "Warm replay used a second bundle invocation. It measures a warm "
                "call boundary, not pure calculation time."
            )
            warm_call = warm_fn or fn
            warm_call_start = time.perf_counter()
            warm_result = warm_call()
            warm_call_elapsed_s = time.perf_counter() - warm_call_start
            warm_ready_block = _block_until_ready(warm_result)
            warm_ready_block_s = warm_ready_block.elapsed_s
            warm_ready_blocked_leaves = warm_ready_block.blocked_leaves
            warm_ready_block_note = warm_ready_block.note
            warm_writer = warm_output_writer if warm_output_writer is not None else output_writer
            if warm_writer is not None:
                warm_writer(warm_result)
    except Exception as exc:  # pragma: no cover - kept visible in benchmark JSON.
        summary = {"error": type(exc).__name__, "detail": str(exc)}
        status = "error"
    return TimedBundle(
        bundle=bundle,
        total_elapsed_s=time.perf_counter() - start,
        cold_call_elapsed_s=cold_call_elapsed_s,
        cold_ready_block_s=ready_block.elapsed_s,
        cold_ready_blocked_leaves=ready_block.blocked_leaves,
        cold_ready_block_note=ready_block.note,
        warm_call_elapsed_s=warm_call_elapsed_s,
        warm_ready_block_s=warm_ready_block_s,
        warm_ready_blocked_leaves=warm_ready_blocked_leaves,
        warm_ready_block_note=warm_ready_block_note,
        warm_replay_status=warm_replay_status,
        warm_replay_note=warm_replay_note,
        output_write_elapsed_s=output_write.elapsed_s,
        output_write_mode=output_write.mode,
        output_write_note=output_write.note,
        summary_elapsed_s=summary_elapsed_s,
        status=status,
        summary=summary,
    )


def _block_until_ready(value: Any) -> ReadyBlockTiming:
    """Block on JAX leaves in ``value`` and report what was blocked."""

    leaves = [
        leaf for leaf in jt.leaves(value) if callable(getattr(leaf, "block_until_ready", None))
    ]
    if not leaves:
        return ReadyBlockTiming(
            elapsed_s=0.0,
            blocked_leaves=0,
            note="no JAX leaves with block_until_ready",
        )

    start = time.perf_counter()
    for leaf in leaves:
        leaf.block_until_ready()
    return ReadyBlockTiming(
        elapsed_s=time.perf_counter() - start,
        blocked_leaves=len(leaves),
        note="blocked JAX leaves with block_until_ready",
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


def _phase_definitions() -> dict[str, str]:
    return {
        "process_startup_elapsed_s": (
            "Not measured from inside Python. This harness begins after the "
            "interpreter and module imports have already started."
        ),
        "setup_timing.elapsed_s": (
            "Benchmark harness setup before bundles: output path creation, run "
            "resolution, bank slicing, and feedback-bin selection."
        ),
        "bundle.cold_call_elapsed_s": (
            "First invocation of the bundle. This is the earliest honest boundary "
            "available here and may include XLA compile, first execution, internal "
            "sync/host transfer, Python work, and materializer-owned writes."
        ),
        "bundle.cold_ready_block_s": (
            "Explicit block_until_ready on JAX leaves returned by the cold call. "
            "A zero value with no leaves means the bundle returned host summaries, "
            "so readiness already happened inside cold_call_elapsed_s."
        ),
        "bundle.warm_call_elapsed_s": (
            "Second invocation after the cold call, only when --warm-replay is "
            "enabled. This is a warm call boundary, not pure calculation time."
        ),
        "bundle.output_write_elapsed_s": (
            "Measured only when the benchmark can separate a benchmark-owned write "
            "from the materializer call; otherwise the bundle records that writes "
            "are included in cold_call_elapsed_s or not applicable."
        ),
        "bundle.summary_elapsed_s": (
            "Small report-summary extraction after cold readiness and any separated "
            "benchmark-owned output write."
        ),
        "report_serialization_timing.elapsed_s": (
            "Serialization and writing of the benchmark timing JSON report."
        ),
    }


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


if __name__ == "__main__":
    raise SystemExit(main())

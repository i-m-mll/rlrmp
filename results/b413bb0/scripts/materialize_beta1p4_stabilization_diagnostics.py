"""Materialize beta 1.4 stabilization-task endpoint diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.eval.robustness_diagnostics import (
    evaluate_stabilization_row as canonical_evaluate_stabilization_row,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE = "b413bb0"
TAG = "beta1p4_stabilization_diagnostics"
MARKER = "beta1p4_stabilization_diagnostics"
SCHEMA_VERSION = "rlrmp.b413bb0.beta1p4_stabilization_diagnostics.v1"
NOTES_DIR = REPO_ROOT / "results" / ISSUE / "notes"
BULK_DIR = REPO_ROOT / "_artifacts" / ISSUE / "stabilization_diagnostics" / TAG
DETAIL_JSON = BULK_DIR / "per_probe_detail.json"
SUMMARY_JSON = NOTES_DIR / f"{TAG}.json"
SUMMARY_MD = NOTES_DIR / f"{TAG}.md"


@dataclass(frozen=True)
class StabilizationRowSource:
    """One trained row to evaluate on endpoint stabilization probes."""

    row_key: str
    source_experiment: str
    run_id: str
    training_key: str
    training_condition: str
    physical_level: str = "moderate"


DEFAULT_ROWS: tuple[StabilizationRowSource, ...] = (
    StabilizationRowSource(
        row_key="baseline_no_pgd_h0_const_band16",
        source_experiment="33b0dcb",
        run_id="h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64",
        training_key="baseline_no_pgd_h0_const_band16",
        training_condition="no-PGD H0 6D open-loop moderate const_band16 baseline",
    ),
    StabilizationRowSource(
        row_key="direct_epsilon",
        source_experiment=ISSUE,
        run_id="direct_epsilon",
        training_key="beta1p4_direct_epsilon",
        training_condition="beta 1.4 direct-epsilon PGD",
    ),
    StabilizationRowSource(
        row_key="linear_no_bias",
        source_experiment=ISSUE,
        run_id="linear_no_bias",
        training_key="beta1p4_linear_no_bias",
        training_condition="beta 1.4 finite linear no-bias adversary",
    ),
    StabilizationRowSource(
        row_key="affine",
        source_experiment=ISSUE,
        run_id="affine",
        training_key="beta1p4_affine",
        training_condition="beta 1.4 finite affine adversary",
    ),
)


def main() -> None:
    """CLI entry point."""

    summary = materialize_stabilization_diagnostics()
    print(
        {
            "summary_json": repo_rel(SUMMARY_JSON),
            "summary_markdown": repo_rel(SUMMARY_MD),
            "detail_json": repo_rel(DETAIL_JSON),
            "rows": [row["row"] for row in summary["rows"]],
        }
    )


def materialize_stabilization_diagnostics(
    rows: Sequence[StabilizationRowSource] = DEFAULT_ROWS,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate endpoint stabilization probes and write b413-local outputs."""

    mkdir_p(NOTES_DIR)
    mkdir_p(BULK_DIR)
    row_keys = tuple(row.row_key for row in rows)
    if DETAIL_JSON.exists() and SUMMARY_JSON.exists() and not force:
        summary = load_json(SUMMARY_JSON)
        if tuple(summary.get("row_order", ())) == row_keys:
            return summary

    helper = load_c92_stabilization_helper()
    detail_rows: dict[str, Any] = {}
    summary_rows = []
    for row in rows:
        helper.ISSUE = row.source_experiment
        result = evaluate_row_allowing_missing_families(
            helper,
            helper.RowSpec(row.run_id, row.training_key, row.physical_level),
            repo_root=Path(REPO_ROOT).resolve(),
        )
        per_probe_detail = result.pop("per_probe_detail")
        result = {
            "row": row.row_key,
            "source_experiment": row.source_experiment,
            "source_run_id": row.run_id,
            "training_condition": row.training_condition,
            **result,
        }
        detail_rows[row.row_key] = {
            "source_experiment": row.source_experiment,
            "source_run_id": row.run_id,
            "per_probe_detail": per_probe_detail,
        }
        summary_rows.append(result)

    detail = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "detail_role": "stabilization endpoint per-probe scalar and trajectory diagnostics",
        "probe_contract": helper.probe_contract(),
        "row_order": list(row_keys),
        "rows": detail_rows,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "scope": (
            "beta 1.4 b413 rows plus the 33b0dcb H0 no-PGD const_band16 baseline "
            "evaluated on stabilization-task endpoint probes"
        ),
        "row_order": list(row_keys),
        "baseline_row": "baseline_no_pgd_h0_const_band16",
        "probe_contract": helper.probe_contract(),
        "rows": summary_rows,
        "comparisons_vs_baseline": comparisons_vs_baseline(summary_rows),
        "outputs": {
            "summary_json": repo_rel(SUMMARY_JSON),
            "summary_markdown": repo_rel(SUMMARY_MD),
            "detail_json": repo_rel(DETAIL_JSON),
        },
    }
    write_compact_json(DETAIL_JSON, detail)
    write_compact_json(SUMMARY_JSON, summary)
    update_marked_section(SUMMARY_MD, MARKER, render_markdown(summary))
    return summary


def load_c92_stabilization_helper() -> Any:
    """Load the existing c92 stabilization probe implementation by file path."""

    helper_path = (
        REPO_ROOT
        / "results"
        / "c92ebd8"
        / "scripts"
        / "materialize_pgd_1p05_stabilization_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "c92_pgd_1p05_stabilization_diagnostics",
        helper_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load stabilization helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_row_allowing_missing_families(
    helper: Any,
    row_spec: Any,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate one row while preserving unsupported stabilization probes as missing."""

    return canonical_evaluate_stabilization_row(
        row_spec,
        repo_root=repo_root,
        hooks=helper,
        source_experiment=helper.ISSUE,
        row_metadata=lambda row: {
            "run_id": row.run_id,
            "training": row.training,
            "physical_level": row.physical_level,
        },
        allowed_missing_families=("process_epsilon_force_state_xy",),
    )


def comparisons_vs_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return numeric deltas and ratios for each beta row against baseline."""

    by_row = {str(row["row"]): row for row in rows}
    baseline = by_row["baseline_no_pgd_h0_const_band16"]
    fields = (
        "feedback_auc_mm_s",
        "mechanical_auc_mm_s",
        "command_input_auc_mm_s",
        "process_force_auc_mm_s",
        "feedback_peak_mm",
        "mechanical_peak_mm",
    )
    comparisons: dict[str, Any] = {}
    for row_key, row in by_row.items():
        if row_key == "baseline_no_pgd_h0_const_band16":
            continue
        comparisons[row_key] = {
            f"{field}_delta": maybe_delta(row.get(field), baseline.get(field))
            for field in fields
        } | {
            f"{field}_ratio": maybe_ratio(row.get(field), baseline.get(field))
            for field in fields
        }
    return comparisons


def ratio(value: float, reference: float) -> float | None:
    """Return value/reference with a zero guard."""

    if abs(reference) < 1e-12:
        return None
    return float(value / reference)


def maybe_delta(value: Any, reference: Any) -> float | None:
    """Return value-reference when both values are numeric."""

    if not isinstance(value, int | float) or not isinstance(reference, int | float):
        return None
    return float(value) - float(reference)


def maybe_ratio(value: Any, reference: Any) -> float | None:
    """Return value/reference when both values are numeric."""

    if not isinstance(value, int | float) or not isinstance(reference, int | float):
        return None
    return ratio(float(value), float(reference))


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact table note."""

    lines = [
        "# Beta 1.4 Stabilization Diagnostics",
        "",
        "Endpoint stabilization probes reuse the c92 probe contract. AUC values are "
        "mean signed-direction-aligned absolute hand-position displacement after "
        "probe onset in `mm*s`.",
        "",
        "| Row | Source | Training condition | Feedback AUC | Mechanical AUC | "
        "Command AUC | Process-force AUC | Feedback peak | Mechanical peak |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            f"`{row['row']}` | `{row['source_experiment']}` | "
            f"{row['training_condition']} | "
            f"{row['feedback_auc_mm_s']:.4g} | "
            f"{row['mechanical_auc_mm_s']:.4g} | "
            f"{row['command_input_auc_mm_s']:.4g} | "
            f"{fmt(row['process_force_auc_mm_s'])} | "
            f"{row['feedback_peak_mm']:.4g} | "
            f"{row['mechanical_peak_mm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparisons",
            "",
            "Negative AUC deltas mean lower endpoint displacement than the no-PGD H0 "
            "const_band16 baseline.",
            "",
            "| Row | Feedback AUC delta | Mechanical AUC delta | Command AUC delta | "
            "Process-force AUC delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row_key, delta in summary["comparisons_vs_baseline"].items():
        lines.append(
            "| "
            f"`{row_key}` | {fmt(delta['feedback_auc_mm_s_delta'])} | "
            f"{fmt(delta['mechanical_auc_mm_s_delta'])} | "
            f"{fmt(delta['command_input_auc_mm_s_delta'])} | "
            f"{fmt(delta['process_force_auc_mm_s_delta'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    """Format scalar table values."""

    if value is None:
        return "not available"
    if isinstance(value, int | float):
        return f"{float(value):.4g}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    import json

    return json.loads(path.read_text(encoding="utf-8"))


def repo_rel(path: Path) -> str:
    """Return a repo-relative path string."""

    repo_root = Path(REPO_ROOT)
    path = Path(path)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.relative_to(repo_root))


if __name__ == "__main__":
    main()

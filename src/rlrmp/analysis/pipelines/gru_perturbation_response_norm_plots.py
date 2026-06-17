"""Plot perturbation-response norm curves from materialized GRU bulk arrays."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative, write_regeneration_spec
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    _build_extlqg_comparator_context,
    _simulate_extlqg_perturbed,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


SCHEMA_VERSION = "rlrmp.gru_perturbation_response_norm_plots.v1"
DEFAULT_ISSUE = "b8aa38e"
DEFAULT_SELECTOR = "overnight_robust_proprio_validation_selected_corrected"
DEFAULT_SOURCE_MANIFEST = (
    "results/b8aa38e/notes/"
    "gru_perturbation_response_overnight_robust_proprio_validation_selected_"
    "corrected_manifest.json"
)
DEFAULT_CORRECTED_BULK_ROOT = (
    "_artifacts/b8aa38e/perturbation_response/"
    "gru_overnight_robust_proprio_validation_selected_corrected"
)
DEFAULT_RESULTS_DIR = "results/b8aa38e/figures/perturbation_response_norms_proprio"
DEFAULT_ASSET_DIR = "_artifacts/b8aa38e/figures/perturbation_response_norms_proprio/_assets"
DEFAULT_NOTE_PATH = "results/b8aa38e/notes/gru_perturbation_response_norm_plots_proprio.md"
DEFAULT_MANIFEST_PATH = (
    "results/b8aa38e/notes/gru_perturbation_response_norm_plots_proprio_manifest.json"
)
DEFAULT_REGENERATION_SPEC_PATH = (
    "results/b8aa38e/figures/perturbation_response_norms_proprio/regeneration_spec.json"
)

METRICS = ("delta_position", "delta_action")
STAT_COLUMNS = ("mean_norm", "max_norm")
LR_ORDER = ("lr1e-3", "lr3e-3")
TRAINING_LEVEL_ORDER = (
    "none",
    "small",
    "moderate",
    "stress",
    "sisu_raw_strong_gamma_1p05",
    "sisu_effective_020a65b_pgd",
)
SEVERITY_ORDER = ("small", "moderate", "stress")
TIMING_ORDER = ("initial_condition", "early", "mid", "late")
SUPPORTED_EXTLQG_CHANNELS = {
    "command_input",
    "initial_state",
    "process_epsilon",
    "sensory_feedback",
    "delayed_observation",
}

SEVERITY_COLORS = {
    "small": "#2563eb",
    "moderate": "#d97706",
    "stress": "#dc2626",
}
TIMING_COLORS = {
    "initial_condition": "#525252",
    "early": "#0891b2",
    "mid": "#7c3aed",
    "late": "#16a34a",
}
TRAINING_WIDTHS = {
    "none": 1.4,
    "small": 2.2,
    "moderate": 3.0,
    "stress": 3.8,
    "sisu_raw_strong_gamma_1p05": 2.2,
    "sisu_effective_020a65b_pgd": 3.2,
}
EXTLQG_WIDTH = 2.0


@dataclass(frozen=True)
class RunDescriptor:
    """Compact run metadata needed for grouping response curves."""

    run_id: str
    label: str
    learning_rate: str
    training_level: str
    dt_s: float
    n_time_steps: int


@dataclass(frozen=True)
class CurveStats:
    """Replicate-aggregated curve statistics."""

    time_s: np.ndarray
    mean: np.ndarray
    sem: np.ndarray | None
    per_replicate: np.ndarray
    n_replicates: int
    n_samples: int
    n_component_rows: int
    n_rollouts_per_replicate: int


@dataclass(frozen=True)
class DeterministicCurve:
    """Band-free deterministic comparator curve."""

    time_s: np.ndarray
    value: np.ndarray
    status: str
    reason: str | None = None
    n_component_rows: int = 0


@dataclass(frozen=True)
class PlotInputs:
    """Normalized manifest inputs for plotting."""

    manifest: dict[str, Any]
    runs: dict[str, RunDescriptor]
    rows: dict[str, list[dict[str, Any]]]


def materialize_response_norm_plots(
    *,
    source_manifest_path: Path | str = DEFAULT_SOURCE_MANIFEST,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    asset_dir: Path | str = DEFAULT_ASSET_DIR,
    note_path: Path | str = DEFAULT_NOTE_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    regeneration_spec_path: Path | str | None = DEFAULT_REGENERATION_SPEC_PATH,
    repo_root: Path = REPO_ROOT,
    reconstruct_extlqg: bool = True,
    run_id_contains: Sequence[str] = (),
) -> dict[str, Any]:
    """Write response-norm HTML figures plus tracked reproducibility metadata."""

    repo_root = Path(repo_root)
    source_manifest = _resolve_repo_path(source_manifest_path, repo_root=repo_root)
    results_path = mkdir_p(_resolve_repo_path(results_dir, repo_root=repo_root))
    assets_path = mkdir_p(_resolve_repo_path(asset_dir, repo_root=repo_root))
    note = _resolve_repo_path(note_path, repo_root=repo_root)
    manifest_output = _resolve_repo_path(manifest_path, repo_root=repo_root)
    regeneration_spec_output = (
        None
        if regeneration_spec_path is None
        else _resolve_repo_path(regeneration_spec_path, repo_root=repo_root)
    )
    inputs = load_plot_inputs(
        source_manifest,
        repo_root=repo_root,
        run_id_contains=tuple(run_id_contains),
    )
    ext_cache = ExtlqgCurveCache(enabled=reconstruct_extlqg)
    learning_rates = _available_learning_rates(inputs)
    training_levels = _available_training_levels(inputs)

    figure_records: list[dict[str, Any]] = []
    ext_status_counter: Counter[str] = Counter()
    for metric in METRICS:
        figure_records.extend(
            _write_class_a_figures(
                inputs,
                metric=metric,
                asset_dir=assets_path,
                repo_root=repo_root,
                ext_cache=ext_cache,
                ext_status_counter=ext_status_counter,
                learning_rates=learning_rates,
                training_levels=training_levels,
            )
        )
        figure_records.extend(
            _write_class_b_figures(
                inputs,
                metric=metric,
                asset_dir=assets_path,
                repo_root=repo_root,
                ext_cache=ext_cache,
                ext_status_counter=ext_status_counter,
                learning_rates=learning_rates,
                training_levels=training_levels,
            )
        )

    spec = {
        "schema_version": SCHEMA_VERSION,
        "issue": DEFAULT_ISSUE,
        "selector": DEFAULT_SELECTOR,
        "source_manifest_path": _repo_relative(source_manifest, repo_root=repo_root),
        "bulk_roots": {
            "corrected": DEFAULT_CORRECTED_BULK_ROOT,
        },
        "aggregation_method": {
            "metrics": list(METRICS),
            "alignment": (
                "Responses are sign-equalized and represented in the current "
                "target-relative xy basis before norms are taken. For the current "
                "+x reach screen this basis is numerically identical to world xy; "
                "sign equalization is still applied before norm reduction so future "
                "non-+x rows cannot pool raw opposite directions."
            ),
            "xy_reduction": "Euclidean norm over aligned/equalized xy axis at each time point.",
            "mean_norm": (
                "Pool aligned perturbation rows, replicates, and eval rollouts at each "
                "time point; plot mean +/- SEM over pooled replicate x eval-seed samples."
            ),
            "max_norm": (
                "Plot the unbanded max over pooled aligned perturbation rows, replicates, "
                "and eval rollouts at each time point. This is an extreme-response "
                "sidecar, not an uncertainty estimate."
            ),
            "sem": "mean-norm panels only; sample SD over pooled samples divided by sqrt(n).",
            "timing_normalization": "early_visible/mid_visible/late_visible -> early/mid/late.",
        },
        "plot_classes": {
            "class_a": (
                "One figure per metric x perturbation family x timing bin; rows are "
                "mean norm and max norm, columns are available learning rates; traces "
                "are eval severity x perturbation-training level."
            ),
            "class_b": (
                "One figure per metric x perturbation family x eval severity; rows are "
                "mean norm and max norm, columns are available learning rates; traces "
                "are timing bin x perturbation-training level."
            ),
        },
        "extlqg_trace_policy": {
            "status": "reconstructed" if reconstruct_extlqg else "disabled",
            "supported_channels": sorted(SUPPORTED_EXTLQG_CHANNELS),
            "unsupported_channels": ["target_stream"],
            "band": "none; deterministic comparator curve",
        },
        "known_caveats": {
            "graph_adapter_pairing": (
                "Graph-adapter perturbation diagnostics must compare perturbed rows "
                "against a base row evaluated on the same adapter-augmented graph with "
                "a zero payload. This prevents pre-window differences caused by graph "
                "topology/key-stream changes rather than by the declared perturbation."
            ),
            "initial_state_extlqg_information": (
                "The extLQG initial-state comparator perturbs the plant initial state "
                "while keeping the estimator/controller initial state nominal, matching "
                "the GRU delayed-visibility contract for these initial-state rows."
            ),
        },
    }
    manifest = {
        **spec,
        "figure_count": len(figure_records),
        "figures": figure_records,
        "extlqg_trace_status": dict(sorted(ext_status_counter.items())),
        "runs": {run_id: run.__dict__ for run_id, run in inputs.runs.items()},
        "asset_dir": _repo_relative(assets_path, repo_root=repo_root),
        "filters": {
            "run_id_contains": list(run_id_contains),
        },
    }
    if regeneration_spec_output is not None:
        manifest["regeneration_spec_path"] = _repo_relative(
            regeneration_spec_output,
            repo_root=repo_root,
        )
    spec_path = results_path / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        render_response_norm_note(
            manifest,
            spec_path=spec_path,
            manifest_path=manifest_output,
            repo_root=repo_root,
        )
    )
    if regeneration_spec_output is not None:
        _write_response_norm_regeneration_spec(
            spec_path=regeneration_spec_output,
            source_manifest=source_manifest,
            results_dir=results_path,
            asset_dir=assets_path,
            note_path=note,
            manifest_path=manifest_output,
            figure_spec_path=spec_path,
            reconstruct_extlqg=reconstruct_extlqg,
            run_id_contains=tuple(run_id_contains),
            repo_root=repo_root,
        )
    return manifest


def load_plot_inputs(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    run_id_contains: Sequence[str] = (),
) -> PlotInputs:
    """Load and normalize a perturbation-response manifest for plotting."""

    manifest = json.loads(Path(manifest_path).read_text())
    detail_manifest = _load_bulk_detail_manifest(manifest, repo_root=repo_root)
    source_payload = detail_manifest if detail_manifest is not None else manifest
    runs: dict[str, RunDescriptor] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for run_id, run_payload in source_payload["runs"].items():
        if run_id_contains and not any(token in run_id for token in run_id_contains):
            continue
        label = str(run_payload["label"])
        runs[run_id] = RunDescriptor(
            run_id=run_id,
            label=label,
            learning_rate=_parse_learning_rate(label, run_id=run_id),
            training_level=_parse_training_level(label, run_id=run_id),
            dt_s=float(run_payload["dt_s"]),
            n_time_steps=int(run_payload["n_time_steps"]),
        )
        normalized_rows = []
        bulk_files = dict(run_payload.get("bulk_files", {}))
        for row in run_payload.get("perturbations", ()):
            if row.get("status") != "evaluated":
                continue
            severity = _severity(row)
            timing_bin = _normalized_timing_bin(row.get("timing_bin"))
            if severity not in SEVERITY_ORDER or timing_bin == "not_applicable":
                continue
            row_copy = dict(row)
            row_copy["_severity"] = severity
            row_copy["_timing_bin_normalized"] = timing_bin
            row_copy["_bulk_path"] = _bulk_path_for_row(row_copy, bulk_files)
            if row_copy["_bulk_path"] is None:
                continue
            normalized_rows.append(row_copy)
        rows[run_id] = normalized_rows
    if not runs:
        raise ValueError("no runs matched the requested response-norm plot filters")
    return PlotInputs(manifest=manifest, runs=runs, rows=rows)


def aggregate_response_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: Literal["delta_position", "delta_action"],
    dt_s: float,
    repo_root: Path = REPO_ROOT,
    stat: Literal["mean_norm", "max_norm"],
) -> CurveStats:
    """Aggregate materialized response arrays into replicate-level norm curves."""

    if stat not in STAT_COLUMNS:
        raise ValueError(f"unsupported stat {stat!r}")
    per_row_norms = []
    n_rollouts: int | None = None
    n_time: int | None = None
    for row in rows:
        bulk_path = row.get("_bulk_path") or _bulk_path_for_row(row, {})
        if bulk_path is None:
            raise ValueError(f"row {row.get('perturbation_id')} has no bulk array path")
        with np.load(_resolve_repo_path(str(bulk_path), repo_root=repo_root)) as arrays:
            values = np.asarray(arrays[metric], dtype=np.float64)
        if values.ndim != 4 or values.shape[-1] != 2:
            raise ValueError(
                f"{metric} for {row.get('perturbation_id')} must have shape "
                f"(replicate, rollout, time, xy); got {values.shape}"
            )
        aligned = align_and_equalize_response(values, row)
        norms = np.linalg.norm(aligned, axis=-1)
        n_rollouts = norms.shape[1] if n_rollouts is None else n_rollouts
        n_time = norms.shape[2] if n_time is None else n_time
        if norms.shape[1] != n_rollouts or norms.shape[2] != n_time:
            raise ValueError("all grouped response arrays must share rollout and time shapes")
        per_row_norms.append(norms)
    if not per_row_norms:
        raise ValueError("at least one response row is required")
    stacked = np.stack(per_row_norms, axis=0)
    if stat == "mean_norm":
        pooled = stacked.reshape((-1, stacked.shape[-1]))
        per_replicate = np.mean(stacked, axis=(0, 2))
        mean = np.mean(pooled, axis=0)
        sem = _sem(pooled, axis=0)
        n_samples = int(pooled.shape[0])
    else:
        pooled = stacked.reshape((-1, stacked.shape[-1]))
        per_replicate = np.max(stacked, axis=(0, 2))
        mean = np.max(pooled, axis=0)
        sem = None
        n_samples = int(pooled.shape[0])
    time_s = np.arange(mean.shape[0], dtype=np.float64) * float(dt_s)
    return CurveStats(
        time_s=time_s,
        mean=mean,
        sem=sem,
        per_replicate=per_replicate,
        n_replicates=int(per_replicate.shape[0]),
        n_samples=n_samples,
        n_component_rows=len(per_row_norms),
        n_rollouts_per_replicate=int(n_rollouts or 0),
    )


def align_and_equalize_response(values: np.ndarray, row: Mapping[str, Any]) -> np.ndarray:
    """Return signed responses in the target-relative convention before norming.

    The current calibrated screen uses +x reaches, so the target-relative basis is
    the identity. We still equalize perturbation signs before pooling rows, and
    keep this as a single function so future non-+x rows have one explicit place
    to generalize the radial/tangential rotation.
    """

    if values.ndim != 4 or values.shape[-1] != 2:
        raise ValueError("response values must have shape (replicate, rollout, time, xy)")
    sign = int(row.get("sign") or row.get("perturbation", {}).get("sign") or 1)
    return np.asarray(values, dtype=np.float64) * float(sign)


class ExtlqgCurveCache:
    """Lazy cache for deterministic extLQG response curves."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._context: Mapping[str, Any] | None = None
        self._curves: dict[tuple[str, str], DeterministicCurve] = {}

    def group_curve(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        metric: Literal["delta_position", "delta_action"],
        dt_s: float,
    ) -> DeterministicCurve:
        """Return a band-free deterministic aggregate curve for matching rows."""

        if not self.enabled:
            return DeterministicCurve(
                time_s=np.array([], dtype=np.float64),
                value=np.array([], dtype=np.float64),
                status="disabled",
                reason="extLQG reconstruction disabled",
            )
        supported = [row for row in rows if row.get("channel") in SUPPORTED_EXTLQG_CHANNELS]
        if not supported:
            channels = sorted({str(row.get("channel")) for row in rows})
            return DeterministicCurve(
                time_s=np.array([], dtype=np.float64),
                value=np.array([], dtype=np.float64),
                status="not_applicable",
                reason=f"no extLQG adapter for channel(s): {', '.join(channels)}",
            )
        curves = []
        blocked: Counter[str] = Counter()
        for row in supported:
            curve = self._row_curve(row, metric=metric, dt_s=dt_s)
            if curve.status == "available":
                curves.append(curve.value)
            else:
                blocked[curve.reason or curve.status] += 1
        if not curves:
            reason = "; ".join(f"{key}: {value}" for key, value in sorted(blocked.items()))
            return DeterministicCurve(
                time_s=np.array([], dtype=np.float64),
                value=np.array([], dtype=np.float64),
                status="blocked",
                reason=reason or "all extLQG row reconstructions failed",
            )
        stacked = np.stack(curves, axis=0)
        value = np.mean(stacked, axis=0)
        time_s = np.arange(value.shape[0], dtype=np.float64) * float(dt_s)
        return DeterministicCurve(
            time_s=time_s,
            value=value,
            status="available",
            n_component_rows=len(curves),
        )

    def _row_curve(
        self,
        row: Mapping[str, Any],
        *,
        metric: Literal["delta_position", "delta_action"],
        dt_s: float,
    ) -> DeterministicCurve:
        key = (str(row["perturbation_id"]), metric)
        if key in self._curves:
            return self._curves[key]
        if self._context is None:
            self._context = _build_extlqg_comparator_context()
        try:
            base = self._context["base_evaluation"]
            perturbed, _initial_state, _adapter = _simulate_extlqg_perturbed(
                row["perturbation"],
                context=self._context,
            )
            base_values = base.position if metric == "delta_position" else base.command
            perturbed_values = perturbed.position if metric == "delta_position" else perturbed.command
            aligned = align_and_equalize_response(
                np.asarray(perturbed_values - base_values, dtype=np.float64),
                row,
            )
            norms = np.linalg.norm(aligned, axis=-1)
            value = np.mean(norms, axis=(0, 1))
            curve = DeterministicCurve(
                time_s=np.arange(value.shape[0], dtype=np.float64) * float(dt_s),
                value=value,
                status="available",
                n_component_rows=1,
            )
        except (KeyError, ValueError) as exc:
            curve = DeterministicCurve(
                time_s=np.array([], dtype=np.float64),
                value=np.array([], dtype=np.float64),
                status="blocked",
                reason=str(exc),
            )
        self._curves[key] = curve
        return curve


def render_response_norm_note(
    manifest: Mapping[str, Any],
    *,
    spec_path: Path,
    manifest_path: Path,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Render a short reproducibility note for the generated figures."""

    ext_status = ", ".join(
        f"{key}: {value}" for key, value in manifest.get("extlqg_trace_status", {}).items()
    )
    lines = [
        "# GRU Perturbation-Response Norm Plots",
        "",
        "This sidecar materializes Plotly response-curve HTML from existing calibrated "
        "perturbation-response bulk arrays. It does not rerun GRU diagnostics.",
        "",
        f"- Source manifest: `{manifest['source_manifest_path']}`",
        f"- Selector: `{manifest['selector']}`",
        f"- Spec: `{_repo_relative(spec_path, repo_root=repo_root)}`",
        f"- Manifest: `{_repo_relative(manifest_path, repo_root=repo_root)}`",
        f"- HTML inventory: {manifest['figure_count']} files under "
        f"`{manifest.get('asset_dir', DEFAULT_ASSET_DIR)}`",
        "- Aggregation: target-relative/sign-equalized xy responses are converted to "
        "Euclidean norms before pooling. Mean-norm panels show mean +/- SEM over "
        "pooled replicate x eval-seed samples; max-norm panels are unbanded pooled "
        "extreme-response curves.",
        "- ExtLQG: deterministic dotted traces are reconstructed for command-input, "
        "initial-state, process-epsilon, sensory-feedback, and delayed-observation rows.",
    ]
    if ext_status:
        lines.append(f"- ExtLQG trace status counts: {ext_status}.")
    lines.extend(
        [
            "",
            "## Interpretation Caveats",
            "",
            "- Graph-adapter rows are paired against base rows evaluated on the same "
            "adapter-augmented graph with a zero payload, so pre-window differences "
            "reflect the declared perturbation path rather than a graph-topology change.",
            "- Initial-state extLQG traces use a nominal estimator/controller initial "
            "state while perturbing the plant initial state, matching the delayed "
            "visibility contract used by the GRU rows.",
        ]
    )
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    for figure in manifest.get("figures", []):
        lines.append(
            f"- `{figure['html_path']}` - {figure['class']} / {figure['metric']} / "
            f"{figure['family']} / {figure['facet']}"
        )
    lines.append("")
    return "\n".join(lines)


def _write_class_a_figures(
    inputs: PlotInputs,
    *,
    metric: Literal["delta_position", "delta_action"],
    asset_dir: Path,
    repo_root: Path,
    ext_cache: ExtlqgCurveCache,
    ext_status_counter: Counter[str],
    learning_rates: Sequence[str],
    training_levels: Sequence[str],
) -> list[dict[str, Any]]:
    records = []
    for family, timing_bin in _class_a_facets(inputs):
        html_name = f"class_a__{metric}__{family}__{timing_bin}.html"
        html_path = asset_dir / html_name
        fig = _make_base_figure(
            title=f"{metric}: {family} / {timing_bin}",
            learning_rates=learning_rates,
        )
        trace_count = 0
        ext_count = 0
        for row_idx, stat in enumerate(STAT_COLUMNS, start=1):
            for col_idx, lr in enumerate(learning_rates, start=1):
                for severity in SEVERITY_ORDER:
                    for training_level in training_levels:
                        rows = _matching_rows(
                            inputs,
                            learning_rate=lr,
                            training_level=training_level,
                            family=family,
                            timing_bin=timing_bin,
                            severity=severity,
                        )
                        if not rows:
                            continue
                        run = _run_for(inputs, learning_rate=lr, training_level=training_level)
                        stats = aggregate_response_curves(
                            rows,
                            metric=metric,
                            dt_s=run.dt_s,
                            repo_root=repo_root,
                            stat=stat,
                        )
                        _add_sem_curve(
                            fig,
                            stats,
                            row=row_idx,
                            col=col_idx,
                            color=SEVERITY_COLORS[severity],
                            width=TRAINING_WIDTHS[training_level],
                            name=f"{severity} / train {training_level}",
                            legendgroup=f"a-{severity}-{training_level}",
                            showlegend=row_idx == 1 and col_idx == 1,
                        )
                        trace_count += 1
                    ext_rows = _matching_rows(
                        inputs,
                        learning_rate=lr,
                        family=family,
                        timing_bin=timing_bin,
                        severity=severity,
                    )
                    if ext_rows:
                        run = _run_for(inputs, learning_rate=lr)
                        ext_curve = ext_cache.group_curve(ext_rows, metric=metric, dt_s=run.dt_s)
                        ext_status_counter[ext_curve.status] += 1
                        if ext_curve.status == "available":
                            _add_deterministic_curve(
                                fig,
                                ext_curve,
                                row=row_idx,
                                col=col_idx,
                                color=SEVERITY_COLORS[severity],
                                name=f"extLQG {severity}",
                                legendgroup=f"a-ext-{severity}",
                                showlegend=row_idx == 1 and col_idx == 1,
                            )
                            ext_count += 1
        if trace_count == 0:
            continue
        _finish_figure(fig, metric=metric)
        fig.write_html(html_path, include_plotlyjs="cdn")
        records.append(
            _figure_record(
                "class_a",
                metric=metric,
                family=family,
                facet=timing_bin,
                html_path=html_path,
                repo_root=repo_root,
                trace_count=trace_count,
                extlqg_trace_count=ext_count,
            )
        )
    return records


def _write_class_b_figures(
    inputs: PlotInputs,
    *,
    metric: Literal["delta_position", "delta_action"],
    asset_dir: Path,
    repo_root: Path,
    ext_cache: ExtlqgCurveCache,
    ext_status_counter: Counter[str],
    learning_rates: Sequence[str],
    training_levels: Sequence[str],
) -> list[dict[str, Any]]:
    records = []
    for family, severity in _class_b_facets(inputs):
        html_name = f"class_b__{metric}__{family}__{severity}.html"
        html_path = asset_dir / html_name
        fig = _make_base_figure(
            title=f"{metric}: {family} / {severity}",
            learning_rates=learning_rates,
        )
        trace_count = 0
        ext_count = 0
        timings = _timings_for(inputs, family=family, severity=severity)
        for row_idx, stat in enumerate(STAT_COLUMNS, start=1):
            for col_idx, lr in enumerate(learning_rates, start=1):
                for timing_bin in timings:
                    for training_level in training_levels:
                        rows = _matching_rows(
                            inputs,
                            learning_rate=lr,
                            training_level=training_level,
                            family=family,
                            timing_bin=timing_bin,
                            severity=severity,
                        )
                        if not rows:
                            continue
                        run = _run_for(inputs, learning_rate=lr, training_level=training_level)
                        stats = aggregate_response_curves(
                            rows,
                            metric=metric,
                            dt_s=run.dt_s,
                            repo_root=repo_root,
                            stat=stat,
                        )
                        _add_sem_curve(
                            fig,
                            stats,
                            row=row_idx,
                            col=col_idx,
                            color=TIMING_COLORS[timing_bin],
                            width=TRAINING_WIDTHS[training_level],
                            name=f"{timing_bin} / train {training_level}",
                            legendgroup=f"b-{timing_bin}-{training_level}",
                            showlegend=row_idx == 1 and col_idx == 1,
                        )
                        trace_count += 1
                    ext_rows = _matching_rows(
                        inputs,
                        learning_rate=lr,
                        family=family,
                        timing_bin=timing_bin,
                        severity=severity,
                    )
                    if ext_rows:
                        run = _run_for(inputs, learning_rate=lr)
                        ext_curve = ext_cache.group_curve(ext_rows, metric=metric, dt_s=run.dt_s)
                        ext_status_counter[ext_curve.status] += 1
                        if ext_curve.status == "available":
                            _add_deterministic_curve(
                                fig,
                                ext_curve,
                                row=row_idx,
                                col=col_idx,
                                color=TIMING_COLORS[timing_bin],
                                name=f"extLQG {timing_bin}",
                                legendgroup=f"b-ext-{timing_bin}",
                                showlegend=row_idx == 1 and col_idx == 1,
                            )
                            ext_count += 1
        if trace_count == 0:
            continue
        _finish_figure(fig, metric=metric)
        fig.write_html(html_path, include_plotlyjs="cdn")
        records.append(
            _figure_record(
                "class_b",
                metric=metric,
                family=family,
                facet=severity,
                html_path=html_path,
                repo_root=repo_root,
                trace_count=trace_count,
                extlqg_trace_count=ext_count,
            )
        )
    return records


def _make_base_figure(*, title: str, learning_rates: Sequence[str]) -> go.Figure:
    n_cols = len(learning_rates)
    if n_cols <= 0:
        raise ValueError("at least one learning-rate column is required")
    fig = make_subplots(
        rows=2,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    ).update_layout(
        title={
            "text": title,
            "x": 0.0,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
        },
        width=720 if n_cols == 1 else 1180,
        height=760,
        margin={"l": 130, "r": 30, "t": 120, "b": 60},
        hovermode="x unified",
        legend={"groupclick": "togglegroup"},
    )
    column_headers = tuple(
        (_learning_rate_header(lr), (col + 0.5) / n_cols)
        for col, lr in enumerate(learning_rates)
    )
    row_headers = (
        ("mean", 0.755),
        ("max", 0.245),
    )
    for text, x in column_headers:
        fig.add_annotation(
            text=f"<b>{text}</b>",
            x=x,
            y=1.04,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font={"size": 15},
        )
    for text, y in row_headers:
        fig.add_annotation(
            text=f"<b>{text}</b>",
            x=-0.13,
            y=y,
            xref="paper",
            yref="paper",
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font={"size": 15},
        )
    return fig


def _add_sem_curve(
    fig: go.Figure,
    stats: CurveStats,
    *,
    row: int,
    col: int,
    color: str,
    width: float,
    name: str,
    legendgroup: str,
    showlegend: bool,
) -> None:
    if stats.sem is not None:
        upper = stats.mean + stats.sem
        lower = stats.mean - stats.sem
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([stats.time_s, stats.time_s[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=_rgba(color, 0.14),
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                legendgroup=legendgroup,
                name=f"{name} +/- SEM",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=stats.time_s,
            y=stats.mean,
            mode="lines",
            line={"color": color, "width": width},
            legendgroup=legendgroup,
            name=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _add_deterministic_curve(
    fig: go.Figure,
    curve: DeterministicCurve,
    *,
    row: int,
    col: int,
    color: str,
    name: str,
    legendgroup: str,
    showlegend: bool,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=curve.time_s,
            y=curve.value,
            mode="lines",
            line={"color": color, "width": EXTLQG_WIDTH, "dash": "dot"},
            legendgroup=legendgroup,
            name=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _finish_figure(fig: go.Figure, *, metric: str) -> None:
    label = "||delta x||" if metric == "delta_position" else "||delta u||"
    for row in (1, 2):
        for col in _figure_columns(fig):
            fig.update_yaxes(title_text=label, zeroline=True, row=row, col=col)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    for col in _figure_columns(fig)[1:]:
        fig.update_xaxes(title_text="Time (s)", row=2, col=col)
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
    _apply_common_y_range(fig)


def _apply_common_y_range(fig: go.Figure) -> None:
    """Apply one y-axis range to all four subplots."""

    values: list[float] = []
    for trace in fig.data:
        y = getattr(trace, "y", None)
        if y is None:
            continue
        arr = np.asarray(y, dtype=np.float64)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.extend(float(value) for value in finite)
    if not values:
        return
    lower = min(0.0, min(values))
    upper = max(values)
    if upper <= lower:
        upper = lower + 1.0
    padding = 0.05 * (upper - lower)
    fig.update_yaxes(range=[lower, upper + padding])


def _matching_rows(
    inputs: PlotInputs,
    *,
    learning_rate: str,
    family: str,
    timing_bin: str | None = None,
    severity: str | None = None,
    training_level: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for run_id, run in inputs.runs.items():
        if run.learning_rate != learning_rate:
            continue
        if training_level is not None and run.training_level != training_level:
            continue
        for row in inputs.rows[run_id]:
            if row.get("family") != family:
                continue
            if timing_bin is not None and row["_timing_bin_normalized"] != timing_bin:
                continue
            if severity is not None and row["_severity"] != severity:
                continue
            rows.append(row)
    return rows


def _available_learning_rates(inputs: PlotInputs) -> tuple[str, ...]:
    """Return manifest learning rates in canonical display order."""

    observed = {run.learning_rate for run in inputs.runs.values()}
    ordered = [value for value in LR_ORDER if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return tuple(ordered)


def _available_training_levels(inputs: PlotInputs) -> tuple[str, ...]:
    """Return perturbation-training levels in canonical display order."""

    observed = {run.training_level for run in inputs.runs.values()}
    ordered = [value for value in TRAINING_LEVEL_ORDER if value in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return tuple(ordered)


def _figure_columns(fig: go.Figure) -> tuple[int, ...]:
    """Return one-based subplot columns from a Plotly figure grid."""

    grid = getattr(fig, "_grid_ref", None)
    if grid is None or not grid:
        return (1,)
    return tuple(range(1, len(grid[0]) + 1))


def _class_a_facets(inputs: PlotInputs) -> list[tuple[str, str]]:
    facets = {
        (str(row["family"]), str(row["_timing_bin_normalized"]))
        for run_rows in inputs.rows.values()
        for row in run_rows
    }
    return sorted(facets, key=lambda item: (item[0], _order_index(TIMING_ORDER, item[1])))


def _class_b_facets(inputs: PlotInputs) -> list[tuple[str, str]]:
    facets = {
        (str(row["family"]), str(row["_severity"]))
        for run_rows in inputs.rows.values()
        for row in run_rows
    }
    return sorted(facets, key=lambda item: (item[0], _order_index(SEVERITY_ORDER, item[1])))


def _timings_for(inputs: PlotInputs, *, family: str, severity: str) -> list[str]:
    timings = {
        str(row["_timing_bin_normalized"])
        for run_rows in inputs.rows.values()
        for row in run_rows
        if row.get("family") == family and row["_severity"] == severity
    }
    return sorted(timings, key=lambda item: _order_index(TIMING_ORDER, item))


def _run_for(
    inputs: PlotInputs,
    *,
    learning_rate: str,
    training_level: str | None = None,
) -> RunDescriptor:
    for run in inputs.runs.values():
        if run.learning_rate == learning_rate and (
            training_level is None or run.training_level == training_level
        ):
            return run
    raise KeyError(f"no run for {learning_rate=} {training_level=}")


def _figure_record(
    figure_class: str,
    *,
    metric: str,
    family: str,
    facet: str,
    html_path: Path,
    repo_root: Path,
    trace_count: int,
    extlqg_trace_count: int,
) -> dict[str, Any]:
    return {
        "class": figure_class,
        "metric": metric,
        "family": family,
        "facet": facet,
        "html_path": _repo_relative(html_path, repo_root=repo_root),
        "trace_count": int(trace_count),
        "extlqg_trace_count": int(extlqg_trace_count),
    }


def _severity(row: Mapping[str, Any]) -> str:
    perturbation = row.get("perturbation", {})
    value = row.get("level_name") or perturbation.get("level_name")
    if value is not None:
        return str(value)
    parts = str(row["perturbation_id"]).split("__")
    return parts[1] if len(parts) > 2 else "unknown"


def _normalized_timing_bin(value: Any) -> str:
    text = str(value)
    if text.endswith("_visible"):
        return text.removesuffix("_visible")
    return text


def _bulk_path_for_row(
    row: Mapping[str, Any],
    bulk_files: Mapping[str, str],
) -> str | None:
    perturbation_id = str(row["perturbation_id"])
    if perturbation_id in bulk_files:
        return bulk_files[perturbation_id]
    bulk_arrays = row.get("bulk_arrays") or {}
    path = bulk_arrays.get("path") if isinstance(bulk_arrays, Mapping) else None
    return str(path) if path else None


def _load_bulk_detail_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any] | None:
    """Load the ignored full-detail manifest referenced by a slim tracked manifest."""

    detail = manifest.get("bulk_detail_manifest")
    if not isinstance(detail, Mapping):
        return None
    path = detail.get("path")
    if path is None:
        return None
    detail_path = _resolve_repo_path(str(path), repo_root=repo_root)
    return json.loads(detail_path.read_text())


def _write_response_norm_regeneration_spec(
    *,
    spec_path: Path,
    source_manifest: Path,
    results_dir: Path,
    asset_dir: Path,
    note_path: Path,
    manifest_path: Path,
    figure_spec_path: Path,
    reconstruct_extlqg: bool,
    run_id_contains: Sequence[str],
    repo_root: Path,
) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/materialize_gru_perturbation_response_norm_plots.py",
        "--source-manifest",
        repo_relative(source_manifest, repo_root=repo_root),
        "--results-dir",
        repo_relative(results_dir, repo_root=repo_root),
        "--asset-dir",
        repo_relative(asset_dir, repo_root=repo_root),
        "--note-path",
        repo_relative(note_path, repo_root=repo_root),
        "--manifest-path",
        repo_relative(manifest_path, repo_root=repo_root),
        "--regeneration-spec-path",
        repo_relative(spec_path, repo_root=repo_root),
    ]
    for token in run_id_contains:
        command.extend(["--run-id-contains", token])
    if not reconstruct_extlqg:
        command.append("--no-extlqg")
    return write_regeneration_spec(
        spec_path=spec_path,
        diagnostic_name="gru_perturbation_response_norm_plots",
        materializer=(
            "rlrmp.analysis.pipelines.gru_perturbation_response_norm_plots."
            "materialize_response_norm_plots"
        ),
        command=command,
        parameters={
            "reconstruct_extlqg": reconstruct_extlqg,
            "run_id_contains": list(run_id_contains),
            "metrics": list(METRICS),
            "stat_columns": list(STAT_COLUMNS),
            "selector": DEFAULT_SELECTOR,
        },
        inputs=[
            {"role": "source_manifest", "path": source_manifest},
            {
                "role": "source_bulk_root",
                "path": DEFAULT_CORRECTED_BULK_ROOT,
                "description": "bulk response arrays referenced by the source manifest",
            },
        ],
        outputs=[
            {"role": "figure_spec", "path": figure_spec_path},
            {"role": "plot_manifest", "path": manifest_path},
            {"role": "markdown_note", "path": note_path},
            {"role": "html_asset_directory", "path": asset_dir},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/gru_perturbation_response_norm_plots.py",
            "scripts/materialize_gru_perturbation_response_norm_plots.py",
            "src/rlrmp/analysis/pipelines/diagnostic_provenance.py",
        ],
        notes=[
            "Materialization reads existing perturbation-response bulk arrays.",
            "Large response arrays and full row manifests remain under _artifacts.",
        ],
        repo_root=repo_root,
    )


def _parse_learning_rate(label: str, *, run_id: str | None = None) -> str:
    for source in (label, run_id or ""):
        for token in source.split("_"):
            if token.startswith("lr"):
                return token
    raise ValueError(f"could not parse learning rate from {label!r}")


def _parse_training_level(label: str, *, run_id: str | None = None) -> str:
    for source in (label, run_id or ""):
        normalized = (
            source.lower()
            .replace("-", "_")
            .replace("=", "_")
            .replace(" ", "_")
            .replace(".", "p")
        )
        if "sisu" in normalized and "raw_strong_gamma_1p05" in normalized:
            return "sisu_raw_strong_gamma_1p05"
        if "sisu" in normalized and "effective_020a65b_pgd" in normalized:
            return "sisu_effective_020a65b_pgd"
        if source.startswith("proprio_"):
            return source.split("_", maxsplit=1)[1]
        if source.startswith("none_") or "__none_" in source:
            return "none"
        if source.startswith("cal_small_") or "__cal_small_" in source or "_cal_small_" in source:
            return "small"
        if (
            source.startswith("cal_moderate_")
            or "__cal_moderate_" in source
            or "_cal_moderate_" in source
        ):
            return "moderate"
        if source.startswith("cal_stress_") or "__cal_stress_" in source or "_cal_stress_" in source:
            return "stress"
    raise ValueError(f"could not parse perturbation training level from {label!r}")


def _sem(values: np.ndarray, *, axis: int) -> np.ndarray:
    n = int(values.shape[axis])
    if n <= 1:
        return np.zeros(np.mean(values, axis=axis).shape, dtype=np.float64)
    return np.std(values, axis=axis, ddof=1) / np.sqrt(float(n))


def _order_index(order: Sequence[str], value: str) -> int:
    try:
        return order.index(value)
    except ValueError:
        return len(order)


def _rgba(hex_color: str, alpha: float) -> str:
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _learning_rate_header(value: str) -> str:
    return value.removeprefix("lr")


def _resolve_repo_path(path: Path | str, *, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    absolute = path if path.is_absolute() else repo_root / path
    try:
        return str(absolute.relative_to(repo_root.absolute()))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


__all__ = [
    "CurveStats",
    "DEFAULT_ASSET_DIR",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_NOTE_PATH",
    "DEFAULT_REGENERATION_SPEC_PATH",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SOURCE_MANIFEST",
    "DeterministicCurve",
    "ExtlqgCurveCache",
    "PlotInputs",
    "RunDescriptor",
    "SCHEMA_VERSION",
    "aggregate_response_curves",
    "load_plot_inputs",
    "materialize_response_norm_plots",
    "render_response_norm_note",
]

"""Materialize delayed movement-bank PGD vs no-PGD velocity profiles.

The fixed-bank PGD and no-PGD velocity profiles already exist as self-contained
Plotly HTML files. This issue-local driver extracts those traces, relabels the
two GRU rows, and writes comparison figures under a distinct 4d79e07 topic.
The original PGD-only and no-PGD-only figures are left intact.
"""

from __future__ import annotations
from rlrmp.paths import portable_repo_path
from rlrmp.viz.colors import hex_to_rgba

import base64
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from rlrmp.paths import REPO_ROOT, mkdir_p


EXPERIMENT = "4d79e07"
PGD_RUN_ID = "delayed_movement_bank_pgd_clip5"
PGD_LABEL = "PGD delayed movement-bank clip5"
BASELINE_EXPERIMENT = "6c36536"
BASELINE_RUN_ID = "delayed_movement_bank"
BASELINE_LABEL = "No-PGD delayed movement-bank (6c36536)"
TOPIC = "delayed_movement_bank_pgd_clip5_vs_no_pgd_velocity_profiles"
HINF_SOURCE_HTML = (
    "_artifacts/020a65b/figures/"
    "nominal_velocity_overlay_3e3_pgd_vs_baseline_with_analytical/"
    "nominal_forward_velocity_overlay_3e3_pgd_vs_baseline_with_analytical.html"
)
def main() -> None:
    """CLI entry point."""

    spec_path = REPO_ROOT / "results" / EXPERIMENT / "figures" / TOPIC / "spec.json"
    output_root = REPO_ROOT / "_artifacts" / EXPERIMENT / "figures" / TOPIC
    spec = build_spec(output_root)
    mkdir_p(spec_path.parent)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": "rlrmp.delayed_velocity_profile_comparison.v1",
        "issue": EXPERIMENT,
        "topic": TOPIC,
        "baseline_source": {
            "issue": BASELINE_EXPERIMENT,
            "run_id": BASELINE_RUN_ID,
            "run_spec": f"results/{BASELINE_EXPERIMENT}/runs/{BASELINE_RUN_ID}.json",
            "figure_topic": f"_artifacts/{BASELINE_EXPERIMENT}/figures/{BASELINE_RUN_ID}_velocity_profiles",
            "selection": "previous good delayed_movement_bank baseline, not a gradient-clip repeat",
        },
        "pgd_source": {
            "issue": EXPERIMENT,
            "run_id": PGD_RUN_ID,
            "run_spec": f"results/{EXPERIMENT}/runs/{PGD_RUN_ID}.json",
            "figure_topic": f"_artifacts/{EXPERIMENT}/figures/{PGD_RUN_ID}_velocity_profiles",
        },
        "spec": repo_rel(spec_path),
        "outputs": {},
    }

    for bank_kind in ("no_catch", "catch"):
        output_dir = output_root / bank_kind
        mkdir_p(output_dir)
        pgd_html = (
            REPO_ROOT
            / "_artifacts"
            / EXPERIMENT
            / "figures"
            / f"{PGD_RUN_ID}_velocity_profiles"
            / bank_kind
            / "forward_velocity_profiles_stochastic.html"
        )
        baseline_html = (
            REPO_ROOT
            / "_artifacts"
            / BASELINE_EXPERIMENT
            / "figures"
            / f"{BASELINE_RUN_ID}_velocity_profiles"
            / bank_kind
            / "forward_velocity_profiles_stochastic.html"
        )
        pgd_summary_path = pgd_html.with_name("velocity_profile_summary.json")
        baseline_summary_path = baseline_html.with_name("velocity_profile_summary.json")
        pgd_data, pgd_layout = read_plotly_html(pgd_html)
        baseline_data, _baseline_layout = read_plotly_html(baseline_html)

        traces = [
            relabel_gru_trace(baseline_data[0], label=BASELINE_LABEL, color="#2563eb", is_band=True),
            relabel_gru_trace(baseline_data[1], label=BASELINE_LABEL, color="#2563eb", is_band=False),
            relabel_gru_trace(pgd_data[0], label=PGD_LABEL, color="#dc2626", is_band=True),
            relabel_gru_trace(pgd_data[1], label=PGD_LABEL, color="#dc2626", is_band=False),
        ]
        if bank_kind == "no_catch":
            traces.extend(relabel_reference_trace(trace) for trace in pgd_data[2:])
            hinf_data, _hinf_layout = read_plotly_html(REPO_ROOT / HINF_SOURCE_HTML)
            traces.extend(relabel_hinf_trace(trace) for trace in hinf_data[6:8])

        layout = comparison_layout(pgd_layout, bank_kind=bank_kind)
        output_html = output_dir / "forward_velocity_profiles_stochastic.html"
        go.Figure(data=traces, layout=layout).write_html(output_html)

        bank_summary = {
            "figure": repo_rel(output_html),
            "inputs": {
                "pgd_html": repo_rel(pgd_html),
                "baseline_html": repo_rel(baseline_html),
                "pgd_summary": repo_rel(pgd_summary_path),
                "baseline_summary": repo_rel(baseline_summary_path),
                "hinf_source_html": HINF_SOURCE_HTML if bank_kind == "no_catch" else None,
            },
            "pgd": compact_profile_summary(pgd_summary_path, issue=EXPERIMENT),
            "baseline": compact_profile_summary(
                baseline_summary_path,
                issue=BASELINE_EXPERIMENT,
            ),
        }
        bank_summary["delta"] = profile_delta(bank_summary["pgd"], bank_summary["baseline"])
        summary_path = output_dir / "velocity_profile_comparison_summary.json"
        summary_path.write_text(
            json.dumps(bank_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["outputs"][bank_kind] = bank_summary

    manifest_path = output_root / "velocity_profile_comparison_manifest.json"
    mkdir_p(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_spec(output_root: Path) -> dict[str, Any]:
    """Return the tracked comparison figure spec."""

    return {
        "schema_version": "rlrmp.figure_spec.delayed_velocity_profile_comparison.v1",
        "issue": EXPERIMENT,
        "topic": TOPIC,
        "description": (
            "Overlay existing delayed movement-bank PGD clip5 velocity traces with "
            "the matched no-PGD delayed_movement_bank baseline from 6c36536."
        ),
        "output_root": repo_rel(output_root),
        "source_figures": {
            "pgd": f"_artifacts/{EXPERIMENT}/figures/{PGD_RUN_ID}_velocity_profiles/{{bank}}/forward_velocity_profiles_stochastic.html",
            "baseline": f"_artifacts/{BASELINE_EXPERIMENT}/figures/{BASELINE_RUN_ID}_velocity_profiles/{{bank}}/forward_velocity_profiles_stochastic.html",
        },
        "rows": [
            {
                "label": PGD_LABEL,
                "issue": EXPERIMENT,
                "run_id": PGD_RUN_ID,
                "role": "pgd",
            },
            {
                "label": BASELINE_LABEL,
                "issue": BASELINE_EXPERIMENT,
                "run_id": BASELINE_RUN_ID,
                "role": "baseline",
                "selection_reason": (
                    "previous good delayed_movement_bank baseline, not a "
                    "gradient-clip repeat"
                ),
            },
        ],
        "banks": ["no_catch", "catch"],
        "plot": {
            "x_axis": "time relative to go cue (s)",
            "y_axis": "target-radial velocity (m/s)",
            "band": "mean +/- 1 SD over pooled replicate x fixed-bank go-cue/direction trials",
            "references": "C&S output-feedback extLQG overlays on no_catch only",
            "h_infinity_reference": (
                "output-feedback robust analytical movement-period trace copied "
                f"from {HINF_SOURCE_HTML}"
            ),
        },
    }


def read_plotly_html(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return decoded Plotly data and layout from a self-contained HTML figure."""

    text = path.read_text(encoding="utf-8")
    start = text.rfind("Plotly.newPlot(")
    if start < 0:
        raise ValueError(f"No Plotly.newPlot payload found in {path}")
    decoder = json.JSONDecoder()
    idx = skip_whitespace(text, text.find("(", start) + 1)
    _div_id, consumed = decoder.raw_decode(text[idx:])
    idx = advance_to_next_argument(text, idx + consumed)
    data, consumed = decoder.raw_decode(text[idx:])
    idx = advance_to_next_argument(text, idx + consumed)
    layout, _consumed = decoder.raw_decode(text[idx:])
    return decode_plotly_typed_arrays(data), decode_plotly_typed_arrays(layout)


def skip_whitespace(text: str, idx: int) -> int:
    """Return the next non-whitespace index."""

    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def advance_to_next_argument(text: str, idx: int) -> int:
    """Return the index just after the next comma and following whitespace."""

    comma = text.find(",", idx)
    if comma < 0:
        raise ValueError("Malformed Plotly.newPlot payload")
    return skip_whitespace(text, comma + 1)


def decode_plotly_typed_arrays(value: Any) -> Any:
    """Decode Plotly's compact typed-array JSON encoding into Python lists."""

    if isinstance(value, list):
        return [decode_plotly_typed_arrays(item) for item in value]
    if isinstance(value, dict):
        if set(value) >= {"dtype", "bdata"}:
            dtype = dtype_for_plotly(value["dtype"])
            array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype)
            if "shape" in value:
                array = array.reshape(tuple(value["shape"]))
            return array.tolist()
        return {key: decode_plotly_typed_arrays(item) for key, item in value.items()}
    return value


def dtype_for_plotly(dtype: str) -> np.dtype[Any]:
    """Return a NumPy dtype for Plotly's typed-array abbreviation."""

    mapping = {
        "f8": "<f8",
        "f4": "<f4",
        "i4": "<i4",
        "u4": "<u4",
        "i2": "<i2",
        "u2": "<u2",
        "i1": "i1",
        "u1": "u1",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported Plotly typed-array dtype: {dtype}")
    return np.dtype(mapping[dtype])


def relabel_gru_trace(
    trace: dict[str, Any],
    *,
    label: str,
    color: str,
    is_band: bool,
) -> dict[str, Any]:
    """Return a renamed GRU trace with stable comparison colors."""

    trace = deepcopy(trace)
    trace["legendgroup"] = "pgd" if label == PGD_LABEL else "baseline"
    if is_band:
        trace["name"] = f"{label} mean +/- 1 SD"
        trace["fillcolor"] = rgba(color, 0.12)
        trace["showlegend"] = False
    else:
        trace["name"] = label
        trace["showlegend"] = True
        trace.setdefault("line", {})["color"] = color
        trace["line"]["width"] = 2.4
    return trace


def relabel_reference_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Return a reference trace with the original styling preserved."""

    trace = deepcopy(trace)
    trace["showlegend"] = True
    return trace


def relabel_hinf_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Return a movement-period robust analytical trace for the delayed overlay."""

    trace = deepcopy(trace)
    original_name = str(trace.get("name", "output-feedback robust analytical"))
    trace["legendgroup"] = "hinf-analytical"
    trace["showlegend"] = " +/- " not in original_name
    if " +/- " in original_name:
        trace["name"] = "H-infinity analytical movement period mean +/- 1 SD"
        trace["fillcolor"] = rgba("#7c3aed", 0.10)
    else:
        trace["name"] = "H-infinity analytical movement period"
        trace.setdefault("line", {})["color"] = "#7c3aed"
        trace["line"]["dash"] = "dot"
        trace["line"]["width"] = 2.4
    return trace


def comparison_layout(layout: dict[str, Any], *, bank_kind: str) -> dict[str, Any]:
    """Return layout for the overlaid comparison figure."""

    layout = deepcopy(layout)
    layout["title"] = {
        "text": f"Delayed movement-bank PGD vs no-PGD target-radial velocity ({bank_kind})"
    }
    layout["width"] = 940
    layout["height"] = 560
    layout["hovermode"] = "x unified"
    layout["legend"] = {"groupclick": "togglegroup"}
    layout.setdefault("xaxis", {}).setdefault("title", {})
    layout["xaxis"]["title"] = {"text": "Time relative to go cue (s)"}
    layout.setdefault("yaxis", {}).setdefault("title", {})
    layout["yaxis"]["title"] = {"text": "Target-radial velocity (m/s)"}
    return layout


def compact_profile_summary(path: Path, *, issue: str) -> dict[str, Any]:
    """Return compact profile metadata from an existing summary artifact."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    profile = summary["profile"]
    return {
        "run_id": summary["run_id"],
        "run_label": summary["run_label"],
        "issue": issue,
        "peak_mean_forward_velocity_m_s": profile["peak_mean_forward_velocity_m_s"],
        "time_of_peak_mean_forward_velocity_s": profile["time_of_peak_mean_forward_velocity_s"],
        "mean_forward_velocity_min_m_s": profile["mean_forward_velocity_min_m_s"],
        "n_pooled_samples": profile["n_pooled_samples"],
        "n_replicates": profile["n_replicates"],
        "n_trials_per_replicate": profile["n_trials_per_replicate"],
        "time_start_s": profile["time_start_s"],
        "time_stop_s": profile["time_stop_s"],
    }


def profile_delta(pgd: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Return compact PGD-minus-baseline metrics."""

    return {
        "peak_mean_forward_velocity_m_s": (
            pgd["peak_mean_forward_velocity_m_s"]
            - baseline["peak_mean_forward_velocity_m_s"]
        ),
        "time_of_peak_mean_forward_velocity_s": (
            pgd["time_of_peak_mean_forward_velocity_s"]
            - baseline["time_of_peak_mean_forward_velocity_s"]
        ),
    }


rgba = hex_to_rgba


repo_rel = portable_repo_path


if __name__ == "__main__":
    main()

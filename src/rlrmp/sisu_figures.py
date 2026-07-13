"""Figure-side adapter for the registered SISU spectrum analysis payload."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.figures import FigureSpec

from rlrmp.figures import standard_matrix_profile_spec
from rlrmp.mappings import as_mapping as _mapping


SISU_FIGURE_PAYLOAD_SCHEMA_ID = "rlrmp.figure_data.sisu_spectrum"
SISU_FIGURE_PAYLOAD_SCHEMA_VERSION = "rlrmp.figure_data.sisu_spectrum.v1"
_SISU_COLORS = {0.0: "#64748b", 0.5: "#2563eb", 1.0: "#dc2626"}


def sisu_figure_payload(analysis_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize analysis-owned profiles into data-bound declarative facets."""
    references = [_reference_series(value) for value in analysis_payload.get("references", ())]
    facets: dict[str, Any] = {}
    for profile_value in analysis_payload.get("profiles", ()):
        profile = _mapping(profile_value)
        run_id = str(profile["run_id"])
        curves = [_curve_series(value) for value in profile.get("curves", ())]
        if not curves:
            raise ValueError(f"SISU profile {run_id!r} has no curves")
        facets[run_id] = {
            "run_id": run_id,
            "display_name": str(profile.get("label", run_id)),
            "sisu_spectrum_velocity": {"series": [*curves, *references]},
        }
    if not facets:
        raise ValueError("SISU figure payload has no profile rows")
    return {
        "schema_id": SISU_FIGURE_PAYLOAD_SCHEMA_ID,
        "schema_version": SISU_FIGURE_PAYLOAD_SCHEMA_VERSION,
        "facets": {"sisu_spectrum_velocity_profiles": facets},
        "summary": analysis_payload.get("summary", {}),
    }


def sisu_spectrum_figure_spec(*, name: str = "sisu-spectrum-velocity") -> FigureSpec:
    """Return the native manifest-bound SISU velocity-profile intent."""
    spec = standard_matrix_profile_spec(
        name=name,
        output="sisu_spectrum_velocity_profiles",
        profile_key="sisu_spectrum_velocity",
        title="Forward velocity (m/s)",
        figure_routing={"topic": "sisu_spectrum_velocity_profiles"},
    )
    return spec.model_copy(
        update={
            "panels": [
                {
                    "name": "profile",
                    "title": {"item": "condition"},
                    "axes_labels": {
                        "x": "Time from go cue (s)",
                        "y": "Forward velocity (m/s)",
                    },
                }
            ],
            "metadata": {
                **dict(spec.metadata),
                "schema_id": SISU_FIGURE_PAYLOAD_SCHEMA_ID,
                "schema_version": SISU_FIGURE_PAYLOAD_SCHEMA_VERSION,
                "shared_yaxes": "all",
                "parity_oracle": (
                    "results/518aea3/data_products/sisu_spectrum_figure_parity.json"
                ),
                "parity_product": (
                    "results/518aea3/data_products/"
                    "sisu_spectrum_figure_parity_product.json"
                ),
            },
        }
    )


def _curve_series(value: Any) -> dict[str, Any]:
    curve = _mapping(value)
    mean = list(curve.get("mean_forward_velocity_m_s", ()))
    spread = list(curve.get("std_forward_velocity_m_s", ()))
    if len(mean) != len(spread):
        raise ValueError("SISU curve mean/spread lengths differ")
    sisu = float(curve["sisu"])
    return {
        "label": f"SISU={sisu:g}",
        "color": _SISU_COLORS.get(sisu, "#0f766e"),
        "profile": {
            "time": list(curve.get("time_s", ())),
            "mean": mean,
            "lower": [y - delta for y, delta in zip(mean, spread, strict=True)],
            "upper": [y + delta for y, delta in zip(mean, spread, strict=True)],
        },
    }


def _reference_series(value: Any) -> dict[str, Any]:
    reference = _mapping(value)
    mean = list(reference.get("forward_velocity_m_s", ()))
    spread = list(reference.get("std_forward_velocity_m_s", ()))
    if len(mean) != len(spread):
        raise ValueError("SISU reference mean/spread lengths differ")
    return {
        "label": str(reference.get("label", "Analytical reference")),
        "color": "#111827",
        "line_dash": "dash",
        "profile": {
            "time": list(reference.get("time_s", ())),
            "mean": mean,
            "lower": [y - delta for y, delta in zip(mean, spread, strict=True)],
            "upper": [y + delta for y, delta in zip(mean, spread, strict=True)],
        },
    }


__all__ = [
    "SISU_FIGURE_PAYLOAD_SCHEMA_ID",
    "SISU_FIGURE_PAYLOAD_SCHEMA_VERSION",
    "sisu_figure_payload",
    "sisu_spectrum_figure_spec",
]

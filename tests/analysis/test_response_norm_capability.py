"""Contract tests for the manifest-canonical response-norm capability."""

from __future__ import annotations

from pathlib import Path

import pytest
from feedbax.plot.constructors import get_figure_constructor, get_figure_template

from rlrmp.analysis.response_norm import (
    RESPONSE_NORM_PAYLOAD_ROLE,
    RESPONSE_NORM_PAYLOAD_SCHEMA_VERSION,
    ResponseNormAnalysis,
    response_norm_payload,
)
from rlrmp.figures import register_rlrmp_figure_surfaces


pytestmark = pytest.mark.feedbax_contract


def _row(model: str, *, metric: str = "delta_position") -> dict[str, object]:
    return {
        "row_id": f"row-{model}",
        "model_id": model,
        "metric": metric,
        "condition_class": "class_a",
        "time": [0.0, 0.01],
        "mean": [0.0, 0.2],
        "sem": [0.0, 0.02],
        "max": [0.0, 0.4],
    }


def test_payload_distinguishes_intrinsic_and_data_bound_facets() -> None:
    payload = response_norm_payload([_row("gru-a"), _row("gru-b")])

    assert payload["schema_version"] == RESPONSE_NORM_PAYLOAD_SCHEMA_VERSION
    assert payload["intrinsic_axes"] == {
        "metric": ["delta_position", "delta_action"],
        "condition_class": ["class_a", "class_b"],
    }
    assert payload["data_bound_axes"]["model"] == ["gru-a", "gru-b"]
    assert len(payload["facets"]) == 4


def test_payload_model_cardinality_is_not_hard_coded() -> None:
    two = response_norm_payload([_row("a"), _row("b")])
    three = response_norm_payload([_row("a"), _row("b"), _row("c")])

    assert len(two["data_bound_axes"]["model"]) == 2
    assert len(three["data_bound_axes"]["model"]) == 3


def test_analysis_emits_schema_bearing_custody_payload(tmp_path: Path) -> None:
    class Context:
        def __init__(self) -> None:
            self.recorded: list[dict[str, object]] = []

        def record_json_artifact(self, payload, **metadata):
            self.recorded.append({"payload": payload, **metadata})

    analysis = ResponseNormAnalysis(variant="response_norm", params={"rows": [_row("gru")]})
    result = analysis.compute(type("Data", (), {"states": {}})())
    context = Context()
    analysis.emit_artifacts(context, type("Data", (), {})(), result=result)

    assert context.recorded[0]["role"] == RESPONSE_NORM_PAYLOAD_ROLE
    assert context.recorded[0]["payload"]["schema_version"] == (
        RESPONSE_NORM_PAYLOAD_SCHEMA_VERSION
    )


def test_registered_template_and_constructors_preserve_response_norm_views() -> None:
    register_rlrmp_figure_surfaces()
    template = get_figure_template("rlrmp.response_norm_comparison")

    assert template.facet_by == ["metric", "condition_class"]
    assert template.metadata["data_bound_collections"] == ["model", "comparison_row"]
    assert get_figure_constructor("rlrmp.response_norm_bands", tier="trace").tier == "trace"
    assert get_figure_constructor("rlrmp.response_norm_bars", tier="trace").tier == "trace"


def test_legacy_response_norm_producer_is_retired() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not (
        root / "src/rlrmp/analysis/pipelines/gru_perturbation_response_norm_plots.py"
    ).exists()
    assert not (root / "scripts/materialize_gru_perturbation_response_norm_plots.py").exists()
    assert not (
        root / "tests/analysis/pipelines/test_gru_perturbation_response_norm_plots.py"
    ).exists()

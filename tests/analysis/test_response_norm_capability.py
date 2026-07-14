"""Contract tests for the manifest-canonical response-norm capability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from feedbax.analysis import authenticated_manifest_ref
from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    spec_payload,
    write_manifest,
)
from feedbax.plot.constructors import get_figure_constructor, get_figure_template

from rlrmp.analysis.response_norm import (
    RESPONSE_NORM_PAYLOAD_ROLE,
    RESPONSE_NORM_PAYLOAD_SCHEMA_VERSION,
    ResponseNormAnalysis,
    response_norm_payload,
)
from rlrmp.figures import (
    register_rlrmp_figure_surfaces,
    response_norm_comparison_spec,
)


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
        "metric": {
            "delta_position": "delta_position",
            "delta_action": "delta_action",
        },
        "condition_class": {"class_a": "class_a", "class_b": "class_b"},
    }
    assert payload["data_bound_axes"]["model"] == ["gru-a", "gru-b"]
    assert sum(len(classes) for classes in payload["facets"].values()) == 4


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


@pytest.mark.parametrize("model_count", [2, 3])
def test_response_norm_spec_executes_intrinsic_facets_with_payload_cardinality(
    tmp_path: Path,
    model_count: int,
) -> None:
    register_rlrmp_figure_surfaces()
    rows = []
    for metric in ("delta_position", "delta_action"):
        for condition_class in ("class_a", "class_b"):
            for index in range(model_count):
                row = _row(f"model-{index}", metric=metric)
                row["condition_class"] = condition_class
                rows.append(row)
    payload = response_norm_payload(rows)
    manifest = AnalysisRunManifest(
        id=f"response-norm-{model_count}",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type="rlrmp.response_norm_comparison").model_dump(mode="json"),
        ),
        metadata={"figure_payload": payload},
    )
    manifest_path = write_manifest(manifest, root=tmp_path)
    parent = authenticated_manifest_ref(
        manifest,
        manifest_path,
        "response_norm_analysis",
    )
    spec = response_norm_comparison_spec(name=f"response-norm-{model_count}")
    spec = spec.model_copy(update={"inputs": [parent]})

    figure_manifest, _path = execute_figure_spec(spec, root=tmp_path)

    assert figure_manifest.status == "completed"
    included = [record for record in figure_manifest.binding_records if record.status == "included"]
    assert len(included) == 8  # two slots across four intrinsic facet combinations
    render_artifact = next(
        artifact
        for artifact in figure_manifest.artifacts
        if artifact.role == "figure_render" and artifact.media_type == "application/json"
    )
    rendered = json.loads(Path(render_artifact.uri).read_text())
    rendered_names = {trace.get("name") for trace in rendered["data"]}
    assert {f"model-{index}" for index in range(model_count)} <= rendered_names

"""Declarative scalar-diagnostic figure and payload-analysis contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from feedbax.analysis.figures import execute_figure_spec
from feedbax.analysis.specs import AnalysisRunSpec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    ParentRef,
    spec_payload,
    write_manifest,
)
import pytest

from rlrmp.analysis.scalar_diagnostic import (
    SCALAR_DIAGNOSTIC_ANALYSIS_TYPE,
    SCALAR_DIAGNOSTIC_SCHEMA_VERSION,
    scalar_diagnostic_recipe,
)
from rlrmp.figures import register_rlrmp_figure_surfaces


REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS = (
    ("91a090c", "adaptive_damage_lambda"),
    ("1ab1fef", "adaptive_damage_lambda"),
    ("410d7ac", "delta_v_signature"),
)

pytestmark = pytest.mark.feedbax_contract


def _record(label: str) -> dict[str, object]:
    return {
        "label": label,
        "traces": {
            "damage": {"x": [0.0, 1.0], "y": [0.0, 1.0], "axis": "primary"},
            "lambda": {"x": [0.0, 1.0], "y": [1.0, 2.0], "axis": "secondary"},
        },
    }


def _payload(count: int) -> dict[str, object]:
    return {
        "schema_id": "rlrmp.figure_data.scalar_diagnostic",
        "schema_version": SCALAR_DIAGNOSTIC_SCHEMA_VERSION,
        "intrinsic_axes": {
            "metric": {"adaptive_damage": "Adaptive damage"},
            "condition_class": {"training": "Training"},
        },
        "collections": {f"row-{index}": _record(f"Row {index}") for index in range(count)},
        "headlines": {"row_count": count},
        "provenance": {},
    }


@pytest.mark.parametrize("count", [2, 3])
def test_registered_analysis_keeps_row_cardinality_data_bound(count: int) -> None:
    resolved = SimpleNamespace(
        ref=ParentRef(kind="EvaluationRunManifest", id=f"eval-{count}", role="evaluation"),
        states={
            "scalar_diagnostic": {
                "collections": {
                    f"row-{index}": _record(f"Row {index}") for index in range(count)
                },
                "headlines": {"row_count": count},
            }
        },
    )
    spec = AnalysisRunSpec(
        analysis_type=SCALAR_DIAGNOSTIC_ANALYSIS_TYPE,
        params={
            "intrinsic_axes": {
                "metric": {"adaptive_damage": "Adaptive damage"},
                "condition_class": {"training": "Training"},
            }
        },
    )

    recipe = scalar_diagnostic_recipe(spec, REPO_ROOT, [resolved])
    payload = recipe.analyses["scalar_diagnostic"].compute(recipe.data)

    assert len(payload["collections"]) == count
    assert payload["headlines"]["row_count"] == count
    assert payload["provenance"][f"eval-{count}"]["manifest_kind"] == (
        "EvaluationRunManifest"
    )


def test_all_tracked_specs_are_native_manifest_data_bound() -> None:
    for issue, topic in SPECS:
        path = REPO_ROOT / "results" / issue / "figures" / topic / "spec.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = FigureSpec.model_validate(payload)
        assert spec.template == "rlrmp.scalar_diagnostic"
        assert spec.assembler is None
        assert all(binding.item == "manifest" for binding in spec.facet_bindings.values())
        assert payload["metadata"]["analysis_type"] == SCALAR_DIAGNOSTIC_ANALYSIS_TYPE
        assert (REPO_ROOT / payload["metadata"]["parity_oracle"]).is_file()


@pytest.mark.parametrize("issue,topic", SPECS)
def test_tracked_specs_execute_to_completed_figure_manifests(
    tmp_path: Path,
    issue: str,
    topic: str,
) -> None:
    register_rlrmp_figure_surfaces()
    manifest = AnalysisRunManifest(
        id=f"scalar-diagnostic-{issue}",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type=SCALAR_DIAGNOSTIC_ANALYSIS_TYPE).model_dump(
                mode="json"
            ),
        ),
        metadata={"figure_payload": _payload(2)},
    )
    write_manifest(manifest, root=tmp_path)
    tracked = FigureSpec.model_validate_json(
        (REPO_ROOT / "results" / issue / "figures" / topic / "spec.json").read_text()
    )
    parent = ParentRef(kind="AnalysisRunManifest", id=manifest.id, role="analysis")

    rendered, path = execute_figure_spec(
        tracked.model_copy(update={"inputs": [parent]}),
        root=tmp_path,
        issues=["ae26a79"],
    )

    assert rendered.status == "completed"
    assert path.is_file()
    assert rendered.resolved_inputs == [parent]


def test_legacy_exclusive_producers_are_retired() -> None:
    assert not (
        REPO_ROOT / "results/91a090c/scripts/materialize_adaptive_damage_lambda.py"
    ).exists()
    assert not (
        REPO_ROOT / "results/410d7ac/scripts/analyse_linear_decoupling_mvp.py"
    ).exists()
    assert not (
        REPO_ROOT / "results/1ab1fef/scripts/materialize_post_run_analysis.py"
    ).exists()

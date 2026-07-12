"""Execution contracts for migrated steady-state perturbation figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec

from rlrmp.figures import register_rlrmp_figure_surfaces
from rlrmp.steady_state_figures import (
    STEADY_STATE_COMPARISON_IDS,
    register_steady_state_figure_pieces,
)


pytestmark = pytest.mark.feedbax_contract
REPO_ROOT = Path(__file__).resolve().parents[2]


def _comparison(comparison_id: str, condition_count: int) -> dict[str, object]:
    conditions = {}
    for index in range(condition_count):
        family_summary = {}
        for family in ("position", "velocity", "force_filter"):
            row = {"relative_time_steps": [-1, 0, 1, 2]}
            for profile in ("output", "position", "velocity"):
                row[f"aligned_{profile}_window_profile_mean"] = [0.0, 0.2, 0.1, 0.0]
                row[f"aligned_{profile}_window_profile_sem"] = [0.0, 0.02, 0.01, 0.0]
                row[f"orthogonal_{profile}_window_profile_mean"] = [0.0, 0.05, 0.02, 0.0]
            family_summary[family] = row
        conditions[f"condition_{index}"] = {
            "label": f"Condition {index}",
            "dt_s": 0.01,
            "family_summary": family_summary,
        }
    return {
        "comparison_id": comparison_id,
        "title": comparison_id,
        "pulse_duration_steps": 2,
        "conditions": conditions,
    }


@pytest.mark.parametrize("condition_count", [2, 3])
def test_all_four_specs_execute_with_variable_condition_cardinality(
    tmp_path: Path,
    condition_count: int,
) -> None:
    payload = {
        "schema_version": "rlrmp.gru_steady_state_perturbation_bank.v1",
        "comparisons": {
            comparison_id: _comparison(comparison_id, condition_count)
            for comparison_id in STEADY_STATE_COMPARISON_IDS
        },
    }
    artifact = tmp_path / "steady_state_detail.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    register_rlrmp_figure_surfaces()
    register_steady_state_figure_pieces(artifact_path=artifact)

    for comparison_id in STEADY_STATE_COMPARISON_IDS:
        spec_path = REPO_ROOT / "results/87424a4/figures" / comparison_id / "spec.json"
        spec = FigureSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
        manifest, _path = execute_figure_spec(spec, root=tmp_path / comparison_id)
        assert manifest.status == "completed"
        assert len([r for r in manifest.binding_records if r.status == "included"]) == 9
        render = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "figure_render" and artifact.media_type == "application/json"
        )
        traces = json.loads(Path(render.uri).read_text(encoding="utf-8"))["data"]
        names = {trace.get("name") for trace in traces}
        assert {f"Condition {index} aligned" for index in range(condition_count)} <= names
        assert {f"Condition {index} orthogonal" for index in range(condition_count)} <= names
        assert {"Wash-in window start", "Wash-in window end"} <= names


def test_surviving_specs_are_native_and_legacy_builder_is_absent() -> None:
    for comparison_id in STEADY_STATE_COMPARISON_IDS:
        path = REPO_ROOT / "results/87424a4/figures" / comparison_id / "spec.json"
        spec = FigureSpec.model_validate_json(path.read_text(encoding="utf-8"))
        assert spec.template == "rlrmp.perturbation_response_comparison"

    for path in (REPO_ROOT / "src/rlrmp").rglob("*.py"):
        assert "build_response_figure" not in path.read_text(encoding="utf-8")

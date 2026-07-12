"""Native stabilization-response figure and parity contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec
import pytest

from rlrmp.data_products.envelope import read_data_product
from rlrmp.figures import register_rlrmp_figure_surfaces
from rlrmp.stabilization_figures import (
    STABILIZATION_COMPARISON_IDS,
    register_stabilization_figure_pieces,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS = (
    ("c92ebd8", "pgd_1p05_stabilization_perturbation_responses"),
    ("c92ebd8", "pgd_ofb_budget_stabilization_perturbation_responses"),
    ("d55c5f0", "soft_pgd_stabilization_perturbation_responses"),
)
RETIRED = (
    "results/c92ebd8/scripts/materialize_pgd_1p05_stabilization_diagnostics.py",
    "results/c92ebd8/scripts/materialize_pgd_ofb_budget_stabilization_responses.py",
    "results/d55c5f0/scripts/materialize_soft_pgd_stabilization_responses.py",
)
pytestmark = pytest.mark.feedbax_contract


def _comparison(comparison_id: str, count: int) -> dict[str, object]:
    conditions = {}
    for index in range(count):
        family_summary = {}
        for family in ("position", "velocity", "force_filter"):
            row = {"relative_time_steps": [-2, -1, 0, 1, 2]}
            for output in ("output", "position", "velocity"):
                row[f"aligned_{output}_window_profile_mean"] = [0.0, 0.0, 0.2, 0.1, 0.0]
                row[f"aligned_{output}_window_profile_sem"] = [0.0, 0.0, 0.02, 0.01, 0.0]
                row[f"orthogonal_{output}_window_profile_mean"] = [0.0, 0.0, 0.05, 0.02, 0.0]
            family_summary[family] = row
        conditions[f"condition-{index}"] = {
            "label": f"Condition {index}",
            "dt_s": 0.01,
            "family_summary": family_summary,
        }
    return {"comparison_id": comparison_id, "pulse_duration_steps": 2, "conditions": conditions}


@pytest.mark.parametrize("condition_count", [2, 3])
def test_all_specs_execute_with_payload_bound_conditions(
    tmp_path: Path, condition_count: int
) -> None:
    payload = {
        "comparisons": {
            comparison_id: _comparison(comparison_id, condition_count)
            for comparison_id in STABILIZATION_COMPARISON_IDS
        }
    }
    artifact = tmp_path / "stabilization_responses.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    register_rlrmp_figure_surfaces()
    register_stabilization_figure_pieces(artifact)
    for issue, topic in SPECS:
        spec = FigureSpec.model_validate_json(
            (REPO_ROOT / "results" / issue / "figures" / topic / "spec.json").read_text()
        )
        manifest, _path = execute_figure_spec(spec, root=tmp_path / topic, issues=["8c06ef4"])
        assert manifest.status == "completed"
        render = next(artifact for artifact in manifest.artifacts if artifact.role == "figure_render")
        assert render.uri is not None
        assert hashlib.sha256(Path(render.uri).read_bytes()).hexdigest() == render.sha256
        data = json.loads(Path(render.uri).read_text())["data"]
        names = {trace.get("name") for trace in data}
        assert {f"Condition {index} aligned" for index in range(condition_count)} <= names
        assert {"Wash-in window start", "Wash-in window end"} <= names


def test_specs_are_native_and_governed_parity_hashes_match() -> None:
    product = read_data_product(
        REPO_ROOT / "results/8c06ef4/data_products/stabilization_response_parity_product.json"
    )
    assert len(product.artifacts) == 3
    for artifact in product.artifacts:
        assert artifact.uri is not None
        assert hashlib.sha256((REPO_ROOT / artifact.uri).read_bytes()).hexdigest() == artifact.sha256
    for issue, topic in SPECS:
        spec = FigureSpec.model_validate_json(
            (REPO_ROOT / "results" / issue / "figures" / topic / "spec.json").read_text()
        )
        assert spec.template == "rlrmp.perturbation_response_comparison"
        assert spec.metadata["parity_product"].startswith("results/8c06ef4/")


def test_producers_and_direct_plotly_paths_are_retired() -> None:
    for relative in RETIRED:
        assert not (REPO_ROOT / relative).exists()
    for root in ("results/c92ebd8", "results/d55c5f0"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "canonical_build_family_figure" not in source


def test_archived_specs_preserve_response_windows_and_overlays() -> None:
    root = REPO_ROOT / "results/8c06ef4/data_products/archived_specs"
    pgd = json.loads((root / "c92ebd8__pgd_1p05_stabilization_perturbation_responses.json").read_text())
    ofb = json.loads((root / "c92ebd8__pgd_ofb_budget_stabilization_perturbation_responses.json").read_text())
    soft = json.loads((root / "d55c5f0__soft_pgd_stabilization_perturbation_responses.json").read_text())
    assert pgd["figure_count"] == 5
    assert ofb["figure_count"] == soft["figure_count"] == 5
    assert ofb["plot_contract"]["perturbation_event_marker"]["x_axis_reference"] == (
        "seconds relative to perturbation onset"
    )
    assert ofb["plot_contract"]["trace_sources"]["robust_output_feedback6d"] == (
        "6D output-feedback H-infinity"
    )

"""Native stabilization-response figure and parity contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec
import pytest

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

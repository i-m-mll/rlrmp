"""Native nominal-profile figure stock and wrapper-retirement contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

from feedbax.analysis import authenticated_manifest_ref
from feedbax.analysis.figures import execute_figure_spec
from feedbax.analysis.specs import AnalysisRunSpec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import AnalysisRunManifest, spec_payload, write_manifest
import pytest

from rlrmp.figures import register_rlrmp_figure_surfaces


REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS = (
    ("91a090c", "nominal_velocity_profiles"),
    ("c92ebd8", "nominal_velocity_profiles"),
    ("c92ebd8", "pgd_1p05_nominal_velocity_profiles"),
    ("c92ebd8", "pgd_ofb_budget_moderate_nominal_velocity_profiles"),
    ("d55c5f0", "soft_pgd_nominal_velocity_profiles"),
)


def _figure_payload(row_count: int, replicate_count: int) -> dict[str, object]:
    facets = {}
    for row_index in range(row_count):
        series = [
            {
                "label": f"GRU replicate {replicate}",
                "color": "#2563eb",
                "profile": {"time": [0.0, 0.01], "mean": [0.0, 0.2 + replicate * 0.01]},
            }
            for replicate in range(replicate_count)
        ]
        series.extend(
            [
                {
                    "label": "6D extLQG",
                    "color": "#111827",
                    "line_dash": "dash",
                    "profile": {"time": [0.0, 0.01], "mean": [0.0, 0.15]},
                },
                {
                    "label": "6D output-feedback H-infinity",
                    "color": "#dc2626",
                    "line_dash": "dot",
                    "profile": {"time": [0.0, 0.01], "mean": [0.0, 0.18]},
                },
            ]
        )
        facets[f"row-{row_index}"] = {
            "run_id": f"row-{row_index}",
            "display_name": f"Row {row_index}",
            "nominal_velocity": {"series": series},
        }
    return {"facets": {"nominal_velocity_profiles": facets}}


@pytest.mark.parametrize("replicate_count", [2, 3])
def test_all_specs_execute_with_payload_bound_replicates(
    tmp_path: Path,
    replicate_count: int,
) -> None:
    register_rlrmp_figure_surfaces()
    payload = _figure_payload(2, replicate_count)
    analysis = AnalysisRunManifest(
        id=f"nominal-profiles-{replicate_count}",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type="rlrmp.nominal_profile_payload").model_dump(mode="json"),
        ),
        metadata={"figure_payload": payload},
    )
    analysis_path = write_manifest(analysis, root=tmp_path)
    parent = authenticated_manifest_ref(analysis, analysis_path, "profile_analysis")
    for issue, topic in SPECS:
        tracked = FigureSpec.model_validate_json(
            (REPO_ROOT / "results" / issue / "figures" / topic / "spec.json").read_text()
        )
        manifest, path = execute_figure_spec(
            tracked.model_copy(update={"inputs": [parent]}), root=tmp_path, issues=["70fe304"]
        )
        assert manifest.status == "completed"
        assert path.is_file()
        renders = [artifact for artifact in manifest.artifacts if artifact.role == "figure_render"]
        assert renders
        for artifact in renders:
            assert artifact.uri is not None
            assert hashlib.sha256(Path(artifact.uri).read_bytes()).hexdigest() == artifact.sha256

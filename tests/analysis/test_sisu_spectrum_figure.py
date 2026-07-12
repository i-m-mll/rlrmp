"""Native figure-stage contract for the normalized SISU spectrum payload."""

from __future__ import annotations

import json
from pathlib import Path

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

from rlrmp.analysis.sisu_spectrum import SISU_SPECTRUM_ANALYSIS_TYPE
from rlrmp.figures import register_rlrmp_figure_surfaces
from rlrmp.sisu_figures import (
    SISU_FIGURE_PAYLOAD_SCHEMA_VERSION,
    sisu_figure_payload,
    sisu_spectrum_figure_spec,
)
from rlrmp.data_products.envelope import read_data_product


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_SPEC = REPO_ROOT / "src/rlrmp/config/figure_specs/sisu_spectrum.json"

pytestmark = pytest.mark.feedbax_contract


def _analysis_payload(row_count: int) -> dict[str, object]:
    return {
        "summary": {"verified_low_sisu_behavior": "archived parity"},
        "profiles": [
            {
                "run_id": f"run-{index}",
                "label": f"Run {index}",
                "curves": [
                    {
                        "sisu": sisu,
                        "time_s": [0.0, 0.01],
                        "mean_forward_velocity_m_s": [0.0, 0.1 + sisu],
                        "std_forward_velocity_m_s": [0.0, 0.01],
                    }
                    for sisu in (0.0, 0.5, 1.0)
                ],
            }
            for index in range(row_count)
        ],
        "references": [
            {
                "label": "extLQG analytical reference",
                "time_s": [0.0, 0.01],
                "forward_velocity_m_s": [0.0, 0.8],
                "std_forward_velocity_m_s": [0.0, 0.02],
            }
        ],
    }


@pytest.mark.parametrize("row_count", [2, 3])
def test_sisu_profile_cardinality_is_payload_bound(row_count: int) -> None:
    payload = sisu_figure_payload(_analysis_payload(row_count))

    assert payload["schema_version"] == SISU_FIGURE_PAYLOAD_SCHEMA_VERSION
    facets = payload["facets"]["sisu_spectrum_velocity_profiles"]
    assert len(facets) == row_count
    for cell in facets.values():
        series = cell["sisu_spectrum_velocity"]["series"]
        assert len(series) == 4  # three SISU curves plus one analytical reference


def test_tracked_intent_is_native_manifest_bound_and_preserves_parity_oracle() -> None:
    payload = json.loads(TRACKED_SPEC.read_text(encoding="utf-8"))
    spec = FigureSpec.model_validate(payload)

    assert spec.template == "rlrmp.profile_comparison"
    assert spec.assembler is None
    assert spec.facet_bindings["condition"].item == "manifest"
    assert payload["metadata"]["shared_yaxes"] == "all"
    assert (REPO_ROOT / payload["metadata"]["parity_oracle"]).is_file()
    product = read_data_product(REPO_ROOT / payload["metadata"]["parity_product"])
    assert product.product_schema_id == "rlrmp.figure_parity.sisu_spectrum"
    assert product.artifacts[0].sha256 == (
        "f202c09eee90593aa533378c4048b88c28d9ab8d004c479d4e2cbc4458bedc2a"
    )


@pytest.mark.parametrize("row_count", [2, 3])
def test_living_sisu_spec_executes_to_completed_figure_manifest(
    tmp_path: Path,
    row_count: int,
) -> None:
    register_rlrmp_figure_surfaces()
    payload = sisu_figure_payload(_analysis_payload(row_count))
    analysis = AnalysisRunManifest(
        id=f"sisu-analysis-{row_count}",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type=SISU_SPECTRUM_ANALYSIS_TYPE).model_dump(
                mode="json"
            ),
        ),
        metadata={"figure_payload": payload},
    )
    write_manifest(analysis, root=tmp_path)
    parent = ParentRef(kind="AnalysisRunManifest", id=analysis.id, role="sisu_analysis")
    tracked = FigureSpec.model_validate_json(TRACKED_SPEC.read_text(encoding="utf-8"))

    manifest, manifest_path = execute_figure_spec(
        tracked.model_copy(update={"inputs": [parent]}),
        root=tmp_path,
        issues=["518aea3"],
    )

    assert manifest.status == "completed"
    assert manifest_path.is_file()
    assert manifest.resolved_inputs == [parent]
    included = [record for record in manifest.binding_records if record.status == "included"]
    assert len(included) == row_count


def test_figure_adapter_has_no_analysis_or_custody_side_effects() -> None:
    source = (REPO_ROOT / "src/rlrmp/sisu_figures.py").read_text(encoding="utf-8")
    for forbidden in ("plotly", "record_json_artifact", "write_text", "save_figure"):
        assert forbidden not in source
    assert sisu_spectrum_figure_spec().template == "rlrmp.profile_comparison"


def test_retired_direct_builder_package_remains_absent() -> None:
    retired_source = REPO_ROOT / "src/rlrmp/analysis/pipelines"
    assert not any(retired_source.rglob("*.py"))
    assert not (retired_source / "sisu_spectrum_diagnostics.py").exists()
    assert not (
        REPO_ROOT / "tests/analysis/pipelines/test_sisu_spectrum_diagnostics.py"
    ).exists()


def test_living_sisu_figure_surfaces_do_not_reference_retirement_candidate() -> None:
    retired_issue = "e4800" + "d6"
    for path in (
        REPO_ROOT / "src/rlrmp/sisu_figures.py",
        TRACKED_SPEC,
        REPO_ROOT / "tests/analysis/test_sisu_spectrum_figure.py",
    ):
        assert retired_issue not in path.read_text(encoding="utf-8")

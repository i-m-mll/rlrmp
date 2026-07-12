"""Terminal contracts for moderate and calibrated response figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import AnalysisRunManifest, AnalysisRunSpec, ParentRef, spec_payload, write_manifest
import pytest

from rlrmp.analysis.response_norm import response_norm_payload
from rlrmp.data_products.envelope import read_data_product
from rlrmp.figures import register_rlrmp_figure_surfaces


REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY = REPO_ROOT / "results/a778c65/data_products/figure_parity_oracles.json"
SPECS = (
    REPO_ROOT / "results/c92ebd8/figures/moderate_perturbation_profiles/spec.json",
    REPO_ROOT / "results/c92ebd8/figures/pgd_1p05_moderate_perturbation_profiles_overlay/spec.json",
    REPO_ROOT / "results/c92ebd8/figures/pgd_1p05_moderate_perturbation_response_overlays/spec.json",
    REPO_ROOT / "results/3244f1a/figures/final_calibrated_bank_profiles/small/spec.json",
    REPO_ROOT / "results/3244f1a/figures/final_calibrated_bank_profiles/medium/spec.json",
)


pytestmark = pytest.mark.feedbax_contract


def _rows(model_count: int) -> list[dict]:
    rows = []
    for metric in ("position", "velocity"):
        for condition in ("moderate", "calibrated"):
            for index in range(model_count):
                rows.append({
                    "row_id": f"{metric}-{condition}-{index}",
                    "model_id": f"model-{index}",
                    "metric": metric,
                    "condition_class": condition,
                    "time": [0.0, 0.01],
                    "mean": [0.0, 0.2 + index * 0.1],
                    "sem": [0.0, 0.01],
                    "label": f"Model {index}",
                })
    return rows


def test_all_five_specs_are_independent_manifest_bound_leaves() -> None:
    names = set()
    for path in SPECS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        spec = FigureSpec.model_validate(raw)
        assert spec.template == "rlrmp.response_norm_comparison"
        assert not spec.inputs
        assert all(binding.item == "manifest" for binding in spec.facet_bindings.values())
        assert raw["metadata"]["independent_leaf"] is True
        assert "source_spec" not in raw
        assert "source_figure" not in raw
        names.add(spec.name)
    assert len(names) == 5


@pytest.mark.parametrize("model_count", [2, 3])
def test_all_specs_real_execute_with_data_bound_cardinality(tmp_path: Path, model_count: int) -> None:
    register_rlrmp_figure_surfaces()
    manifest = AnalysisRunManifest(
        id=f"moderate-calibrated-payload-{model_count}",
        status="completed",
        analysis_spec=spec_payload("AnalysisRunSpec", AnalysisRunSpec(analysis_type="rlrmp.response_norm_comparison").model_dump(mode="json")),
        metadata={"figure_payload": response_norm_payload(_rows(model_count), metrics=("position", "velocity"), condition_classes=("moderate", "calibrated"))},
    )
    write_manifest(manifest, root=tmp_path)
    parent = ParentRef(kind="AnalysisRunManifest", id=manifest.id, role="rlrmp-response-norm-comparison-payload")
    rendered = []
    for path in SPECS:
        spec = FigureSpec.model_validate_json(path.read_text(encoding="utf-8"))
        output, manifest_path = execute_figure_spec(spec.model_copy(update={"inputs": [parent]}), root=tmp_path, issues=["a778c65"])
        assert output.status == "completed"
        assert output.resolved_inputs == [parent]
        assert manifest_path.is_file()
        artifacts = [artifact for artifact in output.artifacts if artifact.role == "figure_render"]
        assert artifacts
        for artifact in artifacts:
            assert artifact.uri is not None
            render = Path(artifact.uri)
            assert hashlib.sha256(render.read_bytes()).hexdigest() == artifact.sha256
        rendered.append(output)
    assert len({item.id for item in rendered}) == 5


def test_archived_specs_are_hash_verified_governed_parity() -> None:
    product = read_data_product(PARITY)
    assert product.product_schema_id == "rlrmp.figure_parity_oracles"
    assert len(product.artifacts) == 5
    for artifact in product.artifacts:
        assert artifact.uri is not None
        path = REPO_ROOT / artifact.uri
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256


def test_no_figure_consumes_a_figure_or_candidate_stock() -> None:
    for path in SPECS:
        raw = path.read_text(encoding="utf-8")
        assert "FigureManifest" not in raw
        assert "source_spec" not in raw
        assert "source_figure" not in raw
        assert "results/c92ebd8/" not in raw
        assert "results/3244f1a/" not in raw

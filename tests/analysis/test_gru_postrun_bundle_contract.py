"""Terminal contract checks for the declarative GRU post-run bundle."""

import hashlib
import json
from pathlib import Path

from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ParentRef,
    spec_payload,
    write_manifest,
)
import pytest
from ruamel.yaml import YAML

from rlrmp.analysis.response_norm import response_norm_payload
from rlrmp.data_products.envelope import read_data_product
from rlrmp.figures import register_rlrmp_figure_surfaces


BUNDLE_PATH = Path("src/rlrmp/config/analysis_bundles/gru_postrun.yml")
PARITY_PATH = Path("results/74fac80/data_products/gru_postrun_figure_parity.json")


pytestmark = pytest.mark.feedbax_contract


def _bundle() -> dict[str, object]:
    return YAML(typ="safe").load(BUNDLE_PATH.read_text(encoding="utf-8"))


def test_gru_postrun_bundle_uses_canonical_stage_contract() -> None:
    bundle = _bundle()

    assert bundle["metadata"]["primary_contract"] == "feedbax_analysis_bundle"
    assert all(
        stage["kind"] in {"analysis", "evaluation", "figure", "materialization", "report"}
        for stage in bundle["stages"]
    )


def test_gru_postrun_stage_graph_is_unique_and_topologically_ordered() -> None:
    bundle = _bundle()
    stages = bundle["stages"]
    stage_positions = {stage["name"]: index for index, stage in enumerate(stages)}

    assert len(stage_positions) == len(stages)
    for stage in stages:
        assert all(
            stage_positions[dependency] < stage_positions[stage["name"]]
            for dependency in stage.get("depends_on", [])
        )


def test_gru_postrun_reports_resolve_to_canonical_component_roles() -> None:
    bundle = _bundle()
    stages = {stage["name"]: stage for stage in bundle["stages"]}

    for stage in stages.values():
        for dependency in stage.get("depends_on", []):
            assert dependency in stages

    reports = [stage for stage in stages.values() if stage["kind"] == "report"]
    assert reports
    for report in reports:
        assert report["depends_on"]
        assert "report_render" in {output["role"] for output in report["outputs"]}

    roles = {
        output["role"]
        for stage in stages.values()
        for output in stage.get("outputs", [])
    }
    assert {
        "rlrmp-bridge-standard-certificate",
        "rlrmp-gru-perturbation-response-manifest",
        "rlrmp-feedback-quality-objective-comparator-status",
        "rlrmp-feedback-quality-lens",
        "report_render",
    } <= roles


def test_gru_postrun_bundle_retains_scientific_not_applicable_canaries() -> None:
    bundle = _bundle()
    stages = {stage["name"]: stage for stage in bundle["stages"]}

    for sibling_stage in (
        "robustness_phenotype_aggregation",
        "output_feedback_bridge_diagnostics",
        "training_diagnostics",
    ):
        assert stages[sibling_stage]["not_applicable_reason"]


def test_gru_postrun_figure_is_a_leaf_of_canonical_manifest_stages() -> None:
    bundle = _bundle()
    stages = {stage["name"]: stage for stage in bundle["stages"]}
    payload = stages["response_norm_payload"]
    figure = stages["response_norm_figure"]

    assert payload["kind"] == "analysis"
    assert payload["analysis_type"] == "rlrmp.response_norm_comparison"
    assert payload["depends_on"] == ["perturbation_bank_aggregate"]
    assert figure["kind"] == "figure"
    assert figure["depends_on"] == ["response_norm_payload"]
    assert figure["figure"]["template"] == "rlrmp.response_norm_comparison"
    assert figure["figure"]["metadata"]["parity_oracle"] == str(PARITY_PATH)


def test_gru_postrun_figure_executes_to_hash_verified_manifest(tmp_path: Path) -> None:
    register_rlrmp_figure_surfaces()
    bundle = _bundle()
    stage = next(stage for stage in bundle["stages"] if stage["name"] == "response_norm_figure")
    figure = FigureSpec.model_validate(stage["figure"])
    rows = []
    for metric in ("delta_position", "delta_action"):
        for condition in ("class_a", "class_b"):
            rows.append(
                {
                    "row_id": f"gru-{metric}-{condition}",
                    "model_id": "gru-validation-selected",
                    "metric": metric,
                    "condition_class": condition,
                    "time": [0.0, 0.01],
                    "mean": [0.0, 0.2],
                    "sem": [0.0, 0.01],
                    "label": "GRU validation-selected",
                }
            )
    manifest = AnalysisRunManifest(
        id="gru-postrun-response-norm-payload",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type="rlrmp.response_norm_comparison").model_dump(
                mode="json"
            ),
        ),
        metadata={"figure_payload": response_norm_payload(rows)},
    )
    write_manifest(manifest, root=tmp_path)
    parent = ParentRef(
        kind="AnalysisRunManifest",
        id=manifest.id,
        role="rlrmp-response-norm-comparison-payload",
    )
    rendered, manifest_path = execute_figure_spec(
        figure.model_copy(update={"inputs": [parent]}),
        root=tmp_path,
        issues=["74fac80"],
    )

    assert rendered.status == "completed"
    assert rendered.resolved_inputs == [parent]
    assert manifest_path.is_file()
    artifacts = [artifact for artifact in rendered.artifacts if artifact.role == "figure_render"]
    assert artifacts
    for artifact in artifacts:
        assert artifact.uri is not None
        path = Path(artifact.uri)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256


def test_archived_pilot_outputs_are_governed_parity_oracles() -> None:
    product = read_data_product(PARITY_PATH)
    assert product.product_schema_id == "rlrmp.figure_parity_oracles"
    assert len(product.artifacts) == 2
    for artifact in product.artifacts:
        assert artifact.uri is not None
        assert hashlib.sha256(Path(artifact.uri).read_bytes()).hexdigest() == artifact.sha256
    raw = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    assert raw["parameters"]["legacy_alias_outputs"] is False
    assert raw["parameters"]["legacy_figure_summary_writer"] is False

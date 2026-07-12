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
from ruamel.yaml import YAML

from rlrmp.analysis.response_norm import response_norm_payload
from rlrmp.data_products.envelope import read_data_product
from rlrmp.figures import register_rlrmp_figure_surfaces


BUNDLE_PATH = Path("src/rlrmp/config/analysis_bundles/gru_postrun.yml")
PARITY_PATH = Path("results/6c36536/data_products/gru_postrun_figure_parity.json")


def _bundle() -> tuple[str, dict[str, object]]:
    source = BUNDLE_PATH.read_text(encoding="utf-8")
    return source, YAML(typ="safe").load(source)


def test_gru_postrun_bundle_has_no_retired_dispatcher_contract() -> None:
    source, bundle = _bundle()

    retired_strings = (
        "rlrmp.gru_postrun",
        "gru_postrun_materialization",
        "legacy_regeneration",
        "diagnostic_provenance",
        "analysis.pipelines.gru_postrun_materialization",
        "archive_only_entrypoints",
    )
    assert all(retired not in source for retired in retired_strings)
    assert "legacy_regeneration_contract" not in bundle["metadata"]


def test_gru_postrun_reports_resolve_to_canonical_component_roles() -> None:
    _, bundle = _bundle()
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
    _, bundle = _bundle()
    stages = {stage["name"]: stage for stage in bundle["stages"]}

    for sibling_stage in (
        "robustness_phenotype_aggregation",
        "output_feedback_bridge_diagnostics",
        "training_diagnostics",
    ):
        assert stages[sibling_stage]["not_applicable_reason"]


def test_gru_postrun_figure_is_a_leaf_of_canonical_manifest_stages() -> None:
    source, bundle = _bundle()
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
    for retired in (
        "figure_summary",
        "forward_velocity_profiles_stochastic_with_extlqg",
        "materialize_gru_pilot_figures",
    ):
        assert retired not in source


def test_gru_postrun_figure_executes_to_hash_verified_manifest(tmp_path: Path) -> None:
    register_rlrmp_figure_surfaces()
    _, bundle = _bundle()
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


def test_pilot_pipeline_helpers_wrappers_and_benchmark_are_terminally_retired() -> None:
    retired = (
        "src/rlrmp/analysis/pipelines/gru_pilot_figures.py",
        "src/rlrmp/analysis/pipelines/gru_postrun_materialization.py",
        "scripts/materialize_gru_pilot_figures.py",
        "scripts/materialize_gru_postrun.py",
        "src/rlrmp/benchmarks/postrun_eval_materialization.py",
        "tests/analysis/pipelines/test_gru_pilot_figures.py",
        "tests/analysis/pipelines/test_gru_postrun_materialization.py",
    )
    assert all(not Path(path).exists() for path in retired)
    trial_inputs = Path("src/rlrmp/eval/trial_inputs.py").read_text(encoding="utf-8")
    velocity_profiles = Path("src/rlrmp/eval/velocity_profiles.py").read_text(encoding="utf-8")
    assert "def resolve_evaluation_run_inputs" in trial_inputs
    assert "initial_effector_velocity" in velocity_profiles

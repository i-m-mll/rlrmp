"""Terminal contract checks for the declarative GRU post-run bundle."""

from pathlib import Path

from ruamel.yaml import YAML


BUNDLE_PATH = Path("src/rlrmp/config/analysis_bundles/gru_postrun.yml")


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

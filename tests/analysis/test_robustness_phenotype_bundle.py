"""Tests for the robustness phenotype analysis bundle."""

from __future__ import annotations

import json
from pathlib import Path

import rlrmp
from feedbax.analysis.bundles import execute_staged_analysis_bundle, load_analysis_bundle
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry

from rlrmp.analysis import declarative_materialization as dm
from rlrmp.runtime.spec_migrations import (
    HINF_PHENOTYPE_SIDECAR_KIND,
    HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
)


def _registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    return registry


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(role: str, path: Path) -> ArtifactRef:
    return ArtifactRef(
        role=role,
        logical_name=path.name,
        uri=str(path),
        media_type="application/json",
    )


def _write_gru_postrun_analysis_manifest(
    root: Path,
    source_dir: Path,
) -> AnalysisRunManifest:
    objective = _write_json(
        source_dir / "objective_comparator.json",
        {
            "schema_version": "rlrmp.objective_comparator_sidecar.v6",
            "rows": [
                {
                    "run_id": "robust_perturb_fullqrf",
                    "n_replicates": 2,
                    "gru_mean_selected_validation_full_qrf": 12.0,
                    "shared_rollout_comparator": {
                        "gru_vs_extlqg": {
                            "terms": {"total": {"ratio_to_extlqg": 1.25}}
                        }
                    },
                }
            ],
        },
    )
    perturbation = _write_json(
        source_dir / "perturbation_response.json",
        {
            "schema_version": "rlrmp.gru_perturbation_bank.v3",
            "runs": {
                "robust_perturb_fullqrf": {
                    "status": "available",
                    "robust_response_summary": {
                        "class_summary": {
                            "groups": {
                                "process_epsilon/force": {
                                    "n_rows": 1,
                                    "status_counts": {"evaluated": 1},
                                    "metrics": {
                                        "delta_action_norm": {"mean": 0.2},
                                        "delta_endpoint_error_m": {"mean": 0.01},
                                    },
                                    "gru_extlqg_delta_cost_ratio": {
                                        "status": "available",
                                        "ratio_of_means": 0.9,
                                    },
                                }
                            }
                        }
                    },
                }
            },
        },
    )
    broad_epsilon = _write_json(
        source_dir / "broad_epsilon_attribution.json",
        {
            "schema_version": "rlrmp.gru_broad_epsilon_attribution.v1",
            "rows": [{"run_id": "robust_perturb_fullqrf", "status": "available"}],
        },
    )
    manifest = AnalysisRunManifest(
        id="rlrmp-test-analysis:gru-postrun-for-phenotype",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(
                analysis_type=dm.GRU_POSTRUN_ANALYSIS_TYPE,
                params={"experiment": "unitexp"},
            ).model_dump(mode="json"),
        ),
        artifacts=[
            _artifact("rlrmp-gru-objective-comparator-manifest", objective),
            _artifact("rlrmp-gru-perturbation-response-manifest", perturbation),
            _artifact("rlrmp-gru-broad-epsilon-attribution-manifest", broad_epsilon),
        ],
        metadata={"bundle": {"name": "gru_postrun"}},
    )
    write_manifest(manifest, root=root)
    return manifest


def test_robustness_phenotype_bundle_executes_with_status_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = _registry()
    bundle = load_analysis_bundle("rlrmp/robustness_phenotype", registry=registry)
    upstream = _write_gru_postrun_analysis_manifest(
        tmp_path / "feedbax_runs",
        tmp_path / "source_payloads",
    )
    monkeypatch.setattr(dm, "REPO_ROOT", tmp_path / "repo")

    result = execute_staged_analysis_bundle(
        bundle,
        root=tmp_path / "feedbax_runs",
        run_ids=[upstream.id],
        issues=["769aea6"],
        fig_dump_formats=("json",),
    )

    assert result.bundle_name == "robustness_phenotype"
    assert result.matched_run_ids == [upstream.id]
    stages = {stage.name: stage for stage in result.stages}
    phenotype_stage = stages["phenotype_sidecar"]
    output_statuses = {output.role: output.status for output in phenotype_stage.outputs}
    assert output_statuses["manifest"] == "materialized"
    assert output_statuses["rlrmp-robustness-phenotype-sidecar"] == "materialized"
    assert output_statuses["rlrmp-formal-hinf-certificate"] == "missing"

    report_stage = stages["phenotype_report"]
    report_statuses = {output.role: output.status for output in report_stage.outputs}
    assert report_statuses["report_render"] == "materialized"
    report_manifest = load_manifest(report_stage.manifest_refs[0].uri)
    render_artifact = next(
        artifact for artifact in report_manifest.artifacts if artifact.role == "report_render"
    )
    render_text = Path(render_artifact.uri).read_text(encoding="utf-8")
    assert "# Robustness Phenotype Report" in render_text
    assert "Feedbax-custody report render" in render_text

    formal_stage = stages["formal_hinf_certificate"]
    assert formal_stage.outputs[0].status == "not_applicable"
    assert "interpretive only" in formal_stage.outputs[0].reason
    archive_stage = stages["historical_script_entrypoint"]
    assert archive_stage.outputs[0].status == "skipped"

    manifest_ref = phenotype_stage.manifest_refs[0]
    manifest = load_manifest(manifest_ref.uri)
    roles = {artifact.role for artifact in manifest.artifacts}
    assert roles == {"rlrmp-robustness-phenotype-sidecar"}
    payload_artifact = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "rlrmp-robustness-phenotype-sidecar"
    )
    payload = json.loads(Path(payload_artifact.uri).read_text(encoding="utf-8"))
    assert payload["schema_version"] == HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION
    assert payload["issue"] == "769aea6"
    assert payload["components"]["objective_comparator"]["status"] == "available"
    assert payload["components"]["broad_epsilon_attribution"]["status"] == "available"
    assert payload["components"]["exact_audit"]["status"] == "missing"
    row = payload["rows"][0]
    assert row["formal_hinf_claim"]["status"] == "not_claimed"
    assert payload["interpretation_contract"]["not_standard_certificate"] is True
    assert payload["interpretation_contract"]["formal_hinf_requirements"] == [
        "game_card",
        "gamma_or_budget",
        "disturbance_channel",
        "exact_audit_or_induced_gain",
    ]
    accepted = accept_rlrmp_spec_payload(HINF_PHENOTYPE_SIDECAR_KIND, payload)
    assert accepted.target_version == HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION

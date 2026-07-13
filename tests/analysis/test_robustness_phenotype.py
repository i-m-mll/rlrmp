"""Tests for the GRU H-infinity phenotype sidecar aggregator."""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from rlrmp.analysis.robustness_phenotype import (
    PHENOTYPE_PARENT_ANALYSIS_TYPES,
    SCHEMA_VERSION,
    build_hinf_phenotype_sidecar,
    robustness_phenotype_recipe,
)


def _source(path: str, payload: dict[str, object]) -> dict[str, object]:
    return {"status": "available", "source_path": path, "payload": payload}


def test_sidecar_aggregates_available_components_with_provenance() -> None:
    sources = {
        "objective_comparator": _source(
            "results/unit/objective.json",
            {
                "schema_version": "objective.v1",
                "issue": "unit",
                "rows": [
                    {
                        "run_id": "baseline_fullqrf",
                        "n_replicates": 3,
                        "gru_mean_selected_validation_full_qrf": 10.0,
                        "selected_to_extlqg_deterministic_ratio": 1.1,
                    },
                    {
                        "run_id": "robust_perturb_fullqrf",
                        "n_replicates": 3,
                        "gru_mean_selected_validation_full_qrf": 12.0,
                        "shared_rollout_comparator": {
                            "gru_vs_extlqg": {"terms": {"total": {"ratio_to_extlqg": 1.4}}}
                        },
                    },
                ],
            },
        ),
        "perturbation_response": _source(
            "results/unit/perturb.json",
            {
                "schema_version": "perturb.v1",
                "runs": {
                    "robust_perturb_fullqrf": {
                        "status": "available",
                        "robust_response_summary": {
                            "class_summary": {
                                "groups": {
                                    "process_epsilon/force": {
                                        "n_rows": 2,
                                        "status_counts": {"evaluated": 2},
                                        "metrics": {
                                            "delta_action_norm": {"mean": 0.2},
                                            "delta_endpoint_error_m": {"mean": 0.01},
                                            "delta_velocity_trajectory_norm_m_s": {"mean": 0.4},
                                        },
                                        "gru_extlqg_delta_cost_ratio": {
                                            "status": "available",
                                            "ratio_of_means": 0.8,
                                        },
                                    }
                                }
                            }
                        },
                    }
                },
            },
        ),
        "feedback_ablation": _source(
            "results/unit/feedback.json",
            {
                "schema_version": "feedback.v1",
                "runs": {
                    "robust_perturb_fullqrf": {
                        "status": "available",
                        "status_counts": {"evaluated": 4},
                        "interpretation": {
                            "label": "feedback_sensitive",
                            "max_feedback_delta_action_norm_mean": 0.5,
                        },
                        "ablations": [
                            {
                                "mode": "normal",
                                "bin": "nominal",
                                "metrics": {
                                    "endpoint_error_m": {"mean": 0.003},
                                    "terminal_speed_m_s": {"mean": 0.001},
                                    "rollout_full_qrf": {"base_cost": {"control": {"mean": 2.0}}},
                                },
                            }
                        ],
                    }
                },
            },
        ),
        "map_error_decomposition": _source(
            "results/unit/map.json",
            {
                "format": "map.v1",
                "rows": [
                    {
                        "source_run_id": "robust_perturb_fullqrf",
                        "decomposition": {
                            "summary": {
                                "aggregate_delta_ratio": 0.4,
                                "candidate_reference_cosine": 0.8,
                                "best_scalar_gain": 1.2,
                            },
                            "decision_rule_annotations": ["well_excited_residual"],
                        },
                    }
                ],
            },
        ),
    }

    sidecar = build_hinf_phenotype_sidecar(sources=sources)

    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["components"]["objective_comparator"]["source_path"] == (
        "results/unit/objective.json"
    )
    robust = {row["run_id"]: row for row in sidecar["rows"]}["robust_perturb_fullqrf"]
    assert robust["nominal_efficiency"]["values"]["n_replicates"] == 3
    assert robust["nominal_efficiency"]["values"]["endpoint_error_m"] == 0.003
    assert (
        robust["feedback_competence"]["values"]["feedback_ablation_interpretation"]["label"]
        == "feedback_sensitive"
    )
    assert (
        robust["local_feedback_law"]["values"]["map_decomposition_summary"][
            "candidate_reference_cosine"
        ]
        == 0.8
    )
    assert robust["paired_baseline_vs_robust"]["status"] == "candidate_pair_available"
    assert robust["formal_hinf_claim"]["status"] == "not_claimed"
    assert any(warning["code"] == "formal_hinf_not_claimed" for warning in robust["warnings"])


def test_sidecar_degrades_gracefully_when_components_are_missing() -> None:
    sources = {
        "objective_comparator": _source(
            "results/unit/objective.json",
            {
                "schema_version": "objective.v1",
                "rows": [{"run_id": "run_a", "gru_mean_selected_validation_full_qrf": 1.0}],
            },
        ),
        "perturbation_response": None,
    }

    sidecar = build_hinf_phenotype_sidecar(sources=sources)
    row = sidecar["rows"][0]

    assert sidecar["components"]["perturbation_response"]["status"] == "missing"
    assert row["feedback_competence"]["status"] == "missing"
    assert row["hinf_phenotype_markers"]["status"] == "missing"
    assert row["formal_hinf_claim"]["status"] == "not_claimed"


def test_sidecar_reads_sisu_perturbation_comparison_schema() -> None:
    run_id = "delayed_sisu_spectrum__raw"
    sources = {
        "evaluation_diagnostics": _source(
            "results/abc1234/notes/gru_eval.json",
            {
                "schema_version": "rlrmp.gru_evaluation_diagnostics.v1",
                "runs": {
                    run_id: {
                        "behavior": {
                            "endpoint_error_m": {"mean": 0.003},
                            "terminal_speed_m_s": {"mean": 0.002},
                            "velocity_profile": {
                                "mean_profile_peak_forward_velocity_m_s": 0.75,
                                "mean_profile_time_to_peak_forward_velocity_s": 0.16,
                            },
                        },
                    }
                },
            },
        ),
        "perturbation_response": _source(
            "results/abc1234/notes/sisu_perturbation.json",
            {
                "schema_version": "rlrmp.sisu_perturbation_class_comparison.v1",
                "issue": "abc1234",
                "runs": {
                    run_id: {
                        "label": "raw",
                        "headline": {
                            "full_qrf_delta_cost": {"improved": 2, "equal": 0, "worse": 1},
                            "max_delta_x_m": {"improved": 3, "equal": 0, "worse": 0},
                            "mean_delta_action": {"improved": 0, "equal": 0, "worse": 3},
                        },
                        "class_comparison": {
                            "command_input/command_input_pulse": {
                                "rows_sisu_0": 4,
                                "rows_sisu_1": 4,
                                "status_counts_sisu_0": {"evaluated": 4},
                                "status_counts_sisu_1": {"evaluated": 4},
                                "metrics": {
                                    "mean_delta_action": {"ratio_1_over_0": 1.2},
                                    "max_delta_x_m": {"ratio_1_over_0": 0.7},
                                    "auc_delta_x_m_s": {"ratio_1_over_0": 0.8},
                                    "mean_full_qrf_delta_cost": {
                                        "ratio_1_over_0": 0.5,
                                        "delta_1_minus_0": -2.0,
                                    },
                                },
                            }
                        },
                    }
                },
            },
        )
    }

    sidecar = build_hinf_phenotype_sidecar(sources=sources)
    row = sidecar["rows"][0]

    assert row["run_id"] == run_id
    assert row["nominal_efficiency"]["values"]["endpoint_error_m"] == 0.003
    assert row["nominal_efficiency"]["values"]["peak_velocity_m_s"] == 0.75
    assert row["feedback_competence"]["status"] == "available"
    assert row["hinf_phenotype_markers"]["status"] == "available"
    markers = row["hinf_phenotype_markers"]["values"]["sisu_1_vs_0_perturbation_markers"]
    assert markers["full_qrf_delta_cost"]["improved"] == 2
    summary = row["feedback_competence"]["values"]["sisu_1_vs_0_perturbation_class_summary"]
    assert summary["command_input/command_input_pulse"][
        "full_qrf_delta_cost_ratio_1_over_0"
    ] == 0.5


def test_sidecar_uses_explicit_paired_run_ids_for_opaque_run_names() -> None:
    sources = {
        "objective_comparator": _source(
            "results/unit/objective.json",
            {
                "schema_version": "objective.v1",
                "rows": [
                    {"run_id": "row_a", "gru_mean_selected_validation_full_qrf": 1.0},
                    {"run_id": "row_b", "gru_mean_selected_validation_full_qrf": 2.0},
                ],
            },
        ),
    }

    sidecar = build_hinf_phenotype_sidecar(
        sources=sources,
        paired_run_ids={"row_a": "row_b"},
    )
    rows = {row["run_id"]: row for row in sidecar["rows"]}

    assert sidecar["paired_run_ids"] == {"row_a": "row_b"}
    assert rows["row_a"]["paired_baseline_vs_robust"] == {
        "status": "candidate_pair_available",
        "baseline_run_id": "row_a",
        "robust_run_id": "row_b",
        "current_row_role": "baseline",
        "selection_role": "interpretive_pairing_only",
        "pairing_source": "explicit_paired_run_ids",
        "current_row_evidence_statuses": {
            "nominal_efficiency": "available",
            "feedback_competence": "missing",
            "local_feedback_law": "missing",
            "hinf_phenotype_markers": "missing",
        },
    }
    assert rows["row_b"]["paired_baseline_vs_robust"]["current_row_role"] == "robust"


def test_recipe_declares_parents_and_bundle_routes_analysis_to_report_render() -> None:
    assert robustness_phenotype_recipe.ANALYSIS_DEPENDENCIES == PHENOTYPE_PARENT_ANALYSIS_TYPES
    bundle_path = (
        Path(__file__).parents[2]
        / "src/rlrmp/config/analysis_bundles/robustness_phenotype.yml"
    )
    bundle = YAML(typ="safe").load(bundle_path.read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in bundle["stages"]}
    assert stages["phenotype_report"]["depends_on"] == ["phenotype_sidecar"]
    assert stages["phenotype_report"]["report_type"] == (
        "rlrmp.report.robustness_phenotype_markdown"
    )
    assert stages["phenotype_report"]["local_params"]["source_artifact_roles"] == [
        "rlrmp-robustness-phenotype-sidecar"
    ]
    assert "report_render" in {
        output["role"] for output in stages["phenotype_report"]["outputs"]
    }

"""Tests for the GRU H-infinity phenotype sidecar aggregator."""

from __future__ import annotations

import json

from rlrmp.analysis.hinf_phenotype_sidecar import (
    SCHEMA_VERSION,
    build_hinf_phenotype_sidecar,
    load_hinf_phenotype_sources,
    render_hinf_phenotype_markdown,
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
                            "gru_vs_extlqg": {
                                "terms": {"total": {"ratio_to_extlqg": 1.4}}
                            }
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
                                            "delta_velocity_trajectory_norm_m_s": {
                                                "mean": 0.4
                                            },
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
                                    "rollout_full_qrf": {
                                        "base_cost": {"control": {"mean": 2.0}}
                                    },
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
    robust = {
        row["run_id"]: row
        for row in sidecar["rows"]
    }["robust_perturb_fullqrf"]
    assert robust["nominal_efficiency"]["values"]["n_replicates"] == 3
    assert robust["nominal_efficiency"]["values"]["endpoint_error_m"] == 0.003
    assert robust["feedback_competence"]["values"]["feedback_ablation_interpretation"][
        "label"
    ] == "feedback_sensitive"
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


def test_loader_records_missing_path_and_markdown_renders(tmp_path) -> None:
    present = tmp_path / "objective.json"
    present.write_text(
        json.dumps({"schema_version": "objective.v1", "rows": [{"run_id": "run_a"}]}),
        encoding="utf-8",
    )
    missing = tmp_path / "missing.json"

    sources = load_hinf_phenotype_sources(
        {
            "objective_comparator": present,
            "standard_certificate": missing,
        },
        repo_root=tmp_path,
    )
    sidecar = build_hinf_phenotype_sidecar(sources=sources)
    markdown = render_hinf_phenotype_markdown(sidecar)

    assert sources["standard_certificate"]["status"] == "missing"
    assert sources["standard_certificate"]["source_path"] == "missing.json"
    assert "not a standard certificate" in markdown
    assert "| run_a |" in markdown

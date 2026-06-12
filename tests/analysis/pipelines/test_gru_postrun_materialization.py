"""Tests for one-command GRU post-run materialization orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rlrmp.analysis import gru_postrun_materialization as postrun


def test_plan_gru_postrun_materialization_routes_tracked_and_bulk_outputs(
    tmp_path: Path,
) -> None:
    plan = postrun.plan_gru_postrun_materialization(
        experiment="5f70333",
        run_ids=("run_a", "run_b"),
        output_tag="fullqrf_validation_selected",
        repo_root=tmp_path,
    )

    assert plan.checkpoint_policy == "validation_selected_per_replicate"
    assert plan.checkpoint_selection_source == "validation_selected_per_replicate"
    assert plan.checkpoint_manifest_path == (
        tmp_path / "results" / "5f70333" / "notes" / "validation_selected_checkpoints.json"
    )
    assert plan.fixed_bank_rescore_manifest_path == (
        tmp_path / "results" / "5f70333" / "notes" / "fixed_bank_rescored_checkpoints.json"
    )
    assert plan.standard_manifest_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_standard_certificates_fullqrf_validation_selected_manifest.json"
    )
    assert plan.evaluation_manifest_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_evaluation_diagnostics_fullqrf_validation_selected.json"
    )
    assert plan.figure_output_dir == (
        tmp_path
        / "_artifacts"
        / "5f70333"
        / "figures"
        / "gru_postrun_fullqrf_validation_selected"
    )
    assert plan.evaluation_bulk_dir == (
        tmp_path
        / "_artifacts"
        / "5f70333"
        / "evaluation_diagnostics"
        / "gru_fullqrf_validation_selected"
    )
    assert plan.map_decomposition_json_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_map_error_decomposition_fullqrf_validation_selected.json"
    )
    assert plan.map_decomposition_note_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_map_error_decomposition_fullqrf_validation_selected.md"
    )
    assert plan.perturbation_response_json_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_perturbation_response_fullqrf_validation_selected_manifest.json"
    )
    assert plan.perturbation_response_note_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_perturbation_response_fullqrf_validation_selected.md"
    )
    assert plan.perturbation_response_bulk_dir == (
        tmp_path
        / "_artifacts"
        / "5f70333"
        / "perturbation_response"
        / "gru_fullqrf_validation_selected"
    )
    assert plan.feedback_ablation_json_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_feedback_ablation_fullqrf_validation_selected.json"
    )
    assert plan.feedback_ablation_note_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_feedback_ablation_fullqrf_validation_selected.md"
    )
    assert plan.postrun_regeneration_spec_path == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_postrun_materialization_fullqrf_validation_selected_regeneration_spec.json"
    )


def test_materialize_gru_postrun_analysis_passes_validation_selection_to_materializers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_checkpoint_manifest(**kwargs: Any) -> dict[str, Any]:
        calls["checkpoint"] = kwargs
        kwargs["output_path"].write_text("{}", encoding="utf-8")
        return {"checkpoint_policy": "validation_selected_per_replicate"}

    def fake_standard_result(**kwargs: Any) -> dict[str, Any]:
        calls["standard"] = kwargs
        return {"summary": {"n_rows": 2}}

    def fake_write_standard(
        result: dict[str, Any],
        *,
        note_path: Path,
        manifest_path: Path,
        regeneration_spec_path: Path,
        repo_root: Path,
    ) -> None:
        calls["standard_write"] = {
            "result": result,
            "note_path": note_path,
            "manifest_path": manifest_path,
            "regeneration_spec_path": regeneration_spec_path,
            "repo_root": repo_root,
        }
        note_path.write_text("# standard\n", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")

    def fake_evaluation(**kwargs: Any) -> dict[str, Any]:
        calls["evaluation"] = kwargs
        kwargs["output_path"].write_text("{}", encoding="utf-8")
        return {"schema_version": "rlrmp.gru_evaluation_diagnostics.v1"}

    def fake_figures(**kwargs: Any) -> dict[str, Any]:
        calls["figures"] = kwargs
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "figure_summary.json").write_text("{}", encoding="utf-8")
        return {"checkpoint_policy": "validation_selected_per_replicate"}

    def fake_objective(**kwargs: Any) -> dict[str, Any]:
        calls["objective"] = kwargs
        return {
            "status": "materialized",
            "json_path": (
                "results/5f70333/notes/"
                "objective_comparator_fullqrf_validation_selected.json"
            ),
            "note_path": (
                "results/5f70333/notes/"
                "objective_comparator_fullqrf_validation_selected.md"
            ),
            "result": {
                "schema_version": "rlrmp.objective_comparator_sidecar.v6",
                "standard_split_bank_comparator_status": "available",
            },
        }

    def fake_map_decomposition(**kwargs: Any) -> dict[str, Any]:
        calls["map"] = kwargs
        return {
            "status": "materialized",
            "json_path": (
                "results/5f70333/notes/"
                "gru_map_error_decomposition_fullqrf_validation_selected.json"
            ),
            "note_path": (
                "results/5f70333/notes/"
                "gru_map_error_decomposition_fullqrf_validation_selected.md"
            ),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "result": {
                "schema_version": "rlrmp.gru_map_error_decomposition.v1",
                "n_rows": 2,
            },
        }

    def fake_perturbation_response(**kwargs: Any) -> dict[str, Any]:
        calls["perturbation"] = kwargs
        return {
            "status": "materialized",
            "json_path": (
                "results/5f70333/notes/"
                "gru_perturbation_response_fullqrf_validation_selected_manifest.json"
            ),
            "note_path": (
                "results/5f70333/notes/"
                "gru_perturbation_response_fullqrf_validation_selected.md"
            ),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "result": {
                "schema_version": "rlrmp.gru_perturbation_response.v2",
                "n_runs": 2,
                "n_perturbations": 12,
            },
        }

    def fake_feedback_ablation(**kwargs: Any) -> dict[str, Any]:
        calls["feedback"] = kwargs
        return {
            "status": "materialized",
            "json_path": (
                "results/5f70333/notes/"
                "gru_feedback_ablation_fullqrf_validation_selected.json"
            ),
            "note_path": (
                "results/5f70333/notes/"
                "gru_feedback_ablation_fullqrf_validation_selected.md"
            ),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "result": {
                "schema_version": "rlrmp.gru_feedback_ablation.v1",
                "n_runs": 2,
                "checkpoint_policy": "validation_selected_per_replicate",
                "feedback_checkpoint_selection_audit_status": "available",
            },
            "feedback_checkpoint_selection_audit": {
                "schema_version": "rlrmp.gru_feedback_checkpoint_selection_audit.v1",
                "status": "available",
                "selection_use": "audit_only_not_primary_checkpoint_selection",
                "primary_checkpoint_policy": "validation_selected_per_replicate",
                "selected_candidate": {"run_id": "run_b", "feedback_score": 0.7},
            },
        }

    monkeypatch.setattr(
        postrun,
        "materialize_validation_selected_checkpoint_manifest",
        fake_checkpoint_manifest,
    )
    monkeypatch.setattr(postrun, "materialize_gru_standard_result", fake_standard_result)
    monkeypatch.setattr(postrun, "write_gru_standard_result", fake_write_standard)
    monkeypatch.setattr(postrun, "materialize_gru_evaluation_diagnostics", fake_evaluation)
    monkeypatch.setattr(postrun, "materialize_gru_pilot_figures", fake_figures)
    monkeypatch.setattr(postrun, "materialize_optional_objective_comparator", fake_objective)
    monkeypatch.setattr(
        postrun,
        "materialize_optional_map_error_decomposition",
        fake_map_decomposition,
    )
    monkeypatch.setattr(
        postrun,
        "materialize_optional_perturbation_response",
        fake_perturbation_response,
    )
    monkeypatch.setattr(
        postrun,
        "materialize_optional_feedback_ablation",
        fake_feedback_ablation,
    )

    manifest = postrun.materialize_gru_postrun_analysis(
        experiment="5f70333",
        run_ids=("run_a", "run_b"),
        labels=("A", "B"),
        output_tag="fullqrf_validation_selected",
        repo_root=tmp_path,
    )

    assert calls["standard"]["use_validation_selected_checkpoints"] is True
    assert calls["evaluation"]["use_validation_selected_checkpoints"] is True
    assert calls["figures"]["use_validation_selected_checkpoints"] is True
    assert calls["objective"]["use_validation_selected_checkpoints"] is True
    assert calls["objective"]["checkpoint_policy"] == "validation_selected_per_replicate"
    assert calls["map"]["use_validation_selected_checkpoints"] is True
    assert calls["map"]["standard_manifest_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_standard_certificates_fullqrf_validation_selected_manifest.json"
    )
    assert calls["perturbation"]["n_rollout_trials"] == postrun.DEFAULT_N_ROLLOUT_TRIALS
    assert calls["perturbation"]["output_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_perturbation_response_fullqrf_validation_selected_manifest.json"
    )
    assert calls["perturbation"]["bulk_dir"] == (
        tmp_path
        / "_artifacts"
        / "5f70333"
        / "perturbation_response"
        / "gru_fullqrf_validation_selected"
    )
    assert calls["feedback"]["n_rollout_trials"] == postrun.DEFAULT_N_ROLLOUT_TRIALS
    assert calls["feedback"]["labels"] == ("A", "B")
    assert calls["feedback"]["output_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_feedback_ablation_fullqrf_validation_selected.json"
    )
    assert calls["map"]["output_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_map_error_decomposition_fullqrf_validation_selected.json"
    )
    assert calls["checkpoint"]["preferred_manifest_path"] is None
    assert calls["checkpoint"]["checkpoint_selection_mode"] == "sparse_history"
    assert calls["standard_write"]["manifest_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_standard_certificates_fullqrf_validation_selected_manifest.json"
    )
    assert calls["evaluation"]["bulk_dir"] == (
        tmp_path
        / "_artifacts"
        / "5f70333"
        / "evaluation_diagnostics"
        / "gru_fullqrf_validation_selected"
    )
    assert calls["evaluation"]["regeneration_spec_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_evaluation_diagnostics_fullqrf_validation_selected_regeneration_spec.json"
    )
    assert calls["perturbation"]["regeneration_spec_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_perturbation_response_fullqrf_validation_selected_manifest_regeneration_spec.json"
    )
    assert calls["feedback"]["regeneration_spec_path"] == (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_feedback_ablation_fullqrf_validation_selected_regeneration_spec.json"
    )
    assert calls["figures"]["output_dir"] == (
        tmp_path
        / "_artifacts"
        / "5f70333"
        / "figures"
        / "gru_postrun_fullqrf_validation_selected"
    )
    assert manifest["checkpoint_policy"] == "validation_selected_per_replicate"
    assert manifest["labels"] == ["A", "B"]
    assert manifest["regeneration_specs"]["standard_certificate"].startswith("results/")
    assert manifest["regeneration_specs"]["postrun"].endswith("_regeneration_spec.json")
    assert manifest["checkpoint_selection_source"] == "validation_selected_per_replicate"
    assert manifest["selection_leakage_guard"]["status"] == "audit_only"
    assert manifest["outputs"]["fixed_bank_rescore_manifest"]["status"] == "missing"
    assert (
        manifest["outputs"]["fixed_bank_rescore_manifest"]["selection_use"]
        == "sparse_history_fallback"
    )
    assert manifest["outputs"]["map_decomposition"]["status"] == "materialized"
    assert manifest["outputs"]["map_decomposition"]["selection_role"] == (
        "audit_only_not_used_for_checkpoint_selection"
    )
    assert manifest["outputs"]["split_stress_objective_comparator"]["status"] == "materialized"
    assert manifest["outputs"]["perturbation_response"]["status"] == "materialized"
    assert manifest["outputs"]["feedback_ablation"]["status"] == "materialized"
    assert manifest["outputs"]["feedback_checkpoint_selection"]["status"] == "available"
    assert (
        manifest["outputs"]["feedback_checkpoint_selection"]["selection_use"]
        == "audit_only_not_primary_checkpoint_selection"
    )
    assert (
        manifest["outputs"]["split_stress_objective_comparator"][
            "standard_split_bank_comparator_status"
        ]
        == "available"
    )
    assert (
        manifest["outputs"]["split_stress_objective_comparator"]["selection_role"]
        == "audit_only_not_used_for_checkpoint_selection"
    )
    assert "map_error_decomposition" in manifest["selection_leakage_guard"]["audit_only_metrics"]
    assert (
        "split_stress_bank_objective_comparator"
        in manifest["selection_leakage_guard"]["audit_only_metrics"]
    )
    assert (
        "perturbation_response_bank"
        in manifest["selection_leakage_guard"]["audit_only_metrics"]
    )
    assert "feedback_ablation" in manifest["selection_leakage_guard"]["audit_only_metrics"]
    assert (
        "feedback_selected_checkpoint_audit"
        in manifest["selection_leakage_guard"]["audit_only_metrics"]
    )

    postrun_manifest = (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_postrun_materialization_fullqrf_validation_selected.json"
    )
    written = json.loads(postrun_manifest.read_text(encoding="utf-8"))
    assert written["outputs"]["evaluation_bulk_dir"].startswith("_artifacts/")
    assert written["outputs"]["standard_certificate_manifest"].startswith("results/")
    assert written["outputs"]["map_decomposition"]["json_path"].startswith("results/")
    assert written["outputs"]["perturbation_response"]["json_path"].startswith("results/")
    assert written["outputs"]["feedback_ablation"]["json_path"].startswith("results/")
    postrun_spec = (
        tmp_path
        / "results"
        / "5f70333"
        / "notes"
        / "gru_postrun_materialization_fullqrf_validation_selected_regeneration_spec.json"
    )
    assert postrun_spec.exists()
    assert json.loads(postrun_spec.read_text(encoding="utf-8"))["diagnostic_name"] == (
        "gru_postrun_materialization_bundle"
    )


def test_plan_gru_postrun_materialization_final_checkpoint_override(tmp_path: Path) -> None:
    plan = postrun.plan_gru_postrun_materialization(
        experiment="5f70333",
        run_ids=("run_a",),
        use_validation_selected_checkpoints=False,
        repo_root=tmp_path,
    )

    assert plan.checkpoint_policy == "final_checkpoint"
    assert plan.checkpoint_manifest_path is None
    assert plan.fixed_bank_rescore_manifest_path is None


def test_materialize_gru_postrun_analysis_prefers_provided_fixed_bank_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}
    fixed_manifest_path = tmp_path / "fixed_bank.json"
    fixed_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "rlrmp.fixed_bank_gru_checkpoint_rescore.v1",
                "issue": "5f70333",
                "checkpoint_policy": "fixed_bank_rescored_per_replicate",
                "selection_source": "fixed_bank_rescore",
                "materialization_status": "materialized",
                "validation_bank": {
                    "bank_identity": "fixed-bank:test",
                    "scorer_identity": "rollout_validation_objective:test",
                    "seed": 123,
                },
                "runs": {"run_a": []},
            }
        ),
        encoding="utf-8",
    )

    def fake_checkpoint_manifest(**kwargs: Any) -> dict[str, Any]:
        calls["checkpoint"] = kwargs
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "schema_version": "rlrmp.fixed_bank_gru_checkpoint_rescore.v1",
                    "materialization_status": "materialized",
                    "runs": {"run_a": []},
                }
            ),
            encoding="utf-8",
        )
        return {"checkpoint_policy": "fixed_bank_rescored_per_replicate"}

    def fake_standard_result(**kwargs: Any) -> dict[str, Any]:
        calls["standard"] = kwargs
        return {"summary": {}}

    def fake_write_standard(
        _result: dict[str, Any],
        *,
        note_path: Path,
        manifest_path: Path,
        regeneration_spec_path: Path,
        repo_root: Path,
    ) -> None:
        note_path.write_text("", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")

    def fake_evaluation(**kwargs: Any) -> dict[str, Any]:
        kwargs["output_path"].write_text("{}", encoding="utf-8")
        return {"schema_version": "rlrmp.gru_evaluation_diagnostics.v1"}

    def fake_figures(**kwargs: Any) -> dict[str, Any]:
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "figure_summary.json").write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        postrun,
        "materialize_validation_selected_checkpoint_manifest",
        fake_checkpoint_manifest,
    )
    monkeypatch.setattr(postrun, "materialize_gru_standard_result", fake_standard_result)
    monkeypatch.setattr(postrun, "write_gru_standard_result", fake_write_standard)
    monkeypatch.setattr(postrun, "materialize_gru_evaluation_diagnostics", fake_evaluation)
    monkeypatch.setattr(postrun, "materialize_gru_pilot_figures", fake_figures)
    monkeypatch.setattr(
        postrun,
        "materialize_optional_objective_comparator",
        lambda **_kwargs: {"status": "skipped"},
    )
    monkeypatch.setattr(
        postrun,
        "materialize_optional_map_error_decomposition",
        lambda **_kwargs: {"status": "skipped"},
    )
    monkeypatch.setattr(
        postrun,
        "materialize_optional_perturbation_response",
        lambda **_kwargs: {"status": "skipped"},
    )
    monkeypatch.setattr(
        postrun,
        "materialize_optional_feedback_ablation",
        lambda **_kwargs: {"status": "skipped"},
    )

    manifest = postrun.materialize_gru_postrun_analysis(
        experiment="5f70333",
        run_ids=("run_a",),
        fixed_bank_rescore_manifest_path=fixed_manifest_path,
        repo_root=tmp_path,
    )

    assert calls["checkpoint"]["preferred_manifest_path"] == fixed_manifest_path
    assert calls["checkpoint"]["checkpoint_selection_mode"] == "fixed_bank_manifest"
    assert manifest["checkpoint_selection_source"] == "fixed_bank_rescore"
    assert manifest["outputs"]["fixed_bank_rescore_manifest"]["status"] == "materialized"
    assert (
        manifest["outputs"]["fixed_bank_rescore_manifest"]["selection_use"]
        == "fixed_bank_rescore"
    )


def test_materialize_gru_postrun_analysis_preserves_audit_only_skip_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_standard_result(**_kwargs: Any) -> dict[str, Any]:
        return {"summary": {}}

    def fake_write_standard(
        _result: dict[str, Any],
        *,
        note_path: Path,
        manifest_path: Path,
        regeneration_spec_path: Path,
        repo_root: Path,
    ) -> None:
        note_path.write_text("", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")

    def fake_evaluation(**kwargs: Any) -> dict[str, Any]:
        kwargs["output_path"].write_text("{}", encoding="utf-8")
        return {"schema_version": "rlrmp.gru_evaluation_diagnostics.v1"}

    def fake_figures(**kwargs: Any) -> dict[str, Any]:
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "figure_summary.json").write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(postrun, "materialize_gru_standard_result", fake_standard_result)
    monkeypatch.setattr(postrun, "write_gru_standard_result", fake_write_standard)
    monkeypatch.setattr(postrun, "materialize_gru_evaluation_diagnostics", fake_evaluation)
    monkeypatch.setattr(postrun, "materialize_gru_pilot_figures", fake_figures)
    monkeypatch.setattr(
        postrun,
        "materialize_optional_perturbation_response",
        lambda **_kwargs: {"status": "skipped"},
    )

    manifest = postrun.materialize_gru_postrun_analysis(
        experiment="5f70333",
        run_ids=("run_a",),
        output_tag="audit_skip",
        use_validation_selected_checkpoints=False,
        include_objective_comparator=False,
        include_map_decomposition=False,
        include_perturbation_response=False,
        include_feedback_ablation=False,
        repo_root=tmp_path,
    )

    assert manifest["outputs"]["objective_comparator"] == {
        "status": "skipped",
        "reason": "disabled_by_cli",
    }
    assert manifest["outputs"]["map_decomposition"] == {
        "status": "skipped",
        "reason": "disabled_by_cli",
    }
    assert manifest["outputs"]["perturbation_response"] == {
        "status": "skipped",
        "reason": "disabled_by_cli",
    }
    assert manifest["outputs"]["feedback_ablation"] == {
        "status": "skipped",
        "reason": "disabled_by_cli",
    }
    assert manifest["outputs"]["feedback_checkpoint_selection"] == {
        "status": "skipped",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "selection_use": "audit_only_not_primary_checkpoint_selection",
        "source_sidecar": "feedback_ablation",
        "reason": "disabled_by_cli",
    }
    assert manifest["outputs"]["split_stress_objective_comparator"] == {
        "status": "skipped",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "source_sidecar": "objective_comparator",
        "reason": "disabled_by_cli",
    }
    assert manifest["selection_leakage_guard"]["status"] == "audit_only"
    assert "map_error_decomposition" in manifest["selection_leakage_guard"]["audit_only_metrics"]

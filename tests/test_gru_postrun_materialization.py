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
    assert plan.checkpoint_manifest_path == (
        tmp_path / "results" / "5f70333" / "notes" / "validation_selected_checkpoints.json"
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
    ) -> None:
        calls["standard_write"] = {
            "result": result,
            "note_path": note_path,
            "manifest_path": manifest_path,
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
        return {"status": "skipped", "reason": "test"}

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
    assert calls["figures"]["output_dir"] == (
        tmp_path
        / "_artifacts"
        / "5f70333"
        / "figures"
        / "gru_postrun_fullqrf_validation_selected"
    )
    assert manifest["checkpoint_policy"] == "validation_selected_per_replicate"
    assert manifest["selection_leakage_guard"]["status"] == "audit_only"

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


def test_plan_gru_postrun_materialization_final_checkpoint_override(tmp_path: Path) -> None:
    plan = postrun.plan_gru_postrun_materialization(
        experiment="5f70333",
        run_ids=("run_a",),
        use_validation_selected_checkpoints=False,
        repo_root=tmp_path,
    )

    assert plan.checkpoint_policy == "final_checkpoint"
    assert plan.checkpoint_manifest_path is None

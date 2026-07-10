"""Structural and behavior guards for governed feedback-ablation routing."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlrmp.analysis.pipelines import gru_feedback_ablation as feedback
from rlrmp.runtime.spec_migrations import FEEDBACK_ABLATION_EVAL_PARAMS_SCHEMA_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


@pytest.mark.parametrize(
    "relpath",
    (
        "scripts/materialize_gru_feedback_ablation.py",
        "src/rlrmp/benchmarks/postrun_eval_materialization.py",
        "src/rlrmp/eval/robustness_diagnostics.py",
        "src/rlrmp/analysis/pipelines/gru_postrun_materialization.py",
        "results/b413bb0/scripts/materialize_beta1p4_feedback_robustness_summary.py",
    ),
)
def test_live_feedback_callers_use_manifest_pipeline(relpath: str) -> None:
    calls = _call_names(REPO_ROOT / relpath)

    assert "evaluate_run_feedback_ablation" not in calls
    assert "materialize_gru_feedback_ablation" not in calls
    assert "execute_feedback_ablation_pipeline" in calls


def test_manifest_pipeline_links_evaluation_to_analysis_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text("{}\n", encoding="utf-8")
    payload_path = tmp_path / "payload.json"
    payload = {"schema_version": feedback.SCHEMA_VERSION, "runs": {"run-a": {}}}
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text("{}\n", encoding="utf-8")

    def fake_evaluation(spec, **kwargs):
        calls.append(("evaluation", spec))
        assert spec.params["schema_version"] == FEEDBACK_ABLATION_EVAL_PARAMS_SCHEMA_VERSION
        assert spec.params["source_experiment"] == "source"
        assert kwargs["issues"] == ["d0189db", "result"]
        return SimpleNamespace(id="eval-manifest"), evaluation_path

    def fake_analysis(spec, **kwargs):
        calls.append(("analysis", spec))
        assert spec.inputs[0].kind == "EvaluationRunManifest"
        assert spec.inputs[0].id == "eval-manifest"
        assert spec.inputs[0].uri == str(evaluation_path)
        assert kwargs["issues"] == ["d0189db", "result"]
        artifact = SimpleNamespace(
            role="rlrmp-gru-feedback-ablation-manifest",
            uri=str(payload_path),
        )
        return SimpleNamespace(id="analysis-manifest", artifacts=[artifact]), analysis_path

    monkeypatch.setattr(feedback, "execute_evaluation_run_spec", fake_evaluation)
    monkeypatch.setattr(feedback, "execute_analysis_run_spec", fake_analysis)
    monkeypatch.setattr(
        feedback,
        "_effective_checkpoint_policy_from_manifest",
        lambda *_args, **_kwargs: "validation_selected_per_replicate",
    )

    execution = feedback.execute_feedback_ablation_pipeline(
        source_experiment="source",
        result_experiment="result",
        run_ids=("run-a",),
        repo_root=tmp_path,
        feedbax_runs_root=tmp_path / "feedbax",
        issues=("d0189db",),
    )

    assert [name for name, _spec in calls] == ["evaluation", "analysis"]
    assert execution.evaluation_manifest.id == "eval-manifest"
    assert execution.analysis_manifest.id == "analysis-manifest"
    assert execution.payload == payload
